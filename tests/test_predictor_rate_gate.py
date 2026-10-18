"""All-depth GS proposals use the same predictor residual for gate RD and encoding."""
import copy
import importlib
from types import SimpleNamespace
import unittest

import torch

import test_inherited_image_gate as fixtures
import test_proposal_image_supervision as module_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion

MODULE = importlib.import_module('pipeline.module')


class PredictorRateGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        template = fixtures.InheritedImageGateTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.hierarchical_private = False
        cfg.spatial_gate.proposal_image_weight = 0.
        cfg.spatial_gate.predictor_image_weight = .25
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def packet(model):
        return fixtures.InheritedImageGateTests.packet(model)

    def test_main_fits_complete_motion_while_predictor_image_trains_region_and_gate(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.region_translation[9:73, 0] = .004
            model.gs_translation[:, 0] = .02
        target = model.base_xyz.clone()
        target[:, 0] += .04
        xyz, _ = model(1, training=True)
        p, u, g = torch.autograd.grad((xyz - target).square().mean(),
            [model.region_translation, model.gs_translation, model.gate_logits], allow_unused=True)
        self.assertIsNone(p)
        self.assertIsNone(g)
        self.assertTrue(bool(u[:, 0].ne(0).all()))
        sampler = model._gate_rng.get_state().clone()
        prediction, _ = model.predictor_pose()
        p, u, g = torch.autograd.grad((prediction - target).square().mean(),
            [model.region_translation, model.gs_translation, model.gate_logits], allow_unused=True)
        self.assertIsNone(u)
        self.assertGreater(float(p.abs().sum()), 0.)
        self.assertGreater(float(g.abs().sum()), 0.)
        self.assertTrue(torch.equal(sampler, model._gate_rng.get_state()))

    def test_gs_coding_residual_changes_gate_rate_gradient_in_the_correct_direction(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[9:73, 0] = .02
        gradients = []
        for increment in (0., .02):
            with torch.no_grad():
                model.gs_translation[:, 0] = increment
            terms = model.loss_terms(10, False)
            gate, point = torch.autograd.grad(terms['rate_loss'],
                [model.gate_logits, model.gs_translation], allow_unused=True)
            gradients.append(gate)
            self.assertIsNone(point)  # GS Adam remains photometric.
        self.assertTrue(bool(gradients[0][1:9].gt(0).all()))
        self.assertTrue(bool(gradients[1][1:9].lt(0).all()))

    def test_hard_rate_and_packet_use_actual_coarse_owner_residuals(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
        terms = model.loss_terms(10, False)
        expected = (model._gs_id_lengths + 7).sum() + model._metadata_bytes()
        self.assertEqual(float(terms['rate_bytes_surrogate']), float(expected))
        with torch.no_grad():
            model.region_translation[1:9, 0] = .02
        self.assertFalse(bool(model._private_symbols()[0].ne(0).any()))
        terms = model.loss_terms(10, False)
        self.assertEqual(float(terms['rate_bytes_surrogate']), 64. + model._metadata_bytes())
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(10)[0], rtol=0, atol=0)

    def test_gate_switch_changes_coding_residual_not_complete_integer_motion(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.region_translation[9:73, 0] = .004
            model.gs_translation[:, 0] = .02
        before = tuple(x.clone() for x in model._full_motion_symbols())
        residual = model._private_symbols()[0].clone()
        with torch.no_grad():
            model.gate_logits[1:9] = 2.
        model._select_frontier(freeze=False)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model._full_motion_symbols())))
        self.assertTrue(bool((model._private_symbols()[0][:, 0] == residual[:, 0] - 2).all()))
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(10)[0], rtol=0, atol=0)

    def test_raw_block_proximal_shrinks_to_selected_predictor_instead_of_zero(self):
        model = self.model()
        mid = len(model.owner) // 2
        with torch.no_grad():
            model.region_translation[1:9, 0] = .02
            model.gs_translation[:, 0] = .022
            model.gs_translation[mid:, 0] = .2
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(p),
                                        exp_avg_sq=torch.full_like(p, 1e-12))
        model.last_step = 20
        model._proximal_gs_blocks()
        torch.testing.assert_close(model.gs_translation[:mid], model._private_origins()[0][:mid],
                                   rtol=0, atol=0)
        self.assertGreater(float(model.gs_translation[mid:, 0].min()), .19)

    def test_fixed_gate_conversion_keeps_packet_and_pose_without_removing_coarse_gs(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gs_translation[:, 0] = .02
        before, pose = self.packet(model), model(69)[0].detach()
        model.last_step = 70
        model._select_frontier(freeze=True)
        self.assertFalse(model.private_full_motion)
        self.assertTrue(bool(model.gs_translation[:, 0].ne(0).all()))
        self.assertEqual(before, self.packet(model))
        torch.testing.assert_close(pose, model(71)[0], rtol=0, atol=0)
        self.assertEqual(model.predictor_image_weight, 0.)
        target = model.base_xyz + .04
        p, u = torch.autograd.grad((model(71, training=True)[0] - target).square().mean(),
                                   [model.region_translation, model.gs_translation])
        self.assertGreater(float(p.abs().sum()), 0.)
        self.assertTrue(bool(u.ne(0).all()))

    def test_module_auxiliary_reuses_views_and_does_not_refit_appearance(self):
        model = self.model()
        module, alpha, color = module_fixtures.ProposalImageSupervisionTests().module(model)
        view = SimpleNamespace(image=torch.full((3, 20, 20), .6))
        loss, metrics, package = MODULE.Module.training_step(module, view, 1, regularization_weight=0.)
        torch.testing.assert_close(loss.detach(),
            metrics['image_loss'] + .25 * metrics['predictor_image_loss'], rtol=0, atol=0)
        self.assertEqual(package['marker'], 1)
        self.assertEqual(module.renderer.calls, [True, False])
        main = module.forward(view, training=True)['render']
        expected = torch.autograd.grad(MODULE.motion_photometric_loss(main, view.image, .2), [alpha, color])
        actual = torch.autograd.grad(loss, [alpha, color])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            module.forward(view, training=True, predictor_only=True, proposal_only=True)

    def test_restore_keeps_each_coordinate_stage_and_future_updates(self):
        for step in (20, 70):
            model = self.model()
            target = model.base_xyz + .02
            def update(m, current):
                xyz, _ = m(current, training=True)
                loss = (xyz - target).square().mean() + m.regularization_loss(current)
                if m.predictor_image_weight:
                    loss = loss + .25 * (m.predictor_pose()[0] - target).square().mean()
                loss.backward()
                m.optimizer_step()
            update(model, step)
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(copy.deepcopy(model.capture()))
            self.assertEqual(restored.private_full_motion, step < 70)
            update(model, step+1)
            update(restored, step+1)
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(self.packet(model), self.packet(restored))


if __name__ == '__main__':
    unittest.main()
