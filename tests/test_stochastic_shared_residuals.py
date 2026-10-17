"""Stochastic shared predictors retain sparse GS corrections at any frontier."""
import copy
import unittest

import torch

import test_stochastic_gate_feedback as fixtures
import test_joint_image_rate as codec_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class StochasticSharedResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, refit=0):
        template = fixtures.StochasticGateFeedbackTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.hierarchical_private = False
        cfg.spatial_gate.private_full_motion = False
        cfg.amplitude_refit_steps = refit
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    def test_coarse_frontier_keeps_a_real_image_derivative_for_every_gs_residual(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .004
        self.assertEqual(len(model.owner.unique()), 8)
        self.assertFalse(bool(model._private_support().any()))
        xyz, _ = model(1, training=True)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        gradient = torch.autograd.grad((xyz - target).square().mean(), model.gs_translation)[0]
        self.assertTrue(bool(gradient[:, 0].ne(0).all()))
        self.assertTrue(bool(model._values(hard=True)[2][:, 0].ne(0).all()))

    def test_sampled_shared_region_sees_its_own_prediction_with_a_private_correction_present(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .004
        model(1, training=True)
        draw = model.gate_logits / model._gate_temperature() + model._sampled_gate_noise
        weights = model._path_weights((draw.ge(0) & model.split_eligible).to(model.gate_logits))
        row, layer = weights.nonzero()[0].tolist()
        node = int(model.spatial_paths[row, layer])
        members = model.spatial_paths[:, layer].eq(node)
        target = model.base_xyz.clone()
        target[:, 0] += .004
        target[members, 0] += .02
        gradients = []
        for displacement in (.01, .2):
            with torch.no_grad():
                model.region_translation.zero_()
                model.region_translation[node, 0] = displacement
            xyz, _ = model(1, training=True)
            gradient = torch.autograd.grad((xyz - target).square().mean(), model.region_translation)[0]
            gradients.append(float(gradient[node, 0]))
        self.assertLess(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)

    def test_freezing_keeps_all_gs_corrections_resets_moments_and_decodes_exactly(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
            model.region_translation[9:, 1] = .004
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
                                        exp_avg_sq=torch.ones_like(p))
        before = model.gs_translation.detach().clone()
        model._select_frontier(freeze=True)
        torch.testing.assert_close(model.gs_translation, before, rtol=0, atol=0)
        self.assertNotIn(p, model.optimizer.state)
        state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(71, training=True)[0], rtol=0, atol=0)
        self.assertEqual(model.capture()['regional_image_fitting'], 'joint_image_residual_stochastic_v1')

    def test_sampled_fixed_and_refit_stages_resume_and_retire_exactly(self):
        for step in (20, 70, 85):
            model = self.model(refit=20)
            if step == 85:
                model.last_step = 70
                model._select_frontier(freeze=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            xyz, _ = model(step, training=True)
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(copy.deepcopy(model.capture()))
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
                current.retain_survivors(torch.arange(0, len(current.owner), 2))
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
            torch.testing.assert_close(state.xyz, model(step + 1)[0], rtol=0, atol=0)

    def test_gates_can_learn_to_use_a_helpful_shared_field_without_a_private_capacity_rule(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[9:, 0] = .02
        target = model.base_xyz.clone()
        target[:, 0] += .02
        initial = float((model(1)[0] - target).square().mean())
        optimizer = torch.optim.Adam([model.gate_logits], lr=.3)
        for step in range(1, 61):
            optimizer.zero_grad()
            xyz, _ = model(step, training=True)
            gradient = torch.autograd.grad((xyz - target).square().mean(), model.gate_logits)[0]
            model.gate_logits.grad = gradient
            optimizer.step()
            model._select_frontier(freeze=False)
        final = float((model(60)[0] - target).square().mean())
        self.assertLess(final, initial * .1)
        self.assertTrue(bool(model.opened[1:9].any()))


if __name__ == '__main__':
    unittest.main()
