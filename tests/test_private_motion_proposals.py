"""Independent image proposals with residual coding and predictor-centered sparsity."""
import copy
import unittest

import torch

import test_joint_spatial_motion as fixtures
import test_joint_image_rate as codec_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class PrivateMotionProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, soft=False):
        template = fixtures.JointSpatialMotionTests().model(initial_logit=-.2)
        cfg = copy.deepcopy(template.config)
        cfg.rotation_updates = False
        cfg.proximal_gs_regularization = True
        cfg.differentiable_rate_support = True
        cfg.spatial_gate.joint_image_gradients = True
        cfg.spatial_gate.hierarchical_private = True
        cfg.spatial_gate.private_full_motion = True
        if soft:
            cfg.spatial_gate.image_relaxation = 'soft_anneal'
            cfg.spatial_gate.temperature = 3.
            cfg.spatial_gate.final_temperature = .3
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def open_all(model, freeze=True):
        with torch.no_grad():
            model.gate_logits[model.split_eligible] = 2.
        model._select_frontier(freeze=freeze)
        assert bool(model._private_support().all())

    def test_leaf_predictor_changes_encoding_without_training_the_image_twice(self):
        model = self.model()
        self.open_all(model)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .01
            model.gs_translation[:, 0] = .03
        before, _ = model(71, training=True)
        symbols_before = model._values(hard=True)[2].clone()
        target = model.base_xyz.clone()
        target[:, 0] += .05
        p, u = torch.autograd.grad((before - target).square().mean(),
                                  [model.region_translation, model.gs_translation])
        self.assertEqual(float(p[leaves].abs().sum()), 0.)
        self.assertGreater(float(u.abs().sum()), 0.)
        with torch.no_grad():
            model.region_translation[leaves, 0] += .002
        after, _ = model(71)
        torch.testing.assert_close(before, after, rtol=0, atol=2e-7)
        self.assertTrue(bool((model._values(hard=True)[2][:, 0] == symbols_before[:, 0] - 1).all()))
        decoded = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, after, rtol=0, atol=0)

    def test_closed_leaf_predictors_receive_coding_gradients_and_gs_adam_stays_photometric(self):
        for soft in (False, True):
            model = self.model(soft=soft)
            with torch.no_grad():
                model.gs_translation[:, 0] = .02
            xyz, _ = model(1, training=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            image_gradient = torch.autograd.grad((xyz - target).square().mean(), model.region_translation)[0]
            leaves = model.spatial_paths[:, -1].unique()
            self.assertEqual(float(image_gradient[leaves].abs().sum()), 0.)
            terms = model.loss_terms(2, diagnostics=False)
            regional, point = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
                [model.region_translation, model.gs_translation], allow_unused=True)
            self.assertLess(float(regional[leaves, 0].sum()), 0.)
            self.assertIsNone(point)

    def test_matching_full_motion_and_region_prediction_needs_no_private_symbols(self):
        model = self.model()
        self.open_all(model)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .02
            model.gs_translation[:, 0] = .02
        self.assertTrue(bool(model.gs_translation[:, 0].ne(0).all()))
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))
        model.record_final_diagnostics()
        decoded = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(71)[0], rtol=0, atol=0)
        torch.testing.assert_close(decoded.xyz[:, 0], model.base_xyz[:, 0] +
                                  torch.full_like(model.gs_translation[:, 0], .02), rtol=0, atol=2e-7)

    def test_l1_and_block_sparsity_shrink_towards_the_coding_predictor(self):
        for block in (False, True):
            model = self.model()
            self.open_all(model)
            leaves = model.spatial_paths[:, -1].unique()
            midpoint = len(model.owner) // 2
            with torch.no_grad():
                model.region_translation[leaves, 0] = .02
                model.gs_translation[:, 0] = .022
                if block:
                    model.gs_translation[midpoint:, 0] = .2
            p = model.gs_translation
            model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(p),
                                            exp_avg_sq=torch.full_like(p, 1e-12))
            model.last_step = 20
            if block:
                model._proximal_gs_blocks()
                self.assertGreater(float(model.gs_translation[midpoint:, 0].min()), .19)
            else:
                model.config.relative_motion_loss_weight = 1.
                model._proximal_relative_gs()
            origins = model._private_origins()[0]
            torch.testing.assert_close(model.gs_translation[:midpoint], origins[:midpoint], rtol=0, atol=0)
            self.assertTrue(bool(model._values(hard=True)[2][:midpoint].eq(0).all()))

    def test_migration_warms_complete_motion_including_birth_lineage(self):
        helper = fixtures.JointSpatialMotionTests()
        model = helper.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .01
            model.gs_translation[:, 0] = .02
        previous = model._full_motion_symbols()[0].clone()
        state = helper.commit(model, births=model.base_xyz[:1], sources=torch.tensor([0]))
        model.config.spatial_gate.joint_image_gradients = True
        model.config.spatial_gate.hierarchical_private = True
        model.config.spatial_gate.private_full_motion = True
        model.config.proximal_gs_regularization = True
        model.config.validate()
        self.assertFalse(model.private_full_motion)
        model.setup(state.xyz, state.rotation, model.extent, frame_index=2)
        self.assertTrue(model.private_full_motion)
        actual = torch.round(model.gs_translation / model.gs_steps[:3])
        torch.testing.assert_close(actual, torch.cat((previous, previous[:1])), rtol=0, atol=0)

    def test_soft_and_hard_stage_restore_the_same_future_update(self):
        for step in (20, 70):
            model = self.model(soft=True)
            xyz, _ = model(step, training=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(copy.deepcopy(model.capture()))
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            model.retain_survivors(torch.arange(0, len(model.owner), 2))
            state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
            torch.testing.assert_close(state.xyz, model(step + 1)[0], rtol=0, atol=0)

    def test_residual_range_is_checked_before_converting_to_int16(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[model.spatial_paths[:, -1].unique(), 0] = -100.
            model.gs_translation[:, 0] = 100.
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))
        self.open_all(model)
        with self.assertRaisesRegex(ValueError, 'int16'):
            model._values(hard=True)

    def test_full_motion_configuration_requires_its_matching_capacity_and_proximal_map(self):
        model = self.model()
        cfg = copy.deepcopy(model.config)
        cfg.spatial_gate.hierarchical_private = False
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg = copy.deepcopy(model.config)
        cfg.amplitude_refit_steps = 10
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == '__main__':
    unittest.main()
