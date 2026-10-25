"""Freeze learned gates, preserve exact motion, and fit shared residual fields."""
import copy
import unittest

import torch

import test_stochastic_gate_feedback as fixtures
import test_joint_spatial_motion as lineage_fixtures
import test_joint_image_rate as codec_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class FixedGateResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, refit=0):
        template = fixtures.StochasticGateFeedbackTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.reparameterize_after_gate_freeze = True
        cfg.amplitude_refit_steps = refit
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def open_all(model, freeze):
        with torch.no_grad():
            model.gate_logits[model.split_eligible] = 2.
        model._select_frontier(freeze=freeze)

    def test_conversion_preserves_stream_and_pose_while_shared_image_gradients_return(self):
        model = self.model()
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .0101
            model.gs_translation[:, 0] = .0304
        before = codec_fixtures.JointImageRateTests.packet(model)
        pose = model(70)[0].detach().clone()
        model.last_step = 70
        self.open_all(model, freeze=True)
        self.assertFalse(model.private_full_motion)
        self.assertEqual(before, codec_fixtures.JointImageRateTests.packet(model))
        xyz, _ = model(71, training=True)
        torch.testing.assert_close(xyz, pose, rtol=0, atol=0)
        target = model.base_xyz.clone()
        target[:, 0] += .05
        region, point = torch.autograd.grad((xyz - target).square().mean(),
            [model.region_translation, model.gs_translation])
        self.assertGreater(float(region[leaves].abs().sum()), 0.)
        self.assertGreater(float(point.abs().sum()), 0.)

    def test_zero_residual_points_can_move_by_their_shared_region(self):
        model = self.model()
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .02
            model.gs_translation[:, 0] = .02
        model.last_step = 70
        self.open_all(model, freeze=True)
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))
        model.config.region_translation_lr = .004
        model.config.gs_translation_lr = 0.
        model.config.rate_loss_weight = model.config.relative_motion_loss_weight = 0.
        target = model.base_xyz.clone()
        target[:, 0] += .04
        xyz, _ = model(71, training=True)
        initial = float((xyz - target).square().mean())
        (xyz - target).square().mean().backward()
        model.optimizer_step()
        self.assertLess(float((model(72)[0] - target).square().mean()), initial)
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))

    def test_refit_only_changes_existing_blocks_and_keeps_replay_exact(self):
        model = self.model(refit=20)
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .02
            model.gs_translation[:, 0] = .02
            model.gs_translation[0, 0] += .004
        model.last_step = 70
        self.open_all(model, freeze=True)
        before = model(80)[0].detach().clone()
        pose, _ = model(81, training=True)
        torch.testing.assert_close(pose, before, rtol=0, atol=0)
        self.assertTrue(model.amplitude_refit_active)
        mask = model._amplitude_refit_support['gs_translation'].clone()
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(float(model.regularization_loss(81)), 0.)
        opened = model.opened.clone()
        target = model.base_xyz.clone()
        target[:, 0] += .04
        for step in range(81, 86):
            pose, _ = model(step, training=True)
            (pose - target).square().mean().backward()
            model.optimizer_step()
            self.assertTrue(torch.equal(opened, model.opened))
            self.assertFalse(bool(model._values(hard=True)[2][~mask].ne(0).any()))
        state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(86)[0], rtol=0, atol=0)

    def test_restoration_keeps_each_stage_coordinate_system_and_next_update(self):
        for step in (20, 70, 85):
            model = self.model(refit=20)
            if step == 85:
                model.last_step = 70
                self.open_all(model, freeze=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            xyz, _ = model(step, training=True)
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            packet = copy.deepcopy(model.capture())
            self.assertEqual(packet['private_full_motion_active'], step < 70)
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(packet)
            self.assertEqual(restored.private_full_motion, model.private_full_motion)
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            broken = copy.deepcopy(packet)
            broken['private_full_motion_active'] = not packet['private_full_motion_active']
            with self.assertRaises(ValueError):
                SpatialGatedMotion(copy.deepcopy(model.config)).restore(broken)

    def test_next_frame_restarts_complete_proposals_with_birth_lineage(self):
        model = self.model()
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .01
            model.gs_translation[:, 0] = .03
        model.last_step = 70
        self.open_all(model, freeze=True)
        full = model._full_motion_symbols()[0].clone()
        helper = lineage_fixtures.JointSpatialMotionTests()
        state = helper.commit(model, births=model.base_xyz[:1], sources=torch.tensor([0]))
        model.setup(state.xyz, state.rotation, model.extent, frame_index=2)
        self.assertTrue(model.private_full_motion)
        proposal = torch.round(model.gs_translation / model.gs_steps[:3])
        torch.testing.assert_close(proposal, torch.cat((full, full[:1])), rtol=0, atol=0)

    def test_configuration_requires_a_matching_proposal_stage(self):
        model = self.model(refit=20)
        for field, value in (('private_full_motion', False), ('image_relaxation', 'soft_anneal')):
            cfg = copy.deepcopy(model.config)
            setattr(cfg.spatial_gate, field, value)
            with self.assertRaises(ValueError):
                cfg.validate()


if __name__ == '__main__':
    unittest.main()
