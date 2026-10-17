"""Fixed shared fields permit sparse GS exceptions at every selected depth."""
import copy
import unittest

import torch

import test_fixed_gate_residuals as fixtures
import test_joint_spatial_motion as lineage_fixtures
import test_joint_image_rate as codec_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class FixedGateAllResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, enabled=True, rotation=False):
        template = fixtures.FixedGateResidualTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.fixed_gate_all_gs_residuals = enabled
        cfg.rotation_updates = rotation
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def mixed_frontier(model, freeze=False):
        with torch.no_grad():
            model.gate_logits[model.split_eligible] = 2.
            model.gate_logits[model.spatial_paths[0, 0]] = -2.
        model._select_frontier(freeze=freeze)

    def test_freeze_preserves_exact_pose_and_packet_before_new_residuals_are_learned(self):
        model = self.model()
        self.mixed_frontier(model)
        finest = model.owner.eq(model.spatial_paths[:, -1])
        self.assertTrue(bool(finest.any()) and bool((~finest).any()))
        with torch.no_grad():
            model.region_translation[:, 0] = .01
            model.gs_translation[:, 0] = .0304
        before = codec_fixtures.JointImageRateTests.packet(model)
        pose = model(70)[0].detach().clone()
        model.last_step = 70
        self.mixed_frontier(model, freeze=True)
        self.assertFalse(model.private_full_motion)
        self.assertTrue(bool(model._private_support().all()))
        self.assertTrue(bool(model.gs_translation[~finest].eq(0).all()))
        self.assertEqual(before, codec_fixtures.JointImageRateTests.packet(model))
        torch.testing.assert_close(model(71, training=True)[0], pose, rtol=0, atol=0)

    def test_coarse_owner_points_receive_photo_gradients_only_after_freeze(self):
        model = self.model()
        self.mixed_frontier(model)
        coarse = model.owner.ne(model.spatial_paths[:, -1])
        target = model.base_xyz + .04
        model._sampled_gate_step = 69
        model._sampled_gate_noise = torch.full_like(model.gate_logits, -30.)
        xyz = model(69, training=True)[0]
        before = torch.autograd.grad((xyz - target).square().mean(), model.gs_translation)[0]
        self.assertEqual(float(before[coarse].abs().sum()), 0.)
        model.last_step = 70
        self.mixed_frontier(model, freeze=True)
        xyz = model(71, training=True)[0]
        region, point = torch.autograd.grad((xyz - target).square().mean(),
                                            [model.region_translation, model.gs_translation])
        self.assertTrue(bool(point[coarse].ne(0).any(-1).all()))
        self.assertGreater(float(region[model.owner[coarse].unique()].abs().sum()), 0.)

    def test_coarse_translation_and_orientation_exceptions_use_the_existing_stream(self):
        model = self.model(rotation=True)
        self.mixed_frontier(model, freeze=True)
        row = int(model.owner.ne(model.spatial_paths[:, -1]).nonzero()[0])
        with torch.no_grad():
            model.gs_translation[row, 0] = .006
            model.gs_rotation[row, 1] = .02
        self.assertTrue(bool(model._values(hard=True)[2][row].ne(0).any()))
        self.assertTrue(bool(model._values(hard=True)[3][row].ne(0).any()))
        state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        xyz, rotation = model(71)
        torch.testing.assert_close(state.xyz, xyz, rtol=0, atol=0)
        torch.testing.assert_close(state.rotation, rotation, rtol=0, atol=1e-7)

    def test_restoration_preserves_scope_before_and_after_freeze(self):
        for frozen in (False, True):
            model = self.model()
            self.mixed_frontier(model, freeze=frozen)
            if frozen:
                with torch.no_grad():
                    model.gs_translation[0, 0] = .004
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(copy.deepcopy(model.capture()))
            self.assertEqual(bool(restored._private_support().all()), frozen)
            torch.testing.assert_close(restored(71)[0], model(71)[0], rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))

    def test_next_frame_proposals_include_coarse_exceptions_and_birth_lineage(self):
        model = self.model()
        self.mixed_frontier(model, freeze=True)
        with torch.no_grad():
            model.gs_translation[:, 0] = .006
        full = model._full_motion_symbols()[0].clone()
        state = lineage_fixtures.JointSpatialMotionTests().commit(model,
            births=model.base_xyz[:1], sources=torch.tensor([0]))
        model.setup(state.xyz, state.rotation, model.extent, frame_index=2)
        self.assertTrue(model.private_full_motion)
        self.assertFalse(model._hard_gate_frozen)
        self.assertFalse(bool(model._private_support().all()))
        torch.testing.assert_close(torch.round(model.gs_translation / model.gs_steps[:3]),
                                   torch.cat((full, full[:1])), rtol=0, atol=0)

    def test_disabled_default_and_legacy_checkpoint_retain_finest_only_scope(self):
        model = self.model(enabled=False)
        self.mixed_frontier(model, freeze=True)
        with torch.no_grad():
            model.gs_translation[:, 0] = .006
        coarse = model.owner.ne(model.spatial_paths[:, -1])
        self.assertTrue(bool(model._values(hard=True)[2][coarse].eq(0).all()))
        packet = copy.deepcopy(model.capture())
        packet['config']['spatial_gate'].pop('fixed_gate_all_gs_residuals')
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(packet)
        self.assertTrue(torch.equal(restored._private_support(), ~coarse))

    def test_configuration_requires_a_matching_residual_stage(self):
        model = self.model()
        for value in (1, 'yes', None):
            cfg = copy.deepcopy(model.config)
            cfg.spatial_gate.fixed_gate_all_gs_residuals = value
            with self.assertRaises(ValueError):
                cfg.validate()
        cfg = copy.deepcopy(model.config)
        cfg.spatial_gate.reparameterize_after_gate_freeze = False
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == '__main__':
    unittest.main()
