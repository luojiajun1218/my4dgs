"""Disabling motion rotations retains orientation through warm starts/replay."""
import copy
from pathlib import Path
import tempfile
import unittest
import torch

from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_residual_tree import rotation_vector_quaternion
from pipeline.model.rigid_tree_codec import decode_motion, uncompressed_motion_packet, compress_motion_packet
from pipeline.model.sparse_integer_codec import (encode_motion_layout, pack_layout, MOTION_MAGICS)
import test_joint_spatial_motion as joint_tests


class TranslationOnlyMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        source = joint_tests.JointSpatialMotionTests().model()
        config = copy.deepcopy(source.config)
        config.rotation_updates = False
        config.proximal_gs_regularization = True
        model = SpatialGatedMotion(config)
        rotation = rotation_vector_quaternion(torch.full_like(source.base_xyz, .2))
        model.setup(source.base_xyz, rotation, source.extent, frame_index=1)
        return model

    def test_translation_learns_with_no_rotation_parameters_or_adam_moments(self):
        model = self.model()
        xyz, orientation = model(1, training=True)
        image = (xyz - model.base_xyz - .01).square().mean()
        terms = model.loss_terms(1, False)
        (image + terms['rate_loss'] + terms['relative_motion_loss']).backward()
        model.optimizer_step()
        self.assertIsNotNone(model.optimizer.state.get(model.gs_translation))
        for parameter in (model.region_rotation, model.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertNotIn(parameter, model.optimizer.state)
            self.assertTrue(bool(parameter.eq(0).all()))
        torch.testing.assert_close(orientation, model.base_rotation, rtol=0, atol=0)

    def test_causal_transition_drops_old_increment_and_replays_initial_orientation(self):
        source = joint_tests.JointSpatialMotionTests().model()
        with torch.no_grad():
            source.region_rotation[1:, 1] = .02
            source.gs_rotation[:, 0] = .01
        decoded = joint_tests.JointSpatialMotionTests().commit(source)
        source.config.rotation_updates = False
        source.setup(decoded.xyz, decoded.rotation, source.extent, frame_index=2)
        self.assertTrue(bool(source.gs_rotation.eq(0).all()))
        self.assertTrue(bool(source.region_rotation.eq(0).all()))
        torch.testing.assert_close(source(1)[1], decoded.rotation, rtol=0, atol=0)
        replayed = joint_tests.JointSpatialMotionTests().commit(source)
        torch.testing.assert_close(replayed.rotation, decoded.rotation, rtol=0, atol=0)

    def test_restore_and_survivor_pruning_keep_motion_rotations_disabled(self):
        model = self.model()
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        rows = torch.arange(0, len(model.owner), 2)
        restored.retain_survivors(rows)
        for parameter in (restored.region_rotation, restored.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertTrue(bool(parameter.eq(0).all()))
        replayed = joint_tests.JointSpatialMotionTests().commit(restored)
        torch.testing.assert_close(replayed.rotation, model.base_rotation[rows], rtol=0, atol=0)

    def test_all_rotation_off_replays_births_without_transmitting_quaternions(self):
        for relative in (False, True):
            model = self.model()
            if relative:
                model.config.spatial_gate.region_motion = 'relative'
            prior = copy.deepcopy(model.frame_reference_state)
            xyz, orientation = model.materialized_pose()
            sources = torch.tensor([1, 17, 100])
            xyz = torch.cat((xyz, xyz[sources] + .003))
            orientation = torch.cat((orientation, orientation[sources]))
            model.register_materialized_layout(xyz, orientation, sources)
            with tempfile.TemporaryDirectory() as directory:
                inherited, explicit = Path(directory)/'inherited.bin', Path(directory)/'explicit.bin'
                model.export_motion(inherited, inherit_birth_rotation=True)
                model.export_motion(explicit)
                packet = inherited.read_bytes()
                raw = uncompressed_motion_packet(packet)
                self.assertEqual(raw[:4], b'SPG4' if relative else b'SPG3')
                self.assertEqual(len(uncompressed_motion_packet(explicit.read_bytes())) - len(raw),
                                 16 * len(sources))
                formats = (packet, raw, compress_motion_packet(raw, 'zlib'),
                           pack_layout(encode_motion_layout(raw), len(raw), MOTION_MAGICS))
                for encoded in formats:
                    decoded = decode_motion(encoded, prior)
                    torch.testing.assert_close(decoded.rotation.view(torch.int32),
                                               orientation.view(torch.int32), rtol=0, atol=0)
                    torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)

    def test_inherited_rotation_stream_rejects_hidden_rotation_updates(self):
        model = self.model()
        xyz, orientation = model.materialized_pose()
        model.register_materialized_layout(xyz, orientation, torch.empty(0, dtype=torch.long))
        with torch.no_grad():
            model.region_rotation[1, 0] = .02
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'preserve all'):
            model.export_motion(Path(directory)/'bad.bin', inherit_birth_rotation=True)


if __name__ == '__main__':
    unittest.main()
