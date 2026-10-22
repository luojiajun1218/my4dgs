"""Robust spatial initialization, far-point routing, reference hashes, and legacy layouts."""
import copy
import importlib
from pathlib import Path
import tempfile
import unittest
import torch
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _motion_fixtures import joint_motion
from pipeline.model.rigid_tree_codec import DecodedState
from pipeline.model.spatial_gate import SPATIAL_ROUTING_FIELDS, spatial_route, spatial_state_route


LAYOUT = importlib.import_module('pipeline.model.spatial_gate')


RIGID = importlib.import_module('pipeline.model.rigid_residual_tree')


MOTION = importlib.import_module('pipeline.model.spatial_gated_motion')


CODEC = importlib.import_module('pipeline.model.rigid_tree_codec')


class SpatialInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cloud(self):
        axis = torch.linspace(-1., 1., 16)
        core = torch.cartesian_prod(axis, axis, axis)
        return torch.cat((core, torch.tensor([[-100., -100., -100.],
                                             [100., 100., 100.]])))

    def model(self, xyz):
        config = RIGID.RigidTreeConfig()
        config.spatial_gate.max_depth = 3
        config.spatial_gate.initialization = 'robust_median'
        config.validate()
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        model = MOTION.SpatialGatedMotion(config)
        model.setup(xyz, q, extent=2., frame_index=1)
        return model

    def test_farther_outliers_cannot_move_supported_split_planes(self):
        xyz = self.cloud()
        pushed = xyz.clone()
        pushed[-2:] *= 10000.
        support = LAYOUT.spatial_initialization_support(xyz, 'robust_median')
        self.assertFalse(bool(support[-2:].any()))
        first = LAYOUT.spatial_layout(xyz, 5, True, initialization='robust_median')
        second = LAYOUT.spatial_layout(pushed, 5, True, initialization='robust_median')
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # All points, including the farthest two, retain a valid full path.
        self.assertEqual(first[0].shape, (len(xyz), 5))
        routed = LAYOUT.spatial_route(pushed, first[3], 5)
        torch.testing.assert_close(routed, first[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1][routed[:, 1:]], routed[:, :-1])

    def test_support_never_prunes_or_clamps_model_geometry(self):
        xyz = self.cloud()
        original = xyz.clone()
        model = self.model(xyz)
        torch.testing.assert_close(model.base_xyz, original, rtol=0, atol=0)
        torch.testing.assert_close(model(1)[0], original, rtol=0, atol=0)
        self.assertEqual(len(model.owner), len(original))
        self.assertEqual(len(model.parents), 585)

    def test_degenerate_and_invalid_clouds_have_defined_behavior(self):
        xyz = torch.ones(32, 3)
        result = LAYOUT.spatial_layout(xyz, 3, True, initialization='robust_median')
        self.assertTrue(bool(torch.isfinite(result[3]).all()))
        torch.testing.assert_close(LAYOUT.spatial_route(xyz, result[3], 3), result[0])
        for q in (0., .5, 1., float('nan'), True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                LAYOUT.spatial_layout(xyz, 3, initialization='robust_median', distance_quantile=q)
        for bad in (torch.zeros(0, 3), torch.tensor([[float('nan'), 0., 0.]]),
                    torch.tensor([[float('inf'), 0., 0.]])):
            with self.assertRaises(ValueError):
                LAYOUT.spatial_layout(bad, 3)

    def test_stream_handles_births_and_next_frame_without_new_codec_fields(self):
        model = self.model(self.cloud())
        reference = copy.deepcopy(model.frame_reference_state)
        source = torch.tensor([0, len(model.owner) - 1])
        with torch.no_grad():
            model.gs_translation[0, 0] = 2.5
        xyz, q = model.materialized_pose()
        newborn = xyz[source].clone()
        newborn[0] = torch.tensor([200., -200., 100.])
        model.register_materialized_layout(torch.cat((xyz, newborn)),
                                           torch.cat((q, q[source])), source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.bin'
            model.export_motion(path)
            decoded = CODEC.decode_motion(path.read_bytes(), reference)
            self.assertEqual(len(decoded.xyz), len(xyz) + 2)
            torch.testing.assert_close(decoded.xyz, torch.cat((xyz, newborn)), rtol=0, atol=1e-6)
            expected = LAYOUT.spatial_route(newborn, reference.spatial_centers, 3)
            torch.testing.assert_close(decoded.spatial_paths[-2:], expected, rtol=0, atol=0)
            restored = MOTION.SpatialGatedMotion(model.config)
            restored.restore(copy.deepcopy(model.capture()))
            for current in (model, restored):
                current.setup(decoded.xyz, decoded.rotation, 2., frame_index=2)
            torch.testing.assert_close(model.spatial_paths, restored.spatial_paths, rtol=0, atol=0)

    def test_median_checkpoint_is_compatible_but_cannot_resume_as_robust(self):
        xyz = self.cloud()
        config = RIGID.RigidTreeConfig()
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        original = MOTION.SpatialGatedMotion(config)
        original.setup(xyz, q, 2., frame_index=1)
        old = copy.deepcopy(original.capture())
        old['config']['spatial_gate'].pop('initialization')
        old['config']['spatial_gate'].pop('initialization_distance_quantile')
        resumed = MOTION.SpatialGatedMotion(config)
        resumed.restore(old)
        torch.testing.assert_close(resumed.spatial_paths, original.spatial_paths, rtol=0, atol=0)
        robust = copy.deepcopy(config)
        robust.spatial_gate.initialization = 'robust_median'
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            MOTION.SpatialGatedMotion(robust).restore(old)


class SpatialRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_far_points_use_nearest_supported_leaf_without_changing_geometry(self):
        model = joint_motion()
        reference = model.frame_reference_state
        query = torch.tensor([[100., 100., 100.], [-100., -100., -100.],
                              [200., -200., 100.]])
        distances = (query.double()[:, None] - reference.spatial_leaf_centers.double()[None]).square().sum(-1)
        expected = reference.spatial_leaf_nodes[distances.argmin(1)]
        paths = spatial_state_route(query, reference)
        torch.testing.assert_close(paths[:, -1], expected, rtol=0, atol=0)
        torch.testing.assert_close(reference.parents[paths[:, 1:]], paths[:, :-1])
        torch.testing.assert_close(model(1)[0], model.base_xyz, rtol=0, atol=0)
        self.assertEqual(len(model.owner), 514)
        inside = torch.tensor([[0., 0., 0.], [.1, -.2, .3]])
        torch.testing.assert_close(spatial_state_route(inside, reference),
            spatial_route(inside, reference.spatial_centers, 3), rtol=0, atol=0)

    def test_routing_reference_is_shared_once_restored_and_digest_bound(self):
        reference = joint_motion().frame_reference_state
        packet = {key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
                  for key, value in reference.capture().items() if value is not None}
        restored = DecodedState.from_reference(packet)
        self.assertEqual(restored.digest(), reference.digest())
        restored = DecodedState.restore(reference.capture())
        for name in SPATIAL_ROUTING_FIELDS:
            torch.testing.assert_close(getattr(restored, name), getattr(reference, name), rtol=0, atol=0)
        changed = copy.deepcopy(reference)
        changed.spatial_leaf_centers[0, 0] += .01
        self.assertNotEqual(changed.digest(), reference.digest())


if __name__ == '__main__':
    unittest.main()
