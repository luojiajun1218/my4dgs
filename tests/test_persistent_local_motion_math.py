"""CPU math/codec proof for local edits; lifecycle tests live separately."""

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, path in (('pipeline', ROOT / 'pipeline'), ('pipeline.ReCon-GS', ROOT / 'pipeline/ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name); package.__path__ = [str(path)]; sys.modules[name] = package
RIGID = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
LOCAL = importlib.import_module('pipeline.ReCon-GS.model.persistent_local_motion')
PERSISTENT = importlib.import_module('pipeline.ReCon-GS.model.persistent_motion_regions')
MODEL = (RIGID.RigidResidualTree if issubclass(RIGID.RigidResidualTree, LOCAL.PersistentLocalMotionMixin)
         else type('LocalFixtureTree', (LOCAL.PersistentLocalMotionMixin, RIGID.RigidResidualTree), {}))


class PersistentLocalMotionMathTests(unittest.TestCase):
    def fixture(self, tree=False):
        cfg = RIGID.RigidTreeConfig(online_flat_clustering=True, adaptive_clustering=True,
            region_count=1, motion_steps=100, post_partition_steps=100, tree_edit_step=100,
            gs_xyz_step_extent=.001, gs_rotation_step_rad=.001,
            motion_xyz_step_extent=.00005, motion_rotation_step_rad=.00005,
            online_clustering=RIGID.OnlineMotionClusteringConfig(method='kmeans'),
            persistent_regions=PERSISTENT.PersistentRegionConfig(enabled=True, tree_enabled=tree))
        model = MODEL(cfg)
        source = torch.tensor(np.random.default_rng(8).normal(scale=.1, size=(24, 3)), dtype=torch.float32)
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(24, 1)
        model.setup(source, q, 1., frame_index=1)
        model.owner = torch.tensor([1] * 16 + [2] * 8)
        model.parents = torch.tensor([-1, 3, 3, 0] if tree else [-1, 0, 0])
        model.pivots = RIGID.region_pivots(source, model.owner, model.parents)
        count = len(model.parents)
        rt, rr = torch.zeros(count, 3), torch.zeros(count, 3)
        rt[1, 0], rt[2, 1] = .02, -.01
        if tree:
            rt[3] = torch.tensor([.015, -.003, .002])
            rr[3, 2], rr[1, 0], rr[2, 1] = .05, .01, -.02
        model.region_translation, model.region_rotation = nn.Parameter(rt), nn.Parameter(rr)
        model.gs_translation = nn.Parameter(torch.zeros(24, 3))
        model.gs_rotation = nn.Parameter(torch.zeros(24, 3))
        model.gs_translation.data[:8, 0] = .02 if tree else .006
        model.observations.fill_(1)
        model.persistent_state.region_ids = torch.tensor([0, 11, 44, 88] if tree else [0, 11, 44])
        model.persistent_state.reference_region_ids = torch.tensor([0])
        model.persistent_state.next_region_id = 101
        model.persistent_state.initialized = True
        model.region_map = torch.tensor([0])
        model._make_optimizer()
        model._region_id_lengths = RIGID._id_lengths(count - 1, source)
        model._topology_updated = True
        target = tuple(value.detach().clone() for value in model.forward(200))
        self.assertTrue(model._online_packet_score(*target)['valid'])
        return model, target

    def test_unified_cost_removes_twentyfold_precision_bias(self):
        rt = torch.tensor([[0, 0, 0], [120, 0, 0]])
        zero_r = torch.zeros_like(rt)
        gt = torch.zeros(8, 3, dtype=torch.long); gt[:, 0] = 6
        before = LOCAL.unified_gs_motion_cost(torch.zeros_like(rt), zero_r, gt, torch.zeros_like(gt),
            [.00005] * 6, [.001] * 6)
        after = LOCAL.unified_gs_motion_cost(rt, zero_r, torch.zeros_like(gt), torch.zeros_like(gt),
            [.00005] * 6, [.001] * 6)
        self.assertEqual(before['unified_gs_units_l1'], 48)
        self.assertAlmostEqual(after['unified_gs_units_l1'], 6)
        self.assertGreater(after['regional_raw_integer_l1'], before['affected_gs_integer_l1'])

    def test_new_sibling_preserves_others_and_passes_actual_quantized_gain(self):
        model, target = self.fixture(tree=True)
        original = model._persistent_layout_snapshot()
        old_world = tuple(value.detach().clone() for value in model._global_regions())
        result = model._apply_persistent_local_edit(*target,
            new_leaves=[dict(region_id=101, source_region_id=11, rows=torch.arange(8))])
        self.assertTrue(result['accepted'], result)
        self.assertGreater(result['unified_quantized_gain'], 0)
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 11, 44, 88, 101])
        self.assertEqual(model.parents.tolist(), [-1, 3, 3, 0, 3])
        self.assertEqual(model.persistent_state.next_region_id, 102)
        for key in ('region_translation', 'region_rotation', 'pivots'):
            self.assertTrue(torch.equal(getattr(model, key)[:4], original['partition'][key]))
        for key in ('gs_translation', 'gs_rotation'):
            self.assertTrue(torch.equal(getattr(model, key)[8:], original['partition'][key][8:]))
        new_world = model._global_regions()
        self.assertTrue(all(torch.equal(after[:4], before) for after, before in zip(new_world, old_world)))
        pose = model.forward(200)
        self.assertTrue(torch.equal(pose[0][8:], target[0][8:]))
        self.assertTrue(torch.equal(pose[1][8:], target[1][8:]))
        self.assertTrue(result['after_packet']['valid'])
        self.assertFalse(result['whole_packet_byte_veto'])
        self.assertTrue(all(group['params'][0] is parameter for group, parameter in zip(
            model.optimizer.param_groups, (model.region_translation, model.region_rotation,
                                          model.gs_translation, model.gs_rotation))))
        json.dumps(result, allow_nan=False)

    def test_high_rotation_only_without_position_fit_gain_rejected(self):
        model, _ = self.fixture()
        model.gs_translation.data.zero_()
        model.gs_rotation.data[:8, 0] = torch.tensor([.006, -.006] * 4)
        target = tuple(value.detach().clone() for value in model.forward(200))
        original = model._persistent_layout_snapshot()
        result = model._apply_persistent_local_edit(*target,
            new_leaves=[dict(region_id=101, source_region_id=11, rows=torch.arange(8))])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'no_unified_quantized_gain')
        self.assertGreaterEqual(result['unified_cost_after'], result['unified_cost_before'])
        for key, value in original['partition'].items():
            self.assertTrue(torch.equal(getattr(model, key), value))
        self.assertEqual(model.persistent_state.next_region_id, 101)

    def test_WORLD_zero_recycles_with_nonzero_GS_and_equality_gain(self):
        model, _ = self.fixture()
        model.region_translation.data[1] = 0
        model.gs_translation.data[:16, 0] = .006
        target = tuple(value.detach().clone() for value in model.forward(200))
        old_gs = model.gs_translation.detach().clone()
        result = model._apply_persistent_local_edit(*target, redirects={11: 0})
        self.assertTrue(result['accepted'], result)
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 44])
        self.assertTrue(bool(model.owner[:16].eq(0).all()))
        self.assertTrue(torch.equal(model.gs_translation, old_gs))
        self.assertEqual(result['unified_quantized_gain'], 0.)
        self.assertEqual(model.persistent_state.redirects[11], 0)
        self.assertTrue(result['after_packet']['valid'])

    def test_moving_parent_zero_local_cannot_be_recycled(self):
        model, _ = self.fixture(tree=True)
        model.region_translation.data[1] = 0
        model.region_rotation.data[1] = 0
        target = tuple(value.detach().clone() for value in model.forward(200))
        self.assertFalse(model._persistent_world_zero_measurements()[11]['world_shared_zero'])
        result = model._apply_persistent_local_edit(*target, redirects={11: 0})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'current_WORLD_shared_not_zero_or_unsupported')
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 11, 44, 88])

    def test_late_codec_failure_and_whole_leaf_selection_roll_back(self):
        model, target = self.fixture()
        before = model._persistent_layout_snapshot()
        score = model._online_packet_score(*target)
        with patch.object(model, '_online_packet_score', side_effect=[score, {**score, 'valid': False}]):
            result = model._apply_persistent_local_edit(*target,
                new_leaves=[dict(region_id=101, source_region_id=11, rows=torch.arange(8))])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'full_codec_pose_guard_failed')
        for key, value in before['partition'].items():
            self.assertTrue(torch.equal(getattr(model, key), value))
        self.assertEqual(model.persistent_state.parent_overrides, before['state']['parent_overrides'])
        self.assertEqual(model.persistent_state.next_region_id, 101)
        whole = model._apply_persistent_local_edit(*target,
            new_leaves=[dict(region_id=101, source_region_id=11, rows=torch.arange(16))])
        self.assertEqual(whole['reason'], 'insufficient_retained_members')


if __name__ == '__main__':
    unittest.main()
