"""Real-model local sibling edits, world-identity recycling and codec guards."""

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_persistent_motion_integration as lifecycle_tests


class PersistentLocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.lifecycle = lifecycle_tests.PersistentLifecycleTests()
        self.rigid, _, self.flat, _, self.codec = self.lifecycle.modules()

    def fixture(self):
        template = self.lifecycle.fixture(count=64)
        config = replace(template.config, motion_xyz_step_extent=.00005,
                         motion_rotation_step_rad=.00005)
        model = self.rigid.RigidResidualTree(config)
        generator = torch.Generator().manual_seed(21)
        xyz = torch.zeros(64, 3)
        for start, count, scale in ((0, 8, .003), (8, 40, .1), (48, 8, .1), (56, 8, .1)):
            half = torch.randn((count // 2, 3), generator=generator) * scale
            xyz[start:start + count] = torch.cat((half, -half))
        xyz[:48, 0] -= 2.
        xyz[48:56, 0] += 2.
        model.setup(xyz, template.base_rotation, 1., frame_index=1)
        model.observations.fill_(10)
        with torch.no_grad():
            model.gs_translation[:48, 0] = .02
            model.gs_translation[48:56, 1] = .03
        model.forward(100, training=True)
        model.update_topology(100)
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2])
        self.assertEqual(model._persistent_owner_ids()[:48].tolist(), [1] * 48)
        self.assertEqual(model._persistent_owner_ids()[48:56].tolist(), [2] * 8)
        return model

    @staticmethod
    def target(model):
        return tuple(value.detach().clone() for value in model.forward(100))

    def test_twenty_fold_precision_difference_accepts_sharing_and_preserves_old_nodes(self):
        model = self.fixture()
        with torch.no_grad():
            model.gs_translation[:8, 0] += .006
        target = self.target(model)
        rt, rr = model.region_translation.detach().clone(), model.region_rotation.detach().clone()
        pivots, parents = model.pivots.clone(), model.parents.clone()
        logical = model._persistent_owner_ids().clone()
        frame0 = model.frame_reference_state.digest()
        identifier = model.persistent_state.next_region_id
        result = model._apply_persistent_local_edit(*target, new_leaves=[dict(
            region_id=identifier, source_region_id=1, rows=torch.arange(8))])
        self.assertTrue(result['accepted'], result)
        self.assertLess(result['unified_cost_after'], result['unified_cost_before'])
        self.assertGreater(result['unified_cost_after'], .75 * result['unified_cost_before'],
                           'Strict improvement must pass even when reduction is below 25 percent')
        self.assertEqual(model.persistent_state.next_region_id, identifier + 1)
        torch.testing.assert_close(model.region_translation[:3], rt, rtol=0, atol=0)
        torch.testing.assert_close(model.region_rotation[:3], rr, rtol=0, atol=0)
        torch.testing.assert_close(model.pivots[:3], pivots, rtol=0, atol=0)
        self.assertTrue(torch.equal(model.parents[:3], parents))
        self.assertEqual(int(model.parents[-1]), int(parents[1]), 'New leaf must be a sibling')
        self.assertTrue(torch.equal(model._persistent_owner_ids()[8:], logical[8:]))
        self.assertEqual(model._persistent_owner_ids()[:8].tolist(), [identifier] * 8)
        hard_rt, _, gt, _ = model._values(hard=True)
        self.assertGreater(int(hard_rt[-1].abs().sum()), 48, 'Raw regional integers alone would wrongly reject this gain')
        self.assertEqual(int(gt[:8].abs().sum()), 0)
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
            self.assertTrue(any(getattr(model, name) is p for g in model.optimizer.param_groups for p in g['params']))
        restored = self.rigid.RigidResidualTree(model.config)
        restored.restore(copy.deepcopy(model.capture()))
        self.assertEqual(restored.persistent_state.region_ids.tolist(), [0, 1, 2, identifier])
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.lifecycle.finish_export(restored, Path(directory) / 'motion_001.bin', birth_source=0)
            torch.testing.assert_close(decoded.xyz[:64], target[0], rtol=0, atol=.0009)
            torch.testing.assert_close(decoded.rotation[:64], target[1], rtol=0, atol=.0009)
            self.assertEqual(decoded.owner[-1].item(), decoded.owner[0].item())
            self.assertEqual(restored.frame_reference_state.digest(), frame0)

    def test_pose_valid_new_leaf_without_residual_gain_is_rejected_atomically(self):
        model = self.fixture()
        # Positional sharing already explains these points. Their different
        # covariance rotations cannot be canceled by another positional fit.
        with torch.no_grad():
            model.gs_rotation[:8, 0] = torch.tensor([.006, -.006]).repeat(4)
        target = self.target(model)
        before = copy.deepcopy(model.capture())
        result = model._apply_persistent_local_edit(*target, new_leaves=[dict(
            region_id=3, source_region_id=1, rows=torch.arange(8))])
        self.assertFalse(result['accepted'], result)
        self.assertEqual(model.persistent_state.next_region_id, before['persistent_state']['next_region_id'])
        self.assertTrue(torch.equal(model.owner, before['tensors']['owner']))
        self.assertTrue(torch.equal(model.pivots, before['tensors']['pivots']))
        for name, value in before['parameters'].items():
            torch.testing.assert_close(getattr(model, name), value, rtol=0, atol=0)

    def test_final_packet_guard_failure_rolls_back_entire_local_edit(self):
        model = self.fixture()
        with torch.no_grad():
            model.gs_translation[:8, 0] += .006
        target = self.target(model)
        original = copy.deepcopy(model.capture())
        score = model._online_packet_score

        def invalid_new_layout(*args, **kwargs):
            report = score(*args, **kwargs)
            return {**report, 'valid': False} if len(model.parents) > 3 else report

        with patch.object(model, '_online_packet_score', side_effect=invalid_new_layout):
            result = model._apply_persistent_local_edit(*target, new_leaves=[dict(
                region_id=3, source_region_id=1, rows=torch.arange(8))])
        self.assertFalse(result['accepted'], result)
        self.assertEqual(model.persistent_state.next_region_id, 3)
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2])
        self.assertTrue(torch.equal(model.owner, original['tensors']['owner']))
        for name, value in original['parameters'].items():
            torch.testing.assert_close(getattr(model, name), value, rtol=0, atol=0)

    def test_world_identity_is_composed_not_inferred_from_child_local_zero(self):
        model = self.fixture()
        model.config.persistent_regions.tree_enabled = True
        model.parents = torch.tensor([-1, 3, 3, 0])
        model.pivots = torch.cat((model.pivots, model.base_xyz[:56].mean(0, keepdim=True)))
        model.persistent_state.region_ids = torch.tensor([0, 1, 2, 3])
        model.persistent_state.next_region_id = 4
        model.region_translation = torch.nn.Parameter(torch.zeros(4, 3))
        model.region_rotation = torch.nn.Parameter(torch.zeros(4, 3))
        with torch.no_grad():
            model.region_translation[3, 0] = .02
            model.region_translation[2, 0] = -.02
            model.gs_translation[48, 1] = .003
        model._make_optimizer()
        measure = model._persistent_world_zero_measurements()
        self.assertFalse(measure[1]['world_shared_zero'], 'Local-zero child still moves with its parent')
        self.assertTrue(measure[2]['world_shared_zero'], 'Nonzero local correction can cancel its parent exactly')
        target = self.target(model)
        gs = model.gs_translation.detach().clone(), model.gs_rotation.detach().clone()
        result = model._apply_persistent_local_edit(*target, redirects={2: 0})
        self.assertTrue(result['accepted'], result)
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 3])
        self.assertEqual(model.parents.tolist(), [-1, 2, 0])
        self.assertEqual(model._persistent_owner_ids()[48:56].tolist(), [0] * 8)
        torch.testing.assert_close(model.gs_translation, gs[0], rtol=0, atol=0)
        torch.testing.assert_close(model.gs_rotation, gs[1], rtol=0, atol=0)
        torch.testing.assert_close(model.forward(100)[0], target[0], rtol=0, atol=0)
        torch.testing.assert_close(model.forward(100)[1], target[1], rtol=0, atol=0)

    @staticmethod
    def evidence_count(model, identifier):
        evidence = model.persistent_state.subset_evidence
        where = np.flatnonzero(evidence.ids == identifier)
        return int(evidence.counts[where[0]]) if len(where) else None

    def configure_local(self, model):
        cfg = model.config.persistent_regions
        cfg.local_maintenance_enabled = True
        cfg.structure_interval = 1
        cfg.max_leaves = 2
        cfg.local_subset_max_new_leaves = 1
        model.config.root_motion_discovery.enabled = True
        model.config.validate()

    def local_frame(self, model, decoded, frame, missing=False, migration_case=False):
        model.setup(decoded.xyz, decoded.rotation, 1., frame_index=frame)
        model.observations.fill_(10)
        owners = model._persistent_owner_ids().clone()
        with torch.no_grad():
            if migration_case:
                model.gs_translation[owners == 1, 0] = .006
                model.gs_translation[0, 0] = 0.
            else:
                # All source members share meaningful motion, preventing the
                # legacy migration policy legitimately returning a stationary
                # majority to root before the new subset mechanism is tested.
                model.gs_translation[owners == 1, 0] = .02
                model.gs_translation[:8, 0] += .006
                if len(model.owner) > 64:
                    model.gs_translation[64, 0] = .026
            # Positional sharing of leaf2 is WORLD identity, while one GS
            # retains independent covariance rotation. It is safe to recycle.
            model.gs_rotation[48, 0] = .006
        if missing:
            model.observations[:1 if migration_case else 8] = 0
        target = self.target(model)
        previous_evidence_frame = model.persistent_state.subset_evidence.last_frame
        with patch.object(self.flat, 'propose_motion_regions',
                          side_effect=AssertionError('Local maintenance cannot globally recluster')):
            model.update_topology(100)
        self.assertEqual(model.persistent_state.subset_evidence.last_frame, previous_evidence_frame,
                         'Step100 only peeks; it must not consume this frame')
        return target

    def test_public_three_valid_frames_birth_exclusion_cap_recycle_and_checkpoint(self):
        model = self.fixture()
        self.configure_local(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            decoded = self.lifecycle.finish_export(model, path / 'motion_001.bin')
            self.assertEqual(model.persistent_state.subset_evidence.last_frame, 1)
            for frame in (2, 3, 4, 5):
                target = self.local_frame(model, decoded, frame)
                if frame <= 4:
                    self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2])
                    self.assertEqual(model.persistent_state.next_region_id, 3)
                else:
                    local = model.flat_region_diagnostics['local_maintenance']
                    self.assertEqual(local['accepted_regions'], [dict(region_id=3, source_region_id=1, gs_count=8)])
                    self.assertTrue(local['recycle_trials'][0]['accepted'])
                    self.assertEqual(local['recycle_trials'][0]['region_id'], 2)
                    self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 3])
                    self.assertEqual(model._persistent_owner_ids()[:8].tolist(), [3] * 8)
                    self.assertEqual(model._persistent_owner_ids()[64].item(), 1,
                                     'The newborn has only two completed observations, not three')
                    self.assertEqual(model._persistent_owner_ids()[48:56].tolist(), [0] * 8)
                    self.assertEqual(model.region_map.tolist(), [0, 1, 0])
                    self.assertEqual(model.persistent_state.next_region_id, 4)
                    self.assertEqual(model._values(hard=True)[3][48, 0].item(), 6,
                                     'Returning WORLD-zero sharing to root must retain independent rotation')
                    self.assertFalse(model.flat_region_diagnostics['root_discovery']['accepted_regions'])
                    self.assertEqual(model.flat_region_diagnostics['root_discovery']['available_leaves'], 0)
                restored = self.rigid.RigidResidualTree(model.config)
                restored.restore(copy.deepcopy(model.capture()))
                self.assertEqual(restored.persistent_state.last_local_frame, model.persistent_state.last_local_frame)
                np.testing.assert_array_equal(restored.persistent_state.subset_evidence.counts,
                                              model.persistent_state.subset_evidence.counts)
                model = restored
                decoded = self.lifecycle.finish_export(model, path / f'motion_{frame:03d}.bin',
                                                       birth_source=0 if frame == 2 else None)
                self.assertEqual(model.persistent_state.subset_evidence.last_frame, frame)
                if frame <= 4:
                    self.assertEqual(self.evidence_count(model, 0), frame - 1)
                if frame in (3, 4):
                    self.assertEqual(self.evidence_count(model, 64), frame - 2)
                torch.testing.assert_close(decoded.xyz[:len(target[0])], target[0], rtol=0, atol=.0009)
            self.assertEqual(len(model.online_history.records), 5)
            model.setup(decoded.xyz, decoded.rotation, 1., frame_index=6)
            self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 3])
            self.assertEqual(model.online_ids[-1].item(), 64)
            self.assertEqual(model.frame_reference_state.digest(), decoded.digest())

    def test_missing_final_observation_holds_subset_streak_without_premature_edit(self):
        model = self.fixture()
        self.configure_local(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            decoded = self.lifecycle.finish_export(model, path / 'motion_001.bin')
            for frame in (2, 3, 4, 5, 6):
                self.local_frame(model, decoded, frame, missing=frame == 3)
                if frame < 6:
                    self.assertEqual(model._persistent_owner_ids()[:8].tolist(), [1] * 8)
                else:
                    self.assertEqual(model._persistent_owner_ids()[:8].tolist(), [3] * 8)
                decoded = self.lifecycle.finish_export(model, path / f'motion_{frame:03d}.bin')
                expected = {2: 1, 3: 1, 4: 2, 5: 3, 6: 0}[frame]
                self.assertEqual(self.evidence_count(model, 0), expected)

    def test_actual_migration_and_missing_mask_reset_subset_evidence_by_logical_owner(self):
        model = self.fixture()
        self.configure_local(model)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            decoded = self.lifecycle.finish_export(model, path / 'motion_001.bin')
            for frame in (2, 3, 4):
                self.local_frame(model, decoded, frame, missing=frame == 3, migration_case=True)
                if frame < 4:
                    self.assertEqual(model._persistent_owner_ids()[0].item(), 1)
                    self.assertEqual(model.persistent_state.migration_pending[0], dict(target=0, streak=1))
                else:
                    self.assertEqual(model._persistent_owner_ids()[0].item(), 0)
                    self.assertNotIn(0, model.persistent_state.migration_pending)
                decoded = self.lifecycle.finish_export(model, path / f'motion_{frame:03d}.bin')
                self.assertEqual(self.evidence_count(model, 0), 1 if frame < 4 else 0)

    def test_legacy_split_failed_fit_does_not_leave_parent_override_or_consume_id(self):
        model = self.fixture()
        with torch.no_grad():
            model.gs_translation[24:48, 0] += .006
        target = self.target(model)
        history = model._persistent_current_history(*target)
        original = copy.deepcopy(model.capture())
        with patch.object(model, '_try_persistent_owner_ids', return_value=False) as fit:
            self.assertFalse(model._split_persistent_region(1, *target, history))
        fit.assert_called_once()
        self.assertEqual(model.persistent_state.parent_overrides, original['persistent_state']['parent_overrides'])
        self.assertEqual(model.persistent_state.next_region_id, 3)
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2])
        self.assertTrue(torch.equal(model.owner, original['tensors']['owner']))
        torch.testing.assert_close(model.pivots, original['tensors']['pivots'], rtol=0, atol=0)

    def test_final_evidence_exception_rolls_back_legacy_subset_world_and_marker(self):
        model = self.fixture()
        self.configure_local(model)
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.lifecycle.finish_export(model, Path(directory) / 'motion_001.bin')
            self.local_frame(model, decoded, 2)
            model.forward(200, training=True)
            original = model.persistent_state.capture()
            with patch.object(model, '_persistent_world_zero_measurements', side_effect=RuntimeError('Injected after subset observation')):
                with self.assertRaisesRegex(RuntimeError, 'Injected'):
                    model._finalize_persistent_evidence()
            state = model.persistent_state
            self.assertEqual(state.last_evidence_frame, original['last_evidence_frame'])
            self.assertEqual(state.region_evidence, original['region_evidence'])
            self.assertEqual(state.world_zero_evidence, original['world_zero_evidence'])
            self.assertEqual(state.subset_last_observation, original['subset_last_observation'])
            self.assertEqual(state.subset_evidence.last_frame, original['subset_evidence']['last_frame'])
            np.testing.assert_array_equal(state.subset_evidence.ids, original['subset_evidence']['ids'])
            np.testing.assert_array_equal(state.subset_evidence.leaves, original['subset_evidence']['leaves'])
            np.testing.assert_array_equal(state.subset_evidence.counts, original['subset_evidence']['counts'])
            model._finalize_persistent_evidence()
            self.assertEqual(state.last_evidence_frame, 2)
            self.assertEqual(self.evidence_count(model, 0), 1)
            model._finalize_persistent_evidence()
            self.assertEqual(self.evidence_count(model, 0), 1, 'Same final frame must not advance twice')


if __name__ == '__main__':
    unittest.main()
