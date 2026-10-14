"""CPU tests for an unactivated causal evidence helper; no renderer required."""

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline' / 'ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
SUBSET = importlib.import_module('pipeline.ReCon-GS.model.sustained_residual_subset')


class SustainedResidualSubsetTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(100, 120)
        self.leaves = np.full(20, 70)
        self.symbols = np.zeros((20, 6), np.int16)
        self.symbols[:8, 0] = 30
        self.observed = np.ones(20, bool)
        self.evidence = SUBSET.SustainedResidualSubsetEvidence()

    def observe(self, frame, **kwargs):
        return self.evidence.observe(frame, kwargs.get('ids', self.ids),
            kwargs.get('leaves', self.leaves), kwargs.get('symbols', self.symbols),
            kwargs.get('observed', self.observed), 4.)

    def test_three_valid_observations_and_missing_hold(self):
        self.assertFalse(self.observe(1).candidate_mask.any())
        hidden = self.observed.copy(); hidden[:8] = False
        missing = self.symbols.copy(); missing[:8] = 0
        hold = self.observe(2, observed=hidden, symbols=missing)
        np.testing.assert_array_equal(hold.streaks[:8], 1)
        self.assertFalse(hold.candidate_mask.any())
        self.assertFalse(self.observe(3).candidate_mask.any())
        result = self.observe(4)
        np.testing.assert_array_equal(np.flatnonzero(result.candidate_mask), np.arange(8))
        self.assertEqual(result.groups[0]['region_id'], 70)
        self.assertEqual(result.groups[0]['integer_l1'], 240)
        self.assertEqual(result.groups[0]['retained_count'], 12)
        self.assertEqual(result.groups[0]['active_translation_blocks'], 8)
        self.assertEqual(result.groups[0]['active_rotation_blocks'], 0)
        json.dumps(result.metrics, allow_nan=False)
        # Missing after eligibility preserves evidence but proposes no hidden GS.
        held = self.observe(5, observed=hidden, symbols=missing)
        self.assertEqual(held.metrics['missing_held_gs'], 8)
        self.assertFalse(held.candidate_mask.any())

    def test_observed_low_resets_and_exact_threshold_is_low(self):
        self.observe(1); self.observe(2)
        threshold = self.symbols.copy(); threshold[:8, 0] = 4
        result = self.observe(3, symbols=threshold)
        self.assertFalse(result.streaks.any())
        self.assertFalse(self.observe(4).candidate_mask.any())
        self.assertFalse(self.observe(5).candidate_mask.any())
        self.assertTrue(self.observe(6).candidate_mask[:8].all())

    def test_migration_hidden_reset_and_birth_has_no_past(self):
        self.observe(1); self.observe(2)
        changed = self.leaves.copy(); changed[:8] = 999
        hidden = self.observed.copy(); hidden[:8] = False
        migrated = self.observe(3, leaves=changed, observed=hidden)
        self.assertFalse(migrated.streaks[:8].any())
        self.assertEqual(migrated.metrics['migrated_gs'], 8)
        self.assertFalse(self.observe(4, leaves=changed).candidate_mask.any())
        ids = self.ids.copy(); ids[:8] += 1000
        born = self.observe(5, leaves=changed, ids=ids)
        np.testing.assert_array_equal(born.streaks[:8], 1)
        self.assertEqual(born.metrics['new_gs'], 8)
        self.assertFalse(born.candidate_mask.any())

    def test_stable_ids_reorder_checkpoint_and_unchanged_input(self):
        original = self.symbols.copy()
        self.observe(1); self.observe(2)
        state = self.evidence.state_dict()
        restored = SUBSET.SustainedResidualSubsetEvidence()
        restored.load_state_dict(state)
        state['counts'][:] = 0  # Restore does not retain aliases.
        order = np.random.default_rng(23).permutation(len(self.ids))
        result = restored.observe(3, self.ids[order], self.leaves[order],
            self.symbols[order], self.observed[order], 4.)
        selected_ids = self.ids[order][result.candidate_mask]
        np.testing.assert_array_equal(np.sort(selected_ids), self.ids[:8])
        np.testing.assert_array_equal(self.symbols, original)
        self.assertEqual(result.groups[0]['integer_l1'], 240)

    def test_root_and_entire_leaf_not_split_and_signed_rotation_cost(self):
        root = np.zeros(20, np.int64)
        for frame in range(1, 4):
            result = self.observe(frame, leaves=root)
        self.assertFalse(result.streaks.any())
        self.evidence = SUBSET.SustainedResidualSubsetEvidence()
        all_high = np.full((20, 6), -5, np.int16)
        for frame in range(1, 4):
            result = self.observe(frame, symbols=all_high)
        self.assertFalse(result.candidate_mask.any())
        self.assertEqual(result.metrics['region_rejections'][0]['reason'], 'insufficient_retained_members')
        self.evidence = SUBSET.SustainedResidualSubsetEvidence()
        rotation = np.zeros((20, 6), np.int16); rotation[:8, 5] = -30
        for frame in range(1, 4):
            result = self.observe(frame, symbols=rotation)
        self.assertEqual(result.groups[0]['active_rotation_blocks'], 8)
        self.assertEqual(result.groups[0]['integer_l1'], 240)

    def test_large_leaf_p90_does_not_hide_supported_minority(self):
        ids = np.arange(200)
        leaves = np.full(200, 110)
        symbols = np.zeros((200, 6), np.int16); symbols[:8, :3] = 10
        self.assertEqual(float(np.quantile(np.abs(symbols).mean(-1), .9)), 0.)
        for frame in range(1, 4):
            result = self.evidence.observe(frame, ids, leaves, symbols, np.ones(200, bool), 4.)
        self.assertEqual(result.metrics['candidate_gs'], 8)
        self.assertEqual(result.groups[0]['retained_count'], 192)

    def test_invalid_float_symbols_and_replayed_frames_rejected(self):
        with self.assertRaises(ValueError):
            self.observe(1, symbols=self.symbols.astype(float))
        self.observe(1)
        with self.assertRaises(ValueError):
            self.observe(1)
        with self.assertRaises(ValueError):
            self.observe(0)

    def test_peek_does_not_consume_frame_or_inherit_new_leaf_streak(self):
        self.observe(1); self.observe(2)
        for _ in range(2):
            before_third = self.evidence.peek(3, self.ids, self.leaves, self.symbols, self.observed, 4.)
            self.assertFalse(before_third.candidate_mask.any())
            self.assertFalse(before_third.metrics['current_frame_counted'])
        self.assertEqual(self.evidence.last_frame, 2)
        self.observe(3)
        original = self.evidence.state_dict()
        current = self.evidence.peek(4, self.ids, self.leaves, self.symbols, self.observed, 4.)
        self.assertEqual(current.metrics['candidate_gs'], 8)
        changed = self.leaves.copy(); changed[:8] = 999
        moved = self.evidence.peek(4, self.ids, changed, self.symbols, self.observed, 4.)
        self.assertFalse(moved.candidate_mask.any())
        np.testing.assert_array_equal(self.evidence.counts, original['counts'])
        np.testing.assert_array_equal(self.evidence.leaves, original['leaves'])
        self.assertEqual(self.evidence.last_frame, 3)

    def test_block_l2_does_not_use_old_mean_six_or_mix_translation_rotation(self):
        symbols = np.zeros((20, 6), np.int16)
        symbols[:8, :2] = 3  # L2=sqrt(18)>4, mean6=1, so old gate would miss it.
        for frame in range(1, 4):
            result = self.observe(frame, symbols=symbols)
        self.assertEqual(result.metrics['candidate_gs'], 8)
        separated = np.zeros((20, 6), np.int16)
        separated[:8, [0, 3]] = 3  # Both block norms 3; combined L2 would falsely pass.
        self.assertFalse(self.observe(4, symbols=separated).streaks.any())

    def test_local_grouping_stays_inside_selected_rows_and_bounds_mask_buckets(self):
        clustering = importlib.import_module('pipeline.ReCon-GS.model.online_motion_clustering')
        ids = np.arange(32)
        source = np.random.default_rng(7).normal(scale=.001, size=(32, 3))
        source[:8, 0] -= .4; source[8:16, 0] += .4; source[16:24, 0] += 2.
        history = clustering.OnlineMotionHistory()
        for frame in range(1, 4):
            mask = np.ones(32, bool)
            if frame == 1: mask[8:16] = False
            if frame == 2: mask[16:24] = False
            history.append(frame, ids, source, source + .006, observed=mask)
        groups, metrics = SUBSET.propose_local_residual_groups(history, ids, source, np.arange(24),
            1., clustering.OnlineMotionClusteringConfig(), .05, 8, max_groups=2)
        self.assertLessEqual(len(groups), 2)
        self.assertEqual(sum(len(group) for group in groups), 16)
        self.assertTrue(all(np.isin(group, np.arange(24)).all() for group in groups))
        self.assertFalse(metrics['generic_feature_distance_gate'])
        json.dumps(metrics, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
