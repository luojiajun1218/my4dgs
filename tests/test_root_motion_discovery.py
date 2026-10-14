"""CPU checks for causal, local root-only discovery with real missing masks."""

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

DISCOVERY = importlib.import_module('pipeline.ReCon-GS.model.root_motion_discovery')
CLUSTERING = importlib.import_module('pipeline.ReCon-GS.model.online_motion_clustering')


def configs():
    return (CLUSTERING.OnlineMotionClusteringConfig(method='kmeans',
                translation_scale_extent=.004, static_translation_extent=.004),
            DISCOVERY.RootMotionDiscoveryConfig(enabled=True))


def propose(history, ids, root_mask=None, **kwargs):
    clustering, discovery = configs()
    if root_mask is None:
        root_mask = np.ones(len(ids), bool)
    return DISCOVERY.propose_root_motion_regions(history, ids, root_mask, 1., np.full(3, .001),
                                                 clustering, discovery, **kwargs)


class RootMotionDiscoveryTests(unittest.TestCase):
    def test_frame_zero_one_transition_and_two_valid_motion_observations(self):
        ids, source = np.arange(16), np.zeros((16, 3))
        history = CLUSTERING.OnlineMotionHistory()
        self.assertEqual(propose(history, ids).groups, [])
        history.append(1, ids, source, source + [.006, 0, 0])
        self.assertEqual(propose(history, ids).groups, [])
        history.append(2, ids, source, source + [.006, 0, 0])
        result = propose(history, ids)
        self.assertEqual(result.metrics['sustained_moving_root_gs'], 16)
        self.assertTrue(result.groups)
        np.testing.assert_array_equal(np.sort(np.concatenate(result.groups)), ids)
        json.dumps(result.metrics, allow_nan=False)

    def test_missing_holds_evidence_but_current_missing_never_proposes(self):
        ids, source = np.arange(16), np.zeros((16, 3))
        history = CLUSTERING.OnlineMotionHistory()
        history.append(1, ids, source, source + [.006, 0, 0])
        history.append(2, ids, np.full_like(source, np.nan), np.full_like(source, np.nan),
                       observed=np.zeros(len(ids), bool))
        self.assertEqual(propose(history, ids).groups, [])
        history.append(3, ids, source, source + [.006, 0, 0])
        result = propose(history, ids)
        self.assertEqual(result.metrics['sustained_moving_root_gs'], 16)
        self.assertTrue(result.groups)
        self.assertEqual(result.metrics['patterns'][0]['observed_transitions'], [0, 2])

    def test_observed_stationary_resets_and_old_motion_cannot_fill_a_streak(self):
        ids, source = np.arange(16), np.zeros((16, 3))
        history = CLUSTERING.OnlineMotionHistory()
        history.append(1, ids, source, source + [.006, 0, 0])
        history.append(2, ids, source, source)
        history.append(3, ids, source, source + [.006, 0, 0])
        result = propose(history, ids)
        self.assertEqual(result.groups, [])
        self.assertEqual(result.metrics['stationary_reset_gs'], 16)
        history.append(4, ids, source, source + [.006, 0, 0])
        self.assertTrue(propose(history, ids).groups)

    def test_half_quantization_noise_and_observed_zero_do_not_count_as_motion(self):
        ids, source = np.arange(16), np.zeros((16, 3))
        history = CLUSTERING.OnlineMotionHistory()
        history.append(1, ids, source, source + .00049)
        history.append(2, ids, source, source + [.006, 0, 0])
        self.assertEqual(propose(history, ids).groups, [])
        history.append(3, ids, source, source + .00049)
        result = propose(history, ids)
        self.assertEqual(result.groups, [])
        self.assertTrue(np.all(result.labels == -2))

    def test_birth_has_no_fabricated_history_and_nonroot_is_never_selected(self):
        ids, source = np.arange(24), np.zeros((24, 3))
        history = CLUSTERING.OnlineMotionHistory()
        history.append(1, ids[:16], source[:16], source[:16] + [.006, 0, 0])
        history.append(2, ids, source, source + [.006, 0, 0])
        root_mask = np.ones(24, bool)
        root_mask[8:16] = False
        result = propose(history, ids, root_mask)
        self.assertEqual(result.metrics['sustained_moving_root_gs'], 8)
        self.assertTrue(result.groups)
        self.assertTrue(all(np.all(group < 8) for group in result.groups))
        self.assertTrue(np.all(result.labels[8:16] == -3))  # Outside discovery scope.
        self.assertTrue(np.all(result.labels[16:] == -3))  # Unknown before birth.

    def test_spatial_features_separate_distant_objects_with_identical_motion(self):
        ids = np.arange(32)
        source = np.random.default_rng(41).normal(scale=.001, size=(32, 3))
        source[:16, 0] -= 1
        source[16:, 0] += 1
        history = CLUSTERING.OnlineMotionHistory()
        for frame in (1, 2):
            history.append(frame, ids, source, source + [.006, 0, 0])
        result = propose(history, ids)
        self.assertEqual(len(result.groups), 2)
        for group in result.groups:
            self.assertTrue(np.all(group < 16) or np.all(group >= 16))
        self.assertEqual(result.metrics['patterns'][0]['motion_feature_dimensions'], 6)
        self.assertEqual(result.metrics['patterns'][0]['total_feature_dimensions'], 9)

    def test_multiple_patterns_share_global_cap_and_demand_priority(self):
        ids, source = np.arange(24), np.zeros((24, 3))
        history = CLUSTERING.OnlineMotionHistory()
        first, second = np.ones(24, bool), np.ones(24, bool)
        first[8:16] = False
        second[16:] = False
        history.append(1, ids, source, source + [.006, 0, 0], observed=first)
        history.append(2, ids, source, source + [.006, 0, 0], observed=second)
        history.append(3, ids, source, source + [.006, 0, 0])
        priority = np.repeat([1., 2., 3.], 8)
        result = propose(history, ids, priorities=priority)
        self.assertEqual(result.metrics['candidates_before_cap'], 3)
        self.assertEqual(len(result.groups), 2)
        np.testing.assert_array_equal(result.groups[0], ids[16:])
        np.testing.assert_array_equal(result.groups[1], ids[8:16])
        self.assertEqual(result.metrics['capacity_dropped_gs'], 8)
        one = propose(history, ids, priorities=priority, available_leaves=1)
        self.assertEqual(len(one.groups), 1)
        np.testing.assert_array_equal(one.groups[0], ids[16:])
        none = propose(history, ids, available_leaves=0)
        self.assertEqual(none.groups, [])
        self.assertEqual(none.metrics['rejection_reason'], 'no_available_leaves')
        json.dumps(result.metrics, allow_nan=False)

    def test_stable_id_permutation_preserves_candidate_membership_and_inputs(self):
        ids = np.arange(32) * 13 + 5
        source = np.zeros((32, 3))
        source[:16, 0], source[16:, 0] = -1, 1
        history = CLUSTERING.OnlineMotionHistory()
        for frame in (1, 2):
            permutation = np.random.default_rng(frame).permutation(32)
            history.append(frame, ids[permutation], source[permutation], source[permutation] + [.006, 0, 0])
        state_before = history.state_dict()
        expected = propose(history, ids)
        permutation = np.random.default_rng(92).permutation(32)
        actual = propose(history, ids[permutation])
        np.testing.assert_array_equal(actual.labels, expected.labels[permutation])
        self.assertEqual([set(ids[group]) for group in expected.groups],
                         [set(ids[permutation][group]) for group in actual.groups])
        for before, after in zip(state_before['records'], history.records):
            for field in ('ids', 'source', 'target', 'observed'):
                np.testing.assert_array_equal(before[field], after[field])

    def test_root_keeps_valid_motion_groups_when_generic_feature_distance_would_reject(self):
        ids = np.arange(64)
        source = np.random.default_rng(13).normal(scale=.001, size=(64, 3))
        source[:, 0] = np.linspace(-2., 2., 64)
        history = CLUSTERING.OnlineMotionHistory()
        for frame in (1, 2):
            history.append(frame, ids, source, source + [.006, 0, 0])
        result = propose(history, ids)
        self.assertEqual(result.metrics['sustained_moving_root_gs'], 64)
        self.assertFalse(result.metrics['generic_feature_distance_rejection_applied'])
        self.assertGreater(result.metrics['generic_feature_distance_would_reject_gs'], 0)
        self.assertEqual(sum(len(group) for group in result.groups), 64)
        self.assertLessEqual(len(result.groups), 2)
        # The shared helper's existing default still rejects feature outliers.
        clustering, _ = configs()
        from dataclasses import replace
        clustering = replace(clustering, component_counts=[1, 2])
        window = history.aligned(ids)
        motion = CLUSTERING._features(window, ids, np.arange(2), 1., clustering)
        features = np.concatenate((motion, source / .05), axis=1)
        default_labels, detail = CLUSTERING._statistical_partition(features, clustering, 8192)
        self.assertTrue(detail['feature_distance_rejection_applied'])
        self.assertTrue(np.any(default_labels < 0))


if __name__ == '__main__':
    unittest.main()
