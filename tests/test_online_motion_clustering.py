"""CPU-only causal evidence and flat clustering regression checks."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


PATH = Path(__file__).resolve().parents[1] / 'pipeline/ReCon-GS/model/online_motion_clustering.py'
SPEC = importlib.util.spec_from_file_location('online_motion_clustering_test', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
History = MODULE.OnlineMotionHistory
Config = MODULE.OnlineMotionClusteringConfig
propose = MODULE.propose_motion_regions


def config(method, **kwargs):
    values = dict(method=method, min_region_size=4, hdbscan_min_samples=3,
                  translation_scale_extent=.01, static_translation_extent=.001,
                  greedy_fit_tolerance_extent=.002, component_counts=[1, 2, 4])
    values.update(kwargs)
    return Config(**values)


class OnlineMotionClusteringTests(unittest.TestCase):
    def test_frame_zero_then_one_observation_is_sufficient(self):
        ids, source = np.arange(12), np.zeros((12, 3))
        for method in ('kmeans', 'gmm', 'hdbscan', 'greedy'):
            with self.subTest(method=method):
                history = History()
                empty = propose(history, ids, 1., config(method))
                self.assertEqual(empty.groups, [])
                self.assertTrue(np.all(empty.labels == -3))
                history.append(1, ids, source, source + [.1, 0, 0])
                result = propose(history, ids, 1., config(method))
                self.assertEqual(len(result.groups), 1)
                self.assertTrue(np.all(result.labels == 0))
                self.assertEqual(result.metrics['history_transitions'], 1)

    def test_missing_and_birth_are_not_zero_motion_evidence(self):
        history = History()
        ids = np.arange(12)
        source = np.zeros((12, 3))
        history.append(1, ids[:8], source[:8], source[:8] + [.1, 0, 0])
        observed = np.ones(12, dtype=bool)
        observed[0] = False
        source[0] = np.nan
        history.append(2, ids, source, source + [.1, 0, 0], observed=observed)
        aligned = history.aligned(ids[::-1])
        self.assertFalse(aligned['observed'][0, 0])  # New GS ID11 before birth.
        self.assertFalse(aligned['observed'][-1, 1])  # ID0 lacks current evidence.
        for method in ('kmeans', 'gmm', 'hdbscan', 'greedy'):
            result = propose(history, ids, 1., config(method))
            self.assertEqual(result.labels[0], -3)
            self.assertFalse(np.any(result.labels[1:] == -2))
            self.assertEqual(result.metrics['history_transitions'], 2)
            self.assertEqual(sorted(len(p['observed_transitions'])
                                    for p in result.metrics['patterns']), [1, 2])

    def test_history_windows_two_to_five_use_actual_transitions(self):
        ids = np.arange(24)
        source = np.random.default_rng(3).normal(size=(24, 3))
        history = History()
        for frame in range(1, 7):
            target = source + [.1, -.02 * frame, 0]
            history.append(frame, ids, source, target)
            source = target
            aligned = history.aligned(ids)
            self.assertEqual(aligned['observed'].shape, (24, min(frame, 5)))
            features = MODULE._features(aligned, ids, np.arange(min(frame, 5)), 1., config('kmeans'))
            self.assertEqual(features.shape, (24, 3 * min(frame, 5)))
        self.assertEqual(history.aligned(ids)['frames'], [2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            history.append(6, ids, source, source)
        with self.assertRaises(ValueError):
            history.append(4, ids, source, source)

    def test_full_history_distinguishes_identical_latest_motion(self):
        ids = np.arange(48)
        source = np.random.default_rng(4).normal(size=(48, 3))
        first = np.tile([.12, 0, 0], (48, 1))
        first[24:, 0] = -.12
        history = History()
        history.append(1, ids, source, source + first)
        history.append(2, ids, source + first, source + first + [.1, 0, 0])
        for method in ('kmeans', 'gmm', 'hdbscan', 'greedy'):
            with self.subTest(method=method):
                result = propose(history, ids, 1., config(method))
                self.assertGreaterEqual(len(result.groups), 2)
                for part in result.groups:
                    self.assertTrue(np.all(part < 24) or np.all(part >= 24))

    def test_static_unknown_noise_and_optional_covariance_rotation(self):
        ids, source = np.arange(12), np.zeros((12, 3))
        target = source.copy()
        target[8:] = [.2, 0, 0]
        observed = np.ones(12, dtype=bool)
        observed[11] = False
        history = History()
        q = np.tile([.70710678, 0, 0, .70710678], (12, 1))
        history.append(1, ids, source, target, q, observed)
        for method in ('kmeans', 'gmm', 'hdbscan', 'greedy'):
            result = propose(history, ids, 1., config(method))
            self.assertTrue(np.all(result.labels[:8] == -2))
            self.assertTrue(np.all(result.labels[8:11] == -1))
            self.assertEqual(result.labels[11], -3)
        rotated = propose(history, ids, 1., config('kmeans', rotation_feature_weight=1.))
        self.assertFalse(np.any(rotated.labels[:8] == -2))

    def test_greedy_fits_shared_rotation_and_can_rebuild_split(self):
        ids = np.arange(96)
        source = np.random.default_rng(6).normal(size=(96, 3))
        rotation = Rotation.from_rotvec([0, 0, .2]).as_matrix()
        target = source @ rotation.T + [.12, -.03, .02]
        history = History(1)
        history.append(1, ids, source, target)  # No GS covariance rotation evidence.
        result = propose(history, ids, 1., config('greedy', history_frames=1,
                                                greedy_neighbors=12))
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(len(result.groups[0]), 96)
        displacement = np.tile([.15, 0, 0], (96, 1))
        displacement[48:, 0] = -.15
        history.append(2, ids, target, target + displacement)
        split = propose(history, ids, 1., config('greedy', history_frames=1))
        self.assertEqual(len(split.groups), 2)
        self.assertTrue(all(np.all(p < 48) or np.all(p >= 48) for p in split.groups))

    def test_stable_ids_make_sampling_and_partition_permutation_invariant(self):
        ids = np.arange(120) * 13 + 7
        source = np.random.default_rng(8).normal(size=(120, 3))
        displacement = np.tile([.2, 0, 0], (120, 1))
        displacement[60:, 0] = -.2
        history = History()
        history.append(1, ids, source, source + displacement)
        permutation = np.random.default_rng(9).permutation(120)
        for method in ('kmeans', 'gmm', 'hdbscan', 'greedy'):
            cfg = config(method, max_fit_points=40)
            before = propose(history, ids, 1., cfg)
            after = propose(history, ids[permutation], 1., cfg)
            np.testing.assert_array_equal(after.labels, before.labels[permutation])
            np.testing.assert_array_equal(propose(history, ids, 1., cfg).labels, before.labels)

    def test_history_copies_inputs_and_checkpoint_does_not_alias(self):
        ids, source = np.arange(8), np.zeros((8, 3))
        target = source + [.1, 0, 0]
        history = History()
        history.append(1, ids, source, target)
        target[:] = 99
        state = history.state_dict()
        restored = History()
        restored.load_state_dict(state)
        state['records'][0]['target'][:] = -99
        np.testing.assert_allclose(restored.aligned(ids)['target'][:, 0, 0], .1)
        np.testing.assert_array_equal(restored.aligned(ids)['target'], history.aligned(ids)['target'])
        bad = dict(records=history.state_dict()['records'] * 2)
        with self.assertRaises(ValueError):
            restored.load_state_dict(bad)


if __name__ == '__main__':
    unittest.main()
