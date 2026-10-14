"""Pure CPU causal initialization tests; this helper is not wired to training."""

import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'), ('pipeline.ReCon-GS', ROOT / 'pipeline/ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name); package.__path__ = [str(directory)]; sys.modules[name] = package
PREDICTION = importlib.import_module('pipeline.ReCon-GS.model.causal_translation_prediction')


def history(ids, older, latest, masks=None):
    masks = np.ones((len(ids), 2), bool) if masks is None else masks
    source = np.random.default_rng(13).normal(size=(len(ids), 3))
    records = []
    for frame, displacement in ((1, older), (2, latest)):
        target = source + displacement
        records.append(dict(frame_index=frame, ids=np.asarray(ids).copy(), source=source.copy(),
            target=target.copy(), observed=masks[:, frame - 1].copy()))
        source = target
    return SimpleNamespace(records=records)


class CausalTranslationPredictionTests(unittest.TestCase):
    def propose(self, records, ids, frame=3):
        return PREDICTION.propose_causal_translation_initialization(records, np.asarray(ids), frame,
            [.001] * 3, 1., SimpleNamespace(static_translation_extent=.004), gs_xyz_step_extent=.001)

    def test_only_past_full_WORLD_translation_and_no_state_mutation(self):
        ids = np.array([90, 2, 71])
        previous = np.array([[.006, 0, 0], [0, -.008, 0], [.006, .006, 0]])
        latest = previous * 1.5
        records = history(ids, previous, latest)
        saved = copy.deepcopy(records.records)
        result = self.propose(records, ids)
        self.assertTrue(result.predicted_mask.all())
        self.assertEqual(result.translation.dtype, np.float32)
        np.testing.assert_allclose(result.translation, latest, atol=1e-8)
        self.assertEqual(result.pending_predicted_ids.tolist(), [2, 71, 90])
        self.assertFalse(result.metrics['current_evidence'])
        self.assertFalse(result.metrics['observations_written'])
        self.assertFalse(result.metrics['rotation_or_region_prediction'])
        for old, new in zip(saved, records.records):
            for key in ('ids', 'source', 'target', 'observed'):
                np.testing.assert_array_equal(old[key], new[key])
        json.dumps(result.metrics, allow_nan=False)

    def test_reversal_and_both_amplitude_directions_rejected(self):
        ids = np.arange(4)
        older = np.tile([.01, 0., 0.], (4, 1))
        latest = np.array([[-.01, 0., 0.], [.021, 0., 0.], [.0049, 0., 0.], [.019, 0., 0.]])
        result = self.propose(history(ids, older, latest), ids)
        self.assertEqual(result.predicted_mask.tolist(), [False, False, False, True])
        self.assertEqual(result.metrics['rejection_counts']['direction_inconsistent'], 1)
        self.assertEqual(result.metrics['rejection_counts']['amplitude_inconsistent'], 2)

    def test_birth_missing_and_nan_placeholders_do_not_create_history(self):
        ids = np.arange(5)
        delta = np.tile([.006, 0., 0.], (5, 1))
        masks = np.ones((5, 2), bool); masks[1, 0] = False; masks[2, 1] = False
        records = history(ids, delta, delta, masks)
        for key in ('ids', 'source', 'target', 'observed'):
            records.records[0][key] = records.records[0][key][:-1]  # Born before the latest transition only.
        records.records[0]['source'][1] = np.nan; records.records[0]['target'][1] = np.nan
        result = self.propose(records, np.arange(6))  # ID5 born after latest record.
        self.assertEqual(result.predicted_mask.tolist(), [True, False, False, True, False, False])
        self.assertEqual(result.metrics['rejection_counts']['no_older_transition_id'], 1)
        self.assertEqual(result.metrics['rejection_counts']['no_latest_transition_id'], 1)
        self.assertEqual(result.metrics['rejection_counts']['older_unobserved'], 1)
        self.assertEqual(result.metrics['rejection_counts']['latest_unobserved'], 1)
        self.assertTrue((result.translation[~result.predicted_mask] == 0).all())

    def test_stable_ID_permutation_preserves_prediction_and_pending_IDs(self):
        ids = np.array([4, 2, 7])
        delta = np.array([[.006, 0, 0], [0, .008, 0], [0, 0, .01]])
        records = history(ids, delta, delta)
        result = self.propose(records, ids[[2, 0, 1]])
        np.testing.assert_allclose(result.translation, delta[[2, 0, 1]], atol=1e-8)
        self.assertEqual(result.pending_predicted_ids.tolist(), [2, 4, 7])
        result.pending_predicted_ids[:] = -1
        np.testing.assert_array_equal(records.records[0]['ids'], ids)

    def test_motion_and_direction_and_amplitude_exact_boundaries(self):
        ids = np.arange(5)
        older = np.array([[.004, 0, 0], [.005, 0, 0], [.006, .006, 0], [.006, 0, 0], [.012, 0, 0]])
        latest = np.array([[.005, 0, 0], [.004, 0, 0], [.006, 0, .006], [.012, 0, 0], [.006, 0, 0]])
        result = self.propose(history(ids, older, latest), ids)
        self.assertEqual(result.predicted_mask.tolist(), [False, False, True, True, True])
        self.assertAlmostEqual(result.metrics['supported_cosine_min'], .5)
        self.assertAlmostEqual(result.metrics['supported_symmetric_amplitude_ratio_max'], 2)

    def test_first_frame_short_stale_gapped_and_future_history_are_zero(self):
        ids = np.array([1])
        delta = np.array([[.006, 0, 0]])
        records = history(ids, delta, delta)
        first = self.propose(SimpleNamespace(records=[]), ids, frame=1)
        self.assertEqual(first.metrics['rejection_reason'], 'first_dynamic_frame')
        short = self.propose(SimpleNamespace(records=records.records[-1:]), ids)
        self.assertEqual(short.metrics['rejection_reason'], 'fewer_than_two_decoded_transitions')
        stale = self.propose(records, ids, frame=4)
        self.assertEqual(stale.metrics['rejection_reason'], 'history_not_latest_two_consecutive_transitions')
        gapped = copy.deepcopy(records); gapped.records[-1]['frame_index'] = 3
        self.assertFalse(self.propose(gapped, ids, frame=4).predicted_mask.any())
        future = self.propose(records, ids, frame=2)
        self.assertEqual(future.metrics['rejection_reason'], 'history_not_strictly_past')
        for result in (first, short, stale, future):
            self.assertFalse(result.predicted_mask.any())
            self.assertTrue((result.translation == 0).all())

    def test_half_GS_step_noise_and_int16_overflow_are_zero(self):
        ids = np.array([1, 2])
        delta = np.array([[.00049, 0, 0], [40., 0, 0]])
        result = self.propose(history(ids, delta, delta), ids)
        self.assertFalse(result.predicted_mask.any())
        self.assertEqual(result.metrics['rejection_counts']['latest_not_moving'], 1)
        self.assertEqual(result.metrics['rejection_counts']['GS_quantization_overflow'], 1)

    def test_real_extent_float32_step_keeps_four_rejected_and_five_selected(self):
        extent = 6.493635559082032
        actual_step = float(np.float32(extent * .0001))
        self.assertGreater(4 * actual_step, .0004 * extent)
        ids = np.array([1, 2])
        delta = np.array([[4 * actual_step, 0., 0.], [5 * actual_step, 0., 0.]])
        result = PREDICTION.propose_causal_translation_initialization(history(ids, delta, delta), ids, 3,
            [actual_step] * 3, extent, SimpleNamespace(static_translation_extent=.0004), gs_xyz_step_extent=.0001)
        self.assertEqual(result.predicted_mask.tolist(), [False, True])
        self.assertEqual(result.metrics['moving_threshold_GS_units'], 4.)

    def test_configured_and_actual_GS_step_mismatch_rejected(self):
        ids = np.array([1]); delta = np.array([[.006, 0., 0.]])
        with self.assertRaises(ValueError):
            PREDICTION.propose_causal_translation_initialization(history(ids, delta, delta), ids, 3,
                [.001, .002, .001], 1., SimpleNamespace(static_translation_extent=.004), gs_xyz_step_extent=.001)


if __name__ == '__main__':
    unittest.main()
