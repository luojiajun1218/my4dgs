"""Independent CPU preparation tests; the active run does not import this helper."""

import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline' / 'ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
COMMON = importlib.import_module('pipeline.ReCon-GS.model.common_motion_parent')
MATH = importlib.import_module('pipeline.ReCon-GS.model.online_motion_tree')


def wxyz(rotation):
    return rotation.as_quat()[..., [3, 0, 1, 2]]


def fixture():
    source = np.random.default_rng(76).normal(scale=.1, size=(96, 3))
    source[:48, 0] -= .4
    source[48:, 0] += .4
    owner = np.repeat([1, 2], 48)
    parent_r = Rotation.from_rotvec([0., 0., .1])
    parent_b = np.array([.25, .12, .08])
    q, b = np.zeros((3, 4)), np.zeros((3, 3))
    q[0, 0] = 1
    for leaf, vector, delta in ((1, [.012, 0., 0.], [.02, 0., 0.]),
                                (2, [0., -.012, 0.], [-.02, 0., 0.])):
        relative = Rotation.from_rotvec(vector)
        pivot = source[owner == leaf].mean(0)
        inherited = parent_r.apply(pivot)
        q[leaf] = wxyz(relative * parent_r)
        b[leaf] = parent_b + inherited - relative.apply(inherited) + delta
    return source, owner, np.array([0, 11, 70]), q, b


def propose(data, **kwargs):
    source, owner, ids, q, b = data
    return COMMON.propose_common_motion_parents(source, owner, ids, q, b,
        1., .001, .001, kwargs.pop('config', COMMON.CommonMotionParentConfig()), **kwargs)


def virtual(source, owner, q, b):
    matrices = Rotation.from_quat(q[owner][:, [1, 2, 3, 0]]).as_matrix()
    return np.einsum('nij,nj->ni', matrices, source) + b[owner]


def independent_noise(source, owner):
    noise = np.random.default_rng(85).normal(scale=.02, size=source.shape)
    for leaf in np.unique(owner):
        rows = owner == leaf
        design = np.concatenate((np.ones((int(rows.sum()), 1)), source[rows]), axis=1)
        noise[rows] -= design @ np.linalg.lstsq(design, noise[rows], rcond=None)[0]
    return noise


class CommonMotionParentTests(unittest.TestCase):
    def test_legal_different_child_motion_has_common_part_without_raw_rigid_gate(self):
        data = fixture()
        result = propose(data, next_region_id=200, frame_index=1)
        self.assertTrue(result.metrics['inserted_parents'], result.metrics)
        self.assertFalse(result.metrics['virtual_rms_gate'])
        self.assertFalse(result.metrics['child_equal_rotation_gate'])
        self.assertGreater(result.metrics['inserted_parents'][0]['virtual_rms'], .004)
        self.assertGreaterEqual(result.metrics['inserted_parents'][0]['quantized_reduction'], .25)
        self.assertEqual(result.region_ids.tolist(), [0, 11, 70, 200])
        self.assertFalse(np.any(data[1] == 3))
        # Independent GS pose corrections are untouched by the representation.
        noise = independent_noise(data[0], data[1])
        q, b = MATH.local_to_world_motion(result.parents, result.pivots,
                                          result.local_translation, result.local_rotvec)
        np.testing.assert_allclose(virtual(data[0], data[1], q, b) + noise,
                                   virtual(data[0], data[1], data[3], data[4]) + noise, atol=1e-13)
        self.assertTrue(np.all(np.abs((q[:3] * data[3]).sum(1)) > 1 - 1e-12))
        self.assertLess(result.metrics['after']['integer_l1'], result.metrics['before']['integer_l1'])
        json.dumps(result.metrics, allow_nan=False)

    def test_historical_independent_gs_noise_does_not_become_parent_absolute_error(self):
        source, owner, ids, q, b = fixture()
        noise = independent_noise(source, owner)
        target = virtual(source, owner, q, b) + noise
        history = dict(source=source[:, None], target=target[:, None],
                       observed=np.ones((len(source), 1), bool), frames=[9])
        data = (target, owner, ids, q, b)  # Current reference is the previous decoded target.
        result = propose(data, history=history, frame_index=10)
        self.assertTrue(result.metrics['inserted_parents'], result.metrics)
        evidence = result.metrics['inserted_parents'][0]
        self.assertGreater(evidence['history_integer_l1_before'], 0)
        self.assertLessEqual(evidence['history_integer_l1_after'],
                             .75 * evidence['history_integer_l1_before'])
        self.assertEqual(evidence['history'][0]['evidence'], 'supported_nonzero_common_component')
        # Noise is large, but it is orthogonal to each leaf's established rigid fit.
        self.assertGreater(float(np.sqrt(np.square(noise).sum(1).mean())), .01)

    def test_opposite_motion_and_history_without_common_gain_are_rejected(self):
        points = np.random.default_rng(1).normal(size=(24, 3)) * .1
        source, owner = np.tile(points, (2, 1)), np.repeat([1, 2], 24)
        ids = np.array([0, 1, 2])
        q = np.tile([1., 0., 0., 0.], (3, 1))
        b = np.array([[0., 0., 0.], [.15, 0., 0.], [-.15, 0., 0.]])
        result = propose((source, owner, ids, q, b))
        self.assertEqual(result.metrics['inserted_parents'], [])
        self.assertGreater(result.metrics['rejection_counts']['current_quantized_gain'], 0)
        target = source + b[owner]
        history = dict(source=source[:, None], target=target[:, None],
                       observed=np.ones((len(source), 1), bool), frames=[9])
        shared = np.array([[0., 0., 0.], [.1, 0., 0.], [.1, 0., 0.]])
        rejected = propose((target, owner, ids, q, shared), history=history, frame_index=10)
        self.assertEqual(rejected.metrics['inserted_parents'], [])
        self.assertGreater(rejected.metrics['rejection_counts']['history_quantized_gain'], 0)

    def test_missing_birth_and_zero_history_do_not_manufacture_motion_evidence(self):
        source, owner, ids, q, b = fixture()
        hs = np.repeat(source[:, None], 3, axis=1)
        ht = hs.copy()
        observed = np.ones((len(source), 3), bool)
        observed[owner == 2, 1] = False  # No past observation, e.g. before birth.
        hs[owner == 2, 1] = np.nan
        ht[owner == 2, 1] = np.nan
        history = dict(source=hs, target=ht, observed=observed, frames=[7, 8, 9])
        result = propose((source, owner, ids, q, b), history=history, frame_index=10)
        inserted = result.metrics['inserted_parents'][0]
        self.assertEqual(inserted['history_integer_l1_before'], 0.)
        self.assertEqual(inserted['history_integer_l1_after'], 0.)
        self.assertEqual([row['evidence'] for row in inserted['history']],
                         ['observed_zero_world_motion', 'missing_supported_children', 'observed_zero_world_motion'])
        zero = propose((source, owner, ids, np.tile([1., 0., 0., 0.], (3, 1)), np.zeros((3, 3))))
        self.assertEqual(zero.metrics['inserted_parents'], [])
        self.assertGreater(zero.metrics['rejection_counts']['current_zero_world'], 0)
        json.dumps(result.metrics, allow_nan=False)

    def test_each_historical_pivot_uses_its_actual_observed_source(self):
        source, owner, ids, q, b = fixture()
        # A global coordinate shift requires a correspondingly shifted affine
        # offset to describe the same physical child motion. This exposes any
        # accidental use of current pivots for historical cost.
        shift = np.array([40., -30., 15.])
        historical_source = source + shift
        matrices = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        historical_b = b + shift - np.einsum('nij,j->ni', matrices, shift)
        historical_target = virtual(historical_source, owner, q, historical_b)
        history = dict(source=historical_source[:, None], target=historical_target[:, None],
                       observed=np.ones((len(source), 1), bool), frames=[9])
        translated = propose((source, owner, ids, q, b), history=history, frame_index=10)
        same_source = dict(source=source[:, None], target=virtual(source, owner, q, b)[:, None],
                           observed=np.ones((len(source), 1), bool), frames=[9])
        baseline = propose((source, owner, ids, q, b), history=same_source, frame_index=10)
        shifted_evidence, original_evidence = (result.metrics['inserted_parents'][0]['history'][0]
                                               for result in (translated, baseline))
        self.assertEqual(shifted_evidence['integer_l1_before'], original_evidence['integer_l1_before'])
        self.assertEqual(shifted_evidence['integer_l1_after'], original_evidence['integer_l1_after'])

    def test_bottom_up_refit_keeps_leaves_and_handles_parent_slots_after_children(self):
        source = np.random.default_rng(5).normal(size=(96, 3))
        owner = np.repeat([1, 2, 3], 32)
        parents = np.array([-1, 4, 4, 5, 5, 0, 5])
        _, pivots, _ = COMMON._rows_and_pivots(source, owner, parents, np.ones(96, bool))
        q = np.tile([1., 0., 0., 0.], (7, 1))
        q[1:4] = wxyz(Rotation.from_rotvec([.02, -.01, .03]))
        b = np.zeros((7, 3)); b[1:4] = [.1, -.03, .04]
        valid = np.array([True, True, True, True, False, False, False])
        result = COMMON.refit_parent_world_from_children(source, owner, parents, pivots,
                                                         q, b, world_valid=valid)
        np.testing.assert_allclose(result['world_offset'][1:4], b[1:4], atol=1e-14)
        np.testing.assert_allclose(result['world_offset'][4:6], np.repeat(b[1:2], 2, axis=0), atol=1e-13)
        self.assertTrue(result['valid_slot'][4:6].all())
        self.assertFalse(result['valid_slot'][6])
        self.assertEqual([row['slot'] for row in result['metrics']['internal_nodes']], [4, 5])
        t, r = MATH.world_to_local_motion(parents, pivots, result['world_quaternion'], result['world_offset'])
        rq, rb = MATH.local_to_world_motion(parents, pivots, t, r)
        np.testing.assert_allclose(rb[1:4], b[1:4], atol=1e-13)
        self.assertTrue(np.all(np.abs((rq[1:4] * q[1:4]).sum(1)) > 1 - 1e-12))

    def test_unsupported_internal_preserves_input_world_without_identity_evidence(self):
        source, owner, _, q_leaf, b_leaf = fixture()
        parents = np.array([-1, 3, 3, 0])
        _, pivots, _ = COMMON._rows_and_pivots(source, owner, parents, np.ones(len(source), bool))
        q = np.vstack((q_leaf, wxyz(Rotation.from_rotvec([.2, .1, 0.]))))
        b = np.vstack((b_leaf, [.3, -.2, .1]))
        observed = owner == 1
        result = COMMON.refit_parent_world_from_children(source, owner, parents, pivots, q, b,
                                                         observed=observed)
        np.testing.assert_array_equal(result['world_quaternion'][3], q[3])
        np.testing.assert_array_equal(result['world_offset'][3], b[3])
        self.assertFalse(result['valid_slot'][3])
        self.assertFalse(result['valid_slot'][2])
        self.assertTrue(result['valid_slot'][1])
        self.assertEqual(result['metrics']['unsupported_internal_nodes'][0]['slot'], 3)

    def test_future_history_is_rejected_and_current_duplicate_is_excluded(self):
        data = fixture()
        source = data[0]
        history = dict(source=source[:, None], target=source[:, None],
                       observed=np.ones((len(source), 1), bool), frames=[11])
        with self.assertRaises(ValueError):
            propose(data, history=history, frame_index=10)
        with self.assertRaises(ValueError):
            propose(data, history=history)
        history['frames'] = [10]
        result = propose(data, history=history, frame_index=10)
        self.assertEqual(result.metrics['history_frames'], [])

    def test_neighbor_candidates_and_cost_are_invariant_to_world_origin(self):
        count, per_leaf = 6, 24
        source = np.random.default_rng(102).normal(scale=.03, size=(count * per_leaf, 3))
        owner = np.repeat(np.arange(1, count + 1), per_leaf)
        ids = np.concatenate(([0], np.arange(1, count + 1) * 11))
        q = np.tile([1., 0., 0., 0.], (count + 1, 1))
        b = np.zeros((count + 1, 3))
        for leaf in range(1, count + 1):
            source[owner == leaf, 0] += leaf * .4
            q[leaf] = wxyz(Rotation.from_rotvec([.004 * leaf, -.003 * leaf, .08]))
            b[leaf] = [.2 + .007 * leaf, .1, .05]
        config = COMMON.CommonMotionParentConfig(neighbors=1, max_children=3, max_new_parents=2)
        baseline = propose((source, owner, ids, q, b), config=config)
        self.assertTrue(baseline.metrics['inserted_parents'])
        shift = np.array([40., -30., 15.])
        matrices = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        shifted_b = b + shift - np.einsum('nij,j->ni', matrices, shift)
        translated = propose((source + shift, owner, ids, q, shifted_b), config=config)
        np.testing.assert_array_equal(translated.parents, baseline.parents)
        np.testing.assert_array_equal(translated.region_ids, baseline.region_ids)
        np.testing.assert_allclose(translated.local_translation, baseline.local_translation, atol=1e-12)
        np.testing.assert_allclose(translated.local_rotvec, baseline.local_rotvec, atol=1e-12)
        self.assertEqual(translated.metrics['before'], baseline.metrics['before'])
        self.assertEqual(translated.metrics['after'], baseline.metrics['after'])
        self.assertEqual([row['child_region_ids'] for row in translated.metrics['inserted_parents']],
                         [row['child_region_ids'] for row in baseline.metrics['inserted_parents']])
        unsupported = propose((source, owner, ids, q, b), observed=np.zeros(len(source), bool))
        self.assertEqual(unsupported.metrics['inserted_parents'], [])


if __name__ == '__main__':
    unittest.main()
