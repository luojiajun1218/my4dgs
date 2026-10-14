"""Renderer-free causal motion-tree math, admission and decoder checks."""

import importlib
import sys
from pathlib import Path
from types import ModuleType
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline' / 'ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
TREE = importlib.import_module('pipeline.ReCon-GS.model.online_motion_tree')
RIGID = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
CODEC = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')


def wxyz(rotations):
    return rotations.as_quat()[..., [3, 0, 1, 2]]


def fixture(leaves=4, count=24):
    source = np.random.default_rng(34).normal(size=(leaves * count, 3))
    owner = np.repeat(np.arange(1, leaves + 1), count)
    ids = np.concatenate(([0], 13 + 17 * np.arange(leaves)))
    q = np.tile(wxyz(Rotation.from_rotvec([.001, -.002, .001])), (leaves + 1, 1))
    q[0] = [1, 0, 0, 0]
    b = np.tile([.013, -.007, .009], (leaves + 1, 1))
    b[0] = 0
    matrix = Rotation.from_quat(q[owner][:, [1, 2, 3, 0]]).as_matrix()
    target = np.einsum('nij,nj->ni', matrix, source) + b[owner]
    return source, target, owner, ids, q, b


def propose(data, **kwargs):
    source, target, owner, ids, q, b = data
    return TREE.propose_motion_tree(source, target, owner, ids, q, b, 1., .0001, .0001,
                                    kwargs.pop('config', TREE.MotionTreeConfig()), **kwargs)


class OnlineMotionTreeTests(unittest.TestCase):
    def test_noncommuting_rotations_roundtrip_and_repository_composition(self):
        parents = np.array([-1, 0, 1, 1, 0])
        pivot = np.random.default_rng(8).normal(size=(5, 3))
        rotvec = np.array([[0., 0., 0.], [.4, 0., 0.], [0., .7, 0.],
                          [.2, -.3, .4], [0., 0., -.5]])
        translation = np.random.default_rng(9).normal(size=(5, 3)) * .1
        translation[0] = 0
        q, b = TREE.local_to_world_motion(parents, pivot, translation, rotvec)
        restored_t, restored_r = TREE.world_to_local_motion(parents, pivot, q, b)
        np.testing.assert_allclose(restored_t, translation, atol=1e-14)
        np.testing.assert_allclose(restored_r, rotvec, atol=1e-14)
        actual_q, actual_b = RIGID.region_transforms(torch.zeros(1, 3, dtype=torch.float64),
            torch.tensor(parents), torch.tensor(pivot), torch.tensor(translation), torch.tensor(rotvec))
        np.testing.assert_allclose(actual_b.numpy(), b, atol=1e-13)
        self.assertTrue(np.all(np.abs(np.sum(actual_q.numpy() * q, axis=1)) > 1 - 1e-12))

    def test_parent_insertion_preserves_world_leaves_and_stable_ids(self):
        data = fixture()
        result = propose(data, next_region_id=100, frame_index=1)
        self.assertEqual(len(result.metrics['inserted_parents']), 1)
        self.assertEqual(result.region_ids[-1], 100)
        self.assertEqual(result.next_region_id, 101)
        np.testing.assert_array_equal(result.region_ids[:-1], data[3])
        self.assertTrue(np.all(result.parents[1:5] == 5))
        TREE.validate_motion_tree(result.parents, data[2], max_depth=3)
        reconstructed_q, reconstructed_b = TREE.local_to_world_motion(result.parents,
            result.pivots, result.local_translation, result.local_rotvec)
        np.testing.assert_allclose(reconstructed_b[:5], data[5], atol=1e-13)
        self.assertTrue(np.all(np.abs((reconstructed_q[:5] * data[4]).sum(1)) > 1 - 1e-12))
        for parent in result.metrics['inserted_parents']:
            self.assertGreaterEqual(parent['quantized_reduction'], .25)
        self.assertLess(result.metrics['after']['integer_l1'], result.metrics['before']['integer_l1'])
        # A valid existing tree is not rebuilt merely to allocate another ID.
        again = TREE.propose_motion_tree(*data[:3], result.region_ids,
            result.world_quaternion, result.world_offset, 1., .0001, .0001,
            TREE.MotionTreeConfig(), parents=result.parents, pivots=result.pivots,
            next_region_id=result.next_region_id, frame_index=10)
        np.testing.assert_array_equal(again.parents, result.parents)
        self.assertEqual(again.metrics['inserted_parents'], [])

    def test_reparenting_preserves_noncommuting_world_transforms(self):
        old_parents = np.array([-1, 0, 0, 1, 0, 0])
        pivot = np.random.default_rng(11).normal(size=(6, 3))
        rotvec = np.random.default_rng(12).normal(size=(6, 3)) * .3
        translation = np.random.default_rng(13).normal(size=(6, 3)) * .1
        rotvec[0] = translation[0] = 0
        world_q, world_b = TREE.local_to_world_motion(old_parents, pivot, translation, rotvec)
        new_parents = np.array([-1, 5, 5, 1, 0, 4])
        local_t, local_r = TREE.world_to_local_motion(new_parents, pivot, world_q, world_b)
        q, b = TREE.local_to_world_motion(new_parents, pivot, local_t, local_r)
        np.testing.assert_allclose(b, world_b, atol=1e-13)
        self.assertTrue(np.all(np.abs((q * world_q).sum(1)) > 1 - 1e-12))

    def test_invalid_tree_cycle_depth_and_internal_ownership_fail(self):
        for parents, owner in (([-1, 2, 1], None), ([-1, 0, 1, 2, 3], None),
                               ([-1, 0, 1], [1, 2])):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                TREE.validate_motion_tree(np.array(parents), None if owner is None else np.array(owner))

    def test_depth_limit_includes_empty_descendants_and_keeps_existing_tree(self):
        source, target, owner, _, q, b = fixture(leaves=2)
        ids = np.array([0, 13, 30, 40, 50, 60])
        parents = np.array([-1, 3, 3, 4, 0, 1])
        # This existing empty descendant would exceed depth if node1 moved.
        parents[5] = 3
        world_q = np.vstack((q, q[1:], q[1]))[:6]
        world_b = np.vstack((b, b[1:], b[1]))[:6]
        result = TREE.propose_motion_tree(source, target, owner, ids, world_q, world_b,
            1., .0001, .0001, TREE.MotionTreeConfig(), parents=parents)
        np.testing.assert_array_equal(result.parents, parents)
        self.assertEqual(result.metrics['inserted_parents'], [])

    def test_no_common_evidence_and_no_quantized_gain_keep_flat(self):
        data = fixture(leaves=2)
        missing = list(data)
        missing[1] = np.full_like(data[1], np.nan)
        result = propose(missing, observed=np.zeros(len(data[0]), bool))
        self.assertEqual(result.metrics['inserted_parents'], [])
        self.assertEqual(len(result.parents), 3)
        source = np.tile(np.random.default_rng(4).normal(size=(24, 3)), (2, 1))
        owner = np.repeat([1, 2], 24)
        q = np.tile([1., 0., 0., 0.], (3, 1))
        b = np.array([[0., 0., 0.], [.01, 0., 0.], [-.01, 0., 0.]])
        result = propose((source, source + b[owner], owner, np.array([0, 1, 2]), q, b),
                         config=TREE.MotionTreeConfig(fit_tolerance_extent=.1))
        self.assertEqual(result.metrics['inserted_parents'], [])

    def test_history_masks_are_not_zero_evidence_and_future_is_rejected(self):
        data = fixture(leaves=2)
        source, _, owner = data[:3]
        old_target = source + np.where((owner == 1)[:, None], [.02, 0, 0], [-.02, 0, 0])
        historical = dict(source=source[:, None], target=old_target[:, None],
                          observed=np.ones((len(source), 1), bool), frames=[1])
        self.assertEqual(propose(data, history=historical, frame_index=10).metrics['inserted_parents'], [])
        historical['observed'][owner == 2] = False
        historical['target'][owner == 2] = np.nan
        self.assertTrue(propose(data, history=historical, frame_index=10).metrics['inserted_parents'])
        historical['frames'] = [11]
        with self.assertRaises(ValueError):
            propose(data, history=historical, frame_index=10)
        historical['frames'] = [10]  # Explicit current target wins over duplicate current slot.
        result = propose(data, history=historical, frame_index=10)
        self.assertEqual(result.metrics['history_transitions_checked'], 0)

    def test_update_interval_and_sparse_byte_metric_match_codec(self):
        self.assertTrue(propose(fixture(), frame_index=2).metrics['update_skipped'])
        self.assertFalse(propose(fixture(), frame_index=10).metrics['update_skipped'])
        t, r = np.zeros((301, 3)), np.zeros((301, 3))
        t[3] = [2, -4, 0]
        r[300] = [0, 3, 0]
        metric = TREE.quantized_regional_cost(t, r, 1., 1.)
        payload = bytearray()
        CODEC._sparse_write(payload, torch.tensor(t[1:], dtype=torch.int16),
                            torch.tensor(r[1:], dtype=torch.int16))
        self.assertEqual(metric['sparse_bytes'], len(payload))
        self.assertEqual(metric['integer_l1'], 9.)

    def test_checkpoint_decoder_with_inserted_parent_and_quantized_gs_retarget(self):
        data = fixture()
        source, target, owner, ids, _, _ = data
        proposal = propose(data)
        prior_q = torch.tensor([[1., 0., 0., 0.]]).repeat(len(source), 1)
        prior = CODEC.DecodedState(torch.tensor(source, dtype=torch.float32), prior_q,
                                  torch.tensor(owner), torch.tensor([-1] + [0] * 4), 0)
        rt = torch.tensor(np.rint(proposal.local_translation / .0001), dtype=torch.int16)
        rr = torch.tensor(np.rint(proposal.local_rotvec / .0001), dtype=torch.int16)
        parents, pivots = torch.tensor(proposal.parents), torch.tensor(proposal.pivots, dtype=torch.float32)
        q, b = RIGID.region_transforms(prior.xyz, parents, pivots, rt.float() * .0001, rr.float() * .0001)
        current_owner = torch.tensor(owner)
        predicted = RIGID.quaternion_rotate(q[current_owner], prior.xyz) + b[current_owner]
        gt = ((torch.tensor(target, dtype=torch.float32) - predicted) / .0001).round().to(torch.int16)
        target_q = torch.tensor(proposal.world_quaternion[owner], dtype=torch.float32)
        relative = RIGID.quaternion_multiply(target_q, RIGID.quaternion_inverse(q[current_owner]))
        gr = (RIGID.quaternion_rotvec(relative) / .0001).round().to(torch.int16)
        packet = CODEC.encode_motion(prior, 1, 1., torch.arange(len(source)), current_owner,
            parents, rt, rr, gt, gr, torch.empty(0, dtype=torch.long), torch.empty(0, 3),
            torch.empty(0, 4), .0001, .0001, gs_translation_step_extent=.0001,
            gs_rotation_step_rad=.0001, region_map=torch.arange(len(ids)), pivots=pivots,
            compression='auto')
        restored_prior = CODEC.DecodedState.restore(prior.capture())
        decoded = CODEC.decode_motion(packet, restored_prior)
        expected_xyz, expected_q = RIGID.compose_pose(prior.xyz, prior.rotation, current_owner,
            parents, pivots, rt.float() * .0001, rr.float() * .0001,
            gt.float() * .0001, gr.float() * .0001)
        torch.testing.assert_close(decoded.xyz, expected_xyz)
        torch.testing.assert_close(decoded.rotation, expected_q)
        self.assertLess(float((decoded.xyz - torch.tensor(target)).norm(dim=-1).max()), .00009)
        self.assertEqual(decoded.digest(), CODEC.DecodedState.restore(decoded.capture()).digest())
        self.assertFalse(bool((decoded.owner == len(ids)).any()))


if __name__ == '__main__':
    unittest.main()
