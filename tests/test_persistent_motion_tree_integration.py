"""Real persistent-tree fitting, quantization, identity and codec checks on CPU."""

import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline' / 'ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package

RIGID = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
TREE = importlib.import_module('pipeline.ReCon-GS.model.online_motion_tree')
INTEGRATION = importlib.import_module('pipeline.ReCon-GS.model.persistent_motion_tree_integration')
PERSISTENT = importlib.import_module('pipeline.ReCon-GS.model.persistent_motion_regions')
CLUSTERING = importlib.import_module('pipeline.ReCon-GS.model.online_motion_clustering')
CODEC = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')


class PersistentTreeIntegrationTests(unittest.TestCase):
    def fixture(self, tree=False):
        config = RIGID.RigidTreeConfig(online_flat_clustering=True, adaptive_clustering=True,
            region_count=1, motion_steps=100, post_partition_steps=100, birth_steps=100,
            shared_steps=0, tree_edit_step=100, rate_loss_weight=0.,
            relative_motion_loss_weight=0., gs_xyz_step_extent=.0005, gs_rotation_step_rad=.0005,
            motion_xyz_step_extent=.0001, motion_rotation_step_rad=.0001,
            persistent_regions=PERSISTENT.PersistentRegionConfig(enabled=True, tree_enabled=True),
            online_clustering=CLUSTERING.OnlineMotionClusteringConfig(method='kmeans'),
            motion_tree=TREE.MotionTreeConfig())
        model = RIGID.RigidResidualTree(config)
        source = torch.from_numpy(np.random.default_rng(19).normal(scale=.2, size=(100, 3))).float()
        base_q = torch.tensor([[1., 0., 0., 0.]]).repeat(len(source), 1)
        model.setup(source, base_q, 1., frame_index=1)
        model.owner = torch.cat((torch.arange(1, 5).repeat_interleave(24), torch.zeros(4, dtype=torch.long)))
        model.parents = torch.tensor([-1, 0, 0, 0, 0] + ([0] if tree else []))
        if tree:
            model.parents[1:5] = 5  # Internal slot follows all leaf slots.
        model.pivots = RIGID.region_pivots(source, model.owner, model.parents)
        model.persistent_state.region_ids = torch.tensor([0, 11, 33, 55, 77] + ([88] if tree else []))
        model.persistent_state.reference_region_ids = torch.tensor([0])
        model.persistent_state.next_region_id = 101
        model.persistent_state.initialized = True
        model.region_map = torch.tensor([0])
        model.observations.fill_(10)
        shared_q = RIGID.rotation_vector_quaternion(torch.tensor([.001, -.002, .001]))
        target = RIGID.quaternion_rotate(shared_q.expand(96, -1), source[:96]) + torch.tensor([.013, -.007, .009])
        target = torch.cat((target, source[96:]))
        target_q = torch.cat((shared_q.expand(96, -1), base_q[96:]))
        return model, target, target_q

    def refit(self, model, target, target_q):
        self.assertTrue(model._refit_persistent_tree(target, target_q))
        model._make_optimizer()
        model._region_id_lengths = RIGID._id_lengths(len(model.parents) - 1, model.base_xyz)
        self.assertTrue(model._online_packet_score(target, target_q)['valid'])

    def test_existing_internal_parent_refit_preserves_target_and_actual_gradients(self):
        model, target, target_q = self.fixture(tree=True)
        self.refit(model, target, target_q)
        self.assertFalse(bool((model.owner == 5).any()))
        self.assertTrue(bool(model._values(hard=True)[0][5].ne(0).any()))
        world_q, world_b = model._global_regions()
        self.assertGreater(float(world_b[5].norm()), .01)
        xyz, q = model.forward(101, training=True)
        self.assertLess(float((xyz - target).norm(dim=-1).max()), .00044)
        parameters = (model.region_translation, model.region_rotation,
                      model.gs_translation, model.gs_rotation)
        gradients = torch.autograd.grad(xyz.square().mean() + q[:, 1:].mean(), parameters)
        self.assertTrue(all(bool(gradient.abs().sum() > 0) for gradient in gradients))
        self.assertGreater(float(gradients[0][5].abs().sum()), 0.)
        self.assertGreater(float(gradients[0][1:5].abs().sum()), 0.)
        self.assertTrue(all(group['params'][0] is parameter
                            for group, parameter in zip(model.optimizer.param_groups, parameters)))

    def test_parent_admission_appends_slots_preserves_logical_ids_and_redirect_mapping(self):
        model, target, target_q = self.fixture()
        self.refit(model, target, target_q)
        # A removed historical logical ID redirects through another removed ID.
        model.persistent_state.reference_region_ids = torch.tensor([0, 11, 123])
        old_owner = torch.where(model.owner == 1, 1, torch.where(model.owner > 0, 2, 0))
        model.frame_reference_state = CODEC.DecodedState(model.base_xyz.cpu(), model.base_rotation.cpu(),
                                                        old_owner.cpu(), torch.tensor([-1, 0, 0]), 0)
        model.persistent_state.next_region_id = 200
        model.region_map = torch.tensor([0, 1, 2])
        model.persistent_state.redirects = {123: 99, 99: 33}
        before_ids, before_owner = model.persistent_state.region_ids.clone(), model.owner.clone()
        report = model._propose_persistent_tree(target, target_q, history=None)
        self.assertTrue(report['accepted'], report)
        torch.testing.assert_close(model.persistent_state.region_ids[:len(before_ids)], before_ids)
        torch.testing.assert_close(model.owner, before_owner)
        self.assertEqual(model.persistent_state.region_ids[-1], 200)
        self.assertEqual(model.persistent_state.next_region_id, 201)
        self.assertEqual(model.region_map.tolist(), [0, 1, 2])
        TREE.validate_motion_tree(model.parents.numpy(), model.owner.numpy())
        self.assertTrue(report['after_packet']['valid'])
        self.assertFalse(report['whole_packet_byte_veto'])
        self.assertGreaterEqual(report['helper']['inserted_parents'][0]['quantized_reduction'], .25)
        # Dense compaction sorts logical IDs; the larger internal ID still has
        # a slot AFTER its children, and world fitting must respect that tree.
        target_before = tuple(value.detach().clone() for value in model.forward(200))
        owner_ids = model._persistent_owner_ids()
        self.assertTrue(model._fit_persistent_partition(owner_ids, *target_before))
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 11, 33, 55, 77, 200])
        self.assertTrue(bool(model.parents[1:5].eq(5).all()))
        self.assertTrue(model._online_packet_score(*target_before)['valid'])

    def test_invalid_packet_rolls_back_tree_ids_parameters_and_counter(self):
        model, target, target_q = self.fixture()
        self.refit(model, target, target_q)
        original = model._persistent_layout_snapshot()
        score = model._online_packet_score(target, target_q)
        with patch.object(model, '_online_packet_score', side_effect=[score, {**score, 'valid': False}]):
            report = model._propose_persistent_tree(target, target_q, history=None)
        self.assertFalse(report['accepted'])
        self.assertEqual(report['rejection_reason'], 'tree_packet_pose_invalid')
        torch.testing.assert_close(model.persistent_state.region_ids, original['state']['region_ids'])
        self.assertEqual(model.persistent_state.next_region_id, original['state']['next_region_id'])
        for name, value in original['partition'].items():
            torch.testing.assert_close(getattr(model, name), value)
        self.assertTrue(model._online_packet_score(target, target_q)['valid'])

    def test_unobserved_refit_uses_identity_sharing_and_independent_residuals(self):
        model, target, target_q = self.fixture(tree=True)
        model.observations.zero_()
        self.refit(model, target, target_q)
        rt, rr, gt, gr = model._values(hard=True)
        self.assertFalse(bool(rt.ne(0).any()))
        self.assertFalse(bool(rr.ne(0).any()))
        self.assertTrue(bool(gt.ne(0).any()))
        self.assertTrue(bool(gr.ne(0).any()))


if __name__ == '__main__':
    unittest.main()
