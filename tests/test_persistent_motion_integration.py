"""Renderer-free persistent region lifecycle, tree, codec and checkpoint checks."""

import importlib
import json
import copy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

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

DIAGNOSTICS = importlib.import_module('pipeline.ReCon-GS.model.persistent_motion_diagnostics')


class PersistentDiagnosticsTests(unittest.TestCase):
    def fixture(self):
        xyz = np.zeros((4, 3))
        q = np.tile([1., 0., 0., 0.], (4, 1))
        rq = np.tile([1., 0., 0., 0.], (3, 1))
        # Root owns one survivor, slot1 is internal, slot2 owns two survivors
        # and one newborn. Logical identities need not equal compact slots.
        return xyz, q, rq

    def test_internal_direct_subtree_counts_exact_symbols_and_birth_exclusion(self):
        xyz, q, rq = self.fixture()
        target = xyz.copy()
        target[1, 0] = .001
        target_q = q.copy()
        target_q[2] = Rotation.from_rotvec([0., .001, 0.]).as_quat()[[3, 0, 1, 2]]
        world_t = np.zeros((3, 3))
        world_t[1, 0] = .001
        observed = np.ones(4, dtype=bool)
        report = DIAGNOSTICS.summarize_persistent_regions(
            xyz, target, q, target_q, [0, 2, 2, 2], [-1, 0, 1], rq, world_t,
            region_ids=[0, 10, 42], surviving_count=3, observed_mask=observed,
            gs_translation_i16=np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], np.int16),
            gs_rotation_i16=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], np.int16),
            region_translation_i16=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], np.int16),
            region_rotation_i16=np.zeros((3, 3), np.int16),
            edit_metrics=dict(changed=np.bool_(True), migrations=np.int64(2)),
            packet_metrics=dict(bytes=np.int64(123), raw_bytes=200))
        self.assertTrue(observed.all(), 'Diagnostic birth masking must not mutate input masks')
        root, internal, leaf = report['regions']
        self.assertEqual(root['subtree_gs_count'], 4)
        self.assertEqual(internal['gs_count'], 0)
        self.assertEqual(internal['subtree_gs_count'], 3)
        self.assertTrue(internal['is_internal'])
        self.assertEqual(leaf['logical_region_id'], 42)
        self.assertEqual(leaf['parent_region_slot'], 1)
        self.assertEqual(leaf['parent_id'], 10)
        self.assertEqual(leaf['depth'], 2)
        self.assertEqual(leaf['surviving_gs_count'], 2)
        self.assertEqual(leaf['newborn_gs_count'], 1)
        self.assertEqual(leaf['gs_proportion_all'], .75)
        self.assertEqual(leaf['independent_translation_residual']['count'], 2)
        self.assertEqual(leaf['independent_quantized']['translation_nonzero'], 1)
        self.assertEqual(leaf['independent_quantized']['rotation_nonzero'], 1)
        self.assertEqual(leaf['independent_quantized']['residual_nonzero'], 2)
        self.assertTrue(leaf['encoded_local_shared_nonzero'])
        self.assertTrue(leaf['encoded_shared_path_nonzero'])
        self.assertEqual(leaf['shared_translation_norm'], 0.)
        self.assertAlmostEqual(leaf['independent_translation_residual']['mean'], .0005)
        self.assertAlmostEqual(leaf['independent_rotation_residual_rad']['mean'], .0005)
        self.assertEqual(report['max_depth'], 2)
        self.assertEqual(report['internal_region_count'], 1)
        json.dumps(report, allow_nan=False)

    def test_missing_symbols_remain_unavailable_and_unobserved_stats_are_null(self):
        xyz, q, rq = self.fixture()
        report = DIAGNOSTICS.summarize_persistent_regions(
            xyz, xyz, q, q, [0, 2, 2, 2], [-1, 0, 1], rq, np.zeros((3, 3)),
            surviving_count=3, observed_mask=[False] * 4)
        self.assertEqual(report['observed_gs'], 0)
        self.assertIsNone(report['regions'][2]['independent_translation_residual']['mean'])
        self.assertFalse(report['regions'][2]['independent_quantized']['available'])
        self.assertIsNone(report['regions'][2]['independent_quantized']['residual_zero'])
        json.dumps(report, allow_nan=False)

    def test_cycles_internal_gs_ownership_and_incomplete_symbols_are_rejected(self):
        xyz, q, rq = self.fixture()
        for owners, parents in (([0, 2, 2, 2], [-1, 2, 1]), ([0, 1, 2, 2], [-1, 0, 1])):
            with self.subTest(parents=parents):
                with self.assertRaises(ValueError):
                    DIAGNOSTICS.summarize_persistent_regions(
                        xyz, xyz, q, q, owners, parents, rq, np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            DIAGNOSTICS.summarize_persistent_regions(
                xyz, xyz, q, q, [0, 2, 2, 2], [-1, 0, 1], rq, np.zeros((3, 3)),
                surviving_count=3, gs_translation_i16=np.zeros((3, 3), np.int16))


class PersistentLifecycleTests(unittest.TestCase):
    @staticmethod
    def modules():
        return tuple(importlib.import_module('pipeline.ReCon-GS.model.' + name) for name in (
            'rigid_residual_tree', 'persistent_motion_regions', 'online_flat_motion',
            'online_motion_clustering', 'rigid_tree_codec'))

    def fixture(self, count=32, **persistent_options):
        rigid, persistent, _, clustering, _ = self.modules()
        options = dict(enabled=True, min_region_size=8)
        options.update(persistent_options)
        config = rigid.RigidTreeConfig(
            online_flat_clustering=True, adaptive_clustering=True, region_count=1,
            motion_steps=100, post_partition_steps=100, birth_steps=100,
            tree_edit_step=100, shared_steps=0, rate_loss_weight=0.,
            relative_motion_loss_weight=0., gs_xyz_step_extent=.001,
            gs_rotation_step_rad=.001, motion_xyz_step_extent=.001,
            motion_rotation_step_rad=.001, gs_translation_lr=.001,
            gs_rotation_lr=.001, region_translation_lr=.001, region_rotation_lr=.001,
            persistent_regions=persistent.PersistentRegionConfig(**options),
            online_clustering=clustering.OnlineMotionClusteringConfig(
                method='kmeans', min_region_size=8, component_counts=[2]))
        model = rigid.RigidResidualTree(config)
        xyz = torch.from_numpy(np.random.default_rng(3).normal(scale=.1, size=(count, 3))).float()
        xyz[:16, 0] -= 2.
        xyz[16:24, 0] += 2.
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1)
        model.setup(xyz, q, 1., frame_index=1)
        model.observations.fill_(10)
        return model

    def initialize(self, model):
        with torch.no_grad():
            model.gs_translation[:16, 0] = .02
            model.gs_translation[16:24, 1] = .03
        model.last_step = 100
        model.update_topology(100)
        self.assertTrue(model.persistent_state.initialized)
        self.assertEqual(len(model.persistent_state.region_ids), 3)
        self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2])
        self.assertEqual(model.owner[:16].unique().numel(), 1)
        self.assertEqual(model.owner[16:24].unique().numel(), 1)
        self.assertEqual(model.owner[24:].tolist(), [0] * (len(model.owner) - 24))

    def finish_export(self, model, path, birth_source=None):
        _, _, _, _, codec = self.modules()
        model.forward(200, training=True)
        model.finalize_online_refinement()
        xyz, q = (value.detach().clone() for value in model.materialized_pose())
        if birth_source is not None:
            xyz = torch.cat((xyz, xyz[birth_source:birth_source + 1] + .001))
            q = torch.cat((q, q[birth_source:birth_source + 1]))
        model.register_materialized_layout(xyz, q, None if birth_source is None else [birth_source])
        prior = model.frame_reference_state
        result = model.export_motion(path)
        decoded = codec.decode_motion(path.read_bytes(), prior)
        self.assertEqual(result['bytes'], path.stat().st_size)
        self.assertEqual(decoded.digest(), model.codec_previous_state.digest())
        self.assertTrue(torch.equal(decoded.owner, model.materialized_owner))
        json.dumps(model.flat_region_diagnostics, allow_nan=False)
        return decoded

    def test_real_full_iterations_persistent_second_frame_births_and_checkpoint(self):
        rigid, _, flat, _, _ = self.modules()
        module_source = importlib.import_module('pipeline.ReCon-GS.module')
        model = self.fixture()
        frame0 = model.frame_reference_state
        original_digest = frame0.digest()
        desired = model.base_xyz.clone()
        desired[:16, 0] += .02
        desired[16:24, 1] += .03
        qtarget = model.base_rotation.clone()
        with tempfile.TemporaryDirectory() as directory:
            birth_calls = []
            harness = SimpleNamespace(motion=model, motion_committed=False,
                trainer=SimpleNamespace(global_step=0, bootstrap_pending=False),
                config=SimpleNamespace(rigid_tree=model.config,
                    birth=SimpleNamespace(densification_stats_steps=30)),
                output_dir=Path(directory), current_frame=1, motion_steps=200, total_steps=300,
                birth_control={}, gaussians=SimpleNamespace(
                    xyz_gradient_accum=torch.zeros(32, 1), denom=torch.zeros(32, 1),
                    add_densification_stats=Mock()))

            def begin_births():
                birth_calls.append(harness.trainer.global_step)
                self.assertTrue(model.flat_region_diagnostics['refinement']['finalized'])
                harness.motion_committed = True
            harness._begin_births = begin_births
            package = dict(visibility_filter=torch.ones(32, dtype=torch.bool),
                           viewspace_points=torch.zeros(32, 3))
            resume = None
            for step in range(1, 201):
                xyz, q = model.forward(step, training=True)
                loss = ((xyz - desired).square().mean() + (q - qtarget).square().mean())
                loss.backward()
                if resume is not None:
                    rx, rq = resume.forward(step, training=True)
                    ((rx - desired).square().mean() + (rq - qtarget).square().mean()).backward()
                    resume.optimizer_step()
                harness.trainer.global_step = step
                module_source.Module.post_training_step(harness, package)
                if step in (99, 100, 150):
                    restored = rigid.RigidResidualTree(model.config)
                    restored.restore(copy.deepcopy(model.capture()))
                    self.assertEqual(restored.persistent_state.region_ids.tolist(),
                                     model.persistent_state.region_ids.tolist())
                    self.assertEqual(restored.stage(100), model.stage(100))
                    if step == 150:
                        resume = restored
            self.assertEqual(birth_calls, [200])
            self.assertEqual(len(model.topology_edits), 1)
            self.assertEqual(model.persistent_state.last_evidence_frame, 1)
            for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
                torch.testing.assert_close(getattr(model, name), getattr(resume, name), rtol=0, atol=0)
            owner_before_birth = model._persistent_owner_ids().clone()
            decoded = self.finish_export(model, Path(directory) / 'motion_001.bin', birth_source=0)
            self.assertEqual(model.materialized_online_ids.tolist(), list(range(33)))
            self.assertEqual(len(model.online_history.records[0]['ids']), 32)
            self.assertEqual(decoded.owner[-1], decoded.owner[0])
            saved = rigid.RigidResidualTree(model.config)
            saved.restore(copy.deepcopy(model.capture()))
            self.assertEqual(saved.codec_previous_state.digest(), decoded.digest())
            self.assertEqual(saved.persistent_state.capture()['region_evidence'],
                             model.persistent_state.capture()['region_evidence'])
            # Delete one older row: stable GS IDs must align history/pending
            # across shifted dense rows and a newborn with no past evidence.
            survivors = torch.tensor([i for i in range(33) if i != 3])
            model.setup(decoded.xyz[survivors], decoded.rotation[survivors], 1.,
                        frame_index=2, previous_rows=survivors)
            expected_ids = [i for i in range(33) if i != 3]
            self.assertEqual(model.online_ids.tolist(), expected_ids)
            self.assertEqual(model.persistent_state.reference_region_ids.tolist(), [0, 1, 2])
            inherited = torch.cat((owner_before_birth, owner_before_birth[:1]))[survivors]
            self.assertTrue(torch.equal(model._persistent_owner_ids(), inherited))
            aligned = model.online_history.aligned(model.online_ids.numpy())
            self.assertFalse(aligned['observed'][-1, 0])
            model.observations.fill_(10)
            model.observations[-1] = 0
            with torch.no_grad():
                model.gs_translation[inherited == int(inherited[0]), 0] = .01
                other = int(owner_before_birth[16])
                model.gs_translation[inherited == other, 1] = .02
            target = tuple(value.detach().clone() for value in model.forward(100))
            with patch.object(flat, 'propose_motion_regions',
                              side_effect=AssertionError('Persistent frame2 must not globally recluster')):
                model.update_topology(100)
            self.assertTrue(torch.equal(model._persistent_owner_ids(), inherited))
            self.assertEqual(model.flat_region_diagnostics['persistent']['migrated_gs'], 0)
            self.assertEqual(model.persistent_state.migration_pending, {})
            decoded2 = self.finish_export(model, Path(directory) / 'motion_002.bin')
            self.assertEqual(len(model.online_history.records), 2)
            self.assertEqual(model.persistent_state.last_evidence_frame, 2)
            self.assertEqual(frame0.digest(), original_digest)
            self.assertLessEqual(float((decoded2.xyz - target[0]).norm(dim=-1).max()), .000867)

    def test_migration_two_observed_frames_missing_mask_and_stable_gs_identity(self):
        rigid, _, flat, _, _ = self.modules()
        model = self.fixture()
        self.initialize(model)
        destination = int(model._persistent_owner_ids()[0])
        other = int(model._persistent_owner_ids()[16])
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.finish_export(model, Path(directory) / 'motion_001.bin')
            for frame in (2, 3, 4):
                model.setup(decoded.xyz, decoded.rotation, 1., frame_index=frame)
                model.observations.fill_(10)
                if frame == 3:
                    model.observations[24] = 0
                logical = model._persistent_owner_ids().clone()
                with torch.no_grad():
                    model.gs_translation[logical == destination, 0] = .02
                    model.gs_translation[logical == other, 1] = .03
                    model.gs_translation[24, 0] = .02
                target = tuple(value.detach().clone() for value in model.forward(100))
                stable_gs = int(model.online_ids[24])
                with patch.object(flat, 'propose_motion_regions',
                                  side_effect=AssertionError('Migration must not recluster')):
                    model.update_topology(100)
                self.assertEqual(model.flat_region_diagnostics['persistent']['migrated_gs'],
                                 1 if frame == 4 else 0)
                if frame < 4:
                    self.assertEqual(model.persistent_state.migration_pending[stable_gs],
                                     dict(target=destination, streak=1))
                    self.assertEqual(int(model._persistent_owner_ids()[24]), 0)
                else:
                    self.assertEqual(int(model._persistent_owner_ids()[24]), destination)
                    self.assertNotIn(stable_gs, model.persistent_state.migration_pending)
                saved = rigid.RigidResidualTree(model.config)
                saved.restore(copy.deepcopy(model.capture()))
                self.assertEqual(saved.persistent_state.migration_pending,
                                 model.persistent_state.migration_pending)
                self.assertTrue(model._online_packet_score(*target)['valid'])
                self.assertTrue(torch.equal(model.online_ids, torch.arange(32)))
                decoded = self.finish_export(model, Path(directory) / f'motion_{frame:03d}.bin')

    def test_compaction_redirect_map_pose_codec_and_logical_ids_are_not_reused(self):
        rigid, _, _, _, _ = self.modules()
        model = self.fixture()
        self.initialize(model)
        # A region slot disappears after frame1 while a surviving logical leaf
        # is remapped from dense slot2 to slot1.
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.finish_export(model, Path(directory) / 'motion_001.bin')
            model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
            model.observations.fill_(10)
            owner_ids = model._persistent_owner_ids()
            remove_id, keep_id = int(owner_ids[0]), int(owner_ids[16])
            owner_ids[owner_ids == remove_id] = 0
            target = tuple(value.detach().clone() for value in model.forward(100))
            next_id = model.persistent_state.next_region_id
            self.assertTrue(model._try_persistent_owner_ids(owner_ids, *target,
                                                           redirects={remove_id: 0}))
            self.assertEqual(model.persistent_state.region_ids.tolist(), [0, keep_id])
            self.assertEqual(model.region_map.tolist(), [0, 0, 1])
            self.assertEqual(model.persistent_state.next_region_id, next_id)
            self.assertEqual(model._persistent_owner_ids().tolist(), owner_ids.tolist())
            self.assertTrue(model._online_packet_score(*target)['valid'])
            saved = rigid.RigidResidualTree(model.config)
            saved.restore(copy.deepcopy(model.capture()))
            self.assertEqual(saved.region_map.tolist(), [0, 0, 1])
            self.assertEqual(saved.persistent_state.region_ids.tolist(), [0, keep_id])
            decoded2 = self.finish_export(model, Path(directory) / 'motion_002.bin', birth_source=16)
            self.assertEqual(decoded2.parents.tolist(), [-1, 0])
            self.assertEqual(decoded2.owner[-1], decoded2.owner[16])
            self.assertEqual(model.materialized_online_ids[-1], 32)

    def test_valid_observation_evidence_split_and_idle_compaction(self):
        _, persistent, _, clustering, _ = self.modules()
        model = self.fixture()
        self.initialize(model)
        logical = model._persistent_owner_ids()
        first_id, second_id = int(logical[0]), int(logical[16])
        first_slot = model.persistent_state.region_ids.tolist().index(first_id)
        with torch.no_grad():
            model.gs_translation[:8, 0] = .06
            model.gs_translation[8:16, 0] = -.06
        for frame, observed in ((1, True), (2, False), (3, True), (4, True)):
            model.frame_index = frame
            model.last_step = 200
            model.observations[:16] = 10 if observed else 0
            model._finalize_persistent_evidence()
            if frame == 2:
                self.assertEqual(model.persistent_state.region_evidence[first_id]['last_valid_frame'], 1)
                self.assertEqual(model.persistent_state.region_evidence[first_id]['split_streak'], 1)
            if frame == 3:
                model._finalize_persistent_evidence()
                self.assertEqual(model.persistent_state.region_evidence[first_id]['split_streak'], 2)
        self.assertEqual(model.persistent_state.region_evidence[first_id]['split_streak'], 3)
        self.assertGreater(model.persistent_state.region_evidence[first_id]['p90_units'], 4.)
        self.assertNotIn(0, model.persistent_state.region_evidence)
        # Actual binary split, supplied with only causal observed transitions.
        model.frame_index = 10
        target = tuple(value.detach().clone() for value in model.forward(100))
        history = clustering.OnlineMotionHistory(5)
        history.append(10, model.online_ids.numpy(), model.base_xyz.numpy(),
                       target[0].numpy(), observed=np.ones(32, dtype=bool))
        next_id = model.persistent_state.next_region_id
        self.assertTrue(model._split_persistent_region(first_id, *target, history))
        self.assertEqual(model.persistent_state.next_region_id, next_id + 1)
        self.assertIn(next_id, model.persistent_state.region_ids.tolist())
        self.assertEqual(int((model._persistent_owner_ids() == first_id).sum()), 8)
        self.assertEqual(int((model._persistent_owner_ids() == next_id).sum()), 8)
        self.assertNotIn(first_id, model.persistent_state.region_evidence)
        self.assertLessEqual(float((model.forward(100)[0] - target[0]).norm(dim=-1).max()), .000867)
        # Independent idle evidence retains missing frames, and a logical leaf
        # is recycled to root only after three valid refined observations.
        idle = self.fixture()
        self.initialize(idle)
        with torch.no_grad():
            idle.region_translation[first_slot] = 0
            idle.region_rotation[first_slot] = 0
            idle.gs_translation[:16] = 0
            idle.gs_rotation[:16] = 0
        for frame, observed in ((1, True), (2, False), (3, True), (4, True)):
            idle.frame_index = frame
            idle.observations[:16] = 10 if observed else 0
            idle._finalize_persistent_evidence()
        self.assertEqual(idle.persistent_state.region_evidence[first_id]['idle_streak'], 3)
        idle.frame_index = 10
        target = tuple(value.detach().clone() for value in idle.forward(100))
        history = clustering.OnlineMotionHistory(5)
        history.append(10, idle.online_ids.numpy(), idle.base_xyz.numpy(), target[0].numpy(),
                       observed=np.ones(32, dtype=bool))
        before_next = idle.persistent_state.next_region_id
        edits = idle._edit_persistent_structure(*target, history)
        self.assertIn(dict(kind='persistent_idle_to_root', region_id=first_id), edits)
        self.assertEqual(idle.persistent_state.region_ids.tolist(), [0, second_id])
        self.assertEqual(idle.persistent_state.next_region_id, before_next)
        self.assertEqual(idle._persistent_owner_ids()[:16].tolist(), [0] * 16)
        self.assertLessEqual(float((idle.forward(100)[0] - target[0]).norm(dim=-1).max()), .000867)

    def test_compaction_keeps_owner_empty_internal_ancestors_and_rejects_cycles(self):
        _, persistent, _, _, _ = self.modules()
        model = self.fixture()
        self.initialize(model)
        model.owner[:16] = 2
        model.owner[16:24] = 3
        model.parents = torch.tensor([-1, 0, 1, 0])
        model.persistent_state.region_ids = torch.tensor([0, 11, 22, 33])
        model.persistent_state.reference_region_ids = torch.tensor([0, 11, 22, 33])
        logical = model._persistent_owner_ids()
        logical[logical == 33] = 0
        compact = model._compact_persistent_layout(logical, {33: 0})
        self.assertEqual(compact['region_ids'].tolist(), [0, 11, 22])
        self.assertEqual(compact['parents'].tolist(), [-1, 0, 1])
        self.assertEqual(compact['region_map'].tolist(), [0, 1, 2, 0])
        self.assertFalse(bool(compact['owner'].eq(1).any()))
        self.assertEqual(int(compact['owner'].eq(2).sum()), 16)
        self.assertEqual(persistent.ancestry_closure([2], [-1, 0, 1, 0]), [0, 1, 2])
        for parents in ([-1, 1], [-1, 2, 1]):
            with self.assertRaises(ValueError):
                persistent.ancestry_closure([1], parents)

    def test_shared_motion_merge_reduces_quantized_cost_and_decodes_dense_redirect(self):
        _, _, _, clustering, _ = self.modules()
        model = self.fixture()
        self.initialize(model)
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.finish_export(model, Path(directory) / 'motion_001.bin')
            model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
            model.observations.fill_(10)
            owner_ids = model._persistent_owner_ids()
            with torch.no_grad():
                model.gs_translation[owner_ids > 0, 0] = .02
            target = tuple(value.detach().clone() for value in model.forward(100))
            self.assertTrue(model._try_persistent_owner_ids(owner_ids, *target))
            cost = model._persistent_quantized_cost()
            history = clustering.OnlineMotionHistory(5)
            history.append(2, model.online_ids.numpy(), model.base_xyz.numpy(), target[0].numpy(),
                           observed=np.ones(32, dtype=bool))
            edits = model._edit_persistent_structure(*target, history)
            self.assertIn(dict(kind='persistent_merge', source_region_id=2, destination_region_id=1), edits)
            self.assertLess(model._persistent_quantized_cost(), cost)
            self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1])
            self.assertEqual(model.region_map.tolist(), [0, 1, 1])
            self.assertEqual(model.persistent_state.next_region_id, 3)
            self.assertLessEqual(float((model.forward(100)[0] - target[0]).norm(dim=-1).max()), .000867)
            decoded2 = self.finish_export(model, Path(directory) / 'motion_002.bin')
            self.assertEqual(decoded2.parents.tolist(), [-1, 0])
            self.assertEqual(decoded2.owner[:24].tolist(), [1] * 24)

    def test_existing_tree_joint_losses_checkpoint_and_next_frame_world_refit(self):
        rigid, _, _, _, _ = self.modules()
        tree = importlib.import_module('pipeline.ReCon-GS.model.online_motion_tree')
        base = self.fixture()
        config = replace(base.config,
            persistent_regions=replace(base.config.persistent_regions, tree_enabled=True),
            relative_motion_loss_weight=1., tree_child_relative_loss_weight=1.)
        model = rigid.RigidResidualTree(config)
        model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
        model.observations.fill_(10)
        self.initialize(model)
        with tempfile.TemporaryDirectory() as directory:
            decoded1 = self.finish_export(model, Path(directory) / 'motion_001.bin')
            model.setup(decoded1.xyz, decoded1.rotation, 1., frame_index=2)
            model.observations.fill_(10)
            # Install a legitimate accepted-tree checkpoint fixture. Admission
            # and rollback are tested by the independent mixin test file.
            model.parents = torch.tensor([-1, 3, 3, 0])
            model.persistent_state.region_ids = torch.tensor([0, 1, 2, 3])
            model.persistent_state.next_region_id = 4
            model.pivots = rigid.region_pivots(model.base_xyz, model.owner, model.parents)
            shared = rigid.rotation_vector_quaternion(torch.tensor([0., 0., .01]))
            q_world = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
            q_world[1:] = shared
            b_world = torch.zeros(4, 3)
            b_world[1:] = torch.tensor([.02, .01, 0.])
            target_xyz = model.base_xyz.clone()
            target_q = model.base_rotation.clone()
            members = model.owner > 0
            target_xyz[members] = rigid.quaternion_rotate(shared.expand(int(members.sum()), -1),
                model.base_xyz[members]) + b_world[1]
            target_q[members] = rigid.quaternion_multiply(shared.expand(int(members.sum()), -1),
                model.base_rotation[members])
            local_t, local_r = tree.world_to_local_motion(model.parents.numpy(), model.pivots.numpy(),
                q_world.numpy(), b_world.numpy())
            self.assertTrue(model._install_tree_motion_and_retarget(local_t, local_r, target_xyz, target_q))
            model._make_optimizer()
            model._region_id_lengths = rigid._id_lengths(3, model.base_xyz)
            model._topology_updated = True
            self.assertFalse(bool(model.owner.eq(3).any()))
            with torch.no_grad():
                model.region_translation[1, 0] = .004
                model.region_rotation[1, 1] = .003
                model.region_translation[2, 1] = .002
                model.region_rotation[2, 0] = .001
                model.gs_translation[0, 2] = .003
                model.gs_rotation[0, 0] = .002
            model.forward(101, training=True)
            terms = model.loss_terms(101, diagnostics=False)
            self.assertAlmostEqual(float(terms['relative_motion_loss']), .005 / 32, places=6)
            self.assertAlmostEqual(float(terms['child_relative_motion_loss']), .005, places=6)
            (terms['relative_motion_loss'] + terms['child_relative_motion_loss']).backward()
            self.assertGreater(float(model.gs_translation.grad.abs().sum()), 0.)
            self.assertGreater(float(model.gs_rotation.grad.abs().sum()), 0.)
            self.assertGreater(float(model.region_translation.grad[1:3].abs().sum()), 0.)
            self.assertGreater(float(model.region_rotation.grad[1:3].abs().sum()), 0.)
            self.assertEqual(float(model.region_translation.grad[3].abs().sum()), 0.)
            self.assertEqual(float(model.region_rotation.grad[3].abs().sum()), 0.)
            model.optimizer.zero_grad(set_to_none=True)
            xyz, q = model.forward(101, training=True)
            desired_xyz, desired_q = xyz.detach().clone(), q.detach().clone()
            desired_xyz[members, 1] += .005
            delta = rigid.rotation_vector_quaternion(torch.tensor([.004, 0., 0.]))
            desired_q[members] = rigid.quaternion_multiply(delta.expand(int(members.sum()), -1),
                                                           desired_q[members])
            ((xyz - desired_xyz).square().sum() + (q - desired_q).square().sum()).backward()
            self.assertGreater(float(model.region_translation.grad[3].abs().sum()), 0.)
            self.assertGreater(float(model.region_rotation.grad[3].abs().sum()), 0.)
            old_parent_t, old_parent_r = model.region_translation[3].detach().clone(), model.region_rotation[3].detach().clone()
            model.optimizer_step()
            self.assertFalse(torch.equal(model.region_translation[3], old_parent_t))
            self.assertFalse(torch.equal(model.region_rotation[3], old_parent_r))
            restored = rigid.RigidResidualTree(config)
            restored.restore(copy.deepcopy(model.capture()))
            self.assertEqual(restored.parents.tolist(), [-1, 3, 3, 0])
            self.assertEqual(restored.persistent_state.region_ids.tolist(), [0, 1, 2, 3])
            for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            decoded2 = self.finish_export(model, Path(directory) / 'motion_002.bin', birth_source=0)
            self.assertEqual(decoded2.parents.tolist(), [-1, 3, 3, 0])
            model.setup(decoded2.xyz, decoded2.rotation, 1., frame_index=3)
            model.observations.fill_(10)
            model.observations[-1] = 0  # Newly born GS has no previous transition evidence.
            target_xyz, target_q = model.base_xyz.clone(), model.base_rotation.clone()
            members = model.owner > 0
            delta = rigid.rotation_vector_quaternion(torch.tensor([0., 0., .005]))
            target_xyz[members] = rigid.quaternion_rotate(delta.expand(int(members.sum()), -1),
                model.base_xyz[members]) + torch.tensor([.015, .01, -.002])
            target_q[members] = rigid.quaternion_multiply(delta.expand(int(members.sum()), -1),
                model.base_rotation[members])
            logical = model._persistent_owner_ids().clone()
            self.assertTrue(model._try_persistent_owner_ids(logical, target_xyz, target_q))
            self.assertEqual(model.parents.tolist(), [-1, 3, 3, 0])
            self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2, 3])
            self.assertTrue(model._online_packet_score(target_xyz, target_q)['valid'])
            self.assertFalse(bool(model.owner.eq(3).any()))
            self.assertGreater(float(model.region_translation[3].abs().sum()), 0.)
            self.assertGreater(float(model.region_rotation[3].abs().sum()), 0.)
            self.assertTrue(torch.equal(model._persistent_owner_ids(), logical))
            aligned = model.online_history.aligned(model.online_ids.numpy())
            self.assertFalse(bool(aligned['observed'][-1].any()))
            decoded3 = self.finish_export(model, Path(directory) / 'motion_003.bin')
            self.assertEqual(decoded3.parents.tolist(), [-1, 3, 3, 0])
            self.assertEqual(len(model.online_history.records), 3)


if __name__ == '__main__':
    unittest.main()
