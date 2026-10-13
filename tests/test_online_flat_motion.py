"""Renderer-free numerical checks for causal flat shared-motion integration."""

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

DIAGNOSTICS = importlib.import_module('pipeline.ReCon-GS.model.online_motion_diagnostics')


def wxyz(rotation):
    return rotation.as_quat()[..., [3, 0, 1, 2]]


class FlatMotionDiagnosticsTests(unittest.TestCase):
    def test_quantized_counts_separate_blocks_and_exclude_identity_root(self):
        rt = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], np.int16)
        rr = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], np.int16)
        gt = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]], np.int16)
        gr = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]], np.int16)
        counts = DIAGNOSTICS.quantized_motion_counts(rt, rr, gt, gr)
        self.assertEqual(counts['region_active_nodes'], 2)
        self.assertEqual(counts['region_translation_active_blocks'], 1)
        self.assertEqual(counts['region_rotation_active_blocks'], 1)
        self.assertEqual(counts['gs_translation_active_blocks'], 2)
        self.assertEqual(counts['gs_rotation_active_blocks'], 2)
        self.assertEqual(counts['gs_active_rows'], 3)
        self.assertEqual(counts['gs_zero_residual_fraction'], .25)
        empty = np.empty((0, 3), np.int16)
        self.assertEqual(DIAGNOSTICS.quantized_motion_counts(empty, empty, empty, empty)['gs_active_rows'], 0)
        with self.assertRaises(ValueError):
            DIAGNOSTICS.quantized_motion_counts(rt, rr, gt.astype(float), gr)
        json.dumps(counts, allow_nan=False)

    def test_label_permutation_is_not_churn_and_splits_are(self):
        metric = DIAGNOSTICS.ownership_changed_fraction
        self.assertEqual(metric([0, 0, 1, 1, -1], [7, 7, 9, 9, -1]), 0.)
        self.assertAlmostEqual(metric([0, 0, 0, 0], [7, 7, 9, 9]), .5)
        self.assertAlmostEqual(metric([-3, -2, -1], [-2, -3, -1]), 2 / 3)
        self.assertEqual(metric([], []), 0.)
        self.assertEqual(metric([], [1]), 0.)
        self.assertEqual(metric([0], [9, 8]), 0.)

    def test_residuals_use_composed_noncommuting_rotations(self):
        source = np.array([[1., 2., 3.], [3., 0., 1.]])
        shared = Rotation.from_rotvec([0., 0., np.pi / 2])
        original = Rotation.from_rotvec([[np.pi / 3, 0., 0.], [0., np.pi / 5, 0.]])
        residual = Rotation.from_rotvec([[0., .1, 0.], [.2, 0., 0.]])
        shift = np.array([.3, -.2, .1])
        target = shared.apply(source) + shift + np.array([[.1, 0., 0.], [0., .2, 0.]])
        target_q = wxyz(residual * shared * original)
        target_q[1] *= -1  # quaternion antipodes encode the same orientation
        report = DIAGNOSTICS.summarize_flat_regions(
            source, target, wxyz(original), target_q, [0, 0],
            wxyz(shared)[None], shift[None], extent=2.)
        region = report['regions'][0]
        self.assertAlmostEqual(region['independent_translation_residual']['mean'], .15)
        self.assertAlmostEqual(region['independent_translation_residual_extent']['mean'], .075)
        self.assertAlmostEqual(region['independent_rotation_residual_rad']['mean'], .15)
        self.assertAlmostEqual(region['independent_rotation_residual_rad']['max'], .2)
        self.assertAlmostEqual(region['shared_rotation_angle_degrees'], 90.)
        self.assertEqual(region['gs_proportion_all'], 1.)
        json.dumps(report, allow_nan=False)

    def test_singleton_new_gs_and_missing_observations_use_all_gs_denominator(self):
        xyz = np.array([[0., 0., 0.], [1., 0., 0.], [np.nan, np.nan, np.nan]])
        q = np.tile([1., 0., 0., 0.], (3, 1))
        report = DIAGNOSTICS.summarize_flat_regions(
            xyz, xyz, q, q, [0, 1, 1], q[:2], np.zeros((2, 3)),
            observed_mask=[True, False, False], static_mask=[True, False, False],
            noise_mask=[False, True, False], unknown_mask=[False, False, True],
            previous_owner=[4, 5], candidate_owner=[3, 3, -3])
        self.assertEqual(report['new_gs_without_prior_owner'], 1)
        self.assertEqual(report['ownership_changed_fraction'], 0.)
        self.assertEqual(report['candidate_ownership_changed_fraction'], .5)
        self.assertEqual(report['regions'][0]['gs_proportion_all'], 1 / 3)
        self.assertEqual(report['regions'][1]['gs_count'], 2)
        self.assertIsNone(report['regions'][1]['independent_rotation_residual_rad']['mean'])
        self.assertEqual(report['regions'][0]['independent_translation_residual']['max'], 0.)
        self.assertEqual(report['static_fraction'], 1 / 3)
        self.assertEqual(report['unknown_fraction'], 1 / 3)
        json.dumps(report, allow_nan=False)

    def test_empty_and_unobserved_regions_have_null_statistics(self):
        for count in (0, 1):
            xyz = np.zeros((count, 3))
            q = np.tile([1., 0., 0., 0.], (count, 1))
            report = DIAGNOSTICS.summarize_flat_regions(
                xyz, xyz, q, q, np.zeros(count, dtype=np.int64),
                [[1., 0., 0., 0.], [1., 0., 0., 0.]], np.zeros((2, 3)),
                observed_mask=np.zeros(count, dtype=bool))
            self.assertEqual(report['residual_valid_gs'], 0)
            for region in report['regions']:
                self.assertEqual(region['independent_translation_residual']['count'], 0)
                self.assertIsNone(region['independent_translation_residual']['mean'])
            json.dumps(report, allow_nan=False)

    def test_missing_mask_defaults_to_finite_rows_and_invalid_mask_fails(self):
        xyz = np.zeros((1, 3))
        q = np.array([[1., 0., 0., 0.]])
        report = DIAGNOSTICS.summarize_flat_regions(xyz, xyz, q, q, [0], q, xyz)
        self.assertEqual(report['residual_valid_gs'], 1)
        with self.assertRaises(ValueError):
            DIAGNOSTICS.summarize_flat_regions(xyz, xyz, q, q, [0], q, xyz,
                                              observed_mask=[])


class OnlineFlatIntegrationTests(unittest.TestCase):
    @staticmethod
    def modules():
        return tuple(importlib.import_module('pipeline.ReCon-GS.model.' + name)
                     for name in ('rigid_residual_tree', 'online_motion_clustering',
                                  'online_flat_motion', 'rigid_tree_codec'))

    def fixture(self, count=48):
        rigid, clustering, _, _ = self.modules()
        config = rigid.RigidTreeConfig(
            online_flat_clustering=True, adaptive_clustering=True,
            region_count=1, shared_steps=0, motion_steps=100, birth_steps=100,
            tree_edit_step=100, motion_xyz_step_extent=.0001,
            motion_rotation_step_rad=.0001, gs_xyz_step_extent=.0001,
            gs_rotation_step_rad=.0001, rate_loss_weight=0.,
            online_clustering=clustering.OnlineMotionClusteringConfig(
                method='kmeans', min_region_size=2))
        model = rigid.RigidResidualTree(config)
        xyz = torch.from_numpy(np.random.default_rng(4).normal(size=(count, 3))).float()
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1)
        model.setup(xyz, q, 1., frame_index=1)
        model.observations.fill_(10)
        return model

    def assert_diagnostics_json_roundtrip(self, report):
        def native_types(value, path='diagnostics'):
            if type(value) is dict:
                for key, item in value.items():
                    self.assertIs(type(key), str, path)
                    native_types(item, path + '.' + key)
            elif type(value) is list:
                for index, item in enumerate(value):
                    native_types(item, f'{path}[{index}]')
            else:
                self.assertIn(type(value), (bool, int, float, str, type(None)), path)
        native_types(report)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame_001.json'
            path.write_text(json.dumps(report, indent=2, allow_nan=False))
            restored = json.loads(path.read_text())
        self.assertEqual(restored, report)
        self.assertIs(type(restored['gate']['accepted']), bool)
        self.assertIs(type(restored['gate']['old']['valid']), bool)
        self.assertIs(type(restored['gate']['old']['bytes']), int)
        self.assertIs(type(restored['gate']['old']['max_translation_error']), float)
        if restored['gate']['candidate'] is not None:
            self.assertIs(type(restored['gate']['candidate']['valid']), bool)
        return restored

    def test_actual_proposal_gate_and_export_diagnostics_are_native_json_types(self):
        rigid, _, _, _ = self.modules()
        for branch in ('accepted', 'not_smaller', 'quantization_error'):
            with self.subTest(branch=branch):
                base = self.fixture(48 if branch == 'accepted' else 4)
                clustering = replace(base.config.online_clustering,
                                     component_counts=[1], max_feature_distance=10000.)
                config = replace(base.config, online_clustering=clustering,
                    online_partition_policy='require_smaller',
                    motion_xyz_step_extent=1e-6 if branch == 'quantization_error' else .0001)
                model = rigid.RigidResidualTree(config)
                model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
                model.observations.fill_(10)
                with torch.no_grad():
                    if branch == 'accepted':
                        rotation = Rotation.from_rotvec([.035, -.04, .08])
                        target = rotation.apply(model.base_xyz.numpy()) + [.04, -.02, .01]
                        model.gs_translation.copy_(torch.from_numpy(target).float() - model.base_xyz)
                        model.gs_rotation[:] = torch.tensor([.035, -.04, .08])
                    elif branch in ('not_smaller', 'quantization_error'):
                        model.gs_translation[:, 0] = .1 if branch == 'quantization_error' else .01
                # Use the actual clustering and complete packet gate, as the
                # frame1 step100 writer does, without mocking candidate metrics.
                model.update_topology(100)
                restored = self.assert_diagnostics_json_roundtrip(model.flat_region_diagnostics)
                self.assertEqual(restored['proposal']['method'], 'kmeans')
                self.assertIs(type(restored['proposal']['candidate_regions']), int)
                self.assertEqual(restored['gate']['accepted'], branch == 'accepted')
                self.assertEqual(restored['gate']['rejection_reason'], {
                    'accepted': None, 'not_smaller': 'complete_packet_not_smaller',
                    'quantization_error': 'quantization_error'}[branch])
                xyz, q = (value.detach().clone() for value in model.materialized_pose())
                model.register_materialized_layout(xyz, q)
                with tempfile.TemporaryDirectory() as directory:
                    model.export_motion(Path(directory) / 'motion_001.bin')
                final = self.assert_diagnostics_json_roundtrip(model.flat_region_diagnostics)
                self.assertEqual(final['accepted_history_transition_count'], 1)

    def test_all_actual_clustering_methods_write_native_json_diagnostics(self):
        rigid, _, _, _ = self.modules()
        for method in ('kmeans', 'gmm', 'hdbscan', 'greedy'):
            with self.subTest(method=method):
                base = self.fixture()
                clustering = replace(base.config.online_clustering, method=method,
                                     component_counts=[1], max_feature_distance=10000.)
                model = rigid.RigidResidualTree(replace(base.config, online_clustering=clustering))
                model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
                model.observations.fill_(10)
                rotation = Rotation.from_rotvec([.035, -.04, .08])
                target = rotation.apply(model.base_xyz.numpy()) + [.04, -.02, .01]
                with torch.no_grad():
                    model.gs_translation.copy_(torch.from_numpy(target).float() - model.base_xyz)
                    model.gs_rotation[:] = torch.tensor([.035, -.04, .08])
                model.update_topology(100)
                restored = self.assert_diagnostics_json_roundtrip(model.flat_region_diagnostics)
                self.assertEqual(restored['proposal']['method'], method)
                gate = restored['gate']
                self.assertEqual(gate['accepted'], bool(gate['candidate'] and
                    gate['candidate']['valid'] and restored['proposal']['candidate_regions'] > 0))
                self.assertEqual(gate['acceptance_policy'], 'accept_valid')
                xyz, q = (value.detach().clone() for value in model.materialized_pose())
                model.register_materialized_layout(xyz, q)
                with tempfile.TemporaryDirectory() as directory:
                    model.export_motion(Path(directory) / 'motion_001.bin')
                self.assert_diagnostics_json_roundtrip(model.flat_region_diagnostics)

    def test_complete_packet_gate_frame0_immutability_and_causal_export(self):
        rigid, _, flat, codec = self.modules()
        model = self.fixture()
        frame0 = model.frame_reference_state
        reference_digest = frame0.digest()
        rotation = Rotation.from_rotvec([.035, -.04, .08])
        target = rotation.apply(model.base_xyz.numpy()) + [.04, -.02, .01]
        with torch.no_grad():
            model.gs_translation.copy_(torch.from_numpy(target).float() - model.base_xyz)
            model.gs_rotation[:] = torch.tensor([.035, -.04, .08])
        original_target = tuple(value.detach().clone() for value in model.forward(100))
        proposal = SimpleNamespace(groups=[np.arange(len(model.owner))],
            labels=np.zeros(len(model.owner), dtype=np.int64), metrics={})
        with patch.object(flat, 'propose_motion_regions', return_value=proposal):
            model.update_topology(100)
        gate = model.flat_region_diagnostics['gate']
        accepted = gate['accepted']
        self.assertTrue(accepted, 'The coherent rigid fixture should save complete packet bytes')
        self.assertTrue(gate['candidate']['valid'])
        self.assertLess(gate['candidate']['bytes'], gate['old']['bytes'])
        for score in (gate['old'], gate['candidate']):
            self.assertEqual(score['raw_bytes'], sum(score[name] for name in (
                'raw_layout_bytes', 'raw_ownership_bytes', 'raw_shared_motion_bytes',
                'raw_residual_motion_bytes', 'raw_header_survivor_checksum_bytes')))
        self.assertEqual(frame0.digest(), reference_digest)
        self.assertEqual(frame0.frame_index, 0)
        self.assertEqual(frame0.parents.tolist(), [-1])
        self.assertTrue(torch.equal(frame0.owner, torch.zeros_like(frame0.owner)))
        self.assertTrue(torch.equal(model.base_xyz, frame0.xyz))
        self.assertTrue(torch.equal(model.parents[1:], torch.zeros_like(model.parents[1:])))
        pose = tuple(value.detach().clone() for value in model.materialized_pose())
        self.assertLessEqual(float((pose[0] - original_target[0]).norm(dim=-1).max()),
                             np.sqrt(3) * .0001 / 2 + 2e-6)
        self.assertLessEqual(float(rigid.quaternion_angle_error(pose[1], original_target[1]).max()),
                             np.sqrt(3) * .0001 / 2 + 2e-6)
        self.assertEqual(model.update_topology(100), [])
        model.register_materialized_layout(*pose)
        with tempfile.TemporaryDirectory() as directory:
            packet_path = Path(directory) / 'frame1.bin'
            result = model.export_motion(packet_path)
            packet = packet_path.read_bytes()
            decoded = codec.decode_motion(packet, frame0)
        self.assertEqual(result['bytes'], gate['candidate' if accepted else 'old']['bytes'])
        self.assertEqual(frame0.digest(), reference_digest)
        self.assertEqual(decoded.digest(), model.codec_previous_state.digest())
        self.assertTrue(torch.equal(decoded.owner, model.owner.cpu()))
        self.assertEqual(len(model.online_history.records), 1)
        restored = rigid.RigidResidualTree(model.config)
        restored.restore(model.capture())
        self.assertEqual(restored.codec_previous_state.digest(), decoded.digest())
        self.assertEqual(restored.materialized_online_ids.tolist(), model.materialized_online_ids.tolist())
        np.testing.assert_array_equal(restored.online_history.records[0]['ids'],
                                      model.online_history.records[0]['ids'])
        model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
        self.assertTrue(torch.equal(model.base_xyz, decoded.xyz))
        self.assertTrue(torch.equal(model.base_rotation, decoded.rotation))
        self.assertIs(model.frame_reference_state, model.codec_previous_state)

    def test_rejected_proposal_records_history_and_births_get_new_stable_ids(self):
        _, _, flat, codec = self.modules()
        model = self.fixture(4)
        proposal = SimpleNamespace(groups=[], labels=np.full(4, -3, dtype=np.int64), metrics={})
        with patch.object(flat, 'propose_motion_regions', return_value=proposal):
            model.update_topology(100)
        self.assertFalse(model.flat_region_diagnostics['gate']['accepted'])
        old_ids = model.online_ids.clone()
        xyz, q = model.materialized_pose()
        xyz = torch.cat((xyz.detach(), xyz[:1].detach() + .1))
        q = torch.cat((q.detach(), q[:1].detach()))
        model.register_materialized_layout(xyz, q, [0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame1.bin'
            model.export_motion(path)
            decoded = codec.decode_motion(path.read_bytes(), model.frame_reference_state)
        self.assertEqual(len(model.online_history.records), 1)
        summary = model.flat_region_diagnostics['regions']
        self.assertEqual(summary['total_gs'], 5)
        self.assertEqual(summary['regions'][0]['gs_count'], 5)
        self.assertEqual(summary['regions'][0]['gs_proportion_all'], 1.)
        self.assertEqual(summary['unknown_fraction'], 1.)
        self.assertEqual(summary['ownership_compared_gs'], 4)
        self.assertEqual(summary['new_gs_without_motion_history'], 1)
        self.assertEqual(model.materialized_online_ids.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(old_ids.tolist(), [0, 1, 2, 3])
        survivors = torch.tensor([4, 1, 0])
        model.setup(decoded.xyz[survivors], decoded.rotation[survivors], 1.,
                    frame_index=2, previous_rows=survivors)
        self.assertEqual(model.online_ids.tolist(), [4, 1, 0])
        self.assertTrue(torch.equal(model.base_xyz, decoded.xyz[survivors]))

    def test_large_shared_translation_does_not_require_unbounded_root_residuals(self):
        rigid, _, _, _ = self.modules()
        base = self.fixture(4)
        config = replace(base.config, gs_xyz_step_extent=1e-6)
        model = rigid.RigidResidualTree(config)
        model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
        model.observations.fill_(10)
        target = model.base_xyz + torch.tensor([.1, 0., 0.])
        owner = torch.ones(4, dtype=torch.long)
        # Total motion exceeds the int16 GS residual range (0.032767), but
        # shared region translation encodes 0.1 with zero independent residual.
        with torch.no_grad():
            fitted = model._fit_online_partition(owner, 2, target, model.base_rotation)
        self.assertTrue(fitted)
        self.assertTrue(model._online_packet_score(target, model.base_rotation)['valid'])
        self.assertLess(float(model.gs_translation.abs().max()), 2e-6)

    def test_unrepresentable_shared_candidate_preserves_valid_old_partition(self):
        rigid, _, flat, _ = self.modules()
        for policy in ('accept_valid', 'require_smaller'):
            with self.subTest(policy=policy):
                base = self.fixture(4)
                model = rigid.RigidResidualTree(replace(base.config,
                    online_partition_policy=policy, motion_xyz_step_extent=1e-6))
                model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
                model.observations.fill_(10)
                with torch.no_grad():
                    model.gs_translation[:, 0] = .1
                pose = tuple(value.detach().clone() for value in model.forward(100))
                proposal = SimpleNamespace(groups=[np.arange(4)], labels=np.zeros(4, np.int64), metrics={})
                with patch.object(flat, 'propose_motion_regions', return_value=proposal):
                    model.update_topology(100)
                gate = model.flat_region_diagnostics['gate']
                self.assertFalse(gate['accepted'])
                self.assertTrue(gate['old']['valid'])
                self.assertIsNone(gate['candidate'])
                self.assertEqual(gate['rejection_reason'], 'quantization_error')
                self.assertEqual(model.parents.tolist(), [-1])
                self.assertEqual(model.owner.tolist(), [0, 0, 0, 0])
                torch.testing.assert_close(model.forward(100)[0], pose[0], rtol=0, atol=1e-7)

    def test_default_accepts_real_larger_partition_and_next_frame_decodes_nonroot(self):
        rigid, _, flat, codec = self.modules()
        for policy in (None, 'require_smaller'):
            with self.subTest(policy='default' if policy is None else policy):
                base = self.fixture(4)
                clustering = replace(base.config.online_clustering,
                                     component_counts=[1], max_feature_distance=10000.)
                config = replace(base.config, online_clustering=clustering)
                if policy is not None:
                    config = replace(config, online_partition_policy=policy)
                else:
                    self.assertEqual(config.online_partition_policy, 'accept_valid')
                model = rigid.RigidResidualTree(config)
                model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
                frame0 = model.frame_reference_state
                frame0_digest = frame0.digest()
                model.observations.fill_(10)
                with torch.no_grad():
                    model.gs_translation[:, 0] = .01
                model.update_topology(100)  # Actual Kmeans proposal, no mocked score.
                gate = self.assert_diagnostics_json_roundtrip(model.flat_region_diagnostics)['gate']
                self.assertTrue(gate['candidate']['valid'])
                self.assertGreater(gate['candidate']['bytes'], gate['old']['bytes'])
                self.assertFalse(gate['packet_improves'])
                self.assertGreater(gate['packet_bytes_delta'], 0)
                accepted = policy is None
                self.assertEqual(gate['accepted'], accepted)
                self.assertEqual(gate['decision_reason'],
                    'valid_candidate' if accepted else 'complete_packet_not_smaller')
                self.assertEqual(model.owner.tolist(), [1, 1, 1, 1] if accepted else [0, 0, 0, 0])
                self.assertEqual(frame0.digest(), frame0_digest)
                xyz, q = (value.detach().clone() for value in model.materialized_pose())
                model.register_materialized_layout(xyz, q)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'motion_001.bin'
                    model.export_motion(path)
                    decoded = codec.decode_motion(path.read_bytes(), frame0)
                    self.assertEqual(decoded.owner.tolist(), model.owner.tolist())
                    self.assertEqual(path.stat().st_size,
                                     gate['candidate' if accepted else 'old']['bytes'])
                    if accepted:
                        previous_owner = model.owner.clone()
                        model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
                        self.assertTrue(torch.equal(model.owner, previous_owner))
                        self.assertTrue(torch.equal(model.base_xyz, decoded.xyz))
                        model.observations.fill_(10)
                        with torch.no_grad():
                            model.region_translation[1, 1] = .02
                            model.region_rotation[1, 2] = .04
                            model.gs_translation[0, 0] = .0001
                        target = tuple(value.detach().clone() for value in model.forward(100))
                        model.update_topology(100)
                        self.assertTrue(model.flat_region_diagnostics['gate']['accepted'])
                        self.assertEqual(model.owner.tolist(), [1, 1, 1, 1])
                        self.assertEqual(model.region_map.tolist(), [0, 1])
                        self.assert_diagnostics_json_roundtrip(model.flat_region_diagnostics)
                        xyz2, q2 = (value.detach().clone() for value in model.materialized_pose())
                        self.assertLessEqual(float((xyz2 - target[0]).norm(dim=-1).max()), .000088)
                        model.register_materialized_layout(xyz2, q2)
                        path2 = Path(directory) / 'motion_002.bin'
                        model.export_motion(path2)
                        decoded2 = codec.decode_motion(path2.read_bytes(), decoded)
                        self.assertEqual(decoded2.digest(), model.codec_previous_state.digest())
                        self.assertEqual(decoded2.owner.tolist(), [1, 1, 1, 1])
                        restored = rigid.RigidResidualTree(model.config)
                        restored.restore(model.capture())
                        self.assertEqual(restored.codec_previous_state.digest(), decoded2.digest())
                        self.assertEqual(len(restored.online_history.records), 2)
                        self.assertEqual(frame0.digest(), frame0_digest)
                        model.setup(decoded2.xyz, decoded2.rotation, 1., frame_index=3)
                        empty = SimpleNamespace(groups=[], labels=np.full(4, -3, np.int64), metrics={})
                        with patch.object(flat, 'propose_motion_regions', return_value=empty):
                            model.update_topology(100)
                        self.assertFalse(model.flat_region_diagnostics['gate']['accepted'])
                        self.assertEqual(model.flat_region_diagnostics['gate']['decision_reason'],
                                         'empty_proposal_keep_previous')
                        self.assertEqual(model.owner.tolist(), [1, 1, 1, 1])

    def test_legacy_online_checkpoint_keeps_strict_policy_and_invalid_policy_fails(self):
        rigid, _, _, _ = self.modules()
        base = self.fixture(4)
        strict = rigid.RigidResidualTree(replace(base.config, online_partition_policy='require_smaller'))
        strict.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
        packet = strict.capture()
        del packet['config']['online_partition_policy']
        restored = rigid.RigidResidualTree(strict.config)
        restored.restore(packet)
        self.assertEqual(restored.config.online_partition_policy, 'require_smaller')
        with self.assertRaises(ValueError):
            rigid.RigidResidualTree(base.config).restore(packet)
        with self.assertRaises(ValueError):
            rigid.RigidResidualTree(replace(base.config, online_partition_policy='accept_anything'))

    def test_refinement_schedule_joint_updates_resume_and_final_decoded_history(self):
        rigid, _, _, codec = self.modules()
        module_source = importlib.import_module('pipeline.ReCon-GS.module')
        base = self.fixture(8)
        clustering = replace(base.config.online_clustering, component_counts=[1],
                             max_feature_distance=10000.)
        config = replace(base.config, online_clustering=clustering, post_partition_steps=100,
            relative_motion_loss_weight=.0001,
            region_translation_lr=.001, region_rotation_lr=.001,
            gs_translation_lr=.001, gs_rotation_lr=.001)
        model = rigid.RigidResidualTree(config)
        model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
        model.observations.fill_(10)
        with torch.no_grad():
            model.gs_translation[:, 0] = .01
            model.gs_translation[:2, 2] = torch.tensor([.0002, -.0002])
        self.assertEqual(config.motion_steps_for_frame(1), 200)
        self.assertEqual(config.total_steps_for_frame(1), 300)
        self.assertEqual(model.stage(99), 'free_motion')
        self.assertEqual(model.stage(100), 'flat_partition')
        self.assertEqual(model.stage(101), 'flat_refine')
        self.assertEqual(model.stage(200), 'flat_refine')
        self.assertEqual(model.stage(201), 'birth_fit')
        model.last_step = 99
        before_partition = rigid.RigidResidualTree(config)
        before_partition.restore(copy.deepcopy(model.capture()))
        self.assertEqual(before_partition.stage(100), 'flat_partition')
        frame0 = model.frame_reference_state
        reference_digest = frame0.digest()
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            harness = SimpleNamespace(motion=model, motion_committed=False,
                trainer=SimpleNamespace(global_step=100, bootstrap_pending=False),
                config=SimpleNamespace(rigid_tree=config,
                    birth=SimpleNamespace(densification_stats_steps=30)),
                output_dir=Path(directory), current_frame=1, motion_steps=200, total_steps=300,
                birth_control={}, gaussians=SimpleNamespace(
                    xyz_gradient_accum=torch.zeros(8, 1), denom=torch.zeros(8, 1),
                    add_densification_stats=Mock()),
                births=SimpleNamespace(_xyz=torch.empty(0, 3)), finalize_frame=Mock())

            def begin_births():
                calls.append(harness.trainer.global_step)
                self.assertTrue(model.flat_region_diagnostics['refinement']['finalized'])
                harness.motion_committed = True
            harness._begin_births = begin_births
            package = dict(visibility_filter=torch.ones(8, dtype=torch.bool),
                           viewspace_points=torch.zeros(8, 3))
            model.last_step = 100
            module_source.Module.post_training_step(harness, package)
            self.assertEqual(calls, [])
            self.assertTrue(model.flat_region_diagnostics['gate']['accepted'])
            self.assertGreater(int((model.owner > 0).sum()), 0)
            partition_owner = model.owner.clone()
            partition_parents = model.parents.clone()
            parameter_start = {name: getattr(model, name).detach().clone() for name in (
                'region_translation', 'region_rotation', 'gs_translation', 'gs_rotation')}
            with self.assertRaises(ValueError):
                model.finalize_online_refinement()
            xyz100, q100 = model.materialized_pose()
            with self.assertRaises(ValueError):
                model.register_materialized_layout(xyz100.detach(), q100.detach())
            after_partition = rigid.RigidResidualTree(config)
            after_partition.restore(copy.deepcopy(model.capture()))
            self.assertNotEqual(after_partition.stage(100), 'flat_partition')
            self.assertEqual(after_partition.stage(101), 'flat_refine')
            self.assertEqual(after_partition.update_topology(100), [])

            rotation = Rotation.from_rotvec([.003, .001, .006])
            desired_xyz = torch.from_numpy(rotation.apply(model.base_xyz.numpy())).float()
            desired_xyz += torch.tensor([.01, .005, -.003])
            desired_xyz[0, 0] += .0003
            desired_q = torch.from_numpy(wxyz(rotation)).float().repeat(8, 1)
            gradient_seen = set()
            resume = None
            for step in range(101, 201):
                xyz, q = model.forward(step, training=True)
                loss = 1000 * ((xyz - desired_xyz).square().mean() +
                               (q - desired_q).square().mean())
                loss += model.loss_terms(step, diagnostics=False)['relative_motion_loss']
                loss.backward()
                for name in parameter_start:
                    gradient = getattr(model, name).grad
                    if gradient is not None and bool(gradient.abs().sum() > 0):
                        gradient_seen.add(name)
                if resume is not None:
                    rxyz, rq = resume.forward(step, training=True)
                    rloss = 1000 * ((rxyz - desired_xyz).square().mean() +
                                    (rq - desired_q).square().mean())
                    rloss += resume.loss_terms(step, diagnostics=False)['relative_motion_loss']
                    rloss.backward()
                    resume.optimizer_step()
                harness.trainer.global_step = step
                module_source.Module.post_training_step(harness, package)
                self.assertTrue(torch.equal(model.owner, partition_owner))
                self.assertTrue(torch.equal(model.parents, partition_parents))
                if step == 150:
                    resume = rigid.RigidResidualTree(config)
                    resume.restore(copy.deepcopy(model.capture()))
            self.assertEqual(gradient_seen, set(parameter_start))
            for name, value in parameter_start.items():
                self.assertFalse(torch.equal(getattr(model, name), value), name)
                torch.testing.assert_close(getattr(model, name), getattr(resume, name), rtol=0, atol=0)
            self.assertEqual(calls, [200])
            self.assertEqual(len(model.topology_edits), 1)
            self.assertEqual(len(model.online_history.records), 0)
            refinement = model.flat_region_diagnostics['refinement']
            self.assertTrue(refinement['membership_fixed'])
            self.assertTrue(refinement['finalized'])
            self.assertEqual(refinement['before']['step'], 100)
            self.assertEqual(refinement['after']['step'], 200)
            self.assert_diagnostics_json_roundtrip(model.flat_region_diagnostics)
            for snapshot in (refinement['before'], refinement['after']):
                self.assertEqual(snapshot['quantized_motion']['total_surviving_gs'], 8)
                for region in snapshot['regions']['regions']:
                    for field in ('independent_translation_residual', 'independent_rotation_residual_rad'):
                        self.assertEqual(set(region[field]), {'count', 'mean', 'p50', 'p90', 'max'})
            finalized = copy.deepcopy(refinement['after'])
            model.finalize_online_refinement()
            self.assertEqual(refinement['after'], finalized)
            saved = rigid.RigidResidualTree(config)
            saved.restore(copy.deepcopy(model.capture()))
            self.assertTrue(saved.flat_region_diagnostics['refinement']['finalized'])
            final_xyz, final_q = (value.detach().clone() for value in model.materialized_pose())
            self.assertFalse(torch.equal(final_xyz, xyz100))
            model.register_materialized_layout(final_xyz, final_q)
            path = Path(directory) / 'motion_001.bin'
            result = model.export_motion(path)
            decoded = codec.decode_motion(path.read_bytes(), frame0)
            self.assertEqual(result['bytes'], refinement['after']['packet_score']['bytes'])
            torch.testing.assert_close(decoded.xyz, final_xyz, rtol=0, atol=1e-6)
            self.assertEqual(len(model.online_history.records), 1)
            np.testing.assert_allclose(model.online_history.records[0]['target'], decoded.xyz.numpy(),
                                       rtol=0, atol=0)
            self.assertEqual(frame0.digest(), reference_digest)
            harness.trainer.global_step = 300
            module_source.Module.post_training_step(harness, package)
            harness.finalize_frame.assert_called_once()

    def test_refinement_relative_loss_regularizes_gs_residuals_only(self):
        rigid, _, _, _ = self.modules()
        base = self.fixture(4)
        model = rigid.RigidResidualTree(replace(base.config, post_partition_steps=100,
                                               relative_motion_loss_weight=1.))
        model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
        with torch.no_grad():
            model.gs_translation[:, 0] = .01
            model.gs_rotation[:, 1] = .02
        model.forward(99, training=True)
        self.assertEqual(float(model.loss_terms(99, diagnostics=False)['relative_motion_loss']), 0.)
        model.forward(101, training=True)
        loss = model.loss_terms(101, diagnostics=False)['relative_motion_loss']
        self.assertAlmostEqual(float(loss), .03, places=6)
        loss.backward()
        self.assertGreater(float(model.gs_translation.grad.abs().sum()), 0.)
        self.assertGreater(float(model.gs_rotation.grad.abs().sum()), 0.)
        self.assertIsNone(model.region_translation.grad)
        self.assertIsNone(model.region_rotation.grad)


if __name__ == '__main__':
    unittest.main()
