"""Real persistent-root proposals, joint fitting, checkpoints and causal codec."""

import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_persistent_motion_integration as lifecycle_tests


class RootDiscoveryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.lifecycle = lifecycle_tests.PersistentLifecycleTests()
        self.rigid, _, self.flat, _, self.codec = self.lifecycle.modules()
        self.discovery = importlib.import_module('pipeline.ReCon-GS.model.root_motion_discovery')

    def fixture(self):
        base = self.lifecycle.fixture(count=64)
        base.config.root_motion_discovery = self.discovery.RootMotionDiscoveryConfig(
            enabled=True, max_candidates=1)
        base.config.validate()
        xyz = base.base_xyz.clone()
        xyz[24:56] = torch.tensor([0., 1., 0.]) + torch.randn(
            (32, 3), generator=torch.Generator().manual_seed(19)) * .003
        model = self.rigid.RigidResidualTree(base.config)
        model.setup(xyz, base.base_rotation, 1., frame_index=1)
        model.observations.fill_(10)
        self.lifecycle.initialize(model)
        return model

    @staticmethod
    def native_json(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert type(key) in (str, int)
                RootDiscoveryIntegrationTests.native_json(item)
        elif isinstance(value, list):
            for item in value:
                RootDiscoveryIntegrationTests.native_json(item)
        else:
            assert type(value) in (str, int, float, bool, type(None)), type(value)
        json.dumps(value, allow_nan=False)

    def next_motion(self, model, decoded, frame, missing=False):
        model.setup(decoded.xyz, decoded.rotation, 1., frame_index=frame)
        model.observations.fill_(10)
        owners = model._persistent_owner_ids().clone()
        with torch.no_grad():
            model.gs_translation[owners == 1, 0] = -.03
            model.gs_translation[owners == 2, 1] = -.03
            model.gs_translation[24:56, 0] = .04
            model.gs_translation[24:56, 1] = .01
            if len(owners) > 64:
                model.gs_translation[64, :2] = torch.tensor([.04, .01])
        if missing:
            model.observations[24:56] = 0
        target = tuple(value.detach().clone() for value in model.forward(100))
        with patch.object(self.flat, 'propose_motion_regions',
                          side_effect=AssertionError('Persistent discovery must not globally recluster')):
            model.update_topology(100)
        self.native_json(model.flat_region_diagnostics)
        return owners, target

    def test_real_two_observations_newborn_joint_steps_and_decoded_history(self):
        model = self.fixture()
        frame0 = model.frame_reference_state
        frame0_digest = frame0.digest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            decoded1 = self.lifecycle.finish_export(model, path / 'motion_001.bin')
            inherited, _ = self.next_motion(model, decoded1, 2)
            discovery2 = model.flat_region_diagnostics['root_discovery']
            self.assertEqual(discovery2['accepted_regions'], [])
            self.assertEqual(discovery2['proposal']['sustained_moving_root_gs'], 0)
            self.assertTrue(torch.equal(inherited, model._persistent_owner_ids()))
            decoded2 = self.lifecycle.finish_export(model, path / 'motion_002.bin', birth_source=24)
            old, target = self.next_motion(model, decoded2, 3)
            self.assertEqual(model.online_ids[-1].item(), 64)
            discovery3 = model.flat_region_diagnostics['root_discovery']
            self.assertEqual(discovery3['proposal']['sustained_moving_root_gs'], 32)
            self.assertEqual(discovery3['accepted_regions'], [dict(region_id=3, gs_count=32)])
            self.assertTrue(discovery3['trials'][0]['after_packet']['valid'])
            self.assertTrue(torch.equal(model._persistent_owner_ids()[:24], old[:24]))
            self.assertEqual(model._persistent_owner_ids()[24:56].tolist(), [3] * 32)
            self.assertEqual(model._persistent_owner_ids()[64].item(), 0)
            self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2, 3])
            self.assertEqual(model.persistent_state.next_region_id, 4)
            self.assertEqual(model.region_map.tolist(), [0, 1, 2])
            # Repeating the hook within a frame cannot consume logical IDs twice.
            repeat = model._discover_persistent_root_regions(*target,
                model._persistent_current_history(*target))
            self.assertTrue(repeat['same_frame_skipped'])
            self.assertEqual(model.persistent_state.next_region_id, 4)
            ownership = model.owner.clone()
            initial = {name: getattr(model, name).detach().clone() for name in (
                'region_translation', 'region_rotation', 'gs_translation', 'gs_rotation')}
            desired_xyz, desired_q = (value.detach().clone() for value in model.forward(100))
            desired_xyz[24:56, 2] += .006
            desired_xyz[24, 0] += .004
            delta = self.rigid.rotation_vector_quaternion(torch.tensor([.008, 0., 0.]))
            desired_q[24:56] = self.rigid.quaternion_multiply(delta.expand(32, -1), desired_q[24:56])
            desired_q[24] = self.rigid.quaternion_multiply(
                self.rigid.rotation_vector_quaternion(torch.tensor([0., .006, 0.])), desired_q[24])
            # Full 100-step refinement on the installed shared AND GS parameters.
            for step in range(101, 201):
                model.optimizer.zero_grad(set_to_none=True)
                xyz, rotation = model.forward(step, training=True)
                loss = ((xyz - desired_xyz).square().sum() +
                        (rotation - desired_q).square().sum()) * 1000
                loss.backward()
                if step == 101:
                    for name in initial:
                        gradient = getattr(model, name).grad
                        self.assertIsNotNone(gradient)
                        self.assertGreater(float(gradient.abs().sum()), 0., name)
                model.optimizer_step()
                model.update_topology(step)
                self.assertTrue(torch.equal(model.owner, ownership))
                if step == 150:
                    resumed = self.rigid.RigidResidualTree(model.config)
                    resumed.restore(copy.deepcopy(model.capture()))
                    self.assertEqual(resumed.persistent_state.last_discovery_result,
                                     model.persistent_state.last_discovery_result)
                    self.assertEqual(resumed.persistent_state.next_region_id, 4)
                    for name in initial:
                        torch.testing.assert_close(getattr(resumed, name), getattr(model, name), rtol=0, atol=0)
                    model = resumed
            for name, original in initial.items():
                self.assertFalse(torch.equal(getattr(model, name), original), name)
            final_xyz, final_q = (value.detach().clone() for value in model.materialized_pose())
            self.assertLess(float((final_xyz - desired_xyz).norm(dim=-1).max()), .002)
            decoded3 = self.lifecycle.finish_export(model, path / 'motion_003.bin')
            torch.testing.assert_close(decoded3.xyz, final_xyz, rtol=0, atol=.0009)
            torch.testing.assert_close(decoded3.rotation, final_q, rtol=0, atol=.0009)
            history = model.online_history.records[-1]
            self.assertEqual(history['frame_index'], 3)
            self.assertEqual(len(history['ids']), 65)
            self.native_json(model.flat_region_diagnostics)
            model.setup(decoded3.xyz, decoded3.rotation, 1., frame_index=4)
            self.assertEqual(model.persistent_state.region_ids.tolist(), [0, 1, 2, 3])
            self.assertEqual(model.persistent_state.next_region_id, 4)
            self.assertEqual(model.frame_reference_state.digest(), decoded3.digest())
            self.assertEqual(frame0.digest(), frame0_digest)

    def test_missing_current_observation_holds_history_and_no_region_creation(self):
        model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            decoded = self.lifecycle.finish_export(model, path / 'motion_001.bin')
            self.next_motion(model, decoded, 2)
            decoded = self.lifecycle.finish_export(model, path / 'motion_002.bin')
            self.next_motion(model, decoded, 3, missing=True)
            self.assertEqual(model.flat_region_diagnostics['root_discovery']['accepted_regions'], [])
            self.assertEqual(model.flat_region_diagnostics['root_discovery']['proposal']['current_moving_root_gs'], 0)
            self.assertEqual(model.persistent_state.next_region_id, 3)
            decoded = self.lifecycle.finish_export(model, path / 'motion_003.bin')
            self.next_motion(model, decoded, 4)
            self.assertEqual(model.flat_region_diagnostics['root_discovery']['accepted_regions'],
                             [dict(region_id=3, gs_count=32)])
            self.native_json(model.flat_region_diagnostics)

    def test_invalid_whole_adjustment_rolls_back_ids_owners_and_adoption_logs(self):
        model = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            decoded = self.lifecycle.finish_export(model, path / 'motion_001.bin')
            self.next_motion(model, decoded, 2)
            decoded = self.lifecycle.finish_export(model, path / 'motion_002.bin')
            original_score = model._online_packet_score
            injected = []

            def fail_completed_adjustment(*args, **kwargs):
                score = original_score(*args, **kwargs)
                discovery = model.persistent_state.last_discovery_result or {}
                if (model.frame_index == 3 and discovery.get('accepted_regions') and not injected):
                    # Actual proposal, leaf fit and its individual decode gate have
                    # already passed. Only the final complete-adjustment guard fails.
                    injected.append(True)
                    score = {**score, 'valid': False}
                return score

            old_ids = model.persistent_state.region_ids.clone()
            next_id = model.persistent_state.next_region_id
            with patch.object(model, '_online_packet_score', side_effect=fail_completed_adjustment):
                inherited, _ = self.next_motion(model, decoded, 3)
            self.assertEqual(injected, [True])
            self.assertTrue(torch.equal(model._persistent_owner_ids(), inherited))
            self.assertTrue(torch.equal(model.persistent_state.region_ids, old_ids))
            self.assertEqual(model.persistent_state.next_region_id, next_id)
            gate = model.flat_region_diagnostics['gate']
            self.assertFalse(gate['accepted'])
            self.assertFalse(gate['failed_candidate']['valid'])
            self.assertEqual(gate['decision_reason'], 'whole_adjustment_decoded_pose_invalid')
            discovery = model.flat_region_diagnostics['root_discovery']
            self.assertTrue(discovery['rolled_back'])
            self.assertEqual(discovery['accepted_regions'], [])
            self.assertEqual(discovery['proposed_accepted_regions'], [dict(region_id=3, gs_count=32)])
            self.assertFalse(discovery['trials'][0]['accepted'])
            self.assertTrue(discovery['trials'][0]['proposed_accepted'])
            self.assertEqual(model.flat_region_diagnostics['persistent']['structural_edits'], [])
            self.assertEqual(model.persistent_state.last_discovery_frame, 3)
            self.assertEqual(model.persistent_state.last_discovery_result, discovery)
            self.native_json(model.flat_region_diagnostics)
            self.lifecycle.finish_export(model, path / 'motion_003.bin')


if __name__ == '__main__':
    unittest.main()
