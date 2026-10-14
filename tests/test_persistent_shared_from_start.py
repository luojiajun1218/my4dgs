"""CPU regressions for inherited sharing before current-frame structure edits.

These tests exercise the actual motion model and codec, without a renderer.
The heldout-stage recorder test supplies a CPU render stub only.
"""

import copy
import importlib
import random
from dataclasses import replace
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_persistent_motion_integration as lifecycle_tests


class PersistentSharedFromStartTests(unittest.TestCase):
    def setUp(self):
        self.lifecycle = lifecycle_tests.PersistentLifecycleTests()
        self.rigid, _, _, _, self.codec = self.lifecycle.modules()

    def first_frame(self, enabled=True):
        template = self.lifecycle.fixture()
        config = replace(template.config, persistent_shared_from_start=enabled)
        model = self.rigid.RigidResidualTree(config)
        model.setup(template.base_xyz, template.base_rotation, 1., frame_index=1)
        model.observations.fill_(10)
        return model

    def inherited_frame(self, enabled=True):
        model = self.first_frame(enabled)
        self.lifecycle.initialize(model)
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.lifecycle.finish_export(model, Path(directory) / 'motion_001.bin')
        model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
        return model

    @staticmethod
    def backward_image_proxy(model, step):
        """A pose-dependent image proxy with shared and independent gradients."""
        xyz, q = model.forward(step, training=True)
        desired = model.base_xyz.clone()
        desired[:16, 0] += .012
        desired[16:24, 1] += .009
        desired[24:, 2] += .003
        # A small independent mismatch ensures the GS correction is useful.
        desired[0, 1] += .005
        qtarget = model.base_rotation.clone()
        angle = q.new_tensor([0., 0., .012])
        rotation_q = importlib.import_module(
            'pipeline.ReCon-GS.model.rigid_residual_tree').rotation_vector_quaternion(angle)
        qtarget[:24] = rotation_q
        loss = (xyz - desired).square().mean() + (q - qtarget).square().mean()
        loss.backward()

    def assert_optimizer_owns_current_parameters(self, model):
        parameters = [p for group in model.optimizer.param_groups for p in group['params']]
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
            self.assertTrue(any(getattr(model, name) is p for p in parameters), name)

    def test_inherited_regions_and_gs_update_from_step_one_with_zero_increment_root(self):
        model = self.inherited_frame()
        self.assertTrue(model.shared_prestructure_active)
        self.assertEqual(len(model.optimizer.state), 0)
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
            self.assertEqual(torch.count_nonzero(getattr(model, name)), 0, name)
        reference_digest = model.frame_reference_state.digest()
        self.backward_image_proxy(model, 1)
        for name in ('region_translation', 'region_rotation'):
            gradient = getattr(model, name).grad
            self.assertGreater(float(gradient[1:].abs().sum()), 0., name)
            torch.testing.assert_close(gradient[0], torch.zeros(3), rtol=0, atol=0)
        for name in ('gs_translation', 'gs_rotation'):
            self.assertGreater(float(getattr(model, name).grad.abs().sum()), 0., name)
        model.optimizer_step()
        for name in ('region_translation', 'region_rotation'):
            self.assertGreater(float(getattr(model, name)[1:].abs().sum()), 0., name)
        for name in ('gs_translation', 'gs_rotation'):
            self.assertGreater(float(getattr(model, name).abs().sum()), 0., name)
        rt, rr, _, _ = model._values(hard=True)
        torch.testing.assert_close(rt[0], torch.zeros(3, dtype=rt.dtype), rtol=0, atol=0)
        torch.testing.assert_close(rr[0], torch.zeros(3, dtype=rr.dtype), rtol=0, atol=0)
        self.assertEqual(model.frame_reference_state.digest(), reference_digest)
        self.assertEqual(int(model.observations.sum()), 0, 'Enabling sharing must not fabricate observations')
        self.assert_optimizer_owns_current_parameters(model)

    def test_first_discovery_does_not_retroactively_enable_the_prestructure_phase(self):
        model = self.first_frame()
        self.assertFalse(model.shared_prestructure_active)
        self.backward_image_proxy(model, 1)
        model.optimizer_step()
        self.assertEqual(torch.count_nonzero(model.region_translation), 0)
        self.assertGreater(float(model.gs_translation.abs().sum()), 0.)
        with torch.no_grad():
            model.gs_translation.zero_()
            model.gs_rotation.zero_()
        self.lifecycle.initialize(model)
        self.assertTrue(model.persistent_state.initialized)
        self.assertTrue(bool(model.owner.ne(0).any()))
        self.assertFalse(model.shared_prestructure_active,
                         'Current step100 layout cannot stand in for the immutable reference layout')
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.lifecycle.finish_export(model, Path(directory) / 'motion_001.bin')
        model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
        self.assertTrue(model.shared_prestructure_active)
        self.assertEqual(torch.count_nonzero(model.region_translation), 0)
        self.assertEqual(torch.count_nonzero(model.region_rotation), 0)

    def test_delayed_first_discovery_remains_free_until_next_decoded_frame(self):
        model = self.first_frame()
        with tempfile.TemporaryDirectory() as directory:
            for frame in (1, 2):
                self.assertFalse(model.shared_prestructure_active)
                model.observations.fill_(10)
                model.forward(100, training=True)
                model.update_topology(100)
                self.assertFalse(model.persistent_state.initialized)
                self.assertFalse(bool(model.owner.ne(0).any()))
                decoded = self.lifecycle.finish_export(
                    model, Path(directory) / f'motion_{frame:03}.bin')
                model.setup(decoded.xyz, decoded.rotation, 1., frame_index=frame + 1)
            model.observations.fill_(10)
            self.lifecycle.initialize(model)
            self.assertTrue(model.persistent_state.initialized)
            self.assertFalse(model.shared_prestructure_active)
            decoded = self.lifecycle.finish_export(model, Path(directory) / 'motion_003.bin')
            model.setup(decoded.xyz, decoded.rotation, 1., frame_index=4)
        self.assertTrue(model.shared_prestructure_active)
        self.assertEqual(model.frame_reference_state.frame_index, 3)

    def test_disabled_flag_keeps_inherited_regions_frozen_until_joint_phase(self):
        model = self.inherited_frame(enabled=False)
        self.assertFalse(model.shared_prestructure_active)
        self.backward_image_proxy(model, 1)
        self.assertGreater(float(model.region_translation.grad[1:].abs().sum()), 0.)
        model.optimizer_step()
        self.assertEqual(torch.count_nonzero(model.region_translation), 0)
        self.assertEqual(torch.count_nonzero(model.region_rotation), 0)
        self.assertGreater(float(model.gs_translation.abs().sum()), 0.)
        self.backward_image_proxy(model, 101)
        model.optimizer_step()
        self.assertGreater(float(model.region_translation[1:].abs().sum()), 0.)
        self.assertGreater(float(model.region_rotation[1:].abs().sum()), 0.)

    def test_flag_validation_and_legacy_disabled_checkpoint(self):
        model = self.first_frame()
        invalid = (
            {'online_flat_clustering': False},
            {'persistent_regions': replace(model.config.persistent_regions, enabled=False)},
            {'post_partition_steps': 0},
            {'motion_warm_start': True},
            {'shared_steps': 1},
            {'persistent_shared_from_start': 1},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(model.config, **changes).validate()
        legacy_model = self.first_frame(enabled=False)
        packet = copy.deepcopy(legacy_model.capture())
        packet['config'].pop('persistent_shared_from_start')
        restored = self.rigid.RigidResidualTree(legacy_model.config)
        restored.restore(packet)
        self.assertFalse(restored.shared_prestructure_active)
        self.assertEqual(restored.config.persistent_shared_from_start, False)

    def test_prestructure_callback_sees_updated_pose_before_real_refit_and_fresh_adam(self):
        model = self.inherited_frame()
        model.observations.fill_(10)
        calls, original_parameters = [], []
        real_update = model.update_topology

        def record_before_structure():
            calls.append('measurement')
            self.assertFalse(model._topology_updated)
            self.assertEqual(model.last_step, 100)
            self.assertGreater(float(model.region_translation[1:].abs().sum()), 0.)
            self.assertGreater(float(model.gs_translation.abs().sum()), 0.)
            self.assertEqual(int(model.optimizer.state[model.gs_translation]['step']), 1)
            original_parameters.extend([model.region_translation, model.gs_translation])
            calls.append(tuple(value.detach().clone() for value in model.forward(100)))

        def refit(step):
            self.assertEqual(calls[0], 'measurement')
            calls.append('structure')
            return real_update(step)

        self.backward_image_proxy(model, 100)
        with patch.object(model, 'update_topology', side_effect=refit):
            model.optimizer_step(pre_topology_callback=record_before_structure)
        self.assertEqual([entry for entry in calls if isinstance(entry, str)],
                         ['measurement', 'structure'])
        self.assertTrue(model._topology_updated)
        self.assertTrue(model.shared_prestructure_active)
        self.assertEqual(len(model.optimizer.state), 0, 'Structure installation must still create fresh Adam')
        self.assertIsNot(model.region_translation, original_parameters[0])
        self.assertIsNot(model.gs_translation, original_parameters[1])
        self.assert_optimizer_owns_current_parameters(model)
        for actual, expected in zip(model.forward(100), calls[1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=.001)
        for step in (99, 101):
            self.backward_image_proxy(model, step)
            model.optimizer_step(pre_topology_callback=lambda: self.fail('Callback outside step100'))

    def test_checkpoint_99_100_150_restores_reference_eligibility_and_exact_updates(self):
        model = self.inherited_frame()
        reference_digest = model.frame_reference_state.digest()
        branches = []
        for step in range(1, 201):
            for current in [model, *branches]:
                current.observations.fill_(10)
                self.backward_image_proxy(current, step)
                current.optimizer_step()
            if step in (99, 100, 150):
                restored = self.rigid.RigidResidualTree(model.config)
                with patch.object(restored, 'setup', side_effect=AssertionError('restore reinitialized frame')):
                    restored.restore(copy.deepcopy(model.capture()))
                self.assertTrue(restored.shared_prestructure_active)
                self.assertEqual(restored.frame_reference_state.digest(), reference_digest)
                self.assert_optimizer_owns_current_parameters(restored)
                branches.append(restored)
        for restored in branches:
            for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
                torch.testing.assert_close(getattr(restored, name), getattr(model, name), rtol=0, atol=0)
            self.assertTrue(torch.equal(restored.owner, model.owner))
            self.assertTrue(torch.equal(restored.pivots, model.pivots))
            self.assertEqual(restored.persistent_state.region_ids.tolist(),
                             model.persistent_state.region_ids.tolist())
        target = tuple(value.detach().clone() for value in model.materialized_pose(200))
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.lifecycle.finish_export(model, Path(directory) / 'motion_002.bin')
            for index, restored in enumerate(branches):
                other = self.lifecycle.finish_export(restored, Path(directory) / f'branch_{index}.bin')
                self.assertEqual(other.digest(), decoded.digest())
        torch.testing.assert_close(decoded.xyz, target[0], rtol=0, atol=.0009)
        torch.testing.assert_close(decoded.rotation, target[1], rtol=0, atol=.0009)
        self.assertEqual([record['frame_index'] for record in model.online_history.records], [1, 2])

    def test_prestructure_and_postpartition_stage_records_coexist_at_step100(self):
        trainer_source = importlib.import_module('pipeline.ReCon-GS.trainer')
        view = SimpleNamespace(path='cam00.png', image=torch.full((3, 2, 2), .5))

        def render(viewpoint, old_only=False):
            # The real recorder must restore CPU RNG even if rendering uses it.
            torch.rand(1)
            np.random.random()
            random.random()
            return {'render': torch.full((3, 2, 2), .6)}

        module = SimpleNamespace(config=SimpleNamespace(rigid_tree=SimpleNamespace(motion_steps=100)),
            motion_steps=200, total_steps=300, stage_image_metrics=[],
            forward=render)
        harness = SimpleNamespace(module=module, current_frame=2)
        harness._capture_rng = MethodType(trainer_source.Trainer._capture_rng, harness)
        harness._restore_rng = trainer_source.Trainer._restore_rng
        recorder = trainer_source.Trainer._record_stage_image
        with patch('torch.cuda.is_available', return_value=False):
            before = harness._capture_rng()
            recorder(harness, 100, view, phase_override='pre_structure')
            recorder(harness, 100, view)
            recorder(harness, 100, view, phase_override='pre_structure')
            after = harness._capture_rng()
        self.assertEqual({(item['step'], item['phase']) for item in module.stage_image_metrics},
                         {(100, 'pre_structure'), (100, 'post_partition')})
        self.assertEqual(len(module.stage_image_metrics), 2)
        torch.testing.assert_close(before['torch'], after['torch'], rtol=0, atol=0)
        np.testing.assert_equal(before['numpy'], after['numpy'])
        self.assertEqual(before['python'], after['python'])
        self.assertTrue(all(not item['participates_in_training_or_selection']
                            for item in module.stage_image_metrics))


if __name__ == '__main__':
    unittest.main()
