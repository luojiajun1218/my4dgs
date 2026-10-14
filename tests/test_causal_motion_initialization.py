"""Real CPU model/codec regression for causal translation initialization."""

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_persistent_motion_integration as lifecycle_tests


class CausalMotionInitializationTests(unittest.TestCase):
    def setUp(self):
        self.lifecycle = lifecycle_tests.PersistentLifecycleTests()
        self.rigid, _, _, _, self.codec = self.lifecycle.modules()

    def history_fixture(self, enabled=True, reverse=True):
        """Generate history through two real topology/codec transitions, with births."""
        template = self.lifecycle.fixture()
        config = replace(template.config, causal_translation_init=enabled)
        config.online_clustering = replace(config.online_clustering,
                                            static_translation_extent=.004)
        model = self.rigid.RigidResidualTree(config)
        model.setup(template.base_xyz, template.base_rotation, 1., frame_index=1)
        self.assertEqual(torch.count_nonzero(model.gs_translation), 0)
        with tempfile.TemporaryDirectory() as directory:
            for frame in (1, 2):
                model.observations.fill_(10)
                if frame == 2:
                    model.observations[1] = 0
                with torch.no_grad():
                    model.gs_translation.zero_()
                    model.gs_translation[:16, 0] = .02
                    model.gs_translation[16:24, 1] = .03
                model.forward(100, training=True)
                model.update_topology(100)
                decoded = self.lifecycle.finish_export(
                    model, Path(directory) / f'motion_{frame:03}.bin', birth_source=0)
                if frame == 1:
                    model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
        saved_history = copy.deepcopy(model.online_history.state_dict())
        reference_digest = model.codec_previous_state.digest()
        rows = torch.arange(len(decoded.xyz))
        if reverse:
            rows = rows.flip(0)
        model.setup(decoded.xyz[rows], decoded.rotation[rows], 1., frame_index=3,
                    previous_rows=rows)
        return model, saved_history, reference_digest

    def assert_history_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            if key == 'records':
                self.assertEqual(len(left[key]), len(right[key]))
                for a, b in zip(left[key], right[key]):
                    self.assertEqual(a.keys(), b.keys())
                    for field in a:
                        np.testing.assert_array_equal(a[field], b[field])
            else:
                np.testing.assert_array_equal(left[key], right[key])

    def test_actual_decoded_history_reordered_ids_birth_missing_and_fresh_adam(self):
        model, history, digest = self.history_fixture()
        self.assert_history_equal(model.online_history.state_dict(), history)
        self.assertEqual(model.frame_reference_state.digest(), digest)
        self.assertEqual(model.codec_previous_state.digest(), digest)
        self.assertEqual(torch.count_nonzero(model.observations), 0)
        self.assertEqual(len(model.optimizer.state), 0)
        for name in ('region_translation', 'region_rotation', 'gs_rotation'):
            self.assertEqual(torch.count_nonzero(getattr(model, name)), 0, name)
        latest = model.online_history.records[-1]
        locations = {int(identifier): row for row, identifier in enumerate(latest['ids'])}
        predicted_ids = set(range(24)) - {1}
        for row, identifier in enumerate(model.online_ids.tolist()):
            actual = model.gs_translation[row].detach().numpy()
            if identifier in predicted_ids:
                index = locations[identifier]
                expected = latest['target'][index] - latest['source'][index]
                np.testing.assert_allclose(actual, expected, atol=1e-8)
            else:
                np.testing.assert_array_equal(actual, np.zeros(3))
        torch.testing.assert_close(model.base_xyz,
            model.frame_reference_state.xyz[model.previous_rows], rtol=0, atol=0)
        torch.testing.assert_close(model.base_rotation,
            model.frame_reference_state.rotation[model.previous_rows], rtol=0, atol=0)
        restored = self.rigid.RigidResidualTree(model.config)
        restored.restore(copy.deepcopy(model.capture()))
        torch.testing.assert_close(restored.gs_translation, model.gs_translation, rtol=0, atol=0)
        self.assert_history_equal(restored.online_history.state_dict(), history)
        self.assertEqual(model.causal_predicted_ids.tolist(), sorted(predicted_ids))
        for index, identifier in enumerate(model.causal_predicted_ids.tolist()):
            row = model.online_ids.tolist().index(identifier)
            torch.testing.assert_close(model.causal_initial_translation[index],
                                       model.gs_translation[row], rtol=0, atol=0)
        self.assertEqual(model.causal_initial_translation.shape, (23, 3))

    def test_disabled_preserves_zero_initialization_even_with_usable_history(self):
        model, history, digest = self.history_fixture(enabled=False)
        self.assertEqual(torch.count_nonzero(model.gs_translation), 0)
        self.assertEqual(len(model.optimizer.state), 0)
        self.assertEqual(model.frame_reference_state.digest(), digest)
        self.assert_history_equal(model.online_history.state_dict(), history)

    def test_fallback_is_after_adam_and_before_topology_and_clears_only_pending_rows(self):
        model, history, _ = self.history_fixture()
        missing = model.online_ids.tolist().index(0)
        seen = model.online_ids.tolist().index(2)
        unsupported = model.online_ids.tolist().index(33)
        model.observations.fill_(10)
        model.observations[[missing, unsupported]] = 0
        model.last_step = 99
        model.gs_translation.grad = torch.ones_like(model.gs_translation)
        model.optimizer_step()
        self.assertFalse(model.causal_prediction_fallback_completed)
        state = model.optimizer.state[model.gs_translation]
        state['max_exp_avg_sq'] = torch.ones_like(model.gs_translation)
        scalar_step = state['step'].clone()
        before = model.gs_translation.detach().clone()
        calls = []

        def inspect_partition_boundary(step):
            calls.append(step)
            self.assertTrue(model.causal_prediction_fallback_completed)
            torch.testing.assert_close(model.gs_translation[missing], torch.zeros(3))
            self.assertFalse(torch.equal(model.gs_translation[seen], before[seen]),
                             'Observed prediction must still receive the Adam update')
            self.assertFalse(torch.equal(model.gs_translation[unsupported], torch.zeros(3)),
                             'Unpredicted missing rows must not be erased by prediction fallback')
            for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                torch.testing.assert_close(state[name][missing], torch.zeros(3))
                self.assertTrue(bool(state[name][seen].gt(0).all()))
                self.assertTrue(bool(state[name][unsupported].gt(0).all()))
            torch.testing.assert_close(state['step'], scalar_step + 1)
            return []

        model.last_step = 100
        model.gs_translation.grad = torch.ones_like(model.gs_translation)
        with patch.object(model, 'update_topology', side_effect=inspect_partition_boundary):
            model.optimizer_step()
        self.assertEqual(calls, [100])
        self.assert_history_equal(model.online_history.state_dict(), history)
        self.assertEqual(model.causal_prediction_metrics['unobserved_fallback_rows'], 1)
        self.assertEqual(model.causal_prediction_metrics['observed_prediction_support'], 22)
        saved = model.gs_translation.detach().clone()
        model._fallback_unobserved_translation_prediction()
        torch.testing.assert_close(model.gs_translation, saved, rtol=0, atol=0)

    @staticmethod
    def train_step(model, step):
        model.observations.fill_(10)
        model.observations[model.online_ids == 0] = 0
        xyz, rotation = model.forward(step, training=True)
        desired = model.base_xyz.clone()
        desired[:, 0] += .015
        loss = (xyz - desired).square().mean() + (rotation - model.base_rotation).square().mean()
        loss.backward()
        model.optimizer_step()

    def test_midframe_99_100_150_resume_does_not_reinitialize_or_repeat_fallback(self):
        # The codec requires monotonically ordered survivor rows; arbitrary
        # stable-ID alignment is separately exercised at initialization/fallback.
        model, history, _ = self.history_fixture(reverse=False)
        branches = []
        for step in range(1, 201):
            self.train_step(model, step)
            for branch in branches:
                self.train_step(branch, step)
            if step in (99, 100, 150):
                saved = copy.deepcopy(model.capture())
                restored = self.rigid.RigidResidualTree(model.config)
                with patch.object(restored, 'setup', side_effect=AssertionError('restore called setup')), \
                     patch.object(restored, '_initialize_causal_translation_prediction',
                                  side_effect=AssertionError('restore reinitialized prediction')):
                    restored.restore(saved)
                self.assertEqual(restored.causal_prediction_fallback_completed, step >= 100)
                self.assertEqual(restored.causal_prediction_frame, 3)
                torch.testing.assert_close(restored.causal_initial_translation,
                                           model.causal_initial_translation, rtol=0, atol=0)
                branches.append(restored)
        model.finalize_online_refinement()
        for branch in branches:
            branch.finalize_online_refinement()
            for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
                torch.testing.assert_close(getattr(branch, name), getattr(model, name), rtol=0, atol=0)
            self.assertEqual(branch.causal_prediction_metrics, model.causal_prediction_metrics)
            self.assertEqual(branch.causal_prediction_metrics['unobserved_fallback_rows'], 1)
            self.assertEqual(set(branch.causal_prediction_metrics['corrections']), {'100', '200'})
            self.assert_history_equal(branch.online_history.state_dict(), history)

    def test_prediction_then_real_topology_export_decodes_pose_and_updates_history_once(self):
        model, history, digest = self.history_fixture(reverse=False)
        model.observations.fill_(10)
        model.observations[model.online_ids == 0] = 0
        model.forward(100, training=True)
        model.optimizer_step()
        self.assert_history_equal(model.online_history.state_dict(), history)
        self.assertEqual(model.frame_reference_state.digest(), digest)
        target = tuple(v.detach().clone() for v in model.materialized_pose(200))
        with tempfile.TemporaryDirectory() as directory:
            decoded = self.lifecycle.finish_export(model, Path(directory) / 'motion_003.bin')
        torch.testing.assert_close(decoded.xyz, target[0], rtol=0, atol=.0009)
        torch.testing.assert_close(decoded.rotation, target[1], rtol=0, atol=.0009)
        self.assertEqual([r['frame_index'] for r in model.online_history.records], [1, 2, 3])
        last = model.online_history.records[-1]
        index = last['ids'].tolist().index(0)
        self.assertFalse(last['observed'][index])
        self.assertEqual(model.frame_reference_state.digest(), digest)

    def test_first_frame_and_config_compatibility(self):
        template = self.lifecycle.fixture()
        model = self.rigid.RigidResidualTree(replace(template.config, causal_translation_init=True))
        model.setup(template.base_xyz, template.base_rotation, 1., frame_index=1)
        self.assertEqual(model.causal_predicted_ids.numel(), 0)
        self.assertEqual(model.causal_prediction_metrics['rejection_reason'], 'first_dynamic_frame')
        self.assertEqual(torch.count_nonzero(model.gs_translation), 0)
        for changes in ({'motion_warm_start': True}, {'online_flat_clustering': False}):
            with self.assertRaises(ValueError):
                replace(model.config, **changes).validate()
        legacy = copy.deepcopy(template.capture())
        legacy['config'].pop('causal_translation_init', None)
        legacy.pop('causal_translation_prediction', None)
        restored = self.rigid.RigidResidualTree(template.config)
        restored.restore(legacy)
        self.assertEqual(restored.causal_predicted_ids.numel(), 0)
        self.assertEqual(torch.count_nonzero(restored.gs_translation), 0)

    def test_enabled_checkpoint_rejects_missing_or_inconsistent_pending_state(self):
        model, _, _ = self.history_fixture(reverse=False)
        for step in range(1, 151):
            self.train_step(model, step)
            if step not in (99, 100, 150):
                continue
            packet = copy.deepcopy(model.capture())
            corruptions = ['missing', 'wrong_frame', 'non_bool', 'wrong_phase']
            for corruption in corruptions:
                damaged = copy.deepcopy(packet)
                state = damaged['causal_translation_prediction']
                if corruption == 'missing':
                    damaged.pop('causal_translation_prediction')
                elif corruption == 'wrong_frame':
                    state['frame'] += 1
                elif corruption == 'non_bool':
                    state['fallback_completed'] = int(state['fallback_completed'])
                else:
                    state['fallback_completed'] = step < 100
                with self.subTest(step=step, corruption=corruption):
                    restored = self.rigid.RigidResidualTree(model.config)
                    with self.assertRaises(ValueError):
                        restored.restore(damaged)


if __name__ == '__main__':
    unittest.main()
