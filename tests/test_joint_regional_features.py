"""Motion and shared DC colors train the same sampled discrete hierarchy."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from pipeline.model.sparse_features import (FeatureDCUpdateConfig, QuantizedFeatureDCUpdate,
                                           decode_feature_dc_delta)
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.spatial_gate import spatial_owner
from pipeline.model.legacy_opacity_codec import HEADER
from scripts.replay_rigid_tree import decode_frame
import test_motion_training as fixtures
from test_codec import export_frame

MODULE = importlib.import_module('pipeline.module')


class JointRegionalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, joint=True):
        motion = fixtures.ProposalImageSupervisionTests().model()
        cfg = FeatureDCUpdateConfig(enabled=True, sharing='spatial_region', gate_joint=joint,
                                    l1_weight=0.)
        update = QuantizedFeatureDCUpdate(cfg, motion=motion)
        reference = torch.zeros(len(motion.owner), 1, 3)
        reference[0, 0, 0] = -0.
        update.setup(reference)
        return update, motion, reference

    @staticmethod
    def sampled_owner(motion):
        draw = motion.gate_logits.detach() / motion._gate_temperature() + motion._sampled_gate_noise
        opened = draw.ge(0) & motion.split_eligible
        opened[0] = True
        return spatial_owner(motion.spatial_paths, opened)

    @staticmethod
    def force_coarse_sample(motion, step=1):
        motion._sampled_gate_step = step
        motion._sampled_gate_noise = torch.full_like(motion.gate_logits, -.1)
        motion(step, training=True)

    def test_color_and_motion_share_one_sample_without_advancing_either_rng(self):
        update, motion, reference = self.model()
        with torch.no_grad():
            motion.region_translation[:, 0] = torch.arange(len(motion.parents)).remainder(7) * .002
            motion.gs_translation[:, 0] = .08
            table = torch.arange(len(motion.parents)).float()[:, None].expand(-1, 3) * .01
            update.scalar.delta.copy_(table.reshape(-1, 1))
        xyz, _ = motion(1, training=True)
        owner = self.sampled_owner(motion)
        state, global_state = motion._gate_rng.get_state().clone(), torch.get_rng_state().clone()
        value = update(training=True)
        second = update(training=True)
        node_delta = torch.round(table[owner] / update.scalar.step) * update.scalar.step
        expected = torch.where(node_delta[:, None].ne(0), reference + node_delta[:, None], reference)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        torch.testing.assert_close(value, second, rtol=0, atol=0)
        full = owner.eq(motion.spatial_paths[:, -1])
        expected_t = motion._regional_predictions()[0][owner, 0].detach()
        expected_t[full] = .08
        torch.testing.assert_close(xyz[:, 0], motion.base_xyz[:, 0] + expected_t, rtol=0, atol=1e-7)
        self.assertTrue(torch.equal(state, motion._gate_rng.get_state()))
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))
        value.sum().backward()
        counts = torch.bincount(owner, minlength=len(motion.parents)).float()
        torch.testing.assert_close(update.scalar.delta.grad,
            counts[:, None].expand(-1, 3).reshape(-1, 1), rtol=0, atol=0)

    def test_color_image_error_trains_gate_even_with_identical_motion_experts(self):
        update, motion, reference = self.model()
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[9:] = .03
        self.force_coarse_sample(motion)
        value = update(training=True)
        gradients = torch.autograd.grad((value - (reference + .03)).square().mean(),
            [motion.gate_logits, motion.gs_translation, motion.region_translation], allow_unused=True)
        self.assertTrue(bool(gradients[0][1:9].lt(0).all()))
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])

    def test_color_rate_prices_selected_rows_and_trains_gates_without_color_adam_gradients(self):
        update, motion, _ = self.model()
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[9:] = .02
        self.force_coarse_sample(motion)
        terms = update.rate_terms(.1)
        self.assertEqual(float(terms['raw_bytes_surrogate']), 0.)
        self.assertEqual(float(terms['active_rows']), 0.)
        gate, color = torch.autograd.grad(terms['loss'],
            [motion.gate_logits, update.scalar.delta], allow_unused=True)
        self.assertTrue(bool(gate[1:9].gt(0).all()))
        self.assertIsNone(color)
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[1:9] = .01
        terms = update.rate_terms(.1)
        self.assertEqual(float(terms['active_rows']), 24.)
        self.assertEqual(float(terms['raw_bytes_surrogate']), 24 * 3 + HEADER.size + 36)

    def test_attributes_cannot_draw_an_independent_sample_and_zero_bits_keep_gradients(self):
        update, motion, reference = self.model()
        state = motion._gate_rng.get_state().clone()
        with self.assertRaises(ValueError):
            update(training=True)
        self.assertTrue(torch.equal(state, motion._gate_rng.get_state()))
        motion(1, training=True)
        with torch.no_grad():
            update.scalar.delta.fill_(.004)
        value = update(training=True)
        torch.testing.assert_close(value.detach().view(torch.int32), reference.view(torch.int32), rtol=0, atol=0)
        value.sum().backward()
        self.assertGreater(float(update.scalar.delta.grad.abs().sum()), 0.)

    def test_freeze_masks_unused_parameters_moments_and_stream_rows(self):
        update, motion, reference = self.model()
        with torch.no_grad():
            update.scalar.delta.fill_(.02)
        motion(1, training=True)
        update(training=True).sum().backward()
        update.optimizer_step(0.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shdc_001.bin'
            self.assertIsNone(update.export(path, 1))
            self.assertFalse(path.exists())
            motion._select_frontier(freeze=True)
            before = update().detach().clone()
            update.scalar.delta.grad = torch.ones_like(update.scalar.delta)
            update.optimizer_step(0., lr_factor=0.)
            used = torch.bincount(motion.owner, minlength=len(motion.parents)).gt(0)
            self.assertTrue(bool(update.scalar.delta.reshape(-1, 3)[~used].eq(0).all()))
            for key in ('exp_avg', 'exp_avg_sq'):
                value = update.scalar.optimizer.state[update.scalar.delta][key]
                self.assertTrue(bool(value.reshape(-1, 3)[~used].eq(0).all()))
            torch.testing.assert_close(update(), before, rtol=0, atol=0)
            info = update.export(path, 1)
            self.assertEqual(info['rows'], int(used.sum()) * 3)
            decoded = decode_feature_dc_delta(path.read_bytes(), reference, frame_index=1,
                owner=motion.owner, node_count=len(motion.parents))
            torch.testing.assert_close(decoded, update.canonical_features(), rtol=0, atol=0)

    def test_restore_reproduces_sampling_color_gradients_and_next_optimizer_update(self):
        for frozen in (False, True):
            update, motion, reference = self.model()
            with torch.no_grad():
                update.scalar.delta.fill_(.02)
            motion(20, training=True)
            update(training=True).square().mean().backward()
            motion.optimizer_step()
            update.optimizer_step(0.)
            if frozen:
                motion.last_step = 70
                motion._select_frontier(freeze=True)
            restored_motion = SpatialGatedMotion(copy.deepcopy(motion.config))
            restored_motion.restore(copy.deepcopy(motion.capture()))
            restored = QuantizedFeatureDCUpdate(copy.deepcopy(update.config), motion=restored_motion)
            restored.restore(copy.deepcopy(update.capture()), torch.device('cpu'))
            rendered = []
            for color, current in ((update, motion), (restored, restored_motion)):
                current(71 if frozen else 21, training=True)
                value = color(training=True)
                rendered.append(value.detach().clone())
                (value - reference - .04).square().mean().backward()
                current.optimizer_step()
                color.optimizer_step(.001, proximal=True)
            torch.testing.assert_close(*rendered, rtol=0, atol=0)
            torch.testing.assert_close(update.scalar.delta, restored.scalar.delta, rtol=0, atol=0)
            torch.testing.assert_close(motion.gate_logits, restored_motion.gate_logits, rtol=0, atol=0)
            self.assertTrue(torch.equal(motion._gate_rng.get_state(), restored_motion._gate_rng.get_state()))

    def test_joint_colors_replay_after_retirement_and_births(self):
        update, motion, reference = self.model()
        preceding = copy.deepcopy(motion.frame_reference_state)
        raw = dict(xyz=motion.base_xyz.clone(), rotation=motion.base_rotation.clone(),
            logscale=torch.zeros(len(reference), 3), opacity=torch.zeros(len(reference), 1),
            sh_dc=reference.clone(), sh_rest=torch.zeros(len(reference), 3, 3))
        motion(1, training=True)
        with torch.no_grad():
            update.scalar.delta.fill_(.02)
        update(training=True).sum().backward()
        update.optimizer_step(0.)
        motion._select_frontier(freeze=True)
        rows = torch.arange(1, len(reference))
        motion.retain_survivors(rows)
        update.commit(rows)
        surviving = {name: value[rows].clone() for name, value in raw.items()}
        surviving['sh_dc'] = update.canonical_features()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            expected = export_frame(directory, motion, surviving, torch.tensor([0, 2]))
            update.export(directory / 'shdc_001.bin', 1)
            (directory / 'frame_001.json').write_text(json.dumps(dict(
                feature_dc_update='shdc_001.bin', feature_dc_update_timing='before_births')))
            _, decoded = decode_frame(directory, 1, preceding, raw)
            for name in raw:
                torch.testing.assert_close(decoded[name], expected[name], rtol=0, atol=0)

    def test_module_passes_joint_mode_and_auxiliary_proposals_detach_appearance(self):
        update, motion, _ = self.model()
        module, _ = fixtures.ProposalImageSupervisionTests().module(motion)
        module.feature_dc_update = update
        module.config.feature_dc_update = update.config
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[9:] = .02

        class Renderer:
            gaussians = module.gaussians
            collect_contributions = True

            def __call__(self, view, background):
                gs = self.gaussians
                value = gs.rigid_xyz.mean() + gs.rigid_alpha + gs.rigid_sh_dc.mean()
                return dict(render=value.expand(3, 20, 20))

        module.renderer = Renderer()
        view = SimpleNamespace(image=torch.full((3, 20, 20), .6))
        loss, _, _ = MODULE.Module.training_step(module, view, 1, regularization_weight=0.)
        main_image = module.forward(view, training=True)['render']
        main_loss = MODULE.motion_photometric_loss(main_image, view.image, .2)
        actual = torch.autograd.grad(loss, [update.scalar.delta, motion.gate_logits])
        expected = torch.autograd.grad(main_loss, [update.scalar.delta, motion.gate_logits])
        for gradient, wanted in zip(actual, expected):
            torch.testing.assert_close(gradient, wanted, rtol=0, atol=0)
        self.assertGreater(float(actual[0].abs().sum()), 0.)

    def test_defaults_restore_old_checkpoints_and_joint_mode_requires_stochastic_regions(self):
        update, motion, reference = self.model(joint=False)
        self.assertFalse(update(training=True).requires_grad)
        torch.testing.assert_close(update().view(torch.int32), reference.view(torch.int32), rtol=0, atol=0)
        packet = copy.deepcopy(update.capture())
        packet['config'].pop('gate_joint')
        restored = QuantizedFeatureDCUpdate(copy.deepcopy(update.config), motion=motion)
        restored.restore(packet, torch.device('cpu'))
        for value in (1, None, 'yes'):
            with self.assertRaises(ValueError):
                FeatureDCUpdateConfig(gate_joint=value).validate()
        with self.assertRaises(ValueError):
            FeatureDCUpdateConfig(gate_joint=True).validate()
        motion.config.spatial_gate.image_relaxation = 'hard_ste'
        joint = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True,
            sharing='spatial_region', gate_joint=True), motion=motion)
        with self.assertRaises(ValueError):
            joint.setup(reference)


if __name__ == '__main__':
    unittest.main()
