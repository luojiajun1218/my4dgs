"""Learn additive hierarchy from images without losing the parent prediction."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

import test_fixed_gate_residuals as fixtures
import test_joint_spatial_motion as lineage
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import encode_motion, decode_motion
from pipeline.model.sparse_features import (FeatureDCUpdateConfig, QuantizedFeatureDCUpdate,
                                          decode_feature_dc_delta)


class InheritedImageGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, refit=0):
        template = fixtures.FixedGateResidualTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.region_motion = 'relative'
        cfg.amplitude_refit_steps = refit
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def sample_depth2(model, step=1):
        model._sampled_gate_step = step
        model._sampled_gate_noise = torch.full_like(model.gate_logits, -20.)
        model._sampled_gate_noise[1:9] = 20.

    @staticmethod
    def packet(model):
        rt, rr, gt, gr = model._values(hard=True)
        return encode_motion(model.frame_reference_state, model.frame_index, model.extent,
            model.previous_rows, model.owner, model.parents, rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            model.config.motion_xyz_step_extent, model.config.motion_rotation_step_rad,
            gs_translation_step_extent=model.config.gs_xyz_step_extent,
            gs_rotation_step_rad=model.config.gs_rotation_step_rad,
            spatial_open=model.opened, spatial_relative=True)

    def colors(self, model):
        update = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True,
            sharing='spatial_region', gate_joint=True, l1_weight=0.), motion=model)
        reference = torch.zeros(len(model.owner), 1, 3)
        reference[0, 0, 0] = -0.
        update.setup(reference)
        return update, reference

    def test_configuration_limits_relative_images_to_the_supported_residual_stage(self):
        cfg = self.model().config
        for name, value in (('reparameterize_after_gate_freeze', False),
                            ('image_relaxation', 'soft_anneal'), ('private_full_motion', False)):
            invalid = copy.deepcopy(cfg)
            setattr(invalid.spatial_gate, name, value)
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_opening_zero_child_preserves_parent_and_image_gradients_fit_both_levels(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gs_translation[:, 0] = .006
        before = model(1)[0].detach()
        self.sample_depth2(model)
        xyz, _ = model(1, training=True)
        torch.testing.assert_close(xyz, before, rtol=0, atol=0)
        gradient = torch.autograd.grad((xyz - model.base_xyz - .02).square().mean(),
                                       model.region_translation)[0]
        self.assertGreater(float(gradient[1:9].abs().sum()), 0.)
        self.assertGreater(float(gradient[9:73].abs().sum()), 0.)
        self.assertEqual(float(gradient[73:].abs().sum()), 0.)
        self.assertEqual(float(gradient[0].abs().sum()), 0.)
        with torch.no_grad():
            model.region_translation[0] = 10.
        torch.testing.assert_close(model(1, training=True)[0], before, rtol=0, atol=0)

    def test_nonzero_child_correction_trains_gate_from_image_error(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[9:73, 0] = .02
            model.gate_logits.fill_(-.2)
        model._sampled_gate_step = 1
        model._sampled_gate_noise = torch.full_like(model.gate_logits, -.1)
        xyz, _ = model(1, training=True)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        grad = torch.autograd.grad((xyz - target).square().mean(), model.gate_logits)[0]
        self.assertTrue(bool(grad[1:9].lt(0).all()))
        self.assertEqual(float(model.loss_terms(1, False)['candidate_motion_loss']), 0.)

    def test_rate_prices_ancestors_and_freezing_preserves_exact_relative_stream(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.region_translation[9:73, 0] = .004
            model.gate_logits[model.split_eligible] = 2.
            model.gs_translation.copy_(model._regional_predictions()[0][model.spatial_paths[:, -1]])
        model._select_frontier(freeze=False)
        terms = model.loss_terms(20)
        with torch.no_grad():
            model.region_translation[1:9] = 0
            model.gs_translation.copy_(model._regional_predictions()[0][model.spatial_paths[:, -1]])
        without_parents = model.loss_terms(20)
        # Eight transmitted T blocks: six value bytes and a one-byte ID plus
        # flag each. The existing raw proxy is not exact compressed length.
        self.assertEqual(float(terms['rate_bytes_surrogate'] - without_parents['rate_bytes_surrogate']), 64.)
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gs_translation.copy_(model._regional_predictions()[0][model.spatial_paths[:, -1]])
        before = self.packet(model)
        pose = model(20)[0].detach()
        model.last_step = 70
        model._select_frontier(freeze=True)
        self.assertFalse(model.private_full_motion)
        self.assertEqual(before, self.packet(model))
        decoded = decode_motion(before, model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, pose, rtol=0, atol=0)
        torch.testing.assert_close(model(71)[0], pose, rtol=0, atol=0)

    def test_inherited_color_reuses_motion_path_and_collapses_to_existing_codec(self):
        model = self.model()
        update, reference = self.colors(model)
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[1:9] = .02
        self.sample_depth2(model)
        model(1, training=True)
        rng = model._gate_rng.get_state().clone()
        color = update(training=True)
        torch.testing.assert_close(color, reference + .02, rtol=0, atol=0)
        color.sum().backward()
        gradient = update.scalar.delta.grad.reshape(-1, 3)
        self.assertGreater(float(gradient[1:9].abs().sum()), 0.)
        self.assertGreater(float(gradient[9:73].abs().sum()), 0.)
        self.assertEqual(float(gradient[0].abs().sum()), 0.)
        self.assertTrue(torch.equal(rng, model._gate_rng.get_state()))
        with torch.no_grad():
            model.gate_logits[1:9] = 2.
        model.last_step = 70
        model._select_frontier(freeze=True)
        before = update.canonical_features().clone()
        update.optimizer_step(0., lr_factor=0.)
        self.assertFalse(update._relative_active)
        torch.testing.assert_close(before, update.canonical_features(), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'dc.bin'
            info = update.export(path, 1)
            self.assertEqual(info['format'], 'SDR1')
            decoded = decode_feature_dc_delta(path.read_bytes(), reference, frame_index=1,
                owner=model.owner, node_count=len(model.parents))
            torch.testing.assert_close(decoded, before, rtol=0, atol=0)

    def test_restore_keeps_inherited_colors_and_next_motion_update(self):
        model = self.model()
        update, _ = self.colors(model)
        xyz, _ = model(20, training=True)
        (xyz.square().mean() + update(training=True).square().mean()).backward()
        model.optimizer_step()
        update.optimizer_step(0.)
        saved_motion, saved_color = copy.deepcopy(model.capture()), copy.deepcopy(update.capture())
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(saved_motion)
        other, _ = self.colors(restored)
        other.restore(saved_color, torch.device('cpu'))
        for m, u in ((model, update), (restored, other)):
            xyz, _ = m(21, training=True)
            (xyz.square().mean() + u(training=True).square().mean() + m.regularization_loss(21)).backward()
            m.optimizer_step()
            u.optimizer_step(0.)
        self.assertTrue(torch.equal(model._sampled_gate_noise, restored._sampled_gate_noise))
        torch.testing.assert_close(update.scalar.delta, other.scalar.delta, rtol=0, atol=0)
        for key in ('region_translation', 'gs_translation', 'gate_logits'):
            torch.testing.assert_close(getattr(model, key), getattr(restored, key), rtol=0, atol=0)
        self.assertEqual(self.packet(model), self.packet(restored))

    def test_absolute_migration_preserves_applied_motion_and_unused_children_inherit(self):
        model = fixtures.FixedGateResidualTests().model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
        model.last_step = 70
        model._select_frontier(freeze=True)
        full = tuple(x.clone() for x in model._full_motion_symbols())
        state = lineage.JointSpatialMotionTests().commit(model)
        model.enable_relative_regions(inherit_unused=True)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(full, model._full_motion_symbols())))
        self.assertFalse(bool(model.region_translation[9:].round(decimals=4).ne(0).any()))
        model.setup(state.xyz, state.rotation, model.extent, frame_index=2)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(full, model._full_motion_symbols())))

    def test_amplitude_refit_keeps_parent_support_and_relative_replay(self):
        model = self.model(refit=20)
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gate_logits[1:9] = 2.
        model.last_step = 70
        model._select_frontier(freeze=True)
        model(81, training=True)
        support = model._amplitude_refit_support['region_translation']
        self.assertTrue(bool(support[1:9].all()))
        self.assertFalse(bool(support[9:].any()))
        xyz, _ = model(81, training=True)
        (xyz - model.base_xyz - .02).square().mean().backward()
        model.optimizer_step()
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(82)[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
