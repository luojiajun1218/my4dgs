"""Joint image gradients, post-freeze fitting, and production-byte rate."""
import copy
import unittest
from unittest.mock import patch

import torch

import test_joint_spatial_motion as joint_tests
from pipeline.model.rigid_tree_codec import encode_motion, decode_motion
from pipeline.model.spatial_gated_motion import SpatialGatedMotion


class JointImageRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_tests.JointSpatialMotionTests().model()
        model.config.spatial_gate.joint_image_gradients = True
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        model.config.rate_model = 'codec'
        model.config.codec_rate_interval = 1
        model.config.validate()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)).remainder(7) * .002
            model.region_rotation[:, 1] = torch.arange(len(model.parents)).remainder(5) * .001
            model.region_translation[0].zero_()
            model.region_rotation[0].zero_()
            model.gs_translation[:, 1] = .004
            model.gs_rotation[:, 2] = .002
        return model

    @staticmethod
    def packet(model):
        rt, rr, gt, gr = model._values(hard=True)
        return encode_motion(model.frame_reference_state, model.frame_index, model.extent,
            model.previous_rows, model.owner, model.parents, rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            model.config.motion_xyz_step_extent, model.config.motion_rotation_step_rad,
            compression='auto', gs_translation_step_extent=model.config.gs_xyz_step_extent,
            gs_rotation_step_rad=model.config.gs_rotation_step_rad, spatial_open=model.opened)

    @staticmethod
    def image_loss(model, step):
        xyz, q = model(step, training=True)
        return (xyz - model.base_xyz - .02).square().mean() + (q[:, 1:] - .01).square().mean()

    def test_image_error_reaches_region_gate_and_point_motion_without_regularizers(self):
        model = self.model()
        gradients = torch.autograd.grad(self.image_loss(model, 1),
            [model.region_translation, model.region_rotation, model.gate_logits,
             model.gs_translation, model.gs_rotation])
        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_regional_image_update_is_not_cancelled_in_point_residuals(self):
        model = self.model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.region_translation_lr = .004
        model.config.relative_motion_loss_weight = model.config.rate_loss_weight = 0.
        model.optimizer.param_groups[-1]['lr'] = 0.
        before = float(self.image_loss(model, 1))
        old_points = model.gs_translation.detach().clone()
        self.image_loss(model, 1).backward()
        model.optimizer_step()
        self.assertLess(float(self.image_loss(model, 1)), before)
        torch.testing.assert_close(model.gs_translation, old_points, rtol=0, atol=0)

    def test_hard_gate_changes_shared_field_without_hidden_point_compensation(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where(model.parents.eq(parent))[0]
        with torch.no_grad():
            model.region_translation[parent, 0] = .002
            model.region_translation[children, 0] = .01
        before = model(1)[0].detach().clone()
        before_symbols = model._full_motion_symbols()[0].detach().clone()
        point = model.gs_translation.detach().clone()
        members = model.owner.eq(parent)
        with torch.no_grad():
            model.gate_logits[parent] = 2.
        model._select_frontier(freeze=False)
        torch.testing.assert_close(model.gs_translation, point, rtol=0, atol=0)
        difference = model._full_motion_symbols()[0] - before_symbols
        self.assertTrue(bool(difference[members, 0].eq(4).all()))
        torch.testing.assert_close(model(1)[0][members, 0] - before[members, 0],
                                   before.new_full((int(members.sum()),), .008), atol=8e-6, rtol=0)

    def test_freezing_gates_keeps_regional_and_point_image_training(self):
        model = self.model()
        model._select_frontier(freeze=True)
        logits = model.gate_logits.detach().clone()
        region = model.region_translation.detach().clone()
        point = model.gs_translation.detach().clone()
        self.image_loss(model, 71).backward()
        self.assertIsNone(model.gate_logits.grad)
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)
        self.assertGreater(float(model.gs_translation.grad.abs().sum()), 0.)
        model.optimizer_step()
        torch.testing.assert_close(model.gate_logits, logits, rtol=0, atol=0)
        self.assertTrue(bool(model.region_translation.ne(region).any()))
        self.assertTrue(bool(model.gs_translation.ne(point).any()))

    def test_forward_rate_matches_production_packet_and_independent_decoder(self):
        model = self.model()
        for logit in (-2., 2.):
            with torch.no_grad():
                model.gate_logits[model.split_eligible] = logit
            model._select_frontier(freeze=False)
            terms = model.loss_terms(2)
            packet = self.packet(model)
            self.assertEqual(float(terms['rate_bytes_surrogate']), len(packet))
            rt, rr, gt, gr = model._values(hard=True)
            self.assertEqual(model._codec_rate.packet(rt, rr, gt, gr, model.opened), packet)
            decoded = decode_motion(packet, model.frame_reference_state)
            torch.testing.assert_close(decoded.xyz, model(2)[0], rtol=0, atol=1e-6)

    def test_rate_rebuilds_survivor_prefix_after_retirement(self):
        model = self.model()
        model.loss_terms(1, False)
        model.retain_survivors(torch.arange(0, len(model.owner), 2))
        terms = model.loss_terms(2, False)
        self.assertEqual(float(terms['rate_bytes_surrogate']), len(self.packet(model)))

    def test_codec_regularizer_keeps_data_only_point_adam_and_finite_gate_gradient(self):
        model = self.model()
        terms = model.loss_terms(3, False)
        gradients = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
            [model.gs_translation, model.gs_rotation, model.region_translation,
             model.region_rotation, model.gate_logits], allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        for gradient in gradients[2:]:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_closed_candidates_fit_real_images_with_a_binary_forward_cut(self):
        model = self.model()
        terms = model.loss_terms(3, False)
        self.assertFalse(terms['relative_motion_loss'].requires_grad)
        self.assertEqual(float(terms['candidate_motion_loss']), 0.)
        self.assertFalse(terms['candidate_motion_loss'].requires_grad)
        xyz, _ = model(3, training=True)
        rt, _ = model._regional_predictions()
        expected = model.base_xyz + rt[model.owner] + torch.round(
            model.gs_translation / model.gs_steps[:3]) * model.gs_steps[:3]
        torch.testing.assert_close(xyz, expected, rtol=0, atol=0)
        region, = torch.autograd.grad(self.image_loss(model, 3), [model.region_translation])
        closed = model.node_counts.gt(0) & ~model._stored_region_support()
        self.assertGreater(int(closed.sum()), 0)
        self.assertGreater(float(region[closed].abs().sum()), 0.)

    def test_byte_context_prices_match_transmitted_streams_at_varint_boundaries(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation.zero_()
            model.gs_rotation.zero_()
            rows = torch.tensor([0, 1, 129, 260, 513])
            model.gs_translation[rows] = torch.tensor([[0, 1, -1], [63, 64, -65],
                [127, 128, -129], [8191, 8192, -8193], [32767, -32767, 0.]]) * model.gs_steps[:3]
            model.gs_rotation[rows[::2], 1] = 128 * model.gs_steps[4]
        model._measure_codec_rate(1)
        rt, rr, gt, gr = model._values(hard=True)
        total = 0.
        for section, t, r in ((0, rt, rr), (1, gt, gr)):
            st, sr = t.ne(0).any(-1).float(), r.ne(0).any(-1).float()
            total += float(model._codec_rate.costs(t.float(), r.float(), st, sr, section).sum())
        self.assertAlmostEqual(total, model._codec_rate.last_entropy_bytes, places=4)

    def test_rate_calibration_reuses_model_and_diagnostics_force_real_encoding(self):
        model = self.model()
        model.config.codec_rate_interval = 20
        model.config.codec_rate_initial_steps = 0
        first = model.loss_terms(1, False)
        self.assertEqual(float(first['rate_codec_measured']), 1.)
        with patch.object(model._codec_rate, 'measure', wraps=model._codec_rate.measure) as measure:
            estimated = model.loss_terms(2, False)
            measure.assert_not_called()
            self.assertEqual(float(estimated['rate_codec_measured']), 0.)
            self.assertAlmostEqual(float(estimated['rate_bytes_surrogate']),
                                   float(first['rate_bytes_surrogate']), delta=.02)
            actual = model.loss_terms(2, True)
            measure.assert_called_once()
            self.assertEqual(float(actual['rate_codec_measured']), 1.)
            self.assertEqual(float(actual['rate_bytes_surrogate']), len(self.packet(model)))

    def test_rapid_initial_motion_uses_real_packet_lengths_before_periodic_calibration(self):
        model = self.model()
        model.config.codec_rate_interval = 20
        model.config.codec_rate_initial_steps = 20
        for step in (1, 5, 10, 19, 20):
            with torch.no_grad():
                model.gs_translation[:, 0] = step * model.gs_steps[0]
            terms = model.loss_terms(step, False)
            self.assertEqual(float(terms['rate_codec_measured']), 1.)
            self.assertEqual(float(terms['rate_bytes_surrogate']), len(self.packet(model)))
        self.assertEqual(float(model.loss_terms(21, False)['rate_codec_measured']), 0.)

    def test_rate_context_and_joint_updates_survive_resume(self):
        model = self.model()
        model.config.codec_rate_interval = 20
        model.config.codec_rate_initial_steps = 0
        (self.image_loss(model, 1) + model.regularization_loss(1)).backward()
        model.optimizer_step()
        saved = copy.deepcopy(model.capture())
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(saved)
        for current in (model, restored):
            (self.image_loss(current, 2) + current.regularization_loss(2)).backward()
            current.optimizer_step()
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation', 'gate_logits'):
            torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
        self.assertEqual(float(model.loss_terms(3, False)['rate_bytes_surrogate']),
                         float(restored.loss_terms(3, False)['rate_bytes_surrogate']))


if __name__ == '__main__':
    unittest.main()
