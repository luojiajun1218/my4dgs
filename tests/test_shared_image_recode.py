"""Shared image steps are physical; learned cut changes only recode residuals."""
import copy
import unittest

import torch

from pipeline.model.spatial_gated_motion import SpatialGatedMotion
import test_gate_candidate_refresh as candidate_tests


class SharedImageRecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model, parent, body = candidate_tests.GateCandidateRefreshTests().model()
        model.config.spatial_gate.shared_image_recode = True
        model.config.validate()
        return model, parent, body

    def test_shared_image_update_is_not_canceled_by_gs_rebasing(self):
        model, _, _ = self.model()
        model.config.relative_motion_loss_weight = 0.
        model.config.rate_loss_weight = 0.
        model.config.region_translation_lr = .01
        xyz, _ = model(1, training=True)
        before = xyz.detach().clone()
        residual = model.gs_translation.detach().clone()
        target = model.base_xyz.clone(); target[:, 0] += .02
        loss = (xyz-target).square().mean()
        gate_gradient, = torch.autograd.grad(loss, [model.gate_logits], allow_unused=True, retain_graph=True)
        self.assertIsNone(gate_gradient)  # A pure representation switch preserves images.
        loss.backward()
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)
        model.gs_translation.grad.zero_()  # Isolate the actual shared update.
        model.optimizer_step()
        self.assertGreater(float((model(1)[0] - before).abs().sum()), 0.)
        torch.testing.assert_close(model.gs_translation, residual, rtol=0, atol=0)

    def test_rate_trains_gate_candidates_without_fictitious_shared_image_gradients(self):
        model, parent, body = self.model()
        model._refresh_unselected_candidates()
        child = model.spatial_paths[body, 1].unique()[0]
        with torch.no_grad():
            model.region_translation[child, 0] += model.steps[0]
        selected = model._stored_region_support()
        region, residual, gate = torch.autograd.grad(model.regularization_loss(21),
            [model.region_translation, model.gs_translation, model.gate_logits], allow_unused=True)
        self.assertIsNone(residual)
        self.assertTrue(bool(region[selected].eq(0).all()))
        self.assertGreater(float(region[~selected].abs().sum()), 0.)
        self.assertLess(float(gate[parent]), 0.)

    def test_hard_gate_switch_preserves_pose_and_checkpoint_mode(self):
        model, parent, _ = self.model()
        model._refresh_unselected_candidates()
        before = model(21)[0].detach().clone()
        with torch.no_grad():
            model.gate_logits[parent] = 2.
            model._select_frontier(freeze=False, preserve_remainder=True)
        torch.testing.assert_close(model(21)[0], before, rtol=0, atol=1e-6)
        saved = copy.deepcopy(model.capture())
        self.assertEqual(saved['regional_image_fitting'], 'shared_image_recode_v1')
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(saved)
        torch.testing.assert_close(restored(21)[0], model(21)[0], rtol=0, atol=0)

    def test_incompatible_modes_are_rejected(self):
        model, _, _ = self.model()
        model.config.proximal_gs_regularization = False
        with self.assertRaises(ValueError):
            model.config.validate()
        model.config.proximal_gs_regularization = True
        model.config.spatial_gate.joint_image_gradients = True
        with self.assertRaises(ValueError):
            model.config.validate()

    def test_motion_evidence_is_not_pruned_before_gate_discovery_finishes(self):
        model, _, _ = self.model()
        model.config.defer_gs_sparsity_until_gate_freeze = True
        model.config.validate()
        with torch.no_grad():
            model.gs_translation.fill_(1e-6)
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(p), exp_avg_sq=torch.full_like(p, 1e-11))
        before = p.detach().clone()
        for step in (20, 60, 70):
            model.last_step = step
            model._proximal_relative_gs()
            model._proximal_gs_blocks()
            torch.testing.assert_close(p, before, rtol=0, atol=0)
        model.last_step = 80
        model._proximal_relative_gs()
        model._proximal_gs_blocks()
        self.assertTrue(bool(p.eq(0).all()))


if __name__ == '__main__':
    unittest.main()
