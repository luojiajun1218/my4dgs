"""Shared predictors use photo proposals and local sparsification costs."""
import copy
import unittest

import torch

import test_gate_candidate_refresh as fixtures
from pipeline.model.predictor_rd import predictor_cost
from pipeline.model.spatial_gated_motion import SpatialGatedMotion


class PredictorRDTests(unittest.TestCase):
    def model(self):
        model, parent, body = fixtures.GateCandidateRefreshTests().model()
        model.config.spatial_gate.candidate_refresh_interval = 0
        model.config.spatial_gate.predictor_objective = 'proximal_rd'
        model.config.validate()
        return model, parent, body

    def test_local_choice_keeps_sensitive_update_and_zeros_cheap_one(self):
        target = torch.zeros(2, 6)
        target[:, 0] = .01
        predictor = torch.zeros_like(target, requires_grad=True)
        metric = torch.ones_like(target)
        metric[1] *= 1e-5
        cost, residual, byte_cost = predictor_cost(target, predictor, metric,
            torch.full((6,), .001), torch.zeros(6), 1e-7, torch.full((2,), 3.))
        torch.testing.assert_close(residual[0], target[0], rtol=0, atol=1e-8)
        self.assertTrue(bool(residual[1].eq(0).all()))
        torch.testing.assert_close(byte_cost, torch.tensor([9., 0.]))
        cost.sum().backward()
        self.assertLess(abs(float(predictor.grad[0, 0])), 1e-8)
        self.assertLess(float(predictor.grad[1, 0]), 0.)

    def test_closed_candidates_fit_without_gate_or_point_target_gradients(self):
        model, parent, body = self.model()
        target = torch.zeros(len(model.owner), 6, requires_grad=True)
        with torch.no_grad():
            target[body, 0] = .0005
            model.gate_logits.fill_(-2.)
            model.region_translation.zero_()
        model._predictor_fit_state = dict(target=target, metric=torch.ones_like(target) * .001)
        terms = model.loss_terms(2, diagnostics=False)
        gradients = torch.autograd.grad(terms['candidate_motion_loss'],
            [model.region_translation, model.gate_logits, model.gs_translation, target],
            allow_unused=True)
        closed = model.spatial_paths[body, 1].unique()
        self.assertGreater(float(gradients[0][closed].abs().sum()), 0.)
        self.assertEqual(gradients[1:], (None, None, None))

    def test_gate_learns_beneficial_split_from_local_costs(self):
        model, parent, body = self.model()
        with torch.no_grad():
            target = torch.zeros(len(model.owner), 6)
            node = model.spatial_paths[body, 1]
            sign = node.remainder(2).float() * 2 - 1
            target[body, 0] = sign * .02
            model.region_translation.zero_()
            model.region_rotation.zero_()
            for child in node.unique():
                model.region_translation[child, 0] = target[body][node.eq(child), 0][0]
            model.gate_logits.fill_(-2.)
        model._predictor_fit_state = dict(target=target, metric=torch.ones_like(target))
        gate_optimizer = torch.optim.Adam([model.gate_logits], lr=.2, eps=1e-12)
        for _ in range(45):
            gate_optimizer.zero_grad()
            terms = model.loss_terms(2, diagnostics=False)
            (terms['rate_loss'] + terms['relative_motion_loss']).backward()
            gate_optimizer.step()
        self.assertGreater(float(model.gate_logits[parent]), 0.)

    def test_proposal_is_saved_before_pruning_and_resume_is_exact(self):
        model, _, _ = self.model()
        xyz, _ = model(1, training=True)
        (xyz - model.base_xyz - .001).square().mean().backward()
        model.optimizer_step()
        self.assertIsNotNone(model._predictor_fit_state)
        state = copy.deepcopy(model._predictor_fit_state)
        saved = copy.deepcopy(model.capture())
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(saved)
        for current in (model, restored):
            xyz, _ = current(2, training=True)
            ((xyz - current.base_xyz - .001).square().mean() +
             current.regularization_loss(2)).backward()
            current.optimizer_step()
        for name in ('region_translation', 'region_rotation', 'gs_translation',
                     'gs_rotation', 'gate_logits'):
            torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
        rows = torch.arange(0, len(restored.owner), 2)
        expected = {name: value[rows].clone() for name, value in restored._predictor_fit_state.items()}
        restored.retain_survivors(rows)
        for name, value in expected.items():
            torch.testing.assert_close(restored._predictor_fit_state[name], value, rtol=0, atol=0)
        diagnostics = restored.loss_terms(2)
        for name in ('gs_active_rows', 'gs_translation_active_blocks', 'gs_rotation_active_blocks',
                     'spatial_active_regions', 'rate_bytes_hard'):
            self.assertIn(name, diagnostics)
        self.assertGreater(float(state['target'].abs().sum()), 0.)

    def test_validation_rejects_coordinate_mismatch(self):
        model, _, _ = self.model()
        model.config.spatial_gate.shared_image_recode = True
        with self.assertRaises(ValueError):
            model.config.validate()

    def test_fixed_gate_regions_and_points_receive_physical_image_updates(self):
        model, _, _ = self.model()
        model._select_frontier(freeze=True)
        model.config.rate_loss_weight = model.config.relative_motion_loss_weight = 0.
        model.config.region_translation_lr = .01
        residual = model.gs_translation.detach().clone()
        xyz, _ = model(71, training=True)
        before = xyz.detach().clone()
        loss = (xyz - model.base_xyz - .02).square().mean()
        region, point, gate = torch.autograd.grad(loss,
            [model.region_translation, model.gs_translation, model.gate_logits],
            allow_unused=True, retain_graph=True)
        self.assertGreater(float(region.abs().sum()), 0.)
        self.assertGreater(float(point.abs().sum()), 0.)
        self.assertIsNone(gate)
        loss.backward()
        model.gs_translation.grad.zero_()
        model.optimizer_step()
        self.assertGreater(float((model(71)[0] - before).abs().sum()), 0.)
        torch.testing.assert_close(model.gs_translation, residual, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
