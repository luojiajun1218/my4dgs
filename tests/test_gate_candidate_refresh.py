"""Closed predictors must be useful before a learned gate can assess sharing."""
import unittest

import torch

import test_joint_spatial_motion as joint_tests


class GateCandidateRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_tests.JointSpatialMotionTests().model(initial_logit=-3.)
        model.config.spatial_gate.candidate_refresh_interval = 20
        model.config.spatial_gate.temporal_logit_decay = .5
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        model.observations.fill_(1)
        parent = int(model.owner[0])
        body = model.owner.eq(parent)
        with torch.no_grad():
            direction = model.spatial_paths[body, 1].remainder(2).float() * 2 - 1
            model.gs_translation[body, 0] = direction * 20 * model.gs_steps[0]
        return model, parent, body

    def test_refresh_preserves_selected_predictors_pose_and_gates(self):
        model, _, _ = self.model()
        before = model(20)[0].detach().clone()
        selected = model._stored_region_support()
        rt, gt = model.region_translation.detach().clone(), model.gs_translation.detach().clone()
        gates = model.gate_logits.detach().clone()
        opened = model.opened.clone()
        p = model.region_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
            exp_avg_sq=torch.ones_like(p))
        self.assertGreater(model._refresh_unselected_candidates(), 0)
        torch.testing.assert_close(model(20)[0], before, rtol=0, atol=0)
        torch.testing.assert_close(model.region_translation[selected], rt[selected], rtol=0, atol=0)
        torch.testing.assert_close(model.gs_translation, gt, rtol=0, atol=0)
        torch.testing.assert_close(model.gate_logits, gates, rtol=0, atol=0)
        self.assertTrue(torch.equal(model.opened, opened))
        self.assertTrue(bool(model.optimizer.state[p]['exp_avg'][selected].eq(1).all()))
        refreshed = ~selected & model.node_counts.gt(0)
        self.assertTrue(bool(model.optimizer.state[p]['exp_avg'][refreshed].eq(0).all()))

    def test_gate_learns_shared_child_motion_and_reduces_coded_blocks(self):
        model, parent, body = self.model()
        before = model(20)[0].detach().clone()
        initial_bytes = float(model.loss_terms(20, False)['rate_bytes_surrogate'])
        model._refresh_unselected_candidates()
        optimizer = torch.optim.Adam([model.gate_logits], lr=.15)
        for step in range(21, 71):
            optimizer.zero_grad(set_to_none=True)
            model.regularization_loss(step).backward()
            optimizer.step()
            with torch.no_grad():
                model.gate_logits.clamp_(-6, 6)
                model._select_frontier(freeze=False, preserve_remainder=True)
        self.assertGreater(float(model.gate_logits[parent]), 0.)
        self.assertTrue(bool(model.owner[body].ne(parent).all()))
        self.assertTrue(bool(torch.round(model.gs_translation[body] / model.gs_steps[:3]).eq(0).all()))
        self.assertLess(float(model.loss_terms(70, False)['rate_bytes_surrogate']), initial_bytes)
        torch.testing.assert_close(model(70)[0], before, rtol=0, atol=1e-6)

    def test_refresh_stops_after_gate_freeze_and_validates_settings(self):
        model, _, _ = self.model()
        model._hard_gate_frozen = True
        with self.assertRaises(ValueError):
            model._refresh_unselected_candidates()
        for value in (True, -1, .5):
            model.config.spatial_gate.candidate_refresh_interval = value
            with self.assertRaises(ValueError):
                model.config.validate()

    def test_temporal_decay_preserves_decisions_with_less_confidence(self):
        helper = joint_tests.JointSpatialMotionTests()
        model = helper.model()
        model.config.spatial_gate.temporal_logit_decay = .5
        with torch.no_grad():
            model.gate_logits.fill_(-6.)
        decoded = helper.commit(model)
        opened = decoded.spatial_open.clone()
        model.setup(decoded.xyz, decoded.rotation, 2., frame_index=2)
        torch.testing.assert_close(model.gate_logits, torch.full_like(model.gate_logits, -3.), rtol=0, atol=0)
        self.assertTrue(torch.equal(model.opened, opened))


if __name__ == '__main__':
    unittest.main()
