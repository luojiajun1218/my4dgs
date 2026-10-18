"""Physical shared image fitting and gate-only changes of residual coding origin."""
import copy
import unittest

import torch

import test_predictor_rate_gate as fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class PredictorRecodeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        template = fixtures.PredictorRateGateTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.private_full_motion = False
        cfg.spatial_gate.reparameterize_after_gate_freeze = False
        cfg.spatial_gate.shared_image_recode = True
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def packet(model):
        return fixtures.PredictorRateGateTests.packet(model)

    def motion_clusters(self):
        model = self.model()
        sign = model.parents.new_ones(len(model.parents)).float()
        sign[9:73] = torch.arange(64).remainder(2).float() * 2 - 1
        with torch.no_grad():
            model.region_translation[9:73, 0] = sign[9:73] * .02
            model.gs_translation[:, 0] = sign[model.spatial_paths[:, 1]] * .02
        return model

    def test_main_image_reaches_region_and_gs_and_region_updates_are_not_compensated(self):
        model = self.model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.region_translation_lr = .004
        model.optimizer.param_groups[-1]['lr'] = 0.
        target = model.base_xyz + .02
        xyz, _ = model(1, training=True)
        p, u, g = torch.autograd.grad((xyz - target).square().mean(),
            [model.region_translation, model.gs_translation, model.gate_logits], allow_unused=True)
        self.assertGreater(float(p.abs().sum()), 0.)
        self.assertTrue(bool(u.ne(0).all()))
        self.assertIsNone(g)
        before, gs = xyz.detach(), model.gs_translation.detach().clone()
        (model(1, training=True)[0] - target).square().mean().backward()
        model.optimizer_step()
        torch.testing.assert_close(model.gs_translation, gs, rtol=0, atol=0)
        self.assertLess(float((model(2)[0] - target).square().mean()),
                        float((before - target).square().mean()))

    def test_rate_trains_gate_without_fictitious_predictor_or_gs_gradient(self):
        model = self.motion_clusters()
        terms = model.loss_terms(10, False)
        gate, region, gs = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
            [model.gate_logits, model.region_translation, model.gs_translation], allow_unused=True)
        self.assertTrue(bool(gate[1:9].lt(0).all()))
        self.assertTrue(region is None or bool(region.eq(0).all()))
        self.assertIsNone(gs)
        self.assertEqual(float(terms['candidate_motion_loss']), 0.)

    def test_gates_learn_correlated_child_motion_and_reduce_actual_packet(self):
        model = self.motion_clusters()
        before = self.packet(model)
        motion = tuple(x.clone() for x in model._full_motion_symbols())
        optimizer = torch.optim.Adam([model.gate_logits], lr=.15)
        for step in range(1, 31):
            model.last_step = step
            optimizer.zero_grad()
            model.regularization_loss(step).backward()
            optimizer.step()
            model._select_frontier(freeze=False)
        self.assertTrue(bool(model.gate_logits[1:9].gt(0).all()))
        self.assertTrue(bool(model._gs_motion(hard=True)[0].eq(0).all()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(motion, model._full_motion_symbols())))
        self.assertLess(len(self.packet(model)), len(before))
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(31)[0], rtol=0, atol=0)

    def test_gate_rebase_retains_gs_adam_state_and_bin_remainder(self):
        model = self.motion_clusters()
        with torch.no_grad():
            model.gs_translation[:, 0] += .0004
            model.gate_logits[1:9] = 2.
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
                                        exp_avg_sq=torch.ones_like(p))
        before = p.detach().clone()
        integer = model._full_motion_symbols()[0].clone()
        model._select_frontier(freeze=False)
        self.assertTrue(torch.equal(model._full_motion_symbols()[0], integer))
        self.assertTrue(bool(model.optimizer.state[p]['exp_avg'].eq(1).all()))
        grid = model.gs_steps[:3]
        torch.testing.assert_close(p - (p/grid).round()*grid,
            before - (before/grid).round()*grid, rtol=0, atol=1e-8)

    def test_predictor_auxiliary_keeps_its_own_region_and_gate_image_gradients(self):
        model = self.motion_clusters()
        model(1, training=True)
        xyz, _ = model.predictor_pose()
        target = model.base_xyz + .02
        region, gate, gs = torch.autograd.grad((xyz - target).square().mean(),
            [model.region_translation, model.gate_logits, model.gs_translation], allow_unused=True)
        self.assertGreater(float(region.abs().sum()), 0.)
        self.assertGreater(float(gate.abs().sum()), 0.)
        self.assertIsNone(gs)

    def test_resume_and_freeze_preserve_real_motion_and_next_update(self):
        model = self.motion_clusters()
        target = model.base_xyz + .02
        xyz, _ = model(69, training=True)
        ((xyz - target).square().mean() + model.regularization_loss(69)).backward()
        model.optimizer_step()
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        for current in (model, restored):
            xyz, _ = current(70, training=True)
            loss = (xyz - target).square().mean() + current.regularization_loss(70)
            loss = loss + .25*(current.predictor_pose()[0] - target).square().mean()
            loss.backward()
            current.optimizer_step()
            self.assertTrue(current._hard_gate_frozen)
            self.assertEqual(current.predictor_image_weight, 0.)
        for name in ('region_translation', 'gs_translation', 'gate_logits'):
            torch.testing.assert_close(getattr(model,name), getattr(restored,name), rtol=0, atol=0)
        self.assertEqual(self.packet(model), self.packet(restored))

    def test_unsampled_predictor_does_not_drift_on_adam_velocity(self):
        model = self.model()
        p = model.region_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
                                        exp_avg_sq=torch.ones_like(p))
        model._sampled_gate_step = 1
        model._sampled_gate_noise = torch.full_like(model.gate_logits, -20.)
        before = p.detach().clone()
        target = model.base_xyz + .02
        xyz, _ = model(1, training=True)
        prediction, _ = model.predictor_pose()
        loss = ((xyz - target).square().mean() + .25*(prediction - target).square().mean() +
                model.regularization_loss(1))
        loss.backward()
        self.assertFalse(bool(p.grad[9:].ne(0).any()))
        model.optimizer_step()
        torch.testing.assert_close(p[9:], before[9:], rtol=0, atol=0)
        self.assertFalse(bool(model.optimizer.state[p]['exp_avg'][9:].ne(0).any()))
        self.assertTrue(bool(model.optimizer.state[p]['exp_avg_sq'][9:].gt(.5).all()))

    def test_temperature_only_changes_relaxation_not_hard_sampling_probability(self):
        model = self.model()
        with torch.no_grad():
            model.gate_logits.fill_(-1.)
        samples, probabilities = [], []
        for step in (1, 70):
            model.last_step = step
            model._sampled_gate_step = step
            model._sampled_gate_noise = torch.full_like(model.gate_logits, .8)
            samples.append(model._sampled_path_weights().detach())
            probabilities.append(model._gate_probabilities().detach())
        torch.testing.assert_close(samples[0], samples[1], rtol=0, atol=0)
        torch.testing.assert_close(probabilities[0], probabilities[1], rtol=0, atol=0)
        torch.testing.assert_close(probabilities[0][model.split_eligible],
            torch.full_like(probabilities[0][model.split_eligible], torch.sigmoid(torch.tensor(-1.))),
            rtol=0, atol=0)
        packet = copy.deepcopy(model.capture())
        self.assertEqual(packet['gate_estimator'], 'recoded_binary_concrete_v2')
        packet['gate_estimator'] = 'stochastic_binary_path_v1'
        with self.assertRaises(ValueError):
            SpatialGatedMotion(copy.deepcopy(model.config)).restore(packet)

    def test_freezing_keeps_photo_gs_moments_for_the_unchanged_image_derivative(self):
        model = self.model()
        state = dict(step=torch.tensor(7.), exp_avg=torch.full_like(model.gs_translation, .1),
                     exp_avg_sq=torch.full_like(model.gs_translation, .3))
        model.optimizer.state[model.gs_translation] = copy.deepcopy(state)
        model.optimizer.state[model.region_translation] = dict(
            step=torch.tensor(7.), exp_avg=torch.ones_like(model.region_translation),
            exp_avg_sq=torch.ones_like(model.region_translation))
        model.last_step = 70
        model._select_frontier(freeze=True)
        self.assertNotIn(model.region_translation, model.optimizer.state)
        after = model.optimizer.state[model.gs_translation]
        for name, value in state.items():
            torch.testing.assert_close(after[name], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
