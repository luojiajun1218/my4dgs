"""Learned hierarchy controls whether individual corrections are available."""
import copy
import unittest

import torch

import test_joint_spatial_motion as fixtures
import test_joint_image_rate as image_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class HierarchicalPrivateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, enabled=True):
        template = fixtures.JointSpatialMotionTests().model(initial_logit=-.2)
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.joint_image_gradients = True
        cfg.spatial_gate.hierarchical_private = enabled
        cfg.proximal_gs_regularization = True
        cfg.differentiable_rate_support = True
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    def test_closed_private_branch_is_zero_in_render_and_stream_but_can_learn(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
        xyz, _ = model(1, training=True)
        torch.testing.assert_close(xyz, model.base_xyz, rtol=0, atol=0)
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))
        loss = (xyz - model.base_xyz - .02).square().mean()
        gs, gates = torch.autograd.grad(loss, [model.gs_translation, model.gate_logits])
        self.assertGreater(float(gs.abs().sum()), 0.)
        self.assertLess(float(gates[model.split_eligible].sum()), 0.)
        decoded = decode_motion(image_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=0)

    def test_private_permission_requires_every_internal_gate(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
            model.gate_logits[model.split_eligible] = 2.
        model._select_frontier(freeze=False)
        self.assertTrue(bool(model._private_support().all()))
        parent = int(model.spatial_paths[0, 1])
        members = model.spatial_paths[:, 1].eq(parent)
        with torch.no_grad():
            model.gate_logits[parent] = -2.
        model._select_frontier(freeze=False)
        self.assertFalse(bool(model._private_support()[members].any()))
        self.assertTrue(bool(model._private_support()[~members].all()))
        symbols = model._values(hard=True)[2]
        self.assertTrue(bool(symbols[members].eq(0).all()))
        self.assertTrue(bool(symbols[~members, 0].ne(0).all()))
        self.assertFalse(bool(model.opened[model.spatial_paths[:, -1]].any()))

    def test_image_and_rate_gradients_judge_private_capacity(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
        terms = model.loss_terms(2, diagnostics=False)
        rate_gate, gs = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
            [model.gate_logits, model.gs_translation], allow_unused=True)
        self.assertIsNone(gs)
        self.assertGreater(float(rate_gate[model.split_eligible].sum()), 0.)

        model.config.rate_loss_weight = 1e-6
        model.config.relative_motion_loss_weight = 0.
        optimizer = torch.optim.Adam([model.gate_logits], lr=.2)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        initial = float((model(1)[0] - target).square().mean())
        for step in range(1, 61):
            optimizer.zero_grad()
            xyz, _ = model(step, training=True)
            loss = (xyz - target).square().mean() + model.regularization_loss(step)
            loss.backward()
            optimizer.step()
            model._select_frontier(freeze=False)
        final = float((model(60)[0] - target).square().mean())
        self.assertLess(final, initial * .1)
        self.assertTrue(bool(model._private_support().any()))

    def test_proximal_cost_scales_with_soft_image_path(self):
        dense, conditional = self.model(enabled=False), self.model()
        probability = conditional._private_probabilities().detach()[:, None]
        for model in (dense, conditional):
            with torch.no_grad():
                model.gs_translation[:, 0] = .002
                model.gs_translation[::2, 0] = .004
            p = model.gs_translation
            variance = torch.full_like(p, 1e-10)
            if model is conditional:
                variance *= probability.square()
            model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(p),
                exp_avg_sq=variance)
            model.last_step = 20
            model._proximal_gs_blocks()
        torch.testing.assert_close(conditional.gs_translation, dense.gs_translation, rtol=0, atol=0)

    def test_freeze_removes_hidden_momentum_and_resume_keeps_permission(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation.fill_(.02)
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
            exp_avg_sq=torch.ones_like(p))
        model._select_frontier(freeze=True)
        self.assertTrue(bool(model.gs_translation.eq(0).all()))
        self.assertNotIn(p, model.optimizer.state)
        saved = copy.deepcopy(model.capture())
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(saved)
        for current in (model, restored):
            xyz, _ = current(71, training=True)
            (xyz - current.base_xyz - .02).square().mean().backward()
            current.optimizer_step()
            self.assertTrue(bool(current.gs_translation.eq(0).all()))
            current.retain_survivors(torch.arange(0, len(current.owner), 2))
        torch.testing.assert_close(model.region_translation, restored.region_translation, rtol=0, atol=0)
        self.assertTrue(torch.equal(model._private_support(), restored._private_support()))

    def test_frame_boundary_migration_preserves_previous_applied_motion_proposal(self):
        helper = fixtures.JointSpatialMotionTests()
        model = self.model(enabled=False)
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
        applied = model._full_motion_symbols()[0].clone()
        state = helper.commit(model, births=model.base_xyz[:1], sources=torch.tensor([0]))
        model.config.spatial_gate.hierarchical_private = True
        self.assertFalse(model.hierarchical_private)  # Still the preceding frame.
        model.setup(state.xyz, state.rotation, model.extent, frame_index=2)
        self.assertTrue(model.hierarchical_private)
        rt, _ = model._regional_predictions(hard=True)
        latent = rt[model.owner].float() + torch.round(model.gs_translation / model.gs_steps[:3])
        torch.testing.assert_close(latent, torch.cat((applied, applied[:1])), rtol=0, atol=0)

    def test_validation_rejects_training_without_coupled_image_gradients(self):
        model = self.model()
        model.config.spatial_gate.joint_image_gradients = False
        with self.assertRaises(ValueError):
            model.config.validate()
        model.config.spatial_gate.joint_image_gradients = True
        model.config.proximal_gs_regularization = False
        with self.assertRaises(ValueError):
            model.config.validate()


if __name__ == '__main__':
    unittest.main()
