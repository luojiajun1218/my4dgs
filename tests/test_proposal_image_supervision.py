"""Unselected private proposals learn images without training the gate twice."""
import copy
import importlib
from types import MethodType, SimpleNamespace
import unittest

import torch

import test_stochastic_gate_feedback as fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion

MODULE = importlib.import_module('pipeline.module')


class ProposalImageSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, weight=1.):
        template = fixtures.StochasticGateFeedbackTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.proposal_image_weight = weight
        cfg.spatial_gate.reparameterize_after_gate_freeze = True
        cfg.validate()
        motion = SpatialGatedMotion(cfg)
        motion.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        motion.observations.fill_(1)
        return motion

    def module(self, motion):
        alpha = torch.tensor(.3, requires_grad=True)
        color = torch.tensor(.1, requires_grad=True)
        gaussians = SimpleNamespace(_xyz=motion.base_xyz, rigid_xyz=None,
            rigid_rotation=None, rigid_alpha=None, rigid_opacity=None, rigid_sh_dc=None)

        class Renderer:
            def __init__(self):
                self.gaussians = gaussians
                self.collect_contributions = True
                self.calls = []

            def __call__(self, view, background):
                self.calls.append(self.collect_contributions)
                gs = self.gaussians
                value = gs.rigid_xyz.mean() + gs.rigid_alpha + gs.rigid_sh_dc
                return dict(render=value.expand(3, 20, 20), marker=len(self.calls))

        module = SimpleNamespace(motion=motion, gaussians=gaussians, births=None,
            renderer=Renderer(), background=torch.zeros(3),
            frame_prepared=True, frame_finalized=False, motion_committed=False,
            motion_steps=motion.motion_steps, trainer=SimpleNamespace(global_step=1),
            opacity_update=SimpleNamespace(alphas=lambda: alpha),
            feature_dc_update=lambda: color,
            config=SimpleNamespace(effective_motion_observations=True,
                opacity_update=SimpleNamespace(mode='alpha', joint_with_birth=False),
                feature_dc_update=SimpleNamespace(joint_with_birth=True),
                motion_coarse_to_fine=False, rigid_tree=motion.config, lambda_dssim=.2))
        module.forward = MethodType(MODULE.Module.forward, module)
        return module, alpha, color

    def test_closed_proposals_get_direct_image_gradients_without_gate_or_region_gradients(self):
        motion = self.model()
        with torch.no_grad():
            motion.gate_logits[motion.split_eligible] = -40.
        target = motion.base_xyz + .02
        image, _ = motion(1, training=True)
        main_gradient = torch.autograd.grad((image - target).square().mean(),
                                            motion.gs_translation)[0]
        self.assertEqual(float(main_gradient.abs().sum()), 0.)
        sampler = motion._gate_rng.get_state().clone()
        noise = motion._sampled_gate_noise.clone()
        global_rng = torch.get_rng_state().clone()
        proposal, rotation = motion.proposal_pose()
        gradients = torch.autograd.grad((proposal - target).square().mean(),
            [motion.gs_translation, motion.region_translation, motion.gate_logits], allow_unused=True)
        self.assertTrue(bool(gradients[0].ne(0).all()))
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        self.assertTrue(torch.equal(rotation, motion.base_rotation))
        self.assertTrue(torch.equal(sampler, motion._gate_rng.get_state()))
        self.assertTrue(torch.equal(noise, motion._sampled_gate_noise))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))

    def test_auxiliary_render_detaches_appearance_and_does_not_collect_birth_evidence(self):
        module, alpha, color = self.module(self.model())
        proposal = module.forward(None, training=True, proposal_only=True)['render']
        gradients = torch.autograd.grad(proposal.sum(),
            [module.motion.gs_translation, alpha, color], allow_unused=True)
        self.assertTrue(bool(gradients[0].ne(0).all()))
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])
        self.assertEqual(module.renderer.calls, [False])
        self.assertTrue(module.renderer.collect_contributions)
        self.assertIs(module.renderer.gaussians, module.gaussians)
        for name in ('rigid_xyz', 'rigid_rotation', 'rigid_alpha', 'rigid_opacity', 'rigid_sh_dc'):
            self.assertIsNone(getattr(module.gaussians, name))
        with self.assertRaises(ValueError):
            module.forward(None, proposal_only=True)

    def test_training_combines_losses_and_returns_only_the_actual_field_package(self):
        module, alpha, color = self.module(self.model(weight=.7))
        view = SimpleNamespace(image=torch.full((3, 20, 20), .6))
        loss, metrics, package = MODULE.Module.training_step(module, view, 1,
                                                           regularization_weight=0.)
        expected = metrics['image_loss'] + .7 * metrics['proposal_image_loss']
        torch.testing.assert_close(loss.detach(), expected, rtol=0, atol=0)
        self.assertEqual(package['marker'], 1)
        self.assertEqual(module.renderer.calls, [True, False])
        # Appearance is trained by the main image alone.
        main = module.forward(view, training=True)['render']
        main_loss = MODULE.motion_photometric_loss(main, view.image, .2)
        expected_gradients = torch.autograd.grad(main_loss, [alpha, color])
        gradients = torch.autograd.grad(loss, [alpha, color])
        for value, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_freeze_disables_auxiliary_render_and_restored_state_preserves_stage(self):
        for frozen in (False, True):
            motion = self.model()
            if frozen:
                motion._select_frontier(freeze=True)
            restored = SpatialGatedMotion(copy.deepcopy(motion.config))
            restored.restore(copy.deepcopy(motion.capture()))
            self.assertEqual(restored.proposal_image_weight, 0. if frozen else 1.)
            module, _, _ = self.module(restored)
            module.trainer.global_step = 71 if frozen else 1
            view = SimpleNamespace(image=torch.full((3, 20, 20), .6))
            _, metrics, _ = MODULE.Module.training_step(module, view,
                module.trainer.global_step, regularization_weight=0.)
            self.assertEqual(len(module.renderer.calls), 1 if frozen else 2)
            self.assertEqual('proposal_image_loss' in metrics, not frozen)
            if frozen:
                with self.assertRaises(ValueError):
                    restored.proposal_pose()

    def test_zero_default_restores_older_config_and_never_adds_a_render(self):
        motion = self.model(weight=0.)
        packet = copy.deepcopy(motion.capture())
        packet['config']['spatial_gate'].pop('proposal_image_weight')
        restored = SpatialGatedMotion(copy.deepcopy(motion.config))
        restored.restore(packet)
        module, _, _ = self.module(restored)
        MODULE.Module.training_step(module,
            SimpleNamespace(image=torch.full((3, 20, 20), .6)), 1, regularization_weight=0.)
        self.assertEqual(module.renderer.calls, [True])

    def test_invalid_weights_and_parameterizations_are_rejected(self):
        template = self.model().config
        for weight in (True, -.1, float('inf'), float('nan')):
            cfg = copy.deepcopy(template)
            cfg.spatial_gate.proposal_image_weight = weight
            with self.assertRaises(ValueError):
                cfg.validate()
        for field, value in (('private_full_motion', False), ('image_relaxation', 'soft_anneal')):
            cfg = copy.deepcopy(template)
            setattr(cfg.spatial_gate, field, value)
            with self.assertRaises(ValueError):
                cfg.validate()


if __name__ == '__main__':
    unittest.main()
