"""Motion Training tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import importlib
import json
import math
import tempfile
import unittest
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.nn.functional import l1_loss

import test_spatial_motion as candidate_tests
import test_spatial_motion as codec_fixtures
import test_spatial_motion as fixtures
import test_spatial_motion as image_fixtures
import test_spatial_motion as lineage
import test_spatial_motion as lineage_fixtures
from pipeline.model.spatial_gated_motion import predictor_cost
from pipeline.model.rigid_tree_codec import decode_motion, encode_motion
from pipeline.model.spatial_gated_motion import SpatialGatedMotion

# ----- test_private_motion_proposals -----
class PrivateMotionProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, soft=False):
        template = fixtures.JointSpatialMotionTests().model(initial_logit=-.2)
        cfg = copy.deepcopy(template.config)
        cfg.rotation_updates = False
        cfg.proximal_gs_regularization = True
        cfg.differentiable_rate_support = True
        cfg.spatial_gate.joint_image_gradients = True
        cfg.spatial_gate.hierarchical_private = True
        cfg.spatial_gate.private_full_motion = True
        if soft:
            cfg.spatial_gate.image_relaxation = 'soft_anneal'
            cfg.spatial_gate.temperature = 3.
            cfg.spatial_gate.final_temperature = .3
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def open_all(model, freeze=True):
        with torch.no_grad():
            model.gate_logits[model.split_eligible] = 2.
        model._select_frontier(freeze=freeze)
        assert bool(model._private_support().all())

    def test_leaf_predictor_changes_encoding_without_training_the_image_twice(self):
        model = self.model()
        self.open_all(model)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .01
            model.gs_translation[:, 0] = .03
        before, _ = model(71, training=True)
        symbols_before = model._values(hard=True)[2].clone()
        target = model.base_xyz.clone()
        target[:, 0] += .05
        p, u = torch.autograd.grad((before - target).square().mean(),
                                  [model.region_translation, model.gs_translation])
        self.assertEqual(float(p[leaves].abs().sum()), 0.)
        self.assertGreater(float(u.abs().sum()), 0.)
        with torch.no_grad():
            model.region_translation[leaves, 0] += .002
        after, _ = model(71)
        torch.testing.assert_close(before, after, rtol=0, atol=2e-7)
        self.assertTrue(bool((model._values(hard=True)[2][:, 0] == symbols_before[:, 0] - 1).all()))
        decoded = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, after, rtol=0, atol=0)

    def test_closed_leaf_predictors_receive_coding_gradients_and_gs_adam_stays_photometric(self):
        for soft in (False, True):
            model = self.model(soft=soft)
            with torch.no_grad():
                model.gs_translation[:, 0] = .02
            xyz, _ = model(1, training=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            image_gradient = torch.autograd.grad((xyz - target).square().mean(), model.region_translation)[0]
            leaves = model.spatial_paths[:, -1].unique()
            self.assertEqual(float(image_gradient[leaves].abs().sum()), 0.)
            terms = model.loss_terms(2, diagnostics=False)
            regional, point = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
                [model.region_translation, model.gs_translation], allow_unused=True)
            self.assertLess(float(regional[leaves, 0].sum()), 0.)
            self.assertIsNone(point)

    def test_matching_full_motion_and_region_prediction_needs_no_private_symbols(self):
        model = self.model()
        self.open_all(model)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .02
            model.gs_translation[:, 0] = .02
        self.assertTrue(bool(model.gs_translation[:, 0].ne(0).all()))
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))
        model.record_final_diagnostics()
        decoded = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(71)[0], rtol=0, atol=0)
        torch.testing.assert_close(decoded.xyz[:, 0], model.base_xyz[:, 0] +
                                  torch.full_like(model.gs_translation[:, 0], .02), rtol=0, atol=2e-7)

    def test_l1_and_block_sparsity_shrink_towards_the_coding_predictor(self):
        for block in (False, True):
            model = self.model()
            self.open_all(model)
            leaves = model.spatial_paths[:, -1].unique()
            midpoint = len(model.owner) // 2
            with torch.no_grad():
                model.region_translation[leaves, 0] = .02
                model.gs_translation[:, 0] = .022
                if block:
                    model.gs_translation[midpoint:, 0] = .2
            p = model.gs_translation
            model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(p),
                                            exp_avg_sq=torch.full_like(p, 1e-12))
            model.last_step = 20
            if block:
                model._proximal_gs_blocks()
                self.assertGreater(float(model.gs_translation[midpoint:, 0].min()), .19)
            else:
                model.config.relative_motion_loss_weight = 1.
                model._proximal_relative_gs()
            origins = model._private_origins()[0]
            torch.testing.assert_close(model.gs_translation[:midpoint], origins[:midpoint], rtol=0, atol=0)
            self.assertTrue(bool(model._values(hard=True)[2][:midpoint].eq(0).all()))

    def test_migration_warms_complete_motion_including_birth_lineage(self):
        helper = fixtures.JointSpatialMotionTests()
        model = helper.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .01
            model.gs_translation[:, 0] = .02
        previous = model._full_motion_symbols()[0].clone()
        state = helper.commit(model, births=model.base_xyz[:1], sources=torch.tensor([0]))
        model.config.spatial_gate.joint_image_gradients = True
        model.config.spatial_gate.hierarchical_private = True
        model.config.spatial_gate.private_full_motion = True
        model.config.proximal_gs_regularization = True
        model.config.validate()
        self.assertFalse(model.private_full_motion)
        model.setup(state.xyz, state.rotation, model.extent, frame_index=2)
        self.assertTrue(model.private_full_motion)
        actual = torch.round(model.gs_translation / model.gs_steps[:3])
        torch.testing.assert_close(actual, torch.cat((previous, previous[:1])), rtol=0, atol=0)

    def test_soft_and_hard_stage_restore_the_same_future_update(self):
        for step in (20, 70):
            model = self.model(soft=True)
            xyz, _ = model(step, training=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(copy.deepcopy(model.capture()))
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            model.retain_survivors(torch.arange(0, len(model.owner), 2))
            state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
            torch.testing.assert_close(state.xyz, model(step + 1)[0], rtol=0, atol=0)

    def test_residual_range_is_checked_before_converting_to_int16(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[model.spatial_paths[:, -1].unique(), 0] = -100.
            model.gs_translation[:, 0] = 100.
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))
        self.open_all(model)
        with self.assertRaisesRegex(ValueError, 'int16'):
            model._values(hard=True)

    def test_full_motion_configuration_requires_its_matching_capacity_and_proximal_map(self):
        model = self.model()
        cfg = copy.deepcopy(model.config)
        cfg.spatial_gate.hierarchical_private = False
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg = copy.deepcopy(model.config)
        cfg.amplitude_refit_steps = 10
        with self.assertRaises(ValueError):
            cfg.validate()


# ----- test_stochastic_gate_feedback -----
class StochasticGateFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        template = PrivateMotionProposalTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.image_relaxation = 'stochastic_ste'
        cfg.spatial_gate.temperature = 3.
        cfg.spatial_gate.final_temperature = .3
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    def test_render_uses_one_actual_expert_and_reuses_the_sample_for_two_views(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)).remainder(7) * .002
            model.gs_translation[:, 0] = .054
        global_rng = torch.get_rng_state().clone()
        xyz, _ = model(1, training=True)
        noise = model._sampled_gate_noise.clone()
        rng = model._gate_rng.get_state().clone()
        second, _ = model(1, training=True)
        torch.testing.assert_close(xyz, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, model._gate_rng.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        regions = model._regional_predictions()[0]
        proposal = torch.round(model.gs_translation / model.gs_steps[:3]) * model.gs_steps[:3]
        choices = torch.cat((regions[model.spatial_paths[:, :-1]], proposal[:, None]), 1)
        choices = model.base_xyz[:, None] + choices
        distance = (xyz[:, None] - choices).abs().amax(-1).amin(-1)
        self.assertEqual(float(distance.max()), 0.)
        model(2, training=True)
        self.assertFalse(torch.equal(noise, model._sampled_gate_noise))

    def test_sampled_expert_gets_its_own_error_and_unselected_experts_have_no_image_gradient(self):
        model = self.model()
        model(1, training=True)
        draw = model.gate_logits / model._gate_temperature() + model._sampled_gate_noise
        weights = model._path_weights((draw.ge(0) & model.split_eligible).to(model.gate_logits))
        row, level = weights[:, :-1].nonzero()[0].tolist()
        node = int(model.spatial_paths[row, level])
        members = model.spatial_paths[:, level].eq(node)
        target = model.base_xyz.clone()
        target[members, 0] += .02
        gradients = []
        for displacement in (.01, .2):
            with torch.no_grad():
                model.region_translation.zero_()
                model.region_translation[node, 0] = displacement
            xyz, _ = model(1, training=True)
            gradient = torch.autograd.grad((xyz - target).square().mean(), model.region_translation)[0]
            gradients.append(float(gradient[node, 0]))
            selected = torch.zeros(len(model.parents), dtype=torch.bool)
            for layer in range(model.spatial_paths.shape[1] - 1):
                selected[model.spatial_paths[weights[:, layer].bool(), layer]] = True
            self.assertEqual(float(gradient[~selected].abs().sum()), 0.)
        self.assertLess(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)

    def test_private_gradients_use_sampled_capacity_and_gate_probabilities_are_learnable(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .01
        xyz, _ = model(1, training=True)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        point, gate = torch.autograd.grad((xyz - target).square().mean(),
            [model.gs_translation, model.gate_logits])
        draw = model.gate_logits / model._gate_temperature() + model._sampled_gate_noise
        allowed = (draw.ge(0) & model.split_eligible)[model.spatial_paths[:, :-1]].all(-1)
        self.assertTrue(torch.equal(point.ne(0).any(-1), allowed))
        self.assertGreater(float(gate.abs().sum()), 0.)

        with torch.no_grad():
            model.gs_translation[:, 0] = .02
        optimizer = torch.optim.Adam([model.gate_logits], lr=.3)
        initial = float((model(1)[0] - target).square().mean())
        for step in range(1, 61):
            optimizer.zero_grad()
            xyz, _ = model(step, training=True)
            (xyz - target).square().mean().backward()
            optimizer.step()
            model._select_frontier(freeze=False)
        final = float((model(60)[0] - target).square().mean())
        self.assertLess(final, initial * .1)

    def test_sampling_never_changes_the_deterministic_stream_and_final_fit_is_exact(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[9:, 0] = .02
            model.gs_translation[:, 1] = .03
        packet = codec_fixtures.JointImageRateTests.packet(model)
        owner = model.owner.clone()
        for step in (1, 2, 3):
            model(step, training=True)
        self.assertTrue(torch.equal(owner, model.owner))
        self.assertEqual(packet, codec_fixtures.JointImageRateTests.packet(model))
        state = decode_motion(packet, model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(3)[0], rtol=0, atol=0)
        model._select_frontier(freeze=True)
        state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(71, training=True)[0], rtol=0, atol=0)

    def test_resume_preserves_current_and_future_gate_samples_and_updates(self):
        for step in (20, 70):
            model = self.model()
            target = model.base_xyz.clone()
            target[:, 0] += .02
            xyz, _ = model(step, training=True)
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            packet = copy.deepcopy(model.capture())
            restored.restore(packet)
            if step == 20:
                torch.testing.assert_close(model(step, training=True)[0],
                                           restored(step, training=True)[0], rtol=0, atol=0)
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
            self.assertTrue(torch.equal(model._sampled_gate_noise, restored._sampled_gate_noise))
            self.assertTrue(torch.equal(model._gate_rng.get_state(), restored._gate_rng.get_state()))
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            broken = copy.deepcopy(packet)
            broken.pop('image_gate_sampler')
            with self.assertRaises(ValueError):
                SpatialGatedMotion(copy.deepcopy(model.config)).restore(broken)

    def test_stochastic_mode_rejects_an_incompatible_private_parameterization(self):
        model = self.model()
        cfg = copy.deepcopy(model.config)
        cfg.spatial_gate.private_full_motion = False
        with self.assertRaises(ValueError):
            cfg.validate()


# ----- test_proposal_image_supervision -----
MODULE = importlib.import_module('pipeline.module')


class ProposalImageSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, weight=1.):
        template = StochasticGateFeedbackTests().model()
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
        gaussians = SimpleNamespace(_xyz=motion.base_xyz, rigid_xyz=None,
            rigid_rotation=None, rigid_alpha=None, rigid_opacity=None)

        class Renderer:
            def __init__(self):
                self.gaussians = gaussians
                self.collect_contributions = True
                self.calls = []

            def __call__(self, view, background):
                self.calls.append(self.collect_contributions)
                gs = self.gaussians
                value = gs.rigid_xyz.mean() + gs.rigid_alpha
                return dict(render=value.expand(3, 20, 20), marker=len(self.calls))

        module = SimpleNamespace(motion=motion, gaussians=gaussians, births=None,
            renderer=Renderer(), background=torch.zeros(3),
            frame_prepared=True, frame_finalized=False, motion_committed=False,
            motion_steps=motion.motion_steps, trainer=SimpleNamespace(global_step=1),
            opacity_update=SimpleNamespace(alphas=lambda: alpha),
            config=SimpleNamespace(effective_motion_observations=True,
                birth=MODULE.BirthConfig(),
                opacity_update=SimpleNamespace(mode='alpha', joint_with_birth=False),
                motion_coarse_to_fine=False, rigid_tree=motion.config, lambda_dssim=.2))
        module.forward = MethodType(MODULE.Module.forward, module)
        return module, alpha

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
        module, alpha = self.module(self.model())
        proposal = module.forward(None, training=True, proposal_only=True)['render']
        gradients = torch.autograd.grad(proposal.sum(),
            [module.motion.gs_translation, alpha], allow_unused=True)
        self.assertTrue(bool(gradients[0].ne(0).all()))
        self.assertIsNone(gradients[1])
        self.assertEqual(module.renderer.calls, [False])
        self.assertTrue(module.renderer.collect_contributions)
        self.assertIs(module.renderer.gaussians, module.gaussians)
        for name in ('rigid_xyz', 'rigid_rotation', 'rigid_alpha', 'rigid_opacity'):
            self.assertIsNone(getattr(module.gaussians, name))
        with self.assertRaises(ValueError):
            module.forward(None, proposal_only=True)

    def test_training_combines_losses_and_returns_only_the_actual_field_package(self):
        module, alpha = self.module(self.model(weight=.7))
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
        expected_gradients = torch.autograd.grad(main_loss, [alpha])
        gradients = torch.autograd.grad(loss, [alpha])
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
            module, _ = self.module(restored)
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
        module, _ = self.module(restored)
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


# ----- test_hierarchical_private -----
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


# ----- test_fixed_gate_residuals -----
class FixedGateResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, refit=0):
        template = StochasticGateFeedbackTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.reparameterize_after_gate_freeze = True
        cfg.amplitude_refit_steps = refit
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def open_all(model, freeze):
        with torch.no_grad():
            model.gate_logits[model.split_eligible] = 2.
        model._select_frontier(freeze=freeze)

    def test_conversion_preserves_stream_and_pose_while_shared_image_gradients_return(self):
        model = self.model()
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .0101
            model.gs_translation[:, 0] = .0304
        before = codec_fixtures.JointImageRateTests.packet(model)
        pose = model(70)[0].detach().clone()
        model.last_step = 70
        self.open_all(model, freeze=True)
        self.assertFalse(model.private_full_motion)
        self.assertEqual(before, codec_fixtures.JointImageRateTests.packet(model))
        xyz, _ = model(71, training=True)
        torch.testing.assert_close(xyz, pose, rtol=0, atol=0)
        target = model.base_xyz.clone()
        target[:, 0] += .05
        region, point = torch.autograd.grad((xyz - target).square().mean(),
            [model.region_translation, model.gs_translation])
        self.assertGreater(float(region[leaves].abs().sum()), 0.)
        self.assertGreater(float(point.abs().sum()), 0.)

    def test_zero_residual_points_can_move_by_their_shared_region(self):
        model = self.model()
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .02
            model.gs_translation[:, 0] = .02
        model.last_step = 70
        self.open_all(model, freeze=True)
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))
        model.config.region_translation_lr = .004
        model.config.gs_translation_lr = 0.
        model.config.rate_loss_weight = model.config.relative_motion_loss_weight = 0.
        target = model.base_xyz.clone()
        target[:, 0] += .04
        xyz, _ = model(71, training=True)
        initial = float((xyz - target).square().mean())
        (xyz - target).square().mean().backward()
        model.optimizer_step()
        self.assertLess(float((model(72)[0] - target).square().mean()), initial)
        self.assertTrue(bool(model._values(hard=True)[2].eq(0).all()))

    def test_refit_only_changes_existing_blocks_and_keeps_replay_exact(self):
        model = self.model(refit=20)
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .02
            model.gs_translation[:, 0] = .02
            model.gs_translation[0, 0] += .004
        model.last_step = 70
        self.open_all(model, freeze=True)
        before = model(80)[0].detach().clone()
        pose, _ = model(81, training=True)
        torch.testing.assert_close(pose, before, rtol=0, atol=0)
        self.assertTrue(model.amplitude_refit_active)
        mask = model._amplitude_refit_support['gs_translation'].clone()
        self.assertEqual(int(mask.sum()), 1)
        self.assertEqual(float(model.regularization_loss(81)), 0.)
        opened = model.opened.clone()
        target = model.base_xyz.clone()
        target[:, 0] += .04
        for step in range(81, 86):
            pose, _ = model(step, training=True)
            (pose - target).square().mean().backward()
            model.optimizer_step()
            self.assertTrue(torch.equal(opened, model.opened))
            self.assertFalse(bool(model._values(hard=True)[2][~mask].ne(0).any()))
        state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(86)[0], rtol=0, atol=0)

    def test_restoration_keeps_each_stage_coordinate_system_and_next_update(self):
        for step in (20, 70, 85):
            model = self.model(refit=20)
            if step == 85:
                model.last_step = 70
                self.open_all(model, freeze=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            xyz, _ = model(step, training=True)
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            packet = copy.deepcopy(model.capture())
            self.assertEqual(packet['private_full_motion_active'], step < 70)
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(packet)
            self.assertEqual(restored.private_full_motion, model.private_full_motion)
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            broken = copy.deepcopy(packet)
            broken['private_full_motion_active'] = not packet['private_full_motion_active']
            with self.assertRaises(ValueError):
                SpatialGatedMotion(copy.deepcopy(model.config)).restore(broken)

    def test_next_frame_restarts_complete_proposals_with_birth_lineage(self):
        model = self.model()
        self.open_all(model, freeze=False)
        leaves = model.spatial_paths[:, -1].unique()
        with torch.no_grad():
            model.region_translation[leaves, 0] = .01
            model.gs_translation[:, 0] = .03
        model.last_step = 70
        self.open_all(model, freeze=True)
        full = model._full_motion_symbols()[0].clone()
        helper = lineage_fixtures.JointSpatialMotionTests()
        state = helper.commit(model, births=model.base_xyz[:1], sources=torch.tensor([0]))
        model.setup(state.xyz, state.rotation, model.extent, frame_index=2)
        self.assertTrue(model.private_full_motion)
        proposal = torch.round(model.gs_translation / model.gs_steps[:3])
        torch.testing.assert_close(proposal, torch.cat((full, full[:1])), rtol=0, atol=0)

    def test_configuration_requires_a_matching_proposal_stage(self):
        model = self.model(refit=20)
        for field, value in (('private_full_motion', False), ('image_relaxation', 'soft_anneal')):
            cfg = copy.deepcopy(model.config)
            setattr(cfg.spatial_gate, field, value)
            with self.assertRaises(ValueError):
                cfg.validate()


# ----- test_inherited_image_gate -----
class InheritedImageGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, refit=0):
        template = FixedGateResidualTests().model()
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

    def shared_residual_model(self, refit=0):
        template = self.model(refit=refit)
        cfg = copy.deepcopy(template.config)
        gate = cfg.spatial_gate
        gate.hierarchical_private = gate.private_full_motion = False
        gate.reparameterize_after_gate_freeze = False
        gate.proposal_image_weight = 0.
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

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

    def test_absolute_migration_preserves_applied_motion_and_unused_children_inherit(self):
        model = FixedGateResidualTests().model()
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

    def test_shared_residual_gates_keep_gs_image_gradients_at_coarse_and_fine_levels(self):
        model = self.shared_residual_model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gs_translation[:, 0] = .004
        model._sampled_gate_step = 1
        model._sampled_gate_noise = torch.full_like(model.gate_logits, -20.)
        coarse, _ = model(1, training=True)
        self.sample_depth2(model)
        fine, _ = model(1, training=True)
        torch.testing.assert_close(coarse, fine, rtol=0, atol=0)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        point, parent = torch.autograd.grad((fine - target).square().mean(),
            [model.gs_translation, model.region_translation])
        self.assertTrue(bool(point[:, 0].ne(0).all()))
        self.assertGreater(float(parent[1:9].abs().sum()), 0.)
        self.assertGreater(float(parent[9:73].abs().sum()), 0.)
        self.assertTrue(bool(model._private_support().all()))

    def test_shared_residual_gate_freeze_and_refit_keep_coarse_exceptions(self):
        model = self.shared_residual_model(refit=20)
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gs_translation[0, 0] = .004
        before = self.packet(model)
        model.last_step = 70
        model._select_frontier(freeze=True)
        self.assertEqual(before, self.packet(model))
        model(81, training=True)
        self.assertTrue(bool(model._amplitude_refit_support['gs_translation'][0]))
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(82)[0], rtol=0, atol=0)
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        self.assertEqual(self.packet(model), self.packet(restored))


# ----- test_predictor_rate_gate -----
class PredictorRateGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        template = InheritedImageGateTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.hierarchical_private = False
        cfg.spatial_gate.proposal_image_weight = 0.
        cfg.spatial_gate.predictor_image_weight = .25
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    @staticmethod
    def packet(model):
        return InheritedImageGateTests.packet(model)

    def test_main_fits_complete_motion_while_predictor_image_trains_region_and_gate(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.region_translation[9:73, 0] = .004
            model.gs_translation[:, 0] = .02
        target = model.base_xyz.clone()
        target[:, 0] += .04
        xyz, _ = model(1, training=True)
        p, u, g = torch.autograd.grad((xyz - target).square().mean(),
            [model.region_translation, model.gs_translation, model.gate_logits], allow_unused=True)
        self.assertIsNone(p)
        self.assertIsNone(g)
        self.assertTrue(bool(u[:, 0].ne(0).all()))
        sampler = model._gate_rng.get_state().clone()
        prediction, _ = model.predictor_pose()
        p, u, g = torch.autograd.grad((prediction - target).square().mean(),
            [model.region_translation, model.gs_translation, model.gate_logits], allow_unused=True)
        self.assertIsNone(u)
        self.assertGreater(float(p.abs().sum()), 0.)
        self.assertGreater(float(g.abs().sum()), 0.)
        self.assertTrue(torch.equal(sampler, model._gate_rng.get_state()))

    def test_gs_coding_residual_changes_gate_rate_gradient_in_the_correct_direction(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[9:73, 0] = .02
        gradients = []
        for increment in (0., .02):
            with torch.no_grad():
                model.gs_translation[:, 0] = increment
            terms = model.loss_terms(10, False)
            gate, point = torch.autograd.grad(terms['rate_loss'],
                [model.gate_logits, model.gs_translation], allow_unused=True)
            gradients.append(gate)
            self.assertIsNone(point)  # GS Adam remains photometric.
        self.assertTrue(bool(gradients[0][1:9].gt(0).all()))
        self.assertTrue(bool(gradients[1][1:9].lt(0).all()))

    def test_hard_rate_and_packet_use_actual_coarse_owner_residuals(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
        terms = model.loss_terms(10, False)
        expected = (model._gs_id_lengths + 7).sum() + model._metadata_bytes()
        self.assertEqual(float(terms['rate_bytes_surrogate']), float(expected))
        with torch.no_grad():
            model.region_translation[1:9, 0] = .02
        self.assertFalse(bool(model._private_symbols()[0].ne(0).any()))
        terms = model.loss_terms(10, False)
        self.assertEqual(float(terms['rate_bytes_surrogate']), 64. + model._metadata_bytes())
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(10)[0], rtol=0, atol=0)

    def test_gate_switch_changes_coding_residual_not_complete_integer_motion(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.region_translation[9:73, 0] = .004
            model.gs_translation[:, 0] = .02
        before = tuple(x.clone() for x in model._full_motion_symbols())
        residual = model._private_symbols()[0].clone()
        with torch.no_grad():
            model.gate_logits[1:9] = 2.
        model._select_frontier(freeze=False)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model._full_motion_symbols())))
        self.assertTrue(bool((model._private_symbols()[0][:, 0] == residual[:, 0] - 2).all()))
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(10)[0], rtol=0, atol=0)

    def test_raw_block_proximal_shrinks_to_selected_predictor_instead_of_zero(self):
        model = self.model()
        mid = len(model.owner) // 2
        with torch.no_grad():
            model.region_translation[1:9, 0] = .02
            model.gs_translation[:, 0] = .022
            model.gs_translation[mid:, 0] = .2
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.zeros_like(p),
                                        exp_avg_sq=torch.full_like(p, 1e-12))
        model.last_step = 20
        model._proximal_gs_blocks()
        torch.testing.assert_close(model.gs_translation[:mid], model._private_origins()[0][:mid],
                                   rtol=0, atol=0)
        self.assertGreater(float(model.gs_translation[mid:, 0].min()), .19)

    def test_fixed_gate_conversion_keeps_packet_and_pose_without_removing_coarse_gs(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gs_translation[:, 0] = .02
        before, pose = self.packet(model), model(69)[0].detach()
        model.last_step = 70
        model._select_frontier(freeze=True)
        self.assertFalse(model.private_full_motion)
        self.assertTrue(bool(model.gs_translation[:, 0].ne(0).all()))
        self.assertEqual(before, self.packet(model))
        torch.testing.assert_close(pose, model(71)[0], rtol=0, atol=0)
        self.assertEqual(model.predictor_image_weight, 0.)
        target = model.base_xyz + .04
        p, u = torch.autograd.grad((model(71, training=True)[0] - target).square().mean(),
                                   [model.region_translation, model.gs_translation])
        self.assertGreater(float(p.abs().sum()), 0.)
        self.assertTrue(bool(u.ne(0).all()))

    def test_module_auxiliary_reuses_views_and_does_not_refit_appearance(self):
        model = self.model()
        module, alpha = ProposalImageSupervisionTests().module(model)
        view = SimpleNamespace(image=torch.full((3, 20, 20), .6))
        loss, metrics, package = MODULE.Module.training_step(module, view, 1, regularization_weight=0.)
        torch.testing.assert_close(loss.detach(),
            metrics['image_loss'] + .25 * metrics['predictor_image_loss'], rtol=0, atol=0)
        self.assertEqual(package['marker'], 1)
        self.assertEqual(module.renderer.calls, [True, False])
        main = module.forward(view, training=True)['render']
        expected = torch.autograd.grad(MODULE.motion_photometric_loss(main, view.image, .2), [alpha])
        actual = torch.autograd.grad(loss, [alpha])
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            module.forward(view, training=True, predictor_only=True, proposal_only=True)

    def test_restore_keeps_each_coordinate_stage_and_future_updates(self):
        for step in (20, 70):
            model = self.model()
            target = model.base_xyz + .02
            def update(m, current):
                xyz, _ = m(current, training=True)
                loss = (xyz - target).square().mean() + m.regularization_loss(current)
                if m.predictor_image_weight:
                    loss = loss + .25 * (m.predictor_pose()[0] - target).square().mean()
                loss.backward()
                m.optimizer_step()
            update(model, step)
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(copy.deepcopy(model.capture()))
            self.assertEqual(restored.private_full_motion, step < 70)
            update(model, step+1)
            update(restored, step+1)
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(self.packet(model), self.packet(restored))


# ----- test_predictor_recode_gate -----
class PredictorRecodeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        template = PredictorRateGateTests().model()
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
        return PredictorRateGateTests.packet(model)

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

    def test_candidate_warmup_fits_motion_before_learning_logits(self):
        model = self.model()
        model.config.spatial_gate.gate_start_step = 20
        model.config.validate()
        gates = model.gate_logits.detach().clone()
        initial = model.region_translation.detach().clone()
        target = model.base_xyz + .04
        for step in (1, 10, 20, 21):
            xyz, _ = model(step, training=True)
            prediction, _ = model.predictor_pose()
            loss = ((xyz - target).square().mean() + .25*(prediction - target).square().mean() +
                    model.regularization_loss(step))
            loss.backward()
            model.optimizer_step()
            if step <= 20:
                torch.testing.assert_close(model.gate_logits, gates, rtol=0, atol=0)
                self.assertEqual(len(model.owner.unique()), 8)
        self.assertTrue(bool(model.region_translation.ne(initial).any()))
        self.assertTrue(bool(model.gate_logits.ne(gates).any()))
        for step in (-1, 70):
            invalid = copy.deepcopy(model.config)
            invalid.spatial_gate.gate_start_step = step
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_shared_discovery_keeps_zero_residuals_and_warms_photo_moments(self):
        model = self.model()
        model.config.spatial_gate.shared_discovery = True
        model.config.spatial_gate.predictor_image_weight = 0.
        model.config.validate()
        target = model.base_xyz + .02
        for step in (1, 69, 70):
            xyz, _ = model(step, training=True)
            loss = (xyz - target).square().mean() + model.regularization_loss(step)
            loss.backward()
            model.optimizer_step()
            self.assertFalse(bool(model.gs_translation.ne(0).any()))
            self.assertIn('exp_avg_sq', model.optimizer.state[model.gs_translation])
        self.assertTrue(model._hard_gate_frozen)
        state_step = model.optimizer.state[model.gs_translation]['step'].clone()
        self.assertFalse(bool(model.optimizer.state[model.gs_translation]['exp_avg'].ne(0).any()))
        self.assertTrue(bool(model.optimizer.state[model.gs_translation]['exp_avg_sq'].gt(0).any()))
        self.assertGreater(model.optimizer.param_groups[2]['lr'], 0.)
        # Isolate release from the separate sparsity penalty, which may
        # legitimately keep this simple constant-motion residual at zero.
        model.config.relative_motion_loss_weight = 0.
        xyz, _ = model(71, training=True)
        ((xyz - target).square().mean() + model.regularization_loss(71)).backward()
        model.optimizer_step()
        self.assertTrue(bool(model.gs_translation.ne(0).any()))
        self.assertEqual(model.optimizer.state[model.gs_translation]['step'], state_step + 1)
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(72)[0], rtol=0, atol=0)

    def test_shared_discovery_gate_changes_the_shared_field_without_creating_gs_updates(self):
        model = self.model()
        model.config.spatial_gate.shared_discovery = True
        model.config.spatial_gate.predictor_image_weight = 0.
        model.config.validate()
        with torch.no_grad():
            model.region_translation[9:73, 0] = .02
            model.gate_logits[1:9] = 2.
        before = model._full_motion_symbols()[0].clone()
        model._select_frontier(freeze=False)
        self.assertTrue(bool(model._full_motion_symbols()[0].ne(before).any()))
        self.assertFalse(bool(model.gs_translation.ne(0).any()))
        terms = model.loss_terms(10, False)
        self.assertEqual(float(terms['relative_motion_loss']), 0.)
        decoded = decode_motion(self.packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model(11)[0], rtol=0, atol=0)


# ----- test_predictor_rd -----
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


# ----- test_soft_gate_feedback -----
class SoftGateFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, relaxation='soft_anneal', private=False):
        template = fixtures.JointSpatialMotionTests().model(initial_logit=-.2)
        cfg = copy.deepcopy(template.config)
        cfg.rotation_updates = False
        cfg.proximal_gs_regularization = True
        cfg.differentiable_rate_support = True
        gate = cfg.spatial_gate
        gate.joint_image_gradients = True
        gate.hierarchical_private = private
        gate.image_relaxation = relaxation
        gate.temperature = 3.
        gate.final_temperature = .3
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    def test_closed_candidate_gets_feedback_when_its_prediction_overshoots(self):
        gradients = {}
        for relaxation in ('hard_ste', 'soft_anneal'):
            model = self.model(relaxation)
            node = int(model.spatial_paths[0, 1])
            members = model.spatial_paths[:, 1].eq(node)
            target = model.base_xyz.clone()
            target[members, 0] += .02
            gradients[relaxation] = []
            for displacement in (.02, .2):
                with torch.no_grad():
                    model.region_translation.zero_()
                    model.region_translation[node, 0] = displacement
                xyz, _ = model(1, training=True)
                loss = (xyz - target).square().mean()
                grad = torch.autograd.grad(loss, model.region_translation)[0][node, 0]
                gradients[relaxation].append(float(grad))
        self.assertEqual(*gradients['hard_ste'])
        self.assertLess(gradients['soft_anneal'][0], 0.)
        self.assertGreater(gradients['soft_anneal'][1], 0.)

    def test_private_candidate_also_receives_its_own_prediction_error(self):
        model = self.model(private=True)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        gradients = []
        for displacement in (.02, .2):
            with torch.no_grad():
                model.gs_translation[:, 0] = displacement
            xyz, _ = model(1, training=True)
            grad = torch.autograd.grad((xyz - target).square().mean(), model.gs_translation)[0]
            gradients.append(float(grad[:, 0].sum()))
        self.assertLess(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)

    def test_evaluation_and_packet_stay_discrete_while_training_is_relaxed(self):
        for private in (False, True):
            with self.subTest(private=private):
                model = self.model(private=private)
                with torch.no_grad():
                    model.region_translation[9:, 0] = .02
                    model.gs_translation[:, 1] = .02
                soft, _ = model(10, training=True)
                hard, _ = model(10, training=False)
                self.assertGreater(float((soft - hard).abs().max()), 0.)
                state = decode_motion(codec_fixtures.JointImageRateTests.packet(model),
                                      model.frame_reference_state)
                torch.testing.assert_close(state.xyz, hard, rtol=0, atol=0)
                model._select_frontier(freeze=True)
                fitted, _ = model(71, training=True)
                state = decode_motion(codec_fixtures.JointImageRateTests.packet(model),
                                      model.frame_reference_state)
                torch.testing.assert_close(state.xyz, fitted, rtol=0, atol=0)

    def test_temperature_anneals_without_changing_discrete_threshold(self):
        model = self.model()
        temperatures = []
        owners = []
        for step in (1, 20, 40, 70):
            model(step, training=True)
            temperatures.append(model._gate_temperature())
            model._select_frontier(freeze=False)
            owners.append(model.owner.clone())
        self.assertEqual(temperatures[0], 3.)
        self.assertAlmostEqual(temperatures[-1], .3)
        self.assertTrue(all(a > b for a, b in zip(temperatures, temperatures[1:])))
        self.assertTrue(all(torch.equal(owners[0], owner) for owner in owners[1:]))

    def test_resume_preserves_soft_stage_and_fixed_stage_moments(self):
        for step in (20, 70):
            with self.subTest(step=step):
                model = self.model()
                xyz, _ = model(step, training=True)
                loss = (xyz - model.base_xyz - .02).square().mean()
                (loss + model.regularization_loss(step)).backward()
                model.optimizer_step()
                self.assertEqual(model._hard_gate_frozen, step == 70)
                if step == 70:
                    self.assertNotIn(model.region_translation, model.optimizer.state)
                    self.assertNotIn(model.gs_translation, model.optimizer.state)
                packet = copy.deepcopy(model.capture())
                restored = SpatialGatedMotion(copy.deepcopy(model.config))
                restored.restore(packet)
                self.assertEqual(model._gate_temperature(), restored._gate_temperature())
                for current in (model, restored):
                    xyz, _ = current(step + 1, training=True)
                    loss = (xyz - current.base_xyz - .02).square().mean()
                    (loss + current.regularization_loss(step + 1)).backward()
                    current.optimizer_step()
                for name in ('region_translation', 'gs_translation', 'gate_logits'):
                    torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
                incompatible = copy.deepcopy(model.config)
                incompatible.spatial_gate.image_relaxation = 'hard_ste'
                with self.assertRaises(ValueError):
                    SpatialGatedMotion(incompatible).restore(packet)

    def test_relaxation_configuration_rejects_incompatible_modes(self):
        model = self.model()
        for temperature in (0., -1., float('nan'), 4., True):
            cfg = copy.deepcopy(model.config)
            cfg.spatial_gate.final_temperature = temperature
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                cfg.validate()
        cfg = copy.deepcopy(model.config)
        cfg.spatial_gate.joint_image_gradients = False
        with self.assertRaises(ValueError):
            cfg.validate()


# ----- test_stochastic_shared_residuals -----
class StochasticSharedResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, refit=0):
        template = StochasticGateFeedbackTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.hierarchical_private = False
        cfg.spatial_gate.private_full_motion = False
        cfg.amplitude_refit_steps = refit
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    def test_coarse_frontier_keeps_a_real_image_derivative_for_every_gs_residual(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .004
        self.assertEqual(len(model.owner.unique()), 8)
        self.assertTrue(bool(model._private_support().all()))
        xyz, _ = model(1, training=True)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        gradient = torch.autograd.grad((xyz - target).square().mean(), model.gs_translation)[0]
        self.assertTrue(bool(gradient[:, 0].ne(0).all()))
        self.assertTrue(bool(model._values(hard=True)[2][:, 0].ne(0).all()))

    def test_sampled_shared_region_sees_its_own_prediction_with_a_private_correction_present(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .004
        model(1, training=True)
        draw = model.gate_logits / model._gate_temperature() + model._sampled_gate_noise
        weights = model._path_weights((draw.ge(0) & model.split_eligible).to(model.gate_logits))
        row, layer = weights.nonzero()[0].tolist()
        node = int(model.spatial_paths[row, layer])
        members = model.spatial_paths[:, layer].eq(node)
        target = model.base_xyz.clone()
        target[:, 0] += .004
        target[members, 0] += .02
        gradients = []
        for displacement in (.01, .2):
            with torch.no_grad():
                model.region_translation.zero_()
                model.region_translation[node, 0] = displacement
            xyz, _ = model(1, training=True)
            gradient = torch.autograd.grad((xyz - target).square().mean(), model.region_translation)[0]
            gradients.append(float(gradient[node, 0]))
        self.assertLess(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)

    def test_freezing_keeps_all_gs_corrections_resets_moments_and_decodes_exactly(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .02
            model.region_translation[9:, 1] = .004
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
                                        exp_avg_sq=torch.ones_like(p))
        before = model.gs_translation.detach().clone()
        model._select_frontier(freeze=True)
        torch.testing.assert_close(model.gs_translation, before, rtol=0, atol=0)
        self.assertNotIn(p, model.optimizer.state)
        state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(71, training=True)[0], rtol=0, atol=0)
        self.assertEqual(model.capture()['regional_image_fitting'], 'joint_image_residual_stochastic_v1')

    def test_sampled_fixed_and_refit_stages_resume_and_retire_exactly(self):
        for step in (20, 70, 85):
            model = self.model(refit=20)
            if step == 85:
                model.last_step = 70
                model._select_frontier(freeze=True)
            target = model.base_xyz.clone()
            target[:, 0] += .02
            xyz, _ = model(step, training=True)
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            restored.restore(copy.deepcopy(model.capture()))
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
                current.retain_survivors(torch.arange(0, len(current.owner), 2))
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
            torch.testing.assert_close(state.xyz, model(step + 1)[0], rtol=0, atol=0)

    def test_gates_can_learn_to_use_a_helpful_shared_field_without_a_private_capacity_rule(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[9:, 0] = .02
        target = model.base_xyz.clone()
        target[:, 0] += .02
        initial = float((model(1)[0] - target).square().mean())
        optimizer = torch.optim.Adam([model.gate_logits], lr=.3)
        for step in range(1, 61):
            optimizer.zero_grad()
            xyz, _ = model(step, training=True)
            gradient = torch.autograd.grad((xyz - target).square().mean(), model.gate_logits)[0]
            model.gate_logits.grad = gradient
            optimizer.step()
            model._select_frontier(freeze=False)
        final = float((model(60)[0] - target).square().mean())
        self.assertLess(final, initial * .1)
        self.assertTrue(bool(model.opened[1:9].any()))


# ----- test_shared_image_recode -----
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


# ----- test_motion_image_schedule -----
class MotionImageScheduleTests(unittest.TestCase):
    def test_rmse_follows_psnr_error_order_when_l1_prefers_a_large_outlier(self):
        target = torch.zeros(3, 24, 24)
        outlier = target.clone()
        outlier[:, 12, 12] = .3
        diffuse = target + .005
        self.assertLess(float(l1_loss(outlier, target)), float(l1_loss(diffuse, target)))
        self.assertGreater(float((outlier - target).square().mean()),
                           float((diffuse - target).square().mean()))
        self.assertGreater(float(MODULE.motion_photometric_loss(outlier, target, 0., metric='rmse')),
                           float(MODULE.motion_photometric_loss(diffuse, target, 0., metric='rmse')))

    def test_rmse_gives_weak_pixel_errors_less_gradient_than_large_errors(self):
        image = torch.full((3, 24, 24), .001)
        image[:, 0, 0] = .1
        image.requires_grad_()
        target = torch.zeros_like(image)
        loss = MODULE.motion_photometric_loss(image, target, 0., metric='rmse')
        grad = torch.autograd.grad(loss, image)[0]
        self.assertAlmostEqual(float(grad[0, 0, 0] / grad[0, 1, 1]), 100., places=4)
        l1_grad = torch.autograd.grad(l1_loss(image, target), image)[0]
        self.assertEqual(float(l1_grad[0, 0, 0]), float(l1_grad[0, 1, 1]))

    def test_rmse_exact_match_has_zero_finite_gradient_and_rejects_unknown_metric(self):
        image = torch.full((3, 24, 24), .37, requires_grad=True)
        loss = MODULE.motion_photometric_loss(image, image.detach(), 0., metric='rmse')
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(image.grad).all()))
        self.assertTrue(bool(image.grad.eq(0).all()))
        with self.assertRaises(ValueError):
            MODULE.motion_photometric_loss(image, image.detach(), .2, metric='unknown')

    def test_module_uses_rmse_for_both_main_and_auxiliary_proposal_images(self):
        helper = ProposalImageSupervisionTests()
        module, _ = helper.module(helper.model())
        module.config.photometric_loss = 'rmse'
        module.config.lambda_dssim = 0.
        target = torch.linspace(0., 1., 3 * 20 * 20).reshape(3, 20, 20)
        loss, metrics, package = MODULE.Module.training_step(module,
            SimpleNamespace(image=target), 1, regularization_weight=0.)
        expected = ((package['render'].double() - target.double()).square().mean() + 1e-8).sqrt() - 1e-4
        torch.testing.assert_close(metrics['image_loss'].double(), expected.detach(), rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(loss.double(), expected * 2, rtol=1e-6, atol=1e-8)
        self.assertEqual(MODULE.ModuleConfig().photometric_loss, 'l1')

    def test_coarse_tracking_ends_before_coding_and_birth_statistics(self):
        for step, factor in ((1, 4), (20, 4), (21, 2), (40, 2), (41, 1),
                             (60, 1), (71, 1), (100, 1)):
            self.assertEqual(MODULE.motion_image_factor(step, 100, True, True), factor)
        for step in (1, 20, 100, 150, 201, 300):
            self.assertEqual(MODULE.motion_image_factor(step, 100, True, False), 1)
            self.assertEqual(MODULE.motion_image_factor(step, 100, False, True), 1)

    def test_full_resolution_loss_and_gradients_are_unchanged(self):
        generator = torch.Generator().manual_seed(81)
        image = torch.rand(3, 24, 25, generator=generator, requires_grad=True)
        target = torch.rand(3, 24, 25, generator=generator)
        old = .8 * l1_loss(image, target) + .2 * (1 - MODULE.ssim(image, target))
        new = MODULE.motion_photometric_loss(image, target, .2)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(new, image, retain_graph=True)[0],
                                   torch.autograd.grad(old, image)[0], rtol=0, atol=0)

    def test_cached_ssim_matches_loss_and_gradients_across_resolutions(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                image = torch.rand(3, 25, 27, device=device, requires_grad=True)
                target = torch.rand_like(image)
                cache = MODULE._TrainingSSIMCache()
                for factor in (1, 2, 4):
                    expected = MODULE.motion_photometric_loss(image, target, .2, factor)
                    actual = MODULE.motion_photometric_loss(image, target, .2, factor, cache=cache)
                    expected_grad = torch.autograd.grad(expected, image)[0]
                    actual_grad = torch.autograd.grad(actual, image)[0]
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)
                first = cache.prepare(image, target, image.shape[-2:])
                second = cache.prepare(image, target, image.shape[-2:])
                self.assertIs(first[1], second[1])
                target.add_(.1)
                changed = cache.prepare(image, target, image.shape[-2:])
                self.assertIsNot(first[1], changed[1])
                torch.testing.assert_close(
                    MODULE.motion_photometric_loss(image, target, .2, cache=cache),
                    MODULE.motion_photometric_loss(image, target, .2), rtol=1e-6, atol=1e-7)
                cache.clear()
                self.assertEqual(cache._target_bytes, 0)
                self.assertFalse(cache._targets)

    def test_cached_ssim_keeps_target_gradients_and_honours_budget(self):
        image = torch.rand(3, 17, 19, requires_grad=True)
        target = torch.rand_like(image, requires_grad=True)
        cache = MODULE._TrainingSSIMCache(max_bytes=1)
        expected = MODULE.motion_photometric_loss(image, target, .2)
        actual = MODULE.motion_photometric_loss(image, target, .2, cache=cache)
        expected_grads = torch.autograd.grad(expected, (image, target))
        actual_grads = torch.autograd.grad(actual, (image, target))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        MODULE.motion_photometric_loss(image, target.detach(), .2, cache=cache)
        self.assertFalse(cache._targets)
        if torch.cuda.is_available():
            gpu_image = image.detach().cuda()
            gpu_target = target.detach().cuda()
            limited = MODULE._TrainingSSIMCache()
            with patch.object(torch.cuda, 'mem_get_info', return_value=(0, 1024)):
                self.assertIsNone(limited.prepare(gpu_image, gpu_target, gpu_image.shape[-2:]))
            self.assertFalse(limited._targets)

    def test_cached_ssim_falls_back_during_autocast(self):
        image = torch.rand(3, 19, 21, requires_grad=True)
        target = torch.rand_like(image)
        cache = MODULE._TrainingSSIMCache()
        normal = MODULE.motion_photometric_loss(image, target, .2, cache=cache)
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            expected = MODULE.motion_photometric_loss(image, target, .2)
            actual = MODULE.motion_photometric_loss(image, target, .2, cache=cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, image)[0],
                                   torch.autograd.grad(expected, image)[0], rtol=1e-5, atol=1e-9)
        self.assertEqual(len(cache._targets), 1)
        torch.testing.assert_close(normal, MODULE.motion_photometric_loss(image, target, .2, cache=cache),
                                   rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA separable SSIM check')
    def test_cuda_cached_ssim_on_flat_and_nearly_matching_images(self):
        torch.manual_seed(91)
        for value in (0., .5, 1.):
            target = torch.full((3, 25, 27), value, device='cuda')
            image = (target + torch.randn_like(target) * .01).clamp(0, 1).requires_grad_()
            expected = MODULE.motion_photometric_loss(image, target, .2)
            actual = MODULE.motion_photometric_loss(image, target, .2, cache=MODULE._TrainingSSIMCache())
            expected_grad = torch.autograd.grad(expected, image)[0]
            actual_grad = torch.autograd.grad(actual, image)[0]
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-3, atol=1e-6)

    def test_extra_motion_budget_keeps_gate_and_residual_fitting_at_full_resolution(self):
        for budget in (100, 200):
            for step, factor in ((1, 4), (20, 4), (21, 2), (40, 2),
                                 (41, 1), (60, 1), (80, 1), (100, 1), (200, 1)):
                self.assertEqual(MODULE.motion_image_factor(step, budget, True, True,
                                                            bootstrap_steps=40), factor)
        # A different bootstrap boundary remains tied to that phase, too.
        self.assertEqual(MODULE.motion_image_factor(35, 200, True, True, 35), 2)
        self.assertEqual(MODULE.motion_image_factor(36, 200, True, True, 35), 1)

    def test_resume_rejects_a_different_image_schedule_but_accepts_equivalent_legacy(self):
        compatible = MODULE.motion_image_schedule_compatible
        self.assertTrue(compatible(None, True, 100, 40))
        self.assertFalse(compatible(None, True, 200, 40))
        self.assertTrue(compatible(MODULE.MOTION_IMAGE_SCHEDULE, True, 200, 40))
        self.assertTrue(compatible(None, False, 200, 40))
        self.assertFalse(compatible('unknown_schedule', True, 200, 40))

    def test_coarse_loss_keeps_gradient_to_original_full_image(self):
        image = torch.rand(3, 25, 27, requires_grad=True)
        target = torch.rand_like(image)
        MODULE.motion_photometric_loss(image, target, .2, 4).backward()
        self.assertEqual(image.grad.shape, image.shape)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'Native rasterizer capture-range check')
    def test_coarse_image_provides_direction_for_disjoint_gaussian_motion(self):
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
        from libgs.data.utils.graphics import getProjectionMatrix
        settings = GaussianRasterizationSettings(image_height=64, image_width=64,
            tanfovx=math.tan(.5), tanfovy=math.tan(.5), bg=torch.zeros(3, device='cuda'),
            scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'),
            projmatrix=getProjectionMatrix(.01, 100, 1., 1.).T.cuda(), sh_degree=0,
            campos=torch.zeros(3, device='cuda'), prefiltered=False, debug=False)
        def render(x):
            xyz = torch.tensor([[x, 0., 2.]], device='cuda', requires_grad=True)
            image, _ = GaussianRasterizer(settings)(means3D=xyz, means2D=torch.zeros_like(xyz),
                colors_precomp=torch.ones(1, 3, device='cuda'), shs=None,
                opacities=torch.tensor([[.7]], device='cuda'),
                scales=torch.tensor([[.05]*3], device='cuda'),
                rotations=torch.tensor([[1., 0., 0., 0.]], device='cuda'), cov3D_precomp=None)
            return xyz, image
        _, target = render(.75)
        target = target.detach()
        gradients = []
        for factor in (1, 4):
            xyz, image = render(0.)
            MODULE.motion_photometric_loss(image, target, .2, factor).backward()
            gradients.append(float(xyz.grad[0, 0]))
        self.assertLess(gradients[1], -1e-4)
        self.assertGreater(abs(gradients[1]), 100 * abs(gradients[0]))


# ----- test_motion_view_updates -----
TRAINER = importlib.import_module('pipeline.trainer')


class MotionViewUpdates(unittest.TestCase):
    def test_historical_view_counts_resolve_to_one(self):
        config = TRAINER.TrainerConfig(motion_views_per_step=4, birth_views_per_step=2)
        config.use_single_training_view()
        self.assertEqual((config.motion_views_per_step, config.birth_views_per_step), (1, 1))
        for invalid in (0, -1, True):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                TRAINER.TrainerConfig(motion_views_per_step=invalid).use_single_training_view()

    def test_saved_run_config_records_the_actual_single_view(self):
        import yaml
        from pipeline.entry import Config, Pipeline
        with TemporaryDirectory() as directory:
            config = Config(gui_enabled=False,
                trainer=TRAINER.TrainerConfig(motion_views_per_step=4, birth_views_per_step=4))
            def setup(pipeline):
                pipeline.trainer = SimpleNamespace(output_dir=Path(directory), is_global_zero=True)
            with patch('pipeline.entry.safe_state'), patch.object(Pipeline, 'setup_trainer_and_modules', setup):
                Pipeline(config)
            recorded = yaml.safe_load((Path(directory) / 'config.yaml').read_text())
            self.assertEqual(recorded['trainer']['motion_views_per_step'], 1)
            self.assertEqual(recorded['trainer']['birth_views_per_step'], 1)

    def test_birth_camera_schedule_is_random_and_resumable(self):
        names = [f'cam{i:02d}.png' for i in range(1, 9)]
        views = {name: object() for name in names}
        for adaptive, first_step in ((False, 201), (True, 71)):
            with self.subTest(adaptive=adaptive):
                birth = SimpleNamespace(mechanism='residual_transaction',
                    radiance_mode='adaptive_residual_birth' if adaptive else 'free')
                module = SimpleNamespace(config=SimpleNamespace(birth=birth), current_frame=2,
                    datamodule=SimpleNamespace(config=SimpleNamespace(training_view_seed=13)),
                    birth_transaction=SimpleNamespace(fit_names=names), motion_steps=200,
                    transaction_camera_views=lambda: views)
                before = torch.get_rng_state().clone()
                schedule = [MODULE.Module.transaction_training_view(module, step, 0, 1)[1]
                            for step in range(first_step, first_step + 2 * len(names))]
                self.assertEqual(set(schedule[:len(names)]), set(range(len(names))))
                self.assertEqual(set(schedule[len(names):]), set(range(len(names))))
                self.assertNotEqual(schedule[:len(names)], list(range(len(names))))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(MODULE.Module.transaction_training_view(
                    module, first_step + 3, 0, 1)[1], schedule[3])

    def trainer(self, views=4):
        cfg = SimpleNamespace(birth_steps=100,
            spatial_gate=SimpleNamespace(gate_start_step=60, freeze_step=90))
        trainer = SimpleNamespace(config=SimpleNamespace(motion_views_per_step=views),
            module=SimpleNamespace(config=SimpleNamespace(rigid_tree=cfg),
                                   motion_steps=100, total_steps=200))
        trainer._views_per_step = lambda step: TRAINER.Trainer._views_per_step(trainer, step)
        return trainer

    def test_one_random_view_produces_one_backward_and_optimizer_update(self):
        parameter = torch.tensor(.5, requires_grad=True)
        trainer = self.trainer()
        optimizer = torch.optim.SGD([parameter], lr=.1)
        received = []
        targets = [-3., 1., 2., 4.]
        fetcher = SimpleNamespace(next=Mock(side_effect=iter(targets)))
        regularizer_calls = []
        def train(target, step, regularization_weight=1.):
            regularizer_calls.append(regularization_weight)
            loss = (parameter - target).square() + regularization_weight * .3 * parameter.square()
            return loss, dict(image_loss=float(loss.detach()), psnr=1., gs=2), {
                'render': torch.zeros(5), 'visibility_filter': torch.tensor([True, False]),
                'motion_contribution_filter': torch.tensor([False, True]),
                'viewspace_points': torch.zeros(2, 3)}
        def post(packages):
            received.append((float(parameter.grad), packages))
            optimizer.step()
        trainer.module.training_step = train
        trainer.module.post_training_step = post
        loss, metrics = TRAINER.Trainer._fit_views(trainer, fetcher, 1)
        expected = (.5-targets[0])**2 + .3*.5**2
        self.assertAlmostEqual(loss, expected, places=5)
        self.assertAlmostEqual(received[0][0], 7.3, places=5)
        self.assertAlmostEqual(float(parameter), -.23, places=5)
        fetcher.next.assert_called_once_with()
        self.assertEqual(len(received), 1)
        self.assertIn('render', received[0][1])
        self.assertEqual(metrics['gs'], 2)
        self.assertEqual(regularizer_calls, [1])

    def test_device_metrics_are_returned_as_python_values_after_one_view(self):
        trainer = self.trainer(views=2)
        targets = iter([1., 3.])
        trainer.module.training_step = lambda target, step, **kwargs: (
            torch.tensor(target), dict(image_loss=torch.tensor(target),
                psnr=torch.tensor(target * 10), gs=7), {})
        trainer.module.post_training_step = Mock()
        loss, metrics = TRAINER.Trainer._fit_views(trainer,
            SimpleNamespace(next=targets.__next__), 1)
        self.assertEqual(loss, 1.)
        self.assertEqual(metrics, dict(image_loss=1., psnr=10., gs=7))
        self.assertIsInstance(metrics['image_loss'], float)

    def test_deferred_metrics_preserve_gradients_and_materialized_values(self):
        results = []
        for deferred in (False, True):
            parameter = torch.tensor(.5, requires_grad=True)
            trainer = self.trainer(views=2)
            optimizer = torch.optim.SGD([parameter], lr=.1)
            targets = iter([1., 3., -2., 4., .25, .75])
            def train(target, step, **kwargs):
                loss = (parameter - target).square()
                metrics = dict(image_loss=loss.detach(), psnr=loss.detach() * 2,
                               gs=7, motion_image_factor=1)
                if step == 2:
                    metrics['optional_loss'] = loss.detach() * .1
                return loss, metrics, {}
            def update(_):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            trainer.module.training_step = train
            trainer.module.post_training_step = update
            rows = [TRAINER.Trainer._fit_views(trainer,
                    SimpleNamespace(next=targets.__next__), step, defer_metrics=deferred)
                    for step in range(1, 4)]
            if deferred:
                self.assertTrue(all(torch.is_tensor(loss) and not loss.requires_grad
                                    for loss, _ in rows))
                rows = TRAINER.Trainer._materialize_step_metrics(rows)
            results.append((parameter.detach().clone(), rows))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        self.assertEqual(results[0][1], results[1][1])

    def test_frame_batches_diagnostics_across_resume_tail_without_changing_reports(self):
        class Progress:
            def __init__(self, **kwargs):
                self.n = kwargs['initial']
                self.postfix = []
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def set_postfix(self, values, refresh=False):
                self.postfix.append(values)
            def update(self, count):
                self.n += count

        with TemporaryDirectory() as directory:
            fetcher = object()
            def fit(_, step, *, defer_metrics=False):
                self.assertTrue(defer_metrics)
                return (torch.tensor(step / 7),
                        dict(image_loss=torch.tensor(step / 11),
                             psnr=torch.tensor(step / 13), gs=7))
            module = SimpleNamespace(pre_training_step=Mock(), motion_steps=20,
                total_steps=23, frame_report={}, stage_image_metrics=[],
                config=SimpleNamespace(rigid_tree=SimpleNamespace(
                    spatial_gate=SimpleNamespace(gate_start_step=4, freeze_step=10),
                    amplitude_refit_steps=6)))
            trainer = SimpleNamespace(current_frame=1, global_step=4, num_steps=23,
                config=SimpleNamespace(stage_image_camera=None), module=module,
                output_dir=Path(directory), _make_fetcher=Mock(return_value=fetcher),
                _resume_fetcher=Mock(), _fit_views=fit,
                _materialize_step_metrics=TRAINER.Trainer._materialize_step_metrics,
                validation_loop=Mock(return_value={'test': {'psnr': 34.2}}))
            trainer._checkpoint_due = lambda: trainer.global_step % 7 == 0 or trainer.global_step == 23
            trainer.ckpt_save_path = Path(directory) / 'checkpoint.pth'
            saved_steps = []
            trainer.save_checkpoint = lambda: saved_steps.append(trainer.global_step)
            progress = []
            def make_progress(**kwargs):
                bar = Progress(**kwargs)
                progress.append(bar)
                return bar
            transfers = []
            original_tolist = torch.Tensor.tolist
            def tolist(tensor):
                transfers.append(tensor.numel())
                return original_tolist(tensor)
            with patch.object(TRAINER, 'tqdm', make_progress), \
                    patch.object(torch.Tensor, 'tolist', tolist):
                report = TRAINER.Trainer.frame_training_loop(trainer)
            # Resume at step 5: six rows to the first refresh, ten, then a
            # three-step tail. Each row has the loss and two tensor metrics.
            self.assertEqual(transfers, [18, 30, 9])
            self.assertEqual(saved_steps, [7, 14, 21, 23])
            self.assertEqual(report['train_loss_steps'], 19)
            expected_loss, image_ema, psnr_ema = 0., 0., 0.
            for step in range(5, 24):
                expected_loss += float(torch.tensor(step / 7))
                image_ema = .6 * image_ema + .4 * float(torch.tensor(step / 11))
                psnr_ema = .6 * psnr_ema + .4 * float(torch.tensor(step / 13))
            self.assertEqual(report['mean_train_loss'], expected_loss / 19)
            self.assertEqual(report['final_train_metrics'],
                dict(image_loss=float(torch.tensor(23 / 11)),
                     psnr=float(torch.tensor(23 / 13)), gs=7))
            self.assertEqual(progress[0].postfix[-1],
                dict(image=f'{image_ema:.5f}', psnr=f'{psnr_ema:.2f}', gs=7))
            self.assertEqual(json.loads((Path(directory) / 'frame_summary/frame_001.json').read_text()), report)

    def test_schedule_and_resume_replay_exact_view_consumption(self):
        trainer = self.trainer()
        for step, wanted in [(1,1),(60,1),(61,1),(90,1),(91,1),(100,1),
                             (101,1),(200,1)]:
            self.assertEqual(trainer._views_per_step(step), wanted)
        self.assertEqual(sum(trainer._views_per_step(s) for s in range(1,201)), 200)
        trainer.global_step = 95
        trainer._resume_rng = {'marker': True}
        trainer._restore_rng = Mock()
        fetcher = SimpleNamespace(next=Mock())
        TRAINER.Trainer._resume_fetcher(trainer, fetcher)
        self.assertEqual(fetcher.next.call_count, 95)
        trainer._restore_rng.assert_called_once_with({'marker': True})
        self.assertIsNone(trainer._resume_rng)
        old = self.trainer(views=1)
        self.assertEqual(sum(old._views_per_step(s) for s in range(1,201)), 200)

    def test_all_camera_evidence_and_unscaled_birth_scores_reach_one_update(self):
        captured = []
        model = SimpleNamespace(add_densification_stats=lambda point, mask:
                                captured.append(point.grad.clone()))
        motion = SimpleNamespace(mark_observed=Mock(), optimizer_step=Mock(return_value=[]))
        module = SimpleNamespace(trainer=SimpleNamespace(global_step=99),
            motion_committed=False, motion_steps=100,
            config=SimpleNamespace(effective_motion_observations=True,
                birth=SimpleNamespace(densification_stats_steps=30)),
            gaussians=model, motion=motion, birth_control={})
        packages = []
        for i in range(4):
            points = torch.zeros(2,3,requires_grad=True)
            points.grad = torch.full_like(points, .25)
            packages.append(dict(viewspace_points=points,
                visibility_filter=torch.tensor([True,True]),
                motion_contribution_filter=torch.tensor([i%2==0,i%2==1])))
        MODULE.Module.post_training_step(module, packages)
        self.assertEqual(motion.mark_observed.call_count, 4)
        motion.optimizer_step.assert_called_once()
        self.assertEqual(module.birth_control['motion_observed_samples'], 4)
        self.assertEqual(module.birth_control['motion_frustum_samples'], 8)
        self.assertEqual(len(captured), 4)
        for value in captured:
            torch.testing.assert_close(value, torch.ones_like(value))


if __name__ == '__main__':
    unittest.main()
