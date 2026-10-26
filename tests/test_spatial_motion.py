"""Spatial Motion tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import importlib
import itertools
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from pipeline.model.rigid_residual_tree import (
    RigidTreeConfig,
    _quantized,
    rotation_vector_quaternion,
)
from pipeline.model.rigid_tree_codec import (
    DecodedState,
    compress_motion_packet,
    decode_motion,
    encode_motion,
    uncompressed_motion_packet,
)
from pipeline.model.sparse_integer_codec import (
    MOTION_MAGICS,
    encode_motion_layout,
    pack_layout,
)
from pipeline.model.spatial_gate import (
    SPATIAL_ROUTING_FIELDS,
    spatial_accumulate,
    spatial_owner,
    spatial_route,
    spatial_state_route,
)
from pipeline.model.spatial_gated_motion import SpatialGatedMotion

# ----- test_spatial_gated_motion -----
RIGID = importlib.import_module('pipeline.model.rigid_residual_tree')
SPATIAL = importlib.import_module('pipeline.model.spatial_gated_motion')
CODEC = importlib.import_module('pipeline.model.rigid_tree_codec')
LAYOUT = importlib.import_module('pipeline.model.spatial_gate')


class SpatialGatedMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, repetitions=16, extent=1.):
        coords = torch.cartesian_prod(torch.tensor([.1, .3, .6, .9]),
                                     torch.tensor([.1, .3, .6, .9]),
                                     torch.tensor([.1, .3, .6, .9]))
        xyz = coords.repeat_interleave(repetitions, 0)
        q = torch.zeros(len(xyz), 4); q[:, 0] = 1
        cfg = RIGID.RigidTreeConfig(region_count=9,
            motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
            motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
            rate_loss_weight=.015, relative_motion_loss_weight=0.)
        cfg.spatial_gate.enabled = True
        cfg.spatial_gate.max_depth = 3
        cfg.spatial_gate.min_region_size = 8
        model = SPATIAL.SpatialGatedMotion(cfg)
        model.setup(xyz, q, extent, frame_index=1)
        return model

    @staticmethod
    def representative_cost(symbols, candidate, rows, regional_row, price, scales):
        residual = symbols - candidate
        t, r = residual[:, :3].any(1), residual[:, 3:].any(1)
        rt, rr = candidate[:3].any(), candidate[3:].any()
        byte_count = ((t | r) * rows + 6 * (t.astype(int) + r)).sum()
        byte_count += regional_row * bool(rt or rr) + 6 * (int(rt) + int(rr))
        return price * byte_count + (np.abs(residual) * scales).sum()

    def test_statistical_representative_matches_exhaustive_byte_and_l1_objective(self):
        rng = np.random.default_rng(7)
        candidates = np.array(list(itertools.product(range(-1,2), repeat=6)))
        for repeat in range(12):
            symbols = rng.integers(-1,2,size=(8,6),dtype=np.int32)
            rows = rng.integers(2,5,size=8)
            price = 0. if repeat == 0 else float(rng.uniform(0,1))
            scales = np.zeros(6) if repeat == 1 else rng.uniform(0,1,size=6)
            chosen, actual = SPATIAL._rate_relative_representative(symbols,rows,3,price,scales)
            expected = min(self.representative_cost(symbols,c,rows,3,price,scales)
                           for c in candidates)
            self.assertAlmostEqual(actual,expected,places=10)
            self.assertAlmostEqual(actual,self.representative_cost(
                symbols,chosen,rows,3,price,scales),places=10)

    def test_shared_translation_is_not_lost_when_rotations_have_different_symbols(self):
        symbols = np.zeros((10,6),dtype=np.int32)
        symbols[:8,0] = 20
        symbols[:8,3] = np.arange(1,9)
        rows = np.full(10,3)
        chosen, cost = SPATIAL._rate_relative_representative(
            symbols,rows,3,1.,np.zeros(6))
        self.assertEqual(chosen[0],20)
        old = self.representative_cost(symbols,np.zeros(6,int),rows,3,1.,np.zeros(6))
        self.assertLess(cost,old)

    def test_representative_respects_both_region_and_residual_integer_ranges(self):
        symbols = np.zeros((5,6),dtype=np.int32)
        symbols[:,0] = [45000,45001,45002,45003,45004]
        chosen,_ = SPATIAL._rate_relative_representative(
            symbols,np.full(5,3),3,.1,np.ones(6))
        self.assertTrue((np.abs(chosen)<=32767).all())
        self.assertTrue((np.abs(symbols-chosen)<=32767).all())

    def test_previous_joint_mode_objective_checkpoint_cannot_silently_resume(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet.pop('representative_objective')
        with self.assertRaisesRegex(ValueError,'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_fit_window_keeps_persistent_motion_and_discards_unprofitable_motion(self):
        model = self.model(extent=6.493635559082032)
        model.config.motion_fit_window_steps = 20
        model.config.rate_loss_weight = .03
        model.config.rate_normalization = 'reference'
        model.rate_reference_count = 206237
        model.config.gs_translation_lr = .0006493635559082032
        model.config.motion_xyz_step_extent = model.config.gs_xyz_step_extent = 3e-5
        model.config.validate()
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        wanted = model.gs_translation.new_tensor([.02, .0002])
        for step in range(1, 101):
            model.last_step = step
            model.optimizer.zero_grad(set_to_none=True)
            (.001 * (model.gs_translation[:2, 0] - wanted).square().sum()).backward()
            model.optimizer_step()
        encoded = model._values(hard=True)[2].float() * model.gs_steps[:3]
        self.assertLess(abs(float(encoded[0, 0]) - .02), .002)
        self.assertEqual(float(encoded[1].abs().sum()), 0.)
        rate = .03 * 8 / model.rate_normalization_bytes
        self.assertLess(.001 * (float(encoded[0, 0]) - .02) ** 2 + rate, .001 * .02 ** 2)

    def test_single_moving_point_does_not_drag_a_static_regional_majority(self):
        model = self.model(extent=6.493635559082032)
        model.config.motion_fit_window_steps = 20
        model.config.rate_loss_weight = .03
        model.config.rate_normalization = 'reference'
        model.rate_reference_count = 206237
        model.config.motion_xyz_step_extent = model.config.gs_xyz_step_extent = 3e-5
        model.config.region_translation_lr = .0003246817779541016
        model.config.gs_translation_lr = .0006493635559082032
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        desired = model.base_xyz.clone()
        desired[0,0] += .02
        for step in range(1,101):
            position,_ = model(step,training=True)
            (.001*(position-desired).square().sum()).backward()
            model.optimizer_step()
        final = model.materialized_pose()[0]
        torch.testing.assert_close(final[1:],model.base_xyz[1:],rtol=0,
                                   atol=float(model.gs_steps[0])*1.001)
        self.assertLess(abs(float(final[0,0]-model.base_xyz[0,0])-.02), .002)
        # Finite-step quantized fitting may finish one bin from the static
        # majority optimum; a zero initialization is no longer a hard freeze.

    def test_confirmed_shared_motion_can_fit_before_births_with_static_regions_exact(self):
        model = self.model()
        model.config.rate_loss_weight = 0.
        model.config.relative_motion_loss_weight = .0001
        model.config.gs_translation_lr = 1e-12
        model.observations.fill_(1)
        node = int(model.owner[0])
        members = model.owner == node
        with torch.no_grad():
            model.gs_translation[members, 0] = .005
        full = model._full_motion_symbols()
        model._initialize_candidates(full_motion=full)
        model._select_frontier(freeze=True, full_motion=full)
        initial = model.materialized_pose()[0].detach().clone()
        target = initial.clone()
        target[members, 0] += .01
        for step in range(41, 60):
            position, _ = model(step, training=True)
            (position - target).square().mean().backward()
            model.optimizer_step()
        final = model.materialized_pose()[0]
        self.assertGreater(float(model.region_translation[node, 0]), .01)
        self.assertLess(float((final[members] - target[members]).square().mean()),
                        float((initial[members] - target[members]).square().mean()) / 10)
        torch.testing.assert_close(final[~members], initial[~members], rtol=0, atol=0)

    def test_unselected_regional_predictors_keep_values_despite_old_adam_momentum(self):
        model = self.model()
        model.observations.fill_(1)
        with torch.no_grad():
            model.gs_translation[:, 0] = .005
        full = model._full_motion_symbols()
        model._initialize_candidates(full_motion=full)
        for parameter in (model.region_translation, model.region_rotation):
            parameter.grad = torch.ones_like(parameter)
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        model._select_frontier(freeze=True, full_motion=full)
        selected = torch.bincount(model.owner, minlength=len(model.parents)).gt(0)
        before = model.region_translation.detach().clone()
        position, _ = model(45, training=True)
        (position - model.base_xyz - .02).square().mean().backward()
        model.optimizer_step()
        torch.testing.assert_close(model.region_translation[~selected], before[~selected], rtol=0, atol=0)
        self.assertFalse(model.optimizer.state[model.region_translation]['exp_avg'][~selected].ne(0).any())

    def test_restored_bootstrap_keeps_confirmed_support_through_quantized_zero(self):
        model = self.model()
        model.observations.fill_(1)
        node = int(model.owner[0])
        members = model.owner == node
        with torch.no_grad():
            model.gs_translation[members, 0] = .005
        full = model._full_motion_symbols()
        model._initialize_candidates(full_motion=full)
        model._select_frontier(freeze=True, full_motion=full)
        with torch.no_grad():
            model.region_translation[node].zero_()
        restored = SPATIAL.SpatialGatedMotion(model.config)
        restored.restore(copy.deepcopy(model.capture()))
        self.assertTrue(restored._regional_fit_support[node])
        position, _ = restored(45, training=True)
        target = restored.base_xyz.clone()
        target[members, 0] += .01
        (position - target).square().mean().backward()
        restored.optimizer_step()
        self.assertGreater(float(restored.region_translation[node, 0]), 0.)

    def test_old_late_only_regional_fitting_checkpoint_is_replay_only(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet.pop('regional_image_fitting')
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_shared_field_gate_learns_image_resolution_without_individual_residuals(self):
        model = self.model()
        model.config.rate_normalization = 'reference'
        model.rate_reference_count = 206237
        model.config.spatial_gate.gate_start_step = model.config.spatial_gate.start_step
        model.config.spatial_gate.freeze_step = 90
        model.config.spatial_gate.temperature = 3.
        model.observations.fill_(1)
        parent = int(model.owner[0])
        members = model.owner == parent
        target = model.base_xyz.clone()
        children = torch.where((model.parents == parent) & model.node_counts.gt(0))[0]
        with torch.no_grad():
            model.gate_logits.fill_(-2.)
            for index, child in enumerate(children):
                rows = model.spatial_paths[:,1] == child
                sign = 1 if index % 2 else -1
                model.gs_translation[rows,0] = sign*.018
                target[rows,0] += sign*.02
        model.last_step = model.config.spatial_gate.start_step
        model.optimizer_step()
        before = float((model.materialized_pose()[0]-target).square().mean())
        photo_gate_gradients = []
        for step in range(21,71):
            position,_ = model(step,training=True)
            (.001*(position-target).square().sum()).backward()
            photo_gate_gradients.append(float(model.gate_logits.grad[parent]))
            self.assertIsNone(model.gs_translation.grad)
            model.optimizer_step()
        after = model.materialized_pose()[0]
        self.assertTrue(model.opened[parent])
        self.assertTrue(any(value<0 for value in photo_gate_gradients))
        self.assertLess(float((after-target).square().mean()),before/100)
        self.assertFalse(model.gs_translation.detach().ne(0).any())
        torch.testing.assert_close(after[~members],model.base_xyz[~members],rtol=0,atol=0)

    def test_shared_field_forward_uses_one_region_not_a_soft_motion_mixture(self):
        model = self.model()
        model.observations.fill_(1)
        with torch.no_grad():model.gs_translation[:,0]=.005
        model.last_step=model.config.spatial_gate.start_step
        model.optimizer_step()
        parent=int(model.owner[0]);children=model.parents==parent
        with torch.no_grad():
            model.region_translation[parent,0]=.01
            model.region_translation[children,0]=-.02
        position,_=model(30,training=True)
        rows=model.owner==parent
        torch.testing.assert_close(position[rows,0]-model.base_xyz[rows,0],
                                   torch.full((int(rows.sum()),),.01),rtol=0,atol=1e-7)
        self.assertEqual(float(model.loss_terms(30,False)['relative_motion_loss']),0.)
        with torch.no_grad():model.gate_logits[parent]=2.
        model._select_frontier(freeze=False)
        position,_=model(31,training=True)
        torch.testing.assert_close(position[rows,0]-model.base_xyz[rows,0],
                                   torch.full((int(rows.sum()),),-.02),rtol=0,atol=1e-7)

    def test_shared_field_checkpoint_keeps_region_motion_and_point_freeze(self):
        model=self.model();model.observations.fill_(1)
        with torch.no_grad():model.gs_translation[0,0]=.02
        model.last_step=model.config.spatial_gate.start_step
        model.optimizer_step()
        restored=SPATIAL.SpatialGatedMotion(model.config)
        restored.restore(copy.deepcopy(model.capture()))
        self.assertTrue(restored._shared_field_active)
        self.assertFalse(restored.gs_translation.requires_grad)
        for left,right in zip(model.materialized_pose(),restored.materialized_pose()):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        packet=copy.deepcopy(model.capture());packet.pop('motion_schedule')
        with self.assertRaisesRegex(ValueError,'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)


    def test_starting_residual_fit_preserves_learned_region_field(self):
        model = self.model()
        model.observations.fill_(1)
        with torch.no_grad():model.gs_translation[:,0] = .018
        model.last_step = model.config.spatial_gate.start_step
        model.optimizer_step()
        # The learned field has improved after the proposal stage.
        with torch.no_grad():model.region_translation[1:9,0] = .023
        model._select_frontier(freeze=True)
        before = model.materialized_pose()
        model._begin_point_residual_fit()
        for left,right in zip(before,model.materialized_pose()):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        self.assertFalse(model.gs_translation.detach().ne(0).any())
        self.assertFalse(model.gs_rotation.detach().ne(0).any())
        self.assertTrue(model.gs_translation.requires_grad)
        self.assertNotIn(model.gs_translation, model.optimizer.state)
        self.assertFalse(model._shared_field_active)
        self.assertTrue(model._hard_gate_frozen)

    def test_observed_zero_region_can_learn_a_coherent_motion(self):
        model = self.model()
        model.config.rate_loss_weight = 0.
        model.observations.fill_(1)
        node = int(model.owner[0]);members = model.owner.eq(node)
        model.last_step = model.config.spatial_gate.start_step
        model.optimizer_step()
        self.assertTrue(model._regional_fit_support[node])
        self.assertEqual(float(model.region_translation.abs().sum()),0.)
        target = model.base_xyz.clone();target[members,0] += .01
        for step in range(21,51):
            position,_ = model(step,training=True)
            (position-target).square().sum().backward()
            model.optimizer_step()
        final = model.materialized_pose()[0]
        self.assertLess(float((final[members]-target[members]).square().mean()),1e-6)
        torch.testing.assert_close(final[~members],target[~members],rtol=0,atol=0)
        self.assertFalse(model.gs_translation.requires_grad)

    def test_stationary_scene_stays_exact_with_learnable_zero_candidates(self):
        model = self.model(repetitions=2)
        model.observations.fill_(1)
        for step in range(1,101):
            xyz,rotation = model(step,training=True)
            terms = model.loss_terms(step,False)
            loss = ((xyz-model.base_xyz).square().sum()+
                    (rotation-model.base_rotation).square().sum()+
                    terms['rate_loss']+terms['relative_motion_loss'])
            loss.backward();model.optimizer_step()
        for before,after in zip((model.base_xyz,model.base_rotation),model.materialized_pose()):
            torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_candidate_initialization_does_not_precharge_unused_region_bytes(self):
        model = self.model(repetitions=4)
        model.observations.fill_(1)
        child = int(model.spatial_paths[0,1]);rows = model.spatial_paths[:,1].eq(child)
        values = torch.arange(int(rows.sum()))*2+1
        with torch.no_grad():model.gs_translation[rows,0] = values*model.gs_steps[0]
        model._initialize_candidates()
        expected = float(np.rint(np.median(values.numpy())))
        self.assertEqual(float(RIGID._integer(model.region_translation,model.steps[:3])[child,0]),expected)
        self.assertTrue(model._regional_fit_support[child])
        packet = copy.deepcopy(model.capture());packet.pop('candidate_initialization')
        with self.assertRaisesRegex(ValueError,'initializations are replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_retired_refinement_is_readable_as_metadata_but_cannot_resume(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet['config'].update(post_birth_motion_steps=100, fixed_support_refinement=True)
        normalized = RIGID.saved_config_with_default_motion_lr(packet['config'])
        self.assertNotIn('post_birth_motion_steps', normalized)
        self.assertNotIn('fixed_support_refinement', normalized)
        with self.assertRaisesRegex(ValueError, 'refinement checkpoints are replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_shared_shift_preserves_statistical_representative_of_relative_distribution(self):
        symbols = np.tile(np.array([20,0,0,3,0,0]),(12,1))
        symbols[-2:,0] += 2
        rows = np.full(12,3);scales=np.full(6,.01)
        before,_ = SPATIAL._rate_relative_representative(symbols,rows,3,.3,scales)
        shift = np.array([2,-1,1,-1,2,0])
        after,_ = SPATIAL._rate_relative_representative(symbols+shift,rows,3,.3,scales)
        np.testing.assert_array_equal(after,before+shift)


    def test_shared_phase_defers_point_sparsification_and_fits_real_exceptions(self):
        from unittest.mock import patch
        model = self.model(extent=6.493635559082032)
        model.config.motion_fit_window_steps = 20
        model.config.rate_loss_weight = .03
        model.config.rate_normalization = 'reference'
        model.rate_reference_count = 206237
        model.config.motion_xyz_step_extent = model.config.gs_xyz_step_extent = 3e-5
        model.config.gs_translation_lr = .0006493635559082032
        model.config.spatial_gate.start_step = 40
        model.config.spatial_gate.gate_start_step = 40
        model.config.spatial_gate.freeze_step = 60
        model.observations.fill_(1)
        wanted = model.base_xyz.clone()
        wanted[:2,0] += model.base_xyz.new_tensor([.02,.0002])
        calls=[]
        original=RIGID.MotionBase._proximal_gs_blocks
        def project(instance):
            calls.append(instance.last_step)
            return original(instance)
        with patch.object(RIGID.MotionBase,'_proximal_gs_blocks',project):
            for step in range(1,101):
                position,_=model(step,training=True)
                (.001*(position-wanted).square().sum()).backward()
                model.optimizer_step()
                if step==50:
                    self.assertTrue(model._shared_field_active)
                    self.assertFalse(model.gs_translation.requires_grad)
                    self.assertFalse(model.gs_translation.detach().ne(0).any())
        self.assertEqual(calls,[80,100])
        actual=model.materialized_pose()[0]-model.base_xyz
        self.assertLess(abs(float(actual[0,0])-.02),.002+float(model.gs_steps[0]))
        torch.testing.assert_close(actual[2:],torch.zeros_like(actual[2:]),
                                   rtol=0,atol=float(model.gs_steps[0])*1.001)

    def test_old_gate_sparse_pause_schedule_cannot_silently_resume(self):
        model=self.model();packet=copy.deepcopy(model.capture())
        packet.pop('gate_sparsification')
        with self.assertRaisesRegex(ValueError,'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_initial_region_statistics_use_free_motion_before_point_sparsification(self):
        model = self.model()
        model.config.motion_fit_window_steps = 20
        model.config.rate_warmup_steps = 20
        model.config.rate_ramp_steps = 15
        model.config.spatial_gate.start_step = 35
        model.config.validate()
        model.observations.fill_(1)
        with torch.no_grad():
            model.gs_translation[:,0] = .001
        model.last_step = 35
        model.optimizer_step()
        self.assertTrue(model._candidate_initialized)
        self.assertTrue(model._shared_field_active)
        self.assertFalse(model.gs_translation.requires_grad)
        self.assertFalse(model.gs_translation.detach().ne(0).any())
        torch.testing.assert_close(model.materialized_pose()[0][:,0]-model.base_xyz[:,0],
                                   torch.full((len(model.owner),),.001),rtol=0,atol=1e-7)

    def test_gradient_refined_region_checkpoint_is_replay_only(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet['config']['spatial_gate']['learn_region_motion'] = True
        with self.assertRaisesRegex(ValueError, 'replayed'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_complete_motion_is_not_penalized_as_a_shared_residual_before_clustering(self):
        model = self.model()
        model.config.relative_motion_loss_weight = 1.
        model.observations.fill_(1)
        with torch.no_grad():
            model.gs_translation[0,0] = .005
        self.assertEqual(float(model.loss_terms(30,False)['relative_motion_loss']), 0.)
        model._initialize_candidates()
        self.assertGreater(float(model.loss_terms(30,False)['relative_motion_loss']), 0.)

    def test_free_motion_training_skips_zero_weight_regularizer_graph(self):
        model = self.model()
        with patch.object(model, 'loss_terms', side_effect=AssertionError('unused objective')):
            self.assertEqual(float(model.regularization_loss(30)), 0.)

    def test_frozen_regularizer_matches_all_levels_in_value_and_gradient(self):
        model = self.model(repetitions=2)
        model.config.relative_motion_loss_weight = .7
        model.observations.fill_(1)
        model._initialize_candidates()
        with torch.no_grad():
            model.gate_logits[1:9] = 4
            for parameter in (model.region_translation, model.region_rotation,
                              model.gs_translation, model.gs_rotation):
                parameter.uniform_(-.01, .01)
        model._select_frontier(freeze=True)
        parameters = (model.region_translation, model.region_rotation,
                      model.gs_translation, model.gs_rotation)
        fast = model.loss_terms(80, diagnostics=False)
        fast_grad = torch.autograd.grad(fast['relative_motion_loss'], parameters)
        model._hard_gate_frozen = False
        with patch.object(model, '_frontier_probabilities',
                          return_value=model.opened.to(model.gate_logits)):
            full = model.loss_terms(80, diagnostics=False)
            full_grad = torch.autograd.grad(full['relative_motion_loss'], parameters)
        for name in fast:
            torch.testing.assert_close(fast[name], full[name])
        for actual, expected in zip(fast_grad, full_grad):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shared_field_regularizer_does_not_build_unused_point_motion(self):
        model = self.model()
        model.observations.fill_(1)
        model._begin_shared_field_fit()
        with patch.object(model, '_shared_motion', side_effect=AssertionError('unused point motion')):
            terms = model.loss_terms(50, diagnostics=False)
        self.assertEqual(float(terms['relative_motion_loss']), 0.)
        self.assertTrue(terms['rate_loss'].requires_grad)

    def test_mode_refresh_preserves_motion_moments_and_tree_and_absorbs_common_residual(self):
        model = self.model()
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        params = (model.region_translation, model.region_rotation,
                  model.gs_translation, model.gs_rotation)
        sum(p.sum() for p in params).backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        parent = int(model.owner[0])
        rows = model.owner == parent
        with torch.no_grad():
            for p in params:
                p.zero_()
            model.region_translation[parent, 0] = .01
            model.gs_translation[rows, 0] = .003
            model.region_rotation[parent, 1] = .02
            model.gs_rotation[rows, 1] = -.004
        before = model.materialized_pose()
        symbols = model._full_motion_symbols()
        bytes_before = model._hard_motion_bytes()
        states = [{k: v.clone() for k, v in model.optimizer.state[p].items()} for p in params]
        owner, opened, gates = model.owner.clone(), model.opened.clone(), model.gate_logits.detach().clone()
        model._refresh_active_representatives()
        for a, b in zip(symbols, model._full_motion_symbols()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(before, model.materialized_pose()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        self.assertEqual(float(model.gs_translation[rows].abs().sum()), 0.)
        self.assertEqual(float(model.gs_rotation[rows].abs().sum()), 0.)
        self.assertLess(model._hard_motion_bytes(), bytes_before)
        for p, state in zip(params, states):
            for key, value in state.items():
                torch.testing.assert_close(model.optimizer.state[p][key], value, rtol=0, atol=0)
        torch.testing.assert_close(model.owner, owner, rtol=0, atol=0)
        torch.testing.assert_close(model.opened, opened, rtol=0, atol=0)
        torch.testing.assert_close(model.gate_logits, gates, rtol=0, atol=0)

    def test_mode_refresh_keeps_valid_origin_if_the_modal_origin_would_overflow(self):
        model = self.model()
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        parent = int(model.owner[0])
        rows = torch.where(model.owner == parent)[0]
        with torch.no_grad():
            model.gs_translation[rows, 0] = 32767 * model.gs_steps[0]
            model.gs_translation[rows[0], 0] = -32767 * model.gs_steps[0]
        before = model._full_motion_symbols()
        model._refresh_active_representatives()
        self.assertEqual(float(model.region_translation[parent].abs().sum()), 0.)
        for a, b in zip(before, model._full_motion_symbols()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_fit_window_uses_phase_local_steps_and_closes_short_final_windows(self):
        model = self.model()
        model.config.motion_fit_window_steps = 20
        model.config.motion_steps = 97
        model.config.birth_steps = 83
        model.config.validate()
        for step, due in ((39, False), (40, True), (61, False), (80, True),
                          (96, False), (97, True)):
            model.last_step = step
            self.assertEqual(model._motion_fit_window_due(), due)

    def test_mode_refresh_preserves_pending_sub_bin_updates(self):
        model = self.model()
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        parent = int(model.owner[0])
        rows = model.owner == parent
        with torch.no_grad():
            model.region_translation[parent, 0] = .0003
            model.gs_translation[rows, 0] = .0013
        continuous = model.region_translation[model.owner] + model.gs_translation
        symbols = model._full_motion_symbols()
        model._refresh_active_representatives()
        for a, b in zip(symbols, model._full_motion_symbols()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(model.region_translation[model.owner] + model.gs_translation,
                                   continuous, rtol=1e-5, atol=1e-7)
        self.assertGreater(float(model.gs_translation[rows, 0].abs().min()), .0002)
        self.assertGreater(float(model.region_translation[parent, 0]), .0012)

    def test_mode_refresh_preserves_exact_half_bin_motion(self):
        model = self.model()
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        parent = int(model.owner[0])
        rows = model.owner == parent
        with torch.no_grad():
            model.region_translation[parent, 0] = .001
            model.gs_translation[rows, 0] = .0013
            model.gs_translation[0, 0] = model.gs_steps[0] * .5
        before = model._full_motion_symbols()
        model._refresh_active_representatives()
        for a, b in zip(before, model._full_motion_symbols()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_perfectly_fitted_shared_motion_survives_zero_image_gradient(self):
        model = self.model(extent=6.493635559082032)
        model.config.motion_fit_window_steps = 20
        model.config.rate_loss_weight = .03
        model.config.rate_normalization = 'reference'
        model.rate_reference_count = 206237
        model.config.region_translation_lr = .0003246817779541016
        model.config.motion_xyz_step_extent = model.config.gs_xyz_step_extent = 3e-5
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        parent = int(model.owner[0])
        with torch.no_grad():
            model.region_translation[parent, 0] = model.steps[0] * 50
        wanted = model.materialized_pose()[0].detach().clone()
        symbols = model._full_motion_symbols()
        model.last_step = 100
        model.optimizer.zero_grad(set_to_none=True)
        (.001 * (model.materialized_pose()[0] - wanted).square().sum()).backward()
        self.assertEqual(float(model.region_translation.grad.abs().sum()), 0.)
        model.optimizer_step()
        for a, b in zip(symbols, model._full_motion_symbols()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(model.materialized_pose()[0], wanted, rtol=1e-5, atol=1e-7)

    def test_cancelled_shared_motion_becomes_zero_mode_without_pose_change(self):
        model = self.model()
        model.config.motion_fit_window_steps = 20
        model.observations.fill_(1)
        model._initialize_candidates()
        model._select_frontier(freeze=True)
        parent = int(model.owner[0])
        rows = model.owner == parent
        with torch.no_grad():
            model.region_translation[parent, 0] = .01
            model.gs_translation[rows, 0] = -.01
        wanted = model.materialized_pose()[0].detach().clone()
        model.last_step = 100
        model.optimizer.zero_grad(set_to_none=True)
        (model.materialized_pose()[0] - wanted).square().sum().backward()
        model.optimizer_step()
        region, _, point, _ = model._values(hard=True)
        self.assertEqual(int(region[parent].abs().sum()), 0)
        self.assertEqual(int(point[rows].abs().sum()), 0)
        torch.testing.assert_close(model.materialized_pose()[0], wanted, rtol=1e-5, atol=1e-7)

    def test_initial_eight_regions_and_200_step_schedule(self):
        model = self.model()
        self.assertEqual(model.owner.unique().numel(), 8)
        self.assertFalse(model.opened[1:].any())
        for frame in (1, 10, 299):
            self.assertEqual(model.config.motion_steps_for_frame(frame), 100)
            self.assertEqual(model.config.total_steps_for_frame(frame), 200)

    def test_rotations_change_orientation_without_rotating_centers(self):
        model = self.model(1)
        with torch.no_grad():
            model.region_rotation[1:, 2] = .3
            model.gs_rotation[:, 2] = .3
        xyz, q = model(1)
        torch.testing.assert_close(xyz, model.base_xyz, rtol=0, atol=0)
        self.assertGreater(float((q - model.base_rotation).abs().max()), .1)

    def test_gate_open_and_close_preserve_pose_and_children_have_absolute_motion(self):
        model = self.model()
        parent = int(model.owner[0])
        with torch.no_grad():
            model.region_translation[parent, 0] = .01
            model.region_rotation[parent, 1] = .02
            rows = model.owner == parent
            model.gs_translation[rows, 0] = 0.
            model.gs_rotation[rows, 1] = 0.
            children = torch.where(model.parents == parent)[0]
            model.region_translation[children, 0] = .025
            model.region_rotation[children, 1] = -.03
        before = model.materialized_pose()
        with torch.no_grad():
            model.gate_logits[parent] = 4
        model._select_frontier()
        after = model.materialized_pose()
        for old, new in zip(before, after):
            torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-7)
        rows = model.parents[model.owner] == parent
        encoded_residual = model._values(hard=True)[2].float() * model.gs_steps[:3]
        torch.testing.assert_close(encoded_residual[rows, 0], torch.full((int(rows.sum()),), -.015))
        # With zero point residual, only the child absolute increment is used.
        with torch.no_grad():
            model.gs_translation.zero_()
            model.gs_rotation.zero_()
        pos, _ = model(2)
        torch.testing.assert_close(pos[rows, 0] - model.base_xyz[rows, 0], torch.full((int(rows.sum()),), .025))
        before = model.materialized_pose()
        with torch.no_grad():
            model.gate_logits[parent] = -4
        model._select_frontier()
        for old, new in zip(before, model.materialized_pose()):
            torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-7)

    def test_gate_gradient_prefers_split_when_children_explain_point_residuals(self):
        model = self.model()
        model._candidate_initialized = True
        model._coding_initialized = True
        parent = int(model.owner[0])
        children = torch.where((model.parents == parent) & model.node_counts.gt(0))[0]
        with torch.no_grad():
            for index, child in enumerate(children):
                rows = model.spatial_paths[:, 1] == child
                amount = .01 if index % 2 else -.01
                model.gs_translation[rows, 0] = amount
                model.region_translation[child, 0] = amount
            model.gate_logits.fill_(-4)
            model.gate_logits[parent] = -.2
        terms = model.loss_terms(50, diagnostics=False)
        terms['rate_loss'].backward()
        self.assertLess(float(model.gate_logits.grad[parent]), 0,
                        'Gradient descent must increase the probability of the useful split')
        self.assertTrue(terms['rate_loss'].requires_grad,
                        'Hard coding costs must still train the gate probabilities')


    def test_optimization_actually_opens_a_useful_gate(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where((model.parents == parent) & model.node_counts.gt(0))[0]
        with torch.no_grad():
            for index, child in enumerate(children):
                rows = model.spatial_paths[:, 1] == child
                model.gs_translation[rows, 0] = 0. if index % 2 else -.01
        model.observations.fill_(1)
        total_t = model.gs_translation.detach().clone()
        model._initialize_candidates()
        optimizer = torch.optim.Adam([model.gate_logits], lr=.15, eps=1e-12)
        for _ in range(20):
            with torch.no_grad():
                shared_t, _ = model._shared_motion()
                model.gs_translation.copy_(total_t - shared_t)
            optimizer.zero_grad(set_to_none=True)
            model.loss_terms(50, False)['rate_loss'].backward()
            optimizer.step()
        before = model.materialized_pose()
        model._select_frontier()
        self.assertTrue(model.opened[parent], 'A useful split must become a hard open gate')
        self.assertGreater(model.owner.unique().numel(), 8)
        for old, new in zip(before, model.materialized_pose()):
            torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-7)

    def test_gate_recoding_preserves_half_bin_ties(self):
        model = self.model()
        model.config.motion_xyz_step_extent = model.config.gs_xyz_step_extent = .125
        parent = int(model.owner[0])
        rows = model.owner == parent
        with torch.no_grad():
            model.gs_translation[rows, 0] = .0625
            model.region_translation[model.parents == parent, 0] = -.125
        before = model.materialized_pose()
        with torch.no_grad():
            model.gate_logits[parent] = 4
        model._select_frontier()
        for old, new in zip(before, model.materialized_pose()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertTrue(model._values(hard=True)[2][rows, 0].eq(1).all())


    def test_forward_uses_a_complete_region_and_inactive_motion_cannot_cancel_it(self):
        model = self.model()
        model._candidate_initialized = True
        model._coding_initialized = True
        parent = int(model.owner[0])
        child = int(model.spatial_paths[0, 1])
        with torch.no_grad():
            model.region_translation[parent, 0] = .02
            model.region_translation[child, 0] = -.02
        pos, _ = model(30, training=True)
        rows = model.owner == parent
        torch.testing.assert_close(pos[rows, 0] - model.base_xyz[rows, 0],
                                   torch.full((int(rows.sum()),), .02))
        (pos - model.base_xyz).square().mean().backward()
        self.assertGreater(float(model.region_translation.grad[parent, 0]), 0.)
        self.assertEqual(float(model.region_translation.grad[child, 0]), 0.)

    def test_coding_region_switch_preserves_pose_and_checkpoint_state(self):
        model = self.model()
        model._candidate_initialized = True
        model._coding_initialized = True
        with torch.no_grad():
            model.region_translation[1:9, 0] = .01
            model.region_translation[9:73, 0] = -.02
            model.region_rotation[9:73, 2] = .03
        before = model.materialized_pose()
        model._recode_owner(model.spatial_paths[:, 1])
        for left, right in zip(before, model.materialized_pose()):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
        restored = SPATIAL.SpatialGatedMotion(model.config)
        restored.restore(copy.deepcopy(model.capture()))
        self.assertTrue(restored._coding_initialized)
        for left, right in zip(model.materialized_pose(), restored.materialized_pose()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_live_gate_recoding_preserves_sub_bin_image_updates(self):
        model = self.model()
        model._candidate_initialized = model._coding_initialized = True
        with torch.no_grad():
            model.gs_translation[:,0] = .0102
            model.region_translation[9:73,0] = .003
        full = model._full_motion_symbols()
        before = model.materialized_pose()
        model._recode_owner(model.spatial_paths[:,1], full, preserve_remainder=True)
        for left, right in zip(before, model.materialized_pose()):
            torch.testing.assert_close(left,right,rtol=0,atol=1e-7)
        remainder = model.gs_translation[:,0] - torch.round(
            model.gs_translation[:,0]/model.gs_steps[0])*model.gs_steps[0]
        torch.testing.assert_close(remainder,torch.full_like(remainder,.0002),rtol=0,atol=1e-8)

    def test_old_frozen_motion_schedule_cannot_silently_resume_as_live_fitting(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet['config']['spatial_gate'].pop('image_fit_during_gate')
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_coordinate_recoding_preserves_gs_optimizer_moments(self):
        model = self.model()
        model._candidate_initialized = model._coding_initialized = True
        model.gs_translation.grad = torch.ones_like(model.gs_translation) * .001
        model.gs_rotation.grad = torch.ones_like(model.gs_rotation) * .001
        model.optimizer.step()
        saved = {name: {key: model.optimizer.state[getattr(model, name)][key].clone()
                       for key in ('exp_avg', 'exp_avg_sq')}
                 for name in ('gs_translation', 'gs_rotation')}
        model._recode_owner(model.spatial_paths[:, 1])
        for name, moments in saved.items():
            for key, before in moments.items():
                torch.testing.assert_close(model.optimizer.state[getattr(model, name)][key], before, rtol=0, atol=0)


    def test_region_cost_matches_soft_selection_and_hard_frontier(self):
        model = self.model()
        model._candidate_initialized = True
        probability = model._gate_probabilities()
        expected = model.base_xyz.new_zeros(len(model.parents))
        expected.scatter_add_(0, model.spatial_paths.flatten(), model._path_weights(probability).flatten())
        expected /= model.node_counts.clamp_min(1)
        torch.testing.assert_close(model._region_selection_probabilities(), expected)
        model._select_frontier()
        selected = model._region_selection_probabilities()
        self.assertTrue(selected.eq(0).logical_or(selected.eq(1)).all())
        self.assertEqual(int(selected.sum()), model.owner.unique().numel())
        self.assertFalse(model.loss_terms(80, False)['rate_loss'].requires_grad)

    def test_new_frame_does_not_inherit_saturated_gate_logits(self):
        model = self.model()
        with torch.no_grad():
            model.gate_logits.fill_(-6)
        model._select_frontier()
        xyz, rotation = model.materialized_pose()
        model.register_materialized_layout(xyz, rotation)
        with tempfile.TemporaryDirectory() as directory:
            model.export_motion(Path(directory) / 'motion.bin')
        prior = model.codec_previous_state
        model.setup(prior.xyz, prior.rotation, 1., frame_index=2)
        self.assertTrue(model.gate_logits[1:9].eq(model.config.spatial_gate.initial_logit).all())
        self.assertFalse(model._hard_gate_frozen)
        self.assertEqual(model.owner.unique().numel(), 8)

    def test_previous_increment_initialization_preserves_lineage_and_resume(self):
        model = self.model(1)
        with torch.no_grad():
            model.region_translation[1:9, 0] = .005
            model.region_rotation[1:9, 2] = .01
            model.gs_translation[0, 1] = .003
        t, r = [x.clone() for x in model._full_motion_symbols()]
        sources = torch.tensor([0, 42])
        xyz, q = model.materialized_pose()
        model.register_materialized_layout(torch.cat((xyz, xyz[sources])),
                                           torch.cat((q, q[sources])), sources)
        with tempfile.TemporaryDirectory() as directory:
            model.export_motion(Path(directory) / 'motion.bin')
        restored = SPATIAL.SpatialGatedMotion(model.config)
        restored.restore(copy.deepcopy(model.capture()))
        prior = model.codec_previous_state
        # Deliberately reorder survivors and include both newborns.
        rows = torch.tensor([65, 0, 64, 42, 1])
        expected_t = torch.round(.6 * torch.cat((t, t[sources]))[rows])
        expected_r = torch.cat((r, r[sources]))[rows]
        for current in (model, restored):
            current.setup(prior.xyz[rows], prior.rotation[rows], 1.,
                          frame_index=2, previous_rows=rows)
            torch.testing.assert_close(current._full_motion_symbols()[0], expected_t)
            torch.testing.assert_close(current._full_motion_symbols()[1], expected_r)
            self.assertFalse(current.region_translation.detach().ne(0).any())
            self.assertFalse(current.optimizer.state)
        torch.testing.assert_close(model(1)[0], restored(1)[0], rtol=0, atol=0)

    def test_previous_increment_is_trainable_when_motion_stops(self):
        model = self.model(1)
        with torch.no_grad():
            model.gs_translation[0, 0] = .01
        xyz, q = model.materialized_pose()
        model.register_materialized_layout(xyz, q)
        with tempfile.TemporaryDirectory() as directory:
            model.export_motion(Path(directory) / 'motion.bin')
        prior = model.codec_previous_state
        model.setup(prior.xyz, prior.rotation, 1., frame_index=2)
        initial = float((model(1)[0] - model.base_xyz).square().sum())
        self.assertGreater(initial, 0.)
        for step in range(1, 40):
            position, _ = model(step, training=True)
            (position - model.base_xyz).square().sum().backward()
            model.optimizer_step()
        final = model(39)[0]
        self.assertLess(float((final-model.base_xyz).square().sum()), initial / 10)
        torch.testing.assert_close(final[1:], model.base_xyz[1:], rtol=0, atol=0)

    def test_hard_frontier_recodes_pose_and_restores_checkpoint(self):
        model = self.model()
        model._candidate_initialized = True
        with torch.no_grad():
            model.region_translation[1:9, 0] = .01
            model.region_translation[9:73, 0] = -.015
            model.region_translation[73:, 0] = .02
            model.region_rotation[9:73, 2] = .01
            model.gs_translation[:, 1] = .003
        before = model.materialized_pose()
        model._select_frontier()
        self.assertTrue(model._hard_gate_frozen)
        for old, new in zip(before, model.materialized_pose()):
            torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-7)
        restored = SPATIAL.SpatialGatedMotion(model.config)
        restored.restore(copy.deepcopy(model.capture()))
        self.assertTrue(restored._hard_gate_frozen)
        for old, new in zip(model.materialized_pose(), restored.materialized_pose()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)


    def test_sequential_codec_birth_paths_and_checkpoint_resume(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:, 0] = .005
            model.region_rotation[1:, 2] = .01
            model.gate_logits[model.split_eligible] = 2
        model.last_step = 40
        # Populate the optimizer to check that it restores the gate's moments.
        model.loss_terms(50, False)['rate_loss'].backward()
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        model._select_frontier()
        captured = copy.deepcopy(model.capture())
        resumed = SPATIAL.SpatialGatedMotion(model.config)
        resumed.restore(captured)
        torch.testing.assert_close(resumed.gate_logits, model.gate_logits)
        self.assertEqual(len(resumed.optimizer.param_groups), 5)
        self.assertIn(resumed.gate_logits, resumed.optimizer.state)
        model.record_final_diagnostics()
        xyz, q = model.materialized_pose()
        source = torch.tensor([0, 42])
        model.register_materialized_layout(torch.cat((xyz, xyz[source])),
                                           torch.cat((q, q[source])), source)
        with tempfile.TemporaryDirectory() as directory:
            info = model.export_motion(Path(directory) / 'motion.bin')
            self.assertTrue(info['format'].startswith(('SPG', 'SPC', 'SPL')))
            decoded = model.codec_previous_state
            expected_paths = LAYOUT.spatial_route(decoded.xyz[-2:], decoded.spatial_centers,
                                                  model.config.spatial_gate.max_depth)
            torch.testing.assert_close(decoded.spatial_paths[-2:], expected_paths)
            t, r = model._full_motion_symbols()
            expected = torch.round(.6 * torch.cat((t, t[source])))
            model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
            expected_paths = LAYOUT.spatial_route(decoded.xyz, decoded.spatial_centers,
                                                  model.config.spatial_gate.max_depth)
            torch.testing.assert_close(model.spatial_paths, expected_paths)
            torch.testing.assert_close(model.owner, LAYOUT.spatial_owner(expected_paths, model.opened))
            torch.testing.assert_close(model._full_motion_symbols()[0], expected)
            self.assertEqual(model.frame_reference_state.digest(), decoded.digest())

    def test_static_planes_reproduce_initial_paths_and_route_empty_cells(self):
        model = self.model(1)
        reference = model.frame_reference_state
        paths = LAYOUT.spatial_route(reference.xyz, reference.spatial_centers,
                                     model.config.spatial_gate.max_depth)
        torch.testing.assert_close(paths, reference.spatial_paths, rtol=0, atol=0)
        # Even a previously empty cell has deterministic descendants.
        outside = torch.tensor([[-10., -10., -10.], [10., 10., 10.]])
        routed = LAYOUT.spatial_route(outside, reference.spatial_centers,
                                      model.config.spatial_gate.max_depth)
        torch.testing.assert_close(reference.parents[routed[:, 1:]], routed[:, :-1])

    def test_crossing_split_changes_next_frame_membership_and_replays_exactly(self):
        model = self.model(1)
        original = model.spatial_paths[0].clone()
        with torch.no_grad():
            model.gs_translation[0] = torch.tensor([.8, .8, .8])
        xyz, q = model.materialized_pose()
        # A newborn can move away from its source's region during birth fitting.
        newborn = xyz[0:1] - 1.5
        model.register_materialized_layout(torch.cat((xyz, newborn)),
                                           torch.cat((q, q[0:1])), torch.tensor([0]))
        reference = model.frame_reference_state
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'first.bin'
            model.export_motion(path)
            decoded = CODEC.decode_motion(path.read_bytes(), reference)
            self.assertTrue(torch.equal(decoded.spatial_paths[0], original))
            self.assertFalse(torch.equal(decoded.spatial_paths[0], decoded.spatial_paths[-1]))
            resumed = SPATIAL.SpatialGatedMotion(model.config)
            resumed.restore(copy.deepcopy(model.capture()))
            for current in (model, resumed):
                current.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
                self.assertFalse(torch.equal(current.spatial_paths[0], original))
            torch.testing.assert_close(model.spatial_paths, resumed.spatial_paths)
            with torch.no_grad():
                model.region_translation[1:9, 0] = torch.arange(1, 9) * .001
            xyz, q = model.materialized_pose()
            model.register_materialized_layout(xyz, q)
            model.export_motion(Path(directory) / 'second.bin')
            second = CODEC.decode_motion((Path(directory) / 'second.bin').read_bytes(), decoded)
            torch.testing.assert_close(second.xyz, xyz, rtol=0, atol=1e-7)
            torch.testing.assert_close(second.spatial_paths, model.spatial_paths)

    def test_legacy_streams_keep_fixed_membership_and_original_reference_hash(self):
        model = self.model(1)
        prior = copy.deepcopy(model.frame_reference_state)
        prior.spatial_centers = None
        legacy = prior.capture()
        legacy.pop('spatial_centers')
        restored = CODEC.DecodedState.restore(legacy)
        self.assertEqual(prior.digest(), restored.digest())
        rt, rr, gt, gr = [x.cpu() for x in model._values(hard=True)]
        gt[0] = 800
        packet = CODEC.encode_motion(prior, 1, 1., model.previous_rows, model.owner,
            model.parents, rt, rr, gt, gr, torch.empty(0, dtype=torch.long),
            torch.empty(0, 3), torch.empty(0, 4), .001, .001,
            gs_translation_step_extent=.001, gs_rotation_step_rad=.001,
            spatial_open=model.opened)
        decoded = CODEC.decode_motion(packet, restored)
        self.assertIsNone(decoded.spatial_centers)
        torch.testing.assert_close(decoded.spatial_paths, prior.spatial_paths)
        self.assertGreater(float(decoded.xyz[0, 0]), .8)
        changed = copy.deepcopy(model.frame_reference_state)
        changed.spatial_centers[0, 0] += .01
        self.assertNotEqual(changed.digest(), model.frame_reference_state.digest())

    def test_fixed_membership_training_checkpoint_is_replay_only(self):
        model = self.model(1)
        packet = copy.deepcopy(model.capture())
        packet.pop('spatial_assignment')
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_warmup_fits_point_motion_and_freezes_region_optimizer(self):
        model = self.model(1)
        model.config.rate_warmup_steps = model.config.spatial_gate.start_step
        pos, q = model(1, training=True)
        loss = (pos - model.base_xyz - .01).square().mean() + (q[:, 2] - .01).square().mean()
        loss.backward()
        for name in ('gs_translation', 'gs_rotation'):
            self.assertGreater(float(getattr(model, name).grad.abs().sum()), 0, name)
        model.optimizer_step()
        for name in ('region_translation', 'region_rotation'):
            torch.testing.assert_close(getattr(model, name), torch.zeros_like(getattr(model, name)), rtol=0, atol=0)
        self.assertGreater(float(model.gs_translation.detach().abs().sum()), 0)


# ----- test_joint_spatial_motion -----
class JointSpatialMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, initial_logit=-2.):
        axis = torch.linspace(-1., 1., 8)
        xyz = torch.cat((torch.cartesian_prod(axis, axis, axis),
                         torch.tensor([[-100., -100., -100.], [100., 100., 100.]])))
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        config = RigidTreeConfig(motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
            motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
            rate_loss_weight=.002, relative_motion_loss_weight=.01,
            rate_warmup_steps=0, rate_ramp_steps=1, motion_fit_window_steps=20)
        gate = config.spatial_gate
        gate.max_depth = 3
        gate.initialization = 'robust_median'
        gate.initialization_distance_quantile = .99
        gate.outlier_assignment = 'nearest_region'
        gate.training_schedule = 'joint'
        gate.learn_region_motion = True
        gate.min_region_size = 2
        gate.initial_logit = initial_logit
        gate.start_step = gate.gate_start_step = 0
        gate.freeze_step = 70
        config.validate()
        model = SpatialGatedMotion(config)
        model.setup(xyz, q, 2., frame_index=1)
        return model

    def commit(self, model, births=None, sources=None):
        reference = copy.deepcopy(model.frame_reference_state)
        xyz, q = model.materialized_pose()
        sources = torch.empty(0, dtype=torch.long) if sources is None else sources
        if births is not None:
            xyz = torch.cat((xyz, births))
            q = torch.cat((q, q[sources]))
        model.register_materialized_layout(xyz, q, sources)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.bin'
            model.export_motion(path)
            decoded = decode_motion(path.read_bytes(), reference)
        torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)
        torch.testing.assert_close(decoded.rotation, q, rtol=0, atol=1e-6)
        return decoded

    def test_far_points_use_nearest_supported_leaf_without_changing_geometry(self):
        model = self.model()
        reference = model.frame_reference_state
        query = torch.tensor([[100., 100., 100.], [-100., -100., -100.],
                              [200., -200., 100.]])
        distances = (query.double()[:, None] - reference.spatial_leaf_centers.double()[None]).square().sum(-1)
        expected = reference.spatial_leaf_nodes[distances.argmin(1)]
        paths = spatial_state_route(query, reference)
        torch.testing.assert_close(paths[:, -1], expected, rtol=0, atol=0)
        torch.testing.assert_close(reference.parents[paths[:, 1:]], paths[:, :-1])
        torch.testing.assert_close(model(1)[0], model.base_xyz, rtol=0, atol=0)
        self.assertEqual(len(model.owner), 514)
        inside = torch.tensor([[0., 0., 0.], [.1, -.2, .3]])
        torch.testing.assert_close(spatial_state_route(inside, reference),
            spatial_route(inside, reference.spatial_centers, 3), rtol=0, atol=0)

    def test_routing_reference_is_shared_once_restored_and_digest_bound(self):
        reference = self.model().frame_reference_state
        packet = {key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
                  for key, value in reference.capture().items() if value is not None}
        restored = DecodedState.from_reference(packet)
        self.assertEqual(restored.digest(), reference.digest())
        restored = DecodedState.restore(reference.capture())
        for name in SPATIAL_ROUTING_FIELDS:
            torch.testing.assert_close(getattr(restored, name), getattr(reference, name), rtol=0, atol=0)
        changed = copy.deepcopy(reference)
        changed.spatial_leaf_centers[0, 0] += .01
        self.assertNotEqual(changed.digest(), reference.digest())

    def test_first_dynamic_frame_closes_gates_and_both_motion_fields_learn(self):
        model = self.model()
        self.assertFalse(bool(model.opened[1:].any()))
        self.assertTrue(bool(model.opened[0]))
        self.assertEqual(len(model.owner.unique()), 8)
        self.assertTrue(bool(model.gate_logits.eq(-2.).all()))
        self.assertEqual(model.flat_region_diagnostics['initial_gates'], 'all_eligible_closed')
        self.assertFalse(model._hard_gate_frozen)
        xyz, q = model(1, training=True)
        loss = (xyz[:, 0] - model.base_xyz[:, 0] - .1).square().mean()
        loss = loss + (q[:, 1] - .01).square().mean()
        loss.backward()
        self.assertIsNone(model.gate_logits.grad)
        self.assertIsNone(model.region_translation.grad)
        self.assertIsNone(model.region_rotation.grad)
        model.optimizer_step()
        for step in (2, 3):
            xyz, q = model(step, training=True)
            loss = (xyz[:, 0] - model.base_xyz[:, 0] - .1).square().mean()
            loss = loss + (q[:, 1] - .01).square().mean() + model.regularization_loss(step)
            loss.backward()
            if step == 2:
                model.optimizer_step()
        for parameter in (model.region_translation, model.region_rotation,
                          model.gs_translation, model.gs_rotation):
            self.assertTrue(parameter.requires_grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(model._candidate_initialized)

    def test_first_gate_frontier_uses_configured_initial_logit(self):
        model = self.model(initial_logit=2.)
        self.assertTrue(bool(model.opened[model.split_eligible].all()))
        self.assertTrue(bool(model.gate_logits.eq(2.).all()))
        self.assertEqual(model.flat_region_diagnostics['initial_gates'], 'all_eligible_open')

    def test_region_updates_change_residual_origin_without_changing_full_pose(self):
        model = self.model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.region_translation_lr = .004
        with torch.no_grad():
            model.gs_translation[:, 0] = .03
        before = tuple(value.detach().clone() for value in model(3, training=True))
        model.regularization_loss(3).backward()
        model.optimizer_step()
        self.assertGreater(float(model.region_translation.abs().sum()), 0.)
        self.assertLess(float(model.gs_translation[:, 0].mean()), .03)
        for actual, expected in zip(model(3), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_joint_loss_learns_merges_and_splits_from_either_initialization(self):
        for varying_children in (False, True):
            for initial_logit in (-2., 2.):
                with self.subTest(varying_children=varying_children, initial_logit=initial_logit):
                    model = self.model(initial_logit=initial_logit)
                    parent = int(model.spatial_paths[0, 0])
                    target = model.base_xyz.clone()
                    if varying_children:
                        children = torch.where(model.parents.eq(parent) & model.node_counts.gt(0))[0]
                        for index, child in enumerate(children):
                            target[model.spatial_paths[:, 1].eq(child), 0] += .02 if index % 2 else -.02
                    else:
                        target[:, 0] += .02
                    for step in range(1, 71):
                        model.observations.fill_(1)
                        xyz, _ = model(step, training=True)
                        loss = .5 * ((xyz - target) / .01).square().mean()
                        loss = loss + model.regularization_loss(step)
                        loss.backward()
                        model.optimizer_step()
                    self.assertEqual(bool(model.opened[parent]), varying_children)
                    self.assertLess(float((model(70)[0] - target).square().mean()), 1e-6)

    def test_live_joint_gate_cost_includes_point_residuals(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.linspace(0., .01, len(model.parents))
        zero = model.loss_terms(1)
        with torch.no_grad():
            model.gs_translation[:, 0] = .1
        nonzero = model.loss_terms(1)
        self.assertGreater(float(nonzero['relative_motion']), float(zero['relative_motion']))
        self.assertGreater(float(nonzero['rate_bytes_surrogate']), float(zero['rate_bytes_surrogate']))
        model.regularization_loss(1).backward()
        self.assertGreater(float(model.gate_logits.grad.abs().sum()), 0.)
        self.assertGreater(float(model.gs_translation.grad.abs().sum()), 0.)
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)

    def test_gate_ste_has_discrete_cost_and_the_soft_binary_derivative(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where(model.parents.eq(parent))[0]
        with torch.no_grad():
            model.region_translation[parent, 0] = .004
            model.region_translation[children, 0] = .024
            model.gs_translation[model.owner.eq(parent), 0] = .02
            model.gate_logits[parent] = -.2
        captured = copy.deepcopy(model.capture())
        model._hard_gate_frozen = True
        terms = model.loss_terms(10, False)
        closed_cost = float(terms['rate_loss'] + terms['relative_motion_loss'])
        model._hard_gate_frozen = False
        with torch.no_grad():
            model.gate_logits[parent] = 2.
        model._select_frontier(freeze=True)
        terms = model.loss_terms(10, False)
        open_cost = float(terms['rate_loss'] + terms['relative_motion_loss'])
        model.restore(captured)
        probability = torch.sigmoid(model.gate_logits[parent] / model.config.spatial_gate.temperature)
        probabilities = torch.zeros_like(model.gate_logits)
        probabilities[parent] = probability
        with patch.object(model, '_frontier_probabilities', return_value=probabilities):
            terms = model.loss_terms(10, False)
            loss = terms['rate_loss'] + terms['relative_motion_loss']
            self.assertAlmostEqual(float(loss), closed_cost, places=9)
            actual = torch.autograd.grad(loss, model.gate_logits)[0][parent]
        expected = (open_cost - closed_cost) * probability * (1 - probability) / model.config.spatial_gate.temperature
        self.assertAlmostEqual(float(actual), float(expected), places=9)
        self.assertLess(float(actual), 0.)

    def test_closed_ancestor_does_not_block_descendant_gate_gradient(self):
        model = self.model()
        node = int(model.spatial_paths[0, 1])
        parent = int(model.parents[node])
        self.assertFalse(bool(model.opened[parent]))
        children = torch.where(model.parents.eq(node) & model.node_counts.gt(0))[0]
        with torch.no_grad():
            for index, child in enumerate(children):
                rows = model.spatial_paths[:, -1].eq(child)
                value = .02 if index % 2 else -.02
                model.region_translation[child, 0] = value
                model.gs_translation[rows, 0] = value
        terms = model.loss_terms(10, False)
        (terms['rate_loss'] + terms['relative_motion_loss']).backward()
        self.assertLess(float(model.gate_logits.grad[node]), 0.)

    def test_gate_switch_recodes_residuals_without_pose_jump(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)) % 7 * .002
            model.region_rotation[:, 2] = torch.arange(len(model.parents)) % 5 * .001
            model.gs_translation[:, 1] = .0073
        before = tuple(value.detach().clone() for value in model(5))
        for logit in (2., -2.):
            old_owner = model.owner.clone()
            with torch.no_grad():
                model.gate_logits[1] = logit
            model._select_frontier(freeze=False)
            self.assertTrue(bool(model.owner.ne(old_owner).any()))
            for actual, expected in zip(model(5), before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertGreater(float(model.gs_translation.abs().sum()), 0.)

    def test_step70_freezes_only_gates_and_keeps_point_values_and_optimizer_history(self):
        model = self.model()
        model.config.region_translation_lr = model.config.region_rotation_lr = 0.
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.spatial_gate.learning_rate = 0.
        model.optimizer.param_groups[-1]['lr'] = 0.
        with torch.no_grad():
            model.gs_translation[:, 0] = .0373
            model.gs_rotation[:, 1] = .0023
        before = (model.gs_translation.detach().clone(), model.gs_rotation.detach().clone())
        xyz, q = model(70, training=True)
        (xyz.sum() + q.sum()).backward()
        model.optimizer_step()
        self.assertTrue(model._hard_gate_frozen)
        self.assertFalse(model._shared_field_active)
        for parameter, expected in zip((model.gs_translation, model.gs_rotation), before):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            self.assertTrue(parameter.requires_grad)
            self.assertIn('exp_avg', model.optimizer.state[parameter])
        gates = model.gate_logits.detach().clone()
        model.optimizer.param_groups[-1]['lr'] = .15
        xyz, _ = model(71, training=True)
        xyz.sum().backward()
        model.gate_logits.grad = torch.ones_like(model.gate_logits)
        model.optimizer_step()
        torch.testing.assert_close(model.gate_logits, gates, rtol=0, atol=0)

    def test_next_frame_inherits_full_motion_with_births_and_recomputed_owners(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = .01
            model.region_rotation[:, 2] = .002
            model.gs_translation[0, 1] = .0063
            model.gs_rotation[0, 0] = .0032
            model.gate_logits[1] = 3.25
            model.gate_logits[2] = -3.25
        model._select_frontier(freeze=False)
        previous_gates = model.gate_logits.detach().clone()
        total = tuple(value.detach().clone() for value in model._full_motion_symbols())
        source = torch.tensor([0], dtype=torch.long)
        decoded = self.commit(model, torch.tensor([[200., -200., 100.]]), source)
        captured = copy.deepcopy(model.capture())
        resumed = SpatialGatedMotion(copy.deepcopy(model.config))
        resumed.restore(captured)
        for current in (model, resumed):
            current.setup(decoded.xyz, decoded.rotation, 2., frame_index=2)
            for actual, previous in zip(current._full_motion_symbols(), total):
                expected = torch.cat((previous, previous[source]))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(current.spatial_paths,
                spatial_state_route(decoded.xyz, decoded), rtol=0, atol=0)
            torch.testing.assert_close(current.gate_logits, previous_gates, rtol=0, atol=0)
            self.assertTrue(bool(current.opened[1]))
            self.assertFalse(bool(current.opened[2]))
            self.assertEqual(current.flat_region_diagnostics['initial_gates'], 'previous_frame')
        torch.testing.assert_close(model.gate_logits, resumed.gate_logits, rtol=0, atol=0)

    def test_mid_joint_checkpoint_restores_trainable_residuals_and_rejects_old_schedule(self):
        model = self.model()
        captured = copy.deepcopy(model.capture())
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(captured)
        self.assertTrue(restored.gs_translation.requires_grad)
        self.assertTrue(restored.gs_rotation.requires_grad)
        self.assertTrue(restored._shared_field_active)
        old = copy.deepcopy(captured)
        old['motion_schedule'] = 'shared_field_zero_residual_v2'
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            restored.restore(old)
        old['motion_schedule'] = 'joint_motion_v1'
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            restored.restore(old)


# ----- test_joint_image_rate -----
class JointImageRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = JointSpatialMotionTests().model()
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


# ----- test_relative_spatial_motion -----
class RelativeSpatialMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        m = JointSpatialMotionTests().model()
        m.config.spatial_gate.region_motion = 'relative'
        m.config.proximal_gs_regularization = True
        m.config.differentiable_rate_support = True
        m.observations.fill_(1)
        return m

    def packet(self, m, compression='auto'):
        rt, rr, gt, gr = m._values(hard=True)
        return encode_motion(m.frame_reference_state, m.frame_index, m.extent,
            m.previous_rows, m.owner, m.parents, rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            m.config.motion_xyz_step_extent, m.config.motion_rotation_step_rad,
            compression=compression, gs_translation_step_extent=m.config.gs_xyz_step_extent,
            gs_rotation_step_rad=m.config.gs_rotation_step_rad, spatial_open=m.opened,
            spatial_relative=True)

    def test_integer_corrections_accumulate_without_int16_overflow(self):
        m = self.model()
        with torch.no_grad():
            m.region_translation[1:, 0] = 20000 * m.steps[0]
            m.gate_logits[m.split_eligible] = 4.
        # Construct a large valid pose directly. Recoding a coarse pose into
        # this field would itself need a residual outside signed-int16 range.
        m.opened = m.split_eligible.clone()
        m.opened[0] = True
        m.owner = spatial_owner(m.spatial_paths, m.opened)
        m._hard_gate_frozen = True
        pred = m._regional_predictions(hard=True)[0]
        self.assertTrue(bool(pred[m.owner, 0].eq(60000).all()))
        for compression in ('none', 'auto', 'zlib'):
            data = self.packet(m, compression)
            self.assertEqual(uncompressed_motion_packet(data)[:4], b'SPG2')
            decoded = decode_motion(data, m.frame_reference_state)
            torch.testing.assert_close(decoded.xyz, m(100)[0], atol=1e-6, rtol=0)

    def test_selected_path_keeps_parent_corrections_and_discards_hidden_nodes(self):
        m = self.model()
        parent = int(m.owner[0])
        child = int(m.spatial_paths[0, 1])
        with torch.no_grad():
            m.region_translation[parent, 0] = .004
            m.region_translation[child, 0] = .006
            m.gate_logits[parent] = 4.
        m._select_frontier(freeze=True)
        rt, _, _, _ = m._values(hard=True)
        self.assertEqual(int(rt[parent, 0]), 2)
        self.assertEqual(int(rt[child, 0]), 3)
        self.assertFalse(bool(m.owner.eq(parent).any()))
        self.assertGreater(int(m._stored_region_support().sum()), m.owner.unique().numel())
        decoded = decode_motion(self.packet(m), m.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, m(100)[0], atol=1e-6, rtol=0)

    def test_ancestor_support_stops_at_identity_in_mixed_depth_frontiers(self):
        m = self.model()
        for logit in (-4., 4.):
            with torch.no_grad():
                m.gate_logits[m.split_eligible] = logit
                # Retain a coarse region while other branches can be deep.
                m.gate_logits[1] = -4.
            m._select_frontier(freeze=False)
            expected = torch.zeros_like(m.opened)
            reach = torch.ones(len(m.owner), dtype=torch.bool)
            for nodes in m.spatial_paths.unbind(1):
                expected[nodes[reach]] = True
                reach &= m.opened[nodes]
            torch.testing.assert_close(m._stored_region_support(), expected, atol=0, rtol=0)
            self.assertEqual(int(expected.sum()), len(m.owner.unique()) +
                sum(bool(expected[n] and m.opened[n]) for n in range(1,len(expected))))

    def test_closed_candidate_fits_without_training_gate_target_or_parent(self):
        m = self.model()
        with torch.no_grad():
            m.gs_translation[:, 0] = .012
            m.gate_logits.fill_(-6)
        m._select_frontier(freeze=False)
        loss = m.loss_terms(5, False)['candidate_motion_loss']
        grad = torch.autograd.grad(loss, [m.region_translation, m.gs_translation,
            m.gate_logits], allow_unused=True)
        self.assertIsNone(grad[1])
        self.assertIsNone(grad[2])
        self.assertEqual(float(grad[0][1:9].abs().sum()), 0.)
        self.assertGreater(float(grad[0][9:].abs().sum()), 0.)

    def test_gate_gradient_opens_shared_correction_without_rule_edits(self):
        m = self.model()
        parent = int(m.owner[0])
        children = torch.where(m.parents.eq(parent) & m.node_counts.gt(0))[0]
        with torch.no_grad():
            m.region_translation[parent, 0] = .02
            m.gs_translation[m.owner.eq(parent), 0] = .01
            for k, child in enumerate(children):
                rows = m.spatial_paths[:, 1].eq(child)
                correction = -.01 if k % 2 else .01
                m.gs_translation[rows, 0] = correction
                m.region_translation[child, 0] = correction
            m.gate_logits.fill_(-4.)
            m.gate_logits[parent] = -.2
        full = tuple(x.detach().clone() for x in m._full_motion_symbols())
        optimizer = torch.optim.Adam([m.gate_logits], lr=.15, eps=1e-12)
        initial = float(m.loss_terms(5, False)['rate_bytes_surrogate'])
        for step in range(1, 31):
            optimizer.zero_grad(set_to_none=True)
            m.loss_terms(step, False)['rate_loss'].backward()
            optimizer.step()
            m._select_frontier(freeze=False)
        self.assertTrue(bool(m.opened[parent]))
        self.assertLess(float(m.loss_terms(31, False)['rate_bytes_surrogate']), initial)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))

    def test_fixed_frontier_learns_ancestors_and_refits_without_pose_change(self):
        m = self.model()
        with torch.no_grad():
            m.gate_logits[m.split_eligible] = 4.
            m.gs_translation[:, 0] = .006
        m._select_frontier(freeze=True)
        loss = m.regularization_loss(80)
        loss.backward()
        self.assertGreater(float(m.region_translation.grad[1:9].abs().sum()), 0.)
        self.assertIsNone(m.gate_logits.grad)
        full = tuple(x.clone() for x in m._full_motion_symbols())
        m._refresh_active_representatives()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))

    def test_explicit_migration_preserves_all_predictions_and_next_frame_warm_start(self):
        m = JointSpatialMotionTests().model()
        with torch.no_grad():
            m.region_translation[:, 0] = torch.arange(len(m.parents)).remainder(5) * .002
            m.region_rotation[:, 1] = torch.arange(len(m.parents)).remainder(7) * .001
            m.gs_translation[:, 2] = .0024
        before = tuple(x.detach().clone() for x in m._regional_predictions(hard=True))
        full = tuple(x.clone() for x in m._full_motion_symbols())
        JointSpatialMotionTests().commit(m)
        m.enable_relative_regions()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, m._regional_predictions(hard=True))))
        xyz, q = m.codec_previous_state.xyz, m.codec_previous_state.rotation
        m.setup(xyz, q, m.extent, frame_index=2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))

    def test_relative_checkpoint_resumes_the_same_update(self):
        m = self.model()
        with torch.no_grad():
            m.gs_translation[:, 0] = .003
        m(1, training=True)
        (m(1)[0].square().mean() + m.regularization_loss(1)).backward()
        m.optimizer_step()
        saved = copy.deepcopy(m.capture())
        restored = SpatialGatedMotion(copy.deepcopy(m.config))
        restored.restore(saved)
        for model in (m, restored):
            (model(2, training=True)[0].square().mean() + model.regularization_loss(2)).backward()
            model.optimizer_step()
        for key in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation', 'gate_logits'):
            torch.testing.assert_close(getattr(m, key), getattr(restored, key), atol=0, rtol=0)

    def test_relative_chain_with_births_and_survivor_retirement(self):
        m = self.model()
        with torch.no_grad():
            m.region_translation[1:9, 0] = .004
            m.region_rotation[1:9, 1] = .002
        first = JointSpatialMotionTests().commit(m,
            births=m.base_xyz[:1] + .01, sources=torch.tensor([0]))
        m.setup(first.xyz, first.rotation, m.extent, frame_index=2)
        m.retain_survivors(torch.arange(0, len(m.owner), 2))
        second = JointSpatialMotionTests().commit(m)
        self.assertEqual(len(second.xyz), (len(first.xyz) + 1) // 2)


# ----- test_translation_only_motion -----
class TranslationOnlyMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        source = JointSpatialMotionTests().model()
        config = copy.deepcopy(source.config)
        config.rotation_updates = False
        config.proximal_gs_regularization = True
        model = SpatialGatedMotion(config)
        rotation = rotation_vector_quaternion(torch.full_like(source.base_xyz, .2))
        model.setup(source.base_xyz, rotation, source.extent, frame_index=1)
        return model

    def test_translation_learns_with_no_rotation_parameters_or_adam_moments(self):
        model = self.model()
        xyz, orientation = model(1, training=True)
        image = (xyz - model.base_xyz - .01).square().mean()
        terms = model.loss_terms(1, False)
        (image + terms['rate_loss'] + terms['relative_motion_loss']).backward()
        model.optimizer_step()
        self.assertIsNotNone(model.optimizer.state.get(model.gs_translation))
        for parameter in (model.region_rotation, model.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertNotIn(parameter, model.optimizer.state)
            self.assertTrue(bool(parameter.eq(0).all()))
        torch.testing.assert_close(orientation, model.base_rotation, rtol=0, atol=0)

    def test_causal_transition_drops_old_increment_and_replays_initial_orientation(self):
        source = JointSpatialMotionTests().model()
        with torch.no_grad():
            source.region_rotation[1:, 1] = .02
            source.gs_rotation[:, 0] = .01
        decoded = JointSpatialMotionTests().commit(source)
        source.config.rotation_updates = False
        source.setup(decoded.xyz, decoded.rotation, source.extent, frame_index=2)
        self.assertTrue(bool(source.gs_rotation.eq(0).all()))
        self.assertTrue(bool(source.region_rotation.eq(0).all()))
        torch.testing.assert_close(source(1)[1], decoded.rotation, rtol=0, atol=0)
        replayed = JointSpatialMotionTests().commit(source)
        torch.testing.assert_close(replayed.rotation, decoded.rotation, rtol=0, atol=0)

    def test_restore_and_survivor_pruning_keep_motion_rotations_disabled(self):
        model = self.model()
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        rows = torch.arange(0, len(model.owner), 2)
        restored.retain_survivors(rows)
        for parameter in (restored.region_rotation, restored.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertTrue(bool(parameter.eq(0).all()))
        replayed = JointSpatialMotionTests().commit(restored)
        torch.testing.assert_close(replayed.rotation, model.base_rotation[rows], rtol=0, atol=0)

    def test_all_rotation_off_replays_births_without_transmitting_quaternions(self):
        for relative in (False, True):
            model = self.model()
            if relative:
                model.config.spatial_gate.region_motion = 'relative'
            prior = copy.deepcopy(model.frame_reference_state)
            xyz, orientation = model.materialized_pose()
            sources = torch.tensor([1, 17, 100])
            xyz = torch.cat((xyz, xyz[sources] + .003))
            orientation = torch.cat((orientation, orientation[sources]))
            model.register_materialized_layout(xyz, orientation, sources)
            with tempfile.TemporaryDirectory() as directory:
                inherited, explicit = Path(directory)/'inherited.bin', Path(directory)/'explicit.bin'
                model.export_motion(inherited, inherit_birth_rotation=True)
                model.export_motion(explicit)
                packet = inherited.read_bytes()
                raw = uncompressed_motion_packet(packet)
                self.assertEqual(raw[:4], b'SPG4' if relative else b'SPG3')
                self.assertEqual(len(uncompressed_motion_packet(explicit.read_bytes())) - len(raw),
                                 16 * len(sources))
                formats = (packet, raw, compress_motion_packet(raw, 'zlib'),
                           pack_layout(encode_motion_layout(raw), len(raw), MOTION_MAGICS))
                for encoded in formats:
                    decoded = decode_motion(encoded, prior)
                    torch.testing.assert_close(decoded.rotation.view(torch.int32),
                                               orientation.view(torch.int32), rtol=0, atol=0)
                    torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)

    def test_inherited_rotation_stream_rejects_hidden_rotation_updates(self):
        model = self.model()
        xyz, orientation = model.materialized_pose()
        model.register_materialized_layout(xyz, orientation, torch.empty(0, dtype=torch.long))
        with torch.no_grad():
            model.region_rotation[1, 0] = .02
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'preserve all'):
            model.export_motion(Path(directory)/'bad.bin', inherit_birth_rotation=True)


# ----- test_amplitude_refit -----
class AmplitudeRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = JointSpatialMotionTests().model()
        model.config.amplitude_refit_steps = 20
        model.config.proximal_gs_regularization = True
        model._hard_gate_frozen = True
        with torch.no_grad():
            model.region_translation[model.owner[0], 0] = 3 * model.steps[0]
            model.gs_translation[0, 0] = 2 * model.gs_steps[0]
            model.gs_translation[1, 0] = .2 * model.gs_steps[0]
        return model

    def test_refit_preserves_forward_then_updates_selected_blocks_from_images(self):
        model = self.model()
        before = model(80)[0].detach().clone()
        xyz, _ = model(81, training=True)
        torch.testing.assert_close(xyz, before, rtol=0, atol=0)
        self.assertTrue(model.amplitude_refit_active)
        self.assertEqual(float(model.regularization_loss(81)), 0.)
        support = model._amplitude_refit_support
        region_before = model.region_translation.detach().clone()
        gs_before = model.gs_translation.detach().clone()
        gates = model.opened.clone()
        ((xyz - model.base_xyz - .03).square().mean()).backward()
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)
        model.optimizer_step()
        self.assertTrue(torch.equal(model.opened, gates))
        self.assertGreater(float((model.region_translation - region_before).abs().sum()), 0.)
        self.assertGreater(float((model.gs_translation - gs_before).abs().sum()), 0.)
        self.assertFalse(bool(model.gs_translation[~support['gs_translation']].ne(0).any()))
        torch.testing.assert_close(model.region_translation[~support['region_translation']],
            region_before[~support['region_translation']], rtol=0, atol=0)

    def test_refit_masks_adam_momentum_and_survives_resume_and_retirement(self):
        model = self.model()
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
            exp_avg_sq=torch.ones_like(p))
        model(81, training=True)
        self.assertFalse(bool(model.optimizer.state[p]['exp_avg'][1:].ne(0).any()))
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        for current in (model, restored):
            xyz, _ = current(82, training=True)
            ((xyz - current.base_xyz - .03).square().mean()).backward()
            current.optimizer_step()
            current.retain_survivors(torch.arange(len(current.owner))[::2])
            self.assertEqual(len(current._amplitude_refit_support['gs_translation']), len(current.owner))
        torch.testing.assert_close(model.gs_translation, restored.gs_translation, rtol=0, atol=0)
        torch.testing.assert_close(model.region_translation, restored.region_translation, rtol=0, atol=0)

    def test_refit_requires_frozen_gates_and_valid_schedule(self):
        model = self.model()
        model._hard_gate_frozen = False
        with self.assertRaises(ValueError):
            model(81, training=True)
        for count in (True, -1, 31, 100):
            model.config.amplitude_refit_steps = count
            with self.assertRaises(ValueError):
                model.config.validate()


# ----- test_gate_candidate_refresh -----
class GateCandidateRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = JointSpatialMotionTests().model(initial_logit=-3.)
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
        helper = JointSpatialMotionTests()
        model = helper.model()
        model.config.spatial_gate.temporal_logit_decay = .5
        with torch.no_grad():
            model.gate_logits.fill_(-6.)
        decoded = helper.commit(model)
        opened = decoded.spatial_open.clone()
        model.setup(decoded.xyz, decoded.rotation, 2., frame_index=2)
        torch.testing.assert_close(model.gate_logits, torch.full_like(model.gate_logits, -3.), rtol=0, atol=0)
        self.assertTrue(torch.equal(model.opened, opened))


# ----- test_spatial_initialization -----
MOTION = importlib.import_module('pipeline.model.spatial_gated_motion')


class SpatialInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cloud(self):
        axis = torch.linspace(-1., 1., 16)
        core = torch.cartesian_prod(axis, axis, axis)
        return torch.cat((core, torch.tensor([[-100., -100., -100.],
                                             [100., 100., 100.]])))

    def model(self, xyz):
        config = RIGID.RigidTreeConfig()
        config.spatial_gate.max_depth = 3
        config.spatial_gate.initialization = 'robust_median'
        config.validate()
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        model = MOTION.SpatialGatedMotion(config)
        model.setup(xyz, q, extent=2., frame_index=1)
        return model

    def test_farther_outliers_cannot_move_supported_split_planes(self):
        xyz = self.cloud()
        pushed = xyz.clone()
        pushed[-2:] *= 10000.
        support = LAYOUT.spatial_initialization_support(xyz, 'robust_median')
        self.assertFalse(bool(support[-2:].any()))
        first = LAYOUT.spatial_layout(xyz, 5, True, initialization='robust_median')
        second = LAYOUT.spatial_layout(pushed, 5, True, initialization='robust_median')
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # All points, including the farthest two, retain a valid full path.
        self.assertEqual(first[0].shape, (len(xyz), 5))
        routed = LAYOUT.spatial_route(pushed, first[3], 5)
        torch.testing.assert_close(routed, first[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1][routed[:, 1:]], routed[:, :-1])

    def test_support_never_prunes_or_clamps_model_geometry(self):
        xyz = self.cloud()
        original = xyz.clone()
        model = self.model(xyz)
        torch.testing.assert_close(model.base_xyz, original, rtol=0, atol=0)
        torch.testing.assert_close(model(1)[0], original, rtol=0, atol=0)
        self.assertEqual(len(model.owner), len(original))
        self.assertEqual(len(model.parents), 585)

    def test_degenerate_and_invalid_clouds_have_defined_behavior(self):
        xyz = torch.ones(32, 3)
        result = LAYOUT.spatial_layout(xyz, 3, True, initialization='robust_median')
        self.assertTrue(bool(torch.isfinite(result[3]).all()))
        torch.testing.assert_close(LAYOUT.spatial_route(xyz, result[3], 3), result[0])
        for q in (0., .5, 1., float('nan'), True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                LAYOUT.spatial_layout(xyz, 3, initialization='robust_median', distance_quantile=q)
        for bad in (torch.zeros(0, 3), torch.tensor([[float('nan'), 0., 0.]]),
                    torch.tensor([[float('inf'), 0., 0.]])):
            with self.assertRaises(ValueError):
                LAYOUT.spatial_layout(bad, 3)

    def test_stream_handles_births_and_next_frame_without_new_codec_fields(self):
        model = self.model(self.cloud())
        reference = copy.deepcopy(model.frame_reference_state)
        source = torch.tensor([0, len(model.owner) - 1])
        with torch.no_grad():
            model.gs_translation[0, 0] = 2.5
        xyz, q = model.materialized_pose()
        newborn = xyz[source].clone()
        newborn[0] = torch.tensor([200., -200., 100.])
        model.register_materialized_layout(torch.cat((xyz, newborn)),
                                           torch.cat((q, q[source])), source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.bin'
            model.export_motion(path)
            decoded = CODEC.decode_motion(path.read_bytes(), reference)
            self.assertEqual(len(decoded.xyz), len(xyz) + 2)
            torch.testing.assert_close(decoded.xyz, torch.cat((xyz, newborn)), rtol=0, atol=1e-6)
            expected = LAYOUT.spatial_route(newborn, reference.spatial_centers, 3)
            torch.testing.assert_close(decoded.spatial_paths[-2:], expected, rtol=0, atol=0)
            restored = MOTION.SpatialGatedMotion(model.config)
            restored.restore(copy.deepcopy(model.capture()))
            for current in (model, restored):
                current.setup(decoded.xyz, decoded.rotation, 2., frame_index=2)
            torch.testing.assert_close(model.spatial_paths, restored.spatial_paths, rtol=0, atol=0)

    def test_median_checkpoint_is_compatible_but_cannot_resume_as_robust(self):
        xyz = self.cloud()
        config = RIGID.RigidTreeConfig()
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        original = MOTION.SpatialGatedMotion(config)
        original.setup(xyz, q, 2., frame_index=1)
        old = copy.deepcopy(original.capture())
        old['config']['spatial_gate'].pop('initialization')
        old['config']['spatial_gate'].pop('initialization_distance_quantile')
        resumed = MOTION.SpatialGatedMotion(config)
        resumed.restore(old)
        torch.testing.assert_close(resumed.spatial_paths, original.spatial_paths, rtol=0, atol=0)
        robust = copy.deepcopy(config)
        robust.spatial_gate.initialization = 'robust_median'
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            MOTION.SpatialGatedMotion(robust).restore(old)


# ----- test_motion_contributions -----
OBSERVED = importlib.import_module('pipeline.model.contribution_rasterizer')


class MotionEvidenceHookTests(unittest.TestCase):


    def test_actual_motion_counter_uses_compositing_not_frustum(self):
        source = importlib.import_module('pipeline.module')
        motion = SpatialGatedMotionTests().model()
        motion.observations.zero_()
        config = source.ModuleConfig(effective_motion_observations=True)
        adapter = SimpleNamespace(mark_observed=motion.mark_observed, stage=lambda step: 'free',
                                  optimizer_step=lambda **kwargs: [])
        module = SimpleNamespace(trainer=SimpleNamespace(global_step=1), motion_committed=False,
            motion=adapter, config=config, birth_control={}, motion_steps=200)
        frustum = torch.ones_like(motion.observations, dtype=torch.bool)
        used = torch.zeros_like(frustum)
        used[1:3] = True
        source.Module.post_training_step(module, dict(visibility_filter=frustum,
                                                    motion_contribution_filter=used))
        self.assertTrue(torch.equal(motion.observations.gt(0), used))
        self.assertEqual(module.birth_control['motion_observed_samples'], 2)
        self.assertEqual(module.birth_control['motion_frustum_samples'], len(frustum))
        before = motion.observations.clone()
        with self.assertRaisesRegex(ValueError, 'compositing mask'):
            source.Module.post_training_step(module, dict(visibility_filter=frustum))
        self.assertTrue(torch.equal(before, motion.observations))

    def test_old_configuration_still_uses_frustum(self):
        source = importlib.import_module('pipeline.module')
        config = source.ModuleConfig()
        mark = Mock()
        module = SimpleNamespace(trainer=SimpleNamespace(global_step=1), motion_committed=False,
            motion=SimpleNamespace(mark_observed=mark, stage=lambda step: 'free',
                                   optimizer_step=lambda **kwargs: []),
            config=config, birth_control={}, motion_steps=200)
        visible = torch.tensor([True, False])
        source.Module.post_training_step(module, dict(visibility_filter=visible))
        self.assertIs(mark.call_args.args[0], visible)


@unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA contribution tests')
class MotionContributionGPUChecks(unittest.TestCase):
    def settings(self):
        from diff_gaussian_rasterization import GaussianRasterizationSettings
        from libgs.data.utils.graphics import getProjectionMatrix
        return GaussianRasterizationSettings(image_height=64, image_width=64,
            tanfovx=math.tan(.5), tanfovy=math.tan(.5), bg=torch.zeros(3, device='cuda'),
            scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'),
            projmatrix=getProjectionMatrix(.01,100,1.,1.).transpose(0,1).cuda(),
            sh_degree=1, campos=torch.zeros(3,device='cuda'), prefiltered=False, debug=False)

    def arguments(self, count=12):
        generator = torch.Generator(device='cuda').manual_seed(916)
        xyz = torch.randn(count,3,generator=generator,device='cuda')*.15
        xyz[:,2] += 2
        quaternion = torch.randn(count,4,generator=generator,device='cuda')
        quaternion /= quaternion.norm(dim=1,keepdim=True)
        return dict(means3D=xyz, means2D=torch.zeros_like(xyz),
            shs=torch.randn(count,4,3,generator=generator,device='cuda')*.1,
            opacities=torch.full((count,1),.7,device='cuda'),
            scales=torch.full((count,3),.06,device='cuda'), rotations=quaternion,
            cov3D_precomp=None, colors_precomp=None)

    def test_image_exact_and_all_image_gradients_match_native(self):
        from diff_gaussian_rasterization import GaussianRasterizer
        settings = self.settings()
        original = self.arguments()
        native = {k: None if v is None else v.detach().clone().requires_grad_(True)
                  for k,v in original.items()}
        observed = {k: None if v is None else v.detach().clone().requires_grad_(True)
                    for k,v in original.items()}
        native_image, native_radii = GaussianRasterizer(settings)(**native)
        image, radii, mask = OBSERVED.GaussianRasterizer(settings)(**observed)
        self.assertTrue(torch.equal(image, native_image))
        self.assertTrue(torch.equal(radii, native_radii))
        self.assertFalse(bool((mask.gt(0) & radii.eq(0)).any()))
        weights=torch.linspace(.1,1,image.numel(),device='cuda').reshape_as(image)
        (native_image.square()*weights).sum().backward()
        (image.square()*weights).sum().backward()
        for name in ('means3D','means2D','shs','opacities','scales','rotations'):
            torch.testing.assert_close(observed[name].grad,native[name].grad,rtol=2e-5,atol=2e-5)

    def test_transparent_occluded_out_of_view_and_visible(self):
        args=self.arguments(7)
        args.update(shs=None,colors_precomp=torch.ones(7,3,device='cuda'),
                    opacities=torch.ones(7,1,device='cuda'),
                    rotations=torch.tensor([[1.,0.,0.,0.]]*7,device='cuda'))
        args['means3D']=torch.tensor([[0.,0.,2.]]*4+[[0.,0.,3.],[.8,0.,2.],[100.,0.,2.]],device='cuda')
        args['scales']=torch.tensor([[.25]*3]*4+[[.0001]*3,[.1]*3,[.1]*3],device='cuda')
        args['opacities'][5]=0
        image,radii,mask=OBSERVED.GaussianRasterizer(self.settings())(**args)
        self.assertGreater(int(radii[4]),0)
        self.assertGreater(int(radii[5]),0)
        self.assertFalse(bool(mask[4]))  # Small rear point hidden by the front layers.
        self.assertFalse(bool(mask[5]))  # In-frustum transparent point.
        self.assertFalse(bool(mask[6]))  # Outside image bounds.
        self.assertTrue(bool(mask[:4].any()))


# ----- test_proximal_gs_regularization -----
class ProximalGSRegularizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = JointSpatialMotionTests().model()
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        return model

    def test_regularizers_train_regions_and_gates_but_not_gs_adam_moments(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:, 0] = .6 * model.steps[0]
            model.region_rotation[1:, 1] = .6 * model.steps[3]
            model.gs_translation[:, 0] = .7 * model.gs_steps[0]
            model.gs_rotation[:, 1] = .8 * model.gs_steps[3]
        terms = model.loss_terms(5, False)
        gradients = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
            [model.gs_translation, model.gs_rotation, model.region_translation,
             model.region_rotation, model.gate_logits], allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        for gradient in gradients[2:]:
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_soft_threshold_is_exact_metric_prox_and_cannot_cross_zero(self):
        model = self.model()
        model.config.relative_motion_loss_weight = 1.
        model.last_step = 1
        for p, group, scale in [(model.gs_translation, model.optimizer.param_groups[2], model.extent),
                                (model.gs_rotation, model.optimizer.param_groups[3], 1.)]:
            group['lr'] = .01
            model.optimizer.state[p] = dict(step=torch.tensor(1.),
                exp_avg=torch.zeros_like(p), exp_avg_sq=torch.full_like(p, 1-model.config.adam_betas[1]))
            with torch.no_grad():
                p.zero_()
                p[0] = torch.tensor([1e-9, -1e-9, 0.])
                p[1] = torch.tensor([.000501, -.000501, .2])
        original_t, original_r = model.gs_translation.detach().clone(), model.gs_rotation.detach().clone()
        model._proximal_relative_gs()
        for original, p, scale in [(original_t, model.gs_translation, model.extent),
                                   (original_r, model.gs_rotation, 1.)]:
            threshold = .01 / (len(model.owner) * scale * (1 + model.config.adam_eps))
            expected = original.sign() * (original.abs() - threshold).clamp_min(0)
            torch.testing.assert_close(p, expected, rtol=1e-6, atol=1e-10)
            self.assertTrue(bool((p * original >= 0).all()))
            self.assertTrue(bool(p.abs().le(original.abs()).all()))
            self.assertTrue(bool(p[0].eq(0).all()))
        self.assertEqual(float(_quantized(model.gs_rotation, model.gs_steps[3:])[1,0]), 0.)

    def test_one_real_update_stores_only_photometric_gs_second_moment(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = torch.linspace(-.003,.003,len(model.owner))
            model.gs_rotation[:, 1] = torch.linspace(-.002,.002,len(model.owner))
        xyz, rotation = model(1, training=True)
        image_proxy = (xyz-model.base_xyz-.0001).square().mean() + rotation[:,1:].square().mean()
        expected = torch.autograd.grad(image_proxy,[model.gs_translation,model.gs_rotation],retain_graph=True)
        terms = model.loss_terms(1,False)
        (image_proxy+terms['rate_loss']+terms['relative_motion_loss']).backward()
        model.optimizer_step()
        for p,gradient in zip((model.gs_translation,model.gs_rotation),expected):
            torch.testing.assert_close(model.optimizer.state[p]['exp_avg_sq'],
                gradient.square()*(1-model.config.adam_betas[1]),rtol=1e-6,atol=1e-20)

    def test_metric_and_shrink_survive_checkpoint_restore(self):
        model = self.model()
        with torch.no_grad():
            model.gs_rotation[:,0]=.000501
        model.last_step=1
        model.gs_rotation.grad=torch.full_like(model.gs_rotation,1e-6)
        model.optimizer_step()
        captured=copy.deepcopy(model.capture())
        restored=SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(captured)
        self.assertTrue(restored.proximal_gs_regularization)
        for a,b in [(model.gs_translation,restored.gs_translation),(model.gs_rotation,restored.gs_rotation)]:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for current in [model,restored]:
            current.last_step=2
            current.gs_rotation.grad=torch.full_like(current.gs_rotation,2e-6)
            current.optimizer_step()
        torch.testing.assert_close(model.gs_rotation,restored.gs_rotation,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
