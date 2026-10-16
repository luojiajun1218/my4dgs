"""Behavioral checks for learned subdivision, recoding, and sequential decoding."""

import copy
import importlib
import itertools
from pathlib import Path
import tempfile
import unittest

import torch
import numpy as np


RIGID = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
SPATIAL = importlib.import_module('pipeline.ReCon-GS.model.spatial_gated_motion')
CODEC = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
LAYOUT = importlib.import_module('pipeline.ReCon-GS.model.spatial_gate')


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


if __name__ == '__main__':
    unittest.main()
