"""CPU checks for regional alpha repair, reanchoring and recovery."""
import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from pipeline.model.birth_quantization import RAW_FIELDS, materialize_quantized_gaussians, model_raw
from pipeline.model.frame_plan_birth import FramePlanBirthModel
from pipeline.model.gaussian import BirthGaussianModel
from pipeline.model.spatial_gate import spatial_layout
from pipeline.module import BirthConfig, Module
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import decode_frame
from tests.test_frame_plan import checkpoint_harness, controller_fixture, cpu_gaussian_setup


class RegionalAlphaRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, directory, *, retire=False):
        module, views = controller_fixture(directory)
        module.config.birth.regional_color_birth = True
        module.config.birth.regional_alpha_repair = True
        module.config.birth.keep_incumbent_fixed_after_motion = True
        module.config.birth.keep_incumbent_alpha_after_motion = True
        module.config.rigid_tree.spatial_gate.final_joint_refine_steps = 0
        module.config.opacity_update.l1_weight = .05
        module.config.opacity_update.proximal_regularization = True
        module.motion_committed = True
        module.early_initialized = True
        module.births = FramePlanBirthModel(1, logit_opacity=True)
        module._plan_events = []
        module._plan_regional_alpha_state = None
        checkpoint_harness(module)
        parameter = module.opacity_update.delta
        with torch.no_grad():
            parameter.fill_(.04)
        module.motion.optimizer.state[parameter] = dict(step=torch.tensor(200.),
            exp_avg=torch.full_like(parameter, .2), exp_avg_sq=torch.full_like(parameter, .3))
        module.motion.prepare_plan_step(200)
        if retire:
            with torch.no_grad():
                parameter[0] = -module.opacity_update.alpha_reference[0]
            module._plan_cleanup(200)
        return module, views

    @staticmethod
    def units(module, triggered, *, new_points=0):
        count = len(module.gaussians._xyz)
        point_unit = np.zeros(count + new_points, dtype=np.int64)
        point_unit[np.asarray(triggered, dtype=np.int64)] = 1
        if new_points:
            point_unit[count:] = 2  # triggered new rows must never enable old rows
        return dict(point_unit=point_unit, unit_reason=np.array([1, 2, 2], dtype=np.int8))

    def image_step(self, module, views, step):
        module.trainer.global_step = step
        module._plan_prepare_rates(step)
        module.motion.optimizer.zero_grad(set_to_none=True)
        module.gaussians.rigid_alpha = module.opacity_update.alphas()
        image = module.renderer(next(iter(views.values())), module.background)['render']
        (image - .55).square().mean().backward()
        module.gaussians.rigid_alpha = None
        gradient = module.opacity_update.delta.grad.detach().clone()
        module._plan_regional_alpha_mask_gradient(step)
        return gradient

    def adam_and_prox(self, module, step):
        update = module.opacity_update
        module.motion.optimizer.step()
        update.optimizer_step(.00001, proximal=True, perform_step=False,
                              external_state=module.motion.optimizer.state[update.delta])
        module._plan_regional_alpha_restore(step)
        module.motion.optimizer.zero_grad(set_to_none=True)

    def test_mask_maps_survivors_and_freezes_parameter_moments_alpha_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory, retire=True)
            module._plan_regional_alpha_select(200, self.units(module, [0, 2], new_points=3))
            update, state = module.opacity_update, module._plan_regional_alpha_state
            rows = update.rows.clone()
            eligible = state['eligible']
            self.assertEqual(eligible.nonzero().flatten().tolist(), [int(rows[0]), int(rows[2])])
            self.assertFalse(bool(eligible[0]))  # old source row 0 was retired
            baseline_delta = update.delta.detach().clone()
            baseline_alpha = update.alphas().detach().clone()
            baseline_symbols = update.symbols().clone()
            adam = module.motion.optimizer.state[update.delta]
            initial_moments = {k: adam[k].clone() for k in ('exp_avg', 'exp_avg_sq')}
            gradient = self.image_step(module, views, 201)
            self.assertGreater(float(gradient[eligible].abs().sum()), 0.)
            self.assertGreater(float(gradient[~eligible].abs().sum()), 0.)
            self.assertEqual(float(update.delta.grad[~eligible].abs().sum()), 0.)
            self.adam_and_prox(module, 201)
            self.assertTrue(torch.equal(update.delta[~eligible], baseline_delta[~eligible]))
            self.assertTrue(torch.equal(update.alphas().detach()[~eligible[rows]], baseline_alpha[~eligible[rows]]))
            self.assertTrue(torch.equal(update.symbols()[~eligible], baseline_symbols[~eligible]))
            for key in initial_moments:
                self.assertTrue(torch.equal(adam[key][~eligible], initial_moments[key][~eligible]))
            self.assertGreater(float((update.delta[eligible] - baseline_delta[eligible]).abs().sum()), 0.)
            self.assertEqual(int(adam['step']), 201)  # scalar Adam clock remains shared
            audit = module.birth_control['regional_alpha_repair']['invariant_checks'][-1]
            for key in ('max_frozen_alpha_change', 'max_frozen_symbol_change',
                        'max_frozen_delta_change', 'max_frozen_moment_change'):
                self.assertEqual(audit[key], 0.)

    def test_250_unions_eligibility_and_reactivates_from_step200_state(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory, retire=True)
            module._plan_regional_alpha_select(200, self.units(module, [0]))
            state = module._plan_regional_alpha_state
            old_rows = module.opacity_update.rows
            new_row = int(old_rows[1])
            original_moment = state['baseline_moments']['exp_avg'][new_row].clone()
            self.image_step(module, views, 201)
            self.adam_and_prox(module, 201)
            torch.testing.assert_close(module.opacity_update.delta[new_row], state['baseline_delta'][new_row], rtol=0, atol=0)
            torch.testing.assert_close(module.motion.optimizer.state[module.opacity_update.delta]['exp_avg'][new_row],
                                       original_moment, rtol=0, atol=0)
            module._plan_regional_alpha_select(250, self.units(module, [1], new_points=2))
            self.assertTrue(bool(state['eligible'][old_rows[0]]))
            self.assertTrue(bool(state['eligible'][new_row]))
            event = module.birth_control['regional_alpha_repair']['mask_events'][-1]
            self.assertEqual(event['newly_eligible_old_count'], 1)
            self.assertEqual(event['eligible_old_count'], 2)
            gradient = self.image_step(module, views, 251)
            self.assertGreater(float(gradient[new_row].abs().sum()), 0.)
            self.adam_and_prox(module, 251)
            self.assertGreater(float((module.opacity_update.delta[new_row]-state['baseline_delta'][new_row]).abs()), 0.)

    def test_no_trigger_freezes_all_and_full_update_has_no_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory)
            module._plan_regional_alpha_select(200, self.units(module, []))
            baseline = module.opacity_update.delta.detach().clone()
            self.image_step(module, views, 201)
            self.adam_and_prox(module, 201)
            self.assertTrue(torch.equal(module.opacity_update.delta, baseline))
            self.assertEqual(module.birth_control['regional_alpha_repair']['mask_events'][0]['eligible_old_count'], 0)
        with tempfile.TemporaryDirectory() as directory:
            module, _ = self.fixture(directory)
            module.full_update_active = True
            module._plan_regional_alpha_select(200, self.units(module, []))
            self.assertIsNone(module._plan_regional_alpha_state)
            parameter = module.opacity_update.delta
            parameter.grad = torch.ones_like(parameter)
            module._plan_regional_alpha_mask_gradient(201)
            self.assertTrue(bool(parameter.grad.eq(1).all()))
            before = parameter.detach().clone()
            module.motion.optimizer.step()
            module._plan_regional_alpha_restore(201)
            self.assertGreater(float((parameter-before).abs().sum()), 0.)

    def test_config_requires_regional_frame_plan_and_postmotion_alpha(self):
        self.assertFalse(BirthConfig().regional_alpha_repair)
        for values in (dict(), dict(frame_training_plan=True),
                       dict(frame_training_plan=True, regional_color_birth=True)):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'Regional alpha repair'):
                BirthConfig(regional_alpha_repair=True, **values).validate(200)
        BirthConfig(frame_training_plan=True, regional_color_birth=True,
                    keep_incumbent_alpha_after_motion=True, residual_support_init=True,
                    regional_alpha_repair=True).validate(200)

    def test_selection_uses_real_birth_units_even_with_zero_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            module, views = self.fixture(directory)
            module.config.birth.residual_support_init = True
            module.config.birth.max_new_gaussians = 0
            module.config.birth.residual.max_candidates = 0
            module.config.birth.regional_color_min_support = 32.
            module.config.birth.regional_color_min_views = 2
            module.config.birth.regional_color_l1_threshold = .03
            module._plan_used_quota = 0
            module._plan_events = [{}]
            module.birth_control['early_replacement'] = {}
            module._plan_step_views = list(views.values())[:4]
            xyz = module.gaussians._xyz.detach()
            paths, parents, opened, centers, routing = spatial_layout(xyz, 5, return_centers=True,
                                                                       return_routing=True)
            layout = SimpleNamespace(spatial_paths=paths, spatial_centers=centers,
                **{k:routing.get(k) for k in ('spatial_support_ball','spatial_leaf_centers','spatial_leaf_nodes')})
            module._regional_color_layout = (layout, 'fixed fixture layout')
            current_step = [200]
            def evidence(proposal, cache):
                positions = proposal.gaussians.get_xyz.detach().cpu().numpy()
                den = np.full((len(cache), len(positions)), 40.)
                trigger = positions[:, 0] < .5 if current_step[0] == 200 else positions[:, 0] >= .5
                return den*np.where(trigger, .1, .01)[None], den, [dict(camera=x['name']) for x in cache]
            with patch('pipeline.regional_topology.native_colour_evidence', side_effect=evidence):
                module._plan_birth(200)
                state = module._plan_regional_alpha_state
                expected = xyz[:, 0] < .5
                self.assertTrue(torch.equal(state['eligible'][module.opacity_update.rows], expected))
                self.assertEqual(module._plan_events[-1]['born'], 0)
                # The 250 statistics use the complete old+new scene, but the
                # alpha mask can only refer to current old predecessor rows.
                reference = module._adaptive_reference()
                seed = {name:value[[0]].clone() for name,value in reference.items()}
                seed['opacity'].fill_(torch.logit(torch.tensor(.1)))
                module.births.add(seed, torch.tensor([0]), torch.tensor([0]), 200,
                                  module.motion.optimizer, module.config.gaussian_stage2)
                module._plan_events.append({})
                current_step[0] = 250
                module._plan_birth(250)
                self.assertTrue(bool(state['eligible'][module.opacity_update.rows].all()))
                self.assertEqual(len(state['eligible']), len(module.opacity_update.delta))
                self.assertEqual(module._plan_events[-1]['born'], 0)
                events = module.birth_control['regional_alpha_repair']['mask_events']
                self.assertEqual([row['step'] for row in events], [200, 250])
                self.assertEqual(events[-1]['old_count'], len(xyz))
                self.assertEqual(events[-1]['eligible_old_count'], len(xyz))

    def test_step250_checkpoint_restore_next_update_is_identical(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            module, views = self.fixture(directory, retire=True)
            module._plan_regional_alpha_select(200, self.units(module, [0, 2]))
            self.image_step(module, views, 201)
            self.adam_and_prox(module, 201)
            module._plan_regional_alpha_select(250, self.units(module, [1], new_points=2))
            module.early_shared_steps = module.trainer.global_step = 250
            packet = dict(global_step=250, current_frame=1)
            Module.on_save_checkpoint(module, packet)
            clone, clone_views = self.fixture(directory)
            Module.on_load_checkpoint(clone, copy.deepcopy(packet))
            for instance, camera in ((module, views), (clone, clone_views)):
                instance.early_shared_steps = 250
                instance.trainer.global_step = 251
                instance._plan_prepare_rates(251)
                instance.motion.optimizer.zero_grad(set_to_none=True)
                instance.gaussians.rigid_alpha = instance.opacity_update.alphas()
                image = instance.renderer(next(iter(camera.values())), instance.background)['render']
                (image - .55).square().mean().backward()
                instance.gaussians.rigid_alpha = None
                instance._plan_post_step([], 251)
            self.assertTrue(torch.equal(module.opacity_update.delta, clone.opacity_update.delta))
            self.assertTrue(torch.equal(module.opacity_update.alphas(), clone.opacity_update.alphas()))
            self.assertTrue(torch.equal(module._plan_regional_alpha_state['eligible'],
                                        clone._plan_regional_alpha_state['eligible']))
            for key in ('exp_avg', 'exp_avg_sq', 'step'):
                a = module.motion.optimizer.state[module.opacity_update.delta][key]
                b = clone.motion.optimizer.state[clone.opacity_update.delta][key]
                self.assertTrue(torch.equal(a, b), key)

    def test_full_frame_retirement_resume_and_real_six_attribute_decode(self):
        def configure(module):
            module.config.birth.regional_color_birth = True
            module.config.birth.regional_alpha_repair = True
            module.config.birth.residual_support_init = True
            module.config.birth.keep_incumbent_fixed_after_motion = True
            module.config.birth.keep_incumbent_alpha_after_motion = True
            module.config.birth.regional_color_min_support = 32.
            module.config.birth.regional_color_min_views = 2
            module.config.birth.regional_color_l1_threshold = .03
            module.config.rigid_tree.spatial_gate.final_joint_refine_steps = 0
            module._plan_regional_alpha_state = None
            checkpoint_harness(module)

        def evidence(proposal, cache):
            positions = proposal.gaussians.get_xyz.detach().cpu().numpy()
            den = np.full((len(cache), len(positions)), 40.)
            num = den * np.where(positions[:, 0] < .5, .1, .01)[None]
            return num, den, [dict(camera=x['name']) for x in cache]

        def geometry(raw, rows, cache, step):
            seed = {name:raw[name][rows].clone() for name in ('xyz','rotation','logscale')}
            return seed, torch.ones(len(rows), dtype=torch.bool), dict(policy='synthetic')

        with tempfile.TemporaryDirectory() as directory, patch.object(
                BirthGaussianModel, 'training_setup', cpu_gaussian_setup), \
                patch('pipeline.regional_topology.native_colour_evidence', side_effect=evidence), \
                patch('pipeline.frame_plan.residual_support_seed', side_effect=geometry):
            module, _ = controller_fixture(directory)
            configure(module)
            xyz = module.gaussians._xyz.detach()
            paths, parents, opened, centers, routing = spatial_layout(xyz, 5, return_centers=True,
                                                                       return_routing=True)
            layout = SimpleNamespace(spatial_paths=paths, spatial_centers=centers,
                **{k:routing.get(k) for k in ('spatial_support_ball','spatial_leaf_centers','spatial_leaf_nodes')})
            module._regional_color_layout = (layout, 'fixed fixture layout')
            module.finalize_frame = MethodType(Module.finalize_frame, module)
            prior, initial = copy.deepcopy(module.motion.frame_reference_state), model_raw(module.gaussians)
            writer = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(writer, 0, prior)
            module.trainer.decoder_reference_path = writer.decoder_reference_path
            runner = SimpleNamespace(module=module,
                config=SimpleNamespace(motion_views_per_step=4, birth_views_per_step=4),
                _views_per_step=lambda step:4)
            cleanup = module._plan_cleanup
            @torch.no_grad()
            def retire(step):
                if step == 250:
                    rows = module.opacity_update.rows
                    eligible = module._plan_regional_alpha_state['eligible'][rows]
                    row = rows[eligible][0]
                    module.opacity_update.delta[row] = -module.opacity_update.alpha_reference[row]
                return cleanup(step)
            module._plan_cleanup = retire
            clone = None
            for step in range(1, 301):
                module.trainer.global_step = step
                module.pre_training_step()
                Trainer._fit_views(runner, None, step)
                if step >= 200:
                    state = module._plan_regional_alpha_state
                    rows = module.opacity_update.rows
                    frozen = ~state['eligible'][rows]
                    self.assertTrue(torch.equal(module.opacity_update.delta[rows[frozen]],
                                                state['baseline_delta'][rows[frozen]]))
                if step == 250:
                    self.assertEqual(module._plan_events[-1]['removed_old'], 1)
                    packet = dict(global_step=250, current_frame=1)
                    Module.on_save_checkpoint(module, packet)
                    clone, _ = controller_fixture(directory)
                    configure(clone)
                    Module.on_load_checkpoint(clone, copy.deepcopy(packet))
                    clone.trainer.global_step = 251
                    clone.pre_training_step()
                    other = SimpleNamespace(module=clone, config=runner.config, _views_per_step=lambda step:4)
                    Trainer._fit_views(other, None, 251)
                if step == 251:
                    self.assertTrue(torch.equal(module.opacity_update.delta, clone.opacity_update.delta))
                    self.assertTrue(torch.equal(module.births.source_rows, clone.births.source_rows))
                    for name in RAW_FIELDS:
                        self.assertTrue(torch.equal(model_raw(module.births)[name], model_raw(clone.births)[name]))
            self.assertTrue(module.frame_finalized)
            self.assertGreater(module.frame_report['birth_count'], 0)
            decoded, raw = decode_frame(Path(directory)/'frame_storage'/'rigid_tree', 1, prior, initial)
            self.assertEqual(decoded.digest(), module.motion.codec_previous_state.digest())
            self.assertTrue(torch.equal(decoded.birth_sources, module.motion.pending_birth_sources))
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], model_raw(module.gaussians)[name].cpu()), name)
            checks = module.birth_control['regional_alpha_repair']['invariant_checks']
            self.assertEqual([row['step'] for row in checks], [201, 250, 251, 300])
            for row in checks:
                self.assertEqual(row['max_frozen_alpha_change'], 0.)
                self.assertEqual(row['max_frozen_symbol_change'], 0.)
            Module.begin_next_frame(module)
            module.current_frame = 2
            Module.prepare_frame(module)
            self.assertIsNone(module._plan_regional_alpha_state)


class ReanchorAlphaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, directory, *, enabled=True, all_low=False):
        module, _ = controller_fixture(directory)
        module.config.birth.regional_color_birth = enabled
        module.config.birth.keep_incumbent_fixed_after_motion = True
        module.config.birth.keep_incumbent_alpha_after_motion = True
        module.motion_committed = True
        module.early_initialized = True
        module.births = FramePlanBirthModel(1, logit_opacity=True)
        module.births.spatial_lr_scale = 1.
        module._plan_events = []
        reference = module._adaptive_reference()
        reference['opacity'] = module.opacity_update.canonical_logits().to(module.device)
        # Source 1 can represent a value just above .01. Reanchoring the
        # same value to source 0 rounds it below .01 on the .01 logit grid.
        source_logit = reference['opacity'][1, 0]
        threshold = math.log(.01 / .99)
        k = math.ceil((threshold - float(source_logit)) / module.config.birth.new_opacity_step)
        low = source_logit + k * module.config.birth.new_opacity_step
        target = reference['opacity'][0, 0]
        reanchored = target + torch.round((low-target) / module.config.birth.new_opacity_step) * module.config.birth.new_opacity_step
        self.assertGreaterEqual(float(low.sigmoid()), .01)
        self.assertLess(float(reanchored.sigmoid()), .01)
        safe = torch.logit(torch.tensor(.1))
        for index, step in enumerate((200, 250)):
            raw = {name: value[[0, 0]].clone() for name, value in reference.items()}
            raw['opacity'][:, 0] = torch.tensor(([low, low] if all_low else
                                                  [low, safe] if index == 0 else [safe, low]))
            parents = torch.tensor([3 + 2*index, 4 + 2*index])
            module.births.add(raw, parents, torch.tensor([1, 1]), step,
                              module.motion.optimizer, module.config.gaussian_stage2)
        # Give every parameter identifiable row-wise state in both cohorts.
        # Subsequent selection must retain the right source row and moments.
        optimizer = module.motion.optimizer
        for cohort, batch in enumerate(module.births.cohorts):
            for parameter in batch['parameters'].values():
                shape = (2,) + (1,) * (parameter.ndim - 1)
                state_values = torch.arange(2, dtype=parameter.dtype).reshape(shape).expand_as(parameter) + 10*cohort
                optimizer.state[parameter] = dict(step=torch.tensor(7.),
                    exp_avg=state_values.clone(), exp_avg_sq=(state_values + 1).clone())
        return module

    def assert_survivors_and_state(self, module, step, old_removed=0):
        before = module.make_birth_view(reference=module._adaptive_reference()).get_opacity.flatten()
        self.assertTrue(bool((before >= .01).all()))
        old_count = len(module.gaussians._xyz)
        module._plan_cleanup(step)
        event = module._plan_events[-1]
        self.assertEqual(event['removed_old'], old_removed)
        self.assertEqual(event['removed_new'], 2)
        self.assertEqual(event['new_count_before_cleanup'], 4)
        self.assertEqual(event['new_count_before_reanchor'], 4)
        self.assertEqual(event['post_reanchor_low_alpha_pruned'], 2)
        self.assertEqual(event['new_count_after_reanchor'], 2)
        self.assertFalse(event['post_reanchor_alpha_check_deferred'])
        self.assertEqual(len(module.gaussians._xyz), old_count-old_removed)
        self.assertEqual(module.births.parent_rows.tolist(), [4, 5])
        self.assertEqual(module.births.source_rows.tolist(), [0, 0])
        self.assertEqual([len(c['parameters']['_xyz']) for c in module.births.cohorts], [1, 1])
        for cohort, batch in enumerate(module.births.cohorts):
            for name, parameter in batch['parameters'].items():
                state = module.motion.optimizer.state[parameter]
                self.assertEqual(int(state['step']), 7)
                expected = 1 if cohort == 0 else 10
                torch.testing.assert_close(state['exp_avg'], torch.full_like(parameter, expected))
                torch.testing.assert_close(state['exp_avg_sq'], torch.full_like(parameter, expected+1))
                group = next(g for g in module.motion.optimizer.param_groups
                             if g.get('name') == f'early_plan_{cohort}_{name}')
                self.assertIs(group['params'][0], parameter)
        view = module.make_birth_view(reference=module._adaptive_reference())
        self.assertTrue(bool((view.get_opacity.flatten() >= .01).all()))

    def test_reanchor_alpha_at_all_three_cleanup_boundaries(self):
        for step in (200, 250, 300):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                self.assert_survivors_and_state(self.fixture(directory), step)

    def test_no_surviving_newborns_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.fixture(directory, all_low=True)
            module._plan_cleanup(300)
            self.assertEqual(module._plan_events[-1]['post_reanchor_low_alpha_pruned'], 4)
            self.assertEqual(module._plan_events[-1]['new_count_after_reanchor'], 0)
            self.assertIsNone(module.make_birth_view(reference=module._adaptive_reference()))
            self.assertEqual(module.births.source_rows.numel(), 0)
            self.assertEqual([len(batch['parameters']['_xyz']) for batch in module.births.cohorts], [0, 0])

    def test_counts_separate_initial_and_post_reanchor_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.fixture(directory)
            with torch.no_grad():
                module.births.cohorts[0]['parameters']['_opacity'][0] = torch.logit(torch.tensor(.005))
            module._plan_cleanup(250)
            event = module._plan_events[-1]
            self.assertEqual(event['new_count_before_cleanup'], 4)
            self.assertEqual(event['new_count_before_reanchor'], 3)
            self.assertEqual(event['post_reanchor_low_alpha_pruned'], 1)
            self.assertEqual(event['new_count_after_reanchor'], 2)
            self.assertEqual(event['removed_new'], 2)
            self.assertEqual(module.births.parent_rows.tolist(), [4, 5])

    def test_default_off_keeps_existing_reanchor_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            module = self.fixture(directory, enabled=False)
            module._plan_cleanup(300)
            self.assertEqual(len(module.births._xyz), 4)
            alpha = module.make_birth_view(reference=module._adaptive_reference()).get_opacity.flatten()
            self.assertEqual(int((alpha < .01).sum()), 2)
            self.assertNotIn('post_reanchor_low_alpha_pruned', module._plan_events[-1])

    def test_final_encoded_and_decoded_births_match_filtered_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                BirthGaussianModel, 'training_setup', cpu_gaussian_setup):
            module = self.fixture(directory)
            checkpoint_harness(module)
            prior = copy.deepcopy(module.motion.frame_reference_state)
            initial = model_raw(module.gaussians)
            writer = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(writer, 0, prior)
            module.trainer.decoder_reference_path = writer.decoder_reference_path
            module.trainer.global_step = module.early_shared_steps = 300
            module.birth_control['early_replacement'] = dict(actual_birth_cohorts=2, source_count=4)
            # Retire a real old GS too, so the motion packet survivor mapping
            # and the rebased birth-source IDs are independently decoded.
            index = module.opacity_update.rows[-1]
            with torch.no_grad():
                module.opacity_update.delta[index] = -module.opacity_update.alpha_reference[index]
            self.assert_survivors_and_state(module, 300, old_removed=1)
            reference = module._adaptive_reference()
            expected, sources = materialize_quantized_gaussians(reference,
                module.make_birth_view(reference=reference))
            Module.finalize_frame(module)
            stream_dir = Path(directory) / 'frame_storage' / 'rigid_tree'
            state, decoded = decode_frame(stream_dir, 1, prior, initial)
            torch.testing.assert_close(state.birth_sources, sources.cpu(), rtol=0, atol=0)
            self.assertEqual(len(sources), 2)
            self.assertEqual(module.frame_report['birth_count'], 2)
            self.assertTrue(bool((decoded['opacity'][-2:].sigmoid() >= .01).all()))
            for name in RAW_FIELDS:
                torch.testing.assert_close(decoded[name], expected[name].cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
