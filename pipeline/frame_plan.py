"""Implementation of docs/frame_training_plan.md, selected explicitly in YAML."""
import math
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from libgs.model.merged_gaussian import MergedGaussianModel
from .model.birth_quantization import (RAW_FIELDS, RAW_ATTRIBUTES, model_raw,
    QuantizedNewGaussianView, attribute_steps,
    reanchor_reconstruction_tolerance)
from .model.frame_plan_birth import FramePlanBirthModel, nearest_old_references
from .model.early_replacement_model import _rotate
from .model.residual_birth import RawView
from .model.support_reincarnation_proposal import _native_responsibility, _sample
from .model.adaptive_birth_storage import newborn_storage_terms
from .model.residual_support_geometry import residual_support_seed,residual_support_opacity


def effective_final_joint_refine_steps(module):
    if getattr(module.config.birth, 'keep_incumbent_fixed_after_motion', False):
        return 0
    return module.config.rigid_tree.spatial_gate.final_joint_refine_steps


def final_joint_scheduled(module, step):
    """The optional tail fits old motion with the complete newborn population."""
    if not getattr(module.config.birth, 'frame_training_plan', False):
        return False
    count = effective_final_joint_refine_steps(module)
    return bool(count and 301-count <= step <= 300)


def regional_attribute_repair_active(module):
    return bool(getattr(module, 'full_update_active', False) and
                getattr(getattr(module.config, 'full_update', None), 'regional_repair', False))


def gradient_probe_step(module, step):
    return bool(regional_attribute_repair_active(module) and step in regional_probe_steps(module))


def regional_probe_steps(module):
    count = getattr(module.config.full_update, 'probe_views_per_round', 2)
    if getattr(module.config.full_update, 'early_joint_repair', False):
        return tuple(range(101, 101 + count))
    return tuple(range(201, 201 + count)) + tuple(range(251, 251 + count))


def regional_point_budget(module, count):
    fraction = getattr(module.config.full_update, 'point_budget_fraction', None)
    return min(1000, count // 10) if fraction is None else int(count * fraction)


def fitting_old_motion(module):
    # motion_committed is also the newborn-render/checkpoint readiness flag.
    # Keep it true after birth200, including while old motion learns again.
    full_update = bool(getattr(module, 'full_update_active', False))
    if regional_attribute_repair_active(module):
        return module.frame_prepared and not module.frame_finalized
    return module.frame_prepared and (not module.motion_committed or (
        not module.frame_finalized and getattr(module, 'early_initialized', False) and
        (full_update or final_joint_scheduled(module, module.motion.last_step))))


def _fixed_incumbent(module, step):
    if regional_attribute_repair_active(module):
        return False
    return bool(step > 200 and
        getattr(module.config.birth, 'keep_incumbent_fixed_after_motion', False) and
        not getattr(module, 'full_update_active', False))


def _fixed_old_alpha(module, step):
    color_only = (regional_attribute_repair_active(module) and step > 200 and
                  getattr(module.config.full_update, 'color_only', False))
    return bool(color_only or (_fixed_incumbent(module, step) and not
                getattr(module.config.birth, 'keep_incumbent_alpha_after_motion', False)))


def frame_plan_phase(module, step):
    """Name the actual optimization phase consistently in timings and logs."""
    tree = getattr(module.config, 'rigid_tree', None)
    discovery_steps = getattr(tree, 'shared_discovery_steps', 100)
    if step <= discovery_steps:
        return 'shared_gate'
    if step <= 100:
        return 'joint_gate_residual_early'
    if step <= 150:
        return 'fixed_gate_residual'
    if step <= 180:
        return 'joint_gate_residual'
    if step <= 200:
        return 'fixed_gate_motion'
    if final_joint_scheduled(module, step):
        return 'final_joint_refine'
    if regional_attribute_repair_active(module):
        return 'regional_attribute_repair'
    return 'newborn_fixed_incumbent' if _fixed_old_alpha(module, step) else 'newborn_old_alpha'


def regional_alpha_repair_active(module):
    return bool(getattr(module.config.birth, 'regional_alpha_repair', False) and
                not getattr(module, 'full_update_active', False))


class FramePlanController:
    @torch.no_grad()
    def _plan_gradient_probe(self, step, package):
        state = self._plan_regional_attribute_state
        count = len(state['eligible'])
        score = torch.zeros(count, device=self.device)
        scales = {}
        early_attributes = (getattr(self.config.full_update, 'early_joint_repair', False) and
                            getattr(self.config.full_update, 'early_probe_attribute_only', False))
        for name, parameter in self._regional_attribute_parameters().items():
            if early_attributes and name not in ('logscale', 'sh_dc', 'sh_rest'):
                continue
            grad = parameter.grad
            if grad is None:
                scales[name] = None
                continue
            magnitude = grad.detach().reshape(count, -1).norm(dim=1)
            if not torch.isfinite(magnitude).all():
                raise ValueError(f'Non-finite gradient during point probe: {name}')
            positive = magnitude[magnitude > 0]
            if positive.numel():
                scale = positive.median().clamp_min(1e-20)
                score.add_(magnitude / scale)
                scales[name] = float(scale)
            else:
                scales[name] = None
        views_per_round = getattr(self.config.full_update, 'probe_views_per_round', 2)
        if views_per_round == 4 and getattr(self.config.full_update, 'probe_opacity_weight', True):
            alpha = self._render_opacity_alpha().detach().reshape(-1)
            if alpha.shape != (count,) or not bool(torch.isfinite(alpha).all()):
                raise ValueError('Probe opacity does not match the old GS layout')
            score *= alpha.clamp(0, 1).sqrt()
        previous = state.get('probe_score')
        state['probe_score'] = score if previous is None else previous + score
        state.setdefault('probe_steps', []).append(int(step))
        self.birth_control['regional_attribute_repair'].setdefault('probes', []).append(dict(
            step=int(step), camera=str(self._plan_step_views[0].path), median_gradient=scales))
        info = self.birth_control['regional_attribute_repair']
        info['selection_only_views'] = len(info['probes'])
        info['training_views_at_selection'] = len({item['camera'] for item in info['probes']})
        self.motion.optimizer.zero_grad(set_to_none=True)
        early = getattr(self.config.full_update, 'early_joint_repair', False)
        finish_steps = (100 + views_per_round,) if early else (200 + views_per_round, 250 + views_per_round)
        if step in finish_steps:
            first_round = early or step == 200 + views_per_round
            start = 101 if early else 201 if first_round else 251
            if state['probe_steps'] != list(range(start, start + views_per_round)):
                raise ValueError('Gradient probe steps are incomplete or out of order')
            scores = state.pop('probe_score')
            state['probe_steps'] = []
            eligible = state['eligible']
            available = (~eligible) & (scores > 0)
            total_budget = regional_point_budget(self, count)
            # Reserve half the budget for the second probe. If the first
            # probe has fewer visible candidates, the second can use the rest.
            round_limit = (total_budget if early else (total_budget + 1) // 2) if first_round else total_budget - int(eligible.sum())
            budget = max(0, min(round_limit, total_budget - int(eligible.sum()), int(available.sum())))
            if budget:
                ranked = torch.argsort(scores.masked_fill(~available, -1), descending=True, stable=True)
                eligible[ranked[:budget]] = True
            info = self.birth_control['regional_attribute_repair']
            info['selection_step'] = int(step)
            info['selected_count'] = int(eligible.sum())
            info['selected_fraction'] = info['selected_count'] / count if count else 0.
            info.setdefault('selection_rounds', []).append(dict(step=int(step), added=budget,
                selected_count=info['selected_count']))

    def _regional_attribute_parameters(self):
        return dict(xyz=self.motion.gs_translation, rotation=self.motion.gs_rotation,
                    opacity=self.opacity_update.delta,
                    **self.full_attribute_update.parameters)

    @torch.no_grad()
    def _plan_snapshot_regional_attributes(self, eligible):
        count = len(self.gaussians._xyz)
        identity = torch.arange(count, device=self.device)
        if (not torch.equal(self.motion.previous_rows, identity) or
                not torch.equal(self.opacity_update.rows, identity)):
            raise ValueError('Regional attribute repair needs the unchanged old-GS layout')
        if eligible.dtype != torch.bool or eligible.shape != (count,) or eligible.device != identity.device:
            raise ValueError('Regional attribute repair selection must match the old-GS layout')
        snapshots = {}
        for name, parameter in self._regional_attribute_parameters().items():
            if len(parameter) != count:
                raise ValueError(f'Regional attribute {name} lost row alignment')
            adam = self.motion.optimizer.state.get(parameter, {})
            snapshots[name] = dict(value=parameter.detach().clone(), moments={
                key: value.detach().clone() for key, value in adam.items()
                if torch.is_tensor(value) and value.shape == parameter.shape})
        self._plan_regional_attribute_state = dict(
            eligible=eligible, parameters=snapshots)
        scale_shared = getattr(self, 'regional_logscale', None)
        if scale_shared is not None:
            scale_shared.bind(self.motion.owner, len(self.motion.parents), torch.unique(self.motion.owner), self.motion.optimizer)
        shared = getattr(self, 'feature_dc_update', None)
        if (shared is not None and shared.regional and
                not (self.config.full_update.early_joint_repair and not self.motion_committed)):
            if shared.ready:
                shared.freeze_owner()
            else:
                shared.bind(self.motion.owner, len(self.motion.parents),
                            torch.unique(self.motion.owner), self.motion.optimizer)

    @torch.no_grad()
    def _plan_select_regional_attributes(self, packages):
        if self._plan_regional_attribute_state is not None:
            raise ValueError('Regional attribute selection already exists for this frame')
        count = len(self.gaussians._xyz)
        self._plan_snapshot_regional_attributes(torch.zeros(count, dtype=torch.bool, device=self.device))
        self.birth_control['regional_attribute_repair'] = dict(
            selection_step=None, selected_count=0, total_count=count, selected_fraction=0.,
            grouping=('one early round of single-view gradient probes' if
                      self.config.full_update.early_joint_repair else
                      'two rounds of single-view gradient probes'), training_views_at_selection=0,
            selection_only_views=0, measurement=(('logscale and SH only; ' if
                getattr(self.config.full_update, 'early_probe_attribute_only', False) else '') +
                ('per-attribute median-normalized image gradient '
                'and sqrt opacity; no explicit visibility weighting' if (self.config.full_update.probe_views_per_round == 4 and
                getattr(self.config.full_update, 'probe_opacity_weight', True)) else
                'per-attribute median-normalized image gradient')),
            diagnostics=None, invariant_checks=[])

    def _plan_validate_regional_attribute_state(self, state, *, attributes_only=False):
        parameters = self._regional_attribute_parameters()
        count = len(self.gaussians._xyz)
        if (not isinstance(state, dict) or not torch.is_tensor(state.get('eligible')) or
                state['eligible'].dtype != torch.bool or state['eligible'].shape != (count,) or
                not isinstance(state.get('parameters'), dict) or
                set(state['parameters']) != set(parameters)):
            raise ValueError('Regional attribute checkpoint lost its fixed selection')
        identity = torch.arange(count, device=self.device)
        if not torch.equal(self.motion.previous_rows, identity) or not torch.equal(self.opacity_update.rows, identity):
            raise ValueError('Regional attribute checkpoint changed the old-GS layout')
        frozen = ~state['eligible']
        for name, parameter in parameters.items():
            if attributes_only and name not in self.full_attribute_update.parameters:
                continue
            saved = state['parameters'][name]
            if (not isinstance(saved, dict) or not torch.is_tensor(saved.get('value')) or
                    saved['value'].shape != parameter.shape or
                    not bool(torch.isfinite(saved['value']).all()) or
                    not isinstance(saved.get('moments'), dict) or
                    not torch.equal(parameter.detach()[frozen], saved['value'][frozen])):
                raise ValueError(f'Regional attribute checkpoint changed frozen {name}')
            actual = self.motion.optimizer.state.get(parameter, {})
            for key, value in saved['moments'].items():
                if (not torch.is_tensor(value) or value.shape != parameter.shape or
                        not bool(torch.isfinite(value).all()) or key not in actual or
                        not torch.equal(actual[key][frozen], value[frozen])):
                    raise ValueError(f'Regional attribute checkpoint changed frozen {name} moments')

    @torch.no_grad()
    def _plan_mask_regional_attribute_gradients(self, *, attributes_only=False):
        state = self._plan_regional_attribute_state
        if state is None:
            raise ValueError('Regional attribute training lost its step-200 selection')
        frozen = ~state['eligible']
        parameters = (self.full_attribute_update.parameters if attributes_only else self._regional_attribute_parameters())
        for parameter in parameters.values():
            if parameter.grad is not None:
                mask = frozen.reshape((-1,) + (1,) * (parameter.grad.ndim - 1))
                parameter.grad.masked_fill_(mask, 0)

    @torch.no_grad()
    def _plan_record_regional_image_gradients(self, step, viewpoint):
        state = self._plan_regional_attribute_state
        if state is None:
            raise ValueError('Regional gradient diagnostics lost the selected rows')
        selected = state['eligible']
        fields = {}
        for name, parameter in self._regional_attribute_parameters().items():
            gradient = parameter.grad
            fields[name] = dict(has_gradient=gradient is not None,
                selected_norm=0. if gradient is None else float(gradient[selected].norm()),
                unselected_norm=0. if gradient is None else float(gradient[~selected].norm()))
        self.birth_control['regional_attribute_repair'].setdefault('image_gradient_checks', []).append(
            dict(step=step, camera=Path(str(viewpoint.path)).name,
                 semantics='first train view image L1+DSSIM, divided by view count; before row mask; no storage loss',
                 fields=fields))

    @torch.no_grad()
    def _plan_restore_regional_attribute_frozen_rows(self, step, *, attributes_only=False):
        state = self._plan_regional_attribute_state
        eligible = state['eligible']
        for name, parameter in self._regional_attribute_parameters().items():
            if attributes_only and name not in self.full_attribute_update.parameters:
                continue
            saved = state['parameters'][name]
            mask = eligible.reshape((-1,) + (1,) * (parameter.ndim - 1))
            # Fixed-shape selection preserves frozen rows without nonzero's CUDA synchronization.
            torch.where(mask, parameter, saved['value'], out=parameter)
            for key, value in self.motion.optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    baseline = saved['moments'].get(key)
                    if baseline is None:
                        baseline = saved['moments'][key] = torch.zeros_like(value)
                    torch.where(mask, value, baseline, out=value)
        if not attributes_only and step in (201, 250, 251, 300):
            self._plan_validate_regional_attribute_state(state)
            self.birth_control['regional_attribute_repair']['invariant_checks'].append(
                dict(step=step, frozen_count=int((~eligible).sum()), frozen_values_equal=True,
                     frozen_adam_moments_equal=True))

    def _plan_validate_regional_alpha_state(self, state):
        delta = self.opacity_update.delta
        size = len(delta)
        if (not isinstance(state, dict) or not isinstance(state.get('baseline_moments'), dict) or
                not torch.is_tensor(state.get('eligible')) or
                state['eligible'].dtype != torch.bool or state['eligible'].shape != (size,)):
            raise ValueError('Regional alpha checkpoint has an invalid eligibility mask')
        for name, shape in (('baseline_delta', delta.shape), ('baseline_alpha', (size,)),
                            ('baseline_symbols', (size,)),
                            ('max_frozen_alpha_change', ()), ('max_frozen_symbol_change', ()),
                            ('max_frozen_delta_change', ()), ('max_frozen_moment_change', ())):
            value = state.get(name)
            if (not torch.is_tensor(value) or value.shape != shape or
                    not bool(torch.isfinite(value).all())):
                raise ValueError(f'Regional alpha checkpoint has invalid {name}')
        for name, value in state['baseline_moments'].items():
            if (not isinstance(name, str) or not torch.is_tensor(value) or
                    value.shape != delta.shape or not bool(torch.isfinite(value).all())):
                raise ValueError('Regional alpha checkpoint has invalid optimizer moments')
        frozen = ~state['eligible']
        if not torch.equal(delta.detach()[frozen], state['baseline_delta'][frozen]):
            raise ValueError('Regional alpha checkpoint changed a frozen opacity parameter')

    @torch.no_grad()
    def _plan_regional_alpha_select(self, step, units):
        """Snapshot step-200 state and union eligible surviving old rows."""
        if not regional_alpha_repair_active(self):
            return
        if step not in (200, 250):
            raise ValueError('Regional alpha selection occurs at step 200 or 250')
        update = self.opacity_update
        delta, rows = update.delta, update.rows.long()
        old_count = len(self.gaussians._xyz)
        point_unit = np.asarray(units['point_unit'])
        reasons = np.asarray(units['unit_reason'])
        if (rows.shape != (old_count,) or point_unit.ndim != 1 or len(point_unit) < old_count or
                reasons.ndim != 1 or ((point_unit[:old_count] < 0) |
                 (point_unit[:old_count] >= len(reasons))).any()):
            raise ValueError('Regional alpha evidence lost the current survivor layout')
        selected = torch.as_tensor(reasons[point_unit[:old_count]] == 2,
                                   dtype=torch.bool, device=delta.device)
        state = getattr(self, '_plan_regional_alpha_state', None)
        if step == 200:
            if state is not None:
                raise ValueError('Regional alpha step-200 baseline already exists')
            physical = torch.zeros(len(delta), dtype=delta.dtype, device=delta.device)
            physical[rows] = update.alphas().detach().flatten()
            adam = self.motion.optimizer.state.get(delta, {})
            state = dict(eligible=torch.zeros(len(delta), dtype=torch.bool, device=delta.device),
                baseline_delta=delta.detach().clone(), baseline_alpha=physical,
                baseline_symbols=update.symbols().flatten().clone(),
                baseline_moments={key: value.detach().clone() for key, value in adam.items()
                    if torch.is_tensor(value) and value.shape == delta.shape},
                max_frozen_alpha_change=delta.new_zeros(()),
                max_frozen_symbol_change=delta.new_zeros(()),
                max_frozen_delta_change=delta.new_zeros(()),
                max_frozen_moment_change=delta.new_zeros(()))
            self._plan_regional_alpha_state = state
        elif state is None:
            raise ValueError('Regional alpha step-250 selection lost its step-200 baseline')
        eligibility = state['eligible']
        if eligibility.shape != (len(delta),) or eligibility.dtype != torch.bool:
            raise ValueError('Regional alpha eligibility no longer matches predecessor rows')
        newly = selected & ~eligibility[rows]
        eligibility[rows[selected]] = True
        eligible_count = int(eligibility[rows].sum())
        report = self.birth_control.setdefault('regional_alpha_repair', dict(
            mask_events=[], invariant_checks=[],
            mapping='triggered D5 units -> current old GS rows -> opacity_update.rows predecessor parameter rows',
            baseline='step200 after opacity Adam, proximal update and old-GS cleanup',
            freeze='ineligible predecessor delta and all row-shaped shared-Adam moments restored after Adam and proximal; scalar Adam step remains shared',
            full_update_exempt=True))
        report['mask_events'].append(dict(step=step, old_count=old_count,
            triggered_old_count=int(selected.sum()), newly_eligible_old_count=int(newly.sum()),
            eligible_old_count=eligible_count,
            eligible_old_fraction=eligible_count / old_count if old_count else 0.,
            source='real RGB L1 training-view regional units; not birth quota or fitted-candidate success'))

    @torch.no_grad()
    def _plan_regional_alpha_mask_gradient(self, step):
        if step <= 200 or not regional_alpha_repair_active(self):
            return
        state = getattr(self, '_plan_regional_alpha_state', None)
        if state is None:
            raise ValueError('Regional alpha training lost its step-200 eligibility')
        delta = self.opacity_update.delta
        if state['eligible'].shape != (len(delta),):
            raise ValueError('Regional alpha training mask no longer matches opacity rows')
        if delta.grad is not None:
            delta.grad.masked_fill_(~state['eligible'][:, None], 0.)

    @torch.no_grad()
    def _plan_regional_alpha_restore(self, step):
        if step <= 200 or not regional_alpha_repair_active(self):
            return
        state = getattr(self, '_plan_regional_alpha_state', None)
        if state is None:
            raise ValueError('Regional alpha training lost its step-200 baseline')
        update, delta = self.opacity_update, self.opacity_update.delta
        frozen = ~state['eligible']
        delta[frozen] = state['baseline_delta'][frozen]
        optimizer_state = self.motion.optimizer.state.get(delta, {})
        for name, value in optimizer_state.items():
            if torch.is_tensor(value) and value.shape == delta.shape:
                baseline = state['baseline_moments'].get(name)
                if baseline is None:
                    baseline = torch.zeros_like(value)
                    state['baseline_moments'][name] = baseline
                value[frozen] = baseline[frozen]
        rows = update.rows.long()
        frozen_current = frozen[rows]
        frozen_rows = rows[frozen_current]
        if len(frozen_rows):
            alpha = update.alphas().detach().flatten()[frozen_current]
            max_alpha = (alpha-state['baseline_alpha'][frozen_rows]).abs().max()
            max_symbol = (update.symbols().flatten()[frozen_rows]-
                          state['baseline_symbols'][frozen_rows]).abs().max()
            max_delta = (delta[frozen_rows]-state['baseline_delta'][frozen_rows]).abs().max()
            changes = [(value[frozen_rows]-state['baseline_moments'][name][frozen_rows]).abs().max()
                       for name, value in optimizer_state.items() if torch.is_tensor(value) and
                       value.shape == delta.shape]
            max_moment = torch.stack(changes).max() if changes else delta.new_zeros(())
            for name, value in (('max_frozen_alpha_change', max_alpha),
                                ('max_frozen_symbol_change', max_symbol),
                                ('max_frozen_delta_change', max_delta),
                                ('max_frozen_moment_change', max_moment)):
                state[name] = torch.maximum(state[name], value)
        if step in (201, 250, 251, 300):
            report = self.birth_control['regional_alpha_repair']
            report['invariant_checks'].append(dict(step=step,
                frozen_old_count=int(frozen_current.sum()),
                max_frozen_alpha_change=float(state['max_frozen_alpha_change']),
                max_frozen_symbol_change=float(state['max_frozen_symbol_change']),
                max_frozen_delta_change=float(state['max_frozen_delta_change']),
                max_frozen_moment_change=float(state['max_frozen_moment_change']),
                scope='cumulative after each step Adam and opacity proximal; only currently surviving frozen old rows'))

    def frame_plan_training_views(self, step, count):
        # Existing per-frame cache loads training images once; it does not render them.
        cached = self.transaction_camera_views()
        dataset = list(cached.values())
        admission = getattr(self.config.birth, 'regional_postfit_admission', False)
        if admission:
            from .regional_birth_admission import checked_training_pool
            checked_training_pool(self, cached)
            if step > 200:
                names = getattr(self, '_plan_admission_camera_names', None)
                if not names or len(names) != 4 or any(name not in cached for name in names):
                    raise ValueError('Post-fit admission needs its four saved training-camera names')
                dataset = [view for name, view in cached.items() if name not in names]
        seed = int(self.datamodule.config.training_view_seed or 0)
        if count == 1:
            if len(dataset) < 2 or not 1 <= step <= 300:
                raise ValueError('Single-view frame plan needs two training cameras and a valid step')
            # Each optimization step uses one camera; shuffle once per full
            # camera cycle, deterministically from the frame and seed.
            epoch, position = divmod(int(step) - 1, len(dataset))
            generator = torch.Generator().manual_seed(seed + 1000003*self.current_frame + epoch)
            order = torch.randperm(len(dataset), generator=generator).tolist()
            indices = [order[position]]
            if gradient_probe_step(self, step) and self.config.full_update.probe_views_per_round == 4:
                if len(dataset) < 4:
                    raise ValueError('Four-view probes need at least four training cameras')
                start = 101 if self.config.full_update.early_joint_repair else 201 if step < 251 else 251
                generator = torch.Generator().manual_seed(seed + 1000003*self.current_frame + 500000 + start)
                indices = [torch.randperm(len(dataset), generator=generator).tolist()[step - start]]
            auxiliary_count = 0
            if step in (200, 250) and (admission or
                    getattr(self.config.birth, 'regional_color_birth', False)):
                auxiliary_count = 3
            if auxiliary_count and len(dataset) <= auxiliary_count:
                raise ValueError('Regional selection needs distinct training cameras')
            # Additional cameras provide selection evidence only; they are not
            # returned to the optimizer and do not perform another backward.
            indices.extend(order[(position + offset) % len(order)]
                           for offset in range(1, auxiliary_count + 1))
        elif count == 4 and len(dataset) >= count:
            # Preserve the historical four-view sampler and checkpoint behavior.
            generator = torch.Generator().manual_seed(seed+1000003*self.current_frame+int(step))
            indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        else:
            raise ValueError('Frame plan requires one or four distinct training views per step')
        self._plan_step_views = [dataset[i] for i in indices]
        names = [Path(str(view.path)).name for view in self._plan_step_views]
        if len(set(names)) != len(indices) or 'cam00.png' in names:
            raise ValueError('Frame plan sampled duplicate or held-out training cameras')
        if (regional_attribute_repair_active(self) and
                (getattr(self.config.full_update, 'point_training_schedule', 'standard') != 'standard' or
                 self.config.full_update.audit_camera_sequence)):
            if count != 1:
                raise ValueError('Recorded point experiments require one optimization view per step')
            self.birth_control.setdefault('point_schedule_camera_trace', []).append(
                dict(step=int(step), camera=names[0]))
        return self._plan_step_views[:count]

    def _plan_record_phase_time(self):
        started = getattr(self,'_plan_phase_started',None)
        if started is None:
            return
        if torch.device(self.device).type == 'cuda':
            torch.cuda.synchronize(self.device)
        phase = self._plan_timed_phase
        times = self.birth_control.setdefault('phase_seconds',{})
        times[phase] = times.get(phase,0.)+time.perf_counter()-started
        self._plan_phase_started = None

    def _plan_prepare_rates(self, step):
        tail = final_joint_scheduled(self, step)
        full = bool(getattr(self, 'full_update_active', False))
        regional = regional_attribute_repair_active(self)
        alpha_frozen = _fixed_old_alpha(self, step)
        phase = frame_plan_phase(self, step)
        if getattr(self,'_plan_timed_phase',None) != phase:
            self._plan_record_phase_time()
        if getattr(self,'_plan_phase_started',None) is None:
            self._plan_timed_phase = phase
            self._plan_phase_started = time.perf_counter()
        if not self.motion_committed or tail or full:
            self.motion.prepare_plan_step(step)
        update = self.opacity_update
        update.delta.requires_grad_((step >= 101 or (full and not regional)) and not alpha_frozen)
        if (step <= 100 and (not full or regional)) or alpha_frozen:
            update.delta.grad = None
        self._early_prepare_rates_legacy(step)
        if isinstance(self.births, FramePlanBirthModel):
            self.births.set_rates(step, self.config.gaussian_stage2, self.config.birth.lr_final_ratio)
        hook = getattr(self, 'full_update_prepare_step', None)
        if hook is not None:
            hook(step)
        if step > 200 and regional_attribute_repair_active(self):
            self.motion.optimizer.param_groups[2]['lr'] = (self.config.rigid_tree.gs_translation_lr *
                self.motion._motion_lr_factor(200) * self.config.full_update.gradient_position_lr_scale)

    @torch.no_grad()
    def _plan_post_step(self, packages, step):
        if step != self.early_shared_steps+1 or not 1 <= step <= 300:
            raise ValueError('Frame plan requires one shared Adam update per iteration')
        if gradient_probe_step(self, step):
            self._plan_gradient_probe(step, packages[0])
            self.early_shared_steps += 1
            return
        if regional_alpha_repair_active(self) and step > 200:
            self._plan_regional_alpha_mask_gradient(step)
        early_repair = (regional_attribute_repair_active(self) and
                        getattr(self.config.full_update, 'early_joint_repair', False))
        if regional_attribute_repair_active(self) and (step > 200 or (early_repair and step > 100)):
            self._plan_mask_regional_attribute_gradients(attributes_only=step <= 200)
        tail_start = 301-effective_final_joint_refine_steps(self)
        full = bool(getattr(self, 'full_update_active', False))
        shared = getattr(self, 'feature_dc_update', None)
        frozen = _fixed_incumbent(self, step)
        alpha_frozen = _fixed_old_alpha(self, step)
        boundaries = (1,50,51,100,101,150,151,180,181,200,201,250,251,300)
        diagnostic_steps = step in boundaries or (tail_start <= 300 and step in (tail_start-1,tail_start))
        if diagnostic_steps and getattr(getattr(self.config, 'full_update', None), 'gradient_diagnostics', True):
            parameters = {name: getattr(self.motion,name) for name in
                          ('gate_logits','region_translation','gs_translation','region_rotation','gs_rotation')}
            parameters['old_alpha'] = self.opacity_update.delta
            full_attributes = getattr(self, 'full_attribute_update', None)
            if full_attributes is not None and full:
                parameters.update({f'old_full_{name}': parameter for name, parameter in
                                   full_attributes.parameters.items()})
            shared = getattr(self, 'feature_dc_update', None)
            if shared is not None and shared.regional and shared.ready:
                parameters['regional_dc'] = shared.delta
            if (getattr(self, 'logscale_update', None) is not None and
                    self.logscale_update.ready):
                parameters['old_logscale'] = self.logscale_update.delta
            if isinstance(self.births,FramePlanBirthModel):
                for index,batch in enumerate(self.births.cohorts):
                    parameters.update({f'birth{index}_{name}': value for name,value in batch['parameters'].items()})
            def optimizer_updates(name, value):
                return int(self.motion.optimizer.state.get(value, {}).get('step', 0))
            self.birth_control.setdefault('frame_plan_gradients',[]).append(dict(step=step,
                semantics='before this step Adam; old_full_sh_dc is per-GS DC',
                parameters={
                name: dict(trainable=value.requires_grad, has_gradient=value.grad is not None,
                           gradient_norm=float(value.grad.norm()) if value.grad is not None else 0.,
                           adam_updates=optimizer_updates(name, value))
                for name,value in parameters.items()}))
        count = len(self.gaussians._xyz)
        for package in packages:
            observed = package['motion_contribution_filter'][:count]
            self.motion.mark_observed(observed)
            self.birth_control['motion_frustum_samples'] = torch.as_tensor(self.birth_control.get('motion_frustum_samples',0),device=self.device)+package['visibility_filter'][:count].sum()
            self.birth_control['motion_observed_samples'] = torch.as_tensor(self.birth_control.get('motion_observed_samples',0),device=self.device)+observed.sum()
        if fitting_old_motion(self):
            edits = self.motion.optimizer_step()
            self.birth_control['tree_edit_count'] = self.birth_control.get('tree_edit_count', 0)+len(edits)
        else:
            logscale_update = getattr(self, 'logscale_update', None)
            if logscale_update is not None and logscale_update.ready and not frozen:
                logscale_update.before_optimizer_step()
            self.motion.optimizer.step()
            if logscale_update is not None and logscale_update.ready and not frozen:
                logscale_update.after_optimizer_step(
                    step, self.motion.optimizer, self._rate_byte_price(step))
            self.motion.optimizer.zero_grad(set_to_none=True)
        hook = getattr(self, 'full_update_after_step', None)
        if hook is not None:
            hook(step)
        if (step >= 101 or (full and not regional_attribute_repair_active(self))) and not alpha_frozen:
            self._early_sync_opacity_state()
            window = self.config.rigid_tree.motion_fit_window_steps
            # The old-alpha carrier keeps learning after old motion commits.
            # Its rate terms detach in proximal mode, so skipping these birth-
            # phase windows silently disables its nonzero-update byte price.
            opacity_proximal = (self.motion._motion_fit_window_due() if step <= 200 else
                window is None or (step-200) % window == 0 or step == 300)
            opacity_price = (0. if getattr(self.config.birth, 'population_rate_exempt', False)
                             else self._rate_byte_price(step))
            self.opacity_update.optimizer_step(opacity_price,
                proximal=opacity_proximal,
                lr_factor=self.motion._motion_lr_factor(min(step, 200))
                    if self.config.opacity_update.decay_with_motion else 1.,
                perform_step=False, external_state=self.motion.optimizer.state.get(self.opacity_update.delta, {}))
        if regional_alpha_repair_active(self) and step > 200:
            self._plan_regional_alpha_restore(step)
        if regional_attribute_repair_active(self) and (step > 200 or (early_repair and step > 100)):
            self._plan_restore_regional_attribute_frozen_rows(step, attributes_only=step <= 200)
        self.early_shared_steps += 1
        if self.early_initialized:
            self.births.project_parameters()
            self.early_birth_updates += 1
        if getattr(self.config.birth, 'regional_color_birth', False) and step in (100, 150, 180):
            self._plan_record_stage(step, 'old_only', old_only=True)
        if early_repair and step == 100:
            self._plan_select_regional_attributes(packages)
        if step == 200:
            xyz, rotation = self.motion.decoded_survivor_pose()
            self.gaussians.apply_selected_motion(xyz.to(self.device), rotation.to(self.device))
            if not full:
                for parameter in self.motion.parameters():
                    parameter.requires_grad_(False)
                    parameter.grad = None
            self.motion_committed = True
            self.births = FramePlanBirthModel(self.config.sh_degree, self.device,
                logit_opacity=getattr(self.config.birth,'residual_support_init',False))
            self.births.spatial_lr_scale = self.gaussians.spatial_lr_scale
            self.births.active_sh_degree = self.gaussians.active_sh_degree
            self.early_initialized = True
            self._plan_used_quota = 0
            self._plan_events = []
            self.birth_control['early_replacement'] = dict(mode='frame_training_plan', source_count=0,
                birth_steps=[200, 250], cleanup_steps=[200, 250, 300], population_screening=False,
                colour_initialization='responsibility-weighted training RGB to SH DC; higher SH inherited from codec reference',
                reference_distance='0.5 squared position/extent + 0.5 mean squared RGB DC; ties lowest old row',
                storage_loss=dict(units='surrogate bytes; not compressed bytes',
                                  enabled=bool(self.config.birth.storage_loss_enabled)))
            if getattr(self.config, 'ablation_motion_only', False):
                self.birth_control['early_replacement'].update(mode='motion_only_ablation',
                    birth_steps=[], cleanup_steps=[], colour_initialization=None,
                    reference_distance=None)
                self._plan_finish()
                return
            if regional_attribute_repair_active(self):
                self.birth_control['early_replacement'].update(mode='regional_attribute_repair',
                    birth_steps=[], cleanup_steps=[], colour_initialization=None,
                    reference_distance=None)
                if early_repair:
                    if shared is not None and shared.regional and not shared.ready:
                        shared.bind(self.motion.owner, len(self.motion.parents),
                                    torch.unique(self.motion.owner), self.motion.optimizer)
                    # Motion and opacity remain trainable for every point until
                    # step 200; freeze them relative to this committed state.
                    state = self._plan_regional_attribute_state
                    for name, parameter in self._regional_attribute_parameters().items():
                        if name in self.full_attribute_update.parameters:
                            continue
                        adam = self.motion.optimizer.state.get(parameter, {})
                        state['parameters'][name] = dict(value=parameter.detach().clone(), moments={
                            key: value.detach().clone() for key, value in adam.items()
                            if torch.is_tensor(value) and value.shape == parameter.shape})
                    self._plan_validate_regional_attribute_state(state)
                else:
                    self._plan_select_regional_attributes(packages)
        if regional_attribute_repair_active(self):
            if step in (200, 250, 300):
                self._plan_record_stage(step, f'regional_repair_{step}', old_only=True)
            if step == 300 and shared is not None and shared.regional and shared.ready:
                # A read-only final-image intervention, not a retrained ablation.
                self._diagnostic_regional_dc_disabled = True
                try:
                    self._plan_record_stage(step, 'regional_dc_off', old_only=True)
                finally:
                    self._diagnostic_regional_dc_disabled = False
            if step == 300 or (step == 200 and self.config.full_update.stop_after_motion):
                self.full_update_finish_pose_attributes()
                self._plan_finish()
            return
        pre_tail_pose = None
        if step == 300 and full:
            self.full_update_finish_pose_attributes()
        if step == 300 and final_joint_scheduled(self, step):
            if getattr(getattr(self.trainer, 'config', None), 'stage_image_camera', None) is not None:
                # The base pose remains at step200 while the optional tail
                # learns its motion. Capture it before materializing step300.
                pre_tail_pose = (self.gaussians._xyz.detach().clone(),
                                 self.gaussians._rotation.detach().clone())
            # Canonical old attributes must reflect the final quantized pose
            # before deletion/reanchoring and before the codec checks it.
            xyz, rotation = self.motion.decoded_survivor_pose()
            self.gaussians.apply_selected_motion(xyz.to(self.device), rotation.to(self.device))
        if step in (200, 250, 300):
            self._plan_record_stage(step, 'pre_cleanup')
            if (step < 300 and
                    getattr(self.config.birth, 'regional_color_birth', False)):
                self._plan_record_stage(step, 'pre_cleanup_old_only', old_only=True)
            if step == 300:
                self._plan_record_stage(step, 'pre_cleanup_old_only', old_only=True)
                if pre_tail_pose is not None:
                    self._plan_record_stage(step, 'pre_cleanup_old_pre_tail_pose',
                                            old_only=True, old_pose=pre_tail_pose)
            self._plan_cleanup(step)
            self._plan_record_stage(step, 'post_cleanup')
            if (step == 200 and getattr(self, 'logscale_update', None) is not None and
                    not getattr(self.config.birth, 'keep_incumbent_fixed_after_motion', False)):
                _, _, point_translation, _ = self.motion._values(hard=True)
                candidates = point_translation.ne(0).any(-1).nonzero(as_tuple=True)[0]
                self.logscale_update.setup(len(self.gaussians._xyz), candidates,
                    self.motion.optimizer, start_step=200)
            if step < 300:
                self._plan_record_stage(step, 'pre_birth')
                if step == 200 and getattr(self.config.birth, 'regional_postfit_admission', False):
                    from .regional_birth_admission import reserve_admission_cameras
                    reserve_admission_cameras(self)
                self._plan_birth(step)
                self._plan_record_stage(step, 'post_birth')
            else:
                if getattr(self.config.birth, 'regional_postfit_admission', False):
                    from .regional_birth_admission import admit_fitted_births
                    self._plan_record_stage(step, 'pre_admission')
                    admit_fitted_births(self)
                    self._plan_record_stage(step, 'post_admission')
                recorder = getattr(self.trainer, 'record_birth_comparison', None)
                if recorder is not None:
                    recorder()
                self._plan_finish()
        tail_steps = effective_final_joint_refine_steps(self)
        if tail_steps and step == 300-tail_steps:
            # The tail may start immediately after the second birth at step250.
            self._plan_record_stage(step, 'pre_tail')
            self._plan_record_stage(step, 'pre_tail_old_only', old_only=True)

    def _plan_record_stage(self, step, stage, *, old_only=False, old_pose=None):
        recorder = getattr(self.trainer, 'record_frame_plan_stage', None)
        if recorder is not None:
            recorder(step, stage, old_only=old_only, old_pose=old_pose)

    @torch.no_grad()
    def _plan_cleanup(self, step):
        # Snapshot quantized children before changing any references/old rows.
        reference = self._adaptive_reference()
        reference['opacity'] = self.opacity_update.canonical_logits().to(self.device)
        view = self.make_birth_view(reference=reference)
        newborn = ({name: view.raw_value(name).detach().clone() for name in RAW_FIELDS}
                   if view is not None else None)
        regional_alpha_check = bool(getattr(self.config.birth, 'regional_color_birth', False))
        initial_new_count = len(newborn['xyz']) if newborn is not None else 0
        post_reanchor_pruned = 0
        if newborn is not None:
            # Match the rotation materialized by the attribute/pose codecs.
            newborn['rotation'] = view.get_rotation.detach().clone()
            alive = newborn['opacity'].sigmoid().flatten() >= .01
            selected = alive.nonzero(as_tuple=True)[0]
            self.births.select_rows_inplace(selected, self.motion.optimizer)
            newborn = {k: v[selected] for k, v in newborn.items()}
            self.births.set_actual(newborn)
            new_removed = int((~alive).sum())
        else:
            new_removed = 0
        keep = (reference['opacity'].sigmoid().flatten() >= .01).nonzero(as_tuple=True)[0]
        if not len(keep):
            raise ValueError('Frame plan: all old GS have alpha < .01; no decodable birth anchor remains')
        remove = torch.ones(len(reference['xyz']), dtype=torch.bool, device=self.device)
        remove[keep] = False
        if remove.any():
            # Keep deleted static attributes indexed in the original previous
            # frame layout, allowing a post-deletion checkpoint to bootstrap.
            deleted_ids = self.motion.previous_rows[remove].cpu()
            attributes = {name: getattr(self.gaussians, attr).detach()[remove].cpu().clone()
                          for name, attr in zip(RAW_FIELDS[2:], RAW_ATTRIBUTES[2:])}
            if self.retired_reference is None:
                self.retired_reference = dict(rows=deleted_ids, attributes=attributes)
            else:
                self.retired_reference['rows'] = torch.cat((self.retired_reference['rows'], deleted_ids))
                for name in attributes:
                    self.retired_reference['attributes'][name] = torch.cat((
                        self.retired_reference['attributes'][name], attributes[name]))
            self.gaussians.prune_points(remove)
            self.motion.retain_survivors(keep)
            logscale_update = getattr(self, 'logscale_update', None)
            if logscale_update is not None and logscale_update.ready:
                logscale_update.retain_survivors(keep)
        update = self.opacity_update
        update.commit(update.rows[keep])  # Keep original delta parameter/moments intact.
        fixed = _fixed_incumbent(self, step)
        alpha_fixed = _fixed_old_alpha(self, step)
        update.delta.requires_grad_(step < 300 and not alpha_fixed)
        if alpha_fixed:
            update.delta.grad = None
        self.gaussians._opacity.copy_(update.canonical_logits().to(self.device))
        reference = {name: value[keep] for name, value in reference.items()}
        reanchor = {}
        reanchor_tolerance = {}
        new_count_before_reanchor = len(newborn['xyz']) if newborn is not None else 0
        if newborn is not None and len(newborn['xyz']):
            self.births.source_rows = nearest_old_references(newborn, reference, self.datamodule.cameras_extent)
            changed_view = self.make_birth_view(reference=reference)
            if regional_alpha_check:
                # Re-quantizing against a different source can move an alpha
                # that survived the first check just below the threshold.
                alpha = changed_view.get_opacity.detach().flatten()
                if not bool(torch.isfinite(alpha).all()):
                    raise ValueError('Frame plan: reanchored newborn alpha is nonfinite')
                below = alpha < .01
                post_reanchor_pruned = int(below.sum())
                if post_reanchor_pruned:
                    selected = (~below).nonzero(as_tuple=True)[0]
                    self.births.select_rows_inplace(selected, self.motion.optimizer)
                    newborn = {name: value[selected] for name, value in newborn.items()}
                    changed_view = self.make_birth_view(reference=reference)
                    new_removed += post_reanchor_pruned
            if changed_view is not None:
                for name in RAW_FIELDS:
                    error = (changed_view.raw_value(name).double()-newborn[name].double()).abs()
                    reanchor[name] = float(error.max()) if error.numel() else 0.
                    # Quaternion sign changes are handled by the quantizer.
                    if name != 'rotation':
                        tolerance = reanchor_reconstruction_tolerance(
                            newborn[name], changed_view.predictor[name], changed_view.steps[name])
                        reanchor_tolerance[name] = float(tolerance.max()) if tolerance.numel() else 0.
                        if bool((~torch.isfinite(error) | (error > tolerance)).any()):
                            raise ValueError(f'Reanchored birth {name} exceeded quantization and '
                                f'floating-point reconstruction bound: max error={reanchor[name]:.9g}, '
                                f'max bound={reanchor_tolerance[name]:.9g}')
        event = dict(step=step, removed_old=int(remove.sum()), removed_new=new_removed,
                     old_count=len(keep), new_count=len(self.births._xyz), reanchor_max_error=reanchor,
                     reanchor_max_tolerance=reanchor_tolerance,
                     population_confirmation_deferred=False,
                     evidence_time='after optimizer update and cleanup; before birth')
        if regional_alpha_check:
            event.update(new_count_before_cleanup=initial_new_count,
                new_count_before_reanchor=new_count_before_reanchor,
                post_reanchor_low_alpha_pruned=post_reanchor_pruned,
                new_count_after_reanchor=len(self.births._xyz),
                post_reanchor_alpha_check_deferred=False)
        self._plan_events.append(event)
        self.birth_control['frame_plan_events'] = self._plan_events

    @torch.no_grad()
    def _plan_birth(self, step):
        reference = self._adaptive_reference()
        reference['opacity'] = self.opacity_update.canonical_logits().to(self.device)
        view = self.make_birth_view(reference=reference)
        raw = {name: value.clone() for name, value in reference.items()}
        if view is not None:
            for name in RAW_FIELDS:
                raw[name] = torch.cat((raw[name], view.raw_value(name).detach()))
        old = RawView(raw, self.config.sh_degree, self.gaussians)
        proposal = SimpleNamespace(config=self.config, gaussians=old, renderer=self.renderer, background=self.background)
        views = {Path(str(v.path)).name: v for v in self._plan_step_views}
        support_init=getattr(self.config.birth,'residual_support_init',False)
        regional = bool(getattr(self.config.birth, 'regional_color_birth', False))
        evidence_args=dict(residual_floor=(0. if regional else self.config.birth.residual.residual_floor),
                           collect_colour_evidence=True)
        if support_init:evidence_args['collect_support_moments']=True
        scores, observed, _, report, cache = _native_responsibility(proposal, views,**evidence_args)
        eligible = ((scores > 0) & observed.ne(0)).nonzero(as_tuple=True)[0]
        limit = min(self.config.birth.max_new_gaussians, self.config.birth.residual.max_candidates)
        cap = min(limit-self._plan_used_quota,
                  max(0, self.config.birth.residual.max_gaussians-len(raw['xyz'])))
        if step == 200:
            cap = min(cap, limit//2)
        region_context = None
        if regional:
            from .regional_topology import (measurement_layout, assign_measurement_units,
                native_colour_evidence, select_region_candidates)
            from .model.spatial_gate import spatial_state_route
            cfg = self.config.birth
            layout, layout_label = measurement_layout(self)
            numerator, denominator, uncovered = native_colour_evidence(proposal, cache)
            paths = spatial_state_route(raw['xyz'], layout, cfg.regional_color_depth).numpy()
            units = assign_measurement_units(paths, numerator, denominator,
                cfg.regional_color_l1_threshold, cfg.regional_color_min_support,
                cfg.regional_color_min_views)
            if regional_alpha_repair_active(self):
                self._plan_regional_alpha_select(step, units)
            # A candidate needs genuine positive residual responsibility in at
            # least two training views, not only a frustum/bounding-box hit.
            support_views = torch.zeros_like(observed)
            for index, item in enumerate(cache):
                camera = item['view']
                matrix = camera.world_view_transform.to(raw['xyz'])
                depth = (raw['xyz'] @ matrix[:3, :3] + matrix[3, :3])[:, 2]
                positive_l1 = torch.as_tensor(numerator[index] > 0, device=self.device)
                valid = (item['observed'] & positive_l1 & depth.gt(.2))
                support_views += valid.long()
            selected = select_region_candidates(units, numerator.sum(axis=0),
                support_views.detach().cpu().numpy(), max(0, cap))
            rows = torch.as_tensor(selected, dtype=torch.long, device=self.device)
            region_context = dict(layout=layout, layout_label=layout_label, units=units,
                numerator=numerator, uncovered=uncovered, chosen_rows=selected.copy(),
                pre_geometry_count=len(selected), geometry_rejected_count=0,
                geometry_rejected_l1=0., boundary_rejected_count=0,
                boundary_rejected_l1=0., colour_rejected_count=0,
                colour_rejected_l1=0.)
        else:
            # Legacy behavior remains unchanged when regional selection is off.
            rows = eligible[scores[eligible].argsort(descending=True, stable=True)[:max(0, cap)]]
        seed = {name: value[rows].clone() for name, value in raw.items()}
        support_report={}
        if support_init:
            geometry,valid_geometry,support_report=residual_support_seed(raw,rows,cache,
                self.config.birth.new_xyz_step_extent*self.datamodule.cameras_extent)
            seed.update(geometry)
            if regional:
                seed['opacity'] = torch.full_like(seed['opacity'], math.log(.1 / .9))
            else:
                seed['opacity']=residual_support_opacity(seed['opacity'],rows,cache)
            support_report.update(alpha_semantics=('fixed trainable initial alpha 0.1; rendered value follows birth quantization' if regional else
                'minimum of parent alpha and native support-weighted residual RGB RMS prior; not measured opacity'),
                initial_alpha_mean=float(seed['opacity'].sigmoid()[valid_geometry].mean()) if bool(valid_geometry.any()) else None)
            if regional:
                rejected = rows[~valid_geometry].detach().cpu().numpy()
                region_context['geometry_rejected_count'] = len(rejected)
                region_context['geometry_rejected_l1'] = float(numerator[:, rejected].sum())
            rows=rows[valid_geometry]
            seed={name:value[valid_geometry] for name,value in seed.items()}
        else:
            index = torch.arange(len(rows), device=self.device, dtype=raw['xyz'].dtype)+self._plan_used_quota
            direction = torch.stack((torch.sin(index*2.399), torch.cos(index*2.399), torch.sin(index*1.37)), 1)
            seed['xyz'] += _rotate(seed['rotation'], direction*seed['logscale'].exp()*self.config.birth.clone_sigma_scale)
        # Legacy alpha is copied; relocated support starts with a conservative
        # image-evidence amplitude prior, then learns through the same alpha carrier.
        rgb = seed['xyz'].new_zeros(len(rows), 3)
        mass = seed['xyz'].new_zeros(len(rows))
        for item in cache:
            camera = item['view']
            matrix = camera.world_view_transform.to(seed['xyz'])
            projected = seed['xyz']@matrix[:3, :3]+matrix[3, :3]
            z = projected[:, 2]
            grid = torch.stack((projected[:, 0]/z.clamp_min(1e-12)/math.tan(float(camera.fovx)*.5),
                                projected[:, 1]/z.clamp_min(1e-12)/math.tan(float(camera.fovy)*.5)), -1)
            valid = (z > 0) & (grid.abs() < 1).all(-1) & item['observed'][rows]
            weight = item['native_mass'][rows]*valid
            rgb += _sample(item['image'], grid)*weight[:, None]
            mass += weight
        supported = mass > 0
        if regional:
            rejected = rows[~supported].detach().cpu().numpy()
            region_context['colour_rejected_count'] = len(rejected)
            region_context['colour_rejected_l1'] = float(numerator[:, rejected].sum())
        rows = rows[supported]
        seed = {name: value[supported] for name, value in seed.items()}
        rgb = (rgb[supported]/mass[supported, None]).clamp(0, 1)
        seed['sh_dc'] = ((rgb-.5)/.28209479177387814)[:, None]
        refs = nearest_old_references(seed, reference, self.datamodule.cameras_extent)
        if regional:
            from .regional_topology import route_seed_units
            from .model.birth_quantization import quantized_symbols
            point_units = region_context['units']['point_unit']
            source_units = point_units[rows.detach().cpu().numpy()]
            # Enforce the partition on the codec-grid position used for actual
            # rendering, not only on the unquantized proposal centre.
            xyz_step = self.config.birth.new_xyz_step_extent * self.datamodule.cameras_extent
            predictor = reference['xyz'][refs]
            quantized_xyz = predictor + quantized_symbols(seed['xyz'] - predictor, xyz_step) * xyz_step
            seed_leaf, seed_units = route_seed_units(region_context['layout'], quantized_xyz,
                                                      region_context['units'])
            within = seed_units == source_units
            rejected = rows.detach().cpu().numpy()[~within]
            region_context['boundary_rejected_count'] = len(rejected)
            region_context['boundary_rejected_l1'] = float(numerator[:, rejected].sum())
            keep = torch.as_tensor(within, device=self.device, dtype=torch.bool)
            rows, refs = rows[keep], refs[keep]
            seed = {name: value[keep] for name, value in seed.items()}
            region_context['seed_leaf'] = seed_leaf[within]
            region_context['seed_unit'] = seed_units[within]
        # A newborn is a small geometric repair near an existing surface.  Its
        # source is already the packet predictor, so inherit the source's
        # view-dependent coefficients exactly.  Zeroing them used to encode a
        # cancellation residual for every SH coefficient (especially costly
        # for a shared SH3 reference) and discarded useful angular appearance
        # before the child had received a single image update.
        seed['sh_rest'] = reference['sh_rest'][refs].clone()
        self.births.add(seed, rows, refs, step, self.motion.optimizer, self.config.gaussian_stage2)
        self._plan_used_quota += len(rows)
        self.birth_control['early_replacement']['source_count'] = self._plan_used_quota
        self.birth_control['proposal_count'] = self._plan_used_quota
        self._plan_events[-1].update(born=len(rows), used_quota=self._plan_used_quota,
            proposal_cameras=list(views), proposal=report, image_colour_supported=len(rows),
            error_before_birth=report['raw_old_RGB_SSE'],
            birth_jump_error_semantics='unfiltered RGB SSE before and after on identical proposal training cameras')
        if support_init:self._plan_events[-1]['support_initialization']=support_report
        after = dict(reference)
        quantized = self.make_birth_view(reference=reference)
        if quantized is not None:
            after = {name: torch.cat((after[name], quantized.raw_value(name).detach())) for name in RAW_FIELDS}
            if regional and len(rows):
                alpha = quantized.get_opacity[-len(rows):].detach().flatten()
                self._plan_events[-1]['regional_initial_quantized_alpha'] = dict(
                    minimum=float(alpha.min()), maximum=float(alpha.max()), mean=float(alpha.mean()),
                    trainable_initial_alpha=.1,
                    semantics='the trainable seed is 0.1; codec-grid rendering may differ')
        previous = self.renderer.gaussians
        collection = self.renderer.collect_contributions
        try:
            self.renderer.gaussians = RawView(after, self.config.sh_degree, self.gaussians)
            self.renderer.collect_contributions = False
            error = sum(float((self.renderer(v, self.background)['render']-v.image).square().sum())
                        for v in views.values())
        finally:
            self.renderer.gaussians, self.renderer.collect_contributions = previous, collection
        self._plan_events[-1]['error_after_birth'] = error
        if regional:
            from .regional_topology import save_region_diagnostics
            cfg = self.config.birth
            units = region_context['units']
            metadata = dict(camera_names=[item['name'] for item in cache],
                threshold=cfg.regional_color_l1_threshold,
                min_support=cfg.regional_color_min_support, min_views=cfg.regional_color_min_views)
            audit = save_region_diagnostics(self, step, units, metadata,
                candidate_rows=region_context['chosen_rows'],
                seed_rows=rows.detach().cpu().numpy(), seed_reference_rows=refs.detach().cpu().numpy(),
                seed_leaf=region_context['seed_leaf'], seed_unit=region_context['seed_unit'])
            reason = units['unit_reason']
            self._plan_events[-1]['regional_color'] = dict(
                layout=region_context['layout_label'], depth=cfg.regional_color_depth,
                cameras=metadata['camera_names'], l1_threshold=metadata['threshold'],
                min_effective_pixels_per_view=metadata['min_support'],
                min_reliable_views=metadata['min_views'], diagnostic_sidecar=audit,
                reason_codes={'0': 'insufficient evidence', '1': 'at or below threshold', '2': 'triggered'},
                measurement_units=len(reason), triggered_units=int((reason == 2).sum()),
                insufficient_units=int((reason == 0).sum()),
                parent_remainder_units=int((units['unit_depth'] < 5).sum()),
                occupied_leaf_count=len(units['leaf_id']),
                candidate_count=region_context['pre_geometry_count'],
                candidate_priority='sum of per-view native Talpha-weighted RGB L1; at least two positive supported views',
                geometry_rejected_count=region_context['geometry_rejected_count'],
                geometry_rejected_l1=region_context['geometry_rejected_l1'],
                colour_rejected_count=region_context['colour_rejected_count'],
                colour_rejected_l1=region_context['colour_rejected_l1'],
                boundary_rejected_count=region_context['boundary_rejected_count'],
                boundary_rejected_l1=region_context['boundary_rejected_l1'],
                born=len(rows), initial_trainable_alpha=.1, unassigned=region_context['uncovered'],
                seed_provenance='native Talpha RGB SSE spatial moments (no extra residual floor) with inherited source-centre depth; not triangulated',
                limitation='repairs only existing supported regions; unassigned residual lacks independent 3D localization, and covered residual may be attributed to a wrong-depth surface',
                unit_semantics='disjoint fixed D5 leaves or pooled unresolved ancestor remainder; per-view Talpha-weighted RGB L1 divided by Talpha support')

    def _plan_storage_regularization(self, step, birth_view=None):
        view = self.make_birth_view(reference=self._adaptive_reference()) if birth_view is None else birth_view
        if view is None:
            zero = self.gaussians._xyz.new_zeros(())
            return zero, dict(birth_storage_loss=zero)
        # Each cohort's own age controls warmup, never the global step250 clock.
        ramps = []
        for batch in self.births.cohorts:
            age = step-batch['birth_step']
            cfg = self.config.birth
            ramp = min(max((age-cfg.storage_warmup_steps)/max(1, cfg.storage_ramp_steps), 0.), 1.)
            ramps.append(view.source.new_full((len(batch['parameters']['_xyz']),), ramp, dtype=view.get_xyz.dtype))
        view.storage_row_weights = torch.cat(ramps)
        terms = newborn_storage_terms(view, self.config.birth.opacity_threshold)
        price = self.config.birth.storage_loss_strength*self._rate_byte_price(step)
        loss = terms['surrogate_bytes']*price
        return loss, {**{f'birth_storage_{name}': value.detach() for name, value in terms.items()},
                      'birth_storage_loss': loss.detach()}

    def _plan_population_regularization(self, step):
        # Uniform representation pressure complements survival-priced bytes.
        # The same a/(a+.01) penalty applies to old/new decoded alpha.
        old = self.opacity_update.alphas().flatten()
        view = self.make_birth_view(reference=self._adaptive_reference()) if self.early_initialized else None
        new = view.get_opacity.flatten() if view is not None else old[:0]
        threshold = self.config.birth.opacity_threshold
        old_support = (old/(old+threshold)).sum()
        new_support = (new/(new+threshold)).sum()
        penalty = (old_support+new_support)*self.config.birth.population_loss_weight/max(1,len(old)+len(new))
        return penalty, dict(population_loss=penalty.detach(), population_old_support=old_support.detach(),
                             population_new_support=new_support.detach())

    @torch.no_grad()
    def _plan_finish(self):
        self._plan_record_phase_time()
        for name in ('motion_frustum_samples','motion_observed_samples'):
            self.birth_control[name] = int(self.birth_control.get(name,0))
        self.birth_control['phase_timing_scope'] = 'synchronized wall seconds; includes topology events; excludes checkpoint writes and final encoding'
        self.motion.record_final_diagnostics()
        tail_steps = effective_final_joint_refine_steps(self)
        self.birth_control['early_replacement'].update(global_iterations=self.total_steps, motion_iterations=200+tail_steps,
            final_joint_refine_steps=tail_steps,
            configured_final_joint_refine_steps=self.config.rigid_tree.spatial_gate.final_joint_refine_steps,
            incumbent_fixed_after_motion=getattr(self.config.birth, 'keep_incumbent_fixed_after_motion', False),
            newborn_iterations=self.early_birth_updates, extra_SGD_steps=0, codec_trials200=0,
            actual_birth_cohorts=len(self.births.cohorts),
            birth_updates_by_cohort=[300-batch['birth_step'] for batch in self.births.cohorts])
        if regional_attribute_repair_active(self):
            self.birth_control['early_replacement'].update(
                motion_iterations=(200 if self.config.full_update.color_only else self.total_steps),
                newborn_iterations=0, incumbent_fixed_after_motion=self.config.full_update.color_only)
        self.finalize_frame()
        if getattr(self.config.birth, 'regional_postfit_admission', False):
            from .regional_birth_admission import verify_encoded_admission
            verify_encoded_admission(self)
        recorder = getattr(self.trainer, 'record_frame_plan_decoded', None)
        if recorder is not None:
            recorder()
