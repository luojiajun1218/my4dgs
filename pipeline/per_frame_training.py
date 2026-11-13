"""当前逐帧训练方案：200 步、固定 GS 数量、区域与逐点全属性更新。

本文件只实现这一套方案；新逐帧训练方案应新建独立 Python 文件，
并由 pipeline/module.py 选择对应的控制器，而不是堆叠旧方案分支。
"""
import time
from pathlib import Path
from typing import Any, Protocol

import torch


def effective_final_joint_refine_steps(module):
    """Expose the configured tail to validation; this plan requires it to be zero."""
    return module.config.rigid_tree.spatial_gate.final_joint_refine_steps


def regional_attribute_repair_active(module):
    return bool(getattr(module, 'full_update_active', False))


def fitting_old_motion(module):
    return module.frame_prepared and not module.frame_finalized


def frame_plan_phase(module, step):
    tree = module.config.rigid_tree
    if step <= tree.shared_discovery_steps:
        return 'shared_gate'
    if step <= 100:
        return 'joint_gate_residual_early'
    if step < tree.spatial_gate.joint_gate_start_step:
        return 'fixed_gate_residual'
    if step <= 180:
        return 'joint_gate_residual'
    return 'fixed_gate_motion'


class PerFrameContext(Protocol):
    """逐帧方案使用的模型状态与操作，不依赖 Module 的继承关系。"""

    config: Any
    datamodule: Any
    device: Any
    trainer: Any
    motion: Any
    gaussians: Any
    opacity_update: Any
    full_attribute_update: Any
    regional_full_attributes: Any
    birth_control: dict
    current_frame: int
    total_steps: int
    motion_committed: bool
    early_initialized: bool
    early_birth_updates: int
    early_shared_steps: int
    _plan_used_quota: int
    _plan_events: list

    def full_update_prepare_step(self, step: int) -> None: ...
    def full_update_after_step(self, step: int) -> None: ...
    def full_update_finish_pose_attributes(self) -> None: ...
    def _rate_byte_price(self, step: int) -> float: ...
    def transaction_camera_views(self) -> Any: ...
    def finalize_frame(self) -> None: ...


class PerFrameTraining:
    """执行当前训练方案；模型实现通过显式上下文提供。"""

    def __init__(self, context: PerFrameContext):
        self.context = context
        self.reset_frame_state()

    def reset_frame_state(self):
        self._plan_phase_started = None
        self._plan_timed_phase = None
        self._plan_camera_order = None
        self._plan_step_views = None

    def validate_per_frame_training(self, config):
        if (not config.full_update.enabled or not config.full_update.early_joint_repair or
                not config.full_update.dense_color_200 or not config.full_update.regional_full_200 or
                config.ablation_motion_only or config.rigid_tree.motion_steps != 200 or
                config.rigid_tree.spatial_gate.final_joint_refine_steps):
            raise ValueError('The selected per-frame training plan requires 200-step regional and per-GS full-attribute updates')

    def per_frame_phase(self, step):
        context = self.context
        return frame_plan_phase(context, step)

    def _early_attach_opacity(self):
        context = self.context
        if context.opacity_update is None or context.opacity_update.delta is None:return
        optimizer=context.motion.optimizer
        groups={group.get('name'):group for group in optimizer.param_groups}
        delta=context.opacity_update.delta
        if 'early_old_opacity' not in groups:
            optimizer.add_param_group(dict(params=[delta],name='early_old_opacity',
                lr=context.opacity_update.config.learning_rate,betas=(.9,.999),eps=1e-8))
        else:groups['early_old_opacity']['params']=[delta]
        saved=context.opacity_update.optimizer.state.get(delta,{})
        if saved:optimizer.state[delta]=saved

    def _early_sync_opacity_state(self):
        context = self.context
        update=context.opacity_update
        if update is None or update.delta is None or context.motion.optimizer is None:return
        state=context.motion.optimizer.state.get(update.delta)
        if state is not None:update.optimizer.state[update.delta]=state
        for group in context.motion.optimizer.param_groups:
            if group.get('name')=='early_old_opacity':update.optimizer.param_groups[0]['lr']=group['lr']

    def _plan_record_phase_time(self):
        context = self.context
        started = getattr(self,'_plan_phase_started',None)
        if started is None:
            return
        if torch.device(context.device).type == 'cuda':
            torch.cuda.synchronize(context.device)
        phase = self._plan_timed_phase
        times = context.birth_control.setdefault('phase_seconds',{})
        times[phase] = times.get(phase,0.)+time.perf_counter()-started
        self._plan_phase_started = None

    def _plan_record_stage(self, step, stage, *, old_only=False, old_pose=None):
        context = self.context
        recorder = getattr(context.trainer, 'record_frame_plan_stage', None)
        if recorder is not None:
            recorder(step, stage, old_only=old_only, old_pose=old_pose)

    def _adaptive_population_regularization(self, step):
        context = self.context
        alpha = context.opacity_update.alphas().flatten()
        threshold = context.config.birth.opacity_threshold
        support = (alpha / (alpha + threshold)).sum()
        penalty = support * context.config.birth.population_loss_weight / max(1, len(alpha))
        return penalty, dict(population_loss=penalty.detach(),
                             population_old_support=support.detach(),
                             population_new_support=support.new_zeros(()))

    def _early_prepare_rates(self, step):
        context = self.context
        if not 1 <= step <= 200:
            raise ValueError('Full-attribute training requires steps 1 through 200')
        phase = frame_plan_phase(context, step)
        if getattr(self, '_plan_timed_phase', None) != phase:
            self._plan_record_phase_time()
        if getattr(self, '_plan_phase_started', None) is None:
            self._plan_timed_phase = phase
            self._plan_phase_started = time.perf_counter()
        context.motion.prepare_plan_step(step)
        context.opacity_update.delta.requires_grad_(step >= 101)
        if step <= 100:
            context.opacity_update.delta.grad = None
        self._early_attach_opacity()
        for group in context.motion.optimizer.param_groups:
            if group.get('name') == 'early_old_opacity':
                group['lr'] = context.opacity_update.config.learning_rate * (
                    context.motion._motion_lr_factor(step) if context.config.opacity_update.decay_with_motion else 1.)
        context.full_update_prepare_step(step)

    @torch.no_grad()
    def _early_post_training_step(self, packages, step):
        context = self.context
        if step != context.early_shared_steps + 1 or not 1 <= step <= 200:
            raise ValueError('Per-frame training needs one shared Adam update per step')
        boundaries = (1, 50, 51, 100, 101, 150, 151, 180, 181, 200,
                      context.config.rigid_tree.spatial_gate.joint_gate_start_step)
        if step in boundaries and context.config.full_update.gradient_diagnostics:
            parameters = {name: getattr(context.motion, name) for name in
                          ('gate_logits', 'region_translation', 'gs_translation',
                           'region_rotation', 'gs_rotation')}
            parameters['old_alpha'] = context.opacity_update.delta
            parameters.update({f'old_full_{name}': parameter for name, parameter in
                               context.full_attribute_update.parameters.items()})
            context.birth_control.setdefault('frame_plan_gradients', []).append(dict(
                step=step, semantics='before this step Adam; old_full_sh_dc is per-GS DC',
                parameters={name: dict(trainable=value.requires_grad,
                    has_gradient=value.grad is not None,
                    gradient_norm=float(value.grad.norm()) if value.grad is not None else 0.,
                    adam_updates=int(context.motion.optimizer.state.get(value, {}).get('step', 0)))
                    for name, value in parameters.items()}))
        count = len(context.gaussians._xyz)
        for package in packages:
            observed = package['motion_contribution_filter'][:count]
            context.motion.mark_observed(observed)
            context.birth_control['motion_frustum_samples'] = torch.as_tensor(
                context.birth_control.get('motion_frustum_samples', 0), device=context.device) + package['visibility_filter'][:count].sum()
            context.birth_control['motion_observed_samples'] = torch.as_tensor(
                context.birth_control.get('motion_observed_samples', 0), device=context.device) + observed.sum()
        regional = context.regional_full_attributes
        rebase = (regional is not None and step > 100 and context.config.full_update.attribute_aware_gate
                  and not context.motion._hard_gate_frozen)
        if rebase and step in (context.config.rigid_tree.spatial_gate.joint_gate_start_step, 151, 165, 180) and context.motion.gate_logits.grad is not None:
            reachable = regional.reachable_gate_mask()
            logits, gradient = context.motion.gate_logits.detach(), context.motion.gate_logits.grad.detach()
            context.birth_control.setdefault('attribute_gate_total_gradient_checks', []).append(dict(
                step=step, reachable_gates=int(reachable.sum()),
                reachable_nonzero=int((reachable & gradient.ne(0)).sum()),
                reachable_towards_switch=int((reachable & (gradient * logits).gt(0)).sum()),
                reachable_within_one=int((reachable & logits.abs().lt(1.)).sum())))
        previous_owner = context.motion.owner.detach().clone() if rebase else None
        edits = context.motion.optimizer_step()
        context.birth_control['tree_edit_count'] = context.birth_control.get('tree_edit_count', 0) + len(edits)
        if rebase:
            regional.rebase_points(previous_owner, context.full_attribute_update, context.opacity_update)
            if step in (context.config.rigid_tree.spatial_gate.joint_gate_start_step, 151, 165, 180):
                context.birth_control.setdefault('attribute_gate_rebase_checks', []).append(dict(
                    step=step, changed_points=int(previous_owner.ne(context.motion.owner).sum())))
        context.full_update_after_step(step)
        if step >= 101:
            self._early_sync_opacity_state()
            price = (0. if getattr(context.config.birth, 'population_rate_exempt', False)
                     else context._rate_byte_price(step))
            context.opacity_update.optimizer_step(price, proximal=context.motion._motion_fit_window_due(),
                lr_factor=context.motion._motion_lr_factor(step) if context.config.opacity_update.decay_with_motion else 1.,
                perform_step=False, external_state=context.motion.optimizer.state.get(context.opacity_update.delta, {}))
        context.early_shared_steps += 1
        if context.early_initialized:
            context.early_birth_updates += 1
        if step == 200:
            xyz, rotation = context.motion.decoded_survivor_pose()
            context.gaussians.apply_selected_motion(xyz.to(context.device), rotation.to(context.device))
            context.motion_committed = True
            context.early_initialized = True
            context._plan_used_quota = 0
            context._plan_events = []
            context.birth_control['early_replacement'] = dict(
                mode='dense_color_200', source_count=0, birth_steps=[], cleanup_steps=[],
                population_screening=False, colour_initialization=None, reference_distance=None,
                storage_loss=dict(units='surrogate bytes; not compressed bytes', enabled=False))
            self._plan_record_stage(step, 'regional_repair_200', old_only=True)
            context.full_update_finish_pose_attributes()
            self._plan_finish()

    def frame_plan_training_views(self, step, count):
        context = self.context
        cached = context.transaction_camera_views()
        names = list(cached)
        if count != 1 or len(names) < 2 or not 1 <= step <= 200:
            raise ValueError('Per-frame training requires one view and a step in 1..200')
        seed = int(context.datamodule.config.training_view_seed or 0)
        epoch, position = divmod(int(step) - 1, len(names))
        order_key = (seed, context.current_frame, epoch, len(names))
        cached_order = getattr(self, '_plan_camera_order', None)
        if cached_order is None or cached_order[0] != order_key:
            generator = torch.Generator().manual_seed(seed + 1000003 * context.current_frame + epoch)
            cached_order = (order_key, torch.randperm(len(names), generator=generator).tolist())
            self._plan_camera_order = cached_order
        self._plan_step_views = [cached[names[cached_order[1][position]]]]
        camera = Path(str(self._plan_step_views[0].path)).name
        if camera == 'cam00.png':
            raise ValueError('Frame plan sampled a held-out training camera')
        if context.config.full_update.audit_camera_sequence:
            context.birth_control.setdefault('point_schedule_camera_trace', []).append(
                dict(step=int(step), camera=camera))
        return self._plan_step_views

    @torch.no_grad()
    def _plan_finish(self):
        context = self.context
        self._plan_record_phase_time()
        for name in ('motion_frustum_samples', 'motion_observed_samples'):
            context.birth_control[name] = int(context.birth_control.get(name, 0))
        context.birth_control['phase_timing_scope'] = 'synchronized wall seconds; includes topology events; excludes checkpoint writes and final encoding'
        context.motion.record_final_diagnostics()
        context.birth_control['early_replacement'].update(
            global_iterations=context.total_steps, motion_iterations=context.total_steps,
            final_joint_refine_steps=0, configured_final_joint_refine_steps=0,
            incumbent_fixed_after_motion=False, newborn_iterations=0, extra_SGD_steps=0,
            codec_trials200=0, actual_birth_cohorts=0, birth_updates_by_cohort=[])
        context.finalize_frame()
        recorder = getattr(context.trainer, 'record_frame_plan_decoded', None)
        if recorder is not None:
            recorder()
