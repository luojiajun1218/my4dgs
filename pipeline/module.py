"""Spatial gate motion, GS residuals, and quantized newborn Gaussians."""

from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
import json
import math
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.nn.functional import interpolate, l1_loss
from torchvision.utils import save_image

from libgs.metric import lpips, psnr, ssim
from libgs.model.merged_gaussian import MergedGaussianModel
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import Renderer, RendererConfig

from .model.gaussian import BirthGaussianModel, GaussianModel, select_budget_mask
from .model.recongs_birth import ReConGSBirthGaussianModel, recongs_survivor_rows
from .model.rigid_tree_birth_codec import export_birth_attributes
from .model.birth_quantization import (RAW_ATTRIBUTES, RAW_FIELDS,
    QuantizedNewGaussianView, materialize_quantized_gaussians, model_raw,
    reference_digest)
from .model.rigid_residual_tree import (RigidTreeConfig,
    quaternion_angle_error, saved_config_with_default_motion_lr)
from .model.contribution_renderer import ContributionRenderer
from .model.spatial_gated_motion import SpatialGatedMotion
from .model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from .model.multiview_birth import MultiViewBirthConfig, propose_multiview_births
from .model.birth_retirement import (birth_retirement_rows,
                                     compositing_contributions)


BIRTH_ALLOCATION = 'covariance_small_source_v20'


def materialize_motion_evidence_counts(control):
    """Transfer diagnostic counters once, without synchronizing each update."""
    keys = ('motion_frustum_samples', 'motion_observed_samples')
    values = [control.get(key, 0) for key in keys]
    tensors = [value for value in values if torch.is_tensor(value)]
    if tensors:
        counts = torch.stack([torch.as_tensor(value, device=tensors[0].device)
                              for value in values]).tolist()
        control.update(zip(keys, counts))


def restore_retired_reference(raw, state, previous_rows, retired):
    """Rebuild the prior-frame raw reference for a checkpoint after deletion."""
    count = len(state.xyz)
    rows = torch.as_tensor(previous_rows).detach().cpu().long()
    deleted = torch.as_tensor(retired['rows']).detach().cpu().long()
    all_rows = torch.cat((rows, deleted)).sort().values
    if (rows.ndim != 1 or deleted.ndim != 1 or len(rows) != len(raw['xyz']) or
            not torch.equal(all_rows, torch.arange(count))):
        raise ValueError('Retired checkpoint rows do not partition the previous reference')
    restored = dict(xyz=state.xyz, rotation=state.rotation)
    for name in RAW_FIELDS[2:]:
        value = raw[name].detach().cpu()
        removed = retired['attributes'][name].detach().cpu()
        if removed.shape != (len(deleted), *value.shape[1:]):
            raise ValueError(f'Retired checkpoint lost {name} attributes')
        full = value.new_empty((count, *value.shape[1:]))
        full[rows], full[deleted] = value, removed
        restored[name] = full
    return restored


@dataclass
class GaussianConfig:
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 15000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    percent_dense: float = 0.01


@dataclass
class BirthConfig:
    """One covariance-sampled birth pool, split/prune, and balanced retirement."""

    method: str = 'covariance'
    replacement_count_tolerance: float = 0.2
    refit_steps: int = 0
    rotation_updates: bool = False
    clone_sigma_scale: float = 0.5
    initial_alpha: float = 0.1
    position_step_sigma_cap: Optional[float] = None
    max_new_gaussians: Optional[int] = 400
    residual_proposal_views: int = 0
    residual_candidate_min_alpha: float = 0.5
    source_gradient_rescore_views: int = 0
    densification_stats_steps: int = 30
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01
    terminal_opacity_threshold: float = 0.01
    terminal_max_gaussians: Optional[int] = None
    new_max_screen_size: float = 20.0
    new_xyz_step_extent: float = 1e-5
    new_rotation_step: float = 0.001
    new_logscale_step: float = 0.001
    new_opacity_step: float = 0.01
    new_sh_step: float = 0.001
    lr_final_ratio: float = 1.0
    protect_previous_births: bool = True

    def validate(self, motion_steps):
        if self.method not in ('covariance', 'recongs'):
            raise ValueError('birth.method must be covariance or recongs')
        if self.method == 'recongs' and (
                self.refit_steps or self.residual_proposal_views or
                self.source_gradient_rescore_views or self.max_new_gaussians is not None or
                self.position_step_sigma_cap is not None or self.terminal_max_gaussians is not None or
                self.lr_final_ratio != 1. or not self.rotation_updates):
            raise ValueError('ReCon-GS birth requires uncapped native fitting with rotation, '
                             'without residual proposals, step caps or refitting')
        if self.max_new_gaussians is not None and (
                isinstance(self.max_new_gaussians, bool) or
                not isinstance(self.max_new_gaussians, int) or self.max_new_gaussians < 1):
            raise ValueError('birth.max_new_gaussians must be positive or null')
        if (isinstance(self.source_gradient_rescore_views, bool) or
                not isinstance(self.source_gradient_rescore_views, int) or
                self.source_gradient_rescore_views < 0):
            raise ValueError('birth.source_gradient_rescore_views must be a nonnegative integer')
        if (isinstance(self.residual_proposal_views, bool) or
                not isinstance(self.residual_proposal_views, int) or
                self.residual_proposal_views < 0 or
                0 < self.residual_proposal_views < 3):
            raise ValueError('birth.residual_proposal_views must be zero or at least three')
        if (isinstance(self.residual_candidate_min_alpha, bool) or
                not isinstance(self.residual_candidate_min_alpha, (int, float)) or
                not math.isfinite(self.residual_candidate_min_alpha) or
                not 0 < self.residual_candidate_min_alpha <= 1):
            raise ValueError('birth.residual_candidate_min_alpha must be in (0, 1]')
        if self.terminal_max_gaussians is not None and (
                isinstance(self.terminal_max_gaussians, bool) or
                not isinstance(self.terminal_max_gaussians, int) or
                self.terminal_max_gaussians < 1):
            raise ValueError('birth.terminal_max_gaussians must be positive or null')
        if (isinstance(self.initial_alpha, bool) or
                not isinstance(self.initial_alpha, (int, float)) or
                not math.isfinite(self.initial_alpha) or
                not 0 < self.initial_alpha < 1):
            raise ValueError('birth.initial_alpha must be finite and in (0, 1)')
        if self.position_step_sigma_cap is not None and (
                isinstance(self.position_step_sigma_cap, bool) or
                not isinstance(self.position_step_sigma_cap, (int, float)) or
                not math.isfinite(self.position_step_sigma_cap) or
                self.position_step_sigma_cap <= 0):
            raise ValueError('birth.position_step_sigma_cap must be positive or null')
        if not isinstance(self.protect_previous_births, bool):
            raise ValueError('birth.protect_previous_births must be boolean')
        if not 0 < self.lr_final_ratio <= 1:
            raise ValueError('birth.lr_final_ratio must be in (0, 1]')
        if (isinstance(self.replacement_count_tolerance, bool) or
                not isinstance(self.replacement_count_tolerance, (int, float)) or
                not math.isfinite(self.replacement_count_tolerance) or
                not 0 <= self.replacement_count_tolerance < 1):
            raise ValueError('birth.replacement_count_tolerance must be in [0, 1)')
        if (isinstance(self.refit_steps, bool) or not isinstance(self.refit_steps, int) or
                self.refit_steps < 0):
            raise ValueError('birth.refit_steps must be a nonnegative integer')
        if not isinstance(self.rotation_updates, bool):
            raise ValueError('birth.rotation_updates must be boolean')
        if (isinstance(self.densification_stats_steps, bool) or
                not isinstance(self.densification_stats_steps, int) or
                not 1 <= self.densification_stats_steps <= motion_steps):
            raise ValueError('birth.densification_stats_steps must fit the motion phase')
        if (isinstance(self.densify_interval, bool) or
                not isinstance(self.densify_interval, int) or
                self.densify_interval < 1 or
                not math.isfinite(self.densify_grad_threshold) or
                self.densify_grad_threshold <= 0):
            raise ValueError('Birth densification interval and threshold must be positive')
        for name in ('clone_sigma_scale', 'new_max_screen_size',
                     'new_xyz_step_extent', 'new_rotation_step',
                     'new_logscale_step', 'new_opacity_step', 'new_sh_step'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'birth.{name} must be positive')
        if not 0 < self.opacity_threshold < 1:
            raise ValueError('birth.opacity_threshold must be in (0, 1)')
        if not 0 < self.terminal_opacity_threshold < 1:
            raise ValueError('birth.terminal_opacity_threshold must be in (0, 1)')


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 1
    effective_motion_observations: bool = False
    motion_coarse_to_fine: bool = False
    initial_opacity_threshold: float = 0.0
    lambda_dssim: float = 0.2
    photometric_loss: str = 'l1'  # Retained for existing L1 checkpoint identity.
    num_saving_images: int = 5
    lpips_required: bool = True
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    gaussian_stage2: GaussianConfig = field(default_factory=lambda: GaussianConfig(
        position_lr_init=0.0024, position_lr_final=2.4e-5,
        feature_lr=0.075, opacity_lr=0.05, scaling_lr=0.075,
        rotation_lr=0.015))
    renderer: RendererConfig = field(default_factory=RendererConfig)
    rigid_tree: RigidTreeConfig = field(default_factory=RigidTreeConfig)
    birth: BirthConfig = field(default_factory=BirthConfig)
    opacity_update: OpacityUpdateConfig = field(default_factory=OpacityUpdateConfig)


MOTION_IMAGE_SCHEDULE = 'bootstrap_aligned_v1'


def motion_image_schedule_compatible(saved, enabled, motion_steps, bootstrap_steps):
    """Legacy resumes are safe only when their image boundaries are identical."""
    return (not enabled or saved == MOTION_IMAGE_SCHEDULE or
            (saved is None and bootstrap_steps == 2 * motion_steps // 5))


def motion_image_factor(step, motion_steps, enabled=False, fitting_motion=False,
                        bootstrap_steps=None):
    """Track broad image structure first, then fit full-resolution details."""
    if not enabled or not fitting_motion:
        return 1
    bootstrap_steps = (2 * motion_steps // 5 if bootstrap_steps is None else bootstrap_steps)
    if step <= bootstrap_steps // 2:
        return 4
    if step <= bootstrap_steps:
        return 2
    return 1


def motion_photometric_loss(image, target, lambda_dssim, factor=1):
    if factor > 1:
        size = tuple(max(1, value // factor) for value in image.shape[-2:])
        image = interpolate(image[None], size=size, mode='area')[0]
        target = interpolate(target[None], size=size, mode='area')[0]
    return ((1 - lambda_dssim) * l1_loss(image, target) +
            lambda_dssim * (1 - ssim(image, target)))


def validate_initial_opacity_threshold(threshold):
    if (isinstance(threshold, bool) or
            not isinstance(threshold, (int, float)) or
            not math.isfinite(threshold) or
            not 0 <= threshold < 1):
        raise ValueError('module.initial_opacity_threshold must be in [0, 1)')


@torch.no_grad()
def prune_native_initial_gaussians(model, threshold):
    """Prune only the loaded native frame0, before the decoder reference exists."""
    validate_initial_opacity_threshold(threshold)
    original_count = len(model._xyz)
    if original_count < 1:
        raise ValueError('Native first frame contains no Gaussians')
    removed_count = 0
    if threshold > 0:
        remove = torch.sigmoid(model._opacity.detach().flatten()) < threshold
        removed_count = int(remove.sum())
        if removed_count == original_count:
            raise ValueError('Initial opacity pruning would remove every Gaussian')
        if removed_count:
            model.prune_points(remove)
    retained_count = len(model._xyz)
    if retained_count != original_count - removed_count:
        raise ValueError('Initial opacity pruning produced an inconsistent GS count')
    return dict(source='native_frame0_checkpoint', threshold=float(threshold),
                original_count=original_count, retained_count=retained_count,
                removed_count=removed_count)


def old_gaussian_pose_activity(decoded_xyz, decoded_rotation,
                               prior_xyz, prior_rotation, region_steps, gs_steps):
    """Count old GS with decoded pose changes above half a quantization step."""
    translation_threshold = min(float(region_steps[0]), float(gs_steps[0])) / 2
    rotation_threshold = min(float(region_steps[3]), float(gs_steps[3])) / 2
    moved_xyz = (decoded_xyz - prior_xyz).norm(dim=-1)
    moved_rotation = quaternion_angle_error(decoded_rotation, prior_rotation)
    return dict(
        old_gs_with_changed_pose=int(((moved_xyz > translation_threshold) |
                                      (moved_rotation > rotation_threshold)).sum()),
        changed_pose_translation_threshold_world=translation_threshold,
        changed_pose_rotation_threshold_rad=rotation_threshold)


class Module(BaseModule):
    """Old GS learn motion; newborn GS fit quantized geometry and appearance."""

    def setup(self):
        cfg = self.config
        validate_initial_opacity_threshold(cfg.initial_opacity_threshold)
        cfg.birth.validate(cfg.rigid_tree.motion_steps)
        cfg.opacity_update.validate()
        if cfg.birth.method == 'recongs' and cfg.opacity_update.joint_with_birth:
            raise ValueError('ReCon-GS birth keeps old opacity frozen during newborn fitting')
        if cfg.birth.refit_steps:
            if (cfg.birth.refit_steps >= cfg.rigid_tree.birth_steps or
                    not cfg.rigid_tree.amplitude_refit_steps):
                raise ValueError('Birth refitting requires fixed motion support and fitting steps')
        if cfg.opacity_update.amplitude_refit_steps >= cfg.rigid_tree.motion_steps:
            raise ValueError('Opacity amplitude refitting must leave selection steps')
        if not isinstance(cfg.effective_motion_observations, bool):
            raise ValueError('effective_motion_observations must be boolean')
        if not isinstance(cfg.motion_coarse_to_fine, bool):
            raise ValueError('motion_coarse_to_fine must be boolean')
        if not 0 <= cfg.lambda_dssim <= 1:
            raise ValueError('module.lambda_dssim must be in [0, 1]')
        if cfg.photometric_loss != 'l1':
            raise ValueError('Only L1 photometric training is supported')
        bg = torch.ones if self.datamodule.config.white_background else torch.zeros
        self.register_buffer('background', bg(3, device=self.device))
        self.gaussians = GaussianModel(cfg.sh_degree)
        self.gaussians.create_from_pcd(self.datamodule.scene.point_cloud,
                                       self.datamodule.cameras_extent)
        self.gaussians.training_setup(cfg.gaussian)
        self.births = self._new_birth_model()
        renderer_type = ContributionRenderer if cfg.effective_motion_observations else Renderer
        self.renderer = renderer_type(cfg.renderer, self.gaussians)
        self.motion = SpatialGatedMotion(cfg.rigid_tree)
        self.opacity_update = QuantizedOpacityUpdate(cfg.opacity_update) if cfg.opacity_update.enabled else None
        self.frame_prepared = False
        self.frame_finalized = False
        self.motion_committed = False
        self.reference_count = None
        self.initial_reference = None
        self.frame_report = None
        self.birth_control = {}
        self.retired_reference = None
        self.stage_image_metrics = []
        self._birth_diagnostic_views = {}
        self._freeze_base()

    @property
    def current_frame(self):
        return self.datamodule.current_frame

    @property
    def motion_steps(self):
        return self.config.rigid_tree.motion_steps_for_frame(self.current_frame)

    @property
    def total_steps(self):
        return self.motion_steps + self.config.rigid_tree.birth_steps

    @property
    def birth_commit_step(self):
        return self.total_steps - self.config.birth.refit_steps

    @property
    def birth_refit_active(self):
        return self.birth_control.get('refit_active', False) and not self.frame_finalized

    @property
    def birth_opacity_active(self):
        return (self.motion_committed and not self.frame_finalized and
                getattr(self, 'opacity_update', None) is not None and
                self.config.opacity_update.joint_with_birth and
                not self.birth_control.get('replacement_committed', False))

    @property
    def birth_allocation(self):
        if self.config.birth.method == 'recongs':
            return 'recongs_native_spawn_next_frame_merge_v1'
        if self.config.birth.residual_proposal_views:
            return (f'multiview_residual_proposal_v28_'
                    f'{self.config.birth.residual_proposal_views}_views')
        return BIRTH_ALLOCATION

    def _freeze_base(self):
        for attr in RAW_ATTRIBUTES:
            getattr(self.gaussians, attr).requires_grad_(False)

    def _configure_birth_phase(self, step):
        """Derive trainability from the step so boundary checkpoints resume safely."""
        if self.births.optimizer is not None:
            for name in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling'):
                getattr(self.births, name).requires_grad_(True)
            self.births._rotation.requires_grad_(self.config.birth.rotation_updates)
            if self.config.birth.method == 'recongs':
                # Upstream restores base group settings and only schedules xyz.
                self.births.update_learning_rate(max(0, step - self.motion_steps))
                return
            for group in self.births.optimizer.param_groups:
                rates = self.config.gaussian_stage2
                base = dict(xyz=rates.position_lr_init*self.births.spatial_lr_scale,
                    f_dc=rates.feature_lr, f_rest=rates.feature_lr/20,
                    opacity=rates.opacity_lr, scaling=rates.scaling_lr, rotation=rates.rotation_lr)
                group['lr'] = base[group['name']]

    @torch.no_grad()
    def _record_birth_learning(self, step, gradients=None):
        """Sparse diagnostics only; no sampling, selection or optimizer changes."""
        count = len(self.births._xyz)
        alpha = (self.make_birth_view().get_opacity.flatten() if count else
                 self.gaussians._opacity.new_empty(0))
        gradients = gradients or {}
        values = [alpha.mean() if count else alpha.new_zeros(()),
                  alpha.median() if count else alpha.new_zeros(()),
                  alpha.lt(self.config.birth.opacity_threshold).sum().float()]
        values.extend(gradients.get(name, alpha.new_zeros(())) for name in
                      ('xyz_grad_norm', 'color_grad_norm'))
        values = torch.stack(values).tolist()
        row = dict(step=step, new_count=count, alpha_mean=values[0], alpha_median=values[1],
            below_opacity_threshold=int(values[2]), xyz_grad_norm=values[3],
            color_grad_norm=values[4],
            new_opacity_trainable=self.births._opacity.requires_grad,
            new_scale_trainable=self.births._scaling.requires_grad,
            birth_lr_scale_mean=(float(self._birth_learning_rate_scale(step).mean())
                                 if count else 0.0))
        self.birth_control.setdefault('learning_trace', []).append(row)

    def on_save_checkpoint(self, ckpt):
        materialize_motion_evidence_counts(self.birth_control)
        ckpt['birth_allocation'] = self.birth_allocation
        ckpt['motion_image_schedule'] = MOTION_IMAGE_SCHEDULE
        ckpt['rigid_tree_version'] = 1
        ckpt['rigid_tree_config'] = asdict(self.config.rigid_tree)
        ckpt['rigid_tree_module_config'] = asdict(self.config)
        ckpt['rigid_tree_initial_reference'] = self.initial_reference
        ckpt['gaussians'] = self.gaussians.capture()
        ckpt['rigid_tree_motion'] = self.motion.capture() if self.frame_prepared else None
        ckpt['rigid_tree_frame'] = dict(prepared=self.frame_prepared,
            finalized=self.frame_finalized, motion_committed=self.motion_committed,
            reference_count=self.reference_count, report=self.frame_report,
            birth_control=self.birth_control)
        ckpt['rigid_tree_frame']['stage_image_metrics'] = self.stage_image_metrics
        ckpt['rigid_tree_retired_reference'] = self.retired_reference
        opacity_update = getattr(self, 'opacity_update', None)
        ckpt['quantized_opacity_update'] = opacity_update.capture() if opacity_update is not None else None
        ckpt['births'] = (self.births.capture() if self.motion_committed and
                          not self.frame_finalized and self.births.optimizer is not None else None)
        ckpt['birth_source_rows'] = (self.births.source_rows.detach().cpu()
                                     if self.births.source_rows is not None else None)
        ckpt['birth_created_steps'] = (self.births.created_steps.detach().cpu()
                                       if self.births.created_steps is not None else None)

    def on_load_checkpoint(self, ckpt):
        self.gaussians.restore(ckpt['gaussians'], self.config.gaussian)
        self.renderer.gaussians = self.gaussians
        version = ckpt.get('rigid_tree_version')
        if version is None:
            if self.current_frame != 0 or ckpt['global_step'] != self.trainer.config.num_init_steps:
                raise ValueError('Only a completed native first-frame checkpoint can bootstrap this method')
            self.initial_reference = prune_native_initial_gaussians(
                self.gaussians, self.config.initial_opacity_threshold)
            self.reference_count = self.initial_reference['retained_count']
            self._freeze_base()
            return
        if (ckpt.get('current_frame', 0) >= 1 and
                ckpt.get('birth_allocation', 'clone_small_v1') != self.birth_allocation):
            raise ValueError('Earlier newborn allocation checkpoints are replay-only')
        if self.config.motion_coarse_to_fine and not motion_image_schedule_compatible(
                ckpt.get('motion_image_schedule'), True, self.motion_steps,
                self.config.rigid_tree.spatial_gate.start_step):
            raise ValueError('Different motion image schedules are replay-only')
        self._freeze_base()
        if (version != 1 or
                saved_config_with_default_motion_lr(ckpt['rigid_tree_config']) !=
                asdict(self.config.rigid_tree)):
            raise ValueError('Rigid-tree checkpoint needs the same method configuration')
        saved_config = ckpt.get('rigid_tree_module_config')
        if saved_config is not None and 'initial_opacity_threshold' not in saved_config:
            saved_config = {**saved_config, 'initial_opacity_threshold': 0.0}
        if isinstance(saved_config, dict) and 'effective_motion_observations' not in saved_config:
            saved_config = {**saved_config, 'effective_motion_observations': False}
        if isinstance(saved_config, dict):
            saved_config = dict(saved_config)
            saved_config.setdefault('motion_coarse_to_fine', False)
            saved_config.setdefault('photometric_loss', 'l1')
            saved_config.setdefault('opacity_update', asdict(OpacityUpdateConfig()))
            saved_config['opacity_update'] = {**asdict(OpacityUpdateConfig()), **saved_config['opacity_update']}
            # Historical color-update streams remain readable by the replay
            # decoder, but their optimizer state cannot resume this method.
            color_config = saved_config.pop('feature_dc_update', {})
            if color_config.get('enabled', False) or ckpt.get('quantized_feature_dc_update') is not None:
                raise ValueError('DC color-update checkpoints are replay-only; bootstrap from frame 0')
            opacity = saved_config.pop('old_opacity', {})
            if opacity.get('enabled', False) or ckpt.get('old_opacity_delta') is not None:
                raise ValueError('Opacity-correction checkpoints are replay-only')
        if isinstance(saved_config, dict) and 'rigid_tree' in saved_config:
            saved_config = {**saved_config, 'rigid_tree':
                saved_config_with_default_motion_lr(saved_config['rigid_tree'])}
        if isinstance(saved_config, dict) and isinstance(saved_config.get('birth'), dict):
            saved_config = {**saved_config, 'birth': {
                **asdict(BirthConfig()), **saved_config['birth']}}
        if saved_config != asdict(self.config):
            raise ValueError('Rigid-tree checkpoint needs the same module configuration')
        state = ckpt['rigid_tree_frame']
        self.frame_prepared = state['prepared']
        self.frame_finalized = state['finalized']
        self.motion_committed = state['motion_committed']
        if state.get('motion_refining', False):
            raise ValueError('Post-birth refinement checkpoints are replay-only')
        self.reference_count = state['reference_count']
        self.initial_reference = ckpt.get('rigid_tree_initial_reference')
        if self.initial_reference is None:
            self.initial_reference = dict(
                source='native_frame0_checkpoint', threshold=0.0,
                original_count=self.reference_count,
                retained_count=self.reference_count, removed_count=0)
        self.frame_report = state['report']
        self.birth_control = state['birth_control']
        self.stage_image_metrics = state.get('stage_image_metrics', [])
        self.retired_reference = ckpt.get('rigid_tree_retired_reference')
        if self.frame_prepared:
            self.motion.restore(ckpt['rigid_tree_motion'])
            if self.config.opacity_update.enabled:
                self.opacity_update.restore(ckpt.get('quantized_opacity_update'), self.device)
            if self.motion_committed:
                for parameter in self.motion.parameters():
                    parameter.requires_grad_(False)
                self.motion.optimizer.zero_grad(set_to_none=True)
        if ckpt['births'] is not None:
            self.births.restore(ckpt['births'], self.config.gaussian_stage2)
            source = ckpt['birth_source_rows']
            if source is None or len(source) != len(self.births._xyz):
                raise ValueError('Birth checkpoint lost source lineage')
            self.births.source_rows = source.to(self.device)
            created = ckpt.get('birth_created_steps')
            if created is None or len(created) != len(self.births._xyz):
                raise ValueError('Birth checkpoint lost per-point learning age')
            self.births.created_steps = created.to(self.device)
        if getattr(self, 'birth_refit_active', False):
            self._set_birth_refit_trainable()
        if self.motion_committed and not self.frame_finalized:
            self._configure_birth_phase(max(self.trainer.global_step, self.motion_steps + 1))

    def prepare_frame(self):
        if self.frame_prepared:
            return
        if self.current_frame < 1:
            raise ValueError('Rigid motion starts after the shared first frame')
        self.retired_reference = None
        retirement = dict(previous_count=len(self.gaussians._xyz), retired_count=0)
        previous_birth_sources = self.motion.pending_birth_sources
        previous_birth_count = 0 if previous_birth_sources is None else len(previous_birth_sources)
        opacity_reference = (self.gaussians._opacity.detach().clone()
                             if self.config.birth.method == 'recongs' else None)
        merge_rows = None
        if self.config.birth.method == 'recongs':
            merge_rows = recongs_survivor_rows(self.gaussians.get_opacity,
                                              previous_birth_count)
            remove = torch.ones(len(self.gaussians._xyz), dtype=torch.bool, device=self.device)
            remove[merge_rows] = False
            if previous_birth_count:
                self.retired_reference = dict(rows=remove.nonzero(as_tuple=True)[0].cpu(),
                    attributes={name: getattr(self.gaussians, attr)[remove].detach().cpu().clone()
                                for name, attr in zip(RAW_FIELDS[2:], RAW_ATTRIBUTES[2:])})
                self.gaussians.prune_points(remove)
            retirement.update(retired_count=previous_birth_count,
                timing='before_motion', policy='recongs_merge_lowest_alpha',
                natural_retirements=int(opacity_reference.sigmoid().lt(
                    self.config.birth.opacity_threshold).sum()))
            if len(self.gaussians._xyz) != self.reference_count:
                raise ValueError('ReCon-GS merge lost the fixed base capacity')
        xyz, rotation = self.gaussians._xyz.detach(), self.gaussians._rotation.detach()
        previous_rows = (merge_rows if merge_rows is not None else
                         torch.arange(len(xyz), device=xyz.device)
                         if self.current_frame > 1 else None)
        self.motion.setup(xyz, rotation, self.datamodule.cameras_extent,
                          frame_index=self.current_frame, previous_rows=previous_rows)
        self.frame_prepared = True
        self.frame_finalized = False
        self.motion_committed = False
        self.frame_report = None
        self.births = self._new_birth_model()
        self.birth_control = dict(clone_added=0, split_added=0,
                                  proposal_added=0,
                                  split_children=0, split_parents_replaced=0,
                                  pruned=0,
                                  source_tracked=True, retirement=retirement,
                                  previous_birth_count=previous_birth_count)
        self.stage_image_metrics = []
        self._birth_diagnostic_views = {}
        self._freeze_base()
        if self.opacity_update is not None:
            self.opacity_update.setup(opacity_reference if merge_rows is not None
                                      else self.gaussians._opacity)
            if merge_rows is not None:
                # Retain the full preceding reference for exact checkpoint bootstrap.
                self.opacity_update.rows = merge_rows.clone()

    def pre_training_step(self):
        self.prepare_frame()
        if not self.motion_committed and self.opacity_update is not None:
            self.opacity_update.prepare_step(self.trainer.global_step, self.motion_steps)
        elif self.motion_committed and not self.frame_finalized:
            self._configure_birth_phase(self.trainer.global_step)

    def begin_next_frame(self):
        if not self.frame_finalized:
            raise ValueError('Cannot advance before quantized finalization')
        self.frame_prepared = False
        self.frame_finalized = False
        self.motion_committed = False
        self.frame_report = None


    def _new_birth_model(self):
        model = (ReConGSBirthGaussianModel if self.config.birth.method == 'recongs'
                 else BirthGaussianModel)
        return model(self.config.sh_degree)

    def make_birth_view(self, *, quantized=False):
        if not len(self.births._xyz):
            return None
        if self.config.birth.method == 'recongs' and not quantized:
            # Native ReCon-GS trains continuous newborns; quantize only at export.
            return self.births
        # The view immediately gathers its few source rows into owned tensors;
        # cloning every old GS attribute first only duplicates GPU traffic.
        reference = model_raw(self.gaussians, clone=False)
        return QuantizedNewGaussianView(self.births, self.config.birth,
            self.datamodule.cameras_extent, reference,
            source_rows=self.births.source_rows)

    def forward(self, viewpoint, training=False, old_only=False, proposal_only=False,
                predictor_only=False):
        auxiliary = proposal_only or predictor_only
        if (proposal_only and predictor_only) or (auxiliary and (
                not training or not self.frame_prepared or self.motion_committed)):
            raise ValueError('Auxiliary render is only available during motion training')
        old_model = self.renderer.gaussians
        effective = getattr(self.config, 'effective_motion_observations', False)
        previous_collection = getattr(self.renderer, 'collect_contributions', False)
        if effective:
            self.renderer.collect_contributions = bool(training and not auxiliary and self.frame_prepared and
                                                       not self.motion_committed)
        try:
            if self.frame_prepared and not self.motion_committed:
                xyz, rotation = (self.motion.predictor_pose() if predictor_only else
                                 self.motion.proposal_pose() if proposal_only else
                                 self.motion(self.trainer.global_step, training=training))
                self.gaussians.rigid_xyz = xyz
                self.gaussians.rigid_rotation = rotation
                if getattr(self, 'opacity_update', None) is not None:
                    if self.config.opacity_update.mode=='alpha':
                        self.gaussians.rigid_alpha = self.opacity_update.alphas()
                    else:
                        self.gaussians.rigid_opacity = self.opacity_update()
                    if auxiliary:
                        # Fit the auxiliary motion field with the main
                        # appearance solution, without fitting it twice.
                        for field in ('rigid_alpha', 'rigid_opacity'):
                            value = getattr(self.gaussians, field, None)
                            if value is not None:
                                setattr(self.gaussians, field, value.detach())
            elif self.birth_refit_active:
                # Motion is still relative to the preceding decoded frame.
                # Never add it to the already moved stage-one GS a second time.
                xyz, rotation = self.motion(self.trainer.global_step, training=training)
                self.gaussians.rigid_xyz = xyz
                self.gaussians.rigid_rotation = rotation
            if self.birth_opacity_active:
                if self.config.opacity_update.mode == 'alpha':
                    self.gaussians.rigid_alpha = self.opacity_update.alphas()
                else:
                    self.gaussians.rigid_opacity = self.opacity_update()
            if self.motion_committed and not self.frame_finalized and not old_only:
                birth_view = self.make_birth_view()
                if birth_view is not None:
                    self.renderer.gaussians = MergedGaussianModel([self.gaussians, birth_view])
            return self.renderer(viewpoint, self.background)
        finally:
            self.gaussians.rigid_xyz = None
            self.gaussians.rigid_rotation = None
            self.gaussians.rigid_opacity = None
            self.gaussians.rigid_alpha = None
            self.renderer.gaussians = old_model
            if effective:
                self.renderer.collect_contributions = previous_collection

    def training_step(self, viewpoint, step, regularization_weight=1.):
        package = self.forward(viewpoint, training=True)
        image = package['render']
        factor = motion_image_factor(step, self.motion_steps,
            enabled=self.config.motion_coarse_to_fine,
            fitting_motion=self.frame_prepared and not self.motion_committed,
            bootstrap_steps=self.config.rigid_tree.spatial_gate.start_step)
        image_loss = motion_photometric_loss(image, viewpoint.image,
                                              self.config.lambda_dssim, factor)
        loss = image_loss
        # Leave scalar diagnostics on device until all views have finished.
        # Converting each one to float would synchronize CUDA before backward.
        metrics = dict(image_loss=image_loss.detach(), motion_image_factor=factor,
                       psnr=psnr(image.detach(), viewpoint.image).detach(),
                       gs=len(self.gaussians._xyz) +
                       (len(self.births._xyz) if self.motion_committed else 0))
        proposal_weight = (getattr(self.motion, 'proposal_image_weight', 0.)
                           if not self.motion_committed else 0.)
        if proposal_weight:
            proposal = self.forward(viewpoint, training=True, proposal_only=True)['render']
            proposal_loss = motion_photometric_loss(proposal, viewpoint.image,
                                                    self.config.lambda_dssim, factor)
            loss = loss + proposal_weight * proposal_loss
            metrics['proposal_image_loss'] = proposal_loss.detach()
        predictor_weight = (getattr(self.motion, 'predictor_image_weight', 0.)
                            if not self.motion_committed else 0.)
        if predictor_weight:
            prediction = self.forward(viewpoint, training=True, predictor_only=True)['render']
            prediction_loss = motion_photometric_loss(prediction, viewpoint.image,
                self.config.lambda_dssim, factor)
            loss = loss + predictor_weight * prediction_loss
            metrics['predictor_image_loss'] = prediction_loss.detach()
        if not self.motion_committed and regularization_weight:
            loss = loss + regularization_weight * self.motion.regularization_loss(step)
        elif self.birth_refit_active and regularization_weight:
            refit_regularization = self.motion.fixed_support_regularization_loss(step)
            loss = loss + regularization_weight * refit_regularization
            metrics['motion_refit_regularization'] = (
                refit_regularization.detach() * regularization_weight)
        if (regularization_weight and getattr(self, 'opacity_update', None) is not None and
                (not self.motion_committed or self.birth_opacity_active)):
            terms = self.opacity_update.rate_terms(self._rate_byte_price(step),
                width=self.config.rigid_tree.sparse_rate_width_bins)
            loss = loss + regularization_weight * terms['loss']
        return loss, metrics, package

    def _rate_byte_price(self, step):
        cfg = self.config.rigid_tree
        ramp = min(max((step - cfg.rate_warmup_steps) / cfg.rate_ramp_steps, 0.), 1.)
        return cfg.rate_loss_weight * ramp / self.motion.rate_normalization_bytes

    @torch.no_grad()
    def _commit_opacity_and_retire(self):
        """Commit fitted old alpha, leaving point retirement for the birth boundary."""
        update = self.opacity_update
        logits = update.canonical_logits().to(self.device)
        rows = update.rows.clone()
        update.commit(rows)
        self.gaussians._opacity.copy_(logits)
        return rows

    @torch.no_grad()
    def _update_birth_opacity(self, step):
        """Let old alpha compete with births on the existing sparse codec grid."""
        if not self.birth_opacity_active:
            return
        birth_step = step - self.motion_steps
        window = self.config.rigid_tree.motion_fit_window_steps
        update = self.opacity_update
        before = support = None
        if self.config.opacity_update.birth_sources_only:
            before = update.delta.detach().clone()
            support = torch.zeros_like(update.delta, dtype=torch.bool)
            if len(self.births._xyz):
                support[update.rows[self.births.source_rows]] = True
            if update.delta.grad is not None:
                update.delta.grad.mul_(support)
        self.opacity_update.optimizer_step(
            self._rate_byte_price(step),
            proximal=(window is None or birth_step % window == 0 or
                      step == self.birth_commit_step),
            lr_factor=(self.motion._motion_lr_factor(self.motion_steps)
                       if self.config.opacity_update.decay_with_motion else 1.))
        if before is not None:
            # Zero gradients alone leave Adam momentum and proximal shrinkage
            # active. Preserve every unrelated row exactly after both act.
            update.delta.copy_(torch.where(support, update.delta, before))
        # Newborn predictors and the final packet use these same canonical
        # logits, while the alpha override carries image gradients above.
        self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
        self.birth_control['joint_opacity_steps'] = (
            self.birth_control.get('joint_opacity_steps', 0) + 1)
        self.birth_control['joint_opacity_scope'] = (
            'birth_sources' if self.config.opacity_update.birth_sources_only else 'all_old')

    def _training_views(self, frame, limit=None):
        """Load every training camera for one frame in stable name order."""
        previous_frame = self.datamodule.current_frame
        try:
            self.datamodule.current_frame = frame
            dataset = self.datamodule.get_train_dataset()
            indices = sorted(range(len(dataset)),
                             key=lambda i: Path(dataset.items[i].path).name)
            if limit is not None and len(indices) > limit:
                stride = len(indices) / limit
                indices = [indices[min(int(index * stride), len(indices) - 1)]
                           for index in range(limit)]
            return [dataset[index].to(self.device) for index in indices]
        finally:
            self.datamodule.current_frame = previous_frame

    @torch.no_grad()
    def _record_birth_pool_diagnostic(self, phase, step):
        """Measure a deterministic training-view sample for diagnosis only."""
        if not hasattr(self.datamodule, 'get_train_dataset'):
            return None
        views = self._birth_diagnostic_views.get(self.current_frame)
        if views is None:
            previous_frame = self.datamodule.current_frame
            try:
                self.datamodule.current_frame = self.current_frame
                dataset = self.datamodule.get_train_dataset()
                indices = sorted(range(len(dataset)),
                    key=lambda i: Path(dataset.items[i].path).name)
                indices = indices[:min(4, len(indices))]
                views = [dataset[index].to(self.device) for index in indices]
                self._birth_diagnostic_views[self.current_frame] = views
            finally:
                self.datamodule.current_frame = previous_frame
        rows = []
        for viewpoint in views:
            image = self.forward(viewpoint, training=False)['render']
            target = viewpoint.image
            l1 = l1_loss(image, target)
            dssim = 1 - ssim(image, target)
            objective = ((1 - self.config.lambda_dssim) * l1 +
                         self.config.lambda_dssim * dssim)
            rows.append((Path(viewpoint.path).name, l1, dssim,
                         psnr(image, target), objective))
        if not rows:
            return None
        record = dict(step=int(step), phase=str(phase),
            scope='training_views_diagnostic_only',
            views=[row[0] for row in rows], view_count=len(rows),
            old_count=len(self.gaussians._xyz), birth_count=len(self.births._xyz),
            gaussian_count=len(self.gaussians._xyz) + len(self.births._xyz),
            l1=float(torch.stack([row[1] for row in rows]).mean()),
            dssim=float(torch.stack([row[2] for row in rows]).mean()),
            psnr=float(torch.stack([row[3] for row in rows]).mean()),
            objective=float(torch.stack([row[4] for row in rows]).mean()))
        self.birth_control.setdefault('training_view_diagnostics', []).append(record)
        logging.debug('Birth pool training-view diagnostic: %s',
                      json.dumps(record, sort_keys=True))
        return record

    def _raw_gaussian_model(self, raw):
        """Wrap materialized attributes for ordinary contribution renders."""
        result = GaussianModel(self.config.sh_degree)
        result.active_sh_degree = self.gaussians.active_sh_degree
        result.spatial_lr_scale = self.gaussians.spatial_lr_scale
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(result, attr, torch.nn.Parameter(
                raw[name].detach(), requires_grad=False))
        return result

    @torch.no_grad()
    def _retirement_contributions(self):
        """Measure actual current/prior compositing for the same training cameras."""
        current_raw = model_raw(self.gaussians, clone=False)
        current_model = self._raw_gaussian_model(current_raw)
        current_renderer = Renderer(self.config.renderer, current_model)
        current = compositing_contributions(
            current_renderer, self._training_views(self.current_frame), self.background)

        previous_raw = dict(current_raw)
        reference = self.motion.frame_reference_state
        previous_raw['xyz'] = reference.xyz.to(self.device)
        previous_raw['rotation'] = reference.rotation.to(self.device)
        if self.opacity_update is not None:
            previous_raw['opacity'] = self.opacity_update.reference.to(self.device)
        previous_model = self._raw_gaussian_model(previous_raw)
        previous_renderer = Renderer(self.config.renderer, previous_model)
        previous = compositing_contributions(
            previous_renderer, self._training_views(self.current_frame - 1), self.background)
        return current, previous

    @torch.no_grad()
    def commit_birth_replacement(self):
        """Fix the final birth set and retire the required number of old rows."""
        control = self.birth_control
        if control.get('replacement_committed', False):
            retired = control['retirement']
            if (len(self.births._xyz) != retired['birth_count'] or
                    len(self.gaussians._xyz) + len(self.births._xyz) !=
                    retired['previous_count'] + retired['birth_count'] - retired['retired_count']):
                raise ValueError('Committed replacement requires an unchanged final birth set')
            return
        if not self.motion_committed or self.frame_finalized:
            raise ValueError('GS replacement requires fitted motion and an unfinalized birth pool')

        count = len(self.gaussians._xyz)
        candidates = len(self.births._xyz)
        expected_candidates = (control.get('clone_added', 0) +
                               control.get('proposal_added', 0) +
                               control.get('split_added', 0) -
                               control.get('pruned', 0))
        if expected_candidates != candidates:
            raise ValueError('Birth topology counters disagree with the final candidate pool')
        if self.config.birth.method == 'recongs':
            # Upstream renders all fitted additions in this frame. Their merge
            # and lowest-alpha eviction happen at the NEXT frame's start.
            control['retirement'].update(birth_count=candidates,
                candidate_count=candidates, surviving_count=count, rejected_birth_count=0)
            control['replacement_committed'] = True
            if self.opacity_update is not None:
                self.opacity_update.finish()
            return
        sources = (self.births.source_rows if candidates else
                   torch.empty(0, dtype=torch.long, device=self.device))
        if candidates:
            current, previous = self._retirement_contributions()
        else:
            current = previous = self.gaussians._xyz.new_zeros(count)
        rows, details = birth_retirement_rows(
            self.gaussians.get_opacity.flatten(), sources, current, previous,
            previous_birth_count=control.get('previous_birth_count', 0),
            count_tolerance=self.config.birth.replacement_count_tolerance,
            opacity_threshold=self.config.birth.opacity_threshold,
            protect_previous_births=self.config.birth.protect_previous_births)
        diagnose_retirement = (self.trainer.global_step == self.birth_commit_step)
        if diagnose_retirement:
            self._record_birth_pool_diagnostic('before_old_retirement',
                                               self.trainer.global_step)
        remove = torch.ones(count, dtype=torch.bool, device=self.device)
        remove[rows] = False
        retired = int(remove.sum())
        details.update(previous_count=count, candidate_count=candidates,
                       surviving_count=len(rows), rejected_birth_count=0,
                       retired_older_count=int(remove[:count - control.get('previous_birth_count', 0)].sum()),
                       retired_previous_birth_count=int(remove[count - control.get('previous_birth_count', 0):].sum()))

        if retired:
            self.retired_reference = dict(
                rows=remove.nonzero(as_tuple=True)[0].detach().cpu(),
                attributes={name: getattr(self.gaussians, attr).detach()[remove].cpu().clone()
                            for name, attr in zip(RAW_FIELDS[2:], RAW_ATTRIBUTES[2:])})
            remap = torch.full((count,), -1, dtype=torch.long, device=self.device)
            remap[rows] = torch.arange(len(rows), device=self.device)
            if candidates:
                self.births.source_rows = remap[self.births.source_rows]
                if bool(self.births.source_rows.lt(0).any()):
                    raise AssertionError('Retirement removed a required birth predictor')
            self.gaussians.prune_points(remove)
            self.motion.retain_survivors(rows)
            if self.opacity_update is not None:
                self.opacity_update.commit(self.opacity_update.rows[rows])
        control['retirement'] = details
        control['replacement_committed'] = True
        if self.opacity_update is not None:
            self.opacity_update.finish()
        if diagnose_retirement:
            self._record_birth_pool_diagnostic('after_old_retirement',
                                               self.trainer.global_step)

    def _set_birth_refit_trainable(self):
        if not self.motion.amplitude_refit_active or not self.motion._hard_gate_frozen:
            raise ValueError('Birth refitting requires frozen gates and selected motion support')
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
            getattr(self.motion, name).requires_grad_(True)
        self.motion.gate_logits.requires_grad_(False)
        self.motion._enforce_rotation_policy()

    @torch.no_grad()
    def _begin_birth_refit(self):
        if not self.birth_control.get('replacement_committed', False):
            raise ValueError('Birth refitting requires a committed fixed GS pool')
        self._set_birth_refit_trainable()
        # The new point set changes compositing. Discard the stage-one Adam
        # history once; a resumed refit instead restores its own saved moments.
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
            self.motion.optimizer.state.pop(getattr(self.motion, name), None)
        self.birth_control['refit_active'] = True

    def _birth_learning_rate_scale(self, step):
        """Cosine-decay each point from its own creation step through frame end."""
        if self.births.created_steps is None or not len(self.births.created_steps):
            return self.gaussians._xyz.new_empty(0)
        horizon = max(1, self.total_steps - self.motion_steps)
        age = (int(step) - self.births.created_steps).clamp(min=0)
        progress = (age.to(self.births._xyz) / horizon).clamp(max=1)
        final = self.config.birth.lr_final_ratio
        return final + (1 - final) * (1 + torch.cos(math.pi * progress)) / 2

    @torch.no_grad()
    def _birth_optimizer_step(self, step):
        if self.births.optimizer is None:
            return
        self._configure_birth_phase(step)
        cap = self.config.birth.position_step_sigma_cap
        if self.config.birth.lr_final_ratio == 1.0 and cap is None:
            self.births.optimizer.step()
            self.births.optimizer.zero_grad(set_to_none=True)
            return
        row_scale = self._birth_learning_rate_scale(step)
        parameters = [(group['name'], group['params'][0], group['params'][0].detach().clone())
                      for group in self.births.optimizer.param_groups]
        self.births.optimizer.step()
        if self.config.birth.lr_final_ratio != 1.0:
            for _, parameter, before in parameters:
                scale = row_scale.reshape((len(row_scale),) + (1,) * (parameter.ndim - 1))
                parameter.copy_(before + (parameter - before) * scale)
        if cap is not None:
            before = {name: value for name, _, value in parameters}
            self.births.constrain_position_step(
                before['xyz'], before['scaling'], before['rotation'], cap,
                self.config.birth.new_xyz_step_extent * self.datamodule.cameras_extent)
        self.births.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _reset_densification_stats(self):
        self.gaussians.xyz_gradient_accum.zero_()
        self.gaussians.denom.zero_()
        if len(self.births._xyz):
            self.births.xyz_gradient_accum.zero_()
            self.births.denom.zero_()

    def _old_birth_sources(self):
        """Build a broad pool from observed, high-gradient small Gaussians."""
        denominator = self.gaussians.denom.flatten()
        grads = self.gaussians.xyz_gradient_accum / denominator.clamp_min(1)[:, None]
        score = torch.linalg.vector_norm(grads, dim=-1)
        observed = denominator.gt(0)
        high_gradient = score.ge(self.config.birth.densify_grad_threshold)
        large_scale = self.gaussians.get_scaling.max(dim=1).values
        small = large_scale.le(self.config.gaussian_stage2.percent_dense *
                               self.datamodule.cameras_extent)
        gradient_eligible = observed & high_gradient
        small_eligible = gradient_eligible & small
        active_source = torch.zeros_like(observed)
        if (getattr(self, 'births', None) is not None and
                getattr(self.births, 'source_rows', None) is not None and
                len(self.births.source_rows)):
            active_rows = self.births.source_rows.to(device=active_source.device,
                                                      dtype=torch.long)
            active_source[active_rows] = True
        # A large source produces a 1/1.6-scale refinement child. Do not spend
        # the budget on children already beyond the existing world-size limit.
        child_scale = torch.where(small, large_scale, large_scale / 1.6)
        valid_size = child_scale.le(0.1 * self.datamodule.cameras_extent)
        duplicate = gradient_eligible & active_source
        # Keep the established small-GS densification rule. Let birth fitting
        # and opacity pruning decide which candidates are useful instead of
        # truncating the pool with a noisy per-frame residual score.
        eligible = small_eligible & valid_size & ~active_source
        selected = select_budget_mask(
            eligible, score, self.config.birth.max_new_gaussians)
        self._old_birth_source_diagnostics = dict(
            observed=int(observed.sum()),
            high_gradient_observed=int(gradient_eligible.sum()),
            small_high_gradient=int(small_eligible.sum()),
            large_high_gradient=int((gradient_eligible & ~small).sum()),
            duplicate_active_source=int(duplicate.sum()),
            oversized_child_sources=int((gradient_eligible & ~valid_size).sum()),
            budget_blocked=int(eligible.sum() - selected.sum()),
            selected=int(selected.sum()), selected_small=int((selected & small).sum()),
            selected_large=int((selected & ~small).sum()),
            selected_gradient_mean=(float(score[selected].mean())
                                    if bool(selected.any()) else 0.))
        return selected.nonzero(as_tuple=True)[0]

    def _rescore_old_birth_sources_after_motion(self):
        """Measure old-GS screen gradients after final motion has been frozen."""
        view_limit = self.config.birth.source_gradient_rescore_views
        if view_limit == 0:
            return
        views = self._training_views(self.current_frame, limit=view_limit)
        if not views:
            raise ValueError('Post-motion source scoring requires training views')
        self.gaussians.xyz_gradient_accum.zero_()
        self.gaussians.denom.zero_()
        self.gaussians.max_radii2D.zero_()
        if self.gaussians.optimizer is not None:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.motion.optimizer is not None:
            self.motion.optimizer.zero_grad(set_to_none=True)
        view_names = []
        with torch.enable_grad():
            for viewpoint in views:
                package = self.forward(viewpoint, training=True, old_only=True)
                loss = motion_photometric_loss(package['render'], viewpoint.image,
                    self.config.lambda_dssim)
                loss.backward()
                visible = package['visibility_filter']
                viewspace = package['viewspace_points']
                if viewspace.grad is None:
                    raise RuntimeError('Post-motion source scoring produced no screen gradient')
                self.gaussians.add_densification_stats(viewspace, visible)
                radii = package['radii']
                self.gaussians.max_radii2D[visible] = torch.maximum(
                    self.gaussians.max_radii2D[visible], radii[visible])
                view_names.append(Path(viewpoint.path).name)
        if self.gaussians.optimizer is not None:
            self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.motion.optimizer is not None:
            self.motion.optimizer.zero_grad(set_to_none=True)
        diagnostic = dict(step=int(self.motion_steps), requested_views=int(view_limit),
                          used_views=len(view_names), views=view_names,
                          old_gs=int(len(self.gaussians._xyz)))
        self.birth_control['source_gradient_rescore'] = diagnostic
        logging.debug('Post-motion source gradient rescore: %s',
                      json.dumps(diagnostic, sort_keys=True))

    def _record_old_birth_source_filter(self, step):
        diagnostic = dict(step=int(step), **getattr(
            self, '_old_birth_source_diagnostics', {}))
        self.birth_control.setdefault('old_source_filter_events', []).append(diagnostic)
        logging.debug('Old-source birth filter: %s', json.dumps(diagnostic, sort_keys=True))
        return diagnostic

    @torch.no_grad()
    def _begin_births(self):
        xyz, rotation = self.motion(self.motion_steps, training=False)
        self.gaussians.apply_selected_motion(xyz, rotation)
        self.motion_committed = True
        for parameter in self.motion.parameters():
            parameter.requires_grad_(False)
        self.motion.optimizer.zero_grad(set_to_none=True)
        self.births = self._new_birth_model()
        self.births.active_sh_degree = self.gaussians.active_sh_degree
        self.births.spatial_lr_scale = self.gaussians.spatial_lr_scale
        self._freeze_base()
        self._record_birth_pool_diagnostic('before_initial_births', self.motion_steps)
        if self.config.birth.method == 'recongs':
            self.births.initialize_recongs_sources(self.gaussians, self.motion_steps,
                self.config.gaussian_stage2, self.config.birth.densify_grad_threshold,
                self.datamodule.cameras_extent)
        elif self.config.birth.residual_proposal_views:
            views = self._training_views(
                self.current_frame, limit=self.config.birth.residual_proposal_views)
            raw = model_raw(self.gaussians, clone=False)
            proposal = propose_multiview_births(
                self.renderer, views, raw, self.background,
                self.datamodule.cameras_extent,
                MultiViewBirthConfig(candidate_min_alpha=(
                    self.config.birth.residual_candidate_min_alpha)),
                source_xyz=raw['xyz'], existing_xyz=raw['xyz'])
            born, sources, report = proposal
            self.births.initialize_from_sources(
                self.gaussians, None, self.motion_steps,
                self.config.gaussian_stage2, proposal=(born, sources))
            self.birth_control['proposal_added'] = len(sources)
            self.birth_control['multiview_proposal'] = report
            self.birth_control['first_birth_sources'] = len(sources)
        else:
            self._rescore_old_birth_sources_after_motion()
            sources = self._old_birth_sources()
            self._record_old_birth_source_filter(self.motion_steps)
            self.births.initialize_covariance_sources(
                self.gaussians, sources, self.motion_steps,
                self.config.gaussian_stage2,
                sigma_scale=self.config.birth.clone_sigma_scale,
                initial_alpha=self.config.birth.initial_alpha,
                split_scale_threshold=(self.config.gaussian_stage2.percent_dense *
                                       self.datamodule.cameras_extent),
                max_screen_size=self.config.birth.new_max_screen_size)
        added = len(self.births._xyz)
        if not self.config.birth.residual_proposal_views:
            self.birth_control['clone_added'] = added
            self.birth_control['first_birth_sources'] = added
        self._configure_birth_phase(self.motion_steps + 1)
        self._reset_densification_stats()
        self._record_birth_learning(self.motion_steps)
        self._record_birth_pool_diagnostic('after_initial_births', self.motion_steps)

    @torch.no_grad()
    def _densify_births(self, step):
        """Split and prune the current pool; old points seed births only once."""
        self._record_birth_pool_diagnostic(f'before_densify_{step}', step)
        result = (self.births.densify_and_prune(
            self.config.birth.densify_grad_threshold,
            self.config.birth.opacity_threshold,
            self.datamodule.cameras_extent,
            self.config.birth.new_max_screen_size,
            step, max_points=self.config.birth.max_new_gaussians)
            if self.births.optimizer is not None else
            dict(clone_added=0, split_added=0, split_children=0,
                 split_parents_replaced=0, pruned=0, prune_reasons={}))
        self.birth_control['clone_added'] += result['clone_added']
        self.birth_control['split_added'] += result['split_added']
        self.birth_control['split_children'] = (
            self.birth_control.get('split_children', 0) + result['split_children'])
        self.birth_control['split_parents_replaced'] = (
            self.birth_control.get('split_parents_replaced', 0) +
            result['split_parents_replaced'])
        self.birth_control['pruned'] += result['pruned']
        reasons = self.birth_control.setdefault('prune_reasons', {})
        for name, count in result.get('prune_reasons', {}).items():
            reasons[name] = reasons.get(name, 0) + count
        self.birth_control.setdefault('densification_events', []).append(dict(
            step=step,
            newborn_clones=result['clone_added'],
            newborn_split_children_count=result['split_children'],
            newborn_split_parents_replaced=result['split_parents_replaced'],
            split_net_added=result['split_added'],
            pruned=result['pruned']))
        self._reset_densification_stats()
        self._configure_birth_phase(step + 1)
        self._record_birth_pool_diagnostic(f'after_densify_{step}', step)

    @torch.no_grad()
    def _prune_terminal_births(self):
        """Keep useful, compact newborns before fixing K and retiring old rows."""
        model = self.births
        cfg = self.config.birth
        if not len(model._xyz):
            return 0
        view = self.make_birth_view()
        low_opacity = (view.get_opacity.flatten() <
                       cfg.terminal_opacity_threshold)
        big_screen = model.max_radii2D > cfg.new_max_screen_size
        big_world = (view.get_scaling.max(dim=1).values >
                     0.1 * self.datamodule.cameras_extent)
        remove = low_opacity | big_screen | big_world
        budget_removed = torch.zeros_like(remove)
        if cfg.terminal_max_gaussians is not None:
            # Densification statistics were reset at step 260, so this score
            # measures average projected-position gradient over the final
            # 20 fitting steps. Alpha/size filters run first; the point cap
            # then keeps the candidates that still affect the rendered loss.
            denominator = model.denom.flatten()
            gradient_score = (model.xyz_gradient_accum.flatten() /
                              denominator.clamp_min(1))
            gradient_score = torch.nan_to_num(
                gradient_score, nan=0.0, posinf=0.0, neginf=0.0)
            candidates = ~remove
            selected = select_budget_mask(
                candidates, gradient_score, cfg.terminal_max_gaussians)
            budget_removed = candidates & ~selected
            remove |= budget_removed
            candidate_scores = gradient_score[candidates]
            kept_scores = gradient_score[selected]
            self.birth_control.setdefault('terminal_selection_events', []).append({
                'step': int(self.trainer.global_step),
                'cap': cfg.terminal_max_gaussians,
                'candidate_count': int(candidates.sum()),
                'kept_count': int(selected.sum()),
                'removed_by_gradient_budget': int(budget_removed.sum()),
                'observed_candidates': int((candidates & denominator.gt(0)).sum()),
                'candidate_gradient_mean': (float(candidate_scores.mean())
                                            if candidate_scores.numel() else 0.0),
                'kept_gradient_mean': (float(kept_scores.mean())
                                       if kept_scores.numel() else 0.0),
            })
        reasons = self.birth_control.setdefault('prune_reasons', {})
        for name, mask in (('low_opacity', low_opacity), ('screen_size', big_screen),
                           ('world_size', big_world),
                           ('multiple', (low_opacity.int() + big_screen.int() +
                                         big_world.int()) > 1)):
            reasons[name] = reasons.get(name, 0) + int(mask.sum())
        if budget_removed.any():
            reasons['terminal_gradient_budget'] = (
                reasons.get('terminal_gradient_budget', 0) + int(budget_removed.sum()))
        removed = int(remove.sum())
        if removed:
            model.prune_points(remove)
        return removed

    @torch.no_grad()
    def post_training_step(self, package):
        step = self.trainer.global_step
        packages = package if isinstance(package, list) else [package]
        if not self.motion_committed:
            evidence_counts = []
            for item in packages:
                frustum = item['visibility_filter']
                observed = (item.get('motion_contribution_filter')
                    if getattr(self.config, 'effective_motion_observations', False) else frustum)
                if observed is None or observed.dtype != torch.bool or observed.shape != frustum.shape:
                    raise ValueError('Motion evidence requires an aligned actual compositing mask')
                self.motion.mark_observed(observed)
                evidence_counts.append(torch.stack((frustum.sum(), observed.sum())))
            # These counts are diagnostics only. Keep them on device until a
            # checkpoint or phase boundary instead of stalling every update.
            frustum_count, observed_count = torch.stack(evidence_counts).sum(0).unbind()
            self.birth_control['motion_frustum_samples'] = (
                self.birth_control.get('motion_frustum_samples', 0) + frustum_count)
            self.birth_control['motion_observed_samples'] = (
                self.birth_control.get('motion_observed_samples', 0) + observed_count)
            edits = self.motion.optimizer_step()
            if getattr(self, 'opacity_update', None) is not None:
                self.opacity_update.optimizer_step(self._rate_byte_price(step),
                    proximal=self.motion._motion_fit_window_due(),
                    lr_factor=(self.motion._motion_lr_factor(step)
                        if self.config.opacity_update.decay_with_motion else 1.))
            if edits:
                self.birth_control['tree_edit_count'] = (
                    self.birth_control.get('tree_edit_count', 0) + len(edits))
            start = (1 if getattr(self.config.birth, 'method', 'covariance') == 'recongs' else
                     self.motion_steps - self.config.birth.densification_stats_steps + 1)
            if step == start:
                self.gaussians.xyz_gradient_accum.zero_()
                self.gaussians.denom.zero_()
                self.gaussians.max_radii2D.zero_()
            if step >= start:
                for item in packages:
                    visible = item['visibility_filter']
                    self.gaussians.max_radii2D[visible] = torch.maximum(
                        self.gaussians.max_radii2D[visible], item['radii'][visible])
                    if len(packages) > 1:
                        # Average optimizer gradients; preserve per-view birth
                        # score units and the existing densification threshold.
                        item['viewspace_points'].grad.mul_(len(packages))
                    self.gaussians.add_densification_stats(
                        item['viewspace_points'], item['visibility_filter'])
            if step == self.motion_steps:
                materialize_motion_evidence_counts(self.birth_control)
                if getattr(self, 'opacity_update', None) is not None:
                    self._commit_opacity_and_retire()
                self.motion.record_final_diagnostics()
                directory = self.output_dir / 'region_statistics'
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'frame_{self.current_frame:03d}.json').write_text(
                    json.dumps(self.motion.flat_region_diagnostics, indent=2))
                self._begin_births()
        else:
            birth_step = step - self.motion_steps
            commit_step = self.birth_commit_step
            trace = (step == commit_step or step == self.total_steps or
                     birth_step % self.config.birth.densify_interval == 0)
            gradients = {}
            if trace and len(self.births._xyz):
                gradients = {
                    name: parameter.grad.detach().norm()
                    for name, parameter in (
                        ('xyz_grad_norm', self.births._xyz),
                        ('color_grad_norm', self.births._features_dc))
                    if parameter.grad is not None}

            old_count = len(self.gaussians._xyz)
            for item in packages:
                visible = item['visibility_filter']
                radii = item['radii']
                if len(self.births._xyz):
                    birth_visible = visible[old_count:]
                    birth_radii = radii[old_count:]
                    self.births.max_radii2D[birth_visible] = torch.maximum(
                        self.births.max_radii2D[birth_visible], birth_radii[birth_visible])
                if step <= commit_step:
                    if len(packages) > 1:
                        item['viewspace_points'].grad.mul_(len(packages))
                    if len(self.births._xyz):
                        self.births.add_densification_stats(
                            item['viewspace_points'], visible[old_count:], align='right')

            native_densify = (self.config.birth.method == 'recongs' and
                birth_step > 0 and birth_step % self.config.birth.densify_interval == 0)
            if native_densify:
                self._densify_births(step)
            self._birth_optimizer_step(step)
            self._update_birth_opacity(step)
            if self.birth_refit_active:
                self.motion.optimizer_step()
                xyz, rotation = self.motion(step, training=False)
                self.gaussians.apply_selected_motion(xyz, rotation)

            # Create the old-source pool once, split/prune at configured
            # intervals, and fit the final gradient before fixing K and retiring.
            if (self.config.birth.method != 'recongs' and
                    step < commit_step and birth_step > 0 and
                    birth_step % self.config.birth.densify_interval == 0):
                self._densify_births(step)
            if step == commit_step:
                if self.config.birth.method != 'recongs':
                    self._record_birth_pool_diagnostic('before_terminal_birth_prune', step)
                    self.birth_control['pruned'] += self._prune_terminal_births()
                    self._record_birth_pool_diagnostic('after_terminal_birth_prune', step)
                self.commit_birth_replacement()
                if self.config.birth.refit_steps:
                    self._begin_birth_refit()
            if trace:
                self._record_birth_learning(step, gradients)
            if step == self.total_steps:
                timer = getattr(self.trainer, '_frame_timer', None)
                with timer.phase('encoding') if timer else nullcontext():
                    self.finalize_frame()

    @torch.no_grad()
    def finalize_frame(self):
        if self.frame_finalized:
            return
        self.commit_birth_replacement()
        if self.opacity_update is not None:
            self.opacity_update.finish()
        if self.config.birth.refit_steps:
            self.motion.record_final_diagnostics()
            directory = self.output_dir / 'region_statistics'
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f'frame_{self.current_frame:03d}.json').write_text(
                json.dumps(self.motion.flat_region_diagnostics, indent=2))
        view = (self.make_birth_view(quantized=True) if self.config.birth.method == 'recongs'
                else self.make_birth_view())
        raw, sources = materialize_quantized_gaussians(
            model_raw(self.gaussians, clone=False), view)
        stream_dir = self.output_dir / 'frame_storage' / 'rigid_tree'
        stream_dir.mkdir(parents=True, exist_ok=True)
        motion_path = stream_dir / f'motion_{self.current_frame:03d}.bin'
        region_xyz, region_rot, gs_xyz, gs_rot = self.motion._values(hard=True)
        activity = dict(
            moving_regions=int((region_xyz[1:].ne(0).any(-1) |
                                region_rot[1:].ne(0).any(-1)).sum()),
            old_gs_with_local_residual=int((gs_xyz.ne(0).any(-1) |
                                            gs_rot.ne(0).any(-1)).sum()),
            old_gs=len(gs_xyz), topology_edits=self.birth_control.get('tree_edit_count', 0))
        self.motion.register_materialized_layout(raw['xyz'], raw['rotation'],
                                                 birth_source_rows=sources)
        rate_terms = self.motion.loss_terms(self.motion_steps, diagnostics=True)
        motion_info = self.motion.export_motion(motion_path,
            inherit_birth_rotation=(not self.config.birth.rotation_updates and
                                    not self.config.rigid_tree.rotation_updates))
        opacity_info = None
        if self.opacity_update is not None:
            opacity_info = self.opacity_update.export(
                stream_dir / f'opacity_{self.current_frame:03d}.bin', self.current_frame)
            expected_logits = self.opacity_update.canonical_logits().to(raw['opacity'])
            if not torch.equal(raw['opacity'][:len(self.motion.owner)], expected_logits):
                raise ValueError('Materialized old opacity differs from its sparse packet')
        decoded = self.motion.codec_previous_state
        prior = self.motion.frame_reference_state
        old_count = len(self.motion.owner)
        previous_rows = self.motion.previous_rows.detach().cpu().long()
        retired_count = len(prior.xyz) - old_count
        activity.update(old_gaussian_pose_activity(
            decoded.xyz[:old_count], decoded.rotation[:old_count],
            prior.xyz[previous_rows], prior.rotation[previous_rows],
            self.motion.steps, self.motion.gs_steps))
        raw['xyz'] = decoded.xyz.to(device=self.device, dtype=raw['xyz'].dtype)
        raw['rotation'] = decoded.rotation.to(device=self.device,
                                               dtype=raw['rotation'].dtype)
        birth_info = None
        if len(sources):
            birth_path = stream_dir / f'birth_{self.current_frame:03d}.bin'
            birth_info = export_birth_attributes(birth_path,
                {name: raw[name][-len(sources):] for name in
                 ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                len(raw['xyz']) - len(sources),
                symbols={name: view.symbols[name] for name in
                         ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                steps={name: view.steps[name] for name in
                       ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                predictor={name: view.predictor[name] for name in
                           ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                source_rows=sources)
        if self.config.birth.method == 'recongs':
            if len(sources):
                # ReCon-GS merges via densification_postfix: retain old Adam
                # steps/settings and append zero moments for the fitted births.
                self.gaussians.densification_postfix(
                    raw['xyz'][-len(sources):], raw['sh_dc'][-len(sources):],
                    raw['sh_rest'][-len(sources):], raw['opacity'][-len(sources):],
                    raw['logscale'][-len(sources):], raw['rotation'][-len(sources):])
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                getattr(self.gaussians, attr).copy_(raw[name])
        else:
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                setattr(self.gaussians, attr,
                        torch.nn.Parameter(raw[name].detach().clone()))
            self.gaussians.training_setup(self.config.gaussian)
        self.gaussians.max_radii2D = self.gaussians._xyz.new_zeros(len(raw['xyz']))
        self._freeze_base()
        self.renderer.gaussians = self.gaussians
        self.births = self._new_birth_model()
        self.frame_finalized = True
        self.retired_reference = None
        manifest = dict(version=1, frame=self.current_frame,
            reference=str(self.trainer.decoder_reference_path.relative_to(
                self.output_dir)),
            motion=motion_path.name,
            birth=(Path(birth_info['path']).name if birth_info else None),
            old_count=len(raw['xyz']) - len(sources), birth_count=len(sources),
            total_count=len(raw['xyz']),
            decoded_state_sha256=decoded.digest().hex(),
            raw_state_sha256=reference_digest(raw),
            previous_count=len(prior.xyz), retired_count=retired_count,
            no_old_deletions=retired_count == 0)
        if self.opacity_update is not None:
            manifest.update(opacity_update=Path(opacity_info['path']).name if opacity_info else None,
                            opacity_update_timing='before_births')
        manifest_path = stream_dir / f'frame_{self.current_frame:03d}.json'
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.frame_report = dict(frame=self.current_frame,
            initial_reference=self.initial_reference,
            schedule=dict(motion_steps=self.motion_steps,
                densification_stats_start_step=(self.motion_steps -
                    self.config.birth.densification_stats_steps + 1),
                initial_birth_step=self.motion_steps,
                densify_steps=[step for step in range(
                    self.motion_steps + self.config.birth.densify_interval,
                    self.birth_commit_step, self.config.birth.densify_interval)],
                final_pool_step=self.birth_commit_step,
                refit_steps=self.config.birth.refit_steps,
                refit_start_step=(self.birth_commit_step + 1
                                  if self.config.birth.refit_steps else None),
                birth_allocation=self.birth_allocation,
                total_steps=self.total_steps),
            region_count=int(self.motion.owner.unique().numel()),
            candidate_region_count=len(self.motion.parents),
            initialization=dict(spatial=True, initial_regions=8,
                gate_start_step=self.config.rigid_tree.spatial_gate.start_step,
                gate_freeze_step=self.config.rigid_tree.spatial_gate.freeze_step,
                optimization_steps=self.total_steps),
            old_count=len(raw['xyz']) - len(sources), birth_count=len(sources),
            previous_count=len(prior.xyz), retired_count=retired_count,
            total_count=len(raw['xyz']), motion=motion_info, birth=birth_info,
            opacity_update=opacity_info,
            motion_observation_diagnostics=dict(
                kind=('actual_compositing' if self.config.effective_motion_observations else 'frustum'),
                frustum_samples=self.birth_control.get('motion_frustum_samples', 0),
                used_samples=self.birth_control.get('motion_observed_samples', 0),
                old_gs_with_no_motion_observation=int(self.motion.observations.eq(0).sum()),
                scope='surviving GS; current motion optimization views; excludes birth and heldout renders'),
            stage_image_metrics=self.stage_image_metrics,
            encoded_bytes=motion_path.stat().st_size +
                (birth_info['archive_bytes'] if birth_info else 0) +
                (opacity_info['archive_bytes'] if opacity_info else 0),
            shared_reference_bytes=self.trainer.decoder_reference_path.stat().st_size,
            materialized_sha256=reference_digest(raw),
            decoder_reference=manifest['reference'],
            frame_manifest=str(manifest_path.relative_to(self.output_dir)),
            motion_activity=activity,
            motion_rate_diagnostics={name: float(value.detach())
                                     for name, value in rate_terms.items()},
            opacity_rate_diagnostics=({name: float(value.detach()) for name, value in
                self.opacity_update.rate_terms(self._rate_byte_price(self.motion_steps),
                    width=self.config.rigid_tree.sparse_rate_width_bins).items()}
                if self.opacity_update is not None else None),
            flat_region_diagnostics=self.motion.flat_region_diagnostics,
            birth_control=self.birth_control,
            semantics='Spatial gate motion and quantized births; shared first frame counted once. '
                      'Previous-row deletions are encoded in motion; optional quantized old-opacity deltas; '
                      'old scale and higher-order SH fixed.')

    @torch.no_grad()
    def validation_step(self, viewpoint, idx, loader_idx):
        image = self.forward(viewpoint)['render'].clamp(0, 1)
        if idx < self.config.num_saving_images and self.trainer.global_step == self.total_steps:
            root = Path(self.datamodule.config.root)
            path = (self.output_dir / 'image' /
                    viewpoint.path.relative_to(root))
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(image, path.with_name(path.stem +
                       f'_step{self.trainer.global_step}' + path.suffix))
            gt_path = path.with_name(path.stem + '_gt' + path.suffix)
            if not gt_path.exists() and not gt_path.is_symlink():
                save_image(viewpoint.image, gt_path)
        metrics = dict(l1=float(l1_loss(image, viewpoint.image)),
                       psnr=float(psnr(image, viewpoint.image)),
                       ssim=float(ssim(image, viewpoint.image)))
        if self.config.lpips_required:
            metrics['lpips'] = float(lpips(image, viewpoint.image,
                                          net_type='vgg').mean())
        return metrics

    def validation_end(self, results, num_loaders=1):
        groups = [results] if num_loaders == 1 else results
        summary = {}
        for name, rows in zip(self.datamodule.eval_names, groups):
            if rows:
                summary[name] = {key: sum(row[key] for row in rows) / len(rows)
                                 for key in rows[0]}
                self.log_dict({f'eval-{name}/{key}': value
                               for key, value in summary[name].items()})
        return summary
