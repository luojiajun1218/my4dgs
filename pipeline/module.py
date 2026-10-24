"""Spatial gate motion, GS residuals, and quantized newborn Gaussians."""

from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
import json
import math
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

from .model.gaussian import BirthGaussianModel, GaussianModel
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
from .model.birth_retirement import (retirement_count_trials, rank_retirement_candidates,
    choose_validation_trial, compositing_contributions)
from .model.birth_selection import (birth_image_loss, midpoint_image_utility,
    useful_birth_subsets)


BIRTH_ALLOCATION = 'clone_small_v1'
CURRENT_BIRTH_POLICIES = ('balanced_current_births', 'birth_driven_retirement',
                        'multiview_gain_retirement')


def materialize_motion_evidence_counts(control):
    """Transfer diagnostic counters once, without synchronizing each update."""
    keys = ('motion_frustum_samples', 'motion_observed_samples')
    values = [control.get(key, 0) for key in keys]
    tensors = [value for value in values if torch.is_tensor(value)]
    if tensors:
        counts = torch.stack([torch.as_tensor(value, device=tensors[0].device)
                              for value in values]).tolist()
        control.update(zip(keys, counts))


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
    """Stage-two candidate fitting and exact quantized attribute grid."""

    source_covariance_init: bool = False
    replace_low_opacity: bool = False
    replacement_policy: str = 'none'
    replacement_count_tolerance: float = 0.0
    refit_steps: int = 0
    warmup_steps: int = 0
    initial_opacity: Optional[float] = None  # Actual alpha; None preserves legacy logit=0.1.
    rotation_updates: bool = True
    clone_sigma_scale: float = 2.0
    densification_stats_steps: int = 30
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01
    new_max_screen_size: float = 20.0
    max_new_gaussians: Optional[int] = None
    new_xyz_step_extent: float = 1e-5
    new_rotation_step: float = 0.001
    new_logscale_step: float = 0.001
    new_opacity_step: float = 0.01
    new_sh_step: float = 0.001
    strategy: str = 'gradient_clone'
    multiview: MultiViewBirthConfig = field(default_factory=MultiViewBirthConfig)
    validation_cameras: list[str] = field(default_factory=list)
    validation_min_gain_db: float = .005
    validation_max_ssim_drop: float = 1e-4
    fit_loss: str = 'photo'
    fit_ssim_weight: float = 0.
    admission_useful_fractions: list[float] = field(default_factory=list)
    refit_all_training_views: bool = False
    refit_motion: bool = True
    lr_final_ratio: float = 1.
    protect_previous_births: bool = True

    def validate(self, motion_steps):
        if self.strategy not in ('gradient_clone', 'multiview_residual'):
            raise ValueError('Unknown birth.strategy')
        if self.fit_loss not in ('photo', 'mse', 'mse_ssim'):
            raise ValueError('Unknown birth.fit_loss')
        if (isinstance(self.fit_ssim_weight, bool) or not math.isfinite(self.fit_ssim_weight)
                or self.fit_ssim_weight < 0 or
                (self.fit_loss == 'mse_ssim' and self.fit_ssim_weight == 0)):
            raise ValueError('mse_ssim needs a finite positive birth.fit_ssim_weight')
        if (not isinstance(self.refit_all_training_views, bool) or
                len(set(self.admission_useful_fractions)) != len(self.admission_useful_fractions) or
                any(isinstance(f, bool) or not math.isfinite(f) or not 0 < f <= 1
                    for f in self.admission_useful_fractions)):
            raise ValueError('Invalid useful birth fractions or refit camera policy')
        if self.admission_useful_fractions and self.strategy != 'multiview_residual':
            raise ValueError('Image-utility selection requires multi-view births')
        if (not isinstance(self.refit_motion, bool) or
                not isinstance(self.protect_previous_births, bool)):
            raise ValueError('Birth refit/protection flags must be boolean')
        for name in ('validation_min_gain_db', 'validation_max_ssim_drop'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'birth.{name} must be finite and nonnegative')
        if not 0 < self.lr_final_ratio <= 1:
            raise ValueError('birth.lr_final_ratio must be in (0, 1]')
        if self.strategy == 'multiview_residual':
            self.multiview.validate()
            if (not self.validation_cameras or len(set(self.validation_cameras)) != len(self.validation_cameras)
                    or self.replacement_policy != 'multiview_gain_retirement' or self.warmup_steps):
                raise ValueError('Multi-view births require validation cameras, gain retirement and no alpha warmup')
        elif self.replacement_policy == 'multiview_gain_retirement':
            raise ValueError('Multi-view retirement requires multi-view births')
        if not isinstance(self.source_covariance_init, bool):
            raise ValueError('source_covariance_init must be boolean')
        if not isinstance(self.replace_low_opacity, bool):
            raise ValueError('birth.replace_low_opacity must be boolean')
        if self.replacement_policy not in ('none', *CURRENT_BIRTH_POLICIES):
            raise ValueError('Unknown birth.replacement_policy')
        if self.replace_low_opacity and self.replacement_policy != 'none':
            raise ValueError('Legacy and current-frame GS replacement are mutually exclusive')
        if (isinstance(self.replacement_count_tolerance, bool) or
                not isinstance(self.replacement_count_tolerance, (int, float)) or
                not math.isfinite(self.replacement_count_tolerance) or
                not 0 <= self.replacement_count_tolerance < 1):
            raise ValueError('birth.replacement_count_tolerance must be in [0, 1)')
        if self.replacement_count_tolerance and self.replacement_policy not in CURRENT_BIRTH_POLICIES:
            raise ValueError('Replacement count tolerance requires a current-birth retirement policy')
        if (isinstance(self.refit_steps, bool) or not isinstance(self.refit_steps, int) or
                self.refit_steps < 0):
            raise ValueError('birth.refit_steps must be a nonnegative integer')
        if (isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or
                self.warmup_steps < 0):
            raise ValueError('birth.warmup_steps must be a nonnegative integer')
        if self.initial_opacity is not None and (
                isinstance(self.initial_opacity, bool) or
                not isinstance(self.initial_opacity, (int, float)) or
                not math.isfinite(self.initial_opacity) or not 0 < self.initial_opacity < 1):
            raise ValueError('birth.initial_opacity must be an alpha in (0, 1) or null')
        if self.warmup_steps and self.initial_opacity is None:
            raise ValueError('Birth warmup needs an explicit initial_opacity alpha')
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
        if self.max_new_gaussians is not None and (
                isinstance(self.max_new_gaussians, bool) or
                not isinstance(self.max_new_gaussians, int) or
                self.max_new_gaussians < 1):
            raise ValueError('birth.max_new_gaussians must be positive or null')


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
    gaussian_stage2: GaussianConfig = field(default_factory=GaussianConfig)
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


@torch.no_grad()
def replace_low_opacity_gaussians(model, previous_birth_count):
    """ReCon-GS replacement: retire as many combined rows as were just added.

    Keep the surviving order so the existing motion packet can encode deleted
    previous-frame row IDs. Only retired appearance is retained for mid-frame
    checkpoint bootstraps; it is not an extra transmitted appearance update.
    """
    count = len(model._xyz)
    if (isinstance(previous_birth_count, bool) or
            not isinstance(previous_birth_count, int) or
            not 0 <= previous_birth_count < count):
        raise ValueError('Previous birth count must leave at least one surviving GS')
    remove = torch.zeros(count, dtype=torch.bool, device=model._xyz.device)
    retired = None
    if previous_birth_count:
        opacity = torch.sigmoid(model._opacity.detach().flatten())
        if opacity.shape != (count,) or not bool(torch.isfinite(opacity).all()):
            raise ValueError('Replacement requires finite per-GS opacity')
        weakest = torch.argsort(opacity, stable=True)[:previous_birth_count]
        remove[weakest] = True
        retired = dict(rows=remove.nonzero(as_tuple=True)[0].cpu(),
            attributes={name: getattr(model, attr).detach()[remove].cpu().clone()
                        for name, attr in zip(RAW_FIELDS[2:], RAW_ATTRIBUTES[2:])})
        model.prune_points(remove)
    rows = (~remove).nonzero(as_tuple=True)[0]
    return rows, retired, dict(
        previous_count=count, previous_birth_count=previous_birth_count,
        retired_count=previous_birth_count,
        retired_older_count=int(remove[:count - previous_birth_count].sum()),
        retired_previous_birth_count=int(remove[count - previous_birth_count:].sum()),
        surviving_count=len(rows))


@torch.no_grad()
def balanced_replacement_rows(old_alpha, birth_alpha, source_rows, *, count_tolerance=0.0):
    """Select a constant-size pool by quantized alpha, retaining birth sources.

    Start with the strongest N rows of N old + M newborn GS. A retained birth
    needs its old source as a decoder predictor: restore missing sources one
    at a time, then remove the weakest unprotected row if the allowed count
    difference is exceeded. The tolerance is relative to final births, not N.
    Weak newborns can lose
    either competition. Equal alpha favors keeping old rows, avoiding a
    replacement with no alpha benefit. Preserve row order for serialization.
    """
    old_alpha = torch.as_tensor(old_alpha).detach().flatten()
    birth_alpha = torch.as_tensor(birth_alpha, device=old_alpha.device).detach().flatten()
    sources = torch.as_tensor(source_rows, device=old_alpha.device)
    if (isinstance(count_tolerance, bool) or not isinstance(count_tolerance, (int, float)) or
            not math.isfinite(count_tolerance) or not 0 <= count_tolerance < 1):
        raise ValueError('Replacement count tolerance must be in [0, 1)')
    count, candidates = len(old_alpha), len(birth_alpha)
    if (count < 1 or sources.shape != (candidates,) or
            sources.dtype not in (torch.int32, torch.int64) or
            bool(((sources < 0) | (sources >= count)).any()) or
            not bool(torch.isfinite(old_alpha).all()) or
            not bool(torch.isfinite(birth_alpha).all()) or
            bool(((old_alpha < 0) | (old_alpha > 1)).any()) or
            bool(((birth_alpha < 0) | (birth_alpha > 1)).any())):
        raise ValueError('Balanced replacement requires finite alpha and aligned birth sources')
    alpha = torch.cat((old_alpha, birth_alpha))
    keep = torch.zeros(count + candidates, dtype=torch.bool, device=old_alpha.device)
    keep[torch.argsort(alpha, descending=True, stable=True)[:count]] = True
    protected = torch.zeros_like(keep)
    while True:
        selected_sources = sources[keep[count:]].long()
        present = keep[selected_sources]
        protected[selected_sources[present]] = True
        missing = keep[count:] & ~keep[sources.long()]
        if not bool(missing.any()):
            break
        # Handle the strongest dependent birth first. Restoring all parents
        # together could unnecessarily reject every birth in a crowded pool.
        strongest = birth_alpha.masked_fill(~missing, -torch.inf).argmax()
        source = sources[strongest].long()
        keep[source] = protected[source] = True
        # Source protection may keep a few extra old rows. Recompute the
        # integer allowance if a newborn loses this competition as well.
        while (int(keep.sum()) - count >
               math.floor(count_tolerance * int(keep[count:].sum()) + 1e-9)):
            # Reverse pool order so equal-alpha newborns lose before old rows.
            eligible_alpha = alpha.masked_fill(~keep | protected, torch.inf).flip(0)
            weakest = len(alpha) - 1 - eligible_alpha.argmin()
            keep[weakest] = False
    survivors = keep[:count].nonzero(as_tuple=True)[0]
    selected = keep[count:].nonzero(as_tuple=True)[0]
    retired = count - len(survivors)
    allowance = math.floor(count_tolerance * len(selected) + 1e-9)
    if not len(survivors) or not 0 <= len(selected) - retired <= allowance:
        raise AssertionError('Balanced replacement did not preserve a nonempty source pool')
    return survivors, selected


@torch.no_grad()
def birth_driven_retirement_rows(old_alpha, source_rows, *, count_tolerance, opacity_threshold):
    """Retain admitted births and clamp threshold-based old retirement to K±tK.

    If too few old rows fall below the alpha threshold, also retire the next
    weakest rows. If too many qualify, retire only the weakest within the band.
    Birth predictor sources remain present for exact decoding.
    """
    old_alpha = torch.as_tensor(old_alpha).detach().flatten()
    sources = torch.as_tensor(source_rows, device=old_alpha.device)
    count, births = len(old_alpha), sources.numel()
    if (isinstance(count_tolerance, bool) or not isinstance(count_tolerance, (int, float)) or
            not math.isfinite(count_tolerance) or not 0 <= count_tolerance < 1 or
            isinstance(opacity_threshold, bool) or not isinstance(opacity_threshold, (int, float)) or
            not math.isfinite(opacity_threshold) or not 0 <= opacity_threshold < 1 or
            count < 1 or sources.ndim != 1 or sources.dtype not in (torch.int32, torch.int64) or
            bool(((sources < 0) | (sources >= count)).any()) or
            not bool(torch.isfinite(old_alpha).all()) or
            bool(((old_alpha < 0) | (old_alpha > 1)).any())):
        raise ValueError('Birth-driven retirement requires valid alpha, sources and count bounds')
    protected = torch.zeros(count, device=old_alpha.device, dtype=torch.bool)
    protected[sources.long()] = True
    eligible = (~protected).nonzero(as_tuple=True)[0]
    difference = math.floor(count_tolerance * births + 1e-9)
    lower, upper = births - difference, births + difference
    capacity = min(len(eligible), count - 1)
    if lower > capacity:
        raise ValueError(f'Cannot retire at least {lower} old GS for {births} admitted births: '
                         f'only {capacity} old rows can retire while preserving birth sources. '
                         'No admitted births have been discarded.')
    natural = int(old_alpha[eligible].lt(opacity_threshold).sum())
    retired = min(max(natural, lower), upper, capacity)
    ranked = eligible[torch.argsort(old_alpha[eligible], stable=True)]
    remove = torch.zeros_like(protected)
    remove[ranked[:retired]] = True
    rows = (~remove).nonzero(as_tuple=True)[0]
    return rows, dict(natural_retirements=natural, min_retirements=lower,
        max_retirements=upper, eligible_old_count=len(eligible),
        opacity_threshold=opacity_threshold)


def restore_retired_reference(raw, state, previous_rows, retired):
    """Reconstruct the preceding raw reference when resuming after retirement."""
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


class Module(BaseModule):
    """Old GS learn motion; newborn GS fit quantized geometry and appearance."""

    def setup(self):
        cfg = self.config
        validate_initial_opacity_threshold(cfg.initial_opacity_threshold)
        cfg.birth.validate(cfg.rigid_tree.motion_steps)
        cfg.opacity_update.validate()
        if cfg.birth.warmup_steps >= cfg.rigid_tree.birth_steps - cfg.birth.refit_steps:
            raise ValueError('Birth warmup must leave candidate-selection steps')
        if cfg.birth.refit_steps:
            if (cfg.birth.refit_steps >= cfg.rigid_tree.birth_steps or
                    cfg.birth.replacement_policy not in CURRENT_BIRTH_POLICIES or
                    not cfg.rigid_tree.amplitude_refit_steps):
                raise ValueError('Birth refitting requires current-birth retirement, fixed motion support, and selection steps')
        if cfg.opacity_update.amplitude_refit_steps >= cfg.rigid_tree.motion_steps:
            raise ValueError('Opacity amplitude refitting must leave selection steps')
        if cfg.opacity_update.enabled and cfg.birth.replace_low_opacity:
            raise ValueError('Current-frame alpha pruning requires birth.replace_low_opacity=false')
        if cfg.birth.strategy == 'multiview_residual' and cfg.opacity_update.joint_with_birth:
            raise ValueError('Multi-view birth validation requires frozen old opacity')
        if cfg.birth.strategy == 'multiview_residual' and cfg.sh_degree != 1:
            raise ValueError('Multi-view appearance initialization currently requires SH degree 1')
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
        self.births = BirthGaussianModel(cfg.sh_degree)
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
        self._birth_views_cache = None
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
    def birth_selection_end_step(self):
        return self.total_steps - self.config.birth.refit_steps

    @property
    def birth_warmup_end_step(self):
        return self.motion_steps + self.config.birth.warmup_steps

    @property
    def birth_refit_active(self):
        return self.birth_control.get('refit_active', False) and not self.frame_finalized

    @property
    def birth_allocation(self):
        return ('multiview_residual_v1' if self.config.birth.strategy == 'multiview_residual'
                else BIRTH_ALLOCATION)

    def _freeze_base(self):
        for attr in RAW_ATTRIBUTES:
            getattr(self.gaussians, attr).requires_grad_(False)

    def _birth_warmup_active(self, step):
        return self.config.birth.warmup_steps > 0 and step <= (
            self.motion_steps + self.config.birth.warmup_steps)

    def _birth_joint_opacity_active(self, step):
        return (getattr(self, 'opacity_update', None) is not None and
                self.motion_committed and not self.frame_finalized and
                self.config.opacity_update.joint_with_birth and
                not self.birth_control.get('replacement_committed', False) and
                step > self.motion_steps + self.config.birth.warmup_steps and
                step <= getattr(self, 'birth_selection_end_step', self.total_steps))

    def _configure_birth_phase(self, step):
        """Derive trainability from the step so boundary checkpoints resume safely."""
        warmup = self._birth_warmup_active(step)
        if self.births.optimizer is not None:
            for name in ('_opacity', '_scaling'):
                parameter = getattr(self.births, name)
                parameter.requires_grad_(not warmup)
                if warmup:
                    parameter.grad = None
            self.births._rotation.requires_grad_(self.config.birth.rotation_updates)
            phase_step = max(0, step-self.motion_steps-1)
            ratio = self.config.birth.lr_final_ratio ** (phase_step / max(1, self.total_steps-self.motion_steps-1))
            for group in self.births.optimizer.param_groups:
                # Restored optimizer rates may already have decayed.
                rates = self.config.gaussian_stage2
                base = dict(xyz=rates.position_lr_init*self.births.spatial_lr_scale,
                    f_dc=rates.feature_lr, f_rest=rates.feature_lr/20,
                    opacity=rates.opacity_lr, scaling=rates.scaling_lr, rotation=rates.rotation_lr)
                group['lr'] = base[group['name']] * ratio
        if self.opacity_update is not None:
            active = self._birth_joint_opacity_active(step)
            self.opacity_update.delta.requires_grad_(active)
            if not active:
                self.opacity_update.delta.grad = None

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
                      ('xyz_grad_norm', 'color_grad_norm', 'old_opacity_grad_norm'))
        values = torch.stack(values).tolist()
        row = dict(step=step, new_count=count, alpha_mean=values[0], alpha_median=values[1],
            below_opacity_threshold=int(values[2]), xyz_grad_norm=values[3],
            color_grad_norm=values[4], old_opacity_grad_norm=values[5],
            warmup=self._birth_warmup_active(step),
            new_opacity_trainable=self.births._opacity.requires_grad,
            new_scale_trainable=self.births._scaling.requires_grad,
            old_opacity_trainable=(self.opacity_update.delta.requires_grad
                                  if self.opacity_update is not None else False))
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
                ckpt.get('birth_allocation', 'clone_small_v1') != getattr(self, 'birth_allocation', BIRTH_ALLOCATION)):
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
        previous_rows = None
        if self.config.birth.replace_low_opacity and self.current_frame > 1:
            prior = self.motion.codec_previous_state
            sources = self.motion.pending_birth_sources
            if (prior is None or prior.frame_index != self.current_frame - 1 or
                    len(prior.xyz) != len(self.gaussians._xyz) or sources is None):
                raise ValueError('GS replacement requires the finalized previous frame and birth lineage')
            previous_rows, self.retired_reference, retirement = replace_low_opacity_gaussians(
                self.gaussians, len(sources))
        xyz, rotation = self.gaussians._xyz.detach(), self.gaussians._rotation.detach()
        if previous_rows is None and self.current_frame > 1:
            previous_rows = torch.arange(len(xyz), device=xyz.device)
        self.motion.setup(xyz, rotation, self.datamodule.cameras_extent,
                          frame_index=self.current_frame, previous_rows=previous_rows)
        self.frame_prepared = True
        self.frame_finalized = False
        self.motion_committed = False
        self.frame_report = None
        self.birth_control = dict(clone_added=0, split_added=0, pruned=0,
                                  source_tracked=True, retirement=retirement,
                                  previous_birth_count=previous_birth_count)
        self.stage_image_metrics = []
        self._freeze_base()
        if self.opacity_update is not None:
            self.opacity_update.setup(self.gaussians._opacity)

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
        self._birth_views_cache = None


    def make_birth_view(self):
        if not len(self.births._xyz):
            return None
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
            elif self._birth_joint_opacity_active(self.trainer.global_step):
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
        image_loss = (birth_image_loss(image, viewpoint.image,
                          self.config.birth.fit_ssim_weight if self.config.birth.fit_loss == 'mse_ssim' else 0.)
                      if self.motion_committed and self.config.birth.fit_loss in ('mse', 'mse_ssim')
                      else motion_photometric_loss(image, viewpoint.image,
                                                  self.config.lambda_dssim, factor))
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
        if (regularization_weight and getattr(self, 'opacity_update', None) is not None and
                (not self.motion_committed or
                 self._birth_joint_opacity_active(step))):
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
        """Commit fitted alpha; defer count-coupled retirement until births fit."""
        update = self.opacity_update
        logits = update.canonical_logits(full=True).to(self.device)
        if getattr(getattr(self.config, 'birth', None), 'replacement_policy', 'none') in CURRENT_BIRTH_POLICIES:
            # Final surviving births determine the quota. Committing alpha at
            # the phase boundary must not independently shrink the old pool.
            rows = torch.arange(len(logits), device=self.device)
            update.commit(rows)
            self.gaussians._opacity.copy_(logits)
            return rows
        remove = logits.sigmoid().flatten() < self.config.opacity_update.prune_threshold
        prevented_empty = bool(remove.all())
        if prevented_empty:
            remove[logits.flatten().argmax()] = False
        rows = (~remove).nonzero(as_tuple=True)[0]
        count, removed = len(remove), int(remove.sum())
        previous_births = self.birth_control.get('previous_birth_count', 0)
        self.birth_control['retirement'] = dict(
            rule='current_quantized_alpha', threshold=self.config.opacity_update.prune_threshold,
            previous_count=count, retired_count=removed, surviving_count=len(rows),
            retired_older_count=int(remove[:count - previous_births].sum()),
            retired_previous_birth_count=int(remove[count - previous_births:].sum()),
            prevented_empty=prevented_empty)
        if removed:
            self.retired_reference = dict(rows=remove.nonzero(as_tuple=True)[0].cpu(),
                attributes={name: getattr(self.gaussians, attr).detach()[remove].cpu().clone()
                            for name, attr in zip(RAW_FIELDS[2:], RAW_ATTRIBUTES[2:])})
            self.gaussians.prune_points(remove)
            self.motion.retain_survivors(rows)
        update.commit(rows)
        self.gaussians._opacity.copy_(logits[rows])
        return rows

    def _birth_camera_views(self):
        """Cache only training cameras; held-out evaluation cannot seed births."""
        cached = getattr(self, '_birth_views_cache', None)
        if cached is not None and cached[0] == self.current_frame:
            return cached[1:]
        validation = set(self.config.birth.validation_cameras)
        dataset = self.datamodule.get_train_dataset()
        names = {Path(item.path).name for item in dataset.items}
        if not validation.issubset(names) or validation == names:
            raise ValueError('Birth validation cameras must be a proper subset of training cameras')
        fit, check = [], []
        for index in sorted(range(len(dataset)), key=lambda i: Path(dataset.items[i].path).name):
            view = dataset[index].to(self.device)
            (check if Path(view.path).name in validation else fit).append(view)
        if len(fit) < self.config.birth.multiview.min_support_views:
            raise ValueError('Too few fitting cameras for multi-view births')
        self._birth_views_cache = (self.current_frame, fit, check)
        return fit, check

    def _raw_gaussian_model(self, raw):
        result = GaussianModel(self.config.sh_degree)
        result.active_sh_degree = self.gaussians.active_sh_degree
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(result, attr, torch.nn.Parameter(raw[name].detach(), requires_grad=False))
        return result

    @torch.no_grad()
    def _birth_layout_metrics(self, raw, rows, newborn, cameras):
        model = self._raw_gaussian_model({name: value[rows] for name, value in raw.items()})
        if newborn is not None:
            model = MergedGaussianModel([model, newborn])
        renderer = Renderer(self.config.renderer, model)
        metrics = dict(psnr=0., ssim=0., mse=0.)
        for camera in cameras:
            image = renderer(camera, self.background)['render'].clamp(0, 1)
            metrics['psnr'] += float(psnr(image, camera.image)) / len(cameras)
            metrics['ssim'] += float(ssim(image, camera.image)) / len(cameras)
            metrics['mse'] += float((image-camera.image).square().mean()) / len(cameras)
        return metrics

    @torch.no_grad()
    def _multiview_retirement_rows(self, view):
        count = len(self.gaussians._xyz)
        all_rows = torch.arange(count, device=self.device)
        births = len(self.births._xyz)
        if not births:
            return all_rows, all_rows[:0], dict(reason='empty_birth_pool')
        fit, check = self._birth_camera_views()
        raw = model_raw(self.gaussians, clone=False)
        baseline = self._birth_layout_metrics(raw, all_rows, None, check)
        current_renderer = Renderer(self.config.renderer, self._raw_gaussian_model(raw))
        pools = [torch.arange(births, device=self.device)]
        newborn_raw = None
        utility_details = {}
        if self.config.birth.admission_useful_fractions:
            newborn_raw = {name: view.raw_value(name).detach() for name in RAW_FIELDS}
            full_alpha = newborn_raw['opacity'].sigmoid()
            half_model = self._raw_gaussian_model(newborn_raw)
            half_model._opacity = torch.nn.Parameter(
                torch.logit((full_alpha*.5).clamp(1e-6, 1-1e-6)))
            utility_renderer = Renderer(self.config.renderer,
                MergedGaussianModel([self._raw_gaussian_model(raw), half_model]))
            baseline_errors = [float((current_renderer(camera, self.background)['render'].clamp(0, 1)
                                      - camera.image).square().mean()) for camera in check]
            utility = midpoint_image_utility(utility_renderer, check, self.background,
                half_model._opacity, full_alpha, baseline_errors,
                self.config.birth.fit_ssim_weight)
            pools = useful_birth_subsets(utility, self.config.birth.admission_useful_fractions)
            utility_details = dict(positive_count=int((utility > 0).sum()),
                nonpositive_count=int((utility <= 0).sum()),
                pool_counts=[len(pool) for pool in pools],
                estimate='half_alpha_image_derivative', ssim_weight=self.config.birth.fit_ssim_weight)
        current = compositing_contributions(current_renderer, fit, self.background)
        previous = dict(raw)
        reference = self.motion.frame_reference_state
        previous['xyz'] = reference.xyz.to(self.device)
        previous['rotation'] = reference.rotation.to(self.device)
        if self.opacity_update is not None:
            previous['opacity'] = self.opacity_update.reference
        # Color/scale are fixed, so the preceding raw state is fully available.
        previous_frame = self.current_frame
        previous_views = []
        try:
            self.datamodule.current_frame = previous_frame - 1
            dataset = self.datamodule.get_train_dataset()
            fit_names = {Path(camera.path).name for camera in fit}
            for index in sorted(range(len(dataset)), key=lambda i: Path(dataset.items[i].path).name):
                if Path(dataset.items[index].path).name in fit_names:
                    previous_views.append(dataset[index].to(self.device))
        finally:
            self.datamodule.current_frame = previous_frame
        if not previous_views:
            raise ValueError('No preceding fitting cameras for retirement history')
        prior_renderer = Renderer(self.config.renderer, self._raw_gaussian_model(previous))
        prior = compositing_contributions(prior_renderer, previous_views, self.background)
        recent = (self.birth_control.get('previous_birth_count', 0)
                  if self.config.birth.protect_previous_births else 0)
        trials, layouts, counts, selections, protections = [], [], [], [], []
        for selected in pools:
            ranked, protected = rank_retirement_candidates(current, prior,
                self.births.source_rows[selected], previous_birth_count=recent)
            selected_view = (view if len(selected) == births else self._raw_gaussian_model(
                {name: value[selected] for name, value in newborn_raw.items()}))
            for deleted in retirement_count_trials(len(selected), self.config.birth.replacement_count_tolerance):
                if deleted > len(ranked):
                    continue
                keep = torch.ones(count, device=self.device, dtype=torch.bool)
                keep[ranked[:deleted]] = False
                rows = keep.nonzero(as_tuple=True)[0]
                layouts.append(rows)
                counts.append(deleted)
                selections.append(selected)
                protections.append(int(protected.sum()))
                trials.append(self._birth_layout_metrics(raw, rows, selected_view, check))
        choice = choose_validation_trial(baseline, trials,
            self.config.birth.validation_min_gain_db, self.config.birth.validation_max_ssim_drop)
        details = dict(contribution_rule='two_frame_compositing_max', baseline=baseline,
            trial_retirements=counts, trial_metrics=trials, accepted=choice is not None,
            trial_birth_counts=[len(selected) for selected in selections],
            newborn_utility=utility_details,
            protected_count=protections[choice if choice is not None else 0] if protections else 0,
            protected_recent_births=recent,
            min_gain_db=self.config.birth.validation_min_gain_db,
            max_ssim_drop=self.config.birth.validation_max_ssim_drop,
            validation_cameras=[Path(camera.path).name for camera in check])
        self.birth_control['birth_admission'] = details
        if choice is None:
            return all_rows, all_rows[:0], details
        details['selected_trial'] = choice
        return layouts[choice], selections[choice], details

    @torch.no_grad()
    def commit_birth_replacement(self):
        """Commit the final birth pool once; later steps fit this fixed row set.

        This boundary is independent of the final training step. It is called
        at finalization by default, and can precede a later fixed-set refit.
        """
        if self.config.birth.replacement_policy not in CURRENT_BIRTH_POLICIES:
            return
        control = self.birth_control
        if control.get('replacement_committed', False):
            retired = control['retirement']
            if (len(self.births._xyz) != retired['birth_count'] or
                    len(self.gaussians._xyz) + len(self.births._xyz) !=
                    retired['previous_count'] + retired['birth_count'] - retired['retired_count']):
                raise ValueError('Committed GS replacement requires a fixed birth row set')
            return
        if not self.motion_committed or self.frame_finalized:
            raise ValueError('GS replacement requires fitted motion and an unfinalized birth pool')
        count = len(self.gaussians._xyz)
        view = self.make_birth_view()
        sources = (self.births.source_rows if view is not None else
                   torch.empty(0, device=self.device, dtype=torch.long))
        birth_alpha = (view.get_opacity if view is not None else
                       self.gaussians._opacity.new_empty(0))
        retirement_details = {}
        if self.config.birth.replacement_policy == 'multiview_gain_retirement':
            rows, selected_births, retirement_details = self._multiview_retirement_rows(view)
        elif self.config.birth.replacement_policy == 'birth_driven_retirement':
            rows, retirement_details = birth_driven_retirement_rows(
                self.gaussians._opacity.sigmoid(), sources,
                count_tolerance=self.config.birth.replacement_count_tolerance,
                opacity_threshold=self.config.opacity_update.prune_threshold)
            selected_births = torch.arange(len(self.births._xyz), device=self.device)
        else:
            rows, selected_births = balanced_replacement_rows(
                self.gaussians._opacity.sigmoid(), birth_alpha, sources,
                count_tolerance=self.config.birth.replacement_count_tolerance)
        candidates = len(self.births._xyz)
        rejected = candidates - len(selected_births)
        if rejected:
            discard = torch.ones(candidates, device=self.device, dtype=torch.bool)
            discard[selected_births] = False
            self.births.prune_points(discard)
            control['pruned'] = control.get('pruned', 0) + rejected
        remove = torch.ones(count, device=self.device, dtype=torch.bool)
        remove[rows] = False
        removed = int(remove.sum())
        previous_births = control.get('previous_birth_count', 0)
        control['retirement'] = dict(
            rule=self.config.birth.replacement_policy, previous_count=count,
            candidate_count=candidates, birth_count=len(selected_births),
            rejected_birth_count=rejected, retired_count=removed,
            count_tolerance=self.config.birth.replacement_count_tolerance,
            count_difference=len(selected_births) - removed,
            allowed_count_difference=math.floor(
                self.config.birth.replacement_count_tolerance * len(selected_births) + 1e-9),
            retired_older_count=int(remove[:count - previous_births].sum()),
            retired_previous_birth_count=int(remove[count - previous_births:].sum()),
            surviving_count=len(rows),
            protected_source_count=int(self.births.source_rows.unique().numel()) if removed else 0,
            **retirement_details)
        if removed:
            self.retired_reference = dict(rows=remove.nonzero(as_tuple=True)[0].cpu(),
                attributes={name: getattr(self.gaussians, attr).detach()[remove].cpu().clone()
                            for name, attr in zip(RAW_FIELDS[2:], RAW_ATTRIBUTES[2:])})
            remap = torch.full((count,), -1, dtype=torch.long, device=self.device)
            remap[rows] = torch.arange(len(rows), device=self.device)
            self.births.source_rows = remap[self.births.source_rows]
            if bool(self.births.source_rows.lt(0).any()):
                raise AssertionError('GS replacement retired a required birth predictor')
            self.gaussians.prune_points(remove)
            self.motion.retain_survivors(rows)
            if self.opacity_update is not None:
                self.opacity_update.commit(self.opacity_update.rows[rows])
        control['replacement_committed'] = True
        if self.opacity_update is not None:
            self.opacity_update.finish()

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
        if not self.config.birth.refit_motion:
            self.birth_control['fixed_pool_refit'] = True
            self.birth_control['refit_active'] = False
            return
        self._set_birth_refit_trainable()
        # The new point set changes compositing. Discard the stage-one Adam
        # history once; a resumed refit instead restores its own saved moments.
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
            self.motion.optimizer.state.pop(getattr(self.motion, name), None)
        self.birth_control['refit_active'] = True

    @torch.no_grad()
    def _begin_births(self):
        xyz, rotation = self.motion(self.motion_steps, training=False)
        self.gaussians.apply_selected_motion(xyz, rotation)
        self.motion_committed = True
        for parameter in self.motion.parameters():
            parameter.requires_grad_(False)
        self.motion.optimizer.zero_grad(set_to_none=True)
        if self.config.birth.strategy == 'multiview_residual':
            fit, _ = self._birth_camera_views()
            raw, sources, report = propose_multiview_births(
                self.renderer, fit, model_raw(self.gaussians, clone=False), self.background,
                self.datamodule.cameras_extent, self.config.birth.multiview)
            self.births = BirthGaussianModel(self.config.sh_degree)
            self.births.active_sh_degree = self.gaussians.active_sh_degree
            self.births.spatial_lr_scale = self.gaussians.spatial_lr_scale
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                setattr(self.births, attr, torch.nn.Parameter(raw[name]))
            self.births.source_rows = sources
            self.births.max_radii2D = raw['xyz'].new_zeros(len(sources))
            self.births.initialize_incremental_optimizer(self.config.gaussian_stage2)
            self.birth_control['multiview_added'] = len(sources)
            self.birth_control['multiview_proposal'] = report
            self._freeze_base()
            self._configure_birth_phase(self.motion_steps + 1)
            return
        count = len(self.gaussians._xyz)
        clone, source_rows = self.gaussians.adaptive_densify_and_clone(
            self.config.birth.densify_grad_threshold,
            self.datamodule.cameras_extent,
            sigma_scale=self.config.birth.clone_sigma_scale,
            max_new_gaussians=self.config.birth.max_new_gaussians,
            source_covariance_init=self.config.birth.source_covariance_init,
            return_source_rows=True)
        if clone['added'] != len(source_rows):
            raise AssertionError('Initial birth source rows disagree with clones')
        self.birth_control['clone_added'] = clone['added']
        if clone['added']:
            self.births.restore(self.gaussians.capture(),
                                self.config.gaussian_stage2)
            new_mask = torch.arange(len(self.gaussians._xyz),
                                    device=self.device) >= count
            self.gaussians.prune_points(new_mask)
            self.births.prune_points(~new_mask)
            self.births.source_rows = source_rows
            if self.config.birth.initial_opacity is not None:
                alpha = self.config.birth.initial_opacity
                self.births._opacity.fill_(math.log(alpha / (1 - alpha)))
            self.births.initialize_incremental_optimizer(
                self.config.gaussian_stage2)
        self._freeze_base()
        self._configure_birth_phase(self.motion_steps + 1)
        if self.config.birth.warmup_steps:
            self._record_birth_learning(self.motion_steps)


    @torch.no_grad()
    def _prune_terminal_births(self):
        """Remove weak births without adding unfitted GS on the final step."""
        model = self.births
        cfg = self.config.birth
        # Admission, rendering and export must judge the same quantized alpha.
        low_opacity = self.make_birth_view().get_opacity.flatten() < cfg.opacity_threshold
        big_screen = model.max_radii2D > cfg.new_max_screen_size
        big_world = (model.get_scaling.max(dim=1).values >
                     0.1 * self.datamodule.cameras_extent)
        remove = low_opacity | big_screen | big_world
        reasons = self.birth_control.setdefault('prune_reasons', {})
        for name, mask in (('low_opacity', low_opacity), ('screen_size', big_screen),
                           ('world_size', big_world),
                           ('multiple', (low_opacity.int() + big_screen.int() + big_world.int()) > 1)):
            reasons[name] = reasons.get(name, 0) + int(mask.sum())
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
            start = self.motion_steps - self.config.birth.densification_stats_steps + 1
            if step == start:
                self.gaussians.xyz_gradient_accum.zero_()
                self.gaussians.denom.zero_()
            if step >= start:
                for item in packages:
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
            selection_end = getattr(self, 'birth_selection_end_step', self.total_steps)
            selection_terminal = step == selection_end
            fixed_births = self.birth_control.get('replacement_committed', False)
            warmup = Module._birth_warmup_active(self, step)
            trace = (self.config.birth.warmup_steps > 0 and birth_step in
                     {5, 10, self.config.birth.warmup_steps,
                      self.config.birth.warmup_steps + 1,
                      self.config.birth.densify_interval,
                      selection_end - self.motion_steps, self.total_steps - self.motion_steps})
            gradients = {}
            if trace:
                parameters = dict(xyz_grad_norm=self.births._xyz,
                                  color_grad_norm=self.births._features_dc)
                if getattr(self, 'opacity_update', None) is not None:
                    parameters['old_opacity_grad_norm'] = self.opacity_update.delta
                gradients = {name: parameter.grad.detach().norm()
                             for name, parameter in parameters.items() if parameter.grad is not None}
            if len(self.births._xyz):
                old_count = len(self.gaussians._xyz)
                for item in packages:
                    visible = item['visibility_filter'][old_count:]
                    radii = item['radii'][old_count:]
                    self.births.max_radii2D[visible] = torch.maximum(
                        self.births.max_radii2D[visible], radii[visible])
                    if self.config.birth.strategy == 'gradient_clone':
                        if len(packages) > 1:
                            item['viewspace_points'].grad.mul_(len(packages))
                        self.births.add_densification_stats(
                            item['viewspace_points'], visible, align='right')
                terminal = step == self.total_steps
                if (self.config.birth.strategy == 'gradient_clone' and
                        not terminal and not selection_terminal and not fixed_births and not warmup and
                        birth_step % self.config.birth.densify_interval == 0):
                    result = self.births.adaptive_densify_and_prune(
                        self.config.birth.densify_grad_threshold,
                        self.config.birth.opacity_threshold,
                        self.datamodule.cameras_extent,
                        self.config.birth.new_max_screen_size,
                        max_points=self.config.birth.max_new_gaussians)
                    self.birth_control['split_added'] += result['added']
                    self.birth_control['pruned'] += result['pruned']
                    reasons = self.birth_control.setdefault('prune_reasons', {})
                    for name, count in result.get('prune_reasons', {}).items():
                        reasons[name] = reasons.get(name, 0) + count
                self.births.optimizer.step()
                self.births.optimizer.zero_grad(set_to_none=True)
            if getattr(self, 'birth_refit_active', False):
                self.motion.optimizer_step()
                xyz, rotation = self.motion(step, training=False)
                self.gaussians.apply_selected_motion(xyz, rotation)
            if Module._birth_joint_opacity_active(self, step):
                window = self.config.rigid_tree.motion_fit_window_steps
                self.opacity_update.optimizer_step(self._rate_byte_price(step),
                    proximal=(window is None or birth_step % window == 0 or selection_terminal),
                    lr_factor=(self.motion._motion_lr_factor(self.motion_steps)
                        if self.config.opacity_update.decay_with_motion else 1.))
                # Keep source predictors, export and rendering on the same
                # canonical alpha grid while newborn attributes keep fitting.
                self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
            if selection_terminal and not fixed_births and len(self.births._xyz):
                # Judge admission after BOTH optimizers have updated: changing
                # old source alpha also changes the newborn prediction grid.
                self.birth_control['pruned'] += self._prune_terminal_births()
            if (step == getattr(self, 'birth_selection_end_step', self.total_steps) and
                    self.config.birth.replacement_policy in CURRENT_BIRTH_POLICIES):
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
        view = self.make_birth_view()
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
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self.gaussians, attr,
                    torch.nn.Parameter(raw[name].detach().clone()))
        self.gaussians.training_setup(self.config.gaussian)
        self.gaussians.max_radii2D = self.gaussians._xyz.new_zeros(len(raw['xyz']))
        self._freeze_base()
        self.renderer.gaussians = self.gaussians
        self.births = BirthGaussianModel(self.config.sh_degree)
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
                birth_warmup_steps=self.config.birth.warmup_steps,
                birth_selection_steps=self.birth_selection_end_step - self.motion_steps,
                refit_steps=self.config.birth.refit_steps,
                birth_selection_end_step=self.birth_selection_end_step),
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
