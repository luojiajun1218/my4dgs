"""Spatial gate motion, GS residuals, and quantized newborn Gaussians."""

from dataclasses import asdict, dataclass, field
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
    reference_digest, family_quantization_diagnostics, patch_old_state,LiveReplacementView)
from .model.rigid_residual_tree import (RigidTreeConfig,
    quaternion_angle_error, quaternion_rotate,
    saved_config_with_default_motion_lr)
from .model.contribution_renderer import ContributionRenderer
from .model.spatial_gated_motion import SpatialGatedMotion
from .model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from .model.sparse_features import FeatureDCUpdateConfig, QuantizedFeatureDCUpdate
from .model.residual_birth import (ResidualBirthConfig, BirthTransaction,
    SubsetView, propose as propose_residual_births, constrain_candidates,
    admit as admit_residual_births, fitting_errors, refine_transaction, RawView,
    selected_population, CommitTrial, encode_trial as encode_birth_trial,
    validation_errors as birth_validation_errors, qualify_crossfit_novel,nearest_sources)
from .early_replacement import EarlyReplacementController
from .frame_plan import FramePlanController, fitting_old_motion


BIRTH_ALLOCATION = 'clone_small_v1'
RESIDUAL_BIRTH_ALLOCATION = 'quick_source_residual_300_v12'
SHADOW_BIRTH_ALLOCATION = 'shadow_source_residual_300_v13'
PERSISTENT_BIRTH_ALLOCATION = 'persistent_anchored_bank_300_v14'
EARLY_BIRTH_ALLOCATION = 'early_source_replacement_300_v15'


def uses_residual_birth(config):
    return getattr(getattr(config, 'birth', None), 'mechanism', 'clone') == 'residual_transaction'


def uses_intrinsic_birth(config):
    return uses_residual_birth(config) and getattr(config.birth, 'radiance_mode', 'free') in (
        'inherited_intrinsic', 'conservative_detail')


def uses_conservative_birth(config):
    return uses_residual_birth(config) and getattr(config.birth, 'radiance_mode', 'free') == 'conservative_detail'


def uses_surface_patch(config):
    return uses_residual_birth(config) and getattr(config.birth, 'radiance_mode', 'free') == 'surface_patch'


def uses_reincarnation(config):
    return uses_residual_birth(config) and getattr(config.birth, 'radiance_mode', 'free') == 'support_reincarnation'


def uses_dual_predictive(config):
    return uses_residual_birth(config) and getattr(config.birth,'radiance_mode','free') in ('dual_predictive_reincarnation','source_residual')


def uses_source_residual(config):
    return uses_residual_birth(config) and getattr(config.birth,'radiance_mode','free') == 'source_residual'


def uses_quick_source(config):
    return uses_residual_birth(config) and getattr(config.birth,'radiance_mode','free') in ('quick_source_residual','shadow_source_residual')


def uses_shadow_source(config):
    return uses_residual_birth(config) and getattr(config.birth,'radiance_mode','free') == 'shadow_source_residual'


def uses_persistent_birth(config):
    return uses_residual_birth(config) and getattr(config.birth,'radiance_mode','free') == 'persistent_anchored_newborn'


def uses_early_replacement(config):
    return uses_residual_birth(config) and getattr(config.birth,'radiance_mode','free') in ('early_source_replacement','adaptive_residual_birth')


def birth_allocation(config):
    if getattr(config.birth,'frame_training_plan',False):
        return 'frame_training_plan_v1'
    return ('adaptive_residual_birth_300_v16' if getattr(config.birth,'radiance_mode','free')=='adaptive_residual_birth' else
            EARLY_BIRTH_ALLOCATION if uses_early_replacement(config) else
            PERSISTENT_BIRTH_ALLOCATION if uses_persistent_birth(config) else
            SHADOW_BIRTH_ALLOCATION if uses_shadow_source(config) else
            RESIDUAL_BIRTH_ALLOCATION if uses_residual_birth(config) else BIRTH_ALLOCATION)


def freeze_birth_radiance(model, config):
    """Preserve appearance parameter identity while geometry fits independently."""
    if uses_intrinsic_birth(config):
        for name in ('_features_dc', '_features_rest'):
            parameter = getattr(model, name)
            parameter.requires_grad_(False)
            parameter.grad = None
        if hasattr(model, 'family_contrast'):
            model.family_contrast.requires_grad_(False)
            model.family_contrast.grad = None


def set_family_training_stage(model, terminal):
    """Details receive a full probation cohort before mass can return to mother."""
    if hasattr(model, 'family_gate_logit'):
        model.family_gate_logit.requires_grad_(bool(terminal))
        model.family_gate_logit.grad = None


def birth_model_factory(sh_degree, capture=None, family_state=None, patch_state=None, reincarnation_state=None, predictive_model=False, source_residual=False, quick_source=False, shadow_source=False, persistent=False):
    if persistent or (isinstance(capture,dict) and 'persistent_newborn' in capture):
        from .model.persistent_newborn_model import PersistentNewbornModel
        return PersistentNewbornModel(sh_degree)
    if shadow_source or (isinstance(capture,dict) and 'shadow_source_residual' in capture):
        from .model.shadow_source_residual_model import ShadowSourceResidualModel
        return ShadowSourceResidualModel(sh_degree)
    if quick_source or (isinstance(capture,dict) and 'quick_source_residual' in capture):
        from .model.quick_source_residual_model import QuickSourceResidualModel
        return QuickSourceResidualModel(sh_degree)
    if source_residual or (isinstance(capture,dict) and 'source_residual' in capture):
        from .model.source_residual_model import SourceResidualModel
        return SourceResidualModel(sh_degree)
    if predictive_model or (isinstance(capture,dict) and 'predictive_reincarnation' in capture):
        from .model.predictive_reincarnation_model import PredictiveReincarnationModel
        return PredictiveReincarnationModel(sh_degree)
    if reincarnation_state is not None or (isinstance(capture, dict) and 'reincarnation' in capture):
        from .model.reincarnation_model import ReincarnationModel
        return ReincarnationModel(sh_degree)
    if patch_state is not None or (isinstance(capture, dict) and 'patch' in capture):
        from .model.surface_patch_model import SurfacePatchModel
        return SurfacePatchModel(sh_degree)
    if family_state is not None or isinstance(capture, dict):
        from .model.conservative_family import ConservativeFamilyModel
        return ConservativeFamilyModel(sh_degree)
    return BirthGaussianModel(sh_degree)


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
    mechanism: str = 'clone'
    radiance_mode: str = 'conservative_detail'
    residual: ResidualBirthConfig = field(default_factory=ResidualBirthConfig)
    replace_low_opacity: bool = False
    source_handoff: bool = False
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
    densify_until_step: Optional[int] = None
    lr_final_ratio: float = 1.0
    photometric_loss: Optional[str] = None
    source_position_limit_sigma: Optional[float] = None
    source_position_min_extent: float = 0.0
    initial_alpha: Optional[float] = None
    alpha_sparsity_weight: float = 0.0
    storage_loss_enabled: bool = False
    survival_aware_storage: bool = False
    storage_loss_strength: float = 1.0
    storage_warmup_steps: int = 20
    storage_ramp_steps: int = 70
    frame_training_plan: bool = False
    residual_support_init: bool = False
    joint_old_opacity: bool = False
    population_loss_weight: float = 0.0
    residual_colour_init: bool = False
    residual_colour_gain: float = 0.5
    residual_colour_max_rgb_delta: float = 0.05
    appearance_candidates: int = 0
    appearance_source_max_scale_extent: Optional[float] = None
    anchor_sfm_path: Optional[str] = None
    prune_large: bool = True

    def validate(self, motion_steps):
        if self.mechanism not in ('clone', 'residual_transaction'):
            raise ValueError('Unknown birth mechanism')
        if self.radiance_mode not in ('free','inherited_intrinsic','conservative_detail','surface_patch','support_reincarnation','dual_predictive_reincarnation','source_residual','quick_source_residual','shadow_source_residual','persistent_anchored_newborn','early_source_replacement','adaptive_residual_birth'):
            raise ValueError('Unknown newborn radiance mode')
        self.residual.validate()
        for name in ('storage_loss_enabled','survival_aware_storage','residual_support_init','residual_colour_init','joint_old_opacity'):
            if not isinstance(getattr(self,name),bool):raise ValueError(f'birth.{name} must be boolean')
        if self.survival_aware_storage and not (self.frame_training_plan and self.storage_loss_enabled):
            raise ValueError('Survival-aware storage requires the frame plan with newborn storage enabled')
        if self.residual_support_init and not self.frame_training_plan:
            raise ValueError('Residual-support initialization requires the frame training plan')
        for name in ('storage_loss_strength','residual_colour_gain','residual_colour_max_rgb_delta'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'birth.{name} must be finite and positive')
        if self.residual_colour_gain>1 or self.residual_colour_max_rgb_delta>1:
            raise ValueError('Birth colour gain and RGB delta must not exceed one')
        for name in ('storage_warmup_steps','storage_ramp_steps'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<(1 if name=='storage_ramp_steps' else 0):
                raise ValueError(f'birth.{name} must be a valid non-negative warmup / positive ramp')
        if (self.storage_loss_enabled or self.residual_colour_init) and self.radiance_mode!='adaptive_residual_birth':
            raise ValueError('Newborn storage and residual colour initialization require adaptive_residual_birth')
        if (isinstance(self.population_loss_weight,bool) or
                not isinstance(self.population_loss_weight,(int,float)) or
                not math.isfinite(self.population_loss_weight) or self.population_loss_weight<0):
            raise ValueError('birth.population_loss_weight must be finite and non-negative')
        if self.joint_old_opacity and self.radiance_mode!='adaptive_residual_birth':
            raise ValueError('Joint old opacity requires adaptive_residual_birth')
        if self.population_loss_weight and not self.joint_old_opacity:
            raise ValueError('Unified population pressure requires joint_old_opacity')
        if self.population_loss_weight and self.opacity_threshold<=0:
            raise ValueError('Unified population pressure requires a positive opacity threshold')
        if self.radiance_mode == 'support_reincarnation' and self.residual.max_cohorts != 2:
            raise ValueError('Independent geometry cross-fitting requires two complete exploration cohorts')
        if not isinstance(self.source_covariance_init, bool):
            raise ValueError('source_covariance_init must be boolean')
        if not isinstance(self.replace_low_opacity, bool):
            raise ValueError('birth.replace_low_opacity must be boolean')
        if not isinstance(self.source_handoff, bool):
            raise ValueError('birth.source_handoff must be boolean')
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
        if self.densify_until_step is not None and (
                isinstance(self.densify_until_step, bool) or
                not isinstance(self.densify_until_step, int) or
                self.densify_until_step < 0):
            raise ValueError('birth.densify_until_step must be a non-negative integer or null')
        if not math.isfinite(self.lr_final_ratio) or not 0 < self.lr_final_ratio <= 1:
            raise ValueError('birth.lr_final_ratio must be in (0, 1]')
        if self.photometric_loss not in (None, 'l1', 'rmse'):
            raise ValueError('birth.photometric_loss must be null, l1 or rmse')
        if self.source_position_limit_sigma is not None and (
                isinstance(self.source_position_limit_sigma, bool) or
                not isinstance(self.source_position_limit_sigma, (int, float)) or
                not math.isfinite(self.source_position_limit_sigma) or
                self.source_position_limit_sigma <= 0):
            raise ValueError('birth.source_position_limit_sigma must be positive or null')
        if (isinstance(self.source_position_min_extent, bool) or
                not isinstance(self.source_position_min_extent, (int, float)) or
                not math.isfinite(self.source_position_min_extent) or
                self.source_position_min_extent < 0):
            raise ValueError('birth.source_position_min_extent must be a finite non-negative number')
        if self.initial_alpha is not None and (
                isinstance(self.initial_alpha, bool) or
                not isinstance(self.initial_alpha, (int, float)) or
                not math.isfinite(self.initial_alpha) or
                not 0 < self.initial_alpha < 1):
            raise ValueError('birth.initial_alpha must be in (0, 1) or null')
        if (isinstance(self.alpha_sparsity_weight, bool) or
                not isinstance(self.alpha_sparsity_weight, (int, float)) or
                not math.isfinite(self.alpha_sparsity_weight) or
                self.alpha_sparsity_weight < 0):
            raise ValueError('birth.alpha_sparsity_weight must be non-negative')
        if (isinstance(self.appearance_candidates, bool) or
                not isinstance(self.appearance_candidates, int) or
                self.appearance_candidates < 0):
            raise ValueError('birth.appearance_candidates must be a non-negative integer')
        if self.appearance_source_max_scale_extent is not None and (
                isinstance(self.appearance_source_max_scale_extent, bool) or
                not isinstance(self.appearance_source_max_scale_extent, (int, float)) or
                not math.isfinite(self.appearance_source_max_scale_extent) or
                self.appearance_source_max_scale_extent <= 0):
            raise ValueError(
                'birth.appearance_source_max_scale_extent must be positive or null')
        if not isinstance(self.prune_large, bool):
            raise ValueError('birth.prune_large must be boolean')


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 1
    effective_motion_observations: bool = False
    motion_coarse_to_fine: bool = False
    initial_opacity_threshold: float = 0.0
    lambda_dssim: float = 0.2
    photometric_loss: str = 'l1'
    num_saving_images: int = 5
    lpips_required: bool = True
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    gaussian_stage2: GaussianConfig = field(default_factory=GaussianConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    rigid_tree: RigidTreeConfig = field(default_factory=RigidTreeConfig)
    birth: BirthConfig = field(default_factory=BirthConfig)
    opacity_update: OpacityUpdateConfig = field(default_factory=OpacityUpdateConfig)
    feature_dc_update: FeatureDCUpdateConfig = field(default_factory=FeatureDCUpdateConfig)


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


def motion_photometric_loss(image, target, lambda_dssim, factor=1, *, metric='l1'):
    if metric not in ('l1', 'rmse'):
        raise ValueError('Photometric pixel loss must be l1 or rmse')
    if factor > 1:
        size = tuple(max(1, value // factor) for value in image.shape[-2:])
        image = interpolate(image[None], size=size, mode='area')[0]
        target = interpolate(target[None], size=size, mode='area')[0]
    # Global RMS keeps pixel-scale loss units while giving small errors small
    # gradients. The offset makes an exact match have zero finite loss/gradient.
    if metric == 'l1':
        pixel = l1_loss(image, target)
    else:
        mse = (image - target).square().mean()
        # Rationalize sqrt(mse+eps)-sqrt(eps) to keep exact zero in float32.
        pixel = mse / ((mse + 1e-8).sqrt() + 1e-4)
    return ((1 - lambda_dssim) * pixel +
            lambda_dssim * (1 - ssim(image, target)))


def set_birth_learning_rates(model, args, step, total, ratio):
    factor = ratio ** min(1, max(0, (step - 1) / max(1, total - 1)))
    base = {
        'xyz': args.position_lr_init * model.spatial_lr_scale,
        'f_dc': args.feature_lr,
        'f_rest': args.feature_lr / 20.,
        'opacity': args.opacity_lr,
        'scaling': args.scaling_lr,
        'rotation': args.rotation_lr,
        'family_tangent_offset': args.position_lr_init * model.spatial_lr_scale,
        'family_gate_logit': args.opacity_lr,
        'patch_retention': args.opacity_lr,
        'p_common_dc': args.feature_lr,
        'p_common_sh1': args.feature_lr / 20.,
        'family_contrast': args.feature_lr,
    }
    for group in model.optimizer.param_groups:
        lr = base.get(group.get('name'))
        if lr is not None:
            group['lr'] = lr * factor


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


@torch.no_grad()
def constrain_birth_positions(model, reference, sigma, min_limit=0.0):
    """Project newborn xyz into the source oriented ellipsoid, world space only."""
    rows = model.source_rows
    if rows is None or not len(rows):
        return
    srcq = torch.nn.functional.normalize(reference._rotation[rows], dim=-1)
    inverseq = srcq.clone()
    inverseq[..., 1:] = -inverseq[..., 1:]
    reference_xyz = reference._xyz[rows]
    local = quaternion_rotate(inverseq, model._xyz - reference_xyz)
    limit = (reference._scaling[rows].exp() * sigma).clamp_min(min_limit)
    clamped = local.clamp(-limit, limit)
    model._xyz.copy_(reference_xyz + quaternion_rotate(srcq, clamped))


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


class Module(FramePlanController,EarlyReplacementController,BaseModule):
    """Old GS learn motion; newborn GS fit quantized geometry and appearance."""

    def setup(self):
        cfg = self.config
        validate_initial_opacity_threshold(cfg.initial_opacity_threshold)
        cfg.birth.validate(cfg.rigid_tree.motion_steps)
        cfg.opacity_update.validate()
        cfg.feature_dc_update.validate()
        if cfg.opacity_update.amplitude_refit_steps >= cfg.rigid_tree.motion_steps:
            raise ValueError('Opacity amplitude refitting must leave selection steps')
        if cfg.opacity_update.enabled and cfg.birth.replace_low_opacity:
            raise ValueError('Current-frame alpha pruning requires birth.replace_low_opacity=false')
        if uses_residual_birth(cfg) and (not cfg.opacity_update.enabled or
                cfg.opacity_update.joint_with_birth or cfg.birth.source_handoff or
                (not cfg.birth.rotation_updates and cfg.birth.radiance_mode != 'adaptive_residual_birth') or
                not cfg.effective_motion_observations):
            raise ValueError('Residual transactions require frozen old alpha, actual visibility, '
                             'new orientation fitting and no destructive source handoff')
        if cfg.feature_dc_update.enabled and uses_residual_birth(cfg) and cfg.birth.radiance_mode != 'adaptive_residual_birth':
            raise ValueError('Shared colors support ordinary clone and adaptive residual births')
        if cfg.birth.source_handoff and (not cfg.opacity_update.enabled or
                cfg.opacity_update.mode != 'alpha' or
                not cfg.opacity_update.joint_with_birth or
                cfg.birth.initial_alpha is not None):
            raise ValueError(
                'birth.source_handoff requires alpha joint opacity updates without initial_alpha')
        if not isinstance(cfg.effective_motion_observations, bool):
            raise ValueError('effective_motion_observations must be boolean')
        if not isinstance(cfg.motion_coarse_to_fine, bool):
            raise ValueError('motion_coarse_to_fine must be boolean')
        if not 0 <= cfg.lambda_dssim <= 1:
            raise ValueError('module.lambda_dssim must be in [0, 1]')
        if cfg.photometric_loss not in ('l1', 'rmse'):
            raise ValueError('module.photometric_loss must be l1 or rmse')
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
        self.feature_dc_update = (QuantizedFeatureDCUpdate(cfg.feature_dc_update, motion=self.motion)
                                  if cfg.feature_dc_update.enabled else None)
        self.frame_prepared = False
        self.frame_finalized = False
        self.motion_committed = False
        self.reference_count = None
        self.initial_reference = None
        self.frame_report = None
        self.birth_control = {}
        self.birth_appearance_gradient = None
        self.birth_handoff_sources = None
        self.birth_transaction = None
        self.birth_cohort = 0
        self.birth_cohort_start = 0
        self.birth_end_step = None
        self.coarse_birth_snapshot = None
        self.birth_terminal = False
        self.selected_birth_state = None
        self._transaction_views = None
        self.retired_reference = None
        self.stage_image_metrics = []
        self.persistent_bank_state = None
        self.persistent_uid_layout = {}
        self.persistent_specs = {}
        self.persistent_candidate_specs = {}
        self.persistent_pretrain_state = None
        self.persistent_birth_optimizer_steps = 0
        self.persistent_training_population_count = 0
        self.persistent_initialized = False
        self.early_initialized=False
        self.early_birth_updates=0
        self.early_shared_steps=0
        self._rate_budget_state=None
        self.early_selected=None
        self.early_pinned_rows=None
        self.early_pinned_values=None
        self.early_focus_weights={}
        self._freeze_base()

    @property
    def current_frame(self):
        return self.datamodule.current_frame

    @property
    def motion_steps(self):
        return self.config.rigid_tree.motion_steps_for_frame(self.current_frame)

    @property
    def total_steps(self):
        boundary = getattr(self, 'birth_end_step', None)
        return (boundary if uses_residual_birth(self.config) and boundary is not None else
                self.motion_steps + self.config.rigid_tree.birth_steps)

    def _freeze_base(self):
        for attr in RAW_ATTRIBUTES:
            getattr(self.gaussians, attr).requires_grad_(False)

    def on_save_checkpoint(self, ckpt):
        if getattr(self.config.birth,'frame_training_plan',False) and self.frame_prepared:
            self._plan_record_phase_time()
        ckpt['birth_allocation'] = birth_allocation(self.config)
        ckpt['motion_image_schedule'] = MOTION_IMAGE_SCHEDULE
        ckpt['rigid_tree_version'] = 1
        ckpt['rigid_tree_config'] = asdict(self.config.rigid_tree)
        ckpt['rate_budget_controller'] = BirthTransaction.tensor_tree(getattr(self,'_rate_budget_state',None),'cpu')
        ckpt['rigid_tree_module_config'] = asdict(self.config)
        ckpt['rigid_tree_initial_reference'] = self.initial_reference
        ckpt['gaussians'] = self.gaussians.capture()
        ckpt['rigid_tree_motion'] = self.motion.capture() if self.frame_prepared else None
        if uses_early_replacement(self.config) and ckpt['rigid_tree_motion'] is not None:
            payload=ckpt['rigid_tree_motion']['optimizer']
            groups=[group for group in payload['param_groups'] if not group.get('name','').startswith('early_')]
            ids={index for group in groups for index in group['params']}
            ckpt['rigid_tree_motion']['optimizer']=dict(param_groups=groups,
                state={key:value for key,value in payload['state'].items() if key in ids})
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
        feature_update = getattr(self, 'feature_dc_update', None)
        ckpt['quantized_feature_dc_update'] = feature_update.capture() if feature_update is not None else None
        ckpt['birth_handoff_sources'] = (self.birth_handoff_sources.detach().cpu()
            if self.birth_handoff_sources is not None else None)
        transaction = getattr(self, 'birth_transaction', None)
        ckpt['birth_transaction'] = transaction.capture() if transaction is not None else None
        ckpt['birth_cohort_state'] = dict(index=getattr(self, 'birth_cohort', 0),
            start_step=getattr(self, 'birth_cohort_start', getattr(self, 'motion_steps', 0)),
            end_step=getattr(self, 'birth_end_step', None),
            coarse_snapshot=getattr(self, 'coarse_birth_snapshot', None))
        ckpt['birth_terminal_state'] = dict(active=getattr(self, 'birth_terminal', False),
            selected=BirthTransaction.tensor_tree(getattr(self, 'selected_birth_state', None), 'cpu'))
        ckpt['birth_appearance_gradient'] = (
            self.birth_appearance_gradient.detach().cpu()
            if self.birth_appearance_gradient is not None else None)
        if uses_dual_predictive(self.config):
            ckpt['dual_birth_state'] = self._capture_dual_checkpoint()
        if uses_quick_source(self.config):
            ckpt['quick_birth_state'] = dict(phase=getattr(self,'quick_birth_phase',1),
                optimizer_steps=getattr(self,'quick_birth_optimizer_steps',0),
                initialized=getattr(self,'quick_birth_initialized',False))
            if uses_shadow_source(self.config):
                ckpt['shadow_initialized_state'] = BirthTransaction.tensor_tree(
                    getattr(self,'shadow_initialized_state',None),'cpu')
        if uses_persistent_birth(self.config):
            ckpt['persistent_newborn_bank'] = BirthTransaction.tensor_tree(dict(
                state=getattr(self,'persistent_bank_state',None),
                layout=getattr(self,'persistent_uid_layout',{}),specs=getattr(self,'persistent_specs',{}),
                candidate_specs=getattr(self,'persistent_candidate_specs',{}),
                pretrain=getattr(self,'persistent_pretrain_state',None)), 'cpu')
            ckpt['persistent_birth_state'] = dict(optimizer_steps=getattr(self,'persistent_birth_optimizer_steps',0),
                training_population_count=getattr(self,'persistent_training_population_count',0),
                initialized=getattr(self,'persistent_initialized',False))
        if uses_early_replacement(self.config):
            self._early_sync_opacity_state()
            ckpt['quantized_opacity_update']=self.opacity_update.capture()
            ckpt['births']=(self.births.capture() if self.early_initialized and not self.frame_finalized else None)
            ckpt['early_replacement_state']=BirthTransaction.tensor_tree(dict(
                initialized=self.early_initialized,birth_updates=self.early_birth_updates,
                shared_steps=self.early_shared_steps,selected=self.early_selected,
                pinned_rows=self.early_pinned_rows,pinned_values=self.early_pinned_values,
                focus_weights=self.early_focus_weights,
                plan_used_quota=getattr(self,'_plan_used_quota',0),
                plan_events=getattr(self,'_plan_events',[])),'cpu')

    def on_load_checkpoint(self, ckpt):
        self._rate_budget_state=ckpt.get('rate_budget_controller')
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
        if (getattr(self.config.rigid_tree,'target_update_bytes',0) and
                ckpt.get('current_frame',0)>=1 and self._rate_budget_state is None):
            raise ValueError('Budget-controlled checkpoint lost its controller state')
        if self._rate_budget_state is not None:
            from .model.rate_budget import RateBudgetState
            RateBudgetState.restore(self._rate_budget_state,self.config.rigid_tree.max_rate_multiplier)
        allocation = birth_allocation(self.config)
        if (ckpt.get('current_frame', 0) >= 1 and
                ckpt.get('birth_allocation', 'clone_small_v1') != allocation):
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
            saved_config['feature_dc_update'] = {
                **asdict(FeatureDCUpdateConfig()), **saved_config.get('feature_dc_update', {})}
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
        saved_appearance_gradient = ckpt.get('birth_appearance_gradient')
        self.birth_appearance_gradient = (saved_appearance_gradient.to(self.device)
            if saved_appearance_gradient is not None else None)
        saved_handoff = ckpt.get('birth_handoff_sources')
        self.birth_handoff_sources = (saved_handoff.to(self.device).long()
            if saved_handoff is not None else None)
        transaction = ckpt.get('birth_transaction')
        self.birth_transaction = (BirthTransaction.restore(transaction, self.device)
                                  if transaction is not None else None)
        cohort = ckpt.get('birth_cohort_state', {})
        self.birth_cohort = cohort.get('index', 0)
        self.birth_cohort_start = cohort.get('start_step', getattr(self, 'motion_steps', 0))
        self.birth_end_step = cohort.get('end_step')
        self.coarse_birth_snapshot = cohort.get('coarse_snapshot')
        terminal = ckpt.get('birth_terminal_state', {})
        self.birth_terminal = terminal.get('active', False)
        self.selected_birth_state = terminal.get('selected')
        self._transaction_views = None
        if self.frame_prepared:
            self.motion.restore(ckpt['rigid_tree_motion'])
            if self.config.opacity_update.enabled:
                self.opacity_update.restore(ckpt.get('quantized_opacity_update'), self.device)
            if self.config.feature_dc_update.enabled:
                self.feature_dc_update.restore(ckpt.get('quantized_feature_dc_update'), self.device)
            if self.motion_committed:
                for parameter in self.motion.parameters():
                    parameter.requires_grad_(False)
                self.motion.optimizer.zero_grad(set_to_none=True)
        if ckpt['births'] is not None and not uses_early_replacement(self.config):
            self.births = birth_model_factory(self.config.sh_degree, capture=ckpt['births'])
            self.births.restore(ckpt['births'], self.config.gaussian_stage2)
            source = ckpt['birth_source_rows']
            if source is None or len(source) != len(self.births._xyz):
                raise ValueError('Birth checkpoint lost source lineage')
            self.births.source_rows = source.to(self.device)
            freeze_birth_radiance(self.births, self.config)
            if not uses_quick_source(self.config):
                set_family_training_stage(self.births,getattr(self,'birth_terminal',False))
            if hasattr(self.births, 'patch_retention'):
                self._set_patch_stage()
            if hasattr(self.births,'reincarnation_state') and not uses_dual_predictive(self.config) and not uses_quick_source(self.config):
                self.births.set_training_stage(getattr(self, 'birth_terminal', False))
        if uses_dual_predictive(self.config):
            self._restore_dual_checkpoint(ckpt.get('dual_birth_state'))
        if uses_quick_source(self.config):
            quick = ckpt.get('quick_birth_state') or {}
            self.quick_birth_phase = quick.get('phase',1)
            self.quick_birth_optimizer_steps = quick.get('optimizer_steps',0)
            self.quick_birth_initialized = quick.get('initialized',False)
            self.shadow_initialized_state = BirthTransaction.tensor_tree(ckpt.get('shadow_initialized_state'),'cpu')
            if self.motion_committed and not self.frame_finalized:
                if not self.quick_birth_initialized or self.birth_end_step!=300:
                    raise ValueError('Quick checkpoint lost its one-time initialization/300-step schedule')
                self.births.set_training_stage(self.quick_birth_phase)
        if uses_persistent_birth(self.config):
            bank=ckpt.get('persistent_newborn_bank') or {}
            self.persistent_bank_state=bank.get('state')
            self.persistent_uid_layout=dict(bank.get('layout',{}))
            self.persistent_specs=dict(bank.get('specs',{}))
            self.persistent_candidate_specs=dict(bank.get('candidate_specs',{}))
            self.persistent_pretrain_state=bank.get('pretrain')
            progress=ckpt.get('persistent_birth_state') or {}
            self.persistent_birth_optimizer_steps=progress.get('optimizer_steps',0)
            self.persistent_training_population_count=progress.get('training_population_count',0)
            self.persistent_initialized=progress.get('initialized',False)
            if self.motion_committed and not self.frame_finalized and not self.persistent_initialized:
                raise ValueError('Persistent bank checkpoint lost its prepared single birth stage')
            self._validate_persistent_layout()
        if uses_early_replacement(self.config):
            state=BirthTransaction.tensor_tree(ckpt.get('early_replacement_state') or {},self.device)
            self.early_initialized=state.get('initialized',False)
            self.early_birth_updates=state.get('birth_updates',0)
            self.early_shared_steps=state.get('shared_steps',0)
            self.early_selected=state.get('selected')
            self.early_pinned_rows=state.get('pinned_rows')
            self.early_pinned_values=state.get('pinned_values')
            self.early_focus_weights=state.get('focus_weights',{})
            self._plan_used_quota=state.get('plan_used_quota',0)
            self._plan_events=state.get('plan_events',[])
            if self.frame_prepared:
                self._early_attach_opacity()
            if ckpt['births'] is not None:
                from .model.early_replacement_model import EarlyReplacementModel
                if self.config.birth.radiance_mode=='adaptive_residual_birth':
                    from .model.adaptive_birth_model import AdaptiveBirthModel
                    EarlyReplacementModel=AdaptiveBirthModel
                if getattr(self.config.birth,'frame_training_plan',False):
                    from .model.frame_plan_birth import FramePlanBirthModel
                    self.births=FramePlanBirthModel(self.config.sh_degree,self.device).restore(
                        ckpt['births'],self.motion.optimizer,self.config.gaussian_stage2,self.device)
                else:
                    self.births=EarlyReplacementModel(self.config.sh_degree)
                    self.births.restore(ckpt['births'],self.motion.optimizer,self.config.gaussian_stage2)
                self.births.source_rows=ckpt['birth_source_rows'].to(self.device)
                if not getattr(self.config.birth,'frame_training_plan',False):
                    self.births._opacity.requires_grad_(self.motion_committed or self.config.birth.radiance_mode=='adaptive_residual_birth')
            if self.motion_committed and fitting_old_motion(self):
                self.motion.prepare_plan_step(self.motion.last_step)

    def prepare_frame(self):
        if self.frame_prepared:
            return
        if self.current_frame < 1:
            raise ValueError('Rigid motion starts after the shared first frame')
        if uses_quick_source(self.config):
            if self.motion_steps!=200 or self.config.rigid_tree.birth_steps!=100 or self.config.opacity_update.joint_with_birth:
                raise ValueError('Quick training requires exactly motion200 + single-model birth100 and no old-alpha birth refit')
        if uses_persistent_birth(self.config):
            if (self.motion_steps!=200 or self.config.rigid_tree.birth_steps!=100 or
                    self.config.opacity_update.joint_with_birth or not self.config.birth.anchor_sfm_path):
                raise ValueError('Persistent births require motion200 + birth100, fixed existing old-alpha phase and train-SfM anchors')
            self._validate_persistent_layout()
        if uses_early_replacement(self.config) and (
                self.motion_steps!=200 or self.config.rigid_tree.birth_steps!=100 or
                self.config.rigid_tree.spatial_gate.freeze_step!=(180 if getattr(self.config.birth,'frame_training_plan',False) else 70)):
            raise ValueError('Early replacement requires unchanged gate70, motion200 and global300 schedule')
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
        if getattr(self.config.birth,'frame_training_plan',False):
            self._plan_setup_rate_budget()
        self.motion.setup(xyz, rotation, self.datamodule.cameras_extent,
                          frame_index=self.current_frame, previous_rows=previous_rows)
        self.frame_prepared = True
        self.frame_finalized = False
        self.motion_committed = False
        self.frame_report = None
        self.birth_appearance_gradient = None
        self.birth_handoff_sources = None
        self.birth_transaction = None
        self.birth_cohort = 0
        self.birth_cohort_start = self.motion_steps
        self.birth_end_step = None
        self.coarse_birth_snapshot = None
        self.birth_terminal = False
        self.selected_birth_state = None
        self._transaction_views = None
        self.birth_branches = None
        self.dual_coarse_branches = None
        self.dual_training_totals = None
        self.quick_birth_phase = 1
        self.quick_birth_optimizer_steps = 0
        self.quick_birth_initialized = False
        self.shadow_initialized_state = None
        self.persistent_birth_optimizer_steps=0
        self.persistent_training_population_count=0
        self.persistent_initialized=False
        self.persistent_pretrain_state=None
        self.persistent_candidate_specs={}
        self.early_initialized=False;self.early_birth_updates=0;self.early_shared_steps=0
        self.early_selected=None;self.early_pinned_rows=None;self.early_pinned_values=None
        self.early_focus_weights={}
        self._plan_used_quota=0;self._plan_events=[]
        self._plan_phase_started=None;self._plan_timed_phase=None
        self.birth_control = dict(clone_added=0, split_added=0, pruned=0,
                                  source_tracked=True, retirement=retirement,
                                  previous_birth_count=previous_birth_count)
        self.stage_image_metrics = []
        self._freeze_base()
        if self.opacity_update is not None:
            self.opacity_update.setup(self.gaussians._opacity)
        if self.feature_dc_update is not None:
            self.feature_dc_update.setup(self.gaussians._features_dc)
        if uses_early_replacement(self.config):
            self._early_attach_opacity()

    def pre_training_step(self):
        if getattr(self.config.birth,'frame_training_plan',False):
            if (not self.config.rigid_tree.spatial_gate.frame_training_plan or
                    self.config.birth.radiance_mode != 'adaptive_residual_birth' or
                    self.config.rigid_tree.amplitude_refit_steps or
                    self.config.opacity_update.amplitude_refit_steps or
                    self.config.feature_dc_update.enabled or
                    not self.config.birth.rotation_updates or
                    not self.config.birth.joint_old_opacity or self.config.lambda_dssim != .2 or
                    self.config.photometric_loss != 'l1' or self.config.birth.photometric_loss != 'l1'):
                raise ValueError('Frame plan needs its gate schedule, independent residuals, no shared colour, new rotation and L1/DSSIM')
        self.prepare_frame()
        if not self.motion_committed and self.opacity_update is not None:
            self.opacity_update.prepare_step(self.trainer.global_step, self.motion_steps)
        if uses_early_replacement(self.config):
            self._early_prepare_rates(self.trainer.global_step)
            return
        if self.motion_committed and len(self.births._xyz):
            birth_step = self.trainer.global_step - (self.birth_cohort_start
                if uses_residual_birth(self.config) else self.motion_steps)
            models = ([value['model'] for value in self.birth_branches.values()]
                      if uses_dual_predictive(self.config) and self.birth_branches else [self.births])
            for model in models:
                set_birth_learning_rates(model,self.config.gaussian_stage2,birth_step,
                    self.config.rigid_tree.birth_steps,self.config.birth.lr_final_ratio)

    def begin_next_frame(self):
        if not self.frame_finalized:
            raise ValueError('Cannot advance before quantized finalization')
        self.frame_prepared = False
        self.frame_finalized = False
        self.motion_committed = False
        self.frame_report = None


    def make_birth_view(self, reference=None, viewpoint=None, training=False):
        if not len(self.births._xyz):
            return None
        transaction = getattr(self, 'birth_transaction', None)
        if reference is None:
            if hasattr(self.births, 'raw_old_alpha'):
                reference, _, _ = self._surface_old_reference()
            else:
                reference = (transaction.fixed_reference if transaction is not None and
                             transaction.fixed_reference is not None else
                             ({name:getattr(self.gaussians,attr).detach() for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES)}
                              if self.config.birth.radiance_mode=='adaptive_residual_birth' else model_raw(self.gaussians)))
        feature_update = getattr(self, 'feature_dc_update', None)
        if feature_update is not None and not self.frame_finalized:
            reference = dict(reference)
            reference['sh_dc'] = feature_update().detach()
        result = QuantizedNewGaussianView(self.births, self.config.birth,
            self.datamodule.cameras_extent, reference,
            source_rows=self.births.source_rows)
        if uses_reincarnation(self.config):
            novel = self.births.novel_mask
            if training and not self.birth_terminal:
                crossfit = transaction.crossfit_state
                branch = crossfit.get('training_branch', 'A')
                names = crossfit['camera_groups'][branch]
                active = viewpoint is not None and Path(viewpoint.path).name in names
                scale = result.get_xyz.new_ones(len(novel))
                scale[novel] = len(transaction.fit_names)/len(names) if active else 0.
                result.row_gradient_scale = scale
            if self.birth_terminal:
                result.locked_geometry_rows = novel
        return result

    def _surface_old_reference(self):
        transaction = self.birth_transaction
        logits, alpha, _ = patch_old_state(self)
        if transaction.fixed_reference is not None:
            keep = transaction.fixed_keep
            reference = dict(transaction.fixed_reference)
        else:
            keep = torch.arange(len(self.gaussians._xyz), device=self.device)
            reference = dict(transaction.old_render_reference or model_raw(self.gaussians))
        reference['opacity'] = logits[keep]
        return reference, alpha[keep], keep

    def forward(self, viewpoint, training=False, old_only=False, proposal_only=False,
                predictor_only=False):
        auxiliary = proposal_only or predictor_only
        if (proposal_only and predictor_only) or (auxiliary and (
                not training or not self.frame_prepared or self.motion_committed)):
            raise ValueError('Auxiliary render is only available during motion training')
        if uses_dual_predictive(self.config) and training and self.motion_committed and not self.frame_finalized and not old_only:
            self._activate_dual_for_view(viewpoint)
        old_model = self.renderer.gaussians
        effective = getattr(self.config, 'effective_motion_observations', False)
        previous_collection = getattr(self.renderer, 'collect_contributions', False)
        early_pose=None
        birth_view=None
        if effective:
            self.renderer.collect_contributions = bool(training and not auxiliary and self.frame_prepared and
                (not self.motion_committed or (uses_residual_birth(self.config) and
                  not self.frame_finalized)))
        try:
            if fitting_old_motion(self):
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
            elif (self.motion_committed and not self.frame_finalized and
                    getattr(self, 'opacity_update', None) is not None and
                    (self.config.opacity_update.joint_with_birth or self.config.birth.joint_old_opacity)):
                if self.config.opacity_update.mode == 'alpha':
                    self.gaussians.rigid_alpha = self.opacity_update.alphas()
                else:
                    self.gaussians.rigid_opacity = self.opacity_update()
            if (getattr(self, 'feature_dc_update', None) is not None and self.frame_prepared and
                    not self.frame_finalized and
                    (not self.motion_committed or self.config.feature_dc_update.joint_with_birth)):
                self.gaussians.rigid_sh_dc = self.feature_dc_update(training=training)
                if auxiliary:
                    self.gaussians.rigid_sh_dc = self.gaussians.rigid_sh_dc.detach()
            if (uses_early_replacement(self.config) and getattr(self,'early_initialized',False) and
                    not self.motion_committed and not old_only):
                xyz=self.gaussians.rigid_xyz
                q=self.gaussians.rigid_rotation
                self.births.bind_source_pose(xyz,q,detach=False)
                early_pose=(xyz,q)
                if self.config.birth.radiance_mode=='adaptive_residual_birth':
                    birth_view=self.make_birth_view(reference=self._adaptive_reference())
                    self.renderer.gaussians=(MergedGaussianModel([self.gaussians,birth_view])
                        if birth_view is not None else self.gaussians)
                else:self.renderer.gaussians=LiveReplacementView(self.gaussians,self.births)
            if self.motion_committed and not self.frame_finalized and not old_only:
                patch = hasattr(self.births, 'raw_old_alpha')
                if patch:
                    patch_reference, patch_alpha, _ = self._surface_old_reference()
                    birth_view = self.make_birth_view(reference=patch_reference)
                else:
                    birth_view = (self.make_birth_view(reference=self._adaptive_reference())
                                  if self.config.birth.joint_old_opacity else
                                  self.make_birth_view(viewpoint=viewpoint, training=training)
                                  if uses_reincarnation(self.config) else self.make_birth_view())
                old_view = self.gaussians
                transaction = getattr(self, 'birth_transaction', None)
                if patch:
                    old_view = RawView(patch_reference, self.config.sh_degree, self.gaussians, patch_alpha)
                elif transaction is not None and transaction.fixed_reference is not None:
                    old_view = RawView(transaction.fixed_reference, self.config.sh_degree, self.gaussians)
                elif (transaction is not None and birth_view is not None and
                      self.config.birth.radiance_mode!='adaptive_residual_birth'):
                    keep = torch.ones(len(self.gaussians._xyz), dtype=torch.bool, device=self.device)
                    parents = transaction.replacements
                    if hasattr(self.births,'shadow_entity_mask'):
                        parents = parents[self.births.shadow_entity_mask()]
                    keep[parents[parents >= 0]] = False
                    old_view = SubsetView(self.gaussians, keep)
                if birth_view is not None:
                    self.renderer.gaussians = MergedGaussianModel([old_view, birth_view])
                else:
                    self.renderer.gaussians = old_view
            probe = None
            if (training and not auxiliary and self.frame_prepared and
                    not self.motion_committed and
                    getattr(getattr(self.config, 'birth', None),
                            'appearance_candidates', 0) > 0 and
                    self.trainer.global_step >= (self.motion_steps -
                        self.config.birth.densification_stats_steps + 1)):
                probe = self.gaussians._features_dc.detach().requires_grad_(True)
                self.gaussians.rigid_sh_dc = probe
            package = self.renderer(viewpoint, self.background)
            if (training and not auxiliary and self.config.birth.storage_loss_enabled and
                    self.config.birth.radiance_mode=='adaptive_residual_birth'):
                package['_adaptive_birth_rate_view']=birth_view
            if uses_dual_predictive(self.config) and training and self.motion_committed:
                package['birth_branch'] = self.birth_active_branch
            if probe is not None:
                package['birth_feature_probe'] = probe
            return package
        finally:
            if early_pose is not None:
                self.births.bind_source_pose(*early_pose,detach=True)
            self.gaussians.rigid_xyz = None
            self.gaussians.rigid_rotation = None
            self.gaussians.rigid_opacity = None
            self.gaussians.rigid_alpha = None
            self.gaussians.rigid_sh_dc = None
            self.renderer.gaussians = old_model
            if effective:
                self.renderer.collect_contributions = previous_collection

    def training_step(self, viewpoint, step, regularization_weight=1.):
        package = self.forward(viewpoint, training=True)
        adaptive_rate_view=package.pop('_adaptive_birth_rate_view',None)
        image = package['render']
        factor = motion_image_factor(step, self.motion_steps,
            enabled=self.config.motion_coarse_to_fine,
            fitting_motion=fitting_old_motion(self),
            bootstrap_steps=self.config.rigid_tree.spatial_gate.start_step)
        metric = getattr(self.config, 'photometric_loss', 'l1')
        if (self.motion_committed or (uses_early_replacement(self.config) and self.early_initialized)) and self.config.birth.photometric_loss is not None:
            metric = self.config.birth.photometric_loss
        image_loss = motion_photometric_loss(image, viewpoint.image,
                                             self.config.lambda_dssim, factor, metric=metric)
        loss = image_loss
        # Leave scalar diagnostics on device until all views have finished.
        # Converting each one to float would synchronize CUDA before backward.
        metrics = dict(image_loss=image_loss.detach(), motion_image_factor=factor,
                       psnr=psnr(image.detach(), viewpoint.image).detach(),
                       gs=len(self.gaussians._xyz) +
                       (len(self.births._xyz) if self.motion_committed else 0))
        if uses_early_replacement(self.config) and self.early_initialized:
            metrics['gs']=(len(self.gaussians._xyz)+len(self.births.parent_rows)
                if self.config.birth.radiance_mode=='adaptive_residual_birth' else len(self.birth_transaction.fixed_keep)+len(self.births.parent_rows)
                if self.motion_committed else len(self.gaussians._xyz)+(len(self.births.parent_rows) if self.config.birth.radiance_mode=='adaptive_residual_birth' else 0))
            error=(image-viewpoint.image).square().mean(0)
            weight=self.early_focus_weights.get(Path(viewpoint.path).name)
            metrics['early_batch_mse']=error.mean().detach()
            if weight is not None:
                metrics['early_focus_mse']=((error*weight).sum()/weight.sum().clamp_min(1e-12)).detach()
        proposal_weight = (getattr(self.motion, 'proposal_image_weight', 0.)
                           if not self.motion_committed else 0.)
        if proposal_weight:
            proposal = self.forward(viewpoint, training=True, proposal_only=True)['render']
            proposal_loss = motion_photometric_loss(proposal, viewpoint.image,
                                                    self.config.lambda_dssim, factor, metric=metric)
            loss = loss + proposal_weight * proposal_loss
            metrics['proposal_image_loss'] = proposal_loss.detach()
        predictor_weight = (getattr(self.motion, 'predictor_image_weight', 0.)
                            if not self.motion_committed else 0.)
        if predictor_weight:
            prediction = self.forward(viewpoint, training=True, predictor_only=True)['render']
            prediction_loss = motion_photometric_loss(prediction, viewpoint.image,
                self.config.lambda_dssim, factor, metric=metric)
            loss = loss + predictor_weight * prediction_loss
            metrics['predictor_image_loss'] = prediction_loss.detach()
        if fitting_old_motion(self) and regularization_weight:
            loss = loss + regularization_weight * self.motion.regularization_loss(step)
        if (regularization_weight and getattr(self, 'opacity_update', None) is not None and
                (not getattr(self.config.birth,'frame_training_plan',False) or step >= 101) and
                (not self.motion_committed or
                 (not self.frame_finalized and (self.config.opacity_update.joint_with_birth or
                                               self.config.birth.joint_old_opacity)))):
            terms = self.opacity_update.rate_terms(self._rate_byte_price(step),
                width=self.config.rigid_tree.sparse_rate_width_bins)
            loss = loss + regularization_weight * terms['loss']
        if (regularization_weight and getattr(self, 'feature_dc_update', None) is not None and
                (not self.motion_committed or
                 (not self.frame_finalized and self.config.feature_dc_update.joint_with_birth))):
            terms = self.feature_dc_update.rate_terms(self._rate_byte_price(step),
                width=self.config.rigid_tree.sparse_rate_width_bins)
            loss = loss + regularization_weight * terms['loss']
        if (self.motion_committed and regularization_weight and len(self.births._xyz) and
                self.config.birth.alpha_sparsity_weight):
            penalty = (self.config.birth.alpha_sparsity_weight *
                self.make_birth_view().get_opacity.sum() /
                max(1, len(self.gaussians._xyz)))
            loss = loss + regularization_weight * penalty
            metrics['birth_alpha_sparsity_loss'] = penalty.detach()
        if (regularization_weight and self.config.birth.storage_loss_enabled and
                self.config.birth.radiance_mode=='adaptive_residual_birth' and self.early_initialized):
            penalty,storage_metrics=self._adaptive_storage_regularization(step,adaptive_rate_view)
            loss=loss+regularization_weight*penalty
            # Trainer divides both losses and metrics by the camera count.
            metrics.update({name:value*regularization_weight for name,value in storage_metrics.items()})
        if (regularization_weight and self.config.birth.population_loss_weight and
                ((getattr(self.config.birth,'frame_training_plan',False) and step >= 101) or self.early_initialized)):
            penalty,population_metrics=self._adaptive_population_regularization(step)
            loss=loss+regularization_weight*penalty
            metrics.update({name:value*regularization_weight for name,value in population_metrics.items()})
        if uses_dual_predictive(self.config) and self.motion_committed and not self.frame_finalized:
            package['birth_image_loss'] = image_loss.detach()
            package['birth_psnr'] = metrics['psnr']
            groups = self.birth_transaction.crossfit_state['camera_groups']
            loss = loss*(len(self.birth_transaction.fit_names)/len(groups[self.birth_active_branch]))
        return loss, metrics, package

    def _rate_byte_price(self, step):
        cfg = self.config.rigid_tree
        ramp = min(max((step - cfg.rate_warmup_steps) / cfg.rate_ramp_steps, 0.), 1.)
        weight=getattr(self.motion,'effective_rate_loss_weight',cfg.rate_loss_weight)
        return weight * ramp / self.motion.rate_normalization_bytes

    @torch.no_grad()
    def _step_feature_dc(self, step):
        update = getattr(self, 'feature_dc_update', None)
        if update is None:
            return
        window = self.config.rigid_tree.motion_fit_window_steps
        update.optimizer_step(self._rate_byte_price(step),
            proximal=(window is None or step % window == 0 or step == self.total_steps),
            lr_factor=self.motion._motion_lr_factor(min(step, self.motion_steps))
                if self.config.feature_dc_update.decay_with_motion else 1.)

    @torch.no_grad()
    def _commit_opacity_and_retire(self, forced_remove=None):
        """Retire only low CURRENT quantized alpha, before creating new clones."""
        update = self.opacity_update
        logits = update.canonical_logits(full=True).to(self.device)
        remove = (logits.sigmoid().flatten() < self.config.opacity_update.prune_threshold
                  if forced_remove is None else forced_remove.clone())
        if remove.shape != (len(self.gaussians._xyz),) or remove.dtype != torch.bool:
            raise ValueError('Retirement mask must match the current incumbent rows')
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
            if getattr(self, 'birth_appearance_gradient', None) is not None:
                self.birth_appearance_gradient = self.birth_appearance_gradient[rows]
        update.commit(rows)
        self.gaussians._opacity.copy_(logits[rows])
        if getattr(self, 'feature_dc_update', None) is not None:
            self.feature_dc_update.commit(rows)
            self.gaussians._features_dc.copy_(self.feature_dc_update.canonical_features().to(self.device))
        return rows

    def transaction_camera_views(self):
        """Use every standard training camera for both fitting and selection."""
        if self._transaction_views is None:
            dataset = self.datamodule.get_train_dataset()
            ordered = sorted((Path(item.path).name, index)
                             for index, item in enumerate(dataset.items))
            test = self.datamodule.get_test_dataset()
            test_names = {Path(item.path).name for item in test.items} if test is not None else set()
            names = [name for name, _ in ordered]
            if test_names.intersection(names):
                raise ValueError('Birth transactions cannot use external test cameras')
            if len(names) < 3:
                raise ValueError('Residual birth needs at least three training cameras')
            if self.birth_transaction is None:
                fitting = validation = tuple(names)
            else:
                fitting = self.birth_transaction.fit_names
                validation = self.birth_transaction.validation_names
            self._transaction_views = {name: dataset[index].to(self.device)
                                       for name, index in ordered}
            self._transaction_fit_names = fitting
            self._transaction_validation_names = validation
        return self._transaction_views

    def transaction_training_view(self, step, index, count):
        views = self.transaction_camera_views()
        names = self.birth_transaction.fit_names
        if (uses_reincarnation(self.config) or uses_dual_predictive(self.config)) and not self.birth_terminal:
            groups = self.birth_transaction.crossfit_state['camera_groups']
            # Three-view updates always include both complementary groups,
            # while each of the twenty P cameras has identical frequency.
            a, b = groups['A'], groups['B']
            names = tuple(item for pair in zip(a, b) for item in pair)
        camera = ((step-self.motion_steps-1)*count+index) % len(names)
        name = names[camera]
        return views[name], self.birth_transaction.fit_names.index(name)

    @torch.no_grad()
    def _begin_residual_births(self):
        views = self.transaction_camera_views()
        transaction = propose_residual_births(self, self.config.birth.residual,
            list(views.values()), self._transaction_fit_names,
            self._transaction_validation_names)
        self.birth_cohort = 0
        self.birth_cohort_start = self.trainer.global_step
        self.birth_end_step = self.birth_cohort_start+self.config.rigid_tree.birth_steps
        self.coarse_birth_snapshot = []
        self._install_birth_transaction(transaction)
        if uses_surface_patch(self.config) or uses_reincarnation(self.config) or uses_dual_predictive(self.config) or uses_quick_source(self.config):
            baseline = encode_birth_trial(self, torch.zeros(len(self.births._xyz), dtype=torch.bool, device=self.device),
                                          torch.zeros(len(self.gaussians._xyz), dtype=torch.bool, device=self.device))
            self.birth_transaction.old_render_reference = baseline.raw
        self.birth_control.update(proposal_count=len(transaction.sources),
            spatial_proposals=transaction.proposed_spatial,
            repair_proposals=transaction.proposed_repairs,
            region_support=transaction.region_diagnostics,
            selection_camera_scope='all_standard_training_cameras',
            source_mutated_before_acceptance=False)
        if uses_reincarnation(self.config) or uses_dual_predictive(self.config):
            self.birth_control['native_support_initialization'] = self._check_reincarnation_initialization(views)
        if hasattr(self.births, 'patch_rows'):
            self.birth_control['initial_patch_exposure'] = self._patch_training_diagnostics()
        self._freeze_base()
        fitting = [views[name] for name in transaction.fit_names]
        self.birth_control['fitting_camera_names'] = list(transaction.fit_names)
        self.birth_control['fitting_baseline_mse'] = fitting_errors(self, fitting, True).tolist()
        self.birth_control['fitting_initial_proposal_mse'] = fitting_errors(self, fitting, False).tolist()
        if uses_dual_predictive(self.config):
            self._begin_dual_branches()
        if uses_shadow_source(self.config):
            self._initialize_shadow_births(views)
        elif uses_quick_source(self.config):
            self._initialize_quick_births(views)

    @torch.no_grad()
    def _check_reincarnation_initialization(self,views):
        """Measure native clone equivalence, including depth-tie ordering effects."""
        mask = self.birth_transaction.replacements>=0
        deleted = torch.zeros(len(self.gaussians._xyz),dtype=torch.bool,device=self.device)
        deleted[self.birth_transaction.replacements[mask]]=True
        trial = encode_birth_trial(self,mask,deleted)
        baseline = self.birth_transaction.old_render_reference
        previous = self.renderer.gaussians
        collect = self.renderer.collect_contributions
        self.renderer.collect_contributions = False
        per_view = []
        try:
            for name,view in views.items():
                self.renderer.gaussians = RawView(baseline,self.config.sh_degree,self.gaussians)
                before = self.renderer(view,self.background)['render']
                self.renderer.gaussians = RawView(trial.raw,self.config.sh_degree,self.gaussians)
                after = self.renderer(view,self.background)['render']
                per_view.append(dict(camera=name,max_abs_difference=float((after-before).abs().max()),
                    mse_difference=float((after-before).square().mean()),bitwise_equal=bool(torch.equal(after,before))))
        finally:
            self.renderer.gaussians,self.renderer.collect_contributions=previous,collect
        return dict(per_view=per_view,all_native_bitwise_equal=all(x['bitwise_equal'] for x in per_view),
            max_abs_difference=max(x['max_abs_difference'] for x in per_view),
            semantics='P-only actual codec replacement versus original motion scene; equal-depth stable sort may change order',
            independent_novel_excluded=True,incumbent_mutated=False)

    def _install_birth_transaction(self, transaction):
        """Every topology generation starts a fresh optimizer for the whole pool."""
        self.birth_transaction = transaction
        if getattr(transaction,'persistent_state',None) is not None:
            self.births=birth_model_factory(self.config.sh_degree,persistent=True)
            self.births.restore(BirthTransaction.tensor_tree(transaction.persistent_state,self.device),
                                self.config.gaussian_stage2)
            self.births.source_rows=transaction.sources.clone()
            return
        self.births = birth_model_factory(self.config.sh_degree, family_state=transaction.family_state,
                                          patch_state=transaction.patch_state,
                                          reincarnation_state=transaction.reincarnation_state,
                                          predictive_model=transaction.predictive_model,
                                          source_residual=transaction.predictive_model and uses_source_residual(self.config),
                                          quick_source=transaction.predictive_model and uses_quick_source(self.config),
                                          shadow_source=transaction.predictive_model and uses_shadow_source(self.config))
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self.births, attr, torch.nn.Parameter(transaction.raw[name].clone()))
        if transaction.family_state is not None:
            self.births.install_family_state(transaction.family_state)
        if transaction.patch_state is not None:
            self.births.install_patch_state(transaction.patch_state)
            self._set_patch_stage()
        if transaction.reincarnation_state is not None:
            self.births.install_reincarnation_state(transaction.reincarnation_state)
            if not uses_dual_predictive(self.config) and not uses_quick_source(self.config):
                self.births.set_training_stage(getattr(self,'birth_terminal',False))
        freeze_birth_radiance(self.births, self.config)
        if not uses_dual_predictive(self.config) and not uses_quick_source(self.config):
            set_family_training_stage(self.births,getattr(self,'birth_terminal',False))
        self.births.active_sh_degree = self.gaussians.active_sh_degree
        self.births.spatial_lr_scale = self.gaussians.spatial_lr_scale
        self.births.source_rows = transaction.sources.clone()
        self.births.max_radii2D = self.gaussians._xyz.new_zeros(len(transaction.sources))
        if len(transaction.sources) and not getattr(self,'_quick_finalizing',False):
            self.births.initialize_incremental_optimizer(self.config.gaussian_stage2)

    def _restore_birth_pool(self, snapshot):
        self.birth_transaction = BirthTransaction.restore(snapshot['transaction'], self.device)
        self.births = birth_model_factory(self.config.sh_degree, capture=snapshot['model'])
        self.births.restore(BirthTransaction.tensor_tree(snapshot['model'], self.device),
                            self.config.gaussian_stage2)
        self.births.source_rows = self.birth_transaction.sources.clone()
        freeze_birth_radiance(self.births, self.config)
        if not uses_dual_predictive(self.config) and not uses_quick_source(self.config):
            set_family_training_stage(self.births,getattr(self,'birth_terminal',False))
        if hasattr(self.births, 'patch_retention'):
            self._set_patch_stage()
        if hasattr(self.births,'reincarnation_state') and not uses_dual_predictive(self.config) and not uses_quick_source(self.config):
            self.births.set_training_stage(getattr(self, 'birth_terminal', False))

    def _validate_persistent_layout(self):
        layout=getattr(self,'persistent_uid_layout',{})
        if (any(not isinstance(uid,str) or isinstance(row,bool) or not isinstance(row,int) or
                row<0 or row>=len(self.gaussians._xyz) for uid,row in layout.items()) or
                len(set(layout.values()))!=len(layout) or
                set(layout)!=set(getattr(self,'persistent_specs',{})) or
                (layout and getattr(self,'persistent_bank_state',None) is None)):
            raise ValueError('Persistent UID bank must map uniquely onto actual committed Gaussian rows')

    @torch.no_grad()
    def _begin_persistent_births(self):
        """Bind one persistent UID pool to current motion, then add train-track geometry."""
        from .model.persistent_newborn_model import PersistentNewbornModel
        from .model.tracked_surface_proposal import propose_tracked_surface_births
        if self.trainer.global_step!=200 or len(self.transaction_camera_views())!=20:
            raise ValueError('Persistent anchor births start at motion200 with all20 standard training cameras')
        self._validate_persistent_layout()
        views=self.transaction_camera_views();names=tuple(views)
        base=model_raw(self.gaussians)
        active=dict(self.persistent_uid_layout)
        capacity=self.config.birth.residual.max_candidates
        if len(active)>self.config.birth.residual.max_gaussians:
            raise ValueError('Persistent UID bank exceeds the actual scene population cap')
        residuals={name:(self.forward(view,old_only=True)['render'].clamp(0,1)-view.image)
                   for name,view in views.items()}
        proposal,geometry_report=propose_tracked_surface_births(self.current_frame,views,
            self.config.birth.anchor_sfm_path,capacity,active_uids=tuple(active),residuals=residuals,
            image_root=Path(self.datamodule.config.root)/'frames')
        if not isinstance(geometry_report,dict):raise ValueError('Geometry proposal must provide its actual support evidence')
        by_uid={spec['uid']:spec for spec in proposal}
        if len(by_uid)!=len(proposal):raise ValueError('Tracked geometry proposed duplicate persistent identities')
        def convert_raw(raw):
            return {name:torch.as_tensor(raw[name],device=self.device,dtype=base[name].dtype).clone() for name in RAW_FIELDS}
        # The persistent bank may be much larger than this frame's train pool.
        # Reliable current bindings and fresh anchors compete by the same SSE.
        ranked=[]
        for uid,spec in by_uid.items():
            score=float(spec['score'])
            if not math.isfinite(score) or score<0:raise ValueError('Anchor support scores must be finite current residual SSE')
            if bool(spec.get('binding_only',False))!=(uid in active):
                raise ValueError('Tracked anchor ownership disagrees with the actual UID bank')
            if score>0:ranked.append(spec)
        scheduled=sorted(ranked,key=lambda spec:(-float(spec['score']),spec['uid']))[:capacity]
        specs=[];rows=[]
        for spec in scheduled:
            uid=spec['uid']
            if uid not in active:
                specs.append({**spec,'raw':convert_raw(spec['raw'])});rows.append(-1)
        fresh=len(specs)
        for spec in scheduled:
            uid=spec['uid']
            if uid not in active:continue
            row=active[uid]
            raw={name:value[row].clone() for name,value in base.items()}
            # Material and Adam are continued. A trusted current track may
            # bind geometry; it cannot reset learned radiance to its seed.
            geometry=convert_raw(spec['raw'])
            for name in ('xyz','rotation','logscale'):raw[name]=geometry[name]
            specs.append({**self.persistent_specs[uid],'uid':uid,'raw':raw,'score':float(spec['score'])});rows.append(row)
        model=PersistentNewbornModel(self.config.sh_degree).install(specs,
            uids=[spec['uid'] for spec in specs],previous_state=self.persistent_bank_state,
            old_rows=torch.tensor(rows,device=self.device,dtype=torch.long),
            frame_index=self.current_frame,device=self.device,dtype=base['xyz'].dtype)
        if specs:
            # Previous moments/priors survive, but material starts from exactly
            # the physical population actually decoded before this birth stage.
            model.rebind(model.uids,{name:torch.stack([spec['raw'][name] for spec in specs]) for name in RAW_FIELDS},
                         old_rows=torch.tensor(rows,device=self.device,dtype=torch.long),frame_index=self.current_frame)
        model.active_sh_degree=self.gaussians.active_sh_degree
        model.spatial_lr_scale=self.gaussians.spatial_lr_scale
        model.training_setup(self.config.gaussian_stage2)
        raw=model_raw(model);n=len(rows)
        sources=nearest_sources(raw['xyz'],base['xyz'])
        model.source_rows=sources.clone()
        scores=raw['xyz'].new_tensor([float(spec.get('score',spec.get('residual_mass',1.))) for spec in specs])
        self.births=model
        self.birth_transaction=BirthTransaction(raw,sources,torch.tensor(rows,device=self.device,dtype=torch.long),scores,
            raw['xyz'].clone(),raw['rotation'].clone(),raw['logscale'].exp(),
            torch.zeros(n,device=self.device,dtype=torch.int64),names,names,fresh,n-fresh,
            group_ids=torch.arange(n,device=self.device),region_diagnostics=geometry_report,
            persistent_state=model.capture())
        baseline=encode_birth_trial(self,torch.zeros(n,device=self.device,dtype=torch.bool),
                                     torch.zeros(len(base['xyz']),device=self.device,dtype=torch.bool))
        self.birth_transaction.old_render_reference=baseline.raw
        baseline_errors=birth_validation_errors(self,list(views.values()),baseline.raw)
        if n:
            deleted=torch.zeros(len(base['xyz']),device=self.device,dtype=torch.bool)
            oldrows=self.birth_transaction.replacements;deleted[oldrows[oldrows>=0]]=True
            canonical=encode_birth_trial(self,torch.ones(n,device=self.device,dtype=torch.bool),deleted)
            physical={name:value[len(canonical.keep):].clone() for name,value in canonical.raw.items()}
            model.rebind(model.uids,physical,old_rows=oldrows,frame_index=self.current_frame)
            replay=encode_birth_trial(self,torch.ones(n,device=self.device,dtype=torch.bool),deleted)
            if not all(torch.equal(canonical.raw[name],replay.raw[name]) for name in RAW_FIELDS):
                raise ValueError('Persistent SGD geometry is not fixed at its actual pose-codec canonical value')
        self.persistent_pretrain_state=model.bank_state()
        self.persistent_candidate_specs={spec['uid']:spec for spec in specs}
        self.persistent_training_population_count=n;self.persistent_birth_optimizer_steps=0
        self.persistent_initialized=True
        self.birth_cohort=0;self.birth_cohort_start=200;self.birth_end_step=300 if n else 200
        self.birth_terminal=False;self.coarse_birth_snapshot=[]
        self.birth_control.update(proposal_count=n,spatial_proposals=fresh,repair_proposals=n-fresh,
            selection_camera_scope='all_standard_training_cameras',source_mutated_before_acceptance=False,
            persistent_birth=dict(active_bank_count=len(active),scheduled_active_count=n-fresh,
                inactive_bank_count=len(active)-(n-fresh),fresh_count=fresh,training_population_count=n,
                geometry=geometry_report,baseline_all20_mse=baseline_errors.tolist(),
                initial_all20_mse=fitting_errors(self,list(views.values()),False).tolist(),
                model_count=1,optimizer_count=1,initializer_optimizer_updates=0,post_admission_optimizer_updates=0,
                total_steps=self.birth_end_step,SGD_initial_pose_codec_replay_exact=True))
        if not n:
            self._finish_persistent_births()
            recorder=getattr(self.trainer,'record_birth_comparison',None)
            if recorder is not None:recorder()
            self.finalize_frame()

    @torch.no_grad()
    def _persistent_optimizer_update_done(self,step):
        if step-200!=self.persistent_birth_optimizer_steps+1 or not 201<=step<=300:
            raise ValueError('Persistent bank permits one optimizer update per birth iteration, at most100')
        self.persistent_birth_optimizer_steps+=1

    @torch.no_grad()
    def _finish_persistent_births(self):
        end=self.total_steps
        if (end not in (200,300) or self.trainer.global_step!=end or
                self.persistent_birth_optimizer_steps!=end-200 or
                bool(self.persistent_training_population_count)!=(end==300)):
            raise ValueError('Persistent completion must match its actual single-model 200+0/100 ledger')
        views=list(self.transaction_camera_views().values());bank=self.births;tx=self.birth_transaction
        _,report=admit_residual_births(self,self.config.birth.residual,views,views,
            fitting_views=views,capture_prefixes=True,atomic_lineage=True)
        chosen=None;chosen_model=None;chosen_tx=None
        for witness in reversed(self._birth_admission_witnesses):
            self.births,self.birth_transaction=bank,tx
            original=CommitTrial.restore(witness['trial'],self.device)
            selected=selected_population(self,original)
            self._install_birth_transaction(selected)
            replay=encode_birth_trial(self,torch.ones(len(self.births._xyz),device=self.device,dtype=torch.bool),
                                       selected.provisional_deleted)
            if (all(torch.equal(original.raw[name],replay.raw[name]) for name in RAW_FIELDS) and
                    replay.complete_bytes<=report['complete_byte_budget']):
                chosen=replay;chosen_model=self.births;chosen_tx=self.birth_transaction;break
        if chosen is None:raise ValueError('Persistent incumbent failed its actual-codec physical replay')
        # Build the next encoder bank from the *actual* committed population.
        selected_uids={bank.uids[int(row)] for row in original.candidates}
        inverse=torch.full((len(self.gaussians._xyz),),-1,device=self.device,dtype=torch.long)
        inverse[chosen.keep]=torch.arange(len(chosen.keep),device=self.device)
        layout={};committed={};live=[];rejected=[]
        newrow={uid:len(chosen.keep)+index for index,uid in enumerate(chosen_model.uids)}
        old_reference=tx.old_render_reference
        for index,uid in enumerate(bank.uids):
            parent=int(tx.replacements[index])
            if uid in selected_uids:
                row=newrow[uid];physical={name:value[row] for name,value in chosen.raw.items()}
            elif parent>=0 and int(inverse[parent])>=0:
                row=int(inverse[parent]);physical={name:value[parent] for name,value in old_reference.items()}
                rejected.append(uid)
            else:continue
            live.append(uid);layout[uid]=row;committed[uid]=physical
        bank.rollback_uids(rejected,self.persistent_pretrain_state)
        bank.commit(live,old_rows=torch.tensor([layout[uid] for uid in live],device=self.device,dtype=torch.long))
        physical={name:torch.stack([committed[uid][name] for uid in live]) if live else
                  old_reference[name][:0] for name in RAW_FIELDS}
        bank.rebind(live,physical,old_rows=torch.tensor([layout[uid] for uid in live],device=self.device,dtype=torch.long),
                    frame_index=self.current_frame)
        updated=bank.bank_state()
        # Inactive bank identities were never part of the train pool. Their
        # original rows and Adam state survive; only actual decoded deaths or
        # accepted 1:1 updates alter this mapping.
        all_layout={uid:int(inverse[parent]) for uid,parent in self.persistent_uid_layout.items()
                    if int(inverse[parent])>=0}
        all_layout.update(layout)
        all_uids=sorted(all_layout,key=lambda uid:all_layout[uid])
        physical_all={name:value[torch.tensor([all_layout[uid] for uid in all_uids],
                      device=value.device,dtype=torch.long)].detach().cpu() for name,value in chosen.raw.items()}
        self.persistent_bank_state=type(bank).merge_bank_states(self.persistent_bank_state,updated,
            all_uids,physical_all,[all_layout[uid] for uid in all_uids],frame_index=self.current_frame)
        self.persistent_uid_layout=all_layout
        metadata={**self.persistent_specs,**self.persistent_candidate_specs}
        self.persistent_specs={uid:metadata[uid] for uid in all_uids}
        self.births,self.birth_transaction=chosen_model,chosen_tx
        committed_errors=birth_validation_errors(self,views,chosen.raw)
        self._commit_population_trial(chosen)
        self._birth_admission_witnesses=None
        self.birth_control['birth_transaction']=report
        self.birth_control['persistent_birth'].update(committed_all20_mse=committed_errors.tolist(),
            live_bank_count=len(all_uids),trained_bank_retained=len(live),accepted_wire_updates=len(chosen.candidates),
            accepted_fresh=sum(int(tx.replacements[int(row)])<0 for row in original.candidates),
            accepted_active=sum(int(tx.replacements[int(row)])>=0 for row in original.candidates),
            active_unchanged_or_rejected=len(rejected),
            birth_optimizer_updates=self.persistent_birth_optimizer_steps,total_steps=end,
            physical_codec_replay_exact=True,actual_training_steps=dict(motion=200,birth=end-200,total=end),
            semantics='one persistent UID pool; rejected updates retain original row; no repeated coverage or extra refit')

    @torch.no_grad()
    def _initialize_quick_births(self,views):
        from .model.child_dc_initializer import initialize_child_dc_once
        if self.quick_birth_initialized:
            raise ValueError('Quick child DC initialization may run only once per frame')
        if self.trainer.global_step!=200 or self.total_steps!=300 or getattr(self,'birth_branches',None) is not None or len(views)!=20:
            raise ValueError('Quick births require a single model at step200 with fixed total300')
        self.births.set_training_stage(1)
        mask = torch.ones(len(self.births._xyz),dtype=torch.bool,device=self.device)
        deleted = torch.zeros(len(self.gaussians._xyz),dtype=torch.bool,device=self.device)
        parents = self.birth_transaction.replacements[mask];deleted[parents[parents>=0]]=True
        reference = None
        def validate_scene():
            nonlocal reference
            trial = encode_birth_trial(self,mask,deleted)
            if reference is None:
                reference = trial
            elif not torch.equal(trial.keep,reference.keep) or not torch.equal(trial.sources,reference.sources):
                raise ValueError('One-shot initializer changed its actual codec population/source mapping')
            errors = birth_validation_errors(self,list(views.values()),trial.raw)
            return dict(mse=float(errors.mean()),per_view_mse=errors.tolist(),codec_bytes=trial.complete_bytes)
        report = initialize_child_dc_once(self,views,validate_scene=validate_scene)
        self.quick_birth_initialized = True
        self.birth_control['quick_initializer'] = report
        self.birth_control['quick_schedule'] = dict(model_count=1,birth_optimizer_count=1,
            training_camera_scope='all20',motion_steps=200,birth_steps=100,total_steps=300,
            protected_global_steps=[201,250],competition_global_steps=[251,300],
            common_fit_steps=0,MVS_candidates=0,post_validation_SGD_steps=0)

    @torch.no_grad()
    def _initialize_shadow_births(self,views):
        """Birth is an accepted finite-density scene change, never a dense trial pool."""
        from .model.shadow_birth_initializer import propose_shadow_family
        if (self.quick_birth_initialized or self.trainer.global_step!=200 or
                self.total_steps!=300 or len(views)!=20 or getattr(self,'birth_branches',None) is not None):
            raise ValueError('Shadow initialization requires one uninitialized all20 model at step200')
        model,tx=self.births,self.birth_transaction
        model.set_training_stage(1)
        accepted=torch.zeros(len(model._xyz),dtype=torch.bool,device=self.device)
        deleted=torch.zeros(len(self.gaussians._xyz),dtype=torch.bool,device=self.device)
        best=encode_birth_trial(self,accepted,deleted)
        errors=birth_validation_errors(self,list(views.values()),best.raw)
        baseline=errors.clone()
        config=self.config.birth.residual
        available=(config.birth_budget_bytes if self.current_frame==1 else
                   min(config.birth_budget_bytes,max(2048,config.target_frame_bytes-best.complete_bytes)))
        budget=best.complete_bytes+available
        records=[]
        roots=sorted(range(len(model.reincarnation_groups)),
                     key=lambda root:float(tx.scores[model.reincarnation_groups[root]].sum()),reverse=True)
        for root in roots:
            rows=model.reincarnation_groups[root]
            if len(rows)!=3:
                continue
            initial_dc=model._features_dc[rows[1:]].detach().clone()
            mask=accepted.clone();mask[rows]=True
            trial_deleted=deleted.clone()
            parents=tx.replacements[mask];trial_deleted[parents[parents>=0]]=True
            cache={}
            def density_trial(rho):
                # The zero endpoint is the exact already accepted codec scene;
                # it does not append a duplicate mother or reorder a depth tie.
                if float(rho)==0.:
                    model.clear_shadow_probe()
                    return best
                model.set_shadow_probe(root,float(rho))
                key=(float(rho),model._features_dc._version)
                if key not in cache:
                    cache[key]=encode_birth_trial(self,mask,trial_deleted)
                return cache[key]
            def probe_scene():
                trial=density_trial(1.)
                selected=mask.nonzero(as_tuple=True)[0]
                local=torch.searchsorted(selected,rows[1:])
                return trial.raw,local+len(trial.keep)
            def render_density(view,rho):
                trial=density_trial(rho)
                previous=self.renderer.gaussians
                collect=self.renderer.collect_contributions
                try:
                    self.renderer.collect_contributions=False
                    self.renderer.gaussians=RawView(trial.raw,self.config.sh_degree,self.gaussians)
                    return self.renderer(view,self.background)['render']
                finally:
                    self.renderer.gaussians=previous
                    self.renderer.collect_contributions=collect
            model.set_shadow_probe(root,1.)
            try:
                proposal=propose_shadow_family(self,views,root,render_density,probe_scene)
                rho=float(proposal['rho'])
                trial=density_trial(rho) if math.isfinite(rho) and 0.<rho<=1. else None
                candidate_errors=(birth_validation_errors(self,list(views.values()),trial.raw)
                                  if trial is not None else None)
                legal=bool(trial is not None and trial.complete_bytes<=budget and
                    len(trial.raw['xyz'])<=config.max_gaussians and
                    torch.isfinite(candidate_errors).all() and candidate_errors.mean()<errors.mean())
                records.append(dict(root=root,parent=int(tx.replacements[rows[0]]),rho=rho,
                    accepted=legal,before_mean_mse=float(errors.mean()),
                    candidate_mean_mse=float(candidate_errors.mean()) if candidate_errors is not None else None,
                    codec_bytes=trial.complete_bytes if trial is not None else None,proposal=proposal))
                if legal:
                    model.accept_shadow_family(root,rho)
                    accepted,best,errors,deleted=mask,trial,candidate_errors,trial_deleted
                else:
                    model._features_dc[rows[1:]]=initial_dc
                    model._clear_reincarnation_cache()
            finally:
                model.clear_shadow_probe()
        self.quick_birth_initialized=True
        self.birth_control['shadow_initializer']=dict(method='sequential native-responsibility source families',
            baseline_mse=baseline.tolist(),initialized_mse=errors.tolist(),codec_bytes=best.complete_bytes,
            complete_byte_budget=budget,accepted_families=sum(item['accepted'] for item in records),
            accepted_points=int(accepted.sum()),families=records,optimizer_steps=0,SGD_steps=0,
            hidden_gradient_loops=False,training_camera_scope='all20',unaccepted_children='transparent')
        self.birth_control['quick_schedule']=dict(model_count=1,birth_optimizer_count=1,
            training_camera_scope='all20',motion_steps=200,birth_steps=100 if bool(accepted.any()) else 0,
            total_steps=300 if bool(accepted.any()) else 200,protected_global_steps=[201,250],
            competition_global_steps=[251,300],common_fit_steps=0,MVS_candidates=0,post_validation_SGD_steps=0)
        # Remove every rejected shadow before SGD, and train with the exact
        # decoded surviving predictor of the accepted real-codec population.
        selected=selected_population(self,best)
        self._install_birth_transaction(selected)
        replay=encode_birth_trial(self,torch.ones(len(self.births._xyz),dtype=torch.bool,device=self.device),
                                  selected.provisional_deleted)
        if not all(torch.equal(replay.raw[name],best.raw[name]) for name in RAW_FIELDS):
            raise ValueError('Shadow SGD population does not replay its accepted initialization')
        self.birth_control['shadow_initializer']['SGD_initial_codec_replay_exact']=True
        # One trajectory's accepted starting point, not a second trained model.
        self.shadow_initialized_state=dict(pool=self._capture_birth_pool(),trial=replay.capture(),
            mse=errors.tolist(),budget=budget)
        if not bool(accepted.any()):
            self.birth_end_step=200
            self.birth_control['quick_schedule']['early_stop']='no beneficial finite-density family'
            self._finish_shadow_births()
            recorder=getattr(self.trainer,'record_birth_comparison',None)
            if recorder is not None:recorder()
            self.finalize_frame()

    @torch.no_grad()
    def _finish_shadow_births(self):
        end=self.total_steps
        if end not in (200,300) or self.trainer.global_step!=end or self.quick_birth_optimizer_steps!=end-200:
            raise ValueError('Shadow completion permits only motion200 or motion200 + single birth100')
        views=list(self.transaction_camera_views().values())
        # SGD used a fixed surviving predictor. Final per-family withdrawal
        # must restore the actual original mother, not leave a fixed-ref hole.
        tx=self.birth_transaction
        if tx.fixed_reference is not None:
            tx.sources=tx.fixed_keep[tx.sources]
            self.births.source_rows=tx.sources.clone()
            tx.fixed_reference=None;tx.fixed_keep=None;tx.provisional_deleted=None
        _,report=admit_residual_births(self,self.config.birth.residual,views,views,
            fitting_views=views,capture_prefixes=True,atomic_lineage=True)
        full_model,full_transaction=self.births,self.birth_transaction
        initial=self.shadow_initialized_state
        witnesses=[dict(pool=None,trial=item['trial'],kind='trained_prefix')
                   for item in reversed(self._birth_admission_witnesses)]
        witnesses.append(dict(pool=initial['pool'],trial=initial['trial'],kind='accepted_initialization'))
        baseline=self.gaussians._xyz.new_tensor(report['validation_baseline_mse']).mean()
        best_value=float('inf');chosen=None;chosen_state=None;chosen_kind=None;chosen_errors=None;checks=[]
        legal_trained=False
        self._quick_finalizing=True
        try:
            for witness in witnesses:
                if witness['kind']=='trained_prefix' and legal_trained:
                    continue
                if witness['pool'] is None:
                    self.births,self.birth_transaction=full_model,full_transaction
                else:
                    self._restore_birth_pool(witness['pool'])
                original=CommitTrial.restore(witness['trial'],self.device)
                transaction=selected_population(self,original)
                self._install_birth_transaction(transaction)
                replay=encode_birth_trial(self,torch.ones(len(self.births._xyz),dtype=torch.bool,device=self.device),
                                           transaction.provisional_deleted)
                exact=all(torch.equal(replay.raw[name],original.raw[name]) for name in RAW_FIELDS)
                errors=birth_validation_errors(self,views,replay.raw)
                value=float(errors.mean())
                legal=bool(exact and torch.isfinite(errors).all() and errors.mean()<=baseline+1.e-10 and
                    replay.complete_bytes<=initial['budget'])
                checks.append(dict(kind=witness['kind'],candidate_count=len(replay.candidates),
                    exact_codec_replay=exact,mse=value,codec_bytes=replay.complete_bytes,accepted=legal))
                if legal and witness['kind']=='trained_prefix':legal_trained=True
                if legal and value<best_value:
                    chosen=replay;chosen_state=self._capture_birth_pool();best_value=value
                    chosen_kind=witness['kind'];chosen_errors=errors.tolist()
            if chosen is None:
                raise ValueError('Shadow incumbent and accepted initialization both failed exact codec replay')
            self._restore_birth_pool(chosen_state)
            self._commit_population_trial(chosen)
        finally:
            self._quick_finalizing=False
        self._birth_admission_witnesses=None
        report['pre_rollback_admission_mse']=report['validation_committed_mse']
        report['validation_committed_mse']=chosen_errors
        report['retirement_committed_mse']=chosen_errors
        report['accepted']=len(chosen.candidates)
        report['trial_complete_bytes']=chosen.complete_bytes
        self.birth_control['birth_transaction']=report
        self.birth_control['quick_committed']=dict(candidate_count=len(chosen.candidates),
            actual_training_steps=dict(motion=200,birth_iterations=end-200,
                birth_optimizer_updates=self.quick_birth_optimizer_steps,total=end),
            replay_checks=checks,post_admission_SGD_steps=0,
            selected_trajectory_state=chosen_kind,committed_mse=chosen_errors,
            selection_objective='per-source-family all20 mean MSE and actual codec bytes',
            material='single shadow-born trajectory with accepted initialization rollback')

    @torch.no_grad()
    def _quick_optimizer_update_done(self,step):
        expected = step-self.motion_steps
        if not 1<=expected<=100 or self.quick_birth_optimizer_steps!=expected-1:
            raise ValueError('Quick births permit exactly one optimizer update in each of the 100 birth steps')
        self.quick_birth_optimizer_steps += 1
        if step==250:
            self.births.set_training_stage(2)
            # Same single Adam instance; newly released parent-alpha rows must
            # not inherit a 50-step bias correction from already-trained rows.
            self.births.optimizer.state.clear()
            self.quick_birth_phase = 2
            self.birth_control['quick_schedule']['competition_optimizer_state_reset_at'] = 250

    @torch.no_grad()
    def _finish_quick_births(self):
        if self.trainer.global_step!=300 or self.total_steps!=300 or getattr(self,'birth_branches',None) is not None:
            raise ValueError('Quick final admission must follow exactly motion200 + birth100 in one model')
        trained_count = len(self.births._xyz)
        if trained_count and self.quick_birth_optimizer_steps!=100:
            raise ValueError('Quick model did not receive its complete, bounded 100 optimizer updates')
        views = list(self.transaction_camera_views().values())
        _,report = admit_residual_births(self,self.config.birth.residual,views,views,
            fitting_views=views,capture_prefixes=True)
        full_model,full_transaction = self.births,self.birth_transaction
        baseline = self.gaussians._xyz.new_tensor(report['validation_baseline_mse'])
        candidates = self._birth_admission_witnesses
        chosen = None
        self._quick_finalizing = True
        replay_checks = []
        try:
            for witness in reversed(candidates):
                self.births,self.birth_transaction = full_model,full_transaction
                original = CommitTrial.restore(witness['trial'],self.device)
                transaction = selected_population(self,original)
                self._install_birth_transaction(transaction)
                replay = encode_birth_trial(self,torch.ones(len(self.births._xyz),dtype=torch.bool,device=self.device),
                                            transaction.provisional_deleted)
                exact = all(torch.equal(replay.raw[name],original.raw[name]) for name in RAW_FIELDS)
                errors = birth_validation_errors(self,views,replay.raw)
                legal = bool(exact and replay.complete_bytes<=report['complete_byte_budget'] and
                             errors.mean()<=baseline.mean()+1.e-10)
                replay_checks.append(dict(candidate_count=len(original.candidates),exact_codec_replay=exact,
                    mse=float(errors.mean()),codec_bytes=replay.complete_bytes,accepted=legal))
                if legal:
                    chosen = replay
                    break
            if chosen is None:
                raise ValueError('Even the unmodified quick incumbent failed actual-codec prefix replay')
            # replay candidates are already 0..selected_count-1 against the
            # fixed decoded surviving predictor. Only now physically retire.
            self._commit_population_trial(chosen)
        finally:
            self._quick_finalizing = False
        self._birth_admission_witnesses = None
        self.birth_control['birth_transaction'] = report
        self.birth_control['quick_committed'] = dict(candidate_count=len(chosen.candidates),
            actual_training_steps=dict(motion=200,birth_iterations=100,birth_optimizer_updates=self.quick_birth_optimizer_steps,total=300),
            replay_checks=replay_checks,post_admission_SGD_steps=0,selection_objective='all20 mean MSE and actual codec bytes',
            material='single trained theta0 mother plus child D; no branch or global colour calibration')

    def _capture_birth_pool(self):
        return dict(model=BirthTransaction.tensor_tree(self.births.capture(),'cpu'),
                    transaction=self.birth_transaction.capture(),cohort=self.birth_cohort)

    def _activate_dual_branch(self,branch):
        packet = self.birth_branches[branch]
        self.births,self.birth_transaction = packet['model'],packet['transaction']
        self.birth_active_branch = branch

    def _activate_dual_for_view(self,view):
        name = Path(view.path).name
        groups = self.birth_transaction.crossfit_state['camera_groups']
        branches = [key for key,names in groups.items() if name in names]
        if len(branches)!=1:
            raise ValueError('Dual birth training requires one standard own-group camera')
        self._activate_dual_branch(branches[0])

    def _begin_dual_branches(self):
        initial = self.birth_transaction.capture()
        a = dict(model=self.births,transaction=self.birth_transaction)
        self._install_birth_transaction(BirthTransaction.restore(initial,self.device))
        self.birth_branches = dict(A=a,B=dict(model=self.births,transaction=self.birth_transaction))
        self.dual_coarse_branches = {}
        self.dual_training_totals = {}
        self._activate_dual_branch('A')
        self.birth_control['dual_protocol'] = dict(camera_groups=self.birth_transaction.crossfit_state['camera_groups'],
            phases=['native_parent_radiance_N_inactive','residual_details_and_N_parent_frozen',
                    'mature_details_and_alpha_competition_N_geometry_fixed'],
            endpoint_scope='P_whole_family_only_N_current_constant',
            initialization_scope='shared all20 MVS proposal; conditional cross-calibration, not unseen initialization holdout')
        if uses_source_residual(self.config):
            self.birth_control['dual_protocol']['material_lineage'] = ('stage0 common is optional C; '
                'fine parent fixed at original source theta0, children independently learn D from theta0; '
                'final fine endpoints prior/common/detail/full, coarse retains DC/SH/full')

    def _capture_dual_checkpoint(self):
        if self.frame_finalized or not getattr(self,'birth_branches',None):
            return None
        active = self.birth_active_branch
        branches = {}
        for branch in ('A','B'):
            self._activate_dual_branch(branch)
            branches[branch] = self._capture_birth_pool()
        self._activate_dual_branch(active)
        return dict(active=active,branches=branches,coarse=self.dual_coarse_branches,
                    training_totals=BirthTransaction.tensor_tree(self.dual_training_totals,'cpu'))

    def _restore_dual_checkpoint(self,state):
        self.birth_branches = None
        self.dual_coarse_branches = None
        self.dual_training_totals = None
        if state is None:
            if self.motion_committed and not self.frame_finalized:
                raise ValueError('Dual birth resume lost its independent model/optimizer branches')
            return
        branches = {}
        for name,packet in state['branches'].items():
            self._restore_birth_pool(packet)
            branches[name] = dict(model=self.births,transaction=self.birth_transaction)
        if set(branches)!=set(('A','B')):
            raise ValueError('Dual birth resume requires both complete branches')
        self.birth_branches = branches
        self.dual_coarse_branches = state['coarse']
        self.dual_training_totals = BirthTransaction.tensor_tree(state['training_totals'],self.device)
        self._activate_dual_branch(state['active'])

    @torch.no_grad()
    def _post_dual_birth_step(self,packages,step):
        for branch in ('A','B'):
            self._activate_dual_branch(branch)
            items = [item for item in packages if item.get('birth_branch')==branch]
            if not items:
                continue
            if branch not in self.dual_training_totals:
                self.dual_training_totals[branch] = dict(views=0,loss=self.gaussians._xyz.new_zeros(()),
                    psnr=self.gaussians._xyz.new_zeros(()),observed=self.gaussians._xyz.new_zeros(3),
                    initial_observed=None,last_observed=None)
            totals = self.dual_training_totals[branch]
            count = len(self.births._xyz)
            detail = self.births.detail_mask
            novel = self.births.novel_mask
            for item in items:
                totals['views'] += 1
                totals['loss'] += item['birth_image_loss']
                totals['psnr'] += item['birth_psnr']
                if count:
                    visible = item['visibility_filter'][-count:]
                    self.births.max_radii2D[visible] = torch.maximum(self.births.max_radii2D[visible],item['radii'][-count:][visible])
                    observed = item['motion_contribution_filter'][-count:]
                    self.birth_transaction.observations[observed] |= 1<<item['birth_camera_index']
                    sample = torch.stack((observed[~(detail|novel)].sum(),observed[detail].sum(),observed[novel].sum()))
                    totals['observed'] += sample
                    if totals['initial_observed'] is None:
                        totals['initial_observed'] = sample.clone()
                    totals['last_observed'] = sample.clone()
            if count:
                self.births.optimizer.step()
                self.births.optimizer.zero_grad(set_to_none=True)
                self.births.project_parameters()
                if self.birth_cohort>0:
                    rows = (detail|novel if self.birth_cohort==1 else detail).nonzero(as_tuple=True)[0]
                    floor = math.sqrt(3)*.5*self.config.rigid_tree.gs_xyz_step_extent*self.datamodule.cameras_extent
                    constrain_candidates(self.births,self.birth_transaction,floor,rows=rows)
        if step!=self.total_steps:
            return
        stage = dict(index=self.birth_cohort,start_step=self.birth_cohort_start,end_step=step,branches={},
                     optimizer_reset='independent whole-pool Adam at each stage boundary')
        for branch in ('A','B'):
            totals = self.dual_training_totals.get(branch)
            if totals:
                stage['branches'][branch] = dict(own_view_count=totals['views'],
                    mean_image_loss=float(totals['loss']/max(1,totals['views'])),
                    mean_psnr=float(totals['psnr']/max(1,totals['views'])),
                    native_observed_parent_detail_N_point_views=totals['observed'].tolist(),
                    initial_native_observed_parent_detail_N=(totals['initial_observed'].tolist() if totals['initial_observed'] is not None else [0,0,0]),
                    mature_native_observed_parent_detail_N=(totals['last_observed'].tolist() if totals['last_observed'] is not None else [0,0,0]),
                    semantics='native actual contribution bits during existing training renders, not projected footprint area')
        self.birth_control.setdefault('dual_cohorts',[]).append(stage)
        if self.birth_cohort<2:
            self._advance_dual_stage(step)
            return
        self._finish_dual_births()
        recorder = getattr(self.trainer,'record_birth_comparison',None)
        if recorder is not None:
            recorder()
        self.finalize_frame()

    @torch.no_grad()
    def _advance_dual_stage(self,step):
        from .model.residual_birth import _reincarnation_transaction,_copy_novel_raw
        next_branches = {}
        for branch in ('A','B'):
            self._activate_dual_branch(branch)
            if self.birth_cohort==0:
                self.dual_coarse_branches[branch] = self._capture_birth_pool()
                transaction = self.birth_transaction
                roots = [i for i,spec in enumerate(transaction.reincarnation_specs) if spec.get('split_eligible',False)]
                roots = roots[:max(0,(self.config.birth.residual.max_candidates-len(self.births._xyz))//2)]
                alpha = self.config.birth.initial_alpha
                model = self.births.install_details_from_coarse(roots,detail_alpha=.1 if alpha is None else alpha)
                _copy_novel_raw(model,transaction.crossfit_state['novel_initial_raw'])
                model.set_training_stage(1)
                state = dict(transaction.crossfit_state);state['eligible']=True
                refined = _reincarnation_transaction(self,model,transaction.reincarnation_specs,transaction,
                    state,dict(transaction.region_diagnostics),previous=transaction)
                refined.predictive_model = True
                self._install_birth_transaction(refined)
            else:
                self.births.set_training_stage(2)
                self.births.initialize_incremental_optimizer(self.config.gaussian_stage2)
            next_branches[branch] = dict(model=self.births,transaction=self.birth_transaction)
        self.birth_branches = next_branches
        self.birth_cohort += 1
        self.birth_cohort_start = step
        self.birth_end_step = step+self.config.rigid_tree.birth_steps
        self.dual_training_totals = {}
        self._activate_dual_branch('A')

    @torch.no_grad()
    def _calibrate_dual_population(self,branch,mask,deleted):
        """Cross-calibrate frozen endpoints in one exact codec population."""
        from .model.predictive_colour import calibrate_predictive_colour
        model = self.births
        model.freeze_predictive_endpoints()
        views = self.transaction_camera_views()
        names = list(views)
        groups = self.birth_transaction.crossfit_state['camera_groups']
        opposite = 'B' if branch=='A' else 'A'
        calibration = {name:views[name] for name in groups[opposite]}
        endpoint_kinds = getattr(model,'endpoint_kinds',('prior','dc','sh','full'))
        endpoint_kinds = tuple(endpoint_kinds() if callable(endpoint_kinds) else endpoint_kinds)
        vertices = dict(zip(endpoint_kinds,((0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.))))
        packets = {}
        reference = None
        def trial_at(weights):
            nonlocal reference
            key = tuple(float(x) for x in weights)
            if key not in packets:
                previous = (model.predictive_weights.clone() if model.predictive_weights is not None
                            else model._xyz.new_tensor([0.,0.,1.]))
                model.apply_predictive_weights(weights)
                trial = encode_birth_trial(self,mask,deleted)
                if reference is None:
                    reference = trial
                elif not torch.equal(trial.keep,reference.keep) or not torch.equal(trial.sources,reference.sources):
                    raise ValueError('Colour calibration changed its population/surviving predictor')
                for field in ('xyz','rotation','logscale','opacity'):
                    if reference is not trial and not torch.equal(trial.raw[field],reference.raw[field]):
                        raise ValueError('Colour calibration changed frozen geometry or alpha')
                packets[key] = trial
                model.apply_predictive_weights(previous)
            return packets[key]
        def render_endpoint(view,kind):
            trial = trial_at(vertices[kind])
            previous,collect = self.renderer.gaussians,self.renderer.collect_contributions
            try:
                self.renderer.collect_contributions = False
                self.renderer.gaussians = RawView(trial.raw,self.config.sh_degree,self.gaussians)
                return self.renderer(view,self.background)['render']
            finally:
                self.renderer.gaussians,self.renderer.collect_contributions = previous,collect
        def validate(weights):
            trial = trial_at(weights)
            errors = birth_validation_errors(self,list(calibration.values()),trial.raw)
            # Legality is separate from colour-control/incumbent improvement.
            return dict(mse=float(errors.mean()),eligible=bool(torch.isfinite(errors).all()),
                        codec_bytes=trial.complete_bytes,per_view_mse=errors.tolist())
        report = calibrate_predictive_colour(calibration,render_endpoint,
            model.apply_predictive_weights,validate,endpoint_kinds=endpoint_kinds)
        trial = trial_at(report['weights'])
        control = trial_at(vertices['prior'])
        empty = torch.zeros(len(model._xyz),dtype=torch.bool,device=self.device)
        baseline = encode_birth_trial(self,empty,torch.zeros(len(self.gaussians._xyz),dtype=torch.bool,device=self.device))
        actual = birth_validation_errors(self,list(views.values()),trial.raw)
        prior = birth_validation_errors(self,list(views.values()),control.raw)
        incumbent = birth_validation_errors(self,list(views.values()),baseline.raw)
        folds = {}
        for key,cameras in groups.items():
            ids = [names.index(name) for name in cameras]
            folds[key] = dict(final_mean_mse=float(actual[ids].mean()),
                colour_control_mean_mse=float(prior[ids].mean()),incumbent_mean_mse=float(incumbent[ids].mean()),
                improves_incumbent=bool(actual[ids].mean()<incumbent[ids].mean()-1.e-10),
                no_worse_than_colour_control=bool(actual[ids].mean()<=prior[ids].mean()+1.e-10))
        report.update(source_branch=branch,calibration_cameras=list(calibration),
            endpoint_scope='P_whole_family_only_N_current_constant',population_count=int(mask.sum()),
            survivor_count=len(trial.keep),folds=folds,codec_bytes=trial.complete_bytes,
            final_per_view_mse=actual.tolist(),colour_control_per_view_mse=prior.tolist(),
            incumbent_per_view_mse=incumbent.tolist(),
            certificate=all(x['improves_incumbent'] and x['no_worse_than_colour_control'] for x in folds.values()),
            semantics='conditional cross-calibration/model selection; same actual codec population; no SGD follows')
        return trial,report

    @torch.no_grad()
    def _dual_N_increment(self,trial):
        n = self.birth_transaction.proposed_spatial
        selected = trial.candidates
        if not bool((selected<n).any()):
            return dict(present=False,accepted=True)
        mask = torch.zeros(len(self.births._xyz),dtype=torch.bool,device=self.device)
        mask[selected] = True
        if not bool(mask[:n].all()):
            raise ValueError('Independent N hypothesis was split after its transfer test')
        mask[:n] = False
        deleted = torch.ones(len(self.gaussians._xyz),dtype=torch.bool,device=self.device);deleted[trial.keep]=False
        without = encode_birth_trial(self,mask,deleted)
        views = self.transaction_camera_views();names=list(views)
        before = birth_validation_errors(self,list(views.values()),without.raw)
        after = birth_validation_errors(self,list(views.values()),trial.raw)
        folds = {}
        for key,cameras in self.birth_transaction.crossfit_state['camera_groups'].items():
            rows = [names.index(name) for name in cameras]
            folds[key] = dict(without_N_mean_mse=float(before[rows].mean()),
                with_N_mean_mse=float(after[rows].mean()),improves=bool(after[rows].mean()<before[rows].mean()-1.e-10))
        return dict(present=True,accepted=all(x['improves'] for x in folds.values()),folds=folds,
            semantics='actual N increment on identical calibrated P/old-survivor population; both standard camera groups')

    @torch.no_grad()
    def _finish_dual_births(self):
        """Select one complete calibrated branch; no geometry or colour average."""
        fine = {}
        for branch in ('A','B'):
            self._activate_dual_branch(branch)
            fine[branch] = self._capture_birth_pool()
        options = []
        winner = None
        views = self.transaction_camera_views()
        for branch in ('A','B'):
            for kind,snapshot in (('coarse',self.dual_coarse_branches[branch]),('residual',fine[branch])):
                self._restore_birth_pool(snapshot)
                self.birth_active_branch = branch
                n = self.birth_transaction.proposed_spatial
                self.birth_transaction.crossfit_state['eligible'] = kind=='residual'
                full = torch.ones(len(self.births._xyz),dtype=torch.bool,device=self.device)
                if kind=='coarse':full[:n]=False
                deleted = torch.zeros(len(self.gaussians._xyz),dtype=torch.bool,device=self.device)
                parents = self.birth_transaction.replacements[full];deleted[parents[parents>=0]]=True
                _,initial = self._calibrate_dual_population(branch,full,deleted)
                candidate,evidence = admit_residual_births(self,self.config.birth.residual,
                    list(views.values()),list(views.values()),fitting_views=list(views.values()))
                mask = torch.zeros(len(self.births._xyz),dtype=torch.bool,device=self.device);mask[candidate.candidates]=True
                deleted = torch.ones(len(self.gaussians._xyz),dtype=torch.bool,device=self.device);deleted[candidate.keep]=False
                actual,certificate = self._calibrate_dual_population(branch,mask,deleted)
                increment = self._dual_N_increment(actual)
                if increment['present'] and not increment['accepted']:
                    # Remove an untransferable N population, restore its absence,
                    # and re-prove P in the actual P-only selected structure.
                    self.birth_transaction.crossfit_state['eligible'] = False
                    p_only = torch.ones(len(self.births._xyz),dtype=torch.bool,device=self.device);p_only[:n]=False
                    p_deleted = torch.zeros(len(self.gaussians._xyz),dtype=torch.bool,device=self.device)
                    p_parents = self.birth_transaction.replacements[p_only];p_deleted[p_parents[p_parents>=0]]=True
                    _,p_recalibration = self._calibrate_dual_population(branch,p_only,p_deleted)
                    candidate,evidence = admit_residual_births(self,self.config.birth.residual,
                        list(views.values()),list(views.values()),fitting_views=list(views.values()))
                    mask.zero_();mask[candidate.candidates]=True
                    deleted.fill_(True);deleted[candidate.keep]=False
                    actual,certificate = self._calibrate_dual_population(branch,mask,deleted)
                    increment = dict(rejected_N=increment,P_only_pool_recalibration=p_recalibration,
                                     **self._dual_N_increment(actual))
                legal = bool(certificate['certificate'] and increment['accepted'] and
                    actual.complete_bytes<=evidence['complete_byte_budget'])
                entry = dict(branch=branch,kind=kind,initial_pool_calibration=initial,
                    actual_population_calibration=certificate,N_increment=increment,
                    accepted=legal,selected_count=len(actual.candidates),codec_bytes=actual.complete_bytes,
                    byte_budget=evidence['complete_byte_budget'],transaction=evidence)
                options.append(entry)
                mse = sum(certificate['final_per_view_mse'])/len(certificate['final_per_view_mse'])
                if legal and (winner is None or mse<winner['mse']-1.e-10):
                    winner = dict(mse=mse,branch=branch,kind=kind,trial=actual.capture(),
                        pool=self._capture_birth_pool(),proof=certificate,evidence=evidence)
        self.birth_control['dual_selection'] = options
        if winner is None:
            # Whole atomic rollback, including provisional source retirements.
            self._restore_birth_pool(fine['A'])
            empty = torch.zeros(len(self.births._xyz),dtype=torch.bool,device=self.device)
            trial = encode_birth_trial(self,empty,torch.zeros(len(self.gaussians._xyz),dtype=torch.bool,device=self.device))
            self.birth_control['dual_committed'] = dict(accepted=False,reason='no_actual_population_predictive_certificate')
        else:
            self._restore_birth_pool(winner['pool'])
            trial = CommitTrial.restore(winner['trial'],self.device)
            self.birth_control['dual_committed'] = dict(accepted=True,branch=winner['branch'],kind=winner['kind'],
                proof=winner['proof'],codec_bytes=trial.complete_bytes,no_training_after_calibration=True)
        transaction = selected_population(self,trial)
        normalized = CommitTrial(trial.keep,torch.arange(len(trial.candidates),device=self.device),trial.sources,
            trial.raw,trial.complete_bytes,trial.motion_bytes,trial.birth_bytes,trial.opacity_bytes,trial.old_symbols)
        self._install_birth_transaction(transaction)
        self._commit_population_trial(normalized)
        self.birth_branches = None
        self.dual_coarse_branches = None
        self.dual_training_totals = None

    def _set_patch_stage(self):
        cap = 1. if self.birth_terminal else .5
        if hasattr(self.births, 'set_retention_cap'):
            self.births.set_retention_cap(cap)
        else:
            self.births.patch_retention_cap = cap
            with torch.no_grad():
                self.births.patch_retention.clamp_(0, cap)
        self.births.set_radiance_views(self.transaction_camera_views())

    def _calibrate_birth_radiance(self, views):
        if not uses_intrinsic_birth(self.config):
            return
        from .model.birth_radiance import calibrate_intrinsic_dc
        dc, rest = self.births._features_dc, self.births._features_rest
        conservative = uses_conservative_birth(self.config)
        mask = self.birth_transaction.replacements < 0 if conservative else None
        report = calibrate_intrinsic_dc(self, views, point_mask=mask) if conservative else calibrate_intrinsic_dc(self, views)
        if conservative:
            families = getattr(self.births, 'family_rows', None)
            contrast_report = None
            if families is not None and len(families):
                from .model.family_radiance import calibrate_family_contrast_dc
                original_contrast = self.births.family_contrast
                contrast_report = calibrate_family_contrast_dc(self, views)
                if self.births.family_contrast is not original_contrast:
                    raise ValueError('Family contrast solver must preserve its leaf Parameter')
            report = dict(novel_intrinsic=report, family_contrast=contrast_report,
                          repair_common_dc_fixed=True)
        if self.births._features_dc is not dc or self.births._features_rest is not rest:
            raise ValueError('Intrinsic calibration must preserve optimizer parameter identity')
        freeze_birth_radiance(self.births, self.config)
        if conservative:
            report['quantized_family'] = family_quantization_diagnostics(self.births, self.make_birth_view())
        self.birth_control.setdefault('radiance_calibrations', []).append(dict(
            cohort=self.birth_cohort, step=self.trainer.global_step,
            phase='population_refit' if self.birth_terminal else 'exploration',
            radiance_mode=self.config.birth.radiance_mode, candidate_count=len(self.births._xyz),
            report=report))

    @torch.no_grad()
    def _advance_birth_cohort(self):
        views = self.transaction_camera_views()
        if uses_reincarnation(self.config) and self.birth_cohort == 1:
            n = self.birth_transaction.proposed_spatial
            self.birth_transaction.crossfit_state['novel_B_raw'] = {
                name:value[:n].clone() for name,value in model_raw(self.births).items()}
        if uses_intrinsic_birth(self.config):
            self._calibrate_birth_radiance(views)
        fitting = [views[name] for name in self.birth_transaction.fit_names]
        self.birth_control.setdefault('cohorts', []).append(dict(index=self.birth_cohort,
            start_step=self.birth_cohort_start, end_step=self.trainer.global_step,
            optimization_steps=self.trainer.global_step-self.birth_cohort_start,
            candidate_count=len(self.births._xyz),
            fitting_mse=fitting_errors(self, fitting, False).tolist()))
        if hasattr(self.births, 'patch_rows'):
            self.birth_control['cohorts'][-1]['patches'] = self._patch_training_diagnostics()
        if self.birth_cohort+1 >= self.config.birth.residual.max_cohorts or not len(self.births._xyz):
            return False
        transaction, report = refine_transaction(self, self.config.birth.residual, views)
        self.birth_control.setdefault('refinements', []).append(report)
        if transaction is None:
            return False
        self.coarse_birth_snapshot.append(dict(
            model=BirthTransaction.tensor_tree(self.births.capture(), 'cpu'),
            transaction=self.birth_transaction.capture(), cohort=self.birth_cohort))
        self._install_birth_transaction(transaction)
        self.birth_cohort += 1
        self.birth_cohort_start = self.trainer.global_step
        self.birth_end_step = self.birth_cohort_start+self.config.rigid_tree.birth_steps
        return True

    def _patch_training_diagnostics(self):
        transaction = self.birth_transaction
        observations = sum((transaction.observations >> index)&1 for index in range(len(transaction.fit_names)))
        return [dict(kernels=len(row), owned_old_count=len(old), retention=float(self.births.patch_retention[index]),
            exposure_cap=float(self.births.patch_retention_cap),
            newborns_observed=int((observations[row] > 0).sum()),
            newborns_in_multiple_views=int((observations[row] >= 2).sum()))
            for index,(row,old) in enumerate(zip(self.births.patch_rows,self.births.patch_old_rows))]

    @torch.no_grad()
    def _select_residual_births(self):
        views = self.transaction_camera_views()
        validation = [views[name] for name in self.birth_transaction.validation_names]
        fitting = [views[name] for name in self.birth_transaction.fit_names]
        fine_pool = self.births, self.birth_transaction
        crossfit_state = (BirthTransaction.tensor_tree(self.birth_transaction.crossfit_state,self.device)
                          if uses_reincarnation(self.config) else None)
        options = list(self.coarse_birth_snapshot or [])+[None]
        best_errors = None
        selection = []
        for snapshot in options:
            if snapshot is None:
                self.births, self.birth_transaction = fine_pool
                cohort = self.birth_cohort
            else:
                self._restore_birth_pool(snapshot)
                cohort = snapshot['cohort']
            crossfit_report = (qualify_crossfit_novel(self,views,crossfit_state)
                               if crossfit_state is not None else None)
            candidate, evidence = admit_residual_births(self, self.config.birth.residual,
                validation, list(views.values()), fitting_views=fitting)
            if crossfit_report is not None and crossfit_report['eligible']:
                n = self.birth_transaction.proposed_spatial
                if bool((candidate.candidates<n).any()):
                    mask = torch.zeros(len(self.births._xyz),dtype=torch.bool,device=self.device)
                    mask[candidate.candidates] = True
                    deleted = torch.ones(len(self.gaussians._xyz),dtype=torch.bool,device=self.device)
                    deleted[candidate.keep] = False
                    actual = qualify_crossfit_novel(self,views,crossfit_state,mask,deleted,
                                                     preferred_branch=crossfit_report['selected_branch'])
                    crossfit_report = dict(exploration_pool=crossfit_report,selected_population=actual)
                    if not actual['eligible']:
                        candidate, evidence = admit_residual_births(self,self.config.birth.residual,
                            validation,list(views.values()),fitting_views=fitting)
            errors = self.gaussians._xyz.new_tensor(evidence['validation_committed_mse'])
            # Each option has independently passed incumbent coverage/rate
            # guards. Choose the lower validation MSE among these legal scenes;
            # do not impose another per-camera Pareto guard between options.
            select = best_errors is None or bool(errors.mean() < best_errors.mean()-1.e-10)
            selection.append(dict(cohort=cohort, accepted=evidence['accepted'],
                independent_geometry=crossfit_report,
                validation_mse=errors.tolist(), selected_at_comparison=select,
                full_pool_fitting_mse=evidence['full_pool_fitting_mse'],
                full_pool_validation_mse=evidence['full_pool_validation_mse'],
                full_pool_codec_bytes=evidence.get('full_pool_codec_bytes'),
                accepted_trial_bytes=evidence.get('trial_complete_bytes'),
                full_pool_opacity_bytes=evidence.get('full_pool_opacity_bytes'),
                patch_takeover=evidence.get('patch_takeover'),
                full_pool_baseline_fitting_mse=evidence['full_pool_baseline_fitting_mse']))
            if select:
                best_errors = errors
                trial, report = candidate, evidence
                chosen_pool = self.births, self.birth_transaction
                chosen_cohort = cohort
        self.births, self.birth_transaction = chosen_pool
        self.birth_control['cohort_selection'] = selection
        self.birth_control['selected_cohort'] = chosen_cohort
        self.birth_control['fitting_trained_proposal_mse'] = fitting_errors(self, fitting, False).tolist()
        self.birth_control['candidate_rejected'] = len(self.births._xyz)-len(trial.candidates)
        self.birth_control['birth_transaction'] = report
        return trial, report

    @torch.no_grad()
    def _accept_residual_births(self):
        """Select structure now; physically commit only after population refit."""
        trial, report = self._select_residual_births()
        transaction = selected_population(self, trial)
        normalized = CommitTrial(trial.keep.clone(), torch.arange(len(trial.candidates), device=self.device),
            trial.sources.clone(), {name: value.clone() for name, value in trial.raw.items()},
            trial.complete_bytes, trial.motion_bytes, trial.birth_bytes, trial.opacity_bytes, trial.old_symbols)
        self.selected_birth_state = dict(transaction=transaction.capture(),
            trial=normalized.capture(), budget=report['complete_byte_budget'],
            baseline_mse=report['validation_baseline_mse'], selected_mse=report['validation_committed_mse'],
            selected_cohort=self.birth_control['selected_cohort'])
        self._install_birth_transaction(transaction)
        self.coarse_birth_snapshot = None
        if not len(transaction.sources):
            self.birth_control['population_refit'] = dict(skipped='no_selected_births',
                selected_mse=report['validation_committed_mse'], selected_bytes=trial.complete_bytes)
            self._commit_population_trial(normalized)
            return False
        self.birth_terminal = True
        set_family_training_stage(self.births, True)
        if hasattr(self.births, 'reincarnation_state'):
            self.births.set_training_stage(True)
        if hasattr(self.births, 'patch_retention'):
            self._set_patch_stage()
        self.birth_cohort += 1
        self.birth_cohort_start = self.trainer.global_step
        self.birth_end_step = self.birth_cohort_start+self.config.rigid_tree.birth_steps
        fitting = [self.transaction_camera_views()[name] for name in transaction.fit_names]
        self.birth_control['population_refit'] = dict(start_step=self.birth_cohort_start,
            end_step=self.birth_end_step, candidate_count=len(transaction.sources),
            provisional_deleted=int(transaction.provisional_deleted.sum()),
            predictor='fixed_decoded_survivor_reference',
            initial_forward_mse=fitting_errors(self, fitting, False).tolist(),
            selected_mse=report['validation_committed_mse'], selected_bytes=trial.complete_bytes)
        if hasattr(self.births, 'patch_rows'):
            self.birth_control['population_refit']['initial_patches'] = self._patch_training_diagnostics()
        return True

    @torch.no_grad()
    def _finish_population_refit(self):
        transaction = self.birth_transaction
        if uses_intrinsic_birth(self.config):
            self._calibrate_birth_radiance(self.transaction_camera_views())
        fitting = [self.transaction_camera_views()[name] for name in transaction.fit_names]
        selected = CommitTrial.restore(self.selected_birth_state['trial'], self.device)
        refitted = encode_birth_trial(self, torch.ones(len(self.births._xyz), dtype=torch.bool,
            device=self.device), transaction.provisional_deleted)
        errors = birth_validation_errors(self, fitting, refitted.raw)
        original = errors.new_tensor(self.selected_birth_state['baseline_mse'])
        selected_errors = errors.new_tensor(self.selected_birth_state['selected_mse'])
        use_refit = bool(refitted.complete_bytes <= self.selected_birth_state['budget'] and
            errors.mean() < original.mean()-1.e-10 and errors.mean() < selected_errors.mean()-1.e-10)
        self.birth_control.setdefault('cohorts', []).append(dict(index=self.birth_cohort,
            phase='population_refit', start_step=self.birth_cohort_start, end_step=self.trainer.global_step,
            optimization_steps=self.trainer.global_step-self.birth_cohort_start,
            candidate_count=len(self.births._xyz), full_codec_train_mse=errors.tolist()))
        if hasattr(self.births, 'patch_rows'):
            self.birth_control['cohorts'][-1]['patches'] = self._patch_training_diagnostics()
        self.birth_control['population_refit'].update(refit_mse=errors.tolist(),
            baseline_mse=original.tolist(), refit_bytes=refitted.complete_bytes,
            byte_budget=self.selected_birth_state['budget'], accepted_refit=use_refit,
            final_mse=(errors if use_refit else selected_errors).tolist(),
            final_bytes=refitted.complete_bytes if use_refit else selected.complete_bytes)
        if not use_refit:
            restored = BirthTransaction.restore(self.selected_birth_state['transaction'], self.device)
            self._install_birth_transaction(restored)
        self.birth_control['fitting_trained_proposal_mse'] = fitting_errors(self, fitting, False).tolist()
        self._commit_population_trial(refitted if use_refit else selected)

    @torch.no_grad()
    def _commit_population_trial(self, trial):
        """The only physical old-row retirement in the birth transaction."""
        if not torch.equal(trial.candidates, torch.arange(len(self.births._xyz), device=self.device)):
            raise ValueError('Population commit must preserve the selected complete subtree')
        if (not torch.equal(trial.keep, self.birth_transaction.fixed_keep) or
                not torch.equal(trial.sources, self.birth_transaction.sources)):
            raise ValueError('Population commit changed its fixed survivor predictor')
        delete = torch.ones(len(self.gaussians._xyz), dtype=torch.bool, device=self.device)
        delete[trial.keep] = False
        if trial.old_symbols is not None:
            self.opacity_update.delta.copy_(trial.old_symbols.to(self.opacity_update.delta)*self.opacity_update.step)
        self._commit_opacity_and_retire(forced_remove=delete)
        self.births.source_rows = trial.sources.to(self.device)
        self.motion.record_final_diagnostics()
        self.birth_control['committed_population_count'] = len(trial.candidates)
        self.birth_control['committed_trial_bytes'] = trial.complete_bytes

    @torch.no_grad()
    def _pin_birth_handoff_sources(self):
        """Drive newborn source predictors to exactly zero quantized alpha."""
        if self.birth_handoff_sources is None or not len(self.birth_handoff_sources):
            return
        update = self.opacity_update
        full_rows = update.rows[self.birth_handoff_sources]
        alpha_ref = update.alpha_reference[full_rows]
        # Ceil, not round: alpha_ref is continuous, so a rounded negated
        # target can leave a positive residual instead of clamping to zero.
        update.delta[full_rows] = -torch.ceil(alpha_ref / update.step) * update.step

    @torch.no_grad()
    def _begin_births(self):
        xyz, rotation = self.motion(self.motion_steps, training=False)
        self.gaussians.apply_selected_motion(xyz, rotation)
        self.motion_committed = True
        for parameter in self.motion.parameters():
            parameter.requires_grad_(False)
        self.motion.optimizer.zero_grad(set_to_none=True)
        if uses_persistent_birth(self.config):
            self._begin_persistent_births()
            return
        if uses_residual_birth(self.config):
            self._begin_residual_births()
            return
        count = len(self.gaussians._xyz)
        appearance_scores = (self.birth_appearance_gradient.norm(dim=-1)
            if self.birth_appearance_gradient is not None else None)
        if (appearance_scores is not None and
                self.config.birth.appearance_source_max_scale_extent is not None):
            max_source_scale = (
                self.config.birth.appearance_source_max_scale_extent
                * self.datamodule.cameras_extent)
            appearance_scores = appearance_scores.masked_fill(
                self.gaussians.get_scaling.amax(dim=1) > max_source_scale, 0)
        clone, source_rows = self.gaussians.adaptive_densify_and_clone(
            self.config.birth.densify_grad_threshold,
            self.datamodule.cameras_extent,
            sigma_scale=self.config.birth.clone_sigma_scale,
            max_new_gaussians=self.config.birth.max_new_gaussians,
            source_covariance_init=self.config.birth.source_covariance_init,
            return_source_rows=True,
            appearance_scores=appearance_scores,
            max_appearance_gaussians=self.config.birth.appearance_candidates)
        self.birth_appearance_gradient = None
        if clone['added'] != len(source_rows):
            raise AssertionError('Initial birth source rows disagree with clones')
        self.birth_control['clone_added'] = clone['added']
        self.birth_control['appearance_added'] = clone.get('appearance_added', 0)
        if clone['added']:
            self.births.restore(self.gaussians.capture(),
                                self.config.gaussian_stage2)
            new_mask = torch.arange(len(self.gaussians._xyz),
                                    device=self.device) >= count
            self.gaussians.prune_points(new_mask)
            self.births.prune_points(~new_mask)
            self.births.source_rows = source_rows
            if self.config.birth.source_handoff:
                for attr in RAW_ATTRIBUTES:
                    getattr(self.births, attr).copy_(
                        getattr(self.gaussians, attr)[source_rows])
                self.births._opacity.clamp_(-9.21024, 9.21024)
                self.birth_handoff_sources = source_rows.detach().clone()
                self._pin_birth_handoff_sources()
                self.gaussians._opacity.copy_(
                    self.opacity_update.canonical_logits().to(self.device))
                self.birth_control['source_handoff_count'] = len(source_rows)
            elif self.config.birth.initial_alpha is not None:
                alpha = self.config.birth.initial_alpha
                self.births._opacity.fill_(math.log(alpha / (1. - alpha)))
            if self.config.birth.source_position_limit_sigma is not None:
                constrain_birth_positions(self.births, self.gaussians,
                    self.config.birth.source_position_limit_sigma,
                    self.config.birth.source_position_min_extent * self.datamodule.cameras_extent)
            self.births.initialize_incremental_optimizer(
                self.config.gaussian_stage2)
        self._freeze_base()


    @torch.no_grad()
    def _prune_terminal_births(self):
        """Remove weak births without adding unfitted GS on the final step."""
        model = self.births
        cfg = self.config.birth
        low_opacity = model.get_opacity.flatten() < cfg.opacity_threshold
        remove = low_opacity
        if cfg.prune_large:
            big_screen = model.max_radii2D > cfg.new_max_screen_size
            big_world = (model.get_scaling.max(dim=1).values >
                         0.1 * self.datamodule.cameras_extent)
            remove = remove | big_screen | big_world
        removed = int(remove.sum())
        if removed:
            model.prune_points(remove)
        return removed


    @torch.no_grad()
    def post_training_step(self, package):
        step = self.trainer.global_step
        packages = package if isinstance(package, list) else [package]
        if uses_early_replacement(self.config):
            self._early_post_training_step(packages,step)
            return
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
            # All views contribute evidence, but transfer only two final totals.
            frustum_count, observed_count = torch.stack(evidence_counts).sum(0).tolist()
            self.birth_control['motion_frustum_samples'] = (
                self.birth_control.get('motion_frustum_samples', 0) + frustum_count)
            self.birth_control['motion_observed_samples'] = (
                self.birth_control.get('motion_observed_samples', 0) + observed_count)
            edits = self.motion.optimizer_step()
            if getattr(self, 'feature_dc_update', None) is not None:
                self._step_feature_dc(step)
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
                if getattr(self.config.birth, 'appearance_candidates', 0) > 0:
                    self.birth_appearance_gradient = torch.zeros(
                        len(self.gaussians._xyz), 3, device=self.device)
            if step >= start:
                for item in packages:
                    if len(packages) > 1:
                        # Average optimizer gradients; preserve per-view birth
                        # score units and the existing densification threshold.
                        item['viewspace_points'].grad.mul_(len(packages))
                    self.gaussians.add_densification_stats(
                        item['viewspace_points'], item['visibility_filter'])
                    if getattr(self.config.birth, 'appearance_candidates', 0) > 0:
                        probe = item.get('birth_feature_probe')
                        if probe is None or probe.grad is None:
                            raise ValueError('Enabled appearance births require birth_feature_probe gradients')
                        self.birth_appearance_gradient.add_(
                            probe.grad[:, 0, :].detach() * len(packages))
            if step == self.motion_steps:
                rows = torch.arange(len(self.gaussians._xyz), device=self.device)
                if getattr(self, 'opacity_update', None) is not None:
                    rows = self._commit_opacity_and_retire(
                        forced_remove=torch.zeros(len(rows), dtype=torch.bool, device=self.device)) \
                        if uses_residual_birth(self.config) else self._commit_opacity_and_retire()
                self.motion.record_final_diagnostics()
                directory = self.output_dir / 'region_statistics'
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'frame_{self.current_frame:03d}.json').write_text(
                    json.dumps(self.motion.flat_region_diagnostics, indent=2))
                self._begin_births()
        else:
            if uses_dual_predictive(self.config):
                self._post_dual_birth_step(packages,step)
                return
            if len(self.births._xyz):
                for item in packages:
                    visible = item['visibility_filter'][-len(self.births._xyz):]
                    radii = item['radii'][-len(self.births._xyz):]
                    self.births.max_radii2D[visible] = torch.maximum(
                        self.births.max_radii2D[visible], radii[visible])
                    if len(packages) > 1:
                        item['viewspace_points'].grad.mul_(len(packages))
                    self.births.add_densification_stats(
                        item['viewspace_points'], visible, align='right')
                    if uses_residual_birth(self.config):
                        observed = item['motion_contribution_filter'][-len(self.births._xyz):]
                        camera = item['birth_camera_index']
                        self.birth_transaction.observations[observed] |= 1 << camera
                birth_step = step - self.motion_steps
                # Densification/pruning replaces Parameter tensors: apply the
                # fitted gradient exactly once before either happens.
                self.births.optimizer.step()
                self.births.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.births, 'project_parameters'):
                    self.births.project_parameters()
                    if uses_reincarnation(self.config):
                        floor = math.sqrt(3)*.5*self.config.rigid_tree.gs_xyz_step_extent*self.datamodule.cameras_extent
                        constrain_candidates(self.births,self.birth_transaction,floor,
                            rows=self.births.novel_mask.nonzero(as_tuple=True)[0])
                    if uses_quick_source(self.config):
                        floor = math.sqrt(3)*.5*self.config.rigid_tree.gs_xyz_step_extent*self.datamodule.cameras_extent
                        constrain_candidates(self.births,self.birth_transaction,floor,rows=self.births.detail_rows)
                elif uses_residual_birth(self.config):
                    # The inherited motion grid itself leaves up to sqrt(3)/2
                    # of a cell of uncertainty along a rotated surface normal.
                    floor = (math.sqrt(3)*.5*self.config.rigid_tree.gs_xyz_step_extent*
                             self.datamodule.cameras_extent)
                    constrain_candidates(self.births, self.birth_transaction, floor)
                elif self.config.birth.source_position_limit_sigma is not None:
                    constrain_birth_positions(self.births, self.gaussians,
                        self.config.birth.source_position_limit_sigma,
                        self.config.birth.source_position_min_extent * self.datamodule.cameras_extent)
                if uses_residual_birth(self.config):
                    pass  # A fixed probation pool: admission owns both birth and death.
                elif step == self.total_steps:
                    self.birth_control['pruned'] += self._prune_terminal_births()
                elif (birth_step % self.config.birth.densify_interval == 0 and
                        (self.config.birth.densify_until_step is None or
                         birth_step <= self.config.birth.densify_until_step)):
                    result = self.births.adaptive_densify_and_prune(
                        self.config.birth.densify_grad_threshold,
                        self.config.birth.opacity_threshold,
                        self.datamodule.cameras_extent,
                        self.config.birth.new_max_screen_size
                        if self.config.birth.prune_large else None,
                        max_points=self.config.birth.max_new_gaussians)
                    self.birth_control['split_added'] += result['added']
                    self.birth_control['pruned'] += result['pruned']
                if uses_quick_source(self.config):
                    self._quick_optimizer_update_done(step)
                if uses_persistent_birth(self.config):
                    self._persistent_optimizer_update_done(step)
            if (getattr(self, 'opacity_update', None) is not None and
                    self.config.opacity_update.joint_with_birth):
                birth_step = step - self.motion_steps
                birth_lr_factor = self.config.birth.lr_final_ratio ** min(
                    1, max(0, (birth_step - 1) /
                    max(1, self.config.rigid_tree.birth_steps - 1)))
                window = self.config.rigid_tree.motion_fit_window_steps
                self.opacity_update.optimizer_step(self._rate_byte_price(step),
                    proximal=(window is None or birth_step % window == 0 or step == self.total_steps),
                    lr_factor=(self.motion._motion_lr_factor(self.motion_steps) * birth_lr_factor
                        if self.config.opacity_update.decay_with_motion else 1.))
                if self.config.birth.source_handoff:
                    self._pin_birth_handoff_sources()
                # Keep source predictors, export and rendering on the same
                # canonical alpha grid while newborn attributes keep fitting.
                self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
            if getattr(self, 'feature_dc_update', None) is not None and self.config.feature_dc_update.joint_with_birth:
                self._step_feature_dc(step)
                self.gaussians._features_dc.copy_(self.feature_dc_update.canonical_features().to(self.device))
            if step == self.total_steps:
                if uses_residual_birth(self.config):
                    if uses_persistent_birth(self.config):
                        self._finish_persistent_births()
                    elif uses_shadow_source(self.config):
                        self._finish_shadow_births()
                    elif uses_quick_source(self.config):
                        self._finish_quick_births()
                    elif self.birth_terminal:
                        self._finish_population_refit()
                    else:
                        if self._advance_birth_cohort():
                            return
                        if self._accept_residual_births():
                            return
                recorder = getattr(self.trainer, 'record_birth_comparison', None)
                if recorder is not None:
                    recorder()
                self.finalize_frame()

    @torch.no_grad()
    def finalize_frame(self):
        if self.frame_finalized:
            return
        if uses_quick_source(self.config):
            allowed=(200,300) if uses_shadow_source(self.config) else (300,)
            if self.trainer.global_step!=self.total_steps or self.total_steps not in allowed:
                raise ValueError('Quick manifest cannot be finalized outside its motion200 / total300 budget')
        if uses_persistent_birth(self.config) and (
                self.trainer.global_step!=self.total_steps or self.total_steps not in (200,300) or
                self.persistent_birth_optimizer_steps!=self.total_steps-200):
            raise ValueError('Persistent manifest must account for exactly motion200 and at most birth100')
        if uses_early_replacement(self.config) and (
                self.trainer.global_step not in (200,300) or self.early_shared_steps!=self.trainer.global_step):
            raise ValueError('Early replacement manifest requires a truthful single shared-Adam global200/300 ledger')
        if getattr(self.config.birth,'frame_training_plan',False):
            if self.trainer.global_step != 300:
                raise ValueError('Frame plan can only encode after step300 cleanup')
            self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
        if self.opacity_update is not None:
            self.opacity_update.finish()
        if getattr(self, 'feature_dc_update', None) is not None:
            self.feature_dc_update.finish()
        view = self.make_birth_view()
        raw, sources = materialize_quantized_gaussians(
            model_raw(self.gaussians), view)
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
                                    not self.config.rigid_tree.rotation_updates),
            birth_pose_steps=((self.config.birth.new_xyz_step_extent*self.datamodule.cameras_extent,self.config.birth.new_rotation_step)
                if self.config.birth.radiance_mode=='adaptive_residual_birth' else None))
        opacity_info = None
        if self.opacity_update is not None:
            opacity_info = self.opacity_update.export(
                stream_dir / f'opacity_{self.current_frame:03d}.bin', self.current_frame)
            expected_logits = self.opacity_update.canonical_logits().to(raw['opacity'])
            if not torch.equal(raw['opacity'][:len(self.motion.owner)], expected_logits):
                raise ValueError('Materialized old opacity differs from its sparse packet')
        feature_info = None
        if getattr(self, 'feature_dc_update', None) is not None:
            feature_info = self.feature_dc_update.export(
                stream_dir / f'shdc_{self.current_frame:03d}.bin', self.current_frame)
            expected_features = self.feature_dc_update.canonical_features().to(raw['sh_dc'])
            if not torch.equal(raw['sh_dc'][:len(expected_features)], expected_features):
                raise ValueError('Materialized old DC differs from its shared color packet')
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
        self.birth_transaction = None
        self.coarse_birth_snapshot = None
        self.birth_terminal = False
        self.selected_birth_state = None
        self._transaction_views = None
        self.retired_reference = None
        raw_digest=reference_digest(raw)
        manifest = dict(version=1, frame=self.current_frame,
            optimization_steps=self.total_steps,
            birth_cohorts=(self.birth_cohort+1 if uses_residual_birth(self.config) else 1),
            reference=str(self.trainer.decoder_reference_path.relative_to(
                self.output_dir)),
            motion=motion_path.name,
            birth=(Path(birth_info['path']).name if birth_info else None),
            old_count=len(raw['xyz']) - len(sources), birth_count=len(sources),
            total_count=len(raw['xyz']),
            decoded_state_sha256=decoded.digest().hex(),
            raw_state_sha256=raw_digest,
            previous_count=len(prior.xyz), retired_count=retired_count,
            no_old_deletions=retired_count == 0)
        if uses_quick_source(self.config):
            manifest['training_step_accounting'] = dict(motion=200,birth=self.total_steps-200,total=self.total_steps,
                birth_optimizer_updates=self.quick_birth_optimizer_steps,
                initializer_optimizer_updates=0,post_admission_optimizer_updates=0)
        if uses_persistent_birth(self.config):
            manifest['training_step_accounting']=dict(motion=200,birth=self.total_steps-200,total=self.total_steps,
                birth_optimizer_updates=self.persistent_birth_optimizer_steps,
                initializer_optimizer_updates=0,post_admission_optimizer_updates=0)
        if uses_early_replacement(self.config):
            manifest['training_step_accounting']=dict(warm_gate=70,joint=130,final_population=self.total_steps-200,
                global_iterations=self.total_steps,shared_adam_steps=self.early_shared_steps,motion_participation=200,
                birth_participation=self.early_birth_updates,
                joint_overlap=130 if self.birth_control['early_replacement']['source_count'] else 0,extra_SGD_steps=0,
                topology_born_step=70,population_frozen_step=200)
            if self.config.birth.radiance_mode=='adaptive_residual_birth':
                manifest['training_step_accounting'].update(population_frozen_step=70,final_alpha_cleanup_step=300)
        if getattr(self.config.birth,'frame_training_plan',False):
            manifest['training_step_accounting'] = dict(global_iterations=300, shared_adam_steps=self.early_shared_steps,
                shared_gate=100, fixed_gate_residual=50, joint_gate_residual=30, final_motion=20,
                birth_steps=[200,250], cleanup_steps=[200,250,300], old_opacity=200,
                newborn_updates=[100,50], extra_SGD_steps=0)
            manifest['frame_training_plan_version'] = 1
            manifest['birth_cohorts'] = self.birth_control['early_replacement']['actual_birth_cohorts']
            manifest['birth_events'] = [200,250]
        if self.opacity_update is not None:
            manifest.update(opacity_update=Path(opacity_info['path']).name if opacity_info else None,
                            opacity_update_timing='before_births')
        if getattr(self, 'feature_dc_update', None) is not None:
            manifest.update(feature_dc_update=Path(feature_info['path']).name if feature_info else None,
                            feature_dc_update_timing='before_births')
        manifest_path = stream_dir / f'frame_{self.current_frame:03d}.json'
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.frame_report = dict(frame=self.current_frame,
            initial_reference=self.initial_reference,
            region_count=int(self.motion.owner.unique().numel()),
            candidate_region_count=len(self.motion.parents),
            initialization=dict(spatial=True, initial_regions=8,
                gate_start_step=self.config.rigid_tree.spatial_gate.start_step,
                gate_freeze_step=self.config.rigid_tree.spatial_gate.freeze_step,
                optimization_steps=self.total_steps),
            old_count=len(raw['xyz']) - len(sources), birth_count=len(sources),
            previous_count=len(prior.xyz), retired_count=retired_count,
            total_count=len(raw['xyz']), motion=motion_info, birth=birth_info,
            opacity_update=opacity_info, feature_dc_update=feature_info,
            motion_observation_diagnostics=dict(
                kind=('actual_compositing' if self.config.effective_motion_observations else 'frustum'),
                frustum_samples=self.birth_control.get('motion_frustum_samples', 0),
                used_samples=self.birth_control.get('motion_observed_samples', 0),
                old_gs_with_no_motion_observation=int(self.motion.observations.eq(0).sum()),
                scope='surviving GS; current motion optimization views; excludes birth and heldout renders'),
            stage_image_metrics=self.stage_image_metrics,
            encoded_bytes=motion_path.stat().st_size +
                (birth_info['archive_bytes'] if birth_info else 0) +
                (opacity_info['archive_bytes'] if opacity_info else 0) +
                (feature_info['archive_bytes'] if feature_info else 0),
            shared_reference_bytes=self.trainer.decoder_reference_path.stat().st_size,
            materialized_sha256=raw_digest,
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
                      'Previous-row deletions are encoded in motion; optional quantized old-opacity/DC deltas; '
                      'old scale and higher-order SH fixed.')
        if getattr(self.config.birth,'frame_training_plan',False):
            self.frame_report['packet_bytes'] = self.frame_report['encoded_bytes']
            self.frame_report['manifest_bytes'] = manifest_path.stat().st_size
            self.frame_report['encoded_bytes'] += manifest_path.stat().st_size
            self.frame_report['complete_dynamic_update_bytes'] = self.frame_report['encoded_bytes']

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
