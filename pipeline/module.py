"""Spatial gate motion, GS residuals, and quantized newborn Gaussians."""

from dataclasses import asdict, dataclass, field, replace
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
    quaternion_angle_error, quaternion_rotate,
    saved_config_with_default_motion_lr)
from .model.contribution_renderer import ContributionRenderer
from .model.full_attribute_update import (FullUpdateConfig, FullAttributeUpdate,
    FullUpdateSpatialGatedMotion, FIELDS as FULL_ATTRIBUTE_FIELDS)
from .model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from .model.sparse_features import FeatureDCUpdateConfig, QuantizedFeatureDCUpdate
from .model.sparse_logscale import (LogScaleUpdateConfig,
    MotionSupportLogScaleUpdate)
from .model.residual_birth import ResidualBirthConfig, BirthTransaction, RawView
from .early_replacement import EarlyReplacementController
from .frame_plan import (FramePlanController, fitting_old_motion, regional_attribute_repair_active,
                         effective_final_joint_refine_steps)


BIRTH_ALLOCATION = 'clone_small_v1'


def uses_residual_birth(config):
    return getattr(getattr(config, 'birth', None), 'mechanism', 'clone') == 'residual_transaction'


def uses_early_replacement(config):
    return (uses_residual_birth(config) and
            getattr(config.birth, 'radiance_mode', 'free') == 'adaptive_residual_birth')


def birth_allocation(config):
    if getattr(config.birth,'frame_training_plan',False):
        return 'frame_training_plan_v1'
    return 'adaptive_residual_birth_300_v16' if uses_early_replacement(config) else BIRTH_ALLOCATION


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
    keep_incumbent_fixed_after_motion: bool = False
    keep_incumbent_alpha_after_motion: bool = False
    regional_color_birth: bool = False
    regional_alpha_repair: bool = False
    regional_color_depth: int = 5
    regional_color_l1_threshold: float = .03
    regional_color_min_support: float = 32.
    regional_color_min_views: int = 2
    regional_color_reference_layout: Optional[str] = None
    regional_postfit_admission: bool = False
    # Allows population experiments to keep the motion/gate rate objective
    # while fitting incumbent opacity without a byte-price objective.
    population_rate_exempt: bool = False
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
        if self.radiance_mode not in ('free', 'conservative_detail', 'adaptive_residual_birth'):
            raise ValueError('Unknown newborn radiance mode')
        if self.mechanism == 'residual_transaction' and self.radiance_mode != 'adaptive_residual_birth':
            raise ValueError('Retired residual birth training modes are replay-only')
        self.residual.validate()
        for name in ('storage_loss_enabled','survival_aware_storage','population_rate_exempt',
                     'keep_incumbent_fixed_after_motion','keep_incumbent_alpha_after_motion',
                     'regional_color_birth','regional_alpha_repair','regional_postfit_admission',
                     'residual_support_init','residual_colour_init','joint_old_opacity'):
            if not isinstance(getattr(self,name),bool):raise ValueError(f'birth.{name} must be boolean')
        if self.keep_incumbent_fixed_after_motion and not self.frame_training_plan:
            raise ValueError('Fixed incumbent after motion requires the frame training plan')
        if self.keep_incumbent_alpha_after_motion and not self.frame_training_plan:
            raise ValueError('Post-motion old alpha requires the frame training plan')
        if self.regional_alpha_repair and not (self.frame_training_plan and
                self.regional_color_birth and self.keep_incumbent_alpha_after_motion):
            raise ValueError('Regional alpha repair requires the frame plan, regional color birth and post-motion old alpha')
        if self.regional_color_birth:
            if (not self.frame_training_plan or not self.residual_support_init or
                    self.regional_postfit_admission or self.regional_color_depth != 5):
                raise ValueError('Regional color births require the frame plan, support initialization, depth 5 and no postfit admission')
            if (isinstance(self.regional_color_min_views, bool) or
                    not isinstance(self.regional_color_min_views, int) or self.regional_color_min_views < 1):
                raise ValueError('regional_color_min_views must be a positive integer')
            for name in ('regional_color_l1_threshold', 'regional_color_min_support'):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
            if self.regional_color_reference_layout is not None and not isinstance(self.regional_color_reference_layout, str):
                raise ValueError('regional_color_reference_layout must be a path or null')
        if self.regional_postfit_admission and not (
                self.frame_training_plan and self.keep_incumbent_fixed_after_motion):
            raise ValueError('Regional post-fit admission requires the frame plan with a fixed incumbent')
        if self.population_rate_exempt and not self.frame_training_plan:
            raise ValueError('Population rate exemption currently requires the frame training plan')
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
    logscale_update: LogScaleUpdateConfig = field(default_factory=LogScaleUpdateConfig)
    full_update: FullUpdateConfig = field(default_factory=FullUpdateConfig)


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
        cfg.logscale_update.validate()
        cfg.full_update.validate()
        if cfg.full_update.enabled and not cfg.full_update.regional_repair:
            raise ValueError('Periodic full-update training is retired; historical packets remain decodable')
        if cfg.full_update.enabled and (
                not cfg.birth.frame_training_plan or cfg.logscale_update.enabled or
                not cfg.opacity_update.enabled or
                cfg.rigid_tree.spatial_gate.training_schedule != 'joint' or
                cfg.rigid_tree.spatial_gate.hierarchical_private or
                cfg.rigid_tree.spatial_gate.private_full_motion):
            raise ValueError('Full updates require the joint frame plan, unrestricted point residuals, old opacity and no separate logscale update')
        if cfg.full_update.regional_repair:
            birth = cfg.birth
            if (cfg.feature_dc_update.enabled or birth.replace_low_opacity or
                    birth.regional_color_birth or birth.regional_alpha_repair or
                    birth.regional_postfit_admission or not birth.joint_old_opacity or
                    cfg.rigid_tree.spatial_gate.final_joint_refine_steps or
                    cfg.rigid_tree.spatial_gate.max_depth != 5):
                raise ValueError('Regional attribute repair requires fixed population, no shared colour and the D5 frame plan')
            if (isinstance(birth.regional_color_min_views, bool) or
                    not isinstance(birth.regional_color_min_views, int) or birth.regional_color_min_views < 2):
                raise ValueError('Regional attribute repair requires at least two reliable training views')
            for name in ('regional_color_l1_threshold', 'regional_color_min_support'):
                value = getattr(birth, name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
            if birth.regional_color_min_support < 32:
                raise ValueError('Regional attribute repair needs at least 32 effective pixels per view')
        if cfg.logscale_update.enabled and (
                not getattr(cfg.birth, 'frame_training_plan', False) or
                effective_final_joint_refine_steps(self) or
                cfg.logscale_update.selection_steps + cfg.logscale_update.refit_steps !=
                cfg.rigid_tree.birth_steps):
            raise ValueError('Log-scale updates require the fixed frame plan and exactly one '
                             'selection/refit fit over the birth stage')
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
        self.motion = FullUpdateSpatialGatedMotion(cfg.rigid_tree)
        self.full_attribute_update = None
        self.opacity_update = QuantizedOpacityUpdate(cfg.opacity_update) if cfg.opacity_update.enabled else None
        self.feature_dc_update = (QuantizedFeatureDCUpdate(cfg.feature_dc_update, motion=self.motion)
                                  if cfg.feature_dc_update.enabled else None)
        self.logscale_update = (MotionSupportLogScaleUpdate(cfg.logscale_update)
                                if cfg.logscale_update.enabled else None)
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

    @property
    def full_update_active(self):
        cfg = getattr(self.config, 'full_update', None)
        return bool(cfg is not None and cfg.enabled and cfg.regional_repair and self.current_frame > 0)

    def _configure_full_update_motion(self, frame):
        cfg = getattr(self.config, 'full_update', None)
        active = bool(cfg is not None and cfg.enabled and cfg.regional_repair and frame > 0)
        # Copy per-frame policy; the configured flag and checkpoint contract do not change.
        self.motion.config = (replace(self.config.rigid_tree, rotation_updates=True)
                              if active else self.config.rigid_tree)
        self.motion.regional_repair = active
        self.motion.original_rotation_updates = self.config.rigid_tree.rotation_updates

    def full_update_prepare_step(self, step):
        if self.full_update_active and self.full_attribute_update is not None:
            trainable = step > self.motion_steps
            for parameter in self.full_attribute_update.parameters.values():
                parameter.requires_grad_(trainable)
                if not trainable:
                    parameter.grad = None
            args = self.config.gaussian
            rates = dict(logscale=args.scaling_lr, sh_dc=args.feature_lr,
                         sh_rest=args.feature_lr / 20.)
            # These three parameters first become trainable at step 201.
            # Their own 100-step schedule must not inherit a completed motion clock.
            progress = min(max((step - 201) / 99., 0.), 1.)
            factor = self.config.rigid_tree.motion_lr_final_ratio ** progress
            for group in self.motion.optimizer.param_groups:
                name = group.get('name', '')
                if name.startswith('early_full_'):
                    group['lr'] = rates[name[len('early_full_'):]] * factor

    def full_update_after_step(self, step):
        if self.full_update_active and self.full_attribute_update is not None:
            if step <= self.motion_steps:
                return
            window = self.config.rigid_tree.motion_fit_window_steps
            self.full_attribute_update.after_step(self.motion.optimizer, self._rate_byte_price(step),
                proximal=(window is None or step % window == 0 or step == self.total_steps))

    @torch.no_grad()
    def full_update_finish_pose_attributes(self):
        if not self.full_update_active:
            return
        xyz, rotation = self.motion.decoded_survivor_pose()
        self.gaussians.apply_selected_motion(xyz.to(self.device), rotation.to(self.device))

    def full_update_reference(self, *, canonical=False):
        return dict(logscale=self.gaussians._scaling, sh_dc=self.gaussians._features_dc,
                    sh_rest=self.gaussians._features_rest)

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
        if (getattr(self.config.birth, 'frame_training_plan', False) and
                self.frame_prepared and not self.frame_finalized):
            ckpt['frame_plan_diagnostic_reference'] = getattr(
                self, '_plan_diagnostic_reference', None)
        ckpt['rigid_tree_retired_reference'] = self.retired_reference
        opacity_update = getattr(self, 'opacity_update', None)
        ckpt['quantized_opacity_update'] = opacity_update.capture() if opacity_update is not None else None
        ckpt['births'] = (self.births.capture() if self.motion_committed and
                          not self.frame_finalized and self.births.optimizer is not None else None)
        ckpt['birth_source_rows'] = (self.births.source_rows.detach().cpu()
                                     if self.births.source_rows is not None else None)
        feature_update = getattr(self, 'feature_dc_update', None)
        ckpt['quantized_feature_dc_update'] = feature_update.capture() if feature_update is not None else None
        logscale_update = getattr(self, 'logscale_update', None)
        ckpt['quantized_logscale_update'] = (logscale_update.capture(self.motion.optimizer)
            if logscale_update is not None and logscale_update.ready else None)
        full = getattr(self, 'full_attribute_update', None)
        if getattr(self, 'full_update_active', False) and self.frame_prepared and not self.frame_finalized:
            if full is None:
                raise ValueError('Keyframe checkpoint lost full attributes')
            ckpt['full_attribute_update'] = full.capture(self.motion.optimizer)
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
                plan_events=getattr(self,'_plan_events',[]),
                plan_admission_camera_names=getattr(self,'_plan_admission_camera_names',None)),'cpu')
        regional_alpha = getattr(self, '_plan_regional_alpha_state', None)
        if (getattr(self.config.birth, 'regional_alpha_repair', False) and
                not getattr(self, 'full_update_active', False) and self.frame_prepared and
                not self.frame_finalized and regional_alpha is not None):
            ckpt['regional_alpha_repair_state'] = BirthTransaction.tensor_tree(regional_alpha, 'cpu')
        regional_attributes = getattr(self, '_plan_regional_attribute_state', None)
        if (self.full_update_active and self.config.full_update.regional_repair and
                self.frame_prepared and not self.frame_finalized and regional_attributes is not None):
            ckpt['regional_attribute_repair_state'] = BirthTransaction.tensor_tree(regional_attributes, 'cpu')

    def on_load_checkpoint(self, ckpt):
        self._plan_regional_alpha_state = None
        self._plan_regional_attribute_state = None
        self._rate_budget_state=ckpt.get('rate_budget_controller')
        diagnostic_reference = ckpt.get('frame_plan_diagnostic_reference')
        self._plan_diagnostic_reference = (None if diagnostic_reference is None else
            {name: value.detach().cpu().clone() for name, value in diagnostic_reference.items()})
        self.gaussians.restore(ckpt['gaussians'], self.config.gaussian)
        self.renderer.gaussians = self.gaussians
        version = ckpt.get('rigid_tree_version')
        if version is None:
            if self.trainer.config.num_init_steps != 15_000:
                raise ValueError(
                    'Native first-frame training protocol requires exactly 15000 steps')
            if self.current_frame != 0 or ckpt['global_step'] != 15_000:
                raise ValueError(
                    'Only a completed 15000-step native first-frame checkpoint can bootstrap this method')
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
            saved_config['logscale_update'] = {
                **asdict(LogScaleUpdateConfig()), **saved_config.get('logscale_update', {})}
            saved_config['full_update'] = {
                **asdict(FullUpdateConfig()), **saved_config.get('full_update', {})}
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
            Module._configure_full_update_motion(self, ckpt['rigid_tree_motion']['frame_index'])
            self.motion.restore(ckpt['rigid_tree_motion'])
            if self.config.opacity_update.enabled:
                self.opacity_update.restore(ckpt.get('quantized_opacity_update'), self.device)
            if self.config.feature_dc_update.enabled:
                self.feature_dc_update.restore(ckpt.get('quantized_feature_dc_update'), self.device)
            if self.config.logscale_update.enabled:
                self.logscale_update.restore(ckpt.get('quantized_logscale_update'), self.device,
                                             self.motion.optimizer)
            if self.motion_committed:
                for parameter in self.motion.parameters():
                    parameter.requires_grad_(False)
                self.motion.optimizer.zero_grad(set_to_none=True)
            self.full_attribute_update = None
            if getattr(self, 'full_update_active', False) and not self.frame_finalized:
                self.full_attribute_update = FullAttributeUpdate(self.config.full_update)
                self.full_attribute_update.restore(ckpt.get('full_attribute_update'),
                    self.motion.optimizer, self.device, len(self.gaussians._xyz))
        if ckpt['births'] is not None and not uses_early_replacement(self.config):
            self.births = BirthGaussianModel(self.config.sh_degree)
            self.births.restore(ckpt['births'], self.config.gaussian_stage2)
            source = ckpt['birth_source_rows']
            if source is None or len(source) != len(self.births._xyz):
                raise ValueError('Birth checkpoint lost source lineage')
            self.births.source_rows = source.to(self.device)
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
            self._plan_admission_camera_names=state.get('plan_admission_camera_names')
            if (getattr(self.config.birth, 'regional_postfit_admission', False) and
                    self.frame_prepared and not self.frame_finalized and
                    self.early_shared_steps >= 200 and not self._plan_admission_camera_names):
                raise ValueError('Post-fit admission checkpoint lost its held-out training cameras')
            if self.frame_prepared:
                self._early_attach_opacity()
            if (self.full_update_active and self.config.full_update.regional_repair and
                    self.frame_prepared and not self.frame_finalized and self.early_shared_steps >= 200):
                saved = ckpt.get('regional_attribute_repair_state')
                if saved is None:
                    raise ValueError('Regional attribute checkpoint lost its step-200 selection')
                self._plan_regional_attribute_state = BirthTransaction.tensor_tree(saved, self.device)
                self._plan_validate_regional_attribute_state(self._plan_regional_attribute_state)
            if (getattr(self.config.birth, 'regional_alpha_repair', False) and
                    not getattr(self, 'full_update_active', False) and self.frame_prepared and
                    not self.frame_finalized and self.early_shared_steps >= 200):
                saved_alpha = ckpt.get('regional_alpha_repair_state')
                if saved_alpha is None:
                    raise ValueError('Regional alpha checkpoint lost its step-200 state')
                self._plan_regional_alpha_state = BirthTransaction.tensor_tree(saved_alpha, self.device)
                self._plan_validate_regional_alpha_state(self._plan_regional_alpha_state)
            if ckpt['births'] is not None:
                if getattr(self.config.birth,'frame_training_plan',False):
                    from .model.frame_plan_birth import FramePlanBirthModel
                    self.births=FramePlanBirthModel(self.config.sh_degree,self.device).restore(
                        ckpt['births'],self.motion.optimizer,self.config.gaussian_stage2,self.device)
                else:
                    from .model.adaptive_birth_model import AdaptiveBirthModel
                    self.births=AdaptiveBirthModel(self.config.sh_degree)
                    self.births.restore(ckpt['births'],self.motion.optimizer,self.config.gaussian_stage2)
                self.births.source_rows=ckpt['birth_source_rows'].to(self.device)
                if not getattr(self.config.birth,'frame_training_plan',False):
                    self.births._opacity.requires_grad_(self.motion_committed or self.config.birth.radiance_mode=='adaptive_residual_birth')
            if (self.full_update_active and self.config.full_update.regional_repair and
                    self.full_attribute_update is not None and self.early_shared_steps >= self.motion_steps):
                self.full_update_prepare_step(self.early_shared_steps + 1)
                self.motion.prepare_plan_step(self.early_shared_steps + 1)
            if self.motion_committed and fitting_old_motion(self):
                if not (self.full_update_active and self.config.full_update.regional_repair):
                    self.motion.prepare_plan_step(self.motion.last_step)

    def prepare_frame(self):
        if self.frame_prepared:
            return
        if self.current_frame < 1:
            raise ValueError('Rigid motion starts after the shared first frame')
        if uses_early_replacement(self.config) and (
                self.motion_steps!=200 or self.config.rigid_tree.birth_steps!=100 or
                self.config.rigid_tree.spatial_gate.freeze_step!=(180 if getattr(self.config.birth,'frame_training_plan',False) else 70)):
            raise ValueError('Early replacement requires unchanged gate70, motion200 and global300 schedule')
        if getattr(self.config.birth, 'frame_training_plan', False):
            snapshot = getattr(self.trainer, 'snapshot_frame_plan_reference', None)
            if snapshot is not None:
                snapshot()
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
        Module._configure_full_update_motion(self, self.current_frame)
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
        self._plan_admission_camera_names=None
        self._plan_regional_alpha_state=None
        self._plan_regional_attribute_state=None
        self._plan_error_owner=None
        self._plan_phase_started=None;self._plan_timed_phase=None
        self.birth_control = dict(clone_added=0, split_added=0, pruned=0,
                                  source_tracked=True, retirement=retirement,
                                  previous_birth_count=previous_birth_count)
        self.stage_image_metrics = []
        self._freeze_base()
        logscale_update = getattr(self, 'logscale_update', None)
        if logscale_update is not None:
            logscale_update.reset()
        if self.opacity_update is not None:
            self.opacity_update.setup(self.gaussians._opacity)
        if self.feature_dc_update is not None:
            self.feature_dc_update.setup(self.gaussians._features_dc)
        if uses_early_replacement(self.config):
            self._early_attach_opacity()
        self.full_attribute_update = None
        if getattr(self, 'full_update_active', False):
            self.full_attribute_update = FullAttributeUpdate(self.config.full_update)
            args = self.config.gaussian
            self.full_attribute_update.setup(self.full_update_reference(), self.motion.optimizer,
                (args.scaling_lr, args.feature_lr, args.feature_lr / 20.))

    def pre_training_step(self):
        if getattr(self.config.birth,'frame_training_plan',False):
            supported_colour = not self.config.feature_dc_update.enabled
            if (not self.config.rigid_tree.spatial_gate.frame_training_plan or
                    self.config.birth.radiance_mode != 'adaptive_residual_birth' or
                    self.config.rigid_tree.amplitude_refit_steps or
                    self.config.opacity_update.amplitude_refit_steps or
                    not supported_colour or
                    not self.config.birth.rotation_updates or
                    not self.config.birth.joint_old_opacity or self.config.lambda_dssim != .2 or
                    self.config.photometric_loss != 'l1' or self.config.birth.photometric_loss != 'l1'):
                raise ValueError('Frame plan needs its gate schedule, independent residuals, '
                    'no shared colour, new rotation and L1/DSSIM')
        self.prepare_frame()
        if not self.motion_committed and self.opacity_update is not None:
            self.opacity_update.prepare_step(self.trainer.global_step, self.motion_steps)
        if uses_early_replacement(self.config):
            self._early_prepare_rates(self.trainer.global_step)
            return
        if self.motion_committed and len(self.births._xyz):
            birth_step = self.trainer.global_step - (self.birth_cohort_start
                if uses_residual_birth(self.config) else self.motion_steps)
            set_birth_learning_rates(self.births,self.config.gaussian_stage2,birth_step,
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
        implicit_reference = reference is None
        if implicit_reference and getattr(self, 'full_update_active', False) and not self.frame_finalized:
            reference = self._adaptive_reference()
        transaction = getattr(self, 'birth_transaction', None)
        if reference is None:
            reference = (transaction.fixed_reference if transaction is not None and
                         transaction.fixed_reference is not None else
                         ({name:getattr(self.gaussians,attr).detach() for name,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES)}
                          if self.config.birth.radiance_mode=='adaptive_residual_birth' else model_raw(self.gaussians)))
        feature_update = getattr(self, 'feature_dc_update', None)
        if (feature_update is not None and not self.frame_finalized and
                not getattr(self, 'full_update_active', False)):
            reference = dict(reference)
            reference['sh_dc'] = feature_update().detach()
        logscale_update = getattr(self, 'logscale_update', None)
        if (implicit_reference and logscale_update is not None and
                logscale_update.ready and not self.frame_finalized):
            reference = dict(reference)
            reference['logscale'] = logscale_update.canonical_logscale(
                reference['logscale'])
        return QuantizedNewGaussianView(self.births, self.config.birth,
            self.datamodule.cameras_extent, reference,
            source_rows=self.births.source_rows)

    def forward(self, viewpoint, training=False, old_only=False, proposal_only=False,
                predictor_only=False, collect_error_stats=False):
        auxiliary = proposal_only or predictor_only
        if (proposal_only and predictor_only) or (auxiliary and (
                not training or not self.frame_prepared or self.motion_committed)):
            raise ValueError('Auxiliary render is only available during motion training')
        old_model = self.renderer.gaussians
        effective = getattr(self.config, 'effective_motion_observations', False)
        previous_collection = getattr(self.renderer, 'collect_contributions', False)
        previous_error_collection = getattr(self.renderer, 'collect_error_stats', False)
        early_pose=None
        birth_view=None
        if effective:
            self.renderer.collect_contributions = bool(training and not auxiliary and self.frame_prepared and
                (not self.motion_committed or (uses_residual_birth(self.config) and
                  not self.frame_finalized)))
        if collect_error_stats:
            self.renderer.collect_error_stats = True
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
            logscale_update = getattr(self, 'logscale_update', None)
            if (logscale_update is not None and logscale_update.ready and
                    not self.frame_finalized):
                self.gaussians.rigid_logscale = logscale_update.current_logscale(
                    self.gaussians._scaling)
            if (getattr(self, 'feature_dc_update', None) is not None and self.frame_prepared and
                    not self.frame_finalized and
                    (not self.motion_committed or self.config.feature_dc_update.joint_with_birth)):
                self.gaussians.rigid_sh_dc = self.feature_dc_update()
                if auxiliary:
                    self.gaussians.rigid_sh_dc = self.gaussians.rigid_sh_dc.detach()
            full = getattr(self, 'full_attribute_update', None)
            if getattr(self, 'full_update_active', False) and full is not None and not self.frame_finalized:
                dc = getattr(self.gaussians, 'rigid_sh_dc', None)
                if dc is None:
                    dc = self.gaussians._features_dc
                values = full.apply(dict(logscale=self.gaussians._scaling,
                    sh_dc=dc, sh_rest=self.gaussians._features_rest))
                if auxiliary:
                    values = {name: value.detach() for name, value in values.items()}
                self.gaussians.rigid_logscale = values['logscale']
                self.gaussians.rigid_sh_dc = values['sh_dc']
                self.gaussians.rigid_sh_rest = values['sh_rest']
            if (uses_early_replacement(self.config) and getattr(self,'early_initialized',False) and
                    not self.motion_committed and not old_only):
                xyz=self.gaussians.rigid_xyz
                q=self.gaussians.rigid_rotation
                self.births.bind_source_pose(xyz,q,detach=False)
                early_pose=(xyz,q)
                birth_view=self.make_birth_view(reference=self._adaptive_reference())
                self.renderer.gaussians=(MergedGaussianModel([self.gaussians,birth_view])
                    if birth_view is not None else self.gaussians)
            if self.motion_committed and not self.frame_finalized and not old_only:
                birth_view = (self.make_birth_view(reference=self._adaptive_reference())
                              if self.config.birth.joint_old_opacity else self.make_birth_view())
                old_view = self.gaussians
                transaction = getattr(self, 'birth_transaction', None)
                if transaction is not None and transaction.fixed_reference is not None:
                    old_view = RawView(transaction.fixed_reference, self.config.sh_degree, self.gaussians)
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
            self.gaussians.rigid_sh_rest = None
            self.gaussians.rigid_logscale = None
            self.renderer.gaussians = old_model
            if effective:
                self.renderer.collect_contributions = previous_collection
            if collect_error_stats:
                self.renderer.collect_error_stats = previous_error_collection

    def training_step(self, viewpoint, step, regularization_weight=1.):
        collect = step == 200 and regional_attribute_repair_active(self)
        package = self.forward(viewpoint, training=True, collect_error_stats=collect)
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
        if collect:
            stats = package.pop('regional_error_stats', None)
            if stats is None:
                raise ValueError('Step-200 renderer did not return native regional error statistics')
            error_map = (image.detach()-viewpoint.image.detach()).abs().mean(0)
            numerator, denominator = stats.accumulate(error_map)
            count = len(self.gaussians._xyz)
            if (numerator.shape != (count,) or denominator.shape != (count,) or
                    numerator.requires_grad or denominator.requires_grad):
                raise ValueError('Step-200 regional evidence must match the old-GS layout')
            if self._plan_error_owner is None:
                self._plan_error_owner = self.motion.owner.detach().clone()
            elif not torch.equal(self._plan_error_owner, self.motion.owner):
                raise ValueError('Step-200 gate owners changed between training views')
            package['regional_error_evidence'] = dict(camera=Path(str(viewpoint.path)).name,
                numerator=numerator.detach(), denominator=denominator.detach(),
                total_pixel_l1_sum=error_map.sum(dtype=torch.float64).detach(),
                image_pixel_count=error_map.numel())
        loss = image_loss
        # Leave scalar diagnostics on device until all views have finished.
        # Converting each one to float would synchronize CUDA before backward.
        metrics = dict(image_loss=image_loss.detach(), motion_image_factor=factor,
                       psnr=psnr(image.detach(), viewpoint.image).detach(),
                       gs=len(self.gaussians._xyz) +
                       (len(self.births._xyz) if self.motion_committed else 0))
        if uses_early_replacement(self.config) and self.early_initialized:
            metrics['gs']=len(self.gaussians._xyz)+len(self.births.parent_rows)
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
                (not getattr(self.config.birth,'frame_training_plan',False) or step >= 101 or
                 (getattr(self, 'full_update_active', False) and
                  not self.config.full_update.regional_repair)) and
                (not self.motion_committed or
                 (not self.frame_finalized and (self.config.opacity_update.joint_with_birth or
                                               self.config.birth.joint_old_opacity)))):
            terms = self.opacity_update.rate_terms(self._opacity_rate_price(step),
                width=self.config.rigid_tree.sparse_rate_width_bins)
            loss = loss + regularization_weight * terms['loss']
        if (regularization_weight and getattr(self, 'feature_dc_update', None) is not None and
                (not self.motion_committed or
                 (not self.frame_finalized and self.config.feature_dc_update.joint_with_birth))):
            terms = self.feature_dc_update.rate_terms(self._rate_byte_price(step),
                width=self.config.rigid_tree.sparse_rate_width_bins)
            loss = loss + regularization_weight * terms['loss']
        logscale_update = getattr(self, 'logscale_update', None)
        if (regularization_weight and logscale_update is not None and
                logscale_update.ready and not self.frame_finalized):
            terms = logscale_update.rate_terms(self._rate_byte_price(step))
            loss = loss + regularization_weight * terms['loss']
        full = getattr(self, 'full_attribute_update', None)
        if (regularization_weight and getattr(self, 'full_update_active', False) and full is not None and
                full.config.storage_loss_enabled and not self.frame_finalized):
            terms = full.rate_terms(self._rate_byte_price(step))
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
        return loss, metrics, package

    def _rate_byte_price(self, step):
        cfg = self.config.rigid_tree
        ramp = min(max((step - cfg.rate_warmup_steps) / cfg.rate_ramp_steps, 0.), 1.)
        weight=getattr(self.motion,'effective_rate_loss_weight',cfg.rate_loss_weight)
        return weight * ramp / self.motion.rate_normalization_bytes

    def _opacity_rate_price(self, step):
        if getattr(self.config.birth, 'population_rate_exempt', False):
            return 0.
        return self._rate_byte_price(step)

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
        camera = ((step-self.motion_steps-1)*count+index) % len(names)
        name = names[camera]
        return views[name], self.birth_transaction.fit_names.index(name)


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
                if getattr(self, 'opacity_update', None) is not None:
                    self._commit_opacity_and_retire()
                self.motion.record_final_diagnostics()
                directory = self.output_dir / 'region_statistics'
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'frame_{self.current_frame:03d}.json').write_text(
                    json.dumps(self.motion.flat_region_diagnostics, indent=2))
                self._begin_births()
        else:
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
                birth_step = step - self.motion_steps
                # Densification/pruning replaces Parameter tensors: apply the
                # fitted gradient exactly once before either happens.
                self.births.optimizer.step()
                self.births.optimizer.zero_grad(set_to_none=True)
                if hasattr(self.births, 'project_parameters'):
                    self.births.project_parameters()
                elif self.config.birth.source_position_limit_sigma is not None:
                    constrain_birth_positions(self.births, self.gaussians,
                        self.config.birth.source_position_limit_sigma,
                        self.config.birth.source_position_min_extent * self.datamodule.cameras_extent)
                if step == self.total_steps:
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
                recorder = getattr(self.trainer, 'record_birth_comparison', None)
                if recorder is not None:
                    recorder()
                self.finalize_frame()

    @torch.no_grad()
    def finalize_frame(self):
        if self.frame_finalized:
            return
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
        logscale_update = getattr(self, 'logscale_update', None)
        if logscale_update is not None and logscale_update.ready:
            logscale_update.finish()
        old_raw = model_raw(self.gaussians)
        logscale_reference = old_raw['logscale'].detach().clone()
        if logscale_update is not None and logscale_update.ready:
            old_raw['logscale'] = logscale_update.canonical_logscale(
                logscale_reference)
        full = getattr(self, 'full_attribute_update', None)
        full_reference = None
        if getattr(self, 'full_update_active', False) and full is not None:
            full_reference = self.full_update_reference(canonical=True)
            old_raw.update({name: value.to(old_raw[name]) for name, value in
                            full.canonical(full_reference).items()})
        view = self.make_birth_view(reference=old_raw) if full_reference is not None else self.make_birth_view()
        raw, sources = materialize_quantized_gaussians(
            old_raw, view)
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
        logscale_info = None
        if logscale_update is not None and logscale_update.ready:
            decoded_motion = self.motion.codec_previous_state
            if decoded_motion.gs_translation_rows is None:
                raise ValueError('Exported motion lost point-translation support')
            logscale_info = logscale_update.export(
                stream_dir / f'logscale_{self.current_frame:03d}.bin',
                logscale_reference, frame_index=self.current_frame,
                allowed_rows=decoded_motion.gs_translation_rows)
            expected_logscale = logscale_update.canonical_logscale(
                logscale_reference).to(raw['logscale'])
            if not torch.equal(raw['logscale'][:len(expected_logscale)], expected_logscale):
                raise ValueError('Materialized old log-scale differs from its sparse packet')
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
            if full_reference is None and not torch.equal(raw['sh_dc'][:len(expected_features)], expected_features):
                raise ValueError('Materialized old DC differs from its shared color packet')
        full_info = None
        if full_reference is not None:
            full_info = full.export(stream_dir / f'fullattr_{self.current_frame:03d}.bin',
                                    full_reference, self.current_frame)
            expected_full = full.canonical(full_reference)
            for name in FULL_ATTRIBUTE_FIELDS:
                if not torch.equal(raw[name][:len(expected_full[name])].cpu(), expected_full[name]):
                    raise ValueError(f'Materialized old {name} differs from its full attribute packet')
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
                birth_steps=[200,250], cleanup_steps=[200,250,300],
                old_opacity=(100 if self.config.birth.keep_incumbent_fixed_after_motion else 200),
                old_attribute_optimizer_steps=200,
                newborn_updates=[100,50], extra_SGD_steps=0)
            manifest['training_step_accounting']['motion_optimizer_steps'] = (
                200 + effective_final_joint_refine_steps(self))
            if self.config.birth.keep_incumbent_fixed_after_motion:
                manifest['training_step_accounting'].update(incumbent_fixed_after_motion=True,
                    configured_final_joint_refine_steps=(
                        self.config.rigid_tree.spatial_gate.final_joint_refine_steps))
            manifest['frame_training_plan_version'] = 3
            manifest['birth_cohorts'] = self.birth_control['early_replacement']['actual_birth_cohorts']
            manifest['birth_events'] = [200,250]
            if (not getattr(self, 'full_update_active', False) and
                    getattr(self.config.birth, 'keep_incumbent_alpha_after_motion', False)):
                manifest['training_step_accounting']['old_opacity'] = 200
            if self.full_update_active and self.config.full_update.regional_repair:
                factor = self.motion._motion_lr_factor(200)
                rates = dict(
                    xyz=self.config.rigid_tree.gs_translation_lr * factor,
                    rotation=self.config.rigid_tree.gs_rotation_lr * factor,
                    opacity=self.config.opacity_update.learning_rate * (
                        factor if self.config.opacity_update.decay_with_motion else 1.),
                    logscale=self.config.gaussian.scaling_lr,
                    sh_dc=self.config.gaussian.feature_lr,
                    sh_rest=self.config.gaussian.feature_lr / 20.)
                final_rates = {**rates, 'logscale': rates['logscale'] * factor,
                               'sh_dc': rates['sh_dc'] * factor,
                               'sh_rest': rates['sh_rest'] * factor}
                manifest['birth_events'] = []
                manifest['training_step_accounting'] = dict(
                    global_iterations=300, shared_adam_steps=self.early_shared_steps,
                    shared_gate=100, fixed_gate_residual=50, joint_gate_residual=30,
                    final_motion=20, fixed_region_selected_gs_repair=100,
                    old_opacity_all_rows_steps=100, old_opacity_regional_repair_steps=100,
                    old_attribute_regional_repair_steps=100, birth_steps=[], cleanup_steps=[],
                    newborn_updates=[], extra_SGD_steps=0)
                manifest['regional_attribute_repair'] = dict(
                    selection_step=200, selected_count=self.birth_control['regional_attribute_repair']['selected_count'],
                    selected_fraction=self.birth_control['regional_attribute_repair']['selected_fraction'],
                    shared_region_and_gate_frozen_after=200,
                    diagnostic_sidecar=self.birth_control['regional_attribute_repair']['diagnostics'],
                    motion_learning_rate_factor=factor,
                    attribute_learning_rate_schedule='base * final_ratio ** ((step - 201) / 99)',
                    learning_rates_step_201=rates, learning_rates_step_300=final_rates,
                    reference='previous decoded frame', attribute_packet='shortest of FUA1/FUA2/FUA3 or absent for zero')
        if self.opacity_update is not None:
            manifest.update(opacity_update=Path(opacity_info['path']).name if opacity_info else None,
                            opacity_update_timing='before_births')
        if getattr(self, 'feature_dc_update', None) is not None:
            manifest.update(feature_dc_update=Path(feature_info['path']).name if feature_info else None,
                            feature_dc_update_timing='before_births')
        if logscale_update is not None:
            manifest.update(logscale_update=(Path(logscale_info['path']).name
                                             if logscale_info else None),
                            logscale_update_timing='before_births')
        if full_reference is not None:
            manifest.update(full_update=True,
                full_attribute_update=Path(full_info['path']).name if full_info else None,
                full_attribute_update_timing='after_regional_dc_before_births')
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
            logscale_update=logscale_info,
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
                (feature_info['archive_bytes'] if feature_info else 0) +
                (logscale_info['archive_bytes'] if logscale_info else 0) +
                (full_info['archive_bytes'] if full_info else 0),
            shared_reference_bytes=self.trainer.decoder_reference_path.stat().st_size,
            materialized_sha256=raw_digest,
            decoder_reference=manifest['reference'],
            frame_manifest=str(manifest_path.relative_to(self.output_dir)),
            motion_activity=activity,
            motion_rate_diagnostics={name: float(value.detach())
                                     for name, value in rate_terms.items()},
            opacity_rate_diagnostics=({name: float(value.detach()) for name, value in
                self.opacity_update.rate_terms(self._opacity_rate_price(self.motion_steps),
                    width=self.config.rigid_tree.sparse_rate_width_bins).items()}
                if self.opacity_update is not None else None),
            logscale_rate_diagnostics=({name: float(value.detach()) for name, value in
                logscale_update.rate_terms(self._rate_byte_price(self.total_steps)).items()}
                if logscale_update is not None and logscale_update.ready else None),
            flat_region_diagnostics=self.motion.flat_region_diagnostics,
            birth_control=self.birth_control,
            semantics='Spatial gate motion and quantized births; shared first frame counted once. '
                      'Previous-row deletions are encoded in motion; optional quantized old-opacity/DC and '
                      'motion-supported sparse log-scale deltas; higher-order SH fixed.')
        if self.full_update_active and self.config.full_update.regional_repair:
            self.frame_report['semantics'] = ('Spatial gate motion with fixed population and selected old-GS '
                'pose, opacity, logscale, DC and higher-order SH updates; shared first frame counted once.')
        if full_reference is not None:
            self.frame_report['full_attribute_update'] = full_info
            terms = full.rate_terms(self._rate_byte_price(self.total_steps))
            self.frame_report['full_attribute_rate_diagnostics'] = {
                'storage_loss_enabled': full.config.storage_loss_enabled,
                'byte_price': self._rate_byte_price(self.total_steps),
                'raw_bytes_surrogate': float(terms['raw_bytes_surrogate']),
                'l1_loss': float(terms['l1_loss']),
                'loss': float(terms['loss']),
                'actual_packet_bytes': full_info['archive_bytes'] if full_info else 0,
                'fields': {name: {key: (int(value) if key.startswith('active_') else float(value))
                                  for key, value in item.items()}
                           for name, item in terms['fields'].items()}}
            self.frame_report['semantics'] = ('Spatial gate motion and quantized births; '
                'keyframe old log-scale, DC and higher-order SH use an additional coded delta.')
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
