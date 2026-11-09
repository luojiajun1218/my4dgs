"""Spatial gate motion, GS residuals, and quantized newborn Gaussians."""

from dataclasses import asdict, dataclass, field, replace
from collections import OrderedDict
import json
import math
from pathlib import Path
from typing import Optional
import weakref

import torch
from torch.nn.functional import conv2d, interpolate, l1_loss
from torchvision.utils import save_image

from libgs.metric import psnr, ssim
from libgs.metric.lpips import LPIPS
from libgs.metric.ssim import create_window
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
    FullUpdateSpatialGatedMotion, normalize_full_update_config, FIELDS as FULL_ATTRIBUTE_FIELDS)
from .model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from .model.sparse_features import FeatureDCUpdateConfig, QuantizedFeatureDCUpdate, RegionalFeatureDCUpdate
from .model.regional_logscale import RegionalLogScaleConfig, RegionalLogScaleUpdate
from .model.sparse_logscale import (LogScaleUpdateConfig,
    MotionSupportLogScaleUpdate)
from .model.residual_birth import ResidualBirthConfig, BirthTransaction, RawView
from .early_replacement import EarlyReplacementController
from .frame_plan import (FramePlanController, fitting_old_motion, regional_attribute_repair_active,
                         effective_final_joint_refine_steps, regional_point_budget, regional_probe_steps)


BIRTH_ALLOCATION = 'clone_small_v1'


def uses_residual_birth(config):
    return getattr(getattr(config, 'birth', None), 'mechanism', 'clone') == 'residual_transaction'


def uses_early_replacement(config):
    return (uses_residual_birth(config) and
            getattr(config.birth, 'radiance_mode', 'free') == 'adaptive_residual_birth')


def _full_attribute_learning_rates(args):
    return dict(logscale=args.scaling_lr, sh_dc=args.feature_lr,
                sh_rest=args.feature_lr / 20.)


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
    ablation_motion_only: bool = False
    effective_motion_observations: bool = False
    stable_tail_covariance_backward: bool = False
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
    regional_logscale: RegionalLogScaleConfig = field(default_factory=RegionalLogScaleConfig)
    full_update: FullUpdateConfig = field(default_factory=FullUpdateConfig)


MOTION_IMAGE_SCHEDULE = 'bootstrap_aligned_v1'


def validate_early_point_checkpoint(schedule, state, global_step):
    if (schedule != 'standard' and state.get('prepared') and
            not state.get('finalized') and int(global_step) >= 100):
        raise ValueError('Early-point mid-frame checkpoints at step100 or later are not resumable')


def early_schedule_resume_allowed(saved, requested, state, verified_prefix):
    """Permit only standard-to-early scheduling at a verified finalized prefix."""
    if not (verified_prefix is not None and isinstance(saved, dict) and
            isinstance(saved.get('full_update'), dict) and
            isinstance(requested.get('full_update'), dict)):
        return False
    before, after = saved['full_update'], requested['full_update']
    schedule = after.get('point_training_schedule')
    return bool(before.get('point_training_schedule', 'standard') == 'standard' and
        schedule in ('select100', 'joint101') and
        {**saved, 'full_update': {**before, 'point_training_schedule': schedule}} == requested and
        state.get('prepared') is True and state.get('finalized') is True and
        state.get('motion_committed') is True)


def verified_color_resume_allowed(saved, requested, state, global_step, verified_prefix):
    """Existing stop-at-200 color fork, with only explicit requested changes."""
    if not (isinstance(saved, dict) and isinstance(saved.get('full_update'), dict) and
            isinstance(requested.get('full_update'), dict)):
        return False
    previous, target = saved['full_update'], requested['full_update']
    permitted = ('stop_after_motion', 'color_only', 'resume_from_finalized_200',
                 'train_attributes', 'train_point_dc', 'color_point_budget',
                 'sparse_color_stream_v2', 'sparse_color_logscale_v3',
                 'point_training_schedule', 'shared_dc_rate_scale', 'point_dc_rate_scale')
    comparable = {**saved, 'feature_dc_update': requested['feature_dc_update'],
        'regional_logscale': requested['regional_logscale'] if (
            saved['regional_logscale'] == asdict(RegionalLogScaleConfig()) and
            requested['regional_logscale'] == {**asdict(RegionalLogScaleConfig()), 'enabled': True})
            else saved['regional_logscale'],
        'full_update': {**previous, **{key: target[key] for key in permitted}}}
    return bool(verified_prefix is not None and state.get('prepared') is True and
        state.get('finalized') is True and state.get('motion_committed') is True and
        global_step == 200 and previous.get('stop_after_motion') is True and
        previous.get('point_training_schedule', 'standard') == 'standard' and
        target['point_training_schedule'] in ('standard', 'select100', 'joint101') and
        not saved['feature_dc_update']['enabled'] and comparable == requested)


def regional_scale_resume_allowed(saved, requested, state, verified_prefix):
    """Only enable the default regional scale on a verified finalized prefix."""
    defaults = asdict(RegionalLogScaleConfig())
    enabled = {**defaults, 'enabled': True}
    return bool(verified_prefix is not None and isinstance(saved, dict) and
        saved.get('regional_logscale') == defaults and
        requested.get('regional_logscale') == enabled and
        {**saved, 'regional_logscale': enabled} == requested and
        state.get('prepared') is True and state.get('finalized') is True and
        state.get('motion_committed') is True)


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


class _LPIPSEvalCache:
    """Keep the evaluation network outside the trained module and checkpoints."""

    def __init__(self):
        self._model = None
        self._device = None

    def __call__(self, image, target):
        if self._model is None or self._device != image.device:
            # torchvision initializes random weights before loading pretrained
            # ones. Evaluation and cache warm-up must not perturb training RNG,
            # including when a resumed run starts with an empty cache.
            with torch.random.fork_rng(devices=[]):
                model = LPIPS('vgg', '0.1').to(image.device)
            self._model, self._device = model, image.device
        return self._model(image, target)


class _TrainingSSIMCache:
    """Cache fixed target-image SSIM terms for the current frame."""

    def __init__(self, max_bytes=768 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._windows = {}
        self._targets = OrderedDict()
        self._target_bytes = 0
        self._budgets = {}

    def clear(self):
        self._targets.clear()
        self._target_bytes = 0
        self._budgets.clear()

    def _window(self, image):
        key = (image.shape[-3], image.device, image.dtype)
        if key not in self._windows:
            self._windows[key] = create_window(11, image.shape[-3]).to(
                device=image.device, dtype=image.dtype)
        return self._windows[key]

    def prepare(self, image, target, size):
        if (target.requires_grad or image.device != target.device or image.dtype != target.dtype or
                torch.is_autocast_enabled(image.device.type)):
            return None
        if image.device not in self._budgets:
            # Leave room for the renderer and saved backward buffers.
            free = torch.cuda.mem_get_info(image.device)[0] if image.is_cuda else 4 * self.max_bytes
            self._budgets[image.device] = min(self.max_bytes, free // 4)
        budget = self._budgets[image.device]
        pixels = image.numel() // (image.shape[-2] * image.shape[-1]) * size[0] * size[1]
        resized_bytes = pixels * image.element_size()
        needed = resized_bytes * (2 + (tuple(target.shape[-2:]) != tuple(size)))
        if needed > budget:
            return None
        try:
            version = target._version
        except RuntimeError:  # Inference tensors do not expose a version counter.
            return None
        key = (id(target), version, tuple(size))
        entry = self._targets.get(key)
        if entry is not None and entry[0]() is target:
            self._targets.move_to_end(key)
            return (target if entry[1] is None else entry[1], *entry[2:5])
        if entry is not None:
            self._target_bytes -= entry[5]
            del self._targets[key]
        resized = (interpolate(target[None], size=size, mode='area')[0]
                   if tuple(target.shape[-2:]) != tuple(size) else target)
        window = self._window(image)
        with torch.no_grad():
            mean = conv2d(resized, window, padding=5, groups=image.shape[-3])
            variance = conv2d(resized * resized, window, padding=5,
                              groups=image.shape[-3]) - mean.pow(2)
        stored_resized = None if resized is target else resized
        nbytes = sum(t.numel() * t.element_size() for t in
                     (mean, variance) + (() if stored_resized is None else (stored_resized,)))
        if nbytes <= budget:
            while self._targets and self._target_bytes + nbytes > budget:
                _, old = self._targets.popitem(last=False)
                self._target_bytes -= old[5]
            self._targets[key] = (weakref.ref(target), stored_resized, mean, variance, window, nbytes)
            self._target_bytes += nbytes
        return resized, mean, variance, window

    @staticmethod
    def ssim(image, target, mean_target, variance_target, window):
        channel = image.shape[-3]
        mean_image = conv2d(image, window, padding=5, groups=channel)
        mean_image_sq = mean_image.pow(2)
        mean_target_sq = mean_target.pow(2)
        mean_product = mean_image * mean_target
        variance_image = conv2d(image * image, window, padding=5, groups=channel) - mean_image_sq
        covariance = conv2d(image * target, window, padding=5, groups=channel) - mean_product
        return (((2 * mean_product + 0.01**2) * (2 * covariance + 0.03**2)) /
                ((mean_image_sq + mean_target_sq + 0.01**2) *
                 (variance_image + variance_target + 0.03**2))).mean()


def stabilized_rmse(image, target):
    """Pixel-scale RMSE with finite gradient and an exact zero at a match."""
    mse = (image - target).square().mean()
    return mse / ((mse + 1e-8).sqrt() + 1e-4)


@torch.no_grad()
def temporal_repair_weights(target, previous, strength, scale):
    """Soft temporal evidence in a fixed training camera; never a hard mask."""
    if target.shape != previous.shape or target.ndim != 3:
        raise ValueError('Temporal repair images must have matching CHW shapes')
    change = (target.detach() - previous.detach()).abs().mean(dim=0, keepdim=True)
    change = torch.nn.functional.avg_pool2d(change.unsqueeze(0), 5, stride=1, padding=2)[0]
    return 1 + strength * (change / scale).clamp(0, 1)


def weighted_repair_rmse(image, target, weights):
    if image.shape != target.shape or weights.shape != (1, *image.shape[-2:]):
        raise ValueError('Repair weights must match the image')
    mse = ((image - target).square() * weights).sum() / (weights.sum() * image.shape[0])
    return mse / ((mse + 1e-8).sqrt() + 1e-4)


def decoder_exact_rotation(motion, rotation):
    """Use the CPU codec's quaternion values while preserving training gradients."""
    _, decoded = motion.decoded_survivor_pose()
    if decoded.shape != rotation.shape:
        raise ValueError('Decoded motion rotation does not match training pose')
    decoded = decoded.to(device=rotation.device, dtype=rotation.dtype)
    return decoded + (rotation - rotation.detach())


def motion_photometric_loss(image, target, lambda_dssim, factor=1, *, metric='l1', cache=None):
    if metric not in ('l1', 'rmse'):
        raise ValueError('Photometric pixel loss must be l1 or rmse')
    size = (tuple(max(1, value // factor) for value in image.shape[-2:])
            if factor > 1 else tuple(image.shape[-2:]))
    cached = cache.prepare(image, target, size) if cache is not None else None
    if factor > 1:
        image = interpolate(image[None], size=size, mode='area')[0]
        if cached is None:
            target = interpolate(target[None], size=size, mode='area')[0]
    if cached is not None:
        target, mean_target, variance_target, window = cached
    # Global RMS keeps pixel-scale loss units while giving small errors small
    # gradients. The offset makes an exact match have zero finite loss/gradient.
    if metric == 'l1':
        pixel = l1_loss(image, target)
    else:
        pixel = stabilized_rmse(image, target)
    structural = (ssim(image, target) if cached is None else
                  cache.ssim(image, target, mean_target, variance_target, window))
    return (1 - lambda_dssim) * pixel + lambda_dssim * (1 - structural)


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
        if not isinstance(cfg.ablation_motion_only, bool):
            raise ValueError('ablation_motion_only must be boolean')
        if cfg.ablation_motion_only and (
                cfg.full_update.enabled or not cfg.birth.frame_training_plan or
                not uses_early_replacement(cfg) or not cfg.opacity_update.enabled or
                cfg.feature_dc_update.enabled or cfg.logscale_update.enabled or
                cfg.regional_logscale.enabled or cfg.rigid_tree.motion_steps != 200 or
                cfg.birth.replace_low_opacity or cfg.birth.regional_color_birth or
                cfg.birth.regional_alpha_repair or cfg.birth.regional_postfit_admission or
                cfg.rigid_tree.spatial_gate.final_joint_refine_steps):
            raise ValueError('Motion-only ablation requires the fixed-population 200-step frame plan without attribute repair')
        validate_initial_opacity_threshold(cfg.initial_opacity_threshold)
        cfg.birth.validate(cfg.rigid_tree.motion_steps)
        cfg.opacity_update.validate()
        cfg.feature_dc_update.validate()
        cfg.logscale_update.validate()
        regional_scale_config = getattr(cfg, 'regional_logscale', RegionalLogScaleConfig())
        regional_scale_config.validate()
        if regional_scale_config.enabled and not (cfg.full_update.enabled and cfg.full_update.regional_repair and cfg.full_update.color_only and cfg.full_update.sparse_color_logscale_v3 and not cfg.logscale_update.enabled):
            raise ValueError("Regional scale requires fixed-population color-only FUA3")
        cfg.full_update.validate()
        supported_regional_dc = (cfg.feature_dc_update.enabled and
            cfg.feature_dc_update.sharing == 'spatial_region' and cfg.feature_dc_update.post_motion_only)
        if cfg.full_update.enabled and ((cfg.feature_dc_update.enabled and not supported_regional_dc) or
                regional_scale_config.enabled or cfg.logscale_update.enabled):
            raise ValueError('Gradient probing supports only post-motion regional DC as a shared appearance update')
        if cfg.full_update.point_training_schedule != 'standard' and (regional_scale_config.enabled or
                cfg.rigid_tree.shared_discovery_steps > 100 or not cfg.rigid_tree.spatial_gate.frame_training_plan):
            raise ValueError('Early points need the fixed 100-step motion plan and no regional scale')
        if cfg.full_update.color_only:
            if cfg.feature_dc_update.enabled and not cfg.feature_dc_update.post_motion_only:
                raise ValueError('Color-only shared DC must start after the step-200 motion plan')
            if not cfg.feature_dc_update.enabled and not (cfg.full_update.train_attributes and
                    (cfg.full_update.train_point_dc or cfg.full_update.sparse_color_logscale_v3)):
                raise ValueError('Color-only fitting needs shared DC or point DC enabled')
        regional_dc = (cfg.feature_dc_update.enabled and cfg.feature_dc_update.sharing == 'spatial_region')
        if regional_dc and not cfg.full_update.regional_repair:
            raise ValueError('Regional DC requires the fixed-population regional repair pipeline')
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
            if ((cfg.feature_dc_update.enabled and not regional_dc) or birth.replace_low_opacity or
                    birth.regional_color_birth or birth.regional_alpha_repair or
                    birth.regional_postfit_admission or not birth.joint_old_opacity or
                    cfg.rigid_tree.spatial_gate.final_joint_refine_steps):
                raise ValueError('Regional attribute repair requires fixed population and a supported frame plan')
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
        if not isinstance(cfg.stable_tail_covariance_backward, bool):
            raise ValueError('stable_tail_covariance_backward must be boolean')
        if (cfg.stable_tail_covariance_backward and
                not (cfg.effective_motion_observations and cfg.full_update.enabled and
                     cfg.full_update.regional_repair)):
            raise ValueError('Stable tail covariance backward needs observed regional attribute repair')
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
        self.regional_logscale = (RegionalLogScaleUpdate(regional_scale_config, self.motion) if regional_scale_config.enabled else None)
        self.opacity_update = QuantizedOpacityUpdate(cfg.opacity_update) if cfg.opacity_update.enabled else None
        dc_type = RegionalFeatureDCUpdate if regional_dc else QuantizedFeatureDCUpdate
        self.feature_dc_update = (dc_type(cfg.feature_dc_update, motion=self.motion)
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
        if getattr(self.config, 'ablation_motion_only', False):
            return self.motion_steps
        if self.full_update_active and self.config.full_update.stop_after_motion:
            return self.motion_steps
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
        if active and cfg.shared_region_orientation and self.config.rigid_tree.rotation_updates:
            raise ValueError('Shared-only orientation requires per-point motion rotation to be disabled')
        # Copy per-frame policy; the configured flag and checkpoint contract do not change.
        self.motion.config = (replace(self.config.rigid_tree, rotation_updates=True)
                              if active else self.config.rigid_tree)
        self.motion.regional_repair = active
        self.motion.color_only = False
        self.motion.original_rotation_updates = self.config.rigid_tree.rotation_updates
        self.motion.shared_region_orientation = bool(active and cfg.shared_region_orientation)
        self.motion.refine_image_gate_ste = bool(getattr(self, 'gate_refine_image_ste', False))

    def full_update_prepare_step(self, step):
        if self.full_update_active and self.full_attribute_update is not None:
            start = 101 if self.config.full_update.early_joint_repair else self.motion_steps + 1
            trainable = step >= start
            for name, parameter in self.full_attribute_update.parameters.items():
                parameter.requires_grad_(trainable)
                if not trainable:
                    parameter.grad = None
            args = self.config.gaussian
            rates = _full_attribute_learning_rates(args)
            # Use the actual repair start rather than a completed motion clock.
            progress = min(max((step - start) / max(1, 300 - start), 0.), 1.)
            factor = self.config.rigid_tree.motion_lr_final_ratio ** progress
            for group in self.motion.optimizer.param_groups:
                name = group.get('name', '')
                if name.startswith('early_full_'):
                    group['lr'] = rates[name[len('early_full_'):]] * factor
        scale_shared = getattr(self, 'regional_logscale', None)
        if scale_shared is not None and scale_shared.ready:
            scale_shared.prepare_step(step, self.config.rigid_tree.motion_lr_final_ratio)
            if scale_shared.disabled_for_frame:
                scale_shared.delta.requires_grad_(False)
                scale_shared.delta.grad = None
        shared = getattr(self, 'feature_dc_update', None)
        if isinstance(shared, RegionalFeatureDCUpdate):
            shared.prepare_step(step, self.config.rigid_tree.motion_lr_final_ratio)
            branch = getattr(self, 'color_drift_branch', None)
            if shared.ready and step > self.motion_steps and branch in ('none', 'point'):
                shared.delta.requires_grad_(False)
                shared.delta.grad = None

    def full_update_after_step(self, step):
        scale_shared = getattr(self, 'regional_logscale', None)
        if scale_shared is not None and scale_shared.ready and step > self.motion_steps and not scale_shared.disabled_for_frame:
            window = self.config.rigid_tree.motion_fit_window_steps
            scale_shared.after_step(self._rate_byte_price(step) * self.config.full_update.shared_dc_rate_scale,
                proximal=(window is None or step % window == 0 or step == self.total_steps))
        shared = getattr(self, 'feature_dc_update', None)
        if isinstance(shared, RegionalFeatureDCUpdate) and shared.ready and (step > self.motion_steps or not shared.config.post_motion_only):
            shared.sync_owner()
            window = self.config.rigid_tree.motion_fit_window_steps
            shared.after_step(self._rate_byte_price(step) * self.config.full_update.shared_dc_rate_scale,
                proximal=(window is None or step % window == 0 or step == self.total_steps))
        if self.full_update_active and self.full_attribute_update is not None:
            start = 101 if self.config.full_update.early_joint_repair else self.motion_steps + 1
            if step < start:
                return
            window = self.config.rigid_tree.motion_fit_window_steps
            proximal = window is None or step % window == 0 or step == self.total_steps
            self.full_attribute_update.after_step(self.motion.optimizer, self._rate_byte_price(step) * self.config.full_update.point_dc_rate_scale,
                proximal=proximal)

    @torch.no_grad()
    def full_update_finish_pose_attributes(self):
        if not self.full_update_active:
            return
        xyz, rotation = self.motion.decoded_survivor_pose()
        self.gaussians.apply_selected_motion(xyz.to(self.device), rotation.to(self.device))

    def full_update_reference(self, *, canonical=False):
        dc = self.gaussians._features_dc
        shared = getattr(self, 'feature_dc_update', None)
        if isinstance(shared, RegionalFeatureDCUpdate) and shared.ready:
            dc = shared.canonical_features().to(dc) if canonical else shared()
        scale = self.gaussians._scaling
        shared_scale = getattr(self, 'regional_logscale', None)
        if shared_scale is not None and shared_scale.ready:
            scale = shared_scale.canonical_features().to(scale) if canonical else shared_scale()
        return dict(logscale=scale, sh_dc=dc, sh_rest=self.gaussians._features_rest)

    def on_save_checkpoint(self, ckpt):
        ckpt['gate_refine_image_ste'] = bool(getattr(self, 'gate_refine_image_ste', False))
        if getattr(self.config.birth,'frame_training_plan',False) and self.frame_prepared:
            self._plan_record_phase_time()
        ckpt['birth_allocation'] = birth_allocation(self.config)
        ckpt['motion_image_schedule'] = MOTION_IMAGE_SCHEDULE
        ckpt['rigid_tree_version'] = 1
        ckpt['rigid_tree_config'] = asdict(self.config.rigid_tree)
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
        scale_shared = getattr(self, 'regional_logscale', None)
        ckpt['regional_logscale_update'] = scale_shared.capture() if scale_shared is not None else None
        ckpt['regional_logscale_disabled_for_frame'] = bool(scale_shared is not None and scale_shared.disabled_for_frame)
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
        cache = getattr(self, '_training_ssim_cache', None)
        if cache is not None:
            cache.clear()
        self._plan_regional_alpha_state = None
        self._plan_regional_attribute_state = None
        self._plan_early_point_state = None
        if ckpt.get('learnable_regional_repair_gate') is not None:
            raise ValueError('Learnable attribute-repair checkpoints are replay-only')
        if ckpt.get('rate_budget_controller') is not None:
            raise ValueError('Feedback-trained checkpoints are replay-only; fixed-loss training needs a new experiment')
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
        allocation = birth_allocation(self.config)
        if (ckpt.get('current_frame', 0) >= 1 and
                ckpt.get('birth_allocation', 'clone_small_v1') != allocation):
            raise ValueError('Earlier newborn allocation checkpoints are replay-only')
        if self.config.motion_coarse_to_fine and not motion_image_schedule_compatible(
                ckpt.get('motion_image_schedule'), True, self.motion_steps,
                self.config.rigid_tree.spatial_gate.start_step):
            raise ValueError('Different motion image schedules are replay-only')
        self._freeze_base()
        from .sequence_extension import relocate_checkpoint_reference
        ckpt = relocate_checkpoint_reference(
            ckpt, self.config.rigid_tree.spatial_gate.reference_layout)
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
            saved_config.setdefault('ablation_motion_only', False)
            saved_config.setdefault('stable_tail_covariance_backward', False)
            saved_config.setdefault('motion_coarse_to_fine', False)
            saved_config.setdefault('photometric_loss', 'l1')
            saved_config.setdefault('opacity_update', asdict(OpacityUpdateConfig()))
            saved_config['opacity_update'] = {**asdict(OpacityUpdateConfig()), **saved_config['opacity_update']}
            saved_config['feature_dc_update'] = {
                **asdict(FeatureDCUpdateConfig()), **saved_config.get('feature_dc_update', {})}
            saved_config['regional_logscale'] = {**asdict(RegionalLogScaleConfig()), **saved_config.get('regional_logscale', {})}
            saved_config['logscale_update'] = {
                **asdict(LogScaleUpdateConfig()), **saved_config.get('logscale_update', {})}
            saved_config['full_update'] = normalize_full_update_config(saved_config.get('full_update', {}))
            opacity = saved_config.pop('old_opacity', {})
            if opacity.get('enabled', False) or ckpt.get('old_opacity_delta') is not None:
                raise ValueError('Opacity-correction checkpoints are replay-only')
        if isinstance(saved_config, dict) and 'rigid_tree' in saved_config:
            saved_config = {**saved_config, 'rigid_tree':
                saved_config_with_default_motion_lr(saved_config['rigid_tree'])}
        if isinstance(saved_config, dict) and isinstance(saved_config.get('birth'), dict):
            saved_config = {**saved_config, 'birth': {
                **asdict(BirthConfig()), **saved_config['birth']}}
        requested_config = asdict(self.config)
        state = ckpt['rigid_tree_frame']
        validate_early_point_checkpoint(getattr(self.config.full_update, 'point_training_schedule', 'standard'),
                                        state, ckpt.get('global_step', 0))
        color_resume = False
        early_resume = early_schedule_resume_allowed(saved_config, requested_config, state, getattr(self.trainer, '_verified_prefix_plan', None))
        scale_resume = regional_scale_resume_allowed(saved_config, requested_config, state, getattr(self.trainer, '_verified_prefix_plan', None))
        if saved_config != requested_config and self.config.full_update.resume_from_finalized_200:
            color_resume = verified_color_resume_allowed(saved_config, requested_config, state,
                ckpt.get('global_step'), getattr(self.trainer, '_verified_prefix_plan', None))
        if saved_config != requested_config and not color_resume and not scale_resume and not early_resume:
            raise ValueError('Rigid-tree checkpoint needs the same module configuration; color-only changes require a verified finalized 200-step prefix')
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
            if self.config.feature_dc_update.enabled and not (color_resume and ckpt.get('quantized_feature_dc_update') is None):
                if isinstance(self.feature_dc_update, RegionalFeatureDCUpdate):
                    self.feature_dc_update.restore(ckpt.get('quantized_feature_dc_update'), self.device,
                                                   self.motion.optimizer)
                    if self.feature_dc_update.ready and not torch.equal(self.feature_dc_update.owner, self.motion.owner):
                        raise ValueError('Regional DC checkpoint ownership differs from its motion cut')
                else:
                    self.feature_dc_update.restore(ckpt.get('quantized_feature_dc_update'), self.device)
            if getattr(self.config, 'regional_logscale', RegionalLogScaleConfig()).enabled and not ((color_resume or scale_resume) and ckpt.get('regional_logscale_update') is None):
                self.regional_logscale.restore(ckpt.get('regional_logscale_update'), self.device, self.motion.optimizer)
                self.regional_logscale.disabled_for_frame = bool(ckpt.get('regional_logscale_disabled_for_frame', False))
                if self.regional_logscale.ready and not torch.equal(self.regional_logscale.owner, self.motion.owner):
                    raise ValueError('Regional scale checkpoint owner differs from motion cut')
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
                if getattr(self, 'regional_logscale', None) is not None:
                    self.full_attribute_update.regional_scale_symbols = self.regional_logscale.point_symbols
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
            regional_start = 100 if self.config.full_update.early_joint_repair else 200
            if (self.full_update_active and self.config.full_update.regional_repair and
                    self.frame_prepared and not self.frame_finalized and self.early_shared_steps >= regional_start):
                saved = ckpt.get('regional_attribute_repair_state')
                if saved is None:
                    raise ValueError('Regional attribute checkpoint lost its step-200 selection')
                self._plan_regional_attribute_state = BirthTransaction.tensor_tree(saved, self.device)
                self._plan_validate_regional_attribute_state(self._plan_regional_attribute_state,
                    attributes_only=(self.config.full_update.early_joint_repair and self.early_shared_steps < 200))
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
        cache = getattr(self, '_training_ssim_cache', None)
        if cache is not None:
            cache.clear()
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
        self._plan_early_point_state=None
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
        if getattr(self, 'regional_logscale', None) is not None:
            self.regional_logscale.setup(self.gaussians._scaling)
        if self.feature_dc_update is not None:
            self.feature_dc_update.setup(self.gaussians._features_dc)
            if isinstance(self.feature_dc_update, RegionalFeatureDCUpdate) and not self.config.feature_dc_update.post_motion_only:
                self.feature_dc_update.bind_live(self.motion, self.motion.optimizer)
        if uses_early_replacement(self.config):
            self._early_attach_opacity()
        self.full_attribute_update = None
        if getattr(self, 'full_update_active', False):
            self.full_attribute_update = FullAttributeUpdate(self.config.full_update)
            args = self.config.gaussian
            rates = _full_attribute_learning_rates(args)
            self.full_attribute_update.setup(self.full_update_reference(), self.motion.optimizer,
                tuple(rates[name] for name in FULL_ATTRIBUTE_FIELDS))
            if getattr(self, 'regional_logscale', None) is not None:
                self.full_attribute_update.regional_scale_symbols = self.regional_logscale.point_symbols

    def pre_training_step(self):
        if getattr(self.config.birth,'frame_training_plan',False):
            supported_colour = (not self.config.feature_dc_update.enabled or
                self.config.feature_dc_update.sharing == 'spatial_region')
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
        cache = getattr(self, '_training_ssim_cache', None)
        if cache is not None:
            cache.clear()
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

    def _begin_shared_training_views(self, step):
        # This plan samples one gate per step and has no newborns. All views
        # therefore see the same motion graph until the shared Adam update.
        if (not regional_attribute_repair_active(self) or not fitting_old_motion(self) or
                self.motion.proposal_image_weight or self.motion.predictor_image_weight or
                (self.early_initialized and len(self.births._xyz))):
            return False
        if getattr(self, '_shared_training_pose', None) is not None:
            raise ValueError('A shared training pose is already active')
        # These diagnostic steps inspect gradients after the first view.
        mode = 'shared' if step in (201, 250, 300) else 'proxy'
        self._shared_training_pose = dict(step=int(step), pose=self.motion(step, training=True),
                                          attributes=None, mode=mode, proxy_pairs=[])
        return mode

    def _finish_shared_training_views(self):
        state = self._shared_training_pose
        # Each proxy accumulated the four image gradients; traverse the
        # shared gate/attribute graph only once.
        pairs = [(original, proxy.grad) for original, proxy in state['proxy_pairs']
                 if proxy.grad is not None]
        if pairs:
            torch.autograd.backward([item[0] for item in pairs], [item[1] for item in pairs])

    def _end_shared_training_views(self):
        self._shared_training_pose = None

    def _render_opacity_alpha(self):
        # Color-only repair freezes opacity after motion. Render the exact
        # decoder logit -> sigmoid path in that phase: direct alpha-grid
        # rendering can differ by an ULP and cross the rasterizer cutoff.
        color = getattr(getattr(self.config, 'full_update', None), 'color_only', False)
        frozen = (regional_attribute_repair_active(self) and color and
                  self.trainer.global_step > self.motion_steps)
        update = self.opacity_update
        if ((regional_attribute_repair_active(self) and
             self.trainer.global_step > self.motion_steps) or
                (getattr(self.config, 'ablation_motion_only', False) and self.motion_committed)):
            # The codec materializes alpha through CPU logit, followed by the
            # renderer's device sigmoid. Use those exact forward values while
            # keeping the original quantized-alpha straight-through gradient.
            alpha = update.alphas()
            decoded = update.canonical_logits().to(device=alpha.device, dtype=alpha.dtype).sigmoid()
            return decoded + (alpha - alpha.detach())
        if not frozen:
            return update.alphas()
        if update.delta.requires_grad:
            raise ValueError('Decoded opacity cache requires frozen opacity')
        cached = getattr(self, '_frozen_decoded_alpha_cache', None)
        if cached is None or cached[0] != self.current_frame or cached[1] is not update:
            decoded = update.canonical_logits().to(device=update.reference.device).sigmoid()
            cached = (self.current_frame, update, decoded)
            self._frozen_decoded_alpha_cache = cached
        return cached[2]

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
        previous_stable_backward = getattr(self.renderer, 'stable_backward', False)
        stable_tail = getattr(self.config, 'stable_tail_covariance_backward', False)
        if stable_tail:
            self.renderer.stable_backward = bool(
                training and self.frame_prepared and not self.frame_finalized and
                regional_attribute_repair_active(self) and self.trainer.global_step > self.motion_steps)
        early_pose=None
        birth_view=None
        if effective:
            self.renderer.collect_contributions = bool(training and not auxiliary and self.frame_prepared and
                (not self.motion_committed or (uses_residual_birth(self.config) and
                  not self.frame_finalized)))
        if collect_error_stats:
            self.renderer.collect_error_stats = True
        try:
            shared = (getattr(self, '_shared_training_pose', None)
                      if training and not auxiliary and not old_only else None)
            if shared is not None and shared['step'] != self.trainer.global_step:
                raise ValueError('Shared training pose belongs to a different step')
            if shared is not None and shared['attributes'] is not None:
                for name, value in shared['attributes'].items():
                    setattr(self.gaussians, name, value)
            else:
                if fitting_old_motion(self):
                    xyz, rotation = (shared['pose'] if shared is not None else
                                     self.motion.predictor_pose() if predictor_only else
                                     self.motion.proposal_pose() if proposal_only else
                                     self.motion(self.trainer.global_step, training=training))
                    if (self.motion_committed and self.trainer.global_step > self.motion_steps and
                            getattr(self.config.full_update, 'shared_region_orientation', False)):
                        rotation = decoder_exact_rotation(self.motion, rotation)
                    self.gaussians.rigid_xyz = xyz
                    self.gaussians.rigid_rotation = rotation
                    if getattr(self, 'opacity_update', None) is not None:
                        if self.config.opacity_update.mode=='alpha':
                            self.gaussians.rigid_alpha = Module._render_opacity_alpha(self)
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
                    if self.config.opacity_update.mode=='alpha':
                        self.gaussians.rigid_alpha = Module._render_opacity_alpha(self)
                    else:
                        self.gaussians.rigid_opacity = self.opacity_update()
                logscale_update = getattr(self, 'logscale_update', None)
                if (logscale_update is not None and logscale_update.ready and
                        not self.frame_finalized):
                    self.gaussians.rigid_logscale = logscale_update.current_logscale(
                        self.gaussians._scaling)
                if (getattr(self, 'feature_dc_update', None) is not None and self.frame_prepared and
                        not self.frame_finalized and
                        not getattr(self, '_diagnostic_regional_dc_disabled', False) and
                        (not isinstance(self.feature_dc_update, RegionalFeatureDCUpdate) or self.feature_dc_update.ready) and
                        (not self.motion_committed or self.config.feature_dc_update.joint_with_birth)):
                    self.gaussians.rigid_sh_dc = self.feature_dc_update(training=training and not auxiliary)
                    if auxiliary:
                        self.gaussians.rigid_sh_dc = self.gaussians.rigid_sh_dc.detach()
                full = getattr(self, 'full_attribute_update', None)
                full_active = getattr(self, 'full_update_active', False)
                zero_frozen = (full_active and full is not None and
                               self.trainer.global_step <= self.motion_steps and full.initial_zero_frozen())
                if full_active and full is not None and not self.frame_finalized and not zero_frozen:
                    dc = getattr(self.gaussians, 'rigid_sh_dc', None)
                    if dc is None:
                        dc = self.gaussians._features_dc
                    shared_scale = getattr(self, 'regional_logscale', None)
                    scale = shared_scale() if shared_scale is not None and shared_scale.ready else self.gaussians._scaling
                    values = full.apply(dict(logscale=scale,
                        sh_dc=dc, sh_rest=self.gaussians._features_rest))
                    if auxiliary:
                        values = {name: value.detach() for name, value in values.items()}
                    self.gaussians.rigid_logscale = values['logscale']
                    self.gaussians.rigid_sh_dc = values['sh_dc']
                    self.gaussians.rigid_sh_rest = values['sh_rest']
                if shared is not None:
                    attributes = {name: getattr(self.gaussians, name, None) for name in
                        ('rigid_xyz', 'rigid_rotation', 'rigid_alpha', 'rigid_opacity',
                         'rigid_sh_dc', 'rigid_sh_rest', 'rigid_logscale')}
                    if shared['mode'] == 'proxy':
                        for name, value in attributes.items():
                            if value is not None and value.requires_grad:
                                proxy = value.detach().requires_grad_(True)
                                shared['proxy_pairs'].append((value, proxy))
                                attributes[name] = proxy
                                setattr(self.gaussians, name, proxy)
                    shared['attributes'] = attributes
            if (uses_early_replacement(self.config) and getattr(self,'early_initialized',False) and
                    not self.motion_committed and not old_only):
                xyz=self.gaussians.rigid_xyz
                q=self.gaussians.rigid_rotation
                self.births.bind_source_pose(xyz,q,detach=False)
                early_pose=(xyz,q)
                birth_view=(self.make_birth_view(reference=self._adaptive_reference())
                            if len(self.births._xyz) else None)
                self.renderer.gaussians=(MergedGaussianModel([self.gaussians,birth_view])
                    if birth_view is not None else self.gaussians)
            if self.motion_committed and not self.frame_finalized and not old_only:
                if len(self.births._xyz):
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
            if getattr(self, '_capture_regional_full_debug', False):
                g = self.gaussians
                self._regional_full_debug = {name: getattr(g, name).detach().cpu().clone() for name in
                    ('get_xyz', 'get_rotation', 'get_scaling', 'get_opacity', 'get_features')}
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
            if stable_tail:
                self.renderer.stable_backward = previous_stable_backward

    @torch.no_grad()
    def _temporal_repair_weights(self, viewpoint):
        frame = self.current_frame
        if frame < 1:
            raise ValueError('Temporal repair requires a previous frame')
        cache = getattr(self, '_repair_temporal_cache', None)
        if cache is None or cache['frame'] != frame:
            lag = self.config.full_update.repair_temporal_lag
            previous = self.datamodule.scene.train_dataset.get_frame_dataset(max(0, frame - lag))
            cache = self._repair_temporal_cache = dict(frame=frame, dataset=previous, weights={})
        name = Path(viewpoint.path).name
        if name not in cache['weights']:
            previous = cache['dataset']
            indices = [i for i, item in enumerate(previous.items) if Path(item.path).name == name]
            if len(indices) != 1:
                raise ValueError(f'Previous training frame has no unique camera: {name}')
            old_image = previous[indices[0]].image.to(device=viewpoint.image.device,
                                                       dtype=viewpoint.image.dtype)
            cfg = self.config.full_update
            cache['weights'][name] = temporal_repair_weights(viewpoint.image, old_image,
                cfg.repair_temporal_weight, cfg.repair_temporal_scale)
        return cache['weights'][name]

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
        cache = getattr(self, '_training_ssim_cache', None)
        if cache is None:
            cache = self._training_ssim_cache = _TrainingSSIMCache()
        image_loss = motion_photometric_loss(image, viewpoint.image,
                                             self.config.lambda_dssim, factor, metric=metric, cache=cache)
        loss = image_loss
        full_cfg = getattr(self.config, 'full_update', None)
        mix = ((full_cfg.repair_rmse_mix if step > self.motion_steps else full_cfg.motion_rmse_mix)
               if full_cfg is not None else 0.)
        temporal_objective = bool((regional_attribute_repair_active(self) or
                                   getattr(self.config, 'ablation_motion_only', False)) and mix > 0)
        if temporal_objective:
            if full_cfg.repair_temporal_weight:
                weights = self._temporal_repair_weights(viewpoint)
                repair_rmse = weighted_repair_rmse(image, viewpoint.image, weights)
            else:
                repair_rmse = stabilized_rmse(image, viewpoint.image)
            loss = (1 - mix) * image_loss + mix * repair_rmse
        # Leave scalar diagnostics on device until all views have finished.
        # Converting each one to float would synchronize CUDA before backward.
        early = uses_early_replacement(self.config) and self.early_initialized
        if early:
            squared_error = (image.detach() - viewpoint.image.detach()).square()
            channel_mse = squared_error.view(image.shape[0], -1).mean(1, keepdim=True)
            image_psnr = (20 * torch.log10(1. / torch.sqrt(channel_mse))).mean()
        else:
            image_psnr = psnr(image.detach(), viewpoint.image).detach()
        metrics = dict(image_loss=image_loss.detach(), motion_image_factor=factor,
                       psnr=image_psnr,
                       gs=len(self.gaussians._xyz) +
                       (len(self.births._xyz) if self.motion_committed else 0))
        if temporal_objective:
            metrics['repair_rmse'] = repair_rmse.detach()
        if early:
            metrics['gs']=len(self.gaussians._xyz)+len(self.births.parent_rows)
            error=squared_error.mean(0)
            weight=self.early_focus_weights.get(Path(viewpoint.path).name)
            metrics['early_batch_mse']=error.mean().detach()
            if weight is not None:
                metrics['early_focus_mse']=((error*weight).sum()/weight.sum().clamp_min(1e-12)).detach()
        proposal_weight = (getattr(self.motion, 'proposal_image_weight', 0.)
                           if not self.motion_committed else 0.)
        if proposal_weight:
            proposal = self.forward(viewpoint, training=True, proposal_only=True)['render']
            proposal_loss = motion_photometric_loss(proposal, viewpoint.image,
                                                    self.config.lambda_dssim, factor, metric=metric, cache=cache)
            loss = loss + proposal_weight * proposal_loss
            metrics['proposal_image_loss'] = proposal_loss.detach()
        predictor_weight = (getattr(self.motion, 'predictor_image_weight', 0.)
                            if not self.motion_committed else 0.)
        if predictor_weight:
            prediction = self.forward(viewpoint, training=True, predictor_only=True)['render']
            prediction_loss = motion_photometric_loss(prediction, viewpoint.image,
                self.config.lambda_dssim, factor, metric=metric, cache=cache)
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
                (not isinstance(self.feature_dc_update, RegionalFeatureDCUpdate) or self.feature_dc_update.ready) and
                (not self.motion_committed or
                 (not self.frame_finalized and self.config.feature_dc_update.joint_with_birth))):
            terms = self.feature_dc_update.rate_terms(self._rate_byte_price(step) * self.config.full_update.shared_dc_rate_scale,
                width=self.config.rigid_tree.sparse_rate_width_bins)
            loss = loss + regularization_weight * terms['loss']
        logscale_update = getattr(self, 'logscale_update', None)
        if (regularization_weight and logscale_update is not None and
                logscale_update.ready and not self.frame_finalized):
            terms = logscale_update.rate_terms(self._rate_byte_price(step))
            loss = loss + regularization_weight * terms['loss']
        shared_scale = getattr(self, 'regional_logscale', None)
        if regularization_weight and shared_scale is not None and shared_scale.ready and not self.frame_finalized:
            loss = loss + regularization_weight * shared_scale.rate_terms(self._rate_byte_price(step) * self.config.full_update.shared_dc_rate_scale)['loss']
        full = getattr(self, 'full_attribute_update', None)
        if (regularization_weight and getattr(self, 'full_update_active', False) and full is not None and
                full.config.storage_loss_enabled and not self.frame_finalized and
                not (step <= self.motion_steps and full.initial_zero_frozen())):
            terms = full.rate_terms(self._rate_byte_price(step) * self.config.full_update.point_dc_rate_scale)
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
        if count != 1 or index != 0 or not names:
            raise ValueError('Birth fitting requires one random training view per step')
        start = 71 if uses_early_replacement(self.config) else self.motion_steps + 1
        if step < start:
            raise ValueError('Birth fitting view requested before its training phase')
        epoch, position = divmod(step - start, len(names))
        data_config = getattr(self.datamodule, 'config', None)
        seed = int(getattr(data_config, 'training_view_seed', None) or 0)
        generator = torch.Generator().manual_seed(seed + 1000003 * self.current_frame + epoch)
        camera = int(torch.randperm(len(names), generator=generator)[position])
        name = names[camera]
        return views[name], camera


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
        forward_check = None
        cache = getattr(self, '_frozen_decoded_alpha_cache', None)
        if cache is not None and cache[0] == self.current_frame and cache[1] is self.opacity_update:
            actual = self.opacity_update.canonical_logits().to(device=self.opacity_update.reference.device).sigmoid()
            if not torch.equal(cache[2], actual):
                raise ValueError('Frozen decoded opacity changed after it was cached')
        if self.full_update_active or getattr(self.config, 'ablation_motion_only', False) or getattr(self, 'color_verify_forward', False):
            if not self.full_update_active and not getattr(self.config, 'ablation_motion_only', False):
                raise ValueError('Forward/decode check requires full attribute updates')
            views = self.transaction_camera_views()
            camera = getattr(self.trainer, 'diagnostic_train_camera', None) or sorted(views)[0]
            if camera not in views or camera == 'cam00.png':
                raise ValueError('Forward/decode check requires a training camera')
            full_debug = bool(self.config.full_update.regional_full_point_budget)
            self._capture_regional_full_debug = full_debug
            before = self.forward(views[camera], training=True)['render'].detach().clamp(0, 1).clone()
            debug_before = getattr(self, '_regional_full_debug', None) if full_debug else None
            self._capture_regional_full_debug = False
            forward_check = dict(camera=camera, before=before)
        if uses_early_replacement(self.config) and (
                self.trainer.global_step not in (200,300) or self.early_shared_steps!=self.trainer.global_step):
            raise ValueError('Early replacement manifest requires a truthful single shared-Adam global200/300 ledger')
        if getattr(self.config.birth,'frame_training_plan',False):
            stop_at_motion = (getattr(self.config, 'ablation_motion_only', False) or
                              (self.full_update_active and self.config.full_update.stop_after_motion))
            expected_step = 200 if stop_at_motion else 300
            if self.trainer.global_step != expected_step:
                raise ValueError(f'Frame plan can only encode after step{expected_step}')
            self.gaussians._opacity.copy_(self.opacity_update.canonical_logits().to(self.device))
        if self.opacity_update is not None:
            self.opacity_update.finish()
        if getattr(self, 'feature_dc_update', None) is not None:
            self.feature_dc_update.finish()
        logscale_update = getattr(self, 'logscale_update', None)
        if logscale_update is not None and logscale_update.ready:
            logscale_update.finish()
        old_raw = model_raw(self.gaussians)
        shared_scale = getattr(self, 'regional_logscale', None)
        if shared_scale is not None and shared_scale.ready:
            shared_scale.finish()
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
        shared_scale_info = None
        if shared_scale is not None and shared_scale.ready:
            shared_scale_info = shared_scale.export(stream_dir / f'regscale_{self.current_frame:03d}.bin', self.current_frame)
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
            selected = self._plan_regional_attribute_state['eligible'].detach().cpu()
            if selected.dtype != torch.bool or selected.shape != (len(full_reference['sh_dc']),):
                raise ValueError('Invalid gradient-probed full-attribute selection')
            if int(selected.sum()) > regional_point_budget(self, len(selected)):
                raise ValueError('Gradient-probed full-attribute selection exceeds its point budget')
            for name, symbols in full.symbols().items():
                if bool(symbols[~selected].ne(0).any()):
                    raise ValueError(f'Full attribute {name} changed outside the selection')
            sparse_color = (self.config.full_update.sparse_color_stream_v2 or
                            self.config.full_update.sparse_color_logscale_v3)
            if sparse_color:
                from .model.full_attribute_update import validate_sparse_color_selection
                selected = self._plan_regional_attribute_state['eligible'].detach().cpu()
                if selected.dtype != torch.bool or selected.shape != (len(full_reference['sh_dc']),):
                    raise ValueError('Invalid sparse color training selection')
                allowed = ('sh_dc', 'logscale') if self.config.full_update.sparse_color_logscale_v3 else ('sh_dc',)
                validate_sparse_color_selection(full.symbols(), selected, self.config.full_update.color_point_budget, allowed)
            if self.config.full_update.regional_full_point_budget:
                selected = self._plan_regional_attribute_state['eligible'].detach().cpu()
                selected_count = int(selected.sum())
                if (selected_count > self.config.full_update.regional_full_point_budget or
                        10 * selected_count >= len(selected)):
                    raise ValueError('Full-attribute selection must be below 10 percent and within point budget')
                for name, symbols in full.symbols().items():
                    if bool(symbols[~selected].ne(0).any()):
                        raise ValueError(f'Full attribute {name} changed outside the selection')
            full_info = full.export(stream_dir / f'fullattr_{self.current_frame:03d}.bin',
                                    full_reference, self.current_frame, sparse_only=sparse_color)
            expected_full = full.canonical(full_reference)
            for name in FULL_ATTRIBUTE_FIELDS:
                if not torch.equal(raw[name][:len(expected_full[name])].cpu(), expected_full[name]):
                    raise ValueError(f'Materialized old {name} differs from its full attribute packet')
        point_mask_info = None
        if (self.full_update_active and self.config.full_update.color_only and
                self.config.full_update.color_point_budget and
                not (self.config.full_update.sparse_color_stream_v2 or self.config.full_update.sparse_color_logscale_v3) and
                getattr(self, 'color_drift_branch', None) not in ('none', 'shared')):
            from .model.color_point_mask import encode_color_point_mask, decode_color_point_mask
            mask = self._plan_regional_attribute_state['eligible'].detach().cpu().numpy()
            if int(mask.sum()) > self.config.full_update.color_point_budget:
                raise ValueError('Color point mask exceeded its budget')
            packet = encode_color_point_mask(mask, int(self.current_frame))
            if packet is not None:
                path = stream_dir / f'pointmask_{self.current_frame:03d}.bin'
                path.write_bytes(packet)
                if (decode_color_point_mask(packet, int(self.current_frame), len(mask)) != mask).any():
                    raise ValueError('Color point mask packet did not round-trip')
                point_mask_info = dict(path=str(path), archive_bytes=len(packet), selected_points=int(mask.sum()))
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
            if self.full_update_active:
                factor = self.motion._motion_lr_factor(200)
                rates = dict(
                    xyz=self.config.rigid_tree.gs_translation_lr * factor * self.config.full_update.gradient_position_lr_scale,
                    rotation=self.config.rigid_tree.gs_rotation_lr * factor,
                    opacity=self.config.opacity_update.learning_rate * (
                        factor if self.config.opacity_update.decay_with_motion else 1.),
                    logscale=self.config.gaussian.scaling_lr,
                    sh_dc=self.config.gaussian.feature_lr,
                    sh_rest=self.config.gaussian.feature_lr / 20.)
                final_rates = {**rates, 'logscale': rates['logscale'] * factor,
                               'sh_dc': rates['sh_dc'] * factor,
                               'sh_rest': rates['sh_rest'] * factor}
                info = self.birth_control['regional_attribute_repair']
                manifest['birth_events'] = []
                probe_count = self.config.full_update.probe_views_per_round
                probe_steps = list(regional_probe_steps(self))
                early = self.config.full_update.early_joint_repair
                if early:
                    at_201 = self.config.rigid_tree.motion_lr_final_ratio ** (100 / 199)
                    rates = {**rates, **{key: rates[key] * at_201 for key in ('logscale', 'sh_dc', 'sh_rest')}}
                repair_steps = 100 - sum(step > 200 for step in probe_steps)
                early_attribute_steps = 100 - sum(step <= 200 for step in probe_steps) if early else 0
                manifest['training_step_accounting'] = dict(
                    global_iterations=300, shared_adam_steps=self.early_shared_steps - len(probe_steps),
                    shared_gate=self.config.rigid_tree.shared_discovery_steps,
                    joint_gate_residual_early=100-self.config.rigid_tree.shared_discovery_steps,
                    fixed_gate_residual=50 - sum(101 <= step <= 150 for step in probe_steps),
                    joint_gate_residual=30, final_motion=20,
                    fixed_region_selected_gs_repair=repair_steps,
                    old_opacity_all_rows_steps=100 - sum(step <= 200 for step in probe_steps),
                    old_opacity_regional_repair_steps=repair_steps,
                    old_attribute_regional_repair_steps=repair_steps + early_attribute_steps,
                    gradient_probe_steps=probe_steps,
                    birth_steps=[], cleanup_steps=[], newborn_updates=[], extra_SGD_steps=0)
                manifest['regional_attribute_repair'] = dict(
                    selection_step=([100 + probe_count] if early else [200 + probe_count, 250 + probe_count]), train_attributes=True,
                    selected_count=info['selected_count'], selected_fraction=info['selected_fraction'],
                    selection_rounds=info['selection_rounds'], shared_region_and_gate_frozen_after=200,
                    diagnostic_sidecar=info['diagnostics'],
                    selection=f'{1 if early else 2} rounds of {probe_count} single-view image-gradient probes',
                    motion_learning_rate_factor=factor,
                    position_learning_rate_scale=self.config.full_update.gradient_position_lr_scale,
                    attribute_learning_rate_schedule=('base * final_ratio ** ((step - 101) / 199)' if early else
                        'base * final_ratio ** ((step - 201) / 99)'),
                    learning_rates_step_201=rates, learning_rates_step_300=final_rates,
                    reference='previous decoded frame',
                    attribute_packet='shortest of FUA1/FUA2/FUA3 or absent for zero')
        if getattr(self.config, 'ablation_motion_only', False):
            manifest['birth_events'] = []
            manifest['training_step_accounting'] = dict(
                global_iterations=200, shared_adam_steps=self.early_shared_steps,
                shared_gate=self.config.rigid_tree.shared_discovery_steps,
                joint_gate_residual_early=100-self.config.rigid_tree.shared_discovery_steps,
                fixed_gate_residual=50, joint_gate_residual=30, final_motion=20,
                motion_optimizer_steps=200, old_opacity_all_rows_steps=100,
                sparse_attribute_update_steps=0, gradient_probe_steps=[],
                birth_steps=[], cleanup_steps=[], newborn_updates=[], extra_SGD_steps=0)
            manifest['motion_only_ablation'] = dict(
                sparse_attribute_repair=False, per_point_motion_residual=True,
                quantized_opacity_update=True, train_views_per_step=1)
        if getattr(self, 'gate_refine_image_ste', False):
            manifest['gate_refine_image_ste'] = 'hard cut forward; candidate-total path-probability gate derivative at steps 151-180'
        if self.opacity_update is not None:
            manifest.update(opacity_update=Path(opacity_info['path']).name if opacity_info else None,
                            opacity_update_timing='before_births')
        if getattr(self, 'feature_dc_update', None) is not None:
            manifest.update(feature_dc_update=Path(feature_info['path']).name if feature_info else None,
                            feature_dc_update_timing='before_births')
        if shared_scale is not None:
            manifest.update(regional_logscale_update=(Path(shared_scale_info['path']).name if shared_scale_info else None), regional_logscale_update_timing='before_full_attribute_update')
        if logscale_update is not None:
            manifest.update(logscale_update=(Path(logscale_info['path']).name
                                             if logscale_info else None),
                            logscale_update_timing='before_births')
        if self.full_update_active and self.config.full_update.color_only and self.config.full_update.color_point_budget:
            selected=self._plan_regional_attribute_state['eligible']
            if self.config.full_update.sparse_color_logscale_v3:
                manifest['color_stream_version'] = 3
                manifest['color_attribute_fields'] = ['sh_dc', 'logscale']
            elif self.config.full_update.sparse_color_stream_v2:
                manifest['color_stream_version'] = 2
            else:
                manifest['color_point_mask'] = Path(point_mask_info['path']).name if point_mask_info else None
                manifest['color_point_mask_count'] = int(selected.sum()) if getattr(self, 'color_drift_branch', None) not in ('none', 'shared') else 0
            manifest['color_point_budget'] = self.config.full_update.color_point_budget
            manifest['color_drift_branch'] = getattr(self, 'color_drift_branch', None) or 'shared_point'
        if full_reference is not None:
            manifest.update(full_update=True,
                full_attribute_update=Path(full_info['path']).name if full_info else None,
                full_attribute_update_timing='after_regional_dc_before_births')
        if isinstance(getattr(self, 'feature_dc_update', None), RegionalFeatureDCUpdate):
            dc_start = 201 if self.config.feature_dc_update.post_motion_only else 1
            manifest['regional_dc_update'] = dict(start_step=dc_start, end_step=300,
                reference='previous decoded GS DC', ownership='live motion cut, frozen by step200' if dc_start == 1 else 'frozen step200 motion owners',
                tail_learning_rate_scale=self.config.feature_dc_update.tail_learning_rate_scale,
                eligible_regions=int(self.feature_dc_update.allowed.sum()),
                active_regions=int(self.feature_dc_update.symbols().ne(0).any(-1).sum()),
                zero_region_update_is_implicit=True, point_dc_reference='after regional DC')
            shared_on=getattr(self, 'color_drift_branch', None) not in ('none', 'point')
            manifest['regional_dc_update']['trainable'] = shared_on
            manifest['training_step_accounting']['regional_dc_steps'] = (301 - dc_start) if shared_on else 0
        manifest_path = stream_dir / f'frame_{self.current_frame:03d}.json'
        # The manifest is transmitted per frame; whitespace is not needed
        # for parsing and every field remains available to the decoder.
        manifest_path.write_text(json.dumps(manifest, separators=(',', ':')))
        if self.full_update_active:
            # Commit precisely what an independent receiver reconstructed, not
            # merely an algebraically equivalent training-side quantization.
            prior_raw = getattr(self, '_plan_diagnostic_reference', None)
            if prior_raw is None:
                raise ValueError('Bounded attribute updates require the previous decoded raw-state snapshot')
            from scripts.replay_rigid_tree import decode_frame
            replayed_state, replayed_raw = decode_frame(stream_dir, self.current_frame, prior, prior_raw)
            if replayed_state.digest().hex() != manifest['decoded_state_sha256'] or reference_digest(replayed_raw) != raw_digest:
                raise ValueError('Bounded attribute packet replay differs from the materialized training state')
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                target = getattr(self.gaussians, attr)
                target.copy_(replayed_raw[name].to(device=target.device, dtype=target.dtype))
            self.motion.codec_previous_state = replayed_state
        if forward_check is not None:
            self._capture_regional_full_debug = bool(self.config.full_update.regional_full_point_budget)
            after=self.forward(views[forward_check['camera']],training=False)['render'].detach().clamp(0,1)
            debug_after = getattr(self, '_regional_full_debug', None) if self._capture_regional_full_debug else None
            self._capture_regional_full_debug = False
            difference=(forward_check.pop('before')-after).abs()
            forward_check['max_abs'] = float(difference.max())
            forward_check['mean_abs'] = float(difference.mean())
            forward_check['tolerance'] = 1e-5
            if not bool(torch.isfinite(difference).all()) or forward_check['max_abs'] > forward_check['tolerance']:
                if self.config.full_update.regional_full_point_budget:
                    path = Path(self.output_dir) / f'full_forward_debug_{self.current_frame:03d}.pth'
                    torch.save(dict(before=debug_before, after=debug_after, camera=forward_check['camera'],
                                    max_abs=forward_check['max_abs'], mean_abs=forward_check['mean_abs']), path)
                raise ValueError(f"Training forward image differs from decoded-state image: max={forward_check['max_abs']}, mean={forward_check['mean_abs']}")
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
            logscale_update=logscale_info, regional_logscale_update=shared_scale_info, color_point_mask=point_mask_info, forward_decode_check=forward_check,
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
                (shared_scale_info['archive_bytes'] if shared_scale_info else 0) +
                (full_info['archive_bytes'] if full_info else 0) +
                (point_mask_info['archive_bytes'] if point_mask_info else 0),
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
            terms = full.rate_terms(self._rate_byte_price(self.total_steps) * self.config.full_update.point_dc_rate_scale)
            self.frame_report['full_attribute_rate_diagnostics'] = {
                'storage_loss_enabled': full.config.storage_loss_enabled,
                'byte_price': self._rate_byte_price(self.total_steps) * self.config.full_update.point_dc_rate_scale,
                'raw_bytes_surrogate': float(terms['raw_bytes_surrogate']),
                'l1_loss': float(terms['l1_loss']),
                'loss': float(terms['loss']),
                'actual_packet_bytes': full_info['archive_bytes'] if full_info else 0,
                'fields': {name: {key: (int(value) if key.startswith('active_') else float(value))
                                  for key, value in item.items()}
                           for name, item in terms['fields'].items()}}
            self.frame_report['semantics'] = ('Spatial gate motion and quantized births; '
                'keyframe old log-scale, DC and higher-order SH use an additional coded delta.')
        if self.full_update_active and self.config.full_update.color_only and not self.config.full_update.stop_after_motion:
            self.frame_report['semantics'] = ('Motion and opacity frozen after step200; quantized regional DC and selected point attributes fitted through step300; previous decoded frame used as reference.')
        if self.full_update_active and self.config.full_update.stop_after_motion:
            self.frame_report['semantics'] = ('Spatial gate motion and opacity through step200; '
                'fixed population; no post-motion updates; shared first frame counted once.')
        if getattr(self.config, 'ablation_motion_only', False):
            self.frame_report['semantics'] = ('200 steps of shared gate motion, per-point motion residuals '
                'and quantized opacity updates; no sparse attribute repair or births.')
        if getattr(self.config.birth,'frame_training_plan',False):
            self.frame_report['packet_bytes'] = self.frame_report['encoded_bytes']
            self.frame_report['manifest_bytes'] = manifest_path.stat().st_size
            self.frame_report['encoded_bytes'] += manifest_path.stat().st_size
            self.frame_report['complete_dynamic_update_bytes'] = self.frame_report['encoded_bytes']
        if isinstance(getattr(self, 'feature_dc_update', None), RegionalFeatureDCUpdate):
            terms = self.feature_dc_update.rate_terms(self._rate_byte_price(self.total_steps) * self.config.full_update.shared_dc_rate_scale)
            self.frame_report['regional_dc_rate_diagnostics'] = {k:float(v) for k,v in terms.items()}

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
            cache = getattr(self, '_lpips_eval_cache', None)
            if cache is None:
                cache = self._lpips_eval_cache = _LPIPSEvalCache()
            metrics['lpips'] = float(cache(image, viewpoint.image).mean())
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
