"""训练模型的组装与逐帧执行。

负责组织 GS、区域运动和属性更新模块，完成渲染、帧状态管理、
检查点及帧结果导出；训练损失由 loss.py 组合，逐帧训练方案由 per_frame_training.py 控制，
底层运动、属性和码流实现放在 model/ 对应文件中。
"""

from dataclasses import asdict, dataclass, field, replace
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.metric import psnr, ssim
from libgs.metric.lpips import LPIPS
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import Renderer, RendererConfig

from .model.gaussian import GaussianModel, clone_tensor_tree, model_raw
from .model.frame_stream_format import RAW_ATTRIBUTES, RAW_FIELDS, reference_digest
from .model.rigid_residual_tree import (RigidTreeConfig,
    quaternion_angle_error, quaternion_rotate,
    saved_config_with_default_motion_lr)
from .model.contribution_renderer import ContributionRenderer
from .model.residual_gs_attributes_update import (FullUpdateConfig, ResidualAppearanceAttributes,
    normalize_full_update_config, FIELDS as FULL_ATTRIBUTE_FIELDS,
    OpacityUpdateConfig, QuantizedOpacityUpdate, FeatureDCUpdateConfig,
    QuantizedFeatureDCUpdate, LogScaleUpdateConfig, MotionSupportLogScaleUpdate)
from .model.spatial_motion_training import FullUpdateSpatialMotionTraining
from .model.regional_attributes_update import RegionalAttributesUpdate
from .data import _collate_single_view
from .loss import compute_training_loss, temporal_repair_weights
from .per_frame_training import (PerFrameTraining, fitting_old_motion,
    regional_attribute_repair_active, effective_final_joint_refine_steps)


BIRTH_ALLOCATION = 'clone_small_v1'


class _FrameCameraViews(Mapping):
    """Keep a frame's decoded views on CPU unless device caching is requested."""
    def __init__(self, views, device, cache_on_device):
        self.views = views
        self.device = device
        self.cache_on_device = cache_on_device
        if cache_on_device:
            self.views = {name: view.to(device) for name, view in views.items()}

    def __iter__(self):
        return iter(self.views)

    def __len__(self):
        return len(self.views)

    def __getitem__(self, name):
        view = self.views[name]
        return view if self.cache_on_device else view.to(self.device)

    def metadata_items(self):
        return self.views.items()


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
    """逐帧方案仍使用的布局、透明度和损失设置。"""

    mechanism: str = 'clone'
    radiance_mode: str = 'conservative_detail'
    replace_low_opacity: bool = False
    source_handoff: bool = False
    rotation_updates: bool = True
    opacity_threshold: float = 0.01
    new_xyz_step_extent: float = 1e-5
    new_rotation_step: float = 0.001
    photometric_loss: Optional[str] = None
    initial_alpha: Optional[float] = None
    storage_loss_enabled: bool = False
    frame_training_plan: bool = False
    regional_color_birth: bool = False
    regional_alpha_repair: bool = False
    regional_postfit_admission: bool = False
    population_rate_exempt: bool = False
    joint_old_opacity: bool = False
    population_loss_weight: float = 0.0

    def validate(self, motion_steps):
        if self.mechanism not in ('clone', 'residual_transaction'):
            raise ValueError('Unknown birth mechanism')
        if self.radiance_mode not in ('free', 'conservative_detail', 'adaptive_residual_birth'):
            raise ValueError('Unknown radiance mode')
        if self.mechanism == 'residual_transaction' and self.radiance_mode != 'adaptive_residual_birth':
            raise ValueError('Retired residual birth training modes are replay-only')
        for name in ('replace_low_opacity', 'source_handoff', 'rotation_updates',
                     'storage_loss_enabled', 'frame_training_plan', 'regional_color_birth',
                     'regional_alpha_repair', 'regional_postfit_admission',
                     'population_rate_exempt', 'joint_old_opacity'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'birth.{name} must be boolean')
        for name in ('new_xyz_step_extent', 'new_rotation_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'birth.{name} must be finite and positive')
        if (isinstance(self.opacity_threshold, bool) or not isinstance(self.opacity_threshold, (int, float)) or
                not math.isfinite(self.opacity_threshold) or not 0 < self.opacity_threshold < 1):
            raise ValueError('birth.opacity_threshold must be in (0, 1)')
        if self.photometric_loss not in (None, 'l1', 'rmse'):
            raise ValueError('birth.photometric_loss must be null, l1 or rmse')
        if self.initial_alpha is not None and (isinstance(self.initial_alpha, bool) or
                not isinstance(self.initial_alpha, (int, float)) or not math.isfinite(self.initial_alpha) or
                not 0 < self.initial_alpha < 1):
            raise ValueError('birth.initial_alpha must be in (0, 1) or null')
        if (isinstance(self.population_loss_weight, bool) or
                not isinstance(self.population_loss_weight, (int, float)) or
                not math.isfinite(self.population_loss_weight) or self.population_loss_weight < 0):
            raise ValueError('birth.population_loss_weight must be finite and non-negative')
        if self.population_rate_exempt and not self.frame_training_plan:
            raise ValueError('Population rate exemption requires the frame training plan')
        if self.joint_old_opacity and self.radiance_mode != 'adaptive_residual_birth':
            raise ValueError('Joint old opacity requires adaptive_residual_birth')
        if self.population_loss_weight and not self.joint_old_opacity:
            raise ValueError('Population pressure requires joint old opacity')


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
    full_update: FullUpdateConfig = field(default_factory=FullUpdateConfig)


MOTION_IMAGE_SCHEDULE = 'bootstrap_aligned_v1'


def motion_image_schedule_compatible(saved, enabled, motion_steps, bootstrap_steps):
    """Legacy resumes are safe only when their image boundaries are identical."""
    return (not enabled or saved == MOTION_IMAGE_SCHEDULE or
            (saved is None and bootstrap_steps == 2 * motion_steps // 5))




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






def stable_covariance_backward_active(module, training):
    """Use the stable backward during full-attribute fitting and legacy tails."""
    if not (getattr(module.config, 'stable_tail_covariance_backward', False) and training and
            module.frame_prepared and not module.frame_finalized and
            regional_attribute_repair_active(module)):
        return False
    step = module.trainer.global_step
    return (step > module.motion_steps or
            (getattr(module.config.full_update, 'regional_full_200', False) and
             1 <= step <= module.motion_steps))








def decoder_exact_rotation(motion, rotation):
    """Use the CPU codec's quaternion values while preserving training gradients."""
    _, decoded = motion.decoded_survivor_pose()
    if decoded.shape != rotation.shape:
        raise ValueError('Decoded motion rotation does not match training pose')
    decoded = decoded.to(device=rotation.device, dtype=rotation.dtype)
    return decoded + (rotation - rotation.detach())






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
    """组织 GS 状态、逐帧训练、渲染、损失和码流输出。"""

    def setup(self):
        cfg = self.config
        self.per_frame_training = PerFrameTraining(self)
        self.per_frame_training.validate_per_frame_training(cfg)
        if (not cfg.birth.frame_training_plan or not uses_early_replacement(cfg) or
                cfg.birth.regional_color_birth or cfg.birth.regional_postfit_admission or
                cfg.birth.regional_alpha_repair or cfg.birth.storage_loss_enabled):
            raise ValueError('New Gaussian training has been removed; use the fixed-population frame plan')
        if not isinstance(cfg.ablation_motion_only, bool):
            raise ValueError('ablation_motion_only must be boolean')
        validate_initial_opacity_threshold(cfg.initial_opacity_threshold)
        cfg.birth.validate(cfg.rigid_tree.motion_steps)
        cfg.opacity_update.validate()
        cfg.feature_dc_update.validate()
        cfg.logscale_update.validate()
        cfg.full_update.validate()
        if cfg.full_update.regional_full_200 and (cfg.sh_degree != 1 or not cfg.rigid_tree.rotation_updates or
                cfg.opacity_update.mode != 'alpha' or cfg.feature_dc_update.enabled or
                cfg.logscale_update.enabled):
            raise ValueError('Regional full experiment requires SH1, rotations, alpha updates and no other shared attributes')
        if cfg.feature_dc_update.enabled and cfg.feature_dc_update.sharing == 'spatial_region':
            raise ValueError('Separate regional DC updates were removed; use regional attributes instead')
        if cfg.full_update.enabled and (cfg.feature_dc_update.enabled or cfg.logscale_update.enabled):
            raise ValueError('Full updates cannot be combined with separate feature or log-scale updates')
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
                    cfg.rigid_tree.spatial_gate.final_joint_refine_steps):
                raise ValueError('Regional attribute repair requires fixed population and a supported frame plan')
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
        renderer_type = ContributionRenderer if cfg.effective_motion_observations else Renderer
        self.renderer = renderer_type(cfg.renderer, self.gaussians)
        self.motion = FullUpdateSpatialMotionTraining(cfg.rigid_tree)
        self.full_attribute_update = None
        self.regional_full_attributes = None
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
        self.motion.regional_full_200 = bool(active and cfg.regional_full_200)
        self.motion.refine_image_gate_ste = bool(getattr(self, 'gate_refine_image_ste', False))

    def full_update_prepare_step(self, step):
        if self.full_update_active and self.full_attribute_update is not None:
            start = 101 if self.config.full_update.early_joint_repair else self.motion_steps + 1
            trainable = step >= start
            for name, parameter in self.full_attribute_update.parameters.items():
                enabled = trainable and not (getattr(self.config.full_update, 'dense_color_200', False) and
                    not getattr(self.config.full_update, 'regional_full_200', False) and name == 'logscale')
                parameter.requires_grad_(enabled)
                if not enabled:
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
        regional_full = getattr(self, 'regional_full_attributes', None)
        if regional_full is not None:
            regional_full.prepare(step, self.config.rigid_tree.motion_lr_final_ratio)

    def full_update_after_step(self, step):
        regional_full = getattr(self, 'regional_full_attributes', None)
        if regional_full is not None:
            regional_full.after_step(step, self._rate_byte_price(step))
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
        return dict(logscale=self.gaussians._scaling, sh_dc=dc, sh_rest=self.gaussians._features_rest)

    def on_save_checkpoint(self, ckpt):
        ckpt['gate_refine_image_ste'] = bool(getattr(self, 'gate_refine_image_ste', False))
        if getattr(self.config.birth,'frame_training_plan',False) and self.frame_prepared:
            self.per_frame_training._plan_record_phase_time()
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
        ckpt['births'] = None
        ckpt['birth_source_rows'] = None
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
            if self.config.full_update.regional_full_200:
                if self.regional_full_attributes is None:
                    raise ValueError('Checkpoint lost regional full attributes')
                ckpt['regional_full_attributes'] = self.regional_full_attributes.capture()
        ckpt['birth_handoff_sources'] = (self.birth_handoff_sources.detach().cpu()
            if self.birth_handoff_sources is not None else None)
        ckpt['birth_transaction'] = None
        ckpt['birth_cohort_state'] = dict(index=getattr(self, 'birth_cohort', 0),
            start_step=getattr(self, 'birth_cohort_start', getattr(self, 'motion_steps', 0)),
            end_step=getattr(self, 'birth_end_step', None),
            coarse_snapshot=getattr(self, 'coarse_birth_snapshot', None))
        ckpt['birth_terminal_state'] = dict(active=getattr(self, 'birth_terminal', False),
            selected=clone_tensor_tree(getattr(self, 'selected_birth_state', None), 'cpu'))
        ckpt['birth_appearance_gradient'] = (
            self.birth_appearance_gradient.detach().cpu()
            if self.birth_appearance_gradient is not None else None)
        if uses_early_replacement(self.config):
            self.per_frame_training._early_sync_opacity_state()
            ckpt['quantized_opacity_update']=self.opacity_update.capture()
            ckpt['early_replacement_state']=clone_tensor_tree(dict(
                initialized=self.early_initialized,birth_updates=self.early_birth_updates,
                shared_steps=self.early_shared_steps,selected=self.early_selected,
                pinned_rows=self.early_pinned_rows,pinned_values=self.early_pinned_values,
                focus_weights=self.early_focus_weights,
                plan_used_quota=getattr(self,'_plan_used_quota',0),
                plan_events=getattr(self,'_plan_events',[]),
                plan_admission_camera_names=getattr(self,'_plan_admission_camera_names',None)),'cpu')
        regional_attributes = getattr(self, '_plan_regional_attribute_state', None)
        if (self.full_update_active and self.config.full_update.regional_repair and
                self.frame_prepared and not self.frame_finalized and regional_attributes is not None):
            ckpt['regional_attribute_repair_state'] = clone_tensor_tree(regional_attributes, 'cpu')

    def on_load_checkpoint(self, ckpt):
        cache = getattr(self, '_training_ssim_cache', None)
        if cache is not None:
            cache.clear()
        self._plan_regional_attribute_state = None
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
        from .continue_trainer import relocate_checkpoint_reference
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
        if saved_config != requested_config:
            raise ValueError('Rigid-tree checkpoint needs the same module configuration')
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
        if ckpt.get('birth_transaction') is not None:
            raise ValueError('Checkpoints containing new-GS transactions cannot be resumed')
        self.birth_transaction = None
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
            self.regional_full_attributes = None
            if getattr(self, 'full_update_active', False) and not self.frame_finalized:
                self.full_attribute_update = ResidualAppearanceAttributes(self.config.full_update)
                self.full_attribute_update.restore(ckpt.get('full_attribute_update'),
                    self.motion.optimizer, self.device, len(self.gaussians._xyz))
                if self.config.full_update.regional_full_200:
                    self.regional_full_attributes = RegionalAttributesUpdate(self.motion, self.config.full_update,
                        self.config.gaussian, self.config.opacity_update)
                    self.regional_full_attributes.restore(ckpt.get('regional_full_attributes'))
        saved_births = ckpt.get('births')
        if saved_births is not None:
            state = saved_births.get('frame_plan_birth') if isinstance(saved_births, dict) else None
            if (state is None or len(state.get('source_rows', ())) or
                    any(len(batch['parameters']['_xyz']) for batch in state.get('cohorts', []))):
                raise ValueError('Checkpoints containing new Gaussians cannot be resumed for training')
        if uses_early_replacement(self.config):
            state=clone_tensor_tree(ckpt.get('early_replacement_state') or {},self.device)
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
                self.per_frame_training._early_attach_opacity()
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
        self.early_initialized=False;self.early_birth_updates=0;self.early_shared_steps=0
        self.early_selected=None;self.early_pinned_rows=None;self.early_pinned_values=None
        self.early_focus_weights={}
        self._plan_used_quota=0;self._plan_events=[]
        self._plan_admission_camera_names=None
        self._plan_regional_attribute_state=None
        self._plan_error_owner=None
        self.per_frame_training.reset_frame_state()
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
            self.per_frame_training._early_attach_opacity()
        self.full_attribute_update = None
        self.regional_full_attributes = None
        if getattr(self, 'full_update_active', False):
            self.full_attribute_update = ResidualAppearanceAttributes(self.config.full_update)
            args = self.config.gaussian
            rates = _full_attribute_learning_rates(args)
            self.full_attribute_update.setup(self.full_update_reference(), self.motion.optimizer,
                tuple(rates[name] for name in FULL_ATTRIBUTE_FIELDS))
            if self.config.full_update.regional_full_200:
                self.regional_full_attributes = RegionalAttributesUpdate(self.motion, self.config.full_update,
                    self.config.gaussian, self.config.opacity_update)

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
            self.per_frame_training._early_prepare_rates(self.trainer.global_step)
            return
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
                ((getattr(self.config, 'ablation_motion_only', False) or
                  getattr(getattr(self.config, 'full_update', None), 'dense_color_200', False)) and self.motion_committed)):
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
                predictor_only=False):
        auxiliary = proposal_only or predictor_only
        if (proposal_only and predictor_only) or (auxiliary and (
                not training or not self.frame_prepared or self.motion_committed)):
            raise ValueError('Auxiliary render is only available during motion training')
        old_model = self.renderer.gaussians
        effective = getattr(self.config, 'effective_motion_observations', False)
        previous_collection = getattr(self.renderer, 'collect_contributions', False)
        previous_stable_backward = getattr(self.renderer, 'stable_backward', False)
        stable_tail = getattr(self.config, 'stable_tail_covariance_backward', False)
        if stable_tail:
            self.renderer.stable_backward = stable_covariance_backward_active(self, training)
        if effective:
            self.renderer.collect_contributions = bool(training and not auxiliary and self.frame_prepared and
                (not self.motion_committed or (uses_residual_birth(self.config) and
                  not self.frame_finalized)))
        try:
            if fitting_old_motion(self):
                xyz, rotation = (self.motion.predictor_pose() if predictor_only else
                                 self.motion.proposal_pose() if proposal_only else
                                 self.motion(self.trainer.global_step, training=training))
                if (self.motion_committed and ((self.trainer.global_step > self.motion_steps and
                        getattr(self.config.full_update, 'shared_region_orientation', False)) or
                        getattr(self.config.full_update, 'regional_full_200', False) or
                        (getattr(self, 'full_update_active', False) and
                         self.trainer.global_step == self.total_steps))):
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
                values = full.apply(dict(logscale=self.gaussians._scaling,
                    sh_dc=dc, sh_rest=self.gaussians._features_rest))
                if auxiliary:
                    values = {name: value.detach() for name, value in values.items()}
                self.gaussians.rigid_logscale = values['logscale']
                self.gaussians.rigid_sh_dc = values['sh_dc']
                self.gaussians.rigid_sh_rest = values['sh_rest']
            regional_full = getattr(self, 'regional_full_attributes', None)
            if regional_full is not None and self.frame_prepared and not self.frame_finalized:
                gate_weights = regional_full.gate_image_weights()
                for name, rigid, base in (('logscale', 'rigid_logscale', '_scaling'),
                        ('sh_dc', 'rigid_sh_dc', '_features_dc'), ('sh_rest', 'rigid_sh_rest', '_features_rest')):
                    reference = getattr(self.gaussians, rigid, None)
                    if reference is None:
                        reference = getattr(self.gaussians, base)
                    setattr(self.gaussians, rigid, regional_full.apply(name, reference, gate_weights=gate_weights))
                alpha = getattr(self.gaussians, 'rigid_alpha', None)
                if alpha is not None:
                    logits = self.opacity_update.canonical_logits() if self.motion_committed else None
                    self.gaussians.rigid_alpha = regional_full.alpha(alpha, self.motion_committed, logits,
                                                                    gate_weights=gate_weights)
            if getattr(self, '_capture_regional_full_debug', False):
                g = self.gaussians
                self._regional_full_debug = {name: getattr(g, name).detach().cpu().clone() for name in
                    ('get_xyz', 'get_rotation', 'get_scaling', 'get_opacity', 'get_features')}
            package = self.renderer(viewpoint, self.background)
            return package
        finally:
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
        return compute_training_loss(self, viewpoint, step, regularization_weight,
                                     early_replacement=uses_early_replacement(self.config))

    def _rate_byte_price(self, step):
        from .model.storage_rate import frame_byte_price
        return frame_byte_price(self, step)

    def _opacity_rate_price(self, step):
        from .model.storage_rate import opacity_byte_price
        return opacity_byte_price(self, step)

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
            data_config = self.datamodule.config
            indices = range(len(dataset.items))
            ordered = sorted((Path(dataset.items[index].path).name, index) for index in indices)
            names = [name for name, _ in ordered]
            if len(set(names)) != len(names):
                raise ValueError('Training camera filenames must be unique within a frame')
            if len(names) < 3:
                raise ValueError('Residual birth needs at least three training cameras')
            fitting = validation = tuple(names)
            expected = getattr(data_config, 'expected_image_size', None)
            views = {name: _collate_single_view([dataset[index]], expected)
                     for name, index in ordered}
            trainer = getattr(self, 'trainer', None)
            cache_on_device = getattr(getattr(trainer, 'config', None), 'cache_data_on_device', True)
            self._transaction_views = _FrameCameraViews(views, self.device, cache_on_device)
            self._transaction_fit_names = fitting
            self._transaction_validation_names = validation
        return self._transaction_views

    @torch.no_grad()
    def post_training_step(self, package):
        packages = package if isinstance(package, list) else [package]
        self.per_frame_training._early_post_training_step(packages, self.trainer.global_step)


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
        if self.full_update_active:
            views = self.transaction_camera_views()
            camera = getattr(self.trainer, 'diagnostic_train_camera', None) or sorted(views)[0]
            if camera not in views or camera == 'cam00.png':
                raise ValueError('Forward/decode check requires a training camera')
            full_debug = bool(self.config.full_update.regional_full_point_budget or
                os.environ.get('MY4DGS_DEBUG_ATTRGATE') == '1')
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
        from .model.frame_stream_encode import encode_frame, add_frame_packet_manifest
        encoded = encode_frame(self, old_raw, full_reference, logscale_reference,
                               logscale_update, full)
        (stream_dir, regional_full_info, regional_full, raw, sources, motion_path, motion_info,
         activity, rate_terms, logscale_info, opacity_info,
         feature_info, full_info, point_mask_info) = encoded
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
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self.gaussians, attr,
                    torch.nn.Parameter(raw[name].detach().clone()))
        self.gaussians.training_setup(self.config.gaussian)
        self.gaussians.max_radii2D = self.gaussians._xyz.new_zeros(len(raw['xyz']))
        self._freeze_base()
        self.renderer.gaussians = self.gaussians
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
            birth_cohorts=0,
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
        gate_restart = self.config.rigid_tree.spatial_gate.joint_gate_start_step
        manifest['frame_training_plan_version'] = 3
        manifest['birth_cohorts'] = self.birth_control['early_replacement']['actual_birth_cohorts']
        manifest['birth_events'] = []
        manifest['training_step_accounting'] = dict(
            global_iterations=200, shared_adam_steps=self.early_shared_steps,
            shared_gate=self.config.rigid_tree.shared_discovery_steps,
            joint_gate_residual_early=100-self.config.rigid_tree.shared_discovery_steps,
            fixed_gate_residual=gate_restart-101, joint_gate_residual=181-gate_restart, final_motion=20,
            motion_optimizer_steps=200, old_opacity_all_rows_steps=100,
            all_point_color_steps=100, gradient_probe_steps=[],
            all_point_full_attribute_steps=100, regional_full_attribute_steps=100,
            birth_steps=[], cleanup_steps=[], newborn_updates=[], extra_SGD_steps=0)
        manifest['dense_color_200'] = dict(fields=['logscale', 'sh_dc', 'sh_rest'],
            start_step=101, end_step=200, selection='all points',
            reference='previous decoded frame', train_views_per_step=1)
        manifest['regional_full_attributes'] = (Path(regional_full_info['path']).name if regional_full_info else None)
        manifest['regional_full_timing'] = 'after_point_attributes_before_births'
        manifest['regional_full_rate_config'] = dict(
            attribute_aware_gate=self.config.full_update.attribute_aware_gate,
            gate_cost=('motion plus estimated regional attributes and point residuals after candidate rebase'
                       if self.config.full_update.attribute_aware_gate else 'motion only'),
            regional_opacity_l1_weight=regional_full.opacity_l1_weight,
            point_sh_rest_rate_multiplier=self.config.full_update.point_sh_rest_rate_multiplier,
            joint_gate_start_step=gate_restart,
            gate_reopen_logit_limit=self.config.rigid_tree.spatial_gate.gate_reopen_logit_limit,
            gate_alpha_rebase='alpha-grid approximation after regional logit update')
        if getattr(self, 'gate_refine_image_ste', False):
            manifest['gate_refine_image_ste'] = (f'hard cut forward; candidate-total path-probability gate derivative '
                f'at steps {self.config.rigid_tree.spatial_gate.joint_gate_start_step}-180')
        add_frame_packet_manifest(manifest, encoded, self, full_reference, logscale_update)
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
            from .model.frame_stream_decode import (decode_frame)
            replayed_state, replayed_raw = decode_frame(stream_dir, self.current_frame, prior, prior_raw)
            if replayed_state.digest().hex() != manifest['decoded_state_sha256'] or reference_digest(replayed_raw) != raw_digest:
                raise ValueError('Bounded attribute packet replay differs from the materialized training state')
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                target = getattr(self.gaussians, attr)
                target.copy_(replayed_raw[name].to(device=target.device, dtype=target.dtype))
            self.motion.codec_previous_state = replayed_state
        if forward_check is not None:
            self._capture_regional_full_debug = full_debug
            after=self.forward(views[forward_check['camera']],training=False)['render'].detach().clamp(0,1)
            debug_after = getattr(self, '_regional_full_debug', None) if self._capture_regional_full_debug else None
            self._capture_regional_full_debug = False
            difference=(forward_check.pop('before')-after).abs()
            forward_check['max_abs'] = float(difference.max())
            forward_check['mean_abs'] = float(difference.mean())
            forward_check['tolerance'] = 1e-5
            if full_debug and debug_before is not None and debug_after is not None:
                forward_check['state_check'] = {name: dict(
                    max_abs=float((value-debug_after[name]).abs().max()),
                    changed_scalars=int(value.ne(debug_after[name]).sum()))
                    for name, value in debug_before.items()}
            if not bool(torch.isfinite(difference).all()) or forward_check['max_abs'] > forward_check['tolerance']:
                if full_debug:
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
            logscale_update=logscale_info, regional_full_attributes=regional_full_info, color_point_mask=point_mask_info, forward_decode_check=forward_check,
            motion_observation_diagnostics=dict(
                kind=('actual_compositing' if self.config.effective_motion_observations else 'frustum'),
                frustum_samples=self.birth_control.get('motion_frustum_samples', 0),
                used_samples=self.birth_control.get('motion_observed_samples', 0),
                old_gs_with_no_motion_observation=int(self.motion.observations.eq(0).sum()),
                scope='surviving GS; current motion optimization views; excludes birth and heldout renders'),
            stage_image_metrics=self.stage_image_metrics,
            encoded_bytes=encoded.payload_bytes,
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
            semantics='Spatial gate motion and existing Gaussian attributes; shared first frame counted once. '
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
            self.frame_report['semantics'] = ('Spatial gate motion and existing Gaussian attributes; '
                'keyframe old log-scale, DC and higher-order SH use an additional coded delta.')
        if regional_full is not None:
            terms = regional_full.rate_terms(self._rate_byte_price(self.total_steps))
            self.frame_report['regional_full_rate_diagnostics'] = dict(
                semantics='raw sparse row estimate excluding packet headers; post-Adam proximal penalties',
                raw_bytes_surrogate=float(terms['raw_bytes_surrogate']), l1_loss=float(terms['l1_loss']),
                loss=float(terms['loss']), actual_packet_bytes=regional_full_info['archive_bytes'] if regional_full_info else 0,
                opacity_l1_weight=regional_full.opacity_l1_weight,
                fields={name:{key:(int(value) if key == 'active_regions' else float(value))
                    for key,value in field.items()} for name,field in terms['fields'].items()})
        if self.full_update_active and self.config.full_update.color_only and not self.config.full_update.stop_after_motion:
            self.frame_report['semantics'] = ('Motion and opacity frozen after step200; quantized regional DC and selected point attributes fitted through step300; previous decoded frame used as reference.')
        if self.full_update_active and self.config.full_update.stop_after_motion:
            self.frame_report['semantics'] = (('Regional and per-point quantized pose, scale, opacity and SH updates through step200; '
                if self.config.full_update.regional_full_200 else 'Spatial gate motion and opacity through step200; all-point quantized SH color updates during steps 101–200; ')
                + 'fixed population; no post-motion updates; shared first frame counted once.')
        if getattr(self.config, 'ablation_motion_only', False):
            self.frame_report['semantics'] = ('200 steps of shared gate motion, per-point motion residuals '
                'and quantized opacity updates; no sparse attribute repair or births.')
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
