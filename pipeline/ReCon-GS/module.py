from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import torch
from absl import logging
from jaxtyping import Bool, Float
from torch import Tensor
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.data.types import TensorSpace
from libgs.metric import psnr, ssim, lpips
from libgs.model.merged_gaussian import MergedGaussianModel
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import Renderer, RendererConfig
from libgs.renderer.network_gui import interact_with_gui
from libgs.utils.time import Timer
from libgs.utils.general import quaternion_multiply

from .data import DataModule
from .model.deformation import Deformation, DeformConfig
from .model.gated_four import GatedFourDeformation
from .model.gaussian import GaussianModel
from .model.selective_motion import SelectiveMotion, SelectiveMotionConfig
from .model.selective_diagnostics import summarize, concise
from .model.learned_selective import LearnedSelectiveMotion, LearnedSelectiveConfig
from .model.residual_motion import (QuantizedResidualConfig, QuantizedResidualMotion,
    QuantizedNewGaussianView, RAW_FIELDS, RAW_ATTRIBUTES,
    model_raw, materialize_quantized_gaussians, reference_digest, survivor_rows,
    inverse_unit_quaternion)
from .model.gated_residual import GatedResidualMotion, actual_motion_statistics
from .model.sparse_hierarchical import SparseHierarchicalConfig, SparseHierarchicalMotion

import json
import csv
import numpy as np
import time
import math

QUANTIZED_MOTION_MODES = {'quantized_residual', 'sparse_hierarchical'}

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
class DensifyConfig:
    from_step: int = 500
    until_step: int = 5000
    interval: int = 100
    grad_threshold: float = 2e-4
    opacity_reset_interval: int = 30000
    max_gaussians: int = 10000000


@dataclass
class QuantizedDiagnosticsConfig:
    deletion_damage_enabled: bool = False
    deletion_damage_view_name: Optional[str] = None
    deletion_damage_every_n_frames: Optional[int] = None
    control_quality_enabled: bool = False
    control_quality_camera_names: Optional[List[str]] = None
    prediction_quality_enabled: bool = False
    gradient_steps: Optional[List[int]] = None
    actual_motion_enabled: bool = False
    translation_extent_thresholds: Optional[List[float]] = None
    rotation_radian_thresholds: Optional[List[float]] = None

    def validate(self):
        if self.control_quality_camera_names is not None:
            if (not self.control_quality_camera_names or
                    len(set(self.control_quality_camera_names)) != len(self.control_quality_camera_names)):
                raise ValueError('control_quality_camera_names must be nonempty and unique')
        if self.actual_motion_enabled:
            for name in ('translation_extent_thresholds', 'rotation_radian_thresholds'):
                values = getattr(self, name)
                if not values or any(not math.isfinite(v) or v < 0 for v in values):
                    raise ValueError(f'{name} requires finite nonnegative thresholds')
        if self.gradient_steps is not None and any(
                isinstance(step, bool) or not isinstance(step, int) or step < 1
                for step in self.gradient_steps):
            raise ValueError('gradient_steps must contain positive integer steps')
        if not self.deletion_damage_enabled:
            return
        interval = self.deletion_damage_every_n_frames
        if not self.deletion_damage_view_name:
            raise ValueError('Deletion diagnostics require an explicit test camera filename')
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError('Deletion diagnostic interval must be a positive integer')


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 3
    random_background: bool = False  # for training
    lambda_dssim: float = 0.2
    lambda_deform: float = 1.0
    noise_scale: Optional[float] = 0.01
    saving_gs_steps: List[int] = field(default_factory=lambda: [5000])
    saving_gs_every_n_frames: int = 1000
    saving_gs_frames: Optional[List[int]] = None
    num_saving_images: int = 5
    full_eval: bool = True
    lpips_required: bool = False
    merge_to_base: bool = True
    re_hierarchization: bool = True
    densify: DensifyConfig = field(default_factory=DensifyConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    gaussian_stage2: GaussianConfig = field(default_factory=GaussianConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    deformation: DeformConfig = field(default_factory=DeformConfig)
    motion_mode: str = 'legacy'
    sparse_export_index_encoding: str = 'per_row'
    selective_motion: SelectiveMotionConfig = field(default_factory=SelectiveMotionConfig)
    learned_selective_motion: LearnedSelectiveConfig = field(default_factory=LearnedSelectiveConfig)
    quantized_residual: QuantizedResidualConfig = field(default_factory=QuantizedResidualConfig)
    sparse_hierarchical: SparseHierarchicalConfig = field(default_factory=SparseHierarchicalConfig)
    quantized_diagnostics: QuantizedDiagnosticsConfig = field(default_factory=QuantizedDiagnosticsConfig)
    selective_diagnostics_config: Optional[str] = 'config/diagnostics/selective_motion.yaml'


class Module(BaseModule):
    datamodule: DataModule
    trainer: "Trainer"
    gaussians: GaussianModel
    background: torch.Tensor  # TODO remove
    timer: Timer

    @property
    def current_frame(self) -> int:
        return self.datamodule.current_frame

    @property
    def num_steps(self) -> int:
        return self.trainer.num_steps

    @property
    def motion_config(self):
        return (self.config.learned_selective_motion if self.config.motion_mode == 'learned_selective'
                else self.config.selective_motion)

    def setup(self):
        if self.config.motion_mode not in {'legacy', 'selective', 'learned_selective', *QUANTIZED_MOTION_MODES}:
            raise ValueError('Unsupported motion_mode')
        if self.config.sparse_export_index_encoding not in {'per_row', 'shared_table'}:
            raise ValueError('sparse_export_index_encoding must be per_row or shared_table')
        if self.config.motion_mode in {'selective', 'learned_selective'}:
            self.motion_config.validate()
            if self.trainer.config.num_incr_steps != self.motion_config.total_steps:
                raise ValueError('num_incr_steps must equal selective stage budgets')
            if self.config.deformation.num_stage1_steps != self.motion_config.motion_steps:
                raise ValueError('num_stage1_steps must equal selective motion stage budget')
            if self.config.random_background:
                raise ValueError('Selective comparisons require a fixed background')
            self.selective = (LearnedSelectiveMotion(self.motion_config) if self.config.motion_mode == 'learned_selective'
                              else SelectiveMotion(self.motion_config))
            self.change_reference = {}
        if self.config.motion_mode in QUANTIZED_MOTION_MODES:
            cfg = (self.config.sparse_hierarchical
                   if self.config.motion_mode == 'sparse_hierarchical'
                   else self.config.quantized_residual)
            # Shared post-processing and birth code consumes one residual-config
            # alias. Keep old quantized mode pointed at its original config.
            self.config.quantized_residual = cfg
            cfg.validate()
            self.config.quantized_diagnostics.validate()
            if self.trainer.config.num_incr_steps != cfg.total_steps or self.config.deformation.num_stage1_steps != cfg.motion_steps:
                raise ValueError('Quantized residual budgets must agree with trainer and deformation stage boundary')
            if self.config.random_background:
                raise ValueError('Quantized residual comparisons require a fixed background')
            if self.config.motion_mode == 'sparse_hierarchical':
                if self.config.selective_diagnostics_config is not None:
                    raise ValueError('sparse_hierarchical does not support legacy selective overlays; '
                                     'set selective_diagnostics_config: null')
                self.residual_motion = SparseHierarchicalMotion(cfg)
            else:
                self.residual_motion = (GatedResidualMotion(cfg) if cfg.enable_learned_gates
                                        else QuantizedResidualMotion(cfg))
            self.quant_previous_new_count = 0
            self.quant_first_frame_gs_count = None
            self.quant_previous_reference = None
            self.quant_previous_sha256 = None
            self.quant_deleted_indices = None
            self.quant_report = None
            self.quant_predictor_tree = None
            self.quant_finalized = False
            self.quant_stats_window_initialized = False
            self.quant_birth_control = None
            self.quant_point_birth_frame = None
        white_background = self.datamodule.config.white_background
        bg_color_fn = torch.ones if white_background else torch.zeros
        self.register_buffer("background", bg_color_fn(3, device=self.device))

        self.gaussians = GaussianModel(self.config.sh_degree)
        self.gaussians.create_from_pcd(
            self.datamodule.scene.point_cloud, self.datamodule.cameras_extent
        )
        self.gaussians.training_setup(self.config.gaussian)
        self.gaussians_incr = GaussianModel(self.config.sh_degree)
        self.renderer = Renderer(self.config.renderer, self.gaussians)
        deformation_cls = (GatedFourDeformation
                           if self.config.deformation.hierarchy_mode == "nested" and
                           self.config.deformation.num_grid_levels == 4 else Deformation)
        self.deform = deformation_cls(self.config.deformation)
        self.timer = Timer()
        self._sparse_motion_export = None

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["gaussians"] = self.gaussians.capture()
        if not self.trainer.is_first_frame:
            if self.config.motion_mode in {'selective', 'learned_selective', *QUANTIZED_MOTION_MODES} and self.gaussians_incr.optimizer is None:
                ckpt['gaussians_incr'] = None
            else:
                ckpt["gaussians_incr"] = self.gaussians_incr.capture()
        ckpt["deformation"] = self.deform.capture()
        if self.config.motion_mode in {'selective', 'learned_selective'}:
            ckpt['selective_motion'] = self.selective.capture()
            if self.config.motion_mode == 'learned_selective':
                ckpt['learned_change_reference'] = self.change_reference

        if self.config.motion_mode in QUANTIZED_MOTION_MODES:
            ckpt['motion_mode'] = self.config.motion_mode
            ckpt['quantized_residual_motion'] = self.residual_motion.capture()
            ckpt['quantized_frame_state'] = dict(previous_new_count=self.quant_previous_new_count,
                first_frame_gs_count=self.quant_first_frame_gs_count, previous_reference=self.quant_previous_reference,
                deleted_indices=self.quant_deleted_indices, report=self.quant_report, finalized=self.quant_finalized,
                stats_window_initialized=self.quant_stats_window_initialized,
                birth_control=self.quant_birth_control,
                point_birth_frame=self.quant_point_birth_frame)

    def on_load_checkpoint(self, ckpt: dict):
        if (not self.trainer.is_first_frame and ckpt.get('motion_mode') in QUANTIZED_MOTION_MODES
                and self.config.motion_mode != ckpt.get('motion_mode')):
            raise ValueError('Incremental quantized checkpoint cannot resume a different motion mode')
        self.gaussians.restore(ckpt["gaussians"], self.config.gaussian)
        if not self.trainer.is_first_frame:
            if self.config.motion_mode in {'selective', 'learned_selective', *QUANTIZED_MOTION_MODES} and ckpt.get('gaussians_incr') is None:
                self.gaussians_incr = GaussianModel(self.config.sh_degree)
            else:
                self.gaussians_incr.restore(
                    ckpt["gaussians_incr"], self.config.gaussian_stage2
                )
        if self.config.motion_mode in QUANTIZED_MOTION_MODES:
            packet = ckpt.get('quantized_residual_motion')
            if not self.trainer.is_first_frame and packet is None:
                raise ValueError('Incremental quantized residual resume requires matching motion-mode checkpoint')
            self.residual_motion.restore(packet)
            if packet is not None:
                frame_state = ckpt['quantized_frame_state']
                self.quant_previous_new_count = frame_state['previous_new_count']
                self.quant_first_frame_gs_count = frame_state['first_frame_gs_count']
                self.quant_previous_reference = frame_state['previous_reference']
                self.quant_previous_sha256 = reference_digest(self.quant_previous_reference)
                self.quant_deleted_indices = frame_state['deleted_indices']
                self.quant_report, self.quant_finalized = frame_state['report'], frame_state['finalized']
                self.quant_stats_window_initialized = frame_state.get(
                    'stats_window_initialized', packet['last_step'] >= self.quant_stats_window_start())
                self.quant_birth_control = frame_state.get('birth_control')
                self.quant_point_birth_frame = frame_state.get('point_birth_frame')
                if self.quant_point_birth_frame is None:
                    self.quant_point_birth_frame = torch.full(
                        (len(self.gaussians.get_xyz),), -1, dtype=torch.long)
                    if self.quant_finalized and self.quant_previous_new_count:
                        self.quant_point_birth_frame[-self.quant_previous_new_count:] = self.current_frame
                if self.quant_birth_control is None:
                    self.quant_birth_control = self.new_quant_birth_control()
                    if self.config.quantized_residual.enable_incremental:
                        self.quant_birth_control['stats_steps_observed'] = max(0, min(
                            packet['last_step'], self.config.quantized_residual.motion_steps)
                            - self.quant_birth_control['stats_window_start'] + 1)
                else:
                    defaults = self.new_quant_birth_control()
                    defaults.update(self.quant_birth_control)
                    self.quant_birth_control = defaults
                self.freeze_quantized_base()
                if self.residual_motion.committed:
                    for parameter in self.residual_motion.parameters():
                        parameter.requires_grad_(False)
                self.renderer.gaussians = self.gaussians
            return
        if self.config.motion_mode in {'selective', 'learned_selective'}:
            packet = ckpt.get('selective_motion')
            if not self.trainer.is_first_frame and packet is None:
                raise ValueError('An incremental selective resume requires selective_motion state')
            if packet is not None and packet.get('mode', 'selective') != self.config.motion_mode:
                raise ValueError('Incremental checkpoint motion mode does not match configured motion_mode')
            self.selective.restore(packet)
            self.change_reference = ckpt.get('learned_change_reference', {})
            if packet is not None and self.selective.committed and self.motion_config.enable_incremental:
                self.renderer.gaussians = MergedGaussianModel([self.gaussians, self.gaussians_incr])
        else:
            self.deform.restore(ckpt["deformation"])

    @torch.no_grad()
    def load_sparse_motion_export(self, motion_path, incremental_ply=None):
        """Load one sparse-v2/v3 frame for playback on the pre-frame base model.

        The caller must advance the base Gaussian state to the start of this
        frame. The optional PLY contains only this frame's newly added points.
        """
        base = self.gaussians.get_xyz_ori
        if getattr(self.config, 'motion_mode', 'legacy') == 'sparse_hierarchical':
            moved_xyz, moved_rotation = SparseHierarchicalMotion.decode_export_motion(
                motion_path, base, self.gaussians._rotation)
            moved_xyz = torch.as_tensor(moved_xyz, device=base.device, dtype=base.dtype)
            moved_rotation = torch.as_tensor(moved_rotation, device=base.device, dtype=base.dtype)
            base_rotation = self.gaussians.get_rotation
            delta_rotation = quaternion_multiply(
                inverse_unit_quaternion(base_rotation), moved_rotation)
            field = torch.cat((moved_xyz - base,
                               torch.nn.functional.normalize(delta_rotation, dim=-1)), dim=-1)
        else:
            if not isinstance(self.deform, GatedFourDeformation):
                raise RuntimeError("Sparse gated-four playback requires GatedFourDeformation")
            field = GatedFourDeformation.decode_export_motion(motion_path)
        if field.shape != (len(base), 7):
            raise ValueError(f"Sparse motion has shape {field.shape}; expected ({len(base)}, 7)")
        motion = torch.as_tensor(field, dtype=base.dtype, device=base.device).contiguous()
        delta_rotation = torch.nn.functional.normalize(motion[:, 3:], dim=-1)
        incremental = None
        if incremental_ply is not None:
            incremental = GaussianModel(self.config.sh_degree)
            incremental.load_ply(str(incremental_ply))
            # libgs GaussianModel is a plain Python container, not nn.Module;
            # load_ply hardcodes CUDA, so move its renderer tensors explicitly.
            for name in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                         '_scaling', '_rotation', 'max_radii2D',
                         'xyz_gradient_accum', 'denom'):
                value = getattr(incremental, name, None)
                if torch.is_tensor(value):
                    dtype = base.dtype if value.is_floating_point() else value.dtype
                    setattr(incremental, name, value.to(device=base.device, dtype=dtype))
        self._sparse_motion_export = (motion[:, :3], delta_rotation, incremental)
        return dict(motion_rows=len(motion),
                    incremental_rows=0 if incremental is None else len(incremental.get_xyz))

    @torch.no_grad()
    def forward_sparse_motion_export(self, viewpoint: TensorSpace) -> dict:
        """Render the currently loaded sparse-v2 frame through the normal rasterizer."""
        if self._sparse_motion_export is None:
            raise RuntimeError("Load a sparse motion export before rendering it")
        delta_xyz, delta_rotation, incremental = self._sparse_motion_export
        original_model = self.renderer.gaussians
        original_delta_xyz = getattr(self.gaussians, "delta_xyz", None)
        original_delta_rotation = getattr(self.gaussians, "delta_rot", None)
        try:
            self.gaussians.delta_xyz = delta_xyz
            self.gaussians.delta_rot = delta_rotation
            if incremental is not None:
                self.renderer.gaussians = MergedGaussianModel([self.gaussians, incremental])
            else:
                self.renderer.gaussians = self.gaussians
            return self.renderer(viewpoint, self.background)
        finally:
            self.gaussians.delta_xyz = original_delta_xyz
            self.gaussians.delta_rot = original_delta_rotation
            self.renderer.gaussians = original_model

    def clear_sparse_motion_export(self):
        """Release the decoded motion field and any loaded incremental points."""
        self._sparse_motion_export = None

    @torch.no_grad()
    def advance_sparse_motion_export(self):
        """Commit the loaded frame and move base state to the next frame.

        This mirrors the legacy training transition: apply motion to the old
        base, merge this frame's births, then prune the same number of lowest
        opacity points before the next frame's motion is decoded.
        """
        if self._sparse_motion_export is None:
            raise RuntimeError("Load and render a sparse motion export before advancing")
        if not self.config.merge_to_base:
            raise RuntimeError("Sparse sequential playback currently requires merge_to_base")
        if any(getattr(self.gaussians, name, None) is not None for name in
               ("delta_xyz", "delta_rot", "rigid_xyz", "rigid_rotation", "noise_scale")):
            raise RuntimeError("Clear temporary Gaussian motion overrides before advancing")
        delta_xyz, delta_rotation, incremental = self._sparse_motion_export
        self.gaussians.apply_deformation(delta_xyz, delta_rotation)
        incremental_rows = 0 if incremental is None else len(incremental.get_xyz)
        if incremental_rows:
            self.gaussians.extend(incremental)
            lowest_opacity = self.gaussians.get_opacity.flatten().sort().indices
            self.gaussians.prune_points(lowest_opacity[:incremental_rows])
        self.renderer.gaussians = self.gaussians
        self.clear_sparse_motion_export()
        return dict(base_rows=len(self.gaussians.get_xyz),
                    merged_incremental_rows=incremental_rows)

    def forward(self, viewpoint: TensorSpace, training: bool = False, **kwargs) -> dict:
        bg_color = self.background
        if training and self.config.random_background:
            bg_color = torch.rand(3, device=bg_color.device)

        results = {}
        if not self.trainer.is_first_frame:
            if self.config.motion_mode in QUANTIZED_MOTION_MODES:
                return self.forward_quantized_residual(viewpoint, training=training, **kwargs)
            if self.config.motion_mode in {'selective', 'learned_selective'} and not self.selective.committed:
                stage, owner = self.selective.stage(self.global_step)
                if self.config.motion_mode == 'learned_selective':
                    xyz, rotation = self.selective.transform(step=self.global_step, training=training, update_diagnostics=training)
                    owner = self.selective.owner.clone()
                else:
                    xyz, rotation = self.selective.transform(owner)
                self.gaussians.rigid_xyz = xyz
                self.gaussians.rigid_rotation = rotation
                results['selective_owner'] = owner
                results['selective_stage'] = stage
            num_stage1_steps = self.deform.config.num_stage1_steps
            enable_deform = False
            if self.global_step < num_stage1_steps:
                enable_deform = True
            elif self.global_step == num_stage1_steps and training:
                enable_deform = True

            if enable_deform and self.config.motion_mode == 'legacy':
                delta_xyz, delta_rot = self.deform(self.gaussians.get_xyz_ori)
                if training and delta_xyz.requires_grad and self.deform.collecting_gradients(self.global_step):
                    delta_xyz.retain_grad()
                results["delta_xyz"] = delta_xyz
                results["delta_rotation"] = delta_rot
                self.gaussians.delta_xyz = delta_xyz
                self.gaussians.delta_rot = delta_rot
        elif training:
            self.gaussians.noise_scale = self.config.noise_scale

        results.update(self.renderer(viewpoint, bg_color, **kwargs))
        for attr in ["delta_xyz", "delta_rot", "noise_scale", 'rigid_xyz', 'rigid_rotation']:
            setattr(self.gaussians, attr, None)
        return results

    def pre_training_step(self):
        interact_with_gui(
            self.global_step,
            self.renderer.config,  # can be modified
            self,
            self.datamodule.config.root,
            self.num_steps,
        )

        if self.trainer.is_first_step:
            self.timer.clock("frame").start()

        if (not self.trainer.is_first_frame and
                self.config.motion_mode == 'sparse_hierarchical' and
                self.global_step == (self.config.sparse_hierarchical.tree_probe_steps +
                                     self.config.sparse_hierarchical.dense_fit_steps +
                                     self.config.sparse_hierarchical.gate_compress_steps + 1)):
            # Finish the training phase transition before its read-only probe.
            self.residual_motion._configure_phase(self.global_step)
            self.quantized_quality_snapshot('after_gate_hardening')

        if not self.trainer.is_first_frame and self.trainer.is_first_step:
            self.timer.clock("frame setup").start()
            num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
            if self.config.motion_mode in QUANTIZED_MOTION_MODES:
                self.prepare_quantized_frame()
                self.timer.clock('frame setup').stop()
                return
            affected_old = {}
            if self.config.motion_mode == 'learned_selective':
                reference_started = time.perf_counter()
                self.prepare_change_reference()
                reference_seconds = time.perf_counter() - reference_started
            if self.config.merge_to_base and num_incr_gs > 0:
                self.trainer.routine_info(f"Merge and prune {num_incr_gs} gaussians")
                self.gaussians.extend(self.gaussians_incr)
                indices = self.gaussians.get_opacity.flatten().sort().indices
                if self.config.motion_mode in {'selective', 'learned_selective'} and self.selective.members:
                    old_count = len(self.selective.owner)
                    deleted_old = indices[:num_incr_gs]
                    deleted_old = deleted_old[deleted_old < old_count]
                    previous_owner = self.selective.owner[deleted_old]
                    affected_old = dict(pruned_old=int(len(deleted_old)),
                                        pruned_keep=int((self.selective.states[previous_owner] == 0).sum()))
                    if self.config.motion_mode == 'selective':
                        affected_old.update(pruned_patch_static=int((self.selective.evidence[previous_owner] == 1).sum()),
                                            pruned_unknown=int((self.selective.evidence[previous_owner] == 0).sum()))
                    else:
                        affected_old['pruned_unsupported'] = int((self.selective.observation_count[previous_owner] == 0).sum())
                self.gaussians.prune_points(indices[:num_incr_gs])

            if self.config.motion_mode in {'selective', 'learned_selective'}:
                self.selective.setup(self.gaussians.get_xyz_ori.detach(), self.gaussians._rotation.detach())
                if self.config.merge_to_base and num_incr_gs > 0:
                    self.selective.topology_events.append(dict(event='merge_prune', merged=num_incr_gs,
                                                               pruned=num_incr_gs, **affected_old))
                self.gaussians_incr = GaussianModel(self.config.sh_degree)
                self.renderer.gaussians = self.gaussians
                evidence_started = time.perf_counter()
                if self.config.motion_mode == 'selective':
                    self.selective.gather_evidence(self.selective_pairs(), self.render_selective_reference)
                elif not self.motion_config.fixed_leaf_ablation:
                    # Geometric renderer support only; never image-change votes.
                    with torch.no_grad():
                        for view in self.datamodule.train_dataloader(random=False):
                            self.selective.observe(self.renderer(view.to(self.device), self.background)['visibility_filter'])
                self.selective.stage_seconds['observation_setup'] = time.perf_counter() - evidence_started
                if self.config.motion_mode == 'learned_selective':
                    self.selective.stage_seconds['change_reference_setup'] = reference_seconds
                self.timer.clock('frame setup').stop()
                return

            reset_grid = self.current_frame == 1
            if self.current_frame == 1:
                self.deform.setup(self.gaussians.get_xyz.detach(), reset_grid)
            elif self.current_frame % self.deform.config.grid_reset_interval == 0 and self.config.re_hierarchization:
                self.deform.reset_grid(self.gaussians.get_xyz.detach())
            else:
                self.deform.reset(self.gaussians.get_xyz.detach())
            self.deform.to(self.device)
            self.gaussians_incr = GaussianModel(self.config.sh_degree)
            logging.info("Reset gaussians of renderer")
            self.renderer.gaussians = self.gaussians
            self.timer.clock("frame setup").stop()

        if self.config.motion_mode in QUANTIZED_MOTION_MODES and not self.trainer.is_first_frame:
            return
        if self.config.motion_mode == 'learned_selective' and not self.trainer.is_first_frame:
            if self.global_step > self.motion_config.warmup_steps + self.motion_config.joint_steps and not self.selective.selected:
                self.selective.select()
        if self.config.motion_mode == 'selective' and not self.trainer.is_first_frame:
            selection_step = (self.motion_config.probe_steps +
                              self.motion_config.max_depth *
                              self.motion_config.shared_fit_steps_per_depth + 1)
            if self.global_step >= selection_step and not self.selective.selected:
                self.timer.clock('selective comparison').start()
                comparison_started = time.perf_counter()
                pairs = self.selective_pairs()
                self.selective.select(lambda node, candidates: self.selective_candidate_losses(node, candidates, pairs))
                self.selective.stage_seconds['comparison'] = time.perf_counter() - comparison_started
                self.timer.clock('selective comparison').stop()
                self.save_selective_diagnostics('selection')

        num_stage2_steps = self.global_step - self.deform.config.num_stage1_steps
        if not self.trainer.is_first_frame and num_stage2_steps == 1:
            self.gaussians_incr.training_setup(self.config.gaussian_stage2)
            self.renderer.gaussians = MergedGaussianModel(
                [self.gaussians, self.gaussians_incr]
            )

    def training_step(
        self, viewpoint: TensorSpace, current_step: int
    ) -> Tuple[Tensor, dict, dict]:
        self.timer.clock("training step").start()
        selective_started = time.perf_counter()

        num_stage1_steps = self.deform.config.num_stage1_steps
        if self.trainer.is_first_frame:
            self.gaussians.update_learning_rate(current_step)
        elif (current_stage2_step := current_step - num_stage1_steps) > 0:
            self.gaussians_incr.update_learning_rate(current_stage2_step)
            if self.config.motion_mode in QUANTIZED_MOTION_MODES and current_stage2_step == 1:
                self.quant_birth_control['stage2_optimizer_lrs'] = {
                    group['name']: float(group['lr']) for group in self.gaussians_incr.optimizer.param_groups}

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if current_step % 1000 == 0 and self.trainer.is_first_frame:
            self.gaussians.oneupSHdegree()

        # Render
        if (current_step - 1) == self.renderer.config.debug_from:
            self.renderer.config.debug = True

        render_pkg = self.forward(viewpoint, training=True)
        image = render_pkg["render"]
        gt_image = viewpoint.image
        loss_rgb = l1_loss(image, gt_image)
        loss_dssim = 1.0 - ssim(image, gt_image)

        lambda_dssim = self.config.lambda_dssim
        loss = (1.0 - lambda_dssim) * loss_rgb + lambda_dssim * loss_dssim

        metrics = {
            "loss/image": loss,
            "loss/rgb": loss_rgb,
            "loss/dssim": loss_dssim,
            "psnr": psnr(image, gt_image).mean().double(),
            "gs": (len(render_pkg["radii"]) if self.config.motion_mode in QUANTIZED_MOTION_MODES
                   else self.renderer.gaussians.get_xyz.shape[0]),
        }

        if "delta_xyz" in render_pkg and self.config.lambda_deform > 0:
            loss_reg = self.deform.reg_loss()
            loss = loss + self.config.lambda_deform * loss_reg
            metrics["loss/reg"] = loss_reg

        if 'selective_owner' in render_pkg:
            render_pkg['selective_started'] = selective_started
            if self.config.motion_mode == 'learned_selective':
                self.selective.observe(render_pkg['visibility_filter'])
                if self.motion_config.fixed_leaf_ablation:
                    loss_reg, structure_metrics = loss_rgb.new_zeros(()), {}
                else:
                    loss_reg, structure_metrics = self.selective.regularization(current_step)
                metrics.update({'learned/' + key: value for key, value in structure_metrics.items()})
                if self.motion_config.enable_change_loss:
                    reference, old_gt = self.change_reference[viewpoint.path.name]
                    loss_change = ((image - reference.to(image)) - (gt_image - old_gt.to(image))).abs().mean()
                    loss = loss + self.motion_config.change_loss_weight * loss_change
                    metrics['loss/change_matching'] = loss_change
            else:
                loss_reg = self.selective.regularization(render_pkg['selective_owner'])
            loss = loss + loss_reg
            metrics['loss/selective_reg'] = loss_reg
            if self.config.motion_mode == 'learned_selective':
                self.selective.last_metrics.update({key: float(value.detach()) for key, value in metrics.items() if torch.is_tensor(value)})
        elif self.config.motion_mode in {'selective', 'learned_selective'} and not self.trainer.is_first_frame:
            render_pkg['selective_started'] = selective_started
            if self.config.motion_mode == 'learned_selective':
                self.selective.last_metrics['incremental_loss_rgb'] = float(loss_rgb.detach())
                self.selective.last_metrics['incremental_loss_dssim'] = float(loss_dssim.detach())
                self.selective.last_metrics['incremental_step'] = current_step

        if self.config.motion_mode in QUANTIZED_MOTION_MODES and not self.trainer.is_first_frame:
            render_pkg['quantized_started'] = selective_started
            cfg = self.config.quantized_residual
            storage_active = (self.config.motion_mode != 'sparse_hierarchical' or
                              self.residual_motion.stage(current_step) in
                              {'gate_compress', 'incremental'})
            storage_weight = (cfg.storage_loss_weight * self.residual_motion.storage_scale(current_step)
                              if storage_active else 0.)
            if self.config.motion_mode == 'sparse_hierarchical' and storage_weight <= 0:
                # Disabled/frozen sparse phases build no storage backward graph.
                storage = loss_rgb.new_zeros(())
                storage_details = {}
            else:
                # Preserve the existing storage path for other motion modes.
                with torch.set_grad_enabled(torch.is_grad_enabled() and storage_weight > 0):
                    storage, storage_details = self.quantized_storage(render_pkg.get('quantized_new_view'))
            reference_bytes = self.residual_motion.reference_storage_bytes(self.quant_first_frame_gs_count)
            loss_storage = loss_rgb.new_zeros(())
            if storage_weight > 0:
                loss_storage = storage_weight * storage / reference_bytes
                loss = loss + loss_storage
            loss_activity = loss_storage.new_zeros(())
            activity_active = (self.config.motion_mode != 'sparse_hierarchical' or
                               self.residual_motion.stage(current_step) == 'gate_compress')
            if cfg.gs_activity_loss_weight > 0 and activity_active:
                if self.residual_motion.committed:
                    # Gates/motion are fixed during newborn fitting; reuse their final cost.
                    cached = self.residual_motion.last_metrics.get('gs_activity/committed_fraction')
                    if cached is None:
                        expected_activity = self.residual_motion.gs_activity_fraction(current_step)
                        self.residual_motion.last_metrics['gs_activity/committed_fraction'] = float(
                            expected_activity.detach())
                    else:
                        expected_activity = loss_storage.new_tensor(cached)
                else:
                    expected_activity = self.residual_motion.gs_activity_fraction(current_step)
                loss_activity = (cfg.gs_activity_loss_weight *
                                 self.residual_motion.storage_scale(current_step) * expected_activity)
                loss = loss + loss_activity
                metrics['gs_activity/expected_fraction'] = expected_activity
            metrics['loss/gs_activity'] = loss_activity
            metrics.update({'storage/' + name: value for name, value in storage_details.items()})
            metrics['storage/expected_total_bytes'] = storage
            metrics['storage/reference_bytes'] = loss_rgb.new_tensor(reference_bytes)
            metrics['loss/storage'] = loss_storage
            loss_candidate = loss_storage.new_zeros(())
            if (self.config.motion_mode != 'sparse_hierarchical' and
                    not self.residual_motion.committed and cfg.keep_candidate_fit_due(current_step)):
                candidate = self.keep_candidate_image_loss(viewpoint)
                if candidate is not None:
                    loss_candidate = cfg.keep_candidate_fit_weight * candidate
                    loss = loss + loss_candidate
            metrics['loss/keep_candidate'] = loss_candidate
            if self.residual_motion.committed and cfg.enable_incremental:
                # An empty newborn pool must still allow the stage's backward call.
                loss = loss + self.gaussians_incr._xyz.sum() * 0
            metrics['loss/total'] = loss
            self.residual_motion.last_metrics.update({name: float(value.detach()) for name, value in metrics.items() if torch.is_tensor(value)})

        self.timer.clock("training step").stop()

        self.timer.clock("tensorboard logging").start()
        self.log_dict(metrics)
        self.timer.clock("tensorboard logging").stop()

        return loss, metrics, render_pkg

    def keep_candidate_image_loss(self, viewpoint):
        """One extra actual candidate render, with no gate/active-motion gradient."""
        pose = self.residual_motion.keep_share_candidate_pose()
        if pose is None:
            return None
        xyz, rotation, nodes = pose
        original = self.renderer.gaussians
        original_xyz = getattr(self.gaussians, 'rigid_xyz', None)
        original_rotation = getattr(self.gaussians, 'rigid_rotation', None)
        self.trainer.synchronize()
        started = time.perf_counter()
        try:
            self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = xyz, rotation
            self.renderer.gaussians = self.gaussians
            image = self.renderer(viewpoint, self.background)['render']
            weight = self.config.lambda_dssim
            loss = ((1 - weight) * l1_loss(image, viewpoint.image) +
                    weight * (1 - ssim(image, viewpoint.image)))
        finally:
            self.gaussians.rigid_xyz = original_xyz
            self.gaussians.rigid_rotation = original_rotation
            self.renderer.gaussians = original
        self.trainer.synchronize()
        stats = self.residual_motion.last_metrics
        stats['candidate_fit/extra_renders'] = stats.get('candidate_fit/extra_renders', 0) + 1
        stats['candidate_fit/forward_loss_seconds'] = stats.get('candidate_fit/forward_loss_seconds', 0.) + time.perf_counter() - started
        stats['candidate_fit/last_nodes'] = nodes
        stats['candidate_fit/last_image_loss'] = float(loss.detach())
        stats['candidate_fit/image_loss_sum'] = stats.get('candidate_fit/image_loss_sum', 0.) + float(loss.detach())
        return loss

    def selective_pairs(self):
        cfg = self.motion_config
        return self.datamodule.selective_training_pairs(cfg.comparison_view_names, self.device,
                                                       cfg.camera_match_atol)

    def render_selective_reference(self, viewpoint):
        return self.renderer(viewpoint, self.background)

    @torch.no_grad()
    def selective_render_loss(self, owner, keep, pairs):
        xyz, rotation = self.selective.transform(owner, keep)
        self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = xyz, rotation
        try:
            total = 0
            for viewpoint, _ in pairs:
                image = self.renderer(viewpoint, self.background)['render']
                weight = self.config.lambda_dssim
                total += ((1 - weight) * l1_loss(image, viewpoint.image) +
                          weight * (1 - ssim(image, viewpoint.image))).item()
            return total / len(pairs)
        finally:
            self.gaussians.rigid_xyz = self.gaussians.rigid_rotation = None

    @torch.no_grad()
    def selective_candidate_losses(self, node, candidates, pairs):
        """One common ROI covers old centers and every candidate's moved centers."""
        indices = self.selective.members[node]
        transformed = [self.selective.transform(owner, keep) for owner, keep in candidates]
        positions = [self.selective.base_xyz[indices]] + [xyz[indices] for xyz, _ in transformed]
        totals = [0.0 for _ in candidates]
        padding = self.motion_config.roi_padding_pixels
        radius = self.selective.reference_radii[indices] + padding
        for viewpoint, _ in pairs:
            height, width = viewpoint.image.shape[-2:]
            lower, upper = [], []
            for points in positions:
                homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], -1)
                clip = homogeneous @ viewpoint.full_proj_transform
                valid = clip[:, 3] > 0
                if valid.any():
                    pixel = ((clip[valid, :2] / clip[valid, 3:4] + 1) *
                             points.new_tensor([width, height]) / 2 - .5)
                    lower.append((pixel - radius[valid, None]).min(0).values)
                    upper.append((pixel + radius[valid, None]).max(0).values)
            if lower:
                lo = torch.stack(lower).min(0).values.floor().long()
                hi = torch.stack(upper).max(0).values.ceil().long() + 1
                x0, y0 = int(lo[0].clamp(0, width - 1)), int(lo[1].clamp(0, height - 1))
                x1, y1 = int(hi[0].clamp(x0 + 1, width)), int(hi[1].clamp(y0 + 1, height))
            else:
                x0, y0, x1, y1 = 0, 0, width, height
            target = viewpoint.image[:, y0:y1, x0:x1]
            for index, (xyz, rotation) in enumerate(transformed):
                self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = xyz, rotation
                try:
                    image = self.renderer(viewpoint, self.background)['render'][:, y0:y1, x0:x1]
                    weight = self.config.lambda_dssim
                    totals[index] += ((1 - weight) * l1_loss(image, target) +
                                      weight * (1 - ssim(image, target))).item()
                finally:
                    self.gaussians.rigid_xyz = self.gaussians.rigid_rotation = None
        return [value / len(pairs) for value in totals]

    @torch.no_grad()
    def review_selective_refinement(self):
        pairs = self.selective_pairs()
        keep = self.selective.states[self.selective.owner] == 0
        refined = self.selective_render_loss(self.selective.owner, keep, pairs)
        translation = self.selective.translation.detach().clone()
        rotation = self.selective.rotation_vector.detach().clone()
        self.selective.translation.copy_(self.selective.accepted_translation)
        self.selective.rotation_vector.copy_(self.selective.accepted_rotation_vector)
        accepted = self.selective_render_loss(self.selective.owner, keep, pairs)
        rollback = refined > accepted + self.motion_config.final_loss_tolerance
        if not rollback:
            self.selective.translation.copy_(translation)
            self.selective.rotation_vector.copy_(rotation)
        old_loss = self.selective_render_loss(self.selective.owner, torch.ones_like(keep), pairs)
        chosen_loss = accepted if rollback else refined
        old_fallback = old_loss + self.motion_config.final_loss_tolerance < chosen_loss
        if old_fallback:
            self.selective.states[self.selective.owner.unique()] = 0
        self.selective.final_review = dict(refined_loss=refined, accepted_loss=accepted,
                                           old_loss=old_loss, rolled_back=rollback,
                                           all_keep_fallback=old_fallback)

    @torch.no_grad()
    def post_training_step_init(self, render_pkg):
        if self.global_step < self.config.densify.until_step:
            # Densification, depend on grad
            self.densify_gaussians(
                self.global_step,
                render_pkg["visibility_filter"],
                render_pkg["viewspace_points"],
                render_pkg["radii"],
            )

        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def post_training_step_incr(
        self, render_pkg, gate_progress: Optional[Callable[[str], None]] = None
    ):
        num_stage1_steps = self.deform.config.num_stage1_steps
        if self.config.motion_mode in QUANTIZED_MOTION_MODES:
            self.post_quantized_residual(render_pkg)
            return
        if self.config.motion_mode in {'selective', 'learned_selective'}:
            if self.config.motion_mode == 'learned_selective':
                self.post_learned_selective_motion(render_pkg)
            else:
                self.post_selective_motion(render_pkg)
            return

        # Collect output gradients after backward, update motion, then freeze the
        # parent split decision for subsequent steps.
        if self.global_step <= num_stage1_steps and render_pkg.get("delta_xyz") is not None:
            if self.deform.collecting_gradients(self.global_step):
                self.deform.accumulate_gradient_score(render_pkg["delta_xyz"].grad,
                                                     render_pkg["visibility_filter"], step=self.global_step)
            self.deform.optimizer.step()
            nested_split = (self.deform.config.hierarchy_mode == "nested" and
                            self.deform.config.num_grid_levels == 3 and
                            self.global_step == self.deform.config.split_step)
            gated_split = (isinstance(self.deform, GatedFourDeformation) and
                           self.deform.gate_due(self.global_step))
            legacy_split = (self.deform.config.hierarchy_mode == "legacy" and
                            self.global_step == self.deform.config.mask_build_step)
            if nested_split or legacy_split:
                if nested_split:
                    before_xyz, before_rot = self.deform(self.gaussians.get_xyz_ori)
                hierarchy_stats = (self.deform.activate_refinement() if nested_split
                                   else self.deform.select_random_middle_regions())
                if nested_split:
                    after_xyz, after_rot = self.deform(self.gaussians.get_xyz_ori)
                    xyz_change = (after_xyz - before_xyz).abs().max().item()
                    rot_change = (after_rot - before_rot).abs().max().item()
                    hierarchy_stats.update(activation_delta_xyz_max=xyz_change,
                                           activation_delta_rotation_max=rot_change)
                    logging.info("Fine activation continuity: max xyz change %.6g, max rotation change %.6g",
                                 xyz_change, rot_change)
                    if xyz_change > 1e-7 or rot_change > 1e-7:
                        raise RuntimeError("Activating untrained Fine changed the deformation output")
                stats_path = (
                    self.output_dir
                    / "hierarchy_stats"
                    / f"frame_{self.current_frame:03d}_hierarchy_stats.pt"
                )
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                cpu_stats = {key: value.detach().cpu() if torch.is_tensor(value) else value
                             for key, value in hierarchy_stats.items()}
                torch.save(cpu_stats, stats_path)
                if nested_split:
                    columns = dict(gs_count='count', child_count='child_count',
                                   observations='score_observations', visible_gs_sum='score_visible_sum',
                                   H_before='H_before', H_after='H_after', H_reduction='H_reduction',
                                   diagnostic_valid='diagnostic_valid', eligible='eligible', active='mid_active')
                    values = {label: cpu_stats[key].tolist() for label, key in columns.items()}
                    csv_path = stats_path.with_suffix('.csv')
                    with csv_path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=['middle_id', 'x', 'y', 'z', *columns])
                        writer.writeheader()
                        for middle_id, xyz in enumerate(cpu_stats['middle_xyz'].tolist()):
                            row = {label: array[middle_id] for label, array in values.items()}
                            if not row['diagnostic_valid']:
                                for label in ('H_before', 'H_after', 'H_reduction'):
                                    row[label] = ''
                            writer.writerow(dict(middle_id=middle_id, x=xyz[0], y=xyz[1], z=xyz[2], **row))
                    logging.info("Saved readable gradient diagnostics to %s", csv_path)
                logging.info(f"Saved hierarchy statistics to {stats_path}")
            if gated_split:
                before_xyz, before_rot = self.deform(self.gaussians.get_xyz_ori)
                if gate_progress is not None:
                    gate_progress("baseline")
                hierarchy_stats = self.deform.activate_refinement(self.global_step)
                if gate_progress is not None:
                    gate_progress("selection")
                after_xyz, after_rot = self.deform(self.gaussians.get_xyz_ori)
                xyz_change = (after_xyz - before_xyz).abs().max().item()
                rot_change = (after_rot - before_rot).abs().max().item()
                hierarchy_stats.update(activation_delta_xyz_max=xyz_change,
                                       activation_delta_rotation_max=rot_change)
                if xyz_change > 1e-7 or rot_change > 1e-7:
                    raise RuntimeError("Opening a four-level gate changed the deformation output")
                if gate_progress is not None:
                    gate_progress("continuity")
                level = hierarchy_stats["opened_level"]
                stats_path = (self.output_dir / "hierarchy_stats" /
                              f"frame_{self.current_frame:03d}_gate_l{level}_stats.pt")
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                cpu_stats = {key: value.detach().cpu() if torch.is_tensor(value) else value
                             for key, value in hierarchy_stats.items()}
                torch.save(cpu_stats, stats_path)
                with stats_path.with_suffix(".csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["parent_id", "x", "y", "z",
                                                                "count", "child_count", "observed",
                                                                "H_before", "H_after", "eligible", "active"])
                    writer.writeheader()
                    for parent_id, xyz in enumerate(cpu_stats["parent_xyz"].tolist()):
                        writer.writerow(dict(parent_id=parent_id, x=xyz[0], y=xyz[1], z=xyz[2],
                                             count=int(cpu_stats["count"][parent_id]),
                                             child_count=int(cpu_stats["child_count"][parent_id]),
                                             observed=int(cpu_stats["observed"][parent_id]),
                                             H_before=float(cpu_stats["H_before"][parent_id]),
                                             H_after=float(cpu_stats["H_after"][parent_id]),
                                             eligible=bool(cpu_stats["eligible"][parent_id]),
                                             active=bool(cpu_stats["active"][parent_id])))
                if gate_progress is not None:
                    gate_progress("statistics")
                logging.info("Opened L%d gate: %d/%d parents; saved %s", level,
                             hierarchy_stats["selected"], hierarchy_stats["budget"], stats_path)
            self.deform.optimizer.zero_grad(set_to_none=True)

        if self.global_step < self.config.densify.until_step:
            self.adaptive_densify_gaussians(
                render_pkg["visibility_filter"],
                render_pkg["viewspace_points"],
                render_pkg["radii"],
            )

        if self.global_step > num_stage1_steps:
            self.gaussians_incr.optimizer.step()
            self.gaussians_incr.optimizer.zero_grad(set_to_none=True)

    def forward_quantized_residual(self, viewpoint, training=False, **kwargs):
        motion = self.residual_motion
        results = {}
        original_renderer_model = self.renderer.gaussians
        try:
            if not motion.committed:
                xyz, rotation = motion(self.global_step, training=training)
                self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = xyz, rotation
                self.renderer.gaussians = self.gaussians
                results['quantized_stage'] = motion.stage(self.global_step)
                if training and results['quantized_stage'] == 'tree_probe':
                    results['tree_probe_view_name'] = viewpoint.path.name
            elif (not training and self.global_step == self.config.quantized_residual.motion_steps
                  and not self.quant_finalized):
                # Clones already exist at this boundary, but this evaluation measures old-GS motion only.
                self.renderer.gaussians = self.gaussians
                results['quantized_stage'] = motion.stage(self.global_step)
            elif not self.quant_finalized and self.config.quantized_residual.enable_incremental:
                view = self.make_quantized_new_view(training=training)
                self.renderer.gaussians = MergedGaussianModel([self.gaussians, view])
                results['quantized_new_view'] = view
                results['quantized_stage'] = 'incremental'
            else:
                self.renderer.gaussians = self.gaussians
            results.update(self.renderer(viewpoint, self.background, **kwargs))
        finally:
            self.gaussians.rigid_xyz = self.gaussians.rigid_rotation = None
            self.renderer.gaussians = original_renderer_model
        return results

    def make_quantized_new_view(self, training=False):
        from scipy.spatial import cKDTree
        started = time.perf_counter()
        if self.quant_predictor_tree is None:
            self.quant_predictor_tree = cKDTree(self.gaussians.get_xyz_ori.detach().cpu().numpy())
        reference = {name: getattr(self.gaussians, attr).detach() for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}
        view = QuantizedNewGaussianView(self.gaussians_incr, self.config.quantized_residual,
            self.datamodule.cameras_extent, reference, self.quant_predictor_tree)
        key = 'birth_prediction_train_cpu' if training else 'birth_prediction_eval_cpu'
        self.residual_motion.stage_seconds[key] = self.residual_motion.stage_seconds.get(key, 0.) + time.perf_counter() - started
        return view

    @torch.no_grad()
    def freeze_quantized_base(self):
        for attr in RAW_ATTRIBUTES:
            getattr(self.gaussians, attr).requires_grad_(False)

    @torch.no_grad()
    def install_quantized_raw(self, raw):
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self.gaussians, attr, torch.nn.Parameter(raw[name].to(self.device).clone()))
        self.gaussians.training_setup(self.config.gaussian)
        # training_setup resets gradients/counts, but not the per-point radii.
        self.gaussians.max_radii2D = self.gaussians._xyz.new_zeros(self.gaussians._xyz.shape[0])
        self.freeze_quantized_base()
        self.renderer.gaussians = self.gaussians

    @torch.no_grad()
    def deletion_diagnostic_view(self, deleted_count):
        cfg = self.config.quantized_diagnostics
        if (not cfg.deletion_damage_enabled or not deleted_count or
                self.current_frame % cfg.deletion_damage_every_n_frames):
            return None
        # DataLoader draws a worker seed. Diagnostics must not change training RNG.
        with torch.random.fork_rng(devices=[]):
            loader = self.datamodule.test_dataloader()
            if loader is not None:
                for viewpoint in loader:
                    if viewpoint.path.name == cfg.deletion_damage_view_name:
                        return viewpoint.to(self.device)
        raise ValueError(f'Deletion diagnostic test camera not found: {cfg.deletion_damage_view_name}')

    @torch.no_grad()
    def deletion_diagnostic_render(self, viewpoint):
        original = self.renderer.gaussians
        try:
            self.renderer.gaussians = self.gaussians
            image = self.renderer(viewpoint, self.background)['render'].clamp(0., 1.)
        finally:
            self.renderer.gaussians = original
        gt = viewpoint.image
        metrics = dict(psnr=float(psnr(image, gt)), l1=float(l1_loss(image, gt)),
                       ssim=float(ssim(image, gt)))
        return image, metrics

    @torch.no_grad()
    def quantized_quality_snapshot(self, stage, view=None, preserve_motion=False):
        """Read-only metrics on all train/test cameras; never selects a branch."""
        if not self.config.quantized_diagnostics.control_quality_enabled:
            return
        original = self.renderer.gaussians
        original_xyz = getattr(self.gaussians, 'rigid_xyz', None)
        original_rot = getattr(self.gaussians, 'rigid_rotation', None)
        if not preserve_motion:
            mode = getattr(self.config, 'motion_mode', 'legacy')
            motion = getattr(self, 'residual_motion', None)
            can_probe_sparse_motion = (
                mode == 'sparse_hierarchical' and
                stage not in {'before_deletion', 'after_deletion'} and
                motion is not None and getattr(motion, 'index', None) is not None and
                not getattr(motion, 'committed', True))
            if can_probe_sparse_motion:
                self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = self.residual_motion(
                    self.global_step, training=False)
            else:
                self.gaussians.rigid_xyz = self.gaussians.rigid_rotation = None
        result = {}
        try:
            self.renderer.gaussians = (self.gaussians if view is None else
                                      MergedGaussianModel([self.gaussians, view]))
            # Worker seeds must not advance model/optimization random streams.
            with torch.random.fork_rng(devices=[]):
                names = getattr(self.config.quantized_diagnostics, 'control_quality_camera_names', None)
                if names is not None:
                    # Load the small read-only diagnostic subset once per frame.
                    # Explicit indices may include validation cameras; this loader
                    # never supplies an optimization step.
                    cached = getattr(self, 'quant_phase_views', None)
                    if cached is None or cached[0] != self.current_frame:
                        dataset = self.datamodule.get_train_dataset()
                        indices = self.datamodule._camera_indices(dataset, names)
                        views = [v.to(self.device) for v in self.datamodule.train_dataloader(
                            random=False, indices=indices)]
                        self.quant_phase_views = (self.current_frame, views)
                    validation = set(self.datamodule.config.validation_camera_names or [])
                    views = self.quant_phase_views[1]
                    loaders = [('train_probe', [v for v in views if v.path.name not in validation]),
                               ('validation_probe', [v for v in views if v.path.name in validation])]
                else:
                    loaders = [('train', self.datamodule.train_dataloader(random=False)),
                               ('test', self.datamodule.test_dataloader())]
                for split, loader in loaders:
                    if loader is None or (isinstance(loader, list) and not loader):
                        continue
                    rows = []
                    for viewpoint in loader:
                        viewpoint = viewpoint.to(self.device)
                        raw_image = self.renderer(viewpoint, self.background)['render']
                        gt = viewpoint.image
                        image = raw_image.clamp(0., 1.)
                        weight = self.config.lambda_dssim
                        objective = ((1 - weight) * l1_loss(raw_image, gt)
                                     + weight * (1 - ssim(raw_image, gt)))
                        rows.append(dict(camera=viewpoint.path.name, psnr=float(psnr(image, gt)),
                            ssim=float(ssim(image, gt)), l1=float(l1_loss(image, gt)),
                            image_objective=float(objective)))
                    result[split] = dict(views=rows, mean={key: float(np.mean([r[key] for r in rows]))
                        for key in ('psnr', 'ssim', 'l1', 'image_objective')})
        finally:
            self.renderer.gaussians = original
            self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = original_xyz, original_rot
        self.quant_control_diagnostics[stage] = result
        self.save_quantized_control_diagnostics()

    def save_quantized_control_diagnostics(self):
        path = self.output_dir / 'frame_summary' / f'control_{self.current_frame:03d}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = getattr(self.config, 'motion_mode', 'legacy')
        semantics = ('Sparse hierarchical snapshots use direct soft-gated image forwards in dense/compress '
                     'and the fixed binary gates in refine. The m/z diagnostics are independent binary gates; '
                     'they are not KEEP/SHARE/SPLIT states.'
                     if mode == 'sparse_hierarchical' else
                     'Quality snapshots use all train/test cameras. Image objective is the unclamped training '
                     'L1/DSSIM objective without storage; PSNR/SSIM/L1 use clamped images. Gradient summaries '
                     'are combined-loss gradients before optimizer.step at sampled steps; gate changes are '
                     'between sampled steps. No background mask.')
        path.write_text(json.dumps(dict(frame=self.current_frame,
            motion_gradient_mode=self.config.quantized_residual.motion_gradient_mode,
            quality=self.quant_control_diagnostics, gradients=self.quant_gradient_diagnostics,
            semantics=semantics), indent=2))

    @torch.no_grad()
    def record_quantized_gradients(self):
        if self.trainer.is_first_frame or self.residual_motion.committed:
            return
        motion = self.residual_motion
        groups = {f'region_translation_{i}': p for i, p in enumerate(motion.region_translation)}
        groups.update({f'region_rotation_{i}': p for i, p in enumerate(motion.region_rotation)})
        if self.config.motion_mode == 'sparse_hierarchical':
            groups.update(m_gate=motion.m_logits, z_local=motion.z_local_logits)
            if motion.config.m_fine_sharing:
                groups['m_fine_gate'] = motion.m_fine_logits
            groups.update({f'z_region_{i}': p for i, p in enumerate(motion.z_region_logits)})
        else:
            groups.update({f'gate_{i}': p for i, p in enumerate(motion.gate_logits)})
        groups.update(gs_translation=motion.gs_translation, gs_rotation=motion.gs_rotation)
        gradients = {}
        for name, parameter in groups.items():
            grad = parameter.grad
            if grad is None:
                nonzero_rows = None
            elif grad.ndim == 1:
                nonzero_rows = int((grad != 0).sum())
            else:
                nonzero_rows = int((grad.abs().sum(-1) > 0).sum())
            gradients[name] = (None if grad is None else dict(
                l2=float(grad.norm()), max_abs=float(grad.abs().max()),
                nonzero_rows=nonzero_rows, rows=len(grad)))
        if self.config.motion_mode == 'sparse_hierarchical':
            masks = motion._hard_masks_for_step(self.global_step)
            states = [masks['m'].cpu().clone(),
                      *[value.cpu().clone() for value in masks['z_regions']],
                      masks['z_local'].cpu().clone()]
        else:
            states = [s.cpu().clone() for s in motion.gate_state_arrays()['states']]
        previous = getattr(self, 'quant_last_sampled_gates', None)
        changes = ([int((a != b).sum()) for a, b in zip(states, previous)]
                   if previous is not None else None)
        self.quant_last_sampled_gates = states
        self.quant_gradient_diagnostics.append(dict(step=self.global_step,
            gradients=gradients, sampled_gate_changes=changes,
            activity=motion.statistics(), training_metrics=dict(motion.last_metrics)))
        self.save_quantized_control_diagnostics()

    @torch.no_grad()
    def prepare_quantized_frame(self):
        started = time.perf_counter()
        self.quant_control_diagnostics = {}
        self.quant_gradient_diagnostics = []
        self.quant_last_sampled_gates = None
        self.quantized_quality_snapshot('before_deletion')
        self.quant_previous_reference = {name: value.cpu() for name, value in model_raw(self.gaussians).items()}
        self.quant_previous_sha256 = reference_digest(self.quant_previous_reference)
        if self.quant_first_frame_gs_count is None:
            self.quant_first_frame_gs_count = len(self.gaussians.get_xyz)
        count = self.quant_previous_new_count
        if not self.config.quantized_residual.enable_incremental and count:
            raise ValueError('Incremental-disabled residual run cannot delete previous Gaussians')
        if count >= len(self.gaussians.get_xyz):
            raise ValueError('Previous new-point compensation would remove all reference Gaussians')
        # Deletion precedes this frame's gate fitting, so protect the previous
        # frame's final KEEP decisions, in the exact materialized row order.
        if (self.config.quantized_residual.enable_learned_gates and
                self.residual_motion.index is not None):
            deleted, deletion_protection = self.residual_motion.compensation_deletions(
                self.gaussians.get_opacity, count)
        else:
            deleted = torch.argsort(self.gaussians.get_opacity.detach().flatten(), stable=True)[:count]
            deletion_protection = dict(
                enabled=False, protected_keep_old_count=0,
                eligible_count=len(self.gaussians.get_xyz), requested_count=count,
                deleted_count=len(deleted), deleted_keep_count=0,
                semantics='No previous learned gate decisions are available.')
        self.quant_deleted_indices = deleted.to(torch.int32).cpu()
        previous_rows = survivor_rows(len(self.quant_previous_reference['xyz']), deleted)
        if self.quant_point_birth_frame is None:
            # Older checkpoints did not record point origins. Keep that uncertainty explicit.
            first_origin = 0 if self.current_frame == 1 else -1
            self.quant_point_birth_frame = torch.full(
                (len(self.quant_previous_reference['xyz']),), first_origin, dtype=torch.long)
        if len(self.quant_point_birth_frame) != len(self.quant_previous_reference['xyz']):
            raise ValueError('Point origin rows disagree with previous materialized Gaussian order')
        origins = self.quant_point_birth_frame.cpu()
        removed_origins = origins[self.quant_deleted_indices.long()]
        point_origins = dict(
            deleted_first_frame=int((removed_origins == 0).sum()),
            deleted_previous_birth=int((removed_origins == self.current_frame - 1).sum())
                if self.current_frame > 1 else 0,
            deleted_older_birth=int(((removed_origins > 0) &
                                     (removed_origins < self.current_frame - 1)).sum()),
            deleted_untracked=int((removed_origins < 0).sum()))
        diagnostic_started = time.perf_counter()
        viewpoint = self.deletion_diagnostic_view(count)
        before_image, before_metrics = self.deletion_diagnostic_render(viewpoint) if viewpoint is not None else (None, None)
        if count:
            self.gaussians.prune_points(deleted)
        self.quantized_quality_snapshot('after_deletion')
        self.quant_point_birth_frame = origins[previous_rows.cpu()]
        point_origins.update(
            remaining_first_frame=int((self.quant_point_birth_frame == 0).sum()),
            remaining_incremental=int((self.quant_point_birth_frame > 0).sum()),
            remaining_untracked=int((self.quant_point_birth_frame < 0).sum()))
        deletion_damage = dict(measured=False, deleted_count=count)
        if viewpoint is not None:
            after_image, after_metrics = self.deletion_diagnostic_render(viewpoint)
            deletion_damage.update(measured=True, camera=viewpoint.path.name, frame=self.current_frame,
                before=before_metrics, after=after_metrics,
                delta_psnr=after_metrics['psnr'] - before_metrics['psnr'],
                delta_l1=after_metrics['l1'] - before_metrics['l1'],
                delta_ssim=after_metrics['ssim'] - before_metrics['ssim'],
                render_change_mse=float((after_image - before_image).square().mean()),
                semantics='Same current-frame held-out camera, before/after deletion and before motion fitting. Read-only diagnostic.')
        diagnostic_seconds = time.perf_counter() - diagnostic_started
        self.gaussians_incr = GaussianModel(self.config.sh_degree)
        self.renderer.gaussians = self.gaussians
        self.freeze_quantized_base()
        self.trainer.synchronize()
        layout_started = time.perf_counter()
        prior_motion_count = (len(self.residual_motion.base_xyz)
                              if self.residual_motion.index is not None else None)
        if (self.config.quantized_residual.motion_init_decay and prior_motion_count is not None and
                len(self.quant_previous_reference['xyz']) != prior_motion_count + count):
            raise ValueError('Motion warm start previous reference count disagrees with old motion and newborn count')
        tree_kwargs = (dict(frame_index=self.current_frame)
                       if self.config.quantized_residual.enable_learned_gates else {})
        self.residual_motion.setup(self.gaussians.get_xyz_ori, self.gaussians._rotation,
            self.datamodule.cameras_extent, previous_rows=previous_rows,
            previous_new_count=count, **tree_kwargs)
        self.quant_predictor_tree = None
        self.quant_report = None
        self.quant_finalized = False
        self.quant_stats_window_initialized = False
        self.quant_birth_control = self.new_quant_birth_control()
        self.quant_birth_control['point_origins'] = point_origins
        self.quant_birth_control['deletion_damage'] = deletion_damage
        self.quant_birth_control['deletion_protection'] = deletion_protection
        self.quant_birth_control['motion_history'] = dict(
            old_motion_rows=(int((previous_rows < prior_motion_count).sum())
                             if prior_motion_count is not None else 0),
            newborn_rows=(int((previous_rows >= prior_motion_count).sum())
                          if prior_motion_count is not None else 0),
            first_frame_rows=len(previous_rows) if prior_motion_count is None else 0,
            initialization_decay=(self.residual_motion.tree_setup['initialization_decay']
                if self.config.quantized_residual.enable_learned_gates
                else self.config.quantized_residual.motion_init_decay),
            semantics='Disjoint counts of surviving rows by available motion history. '
                      'Old-row mapping does not imply a nonzero inherited motion; '
                      'newborn and first-frame rows have no own previous motion residual.')
        self.trainer.synchronize()
        self.residual_motion.stage_seconds['layout'] = time.perf_counter() - layout_started
        self.residual_motion.stage_seconds['deletion_diagnostics'] = diagnostic_seconds if viewpoint is not None else 0.
        if getattr(getattr(self.config, 'quantized_diagnostics', None),
                   'prediction_quality_enabled', False):
            original_xyz = getattr(self.gaussians, 'rigid_xyz', None)
            original_rotation = getattr(self.gaussians, 'rigid_rotation', None)
            try:
                self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = self.residual_motion(1)
                self.quantized_quality_snapshot('after_initialization', preserve_motion=True)
            finally:
                self.gaussians.rigid_xyz = original_xyz
                self.gaussians.rigid_rotation = original_rotation
        self.trainer.synchronize()
        self.residual_motion.stage_seconds['frame_prepare'] = time.perf_counter() - started

    def quant_stats_window_start(self):
        return self.config.quantized_residual.densification_stats_start

    def new_quant_birth_control(self):
        cfg = self.config.quantized_residual
        return dict(
            enabled=cfg.enable_incremental,
            stats_window_start=self.quant_stats_window_start(), stats_window_end=cfg.motion_steps,
            stats_steps_observed=0, clone_eligible=0, clone_added=0,
            clone_blocked=0, split_added=0, split_blocked=0,
            split_eligible=0, candidate_pool_peak=0, candidate_pruned=0,
            candidate_prune_reasons=dict(low_opacity=0, screen_size=0, world_size=0, multiple=0),
            candidate_prune_events=[],
            final_candidate_count=0, opacity_pruned=0,
            final_retained=0, max_new_gaussians=cfg.max_new_gaussians,
            max_retained_gaussians=cfg.max_retained_gaussians, retention_budget_pruned=0,
            motion_history=None,
            point_origins={},
            budget_touched=False, previous_compensation_deleted=int(self.quant_deleted_indices.numel()),
            split_count_semantics='split_added counts net new Gaussians; split_blocked counts parent candidates blocked by the budget.',
            capacity_reuse='Pruning follows split; freed slots are available at the next densification interval.')

    def quantized_storage(self, view=None, hard=False):
        """Count effective motion, gate, birth and deletion parameter bytes."""
        parts = self.residual_motion.storage_parts(self.global_step, hard=hard)
        zero = next(iter(parts.values())) * 0
        # Raw sparse motion stores ownership directly, so a birth source used
        # only during training is not an inference parameter in this mode.
        birth_bytes_per_row = 0 if self.config.motion_mode == 'sparse_hierarchical' else 4
        for attr in RAW_ATTRIBUTES:
            value = getattr(self.gaussians, attr)
            birth_bytes_per_row += int(np.prod(value.shape[1:])) * value.element_size()
        if view is None:
            birth_bytes = zero
        else:
            count = view.keep.sum().to(zero) if hard else view.keep_st.sum()
            birth_bytes = count * birth_bytes_per_row
        details = dict(parts, new_bytes=birth_bytes,
                       deletion_bytes=zero + self.quant_deleted_indices.numel() * 4)
        return sum(details.values()), details

    @torch.no_grad()
    def post_quantized_residual(self, render_pkg):
        cfg = self.config.quantized_residual
        motion = self.residual_motion
        if not motion.committed:
            if self.config.motion_mode == 'sparse_hierarchical':
                if 'tree_probe_view_name' in render_pkg:
                    motion.observe_tree_probe(render_pkg['visibility_filter'],
                                              render_pkg['tree_probe_view_name'])
            motion.optimizer_step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            birth = self.quant_birth_control
            if cfg.enable_incremental and self.global_step >= birth['stats_window_start']:
                if not self.quant_stats_window_initialized:
                    if cfg.densification_stats_steps is not None:
                        self.gaussians.xyz_gradient_accum.zero_()
                        self.gaussians.denom.zero_()
                    self.quant_stats_window_initialized = True
                self.gaussians.add_densification_stats(render_pkg['viewspace_points'], render_pkg['visibility_filter'])
                birth['stats_steps_observed'] += 1
            stage = render_pkg['quantized_stage']
            self.trainer.synchronize()
            motion.stage_seconds[stage] = motion.stage_seconds.get(stage, 0.) + time.perf_counter() - render_pkg['quantized_started']
            if self.config.motion_mode == 'sparse_hierarchical':
                probe_steps = getattr(cfg, 'tree_probe_steps', 0)
                if probe_steps and self.global_step == probe_steps:
                    self.quantized_quality_snapshot('after_tree_probe')
                elif self.global_step == probe_steps + cfg.dense_fit_steps:
                    self.quantized_quality_snapshot('after_dense_fit')
                elif self.global_step == probe_steps + cfg.dense_fit_steps + cfg.gate_compress_steps:
                    self.quantized_quality_snapshot('after_gate_compress_soft')
            if self.global_step == cfg.motion_steps:
                xyz, rotation = motion(self.global_step)
                self.gaussians.apply_selected_motion(xyz, rotation)
                motion.committed = True
                self.quantized_quality_snapshot('after_motion')
                for parameter in motion.parameters():
                    parameter.requires_grad_(False)
                if cfg.enable_incremental:
                    before = len(xyz)
                    clone = self.gaussians.adaptive_densify_and_clone(self.deform.config.densify_grad_threshold,
                        self.datamodule.cameras_extent, sigma_scale=cfg.clone_sigma_scale,
                        max_new_gaussians=cfg.max_new_gaussians)
                    birth['clone_eligible'] = clone['eligible']
                    birth['clone_added'] = clone['added']
                    birth['clone_blocked'] = clone['blocked']
                    birth['candidate_pool_peak'] = clone['added']
                    self.gaussians_incr.restore(self.gaussians.capture(), self.config.gaussian_stage2)
                    mask = torch.arange(len(self.gaussians.get_xyz), device=self.device) >= before
                    self.gaussians.prune_points(mask)
                    self.gaussians_incr.prune_points(~mask)
                    self.gaussians_incr.initialize_incremental_optimizer(self.config.gaussian_stage2)
                    birth['stage2_optimizer_lrs_after_reset'] = {
                        group['name']: float(group['lr']) for group in self.gaussians_incr.optimizer.param_groups}
                self.freeze_quantized_base()
        else:
            # Base parameters and motion stay fixed; only quantized birth candidates learn.
            num_old = len(self.gaussians.get_xyz)
            visibility = render_pkg['visibility_filter']
            radii = render_pkg['radii']
            stage2_step = self.global_step - cfg.motion_steps
            if len(self.gaussians_incr.get_xyz):
                self.gaussians_incr.max_radii2D[visibility[num_old:]] = torch.maximum(
                    self.gaussians_incr.max_radii2D[visibility[num_old:]], radii[num_old:][visibility[num_old:]])
                self.gaussians_incr.add_densification_stats(render_pkg['viewspace_points'], visibility[num_old:], align='right')
                if stage2_step % self.deform.config.densify_interval == 0:
                    result = self.gaussians_incr.adaptive_densify_and_prune(
                        self.deform.config.densify_grad_threshold,
                        self.deform.config.opacity_threshold, self.datamodule.cameras_extent,
                        cfg.new_max_screen_size, max_points=cfg.max_new_gaussians)
                    birth = self.quant_birth_control
                    birth['split_eligible'] += result['eligible']
                    birth['split_added'] += result['added']
                    birth['split_blocked'] += result['blocked']
                    birth['candidate_pruned'] += result['pruned']
                    for reason, count in result['prune_reasons'].items():
                        birth['candidate_prune_reasons'][reason] += count
                    birth['candidate_prune_events'].append(dict(
                        incremental_step=stage2_step, candidate_count=result['peak_before_prune'],
                        pruned=result['pruned'], remaining=len(self.gaussians_incr.get_xyz),
                        reasons=result['prune_reasons'],
                        semantics='Reason counts overlap; multiple counts points matching at least two reasons.'))
                    birth['candidate_pool_peak'] = max(birth['candidate_pool_peak'], result['peak_before_prune'])
            self.gaussians_incr.optimizer.step()
            self.gaussians_incr.optimizer.zero_grad(set_to_none=True)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            motion.optimizer.zero_grad(set_to_none=True)
            self.trainer.synchronize()
            motion.stage_seconds['incremental'] = motion.stage_seconds.get('incremental', 0.) + time.perf_counter() - render_pkg['quantized_started']
        if self.trainer.is_last_step:
            self.finalize_quantized_frame()

    @torch.no_grad()
    def finalize_quantized_frame(self):
        finalize_started = time.perf_counter()
        cfg = self.config.quantized_residual
        view = self.make_quantized_new_view() if cfg.enable_incremental else None
        self.quantized_quality_snapshot('before_retention', view)
        opacity_eligible = int(view.keep.sum()) if view is not None else 0
        retention = (view.apply_retention_budget(cfg.max_retained_gaussians)
                     if view is not None else dict(eligible=0, pruned=0,
                         limit=cfg.max_retained_gaussians, selection='no_candidates'))
        self.quant_birth_control['retention_budget'] = retention
        self.quant_birth_control['retention_budget_pruned'] = retention['pruned']
        expected_storage, expected_parts = self.quantized_storage(view)
        hard_storage, hard_parts = self.quantized_storage(view, hard=True)
        reference_bytes = self.residual_motion.reference_storage_bytes(self.quant_first_frame_gs_count)
        motion_export = None
        incremental_export = None
        frame_manifest = None
        if self.config.motion_mode == 'sparse_hierarchical':
            storage_dir = self.output_dir / 'frame_storage' / 'sparse_hierarchical'
            motion_path = storage_dir / f'motion_{self.current_frame:03d}.npz'
            motion_export = self.residual_motion.export_motion(motion_path)
        self.quant_birth_control['final_candidate_count'] = (
            len(self.gaussians_incr.get_xyz) if cfg.enable_incremental else 0)

        self.trainer.synchronize()
        state_started = time.perf_counter()
        raw, retained_sources = materialize_quantized_gaussians(model_raw(self.gaussians), view)
        retained_count = len(retained_sources)
        if cfg.enable_learned_gates:
            self.residual_motion.register_materialized_layout(raw['xyz'], retained_sources)
        self.install_quantized_raw(raw)
        if self.config.motion_mode == 'sparse_hierarchical' and retained_count:
            birth_model = GaussianModel(self.config.sh_degree)
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                setattr(birth_model, attr, torch.nn.Parameter(raw[name][-retained_count:].detach().clone()))
            storage_dir.mkdir(parents=True, exist_ok=True)
            incremental_path = storage_dir / f'incremental_{self.current_frame:03d}.ply'
            birth_model.save_ply(incremental_path)
            incremental_export = dict(path=str(incremental_path),
                                      archive_bytes=incremental_path.stat().st_size,
                                      rows=retained_count)
        if self.config.motion_mode == 'sparse_hierarchical':
            manifest_path = storage_dir / f'frame_{self.current_frame:03d}.json'
            manifest = dict(
                layout_version='independent_binary_hierarchical_motion_v1',
                frame=self.current_frame,
                playback_scope=('Single-frame post-deletion motion override plus this frame\'s birth PLY. '
                    'First confirm previous_reference_sha256 against the previous materialized Gaussian state; '
                    'then apply deleted_indices and confirm motion_base_digest against the post-deletion base '
                    'before decoding motion. This manifest does not implement automatic sequential playback.'),
                previous_reference_sha256=self.quant_previous_sha256,
                motion_base_digest=self.residual_motion._base_digest(),
                deleted_indices=self.quant_deleted_indices.detach().cpu().to(torch.int32).tolist(),
                deleted_indices_dtype='int32',
                deleted_indices_payload_bytes=int(self.quant_deleted_indices.numel() * 4),
                motion_export=motion_export,
                incremental_export=incremental_export)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            frame_manifest = dict(path=str(manifest_path), archive_bytes=manifest_path.stat().st_size,
                                  deleted_indices_payload_bytes=manifest['deleted_indices_payload_bytes'])
        self.quantized_quality_snapshot('after_retention')
        self.trainer.synchronize()
        state_update_seconds = time.perf_counter() - state_started
        self.residual_motion.stage_seconds['state_update'] = state_update_seconds

        self.quant_birth_control['final_retained'] = retained_count
        self.quant_birth_control['opacity_pruned'] = (
            self.quant_birth_control['final_candidate_count'] - opacity_eligible)
        limit = cfg.max_new_gaussians
        self.quant_birth_control['budget_touched'] = (limit is not None and
            (self.quant_birth_control['candidate_pool_peak'] >= limit or
             self.quant_birth_control['clone_blocked'] > 0 or
             self.quant_birth_control['split_blocked'] > 0))
        self.gaussians_incr = GaussianModel(self.config.sh_degree)
        self.quant_previous_new_count = retained_count
        self.quant_point_birth_frame = torch.cat((self.quant_point_birth_frame.cpu(),
            torch.full((retained_count,), self.current_frame, dtype=torch.long)))
        if len(self.quant_point_birth_frame) != len(raw['xyz']):
            raise ValueError('Final point origins disagree with quantized Gaussian order')
        self.quant_birth_control['point_origins'].update(
            new_retained=retained_count, final_total=len(raw['xyz']))
        self.quant_finalized = True
        reference_sha256 = reference_digest(raw)
        actual_motion = None
        diagnostics = getattr(self.config, 'quantized_diagnostics', None)
        if getattr(diagnostics, 'actual_motion_enabled', False):
            if not cfg.enable_learned_gates:
                raise ValueError('Actual motion diagnostics require gated motion')
            xyz, rotation = self.residual_motion(cfg.motion_steps)
            terminal = (self.residual_motion._hard_masks_for_step(cfg.motion_steps)['m'].long()
                        if self.config.motion_mode == 'sparse_hierarchical'
                        else self.residual_motion.gate_state_arrays()['gs_terminal'])
            actual_motion = actual_motion_statistics(
                self.residual_motion.base_xyz, self.residual_motion.base_rotation,
                xyz, rotation, self.residual_motion.extent,
                diagnostics.translation_extent_thresholds,
                diagnostics.rotation_radian_thresholds,
                terminal)
            if self.config.motion_mode == 'sparse_hierarchical':
                actual_motion['semantics'] = ('m=0 rows are exact previous poses; m=1 rows use the '
                    'quantized hierarchical candidate. This is binary m reporting, not a KEEP/SHARE/SPLIT cut.')
        sparse_mode = self.config.motion_mode == 'sparse_hierarchical'
        if sparse_mode:
            candidate_fit_report = dict(enabled=False, extra_renders=0,
                semantics='Disabled in sparse_hierarchical v1. Gate compression receives its image loss '
                          'through the same direct soft-gated rendering graph.')
            motion_gradient_report = dict(
                stages=dict(dense_fit='all m/z forced open; optimize motion only',
                            gate_compress='freeze motion; optimize soft m/z with direct rendering and storage',
                            gate_refine='freeze thresholded m/z; optimize selected motion only',
                            incremental='freeze motion/gates; fit retained newborn Gaussians'),
                semantics='The current phase is selected from the explicit training step. No hard-pose '
                          'soft-gate Jacobian proxy or KEEP/SHARE/SPLIT gradient schedule is used.')
            method_semantics = (
                'Independent binary hierarchy. Every region has an independent z residual gate; a child z=1 '
                'can add motion even when its parent z=0. Local z controls each GS residual. Final m=0 returns '
                'the exact base xyz and rotation. Motion is composed coarse to fine around parent-advected pivots, '
                'then the per-GS residual is applied. Storage counts selected float32 motion plus plain masks '
                'and routing arrays. A selected identity/canceled transform still costs a motion row; '
                'the m-open activity objective is recorded separately.' )
        else:
            candidate_fit_report = dict(mode=cfg.motion_gradient_mode,
                interval=cfg.keep_candidate_fit_interval, weight=cfg.keep_candidate_fit_weight,
                extra_renders=int(self.residual_motion.last_metrics.get('candidate_fit/extra_renders', 0)),
                forward_loss_seconds=self.residual_motion.last_metrics.get('candidate_fit/forward_loss_seconds', 0.),
                last_nodes=self.residual_motion.last_metrics.get('candidate_fit/last_nodes', 0),
                image_loss_sum=self.residual_motion.last_metrics.get('candidate_fit/image_loss_sum', 0.),
                semantics='Extra candidate render shares this step training view; only reachable KEEP '
                          'terminal motion fits its own image objective. Gate/active motion detached. '
                          'Forward time excludes auxiliary backward; full training time includes it.')
            motion_gradient_report = dict(
                configured_mode=cfg.motion_gradient_mode,
                all_split_warmup_steps=cfg.motion_gradient_warmup_steps,
                semantics=('Use all candidate motion Jacobians through the warmup step, then the configured mode. '
                           'Hard forward poses and the gate proxy are unchanged; this schedule affects only motion '
                           'parameter gradients during differentiable training forwards. keep_probe fits SHARE '
                           'candidates at reachable KEEP terminals while retaining learned gate priors; '
                           'KEEP and SHARE both stop descent, so unvisited descendants stay closed. '
                           'keep_local_probe additionally detaches ancestor pose/pivots only for a '
                           'descendant KEEP terminal candidate, preserving its own SHARE probe gradient.'))
            method_semantics = ('Learned KEEP/SHARE/SPLIT hierarchy with optional periodic spatial rebuild. '
                'Rebuild inherits same-level KNN motion priors and gates by surviving member overlap; '
                'otherwise region membership persists. KEEP cancels all ancestor motion; '
                'SHARE closes descendants; SPLIT retains shared rigid motion and opens children/local residuals. '
                'Region rotations move centers around parent-advected pivots. Only reachable branches carry motion.'
                if cfg.enable_learned_gates else
                'Fixed nested spatial candidates with dense quantized residual-motion tensors. '
                'Position offsets add; world orientation composes coarse, middle, fine, then GS residual.')

        self.quant_report = dict(
            frame=self.current_frame, storage_only=True,
            candidate_fit=candidate_fit_report,
            state_update_seconds=state_update_seconds,
            expected_storage_bytes=float(expected_storage),
            hard_storage_bytes=float(hard_storage),
            storage_bytes_by_class={name: float(value) for name, value in hard_parts.items()},
            expected_storage_bytes_by_class={name: float(value) for name, value in expected_parts.items()},
            reference_storage_bytes=reference_bytes,
            motion_export=motion_export,
            incremental_export=incremental_export,
            frame_manifest=frame_manifest,
            dense_motion_training_parameter_bytes=(self.residual_motion.dense_training_parameter_bytes()
                if self.config.motion_mode == 'sparse_hierarchical' else None),
            storage_accounting=('sparse_inference_arrays_v1'
                if sparse_mode and cfg.motion_export_format == 'sparse_parameters' else
                'uncompressed_inference_arrays_v1'
                if self.config.motion_mode == 'sparse_hierarchical' else None),
            inference_file_bytes=(sum(item['archive_bytes'] for item in
                (motion_export, incremental_export, frame_manifest) if item is not None)
                if sparse_mode else None),
            inference_file_semantics=('Necessary motion NPZ, retained birth PLY and deletion/frame manifest '
                'file bytes for this dynamic frame. Includes container headers and verification metadata; '
                'excludes the shared first-frame model, images, full-frame debug PLYs, checkpoints and logs.'
                if sparse_mode else None),
            storage_gate_bits=(self.residual_motion.storage_bit_counts(
                self.global_step, hard=True)
                if self.config.motion_mode == 'sparse_hierarchical' and cfg.motion_export_format == 'legacy' else None),
            gs_activity_cost=(dict(
                weight=cfg.gs_activity_loss_weight,
                expected_fraction=float(self.residual_motion.gs_activity_fraction(self.global_step)),
                hard_fraction=float(self.residual_motion.gs_activity_fraction(self.global_step, hard=True)),
                semantics='Old-GS m-open fraction. Selected zero or canceled motion still counts; '
                          'actual pose changes are reported separately. This cost selects m gates only '
                          'during gate_compress; frozen phases report the committed hard mask.')
                if self.config.motion_mode == 'sparse_hierarchical' else dict(
                weight=cfg.gs_activity_loss_weight,
                expected_fraction=float(self.residual_motion.gs_activity_fraction(self.global_step)),
                hard_fraction=float(self.residual_motion.gs_activity_fraction(self.global_step, hard=True)),
                translation_threshold_world=self.residual_motion.extent * cfg.motion_xyz_step_extent / 2,
                rotation_threshold_radians=cfg.motion_rotation_step_rad / 2,
                semantics='Expected changed old-GS fraction over full quantized terminal poses; '
                          'KEEP and canceled candidate poses cost zero. Geometry support is detached; '
                          'only gate probabilities receive this cost gradient. No masks or visibility oracle.')
                if cfg.enable_learned_gates else None),
            storage_semantics=('Sparse necessary inference arrays: six float32 motion values for relevant '
                'z=1 regions and m=1/z=1 local rows; uint32 m-open GS IDs; uint16/uint32 fine owners '
                'and parent tables; bool local z on m-open GSs; bool regional z and float32 pivots '
                'on relevant regions; counts/source row count; retained newborn raw attributes and '
                'int32 deletion IDs. Soft loss charges expected selected rows and relevant paths. '
                'Parameter bytes exclude file headers and verification metadata; inference_file_bytes '
                'reports necessary files separately. Shared first-frame model is counted once separately. '
                'Images, checkpoints, gradients and optimizer state are excluded.'
                if sparse_mode and cfg.motion_export_format == 'sparse_parameters' else
                'Uncompressed necessary inference arrays: six float32 motion values for '
                'each relevant z=1 region and m=1/z=1 local row; full bool m/z masks; int32 fine ownership, '
                'parent tables and counts; float32 regional pivots; retained newborn raw attributes and '
                'int32 deletion identities. Soft loss uses expected selected rows. First-frame GS is counted '
                'once separately. File headers, diagnostic manifests, dense training tensors, gradients and '
                'optimizer state are excluded. Dense motion/gate training parameter bytes are a separate field.'
                if self.config.motion_mode == 'sparse_hierarchical' else
                'Estimated effective parameter bytes: reachable gate states, SHARE/SPLIT region motion, '
                'open GS residuals, retained newborn raw attributes and int32 source, and int32 deletions. '
                'Hierarchy membership is deterministically reconstructed from the previous materialized GS, '
                'ordered surviving rows, configured rebuild interval and stable_layout algorithm; no membership '
                'table bytes are charged. First-frame reference, configuration/algorithm metadata, file headers, '
                'logs and optimizer are excluded.'
                if cfg.enable_learned_gates else
                'Estimated dense raw regional and per-GS motion parameter bytes, plus retained newborn raw attributes, '
                'int32 source and int32 deletions. No nongated sparse positions are assumed; file headers, first-frame '
                'reference, logs and optimizer are excluded.'),
            activities=self.residual_motion.statistics(),
            actual_motion=actual_motion,
            gate_policy=cfg.gate_policy,
            motion_gradient_schedule=motion_gradient_report,
            last_training_metrics=dict(self.residual_motion.last_metrics),
            stage_seconds=None,
            new_count=retained_count,
            deleted_count=int(self.quant_deleted_indices.numel()),
            normalization_first_frame_gs=self.quant_first_frame_gs_count,
            birth_control=self.quant_birth_control,
            tree_setup=(self.residual_motion.tree_setup if cfg.enable_learned_gates else None),
            method_semantics=method_semantics,
            timing_semantics=('frame_prepare includes deletion/layout; layout is a subcomponent. '
                'tree_topology and tree_initialization are inside layout; tree_rebuild is their sum only on rebuild '
                'frames and must not be added again. Topology includes row/pose validation and pivots. '
                'finalize includes quantized state materialization/model installation/report preparation; '
                'state_update is a subcomponent. birth_prediction_train_cpu is included in training; '
                'birth_prediction_eval_cpu is included in evaluation or finalization. '
                'Reported pure training excludes frame_prepare and finalize; rendering/backward/data/storage '
                'and optimizer remain included. Frame summary JSON writing is measured under saving.'),
            reference_sha256=reference_sha256)
        self.trainer.synchronize()
        self.residual_motion.stage_seconds['finalize'] = time.perf_counter() - finalize_started
        self.quant_report['stage_seconds'] = dict(self.residual_motion.stage_seconds)
        save_started = time.perf_counter()
        report_path = self.output_dir / 'frame_summary' / f'quantized_frame_{self.current_frame:03d}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(self.quant_report, indent=2))
        self.trainer.record_io('saving', save_started)

    def save_quantized_image_diagnostics(self, viewpoint, image_path, camera_name):
        if (self.trainer.is_first_frame or not self.quant_finalized or
                not self.config.selective_diagnostics_config):
            return
        import yaml
        from pathlib import Path
        config = yaml.safe_load(Path(self.config.selective_diagnostics_config).read_text())
        if not config.get('enabled', True) or not config.get('online_enabled', True):
            return
        if config.get('online_steps') is not None and self.global_step not in config['online_steps']:
            return
        if config.get('online_frames') is not None and self.current_frame not in config['online_frames']:
            return
        if config.get('online_cameras') is not None and camera_name not in config['online_cameras']:
            return
        self.trainer.synchronize()
        started = time.perf_counter()
        from .model.residual_plot import draw_residual_frame
        draw_residual_frame(self.residual_motion, self.gaussians.get_xyz_ori, viewpoint, image_path, config)
        self.trainer.record_io('drawing', started)

    @torch.no_grad()
    def prepare_change_reference(self):
        """Freeze full previous renderer before this frame's merge/prune; cache RGB only."""
        self.change_reference = {}
        if not self.motion_config.enable_change_loss:
            return
        names = [item.path.name for item in self.datamodule.get_train_dataset().items]
        pairs = self.datamodule.selective_training_pairs(names, 'cpu', self.motion_config.camera_match_atol)
        cache_device = self.device if self.motion_config.change_cache_device == 'cuda' else 'cpu'
        for current, previous in pairs:
            frozen = self.renderer(current.to(self.device), self.background)['render'].detach().to(cache_device)
            self.change_reference[current.path.name] = (frozen, previous.image.detach().to(cache_device))
        # Paired views need not remain on GPU after RGB reference extraction.
        self.datamodule._selective_pairs_cache = None

    @torch.no_grad()
    def post_learned_selective_motion(self, render_pkg):
        cfg = self.motion_config
        if not self.selective.committed:
            self.selective.optimizer_step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            stage = render_pkg['selective_stage']
            self.trainer.synchronize()
            self.selective.stage_seconds[stage] = self.selective.stage_seconds.get(stage, 0.) + time.perf_counter() - render_pkg['selective_started']
            if cfg.enable_incremental:
                self.gaussians.add_densification_stats(render_pkg['viewspace_points'], render_pkg['visibility_filter'])
            if self.global_step == cfg.motion_steps:
                xyz, rotation = self.selective.transform(step=self.global_step, training=False)
                self.gaussians.apply_selected_motion(xyz, rotation)
                self.selective.committed = True
                self.change_reference = {}
                if cfg.enable_incremental:
                    before = len(xyz)
                    self.gaussians.adaptive_densify_and_clone(self.deform.config.densify_grad_threshold,
                        self.datamodule.cameras_extent, sigma_scale=cfg.clone_sigma_scale)
                    added = len(self.gaussians.get_xyz) - before
                    self.gaussians_incr.restore(self.gaussians.capture(), self.config.gaussian_stage2)
                    mask = torch.arange(len(self.gaussians.get_xyz), device=self.device) >= before
                    self.gaussians.prune_points(mask)
                    self.gaussians_incr.prune_points(~mask)
                    self.selective.topology_events.append(dict(event='clone', added=added))
        else:
            before = len(self.gaussians_incr.get_xyz)
            self.adaptive_densify_gaussians(render_pkg['visibility_filter'], render_pkg['viewspace_points'], render_pkg['radii'])
            self.gaussians_incr.optimizer.step()
            self.gaussians_incr.optimizer.zero_grad(set_to_none=True)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            after = len(self.gaussians_incr.get_xyz)
            self.trainer.synchronize()
            self.selective.stage_seconds['incremental'] = self.selective.stage_seconds.get('incremental', 0.) + time.perf_counter() - render_pkg['selective_started']
            if before != after:
                self.selective.topology_events.append(dict(event='incremental_densify_prune', before=before, after=after))
        if self.trainer.is_last_step:
            started = time.perf_counter()
            path = self.output_dir / 'selective_motion' / f'frame_{self.current_frame:03d}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cfg.fixed_leaf_ablation or self.trainer.checkpoint_frame_due():
                torch.save(self.selective.capture(), path)
            nested_io_before = self.trainer.frame_times['saving']
            summary = self.save_selective_diagnostics('frame_end')
            self.trainer.frame_times['saving'] = nested_io_before
            path.with_suffix('.json').write_text(json.dumps(dict(frame=self.current_frame, diagnostics=summary,
                stage_seconds=self.selective.stage_seconds,
                stage_budgets=(dict(fixed_leaf=cfg.motion_steps, incremental=cfg.incremental_steps)
                               if cfg.fixed_leaf_ablation else
                               dict(soft_warmup=cfg.warmup_steps, hard_joint=cfg.joint_steps,
                                    fixed_refine=cfg.refine_steps, incremental=cfg.incremental_steps))), indent=2))
            self.trainer.record_io('saving', started)

    @torch.no_grad()
    def post_selective_motion(self, render_pkg):
        cfg = self.motion_config
        if not self.selective.committed:
            self.selective.optimizer_step(render_pkg['selective_owner'])
            # The base GS optimizer never steps, including during KEEP training.
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            stage = render_pkg['selective_stage']
            self.selective.stage_seconds[stage] = self.selective.stage_seconds.get(stage, 0.0) + (
                time.perf_counter() - render_pkg['selective_started'])
            if cfg.enable_incremental:
                self.gaussians.add_densification_stats(render_pkg['viewspace_points'],
                                                       render_pkg['visibility_filter'])
            if self.global_step == cfg.motion_steps:
                self.timer.clock('selective final review').start()
                review_started = time.perf_counter()
                self.review_selective_refinement()
                self.selective.stage_seconds['final_review'] = time.perf_counter() - review_started
                self.timer.clock('selective final review').stop()
                self.save_selective_diagnostics('final_review')
                xyz, rotation = self.selective.transform()
                self.gaussians.apply_selected_motion(xyz, rotation)
                self.selective.committed = True
                if cfg.enable_incremental:
                    before = len(xyz)
                    gradient = self.gaussians.xyz_gradient_accum / self.gaussians.denom
                    gradient = torch.nan_to_num(gradient, nan=0.0)
                    source = (gradient.norm(dim=-1) >= self.deform.config.densify_grad_threshold) & (
                        self.gaussians.get_scaling.max(dim=-1).values <=
                        self.gaussians.percent_dense * self.datamodule.cameras_extent)
                    source_owner = self.selective.owner[source]
                    source_stats = dict(keep_sources=int((self.selective.states[source_owner] == 0).sum()),
                                        patch_static_sources=int((self.selective.evidence[source_owner] == 1).sum()),
                                        unknown_sources=int((self.selective.evidence[source_owner] == 0).sum()))
                    self.gaussians.adaptive_densify_and_clone(
                        self.deform.config.densify_grad_threshold, self.datamodule.cameras_extent,
                        sigma_scale=cfg.clone_sigma_scale)
                    added = len(self.gaussians.get_xyz) - before
                    self.gaussians_incr.restore(self.gaussians.capture(), self.config.gaussian_stage2)
                    mask = torch.arange(len(self.gaussians.get_xyz), device=self.device) >= before
                    self.gaussians.prune_points(mask)
                    self.gaussians_incr.prune_points(~mask)
                    self.selective.topology_events.append(dict(event='clone', added=added, **source_stats))
        else:
            count_before = len(self.gaussians_incr.get_xyz)
            self.adaptive_densify_gaussians(render_pkg['visibility_filter'],
                                            render_pkg['viewspace_points'], render_pkg['radii'])
            self.gaussians_incr.optimizer.step()
            self.gaussians_incr.optimizer.zero_grad(set_to_none=True)
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            count_after = len(self.gaussians_incr.get_xyz)
            self.selective.stage_seconds['incremental'] = self.selective.stage_seconds.get('incremental', 0.0) + (
                time.perf_counter() - render_pkg['selective_started'])
            if count_before != count_after:
                self.selective.topology_events.append(dict(event='incremental_densify_prune',
                                                           before=count_before, after=count_after))
        if self.trainer.is_last_step:
            self.trainer.synchronize()
            save_started = time.perf_counter()
            path = self.output_dir / 'selective_motion' / f'frame_{self.current_frame:03d}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.selective.capture(), path)
            selected_nodes = self.selective.owner.unique()
            shared_nodes = selected_nodes[(self.selective.states[selected_nodes] == 1) &
                                           (self.selective.evidence[selected_nodes] == 2)]
            stats = dict(frame=self.current_frame, nodes=len(self.selective.members),
                         selected_regions=int(self.selective.owner.unique().numel()),
                         shared_motion_regions=int(len(shared_nodes)), motion_parameters=int(len(shared_nodes) * 6),
                         comparisons=self.selective.comparison_records,
                         unknown=int((self.selective.evidence == 0).sum()),
                         projected_patch_static=int((self.selective.evidence == 1).sum()),
                         final_review=self.selective.final_review,
                         stage_seconds=self.selective.stage_seconds,
                         stage_budgets=dict(probe=cfg.probe_steps,
                                            shared_fit=cfg.max_depth * cfg.shared_fit_steps_per_depth,
                                            refine=cfg.refine_steps, incremental=cfg.incremental_steps),
                         topology_events=self.selective.topology_events)
            nested_io_before = self.trainer.frame_times['saving']
            stats['diagnostics'] = self.save_selective_diagnostics('frame_end')
            self.trainer.frame_times['saving'] = nested_io_before
            path.with_suffix('.json').write_text(json.dumps(stats, indent=2))
            self.trainer.record_io('saving', save_started)

    def save_selective_diagnostics(self, stage):
        import yaml
        from pathlib import Path
        config = yaml.safe_load(Path(self.config.selective_diagnostics_config).read_text())
        if not config.get('enabled', True):
            return None
        self.trainer.synchronize()
        started = time.perf_counter()
        summary = summarize(self.selective.capture())
        summary.update(frame=self.current_frame, stage=stage)
        self.trainer.routine_info("Selective %s frame %s: %s", stage, self.current_frame, concise(summary))
        path = self.output_dir / 'selective_diagnostics' / f'frame_{self.current_frame:03d}_{stage}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
        self.trainer.record_io('saving', started)
        return summary

    @torch.no_grad()
    def post_training_step(
        self, render_pkg: dict,
        gate_progress: Optional[Callable[[str], None]] = None,
    ):
        self.timer.clock("post training step").start()
        if self.trainer.is_first_frame:
            self.post_training_step_init(render_pkg)
        else:
            self.post_training_step_incr(render_pkg, gate_progress=gate_progress)
        self.timer.clock("post training step").stop()

        self.trainer.synchronize()
        save_started = time.perf_counter()
        self.save_point_cloud()
        self.trainer.record_io('saving', save_started)

        if self.trainer.is_last_step:
            self.timer.clock("frame").stop()
            if not self.trainer.is_first_frame:
                num_incr_gs = (self.quant_report['new_count']
                               if self.config.motion_mode in QUANTIZED_MOTION_MODES
                               else self.gaussians_incr.get_xyz.shape[0])
                self.trainer.routine_info(f"Number of incremental gaussians: {num_incr_gs}")
                if self.config.motion_mode == 'legacy' and isinstance(self.deform, GatedFourDeformation):
                    storage_dir = self.output_dir / "frame_storage"
                    storage_dir.mkdir(parents=True, exist_ok=True)
                    motion_path = storage_dir / f"motion_{self.current_frame:03d}.npz"
                    stats = self.deform.export(
                        motion_path, index_encoding=self.config.sparse_export_index_encoding)
                    gs_path = storage_dir / f"incremental_{self.current_frame:03d}.ply"
                    self.gaussians_incr.save_ply(gs_path)
                    stats.update(frame=self.current_frame, incremental_gs=num_incr_gs,
                                 incremental_bytes=gs_path.stat().st_size,
                                 total_bytes=stats["motion_bytes"] + gs_path.stat().st_size)
                    (storage_dir / f"frame_{self.current_frame:03d}.json").write_text(
                        json.dumps(stats, indent=2))
                if (self.config.motion_mode == 'legacy' and self.deform.config.hierarchy_mode == "nested" and
                        self.deform.config.num_grid_levels == 3):
                    storage_dir = self.output_dir / "frame_storage"
                    storage_dir.mkdir(parents=True, exist_ok=True)
                    active = torch.cat([self.deform.fine_active_mask, torch.ones(
                        self.deform.delta.shape[0] - self.deform.fine_active_mask.numel(),
                        device=self.device, dtype=torch.bool)])
                    remap = torch.full_like(active, -1, dtype=torch.long)
                    remap[active] = torch.arange(active.sum(), device=self.device)
                    motion_path = storage_dir / f"motion_{self.current_frame:03d}.npz"
                    # -1 in the fine index means no contribution; includes mapping overhead.
                    exported_delta = self.deform.delta.detach().clone()
                    exported_delta[:self.deform.fine_active_mask.numel(), 3] -= 1
                    fine = self.deform.grids[0]
                    np.savez_compressed(
                        motion_path, representation=np.array("nested_fine_residual_v2"),
                        delta=exported_delta[active].cpu().numpy(),
                        index=remap[self.deform.index[1:]].int().cpu().numpy(),
                        fine_indices=remap[fine.blend_indices].int().cpu().numpy(),
                        fine_weights=fine.blend_weights.cpu().numpy())
                    gs_path = storage_dir / f"incremental_{self.current_frame:03d}.ply"
                    self.gaussians_incr.save_ply(gs_path)
                    stats = dict(frame=self.current_frame, incremental_gs=num_incr_gs,
                                 active_fine=int(self.deform.fine_active_mask.sum()),
                                 candidate_fine=self.deform.fine_active_mask.numel(),
                                 motion_bytes=motion_path.stat().st_size,
                                 incremental_bytes=gs_path.stat().st_size,
                                 total_bytes=motion_path.stat().st_size + gs_path.stat().st_size)
                    (storage_dir / f"frame_{self.current_frame:03d}.json").write_text(json.dumps(stats, indent=2))
            time_stats = self.timer.display()
            self.trainer.routine_info(f"Time stats at {self.global_step} step:\n{time_stats}\n")
            self.timer.reset()

    def save_selective_image_diagnostics(self, viewpoint, image_path, camera_name):
        if self.config.motion_mode in QUANTIZED_MOTION_MODES:
            self.save_quantized_image_diagnostics(viewpoint, image_path, camera_name)
            return
        if self.config.motion_mode == 'legacy':
            return
        import yaml
        from pathlib import Path
        config = yaml.safe_load(Path(self.config.selective_diagnostics_config).read_text())
        if not config.get('enabled', True) or not config.get('online_enabled', True):
            return
        if config.get('online_steps') is not None and self.global_step not in config['online_steps']:
            return
        if config.get('online_frames') is not None and self.current_frame not in config['online_frames']:
            return
        if config.get('online_cameras') is not None and camera_name not in config['online_cameras']:
            return
        if self.trainer.is_first_frame or not self.selective.members:
            self.trainer.frame_diagnostic_note = 'First frame has no selective tree; image overlays skipped.'
            return
        self.trainer.synchronize()
        started = time.perf_counter()
        from .model.selective_plot import draw_frame, project_with_matrix
        packet = self.selective.capture()
        packet = {k: v for k, v in packet.items() if k != 'optimizer'}
        packet['state'] = {k: v.detach().cpu().clone() for k, v in packet['state'].items()}
        packet['members'] = [indices.detach().cpu() for indices in packet['members']]
        if self.selective.committed:
            # Base GS already contains committed motion; do not apply it twice.
            xyz = self.gaussians.get_xyz_ori[:len(self.selective.owner)].detach().cpu().numpy()
            label = ('fixed fine-leaf SHARED cut (committed)'
                     if self.config.motion_mode == 'learned_selective' and self.motion_config.fixed_leaf_ablation
                     else 'committed cut')
            extra_gs = len(self.gaussians_incr.get_xyz)
        else:
            stage, owner = self.selective.stage(self.global_step)
            if self.config.motion_mode == 'learned_selective':
                xyz = self.selective.transform(step=self.global_step, training=False, update_diagnostics=False)[0].detach().cpu().numpy()
                states, owner = self.selective.cut_from_gates(self.selective.gates(self.global_step, False))
                packet['state']['states'] = states.detach().cpu().clone()
            else:
                xyz = self.selective.transform(owner)[0].detach().cpu().numpy()
            packet['state']['owner'] = owner.detach().cpu()
            extra_gs = 0
            label = 'selected cut / current refinement' if self.selective.selected else f'provisional {stage} owners'
            if self.config.motion_mode == 'learned_selective' and self.motion_config.fixed_leaf_ablation:
                label = 'fixed fine-leaf SHARED cut'
            if not self.selective.selected and self.config.motion_mode == 'selective':
                # Current fitting owners are candidate SHARED/KEEP regions, not the final cut.
                packet['state']['states'].fill_(2)
                nodes = packet['state']['owner'].unique()
                packet['state']['states'][nodes] = torch.where(
                    packet['state']['evidence'][nodes] == 2, 1, 0)
        matrix = viewpoint.full_proj_transform.detach().cpu().numpy()
        height, width = viewpoint.image.shape[-2:]
        config['camera'] = camera_name
        draw_frame(packet, None, image_path, image_path.parent, self.current_frame, config,
                   xyz=xyz, projection=lambda points: project_with_matrix(points, matrix, (width, height)),
                   output_stem=image_path.stem, stage_label=label, extra_gs=extra_gs, write_json=True)
        self.trainer.record_io('drawing', started)

    def validation_step(
        self, viewpoint: TensorSpace, idx: int, loader_idx: int
    ) -> dict:
        if viewpoint.image.is_cuda:
            torch.cuda.synchronize(viewpoint.image.device)
        t0 = time.perf_counter()
        image = self.forward(viewpoint)["render"].clamp(0.0, 1.0)
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
        t1 = time.perf_counter()
        gt_image = viewpoint.image
        if (self.config.motion_mode == 'learned_selective' and not self.trainer.is_first_frame
                and not self.selective.committed and self.motion_config.diagnose_soft_hard):
            xyz, rotation = self.selective.transform(step=self.global_step, training=False, soft=True,
                                                      update_diagnostics=False)
            self.gaussians.rigid_xyz, self.gaussians.rigid_rotation = xyz, rotation
            try:
                soft_image = self.renderer(viewpoint, self.background)['render'].clamp(0., 1.)
            finally:
                self.gaussians.rigid_xyz = self.gaussians.rigid_rotation = None
            self.log_dict({'learned/soft_minus_hard_l1': l1_loss(soft_image, gt_image) - l1_loss(image, gt_image),
                           'learned/soft_minus_hard_psnr': psnr(soft_image, gt_image) - psnr(image, gt_image)})
            self.selective.last_metrics['soft_hard_diagnostic_step'] = self.global_step
            self.selective.last_metrics['soft_minus_hard_l1'] = float(l1_loss(soft_image, gt_image) - l1_loss(image, gt_image))
            self.selective.last_metrics['soft_minus_hard_psnr'] = float(psnr(soft_image, gt_image) - psnr(image, gt_image))
        if idx <= self.config.num_saving_images:
            root = self.datamodule.config.root
            path = self.output_dir / f"image" / viewpoint.path.relative_to(root)
            save_dir, name, ext = path.parent, path.stem, path.suffix
            save_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.synchronize()
            io_started = time.perf_counter()
            image_path = save_dir / f"{name}_step{self.global_step}{ext}"
            save_image(image, image_path)
            self.trainer.record_io('saving', io_started)
            self.save_selective_image_diagnostics(viewpoint, image_path, name)

            testing_steps = self.trainer.testing_steps_for_frame + [self.num_steps]
            if self.global_step == testing_steps[0]:
                io_started = time.perf_counter()
                save_image(gt_image, save_dir / f"{name}_gt{ext}")
                self.trainer.record_io('saving', io_started)

        metrics = dict(l1=l1_loss(image, gt_image), psnr=psnr(image, gt_image))
        if self.config.full_eval:
            metrics["ssim"] = ssim(image, gt_image)
            if self.config.lpips_required:
                metrics["lpips"] = lpips(image, gt_image,net_type='vgg').mean()
            metrics["forward_seconds"] = t1 - t0
        return metrics

    def validation_end(self, results: Union[dict, List[dict]], num_loaders: int = 1):
        def fn(metrics_list, idx):
            mean = lambda xs: sum(xs) / len(xs)
            metrics = {k: mean([ms[k] for ms in metrics_list]) for k in metrics_list[0]}
            if 'forward_seconds' in metrics:
                total_forward_seconds = sum(ms['forward_seconds'] for ms in metrics_list)
                metrics['mean_forward_seconds'] = metrics.pop('forward_seconds')
                metrics['fps'] = len(metrics_list) / total_forward_seconds
            name = self.datamodule.eval_names[idx]
            self.log_dict({f"eval-{name}/{k}": v for k, v in metrics.items()})
            self.trainer.frame_eval_metrics[name] = {k: float(v) for k, v in metrics.items()}
            self.trainer.routine_info(
                f"Evaluate {name} dataset at step {self.global_step}:\n\t"
                + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
            )

        results = [results] if num_loaders == 1 else results
        for idx, result in enumerate(results):
            fn(result, idx)

        self.log_histogram("scene/opacity_histogram", self.gaussians.get_opacity)
        self.log("total_points", self.gaussians.get_xyz.shape[0])

    def save_point_cloud(self):
        skiping = (self.current_frame not in self.config.saving_gs_frames
                   if self.config.saving_gs_frames is not None else
                   self.current_frame % self.config.saving_gs_every_n_frames > 0)
        if not self.trainer.is_last_frame and skiping:
            return

        skiping = self.global_step not in self.config.saving_gs_steps
        if not self.trainer.is_last_step and skiping:
            return

        filename = f"gaussians-{self.current_frame}-{self.global_step}.ply"
        save_path = self.output_dir / "point_cloud" / filename
        self.trainer.routine_info(f"Save gaussians to {save_path}")
        if self.trainer.is_first_frame or self.config.motion_mode in {'selective', 'learned_selective', *QUANTIZED_MOTION_MODES}:
            self.gaussians.save_ply(save_path)
        else:
            self.gaussians_incr.save_ply(save_path)

        if not self.trainer.is_first_frame and self.trainer.is_last_step:
            filename = f"initial-gaussians-{self.current_frame}.ply"
            save_path = self.output_dir / "point_cloud" / filename
            # No file is written here; do not report a completed save.

    def densify_gaussians(
        self,
        current_step: int,
        visibility_filter: Float[Tensor, "n"],
        viewspace_point_tensor: Float[Tensor, "n 3"],
        radii: Float[Tensor, "n"],
    ):
        # Keep track of max radii in image-space for pruning
        self.gaussians.max_radii2D[visibility_filter] = torch.max(
            self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        self.gaussians.add_densification_stats(
            viewspace_point_tensor, visibility_filter
        )

        opacity_reset_interval = self.config.densify.opacity_reset_interval
        max_gaussians = self.config.densify.max_gaussians

        densify_started = current_step > self.config.densify.from_step
        at_interval = current_step % self.config.densify.interval == 0
        within_limit = self.gaussians.get_xyz.shape[0] < max_gaussians
        if densify_started and at_interval and within_limit:
            exceed_reset_inverval = current_step > opacity_reset_interval
            size_threshold = 20 if exceed_reset_inverval else None
            self.gaussians.densify_and_prune(
                self.config.densify.grad_threshold,
                0.005,
                self.datamodule.cameras_extent,
                size_threshold,
            )

        white_background = self.datamodule.config.white_background
        at_reset_interval = current_step % opacity_reset_interval == 0
        at_densify_start = current_step == self.config.densify.from_step
        if at_reset_interval or (white_background and at_densify_start):
            self.gaussians.reset_opacity()

    def adaptive_densify_gaussians(
        self,
        visibility_filter: Bool[Tensor, "n"],
        viewspace_point_tensor: Float[Tensor, "n 3"],
        radii: Float[Tensor, "n"],
    ):
        num_init_gs = self.gaussians.get_xyz.shape[0]
        num_stage1_steps = self.deform.config.num_stage1_steps
        current_stage2_step = self.global_step - num_stage1_steps

        # Keep track of max radii in image-space for pruning
        visibility_filter_init = visibility_filter[:num_init_gs]
        radii_init = radii[:num_init_gs]
        self.gaussians.max_radii2D[visibility_filter_init] = torch.max(
            self.gaussians.max_radii2D[visibility_filter_init],
            radii_init[visibility_filter_init],
        )
        self.gaussians.add_densification_stats(
            viewspace_point_tensor, visibility_filter_init, align="left"
        )

        if current_stage2_step > 0:
            visibility_filter_incr = visibility_filter[num_init_gs:]
            radii_incr = radii[num_init_gs:]
            self.gaussians_incr.max_radii2D[visibility_filter_incr] = torch.max(
                self.gaussians_incr.max_radii2D[visibility_filter_incr],
                radii_incr[visibility_filter_incr],
            )
            self.gaussians_incr.add_densification_stats(
                viewspace_point_tensor, visibility_filter_incr, align="right"
            )

        if self.global_step == num_stage1_steps:
            logging.info("Apply deformation to previous gaussians")
            delta_xyz, delta_rotation = self.deform(self.gaussians.get_xyz.detach())
            self.gaussians.apply_deformation(delta_xyz, delta_rotation)
            self.gaussians.adaptive_densify_and_clone(
                self.deform.config.densify_grad_threshold,
                self.datamodule.cameras_extent,
            )
            self.gaussians_incr.restore(
                self.gaussians.capture(), self.config.gaussian_stage2
            )
            mask = torch.zeros_like(self.gaussians.get_xyz[:, 0], dtype=torch.bool)
            mask[num_init_gs:] = True
            self.gaussians.prune_points(mask)
            self.gaussians_incr.prune_points(~mask)

        densify_interval = self.deform.config.densify_interval
        if current_stage2_step > 0 and current_stage2_step % densify_interval == 0:
            self.gaussians_incr.adaptive_densify_and_prune(
                self.deform.config.densify_grad_threshold,
                self.deform.config.opacity_threshold,
                self.datamodule.cameras_extent,
                20,
            )
