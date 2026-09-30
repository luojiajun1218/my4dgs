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

from .data import DataModule
from .model.deformation import Deformation, DeformConfig
from .model.gated_four import GatedFourDeformation
from .model.gaussian import GaussianModel
from .model.selective_motion import SelectiveMotion, SelectiveMotionConfig
from .model.selective_diagnostics import summarize, concise
from .model.learned_selective import LearnedSelectiveMotion, LearnedSelectiveConfig
from .model.residual_motion import (QuantizedResidualConfig, QuantizedResidualMotion,
    QuantizedNewGaussianView, RAW_FIELDS, RAW_ATTRIBUTES, LAYOUT_VERSION,
    model_raw, reference_digest, replay_frame)
from .model import residual_codec

import json
import csv
import numpy as np
import time

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
    selective_motion: SelectiveMotionConfig = field(default_factory=SelectiveMotionConfig)
    learned_selective_motion: LearnedSelectiveConfig = field(default_factory=LearnedSelectiveConfig)
    quantized_residual: QuantizedResidualConfig = field(default_factory=QuantizedResidualConfig)
    selective_diagnostics_config: str = 'config/diagnostics/selective_motion.yaml'


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
        if self.config.motion_mode not in {'legacy', 'selective', 'learned_selective', 'quantized_residual'}:
            raise ValueError('motion_mode must be legacy, selective, learned_selective or quantized_residual')
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
        if self.config.motion_mode == 'quantized_residual':
            cfg = self.config.quantized_residual
            cfg.validate()
            if self.trainer.config.num_incr_steps != cfg.total_steps or self.config.deformation.num_stage1_steps != cfg.motion_steps:
                raise ValueError('Quantized residual budgets must agree with trainer and deformation stage boundary')
            if self.config.random_background:
                raise ValueError('Quantized residual comparisons require a fixed background')
            self.residual_motion = QuantizedResidualMotion(cfg)
            self.quant_previous_new_count = 0
            self.quant_first_frame_gs_count = None
            self.quant_previous_reference = None
            self.quant_previous_sha256 = None
            self.quant_deleted_indices = None
            self.quant_report = None
            self.quant_predictor_tree = None
            self.quant_finalized = False
            self.quant_header_cache = {}
            self.quant_bootstrap_bits = 0
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

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["gaussians"] = self.gaussians.capture()
        if not self.trainer.is_first_frame:
            if self.config.motion_mode in {'selective', 'learned_selective', 'quantized_residual'} and self.gaussians_incr.optimizer is None:
                ckpt['gaussians_incr'] = None
            else:
                ckpt["gaussians_incr"] = self.gaussians_incr.capture()
        ckpt["deformation"] = self.deform.capture()
        if self.config.motion_mode in {'selective', 'learned_selective'}:
            ckpt['selective_motion'] = self.selective.capture()
            if self.config.motion_mode == 'learned_selective':
                ckpt['learned_change_reference'] = self.change_reference

        if self.config.motion_mode == 'quantized_residual':
            ckpt['motion_mode'] = 'quantized_residual'
            ckpt['quantized_residual_motion'] = self.residual_motion.capture()
            ckpt['quantized_frame_state'] = dict(previous_new_count=self.quant_previous_new_count,
                first_frame_gs_count=self.quant_first_frame_gs_count, previous_reference=self.quant_previous_reference,
                deleted_indices=self.quant_deleted_indices, report=self.quant_report, finalized=self.quant_finalized,
                bootstrap_bits=self.quant_bootstrap_bits)

    def on_load_checkpoint(self, ckpt: dict):
        if (not self.trainer.is_first_frame and ckpt.get('motion_mode') == 'quantized_residual'
                and self.config.motion_mode != 'quantized_residual'):
            raise ValueError('Incremental quantized residual checkpoint cannot resume a different motion mode')
        self.gaussians.restore(ckpt["gaussians"], self.config.gaussian)
        if not self.trainer.is_first_frame:
            if self.config.motion_mode in {'selective', 'learned_selective', 'quantized_residual'} and ckpt.get('gaussians_incr') is None:
                self.gaussians_incr = GaussianModel(self.config.sh_degree)
            else:
                self.gaussians_incr.restore(
                    ckpt["gaussians_incr"], self.config.gaussian_stage2
                )
        if self.config.motion_mode == 'quantized_residual':
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
                self.quant_header_cache = {}
                self.quant_bootstrap_bits = frame_state.get('bootstrap_bits', 0)
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

    def forward(self, viewpoint: TensorSpace, training: bool = False, **kwargs) -> dict:
        bg_color = self.background
        if training and self.config.random_background:
            bg_color = torch.rand(3, device=bg_color.device)

        results = {}
        if not self.trainer.is_first_frame:
            if self.config.motion_mode == 'quantized_residual':
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

        if not self.trainer.is_first_frame and self.trainer.is_first_step:
            self.timer.clock("frame setup").start()
            num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
            if self.config.motion_mode == 'quantized_residual':
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

        if self.config.motion_mode == 'quantized_residual' and not self.trainer.is_first_frame:
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
            "loss/rgb": loss_rgb,
            "loss/dssim": loss_dssim,
            "psnr": psnr(image, gt_image).mean().double(),
            "gs": (len(render_pkg["radii"]) if self.config.motion_mode == "quantized_residual"
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

        if self.config.motion_mode == 'quantized_residual' and not self.trainer.is_first_frame:
            render_pkg['quantized_started'] = selective_started
            bits, rate_details = self.quantized_rate(render_pkg.get('quantized_new_view'))
            cfg = self.config.quantized_residual
            multiplier = cfg.lambda_rate * self.residual_motion.rate_scale(current_step)
            loss_rate = multiplier * bits / self.quant_first_frame_gs_count
            loss = loss + loss_rate
            metrics.update({'rate/' + name: value for name, value in rate_details.items()})
            metrics['rate/estimated_total_bits'] = bits
            metrics['loss/rate'] = loss_rate
            self.residual_motion.last_metrics.update({name: float(value.detach()) for name, value in metrics.items() if torch.is_tensor(value)})

        self.timer.clock("training step").stop()

        self.timer.clock("tensorboard logging").start()
        self.log_dict(metrics)
        self.timer.clock("tensorboard logging").stop()

        return loss, metrics, render_pkg

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
        if self.config.motion_mode == 'quantized_residual':
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
    def prepare_quantized_frame(self):
        started = time.perf_counter()
        self.quant_previous_reference = {name: value.cpu() for name, value in model_raw(self.gaussians).items()}
        self.quant_previous_sha256 = reference_digest(self.quant_previous_reference)
        if self.quant_first_frame_gs_count is None:
            self.quant_first_frame_gs_count = len(self.gaussians.get_xyz)
        count = self.quant_previous_new_count
        if not self.config.quantized_residual.enable_incremental and count:
            raise ValueError('Incremental-disabled residual run cannot delete previous Gaussians')
        if count >= len(self.gaussians.get_xyz):
            raise ValueError('Previous new-point compensation would remove all reference Gaussians')
        # Stable activated-opacity order references exact previous decoded point order.
        deleted = torch.argsort(self.gaussians.get_opacity.detach().flatten(), stable=True)[:count]
        self.quant_deleted_indices = deleted.to(torch.int32).cpu()
        if count:
            self.gaussians.prune_points(deleted)
        self.gaussians_incr = GaussianModel(self.config.sh_degree)
        self.renderer.gaussians = self.gaussians
        self.freeze_quantized_base()
        layout_started = time.perf_counter()
        self.residual_motion.setup(self.gaussians.get_xyz_ori, self.gaussians._rotation, self.datamodule.cameras_extent)
        self.quant_predictor_tree = None
        self.quant_report = None
        self.quant_finalized = False
        self.quant_header_cache = {}
        self.quant_bootstrap_bits = 0
        self.trainer.synchronize()
        self.residual_motion.stage_seconds['layout'] = time.perf_counter() - layout_started
        if self.current_frame == 1:
            bootstrap = dict(layout_version=LAYOUT_VERSION, config=self.quantized_metadata()['config'],
                extent=float(self.datamodule.cameras_extent), first_frame_gs_count=self.quant_first_frame_gs_count,
                reference_sha256=reference_digest(self.quant_previous_reference),
                reference_kind='External shared first-frame raw Gaussian model in stable tensor order',
                active_sh_degree=self.gaussians.active_sh_degree, max_sh_degree=self.gaussians.max_sh_degree)
            path = self.output_dir / 'incremental_stream' / 'bootstrap.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bootstrap, separators=(',', ':')))
            self.quant_bootstrap_bits = path.stat().st_size * 8
        self.trainer.synchronize()
        self.residual_motion.stage_seconds['frame_prepare'] = time.perf_counter() - started

    def quantized_metadata(self):
        from dataclasses import asdict
        return dict(frame=self.current_frame, layout_version=LAYOUT_VERSION,
                    config=asdict(self.config.quantized_residual), extent=float(self.datamodule.cameras_extent),
                    first_frame_gs_count=self.quant_first_frame_gs_count,
                    previous_reference_sha256=self.quant_previous_sha256,
                    active_sh_degree=self.gaussians.active_sh_degree, max_sh_degree=self.gaussians.max_sh_degree)

    def quantized_components(self, view=None, integer=False):
        packet = self.residual_motion.symbol_tensors(integer=integer)
        packet['deleted_prev_indices'] = self.quant_deleted_indices
        models = self.residual_motion.entropy_models()
        if view is not None:
            packet.update(view.integer_packet())
            models.update(view.entropy_models())
        else:
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                raw = getattr(self.gaussians, attr)
                channels = int(np.prod(raw.shape[1:]))
                packet['new_' + name] = torch.empty((0, channels), dtype=torch.int32)
                models['new_' + name] = dict(scale=[self.config.quantized_residual.new_entropy_scales[name]] * channels, sparse=False)
            packet['new_source_old_index'] = torch.empty(0, dtype=torch.int32)
        return packet, models

    def quantized_header_bits(self, view=None):
        retained = int(view.keep.sum()) if view is not None else 0
        if retained in self.quant_header_cache:
            return self.quant_header_cache[retained]
        packet = {f'region_{n}': torch.empty((count, 6), dtype=torch.int32, device='meta')
                  for n, count in enumerate(self.residual_motion.counts)}
        packet['gs_local'] = torch.empty((len(self.residual_motion.base_xyz), 6), dtype=torch.int32, device='meta')
        packet['deleted_prev_indices'] = self.quant_deleted_indices
        models = self.residual_motion.entropy_models()
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            channels = int(np.prod(getattr(self.gaussians, attr).shape[1:]))
            packet['new_' + name] = torch.empty((retained, channels), dtype=torch.int32, device='meta')
            models['new_' + name] = dict(scale=[self.config.quantized_residual.new_entropy_scales[name]] * channels, sparse=False)
        packet['new_source_old_index'] = torch.empty(retained, dtype=torch.int32, device='meta')
        cfg = self.config.quantized_residual
        cost = residual_codec.container_overhead_bits(packet, self.quantized_metadata(), models,
            cfg.entropy_radius, cfg.entropy_chunk_rows) + self.quant_bootstrap_bits
        self.quant_header_cache[retained] = cost
        return cost

    def quantized_rate(self, view=None):
        cfg = self.config.quantized_residual
        motion_symbols = self.residual_motion.symbol_tensors(self.global_step)
        parts = {}
        for name, symbols in motion_symbols.items():
            parts[name] = residual_codec.expected_torchac_bits(symbols, cfg.motion_entropy_scales,
                entropy_radius=cfg.entropy_radius, mask_probability=cfg.tuple_mask_probability,
                support_temperature=cfg.support_temperature, sparse=True)
        zero = next(iter(parts.values())) * 0
        new_bits = zero
        if view is not None:
            point_cost = view.keep_st * 32  # Prediction source index for every retained birth.
            for name, symbols in view.symbols.items():
                row_bits = residual_codec.expected_torchac_bits(symbols.flatten(1), cfg.new_entropy_scales[name],
                    entropy_radius=cfg.entropy_radius, sparse=False, reduction='none')
                point_cost = point_cost + view.keep_st * row_bits
            new_bits = point_cost.sum()
        overhead = self.quantized_header_bits(view)
        deletion_bits = self.quant_deleted_indices.numel() * 32
        total = sum(parts.values()) + new_bits + overhead + deletion_bits
        return total, dict(motion_bits=sum(parts.values()), new_bits=new_bits,
                           deletion_bits=zero + deletion_bits, header_bits=zero + overhead)

    @torch.no_grad()
    def post_quantized_residual(self, render_pkg):
        cfg = self.config.quantized_residual
        motion = self.residual_motion
        if not motion.committed:
            motion.optimizer_step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            if cfg.enable_incremental:
                self.gaussians.add_densification_stats(render_pkg['viewspace_points'], render_pkg['visibility_filter'])
            stage = render_pkg['quantized_stage']
            self.trainer.synchronize()
            motion.stage_seconds[stage] = motion.stage_seconds.get(stage, 0.) + time.perf_counter() - render_pkg['quantized_started']
            if self.global_step == cfg.motion_steps:
                xyz, rotation = motion(self.global_step)
                self.gaussians.apply_selected_motion(xyz, rotation)
                motion.committed = True
                for parameter in motion.parameters():
                    parameter.requires_grad_(False)
                if cfg.enable_incremental:
                    before = len(xyz)
                    self.gaussians.adaptive_densify_and_clone(self.deform.config.densify_grad_threshold,
                        self.datamodule.cameras_extent, sigma_scale=cfg.clone_sigma_scale)
                    self.gaussians_incr.restore(self.gaussians.capture(), self.config.gaussian_stage2)
                    mask = torch.arange(len(self.gaussians.get_xyz), device=self.device) >= before
                    self.gaussians.prune_points(mask)
                    self.gaussians_incr.prune_points(~mask)
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
                    self.gaussians_incr.adaptive_densify_and_prune(self.deform.config.densify_grad_threshold,
                        self.deform.config.opacity_threshold, self.datamodule.cameras_extent, cfg.new_max_screen_size)
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
        estimated, details = self.quantized_rate(view)
        packet, models = self.quantized_components(view, integer=True)
        path = self.output_dir / 'incremental_stream' / f'frame_{self.current_frame:03d}.bin'
        self.trainer.synchronize()
        started = time.perf_counter()
        actual = residual_codec.encode_entropy_frame(path, packet, self.quantized_metadata(),
            entropy_radius=cfg.entropy_radius, chunk_rows=cfg.entropy_chunk_rows, entropy_models=models)
        self.residual_motion.stage_seconds['encode'] = time.perf_counter() - started
        started = time.perf_counter()
        decoded = residual_codec.decode_entropy_frame(path)
        raw = replay_frame(self.quant_previous_reference, decoded['tensors'], decoded['metadata'])
        self.install_quantized_raw(raw)
        self.gaussians_incr = GaussianModel(self.config.sh_degree)
        self.quant_previous_new_count = len(packet['new_source_old_index'])
        self.quant_finalized = True
        self.trainer.synchronize()
        self.residual_motion.stage_seconds['decode'] = time.perf_counter() - started
        chunk_costs = {}
        for name, model in models.items():
            rows, channels = packet[name].shape
            chunks = (rows + cfg.entropy_chunk_rows - 1) // cfg.entropy_chunk_rows
            chunk_costs[name] = (chunks * (residual_codec.ENTROPY_CHUNK.size + residual_codec.UINT32.size * channels)
                                 if channels else 0)
        motion_names = ('region_0', 'region_1', 'region_2', 'gs_local')
        payload_costs = {name: cost - chunk_costs.get(name, 0) for name, cost in actual['tensor_bytes'].items()}
        actual_classes = dict(motion_bytes=sum(payload_costs[name] for name in motion_names),
            new_bytes=sum(cost for name, cost in payload_costs.items() if name.startswith('new_')),
            deletion_bytes=payload_costs['deleted_prev_indices'],
            header_bytes=actual['header_bytes'] + actual['chunk_header_bytes'] + self.quant_bootstrap_bits // 8)
        self.quant_report = dict(frame=self.current_frame, estimated_total_bits=float(estimated),
            bootstrap_bytes=self.quant_bootstrap_bits // 8,
            total_stream_bytes=actual['total_bytes'] + self.quant_bootstrap_bits // 8,
            estimated_bits_by_class={name: float(value) for name, value in details.items()},
            actual=actual, actual_bytes_by_class=actual_classes, activities=self.residual_motion.statistics(),
            stage_seconds=self.residual_motion.stage_seconds, new_count=self.quant_previous_new_count,
            deleted_count=self.quant_deleted_indices.numel(), normalization_first_frame_gs=self.quant_first_frame_gs_count,
            rate_semantics='Same quantized tuples, priors and 16bit CDF. Arithmetic termination/rounding is represented in actual bytes.',
            method_semantics='Fixed nested spatial candidates with quantized residual-support effective hierarchy. '
            'Position offsets add; world orientation composes coarse, middle, fine, then GS residual. '
            'Regional rotations do not rotate Gaussian centers around regional pivots.',
            timing_semantics='frame_prepare includes deletion/layout/bootstrap; layout is a subcomponent. '
            'finalize includes packet preparation/encode/decode/model installation/report preparation; encode and decode are subcomponents. '
            'birth_prediction_train_cpu is included in training; birth_prediction_eval_cpu is included in evaluation or finalization. '
            'Reported pure training excludes frame_prepare and finalize; rendering/backward/data/rate and optimizer remain included.',
            decoded_reference_sha256=reference_digest(raw))
        self.residual_motion.stage_seconds['finalize'] = time.perf_counter() - finalize_started
        path.with_suffix('.json').write_text(json.dumps(self.quant_report, indent=2))

    def save_quantized_image_diagnostics(self, viewpoint, image_path, camera_name):
        if self.trainer.is_first_frame or not self.quant_finalized:
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
                num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
                self.trainer.routine_info(f"Number of incremental gaussians: {num_incr_gs}")
                if self.config.motion_mode == 'legacy' and isinstance(self.deform, GatedFourDeformation):
                    storage_dir = self.output_dir / "frame_storage"
                    storage_dir.mkdir(parents=True, exist_ok=True)
                    motion_path = storage_dir / f"motion_{self.current_frame:03d}.npz"
                    stats = self.deform.export(motion_path)
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
        if self.config.motion_mode == 'quantized_residual':
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
        t0 = time.time()
        image = self.forward(viewpoint)["render"].clamp(0.0, 1.0)
        t1 = time.time()
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
            metrics["fps"] = 1/(t1-t0)
        return metrics

    def validation_end(self, results: Union[dict, List[dict]], num_loaders: int = 1):
        def fn(metrics_list, idx):
            mean = lambda xs: sum(xs) / len(xs)
            metrics = {k: mean([ms[k] for ms in metrics_list]) for k in metrics_list[0]}
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
        if self.trainer.is_first_frame or self.config.motion_mode in {'selective', 'learned_selective', 'quantized_residual'}:
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
