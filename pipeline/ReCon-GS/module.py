from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

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
from .model.binary_motion import BinaryMotion
from .model.rd_tree_motion import RDTreeMotion
from .model.gaussian import GaussianModel

import json
import csv
import numpy as np
import os
from libgs.data.types import storePly
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

    def setup(self):
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
        deformation_cls = (BinaryMotion if self.config.deformation.hierarchy_mode == 'binary'
                           else Deformation)
        if self.config.deformation.hierarchy_mode == 'rd_tree':
            deformation_cls = RDTreeMotion
        self.deform = deformation_cls(self.config.deformation)
        self.binary_views = {}
        self.timer = Timer()

    def on_save_checkpoint(self, ckpt: dict):
        ckpt["gaussians"] = self.gaussians.capture()
        if not self.trainer.is_first_frame:
            ckpt["gaussians_incr"] = self.gaussians_incr.capture()
        ckpt["deformation"] = self.deform.capture()

    def on_load_checkpoint(self, ckpt: dict):
        self.gaussians.restore(ckpt["gaussians"], self.config.gaussian)
        if not self.trainer.is_first_frame:
            self.gaussians_incr.restore(
                ckpt["gaussians_incr"], self.config.gaussian_stage2
            )
        self.deform.restore(ckpt["deformation"])

    def forward(self, viewpoint: TensorSpace, training: bool = False, **kwargs) -> dict:
        bg_color = self.background
        if training and self.config.random_background:
            bg_color = torch.rand(3, device=bg_color.device)

        results = {}
        if not self.trainer.is_first_frame:
            num_stage1_steps = self.deform.config.num_stage1_steps
            enable_deform = False
            if self.global_step < num_stage1_steps:
                enable_deform = True
            elif self.global_step == num_stage1_steps and training:
                enable_deform = True

            if enable_deform:
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
        for attr in ["delta_xyz", "delta_rot", "noise_scale"]:
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
            self.binary_views = {}
            self.timer.clock("frame setup").start()
            num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
            if self.config.merge_to_base and num_incr_gs > 0:
                logging.info(f"Merge and prune {num_incr_gs} gaussians")
                self.gaussians.extend(self.gaussians_incr)
                indices = self.gaussians.get_opacity.flatten().sort().indices
                self.gaussians.prune_points(indices[:num_incr_gs])

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

        if (self.config.deformation.hierarchy_mode == 'binary' and
                not self.trainer.is_first_frame and
                current_step <= self.deform.config.num_stage1_steps):
            # Only training cameras enter proposal evaluation; never test views.
            self.binary_views[str(viewpoint.path)] = viewpoint

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
            "gs": self.renderer.gaussians.get_xyz.shape[0],
        }

        if "delta_xyz" in render_pkg and self.config.lambda_deform > 0:
            loss_reg = self.deform.reg_loss()
            loss = loss + self.config.lambda_deform * loss_reg
            metrics["loss/reg"] = loss_reg

        if "delta_xyz" in render_pkg and self.config.deformation.hierarchy_mode == 'binary':
            bits = self.deform.rate_bits()
            rate_loss = self.deform.config.binary_lambda_rate * bits / len(self.deform.xyz)
            loss = loss + rate_loss
            metrics.update({'loss/rate': rate_loss, 'motion/estimated_bits': bits.detach(),
                            'motion/coefficients': self.deform.get_grid_num,
                            'motion/splits': self.deform.split_count})

        if "delta_xyz" in render_pkg and self.config.deformation.hierarchy_mode == 'rd_tree':
            envelope = self.deform.envelope_loss()
            loss = loss + envelope
            projection = self.deform.project()
            metrics.update({'loss/tree_envelope': envelope,
                            'motion/payload_bits': projection.payload_bits,
                            'motion/coefficients': self.deform.get_grid_num,
                            'motion/splits': len(projection.split_nodes)})

        self.timer.clock("training step").stop()

        self.timer.clock("tensorboard logging").start()
        self.log_dict(metrics)
        self.timer.clock("tensorboard logging").stop()

        return loss, metrics, render_pkg

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
    def post_training_step_incr(self, render_pkg):
        num_stage1_steps = self.deform.config.num_stage1_steps

        # Collect output gradients after backward, update motion, then freeze the
        # parent split decision for subsequent steps.
        if self.global_step <= num_stage1_steps and render_pkg.get("delta_xyz") is not None:
            if self.deform.collecting_gradients(self.global_step):
                self.deform.accumulate_gradient_score(render_pkg["delta_xyz"].grad,
                                                     render_pkg["visibility_filter"], step=self.global_step)
            self.deform.optimizer.step()
            nested_split = (self.deform.config.hierarchy_mode == "nested" and
                            self.global_step == self.deform.config.split_step)
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
            self.deform.optimizer.zero_grad(set_to_none=True)

            if (self.deform.config.hierarchy_mode == 'binary' and
                    self.deform.split_due(self.global_step)):
                self.binary_refinement_trial()

        if self.global_step < self.config.densify.until_step:
            self.adaptive_densify_gaussians(
                render_pkg["visibility_filter"],
                render_pkg["viewspace_points"],
                render_pkg["radii"],
            )

        if self.global_step > num_stage1_steps:
            self.gaussians_incr.optimizer.step()
            self.gaussians_incr.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def binary_refinement_trial(self):
        """Matched short no-split/split trials on TRAINING cameras only.

        A batch is accepted as a whole: no per-node attribution is claimed.
        Both branches optimize from the same state for the same number of steps.
        All trial backward calls and their wall time are additional overhead.
        """
        views = list(self.binary_views.values())
        config = self.deform.config
        if len(views) < (2 if config.binary_decision_all_views else config.binary_eval_views + 1):
            return
        proposal = self.deform.propose()
        if not len(proposal['nodes']):
            self.deform.clear_scores()
            return
        snapshot = self.deform.capture()
        eval_views = views if config.binary_decision_all_views else views[-config.binary_eval_views:]
        train_views = views if config.binary_decision_all_views else views[:-config.binary_eval_views]
        offset = len(self.deform.events) * config.binary_probe_steps
        probe_views = [train_views[(offset + i) % len(train_views)]
                       for i in range(config.binary_probe_steps)]
        torch.cuda.synchronize()
        started = time.perf_counter()

        def photo(view):
            image = self.forward(view, training=True)['render']
            return ((1 - self.config.lambda_dssim) * l1_loss(image, view.image) +
                    self.config.lambda_dssim * (1 - ssim(image, view.image)))

        def optimize_branch(split):
            self.deform.restore(snapshot)
            if split:
                before = self.deform(self.gaussians.get_xyz_ori)
                self.deform.activate(proposal)
                after = self.deform(self.gaussians.get_xyz_ori)
                for a, b in zip(before, after):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            with torch.enable_grad():
                for view in probe_views:
                    loss = photo(view) + self.config.lambda_deform * self.deform.reg_loss()
                    loss = loss + config.binary_lambda_rate * self.deform.rate_bits() / len(self.deform.xyz)
                    parameters = list(self.deform.parameters())
                    gradients = torch.autograd.grad(loss, parameters)
                    self.deform.optimizer.zero_grad(set_to_none=True)
                    for parameter, gradient in zip(parameters, gradients):
                        parameter.grad = gradient
                    self.deform.optimizer.step()
                self.deform.optimizer.zero_grad(set_to_none=True)
            distortion = sum(float(photo(view)) for view in eval_views) / len(eval_views)
            bits = float(self.deform.rate_bits())
            cost = distortion + config.binary_lambda_rate * bits / len(self.deform.xyz)
            return self.deform.capture(), distortion, bits, cost

        # Preserve the outer training RNG; trial camera streams are explicit.
        with torch.random.fork_rng(devices=[self.device]):
            baseline, d0, r0, j0 = optimize_branch(False)
        with torch.random.fork_rng(devices=[self.device]):
            candidate, d1, r1, j1 = optimize_branch(True)
        accepted = bool(np.isfinite(j1) and j1 < j0)
        self.deform.restore(candidate if accepted else baseline)
        self.deform.probe_backward_steps += 2 * config.binary_probe_steps
        self.deform.clear_scores()
        torch.cuda.synchronize()
        event = dict(step=self.global_step, accepted=accepted, candidate_nodes=len(proposal['nodes']),
                     eligible=proposal['eligible'], depths=proposal['depths'].cpu().tolist(),
                     distortion_before=d0, distortion_after=d1, bits_before=r0, bits_after=r1,
                     objective_before=j0, objective_after=j1, wall_seconds=time.perf_counter() - started,
                     train_views=[str(v.path) for v in probe_views],
                     decision_views=[str(v.path) for v in eval_views],
                     decision_uses_all_training_views=config.binary_decision_all_views,
                     extra_backward_steps=2 * config.binary_probe_steps,
                     active_splits=self.deform.split_count)
        self.deform.events.append(event)
        path = self.output_dir / 'hierarchy_stats'
        path.mkdir(exist_ok=True)
        (path / f'frame_{self.current_frame:03d}_binary.json').write_text(
            json.dumps(self.deform.events, indent=2))
        logging.info('Binary trial frame=%d step=%d candidates=%d accepted=%s '
                     'D %.7f -> %.7f bits %.0f -> %.0f splits=%d',
                     self.current_frame, self.global_step, len(proposal['nodes']), accepted,
                     d0, d1, r0, r1, self.deform.split_count)

    @torch.no_grad()
    def post_training_step(self, render_pkg: dict):
        self.timer.clock("post training step").start()
        if self.trainer.is_first_frame:
            self.post_training_step_init(render_pkg)
        else:
            self.post_training_step_incr(render_pkg)
        self.timer.clock("post training step").stop()

        self.save_point_cloud()

        if self.trainer.is_last_step:
            self.timer.clock("frame").stop()
            if not self.trainer.is_first_frame:
                num_incr_gs = self.gaussians_incr.get_xyz.shape[0]
                logging.info(f"Numer of increment gaussians: {num_incr_gs}")
                if self.deform.config.hierarchy_mode == 'rd_tree':
                    storage_dir = self.output_dir / 'frame_storage'
                    storage_dir.mkdir(exist_ok=True)
                    codec_started = time.perf_counter()
                    stats = self.deform.export(storage_dir / f'motion_{self.current_frame:03d}.rdp')
                    gs_path = storage_dir / f'incremental_{self.current_frame:03d}.ply'
                    self.gaussians_incr.save_ply(gs_path)
                    stats.update(frame=self.current_frame, incremental_gs=num_incr_gs,
                                 incremental_bytes=gs_path.stat().st_size,
                                 total_bytes=stats['motion_bytes'] + gs_path.stat().st_size,
                                 export_and_verify_seconds=time.perf_counter() - codec_started)
                    (storage_dir / f'frame_{self.current_frame:03d}.json').write_text(
                        json.dumps(stats, indent=2))
                if self.deform.config.hierarchy_mode == 'binary':
                    storage_dir = self.output_dir / 'frame_storage'
                    storage_dir.mkdir(exist_ok=True)
                    stats = self.deform.export(storage_dir / f'motion_{self.current_frame:03d}.npz')
                    gs_path = storage_dir / f'incremental_{self.current_frame:03d}.ply'
                    self.gaussians_incr.save_ply(gs_path)
                    stats.update(frame=self.current_frame, incremental_gs=num_incr_gs,
                                 incremental_bytes=gs_path.stat().st_size,
                                 total_bytes=stats['motion_bytes'] + gs_path.stat().st_size)
                    (storage_dir / f'frame_{self.current_frame:03d}.json').write_text(
                        json.dumps(stats, indent=2))
                if self.deform.config.hierarchy_mode == "nested":
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
            logging.info(f"Time stats at {self.global_step} step:\n{time_stats}\n")
            self.timer.reset()

    def validation_step(
        self, viewpoint: TensorSpace, idx: int, loader_idx: int
    ) -> dict:
        t0 = time.time()
        image = self.forward(viewpoint)["render"].clamp(0.0, 1.0)
        t1 = time.time()
        gt_image = viewpoint.image
        if idx <= self.config.num_saving_images:
            root = self.datamodule.config.root
            path = self.output_dir / f"image" / viewpoint.path.relative_to(root)
            save_dir, name, ext = path.parent, path.stem, path.suffix
            save_dir.mkdir(parents=True, exist_ok=True)
            save_image(image, save_dir / f"{name}_step{self.global_step}{ext}")

            testing_steps = self.trainer.config.testing_steps + [self.num_steps]
            if self.global_step == testing_steps[0]:
                save_image(gt_image, save_dir / f"{name}_gt{ext}")

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
            logging.info(
                f"Evaluate {name} dataset at step {self.global_step}:\n\t"
                + " | ".join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
            )

        results = [results] if num_loaders == 1 else results
        for idx, result in enumerate(results):
            fn(result, idx)

        self.log_histogram("scene/opacity_histogram", self.gaussians.get_opacity)
        self.log("total_points", self.gaussians.get_xyz.shape[0])

    def save_point_cloud(self):
        skiping = self.current_frame % self.config.saving_gs_every_n_frames > 0
        if not self.trainer.is_last_frame and skiping:
            return

        skiping = self.global_step not in self.config.saving_gs_steps
        if not self.trainer.is_last_step and skiping:
            return

        filename = f"gaussians-{self.current_frame}-{self.global_step}.ply"
        save_path = self.output_dir / "point_cloud" / filename
        logging.info(f"\nSave gaussians to {save_path}")
        if self.trainer.is_first_frame:
            self.gaussians.save_ply(save_path)
        else:
            self.gaussians_incr.save_ply(save_path)

        if not self.trainer.is_first_frame and self.trainer.is_last_step:
            filename = f"initial-gaussians-{self.current_frame}.ply"
            save_path = self.output_dir / "point_cloud" / filename
            logging.info(f"Save deformed initial gaussians to {save_path}")

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
