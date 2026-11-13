"""首帧静态 GS 模型及单步训练逻辑。

负责模型与渲染器初始化、图像损失、优化器更新、首帧增密和裁剪、验证及
模型检查点状态；训练循环和保存时机由同目录 trainer.py 调度。
本文件不负责动态帧运动、逐帧更新或码流编解码。
"""

from dataclasses import dataclass, field
import time
from typing import List, Optional

import torch
from absl import logging
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.metric import lpips, psnr, ssim
from libgs.pipeline import Module as BaseModule, ModuleConfig as BaseModuleConfig
from libgs.renderer import Renderer, RendererConfig
from libgs.renderer.network_gui import interact_with_gui
from libgs.utils.time import Timer

from ..model.gaussian import GaussianModel as BaseGaussianModel


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
class DeformConfig:
    """保留现有静态配置的字段；首帧训练不执行动态形变。"""
    quantile: float = 0.05
    lr: float = 0.0005
    num_stage1_steps: int = 100
    max_gs_per_grid: int = 5
    num_grid_levels: int = 3
    grid_level_ratio: int = 2
    momentum: Optional[float] = 0.6
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01
    grid_reset_interval: int = 4


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 3
    random_background: bool = False
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


class GaussianModel(BaseGaussianModel):
    """复用项目的 GS 属性，只在静态训练渲染时加入可选位置噪声。"""

    @property
    def get_xyz(self):
        xyz = super().get_xyz
        noise_scale = getattr(self, 'noise_scale', None)
        return xyz if noise_scale is None else xyz + torch.randn_like(xyz.detach()) * noise_scale


class Module(BaseModule):
    @property
    def current_frame(self):
        return self.datamodule.current_frame

    @property
    def num_steps(self):
        return self.trainer.num_steps

    def setup(self):
        self._eval_history = {}
        self._eval_frames = set()
        color = torch.ones if self.datamodule.config.white_background else torch.zeros
        self.register_buffer('background', color(3, device=self.device))
        self.gaussians = GaussianModel(self.config.sh_degree)
        self.gaussians.create_from_pcd(self.datamodule.scene.point_cloud, self.datamodule.cameras_extent)
        self.gaussians.training_setup(self.config.gaussian)
        self.renderer = Renderer(self.config.renderer, self.gaussians)
        self.timer = Timer()

    def on_save_checkpoint(self, ckpt):
        ckpt['gaussians'] = self.gaussians.capture()
        ckpt['deformation'] = {}

    def on_load_checkpoint(self, ckpt):
        if ckpt.get('current_frame') != 0:
            raise ValueError('First-frame trainer requires a frame-0 checkpoint')
        self.gaussians.restore(ckpt['gaussians'], self.config.gaussian)

    def forward(self, viewpoint, training=False, **kwargs):
        background = self.background
        if training and self.config.random_background:
            background = torch.rand(3, device=background.device)
        if training:
            self.gaussians.noise_scale = self.config.noise_scale
        try:
            return self.renderer(viewpoint, background, **kwargs)
        finally:
            self.gaussians.noise_scale = None

    def pre_training_step(self):
        interact_with_gui(self.global_step, self.renderer.config, self,
                          self.datamodule.config.root, self.num_steps)
        if self.trainer.is_first_step:
            self.timer.clock('frame').start()

    def training_step(self, viewpoint, current_step):
        self.timer.clock('training step').start()
        self.gaussians.update_learning_rate(current_step)
        if current_step % 1000 == 0:
            self.gaussians.oneupSHdegree()
        if current_step - 1 == self.renderer.config.debug_from:
            self.renderer.config.debug = True
        package = self.forward(viewpoint, training=True)
        image = package['render']
        loss_rgb = l1_loss(image, viewpoint.image)
        loss_dssim = 1. - ssim(image, viewpoint.image)
        loss = (1. - self.config.lambda_dssim) * loss_rgb + self.config.lambda_dssim * loss_dssim
        metrics = {'loss/rgb': loss_rgb, 'loss/dssim': loss_dssim,
                   'psnr': psnr(image, viewpoint.image).mean().double(),
                   'gs': self.renderer.gaussians.get_xyz.shape[0]}
        self.timer.clock('training step').stop()
        self.timer.clock('tensorboard logging').start()
        self.log_dict(metrics)
        self.timer.clock('tensorboard logging').stop()
        return loss, metrics, package

    @torch.no_grad()
    def post_training_step_init(self, package):
        if self.global_step < self.config.densify.until_step:
            self.densify_gaussians(self.global_step, package['visibility_filter'],
                                   package['viewspace_points'], package['radii'])
        self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def post_training_step(self, package):
        self.timer.clock('post training step').start()
        self.post_training_step_init(package)
        self.timer.clock('post training step').stop()
        self.save_point_cloud()
        if self.trainer.is_last_step:
            self.timer.clock('frame').stop()
            logging.info('Time stats at %d step:\n%s\n', self.global_step, self.timer.display())
            self.timer.reset()

    def validation_step(self, viewpoint, idx, loader_idx):
        start = time.time()
        image = self.forward(viewpoint)['render'].clamp(0., 1.)
        elapsed = time.time() - start
        target = viewpoint.image
        if idx <= self.config.num_saving_images:
            path = self.output_dir / 'image' / viewpoint.path.relative_to(self.datamodule.config.root)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(image, path.parent / f'{path.stem}_step{self.global_step}{path.suffix}')
            testing_steps = self.trainer.config.testing_steps + [self.num_steps]
            if self.global_step == testing_steps[0]:
                save_image(target, path.parent / f'{path.stem}_gt{path.suffix}')
        metrics = dict(l1=l1_loss(image, target), psnr=psnr(image, target))
        if self.config.full_eval:
            metrics['ssim'] = ssim(image, target)
            if self.config.lpips_required:
                metrics['lpips'] = lpips(image, target, net_type='vgg').mean()
            metrics['fps'] = 1 / elapsed
        return metrics

    def validation_end(self, results, num_loaders=1):
        results = [results] if num_loaders == 1 else results
        for index, rows in enumerate(results):
            metrics = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
            name = self.datamodule.eval_names[index]
            self.log_dict({f'eval-{name}/{key}': value for key, value in metrics.items()})
            logging.info('Evaluate %s dataset at step %d:\n\t%s', name, self.global_step,
                         ' | '.join(f'{key.upper()}: {value:.4f}' for key, value in metrics.items()))
        self.log_histogram('scene/opacity_histogram', self.gaussians.get_opacity)
        self.log('total_points', self.gaussians.get_xyz.shape[0])

    def save_point_cloud(self):
        if not self.trainer.is_last_frame and self.current_frame % self.config.saving_gs_every_n_frames:
            return
        if not self.trainer.is_last_step and self.global_step not in self.config.saving_gs_steps:
            return
        path = self.output_dir / 'point_cloud' / f'gaussians-{self.current_frame}-{self.global_step}.ply'
        logging.info('\nSave gaussians to %s', path)
        self.gaussians.save_ply(path)

    def densify_gaussians(self, current_step, visibility_filter, viewspace_point_tensor, radii):
        self.gaussians.max_radii2D[visibility_filter] = torch.max(
            self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        config = self.config.densify
        if (current_step > config.from_step and current_step % config.interval == 0 and
                self.gaussians.get_xyz.shape[0] < config.max_gaussians):
            size_threshold = 20 if current_step > config.opacity_reset_interval else None
            self.gaussians.densify_and_prune(config.grad_threshold, 0.005,
                                             self.datamodule.cameras_extent, size_threshold)
        if (current_step % config.opacity_reset_interval == 0 or
                (self.datamodule.config.white_background and current_step == config.from_step)):
            self.gaussians.reset_opacity()
