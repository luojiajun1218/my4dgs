"""首帧静态训练循环与检查点调度。

负责取训练视角、执行前向与反向传播、按步骤调用模型更新、验证和保存；
具体渲染、损失与 GS 更新由同目录 module.py 实现。本训练器只接受首帧。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
from absl import logging
from tqdm import tqdm

from libgs.data.utils.fetch import Fetcher
from libgs.pipeline import Trainer as BaseTrainer, TrainerConfig as BaseTrainerConfig


@dataclass
class TrainerConfig(BaseTrainerConfig):
    cache_data_on_device: bool = True
    testing_steps: List[int] = field(default_factory=lambda: [5000])
    saving_ckpt_steps: List[int] = field(default_factory=lambda: [5000])
    saving_ckpt_every_n_frames: int = 1000
    num_init_steps: int = 15000
    num_incr_steps: int = 200


class Trainer(BaseTrainer):
    @property
    def num_frames(self):
        return self.datamodule.num_frames

    @property
    def current_frame(self):
        return self.datamodule.current_frame

    @property
    def is_first_frame(self):
        return self.current_frame == 0

    @property
    def is_last_frame(self):
        return self.current_frame == self.num_frames - 1

    @property
    def num_steps(self):
        return self.config.num_init_steps

    @property
    def is_first_step(self):
        return self.global_step == 1

    @property
    def is_last_step(self):
        return self.global_step == self.num_steps

    @property
    def ckpt_save_path(self) -> Path:
        return self.output_dir / f'ckpt-{self.current_frame}-{self.global_step}.pth'

    def training_loop(self):
        next_frame = self.current_frame if self.global_step > 0 else self.current_frame + 1
        for current_frame in range(next_frame, self.num_frames):
            if current_frame != 0:
                raise ValueError('First-frame trainer cannot train dynamic frames')
            logging.info('\n==> Start learning frame %d ...', current_frame)
            self.datamodule.current_frame = current_frame
            self.frame_training_loop()

    def frame_training_loop(self):
        train_dataloader = self.datamodule.train_dataloader()
        logging.info('Number of views in training dataset: %d', len(train_dataloader))
        fetcher = Fetcher(train_dataloader, self.config.cache_data_on_device,
                          lambda view: view.to(self.device, True))
        metrics_ema = {'loss': 0., 'psnr': 0.}
        progress = tqdm(range(self.global_step, self.num_steps),
                        desc=f'Training frame {self.current_frame}')
        for self.global_step in range(self.global_step + 1, self.num_steps + 1):
            self.module.pre_training_step()
            self.module.timer.clock('data').start()
            view = fetcher.next()
            self.module.timer.clock('data').stop()
            loss, metrics, package = self.module.training_step(view, self.global_step)
            self.module.timer.clock('backward').start()
            loss.backward()
            self.module.timer.clock('backward').stop()
            self.module.timer.clock('progress updating').start()
            for key, value in (('loss', loss), ('psnr', metrics['psnr'])):
                metrics_ema[key] = .6 * metrics_ema[key] + .4 * value.item()
            if self.global_step % 10 == 0:
                displayed = {key: f'{value:.4f}' for key, value in metrics_ema.items()}
                displayed['gs'] = metrics['gs']
                progress.set_postfix(displayed)
                progress.update(10)
            if self.global_step == self.num_steps:
                progress.close()
            self.module.timer.clock('progress updating').stop()
            self.module.post_training_step(package)
            if self.global_step in self.config.saving_ckpt_steps + [self.num_steps]:
                self.save_checkpoint()
            if self.global_step in self.config.testing_steps + [self.num_steps]:
                self.validation_loop()
        self.global_step = 0

    @torch.no_grad()
    def validation_loop(self):
        torch.cuda.empty_cache()
        loaders = self.datamodule.val_dataloader()
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        results = []
        for loader_index, loader in enumerate(loaders):
            logging.info('Number of views in testing dataset %d: %d', loader_index, len(loader))
            results.append([self.module.validation_step(view.to(self.device), index, loader_index)
                            for index, view in enumerate(loader)])
        self.module.validation_end(results[0] if len(loaders) == 1 else results, len(loaders))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def save_checkpoint(self):
        interval = self.config.saving_ckpt_every_n_frames
        if self.is_last_frame or self.current_frame % interval == 0:
            super().save_checkpoint()
