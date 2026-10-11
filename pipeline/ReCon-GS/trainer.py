"""Frame trainer for the single rigid residual tree production method."""

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from absl import logging

from libgs.data.utils.fetch import Fetcher
from libgs.pipeline import Trainer as BaseTrainer
from libgs.pipeline import TrainerConfig as BaseTrainerConfig

from .data import PairedFetcher
from .model.birth_quantization import RAW_FIELDS, model_raw
from .model.rigid_residual_tree import initial_layout


@dataclass
class TrainerConfig(BaseTrainerConfig):
    cache_data_on_device: bool = True
    num_init_steps: int = 15000
    saving_ckpt_steps: List[int] = field(default_factory=lambda: [100, 200])
    saving_ckpt_frames: Optional[List[int]] = field(default_factory=lambda: [
        1, 29, 49, 63, 99, 149, 199, 249, 299])
    saving_ckpt_every_n_frames: int = 1000


class Trainer(BaseTrainer):
    def setup(self, module, ckpt_path=None):
        self.module, module.trainer = module, self
        self.init_logger()
        self.datamodule.setup(self.output_dir)
        self.module.to(self.device).setup()
        self._resume_rng = None
        if ckpt_path is None:
            raise ValueError('The active method requires a shared completed first-frame checkpoint')
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.global_step = int(ckpt['global_step'])
        self.datamodule.on_load_checkpoint(ckpt)
        if ckpt.get('rigid_tree_version') is not None:
            if (ckpt.get('rigid_tree_data_config') != asdict(self.datamodule.config) or
                    ckpt.get('rigid_tree_trainer_config') != asdict(self.config)):
                raise ValueError('Rigid checkpoint needs matching data and trainer settings')
        self.module.on_load_checkpoint(ckpt)
        self._resume_rng = ckpt.get('rng_state')
        if ckpt.get('rigid_tree_version') is None:
            self._write_reference(frame_index=0)
        else:
            existing = self.output_dir / 'frame_storage' / 'rigid_tree' / 'reference_frame0.npz'
            if existing.exists():
                self.decoder_reference_path = existing
            else:
                state = self.module.motion.codec_previous_state
                if state is None:
                    raise ValueError('Rigid checkpoint has no decoded bootstrap state')
                self._write_reference(frame_index=state.frame_index, state=state)
        logging.info('Loaded frame %d step %d from %s',
                     self.current_frame, self.global_step, ckpt_path)

    def _write_reference(self, frame_index, state=None):
        """Lossless decoder bootstrap, including layout and static GS fields."""
        directory = self.output_dir / 'frame_storage' / 'rigid_tree'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'reference_frame{frame_index}.npz'
        raw = model_raw(self.module.gaussians)
        if state is None:
            owner, parents = initial_layout(raw['xyz'],
                self.module.config.rigid_tree.region_count)
        else:
            if len(raw['xyz']) != len(state.xyz):
                raise ValueError('Checkpoint GS count differs from decoder bootstrap')
            raw['xyz'] = state.xyz
            raw['rotation'] = state.rotation
            owner, parents = state.owner, state.parents
        fields = {name: raw[name].detach().cpu().numpy() for name in RAW_FIELDS}
        fields.update(owner=owner.detach().cpu().numpy(),
                      parents=parents.detach().cpu().numpy(),
                      frame_index=np.array(frame_index, dtype=np.int32),
                      extent=np.array(self.datamodule.cameras_extent, dtype=np.float64),
                      region_count=np.array(self.module.config.rigid_tree.region_count,
                                            dtype=np.int32))
        initial = self.module.initial_reference
        if initial is None:
            raise ValueError('Decoder bootstrap lost initial reference metadata')
        fields.update(
            initial_original_count=np.array(initial['original_count'], dtype=np.int64),
            initial_retained_count=np.array(initial['retained_count'], dtype=np.int64),
            initial_removed_count=np.array(initial['removed_count'], dtype=np.int64),
            initial_opacity_threshold=np.array(initial['threshold'], dtype=np.float64))
        np.savez_compressed(path, **fields)
        self.decoder_reference_path = path
        logging.info('Decoder reference: %s (%d bytes); native frame0 GS %d -> %d '
                     'at opacity threshold %.6g', path, path.stat().st_size,
                     initial['original_count'], initial['retained_count'],
                     initial['threshold'])

    @property
    def current_frame(self):
        return self.datamodule.current_frame

    @property
    def is_first_frame(self):
        return self.current_frame == 0

    @property
    def num_steps(self):
        return (self.config.num_init_steps if self.is_first_frame
                else self.module.total_steps)

    @property
    def ckpt_save_path(self):
        return self.output_dir / f'rigid-ckpt-{self.current_frame}-{self.global_step}.pth'

    def _capture_rng(self):
        return dict(torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    python=random.getstate(), numpy=np.random.get_state())

    @staticmethod
    def _restore_rng(state):
        if state is None:
            return
        torch.set_rng_state(state['torch'].cpu())
        if state['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])

    @torch.no_grad()
    def save_checkpoint(self):
        ckpt = dict(global_step=self.global_step,
                    rng_state=self._capture_rng(),
                    rigid_tree_data_config=asdict(self.datamodule.config),
                    rigid_tree_trainer_config=asdict(self.config))
        self.module.on_save_checkpoint(ckpt)
        self.datamodule.on_save_checkpoint(ckpt)
        torch.save(ckpt, self.ckpt_save_path)

    def _checkpoint_due(self):
        frames = self.config.saving_ckpt_frames
        frame_due = (self.current_frame in frames if frames is not None else
                     self.current_frame % self.config.saving_ckpt_every_n_frames == 0)
        return frame_due and self.global_step in self.config.saving_ckpt_steps

    def _make_fetcher(self):
        loader = self.datamodule.train_dataloader()
        move = lambda view: view.to(self.device, True)
        seed = self.datamodule.config.training_view_seed
        if seed is None:
            return Fetcher(loader, self.config.cache_data_on_device, move)
        return PairedFetcher(loader, self.config.cache_data_on_device,
                             move, seed + self.current_frame)

    def _resume_fetcher(self, fetcher):
        if self.global_step:
            for _ in range(self.global_step):
                fetcher.next()
            # Rebuilding a DataLoader and replaying views can draw host RNG.
            # The optimizer and cloning resume from the saved post-step state.
            self._restore_rng(self._resume_rng)
            self._resume_rng = None

    def training_loop(self):
        if self.is_first_frame:
            if self.global_step != self.config.num_init_steps:
                raise ValueError('The shared first frame must already be fully trained')
            self.global_step = 0
            start_frame = 1
        else:
            start_frame = self.current_frame
        if start_frame >= self.datamodule.num_frames:
            return []
        reports = []
        for frame in range(start_frame, self.datamodule.num_frames):
            self.datamodule.current_frame = frame
            if self.global_step >= self.num_steps:
                logging.info('Frame %d is complete in checkpoint; continuing', frame)
                self._restore_rng(self._resume_rng)
                self._resume_rng = None
                self.module.begin_next_frame()
                self.global_step = 0
                continue
            reports.append(self.frame_training_loop())
            self.module.begin_next_frame()
            self.global_step = 0
        return reports

    def frame_training_loop(self):
        frame = self.current_frame
        started = time.perf_counter()
        fetcher = self._make_fetcher()
        self._resume_fetcher(fetcher)
        executed_steps = self.num_steps - self.global_step
        train_loss = 0.0
        last_metrics = {}
        for step in range(self.global_step + 1, self.num_steps + 1):
            self.global_step = step
            self.module.pre_training_step()
            viewpoint = fetcher.next()
            loss, metrics, package = self.module.training_step(viewpoint, step)
            if loss.requires_grad:
                loss.backward()
            self.module.post_training_step(package)
            train_loss += float(loss.detach())
            last_metrics = metrics
            if step % 20 == 0 or step == self.num_steps:
                logging.info('Frame %d step %d/%d image=%.5f PSNR=%.2f GS=%d',
                             frame, step, self.num_steps,
                             metrics['image_loss'], metrics['psnr'], metrics['gs'])
            if self._checkpoint_due() and step != self.num_steps:
                self.save_checkpoint()
        evaluation = self.validation_loop()
        report = dict(self.module.frame_report or {},
            evaluation=evaluation, final_train_metrics=last_metrics,
            mean_train_loss=train_loss / max(1, executed_steps),
            train_loss_steps=executed_steps,
            elapsed_seconds=time.perf_counter() - started,
            checkpoint=str(self.ckpt_save_path) if self._checkpoint_due() else None)
        directory = self.output_dir / 'frame_summary'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'frame_{frame:03d}.json').write_text(
            json.dumps(report, indent=2, default=str))
        if self._checkpoint_due():
            self.save_checkpoint()
        logging.info('Frame %d test=%s encoded=%d bytes births=%d edits=%d',
                     frame, evaluation.get('test'), report.get('encoded_bytes', 0),
                     report.get('birth_count', 0),
                     report.get('birth_control', {}).get('tree_edit_count', 0))
        return report

    @torch.no_grad()
    def validation_loop(self):
        loaders = self.datamodule.val_dataloader()
        if not isinstance(loaders, (tuple, list)):
            loaders = [loaders]
        groups = []
        for loader_idx, loader in enumerate(loaders):
            rows = []
            for index, viewpoint in enumerate(loader):
                rows.append(self.module.validation_step(
                    viewpoint.to(self.device), index, loader_idx))
            groups.append(rows)
        return self.module.validation_end(
            groups[0] if len(groups) == 1 else groups,
            num_loaders=len(groups))
