"""Frame trainer for spatial gates, GS motion residuals, and quantized births."""

import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from absl import logging
from tqdm import tqdm

from libgs.data.utils.fetch import Fetcher
from libgs.pipeline import Trainer as BaseTrainer
from libgs.pipeline import TrainerConfig as BaseTrainerConfig

from .data import PairedFetcher
from .model.birth_quantization import RAW_FIELDS, model_raw
from libgs.metric import psnr


@dataclass
class TrainerConfig(BaseTrainerConfig):
    cache_data_on_device: bool = True
    num_init_steps: int = 15000
    saving_ckpt_steps: List[int] = field(default_factory=lambda: [100, 200])
    saving_ckpt_frames: Optional[List[int]] = field(default_factory=lambda: [
        1, 29, 49, 63, 99, 149, 199, 249, 299])
    saving_ckpt_every_n_frames: int = 1000
    stage_image_camera: Optional[str] = None
    motion_views_per_step: int = 1


class Trainer(BaseTrainer):
    def setup(self, module, ckpt_path=None):
        if (isinstance(self.config.motion_views_per_step, bool) or
                not isinstance(self.config.motion_views_per_step, int) or
                self.config.motion_views_per_step < 1):
            raise ValueError('motion_views_per_step must be a positive integer')
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
        extension_plan = None
        prefix = getattr(self, 'resume_prefix_dir', None)
        if prefix is not None:
            from .sequence_extension import validate_sequence_extension
            from libgs.data.types import getNerfppNorm
            if Path(ckpt_path).resolve().parent != Path(prefix).resolve():
                raise ValueError('Extension checkpoint must belong to its source prefix run')
            items = list(self.datamodule.scene.train_dataset.items)
            if self.datamodule.scene.test_dataset is not None:
                items += list(self.datamodule.scene.test_dataset.items)
            first = [item for item in self.datamodule.scene.train_dataset.items if item.frame == 0]
            if not first or getNerfppNorm(first)['radius'] != self.datamodule.cameras_extent:
                raise ValueError('Extension scene extent must already be determined by frame0 cameras')
            extension_plan = validate_sequence_extension(
                checkpoint=ckpt, current_data_config=asdict(self.datamodule.config),
                current_trainer_config=asdict(self.config), source_run=prefix,
                destination_run=self.output_dir, current_extent=self.datamodule.cameras_extent,
                current_images=[(item.frame, item.path) for item in items],
                completed_prefix_frames=getattr(self, 'resume_completed_prefix_frames', None))
        if ckpt.get('rigid_tree_version') is not None:
            saved_trainer = ckpt.get('rigid_tree_trainer_config')
            if isinstance(saved_trainer, dict) and 'stage_image_camera' not in saved_trainer:
                saved_trainer = {**saved_trainer, 'stage_image_camera': None}
            if isinstance(saved_trainer, dict):
                saved_trainer = {'motion_views_per_step': 1, **saved_trainer}
            if extension_plan is None and (
                    ckpt.get('rigid_tree_data_config') != asdict(self.datamodule.config) or
                    saved_trainer != asdict(self.config)):
                raise ValueError('Rigid checkpoint needs matching data and trainer settings')
        self.module.on_load_checkpoint(ckpt)
        if extension_plan is not None:
            from .sequence_extension import copy_verified_prefix
            provenance = copy_verified_prefix(extension_plan)
            with (self.output_dir / 'prefix_provenance.json').open('x') as file:
                json.dump(provenance, file, indent=2)
            logging.info('Extending completed prefix 0–%d to %d frames; past packets copied unchanged',
                         extension_plan.resume_frame, extension_plan.extended_frames)
        self._resume_rng = ckpt.get('rng_state')
        if ckpt.get('rigid_tree_version') is None:
            self._write_reference(frame_index=0)
            logging.info('Spatial gates: eight initial cells, candidate depth=%d, '
                         '%d motion + %d birth steps',
                         self.module.config.rigid_tree.spatial_gate.max_depth,
                         self.module.config.rigid_tree.motion_steps,
                         self.module.config.rigid_tree.birth_steps)
        else:
            existing = self.output_dir / 'frame_storage' / 'rigid_tree' / 'reference_frame0.npz'
            if existing.exists():
                self.decoder_reference_path = existing
            else:
                state = self.module.motion.codec_previous_state
                if state is None:
                    raise ValueError('Spatial checkpoint has no decoded bootstrap state')
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
            state = self.module.motion.initialize_reference(raw['xyz'], raw['rotation'],
                self.datamodule.cameras_extent, frame_index)
        if len(raw['xyz']) != len(state.xyz):
            raise ValueError('Checkpoint GS count differs from decoder bootstrap')
        raw['xyz'], raw['rotation'] = state.xyz, state.rotation
        owner, parents = state.owner, state.parents
        fields = {name: raw[name].detach().cpu().numpy() for name in RAW_FIELDS}
        fields.update(owner=owner.detach().cpu().numpy(),
                      parents=parents.detach().cpu().numpy(),
                      frame_index=np.array(frame_index, dtype=np.int32),
                      extent=np.array(self.datamodule.cameras_extent, dtype=np.float64),
                      region_count=np.array(len(parents),
                                            dtype=np.int32))
        if state is not None and state.spatial_paths is not None:
            fields.update(spatial_paths=state.spatial_paths.cpu().numpy(),
                          spatial_open=state.spatial_open.cpu().numpy())
            if state.spatial_centers is not None:
                fields['spatial_centers'] = state.spatial_centers.cpu().numpy()
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
        with tqdm.external_write_mode(file=sys.stderr):
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
            for step in range(1, self.global_step + 1):
                for _ in range(self._views_per_step(step)):
                    fetcher.next()
            # Rebuilding a DataLoader and replaying views can draw host RNG.
            # The optimizer and cloning resume from the saved post-step state.
            self._restore_rng(self._resume_rng)
            self._resume_rng = None

    def _views_per_step(self, step):
        return (self.config.motion_views_per_step
                if step <= self.module.motion_steps else 1)

    def _fit_views(self, fetcher, step):
        """Average image objectives before one shared-motion optimizer update."""
        count = self._views_per_step(step)
        packages, metrics, loss_value = [], {}, 0.
        for _ in range(count):
            loss, values, package = self.module.training_step(fetcher.next(), step)
            if loss.requires_grad:
                (loss / count).backward()
            loss_value += float(loss.detach()) / count
            for name, value in values.items():
                metrics[name] = metrics.get(name, 0.) + value / count
            if count > 1:
                # Keep only evidence needed after backward, not full images.
                package = {name: package[name] for name in
                    ('visibility_filter', 'motion_contribution_filter', 'viewspace_points')
                    if name in package}
            packages.append(package)
        metrics['gs'] = values['gs']
        self.module.post_training_step(packages[0] if count == 1 else packages)
        return loss_value, metrics

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
        metrics_ema = dict(image=0., psnr=0.)
        stage_view = None

        with tqdm(total=self.num_steps, initial=self.global_step,
                  desc=f'Training frame {frame}', dynamic_ncols=True,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| '
                             '{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as progress_bar:
            for step in range(self.global_step + 1, self.num_steps + 1):
                self.global_step = step
                self.module.pre_training_step()
                loss_value, metrics = self._fit_views(fetcher, step)
                if (self.config.stage_image_camera is not None and
                        step in (self.module.motion_steps, self.module.total_steps)):
                    stage_view = self._record_stage_image(step, stage_view)
                train_loss += loss_value
                last_metrics = metrics
                for name, value in (('image', metrics['image_loss']), ('psnr', metrics['psnr'])):
                    metrics_ema[name] = .6 * metrics_ema[name] + .4 * value
                if step % 10 == 0 or step == self.num_steps:
                    progress_bar.set_postfix(dict(image=f"{metrics_ema['image']:.5f}",
                        psnr=f"{metrics_ema['psnr']:.2f}", gs=metrics['gs']), refresh=False)
                    progress_bar.update(step - progress_bar.n)
                if self._checkpoint_due() and step != self.num_steps:
                    self.save_checkpoint()
        evaluation = self.validation_loop()
        report = dict(self.module.frame_report or {},
            stage_image_metrics=self.module.stage_image_metrics,
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
        quality = ' | '.join(
            f'{name} PSNR={row.get("psnr", float("nan")):.4f} '
            f'SSIM={row.get("ssim", float("nan")):.4f}'
            + (f' LPIPS={row["lpips"]:.4f}' if 'lpips' in row else '')
            for name, row in evaluation.items())
        logging.info('Frame %d | %s | %d steps | %.2fs', frame,
                     quality or 'no evaluation', executed_steps, report['elapsed_seconds'])
        logging.info('  Storage=%.2f KB | GS=%d (+%d) | regions=%d | tree edits=%d',
                     report.get('encoded_bytes', 0) / 1000,
                     report.get('total_count', last_metrics.get('gs', 0)),
                     report.get('birth_count', 0), report.get('region_count', 0),
                     report.get('birth_control', {}).get('tree_edit_count', 0))
        return report

    @torch.no_grad()
    def _record_stage_image(self, step, viewpoint=None):
        """One fixed heldout view; no gradients, selection, or RNG side effects."""
        rng = self._capture_rng()
        try:
            if viewpoint is None:
                loader = self.datamodule.test_dataloader()
                if loader is None:
                    raise ValueError('Stage image diagnostics require a heldout test dataset')
                dataset = loader.dataset
                indices = [index for index, item in enumerate(dataset.items)
                           if Path(str(item.path)).name == self.config.stage_image_camera]
                if len(indices) != 1:
                    raise ValueError('Stage image camera must identify exactly one heldout view')
                iterator = iter(loader)
                try:
                    for candidate in iterator:
                        if Path(str(candidate.path)).name == self.config.stage_image_camera:
                            viewpoint = candidate.to(self.device)
                            break
                    if viewpoint is None:
                        raise ValueError('Stage image camera was not returned by the evaluation loader')
                finally:
                    shutdown = getattr(iterator, '_shutdown_workers', None)
                    if shutdown is not None:
                        shutdown()
            old_only = step == self.module.motion_steps
            rendered = self.module.forward(viewpoint, old_only=old_only)['render'].clamp(0, 1)
            difference = rendered - viewpoint.image
            labels = {self.module.motion_steps: 'hard_spatial_motion_old_only',
                      self.module.total_steps: 'final_decoded_with_birth'}
            record = dict(step=int(step), phase=labels[step], camera=str(viewpoint.path),
                          mse=float(difference.square().mean()),
                          psnr=float(psnr(rendered, viewpoint.image)), old_only=old_only,
                          shared_prestructure_active=bool(getattr(getattr(self.module, 'motion', None),
                                                                 'shared_prestructure_active', False)),
                          participates_in_training_or_selection=False)
            records = self.module.stage_image_metrics
            records[:] = [item for item in records
                          if (item['step'], item['phase']) != (step, record['phase'])]
            records.append(record)
            logging.info('Stage heldout: frame=%d step=%d %s PSNR=%.6f MSE=%.9g',
                         self.current_frame, step, record['phase'], record['psnr'], record['mse'])
            return viewpoint
        finally:
            self._restore_rng(rng)

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
