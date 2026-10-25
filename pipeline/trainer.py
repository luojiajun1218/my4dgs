"""Frame trainer for spatial gates, GS motion residuals, and quantized births."""

import json
import hashlib
import random
import sys
import time
from contextlib import nullcontext
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
from .module import restore_retired_reference
from .timing import FrameTimer
from .run_summary import build_run_summary, format_run_summary, time_render
from libgs.metric import psnr


def frame_training_phase(module, step):
    if step <= module.motion_steps:
        return 'motion'
    return 'birth' if step <= module.birth_commit_step else 'refit'


def stage_image_steps(module):
    birth = getattr(getattr(module, 'config', None), 'birth', None)
    interval = getattr(birth, 'densify_interval', 30)
    stats_steps = getattr(birth, 'densification_stats_steps', 31)
    steps = {module.motion_steps - stats_steps + 1, module.motion_steps,
             *range(module.motion_steps + interval, module.birth_commit_step, interval),
             module.birth_commit_step, module.total_steps}
    return tuple(sorted(steps))


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
    birth_views_per_step: int = 1


class Trainer(BaseTrainer):
    def fit(self, module, ckpt_path=None):
        started = time.perf_counter()
        self.setup(module, ckpt_path)
        initial_report = None
        if self.is_first_frame:
            # This extra evaluation must not consume the training RNG stream.
            rng = self._capture_rng()
            try:
                evaluation = self.validation_loop()
                initial_report = dict(frame=0, evaluation=evaluation,
                    total_count=len(self.module.gaussians._xyz),
                    render_timing=self._validation_render_timing,
                    reference_sha256=hashlib.sha256(
                        self.decoder_reference_path.read_bytes()).hexdigest())
                (self.output_dir / 'reference_evaluation.json').write_text(
                    json.dumps(initial_report, indent=2))
            finally:
                self._restore_rng(rng)
        elif getattr(self, 'resume_prefix_dir', None) is not None:
            source = Path(self.resume_prefix_dir) / 'reference_evaluation.json'
            if source.is_file() and self.decoder_reference_path.name == 'reference_frame0.npz':
                candidate = json.loads(source.read_text())
                if candidate.get('reference_sha256') == hashlib.sha256(
                        self.decoder_reference_path.read_bytes()).hexdigest():
                    initial_report = candidate
                    (self.output_dir / 'reference_evaluation.json').write_text(
                        json.dumps(initial_report, indent=2))
        reports = self.training_loop()
        summary = build_run_summary(self.output_dir, self.datamodule.num_frames,
            initial_report=initial_report, reference_path=self.decoder_reference_path,
            elapsed_seconds=time.perf_counter() - started)
        path = self.output_dir / 'run_summary.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(summary, indent=2))
        temporary.replace(path)
        with tqdm.external_write_mode(file=sys.stderr):
            logging.info('\n%s', '\n'.join(format_run_summary(summary)))
        return reports

    def setup(self, module, ckpt_path=None):
        if (isinstance(self.config.motion_views_per_step, bool) or
                not isinstance(self.config.motion_views_per_step, int) or
                self.config.motion_views_per_step < 1):
            raise ValueError('motion_views_per_step must be a positive integer')
        if (isinstance(self.config.birth_views_per_step, bool) or
                not isinstance(self.config.birth_views_per_step, int) or
                self.config.birth_views_per_step < 1):
            raise ValueError('birth_views_per_step must be a positive integer')
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
                saved_trainer = {'motion_views_per_step': 1,
                                 'birth_views_per_step': 4, **saved_trainer}
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
            birth_cfg = self.module.config.birth
            dense_steps = list(range(self.module.motion_steps + birth_cfg.densify_interval,
                                     self.module.birth_commit_step,
                                     birth_cfg.densify_interval))
            logging.info('Spatial gates: candidate depth=%d; schedule: '
                         '%d motion, iterative birth through %d, fixed-set refit %d steps; '
                         'birth stats %d–%d, densify at %s',
                         self.module.config.rigid_tree.spatial_gate.max_depth,
                         self.module.motion_steps, self.module.birth_commit_step,
                         birth_cfg.refit_steps, self.module.motion_steps -
                         birth_cfg.densification_stats_steps + 1,
                         self.module.motion_steps, dense_steps)
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
        elif (not getattr(self.module, 'frame_finalized', False) and
              getattr(self.module, 'retired_reference', None) is not None):
            raw = restore_retired_reference(raw, state,
                self.module.motion.previous_rows, self.module.retired_reference)
        if len(raw['xyz']) != len(state.xyz):
            raise ValueError('Checkpoint GS count differs from decoder bootstrap')
        opacity_update = getattr(self.module, 'opacity_update', None)
        if (not getattr(self.module, 'frame_finalized', False) and opacity_update is not None and
                opacity_update.reference is not None):
            if len(opacity_update.reference) != len(state.xyz):
                raise ValueError('Opacity checkpoint lost preceding reference logits')
            raw['opacity'] = opacity_update.reference.detach().cpu()
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
            from .model.spatial_gate import SPATIAL_ROUTING_FIELDS
            fields.update({name: getattr(state, name).cpu().numpy()
                           for name in SPATIAL_ROUTING_FIELDS
                           if getattr(state, name) is not None})
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
                    Trainer._next_training_view(self, fetcher, step)
            # Rebuilding a DataLoader and replaying views can draw host RNG.
            # The optimizer and cloning resume from the saved post-step state.
            self._restore_rng(self._resume_rng)
            self._resume_rng = None

    def _views_per_step(self, step):
        return (self.config.motion_views_per_step if step <= self.module.motion_steps
                else self.config.birth_views_per_step)

    def _next_training_view(self, fetcher, step):
        return fetcher.next()

    def _fit_views(self, fetcher, step, *, defer_metrics=False):
        """Average image objectives before one shared-motion optimizer update."""
        count = self._views_per_step(step)
        packages, metrics, loss_value = [], {}, 0.
        timer = getattr(self, '_frame_timer', None)
        for index in range(count):
            # The regularizer does not depend on the camera. Charging it on
            # the last view with weight=count preserves the averaged objective
            # and avoids rebuilding the same graph once per camera.
            with timer.phase('data_loading') if timer else nullcontext():
                viewpoint = Trainer._next_training_view(self, fetcher, step)
            loss, values, package = self.module.training_step(viewpoint, step,
                regularization_weight=count if index == count - 1 else 0.)
            if loss.requires_grad:
                (loss / count).backward()
            loss_value = loss_value + loss.detach() / count
            for name, value in values.items():
                metrics[name] = metrics.get(name, 0.) + value / count
            if count > 1:
                # Keep only evidence needed after backward, not full images.
                package = {name: package[name] for name in
                    ('visibility_filter', 'motion_contribution_filter', 'viewspace_points', 'radii')
                    if name in package}
            packages.append(package)
        metrics['gs'] = values['gs']
        self.module.post_training_step(packages[0] if count == 1 else packages)
        model = getattr(self.module, 'gaussians', None)
        if model is not None and hasattr(model, '_xyz'):
            births = getattr(self.module, 'births', None)
            metrics['gs'] = len(model._xyz) + (
                len(births._xyz) if getattr(self.module, 'motion_committed', False)
                and births is not None else 0)
        if defer_metrics:
            return loss_value, metrics
        return Trainer._materialize_step_metrics([(loss_value, metrics)])[0]

    @staticmethod
    def _materialize_step_metrics(rows):
        """Transfer detached diagnostics together without changing Python sums."""
        names, tensors = [], []
        for loss, metrics in rows:
            keys = [name for name, value in metrics.items() if torch.is_tensor(value)]
            names.append(keys)
            tensors.append(loss)
            tensors.extend(metrics[name] for name in keys)
        if not tensors:
            return []
        scalars = iter(torch.stack(tensors).tolist())
        materialized = []
        for (_, metrics), keys in zip(rows, names):
            loss = next(scalars)
            metrics = dict(metrics)
            metrics.update((name, next(scalars)) for name in keys)
            materialized.append((loss, metrics))
        return materialized

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
        device = getattr(self, 'device', None)
        synchronize = (lambda: torch.cuda.synchronize(device)) if (
            device is not None and torch.device(device).type == 'cuda') else None
        timer = FrameTimer(synchronize=synchronize)
        self._frame_timer = timer
        started = time.perf_counter()
        fetcher = self._make_fetcher()
        self._resume_fetcher(fetcher)
        executed_steps = self.num_steps - self.global_step
        train_loss = 0.0
        last_metrics = {}
        metrics_ema = dict(image=0., psnr=0.)
        pending_metrics = []
        stage_view = None

        with tqdm(total=self.num_steps, initial=self.global_step,
                  desc=f'Training frame {frame}', dynamic_ncols=True,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| '
                             '{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as progress_bar:
            for step in range(self.global_step + 1, self.num_steps + 1):
                phase = frame_training_phase(self.module, step)
                timer.switch(phase)
                self.global_step = step
                with timer.phase('model_setup') if step == 1 else nullcontext():
                    self.module.pre_training_step()
                pending_metrics.append(self._fit_views(fetcher, step, defer_metrics=True))
                if (self.config.stage_image_camera is not None and
                        step in stage_image_steps(self.module)):
                    with timer.phase('stage_evaluation'):
                        stage_view = self._record_stage_image(step, stage_view)
                if step % 10 == 0 or step == self.num_steps:
                    # The progress display only refreshes every ten steps. Keep
                    # detached scalars on device until then instead of forcing
                    # a CUDA synchronization after every optimizer update.
                    with timer.phase('diagnostics'):
                        for loss_value, metrics in self._materialize_step_metrics(pending_metrics):
                            train_loss += loss_value
                            last_metrics = metrics
                            for name, value in (('image', metrics['image_loss']),
                                                ('psnr', metrics['psnr'])):
                                metrics_ema[name] = .6 * metrics_ema[name] + .4 * value
                        pending_metrics.clear()
                        progress_bar.set_postfix(dict(phase=phase, image=f"{metrics_ema['image']:.5f}",
                            psnr=f"{metrics_ema['psnr']:.2f}", gs=metrics['gs']), refresh=False)
                        progress_bar.update(step - progress_bar.n)
                if self._checkpoint_due() and step != self.num_steps:
                    with timer.phase('checkpoint'):
                        self.save_checkpoint()
        timer.switch('validation')
        evaluation = self.validation_loop()
        timer.switch('bookkeeping')
        report = dict(self.module.frame_report or {},
            stage_image_metrics=self.module.stage_image_metrics,
            evaluation=evaluation, final_train_metrics=last_metrics,
            render_timing=getattr(self, '_validation_render_timing', None),
            mean_train_loss=train_loss / max(1, executed_steps),
            train_loss_steps=executed_steps,
            elapsed_seconds=time.perf_counter() - started,
            checkpoint=str(self.ckpt_save_path) if self._checkpoint_due() else None)
        if self._checkpoint_due():
            with timer.phase('checkpoint'):
                self.save_checkpoint()
        timing = timer.finish()
        self._frame_timer = None
        report.update(timing=timing, **{name: timing[name] for name in (
            'training_seconds', 'optimization_seconds', 'evaluation_seconds',
            'checkpoint_seconds', 'frame_wall_seconds', 'preparation_seconds',
            'data_loading_seconds', 'encoding_seconds', 'diagnostics_seconds',
            'training_loop_seconds')})
        directory = self.output_dir / 'frame_summary'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'frame_{frame:03d}.json').write_text(
            json.dumps(report, indent=2, default=str))
        quality = ' | '.join(
            f'{name} PSNR={row.get("psnr", float("nan")):.4f} '
            f'SSIM={row.get("ssim", float("nan")):.4f}'
            + (f' LPIPS={row["lpips"]:.4f}' if 'lpips' in row else '')
            for name, row in evaluation.items())
        logging.info('Frame %d | %s | %d steps | %.2fs', frame,
                     quality or 'no evaluation', executed_steps, report['elapsed_seconds'])
        logging.info('  Time: training=%.3fs evaluation=%.3fs checkpoint=%.3fs wall=%.3fs',
                     report['training_seconds'], report['evaluation_seconds'],
                     report['checkpoint_seconds'], report['frame_wall_seconds'])
        logging.info('  Storage=%.2f KB | GS=%d (new=%d, deleted=%d, net=%+d) | regions=%d | tree edits=%d',
                     report.get('encoded_bytes', 0) / 1000,
                     report.get('total_count', last_metrics.get('gs', 0)),
                     report.get('birth_count', 0), report.get('retired_count', 0),
                     report.get('birth_count', 0) - report.get('retired_count', 0),
                     report.get('region_count', 0),
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
            labels = {
                stage_image_steps(self.module)[0]: 'motion_gradient_window_start',
                self.module.motion_steps: 'first_birth_pool',
                self.module.birth_commit_step: 'final_birth_pool_fixed',
                self.module.total_steps: 'final_decoded_with_birth',
            }
            interval = getattr(getattr(getattr(self.module, 'config', None), 'birth', None),
                               'densify_interval', 30)
            for index, boundary in enumerate(range(
                    self.module.motion_steps + interval,
                    self.module.birth_commit_step, interval), 1):
                labels[boundary] = f'birth_split_prune_{index}'
            record = dict(step=int(step), phase=labels[step], camera=str(viewpoint.path),
                          mse=float(difference.square().mean()),
                          psnr=float(psnr(rendered, viewpoint.image)), old_only=old_only,
                          gaussian_count=len(self.module.gaussians._xyz) + (
                              len(self.module.births._xyz) if not old_only else 0),
                          shared_prestructure_active=bool(getattr(getattr(self.module, 'motion', None),
                                                                 'shared_prestructure_active', False)),
                          participates_in_training_or_selection=False)
            records = self.module.stage_image_metrics
            records[:] = [item for item in records
                          if (item['step'], item['phase']) != (step, record['phase'])]
            records.append(record)
            logging.debug('Stage heldout: frame=%d step=%d %s PSNR=%.6f MSE=%.9g',
                          self.current_frame, step, record['phase'], record['psnr'], record['mse'])
            return viewpoint
        finally:
            self._restore_rng(rng)

    @torch.no_grad()
    def _measure_render(self, viewpoint):
        rng = self._capture_rng()
        try:
            device = torch.device(self.device)
            synchronize = ((lambda: torch.cuda.synchronize(device))
                           if device.type == 'cuda' else (lambda: None))
            result = time_render(
                lambda: self.module.forward(viewpoint)['render'].clamp(0, 1), synchronize)
            return dict(result, frame=self.current_frame, camera=Path(viewpoint.path).name,
                        image_size=list(reversed(viewpoint.image.shape[-2:])),
                        device=str(device))
        finally:
            self._restore_rng(rng)

    @torch.no_grad()
    def validation_loop(self):
        loaders = self.datamodule.val_dataloader()
        if not isinstance(loaders, (tuple, list)):
            loaders = [loaders]
        self._validation_render_timing = None
        preferred = next((name for name in ('test', 'validation', 'train')
                          if name in self.datamodule.eval_names), None)
        selected_camera = self.config.stage_image_camera
        groups = []
        for loader_idx, loader in enumerate(loaders):
            rows = []
            for index, viewpoint in enumerate(loader):
                viewpoint = viewpoint.to(self.device)
                rows.append(self.module.validation_step(
                    viewpoint, index, loader_idx))
                if (self._validation_render_timing is None and
                        self.datamodule.eval_names[loader_idx] == preferred and
                        (selected_camera is None or Path(viewpoint.path).name == selected_camera)):
                    self._validation_render_timing = self._measure_render(viewpoint)
            groups.append(rows)
        return self.module.validation_end(
            groups[0] if len(groups) == 1 else groups,
            num_loaders=len(groups))
