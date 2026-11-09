"""Frame trainer for spatial gates, GS motion residuals, and quantized births."""

import csv
import json
import math
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
from .frame_plan import frame_plan_phase, gradient_probe_step
from .model.birth_quantization import RAW_FIELDS, model_raw, reference_digest
from .model.residual_birth import RawView
from .module import restore_retired_reference, uses_residual_birth,uses_early_replacement
from libgs.metric import psnr


def _validate_extension_frame_coverage(items, requested):
    """Reject silently truncated training datasets before resuming a sequence."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 2:
        raise ValueError('Sequence extension needs an explicit frame count of at least two')
    frames = set()
    for item in items:
        frame = getattr(item, 'frame', None)
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise ValueError('Extension training dataset has an invalid frame index')
        frames.add(int(frame))
    expected = set(range(requested))
    if frames != expected:
        missing = sorted(expected - frames)
        extra = sorted(frames - expected)
        raise ValueError(f'Extension training dataset does not cover requested frames; '
                         f'missing={missing[:10]}, extra={extra[:10]}')


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
    # Kept for loading checkpoints from before population-window retirement.
    population_window_size: int = 0
    population_window_joint_steps: int = 10
    population_window_views_per_step: int = 1
    population_window_selection_views: int = 4

    def use_single_training_view(self):
        """Record the actual per-step view count in resolved run configs."""
        names = ('motion_views_per_step', 'birth_views_per_step')
        previous = {name: getattr(self, name) for name in names}
        for name, value in previous.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if any(value != 1 for value in previous.values()):
            logging.warning('Single-view training overrides historical view counts %s', previous)
        for name in names:
            setattr(self, name, 1)


def step_phase(step, motion_steps, gate_start, freeze, refit_steps):
    if step > motion_steps:
        return 'birth'
    if step <= gate_start:
        return 'warmup'
    if step <= freeze:
        return 'gate'
    if refit_steps and step > motion_steps - refit_steps:
        return 'refit'
    return 'joint'


def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value))


def _finite_mean(values):
    values = [float(value) for value in values if _finite_number(value)]
    return sum(values) / len(values) if values else None


def _frame_number(row):
    value = row.get('frame') if isinstance(row, dict) else None
    return int(value) if _finite_number(value) else None


def _evaluation_metrics(report):
    evaluation = report.get('evaluation') if isinstance(report, dict) else None
    if not isinstance(evaluation, dict):
        return {}
    for name in ('test', 'validation', 'train'):
        row = evaluation.get(name)
        if isinstance(row, dict):
            return row
    return next((row for row in evaluation.values() if isinstance(row, dict)), {})


def _log_value(value, pattern):
    return 'n/a' if value is None else format(value, pattern)


def _extension_end_frame(last_frame, plan, total_frames):
    if last_frame is None:
        return total_frames
    if (plan is None or isinstance(last_frame, bool) or not isinstance(last_frame, int) or
            not plan.resume_frame < last_frame < total_frames):
        raise ValueError('The last frame must follow the verified prefix and belong to the dataset')
    return last_frame + 1


class Trainer(BaseTrainer):
    def fit(self, module, ckpt_path=None):
        self.config.use_single_training_view()
        return super().fit(module, ckpt_path)

    @staticmethod
    def _assert_frame_step_budget(trainer):
        if trainer.current_frame>=1 and (trainer.num_steps>300 or trainer.global_step>300):
            raise ValueError('Hard per-frame training budget is 300 total steps including motion, birth and refit')

    def setup(self, module, ckpt_path=None):
        if (getattr(getattr(module.config, 'birth', None), 'frame_training_plan', False) and
                self.config.stage_image_camera not in (None, 'cam00.png')):
            raise ValueError('Frame-plan stage diagnostics require heldout cam00.png')
        if self.config.population_window_size:
            raise ValueError('Population-window training is retired; historical packets remain decodable')
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
        self._stage_viewpoint = None
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
            _validate_extension_frame_coverage(items,
                self.datamodule.config.extra_dataset_kwargs.get('num_frames'))
            if self.datamodule.scene.test_dataset is not None:
                items += list(self.datamodule.scene.test_dataset.items)
            first = [item for item in self.datamodule.scene.train_dataset.items if item.frame == 0]
            if not first or not math.isclose(float(getNerfppNorm(first)['radius']),
                    float(self.datamodule.cameras_extent), rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError('Extension scene extent must already be determined by frame0 cameras')
            extension_plan = validate_sequence_extension(
                checkpoint=ckpt, current_data_config=asdict(self.datamodule.config),
                current_trainer_config=asdict(self.config), source_run=prefix,
                destination_run=self.output_dir, current_extent=self.datamodule.cameras_extent,
                current_images=[(item.frame, item.path) for item in items],
                completed_prefix_frames=getattr(self, 'resume_completed_prefix_frames', None))
        if ckpt.get('rigid_tree_version') is not None:
            previous_ste = ckpt.get('gate_refine_image_ste', False)
            current_ste = bool(getattr(self.module, 'gate_refine_image_ste', False))
            fork = bool(getattr(self, 'gate_fork_from_step150', False) and self.global_step == 150 and
                        ckpt.get('rigid_tree_frame', {}).get('prepared') is True and
                        ckpt.get('rigid_tree_frame', {}).get('finalized') is False)
            if previous_ste != current_ste and not fork:
                raise ValueError('Gate STE changes require an explicitly authorized step-150 fork')
        self._verified_prefix_plan = extension_plan
        self._end_frame_exclusive = _extension_end_frame(getattr(self, 'resume_last_frame', None),
                                                          extension_plan, self.datamodule.num_frames)
        color_last = getattr(self, 'color_last_frame', None)
        if color_last is not None:
            if (extension_plan is not None or isinstance(color_last, bool) or not isinstance(color_last, int) or
                    not self.current_frame <= color_last < self.datamodule.num_frames or
                    self.global_step != 200 or not ckpt.get('rigid_tree_frame', {}).get('prepared') or
                    ckpt.get('rigid_tree_frame', {}).get('finalized')):
                raise ValueError('Color branch needs a mid-frame step-200 fork checkpoint and valid last frame')
            self._end_frame_exclusive = color_last + 1
        gate_last = getattr(self, 'gate_last_frame', None)
        if gate_last is not None:
            if (extension_plan is not None or not getattr(self, 'gate_fork_from_step150', False) or
                    isinstance(gate_last, bool) or not isinstance(gate_last, int) or
                    not self.current_frame <= gate_last < self.datamodule.num_frames or self.global_step != 150 or
                    ckpt.get('rigid_tree_frame', {}).get('finalized') is not False):
                raise ValueError('Gate fork needs an unfinalized step-150 checkpoint and a valid last frame')
            self._end_frame_exclusive = gate_last + 1
        if ckpt.get('rigid_tree_version') is not None:
            saved_trainer = ckpt.get('rigid_tree_trainer_config')
            if isinstance(saved_trainer, dict) and 'stage_image_camera' not in saved_trainer:
                saved_trainer = {**saved_trainer, 'stage_image_camera': None}
            if isinstance(saved_trainer, dict):
                saved_trainer = {'birth_views_per_step': 1, 'motion_views_per_step': 1, **saved_trainer}
                for name in ('population_window_size', 'population_window_joint_steps',
                             'population_window_views_per_step', 'population_window_selection_views'):
                    saved_trainer.setdefault(name, getattr(TrainerConfig(), name))
            if extension_plan is None and (
                    ckpt.get('rigid_tree_data_config') != asdict(self.datamodule.config) or
                    saved_trainer != asdict(self.config)):
                raise ValueError('Rigid checkpoint needs matching data and trainer settings')
        self.module.on_load_checkpoint(ckpt)
        if extension_plan is not None:
            from .sequence_extension import copy_verified_prefix
            provenance = copy_verified_prefix(extension_plan)
            if getattr(self, 'resume_last_frame', None) is not None:
                provenance['requested_last_frame'] = self.resume_last_frame
            with (self.output_dir / 'prefix_provenance.json').open('x') as file:
                json.dump(provenance, file, indent=2)
            logging.info('Extending completed prefix 0–%d to %d frames; past packets copied unchanged',
                         extension_plan.resume_frame, extension_plan.extended_frames)
        self._resume_rng = ckpt.get('rng_state')
        if ckpt.get('rigid_tree_version') is None:
            self._write_reference(frame_index=0)
            full = self.module.config.full_update
            if getattr(full, 'regional_full_200', False):
                logging.info('Spatial gates: eight initial cells, candidate depth=%d, '
                             '200 total steps; regional and per-point attributes jointly optimized at steps 101-200',
                             self.module.config.rigid_tree.spatial_gate.max_depth)
            elif getattr(full, 'dense_color_200', False):
                logging.info('Spatial gates: eight initial cells, candidate depth=%d, '
                             '200 total steps; all-point color jointly optimized at steps 101-200',
                             self.module.config.rigid_tree.spatial_gate.max_depth)
            else:
                tail = 'regional attribute repair' if full.regional_repair else 'birth'
                logging.info('Spatial gates: eight initial cells, candidate depth=%d, '
                             '%d motion + %d %s steps',
                             self.module.config.rigid_tree.spatial_gate.max_depth,
                             self.module.config.rigid_tree.motion_steps,
                             (0 if full.stop_after_motion else self.module.config.rigid_tree.birth_steps), tail)
        else:
            existing = self.output_dir / 'frame_storage' / 'rigid_tree' / 'reference_frame0.npz'
            if existing.exists():
                self.decoder_reference_path = existing
            else:
                state = self.module.motion.codec_previous_state
                if state is None:
                    raise ValueError('Spatial checkpoint has no decoded bootstrap state')
                self._write_reference(frame_index=state.frame_index, state=state)
        if getattr(self, 'color_last_frame', None) is not None:
            # The fork bootstraps from the previous decoded frame, not the
            # in-progress step-200 pose; compare all attributes bit for bit.
            saved = ckpt.get('frame_plan_diagnostic_reference')
            if saved is None:
                raise ValueError('Color fork checkpoint lost the previous decoded raw state')
            with np.load(self.decoder_reference_path, allow_pickle=False) as packet:
                bootstrap={name:torch.from_numpy(packet[name].copy()) for name in RAW_FIELDS}
            if reference_digest(bootstrap) != reference_digest(saved):
                raise ValueError('Color fork bootstrap differs from the saved decoded predecessor')
            from .model.rigid_tree_codec import DecodedState
            with np.load(self.decoder_reference_path, allow_pickle=False) as packet:
                bootstrap_state=DecodedState.from_reference(packet)
            if bootstrap_state.digest() != self.module.motion.frame_reference_state.digest():
                raise ValueError('Color fork motion reference differs from the saved predecessor')
            import hashlib
            digest=hashlib.sha256()
            with Path(ckpt_path).open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''): digest.update(block)
            selected=self.module._plan_regional_attribute_state['eligible'].detach().cpu().numpy()
            provenance=dict(checkpoint=str(Path(ckpt_path).resolve()), checkpoint_sha256=digest.hexdigest(),
                frame=int(self.current_frame),step=int(self.global_step),branch=self.module.color_drift_branch,
                prior_raw_sha256=reference_digest(bootstrap),prior_state_sha256=bootstrap_state.digest().hex(),
                selected_points=int(selected.sum()), selection_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                reference_kind='lossless diagnostic bootstrap at previous decoded frame; original prefix charged separately')
            with (self.output_dir/'color_fork_provenance.json').open('x') as file:
                json.dump(provenance,file,indent=2)
        if getattr(self, 'gate_last_frame', None) is not None:
            import hashlib
            from .model.rigid_tree_codec import DecodedState
            source_prefix=Path(ckpt_path).resolve().parent/'prefix_provenance.json'
            if not source_prefix.is_file():
                raise ValueError('Gate fork checkpoint needs its verified source-prefix provenance')
            origin=json.loads(source_prefix.read_text())
            with np.load(self.decoder_reference_path, allow_pickle=False) as packet:
                bootstrap={name:torch.from_numpy(packet[name].copy()) for name in RAW_FIELDS}
                state=DecodedState.from_reference(packet)
            if (reference_digest(bootstrap) != origin['raw_sha256'] or
                    state.digest().hex() != origin['decoded_sha256'] or
                    state.digest() != self.module.motion.frame_reference_state.digest()):
                raise ValueError('Gate fork bootstrap does not match its verified decoded predecessor')
            digest=hashlib.sha256()
            with Path(ckpt_path).open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''): digest.update(block)
            motion=self.module.motion
            state_hash=hashlib.sha256()
            for v in (motion.gate_logits,motion.opened,motion.owner,motion.region_translation,
                      motion.region_rotation,motion.gs_translation,motion.gs_rotation):
                state_hash.update(v.detach().cpu().contiguous().numpy().tobytes())
            record=dict(checkpoint=str(Path(ckpt_path).resolve()),checkpoint_sha256=digest.hexdigest(),
                frame=int(self.current_frame),step=int(self.global_step),
                gate_refine_image_ste=bool(getattr(self.module,'gate_refine_image_ste',False)),
                predecessor_raw_sha256=origin['raw_sha256'],predecessor_state_sha256=origin['decoded_sha256'],
                loaded_motion_sha256=state_hash.hexdigest(),rng_state_present=bool(ckpt.get('rng_state')),
                reference_kind='diagnostic lossless previous-frame bootstrap; original reference charged separately')
            with (self.output_dir/'gate_fork_provenance.json').open('x') as file:
                json.dump(record,file,indent=2)
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
        elif (not self.module.frame_finalized and
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
        from .model.rigid_tree_codec import compact_reference_fields, write_reference_npz
        write_reference_npz(path, compact_reference_fields(fields))
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
        return frame_due and (self.global_step in self.config.saving_ckpt_steps or
            (uses_residual_birth(self.module.config) and self.module.frame_finalized and
             self.global_step == self.num_steps))

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
                if getattr(getattr(self.module.config,'birth',None),'frame_training_plan',False):
                    continue
                if uses_early_replacement(self.module.config) and step>70:
                    continue
                if uses_residual_birth(self.module.config) and step > self.module.motion_steps:
                    continue
                for _ in range(self._views_per_step(step)):
                    fetcher.next()
            # Rebuilding a DataLoader and replaying views can draw host RNG.
            # The optimizer and cloning resume from the saved post-step state.
            self._restore_rng(self._resume_rng)
            self._resume_rng = None

    def _views_per_step(self, step):
        return 1

    def _fit_views(self, fetcher, step, *, defer_metrics=False):
        """Average image objectives before one shared-motion optimizer update."""
        count = self._views_per_step(step)
        packages, metrics, loss_value = [], {}, 0.
        plan = getattr(getattr(self.module.config,'birth',None), 'frame_training_plan', False)
        plan_views = self.module.frame_plan_training_views(step, count) if plan else None
        begin_shared = getattr(self.module, '_begin_shared_training_views', None)
        shared_pose = begin_shared(step) if count > 1 and begin_shared is not None else False
        try:
            for index in range(count):
                # The regularizer does not depend on the camera. Charging it on
                # the last view with weight=count preserves the averaged objective.
                camera = None
                if plan:
                    viewpoint = plan_views[index]
                elif uses_residual_birth(self.module.config) and (step > self.module.motion_steps or
                        (uses_early_replacement(self.module.config) and step>70)):
                    viewpoint, camera = self.module.transaction_training_view(step, index, count)
                else:
                    viewpoint = fetcher.next()
                probe = plan and gradient_probe_step(self.module, step)
                loss, values, package = self.module.training_step(viewpoint, step,
                    regularization_weight=0. if probe else count if index == count - 1 else 0.)
                if camera is not None:
                    package['birth_camera_index'] = camera
                if loss.requires_grad:
                    (loss / count).backward(retain_graph=shared_pose == 'shared' and index < count - 1)
                if (index == 0 and count > 1 and step in (201, 250, 300) and
                        getattr(getattr(self.module.config, 'full_update', None), 'regional_repair', False) and
                        getattr(getattr(self.module.config, 'full_update', None), 'gradient_diagnostics', True)):
                    self.module._plan_record_regional_image_gradients(step, viewpoint)
                loss_value = loss_value + loss.detach() / count
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0.) + value / count
                if count > 1:
                    # Keep only evidence needed after backward, not full images.
                    package = {name: package[name] for name in
                        ('visibility_filter', 'motion_contribution_filter', 'viewspace_points',
                         'radii','birth_feature_probe','birth_camera_index','birth_branch','birth_image_loss','birth_psnr',
                         'regional_error_evidence')
                        if name in package}
                packages.append(package)
            if shared_pose == 'proxy':
                self.module._finish_shared_training_views()
        finally:
            if shared_pose:
                self.module._end_shared_training_views()
        metrics['gs'] = values['gs']
        self.module.post_training_step(packages[0] if count == 1 else packages)
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
        end_frame = getattr(self, '_end_frame_exclusive', self.datamodule.num_frames)
        if start_frame >= end_frame:
            self._log_sequence_summary([])
            return []
        reports = []
        for frame in range(start_frame, end_frame):
            self.datamodule.current_frame = frame
            plan = getattr(self, '_verified_prefix_plan', None)
            color_prefix_complete = bool(plan is not None and frame == plan.resume_frame and
                self.module.config.full_update.resume_from_finalized_200 and
                self.module.frame_finalized and self.global_step == 200)
            if self.global_step >= self.num_steps or color_prefix_complete:
                logging.info('Frame %d is complete in checkpoint; continuing', frame)
                self._restore_rng(self._resume_rng)
                self._resume_rng = None
                self.module.begin_next_frame()
                self.global_step = 0
                continue
            report = self.frame_training_loop()
            if report is None and getattr(self, 'color_fork_capture', False):
                return reports
            reports.append(report)
            self.module.begin_next_frame()
            self.global_step = 0
        self._log_sequence_summary(reports)
        return reports

    def _sequence_summary_rows(self, reports):
        directory = getattr(self, 'output_dir', None)
        rows = {}
        if directory is not None:
            directory = Path(directory) / 'frame_summary'
            if directory.is_dir():
                for path in sorted(directory.glob('frame_*.json')):
                    try:
                        report = json.loads(path.read_text())
                    except (OSError, ValueError) as error:
                        logging.warning('Skipping unreadable frame summary %s: %s',
                                        path, error)
                        continue
                    if not isinstance(report, dict):
                        continue
                    frame = _frame_number(report)
                    if frame is None:
                        try:
                            frame = int(path.stem.rsplit('_', 1)[-1])
                        except ValueError:
                            continue
                    rows[frame] = report
        for report in reports:
            if not isinstance(report, dict):
                continue
            frame = _frame_number(report)
            if frame is not None:
                rows[frame] = report
        return [rows[frame] for frame in sorted(rows)]

    def _log_sequence_summary(self, reports):
        if not self.is_global_zero:
            return
        rows = self._sequence_summary_rows(reports)
        if not rows:
            return
        frames = [frame for row in rows
                  if (frame := _frame_number(row)) is not None]
        dynamic = [row for row in rows
                   if (_frame_number(row) or 0) >= 1] or rows
        evaluation = [_evaluation_metrics(row) for row in rows]
        psnr = _finite_mean(row.get('psnr') for row in evaluation)
        ssim = _finite_mean(row.get('ssim') for row in evaluation)
        lpips = _finite_mean(row.get('lpips') for row in evaluation)

        storage = [row.get('encoded_bytes') for row in dynamic]
        mean_storage = _finite_mean(storage)
        storage_values = [float(value) for value in storage
                          if _finite_number(value)]
        total_storage = sum(storage_values) if storage_values else None
        references = [row.get('shared_reference_bytes') for row in rows]
        reference = max((float(value) for value in references
                         if _finite_number(value)), default=None)
        if reference is None:
            directory = getattr(self, 'output_dir', None)
            if directory is not None:
                path = (Path(directory) / 'frame_storage' / 'rigid_tree' /
                        'reference_frame0.npz')
                if path.is_file():
                    reference = path.stat().st_size

        mean_train = _finite_mean(row.get('training_seconds') for row in dynamic)
        mean_wall = _finite_mean(row.get('elapsed_seconds') for row in dynamic)
        wall = [float(row['elapsed_seconds']) for row in dynamic
                if _finite_number(row.get('elapsed_seconds'))]
        total_wall = sum(wall) if wall else None
        if mean_train is None:
            mean_train = mean_wall

        if all(value is None for value in
               (psnr, ssim, lpips, mean_storage, reference, mean_train, mean_wall)):
            return
        span = f' ({min(frames)}-{max(frames)})' if frames else ''
        logging.info('Sequence summary | %d frames%s', len(rows), span)
        if any(value is not None for value in (psnr, ssim, lpips)):
            logging.info('  PSNR mean=%s dB | SSIM mean=%s | LPIPS mean=%s',
                         _log_value(psnr, '.4f'), _log_value(ssim, '.4f'),
                         _log_value(lpips, '.4f'))
        if any(value is not None for value in
               (mean_storage, total_storage, reference)):
            sequence_bytes = (None if total_storage is None else
                              total_storage + (reference or 0.))
            logging.info('  Storage mean=%s KB/frame | dynamic total=%s MB | '
                         'sequence total=%s MB | reference=%s KB',
                         _log_value(None if mean_storage is None else mean_storage / 1000, '.2f'),
                         _log_value(None if total_storage is None else total_storage / 1e6, '.2f'),
                         _log_value(None if sequence_bytes is None else sequence_bytes / 1e6, '.2f'),
                         _log_value(None if reference is None else reference / 1000, '.2f'))
        if any(value is not None for value in (mean_train, mean_wall, total_wall)):
            logging.info('  Training mean=%s s/frame | wall mean=%s s/frame | '
                         'wall total=%s s',
                         _log_value(mean_train, '.2f'),
                         _log_value(mean_wall, '.2f'),
                         _log_value(total_wall, '.1f'))

    def frame_training_loop(self):
        Trainer._assert_frame_step_budget(self)
        frame = self.current_frame
        started = time.perf_counter()
        if getattr(self, 'gate_fork_from_step150', False) and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        fetcher = self._make_fetcher()
        self._resume_fetcher(fetcher)
        starting_step = self.global_step
        train_loss = 0.0
        last_metrics = {}
        metrics_ema = dict(image=0., psnr=0.)
        pending_metrics = []
        pending_steps = []
        stage_view = None
        self._stage_viewpoint = None
        self._frame_diagnostic_seconds = 0.

        step_log = self.output_dir / 'step_diagnostics' / f'frame_{frame:03d}.csv'
        step_log.parent.mkdir(parents=True, exist_ok=True)
        step_handle = step_log.open('w', newline='')
        step_writer = csv.writer(step_handle)
        step_writer.writerow(('frame', 'step', 'phase', 'loss', 'image_loss', 'psnr','early_focus_mse','early_batch_mse',
            'birth_storage_loss','birth_storage_surrogate_bytes','birth_storage_support_points','birth_storage_ramp',
            'population_loss','population_old_support','population_new_support','population_ramp'))

        with tqdm(total=self.num_steps, initial=self.global_step,
                  desc=f'Training frame {frame}', dynamic_ncols=True,
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| '
                             '{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as progress_bar:
            try:
                while self.global_step < self.num_steps:
                    Trainer._assert_frame_step_budget(self)
                    step = self.global_step+1
                    self.global_step = step
                    self.module.pre_training_step()
                    Trainer._assert_frame_step_budget(self)
                    pending_steps.append(step)
                    pending_metrics.append(self._fit_views(fetcher, step, defer_metrics=True))
                    if getattr(progress_bar, 'total', self.num_steps) != self.num_steps:
                        progress_bar.total = self.num_steps
                    if (self.config.stage_image_camera is not None and
                            not getattr(getattr(self.module.config, 'birth', None),
                                        'frame_training_plan', False) and
                            step == self.module.motion_steps):
                        handoff = getattr(getattr(self.module.config, 'birth', None),
                                          'source_handoff', False)
                        if uses_residual_birth(self.module.config):
                            stage_view = self._record_stage_image(
                                step, stage_view, label='transaction_baseline', old_only=True)
                        elif handoff:
                            stage_view = self._record_stage_image(
                                step, stage_view, label='handoff_initialization',
                                old_only=False)
                        else:
                            stage_view = self._record_stage_image(
                                step, stage_view, label='motion_old_only', old_only=True)
                    if step % 10 == 0 or step == self.num_steps:
                        # The progress display only refreshes every ten steps. Keep
                        # detached scalars on device until then instead of forcing
                        # a CUDA synchronization after every optimizer update.
                        for pending_step, (loss_value, metrics) in zip(
                                pending_steps,
                                self._materialize_step_metrics(pending_metrics)):
                            train_loss += loss_value
                            last_metrics = metrics
                            write_started = time.perf_counter()
                            step_writer.writerow((
                                frame, pending_step,
                                Trainer._phase_name(self, pending_step),
                                loss_value, metrics['image_loss'], metrics['psnr'],
                                metrics.get('early_focus_mse'),metrics.get('early_batch_mse'),
                                metrics.get('birth_storage_loss'),metrics.get('birth_storage_surrogate_bytes'),
                                metrics.get('birth_storage_support_points'),metrics.get('birth_storage_ramp'),
                                metrics.get('population_loss'),metrics.get('population_old_support'),
                                metrics.get('population_new_support'),metrics.get('population_ramp')))
                            self._frame_diagnostic_seconds += time.perf_counter() - write_started
                            for name, value in (('image', metrics['image_loss']),
                                                ('psnr', metrics['psnr'])):
                                metrics_ema[name] = .6 * metrics_ema[name] + .4 * value
                        pending_metrics.clear()
                        pending_steps.clear()
                        flush_started = time.perf_counter()
                        step_handle.flush()
                        self._frame_diagnostic_seconds += time.perf_counter() - flush_started
                        progress_bar.set_postfix(dict(image=f"{metrics_ema['image']:.5f}",
                            psnr=f"{metrics_ema['psnr']:.2f}", gs=metrics['gs']), refresh=False)
                        progress_bar.update(step - progress_bar.n)
                    if getattr(self, 'color_fork_capture', False) and step == 200:
                        if not (self.module.config.full_update.color_only and self.module.config.full_update.color_point_budget and
                                self.module.frame_prepared and not self.module.frame_finalized):
                            raise ValueError('Color fork requires a completed, unfinalized step 200')
                        self.save_checkpoint()
                        logging.info('Saved common color fork at frame %d step 200: %s', frame, self.ckpt_save_path)
                        return None
                    if self._checkpoint_due() and step != self.num_steps:
                        self.save_checkpoint()
            finally:
                step_handle.close()
        evaluation = self.validation_loop()
        train_camera = getattr(self, 'diagnostic_train_camera', None)
        if train_camera is not None:
            rng = self._capture_rng()
            try:
                dataset = self.datamodule.get_train_dataset()
                matches = [i for i, item in enumerate(dataset.items) if Path(str(item.path)).name == train_camera]
                if len(matches) != 1:
                    raise ValueError(f'Train diagnostic camera {train_camera!r} must match exactly one training view')
                with torch.no_grad():
                    sample = self.module.validation_step(dataset[matches[0]].to(self.device), 1000000, 0)
                evaluation['train_sample'] = {**sample, 'camera': train_camera}
            finally:
                self._restore_rng(rng)
        executed_steps = self.global_step-starting_step
        elapsed = time.perf_counter() - started
        diagnostics = self._frame_diagnostic_seconds
        report = dict(self.module.frame_report or {},
            stage_image_metrics=self.module.stage_image_metrics,
            evaluation=evaluation, validation_per_view=getattr(self, '_validation_per_view', {}), final_train_metrics=last_metrics,
            mean_train_loss=train_loss / max(1, executed_steps),
            train_loss_steps=executed_steps,
            elapsed_seconds=elapsed,
            diagnostic_seconds=diagnostics,
            training_seconds=max(0., elapsed - diagnostics),
            peak_gpu_allocated_bytes=(int(torch.cuda.max_memory_allocated(self.device))
                if getattr(self, 'gate_fork_from_step150', False) and self.device.type == 'cuda' else None),
            checkpoint=(str(self.ckpt_save_path) if self._checkpoint_due() or
                        getattr(self, 'gate_fork_from_step150', False) else None))
        directory = self.output_dir / 'frame_summary'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'frame_{frame:03d}.json').write_text(
            json.dumps(report, indent=2, default=str))
        if self._checkpoint_due() or getattr(self, 'gate_fork_from_step150', False):
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

    def _phase_name(self, step):
        if getattr(getattr(self.module.config,'birth',None), 'frame_training_plan', False):
            return frame_plan_phase(self.module, step)
        if uses_early_replacement(self.module.config) and step>70:
            return 'early_joint' if step<=200 else 'early_fixed_population'
        gate = self.module.config.rigid_tree.spatial_gate
        return step_phase(step, self.module.motion_steps, gate.gate_start_step,
                          gate.freeze_step,
                          self.module.config.rigid_tree.amplitude_refit_steps)

    @torch.no_grad()
    def record_birth_comparison(self):
        """Same final fitted state, rendered with and without newborn GS."""
        if self.config.stage_image_camera is None:
            return
        if getattr(getattr(self.module.config, 'birth', None), 'frame_training_plan', False):
            self._record_stage_image(self.module.total_steps, self._stage_viewpoint,
                                     label='birth_off', old_only=True)
            self._record_stage_image(self.module.total_steps, self._stage_viewpoint,
                                     label='birth_on', old_only=False)
            return
        if uses_residual_birth(self.module.config):
            self._record_stage_image(self.module.total_steps, self._stage_viewpoint,
                                     label='transaction_final', old_only=False)
            return
        if getattr(getattr(self.module.config, 'birth', None),
                   'source_handoff', False):
            self._record_stage_image(self.module.total_steps, self._stage_viewpoint,
                                     label='handoff_final', old_only=False)
            return
        self._record_stage_image(self.module.total_steps, self._stage_viewpoint,
                                 label='birth_off', old_only=True)
        self._record_stage_image(self.module.total_steps, self._stage_viewpoint,
                                 label='birth_on', old_only=False)

    @torch.no_grad()
    def snapshot_frame_plan_reference(self):
        """Keep the preceding decoded population for optional independent replay."""
        self.module._plan_diagnostic_reference = None
        if (self.config.stage_image_camera is None and not
                getattr(getattr(self.module.config, 'birth', None), 'regional_postfit_admission', False) and not
                getattr(self.module, 'full_update_active', False) and not
                getattr(getattr(self.module.config, 'full_update', None), 'color_only', False) and not
                getattr(getattr(self.module.config, 'full_update', None), 'regional_full_point_budget', 0)):
            return
        started = time.perf_counter()
        try:
            self.module._plan_diagnostic_reference = {
                name: value.detach().cpu().clone()
                for name, value in model_raw(self.module.gaussians).items()}
        finally:
            self._frame_diagnostic_seconds += time.perf_counter() - started

    @torch.no_grad()
    def record_frame_plan_stage(self, step, stage, *, old_only=False, old_pose=None):
        if self.config.stage_image_camera is not None:
            self._record_stage_image(step, self._stage_viewpoint, label=f'plan_{stage}',
                                     old_only=old_only, old_pose=old_pose)

    @torch.no_grad()
    def record_frame_plan_decoded(self):
        """Render the actual independently decoded packet, including all updates."""
        if self.config.stage_image_camera is None:
            return
        started = time.perf_counter()
        rng = self._capture_rng()
        try:
            # The same decoder is used for offline sequential stream verification.
            # Its command-line module adds the repository root to sys.path;
            # restore the process search path after loading it.
            path = list(sys.path)
            try:
                from scripts.replay_rigid_tree import decode_frame, replay
            finally:
                sys.path[:] = path
            prior = self.module.motion.frame_reference_state
            raw = getattr(self.module, '_plan_diagnostic_reference', None)
            if raw is None:
                # Older mid-frame checkpoints predate the optional snapshot.
                previous, raw = replay(self.output_dir / 'frame_storage' / 'rigid_tree',
                                       self.decoder_reference_path, self.current_frame - 1)
                if previous.digest() != prior.digest():
                    raise ValueError('Diagnostic replay reference differs from the training reference')
            directory = self.output_dir / 'frame_storage' / 'rigid_tree'
            decoded, raw = decode_frame(directory, self.current_frame, prior, raw)
            manifest = json.loads((directory / f'frame_{self.current_frame:03d}.json').read_text())
            metadata = dict(
                decoded_state_matches_manifest=decoded.digest().hex() == manifest['decoded_state_sha256'],
                decoded_raw_matches_manifest=reference_digest(raw) == manifest['raw_state_sha256'],
                population_old_count=int(manifest['old_count']),
                population_birth_count=int(manifest['birth_count']))
        finally:
            self._restore_rng(rng)
            self._frame_diagnostic_seconds += time.perf_counter() - started
        self._record_stage_image(self.module.total_steps, self._stage_viewpoint,
                                 label='plan_decoded', raw=raw, metadata=metadata)

    @torch.no_grad()
    def _record_stage_image(self, step, viewpoint=None, label=None, old_only=False,
                            raw=None, metadata=None, old_pose=None):
        """One fixed heldout view; no gradients, selection, or RNG side effects."""
        started = time.perf_counter()
        rng = self._capture_rng()
        renderer = self.module.renderer
        previous_model = renderer.gaussians
        missing = object()
        previous_collection = getattr(renderer, 'collect_contributions', missing)
        previous_contributions = getattr(renderer, 'last_contributions', missing)
        render_cache = getattr(self.module.gaussians, '_render_cache', missing)
        previous_cache = render_cache.copy() if render_cache is not missing else None
        fields = ('rigid_xyz', 'rigid_rotation', 'rigid_opacity', 'rigid_alpha',
                  'rigid_sh_dc', 'rigid_logscale')
        previous_fields = {name: getattr(self.module.gaussians, name, missing) for name in fields}
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
            self._stage_viewpoint = viewpoint
            if raw is None and old_pose is None:
                rendered = self.module.forward(viewpoint, old_only=old_only)['render'].clamp(0, 1)
            else:
                if old_pose is not None:
                    if raw is not None or not old_only:
                        raise ValueError('A diagnostic pose override requires the old-only population')
                    current = model_raw(self.module.gaussians)
                    xyz, rotation = old_pose
                    if (xyz.shape != current['xyz'].shape or
                            rotation.shape != current['rotation'].shape):
                        raise ValueError('Diagnostic pose lost the incumbent row layout')
                    current['xyz'], current['rotation'] = xyz, rotation
                    feature = getattr(self.module, 'feature_dc_update', None)
                    if feature is not None:
                        current['sh_dc'] = feature(training=False)
                    scale = getattr(self.module, 'logscale_update', None)
                    if scale is not None and scale.ready:
                        current['logscale'] = scale.current_logscale(current['logscale'])
                    opacity = getattr(self.module, 'opacity_update', None)
                    alpha = opacity.alphas() if opacity is not None else None
                else:
                    current = {name: value.to(self.device) for name, value in raw.items()}
                    alpha = None
                renderer.gaussians = RawView(current, self.module.gaussians.active_sh_degree,
                                             self.module.gaussians, alpha)
                if previous_collection is not missing:
                    renderer.collect_contributions = False
                rendered = renderer(viewpoint, self.module.background)['render'].clamp(0, 1)
            difference = rendered - viewpoint.image
            old_count = len(self.module.gaussians._xyz)
            birth_count = (len(self.module.births._xyz) if
                           getattr(self.module, 'motion_committed', False) and
                           not getattr(self.module, 'frame_finalized', False) else 0)
            record = dict(step=int(step), phase=label, camera=str(viewpoint.path),
                          mse=float(difference.square().mean()),
                          psnr=float(psnr(rendered, viewpoint.image)), old_only=old_only,
                          representation='decoded_stream' if raw is not None else 'training_quantized',
                          pose='pre_tail' if old_pose is not None else 'current',
                          population_old_count=old_count, population_birth_count=birth_count,
                          shared_prestructure_active=bool(getattr(getattr(self.module, 'motion', None),
                                                                 'shared_prestructure_active', False)),
                          participates_in_training_or_selection=False)
            if metadata:
                record.update(metadata)
            records = self.module.stage_image_metrics
            records[:] = [item for item in records
                          if (item['step'], item['phase']) != (step, record['phase'])]
            records.append(record)
            return viewpoint
        finally:
            renderer.gaussians = previous_model
            if previous_collection is not missing:
                renderer.collect_contributions = previous_collection
            if previous_contributions is not missing:
                renderer.last_contributions = previous_contributions
            elif hasattr(renderer, 'last_contributions'):
                delattr(renderer, 'last_contributions')
            for name, value in previous_fields.items():
                if value is not missing:
                    setattr(self.module.gaussians, name, value)
                elif hasattr(self.module.gaussians, name):
                    delattr(self.module.gaussians, name)
            if render_cache is not missing:
                render_cache.clear()
                render_cache.update(previous_cache)
                self.module.gaussians._render_cache = render_cache
            elif hasattr(self.module.gaussians, '_render_cache'):
                delattr(self.module.gaussians, '_render_cache')
            self._restore_rng(rng)
            self._frame_diagnostic_seconds += time.perf_counter() - started

    @torch.no_grad()
    def validation_loop(self):
        loaders = self.datamodule.val_dataloader()
        if not isinstance(loaders, (tuple, list)):
            loaders = [loaders]
        groups = []
        per_view = {}
        for loader_idx, loader in enumerate(loaders):
            rows = []
            for index, viewpoint in enumerate(loader):
                row = self.module.validation_step(viewpoint.to(self.device), index, loader_idx)
                rows.append(row)
                name = Path(str(viewpoint.path)).name
                if name in per_view:
                    raise ValueError('Duplicate validation camera in frame')
                per_view[name] = {key: float(value) for key, value in row.items()}
            groups.append(rows)
        self._validation_per_view = per_view
        return self.module.validation_end(
            groups[0] if len(groups) == 1 else groups,
            num_loaders=len(groups))
