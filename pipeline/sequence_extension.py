"""Validate and copy a completed causal prefix before extending its sequence.

No model state or encoded frame is rewritten. Ordinary checkpoint restoration
remains strict; callers must explicitly request this completed-prefix path.
"""

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .model.birth_quantization import reference_digest
from .model.rigid_tree_codec import DecodedState


@dataclass(frozen=True)
class PrefixFile:
    relative_path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class SequenceExtensionPlan:
    source_run: Path
    destination_run: Path
    resume_frame: int
    previous_frames: int
    extended_frames: int
    files: tuple
    images: tuple
    decoded_sha256: str
    raw_sha256: str
    source_planned_frames: int | None = None


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def relocate_checkpoint_reference(checkpoint, reference_layout):
    """Accept a moved layout only when its migration manifest proves its bytes."""
    config = checkpoint.get('rigid_tree_config')
    if not isinstance(config, dict) or not isinstance(config.get('spatial_gate'), dict):
        return checkpoint
    previous = config['spatial_gate'].get('reference_layout')
    if not isinstance(previous, str) or not isinstance(reference_layout, str) or previous == reference_layout:
        return checkpoint
    reference = Path(reference_layout)
    manifest = reference.parent / 'manifest.json'
    if not reference.is_file() or not manifest.is_file():
        return checkpoint
    try:
        entry = json.loads(manifest.read_text())['files']['reference_layout']
        valid_origin = previous in (entry['original_path'], entry['original_resolved_path'])
        expected_hash = entry['sha256']
        valid_name = entry['file'] == reference.name
    except (OSError, ValueError, KeyError, TypeError):
        return checkpoint
    if not valid_origin or not valid_name:
        return checkpoint
    if not isinstance(expected_hash, str) or _hash(reference) != expected_hash:
        raise ValueError('Relocated reference layout does not match its recorded SHA256')

    module = checkpoint.get('rigid_tree_module_config')
    module_config = module.get('rigid_tree') if isinstance(module, dict) else None
    motion = checkpoint.get('rigid_tree_motion')
    motion_config = motion.get('config') if isinstance(motion, dict) else None
    for saved in (module_config, motion_config):
        if saved is not None and (not isinstance(saved, dict) or
                not isinstance(saved.get('spatial_gate'), dict) or
                saved['spatial_gate'].get('reference_layout') != previous):
            return checkpoint

    def changed(saved):
        return {**saved, 'spatial_gate': {**saved['spatial_gate'],
                                         'reference_layout': reference_layout}}

    relocated = {**checkpoint, 'rigid_tree_config': changed(config)}
    if module_config is not None:
        relocated['rigid_tree_module_config'] = {**module, 'rigid_tree': changed(module_config)}
    if motion_config is not None:
        relocated['rigid_tree_motion'] = {**motion, 'config': changed(motion_config)}
    return relocated


def _plain_config(value):
    # asdict preserves OmegaConf containers loaded by the actual CLI.
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {key: _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'Missing or invalid prefix metadata: {path}') from error


def _directory_guard(root, relative=None):
    """Reject directory aliases; individual source file links stay read-only."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f'Run must be an actual directory: {root}')
    if relative is not None:
        current = root
        for part in Path(relative).parts[:-1]:
            current = current / part
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise ValueError(f'Prefix directory is a link or file: {current}')


def _camera_map(path, frames):
    result = {}
    for item in _read_json(path):
        frame = item.get('frame')
        name = item.get('img_name')
        if (isinstance(frame, bool) or not isinstance(frame, int) or
                not 0 <= frame < frames or not isinstance(name, str)):
            raise ValueError('Invalid camera frame identity')
        key = (frame, name)
        if key in result:
            raise ValueError('Duplicate frame/camera identity')
        result[key] = {k: v for k, v in item.items() if k != 'id'}
    if {key[0] for key in result} != set(range(frames)):
        raise ValueError('Camera metadata does not cover the sequence')
    return result


def _empty_destination_stream(destination):
    directory = destination / 'frame_storage' / 'rigid_tree'
    _directory_guard(destination, 'frame_storage/rigid_tree/placeholder')
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Extension destination already contains stream files')


def _validate_adaptive_birth_completion(checkpoint, manifest, terminal_step):
    """Validate the v19 adaptive-birth ledger, which has no population screening."""
    birth_control = ((checkpoint.get('rigid_tree_frame') or {}).get('birth_control') or {})
    control = birth_control.get('early_replacement') or {}
    state = checkpoint.get('early_replacement_state') or {}
    ledger = manifest.get('training_step_accounting') or {}
    source = control.get('source_count')
    population = control.get('training_population')
    capacity = control.get('population_cap')
    removed = control.get('low_alpha_newborns_removed')
    committed = birth_control.get('committed_population_count')
    encoded = manifest.get('birth_count')
    counts = (source, population, capacity, removed, committed, encoded,
              birth_control.get('proposal_count'))
    if (terminal_step != 300 or any(isinstance(n, bool) or not isinstance(n, int) or n < 0
                                    for n in counts)):
        raise ValueError('Adaptive prefix has invalid completed population counts')
    expected_birth = 230 if source else 0
    expected = dict(warm_gate=70, joint=130, final_population=100,
                    global_iterations=300, shared_adam_steps=300,
                    motion_participation=200, birth_participation=expected_birth,
                    joint_overlap=130 if source else 0, extra_SGD_steps=0,
                    topology_born_step=70, population_frozen_step=70,
                    final_alpha_cleanup_step=300)
    expected_control = dict(global_iterations=300, motion_iterations=200,
                            newborn_iterations=expected_birth,
                            joint_overlap_iterations=130 if source else 0,
                            final_population_iterations=100,
                            new_alpha_iterations=expected_birth,
                            population_screening=False, codec_trials200=0,
                            finish_render_calls=0, extra_SGD_steps=0)
    if (source != population or source != birth_control['proposal_count'] or source > capacity or
            committed != encoded or committed + removed != source or
            state.get('initialized') is not True or state.get('shared_steps') != 300 or
            state.get('birth_updates') != expected_birth or
            any(ledger.get(key) != value for key, value in expected.items()) or
            any(control.get(key) != value for key, value in expected_control.items()) or
            checkpoint.get('dual_birth_state') is not None):
        raise ValueError('Adaptive prefix has inconsistent no-screening population or training ledger')


def validate_sequence_extension(checkpoint, current_data_config,
                                current_trainer_config, source_run,
                                destination_run, current_extent, current_images,
                                completed_prefix_frames=None):
    source, destination = Path(source_run).absolute(), Path(destination_run).absolute()
    _directory_guard(source)
    _directory_guard(destination)
    source, destination = source.resolve(), destination.resolve()
    if (source == destination or source in destination.parents or
            destination in source.parents):
        raise ValueError('Extension source and destination must be separate runs')
    _empty_destination_stream(destination)
    old_data = _plain_config(checkpoint.get('rigid_tree_data_config', {}))
    old_trainer = _plain_config(checkpoint.get('rigid_tree_trainer_config', {}))
    current_data_config = _plain_config(current_data_config)
    current_trainer_config = _plain_config(current_trainer_config)
    old_frames = old_data.get('extra_dataset_kwargs', {}).get('num_frames')
    new_frames = current_data_config.get('extra_dataset_kwargs', {}).get('num_frames')
    if (any(isinstance(n, bool) or not isinstance(n, int) for n in (old_frames, new_frames)) or
            old_frames < 2 or new_frames < 2):
        raise ValueError('Sequence extension requires valid frame counts')
    prefix_frames = old_frames if completed_prefix_frames is None else completed_prefix_frames
    if (isinstance(prefix_frames, bool) or not isinstance(prefix_frames, int) or
            not 2 <= prefix_frames <= old_frames):
        raise ValueError('Completed prefix length must lie within the source planned sequence')
    # An interrupted run may recover its unchanged planned length from an
    # explicitly selected completed prefix. Ordinary extension stays strict;
    # a complete same-length sequence has no future frame to resume.
    if (new_frames < old_frames or
            (new_frames == old_frames and
             (completed_prefix_frames is None or prefix_frames == old_frames))):
        raise ValueError('Sequence extension needs more frames or an explicit incomplete prefix for recovery')
    expected_data = {**old_data, 'extra_dataset_kwargs': {
        **old_data['extra_dataset_kwargs'], 'num_frames': new_frames}}
    old_root = Path(old_data['root']).absolute()
    new_root = Path(current_data_config['root']).absolute()
    relocated = old_root != new_root
    if relocated:
        # A separate symlink view can extend a dataset without changing the
        # historical view. Every prefix image and shared calibration file is
        # checked against the original target below.
        expected_data['root'] = current_data_config['root']
    if expected_data != current_data_config:
        raise ValueError('Only num_frames and an equivalent symlink dataset root may change')
    if relocated:
        for relative in ('sparse/0/cameras.bin', 'sparse/0/images.bin',
                         'sparse/0/points3D.bin', 'points3D.ply', 'test_split.txt'):
            old_path, new_path = old_root / relative, new_root / relative
            if old_path.is_file() or new_path.is_file():
                if (not old_path.is_file() or not new_path.is_file() or
                        old_path.resolve(strict=True) != new_path.resolve(strict=True)):
                    raise ValueError('Relocated dataset changed a shared calibration asset')
    old_trainer = {**old_trainer}
    old_trainer.setdefault('stage_image_camera', None)
    old_trainer.setdefault('motion_views_per_step', 1)
    current_trainer_config.setdefault('motion_views_per_step', 1)
    old_saves, new_saves = old_trainer.get('saving_ckpt_frames'), current_trainer_config.get('saving_ckpt_frames')
    if (not isinstance(old_saves, list) or not isinstance(new_saves, list) or
            any(isinstance(n, bool) or not isinstance(n, int) for n in new_saves) or
            len(new_saves) != len(set(new_saves)) or
            not set(old_saves).issubset(new_saves) or
            any(n < prefix_frames or n >= new_frames for n in set(new_saves) - set(old_saves)) or
            {**old_trainer, 'saving_ckpt_frames': new_saves} != current_trainer_config):
        raise ValueError('Only future checkpoint frame appends are allowed')
    resume = prefix_frames - 1
    frame_state = checkpoint.get('rigid_tree_frame', {})
    motion = checkpoint.get('rigid_tree_motion') or {}
    config = checkpoint.get('rigid_tree_config') or {}
    terminal_step = config.get('motion_steps', 0) + config.get('birth_steps', 0)
    terminal_step += config.get('post_birth_motion_steps', 0)
    if (checkpoint.get('birth_allocation') == 'frame_training_plan_v1' and
            (checkpoint.get('rigid_tree_module_config') or {}).get('full_update', {}).get('stop_after_motion') is True):
        terminal_step = config.get('motion_steps', 0)
    if config.get('online_flat_clustering'):
        terminal_step += config.get('post_partition_steps', 0)
    allocation = checkpoint.get('birth_allocation')
    population_allocations = ('selected_population_refit_v5', 'selected_population_refit_v5_1',
                              'inherited_intrinsic_radiance_v6', 'bounded_intrinsic_radiance_v6_1',
                              'conservative_detail_family_v7', 'surface_patch_takeover_v8',
                              'support_reincarnation_crossfit_v9','dual_predictive_support_v10','source_residual_support_v11',
                              'quick_source_residual_300_v12','shadow_source_residual_300_v13',
                              'persistent_anchored_bank_300_v14','early_source_replacement_300_v15','adaptive_residual_birth_300_v16')
    if allocation in ('residual_surface_cohorts_v4', *population_allocations):
        method = checkpoint.get('rigid_tree_module_config') or {}
        birth = method.get('birth') or {}
        max_cohorts = (birth.get('residual') or {}).get('max_cohorts', 2)
        schedule = checkpoint.get('birth_cohort_state') or {}
        terminal_step = schedule.get('end_step')
        index = schedule.get('index')
        start = schedule.get('start_step')
        motion_steps, birth_steps = config.get('motion_steps'), config.get('birth_steps')
        quick_allocation=allocation in ('quick_source_residual_300_v12','shadow_source_residual_300_v13')
        shadow=allocation=='shadow_source_residual_300_v13'
        persistent=allocation=='persistent_anchored_bank_300_v14'
        early=allocation in ('early_source_replacement_300_v15','adaptive_residual_birth_300_v16')
        if quick_allocation and (
                motion_steps!=200 or birth_steps!=100 or index!=0 or start!=200 or
                terminal_step not in ((200,300) if shadow else (300,)) or
                birth.get('radiance_mode')!=('shadow_source_residual' if shadow else 'quick_source_residual')):
            raise ValueError('Quick completed prefix must be exactly one motion200 + birth100 frame')
        if persistent and (motion_steps!=200 or birth_steps!=100 or index!=0 or start!=200 or
                terminal_step not in (200,300) or birth.get('radiance_mode')!='persistent_anchored_newborn'):
            raise ValueError('Persistent bank prefix must complete one bounded motion200 + birth0/100 stage')
        if early and (motion_steps!=200 or birth_steps!=100 or index!=0 or start!=70 or
                terminal_step not in (200,300) or birth.get('radiance_mode')!=('adaptive_residual_birth' if allocation=='adaptive_residual_birth_300_v16' else 'early_source_replacement')):
            raise ValueError('Early support transfer prefix must match gate70 / motion200 / global<=300')
        values = (motion_steps, birth_steps, max_cohorts, terminal_step, index, start)
        if (any(isinstance(value, bool) or not isinstance(value, int) for value in values) or
                min(motion_steps, birth_steps, max_cohorts, terminal_step) < 1 or index < 0 or
                birth.get('mechanism') != 'residual_transaction'):
            raise ValueError('Dynamic birth checkpoint has invalid cohort completion metadata')
        extra = int(allocation in population_allocations)
        if not early and (index >= max_cohorts+extra or start != motion_steps+birth_steps*index or
                (terminal_step != start+birth_steps and not ((shadow or persistent) and terminal_step==200))):
            raise ValueError('Dynamic birth completion exceeds its configured cohort schedule')
        relative = f'frame_storage/rigid_tree/frame_{resume:03d}.json'
        _directory_guard(source, relative)
        manifest = _read_json(source / relative)
        if quick_allocation:
            ledger = manifest.get('training_step_accounting') or {}
            quick = checkpoint.get('quick_birth_state') or {}
            updates = ledger.get('birth_optimizer_updates')
            control=((checkpoint.get('rigid_tree_frame') or {}).get('birth_control') or {})
            proposed = control.get('proposal_count',1)
            expected_updates = 100 if proposed else 0
            if shadow:
                accepted=(control.get('shadow_initializer') or {}).get('accepted_points')
                if (isinstance(accepted,bool) or not isinstance(accepted,int) or accepted<0 or
                        terminal_step!=(300 if accepted else 200)):
                    raise ValueError('Shadow prefix must match its finite-density accepted population')
                expected_updates=terminal_step-200
            if (any(ledger.get(key)!=value for key,value in dict(motion=200,birth=terminal_step-200,total=terminal_step,
                    initializer_optimizer_updates=0,post_admission_optimizer_updates=0).items()) or
                    isinstance(updates,bool) or not isinstance(updates,int) or updates!=expected_updates or
                    quick.get('optimizer_steps')!=updates or quick.get('initialized') is not True or
                    checkpoint.get('dual_birth_state') is not None):
                raise ValueError('Quick prefix training ledger must match one bounded single-model birth optimizer')
        if persistent:
            ledger=manifest.get('training_step_accounting') or {}
            progress=checkpoint.get('persistent_birth_state') or {}
            population=progress.get('training_population_count')
            expected=terminal_step-200
            bank=checkpoint.get('persistent_newborn_bank') or {}
            layout=bank.get('layout');specs=bank.get('specs')
            if (isinstance(population,bool) or not isinstance(population,int) or population<0 or
                    bool(population)!=(terminal_step==300) or progress.get('optimizer_steps')!=expected or
                    progress.get('initialized') is not True or
                    any(ledger.get(key)!=value for key,value in dict(motion=200,birth=expected,total=terminal_step,
                        birth_optimizer_updates=expected,initializer_optimizer_updates=0,post_admission_optimizer_updates=0).items()) or
                    not isinstance(layout,dict) or not isinstance(specs,dict) or set(layout)!=set(specs) or
                    len(set(layout.values()))!=len(layout) or
                    any(not isinstance(uid,str) or isinstance(row,bool) or not isinstance(row,int) or
                        row<0 or row>=len(checkpoint['gaussians'][1]) for uid,row in layout.items()) or
                    (bool(layout) and bank.get('state') is None) or
                    checkpoint.get('dual_birth_state') is not None):
                raise ValueError('Persistent prefix lost its single-model optimizer ledger or UID population')
        if allocation == 'adaptive_residual_birth_300_v16':
            _validate_adaptive_birth_completion(checkpoint, manifest, terminal_step)
        elif early:
            ledger=manifest.get('training_step_accounting') or {}
            state=checkpoint.get('early_replacement_state') or {}
            control=((checkpoint.get('rigid_tree_frame') or {}).get('birth_control') or {}).get('early_replacement') or {}
            source_count=control.get('source_count')
            selected_count=control.get('selected_population')
            expected_birth=(terminal_step-70 if source_count else 0)
            expected=dict(warm_gate=70,joint=130,final_population=terminal_step-200,
                global_iterations=terminal_step,shared_adam_steps=terminal_step,motion_participation=200,
                birth_participation=expected_birth,joint_overlap=130 if source_count else 0,extra_SGD_steps=0,
                topology_born_step=70,population_frozen_step=200)
            if (isinstance(source_count,bool) or not isinstance(source_count,int) or source_count<0 or
                    isinstance(selected_count,bool) or not isinstance(selected_count,int) or selected_count<0 or
                    bool(selected_count)!=(terminal_step==300) or state.get('initialized') is not True or
                    state.get('shared_steps')!=terminal_step or state.get('birth_updates')!=expected_birth or
                    any(ledger.get(key)!=value for key,value in expected.items()) or
                    control.get('SGD_200_codec_replay_exact') is not True or
                    checkpoint.get('dual_birth_state') is not None):
                raise ValueError('Early prefix lost its one-trajectory shared-Adam participation ledger')
        actual_step = manifest.get('optimization_steps')
        if (isinstance(actual_step, bool) or not isinstance(actual_step, int) or
                actual_step != terminal_step or
                (allocation in population_allocations and
                 (checkpoint.get('birth_terminal_state') or {}).get('active', False))):
            raise ValueError('Dynamic birth completion disagrees with its finalized prefix manifest')
    if (checkpoint.get('rigid_tree_version') != 1 or
            checkpoint.get('current_frame') != resume or
            checkpoint.get('global_step') != terminal_step or
            motion.get('frame_index') != resume or
            not all(frame_state.get(k) is True for k in ('prepared', 'finalized', 'motion_committed')) or
            not {'torch', 'numpy', 'python', 'cuda'}.issubset(checkpoint.get('rng_state') or {}) or
            checkpoint.get('births') is not None or checkpoint.get('old_opacity_delta') is not None):
        raise ValueError('Extension needs the completed last frame of its explicitly selected prefix')
    if not (np.isfinite(current_extent) and current_extent > 0 and
            float(current_extent) == motion.get('extent')):
        raise ValueError('Extension changed the scene extent')
    old_cameras = _camera_map(source / 'cameras.json', old_frames)
    new_cameras = _camera_map(destination / 'cameras.json', new_frames)
    if any(new_cameras.get(key) != value for key, value in old_cameras.items()):
        raise ValueError('Extension changed a prefix camera')
    old_cameras = {key: value for key, value in old_cameras.items() if key[0] < prefix_frames}
    images, seen = [], set()
    for frame, filename in current_images:
        if frame >= prefix_frames:
            continue
        key = (frame, Path(filename).stem)
        if key not in old_cameras or key in seen:
            raise ValueError('Dataset frame/camera identity differs from prefix')
        seen.add(key)
        try:
            resolved = Path(filename).resolve(strict=True)
            if not resolved.is_file():
                raise ValueError('Dataset image is not a file')
            if relocated:
                try:
                    relative_image = Path(filename).absolute().relative_to(new_root)
                    original_image = (old_root / relative_image).resolve(strict=True)
                except (ValueError, OSError) as error:
                    raise ValueError('Relocated image does not map to the original dataset') from error
                if original_image != resolved:
                    raise ValueError('Relocated dataset changed a prefix image')
            stat = resolved.stat()
        except OSError as error:
            raise ValueError('Missing prefix image in current dataset') from error
        images.append(dict(frame=frame, camera=key[1], resolved_path=str(resolved),
                           device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns))
    if seen != set(old_cameras):
        raise ValueError('Current dataset is missing prefix camera images')
    relative_files = {'frame_storage/rigid_tree/reference_frame0.npz'}
    for frame in range(1, prefix_frames):
        index = f'frame_storage/rigid_tree/frame_{frame:03d}.json'
        manifest = _read_json(source / index)
        if (manifest.get('frame') != frame or
                manifest.get('reference') != 'frame_storage/rigid_tree/reference_frame0.npz' or
                manifest.get('motion') != f'motion_{frame:03d}.bin'):
            raise ValueError('Prefix frame references a different bootstrap or packet')
        relative_files.update((index, f'frame_storage/rigid_tree/motion_{frame:03d}.bin',
                               f'frame_summary/frame_{frame:03d}.json'))
        for field, stem in (('birth', 'birth'), ('opacity_update', 'opacity'),
                            ('feature_dc_update', 'shdc'),
                            ('logscale_update', 'logscale'),
                            ('full_attribute_update', 'fullattr')):
            name = manifest.get(field)
            if name is not None:
                if name != f'{stem}_{frame:03d}.bin':
                    raise ValueError('Unexpected prefix attribute packet name')
                if field == 'full_attribute_update' and (
                        manifest.get('full_update') is not True or
                        manifest.get('full_attribute_update_timing') != 'after_regional_dc_before_births'):
                    raise ValueError('Prefix full attributes have an invalid decode order')
                relative_files.add(f'frame_storage/rigid_tree/{name}')
        diagnostic = (manifest.get('regional_attribute_repair') or {}).get('diagnostic_sidecar')
        if diagnostic is not None:
            permitted = {
                f'frame_diagnostics/regional_attribute_repair/frame_{frame:03d}_step_200.npz',
                f'frame_diagnostics/learnable_attribute_gate/frame_{frame:03d}.npz',
            }
            if diagnostic not in permitted:
                raise ValueError('Unexpected regional repair diagnostic path')
            relative_files.add(diagnostic)
        region = f'region_statistics/frame_{frame:03d}.json'
        if (source / region).exists():
            relative_files.add(region)
    files = []
    for relative in sorted(relative_files):
        _directory_guard(source, relative)
        _directory_guard(destination, relative)
        original, target = source / relative, destination / relative
        if not original.is_file():
            raise ValueError(f'Missing prefix file: {relative}')
        if target.exists() or target.is_symlink():
            raise ValueError(f'Extension would overwrite a prefix target: {relative}')
        files.append(PrefixFile(relative, _hash(original), original.stat().st_size))
    reference = source / 'frame_storage/rigid_tree/reference_frame0.npz'
    with np.load(reference, allow_pickle=False) as packet:
        if int(packet['frame_index']) != 0 or float(packet['extent']) != float(current_extent):
            raise ValueError('Prefix reference must be the same shared frame0')
    replay = importlib.import_module('scripts.replay_rigid_tree').replay
    decoded, raw = replay(reference.parent, reference, through=resume)
    try:
        saved = DecodedState.restore(motion['codec_previous_state'])
        gaussian = checkpoint['gaussians']
        checkpoint_raw = dict(xyz=gaussian[1], rotation=gaussian[5], logscale=gaussian[4],
                              opacity=gaussian[6], sh_dc=gaussian[2], sh_rest=gaussian[3])
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError('Checkpoint is missing its completed raw/decoded state') from error
    if (decoded.frame_index != resume or decoded.digest() != saved.digest() or
            reference_digest(raw) != reference_digest(checkpoint_raw)):
        raise ValueError('Checkpoint and decoded prefix do not match')
    return SequenceExtensionPlan(source, destination, resume, prefix_frames, new_frames,
                                 tuple(files), tuple(images), decoded.digest().hex(), reference_digest(raw),
                                 source_planned_frames=old_frames)


def copy_verified_prefix(plan):
    """Copy prevalidated packets without links, overwrites or source writes."""
    created = []
    try:
        _empty_destination_stream(plan.destination_run)
        for entry in plan.files:
            _directory_guard(plan.source_run, entry.relative_path)
            _directory_guard(plan.destination_run, entry.relative_path)
            source, target = plan.source_run / entry.relative_path, plan.destination_run / entry.relative_path
            if target.exists() or target.is_symlink():
                raise ValueError(f'Prefix target exists: {entry.relative_path}')
            if source.stat().st_size != entry.bytes or _hash(source) != entry.sha256:
                raise ValueError(f'Prefix source changed: {entry.relative_path}')
        for entry in plan.files:
            source, target = plan.source_run / entry.relative_path, plan.destination_run / entry.relative_path
            _directory_guard(plan.destination_run, entry.relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                created.append(target)
                for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
                    outgoing.write(chunk)
            if _hash(target) != entry.sha256 or _hash(source) != entry.sha256:
                raise ValueError(f'Prefix copy hash mismatch: {entry.relative_path}')
        return dict(source_run=str(plan.source_run), resume_frame=plan.resume_frame,
                    previous_frames=plan.previous_frames, extended_frames=plan.extended_frames,
                    source_planned_frames=plan.source_planned_frames,
                    decoded_sha256=plan.decoded_sha256, raw_sha256=plan.raw_sha256,
                    files=[vars(entry) for entry in plan.files], images=list(plan.images),
                    identity_scope='current_resolved_dataset_paths_no_historical_image_hash',
                    past_frames_reencoded=False)
    except Exception:
        for target in reversed(created):
            target.unlink()
        raise
