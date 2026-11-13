"""已完成序列的续训辅助工具。

核对数据、检查点和已完成帧的码流，在新实验目录复制并记录前序结果；
也提供检查点引用和数据路径迁移的校验。训练循环仍由 trainer.py 调度。
"""

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .model.frame_stream_format import reference_digest
from .model.frame_stream_format import (DecodedState)


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
    historical_images_unverified: bool = False


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


def validate_missing_dataset_root(source, destination, new_root, old_frames, new_frames):
    """Check retained camera and point-cloud evidence when the old root is gone.

    Historical pixel bytes are not in these records; callers must require an
    explicit opt-in and must not present this check as an image hash audit.
    """
    source, destination, new_root = Path(source), Path(destination), Path(new_root)
    if not isinstance(old_frames, int) or not isinstance(new_frames, int):
        raise ValueError('Dataset relocation requires explicit frame counts')
    old_cameras = _camera_map(source / 'cameras.json', old_frames)
    new_cameras = _camera_map(destination / 'cameras.json', new_frames)
    if any(new_cameras.get(key) != value for key, value in old_cameras.items()):
        raise ValueError('Relocated dataset changed recorded camera metadata')
    previous_cloud = source / 'input.ply'
    current_cloud = new_root / 'sparse/0/points3D.ply'
    if not previous_cloud.is_file() or not current_cloud.is_file() or _hash(previous_cloud) != _hash(current_cloud):
        raise ValueError('Relocated dataset point cloud differs from saved input.ply')


def _empty_destination_stream(destination):
    directory = destination / 'frame_storage' / 'rigid_tree'
    _directory_guard(destination, 'frame_storage/rigid_tree/placeholder')
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Extension destination already contains stream files')


def validate_sequence_extension(checkpoint, current_data_config,
                                current_trainer_config, source_run,
                                destination_run, current_extent, current_images,
                                completed_prefix_frames=None,
                                allow_unverified_dataset_root=False):
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
    missing_old_root = relocated and not old_root.exists()
    if missing_old_root:
        if not allow_unverified_dataset_root:
            raise ValueError('Previous dataset root is missing; explicit unverified relocation opt-in is required')
        validate_missing_dataset_root(source, destination, new_root, old_frames, new_frames)
    if relocated:
        # A separate symlink view can extend a dataset without changing the
        # historical view. Every prefix image and shared calibration file is
        # checked against the original target below.
        expected_data['root'] = current_data_config['root']
    if expected_data != current_data_config:
        raise ValueError('Only num_frames and an equivalent symlink dataset root may change')
    if relocated and not missing_old_root:
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
    terminal_manifest_path = f'frame_storage/rigid_tree/frame_{resume:03d}.json'
    _directory_guard(source, terminal_manifest_path)
    terminal_manifest = _read_json(source / terminal_manifest_path)
    terminal_step = terminal_manifest.get('optimization_steps')
    if isinstance(terminal_step, bool) or not isinstance(terminal_step, int) or terminal_step < 1:
        raise ValueError('Completed prefix manifest needs a valid training step count')
    if (checkpoint.get('rigid_tree_version') != 1 or
            checkpoint.get('current_frame') != resume or
            checkpoint.get('global_step') != terminal_step or
            motion.get('frame_index') != resume or
            not all(frame_state.get(k) is True for k in ('prepared', 'finalized', 'motion_committed')) or
            not {'torch', 'numpy', 'python', 'cuda'}.issubset(checkpoint.get('rng_state') or {})):
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
            if relocated and not missing_old_root:
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
                            ('regional_full_attributes', 'regfull'),
                            ('full_attribute_update', 'fullattr')):
            name = manifest.get(field)
            if name is not None:
                if name != f'{stem}_{frame:03d}.bin':
                    raise ValueError('Unexpected prefix attribute packet name')
                if field == 'full_attribute_update' and (
                        manifest.get('full_update') is not True or
                        manifest.get('full_attribute_update_timing') != 'after_regional_dc_before_births'):
                    raise ValueError('Prefix full attributes have an invalid decode order')
                if field == 'regional_full_attributes' and manifest.get('regional_full_timing') != 'after_point_attributes_before_births':
                    raise ValueError('Prefix regional full attributes have an invalid decode order')
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
    replay = importlib.import_module('pipeline.model.frame_stream_decode').replay
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
                                 source_planned_frames=old_frames,
                                 historical_images_unverified=missing_old_root)


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
                    identity_scope=('current_paths_and_recorded_cameras; historical image bytes unverified'
                                    if plan.historical_images_unverified else
                                    'current_resolved_dataset_paths_no_historical_image_hash'),
                    past_frames_reencoded=False)
    except Exception:
        for target in reversed(created):
            target.unlink()
        raise
