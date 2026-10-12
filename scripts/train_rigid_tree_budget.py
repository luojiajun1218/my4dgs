"""Select a rigid-tree rate per frame using actual motion and birth packets.

This is an experiment driver around the unchanged ReCon-GS training path.
Every trial starts from the same selected preceding checkpoint, runs in its
own directory, and is selected using only complete training-view image loss.
The selected packets form one sequentially decoded frame-0 stream.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config/fs_rigid_tree_actual_budget_300f_20261012.yaml'
RAW_FIELDS = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')
CAPTURE_INDEX = dict(xyz=1, sh_dc=2, sh_rest=3, logscale=4,
                     rotation=5, opacity=6)
REFERENCE = 'frame_storage/rigid_tree/reference_frame0.npz'


class BudgetError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _required_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a YAML mapping')
    return value


def load_budget_config(path: Path) -> tuple[dict, dict]:
    path = path.resolve(strict=True)
    plan = _required_mapping(yaml.safe_load(path.read_text()), 'budget config')
    expected = {'base_config', 'source_snapshot', 'experiment_name', 'output_dir',
                'total_frames', 'budget_bytes_exclusive', 'rate_candidates',
                'fallback_rate_candidates', 'train_loss_tie_tolerance',
                'retain_unselected_checkpoints'}
    if set(plan) != expected:
        raise ValueError(f'Budget config fields differ: {sorted(set(plan) ^ expected)}')
    for name in ('base_config', 'source_snapshot', 'output_dir'):
        value = Path(plan[name]).expanduser()
        plan[name] = str((path.parent / value).resolve() if not value.is_absolute()
                         else value.resolve())
    if (plan['total_frames'] != 300 or
            not isinstance(plan['budget_bytes_exclusive'], int) or
            isinstance(plan['budget_bytes_exclusive'], bool) or
            plan['budget_bytes_exclusive'] < 1 or
            not isinstance(plan['experiment_name'], str) or
            not plan['experiment_name'].startswith('fs_rigid_tree_') or
            not isinstance(plan['retain_unselected_checkpoints'], bool)):
        raise ValueError('Invalid complete-sequence budget settings')
    for key in ('rate_candidates', 'fallback_rate_candidates'):
        rates = plan[key]
        if (not isinstance(rates, list) or not rates or
                any(isinstance(rate, bool) or not isinstance(rate, (int, float)) or
                    not math.isfinite(rate) or rate <= 0 for rate in rates) or
                len(set(rates)) != len(rates)):
            raise ValueError(f'{key} must be distinct positive rate weights')
    if (set(plan['rate_candidates']) & set(plan['fallback_rate_candidates']) or
            min(plan['fallback_rate_candidates']) <= max(plan['rate_candidates'])):
        raise ValueError('Fallback rates must be distinct and stronger')
    tolerance = plan['train_loss_tie_tolerance']
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
            not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Invalid training-loss tie tolerance')
    base_path = Path(plan['base_config'])
    base = _required_mapping(yaml.safe_load(base_path.read_text()), 'base config')
    if (base['mode'] != 'train' or
            base['data']['extra_dataset_kwargs']['num_frames'] != 300 or
            base['module']['rigid_tree']['region_count'] < 2 or
            base['data']['training_view_seed'] is None or
            not base['data']['canonical_training_view_order'] or
            not base['data']['eval_test']):
        raise ValueError('Base config must be the 300-frame canonical rigid run')
    if Path(base['output_dir']).resolve() != Path(plan['output_dir']):
        raise ValueError('Base output root differs from budget output root')
    if not Path(base['ckpt_path']).is_file():
        raise FileNotFoundError(base['ckpt_path'])
    snapshot = Path(plan['source_snapshot'])
    manifest = json.loads((snapshot / 'source_manifest.json').read_text())
    for name in manifest['hashes']:
        if file_sha256(snapshot / name) != manifest['hashes'][name]:
            raise ValueError(f'Frozen source changed: {name}')
    if base_path.parent.name != 'config':
        raise ValueError('Base config must be a frozen config/ YAML')
    base_relative = f'config/{base_path.name}'
    if manifest['hashes'].get(base_relative) != file_sha256(base_path):
        raise ValueError('Base config differs from frozen source')
    plan['source_commit'] = manifest['commit']
    plan['source_manifest_sha256'] = file_sha256(snapshot / 'source_manifest.json')
    plan['base_config_sha256'] = file_sha256(base_path)
    plan['budget_config_sha256'] = file_sha256(path)
    return plan, base


def trial_config(base: dict, rate: float, frame: int, plan: dict,
                 checkpoint: Path, trial_name: str, run: Path) -> dict:
    """Vary only rate, evaluation availability, and single-frame saving plan."""
    result = deepcopy(base)
    result['ckpt_path'] = str(checkpoint)
    result['output_dir'] = str(run / 'trials')
    result['experiment_name'] = trial_name
    result['data']['eval_train'] = True
    result['data']['eval_test'] = True
    result['module']['rigid_tree']['rate_loss_weight'] = float(rate)
    steps = (result['module']['rigid_tree']['motion_steps'] +
             result['module']['rigid_tree']['birth_steps'])
    result['trainer']['saving_ckpt_frames'] = [frame]
    result['trainer']['saving_ckpt_steps'] = [steps]
    return result


def adapt_rigid_checkpoint(packet: dict, config: dict, frame: int) -> dict:
    """Copy only metadata dictionaries; all Gaussian, tree and RNG objects stay shared.

    The input checkpoint is never mutated. The strict production restore then
    checks every resulting method/data/trainer field against trial config.
    """
    steps = (config['module']['rigid_tree']['motion_steps'] +
             config['module']['rigid_tree']['birth_steps'])
    if (packet.get('rigid_tree_version') != 1 or
            packet.get('current_frame') != frame - 1 or
            packet.get('global_step') != steps or
            not packet.get('rigid_tree_frame', {}).get('finalized')):
        raise ValueError('Candidate source must be the preceding finalized frame')
    output = packet.copy()
    data = deepcopy(packet['rigid_tree_data_config'])
    trainer = deepcopy(packet['rigid_tree_trainer_config'])
    module = deepcopy(packet['rigid_tree_module_config'])
    tree = deepcopy(packet['rigid_tree_config'])
    motion = packet['rigid_tree_motion'].copy()
    motion['config'] = deepcopy(motion['config'])
    rate = config['module']['rigid_tree']['rate_loss_weight']
    data['eval_train'] = True
    data['eval_test'] = True
    trainer['saving_ckpt_frames'] = [frame]
    trainer['saving_ckpt_steps'] = [steps]
    tree['rate_loss_weight'] = rate
    module['rigid_tree']['rate_loss_weight'] = rate
    motion['config']['rate_loss_weight'] = rate
    output.update(rigid_tree_data_config=data,
                  rigid_tree_trainer_config=trainer,
                  rigid_tree_config=tree,
                  rigid_tree_module_config=module,
                  rigid_tree_motion=motion)
    untouched = set(packet) - {'rigid_tree_data_config',
                               'rigid_tree_trainer_config',
                               'rigid_tree_config',
                               'rigid_tree_module_config',
                               'rigid_tree_motion'}
    if any(output[key] is not packet[key] for key in untouched):
        raise AssertionError('Adapter touched checkpoint state or RNG')
    if any(motion[key] is not packet['rigid_tree_motion'][key]
           for key in set(motion) - {'config'}):
        raise AssertionError('Adapter touched motion tensors or optimizer')
    return output


def train_image_loss(summary: dict, lambda_dssim: float) -> tuple[float, dict]:
    """Read only training metrics; test fields cannot affect selection."""
    train = summary['evaluation']['train']
    l1, ssim = float(train['l1']), float(train['ssim'])
    if not math.isfinite(l1) or not math.isfinite(ssim):
        raise ValueError('Nonfinite train evaluation')
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim), {
        'l1': l1, 'ssim': ssim, 'psnr': float(train['psnr'])}


def select_candidate(candidates: list[dict], budget_exclusive: int,
                     tie_tolerance: float) -> dict | None:
    """Strict budget first, then smallest train image loss, then packet bytes."""
    selected = None
    for candidate in candidates:
        if candidate['encoded_bytes'] >= budget_exclusive:
            continue
        if (selected is None or
                candidate['train_image_loss'] <
                selected['train_image_loss'] - tie_tolerance or
                (abs(candidate['train_image_loss'] -
                     selected['train_image_loss']) <= tie_tolerance and
                 candidate['encoded_bytes'] < selected['encoded_bytes'])):
            selected = candidate
    return selected


def _sources(snapshot: Path):
    if str(snapshot) not in sys.path:
        sys.path.insert(0, str(snapshot))
    return (importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec'),
            importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_birth_codec'),
            importlib.import_module('pipeline.ReCon-GS.model.birth_quantization'))


def _reference(path: Path, snapshot: Path):
    import numpy as np
    import torch

    codec, _, _ = _sources(snapshot)
    with np.load(path, allow_pickle=False) as saved:
        raw = {key: torch.from_numpy(saved[key].copy()) for key in RAW_FIELDS}
        state = codec.DecodedState(
            raw['xyz'], raw['rotation'],
            torch.from_numpy(saved['owner'].copy()).long(),
            torch.from_numpy(saved['parents'].copy()).long(),
            int(saved['frame_index']))
        extent = float(saved['extent'])
        regions = int(saved['region_count'])
    return state, raw, extent, regions


def _checkpoint_raw(packet):
    return {name: packet['gaussians'][index].detach().cpu().float().contiguous()
            for name, index in CAPTURE_INDEX.items()}


def _rng_sha256(packet):
    """Hash RNG values, independent of pickle storage IDs and object identity."""
    import numpy as np
    import torch

    state = packet.get('rng_state')
    if state is None:
        return None
    digest = hashlib.sha256()

    def blob(value):
        digest.update(len(value).to_bytes(8, 'little'))
        digest.update(value)

    def visit(value):
        if value is None:
            digest.update(b'N')
        elif isinstance(value, bool):
            digest.update(b'B1' if value else b'B0')
        elif isinstance(value, int):
            digest.update(b'I')
            blob(str(value).encode('ascii'))
        elif isinstance(value, float):
            digest.update(b'F')
            digest.update(struct.pack('<d', value))
        elif isinstance(value, str):
            digest.update(b'S')
            blob(value.encode('utf8'))
        elif isinstance(value, bytes):
            digest.update(b'Y')
            blob(value)
        elif isinstance(value, torch.Tensor):
            digest.update(b'T')
            tensor = value.detach().cpu().contiguous()
            blob(str(tensor.dtype).encode('ascii'))
            visit(tuple(tensor.shape))
            blob(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(b'A')
            blob(value.dtype.str.encode('ascii'))
            visit(tuple(value.shape))
            blob(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, np.generic):
            digest.update(b'G')
            blob(value.dtype.str.encode('ascii'))
            blob(value.tobytes())
        elif isinstance(value, (tuple, list)):
            digest.update(b'U' if isinstance(value, tuple) else b'L')
            digest.update(len(value).to_bytes(8, 'little'))
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError('RNG dictionary keys must be strings')
            digest.update(b'D')
            digest.update(len(value).to_bytes(8, 'little'))
            for key in sorted(value):
                visit(key)
                visit(value[key])
        else:
            raise TypeError(f'Unsupported RNG state value: {type(value).__name__}')

    visit(state)
    return digest.hexdigest()


def _train_view_ids(snapshot: Path, config: dict, frame: int) -> list[str]:
    data = importlib.import_module('pipeline.ReCon-GS.data')
    data_config = deepcopy(config['data'])
    data_config['shuffle'] = False  # Only enumerate paths, never change trial data.
    module = data.DataModule(data.DataConfig(**data_config))
    module.setup()
    module.current_frame = frame
    dataset = module.train_dataloader(random=False).dataset
    if hasattr(dataset, 'indices'):
        items = [dataset.dataset.items[index] for index in dataset.indices]
    else:
        items = dataset.items
    ids = [str(Path(item.path)) for item in items]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Training view IDs are empty or duplicated')
    return ids


def _trial_result(trial_run: Path, frame: int, rate: float,
                  lambda_dssim: float, source_sha: str,
                  source_rng_sha: str, config_sha: str) -> dict:
    summary_path = trial_run / 'frame_summary' / f'frame_{frame:03d}.json'
    summary = json.loads(summary_path.read_text())
    if summary['frame'] != frame:
        raise ValueError('Trial summary frame differs')
    stream = trial_run / 'frame_storage' / 'rigid_tree'
    motion = stream / f'motion_{frame:03d}.bin'
    birth = stream / f'birth_{frame:03d}.bin'
    manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
    motion_bytes = motion.stat().st_size
    birth_bytes = birth.stat().st_size if birth.exists() else 0
    if (manifest['frame'] != frame or
            manifest['motion'] != motion.name or
            manifest['birth'] != (birth.name if birth.exists() else None) or
            summary['motion']['bytes'] != motion_bytes or
            (summary['birth']['archive_bytes'] if summary['birth'] else 0) != birth_bytes or
            summary['encoded_bytes'] != motion_bytes + birth_bytes):
        raise ValueError('Trial packet sizes or manifest disagree')
    loss, metrics = train_image_loss(summary, lambda_dssim)
    result = dict(frame=frame, rate_loss_weight=rate,
                  trial_run=str(trial_run), trial_config_sha256=config_sha,
                  source_checkpoint_sha256=source_sha,
                  source_rng_sha256=source_rng_sha,
                  train_metrics=metrics, train_image_loss=loss,
                  motion_bytes=motion_bytes, birth_bytes=birth_bytes,
                  encoded_bytes=motion_bytes + birth_bytes,
                  motion_sha256=file_sha256(motion),
                  birth_sha256=file_sha256(birth) if birth.exists() else None,
                  decoded_state_sha256=manifest['decoded_state_sha256'],
                  raw_state_sha256=manifest['raw_state_sha256'],
                  birth_count=manifest['birth_count'],
                  total_count=manifest['total_count'])
    return result


def _same_reference(reference: Path, prior_state, prior_raw: dict,
                    snapshot: Path, expected_extent: float | None,
                    expected_regions: int):
    _, _, birth_quantization = _sources(snapshot)
    state, raw, extent, regions = _reference(reference, snapshot)
    if (state.digest() != prior_state.digest() or
            birth_quantization.reference_digest(raw) !=
            birth_quantization.reference_digest(prior_raw) or
            regions != expected_regions or
            (expected_extent is not None and extent != expected_extent)):
        raise ValueError('Trial reference differs from selected decoded prefix')
    return extent


def _decode_trial_frame(trial_stream: Path, frame: int, prior_state,
                        prior_raw: dict, snapshot: Path,
                        expected_reference: str | None = None):
    import torch

    codec, birth_codec, birth_quantization = _sources(snapshot)
    motion_path = trial_stream / f'motion_{frame:03d}.bin'
    decoded = codec.decode_motion(motion_path.read_bytes(), prior_state)
    old_count = len(prior_state.xyz)
    if len(decoded.xyz) < old_count:
        raise ValueError('Selected stream has invalid old GS layout')
    birth_count = len(decoded.xyz) - old_count
    birth_path = trial_stream / f'birth_{frame:03d}.bin'
    raw = {name: value.clone() for name, value in prior_raw.items()}
    raw['xyz'] = decoded.xyz[:old_count]
    raw['rotation'] = decoded.rotation[:old_count]
    if birth_count:
        if not birth_path.is_file():
            raise ValueError('Selected motion has births without appearance packet')
        sources = decoded.birth_sources
        if (sources is None or len(sources) != birth_count or
                not torch.equal(decoded.owner[old_count:],
                                decoded.owner[:old_count][sources])):
            raise ValueError('Birth source/owner relation differs')
        newborn = birth_codec.decode_birth_attributes(
            birth_path.read_bytes(), raw, decoded.xyz[old_count:],
            decoded.rotation[old_count:], sources)
        for name in RAW_FIELDS:
            raw[name] = torch.cat((raw[name], newborn[name]))
    elif birth_path.exists():
        raise ValueError('Empty birth frame has an unexpected packet')
    if (not torch.equal(raw['xyz'], decoded.xyz) or
            not torch.equal(raw['rotation'], decoded.rotation)):
        raise ValueError('Decoded birth pose differs from motion packet')
    manifest = json.loads((trial_stream / f'frame_{frame:03d}.json').read_text())
    raw_digest = birth_quantization.reference_digest(raw)
    if expected_reference is None:
        expected_reference = f'frame_storage/rigid_tree/reference_frame{frame - 1}.npz'
    if (manifest['frame'] != frame or
            manifest['reference'] != expected_reference or
            manifest['old_count'] != old_count or
            manifest['birth_count'] != birth_count or
            manifest['total_count'] != len(decoded.xyz) or
            manifest['decoded_state_sha256'] != decoded.digest().hex() or
            manifest['raw_state_sha256'] != raw_digest):
        raise ValueError('Selected trial manifest does not match formal replay')
    return decoded, raw, manifest, raw_digest


def append_selected_frame(run: Path, trial_run: Path, frame: int,
                          prior_state, prior_raw: dict, snapshot: Path,
                          expected_extent: float | None,
                          expected_regions: int, rate: float):
    """Verify from the selected prefix, then copy one complete frame atomically."""
    trial_stream = trial_run / 'frame_storage' / 'rigid_tree'
    trial_reference = trial_stream / f'reference_frame{frame - 1}.npz'
    extent = _same_reference(trial_reference, prior_state, prior_raw,
                             snapshot, expected_extent, expected_regions)
    decoded, raw, manifest, raw_digest = _decode_trial_frame(
        trial_stream, frame, prior_state, prior_raw, snapshot)
    source_summary = json.loads((trial_run / 'frame_summary' /
                                 f'frame_{frame:03d}.json').read_text())
    motion_name = f'motion_{frame:03d}.bin'
    birth_name = f'birth_{frame:03d}.bin'
    source_motion_bytes = (trial_stream / motion_name).stat().st_size
    source_birth_bytes = ((trial_stream / birth_name).stat().st_size
                          if manifest['birth'] is not None else 0)
    if (source_summary['frame'] != frame or
            source_summary['materialized_sha256'] != raw_digest or
            source_summary['motion']['bytes'] != source_motion_bytes or
            (source_summary['birth']['archive_bytes']
             if source_summary['birth'] else 0) != source_birth_bytes or
            (source_summary['birth'] is None) != (manifest['birth'] is None) or
            source_summary['encoded_bytes'] !=
            source_motion_bytes + source_birth_bytes):
        raise ValueError('Selected summary differs from formal replay and files')
    stream = run / 'frame_storage' / 'rigid_tree'
    summaries = run / 'frame_summary'
    stream.mkdir(parents=True, exist_ok=True)
    summaries.mkdir(parents=True, exist_ok=True)
    aggregate_reference = run / REFERENCE
    if frame == 1:
        if aggregate_reference.exists():
            raise FileExistsError(aggregate_reference)
        shutil.copyfile(trial_reference, aggregate_reference)
    elif not aggregate_reference.is_file():
        raise FileNotFoundError(aggregate_reference)
    for name in (f'motion_{frame:03d}.bin', f'birth_{frame:03d}.bin',
                 f'frame_{frame:03d}.json'):
        target = stream / name
        if target.exists():
            raise FileExistsError(target)
    target_summary = summaries / f'frame_{frame:03d}.json'
    if target_summary.exists():
        raise FileExistsError(target_summary)
    shutil.copyfile(trial_stream / motion_name, stream / motion_name)
    if manifest['birth'] is not None:
        shutil.copyfile(trial_stream / birth_name, stream / birth_name)
    manifest = deepcopy(manifest)
    manifest['reference'] = REFERENCE
    (stream / f'frame_{frame:03d}.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    summary = deepcopy(source_summary)
    summary['motion']['path'] = str((stream / motion_name).resolve())
    if summary['birth'] is not None:
        summary['birth']['path'] = str((stream / birth_name).resolve())
    summary['shared_reference_bytes'] = aggregate_reference.stat().st_size
    summary['decoder_reference'] = REFERENCE
    summary['frame_manifest'] = f'frame_storage/rigid_tree/frame_{frame:03d}.json'
    summary['budget_selected_rate_loss_weight'] = rate
    motion_bytes = (stream / motion_name).stat().st_size
    birth_bytes = (stream / birth_name).stat().st_size if summary['birth'] else 0
    if motion_bytes != source_motion_bytes or birth_bytes != source_birth_bytes:
        raise ValueError('Copied packet sizes changed')
    target_summary.write_text(json.dumps(summary, indent=2) + '\n')
    copied_state, copied_raw, _, copied_digest = _decode_trial_frame(
        stream, frame, prior_state, prior_raw, snapshot, REFERENCE)
    if copied_state.digest() != decoded.digest() or copied_digest != raw_digest:
        raise ValueError('Copied selected frame differs from verified trial')
    return copied_state, copied_raw, extent, dict(
        decoded_state_sha256=copied_state.digest().hex(),
        raw_state_sha256=copied_digest, motion_bytes=motion_bytes,
        birth_bytes=birth_bytes, encoded_bytes=motion_bytes + birth_bytes,
        reference_bytes=aggregate_reference.stat().st_size)


def _assert_selected_checkpoint(checkpoint: Path, frame: int,
                                decoded_state, raw: dict, snapshot: Path,
                                total_steps: int):
    import torch

    codec, _, birth_quantization = _sources(snapshot)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved.get('current_frame') != frame or
            saved.get('global_step') != total_steps or
            not saved.get('rigid_tree_frame', {}).get('finalized')):
        raise ValueError('Selected final checkpoint has wrong frame or stage')
    raw_digest = birth_quantization.reference_digest(_checkpoint_raw(saved))
    state = codec.DecodedState.restore(
        saved['rigid_tree_motion']['codec_previous_state'])
    if (raw_digest != birth_quantization.reference_digest(raw) or
            state.digest() != decoded_state.digest()):
        raise ValueError('Selected checkpoint differs from formal decoded stream')
    rng_digest = _rng_sha256(saved)
    if rng_digest is None:
        raise ValueError('Selected rigid checkpoint lost RNG state')
    return file_sha256(checkpoint), rng_digest


def _status(path: Path, **fields):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(fields, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def _trial_run_dir(run: Path, name: str) -> Path:
    parent = run / 'trials' / name
    children = [path for path in parent.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise ValueError(f'Trial must have one output directory: {parent}')
    return children[0]


def _run_trial(run: Path, snapshot: Path, base: dict, plan: dict,
               frame: int, rate: float, ordinal: int, source_checkpoint: Path,
               source_packet: dict | None, source_sha: str,
               source_rng_sha: str | None) -> tuple[dict, dict]:
    import torch

    name = f'frame_{frame:03d}_rate_{ordinal:02d}'
    trial_parent = run / 'trials' / name
    if trial_parent.exists():
        raise FileExistsError(trial_parent)
    config_path = run / 'trial_configs' / f'{name}.yaml'
    adapter = run / 'adapters' / f'{name}.pth'
    trial = trial_config(base, rate, frame, plan,
                         source_checkpoint if frame == 1 else adapter,
                         name, run)
    if frame > 1:
        adapted = adapt_rigid_checkpoint(source_packet, trial, frame)
        torch.save(adapted, adapter)
    config_path.write_text(yaml.safe_dump(trial, sort_keys=False))
    log_path = run / 'trial_logs' / f'{name}.log'
    frozen_driver = run / 'budget_driver.py'
    if file_sha256(frozen_driver) != plan['driver_sha256']:
        raise ValueError('Frozen budget worker changed during the run')
    command = [sys.executable, str(frozen_driver),
               '--trial-config', str(config_path), '--trial-frame', str(frame),
               '--source-snapshot', str(snapshot)]
    print(f'Frame {frame}: rate {rate:g}, trial {name}', flush=True)
    try:
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=snapshot, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise BudgetError(f'Trial {name} failed ({result.returncode}); see {log_path}')
        trial_run = _trial_run_dir(run, name)
        record = _trial_result(
            trial_run, frame, rate, base['module']['lambda_dssim'],
            source_sha, source_rng_sha, file_sha256(config_path))
        record['trial_log'] = str(log_path)
        record['trial_config'] = str(config_path)
        record['final_checkpoint'] = str(
            trial_run / f'rigid-ckpt-{frame}-{trial["trainer"]["saving_ckpt_steps"][0]}.pth')
        if not Path(record['final_checkpoint']).is_file():
            raise FileNotFoundError(record['final_checkpoint'])
        return record, trial
    finally:
        if adapter.exists():
            adapter.unlink()


def _create_run(plan: dict, base: dict) -> Path:
    parent = Path(plan['output_dir']) / plan['experiment_name']
    if parent.is_symlink():
        raise ValueError('Budget experiment parent cannot be a symlink')
    run = parent / (datetime.now().strftime('%Y%m%dT%H%M%S') +
                    '_budget_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    for name in ('adapters', 'trial_configs', 'trial_logs', 'trials'):
        (run / name).mkdir()
    shutil.copyfile(Path(__file__).resolve(), run / 'budget_driver.py')
    plan['driver_sha256'] = file_sha256(run / 'budget_driver.py')
    (run / 'budget_config.yaml').write_text(yaml.safe_dump(
        {key: value for key, value in plan.items() if not key.endswith('sha256')
         and key != 'source_commit'}, sort_keys=False))
    (run / 'budget_preflight.json').write_text(json.dumps(dict(
        protocol='rigid_tree_actual_packet_budget_v1',
        source_commit=plan['source_commit'],
        source_manifest_sha256=plan['source_manifest_sha256'],
        frozen_driver_sha256=plan['driver_sha256'],
        base_config_sha256=plan['base_config_sha256'],
        budget_config_sha256=plan['budget_config_sha256'],
        data_num_frames=base['data']['extra_dataset_kwargs']['num_frames'],
        single_frame_worker='Same frozen Pipeline and Trainer setup, then one '
                            'Trainer.frame_training_loop call; the worker reproduces '
                            'the native or finalized-rigid branch of '
                            'Trainer.training_loop before that call.',
        native_rng='The shared native checkpoint has no RNG snapshot; each frame-1 '
                   'trial starts in a fresh subprocess with base random_seed. '
                   'Native checkpoint SHA and frame-0 decoded/raw references '
                   'are compared; bitwise training determinism is not assumed.',
        rigid_rng='Every later trial restores the same selected predecessor '
                  'checkpoint RNG before begin_next_frame.',
        selection='Only mean L1 and SSIM over every training view; test metrics '
                  'are generated for benchmark compatibility but ignored by selector.'
    ), indent=2) + '\n')
    aggregate = deepcopy(base)
    aggregate['experiment_name'] = plan['experiment_name']
    aggregate['data']['eval_train'] = True
    aggregate['data']['eval_test'] = True
    (run / 'config.yaml').write_text(
        '# Render/benchmark config. Per-frame selected rates are in budget_decisions.jsonl.\n'
        + yaml.safe_dump(aggregate, sort_keys=False))
    _status(run / 'budget_status.json', status='running',
            last_selected_frame=0, expected_last_frame=plan['total_frames'] - 1,
            source_commit=plan['source_commit'],
            source_manifest_sha256=plan['source_manifest_sha256'],
            frozen_driver_sha256=plan['driver_sha256'],
            base_config_sha256=plan['base_config_sha256'],
            budget_config_sha256=plan['budget_config_sha256'])
    return run


def run_budget(plan: dict, base: dict, through: int) -> Path:
    import torch

    snapshot = Path(plan['source_snapshot'])
    codec, _, birth_quantization = _sources(snapshot)
    run = _create_run(plan, base)
    total_steps = (base['module']['rigid_tree']['motion_steps'] +
                   base['module']['rigid_tree']['birth_steps'])
    source_checkpoint = Path(base['ckpt_path'])
    selected_state = selected_raw = None
    extent = None
    region_count = base['module']['rigid_tree']['region_count']
    decisions = run / 'budget_decisions.jsonl'
    last_selected_frame = 0
    try:
        for frame in range(1, through + 1):
            source_sha = file_sha256(source_checkpoint)
            source_packet = torch.load(source_checkpoint, map_location='cpu',
                                       weights_only=False)
            source_rng_sha = _rng_sha256(source_packet)
            if frame > 1 and source_rng_sha is None:
                raise ValueError('Selected rigid source checkpoint lacks RNG state')
            candidates = []
            view_ids = None
            for rates, fallback in ((plan['rate_candidates'], False),
                                    (plan['fallback_rate_candidates'], True)):
                if fallback and select_candidate(
                        candidates, plan['budget_bytes_exclusive'],
                        plan['train_loss_tie_tolerance']) is not None:
                    break
                for rate in rates:
                    ordinal = len(candidates)
                    record, trial = _run_trial(
                        run, snapshot, base, plan, frame, rate, ordinal,
                        source_checkpoint, source_packet, source_sha,
                        source_rng_sha)
                    trial_run = Path(record['trial_run'])
                    trial_ref = (trial_run / 'frame_storage' / 'rigid_tree' /
                                 f'reference_frame{frame - 1}.npz')
                    if selected_state is None:
                        selected_state, selected_raw, extent, initial_regions = _reference(
                            trial_ref, snapshot)
                        if (frame != 1 or selected_state.frame_index != 0 or
                                initial_regions != region_count):
                            raise ValueError('Native first-frame bootstrap differs')
                    else:
                        _same_reference(trial_ref, selected_state, selected_raw,
                                        snapshot, extent, region_count)
                    record['source_reference_sha256'] = file_sha256(trial_ref)
                    record['source_decoded_state_sha256'] = selected_state.digest().hex()
                    record['source_raw_state_sha256'] = (
                        birth_quantization.reference_digest(selected_raw))
                    verified_state, verified_raw, _, verified_raw_sha = _decode_trial_frame(
                        trial_run / 'frame_storage' / 'rigid_tree', frame,
                        selected_state, selected_raw, snapshot)
                    checkpoint_sha, end_rng_sha = _assert_selected_checkpoint(
                        Path(record['final_checkpoint']), frame, verified_state,
                        verified_raw, snapshot, total_steps)
                    if (record['decoded_state_sha256'] != verified_state.digest().hex() or
                            record['raw_state_sha256'] != verified_raw_sha):
                        raise ValueError('Candidate recorded state differs from replay')
                    record['final_checkpoint_sha256'] = checkpoint_sha
                    record['final_rng_sha256'] = end_rng_sha
                    if view_ids is None:
                        view_ids = _train_view_ids(snapshot, trial, frame)
                    record['feasible'] = (
                        record['encoded_bytes'] < plan['budget_bytes_exclusive'])
                    record['fallback'] = fallback
                    candidates.append(record)
            selected = select_candidate(
                candidates, plan['budget_bytes_exclusive'],
                plan['train_loss_tie_tolerance'])
            if selected is None:
                raise BudgetError(f'Frame {frame}: no candidate meets strict '
                                  f'{plan["budget_bytes_exclusive"]}-byte budget')
            prior_state_sha = selected_state.digest().hex()
            prior_raw_sha = birth_quantization.reference_digest(selected_raw)
            selected_run = Path(selected['trial_run'])
            selected_state, selected_raw, extent, copied = append_selected_frame(
                run, selected_run, frame, selected_state, selected_raw,
                snapshot, extent, region_count, selected['rate_loss_weight'])
            if copied['encoded_bytes'] >= plan['budget_bytes_exclusive']:
                raise AssertionError('Selected copied packets exceed budget')
            next_checkpoint = Path(selected['final_checkpoint'])
            final_checkpoint_sha, final_rng_sha = _assert_selected_checkpoint(
                next_checkpoint, frame, selected_state, selected_raw,
                snapshot, total_steps)
            if not plan['retain_unselected_checkpoints']:
                for candidate in candidates:
                    if candidate is not selected:
                        Path(candidate['final_checkpoint']).unlink()
                        candidate['checkpoint_retained'] = False
                    else:
                        candidate['checkpoint_retained'] = True
            else:
                for candidate in candidates:
                    candidate['checkpoint_retained'] = True
            decision = dict(frame=frame, prior_state_sha256=prior_state_sha,
                            prior_raw_sha256=prior_raw_sha,
                            prior_checkpoint=str(source_checkpoint),
                            prior_checkpoint_sha256=source_sha,
                            prior_rng_sha256=source_rng_sha,
                            prior_rng_status=('saved_checkpoint' if source_rng_sha
                                              else 'unavailable_native_checkpoint'),
                            native_bootstrap_seed=(base['random_seed']
                                                   if frame == 1 else None),
                            train_view_ids=view_ids,
                            train_view_ids_sha256=hashlib.sha256(
                                json.dumps(view_ids).encode()).hexdigest(),
                            candidates=candidates,
                            selected_trial_run=selected['trial_run'],
                            selected_rate_loss_weight=selected['rate_loss_weight'],
                            selection_reason='minimum full-train-view image loss among '
                                             'actual complete packets below strict budget; '
                                             'smaller packet wins ties',
                            selected_packet=copied,
                            selected_checkpoint_sha256=final_checkpoint_sha,
                            selected_rng_sha256=final_rng_sha)
            with decisions.open('a') as stream:
                stream.write(json.dumps(decision, sort_keys=True) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            last_selected_frame = frame
            source_checkpoint = next_checkpoint
            _status(run / 'budget_status.json', status='running',
                    last_selected_frame=frame,
                    expected_last_frame=plan['total_frames'] - 1,
                    source_commit=plan['source_commit'],
                    source_manifest_sha256=plan['source_manifest_sha256'],
                    frozen_driver_sha256=plan['driver_sha256'],
                    base_config_sha256=plan['base_config_sha256'],
                    budget_config_sha256=plan['budget_config_sha256'],
                    selected_decoded_state_sha256=selected_state.digest().hex(),
                    selected_raw_state_sha256=
                    birth_quantization.reference_digest(selected_raw))
            print(f'Frame {frame} selected rate '
                  f'{selected["rate_loss_weight"]:g}, '
                  f'{copied["encoded_bytes"]} bytes, '
                  f'train image loss {selected["train_image_loss"]:.8f}',
                  flush=True)
        final_status = ('complete' if through == plan['total_frames'] - 1
                        else 'partial')
        _status(run / 'budget_status.json', status=final_status,
                last_selected_frame=through,
                expected_last_frame=plan['total_frames'] - 1,
                source_commit=plan['source_commit'],
                source_manifest_sha256=plan['source_manifest_sha256'],
                frozen_driver_sha256=plan['driver_sha256'],
                base_config_sha256=plan['base_config_sha256'],
                budget_config_sha256=plan['budget_config_sha256'],
                selected_decoded_state_sha256=selected_state.digest().hex(),
                selected_raw_state_sha256=
                birth_quantization.reference_digest(selected_raw))
        return run
    except Exception as error:
        _status(run / 'budget_status.json', status='failed',
                last_selected_frame=last_selected_frame,
                expected_last_frame=plan['total_frames'] - 1,
                failure=str(error))
        raise


def run_single_trial(config_path: Path, snapshot: Path, frame: int):
    """Run exactly one existing Trainer frame with the full 300-frame scene.

    This reproduces the native/rigid transition in Trainer.training_loop, then
    invokes its unchanged frame_training_loop. Keeping all 300 scene frames
    preserves the exact camera extent used by the full-sequence baseline.
    """
    from omegaconf import OmegaConf
    from libgs.pipeline.loader import load_pipeline
    from libgs.utils.config import to_structured

    snapshot = snapshot.resolve(strict=True)
    if str(snapshot) not in sys.path:
        sys.path.insert(0, str(snapshot))
    Config, Pipeline = load_pipeline('ReCon-GS')
    config, extra = to_structured(OmegaConf.load(config_path), Config)
    if extra or config.data.extra_dataset_kwargs.get('num_frames') != 300:
        raise ValueError('Trial must use the complete canonical 300-frame scene')
    if config.mode != 'train' or config.gui_enabled:
        raise ValueError('Budget trial needs non-GUI training mode')
    pipeline = Pipeline(config)
    trainer = pipeline.trainer
    trainer.setup(pipeline.module, ckpt_path=config.ckpt_path)
    if frame == 1:
        if trainer.current_frame != 0 or trainer.global_step != trainer.config.num_init_steps:
            raise ValueError('First trial must start from completed native frame 0')
        trainer.global_step = 0
    else:
        if (trainer.current_frame != frame - 1 or
                trainer.global_step != trainer.num_steps or
                not pipeline.module.frame_finalized):
            raise ValueError('Trial must start from preceding finalized rigid frame')
        trainer._restore_rng(trainer._resume_rng)
        trainer._resume_rng = None
        pipeline.module.begin_next_frame()
        trainer.global_step = 0
    trainer.datamodule.current_frame = frame
    trainer.frame_training_loop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--through', type=int,
                        help='Inclusive last dynamic frame. Omit for full 300.')
    parser.add_argument('--trial-config', type=Path,
                        help=argparse.SUPPRESS)
    parser.add_argument('--trial-frame', type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--source-snapshot', type=Path,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.trial_config is not None:
        if args.trial_frame is None or args.source_snapshot is None:
            parser.error('Trial invocation needs frame and frozen source')
        run_single_trial(args.trial_config, args.source_snapshot,
                         args.trial_frame)
        return
    plan, base = load_budget_config(args.config)
    through = plan['total_frames'] - 1 if args.through is None else args.through
    if not 1 <= through < plan['total_frames']:
        parser.error('--through must be in 1..299')
    run = run_budget(plan, base, through)
    print(f'Budget sequence {run}: '
          f'{"complete" if through == plan["total_frames"] - 1 else "partial"}',
          flush=True)


if __name__ == '__main__':
    main()
