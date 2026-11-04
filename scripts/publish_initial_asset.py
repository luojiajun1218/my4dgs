"""Publish a verified fixed-15k checkpoint/layout pair using only symlinks."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
import yaml


PROTOCOL_STEPS = 15_000
MIN_GAUSSIANS = 100_000
MAX_GAUSSIANS = 200_000
MAX_SH_DEGREE = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if int(checkpoint.get('global_step', -1)) != PROTOCOL_STEPS:
        raise ValueError('Initial checkpoint must be exactly 15000 steps')
    if int(checkpoint.get('current_frame', -1)) != 0:
        raise ValueError('Initial checkpoint must be frame0')
    values = checkpoint.get('gaussians')
    if not isinstance(values, (tuple, list)) or len(values) < 7:
        raise ValueError('Initial checkpoint has no native Gaussian capture')
    count = int(values[1].shape[0])
    if not MIN_GAUSSIANS <= count <= MAX_GAUSSIANS:
        raise ValueError('Initial checkpoint must contain 100k–200k GS')
    if any(not torch.is_tensor(values[index]) or len(values[index]) != count
           for index in range(1, 7)):
        raise ValueError('Initial checkpoint Gaussian attributes have different row counts')
    coefficients = int(values[2].shape[1] + values[3].shape[1])
    side = math.isqrt(coefficients)
    if side * side != coefficients:
        raise ValueError('Initial checkpoint has an invalid SH coefficient count')
    active_degree, max_degree = int(values[0]), side - 1
    if (not 0 <= max_degree <= MAX_SH_DEGREE or
            active_degree != max_degree):
        raise ValueError(
            'Initial checkpoint must have all coefficients active through SH0-SH3')
    return dict(global_step=PROTOCOL_STEPS, current_frame=0,
                original_gaussians=count, active_sh_degree=active_degree,
                max_sh_degree=max_degree)


def _reference_metadata(path: Path, original_count: int,
                        sh_degree: int) -> dict:
    with np.load(path, allow_pickle=False) as packet:
        required = {'xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest',
                    'owner', 'frame_index', 'initial_original_count',
                    'initial_retained_count', 'initial_removed_count'}
        missing = required.difference(packet.files)
        if missing:
            raise ValueError(f'Reference layout is missing fields: {sorted(missing)}')
        if int(packet['frame_index']) != 0:
            raise ValueError('Reference layout must describe frame0')
        retained = int(packet['initial_retained_count'])
        recorded_original = int(packet['initial_original_count'])
        removed = int(packet['initial_removed_count'])
        if recorded_original != original_count or retained + removed != original_count:
            raise ValueError('Reference layout does not match checkpoint population')
        if not MIN_GAUSSIANS <= retained <= MAX_GAUSSIANS:
            raise ValueError('Published reference must retain 100k–200k GS')
        for name in ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc',
                     'sh_rest', 'owner'):
            if len(packet[name]) != retained:
                raise ValueError(f'Reference field {name} has the wrong row count')
        expected_shapes = {
            'xyz': (retained, 3),
            'rotation': (retained, 4),
            'logscale': (retained, 3),
            'opacity': (retained, 1),
            'sh_dc': (retained, 1, 3),
            'sh_rest': (retained, (sh_degree + 1) ** 2 - 1, 3),
            'owner': (retained,),
        }
        for name, shape in expected_shapes.items():
            if packet[name].shape != shape:
                raise ValueError(
                    f'Reference field {name} has shape {packet[name].shape}, expected {shape}')
            if name != 'owner' and not np.isfinite(packet[name]).all():
                raise ValueError(f'Reference field {name} contains nonfinite values')
        region_count = int(packet['region_count']) if 'region_count' in packet else None
        threshold = (float(packet['initial_opacity_threshold'])
                     if 'initial_opacity_threshold' in packet else None)
    return dict(retained_gaussians=retained, removed_gaussians=removed,
                region_count=region_count, initial_opacity_threshold=threshold)


def publish(checkpoint: Path, layout_run: Path, output: Path) -> dict:
    checkpoint, layout_run = checkpoint.resolve(), layout_run.resolve()
    output = output.absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path = layout_run / 'config.yaml'
    reference = layout_run / 'frame_storage/rigid_tree/reference_frame0.npz'
    if not config_path.is_file() or not reference.is_file():
        raise FileNotFoundError('Layout run needs config.yaml and reference_frame0.npz')
    if os.path.lexists(output):
        raise FileExistsError(output)

    checkpoint_report = _checkpoint_metadata(checkpoint)
    config = yaml.safe_load(config_path.read_text())
    configured_checkpoint = Path(config.get('ckpt_path', '')).expanduser().resolve()
    if configured_checkpoint != checkpoint:
        raise ValueError('Layout run was not generated from the supplied checkpoint')
    if int(config['trainer']['num_init_steps']) != PROTOCOL_STEPS:
        raise ValueError('Layout run does not use the fixed 15000-step protocol')
    if int(config['data']['extra_dataset_kwargs']['num_frames']) != 1:
        raise ValueError('Layout run must contain frame0 only')
    configured_degree = int(config['module']['sh_degree'])
    if configured_degree != checkpoint_report['max_sh_degree']:
        raise ValueError('Layout SH degree does not match the supplied checkpoint')
    gate = config['module']['rigid_tree']['spatial_gate']
    if gate.get('reference_layout') is not None:
        raise ValueError('Layout run must build a fresh spatial layout')
    reference_report = _reference_metadata(
        reference, checkpoint_report['original_gaussians'], configured_degree)

    report = dict(schema='verified_initial_asset_v1', link_only=True,
        checkpoint_source=str(checkpoint), layout_run=str(layout_run),
        reference_source=str(reference), checkpoint_sha256=_sha256(checkpoint),
        reference_sha256=_sha256(reference), **checkpoint_report, **reference_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    try:
        (temporary / 'ckpt-0-15000.pth').symlink_to(checkpoint)
        (temporary / 'reference_frame0.npz').symlink_to(reference)
        (temporary / 'layout_config.yaml').symlink_to(config_path.resolve())
        (temporary / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--layout-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.checkpoint, args.layout_run, args.output), indent=2))


if __name__ == '__main__':
    main()
