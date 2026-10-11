"""Summarize completed runs and verify their deployment packet inventory on CPU."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import statistics
import sys
import zlib

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CC = importlib.import_module('pipeline.ReCon-GS.archive.model.common_contrast')
BIRTH = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_codec')
COLOR = importlib.import_module('pipeline.ReCon-GS.archive.model.sparse_color')
OPACITY = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_lifetime')
Q = importlib.import_module('pipeline.ReCon-GS.archive.model.residual_motion')


def check_side_packet(path, codec, base_count, width):
    packet = path.read_bytes()
    magic, count, rows, step, length, checksum = codec.HEADER.unpack_from(packet)
    if magic != codec.MAGIC or count != base_count or rows > count or length != rows * (4 + 2 * width):
        raise ValueError(f'Invalid side packet dimensions: {path}')
    decompressor = zlib.decompressobj()
    body = decompressor.decompress(packet[codec.HEADER.size:], length + 1)
    if len(body) != length or not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail or zlib.crc32(body) != checksum:
        raise ValueError(f'Invalid side packet integrity: {path}')
    ids = np.frombuffer(body[:rows * 4], dtype='<u4')
    if np.any(ids >= count) or np.any(ids[1:] <= ids[:-1]) or not np.isfinite(step) or step <= 0:
        raise ValueError(f'Invalid side packet row IDs or step: {path}')
    return rows


def summarize(rows):
    return {key: statistics.mean(row[key] for row in rows) for key in (
        'psnr', 'ssim', 'lpips', 'bytes', 'motion_bytes', 'birth_bytes', 'color_bytes',
        'opacity_bytes', 'new', 'deleted', 'training_seconds', 'wall_seconds', 'export_seconds')}


def read_run(path, frames):
    rows = []
    previous = None
    previous_count = None
    config = yaml.safe_load((path / 'config.yaml').read_text())
    rest_expected = (config['module']['sh_degree'] + 1) ** 2 - 1
    archive = path / 'first_reference.npz'
    if archive.exists():
        with np.load(archive, allow_pickle=False) as initial:
            raw = {name: torch.from_numpy(initial[name].copy()) for name in Q.RAW_FIELDS}
        previous, previous_count = Q.reference_digest(raw), len(raw['xyz'])
        rest_expected = raw['sh_rest'].shape[1]
    inventory = dict(motion=0, birth=0, color=0, opacity=0)
    for frame in range(1, frames + 1):
        report = json.loads((path / 'frame_summary' / f'frame_{frame:03d}.json').read_text())
        stream = report['stream']
        folder = path / 'frame_storage' / 'common_contrast_binary'
        header = CC.CommonContrastMotion.read_header(folder / f'motion_{frame:03d}.bin')
        if (header['frame'] != frame or (previous is not None and header['previous_digest'] != previous)
                or (previous_count is not None and header['previous_count'] != previous_count)
                or header['base_count'] != header['previous_count'] - header['deleted_count']):
            raise ValueError(f'Header count or recorded digest chain differs at frame {frame}')
        inventory['motion'] += 1
        birth = folder / f'incremental_{frame:03d}.bin'
        if header['birth_count']:
            # The decoder only needs base shape/device/dtype to validate births.
            base = {name: torch.empty(0) for name in Q.RAW_FIELDS}
            base['xyz'] = torch.empty(header['base_count'], 0)
            base['sh_rest'] = torch.empty(0, rest_expected, 3)
            newborn, _ = BIRTH.decode_birth_packet(birth.read_bytes(), base)
            if len(newborn['xyz']) != header['birth_count']:
                raise ValueError('Birth count differs from motion header')
            inventory['birth'] += 1
        sizes = dict(motion_bytes=(folder / f'motion_{frame:03d}.bin').stat().st_size,
                     birth_bytes=birth.stat().st_size if birth.exists() else 0)
        for name, codec, width in [('color', COLOR, 3), ('opacity', OPACITY, 1)]:
            side = folder / f'{"appearance" if name == "color" else name}_{frame:03d}.bin'
            present = header['appearance_present' if name == 'color' else 'opacity_present']
            if side.exists() != present:
                raise ValueError(f'Side packet presence differs: {side}')
            if present:
                check_side_packet(side, codec, header['base_count'], width)
                inventory[name] += 1
            sizes[f'{name}_bytes'] = side.stat().st_size if present else 0
        actual = sum(sizes.values())
        if actual != stream['inference_file_bytes'] or actual != stream['hard_storage_bytes']:
            raise ValueError(f'Storage summary differs from files at frame {frame}')
        if (header['birth_count'], header['deleted_count']) != (stream['new_count'], stream['deleted_count']):
            raise ValueError('Header birth/deletion counts differ from summary')
        control = stream['birth_control']
        quality = {name: report['quantized_evaluation']['test'][name]
                   for name in ('l1', 'psnr', 'ssim', 'lpips')}
        rows.append(dict(frame=frame, **quality, **sizes,
                         bytes=actual, new=stream['new_count'], deleted=stream['deleted_count'],
                         training_seconds=report['timing']['training'], wall_seconds=report['timing']['wall'],
                         export_seconds=stream['stage_seconds']['stream_export'],
                         origins=control['point_origins'],
                         expired=(control.get('deletion_protection') or {}).get('expired_births', 0),
                         moved_old=stream['actual_motion']['any_pose_changed_count']))
        previous = stream['reference_sha256']
        previous_count = header['base_count'] + header['birth_count']
    initial_bytes = archive.stat().st_size if archive.exists() else None
    dynamic_bytes = sum(row['bytes'] for row in rows)
    periodic_gap = [rows[i-1]['psnr'] - (rows[i-2]['psnr'] + rows[i]['psnr']) / 2
                    for i in range(5, frames, 5)]
    return dict(run=str(path.resolve()), frames=rows, mean=summarize(rows),
                first63=summarize(rows[:63]), last50=summarize(rows[-50:]),
                blocks={f'{start}-{min(start+59, frames)}': summarize(rows[start-1:start+59])
                        for start in range(1, frames+1, 60)},
                periodic_neighbor_gap=statistics.mean(periodic_gap),
                initial_bytes=initial_bytes, dynamic_bytes=dynamic_bytes,
                video_bytes=dynamic_bytes + initial_bytes if initial_bytes is not None else None,
                packet_inventory=inventory, packet_inventory_validated=True,
                validation_scope='CRC, lengths, birth attributes/attachment bounds, side row bounds, '
                                 'header counts, recorded digest chain and actual necessary file sizes. '
                                 'Does not reconstruct motion, validate true decoded states or measure FPS.',
                config_sha256=hashlib.sha256((path / 'config.yaml').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='LABEL=RUN_DIRECTORY')
    parser.add_argument('--frames', type=int, default=299)
    parser.add_argument('--native-log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(schema='common_contrast_completed_comparison_v1', runs={})
    for item in args.run:
        label, path = item.split('=', 1)
        if label in result['runs']:
            raise ValueError('Run labels must be unique')
        result['runs'][label] = read_run(Path(path), args.frames)
    if args.native_log:
        rows = []
        for frame, section in re.findall(r'==> Start learning frame (\d+) \.\.\.(.*?)(?===> Start learning frame|\Z)', args.native_log.read_text(), re.S):
            values = re.findall(r'Evaluate test dataset at step 200:\s+L1: [0-9.e+-]+ \| PSNR: ([0-9.e+-]+) \| SSIM: ([0-9.e+-]+)', section)
            if int(frame) and values:
                rows.append(dict(frame=int(frame), psnr=float(values[-1][0]), ssim=float(values[-1][1])))
        if [row['frame'] for row in rows] != list(range(1, args.frames+1)):
            raise ValueError('Native reference frame coverage differs')
        result['native'] = dict(log=str(args.native_log.resolve()), frames=rows,
                                mean_psnr=statistics.mean(row['psnr'] for row in rows),
                                last50_psnr=statistics.mean(row['psnr'] for row in rows[-50:]),
                                semantics='Historical trend reference, differing representation/training budgets. '
                                          'Original asynchronous FPS is excluded.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({label: row['mean'] for label, row in result['runs'].items()}, indent=2))


if __name__ == '__main__':
    main()
