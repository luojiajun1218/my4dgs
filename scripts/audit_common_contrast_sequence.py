"""Replay a completed common/contrast sequence through its real decoder."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import os
import subprocess

import torch
import yaml
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CC = importlib.import_module('pipeline.ReCon-GS.model.common_contrast')
SAB = importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary')
Q = importlib.import_module('pipeline.ReCon-GS.model.residual_motion')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=99)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--initial-model', type=Path, default=None)
    parser.add_argument('--profile', action='store_true', help='Synchronize and time actual decoder calls')
    args = parser.parse_args()
    run = args.run.resolve()
    device = torch.device(args.device)
    config = yaml.safe_load((run / 'config.yaml').read_text())
    motion_config = CC.CommonContrastConfig(**config['module']['common_contrast_binary'])
    steps = int(config['trainer']['num_incr_steps'])
    if args.initial_model is not None:
        with np.load(args.initial_model, allow_pickle=False) as initial:
            raw = {name: torch.from_numpy(initial[name].copy()).to(device) for name in Q.RAW_FIELDS}
        bootstrap = dict(kind='standalone_initial_npz', path=str(args.initial_model.resolve()),
                         bytes=args.initial_model.stat().st_size)
    elif config.get('ckpt_path') and Path(config['ckpt_path']).is_file():
        checkpoint = torch.load(config['ckpt_path'], map_location='cpu', weights_only=False)
        values = checkpoint['gaussians']
        raw = dict(xyz=values[1], sh_dc=values[2], sh_rest=values[3], logscale=values[4],
                   rotation=values[5], opacity=values[6])
        raw = {name: value.detach().to(device) for name, value in raw.items()}
        bootstrap = dict(kind='shared_first_frame_checkpoint', path=config['ckpt_path'])
    else:
        first_checkpoint = torch.load(run / f'ckpt-1-{steps}.pth', map_location='cpu', weights_only=False)
        raw = {name: value.to(device) for name, value in
               first_checkpoint['quantized_frame_state']['previous_reference'].items()}
        bootstrap = dict(kind='legacy_dynamic_checkpoint_reference')
    storage = run / 'frame_storage' / 'common_contrast_binary'
    first_motion = storage / 'motion_001.bin'
    decoder = SAB.decoder_for_binary_stream(first_motion, motion_config)
    rows = []
    if args.profile and device.type == 'cuda':
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader,nounits'], text=True).splitlines()
        foreign = [int(value.strip()) for value in processes if value.strip() and int(value.strip()) != os.getpid()]
        if foreign:
            raise RuntimeError(f'Exclusive decoder timing requires idle GPU; other processes: {foreign}')
        torch.cuda.synchronize()
    for frame in range(1, args.frames + 1):
        motion = storage / f'motion_{frame:03d}.bin'
        birth = storage / f'incremental_{frame:03d}.ply'
        compact_birth = storage / f'incremental_{frame:03d}.bin'
        if compact_birth.is_file():
            birth = compact_birth
        if not motion.is_file():
            raise FileNotFoundError(motion)
        appearance = storage / f'appearance_{frame:03d}.bin'
        opacity = storage / f'opacity_{frame:03d}.bin'
        started = time.perf_counter()
        raw, info = decoder.advance(motion, birth if birth.is_file() else None, raw,
                                   appearance_path=appearance if appearance.is_file() else None,
                                   opacity_path=opacity if opacity.is_file() else None)
        if args.profile:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            info['decoder_seconds'] = time.perf_counter() - started
        rows.append(info)
        print(f"frame={frame} rebuild={info['tree_rebuilt']} "
              f"deleted={len(info['deleted_rows'])} births={info['birth_count']} "
              f"bits={info['coded_payload_bits']}", flush=True)
    final_checkpoint = torch.load(run / f'ckpt-{args.frames}-{steps}.pth',
                                  map_location='cpu', weights_only=False)
    packet = final_checkpoint['gaussians']
    expected = dict(xyz=packet[1].detach().to(device),
        sh_dc=packet[2].detach().to(device), sh_rest=packet[3].detach().to(device),
        logscale=packet[4].detach().to(device), rotation=packet[5].detach().to(device),
        opacity=packet[6].detach().to(device))
    exact = {name: bool(torch.equal(raw[name], expected[name])) for name in Q.RAW_FIELDS}
    if not all(exact.values()):
        error = {name: float((raw[name] - expected[name]).abs().max())
                 for name in Q.RAW_FIELDS}
        raise AssertionError(f'Final decoded checkpoint mismatch: exact={exact}, max_abs={error}')
    final_digest = Q.reference_digest(raw)
    expected_digest = Q.reference_digest(expected)
    if final_digest != expected_digest:
        raise AssertionError('Exact final fields produced differing state digests')
    result = dict(schema='common_contrast_sequential_decode_audit_v1',
        run=str(run), device=str(device), frame_count=args.frames, bootstrap=bootstrap,
        frames=rows, final_fields_exact=exact,
        final_state_sha256=final_digest,
        final_checkpoint_state_sha256=expected_digest,
        final_state_match=(final_digest == expected_digest and all(exact.values())),
        semantics=('Strict hashes are verified on the requested device. Float kernel order can '
                   'make a CPU replay differ in low bits from a CUDA-encoded reference.'))
    if args.profile:
        seconds = sum(row['decoder_seconds'] for row in rows)
        result['decoder_timing'] = dict(seconds=seconds, mean_seconds=seconds / args.frames,
            decoder_fps=args.frames / seconds, includes_render=False,
            semantics='Includes packet reads, integrity hashes, deletion, topology, motion and attribute reconstruction. '
                      'Excludes renderer, checkpoint validation and dataset/image loading. Requires an idle GPU for speed claims.')
    path = args.output or run / f'sequential_decode_audit_frame001_{args.frames:03d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(path)


if __name__ == '__main__':
    main()
