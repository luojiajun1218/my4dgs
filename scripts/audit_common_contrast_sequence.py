"""Replay a completed common/contrast sequence through its real decoder."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch
import yaml


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
    args = parser.parse_args()
    run = args.run.resolve()
    device = torch.device(args.device)
    config = yaml.safe_load((run / 'config.yaml').read_text())
    motion_config = CC.CommonContrastConfig(**config['module']['common_contrast_binary'])
    first_checkpoint = torch.load(run / 'ckpt-1-300.pth', map_location='cpu',
                                  weights_only=False)
    raw = {name: value.to(device) for name, value in
           first_checkpoint['quantized_frame_state']['previous_reference'].items()}
    storage = run / 'frame_storage' / 'common_contrast_binary'
    first_motion = storage / 'motion_001.bin'
    decoder = SAB.decoder_for_binary_stream(first_motion, motion_config)
    rows = []
    for frame in range(1, args.frames + 1):
        motion = storage / f'motion_{frame:03d}.bin'
        birth = storage / f'incremental_{frame:03d}.ply'
        if not motion.is_file():
            raise FileNotFoundError(motion)
        raw, info = decoder.advance(motion, birth if birth.is_file() else None, raw)
        rows.append(info)
        print(f"frame={frame} rebuild={info['tree_rebuilt']} "
              f"deleted={len(info['deleted_rows'])} births={info['birth_count']} "
              f"bits={info['coded_payload_bits']}", flush=True)
    final_checkpoint = torch.load(run / f'ckpt-{args.frames}-300.pth',
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
        run=str(run), device=str(device), frame_count=args.frames,
        frames=rows, final_fields_exact=exact,
        final_state_sha256=final_digest,
        final_checkpoint_state_sha256=expected_digest,
        final_state_match=(final_digest == expected_digest and all(exact.values())),
        semantics=('Strict hashes are verified on the requested device. Float kernel order can '
                   'make a CPU replay differ in low bits from a CUDA-encoded reference.'))
    path = run / f'sequential_decode_audit_frame001_{args.frames:03d}.json'
    path.write_text(json.dumps(result, indent=2))
    print(path)


if __name__ == '__main__':
    main()
