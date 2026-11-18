"""Independently replay and audit a point-only 300-timepoint N3DV experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline.model.frame_stream_decode as decoder
from pipeline.model.frame_stream_format import DecodedState, FRAME_FIELDS, reference_digest

PACKETS = ('motion', 'birth', 'opacity_update', 'feature_dc_update', 'logscale_update',
           'full_attribute_update', 'regional_full_attributes', 'color_point_mask')


def normalized(run):
    cfg = OmegaConf.to_container(OmegaConf.load(run / 'config.yaml'), resolve=True)
    cfg.pop('experiment_name', None)
    cfg.pop('output_dir', None)
    cfg['trainer'].pop('saving_ckpt_frames', None)
    cfg['data']['extra_dataset_kwargs'].pop('num_frames', None)
    return cfg


def audit(scene, baseline, run, count, output):
    if count < 2 or output.exists() or output.is_symlink() or run.resolve() == baseline.resolve():
        raise ValueError('Invalid audit output, frame count or run')
    cfg = normalized(run)
    if cfg['module']['full_update']['regional_full_200'] or cfg['module']['full_update']['attribute_aware_gate']:
        raise ValueError('Regional attributes and their gate must be disabled')
    cfg['module']['full_update']['regional_full_200'] = True
    cfg['module']['full_update']['attribute_aware_gate'] = True
    if cfg != normalized(baseline):
        raise ValueError('Configuration differs from baseline beyond the declared ablation')
    actual_cfg = OmegaConf.load(run / 'config.yaml')
    if int(actual_cfg.data.extra_dataset_kwargs.num_frames) != count:
        raise ValueError('Wrong run length')
    if actual_cfg.get('resume_prefix_dir') is not None or actual_cfg.get('resume_completed_prefix_frames') is not None:
        raise ValueError('The experiment must start from the shared first frame')
    stream = run / 'frame_storage/rigid_tree'
    reference = stream / 'reference_frame0.npz'
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(reference) != sha(baseline / 'frame_storage/rigid_tree/reference_frame0.npz'):
        raise ValueError('Shared first-frame reference differs')
    with np.load(reference, allow_pickle=False) as packet:
        state = DecodedState.from_reference(packet)
        raw = {name: torch.from_numpy(packet[name].copy()) for name in FRAME_FIELDS}
    torch.set_num_threads(4)
    rows = []
    for frame in range(1, count):
        sp, mp = run / f'frame_summary/frame_{frame:03d}.json', stream / f'frame_{frame:03d}.json'
        summary, manifest = json.loads(sp.read_text()), json.loads(mp.read_text())
        prior_summary = json.loads((baseline / f'frame_summary/frame_{frame:03d}.json').read_text())
        if manifest['frame'] != frame or manifest['optimization_steps'] != 200:
            raise ValueError(f'{frame}: wrong step count')
        info, accounting = manifest['point_only_ablation'], manifest['training_step_accounting']
        if (info['region_motion'] or info['gate_learning'] or info['regional_attributes'] or
                info['point_translation_steps'] != 200 or info['point_rotation_steps'] != 100 or
                info['point_attribute_steps'] != 100 or accounting['point_adam_steps'] != 200 or
                accounting['shared_adam_steps'] != 0 or manifest['regional_full_attributes'] is not None):
            raise ValueError(f'{frame}: wrong point-only policy')
        trace = summary['birth_control']['point_schedule_camera_trace']
        if len(trace) != 200 or trace != prior_summary['birth_control']['point_schedule_camera_trace']:
            raise ValueError(f'{frame}: camera trace differs')
        checks = summary['birth_control']['point_only_freeze_audit']
        if [row['step'] for row in checks] != [1, 100, 101, 151, 180, 200]:
            raise ValueError(f'{frame}: incomplete parameter schedule')
        for row in checks:
            second = row['step'] > 100
            if (row['region_translation_trainable'] or row['region_rotation_trainable'] or row['gate_trainable'] or
                    not row['point_translation_trainable'] or row['point_rotation_trainable'] != second or
                    row['opacity_trainable'] != second or set(row['point_attribute_trainable']) != {'logscale', 'sh_dc', 'sh_rest'} or
                    any(value != second for value in row['point_attribute_trainable'].values())):
                raise ValueError(f"{frame}: invalid trainable parameters at step {row['step']}")
        grads = {row['step']: row['parameters'] for row in summary['birth_control']['frame_plan_gradients']}
        for step in (100, 200):
            p = grads[step]
            if (p['gs_translation']['adam_updates'] != step - 1 or p['region_translation']['adam_updates'] or
                    p['region_rotation']['adam_updates'] or p['gate_logits']['adam_updates']):
                raise ValueError(f'{frame}: optimizer update count differs at step {step}')
        if any(grads[200][name]['adam_updates'] != 99 for name in
               ('gs_rotation', 'old_alpha', 'old_full_logscale', 'old_full_sh_dc', 'old_full_sh_rest')):
            raise ValueError(f'{frame}: missing second-stage optimizer updates')
        check = summary['forward_decode_check']
        if check['max_abs'] > check['tolerance']:
            raise ValueError(f'{frame}: training and decoded images differ')
        paths = {key: manifest[key] for key in PACKETS if manifest.get(key)}
        if len(paths) != len(set(paths.values())):
            raise ValueError(f'{frame}: duplicated packet path')
        sizes = {key: (stream / value).stat().st_size for key, value in paths.items()}
        sizes['manifest'] = mp.stat().st_size
        if sum(sizes.values()) != summary['complete_dynamic_update_bytes']:
            raise ValueError(f'{frame}: packet bytes differ')
        original = decoder._sparse_read
        calls = []
        def check_blocks(*args, **kwargs):
            values = original(*args, **kwargs)
            calls.append(values)
            return values
        decoder._sparse_read = check_blocks
        try:
            state, raw = decoder.decode_frame(stream, frame, state, raw, manifest=manifest)
        finally:
            decoder._sparse_read = original
        if len(calls) != 2 or any(bool(value.ne(0).any()) for value in calls[0]):
            raise ValueError(f'{frame}: regional motion symbols are not zero')
        if state.digest().hex() != manifest['decoded_state_sha256'] or reference_digest(raw) != manifest['raw_state_sha256']:
            raise ValueError(f'{frame}: independent decode digest differs')
        metrics = summary['evaluation']['test']
        row = dict(frame=frame, psnr=float(metrics['psnr']), ssim=float(metrics['ssim']),
                   lpips=float(metrics['lpips']), training_seconds=float(summary['training_seconds']),
                   actual_bytes=sum(sizes.values()), components=sizes)
        if any(not math.isfinite(row[k]) for k in ('psnr', 'ssim', 'lpips', 'training_seconds')):
            raise ValueError(f'{frame}: non-finite metric')
        rows.append(row)
    result = dict(run=str(run.resolve()), psnr=mean(r['psnr'] for r in rows), ssim=mean(r['ssim'] for r in rows),
                  lpips=mean(r['lpips'] for r in rows), dynamic_kb=mean(r['actual_bytes'] for r in rows) / 1000,
                  training_seconds=mean(r['training_seconds'] for r in rows),
                  component_kb={k: mean(r['components'].get(k, 0) for r in rows) / 1000 for k in (*PACKETS, 'manifest')},
                  frames=rows, last_state_sha256=state.digest().hex())
    payload = dict(scene=scene, total_timepoints=count, dynamic_frames=count-1,
                   first_frame_reference_sha256=sha(reference), results={'depth0': result},
                   notes=['no shared regional motion or attributes; fixed spatial metadata retained for codec compatibility',
                          'point translation 200 updates, point rotation and attributes 100 updates',
                          'baseline predates refactor and lacks a full original worktree snapshot'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(payload, f, indent=2)
        f.write('\n')
    print(f"AUDITED {scene}: PSNR={result['psnr']:.4f}, KB={result['dynamic_kb']:.3f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    audit(args.scene, args.baseline, args.run, args.count, args.output)
