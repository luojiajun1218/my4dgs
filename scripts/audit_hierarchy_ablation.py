"""Audit independent 100+100 hierarchy ablations and actual dynamic packets."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model.frame_stream_decode import decode_frame
from pipeline.model.frame_stream_format import DecodedState, FRAME_FIELDS, reference_digest

MODES = ('adaptive', 'depth2', 'depth3', 'depth4')
PACKETS = ('motion', 'birth', 'opacity_update', 'feature_dc_update', 'logscale_update',
           'full_attribute_update', 'regional_full_attributes', 'color_point_mask')

def normalized_config(run):
    data = OmegaConf.to_container(OmegaConf.load(run / 'config.yaml'), resolve=True)
    data.pop('experiment_name', None)
    return data

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_run(root, mode, phase, count):
    suffix = f'cook_hierarchy_{mode}_{phase}{count}_20261115'
    runs = list((root / suffix).glob('*/frame_summary/frame_001.json'))
    if len(runs) != 1:
        raise ValueError(f'Expected one independent run for {mode}: {runs}')
    return runs[0].parent.parent

def mean_key(rows, key):
    return mean(x[key] for x in rows)

def audit(root, phase, count, output):
    if count < 2 or output.exists() or output.is_symlink():
        raise ValueError('Invalid frame count or existing output')
    torch.set_num_threads(4)
    runs = {mode: load_run(root, mode, phase, count) for mode in MODES}
    reference = runs['adaptive'] / 'frame_storage/rigid_tree/reference_frame0.npz'
    reference_sha = sha(reference)
    base_cfg = normalized_config(runs['adaptive'])
    traces = {}
    results = {}
    for mode, run in runs.items():
        if normalized_config(run) != base_cfg:
            raise ValueError(f'Configuration differs outside experiment name: {mode}')
        stream = run / 'frame_storage/rigid_tree'
        if sha(stream / 'reference_frame0.npz') != reference_sha:
            raise ValueError(f'First-frame reference differs: {mode}')
        with np.load(stream / 'reference_frame0.npz', allow_pickle=False) as packet:
            raw = {key: torch.from_numpy(packet[key].copy()) for key in FRAME_FIELDS}
            state = DecodedState.from_reference(packet)
        rows = []
        for frame in range(1, count):
            sp = run / f'frame_summary/frame_{frame:03d}.json'
            mp = stream / f'frame_{frame:03d}.json'
            summary, manifest = json.loads(sp.read_text()), json.loads(mp.read_text())
            if manifest['frame'] != frame or manifest['optimization_steps'] != 200:
                raise ValueError(f'Wrong frame or step count: {mode} {frame}')
            metadata = manifest['hierarchy_ablation']
            if metadata['mode'] != mode or metadata['first_stage_steps'] != 100 or metadata['second_stage_steps'] != 100:
                raise ValueError(f'Wrong hierarchy policy: {mode} {frame}')
            trace = summary['birth_control']['point_schedule_camera_trace']
            if len(trace) != 200 or [x['step'] for x in trace] != list(range(1, 201)):
                raise ValueError(f'Incomplete camera trace: {mode} {frame}')
            if mode == 'adaptive':
                traces[frame] = trace
            elif trace != traces[frame]:
                raise ValueError(f'Camera sequence mismatch: {mode} {frame}')
            audit_rows = summary['birth_control']['hierarchy_ablation_freeze_audit']
            if [r['step'] for r in audit_rows] != [1, 100, 101, 151, 180, 200]:
                raise ValueError(f'Incomplete freeze audit: {mode} {frame}')
            for item in audit_rows:
                early = item['step'] <= 100
                if (item['gate_trainable'] != (mode == 'adaptive' and early) or
                    not item['region_translation_trainable'] or item['region_rotation_trainable'] == early or
                    item['point_translation_trainable'] == early or item['point_rotation_trainable'] == early or
                    item['opacity_trainable'] == early or
                    any(flag == early for flag in item['point_attribute_trainable'].values())):
                    raise ValueError(f'Incorrect parameter freeze: {mode} frame {frame}, {item}')
            gradients = {r['step']: r['parameters'] for r in summary['birth_control']['frame_plan_gradients']}
            if gradients[100]['gate_logits']['adam_updates'] != (99 if mode == 'adaptive' else 0):
                raise ValueError(f'Incorrect gate optimizer state at 100: {mode} {frame}')
            if gradients[200]['gate_logits']['adam_updates'] != (100 if mode == 'adaptive' else 0):
                raise ValueError(f'Gate optimizer changed in second stage: {mode} {frame}')
            diag = summary['flat_region_diagnostics']
            changes = diag['reference_changes']
            if any(row['changed_gaussians'] for row in changes if mode != 'adaptive' or row['step'] > 100):
                raise ValueError(f'Frozen cut changed: {mode} {frame}')
            if mode != 'adaptive':
                depth = int(mode[-1]); active = diag['active_regions_by_depth']
                if not int(active[str(depth)]) or any(int(n) for k,n in active.items() if int(k) != depth):
                    raise ValueError(f'Wrong active depth: {mode} {frame} {active}')
            check = summary['forward_decode_check']
            if check['max_abs'] > check['tolerance']:
                raise ValueError(f'Training vs decoded rendering mismatch: {mode} {frame}')
            names = {key:manifest[key] for key in PACKETS if manifest.get(key)}
            if len(set(names.values())) != len(names):
                raise ValueError(f'Duplicate packet name: {mode} {frame}')
            sizes = {key:(stream/name).stat().st_size for key,name in names.items()}
            sizes['manifest'] = mp.stat().st_size
            total = sum(sizes.values())
            if total != summary['complete_dynamic_update_bytes']:
                raise ValueError(f'Actual bytes differ from summary: {mode} {frame}')
            state, raw = decode_frame(stream, frame, state, raw, manifest=manifest)
            if state.digest().hex() != manifest['decoded_state_sha256'] or reference_digest(raw) != manifest['raw_state_sha256']:
                raise ValueError(f'Independent replay digest mismatch: {mode} {frame}')
            metrics = summary['evaluation']['test']
            rows.append(dict(frame=frame, psnr=metrics['psnr'], ssim=metrics['ssim'], lpips=metrics['lpips'],
                             actual_bytes=total, components=sizes, regions=summary['region_count'],
                             training_seconds=summary['training_seconds']))
        components = {name:mean(row['components'].get(name,0) for row in rows)/1000 for name in (*PACKETS,'manifest')}
        results[mode] = dict(run=str(run), psnr=mean_key(rows,'psnr'), ssim=mean_key(rows,'ssim'),
            lpips=mean_key(rows,'lpips'), dynamic_kb=mean_key(rows,'actual_bytes')/1000,
            training_seconds=mean_key(rows,'training_seconds'), active_regions=mean_key(rows,'regions'),
            component_kb=components, last_state_sha256=state.digest().hex(), frames=rows)
        print(f"AUDITED {mode}: PSNR={results[mode]['psnr']:.4f}, KB={results[mode]['dynamic_kb']:.3f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(total_timepoints=count, dynamic_frames=count-1,
        first_frame_reference_sha256=reference_sha, spatial_membership='per_frame_previous_decoded_position',
        results=results), indent=2)+'\n')

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke','short','full'], required=True);p.add_argument('--count', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args();audit(a.root,a.phase,a.count,a.output)
