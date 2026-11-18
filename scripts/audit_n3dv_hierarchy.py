"""Audit one N3DV scene: existing adaptive baseline versus independent fixed depths."""
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
from pipeline.model.frame_stream_decode import decode_frame
from pipeline.model.frame_stream_format import DecodedState, FRAME_FIELDS, reference_digest

PACKETS = ('motion', 'birth', 'opacity_update', 'feature_dc_update', 'logscale_update',
           'full_attribute_update', 'regional_full_attributes', 'color_point_mask')
MODES = ('baseline', 'depth2', 'depth3', 'depth4')
KEY_STEPS = (1, 100, 101, 151, 180, 181, 200)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_config(run):
    config = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'), resolve=True)
    config.pop('experiment_name', None)
    config.pop('output_dir', None)
    config['trainer'].pop('saving_ckpt_frames', None)
    config['data']['extra_dataset_kwargs'].pop('num_frames', None)
    return config


def audit(scene, baseline, fixed, count, output, depths=(2, 3, 4)):
    if count < 2 or output.exists() or output.is_symlink():
        raise ValueError('Frame count or output path is invalid')
    if len(depths) != len(fixed) or len(set(depths)) != len(depths) or any(d not in range(1, 6) for d in depths):
        raise ValueError('Specify one distinct candidate depth from 1 to 5 for each fixed run')
    runs = {'baseline':baseline, **{f'depth{depth}':run for depth,run in zip(depths,fixed)}}
    if len({path.resolve() for path in runs.values()}) != 1 + len(fixed):
        raise ValueError('Each fixed group must be an independent run')
    if any(output.resolve().is_relative_to(path.resolve()) for path in runs.values()):
        raise ValueError('Audit must not write into a training run')
    reference = baseline/'frame_storage/rigid_tree/reference_frame0.npz'
    reference_sha = sha(reference)
    base_cfg = normalized_config(baseline)
    torch.set_num_threads(4)
    traces, baseline_gradients, results = {}, {}, {}
    for mode, run in runs.items():
        cfg = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'), resolve=True)
        expected_frames = 300 if mode == 'baseline' else count
        if int(cfg['data']['extra_dataset_kwargs']['num_frames']) != expected_frames:
            raise ValueError(f'{mode}: unexpected total timepoints')
        if normalized_config(run) != base_cfg:
            raise ValueError(f'{mode}: config differs beyond output, frame count and checkpoint saves')
        if mode != 'baseline' and (cfg.get('resume_prefix_dir') is not None or
                                   cfg.get('resume_completed_prefix_frames') is not None):
            raise ValueError(f'{mode}: experiment must begin from shared first frame')
        stream = run/'frame_storage/rigid_tree'
        if sha(stream/'reference_frame0.npz') != reference_sha:
            raise ValueError(f'{mode}: first frame differs')
        with np.load(stream/'reference_frame0.npz', allow_pickle=False) as packet:
            raw = {name:torch.from_numpy(packet[name].copy()) for name in FRAME_FIELDS}
            state = DecodedState.from_reference(packet)
        rows = []
        for frame in range(1,count):
            sp = run/f'frame_summary/frame_{frame:03d}.json'
            mp = stream/f'frame_{frame:03d}.json'
            summary, manifest = json.loads(sp.read_text()), json.loads(mp.read_text())
            if manifest['frame'] != frame or manifest['optimization_steps'] != 200:
                raise ValueError(f'{mode} frame {frame}: wrong optimization steps')
            trace = summary['birth_control']['point_schedule_camera_trace']
            if len(trace) != 200 or [v['step'] for v in trace] != list(range(1,201)):
                raise ValueError(f'{mode} frame {frame}: incomplete camera trace')
            grads = {v['step']:v['parameters'] for v in summary['birth_control']['frame_plan_gradients']}
            if mode == 'baseline':
                traces[frame] = trace
                baseline_gradients[frame] = grads
                for step in KEY_STEPS:
                    if grads[step]['gate_logits']['trainable'] != (step <= 100 or 151 <= step <= 180):
                        raise ValueError(f'Unexpected baseline gate schedule frame {frame} step {step}')
            else:
                if trace != traces[frame]:
                    raise ValueError(f'{mode} frame {frame}: camera sequence differs from baseline')
                info = manifest['hierarchy_ablation']
                depth = int(mode[-1])
                if (info['mode'] != mode or info['fixed_depth'] != depth or
                        info['first_stage_steps'] != 100 or info['second_stage_steps'] != 100):
                    raise ValueError(f'{mode} frame {frame}: wrong hierarchy policy')
                freeze = summary['birth_control']['hierarchy_ablation_freeze_audit']
                if [v['step'] for v in freeze] != [1,100,101,151,180,200]:
                    raise ValueError(f'{mode} frame {frame}: incomplete freeze trace')
                for step in KEY_STEPS:
                    a, b = baseline_gradients[frame][step], grads[step]
                    if set(a) != set(b) or b['gate_logits']['trainable'] or b['gate_logits']['adam_updates']:
                        raise ValueError(f'{mode} frame {frame}: gate/parameter audit failed at {step}')
                    for name in a:
                        if name != 'gate_logits' and a[name]['trainable'] != b[name]['trainable']:
                            raise ValueError(f'{mode} frame {frame}: non-gate freeze differs at {step}: {name}')
                diag = summary['flat_region_diagnostics']
                active = diag['active_regions_by_depth']
                if not int(active[str(depth)]) or any(int(v) for k,v in active.items() if int(k) != depth):
                    raise ValueError(f'{mode} frame {frame}: wrong depth {active}')
                if any(v['changed_gaussians'] for v in diag['reference_changes']):
                    raise ValueError(f'{mode} frame {frame}: ownership changed during training')
            check = summary['forward_decode_check']
            if check['max_abs'] > check['tolerance']:
                raise ValueError(f'{mode} frame {frame}: training/decoded image mismatch')
            files = {k:manifest[k] for k in PACKETS if manifest.get(k)}
            if len(files) != len(set(files.values())):
                raise ValueError(f'{mode} frame {frame}: duplicate packet path')
            sizes = {k:(stream/v).stat().st_size for k,v in files.items()}
            sizes['manifest'] = mp.stat().st_size
            total = sum(sizes.values())
            if total != summary['complete_dynamic_update_bytes']:
                raise ValueError(f'{mode} frame {frame}: byte accounting mismatch')
            state, raw = decode_frame(stream, frame, state, raw, manifest=manifest)
            if state.digest().hex() != manifest['decoded_state_sha256'] or reference_digest(raw) != manifest['raw_state_sha256']:
                raise ValueError(f'{mode} frame {frame}: decoded state mismatch')
            metrics = summary['evaluation']['test']
            row = dict(frame=frame, psnr=float(metrics['psnr']), ssim=float(metrics['ssim']),
                       lpips=float(metrics['lpips']), actual_bytes=total, components=sizes,
                       regions=int(summary['region_count']), training_seconds=float(summary['training_seconds']))
            if not all(math.isfinite(row[k]) for k in ('psnr','ssim','lpips','training_seconds')):
                raise ValueError(f'{mode} frame {frame}: nonfinite metric')
            rows.append(row)
        components={key:mean(r['components'].get(key,0) for r in rows)/1000 for key in (*PACKETS,'manifest')}
        result=dict(run=str(run.resolve()), psnr=mean(r['psnr'] for r in rows), ssim=mean(r['ssim'] for r in rows),
                    lpips=mean(r['lpips'] for r in rows), dynamic_kb=mean(r['actual_bytes'] for r in rows)/1000,
                    active_regions=mean(r['regions'] for r in rows), training_seconds=mean(r['training_seconds'] for r in rows),
                    component_kb=components, last_state_sha256=state.digest().hex(), frames=rows)
        results[mode]=result
        print(f"AUDITED {scene} {mode}: PSNR={result['psnr']:.4f}, KB={result['dynamic_kb']:.3f}",flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(scene=scene,total_timepoints=count,dynamic_frames=count-1,
        first_frame_reference_sha256=reference_sha,results=results,
        notes=['baseline gate is re-enabled at steps 151–180; fixed gate is always frozen',
               'fixed owners are re-routed from previous decoded positions between frames',
               'baseline predates project refactoring; its full original worktree snapshot was not saved']),indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--fixed',type=Path,nargs='+',required=True);p.add_argument('--depths',type=int,nargs='+',default=(2,3,4))
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit(a.scene,a.baseline,a.fixed,a.count,a.output,a.depths)
