"""Audit adaptive N3DV storage-loss ablations against the saved baseline."""
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
KEY_STEPS = (1, 100, 101, 151, 180, 181, 200)
ZERO_KEYS = (('rigid_tree', 'rate_loss_weight'), ('opacity_update', 'l1_weight'),
             ('full_update', 'per_attribute_l1_weight'), ('full_update', 'regional_opacity_l1_weight'))


def normalized(run):
    cfg = OmegaConf.to_container(OmegaConf.load(run / 'config.yaml'), resolve=True)
    cfg.pop('experiment_name', None); cfg.pop('output_dir', None)
    cfg['trainer'].pop('saving_ckpt_frames', None)
    cfg['data']['extra_dataset_kwargs'].pop('num_frames', None)
    return cfg


def audit(scene, baseline, run, count, output):
    if count < 2 or output.exists() or output.is_symlink() or baseline.resolve() == run.resolve():
        raise ValueError('Invalid output, count or run')
    cfg, base_cfg = normalized(run), normalized(baseline)
    for section, key in ZERO_KEYS:
        if cfg['module'][section][key] != 0:
            raise ValueError(f'{section}.{key} must be zero')
        cfg['module'][section][key] = base_cfg['module'][section][key]
    if cfg != base_cfg:
        raise ValueError('Configuration differs beyond the four declared storage penalties')
    actual_cfg = OmegaConf.load(run/'config.yaml')
    if (int(actual_cfg.data.extra_dataset_kwargs.num_frames) != count or
            actual_cfg.get('resume_prefix_dir') is not None or actual_cfg.get('resume_completed_prefix_frames') is not None):
        raise ValueError('Expected an independent run of the declared frame count')
    stream = run/'frame_storage/rigid_tree';reference = stream/'reference_frame0.npz'
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(reference) != sha(baseline/'frame_storage/rigid_tree/reference_frame0.npz'):
        raise ValueError('Shared first-frame reference differs')
    with np.load(reference, allow_pickle=False) as packet:
        state = DecodedState.from_reference(packet)
        raw = {name:torch.from_numpy(packet[name].copy()) for name in FRAME_FIELDS}
    torch.set_num_threads(4)
    rows=[]
    for frame in range(1,count):
        sp=run/f'frame_summary/frame_{frame:03d}.json';mp=stream/f'frame_{frame:03d}.json'
        summary=json.loads(sp.read_text());manifest=json.loads(mp.read_text())
        prior=json.loads((baseline/f'frame_summary/frame_{frame:03d}.json').read_text())
        if manifest['frame'] != frame or manifest['optimization_steps'] != 200:
            raise ValueError(f'{frame}: incorrect frame or step count')
        trace=summary['birth_control']['point_schedule_camera_trace']
        if len(trace)!=200 or trace!=prior['birth_control']['point_schedule_camera_trace']:
            raise ValueError(f'{frame}: camera sequence differs from baseline')
        actual={x['step']:x['parameters'] for x in summary['birth_control']['frame_plan_gradients']}
        expected={x['step']:x['parameters'] for x in prior['birth_control']['frame_plan_gradients']}
        for step in KEY_STEPS:
            a,b=actual[step],expected[step]
            if set(a)!=set(b) or any(a[k]['trainable']!=b[k]['trainable'] for k in a):
                raise ValueError(f'{frame}: trainability differs at step {step}')
            if a['gate_logits']['trainable'] != (step<=100 or 151<=step<=180):
                raise ValueError(f'{frame}: wrong adaptive gate schedule at step {step}')
        motion=summary['motion_rate_diagnostics']
        if float(motion['rate_loss']) != 0 or float(motion['rate_byte_price']) != 0:
            raise ValueError(f'{frame}: motion byte loss or price is nonzero')
        for key in ('opacity_rate_diagnostics','full_attribute_rate_diagnostics','regional_full_rate_diagnostics'):
            if float(summary[key]['loss']) != 0 or float(summary[key]['l1_loss']) != 0:
                raise ValueError(f'{frame}: {key} not zero')
        for entry in summary['birth_control'].get('attribute_gate_rate_checks', []):
            if entry['gradient_norm'] != 0 or entry['nonzero_gates'] != 0:
                raise ValueError(f'{frame}: storage gate gradient is nonzero')
        check=summary['forward_decode_check']
        if check['max_abs']>check['tolerance']:
            raise ValueError(f'{frame}: training and decoded images differ')
        paths={k:manifest[k] for k in PACKETS if manifest.get(k)}
        if len(paths)!=len(set(paths.values())):
            raise ValueError(f'{frame}: duplicate packet path')
        sizes={k:(stream/v).stat().st_size for k,v in paths.items()};sizes['manifest']=mp.stat().st_size
        if sum(sizes.values())!=summary['complete_dynamic_update_bytes']:
            raise ValueError(f'{frame}: actual byte accounting differs')
        state,raw=decode_frame(stream,frame,state,raw,manifest=manifest)
        if state.digest().hex()!=manifest['decoded_state_sha256'] or reference_digest(raw)!=manifest['raw_state_sha256']:
            raise ValueError(f'{frame}: independent replay digest differs')
        m=summary['evaluation']['test']
        row=dict(frame=frame,psnr=float(m['psnr']),ssim=float(m['ssim']),lpips=float(m['lpips']),
                 training_seconds=float(summary['training_seconds']),actual_bytes=sum(sizes.values()),
                 components=sizes,active_regions=int(summary['region_count']),
                 relative_motion_loss=float(motion['relative_motion_loss']))
        if any(not math.isfinite(row[k]) for k in ('psnr','ssim','lpips','training_seconds','relative_motion_loss')):
            raise ValueError(f'{frame}: non-finite metric')
        rows.append(row)
    result=dict(run=str(run.resolve()),psnr=mean(x['psnr'] for x in rows),ssim=mean(x['ssim'] for x in rows),
                lpips=mean(x['lpips'] for x in rows),dynamic_kb=mean(x['actual_bytes'] for x in rows)/1000,
                training_seconds=mean(x['training_seconds'] for x in rows),active_regions=mean(x['active_regions'] for x in rows),
                relative_motion_loss=mean(x['relative_motion_loss'] for x in rows),
                component_kb={k:mean(x['components'].get(k,0) for x in rows)/1000 for k in (*PACKETS,'manifest')},
                frames=rows,last_state_sha256=state.digest().hex())
    payload=dict(scene=scene,total_timepoints=count,dynamic_frames=count-1,reference_sha256=sha(reference),
                 result=result,notes=['all storage byte and attribute/opacity L1 penalties set to zero',
                    'relative motion constraint, quantization, gates and the 200-step schedule are retained',
                    'baseline was trained before refactoring and its original full worktree was not saved'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(f"AUDITED {scene}: PSNR={result['psnr']:.4f}, KB={result['dynamic_kb']:.3f}")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.scene,a.baseline,a.run,a.count,a.output)
