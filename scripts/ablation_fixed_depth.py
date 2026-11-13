"""汇总自适应区域深度与五个固定深度的对照实验。

读取已有实验，核对配置、首帧、相机顺序、帧码流和存储统计，输出各方案
动态帧的画质、存储、区域数与训练时间。输入帧数包含首帧，汇总从第 1 帧开始。
本文件不负责训练；固定深度实验由 train_fixed_depth.py 独立运行。
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model.frame_stream_decode import (replay)

PACKETS = ('motion', 'birth', 'opacity_update', 'feature_dc_update',
           'logscale_update', 'full_attribute_update',
           'regional_full_attributes', 'color_point_mask')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(run):
    c = OmegaConf.to_container(OmegaConf.load(run/'config.yaml'), resolve=True)
    c.pop('experiment_name', None)
    c['data']['extra_dataset_kwargs'].pop('num_frames', None)
    c['trainer'].pop('saving_ckpt_frames', None)
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fixed', type=Path, nargs=5, required=True,
                        help='Run directories in depth 1..5 order')
    parser.add_argument('--num-frames', type=int, default=31,
                        help='Total frames to audit, including frame 0 (default: 31)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.num_frames < 2:
        parser.error('--num-frames must include frame 0 and at least one dynamic frame')
    runs = [args.baseline, *args.fixed]
    if len({run.resolve() for run in runs}) != len(runs):
        raise ValueError('Baseline and fixed-depth experiments must be separate runs')
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if any(args.output.resolve().is_relative_to(run.resolve()) for run in runs):
        raise ValueError('Write the report outside the source experiment directories')
    torch.set_num_threads(4)
    baseline = args.baseline
    reference_sha = sha(baseline/'frame_storage/rigid_tree/reference_frame0.npz')
    base_cfg = load_config(baseline)
    scene_root = base_cfg['data']['root']
    reference_trace, reference_steps = {}, {}
    for frame in range(1, args.num_frames):
        summary = json.loads((baseline/f'frame_summary/frame_{frame:03d}.json').read_text())
        manifest = json.loads((baseline/f'frame_storage/rigid_tree/frame_{frame:03d}.json').read_text())
        trace = summary.get('birth_control', {}).get('point_schedule_camera_trace')
        steps = manifest.get('optimization_steps')
        if (type(steps) is not int or steps < 1 or not isinstance(trace, list) or
                len(trace) != steps):
            raise ValueError(f'Baseline frame {frame} needs a complete training camera trace and step count; '
                             'enable full_update.audit_camera_sequence when training')
        reference_trace[frame], reference_steps[frame] = trace, steps
    results = []
    for index, run in enumerate(runs):
        depth = index if index else None
        if load_config(run) != base_cfg:
            raise ValueError(f'Configuration differs beyond run name and frame count: {run}')
        stream = run/'frame_storage/rigid_tree'
        if sha(stream/'reference_frame0.npz') != reference_sha:
            raise ValueError(f'Shared frame differs: {run}')
        rows = []
        for frame in range(1, args.num_frames):
            summary = json.loads((run/f'frame_summary/frame_{frame:03d}.json').read_text())
            manifest_path = stream/f'frame_{frame:03d}.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest['frame'] != frame or manifest['optimization_steps'] != reference_steps[frame]:
                raise ValueError(f'Frame/step mismatch {run} frame {frame}')
            if summary.get('birth_control', {}).get('point_schedule_camera_trace') != reference_trace[frame]:
                raise ValueError(f'Training camera trace mismatch {run} frame {frame}')
            if depth:
                active = summary['flat_region_diagnostics']['active_regions_by_depth']
                if any(int(count) for layer, count in active.items() if int(layer) != depth):
                    raise ValueError(f'Wrong active frontier {run} frame {frame}: {active}')
                if int(active[str(depth)]) < 1:
                    raise ValueError(f'Empty fixed frontier {run} frame {frame}')
                changes = summary['flat_region_diagnostics']['reference_changes']
                if any(edit['changed_gaussians'] for edit in changes):
                    raise ValueError(f'Frontier changed during frame {frame}')
            decoded_check = summary['forward_decode_check']
            if decoded_check['max_abs'] > decoded_check['tolerance']:
                raise ValueError(f'Forward/decode render mismatch {run} frame {frame}')
            names = [manifest[k] for k in PACKETS if manifest.get(k)]
            if len(set(names)) != len(names):
                raise ValueError('Duplicate packet name')
            total = manifest_path.stat().st_size + sum((stream/name).stat().st_size for name in names)
            if total != summary['complete_dynamic_update_bytes']:
                raise ValueError(f'Byte accounting mismatch {run} frame {frame}')
            rows.append(dict(frame=frame, psnr=summary['evaluation']['test']['psnr'],
                             complete_bytes=total, motion_bytes=(stream/manifest['motion']).stat().st_size,
                             active_regions=summary['region_count'],
                             training_seconds=summary['training_seconds']))
        state, _ = replay(stream, stream/'reference_frame0.npz', args.num_frames - 1)
        results.append(dict(depth=depth, run=str(run), rows=rows,
            psnr=mean(r['psnr'] for r in rows), complete_kb=mean(r['complete_bytes'] for r in rows)/1000,
            motion_kb=mean(r['motion_bytes'] for r in rows)/1000,
            active_regions=mean(r['active_regions'] for r in rows),
            training_seconds=mean(r['training_seconds'] for r in rows),
            replay_final_sha256=state.digest().hex()))
        print('AUDITED', depth if depth else 'adaptive', results[-1]['psnr'],
              results[-1]['complete_kb'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(dict(scene_root=scene_root, total_frames=args.num_frames,
                                     dynamic_frames=args.num_frames - 1,
                                     first_dynamic_frame=1, last_dynamic_frame=args.num_frames - 1,
                                     reference_sha256=reference_sha, results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
