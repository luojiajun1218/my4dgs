"""Audit a single GS-activity-cost pair from shared frame0 over frames 1-10."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from scripts.archive.analyze_keep_probe_continuous import average, compare


def config_differences(control, treatment, prefix=''):
    if isinstance(control, dict) and isinstance(treatment, dict):
        differences = []
        for key in sorted(set(control) | set(treatment)):
            path = f'{prefix}.{key}' if prefix else key
            differences.extend(config_differences(control.get(key), treatment.get(key), path))
        return differences
    return [] if control == treatment else [dict(key=prefix, control=control, treatment=treatment)]


def rename_keys(value):
    if isinstance(value, dict):
        return {key.replace('keepprobe', 'activity'): rename_keys(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [rename_keys(item) for item in value]
    return value


def analyze_pair(control, treatment, reference):
    configs = [yaml.safe_load((run / 'config.yaml').read_text(encoding='utf8'))
               for run in (control, treatment)]
    differences = config_differences(*configs)
    if {row['key'] for row in differences} != {
            'experiment_name', 'module.quantized_residual.gs_activity_loss_weight'}:
        raise ValueError(f'Expected only experiment name and activity weight differences: {differences}')
    weights = [cfg['module']['quantized_residual']['gs_activity_loss_weight'] for cfg in configs]
    if weights != [0., .001]:
        raise ValueError('This planned pair uses GS activity weights 0 and 0.001')
    if configs[0]['ckpt_path'] != configs[1]['ckpt_path']:
        raise ValueError('Both arms must use the same shared frame0 checkpoint')
    rows, _, summary = compare(control, treatment, reference)
    if [row['frame'] for row in rows] != list(range(1, 11)):
        raise ValueError('Both arms must complete the planned dynamic frames 1-10')
    rows, summary = rename_keys(rows), rename_keys(summary)
    views, phase_views = [], []
    for row in rows:
        frame = row['frame']
        for name, run, weight in (('control', control, 0.), ('activity', treatment, .001)):
            stream = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())['stream']
            cost = stream['gs_activity_cost']
            if cost['weight'] != weight:
                raise ValueError(f'Recorded activity weight differs at frame {frame}')
            for key in ('expected_fraction', 'hard_fraction'):
                row[f'{name}_gs_activity_{key}'] = cost[key]
            for level in stream['activities']['levels']:
                prefix = f"{name}_level{level['level']}"
                row[f'{prefix}_visited_nodes'] = level['visited_count']
                for state, fraction in level['fractions'].items():
                    row[f'{prefix}_{state.lower()}_fraction'] = fraction
            metrics = stream['last_training_metrics']
            row[f'{name}_sample_image_objective'] = (
                metrics['loss/total'] - metrics['loss/storage'] - metrics['loss/gs_activity'])
            row[f'{name}_sample_psnr'] = metrics['psnr']
            row[f'{name}_activity_loss'] = metrics['loss/gs_activity']
            packet = json.loads((run / 'frame_summary' / f'evaluation_{frame:03d}_step200.json').read_text())
            views.extend(dict(arm=name, frame=frame, **item) for item in packet['views'])
            phases = json.loads((run / 'frame_summary' / f'control_{frame:03d}.json').read_text())['quality']
            for stage in ('before_deletion', 'after_deletion', 'after_initialization',
                          'after_motion', 'before_retention', 'after_retention'):
                selected = phases[stage]['validation_probe']['views']
                if {view['camera'] for view in selected} != {
                        'cam17.png', 'cam18.png', 'cam19.png', 'cam20.png'}:
                    raise ValueError(f'Expected all four phase validation views at frame {frame}')
                phase_views.extend(dict(arm=name, frame=frame, stage=stage, **item) for item in selected)
        for key in ('gs_activity_expected_fraction', 'gs_activity_hard_fraction',
                    'sample_image_objective', 'sample_psnr', 'activity_loss'):
            row[f'delta_{key}'] = row[f'activity_{key}'] - row[f'control_{key}']
        for level in range(3):
            for suffix in ('visited_nodes', 'keep_fraction', 'shared_fraction', 'split_fraction'):
                key = f'level{level}_{suffix}'
                row[f'delta_{key}'] = row[f'activity_{key}'] - row[f'control_{key}']
    keys = sorted({key for row in rows for key in row if key != 'frame'})
    view_means = {}
    for camera in sorted({row['camera'] for row in views}):
        means = {arm: {metric: statistics.fmean(row[metric] for row in views
                     if row['arm'] == arm and row['camera'] == camera)
                     for metric in ('psnr', 'ssim', 'l1')}
                 for arm in ('control', 'activity')}
        means['difference'] = {metric: means['activity'][metric] - means['control'][metric]
                               for metric in ('psnr', 'ssim', 'l1')}
        view_means[camera] = means
    stage_pairs = (
        ('deletion', 'before_deletion', 'after_deletion'),
        ('inheritance', 'after_deletion', 'after_initialization'),
        ('motion_fit', 'after_initialization', 'after_motion'),
        ('birth_fit', 'after_motion', 'before_retention'),
        ('retention', 'before_retention', 'after_retention'),
    )
    phase_index = {(item['arm'], item['frame'], item['stage'], item['camera']): item
                   for item in phase_views}
    phase_effects = {}
    for camera in sorted({item['camera'] for item in phase_views}):
        effects = {}
        for name, before, after in stage_pairs:
            means = {
                arm: {metric: statistics.fmean(
                    phase_index[(arm, frame, after, camera)][metric] -
                    phase_index[(arm, frame, before, camera)][metric]
                    for frame in range(1, 11))
                    for metric in ('psnr', 'ssim', 'l1', 'image_objective')}
                for arm in ('control', 'activity')}
            means['difference'] = {metric: means['activity'][metric] - means['control'][metric]
                                   for metric in means['control']}
            effects[name] = means
        phase_effects[camera] = effects
    summary.update(
        complete_short_pair=True, planned_frame_range=[1, 10], config_differences=differences,
        shared_frame0_sha256=hashlib.sha256(Path(configs[0]['ckpt_path']).read_bytes()).hexdigest(),
        means={key: average(rows, key) for key in keys}, endpoint=rows[-1],
        float_view_means=view_means,
        phase_view_effects=phase_effects,
        limitations=[
            'One seed and frames 1-10 only; not a completed long-sequence fix.',
            'Both arms use the same shared frame0, 16 training/4 validation cameras and 200 steps.',
            'Private camera sequences match; method-dependent births can consume different random draws.',
            'Activity measures quantized candidate pose changes, not ground-truth foreground labels.',
            'Level fractions count reachable nodes, not GS fractions or semantic motion labels.',
            'cam00 is passive; no masks or manual image rectangles select the method.',
            'Bytes estimate effective parameters; GS activity is a separate pose-fraction objective.',
            'Sample image objective uses the final sampled train camera, not all 16 camera means.'])
    summary['blocks'] = [dict(start=1, end=10, completed=10, means=summary['means'])]
    return rows, views, phase_views, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('treatment', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows, views, phase_views, summary = analyze_pair(args.control, args.treatment, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gs_activity_pair_summary.json').write_text(json.dumps(summary, indent=2))
    for name, values in (('gs_activity_pair_frames.csv', rows), ('gs_activity_pair_views.csv', views),
                         ('gs_activity_phase_views.csv', phase_views)):
        with (args.output / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(values[0]))
            writer.writeheader()
            writer.writerows(values)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        frames = [row['frame'] for row in rows]
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        for arm in ('control', 'activity'):
            for axis, key in zip(axes[:3], ('validation_psnr', 'actual_changed_fraction', 'hard_bytes')):
                values = [row[f'{arm}_{key}'] for row in rows]
                if key == 'hard_bytes':
                    values = [value / 1024 for value in values]
                axis.plot(frames, values, marker='o', label=arm)
        axes[3].plot(frames, [r['delta_validation_psnr'] for r in rows], marker='o', label='Held-out mean')
        axes[3].plot(frames, [r['delta_test_psnr'] for r in rows], marker='o', label='Passive cam00')
        axes[3].axhline(0, color='gray', linewidth=.8)
        for axis, label in zip(axes, ('Held-out PSNR (dB)', 'Actual changed old-GS fraction',
                                     'Effective parameters (KiB)', 'PSNR difference (dB)')):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
            axis.legend()
        axes[-1].set_xlabel('Dynamic frame')
        fig.suptitle('Same frame0: GS activity cost 0 versus 0.001')
        fig.tight_layout()
        fig.savefig(args.output / 'gs_activity_pair.png', dpi=150)
        plt.close(fig)
    print(json.dumps(dict(means=summary['means'], views=summary['float_view_means']), indent=2))


if __name__ == '__main__':
    main()
