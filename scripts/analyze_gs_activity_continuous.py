"""Compare the GS-cost continuous run without mixing unequal phase probes."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from analyze_adaptive_long_sequence import analyze
from analyze_gs_activity_pair import config_differences, rename_keys
from analyze_keep_probe_continuous import average, compare


def summarize(control, activity, hard_path, reference):
    configs = [yaml.safe_load((run / 'config.yaml').read_text())
               for run in (control, activity)]
    configs[0]['module']['quantized_residual'].setdefault('gs_activity_loss_weight', 0.)
    differences = config_differences(*configs)
    expected = {'experiment_name', 'module.quantized_residual.gs_activity_loss_weight',
                'module.quantized_diagnostics.control_quality_camera_names',
                'module.quantized_diagnostics.gradient_steps'}
    if {r['key'] for r in differences} != expected:
        raise ValueError(f'Unexpected protocol changes: {differences}')
    rows, _, summary = compare(control, activity, reference)
    rows, summary = rename_keys(rows), rename_keys(summary)
    # The control phase probe has only cam17, the new run has four validation
    # cameras. Comparing those aggregate phase means would be misleading.
    for row in rows:
        for key in list(row):
            if 'validation_probe_' in key:
                del row[key]
    third = {row['frame']: row for row in analyze(hard_path, reference)[0]}
    views, phases = [], []
    for row in rows:
        frame = row['frame']
        for key in ('validation_psnr', 'validation_ssim', 'test_psnr', 'test_ssim',
                    'actual_changed_fraction', 'hard_bytes', 'keep_fraction', 'training_s', 'wall_s'):
            row[f'hard_path_{key}'] = third[frame][key]
            row[f'activity_minus_hard_path_{key}'] = row[f'activity_{key}'] - third[frame][key]
        for arm, run in (('control', control), ('activity', activity)):
            trace = json.loads((run / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views']
            third_trace = json.loads((hard_path / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views']
            if trace != third_trace:
                raise ValueError(f'Hard-path camera sequence differs at frame {frame}')
            stream = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())['stream']
            for level in stream['activities']['levels']:
                prefix = f"{arm}_level{level['level']}"
                row[f'{prefix}_visited_nodes'] = level['visited_count']
                for state, value in level['fractions'].items():
                    row[f'{prefix}_{state.lower()}_fraction'] = value
            if arm == 'activity':
                cost = stream['gs_activity_cost']
                if cost['weight'] != .001:
                    raise ValueError(f'Activity weight changed at {frame}')
                row['activity_gs_activity_expected_fraction'] = cost['expected_fraction']
                row['activity_gs_activity_hard_fraction'] = cost['hard_fraction']
                metrics = stream['last_training_metrics']
                row['activity_sample_image_objective'] = (
                    metrics['loss/total'] - metrics['loss/storage'] - metrics['loss/gs_activity'])
                row['activity_loss'] = metrics['loss/gs_activity']
            packet = json.loads((run / 'frame_summary' / f'evaluation_{frame:03d}_step200.json').read_text())
            views.extend(dict(arm=arm, frame=frame, **view) for view in packet['views'])
            quality = json.loads((run / 'frame_summary' / f'control_{frame:03d}.json').read_text())['quality']
            for stage in ('before_deletion', 'after_deletion', 'after_initialization',
                          'after_motion', 'before_retention', 'after_retention'):
                selected = quality[stage]['validation_probe']['views']
                cameras = ({'cam17.png'} if arm == 'control' else
                           {'cam17.png', 'cam18.png', 'cam19.png', 'cam20.png'})
                if {view['camera'] for view in selected} != cameras:
                    raise ValueError(f'Unexpected phase cameras at {frame}')
                phases.extend(dict(arm=arm, frame=frame, stage=stage, **view) for view in selected)
    view_means = {}
    for camera in sorted({v['camera'] for v in views}):
        entry = {arm: {metric: statistics.fmean(v[metric] for v in views
                         if v['camera'] == camera and v['arm'] == arm)
                       for metric in ('psnr', 'ssim', 'l1')} for arm in ('control', 'activity')}
        entry['difference'] = {metric: entry['activity'][metric] - entry['control'][metric]
                               for metric in entry['control']}
        view_means[camera] = entry
    keys = sorted(set(rows[0]) - {'frame'})
    summary['means'] = {key: average(rows, key) for key in keys}
    summary['endpoint'] = rows[-1]
    for block in summary['blocks']:
        selected = [r for r in rows if block['start'] <= r['frame'] <= block['end']]
        block['means'] = {key: average(selected, key) for key in keys}
    summary.update(hard_path=str(hard_path), config_differences=differences,
                   float_view_means=view_means,
                   shared_frame0_sha256=hashlib.sha256(Path(configs[0]['ckpt_path']).read_bytes()).hexdigest(),
                   limitations=[
                       'One seed; same frame0, 16 train/4 validation, 200 steps and private camera sequence.',
                       'Method-dependent births and nondeterministic CUDA can diverge.',
                       'Only GS-cost weight changes optimization; phase/gradient diagnostics are expanded.',
                       'Control phase probe has cam17 only; treatment has four cameras. No cross-camera phase aggregate subtraction.',
                       'Hard-path is a useful anchor but differs in gradient mode as well as activity cost.',
                       'REF trained on 20 cameras; REF gap is descriptive, not a paired method effect.',
                       'Bytes are effective parameter estimates; activity counts poses, not semantic foreground.',
                       'A partial prefix cannot establish the full 299-frame quality target.'])
    return rows, views, phases, summary


def main():
    parser = argparse.ArgumentParser()
    for name in ('control', 'activity', 'hard_path', 'reference'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows, views, phases, summary = summarize(args.control, args.activity, args.hard_path, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gs_activity_continuous_summary.json').write_text(json.dumps(summary, indent=2))
    for name, values in (('gs_activity_continuous_frames.csv', rows),
                         ('gs_activity_continuous_views.csv', views),
                         ('gs_activity_continuous_phases.csv', phases)):
        with (args.output / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(values[0]))
            writer.writeheader()
            writer.writerows(values)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        frames = [r['frame'] for r in rows]
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for arm, label in (('control', 'keep_probe: activity cost 0'),
                           ('activity', 'keep_probe: activity cost .001'),
                           ('hard_path', 'hard_path: activity cost 0')):
            for axis, key in zip((axes[0], axes[2], axes[3]),
                                 ('validation_psnr', 'actual_changed_fraction', 'hard_bytes')):
                values = [r[f'{arm}_{key}'] for r in rows]
                if key == 'hard_bytes':
                    values = [value / 1024 for value in values]
                axis.plot(frames, values, label=label)
        axes[1].plot(frames, [r['delta_validation_psnr'] for r in rows], label='Validation vs keep_probe')
        axes[1].plot(frames, [r['activity_minus_hard_path_validation_psnr'] for r in rows], label='Validation vs hard_path')
        axes[1].plot(frames, [r['activity_delta_psnr'] for r in rows], label='Passive cam00 vs REF (20 train)')
        axes[1].axhline(0, color='gray', linewidth=.8)
        for axis, label in zip(axes, ('Held-out PSNR (dB)', 'PSNR difference (dB)',
                                     'Actual changed old-GS fraction', 'Effective parameters (KiB)')):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        axes[-1].set_xlabel('Dynamic frame')
        fig.suptitle(f'GS activity cost continuous validation: {len(rows)}/299 frames')
        fig.tight_layout()
        fig.savefig(args.output / 'gs_activity_continuous.png', dpi=150)
        plt.close(fig)
    print(json.dumps(dict(completed=summary['completed'], means=summary['means'],
                          views=summary['float_view_means']), indent=2))


if __name__ == '__main__':
    main()
