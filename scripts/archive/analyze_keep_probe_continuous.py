"""Compare completed common frames of two continuous hierarchy runs."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from scripts.archive.analyze_adaptive_long_sequence import analyze


def average(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def transitions(run, frames):
    counts = {}
    for frame in frames:
        stream = json.loads((run / 'frame_summary' /
                             f'frame_{frame:03d}.json').read_text())['stream']
        action = stream['tree_setup']['action']
        item = counts.setdefault(action, dict(frames=0, initial_keep=0, reopened_keep=0))
        item['frames'] += 1
        matrix = stream['activities']['levels'][0].get('initial_to_final_actions')
        if matrix:
            item['initial_keep'] += sum(matrix[0])
            item['reopened_keep'] += sum(matrix[0][1:])
    return counts


def compare(control, treatment, reference):
    original = {row['frame']: row for row in analyze(control, reference)[0]}
    current = {row['frame']: row for row in analyze(treatment, reference)[0]}
    frames = sorted(original.keys() & current.keys())
    if not frames or frames != list(range(1, frames[-1] + 1)):
        raise ValueError('Expected a contiguous common dynamic-frame prefix starting at 1')
    rows = []
    per_view = []
    view_seconds = dict(validation=[], test=[])
    for frame in frames:
        traces = [json.loads((run / 'frame_summary' /
                              f'training_views_{frame:03d}.json').read_text())['views']
                  for run in (control, treatment)]
        if len(traces[0]) != 200 or traces[0] != traces[1]:
            raise ValueError(f'Camera sequence or 200-step budget differs at frame {frame}')
        row = dict(frame=frame)
        for key, value in current[frame].items():
            if key in ('frame', 'tree_action'):
                continue
            row[f'control_{key}'] = original[frame].get(key)
            row[f'keepprobe_{key}'] = value
            if value is not None and original[frame].get(key) is not None:
                row[f'delta_{key}'] = value - original[frame][key]
        rows.append(row)
        packet = json.loads((treatment / 'frame_summary' /
                             f'evaluation_{frame:03d}_step200.json').read_text())
        views = packet.get('views', [])
        if len(views) != 5:
            raise ValueError(f'Expected four validation views and one passive view at {frame}')
        for split, cameras in (('validation', {'cam17.png', 'cam18.png', 'cam19.png', 'cam20.png'}),
                               ('test', {'cam00.png'})):
            selected = [v for v in views if v['split'] == split]
            if {v['camera'] for v in selected} != cameras:
                raise ValueError(f'Unexpected evaluation views for {split} at {frame}')
            for metric in ('psnr', 'ssim', 'l1'):
                value = statistics.fmean(v[metric] for v in selected)
                if abs(value - packet['evaluation'][split][metric]) > 1e-5:
                    raise ValueError(f'Per-view {metric} does not match aggregate at {frame}')
            view_seconds[split].extend(v['forward_seconds'] for v in selected)
        per_view.extend(dict(frame=frame, **v) for v in views)
    keys = sorted({key for row in rows for key in row if key != 'frame'})
    blocks = []
    for start, end in ((1, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299)):
        selected = [row for row in rows if start <= row['frame'] <= end]
        if selected:
            blocks.append(dict(start=start, end=end, completed=len(selected),
                               means={key: average(selected, key) for key in keys}))
    summary = dict(
        control=str(control), keepprobe=str(treatment), completed=len(frames),
        last_frame=frames[-1], complete_299=(frames == list(range(1, 300))),
        training_views_identical=True, total_paired_steps=len(frames) * 200,
        means={key: average(rows, key) for key in keys}, endpoint=rows[-1], blocks=blocks,
        root_transitions={name: transitions(run, frames) for name, run in
                          (('control', control), ('keepprobe', treatment))},
        keepprobe_view_means={camera: {metric: statistics.fmean(
            row[metric] for row in per_view if row['camera'] == camera)
            for metric in ('psnr', 'ssim', 'l1')}
            for camera in sorted({row['camera'] for row in per_view})},
        keepprobe_aggregate_fps={split: 1 / statistics.fmean(values)
                                  for split, values in view_seconds.items()},
        source_manifest_sha256={name: hashlib.sha256(
            (run / 'source_snapshot' / 'manifest.json').read_bytes()).hexdigest()
            for name, run in (('control', control), ('keepprobe', treatment))},
        limitations=[
            'One seed, same shared frame0 and 16 train/4 validation camera protocol; '
            'method-dependent motion, births and later random draws can diverge.',
            'The original run lacks per-camera float metrics; only its four-view aggregate is paired.',
            'cam00 is passive and did not select the intervention.',
            'Bytes estimate effective parameters, not an encoded stream.',
            'FPS covers materialized-GS render/clamp; excludes loading, pose updates and metrics.',
            'A partial prefix is not a completed long-sequence conclusion.'])
    return rows, per_view, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('treatment', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows, views, summary = compare(args.control, args.treatment, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'continuous_pair_summary.json').write_text(json.dumps(summary, indent=2))
    for name, values in (('continuous_pair_frames.csv', rows), ('keepprobe_float_views.csv', views)):
        with (args.output / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(values[0]))
            writer.writeheader()
            writer.writerows(values)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        frames = [row['frame'] for row in rows]
        for arm in ('control', 'keepprobe'):
            axes[0].plot(frames, [r[f'{arm}_validation_psnr'] for r in rows], label=arm)
            axes[2].plot(frames, [r[f'{arm}_keep_fraction'] for r in rows], label=arm)
            axes[3].plot(frames, [r[f'{arm}_hard_bytes'] / 1024 for r in rows], label=arm)
        axes[1].plot(frames, [r['delta_validation_psnr'] for r in rows], label='Held-out mean')
        axes[1].plot(frames, [r['delta_test_psnr'] for r in rows], label='Passive cam00')
        axes[1].axhline(0, color='gray', linewidth=.8)
        for axis, label in zip(axes, ('Held-out PSNR (dB)', 'PSNR difference (dB)',
                                      'Terminal KEEP fraction', 'Effective parameters (KiB)')):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
            axis.legend()
        axes[-1].set_xlabel('Dynamic frame')
        fig.suptitle(f'Continuous keep_probe regression: {len(rows)}/299 frames')
        fig.tight_layout()
        fig.savefig(args.output / 'continuous_pair.png', dpi=150)
        plt.close(fig)
    print(json.dumps(dict(completed=summary['completed'], last_frame=summary['last_frame'],
                         means=summary['means'], root_transitions=summary['root_transitions']), indent=2))


if __name__ == '__main__':
    main()
