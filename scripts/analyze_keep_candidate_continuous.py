"""Read completed continuous candidate runs and their same-frame anchors."""
import argparse
import csv
import json
import statistics
import tempfile
from pathlib import Path

from analyze_adaptive_long_sequence import analyze
from analyze_keep_probe_continuous import average


def completed_rows(run, reference):
    """Exclude a partially written next frame without changing any run files."""
    with tempfile.TemporaryDirectory(prefix='recongs_completed_analysis_', dir='/tmp') as folder:
        summary = Path(folder) / 'frame_summary'
        summary.mkdir()
        for path in (run / 'frame_summary').glob('frame_*.json'):
            try:
                packet = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if packet['frame'] == 0 or 'wall' not in packet.get('timing', {}):
                continue
            (summary / path.name).symlink_to(path.resolve())
            control = path.parent / f'control_{packet["frame"]:03d}.json'
            if control.exists():
                (summary / control.name).symlink_to(control.resolve())
        return analyze(Path(folder), reference)[0]


def training_trajectory(run, frames):
    """Reuse logged optimization renders; this adds no scene renders."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(run), size_guidance={'scalars': 0},
                                   purge_orphaned_data=False).Reload()
    available = set(accumulator.Tags()['scalars'])
    tags = ['psnr', 'loss/rgb', 'loss/dssim', 'loss/image', 'loss/keep_candidate']
    events = {tag: accumulator.Scalars(tag) for tag in tags if tag in available}
    rows = []
    for index, frame in enumerate(frames):
        start = index * 200
        if any(len(events.get(tag, [])) < start + 200 for tag in ('psnr', 'loss/rgb', 'loss/dssim')):
            break
        chunk = {tag: values[start:start + 200] for tag, values in events.items()
                 if tag != 'loss/keep_candidate' and len(values) >= start + 200}
        for tag, values in chunk.items():
            if [value.step for value in values] != list(range(1, 201)):
                raise ValueError(f'Unexpected scalar step sequence: {run}, {frame}, {tag}')
        if 'loss/image' not in chunk:
            image = [.8 * a.value + .2 * b.value
                     for a, b in zip(chunk['loss/rgb'], chunk['loss/dssim'])]
        else:
            image = [value.value for value in chunk['loss/image']]
        psnr = [value.value for value in chunk['psnr']]
        views = json.loads((run / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views']
        expected = {f'cam{camera:02d}.png' for camera in range(1, 17)}
        if set(views) != expected:
            raise ValueError(f'Training samples do not cover each train camera: {frame}')
        row = dict(frame=frame, last16_distinct_cameras=len(set(views[-16:])))
        for stage, lower, upper in (('all', 0, 200), ('motion', 0, 133),
                                     ('birth', 133, 200), ('last16', 184, 200)):
            row[f'{stage}_psnr'] = statistics.fmean(psnr[lower:upper])
            row[f'{stage}_image_objective'] = statistics.fmean(image[lower:upper])
        for camera in sorted(expected):
            indices = [index for index, view in enumerate(views) if view == camera]
            row[f'{camera}_samples'] = len(indices)
            row[f'{camera}_psnr'] = statistics.fmean(psnr[index] for index in indices)
            row[f'{camera}_image_objective'] = statistics.fmean(image[index] for index in indices)
        rows.append(row)
    return rows


def compare(run, anchors, reference):
    source_rows = {name: {row['frame']: row for row in completed_rows(path, reference)}
                   for name, path in dict(candidate=run, **anchors).items()}
    frames = sorted(source_rows['candidate'])
    if frames != list(range(1, frames[-1] + 1)):
        raise ValueError('Candidate must be a contiguous completed prefix starting at frame 1')
    rows, views, phases = [], [], []
    render_seconds = {}
    matched = {}
    trajectories = {}
    for name, path in dict(candidate=run, **anchors).items():
        available_frames = [frame for frame in frames if frame in source_rows[name]]
        trajectories[name] = training_trajectory(path, available_frames)
        render_seconds[name] = dict(validation=[], test=[])
        matched[name] = 0
    for frame in frames:
        row = dict(frame=frame)
        candidate_trace = json.loads((run / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views']
        if len(candidate_trace) != 200 or set(candidate_trace) != {f'cam{i:02d}.png' for i in range(1, 17)}:
            raise ValueError(f'Unexpected candidate training protocol: {frame}')
        for name, path in dict(candidate=run, **anchors).items():
            if frame not in source_rows[name]:
                continue
            trace = json.loads((path / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views']
            if trace != candidate_trace:
                raise ValueError(f'Camera sequence differs: {name}, {frame}')
            matched[name] += 200
            packet = json.loads((path / 'frame_summary' / f'frame_{frame:03d}.json').read_text())
            stream = packet['stream']
            base = source_rows[name][frame]
            for key, value in base.items():
                # Phase validation means cover different camera sets across historical runs.
                if key != 'frame' and key != 'tree_action' and not key.startswith('validation_probe'):
                    row[f'{name}_{key}'] = value
            for key, value in stream['storage_bytes_by_class'].items():
                row[f'{name}_{key}'] = value
            row[f'{name}_expected_bytes'] = stream['expected_storage_bytes']
            for key in ('expected_fraction', 'hard_fraction'):
                row[f'{name}_gs_activity_{key}'] = stream.get('gs_activity_cost', {}).get(key)
            for level in stream['activities']['levels']:
                row[f'{name}_level{level["level"]}_visited'] = level['visited_count']
                for action, value in level['fractions'].items():
                    row[f'{name}_level{level["level"]}_{action}'] = value
            fit = stream.get('candidate_fit', {})
            row[f'{name}_extra_renders'] = fit.get('extra_renders', 0)
            row[f'{name}_candidate_forward_loss_s'] = fit.get('forward_loss_seconds', 0)
            evaluation = json.loads((path / 'frame_summary' / f'evaluation_{frame:03d}_step200.json').read_text())
            for view in evaluation.get('views', []):
                views.append(dict(arm=name, frame=frame, **view))
                render_seconds[name][view['split']].append(view['forward_seconds'])
            phase_path = path / 'frame_summary' / f'control_{frame:03d}.json'
            if phase_path.exists():
                control = json.loads(phase_path.read_text())['quality']
                for stage, splits in control.items():
                    for split, entry in splits.items():
                        for view in entry.get('views', []):
                            phases.append(dict(arm=name, frame=frame, stage=stage, split=split, **view))
        for name in anchors:
            for key in ('validation_psnr', 'test_psnr', 'hard_bytes', 'actual_changed_fraction', 'training_s', 'wall_s'):
                a, b = row.get(f'candidate_{key}'), row.get(f'{name}_{key}')
                row[f'candidate_minus_{name}_{key}'] = a - b if a is not None and b is not None else None
        rows.append(row)
    trajectory_counts = {name: len(items) for name, items in trajectories.items()}
    common_trajectory_frames = set(frames)
    for trajectory in trajectories.values():
        common_trajectory_frames &= {item['frame'] for item in trajectory}
    for name, trajectory in trajectories.items():
        by_frame = {item['frame']: item for item in trajectory}
        for row in rows:
            if row['frame'] not in common_trajectory_frames:
                continue
            for key, value in by_frame.get(row['frame'], {}).items():
                if key != 'frame':
                    row[f'{name}_trajectory_{key}'] = value
    fields = sorted({key for row in rows for key in row if key != 'frame'})
    blocks = []
    for lower, upper in ((1, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299)):
        selected = [row for row in rows if lower <= row['frame'] <= upper]
        if selected:
            blocks.append(dict(start=lower, end=upper, completed=len(selected),
                               means={key: average(selected, key) for key in fields}))
    summary = dict(run=str(run), anchors={key: str(value) for key, value in anchors.items()},
        completed=len(frames), last_frame=frames[-1], complete299=frames[-1] == 299,
        matched_training_steps=matched, means={key: average(rows, key) for key in fields},
        training_trajectory_logged_frames=trajectory_counts,
        training_trajectory_common_frames=sorted(common_trajectory_frames),
        blocks=blocks, endpoint=rows[-1], total_extra_renders=sum(row['candidate_extra_renders'] for row in rows),
        extra_render_fraction=sum(row['candidate_extra_renders'] for row in rows) / (200 * len(rows)),
        aggregate_fps={name: {split: len(values) / sum(values) if values else None
                             for split, values in splits.items()}
                       for name, splits in render_seconds.items()},
        limitations=[
            'Original core/keep_probe are overall method anchors: GS activity lambda differs (0 vs .001).',
            'GS-activity-only failed anchor is available only through frame25.',
            'REF trains on all20 cameras; its cameras17-20 are not held-out validation.',
            'Training trajectory means reuse 200 changing optimization renders; per-camera trajectory means cover all16. Last16 samples may repeat cameras and are not a fixed final-pose evaluation.',
            'Historical phase validation means have different view sets; raw same-camera rows are retained without subtracting incompatible aggregate means.',
            'FPS is synchronized materialized-GS forward/clamp, excluding data/pose/metrics/save.',
            'Added render graphs and measured full training/wall time are counted; no equal-time claim.',
            'Effective parameter bytes exclude membership metadata, file headers and training files.',
            'One seed and GPU/birth branching nondeterminism; short-repeat effect amplitudes varied.'])
    return rows, views, phases, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--original-core', type=Path, required=True)
    parser.add_argument('--keep-probe', type=Path, required=True)
    parser.add_argument('--gs-activity-only', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    anchors = dict(original_core=args.original_core, keep_probe=args.keep_probe)
    if args.gs_activity_only:
        anchors['gs_activity_only'] = args.gs_activity_only
    rows, views, phases, summary = compare(args.run, anchors, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'candidate_continuous_summary.json').write_text(json.dumps(summary, indent=2))
    for name, values in (('candidate_continuous_frames.csv', rows), ('float_views.csv', views), ('phase_views.csv', phases)):
        if values:
            with (args.output / name).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in values for key in row}))
                writer.writeheader()
                writer.writerows(values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(11, 16), sharex=True)
    frames = [row['frame'] for row in rows]
    for name in ('candidate', 'original_core', 'keep_probe', 'gs_activity_only'):
        if name != 'candidate' and name not in anchors:
            continue
        for axis, key, scale in ((axes[0], 'validation_psnr', 1), (axes[2], 'actual_changed_fraction', 1),
                                 (axes[3], 'hard_bytes', 1024), (axes[4], 'training_s', 1),
                                 (axes[5], 'trajectory_last16_psnr', 1)):
            axis.plot(frames, [row.get(f'{name}_{key}') / scale if row.get(f'{name}_{key}') is not None else None
                              for row in rows], label=name)
    for name in anchors:
        axes[1].plot(frames, [row[f'candidate_minus_{name}_validation_psnr'] for row in rows], label=name)
    axes[1].axhline(0, color='gray', linewidth=.8)
    for axis, label in zip(axes, ('Held-out PSNR (dB)', 'Candidate - anchor (dB)', 'Old GS changed fraction',
                                 'Effective parameters (KiB)', 'Full training seconds', 'Last16 trajectory PSNR (dB)')):
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
        axis.legend()
    axes[-1].set_xlabel('Continuous frame from shared frame0')
    fig.suptitle(f'Candidate own image loss: completed {len(frames)}/299')
    fig.tight_layout()
    fig.savefig(args.output / 'candidate_continuous.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(completed=summary['completed'], complete299=summary['complete299'],
                         means=summary['means'], aggregate_fps=summary['aggregate_fps']), indent=2))


if __name__ == '__main__':
    main()
