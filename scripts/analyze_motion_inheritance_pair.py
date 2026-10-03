"""Compare one same-state .6 vs zero applied-motion inheritance replay."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from analyze_adaptive_long_sequence import analyze
from analyze_gs_activity_pair import config_differences
from analyze_keep_candidate_continuous import training_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('zero', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = dict(decay06=args.control, decay00=args.zero)
    configurations = [yaml.safe_load((run / 'config.yaml').read_text()) for run in runs.values()]
    differences = config_differences(*configurations)
    assert {item['key'] for item in differences} == {
        'experiment_name', 'ckpt_path', 'module.quantized_residual.motion_init_decay'}
    assert [cfg['module']['quantized_residual']['motion_init_decay'] for cfg in configurations] == [.6, 0.]
    lineage = [json.loads((run.parent / 'replay_input/lineage.json').read_text()) for run in runs.values()]
    assert lineage[0]['source_sha256'] == lineage[1]['source_sha256']
    for item in lineage:
        assert item['exact_tensors'] == 81 and item['frames'] == [6, 15]
        assert hashlib.sha256(Path(item['input_checkpoint']).read_bytes()).hexdigest() == item['input_sha256']
    manifests = [json.loads((run / 'source_snapshot/manifest.json').read_text()) for run in runs.values()]
    assert manifests[0] == manifests[1]
    for run in runs.values():
        for item in manifests[0]['files']:
            assert hashlib.sha256((run / 'source_snapshot' / item['path']).read_bytes()).hexdigest() == item['sha256']
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
    frames = list(range(6, 16))
    values = {arm: {row['frame']: row for row in analyze(run, args.reference)[0]} for arm, run in runs.items()}
    assert all(sorted(items) == frames for items in values.values())
    trajectories = {arm: {row['frame']: row for row in training_trajectory(run, frames)} for arm, run in runs.items()}
    assert all(sorted(items) == frames for items in trajectories.values())
    rows, phase_rows, view_rows, gate_rows = [], [], [], []
    first_stage = {}
    timings = {arm: dict(validation=[], test=[]) for arm in runs}
    for frame in frames:
        traces = [json.loads((run / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views'] for run in runs.values()]
        assert len(traces[0]) == 200 and traces[0] == traces[1]
        row = dict(frame=frame)
        for arm, run in runs.items():
            record = values[arm][frame]
            for key, value in record.items():
                if key not in ('frame', 'tree_action'):
                    row[f'{arm}_{key}'] = value
            packet = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())
            stream = packet['stream']
            assert stream['candidate_fit']['mode'] == 'keep_candidate_loss'
            assert stream['candidate_fit']['interval'] == 10 and stream['candidate_fit']['weight'] == 1.
            assert stream['candidate_fit']['extra_renders'] <= 14
            row[f'{arm}_extra_renders'] = stream['candidate_fit']['extra_renders']
            row[f'{arm}_candidate_forward_loss_s'] = stream['candidate_fit']['forward_loss_seconds']
            row[f'{arm}_sample_hard_image_objective'] = stream['last_training_metrics']['loss/image']
            for key, value in trajectories[arm][frame].items():
                if key != 'frame':
                    row[f'{arm}_trajectory_{key}'] = value
            evaluation = json.loads((run / 'frame_summary' / f'evaluation_{frame:03d}_step200.json').read_text())
            assert {v['camera'] for v in evaluation['views']} == {f'cam{i:02d}.png' for i in (0, 17, 18, 19, 20)}
            for view in evaluation['views']:
                view_rows.append(dict(frame=frame, arm=arm, **view))
                timings[arm][view['split']].append(view['forward_seconds'])
            control = json.loads((run / 'frame_summary' / f'control_{frame:03d}.json').read_text())
            phase = control['quality']
            for stage, splits in phase.items():
                for split, item in splits.items():
                    for view in item['views']:
                        phase_rows.append(dict(frame=frame, arm=arm, stage=stage, split=split, **view))
            if frame == 6:
                first_stage[arm] = phase
            for diagnostic in control['gradients']:
                gate_rows.append(dict(frame=frame, arm=arm, step=diagnostic['step'],
                    changes=diagnostic['sampled_gate_changes'],
                    hard_gs_changed_fraction=diagnostic['activity']['gs']['fractions']['SHARED'] +
                                             diagnostic['activity']['gs']['fractions']['SPLIT'],
                    training_image_objective=diagnostic['training_metrics']['loss/image'],
                    gradients=diagnostic['gradients']))
        for key in record:
            if key not in ('frame', 'tree_action'):
                a, b = row[f'decay06_{key}'], row[f'decay00_{key}']
                row[f'zero_minus_control_{key}'] = b - a if a is not None and b is not None else None
        rows.append(row)
    initial_checks = []
    for stage in ('before_deletion', 'after_deletion'):
        maximum_difference = 0.
        for split in ('train_probe', 'validation_probe'):
            for a, b in zip(first_stage['decay06'][stage][split]['views'], first_stage['decay00'][stage][split]['views']):
                assert a['camera'] == b['camera']
                for metric in ('psnr', 'ssim', 'l1', 'image_objective'):
                    difference = abs(a[metric] - b[metric])
                    maximum_difference = max(maximum_difference, difference)
                    assert difference <= 1e-5, (stage, split, metric, a, b)
        initial_checks.append(dict(stage=stage, all_five_float_views_exact=maximum_difference == 0.,
                                   max_absolute_difference=maximum_difference, atol=1e-5))
    for split in ('train_probe', 'validation_probe'):
        for a, b in zip(first_stage['decay00']['after_deletion'][split]['views'], first_stage['decay00']['after_initialization'][split]['views']):
            assert a['camera'] == b['camera']
            for metric in ('psnr', 'ssim', 'l1', 'image_objective'):
                assert abs(a[metric] - b[metric]) <= 1e-5, (split, metric, a, b)
    keys = sorted({key for row in rows for key in row if key != 'frame'})
    means = {key: statistics.fmean(row[key] for row in rows if row.get(key) is not None) for key in keys}
    summary = dict(runs={key: str(value) for key, value in runs.items()}, frames=frames, completed10=True,
        config_differences=differences, source_sha256=lineage[0]['source_sha256'],
        same_input_tensors=81, paired_training_steps=2000, initial_checks=initial_checks,
        zero_initialization_render_identity=True, means=means, endpoint=rows[-1],
        extra_renders={arm: sum(row[f'{arm}_extra_renders'] for row in rows) for arm in runs},
        fps={arm: {split: len(times) / sum(times) for split, times in splits.items()} for arm, splits in timings.items()},
        validation_camera_means={arm: {camera: statistics.fmean(item['psnr'] for item in view_rows if item['arm'] == arm and item['camera'] == camera)
                                       for camera in ('cam17.png', 'cam18.png', 'cam19.png', 'cam20.png')} for arm in runs},
        limitations=['Only applied-motion initialization decay differs; gate inheritance/candidate loss/GS costs are fixed.',
                     'Ten-frame freshly-seeded replay does not prove original continuous failure reproduced or continuous recovery.',
                     'Optimization trajectory metrics are changing sampled poses, not a final fixed sixteen-view evaluation.',
                     'Gate transitions/combined gradients are observational evidence, not attribution to a unique update.',
                     'Candidate render counts may differ because actual reachable KEEP counts change; count full costs.'])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, packet in (('motion_inheritance_summary.json', summary), ('gate_gradient_samples.json', gate_rows)):
        (args.output / name).write_text(json.dumps(packet, indent=2))
    for name, data in (('frames.csv', rows), ('phase_views.csv', phase_rows), ('float_views.csv', view_rows)):
        with (args.output / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in data for key in row}))
            writer.writeheader()
            writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for arm in runs:
        for axis, key in zip(axes, ('validation_psnr', 'validation_probe_prediction_delta',
                                    'validation_probe_motion_fit_delta', 'actual_changed_fraction')):
            axis.plot(frames, [row[f'{arm}_{key}'] for row in rows], marker='o', label=arm)
    for axis, label in zip(axes, ('Held-out PSNR (dB)', 'Inheritance phase PSNR change',
                                 'Motion-fit phase PSNR change', 'Actual changed old-GS fraction')):
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('Replay from same frame5')
    fig.tight_layout()
    fig.savefig(args.output / 'motion_inheritance_pair.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(means=means, fps=summary['fps'], extra_renders=summary['extra_renders']), indent=2))


if __name__ == '__main__':
    main()
