"""Audit the same frame5 state with only descendant-probe gradient isolation."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from analyze_adaptive_long_sequence import analyze
from analyze_gate_keep_refresh_replay import gate_counts
from analyze_gs_activity_pair import config_differences


def read_json(path):
    return json.loads(path.read_text())


def analyze_pair(control, isolated, reference, expected_modes=('keep_probe', 'keep_local_probe')):
    runs = dict(control=control, isolated=isolated)
    frames = list(range(6, 16))
    configs = [yaml.safe_load((run / 'config.yaml').read_text()) for run in runs.values()]
    differences = config_differences(*configs)
    if {item['key'] for item in differences} != {
            'experiment_name', 'ckpt_path', 'module.quantized_residual.motion_gradient_mode'}:
        raise ValueError(f'Unexpected configuration differences: {differences}')
    if [cfg['module']['quantized_residual']['motion_gradient_mode'] for cfg in configs] != list(expected_modes):
        raise ValueError(f'Expected modes {expected_modes}')
    if any(cfg['module']['quantized_residual']['gs_activity_loss_weight'] != .001
           for cfg in configs):
        raise ValueError('Both arms must retain GS activity weight .001')
    lineage = [read_json(run.parent / 'replay_input' / 'lineage.json') for run in runs.values()]
    if lineage[0]['source_sha256'] != lineage[1]['source_sha256']:
        raise ValueError('Different source checkpoints')
    for cfg, item in zip(configs, lineage):
        if item['frames'] != [6, 15] or item['exact_tensors'] != 81 or not item['all_other_fields_identical']:
            raise ValueError('Initial-state comparison failed')
        if hashlib.sha256(Path(cfg['ckpt_path']).read_bytes()).hexdigest() != item['input_sha256']:
            raise ValueError('Replay input changed after preparation')
    manifests = [read_json(run / 'source_snapshot' / 'manifest.json') for run in runs.values()]
    if manifests[0] != manifests[1]:
        raise ValueError('Source snapshots differ')
    for run, manifest in zip(runs.values(), manifests):
        for item in manifest['files']:
            expected = item['sha256']
            if hashlib.sha256((run / 'source_snapshot' / item['path']).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Snapshot changed: {item['path']}")
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Workspace source changed during pair: {item['path']}")
    arm_rows = {arm: analyze(run, reference)[0] for arm, run in runs.items()}
    if any([row['frame'] for row in rows] != frames for rows in arm_rows.values()):
        raise ValueError('Both arms must complete exactly frames6–15')
    rows, views, phase_views = [], [], []
    stages = ('before_deletion', 'after_deletion', 'after_initialization',
              'after_motion', 'before_retention', 'after_retention')
    validation_cameras = {'cam17.png', 'cam18.png', 'cam19.png', 'cam20.png'}
    for index, frame in enumerate(frames):
        traces = [read_json(run / 'frame_summary' / f'training_views_{frame:03d}.json')['views']
                  for run in runs.values()]
        if len(traces[0]) != 200 or traces[0] != traces[1]:
            raise ValueError(f'Different camera order or step budget at frame{frame}')
        row = dict(frame=frame)
        for arm, run in runs.items():
            values = dict(arm_rows[arm][index])
            values.pop('frame')
            values.pop('tree_action')
            stream = read_json(run / 'frame_summary' / f'frame_{frame:03d}.json')['stream']
            cost = stream['gs_activity_cost']
            if cost['weight'] != .001:
                raise ValueError('Unexpected GS activity cost weight')
            values.update(gs_activity_expected=cost['expected_fraction'],
                          gs_activity_hard=cost['hard_fraction'])
            metrics = stream['last_training_metrics']
            values.update(sample_image_objective=(metrics['loss/total'] - metrics['loss/storage'] -
                                                 metrics['loss/gs_activity']),
                          sample_psnr=metrics['psnr'])
            for level in stream['activities']['levels']:
                prefix = f"level{level['level']}"
                values[f'{prefix}_visited_nodes'] = level['visited_count']
                for state, fraction in level['fractions'].items():
                    values[f'{prefix}_{state.lower()}_fraction'] = fraction
            row.update({f'{arm}_{key}': value for key, value in values.items()})
            packet = read_json(run / 'frame_summary' / f'evaluation_{frame:03d}_step200.json')
            for split, cameras in (('validation', validation_cameras), ('test', {'cam00.png'})):
                selected = [item for item in packet['views'] if item['split'] == split]
                if {item['camera'] for item in selected} != cameras:
                    raise ValueError('Unexpected independent evaluation views')
                for metric in ('psnr', 'ssim', 'l1'):
                    if abs(statistics.fmean(item[metric] for item in selected) -
                           packet['evaluation'][split][metric]) > 1e-5:
                        raise ValueError('View metrics do not match aggregate')
            views.extend(dict(arm=arm, frame=frame, **item) for item in packet['views'])
            phases = read_json(run / 'frame_summary' / f'control_{frame:03d}.json')['quality']
            for stage in stages:
                for split, cameras in (('validation_probe', validation_cameras),
                                       ('train_probe', {'cam01.png'})):
                    selected = phases[stage][split]['views']
                    if {item['camera'] for item in selected} != cameras:
                        raise ValueError('Phase probe cameras differ')
                    phase_views.extend(dict(arm=arm, frame=frame, stage=stage, split=split, **item)
                                       for item in selected)
        for key in values:
            a, b = row[f'control_{key}'], row[f'isolated_{key}']
            if a is not None and b is not None:
                row[f'delta_{key}'] = b - a
        rows.append(row)
    view_means = {}
    for camera in sorted({item['camera'] for item in views}):
        values = {arm: {metric: statistics.fmean(item[metric] for item in views
                       if item['arm'] == arm and item['camera'] == camera)
                       for metric in ('psnr', 'ssim', 'l1')} for arm in runs}
        values['difference'] = {metric: values['isolated'][metric] - values['control'][metric]
                                for metric in ('psnr', 'ssim', 'l1')}
        view_means[camera] = values
    phase_index = {(item['arm'], item['frame'], item['stage'], item['camera']): item
                   for item in phase_views}
    phase_effects = {}
    stage_pairs = (('deletion', stages[0], stages[1]), ('inheritance', stages[1], stages[2]),
                   ('motion_fit', stages[2], stages[3]), ('birth_fit', stages[3], stages[4]),
                   ('retention', stages[4], stages[5]))
    for camera in sorted({item['camera'] for item in phase_views}):
        phase_effects[camera] = {}
        for effect, before, after in stage_pairs:
            values = {arm: {metric: statistics.fmean(
                phase_index[(arm, frame, after, camera)][metric] -
                phase_index[(arm, frame, before, camera)][metric] for frame in frames)
                for metric in ('psnr', 'ssim', 'l1', 'image_objective')} for arm in runs}
            values['difference'] = {metric: values['isolated'][metric] - values['control'][metric]
                                    for metric in values['control']}
            phase_effects[camera][effect] = values
    first_start = {}
    for stage in stages[:3]:
        first_start[stage] = {
            camera: {metric: phase_index[('isolated', 6, stage, camera)][metric] -
                     phase_index[('control', 6, stage, camera)][metric]
                     for metric in ('psnr', 'ssim', 'l1', 'image_objective')}
            for camera in phase_effects}
    keys = sorted(key for key in rows[0] if key != 'frame')
    summary = dict(control=str(control), isolated=str(isolated), frames=frames,
        complete=True, source_checkpoint_sha256=lineage[0]['source_sha256'], lineage=lineage,
        config_differences=differences, source_snapshots_identical=True, frozen_source_verified=True,
        tests_passed=manifests[0]['tests_passed'], training_views_identical=True,
        total_paired_steps=2000, means={key: statistics.fmean(row[key] for row in rows)
                                      for key in keys if all(row[key] is not None for row in rows)},
        endpoint=rows[-1], float_view_means=view_means, phase_view_effects=phase_effects,
        first_frame_pretraining_difference=first_start,
        gate_counts={arm: gate_counts(run, frames) for arm, run in runs.items()},
        limitations=['One seed, restart from frame5 and only ten frames; not a long-sequence result.',
            'Checkpoint lacks complete RNG state; two fresh seeded arms compare each other.',
            'Hard forward, gate proxy, initial tensors, costs, camera order and budgets are paired.',
            'Method-dependent births can change later random draws.',
            'cam00 is passive; independent cam17–20 select the intervention.',
            'cam01 phase probe and last sampled train camera do not cover all training views.',
            'Node KEEP fractions and actual GS change are distinct; neither supplies motion labels.',
            'Effective storage estimates parameters rather than encoded bytes.',
            'REF used20 training views; its differences are descriptive only.'])
    return rows, views, phase_views, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('isolated', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows, views, phases, summary = analyze_pair(args.control, args.isolated, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'keep_local_pair_summary.json').write_text(json.dumps(summary, indent=2))
    for name, values in (('paired_frames.csv', rows), ('paired_views.csv', views),
                         ('paired_phase_views.csv', phases)):
        with (args.output / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(values[0]))
            writer.writeheader()
            writer.writerows(values)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        for arm in ('control', 'isolated'):
            for axis, key in zip(axes[:3], ('validation_psnr', 'actual_changed_fraction', 'hard_bytes')):
                values = [row[f'{arm}_{key}'] for row in rows]
                if key == 'hard_bytes':
                    values = [value / 1024 for value in values]
                axis.plot(summary['frames'], values, marker='o', label=arm)
        axes[3].plot(summary['frames'], [row['delta_validation_psnr'] for row in rows], marker='o')
        axes[3].axhline(0, color='gray', linewidth=.8)
        for axis, label in zip(axes, ('Held-out PSNR (dB)', 'Actual changed old-GS fraction',
                                     'Effective parameters (KiB)', 'Held-out difference (dB)')):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
        for axis in axes[:3]:
            axis.legend()
        axes[-1].set_xlabel('Replay frame')
        fig.suptitle('Same frame5 state: descendant KEEP probe ancestor isolation')
        fig.tight_layout()
        fig.savefig(args.output / 'keep_local_pair.png', dpi=150)
        plt.close(fig)
    print(json.dumps(dict(means=summary['means'], views=summary['float_view_means']), indent=2))


if __name__ == '__main__':
    main()
