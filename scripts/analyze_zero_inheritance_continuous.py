"""Read-only zero-inheritance continuous analysis with a short .6 anchor."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml

from analyze_gs_activity_pair import config_differences
from analyze_keep_candidate_continuous import compare, completed_rows, training_trajectory


def mean(items, key):
    values = [item[key] for item in items if item.get(key) is not None]
    return statistics.fmean(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--original-core', type=Path, required=True)
    parser.add_argument('--keep-probe', type=Path, required=True)
    parser.add_argument('--candidate-decay06', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old_cfg, new_cfg = [yaml.safe_load((run / 'config.yaml').read_text())
                        for run in (args.candidate_decay06, args.run)]
    differences = config_differences(old_cfg, new_cfg)
    assert {item['key'] for item in differences} == {
        'experiment_name', 'module.quantized_residual.motion_init_decay'}
    assert old_cfg['module']['quantized_residual']['motion_init_decay'] == .6
    assert new_cfg['module']['quantized_residual']['motion_init_decay'] == 0.
    assert old_cfg['ckpt_path'] == new_cfg['ckpt_path']
    anchors = dict(original_core=args.original_core, keep_probe=args.keep_probe)
    rows, views, phases, summary = compare(args.run, anchors, args.reference)
    old_rows = {row['frame']: row for row in completed_rows(args.candidate_decay06, args.reference)}
    common = [row['frame'] for row in rows if row['frame'] in old_rows]
    old_trajectory = {row['frame']: row for row in training_trajectory(args.candidate_decay06, common)}
    paired_fields = ('validation_psnr', 'validation_ssim', 'test_psnr', 'test_ssim',
                     'actual_changed_fraction', 'keep_fraction', 'local_fraction',
                     'hard_bytes', 'training_s', 'wall_s', 'born', 'deleted', 'total_gs')
    for row in rows:
        frame = row['frame']
        if frame not in old_rows:
            continue
        trace_a, trace_b = [json.loads((run / 'frame_summary' /
                            f'training_views_{frame:03d}.json').read_text())['views']
                            for run in (args.run, args.candidate_decay06)]
        assert len(trace_a) == 200 and trace_a == trace_b
        old = old_rows[frame]
        for key in paired_fields:
            row[f'candidate_decay06_{key}'] = old.get(key)
            a, b = row.get(f'candidate_{key}'), old.get(key)
            row[f'zero_minus_decay06_{key}'] = a - b if a is not None and b is not None else None
        for key, value in old_trajectory.get(frame, {}).items():
            if key != 'frame':
                row[f'candidate_decay06_trajectory_{key}'] = value
        packet = json.loads((args.candidate_decay06 / 'frame_summary' /
                             f'evaluation_{frame:03d}_step200.json').read_text())
        views.extend(dict(frame=frame, arm='candidate_decay06', **view)
                     for view in packet.get('views', []))
        control = json.loads((args.candidate_decay06 / 'frame_summary' /
                             f'control_{frame:03d}.json').read_text())['quality']
        for stage, splits in control.items():
            for split, item in splits.items():
                phases.extend(dict(arm='candidate_decay06', frame=frame, stage=stage,
                                   split=split, **view) for view in item.get('views', []))
    fields = sorted({key for row in rows for key in row if key != 'frame'})
    selected = [row for row in rows if row['frame'] in common]
    summary.update(
        experiment='candidate own-loss + GS-cost; only motion_init_decay0',
        exact_config_differences_from_failed_decay06=differences,
        shared_frame0_sha256=hashlib.sha256(Path(new_cfg['ckpt_path']).read_bytes()).hexdigest(),
        candidate_decay06_available=dict(run=str(args.candidate_decay06), frames=common,
            paired_steps=200 * len(common), full299=False,
            means={key: mean(selected, key) for key in fields}),
        means={key: mean(rows, key) for key in fields},
        endpoint=rows[-1],
        limitations=summary['limitations'] + [
            'Failed same-config decay .6 anchor has only frames1-25; its paired differences use exactly that prefix, never a full299 comparison.',
            'No gate inheritance/effective-prior filtering/local candidate changes.',
            'Unchanged83-tested production code; partial short positive results do not establish continuous recovery.'])
    for block in summary['blocks']:
        block_rows = [row for row in rows if block['start'] <= row['frame'] <= block['end']]
        block['means'] = {key: mean(block_rows, key) for key in fields}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'zero_inheritance_continuous_summary.json').write_text(json.dumps(summary, indent=2))
    for name, data in (('frames.csv', rows), ('float_views.csv', views), ('phase_views.csv', phases)):
        with (args.output / name).open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in data for key in row}))
            writer.writeheader()
            writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 1, figsize=(11, 16), sharex=True)
    frames = [row['frame'] for row in rows]
    for name in ('candidate', 'original_core', 'keep_probe', 'candidate_decay06'):
        for axis, key, scale in ((axes[0], 'validation_psnr', 1),
                                 (axes[2], 'actual_changed_fraction', 1),
                                 (axes[3], 'hard_bytes', 1024),
                                 (axes[4], 'training_s', 1),
                                 (axes[5], 'trajectory_all_psnr', 1)):
            axis.plot(frames, [row.get(f'{name}_{key}') / scale
                              if row.get(f'{name}_{key}') is not None else None for row in rows],
                      label='zero_inheritance' if name == 'candidate' else name)
    for name in ('original_core', 'keep_probe'):
        axes[1].plot(frames, [row[f'candidate_minus_{name}_validation_psnr'] for row in rows], label=name)
    axes[1].plot(frames, [row.get('zero_minus_decay06_validation_psnr') for row in rows],
                 label='failed decay .6, prefix25 only')
    axes[1].axhline(0, color='gray', linewidth=.8)
    for axis, label in zip(axes, ('Held-out PSNR (dB)', 'Zero - anchor (dB)',
                                 'Actual changed old-GS fraction', 'Effective parameters (KiB)',
                                 'Full training seconds', 'Training trajectory PSNR (dB)')):
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('Continuous frame from shared frame0')
    fig.suptitle(f'Zero motion inheritance: {len(rows)}/299 completed')
    fig.tight_layout()
    fig.savefig(args.output / 'zero_inheritance_continuous.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(completed=summary['completed'], complete299=summary['complete299'],
        means=summary['means'], blocks=summary['blocks'], fps=summary['aggregate_fps'],
        candidate_decay06_common_frames=len(common)), indent=2))


if __name__ == '__main__':
    main()
