"""Compare two same-state local replays; require paired frames and camera order."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from analyze_adaptive_long_sequence import analyze


def gate_counts(run, frames):
    totals = dict(initial_keep=0, reopened_keep=0, refreshed_root=0,
                  refreshed_root_final_keep=0, refreshed_root_final_open=0)
    for frame in frames:
        packet = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())
        stream = packet['stream']
        root = stream['activities']['levels'][0]
        transitions = root.get('initial_to_final_actions')
        if transitions:
            totals['initial_keep'] += sum(transitions[0])
            totals['reopened_keep'] += sum(transitions[0][1:])
        refreshed = root.get('refreshed_keep_final_counts')
        if refreshed:
            totals['refreshed_root_final_keep'] += refreshed[0]
            totals['refreshed_root_final_open'] += sum(refreshed[1:])
        reset = stream['tree_setup'].get('reset_keep_gate_nodes', [])
        if reset:
            totals['refreshed_root'] += reset[0]
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('resetkeep', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--treatment-name', default='resetkeep')
    args = parser.parse_args()
    if args.treatment_name == 'control':
        parser.error('Treatment name must differ from control')
    arms = {}
    for name in ('control', 'resetkeep'):
        arms[name] = analyze(getattr(args, name), args.reference)
    control, reset = (arms[name][0] for name in ('control', 'resetkeep'))
    frames = [row['frame'] for row in control]
    if not frames or frames != [row['frame'] for row in reset]:
        raise ValueError('Both arms must contain the same completed replay frames')
    lineage = [json.loads((getattr(args, arm).parent / 'replay_input' /
                           'lineage.json').read_text())
               for arm in ('control', 'resetkeep')]
    if lineage[0]['source_sha256'] != lineage[1]['source_sha256']:
        raise ValueError('Replay inputs must originate from the same checkpoint')
    for item in lineage:
        start, end = item['replay_frames']
        if frames != list(range(start, end + 1)):
            raise ValueError('Completed frames do not match the planned replay range')
    traces_equal = True
    for frame in frames:
        traces = [json.loads((getattr(args, arm) / 'frame_summary' /
                              f'training_views_{frame:03d}.json').read_text())['views']
                  for arm in ('control', 'resetkeep')]
        if len(traces[0]) != 200 or traces[0] != traces[1]:
            traces_equal = False
    if not traces_equal:
        raise ValueError('Camera order or step budget differs between arms')
    keys = [key for key in control[0] if key != 'tree_action' and
            all(row.get(key) is not None for row in control + reset)]
    differences = [{key: other[key] - original[key] for key in keys}
                   for original, other in zip(control, reset)]
    for frame, row in zip(frames, differences):
        row['frame'] = frame
    summary = dict(
        frames=frames, source_checkpoint_sha256=lineage[0]['source_sha256'],
        treatment_lineage=lineage[1],
        planned_frame_range_verified=True, training_views_identical=True, steps_per_frame=200,
        control=arms['control'][1], resetkeep=arms['resetkeep'][1],
        paired_mean_difference={key: statistics.fmean(row[key] for row in differences)
                                for key in keys if key != 'frame'},
        paired_endpoint_difference=differences[-1],
        gate_counts={arm: gate_counts(getattr(args, arm), frames)
                     for arm in ('control', 'resetkeep')},
        limitations=['Both arms restart with seed0; original checkpoint has no full RNG state.',
                     'Local replay tests recovery from frame99, not prevention over 300 frames.',
                     'The treatment changes gate priors or candidate gradients as recorded in lineage. '
                     'Separate prediction and fit probes identify their observed contributions.',
                     'cam00 is passive; held-out cam17-cam20 assesses the intervention.'])
    summary['treatment_name'] = args.treatment_name
    if args.treatment_name != 'resetkeep':
        summary[args.treatment_name] = summary.pop('resetkeep')
        summary['gate_counts'][args.treatment_name] = summary['gate_counts'].pop('resetkeep')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'paired_summary.json').write_text(json.dumps(summary, indent=2))
    with (args.output / 'paired_differences.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=sorted(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for arm, rows in (('control', control), ('resetkeep', reset)):
            label = args.treatment_name if arm == 'resetkeep' else arm
            axes[0].plot(frames, [r['validation_psnr'] for r in rows], marker='o', label=label)
            axes[1].plot(frames, [r['keep_fraction'] for r in rows], marker='o', label=label)
            axes[2].plot(frames, [r['hard_bytes'] / 1024 for r in rows], marker='o', label=label)
        axes[0].set_ylabel('Held-out PSNR (dB)')
        axes[1].set_ylabel('Terminal KEEP fraction')
        axes[2].set_ylabel('Effective parameters (KiB)')
        axes[2].set_xlabel('Replay frame')
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend()
        fig.suptitle(f'Same frame99 state: {args.treatment_name} versus original')
        fig.tight_layout()
        fig.savefig(args.output / 'paired_replay.png', dpi=150)
        plt.close(fig)
    print(json.dumps({key: summary[key] for key in
                      ('frames', 'training_views_identical', 'paired_mean_difference',
                       'paired_endpoint_difference', 'gate_counts')}, indent=2))


if __name__ == '__main__':
    main()
