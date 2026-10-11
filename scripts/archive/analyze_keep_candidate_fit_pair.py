"""Same-state candidate-own photometric fitting, including counted render costs."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from scripts.archive.analyze_keep_local_probe_pair import analyze_pair, read_json


def rename(value):
    if isinstance(value, dict):
        return {key.replace('isolated', 'ownloss'): rename(item) for key, item in value.items()}
    if isinstance(value, list):
        return [rename(item) for item in value]
    if value == 'isolated':
        return 'ownloss'
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('ownloss', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows, views, phases, summary = analyze_pair(args.control, args.ownloss, args.reference,
                                               expected_modes=('keep_probe', 'keep_candidate_loss'))
    rows, views, phases, summary = map(rename, (rows, views, phases, summary))
    for row in rows:
        for name, run in (('control', args.control), ('ownloss', args.ownloss)):
            stream = read_json(run / 'frame_summary' / f"frame_{row['frame']:03d}.json")['stream']
            cost = stream['candidate_fit']
            if cost['interval'] != 10 or cost['weight'] != 1.:
                raise ValueError('Unexpected candidate fitting schedule or weight')
            if cost['extra_renders'] > 14 or (name == 'control' and cost['extra_renders'] != 0):
                raise ValueError('Candidate render budget exceeded')
            row[f'{name}_extra_renders'] = cost['extra_renders']
            row[f'{name}_extra_forward_loss_seconds'] = cost['forward_loss_seconds']
            row[f'{name}_last_candidate_nodes'] = cost['last_nodes']
            row[f'{name}_candidate_image_loss_sum'] = cost['image_loss_sum']
            row[f'{name}_sample_hard_image_loss'] = stream['last_training_metrics']['loss/image']
        for key in ('extra_renders', 'extra_forward_loss_seconds', 'last_candidate_nodes',
                    'candidate_image_loss_sum', 'sample_hard_image_loss'):
            row[f'delta_{key}'] = row[f'ownloss_{key}'] - row[f'control_{key}']
    summary['means'] = {key: statistics.fmean(row[key] for row in rows)
                        for key in rows[0] if key != 'frame' and all(row[key] is not None for row in rows)}
    summary['endpoint'] = rows[-1]
    means = summary['means']
    summary['compute_cost'] = dict(base_training_renders_per_frame=200,
        ownloss_extra_render_backward_graphs=sum(row['ownloss_extra_renders'] for row in rows),
        ownloss_extra_render_fraction=means['ownloss_extra_renders'] / 200,
        extra_candidate_forward_loss_seconds=sum(row['ownloss_extra_forward_loss_seconds'] for row in rows),
        total_training_time_change_fraction=means['ownloss_training_s'] / means['control_training_s'] - 1,
        total_wall_time_change_fraction=means['ownloss_wall_s'] / means['control_wall_s'] - 1,
        effective_parameter_change_fraction=means['ownloss_hard_bytes'] / means['control_hard_bytes'] - 1,
        limits='Forward-loss timing omits auxiliary backward and pose construction. Full training/wall '
               'time includes the complete intervention; added render backward graphs are counted.')
    summary['limitations'].extend([
        'Candidate fitting changes only closed-motion learning: hard-path image Jacobian plus sparse '
        'true candidate image loss. Gate proxy remains unchanged.',
        'Reachable KEEP candidates are rendered together; occlusion interactions can differ from '
        'one-node audit. Only their own motion receives this auxiliary gradient.',
        'Added candidate renders are7% of200 base training renders at interval10, not free compute.',
        'This is a ten-frame comparison; positive short quality does not establish continuous recovery.'])
    summary['limitations'][2] = ('Hard forward, gate proxy, initial tensors, costs, camera order and '
        '200 base optimization steps are paired; treatment adds counted candidate render graphs.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'keep_candidate_pair_summary.json').write_text(json.dumps(summary, indent=2))
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
        for name in ('control', 'ownloss'):
            for axis, key in zip(axes[:3], ('validation_psnr', 'hard_bytes', 'training_s')):
                values = [row[f'{name}_{key}'] for row in rows]
                if key == 'hard_bytes':
                    values = [value / 1024 for value in values]
                axis.plot(summary['frames'], values, marker='o', label=name)
        axes[3].plot(summary['frames'], [row['delta_validation_psnr'] for row in rows], marker='o')
        axes[3].axhline(0, color='gray', linewidth=.8)
        for axis, label in zip(axes, ('Held-out PSNR (dB)', 'Effective parameters (KiB)',
                                     'Full training time (s)', 'Held-out difference (dB)')):
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
        for axis in axes[:3]:
            axis.legend()
        axes[-1].set_xlabel('Replay frame')
        fig.suptitle('Same frame5 state: sparse true KEEP candidate image fitting')
        fig.tight_layout()
        fig.savefig(args.output / 'keep_candidate_pair.png', dpi=150)
        plt.close(fig)
    print(json.dumps(dict(means=summary['means'], compute_cost=summary['compute_cost'],
                         views=summary['float_view_means']), indent=2))


if __name__ == '__main__':
    main()
