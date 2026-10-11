"""Compare spatial and shared-SE3 trees using completed frame reports."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml


def mean(values):
    return statistics.fmean(values) if values else None


def read_run(run):
    config = yaml.safe_load((run / 'config.yaml').read_text())
    orientation_extent = config['module']['sparse_hierarchical']['tree_cluster_orientation_extent']
    records = {}
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        packet = json.loads(path.read_text())
        if packet['frame'] == 0 or not packet.get('stream'):
            continue
        frame, stream = packet['frame'], packet['stream']
        evaluation = packet['quantized_evaluation']['test']
        gs = stream['activities']['gs']
        record = dict(frame=frame, psnr=evaluation['psnr'], ssim=evaluation['ssim'],
                      l1=evaluation['l1'], online_fps=evaluation['fps'],
                      online_forward_seconds=evaluation['mean_forward_seconds'],
                      file_mb=stream['inference_file_bytes'] / 1e6,
                      parameter_mb=stream['hard_storage_bytes'] / 1e6,
                      m_fraction=gs['m_open'] / gs['denominator'],
                      old_gs=gs['denominator'],
                      final_gs=stream['birth_control']['point_origins']['final_total'],
                      new_gs=stream['new_count'], deleted_gs=stream['deleted_count'],
                      train_seconds=packet['timing']['training'],
                      wall_seconds=packet['timing']['wall'],
                      tree_action=stream['tree_setup']['action'],
                      cluster_seconds=stream['stage_seconds'].get('tree_cluster', 0.),
                      projection_seconds=stream['stage_seconds'].get('probe_projection', 0.))
        probe = stream['tree_setup']['probe']
        record.update(qualified_fraction=probe['qualified_rows'] / gs['denominator'],
                      projection_xyz_max_extent=probe['projection_xyz_max_extent'],
                      projection_rotation_acos_max_rad=probe['projection_rotation_max_rad'])
        for level in range(3):
            prefix = f'l{level + 1}'
            activity = stream['activities']['levels'][level]
            record[prefix + '_z_fraction'] = (activity['z_on_relevant_regions'] /
                activity['denominator']) if activity['denominator'] else None
            record[prefix + '_owner_change_fraction'] = probe['changed_owners'][level] / gs['denominator']
            before = probe['before_shared_fit'][level]
            after = probe['after_shared_fit'][level]
            for key in ('objective', 'position_rmse_extent', 'rotation_chord_rmse'):
                record[prefix + '_fit_before_' + key] = before[key]
                record[prefix + '_fit_after_' + key] = after[key]
            if stream['tree_setup'].get('cluster_iterations'):
                metrics = stream['tree_setup']['cluster_iterations']['levels'][level]
                record[prefix + '_cluster_before_objective'] = metrics['before']['objective']
                record[prefix + '_cluster_after_objective'] = metrics['after']['objective']
                # Shared fit objective excludes the spatial penalty. Same starting root seed;
                # deeper levels use a seed inside the newly selected parent.
                pure_before = (metrics['before']['position_rmse_extent'] ** 2 +
                    orientation_extent ** 2 * metrics['before']['rotation_chord_rmse'] ** 2)
                record[prefix + '_cluster_before_motion_share'] = (
                    pure_before / metrics['before']['objective']
                    if metrics['before']['objective'] else None)
        control_path = run / 'frame_summary' / f'control_{frame:03d}.json'
        if control_path.exists():
            quality = json.loads(control_path.read_text())['quality']
            for stage, item in quality.items():
                if 'train_probe' in item:
                    record['train_probe_' + stage + '_psnr'] = item['train_probe']['mean']['psnr']
        records[frame] = record
    return records


def summarize(rows):
    keys = {key for row in rows for key, value in row.items()
            if isinstance(value, (int, float)) and key != 'frame'}
    averages = {key: mean([row[key] for row in rows if row.get(key) is not None])
                for key in sorted(keys)}
    refresh = [row for row in rows if row['tree_action'] in {'initial', 'rebuild'}]
    fit_changes = {}
    for level in range(1, 4):
        prefix = f'l{level}'
        before = mean([row[prefix + '_fit_before_objective'] for row in refresh])
        after = mean([row[prefix + '_fit_after_objective'] for row in refresh])
        fit_changes[prefix] = dict(before_objective=before, after_objective=after,
                                  after_over_before=after / before if before else None,
                                  owner_change_fraction=mean([
                                      row[prefix + '_owner_change_fraction'] for row in refresh]),
                                  motion_objective_share=mean([
                                      row[prefix + '_cluster_before_motion_share'] for row in refresh
                                      if row.get(prefix + '_cluster_before_motion_share') is not None]))
    return dict(frames=len(rows), means=averages,
                aggregate_online_fps=1 / averages['online_forward_seconds'] if rows else None,
                total_file_mb=sum(row['file_mb'] for row in rows),
                refresh_frames=len(refresh), refresh_shared_fit=fit_changes,
                z_always_open=all(row[f'l{level}_z_fraction'] in (1., None)
                                 for row in rows for level in range(1, 4)))


def compare(spatial, cluster, allow_partial=False, last_frame=299):
    if not 1 <= last_frame <= 299:
        raise ValueError('last_frame must be between 1 and 299')
    source, treatment = read_run(spatial), read_run(cluster)
    source = {frame: row for frame, row in source.items() if frame <= last_frame}
    treatment = {frame: row for frame, row in treatment.items() if frame <= last_frame}
    frames = sorted(source.keys() & treatment.keys())
    if not frames or frames != list(range(1, max(frames) + 1)):
        raise ValueError('Need a contiguous completed paired sequence starting at 1')
    if not allow_partial and frames != list(range(1, last_frame + 1)):
        raise ValueError(f'Both dynamic windows 1-{last_frame} are required; got {len(frames)}')
    paired, views_equal = [], []
    for frame in frames:
        views = [json.loads((run / 'frame_summary' /
                 f'training_views_{frame:03d}.json').read_text())['views']
                 for run in (spatial, cluster)]
        if not all(len(value) == 340 for value in views):
            raise ValueError(f'Frame {frame}: expected 340 training steps per arm')
        views_equal.append(views[0] == views[1])
        paired.append(dict(frame=frame, spatial_psnr=source[frame]['psnr'],
                           cluster_psnr=treatment[frame]['psnr'],
                           delta_psnr=treatment[frame]['psnr'] - source[frame]['psnr'],
                           spatial_file_mb=source[frame]['file_mb'],
                           cluster_file_mb=treatment[frame]['file_mb']))
    summaries = dict(spatial=summarize([source[f] for f in frames]),
                     rigid_cluster=summarize([treatment[f] for f in frames]))
    blocks = []
    for start, end in [(1, 24), (1, 49), (50, 99), (100, 149), (150, 199),
                       (200, 249), (250, 299), (280, 299)]:
        rows = [row for row in paired if start <= row['frame'] <= end]
        if rows:
            blocks.append(dict(start=start, end=end, frames=len(rows),
                spatial_psnr=mean([row['spatial_psnr'] for row in rows]),
                cluster_psnr=mean([row['cluster_psnr'] for row in rows]),
                delta_psnr=mean([row['delta_psnr'] for row in rows])))
    result = dict(spatial=str(spatial), rigid_cluster=str(cluster),
        complete_299=(frames == list(range(1, 300))), paired_frames=len(frames),
        requested_last_frame=last_frame,
        complete_requested_window=(frames == list(range(1, last_frame + 1))),
        training_view_sequences_equal_frames=sum(views_equal),
        training_view_sequences_all_equal=all(views_equal), summaries=summaries,
        paired_delta_psnr=mean([row['delta_psnr'] for row in paired]),
        positive_delta_frames=sum(row['delta_psnr'] > 0 for row in paired),
        blocks=blocks, worst_frames=sorted(paired, key=lambda row: row['delta_psnr'])[:10],
        best_frames=sorted(paired, key=lambda row: row['delta_psnr'], reverse=True)[:10],
        notes=[f'Averages cover dynamic frames1-{max(frames)}; the shared frame0 is excluded.',
               'Online FPS may include GPU contention. Use a separate paired checkpoint benchmark.',
               'Necessary NPZ, birth PLY and frame/deletion manifests are counted; shared first frame excluded.',
               'Probe-fit diagnostics describe independent teacher motion, not ground-truth trajectories.',
               'tree_cluster/probe_projection are inside the first dense step; do not sum all stage timers.',
               'Rotation acos diagnostic is float32 and has poor precision near identity.'])
    return result, paired, source, treatment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spatial', type=Path)
    parser.add_argument('cluster', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--last-frame', type=int, default=299)
    args = parser.parse_args()
    result, rows, source, treatment = compare(args.spatial, args.cluster,
                                            args.allow_partial, args.last_frame)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2))
    with (args.output / 'paired_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name, records in [('spatial', source), ('rigid_cluster', treatment)]:
        keys = sorted({key for row in records.values() for key in row})
        with (args.output / f'{name}_frames.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(records.values())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    frames = [row['frame'] for row in rows]
    for name, records, color in [('Spatial', source, '#4c78a8'),
                                 ('Rigid cluster', treatment, '#e45756')]:
        for axis, key, title in [(axes[0, 0], 'psnr', 'Test PSNR (dB)'),
                                 (axes[1, 0], 'file_mb', 'Necessary files (MB / dynamic frame)'),
                                 (axes[1, 1], 'm_fraction', 'Old GS motion gate open fraction')]:
            axis.plot(frames, [records[f][key] for f in frames], color=color, label=name, linewidth=1)
            axis.set_title(title); axis.set_xlabel('Frame'); axis.grid(alpha=.2)
    axes[0, 0].legend()
    axes[0, 1].plot(frames, [row['delta_psnr'] for row in rows], color='#e45756', linewidth=1)
    axes[0, 1].axhline(0, color='gray', linewidth=.8)
    axes[0, 1].set_title('Cluster minus spatial PSNR (dB)'); axes[0, 1].grid(alpha=.2)
    axes[0, 1].set_xlabel('Frame')
    figure.savefig(args.output / 'comparison.png', dpi=160)
    plt.close(figure)
    print(json.dumps({key: result[key] for key in ['paired_frames', 'paired_delta_psnr',
                     'training_view_sequences_all_equal', 'positive_delta_frames']}))


if __name__ == '__main__':
    main()
