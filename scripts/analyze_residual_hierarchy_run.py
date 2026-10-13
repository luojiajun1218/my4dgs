"""Summarize one method run, its tree snapshots, and its complete frame range."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def mean(values):
    return statistics.fmean(values) if values else None


def summarize_frames(reports):
    quality = [row['evaluation']['test'] for row in reports]
    return dict(
        frames=len(reports), first_frame=reports[0]['frame'], last_frame=reports[-1]['frame'],
        psnr_mean=mean([row['psnr'] for row in quality]),
        psnr_min=min(row['psnr'] for row in quality),
        ssim_mean=mean([row['ssim'] for row in quality]),
        lpips_mean=mean([row['lpips'] for row in quality if 'lpips' in row]),
        dynamic_bytes_mean=mean([row['encoded_bytes'] for row in reports]),
        dynamic_bytes_total=sum(row['encoded_bytes'] for row in reports),
        elapsed_seconds_mean=mean([row['elapsed_seconds'] for row in reports]),
        elapsed_seconds_total=sum(row['elapsed_seconds'] for row in reports),
        local_residual_fraction_mean=mean([
            row['motion_activity']['old_gs_with_local_residual'] / row['old_count'] for row in reports]),
        changed_pose_fraction_mean=mean([
            row['motion_activity']['old_gs_with_changed_pose'] / row['old_count'] for row in reports]),
        births_total=sum(row['birth_count'] for row in reports),
        final_gs=reports[-1]['total_count'], final_regions=reports[-1]['region_count'])


def tree_snapshot(path):
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    parents = {int(row['node_id']): int(row['parent_id']) for row in rows}
    if parents.get(0) != -1:
        raise ValueError(f'Invalid identity root in {path}')
    depths = {}
    for node in parents:
        seen = set()
        current = node
        while current != 0:
            if current in seen or current not in parents:
                raise ValueError(f'Cyclic or disconnected tree in {path}')
            seen.add(current)
            current = parents[current]
        depths[node] = len(seen)
    if any(int(row['depth']) != depths[int(row['node_id'])] for row in rows):
        raise ValueError(f'CSV depth differs from actual topology in {path}')
    root = next(row for row in rows if int(row['node_id']) == 0)
    old_count = sum(int(row['old_gs_count']) for row in rows)
    original_count = sum(int(row['original_frame0_gs_count']) for row in rows)
    if (old_count < original_count or original_count <= 0 or
            any(int(row['old_gs_with_local_residual']) > int(row['old_gs_count']) for row in rows) or
            abs(sum(float(row['old_gs_percent']) for row in rows) - 100) > 1e-6 or
            abs(sum(float(row['original_frame0_gs_percent']) for row in rows) - 100) > 1e-6):
        raise ValueError(f'Invalid ownership or residual counts in {path}')
    nonroot = [row for row in rows if int(row['node_id']) != 0 and int(row['old_gs_count'])]
    return dict(
        frame=int(root['frame']), step=int(root['step']), nodes=len(rows),
        max_depth=max(depths.values()),
        old_gs=old_count, original_gs=original_count,
        root_old_gs=int(root['old_gs_count']), root_old_gs_percent=float(root['old_gs_percent']),
        root_original_gs_percent=float(root['original_frame0_gs_percent']),
        local_residual_gs=sum(int(row['old_gs_with_local_residual']) for row in rows),
        root_local_residual_gs=int(root['old_gs_with_local_residual']),
        root_zero_residual_gs=int(root['old_gs_count']) - int(root['old_gs_with_local_residual']),
        near_identity_candidates=sum(int(row.get('near_identity_candidates_at_clustering', 0)) for row in rows),
        noise_gs=sum(int(row.get('noise_at_clustering', 0)) for row in rows),
        unobserved_gs=sum(int(row.get('unobserved_at_clustering', 0)) for row in rows),
        populated_motion_regions=len(nonroot),
        motion_region_members_median=statistics.median([int(row['old_gs_count']) for row in nonroot]) if nonroot else 0,
        zero_local_motion_regions=sum(int(row['local_motion_nonzero']) == 0 for row in nonroot),
        nonzero_inherited_motion_regions=sum(int(row['inherited_motion_nonzero']) for row in nonroot),
        path=str(path))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--num-frames', type=int, required=True,
                        help='Total frames, including the shared frame0')
    parser.add_argument('--output', type=Path, required=True,
                        help='New analysis directory; existing directories are rejected')
    parser.add_argument('--allow-prefix', action='store_true',
                        help='Explicitly analyze a finished prefix of a longer run')
    args = parser.parse_args()
    if args.num_frames < 2:
        parser.error('The run needs at least one dynamic frame')
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.run / 'config.yaml')
    configured_frames = int(config.data.extra_dataset_kwargs.num_frames)
    rate_normalization = config.module.rigid_tree.get('rate_normalization', 'frame')
    if rate_normalization not in ('frame', 'reference'):
        raise ValueError('Unknown storage-loss normalization in run config')
    if args.num_frames > configured_frames or (args.num_frames != configured_frames and not args.allow_prefix):
        raise ValueError('Requested range differs from configuration; interrupted prefixes need --allow-prefix')
    report_paths = sorted((args.run / 'frame_summary').glob('frame_*.json'))
    if args.allow_prefix:
        report_paths = [path for path in report_paths
                        if int(path.stem.rsplit('_', 1)[-1]) < args.num_frames]
    reports = [json.loads(path.read_text()) for path in report_paths]
    if [row['frame'] for row in reports] != list(range(1, args.num_frames)):
        raise ValueError('Run is incomplete or has duplicate/nonconsecutive dynamic frames')
    reference_gs_count = reports[0]['initial_reference']['retained_count']
    rate_counts = []
    rate_prices = []
    stream = args.run / 'frame_storage' / 'rigid_tree'
    reference = stream / 'reference_frame0.npz'
    if not reference.is_file():
        raise FileNotFoundError(reference)
    for report in reports:
        frame = report['frame']
        expected_steps = 300 if frame == 1 or frame % 10 == 0 else 200
        if report['train_loss_steps'] != expected_steps:
            raise ValueError(f'Unexpected schedule in frame{frame}: {report["train_loss_steps"]}')
        quality = report['evaluation']['test']
        if any(not math.isfinite(value) for value in quality.values()):
            raise ValueError(f'Nonfinite test metrics in frame{frame}')
        motion = stream / f'motion_{frame:03d}.bin'
        birth = stream / f'birth_{frame:03d}.bin'
        stored = motion.stat().st_size + (birth.stat().st_size if birth.exists() else 0)
        if stored != report['encoded_bytes']:
            raise ValueError(f'Actual stream bytes differ from frame{frame} report')
        count = reference_gs_count if rate_normalization == 'reference' else report['old_count']
        price = float(config.module.rigid_tree.rate_loss_weight) / (24 * count)
        rate = report['motion_rate_diagnostics']
        if ('rate_normalization_gs_count' in rate and
                rate['rate_normalization_gs_count'] != count):
            raise ValueError(f'Storage-loss reference changed in frame{frame}')
        if ('rate_byte_price' in rate and
                not math.isclose(rate['rate_byte_price'], price, rel_tol=1e-6, abs_tol=1e-14)):
            raise ValueError(f'Unexpected byte price in frame{frame}')
        rate_counts.append(count)
        rate_prices.append(price)
    snapshot_paths = sorted((args.run / 'tree_stats').glob('*.csv'))
    if args.allow_prefix:
        snapshot_paths = [path for path in snapshot_paths
                          if int(path.stem.split('_')[1]) < args.num_frames]
    snapshots = [tree_snapshot(path) for path in snapshot_paths]
    maximum_depth = int(config.module.rigid_tree.parent_max_depth)
    if any(row['max_depth'] > maximum_depth for row in snapshots):
        raise ValueError('An exported tree exceeds the configured depth bound')
    expected_snapshots = [(frame, step) for frame in range(1, args.num_frames)
                          if frame == 1 or frame % 10 == 0 for step in (100, 200)]
    if [(row['frame'], row['step']) for row in snapshots] != expected_snapshots:
        raise ValueError('Missing or unexpected tree-edit snapshots')
    summary = summarize_frames(reports)
    segments = [summarize_frames(selected) for start in range(0, args.num_frames, 50)
                if (selected := [row for row in reports if start <= row['frame'] < start + 50])]
    summary.update(run=str(args.run.resolve()), num_frames_including_shared_reference=args.num_frames,
        configured_num_frames=configured_frames, full_sequence=args.num_frames == configured_frames,
        rate_normalization=rate_normalization, shared_reference_gs_count=reference_gs_count,
        rate_normalization_gs_count_initial=rate_counts[0],
        rate_normalization_gs_count_final=rate_counts[-1],
        final_to_initial_rate_byte_price_ratio=(rate_prices[-1] / rate_prices[0]
                                              if rate_prices[0] else None),
        shared_reference_bytes=reference.stat().st_size,
        total_reference_and_dynamic_bytes=reference.stat().st_size + summary['dynamic_bytes_total'],
        tree_update_frames=len(expected_snapshots) // 2,
        max_tree_depth=max(row['max_depth'] for row in snapshots),
        finalized_root_original_percent_min=min(row['root_original_gs_percent'] for row in snapshots if row['step'] == 200),
        finalized_root_original_percent_max=max(row['root_original_gs_percent'] for row in snapshots if row['step'] == 200),
        segments=segments, tree_snapshots=snapshots,
        scope='Single method run. Test metrics cover dynamic frames 1..num_frames-1; '
              'frame0 is the shared trained reference. Root ownership is not the scene static ratio. '
              'Root + zero quantized GS residual is exactly unchanged under the current model.')
    args.output.mkdir(parents=True, exist_ok=False)
    frame_rows = [dict(frame=row['frame'], steps=row['train_loss_steps'],
        psnr=row['evaluation']['test']['psnr'], ssim=row['evaluation']['test']['ssim'],
        lpips=row['evaluation']['test'].get('lpips'), dynamic_bytes=row['encoded_bytes'],
        births=row['birth_count'], gs=row['total_count'], regions=row['region_count'],
        elapsed_seconds=row['elapsed_seconds'],
        local_residual_gs=row['motion_activity']['old_gs_with_local_residual'],
        changed_pose_gs=row['motion_activity']['old_gs_with_changed_pose'],
        local_residual_fraction=row['motion_activity']['old_gs_with_local_residual'] / row['old_count'],
        changed_pose_fraction=row['motion_activity']['old_gs_with_changed_pose'] / row['old_count'],
        selected_components=row['motion_tree_diagnostics'].get('tree_edit/gmm_selected_components'),
        gmm_converged=row['motion_tree_diagnostics'].get('tree_edit/gmm_converged'),
        rejected_motion_groups=row['motion_tree_diagnostics'].get('tree_edit/rejected_motion_groups'),
        deferred_sparse_groups=row['motion_tree_diagnostics'].get('tree_edit/deferred_sparse_motion_groups'),
        rate_normalization_gs_count=rate_counts[index], rate_byte_price=rate_prices[index])
        for index, row in enumerate(reports)]
    write_csv(args.output / 'frames.csv', frame_rows)
    write_csv(args.output / 'tree_snapshots.csv', snapshots)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout='constrained')
    frames = [row['frame'] for row in frame_rows]
    axes[0].plot(frames, [row['psnr'] for row in frame_rows], linewidth=1.)
    axes[0].set_ylabel('PSNR (dB)')
    axes[1].plot(frames, [row['dynamic_bytes'] / 1000 for row in frame_rows], linewidth=1.)
    axes[1].set_ylabel('Dynamic packet (KB)')
    axes[2].step(frames, [row['regions'] for row in frame_rows], where='post', linewidth=1.)
    axes[2].set_ylabel('Tree nodes')
    axes[2].set_xlabel('Dynamic frame')
    for axis in axes:
        axis.grid(alpha=.2)
    figure.savefig(args.output / 'progression.png', dpi=150)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 4), layout='constrained')
    axis.plot(frames, [100 * row['local_residual_fraction'] for row in frame_rows],
              label='Old GS with local residual')
    axis.plot(frames, [100 * row['changed_pose_fraction'] for row in frame_rows],
              label='Old GS with changed model pose')
    finalized = [row for row in snapshots if row['step'] == 200]
    axis.plot([row['frame'] for row in finalized],
              [row['root_original_gs_percent'] for row in finalized], marker='.',
              label='Frame0 GS directly owned by identity root')
    axis.set(xlabel='Dynamic frame', ylabel='GS percentage', ylim=(0, 100))
    axis.legend()
    axis.grid(alpha=.2)
    figure.savefig(args.output / 'motion_activity.png', dpi=150)
    plt.close(figure)
    compact = {key: value for key, value in summary.items() if key not in ('tree_snapshots', 'segments')}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
