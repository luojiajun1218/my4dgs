"""Summarize the completed half-resolution experiment, after stream verification."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def mean(rows, field):
    return statistics.mean(float(row[field]) for row in rows)


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    control = args.benchmark.parent.resolve()
    sys.path.insert(0, str(args.source.resolve()))
    from scripts import benchmark_rigid_residual_tree as benchmark
    import torch
    import numpy as np
    from libgs.metric import lpips, psnr

    measured = json.loads(args.benchmark.read_text())
    count = measured['expected_frames']
    assert measured['measured_frames'] == count and 2 <= count <= 300
    assert measured['image_shape'] == [3, 1008, 1345]
    assert [item['frame'] for item in measured['frames']] == list(range(count))
    config = benchmark._load_config(run)
    assert config.data.extra_dataset_kwargs['num_frames'] == 300
    device = torch.device('cuda')
    module, datamodule = benchmark._make_module(config, device)
    _, raw = benchmark._load_reference(Path(measured['reference']), None)
    benchmark._install_raw(module, raw, device)
    view, test_view_count = benchmark._cam00(datamodule, 0, device)
    assert test_view_count == 1
    with torch.no_grad():
        image = benchmark._render_checked(module, view)
        initial_lpips = float(lpips(image, view.image, net_type='vgg').mean())
        initial_psnr = float(psnr(image, view.image))
    assert abs(initial_psnr - measured['frames'][0]['quality']['psnr']) < 1e-5

    rows = []
    max_quality_differences = dict(psnr=0., ssim=0., l1=0.)
    reference_bytes = measured['storage']['bootstrap_reference_bytes']
    for item in measured['frames']:
        frame = item['frame']
        report = (json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())
                  if frame else None)
        flat = report.get('flat_region_diagnostics', {}) if report else {}
        rate = report.get('motion_rate_diagnostics', {}) if report else {}
        activity = report.get('motion_activity', {}) if report else {}
        for key, difference in (item.get('quality_saved_difference') or {}).items():
            if key in max_quality_differences:
                max_quality_differences[key] = max(max_quality_differences[key], abs(difference))
        row = dict(frame=frame, **item['quality'],
            lpips=(report['evaluation']['test']['lpips'] if report else initial_lpips),
            complete_update_bytes=item['complete_update_bytes'],
            payload_bytes=item['payload_bytes'], motion_bytes=item['motion_bytes'],
            opacity_bytes=item.get('opacity_update_bytes', 0),
            feature_bytes=item.get('feature_dc_update_bytes', 0),
            birth_bytes=item['birth_bytes'], manifest_bytes=item['manifest_index_bytes'],
            shared_reference_bytes=reference_bytes,
            total_gaussians=item['gaussian_count'],
            new_gaussians=report['birth_count'] if report else 0,
            retired_gaussians=report['retired_count'] if report else 0,
            regions=report['region_count'] if report else '',
            open_gates=flat.get('open_gates', ''),
            stored_region_nodes=flat.get('stored_region_nodes', ''),
            old_gaussians=activity.get('old_gs', ''),
            gs_translation_blocks=rate.get('gs_translation_active_blocks', ''),
            gs_rotation_blocks=rate.get('gs_rotation_active_blocks', ''),
            residual_points=activity.get('old_gs_with_local_residual', ''),
            residual_fraction=(activity['old_gs_with_local_residual'] / activity['old_gs']
                               if activity.get('old_gs') else ''),
            pure_render_fps=item['render_fps'],
            decode_verify_seconds=item['file_decode_verify_seconds'],
            upload_seconds=item['upload_seconds'],
            train_wall_seconds=report['elapsed_seconds'] if report else '',
            train_steps=report['train_loss_steps'] if report else '')
        assert all(np.isfinite(float(row[name])) for name in ('psnr', 'ssim', 'l1', 'lpips', 'pure_render_fps'))
        if report:
            assert report['train_loss_steps'] == 300
            assert report['encoded_bytes'] == row['payload_bytes']
            assert row['complete_update_bytes'] == row['payload_bytes'] + row['manifest_bytes']
        rows.append(row)
    assert len(rows) == count
    write_csv(control / 'per_frame_metrics.csv', rows)
    segments = []
    for start in range(0, count, 30):
        part = rows[start:start + 30]
        dynamic = [row for row in part if row['frame']]
        timed = measured['frames'][start:start + 30]
        segments.append(dict(first_frame=start, last_frame=start + 29,
            psnr=mean(part, 'psnr'), ssim=mean(part, 'ssim'), lpips=mean(part, 'lpips'),
            dynamic_complete_kb=mean(dynamic, 'complete_update_bytes') / 1000,
            dynamic_payload_kb=mean(dynamic, 'payload_bytes') / 1000,
            pure_render_fps=sum(item['timed_renders'] for item in timed) /
                sum(item['render_seconds'] for item in timed),
            train_wall_seconds=mean(dynamic, 'train_wall_seconds'),
            mean_gaussians=mean(part, 'total_gaussians'),
            mean_open_gates=mean(dynamic, 'open_gates'),
            mean_regions=mean(dynamic, 'regions')))
    write_csv(control / 'per_30_frames.csv', segments)
    dynamic = rows[1:]
    old_total = sum(int(row['old_gaussians']) for row in dynamic)
    residual_total = sum(int(row['residual_points']) for row in dynamic)
    summary = dict(status='complete', run=str(run), frames=count, dynamic_frames=count-1, full_sequence_evaluation=(count == 300),
        image_size=[1345,1008], psnr=mean(rows,'psnr'), ssim=mean(rows,'ssim'),
        lpips=mean(rows,'lpips'), l1=mean(rows,'l1'),
        dynamic_complete_kb=mean(dynamic,'complete_update_bytes')/1000,
        dynamic_payload_kb=mean(dynamic,'payload_bytes')/1000,
        aggregate_materialized_fps=measured['groups']['all_frames']['aggregate_fps'],
        median_frame_render_fps=measured['groups']['all_frames']['frame_fps_median'],
        mean_train_wall_seconds=mean(dynamic,'train_wall_seconds'),
        mean_train_wall_seconds_after_first_dynamic=mean(dynamic[1:],'train_wall_seconds'),
        first_dynamic_seconds=dynamic[0]['train_wall_seconds'],
        native_frame0_training_steps=config.trainer.num_init_steps,
        mean_decode_verify_seconds=mean(dynamic,'decode_verify_seconds'),
        mean_upload_seconds=mean(dynamic,'upload_seconds'),
        shared_reference_bytes=reference_bytes,
        total_stream_bytes=measured['storage']['total_reference_and_complete_update_bytes'],
        initial_gaussians=rows[0]['total_gaussians'], final_gaussians=rows[-1]['total_gaussians'],
        mean_dynamic_gaussians=mean(dynamic,'total_gaussians'),
        mean_new_gaussians=mean(dynamic,'new_gaussians'),
        mean_retired_gaussians=mean(dynamic,'retired_gaussians'),
        mean_open_gates=mean(dynamic,'open_gates'), mean_regions=mean(dynamic,'regions'),
        frames_with_open_gates=sum(int(row['open_gates'])>0 for row in dynamic),
        frames_with_effective_subdivisions=sum(int(row['regions'])>8 for row in dynamic),
        old_gs_without_local_residual_fraction=1-residual_total/old_total,
        max_saved_quality_differences=max_quality_differences,
        max_cache_pixel_difference=max(item['cache_pixel_max_abs_difference'] for item in measured['frames']),
        gpu_exclusivity_verified=all(item['status']['verified'] for item in measured['gpu_checks']),
        storage_components_mean_kb={key:mean(dynamic,key)/1000 for key in
            ('motion_bytes','opacity_bytes','feature_bytes','birth_bytes','manifest_bytes')},
        goals=dict(psnr_at_least_34_2=mean(rows,'psnr')>=34.2,
            dynamic_complete_at_most_50kb=mean(dynamic,'complete_update_bytes')<=50000,
            pure_render_fps_above_200=measured['groups']['all_frames']['aggregate_fps']>200),
        first_frame_reference_quality=rows[0],
        first_frame_lpips_scope='Recomputed on the decoded pruned reference, not copied from the unpruned native model.',
        dynamic_lpips_scope='Training final decoded-state evaluations. Independent sequence replay rechecks PSNR/SSIM/L1 and all packet digests.',
        comparison_note='Baseline is the unrefined fresh 1345x1008 full run. Candidate changes byte pricing and may refine the fresh first frame; this is a goal improvement experiment, not a single-factor attribution.',
        training_timing_note='All measured dynamic frames remain in the wall-time mean; the mean excluding the first dynamic frame is also reported. First-use CUDA compilation is included when its cache was cold.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    fields=(('psnr','PSNR (dB)',1.),('complete_update_bytes','Dynamic complete update (KB)',.001),
            ('pure_render_fps','Full render FPS',1.),('total_gaussians','Gaussians',1.))
    old_path=Path('/home/gamer-02/Documents/DeepLearning/my4dgs/docs/experiment/evidence/flame_steak_half_fresh_full300_20261018/per_frame_metrics.csv')
    with old_path.open() as file: old=list(csv.DictReader(file))[:count]
    for axis,(field,label,scale) in zip(axes.flat,fields):
        subset=rows[1:] if field=='complete_update_bytes' else rows
        prior=old[1:] if field=='complete_update_bytes' else old
        axis.plot([int(row['frame']) for row in prior],[float(row[field])*scale for row in prior],color='.65',label='Fresh half baseline')
        axis.plot([row['frame'] for row in subset],[float(row[field])*scale for row in subset],color='#187a80',label='Goal candidate (half)',linewidth=1.2)
        axis.set(xlabel='Frame',ylabel=label); axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    figure.suptitle('Flame steak goal: half-resolution quality, bytes, and speed')
    figure.savefig(control/'metrics_comparison.png',dpi=160)
    plt.close(figure)
    manifest=json.loads((args.source/'source_manifest.json').read_text())
    assert all(hashlib.sha256((args.source/name).read_bytes()).hexdigest()==digest
               for name,digest in manifest['hashes'].items())
    summary['frozen_source_unchanged'] = True
    (control/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({key:summary[key] for key in ('psnr','ssim','lpips','dynamic_complete_kb','aggregate_materialized_fps','mean_train_wall_seconds')},indent=2),flush=True)


if __name__=='__main__':
    main()
