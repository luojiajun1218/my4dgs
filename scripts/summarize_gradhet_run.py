"""Summarize a single run without mixing in older experiments."""
import argparse
import csv
import json
import re
from pathlib import Path


def summarize(run_dir):
    log = (run_dir / 'logging.INFO').read_text()
    metrics, frame, evaluation_step = {}, None, None
    for line in log.splitlines():
        start = re.search(r'Start learning frame (\d+)', line)
        if start:
            frame, evaluation_step = int(start[1]), None
        evaluation = re.search(r'Evaluate test dataset at step (\d+):', line)
        if evaluation:
            evaluation_step = int(evaluation[1])
        match = re.search(r'PSNR: ([\d.]+).*SSIM: ([\d.]+)', line)
        if match and frame is not None and evaluation_step is not None:
            metrics.setdefault(frame, {})[evaluation_step] = dict(psnr=float(match[1]), ssim=float(match[2]))
            evaluation_step = None

    rows = []
    for frame, steps in sorted(metrics.items()):
        final_step = 15000 if frame == 0 else 200
        final = steps.get(final_step, {})
        row = dict(frame=frame, final_step=final_step, final_psnr=final.get('psnr'), final_ssim=final.get('ssim'))
        for step in (50, 99, 100):
            row[f'psnr{step}'] = steps.get(step, {}).get('psnr')
        storage = run_dir / 'frame_storage' / f'frame_{frame:03d}.json'
        if storage.exists():
            row.update(json.loads(storage.read_text()))
        diagnostic = run_dir / 'hierarchy_stats' / f'frame_{frame:03d}_hierarchy_stats.pt'
        if diagnostic.exists():
            import torch
            stats = torch.load(diagnostic, map_location='cpu', weights_only=True)
            selected = stats['mid_active']
            valid = stats['diagnostic_valid']
            row.update(active_middle=int(selected.sum()), nonempty_middle=int((stats['count'] > 0).sum()),
                       observed_middle=int(valid.sum()), shared_gs_ratio=float(stats['overlap_gs_ratio']),
                       activation_xyz_change=stats['activation_delta_xyz_max'],
                       activation_rotation_change=stats['activation_delta_rotation_max'])
            for key in ('H_before', 'H_after', 'H_reduction'):
                row[key + '_selected_mean'] = float(stats[key][selected].mean()) if selected.any() else None
        rows.append(row)
    fields = ['frame', 'final_step', 'final_psnr', 'final_ssim', 'psnr50', 'psnr99', 'psnr100',
              'active_middle', 'nonempty_middle', 'observed_middle', 'active_fine', 'candidate_fine',
              'shared_gs_ratio', 'H_before_selected_mean', 'H_after_selected_mean', 'H_reduction_selected_mean',
              'activation_xyz_change', 'activation_rotation_change', 'incremental_gs',
              'motion_bytes', 'incremental_bytes', 'total_bytes']
    with (run_dir / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / 'results.json').write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps(row))
    later = [row['final_psnr'] for row in rows if row['frame'] > 0 and row['final_psnr'] is not None]
    if later:
        print(f'Mean final PSNR over {len(later)} later frames: {sum(later) / len(later):.4f}')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    summarize(parser.parse_args().run_dir)
