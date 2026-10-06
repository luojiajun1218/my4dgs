"""Summarize a completed binary-refinement run without changing source results."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import yaml


def read_run(run):
    config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    rows = []
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        packet = json.loads(path.read_text(encoding='utf-8'))
        if packet['frame'] == 0:
            continue
        stream = packet['stream']
        quality = packet['quantized_evaluation']['test']
        parameter_total = sum(stream['storage_bytes_by_class'].values())
        if abs(parameter_total - stream['hard_storage_bytes']) > 1e-3:
            raise ValueError(f'Parameter totals disagree: {path}')
        actual_files = 0
        for field in ('motion_export', 'incremental_export', 'frame_manifest'):
            export = stream.get(field)
            if export:
                source = Path(export['path'])
                if not source.is_absolute():
                    source = Path.cwd() / source
                size = source.stat().st_size
                if size != export['archive_bytes']:
                    raise ValueError(f'File bytes disagree: {source}')
                actual_files += size
        if actual_files != stream['inference_file_bytes']:
            raise ValueError(f'Necessary file totals disagree: {path}')
        image_sizes = packet.get('evaluation_image_sizes')
        if image_sizes is None:
            image_dir = run / 'image' / 'frames' / f"{packet['frame']:04d}"
            # Older summaries predate size reporting. Verify saved GT pixels.
            image_sizes = sorted({Image.open(image).size for image in image_dir.glob('*_gt.png')})
            image_sizes = [list(size) for size in image_sizes]
            if not image_sizes:
                raise ValueError(f'No actual image-size evidence: {path}')
        row = dict(frame=packet['frame'], psnr=quality['psnr'], ssim=quality['ssim'],
                   fps=quality['fps'], inference_bytes=actual_files,
                   parameter_bytes=stream['hard_storage_bytes'],
                   training_seconds=packet['timing']['training'],
                   wall_seconds=packet['timing']['wall'],
                   new_count=stream['new_count'], deleted_count=stream['deleted_count'],
                   activities=stream['activities'], image_sizes=image_sizes)
        control_path = path.parent / f"control_{packet['frame']:03d}.json"
        if control_path.exists():
            control = json.loads(control_path.read_text(encoding='utf-8'))
            row['phase_quality'] = {stage: value.get('train_probe', {}).get('mean', {})
                                    for stage, value in control.get('quality', {}).items()}
        rows.append(row)
    expected = config['data']['extra_dataset_kwargs']['num_frames'] - 1
    if [row['frame'] for row in rows] != list(range(1, expected + 1)):
        raise ValueError(f'Run incomplete: expected frames 1..{expected}')
    return config, rows


def means(rows):
    return {key: statistics.mean(row[key] for row in rows) for key in (
        'psnr', 'ssim', 'fps', 'inference_bytes', 'parameter_bytes',
        'training_seconds', 'wall_seconds', 'new_count', 'deleted_count')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, rows = read_run(args.run)
    if config['module']['motion_mode'] != 'binary_refinement':
        raise ValueError('Expected a binary_refinement run')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(run=str(args.run), frames=len(rows), means=means(rows),
                   config=config, rows=rows,
                   semantics='Dynamic frames only; shared bootstrap excluded. Historical '
                             'reference is a contextual comparison with changed representation '
                             'and schedule, not a controlled mechanism ablation.')
    reference_rows = None
    if args.reference:
        reference_config, reference_rows = read_run(args.reference)
        for field in ('ckpt_path', 'random_seed'):
            if config[field] != reference_config[field]:
                raise ValueError(f'Reference mismatch: {field}')
        for field in ('root', 'resolution'):
            if config['data'][field] != reference_config['data'][field]:
                raise ValueError(f'Reference data mismatch: {field}')
        if [r['image_sizes'] for r in rows] != [r['image_sizes'] for r in reference_rows]:
            raise ValueError('Reference actual image sizes mismatch')
        reference_means = means(reference_rows)
        summary['reference'] = dict(run=str(args.reference), means=reference_means)
        summary['delta'] = {key: summary['means'][key] - value
                            for key, value in reference_means.items()}
        summary['inference_bytes_reduction_percent'] = (
            100 * (1 - summary['means']['inference_bytes'] / reference_means['inference_bytes']))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    scalar_keys = [key for key in rows[0] if key not in ('activities', 'image_sizes', 'phase_quality')]
    with (args.output / 'frames.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, label in zip(axes.flat,
            ('psnr', 'inference_bytes', 'training_seconds', 'fps'),
            ('Held-out PSNR (dB)', 'Necessary files (MB/frame)', 'Training (s/frame)', 'Render FPS')):
        divisor = 1e6 if key == 'inference_bytes' else 1
        axis.plot([r['frame'] for r in rows], [r[key] / divisor for r in rows],
                  label='Binary quantized refinement')
        if reference_rows:
            axis.plot([r['frame'] for r in reference_rows],
                      [r[key] / divisor for r in reference_rows], label='Historical sparse hierarchy')
        axis.set(xlabel='Dynamic frame', ylabel=label)
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.output / 'comparison.png', dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('rows', 'config')}, indent=2))


if __name__ == '__main__':
    main()
