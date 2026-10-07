"""Summarize a completed storage-aware binary motion run.

Usage: python scripts/analyze_storage_aware_binary.py RUN --output ANALYSIS_DIR
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


def resolve_path(run, value):
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, run / path)
    return next((candidate for candidate in candidates if candidate.exists()), run / path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite analysis output: {output}')
    config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    if config['module']['motion_mode'] != 'storage_aware_binary':
        raise ValueError('Expected a storage_aware_binary run')

    frames = []
    histograms = {str(bits): 0 for bits in config['module']['storage_aware_binary']['precision_bits']}
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        packet = json.loads(path.read_text(encoding='utf-8'))
        frame = packet['frame']
        if frame == 0:
            continue
        stream = packet['stream']
        evaluation = packet['quantized_evaluation']['test']
        storage_parts = stream['storage_bytes_by_class']
        if abs(sum(storage_parts.values()) - stream['hard_storage_bytes']) > 1e-3:
            raise ValueError(f'Hard storage class totals disagree: {path}')
        motion = stream['motion_export']
        motion_path = resolve_path(run, motion['path'])
        if motion_path.stat().st_size != motion['archive_bytes']:
            raise ValueError(f'Motion file byte count mismatch: {motion_path}')
        birth = stream.get('incremental_export')
        birth_bytes = 0
        if birth:
            birth_path = resolve_path(run, birth['path'])
            birth_bytes = birth_path.stat().st_size
            if birth_bytes != birth['archive_bytes']:
                raise ValueError(f'Birth PLY byte count mismatch: {birth_path}')
        inference_bytes = motion_path.stat().st_size + birth_bytes
        if inference_bytes != stream['inference_file_bytes']:
            raise ValueError(f'Necessary file total disagrees: {path}')
        if abs(inference_bytes - stream['hard_storage_bytes']) > 1e-3:
            raise ValueError(f'Hard storage estimate disagrees with deployment files: {path}')
        actual_sizes = packet.get('evaluation_image_sizes', [])
        expected_size = config['data'].get('expected_image_size')
        if expected_size and any(size != expected_size for size in actual_sizes):
            raise ValueError(f'Unexpected evaluated image dimensions: {path}')

        activities = stream['activities']
        precision_counts = {bits: sum(level['precision_bit_histogram'].get(bits, 0)
                                     for level in activities['levels'])
                            for bits in histograms}
        for bits, count in precision_counts.items():
            histograms[bits] += count
        frames.append(dict(frame=frame, psnr=evaluation['psnr'], ssim=evaluation['ssim'],
            fps=evaluation['fps'], inference_bytes=inference_bytes,
            motion_bytes=motion_path.stat().st_size, birth_bytes=birth_bytes,
            hard_storage_bytes=stream['hard_storage_bytes'],
            expected_storage_bytes=stream['expected_storage_bytes'],
            training_seconds=packet['timing']['training'], wall_seconds=packet['timing']['wall'],
            training_with_prepare_finalize_seconds=packet['timing']['training_loop_including_state_update'],
            new_count=stream['new_count'], deleted_count=stream['deleted_count'],
            tree_rebuilt=activities['tree_rebuilt'], precision_counts=precision_counts,
            terminal_depth_counts=activities['gs']['terminal_depth_counts'],
            coded_payload_bits=motion['payload_bits'], image_sizes=actual_sizes))

    expected_frames = config['data']['extra_dataset_kwargs']['num_frames'] - 1
    if [row['frame'] for row in frames] != list(range(1, expected_frames + 1)):
        raise ValueError(f'Run incomplete: expected dynamic frames 1..{expected_frames}')
    mean_keys = ('psnr', 'ssim', 'fps', 'inference_bytes', 'hard_storage_bytes',
                 'expected_storage_bytes', 'training_seconds', 'wall_seconds',
                 'training_with_prepare_finalize_seconds',
                 'new_count', 'deleted_count', 'coded_payload_bits')
    means = {key: statistics.mean(row[key] for row in frames) for key in mean_keys}
    output.mkdir(parents=True)
    summary = dict(run=str(run), frames=len(frames), means=means,
        precision_histogram=histograms, rebuilt_frames=[row['frame'] for row in frames
                                                        if row['tree_rebuilt']],
        config=config, rows=frames,
        interpretation='This run records its final quality and exact coded file size. It is not '
        'a controlled comparison against a prior configuration.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    scalar_keys = [key for key, value in frames[0].items()
                   if not isinstance(value, (dict, list))]
    with (output / 'frames.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in scalar_keys} for row in frames)

    frame_ids = [row['frame'] for row in frames]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(frame_ids, [row['psnr'] for row in frames], label='PSNR')
    axes[0, 0].plot(frame_ids, [row['ssim'] for row in frames], label='SSIM')
    axes[0, 0].set(xlabel='Dynamic frame', ylabel='Image quality')
    axes[0, 0].legend()
    axes[0, 1].plot(frame_ids, [row['inference_bytes'] / 1e6 for row in frames])
    axes[0, 1].set(xlabel='Dynamic frame', ylabel='Necessary files (MB/frame)')
    axes[1, 0].plot(frame_ids, [row['coded_payload_bits'] / 8 for row in frames],
                    label='Motion payload')
    axes[1, 0].plot(frame_ids, [row['birth_bytes'] for row in frames], label='Birth PLY')
    axes[1, 0].set(xlabel='Dynamic frame', ylabel='Bytes/frame')
    axes[1, 0].legend()
    axes[1, 1].plot(frame_ids, [row['training_seconds'] for row in frames])
    axes[1, 1].set(xlabel='Dynamic frame', ylabel='Training seconds/frame')
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.savefig(output / 'final_effect.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(frames=len(frames), means=means,
                          precision_histogram=histograms,
                          rebuilt_frames=summary['rebuilt_frames']), indent=2))


if __name__ == '__main__':
    main()
