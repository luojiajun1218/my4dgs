"""Audit complete controlled runs and compare identical dynamic frame sets."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(run, required_frames=None, write_results=True):
    records = read_jsonl(run / 'evaluation.jsonl')
    frames = {}
    for item in records:
        if item['split'] == 'test' and item['frame'] > 0:
            previous = frames.get(item['frame'])
            if previous is None or previous['step'] <= item['step']:
                frames[item['frame']] = item
    if required_frames is not None and set(frames) != set(range(1, required_frames)):
        raise ValueError(f'Incomplete frame set: got {sorted(frames)}, expected 1..{required_frames - 1}')
    times = {r['frame']: r['wall_seconds'] for r in read_jsonl(run / 'frame_times.jsonl')}
    rows = []
    for frame, item in sorted(frames.items()):
        row = {k: item[k] for k in ('frame', 'step', 'psnr', 'ssim', 'l1')}
        row['wall_seconds'] = times.get(frame)
        storage = run / 'frame_storage' / f'frame_{frame:03d}.json'
        if storage.exists():
            row.update(json.loads(storage.read_text()))
        rows.append(row)
    keys = ('psnr', 'ssim', 'l1', 'wall_seconds', 'motion_bytes', 'incremental_bytes',
            'total_bytes', 'estimated_bits', 'extra_backward_steps', 'splits')
    means = {key: statistics.mean(row[key] for row in rows if row.get(key) is not None)
             for key in keys if any(row.get(key) is not None for row in rows)}
    result = dict(run=str(run.resolve()), dynamic_frames=len(rows), means=means, frames=rows)
    if write_results:
        (run / 'controlled_summary.json').write_text(json.dumps(result, indent=2))
        fields = sorted(set().union(*(row.keys() for row in rows)))
        with (run / 'controlled_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    result = summarize(args.run, args.frames)
    print(json.dumps({k: v for k, v in result.items() if k != 'frames'}, indent=2))
    if args.reference:
        reference = summarize(args.reference, write_results=False)
        current = {row['frame']: row for row in result['frames']}
        baseline = {row['frame']: row for row in reference['frames'] if row['frame'] in current}
        if set(baseline) != set(current):
            raise ValueError('Reference is missing comparison frames')
        a = [r for r in read_jsonl(args.run / 'training_views.jsonl') if r['frame'] in current]
        b = [r for r in read_jsonl(args.reference / 'training_views.jsonl') if r['frame'] in current]
        view_alignment = a == b
        delta = {metric: statistics.mean(current[f][metric] - baseline[f][metric] for f in current)
                 for metric in ('psnr', 'ssim', 'l1')}
        comparison = dict(reference=str(args.reference.resolve()), run=str(args.run.resolve()),
                          frames=sorted(current), regular_training_views_identical=view_alignment,
                          mean_difference=delta,
                          extra_backward_steps=result['means'].get('extra_backward_steps', 0),
                          caveat='Exploratory single-scene comparison; extra trials are NOT free computation.')
        (args.run / 'controlled_comparison.json').write_text(json.dumps(comparison, indent=2))
        print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
