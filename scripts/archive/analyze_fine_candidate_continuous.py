"""Same-frame comparison of the frozen local-candidate treatment and priorformal."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from scripts.archive.analyze_refaligned_core import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    native = analyze(args.run, args.native, args.output / 'native')
    current = list(csv.DictReader((args.output / 'native/frames.csv').open()))
    previous = {int(row['frame']): row for row in csv.DictReader(
        (args.old / 'analysis/refaligned/frames.csv').open())}
    rows = []
    keys = ('test_psnr', 'test_ssim', 'test_l1', 'hard_bytes', 'expected_bytes',
            'actual_changed_fraction', 'local_fraction', 'born', 'deleted',
            'training_s', 'training_loop_s', 'wall_s', 'extra_candidate_renders')
    camera_matches = 0
    for row in current:
        frame = int(row['frame'])
        old = previous[frame]
        traces = [json.loads((root / 'frame_summary' /
                             f'training_views_{frame:03d}.json').read_text())['views']
                  for root in (args.run, args.old)]
        assert traces[0] == traces[1], 'Actual treatment/priorformal camera orders differ'
        camera_matches += len(traces[0])
        item = dict(frame=frame)
        for key in keys:
            item[f'new_{key}'] = float(row[key])
            item[f'old_{key}'] = float(old[key])
            item[f'delta_{key}'] = float(row[key]) - float(old[key])
        for prefix, root in (('new', args.run), ('old', args.old)):
            probe = json.loads((root / 'frame_summary' / f'control_{frame:03d}.json').read_text())['quality']['after_retention']['train_probe']['mean']
            item[f'{prefix}_train_probe_image'] = probe['image_objective']
            item[f'{prefix}_train_probe_psnr'] = probe['psnr']
        item['delta_train_probe_image'] = item['new_train_probe_image'] - item['old_train_probe_image']
        item['delta_train_probe_psnr'] = item['new_train_probe_psnr'] - item['old_train_probe_psnr']
        rows.append(item)
    fields = [key for key in rows[0] if key != 'frame']
    def summary(subset):
        return {key: statistics.fmean(row[key] for row in subset) for key in fields}
    blocks = []
    for start, end in ((1, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299)):
        subset = [row for row in rows if start <= row['frame'] <= end]
        if subset:
            blocks.append(dict(start=start, end=end, completed=len(subset), means=summary(subset)))
    result = dict(completed=len(rows), complete299=len(rows) == 299,
        means=summary(rows), blocks=blocks, actual_camera_samples_identical=camera_matches,
        native_reference_means=native['means'], test_used_for_selection=False,
        limitations=['One seed and potentially nondeterministic CUDA optimization.',
            'Training probes use five fixed training cameras; not all20 final evaluation.',
            'Pure training and inclusive loop/wall are kept separate.',
            'Effective parameter bytes are estimates, not encoded file sizes.'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'fine_continuous_summary.json').write_text(json.dumps(result, indent=2))
    with (args.output / 'fine_continuous_frames.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(completed=len(rows), means=result['means'])))


if __name__ == '__main__':
    main()
