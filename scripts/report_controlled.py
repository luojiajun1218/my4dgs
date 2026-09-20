"""Plot a completed, exactly aligned controlled comparison (no rounded log parsing)."""
import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from summarize_controlled import read_jsonl, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--frames', type=int, default=100)
    args = parser.parse_args()
    baseline = summarize(args.reference)
    candidate = summarize(args.candidate, args.frames)
    baseline['frames'] = [r for r in baseline['frames'] if r['frame'] < args.frames]
    assert [r['frame'] for r in baseline['frames']] == list(range(1, args.frames))
    baseline['means'] = {key: statistics.mean(r[key] for r in baseline['frames']
                                            if r.get(key) is not None)
                         for key in baseline['means']}
    a, b = baseline['frames'], candidate['frames']
    assert [r['frame'] for r in a] == [r['frame'] for r in b]
    assert all(r.get('wall_seconds') is not None and r.get('total_bytes') is not None
               for r in a + b), 'Missing timing or actual storage records'
    frames = [r['frame'] for r in a]
    streams = []
    for run in (args.reference, args.candidate):
        streams.append([r for r in read_jsonl(run / 'training_views.jsonl')
                        if r['frame'] in frames])
    aligned = streams[0] == streams[1]
    output = args.candidate / 'controlled_report'
    output.mkdir(exist_ok=True)
    deltas = [y['psnr'] - x['psnr'] for x, y in zip(a, b)]
    metrics = dict(frames=frames, reference=baseline['means'], candidate=candidate['means'],
                   psnr_delta=statistics.mean(deltas),
                   ssim_delta=statistics.mean(y['ssim'] - x['ssim'] for x, y in zip(a, b)),
                   psnr_wins=sum(d > 0 for d in deltas),
                   regular_training_views_identical=aligned,
                   max_decode_xyz_error=max(r.get('decode_xyz_max_error', 0) for r in b),
                   max_decode_rotation_error=max(r.get('decode_rotation_max_error', 0) for r in b))
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for label, rows in (('ReCon-GS reference', a), ('Adaptive binary + RD', b)):
        axes[0].plot(frames, [r['psnr'] for r in rows], label=label)
        axes[2].plot(frames, [r['total_bytes'] / 1024 for r in rows], label=label)
    axes[1].plot(frames, deltas, color='tab:green')
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[0].set_ylabel('Test PSNR (dB)')
    axes[1].set_ylabel('PSNR difference (dB)')
    axes[2].set_ylabel('Motion + incremental (KiB)')
    axes[2].set_xlabel('Frame (shared static frame 0 excluded)')
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle('flame_steak / held-out cam00 / measured container bytes')
    fig.tight_layout()
    fig.savefig(output / 'quality_storage.png', dpi=160)
    plt.close(fig)

    selected = sorted(set([frames[0], frames[len(frames)//2], frames[-1],
                           frames[deltas.index(min(deltas))]]))
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.5 * len(selected)), squeeze=False)
    index = {r['frame']: i for i, r in enumerate(a)}
    for row, frame in enumerate(selected):
        sub = Path('image/frames') / f'{frame:04d}'
        paths = [args.reference / sub / 'cam00_gt.png',
                 args.reference / sub / f"cam00_step{a[index[frame]]['step']}.png",
                 args.candidate / sub / f"cam00_step{b[index[frame]]['step']}.png"]
        with Image.open(paths[0]) as gt, Image.open(args.candidate / sub / 'cam00_gt.png') as other:
            assert gt.size == other.size and gt.tobytes() == other.tobytes(), frame
        for column, path in enumerate(paths):
            with Image.open(path) as picture:
                axes[row, column].imshow(picture)
            axes[row, column].axis('off')
            title = f'Frame {frame}: ground truth'
            if column:
                data = a if column == 1 else b
                name = 'ReCon-GS' if column == 1 else 'Adaptive binary + RD'
                title = f"{name}: {data[index[frame]]['psnr']:.4f} dB"
            axes[row, column].set_title(title)
    fig.tight_layout()
    fig.savefig(output / 'heldout_examples.png', dpi=120)
    plt.close(fig)

    lines = ['# Controlled flame_steak comparison', '',
             f'{len(frames)} dynamic frames ({frames[0]}–{frames[-1]}); shared frame 0 excluded.', '',
             '| Measurement | ReCon-GS | Adaptive binary + RD |', '|---|---:|---:|']
    for key in ('psnr', 'ssim', 'l1', 'total_bytes', 'wall_seconds'):
        lines.append(f"| {key} | {baseline['means'][key]:.6f} | {candidate['means'][key]:.6f} |")
    lines += ['', f"Mean PSNR difference: {metrics['psnr_delta']:+.6f} dB; "
              f"wins in {metrics['psnr_wins']}/{len(frames)} frames.",
              f'Regular training view streams identical: {aligned}.',
              f"Candidate extra backward steps/frame: {candidate['means'].get('extra_backward_steps', 0):.2f}.",
              '', 'Caveats: single-scene exploratory tuning; search computation is additional. '
              'Reference storage is measured float NPZ, not its best possible entropy codec. '
              'Motion packets are round-trip checked, but this is not a complete standalone video codec.',
              '', f'Reference: {args.reference.resolve()}', f'Candidate: {args.candidate.resolve()}',
              '', 'Examples include first, middle, last and worst-difference frames (not only wins).']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
