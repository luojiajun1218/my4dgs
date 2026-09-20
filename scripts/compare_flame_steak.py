"""Compare final-step metrics and held-out renders of two existing runs."""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def read_metrics(run, num_frames=5):
    rows, frame, split = {}, None, None
    for line in (run / 'logging.INFO').read_text().splitlines():
        match = re.search(r'Start learning frame (\d+)', line)
        if match:
            frame = int(match[1])
        match = re.search(r'Evaluate (train|test) dataset at step (\d+)', line)
        if match:
            split = match[1] if int(match[2]) == 200 else None
        match = re.search(r'L1: ([\d.]+).*PSNR: ([\d.]+).*SSIM: ([\d.]+)', line)
        if match and split and frame in range(1, num_frames + 1):
            rows.setdefault(frame, {})[split] = dict(zip(('l1', 'psnr', 'ssim'), map(float, match.groups())))
            split = None
    assert set(rows) == set(range(1, num_frames + 1)), rows
    assert all(set(v) == {'train', 'test'} for v in rows.values())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('gradhet', type=Path)
    parser.add_argument('--num-frames', type=int, default=5)
    args = parser.parse_args()
    a, b = read_metrics(args.baseline, args.num_frames), read_metrics(args.gradhet, args.num_frames)
    output = args.gradhet / 'comparison'
    output.mkdir(exist_ok=True)
    rows = [dict(frame=f, recongs=a[f], gradhet=b[f], test_psnr_delta=b[f]['test']['psnr']-a[f]['test']['psnr']) for f in a]
    means = {name: {split: {m: sum(data[f][split][m] for f in data)/len(data) for m in ('l1', 'psnr', 'ssim')} for split in ('train', 'test')} for name, data in (('recongs', a), ('gradhet', b))}
    (output / 'metrics.json').write_text(json.dumps(dict(baseline=str(args.baseline), gradhet=str(args.gradhet), frames=rows, means=means), indent=2))
    report = [f'# Flame Steak：共享首帧后续 {args.num_frames} 帧对比', '', f'相同首帧 checkpoint；帧索引 1–{args.num_frames}，每帧 200 步，20 个训练视角、1 个测试视角 cam00。resolution=-1（代码将大图缩放到 1600 宽），PSNR / SSIM 均越高越好。', '', '| 帧索引 | ReCon-GS PSNR | GradHet v2 PSNR | 差值 dB | ReCon-GS SSIM | GradHet v2 SSIM |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        x, y = r['recongs']['test'], r['gradhet']['test']
        report.append(f"| {r['frame']} | {x['psnr']:.4f} | {y['psnr']:.4f} | {r['test_psnr_delta']:+.4f} | {x['ssim']:.4f} | {y['ssim']:.4f} |")
    x, y = means['recongs']['test'], means['gradhet']['test']
    report += [f"| 平均 | {x['psnr']:.4f} | {y['psnr']:.4f} | {y['psnr']-x['psnr']:+.4f} | {x['ssim']:.4f} | {y['ssim']:.4f} |", '', '这是单次运行对比。两份代码均默认使用 seed=0，但不同算法会消耗不同数量的随机数，未逐步强制对齐随机状态，不能据此断言方法具有稳定优势。LPIPS 未启用。', '', f'基线：{args.baseline}', f'GradHet：{args.gradhet}']
    if (args.baseline / 'requested_stop.json').exists():
        report += ['', 'ReCon-GS 原按 50 个后续帧启动，用户中途改为 20 帧，完成第 20 帧保存与评估后请求停止。SIGINT 未生效，随后以 SIGTERM 终止；多出的产物保存在 extra_after_requested_stop，不参与本报告。基线运行时加载 num_frames=51，GradHet 加载 num_frames=21；两组评估均仅取帧 1–20，相机真值逐帧校验一致。数据集长度会影响随机打乱顺序，因此本次并非逐步随机状态对齐的消融。']
    (output / 'report.md').write_text('\n'.join(report)+'\n')
    for f in a:
        sub = Path('image/frames') / f'{f:04d}'
        with Image.open(args.gradhet/sub/'cam00_gt.png') as gt, Image.open(args.baseline/sub/'cam00_gt.png') as other:
            assert gt.size == other.size and gt.tobytes() == other.tobytes(), f
    selected = list(a) if args.num_frames <= 5 else [1, args.num_frames//4, args.num_frames//2, 3*args.num_frames//4, args.num_frames]
    fig, axes = plt.subplots(len(selected), 3, figsize=(15, 3.8*len(selected)), squeeze=False)
    for i, f in enumerate(selected):
        sub = Path('image/frames') / f'{f:04d}'
        paths = [args.gradhet/sub/'cam00_gt.png', args.baseline/sub/'cam00_step200.png', args.gradhet/sub/'cam00_step200.png']
        assert Image.open(paths[0]).tobytes() == Image.open(args.baseline/sub/'cam00_gt.png').tobytes()
        for j, path in enumerate(paths):
            axes[i, j].imshow(Image.open(path))
            axes[i, j].axis('off')
            title = f'Frame {f} / GT' if j == 0 else f"{'ReCon-GS' if j == 1 else 'GradHet v2'} / PSNR {(a if j == 1 else b)[f]['test']['psnr']:.4f}"
            axes[i, j].set_title(title)
    fig.tight_layout()
    fig.savefig(output/'heldout_comparison.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, metric in zip(axes, ('psnr', 'ssim')):
        for label, data in (('ReCon-GS', a), ('GradHet v2', b)):
            ax.plot(list(data), [data[f]['test'][metric] for f in data], label=label)
        ax.set_ylabel('PSNR (dB)' if metric == 'psnr' else 'SSIM')
        ax.grid(alpha=.25)
        ax.legend()
    axes[-1].set_xlabel('Frame index (shared first frame = 0)')
    fig.suptitle(f'Flame Steak / held-out cam00 / {args.num_frames} incremental frames')
    fig.tight_layout()
    fig.savefig(output/'metric_curves.png', dpi=160)
    plt.close(fig)
    print('\n'.join(report))


if __name__ == '__main__':
    main()
