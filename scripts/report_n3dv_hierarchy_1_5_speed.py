"""Summarize full-sequence rendering and playback benchmarks for depth 1 and 5."""
import argparse
import json
from pathlib import Path
from statistics import mean

SCENES = ('coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon', 'flame_steak', 'sear_steak')
MODES = ('baseline', 'depth1', 'depth5')


def main(metadata, docs):
    summary = json.loads((metadata / 'all_scenes_summary.json').read_text())
    rows = {}
    for scene in SCENES:
        rows[scene] = {}
        for mode in MODES:
            path = (Path('output/experiments/n3dv_speed_20261114') / f'baseline_{scene}.json'
                    if mode == 'baseline' else metadata / f'{scene}_{mode}_fps.json')
            d = json.loads(path.read_text())
            if (Path(d['run']).resolve() != Path(summary['scenes'][scene][mode]['run']).resolve()
                    or d['frames'] != 300 or len(d['render']['rows']) != 300
                    or d['render']['renders_per_frame'] != 10 or len(d['playback']['rounds']) != 2
                    or d['playback']['dynamic_frames_per_round'] != 299):
                raise ValueError(f'Benchmark does not match run or full-sequence protocol: {path}')
            rows[scene][mode] = {'render_fps': d['render']['fps'], 'playback_fps': d['playback']['fps'], 'source': str(path)}
    lines = ['# N3DV 固定第 1、5 层：纯渲染与完整播放速度', '',
             'RTX 3090 Ti、各场景 cam00；每个时间点已解码后预热 3 次，计时 10 次渲染；完整播放从首帧顺序读取、解码、更新并渲染 299 个动态帧，重复两轮。播放计时含逐帧状态校验，不含一次性首帧与相机加载，可能受到操作系统文件缓存影响。baseline 使用此前同口径的独立测速。', '',
             '| 场景 | baseline 渲染 / 播放 FPS | depth 1 渲染 / 播放 FPS | depth 5 渲染 / 播放 FPS |', '|---|---:|---:|---:|']
    for scene in SCENES:
        cells = [f"{rows[scene][mode]['render_fps']:.2f} / {rows[scene][mode]['playback_fps']:.2f}" for mode in MODES]
        lines.append('| ' + scene + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '六场景平均：' + '；'.join(
        f"{mode} 已解码渲染 {mean(rows[s][mode]['render_fps'] for s in SCENES):.2f} FPS、完整播放 {mean(rows[s][mode]['playback_fps'] for s in SCENES):.2f} FPS" for mode in MODES) + '。',
        '', '不同日期的机器负载和文件缓存可能影响结果；固定第 2–4 层未在此测速。训练时间和画质、存储结果见同次实验汇总。']
    target = docs / '20261116_n3dv_hierarchy_1_5_speed_full300.md'
    with target.open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with (metadata / 'speed_summary.json').open('x', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--docs', type=Path, default=Path('docs/experiment'))
    args = p.parse_args()
    main(args.metadata, args.docs)
