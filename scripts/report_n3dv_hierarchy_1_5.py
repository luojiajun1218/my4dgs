"""Report independently audited N3DV depth-1/depth-5 runs with earlier controls."""
import argparse
import json
from pathlib import Path
from statistics import mean

SCENES = ('coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon', 'flame_steak', 'sear_steak')
MODES = ('baseline', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5')
METRICS = ('psnr', 'ssim', 'lpips', 'dynamic_kb', 'training_seconds', 'active_regions')
PREVIOUS = Path('output/experiments/n3dv_hierarchy_ablation_20261115_metadata/all_scenes_summary.json')


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(content + '\n')


def data_for_scene(scene, audit):
    data = json.loads(audit.read_text())
    old = json.loads(PREVIOUS.read_text())['scenes'][scene]
    if data['scene'] != scene or data['total_timepoints'] != 300 or data['dynamic_frames'] != 299:
        raise ValueError('Expected a 300-timepoint scene audit')
    if set(data['results']) != {'baseline', 'depth1', 'depth5'}:
        raise ValueError('Expected baseline, depth1 and depth5')
    new = data['results']
    for name in ('psnr', 'ssim', 'lpips', 'dynamic_kb', 'training_seconds', 'active_regions'):
        if abs(new['baseline'][name] - old['baseline'][name]) > 1e-8:
            raise ValueError(f'Existing baseline differs for {scene}: {name}')
    return {**old, **{mode: {k: v for k, v in new[mode].items() if k != 'frames'} for mode in ('depth1', 'depth5')}}


def scene_report(scene, audit, docs):
    rows = data_for_scene(scene, audit)
    base = rows['baseline']
    common = ('使用五层候选树，从共享首帧独立训练；每帧按上一解码位置路由，'
              '训练期间固定区域归属。每帧 200 步，前 100 步训练区域平移，后 100 步训练区域运动、逐点残差与属性；固定组 gate 冻结。'
              '旧 baseline 在第 151–180 步重新训练 gate，且旧版完整源码快照缺失，不能确认其他实现逐行一致。')
    for mode in ('depth1', 'depth5'):
        r = rows[mode]
        parts = '、'.join(f'{k} {v:.3f}' for k, v in r['component_kb'].items() if v > 0)
        text = f'''# N3DV {scene}：{mode} 固定层级，300 时间点

## 方法与代码

{common}

仅扩展 `pipeline/per_frame_hierarchy_ablation.py` 支持选择 {mode}；优化、码流和数据配置沿用此前实验。结果目录：`{r['run']}`。

## 结果与分析

299 动态帧均值：PSNR {r['psnr']:.4f} dB、SSIM {r['ssim']:.5f}、LPIPS {r['lpips']:.5f}；完整动态包 {r['dynamic_kb']:.3f} KB/帧、活动区域 {r['active_regions']:.1f}、训练 {r['training_seconds']:.2f} 秒/帧。包组成（KB/帧）：{parts}。

相对已有自适应 baseline，PSNR {r['psnr']-base['psnr']:+.4f} dB，存储 {r['dynamic_kb']-base['dynamic_kb']:+.3f} KB/帧。共享粒度、内容、量化和优化均可能影响画质与码率；该单次实验不能单独确立原因。KB 按 1000 字节，含完整动态包和 manifest，不含首帧。

逐帧相机、200 步、冻结、指定层归属、训练与解码图像、包字节和完整顺序解码摘要见 `{audit}`。本组纯渲染和完整播放 FPS 未单独测试。
'''
        write_new(docs / f'20261116_n3dv_hierarchy_{scene}_{mode}_full300.md', text.rstrip())
    lines = [f'# N3DV {scene}：自适应与固定 1–5 层，300 时间点', '', common, '',
             '| 组别 | PSNR / dB | SSIM | LPIPS | KB/动态帧 | 活动区域 | 训练秒/帧 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        r = rows[mode]
        lines.append(f"| {mode} | {r['psnr']:.4f} | {r['ssim']:.5f} | {r['lpips']:.5f} | {r['dynamic_kb']:.3f} | {r['active_regions']:.1f} | {r['training_seconds']:.2f} |")
    lines.extend(['', f'每组为 299 动态帧均值；新增组核验明细 `{audit}`。不同日期的负载可能影响训练时间；固定组未测 FPS。单次结果无法独立证明收益来源。'])
    write_new(docs / f'20261116_n3dv_hierarchy_{scene}_comparison_full300.md', '\n'.join(lines))


def aggregate(metadata, docs):
    rows = {scene: data_for_scene(scene, metadata / f'{scene}_full_audit.json') for scene in SCENES}
    averages = {mode: {key: mean(rows[s][mode][key] for s in SCENES) for key in METRICS} for mode in MODES}
    lines = ['# N3DV 六场景：自适应与固定 1–5 层', '',
             '各场景均为 300 时间点，其中 299 个动态帧参与统计。新增固定第 1、5 层独立从首帧训练，其余列为先前结果；每动态帧训练 200 步。', '',
             '| 场景 | baseline PSNR / KB | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        cells = [f"{rows[scene][m]['psnr']:.4f} / {rows[scene][m]['dynamic_kb']:.3f}" for m in MODES]
        lines.append('| ' + scene + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 六场景等权均值', '', '| 组别 | PSNR | SSIM | LPIPS | KB/动态帧 | 活动区域 | 训练秒/帧 |', '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        r = averages[mode]
        lines.append(f"| {mode} | {r['psnr']:.4f} | {r['ssim']:.5f} | {r['lpips']:.5f} | {r['dynamic_kb']:.3f} | {r['active_regions']:.1f} | {r['training_seconds']:.2f} |")
    lines += ['', '训练时间来自不同日期和系统负载。字节包含动态包和 manifest，不含首帧；固定层组尚未测 FPS。旧 baseline 在第 151–180 步重新训练 gate，且旧版完整源码快照缺失，因此不能仅凭数值归因于层级。各场景为单次运行。', '']
    write_new(docs / '20261116_n3dv_hierarchy_all_scenes_1_to_5_full300.md', '\n'.join(lines))
    write_new(metadata / 'all_scenes_summary.json', json.dumps({'scenes': rows, 'means': averages}, indent=2))
    best = json.loads(Path('output/best/index.json').read_text())
    candidates = []
    for scene in SCENES:
        historic = best.get(scene, {})
        for mode in ('depth1', 'depth5'):
            r = rows[scene][mode]
            psnr, size = historic.get('dynamic_psnr'), historic.get('complete_update_bytes')
            valid = historic.get('dynamic_frames', 0) >= 299 and historic.get('last_complete_frame', -1) >= 299 and isinstance(psnr, (int, float)) and isinstance(size, (int, float))
            candidates.append({'scene': scene, 'mode': mode, 'run': r['run'], 'psnr': r['psnr'], 'dynamic_kb': r['dynamic_kb'], 'historical_psnr': psnr, 'historical_dynamic_kb': size / 1000 if isinstance(size, (int, float)) else None, 'comparable': valid, 'dominates_historical_psnr_and_bytes': bool(valid and r['psnr'] > psnr and r['dynamic_kb'] * 1000 <= size), 'historical_availability': historic.get('availability')})
    write_new(metadata / 'best_comparison.json', json.dumps({'results': candidates, 'dominant_candidates': [r for r in candidates if r['dominates_historical_psnr_and_bytes']]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--docs', type=Path, default=Path('docs/experiment'))
    parser.add_argument('--aggregate', action='store_true')
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.metadata, args.docs)
    elif args.scene and args.audit:
        scene_report(args.scene, args.audit, args.docs)
    else:
        parser.error('Provide --aggregate or both --scene and --audit')
