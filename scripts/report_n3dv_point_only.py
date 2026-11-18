"""Create N3DV point-only ablation reports from independently replayed runs."""
import argparse
import json
from pathlib import Path
from statistics import mean

SCENES = ('coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon', 'flame_steak', 'sear_steak')
MODES = ('baseline', 'depth0', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5')
KEYS = ('psnr', 'ssim', 'lpips', 'dynamic_kb', 'training_seconds')
PREVIOUS = Path('output/experiments/n3dv_hierarchy_1_5_20261116_metadata/all_scenes_summary.json')
SPEED_PREVIOUS = Path('output/experiments/n3dv_hierarchy_1_5_20261116_metadata/speed_summary.json')


def write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(text.rstrip() + '\n')


def get_rows(scene, audit):
    data = json.loads(audit.read_text())
    if data['scene'] != scene or data['total_timepoints'] != 300 or data['dynamic_frames'] != 299 or set(data['results']) != {'depth0'}:
        raise ValueError('Expected an audited full point-only sequence')
    old = json.loads(PREVIOUS.read_text())['scenes'][scene]
    row = {k: v for k, v in data['results']['depth0'].items() if k != 'frames'}
    return {**old, 'depth0': row}


def scene_report(scene, metadata, docs):
    audit = metadata / f'{scene}_audit.json'
    rows = get_rows(scene, audit)
    r, base = rows['depth0'], rows['baseline']
    parts = '、'.join(f'{key} {value:.3f}' for key, value in r['component_kb'].items() if value > 0)
    text = f'''# N3DV {scene}：0 层逐点运动消融，300 时间点

## 方法与代码

不学习区域运动、区域 gate 或共享区域属性；第 1–100 步优化逐点平移，第 101–200 步优化逐点运动和属性。使用与对照相同的首帧、相机顺序、量化参数和 200 步。逐点平移有 200 步，其他固定层对照的逐点运动主要在后 100 步更新；关闭共享属性也是差异，不能仅将结果归因于区域运动。

新增 `pipeline/per_frame_point_only.py`，通过 `RECONGS_POINT_ONLY=1` 选择；旧码流仍保留固定空间路由元数据以兼容原解码器，但不传输非零区域运动或区域属性包。该兼容开销计入实际字节。

## 结果与分析

299 个动态帧测试均值：PSNR {r['psnr']:.4f} dB、SSIM {r['ssim']:.5f}、LPIPS {r['lpips']:.5f}；动态包 {r['dynamic_kb']:.3f} KB/帧、训练 {r['training_seconds']:.2f} 秒/帧。各包均值 KB/帧：{parts}。相对 baseline PSNR {r['psnr']-base['psnr']:+.4f} dB，动态存储 {r['dynamic_kb']-base['dynamic_kb']:+.3f} KB/帧；内容、优化步数和属性编码均可能影响差异。

训练与解码图像、200 步相机顺序、逐点与区域参数训练状态、实际字节、零区域运动符号、从共享首帧开始的逐帧 CPU 回放摘要已核验：`{audit}`。完整运行：`{r['run']}`。KB=1000 B，含动态包和 manifest、不含首帧。旧 baseline 的完整历史工作区缺失，不能证明所有非 ablation 代码逐行相同。

测速结果：待运行。
'''
    write_new(docs / f'20261117_n3dv_point_only_{scene}_full300.md', text)


def aggregate(metadata, docs):
    data = {s: get_rows(s, metadata / f'{s}_audit.json') for s in SCENES}
    averages = {m: {k: mean(data[s][m][k] for s in SCENES) for k in KEYS} for m in MODES}
    lines = ['# N3DV 六场景：0 层逐点运动与自适应、固定层级对照', '',
             '0 层只训练逐点运动和逐点属性，保留旧码流的惰性空间元数据并计入存储。每场景 300 时间点，统计 299 动态帧。其他层复用已有结果。', '',
             '| 场景 | baseline | depth 0 | depth 1 | depth 2 | depth 3 | depth 4 | depth 5 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in SCENES:
        cells = [f"{data[s][m]['psnr']:.4f} / {data[s][m]['dynamic_kb']:.3f}" for m in MODES]
        lines.append('| ' + s + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '数值为 PSNR dB / 实际动态包 KB/帧；含 manifest、不含首帧。', '',
              '## 六场景等权均值', '', '| 组别 | PSNR | SSIM | LPIPS | KB/帧 | 训练秒/帧 |', '|---|---:|---:|---:|---:|---:|']
    for m in MODES:
        r=averages[m]
        lines.append(f"| {m} | {r['psnr']:.4f} | {r['ssim']:.5f} | {r['lpips']:.5f} | {r['dynamic_kb']:.3f} | {r['training_seconds']:.2f} |")
    lines += ['', '点运动优化步数、区域属性开关及历史 baseline 代码版本均存在差异，不应将结果单独归因于运动共享层级。训练时间跨日期和负载比较。', '', '测速结果：待运行。']
    write_new(docs / '20261117_n3dv_point_only_all_scenes_full300.md', '\n'.join(lines))
    write_new(metadata / 'all_scenes_summary.json', json.dumps({'scenes':data,'means':averages}, indent=2))
    best = json.loads(Path('output/best/index.json').read_text())
    results=[]
    for scene in SCENES:
        r, b = data[scene]['depth0'], best.get(scene, {})
        p, size = b.get('dynamic_psnr'), b.get('complete_update_bytes')
        valid = b.get('dynamic_frames', 0) >= 299 and b.get('last_complete_frame', -1) >= 299 and isinstance(p, (int,float)) and isinstance(size,(int,float))
        results.append(dict(scene=scene, run=r['run'], psnr=r['psnr'], dynamic_kb=r['dynamic_kb'],
                            historical_psnr=p, historical_dynamic_kb=size/1000 if isinstance(size,(int,float)) else None,
                            comparable=valid, dominates=bool(valid and r['psnr'] > p and r['dynamic_kb']*1000 <= size),
                            historical_availability=b.get('availability')))
    write_new(metadata / 'best_comparison.json', json.dumps({'results':results,'dominant_candidates':[r for r in results if r['dominates']]},indent=2))


def speed_report(metadata, docs):
    summary=json.loads((metadata/'all_scenes_summary.json').read_text())
    prior=json.loads(SPEED_PREVIOUS.read_text())
    results={}
    lines=['# N3DV 六场景：0 层逐点方案速度', '',
           'RTX 3090 Ti，cam00；300 时间点的已解码纯渲染每帧预热 3 次、计时 10 次；完整播放从首帧顺序读包、解码、更新并渲染 299 动态帧，重复两轮，含状态校验、不含一次性首帧及相机加载。基线是此前独立同口径测速。', '',
           '| 场景 | baseline 渲染 / 播放 FPS | depth 0 渲染 / 播放 FPS | depth 1 渲染 / 播放 FPS | depth 5 渲染 / 播放 FPS |', '|---|---:|---:|---:|---:|']
    for scene in SCENES:
        r=json.loads((metadata/f'{scene}_fps.json').read_text())
        if (Path(r['run']).resolve()!=Path(summary['scenes'][scene]['depth0']['run']).resolve() or
                r['frames']!=300 or len(r['render']['rows'])!=300 or r['render']['renders_per_frame']!=10 or
                r['playback']['dynamic_frames_per_round']!=299 or len(r['playback']['rounds'])!=2):
            raise ValueError(f'{scene}: incomplete or mismatched FPS benchmark')
        results[scene]=dict(render_fps=r['render']['fps'],playback_fps=r['playback']['fps'])
        def cell(x): return f"{x['render_fps']:.2f} / {x['playback_fps']:.2f}"
        lines.append('| '+scene+' | '+' | '.join((cell(prior[scene]['baseline']),cell(results[scene]),cell(prior[scene]['depth1']),cell(prior[scene]['depth5'])))+' |')
    lines += ['', f"depth 0 六场景等权平均：纯渲染 {mean(r['render_fps'] for r in results.values()):.2f} FPS；完整播放 {mean(r['playback_fps'] for r in results.values()):.2f} FPS。不同运行日期和文件缓存会影响速度。"]
    write_new(docs/'20261117_n3dv_point_only_speed_full300.md','\n'.join(lines))
    write_new(metadata/'speed_summary.json',json.dumps(results,indent=2))
    overall=docs/'20261117_n3dv_point_only_all_scenes_full300.md';text=overall.read_text();assert text.count('测速结果：待运行。')==1
    overall.write_text(text.replace('测速结果：待运行。','测速见 `docs/experiment/20261117_n3dv_point_only_speed_full300.md`。'))
    for scene,r in results.items():
        path=docs/f'20261117_n3dv_point_only_{scene}_full300.md';text=path.read_text();assert text.count('测速结果：待运行。')==1
        path.write_text(text.replace('测速结果：待运行。',f"RTX 3090 Ti 全序列已解码纯渲染 {r['render_fps']:.2f} FPS，完整播放 {r['playback_fps']:.2f} FPS；口径和逐场景对照见 `docs/experiment/20261117_n3dv_point_only_speed_full300.md`。"))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--docs',type=Path,default=Path('docs/experiment'))
    p.add_argument('--scene',choices=SCENES);p.add_argument('--aggregate',action='store_true');p.add_argument('--speed',action='store_true');a=p.parse_args()
    if a.aggregate: aggregate(a.metadata,a.docs)
    elif a.speed: speed_report(a.metadata,a.docs)
    elif a.scene: scene_report(a.scene,a.metadata,a.docs)
    else: p.error('Select --scene, --aggregate or --speed')
