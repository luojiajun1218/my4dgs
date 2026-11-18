"""Write experiment and speed records for zero-storage-loss N3DV runs."""
import argparse
import json
from pathlib import Path
from statistics import mean

SCENES=('coffee_martini','cook_spinach','cut_roasted_beef','flame_salmon','flame_steak','sear_steak')
PREVIOUS=Path('output/experiments/n3dv_hierarchy_1_5_20261116_metadata/all_scenes_summary.json')
PREVIOUS_SPEED=Path('output/experiments/n3dv_hierarchy_1_5_20261116_metadata/speed_summary.json')


def write_new(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(text.rstrip()+'\n')


def load_scene(scene,metadata):
    p=metadata/f'{scene}_audit.json';data=json.loads(p.read_text())
    if data['scene']!=scene or data['total_timepoints']!=300 or data['dynamic_frames']!=299:
        raise ValueError(f'{scene}: incomplete audit')
    base=json.loads(PREVIOUS.read_text())['scenes'][scene]['baseline']
    result={k:v for k,v in data['result'].items() if k!='frames'}
    return base,result


def scene_report(scene,metadata,docs):
    base,r=load_scene(scene,metadata)
    parts='、'.join(f'{k} {v:.3f}' for k,v in r['component_kb'].items() if v>0)
    text=f'''# N3DV {scene}：自适应层级 storage loss 为零，300 时间点

## 思路与设置

对照已有自适应 baseline；每帧 200 步，保留自适应 gate、首帧、相机采样、量化和运动残差约束。将运动码率权重 `rate_loss_weight` 从 0.008 设为 0，同时将逐点透明度、逐点属性和区域属性的存储 L1 权重设为 0；依赖字节价格的近端筛除也随之关闭。运动残差约束 `relative_motion_loss_weight=1.0` 保留。训练实现未改，只使用独立实验配置。

## 结果与分析

299 个动态帧测试均值：PSNR {r['psnr']:.4f} dB、SSIM {r['ssim']:.5f}、LPIPS {r['lpips']:.5f}；完整动态包 {r['dynamic_kb']:.3f} KB/帧；训练 {r['training_seconds']:.2f} 秒/帧；活动区域 {r['active_regions']:.1f}。相对 baseline，PSNR {r['psnr']-base['psnr']:+.4f} dB、存储 {r['dynamic_kb']-base['dynamic_kb']:+.3f} KB/帧。实际包组成（KB/帧）：{parts}。

关闭的是一整组存储相关约束，不能单独判断某个分项的作用；变化可能来自参数稀疏性、区域选择、量化和优化共同作用。旧 baseline 早于工程重构，且未保存完整历史工作区，无法证明非存储相关代码逐行一致；单次实验不等于因果证明或跨场景结论。

对照配置、逐帧相机顺序、关键步骤参数冻结、存储损失和近端价格、实际字节、训练/解码图像与从共享首帧顺序解码摘要已核验：`{metadata/f'{scene}_audit.json'}`。完整运行：`{r['run']}`。KB 为 1000 字节，动态包含 manifest，不含首帧。

测速结果：待运行。
'''
    write_new(docs/f'20261117_n3dv_storage_zero_{scene}_full300.md',text)


def aggregate(metadata,docs):
    scenes={s:dict(zip(('baseline','zero'),load_scene(s,metadata))) for s in SCENES}
    keys=('psnr','ssim','lpips','dynamic_kb','training_seconds','active_regions')
    averages={mode:{k:mean(scenes[s][mode][k] for s in SCENES) for k in keys} for mode in ('baseline','zero')}
    lines=['# N3DV 六场景：自适应层级 storage loss 为零消融','',
           '关闭运动码率价格以及透明度、逐点和区域属性的存储 L1，保留量化、运动残差约束和原有自适应 gate 训练安排。每组为共享首帧加 299 个动态帧，其他超参数沿用 baseline。','',
           '| 场景 | baseline PSNR / KB | storage loss 0 PSNR / KB | PSNR 差 / dB | 存储差 / KB |','|---|---:|---:|---:|---:|']
    for s in SCENES:
        b,r=scenes[s]['baseline'],scenes[s]['zero']
        lines.append(f"| {s} | {b['psnr']:.4f} / {b['dynamic_kb']:.3f} | {r['psnr']:.4f} / {r['dynamic_kb']:.3f} | {r['psnr']-b['psnr']:+.4f} | {r['dynamic_kb']-b['dynamic_kb']:+.3f} |")
    lines+=['','## 六场景等权均值','','| 组别 | PSNR | SSIM | LPIPS | KB/动态帧 | 训练秒/帧 | 活动区域 |','|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('baseline','zero'):
        r=averages[mode];lines.append(f"| {mode} | {r['psnr']:.4f} | {r['ssim']:.5f} | {r['lpips']:.5f} | {r['dynamic_kb']:.3f} | {r['training_seconds']:.2f} | {r['active_regions']:.1f} |")
    lines+=['','关闭整个存储约束是组合消融，不分辨各分项的贡献；运动残差约束与量化仍生效。旧 baseline 完整工作区源码不可得，跨日期、代码版本及负载比较有局限。真实存储含完整动态包和 manifest，不含首帧。','','测速结果：待运行。']
    write_new(docs/'20261117_n3dv_storage_zero_all_scenes_full300.md','\n'.join(lines))
    write_new(metadata/'all_scenes_summary.json',json.dumps(dict(scenes=scenes,means=averages),indent=2))
    best=json.loads(Path('output/best/index.json').read_text());results=[]
    for s in SCENES:
        r,b=scenes[s]['zero'],best.get(s,{});p,size=b.get('dynamic_psnr'),b.get('complete_update_bytes')
        valid=b.get('dynamic_frames',0)>=299 and b.get('last_complete_frame',-1)>=299 and isinstance(p,(float,int)) and isinstance(size,(float,int))
        results.append(dict(scene=s,run=r['run'],psnr=r['psnr'],dynamic_kb=r['dynamic_kb'],historical_psnr=p,
                            historical_dynamic_kb=size/1000 if isinstance(size,(float,int)) else None,comparable=valid,
                            dominates=bool(valid and r['psnr']>p and r['dynamic_kb']*1000<=size),historical_availability=b.get('availability')))
    write_new(metadata/'best_comparison.json',json.dumps(dict(results=results,dominant_candidates=[r for r in results if r['dominates']]),indent=2))


def speed_report(metadata,docs):
    summary=json.loads((metadata/'all_scenes_summary.json').read_text());prior=json.loads(PREVIOUS_SPEED.read_text());results={}
    lines=['# N3DV 六场景：storage loss 为零的渲染和播放速度','',
           'RTX 3090 Ti、cam00，300 时间点每帧预热 3 次、计时渲染 10 次；完整播放从首帧顺序读包、解码、更新并渲染 299 动态帧，重复两轮，包含状态校验，不含一次性首帧和相机加载。baseline 使用此前同口径测速；缓存与系统负载可能影响比较。','',
           '| 场景 | baseline 渲染 / 播放 FPS | storage loss 0 渲染 / 播放 FPS |','|---|---:|---:|']
    for scene in SCENES:
        d=json.loads((metadata/f'{scene}_fps.json').read_text())
        if (Path(d['run']).resolve()!=Path(summary['scenes'][scene]['zero']['run']).resolve() or d['frames']!=300 or len(d['render']['rows'])!=300 or d['render']['renders_per_frame']!=10 or d['playback']['dynamic_frames_per_round']!=299 or len(d['playback']['rounds'])!=2):
            raise ValueError(f'{scene}: incomplete or wrong benchmark')
        r=results[scene]=dict(render_fps=d['render']['fps'],playback_fps=d['playback']['fps']);b=prior[scene]['baseline']
        lines.append(f"| {scene} | {b['render_fps']:.2f} / {b['playback_fps']:.2f} | {r['render_fps']:.2f} / {r['playback_fps']:.2f} |")
    lines+=['',f"storage loss 0 六场景等权平均：纯渲染 {mean(r['render_fps'] for r in results.values()):.2f} FPS，完整播放 {mean(r['playback_fps'] for r in results.values()):.2f} FPS。"]
    write_new(docs/'20261117_n3dv_storage_zero_speed_full300.md','\n'.join(lines));write_new(metadata/'speed_summary.json',json.dumps(results,indent=2))
    overall=docs/'20261117_n3dv_storage_zero_all_scenes_full300.md';text=overall.read_text();assert text.count('测速结果：待运行。')==1
    overall.write_text(text.replace('测速结果：待运行。','测速口径及各场景结果见 `docs/experiment/20261117_n3dv_storage_zero_speed_full300.md`。'))
    for s,r in results.items():
        path=docs/f'20261117_n3dv_storage_zero_{s}_full300.md';text=path.read_text();assert text.count('测速结果：待运行。')==1
        path.write_text(text.replace('测速结果：待运行。',f"RTX 3090 Ti 全序列已解码纯渲染 {r['render_fps']:.2f} FPS、完整播放 {r['playback_fps']:.2f} FPS；详见 `docs/experiment/20261117_n3dv_storage_zero_speed_full300.md`。"))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--docs',type=Path,default=Path('docs/experiment'))
    p.add_argument('--scene',choices=SCENES);p.add_argument('--aggregate',action='store_true');p.add_argument('--speed',action='store_true');a=p.parse_args()
    if a.aggregate:aggregate(a.metadata,a.docs)
    elif a.speed:speed_report(a.metadata,a.docs)
    elif a.scene:scene_report(a.scene,a.metadata,a.docs)
    else:p.error('Select --scene, --aggregate or --speed')
