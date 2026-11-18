"""Write per-run and cross-scene reports from verified N3DV hierarchy audits."""
import argparse
import json
from pathlib import Path
from statistics import mean

SCENES = ('coffee_martini','cook_spinach','cut_roasted_beef','flame_salmon','flame_steak','sear_steak')
MODES = ('baseline','depth2','depth3','depth4')


def write_new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: f.write(text)


def scene_report(scene, audit, short_audit, docs):
    data=json.loads(audit.read_text())
    short=json.loads(short_audit.read_text()) if short_audit is not None and short_audit.is_file() else None
    if data['scene'] != scene or data['dynamic_frames'] != 299 or (short is not None and short['dynamic_frames'] != 29):
        raise ValueError('Expected verified 300-timepoint report and, if provided, verified 30-timepoint report')
    result=data['results']; base=result['baseline']
    if set(result)!=set(MODES):raise ValueError('Missing a comparison group')
    output=[docs/f'20261115_n3dv_hierarchy_{scene}_{m}_full300.md' for m in MODES[1:]]
    output.append(docs/f'20261115_n3dv_hierarchy_{scene}_comparison_full300.md')
    if any(p.exists() for p in output):raise FileExistsError('A scene report already exists')
    for mode,path in zip(MODES[1:],output):
        r=result[mode];s=short['results'][mode] if short is not None else None
        short_note=(f"此前 30 时间点（29 个动态帧）短实验 PSNR {s['psnr']:.4f} dB、动态 {s['dynamic_kb']:.3f} KB/帧，通过相机、冻结、实际字节和独立解码核验后开始全序列。"
                    if s is not None else '按用户要求跳过剩余短实验，直接独立训练完整序列；完整结果仍逐帧核验。')
        parts='、'.join(f"{k} {v:.3f}" for k,v in r['component_kb'].items() if v>0)
        motion=r['component_kb']['motion']-base['component_kb']['motion']
        region=r['component_kb']['regional_full_attributes']-base['component_kb']['regional_full_attributes']
        body=f'''# N3DV {scene}：{mode} 固定层级，300 时间点

## 方法与设置

从与已有 baseline 相同的共享首帧独立递推训练 299 个动态帧，共用五层候选空间树。每帧 200 步，前 100 步固定区域归属、训练共享平移，区域旋转沿用原方案冻结；后 100 步训练区域运动、逐点运动残差和属性，固定 gate。每帧开始按上一帧解码位置路由，200 步内归属不变。相机采样与 baseline 按 200 步逐帧比较。

使用独立方案 `pipeline/per_frame_hierarchy_ablation.py`，仅选择固定深度 `{mode}`，保留既有优化和码流实现；记录额外冻结和相机诊断。run：`{r['run']}`。{short_note}

## 结果与分析

动态帧测试均值：PSNR {r['psnr']:.4f} dB，SSIM {r['ssim']:.5f}，LPIPS {r['lpips']:.5f}；完整动态包 {r['dynamic_kb']:.3f} KB/帧（1000 B/KB，不含首帧），活动区域 {r['active_regions']:.1f}，项目训练时间 {r['training_seconds']:.2f} 秒/帧。文件分量均值 KB/帧：{parts}。

相对现有 baseline，PSNR {r['psnr']-base['psnr']:+.4f} dB、存储 {r['dynamic_kb']-base['dynamic_kb']:+.3f} KB/帧、活动区域 {r['active_regions']-base['active_regions']:+.1f}；运动包差 {motion:+.3f} KB/帧，区域属性包差 {region:+.3f} KB/帧。该分解反映本次不同共享粒度下的实际字节分布；内容、量化和优化也可能影响结果，不能单独确定原因。

逐帧数据、训练/解码渲染抽查、完整顺序回放状态摘要、相机与冻结核验在 `{audit}`。baseline 的 gate 在第 151–180 步重新训练，固定组不重新选择；其训练时 Git 版本早于工程重构且缺少当时完整工作区源码快照。单场景结果不能说明跨场景结论。
'''
        write_new(path,body)
    lines=[f'# N3DV {scene}：现有自适应 baseline 与固定深度消融（300 时间点）','',
           '共享首帧和同一候选树；固定深度组分别独立从首帧训练。前 100 步训练共享平移，区域旋转沿用原方案冻结；后 100 步训练区域运动、逐点残差和属性。固定组每帧根据上一解码位置路由并在训练中固定归属。原 baseline 在 151–180 步继续训练 gate，固定组始终冻结。','',
           '| 组别 | PSNR / dB | SSIM | LPIPS | 动态 KB/帧 | 活动区域 | 训练秒/帧 | 运动 KB/帧 | 区域属性 KB/帧 | 逐点属性 KB/帧 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        r=result[mode];c=r['component_kb'];lines.append(f"| {mode} | {r['psnr']:.4f} | {r['ssim']:.5f} | {r['lpips']:.5f} | {r['dynamic_kb']:.3f} | {r['active_regions']:.1f} | {r['training_seconds']:.2f} | {c['motion']:.3f} | {c['regional_full_attributes']:.3f} | {c['full_attribute_update']:.3f} |")
    lines += ['', '## 画质与存储','']
    for mode in MODES[1:]:
        r=result[mode];lines.append(f"- 相对 {mode}，baseline PSNR {base['psnr']-r['psnr']:+.4f} dB，存储 {base['dynamic_kb']-r['dynamic_kb']:+.3f} KB/帧；运动包差 {base['component_kb']['motion']-r['component_kb']['motion']:+.3f} KB/帧，区域属性包差 {base['component_kb']['regional_full_attributes']-r['component_kb']['regional_full_attributes']:+.3f} KB/帧。")
    lines += ['', '动态大小统计实际帧包与 manifest，不含首帧；所列文件分量不含不透明度等其他文件。时间为各自记录的训练时间，跨日期可能受负载影响。分量差异描述实际字节，不能据此单独归因。',
              '', '逐帧相机顺序、关键步骤非 gate 参数冻结、固定层归属、训练与解码的一个训练视角渲染、包字节和首帧起顺序回放摘要均已核验。共享首帧 SHA256 相同；配置除输出位置、时间点数和 checkpoint 保存帧外一致。原 baseline 早于工程重构且没有当时完整工作区源码快照，不能证明非 gate 实现逐行相同。',
              '',f'单场景、单随机种子结果；不可作为跨场景结论。逐帧明细：`{audit}`。','']
    write_new(output[-1],'\n'.join(lines))
    print(f'WROTE {scene}: '+', '.join(str(x) for x in output),flush=True)


def aggregate(metadata, docs):
    files={scene:(Path('output/experiments/cook_hierarchy_ablation_20261115_metadata/baseline_comparison_audit.json')
                  if scene=='cook_spinach' else metadata/f'{scene}_full_audit.json') for scene in SCENES}
    scenes={name:json.loads(path.read_text()) for name,path in files.items()}
    if any(item['dynamic_frames']!=299 or set(item['results'])!=set(MODES) for item in scenes.values()):
        raise ValueError('All scenes require four independently audited 299-frame sequences')
    path=docs/'20261115_n3dv_hierarchy_all_scenes_full300.md'
    lines=['# N3DV 六场景：自适应 baseline 与固定深度消融','',
           '每个场景均使用已有自适应 baseline、同一共享首帧与 200 步训练，相机采样按动态帧逐步核验。固定组每帧根据上一解码位置路由，训练中固定指定深度。baseline 在 151–180 步继续训练 gate；固定组 gate 始终冻结。不同日期、不同版本历史 baseline 的非 gate 代码不能凭现存快照完全证明逐行一致。','',
           '| 场景 | baseline PSNR / KB | depth 2 PSNR / KB | depth 3 PSNR / KB | depth 4 PSNR / KB |','|---|---:|---:|---:|---:|']
    for scene in SCENES:
        r=scenes[scene]['results'];cells=[f"{r[m]['psnr']:.4f} / {r[m]['dynamic_kb']:.3f}" for m in MODES];lines.append('| '+scene+' | '+' | '.join(cells)+' |')
    avgs={m:{k:mean(scenes[s]['results'][m][k] for s in SCENES) for k in ('psnr','ssim','lpips','dynamic_kb','training_seconds','active_regions')} for m in MODES}
    lines += ['','## 六场景等权均值','', '| 组别 | PSNR | SSIM | LPIPS | KB/帧 | 活动区域 | 训练秒/帧 |','|---|---:|---:|---:|---:|---:|---:|']
    for m in MODES:
        r=avgs[m];lines.append(f"| {m} | {r['psnr']:.4f} | {r['ssim']:.5f} | {r['lpips']:.5f} | {r['dynamic_kb']:.3f} | {r['active_regions']:.1f} | {r['training_seconds']:.2f} |")
    lines += ['', '## 各场景结果','']
    for scene in SCENES:
        r=scenes[scene]['results'];base=r['baseline'];sub=[]
        for m in MODES[1:]:
            v=r[m];sub.append(f"{m}: PSNR {v['psnr']-base['psnr']:+.3f} dB、{v['dynamic_kb']-base['dynamic_kb']:+.2f} KB/帧")
        lines.append(f"- {scene}："+'；'.join(sub)+'。')
    lines += ['', '每项为 299 个动态帧测试视角均值；跨场景均值为六场景等权平均，KB=1000 B，包含完整实际动态包与 manifest、不含首帧。不同运行日期的训练时间不保证相同系统负载。运动和属性码流、分段结果与单次实验限制见各场景报告和审计 JSON。','',
              '这些结果反映各场景的单次运行，画质与码率可能同时受动态内容、区域数、量化、优化和代码版本影响；不能仅凭平均值改善归因于自适应层级，也不构成跨数据集或超过 SOTA 的结论。','']
    write_new(path,'\n'.join(lines)); print(f'WROTE {path}',flush=True)
    payload=dict(scenes={s:{m:{k:v for k,v in scenes[s]['results'][m].items() if k not in ('frames',)} for m in MODES} for s in SCENES},means=avgs,source_audits={s:str(p) for s,p in files.items()})
    write_new(metadata/'all_scenes_summary.json',json.dumps(payload,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene');p.add_argument('--audit',type=Path);p.add_argument('--short-audit',type=Path);p.add_argument('--metadata',type=Path);p.add_argument('--docs',type=Path,default=Path('docs/experiment'));p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    if a.aggregate:
        if a.metadata is None:p.error('--metadata required for aggregation')
        aggregate(a.metadata,a.docs)
    else:
        if not (a.scene and a.audit):p.error('--scene and --audit required')
        scene_report(a.scene,a.audit,a.short_audit,a.docs)
