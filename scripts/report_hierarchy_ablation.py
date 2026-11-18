"""Write 300-timepoint hierarchy experiment records from a verified audit."""
import argparse
import json
from pathlib import Path

MODES=('adaptive','depth2','depth3','depth4')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--short-audit',type=Path,required=True);p.add_argument('--docs',type=Path,required=True);a=p.parse_args()
    full=json.loads(a.audit.read_text());short=json.loads(a.short_audit.read_text())
    if full['total_timepoints']!=300 or full['dynamic_frames']!=299 or short['total_timepoints']!=30:
        raise ValueError('Require audited 300/30-timepoint experiments')
    if set(full['results'])!=set(MODES) or set(short['results'])!=set(MODES):
        raise ValueError('Missing an ablation group')
    dest=a.docs;dest.mkdir(parents=True,exist_ok=True)
    targets=[dest/f'20261115_cook_hierarchy_ablation_{mode}_full300.md' for mode in MODES]+[dest/'20261115_cook_hierarchy_ablation_comparison_full300.md']
    if any(t.exists() for t in targets): raise FileExistsError('An experiment report already exists')
    for mode,path in zip(MODES,targets):
        d=full['results'][mode];s=short['results'][mode]
        components='、'.join(f"{k} {v:.3f}" for k,v in d['component_kb'].items() if v>0)
        body=f'''# Cook Spinach：{mode} 层级消融，300 时间点

## 设置与改动

同一共享首帧和五层候选空间树，从首帧独立递推训练 299 个动态帧；每帧 200 步，前 100 步训练共享平移（自适应组同时训练 gate），后 100 步训练区域运动、逐点残差和属性。前 100 步区域旋转按原方案冻结。gate 在第 100 步后冻结；固定组每帧的 200 步中始终处于指定深度，下一帧按上一帧解码位置重新路由。各组的相机顺序逐帧核对一致。

使用独立逐帧方案 `pipeline/per_frame_hierarchy_ablation.py`，选择策略为 `{mode}`。训练使用原优化、属性与编码流程；额外记录冻结状态及相机轨迹。结果 `{d['run']}`。原自适应 baseline 会在第 151–180 步重新训练 gate，与此阶段不一致，因此本次重新训练自适应对照。短实验为 30 时间点（29 个动态帧），经独立顺序解码和状态摘要校验后运行完整实验。

## 结果

测试视角动态帧均值：PSNR {d['psnr']:.4f} dB，SSIM {d['ssim']:.5f}，LPIPS {d['lpips']:.5f}；实际完整动态包 {d['dynamic_kb']:.3f} KB/帧，活动区域 {d['active_regions']:.1f}，记录训练时间 {d['training_seconds']:.2f} 秒/帧。KB 为 1000 字节，动态包包括各文件与 manifest、不含共享首帧。主要码流分量均值（KB/帧）：{components}。

短实验：PSNR {s['psnr']:.4f} dB、{s['dynamic_kb']:.3f} KB/帧。完整实验所有帧训练与解码渲染检查、包大小检查、顺序解码摘要和冻结/相机轨迹核验通过；逐帧证据与组间对比见 `{a.audit}`。

仅为单场景、单随机种子的结果；画质和码率差异不能单凭总指标归因于自适应层级，也不能推广为跨场景结论。
'''
        path.write_text(body)
    lines=['# Cook Spinach：固定与自适应层级消融，300 时间点','','## 共同设置','', '共享首帧、同一候选树和每帧 200 步训练顺序；首 100 步仅共享平移及自适应 gate，区域旋转在这一阶段沿用原方案冻结；后 100 步训练区域运动、逐点运动和属性，gate 固定。固定深度组每帧按上一解码位置路由到指定层，200 步内归属不变。四组独立从首帧递推，不复用历史动态前缀。','', '| 层级 | PSNR / dB | SSIM | LPIPS | 动态 KB/帧 | 活动区域 | 训练秒/帧 | motion KB/帧 | 区域属性 KB/帧 | 逐点属性 KB/帧 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        d=full['results'][mode];c=d['component_kb'];lines.append(f"| {mode} | {d['psnr']:.4f} | {d['ssim']:.5f} | {d['lpips']:.5f} | {d['dynamic_kb']:.3f} | {d['active_regions']:.1f} | {d['training_seconds']:.2f} | {c['motion']:.3f} | {c['regional_full_attributes']:.3f} | {c['full_attribute_update']:.3f} |")
    adaptive=full['results']['adaptive'];lines+=['','## 画质与存储','']
    for mode in MODES[1:]:
        d=full['results'][mode]; lines.append(f"- 相对 {mode}，自适应 PSNR {adaptive['psnr']-d['psnr']:+.4f} dB，动态存储 {adaptive['dynamic_kb']-d['dynamic_kb']:+.3f} KB/帧，活动区域 {adaptive['active_regions']-d['active_regions']:+.1f}。")
    lines+=['','码率为实际文件大小与 manifest 的总和，KB=1000 字节，不含首帧；motion、区域属性和逐点属性只列主要分量。计时为项目 `training_seconds`，包括验证并扣除诊断日志，排除最终 checkpoint 写入。逐帧相机顺序、参数冻结、固定层归属、训练/解码渲染一致性、逐帧实际字节和独立顺序回放摘要均已核验。','', '自适应旧 baseline 的后半段仍重新训练 gate，本次按相同 100+100 方案从首帧重跑；不可直接把旧 baseline 当成本次对照。仅为单场景、单随机种子；结果不证明改进由自适应层级单独造成，也不能推广到其他场景。',f'\n逐帧明细：`{a.audit}`。四组实验各有独立记录。','']
    targets[-1].write_text('\n'.join(lines));print('WROTE',*(str(x) for x in targets),sep='\n')
if __name__=='__main__':main()
