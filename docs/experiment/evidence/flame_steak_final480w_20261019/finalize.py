"""Archive only lightweight evidence after the final 480 W experiment finishes."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path('/home/gamer-02/Documents/DeepLearning/my4dgs')
C = Path(__file__).resolve().parent
M = C / 'measurement'
E = ROOT / 'docs/experiment/evidence/flame_steak_final480w_20261019'
DOC = ROOT / 'docs/experiment/20261019_flame_steak_final480w_full300.md'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


launch = load(C / 'launch.json')
RUN = Path(launch['run'])
assert load(C / 'progress.json')['phase'] == 'complete'
S = load(M / 'summary.json')
B = load(M / 'decoded_sequence.json')
with (M / 'per_frame_metrics.csv').open() as f:
    rows = list(csv.DictReader(f))
with (M / 'per_30_frames.csv').open() as f:
    segments = list(csv.DictReader(f))
dynamic = rows[1:]
assert S['status'] == 'complete' and S['full_sequence_evaluation']
assert B['sequence_complete'] and B['no_training_updates']
assert S['frames'] == B['measured_frames'] == B['planned_frames'] == 300
assert S['dynamic_frames'] == 299 and len(segments) == 10
assert [int(r['frame']) for r in rows] == [r['frame'] for r in B['frames']] == list(range(300))
assert S['max_cache_pixel_difference'] == 0
assert S['gpu_exclusivity_verified']
assert all(v == 0 for v in S['max_saved_quality_differences'].values())
assert all(int(r['train_steps']) == 300 and float(r['gs_rotation_blocks']) == 0 for r in dynamic)
assert sum(int(r['new_gaussians']) - int(r['retired_gaussians']) for r in dynamic) == S['final_gaussians'] - S['initial_gaussians']
assert abs(statistics.mean(float(r['psnr']) for r in rows) - S['psnr']) < 1e-12
assert sum(int(r['complete_update_bytes']) for r in dynamic) == B['storage']['complete_update_bytes_total']
for r in rows:
    assert sum(int(r[k]) for k in ('motion_bytes', 'opacity_bytes', 'feature_bytes', 'birth_bytes', 'manifest_bytes')) == int(r['complete_update_bytes'])
source = C / 'source'
manifest = load(source / 'source_manifest.json')
assert all(sha(source / name) == digest for name, digest in manifest['hashes'].items())
training = {n: d for n, d in manifest['hashes'].items() if n.startswith(('pipeline/', 'extensions/')) or n == 'main.py'}
assert all(sha(ROOT / name) == digest for name, digest in training.items())
actual = yaml.safe_load((RUN / 'config.yaml').read_text())
base = yaml.safe_load((ROOT / 'config/n3dv/flame_steak_half_byte_budget_20261019.yaml').read_text())
assert not actual.get('resume_prefix_dir') and not launch['continuation']
assert actual['ckpt_path'] == base['ckpt_path']
assert all(actual[k] == base[k] for k in ('module', 'trainer', 'data'))
assert sha(Path(actual['ckpt_path'])) == '61f38b3ff4ffc40655b1bad75834ce92c1540537962f0a65438b5a2646b4384e'
assert (RUN / 'rigid-ckpt-299-300.pth').is_file()
assert 'Finished pipeline my4dgs.' in (C / 'training.log').read_text(errors='replace')
provenance = load(M / 'measurement_provenance.json')
assert all(provenance[k].split(',')[0].strip() == '480.00 W' for k in ('power_before', 'power_after'))
power = [load_line for line in (C / 'power_samples.jsonl').read_text().splitlines() if (load_line := json.loads(line))]
assert power and all(p['power'].split(',')[0].strip() == '480.00 W' for p in power)
previous_root = ROOT / 'docs/experiment/evidence/half_byte_budget_full300_20261019'
previous = load(previous_root / 'summary.json')
assert sha(RUN / 'frame_storage/rigid_tree/reference_frame0.npz') == sha(Path(previous['run']) / 'frame_storage/rigid_tree/reference_frame0.npz')
sizes = [int(r['complete_update_bytes']) for r in dynamic]
worst = min(rows, key=lambda r: float(r['psnr']))
best = max(rows, key=lambda r: float(r['psnr']))
audit = dict(status='verified', checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    final_result_designated_by_user=True, run=str(RUN), frames=300, dynamic_frames=299,
    no_dynamic_prefix=True, fixed_parameters=True, unchanged_training_source_files=len(training),
    frozen_source_files_verified=len(manifest['hashes']),
    initial_checkpoint=actual['ckpt_path'], initial_checkpoint_sha256=sha(Path(actual['ckpt_path'])),
    initial_reference_identical_to_350w=True, reference_bytes=S['shared_reference_bytes'],
    psnr=S['psnr'], dynamic_complete_kb=S['dynamic_complete_kb'], pure_render_fps=S['aggregate_materialized_fps'],
    complete_update_median_kb=statistics.median(sizes)/1000,
    complete_update_p95_kb=float(np.percentile(sizes, 95))/1000,
    complete_update_max_kb=max(sizes)/1000,
    frames_at_most_50kb=sum(size <= 50000 for size in sizes),
    reference_amortized_kb_per_frame=S['total_stream_bytes']/300/1000,
    worst_quality_frame={'frame': int(worst['frame']), 'psnr': float(worst['psnr'])},
    best_quality_frame={'frame': int(best['frame']), 'psnr': float(best['psnr'])},
    power_limit_watts=480, training_power_samples=len(power),
    sampled_peak_power_watts=max(float(p['power'].split(',')[1].strip().split()[0]) for p in power),
    sampled_peak_temperature_c=max(float(p['power'].split(',')[2].strip()) for p in power),
    measurement_provenance=provenance,
    comparison_to_350w={k: {'previous': previous[k], 'final': S[k], 'delta': S[k]-previous[k]}
        for k in ('psnr', 'ssim', 'lpips', 'dynamic_complete_kb', 'aggregate_materialized_fps', 'mean_train_wall_seconds')},
    target_checks=S['goals'])
(M / 'final_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
assert not E.exists() and not DOC.exists()
E.mkdir(parents=True)
files = {n: M/n for n in ('summary.json', 'per_frame_metrics.csv', 'per_30_frames.csv', 'decoded_sequence.json',
    'metrics_comparison.png', 'measurement_provenance.json', 'final_audit.json')}
files.update({
    'config.yaml': RUN/'config.yaml', 'source_manifest.json': source/'source_manifest.json',
    'launch.json': C/'launch.json', 'progress.json': C/'progress.json', 'preflight.json': C/'preflight.json',
    'standard_recipe_check.json': C/'standard_recipe_check.json', 'power_samples.jsonl': C/'power_samples.jsonl',
    'runtime_environment.json': C/'runtime_environment.json',
    'watch.py': C/'watch.py', 'postprocess_measurement.py': C/'postprocess_measurement.py',
    'finalize.py': Path(__file__),
})
for name, path in files.items():
    assert path.is_file() and path.stat().st_size < 5_000_000
    shutil.copy2(path, E/name)
evidence = dict(run=str(RUN), scope='Only lightweight final-run evidence; raw models/packets/images remain in output.',
    files={p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(E.iterdir())})
(E/'evidence_manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
table = '\n'.join(f"| {r['first_frame']}–{r['last_frame']} | {float(r['psnr']):.4f} | {float(r['dynamic_complete_kb']):.3f} | {float(r['pure_render_fps']):.1f} | {float(r['train_wall_seconds']):.2f} |" for r in segments)
comparison = '\n'.join(f"| {label} | {previous[key]:.{precision}f} | {S[key]:.{precision}f} |" for key, label, precision in (
    ('psnr','cam00 PSNR / dB',6), ('ssim','SSIM',6), ('lpips','LPIPS（VGG）',6),
    ('dynamic_complete_kb','动态完整更新 / KB',6), ('aggregate_materialized_fps','纯渲染 FPS',3),
    ('mean_train_wall_seconds','训练与评估秒/动态帧',4)))
components = '\n'.join(f"| {label} | {S['storage_components_mean_kb'][key]:.3f} | {100*S['storage_components_mean_kb'][key]/S['dynamic_complete_kb']:.2f}% |" for key,label in (
    ('motion_bytes','运动包'),('opacity_bytes','旧点透明度'),('feature_bytes','区域 DC'),('birth_bytes','新增属性'),('manifest_bytes','manifest')))
target_text = f"PSNR ≥34.2 dB：{'达到' if S['goals']['psnr_at_least_34_2'] else '未达到'}；纯渲染 FPS >200：{'达到' if S['goals']['pure_render_fps_above_200'] else '未达到'}；动态完整更新严格 ≤50 KB：{'达到' if S['goals']['dynamic_complete_at_most_50kb'] else '未达到'}。"
text = f'''# Flame steak：480 W 标准配置，最终完整 300 帧

作者指定本次为 flame_steak 最终结果。恢复显卡默认 480 W 后，用固定标准半分辨率配方，从新训练的首帧开始完整运行第 0–299 帧，没有使用动态前缀，也没有根据中间结果调整参数。

最终 **{S['psnr']:.6f} dB、{S['dynamic_complete_kb']:.6f} KB/动态帧、{S['aggregate_materialized_fps']:.3f} 纯渲染 FPS**。具体阈值是否达到，以[完整核验](evidence/flame_steak_final480w_20261019/final_audit.json)的 target_checks 为准，不用此前 350 W 的结果替代本次指标。

{target_text}

## 配方与数据

配置 [flame_steak_half_final480w.yaml](../../config/n3dv/flame_steak_half_final480w.yaml)。与[上一轮](20261019_half_byte_budget_full300.md) module/trainer/data 字典完全一致，仅改实验名。每字节代价 `3e-5/50000=6e-10`，50000 是损失单位，不是硬码流上限。learnable gate 仍使用随机离散前向与 STE，最大深度 4、robust_median、远点连最近区域，父子运动相对表示，旋转更新全部关闭；透明度量化、近端稀疏与低透明度删除保持开启。本轮未改方法代码。

1345×1008，21 相机，每相机 300 帧，cam00 留出，其余 20 相机训练。6300 张图像保持软链接，全部 PNG 头尺寸及链接检查通过，处理来源和像素检查见[预处理审计](20261018_n3dv_preprocessing_audit.md)。原始种子仍为相同 40000 点世界坐标 PLY；本次不重新进行 SfM/MVS。

首帧直接加载此前这轮重新训练的半分辨率 15000 步 checkpoint；本次未重复 native 首帧训练，也未使用历史 1600 模型或 17000 步精修模型；SHA256 `61f38b3ff4ffc40655b1bad75834ce92c1540537962f0a65438b5a2646b4384e`。初始保留 {S['initial_gaussians']} GS，参考包与上一轮逐字节一致。480 W 的变化不能单独解释两次所有质量差异：GPU 优化有非确定性，上一轮还存在经验证的动态前缀续跑。

每动态帧 300 步：1–20 共享区域/DC 预热，21–70 区域与 gate 学习，71–100 固定 gate 后的区域/逐点稀疏联合拟合，101–200 固定非零支持精修幅值，201–300 普通新增及区域 DC 更新。前 200 步每步 2 视角，后 100 步每步 1 视角。

## 完整结果

| 指标 | 此前 350 W 同配方 | 本次最终 480 W |
| --- | ---: | ---: |
{comparison}

质量均值覆盖 cam00 第 0–299 帧，动态体积与时间覆盖第 1–299 帧。KB=1000B。完整更新包含运动、新增、透明度、区域 DC 和必要 manifest，首帧参考另计。中位数 {audit['complete_update_median_kb']:.3f} KB，P95 {audit['complete_update_p95_kb']:.3f} KB，最大 {audit['complete_update_max_kb']:.3f} KB，{audit['frames_at_most_50kb']}/299 帧不超过 50 KB。首帧参考 {S['shared_reference_bytes']/1e6:.6f} MB，整段完整码流 {S['total_stream_bytes']/1e6:.6f} MB，摊入首帧后 {audit['reference_amortized_kb_per_frame']:.6f} KB/帧。

| 成分 | 平均 KB/动态帧 | 占比 |
| --- | ---: | ---: |
{components}

{S['frames_with_effective_subdivisions']}/299 个动态帧存在有效细分，平均 {S['mean_regions']:.2f} 个选中区域，原始 gate 开启位平均 {S['mean_open_gates']:.2f}。按旧点数加权，{100*S['old_gs_without_local_residual_fraction']:.3f}% 无逐点平移残差，包含静止点，不能把它当作 gate 单独贡献的因果比例。所有 GS 旋转残差块为零。最终 {S['final_gaussians']} GS，平均新增 {S['mean_new_gaussians']:.2f}、删除 {S['mean_retired_gaussians']:.2f} 个/帧。

最低 PSNR：第 {worst['frame']} 帧 {float(worst['psnr']):.6f} dB；最高：第 {best['frame']} 帧 {float(best['psnr']):.6f} dB。本次是固定配方最终复跑，不是功率消融或新的机制归因实验。

## 独立核验与计时范围

连续解码全部 299 个动态更新并独立重渲染，packet/manifest 摘要通过，PSNR/SSIM/L1 与保存值最大差为 0，缓存逐像素差为 0。第 0 帧 LPIPS 独立复算；动态 LPIPS 使用训练终态 decoded-state 评估，未独立复算全部动态 LPIPS。

每帧预热 10 次、计时 50 次完整 `Module.forward(cam00)+clamp`，15000 次渲染除以总秒数得到 FPS，GPU 独占检查通过，测量前后功率上限均为 480 W。**纯渲染 FPS 排除解码、上传、图像 I/O 与质量计算**。CPU 解码/校验 {S['mean_decode_verify_seconds']:.4f} 秒/动态帧，上传 {S['mean_upload_seconds']:.6f} 秒。

训练墙钟包含数据准备、训练与评估，终态 checkpoint 写入在计时外；首动态帧 {S['first_dynamic_seconds']:.2f} 秒，排除该帧后平均 {S['mean_train_wall_seconds_after_first_dynamic']:.4f} 秒/动态帧。固定首帧原训练耗时另计。训练期间 {len(power)} 次功率抽查上限均为 480 W，抽样最高功耗 {audit['sampled_peak_power_watts']:.2f} W、最高温度 {audit['sampled_peak_temperature_c']:.0f}°C，抽样峰值不代表全部瞬间最大值。

冻结源码 {len(manifest['hashes'])} 个文件摘要全部通过，基于提交 `{manifest['commit']}` 加已记录的工作区改动；当前 {len(training)} 个训练/extension 源文件与快照一致。

## 每 30 帧

| 帧范围 | PSNR / dB | 动态完整 KB | 纯渲染 FPS | 训练评估秒/动态帧 |
| --- | ---: | ---: | ---: | ---: |
{table}

第一段质量包含第 0 帧，动态存储与时间仅覆盖第 1–29 帧。

![最终复跑与此前350W同配方比较](evidence/flame_steak_final480w_20261019/metrics_comparison.png)

## 输出与复现

原始输出：`{RUN.relative_to(ROOT)}`。统计：`{M.relative_to(ROOT)}`。

```bash
conda activate ReConGS
python main.py --config=config/n3dv/flame_steak_half_final480w.yaml
```

[汇总](evidence/flame_steak_final480w_20261019/summary.json) · [逐帧CSV](evidence/flame_steak_final480w_20261019/per_frame_metrics.csv) · [每30帧CSV](evidence/flame_steak_final480w_20261019/per_30_frames.csv) · [顺序解码](evidence/flame_steak_final480w_20261019/decoded_sequence.json) · [完整核验](evidence/flame_steak_final480w_20261019/final_audit.json) · [文件摘要](evidence/flame_steak_final480w_20261019/evidence_manifest.json)。原始模型和训练图片保留在本地 output，未加入 Git。其他五景的完整动态训练尚未启动，不能将本次指标当作 N3DV 六景平均。
'''
DOC.write_text(text)
pointer = dict(status='final_verified', designated_by='user', scene='flame_steak', date='2026-10-19',
    run=str(RUN), config=str(ROOT/'config/n3dv/flame_steak_half_final480w.yaml'),
    document=str(DOC), summary=str(M/'summary.json'), metrics=S,
    audit_sha256=sha(M/'final_audit.json'))
(C/'final_result.json').write_text(json.dumps(pointer, indent=2)+'\n')
print(json.dumps({'status':'final_verified','document':str(DOC),'run':str(RUN),
    'psnr':S['psnr'],'complete_update_kb':S['dynamic_complete_kb'],'fps':S['aggregate_materialized_fps']},indent=2))
