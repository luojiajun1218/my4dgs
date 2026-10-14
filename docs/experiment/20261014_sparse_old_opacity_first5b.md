# 稀疏旧opacity：首5帧完成记录

## Idea与改动

旧GS只有位姿变化时，消失表面可能继续留下重影。本轮仅在201–300步birth阶段增加旧opacity logit的双向量化增量（step=.01、Adam lr=.05），复用运动的归一化存储价格和Adam稀疏prox；不增加SH/scale更新，不叠改运动/区域/tree规则。OPD1包只传非零旧行，零更新没有文件。birth预测器使用更新前外观，之后才应用旧行opacity；新生点本帧不受旧行增量影响，下一帧成为普通旧行。

新增sparse_opacity.py，module/checkpoint/最终物化接线；replay/benchmark按motion→birth→旧opacity→全raw hash顺序解码，完整计费包含opacity实文件。新增实流与CPU实模块回归，覆盖250双Adam恢复、300最终状态、下一帧参考、birth0、step200排未训练clone与随机状态。

首轮 output/fs_persistent_sparseopacity_first5_20261014/20261014T023228_3f6dd2 在阶段诊断直接访问test dataset时因主进程get_worker_info为空退出1（20.72s），完成动态帧0、未到opacity优化；这是启动接口故障，不是方法效果。改为现有test_dataloader worker加载，并通过真实worker/library warning及标准数据cam00验证后，使用新experiment_name重试。

- 配置：config/fs_persistent_sparseopacity_first5b_20261014.yaml
- Run：output/fs_persistent_sparseopacity_first5b_20261014/20261014T023654_126845
- 命令：激活ReConGS后 python main.py --config=config/fs_persistent_sparseopacity_first5b_20261014.yaml
- 实际exit0，wall163.51s；完成共享frame0及动态1–4，source_snapshot保存34文件，运行中实现冻结。

## 独立解码、质量、完整字节与速度

| 指标 | 实测 |
|---|---:|
| 全frame0–4 PSNR | **34.22474 dB** |
| 动态frame1–4 PSNR | 34.27810 dB |
| 动态完整更新平均 | **74311.50 B** |
| motion / birth / opacity / manifest平均 | 68687.00 /3452.25 /1697.25 /475.00 B |
| 全0–4纯完整渲染，10预热+50次/帧 | **202.132 FPS** |
| 动态纯渲染 | 202.438 FPS |
| frame4物化渲染，5预热+30次×3 | 200.940 FPS |
| 2轮顺序文件读取/校验解码/上传/渲染 | 4.583 FPS |

全部4帧motion/birth/opacity和最终raw hash、saved质量一致，cache开关图像逐位一致。完整动态合计297246 B，payload295346 B；共享reference16,656,569 B另计一次。短前缀质量/FPS达数值门槛，但完整字节未到50000 B，更没有全300验收。

## 同一heldout cam00的阶段变化

| 帧 | 100步建区后PSNR | 200步joint后，仅旧GS | 300步birth+opacity后 | 200→300 dB | +opacity/-opacity行 | births | opacity包B | 完整B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|34.37720|34.40183|34.29924|-0.10259|583 /313|139|2560|114475|
|2|34.23931|34.25097|34.05230|-0.19867|233 /186|162|1337|72794|
|3|34.15343|34.21228|34.45393|+0.24166|255 /249|184|1599|56002|
|4|34.35790|34.35947|34.30692|-0.05254|171 /236|228|1293|53975|

100→200四帧均微增；200→300方向混合，平均-0.02804 dB。**该阶段同时优化birth和旧opacity，不能把阶段差全部归到opacity单项。** opacity最终active896/419/504/407行，约旧GS的0.43%/0.20%/0.24%/0.20%，正更新合计1242、负更新984；有双向更新，但不等于本轮已证明质量改善。

final_train_metrics PSNR为38.52/38.48/38.25/32.01，来自最后采样训练视角、优化/末尾剪枝前；与固定cam00并非同视角同阶段，不能仅据此声称训练改善而测试下降或证明过拟合。背景/表面alpha漂移及视角泛化仍是原因候选，需长时阶段轨迹判断。区域slots9/10/12/14，未实际采用内部parent；暂不增加新的区域维护改动或消融。

下一步按同一机制推进30帧，检查较迟消失表面的累计影响，而不由这4帧保证长序列收益。

## 新产物

- 全5帧重放/FPS与真实文件计费：output/persistent_sparseopacity_first5b_benchmarks_20261014/sequence.json
- frame4物化/顺序播放：output/persistent_sparseopacity_first5b_benchmarks_20261014/frame4_playback.json
- 整数残差与实际motion/birth/opacity/index字节复核：output/persistent_sparseopacity_first5b_analysis_20261014/packet_analysis.json
