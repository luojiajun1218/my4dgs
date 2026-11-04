# Cook Spinach：2026-10-29 至 31 日前30帧实验

## Cook Spinach 前 30 帧：运动支持上的稀疏各向同性尺度流

日期：2026-10-30

### 核心想法

区域刚体运动和逐点位置残差只能修改 GS 位姿，不能修正随动态形变变化的协方差。此前对 Cook Spinach 第 13、16 帧的诊断显示，允许尺度更新能明显改善 held-out `cam00`，但无约束逐点尺度流会将单帧存储推到约 95KB。

本实验只在最终逐点平移残差的支持集内学习一个各向同性 log-scale 标量，每帧最多选择 1024 个旧 GS：

- 第 201–280 步在运动支持上联合优化，并用 Adam 二阶矩估计潜在图像收益；
- 第 280 步硬选最多 1024 行，第 281–300 步固定支持重拟合幅值；
- 使用步长 0.005 的有符号 int8 符号，每行只传一个各向同性值；
- 尺度包在出生包之前应用，因此新生 GS 的预测参考与训练态一致；
- 包含参考尺度和运动支持摘要、CRC、严格行号及 int8 边界检查。

这不是任意学习新的空间边界；尺度流复用已经传输的逐点运动支持，额外结构开销受到硬上限约束。

### 配置与代码改动

- 配置：`config/cook_spinach_scale1024_30_20261030.yaml`
- 结果：`output/goal_n3dv_dynamic_20261030/cook_spinach_scale1024_30_20261030/20261030T062332_c6765f`
- 对照：`output/goal_n3dv_dynamic_20261030/cook_spinach_warm3_regionthird_regdc_30_20261030/20261030T050727_2c88ec`

主要实现：

- `pipeline/model/sparse_logscale.py`：训练、硬支持选择、int8 量化和 `SLS1/SLZ1` 精确码流；
- `pipeline/frame_plan.py`、`pipeline/module.py`：在第 200 步清理后建立支持，接入联合优化、出生参考、checkpoint 和帧报告；
- `pipeline/model/rigid_tree_codec.py`：解码器暴露本帧逐点平移支持，但不改变连续状态摘要；
- `pipeline/model/stream_packet.py`、`pipeline/model/gpu_stream.py`、`scripts/replay_rigid_tree.py`：CPU/GPU 均在出生前应用尺度流；
- `scripts/benchmark_rigid_residual_tree.py`：把尺度包计入完整动态存储；
- `tests/test_sparse_features.py`、`tests/test_gpu_stream.py`：覆盖摘要绑定、损坏包、硬预算、checkpoint、出生继承和 CPU/GPU 精确一致性。

### 结果

统计口径为动态帧 1–29；KB 使用十进制字节，且完整存储包含每帧必要 manifest。

| 方法 | PSNR | SSIM | LPIPS | 完整存储/帧 |
|---|---:|---:|---:|---:|
| 旧机制：region-rate 1/3 + regional DC | 33.6156 dB | 0.9622 | 0.1239 | 49.26 KB |
| 本实验：再加稀疏尺度流 | **33.7275 dB** | **0.9623** | **0.1233** | 49.68 KB |
| 差值 | **+0.1119 dB** | +0.0001 | -0.0006 | +0.42 KB |

完整 30 帧（含共享首帧）平均 PSNR 为 33.7313dB。动态包总计 1,440,855B，共享参考 13,112,076B，参考加完整动态流总计 14,552,931B。

逐帧差分中 20/29 帧提升、9/29 帧下降。最大提升为第 16 帧 +0.5020dB，最大下降为第 2 帧 -0.2112dB。分段结果：

- 第 1–8 帧：平均 +0.0262dB，+0.78KB/帧；
- 第 9–16 帧：平均 **+0.3088dB**，+0.53KB/帧；
- 第 17–29 帧：平均 +0.0436dB，+0.14KB/帧。

尺度包平均 2.291KB、1010 个非零行。与此同时，运动、出生、透明度包相对旧 run 平均分别减少约 1.42、0.23、0.31KB；所以完整码率只增加 0.42KB，而不是直接增加 2.29KB。这说明 rate controller 能部分吸收新流，但也意味着不同 run 的门控/支持轨迹并不完全相同。

本地 `sota.png` 中 S²GS-full 的 Cook Spinach 为 33.96dB / 0.958 SSIM。本实验 PSNR 仍低约 0.23dB，尚未达到该场景 SOTA；SSIM 高约 0.004。不能据此声称整体超过 SOTA。

### 正确性与速度

- CPU 从共享参考逐帧重放到第 29 帧，得到 143,323 个 GS，decoder SHA256 与 manifest 完全一致；
- 最终 checkpoint 与码流在 `xyz/rotation/logscale/opacity/sh_dc/sh_rest` 上逐项完全相等；
- checkpoint 与码流渲染像素最大绝对差为 0，逐帧重放画质与训练保存指标差为 0；
- 动态帧已解码渲染 331.94 FPS，完整 30 帧聚合 331.40 FPS，超过 200 FPS 验收线；
- 包含读文件、校验、CPU 解码、上传、更新和渲染的完整播放为 11.26 FPS，单独列出，未用渲染 FPS 代替；
- 证据：`tmp/cook_spinach_scale1024_first30_20261030/sequence_validation.json` 与 `final_checkpoint_validation.json`。

完整 `unittest` 共运行 691 项，其中 690 项通过；唯一错误是历史测试仍引用工作区中已删除的 `config/fs_sd4_300.yaml`，与本次尺度机制无关。新增尺度 codec、训练调度、CPU/GPU 流测试均通过。

### 结论边界与下一步

已验证结论：在 Cook Spinach 前 30 帧，受运动支持约束的 1024 行各向同性尺度流改善了画质—存储折中，尤其稳定改善第 9–16 帧的困难段，并保持精确解码与目标渲染速度。

尚未验证：总提升不能全部归因于尺度流，因为 CUDA/门控轨迹存在非确定性；这次结果也不能证明所有 N3DV 场景或完整 300 帧都会提升，更不能证明自适应层级本身已成立。第 2、3、6、15、17、20、23、26、29 帧仍有局部下降。

下一步优先诊断启用尺度流后仍残留的协方差误差：比较同一运动支持上的各向异性尺度与小旋转残差的边际收益，继续使用硬行预算和精确码流。若边际收益不足，再考虑把后半段低于 50KB 的预算余量更主动地分配给困难帧，而不是盲目增加尺度行数。

---

## Cook Spinach：区域选择码率梯度折扣的前 30 帧结果

本轮针对 Cook Spinach 在高码率 multiplier 下空间 gate 过度合并的问题，将“选择更多区域”的码率梯度与逐点残差码率压力分开。正式 frame0–29 的平均 PSNR 为 33.623064 dB，动态完整更新为 49.259690 KB/帧。相对旧 warm1 基线，动态 PSNR 提高 0.022475 dB，同时完整更新减少 6.613931 KB/帧（−11.84%）。因此本轮是有效的率失真改进，主要价值是画质不降时显著降低存储；它仍低于本地 `sota.png` 中 S²GS-Full 的 Cook Spinach 33.96 dB，不能宣称达到 SOTA。

### 核心想法与实现

旧 warm3 首 7 帧把平均空间区域从 warm1 的 419.9 压到 131.6；frame6 的 motion-only 画质已经比 warm1 低约 0.54 dB，说明损失发生在运动阶段，而不是后续 birth 或区域 DC。进一步检查代码后确认，逐点 residual 的 proximal 决策已经使用 Adam 二阶量度，并不是对所有点等价删减；更直接的问题是同一个高全局 multiplier 也强烈惩罚区域细分，使共享运动场过粗。

本轮在 `SpatialGateConfig` 新增 `region_selection_rate_gradient_scale`，默认值为 1，保证旧配置兼容。训练时只缩放区域 selection weight 到 gate 的码率梯度：

- 前向 `rate_bytes_surrogate` 仍是完整、真实的区域字节代理，实际 packet 与闭环反馈没有打折；
- 逐点 residual 仍承受完整 warm3 码率压力；
- 区域运动参数本身仍按完整区域码率训练，只有“是否继续细分”的梯度被折扣；
- 主 `loss_terms` 与备用 `proximal_rd` 路径使用相同语义，并输出诊断字段；
- 配置校验限制 scale 在 `(0, 1]`。

Cook 使用 scale=`1/3`，其依据是抵消 initial multiplier=3 对 gate 的额外初始压力，而不是继续做参数网格。完整相关回归测试为 266 项通过；新增测试确认前向字节逐位不变、gate 的区域码率梯度按比例缩放，并拒绝非法 scale。

- 配置：cook_spinach_warm3_regionthird_regdc_30_20261030.yaml（原路径 `../../config/cook_spinach_warm3_regionthird_regdc_30_20261030.yaml` 已不在工作区）
- 新运行：`output/goal_n3dv_dynamic_20261030/cook_spinach_warm3_regionthird_regdc_30_20261030/20261030T050727_2c88ec`
- 旧 warm1 基线：`output/cook_spinach_20261028/cook_spinach_cov150_sh_30_resume_f2_20261029/20261029T125521_993060`
- 被提前停止的旧 warm3：`output/goal_n3dv_dynamic_20261030/cook_spinach_cov150_warm3_regdc_30_20261030/20261030T044532_c8f3af`

正式新旧运行使用相同 150k 首帧、相机、分辨率、随机种子和 200+100 动态训练步。相对 warm1 的变化同时包含 warm-start rate、区域 DC 和本轮 gate 折扣，因此不能把完整 29 帧的总差值全部归因于本轮单一机制；下面另列同为 warm3 的前 7 帧配对证据。

### 正式结果

| 指标 | 旧 warm1 | 本轮 | 变化 |
|---|---:|---:|---:|
| frame0–29 平均 PSNR / dB | 33.601338 | 33.623064 | +0.021726 |
| 动态 frame1–29 PSNR / dB | 33.593092 | 33.615567 | +0.022475 |
| 动态 SSIM | 0.962235 | 0.962224 | −0.000011 |
| 动态 LPIPS | 0.123402 | 0.123533 | +0.000131 |
| 完整动态更新 / KB 每帧 | 55.873621 | 49.259690 | −6.613931（−11.84%） |
| 动态总更新 / bytes | 1,620,335 | 1,428,531 | −191,804 |
| 训练时间 / s 每帧 | 46.2111 | 55.0142 | +19.05% |
| 最终 GS 数 | 144,519 | 143,237 | −1,282 |

KB=1000 bytes。完整更新包含 motion、birth、opacity、区域 DC 和必要的逐帧 manifest。共享首帧为 13,112,076 bytes；首帧加全部完整动态更新为 14,540,607 bytes。

29 个动态帧中 22 帧不超过 50 KB，27 帧不超过 55 KB。线性插值 P95 为 55.387 KB，最大值 92.034 KB，最大值来自控制器尚未收到历史反馈的 frame1。工程目标是序列平均约 50 KB，不表示每一帧都必须低于 50 KB。

| 时间段 | 旧 PSNR / dB | 本轮 PSNR / dB | 旧 KB/帧 | 本轮 KB/帧 |
|---|---:|---:|---:|---:|
| frame1–10 | 33.697208 | 33.799702 | 67.4700 | 54.5894 |
| frame11–19 | 33.535889 | 33.481266 | 51.3414 | 46.6742 |
| frame20–29 | 33.540460 | 33.552304 | 48.3562 | 46.2569 |

前 10 帧同时提高 0.1025 dB 并减少 12.88 KB/帧；中间 9 帧减少 4.67 KB/帧但损失 0.0546 dB；末 10 帧提高 0.0118 dB 并减少 2.10 KB/帧。整体收益不是每帧一致：29 帧中 16 帧 PSNR 更高，最明显回退是 frame13 的 −0.3821 dB（同时少 6.195 KB）和 frame16 的 −0.2917 dB（少 1.608 KB）；最好的是 frame2 的 +0.4269 dB（少 20.943 KB）。因此结论应是平均率失真改善，而不是所有局部时刻都改善。

### 同 warm3 的机制证据

旧 warm3 在确认 frame6/7 明显失真后于 frame8 训练中提前停止，因而只能严格比较正式运行的 frame1–7。两者总帧数均为 30、随机种子与所有其他配置相同；配对差异只有区域选择梯度折扣：

| frame1–7 平均 | 旧 warm3 | 本轮 | 变化 |
|---|---:|---:|---:|
| PSNR / dB | 33.668771 | 33.982049 | +0.313278 |
| 完整更新 / KB | 57.276857 | 56.953714 | −0.323143 |
| 激活区域 | 131.57 | 145.86 | +14.29 |
| 逐点 residual 行 | 19,658.7 | 19,476.1 | −182.6 |
| step200 motion-only PSNR / dB | 33.610463 | 33.840777 | +0.230314 |

该配对结果与假设一致：区域稍细后，逐点 residual 不增反降，画质和总字节同时改善。特别是旧 warm3 的 frame6 为 32.8299 dB / 51.981 KB，本轮为 33.7376 dB / 51.506 KB。它为“高全局 multiplier 误伤空间共享粒度”提供直接证据，但证据范围仍只覆盖 Cook 的前 7 个动态帧，不能外推为全部场景、全部帧已经验证。

### 码流与结构分析

| 完整更新分项 | 旧 warm1 KB/帧 | 本轮 KB/帧 | 变化 |
|---|---:|---:|---:|
| motion | 43.6661 | 38.0947 | −5.5714 |
| birth | 5.0333 | 3.7587 | −1.2746 |
| opacity | 6.0535 | 5.9068 | −0.1467 |
| 区域 DC | 0 | 0.2929 | +0.2929 |
| manifest | 1.1207 | 1.2067 | +0.0860 |

相对 warm1，本轮平均激活区域从 231.5 降到 107.4，逐点 residual 行从 19,298.5 降到 15,682.1，每帧 birth 从 236.7 降到 191.8，retire 基本不变（425.7 对 425.0）。这里包含 warm3 和区域 DC 的共同作用，不能仅凭区域数下降就反向否定 gate 折扣；判断折扣本身应使用上面的同 warm3 前 7 帧配对。

最终完整帧比 step200 motion-only 诊断平均高 0.08752 dB，29 帧中 19 帧为正。该阶段混合了 birth、retire、opacity、区域 DC 和量化提交，不能把这 0.08752 dB 单独归因给 birth 或 DC。

### 解码与速度核验

独立 CPU 从 frame0 reference 顺序解码全部 29 个更新，逐帧 GPU 渲染与保存指标一致；GPU 排他性检查通过。末帧 checkpoint 与码流的 `xyz/rotation/logscale/opacity/sh_dc/sh_rest` 六类原始属性逐位一致，checkpoint、解码码流和缓存/非缓存渲染的像素最大差均为 0。

- 30 帧已解码材质化渲染：334.635 FPS；动态 29 帧为 335.301 FPS，满足 ≥200 FPS 的论文口径目标。
- 末帧单独材质化渲染：342.608 FPS。
- 完整播放：11.605 FPS。该值覆盖热缓存本地文件读取、CPU 解码、host-to-GPU 更新和实际渲染，共两轮、每轮 29 次更新；一次性 reference、相机/GT 加载和 manifest 预检不计入。它必须与纯渲染 FPS 分列。
- 核验：sequence_validation.json（原路径 `../../tmp/cook_spinach_warm3_regionthird_regdc_first30_20261030/sequence_validation.json` 已不在工作区） 与 final_checkpoint_validation.json（原路径 `../../tmp/cook_spinach_warm3_regionthird_regdc_first30_20261030/final_checkpoint_validation.json` 已不在工作区）。

### 与 SOTA 的边界及下一步

本地 `sota.png` 中 S²GS-Full 的 Cook Spinach 为 33.96 dB。本轮 frame0–29 平均为 33.6231 dB，仍低 0.3369 dB，而且只训练前 30 帧，协议范围也未必与论文全序列完全一致。因此当前只验证了 Cook 前 30 帧在约 50 KB/帧下优于本仓库旧基线的率失真表现；尚未达到 >34 dB 的工程画质目标，更不能据此声称 N3DV 整体超过 SOTA。

结果还暴露出逐帧同目标预算的问题：困难帧 11–13 画质回退，而若干后段帧只使用约 42–46 KB。下一步的研究假设是，在序列平均仍约 50 KB 的前提下，用只依赖训练侧信号的累计预算或难度估计，把未使用的码率分配给困难帧，可能比让每帧独立追同一 50 KB 更好。该方向尚未实现或验证，不能写成结论。

另一个工程观察是 8 帧 smoke 与 30 帧正式运行即使 seed 相同，前缀轨迹仍有差异。代码检查确认 frame-plan 相机采样和 gate 噪声已经使用独立、按 frame/step 绑定的生成器，因此目前不能把差异直接归因于数据集长度改变 RNG 消耗；CUDA 渲染反向中的非确定性或其他初始化路径仍可能放大离散 gate 分支。正式比较始终使用同为 30 帧的配对运行，短实验与正式运行的确定性来源需要后续单独定位。

## 10月29日：150k首帧与恢复前缀

150,000点SH1首帧、半分辨率、cam00评估、区域运动与量化出生/透明度更新，frame1–29为 **33.5931 dB / 55.87 KB**，首帧33.8405 dB。前5帧平均75.38 KB，后10帧降至48.36 KB；运动包为主要开销，码率反馈收敛较慢。完整恢复链、29帧码流字节及最终checkpoint/参考属性核验通过；未测渲染或播放FPS。运行：`output/cook_spinach_20261028/cook_spinach_cov150_sh_30_resume_f2_20261029/20261029T125521_993060`。

## 10月31日：两阶段 exact15k 首帧

首帧前14,000步训练全部属性，最后1,000步只校准SH1颜色，几何和透明度冻结；动态入口152,557点。动态沿用warm3、gate价格1/3、区域DC及最多1024行的运动支持尺度流。frame1–29为 **33.913114 dB / 49.351793 KB**，含首帧30帧为33.918133 dB；相对旧scale1024的33.7275 dB / 49.68 KB，整套方案画质提高约0.1856 dB，但首帧及layout也变了，不能仅归因于颜色校准。

29帧连续CPU解码、30帧GPU重渲染和checkpoint六属性逐位核验通过。已解码渲染306.755 FPS，包含读包与解码的完整播放11.046 FPS。首帧Stage A/B及layout配置仍在 `config/cook_spinach_frame0_cap150k_14k_stagea_20261031.yaml`、`cook_spinach_frame0_cap150k_shcal1k_stageb_20261031.yaml`、`cook_spinach_twostage_cap150_layout_20261031.yaml`；动态运行：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cook_spinach_twostage_cap150_gatepricefix_regdc_scale1024_30_20261031/20261031T015149_9df4f1`。这只是30帧，不能代替完整300帧或证明整体超过SOTA。
