# V15：提前接管原生支撑，单轨迹 300 步，完整前 10 帧

本轮已完成 frame0 + 9 个动态帧训练与独立 stream 重解码。动态平均 PSNR **34.127969 dB**，全部 10 帧平均 **34.036958 dB**；完整动态更新平均 **43.065 KB**，全 10 帧渲染 **272.11 FPS**。画质尚未达到 34.2 dB，且同姿态接管收益平均为 **−0.063378 dB**，因此不扩展长序列。

## 本轮机制

本轮检验当前帧新增是否获得真实可见支撑与充分的属性拟合机会。没有使用双分支、SfM 候选、跨帧持久池或额外 SGD/refit。

1. 前 70 步保持原运动/gate 机制。gate 冻结后，从全部 20 个训练相机的原生 `Σ Tα_i × RGB residual²` 责任选择 128 个旧源。
2. 每源建立一个可训练的 1→1 接管行，复制实际六字段与真实 alpha；训练时在旧行原位置覆盖，保持原生排序。所有相机初态图像逐像素 bitwise 相同。旧父只作回滚载体，场景中始终显示一份覆盖。
3. 71–200 步学习局部 XYZ/旋转、尺度、DC、SH1，与旧运动共享同一次照片损失、同一个 Adam。新 alpha 保持源第 70 步的实际值；原父 fallback OPD 值及其 moments 被保护，不能因失去照片梯度而被 rate/prox 掏空。新世界 pose 始终绑定 live source motion，梯度可回传到源运动。
4. 第 200 步按真实 codec、全部 20 相机 MSE 与字节选择接管/死亡操作，固定实际人口、surviving source 与量化参考；切片保留 Adam moments/step。之后 100 步只优化这个人口，同时释放新 alpha。
5. 第 300 步仅提交该轨迹或完整回滚已验收的第 200 步快照；不再重选另一子集、不添加训练轮。f8 实际触发了这一回滚。

这是旧支撑的独立属性接管，**1→1 不增加净表示容量**，不能自动补出原支撑缺少的空间/多色细节。

主要实现：`pipeline/early_replacement.py`、`pipeline/model/early_replacement_model.py`；`module.py`/`trainer.py` 接入共享 Adam、每步全 20 视角累积与阶段路由；`birth_quantization.py` 加 ordered live overlay 与物理 alpha/属性导出；`sequence_extension.py` 校验实际调度账本。沿用 SPC/RTB1/ATTR 与真实计费，没有新增推理期隐状态。

## 固定条件与执行证据

- 使用既有共享 132586 GS 首帧；沿用既有 opacity 过滤，实际 decoder reference 为 128963 GS。首帧没有另训或替换。
- 监督分辨率 1345×1008，全部 20 个标准训练相机；cam00 只用于评估。gate 参数与冻结时机保持原机制。
- 一个模型、一个共享 Adam；每动态帧恰好 **300 次 global 更新**。motion 参与 200，newborn 材质/局部几何参与 230，其中共享 130；`200 + 230 − 130 = 300`。新 alpha 参与最后 100 次；没有额外 SGD。71–300 每次更新前逐视角 fresh graph 累积全部 20 相机梯度，只执行一次 Adam step。
- 实际 9 帧的第 70 步初始化均通过全 20 相机像素 bitwise identity；第 200 步六字段 codec 重放全部 exact。新模型 7 项 CPU、实际 Trainer/Module 的完整 300 步控制器 fixture、相关失败组及 prefix 检查通过。原生证明 `tmp/birth_diagnosis/early_v15_native_validation.json` 另证实源 motion/new 参数真梯度、alpha 冻结无 state、alpha=1 释放后可下降及 ATTR bitexact。

## 实际重解码结果

KB 按 1000 bytes。完整更新含 motion、birth、old-opacity 包和必要 frame manifest，均实测文件大小。下表 baseline 是第 200 步同运动姿态下恢复旧源覆盖的反事实场景，不是另训 motion-only 对照；共同运动本身也受接管学习影响。

| 动态帧 | 同姿态旧场景 PSNR | 最终 stream PSNR | 差值 dB | 接管数 | 完整 KB | 最后100步 |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | 33.905144 | 33.971840 | +0.066696 | 107 | 115.734 | 提交 |
| 2 | 34.280083 | 33.712959 | −0.567123 | 103 | 42.188 | 提交 |
| 3 | 34.347767 | 34.102135 | −0.245632 | 89 | 36.889 | 提交 |
| 4 | 34.193832 | 34.048271 | −0.145561 | 85 | 34.992 | 提交 |
| 5 | 33.900421 | 34.261093 | +0.360672 | 69 | 32.299 | 提交 |
| 6 | 34.224941 | 34.312225 | +0.087284 | 67 | 30.799 | 提交 |
| 7 | 34.313843 | 34.313240 | −0.000603 | 80 | 30.482 | 提交 |
| 8 | 34.373592 | 34.391327 | +0.017735 | 80 | 33.653 | 完整回滚200 |
| 9 | 34.182495 | 34.038628 | −0.143867 | 86 | 30.549 | 提交 |

- 动态 9 帧：PSNR **34.127968682**，SSIM 0.969680117，渲染 **270.537791 FPS**。
- 含共享 frame0 全 10 帧：PSNR **34.036957550**，SSIM 0.969470310，渲染 **272.110635 FPS**。frame0 PSNR 33.217857361。
- 动态 payload 平均 **42.097 KB**；完整更新平均 **43.065 KB**，总计 387585 bytes。第一包完整 115734 bytes，后续较小，不能把第一包峰值当平均值。
- 共享首帧 reference **11081043 bytes** 单独列出；含 reference 与全部动态完整更新总计 **11468628 bytes**。上述 43.065 KB 不包含这个一次性参考。
- 所有动态帧 stream PSNR/SSIM/L1 与保存值差均 **0**，cache/uncached 像素最大差均 **0**。计时采用每帧 10 warmup + 50 CUDA 同步完整 forward+clamp，排除文件解码、上传与图像读取。
- 每动态帧平均总耗时约 **300.28 s**；第 200 步逐组完整 codec/图像验收耗时明显，额外时间不是额外 SGD。

## 当前帧拟合是否有效

128 个源实际承担第 70 步旧图像总 RGB SSE 的 **14.20%–17.12%**，不是投影 bbox 面积或极少数不可见小点。固定源第 70 步的原生 Tα 图作为权重，记录其支撑区域 GT MSE；各相机归一后对 20 视角等权平均。该曲线包含共同运动与接管属性的联合收益，不能把全部下降单独归因于新参数。

支撑区域 GT MSE 在 9 条尝试轨迹分别下降 **18.74%/23.49%/6.62%/3.14%/6.78%/4.85%/6.38%/5.16%/12.69%**。例如 f2 从 3.148541e−4 降至 2.409055e−4；实际全 20 codec MSE 也从旧场景 2.373774e−4 降至最终 2.307949e−4。逐帧原值见 summary JSON。**f8 曲线末端属于被拒绝的训练尝试，最终提交第 200 步模型**。CSV `early_all20_mse` 是未 clamp 原生 RGB，codec 验收使用 clamp 后 RGB，二者口径不同。

这些事实表明当前帧选中支撑确实得到梯度与拟合，不支持将失败简单解释成“新点完全看不到或没有训练”。训练误差下降同时，f2 同姿态 cam00 损失 0.567 dB，说明当前释放的局部几何/材质自由度和训练相机收益仍不能稳定转移到未参与训练的视角。不能因为有 230 次参与就断言充分泛化，也不能凭本轮把问题全部归因于某一个属性。1→1 接管没有增加空间表示容量，尚未解决真正缺失支撑或单核复杂细节。这轮存储/速度通过，质量与稳定收益未通过，未启动 30/300 帧。

gate 仍在 70 步冻结；各动态帧 active regions 为 1000/284/284/236/268/269/273/144/244，open gate 数为 163/61/60/63/81/68/84/53/66。没有为出生修改 gate 规则。

## 路径、冻结与可复查性

- 配置：`tmp/early_source_replacement_v15_first10.yaml`。
- 冻结源：`tmp/experiments/early_source_replacement_v15_first10`，121 个文件 SHA 复核无变化；native 复用不可变 V9 446 文件目录。配置 SHA `133649d61fc9b652b0430cf3a3bba475ada490655a4f216c863fcb3e7be37511`。
- 实际 run：`tmp/early_source_replacement_v15_runs/fs_early_source_replacement_v15_first10/20261026T133549_38ac51`。
- 训练日志：`tmp/birth_diagnosis/early_source_replacement_v15_first10_train.log`；训练 exit0。
- 解码结果：`tmp/birth_diagnosis/early_source_replacement_v15_first10_decoded.json`；摘要：同目录 `early_source_replacement_v15_first10_summary.json`。
- 原 benchmark setup-free 构造不运行 Module.setup，漏初始化 `early_initialized`，首次测量前报 AttributeError。根源码 guard 已最小改为 `getattr(..., False)`。**训练快照、模型、包文件没有修改或重训**；`tmp/run_early_v15_benchmark.py` 仅给 frozen benchmark 构造的 finalized Module 补该 flag=False，再调用原 frozen decoder/forward/timer，成功 exit0。兼容日志 `early_source_replacement_v15_first10_benchmark_compat.log`，source 完整性证明 `early_v15_postrun_source_integrity.json`。
