# Cut Beef：2026-10-29 至 30 日前 30 帧实验

## Cut Roasted Beef：150k 首帧的前 30 帧

### 目的与设置

在 Cut Roasted Beef 上运行现有区域共享运动、逐点残差、层级 gate 与量化码流方法，检查画质与完整动态更新大小。此跑使用 150,000 点首帧；随后生成的其他首帧候选单独保留，不改变本跑结果。没有修改动态方法或运行 300 帧。

官方数据包含 20 路视频，缺少 cam04；按现有文件名排序处理，cam00 留出评估，另外 19 路训练。[作者说明](https://github.com/facebookresearch/Neural_3D_Video/issues/18#issuecomment-1239809513)解释了官方对不合格视频流的过滤，但未单独说明 cam04 的过滤原因。

全分辨率 frame 0 只用于 COLMAP/MVS：20/20 相机注册，平均重投影误差 0.8045 px，融合点云 906,247 点，保留 40,000 点初始化 PLY。所有 GS 拟合、动态训练与评估均为同一标定下的 1343×1006 半分辨率。frame 0–29 共 600 张图像，帧号和缩放内参检查通过；数据入口使用软链接。

本跑首帧只从头拟合一次，共训练 15,000 步，SH1、无 XYZ jitter，5,000 步停止增密。表内前四行是这同一次训练保存的 checkpoint，PSNR 统一为 cam00 的项目指标：

| 保存步数/处理 | GS 点数 | PSNR / dB |
|---|---:|---:|
| 5,000 | 298,918 | 33.6535 |
| 10,000 | 298,918 | 33.5924 |
| 12,500 | 298,918 | 33.4974 |
| 15,000 | 298,918 | 33.6768 |
| 15,000，透明度阈值 0.0075 | 272,174 | 33.6786 |
| 保留训练图像影响较大、且覆盖原区域的 150k 点 | 150,000 | 33.6360 |

裁点后被保留 GS 的全部属性和 optimizer 行保持一致，原有 4,070 个非空最细区域均保留覆盖。裁点静态损失 0.0426 dB。cam00 用于保存点比较，属于模型选择分数。

### 代码与配置

新增场景首帧与 30 帧配置；sequence 准备增加可选分批解码，本跑只处理 30 帧；首帧评估增加 half-only 选项。未改本场景的动态训练和码流实现。动态配置沿用冻结模板，motion/birth 为 200/100 步、每步 4 个训练视角、GS 上限 250k。只保存 frame 29 / step 300 的最终恢复 checkpoint。

```bash
conda activate ReConGS
python main.py --config=config/cut_roasted_beef_first30_legacy_20261029.yaml
```

### 结果

动态均值只统计 frame 1–29，KB=1000 bytes。完整更新包含 motion、birth、opacity、可选 DC 更新和逐帧传输 manifest，共享首帧 reference 单列。

| 指标 | 结果 |
|---|---:|
| frame 0 PSNR | 33.6360 dB |
| 动态平均 PSNR | 33.4400 dB |
| 动态平均 SSIM | 0.964100 |
| 动态平均 LPIPS，VGG | 0.115276 |
| 完整动态更新 | 59.29 KB/帧 |
| 29 帧动态总量 | 1.719 MB |
| 共享首帧 reference | 13.138 MB |
| 平均训练时间 | 44.72 s/帧 |
| 最终 GS 数量 | 147,025 |
| 最差帧 | frame 29，32.9959 dB |
| 最好帧 | frame 1，33.8081 dB |
| frame 29 | 32.9959 dB，52.31 KB |
| 前 10 个动态帧 PSNR / 更新大小 | 33.6476 dB / 68.16 KB |
| 后 10 帧 PSNR / 更新大小 | 33.2989 dB / 54.27 KB |
| 后 10 帧 PSNR 线性斜率 | -0.03108 dB/帧 |

29 个动态帧中，0 帧达到 34 dB，0 帧的完整更新不超过 50 KB。逐帧 packet 和 manifest 的磁盘大小核算全部通过。训练时间口径包含每帧评估与编码，扣除诊断记账，不含终态 checkpoint 序列化；本次未测渲染 FPS 或完整播放 FPS。

### 分析与限制

当前首帧和动态均值均未达到整体 34 dB 目标。比较同一次 15,000 步训练的第 5,000 步与第 15,000 步 checkpoint，训练视角平均 PSNR 从 35.2136 升到 37.5927 dB，cam00 仅从 33.6535 变成 33.6768 dB，提示继续增加训练步数未必能改善新视角；这不是对过拟合原因的独立验证。150k 裁点损失仅 0.0426 dB，不能把动态画质不足全部归因于首帧裁点。

码流主体为 motion：前 10 帧 motion 平均 58.36 KB，后 10 帧为 45.50 KB。rate multiplier 从 1.000 到 3.069，前段超预算和后续反馈调整需分别看待。最终净 GS 变化为 -2,975 点；仅凭点数或总 PSNR 无法确认 retirement、共享层级或某个约束是画质变化原因。

后续首帧比较按画质选择完整候选，不再把 150k 设为必选点数。更充分的增密、SH 表达和延长拟合是否有效，以各自 cam00 实测为准；首帧改善也不能替代动态序列的验证。本记录只覆盖该场景前 30 帧，不代表全 300 帧或 SOTA 结论。

### 产物

- 历史配置快照：`/home/gamer-02/Documents/DeepLearning/my4dgs/config/cut_roasted_beef_first30_legacy_20261029.yaml`；通用 `cut_roasted_beef_first30.yaml` 已切换到新 SH1 首帧入口，本记录仍是原 2026-10-29 实验。
- 运行：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/cut_roasted_beef_20261029/cut_beef_uniform_first30_20261029T114336/20261029T131722_be27df`
- 数值汇总：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/cut_roasted_beef_20261029/cut_beef_uniform_first30_20261029T114336/20261029T131722_be27df/cut_roasted_beef_first30_summary.json`
- 逐帧 CSV：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/cut_roasted_beef_20261029/cut_beef_uniform_first30_20261029T114336/20261029T131722_be27df/cut_roasted_beef_first30_per_frame.csv`
- 最终恢复点：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/cut_roasted_beef_20261029/cut_beef_uniform_first30_20261029T114336/20261029T131722_be27df/rigid-ckpt-29-300.pth`；SHA256 `4cefed83ce493d425eae9a493da8bfdf8607df07c2f349f86d4d619264cfb728`
- 初始化 PLY：`/home/gamer-02/Documents/DeepLearning/my4dgs/data/n3dv_prepared/cut_roasted_beef_frame0_20261029T114336/reconstruction/points3D_downsample2.ply`

2026-10-29 按用户要求清理了原训练 5k/10k/12.5k 和已停止续训 20k/25k 共 5 个多余 checkpoint，释放约 430.5 MB；相应评测日志保留。继续使用 150k 首帧，并保留 15k 原始母版、逐点评分与所需空间布局，供其他点数候选生成和复现使用。目前尚未生成或评测 12 万、18 万、20 万等中间点数候选，动态 30 帧只完成了本记录的 150k 版本。

---

## Cut Roasted Beef：SH3 共享首帧与新生点外观继承的前 30 帧

> 2026-10-31 更新：本文选择 SH3 时比较的是旧 SH1 资产，不能据此断言 SH1 阶数本身不足。后续已按用户要求改用新训 SH1 首帧入口，静态证据见[首帧处理指南](../first_frame_recipe.md)。本文历史指标不变；其中按 cam00 选取 10k 保存点及后续沿用 10k 的建议不再适用于当前固定 15k 终点协议。

### 核心想法

本实验检验两个相连但需要分别表述的假设：第一，原 SH1 共享首帧限制了新视角和后续动态帧的画质上限，改用 SH3 可能提高视角相关外观表达；第二，新生点通常是已有表面附近的几何修补，因此其高阶 SH 可以从码流参考点继承，避免把高阶 SH 清零后再发送整组“抵消残差”。后者是代码检查发现的具体问题，不等同于已经证明所有新生点都应永久共享外观。

数据、训练/测试相机划分和动态训练计划与 2026-10-29 的 Cut Beef 150k 首帧实验一致：frame 0–29，cam00 留出评估，其余相机训练，1343×1006 半分辨率。动态均值只统计 frame 1–29，KB=1000 bytes。

### 代码与配置

- `scripts/run_first_frame.py` 增加独立输出目录、SH 阶数和首帧步数覆盖参数，候选不会复用历史实验目录。
- `pipeline/frame_plan.py` 在新生点确定最近旧点参考后，将该参考的 `sh_rest` 复制为新生点初值。由于同一个旧点也是 packet predictor，出生时高阶 SH 残差为零；DC 仍由责任加权训练图像颜色初始化并编码。
- `tests/test_frame_plan.py` 增加 SH3 新生点零高阶残差及无损继承检查，并更新两轮出生生命周期断言。相关 240 项测试全部通过。
- 首帧配置：`config/cut_roasted_beef_frame0_sh3_goal_20261030.yaml`。
- 动态复现配置：`config/cut_roasted_beef_sh3_inherit_30_20261030.yaml`。

首帧固定沿用已有几何训练配方，改为 SH3。训练产生 280,076 点；用已有影响度与区域覆盖规则压缩为 150,000 点，未复制或改写历史结果。

### 首帧结果

| 候选 | GS | cam00 PSNR / dB |
|---|---:|---:|
| 旧 SH1，15k 原始 | 298,918 | 33.6768 |
| 旧 SH1，150k | 150,000 | 33.6360 |
| SH3，5k，透明度处理后 | 280,076 | 34.1999 |
| SH3，10k，原始 / 透明度处理后 | 280,076 | 34.3280 / 34.3298 |
| SH3，12.5k | 280,076 | 34.2355 |
| SH3，15k | 280,076 | 34.0634 |
| SH3，10k 压缩后 | 150,000 | 34.2458 |

10k SH3 压缩到 150k 后相对处理后模型损失 0.0840 dB，相对旧 150k 首帧提高 0.6098 dB。这里比较过多个 checkpoint 并用 cam00 选中 10k，因此 34.2458 是模型选择后的分数，不是完全独立的测试估计；后续场景应固定使用 10k 配方，避免逐场景重复选择测试帧。

### 30 帧结果

| 指标（frame 1–29） | 旧 SH1 150k | SH3 + 外观继承 | 变化 |
|---|---:|---:|---:|
| 平均 PSNR | 33.4400 dB | 34.1627 dB | +0.7227 dB |
| 平均 SSIM | 0.964100 | 0.965029 | +0.000929 |
| 平均 LPIPS，VGG | 0.115276 | 0.114817 | -0.000459 |
| 完整动态更新 | 59.2925 KB | 58.0822 KB | -1.2103 KB |
| motion | 50.2704 KB | 48.9504 KB | -1.3200 KB |
| birth | 1.8332 KB | 2.9704 KB | +1.1372 KB |
| opacity | 6.0688 KB | 5.0416 KB | -1.0272 KB |
| 前 10 个动态帧 PSNR / 更新 | 33.6476 / 68.1572 | 34.4371 / 65.1510 | +0.7895 / -3.0062 |
| 后 10 帧 PSNR / 更新 | 33.2989 / 54.2702 | 34.0205 / 53.7793 | +0.7216 / -0.4909 |

新实验 29 个动态帧共 1,684,383 bytes；平均训练时间 41.71 s/帧，最终 147,329 GS，最终 rate multiplier 为 2.74765。动态帧最低/最高 PSNR 为 33.8735/34.5677 dB。只有 1 帧完整更新不超过 50 KB，因此平均 58.08 KB 仍未达到目标；后 10 帧降至 53.78 KB，说明反馈在收敛，但不能据此保证完整 300 帧均值会低于 50 KB。

S²GS-Full 表中 Cut Roasted Beef 为 34.12 dB；本实验前 30 帧动态均值高 0.0427 dB。这个差值很小，而且本次使用 cam00 选择了首帧 checkpoint，尚未统一验证六场景、完整 300 帧和相同口径的存储/FPS，所以不能表述为整体超过 SOTA。

### 分析与限制

已验证的是：在 Cut Beef 的这次 30 帧运行中，SH3 共享首帧加上修正后的新生点高阶 SH 继承，同时提高了画质并略微减少动态更新；出生包平均仅 2.97 KB，没有出现 SH3 预期的整组抵消残差膨胀。尚未通过独立对照分离“首帧 SH3”和“新生点继承”各自贡献，也不能仅凭总指标把收益归因于自适应空间层级。

共享 reference 从旧实验的 13.138 MB 增至 31.636 MB，是明确代价。它只传一次，应与逐帧更新分列；对于短序列，若把 reference 摊入每帧，总存储未必更优。当前主要逐帧瓶颈仍是 motion（48.95 KB），继续改进需优先降低运动/结构包，同时守住约 34.16 dB，而不是压缩仅约 3 KB 的出生包。

本轮未测已解码渲染 FPS 和包含读包、解码、更新、渲染的完整播放 FPS；也未运行 frame 30–299。上述结论只覆盖一个场景前 30 帧。

### 产物

- 首帧原始运行：`output/goal_n3dv_frame0_20261030/cut_beef_f0_sh3_goal_20261030/20261030T012803_fab7ec`
- 150k 共享首帧：`output/goal_n3dv_assets_20261030/cut_beef_sh3_10k_cov150/ckpt-0-10000.pth`
- 动态运行：`output/goal_n3dv_dynamic_20261030/cut_beef_sh3_inherit_30_20261030/20261030T013802_23b79a`
- 动态恢复点：上述目录的 `rigid-ckpt-29-300.pth`
- 共享 reference：31,636,099 bytes；与逐帧更新分列。

---

## Cut Roasted Beef：码率 warm-start 与区域 DC 的前 30 帧

### 核心想法

上一版 SH3 Cut Beef 的动态画质已经达到 34.1627 dB，但前段码率反馈收敛慢，完整动态更新仍为 58.08 KB/帧。本轮做两个可联合工作的改动：首个动态帧直接以 `rate_multiplier=2` 启动；允许 existing GS 的 DC 颜色在同一预构建空间层级上按区域共享、量化传输。目标是在保留上一版画质的同时把完整动态更新推近 50 KB/帧。

本实验沿用同一个 150,000 GS、SH3 共享首帧和新生点高阶 SH 继承。frame 0–29 使用 1343×1006 半分辨率图像，cam00 留出评估，其余相机训练。下文动态均值统计 frame 1–29，KB=1000 bytes。

### 代码改动

- `RigidTreeConfig.initial_rate_multiplier` 为字节反馈控制器提供显式 warm-start；checkpoint 恢复继续使用保存的控制器状态，不会重新 warm-start。
- frame-plan 支持 `spatial_region + gate_joint + joint_with_birth` 的 DC 更新。共享发现阶段使用随机 gate；残差感知阶段与 motion 一样使用 hard owner，避免颜色和运动引用不同的区域。
- 区域 DC 进入每步优化、梯度诊断、retirement 后重排、量化码流、独立解码和 checkpoint 恢复链路。
- 配置：`config/cut_roasted_beef_sh3_warm2_regdc_30_20261030.yaml`。

### 结果

对照为同一 SH3 reference、没有 warm-start 和 DC 更新的 `cut_beef_sh3_inherit_30_20261030`。

| 动态 frame 1–29 | 上一版 | 本轮 | 变化 |
|---|---:|---:|---:|
| PSNR | 34.162741 dB | **34.131575 dB** | -0.031166 dB |
| SSIM | 0.965029 | 0.964855 | -0.000174 |
| LPIPS | 0.114817 | 0.114858 | +0.000041 |
| 完整动态更新 | 58.0822 KB/帧 | **54.1781 KB/帧** | -3.9041 KB，-6.72% |
| motion | 48.9504 KB/帧 | 45.6496 KB/帧 | -3.3008 KB |
| birth | 2.9704 KB/帧 | 2.0407 KB/帧 | -0.9297 KB |
| opacity | 5.0416 KB/帧 | 4.9501 KB/帧 | -0.0914 KB |
| DC / manifest | 0 / 1.1198 KB/帧 | 0.3319 / 1.2058 KB/帧 | +0.4179 KB |
| 平均区域数 | 362.28 | 189.59 | -47.7% |
| 平均逐点残差行 | 24,189.2 | 21,262.2 | -12.1% |
| 平均 birth 数 | 84.55 | 61.86 | -26.8% |
| 最终 GS | 147,329 | 146,879 | -450 |
| 平均训练时间 | 41.71 s/帧 | 53.82 s/帧 | +12.12 s/帧 |

本轮 29 个动态更新共 1,571,165 bytes。前 10 / 中间 9 / 后 10 帧分别为 34.4038 dB、57.2709 KB；34.0101 dB、52.8582 KB；33.9687 dB、52.2732 KB。5/29 帧不超过 50 KB，21/29 帧不超过 55 KB；最低/最高为 48.728/85.646 KB。rate multiplier 从 2.0 增至 3.4140，下一帧状态为 3.4307。

共享 frame0 reference 为 31,636,099 bytes，继续与逐帧动态更新分列。frame0 的已记录 cam00 PSNR 为 34.2458 dB，若仅为便于比较把它和 29 个动态帧算术平均，frame0–29 为约 34.1354 dB；这不是完整 300 帧结论。

### 分析与结论

本轮完整动态更新比上一版少 6.72%，画质只下降 0.031 dB，属于当前目标下更好的率失真点。S²GS-Full 图中 Cut Roasted Beef 为 34.12 dB；本轮动态均值高约 0.0116 dB，包含 frame0 的 30 帧均值高约 0.0154 dB，但差值很小。尚未统一完成六场景、完整 300 帧和相同存储/FPS口径，因此不能表述为整体超过 SOTA。

实际节省主要来自 motion 和 birth，而区域 DC 自身增加约 0.332 KB/帧。区域数、逐点残差行和 birth 数同时下降，与更强的首帧码率约束一致；但本轮同时启用了 warm-start 和区域 DC，逐帧训练也存在随机波动，不能把这些变化单独归因给其中一个模块，更不能仅凭总指标证明可学习层级机制有效。

54.18 KB/帧仍高于 50 KB 目标约 8.36%，且首帧动态更新仍有 85.65 KB。下一步应把同一机制验证到高前段码率、颜色变化明显的 Coffee Martini；warm-start 强度应依据已有控制器收敛状态设置或改成可自校准，而不是继续仅压缩已经很小的 DC 包。区域 DC 带来约 29% 的训练时间开销，也需要后续优化，但本轮没有测渲染 FPS 或完整播放 FPS，不能用训练速度替代。

### 完整性验证与产物

- 运行：`output/goal_n3dv_dynamic_20261030/cut_beef_sh3_warm2_regdc_30_20261030/20261030T025021_e69195`
- 最终 checkpoint：`rigid-ckpt-29-300.pth`。
- 独立 CPU 解码从共享 frame0 连续重放至 frame29；最终 146,879 GS，decoder SHA256 为 `00eeb3c36ad810a0b6cb60dc3203337d7ababfb0610e84d03dff062471effad5`，逐帧 motion/raw 属性哈希全部通过。
- 相关回归：`tests.test_frame_plan`、区域/稀疏颜色、frame-plan、motion training 和 spatial motion，共 264 项通过。
