# 首帧处理指南

目标是先得到质量可靠的首帧，再删掉影响较小的 GS。点数越少越好，但最终以动态 30 帧的画质和存储为准，不能只看首帧 PSNR。

## Cut Beef 当前首帧入口（2026-10-31 更新）

按用户要求，后续 Cut Beef 实验采用新训练的 **SH1 / exact15k / 150,000 GS** 首帧。当前配置入口为 [cut_roasted_beef_first30.yaml](../config/cut_roasted_beef_first30.yaml)。配置采用仓库当前统一 RD 与末段联合训练机制，所有 19 个非 cam00 相机参与训练；没有继承历史 ABCD 提交或 internal-probe 选择策略。配置已准备，新的动态 30 帧尚未运行。

- 首帧资产：`/mnt/SSD500GB/my4dgs_assets/cut_beef_sh1_supportsafe15k_cov150_20261031T190839`。
- checkpoint / reference SHA256：`65e5abf049067e801dc6b0b321e3f336c1a58766f043f42a998df958414cf424` / `6de15990b9b96fed6c200fbb90dcbcf5b6434a7d2bbb0542b7845a6eddcc8d07`。
- 新 SH1 的 cam00 首帧为 **34.504936 dB**，共享 reference 为 **13.035366 MB**；同一输入和评测程序重测当前 SH3 为 **34.458683 dB / 31.221183 MB**。MB=1,000,000 bytes，checkpoint 内的优化器状态不计入 reference 大小。
- 评测与六属性一致性核验：`/mnt/SSD500GB/my4dgs_output/cut_beef_sh1_retrain_20261031T190839/comparison.json`、`/mnt/SSD500GB/my4dgs_output/cut_beef_sh1_retrain_20261031T190839/verification.json`。这里只确认首帧表现，不能沿用历史 SH3 的动态指标。

历史选择 SH3 的依据是旧 SH1 首帧仅 33.6360 dB、动态前缀仅 33.4400 dB；这不是“SH1 本身不够用”的结论。新 SH1 从零训练 15,000 次优化，每步使用两个视角；采用 40k→200k 渐进上限及多视角覆盖支持的增密，再由训练相机影响度和空间覆盖保留 150k。旧 SH1 使用每步一个视角与普通增密，因此同为 SH1 不代表训练出的几何、透明度和颜色权重相同。提升在裁点前的母版中已经存在：旧 33.6786 dB，新 34.5065 dB。旧图像缓存已清理，当前只能确认相机文件同源，不能补作新旧输入逐像素等价验证，也不能将全部提升归因于某个单独改动。

上述是首帧资产入口与已有静态证据的更新，不是一份已完成的动态实验记录。历史 SH3 实验保持独立；旧的 2026-10-29 通用配置已保存为 legacy 配置（原路径 `../config/cut_roasted_beef_first30_legacy_20261029.yaml` 已不在工作区）。

## Coffee Martini 当前首帧（2026-11-02）

选用 **SH1 / 15,000 步 / 160,000 GS**，cam00 首帧 PSNR 为 **29.3362 dB**。训练数据排除 cam13；首帧资产为 `/mnt/SSD500GB/my4dgs_assets/coffee_no13_baseline_160k_frame0_20261102`。

后续动态训练入口为 [coffee_martini_no13_first30.yaml](../config/coffee_martini_no13_first30.yaml)，沿用原有 Coffee 动态参数。这个首帧尚未验证动态 30 帧。

## Coffee Martini 历史首帧比较

按 Cut Beef 同一配方从零训练 SH1：固定 15,000 步、每步两个视角、40k→200k 渐进增密上限，最后按训练视角影响和母版区域覆盖保留 150k。沿用已核对的 no13 图像与 40k 初始化点云，16 路训练、cam00 评估；本次没有重做 COLMAP。

- 母版加载后 190,307 点，cam00 **29.343124 dB**；保留 150,000 点后为 **29.323029 dB / SSIM 0.923229**，共享 reference **13.086370 MB**。
- 当前保留基线同程序重测为 **29.023022 dB**，新候选高 0.300007 dB；相对历史 15k 最好记录 29.312492 dB 仅高 0.010536 dB，不能称为明显突破。动态帧尚未训练，现有动态入口未切换。
- 候选资产：`/mnt/SSD500GB/my4dgs_assets/coffee_no13_sh1_supportsafe15k_cov150_20261031T214049`。六类属性与 reference 逐位一致，16 路评分中没有 cam00 或 cam13；裁点保留了母版区域覆盖，随后按 Cut Beef 流程为使用版生成独立布局。
- [训练配置](../config/coffee_martini_no13_frame0_sh1_mv2_supportsafe200k_15k_20261031T214049.yaml)；完整命令、输入及源码指纹见 `/mnt/SSD500GB/my4dgs_output/coffee_no13_sh1_retrain_20261031T214049/recipe.json`，对比和核验见同目录 `comparison.json`、`verification.json`。这是首帧准备记录，不是动态实验报告。
- 2026-11-02 点云重建对照：重新用 17 路无 cam13 图像生成点云，对齐后使用相同的训练图像、相机和配方。母版为 **28.9079 dB**（旧点云 **29.3431 dB**）；裁到 15 万点后为 **28.8942 dB**（旧版 **29.3230 dB**），暂保留旧版。两次训练有随机性，新点云的坐标对齐也有误差，不能据此否定所有重建方法。[对照报告](/mnt/SSD500GB/my4dgs_output/coffee_no13_rebuild_aligned_20261102/comparison.json)。
- 2026-11-02 补充：旧母版裁到 **16 万点为 29.3362 dB**，裁到 **19 万点为 29.3431 dB**；前者少 3 万点，仅低 0.0069 dB。两者都是 SH1、15,000 步、未用 cam13，checkpoint 和 reference 已核对一致。可选资产分别为 `/mnt/SSD500GB/my4dgs_assets/coffee_no13_baseline_160k_frame0_20261102` 和 `/mnt/SSD500GB/my4dgs_assets/coffee_no13_best_frame0_20261102`；尚未用这两个资产跑动态 30 帧。
- 本轮另试了窗外补点和借鉴 [OR2](https://github.com/bbangsik13/OR2) 的训练视角误差图，均未超过上述 19 万点结果。误差图方案在 12 万点时为 **29.2341 dB**，同点数旧母版为 **29.1792 dB**；各为单次训练，不能把差异直接归因于某一改动。[点数比较](/mnt/SSD500GB/my4dgs_output/coffee_no13_baseline_point_sweep_20261102/comparison.json)、[误差图结果](/mnt/SSD500GB/my4dgs_output/coffee_no13_or2_sh1_20261102/comparison.json)。

## 基本原则

1. **先确认数据正确。** 相机、帧号、图像尺寸和内参必须一致。Coffee Martini 排除不同步的 cam13；16 路用于 GS 训练，cam00 用于评估。现有点云融合和模型选择用到了 cam00，因此这不是完全独立的测试结果。
2. **按动态训练的实际分辨率训练首帧。** 不要先在另一种分辨率训练，再直接拿来跑动态序列。
3. **训练预算保持 15,000 step。** 改变 GS 数量上限时，只调整 `module.densify.max_gaussians`，不同时延长训练步数或改变其他训练参数；不要改成 17,500、20,000 step。
4. **统一使用 15,000 步终点，不能按 PSNR 改选中间步数。** 中间保存点仅用于排障和观察收敛；即使 10,000 或 12,500 步的 PSNR 更高，也不能替代 `ckpt-0-15000.pth`。GS 数量上限与训练步数是独立参数。
5. **删点后保留原属性和原空间划分。** 不改被保留 GS 的位置、尺度、旋转、透明度和 SH，也不重新生成候选区域。
6. **先跑 30 帧，再决定是否跑 300 帧。** 静态首帧好，不代表后续动态一定好。

## 操作流程

### 1. 训练一个质量可靠的首帧

首帧训练要使用动态序列相同的图像、相机划分和分辨率。`trainer.num_init_steps` 固定为 `15000`。在只比较 GS 点数预算的实验中，学习率日程、增密开始/结束步数、SH 阶数等沿用原训练配置，只改变 GS 数量上限；用户明确要求调整 SH 或首帧训练方法时，应另存新配方并记录差异，训练预算仍固定。处理和动态训练的源必须是实际完成 15,000 步的终点 checkpoint，不能通过选择较早保存点或增加训练步数来改变实验协议。

**统一协议固定使用 15,000 步终点。** 原 checkpoint、裁点后的 checkpoint 都必须满足 `current_frame == 0`、`global_step == 15000`，动态配置的 `trainer.num_init_steps` 也必须是 `15000`。不是“最多训练 15,000 步后在预算内挑最好的一步”。

GS 数量与训练步数是两个独立参数。例如 `module.densify.max_gaussians: 180000` 表示增密阈值约 18 万点，`trainer.num_init_steps: 15000` 表示训练 15,000 步。原生增密在每批操作前检查数量，实际 GS 数可能略高于阈值，因此必须记录实际点数，并核验最终入口满足约定预算。

Coffee Martini 当前使用：

- 数据：`data/colmap4d/coffee_martini_half_no13`
- 完整首帧入口：`data/colmap4d/coffee_martini_full_first_no13`，只在重建初始化点云时使用
- 图像尺寸：`1341 × 1006`
- 首帧训练配置：[coffee_martini_no13_frame0_sh1_mv2_supportsafe200k_15k_20261031T214049.yaml](../config/coffee_martini_no13_frame0_sh1_mv2_supportsafe200k_15k_20261031T214049.yaml)
- 不使用 cam13，不添加训练专用 XYZ 噪声

```bash
conda activate ReConGS
# 新配置需设置 num_init_steps: 15000，测试和保存步数均不超过15000。
# 当前 main.py 的动态入口要求已有首帧；从零训练使用原生静态入口。
frame0_config="config/your_scene_frame0.yaml"
python scripts/run_first_frame.py --config "$frame0_config" \
  --name "frame0_$(date +%Y%m%dT%H%M%S)"
```

### 2. 比较不同点数

[compact_initial_model.py](../scripts/compact_initial_model.py)会估计每个 GS 对训练视角图像的影响，优先保留影响大的点，同时保证原有最细区域不会被清空。cam00 和未来帧不参与逐点打分。

下面是 Coffee Martini 历史 16 万点候选的复现形式，其源 checkpoint 为历史 17,500 步结果，不能作为新实验的步数设置依据。新实验必须替换为本次 15,000 步终点 checkpoint、相应原空间模板和渲染配置。输出必须使用新路径，不能覆盖已有结果。

```bash
source_ckpt="output/coffee_martini_no13_dynamic_20261029/assets/frame0_source/ckpt-0-17500.pth"
reference="output/coffee_martini_no13_dynamic_20261029/assets/reference_layout/reference_frame0.npz"
output_ckpt="output/coffee_martini_no13_dynamic_20261029/assets/candidates/cov160_$(date +%Y%m%dT%H%M%S)/ckpt-0-17500.pth"

python scripts/compact_initial_model.py \
  --checkpoint "$source_ckpt" \
  --render-config output/coffee_martini_no13_dynamic_20261029/assets/frame0_source/config.yaml \
  --coverage-reference "$reference" \
  --target-count 160000 \
  --opacity-threshold 0.0075 \
  --probes 2 \
  --seed 1729 \
  --output "$output_ckpt"
```

Coffee Martini 已比较 10 万、12 万、14 万、15 万、16 万、17 万和 18 万点。其他场景可在约 10 万到 20 万点之间取样，但不能超过透明度过滤后的源点数。静态 PSNR 接近时，优先把点数更少的候选送入动态 30 帧验证。

### 3. 跑动态 30 帧

只替换首帧 checkpoint 和对应的 `reference_frame0.npz`，新增 GS、损失、量化和训练步数保持不变。这样才能判断差异是否来自首帧。

Coffee Martini 当前 16 万点配置为 [coffee_martini_no13_first30.yaml](../config/coffee_martini_no13_first30.yaml)：

```bash
conda activate ReConGS
python main.py --config=config/coffee_martini_no13_first30.yaml \
  experiment_name="coffee_no13_cov160_first30_$(date +%Y%m%dT%H%M%S)"
```

30 帧完成后比较平均 PSNR、每帧存储、最终 GS 数量和困难区域画质。通过后再用同一首帧和同一动态配置扩展到 300 帧。

## 当前结果

- **Flame Steak：** 从加载后的 206,237 点保留 150,000 点，完整 300 帧 PSNR 从 34.0647 提高到 34.1203 dB，动态存储从 50.97 降到 48.96 KB/帧。这只证明 15 万点适合该场景。
- **Coffee Martini（历史实验）：** 17,500 步首帧经过透明度过滤后有 180,741 点，cam00 PSNR 为 29.3634 dB；裁到 16 万点后为 29.3585 dB。其动态 frame 1–29 平均为 29.5782 dB、64.54 KB/帧，未达到当前画质和存储目标，因此不直接扩展到 300 帧。该历史训练预算不沿用到新实验。

## 检查清单

- 数据没有错帧或不同步，相机数量和留出视角正确。
- 首帧固定使用实际完成 15,000 step 的终点；只调整 GS 上限，不联动改变训练步数、学习率日程或增密日程。源、裁点后 checkpoint 与动态入口的步数必须都等于 15,000，不能只检查“不超过 15,000”。
- checkpoint 的分辨率、SH 阶数和透明度阈值与动态配置一致。
- 裁点前的 checkpoint 与 `reference_frame0.npz` 属于同一首帧；脚本的 XYZ 校验必须通过。
- 被保留 GS 的全部属性不变，所有原有非空区域仍有覆盖。
- 每次实验使用新的输出目录，不覆盖历史产物。
- 只有完成至少 30 帧后，才在 `docs/experiment/` 记录实验结论。
