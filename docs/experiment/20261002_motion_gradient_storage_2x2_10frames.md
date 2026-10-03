# 运动梯度路径 × storage：四组 10 帧控制实验

实验与整理日期：2026-10-02。**四组均完成共享首帧后的第 1–10 帧，共 40 个训练帧。** 用户明确授权实现、训练、评估和本记录；启动前由 GPT-6 luna xhigh 只读审查。没有追加种子、长序列训练、提交或推送。

## 1. 先看结果

本轮回答：避免深层梯度被概率连乘压小，是否一定需要让关闭的运动分支也接收梯度？同时检查 storage 在两种运动梯度下的作用。

本次短序列中，**实际硬路径＋storage=0.001** 的质量和参数估算组合最好。同权重下，它比全 SPLIT 高 **1.0151 dB**，10/10 帧更高；参数估算低约 **53.05 倍**。但两组硬路径均没有打开逐 GS 残差门。它们避开了大量逐点候选更新，也缺少逐点候选探索，不能由这个结果宣称自适应深层表达成功。

关闭 storage 在全 SPLIT 下提高约 0.1850 dB，在硬路径下却降低约 0.6193 dB。不能把失败简单归为“storage 太强”。两种运动反向路径与 gate/storage 存在明显交互。

所有组仍低于同帧原版参考 34.3010 dB。参考的训练预算和方法不同，仅作画质背景，不是本轮等预算消融。短窗口也不证明长期稳定，尤其历史预算 500 实验曾在前 79 帧接近参考，后 50 帧落后约 5.37 dB。

## 2. 概率连乘的动机与三种路径

用户担心的梯度衰减确实存在。若用软概率混合运动，第三层区域运动要经过两个祖先 SPLIT 概率；逐 GS 残差还要经过第三层 SPLIT。例：三个概率都为 0.2，逐点平移的梯度系数是 `0.2³ = 0.008`。本次数值测试验证了这个例子。

| 路径 | 实际硬门打开时 | 实际硬门关闭时 | 本轮是否训练 |
| --- | --- | --- | --- |
| 软概率混合运动 | 被祖先概率缩小 | 仍可能有软梯度 | 否，只用于数值解释 |
| `all_split` | 使用全候选刚体链 Jacobian | 仍使用全候选链 Jacobian | A、B |
| `hard_path` | 使用实际选中的刚体链 Jacobian，无软概率乘子 | 该分支的运动梯度为 0 | C、D |

四组**硬门前向不变，gate 自身代理不变**。gate 仍使用原来的软递归，候选运动在这条图里 detach。改变的只有运动参数的 Jacobian。因而当前 all_split 消除的是运动参数的概率衰减，**并没有同时消除深层 gate logits 的祖先概率衰减**。

all_split 给关闭候选的梯度来自当前硬门画面的误差，可能与该候选真正启用后的误差不同；它是代理梯度，并非实际硬分支的精确导数。hard_path 符合实际分支，但关闭候选不能预先拟合。本轮比较这两种取舍，没有把 hard_path 换成软权重，也没有偷偷增加候选预热。

## 3. 控制条件、身份与代码版本

- 共享首帧：`/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf/ckpt-0-15000.pth`。214,996 个 GS，首帧直接加载，没有重训。
- 共享 checkpoint SHA256：`0715e9f0ce2eaea16633207fb9f1e87170162f9fb2974f8b9f1a20206da35c7d`。
- 数据通过既有软链接 `data/colmap4d/flame_steak` 访问；外部入口为 ReCon-GS 对应 data 目录，帧图像/COLMAP 来源为 S2GS 的 `data/standard_flame_steak_300`。未复制或修改大型数据。
- `num_frames: 11` 表示首帧 0＋后续 1–10；`resolution: -1`，自动缩小后 1600×1199；SH degree=1；20 个训练相机、1 个测试相机 cam00。
- 每帧 200 步 gate/运动联合学习＋100 步新增 GS；`gate_refine_steps: 0`。温度 1.5→0.5，gate LR=0.02；fine_region_gs=24；每 5 帧重建；motion_init_decay=0.6。
- 区域平移/旋转 LR=0.0005；逐点平移/旋转 LR=0.0001；全部新增学习率、量化步长、裁剪规则沿用最新配置。候选数量无限，最终最多 200 点，无最低保留数。
- Storage 前 30 步关闭，再用 50 步 ramp。唯一方法变量是 `motion_gradient_mode` 与 `storage_loss_weight`；其余配置逐项相同，只有 experiment_name 作为身份不同。
- 全局 random_seed=0、PYTHONHASHSEED=0。视角使用独立 RandomSampler/worker generator 和 PairedFetcher 私有 Python RNG；模型随机数消耗不改变相机顺序。四组每帧 300 步、10 帧的相机 trace SHA256 完全相同。
- 训练源码基准：Git `b9a778fc5ad74aef24851a47dacc1b096a193601` **加未提交 diff**，不是仅凭 Git hash 声称源码相同。[manifest](../../output/gradient_ablation_20261002/manifest.json) 保存源文件 SHA256，[source_snapshot](../../output/gradient_ablation_20261002/source_snapshot/) 保存代码、测试、配置及 working_tree.patch。四组使用冻结的同一实现。

只读核对 `~/.local/share/Trash`：没有发现从仓库 `config/` 目录删除的 YAML；唯一独立 YAML 命中是根目录 `environment.yml`（2026-09-23 删除）。另有 70 条原路径位于项目 `output/` 的删除记录。与其区分，实际在被删运行目录中找到并读取了 7 份 `config.yaml` 快照：`flame_steak_gradhet4_recongs_first_checkpoint`、`flame_steak_selective_motion_21`、`fs_motion_only_300_steps200`、`fs_birth_budget500_300_steps300`、`fs_residual_rate0001_300_steps200`、`fs_learned_selective_300`、`fs_residual_rate0_300_steps200`；对应运行目录均于 2026-10-01 18:15:34 删除。快照是各次运行保存的配置副本，不是仓库 `config/` 原文件；Trash 元数据只记录原路径和删除时间，不含配置内容。未恢复文件，也未查看非项目被删文件内容。



| 组 | 仓库 YAML | 实际运行输出 |
| --- | --- | --- |
| A：全 SPLIT，λ=0.001 | [fs_gradpath_all_split_storage001_seed0_11.yaml](../../config/fs_gradpath_all_split_storage001_seed0_11.yaml) | [20261002T160238_720236](../../output/fs_gradpath_all_split_storage001_seed0_11/20261002T160238_720236/) |
| B：全 SPLIT，λ=0 | [fs_gradpath_all_split_storage0_seed0_11.yaml](../../config/fs_gradpath_all_split_storage0_seed0_11.yaml) | [20261002T160910_913299](../../output/fs_gradpath_all_split_storage0_seed0_11/20261002T160910_913299/) |
| C：硬路径，λ=0.001 | [fs_gradpath_hard_path_storage001_seed0_11.yaml](../../config/fs_gradpath_hard_path_storage001_seed0_11.yaml) | [20261002T161549_cf83b0](../../output/fs_gradpath_hard_path_storage001_seed0_11/20261002T161549_cf83b0/) |
| D：硬路径，λ=0 | [fs_gradpath_hard_path_storage0_seed0_11.yaml](../../config/fs_gradpath_hard_path_storage0_seed0_11.yaml) | [20261002T162228_0035ff](../../output/fs_gradpath_hard_path_storage0_seed0_11/20261002T162228_0035ff/) |



简要代码改动：`QuantizedResidualConfig.motion_gradient_mode` 新增 all_split/hard_path，生产默认仍为 all_split；`GatedResidualMotion.forward` 只切运动反向分支；pipeline 增加显式 seed；data/trainer 增加独立相机 RNG 和逐步 trace；module 增加默认关闭的质量/梯度诊断。历史输出未修改。

## 4. 验证和测量口径

启动前完整 **53 项 `test_nested_prototype.py`** 与 **13 项 `test_experiment_inventory.py`** 通过，`git diff --check` 通过。新增测试覆盖：两模式硬输出和 gate 梯度逐项一致；KEEP/SHARE/SPLIT 下的运动梯度正确；软祖先概率连乘；相机 RNG 不受模型随机消耗影响；诊断恢复 RNG、renderer 和 rigid override。luna 审查没有发现启动阻塞。

运行冻结核验：manifest 列出的 **23 个源码、测试和配置文件**，其 SHA256 均与冻结快照及当前工作区逐项匹配。刚体 override 恢复边界使用 `getattr(..., None)` 的修复已在首组运行前进入冻结源码：源码文件时间为本地 15:58:39，快照补丁生成于 16:02:35.580，首组训练实际启动于 16:02:35.684。四组指标对应的实现与当前实现之间没有这些文件层面的差异。

每帧记录 5 个阶段：删点前、删点后、运动完成、最终预算筛点前、最终预算筛点后。每阶段测全部 20 个训练相机和 cam00。图像目标为未 clamp 图像的 `0.8×L1 + 0.2×(1−SSIM)`，不含 storage；PSNR/SSIM/L1 使用 clamp 图像，与正式评估一致。没有可靠背景 mask，未报告背景专属误差。

采样第 1、30、80、150、200 步的 combined-loss 梯度，在 optimizer.step 前记录各层运动/gate 和逐点运动的 L2、max_abs、非零行数；门变化只在这些采样点之间统计，不是逐步切换次数。

已自动核验：四组首帧拟合前全部相机质量逐项相同；相机 trace 一致；after_motion 与正式 step200 评估匹配；after_retention 与正式 step300 匹配。检查结果见 [results.json](../../output/gradient_ablation_20261002/summary/results.json)。

**存储都是参数字节估算，不是编码文件大小。** 计入可达 gate、有效区域运动、开放逐点运动、新点属性/来源和删除索引；不计共享首帧、文件头、优化器等。本轮未运行 codec，不能将下表与历史 `.bin` 平均 bytes 直接排大小，也不能称为实际视频压缩率。

## 5. 最终结果：第 1–10 帧等权均值

| 组 | PSNR/dB | SSIM | 硬门估算 KiB/帧 | 训练图像目标↓ | KEEP | 逐点残差开放 | 新点/帧 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A：全 SPLIT，λ=0.001 | 32.4976 | 0.964172 | 624.86 | 0.017496 | 55.66% | 9.97% | 159.0 |
| B：全 SPLIT，λ=0 | 32.6826 | 0.966181 | 1988.03 | 0.016701 | 49.10% | 36.16% | 200.0 |
| C：硬路径，λ=0.001 | 33.5127 | 0.970867 | 11.78 | 0.014077 | 83.36% | 0.00% | 29.1 |
| D：硬路径，λ=0 | 32.8935 | 0.969532 | 30.85 | 0.015479 | 80.90% | 0.00% | 113.1 |



相同 λ=0.001 下，C 比 A 的测试质量更好，训练图像目标也更低，因此优势不只体现在 PSNR。λ=0 下，D 比 B 的平均 PSNR 高 0.2109 dB，但只有 5/10 帧更高，不应把这个较弱对照描述为逐帧一致优势。

### 每帧最终 PSNR

| 帧 | REF（不同预算） | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 34.2636 | 33.6581 | 33.6516 | 34.1549 | 34.2244 |
| 2 | 34.2554 | 33.4001 | 33.4879 | 34.1284 | 34.1511 |
| 3 | 34.2747 | 30.3591 | 30.6980 | 33.9954 | 33.1618 |
| 4 | 34.3576 | 31.2708 | 31.3656 | 33.0926 | 31.4571 |
| 5 | 34.4741 | 32.2084 | 32.7597 | 33.4005 | 32.4666 |
| 6 | 34.2409 | 32.6842 | 33.1080 | 33.5005 | 32.3731 |
| 7 | 34.4019 | 32.9354 | 33.3111 | 33.4886 | 32.3958 |
| 8 | 34.3717 | 32.8693 | 32.4670 | 33.4037 | 33.1786 |
| 9 | 33.8565 | 32.5056 | 32.6978 | 32.8152 | 32.5594 |
| 10 | 34.5132 | 33.0851 | 33.2794 | 33.1475 | 32.9668 |



### 同帧阶段变化

每项先计算同帧后一阶段减前一阶段，再对 10 帧平均。训练图像目标下降才是改善，PSNR 上升才是改善。

| 组 | 删除 ΔPSNR | 运动 ΔPSNR | 运动 Δ训练图像目标 | 运动令训练目标变差 | 候选拟合 ΔPSNR | 最终预算 ΔPSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A：全 SPLIT，λ=0.001 | +0.000000 | -0.066090 | +0.000135 | 4/10 | +0.010502 | -0.000001 |
| B：全 SPLIT，λ=0 | +0.000000 | -0.038588 | +0.000017 | 5/10 | +0.006898 | -0.000147 |
| C：硬路径，λ=0.001 | +0.000000 | -0.020798 | -0.000203 | 3/10 | -0.000163 | +0.000000 |
| D：硬路径，λ=0 | +0.000000 | -0.048480 | -0.000057 | 2/10 | +0.000744 | +0.000000 |



第 3 帧尤其有信息：A 从运动前 33.3643 降到 30.3485 dB，训练图像目标增加 0.002745；C 从 34.0823 到 33.9953，训练目标增加仅 0.0000667。这支持当前全 SPLIT 路径在本条件下存在优化风险。后续帧输入状态已经不同，所以这也是跨帧闭环效果，不能当作四组第 3 帧都从同一状态启动的瞬时消融。

硬路径仍有失败：C 第 4 帧运动阶段下降约 0.7364 dB；D 第 4 帧下降约 1.6015 dB，训练目标也变差。实际 motion Jacobian 并没有自动保证整个 gate＋motion 更新是图像目标的下降步骤，因为 gate 自身仍使用代理并会切换硬选择。

四组全部训练相机和 cam00 测试视角的删点前后 PSNR 差均为 0（10 帧×21 视角×4 组）；最终预算损伤也很小。因此本轮的主要差异不能由这两项解释。补点净收益很小，C 甚至略为负；“新点数量够了就能修复运动错误”仍无支持。

### 各类硬门参数字节估算

| 组 | gate/KiB | 区域运动/KiB | 逐点运动/KiB | 新点/KiB | 删除/KiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| A：全 SPLIT，λ=0.001 | 5.352 | 101.810 | 502.254 | 14.906 | 0.543 |
| B：全 SPLIT，λ=0 | 6.878 | 139.678 | 1822.024 | 18.750 | 0.703 |
| C：硬路径，λ=0.001 | 1.196 | 7.760 | 0.000 | 2.728 | 0.095 |
| D：硬路径，λ=0 | 1.738 | 18.113 | 0.000 | 10.603 | 0.395 |



C/D 的低估算主要来自零逐点残差开放与较少有效区域运动。KEEP 是门的决定，不是背景标签，也不等于像素不会受其他移动点影响。C 的新点平均只有 29.1 个，未强行补足 150–200；这是上限规则的实际结果，可能同时反映较少运动残差梯度与候选淘汰，不宜单独归因。

![四组逐帧画质、参数估算、训练目标和运动阶段变化](../../output/gradient_ablation_20261002/summary/ablation_curves.png)

图中参考为不同预算的已有结果；参数轴采用对数刻度，展示估算范围。

## 6. 梯度诊断与证据边界

首帧第 1 步所有门初始化为 SHARE。A/B 的逐点运动梯度已有非零行，C/D 为 0；C/D 的 gate 梯度仍存在。这个实测结果与数值测试相符：hard_path 没有误把 gate 梯度关闭，而是在关闭的运动路径上不给候选更新。完整采样见 [gradients.csv](../../output/gradient_ablation_20261002/summary/gradients.csv)。

两组硬路径最终逐点开放率均为 0。它们仍可打开区域子层，不能说整个树固定在最粗层；但最末逐点残差没有得到启用。本轮不能区分“逐点候选原本不需要”与“关闭候选未训练，所以 gate 没有看到它的收益”。下一轮应优先检查这个探索瓶颈。

四组只各跑一次 seed0。相同 seed 和相机序列不保证 GPU 训练逐位确定：尚未启用 storage 的首帧 step30，A/B 已分别为 34.234051/34.238819，差 0.004768 dB；C/D 为 34.221546/34.224419，差 0.002872 dB。没有通过重复实验估计方差，小差异不要作强因果解释。模型随机数仍会受各组候选数量影响；我们保证的是初始 seed 与独立相机序列，不是不同候选集合的一一随机匹配。

本次只有 flame_steak 的前 10 帧、一个测试相机，没有额外背景 mask、独立实际 codec 或晚期验证。不能宣称论文整体假设通过，不能宣称长期消除误差累积。

## 7. 本轮决定与下一步

1. 保留本轮四组作为有效失败/改善证据。生产默认暂不自动切换；当前 all_split＋storage 的问题值得优先处理，不能只靠扩大补点。

2. 上述 warmup 候选已按建议完成，不应重跑。相同起点与相机轨迹下，control（hard_path、warmup=0）5 帧平均 PSNR/SSIM 为 33.8493 dB/0.972117，`hard_storage_bytes` 均值 13,146 B；warm50（前 50 步 all_split）为 33.0317 dB/0.970921，估算 193,209 B。差值为 −0.8175 dB，硬存储参数估算扩大 14.70 倍；warm50 每帧 GS split 数为 3,456、3,901、3,889、5,804、5,634。这里的字节数是参数存储估算，不是实际输出文件或压缩码流体积。该预热路径在本组设置下失败：它开出了更深的运动行，却显著降低画质并大幅增加估算存储；不能据此否定自适应层级本身。

   下一步转为补齐 ReCon-GS 基线的真实动态码流：复用已有训练好的 frame-0 checkpoint，只训练并保存 frame 1–299 的原生 delta/index、增量点云和精确转移元数据，以实际文件、可重放性和同口径计时评估当前方法的收益。该基线流此前缺失，不能用 checkpoint 或参数估算代替。

3. 保留 storage 机制继续验证。此次关闭它在 hard_path 下同时损害质量并增加估算参数，不能据此推荐删除 storage。它的收益是否稳定，需要后续受控重复验证。

4. 进入论文收益验证前，增加相同预算、量化与补点规则下的固定分层/全开放对照，以及晚期共同状态窗口与连续长序列。第 199 帧公平起点尚未保存；重建历史流需旧版匹配 decoder，当前代码不支持旧 codec。不能用本轮前缀代替晚期稳定性。

5. attribute-specific gate 暂缓，符合 AGENTS 的研究顺序。本次未追加以上实验。

## 8. 命令、产物与重算

环境为 conda ReConGS，3090 Ti；四组顺序执行，总耗时约 26 分 33 秒，各组平均 wall 约 38.8–39.9 秒/帧。新增全视角诊断已计入相关训练/准备/状态更新时间，不与历史无诊断训练做速度排名。机器日志显示本地时间 16:02–16:29，manifest 明确记录 UTC 08:02–08:29。

```bash
conda activate ReConGS
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python tests/test_nested_prototype.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests/test_experiment_inventory.py
```

实际训练命令如下，均已执行。历史身份仅用于追溯；后续实验应使用新 experiment_name。

```bash
python main.py --config=config/fs_gradpath_all_split_storage001_seed0_11.yaml
python main.py --config=config/fs_gradpath_all_split_storage0_seed0_11.yaml
python main.py --config=config/fs_gradpath_hard_path_storage001_seed0_11.yaml
python main.py --config=config/fs_gradpath_hard_path_storage0_seed0_11.yaml
```

结果汇总可只读源运行后重算到本轮 summary：

```bash
MPLCONFIGDIR=/tmp/my4dgs-mpl-cache python output/gradient_ablation_20261002/summarize_gradient_ablation.py
```

| 产物 | 内容 |
| --- | --- |
| [manifest.json](../../output/gradient_ablation_20261002/manifest.json) | 版本、源 SHA、首帧 SHA、命令、完成范围、相机 trace SHA |
| [summary.csv](../../output/gradient_ablation_20261002/summary/summary.csv) | 四组均值与诊断计数 |
| [frames.csv](../../output/gradient_ablation_20261002/summary/frames.csv) | 40 行逐帧最终指标、阶段变化和各类字节 |
| [stages.csv](../../output/gradient_ablation_20261002/summary/stages.csv) | 400 行：四组×10帧×5阶段×train/test |
| [gradients.csv](../../output/gradient_ablation_20261002/summary/gradients.csv) | 各采样步和参数组梯度 |
| 各运行 `frame_summary/control_NNN.json` | 全部相机逐视角阶段指标与梯度记录 |
| 各运行 `training_views_NNN.json` | 每帧300步完整相机顺序 |
| 各运行 `quantized_frame_NNN.json`、`evaluation_NNN_stepS.json` | 门、补点、删除、存储与正式阶段评估 |

原始 checkpoint、图像、日志、快照和汇总 CSV/图都保存在本地 output，不纳入 Git。本记录、代码和实验配置保留为可审核工作区改动；没有提交或推送。
