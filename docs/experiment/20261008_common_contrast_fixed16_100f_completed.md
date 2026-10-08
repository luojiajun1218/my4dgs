# 共同运动—单对比树固定 16 bit：100 帧端到端实验

## 要解决的问题

原来的树上每个节点都有一份独立运动。子层开启时，父节点和子节点可以重复表达共同运动，gate 截断子树时，也不容易直接解释它省掉的是共同运动还是局部差异。

本轮把运动改成两类参数：每棵树的 root 保存共同刚体运动；每个有效二叉 split 只保存一份左右分支的 motion contrast。subtree gate 仍是可学习参数，关闭后截断整棵后代。这样 gate 直接决定是否为一整棵子树继续支付局部差异的码流成本。

这是第一个完整端到端版本。它包含训练、保存、恢复、真实 writer、连续 decoder、删除/新增 GS、评测和 100 帧实验。motion 固定为 16 bit，本轮没有学习 precision，也没有 attribute gate。

## 方法

对一个左右子树成员权重为 `w_left + w_right = 1` 的 split，只学习一份平移对比 `tau` 和旋转向量对比 `omega`：

```text
t_left  = -w_right * tau       r_left  = -w_right * omega
t_right =  w_left  * tau       r_right =  w_left  * omega

Q_child     = Exp(r_child) * Q_parent
delta_child = (R_parent - I) * (pivot_child - pivot_parent)
              + delta_parent + t_child
```

公式使用完整 pivot 项，不假定缓存的 parent pivot 恰好等于当前成员的加权重心。只有一个 occupied child 的 unary 分支直接继承，不消耗 contrast 自由度，也不写 split record。左右派生的平移和旋转向量加权和在数学上为零，数值检查使用明确的浮点容差；旋转的“共同量 + 唯一局部差异”解释限于选定的 principal-log 分支、相对角小于 `pi` 的区域。生产参数不额外限制 `omega` 范数，因此不能把这个局部解释扩展到全部 SO(3) 状态。

每个有效 split 有一个 subtree gate。hard gate 关闭时，所有后代 contrast 都不可达；当前 split 的 contrast 为零时，gate 仍可开启以访问更深层差异。gate 的训练使用 hard-forward STE，实际 forward 严格得到 0/1。storage proxy 按 root、gate presence 和实际可达 split payload 计费；hard 状态下与 writer 的实际文件成本逐帧一致。

固定量化参数为：

- motion：16 bit；
- translation step：场景 extent 的 `1e-5`；
- rotation step：`1e-4 rad`；
- storage loss 权重：`0.01`；
- 每个动态帧 300 步：100 步 dense fit、80 步 gate compress、20 步 fixed-gate refine、100 步 incremental；
- 训练视角 seed 为 0，分辨率为原协议 `resolution: -1`。

这些值是首版工程设置，不是扫描得到的最优值。

writer 使用有符号 16 bit symbol，取值范围为 `[-32768, 32767]`。因此平移 contrast 的可表示范围约为 `[-0.32768, 0.32767] * extent`，旋转 contrast 每个分量的范围约为 `[-3.2768, 3.2767] rad`，超出后会 clamp。每个旋转分量的范围略跨过 `pi`，向量范数可以更大，所以 codec 可正常解码不代表每个量化状态都仍处于唯一 principal-log 分支。

## 代码改动

- `pipeline/ReCon-GS/model/common_contrast.py`：新 motion backend、批量树 forward、分解/重建数学 helper、fixed16 量化、capture/restore、真实 writer 和 decoder。
- `pipeline/ReCon-GS/module.py`、`trainer.py`：接入训练阶段、删除/新增 GS、layout 复用与重建、checkpoint、manifest、实际成本和导出生命周期。
- `pipeline/ReCon-GS/model/storage_aware_binary.py`：按 codec magic 分发新旧 decoder，历史 SARB 流保持可读。
- `scripts/train_common_contrast.py`：独立正式训练入口，不恢复已删除的旧 `main.py`。
- `scripts/audit_common_contrast_sequence.py`：从第一帧开始连续消费 99 个 motion/birth 流，并把最终状态与 `ckpt-99-300.pth` 比较。
- `config/fs_common_contrast_fixed16_100f.yaml`：learned-gate 配置。
- `config/fs_common_contrast_fixed16_allopen_baseline_100f.yaml`：同表示、同固定 16 bit、同训练预算的 all-open 对照。除实验名和 `force_all_gates_open` 外，两份计划配置一致。

训练会把配置和 9 个实际依赖源码复制到各自的 `source_snapshot`。两臂的训练源码哈希逐项相同；配置差异已限制为实验名和 all-open 开关。99 帧训练相机序列也逐帧相同。

## 测试与 smoke

正式训练前通过：

```text
tests/test_nested_prototype.py: 206/206 passed
tests/test_experiment_inventory.py: 16/16 passed
tests/test_recongs_native_stream.py: 16/16 passed
```

数值测试覆盖非均衡权重、非重心 pivot 的任意 terminal pose 分解/重建、共同世界刚体的零 contrast、浅层零 contrast 与深层非零差异、ancestor 关闭后的梯度、空/单子分支、quaternion 符号和 near-pi 边界、量化后的加权零和、真实 writer/decoder、checkpoint 的 Adam moments 和恢复后的下一步更新。

第一次 smoke `output/fs_common_contrast_fixed16_smoke_2f/20261008T123809_5755b7` 在训练前因配置缺少 `initialization_decay` 退出，没有产生实验结果。补齐配置后，`output/fs_common_contrast_fixed16_smoke_v2_2f/20261008T123850_725435` 正常 exit 0，完成一个动态帧，并验证 CUDA 路径的 writer、dispatcher 和 checkpoint reference 逐值一致。

## 完整实验

learned-gate：

```text
output/fs_common_contrast_fixed16_100f/20261008T124108_2dea25
```

all-open：

```text
output/fs_common_contrast_fixed16_allopen_baseline_100f/20261008T132448_66ad5e
```

两臂均完成 99/99 个动态帧并 exit 0。成对汇总及可重跑脚本位于：

```text
output/fs_common_contrast_analysis_20261008/paired_v1
```

其中 `summary.json`、`learned_frames.csv`、`all_open_frames.csv` 和 `paired_frames.csv` 保存完整数值，`source_sha256.json` 保存输入、源码快照和关键产物哈希。

## 结果

下表的 PSNR/L1/SSIM 是每个动态帧最终的 held-out `cam00` 评测均值。必要字节为该帧实际 `motion.bin + incremental birth PLY`；motion header 包含点数、删除数和拓扑校验；布局与删除行由 decoder 确定性重算，不传逐节点或删除行 ID。共享首帧、外部 manifest/metadata、checkpoint 和诊断文件不计入。RD 目标使用训练协议的 `image objective + 0.01 * bytes / reference_bytes`。

| 指标 | learned gate | all open | learned - all open |
|---|---:|---:|---:|
| cam00 PSNR | 34.044563 dB | 34.047302 dB | -0.002739 dB |
| cam00 L1 | 0.01450047 | 0.01445404 | +0.00004642 |
| cam00 SSIM | 0.96977464 | 0.97011394 | -0.00033930 |
| test RD objective | 0.01772024 | 0.01793600 | -0.00021575 |
| motion bytes / frame | 34,983.71 | 200,555.88 | -165,572.17 |
| birth bytes / frame | 3,611.24 | 3,543.98 | +67.26 |
| necessary bytes / frame | 38,594.95 | 204,099.86 | -165,504.91 |
| 99 帧 necessary bytes | 3,820,900 | 20,205,886 | -16,384,986 |
| pure train median / frame | 15.734 s | 16.150 s | -0.416 s |
| wall median / frame | 22.184 s | 23.060 s | -0.875 s |

learned gate 使用 all-open `18.91%` 的必要字节，减少 `81.09%`。平均 cam00 PSNR 只差 `-0.00274 dB`，但逐帧差值中位数为 `-0.008522 dB`，只有 42/99 帧为正。最差是 frame 82 的 `-0.247845 dB`，最好是 frame 60 的 `+0.286861 dB`，不能把均值接近零理解为每一帧都等价。

由于实际存储项大幅下降，learned 的平均 RD 目标比 all-open 低 `2.16e-4`。这里定义 `J gain = J_all-open - J_learned`，正值表示 learned 较好；83/99 帧为正，最差是 frame 88 的 `-1.154144e-4`，frame 82 为 `-7.565243e-5`。必要字节在 99/99 帧都更低。

为检查均值是否只来自某一段，按预先固定的三个等长区间做描述性汇总：

| frame | mean PSNR gain | mean J gain |
|---|---:|---:|
| 1–33 | -0.035358 dB | +2.443007e-4 |
| 34–66 | +0.083611 dB | +3.133623e-4 |
| 67–99 | -0.056470 dB | +8.959543e-5 |

三个区间的 J 均值都改善，但首尾两段的 PSNR 均值下降。这是固定分段的描述，不是挑选窗口，也没有统计显著性估计。

![learned gate 与 all-open 的逐帧成对结果](../../output/fs_common_contrast_analysis_20261008/paired_v1/paired_results.png)

结构上，all-open 每帧固定开启 14,144 个有效 split，得到 16,384 个 terminal group。learned 平均开启 604.79 个 split，得到 2,844.79 个 terminal group。learned 的 level-0 gate 每帧都以 2,240 个 root gate 为可达分母；更深层只统计 ancestor 已开的可达 gate，没有把不可达后代混入分母。

两臂的 actual writer bytes、hard storage accounting 和 inference bytes 在全部帧逐值一致。learned 的训练中曾有一次独立 CUDA 审计与主进程短暂重叠，主要影响约 frame 9/10 的 wall time；因此这里报告中位数，不把个别 wall outlier 解释成机制开销。

## 顺序解码审计

两臂都从 frame 1 开始连续解码到 frame 99，覆盖 cached layout、定期 rebuild、删除和 newborn 续接。最终解码状态与各自 `ckpt-99-300.pth` 的以下六个原始字段逐值相等：

```text
xyz, rotation, logscale, opacity, sh_dc, sh_rest
```

审计文件：

```text
learned:
output/fs_common_contrast_fixed16_100f/20261008T124108_2dea25/
sequential_decode_audit_frame001_099.json

all-open:
output/fs_common_contrast_fixed16_allopen_baseline_100f/20261008T132448_66ad5e/
sequential_decode_audit_frame001_099.json
```

两份解码审计的 `final_state_match=true` 且 `final_fields_exact` 全部为 true；独立汇总据此记录 `final_checkpoint_match=true`。all-open 的解码终态和 checkpoint SHA-256 都是 `aff02494e942082c4f8c1cc13b16743cfe716e6b15862925f844838e6a90159e`；learned 也通过同样的六字段和终态哈希检查。

严格逐 bit/逐值的完整审计目前只在与训练相同的 CUDA 数值路径上验证。smoke 的 CPU replay 在浮点 kernel 低位出现 digest 差异，因此目前不能声称跨 CPU/CUDA 的 bit-exact 可移植解码已经验证。这不影响本轮同 CUDA 路径的 99 帧一致性结论，但属于 codec 的实际限制。

## 如何解释

这组受控比较支持一个窄结论：在同一个 common+contrast 表示、同一个 fixed16 codec 和相同训练协议内，可学习 subtree gate 找到了明显的存储—质量取舍。它把必要字节降到 all-open 的约五分之一，平均 held-out PSNR 变化很小，并改善了含真实存储项的 RD 目标。

它没有证明 common+contrast 表示优于旧的独立 parent/child 参数，也不能拿旧 mixed-precision 或 native run 宣称公平压缩比。all-open 对照只回答“新表示内部 gate 是否有用”。共同量—对比量的临界参数化与树 wavelet/lifting、HEIR 的 parent inheritance/local residual、Temporal Smoothness-Aware RD 4DGS 的时域 Haar+RD，以及 ReCon-GS 的层级 anchor 思路都有相邻关系；本轮不据此宣称新颖性。相关工作边界可参考 [Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](https://arxiv.org/html/2507.17336v1)、[HEIR](https://arxiv.org/html/2510.26786v1) 和 [ReCon-GS](https://arxiv.org/html/2509.24325v2)。

此外，本轮只有 flame_steak 一个场景和一个训练 seed，没有显著性估计；16 bit 是首版固定选择；precision learning 和 attribute gate 尚未接入。下一阶段若要形成论文证据，首先需要与旧 parent/child 表示的同协议对照和多场景复现，而不是用本轮 learned-vs-all-open 结果替代这些证据。

最直接的下一项核心证据有两个部分：先做 matched 的旧 independent parent/child 表示对照，检验 common/contrast 分工本身；再用已知运动构造受控 gate 学习案例，特别覆盖“浅层 contrast 为零、深层 contrast 非零”的 zerotree 情况。现有数学测试和单 gate 梯度测试只证明信号存在，不能替代多层 gate 联合训练正确性的验证。下一步不需要先扫描 PSNR 权重或 bit 数。

本轮已落盘 summary 的 `gs_activity_cost` 文字仍沿用历史 `m-gate` 说法，但 common+contrast backend 没有 per-GS `m` gate。实验完成后已修正文案，不改变训练或数值结果。实际可用的字段是独立的 hard-pose change 诊断；root 的共同运动可以让大量 GS 都发生非零 pose 变化，不能据此解释为“整场景物体都在运动”或新的 activity-gate 证据。
