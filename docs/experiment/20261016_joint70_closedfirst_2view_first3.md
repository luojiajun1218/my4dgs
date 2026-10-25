# 首个动态帧关闭 gate，后续继承 gate：两视角短流程核验

按新的初始化要求完成共享首帧加两个动态帧的 GPU 训练及独立码流核验。实际新训练 frame 1/2，各 200 步，没有执行完整 300 帧。

## 想法与改动

首个动态帧不再强制将所有细分 gate 打开，而是从 8 个大区域开始，让联合优化决定是否细分。后续帧继续继承上一帧完整运动、区域参数、gate 开关与 logits，作为初始化；没有增加额外预热步骤。

`pipeline/model/spatial_gated_motion.py::_begin_joint_frame` 去掉硬编码的正 logits 和全开掩码，直接根据配置 `initial_logit` 的符号生成首帧硬 gate。当前值为 **-2.0**，所以首个动态帧细分 gate 全关；正值仍可配置为全开。根节点是进入 8 个初始区域的固定入口，不属于待学习的细分开关。

`config/fs_spatial_gate_depth4_200_full300_20261016.yaml` 使用新的实验名 `fs_spatial_gate_joint70_closedfirst_robustnearest_depth4_full300_20261016`。其余约定保持：robust_median、远点最近非空叶路由、深度 4、storage 权重 0.002；1–70 步联合训练 gate/区域运动/GS 残差，71–100 步固定 gate 继续训练运动，每步两个视角；101–200 步新增 GS，每步一个视角。

更新 `tests/test_joint_spatial_motion.py`，验证负 logits 从 8 个区域开始、正 logits 仍能全开、开关双向切换时姿态不跳变，以及含新增点的下一帧同时继承正负 gate logits。该文件 9 项检查全部通过；配置 `--print` 和 `git diff --check` 通过。本次没有重跑整个测试集。

## 已完成结果

RTX 3090 Ti，350 W，随机种子 0，1600×1199，复用相同原生首帧。独立进程从参考和增量解码，逐帧验证状态哈希；动态帧 PSNR/SSIM/L1 与训练保存值差异均为 0。

| 动态帧 | PSNR / dB | 运动包 / B | 新增属性 / B | 完整更新 / B | 训练报告耗时 / 秒 | 实际共享区域 | 有非零局部残差的 GS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.4453 | 164235 | 1872 | 166542 | 23.3018 | 8 | 52328 |
| 2 | 34.4013 | 106643 | 2006 | 109084 | 22.3568 | 8 | 33905 |
| 平均 | **34.4233** | 135439 | 1939 | **137813** | **22.8293** | 8 | — |

完整更新包括运动、新增属性和每帧 435 B 的 manifest，不含一次性共享参考 17720236 B。KB 使用十进制，平均为 **137.81 KB/动态帧**。共享首帧 PSNR 为 34.0113 dB；含首帧的三帧平均为 34.2860 dB。

诊断确认 frame 1 从 `all_eligible_closed` 开始，frame 2 为 `previous_frame`，运动初始化为 `previous_full_increment`。两帧结束时实际 owner 都停留在第 1 层的 8 个区域。原始 `open_gates` 计数分别为 130/261：这些是上层 gate 尚未打开的下层开关，不代表已经启用了 130/261 个共享区域。区域使用情况应看实际 owner/frontier。

每帧 10 次预热、50 次同步计时完整渲染，动态帧纯渲染合计 201.93 FPS；不包含文件读取、CPU 解码与验证、上传。训练耗时包含本次评估及选定 checkpoint 写入，不作为严格提速对照。

## 对照已有短测与分析

与 [此前全开初始化短测](20261016_joint70_robustnearest_2view_first3.md) 使用相同首帧、相机顺序、训练超参数和两动态帧范围；配置的 trainer/module/data/seed/首帧 checkpoint 已逐项核对相同。运动实现仅修改首帧 gate 初始化及对应诊断。

两动态帧平均 PSNR 从 34.4122 到 34.4233 dB，完整更新从 139720.5 到 137813 B，约减少 **1.37%**。活动区域从约 4050 降到 8，但有非零残差的 GS 从 50382/29590 增加到 52328/33905，因此没有出现明显的存储下降。这个结果表明，在这两帧上仅改变 gate 的初始开关不足以解决约 140 KB 的更新量；GS 残差仍是重要开销。

本次仅验证用户指定的新初始化及其短期行为，没有据此修改 storage 权重、量化步长或训练调度，也不能据此证明 300 帧最终画质或压缩达标。此前全开短测曾并行运行 CPU 回归，本次没有，因此不能将两次训练耗时差全部归因于区域数变化。

## 证据

配置、逐帧摘要和区域诊断、独立 benchmark、日志、定向检查记录、初始化代码差异与关键源码快照见 [轻量归档](../archive/joint70_closedfirst_2view_20261016.tar.gz)，不包含模型 checkpoint、大型共享参考或训练数据。

原始临时结果：`/tmp/my4dgs_joint70_closedfirst_2view_20261016/joint70_closedfirst_2view_first3/20261016T232233_9c1493`。复现训练可使用归档中的短测配置；独立核验命令如下，输出文件应选择尚不存在的路径：

```bash
conda activate ReConGS
python scripts/benchmark_rigid_residual_tree.py \
  --run=/tmp/my4dgs_joint70_closedfirst_2view_20261016/joint70_closedfirst_2view_first3/20261016T232233_9c1493 \
  --sequence --mode=stream --sequence-warmups=10 --sequence-renders=50 \
  --output=/tmp/my4dgs_joint70_closedfirst_2view_20261016/benchmark_new.json
```
