# 2026-10-08 bounded 25% shadow frontier 六帧实验（完成）

## 要解决的问题

上一轮每次 shadow 都强开同层全部 closed frontier。root 层一次会同时强开约 1,900–2,100 个 gate，完整渲染中的投影、遮挡和 alpha 混合可能让训练上下文偏离主 hard 路径。

本轮只改一个因素：每个事件从当前同层 `reachable + closed + occupied` frontier 中，用 seed 0 的私有随机数均匀无放回选择 `ceil(0.25 * N)` 个。`0.25` 是一次可证伪设置，不是调参得到的最优值。帧、快照、10 个触发步、层轮转、相机序列、LR、storage、precision 和 source 主训练均保持原值。

## 实现与诊断

沿用现有代码，没有再复制一套实验入口：

- `storage_aware_binary.py` 的 shadow plan 支持显式 frontier 子集，并检查子集唯一且都属于当前合法 frontier。
- `shadow_exploration.py` 增加 `fraction`。随机种子由 `seed/frame/step/level` 组成，不读 loss，不改变全局 NumPy 或 PyTorch RNG。采样后排序只用于稳定记录，不改变均匀无放回分布。
- `continue_shadow_frontier_pair.py` 仍运行 source/bounded-shadow 两臂。每个事件在更新前用同一状态、同一相机分别计算：主 hard loss、全 frontier 强开 loss、实际 25% 子集 loss。前两项是分开记录的 `no_grad` 诊断 forward；只有预先采好的子集 loss 反传。
- 事件记录完整候选和选中 gate position/node、选中时概率、target row IDs。末尾记录两臂 hard gate 和 reachable path IDs，方便身份级复盘。

配置为 `config/fs_shadow_frontier_bounded25_prehardening_6frames.yaml`。完整 run：

`output/fs_shadow_frontier_bounded25_prehardening_6frames/20261008T113319`

## 完整性

- 6/6 帧完成，进程退出码 0，耗时 115.81 秒，analysis `issues=[]`。
- 60/60 个事件实际执行 shadow backward；另有 120 次明确分账的 `no_grad` 诊断 forward。
- 53,116 次候选 gate-event 中选择 13,301 次。每个事件都满足 `ceil(0.25*N)`、无重复且属于候选集合。
- 每次事件的目标外参数、主 motion optimizer、主 gate optimizer 和主 hard pose 都逐值不变。
- 六帧初始 source/shadow runtime 逐值相等；11 个实际依赖源码快照哈希通过，历史 step-120 snapshot 哈希未变。
- 测试：数值检查 191 项、实验清单 16 项、原生数据流 16 项全部通过。新增测试覆盖固定随机、均匀无放回规则、ceil 数量、同层子树关系和空 frontier。

## 同状态上下文诊断

下表对每帧 10 个事件取平均绝对差。两项都相对事件更新前的同状态主 hard loss。

| frame | `abs(full - hard)` | `abs(25% - hard)` | 25% 更接近的事件数 |
|---:|---:|---:|---:|
| 1 | 4.955e-5 | 1.434e-5 | 9/10 |
| 5 | 5.046e-5 | 1.422e-5 | 8/10 |
| 25 | 6.391e-5 | 1.618e-5 | 10/10 |
| 50 | 3.035e-5 | 1.102e-5 | 8/10 |
| 75 | 1.948e-5 | 4.369e-6 | 7/10 |
| 99 | 3.470e-5 | 7.480e-6 | 10/10 |
| 六帧事件均值 | 4.141e-5 | 1.127e-5 | 52/60 |

25% 子集的 render loss 通常更接近主 hard 状态，平均绝对差降低约 72.8%。这只证明干预规模变小后即时输出更接近；少开 gate 天然会缩小差值。它没有单独证明批量分支之间的渲染交互，也不代表训练收益。

## 最终结果

RD gain 为 source RD 减 bounded-shadow RD，正值较好。所有结果都与本轮自己的 source 臂配对。

| frame | all20 RD gain | all20 PSNR 差 (dB) | bytes 差 | cam00 PSNR 差 (dB) | hard gate 差 `[L0,L1,L2]` |
|---:|---:|---:|---:|---:|---:|
| 1 | -3.148e-6 | -0.002141 | -571 | -0.003433 | `[16,15,12]` |
| 5 | +6.394e-7 | +0.003610 | +231 | +0.002159 | `[12,8,8]` |
| 25 | -4.390e-6 | -0.002210 | -404 | -0.009262 | `[7,6,2]` |
| 50 | +2.094e-6 | +0.000437 | +48 | -0.000900 | `[7,2,1]` |
| 75 | -1.549e-6 | -0.003155 | -555 | -0.003159 | `[3,5,4]` |
| 99 | +1.611e-6 | +0.000860 | +460 | -0.002174 | `[8,7,5]` |
| 均值 | -7.904e-7 | -0.000433 | -131.83 | -0.002795 | — |

训练 RD 为 3 正 3 负，均值和中位数都为负。cam00 只有 frame 5 为正，整体为 1 正 5 负。六帧合计少 791 bytes，但同时训练视角和 cam00 平均画质下降，不能把它解释成更好的压缩结果。

身份记录显示，六帧共有 9,289 个不同 gate 被选中过，其中 128 个在各自 bounded-shadow 末尾为 open；20,210 个不同 target motion row 被训练，其中 350 个末尾可达。这些是机制轨迹，不是好坏指标。关闭候选也可能在中途影响 gate 梯度，最终开启率不能作为优化目标。

上一轮 full-frontier 的训练 RD 是 5 正 1 负、均值 -2.94e-7，cam00 为 3 正 3 负、均值 -0.001424 dB。本轮 frame 1 的退步较小，但 frame 25、75 等没有改善，整体均值更差。两轮 source continuation 是独立 GPU 执行，source 自身存在数值分歧，因此跨 run 对比只能作为描述；本轮内部与 source 的配对结论已经足以说明收益不稳定。

## 判断

“限制单次干预规模会让 shadow 上下文更接近主路径”这一行为被观察到；“因此能让 gate 学得更好、改善最终 RD 或 cam00”没有成立。即时 loss 更接近不足以解释训练效果，当前证据也没有给出可用的候选排序规则。

本轮仍执行 10 个完整 shadow render/backward，并额外执行 20 个诊断 forward/帧。减少 target row 数没有减少主要全图渲染次数，也没有自动带来 render 提速。

因此停止继续扫描 fraction，也不把 frame 1 的局部改善包装成整体成功。当前 shadow 路线已经回答了批量曝光和干预规模两个问题；在有新的、可识别的决策信号之前，不继续堆同类变体。precision 仍走原训练路径，本轮没有解决 joint precision。
