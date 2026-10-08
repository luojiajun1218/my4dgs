# 父精度变化后的子残差几何补偿与真实码流审计

## 要解决的问题

父节点换一个量化 bit 后，它的累计旋转和平移会改变。当前 child residual 仍按旧父节点定义，因此整棵已开启子树的姿态也会一起偏移。本实验检查一个解析补偿是否能把原 source 的已量化累计姿态重新表达到 immediate children，并检查补偿后的真实 `motion.bin` 成本。

这是 CPU 几何与 codec 实验。没有训练、渲染或 PSNR 评测，也没有接入生产默认路径。

## 方法

当前累计变换为：

```text
q_c     = q_local_c * q_p
delta_c = rot_delta(q_p, pivot_c - pivot_p) + delta_p + t_c
```

父节点量化后变为 `q'_p, delta'_p`。对每个实际 occupied immediate child，设置：

```text
q'_local_c = q_c * inverse(q'_p)
t'_c       = delta_c - delta'_p - rot_delta(q'_p, pivot_c - pivot_p)
```

未再次量化时，这个变换保持 child 的累计姿态。实验随后使用 child 原来的 hard precision 重新量化 `q'_local_c, t'_c`。没有向 grandchildren 递归下推误差。

STOP 父节点没有可达 child，不能用这个办法补偿。因此本实验只使用原诊断采样中的 `reachable_open` 节点。

## 固定协议

- 来源：`fs_joint_precision_mechanism_100f/20261007T235547_f9a241` 的 12 个不可变 snapshot。
- frame：`1, 5, 25, 50, 75, 99`。
- snapshot step：`120, 180`。
- 每个 snapshot 使用原始、image-blind 的 3 个 `reachable_open` sample，共 36 个节点；不按收益筛选。
- 32 个节点的 source parent 为 32 bit，4 个为 12 bit。
- 每个节点比较 source 之外的 4 种 parent bit，共 144 组。其中 132 组降低 parent bit，12 组提高 parent bit。
- 每组从 source snapshot 独立恢复 `naive` 和 `compensated` 两个候选。
- gate 与所有 precision argmax 在 snapshot step 显式冻结；forward、writer 和 decode 统一使用会读取 hard mask 的 `config.motion_steps`。
- `naive` 只换 parent bit。`compensated` 再重表达 occupied immediate-child residual；更深 residual、gate 和其他 precision 不变。
- 目标是 source 经过生产 writer 解码后的姿态，不是未量化 raw pose。
- 每个候选调用现有 `StorageAwareBinaryMotion.export_motion` 和 `decode_motion`，没有用自写 bit 公式替代真实 writer。

配置：`config/fs_parent_precision_compensation_codec_audit_36open.yaml`

脚本：`scripts/audit_parent_precision_compensation_codec.py`

结果：`output/fs_parent_precision_compensation_codec_audit_36open/20261008T120545`

## 结果

下表只在被选 parent 的 GS 范围内汇总。全 GS 均值会被大量不变点稀释，因此不作为主要几何指标。

| 指标 | naive | compensated | 相对下降 |
|---|---:|---:|---:|
| position mean | 9.582e-4 | 1.100e-5 | 98.85% |
| position RMS | 1.793e-3 | 6.171e-5 | 96.56% |
| position max | 1.412e-2 | 5.819e-4 | 95.88% |
| rotation mean | 5.507e-4 rad | 1.112e-5 rad | 97.98% |
| rotation RMS | 1.033e-3 rad | 7.331e-5 rad | 92.90% |
| rotation max | 3.440e-3 rad | 6.162e-4 rad | 82.09% |

144 组中，补偿后的 position mean 全部更低。rotation mean 有 140 组更低，其余是 3 组相等和 1 组两边均为零。对最激进的 target 8 bit 共 36 组，parent 区域 position mean 从 `3.279e-3` 降到 `9.614e-6`，rotation mean 从 `1.639e-3` 降到 `1.595e-6 rad`。

补偿前、child 尚未重新量化的连续几何检查中，最大 position 残差为 `7.45e-9`，最大 rotation 残差为 `1.67e-9 rad`。这验证了解析式和当前旋转乘法约定。最终剩余误差来自 child 原 bit 的重新取整。

所有 naive 和 compensated 变化都局限在对应 parent 的 GS 范围；范围外 quaternion/XYZ 分量逐值不变。production writer 解码 pose 与对应 candidate forward 在全部 300 个 `motion.bin` 上逐值一致，position 和 rotation 检查的最大差均为零。

真实 whole-tree `motion.bin` 结果：

- `compensated` 与 `naive` 在 144/144 组中 bytes 完全相同。
- 全部 144 组相对 source 的平均差为 `-11.160 bytes`，范围 `[-25, +15]`。
- 132 个降 bit 比较平均 `-12.992 bytes`，范围 `[-25, -3]`。
- 12 个升 bit 比较平均 `+9.000 bytes`，范围 `[+3, +15]`。
- 补偿改变了 276 个 child symbol row，但没有出现 child zero→nonzero、nonzero→zero 或 saturation。

这批样本中没有发生码率向 child 迁移。原因是 writer 对非零 motion 使用固定 bit 宽，symbol 数值改变不会改变长度；而这次重新取整没有改变 child 的 presence。这个结果不能推广到会跨过零桶或发生饱和的节点。

## 代码与测试

新增的 `parent_precision_compensation.py` 是离线 helper，不被训练或生产 export 默认路径调用。它包含 hard-choice 冻结、累计 node pose、immediate-child 覆盖检查、解析 rebase 和生产一致的 round/clamp helper。

`tests/test_nested_prototype.py` 增加了以下确定性检查：

- 解析补偿与当前 quaternion 顺序一致。
- 连续补偿在 production reference `_conditional_transform` 上保持全 GS pose。
- immediate children 必须完整覆盖 parent GS。
- STOP parent 明确拒绝补偿。
- 空 structural child 被保留但不写 residual；只补偿 occupied child。
- child 重新取整覆盖零桶与 saturation。
- gate-compress snapshot step 不能直接拿来消费 frozen precision，避免 hard mask 被阶段逻辑忽略。

验证结果：

```text
tests/test_nested_prototype.py: 195 checks passed
tests/test_experiment_inventory.py: 16 tests passed
tests/test_recongs_native_stream.py: 16 tests passed
```

## 解释边界

本轮 `motion.bin` 使用真实生产 writer 和 decoder，但 header 中 previous/post digest 是 `00` 占位，`deleted_count=0`、`birth_count=0`。它验证的是每个独立 snapshot 的 motion payload、实际文件字节和解码 pose。它不是可顺序播放的完整帧流，也不是 99-frame sequence decoder 审计。birth 数据固定，因此没有进入候选差值。

结果说明：固定 open 层级中，父 bit 改变后可以用现有 child residual 重表达 source 的已量化累计 pose；child 再量化后的误差明显小于直接换 bit，并且本样本集没有增加实际码流。它还没有证明渲染 RD 会改善，也没有让 gate 学会利用补偿。

HiFi4G 已使用 warped keyframe Gaussian 的 motion-compensated residual 再量化，因此“残差补偿”本身不能作为新颖性主张：[HiFi4G, CVPR 2024, Section 3.3](https://arxiv.org/html/2312.03461v2)。本轮只验证更窄的对象：固定 open 层级中，parent bit 变化后对现有 child 参数做条件重表达。是否形成论文机制仍需要后续的统一候选、视角和最终兑现协议验证。
