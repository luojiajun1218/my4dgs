# 2026-10-08 批量 shadow frontier 六帧实验（完成）

## 要解决的问题

前一轮逐节点实验要对每个节点跑 32 个候选和双折续训。六个节点仅搜索就用了 678 秒，且 f25、f99 的末期变化小于一次无干预独立重放差异。这个成本和可靠性都不适合接入正式训练。

本实验检查一个更便宜的机制：在 gate 学习阶段，用少量额外全图 backward 给当前关闭分支的后代 motion 行训练机会。主 gate、storage loss 和 precision 仍按原训练更新，不用交叉熵硬推某个候选。

## 实现

代码入口：

- `pipeline/ReCon-GS/model/storage_aware_binary.py`：生成当前可达、关闭、occupied 的同层 frontier。shadow 只强开 frontier；更深 gate 保留源 hard 决策。训练目标是 `shadow_paths & ~source_paths & occupancy`，不包含 frontier 父行。
- `pipeline/ReCon-GS/model/shadow_exploration.py`：独立 Adam，只更新目标 motion 行，并清零目标外梯度和 Adam moments。precision 使用当前 hard argmax，只向 motion 传梯度。
- `scripts/continue_shadow_frontier_pair.py`：从同一 step-120 runtime 深拷贝 source/shadow 两臂，执行受控续训，并逐事件检查参数和优化器隔离。
- `config/fs_shadow_frontier_prehardening_6frames.yaml`：保存全部实验参数。

shadow 在 steps 120, 126, ..., 174 运行，每帧 10 次，depth `[0,1,2]` 轮转。camera 使用 seed 0 的独立均衡序列且不含 cam00。每次 shadow 后重新计算正常主 forward/backward，不复用旧梯度。六帧为 1、5、25、50、75、99，均从 step 120 续到 200。cam00 只在末尾评估。

## 完整性检查

完整 run：`output/fs_shadow_frontier_prehardening_6frames/20261008T114000`

- 6/6 帧完成，进程退出码 0，耗时 113.85 秒。
- 60/60 个计划事件都有非空 frontier，并实际执行 backward。
- 每个事件的目标外参数、主 motion optimizer、主 gate optimizer 和关闭 frontier 下的主 hard pose 都逐值相等。
- 初始 source/shadow runtime 六帧均逐值相等。
- 11 个实际依赖源码文件已快照并校验哈希；analysis `issues=[]`。
- 测试：`tests/test_nested_prototype.py` 189 项通过；实验清单 16 项通过；原生数据流 16 项通过。

较早的 `20261008T113500` 因启动器提前创建目标目录而退出；`20261008T113600` 在补充机制指标后主动中止。两者都不是结果 run，目录和退出码保留。

## 结果

下表是 step 200 的 all20 训练相机结果。RD gain 为 source RD 减 shadow RD，正值较好。

| frame | RD gain | PSNR 差 (dB) | bytes 差 | cam00 PSNR 差 (dB) | 最终 hard gate 差 `[L0,L1,L2]` |
|---:|---:|---:|---:|---:|---:|
| 1 | -8.216e-6 | -0.007916 | -476 | -0.011375 | `[29,19,12]` |
| 5 | +6.000e-7 | +0.002111 | +103 | +0.003971 | `[12,11,11]` |
| 25 | +9.022e-7 | +0.000830 | +202 | -0.002174 | `[10,13,1]` |
| 50 | +1.283e-6 | +0.007179 | +660 | +0.002186 | `[13,17,7]` |
| 75 | +3.490e-6 | +0.004230 | +242 | -0.005417 | `[12,9,12]` |
| 99 | +1.766e-7 | +0.001808 | +516 | +0.004265 | `[7,11,14]` |

训练相机 RD 为 5 正 1 负，但均值为 -2.94e-7，主要被 frame 1 的退化拉低。cam00 为 3 正 3 负，平均 -0.001424 dB。结果不能表述为稳定提升。

shadow 确实改变了层级决策：每帧最终有 24 到 60 个 occupied hard gate 与 source 不同。checkpoint 记录的是各层 aggregate，包括 occupied/reachable motion 行、可达 open/closed gate 和 precision bit histogram。它能证明整体决策改变，但不能追踪单个 frontier 的决策轨迹。

### 完成后的只读 CPU 复盘

这里必须区分三种计数：frontier 是 gate；target 是后代 motion row；同一个 motion row 可以在多个事件中重复曝光。六帧共有 113,380 次 motion-row 事件曝光，其中 35,216 行是各帧内第一次曝光，78,164 次是重复曝光。按所选 frontier 层汇总如下。“归一化梯度”是每个事件合并 translation/rotation 后的 L2 除以目标行数平方根，只用于比较记录量级。

| frontier 层 | passes | gate-event 数 | motion-row 曝光 | 首次曝光 | 重复曝光 | 事件梯度 L2 中位数 | 归一化中位数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 24 | 50,876 | 108,340 | 32,006 | 76,334 | 1.115e-2 | 1.680e-4 |
| 1 | 18 | 1,419 | 3,384 | 2,178 | 1,206 | 1.704e-3 | 1.354e-4 |
| 2 | 18 | 828 | 1,656 | 1,032 | 624 | 5.267e-4 | 6.080e-5 |

原先的 3.02% 是“曾曝光的不同 motion row 与最终 shadow hard 路径的重合率”，不是算力利用率，也不等于 97% 的计算浪费。分层看，level 1 为 364/26,224（1.39%），level 2 为 354/4,650（7.61%），level 3 为 344/4,342（7.92%）。target 在创建时都 occupied；若最终不可达，表示最终路径上至少一个祖先 gate 关闭。但现有记录没有保存最终 row 身份，无法继续区分由哪一层祖先关闭。未进入最终路径的 residual 也可能在中间训练阶段影响 gate 梯度，所以不能把它们直接判作无效更新。

frame 1 的负例也不能由 3.02% 解释。它的最终重合率反而是六帧最高的 4.86%，总 row-event 数 19,332 也处于其他帧的 18,366–19,466 范围内。它在 level 1/2 的 gate-event 数较多：517/274，其他帧分别为 137–199/57–131；最终 hard gate 差为 60 个，也是六帧最大。末尾相对 source 少开 7 个 root gate，level 1/2/3 可达 motion row 分别少 14/4/6 个，同时少 476 bytes、all20 PSNR 低 0.007916 dB。这说明 frame 1 的端点是更强压缩伴随画质损失，但仍只是关联，不能证明是哪一批 frontier 导致。

只有 step 120 的第一次事件能与 pre-update 中同一相机、同一初态的主 hard loss 直接配对。六帧“批量强开 root 后的 loss − 主 hard loss”依次为 `+3.1645e-4`、`+2.2567e-4`、`-1.1877e-4`、`-4.8807e-5`、`+1.2259e-5`、`+7.3750e-5`；frame 1 最大。六个点不足以建立因果。即使同层 GS 子树结构上不重叠，它们在一次完整渲染中仍通过投影、遮挡和 alpha 混合相互影响。后续事件没有记录同状态 hard baseline，保存的 raw shadow loss 不能解释为干预增益或损失。

现有数据也不足以支持 uncertainty top-k。事件只保存 hard-closed frontier 身份、目标行和聚合梯度，没有保存当时 gate logit/probability；最终只保存 aggregate gate 差和可达行数，没有保存最终 gate/row 身份；loss 和梯度也没有拆到单个 frontier。step-120 snapshot 中的初始概率不能冒充后来九个事件的动态 uncertainty。因此无法匹配“当时 uncertainty → 后续开启/关闭 → 单独质量变化”，也不能从这六帧判断 uncertainty 是否优于随机或结构均衡选择。

## 判断

这个原型完成了最小机制验证：固定 10 个额外 backward 就能成批训练关闭候选，且不会直接改主 hard pose或污染主优化器状态；总运行时间也远低于上一轮 384 候选搜索。它没有证明质量收益。frame 1 同时省字节和明显损失画质；其余帧多为增加 103 到 660 bytes 换取很小训练 PSNR 变化，cam00 符号不一致。

现阶段唯一有记录支持、但仍待证伪的机制假设是：一次强开约 1,900–2,100 个 root frontier 使 shadow forward 偏离主 hard 上下文，批量分支之间的渲染交互会污染 residual 学习；在仍保持每帧 10 个 pass 时限制每个 pass 的干预规模，可能缩小同状态 shadow-vs-hard loss gap，并减少 frame 1 这类压缩增强但画质下降的端点。这个假设还没有被验证，也没有证据指定 uncertainty top-k 作为筛选规则。

限制 frontier 数量本身不会减少完整 render/backward 次数。若仍执行 10 个全图 pass，主要渲染成本基本不变；它改变的是干预上下文和目标行混合，除非同时减少 pass 或实现真正的稀疏渲染/反传，不能宣称自动节省算力。precision 仍只是原有学习路径，本实验没有解决 joint precision，也不支持相关创新声明。
