# 联合反馈受控续训实验（已完成）

## 要解决的问题

上一轮已经在独立 clone 中把候选 motion、gate 和 precision 写回，并用 cam01..cam20 的真实码流 RD 验证即时收益。本轮继续检查一个更关键的问题：这些收益进入原训练后能不能保留，还是会被原 optimizer 撤销。

本轮只比较两个已经通过即时事务验证的调试节点：

- frame 5 / step 120 / level 1 / node 2552：反馈目标为 `REFINE + parent 8 bit + descendants 16 bit`。
- frame 99 / step 180 / level 0 / node 43：反馈目标为 `STOP + parent 8 bit`。

frame 75 没有可靠候选，上一轮保留源状态，本轮不重复训练。

## 方法和代码

新增：

- `config/fs_joint_feedback_continuation_pair.yaml`
- `scripts/continue_joint_feedback_pair.py`
- `scripts/analyze_joint_feedback_continuation.py`
- `tests/test_nested_prototype.py` 中的续训调度和原生硬化测试

每个节点有两个独立臂：

- `source`：恢复原阶段快照和原 optimizer state。
- `feedback`：恢复上一轮已提交的反馈 clone，包括候选 motion warm start 和目标行 optimizer moment reset。

原快照位于 main backward 之后、optimizer update 之前。两个臂都清除快照内的旧梯度，在当前步重新计算快照记录的训练相机；这避免把 source 旧梯度用于 feedback 参数。frame 5 从 step 120 训练到 200，共 81 次更新；frame 99 从 step 180 训练到 200，共 21 次更新。后续相机用相同的 seed 0 均衡排列 cam01..cam20。

每一步复用原 motion 路径：`forward(training=True)`、0.8 L1 + 0.2 DSSIM、storage weight 0.01、原 optimizer 和 `optimizer_step()`。源配置的 `gs_activity_loss_weight=0`，脚本对此有运行时断言。稀疏 motion 主路径不会启用 keep-candidate loss。step 181 的 gate/precision 硬化和 motion Adam 重建直接走原模型代码；两个臂实测学习率都是 `[1.5e-4, 1.5e-5]`，等于原 translation/rotation LR 乘 `refine_lr_scale=0.3`。

storage 在 refine 阶段固定 gate 和 precision 选择，但 symbol-support STE 仍可给 motion 行传梯度。本轮保留了这一原行为。

cam01..cam20 用于训练和中途 RD 检查。cam00 只在 step 200 各评一次，从未用于更新或选择。没有执行 frame commit、补点或完整帧训练，因此结果只说明旧 GS motion 的受控续训。

## 运行与核验

运行目录：

`output/fs_joint_feedback_continuation_pair/20261008T102706`

运行命令：

```bash
/home/gamer-02/Documents/Codex/miniforge3/envs/ReConGS/bin/python \
  scripts/continue_joint_feedback_pair.py \
  --config config/fs_joint_feedback_continuation_pair.yaml \
  --output output/fs_joint_feedback_continuation_pair/20261008T102706
```

进程退出码为 0，用时 27.70 秒。运行前完整测试结果：

- `tests/test_nested_prototype.py`：185 项通过。
- `tests/test_experiment_inventory.py`：16 项通过。
- `tests/test_recongs_native_stream.py`：16 项通过。

独立分析得到 `issues=[]`。两个节点合计 204 次训练更新、20 组 all20 评测，即 400 次普通视角渲染，以及 4 次最终 cam00 渲染。两个节点、两个臂都只在 step 181 硬化一次。7 份启动时源码/配置快照、历史阶段快照和反馈 packet 的哈希均通过。数据仍从原软链接读取，没有复制数据集或修改历史输出。

## 结果

### frame 5：即时收益没有保留

反馈相对 source 的真实 RD gain：

| 时点 | feedback RD gain | 字节差（feedback-source） | all20 PSNR 差 |
|---|---:|---:|---:|
| step120 pre-update | +4.5375e-7 | +32 B | +0.000296 dB |
| step120 post-update | +3.8506e-7 | +32 B | +0.000251 dB |
| step150 post-update | +2.1215e-7 | +296 B | +0.000283 dB |
| step180 post-update | -5.6906e-7 | +193 B | -0.001891 dB |
| step181 post-update | -6.0927e-7 | +194 B | -0.002100 dB |
| step200 post-update | -2.7043e-7 | +193 B | -0.001712 dB |

cam00 最终结果也不支持 feedback：image loss 增加 `7.7635e-6`，PSNR 降低 `0.00665 dB`。

结构轨迹说明了收益怎样消失：

- feedback gate 从开始到硬化始终保持打开。
- feedback 的 parent precision 在 step 126 从 8 bit 回到 32 bit；四个 descendant 仍保持 16 bit。
- source gate 在 step 121 自行打开，step 175 又关闭，说明该节点的 gate 本身处在容易反转的边界。
- step 200 时 feedback 的 all20 image loss比 source 略低 `1.0361e-7`，但多 193 B 的码率代价更大，所以总 RD 仍为负。

因此，这里不是“BCE 没把 gate 翻开”。gate 已经兑现并保持打开；同时观察到 parent precision 被原训练改回，继续训练后的图像收益也不足以支付结构和 motion 码率。由于两臂的其他节点和 motion 轨迹也在继续变化，本实验不能把最终 RD 转负单独归因于 parent precision 回升。它能证明的是：step 120 的一次联合 target 不能保证 80 步后仍有正收益。

### frame 99：STOP/低精度反馈保持

反馈相对 source 的真实 RD gain：

| 时点 | feedback RD gain | 字节差（feedback-source） | all20 PSNR 差 |
|---|---:|---:|---:|
| step180 pre-update | +1.1887e-7 | -6 B | +0.000055 dB |
| step180 post-update | +9.1629e-8 | -6 B | +0.000043 dB |
| step181 post-update | +9.4516e-8 | -6 B | +0.000043 dB |
| step200 post-update | +3.0187e-7 | -6 B | +0.000204 dB |

cam00 的 image loss 降低 `7.5437e-7`，PSNR 增加 `0.000488 dB`。source 保持 STOP + 16 bit，feedback 保持 STOP + 8 bit，直到 step 200 都没有反转。

这个案例说明较稳定的 STOP/precision 反馈能经过硬化和后续 motion refine，并保留真实码流收益。不过绝对量仍很小，而且这里只有一个调试节点，不能当成总体效果。

## 结论和原因分析

本轮给出了两种不同结果：

- frame 99 的反馈动作、精度和收益都保留下来。
- frame 5 的 gate 监督保留下来，但 parent precision 在 6 步后被原 optimizer 改回，最终 RD 和 cam00 都变差。

这说明“一次候选搜索 + 一次 logit 写回”已经能形成真实闭环，但还不是稳定的训练机制。frame 5 在 step 120 做选择，距离 step 181 硬化还有 61 次 gate-compress 更新，联合 target 和相对收益都可能继续变化。下一步应在多个预定节点上把当前状态的反馈放到 gate-compress 末端，然后继续短 motion refine 检查，而不是调大 CE 强度来强行保留旧标签。这个实验只能测试“硬化前重新评估”这个机制位置是否值得接入，不能作为 step 180 优于 step 120 的因果消融。

本轮只是两个机制调试节点的受控续训，不是总体统计，也不是历史训练轨迹的精确 replay。未来相机顺序是固定的均衡顺序；当前步虽使用原快照训练相机，但梯度重新计算。完整方法仍需在更多独立帧/节点上验证，并保留 cam00 或其他未参与选择的最终评测。
