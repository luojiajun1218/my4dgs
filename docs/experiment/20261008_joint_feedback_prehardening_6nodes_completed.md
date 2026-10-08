# 硬化前6节点联合反馈实验与数值底噪审计（已完成）

## 要解决的问题

前三个调试节点说明，层级和精度候选可以写回独立 clone，但即时收益可能在后续训练中消失。本轮把反馈位置固定在 step 180，即 gate/precision 硬化前一刻，并扩大到预先选定的6个节点。目标是回答三件事：

1. 双折跨视角选择能否在不同层级、开关状态下找到可靠候选。
2. 候选写回后，经过 step 181 硬化和 step 200 motion refine，真实码流 RD 收益能否保留。
3. 这些微小差值是否大于相同初态、相同训练日程下的独立 GPU 重算差异。

结论需要分开看。候选搜索和事务写回按设计完成：384个候选全部落盘，事务得到3个提交、2个保留 source、1个回滚。继续到 step 200 后，3个提交节点在 cam01..cam20 上只有2个 RD gain 为正，另1个转负；3个节点的 cam00 PSNR 都略低于 source。新增的无反馈独立重复还表明，frame 25 和 frame 99 的末期差值小于一次无干预重复差异，不能当成可靠机制收益或损失。frame 1 的训练视角增益明显大于这次重复差异，但 cam00 下降也明显大于重复差异，表现为训练视角与未参与选择视角之间的真实冲突。

因此，本轮证明了反馈原型的闭环和回滚逻辑，尚未证明它能提升最终训练效果，也不支持直接接入正式训练。

## 实验范围和代码

三个正式阶段使用同一批节点：

| frame | step | level | node | 初始层级状态 |
|---:|---:|---:|---:|---|
| 1 | 180 | 0 | 1160 | reachable closed |
| 5 | 180 | 0 | 1279 | reachable open |
| 25 | 180 | 1 | 618 | reachable closed |
| 50 | 180 | 1 | 2919 | reachable open |
| 75 | 180 | 2 | 934 | reachable closed |
| 99 | 180 | 2 | 1010 | reachable open |

对应产物：

- 双折候选回放：`output/fs_joint_feedback_prehardening_crossfold_6nodes/20261008T103200/`
- 反馈事务：`output/fs_joint_feedback_prehardening_transaction_6nodes/20261008T104500/`
- 受控续训：`output/fs_joint_feedback_prehardening_continuation_6nodes/20261008T105700/`
- 失败路径审计：`output/fs_joint_feedback_noop_failure_audit/20261008T110639/`
- 无反馈独立重复：`output/fs_joint_feedback_noop_stability_3accepted/20261008T110538/`

本轮涉及的实验入口和配置包括：

- `scripts/replay_joint_feedback_crossfold.py` 与 `config/fs_joint_feedback_prehardening_crossfold_6nodes.yaml`
- `scripts/prototype_joint_feedback_transaction.py` 与 `config/fs_joint_feedback_prehardening_transaction_6nodes.yaml`
- `scripts/continue_joint_feedback_pair.py` 与 `config/fs_joint_feedback_prehardening_continuation_6nodes.yaml`
- `scripts/audit_joint_feedback_noop_stability.py` 与 `config/fs_joint_feedback_noop_stability_3accepted.yaml`
- 三个独立分析入口，以及 `tests/test_nested_prototype.py` 中的候选、事务、续训和 no-op 数值检查

这些代码只作用于历史快照的离线 clone。`main.py` 的正式训练尚未接入 feedback；attribute gate、完整消融和大范围 PSNR 扫描均未开始。

## 阶段一：双折候选回放

cam01..cam20 分为奇数和偶数两组，各10张。每一折用一组拟合20步，再到另一组检查；第二折交换。每个节点比较32种 STOP/REFINE 与 parent/descendant precision 组合，总计 `6 × 2 × 32 = 384` 个候选、7680次候选更新。cam00 完全不参与。

回放用时678.51秒。独立分析确认12个 fold 和384个候选完整，6份历史快照哈希一致，`issues=[]`。

| frame | 两折拟合 winner | held-out 层级偏好同号 | 两折 winner 都优于 source | 后续同候选选择 |
|---:|---|---|---|---|
| 1 | refine p12/d12；refine p12/d12 | 是，REFINE | 是 | refine p8/d12 |
| 5 | refine p12/d12；refine p12/d12 | 是，REFINE | 否 | source |
| 25 | refine p12/d12；refine p12/d12 | 是，REFINE | 是 | refine p16/d12 |
| 50 | refine p12/d8；refine p16/d12 | 是，STOP优于REFINE | 否 | source |
| 75 | refine p8/d12；stop p12 | 否 | 否 | stop p8，待事务复核 |
| 99 | refine p12/d8；refine p8/d8 | 是，REFINE | 是 | refine p12/d8 |

“后续同候选选择”同时使用两个方向的检查值做选择，因此不再是 held-out 估计。source 固定为0分兜底；只有同一个候选在两个方向都为正时才有资格写回。这个阶段说明跨视角复核能筛掉 frame 5 和 frame 50，但具体 precision winner 仍不稳定。它没有证明搜索结果能经过硬化，也没有独立测试集意义。

## 阶段二：反馈事务

对选中的候选，从源快照重新用 cam01..cam20 各拟合一步，再把目标 motion 行写入 clone。gate 和 precision logits 通过有限步优化达到候选硬动作；随后用20张普通视角和生产码流成本复算 RD。实际 gain 严格大于0才提交，否则完整回滚。

事务用时28.88秒，结果为3个提交、2个 source 保留和1个回滚：

| frame | 选择 | 两方向 selection gain | 实际字节差 | 即时 realized RD gain | 事务结果 |
|---:|---|---:|---:|---:|---|
| 1 | refine p8/d12 | +2.510e-7 / +2.555e-7 | +1 B | +2.402e-7 | 提交 clone |
| 5 | source | 0 / 0 | 0 B | 0 | 保留 source |
| 25 | refine p16/d12 | +1.519e-7 / +1.899e-7 | +8 B | +2.600e-7 | 提交 clone |
| 50 | source | 0 / 0 | 0 B | 0 | 保留 source |
| 75 | stop p8 | +1.765e-8 / +1.553e-7 | -18 B | -7.827e-8 | 完整回滚 |
| 99 | refine p12/d8 | +8.261e-8 / +1.695e-8 | -21 B | +2.994e-8 | 提交 clone |

frame 75 是必要的负例：双折选择值为正，但在全20张视角重新拟合和真实码流复核后变成负收益，事务没有强行接受。三个提交节点中，候选预测与部署 clone 的图像 loss、字节数和 RD gain 一致，说明 motion warm start、可学习 gate 和 precision 硬动作确实被写回。

第一次 transaction 分析日志退出码为1；修正分析入口后使用同一只读产物重跑，`analysis_retry_20261008T104500.status=0`，最终 `analysis.json` 为 `issues=[]`。成功结论只采用重跑后的审计结果，失败日志仍保留。

## 阶段三：硬化和续训

两个臂都丢弃快照中当前步的旧梯度，并在 step 180 重新计算快照记录的训练相机。随后使用 seed 0 的相同均衡相机日程，沿原 loss、storage weight 0.01、原 optimizer 和原阶段调度继续到 step 200。step 181 由生产代码完成 gate/precision 硬化。cam01..cam20 用于训练和中途检查；cam00 只在最后评一次。

最终运行用时55.71秒，6个节点全部完成，分析 `issues=[]`。实际执行189次训练更新、720次普通视角评价 render 和9次 cam00 render。3个 no-op 节点采用单 source 计算镜像，节省了第二臂计算；这项语义在后面的数值审计中单独说明。

| frame | 事务状态 | step 200 all20 RD gain | 字节差 | all20 PSNR差 | cam00 PSNR差 |
|---:|---|---:|---:|---:|---:|
| 1 | committed | +4.685e-7 | +20 B | +0.000176 dB | -0.001503 dB |
| 5 | source retained | 0（镜像） | 0 B | 0 | 0 |
| 25 | committed | +1.462e-7 | +7 B | +0.000059 dB | -0.000053 dB |
| 50 | source retained | 0（镜像） | 0 B | 0 | 0 |
| 75 | rolled back | 0（镜像） | 0 B | 0 | 0 |
| 99 | committed | -2.091e-8 | -21 B | -0.000030 dB | -0.000385 dB |

frame 1 的 gate 保持开启，但 parent precision 在 step 180 第一次更新后从目标8 bit 回到32 bit，两个 descendant 保持12 bit。尽管最终 all20 RD 为正，它没有完整保留所选 precision 策略。frame 25 的正 gain 下降到即时值的约56%。frame 99 的即时小正 gain 在 step 200 转为小负值。三个提交节点的 cam00 都略差，不能据此宣称反馈改善最终质量。

## no-op 失败定位

最终成功运行之前有两个失败目录：`20261008T105100` 和 `20261008T105500`。失败原因不同，不能统一归因为“CUDA 非确定性”。字段级复测记录在失败路径审计目录中；复测不是原失败现场，审计文件明确保存了这一限制。

### 105100：独立计算从第一次更新开始分叉

frame 5 的 source runtime 与完整 `deepcopy` 在更新前逐字段相等，`INITIAL_DIFFS=[]`。两个臂使用同一视角做 step 180 更新，total loss 都是 `0.010479181073606014`，但更新后出现45处 runtime 差异。代表字段为：

- `region_translation.0`：最大绝对差 `1.278e-6`，909/6720个元素不同。
- `region_rotation.0`：最大绝对差 `4.465e-8`，1216/6720个元素不同。
- `refine_logits.0`：最大绝对差 `1.848e-6`，41/2240个元素不同。
- `precision_logits.0`：最大绝对差 `8.136e-6`，665/11200个元素不同。

这证明初态相等和标量 loss 相等不足以保证两次独立 GPU 更新逐位相同。证据把分叉定位到独立数值执行后的第一次更新，但没有点名某个 CUDA kernel，因此不作更强因果表述。

### 105500：capture/restore 改变 Adam step 张量设备

旧 no-op clone 使用 `capture()` 后重新构造 motion。复测发现 step 180 后 capture/restore 完全相等；step 181 后只有8个字段不等：`motion_optimizer.state.{0..7}.step` 在源 capture 中是 CPU 上的 float32 标量，restore 后位于 `cuda:0`。参数值没有发现丢失。

因此，105500 在构造 clone 时就因 optimizer step 张量设备不同而失败，尚未执行第二臂更新。最终代码改成 `deepcopy` 完整 runtime，避免 capture/restore 的设备变化。

105700 对 source-retained 和 rolled-back 节点采用“source 更新一次，再完整 deepcopy 并镜像指标”。这能证明 no-op 记录与 source 完全一致，但不能用来证明两个独立 GPU 重跑等价。

## 无反馈独立重复与效应量

为估计既有成对协议的数值底噪，对3个已提交节点各做一次无反馈重复。两个臂从同一 source runtime `deepcopy`，初始完整 runtime 和 RNG 指纹完全一致；两臂都独立执行原 step180..200 日程，不施加 feedback，也不镜像结果。运行用时18.38秒。

| frame | 初始最大参数差 | 无反馈末期 \|all20 RD差\| | feedback末期 RD gain | \|feedback\| / 无反馈差 | 无反馈 \|cam00 PSNR差\| | feedback cam00 PSNR差 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 3.865e-8 | +4.685e-7 | 12.1× | 0.000034 dB | -0.001503 dB |
| 25 | 0 | 2.875e-7 | +1.462e-7 | 0.51× | 0.000198 dB | -0.000053 dB |
| 99 | 0 | 3.014e-7 | -2.091e-8 | 0.07× | 0.001343 dB | -0.000385 dB |

无反馈两臂的最终字节数在每个节点都相同，所以表中的 RD 差来自图像数值差。21步后最大参数绝对差分别达到 `5.225e-4`、`1.141e-3` 和 `1.452e-3`。

一次重复不能估计噪声分布，也不能给统计显著性。它足以否定两种过强表述：frame 25 的小正 gain 和 frame 99 的小负 gain 都低于这次无干预重复差异，不能当成可靠机制证据。frame 1 的 all20 改善和 cam00 下降都比本次重复差异大一个数量级以上，更像真实的训练视角收益与未参与选择视角损失，而不是简单的逐位重算误差。

## 完整性和测试

三个正式阶段的最终启动状态均为0；crossfold、transaction retry 和 continuation 的独立分析均为 `issues=[]`。每个阶段在计算前保存实际入口、配置和核心依赖源码副本及 SHA-256。历史快照、fold 记录、部署 packet 和只读 summary 的哈希检查通过，没有修改历史输出或软链接目标。

完成 no-op 审计后重新运行：

- `tests/test_nested_prototype.py`：187项通过。
- `tests/test_experiment_inventory.py`：16项通过。
- `tests/test_recongs_native_stream.py`：16项通过。
- 新审计入口 `py_compile` 通过，3节点无反馈诊断正常退出。

## 对论文主线的判断

本轮闭环的主要价值是暴露了三个方法问题，而不是给出正向总体结果：

1. 关闭分支需要训练机会，否则 STOP/REFINE 比较不公平；给所有32个候选各训练20步虽然能诊断，但代价过高。
2. 选择值只有 `1e-8` 到 `1e-7` 量级时，单次 GPU 续训差异足以改变结论；可靠决策必须显式估计或压过这一底噪。
3. 训练视角 RD 改善不保证 cam00 改善。当前3个提交节点的 cam00 全部略差，反馈不能只在参与选择的20张图上验收。

6节点双折搜索耗时678.51秒，占整套正式三阶段时间的89%左右，只换来几个节点上数十字节量级的不同选择。它适合作为 oracle 和诊断工具，不适合作为生产训练机制。已有 GETA-3DGS 已联合结构稀疏与异质精度，MesonGS++ 已使用离散搜索/ILP做混合精度分配，ReCon/QuARC 也已有层级运动和量化。因此论文不能把“联合结构与精度”或“做离散搜索”本身当作创新点。更具体的缺口应是：在线动态场景里怎样给潜在细层低成本训练机会，并在数值底噪和视角偏差下做可靠、可扩展的层级决策。

## 最小下一步机制候选

下面两项都保留可学习 gate 和 storage loss，也都不依赖调大 CE、改阈值或做 PSNR 扫描。优先做批量 shadow residual；局部 lookahead 只作为后备方案。

### 候选A：批量 shadow residual 探索

首版从已有6个 frame 的 step120 快照继续到 step200。主训练 pass 完全保留原 gate、precision、storage loss 和 optimizer。只在 step120..179 的 gate-compress 窗口增加10次 shadow full-image backward：step120、126、132、138、144、150、156、162、168、174。额外 backward 占正常300步的3.33%。

每次 shadow pass 轮转一个 gate level，并批量选择该层“当前主硬路径可达、gate关闭、occupied”的全部 frontier 节点。同层节点的 GS 子树互不重叠。探索 forward 只临时强开这些 frontier；更深 gate 保留源 hard 选择，避免 direct child 在与真实开门不同的模型下重复补偿。训练目标是因此新可达的全部 occupied 后代 motion 行，不含 frontier 父行。强开只存在于这次 forward，不写入主 gate logit 或 hard mask。shadow 只用普通训练相机的图像 loss；precision 使用当前 argmax 的 detached 选择。原 precision 和 storage 学习仍由主 pass 完成，因此首版只检验层级训练机会，不声称已经解决 joint precision。

参数与 Adam 隔离在现有结构上可实现，但不能直接复用主 motion Adam。每个 level 的 motion 是整张参数，主 Adam 的 `step` 又是整张共享标量；直接调用主 optimizer 会改变父层、兄弟行的 optimizer 语义。首版应建立独立 `shadow_motion_optimizer`。shadow backward 后，只保留新可达后代行的梯度；无目标的 level 设 `grad=None`，非候选行梯度和 shadow moment 清零。这样只有目标后代 translation/rotation 和 shadow 自有状态变化，主 `motion_optimizer`、`gate_optimizer`、父层和兄弟参数保持逐值不变。已有 `_paths`、occupancy、parents/children cache、`_region_motion`、`node_conditional_transform` 和行 mask 逻辑可以复用。

建议新增一个 YAML 块，参数先固定为：

- `start_step: 120`、`end_step: 179`、`interval: 6`、`max_backward_per_frame: 10`
- `levels: [0, 1, 2]`，按事件轮转
- `frontier_policy: reachable_closed_same_level_all`
- `descendant_policy: preserve_source_hard`
- `camera_policy: seeded_balanced_cycle`，排除 cam00
- `precision_policy: current_argmax_detached`
- shadow Adam 的 translation/rotation LR、betas 和 eps 沿用主 motion，即 `5e-4`、`5e-5`、`[0.9,0.999]`、`1e-8`

6个 frame 的 baseline 与 shadow 两臂各执行81个正常更新，共972次正常更新；shadow 臂额外60次 backward，不做32候选枚举。关键验证信号包括 frontier 覆盖率、目标 child 梯度和非零 symbol support、主 gate 自主变化、step181硬化动作、exact bytes、all20 RD 和最终 cam00。测试必须逐值证明：只改目标 child 行；主 gate/precision/hard mask 与两个主 optimizer state 不被 shadow 直接改写；同层子树不重叠；10个触发步、层轮转和相机排除正确；checkpoint 能恢复 shadow optimizer 与调度；临时强开不会泄漏到生产码流。

假设是：当前 hard-forward STE 让关闭 child 的 motion 梯度为0，gate 梯度又依赖未充分训练的 child residual。少量批量 shadow pass 能用一次全图 backward 同时训练许多互不重叠的关闭候选，再让原可学习 gate 和 storage loss自行决定是否开启。有效信号必须明显超过无反馈重复差异，并且不能再出现训练视角小正、cam00 同量级负的情况。

### 候选B：带一步 motion lookahead 的局部 RD advantage

如果批量 shadow 仍不能形成可靠 gate 信号，再考虑对少量 frontier 节点做 STOP/REFINE 局部比较。REFINE 先对目标 child motion 行做一步可逆 inner update，再用下一张普通相机计算 `image loss + storage cost`；detached RD advantage 作为 gate 学习信号，而不是把32候选 argmax 蒸馏成 CE 标签。

这个方案能更直接估计局部价值，但仍保留逐节点估值，且一步 lookahead 可能比20步 oracle 更弱。最小诊断预算为6节点各20对相邻 fit/check 相机，最多120次 inner update 和约240次额外 render。它只在候选A失败后使用。若两个方向都不能给出超过底噪的稳定信号，应停止反馈机制扩展，回到基础 gate/storage 目标本身，而不是继续加标签强度。
