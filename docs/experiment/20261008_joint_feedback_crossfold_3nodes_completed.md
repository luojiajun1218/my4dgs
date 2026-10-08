# 联合候选跨视角反馈：3个代表节点双折回放

这轮先回答一个小问题：候选额外训练20步以后，得到的层级与精度收益能否换一组普通训练视角仍然成立？本轮只回放历史快照，不修改主训练、gate、precision logits或历史输出。

结论很清楚：跨视角复核能区分“值得细化”和“只在拟合图上看起来值得细化”。frame5的节点在两折都支持REFINE；frame75和frame99在两折的检查视角上都更支持STOP。具体precision组合在两折之间不稳定，暂时不适合用单个最优bit做强监督。

## 做法

- 回放输出：`output/fs_joint_feedback_crossfold_3nodes/20261008T100558/`。
- 配置：`config/fs_joint_feedback_crossfold_3nodes.yaml`。
- 节点：frame5/step120/level1/node2552、frame75/step180/level0/node926、frame99/step180/level0/node43。它们只用于机制调试，不能代表总体误选率。
- 每个节点比较完整32候选：2个原precision候选、5个STOP父bit候选、25个REFINE父bit/后代bit候选。
- 每个候选拟合20步，沿用源快照的学习率、`storage_loss_weight=0.01`和`precision_bits=[8,12,16,24,32]`。
- cam01..cam20分成奇数10张和偶数10张。第一折奇数拟合、偶数检查；第二折交换。每个拟合视角恰好出现2次。cam00不参与。
- 候选选择只看拟合侧RD分数。选定后的候选不变，再到另一侧检查。check侧重新选优只保存为选择分析，不作为泛化证据。
- RD使用生产编码器得到的完整运动码流字节。补点PLY对同一局部候选是常数，本轮未加入绝对成本，但不影响候选间字节差。

代码新增独立回放入口`replay_joint_feedback_crossfold.py`、严格审计入口`analyze_joint_feedback_crossfold.py`，并在`tests/test_nested_prototype.py`增加视角划分、平衡调度以及“fit选、check固定评”的检查。生产训练路径没有接入反馈写回。

## 结果

表中收益均为RD gain，正数表示前者更好。“REFINE−STOP”固定比较拟合侧选出的best REFINE和best STOP在对侧10张检查图上的分数。“joint−original”比较拟合侧选出的显式联合precision候选与同20步预算的最佳original-precision候选。

| 节点 | odd-fit winner | even-fit winner | held-out REFINE−STOP（两折） | held-out winner−源状态（两折） | held-out joint−original（两折） |
|---|---|---|---:|---:|---:|
| f5/s120/l1/n2552 | refine p12/d8 | refine p8/d12 | +6.377e-7 / +1.384e-7 | +6.985e-7 / +1.572e-7 | +2.513e-7 / +2.680e-7 |
| f75/s180/l0/n926 | refine p8/d16 | refine p16/d12 | -1.535e-7 / -7.379e-7 | -3.188e-7 / -9.015e-7 | -1.835e-8 / +1.015e-8 |
| f99/s180/l0/n43 | refine p12/d12 | stop p8 | -6.309e-8 / -5.094e-7 | -4.376e-7 / +2.970e-8 | -9.410e-8 / +6.285e-8 |

frame5是可靠正例。两折分别在不同10张图上拟合，换到对侧后仍支持REFINE；显式联合precision候选也都优于同预算original候选。两折的具体bit组合不同，说明现阶段稳定的是层级动作，不是唯一precision标签。

frame75是可靠反例。两折的拟合侧都把REFINE选为总体winner，但换组后REFINE相对STOP都变差，且fit winner相对未额外拟合的源状态也都变差。这说明给关闭分支20步训练机会仍不足以直接形成gate反馈，必须保留不参与本折拟合的多视角复核。

frame99的收益小且分歧明显。一折拟合选REFINE，另一折拟合选STOP；但两边held-out的REFINE−STOP都为负。两折各自的argmax不一致，因此不能直接把单折winner当成稳定precision标签。

三个节点的held-out层级收益符号在双折内都一致：frame5为正，frame75和frame99为负。具体precision winner没有一个节点在双折完全一致，说明不能直接蒸馏单折argmax。

完成实验后的同candidate跨折选择分析把两折check都用于选择，因此不再是held-out验证。对每个相同候选取两个方向的检查RD gain，只保留两边都大于0的候选，再按均值选择：frame5选中`refine_parent8_descendant16`，两个方向为`+6.887e-7/+1.730e-7`；frame75没有双正候选，保留源状态；frame99选中`stop_parent8`，两个方向为`+1.424e-7/+2.970e-8`。这说明frame99仍可能提供可靠的STOP父层8-bit更新，只是不能从两折各自winner直接得到它。这里没有设置`1e-7`等绝对收益门槛。

## 完整性与开销

独立审计重算192个候选的图像收益、实际码流字节差和RD分数。6个fold全部包含32个唯一候选；每个候选20次拟合更新、每张fit图2次，fit/check严格互斥。两个original-precision对照均存在。三份历史快照SHA-256复核一致，新增相机的图像与相机矩阵指纹已写入汇总。审计未发现问题。

回放总耗时344.17秒，包含3840次候选拟合反传和7800次候选/基线评价render。输出约6.3 MB。运行进程退出码为0。

启动前测试：

- `python tests/test_nested_prototype.py`：181项通过。
- `python tests/test_experiment_inventory.py`：16项通过。
- `python tests/test_recongs_native_stream.py`：16项通过。

本次入口在启动时复制了YAML，但没有自动保存脚本副本和脚本哈希。运行中观察到回放脚本SHA-256为`ed325a6df2b4cb140e2eaea31090277546525ff7bccd387bcc59d8229b1ba82f`，文件mtime早于进程启动；结果字段也与该版本一致。这是运行中追溯证据，不能表述成启动时生成的加密源码快照。后续回放入口应在开始候选计算前自动保存源文件副本与哈希。

## 下一步

下一步先在同3节点做真实反馈闭环原型：复用本轮不可变候选记录完成同candidate跨折选择；选中结构再从源状态用全部20张普通训练图拟合20步；在独立clone中用有限BCE/CE更新可学习gate与precision logits，并带回训练后的motion参数。只有硬动作、precision和新主状态实际码流RD都兑现才接受，否则完整回滚。这个闭环通过后，再扩到更广样本或短主训练。当前结果不支持只做一次CE后宣称收益已兑现，也不支持直接覆盖gate或precision argmax。attribute gate和PSNR扫描继续暂缓。
