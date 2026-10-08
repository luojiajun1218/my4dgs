# 联合候选反馈闭环：3节点clone事务验证

上一轮已经证明，候选在拟合视角上变好不等于换视角后仍然变好。这轮继续解决更具体的问题：候选训练出的运动参数怎样回到主状态，gate和precision logits怎样通过学习达到候选的硬选择，以及候选收益能否在写回后的真实状态中兑现。

本轮结论：最小闭环可以兑现。frame5的REFINE候选和frame99的STOP 8-bit候选都在独立源clone中完成了motion warmstart、logit学习和实际码流复核；预测收益与写回后的实际收益完全一致。frame75没有跨折双正候选，保持源状态。生产训练路径仍未修改。

## 选择与写回方法

输入是上一轮不可变的6份fold记录：`output/fs_joint_feedback_crossfold_3nodes/20261008T100558/`。

对同一个候选`c`，记两次“在一半视角拟合、到另一半视角检查”的RD gain为`g1(c)`和`g2(c)`。这一轮把两边都用于选择，因此它们是selection value，不再称held-out验证。

- 只有`g1(c)>0`且`g2(c)>0`的同一候选才有资格写回。
- 候选分数为`(g1(c)+g2(c))/2`。
- 未改动的source固定为0分兜底。

选中候选后，从原快照重新开始，用cam01..cam20各拟合1步，共20步。随后在独立source clone中执行一个事务：

1. 把候选在目标节点和可达后代上训练后的translation/rotation行复制到clone。
2. 用独立Adam对目标gate做BCE，对候选涉及的precision行做CE。LR、betas和eps沿用原配置：`0.02`、`[0.9,0.999]`、`1e-15`。
3. 最多更新128步，第一次达到目标hard gate和precision argmax就停止。
4. 用20张训练/选择图和生产运动编码器复算clone的图像loss、真实字节与RD。
5. 只有硬选择已兑现且实际RD gain严格大于0才保留clone；否则恢复参数、optimizer、梯度、缓存与RNG的完整事务状态。

候选motion optimizer与主gate optimizer只清目标行的`exp_avg`、`exp_avg_sq`和可选`max_exp_avg_sq`。Adam对整张参数共享的scalar step保持不变，未改节点不受重置影响。临时feedback Adam会恢复未选行已有梯度，目标行的旧梯度清零。

配置为`config/fs_joint_feedback_transaction_3nodes.yaml`，入口为`scripts/prototype_joint_feedback_transaction.py`，审计为`scripts/analyze_joint_feedback_transaction.py`。测试增加了同candidate跨折选择、硬动作学习、未选行不变和完整回滚检查。

## 结果

输出：`output/fs_joint_feedback_transaction_3nodes/20261008T101708/`。正的RD gain表示部署clone优于未修改source。

| 节点 | 跨折选择 | 两方向selection gain | logit步数 | gate变化 | precision目标 | 实际字节差 | realized RD gain | 结果 |
|---|---|---:|---:|---|---|---:|---:|---|
| f5/s120/l1/n2552 | refine p8/d16 | +6.887e-7 / +1.730e-7 | 51 | 关→开 | 父8、后代16，共5行 | +32 B | +4.5375e-7 | clone接受 |
| f75/s180/l0/n926 | source | 0 / 0 | 0 | 关→关 | 保持原值 | 0 B | 0 | 保持source |
| f99/s180/l0/n43 | stop p8 | +1.424e-7 / +2.970e-8 | 18 | 关→关 | 父8，共1行 | -6 B | +1.1887e-7 | clone接受 |

frame5解决了关闭候选与源运动参数不匹配的问题。候选经过全部20视角重新拟合后，motion参数随结构一起写回；51步后gate确实从关闭变成开启，相关precision也达到8/16。候选预测gain和clone实际gain都是`4.5374978414675985e-7`，实际运动码流增加32字节。

frame99说明可靠反馈不等于必须开门。gate继续关闭，父节点precision从原16-bit学习为8-bit；写回后码流减少6字节，实际RD gain为`1.188699177795067e-7`。这也验证了同candidate跨折选择能找到两折各自argmax没有直接给出的稳定precision动作。

frame75的32个候选没有任何一个在两个方向都严格为正，source兜底生效，没有额外拟合或logit更新。这避免了把上一轮明显跨视角过拟合的REFINE结果写回。

两个接受事务中，candidate预测图像loss/字节与部署clone逐项一致，predicted gain与realized gain完全相同。这证明本轮收益来自“训练后的motion参数+实际hard gate/precision”的完整状态，而不是只修改logit后沿用不可兑现的候选分数。

## 完整性与开销

独立审计结果：2个clone接受、1个source保留、0个rollback，issues为空。三份历史快照和6份cross-fold记录哈希保持一致；保存的两个部署motion文件用生产成本函数复算后与报告字节一致。

入口在候选计算前保存了4个实际依赖源码副本和SHA-256 manifest，包括事务入口、候选诊断、快照恢复和生产编码器。总运行时间14.18秒；复用了已有192个候选结果，只新增2个候选的40次render/backward、69次轻量logit更新及三节点的全20视角复核。

启动前测试：

- `python tests/test_nested_prototype.py`：183项通过。
- `python tests/test_experiment_inventory.py`：16项通过。
- `python tests/test_recongs_native_stream.py`：16项通过。

真实运行没有触发rollback分支，因为两个部署候选都成功兑现；确定性测试单独强制负收益，逐值验证motion state、optimizer state、requires-grad、grad、缓存以及Python/NumPy/Torch RNG均恢复。

## 限制与下一步

这20张图同时参与候选选择、最终拟合和接受检查，因此realized gain证明的是事务一致性，不是独立泛化效果。三个节点仍只是机制调试样本，收益量级也很小。

frame5最终precision margin只有`3.05e-4`，frame99为`1.77e-2`。虽然当前argmax已经兑现，继续主训练后仍可能反转。下一步应把同一事务接到一个小预算、独立实验名的短主训练中，记录反馈后的hard动作、precision、实际码流和普通训练视角轨迹，并把cam00继续留到最终效果评测。先确认反馈能在后续优化中保持，再扩到更广节点；attribute gate和PSNR扫描继续暂缓。
