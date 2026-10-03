# 实验记录索引

这里按研究问题整理实验入口。每篇实验文档保留完整条件、结果和证据路径；失败、暂停和未完成记录也保留，便于避免重复试验。阅读时先看项目路线，再按主题查看原始记录。

## 建议阅读顺序

1. [项目路线与当前判断](20261003_recongs_project_roadmap.md)
2. [20 个训练视角的完整 ReCon-GS 对齐结果](20261003_refaligned_zero_inheritance.md)
3. [fine 局部候选的短对照](20261003_fine_local_candidate_same_state.md) 与 [连续长程验证](20261003_fine_local_candidate_continuous.md)
4. 编码方向先读 [原生流基线审计](20261003_recongs_native_baseline_audit.md)，再读 [exact-index-delta 在线验收](20261003_recongs_exact_index_delta_online.md)。

## 核心算法：分层、门控与存储

- [可学习门控、storage loss 与周期重建：暂停的 79 帧实验](20261001_learned_gates_storage_rebuild5_paused.md)
- [联合优化与最终补点上限：299 帧实验](20261002_joint200_finalcap200_completed.md)
- [运动梯度路径 × storage：四组 10 帧对照](20261002_motion_gradient_storage_2x2_10frames.md)
- [20 训练视角、cam00 测试的零继承完整实验](20261003_refaligned_zero_inheritance.md)
- [fine SHARE→局部 SPLIT 候选拟合短对照](20261003_fine_local_candidate_same_state.md)
- [fine 局部候选连续长程验证](20261003_fine_local_candidate_continuous.md)

## 编码、回放与公平基线

- [GradHet 稀疏回放验证与存储口径校正](20261002_gradhet_sparse_playback_validation.md)
- [ReCon-GS 原生逐帧基线审计](20261003_recongs_native_baseline_audit.md)
- [ReCon-GS exact-index-delta 在线写盘与验收](20261003_recongs_exact_index_delta_online.md)
- [ReCon-GS 拓扑继承：299 帧验证](20261003_recongs_topology_reuse_full299.md)

## 尚未运行的对照

- [同容量固定门对照计划](20261003_fixed_capacity_control_plan.md)

## 维护约定

- 一篇文档对应一项实验或一条明确的诊断链；不把负结果从记录中删掉。
- 首页路线文档只保留当前结论和下一步；参数、逐帧指标、日志路径放在对应实验记录。
- 新实验使用新的 `experiment_name` 和输出目录。`output/` 中的历史产物只作读取；需要减少磁盘占用时，先建立清单并单独确认保留策略。
