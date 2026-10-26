# 固定刷新700点并开放方向：13个动态帧后停止

日期：2026-10-26。原计划30帧，在13个动态帧完成后停止；中断帧不计指标，不宣称完整30帧结果。

沿用源点移交机制，将每帧刷新上限降到700、外观额度240，允许新点方向更新。输入仍是132586 GS首帧、1345×1008半分辨率监督，gate机制不变。

已完成动态帧平均PSNR **33.985983 dB**，含manifest实际更新 **64.203 KB/帧**，末帧127076个GS。未作完整序列FPS测量。

停止原因是试用点训练发生明确跨视角回退：第6/7/8帧从完整初始化34.0859/34.2012/34.2677降至33.7396/33.9911/33.8695 dB。训练前后指标与最终编码后评估一致，问题发生在拟合阶段，非编码误差。单纯减少刷新数、增加方向自由度没有解决机制问题。

使用SIGINT停止自己的训练进程，保留已保存checkpoint（含第6帧800步）和全部已完成码流。通过第13帧的独立CPU回放验证哈希。

本轮之后用户明确要求转向整体新增/删除机制，暂停参数小修。下一步实施观测驱动空间提出、候选试用、独立训练视角准入、可撤销删除；暂不继续固定每帧替换一批旧点。

- 输出：`output/flame_steak_1025/fs_birth_handoff_rot_v7_first30/20261026T000257_1b3feb`。
- 配置：`tmp/birth_handoff_rot_v7_first30.yaml`；源码：`tmp/experiments/birth_handoff_rot_v7_first30/source_manifest.json`。
- 停止记录：`tmp/birth_diagnosis/v7_stop.json`。
- 摘要：`tmp/birth_diagnosis/v7_stopped_summary.json`。
- 回放：`tmp/birth_diagnosis/v7_replay.json`。
