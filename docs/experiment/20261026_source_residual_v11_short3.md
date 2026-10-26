# 原材质背景学习独立细节：短3帧（旧2000步诊断）

完成日期：2026-10-26。此轮按旧2000步排程完成；用户随后明确每帧总训练≤300步且只允许单模型，因此本轮不满足最终训练预算，不能作为达标结果，也不再续跑长帧。保留记录用于理解失效。固定原132586 GS首帧（既有过滤保留128963）、1345×1008半分辨率和原learnable motion gate。

## 核心想法与改动

V10的共同颜色更新与子核残差绑定，拒绝坏共同颜色也抹掉好细节。本轮用`SourceResidualModel`独立保存共同材质C；细节训练前将母和子重置原θ0，母材质在细节训练中保持θ0，子学习D。最终fine端点为prior/common/detail/full，允许拒C保D；coarse保留旧DC/SH端点。仍采用当时的独立双候选和200＋3×600预算，现已因用户单模型/300步要求弃用该训练排程。

新增`source_residual_model.py`和对应测试，helper仅增加端点语义标签，3×3解法未变。修复源HDR边界投影的float32 roundtrip：逐点真正pristine材质保留θ0原bit，已接收C仍可提交。没有修改native推理kernel、原首帧或数据。

## 实际结果

| 指标 | 结果 |
|---|---:|
| 全3帧平均PSNR | 33.629358 dB |
| 两动态帧平均PSNR | 33.835108 dB |
| frame1新增前→后 | 33.588875→33.588875 dB |
| frame2新增前→后 | 34.081341→34.081341 dB |
| 接收出生点 | 两帧均0 |
| 最终总点数 | 两帧均128963 |
| 完整动态更新均值 | 61.5405 KB/帧 |
| frame1 / 2完整更新 | 90.763 / 32.318 KB |
| 全场景渲染FPS | 287.84 |

KB为十进制，含必要manifest；共享首帧11081043字节另计。FPS为每帧10次预热、50次同步完整forward+clamp，非解码/上传速度。训练与独立码流benchmark均exit0，但本轮全部新增撤回，不能从FPS/存储推断新增机制成功。

## 分析

实际step1999保存状态确认两候选的母raw DC/SH及pcommon leaf均逐bit等于θ0（含signed zero）；C缓冲仍非零。子D的DC RGB RMS约0.142/0.149，SH球面半径RMS约0.0359/0.0362；实际DC/SH Adam为599步，211/218与210/218子有非零颜色创新。说明确有训练与参数更新，失败不能笼统说成没跑optimizer。该状态是提交前完整人口，不能代表最终接收收益。

第一帧A方案admit选122P，最终detail权重0.1017、common0，颜色控制两组都改善；但相对原场景A MSE−0.5265%，B＋0.0356%，最终双组检查全弃。B方案88P、w0，A＋1.1089%、B−1.7928%，属于结构/密度跨视角取舍，非颜色回退或码率失败。

还发现目标错位：admit按all20平均误差挑集合，最后却要求两组分别改善，没有按组回滚个别坏族。已有日志不足以证明一定存在双组可行子集；不能把aggregate改善说成已证实可保留的局部好族。用户随后明确单模型，后续应统一all20训练与验收目标，采用逐源族实际codec增量接收，避免无必要的双模型及终态整池撤销。

## 验证与产物

- 509项完整CPU检查通过（104.546秒）；模型11项、独立23项及实际material-lineage/SPC/ATTR验证通过。
- 原生128父＋256子检查：θ0母＋零子20训练视角RGB逐bit一致，253子有响应、243有DC梯度；可独立保D/提交C，六字段回环通过。
- 运行：`output/flame_steak_1025/fs_source_residual_v11_short3/20261026T082740_398023`。
- 冻结103文件：`tmp/experiments/source_residual_v11_short3`；native复用逐hash核对的V9冻结446文件。
- 解码/汇总：`tmp/birth_diagnosis/source_residual_v11_decoded.json`、`source_residual_v11_summary.json`。
- native：`tmp/source_residual_model_native_validation.json`。
- 实际训练检查：`tmp/birth_diagnosis/v11_frame1_1999_material_cpu.json`。
- 第一帧拒绝原因：`tmp/birth_diagnosis/v11_f1_cpu_certificate_summary.json`。

## 后续硬约束

单模型、所有20训练视角共同监督、每帧200运动＋100新增总300；取消双候选、600步阶段和额外refit。新核通过一次真实响应的有界颜色初始化获得可用起点，随后100步内成熟/竞争；初始化无optimizer或SGD循环。这是待验证的下一轮方法，此处不提前报告收益。
