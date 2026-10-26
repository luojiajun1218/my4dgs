# 共享表面patch出生与旧覆盖接管：短3帧

完成日期：2026-10-26。frame0＋2动态帧训练完成，独立码流解码与质量核对通过。固定132586 GS首帧（既有过滤保留128963）、1345×1008半分辨率监督，运动learnable gate不变。本轮未扩长，300帧目标尚未达成。

## 核心机制与改动

V7审计显示，主要损伤在自由3D粗修复阶段已发生，冻结共同颜色后几何/透明度仍会降低训练损失、损害测试视角。因此本轮把出生、覆盖接管和提交单位改成跨多个旧GS的表面patch。

完整训练残差（含持续低频偏差）与原生可见贡献、深度/法向矩构建共享表面；经过多视角照片、深度和法向验真后铺设新核。新核只能学习切面内坐标、二维尺度、面内旋转、alpha与受界共同DC；共享深度、法向、厚度固定。可信材质SH-rest继承与码流预测源独立，初始DC从各视角GT扣掉继承角向项后估计，避免重复加色。

区域旧透明度与新核联合学习。两个600步探索阶段旧覆盖保留比例最多0.5，给予新核实际合成机会；选定整patch后再训练600步，允许旧覆盖回流到1。未接收patch完整回滚，旧数据直到最终提交才实变。最终旧alpha以原参考上的总OPD符号编码，而非叠加已clamp基准上的增量。细分始终共享同一表面和ownership，精确零alpha后续可退休，非零遮挡点保留。

新增`surface_patch_proposal.py`、`surface_patch_model.py`及接入module、residual transaction、量化、checkpoint/延伸模块。码流仍是普通平坦GS；需要时采用现有RTB1精确属性格式。代码由普通agent实现，未使用Command Code。本轮使用独立冻结的原训练渲染扩展，不混入后续component辅助接口。

## 实际结果

| 指标 | 结果 |
|---|---:|
| 全3帧平均PSNR | 33.608588 dB |
| 两动态帧平均PSNR | 33.803953 dB |
| 新增前→后，frame1 | 33.547363→33.542221 dB |
| 新增前→后，frame2 | 34.099754→34.065685 dB |
| 新增阶段收益，frame1 / 2 | −0.005142 / −0.034069 dB |
| 最终选择 | 两帧均fine |
| 接收新点，frame1 / 2 | 215 / 214 |
| 最终总点数，frame1 / 2 | 128922 / 128880 |
| 动态更新含必要manifest均值 | 75.303 KB/帧 |
| 第二动态帧含manifest | 45.946 KB |
| 全3帧完整场景渲染FPS | 288.96 |
| 动态帧平均训练时间 | 276.81秒 |

KB为十进制，共享首帧11081043字节另计；首动态帧104.660 KB影响短序列均值。FPS每帧10次预热、50次同步完整`Module.forward+clamp`，不包含解码/上传/指标，不能当作完整码流播放吞吐。出生包为实际RTB1精确属性包，7529/7544字节，已经计入存储。

## 训练与覆盖分析

表面新核在多个训练视角实际参与合成，完成既定600＋600＋600步骤。第一帧6个patch拥有287个旧点，第二帧8个patch拥有310个旧点。探索阶段保留比例均在0.5上限；terminal后分别回流到0.723–0.945、0.627–0.925，说明新核没有完全接管旧细节。

由全训练视角平均MSE折算的训练收益，第一帧coarse/fine/terminal为+0.00433/+0.00755/+0.00896 dB，第二帧为+0.02436/+0.02793/+0.03204 dB；测试收益仍为负。不能从“所有点有更新”或训练误差下降推断拟合机制成功，也不能把两动态帧绝对均值与另一轮的不同运动结果直接作因果比较。

冻结前真实13万点只读smoke定位了供给瓶颈：20训练视角的425880个8像素cell，在旧深度spread门槛后仅1118，留下总SSE的0.3599%；加法向一致性为641（0.1429%），planar后156（0.0236%），crossview后133，最终合并/唯一ownership得到7patch、63核、276旧点。预算256核未用满。旧3DGS深度std相对半cell世界尺寸中位17–50倍，平均法向长度中位约0.60，混合渲染矩普遍不能直接当成薄表面几何。

该smoke使用实际canonical OPD4、旧保留比例0.5后，新核原生非零支持为0.4778%像素，触达1.3015%原SSE；响应加权SSE仅0.04463%。这些是未训练核的诊断，不能混同训练后收益、区域分配面积或世界表面积。

本轮没有解决质量问题，因此没有进入30/300帧。下一步需要分解最终真实码流的独立MVS与patch接管收益，同时检查能否提取实际贡献分量的条件深度，避免仅放宽混合矩阈值；后续辅助接口没有计入本轮成绩。

### 完成后的真实码流归因

最终六字段与实际checkpoint及官方渲染一致。省略P时恢复它所拥有旧点的motion-stage alpha，避免把半透明载体留下的洞误作新点收益。N/P场景均保留同一批独立删除D，因此不能直接相加。

| frame2场景，相对新增前 | 训练视角平均PSNR收益 | cam00收益 |
|---|---:|---:|
| N＋D | +0.031382 | −0.033138 |
| P＋D | +0.000290 | −0.000877 |
| D | −0.00011 | +0.000061 |
| 完整 | +0.03179 | −0.034069 |

P相对D的额外测试变化约−0.000938 dB，N约−0.033199 dB。主要损伤已转为独立MVS新点；进一步加P约束或训练cohort不能处理这个问题。匹配coarse整池的测试收益−0.016579 dB，coarse P-only为−0.001770 dB；细分/refit后额外退化主要来自N。第一帧N＋D为−0.004078、P＋D为+0.001011、D为+0.002075 dB，同样不能把P＋D的正数当作P独立改善。

第一帧最终126P＋89N，第二帧130P＋84N。非owned存活旧alpha与motion baseline逐位相同；owned平均alpha分别0.4707→0.3905、0.4138→0.3429，接管确实写回，退化不是遗漏旧alpha编码。

后续独立辅助接口的只读诊断也已完成（不计入本轮训练）：实际贡献分量的条件几何使同门槛供给从7patch/63核增到16patch/206核，触达SSE从1.30%到4.61%。但dominant share中位仅0.179，其余分量质量仍大；27.8%中心查询、41.5%可见性查询跨ID插值，可能制造中间面。它尚不能直接替代混合矩作为可信表面。这一结果支持改用原始GS支持域与真正跨视角出生证明，而非单纯放宽或替换深度门槛。

## 验证与产物

- 冻结前429项CPU检查通过，含整patch筛选/固定terminal、真实码流、细分物理边界和精确零退休。
- 原生小场景确认旧retention、新DC/alpha及切向几何有实际梯度，联合一步MSE下降；局部量化后平面误差0，六类属性精确往返。
- 运行：`output/flame_steak_1025/fs_surface_patch_v8_short3/20261026T052423_5b53e1`。
- 配置/源码：`tmp/surface_patch_v8_short3.yaml`、`tmp/experiments/surface_patch_v8_short3/source_manifest.json`。
- 独立渲染扩展：`tmp/frozen_extensions/surface_patch_v8_20261026_0516/extension_source_manifest.json`。
- 解码/结构化摘要：`tmp/birth_diagnosis/surface_patch_v8_decoded.json`、`surface_patch_v8_summary.json`。
- 实际提案smoke：`tmp/birth_diagnosis/surface_patch_real_smoke/summary.json`。
- 模型原生验证：`tmp/surface_patch_native_validation.json`；码流审查`tmp/surface_patch_wire_cpu_review.json`。
- 完整码流归因：`tmp/birth_diagnosis/v8_wire_audit/wire_audit.json`；metadata审计`tmp/birth_diagnosis/v8_wire_metadata_cpu.json`。
- 后续只读分量诊断：`tmp/birth_diagnosis/component_surface_real_smoke/summary.json`；辅助接口原生验证`tmp/surface_component_aux_native_validation.json`。
