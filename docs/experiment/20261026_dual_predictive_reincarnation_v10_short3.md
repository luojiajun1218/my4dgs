# 完整父覆盖、保护成熟期与独立双分支：短3帧

完成日期：2026-10-26。frame0＋2动态帧训练、实际码流独立解码均已完成。目标未达成，未进入长帧。固定原132586 GS首帧（既有过滤保留128963）、1345×1008半分辨率、SH1、原运动learnable gate和200运动步。不是训练参数扫描。

## 核心机制与代码

V9的world opticalmass代理无法保留native投影覆盖，因此本轮不再出生时扣除母核密度。完整母核保留，新细节与母核同中心，以不同主轴收缩协方差打破对称；子核使用独立颜色、几何和透明度，不把母颜色强制平均。三段各600步：先固定父支持学习辐射（N关闭）；再保留成熟父覆盖，让细节/N学习；最后释放透明度竞争并锁N几何。学习期以各核本次出生alpha作真实phenotype和poststep下界，成熟后解除，防止颜色/形状尚未学会就塌到零密度。

全部候选维护A/B完整独立模型与Adam，各仅对自己的10台训练相机反传，每阶段每支900次训练视图渲染。共同父色和细节仍在同族中；最后才用对侧视图校准prior/DC-only/SH-only/full三方向凸创新。P整族的母与子都回到同一原source prior；N颜色在四端点保持当前成熟值，N另验增量。结构删选或N撤回后在实际人口重校准，最终选择完整一支，验证后无SGD。初始化使用all20，因此是条件cross-calibration，非统计上从未见过的图像holdout。

新增`predictive_reincarnation_model.py`、`predictive_colour.py`，并接入module的双分支、阶段、checkpoint、真实codec和prefix恢复；没有改native推理kernel。审查中修复模型API的Adam动量冻结漏洞：仅detach不能阻止旧动量推N，进入末阶段清其几何moment、restore后重新施加阶段保护。实际controller阶段间也整池重建Adam。

## 实际结果

| 指标 | 结果 |
|---|---:|
| 全3帧平均PSNR | 33.660268 dB |
| 两动态帧平均PSNR | 33.881474 dB |
| frame1新增前→后 | 33.642593→33.671631 dB |
| frame2新增前→后 | 34.091316→34.091316 dB |
| 新增收益 frame1 / 2 | +0.029037 / 0 dB |
| 接收出生点 frame1 / 2 | 74 / 0 |
| 结构选择 | frame1 A residual；frame2全回退 |
| 接收独立新几何N | 两帧均0 |
| 动态完整更新均值 | 63.6705 KB/帧 |
| frame1 / 2完整更新 | 95.067 / 32.274 KB |
| 完整场景渲染FPS | 288.24 |
| 动态帧渲染FPS | 288.71 |
| 每动态帧平均训练 | 471.57秒 |

KB为十进制，完整更新含必要manifest，共享首帧11081043字节另计。FPS每帧10次预热、50次同步完整Module.forward+clamp，不含解码/上传/指标。短2动态帧被首包影响，不能外推299帧平均。frame1出生ATTR/RTB1为2309字节，frame2无出生包。实际decode核对与保存质量一致。

## 结果分析

第一帧仅28个源族、74个普通GS保留，最终校准权重为[0,0,0]：几何/密度带来小收益，所有已学P族颜色均回原prior。两种coarse均无接收。第二帧细节方案在选点后仍不能通过实际双fold均值验收，整体回退。收益不足，不能据此启动300帧。

原生预检的128父＋256子、完整13万点背景中，母＋零子在20训练视角逐bit一致；253个子有真实ΣTα，241有DC梯度、253有alpha梯度。3个源各有单子零响应，无一个源双子皆无响应；零DC不等于零贡献。实际训练每支每阶段900视图，细节真实贡献point-view累积超过16万，说明多数新增核确实有监督机会。

只读CPU优化器诊断也不支持“梯度被eps吞掉”：实际eps=1e−15，step1600的非零moment远高于eps；800→1600子核DC的RGB RMS变化A/B约0.131/0.125，SH球面半径RMS约0.0286/0.0283。投影大多保留有效步。1600仍是末阶段200步，2000 finalize后未保存成熟全候选/Adam，不能断言最后400步收敛。以后新实验需在1999额外保存候选，避免这一证据缺口。

最终74人口中，D/S/F整图创新RMS约0.000643/0.000110/0.000720，对侧RHS三项全负，w0是当前三方向simplex的解析最优，非回退代码误杀。但是当前D/S/F将母核共同创新与child-specific创新绑成同一方向，拒绝共同颜色时也清除了细节颜色。这只是绑定限制的代码事实，尚不能证明detail-only有正native收益；下一步以同人口只读分解验证，不追加训练对照或参数扫。

第一帧555秒中约340秒为梯度训练，约212秒为选点/提交。4个选项中两份residual撤N重选，共12次校准、6次admit；全部有硬上界，未见循环挂死。重复全20渲染与编码是明确开销，本轮为保持结果一致没有中途改码。

## 验证与产物

- 完整494项CPU检查通过（102.801秒），独立25项相关检查与三阶段路由/梯度隔离/完整Adam恢复通过。
- 真实13万点背景原生可见贡献、冻结行为、全向颜色界及RTB1六字段回环通过。
- 运行：`output/flame_steak_1025/fs_dual_predictive_reincarnation_v10_short3/20261026T074756_55d60c`。
- 冻结100文件：`tmp/experiments/dual_predictive_reincarnation_v10_short3/source_manifest.json`；配置及invocation manifest同目录。
- 复用逐文件hash一致的446文件native冻结：`tmp/frozen_extensions/support_reincarnation_v9_20261026_0628`。
- 实际解码/汇总：`tmp/birth_diagnosis/dual_predictive_reincarnation_v10_decoded.json`、`dual_predictive_reincarnation_v10_summary.json`。
- 原生模型报告：`tmp/predictive_reincarnation_model_native_validation.json`。
- 独立审查/优化器诊断：`tmp/birth_diagnosis/v10_independent_cpu_review.json`、`v10_colour_optimizer_cpu.json`。

## 完成后同人口颜色来源分解

只读native审计在step1600（末阶段200步）固定348个P、128835个旧survivor、相同source/pose/cov/alpha，统一排除57个N，用真实codec分别构造prior、common-only、detail-only、full。相对prior的MSE百分比：

| branch / 材质 | 本组 | 对侧 | 全20 |
|---|---:|---:|---:|
| A common | −2.13% | +7.22% | +3.08% |
| A detail | −3.36% | −3.69% | −3.54% |
| A full | −6.36% | +2.71% | −1.30% |
| B common | −10.32% | +12.42% | +0.69% |
| B detail | −8.95% | −3.71% | −6.41% |
| B full | −15.84% | +14.28% | −1.26% |

负号表示MSE下降。detail-only回到θ0后按原sphere envelope投影A28/B39个child，收益属于该可行独立端点，不是未投影属性的可加MSE归因。它证明这个中间固定人口存在当前共享D/S/F遗漏的有用方向，不能冒充最终74点/预算验收后的收益。完整池119KB左右也不能当作已达到50KB。没有新训练或对照优化。报告：`tmp/birth_diagnosis/v10_colour_binding/native_step1600_report.json`、`read_only_findings.txt`。

下一轮采用source residual lineage：细节训练时母材质保持原θ0，stage0共同创新C独立保存，子从θ0学D；fine提交端点改prior/common/detail/full，使拒绝C不抹除D，保留相同双branch、成熟密度、实际人口codec证明。这是待验证的机制，不声称θ0背景训练已证明更好。ResGS的继承式residual split、AbsGS和Pixel-GS为密度与覆盖分开处理提供启发，但不照搬即时削减母alpha。primary阅读记录：`tmp/birth_diagnosis/v10_residual_maturation_primary_notes.md`。
