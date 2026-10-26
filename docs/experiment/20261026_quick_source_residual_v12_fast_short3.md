# 单模型300步的解析颜色出生：短3帧

完成日期：2026-10-26。frame0＋2动态帧训练及独立码流解码完成。用户最新硬约束已经落实：单模型、全部20训练视角、每动态帧200运动＋100出生，总300步。没有双分支、额外refit或内部SGD循环。目标质量未达成，不进入长帧。

## 核心机制与实现

从原GS支持中生出两枚缩小子核，保留母核原位置/尺度/SH1/材质θ0，子核固定继承SH1，只学习DC、几何和alpha。出生先用真实native响应做一次每族、每RGB的有界2×2 DC求解，包含同族子核交叉项，忽略跨族H；一次全20实际codec MSE检查决定接受或回滚，零optimizer/backward训练调用。随后单一Adam100步，前50保护出生alpha并固定母alpha，后50竞争；250只清同一Adam状态，没有新训练轮。最终同一已训练池按all20真实codec误差与字节预算逐源族接收/删除，保存并重放已通过prefix，不再做终态颜色校准。

新`QuickSourceResidualModel`固定物理SH1，避免DC变化时SHrest被codec predictor格点隐式改动；母材质保持θ0，child DC使用固定rest下的有界盒。controller、trainer、账本与prefix硬拒超300及旧多cohort/dual/refit状态。真实不同source DC格点的删父案例，连续preimage与actual survivor predictor六属性和bytes重放一致，排除曾怀疑的replay误杀。

初版在一次初始化内因FP64瘦矩阵乘法极慢而中止；本次仅将H/g改为等价乘积归约。真实形状的单次归约0.3877→0.000730秒，约531倍，双精度相对差约1e−14；不是方法/参数/训练步数对照。完整源manifest比对仅initializer与其test两文件变化。

## 实际结果

| 指标 | 结果 |
|---|---:|
| 全3帧平均PSNR | 33.605453 dB |
| 两动态帧平均PSNR | 33.799252 dB |
| frame1新增前→后 | 33.590141→33.590561 dB |
| frame2新增前→后 | 34.108665→34.007942 dB |
| 新增收益 frame1 / 2 | +0.000420 / −0.100723 dB |
| 实际motion/birth/总步 | 两帧均200/100/300 |
| 实际birth optimizer更新 | 两帧均100，同一Adam |
| 接收GS frame1 / 2 | 54 / 158 |
| 总点数 frame1 / 2 | 128867 / 128711 |
| 退休数（含被替换P父） | 150 / 314 |
| 动态完整更新均值 | 67.978 KB/帧 |
| frame1 / 2完整更新 | 94.287 / 41.669 KB |
| 全场景渲染FPS | 286.51 |
| 每动态帧训练均值 | 71.84秒 |

KB为十进制，含必要manifest，共享首帧11081043字节另计。FPS每帧10次预热、50次同步完整forward+clamp，不含解码/上传/指标。首包较大，短序列平均不能外推299帧；本轮质量下降，不据此开长帧。固定132586 GS首帧（既有过滤保留128963）、1345×1008半分辨率与原learnable gate保持。

## 结果分析

初始化在其当前附加核场景中MSE确有下降：frame1 0.000907571→0.000702322，frame2 0.000730924→0.000521772。但其原场景MSE分别约0.000235604/0.000247285，初始化后仍达原场景2.98/2.11倍；20训练视角仅0/1个胜原场景。因此accepted仅表示修复了一部分出生扰动，不能当作出生成功或图像已拟合。

每子固定初生alpha0.1，而源母alpha中位0.608/0.620、最小0.022/0.035。两子alpha和与母之比p90为3.04/1.99，最大9.10/5.71。299时子alpha中位0.0874/0.0894、最小约0.08，没有降到出生一半；母alpha与原值比例中位约0.9999/0.9992。100步的大部分预算可能被用来处理出生自身过量密度，这是待原生分解核对的机制解释，不能用更多step或简单改alpha常数代替解决。

299的DC pairmean占未加权颜色参数能量91.4%/93.6%，contrast较小；这不是渲染能量分解，但提示新增主要在做共同颜色偏移，不能自动当作跨视角一致纹理。报告明确299是提交前完整人口，非最终入选158点。

完成后的actual-wire因果审计明确f2退化来自P：独立D删除train−0.000041/test−0.000053 dB；新增/替换P train＋0.037738/test−0.100693 dB；full test−0.100723与官方一致。重组六字段逐bit相同，存活旧点六字段完全未变。P父由固定geometry唯一逐族匹配：f1 22父→54born、独立D128；f2 58父→158born、独立D256。不是把删父错算成D，也不是训练后存储改变了旧GS。

## 验证与产物

- 529项完整回归＋最终Quick5项通过，等价加速后21项相关检查通过；真实单Adam201..300恰100次更新、全20每个15次birth渲染、初始化0SGD、无终态refit。
- 一次initializer小native验证已知DC目标恢复、source母/DC/SHrest不变、真实ATTR六字段回环；性能归约的H/g误差约1e−14。
- 运行：`output/flame_steak_1025/fs_quick_source_residual_v12_fast_short3/20261026T093905_4aaa19`。
- 冻结108文件及配置：`tmp/experiments/quick_source_residual_v12_fast_short3`。
- 解码/汇总：`tmp/birth_diagnosis/quick_source_residual_v12_fast_decoded.json`、`quick_source_residual_v12_fast_summary.json`。
- 性能报告：`tmp/birth_diagnosis/child_dc_fast_actual_shape.json`。
- 实际density/颜色统计：`tmp/birth_diagnosis/v12_density_colour_cpu.json`。
- actual-wire分解：`tmp/birth_diagnosis/quick_v12_wire_audit/wire_audit.json`。

下一步只根据该单模型结果改进出生/密度/保留机制，保持总300步，不采用双分支或增加训练轮。

## 完成后的299原生属性检查

在提交前完整池（f1 344P、f2 352P，非最终54/158）保持keep/source/geometry/SH1不变，以真实codec测试。下表为MSE×10⁴，单元格为all20 / cam00；cam00只诊断，不参与选择或训练。

| 控制 | f1 | f2 |
|---|---:|---:|
| full | 3.442 / 7.136 | 3.201 / 6.595 |
| 母alpha恢复source | 3.603 / 7.238 | 3.339 / 6.922 |
| child颜色恢复θ0 | 6.353 / 18.688 | 4.907 / 14.522 |
| pairmean-only | 3.474 / 6.713 | 3.200 / 6.341 |
| 受界±contrast-only | 6.522 / 19.293 | 4.977 / 14.698 |

恢复母alpha不改善，直接去mean保contrast显著更差。因此当前大mean确是维持这批额外密度场景拟合的重要补偿，不能据未加权参数能量猜测就改成纯±纹理。contrast端点发生32/37对盒投影；这是可行控制，非可加归因。报告：`tmp/birth_diagnosis/v12_fast_quick299_controls/native_report.json`。

下一轮采用逐族shadow初始化：原母完整、未接收child透明；在当前已验收场景内给该族一次定色，再用native三点密度响应构造quartic、一次解cubic求初生强度，唯一结果通过实际codec误差/字节才实体出生。成熟保护使用数据alpha，避免重新抬到0.1。仍单模型、总≤300、不加SGD；该方案尚未实际验证，不提前报告收益。

另只读检查发现300份COLMAP三类文件各自同hash，实际为共享首时刻静态几何：7363稀疏点（7314有≥2train对应、4796不含cam00），40000来自无track的独立PLY。不能把它们当逐帧动态真值或40000个可靠多视图锚。现阶段不改几何提案，记录作为后续先验可用性边界。`tmp/birth_diagnosis/colmap_geometry_availability.json`。
