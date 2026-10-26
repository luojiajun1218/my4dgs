# 原支持域重生与互补视图优化转移：短3帧

完成日期：2026-10-26。frame0＋2动态帧训练完成，实际码流独立解码核对通过。固定132586 GS首帧（既有过滤保留128963）、1345×1008半分辨率监督、原运动learnable gate及200运动步保持不变。两帧收益不稳定，未进入30/300帧。

## 核心机制

V8中，独立MVS新点贡献了主要测试退化；薄表面提案又被旧3DGS的多层混合深度限制。因此已有支持的误差改为从原GS完整支持域1→1重生：复制原mu、covariance、q、alpha和SH，临时替换旧父核，先获得原核的完整可见覆盖。coarse600步固定几何和alpha，只学习共同DC/SH1。fine600步再对有残差方向证据的原支持做三成员细分，几何锚点保持原source，共同辐射和成对DC细节共享，保留coarse回退。

候选排序改为真实native误差责任：同一alpha/pose、零预计算颜色且可导的pass，对残差平方求颜色梯度，得到每GS的ΣTα·r²。它不会因有正负纹理误差、DC梯度相消或混合深度不可信而丢掉候选；固定支持复制不再设置source尺度上限。

独立MVS几何N不再在出生前反复拆分。两组互补相机分别从相同初始MVS raw独立优化600步；候选必须在未参与该支梯度拟合的另一组上产生真实量化增量收益。N初版整支原子接收/拒绝，且最终实际P/D人口选择后重放转移证明。terminal600步锁N几何，只拟合辐射与alpha。初始MVS使用全训练多视图，因此这是互补视图的优化转移检查，并非图像从未参与初始化的严格统计holdout。

SH1使用全方向sphere envelope，物理先验独立于codec predictor；common与±DC detail共同投影。原HDR先验保留，N为LDR范围。本任务原native checkpoint active SH degree确为1。新增`support_reincarnation_proposal.py`、`reincarnation_model.py`及交易/训练/量化接入；最终仍为普通GS码流，无推理期隐藏支持域状态。

## 实际结果

| 指标 | 结果 |
|---|---:|
| 全3帧平均PSNR | 33.656787 dB |
| 两动态帧平均PSNR | 33.876251 dB |
| frame1新增前→后 | 33.509476→33.862400 dB |
| frame2新增前→后 | 34.031223→33.890102 dB |
| 新增收益，frame1 / 2 | +0.352924 / −0.141121 dB |
| 最终结构 | 两帧均coarse |
| 接收P / N，frame1 | 126 / 0 |
| 接收P / N，frame2 | 128 / 0 |
| 最终总点数，frame1 / 2 | 128579 / 128259 |
| 动态更新含必要manifest均值 | 72.7335 KB/帧 |
| 第二动态帧含manifest | 43.754 KB |
| 全3帧完整场景渲染FPS | 283.60 |
| 动态帧平均训练时间 | 350.57秒 |

KB为十进制，首动态帧101.713 KB影响短序列均值，共享首帧11081043字节另计。FPS每帧10次预热、50次同步完整`Module.forward+clamp`，不含解码/上传/指标。出生实际采用RTB1精确属性，7518/7400字节，已计入存储。

## 结果分析与证据边界

第一帧+0.353 dB说明在原完整支持中训练辐射可以产生明显收益，候选并非完全没有训练机会。但第二帧−0.141 dB表明它仍不稳定，不能由单帧正数推断300帧达标。独立N两帧均未出生，两帧都回退到coarse；本轮必须继续检查支持域内的辐射创新是否跨视角可靠，以及细分为何不能改善成熟父核，不能继续把问题笼统归因训练步数。

冻结前真实13万点只读检查：128个源点的ΣTα·r²占总残差SSE的16.8608%，并非bbox面积；全源责任归因覆盖99.432%的SSE。选中source最大尺度中位1.24、p90为6.54、最大30.93世界单位，alpha中位0.727，可见视角中位20、最少4；113个有fine方向证据。该供给比薄表面提案明显更广，但不等于这些误差能由共同颜色完全解释。

P-only初始化真实检查中19/20训练视角RGB逐位相同，剩余相机有同深度排序交叠：最大RGB差7.28e-5，但平均差图MSE仅3.22e-15，最大PSNR变化5.72e-6 dB。原六字段一致不一般性保证native深度tie下画面逐位一致，本数据的初始化变化可忽略。

## 验证与产物

- 冻结前469项CPU检查通过；模型12项专项及原生验证确认P几何/alpha固定、DC/SH1有实际梯度、N视图梯度隔离和terminal几何锁定生效。
- 真实SPC/ATTR、source映射、固定selected人口、checkpoint恢复及全向颜色范围验证通过。
- 运行：`output/flame_steak_1025/fs_support_reincarnation_v9_short3/20261026T063033_e06376`。
- 配置/源码：`tmp/support_reincarnation_v9_short3.yaml`、`tmp/experiments/support_reincarnation_v9_short3/source_manifest.json`。
- 原生扩展冻结：`tmp/frozen_extensions/support_reincarnation_v9_20261026_0628/extension_source_manifest.json`。
- 解码/摘要：`tmp/birth_diagnosis/support_reincarnation_v9_decoded.json`、`support_reincarnation_v9_summary.json`。
- 源点真实责任与初始化：`tmp/birth_diagnosis/support_reincarnation_real_smoke/summary.json`。
- 模型原生验证：`tmp/reincarnation_model_native_validation.json`。

## 完成后的原生因果检查

最终同人口、同几何和alpha，仅替换辐射属性的原生审计复现了官方PSNR。frame1的DC-only / SH-only / full测试收益分别+0.256363 / +0.095577 / +0.352924 dB；frame2为−0.184341 / +0.044975 / −0.141121 dB。第二帧DC训练收益+0.043512 dB，测试却下降；SH单独反而有正收益。因此主要问题是共同DC跨视角拟合取舍，不能归因SH外推或未执行优化。P位置、尺度、alpha与source一致，存活旧点六字段未改，颜色球界最大浮点误差1.19e−7。95/128个P来自前帧出生点，其损伤占比约73.6%，与数量占比74.2%接近，不能据此断言重复重生本身更差。

另将成熟coarse的共同颜色逐位复制给fine初态，在无N、无重新量化、contrast为0的同背景中比较。20训练视角平均PSNR，frame1下降2.86427 dB，frame2下降0.39376 dB；实际P的ΣTα积分却增加12.83% / 6.04%。world opticalmass代理守恒精度约2.4e−7，仍无法保持真实投影/遮挡覆盖。固定子核颜色采用母核方向后误差基本相同；完整parent＋零子核控制则所有视角逐像素一致。这解释fine为何从有损初态开始，不能用来解释最终已回退coarse的DC退化。

下一步机制应保留成熟母核原生覆盖，让附加细节独立成熟后再竞争alpha，并在提交前对整族辐射做互补视角预测校准；不以增加步数或继续调world质量分配参数代替。证据：`tmp/birth_diagnosis/v9_wire_audit/wire_audit.json`、`tmp/birth_diagnosis/v9_fine_initial_support/audit.json`。这些是同一已完成实验的只读机制审计，并非重新训练的消融对照。
