# Shadow源族出生：单模型最多300步短3帧

完成日期：2026-10-26。共享frame0＋2个动态帧训练、独立真实码流解码与全场景渲染测速均已完成。本轮解决了“先把一批固定密度新核塞进场景、再训练修复其扰动”的初始化问题，但第二帧测试质量仍下降，未达到目标，不进入长帧。

## 核心机制与代码

保留源GS的完整θ0母核、原姿态/协方差与SH1。step200时所有未接收族都透明，原父仍在场景中。按真实native残差责任排序逐源族提案：当前已接收场景作为背景，一次有界2×2真实响应解初始化两子DC；固定颜色和几何，在共享密度ρ=0、0.5、1处渲染，重建四次MSE多项式，求一次三次导数根得到唯一ρ*。只有该ρ*的真实codec全20训练视角MSE严格改善、字节和点数合法时才出生，否则维持不出生。没有密度网格搜索、反复拟合或额外SGD。

接收后两子alpha为`.99ρ*`，前50个birth更新以这个数据密度作成熟下界，不抬回0.1；后50更新释放母/子密度竞争。初始化后裁掉所有未接收族，固定真实存活源预测器，逐bit重放六属性后才进入单模型、单Adam的100步训练。最终仍按整源族、同一all20真实码流目标选择；撤回一族会恢复它的原父。已通过的初始化trial/model状态只作为同一轨迹的回滚检查点，不另训分支。无族出生时允许200步结束，本轮两帧均有出生、实际各300步。

- 新`ShadowSourceResidualModel`管理透明影子、实体mask、每族ρ/成熟下界及capture/subset；保留固定物理SH1和受界DC。
- 新`shadow_birth_initializer.py`实现一次颜色解与quartic/cubic密度提案；`module.py`集成逐族真实codec验收、实体人口裁切、100步与最终回滚。
- `residual_birth.py`支持最终逐源族原子选择；`sequence_extension.py`核对200或300的完成状态、实际optimizer账本与prefix。
- Shadow专属修正：DC不随codec predictor重选而重新舍入到不同颜色格点，保持固定物理颜色；沿现有ATTR精确回退并真实计费，未改码流格式。旧Quick/V12与历史冻结源不变。

始终单模型、全部20训练相机；cam00只评估。固定132586 GS首帧，沿既有过滤保留128963；1345×1008半分辨率、learnable gate、motion200、学习率及候选上限保持。

## 实际结果

| 指标 | frame1 | frame2 |
|---|---:|---:|
| 新增前cam00 PSNR | 33.522053 | 33.901649 |
| 最终cam00 PSNR | 33.538067 | 33.785553 |
| 新增阶段收益 | +0.016014 dB | −0.116096 dB |
| 原场景all20 MSE | 0.0002356477 | 0.0002454986 |
| 初始化后all20 MSE | 0.0002345700 | 0.0002421262 |
| 最终真实人口all20 MSE | 0.0002334300 | 0.0002417516 |
| 测试源族 / 初始化接收族 | 110 / 49 | 113 / 59 |
| 初始化点 / 最终点 | 147 / 138 | 177 / 162 |
| 初始化ρ min / median / max | 0.004645 / 0.184258 / 1 | 0.005100 / 0.644009 / 1 |
| 退休数（含被替换父） | 366 | 246 |
| 最终总GS | 128735 | 128651 |
| 实际motion / birth更新 / 总步 | 200 / 100 / 300 | 200 / 100 / 300 |
| 完整更新（含manifest） | 99.434 KB | 42.210 KB |
| 训练耗时（含闭式初始化/codec检查） | 569.41 s | 345.41 s |

两帧都选训练后prefix，没有额外训练/refit。initializer optimizer更新和提交后更新均为0。初始化及最终all20 MSE均优于各自原场景，最终也优于初始化。

独立stream解码：动态两帧平均PSNR **33.661810 dB**；全3帧 **33.513826 dB**。动态完整更新平均 **70.822 KB/帧**，payload平均69.989 KB；共享首帧11081043字节另计。全场景渲染 **280.39 FPS**，动态帧277.05 FPS；每帧10次预热、50次同步完整forward＋clamp，不含解码/上传/指标。解码与保存的PSNR/SSIM/L1差均0，属性cache开关像素差0。KB按十进制计，短序列首包较大，不能据此推断299动态帧均值。

Gate分布：frame1 active regions=1000，深度1/2/3/4为1/21/179/799，open/eligible gates=161/583；frame2为298，深度分布3/28/71/196，open/eligible=65/583。候选节点均4681，gate机制未修改。

## 结果含义

本轮新点在进入SGD之前就使完整训练场景MSE下降，避免了V12初始化后误差仍高于原场景的问题。数据密度也覆盖很宽的范围，说明统一固定初生alpha不能代表实际需要的密度。

该改进仍不足以带来测试质量：frame2全20训练MSE持续下降，cam00却下降0.1161 dB。现有证据支持进一步检查源GS支持、遮挡和材质修复的跨视角可迁移性；不能把本轮失败单归为未充分训练，也不能只根据本轮把损伤归给某一个属性。本轮没有额外做属性对照或长序列训练。

闭式初始化的实际计算成本较高。每族都要进行20视角颜色响应、三密度图和实际码流检查，本实现还重复重建完整motion/opacity码流；主要耗时集中在这段有界只读计算。300步是实际梯度更新上限，并不表示总耗时很短。若继续这一机制，需要单独审视实现效率；本轮未在跑中改变方法或参数。

## 验证与产物

- 完整CPU回归553项通过（CUDA禁用，6项原生项跳过）；最后物理DC修复后29项相关检查通过。命令为`CUDA_VISIBLE_DEVICES='' PYTHONPATH=tests:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests`及最终shadow/quick/prefix目标组。
- 三族原生证明：先拒、再接、第三probe后拒；ρ0原生图像严格等于当前prefix，full-row到packed-row映射非trivial，后probe保持已接收族实际codec六字段逐bit不变，裁切后SGD入口重放exact；0SGD/optimizer。小场景MSE为1.31783e−5→2.75581e−7。
- CPU另覆盖：最终拒一族恢复其原父并保留另一族；训练损伤回退同一初始化状态；全拒时真实200步提交、0空SGD；200/300 prefix账本；不同DC predictor删行重定基准的六属性ATTR回环。
- 运行：`output/flame_steak_1025/fs_shadow_source_residual_v13_short3/20261026T104656_f4fb30`。
- 冻结：`tmp/experiments/shadow_source_residual_v13_short3`，113份源文件，复用hash验证的446份V9 immutable native文件；完成后源hash复核一致。
- 训练/benchmark日志：`tmp/birth_diagnosis/shadow_source_residual_v13_short3_train.log`、`shadow_source_residual_v13_benchmark.log`。
- 解码/汇总：`tmp/birth_diagnosis/shadow_source_residual_v13_decoded.json`、`shadow_source_residual_v13_summary.json`。
- 原生证明：`tmp/birth_diagnosis/shadow_v13_native_validation.json`；CPU日志`shadow_v13_full_cpu_tests.log`、`shadow_v13_final_related_tests.log`。
