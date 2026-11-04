# Cut Roasted Beef：support-safe exact15k SH3 / cov150 full300 延伸（frame129 提前停止）

## 结论

本实验原计划从已验收的 30 帧前缀继续到 300 帧，但在 frame119 质量硬门判定失败后提前停止。**这不是完整 300 帧结果，不能作为 full300 指标，也不能用于宣称超过 SOTA。** 停止信号发出前 frame129 已完成并落盘；frame130 只训练到约 step40，没有生成 summary、manifest 或 motion 包，因此最后一个完整动态帧是 frame129。正式保留、已做 checkpoint 严格验收的恢复点仍为 frame119。

质量失败不是码流损坏造成的：截至 frame119，所有动态包与 manifest 字节逐帧闭合，六个 GS 属性 finite，CPU 连续回放与 checkpoint 六属性逐 bit 一致。问题是随时间持续积累的重建质量漂移。动态帧 1–119 平均仅 **33.666541 dB**，最近 30 帧 90–119 只有 **32.897152 dB**；若要让完整 299 个动态帧达到 Cut 的 34.12 dB 目标，剩余 180 帧必须平均达到 **34.419787 dB**。前 119 帧中只有 6 帧达到这一恢复水平，且最近 60 帧最高仅 **34.192188 dB**，因此触发预注册停止门。

## 协议与运行信息

- 配置：`config/cut_roasted_beef_sh3_supportsafe15k_cov150_gatepricefix_regdc_full300_prefix30_20261031.yaml`，SHA256 为 `42fe6626c9dd6bedfc1394db7b4b6d84b01d00b1d81afe9c820264fbb8191f6a`。
- 运行目录：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cut_beef_sh3_supportsafe15k_cov150_gatepricefix_regdc_full300_prefix30_20261031/20261031T115257_aed9f1`。
- 续跑来源：已严格验收的 frame0–29 前缀，未重新编码历史帧；prefix aggregate SHA256 为 `32e2935a359ff489a920bc81aac581f8bc10dd2e9d3312be21efc8041abeb9ac`。
- 初始 checkpoint：first30 的 finalized frame29，SHA256 为 `263306c38efed73a79b2e2e8dee50f5528aae9f411c38dda6983b611521038e6`。
- 首帧资产仍是 exact15000 / SH3 / resolution -1 / exact 150,000 GS 的 support-safe 资产；数据仍通过 standard300 软链接读取，没有复制数据或修改链接目标。
- 动态协议与 first30 逐字段一致：50,000 bytes 目标、initial multiplier 3、gate price 1/3、区域 DC、关闭 log-scale；没有启用 Cook 专用 decoupled 属性策略。
- 运行时冻结源码：`pipeline/frame_plan.py` 为 `99530305baf7a7b081f405728a8ba974fe5262d9749d0f78a155337579bf014c`，`pipeline/module.py` 为 `ee0c18d1fa7b98c99e94cabf78bccc972cd45b8204cccd8ab205dd9088bb4eac`。

## 停止时结果

质量决策点与最后完整落盘点分开报告：

| 范围 | 状态 | 动态 PSNR | 含 frame0 PSNR | 完整更新 KB/帧 | 最终 GS |
| --- | --- | ---: | ---: | ---: | ---: |
| frame1–119 | 预注册停止判定与严格 checkpoint 验收点 | 33.666541 | 33.673142 | 50.041050 | 138,213 |
| frame1–129 | 停止信号生效前最后完整前缀 | 33.611935 | 33.618449 | 50.254705 | 137,673 |

这里 KB 固定为 1000 bytes，完整更新包含 motion、birth、opacity、区域 DC 和逐帧 manifest。frame1–129 的完整更新总量为 6,482,857 bytes，p95 为 56.542 KB，最大值为 59.665 KB。最低 PSNR 是 frame107 的 32.585224 dB，最高是 frame2 的 34.497368 dB。

分段结果清楚显示 first30 的优势没有保持到长序列：

| 动态帧 | PSNR | 完整更新 KB/帧 | 段末 GS |
| --- | ---: | ---: | ---: |
| 1–29 | 34.223070 | 49.133655 | 142,171 |
| 30–59 | 33.899188 | 52.150900 | 140,216 |
| 60–89 | 33.665304 | 48.034533 | 139,158 |
| 90–119 | 32.897152 | 50.814867 | 138,213 |
| 120–129 | 32.962132 | 52.797200 | 137,673 |

frame1–129 的完整更新平均组成是 motion 41.123705 KB、birth 2.151326 KB、opacity 5.316829 KB、区域 DC 0.415217 KB、manifest 1.247628 KB；log-scale 为 0。最近 30 帧 frame100–129 为 **32.874560 dB / 50.921533 KB**，画质与码率均未回到 first30 水平。

## 完整性验收与停止边界

- frame1–129 的 129 份 summary 数值均 finite；每帧实际 packet 文件加 manifest 大小与 summary 的完整更新字节逐帧相等。
- 从共享 reference 连续 CPU 回放到 frame129 成功，得到 137,673 GS，`decoder_sha256=4d42992c82b3cec936efe83752b75c4698ca43347f5c0e6a88c8fec9e5d57054`。
- frame119 checkpoint SHA256 为 `b83bc73113e3c57aa31f199d5950315e858a4b5047bdee1220156f84aade9328`；状态为 `current_frame=119`、`global_step=300`、`prepared=true`、`finalized=true`、`motion_committed=true`、SH3、138,213 GS。
- frame119 的 xyz、rotation、logscale、opacity、SH-DC、SH-rest 均 finite；连续 stream 与 checkpoint 的六属性逐 bit 一致，decoder digest 与 raw digest 均和 manifest 一致。
- frame119 rate multiplier 为 2.786700；controller 到该帧累计 5,954,885 bytes。
- frame130 没有 summary、manifest 或 motion 包，仅保留一次未完成训练的 step diagnostics。未修改或清理任何历史产物。
- 停止后训练主进程及 worker 均退出；GPU 为 0% utilization、400 MiB 基础占用且无 compute process。

由于实验未完成 300 帧，本轮不运行、也不报告 full300 GPUStream 连续验证、已解码渲染 FPS 或完整播放 FPS。first30 的速度和逐像素严格验证仍以本文后附前 30 帧记录为准，不能外推到本次未完成序列。

## 结果分析与下一步

已有只读诊断显示，退化在 birth 之前的 transaction baseline 就已经存在，且 stage200 到 final 的附加损失在前后段近似不变；因此不能把主因归结为某次 birth 编码或解码错误。与此同时，GS 从 frame29 的 142,171 持续下降到 frame119 的 138,213、frame129 的 137,673，训练图像损失也随序列后移而升高。结合连续回放与 bit-exact 验收，当前证据更支持“长时动态拟合与拓扑/属性支持逐渐不足”，而不是 resume 错位、packet 损坏或单纯末段编码误差。

这仍是相关性证据，不能仅凭总指标把失效归因于某一个 gate、区域粒度或 GS retirement 规则。下一步最有价值的是针对长序列的质量保持机制做一次生产级修正，并用本轮 frame119 前缀作为诊断基线；**不应从 frame119 沿相同机制继续生产，也不应因为 first30 达标就直接恢复 full300**。后续候选仍需同时预注册分段 PSNR、完整更新字节、GS 支持变化以及完整300回放/FPS门槛。

## 前30帧资格实验

首帧使用 exact-15000 step、SH3，经过support-safe增密及覆盖与贡献度筛选后为150,000 GS；动态使用50,000 B目标、gate价格1/3和区域DC，关闭log-scale。frame1–29为 **34.223070 dB / 49.133655 KB**，frame0–29为34.230924 dB；动态SSIM/LPIPS为0.962876/0.116542，完整更新P95/最大为54.389/55.000 KB。帧1–10、11–20、21–29分别为34.407993/44.633、34.082829/50.814、34.173423 dB/52.267 KB，后段码率已上升。

旧exact15k SH3候选的动态前缀为33.860999 dB / 51.814724 KB；本轮新首帧资产整体改善，但不能把收益单独归因于某个覆盖排序或层级细节。29帧CPU连续回放、30帧GPU stream及最终checkpoint六属性核验通过；已解码渲染279.370 FPS，含读包和解码的完整播放10.780 FPS。运行：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cut_beef_sh3_supportsafe15k_cov150_gatepricefix_regdc_30_20261031/20261031T110445_f154f8`。后续从此准确前缀续跑并于frame129停止，不能把前缀成绩作为完整300帧结果。
