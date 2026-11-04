# Flame Salmon：固定15k / cap150参考帧与gate-price/regional-DC完整300帧

日期：2026-10-30

## 核心想法与协议

本实验把已经通过严格检查的Flame Salmon前30帧配方原样延伸到frame299。首帧只使用 **global step 15,000** 的SH1终点，densification上限为150k；训练终点原始151,807点，经0.0075透明度过滤后，实际共享reference为143,316点。没有选择中间checkpoint，也没有做点数sweep。

动态表示保持“候选空间层级中的区域共享运动 + 逐点残差”，沿用gate码率价格梯度缩放`1/3`、区域共享DC、量化、birth/opacity事务和50,000B完整动态更新目标。本轮没有修改公共机制代码。full300通过completed-prefix机制验证并复制frame0–29的175个文件，从frame30继续；`prefix_provenance.json`记录`past_frames_reencoded=false`，因此没有重训或重编码已完成前缀。

研究假设仍是不同区域需要不同运动共享粒度。完整序列总指标与只读frontier重编码可以检验当前表示是否被简单统一深度明显支配，但不能单独证明自适应层级带来画质因果收益。

## 配置、数据与运行

- 首帧配置：`config/flame_salmon_frame0_cap150k_15k_20261030.yaml`。
- 独立布局配置：`config/flame_salmon_cap150_layout_20261030.yaml`。
- 前30帧配置：`config/flame_salmon_cap150_gatepricefix_regdc_30_20261030.yaml`。
- completed-prefix配置：`config/flame_salmon_cap150_gatepricefix_regdc_full300_prefix30_20261030.yaml`。
- 前30帧源运行：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261030/flame_salmon_cap150_gatepricefix_regdc_30_20261030/20261030T180410_b234e6`。
- full300运行：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261030/flame_salmon_cap150_gatepricefix_regdc_full300_prefix30_20261030/20261030T183920_b08ddc`。
- 数据入口：`data/colmap4d/flame_salmon_300`软链接；300帧、19路相机、5700个图像软链接，`resolution: -1`，cam00留出评测。
- 稳定首帧资产：`/mnt/SSD500GB/my4dgs_assets/flame_salmon_cap150k_15k_20261030`。首帧checkpoint SHA256为`58bc33af10f0d5e9b7de7febdc14af2d090c28748f65a65558fa4fe3ae82c685`；reference为12,497,890B，SHA256为`ca50157177a8dce057b0defa46e4578828b7fdd2f3d028eaf9593aa77a0ca4cf`。

full300配置相对已运行的30帧保存配置只有6类必要语义差异：新实验名、末态frame29 checkpoint、`num_frames: 300`、追加未来保存帧、`resume_prefix_dir`和`resume_completed_prefix_frames: 30`；其余数据、trainer与module字段逐项一致。

## 完整结果

KB/MB均为十进制字节。动态完整更新包含motion、birth、opacity、区域DC和必要的逐帧manifest；共享reference单列。

| 指标 | 结果 |
|---|---:|
| frame0 PSNR / SSIM | 30.263462 / 0.929527 |
| frame1–299 PSNR / SSIM / LPIPS | 30.278229 / 0.929744 / 0.135936 |
| frame0–299 PSNR / SSIM | **30.278180 / 0.929744** |
| 动态完整更新均值 / P95 / 最大 | **49.871 / 55.943 / 59.247 KB** |
| 动态payload均值 | 48.663 KB |
| 共享reference | 12.497890 MB |
| 299帧动态payload / manifest | 14.550294 / 0.361050 MB |
| 299帧动态完整更新 | 14.911344 MB |
| reference + 全部完整更新 | 27.409234 MB |
| 含reference、摊至300帧 | **91.364 KB/帧** |
| 出生 / 退出GS总数 | 46,690 / 49,196 |
| 动态GS均值 / 最小 / 最大 | 139,314 / 138,255 / 140,867 |
| 最终GS | 140,810 |
| 动态训练时间 | 57.703秒/帧，合计17,253.21秒 |

这里的 **49.870716KB/帧** 是动态帧的完整更新均值，满足50KB平均目标；**91.364113KB/帧** 是把一次性reference也摊入300帧后的总序列口径，两者不能混称。P95和最大值仍超过50KB，因此通过的是平均预算，不是严格逐帧上限。

动态完整更新平均组成为：motion 40.845585KB、birth 2.761753KB、opacity 4.798900KB、区域DC 0.256953KB、manifest 1.207525KB。motion占完整动态字节的81.90%，仍是后续压缩的主要实际来源。

## 30帧分段

首段PSNR统计frame0–29，存储只统计其中动态frame1–29；其他段均统计30个动态帧。

| 帧范围 | PSNR/dB | 完整更新KB/帧 | motion KB/帧 | baseline→final/dB | 段末GS |
|---|---:|---:|---:|---:|---:|
| 0–29 | 30.262400 | 49.285448 | 39.803379 | +0.007333 | 138,590 |
| 30–59 | 30.260083 | 48.150533 | 39.424533 | -0.006185 | 138,402 |
| 60–89 | 30.298440 | 50.323300 | 41.571733 | +0.031884 | 138,778 |
| 90–119 | 30.236714 | 50.438567 | 41.588667 | +0.005554 | 138,650 |
| 120–149 | **30.376357** | 50.632133 | 41.648400 | +0.012528 | 138,963 |
| 150–179 | 30.267292 | 50.445367 | 41.258667 | +0.016908 | 139,237 |
| 180–209 | 30.244446 | 49.403767 | 40.068000 | +0.011866 | 139,429 |
| 210–239 | 30.354006 | 50.543267 | 41.349733 | +0.007741 | 140,044 |
| 240–269 | 30.235393 | 50.110067 | 40.957867 | +0.018444 | 140,695 |
| 270–299 | 30.246664 | 49.355200 | 40.750133 | +0.009741 | 140,810 |

全序列没有持续单向掉分：最高段为120–149，最低段为90–119/240–269附近；动态最差单帧是frame282的29.853674dB，随后最后10帧平均恢复到30.344218dB。该现象描述时间段难度和状态轨迹，不能仅凭分段均值归因于某个模块。

rate controller在各段末使用倍率约1.94–2.42，frame299使用2.020830、下一状态2.044870；最终累计均值49.870716KB说明闭环达到了平均预算。若继续改善，优先目标应是P95/最大值和motion主导的尾部，而不是增加首帧点数。

## 与本地SOTA及工程目标的关系

本地`sota.png`中S²GS-full的Flame Salmon为30.26dB。本实验完整300帧为30.278180dB，高 **0.018180dB**，数值上略高于该场景SOTA；优势仍很小，不能表述为有充分余量的领先。

Flame Salmon本身的SOTA约30.26dB，因此项目“全N3DV平均最好超过34.2dB”的目标不能要求这个单场景单独达到34dB；是否达到全数据集目标必须与其他场景按统一口径汇总。本实验验证的是这个困难场景已达到其本地SOTA附近，同时平均动态完整更新低于50KB。

论文表格的存储是否包含与这里完全相同的reference、manifest和属性更新字段尚未核实，所以不能用91.364KB或49.871KB直接宣称严格超过论文存储口径。

## transaction与画质波动

在frame1–299上，step200的transaction baseline与最终PSNR相关系数为0.9408；baseline→final均值为+0.011596dB，标准差0.049891dB，126/299帧为负。delta方差约为最终PSNR方差的12.5%。截至frame239的独立中途诊断也得到相关系数0.932、约14.1%的方差比例，并且只有30–59窗口的平均delta为负。

这些是相关性证据：大部分时序波动在step200之前已经存在，birth/opacity/DC/提交事务不是当前全局平均画质短板的主要来源。它不能证明具体因果，也不排除transaction对部分单帧有负作用。

## 层级frontier只读重编码

对9个保存checkpoint做了只读反事实重编码：固定该帧总量化逐点运动、survivor和birth，只替换区域frontier；统一深度的区域代表取coordinate median。学习frontier的packet必须逐字节复现原motion文件，且所有对照解码后的点位置和旋转保持一致。

| 帧 | 学习frontier | active regions | residual rows | 最佳统一深度 | 学习值相对差异 |
|---|---:|---:|---:|---:|---:|
| 59 | 42,758B | 81 | 21,759 | 43,506B（depth4） | -748B |
| 89 | 39,654B | 108 | 19,893 | 40,438B（depth4） | -784B |
| 119 | 42,306B | 106 | 21,727 | 43,042B（depth4） | -736B |
| 149 | 40,790B | 152 | 20,425 | 41,598B（depth4） | -808B |
| 179 | 36,726B | 112 | 17,891 | 37,518B（depth4） | -792B |
| 209 | 38,798B | 140 | 19,072 | 39,722B（depth4） | -924B |
| 239 | 40,110B | 61 | 20,036 | 40,650B（depth4） | -540B |
| 269 | 41,930B | 103 | 20,345 | 42,854B（depth4） | -924B |
| 299 | 43,450B | 173 | 21,626 | 44,002B（depth4） | -552B |

学习frontier在9/9个采样帧都比这一定义下的最佳统一深度小，平均少约756B，说明当前层级至少没有被简单统一深度明显支配。但是coordinate median不是全局最优字节下界，这不是重新训练后的固定粒度消融；画质在重编码中按构造保持不变，因此这些结果不能证明PSNR因果，也不能证明gate训练达到全局最优。证据位于`/mnt/SSD500GB/my4dgs_diagnostics/flame_salmon_cap150_frontier_frame{059,089,119,149,179,209,239,269,299}_20261030.json`。

## 严格核验与速度

- CPU从共享reference顺序重放frame1–299，每帧manifest的decoded/raw状态哈希均匹配；最终140,810点，decoder SHA256为`d8c10ff69f84354dfc0d0d247d2cc3b431095e30e88b60a4424a2a4a78f1e8d6`，raw SHA256为`038bd7cd9da4070b87f6b84ca37ebef19bf70cdd60feb27aee3bc7401afa5c3d`。
- 299组motion/birth/opacity/DC/manifest的实际文件字节与summary全部一致；payload为14,550,294B，manifest为361,050B，完整动态更新为14,911,344B。frame1–299的rate feedback输入、使用倍率、下一状态及最终controller checkpoint逐帧一致。
- completed-prefix的175个文件在源、provenance与目标目录三方哈希一致，`past_frames_reencoded=false`。frame59、89、119、149、179、209、239、269、299共9个续跑checkpoint均为frame完成态300步，GS数和累计实际字节一致。
- 最终checkpoint SHA256为`9c1fa4c13f30a929dde80d9676b89a079ae35e38a8ad9b8c937eab7ce05856de`。frame299 checkpoint与连续stream回放的xyz、rotation、logscale、opacity、SH-DC和SH-rest全部逐位一致，渲染像素最大差为0。
- GPU sequence stream复核300/300帧，保存质量差和cache像素差均为0。RTX 3090 Ti上全300帧已解码材质渲染为 **263.960FPS**，动态帧为 **264.009FPS**；逐帧最低249.914FPS、P05 258.132FPS、中位数264.573FPS，达到≥200FPS工程目标。
- frame299物化后渲染为267.681FPS。完整播放连续处理frame1–299，包含本地读包、完整性校验CPU解码、raw GS上传、同步forward与clamp，两轮分别10.614/10.757FPS，合计 **10.685FPS**。它不包含一次性reference、相机/GT加载、manifest验证和质量指标计算，且OS缓存可能有利；该值与纯渲染FPS分列。
- 所有训练及验收阶段均未出现NaN、Traceback、CUDA错误或Xid；GPU基准的每个独占检查点均无foreign compute context。

主要证据：

- `/mnt/SSD500GB/my4dgs_diagnostics/flame_salmon_cap150k_15k_20261030/full300_cpu_stream_rate_checkpoint_audit.json`
- `/mnt/SSD500GB/my4dgs_diagnostics/flame_salmon_cap150k_15k_20261030/full300_sequence_benchmark.json`
- `/mnt/SSD500GB/my4dgs_diagnostics/flame_salmon_cap150k_15k_20261030/full300_frame299_checkpoint_stream_playback.json`
- `/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261030/flame_salmon_cap150_gatepricefix_regdc_full300_prefix30_20261030/20261030T183920_b08ddc/prefix_provenance.json`

## 已验证结论、边界与下一步

已验证：严格15,000-step、cap150共享首帧与当前动态配方可以在Flame Salmon完整300帧上同时达到场景SOTA附近画质、平均动态完整更新低于50KB、已解码渲染高于200FPS；码流、rate链、checkpoint、CPU/GPU解码、实际像素和完整播放均已核验。

尚未达到：P95和最大动态更新仍高于50KB，完整播放10.685FPS与已解码渲染263.960FPS差距明显；全N3DV是否超过34/34.2dB还必须等其他场景统一汇总。

因果边界：本实验验证的是“15k cap150 reference + gate价格梯度修复 + 区域DC + 当前birth/opacity机制”的整体结果，不能把收益全部归因于自适应层级。9个frontier重编码只支持当前层级在定义明确的统一深度对照下具有竞争力，不是全局最优证明或训练消融。

下一步对Flame Salmon本身应优先压缩占81.9%的motion尾部并优化CPU解码/上传播放路径，同时保持当前画质；不建议通过继续增加首帧GS解决问题。整体研究方向应由所有N3DV场景的统一结果决定。

## 前30帧资格实验

前缀使用 exact-15000 step、SH1、150k增密上限，透明度过滤后143,316点；动态训练使用 gate 选择价格系数1/3、区域DC和50,000 B目标。frame1–29为 **30.262363 dB / 49.285448 KB**，含首帧的30帧为30.262400 dB；完整更新P95/最大为56.559/58.922 KB。已解码渲染261.200 FPS，含读包和CPU解码的完整播放11.375 FPS，口径分列。

同为15k首帧的旧175k入口为29.997435 dB / 62.344655 KB，但首帧、gate价格和区域DC同时变化，不能单项归因。29帧CPU连续解码及30帧GPU重渲染通过，checkpoint与码流六属性逐位一致。前缀运行：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261030/flame_salmon_cap150_gatepricefix_regdc_30_20261030/20261030T180410_b234e6`；核验位于 `/mnt/SSD500GB/my4dgs_diagnostics/flame_salmon_cap150k_15k_20261030/`。完整运行直接续接该前缀，最终判断以本文300帧结果为准。

## 更早的前30帧实验

以下均为frame1–29的动态平均，完整更新含必要manifest，首帧另计；这些方案与上面的完整300帧使用不同首帧或训练设置，不能跨范围直接比较。

| 候选 | 首帧来源 | PSNR / dB | 完整更新 / KB | 观察 |
|---|---|---:|---:|---|
| cov125k | 12,500步checkpoint、125,000点 | 30.198476 | 55.084103 | 不符合当前固定15k终点协议 |
| exact15k/cov175k | 15,000步checkpoint、175,000点 | 29.997435 | 62.344655 | 点数和首帧一起改变，画质与码率均退化 |
| cov125k warm3 + 区域DC | 同一12,500步、125,000点首帧 | 30.183858 | 48.028483 | 相对cov125k省12.81%，PSNR下降0.014618 dB |

前两轮cam00参与首帧选择，不能视为独立泛化评估。第三轮同时将码率倍率从3启动，并启用与运动共享owner的区域DC；前10帧更新从60.459降到46.248 KB，符合冷启动开销下降的解释，但不能分离两项机制的贡献。第三轮29帧CPU连续解码及GPU重渲染通过，已解码渲染273.989 FPS，完整播放12.697 FPS；前两轮只测了30帧已解码渲染273.580/241.192 FPS，没有完整序列播放结果。

运行：`output/flame_salmon_recipe30_20261029T144442/flame_salmon_cov125k_first30_20261029T144442/20261029T144715_186756`、`output/flame_salmon_f15000_first30_20261029T151112/flame_salmon_f15000_gs175000_first30_20261029T151112/20261029T152202_90bb68`、`output/goal_n3dv_dynamic_20261030/flame_salmon_cov125k_warm3_regdc_30_20261030/20261030T041614_f5c414`。[125k逐帧数据](assets/20261029_flame_salmon_cov125k_first30_20261029T144442.csv)、[175k逐帧数据](assets/20261029_flame_salmon_f15000_gs175000_first30_20261029T151112.csv)。
