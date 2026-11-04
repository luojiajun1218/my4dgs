# Cook Spinach：trained-probe A/C tail commit 全300帧

日期：2026-10-31

## 核心思路与实验边界

本轮把已经严格验收的前30帧 trained-probe 生产结果原样作为前缀，从 frame30 续跑到
frame299。每帧在固定内部训练 probe `cam05/10/15/20` 上比较：

- A：仅提交运动后的 surviving old GS，不提交本帧旧点 opacity/DC/log-scale 与 newborn；
- C：提交完整旧点属性更新与 newborn。

四个 probe 与其余16个训练相机一样参与其可训练阶段，外部 `cam00` 不参与训练或选择，只在
选择冻结后作事后诊断。连续 A 最多3帧，第4次仍偏好 A 时强制 C 刷新。研究假设仍是区域共享
粒度和逐点残差的权衡；本次结果只验证这一整套生产配置在 Cook 上的表现，不能把总指标单独
归因于自适应空间层级。

## 配置、代码与运行

- 历史训练配置已退役；运行时冻结配置保存在下方运行目录的 `config.yaml`，仅用于记录和回放。
- 运行目录：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cook_spinach_twostage_cap150_gatepricefix_regdc_scale1024_tailac_trainedprobe_full300_prefix30_20261031/20261031T054805_89d18f`。
- 标准300帧软链接入口：`data/colmap4d/cook_spinach_300`；`resolution: -1`，SH1。
- 首帧继续使用 exact-15k 两阶段稳定资产，opacity 0.0075 后为152,557 GS；没有复制数据或
  改写共享首帧。
- 前30帧从已验收 run 复制并逐文件校验，frame30–299 从 frame29 完成态 checkpoint 续跑；
  历史前缀未重新编码。
- 动态配置保持 target 50,000 B、initial rate multiplier 3、gate price 1/3、区域 DC、
  motion-supported isotropic log-scale（最多1024行）。

冻结哈希：

- `pipeline/frame_plan.py`：`2fa25c4af75827ff1ce79e0123567c301c3f892c1e90eb71641db47e07515386`
- `pipeline/module.py`：`6415df5594196cecdc425085e2124326809fbd9a3a961a9ef6b789b699a16cb6`
- full300 配置：`76b325e91fdc06b2f9d76d66e8e8903ddf0e23e04550ad35ac0288f6eceefbf7`

本轮没有再修改训练核心。为在训练结束后对300帧做只读 A/B/C 归因，仅把既有
`scripts/diagnose_tail_commit_variants.py` 的硬编码29动态帧改为从冻结 run config 读取
`num_frames`，严格遍历 `1..N-1`，并把报告分母动态化；该历史诊断脚本及其专项单测现已移除。

## 全300帧结果

KB 均为十进制 1,000 B。完整动态更新包含 motion、birth、opacity、DC、log-scale packet 和
该帧必要 manifest；共享 frame0 reference 单列。

| 指标 | 结果 |
|---|---:|
| frame0 cam00 PSNR | 34.063683 dB |
| frame1–299 PSNR | 33.921797 dB |
| frame0–299 PSNR | 33.922270 dB |
| 动态 SSIM / LPIPS | 0.960516 / 0.130948 |
| 动态 PSNR 最小 / 最大 | 33.002285 / 34.666023 dB |
| 完整动态更新均值 | 50.081886 KB/帧 |
| 完整动态更新 P95 / 最大 | 59.768000 / 80.970000 KB |
| 不超过50 KB的动态帧 | 124/299 |
| 动态 packet payload | 47.976773 KB/帧 |
| 动态完整更新总量 | 14,974,484 B |
| 共享 frame0 reference | 13,285,176 B |
| reference + 动态完整更新 | 28,259,660 B |
| 最终 GS 数 | 145,999 |
| 动态训练 / wall time | 63.618 / 63.910 s/帧 |
| 已解码渲染 FPS，300帧 | 318.010 |
| 已解码渲染 FPS，动态帧 | 318.104 |
| frame299 已解码渲染 FPS | 323.397 |
| 完整播放 FPS | 11.512 |

本地 `sota.png` 中 S²GS-full 的 Cook Spinach 为33.96 dB：

- full300 比33.96低 `0.037730 dB`；
- 比工程底线34.0低 `0.077730 dB`，更未达到最好34.2 dB目标；
- 完整更新均值比50,000 B高 `81.886 B/帧`，即 `0.164%`，属于极小幅放宽，但不能写成
  严格低于50 KB。

因此本轮没有同时达到全300帧的画质和严格50 KB目标。渲染速度超过200 FPS；完整播放速度
按包含读包、CPU解码、更新与渲染的口径单列，没有用纯渲染 FPS 替代。

## 序列趋势与 A/C 行为

累计均值在中段曾明显过线，后段困难内容把全局均值拉回：

| 累计帧 | cam00 PSNR | 完整更新 B/动态帧 |
|---|---:|---:|
| 0–29 | 33.964293 | 49,598.93 |
| 0–149 | 34.066363 | 48,783.35 |
| 0–179 | 34.093984 | 49,293.20 |
| 0–209 | 34.020247 | 50,039.58 |
| 0–239 | 33.995004 | 49,894.86 |
| 0–269 | 33.957567 | 49,747.89 |
| 0–299 | 33.922270 | 50,081.89 |

末段分窗：frame270–279 为33.693437 dB / 49,500.5 B，280–289 为33.567526 dB /
54,944.0 B，290–299 为33.552832 dB / 54,785.7 B。rate controller 已把全局均值压到目标
附近，但最后20帧连续 C 且内容更难，局部码率仍反弹。

生产选择为 C 243/299、A 56/299；内部原始偏好为 C 241、A 58。frame140 与150因连续 A
达到3帧而强制 C，最大 streak=3，最终 streak=0。全部299个 manifest 均声明
`probe_trained_views=true`、motion `steps1..200`、旧属性和 birth/tail 正常参与，且明确外部
test camera 不参与训练和选择。

完整更新分量总量为：motion 11,408,014 B、birth 1,073,026 B、opacity 1,232,478 B、
区域 DC 87,029 B、log-scale 544,508 B、manifest 629,429 B。motion 仍是主要开销。

## 只读 A/B/C 尾阶段归因

训练完成后从每一帧的前一帧 canonical C 状态出发，重建三种一次性反事实：

- A：当前 motion + surviving old GS 的上一帧属性；
- B：A + 当前旧点 opacity/DC/log-scale，不含 newborn；
- C：完整当前帧，包含 newborn。

每帧 B 都与 canonical decoder 的 C-old-prefix 六属性逐位一致。四 probe 的选择在读取 cam00
图像与保存的 cam00 指标前冻结。下表中的字节是“当前帧实际 packet 子集 + 现有 manifest”的
诊断估计；它不是可直接播放的 A/B 替代序列，因为后续帧仍沿 canonical C 状态递推，真正部署
必须重新写 manifest 并重编码后续包。

| 范围 | A cam00 | B cam00 | C cam00 | B−A | C−B | A/B/C 字节均值 |
|---|---:|---:|---:|---:|---:|---:|
| 1–299 | 33.898505 | 33.902936 | 33.921797 | +0.004432 | +0.018861 | 40,259 / 46,493 / 50,081.9 |
| 240–269 | 33.617418 | 33.629179 | 33.658076 | +0.011761 | +0.028897 | 38,070.8 / 44,762.8 / 48,577.0 |
| 270–299 | 33.578433 | 33.587649 | 33.604598 | +0.009216 | +0.016950 | 41,150.4 / 48,254.3 / 53,076.7 |

内部 probe 目标（四相机 mean RGB MSE 转成的 PSNR）在全序列 A/B/C 分别为
35.557511 / 35.619217 / 35.676593 dB；240–269 为35.529626 / 35.580767 /
35.660839，270–299 为35.374103 / 35.455189 / 35.516902。也就是说训练 probe 对旧属性和
newborn 都稳定偏正，但 cam00 的增益小得多，存在明显的训练视角到外部视角泛化差距。

更有诊断力的是只看 `C < A` 的101个失败帧：

- 旧点属性 `B−A` 平均 `-0.140536 dB`，101帧中100帧为负；
- newborn `C−B` 平均 `+0.014355 dB`，101帧中69帧为正；
- 同一批失败帧在内部 probe 上，`B−A` 仍为 `+0.072543 dB`，`C−B` 为
  `+0.060924 dB`，`C−A` 101/101 均为正。

分段也一致：240–269 的10个失败帧，旧属性为 `-0.167305 dB`，newborn 为
`+0.017472 dB`；270–299 的12个失败帧分别为 `-0.120030 dB` 和 `+0.007510 dB`。
因此已验证的归因是：外部负泛化主要来自旧点属性更新，特别是它在训练 probe 上获益、却在
cam00 上恶化；newborn 整体在补偿这一损失，不是主要负因。这个结论针对当前 Cook 序列和
当前反事实定义，不应直接推广到其他场景。

内部 A/B/C 选择为 A58/B5/C236，事后与 cam00 oracle 一致165/299；只允许 B/C 时为
B61/C238，一致238/299。只允许 B/C 的 cam00 均值为33.921862 dB、估计50,007.57 B/帧，
相对保存 C 只高约0.000065 dB，不足以靠简单 B/C commit 补足 SOTA 缺口。

## 严格验收

- CPU 从 reference 连续解码 frame1–299；300/300帧重新渲染与保存的 PSNR/SSIM/L1 最大差
  均为0，缓存/非缓存像素最大差为0。
- 299个 summary、299个 manifest、299个 motion packet 及所有可选属性/birth packet 的
  磁盘实际大小、manifest声明、保存 summary 与 rate controller 累计完全闭合；没有 `.tmp`
  或 `.part` 残包。
- frame299 checkpoint 为 `current_frame=299`、`global_step=300`、SH1、145,999 GS；六属性
  全部有限，prepared/motion_committed/finalized 均为 true。
- frame299 checkpoint、manifest、summary 的 raw SHA256 三方一致：
  `c9834522c3b32cd8a2bfdc92026ba6fb408c5ac639cffd4e569f04edfa2b4df6`；decoded state SHA256
  为 `3fd3c2a63f8a553447e1864a8a0a3203ea13069a72f40d393dcb25bc8a31bd29`，与 manifest 一致。
- frame299 checkpoint 与连续 stream 的 xyz、rotation、logscale、opacity、SH-DC、SH-rest
  六属性逐位一致，checkpoint/stream 渲染像素最大差为0；checkpoint 文件 SHA256 为
  `81de15b5efe2168523379f7db5481fa5c180fe18d205564607c1769f47e18c5e`。
- 300帧渲染 FPS 使用每帧10次 warmup + 50次同步完整 render；完整播放 FPS 使用两次
  frame1–299 pass，逐帧包含本地读包、完整性检查的 CPU decode、host-to-GPU安装、同步渲染。
- A/B/C 诊断覆盖严格 frame1–299，保存前后 source run 文件大小和 mtime 快照完全相同；
  结果文件四项 checksum 均通过，保存 C 的 PSNR 最大复现差为0，raw shape/SH/manifest 与
  B-old-prefix 检查全部通过。
- 诊断通用化 CPU 单测2/2通过，脚本编译检查通过；没有启动新训练或重跑实验。

证据目录：

- `/mnt/SSD500GB/my4dgs_diagnostics/cook_spinach_twostage_cap150k_15k_trainedprobe_full300_20261031/sequence_benchmark.json`
- `/mnt/SSD500GB/my4dgs_diagnostics/cook_spinach_twostage_cap150k_15k_trainedprobe_full300_20261031/frame299_checkpoint_stream_playback.json`
- `/mnt/SSD500GB/my4dgs_diagnostics/cook_spinach_tail_abc_full300_20261031/results.json`
- `/mnt/SSD500GB/my4dgs_diagnostics/cook_spinach_tail_abc_full300_20261031/per_frame.csv`

## 结论与下一步

本轮生产机制在前180帧一度超过34 dB，最终却因后段内容困难和旧点属性的外部负泛化降至
33.922270 dB；50 KB累计控制基本成功，但最后20帧高 C 比例使全局均值略超预算。不能把这次
失败解释成简单的 newborn 过多，也没有证据表明状态/hash损坏。

下一步最有依据的生产改进方向是约束旧点 opacity/DC/log-scale 的外部泛化风险，而不是继续
扩大 newborn 或盲目增加点数：例如构造不依赖 cam00、与图像误差选择解耦的训练内一致性/保守
提交条件，或对旧属性残差施加跨训练视角一致性与更强的近端限制。任何新机制仍须保持 cam00
只作事后验收，并从30帧生产门开始；本轮结果不授权自动重跑 full300。

## 前30帧资格实验

20个训练相机均参与各自可训练阶段；其中固定的 `cam05/10/15/20` 在step300作为内部训练probe选择仅提交运动A或完整旧属性与新点C，外部cam00只在选择后评估。修复此前probe不参与训练导致的分布错位；这是 `pipeline/frame_plan.py` 与 `pipeline/module.py` 的整套机制，不是层级的单项消融。

frame1–29为 **33.960866 dB / 49.598931 KB**，frame0–29为33.964293 dB；动态SSIM/LPIPS为0.962253/0.123668。选择C 26帧、A 3帧，未触发连续A强制刷新；内部probe和cam00事后oracle一致17/29。已解码渲染307.699 FPS，包含读包及CPU解码的完整播放11.169 FPS。29帧顺序解码、30帧重渲染和最终checkpoint六属性核验通过。与历史完整C运行的动态均值相比，PSNR提高0.047752 dB、完整更新增加247.14 B/帧；前30帧接近本地SOTA数字不代表完整序列必然超过。

前缀运行：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cook_spinach_twostage_cap150_gatepricefix_regdc_scale1024_tailac_trainedprobe_30_20261031/20261031T044139_0f29f9`；核验位于 `/mnt/SSD500GB/my4dgs_diagnostics/cook_spinach_twostage_cap150k_15k_trainedprobe_20261031/`。完整运行从该前缀续接，最终判断以本文300帧结果为准。
