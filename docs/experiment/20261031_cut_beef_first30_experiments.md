# Cut Beef：2026-10-31 日前 30 帧实验

## Cut Roasted Beef：严格 15k SH3 覆盖压缩的前 30 帧

> 后续更新：本文只记录当时旧 SH1 与该 SH3 资产的动态比较，不构成对所有 SH1 模型的判断。当前首帧入口已改用新训 SH1 / exact15k / 150k，见[首帧处理指南](../first_frame_recipe.md)；新 SH1 动态结果尚待验证，本文历史数据保持不变。

### 结论

本轮完成 frame0–29，共 30 帧。首帧严格来自 `global_step=15000`、完整激活 SH3 的 checkpoint，经训练视角贡献度与空间覆盖约束压缩到 150,000 GS，再从该 checkpoint 独立构建并发布 layout。29 个动态帧平均为 **33.860999 dB / 51.814724 KB**，包含 frame0 的 30 帧平均为 **33.864572 dB**。KB 均为 1000 bytes，动态存储采用 motion、birth、opacity、区域 DC 和逐帧 manifest 的完整更新口径。

相对旧的合规 exact15k/SH1/150k 结果，本轮同时提高 **0.420996 dB**、减少 **7.477793 KB/帧**，证明严格 15k 条件下 SH3 路径仍有明确价值。但本地 `sota.png` 中 S²GS-full 的 Cut Roasted Beef 为 34.12 dB，本轮动态均值仍低 **0.259001 dB**，完整 30 帧均值低 **0.255428 dB**。因此本轮不通过 full300 扩展门槛，不能进入正式 SOTA 表，也不能仅凭相对 SH1 的改善归因于自适应空间层级。

### 核心想法与固定协议

旧 SH3 结果使用了 10k 中间 checkpoint，虽然画质较好，但不满足当前统一首帧协议。本轮只使用旧严格 15k 母版：

- 母 checkpoint：`output/goal_n3dv_frame0_20261030/cut_beef_f0_sh3_goal_20261030/20261030T012803_fab7ec/ckpt-0-15000.pth`。
- 母 checkpoint SHA256：`197cfc9dbf5c67ef7a9713a2c65a077d96e8c07624c1ddc44067ddbc842c65f2`。
- `global_step=15000`，`current_frame=0`，SH3，296,481 GS；opacity 0.0075 后为 275,155 GS。
- 标准 300 帧软链接数据，动态训练只跑前 30 帧；`resolution: -1`，1343×1006，cam00 留出评估。
- 初始 GS 必须在 100k–200k。本轮固定 150k，不使用 10k checkpoint 作为最终首帧，也不修改历史输出。

压缩以 19 个训练相机的 Rademacher VJP 贡献度排序，并强制每个已占用最细候选区域至少保留两个可用 GS。母 layout 的 4050 个占用叶区全部保留，覆盖不足区域为 0；为满足覆盖约束替换了 87 个纯全局排名行。压缩只选择原始属性行，不优化或改写保留属性。

### 配置与工程改动

- 覆盖母 layout：`config/cut_roasted_beef_sh3_15k_mother_layout_20261031.yaml`。
- 压缩后独立 production layout：`config/cut_roasted_beef_sh3_15k_cov150_layout_20261031.yaml`。
- 生产 30 帧配置：`config/cut_roasted_beef_sh3_15k_cov150_gatepricefix_regdc_30_20261031.yaml`。
- 生产配置相对当前 Cut 统一 SH1 版本的结构化差异只有 `output_dir`、`experiment_name`、checkpoint、reference layout 和 `module.sh_degree: 1 -> 3`。动态机制保持 50,000 bytes 目标、初始 multiplier 3、gate price 1/3、区域共享 DC，且 log-scale 更新关闭。
- frame-plan newborn 先由训练 RGB 初始化 DC，高阶 `sh_rest` 从同一个 codec reference 点完整继承；SH3 下为 15 个高阶系数，不会清零后发送整组抵消残差。
- 当时的 `scripts/publish_initial_asset.py` 严格校验完整激活的 SH0–SH3、layout degree、reference SH 形状与 finite 属性；15k、frame0、100k–200k、同源 fresh layout 约束未放宽。该脚本现已清理。
- `pipeline/model/gpu_stream.py` 从 `sh_rest` 系数数目严格推导 SH degree，`StreamView` 报告真实 degree；新增 SH3 newborn + regional DC 的 CPU/GPU exact 覆盖。

这些工程改动验证的是 SH3 资产与播放链路的正确性，不等于已经证明某种共享粒度选择策略提高了画质。

### 首帧资产

| 项目 | 数值 |
|---|---:|
| 严格 15k 母版 raw cam00 | 34.063446 dB |
| opacity 后 cam00 | 34.062527 dB |
| 150k 压缩后 cam00 | **33.968185 dB** |
| 150k 压缩后 19 个训练相机均值 | 38.196616 dB |
| 压缩损失（相对 opacity 后） | -0.094341 dB |
| 压缩后 SSIM | 0.961740 |
| 初始 GS / SH-rest 形状 | 150,000 / `[150000, 15, 3]` |
| production reference | 31,747,909 bytes |

压缩 checkpoint 为 `/mnt/SSD500GB/my4dgs_assets/cut_beef_sh3_15k_cov150_20261031/ckpt-0-15000.pth`，SHA256 为 `d516f0da51d27be5c145e8ad0d4f21c10ee3546e1f4867845065d697a25da8a2`。所有 xyz、DC、SH-rest、scale、rotation、opacity 均 finite，12 个 Adam 状态行张量均为 150k。

稳定资产位于 `/mnt/SSD500GB/my4dgs_assets/cut_beef_sh3_15k_cov150_asset_20261031`，由发布脚本原子生成；checkpoint、reference 和 layout config 均为软链接。独立 layout run 为 `/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cut_beef_sh3_15k_cov150_layout_20261031/20261031T024735_a35575`。

### 30 帧结果

运行目录：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cut_beef_sh3_15k_cov150_gatepricefix_regdc_30_20261031/20261031T024838_b1d3e4`。

| 指标 | 结果 |
|---|---:|
| frame1–29 PSNR | **33.860999 dB** |
| frame1–29 SSIM / LPIPS | 0.963262 / 0.115710 |
| frame0–29 PSNR / SSIM | 33.864572 dB / 0.963212 |
| 完整动态更新均值 | **51.814724 KB/帧** |
| 完整动态更新 P95 / 最大 | 56.7006 / 85.043 KB |
| 不超过 50 / 55 KB 的帧数 | 8/29 / 27/29 |
| 动态更新总量 | 1,502,627 bytes |
| reference + 完整动态更新 | 33,250,536 bytes |
| 最终 GS | 146,925 |
| 平均训练 / wall 时间 | 52.370 / 52.638 秒/帧 |

完整更新的平均组成如下：

| 组成 | KB/帧 |
|---|---:|
| motion | 43.414138 |
| birth | 1.729897 |
| opacity | 5.076345 |
| 区域 DC | 0.388793 |
| manifest | 1.205552 |

分段结果说明首帧树初始化不是唯一问题；后两段画质仍明显低于前段：

| 动态帧 | PSNR / dB | 完整更新 / KB |
|---|---:|---:|
| 1–10 | 34.071895 | 53.0255 |
| 11–19 | 33.736170 | 50.7941 |
| 20–29 | 33.762450 | 51.5225 |

最低为 frame29 的 33.519203 dB，最高为 frame5 的 34.136543 dB。rate multiplier 从 3.0 增长到 frame29 使用的 3.6950，下一帧状态为 3.6964；码率反馈已基本稳定，但完整均值仍比 50 KB 目标高约 3.63%。

### 与已有结果的关系

| frame1–29 | 旧合规 exact15k SH1 | 本轮 exact15k SH3 | 旧非法 10k SH3，仅机制参考 |
|---|---:|---:|---:|
| PSNR / dB | 33.440003 | **33.860999** | 34.131575 |
| 完整更新 / KB | 59.292517 | **51.814724** | 54.178103 |

相对合规 SH1，本轮是画质与存储双胜。相对旧 10k SH3，本轮低 0.270575 dB，但少 2.363379 KB/帧。旧 10k 结果违反统一首帧协议，只能用于理解机制，不能放入正式统一结果表。

一个重要现象是：旧 10k 压缩首帧约 34.2458 dB，本轮严格 15k 压缩首帧为 33.9682 dB，差约 0.278 dB；两次动态均值相差约 0.271 dB。两次运行从各自 frame0 到动态均值的下降也接近约 0.11 dB。这支持“当前主要画质缺口来自严格 15k 首帧上限，而不是动态表示继续累积额外 0.27 dB 损失”的判断，但两个运行的码率 warm-start 等设置并非完全相同，因此这仍是证据支持的推断，不是因果证明。

已有首帧记录还显示，同一条连续 SH3 训练在 10k、12.5k、15k 的 cam00 分别约为 34.3280、34.2355、34.0634 dB。最终必须使用 15k，不能回退或挑选 10k；这个趋势只用于定位严格 15k 后段训练的留出视角退化。

### 完整性与速度验证

- 发布与首帧相关 CPU 测试通过；完整 `tests.test_gpu_stream` 为 13/13 通过。
- 独立 CPU 顺序解码覆盖 frame0–29，逐帧检查 manifest、raw/state hash、保存质量和实际 cam00 重渲染。
- 修复后的 GPUStream 对本次真实 30 帧逐帧与独立 CPU 解码比较；所有 raw 属性、状态、激活后的 renderer 输入和像素 bit-exact。最终 state SHA 为 `6f1433ad0c06333e02938a4d9b7940b4fa28df3f058dac62896fd316b7ba65f1`，raw SHA 为 `e5b4c9586f6337c616c643a63034f7ace5528cb522794c49f036d8ec792a11d2`。
- frame29 checkpoint 与连续 stream 解码的 xyz、rotation、logscale、opacity、SH-DC、SH-rest 六属性均 bit-exact；重渲染像素最大差为 0。
- 已解码后的 30 帧完整渲染为 **393.664 FPS**，动态帧为 394.100 FPS，满足 ≥200 FPS 要求。
- 包含本地读包、校验、CPU 解码、更新、上传和实际渲染的前缀完整播放为 **10.604 FPS**。它与已解码渲染口径分列，不能用 393.664 FPS 替代。
- 验证产物为运行目录下的 `verification_sequence_20261031.json`、`verification_frame29_playback_20261031.json`、`verification_frame29_checkpoint_stream_20261031.json`、`verification_gpu_stream_prefix30_20261031.json` 和 `analysis_summary_20261031.json`。

### 原因分析与下一步

本记录验证的是：这份合规 SH3 资产在当时设置下明显优于所比较的旧 SH1 资产；它没有覆盖本次新 SH1 的训练配方，不能推广为 SH1 的能力上限。动态 codec 没有出现 SH3 newborn 高阶清零、degree 误报或 CPU/GPU 不一致；本轮 150k 严格首帧仍不足以把前 30 帧推到 SOTA，完整更新也略高于 50 KB。

最有希望的下一步不是继续调动态参数，而是修复 **连续训练到 exact15k 时的首帧后期退化**。下一生产首帧从头训练：前 10k 正常训练全部参数，后 5k 仅训练完整 SH3 颜色；最终只接受 global step 15000 checkpoint，geometry、scale、rotation 和 opacity 在后 5k 必须 bit-exact，不得把旧 10k checkpoint当最终结果，也不得用中间 checkpoint 选择最终模型。随后使用同一覆盖+贡献规则压到 150k，只有压缩后质量过门槛才允许再跑动态 30 帧。

这一路径针对现有证据中最大的质量缺口。是否确实能缓解后期留出视角退化，仍需新 exact15k 结果验证。当前没有证据支持直接扩 full300，也没有证据支持把总指标变化归因于自适应层级本身。

---

## Cut Roasted Beef：support-safe exact15k SH3 / attr-refresh-4 + birth-decoupled first30

### 结论

本轮完成 frame0–29，共 30 帧。29 个动态帧的外部 cam00 平均为 **34.265188 dB / 48.778069 KB**；包含 frame0 的 30 帧平均为 **34.271638 dB**。这里 KB 固定为 1000 bytes，存储采用 motion、birth、old opacity、区域 DC 与逐帧 manifest 的完整更新口径。

本候选通过预注册硬门：相对 Cut 的 34.12 dB 目标高 **0.145188 dB**，完整更新比 50,000 bytes 上限低 **1,221.931 bytes/帧**；也比 34.173 dB 的生产扩展优选线高 **0.092188 dB**。相对同一首帧资产、旧统一动态机制的 first30，本轮提高 **0.042118 dB**，同时减少 **0.355586 KB/帧**。已解码全序列渲染为 **274.382 FPS**，满足 ≥200 FPS；包含本地读包、校验、CPU 解码、更新、上传和实际渲染的 frame1–29 完整播放为 **11.567 FPS**，两种速度口径分列。

这是 Cut 当前最好的合规 first30 生产候选，但仍不能据此宣称 full300 超过 SOTA。上一版机制曾在 first30 达标后出现明显长时漂移，因此本轮只完成 30 帧验收，没有启动 full300。

### 核心机制与协议

本轮使用固定四帧属性刷新周期，并把“旧属性是否刷新”和“newborn 是否提交”解耦：

- frame4、8、12、16、20、24、28 是预先声明的刷新帧，只允许 B/C；旧 opacity 与区域 DC 必须提交，B 不提交 newborn，C 提交 newborn。
- 其他动态帧只允许 A/D；旧 opacity、区域 DC、log-scale 均不提交，A 不提交 newborn，D 提交 newborn。
- 四个固定 internal probes 为 cam05、cam10、cam15、cam20。它们仍对称参与标准训练，只在尾部决定 A/D 或 B/C；cam00 始终是外部测试相机，不参与训练或选择。
- log-scale 更新全程关闭。动态码率仍使用 50,000 bytes 目标、initial multiplier 3、gate price 1/3 和同一 support-safe exact15k SH3/cov150 首帧资产。

实际选择统计为 A=5、B=3、C=4、D=17。A/B 的 birth 总数严格为 0；C/D 分别提交 249/1,021 个 newborn，共 1,270 个。七个刷新帧均只产生 B/C，22 个非刷新帧均只产生 A/D；每帧 streak 严格按 0→1→2→3→0 循环。刷新帧都有 old-opacity 与区域-DC packet，非刷新帧二者均为空，log-scale packet 始终为空。

19 个标准训练相机都实际出现在 birth proposal 记录中；四个 probes 分别出现 12、11、9、13 次，cam00 出现 0 次。外部 cam00 只在选择冻结后用于 posthoc 诊断：internal probes 选择的分支在 cam00 上有 18 帧更好、11 帧更差，选择分支相对另一候选的 cam00 差值均值仅 +0.002929 dB。这既表明 cam00 没有泄漏进选择，也说明 internal probes 与外部视角的偏好并不完全相同。

### 配置与来源

- 原训练配置已退役，原文件 SHA256：`790a7341911d0062aff31af0b64f1e8bedda7a2af0d139809976914c850383c1`；运行时冻结配置保存在下方运行目录的 `config.yaml`，仅用于记录和回放。
- 运行目录：`/mnt/SSD500GB/my4dgs_output/goal_n3dv_dynamic_20261031/cut_beef_sh3_supportsafe15k_cov150_attrrefresh4_birthdecoupled_nocam04_30_20261031/20261031T155033_05d524`。
- 首帧 checkpoint SHA256：`6672ad7ef716d4381d22f82f2f971afa9bb223925a045169889492bac95bd7f1`。
- reference layout SHA256：`89ea2e2aa16afe1a6ade73f4a5844844902fba45247db65f1f0bf82ce0d00888`。
- 首帧严格为 global_step=15000、SH3、resolution -1、exact 150,000 GS；数据仍通过 standard300 软链接读取，没有复制数据或改动链接目标。

最初的配置误把 Cut 数据中不存在的 `cam04.png` 放入 fitting 列表，初始化检查在任何动态训练前即报错退出；失败初始化目录被原样保留。本次正式配置只使用新的 experiment name，并移除不存在的 cam04，使 15 个 fitting 相机与 4 个 probes 无重叠地覆盖全部 19 个标准训练相机，cam00 保持外部。

本实验没有修改共享源码。运行时冻结的关键源码为：

- `pipeline/frame_plan.py`：`99530305baf7a7b081f405728a8ba974fe5262d9749d0f78a155337579bf014c`；
- `pipeline/module.py`：`ee0c18d1fa7b98c99e94cabf78bccc972cd45b8204cccd8ab205dd9088bb4eac`。

### 质量、存储与训练结果

动态帧汇总：

| 指标 | 结果 |
| --- | ---: |
| PSNR | 34.265187757 dB |
| SSIM | 0.962770647 |
| LPIPS | 0.116173798 |
| 完整动态更新均值 | 48,778.069 bytes / 48.778069 KB |
| payload 均值 | 46.380379 KB |
| 完整动态更新总量 | 1,414,564 bytes |
| 完整更新 p95 / 最大值 | 56.691 KB / 58.837 KB |
| reference | 31,221,183 bytes |
| reference + 29 个完整更新 | 32,635,747 bytes |
| 最终 GS | 144,728 |
| 平均训练 / wall 时间 | 59.059 s / 59.343 s 每帧 |

最低 PSNR 为 frame15 的 33.957394 dB，最高为 frame2 的 34.507462 dB；最大完整更新为 frame24 的 58.837 KB。总计 retirement 6,542 GS、birth 1,270 GS，因此 150,000 - 6,542 + 1,270 = 144,728 GS，与最终 checkpoint 完全一致。除 frame1 的初始 809 个清理外，后续 retirement 均发生在预声明刷新帧。

分段结果：

| 动态帧 | PSNR | 完整更新 KB/帧 | final 相对 step200 stage PSNR |
| --- | ---: | ---: | ---: |
| 1–10 | 34.415260 | 43.704300 | +0.004662 dB |
| 11–20 | 34.164872 | 50.602400 | +0.011221 dB |
| 21–29 | 34.209903 | 52.388556 | +0.004489 dB |

完整更新平均组成：motion 42.598138 KB、birth 2.019828 KB、opacity 1.647276 KB、区域 DC 0.115138 KB、manifest 2.397690 KB，log-scale 为 0。刷新帧平均为 34.251510 dB / 53.826571 KB；非刷新帧平均为 34.269540 dB / 47.171727 KB。step200 transaction baseline 到最终状态的外部 cam00 平均变化为 +0.006870 dB，但单帧既有收益也有损失，不能据此声称每次刷新都改善外部视角。

选择类别仅用于验证实现语义，不应作为类别间因果比较：

| 类别 | 帧数 | PSNR | 完整 KB/帧 | birth 总数 |
| --- | ---: | ---: | ---: | ---: |
| A | 5 | 34.234541 | 44.978200 | 0 |
| B | 3 | 34.254325 | 51.752000 | 0 |
| C | 4 | 34.249399 | 55.382500 | 249 |
| D | 17 | 34.279833 | 47.816882 | 1,021 |

### 与旧 Cut first30 比较

| 候选 | 动态 PSNR | 完整 KB/帧 | 已解码渲染 FPS | 完整播放 FPS |
| --- | ---: | ---: | ---: | ---: |
| 旧 support-safe / 每帧旧属性更新 | 34.223070 | 49.133655 | 279.370 | 10.780 |
| 本轮 refresh-4 + birth-decoupled | **34.265188** | **48.778069** | 274.382 | **11.567** |

本轮相对旧候选提高 0.042118 dB，完整更新减少 0.355586 KB/帧。属性包节省是主要存储收益：opacity 平均从 5.108207 KB 降到 1.647276 KB，区域 DC 从 0.387621 KB 降到 0.115138 KB；但 motion 增加约 2.301379 KB，且更完整的选择/恢复元数据使 manifest 增加约 1.149966 KB，因此净节省较小。这个结果与“稀疏刷新旧属性可以把预算留给运动和 newborn，同时保持画质”的生产假设一致，但 first30 总指标不能单独证明该机制是唯一原因，也不能替代 full300 验证。

分段上，本轮相对旧候选分别变化：1–10 为 +0.007267 dB / -0.928900 KB，11–20 为 +0.082043 dB / -0.212000 KB，21–29 为 +0.036480 dB / +0.121889 KB。末段完整更新已经略高于旧候选，说明码率优势并非每段都成立；full300 若获授权，仍需继续按分段门控。

### 严格回放、checkpoint 与速度验收

- 29 份 summary 全部 finite；实际 motion/birth/opacity/DC 文件大小、manifest 大小与 `packet_bytes`、`manifest_bytes`、`complete_dynamic_update_bytes` 逐帧严格闭合。
- CPU 从共享 reference 连续回放到 frame29 成功，最终 `decoder_sha256=9d1d99f5f364b75a0e695f2e847254530900eae4e2c1ad70da0d248091624a4e`，raw SHA256 为 `72068c975d2ce334fc623988889cd27b753908521de7d75fe716751c93651b65`。
- frame29 checkpoint SHA256 为 `806a0e2d3188a7f3e5d066ee5d4527cb81620d2aec8821d795a18bd76a4aa731`；状态为 `current_frame=29`、`global_step=300`、`prepared/finalized/motion_committed=true`、SH3、144,728 GS。
- checkpoint 与连续 stream 的 xyz、rotation、logscale、opacity、SH-DC、SH-rest 六属性全部 bit-exact且 finite；checkpoint/stream 重渲染像素最大差为 0，保存的 PSNR/SSIM/L1 差均为 0。
- frame29 保存的 `tail_motion_only_streak=1`、rate controller `frames=29`。使用真实 `Module.on_load_checkpoint` 路径恢复成功，streak、finalized/motion 状态、rate 状态和 decoder digest 全部保持。
- 真实 GPUStream 对 frame0–29 逐帧检查通过：CPU/GPU raw 六属性、解码状态、激活后的 renderer 输入、实际 RGB 像素和保存 PSNR 全部一致；最终 state/raw SHA 与 CPU 回放相同。
- `tests.test_gpu_stream` 13/13 通过。
- 全30帧已解码渲染为 **274.382 FPS**，动态帧为 **274.717 FPS**；frame29 单独物化渲染为 **281.053 FPS**。
- frame1–29 完整播放为 **11.567 FPS**。该口径包含本地文件读取、校验、CPU 解码、更新、上传和实际渲染；初始 reference、相机/GT 加载、指标计算与预检不计入。
- 所有速度测试前后均由 NVML 确认无其他 compute context；验收结束后 GPU 为 0%/400 MiB 且无 compute process。

验证产物位于运行目录：

- `verification_sequence_attrrefresh4_20261031.json`，SHA256 `762b450b535d686ad687d0dbbca417ed809bcbaeb358fd338e24abbaf869e430`；
- `verification_frame29_checkpoint_stream_attrrefresh4_20261031.json`，SHA256 `eead53541d4cace725d0f3087b695440925e4fcd3623c00c29ceade3f9ca1fc7`。

### 下一步

本候选满足 first30 的画质、完整存储、渲染速度和恢复完整性门槛，具备成为唯一 Cut full300 延伸对象的条件。但上一轮 full300 在 frame30 后持续退化并于 frame119 硬停止，所以不能仅凭本轮 first30 的 +0.042 dB 推断长序列已解决。若后续由主控授权延伸，应从本次 finalized frame29 checkpoint 与完整 prefix 续跑，并继续在固定里程碑检查累计/最近30帧 PSNR、完整字节、属性刷新协议、GS retirement、streak 恢复和像素级回放；本轮不自行启动 full300。
