# 四种在线平面共享运动方法：前 5 帧完成汇总

## 结论与统计范围

四种方法都完成了 frame0–4 的运行及一致速度测量，但 **16 次候选划分全部因为完整压缩包不更小而拒绝**；不是量化精度不合格。最后都只保留身份 root，实际共享平移/旋转为零，运动由独立 GS 残差承担。因此这轮结果说明当前候选生成没有兑现共享存储收益，不能从相近的质量指标选出“最佳共享算法”。

主表只平均动态帧 **1–4**，共享 frame0 不参与该 PSNR 平均。所有数据都是短序列结果，尚不能证明全 300 帧平均 PSNR>34.2 dB、50 kB/帧或全序列 FPS>200。

| 方法 | test PSNR，帧1–4 dB | motion kB/帧 | birth kB/帧 | payload kB/帧 | 加 manifest kB/帧 | 帧耗时 s | 动态纯渲染 FPS | 动态顺序播放 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kmeans | 34.29735 | 79.613 | 4.078 | 83.691 | 84.127 | 16.87 | 201.924 | 4.189 |
| GMM | 34.28824 | 79.493 | 3.952 | 83.445 | 83.881 | 16.70 | 201.476 | 4.201 |
| HDBSCAN | 34.29959 | 79.698 | 4.090 | 83.788 | 84.224 | 35.85 | 201.509 | 4.261 |
| greedy | 34.29546 | 79.751 | 4.033 | 83.784 | 84.220 | 17.23 | 201.762 | 4.329 |

kB=1000 B；payload 为实际最终 motion+birth 文件，motion 包已包含帧头、归属、区域/layout 等 codec 元信息和出生位姿，birth 存出生属性。manifest 为每动态帧 436 B 的解码索引 JSON，单独计入右侧存储口径；训练 checkpoint、图像及诊断日志不计传输量。共享 reference **16,656,569 B = 16.656569 MB** 只计一次，不隐藏摊入 4 个动态帧预算。所有实际文件大小已与 summary 校验一致。帧耗时取 `elapsed_seconds`，包含训练循环及验证评估，不是纯优化时间；新 GMM/HDBSCAN/greedy 进程 wall 分别为 72.20 / 148.92 / 74.25 s。

重新解码测得共同 frame0 test PSNR=34.011318 dB，因此含共享首帧的 5 帧统计如下，不能与主表混用：

| 方法 | test PSNR，帧0–4 dB | 纯渲染 FPS，帧0–4 |
|---|---:|---:|
| Kmeans | 34.24015 | 201.889 |
| GMM | 34.23286 | 201.360 |
| HDBSCAN | 34.24193 | 201.689 |
| greedy | 34.23863 | 201.674 |

## 共同核心方法与代码

1. 每帧先训练自由独立 GS 运动，之后取当前拟合转移和之前已解码转移，历史最多 5 个；用稳定 GS ID 对齐，出生点不补造过去观测。
2. 分别用 Kmeans、GMM、HDBSCAN 或 greedy 提议平面区域，未知/近静止/noise 不强制归入运动组。统计方法主要使用位移历史，协方差旋转默认不作为物体旋转证据；拟合共享运动采用位置刚体拟合。
3. 旧划分与新划分拟合同一当前位姿；使用共同上一帧解码状态、同一量化误差限制及真实 codec 编码/解码。比较完整压缩包，包含 layout、归属、共享与独立残差，不只比较理想残差。
4. frame1 开始每帧提案，接受更小的候选；本轮没有层级树、没有使用未来帧，没有改写共享 frame0。

主要实现：`online_motion_clustering.py`、`online_flat_motion.py`、`online_motion_diagnostics.py`，以及 trainer/module 的因果初始化、存储与日志。此次三项新增运行只改配置中的 method，未针对结果修改算法。之前 `np.bool_` 日志序列化错误已在评分来源转为原生 bool；11 项 numerical 测试中包括真实四方法提案→门控→JSON 写入、接受/不省字节/量化失败路径及解码回退检查。

## 候选区域与实际采用必须区分

| 方法 | 候选运动区域数，帧1/2/3/4 | 采用次数 | 实际非 root 区域 | 每帧拒绝原因 |
|---|---|---:|---:|---|
| Kmeans | 8 / 9 / 11 / 15 | 0/4 | 0 | complete_packet_not_smaller |
| GMM | 8 / 8 / 7 / 14 | 0/4 | 0 | complete_packet_not_smaller |
| HDBSCAN | 1942 / 1526 / 1464 / 1456 | 0/4 | 0 | complete_packet_not_smaller |
| greedy | 19 / 39 / 41 / 42 | 0/4 | 0 | complete_packet_not_smaller |

所有候选量化有效，最终都只有区域 0、全部 GS 占比 100%、实际归属变化率 0%；共享平移 `[0,0,0]`，旋转 wxyz=`[1,0,0,0]`，0°。最终 GS 约 20.64–20.71 万。每帧具体 GS 数、unknown/noise/static 比例和独立残差 mean/p50/p90/max 已记录在各方法文档及 `frame_summary`；残差仅统计有观测存活点，比例以所有当前 GS 为分母。

四 run 的共享 reference 文件 SHA256 完全相同：`21f338ba72de7a8531f25bcc454ebeffcb9074dcecd24d71fb720f2c11e77959`。但任意两方法的四个最终 motion 包逐字节比较均不同，所有逐帧 raw-state SHA256 也不同，不能称最终码流完全相同。均值 PSNR 最大差仅约 0.01134 dB；结合全部 root 回退，这些小差异不足以评价共享算法优劣，也没有重复运行估计方差。

## 机制没有兑现收益的证据

首个动态帧最容易看清问题：

| 方法 | 旧/候选完整压缩包 B | 候选 raw 归属 B | raw 独立残差减少 B |
|---|---:|---:|---:|
| Kmeans | 138107 / 179207 | 161300 | 4422 |
| GMM | 138063 / 176775 | 157943 | 4892 |
| HDBSCAN | 138119 / 184227 | 142588 | 203656 |
| greedy | 138127 / 172755 | 124854 | 5978 |

门控包还没有最终出生阶段数据，不可代替主表实际 payload。raw 分项不能直接当作压缩后的节省量。Kmeans、GMM、greedy 的首帧残差收益不足以支付大量成员归属；后续 raw 残差收益常为负，说明给历史上曾动、当前已零残差的点施加共享运动会引入抵消项。greedy 的拟合误差容差 0.0004×extent 约为 0.002597 世界单位，相当于 4 个 GS 平移量化步长；几何拟合通过仍然可能要存多个残差符号，几何相容不等于码流收益。

HDBSCAN 首帧 raw 总包 **411068→391966 B**，但实际压缩包 **138119→184227 B**。它生成 1942 个小区域，额外 layout/shared 成本分别为 25260/16708 raw B，改变压缩结构后，raw 节省没有兑现。压缩熵及字段相互作用是可能解释，本次没有逐分项压缩归因。随后 raw 残差收益仅 8116/167/-489 B，而归属仍超过 119 kB。这也验证了不能放弃完整 codec 门控。

一个值得优先检验的改进方向是：用当前量化后实际存储收益选择成员、排除当前无需改变的点，允许只尝试接纳部分有收益的区域，再走同一实际 codec 门控。完整划分一次性接受会一起支付所有成员归属，不能发现可能存在的有益子集；本实验没有证明这样的子集一定存在。第二优先级是观测模式的拟合预算分配和避免过多小区域。此处只提出原因及后续方向，没有修改算法或自动追加实验。

## FPS 测量协议和限制

所有训练结束后，四 run 顺序在 **RTX 3090 Ti** 上测量，1600×1199 cam00，冻结 GS 属性缓存开启；GPU 排他检查通过，缓存/非缓存图像逐像素一致，解码重放动态 PSNR 与训练保存值逐帧完全一致。

- 纯渲染 sequence：每帧 10 次预热、50 次完整 forward+clamp，CUDA 前后同步；聚合 FPS=总渲染次数/总渲染秒数。约 201.5–201.9 动态 FPS 不含文件读取、CPU 解码、上传、数据加载或质量评估，最低动态单帧为 200.53–200.84 FPS。这里只略超过 200，且只测早期 5 帧，不代表长序列稳定达标。
- 顺序播放：从共享参考依次读取动态帧 1–4，共 2 轮，包含 motion/birth 本地读取、校验 CPU 解码、GS 上传和完整渲染，平均 **4.19–4.33 FPS**。不包含一次性 reference、相机/GT 加载、manifest 预检查、质量评估，文件预读可能受 OS 缓存帮助，不能称完整应用启动速度。
- 顺序播放平均读取/解码耗时 222.92–230.70 ms/帧，上传+渲染约 7.93–8.10 ms，解码占约 96.5–96.6%。当前端到端瓶颈在 CPU 解码路径；约 202 纯渲染 FPS 不能称作 codec 播放 FPS。

基准结果均写入新的 `output/online_flat_first5_benchmarks_20261013/`，源实验目录只读，八项命令全部 exit=0。每个方法对应 `<method>_sequence.json` 和 `<method>_frame4_playback.json`；精确命令、时间和源路径在 `benchmark_commands.json`。

复现协议（将 `<run>` 换成下列完成目录，输出路径须为新文件）：

```bash
conda activate ReConGS
python scripts/benchmark_rigid_residual_tree.py --run <run> --mode stream --sequence --sequence-warmups 10 --sequence-renders 50 --output <new_sequence.json>
python scripts/benchmark_rigid_residual_tree.py --run <run> --mode stream --frame 4 --warmup-renders 5 --timed-renders 30 --rounds 3 --playback-rounds 2 --output <new_playback.json>
```

## 完成目录和逐项记录

| 方法 | 配置 | 完成 run | 实验记录 |
|---|---|---|---|
| Kmeans | `config/fs_online_flat_kmeans_first5_20261013.yaml` | `output/fs_online_flat_kmeans_first5_20261013/20261013T182601_401a25` | [Kmeans](20261013_online_flat_kmeans_first5.md) |
| GMM | `config/fs_online_flat_gmm_first5_20261013.yaml` | `output/fs_online_flat_gmm_first5_20261013/20261013T182944_18954d` | [GMM](20261013_online_flat_gmm_first5.md) |
| HDBSCAN | `config/fs_online_flat_hdbscan_first5_20261013.yaml` | `output/fs_online_flat_hdbscan_first5_20261013/20261013T183116_80f067` | [HDBSCAN](20261013_online_flat_hdbscan_first5.md) |
| greedy | `config/fs_online_flat_greedy_first5_20261013.yaml` | `output/fs_online_flat_greedy_first5_20261013/20261013T183412_05a5e0` | [greedy](20261013_online_flat_greedy_first5.md) |

训练复现命令为激活 ReConGS 后运行 `python main.py --config=<对应配置>`。此次只完成用户指定的短序列四方法运行和上述速度测量，没有启动全 300 帧或额外消融实验。
