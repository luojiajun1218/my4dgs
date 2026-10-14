# 持久运动区域：首 5 帧完成记录

## 核心想法和实现

只用 frame1 Kmeans 初始化运动区域，后续继承稳定 logical ID，不每帧重聚类。存活 GS 以稳定 GS ID 对齐历史，新生点不补造过去观测；连续两个有效观测中候选量化残差代价改善至少25%才迁移，缺观测保留待确认状态。nonroot 持续高残差区域在结构间隔10帧检查 split/merge，连续三有效帧 idle 可归root并安全压缩slot，不复用logical ID。本轮只运行**持久平面区域**，树尚未接入实际训练。

新增 persistent_motion_regions.py 的状态、证据和局部编辑；persistent_motion_diagnostics.py 记录 parent/logical ID、存活/新生、整数非零计数和组合运动。保留100步自由GS→分区/局部维护→100步联合细化→100步births，最终解码后更新历史。9项CPU回归覆盖实际200步调度、checkpoint精确续训、跨帧/新生/缺观测、slot映射、迁移确认、split/idle/merge与empty internal ancestry。

- 配置：config/fs_persistent_regions_first5_20261014.yaml
- 完成 run：output/fs_persistent_regions_first5_20261014/20261014T003210_f84810
- 命令：激活 ReConGS 后 python main.py --config=config/fs_persistent_regions_first5_20261014.yaml
- exit=0，wall145.58s；source_snapshot保存本次源码/配置与哈希。数据软链接、共享首帧，resolution=-1，种子0。

## 实际质量、完整更新字节和机制

只平均动态帧1–4，frame0另测；B为实际文件字节。**完整更新=motion+birth+必要manifest JSON**，不是旧encoded_bytes的payload口径。reference16,656,569 B只计一次，训练产物不计传输量。原codec逐包校验/decoded digest全部匹配manifest。

| 帧 | test PSNR dB | motion B | birth B | manifest B | 完整更新 B | 迁移GS | 分区后→细化后独立非零GS |
|---|---:|---:|---:|---:|---:|---:|---:|
|1|34.40453|109891|2413|436|112740|0|43767→12092|
|2|34.30782|73347|3548|436|77331|0|42372→18416|
|3|34.43412|52795|3826|436|57057|0|32248→11553|
|4|34.38398|59175|4736|436|64347|10766|25599→9632|
|平均|**34.38261**|**73802**|**3630.75**|**436**|**77868.75**|—|—|

全部四帧保持logical ID 0–8，9个非空slot，其中8个运动叶。frame2/3完全继承，raw归属与layout各只有1 B；frame4迁移10766点，raw归属增加到41955 B、layout126 B。其同目标旧划分包83555 B→迁移后82283 B，实际只节省1272 B；不能把量化幅值改善直接当作大幅字节收益。细化后无birth包分别106047/68163/47115/52743 B，最终更新还包括birth与索引。

目前动态PSNR超过34.2，但**完整更新77.87 kB/帧仍高于50000 B目标**。首4个转移不能证明完整300帧达标；最终PSNR在births后评估，没有阶段test PSNR，不能把所有提升单独归因于某个阶段。

## root 是否是剩余残差的主瓶颈

以下独立非零是最终int16平移/旋转符号并集，分母为全部非零存活GS；新生点无上一帧残差。raw字节份额按全局delta-row间隔、flags及6 B块逐条精确归属，稀疏count header另计，**不等同混合压缩包的区域份额**。

| 帧 | 全部非零GS | root非零GS | root非零份额 | root raw残差条目 B | 全部raw条目 B | root字节份额 |
|---|---:|---:|---:|---:|---:|---:|
|1|12092|1615|13.36%|13930|113648|12.26%|
|2|18416|4895|26.58%|41061|160139|25.64%|
|3|11553|2587|22.39%|22095|102254|21.61%|
|4|9632|1978|20.54%|17131|83977|20.40%|

四帧合计root仅占非零转移行21.42%、raw残差条目20.48%，**本轮没有证据支持root主导**。frame4 nonroot还有7654非零点，其中region2/3分别2111/1815，合计占全部非零40.76%，raw条目合计约41.10%；两区split_streak连续4，p90约5.67/5.15量化单位。结构间隔10使首5帧尚未实际split，后续更应先检验这些叶区域的异质运动与拆分，而非现在盲目改root策略或阈值。

## 独立解码渲染基准与后续

唯一sequence基准在训练完成后、源码短暂冻结时运行，RTX3090Ti、1600×1199 cam00、冻结GS属性缓存、每帧10预热+50次CUDA同步完整forward/clamp。动态纯渲染 **201.602 FPS**，含frame0 **201.424 FPS**；含实际frame0的5帧PSNR为 **34.30835 dB**。纯渲染排除解码、上传和评估，不能称端到端播放速度，也不是全300帧速度结论。

报告 output/persistent_regions_first5_benchmarks_20261014/sequence.json 的 dynamic_packets.complete_update_bytes_mean=77868.75 B；旧encoded_bytes_mean=77432.75 B明确仅payload。离线精确区域字节分析在 output/persistent_regions_first5_analysis_20261014/packet_analysis.json，可用 scripts/analyze_persistent_motion_run.py只读复核。

本轮完成了继承、确认迁移、真实码流和恢复路径验证；主存储压力仍在nonroot独立残差。下一步按已批准计划接树并在30帧检验结构机制，不做对照/消融，不把未运行的树效果写成已实现质量收益。
