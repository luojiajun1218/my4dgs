# 稀疏旧opacity：30帧完成，质量仍未达标

## Idea与本轮范围

保持首5b机制：100自由GS→区域维护→100共享/GS联合优化→100birth，同时在最后100步允许旧opacity logit双向稀疏量化更新。step=.01、lr=.05、复用运动每字节价格；birth先预测更新前外观，再仅对旧行应用opacity包。没有叠改运动、区域/tree维护、量化或预算，也没有启用warm start。

本轮只扩大到30帧并调整checkpoint帧；源码冻结，source_snapshot保留34文件及版本/hash。首5启动时stage loader故障及修复见20261014_sparse_old_opacity_first5b.md，本轮无运行错误。

- 配置：config/fs_persistent_sparseopacity_first30_20261014.yaml
- Run：output/fs_persistent_sparseopacity_first30_20261014/20261014T024217_30159a
- 命令：激活ReConGS后 python main.py --config=config/fs_persistent_sparseopacity_first30_20261014.yaml
- 完成frame0–29，实际exit0，wall1157.73s。

## 全30帧独立解码与验收范围

| 指标 | 实测 | 本阶段判断 |
|---|---:|---|
| 全frame0–29平均test PSNR | **34.15377 dB** | 未达到34.2 |
| 动态frame1–29 PSNR | 34.15868 dB | 未达到34.2 |
| 动态完整更新平均 | **49226.83 B** | 本30范围≤50000 |
| motion /birth /opacity /manifest平均 | 43064.79 /4415.21 /1271.14 /475.69 B | 所有实文件计费 |
| 全0–29纯完整渲染 | **201.833 FPS** | 本30范围>200 |
| 动态纯渲染 | 201.778 FPS | 排除文件读取/解码/上传 |
| frame29物化渲染 | 200.567 FPS | 5预热+30次×3 |
| 2轮顺序1–29读取/校验解码/上传/渲染 | 7.545 FPS | 不混称纯FPS |

全帧逐包重放、motion/birth/opacity最终raw hash、saved PSNR与cache开关图像均验证一致；sequence每帧10预热+50次CUDA同步完整forward/clamp。完整动态合计1,427,578 B，payload1,413,783 B；共享reference16,656,569 B另计一次。100/300未运行，不能把此30帧的字节/FPS通过当成全序列达标，当前明确未达项仍是质量。

## 同一heldout cam00阶段轨迹

| 段 | step100建区后PSNR | step200 joint后，仅旧GS | step300 birth+opacity后 | 100→200 | 200→300 | 完整B/帧 | opacity B/帧 |
|---|---:|---:|---:|---:|---:|---:|---:|
|1–10|34.28621|34.30064|34.29904|+0.01443|-0.00161|57573.8|1426.7|
|11–20|34.25004|34.26227|34.27936|+0.01223|+0.01710|40633.1|1120.1|
|21–29|**33.82636**|33.84974|33.86865|+0.02338|+0.01891|49501.0|1266.1|

后段质量下降主要已经出现在step100，两个后续阶段平均只微修复。birth+opacity每帧方向混合，不能把阶段差全部归为opacity单项。最后frame29为33.89226→33.81279→33.77419；也不能只由单帧推断整个阶段机制。

最终opacity正/负支持累计5844/5591次（跨帧可能重复GS），包平均1271.14 B，占完整更新约2.58%；最后帧380行（+192/-188）、1243 B，旧GS212651。支持稀疏、双向更新和解码正确，但仍未证明解决长时质量退化。累计birth6708，总GS206237→212945；birth持续发生，并非后段停止增点。最后55个区域slot，内部parent实际0；本轮30帧尚未触及64leaf容量。

## 剩余机制证据与限制

29checkpoint按真实ID记录显示，25→29阈值移动GS的解码位移p90从0.01447→0.01899世界单位、p99从0.02256→0.02669，最后最大0.07241；与此同时后段step100质量偏低，joint平均只提升0.02338 dB。这支持检查运动跟踪/分组不足，但历史位移是模型估计，不是GT轨迹；不能仅凭zero-start或lr=.0001断言因果。

region1有1228/21814个非零独立GS，其中685个六通道整数绝对平均>4；整体p90=0、活跃子集p90=9.67、split_streak=0，少数高残差继续被大区域静止多数稀释。region2/3整体p90=5.67/7.33、streak6/11，活跃GS最终独立平移p90约0.01585/0.01565世界单位。下一机制选择局部持久区域维护，以高残差子集和安全回收改善分配，当前不同时调整opacity/预算/量化。

“有效观测”在现实现中累计renderer visibility_filter/radii>0，是投影/视锥可见代理，未必代表遮挡后像素贡献；不能据它断言被遮挡GS具有真实运动证据。固定heldout阶段轨迹也不等同不同训练视角的loss轨迹，所以没有凭训练/测试数字差宣称过拟合。

## 新产物

- 全30重放/质量/纯FPS/完整字节：output/persistent_sparseopacity_first30_benchmarks_20261014/sequence.json
- frame29顺序播放：output/persistent_sparseopacity_first30_benchmarks_20261014/frame29_playback.json
- 整数残差/实际包计费：output/persistent_sparseopacity_first30_analysis_20261014/packet_analysis.json
- 解码历史位移及区域残差：output/persistent_sparseopacity_first30_analysis_20261014/decoded_motion_tracking.json
