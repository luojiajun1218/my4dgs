# 持久区域与树接入：前30帧完成记录

## Idea与实际实现范围

首帧Kmeans初始化，之后继承logical ID，仅凭连续有效观测局部迁移/拆分/合并/回收；内部树节点不直接拥有GS，父运动+子相对运动+GS残差按codec约定组合。新增 persistent_motion_tree_integration.py：每帧用当前观测拟合已有父/叶WORLD运动，转local并量化、GS retarget保位姿；frame1及每10帧尝试建立共同父，maxdepth3，量化幅值收益不足不强制建树。新增child-relative loss，root一级共享不受该项惩罚。

共同300步：100自由GS→局部结构/树候选→100共享+GS联合细化→100births。10项真实生命周期/综合回归及独立树admission/codec检查通过；运行中源码/参数冻结，source_snapshot保留当次版本。

- 配置：config/fs_persistent_tree_first30_20261014.yaml
- 完成run：output/fs_persistent_tree_first30_20261014/20261014T005425_5719e4
- 命令：激活ReConGS后 python main.py --config=config/fs_persistent_tree_first30_20261014.yaml
- 30帧含共享frame0，动态1–29，resolution=-1，种子0，训练exit=0。

## 验收结果：存储、纯渲染过线，质量未过线

完整sequence独立解码/重放每帧质量与saved一致，全部packet最终digest匹配manifest。RTX3090Ti、1600×1199 cam00、冻结属性缓存，每帧10预热+50次CUDA同步完整forward/clamp，源码在基准期间冻结。

| 指标 | 实际结果 | 本阶段判断 |
|---|---:|---|
| 全frame0–29平均test PSNR | **34.11953 dB** | 未达到34.2 |
| 动态frame1–29平均test PSNR | 34.12326 dB | 未达到34.2 |
| 动态完整更新平均 | **46988.86 B** | 此30帧范围达到50000 B |
| 动态payload motion+birth平均 | 46552.17 B | 不作为完整预算口径 |
| manifest索引平均 | 436.69 B | 已计入完整更新 |
| 全frame0–29纯渲染 | **201.629 FPS** | 此短范围>200，余量小 |
| 动态纯渲染 | 201.667 FPS | 排除解码/上传 |
| 2轮顺序frame1–29读取/校验解码/上传/渲染 | **7.661 FPS** | 不与纯渲染混称 |

29个完整更新合计1362677 B，共享reference16,656,569 B另计一次，不把checkpoint/图片/诊断作为传输量。全300帧尚未运行，不能宣布总目标完成。当前未达项是质量，不能继续把主问题说成平均字节超标。

## 结构机制是否真的运行

frame1评估28对tree候选，frame10评估6对并检查4个历史转移，frame20评估3对并检查4个历史转移；均 no_supported_quantized_gain。其他帧 tree_update_not_due。**本轮实际内部节点始终0，不能说树已产生压缩或质量收益**；初始化/结构入口正常，不是忘记调用，但日志未分解位置拟合、历史拟合和量化收益的拒绝数量。

frame10真实合并8→4、4→1、6→1、7→1，slots9→5。frame20真实split3→9、5→10、2→11，再merge11→10、5→1、2→1、3→1，最终logical IDs [0,1,9,10]、slots4。稳定ID和映射压缩均经实际码流复核。

region2/3在frame9 final的split_streak为9/8，p90为6.50/6.17量化单位；frame10 final为10/9、6.17/5.00。但维护发生在100步，final证据来自200步，现有日志无维护时current p90或split trial/rejectioncost，所以不能把未拆直接归因于阈值或cost；frame20的实际split证明该路径能执行。

## 后段质量下降与root残差

| 动态段 | 平均PSNR dB | 完整更新 B/帧 | root占非零残差GS | root占raw残差条目字节 |
|---|---:|---:|---:|---:|
|1–9|34.33114|58567.67|24.45%|23.64%|
|10–19|34.24395|38301.80|43.17%|42.48%|
|20–29|**33.81548**|45255.00|**61.30%**|**60.22%**|

21–29平均PSNR33.75750、完整更新45910.56 B、平均birth217.33个，birth没有停产；独立非零GS平均7946.67，高于10–19的6041.2。frame29 PSNR33.46747，before非零11789→after7560，但root4784/7560=63.28%，raw残差条目root62.28%；root占全部GS87.52%。这与前5帧root仅约20%剩余残差不同，**后段root成为主要残差来源**。

raw份额按实际全局delta-row间隔+flags+int16块逐条归属，共同稀疏count header另计；混合LZMA压缩包不能精确分摊给区域。30帧平均birth位姿/source带来的motion净增6007.72 B，加birth属性4102.28 B与索引436.69 B约占完整更新22.4%，不能只用birth.bin解释出生成本。

保存的同cam00 PNG固定ROI诊断：dog(x0:340,y780:1150) SSE份额frame1约4.03%→frame29约15.15%，MSE .000226→.001043；torch(x650:840,y780:1040)份额1.28%→10.34%，MSE .000182→.001813。图中有后段重影；这是保存PNG诊断，不替代official float渲染PSNR，也不单凭ROI直接证明所有相关GS都在root。

目前机制只对nonroot叶split，迟动对象留在root时未必匹配已有区域，缺乏新共享区域。结合root独立残差份额及dog/torch热点上升，root中持续大残差子集的局部区域生长是有依据的下一改进候选；但量化稀疏/有限优化、旧属性固定也可能参与，不能保证仅这一修改解决质量。没有阶段test PSNR，因此不杜撰哪个阶段使质量掉落。

只读frame29 checkpoint的online_history真实ID对齐还显示：当前root稳定GS在frame28/29两段中有观测且位移超过0.0004×extent的点分别5379/4498，两段交集2427。没有补新生点历史，也不是假造当前标签；它支持局部新区域存在候选，但尚不证明接纳后一定改善质量或存储。

## 产物与后续

- 独立质量/FPS及完整更新口径：output/persistent_tree_first30_benchmarks_20261014/sequence.json
- 同协议顺序播放：output/persistent_tree_first30_benchmarks_20261014/frame29_playback.json
- 29帧实际整数/区域raw字节复核：output/persistent_tree_first30_analysis_20261014/packet_analysis.json

100帧暂不启动，先针对已出现的后段质量机制问题改进，再推进长序列。没有重新定义34.2 dB/50000 B/>200 FPS目标，也没有追加对照或消融。
