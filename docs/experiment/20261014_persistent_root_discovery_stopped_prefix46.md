# 持久区域 root 建区：100帧计划提前停止，完成前缀0–46

## Idea与实现

此前后段root积累新运动，本轮从root中寻找连续两次有效观测的移动GS，每帧最多建立2个局部Kmeans叶区域；缺测保留证据，观测静止重置，新生点不补历史。空间特征加入0.05×extent尺度的位置；root专用入口关闭通用feature-distance硬丢弃，仅保留最小8GS、64leaf容量及真实解码位姿有效校验。原有leaf继承ID，迁移/拆分/合并/树规则不叠加改动。

仍为100自由GS→局部维护→100共享/GS联合优化→100birth；本轮没有旧opacity更新。新增root_motion_discovery.py与接线，3项实模型回归覆盖真实建区、新生/缺测、100joint/checkpoint/codec及整帧失败rollback。源码运行中冻结，source_snapshot保存28个文件及hash。

- 配置：config/fs_persistent_tree_rootdiscover_first100_20261014.yaml
- 命令：激活ReConGS后 python main.py --config=config/fs_persistent_tree_rootdiscover_first100_20261014.yaml
- Run：output/fs_persistent_tree_rootdiscover_first100_20261014/20261014T014405_59616e
- **因连续退化提前终止，实际exit130，wall1890.56s；完整frame1–46，frame47在250/300中断，无47summary。100帧未完成，也不是300帧验收。** run_stop_status.json明确保留此状态；GNU time误印退出0不作为真实退出状态。

## 实际结果与速度范围

| 指标 | 完成前缀实测 |
|---|---:|
| 动态frame1–46 saved test PSNR平均 | **33.67806 dB** |
| frame0–46平均（共享frame0沿用相同reference已测值） | 33.68515 dB |
| 动态完整更新 motion+birth+manifest平均 | **52349.83 B** |
| motion / birth / manifest平均 | 46840.74 / 5072.28 / 436.80 B |
| frame46独立stream重放PSNR | **32.26210 dB** |
| frame46纯完整渲染 | 200.032 FPS |
| 2轮顺序前缀读取/校验解码/上传/渲染 | 7.471 FPS |

46个完整packet与manifest最终digest、整数残差统计及文件大小复核通过。最后完整frame46独立重放还检查全部1–46包和属性，其质量与saved一致；未把缺失后半段伪装为完整sequence基准。frame0 reference与前次30帧reference逐字节SHA256相同，复用其已独立测得34.011318 dB；本次没有逐帧独立重新渲染全前缀。纯FPS仅测最后帧，5预热、30次×3轮；不能宣布全300/>200目标完成。共享reference16,656,569 B另计一次。

| 动态段 | 平均PSNR | 完整更新B/帧 | root占非零GS |
|---|---:|---:|---:|
|1–9|34.35824|59672.11|18.08%|
|10–19|34.28736|38555.30|16.50%|
|20–29|33.87197|47916.10|15.39%|
|30–39|33.08404|59869.50|10.97%|
|40–46|32.50471|58233.43|11.35%|

root建区实际采用74次、6587次GS分配（不声称去重人数）；frame46 root仅占残存非零GS8.88%。因此入口已work，持续质量下降不能继续归因于root独立运动主导。frame10/20/30/40维护有真实合并，30拆分3→66与12→67；稳定logical IDs、dense codec映射均经码流验证。内部父节点实际始终0，tree候选多在当前位置/历史位置拟合阶段被拒，不能说树带来收益。

## 隐藏运动子集与容量证据

29checkpoint的真实25–29 history IDs与各帧最终六维int16残差对齐；高残差定义为六通道绝对整数平均>4，当前必须observed，缺测跳过、观测低残差重置连续证据。

| logical区域 | survivorGS | >4单位GS | 连续2/3有效帧 | 高子集占区域整数L1 | 高子集raw条目B | 全区域p90 / 活跃子集p90 |
|---|---:|---:|---:|---:|---:|---:|
|1|28908|2004 (6.93%)|1403 /1128|84.86%|19952 (70.33%)|1.33 /10.33|
|3|4322|887 (20.52%)|615 /482|73.86%|8953 (57.70%)|6.32 /8.52|

region1的split_streak=0，整体p90被多数静止GS稀释；region3 streak=6且下一维护30确实拆分。不能把所有“未拆”都说成尝试失败：维护发生在100步、此证据来自final200步，现日志没有每次split trial/current100 p90或拒绝cost。raw条目按实际delta-gap、flags、int16块计算，不把混合LZMA压缩字节强行分摊。

64leaf容量首次在36填满，37–40和44–46明确leaf_capacity_reached。37时12个leaf共享全零，但各自都有至少1个GS残差，严格all-zero回收不触发。CPU同位姿假设回收26/65/15（仅1/1/2个active GS）保持其GS残差和其他leaf共享参数，真实解码完全相同，但动态layout一次变更使包增加1464/1436/1320 B；每次可释放1名额，不是立即存储收益，也不能据此否定释放容量的长期价值。回收shared非零的34/17则active从1→73、3→165，raw残差增加576/1285 B，故低独立残差不等于低共享价值。此假设诊断只在CPU内存副本执行，未应用到训练或历史码流，不作为训练实验结果。

同cam00保存PNG里dog/torch热点仍增长：frame1→28 SSE份额4.06→13.48%、1.29→9.09%，对应MSE .000227→.000877、.000183→.001505；这不是official float PSNR，也不能直接由ROI推定GS归属。当前同时存在大区局部运动被p90掩盖、容量长占、旧外观无法消退等候选问题；下一机制选择稀疏旧opacity更新，运动/区域/tree先保持不变，不宣称该修改必然解决质量。

## 新增分析产物

- 最后完整帧独立校验/FPS：output/persistent_tree_rootdiscover_stopped_benchmarks_20261014/frame46_playback.json
- 全46码流整数/字节复核：output/persistent_tree_rootdiscover_stopped_analysis_20261014/packet_analysis.json
- 持续高残差子集与CPU同位姿回收：output/persistent_tree_rootdiscover_stopped_analysis_20261014/hidden_subsets_and_retirement_v2.json
- 复现分析：scripts/diagnose_persistent_motion_subsets.py（只读输入、拒绝覆盖输出）
