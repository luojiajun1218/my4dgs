# Reference与缓存组合：24帧完成记录

本轮尝试使用reference几何路径和完整精度计算，保留静态布局、冻结运动成本、渲染属性缓存与整块CPU导出。原共享首帧、300步、所有学习率和storage参数保持原值。

结果仍未严格守住原PSNR和文件上限。训练时间下降明确，FPS小幅提升；仅切换reference开关没有消除微小的训练结果差异。

## 四项结果

动态帧1–24，1600×1199，cam00评测，seed=0。FPS使用同一独占3090Ti协议，恢复第5/10/24帧，10次预热、100次计时、5轮。

| 指标 | 原结果 | 本轮 |
|---|---:|---:|
| 平均PSNR | 34.273313 dB | 34.246734 dB |
| 必要动态文件/帧 | 49,942.71 B | 50,182.42 B |
| 独立前向+clamp FPS | 202.89 | 205.73 |
| 含准备与收尾训练/帧 | 37.1525 s | 19.7358 s |

PSNR低0.026579 dB，文件多239.71 B（0.48%）。纯训练17.6110 s/帧、wall22.4916 s/帧，SSIM0.972476。纯光栅化208.70 FPS，原为208.54 FPS，基本接近。FPS不含加载、指标、解码和训练；文件不含共享首帧与诊断JSON。

## 实现与解释

[配置](../../config/fs_storage_aware_cache_original_order_24f_20261007.yaml)使用旧15000步首帧、214,996 GS，`motion_compute: reference`、`selected_precision_only: false`，其余三个缓存打开。没有更改训练方法、损失或预算。

本轮冻结代码仍先构造paths、再构造regions，而原forward语句按Python参数求值顺序先regions、再paths；硬mask诊断和近并列精度选择也未完全沿用原语义。因此不能把本轮描述为已精确恢复全部原调用行为，也不能把差异只归因于node几何算术。

实验结束后，工作树补丁恢复了reference的regions→paths顺序、super硬mask路径及原softmax→onehot精度选择，node路径保持不动。已有159项数值检查通过。该补丁尚未用于新的24帧训练，其表现未混入本轮结果。

## 核验与产物

- 全24帧顺序恢复摘要匹配；第5/10/24帧checkpoint六类原始属性逐元素精确一致。硬成本与文件实测逐帧相同。
- 正常训练由根启动，PID948601，运行目录`output/fs_storage_aware_cache_original_order_24f_20261007/20261007T001508_4da373/`。运行内`logging.INFO`以Finished结束；公共`launch.log`混入第二次启动失败信息，未作为训练证据。
- 源码冻结在同实验base的`source_snapshot/`，103个文件哈希核查无变化。结果和FPS在`analysis_completed/`，精确恢复在`audit/sequential_001_024.json`。
- 第一次FPS因检测到另一短时CUDA上下文而中止，未生成或使用结果；GPU空闲后按相同协议重测成功。
- 数据和历史结果只读，训练产物不入Git。没有在本轮后自动启动新的训练。
