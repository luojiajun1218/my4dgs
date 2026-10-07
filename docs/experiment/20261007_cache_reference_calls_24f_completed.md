# 原reference调用恢复与缓存：24帧完成记录

本轮恢复原reference的调用行为，保留缓存和批量CPU导出。文件和速度达到要求，PSNR仍比原值低0.0141 dB，因此未宣称严格画质目标完全达成。

## 四项结果

动态帧1–24、1600×1199、cam00、seed=0。

| 指标 | 原结果 | 本轮 |
|---|---:|---:|
| 平均PSNR | 34.273313 dB | 34.259236 dB |
| 必要动态文件/帧 | 49,942.71 B | 49,941.17 B |
| 独立前向+clamp FPS | 202.89 | 206.49 |
| 含准备与收尾训练/帧 | 37.1525 s | 19.7713 s |

PSNR差−0.014078 dB，文件少1.54 B/帧，FPS提高约1.77%，含准备收尾训练缩短约46.78%。SSIM为0.972518，纯训练17.6455 s/帧，wall22.4859 s/帧。

FPS独立恢复第5/10/24帧，在3090Ti独占条件下预热10次、计时100次、重复5轮。纯光栅化209.43 FPS，原为208.54 FPS。FPS不含加载、指标、码流解码和训练；文件包含运动流及补点PLY，不含共享首帧和诊断JSON。

## 核心改动

[配置](../../config/fs_storage_aware_cache_reference_calls_24f_20261007.yaml)使用旧15000步共享首帧、214,996 GS；原300步调度、学习率、storage、精度档位、相机与诊断参数不变。`motion_compute: reference`、`selected_precision_only: false`，布局/冻结运动成本/渲染属性缓存打开。

最小补丁恢复三处原行为：前向先构造regions再构造paths，避免改变autograd子图建立顺序；硬mask沿用原super路径；reference的符号和码率精度选择沿用softmax→onehot后argmax，避免近并列浮点选择差异。node分支未更改。

新增确定性回归检查前向与硬mask调用顺序，并在联合阶段验证raw logits与softmax近并列时的选择。159项数值检查、32项unittest通过。

缓存减少重复布局、量化成本和属性准备，训练提速明确。恢复调用行为后画质更接近原值，但单次训练仍有微差；不能仅凭本轮确认剩余差异的来源，也没有用降低评测尺寸掩盖差异。

## 产物与恢复证明

- 正式运行：`output/fs_storage_aware_cache_reference_calls_24f_20261007/20261007T005518_7b88a4/`，日志以Finished结束。
- 全24帧顺序解码状态摘要匹配；第5/10/24帧checkpoint六类原始属性逐元素精确一致。硬成本与实际文件逐帧一致。
- 同实验base的`source_snapshot/`冻结104个文件，哈希核查无变化。统计、曲线和FPS在`analysis_completed/`；顺序恢复在`audit/sequential_001_024.json`。
- 数据与历史输出只读，未复制大数据或修改链接目标。原始训练产物不入Git。
