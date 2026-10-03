# ReCon-GS 原生逐帧基线与公平比较审计 — 2026-10-03

## 结论

完成了 299 个动态帧的 ReCon-GS 原生流基线，逐帧重放到最终状态，并按相同序列化格式复核 GradHet sparse-v2 的文件大小。该基线补齐了此前缺少的逐帧 `delta/index`、新生 Gaussian 和删点/行序信息。

现在不能说 GradHet sparse-v2 已同时达到四项目标：与 ReCon-GS 相比，它的全序列 cam00 画质仍低约 0.167 dB / 0.00418 SSIM；在相同压缩序列化边界下，v2 流文件大约多 24.96%。新的 v3 索引表表示已把流大小减少约 6.84%，但仍比 ReCon-GS 多 16.41%。训练时间尚无同预算、同计时边界的 GradHet 全序列数据；渲染速度只在三个抽样帧完成配对，不能据此判定全序列不降速。

## ReCon-GS 基线

- 完整运行目录：`output/recongs_native_stream_20261002/recongs_flame_steak_native_delta_index_seed0_20261002_rerun01/20261002T231157_base02/`。运行使用固定源快照：ReCon-GS commit `36e87f1` 加当时 `main.py`、`pipeline/ReCon-GS/module.py` 两项工作区 overlay；归档和 SHA-256 位于该目录的 `provenance/` 与 `stream_metrics/run_metadata.json`。数据继续通过现有软链接读取，没有复制数据集。
- 从共享训练帧 0 开始，对动态帧 1–299 各训练 200 步。每帧流包含原生 anchor `delta/index`、新增点 PLY、删除索引/行数转换记录。299 个帧记录、motion 包、birth PLY 和 transition 均齐全，最终 `ckpt-299-200.pth` 为 67,528,602 B。
- 共享基底是训练后的 frame-0 PLY，214,996 行、22,360,216 B，只计一次。相邻目录中的 2,040,265 B `input.ply` 是 40,000 点初始输入，不是播放器使用的训练后基底。

| 指标 | ReCon-GS 全序列结果 | 口径 |
|---|---:|---|
| PSNR | 33.94978 dB | 299 帧，每帧唯一 held-out 测试视角 cam00 的均值 |
| SSIM | 0.968486 | 同上 |
| L1 | 0.014275 | 同上 |
| 训练段时间 | 9.44936 s/动态帧 | 200 步；CUDA 同步；从 pre-step 到 final post-step；不含验证、checkpoint、最终 PLY 和 packet/metadata 写盘 |
| 算法时间估计 | 9.44926 s/动态帧 | 从同步墙钟中只扣除测得的 CUDA tensor-snapshot copy；仍含 Python hook/event 开销，须同时看原始墙钟 |
| 含验证和 I/O 的帧循环 | 12.23133 s/帧 | 同步墙钟 |
| renderer FPS | 均值 187.14，median 187.44 | 每个动态帧的 cam00 `module.forward`；5 次预热 + 30 次 CUDA 同步；无 clamp；不含读图、解包或设备传输 |

本次质量配置只有一个 held-out 测试视角 cam00，因此“质量均值”不是多视角均值。FPS 是纯 renderer forward；不能与包含图像读取、packet 解码或状态更新的整段回放 FPS 混用。旧 ReCon-GS 日志约 8.82 s/动态帧、frame 299 约 398 FPS，分别使用未显式同步的旧 timer 和 `time.time()` forward+clamp 计时，不作为本次严格门槛值。

## 重放核验

从训练后共享 frame-0 开始，依次读取 299 组 native motion、transition 与 birth PLY，按 append、prune 和行序规则重建到 frame 299，再与最终 checkpoint 比较。299 帧 motion 和 birth 文件哈希通过；删除索引、行数和索引范围通过；增量 PLY 与 checkpoint 对应字段完全相同。

最终状态中 xyz、SH、opacity、scale 的最大绝对差均为 0；rotation 最大绝对差为 `2.2649765e-6`。因此应报告为数值容差内一致，不能写成逐元素完全相同或 rotation 差为 0。逐帧索引和删除/出生行映射也通过顺序重放校验。

## 存储比较：撤销旧的跨 serializer 推断

所有字节数均来自实际文件。原 ReCon-GS 流用 `np.savez`（ZIP_STORED），GradHet sparse-v2 用 `np.savez_compressed`（ZIP_DEFLATED）；直接比较原目录会把序列化器差异误算成方法差异。**撤销此前基于这两组原始目录大小得出的任何 ReCon-GS 对 GradHet 跨方法节省结论。**

为作同边界比较，审计副本将两种方法的全部 motion/index/weight/row-id 数组都写成 `np.savez_compressed`，对共有 transition 与 manifest 使用相同 JSON schema，并把同一训练后 frame-0 base 计一次。未删去 GradHet decoder 所需的 blend/base indices、row IDs 或 weights；ReCon 的 int64 索引和删点 ID 转成 int32 前先验证范围，结果无损。

| 同格式实际字节 | ReCon-GS | GradHet sparse-v2 |
|---|---:|---:|
| 299 个压缩 motion NPZ | 434,053,663 | 541,627,560 |
| 新生点 PLY | 6,030,886 | 13,946,542 |
| transition + manifest | 192,436 | 192,460 |
| 动态流合计 | 440,276,985 | 555,766,562 |
| 加一次共享训练后 frame-0 base | 462,637,201 | 578,126,778 |

在此边界下，GradHet sparse-v2 比 ReCon-GS 原生流大 24.96%（以 ReCon 为分母）；等价地 ReCon 流比 GradHet 小 19.98%（以 GradHet 为分母）。该离线重序列化耗时不是训练时间，也不是新 codec 的实现结果。

旧记录中 sparse-v2 比 GradHet dense-v1 少 18.54% 的结论仍只适用于 GradHet 内部 dense-v1/sparse-v2 的文件比较；它不是相对 ReCon-GS 的节省率。混合原始 serializer 时测得的 ReCon 1,680,697,544 B 与 GradHet 577,934,318 B 不能用于跨方法结论。

## 下一候选：逐帧共享重复的 child-index 表

先检查了 ReCon-GS 原生流和 GradHet sparse-v2 的真实组成。对 GradHet v2，299 帧压缩 NPZ 中约 88.6% 文件字节来自 `base_index`、blend 行号、逐行重复 child indices 和 weights。每帧的活动 Gaussian 常共享相同 child-index 三元组；这些索引随帧变化，因此 v3 每帧独立构建表，不假设跨帧拓扑固定。

生产导出新增 `index_encoding: shared_table`：对本帧每个活动行的完整 child-index 三元组做精确去重，存一张 `blend_index_table` 和逐行 `blend_index_group`；保留 `blend_rows`、全部 float32 `blend_weights`、`delta`、base map、-1 sentinel 和现有 `np.savez_compressed`。默认仍是旧 `per_row` 表示。decoder 对表做索引展开后沿用原计算；`module.py` 的 sparse playback 也走同一 decoder。它是 NPZ 内的无损结构化表示，没有改容器压缩器或量化数值。

CPU 离线重打包覆盖 299 帧，每行索引表展开与原 v2 的全部 child indices 完全相等，299 帧解码出的 motion 场逐元素一致。index 数组的原始载荷从约 1.029 GB 降至约 351 MB；ZIP 已压缩掉大部分重复，所以实际 motion NPZ 只从 541,627,560 B 降到 502,072,771 B。保持 birth PLY、transitions、manifest 和共享 base 不变后，总流从 578,126,778 B 降到 538,571,989 B，减少 39,554,789 B（6.84%）。它仍比 ReCon-GS 同格式总流 462,637,201 B 大 75,934,788 B（16.41%）。离线去重、写盘和逐帧精确检查合计约 192 s CPU；这不是训练耗时或 playback decoder 延迟。

按 ZIP 实际压缩成员大小拆分（299 帧汇总，另有约 0.52–0.66 MB ZIP 头/目录开销）：

| NPZ 成员 | sparse-v2 | 共享表 v3 |
|---|---:|---:|
| delta | 61,583,470 B | 61,583,470 B |
| base_index | 96,949,549 B | 96,949,549 B |
| blend_rows | 119,604,172 B | 119,604,172 B |
| blend_indices 或 index table + group | 166,325,331 B | 126,629,115 B |
| blend_weights | 96,590,360 B | 96,590,360 B |
| representation、计数与 ZIP 开销 | 574,678 B | 716,105 B |

v3 的 index table + group 包括三层：表分别占 781,662、1,165,755、1,672,232 B；逐行 group 分别占 52,373,407、38,093,270、32,542,789 B。NPZ 总体净降幅是 39,554,789 B。上述真实文件分解也说明，文件中仍有约 119.6 MB 行号和 96.9 MB base index；可进一步研究从父层共享结构恢复行归属，但不能用权重最大项代替真实 assignment。

暖缓存 CPU 解包/解码对照：v2 每帧读入并解压均值 18.31 ms、解码 41.66 ms；v3 分别为 17.52 ms、45.23 ms。共享表让文件读取/解压约快 0.79 ms/帧，索引展开使纯 CPU 解码约慢 3.58 ms/帧（+8.6%）。这不含相机图像加载、GPU 传输或 renderer，也不能推断 GPU renderer-only FPS；完整 renderer FPS 未对 v3 重测。

另外测了按非零权重槽打包索引的方案。它原样保留 `delta`、所有 weights、行号和每个非零权重位置上的索引（包括 -1），将零权重位置当作零贡献省略；同格式总流只减少 8.68 MB（1.50%），不如共享表，不作为当前候选。也排除了用 `argmax(weights)` 推断下一层父节点：frame 1 存在前层最大权重对应同一 child，而下一层 child-index tuple 不同的行，说明这不是可靠映射。

代码入口为 `GatedFourDeformation.export(..., index_encoding='shared_table')`，`ModuleConfig.sparse_export_index_encoding` 可由 YAML 设为 `shared_table`；缺省仍使用 `per_row`，所以历史实验配置行为不变。CPU 定向测试覆盖 v2/v3 等价、三个门全部打开、部分门关闭、两槽 overlap 设置和 -1 sentinel。测试使用 `torch.cdist` 代替 KeOps KNN 初始化，没有启动 GPU。

本候选只改善可验证的磁盘表示；它不改训练算法、画质或 renderer forward，不能称为自适应分层或 storage loss 成功。GradHet v2 的画质差、训练耗时未配对和全序列 renderer FPS 未配对仍未解决。下一步要从 ReCon 质量基线引入能看见索引/权重/出生字节代价的可学习层级与 storage loss，并用训练目标或单独验证集选择门；不可用 held-out cam00 调门。

已有 storage surrogate 仍不是这个实际文件口径：`QuantizedResidualMotion.storage_parts` 按 `region_count` 和 Gaussian 数估运动参数字节；`Module.quantized_storage` 另加新点原始属性、每点 int32 来源号和 int32 删除索引。finalize 报告明确把层级成员映射表、共享首帧、配置/算法元数据、文件头和 optimizer 排除。因此旧实验中约 11.78 KiB/帧是参数估算，不是实际 NPZ/PLY 文件大小；新的 rate proxy 至少应覆盖 base/row/group/child index、weights、delta、birth、deletion，并用落盘 serializer 字节验证，而不能只对容器 ZIP 压缩率宣称论文收益。

## 复现审计材料

- 完整帧级记录和汇总：运行目录 `stream_metrics/frames.jsonl` 与 `experiment_summary.json`。
- 重放与同格式存储审计：`/tmp/recongs_gradhet_samezip_20261003_run01/audit_results.json`、`storage_comparison_summary.json`。
- 固定基线源归档、配置哈希和 shared base 哈希：运行目录 `provenance/` 与 `stream_metrics/run_metadata.json`。
