# ReCon-GS exact index-delta 在线写盘与四项验收 — 2026-10-03

## 目标与实现

让 ReCon-GS 逐帧保存的 motion 包省下索引空间，同时保持原训练结果和播放画质。该改动只重排索引数组，不量化 motion，也不改训练拓扑、loss 或 optimizer。

`scripts/run_recongs_native_stream.py` 现在可在每帧训练结束时在线写 exact-index-delta 包。旧配置仍默认为 `legacy`；配置 `runner_options.native_index_encoding: exact_index_delta`，或用 `RECON_GS_NATIVE_INDEX_ENCODING` 覆盖。第一帧和不划算的帧写完整索引；其余帧记录 surviving old-base rows 中改变的索引 cell，以及新行的完整索引。编码器按两种候选的实际压缩 NPZ 字节数选更小者。写盘后立即解码，并与本帧冻结的索引逐元素比较；删行继续按旧 base、上一帧 birth、再升序保留的行序恢复。

完整压缩包、继承式旧包和 exact-delta 包都可解码。测试覆盖默认兼容、首帧、坏维度/越界 ID、较大差分回退、重复/越界 changed cells、删除及 birth 行映射、旧格式与精确索引恢复。新增 `scripts/benchmark_recongs_index_delta_io.py`，用于在真实帧快照上交错、重复测量两种 writer。

## 四项验收

| 指标 | REF / 对照 | Exact-index-delta | 判断 |
|---|---:|---:|---|
| 画质 | 33.94978 dB PSNR，0.968486 SSIM，L1 0.014275 | 299 帧回放逐帧对照最大差值：PSNR、SSIM、L1 均为 0 | 相同 |
| 实际总存储 | 同为 ZIP_DEFLATED 的完整索引：462,924,997 B | 299,631,425 B；149 帧用差分、150 帧用完整索引 | 少 163,293,572 B（35.27%） |
| 训练时间 | 9.44936 s/动态帧（200 步） | training forward/backward/optimizer 不变；3 帧在线 smoke 为 9.63471 s/帧 | 计算时间相当；短 smoke 慢约 2.0%，样本不足以视为配对回归 |
| 播放时间与 FPS | 全序列回放均值 20.98818 ms/帧，47.646 FPS | 20.50766 ms/帧，48.763 FPS | 全序列端到端回放快约 2.29%；renderer-only 抽样基本持平 |

存储比较包含 299 个 motion NPZ、birth PLY、transition、manifest，以及只计一次的 22,360,216 B 训练后 frame-0 base。所有数字从磁盘文件重新核算。原生未压缩 REF 流为 1,680,697,544 B；它和压缩结果的差别混有 serializer 变化，不作为 35.27% 的公平比较分母。

对 REF 所有 299 个状态做 3 轮交错、共 897 对 writer/磁盘测量。每轮随机打乱帧序与两种 writer 顺序；数组计时前预载一次，输出写入临时目录并 fsync，不清空操作系统页缓存。完整索引均值为 253.88 ms/包，exact writer 为 490.55 ms/包，多 236.67 ms/包（每轮分别多 237.75、237.28、234.97 ms）；轮间变化约 1.2%。折合整段多约 70.76 s，约为 REF 含验证和 I/O 的 12.23133 s/帧的 1.93%。按包选择分组，149 个差分帧的 writer 为 275.74 vs 253.59 ms，平均多 22.14 ms、少 1,095,942 B；150 个 full fallback 帧为 703.93 vs 254.17 ms，平均多 449.76 ms且不省字节。

精确 writer 对动态帧 2–299 的分段计时均值为：差分构造 7.15 ms、完整索引候选压缩 252.91 ms、exact 候选压缩 227.76 ms、实际文件写入 2.28 ms。序列首帧没有差分候选，故不纳入这项分段均值。重复压缩候选是主要额外 CPU 成本，差分构造和磁盘写入占比较小。在线 smoke 的同步训练段为 9.63471 s/帧（原 REF 9.44936 s，约 +1.96%），含验证和 I/O 的帧循环为 12.53953 s（原 REF 12.23133 s，约 +2.52%）；它只覆盖 3 帧，不能替代整段配对训练时间。

profile 还检查了“先压缩 index-only 探针、再压缩胜出完整包”的改法。探针选择出的编码与原算法一致，完整 299 帧实际包字节也逐帧一致，但这一路径 CPU 编码写盘耗时 180.17 s，比原双完整候选的 147.03 s 慢约 22.5%，因此没有保留。当前实现继续按两份完整 NPZ 的实际字节择小者；全帧交错配对计时再次确认其较高 fallback 成本。没有保留实测变慢的 writer 优化。

RTX 3090 Ti 播放测量每种流各两遍、每遍 299 帧。逐帧 cam00 的三项画质差最大值均为 0；两种回放最后的 base/birth Gaussian 状态相对最终 checkpoint 的每字段最大绝对差也均为 0。端到端计时包含包读取/解码、删行与 motion 状态应用、birth PLY 加载和一次同步 render；排除相机/GT 准备。renderer-only 测量为 5 次预热后 30 次同步 forward，exact 与完整索引抽样均值分别约 186.72 和 187.40 FPS，属于持平范围。

因此，这个工程基座达到“质量近似相同、实际存储明显更少、播放 FPS 相当或略高、训练计算时间相当”的工程门槛。在线 writer 会增加 host 编码与写盘时间；完整新 run 中包含验证和 I/O 的帧循环慢了 3.51%，需作为代价一并报告。它不证明自适应分层或 storage loss 的论文贡献已经完成。

## 实际在线 smoke

使用原有 200 步设置、限制为 3 个动态帧，未重训 299 帧。在线记录依次显示 frame 1 写 full（code 0），frame 2 因差分更大回退 full（code 0），frame 3 写 exact-index-delta（code 2，350,090 B）；三个包都通过写后解码检查。输出目录：

`output/recongs_native_stream_online_smoke_delta_20261003/recongs_flame_steak_native_exact_index_delta_smoke_delta_20261003/20261003T050645_531ed1`

训练数据继续通过现有软链接读取，没有复制数据，也没有修改历史输出或链接目标。实现初验时先将 REF motion 快照的 299 帧重编码并逐帧校验，再用 3 帧 smoke 实际运行在线路径；后续为核实完整在线训练开销，另完成了一次 299 帧新训练复验，见下节。

## 完整在线训练复验（299 帧）

为确认 writer 在真实长序列训练中的开销和落盘结果，另启动一次完整的在线 exact-index-delta 训练。使用原有 200 步/帧、同一 frame-0 checkpoint、300 帧配置和相同数据/训练/测试设置；没有重新训练 frame 0。配置只更换 `experiment_name`，没有指定新 seed。运行元数据记录动态范围 `[1,299]` 且没有帧数上限覆盖，最终 299 条记录、manifest 和 run metadata 均显示 completed。

新 run 为 `output/recongs_native_stream_20261003/recongs_flame_steak_native_exact_index_delta_live299_20261003T0622/20261003T063719_a893eb`。299 个包重新打开并独立顺序解码；所有 index 均在 anchor 范围内，写入候选大小与磁盘实际大小逐包相符，记录的 translation 重建最大误差均为 0。149 帧选 exact-index-delta，150 帧按压缩后实际体积回退 full。

| 实测项 | 完整在线 run | 原 REF | 判断 |
|---|---:|---:|---|
| cam00 平均 PSNR | 33.90519 dB | 33.94978 dB | 差 −0.04459 dB；没有显式 seed，按整段平均比较，不把随机差异归因于索引编码 |
| cam00 平均 SSIM / L1 | 0.968579 / 0.014256 | 0.968486 / 0.014275 | SSIM 略高，L1 略低；与 REF 质量范围一致 |
| 实际流总字节（含一次共享 base） | 299,840,238 B | 同 run full-index 候选 463,086,475 B | 节省 163,246,237 B（35.25%） |
| 同步训练段 | 9.43764 s/帧 | 9.44936 s/帧 | −0.12%，计算段相当 |
| 含验证和 I/O 的帧循环 | 12.66066 s/帧 | 12.23133 s/帧 | +3.51%；完整帧的训练计算相近，在线双候选压缩增加 host 时间 |
| 每帧 renderer-only benchmark | 187.35 FPS | 187.14 FPS | 相当；完整端到端播放项仍以同一状态的 REF 双流回放测量为准 |

完整在线 run 的字节分解为：motion NPZ 271,001,058 B、birth PLY 5,880,496 B、transition JSON 535,297 B、manifest 63,171 B、共享 frame-0 base 22,360,216 B。公平的 full-index 总量使用同一 run 每帧记录的 full candidate 压缩包大小求和，并保持相同 birth、transition、manifest、base；不是与另一随机训练流比较。

与旧 REF 配对的完整回放仍是序列化播放差异的严格测量：同一 ReCon 状态每种流两遍、每遍 299 帧，PSNR/SSIM/L1 逐帧最大差均为 0，最终 base/birth 状态的所有 Gaussian 属性最大绝对差也均为 0。完整索引回放为 20.98818 ms/帧、47.646 FPS；exact 为 20.50766 ms/帧、48.763 FPS（快约 2.29%）。此 GPU 回放测试针对 bitwise-identical 的旧 REF 重编码流；没有把它写成新随机 run 的 299 帧完整播放测量。

独立审计 JSON：`/tmp/recongs_live299_exact_index_delta_audit_20261003.json`。运行配置 SHA-256 为 `6905b9699287507d78f17dee5c93b55305d84b44e95f50d900837f1bdfcccc9e`；runner SHA-256 为 `564ab78748eaa5e5d6d7162167abe5272ba8d9f3cc5dc676db203f89dce509ae`。数据仍通过既有外部软链接读取。冻结 ReCon-GS commit 为 `36e87f16fddd5b7e984dba613a3ebb9a3f347f82`，source tree hash `7fd01ec12016f4749d2f5fca725aaa175001319884f40249cb0bf280fa18bc4b`；使用已校验的只读快照 `/tmp/recongs_native_source_snapshot_smoke_delta_20261003`。

## 复现命令与审计材料

`unittest discover` 覆盖 29 项（13 项实验清单测试、16 项 native-stream 测试）；`test_nested_prototype.py` 使用自定义入口，另执行 57 项数值检查。两个入口合计 86 项通过。单跑 unittest discovery 不会执行那 57 项自定义检查。

```bash
timeout 600s conda run --no-capture-output -n ReConGS \
  python -m unittest discover -s tests -v \
  > /tmp/recongs_exact_index_delta_unittest_20261003.log 2>&1

timeout 600s env \
  KEOPS_CACHE_FOLDER=/tmp/recongs_exact_index_delta_keops_20261003 \
  MPLCONFIGDIR=/tmp/recongs_test_mpl_20261003 \
  conda run --no-capture-output -n ReConGS \
  python tests/test_nested_prototype.py \
  > /tmp/recongs_exact_index_delta_nested_host_20261003.log 2>&1
```

自定义数值检查使用多 worker DataLoader，需要本机支持本地 multiprocessing socket；默认受限沙箱会以 `EPERM` 阻止它，故此处在 host 执行，并将 KeOps/Matplotlib 缓存定向到 `/tmp`。本次两条命令分别通过 29 项和 57 项检查。

全 299 帧、3 轮交错的真实包 writer/磁盘计时；每轮 299 对、两种刚写出的包均解码并与 REF 索引逐元素比较：

```bash
conda run -n ReConGS python scripts/benchmark_recongs_index_delta_io.py \
  output/recongs_native_stream_20261003/recongs_flame_steak_ref_exact_index_delta_20261003/compressed_full_control_299 \
  --frames all --repeats 3 \
  --output /tmp/recongs_exact_index_delta_all299_paired_io_20261003.json
```

REF 与两种候选的 299 帧逐项核对：

```bash
conda run -n ReConGS python scripts/verify_recongs_native_stream.py \
  output/recongs_native_stream_20261002/recongs_flame_steak_native_delta_index_seed0_20261002_rerun01/20261002T231157_base02 \
  output/recongs_native_stream_20261003/recongs_flame_steak_ref_exact_index_delta_20261003/compressed_full_control_299 \
  --output /tmp/recongs_compressed_full_control_verify.json

conda run -n ReConGS python scripts/verify_recongs_native_stream.py \
  output/recongs_native_stream_20261002/recongs_flame_steak_native_delta_index_seed0_20261002_rerun01/20261002T231157_base02 \
  output/recongs_native_stream_20261003/recongs_flame_steak_ref_exact_index_delta_20261003/repacked_full299_verified \
  --output /tmp/recongs_exact_index_delta_verify.json
```

实际在线 writer smoke（配置仍是原 200 步；环境变量把序列限为 3 个动态帧）：

```bash
env RECON_GS_MAX_DYNAMIC_FRAMES=3 \
  RECON_GS_ROOT=/tmp/recongs_native_source_snapshot_smoke_delta_20261003 \
  PYTHONDONTWRITEBYTECODE=1 \
  timeout 600s conda run --no-capture-output -n ReConGS \
  python scripts/run_recongs_native_stream.py \
  config/recongs_flame_steak_native_exact_index_delta_smoke_delta_20261003.yaml
```

完整在线 299 帧复验的等价命令（完整运行约一小时；复跑需使用新的 `experiment_name`，以免覆盖产物）：

```bash
timeout 21600s env \
  -u RECON_GS_MAX_DYNAMIC_FRAMES \
  -u RECON_GS_NATIVE_INDEX_ENCODING \
  -u RECON_GS_SOURCE_ARCHIVE \
  -u RECON_GS_SOURCE_ARCHIVE_RECORD \
  RECON_GS_ROOT=/tmp/recongs_native_source_snapshot_smoke_delta_20261003 \
  RECON_GS_BASE_RUN=/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR=/tmp/recongs_live299_mpl_20261003T0622 \
  conda run --no-capture-output -n ReConGS \
  python scripts/run_recongs_native_stream.py \
  config/recongs_flame_steak_native_exact_index_delta_live299_20261003T0622.yaml \
  > /tmp/recongs_live299_exact_20261003T0622_r2.log 2>&1
```

新训练产物的离线包/索引/字节核验命令：

```bash
conda run -n ReConGS python scripts/audit_recongs_native_online_run.py \
  output/recongs_native_stream_20261003/recongs_flame_steak_native_exact_index_delta_live299_20261003T0622/20261003T063719_a893eb \
  --reference-run output/recongs_native_stream_20261002/recongs_flame_steak_native_delta_index_seed0_20261002_rerun01/20261002T231157_base02 \
  --output /tmp/recongs_live299_exact_index_delta_audit_20261003.json
```

全 299 帧 writer 观测保存在 `/tmp/recongs_exact_index_delta_all299_paired_io_20261003.json`。GPU 回放原始记录与独立断言结果分别在 `/tmp/recongs_native_stream_playback_20261003_optimfree.json` 和 `/tmp/recongs_exact_index_delta_playback_verified_20261003.json`。REF→exact 解码核验结果在 `/tmp/recongs_exact_index_delta_existing_candidate_verified_20261003.json`；所选 299 个 exact 输出包还与已有在线方案逐帧比较了 encoding、大小和 SHA-256，均一致。新在线训练的独立审计结果在 `/tmp/recongs_live299_exact_index_delta_audit_20261003.json`。未保留的探针尝试与解码核验记录在 `/tmp/recongs_exact_index_delta_probeoptimized_full299_20261003/stream_metrics/experiment_summary.json` 和 `/tmp/recongs_exact_index_delta_probeoptimized_verify_20261003.json`。项目后续研究顺序见[路线总结](20261003_recongs_project_roadmap.md)。本次完成了一次新的完整 299 帧在线训练复验，但没有把它表述成新训练算法的质量提升。
