# ReCon-GS 拓扑继承候选：299 帧验证

## 想法

在不重建层级的帧上，保留上帧仍存活 Gaussian 的 `p2v` anchor 索引，只对新生且存活的行做 KNN；重建帧仍完整计算索引。顺序解码时从前一帧索引和删点映射恢复存活行，再读入新生行的索引。包中没有重复写存活行索引。

这会改变非重建帧使用的训练拓扑，不只是改变文件格式。训练仍为每个动态帧 200 步。没有增加量化或新 codec。

## 实验材料

- 配置：[recongs_flame_steak_topology_reuse_full299_200step_20261003.yaml](../../config/recongs_flame_steak_topology_reuse_full299_200step_20261003.yaml)
- 实现：[run_recongs_native_stream.py](../../scripts/run_recongs_native_stream.py)
- CPU 包往返回归：[test_recongs_native_stream.py](../../tests/test_recongs_native_stream.py)
- ReCon-GS 对照：[20261003_recongs_native_baseline_audit.md](20261003_recongs_native_baseline_audit.md)
- 候选运行：`output/recongs_native_stream_20261002/recongs_flame_steak_topology_reuse_full299_200step_20261003/20261003T014825_topofu/`
- 同序列化器审计副本：`/tmp/recongs_topology_reuse_samezip_20261003/`；汇总为 `storage_comparison_summary.json`。
- 定向 CPU 包 round-trip 测试 6/6 通过；另做了下文的完整流独立重放。

使用固定 ReCon-GS 快照（commit `36e87f16fddd5b7e984dba613a3ebb9a3f347f82`，源树 SHA-256 `7fd01ec12016f4749d2f5fca725aaa175001319884f40249cb0bf280fa18bc4b`，归档 SHA-256 `e30e4311f4a6ec4cf2135b1fe5752b17fcc7b2669fc0a1ec5a8e01fd63457a58`）、同一训练后 frame-0 base 和相同数据软链接。数据仍指向 `/home/gamer-02/Documents/DeepLearning/ReCon-GS/data/colmap4d/flame_steak`，未复制。候选配置 SHA-256 为 `634bad6d80f021000eae0d3fde113aff7e9a3af991bd3e6c9dbd6dd200cfc247`。299 个动态帧均训练 200 步，运行状态为 `completed`。共享首帧训练 PLY 有 214,996 行、22,360,216 B，只计一次。

之前的 10 帧 pilot 中，训练视角 PSNR 差为 +0.0177 dB、训练时间约 −0.006 s/帧，不能代表完整序列；本报告以下结论以 299 帧配对结果为准。

## 质量、时间与速度

每项都按相同 frame 1–299 配对。区间是逐帧配对差的近似 95% 区间，未建模相邻帧的时间相关性。训练视角指标用于判断算法；cam00 是单一 held-out 测试视角，仅作最终评测。差值均为候选减 REF。

| 指标 | ReCon-GS | 拓扑继承 | 配对差（近似 95% 区间） |
|---|---:|---:|---:|
| 训练视角 PSNR | 35.40929 dB | 35.37216 dB | −0.03713 dB（−0.05430, −0.01996） |
| 训练视角 SSIM | 0.9691463 | 0.9691338 | −0.0000125（−0.0000301, +0.0000051） |
| 训练视角 L1 | 0.0106153 | 0.0106331 | +0.0000178（+0.0000127, +0.0000229） |
| 训练段同步墙钟 | 9.44936 s/帧 | 9.48006 s/帧 | +0.03069 s/帧（+0.02193, +0.03946），约 +0.325% |
| 算法时间估计 | 9.44926 s/帧 | 9.47995 s/帧 | +0.03069 s/帧 |
| cam00 renderer-only FPS | 187.142 | 187.231 | +0.088（−0.106, +0.283） |

训练计时从 pre-step 到最后 post-step，CUDA 同步，包含帧准备、取数和 200 步训练；不含验证、checkpoint、最终 PLY 和 packet/metadata 写盘。算法时间估计只扣除测得的 CUDA snapshot copy，仍包含 Python hook/event 开销。两个时间口径都显示候选略慢；本次没有单独 profile 到足以解释约 31 ms 差值的单一来源。含验证与 I/O 的帧循环均值为 12.2313 s（REF）和 12.3113 s（候选）。

FPS 每帧使用同一个 cam00 相机，先 warmup 5 次，再同步测 30 次 `module.forward`，不含 clamp、图像读取、packet 解码或 GPU 上传。均值略升，但配对区间跨 0；只能说没有测出明确下降，不能证明严格不降。

cam00 最终全序列质量为：PSNR 33.94978 → 33.93419 dB（−0.01560 dB，区间 −0.03134 至 +0.00015）、SSIM 0.9684855 → 0.9685006（+0.0000151，区间 −0.0000070 至 +0.0000372）、L1 0.0142752 → 0.0142900（+0.0000148，区间 +0.0000073 至 +0.0000222）。这是单一 held-out 视角；这些数值没有用于挑选候选。

全序列的训练 PSNR 差没有随帧号单调变差，但中段有明显波动：

| 帧段 | 训练视角 PSNR 差（候选−REF） |
|---|---:|
| 1–50 | +0.0017 dB |
| 51–100 | +0.0140 dB |
| 101–150 | −0.1686 dB |
| 151–200 | −0.0631 dB |
| 201–250 | +0.0195 dB |
| 251–299 | −0.0260 dB |

PSNR 的帧均值差为负且区间低于 0，L1 也变差；全量结果不支持“画质持平”。按 packet 模式分组，full-index 帧训练时间差 +0.0300 s、PSNR 差 −0.0323 dB；inherited 帧时间差 +0.0314 s、PSNR 差 −0.0420 dB。两组训练时间增量接近，现有计时还不能把约 31 ms 的慢化归因到单独的继承分支。质量差也不只出现在继承包当帧，完整状态和时间序列都需要评估。

## 实际文件大小

原始运行输出使用 `np.savez`，299 帧 native motion packet 共 883,509,755 B；出生点 PLY 5,978,256 B；transition JSON 435,691 B；sequence manifest 63,160 B；共享训练 base 22,360,216 B。实际总流为 912,347,078 B。ReCon-GS 原始同类流总计 1,680,697,544 B；原始 on-disk 文件少 45.72%。

为与已经验证的压缩 REF 口径对齐，另把双方所有 packet 数组按 `np.savez_compressed` 实际写入审计副本。保留全部数组；所有 int64 ID 先检查范围，再无损转成 int32；float32 motion/continuous 数值不变。birth PLY 原样复制，transition/manifest 写成双方相同的解码 schema，继承规则、删点行映射和共享 base 都计入。frame-0 base 只计一次。边界针对可恢复 Gaussian 状态流；`input.ply`、训练 checkpoint、TensorBoard 和 provenance 不是逐帧播放状态，因此未计；相机/图像来自双方共用的外部数据软链接。

| 同格式审计副本（实际文件字节） | ReCon-GS | 拓扑继承 |
|---|---:|---:|
| 299 个压缩 motion NPZ | 434,113,164 | 266,789,534 |
| 新生点 PLY | 6,030,886 | 5,978,256 |
| transition + manifest | 245,461 | 245,443 |
| 动态流 | 440,389,511 | 273,013,233 |
| 加一次共享训练 frame-0 base | 462,749,727 | 295,373,449 |

候选比同 schema REF 少 167,376,278 B（36.17%）。先前压缩 REF 数字为 462,637,201 B；这次统一 schema 更明确记录了索引解码规则、删点行映射和 packet code，并在 REF packet 增加 full-index code，因此 REF 总数增加 112,526 B。按先前数字对照，候选仍少 36.15%。这只是同 serializer 的实际 NPZ 文件大小复核，不是训练耗时，也没有引入新 codec。

## 顺序重放与结论

CPU 独立解码从共享 frame-0 开始顺序读取 299 包：150 个完整索引包、149 个继承包。packet/PLY 哈希、删点范围、逐帧行映射、索引范围均通过。重建末帧与 checkpoint 对比，xyz、SH、opacity、scale 的最大绝对差为 0；rotation 最大绝对差为 `2.26498e-6`，在数值容差内一致。

| 目标 | 结果 |
|---|---|
| 画质持平或更好 | 未达到：训练视角 PSNR/L1 有小幅但一致的退化 |
| 实际保存存储减少 | 达到：同 serializer 全流少 36.17% |
| 每帧训练时间不增加 | 未达到：同步训练段慢约 0.325%，配对区间高于 0 |
| renderer FPS 不降低 | 均值略高但区间跨 0；尚不能证明非劣 |

本候选证明了跨帧索引继承可以大幅减少实际文件字节，但以改变拓扑分配为代价。下一版需降低继承路径的 host/row-map 开销，并改善训练 PSNR/L1，再用完整 299 帧配对验证；不能沿用 10 帧 pilot 结论。
