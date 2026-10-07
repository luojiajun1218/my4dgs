# 原版 ReCon-GS 同步 FPS 测速

## 要确认什么

原版日志约 440 FPS，我们的独立测速约 206 FPS。检查统一计时后是否仍有两倍差距。

本次读取已保存的模型进行渲染。新增测速配置，使用临时测速脚本恢复原版模型；项目训练和渲染实现没有改动。

## 测速方式

- RTX 3090 Ti，ReConGS 环境，cam00，1600×1199。
- 每轮预热 10 次，计时 100 次，重复 5 轮。
- 每批计时前后同步 CUDA，FPS 为总渲染次数除以总耗时。
- 完整前向包含属性准备、渲染与 clamp；纯光栅化提前准备属性和相机参数。
- 计时排除模型加载、图片读取、质量指标、码流解码和训练。
- 前后检查 GPU 上没有其他计算进程。

原版只保存了第 0、299 帧的 checkpoint，因此测这两个点。我们的版本重测第 5、10、24 帧。帧号不同，本次不是同帧或完整序列的性能比较。

## 结果

| 模型 | 帧 | GS 数 | 完整前向＋clamp FPS | 纯光栅化 FPS |
|---|---:|---:|---:|---:|
| 原版 ReCon-GS | 0 | 214,996 | 203.08 | 209.25 |
| 原版 ReCon-GS | 299 | 215,235 | 218.85 | 228.48 |
| 当前版本 | 5 | 215,001 | 205.94 | 209.59 |
| 当前版本 | 10 | 215,010 | 206.47 | 208.98 |
| 当前版本 | 24 | 215,017 | 205.94 | 208.62 |

当前版本三个采样帧按总耗时合计为 **206.12 FPS**，纯光栅化为 **209.06 FPS**。此前对应结果为 206.19、208.95 FPS，重测接近。

原版第 0 帧历史日志为 491.77 FPS，第 299 帧为 398.02 FPS；本次同步测速分别为 203.08、218.85 FPS。原版验证计时在 `forward` 和 `clamp` 前后读取 CPU 时间，没有显式同步 CUDA。GPU 尚未完成的工作可能在后续保存图片或质量计算时才被等待，造成日志 FPS 偏高。历史单次计时与本次预热批次计时也有差异。

结论：本次测得的原版和当前版本都在约 200 FPS 的范围，没有出现两倍差距。原版末帧纯光栅化也更快；帧、位姿与屏幕覆盖不同，不能把剩余差距归因于分层方法。GS 总数相近也不能保证渲染耗时相同。

## 恢复检查

- 原版首帧恢复 PSNR 为 34.004742 dB，历史日志为 34.0047 dB。
- 原版末帧恢复 PSNR 为 34.270901 dB，历史日志为 34.2709 dB。
- 原版末帧恢复基础 GS 与新增 GS 的合并渲染视图，包含 239 个新增 GS；仅恢复基础模型会遗漏这些 GS。
- 原版 PSNR、SSIM、L1 与历史日志误差均小于 0.0005；当前版本恢复质量也通过既有测速脚本检查。
- 原版两个 checkpoint 读取前后 SHA-256 相同；没有执行优化器更新。

## 文件与命令

原版运行目录：`/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf/`。

当前运行目录：`output/fs_storage_aware_polish03_24f_20261007/20261007T014058_fadd3f/`。

配置：[原版测速参数](../../config/fps_recongs_same_protocol_20261007_103004.yaml)、[当前版本测速参数](../../config/fps_materialized_3090ti_20261006.yaml)。

本次产物：`output/fps_recongs_same_protocol_20261007_103004/`。其中 `native_fps.json`、`ours_fps.json` 保存逐帧质量与五轮计时；`comparison.json` 保存范围说明和脚本、配置哈希；`native_benchmark.py` 保存临时测速脚本。

```bash
conda activate ReConGS
python output/fps_recongs_same_protocol_20261007_103004/native_benchmark.py \
  --protocol config/fps_recongs_same_protocol_20261007_103004.yaml \
  --output /tmp/recongs_native_fps_new.json
python scripts/benchmark_materialized_fps.py \
  output/fs_storage_aware_polish03_24f_20261007/20261007T014058_fadd3f \
  --protocol config/fps_materialized_3090ti_20261006.yaml \
  --output /tmp/my4dgs_fps_new.json
```

复测时选择尚不存在的输出文件。原版数据通过原有软链接读取，没有复制大型数据或修改链接目标。
