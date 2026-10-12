# 新增 GS 阶段带来多少测试画质收益

完整 300 帧实验没有达到画质目标。本次用已保存的第 199、249 帧检查点，分别渲染运动结束时的旧 GS 和新增阶段结束后的最终模型，判断下一轮应该优先调整哪个阶段。没有重新训练或改变模型。

| 帧 | 运动结束 PSNR | 新增结束 PSNR | 增益 |
|---|---:|---:|---:|
| 199 | 33.120903 | 33.158451 | +0.037548 |
| 249 | 33.384499 | 33.419418 | +0.034920 |

两个样本中的新增阶段收益都很小，不能据此推断所有帧都如此，也不能据此断言新增 GS 无用。它们支持先检查运动阶段的压缩约束，暂不直接增加新增步数或 GS 数。下一轮候选是只降低运动存储权重，需独立对照验证。

第 1200 步 checkpoint 已完成运动 optimizer 更新、运动提交和初始 newborn 创建。因此诊断只装载其中的旧 `gaussians`，排除单独保存的 `births`；直接恢复整个 checkpoint 渲染会混入初始 newborn。第 1400 步 checkpoint 则已完成最终新增优化、剪枝和量化材质化。

使用本轮训练的冻结源码，通过相同完整 `Module.forward + clamp` 渲染 cam00（1600×1199），报告 PSNR、SSIM、L1。最终检查点另与从原始首帧顺序解码到对应帧的正式码流对照：所有中间 manifest 摘要通过，最终原始属性逐值相同，渲染像素完全相同，指标与已保存评测差值为 0。旧 GS 的 SH、logscale、opacity 在两个阶段逐值一致。未测 FPS。

来源实验为 `output/fs_rigid_tree_region8192_long_300f_20261012/20261012T030052_1bbab8`，训练源码提交 `057ac3c9d47ad66ef5988235d3fad245d04d8c6e`。检查点名称为 `rigid-ckpt-{199,249}-{1200,1400}.pth`，完整来源 SHA、阶段语义和结果保存在 `output/rigid_tree_diagnostics_20261012/stage_quality_199_249_region8192_long.json`。

诊断脚本：`/tmp/my4dgs_rigid_stage_quality_199_249.py`。执行前通过 CPU 预检，并等待完整 300 帧 benchmark 正常结束后才启动 GPU。命令：

```bash
OMP_NUM_THREADS=1 python /tmp/my4dgs_rigid_stage_quality_199_249.py --execute
```

日志：`/tmp/my4dgs_stage_quality_199_249.log`。执行正常退出，历史训练产物未修改。
