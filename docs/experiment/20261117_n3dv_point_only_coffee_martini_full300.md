# N3DV coffee_martini：0 层逐点运动消融，300 时间点

## 方法与代码

不学习区域运动、区域 gate 或共享区域属性；第 1–100 步优化逐点平移，第 101–200 步优化逐点运动和属性。使用与对照相同的首帧、相机顺序、量化参数和 200 步。逐点平移有 200 步，其他固定层对照的逐点运动主要在后 100 步更新；关闭共享属性也是差异，不能仅将结果归因于区域运动。

新增 `pipeline/per_frame_point_only.py`，通过 `RECONGS_POINT_ONLY=1` 选择；旧码流仍保留固定空间路由元数据以兼容原解码器，但不传输非零区域运动或区域属性包。该兼容开销计入实际字节。

## 结果与分析

299 个动态帧测试均值：PSNR 29.7006 dB、SSIM 0.92393、LPIPS 0.14405；动态包 39.778 KB/帧、训练 11.53 秒/帧。各包均值 KB/帧：motion 21.153、opacity_update 4.152、full_attribute_update 12.686、manifest 1.788。相对 baseline PSNR -0.0675 dB，动态存储 -15.172 KB/帧；内容、优化步数和属性编码均可能影响差异。

训练与解码图像、200 步相机顺序、逐点与区域参数训练状态、实际字节、零区域运动符号、从共享首帧开始的逐帧 CPU 回放摘要已核验：`output/experiments/n3dv_pointonly_20261117_metadata/coffee_martini_audit.json`。完整运行：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/n3dv_pointonly_coffee_martini_full300_20261117/20261117T111841_3eace9`。KB=1000 B，含动态包和 manifest、不含首帧。旧 baseline 的完整历史工作区缺失，不能证明所有非 ablation 代码逐行相同。

RTX 3090 Ti 全序列已解码纯渲染 187.65 FPS，完整播放 13.93 FPS；口径和逐场景对照见 `docs/experiment/20261117_n3dv_point_only_speed_full300.md`。
