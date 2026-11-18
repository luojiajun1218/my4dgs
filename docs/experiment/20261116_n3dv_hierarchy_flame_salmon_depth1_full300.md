# N3DV flame_salmon：depth1 固定层级，300 时间点

## 方法与代码

使用五层候选树，从共享首帧独立训练；每帧按上一解码位置路由，训练期间固定区域归属。每帧 200 步，前 100 步训练区域平移，后 100 步训练区域运动、逐点残差与属性；固定组 gate 冻结。旧 baseline 在第 151–180 步重新训练 gate，且旧版完整源码快照缺失，不能确认其他实现逐行一致。

仅扩展 `pipeline/per_frame_hierarchy_ablation.py` 支持选择 depth1；优化、码流和数据配置沿用此前实验。结果目录：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/n3dv_hierarchy_flame_salmon_depth1_full300_20261116/20261116T201235_9b5456`。

## 结果与分析

299 动态帧均值：PSNR 29.6239 dB、SSIM 0.92123、LPIPS 0.14832；完整动态包 55.575 KB/帧、活动区域 8.0、训练 17.41 秒/帧。包组成（KB/帧）：motion 32.068、opacity_update 3.820、full_attribute_update 17.497、regional_full_attributes 0.120、manifest 2.070。

相对已有自适应 baseline，PSNR -0.8077 dB，存储 +18.402 KB/帧。共享粒度、内容、量化和优化均可能影响画质与码率；该单次实验不能单独确立原因。KB 按 1000 字节，含完整动态包和 manifest，不含首帧。

逐帧相机、200 步、冻结、指定层归属、训练与解码图像、包字节和完整顺序解码摘要见 `output/experiments/n3dv_hierarchy_1_5_20261116_metadata/flame_salmon_full_audit.json`。RTX 3090 Ti、cam00 全序列测速：已解码纯渲染 240.39 FPS；完整播放 13.52 FPS，包含读包、CPU 解码与校验、GPU 更新和渲染。测速范围与原始结果见 `docs/experiment/20261116_n3dv_hierarchy_1_5_speed_full300.md`。
