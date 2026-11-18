# S2GS：cook_spinach 300 时间点复现

## 方法与设置

S2GS 原生首帧及层级门控稀疏残差；按原生结构重新训练首帧，未使用共享模型权重。修复轨迹图片目录在首帧未创建时导致后续写入失败的问题，不改变训练方法。
运行目录 `/home/gamer-02/Documents/DeepLearning/S2GS/output/n3dv_reproduction_cook_spinach_native_first_300_20261113`。首次时间点和 299 个动态帧；不对图片或共享数据做复制。

## 结果与限制

动态帧测试均值：PSNR 33.4630 dB，SSIM 0.9574，LPIPS 0.1319。

指标口径：cam00 saved 8-bit PNG frames 0002–0300。原项目未实际保存逐帧压缩码流，训练 Size 为估算且数值四舍五入，不能作为实际磁盘码率。没有独立解码回放或完整播放 FPS 测量。PNG 统计存在 8 位量化。

此结果来自单次运行；不同项目首帧、相机读取和表示不同，不能只凭总指标推断因果机制或宣称超过 SOTA。
