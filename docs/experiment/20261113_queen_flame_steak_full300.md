# queen：flame_steak 300 时间点复现

## 方法与设置

QUEEN 原生首帧及量化逐帧残差；按原生结构重新训练首帧，未使用共享模型权重。
运行目录 `/home/gamer-02/Documents/DeepLearning/queen/output/n3dv_reproduction_flame_steak_native_first_300_20261113`。首次时间点和 299 个动态帧；不对图片或共享数据做复制。

## 结果与限制

动态帧测试均值：PSNR 34.4521 dB，SSIM 0.9675，LPIPS 0.1116。

动态压缩文件均值 15069.850 KB/帧（1000 B/KB）；首帧 PLY 37.011 MB。

指标口径：cam00 saved 8-bit PNG frames 0002–0300。码率只统计实际保存的动态 point_cloud.pkl 文件；首帧 PLY 单列。没有独立解码回放或完整播放 FPS 测量。PNG 统计存在 8 位量化。

此结果来自单次运行；不同项目首帧、相机读取和表示不同，不能只凭总指标推断因果机制或宣称超过 SOTA。

训练日志中的模型大小是估算值，不能替代上述文件大小。抽查第 100 个时间点的 `point_cloud.pkl`，文件为 15,021,427 B；重新序列化检查显示其中 `decoder_state_dict` 约 9.74 MB、`latents` 约 5.28 MB，后者包括约 2.66 MB 的 `flow` 张量。编码实现还将完整的 `decoded_att` 写入部分 decoder 状态。这解释了该样本文件远大于日志估算，但不应把单帧字节构成当作全序列逐项均值。
