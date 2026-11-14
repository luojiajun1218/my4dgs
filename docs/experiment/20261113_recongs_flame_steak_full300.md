# ReCon-GS：flame_steak 300 时间点复现

## 方法与设置

区域多层 Anchor 变形与增量高斯；使用共享首帧，仅将 checkpoint 中张量的设备标记转换为 GPU；没有修改训练代码。测试集使用项目当前数据划分，跳过首帧训练。
运行目录 `/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/n3dv_reproduction_flame_steak_shared_first_300_20261113/20261113T231737_a9b508`。首次时间点和 299 个动态帧；不对图片或共享数据做复制。

## 结果与限制

动态帧测试均值：PSNR 33.1460 dB，SSIM 0.9599；本次未启用 LPIPS。

指标口径：TensorBoard test validation, 299 dynamic frames。原项目未保存完整逐帧变形码流，增量高斯 PLY 不包含全部动态信息，不能以其大小作为完整动态码率。训练日志的 FPS 未同步 GPU，不能作为严格渲染 FPS。

此结果来自单次运行；不同项目首帧、相机读取和表示不同，不能只凭总指标推断因果机制或宣称超过 SOTA。
