# N3DV cook_spinach：自适应层级 storage loss 为零，300 时间点

## 思路与设置

对照已有自适应 baseline；每帧 200 步，保留自适应 gate、首帧、相机采样、量化和运动残差约束。将运动码率权重 `rate_loss_weight` 从 0.008 设为 0，同时将逐点透明度、逐点属性和区域属性的存储 L1 权重设为 0；依赖字节价格的近端筛除也随之关闭。运动残差约束 `relative_motion_loss_weight=1.0` 保留。训练实现未改，只使用独立实验配置。

## 结果与分析

299 个动态帧测试均值：PSNR 33.4650 dB、SSIM 0.95682、LPIPS 0.13419；完整动态包 681.672 KB/帧；训练 18.78 秒/帧；活动区域 3779.7。相对 baseline，PSNR -0.1402 dB、存储 +643.000 KB/帧。实际包组成（KB/帧）：motion 36.417、opacity_update 37.338、full_attribute_update 584.392、regional_full_attributes 21.704、manifest 1.822。

关闭的是一整组存储相关约束，不能单独判断某个分项的作用；变化可能来自参数稀疏性、区域选择、量化和优化共同作用。旧 baseline 早于工程重构，且未保存完整历史工作区，无法证明非存储相关代码逐行一致；单次实验不等于因果证明或跨场景结论。

对照配置、逐帧相机顺序、关键步骤参数冻结、存储损失和近端价格、实际字节、训练/解码图像与从共享首帧顺序解码摘要已核验：`output/experiments/n3dv_storage_zero_20261117_metadata/cook_spinach_audit.json`。完整运行：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/n3dv_storage_zero_cook_spinach_full300_20261117/20261117T202105_be42e2`。KB 为 1000 字节，动态包含 manifest，不含首帧。

RTX 3090 Ti 全序列已解码纯渲染 257.99 FPS、完整播放 7.45 FPS；详见 `docs/experiment/20261117_n3dv_storage_zero_speed_full300.md`。
