# 区域共享 DC + 稀疏逐点 DC/尺度：连续 30 帧对照

## 方法与范围

Sear Steak 和 Coffee Martini 分别从各自同一第 200 帧已验证检查点续跑 201–230。每帧先执行原 200 步共享运动、gate、逐点运动和透明度，再固定这些参数；区域共享 DC 与训练视角误差选取的最多 1,000 点逐点 DC 残差、log-scale 更新额外训练 100 步。运动 gate 的图像 STE 关闭。每步一个训练视角；选区额外汇总全部训练相机，cam00 只报告。Coffee 使用 16 训练相机、排除 cam13。训练画质指标仅为固定训练相机 cam06。各分支从上一帧的真实解码状态递推。

FUA2/FUA3 稀疏码流包含 DC 和尺度实际非零行的并集索引；报告包含真实 motion、opacity、区域 DC、FUA 和 manifest 全字节。第 0 帧参考及 1–200 帧动态前缀单独报告。所有 201–230 帧连同之前的前缀均已从第 0 帧独立重放并核对状态摘要。两分支由于训练路径和 GPU 数值非确定性，不能视为第 201 帧后使用相同前驱。

## 结果

| 场景 / 方案 | test PSNR | test SSIM | cam06 PSNR | cam06 SSIM | 动态字节/帧 | 训练秒/帧 |
|---|---:|---:|---:|---:|---:|---:|
| sear / baseline | 32.6601 | 0.963783 | 33.7620 | 0.965419 | 31004.5 | 12.578 |
| sear / shared_dc_scale | 33.4878 | 0.964433 | 34.8730 | 0.966146 | 34918.2 | 16.718 |
| coffee / baseline | 28.9383 | 0.924272 | 32.2671 | 0.943930 | 44806.9 | 13.600 |
| coffee / shared_dc_scale | 29.0355 | 0.924690 | 32.4823 | 0.944358 | 50364.6 | 17.891 |

test 和训练指标都按帧算术平均。该片段并非全 300 帧结果，不能代表 N3DV、MeetRoom 或长期漂移的整体表现。

## 实际字节、时间与检查

- sear: test 差值均值 +0.8277 dB，范围 +0.0008 到 +1.3999，30/30 帧为正；动态总字节 930,136 → 1,047,547，差额 117,411 B。共享首帧参考 11,299,285 B；共同 1–200 动态前缀 6,529,789 B。
  联合方案每帧分包均值（字节）：frame 2541.8, fullattr 2657.3, motion 25672.7, opacity 3919.2, shdc 127.2；实际非零索引并集均值 429.6，最大不超过 1,000；选区平均 0.226 秒；逐帧前向/解码最大像素差 0.0e+00。
  基线目录 `/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/shared_dc_sparse_scale_20261106/runs/sear_baseline_full30/20261106T232636_8efd43`；联合目录 `/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/shared_dc_sparse_scale_20261106/runs/sear_shared_dc_scale_fixed30/20261106T235830_f9af93`。
- coffee: test 差值均值 +0.0972 dB，范围 +0.0136 到 +0.1733，30/30 帧为正；动态总字节 1,344,208 → 1,510,937，差额 166,729 B。共享首帧参考 12,139,245 B；共同 1–200 动态前缀 9,529,529 B。
  联合方案每帧分包均值（字节）：frame 2541.5, fullattr 3442.0, motion 39519.5, opacity 4734.0, shdc 127.6；实际非零索引并集均值 633.6，最大不超过 1,000；选区平均 0.228 秒；逐帧前向/解码最大像素差 0.0e+00。
  基线目录 `/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/shared_dc_sparse_scale_20261106/runs/coffee_baseline_full30/20261107T000825_778d7a`；联合目录 `/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/shared_dc_sparse_scale_20261106/runs/coffee_shared_dc_scale_fixed30/20261107T001545_e1c1cf`。

Coffee 联合方案有 17/30 帧超过 50,000 字节；目标按完整序列平均而非逐帧硬上限，此处 30 帧平均 50,365 B。30 帧均值不能代替完整序列平均。

## 不一致修复与局限

首次联合轨迹在 Sear 第 208 帧因前向与解码图像相差超过 1e-5 停止；后续独立捕获在另一条轨迹的第 203 帧重现 0.0001788 的最大误差。渲染器单变量交叉验证：同输入下两个后端结果相同；训练直接量化 alpha 和解码 logit 后 sigmoid 的微小舍入差使一个高斯跨越光栅化贡献阈值。修复仅作用于 200 步后透明度冻结的属性阶段：渲染解码器对应的透明度并检查缓存未变化；未更改码流格式，也未改变前 200 步。回归包含历史透明度码流、CUDA 和两帧 v3 组合。

已从原失败运行的真实第 207 帧解码状态做第 208 帧诊断，属性及渲染逐位一致，且独立解码通过；因为原运行未保存该帧的完整训练检查点，诊断的运动预热与随机状态来自另一条轨迹，不能称为原训练轨迹逐位复现。失败日志及诊断均保留。

本次没有 MeetRoom 训练结果；现有 discussion、trimming、vrheadset 适配数据缺训练好的首帧参考，官方第四场景 stepin 未取得。仍需在固定配置下扩大 N3DV 及 MeetRoom 跨场景验证，且不得将本结果归因于运动 gate 的因果收益。

逐帧完整汇总：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/shared_dc_sparse_scale_20261106/sear_verified_30frames.json`、`/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/shared_dc_sparse_scale_20261106/coffee_verified_30frames.json`。
