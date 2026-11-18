# N3DV flame_steak：depth2 固定层级，300 时间点

## 方法与设置

从与已有 baseline 相同的共享首帧独立递推训练 299 个动态帧，共用五层候选空间树。每帧 200 步，前 100 步固定区域归属、训练共享平移，区域旋转沿用原方案冻结；后 100 步训练区域运动、逐点运动残差和属性，固定 gate。每帧开始按上一帧解码位置路由，200 步内归属不变。相机采样与 baseline 按 200 步逐帧比较。

使用独立方案 `pipeline/per_frame_hierarchy_ablation.py`，仅选择固定深度 `depth2`，保留既有优化和码流实现；记录额外冻结和相机诊断。run：`/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/n3dv_hierarchy_flame_steak_depth2_full300_20261115/20261115T225121_8a0370`。按用户要求跳过剩余短实验，直接独立训练完整序列；完整结果仍逐帧核验。

## 结果与分析

动态帧测试均值：PSNR 33.7050 dB，SSIM 0.96241，LPIPS 0.12968；完整动态包 40.265 KB/帧（1000 B/KB，不含首帧），活动区域 64.0，项目训练时间 14.38 秒/帧。文件分量均值 KB/帧：motion 15.784、opacity_update 3.851、full_attribute_update 18.273、regional_full_attributes 0.286、manifest 2.071。

相对现有 baseline，PSNR -0.4650 dB、存储 +3.774 KB/帧、活动区域 -123.4；运动包差 +1.647 KB/帧，区域属性包差 -0.134 KB/帧。该分解反映本次不同共享粒度下的实际字节分布；内容、量化和优化也可能影响结果，不能单独确定原因。

逐帧数据、训练/解码渲染抽查、完整顺序回放状态摘要、相机与冻结核验在 `output/experiments/n3dv_hierarchy_ablation_20261115_metadata/flame_steak_full_audit.json`。baseline 的 gate 在第 151–180 步重新训练，固定组不重新选择；其训练时 Git 版本早于工程重构且缺少当时完整工作区源码快照。单场景结果不能说明跨场景结论。
