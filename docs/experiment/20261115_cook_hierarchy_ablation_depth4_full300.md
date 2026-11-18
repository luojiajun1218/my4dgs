# Cook Spinach：depth4 层级消融，300 时间点

## 设置与改动

同一共享首帧和五层候选空间树，从首帧独立递推训练 299 个动态帧；每帧 200 步，前 100 步训练共享平移（自适应组同时训练 gate），后 100 步训练区域运动、逐点残差和属性。前 100 步区域旋转按原方案冻结。gate 在第 100 步后冻结；固定组每帧的 200 步中始终处于指定深度，下一帧按上一帧解码位置重新路由。各组的相机顺序逐帧核对一致。

使用独立逐帧方案 `pipeline/per_frame_hierarchy_ablation.py`，选择策略为 `depth4`。训练使用原优化、属性与编码流程；额外记录冻结状态及相机轨迹。结果 `/home/gamer-02/Documents/DeepLearning/my4dgs/output/experiments/cook_hierarchy_depth4_full300_20261115_retry1/20261115T091022_aaa7b5`。原自适应 baseline 会在第 151–180 步重新训练 gate；本次额外重跑了第 100 步后冻结 gate 的辅助对照，主对比按用户要求使用原 baseline。短实验为 30 时间点（29 个动态帧），经独立顺序解码和状态摘要校验后运行完整实验。

## 结果

测试视角动态帧均值：PSNR 33.4900 dB，SSIM 0.95644，LPIPS 0.13786；实际完整动态包 59.481 KB/帧，活动区域 3413.8，记录训练时间 14.09 秒/帧。KB 为 1000 字节，动态包包括各文件与 manifest、不含共享首帧。主要码流分量均值（KB/帧）：motion 32.858、opacity_update 3.938、full_attribute_update 17.507、regional_full_attributes 3.107、manifest 2.071。

短实验：PSNR 33.7536 dB、62.563 KB/帧。完整实验每帧一个训练视角的训练态与解码渲染检查、包大小检查、顺序解码摘要和冻结/相机轨迹核验通过；逐帧证据与组间对比见 `output/experiments/cook_hierarchy_ablation_20261115_metadata/full_audit.json`。


相对现有自适应 baseline，本组 PSNR -0.1152 dB、动态存储 +20.809 KB/帧；运动包差 +17.406 KB/帧，区域属性包差 +2.682 KB/帧。区域数量、运动和属性包同时变化，可能与共享粒度及量化拟合有关；本次不能证明唯一原因。

仅为单场景、单随机种子的结果；画质和码率差异不能单凭总指标归因于自适应层级，也不能推广为跨场景结论。
