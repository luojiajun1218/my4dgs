# 首帧统一评估：当前13万与0902重要版本

本次只读取现有checkpoint并短渲染，没有训练，也没有改动训练机制或历史output。核心问题是当前动态序列是否继承了较差的首帧，以及用户所说“重要那版约20万首帧”实际是哪一个。

## 统一条件与结果

使用冻结的 `tmp/experiments/residual_tree_v4_short3` 模型/渲染辅助函数，在相同1345×1008原始cam00首帧、相同COLMAP相机、黑背景、SH1下渲染。所有模型均检查原始版本及opacity≥0.0075版本；GT未做曝光、平移或其他配准。

| 模型 | 原始GS | 过滤后GS | 过滤后全RGB浮点PSNR | 过滤后libgs PSNR | 过滤后S2GS 8bit PSNR |
|---|---:|---:|---:|---:|---:|
| 当前frame0 | 132586 | 128963 | 33.18471 | 33.21785 | 33.17322 |
| 0902重要版fullres首帧 | 214996 | 206237 | 33.91817 | 33.95224 | 33.90412 |
| 另一个历史1345模型 | 199719 | 192069 | 32.15062 | 32.17338 | 32.14175 |

0902的重要首帧是 `ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf/ckpt-0-15000.pth`，不是199719那个文件。相同评估条件下它比当前13万高 **0.73346 dB**，全RGB MSE降低约15.5%。这个差距属于首帧本身，不能直接等同于300帧平均收益。

## 指标与S2GS公平性检查

实际代码存在指标口径差异：libgs对CHW输入分别计算三个通道PSNR再平均；S2GS先round到8bit，再对全RGB计算单一MSE。本次同时给出两种历史指标和全RGB浮点指标。当前33.21785准确复现既有33.217857记录。

S2GS flame_steak记录首帧35.15dB、59745个_offset，SH2。其Frame index=1对应原始0000.png；cam00留出，其余20相机训练。图像路径与当前half数据最终指向同一个n3dv数据源，图像1345×1008，cam00外参最大差约1e-16。fx相同730.7152678，但我们fy=727.6929257，S2GS由poses_bounds强制fy=fx=730.7152678，存在0.415%差异。两者黑背景一致。

S2GS flame_steak目录仅548KB，无checkpoint、PLY或首帧PNG，log_ply/log_compressed/log_images均关闭，因此无法对其35.15执行真正相同相机重渲染，也无法定位其误差。现存Num points来自_offset.shape[0]，不能解释为根节点数或实际可见点数；Active gates=0是更新门，不是零个渲染点。当地generate_lod_gaussians可见性为空时使用全部anchor并返回全真mask，但没有模型不能恢复各层及屏幕active数量。

cook_spinach存在首帧PNG：test/gt/cam00/0001.png实际链接原始0000.png；直接统一全RGB 8bit MSE重算 **33.8309325 dB**，吻合日志33.83。它没有checkpoint，本仓库也没有已确认兼容的该场景首帧，故未进行跨方法质量差比较。

## 误差与来源

已查看同色标残差图：当前13万与重要版的差别主要包含窗帘上部和右墙等稳定背景的低频误差，物体边缘、台面及手部仍都有残差。199719模型的人体及右墙残差明显更强。此处是图像空间观察，不能据此断言具体3D机制。

三个模型都SH1、15000步、40000点输入种子。两份旧模型保存配置noise_scale=0.01、densify threshold=0.0002；当前13万记录threshold=0.00033，且训练明确不使用共享颜色。旧199719配置resolution=1345；重要版配置resolution=-1，历史默认宽图会自动缩到1600宽，0902动态实验也是1600×1199；旧相机JSON保留2689×2016原始元数据，不能拿它直接当实际训练tensor尺寸。当前13万用已预缩小1345×1008图像。采样随机状态、当时全部代码版本与预处理差异未完全锁定，因此本评估不把提升归因于点数；199719反而更差也直接否定“更多点必然更好”。

建议下一步先恢复重要版首帧作为候选，在新实验中验证动态继承表现；先核对输入尺寸与其余配置，不能覆盖现有历史输出。S2GS的剩余差距还混有SH2、训练方法与fy口径，缺少模型不能拆分贡献。

## 复现产物

- 脚本：`tmp/first_frame_alignment_20261026/evaluate.py`
- 数值：`tmp/first_frame_alignment_20261026/evaluation.json`
- checkpoint SHA256、旧相机及种子PLY头：`tmp/first_frame_alignment_20261026/provenance.json`
- GT/渲染/固定0..0.10 RGB RMSE比较：`tmp/first_frame_alignment_20261026/comparison.png`
- 固定5×4区域PNG误差：`tmp/first_frame_alignment_20261026/regional_png_errors.json`

图像残差由存储8bit PNG生成，指标表来自浮点渲染；不能把图像量化误差当浮点数值。脚本通过ReConGS环境运行，历史产物未写入Git。
