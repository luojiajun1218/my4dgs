# 保留的实验配置

`best_*.yaml` 根据各场景所保留结果的运行时配置整理；原始配置仍保存在对应结果目录的 `config.yaml`。Coffee 配置已更新为当前完整序列方案；其他场景的历史配置对应当时的代码，使用前应核对兼容性并设置新的 `experiment_name`。

各场景的结果、首帧、前缀和统计见 [output/best/index.json](../output/best/index.json)，也可从 `output/best/` 进入结果目录。动态帧平均字节包含每帧 manifest，共享首帧另计。

首帧模型及对应区域布局集中在 `/home/gamer-02/Documents/DeepLearning/data/n3dv/first_frame/`；迁移后的实验和场景数据分别在 `output/experiments/` 与 `output/datasets/`。

Coffee Martini 保留完整 300 帧、Cut Roasted Beef 保留 88 帧；其余保留完整 300 帧。短序列结果不能作为完整 300 帧结论。有关方法和历史结果的说明见 [实验记录](../docs/experiment/)；其中已清理实验的旧路径可能不再存在。

Cut 的结果包含 frame0–87，但最新的完整 checkpoint 是 frame59。Cook Spinach 和 Cut Roasted Beef 的历史配置包含当前代码已移除的旧属性策略；其他非 Coffee 的历史配置也应先检查当前训练代码兼容性。

运行前使用 `conda activate ReConGS`；数据继续通过现有软链接读取。

当前训练固定每步使用一个随机训练视角。历史配置中的 `motion_views_per_step` 和 `birth_views_per_step` 会在启动训练时统一设为 1，日志会提示，实际运行配置也会记录为 1；本目录的历史 YAML 不被改写。不同视角数的旧动态 checkpoint 不能直接续训，以免把不同训练方式混在同一段序列里；新实验应从保留的共享首帧启动，并设置新的 `experiment_name`。

Coffee 当前推荐的完整配置为 `coffee_early_joint_rmse08_prox10_1pct_300frames_20261109.yaml`：共享首帧加 299 个动态帧，16 万点、depth 5 候选层级，第 101–104 步依次用四个训练视角探测，最多选择 1% 的点，后续联合训练运动和所选点属性；运动与纠偏阶段使用 80% RMSE 混合损失。固定码率 loss 权重为 0.008，逐点运动近端收缩权重为 1.0，不使用码率反馈。本次完整序列动态帧平均为 29.802 dB、41.48 KB/帧，结果和口径见 [实验记录](../docs/experiment/20261109_coffee_rmse08_prox10_full300.md)。

如优先追求 PSNR，`coffee_early_joint_rmse08_1pct_300frames_20261109.yaml` 使用相同训练路径和 0.8 收缩，完整序列为 29.810 dB、46.50 KB/动态帧。两者均不包含首帧存储；运行时应使用新的 `experiment_name`。如需进行 30 个动态帧短实验，可基于完整配置另建配置，将 `num_frames` 设为 31（包含共享首帧），并设置新的 `experiment_name`。

`coffee_motion_only_200step_prox10_300frames_20261109.yaml` 专用于无稀疏属性纠偏消融，保留共享运动、gate、逐点运动残差及透明度更新，每帧训练 200 步；详见 [消融记录](../docs/experiment/20261109_coffee_motion_only_200step_ablation_full300.md)，不作为完整方法配置。

Coffee 的最佳完整序列见 `output/best/coffee_martini`，200 步消融及较高 PSNR 的完整序列见 `output/comparison/`。2026-11-09 清理的 Coffee 短测逐帧摘要、运行配置和源码快照保存在 `docs/experiment/assets/20261109_coffee_short_runs_records.tar.gz`；短测的图像、码流与 checkpoint 未保留。
