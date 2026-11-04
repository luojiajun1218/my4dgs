# 配置入口

当前 Flame Steak 候选为 [flame_steak_gate_owner_repair_10_20261104.yaml](flame_steak_gate_owner_repair_10_20261104.yaml)：10 个时间点包含共享首帧和 9 个动态帧，区域误差阈值为 0.03。
0.02 阈值短试也已完成，可用当前配置覆盖 `module.birth.regional_color_l1_threshold=0.02` 并指定新的 `experiment_name` 复现。相同半图上的 9 个动态帧：0.03 为 34.2534 dB / 39.406 KB，0.02 为 34.2597 dB / 42.418 KB（KB = 1000 B）。当前保留 0.03；两轮在区域选择前已有数值差异，不能把微小画质差异归因于阈值。

该候选保持 5 层候选区域，每个动态帧先训练 200 步共享运动与 gate，再用 100 步进行已有 GS 的区域属性修复；每步使用 4 个训练视角，本次固定 GS 数量。
数据使用 `/mnt/SSD500GB/my4dgs_data/flame_steak_standard300_frame_view_20261104/dataset` 的标准半图，`resolution: -1`，预期图像大小为 1345 × 1008。
该私有数据入口目前只链接了 10 个时间点；后续长序列需补齐链接。数据和首帧资产通过已有路径访问，不复制；后续新实验应使用新的 `experiment_name`。

```bash
conda activate ReConGS
python main.py --config=config/flame_steak_gate_owner_repair_10_20261104.yaml experiment_name=替换为新的实验名
```

## 历史配置和结果

其他 YAML 保留用于追溯，并不都代表当前训练入口。涉及已退役训练分支的实验，应结合对应运行的源码快照、配置和依赖复现。
重要实验记录见 [docs/experiment](../docs/experiment/)；[output/best/index.json](../output/best/index.json) 汇集部分历史结果，并非所有实验或最新最优结果的完整清单。
首帧资产及准备流程见 [首帧处理指南](../docs/first_frame_recipe.md)。

## 清理记录

清理汇总见 `/mnt/SSD500GB/my4dgs_diagnostics/cleanup_20261104/summary.json`：两批共移除 117 个旧短实验检查点，逻辑大小约 9.8 GiB，并归档了 3 份 smoke 配置和 11 个一次性脚本后移除原文件。
被清理的检查点不能再从原路径读取；对应摘要、真实码流、源码快照和首帧资产均保留。完整逐文件记录及归档路径见汇总文件。
