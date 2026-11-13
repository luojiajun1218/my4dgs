# 训练配置

此目录只保留动态场景的 baseline 和 best 配置：

| 配置 | 用途 |
| --- | --- |
| `n3dv_baseline_<场景>_300frames_20261113.yaml` | 六个 N3DV 场景各一份 300 时间点 baseline 配置。 |
| `best_coffee_martini.yaml` | Coffee 完整 300 时间点全属性方案；对应 `output/best/coffee_martini`。 |

Sear Steak 的 best 结果使用对应的 baseline 配置，保存在 `output/best/sear_steak`，因此没有单独的 best YAML。其他场景的历史 best 原始结果不在 `output/best` 中，不把 baseline 标记为这些场景的 best。

运行前使用 `conda activate ReConGS`，并为新实验设置新的 `experiment_name`，不要覆盖历史结果。场景数据入口位于 `data/n3dv/<场景>/`，共享首帧和区域参考位于 `/home/gamer-02/Documents/DeepLearning/data/n3dv/first_frame/`。

六个场景 baseline 的完整结果副本保存在 `/mnt/SSD500GB/n3dv/baseline/`，各场景实验记录位于该目录的 `reports/`。首帧训练入口仍为 `scripts/train_first_frame.py`，需要训练时自行准备静态训练配置。

旧版实验配置已清理；历史实验文档和 `output/best/index.json` 中的旧配置路径或旧运行路径可能不再存在。对应现存实验目录里的 `config.yaml` 是当时运行记录，不应当作现行路径直接使用。
