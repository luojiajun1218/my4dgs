# my4dgs

基于 ReCon-GS 的分层运动压缩原型。目前保留 GradHet 四层 gate、规则选择、可学习选择和量化残差运动。内部 pipeline 名称保持 `ReCon-GS`，供 libgs 动态加载。

## 实验与结果

[2026-09-30 实验总表](docs/experiments/20260930/README.md)核对了 14 个本地实验目录和 1 个外部 ReCon-GS 参考。报告包括实际完成帧数、训练预算、同帧 PSNR / SSIM、真实增量码流、计时、磁盘占用和配置差异。逐帧数值、配置快照和来源 SHA256 一起保存在该目录。

当前 299 个后续帧的测试平均 PSNR：ReCon-GS 参考 33.942 dB，GradHet 四层 33.783 dB，motion-only 33.565 dB，有新增 GS 的残差版本为 32.795 / 32.763 dB。motion-only 的平均 SSIM 高于参考，但 PSNR 低于参考。所有主均值都排除首帧。

`v1–v6` 是 S²GS 分支历史实验，首帧独立训练，SH degree 和图像尺寸与近期实验不同。learned-selective、`v5`、`v6` 未完成整段；两个空目录没有可核验结果。它们在报告中单独列出。[更早的历史记录](docs/experiments/history_20260919_20.md)来自初始 Git 提交，原始输出已不可访问，未混入本次重算。

## 代码与配置

- `main.py`：训练入口。
- `pipeline/ReCon-GS/`：模型、训练、运动编码与诊断实现。
- `config/flame_steak_motion_only.yaml`：当前 motion-only 配置，200 步运动、关闭新增 GS、300 总帧。
- `config/archive/`：历史规则选择、可学习选择和量化残差配置。
- `config/diagnostics/`：选择与残差可视化配置。
- `config/templates/`：其他数据集的配置模板。
- `tests/test_nested_prototype.py`：层级、运动、调度、codec 与 checkpoint 的数值检查。
- `scripts/summarize_experiments.py`：只读取已有结果的统计入口。

归档配置与运行快照有历史路径差异。报告的 `configs/` 是运行时原始快照；`config/archive/` 是整理后的配置。重跑前需要核对当前实现、依赖和参数，并设置新的 `experiment_name`。

## 检查与运行

使用现有 `ReConGS` conda 环境。训练由用户手动启动；先讨论并确定实验参数。以下训练命令使用新的名称，依赖共享首帧 checkpoint：

```bash
conda activate ReConGS
python tests/test_nested_prototype.py
python tests/test_experiment_inventory.py
python main.py --pipeline=ReCon-GS --config=config/flame_steak_motion_only.yaml experiment_name=fs_motion_only_new_run
```

只重算已有实验的统计，不训练、不渲染、不加载 checkpoint：

```bash
python scripts/summarize_experiments.py \
  --reference /home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf \
  --destination docs/experiments/new_inventory \
  --plots
```

## 本地依赖与产物

`data/colmap4d/flame_steak` 软链接指向 `/home/gamer-02/Documents/DeepLearning/ReCon-GS/data/colmap4d/flame_steak`，其帧和 COLMAP 文件再链接到 `/home/gamer-02/Documents/DeepLearning/S2GS/data/standard_flame_steak_300`。近期实验保存的测试图为 1600×1199；源图为 2689×2016。

共享首帧来自 `/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf/ckpt-0-15000.pth`。训练还依赖环境中已有的 libgs、CUDA 渲染扩展及 torchac。本地 `submodules/` 目前只有构建产物和包元数据，不包含可安装的源码，不上传 Git。

`data/`、`output/`、checkpoint、码流、渲染图和缓存保留在本地。Git 保存代码、配置、文档、汇总曲线和紧凑统计记录。历史实验只读，禁止覆盖或修改软链接目标。
