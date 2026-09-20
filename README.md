# my4dgs

基于 ReCon-GS 的 GradHet v2 研究原型：先训练 Coarse + Middle，再根据区域内三维位移梯度方差，选择性开启嵌套 Fine 残差。

## 目录

- `pipeline/ReCon-GS/`：原型源码；内部入口名称保留以兼容 libgs。
- `config/`：`prototype.yaml` 复用共享首帧训练两帧；`prototype_gradhet_scratch6.yaml` 从零训练六帧。
- `scripts/`：运行、结果统计及首帧审计脚本。
- `tests/`：原型数值检查。
- `docs/prototype.md`：方法与实验说明。
- `docs/upstream_README.md`：原始说明及上游出处；LICENSE 保留原许可证。
- `data/colmap4d/coffee_martini`：公共数据软链接。
- `output/00_init`、`output/prototype_gradhet_v2_scratch6_20260916`：原项目的历史实验软链接；新实验仍写入本项目 output。
- `references/`：官方六帧对照、原始备份资料副本及较早版本结果表。
- `docs/source_manifest.json`：复制时逐文件 SHA-256 校验清单，README 原文另存于 docs。

## 运行

```bash
conda activate ReConGS
cd /home/gamer-02/Documents/DeepLearning/my4dgs
python tests/test_nested_prototype.py
# 共享首帧的 GradHet 实验
bash scripts/run_prototype.sh gradhet
# 或从零训练六帧
python main.py --pipeline=ReCon-GS --config=config/prototype_gradhet_scratch6.yaml
```

复用现有 ReConGS 环境及 libgs 依赖。项目目录名为 my4dgs，内部 pipeline 注册名仍为 ReCon-GS。

## 已有结果与限制

GradHet v2 后续五帧平均 PSNR 为 27.6613 dB，官方独立从零实验为 27.5468 dB；两次首帧分别为 27.3887 和 27.2155 dB，起点不同，不能据此归因于 GradHet。较早版本汇总不属于 v2 消融。可靠对照应共享首帧 checkpoint，并匹配随机性、训练安排及 Fine 预算。

本项目从 2026-09-17 备份独立复制；官方项目保留；原备份资料已独立归档到 references/original_backup，原 ReCon-GS-backups 目录已删除。历史日志和配置中的旧路径用于溯源。数据与历史结果使用软链接，依赖原目录；修改链接中的文件会修改原文件。
