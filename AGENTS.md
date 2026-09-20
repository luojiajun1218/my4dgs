项目为基于 ReCon-GS 的 GradHet v2 原型，使用 conda 环境 ReConGS。
数据通过软链接访问，不复制大型数据。
output 中既有实验目录及 references/official_full3_scratch6 为指向原目录的软链接；references/original_backup 是独立副本，历史结果仅用于读取，新实验使用新的 experiment_name。
内部 pipeline 名称保留 ReCon-GS，以兼容 libgs 动态加载。

# 仓库贡献指南

## 项目结构与模块组织

`main.py` 是训练入口。GradHet v2 与 ReCon-GS 的核心代码位于 `pipeline/ReCon-GS/`；`libgs` 会动态加载此名称，因此不得重命名该目录。实验配置放在 `config/`，数值回归检查放在 `tests/`，可复用的启动及结果汇总脚本放在 `scripts/`，方法说明和来源记录放在 `docs/`。`data/`、`output/` 与 `references/` 属于本地数据或实验产物，其中可能包含软链接；不要随意复制大型数据或修改链接指向的原文件。

## 构建、测试与开发命令

执行命令前先激活现有环境：

```bash
conda activate ReConGS
python tests/test_nested_prototype.py
bash scripts/run_prototype.sh gradhet
python main.py --pipeline=ReCon-GS --config=config/prototype_gradhet_scratch6.yaml
```

测试脚本无需加载渲染器，可检查层级采样、权重混合、梯度诊断、训练调度和 checkpoint 恢复。`run_prototype.sh` 运行共享首帧的原型并汇总结果。本项目没有单独的构建步骤。

## 编码风格与命名约定

Python 使用四空格缩进并遵循 PEP 8：函数、变量和模块采用 `snake_case`，类采用 `PascalCase`。在接口不直观时补充类型标注，并明确张量的形状、设备和数据类型。遵循相邻代码的 PyTorch 写法，实验参数写入 YAML，不要硬编码。仓库目前未配置统一格式化或 lint 工具，避免无关的格式调整。

## 测试规范

涉及数值计算或层级结构的改动，应在 `tests/test_nested_prototype.py` 中增加确定性的 `test_*` 函数。随机测试必须设置种子；浮点比较使用 `torch.testing.assert_close`，并明确容差。启动耗时训练前先运行完整测试。训练改动还应使用新的 `experiment_name`，记录配置、输出路径和汇总指标，禁止覆盖历史结果。

## 提交与拉取请求规范

当前快照不含 Git 历史，无法归纳既有提交格式。建议使用简短、祈使式且可带范围的标题，例如 `deformation: fix fine residual blending`，并保证每个提交聚焦单一改动。拉取请求应说明方法变化、列出已执行命令和受影响配置，并报告相关 PSNR 或诊断指标差异。适用时关联 issue；视觉变化需附曲线图或渲染样例。

## 数据与实验安全

不要提交数据集、checkpoint、生成结果或缓存文件。保留现有软链接，并在文档中注明依赖的外部路径。历史产物仅供读取；每次新实验都应创建独立目录。
