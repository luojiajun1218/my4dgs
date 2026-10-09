项目为基于 ReCon-GS 的分层运动压缩原型，使用 conda 环境 ReConGS。
数据通过软链接访问，不复制大型数据。
本地 data/colmap4d/flame_steak 指向 /home/gamer-02/Documents/DeepLearning/ReCon-GS/data/colmap4d/flame_steak；其帧图像与 COLMAP 文件链接到 /home/gamer-02/Documents/DeepLearning/S2GS/data/standard_flame_steak_300。当前 motion-only 配置使用 resolution: -1、num_frames: 300，并加载共享首帧；libgs 会自动缩小宽图。
output 中历史结果仅用于读取；目录可能包含软链接，不修改链接目标。新实验使用新的 experiment_name。Git 不保存原始训练产物。

# 特别须知
实验汇总：最理想的做法是每次做完实验后，在docs/experiment/文件夹下面 写一个对应的md文档（别乱写文档，做完实验 再写文档），
文档中记录这次实验的核心idea，以及对应的代码改动（简要版的，不需要特别详细，可以渐进式披露），运行的结果分析，以及可能造成这种结果的原因。
相当于是一次实验记录文档，不然有些实验 我也记不清楚他们是干嘛的了。

我（作者）一开始的想法很简单就是 让改动的地方动，不改动的地方不动。
具体而言 我大致的做法是：
1.自适应分层，这里的gate是个可学习参数，由模型自动判断是不是要开启下一层。
2.storage loss，存储loss，尽量少的gs变化，让需要改变的gs变化
3.atrribute gate（待做，等自适应分层和storage loss成功了在做），比如灯泡这种就只需要改颜色（位置gate可以关闭），刚体只需要改位置（颜色一般不变化）



# 仓库贡献指南

## 项目结构与模块组织

`main.py` 是训练入口。GradHet v2 与 ReCon-GS 的核心代码位于 `pipeline/ReCon-GS/`；`libgs` 会动态加载此名称，因此不得重命名该目录。实验配置放在 `config/`，数值回归检查放在 `tests/`，方法说明和来源记录放在 `docs/`。`data/`、`output/` 与 `references/` 属于本地数据或实验产物，其中可能包含软链接；不要随意复制大型数据或修改链接指向的原文件。

## 构建、测试与开发命令

执行命令前先激活现有环境：

```bash
conda activate ReConGS
python main.py --config=config/xxxxx.yaml
```

测试脚本无需加载渲染器，可检查层级采样、权重混合、梯度诊断、训练调度和 checkpoint 恢复。本项目没有单独的构建步骤。

## 编码风格与命名约定

Python 使用四空格缩进并遵循 PEP 8：函数、变量和模块采用 `snake_case`，类采用 `PascalCase`。在接口不直观时补充类型标注，并明确张量的形状、设备和数据类型。遵循相邻代码的 PyTorch 写法，实验参数写入 YAML，不要硬编码。仓库目前未配置统一格式化或 lint 工具，避免无关的格式调整。

## 与用户协作

- 文档使用短句、通俗说法和具体例子。先说明要解决什么、怎么做；公式和实现细节放在后面，避免写成长篇术语说明。
- 新增或修改的可调超参数统一写入 YAML，并由代码读取。配置中简短说明参数含义；待讨论的候选值不代表已经选定。
- 运行代码前，先和用户讨论并确定本次使用的超参数，再开始运行。代码修改和候选配置可以先准备好。
- 当前实验的训练由用户手动启动；配置准备好后提供命令，除非用户另行明确要求，不自行启动训练。
- 委派代码实现时优先使用用户指定的 GPT-6 luna max，主 agent 负责检查和解释，控制不必要的 token 开销。

## 测试规范

涉及数值计算或层级结构的改动，应在 `tests/test_nested_prototype.py` 中增加确定性的 `test_*` 函数。随机测试必须设置种子；浮点比较使用 `torch.testing.assert_close`，并明确容差。启动耗时训练前先运行完整测试。训练改动还应使用新的 `experiment_name`，记录配置、输出路径和汇总指标，禁止覆盖历史结果。

## 提交与拉取请求规范

使用简短、祈使式且可带范围的标题，例如 `deformation: fix fine residual blending`，并保证每个提交聚焦单一改动。拉取请求应说明方法变化、列出已执行命令和受影响配置，并报告相关 PSNR 或诊断指标差异。适用时关联 issue；视觉变化需附曲线图或渲染样例。

## 数据与实验安全

不要提交数据集、checkpoint、生成结果或缓存文件。保留现有软链接，并在文档中注明依赖的外部路径。历史产物仅供读取；每次新实验都应创建独立目录。
