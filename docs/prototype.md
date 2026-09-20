# GradHet 单帧原型

代码实现已完成；2026-09-16 数值检查通过，已完成从零训练首帧与后续五帧的 GradHet 实验。首帧最终测试 PSNR 为 27.3887 dB，后续五帧最终平均为 27.6613 dB。

首帧采用 ReCon 自带的静态 3DGS 训练，不启用任何层级 deformation，因此没有首帧两层或三层之分。此次配置为 `config/prototype_gradhet_scratch6.yaml`：不加载 checkpoint，总共六帧，首帧 15000 步，后续每帧 200 步。数据为 coffee_martini，每帧 17 个训练视角、1 个测试视角；仅运行 GradHet。

本次输出目录为 `output/prototype_gradhet_v2_scratch6_20260916/20260916T215258_f2aa0a`。首帧第 5000、10000、15000 步保存 checkpoint，后续每帧结束保存 checkpoint，便于检查和复用本次从零得到的首帧。

## 训练流程

- 当前帧开始时建立 Middle→Fine 子节点和距离权重。每个 Middle 最多三个 Fine；第一个取最接近父节点 GS 均值的成员点，后续做父节点内部最远点采样，重复坐标不产生重复子节点。
- 第 1–30 步只训练 Coarse + Middle。
- 第 31–50 步继续两层训练，累计可见 GS 的三维位移输出梯度诊断。
- 第 50 次参数更新后，按 `H_before` 选择 Middle。最多激活非空 Middle 的 50%；没有观测、分数为零或不足两个空间子节点的父节点跳过，不强行填满预算。
- 第 51–100 步继续训练，同时激活选中的 Fine。
- 后续保持现有 densification 流程。

## 距离权重与重叠

`fine_overlap_ratio: 0.15` 指相邻两个 Fine 间距的 15%，即边界每侧延伸间距的 7.5%，不表示恰好 15% 的 GS 被共享。边界使用三次 smoothstep 距离权重；通过每个子节点相对其他子节点的平滑边界权重乘积并归一化，处理最多三个子节点的交汇。区域内部权重为单节点控制，重叠带内平滑混合，权重和为 1。

位置、GS 归属和权重在当前帧 Phase 1 内固定，只允许同一 Middle 内的 Fine 共同控制 GS。实际共享 GS 数量及每个 Fine 的共享比例写入统计文件。

## 加权梯度诊断

每次仅使用当前可见且三维梯度有限的 GS；至少两个有效 GS 才计入该 Middle 的一次有效观测。

- `H_before`：Middle 内梯度向量的总体方差，即平均平方偏差。
- `H_after`：各 Fine 的平均梯度按距离权重计算，再将每个 GS 对各 Fine 均值的平方偏差按权重求和，除以同一批有效 GS 数量。
- `H_reduction`：上述二者之差。此值只用于分析，不参与激活排序或 loss。

各 Middle 按自身有效观测次数做时间平均。没有观测的 Middle 在 PT 中用 `diagnostic_valid=False` 标注，CSV 的三个诊断量留空，避免误认为零分。

第 50 步会写入：

```text
hierarchy_stats/frame_001_hierarchy_stats.pt
hierarchy_stats/frame_001_hierarchy_stats.csv
```

CSV 每个 Middle 一行，包含坐标、GS 数量、子节点数量、观测次数、有效可见 GS 累计数、三个诊断量、是否有激活资格及是否激活。PT 另含父子归属、每个 Fine 的支持/共享 GS 数量及比例、实际统计步号、开启前后位移和旋转输出的最大差值。每个统计步还会打印诊断均值。

## Fine 从零附加修正接入

保留 Fine 参数的单位变换初始化 `(0,0,0,1,0,0,0)`，在叠加前减去该初始值，按权重混合后作为 Fine 残差添加到 Coarse + Middle。全部叠加完成后才归一化总旋转分量。

Fine 正则仍参照单位变换，未激活行不产生正则梯度；正则平均的分母固定，避免仅因开启零残差子节点而改变父层正则强度。

第 50 步自动检查开启前后所有 GS 的位移和归一化旋转输出，最大差值超过 `1e-7` 时停止并报错。不增加额外 warm-up 或渐进 gate。

## 配置与后续运行

入口为 `config/prototype.yaml`，使用已有首帧 checkpoint 和 coffee_martini 的两帧范围。已有 V0 文件不用于本版恢复；本版 nested checkpoint 标记为 v2，仍支持从共享首帧 checkpoint 开始。

用户最初要求写完代码停下，随后授权运行首帧与后续五帧。数值检查和此次从零实验均已完成；结果见本次输出目录的 `report.md`、`results.csv` 与 `results.json`。在 ReConGS 环境、仓库根目录中，以下命令仍可用于后续复查或单帧实验：

```bash
# 数值检查，不加载场景渲染器。
python tests/test_nested_prototype.py

# 仅 GradHet；默认脚本不会启动 Random。
bash scripts/run_prototype.sh gradhet
```

此次六帧从零运行的命令为：

```bash
python main.py --pipeline=ReCon-GS --config=config/prototype_gradhet_scratch6.yaml
python scripts/summarize_gradhet_run.py output/prototype_gradhet_v2_scratch6_20260916/20260916T215258_f2aa0a
```

## 导出格式

`frame_storage/motion_001.npz` 的 `representation` 为 `nested_fine_residual_v2`。`delta` 存激活 Fine 的零参照残差与 Middle/Coarse 原参数；`index` 只存 Middle/Coarse 两层索引；`fine_indices` 与 `fine_weights` 保存软分配，索引 `-1` 表示零贡献。恢复时先混合 Fine 残差，再加两层参数，最后归一化总旋转。实际大小包含索引、权重与文件格式开销，不是正式 entropy codec 的码率。

尚未评估重叠带是否磨平真实运动差异，按当前约定留待后续实验。
