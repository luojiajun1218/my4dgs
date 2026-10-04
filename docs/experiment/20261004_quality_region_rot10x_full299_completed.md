# 完整 299 帧：平均 PSNR 超过当前 ReCon-GS 基线

## 结果先看这里

动态帧 1–299 全部完成，cam00 是唯一留出评测视角。本版平均 **34.05826 dB**，原版相同帧平均 **33.94978 dB**，高 **0.10848 dB**。224/299 帧高于原版。训练正常退出，最终 checkpoint 已保存。

| 范围 | 本版 PSNR | 原版 PSNR | 差值 |
|---|---:|---:|---:|
| 1–100 | 34.29176 | 34.07523 | +0.21654 dB |
| 101–200 | 34.05749 | 34.00356 | +0.05394 dB |
| 201–299 | 33.82317 | 33.76875 | +0.05442 dB |
| 全部 299 帧 | 34.05826 | 33.94978 | +0.10848 dB |

末尾仍有明显失败帧：第 284 帧比原版低 0.98293 dB，最后 20 帧平均低 0.24554 dB。因此目前达到的是完整序列平均 PSNR 目标，局部质量仍需要改善。

## 想解决什么、最终保留了什么

之前的版本存在门硬化时位姿跳变、精修更新过大，以及缺少观测的点默认运动等问题。区域旋转还会把离中心很远的点带偏。本轮保留独立的分层 m/z 门，针对这些问题调整实现和参数。

1. **变化门默认中性。** 逐 GS 的 m 初始概率为 0.5，硬化阈值为 0.51。没有学到变化证据的点保持上一帧位姿。
2. **细区域共享变化证据。** m 的 logit 由逐 GS 项和细区域项相加。相邻点可以共享观测信号，逐 GS 项仍能单独调整。
3. **保留软门拟合好的位姿。** 硬化时按软 z 缩放区域运动，已开启的局部残差补足软目标。m=0 精确返回上一帧；局部门关闭时无法保证完全补足。
4. **精修与区域旋转更新减小。** 固定门后使用新 Adam，精修学习率倍率 0.1。区域旋转学习率从 0.0005 降到 0.00005，逐 GS 学习率保持不变。
5. **运动量化更细。** 平移步长为 scene extent×1e-6，旋转步长为 0.0001 rad。

每帧仍为 100 步拟合运动、60 步学习门、40 步固定门精修、100 步新增 GS，共 300 步。storage 和 GS activity loss 均关闭；运动继承和二值 m 前向均关闭。没有新增训练损失或候选辅助渲染。

全序列平均约 51.66% 的旧 GS 关闭 m。逐帧检查确认：所有 m=0 旧 GS 的位置和旋转变化计数均为 0。这是位姿保持检查，不代表已准确识别语义背景，也不能据此证明压缩收益。

## 本轮搜索中没有保留的改动

| 改动 | 范围与结果 | 决定 |
|---|---|---|
| 逐 GS 学习率整体降低 10 倍 | 12 帧，比原版低约 0.959 dB | 不保留 |
| 门阶段缩到 20 步，精修增到 80 步 | 12 帧，比原版低约 0.094 dB | 不保留 |
| 继承 0.6 倍上一帧运动 | 24 帧，比原版低约 0.466 dB | 关闭 |
| m 前向直接使用二值，反向用直通梯度 | 24 帧，比原版低约 0.015 dB | 关闭 |

降低区域旋转学习率的单变量 24 帧对照高于原版 0.13335 dB；同一第 24 帧，最大位移从 extent 的 6.26% 降到 1.21%。这支持检查区域旋转带来的大位移，但尚不能认定它是唯一原因。完整结果也包含前面已经保留的门控与精修调整。

## 运行方式和比较边界

- cam01–20 训练，cam00 唯一留出评测；没有新增验证集。控制探针只读训练相机。
- 使用相同共享首帧、原有数据软链接和 resolution=-1，libgs 实际渲染宽度为 1600。
- 前 24 帧使用短对照运行，随后加载 `ckpt-24-300.pth`，在新目录继续 25–299。两段训练参数保持一致，指标按帧合并；检查没有缺帧或重复帧。
- 本版每帧 300 步，当前原版基线每帧 200 步，预算不同。
- 参数搜索查看过 cam00 前缀结果；这是单 seed 的当前场景结果，尚未证明独立重跑、其它场景或相同预算下的优势。
- storage 关闭，已有 float32 storage 精度检查仍 pending。此记录不证明 storage loss 已通过精度回归，也不证明存储压缩率。

## 代码与检查

核心实现位于 `pipeline/ReCon-GS/model/sparse_hierarchical.py`，阶段切换和梯度记录位于 `module.py`。新增可调参数均由 YAML 读取。最后一项区域旋转学习率改动只涉及 YAML。

启动前 108 项数值检查通过，覆盖门梯度、位姿重算、父门关闭时的子运动、m=0、导出、checkpoint 和可选运动继承。上述 pending storage 检查未执行。全序列 299 份最终评测均为有限数值且只有 cam00；日志没有 NaN 异常或异常退出。

## 配置、结果和来源

- 前缀配置：`config/fs_quality_region_rot10x_24f_20261004.yaml`。
- 继续配置：`config/fs_quality_region_rot10x_continue299_20261004.yaml`。
- 方便从共享首帧重跑的配置：`config/fs_sparse_hierarchical_psnr_candidate_300.yaml`。这份完整重跑配置已准备，尚未执行；本次实测使用上述两段运行。
- 前缀输出：`output/fs_quality_region_rot10x_24f_20261004/20261004T025652_2ba6fe/`。
- 继续输出：`output/fs_quality_region_rot10x_continue299_20261004/20261004T030718_29ed5a/`。
- 最终模型：继续输出中的 `ckpt-299-300.pth`；同时保留 50/100/200 帧 checkpoint。
- 完整逐帧 JSON/CSV、曲线 PNG/PDF：继续输出中的 `comparison/full299_*`。
- `series_inputs.json` 记录前缀来源和恢复 checkpoint 的 SHA-256；`source_snapshot/` 保存 10 个代码/配置文件及 SHA-256。
- `reference_evidence/` 保存原版指标来源 `/tmp/recongs_native_stream_playback_20261003.json` 和展开的 299 帧 cam00 指标，避免只依赖临时目录。
- 日志：`/tmp/my4dgs_quality_region_rot10x_24f_20261004.log`、`/tmp/my4dgs_quality_region_rot10x_continue299_20261004.log`。
- 数据依赖 `/home/gamer-02/Documents/DeepLearning/ReCon-GS/data/colmap4d/flame_steak`，其帧图像与 COLMAP 文件链接至 `/home/gamer-02/Documents/DeepLearning/S2GS/data/standard_flame_steak_300`；未复制数据或修改历史训练产物。

重跑命令：

```bash
conda activate ReConGS
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
python main.py --config=config/fs_sparse_hierarchical_psnr_candidate_300.yaml
```
