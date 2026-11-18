# N3DV 六场景速度统计（2026-11-14）

## 设置

统计 2026-11-13 六场景 baseline 和 ReCon-GS、QUEEN、S2GS 的复现实验；显卡为 RTX 3090 Ti。训练时间使用原有日志，取动态帧 1–299 的算术平均，不包含首帧训练。未修改训练算法。

各项目计时范围不同：baseline 的 `training_seconds` 是帧循环墙钟时间，减去诊断记录时间，包含验证；ReCon-GS 为 GPU 同步的 `frame` 计时，覆盖逐帧训练和 PLY 保存，验证在计时之外；QUEEN 与 S2GS 为原项目同步的分段 `Frame time`，验证在计时之外，原始结果中的 `Frame time IO` 另有数据读取时间。这些值不能视为完全统一计时口径。

| 场景 | baseline 秒/动态帧 | ReCon-GS | QUEEN | S2GS |
|---|---:|---:|---:|---:|
| coffee_martini | 19.34 | 15.20 | 7.26 | 4.28 |
| cook_spinach | 18.58 | 13.97 | 7.41 | 9.55 |
| cut_roasted_beef | 19.02 | 14.30 | 6.83 | 8.98 |
| flame_salmon | 18.99 | 15.07 | 7.72 | 8.75 |
| flame_steak | 17.93 | 12.96 | 7.06 | 9.27 |
| sear_steak | 17.34 | 13.71 | 7.04 | 9.00 |

详细原始来源、各项目 I/O 计时和数值见 `output/experiments/n3dv_speed_20261114/training_times.json`。

## baseline 渲染和完整播放

执行 `scripts/fps_render_calculate.py`，测试 RTX 3090 Ti、cam00、原图分辨率，GPU 无其他计算任务；每个场景连续读入 300 时间点。已解码纯渲染对每个时间点预热 3 次并计时 10 次；完整播放从首帧开始重放动态帧 1–299，重复两轮，计时包含磁盘读包、CPU 解码与状态摘要校验、GPU 状态更新、渲染，不含首帧和相机加载。所有帧都校验 GPU 和 CPU 状态；结果见本次独立实验目录 JSON。

| 场景 | 已解码渲染 FPS | 完整播放 FPS |
|---|---:|---:|
| coffee_martini | 172.88 | 12.31 |
| cook_spinach | 275.12 | 13.06 |
| cut_roasted_beef | 254.03 | 12.76 |
| flame_salmon | 238.68 | 13.98 |
| flame_steak | 264.71 | 13.15 |
| sear_steak | 344.06 | 13.92 |

上述完整播放是现有实现及其逐帧验证开销的实测，不是仅渲染 FPS；此测量没有对 CPU 解码、校验、GPU 更新分别做剖析。coffee_martini 的纯渲染 FPS 低于 200，其余五个场景高于 200。

## 其他项目现有输出的限制

ReCon-GS 本轮只保存增量高斯 PLY，缺少完整逐帧变形状态，无法从这些结果独立播放；其训练日志 FPS 未进行 GPU 同步，不能作为严格渲染 FPS。S2GS 本轮没有保存可用的逐帧模型或压缩码流，无法对已保存结果重建序列并实测完整播放。

S2GS 原生训练日志包含逐帧测试视角单次同步渲染 FPS，算术均值如下；这些是训练中单次调用的统计，未做独立多次预热和重放，不能当作完整播放 FPS 或直接作为统一口径的速度验收。

| 场景 | S2GS 日志单次渲染 FPS 均值 |
|---|---:|
| coffee_martini | 244.08 |
| cook_spinach | 276.49 |
| cut_roasted_beef | 243.25 |
| flame_salmon | 247.96 |
| flame_steak | 276.57 |
| sear_steak | 243.60 |

原生首帧训练单列：baseline 与 ReCon-GS 直接加载共享首帧；QUEEN / S2GS 第一帧的项目内 `Frame time` 如下，预处理和启动耗时可能不在此计时内。

| 场景 | QUEEN 首帧秒数 | S2GS 首帧秒数 |
|---|---:|---:|
| coffee_martini | 192.56 | 81.39 |
| cook_spinach | 218.98 | 68.93 |
| cut_roasted_beef | 208.25 | 75.92 |
| flame_salmon | 214.94 | 81.96 |
| flame_steak | 220.11 | 68.15 |
| sear_steak | 214.30 | 75.05 |

## QUEEN 解码与播放测速

从原生首帧 PLY 顺序加载逐帧 PKL、解码并更新状态、渲染测试相机 cam00；每动态帧在 GPU 同步后单独计时并将耗时求和，帧间穿插不计入的渲染测速。已解码渲染每帧预热 3 次并计时 10 次。单轮完整序列；不包含首帧和相机加载。以第 2、150、300 个时间点和原先保存的 PNG 做差异检查；低于 35 dB 者标注为需核查，不能当成有效的原结果回放。

| 场景 | 已解码渲染 FPS | 完整播放 FPS | 样本与原 PNG 的最低 PSNR / dB |
|---|---:|---:|---:|
| coffee_martini | 304.31 | 0.46 | 42.10 |
| cook_spinach | 342.73 | 0.71 | 49.50 |
| cut_roasted_beef | 458.48 | 0.63 | 41.10 |
| flame_salmon | 283.89 | 0.48 | 56.19 |
| flame_steak | 364.65 | 0.77 | 48.56 |
| sear_steak | 368.22 | 0.77 | 57.88 |

同一场景中实际保存的文件约 15～25 MB/动态帧。此结果描述当前项目的原始解码路径和存储格式；未把单次渲染计时当作完整播放。详情与测速脚本位于 `queen/output/n3dv_speed_20261114/`。

## ReCon-GS 末帧渲染样本

从 299 动态帧的最终 checkpoint 重建已解码高斯，仅测最后一帧 cam00 纯渲染：预热 20 次，计时 5 组各 100 次 GPU 同步渲染。每帧重建图像须与原训练结果 PNG 的 PSNR 不低于 40 dB。该值是末帧样本，不能代表完整序列平均渲染 FPS；本次缺少完整逐帧运动状态，无法测完整播放。

| 场景 | 末帧渲染 FPS | 与原 PNG 的 PSNR / dB |
|---|---:|---:|
| coffee_martini | 535.21 | 58.99 |
| cook_spinach | 330.43 | 58.92 |
| cut_roasted_beef | 345.40 | 58.92 |
| flame_salmon | 362.68 | 58.96 |
| flame_steak | 394.23 | 58.92 |
| sear_steak | 448.71 | 58.92 |

结果和测速脚本位于 `ReCon-GS/output/n3dv_speed_20261114_retry1/`。首次启动因测速脚本导入路径错误而退出，未修改训练代码。
