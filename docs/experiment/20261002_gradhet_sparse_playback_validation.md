# GradHet 稀疏回放验证 — 2026-10-02

## 2026-10-03 后续校正

本记录正文反映 10 月 2 日当时的基线缺口；次日已补齐 ReCon-GS 299 帧原生流并完成同边界审计，详细数据见[基线审计](20261003_recongs_native_baseline_audit.md)。原始 ReCon `np.savez` 与 GradHet `np.savez_compressed` 的目录字节数不同，不能据此给出跨方法存储结论；基于混合 serializer 的任何 ReCon-GS 对 GradHet 节省判断现已撤回。

此前 sparse-v2 比 GradHet dense-v1 少 18.54% 的结论只适用于 GradHet 内部两种表示，不是相对 ReCon-GS。相同 `np.savez_compressed` 边界下，GradHet sparse-v2 总流为 578,126,778 B，ReCon-GS 为 462,637,201 B；新共享 child-index 表 v3 暂测 538,571,989 B，仍比 ReCon-GS 大 16.41%。v3 不改变运动或画质，仍不构成论文算法收益。

## 做了什么

验证现有高质量 GradHet 运动结果能否无损稀疏保存和逐帧回放，不重训。v2 只写 gate 打开的行及对应 blend 索引、权重；解码器重建每个 Gaussian 的 7 维运动。代码在 `pipeline/ReCon-GS/model/gated_four.py`，加载、渲染和逐帧状态推进在 `pipeline/ReCon-GS/module.py`。

GPT-6 Luna 独立审查认为，对有限浮点导出，`einsum` 累加与旧公式一致；它指出 `-1` 哨兵在占位锚点含 NaN/Inf 时会因乘零泄漏非有限值。现已改为显式清零并新增回归用例。完整 57 项数值检查通过；因本环境无 KeOps C++ 编译器，测试用 `torch.cdist` 代替测试初始化中的 KNN，decoder 与 renderer 检查仍走项目代码。成对相机 DataLoader 现有 30 秒 worker 超时、顶层 collate 函数，并在成功或异常时关闭一次性 worker；正常读取、超时清理与 spawn worker 测试均通过。没有训练。

## 质量和速度

之前的全序列 dense-v1 / sparse-v2 对照覆盖 299 帧、每帧 held-out `cam00`：motion field 逐元素相同，渲染像素最大差为 0。当前 decoder 在帧 1、150、299 又逐帧与 v1 精确对照；修正后的 sparse-only 生产回放完整走完 299 帧。

| 帧 | cam00 PSNR / SSIM | sparse renderer FPS（4 视角范围） |
|---:|---:|---:|
| 1 | 34.012 / 0.97192 | 140.5–156.2 |
| 150 | 34.160 / 0.96522 | 156.2–173.0 |
| 299 | 34.124 / 0.96161 | 179.8–203.8 |

最终 DataLoader 修复后再次运行了有 90 秒外层上限的多视角 GPU 抽测。视角为 cam00（test）及 cam01、cam10、cam20（train），每帧 5 次预热、30 次 CUDA 同步渲染。ReCon-GS 只有帧 299 检查点，因此只有这一帧能配对比较：GradHet 的 PSNR/SSIM 差值依次为 cam00 `−0.147 dB / −0.00629`、cam01 `−3.364 / −0.01987`、cam10 `−1.246 / −0.01640`、cam20 `−1.999 / −0.01163`。render-only FPS 比值（GradHet/ReCon-GS）为 `1.11–1.13`。除 cam00 外都是训练视角，不能当成独立 held-out 多视角质量结论；不同轮次的 FPS 也有时钟/顺序波动。

之前 299 帧 cam00 汇总为 GradHet PSNR `33.78295 dB`、SSIM `0.964306`；ReCon-GS 历史日志为 `33.94223 dB`、`0.968533`。差值约 `−0.1593 dB / −0.00423`。这是历史训练汇总，硬件计时和输出保存节奏没有完全配对。

decoder 的 `einsum` 优化后，单路 299 帧 GPU 回放完成于 53.50 秒（5.589 帧/秒）。包解码、增量 PLY 读取和设备传输平均 66.47 ms，单次渲染 6.39 ms，状态推进 3.11 ms。旧实现同口径分别为 94.49 ms、6.20 ms、2.72 ms；包加载段降低约 29.7%，但总 FPS 只从 5.516 到 5.589，因为总时间还包含逐帧图像 DataLoader 读取。该 FPS 包含图像读入，文件可能命中系统缓存，不代表纯部署 FPS。

## 存储口径和基线缺口

ReCon-GS `README.md` 建议逐帧保存必要 Gaussian 属性和 deformation 参数，再按实际文件字节计量；它没有要求每帧写完整场景 PLY。回放所用共享帧 0 是训练后的 `gaussians-0-15000.ply`：214,996 行、22,360,216 B。`input.ply` 的 2,040,265 B 只是 40,000 点初始种子，不能代替共享训练帧 0。

| 299 帧实际文件 | dense-v1 | sparse-v2 |
|---|---:|---:|
| motion NPZ + 增量 PLY | 681,988,300 B | 555,574,102 B |
| 加一次共享训练帧 0 | 704,348,516 B | 577,934,318 B |

v2 比 dense-v1 少 126,414,198 B：动态帧文件少 18.54%，计入共同共享帧后少 17.95%。这是同一 GradHet 回放表示之间的真实磁盘体积差，不是参数量估算。

当前 ReCon-GS 历史输出只有 frame 0 与 frame 299 检查点，以及 frame 0 完整 PLY 和 frame 299 增量 PLY；配置每 1000 帧才保存 checkpoint/GS，缺少 frame 1–298 的原生 `delta/index`、出生点及删除/行序记录。因此不能按 README 的逐帧实际文件法计算 ReCon-GS 的 299 帧动态流，也不能声称 v2 比 ReCon-GS 压缩多少。完整场景 PLY 可单列为一种实际资产表示，但不应用它夸大基线。公平的后续基线流需从共享训练帧 0 出发，逐帧写原生 7 值 anchor delta/index、新增点 PLY，以及准确删除索引和行序信息；SH、opacity、scale 沿用的部分不重复计费。

## 对论文方向的影响

稀疏导出证明了一个无需训练的编码收益，并可保持现有 GradHet 画质；decoder CPU 开销仍是实际瓶颈。多视角抽测提示帧 299 的质量差异可能不止 cam00 上的约 0.15 dB，但当前只有一个 held-out 视角。训练耗时也尚未证明变慢：历史全帧 wall time 是 ReCon-GS 8.82 秒、GradHet 13.62 秒，但基线每 1000 帧保存一次，GradHet 每帧保存；200 步的训练加反传均值分别约 7.62 和 7.70 秒。现有墙钟不能用来下训练效率结论。

下一步先以同一批多视角、多个困难帧配对质量和 render-only FPS；然后为 ReCon-GS 保存可重放的原生逐帧流，按相同精度、相同基础资产和文件边界实测字节。当前历史文件不足以完成这项基线存储比较，且本次没有启动训练。
