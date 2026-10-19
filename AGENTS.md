运行项目前记得 conda activate ReConGS
数据通过软链接访问，不复制大型数据。
本地 data/colmap4d/flame_steak 指向 /home/gamer-02/Documents/DeepLearning/ReCon-GS/data/colmap4d/flame_steak；其帧图像与 COLMAP 文件链接到 /home/gamer-02/Documents/DeepLearning/S2GS/data/standard_flame_steak_300。当前 配置使用 resolution: -1、num_frames: 300，并加载共享首帧；libgs 会自动缩小宽图。
output 中历史结果仅用于读取；目录可能包含软链接，不修改链接目标。新实验使用新的 experiment_name。Git 不保存原始训练产物。

## N3DV 全场景数据检查（2026-10-18）

共享数据根目录为 `/home/gamer-02/Documents/DeepLearning/data/n3dv`，六景处理结果在 `<scene>_result`，每相机 300 帧，去畸变后半分辨率宽约 1341–1345。完整检查见 `docs/experiment/20261018_n3dv_preprocessing_audit.md`。图像完整性已通过，但 coffee_martini、flame_salmon_1 的种子点云存在远距离长尾，仍需核实。只有 flame 保留原去畸变 COLMAP 模型，不能混用不同版本的位姿和 PLY。

作者要求本轮六景首帧全部重新训练，不复用历史首帧检查点。旧 flame 的 1600×1199 指标独立保留，半分辨率需要重新统计。`config/n3dv/flame_steak_half_frame0.yaml` 使用 native ReCon-GS 入口，只训练新首帧；当前 my4dgs 动态入口仍要求完成的 native 首帧 checkpoint。

新的 flame 首帧已完成：`output/n3dv_half_fresh_frame0_20261018/flame_steak/20261018T224824_9ca22a/ckpt-0-15000.pth`，1345×1008、15000 步、237495 GS，cam00 PSNR 32.991226 dB；已用 my4dgs 渲染器独立读取核对。这是首帧指标，完整六景动态训练尚未启动。

flame 半分辨率连续 300 帧已于 2026-10-19 完成：配置 `config/n3dv/flame_steak_half_fresh_full300.yaml` 使用 `resolution: 1` 和新首帧；第 0–299 帧平均 PSNR 34.265514 dB，动态完整更新 60.913512 KB/帧，纯渲染 291.513 FPS，训练及评估 29.7221 秒/动态帧。输出 `output/n3dv_flame_steak_half_fresh_full300_20261018/20261018T231032_84d9ab`，详见 `docs/experiment/20261018_flame_steak_half_fresh_full300.md`。存储目标尚未达到；每字节惩罚按初始保留 GS 数归一化，跨分辨率/场景比较需留意点数变化。其余五景尚未启动完整动态训练。

2026-10-19 半分辨率改进已完成连续 300 帧及独立顺序解码：**34.208911 dB、50.796388 KB/动态帧、291.792 纯渲染 FPS**，训练评估 29.4643 秒/动态帧。相比同分辨率基线存储少 16.61%，PSNR 低 0.056603 dB；达到画质、约 50 KB 和纯渲染目标，严格 ≤50 KB 尚未达到。新增可选 `rate_normalization: budget` / `rate_budget_bytes: 50000`，只是损失单位；本次每字节系数 6e-10，比基线提高 64.47%，不能把收益仅归因于归一化。使用原本新训练的 15000 步半分辨率首帧，未采用动态表现更差的 17000 步精修参考。284/299 动态帧存在有效细分，94.688% 旧 GS 无逐点平移残差。输出 `output/fs_goal_half_budget_20261019/absolute_bytes_full300/20261019T023630_678e85`，统计 `output/fs_goal_half_budget_20261019/measurement_full300`，冻结源码 `output/fs_goal_half_budget_20261019/source_v1`；详见 `docs/experiment/20261019_half_byte_budget_full300.md`，前期短测见 `docs/experiment/20261019_half_byte_budget.md`。从首帧复现用 `config/n3dv/flame_steak_half_byte_budget_20261019.yaml`，实际完整输出配置则包含第 29 帧续跑指针。实验及 FPS 测量时为 350 W；作者提供管理员认证后，已于 2026-10-19 恢复默认 480 W，NVML 配置/生效上限均为 480 W，限功率服务 inactive/disabled，遥测服务保持 active/enabled。恢复记录见 `output/fs_goal_half_budget_20261019/power_restore_success.json`；既有 291.792 FPS 仍应标注为 350 W 测得。其余五景完整训练尚未启动。

2026-10-19 作者指定的 **480 W 最终正式复跑已完成并核验**：标准 1345×1008、300 帧、cam00，**34.211170 dB / 51.027672 KB/动态完整更新 / 292.205 纯渲染 FPS**，训练与评估 29.4586 秒/动态帧。今后 flame_steak 最终指标以此次为准，不使用此前 350 W 的结果替代。固定原字节代价配方，加载本轮此前新训的 15000 步半分辨率首帧，从第 0 帧参考开始，未继承动态前缀、未改训练参数。推荐配置 `config/n3dv/flame_steak_half_final480w.yaml`；输出 `output/n3dv_flame_steak_half_final480w_20261019/training/20261019T091653_42b563`；逐帧统计在 `output/n3dv_flame_steak_half_final480w_20261019/measurement`；最终结果指针 `output/n3dv_flame_steak_half_final480w_20261019/final_result.json`。记录 `docs/experiment/20261019_flame_steak_final480w_full300.md`，含独立顺序解码、15000 次渲染计时及源码摘要。质量与纯渲染速度达标，存储比严格 50 KB 高约 2.06%；纯渲染 FPS 不含解码/上传，首帧参考另计。291/299 动态帧存在有效细分，94.705% 旧 GS 无逐点平移残差。其余五景完整训练尚未开始。

# 特别须知
实验汇总：最理想的做法是每次做完实验后，在docs/experiment/文件夹下面 写一个对应的md文档（别乱写文档，做完实验 再写文档），
文档中记录这次实验的核心idea，以及对应的代码改动（简要版的，不需要特别详细，可以渐进式披露），运行的结果分析，以及可能造成这种结果的原因。
相当于是一次实验记录文档，不然有些实验 我也记不清楚他们是干嘛的了。

我（作者）一开始的想法很简单就是 让改动的地方动，不改动的地方不动。然后我想到了共享运动这一出发点，
具体而言 我大致的做法是：
1.先让gs自由残存运动，然后使用聚类算法将残存运动相似的gs聚类成运动区域，然后我们可以对这些运动进行建树，子类区域的真实运动=父类区域运动+自身残存；区域中的gs点运动也是如此。应该每过k帧需要重新建树。
2.storage loss，存储loss，尽量少的gs变化，让需要改变的gs变化（现在还有个量化效果，应该会让小的运动偏向于归0了，这样就不用传输了）；还有就是子类区域的relative motion loss，让残存运动尽量小
3.效果达到SOTA，全300帧平均 psnr在34.2db以上（一定一定要超过34db），每帧存储平均在50kb以下（可以略微放宽），fps在200以上，每帧训练时间目前不限制（但是不能太长了）


# 仓库贡献指南

## 项目结构与模块组织

`main.py` 是训练入口，默认使用 `--pipeline=my4dgs`。实现直接放在 `pipeline/`，其中 `entry.py` 提供 `Config` 和 `Pipeline`，`data.py`、`module.py`、`trainer.py` 分别负责数据、模型训练逻辑和训练调度，`model/` 存放运动与码流模块。`pipeline/__init__.py` 为 `libgs` 提供加载入口，`_compat.py` 兼容旧的 `pipeline.ReCon-GS` 导入，无需保留中间目录。新代码使用 `pipeline.model.*` 等规范导入路径。实验配置放在 `config/`，数值回归检查放在 `tests/`，方法说明和来源记录放在 `docs/`。`data/`、`output/` 与 `references/` 属于本地数据或实验产物，其中可能包含软链接；不要随意复制大型数据或修改链接指向的原文件。

## 构建、测试与开发命令

执行命令前先激活现有环境：

```bash
conda activate ReConGS
python main.py --config=config/xxxxx.yaml
```

测试脚本无需加载渲染器，可检查层级采样、权重混合、梯度诊断、训练调度和 checkpoint 恢复。本项目没有单独的构建步骤。

## 本机故障恢复与代理命令

### 显卡掉卡：`gpu-recover`

- 适用情况：宿主机上的 `nvidia-smi` 无法识别显卡，或内核日志出现 `GPU has fallen off the bus`、Xid 79、要求重启的 Xid 154，且机器仍能执行终端或 SSH 命令。先区分真实掉卡与受限运行环境无法访问 GPU；仅训练报错或 CUDA 显存不足时，应先排查程序。
- 先用 `gpu-recover --check` 检查远程连接所需服务；用 `gpu-recover --status` 查看当前状态和上次开机检查报告。两者都不会重启。
- 确认需要通过重启恢复时运行 `gpu-recover`。它会申请 sudo 权限、收集诊断、配置网络/SSH/Tailscale/Clash 开机恢复，并在诊断完成后安排 **1 分钟后整机重启**。重启会中断训练，执行前应保存可保存的进度并确认其他任务状态。
- 只收集诊断用 `gpu-recover --collect-only`；重启发生前可用 `gpu-recover --cancel` 取消。日志保存在 `/var/log/gpu-recovery/`，不要在仓库中保存 sudo 密码。
- 重启后，从已接入同一 Tailscale 网络的设备尝试 `ssh gamer-02@100.124.74.30`，再检查 `gpu-recover --status` 和 `nvidia-smi`。本地服务检查通过不等于外部 SSH 已验证；整机完全卡死时也无法依靠此脚本恢复。

### 代理变慢或连接超时：`clash change`

- 适用情况：Clash 代理已运行，但访问外部服务超时、下载很慢，或 Codex 持续显示 reconnecting。先用 `clash status` 查看当前节点、实测带宽和连通性，再按需运行 `clash change` 测速并选择节点。
- `clash change` 会临时切换候选节点做带宽实测，再选择节点并检查服务端点连通性；测速期间可能影响现有代理连接。默认实测 3 个候选，可用 `clash change --top N` 调整。
- `clash change --dry-run` 也会临时切换节点，正常结束时恢复原节点。当前切换只作用于运行中的代理内核；需要在 Clash Verge 重启后保留时，在界面中手动选择同名节点。
- 如果提示找不到控制 socket，先检查 Clash Verge 服务及代理内核是否运行；换节点命令需要已有可用的代理内核。

## 与用户协作
- 你的输出最好简短并通俗易懂

## goal模式须知（进入goal模式必读）
- goal模式自动运行时，尽量别做什么对照实验，我们的目的是为了发论文，是为了整体的效果变好才让你继续自动运行的。目前还在改进效果，还没到消融实验的阶段。
- 现在最大的问题应该是这里的机制不work，我需要你自动分析实验结果，找出可能的原因（代码逻辑上的疏忽 / 整体方法的问题）并加以改进。


