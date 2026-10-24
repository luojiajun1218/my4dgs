# 当前训练入口

| 配置 | 用途 |
| --- | --- |
| [flame_steak_1024_multiview_select.yaml](n3dv/flame_steak_1024_multiview_select.yaml) | 已完成300帧：平均PSNR 34.160024 dB、动态含索引39.626836 KB/帧。295帧准入，但出生当帧平均净收益仅+0.000879 dB；详见完整报告。 |
| [flame_steak_1024_multiview.yaml](n3dv/flame_steak_1024_multiview.yaml) | 已完成300帧：平均PSNR 34.137506 dB、动态增量33.508612 KB/帧。仅36/299帧提交新点，供复现及分析。 |
| [flame_steak_1020_full300_multiview.yaml](n3dv/flame_steak_1020_full300_multiview.yaml) | 上一版多视角方案的历史入口。当前运行配置以1024实验目录中的快照为准。 |
| [flame_steak_1020_full300_birthwarmup.yaml](/home/gamer-02/Documents/DeepLearning/my4dgs/config/n3dv/flame_steak_1020_full300_birthwarmup.yaml) | 失败候选，暂不用于300帧：单动态帧180→300下降0.095097 dB。 |
| [flame_steak_1020_full300.yaml](/home/gamer-02/Documents/DeepLearning/my4dgs/config/n3dv/flame_steak_1020_full300.yaml) | 已完成的旧方案：180＋80＋40，新生阶段旧透明度固定，保留作历史配置。 |
| [diagnostic/flame_steak_birth_revision.yaml](diagnostic/flame_steak_birth_revision.yaml) | 独立短诊断，使用`scripts/probe_birth_revision.py`，不能传给`main.py`。只加载保存的运动终态，验证补点和退役候选。 |

最新完整实验平均PSNR为34.160024 dB，比上轮34.137506提高0.022517，距34.2仍差0.039976。准入率从36/299升到295/299，但178个准入帧在cam00下降；80步增删与40步精修的平均净收益分别仅+0.000614、+0.000265 dB。8帧固定属性渲染显示，加入最终新点本身的测试影响更大，删旧直接影响很小；这些样本不代表完整序列的因果结论。详见[最新完整实验报告](../docs/experiment/20261024_flame_steak_multiview_select_full300.md)。[上一版300帧与新版早期短验证](../docs/experiment/20261024_flame_steak_multiview_full300_and_selection.md)保留供追溯，不能替代这次完整结果。

新版300帧命令：

```bash
conda activate ReConGS
cd /home/gamer-02/Documents/DeepLearning/my4dgs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python main.py --config=config/n3dv/flame_steak_1024_multiview_select.yaml
```

180＋80＋40调度保持：出生候选由16台拟合相机的高误差像素提出，每条射线搜索9个深度，至少3个视角共同支持。181–260步保留cam04/08/12/16用于新点图像收益排序和配套增删验收；261–300步固定点集，让新点使用全部20台训练相机精修，旧点属性全部固定。cam00始终只评价。验收试选整批及正收益新点的100%/50%/25%子集，各自删除量与新增量相差不超过10%；旧点仍按两帧合成贡献排序，保护编码来源及上一帧新生点。平均验收PSNR需增加超过0.005 dB、SSIM下降不超过0.0001。早期多视角原型见[历史短记录](../docs/experiment/20261024_flame_steak_multiview_birth.md)。

运动阶段的实际细分如下（当前`shared_discovery: true`）：

| step | 实际更新与作用 |
| --- | --- |
| 1–20 | 预拟合共享区域运动与旧透明度；gate参数不更新，点级运动残差保持零。 |
| 21–70 | 拟合共享运动，同时学习gate选择空间共享层级；第70步末冻结gate。 |
| 71–160 | 固定gate，联合拟合共享运动、点级平移残差与旧透明度。 |
| 161–180 | 固定允许更新的运动块集合，精修这些块的幅值；旧透明度仍拟合至180。180步末一次性生成新点候选。 |
| 181–260 | 固定全部旧点属性，拟合候选新点的位置、SH、尺度和透明度；260步末一次完成新点筛选与旧点退出。 |
| 261–300 | 点集固定，仅精修已接纳的新点属性；300步编码和评价。 |

本轮旋转更新关闭。新点没有透明度预热，也不会在220/240/280步继续clone、split或剪枝；`densify_interval`等梯度增密参数属于保留的旧策略。运动稀疏收缩发生在运动选择阶段，`terminal_motion_proximal: false`关闭额外的运动终清理；GS增删在260步提交。storage loss 3e-5约束运动与旧透明度，本轮新点拟合没有单独的出生码率loss。

下文保留历史失败预热候选的说明，供复现；参数不是新版配置。

当前首帧原生132,586 GS，正式加载初剪后128,963 GS、测试PSNR 33.217857 dB。使用已有首帧，不需要重新生成。共享颜色训练链路已删除，基础SH和新生GS的颜色拟合保留。

历史预热候选参数：

```yaml
module:
  rigid_tree:
    motion_steps: 180
    birth_steps: 120  # 80步筛选＋40步精修
    amplitude_refit_steps: 20  # 第161步才固定运动更新项
    rate_loss_weight: 3.0e-5
  birth:
    replacement_policy: birth_driven_retirement
    replacement_count_tolerance: 0.1
    max_new_gaussians: null
    refit_steps: 40
    warmup_steps: 20
    initial_opacity: 0.1  # 真实alpha，内部转换为logit
  gaussian_stage2:
    feature_lr: 0.075
    opacity_lr: 0.05
  opacity_update:
    prune_threshold: 0.01
    joint_with_birth: true
```

181–200步仅拟合新点位置和SH，新点alpha约0.1且尺度固定，旧点属性固定。201–260步开放新点透明度、尺度，同时更新旧点透明度；旧点位置、颜色、尺度、旋转固定。261–300步固定点集和旧透明度，联合精修已选运动幅值与新点属性。图片取色尚未实现，本轮颜色仍继承来源GS。

新点透明度学习率从0.75降到0.05，颜色学习率仍为0.075。每帧`birth_control.learning_trace`记录180/185/190/200/201/220/260/300步的透明度、位置/颜色梯度及属性训练状态；`birth_control.prune_reasons`区分低alpha、投影尺寸、世界尺度和多重原因（单项计数可重叠）。阶段测试指标增加step200，cam00只评价、不参与训练和剪枝。

第260步先完成新生点自身的透明度、尺寸筛选，存活的新点数记为K；增删配额不再额外淘汰新点。旧点中，当前量化透明度低于0.01的可删除点数记为D0，再将实际删除数D限制到 `ceil(0.9K) ≤ D ≤ floor(1.1K)`：不足下限时继续补删透明度最低的旧点，超过上限时只删最弱的上限数量。新生点编码依赖的旧来源点受到保护；若保护后无法满足下限，会明确报错，不会悄悄丢弃新点。

例如K=100时，D0为0、97、100、150，分别删除90、97、100、110个旧点。允许相等，但不再默认一换一；K小于10时，整数约束仍会导致新增、删除相等。K=0时不额外删除。最后40步固定点集精修，不再增删点。

复现预热候选的命令（该候选短验证失败，暂不建议执行300帧）：

```bash
conda activate ReConGS
cd /home/gamer-02/Documents/DeepLearning/my4dgs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python /home/gamer-02/Documents/DeepLearning/my4dgs/main.py \
  --config=/home/gamer-02/Documents/DeepLearning/my4dgs/config/n3dv/flame_steak_1020_full300_birthwarmup.yaml
```

历史预热候选输出集中到 `/home/gamer-02/Documents/DeepLearning/my4dgs/output/n3dv/flame_steak_1020/full300_130k_birthwarmup20_jointalpha_180_80_40/<时间戳>/`，继续从原`frame0.pth`启动。该候选尚未跑300帧；已完成的单动态帧验证未显示该帧画质收益。

`flame_steak_1020_full300_storage2e-5.yaml`保留历史权重，不作为本轮入口；旧实验精确配置仍在原运行目录内。历史策略 `balanced_current_births` 保留供旧配置恢复，它的新旧点竞争逻辑与当前策略不同。此前不带新生预热的方案已完成300帧：平均PSNR 34.093082 dB、完整动态增量39.205699 KB/帧、GS从128,963增至136,672；详见[实验分析](../docs/experiment/flame_steak_1020_birthdriven10pct_180_80_40.md)。
