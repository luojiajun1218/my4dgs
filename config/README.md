# 当前训练入口

| 配置 | 用途 |
| --- | --- |
| [flame_steak_1020_full300.yaml](/home/gamer-02/Documents/DeepLearning/my4dgs/config/n3dv/flame_steak_1020_full300.yaml) | 正式300帧入口：storage 3e-5、无共享颜色、180步运动＋80步新生筛选＋40步固定点集精修，新增/删除数量允许10%差额。 |

当前首帧原生132,586 GS，正式加载初剪后128,963 GS、测试PSNR 33.217857 dB。使用已有首帧，不需要重新生成。共享颜色训练链路已删除，基础SH和新生GS的颜色拟合保留。

关键参数：

```yaml
module:
  rigid_tree:
    motion_steps: 180
    birth_steps: 120  # 80步筛选＋40步精修
    amplitude_refit_steps: 20  # 第161步才固定运动更新项
    rate_loss_weight: 3.0e-5
  birth:
    replacement_policy: balanced_current_births
    replacement_count_tolerance: 0.1
    refit_steps: 40
```

第260步新旧GS按量化透明度竞争，差的新点可以落选。通常一换一；保护新生GS所需的来源点时，最多允许少删 `floor(0.1 × 最终新增数)` 个旧点。零新增时不额外删除，最后40步不再增删点。

正式 300 帧：

```bash
conda activate ReConGS
cd /home/gamer-02/Documents/DeepLearning/my4dgs
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
python /home/gamer-02/Documents/DeepLearning/my4dgs/main.py \
  --config=/home/gamer-02/Documents/DeepLearning/my4dgs/config/n3dv/flame_steak_1020_full300.yaml
```

输出集中到 `/home/gamer-02/Documents/DeepLearning/my4dgs/output/n3dv/flame_steak_1020/full300_130k_balanced10pct_180_80_40/<时间戳>/`。

上一轮200＋60＋40的短验证见[实验记录](/home/gamer-02/Documents/DeepLearning/my4dgs/docs/experiment/flame_steak_1020_balanced10pct_200_60_40.md)，其中指标不代表本轮180＋80＋40。`flame_steak_1020_full300_storage2e-5.yaml`保留历史权重，不作为本轮入口；旧实验精确配置仍在原运行目录内。
