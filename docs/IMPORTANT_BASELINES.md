# 特别重要的保留版本

## 完整300帧：34.0547 dB / 38.19 KB / 233 FPS

**级别：特别重要。** 后续改进以该版本为基线，已有结果和配置应保留。

| 项目 | 内容 |
| --- | --- |
| Git标签 | `important-full300-34.0547db-38.19kb-20261017` |
| 配置 | `config/fs_alpha_norot_quant6_rate003_full300_20261017.yaml` |
| 完整300帧平均PSNR | 34.0546715037 dB |
| 299个动态帧完整更新均值 | 38.1949297659 KB，KB=1000 B |
| 纯渲染速度 | 233.01442487 FPS，不含解码与上传 |
| 一次性首帧 | 17,720,236 B，另计；摊入全序列为97.135067 KB/帧 |
| 本地结果 | `output/fs_goal_rd_20261017/all_rotation_off_quant6_full300/20261017T183218_af18a4/` |
| 轻量证据 | `docs/experiment/evidence/important_full300_20261017/` |

方法为绝对共享区域＋GS平移残差、图像Adam＋近端收缩、关闭全部旋转更新、透明度直接量化与阈值删除、无损编码。动态更新包含运动、透明度、新增GS属性和manifest。训练使用70步联合学习、30步固定gate、100步新增GS，运动每步2视角。

该标签保存本次实验51个冻结的Python源文件，逐项核验SHA-256；后续原型另行提交。Git保留代码、配置和轻量结果，原始checkpoint、图像及码流仍在本地output。完整核验和逐段分析见[实验记录](experiment/20261017_all_rotation_off_full300.md)。

复现时可创建单独工作树，并指定新的实验名：

```bash
git worktree add ../my4dgs-important important-full300-34.0547db-38.19kb-20261017
conda activate ReConGS
python main.py --config=config/fs_alpha_norot_quant6_rate003_full300_20261017.yaml \
  experiment_name=fs_important_baseline_reproduction_NEW
```

新工作树的数据访问仍需沿用本机软链接；配置中的绝对路径依赖现有数据和共享首帧。当前结果距离34.2 dB目标仍差约0.1453 dB，全部动态帧最终仅选8个区域，learnable gate细分收益尚未得到证明。
