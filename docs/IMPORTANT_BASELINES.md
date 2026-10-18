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

新工作树的数据访问仍需沿用本机软链接；配置中的绝对路径依赖现有数据和共享首帧。该基线结果距离34.2 dB目标仍差约0.1453 dB，全部动态帧最终仅选8个区域，未体现learnable gate细分收益。

## 共享发现完整300帧：34.2444 dB / 50.92 KB / 226.9 FPS

**本轮已验证新方案。原“特别重要”版本继续保留，不覆盖。**

| 项目 | 内容 |
| --- | --- |
| Git标签 | `verified-shared-gate-full300-34.2444db-50.92kb-20261018` |
| 方法源码 | `7c462fb`，70个冻结Python文件与Git逐项哈希一致 |
| 配置 | `config/fs_shared_discovery_depth4_full300_20261018.yaml` |
| 完整300帧平均PSNR | 34.2444124985 dB |
| 动态完整更新均值 | 50.9171872910 KB，含全部负载与manifest；比50KB高1.83% |
| 纯渲染速度 | 226.93125313 FPS，不含解码与上传 |
| 一次性参考 | 17720236 B，另计；摊入全序列为109.814917 KB/帧 |
| 本地结果 | `output/fs_goal_rd_20261017/gate_shared_discovery_warm20_full300/20261018T113940_72a5de/` |
| 实验记录 | [完整300帧记录](experiment/20261018_shared_discovery_full300.md) |
| 轻量证据 | `docs/experiment/evidence/shared_discovery_full300_20261018/` |

方法先学习共享区域/gate，再开放全GS残差。283个动态帧有细分、16帧采用粗层。全部299帧独立顺序解码、渲染及10个终态检查点核验通过；PSNR差和缓存像素差均为0。PSNR和FPS达标，存储按用户允许略微放宽的约50KB口径接受，严格≤50000B标记保留为false。

复现使用现有`ReConGS`环境、数据软链接和共享首帧，以新实验名从第0帧开始；结构化解析已验证完整module配置与实际实验一致。新方案更新比旧基线大，画质更高，两者分别保留。
