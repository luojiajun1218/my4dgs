# 交接：自适应运动层级 + bit loss

更新：2026-09-20。用户要求完成当前实验后交接；不要把研究目标标成已完成。

## 1. 用户真正想要什么

让模型自己判断哪里要细分运动、哪里共享粗运动。核心是**自适应层级和
真实 bit loss**，不是重叠技巧。attribute gate 下次再做；不要加两层前瞻补丁。
优先想清楚方法，不设固定PSNR完成线。完整比较保持接近原开销，短实验可多算。

## 2. 现在的方法，用人话说

父区域存共同运动，子区域只存差异。根据“运动近似误差 + 实际编码bit”
比较整棵候选树：有价值就细分，不划算就合并，各深度一起竞争。
图像梯度通过连续latent推动下一次树选择，被关闭的分支也能重新打开。

注意：候选分组仍来自父区域内的几何二聚类，**没有学到任意运动边界**。
树求解只对二次运动误差代理精确，不是直接精确优化图像PSNR；图像反传
用了有偏STE。详细公式见[方法草案](adaptive_hierarchy_bit_loss_proposal.md)。

## 3. 已有结果

flame_steak，共享首帧，100总帧/99动态帧，cam00测试。字节包含运动包、
协议和增量高斯，不含共享首帧与checkpoint；KB按1000字节。

| 方法 | seed0 PSNR | seed1 PSNR | seed1 KB/帧 | seed1 秒/帧 |
| --- | ---: | ---: | ---: | ---: |
| A：持续自适应 | 31.751 | 31.790 | 38.00 | 17.29 |
| D：固定深度16 | 31.997 | 31.292 | 44.93 | 17.83 |
| E：最后才选树 | 31.834* | **31.908** | **36.52** | 17.76 |

强参考ReCon-GS：32.467 dB，1610.81 KB/帧，16.22秒/帧（seed0，240次反传）。
当前A/D/E为180次反传，因树开销而耗时接近；不能称反传数相同或已胜过参考。

*E seed0中断后从frame60恢复，重算frame61有差异，是分段结果，不是干净重复。
E seed1连续正常结束：PSNR31.907666、SSIM0.955842；比A1高0.117769 dB，
仍比强参考低0.559536 dB。两个种子不足以证明稳定优势，99视频帧也不是99次独立重复。

**关键认识：**

- D两种子排名反转，不能拿D0当稳定的质量上界。
- 固定树去掉全零子树后，D0/D1总量可降至40.40/39.66 KB，解码逐值不变。
  这部分是无损编码去冗余，不是自适应训练贡献；额外编码耗时尚未优化。
- E在20次运动更新中一直用完整树，最后才选结构。因此它是延迟选择对照，
  不能用它证明“训练中持续自适应”优于后处理。
- latent数值大不一定代表解码运动大：叶内零均值部分不影响当前SGD的树选择。
  不要仅凭latent长尾就加入裁剪。

## 4. 下一步怎么接

当前全序列实验已经结束；不要重跑同名目录。独立解码最终状态见本文末尾。

下一项已准备但**没有运行**：只对图像梯度做指数平均，当前bit包络梯度不平均。
假设是减少最近4个视角对细分的影响；也可能因滞后变差，不能预先认定有效。

```bash
cd /home/gamer-02/Documents/DeepLearning/my4dgs
source /home/gamer-02/Documents/Codex/miniforge3/etc/profile.d/conda.sh
conda activate ReConGS
python tests/test_nested_prototype.py
# 确认GPU空闲后再运行；只用训练相机，不用cam00调参
python scripts/probe_rd_tree_render.py --config config/idea_rd_tree_render_ema51.yaml
```

它比较不平均、beta=.5、beta=.75；同一A0 frame50起点、同视角顺序、同树。
看另4个训练相机的误差、真实packet字节及轨迹，不只挑最好一步。
若有收益，再换视角顺序/前帧来源；不直接接进完整训练。
这是优化稳定性对照，不是新的核心贡献。若无收益，如实记录并换假设。

## 5. 接手时找这些文件

- 核心代码：`pipeline/ReCon-GS/model/rd_tree_motion.py`、`rd_tree_projection.py`。
- EMA仅在诊断器：`scripts/gradient_ema.py`、`scripts/probe_rd_tree_render.py`；
  正式训练尚未接入。数值测试已通过，新实验前仍重新跑完整测试。
- 结果检查：`scripts/summarize_controlled.py`、`audit_rd_tree_run.py`、
  `audit_rd_tree_stream.py`。前者查计数不等于后者连续解码。
- 完整研究记录：[日志](rd_structure_search_20260920.md)（目前到第三十二轮）、
  [消融协议](rd_tree_ablation_protocol_20260920.md)、
  [相关工作区别](rd_related_mechanism_check_20260920.md)。不要把通用层级、
  Haar、RD剪枝或STE本身声称为创新。

以下路径均相对仓库根目录：

```text
A0 output/flame_rd_tree_envelope_full100_20260920/20260920T020653_189351
A1 output/flame_rd_tree_envelope_seed1_full100_20260920/20260920T093426_4598cf
D0 output/flame_rd_tree_fixed16_full100_20260920/20260920T041948_54d701
D1 output/flame_rd_tree_fixed16_seed1_full100_20260920/20260920T101919_bd1aae
E0 output/idea_resumed_rd_composite_20260920T092845_921587  （分段，见限制）
E1 output/flame_rd_tree_delayed_seed1_full100_20260920/20260920T110007_d26755
强参考 output/flame_recongs_240backward_full100_20260919/20260919T220421_2f05e2
```

各完整run内有`controlled_summary.json`、配置和源码快照。A1/D1/E1训练源码hash相同，
配置仅预定结构选项不同，训练视角逐项配对已检查。

## 6. 安全约束

先读根目录AGENTS.md；用ReConGS环境；不改历史结果/数据软链接，不复制大数据。
新实验用新experiment_name；GPU任务串行。保留内部名称`ReCon-GS`以兼容加载。
用户要求本轮交接后停止新增实验，由下一agent继续。

## 收尾验证

E1已正常exit0、保存frame99 checkpoint；99帧计数、源码快照、视角和字节
核查通过。独立连续解码也已exit0：从共享首帧重建全部99帧，10个已保存
checkpoint的基础六属性和增量高斯全部逐值一致（不代表跨平台保证）。报告：
`output/idea_rd_tree_decode_audit_20260920T112945_268125/report.json`。
本轮训练和核查均结束，没有留下需要等待的实验进程；EMA实验没有启动。
