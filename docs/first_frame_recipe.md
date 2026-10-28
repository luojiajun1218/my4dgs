# 首帧处理操作指南：减少GS数量，同时保住动态重建质量

给后续agent：当前验证有效的路线是**从已有高质量首帧中，按训练视角的图像影响裁点，保留所选点的全部原始属性，并保持原候选空间划分**。当前Flame Steak从加载后的206237点降到150000点，完整300帧画质基本保持、略有提升。它不是从零训练15万点首帧的方法，也不保证所有场景在15万点时都达到同样结果。

复现时先读本文，再读[完整300帧结果](experiment/20261028_cov150_full300.md)和[每帧训练计划](frame_training_plan.md)。仓库代码、配置和本文进入Git；checkpoint、分数和reference留在本地，通过路径访问，不提交原始训练产物。

## 1. 当前已经验证到什么程度

| 相同动态训练流程 | 原首帧 | 本轮保留原属性的首帧 |
|---|---:|---:|
| 加载后的初始GS数 | 206237 | 150000 |
| 首帧PSNR，项目原生/dB | 33.952248 | 33.958199 |
| 首帧解码参考包/MB | 17.720236 | 13.118597 |
| 全300帧PSNR，项目原生/dB | 34.064716 | 34.120266 |
| 完整动态更新/KB每帧 | 50.970251 | 48.962441 |
| 最后一帧GS数 | 192764 | 152597 |

当前完整300帧按整幅RGB8公式计算为34.091273 dB，尚未达到希望的34.2 dB。点数减少27.27%，动态存储只减少3.94%，因此首帧压缩有效，但不是动态存储问题的全部解法。48.96 KB是299次动态更新的平均；包含13.12 MB首帧后，完整流摊至300帧仍为92.53 KB/帧。KB=1000字节，MB=1000000字节。

## 2. 输入、坐标和分辨率

以下路径是本机已经验证的Flame Steak输入，均只读：

| 用途 | 路径 |
|---|---|
| 原始“重要版”SH1首帧checkpoint | `/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf/ckpt-0-15000.pth` |
| 原首帧候选空间与解码参考 | `output/flame_steak_1027/fs_units_300/20261028T060649_97d995/frame_storage/rigid_tree/reference_frame0.npz` |
| 本轮实际使用的15万点checkpoint | `output/flame_steak_1028/fs_seed150c/20261028T130623/ckpt-0-15000.pth` |
| 完整300帧配置 | [fs_cov150_300.yaml](../config/fs_cov150_300.yaml) |
| 资格30帧配置 | [fs_cov150_30.yaml](../config/fs_cov150_30.yaml) |

原checkpoint实际为214996点；按物理透明度`alpha >= 0.0075`加载后为206237点。本文的筛点、区域覆盖和保留行编号都基于**过滤后的206237行**，不能直接拿这些行编号去索引原214996行。

数据使用`data/colmap4d/flame_steak_half`对应的1345×1008图像，`resolution: 1`，SH degree=1，cam01–cam20用于筛点，cam00单列评估。数据通过软链接访问。不能只改图像尺寸却不核对相机内参、相机划分及相机归一化尺度；GS坐标也不能随意乘图像缩放比例。分辨率影响投影、像素覆盖和损失，不会把GS点变成二维图像点。

原`reference_frame0.npz`含首帧属性及候选空间层级。裁点脚本要求它的`xyz`与当前checkpoint经0.0075过滤后的`xyz`**逐位一致**；动态初始化复用模板时还核验场景extent、空间深度与离群点路由。若文件缺失，应先从**未裁点的同一首帧**构建新的frame0 reference，不能使用其他场景或裁点后重建的划分冒充原模板；不需要为此重跑完整300帧。

本机场景可用下面的初始化命令补建模板。该命令加载已完成的原首帧，仅处理frame0，随后结束；从日志读取新run路径，将其中的`frame_storage/rigid_tree/reference_frame0.npz`作为`first_reference`：

```bash
conda activate ReConGS
python main.py --config=config/fs_units_300.yaml \
  experiment_name="fs_f0ref_$(date +%Y%m%dT%H%M%S)" \
  data.extra_dataset_kwargs.num_frames=1 \
  module.rigid_tree.spatial_gate.reference_layout=null \
  gui_enabled=false
```

这一步的原始checkpoint、透明度阈值、相机与层级设置必须与随后筛点一致；新生成NPZ的文件SHA不一定与历史文件相同，应核对实际属性及布局内容。

本地已验证输入指纹：

```text
原checkpoint SHA256:
0715e9f0ce2eaea16633207fb9f1e87170162f9fb2974f8b9f1a20206da35c7d

原reference_frame0.npz SHA256:
0fed82393ca2efd0d99189361242abf994ea90277ba1c35a896d2772f7bc901b

实际15万点首帧的raw属性摘要（reference_digest，不是pth文件SHA）:
7fa53b8471437c40624a50eebc1df30749918784b6aa219d8c73c64878f34730
```

## 3. 筛点依据：图像影响与区域覆盖

实现为[scripts/compact_initial_model.py](../scripts/compact_initial_model.py)。固定首帧，渲染训练相机；对每幅渲染图做两个固定种子的随机正负投影，通过原生渲染器反向计算每个点的透明度响应，再累加平方能量。它近似衡量“去掉该点可能使图像发生多大变化”，不是按GS大小、透明度或某个视角的颜色误差单独排序。

评分不使用GT像素，也不使用cam00，更不使用未来帧的运动、PSNR或残差来选点。GT用于后续质量评估。原生合成中的遮挡与透明度参与响应计算，但分数仍只是局部近似：多个点同时删除会相互影响，原生有限扰动抽查的相对偏差最高约30%，不能称为精确删除代价。

选择流程：

1. 先做既有`alpha < 0.0075`过滤。
2. 计算全部剩余GS的图像影响分数，按分数从高到低稳定排序。
3. 在原有每个非空最细候选区域中，优先保留`min(2, 该区域原有点数)`个最高分GS。
4. 用剩余名额按全局分数填满到150000点；保持原行顺序输出，选中的GS不重新排序为按分数排列。
5. 只删除未选中的行，所选GS的位置、旋转、logscale、透明度、SH DC和高阶SH全部保持原值。

本轮原有4050个非空最细区域全部保留。单纯全局top150000会清空39个区域，覆盖约束相对该选择只替换119个名额。可细分节点数仍为583，不能把覆盖约束描述为增加了可用gate，也没有验证这119个点独立贡献了多少画质。

这20个视角的评分只在首帧离线做一次；逐动态帧仍按当前计划使用4个训练视角，不会新增每帧20视角候选筛选。

## 4. 复现当前首帧的命令

在仓库根目录使用同一个终端执行。先确认GPU没有正在运行的其他训练；本命令需要GPU，会生成新的小首帧及统计文件，但不启动动态训练、不复制数据、不修改输入checkpoint。

```bash
conda activate ReConGS
cd /home/gamer-02/Documents/DeepLearning/my4dgs

first_source="/home/gamer-02/Documents/DeepLearning/ReCon-GS/output/flame_steak_fullres/20260926T225223_dc5baf/ckpt-0-15000.pth"
first_reference="output/flame_steak_1027/fs_units_300/20261028T060649_97d995/frame_storage/rigid_tree/reference_frame0.npz"
first_seed_dir="output/flame_steak_1028/fs_seed150c_repro_$(date +%Y%m%dT%H%M%S)"

python scripts/compact_initial_model.py \
  --checkpoint "$first_source" \
  --render-config config/fs_units_300.yaml \
  --coverage-reference "$first_reference" \
  --target-count 150000 \
  --opacity-threshold 0.0075 \
  --probes 2 \
  --seed 1729 \
  --output "$first_seed_dir/ckpt-0-15000.pth"
```

如果shell未初始化conda，可先执行：

```bash
source /home/gamer-02/Documents/Codex/miniforge3/etc/profile.d/conda.sh
conda activate ReConGS
```

输出有三个文件：`ckpt-0-15000.pth`为首帧；`ckpt-0-15000.json`记录输入指纹、过滤/目标点数、筛点训练相机、前后画质和覆盖统计；`ckpt-0-15000.scores.pt`保存分数与过滤后原布局中的保留行。输出路径必须是新的，不能覆盖历史run；不要把这些产物加入Git。

GPU浮点归约及分数边界附近的排序可能带来小差异。精确复用已验收版本时直接使用上表的实际15万点checkpoint；重新评分时记录自己的属性摘要与结果，不因行数一致就宣称与历史checkpoint逐位一致。

## 5. 接入动态训练：保留原划分，不续训首帧

当前成功流程**没有在裁点后调用`scripts/run_first_frame.py`继续训练首帧**。保留checkpoint的`current_frame=0`及原完成步数15000，动态配置保持`trainer.num_init_steps: 15000`，随后从该已完成首帧建立解码参考并进入动态帧。

`module.rigid_tree.spatial_gate.reference_layout`必须继续指向与筛点时相同的原始候选模板。新参考按保留下来的GS重新路由，但复用原划分平面、候选区域中心和离群点路由；gate及区域运动重新学习。新reference内嵌这些结构数据，后续解码不依赖外部模板文件。

本机复现实验可直接用现有配置，只覆盖新checkpoint和新的实验名：

```bash
# first_seed_dir沿用上一节同一终端中的变量；如果使用已有首帧则显式重设它。
python main.py --config=config/fs_cov150_30.yaml \
  ckpt_path="$first_seed_dir/ckpt-0-15000.pth" \
  experiment_name="fs_f0r30_$(date +%Y%m%dT%H%M%S)"
```

若把`first_reference`换成新构建的同首帧模板，还要覆盖：

```text
module.rigid_tree.spatial_gate.reference_layout=<该模板的实际路径>
```

30帧完成、质量和码流核验通过后，再以同一配置方法从同一首帧新跑完整300帧：

```bash
python main.py --config=config/fs_cov150_300.yaml \
  ckpt_path="$first_seed_dir/ckpt-0-15000.pth" \
  experiment_name="fs_f0r300_$(date +%Y%m%dT%H%M%S)"
```

两条训练命令依次执行；先分析30帧，合适才执行300帧。不要为了接入首帧同时改动图像损失、量化精度、出生额度、共享颜色或动态训练阶段。当前共享颜色关闭。前30帧过关不能证明后半段过关，完整结论仍需同一方法的全300帧。

## 6. 下一个agent应核验什么

- **输入和行映射：** 原214996点→透明度过滤后206237点→150000点；重新加载后数量不能因为不同阈值或配置再次意外减少。分数/保留行对应过滤后的布局。
- **保留属性：** 六类原始属性必须与原checkpoint经过同样过滤、按保留行索引后的值逐位一致。旋转不能额外改约定或做一次不必要的原地归一化；SH阶数保持1。
- **区域覆盖和参考：** 读取报告`coverage`，检查非空最细区域保留及`guarded_regions_below_available_minimum == 0`；新动态reference的GS属性与新checkpoint一致，候选划分与原模板一致。首帧gate重置并重新学习，不照搬旧训练运动。
- **保存与恢复：** 原有Adam行状态与裁点后行数一致；checkpoint中的Parameter在恢复后仍可训练。现有回归检查为`python -m unittest discover -s tests -p test_initial_checkpoint.py`。本轮没有恢复静态优化，但仍须保持保存格式正确。
- **实际图像：** 相同相机、GT、分辨率、背景、SH阶数下渲染前后首帧，查看图像与PSNR/SSIM；记录项目逐通道dB平均、整幅RGB浮点和RGB8公式的区别。cam00已经参与开发决策，不能宣称独立测试泛化。
- **动态资格：** 同一首帧先跑完整30帧，观察火焰等困难区域、完整动态包、残差行、实际新增/退出及总GS数；通过后做完整300帧、独立CPU/GPU解码/像素核验，并分别测渲染和完整播放速度。

已验收15万点动态首帧的状态摘要为`86f2233dfc4f6a62defef00398107485f1fb9a964bc0f0768fa4eb1b7f2e05b1`，参考包13118597字节；状态摘要还包含布局，不能只靠属性摘要替代它。证据详见上述完整300帧记录，历史诊断文件在`tmp/first_frame_v3/coverage_seed_proof.json`、`cov150_dynamic_bootstrap.json`和`cov150_300_bootstrap.json`，这些tmp文件不是长期唯一依据。

## 7. 已经踩过的坑与其他场景的适用范围

**静态PSNR更高，不保证后续动态更好。** 一版全属性续训15万点首帧的静态PSNR为34.186138 dB，但动态30帧只有34.019742 dB；本轮保留原属性的首帧静态33.958199 dB，动态30帧反而有34.227168 dB。单独续训SH2500步也没有改善首帧。它们是已有实验现象，不能据此证明某一种属性变化必然是唯一原因；后续若要续训，应作为新候选另行验证。

**15万点是当前场景验证有效的工作点，不是所有数据集的最优预算。** 其他场景先准备自身质量可靠的首帧、相机与frame0 reference，再用该场景训练相机评分和候选覆盖约束。保留“筛点、属性不变、空间划分一致、动态验证”这条流程，不能直接套用Flame Steak的空间模板、源checkpoint、绝对点数或预期PSNR。

当前裁点脚本还显式校验cam00和1345×1008评估尺寸，且拒绝用cam00筛点；迁移到不同相机命名、留出划分或分辨率的场景前，要先适配这些校验和相应配置。不要为了通过检查悄悄缩放GT或改测试划分。它只接受已完成的静态frame0 checkpoint，不能把动态rigid checkpoint直接当同类输入。

首帧模板缺失时，可从原首帧配置生成一个`num_frames: 1`、已完成首帧步数匹配、`reference_layout: null`的新初始化run：当前`Trainer.setup()`会写出`frame_storage/rigid_tree/reference_frame0.npz`。先核对原始属性与坐标，再把该文件用于覆盖选择和后续参考；无需训练动态帧。若要从零训练新场景，当前动态`main.py`要求共享已完成首帧，应先使用项目已有的原生静态入口；本文未验证一套适用于所有新场景的低点数从零训练配方。

后续agent的交付应包括输入来源/指纹、命令和配置、点数/区域/属性核验、前后首帧图像，以及实际动态30/300帧结果。只有完成至少30帧的训练实验才在`docs/experiment/`新增实验记录；本文是操作指南，不是额外短实验报告。
