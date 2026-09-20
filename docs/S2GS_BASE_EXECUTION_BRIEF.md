# s2gsBase 执行任务说明：Adaptive Hierarchical Residual Gating V1

> 目标：把研究主线从当前 ReCon-GS / motion-tree 探索中分离出来，回到“只让真正需要变化的 Gaussian 产生 residual”的原始 idea。  
> 工作分支：`s2gsBase`。只在该分支工作，不改 `main`，不覆盖历史实验。  
> 本文只定义 V1；不要擅自加入 anchor、motion tree、entropy model、真实 codec、Geo/App gate、Gumbel 三分类或新的神经网络。

## 1. 研究问题

S²GS 已经通过 structured sparse gating 减少不必要的 residual updates，但 gating 粒度主要在固定的较粗区域上完成。我们要验证的第一个问题只有一个：

**在保持 S²GS 其余训练与 residual 表示不变时，按每帧局部动态比例自适应选择 OFF / UPDATE / SPLIT，能否以接近的重建质量减少 active Gaussian residuals 和实际 storage？**

V1 不追求最终论文方法，只做最小、可解释的 hypothesis test。

## 2. 必须采用的底座

1. 以官方 S²GS 实现为 baseline，保留其：
   - 数据加载与训练流程；
   - Gaussian residual 表示；
   - VGD / dynamic confidence 计算；
   - 原有 photometric / sparse regularization；
   - residual 保存、storage 统计与评估代码；
   - 官方训练/测试视角划分。
2. 不再以 ReCon-GS anchor motion 为主线。
3. 当前仓库中的 `rd_tree_*`、`binary_motion.py`、ReCon-GS prototype 都作为历史探索保留，不接入 V1。
4. 导入 S²GS 后，记录官方仓库 URL、上游 commit SHA、许可证和本分支对应的 baseline commit。

## 3. V1 的核心语义

### 3.1 两层状态，不要混淆

节点状态：

```
a_j^t ∈ {OFF, UPDATE, SPLIT}
```

最终 Gaussian mask：

```
m_i^t ∈ {0, 1}
```

`SPLIT` 只存在于区域决策过程中；递归完成后才生成每个 Gaussian 的最终 0/1 mask。

最终 residual：

```
R_i_final = m_i * R_i
```

V1 对 active Gaussian 的**全部属性 residual 正常更新**。不要做 geometry / appearance 分离。

### 3.2 OFF / UPDATE / SPLIT 的定义

- **OFF**：该区域内最终没有 Gaussian residual update。
- **UPDATE**：该区域停止继续细分；区域内 Gaussian 按当前区域的决策进入 residual update。
- **SPLIT**：当前区域内部动态分布不均匀，继续在更细一级空间网格上判断。

注意：UPDATE 不是“共享一份 motion”。每个 active Gaussian 仍然使用 S²GS 原本自己的 residual。

## 4. Dynamic confidence：第一版直接沿用 S²GS

不要新做 optical flow、SAM、motion network 或其他先验。

直接复用 S²GS 已有的 per-Gaussian dynamic confidence `p_i`（由 VGD 得到）。V1 只改：

**dynamic confidence → residual update decision**

不要改 VGD 本身。

将连续 confidence 转为 binary dynamic evidence：

```
d_i = 1[p_i > tau_dyn]
```

`tau_dyn` 必须放到配置文件，不能硬编码。第一轮只做少量候选值的训练视角诊断（例如 0.5 / 0.8 / 0.9），不得用 test view 挑阈值。

## 5. 不建立显式父子树：使用 level-wise region id

V1 不需要 Node 类、parent pointer、child pointer、motion coefficient tree。

最多使用 **6 层空间粒度**：

```
level = 1, 2, 3, 4, 5, 6
```

实现要求：

1. 复用 S²GS 的空间层级 / octree 坐标体系；
2. 对每个 Gaussian 预计算或快速得到每层的 `region_id[level]`；
3. 区域继续 SPLIT 时，只需在下一层按 `region_id[level+1]` 重新 group；
4. 不额外保存显式 parent-child adjacency；
5. 不为 SPLIT 创建 motion 参数；
6. 同一帧的 OFF / UPDATE / SPLIT 状态每帧重新计算；
7. V1 暂时保持 S²GS 的空间 hierarchy 本身固定，不做 periodic hierarchy rebuild。

也就是说：

**每帧刷新 gating decision；不刷新 spatial topology。**

## 6. Dynamic ratio 决策规则

对于当前区域 j：

```
N_j     = 区域内 Gaussian 数
N_dyn_j = sum(d_i)
r_j     = N_dyn_j / N_j
```

V1 使用最简单、可解释的规则：

```
if N_dyn_j == 0:
    OFF
elif r_j >= tau_update:
    UPDATE
elif level < 6:
    SPLIT
else:
    finest-level fallback
```

其中 `tau_update` 放配置文件，不硬编码。

### 最细层 fallback

到第 6 层后如果仍然是 mixed region，不再创建第 7 层。直接使用该 region 内 per-Gaussian 的 `d_i` 作为最终 mask：

```
m_i = d_i
```

这样最大空间划分深度固定为 6，同时避免第 6 层 mixed cell 被迫整体 UPDATE。

### 第一版 OFF 规则

先不要引入额外 `tau_off`。只有 `N_dyn_j == 0` 才 OFF。

原因：一个大区域里即使只有 1 个真正动态 Gaussian，也应该优先 SPLIT，而不是因为 dynamic ratio 很低直接关掉。

如果之后诊断发现 VGD 噪声导致大量单点误触发，再单独讨论 minimum dynamic count 或 `tau_off`。

## 7. V1 不做的事情

本轮明确禁止自动加入：

- ReCon-GS anchor / hierarchical motion；
- parent motion + child difference；
- `rd_tree_projection` / DP 选树；
- Exp-Golomb / torchac / entropy model；
- true bitrate loss；
- 新的 storage proxy loss；
- Gumbel-Softmax 三分类；
- learnable OFF / UPDATE / SPLIT；
- optical flow / SAM；
- Geo/App gate；
- periodic hierarchy rebuild；
- EMA / latent clipping；
- 任何为了“跑分”而新增的复杂补丁。

V1 先使用 S²GS 原有 loss，只改变 gating granularity。

## 8. 实施顺序

### Stage A：只读代码审计，先不要改

先定位官方 S²GS 的真实调用链，并返回：

1. 训练入口文件；
2. VGD / dynamic confidence 的文件、类、函数；
3. HFP / root-level aggregation 的文件、类、函数；
4. gate 产生与应用到 residual 的文件、类、函数；
5. residual 各属性的数据结构；
6. storage/export 的代码位置；
7. octree / multi-level region index 的代码位置；
8. 当前 root gate 最终如何映射到 leaf Gaussians；
9. S²GS 的默认层数 / root resolution / scene-specific 配置；
10. baseline 一条可执行命令。

如果官方代码结构与论文描述不一致，要明确指出，不要自行假设。

### Stage B：建立可复现 baseline

在 `s2gsBase` 上先保证**未修改算法的 S²GS baseline**能跑。

至少完成：
- 数值/单元 smoke test（若上游有）；
- 一个短序列或少量帧；
- 保存 baseline 配置、commit、日志；
- 确认 PSNR / active Gaussians / storage / training time 的统计路径可用。

baseline 没跑通前不要写 V1。

### Stage C：实现最小 Adaptive Hierarchical Gate

建议单独新增小模块，例如：

```
adaptive_hierarchical_gate.py
```

核心接口尽量简单：

```python
mask, stats = build_adaptive_mask(
    dynamic_confidence=p,
    region_ids=region_ids_by_level,
    tau_dyn=...,
    tau_update=...,
    max_depth=6,
)
```

输出：
- `mask: [N] bool/0-1`
- 每层 OFF / UPDATE / SPLIT 区域数量
- 每层 Gaussian 数量
- 最终 active Gaussian 数量与比例
- depth histogram
- dynamic-ratio histogram / quantiles

然后只把该 mask 接到 S²GS 原 residual update 处。

### Stage D：先做短实验，做完必须停

不要直接跑完整 300 帧。

建议先选 S²GS 已有的一个 N3DV scene，用相同首帧和相同训练视角，跑：
- baseline；
- V1；
- 相同训练 epoch / step；
- 相同随机种子；
- 相同配置，除了 adaptive gate 参数。

先跑少量连续动态帧（例如 5–20 帧，具体按上游成本决定）。

短实验结束后停止，先把下面的信息返回给研究讨论，再决定是否做完整实验。

## 9. 执行 Agent 必须返回的信息

请不要只回复“代码写好了”或“效果不错”。必须按下面格式返回。

### A. 代码与来源

- 当前 branch / commit SHA：
- S²GS upstream URL：
- S²GS upstream commit SHA：
- 新增文件：
- 修改文件：
- 每个修改文件做了什么：
- 是否改动 baseline loss：是/否
- 是否改动 VGD：是/否
- 是否改动 residual representation：是/否
- 是否引入额外模型/网络：是/否

### B. 原 S²GS 调用链

用一段非常具体的流程写清楚：

```
frame images
→ ...
→ VGD
→ p_i
→ ...
→ root gate
→ ...
→ residual
```

并给出对应文件名和函数名。

### C. V1 实现逻辑

必须明确：

- `tau_dyn`
- `tau_update`
- max depth = 6
- 每帧是否重新计算 decision
- hierarchy topology 是否固定
- finest-level fallback 怎么处理
- all-attribute residual mask 接在代码哪里

### D. 单元/数值验证

至少验证这些 toy cases：

1. 全静态 region → OFF；
2. 全动态 region → UPDATE；
3. mixed region → SPLIT；
4. split 到第 6 层后不会再创建第 7 层；
5. 第 6 层 mixed region 使用 per-GS `d_i`；
6. 关闭 adaptive 模式时，baseline 输出不应被修改；
7. adaptive mask 为全 1 时，residual forward 与 baseline 对应路径一致。

### E. 短实验结果

Baseline 与 V1 至少返回：

| metric | Baseline | V1 |
|---|---:|---:|
| PSNR | | |
| SSIM（如已有） | | |
| active Gaussian count | | |
| active Gaussian ratio | | |
| storage / frame | | |
| train time / frame | | |

另外返回：

- 每层 OFF / UPDATE / SPLIT 数量；
- 最终 mask 的 depth 分布；
- dynamic ratio 的分布摘要；
- 每帧 active Gaussian ratio 曲线或表；
- 是否出现 PSNR 突降；
- 是否出现 storage 反而上升；
- 任何异常/失败，不要隐藏。

### F. 结论只能回答三个问题

1. V1 是否真的比 fixed root gate 减少 active residual Gaussians？
2. 画质损失有多大？
3. 实际 storage 是否也跟着下降？

不要提前声称论文贡献成立，也不要自行进入 bitrate loss / Geo-App / 新网络阶段。

## 10. 代码风格与安全

- 所有新参数写入 YAML/config；
- 不覆盖历史 output；
- 新实验名必须唯一；
- 不使用 test view 调阈值；
- 先跑短实验，再决定长实验；
- 新逻辑尽量保持为独立小模块，方便随时删除；
- 不修改 `main` 分支；
- 任何需要大规模重构 S²GS 的情况，先停下来报告原因。

## 11. 当前研究优先级

目前只验证：

**fixed root-level gating 是否确实存在 mixed-region 误更新，以及 adaptive spatial granularity 能否减少这些多余 residual。**

只要这个问题还没有被一个简单、受控实验回答，就不要继续增加方法复杂度。
