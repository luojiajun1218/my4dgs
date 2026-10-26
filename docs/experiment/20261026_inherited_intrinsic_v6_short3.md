# 继承角向辐射与跨视角intrinsic DC校准：短3帧

完成日期：2026-10-26。frame0＋2动态帧全部完成、独立解码通过。固定132586 GS首帧、1345×1008半分辨率监督，learnable gate与码流格式不变，尚未达到300帧目标。

## 核心机制

新增点不再从图像混合RGB直接自由重拟合全部外观。geometry cohort中固定DC/SH_rest，repair继承父角向外观，细分后代继承父DC/SH_rest，MVS初始SH_rest为零。每个既有600步cohort末，用真实合成DC梯度和原生隔离颜色响应计算ΣJr与ΣJ²，按跨视角一致性软收缩DC残差，再以全20训练视角真实量化MSE回溯验证。选择最终结构后仍完整训练600步再编码验收，失败回退。

新helper为`pipeline/model/birth_radiance.py`，`pipeline/module.py`接入阶段顺序/冻结，`residual_birth.py`改变后代颜色继承；新增helper和集成回归。两位普通代码agent分工实现，未使用Command Code。

## 结果

| 指标 | 结果 |
|---|---:|
| 全3帧平均PSNR | 33.523491 dB |
| 两动态帧平均PSNR | 33.676308 dB |
| 新增长阶段增益，frame1 / 2 | +0.036644 / -0.119022 dB |
| 动态更新含manifest均值 | 76.713 KB/帧 |
| 全3帧完整场景渲染FPS | 288.41 |
| 接收birth，frame1 / 2 | 249 / 301 |
| 动态帧平均训练时间 | 285.66 秒 |

全部3帧重解码核对哈希、质量与缓存。FPS为每帧10次预热后50次同步完整渲染+clamp，排除解码/上传/指标。KB为十进制，共享首帧11081043字节另计。短2动态帧存储受第一大运动包影响，不能推断长期平均。

继承SH外观和显式求解仍未解决测试退化，因此未扩到30帧。

## 发现的逆问题

第一轮269候选的颜色残差一致性均值仅0.0397；不少误差不能用同一个intrinsic DC解释。对角响应估计也存在低可见性病态：第一轮最大DC更新152.81，细轮271.61。一个MVS候选原始RGB约0.57/0.38/0.27、alpha约0.035，校准后本色约23/43/9。训练MSE仍可下降，因为它在训练视角受到遮挡，未见视角却可能产生极端亮度。

真实Jacobian计算本身通过数值验证；问题是没有约束的逆解把低alpha/T下的误差吸收到巨大颜色，不能把“训练损失降低”当作可信的材质更新。后续改为source-envelope盒约束：LDR新点不凭空产生HDR或负色，已有父角向HDR/负SH范围保留，真实量化增量落在可行格点内。该修正未追溯计入本轮。

## 验证与证据

冻结前全367项CPU测试通过。原生小GPU场景中，梯度投影误差6.52e-8，H=0.442413、有限差分0.442420，场景差0，校准降低MSE；这些仅验证物理统计实现，不能替代真实序列成绩。

- 输出：`output/flame_steak_1025/fs_inherited_radiance_v6_short3/20261026T030412_a7a108`。
- 配置/源码：`tmp/inherited_radiance_v6_short3.yaml`、`tmp/experiments/inherited_radiance_v6_short3/source_manifest.json`。
- 解码：`tmp/birth_diagnosis/inherited_v6_decoded.json`；摘要`inherited_v6_summary.json`同目录。
- 异常颜色：`tmp/birth_diagnosis/v6_large_dc_audit.json`。
- 原生验证：`tmp/birth_radiance_native_validation.json`。
- 后续source-envelope修正：全371 CPU检查通过；小GPU中无界更新149.08约束为1.77245、量化RGB约0.99931且MSE仍下降，见`tmp/birth_radiance_bounded_native_validation.json`。实际序列效果待另行实测。
