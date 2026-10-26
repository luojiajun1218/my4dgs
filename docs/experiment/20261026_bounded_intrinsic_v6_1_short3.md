# 源点辐射包络约束：短3帧

完成日期：2026-10-26。首帧＋2动态帧完成，全部独立解码通过。固定132586 GS首帧、1345×1008半分辨率监督、learnable gate和码流格式不变，未达到300帧目标。

## 改动

针对v6低可见性导致的巨大颜色逆解，对每点intrinsic DC增量加入源角向辐射包络。新LDR点不凭空产生HDR/负色；已有父SH角向HDR/负范围保留。连续解与真正量化增量共同投影到可行格点，再用全20训练视角完整画面MSE回溯验收；0步始终可行。没有调整学习率或候选预算。

`pipeline/model/birth_radiance.py`增加包络及量化可行集；allocation和动态续跑白名单更新为`bounded_intrinsic_radiance_v6_1`。普通agent修改，冻结v6结果未动。

## 结果

| 指标 | 结果 |
|---|---:|
| 全3帧平均PSNR | 33.578152 dB |
| 两动态帧平均PSNR | 33.758299 dB |
| 新增长阶段frame1 / 2增益 | -0.000504 / -0.126556 dB |
| 含manifest动态更新均值 | 77.7405 KB/帧 |
| 全3帧完整场景渲染FPS | 287.50 |
| 接收birth，frame1 / 2 | 312 / 310 |
| 平均动态帧训练时间 | 279.59 秒 |

第一帧两次探索校准，未受限解最大327.3/601.8，投影后的实际更新最大2.49/2.45；巨大颜色放大已阻断。最终结构初始化与选中真实场景MSE逐视角差为0。数值正确性改善没有转化为测试质量提升，因此没有运行30帧。

KB为十进制，共享首帧11081043字节另计；短序列受第一动态帧大包影响，不能推断长期平均。FPS为每帧10次预热后50次同步完整渲染+clamp，不含解码、上传和指标。全3帧顺序核对哈希、质量及缓存。

## 机制判断

问题已经超出颜色数值极值。之前v6最终真实类型复核：124个repair子点（62个父）单独使测试下降0.11554dB、训练提升0.07692dB；177个MVS点使测试下降0.01012dB。主要损伤仍来自旧点的细分替换。

v6纯CPU族审计还发现：最终光学质量代理/成熟父中位0.707；12/62族中心超父1σ，4族超3σ；极薄法向存在离群变化。质量是opacity-volume代理，不是实际像素覆盖；父法向很薄会夸大标准化位移，不能将所有比值直接当作图像损伤因果。

后续改动应针对出生单元本身：保留粗覆盖核，局部细节与粗平均辐射解耦，约束子点在父支撑中成长，避免全部旧覆盖由独立自由子核接管。尚未有该机制的实测成绩。

## 验证与证据

全371项CPU测试通过。新native小GPU验证中，无界DC149.08被约束到1.77245、实际量化RGB约0.99931且MSE仍下降；梯度/H与有限差分、scene匹配、原参数身份和回退均通过。

- 输出：`output/flame_steak_1025/fs_bounded_intrinsic_v6_1_short3/20261026T031913_169e84`。
- 配置/源码：`tmp/bounded_intrinsic_v6_1_short3.yaml`、`tmp/experiments/bounded_intrinsic_v6_1_short3/source_manifest.json`。
- 解码/摘要：`tmp/birth_diagnosis/bounded_v6_1_decoded.json`、`bounded_v6_1_summary.json`同目录。
- 原生边界验证：`tmp/birth_radiance_bounded_native_validation.json`。
- v6最终类型：`tmp/birth_diagnosis/v6_final_type_audit.json`。
- v6族漂移：`tmp/birth_diagnosis/v6_family_drift_audit.json`。
