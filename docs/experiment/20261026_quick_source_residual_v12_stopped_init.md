# 单模型300步残差出生：初始化性能中止记录

日期：2026-10-26。本次只完成第一动态帧的200步运动，停在出生时的一次解析初始化，尚未进入100步出生更新，未产生完整帧或质量结果。进程SIGINT退出130，总运行约8分38秒。不是达标实验，不能报告该方法PSNR/存储/FPS。

## 核心机制

按用户最新约束：一套模型、全部20训练视角、总200运动＋100出生；无双分支、额外refit或隐藏SGD循环。新核从原GS完整支持域生出两个缩小子核，母材质θ0和原SH1保持，子核训练DC/几何/透明度；出生前50步保护密度，后50步竞争。出生时用真实native响应做一次2×2有界DC初始化，包含同族两个子核的交叉项；忽略跨族H块，以一次全场景实际codec MSE检查决定保留或回滚，初始化不调用optimizer。最终同一训练池按all20真实codec增量逐族接收/拒绝。

模型/控制器/initializer与300账本已验证，529项完整CPU＋最终5项控制器通过；小原生测试确认DC已知目标、固定母与SHrest、真实ATTR回环和零初始化SGD。实际全尺寸初始化却非常慢，因此中止后修复实现，保留旧冻结代码/输出用于追踪。

## 实际问题与修复依据

瓶颈是FP64的`(3,2,1355760)`极瘦矩阵乘法，用来计算每对核的H和g。RTX3090Ti真实shape微基准：原H＋g调用中位0.38772秒，等价逐项乘积/求和0.000731秒，约530倍；分块求和0.001045秒，略慢。H相对差5.86e−14，g相对差4.87e−15。按20视图、约37个family批次、3RGB估算，仅原归约就约14.35分钟，与本次初始化超时量级一致。这个判断来自实测，不是仅凭中断stack猜测。

后续仅替换这段等价计算，不改变模型、数据、解析目标、训练参数、300总步或单模型约束，不增加SGD。新实验使用独立名称继续同一方法，不是训练对照/参数扫描。需要真实运行后另记质量结果。

## 产物

- 中止运行：`output/flame_steak_1025/fs_quick_source_residual_v12_short3/20261026T092415_ce0846`。
- 冻结108文件：`tmp/experiments/quick_source_residual_v12_short3`。
- 日志：`tmp/birth_diagnosis/quick_source_residual_v12_short3_train.log`。
- 微基准：`tmp/birth_diagnosis/child_dc_reduction_microbench.json`。
- 初始原生验证：`tmp/birth_diagnosis/child_dc_initializer_native_validation.json`。
- 单模型300约束审查：`tmp/birth_diagnosis/quick_v12_independent_hard_constraint_review.json`。
