# 当前保留的实验

**特别重要：[34.0547dB / 38.19KB / 233FPS 完整300帧基线](experiment/20261017_all_rotation_off_full300.md)。** 对应Git标签为 `important-full300-34.0547db-38.19kb-20261017`；配置、指标口径和保留规则见[重要版本索引](IMPORTANT_BASELINES.md)。

最新完整序列为[所有旋转增量关闭＋直接α量化](experiment/20261017_all_rotation_off_full300.md)：**300帧34.054672dB，动态帧完整更新38.194930KB，纯渲染233.014FPS**。全部顺序解码、画质与实际文件大小独立核验通过；一次性参考帧另计。存储/FPS达标，PSNR尚未达到34.2，learnable gate的实际细分收益仍未体现。复现入口是`config/fs_alpha_norot_quant6_rate003_full300_20261017.yaml`，当前最完整的已验证候选，以新实验名从共享首帧开始。

最新完成[继承父预测的联合gate短测及原生早期](experiment/20261018_inherited_image_gate_first3.md)：210–212为33.912253dB/28.502333KB/243.068FPS，相比绝对区域联合配方提高0.013812dB、更新减少9.40%；但同代码原生1–3为34.075954dB/29.092KB，画质下降，未进入30/300帧。318项检查及两个窗口独立回放通过。后续优先解除“只有最细层GS才允许残差”的容量约束，让gate只控制共享细化。

此前[RMSE像素项的早晚短窗口](experiment/20261018_joint_dc_rmse_short_windows.md)：210–212为34.015923dB/50.906667KB/243.344FPS，但原生1–3为34.158596dB/66.971667KB，早期画质和存储均未改善，未进入30/300帧。310项检查及两窗口独立回放通过。后续优先诊断和改进learnable gate的实际共享与细分收益。

此前[gate与区域颜色联合学习](experiment/20261018_joint_regional_dc_first3.md)：210–212帧33.898441dB/31.460667KB/243.000FPS，相比原提案监督配方小幅提高0.016340dB、增加0.129KB。306项检查与三帧独立回放通过；随后[共享首帧起步的早期核验](experiment/20261018_joint_regional_dc_native_first3.md)动态1–3帧为34.183821dB/30.664667KB，仍低于同范围基线34.352098dB。未直接进入30/300帧。

最新完成[个体提案图像监督的原生前30帧核验](experiment/20261018_proposal_image_native_first30.md)：34.171395dB/21.346828KB/216.483FPS，相比无辅助监督提高0.022999dB，仍低于同范围基线34.276890dB/38.130724KB。全部29个动态帧独立回放通过，未进入完整300帧。随后[区域候选拟合短测](experiment/20261018_proposal_region_fit_first3.md)和[首动态帧校准诊断](experiment/20261018_bootstrap1_early_first3.md)均未显示足够收益；[固定gate后允许所有GS例外](experiment/20261018_fixed_gate_all_residuals_first3.md)三帧33.896425dB/77.638KB，存储显著回归，原型撤回。下一步检查区域颜色与gate的联合学习时序。

此前[未选中提案的图像监督短测](experiment/20261018_proposal_image_first3.md)：210–212帧33.882102dB/31.331667KB/243.097FPS，优于匹配完整基线33.826224dB/39.609333KB；相对原区域颜色配方提高0.018742dB、增加3.548333KB。297项检查与三帧独立回放通过，其原生前缀结果见上。

最新完成[区域共享颜色的原生前30帧核验](experiment/20261018_regional_dc_native_first30.md)：34.148397dB/19.775172KB/216.817FPS，同范围基线为34.276890dB/38.130724KB。存储降低但画质下降，未推广至完整300帧。全部29个动态帧独立顺序解码通过；新增前checkpoint审计显示raw-block估计高于实际压缩包。随后[减弱稀疏惩罚的短测](experiment/20261018_motion_quality_budget_first3.md)仅提高0.004951dB，字节增加9.774333KB，也未推广，下一步检查gate与个体提案的图像训练机制。

此前[学习区域上的共享颜色增量](experiment/20261018_regional_dc_first3.md)：210–212三帧33.863360dB/27.783333KB，超过匹配基线33.826224dB/39.609333KB；颜色包平均0.435333KB，全部负载均计入。290项全量检查、补充续训检查及三帧独立解码通过。晚期短测收益未在原生前缀重现。

此前[随机共享场与各级GS残差](experiment/20261018_stochastic_shared_residuals_first3.md)恢复任意选中区域下的GS残差，三帧33.763982dB/51.931333KB，低于匹配基线。281项检查和独立解码通过，未推广。

此前[固定gate后切换残差坐标与精修](experiment/20261018_fixed_gate_residual_refit_first3.md)保留原始码流，恢复共享叶子区域的图像梯度；20步精修为33.771741dB/27.370667KB，100步精修为33.772776dB/26.166333KB，均未胜过匹配基线。276项检查和两个三帧窗口独立解码通过，未启动新完整300帧。

此前[随机离散gate与候选反馈](experiment/20261017_stochastic_gate_first3.md)训练实际采样离散专家，STE只传选择梯度。210–212三帧为33.766490dB/26.619667KB；270项检查及独立回放通过，未推广。更低码率并不等于同画质收益已经成立。

此前[完整运动提案与预测器中心近端](experiment/20261017_private_motion_proposals_first3.md)让gate选择共享或个体完整运动提案，编码仅传相对区域预测的差。较好三帧为33.750975dB/30.278667KB，仍未推广。264项检查通过，三个窗口均独立解码；降低惩罚或加细量化都没有补回画质，连续场硬化的位移跳变明显大于量化误差。

此前[连续选择退火与候选反馈](experiment/20261017_soft_gate_self_feedback_first3.md)允许候选实际参与图像再转为离散gate，210–212三帧为33.813229dB/44.648333KB，匹配基线33.826224dB/39.609333KB；256项检查与独立回放通过，未推广。

此前[层级个体通路短测](experiment/20261017_hierarchical_private_gate.md)及[共享首帧起步30帧核验](experiment/20261017_hierarchical_private_native_first30.md)中，三帧较好候选为33.739380dB/31.467KB，但从共享首帧重新训练后，0–29帧为34.042090dB/91.428138KB，差于同范围基线34.276890dB/38.130724KB，未推广至完整300帧。v1冻结源码已通过250项检查。重要基线的精确源码另由上述Git标签保留。

此前[候选提案、关闭区域图像梯度与短测](experiment/20261017_gate_proposals_and_closed_candidate_gradients.md)修复关闭候选图像梯度为0，但其四个三帧候选均未超过匹配基线。更早的[gate与共享运动短测](experiment/20261017_gate_shared_motion_mechanism.md)保留候选训练和稀疏调度检查，属性测试作为辅助证据。

此前[同配方首30帧](experiment/20261017_all_rotation_off_first30.md)为0–29平均34.276890dB、动态更新38.130724KB。其同代码前缀已严格延长并完成上述300帧验证，首30帧成绩不能代替完整序列。

最新短测见 [α 空间量化与关闭旋转](experiment/20261017_alpha_quantization_translation_only.md)：相同旧269检查点续跑270–272，原近端基线32.426898 dB /107.623 KB；直接α＋连续收缩为32.664552 dB /83.314 KB，关闭区域和旧GS旋转为32.645459 dB /80.378 KB。随后平移网格改为6e-5，进一步冻结新点旋转并省去其朝向传输，得到32.654235 dB /71.344 KB /203.990纯渲染FPS。215项检查及三帧独立回放通过；这些270–272帧局部结果与上面的首帧重训300帧成绩分开记录。

历史局部基线为下面的 **GS近端稀疏方案，270–299帧平均33.466945 dB /87.874 KB**。最新完成 [相对区域＋关闭分支拟合短测](experiment/20261017_relative_region_gate_first3.md)：同一270–272帧窗口，基线32.426898 dB /107.623 KB，相对版32.406921 dB /114.020 KB，非零GS残差增加15.89%，有效分裂仍为13；**未作为默认配方**。201项检查与三帧独立回放通过，主配置恢复绝对区域＋近端基线；相对版由独立配置和开关保留。短测不代表完整300帧效果。

此前 [区域/gate 图像联合优化与实际码率校准](experiment/20261017_joint_image_codec_validation.md) 验证了另一种优化参数化：区域与gate接收图像梯度，运动码率采用真实字节与包长校准。192项检查及独立回放通过，但存储回归：帧1为34.3874dB /155.802KB，帧270为32.3780dB /151.022KB。该分支也没有替代近端基线，其历史完整配置保存在对应run和源码快照中。

此前训练改进是 [图像 Adam＋GS 近端稀疏优化，后 30 帧](experiment/20261017_proximal_gs_late270_299.md)：相同旧第 269 帧续跑，PSNR **33.2909 → 33.4669 dB**，完整更新 **151.406 → 87.874 KB，减少 41.96%**。GS 非零残差比例 **21.9% → 5.7%**，旋转残差平均 **36,681 → 687**；94.29% 的移动点无需逐 GS 残差。SSIM/LPIPS 略降，尚未达到完整 300 帧与 50 KB 目标。179 项测试、30 帧独立回放及原始状态检查通过；另完成 0.002 配方从共享首帧启动两个动态帧的核验。该历史配方使用 `proxgs_lossless` 实验名；没有重跑 300 帧。

此前完成 [无损编码接入及 gate/旋转核验](experiment/20261017_lossless_codec_gate_rotation.md)。对既有帧 270–299 通过正式编码器重新导出，完整动态更新 **188.492 → 151.406 KB，减少 19.68%**，连续解码的全部状态逐字节一致，175 项测试通过；未重跑 300 帧。发现大量旋转参数停在半 bin 门槛附近，三帧冻结检查显示其最小旋转残差收益很小。gate 在实际开合，76.53% 的移动点不需 GS 残差；但单帧 learned gate 相比全细分仅省约 0.58%，独立码率优势仍待改进。α＜0.01 删除保持不变。

此前三通道功能实验为 [平移/旋转/透明度联合量化与稀疏更新，后 30 帧续跑](experiment/20261017_sparse_tralpha_late270_299.md)。从旧第 269 帧继续训练 270–299，平均 **32.5682 → 33.2909 dB**，第 299 帧提高约 0.99 dB；每帧完整更新由 **131.257 增至 188.492 KB**。透明度平均新增 41.056 KB，运动包也变大，该历史配方属于**画质改善、存储回归**。171 项测试、全段独立回放与恢复检查通过；不是重新训练的完整 300 帧成绩。

该记录中的码流审计为接入前的离线结果：运动包占 74.74%、透明度占 21.78%，新增属性占 3.17%。离线布局探针为 151.404 KB；当前正式格式增加明确的原始/变换长度等字段后为 **151.406 KB**，并已完成正式导出与连续回放核验。

最新完整序列结果是 [STE gate、storage 0.002 / 0.003 的 300 帧对比](experiment/20261017_ste_storage002_vs003_full300.md)。两组均已训练完成并独立核验全部 300 帧。0.002 为 **33.0724 dB、127.338 KB/动态帧、199.494 纯渲染 FPS**；0.003 为 **33.2902 dB、119.698 KB/动态帧、199.820 纯渲染 FPS**。本次 0.003 的 PSNR 高 0.2178 dB，完整更新量少 6.00%，两组仍未达到画质与存储目标。

针对后段退化，已完成 [定向诊断](experiment/20261017_late_degradation_diagnosis.md)：第 299 帧保留新点从 327 增至 4705 只提高约 0.008 dB，却增加约 192 KB 负载；同点数、无量化/码率约束的透明度诊断可提高约 0.82 dB。历史实现缺少旧点替换，GS 非零块字节项对运动向量没有直接梯度，且压缩效率变化未被块计数充分反映。

此前完成 [ReCon-GS 式固定预算替换的前 30 帧核验](experiment/20261017_recongs_replacement_first30.md)：相同 29 动态帧平均 **34.2249 → 34.2612 dB，106.904 → 101.927 KB**。其按固定 α 排序的问题已在本次实验中改为学习当前 α，再删除低 α 点；不再强制新增 K 个就删除 K 个。固定预算结果保留为历史记录。

| 内容 | 记录 | 结果目录 |
|---|---|---|
| 相对区域与关闭候选拟合，270–272短测，未采用 | [相对表示与GS残差核验](experiment/20261017_relative_region_gate_first3.md) | `output/fs_relative_gate_validation_20261017/` |
| 图像联合优化、实际码率校准与残差诊断 | [联合优化与字节核验](experiment/20261017_joint_image_codec_validation.md) | `output/fs_joint_image_codec_validation_20261017/` |
| GS 近端稀疏优化，续跑 270–299；0.002 启动检查 | [残差优化与独立核验](experiment/20261017_proximal_gs_late270_299.md) | `output/fs_proximal_gs_validation_20261017/` |
| 新编码器重导出 270–299，gate/旋转诊断 | [编码与机制核验](experiment/20261017_lossless_codec_gate_rotation.md) | `output/fs_lossless_gate_rotation_validation_20261017/` |
| 三通道量化与稀疏更新，续跑 270–299 | [画质与实际存储核验](experiment/20261017_sparse_tralpha_late270_299.md) | `output/fs_sparse_tralpha_validation_20261017/` |
| ReCon-GS 式替换，前 30 帧 | [替换与回放核验](experiment/20261017_recongs_replacement_first30.md) | `output/fs_ste_replace_depth4_rate003_first30_20261017/` |
| STE、storage 0.002，完整 300 帧 | [0.002 记录](experiment/20261017_ste_storage002_full300.md) | `output/fs_ste_depth4_rate002_full300_20261017/` |
| STE、storage 0.003，完整 300 帧 | [0.003 记录](experiment/20261017_ste_storage003_full300.md) | `output/fs_ste_depth4_rate003_full300_20261017/` |
| 此前 350 W 稳定性训练，30 帧 | [30 帧记录](experiment/20261016_gpu_350w_training_first30.md) | `output/fs_gpu_stability_350w_worldroute_first30_20261016/` |
| 此前长序列退化检查，实际完成 200 帧 | [200 帧记录](experiment/20261015_spatial_gate_worldroute200_stopped200.md) | `output/fs_spatial_gate_worldroute_200_full300_20261015/` |
| 稳健树初始化 CPU 审计 | [初始化审计](experiment/20261016_spatial_gate_robust_initialization_cpu.md) | `output/fs_spatial_gate_initialization_audit_20261016/` |

完整更新量含每帧 manifest，KB=1000 B，一次性首帧参考 17720236 B 另计。纯渲染 FPS 不含解码、验证和上传。逐帧 CSV、对比曲线和两次 benchmark 见完整对比记录；[轻量证据归档](archive/ste_storage002_003_full300_20261017.tar.gz) 保存固定源码与结果文本，原始训练产物仍在本地 output。

本轮实现提速核验单列为 [训练冗余与提速记录](experiment/20261016_training_redundancy_speed.md)：这是单帧执行速度检查，不是新增的全序列画质结果。

基础配置为 `config/fs_spatial_gate_depth4_200_full300_20261016.yaml`：最大深度4、storage权重0.002、robust_median初始化、远点连最近非空叶区域；首个动态帧细分gate全关，后续继承前帧gate。前70步联合优化总运动与共享表示，后30步只固定gate；图像拟合总GS运动，区域和gate通过候选残差及码率目标学习。运动每步2视角，新增GS再训练100步、每步1视角。开启opacity更新、可微稀疏支持和GS近端收缩；`region_motion: absolute`、`joint_image_gradients: false`、`rate_model: raw_blocks`恢复87.9KB基线的优化与罚价口径，最终仍用生产无损编码器导出。当前α＜0.01删点，旧opacity logit步长0.05，固定数量替换关闭。两份入口使用新`fs_proximal_gs_rate00x_full300_20261017`实验名，0.003入口仍为 `config/fs_ste_sparse_tralpha_rate003_full300_20261017.yaml`，其运动配置与已验证近端基线逐字段一致。相对实验入口为 `config/fs_relative_gate_prox_rate003_20261017.yaml`。未重新训练完整300帧。

此前固定预算替换入口 `config/fs_ste_replace_depth4_rate003_full300_20261017.yaml` 保持其历史设置，可用于读取和复现对应短测。

历史完整序列入口为 `config/fs_ste_depth4_rate002_full300_20261017.yaml` 和 `config/fs_ste_depth4_rate003_full300_20261017.yaml`，两者保持替换关闭，除实验名及权重外设置相同。两次执行同一固定源码，运行来源保存在 `output/fs_ste_rate_sweep_full300_20261017/`。

此前 [learnable gate 的 STE 与优化坐标核验](experiment/20261017_learnable_gate_ste_first6.md) 为共享首帧加五个动态帧：区域数 8、8、287、286、293，动态平均 34.4726 dB、完整更新 126.10 KB。它只用于短测机制检查；完整 300 帧结果已如上，不可用短测均值替代。

此前 [全关初始化短测](experiment/20261016_joint70_closedfirst_2view_first3.md) 为两动态帧平均 34.4233 dB、137.81 KB；[全开初始化短测](experiment/20261016_joint70_robustnearest_2view_first3.md) 为两动态帧平均 34.4122 dB、139.72 KB。这些是 STE/优化坐标修复之前的结果，不可与五帧均值直接比较。

此前保留的 30 帧和 200 帧训练结果来自五层、0.03 配方，不能当作当前配置的结果。初始化 CPU 审计只覆盖建树候选本身，与训练结果分别记录。

`config/` 保留近端基线入口、可选相对区域入口以及此前实验配置。历史200帧续训入口使用保留的最后检查点和新experiment_name；历史run内的原配置保持不变。

演示视频为 `output/4dgs_flame_steak_demo_1920x1080_20261016_122523.mp4`，同名 `.manifest.json` 保存来源帧摘要和视频核验信息。来源链中的旧训练目录已清理，成片仍可直接播放。

2026-10-16 清理前的实验记录、YAML 和轻量诊断文本保存在 `archive/history_before_20261016_cleanup.tar.gz`。归档不包含已删除的原始训练产物；删除清单、保留项和空间统计见 `archive/cleanup_20261016.json`。例如读取旧联合训练记录：

```bash
tar -xOf docs/archive/history_before_20261016_cleanup.tar.gz docs/experiment/20261014_spatial_gate_joint_first5.md
```
