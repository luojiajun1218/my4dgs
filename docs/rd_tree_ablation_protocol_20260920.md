# 树包络方法消融协议（主实验运行期间预先记录）

主实验：`flame_rd_tree_envelope_full100_20260920/20260920T020653_189351`。
本文件在长序列结果尚未完成时建立，不表示以下消融已经执行。

## 冻结条件

共享frame0、100总帧、cam00测试、同一20相机训练池、controlled seed0，
80运动+100增量反传，4视角累积。候选树depth16/min_leaf12/父内批量二聚类；
量化、rho、步长、增量高斯训练及保存协议全部保持主实验设置。
每个运行使用独立experiment_name并存配置与源码快照。
主实验运行时不改训练代码，不并发GPU实验以免污染wall比较。

## 两个核心因果问题

1. bit loss：A为主实验；B仅将normalized_rate从1e-4改为0。
   B配置 `config/flame_rd_tree_nobit_full100_20260920.yaml` 已于03:10启动，运行
   `output/flame_rd_tree_nobit_full100_20260920/20260920T031009_8627ce`。
   启动前完整数值测试通过，配置diff仅experiment_name和normalized_rate。
   B仍采用相同量化和包络commitment；不改为不量化latent渲染。
   B的零码长权重不意味着一定保留全部候选分叉（无失真收益的平局仍可能停止）。
   因而B不能同时充当“固定层级”消融。

2. 局部深度选择：C在相同候选字典内强制几何深度13的cut，D强制到候选深度16，
   自然提前终止的几何叶不再强分。保留相同系数率失真量化、实际packet格式、
   lambda、包络commitment和训练更新。C/D尚未实现，不能用减小max_depth后
   仍运行自适应DP冒充固定层级。13取自先前小容量实验，16为主实验字典上限，
   不根据cam00挑cut。所有cut都必须保持祖先闭包。

固定cut的系数仍可量化为零，但结构不因图像梯度或率失真比较而剪枝；需要
测试真实编码flags保留该cut、packet精确解码、参考与优化路径一致。
仅比较A/B可以说明bit项影响；只有A/C/D才讨论深度分配的收益。

## 报告与解释

报告相同99动态帧的PSNR/SSIM、每帧及分段差值、实际运动包+协议+增量PLY
字节、完整wall、独立解码成本、分叉/叶深度分布。共享首帧与checkpoint另列，
不混入运动流，也不省略增量高斯成本。训练视角序列必须逐项核验。

消融优先保持相同反传和优化调度，记录wall差异；不能自动称为严格等wall。
固定cut与自适应的不同码率必须同时展示，不能只选一个PSNR较好的点。
先完成seed0机制对照，再对有价值的候选用额外固定种子重复；不得把重复中
最好的一次当均值。若主实验长序列失败，保留失败并分析原因，不据cam00
在线改正在运行的参数，也不以短帧结果代替长序列目标。

目前候选边界由几何规则确定。以上实验即使成功，也只支持局部深度/自由度
自适应，不支持任意运动边界已学得，更不单凭此证明论文创新性。

## 运行审计补充

新增只读 `scripts/audit_rd_tree_run.py`，避免旧通用审计对motion.npz的假定。
使用 `--same-schedule-reference` 比较逐条训练frame/step/path，核对21视角
有限评估、wall、源码zip及manifest、RDP1头/有效bit/字节对齐、协议与配置、
实际motion+protocol+incremental字节、更新/投影数。`--through` 明确仅审计
已完成前缀，不宣称整序列通过。它不重建运动，独立解码仍需stream audit。
零bit运行前4动态帧已通过上述检查，训练视角与主实验逐项一致。
该工具在运行期间新增，未改任何正在训练的pipeline代码。
另外比较两次运行已存config和controlled_manifest：配置差异恰好为实验名
及normalized_rate；共同源码文件hash无任何差异，未删除文件，仅新增未被
正式训练导入的rd_latent_acceptance.py和rd_tree_basis.py两个诊断模块。

## 固定cut参考实现准备

`scripts/fixed_cut_reference.py` 是独立慢速CPU oracle，未被训练导入。
保留完整候选树及RDP1编码，强制depth以内的所有可分节点分裂（根深度0），
即使所有差异系数为零也保留分叉和相应码长；自然叶提前结束。
系数依旧按同一rho、量化步长和真实符号码率优化。
`tests/test_fixed_cut_reference.py` 已单独通过：不平衡树各深度结构、零系数
不剪枝、真实包精确往返、自适应二次代理目标不高于任何固定深度cut、零bit
时全深度与自适应最优目标一致。测试也已加入完整测试入口。
尚未接入正式RDTreeMotion或执行固定cut训练，因此不构成消融结果。

后续实施更新：fixed_cut_depth现已作为可选配置接入RDTreeMotion和投影器。
默认None仍为自适应；固定cut时仅替换结构DP的stop/split决定，所有系数率失真
量化、候选字典及编码规则不变。根深度0，depth以内强制分裂，自然叶停止；
大于候选深度等同保留全部候选分叉。四种统计/符号向量化组合均与独立oracle
逐字节一致，包含零target、大rate、不平衡树、depth0及超过树深度的情况。
旧checkpoint缺失此字段按None解释，但固定深度不匹配仍拒绝恢复。训练状态
恢复及独立编码解码测试通过，完整test_nested_prototype入口通过。
配置fixed13/fixed16均已准备，diff仅实验名和新增fixed_cut_depth；正式完整
fixed13已启动，fixed16尚未启动。不得把配置存在当作实验完成。
fixed13实际运行目录：
`output/flame_rd_tree_fixed13_full100_20260920/20260920T034544_0d832f`。

零bit独立连续解码完成，报告
`output/idea_rd_tree_decode_audit_20260920T034405_237020/report.json`。
99包从共享frame0连续重建；frame10/20/.../90/99所有已存checkpoint的六项
基础属性及增量高斯精确匹配，未存checkpoint的帧仅解码、不宣称逐属性验证。
整个审计255.1798秒，当前运行时/设备/顺序协议，不代表跨平台bit-exact。

固定13训练及99帧连续解码均已完成；结果见方法搜索记录第二十轮。
独立解码报告 `output/idea_rd_tree_decode_audit_20260920T041923_284240/report.json`，
耗时239.6962秒；10/20/.../90/99已存检查点基础六属性与增量高斯均精确匹配。
固定16配置与固定13仅实验名和fixed_cut_depth不同；再次完整数值测试通过后
已启动固定16训练。固定16尚未有完整结果，重复仍未执行。
固定16实际目录：
`output/flame_rd_tree_fixed16_full100_20260920/20260920T041948_54d701`。

固定16完整训练与独立连续解码均已完成。完整结果见搜索记录第二十一轮；
独立解码报告 `output/idea_rd_tree_decode_audit_20260920T045348_649013/report.json`，
99包连续重建，所有已存检查点精确匹配，审计253.8140秒。至此A/B/C/D各完成
一次seed0完整序列；必要重复尚未完成，不能声称统计稳定或自适应优势成立。

## 延迟选择与预定重复（当前完整延迟运行尚未结束时记录）

E仅增加structure_warmup_updates=20，其余参数与A一致：训练阶段用完整候选
深度，最后运动更新后恢复自适应结构；随后场景应用、增量训练及实际码流
均使用自适应结果。两动态帧接线与独立连续解码检查已通过，完整seed0运行
`output/flame_rd_tree_delayed_full100_20260920/20260920T050341_d26626`
正在进行。前6动态帧已通过只读协议审计，这不是完整99帧验证。

预先固定下一组重复为A/D/E各seed1，不按E的最终cam00分数决定是否报告：

- `config/flame_rd_tree_envelope_seed1_full100_20260920.yaml`
- `config/flame_rd_tree_fixed16_seed1_full100_20260920.yaml`
- `config/flame_rd_tree_delayed_seed1_full100_20260920.yaml`

这些配置只改实验名；随机种子由run_controlled.py的--seed 1设置，不能
只把seed1写入名字就宣称重复。共享首帧不变，因此是动态训练随机性的重复，
不是重新训练首帧的独立复现。运行串行，启动前确认GPU空闲并运行完整测试。
每组均完成99动态帧及码流审计，再按种子配对报告E-A、E-D，列出每个种子
完整均值及差值；不把99个相关视频帧当成99次独立实验。两个种子仍不足以
支持强统计显著性主张；若结果冲突，保留冲突并追加诊断，不报告最好值。
其中E-D主要检验延迟剪枝是否保持完整层级训练的质量并降低实际码流；
E-A检验训练期间较早选择结构的影响。E不能替代无bit消融B，也不能单凭
后处理剪枝成功就声称图像端到端自适应训练收益；bit的重复仍另行待办。

当前仅准备配置，尚未启动以上seed1运行。

后续状态：E的seed0经历进程退出，在新目录从60恢复，已形成明确来源的
分段99动态帧结果并完成独立连续解码。恢复重算61与原轨迹不精确一致，
故它不是不中断seed0复现；详见研究日志第二十五轮，保留限制。
固定16三个checkpoint的无损零子树对照也已完成，不能将固定cut中的
全部结构冗余视作自适应方法收益，详见第二十六轮。

完整test_nested_prototype及分段组装测试再次通过、GPU确认空闲后，已使用
`--seed 1`启动A的预定100帧重复配置。D/E的seed1仍待串行执行；未基于
种子0测试结果删去某一预定组。当前不修改训练源码。

A seed1实际目录
`output/flame_rd_tree_envelope_seed1_full100_20260920/20260920T093426_4598cf`。
保存配置相对A seed0仅实验名及新增默认structure_warmup_updates=0；manifest
确认seed0->1。源码并非逐文件相同：rd_tree_motion/projection后来加入固定cut
和warmup开关，另新增两个未被正式训练导入的诊断模块。已逐行核对diff，
并从A seed0的校验过hash的zip加载旧motion/projection/codec到隔离模块，
用4组确定性CPU六维梯度流（seed6100..6103，点数63/71/79/87，父内批量
二聚类，depth5/min2，80视角/20更新，包括零bit及原bit）对照当前默认路径：
每步latent逐值相同、所有重算packet逐字节相同。该回归支持新增关闭的开关
未改变被测默认更新，不证明完整CUDA渲染训练确定性或所有输入等价。

A seed1完整训练及独立99帧解码均已完成；结果见研究日志第二十七轮。
D seed0完整无损去冗余对照也完成，运动包减少27.2311%、总动态流减少
11.5757%，所有已存checkpoint精确匹配；须单独报告额外重编码开销。
完整数值测试与分段组装测试再次通过、GPU空闲后，已用--seed 1启动
`config/flame_rd_tree_fixed16_seed1_full100_20260920.yaml`；E seed1仍待执行。

D seed1实际目录
`output/flame_rd_tree_fixed16_seed1_full100_20260920/20260920T101919_bd1aae`。
与A seed1保存配置逐项对照，差异仅experiment_name和fixed_cut_depth None->16；
manifest均seed1且全部pipeline源码hash完全相同。第一动态帧完整视角序列、
评估、快照及码流计数审计通过；这仅是前缀检查，完整配对尚待训练结束。

后续：D1完整训练与99帧独立精简解码均完成（研究日志第二十九/三十轮）。
E1现运行于
`output/flame_rd_tree_delayed_seed1_full100_20260920/20260920T110007_d26755`，
前30帧审计通过。直接对照A1/D1/E1的保存配置和controlled_manifest：
三者seed均1，全部pipeline源码hash逐项相同；相对A1，D1仅实验名及
fixed_cut_depth None->16不同，E1仅实验名及structure_warmup_updates 0->20
不同。这排除了已记录配置/源码的其他变化，不证明CUDA训练本身确定性。
