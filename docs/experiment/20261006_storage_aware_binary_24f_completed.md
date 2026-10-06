# 存储感知二叉运动编码：24帧完成记录

本轮将二叉层级运动、停止/细化门和逐节点精度选择写成可顺序解码的二进制流。目标是降低运动文件大小，同时保持重建画质。

## 配置与实现

- 配置：[fs_storage_aware_binary_24f_20261006.yaml](../../config/fs_storage_aware_binary_24f_20261006.yaml)。共享首帧 checkpoint，24个动态帧，`resolution: -1`，实际评测尺寸1600×1199；使用数据集原生划分，cam01–20训练、cam00测试，seed=0。
- 树使用96 GS粗区、至少12 GS子区、最多4层；第1帧建树，之后每5帧重建。运动精度候选为8/12/16/24/32 bit，位移和旋转步长逐档细化。
- 每帧300步：50步只拟合粗层，50步开放子层，80步联合优化运动/门/精度，20步固定选择后精修，100步拟合补点。Storage权重0.01，从第100步预热、20步升满；门和精度学习率0.02，初始开门概率0.5、初始最高精度。
- 新增 [storage_aware_binary.py](../../pipeline/ReCon-GS/model/storage_aware_binary.py)，按树的前序遍历写入运动存在位、精度选择和量化符号。删点及补点归属由解码器复现，不发送逐GS归属、节点或删点编号。`module.py`、`trainer.py` 接入训练、精确文件成本、报告和顺序解码；测试覆盖训练阶段、STE梯度、码率、树重建、补点和删点。

## 结果

24个动态帧的均值：

| cam00 PSNR | SSIM | 必要推理文件 | 渲染 FPS | 纯训练/帧 | wall/帧 |
|---:|---:|---:|---:|---:|---:|
| 34.2733 dB | 0.972447 | 49,943 B | 134.96 | 33.79 s | 39.92 s |

必要推理文件计入二进制运动流及补点PLY，包含流头，不含共享首帧、checkpoint和诊断JSON。硬成本与文件实测逐帧一致。平均每帧新增9.13个GS、删除8.17个GS。

精度选择共52,671条非零运动记录：32 bit占81.6%，12 bit占13.1%，16 bit占4.1%，8 bit占1.2%，24 bit没有被选中。较多节点保留最高精度，有助于画质；同时说明当前率损失下精度选择偏保守。

上一版二叉原型可作背景参照：PSNR为34.2724 dB，必要文件为1.887 MB/帧。本轮PSNR近似持平，必要文件约缩小37.8倍；SSIM低约0.00019。两轮不是控制变量实验，因此这里只记录整体结果，不将差异归因于单一参数。

可能的主要原因是确定性重建树和删点过程省去了上一版按GS保存的归属与编号表；本轮文件成本主要由量化运动符号和补点属性构成。最高精度档占多数，视觉误差没有随文件缩小而明显上升。24 bit未被选中，后续若需要进一步压缩，应检查精度率损失权重和精度分布，再确认画质影响。

## 核验与复现

- `PYTHONUTF8=1 conda run -n ReConGS python tests/test_nested_prototype.py`：151项数值检查通过。
- 顺序解码审计覆盖24帧，逐帧状态摘要全部匹配；第5、10、24帧checkpoint的六类原始属性逐元素一致。树重建帧为1、6、11、16、21；清单不含逐行GS或节点编号。
- 运行目录：`output/fs_storage_aware_binary_24f_20261006_retry01/20261006T213721_3906a9/`。逐帧汇总和曲线在其`analysis_completed/`，解码审计在`audit/sequential_001_024.json`。
- 数据继续通过本地软链接 `data/colmap4d/flame_steak` 读取，原始数据与历史输出未复制或改写。

```bash
conda activate ReConGS
PYTHONUTF8=1 python main.py --config=config/fs_storage_aware_binary_24f_20261006.yaml
PYTHONUTF8=1 python tests/test_nested_prototype.py
```
