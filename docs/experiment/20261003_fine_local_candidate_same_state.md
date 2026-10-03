# fine SHARE→局部SPLIT的真实候选拟合

## 先解决什么

正式20train/test00的299帧结果为32.84089 dB，REF33.94978，未达标。1196组采样中局部参数梯度均0；局部候选未训练。这轮只打通这个分支的拟合信号，不把它称为全部画质缺口的唯一原因。

## 单变量与预算

新增YAML布尔参数fit_fine_split_candidates，默认false。true时，把可达fine SHARE节点的GS-local SPLIT候选并入已有KEEP→SHARE辅助位姿。两类终端GS不重叠，沿用同一辅助渲染、interval10/weight1，最多14图/帧。主硬gate/位姿不变；gate、父区域、已打开局部参数及非候选局部参数detach。只有关闭的局部候选参数获得这部分图像梯度。

源码改动只涉及QuantizedResidualConfig校验与候选位姿构建。新增确定性检查：关闭局部梯度非零、可比较候选位姿、主硬SHARE/KEEP不变、非候选/active/gate梯度隔离、checkpoint恢复及14图预算。

## 同状态短对照

共同来源为正式run的ckpt-5-200.pth。两份replay输入的81份张量（含optimizer）及其余checkpoint状态完全一致（rtol0/atol0）。严格resume要求配置一致，输入checkpoint只写入各组fit_fine_split_candidates布尔元数据；因此文件SHA不同，不能称文件逐字节相同。源checkpoint只读。

回放6–15帧、每帧133+67基础步。20个训练相机cam01–20，cam00仅测试，无独立validation。control=false，fitted=true；其余候选权重、storage/GS activity .001、motion_init_decay0、相机seed0全部固定。实际200步相机轨迹须逐项一致。

预先训练侧判据：局部候选梯度和非相同位姿出现；五个固定训练探针最终image objective相对control平均下降；KEEP违规0、状态有限、每帧辅助图不超过14。报告配对逐帧图像误差、门开启、字节和完整耗时。不按cam00分数决定推广；局部没开不强制打开。短实验满足判据也不能直接宣称长程34dB或自适应优于固定层级。

测试和运行结果待补充。

## 实际结果

完整86项数值测试通过。默认沙箱限制DataLoader worker socket，随后正式require_escalated/auto-review主机CPU执行通过；GPU训练control/fitted均正常exit0，已确认GPU释放。control输出20261003T192821_f6571b，fitted输出20261003T193202_8c739d。输入来源SHA256 a60aedcf737930df3086b02239786b55bc9de725063789fdc6280f4860ca3c44。

两组2000次实际训练相机序列逐项一致，均含全部20训练相机；cam00仅测试，辅助图均140/2000基础图。8/10帧训练图像objective改善；平均0.013134427→0.013062907，差-0.000071521，下降0.545%。训练探针PSNR 36.72297→36.82304，增加0.10007 dB。

局部非零梯度采样0/40→38/40，硬local开启平均0.08633%；没有强制开gate。frame15控制局部量化非零行0，fitted2128。剩余fine SHARE只读提升SPLIT时，971个位置出现超过1e−6差异，最大位移0.00620044，旋转最大元素差0.00252745；控制只有2.39e−7归一化舍入。已经有不同局部候选，而不再只是未拟合的相同位姿。

有效参数估算12255.30→16785.60 B/帧（+36.97%）；含状态更新循环16.84201→16.88895s，整帧19.88539→20.00415s。辅助图数未增加。KEEP违规0，endpoint状态有限。

被动测试cam00均值34.09843→34.12086 dB（+0.02243），不用于推广判断。优势仍小，只有单seed短对照且CUDA可能非确定；不是完整300帧质量，也不是自适应对固定层级消融。

## 后续配置

预写训练侧判据通过；下一项完整20train/test00配置已准备：`config/fs_core_fine_candidate_refaligned_seed0_300f_20261003.yaml`。从共享native frame0启动1–299，保存新独立实验目录，不从短实验中挑最佳testcheckpoint。当前尚未启动该长程，短对照结果与成本供父任务立即接续。

```bash
conda activate ReConGS
python main.py --config=config/fs_core_fine_candidate_refaligned_seed0_300f_20261003.yaml
```

最终核查两份endpoint各81个张量全部有限，7份训练关键源码/配置在两组之间保持hash冻结。下一长程YAML校验通过。短实验源码快照、输入状态manifest、完整86测试日志和结果JSON均保存在fitted run中，可继续复核。

补充完整训练轨迹：每组2000次变化中优化渲染，每个训练相机100次。control/fitted平均image loss .01367337244/.01361693795；total loss .01470169257/.01464210161（包括候选/storage/GS成本）；RGB loss .00994879779/.00990671569；DSSIM loss .02857166988/.02845782581。记录为training_all20_loss.json，不能当20相机固定终态评估。最终短程test00均值34.098431/34.120857；实际变化GS3.86565%/3.83170%，真实local门0%/.086327%；含状态更新训练循环16.84201/16.88895s，wall19.88539/20.00415s。一次长程已按父任务明确授权启动，详见20261003_fine_local_candidate_continuous.md。

最后frame15的step200测试cam00：control 34.083488 dB，fitted 34.118431，差+0.034943；这与6–15逐帧终评均值不同。frame15真实local门control0/fitted0.11954%，实际变化旧GS3.29588%/3.43913%。
