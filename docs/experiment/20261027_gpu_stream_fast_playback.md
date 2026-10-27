# 并行稀疏播放：完整300帧核验与独占测量

播放器源码：`5e41e4d4e4471327c582980c3203c4d2209f16fb`，目录`/home/gamer-02/Documents/DeepLearning/my4dgs/tmp/research_motion_20261027/play_fast_source`。实现与已完成检查见`20261027_gpu_stream_fast.md`。

两个完整实验均逐帧独立核验raw/状态、原manifest、渲染输入、实际cam00像素与保存PSNR；然后独占GPU，对全部299次动态更新连续计时三轮。

| 序列 | 全300帧PSNR | 完整动态KB/帧 | 原播放器FPS | 本版FPS |
|---|---:|---:|---:|---:|
| v24 | 34.094340 | 60.808809 | 82.470 | 225.720 |
| v25 | 34.098095 | 59.683759 | 85.396 | 238.753 |

KB=1000B，首帧17,720,236B另计。本版只改变解析调度与路由并行方式，码流和画质完全相同。速度变化属于运行时实现，不能归因为新训练方法或压缩收益。

测试显卡为RTX3090 Ti，PyTorch2.5.1+cu121。v24三轮分别为218.047、223.631、236.242 FPS；v25为237.577、239.876、238.816 FPS。表中汇总FPS按总帧数除以三轮总耗时计算。

计时包括文件读取/解压/校验和、两个CPU解析进程与进程传输、顺序属性预测、上传、GPU更新和实际完整渲染。只预取最多两个未来稀疏包。参考包、JIT、worker启动、相机/GT和完整预检为启动成本，单列在外；末帧状态/raw/像素每轮重新核对。

两个完整序列均超过200FPS。该测量未训练新模型，也不构成多场景SOTA结论。

- v24：`/home/gamer-02/Documents/DeepLearning/my4dgs/tmp/research_motion_20261027/gpu_fast_v24_verified.json`；`/home/gamer-02/Documents/DeepLearning/my4dgs/tmp/research_motion_20261027/gpu_fast_v24_playback.json`。
- v25：`/home/gamer-02/Documents/DeepLearning/my4dgs/tmp/research_motion_20261027/gpu_fast_v25_verified.json`；`/home/gamer-02/Documents/DeepLearning/my4dgs/tmp/research_motion_20261027/gpu_fast_v25_playback.json`。
