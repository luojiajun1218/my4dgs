"""独立训练指定固定区域深度，供自适应深度实验对照。

用环境变量 FIXED_SPATIAL_DEPTH=1..5 选择深度，在运行时固定区域划分并冻结
gate；仍使用同一五层候选树及现有训练、损失和码流流程拟合运动与属性。
本文件负责固定深度训练入口，不汇总结果；汇总由 ablation_fixed_depth.py 负责。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from absl import app
from pipeline.model.spatial_region_gate import spatial_owner
from pipeline.model.spatial_motion_training import SpatialMotionTraining
import main

depth = int(os.environ['FIXED_SPATIAL_DEPTH'])
if not 1 <= depth <= 5:
    raise ValueError('FIXED_SPATIAL_DEPTH must be in 1..5')

original_begin = SpatialMotionTraining._begin_joint_frame
original_prepare = SpatialMotionTraining.prepare_plan_step
original_select = SpatialMotionTraining._select_frontier


def probabilities(self):
    result = torch.zeros_like(self.gate_logits)
    if depth > 1:
        result[self.spatial_paths[:, :depth-1].reshape(-1)] = 1
    return result


def sampled_weights(self, *, sample_if_needed=True):
    return self._path_weights(probabilities(self))


def verify_frontier(self):
    if not torch.equal(self.owner, self.spatial_paths[:, depth-1]):
        raise RuntimeError(f'Fixed depth {depth} frontier changed at step {self.last_step}')


@torch.no_grad()
def begin(self, proposal, previous_regions, previous_residuals):
    if self.spatial_paths.shape[1] != 5:
        raise ValueError('Comparison requires the shared depth-five candidate tree')
    self.opened = probabilities(self).bool()
    self.opened[0] = True
    self.owner = spatial_owner(self.spatial_paths, self.opened)
    original_begin(self, proposal, previous_regions, previous_residuals)
    verify_frontier(self)


@torch.no_grad()
def prepare(self, step):
    original_prepare(self, step)
    self.gate_logits.requires_grad_(False)
    self.gate_logits.grad = None
    if step in (1, 100, 101, 151, 180, 200):
        verify_frontier(self)


@torch.no_grad()
def select(self, *args, **kwargs):
    result = original_select(self, *args, **kwargs)
    if self.last_step in (1, 100, 151, 180, 200):
        verify_frontier(self)
    return result


SpatialMotionTraining._gate_probabilities = probabilities
SpatialMotionTraining._sampled_path_weights = sampled_weights
SpatialMotionTraining._begin_joint_frame = begin
SpatialMotionTraining.prepare_plan_step = prepare
SpatialMotionTraining._select_frontier = select

if __name__ == '__main__':
    print(f'Fixed spatial depth: {depth}; independent frame-by-frame training', flush=True)
    app.run(main.main)
