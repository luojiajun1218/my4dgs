"""Batched shadow-motion exploration for closed binary frontiers."""

import copy
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch


@dataclass
class ShadowExplorationConfig:
    start_step: int
    end_step: int
    interval: int
    max_backward_per_frame: int
    levels: list
    seed: int
    camera_policy: str
    frontier_policy: str
    descendant_policy: str
    precision_policy: str
    loss: str
    translation_lr: float
    rotation_lr: float
    adam_betas: list
    adam_eps: float
    fraction: float = 1.0

    def validate(self, motion_config):
        for name in ('start_step', 'end_step', 'interval', 'max_backward_per_frame', 'seed'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.start_step > self.end_step or self.interval < 1:
            raise ValueError('Shadow step interval is invalid')
        steps = list(range(self.start_step, self.end_step + 1, self.interval))
        if len(steps) != self.max_backward_per_frame:
            raise ValueError('Shadow schedule must match max_backward_per_frame exactly')
        gate_start = motion_config.dense_fit_steps + 1
        gate_end = motion_config.dense_fit_steps + motion_config.gate_compress_steps
        if self.start_step < gate_start or self.end_step > gate_end:
            raise ValueError('Shadow schedule must stay inside gate_compress')
        expected_levels = list(range(motion_config.max_motion_levels - 1))
        if self.levels != expected_levels:
            raise ValueError(f'Shadow levels must be {expected_levels}')
        required = {
            'camera_policy': 'seeded_balanced_cycle',
            'descendant_policy': 'preserve_source_hard',
            'precision_policy': 'current_argmax_detached',
            'loss': 'image_only',
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f'{name} must be {expected!r}')
        policies = {
            'reachable_closed_same_level_all': 1.0,
            'reachable_closed_same_level_uniform_without_replacement': None,
        }
        if self.frontier_policy not in policies:
            raise ValueError('Unsupported shadow frontier_policy')
        if (isinstance(self.fraction, bool) or not isinstance(self.fraction, (int, float))
                or not math.isfinite(self.fraction) or not 0 < self.fraction <= 1):
            raise ValueError('fraction must be finite and in (0, 1]')
        required_fraction = policies[self.frontier_policy]
        if required_fraction is not None and self.fraction != required_fraction:
            raise ValueError('The all-frontier policy requires fraction=1')
        for name in ('translation_lr', 'rotation_lr', 'adam_eps'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        if (len(self.adam_betas) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or value >= 1
                for value in self.adam_betas)):
            raise ValueError('adam_betas must contain two finite values in [0, 1)')
        expected = dict(translation_lr=motion_config.region_translation_lr,
                        rotation_lr=motion_config.region_rotation_lr,
                        adam_betas=list(motion_config.adam_betas),
                        adam_eps=motion_config.adam_eps)
        for name, value in expected.items():
            actual = getattr(self, name)
            if actual != value:
                raise ValueError(f'Shadow {name} must match source motion config')
        return self

    def steps(self):
        return list(range(self.start_step, self.end_step + 1, self.interval))

    def level_for_step(self, step):
        steps = self.steps()
        if step not in steps:
            raise ValueError('Requested step is not a shadow event')
        return self.levels[steps.index(step) % len(self.levels)]


class ShadowMotionExplorer:
    """Own an isolated Adam while updating only newly reachable motion rows."""

    def __init__(self, motion, config):
        self.motion = motion
        self.config = config.validate(motion.config)
        options = dict(betas=tuple(config.adam_betas), eps=config.adam_eps)
        self.optimizer = torch.optim.Adam([
            dict(params=motion.region_translation, lr=config.translation_lr),
            dict(params=motion.region_rotation, lr=config.rotation_lr),
        ], **options)
        self.completed_steps = []
        self.row_exposures = [torch.zeros(count, dtype=torch.int64)
                              for count in motion.counts]

    def due(self, step):
        return step in self.config.steps()

    def full_plan(self, step):
        return self.motion.shadow_frontier_plan(
            step, self.config.level_for_step(step))

    def plan(self, step, frame_index=None, full_plan=None):
        """Uniformly sample a fixed fraction without reading losses or global RNG."""
        full_plan = self.full_plan(step) if full_plan is None else full_plan
        if (int(full_plan['step']) != int(step)
                or int(full_plan['gate_level']) != self.config.level_for_step(step)):
            raise ValueError('Full shadow plan has the wrong step or level')
        candidates = full_plan['candidate_gate_positions']
        if self.config.fraction == 1:
            return full_plan
        if (isinstance(frame_index, bool) or not isinstance(frame_index, int)
                or frame_index < 1):
            raise ValueError('Fractional shadow sampling requires a positive frame_index')
        count = int(candidates.numel())
        selected_count = int(math.ceil(self.config.fraction * count)) if count else 0
        rng = np.random.default_rng(np.random.SeedSequence([
            int(self.config.seed), int(frame_index), int(step),
            int(full_plan['gate_level']), 0x53484144,
        ]))
        offsets = np.sort(rng.choice(count, selected_count, replace=False))
        offsets = torch.as_tensor(offsets, device=candidates.device, dtype=torch.long)
        selected = candidates[offsets]
        return self.motion.shadow_frontier_plan(
            step, int(full_plan['gate_level']), gate_positions=selected)

    @staticmethod
    def serializable_plan(plan):
        return dict(
            step=int(plan['step']),
            gate_level=int(plan['gate_level']),
            candidate_gate_positions=(plan['candidate_gate_positions'].detach().cpu().tolist()),
            candidate_gate_nodes=plan['candidate_gate_nodes'].detach().cpu().tolist(),
            gate_positions=plan['gate_positions'].detach().cpu().tolist(),
            gate_nodes=plan['gate_nodes'].detach().cpu().tolist(),
            target_rows=[torch.where(mask)[0].detach().cpu().tolist()
                         for mask in plan['target_rows']],
        )

    def pose(self, step, plan):
        return self.motion.shadow_frontier_pose(step, plan)

    @staticmethod
    def _mask_state_rows(optimizer, parameter, active):
        state = optimizer.state.get(parameter, {})
        shape = (-1,) + (1,) * (parameter.ndim - 1)
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = state.get(name)
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value.mul_(active.reshape(shape))

    def backward_and_step(self, step, loss, plan):
        if int(step) != int(plan['step']) or int(step) != int(self.motion.last_step):
            raise ValueError('Shadow step, plan step, and prepared motion step must match')
        if not any(bool(mask.any()) for mask in plan['target_rows']):
            raise ValueError('Empty shadow frontier must be skipped before backward')
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite shadow image loss')
        self.motion.optimizer_zero_grad()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_rows = []
        for level, active in enumerate(plan['target_rows']):
            active = active.to(device=self.motion.base_xyz.device, dtype=torch.bool)
            self.row_exposures[level][active.detach().cpu()] += 1
            row = dict(level=level, active_rows=int(active.sum()), gradients={})
            for name, parameter in (
                    ('translation', self.motion.region_translation[level]),
                    ('rotation', self.motion.region_rotation[level])):
                if not bool(active.any()):
                    parameter.grad = None
                    row['gradients'][name] = 0.0
                    continue
                if parameter.grad is None:
                    raise AssertionError('Reachable shadow motion row has no gradient tensor')
                mask = active.reshape((-1,) + (1,) * (parameter.ndim - 1))
                parameter.grad.mul_(mask)
                self._mask_state_rows(self.optimizer, parameter, active)
                row['gradients'][name] = float(parameter.grad.norm())
            gradient_rows.append(row)
        self.optimizer.step()
        for level, active in enumerate(plan['target_rows']):
            active = active.to(device=self.motion.base_xyz.device, dtype=torch.bool)
            for parameter in (self.motion.region_translation[level],
                              self.motion.region_rotation[level]):
                self._mask_state_rows(self.optimizer, parameter, active)
        self.optimizer.zero_grad(set_to_none=True)
        self.motion.optimizer_zero_grad()
        self.completed_steps.append(int(step))
        return gradient_rows

    def capture(self):
        return dict(config=asdict(self.config),
                    optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    completed_steps=list(self.completed_steps),
                    row_exposures=[value.clone() for value in self.row_exposures])

    def restore(self, packet):
        if packet['config'] != asdict(self.config):
            raise ValueError('Shadow optimizer config differs from checkpoint')
        self.optimizer.load_state_dict(copy.deepcopy(packet['optimizer']))
        self.completed_steps = list(packet['completed_steps'])
        self.row_exposures = [value.detach().cpu().clone()
                              for value in packet['row_exposures']]
