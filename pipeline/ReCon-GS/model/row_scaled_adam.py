"""Adam with an explicit learning-rate multiplier for a parameter's suffix rows.

Scaling gradients is not equivalent: Adam largely cancels constant gradient
scales in its moments. Scaling the final update changes the effective step size
while retaining ordinary Adam moments and checkpoint serialization.
"""
import copy

import torch


class RowScaledAdam(torch.optim.Adam):
    def __init__(self, parameter, first_scaled_row, multiplier=1., **kwargs):
        if not 0 <= first_scaled_row <= len(parameter):
            raise ValueError('Scaled suffix must lie inside the parameter tensor')
        if not 0 < multiplier <= 1:
            raise ValueError('Suffix learning-rate multiplier must be in (0, 1]')
        super().__init__([dict(params=[parameter], first_scaled_row=first_scaled_row,
                              row_lr_multiplier=float(multiplier))], **kwargs)

    def load_state_dict(self, state_dict):
        # Older binary checkpoints used ordinary Adam and have no row metadata.
        state_dict = copy.deepcopy(state_dict)
        for saved, current in zip(state_dict['param_groups'], self.param_groups):
            for name in ('first_scaled_row', 'row_lr_multiplier'):
                saved.setdefault(name, current[name])
        return super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        snapshots = []
        for group in self.param_groups:
            start, scale = group['first_scaled_row'], group['row_lr_multiplier']
            for parameter in group['params']:
                if scale != 1 and parameter.grad is not None:
                    snapshots.append((parameter, start, scale, parameter[start:].clone()))
        super().step()
        for parameter, start, scale, previous in snapshots:
            parameter[start:].copy_(previous.lerp(parameter[start:], scale))
        return loss
