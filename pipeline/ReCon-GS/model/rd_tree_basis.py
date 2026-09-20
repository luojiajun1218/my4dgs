"""Experimental differentiable coordinates for a fixed candidate motion tree.

One root common-motion coefficient and one contrast per internal node span
exactly the piecewise-constant atomic-leaf space. This does not select a cut;
the existing rate-distortion projector remains responsible for that decision.
"""
import numpy as np
import torch
from torch import nn

from .rd_tree_projection import ProjectionWorkspace


class TreeMotionBasis(nn.Module):
    def __init__(self, tree, workspace=None):
        super().__init__()
        workspace = ProjectionWorkspace(tree) if workspace is None else workspace
        if workspace.tree is not tree:
            raise ValueError('Basis workspace must belong to the same immutable tree')
        left, right, counts, levels = workspace.layout[:4]
        internal = np.flatnonzero(left >= 0)
        self.coefficient_count = 1 + len(internal)
        self.node_count = len(tree.nodes)
        self.level_count = len(levels)
        self.internal_nodes = tuple(internal.tolist())
        coefficient_ids = np.full(len(left), -1, dtype=np.int64)
        coefficient_ids[internal] = np.arange(len(internal)) + 1
        self.register_buffer('point_leaf', torch.tensor(workspace.layout[-1], dtype=torch.long))
        weights = np.concatenate(([tree.point_count],
                                  counts[left[internal]] * counts[right[internal]] / counts[internal]))
        self.register_buffer('energy_weights', torch.tensor(weights, dtype=torch.float64))
        for depth, parents in enumerate(levels):
            a, b = left[parents], right[parents]
            self.register_buffer(f'parents_{depth}', torch.tensor(parents, dtype=torch.long))
            self.register_buffer(f'children_{depth}', torch.tensor(np.concatenate((a, b)), dtype=torch.long))
            self.register_buffer(f'coefficients_{depth}', torch.tensor(coefficient_ids[parents], dtype=torch.long))
            self.register_buffer(f'right_fraction_{depth}',
                                 torch.tensor(counts[b] / counts[parents], dtype=torch.float64))
            self.register_buffer(f'left_fraction_{depth}',
                                 torch.tensor(counts[a] / counts[parents], dtype=torch.float64))

    def forward(self, coefficients):
        """[K,C] physical root/contrast coordinates -> [N,C] point motion.

        Caller moves this module to the coefficients' device. Fraction buffers
        follow their floating dtype without detaching coefficient computation.
        Functional index_copy avoids mutating tensors retained by autograd.
        """
        if coefficients.ndim != 2 or coefficients.shape[0] != self.coefficient_count:
            raise ValueError('Expected [candidate_leaf_count, channels] coefficients')
        if not coefficients.is_floating_point() or coefficients.device != self.point_leaf.device:
            raise ValueError('Floating coefficients and basis must share a device')
        values = coefficients[0].expand(self.node_count, -1)
        for depth in range(self.level_count):
            parents = getattr(self, f'parents_{depth}')
            common = values[parents]
            detail = coefficients[getattr(self, f'coefficients_{depth}')]
            r = getattr(self, f'right_fraction_{depth}').to(dtype=coefficients.dtype)[:, None]
            l = getattr(self, f'left_fraction_{depth}').to(dtype=coefficients.dtype)[:, None]
            children = torch.cat((common + r * detail, common - l * detail))
            values = values.index_copy(0, getattr(self, f'children_{depth}'), children)
        return values[self.point_leaf]
