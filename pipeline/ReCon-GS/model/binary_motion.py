"""Adaptive anchor forest with zero-mean binary motion differences.

Memberships are hard in v1. Each split adds one six-dimensional coefficient;
the weighted child mean is unchanged. Growth is decided by the caller's image
rate-distortion trials, never by a fixed fraction or by a gradient score alone.
"""
import copy
import math

import numpy as np
import torch
from torch import nn

from ..utils import batched_keops_knn, uniform_grid_sampling_optimized
from .motion_codec import (compressed_structure_bits, decode_motion,
                           encode_coefficients, fit_laplace_scales, laplace_bits,
                           quantize_ste, rotation_vector_to_quaternion,
                           structure_arrays)
from .row_scaled_adam import RowScaledAdam


def group_mean(value, group, num_groups):
    count = torch.bincount(group, minlength=num_groups)
    total = value.new_zeros(num_groups, value.shape[-1])
    total.index_add_(0, group, value)
    return total / count.clamp_min(1)[:, None], count


@torch.no_grad()
def binary_clusters(xyz, parent, gradient, observed, num_nodes, gradient_weight=0., iterations=6):
    """Batched parent-restricted two-means; returns [N] right-child indicators.

Spatial and gradient features are normalized separately within each parent.
Unobserved gradients use their parent's observed mean, not an artificial zero.
"""
    center, count = group_mean(xyz, parent, num_nodes)
    space = xyz - center[parent]
    radius, _ = group_mean(space.square().sum(-1, keepdim=True), parent, num_nodes)
    features = space / radius[parent].sqrt().clamp_min(1e-10)
    if gradient_weight > 0:
        mean_g, _ = group_mean(gradient[observed], parent[observed], num_nodes)
        centered = torch.where(observed[:, None], gradient - mean_g[parent], 0.)
        energy, _ = group_mean(centered.square().sum(-1, keepdim=True)[observed],
                               parent[observed], num_nodes)
        features = torch.cat([features, gradient_weight * centered /
                              energy[parent].sqrt().clamp_min(1e-12)], dim=-1)
    ids = torch.arange(len(xyz), device=xyz.device)

    def farthest(distance):
        maximum = distance.new_full((num_nodes,), -1)
        maximum.scatter_reduce_(0, parent, distance, reduce='amax', include_self=True)
        candidate = torch.where(distance == maximum[parent], ids, len(xyz))
        selected = torch.full((num_nodes,), len(xyz), device=xyz.device, dtype=torch.long)
        selected.scatter_reduce_(0, parent, candidate, reduce='amin', include_self=True)
        return features[selected.clamp_max(len(xyz) - 1)]

    left = farthest(features.square().sum(-1))
    right = farthest((features - left[parent]).square().sum(-1))
    for _ in range(iterations):
        label = ((features - right[parent]).square().sum(-1) <
                 (features - left[parent]).square().sum(-1)).long()
        means, sizes = group_mean(features, parent * 2 + label, num_nodes * 2)
        means, sizes = means.reshape(num_nodes, 2, -1), sizes.reshape(num_nodes, 2)
        left = torch.where((sizes[:, :1] > 0), means[:, 0], left)
        right = torch.where((sizes[:, 1:] > 0), means[:, 1], right)
    return ((features - right[parent]).square().sum(-1) <
            (features - left[parent]).square().sum(-1)).long()


class BinaryMotion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.initialized = False
        self.optimizer = None
        self.events = []
        self.probe_backward_steps = 0
        if config.binary_max_depth < 1 or config.binary_max_splits < 1:
            raise ValueError('Binary hierarchy requires positive depth and split limits')
        if config.binary_q_xyz <= 0 or config.binary_q_rotation <= 0:
            raise ValueError('Quantization steps must be positive')
        if (config.binary_root_size < 1 or config.binary_min_child < 1 or
                config.binary_candidate_batch < 1 or config.binary_kmeans_steps < 1 or
                config.binary_probe_steps < 1 or config.binary_eval_views < 1 or
                config.binary_split_interval < 1):
            raise ValueError('Counts and intervals must be positive')
        if not 1 <= config.binary_split_start <= config.binary_split_until < config.num_stage1_steps:
            raise ValueError('Split schedule must finish before the motion stage ends')
        if config.binary_gradient_weight < 0 or config.binary_lambda_rate < 0:
            raise ValueError('Gradient feature and rate weights must be nonnegative')
        if not 0 < config.binary_detail_lr_scale <= 1:
            raise ValueError('Detail learning-rate scale must be in (0, 1]')

    def _buffer(self, name, value):
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value)

    @property
    def get_grid_num(self):
        return self.root_count + self.split_count if self.initialized else 0

    @torch.no_grad()
    def setup(self, xyz, reset_grid=False):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not torch.isfinite(xyz).all():
            raise ValueError('Binary motion needs a nonempty finite [N,3] point tensor')
        old_centers = old_motion = None
        if self.initialized and self.config.momentum is not None:
            old_field = self.field().detach()
            means, counts = group_mean(old_field, self.leaf_index, self.node_count)
            centers, _ = group_mean(self.xyz, self.leaf_index, self.node_count)
            occupied = counts > 0
            old_motion = means[occupied]
            old_centers = centers[occupied] + old_motion[:, :3]
        root_count = min(len(xyz), max(1, math.ceil(len(xyz) / self.config.binary_root_size)))
        sampled = uniform_grid_sampling_optimized(xyz, root_count)
        anchors = xyz[sampled]
        initial = batched_keops_knn(xyz, anchors, 1).flatten()
        _, root_index = torch.unique(initial, sorted=True, return_inverse=True)
        self.root_count = int(root_index.max()) + 1
        self.split_count = 0
        self.node_count = self.root_count
        self.active_depth = 0
        max_nodes = self.root_count + 2 * self.config.binary_max_splits
        capacity = self.root_count + self.config.binary_max_splits
        self.delta = nn.Parameter(xyz.new_zeros(capacity, 6))
        centers, counts = group_mean(xyz, root_index, self.root_count)
        if old_motion is not None:
            k = min(3, len(old_centers))
            nearest = batched_keops_knn(centers, old_centers, k)
            self.delta[:self.root_count] = old_motion[nearest].mean(1) * self.config.momentum
        self._buffer('xyz', xyz.detach().clone())
        self._buffer('root_index', root_index)
        self._buffer('leaf_index', root_index.clone())
        self._buffer('node_depth', torch.zeros(max_nodes, dtype=torch.long, device=xyz.device))
        self._buffer('node_children', torch.full((max_nodes, 2), -1, dtype=torch.long, device=xyz.device))
        self._buffer('node_count_gs', torch.zeros(max_nodes, dtype=torch.long, device=xyz.device))
        self.node_count_gs[:self.root_count] = counts
        self._buffer('split_nodes', torch.full((self.config.binary_max_splits,), -1,
                                              dtype=torch.long, device=xyz.device))
        self._buffer('coefficient_depth', torch.zeros(capacity, dtype=torch.long, device=xyz.device))
        self._buffer('path_index', torch.full((len(xyz), self.config.binary_max_depth), -1,
                                             dtype=torch.long, device=xyz.device))
        self._buffer('path_weight', xyz.new_zeros(len(xyz), self.config.binary_max_depth))
        self._buffer('gradient_sum', torch.zeros_like(xyz))
        self._buffer('gradient_observations', xyz.new_zeros(len(xyz)))
        self._buffer('quant_step', xyz.new_tensor([self.config.binary_q_xyz] * 3 +
                                                 [self.config.binary_q_rotation] * 3))
        self.initialized = True
        self.events = []
        self.probe_backward_steps = 0
        self.refresh_structure_rate()
        self.reset_optimizer()

    def reset(self, xyz):
        self.setup(xyz)

    def reset_grid(self, xyz):
        self.setup(xyz, reset_grid=True)

    def reset_optimizer(self):
        self.optimizer = RowScaledAdam(self.delta, self.root_count,
                                       self.config.binary_detail_lr_scale,
                                       lr=self.config.lr, eps=1e-15)

    def field(self):
        coefficients = quantize_ste(self.delta[:self.get_grid_num], self.quant_step)
        value = coefficients[self.root_index]
        # Explicit loop keeps peak memory small and matches the decoder's order.
        for depth in range(self.active_depth):
            index = self.path_index[:, depth]
            points = torch.where(index >= 0)[0]
            contribution = coefficients[index[points]] * self.path_weight[points, depth, None]
            # Do not gather all absent paths from row zero: its zero-gradient
            # atomic additions cause severe CUDA contention for sparse trees.
            value = value.index_add(0, points, contribution)
        return value

    def forward(self, xyz, normalized=False):
        if len(xyz) != len(self.root_index):
            raise ValueError('Gaussian membership changed without rebuilding binary hierarchy')
        value = self.field()
        return value[:, :3].contiguous(), rotation_vector_to_quaternion(value[:, 3:])

    def reg_loss(self):
        # Penalize the represented motion, not the number of allocated rows.
        # Adding a zero difference must not dilute the existing regularizer.
        return self.field().abs().mean()

    def rate_bits(self):
        coefficients = quantize_ste(self.delta[:self.get_grid_num], self.quant_step)
        symbols = coefficients / self.quant_step
        groups = self.coefficient_depth[:self.get_grid_num]
        num_groups = self.config.binary_max_depth + 1
        scales = fit_laplace_scales(symbols, groups, num_groups)
        bits = laplace_bits(symbols, scales[groups]).sum()
        # CDF tables are actual transmitted side information, not free context.
        with torch.no_grad():
            lower = symbols.new_full((num_groups, 6), float('inf'))
            upper = symbols.new_full((num_groups, 6), -float('inf'))
            expanded = groups[:, None].expand(-1, 6)
            lower.scatter_reduce_(0, expanded, symbols.detach(), reduce='amin', include_self=True)
            upper.scatter_reduce_(0, expanded, symbols.detach(), reduce='amax', include_self=True)
            width = torch.where(torch.isfinite(lower), upper - lower + 2, 0)
            table_bits = torch.where(width > 2, width * 16, 0).sum()
            header_bits = (torch.bincount(groups, minlength=num_groups) > 0).sum() * 6 * 128
        return bits + self.structure_bits + table_bits + header_bits

    @torch.no_grad()
    def refresh_structure_rate(self):
        arrays = structure_arrays(self.root_index, self.split_nodes[:self.split_count],
                                  self.path_index, self.path_weight)
        self.structure_bits = compressed_structure_bits(arrays)

    def collecting_gradients(self, step):
        return 1 <= step <= self.config.binary_split_until

    @torch.no_grad()
    def accumulate_gradient_score(self, gradient, visible, step=None):
        if gradient is None:
            return
        valid = visible.bool() & torch.isfinite(gradient).all(-1)
        self.gradient_sum[valid] += gradient[valid]
        self.gradient_observations[valid] += 1

    def split_due(self, step):
        return (self.config.binary_split_start <= step <= self.config.binary_split_until and
                (step - self.config.binary_split_start) % self.config.binary_split_interval == 0)

    @torch.no_grad()
    def propose(self):
        gradient = self.gradient_sum / self.gradient_observations.clamp_min(1)[:, None]
        observed = self.gradient_observations > 0
        right = binary_clusters(self.xyz, self.leaf_index, gradient, observed, self.node_count,
                                self.config.binary_gradient_weight, self.config.binary_kmeans_steps)
        child_group = self.leaf_index * 2 + right
        mean, seen = group_mean(gradient[observed], child_group[observed], self.node_count * 2)
        mean, seen = mean.reshape(-1, 2, 3), seen.reshape(-1, 2)
        count = torch.bincount(child_group, minlength=self.node_count * 2).reshape(-1, 2)
        score = (mean[:, 0] - mean[:, 1]).square().sum(-1)
        score *= seen[:, 0] * seen[:, 1] / seen.sum(-1).clamp_min(1)
        eligible = ((count.min(-1).values >= self.config.binary_min_child) &
                    (seen.min(-1).values >= 2) &
                    (self.node_depth[:self.node_count] < self.config.binary_max_depth) &
                    (self.node_children[:self.node_count, 0] < 0) &
                    torch.isfinite(score) & (score > 0))
        candidates = torch.where(eligible)[0]
        # Stable sorting gives node-ID tie breaks; no mandatory fill of the budget.
        order = torch.argsort(score[candidates], descending=True, stable=True)
        limit = min(self.config.binary_candidate_batch,
                    self.config.binary_max_splits - self.split_count)
        selected = candidates[order[:limit]]
        return dict(nodes=selected, right=right, scores=score[selected],
                    depths=self.node_depth[selected].clone(), eligible=int(eligible.sum()))

    @torch.no_grad()
    def activate(self, proposal):
        selected = proposal['nodes']
        if len(selected) == 0:
            return
        if (self.node_children[selected, 0] >= 0).any():
            raise ValueError('Only current leaves may split')
        mapping = torch.full((self.node_count,), -1, dtype=torch.long, device=self.xyz.device)
        mapping[selected] = torch.arange(len(selected), device=self.xyz.device)
        relative = mapping[self.leaf_index]
        points = torch.where(relative >= 0)[0]
        ranks = relative[points]
        parent = self.leaf_index[points]
        right = proposal['right'][points]
        children = self.node_count + 2 * ranks + right
        counts = torch.bincount(children, minlength=self.node_count + 2 * len(selected))
        child_ids = self.node_count + torch.arange(2 * len(selected), device=self.xyz.device).reshape(-1, 2)
        nl, nr = counts[child_ids[:, 0]], counts[child_ids[:, 1]]
        if (nl == 0).any() or (nr == 0).any():
            raise ValueError('Empty child is not a split')
        coefficient = self.root_count + self.split_count + ranks
        depth = self.node_depth[parent]
        weights = torch.where(right == 0, nr[ranks], -nl[ranks]).to(self.xyz.dtype)
        weights /= (nl + nr)[ranks]
        self.path_index[points, depth] = coefficient
        self.path_weight[points, depth] = weights
        self.leaf_index[points] = children
        self.node_children[selected] = child_ids
        self.node_depth[child_ids.flatten()] = (self.node_depth[selected] + 1).repeat_interleave(2)
        self.node_count_gs[child_ids.flatten()] = counts[child_ids.flatten()]
        start = self.root_count + self.split_count
        self.delta[start:start + len(selected)] = 0
        self.coefficient_depth[start:start + len(selected)] = self.node_depth[selected] + 1
        self.split_nodes[self.split_count:self.split_count + len(selected)] = selected
        self.split_count += len(selected)
        self.node_count += 2 * len(selected)
        self.active_depth = int(self.node_depth[:self.node_count].max())
        self.refresh_structure_rate()

    @torch.no_grad()
    def clear_scores(self):
        self.gradient_sum.zero_()
        self.gradient_observations.zero_()

    def capture(self):
        if not self.initialized:
            return {}
        return dict(binary_version=1, state={k: v.detach().clone() for k, v in self.state_dict().items()},
                    optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    root_count=self.root_count, split_count=self.split_count,
                    node_count=self.node_count, active_depth=self.active_depth,
                    structure_bits=self.structure_bits,
                    events=copy.deepcopy(self.events), probe_backward_steps=self.probe_backward_steps)

    def restore(self, checkpoint):
        if not checkpoint:
            return
        if checkpoint.get('binary_version') != 1:
            raise ValueError('Binary motion requires its own checkpoint or a shared static first frame')
        if checkpoint['state']['path_index'].shape[1] != self.config.binary_max_depth:
            raise ValueError('Cannot change maximum depth when restoring a binary checkpoint')
        self.delta = nn.Parameter(checkpoint['state']['delta'].clone())
        for key, value in checkpoint['state'].items():
            if key != 'delta':
                self._buffer(key, value.clone())
        for key in ('root_count', 'split_count', 'node_count', 'active_depth',
                    'structure_bits', 'probe_backward_steps'):
            setattr(self, key, checkpoint[key])
        self.events = copy.deepcopy(checkpoint['events'])
        self.initialized = True
        self.reset_optimizer()
        self.optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))

    @torch.no_grad()
    def export(self, path):
        groups = self.coefficient_depth[:self.get_grid_num]
        symbols = torch.round(self.delta[:self.get_grid_num] / self.quant_step)
        scales = fit_laplace_scales(symbols, groups, self.config.binary_max_depth + 1)
        arrays = structure_arrays(self.root_index, self.split_nodes[:self.split_count],
                                  self.path_index, self.path_weight)
        arrays.update(encode_coefficients(symbols.int().cpu().numpy(), groups.cpu().numpy(),
                                          scales.cpu().numpy()))
        arrays.update(representation=np.array('binary_motion_v1'),
                      quant_step=self.quant_step.cpu().numpy(),
                      tree_shape=np.array([self.root_count, self.config.binary_max_depth]))
        np.savez_compressed(path, **arrays)
        decoded_xyz, decoded_rot = decode_motion(path, device=self.xyz.device)
        expected_xyz, expected_rot = self(self.xyz)
        torch.testing.assert_close(decoded_xyz, expected_xyz, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(decoded_rot, expected_rot, rtol=1e-5, atol=1e-7)
        depth_counts = torch.bincount(self.node_depth[self.leaf_index],
                                     minlength=self.config.binary_max_depth + 1)
        return dict(root_anchors=self.root_count, splits=self.split_count,
                    motion_coefficients=self.get_grid_num, estimated_bits=float(self.rate_bits()),
                    structure_estimated_bits=self.structure_bits,
                    motion_bytes=path.stat().st_size,
                    decode_xyz_max_error=float((decoded_xyz - expected_xyz).abs().max()),
                    decode_rotation_max_error=float((decoded_rot - expected_rot).abs().max()),
                    gs_by_depth=depth_counts.tolist(), extra_backward_steps=self.probe_backward_steps,
                    accepted_rounds=sum(int(e.get('accepted', False)) for e in self.events))
