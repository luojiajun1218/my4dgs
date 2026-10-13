"""HEIR Section 3.3 motion-graph learning, independent of GS clustering.

Source: https://light.princeton.edu/wp-content/uploads/2025/10/HEIR.pdf
Equations (3)-(5): attention, relative motion, straight-through hierarchy
sampling, truncated ancestor accumulation, and reconstruction/residual L1.
The supplement uses a fully connected graph at the motion-group level.

This module accepts arbitrary motion features [T,N,D], with identity node 0.
It does not implement HEIR's polar encoder or ARAP editing pipeline. A hard
acyclic projection is added for our stream codec; training samples may contain
cycles, as discussed in the paper. SE(3) pose composition remains separate.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class HeirHierarchyConfig:
    geometry: str = 'linear'  # Legacy checkpoint behavior; se3 uses physical parent costs.
    history_frames: int = 0  # Past observed frames, excluding the current fitting frame.
    heads: int = 4
    attention_channels: int = 3
    learning_rate: float = 0.002
    temperature_start: float = 1.6
    temperature_end: float = 1.6
    hierarchy_samples: int = 1
    propagation_depth: int = 8
    early_stop_epsilon: float = 1e-6
    reconstruction_weight: float = 1.0
    residual_weight: float = 0.01
    negative_slope: float = 0.2

    def validate(self):
        if self.geometry not in ('linear', 'se3'):
            raise ValueError('heir.geometry must be linear or se3')
        if (isinstance(self.history_frames, bool) or not isinstance(self.history_frames, int) or
                self.history_frames < 0):
            raise ValueError('heir.history_frames must be a nonnegative integer')
        for name in ('heads', 'attention_channels', 'hierarchy_samples', 'propagation_depth'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'heir.{name} must be a positive integer')
        for name in ('learning_rate', 'temperature_start', 'temperature_end',
                     'early_stop_epsilon', 'reconstruction_weight', 'residual_weight',
                     'negative_slope'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'heir.{name} must be finite and positive')


def load_heir_config(path):
    """Read the standalone reproduction settings; not a training entrypoint."""
    from omegaconf import OmegaConf

    config = OmegaConf.to_object(OmegaConf.merge(
        OmegaConf.structured(HeirHierarchyConfig), OmegaConf.load(path)))
    config.validate()
    return config


def reconstruct_motion(relative, hierarchy, depth, epsilon=0.):
    """Sum H^l delta for l=0..depth; H has child-row/parent-column convention."""
    term = relative
    reconstructed = term
    for _ in range(min(depth, hierarchy.shape[-1])):
        term = torch.matmul(hierarchy, term)
        reconstructed = reconstructed + term
        if epsilon and float(term.detach().norm()) < epsilon:
            break
    return reconstructed


@torch.no_grad()
def se3_parent_costs(quaternion, offset, pivots, valid, extent, steps):
    """Actual pivot-based local costs [child,parent], over matched times.

    Inputs are [T,N,4], [T,N,3], [T,N,3], [T,N]. Current transforms are last.
    Every admitted edge costs no more than identity inheritance both now and
    over the same valid history. A newer parent cannot explain an older child
    using only its shorter, convenient history. Root remains always available.
    """
    from .rigid_residual_tree import (quaternion_inverse, quaternion_multiply,
        quaternion_rotate, quaternion_rotvec, rotation_delta)

    time, nodes = quaternion.shape[:2]
    if (quaternion.shape != (time, nodes, 4) or offset.shape != (time, nodes, 3) or
            pivots.shape != offset.shape or valid.shape != (time, nodes) or
            not bool(valid[:, 0].all()) or not bool(valid[-1].all())):
        raise ValueError('SE(3) hierarchy needs aligned history and a complete current frame')
    child = quaternion[:, :, None].expand(-1, -1, nodes, -1)
    parent = quaternion[:, None, :].expand(-1, nodes, -1, -1)
    relative = quaternion_multiply(child, quaternion_inverse(parent))
    rotation = quaternion_rotvec(relative)
    inherited_pivot = quaternion_rotate(parent, pivots[:, :, None].expand(-1, -1, nodes, -1))
    translation = (offset[:, :, None] - offset[:, None, :] -
                   rotation_delta(relative, -inherited_pivot))
    overflow = ((translation / steps[:3]).abs().gt(32767.5).any(-1) |
                (rotation / steps[3:]).abs().gt(32767.5).any(-1))
    translation = torch.round(translation / steps[:3]) * steps[:3]
    rotation = torch.round(rotation / steps[3:]) * steps[3:]
    values = translation.abs().sum(-1) / extent + rotation.abs().sum(-1)
    paired = valid[:, :, None] & valid[:, None, :]
    counts = paired.sum(0).clamp_min(1)
    costs = (values * paired).sum(0) / counts
    root_costs = (values[:, :, :1] * paired).sum(0) / counts
    covered = paired.sum(0) == valid.sum(0)[:, None]
    tolerance = torch.finfo(values.dtype).eps * 8
    allowed = (covered & ~((overflow & paired).any(0)) &
               (costs <= root_costs + tolerance) &
               (values[-1] <= values[-1, :, :1] + tolerance))
    allowed.fill_diagonal_(False)
    allowed[0] = False
    allowed[1:, 0] = True
    return costs, allowed


@torch.no_grad()
def acyclic_parent_projection(weights, max_depth):
    """Project probabilities with a standard maximum spanning arborescence.

    Parent->child edge weights are log probabilities. The identity root has no
    incoming edges. This codec adaptation makes the exported graph acyclic;
    the paper's raw row-wise argmax alone does not guarantee that. An additional
    codec depth bound detaches overly deep nodes to the identity root.
    """
    scores = weights.detach().cpu()
    if max_depth < 1:
        raise ValueError('A non-root HEIR node needs a positive depth limit')
    count = len(scores)
    parents = [-1] + [0] * (count - 1)
    if count < 2:
        return torch.tensor(parents, device=weights.device)
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(range(count))
    for node in range(1, count):
        for parent in range(count):
            if parent != node:
                graph.add_edge(parent, node,
                               weight=math.log(max(float(scores[node, parent]), 1e-30)))
    tree = nx.maximum_spanning_arborescence(graph)
    for parent, node in tree.edges:
        parents[node] = parent
    for node in range(1, count):
        cursor, depth = node, 0
        while cursor != 0:
            depth += 1
            cursor = parents[cursor]
        if depth > max_depth:
            parents[node] = 0
    return torch.tensor(parents, dtype=torch.long, device=weights.device)


class HeirMotionHierarchy(nn.Module):
    """Learn parent probabilities on motion groups using native PyTorch."""

    def __init__(self, input_dim, config):
        super().__init__()
        config.validate()
        self.config = config
        self.projection = nn.Parameter(torch.empty(
            config.heads, input_dim, config.attention_channels))
        self.source_attention = nn.Parameter(torch.empty(config.heads, config.attention_channels))
        self.parent_attention = nn.Parameter(torch.empty(config.heads, config.attention_channels))
        nn.init.xavier_uniform_(self.projection)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.parent_attention)

    def make_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def parent_weights(self, motion, valid=None, parent_mask=None):
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if (motion.ndim != 3 or not len(motion) or not motion.shape[1]
                or not bool(torch.isfinite(motion).all())
                or bool(motion[:, 0].ne(0).any())):
            raise ValueError('HEIR needs finite [T,N,D] motion with identity node 0')
        nodes = motion.shape[1]
        if valid is None:
            valid = torch.ones(motion.shape[:2], dtype=torch.bool, device=motion.device)
        if (valid.shape != motion.shape[:2] or valid.dtype != torch.bool or not bool(valid[:, 0].all()) or
                not bool(valid.any(0).all())):
            raise ValueError('HEIR validity must provide each node with observed motion')
        projected = torch.einsum('tnd,hdf->thnf', motion, self.projection)
        source = (projected * self.source_attention[None, :, None]).sum(-1)
        parent = (projected * self.parent_attention[None, :, None]).sum(-1)
        scores = F.leaky_relu(source[..., :, None] + parent[..., None, :],
                             self.config.negative_slope)
        mask = torch.eye(nodes, device=motion.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, -torch.inf)
        if nodes == 1:
            return motion.new_zeros((1, 1)) + self.projection.sum() * 0
        allowed = ~mask
        coverage = (valid[:, :, None] & valid[:, None, :]).sum(0)
        allowed &= coverage == valid.sum(0)[:, None]
        if parent_mask is not None:
            if parent_mask.shape != (nodes, nodes) or not bool(parent_mask[1:, 0].all()):
                raise ValueError('HEIR parent mask must retain the identity parent')
            allowed &= parent_mask
        pair_valid = valid[:, :, None] & valid[:, None, :] & allowed[None]
        # Missing child samples have a harmless root row, discarded below.
        pair_valid[:, 1:, 0] |= ~valid[:, 1:]
        scores = scores.masked_fill(~pair_valid[:, None], -torch.inf)
        # Share the scene hierarchy across observed time steps. Average the
        # normalized heads; only non-root rows participate in parent selection.
        per_time = scores[..., 1:, :].softmax(-1).mean(1)
        weights = ((per_time * valid[:, 1:, None]).sum(0) /
                   valid[:, 1:].sum(0)[:, None])
        return torch.cat((weights.new_zeros((1, nodes)), weights))

    def sample_hierarchy(self, weights, temperature):
        if len(weights) == 1:
            return weights
        scores = weights[1:]
        if self.config.geometry == 'se3':
            # Gumbel-Softmax expects log probabilities, not another softmax of
            # probabilities. Keep exact zero probability for forbidden edges.
            scores = scores.clamp_min(torch.finfo(scores.dtype).tiny).log()
            scores = scores.masked_fill(weights[1:] == 0, -torch.inf)
        scores = scores.masked_fill(
            torch.eye(len(weights), device=weights.device, dtype=torch.bool)[1:], -torch.inf)
        # Linear mode preserves the legacy Eq. (4) literal; SE(3) mode samples
        # the learned categorical distribution using its log probabilities.
        selected = F.gumbel_softmax(scores, tau=temperature, hard=True, dim=-1)
        return torch.cat((selected.new_zeros((1, len(weights))), selected))

    def loss_terms(self, motion, progress, valid=None, parent_mask=None, edge_cost=None):
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if valid is None:
            valid = torch.ones(motion.shape[:2], dtype=torch.bool, device=motion.device)
        motion = torch.where(valid[..., None], motion, torch.zeros_like(motion))
        weights = self.parent_weights(motion, valid, parent_mask)
        relative = motion - torch.matmul(weights, motion)
        progress = min(max(float(progress), 0.), 1.)
        temperature = self.config.temperature_start + progress * (
            self.config.temperature_end - self.config.temperature_start)
        if motion.shape[1] == 1:
            reconstruction = self.projection.sum() * 0
            residual = reconstruction
        else:
            losses = []
            for _ in range(self.config.hierarchy_samples):
                hierarchy = self.sample_hierarchy(weights, temperature)
                reconstructed = reconstruct_motion(relative, hierarchy,
                    self.config.propagation_depth, self.config.early_stop_epsilon)
                errors = (reconstructed[:, 1:] - motion[:, 1:]).abs().mean(-1)
                losses.append((errors * valid[:, 1:]).sum() / valid[:, 1:].sum())
            reconstruction = torch.stack(losses).mean()
            residual = ((relative[:, 1:].abs().mean(-1) * valid[:, 1:]).sum() /
                        valid[:, 1:].sum() if edge_cost is None else
                        (weights[1:] * edge_cost[1:]).sum() / (len(weights) - 1))
        return dict(heir_reconstruction_loss=reconstruction,
                    heir_residual_loss=residual,
                    heir_loss=(self.config.reconstruction_weight * reconstruction +
                               self.config.residual_weight * residual))

    @torch.no_grad()
    def infer_parents(self, motion, max_depth, valid=None, parent_mask=None):
        weights = self.parent_weights(motion, valid, parent_mask)
        parents = acyclic_parent_projection(weights, max_depth)
        if parent_mask is not None:
            rows = torch.arange(1, len(parents), device=parents.device)
            parents[rows[~parent_mask[rows, parents[1:]]]] = 0
        return parents
