"""Spatial binary refinement with shared, quantized regional residuals.

A root's zero quantization bucket is KEEP. Each internal node has one binary
refine decision: closing it retains that node's prediction and closes every
descendant. There are no per-GS motion parameters or pose gates.
"""

from dataclasses import asdict, dataclass
import hashlib
import heapq
import math
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
from torch import nn

from .region_rigid import apply_region_rigid
from .residual_motion import integer_symbols, quantized_symbols
from .sparse_hierarchical import SparseHierarchicalConfig, SparseHierarchicalMotion


LAYOUT_VERSION = 'spatial_binary_conditional_refinement_v1'


@dataclass
class BinaryRefinementConfig(SparseHierarchicalConfig):
    coarse_region_gs: Optional[int] = None
    min_child_gs: Optional[int] = None
    max_motion_levels: Optional[int] = None
    coarse_fit_steps: Optional[int] = None
    gate_adam_eps: Optional[float] = None

    def _validate_binary_config(self, *, max_levels=3, rebuild_interval=1):
        # This inherited validator checks quantization/birth settings and has
        # one historical hierarchy field. Binary seeding reads coarse_region_gs.
        self.fine_region_gs = self.coarse_region_gs
        super().validate()
        for name in ('coarse_region_gs', 'min_child_gs', 'max_motion_levels',
                     'coarse_fit_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_motion_levels > max_levels:
            raise ValueError(f'binary refinement supports at most {max_levels} motion levels')
        if self.coarse_region_gs < self.min_child_gs:
            raise ValueError('coarse_region_gs must be at least min_child_gs')
        if self.coarse_fit_steps >= self.dense_fit_steps:
            raise ValueError('coarse_fit_steps must leave child residual fitting steps')
        if (isinstance(self.gate_adam_eps, bool) or
                not isinstance(self.gate_adam_eps, (int, float)) or
                not math.isfinite(self.gate_adam_eps) or self.gate_adam_eps <= 0):
            raise ValueError('gate_adam_eps must be finite and positive')
        if self.tree_builder != 'spatial' or self.tree_probe_steps:
            raise ValueError('binary_refinement uses a spatial tree without a motion probe')
        if self.tree_rebuild_interval != rebuild_interval:
            raise ValueError(f'binary refinement requires tree_rebuild_interval: {rebuild_interval}')
        if self.motion_init_decay:
            raise ValueError('binary_refinement currently requires motion_init_decay: 0')
        if self.m_fine_sharing or self.m_hard_forward or self.rebase_on_hardening:
            raise ValueError('binary_refinement has no m gates or hardening rebase')
        if self.gs_activity_loss_weight:
            raise ValueError('binary_refinement selects motion with storage loss, without m activity loss')
        if self.motion_export_format != 'sparse_parameters':
            raise ValueError('binary_refinement requires sparse_parameters export')

    def validate(self):
        self._validate_binary_config()


def _copy_layout(layout):
    if layout is None:
        return None
    return dict(index=layout['index'].detach().cpu().clone(),
                counts=list(layout['counts']),
                parents=[value.detach().cpu().clone() for value in layout['parents']],
                pivots=[value.detach().cpu().clone() for value in layout['pivots']],
                refine_rows=[value.detach().cpu().clone()
                             for value in layout['refine_rows']])


def _binary_layout(index, counts, parents, xyz, previous_pivots=None):
    """Validate real nodes; -1 means a row already ended at an earlier level."""
    index = torch.as_tensor(index, device=xyz.device, dtype=torch.long)
    counts = list(map(int, counts))
    depth = len(counts)
    if not depth or index.shape != (depth, len(xyz)) or not len(xyz):
        raise ValueError('Binary layout requires one owner row per level and nonempty xyz')
    if counts[0] < 1 or any(count < 0 for count in counts):
        raise ValueError('Binary layout counts must describe real nonnegative node counts')
    if bool((index[0] < 0).any()):
        raise ValueError('Every training GS must belong to one root')
    parent = [torch.as_tensor(value, device=xyz.device, dtype=torch.long)
              for value in parents]
    if len(parent) != depth - 1:
        raise ValueError('Binary layout parent depth mismatch')
    points = xyz.detach().cpu().double().numpy()
    pivots, refine_rows = [], []
    for level, count in enumerate(counts):
        owner = index[level].detach().cpu().numpy()
        if np.any((owner < -1) | (owner >= count)):
            raise ValueError('Binary layout owner is out of range')
        valid = owner >= 0
        # bincount visits each owner's values in input order, matching add.at
        # while avoiding its repeated multidimensional scatter operations.
        sums = np.stack([np.bincount(owner[valid], weights=points[valid, axis],
                                     minlength=count) for axis in range(3)], axis=-1)
        sizes = np.bincount(owner[valid], minlength=count)[:, None]
        means = sums / np.maximum(sizes, 1)
        if previous_pivots is not None:
            fallback = previous_pivots[level].detach().cpu().double().numpy()
            if fallback.shape != means.shape:
                raise ValueError('Binary layout pivot shape mismatch')
            means = np.where(sizes > 0, means, fallback)
        pivots.append(torch.from_numpy(means).to(xyz))
        if level:
            link = parent[level - 1]
            if (link.shape != (count,) or
                    bool(((link < 0) | (link >= counts[level - 1])).any())):
                raise ValueError('Binary layout parent IDs are out of range')
            members = index[level] >= 0
            if bool((link[index[level, members]] != index[level - 1, members]).any()):
                raise ValueError('Binary layout owners disagree with their parents')
            if bool(((index[level - 1] < 0) & members).any()):
                raise ValueError('Binary layout cannot resume after a terminal level')
            children = torch.bincount(link, minlength=counts[level - 1])
            if bool(((children != 0) & (children != 2)).any()):
                raise ValueError('Every refined binary node must have exactly two children')
            refine_rows.append(torch.where(children == 2)[0])
    return dict(index=index, counts=counts, parents=parent, pivots=pivots,
                refine_rows=refine_rows)


def stable_binary_forest(xyz, coarse_region_gs, min_child_gs, max_motion_levels):
    """Longest-axis median splits, with original GS rows breaking coordinate ties.

    Largest-first spatial splitting first creates ceil(N / coarse_region_gs)
    roots. Within each root, a split is retained only if both children satisfy
    the minimum membership. Unsplittable paths have no padded motion nodes.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError('Binary forest requires nonempty [GS, 3] positions')
    for value in (coarse_region_gs, min_child_gs, max_motion_levels):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Binary forest budgets must be positive integers')
    points = xyz.detach().cpu().double().numpy()
    if not np.isfinite(points).all():
        raise ValueError('Binary forest positions must be finite')

    def split(rows):
        axis = np.ptp(points[rows], axis=0).argmax()
        order = rows[np.lexsort((rows, points[rows, axis]))]
        middle = len(order) // 2
        return order[:middle], order[middle:]

    root_count = math.ceil(len(points) / coarse_region_gs)
    heap = [(-len(points), 0, np.arange(len(points), dtype=np.int64))]
    serial = 1
    while len(heap) < root_count:
        _, _, rows = heapq.heappop(heap)
        for part in split(rows):
            heapq.heappush(heap, (-len(part), serial, part))
            serial += 1
    levels = [[rows for _, _, rows in sorted(heap, key=lambda item: item[1])]]
    parents = []
    for _ in range(1, max_motion_levels):
        groups, links = [], []
        for node, rows in enumerate(levels[-1]):
            if len(rows) < 2 * min_child_gs:
                continue
            for part in split(rows):
                groups.append(part)
                links.append(node)
        levels.append(groups)
        parents.append(torch.tensor(links, device=xyz.device, dtype=torch.long))
    index = torch.full((max_motion_levels, len(xyz)), -1,
                       device=xyz.device, dtype=torch.long)
    for level, groups in enumerate(levels):
        for node, rows in enumerate(groups):
            index[level, torch.as_tensor(rows, device=xyz.device)] = node
    return _binary_layout(index, [len(groups) for groups in levels], parents, xyz)


def _conditional_transform(base_xyz, base_rotation, layout, regions, paths):
    """Apply reachable residuals and carry every finer pivot with its ancestors."""
    index = layout['index']
    xyz, rotation = base_xyz, base_rotation
    pivots = []
    for level in range(len(regions)):
        if not len(regions[level]):
            pivots.append(torch.zeros_like(base_xyz))
        else:
            pivots.append(layout['pivots'][level][index[level].clamp_min(0)])
    identity = base_xyz.new_zeros((len(base_xyz), 4))
    identity[:, 0] = 1
    path_active = torch.zeros(len(base_xyz), device=base_xyz.device, dtype=torch.bool)
    for level, motion in enumerate(regions):
        if not len(motion):
            continue
        owner = index[level]
        valid = owner >= 0
        weight = paths[level][owner.clamp_min(0)] * valid.to(base_xyz)
        values = motion[owner.clamp_min(0)] * weight[:, None]
        path_active |= values.detach().ne(0).any(-1)
        xyz, rotation = apply_region_rigid(xyz, rotation, pivots[level],
                                           values[:, :3], values[:, 3:])
        for child_level in range(level + 1, len(regions)):
            pivots[child_level], _ = apply_region_rigid(
                pivots[child_level], identity, pivots[level], values[:, :3],
                values[:, 3:])
    # libgs quaternion_multiply normalizes even an identity rotation. Preserve
    # the raw previous pose on zero paths while retaining STE opening gradients.
    exact_xyz = torch.where(path_active[:, None], xyz, base_xyz)
    exact_rotation = torch.where(path_active[:, None], rotation, base_rotation)
    return (exact_xyz.detach() + (xyz - xyz.detach()),
            exact_rotation.detach() + (rotation - rotation.detach()))


class BinaryRefinementMotion(SparseHierarchicalMotion):
    """Root motion plus conditional child residuals, without local GS tensors."""

    layout_version = LAYOUT_VERSION
    config_class = BinaryRefinementConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_motion_levels = config.max_motion_levels

    def setup(self, xyz, rotation, extent, warm_start=True, previous_rows=None,
              previous_new_count=None, frame_index=None, layout_override=None):
        if (not len(xyz) or rotation.shape != (len(xyz), 4) or
                not math.isfinite(float(extent)) or extent <= 0):
            raise ValueError('Binary motion requires valid positions, rotations and extent')
        started = time.perf_counter()
        previous_layout = self.materialized_layout
        if layout_override is None:
            layout = stable_binary_forest(xyz, self.config.coarse_region_gs,
                self.config.min_child_gs, self.config.max_motion_levels)
        else:
            layout = _binary_layout(layout_override['index'], layout_override['counts'],
                layout_override['parents'], xyz, layout_override.get('pivots'))
        if len(layout['counts']) != self.num_motion_levels:
            raise ValueError('Binary motion layout depth differs from configured depth')
        self.previous_layout = _copy_layout(previous_layout)
        self.active_layout = layout
        self.materialized_layout = _copy_layout(layout)
        self.index, self.counts, self.parents = (layout['index'], layout['counts'],
                                               layout['parents'])
        self.extent = float(extent)
        for name, value in (('base_xyz', xyz.detach().clone()),
                            ('base_rotation', rotation.detach().clone())):
            if name in self._buffers:
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)
        self.region_translation = nn.ParameterList(
            [nn.Parameter(xyz.new_zeros(count, 3)) for count in self.counts])
        self.region_rotation = nn.ParameterList(
            [nn.Parameter(xyz.new_zeros(count, 3)) for count in self.counts])
        self.refine_logits = nn.ParameterList([
            nn.Parameter(xyz.new_full((len(rows),), float(self.config.initial_gate_logit)))
            for rows in layout['refine_rows']])
        self._make_optimizers()
        self.hard_masks = None
        self.hard_storage_cache = None
        self.last_step = 0
        self.current_phase = None
        self.committed = False
        self.tree_probe_complete = True
        self.probe_seen_views = {}
        self.last_metrics = {}
        topology_seconds = time.perf_counter() - started
        self.stage_seconds = dict(tree_topology=topology_seconds, tree_rebuild=topology_seconds)
        self.tree_setup = dict(frame=frame_index, builder='spatial', probe_steps=0,
            initialization_decay=0., action='initial' if previous_layout is None else 'rebuild',
            old_region_counts=list(previous_layout['counts']) if previous_layout else None,
            region_counts=list(self.counts), topology_seconds=topology_seconds,
            coarse_region_gs=self.config.coarse_region_gs,
            min_child_gs=self.config.min_child_gs,
            max_motion_levels=self.config.max_motion_levels,
            semantics='Deterministic longest-axis median spatial forest rebuilt from surviving '
                'base GS every frame. Only real binary children receive motion. A closed refine '
                'gate inherits its parent prediction and closes all descendants. Root zero '
                'motion may still refine. No per-GS pose gates or motion residuals.')

    def motion_parameters(self):
        return [*self.region_translation, *self.region_rotation]

    def gate_parameters(self):
        return list(self.refine_logits)

    def _make_motion_optimizer(self, scale=1.):
        return torch.optim.Adam([
            dict(params=self.region_translation,
                 lr=self.config.region_translation_lr * scale),
            dict(params=self.region_rotation,
                 lr=self.config.region_rotation_lr * scale)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)

    def _make_optimizers(self):
        self.motion_optimizer = self._make_motion_optimizer()
        # Keep a valid optimizer even when a tiny forest has no internal node.
        self.gate_optimizer = torch.optim.Adam([
            dict(params=self.refine_logits, lr=self.config.gate_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.gate_adam_eps)
        self.optimizer = self.motion_optimizer

    def phase(self, step):
        return self.stage(step)

    def prepare_step(self, step):
        self.last_step = int(step)
        return self._configure_phase(step)

    def _configure_phase(self, step, restoring=False):
        phase = self.stage(step)
        coarse = phase == 'dense_fit' and step <= self.config.coarse_fit_steps
        signature = (phase, coarse)
        if signature == self.current_phase:
            return phase
        self._clear_gradients()
        if phase in {'gate_refine', 'incremental'} and self.hard_masks is None:
            self.hard_masks = self._hard_gate_masks()
        if phase == 'gate_refine' and not restoring:
            self.motion_optimizer = self._make_motion_optimizer(self.config.refine_lr_scale)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if phase in {'dense_fit', 'gate_compress', 'gate_refine'}:
            for level in range(len(self.counts)):
                if not coarse or level == 0:
                    self.region_translation[level].requires_grad_(True)
                    self.region_rotation[level].requires_grad_(True)
        if phase == 'gate_compress':
            for parameter in self.gate_parameters():
                parameter.requires_grad_(True)
        self.optimizer = self.motion_optimizer
        self.current_phase = signature
        return phase

    def gate_probabilities(self, step, *, use_temperature=True):
        temperature = self._temperature(step) if use_temperature else 1.
        return [torch.sigmoid(value / temperature) for value in self.refine_logits]

    def _hard_gate_masks(self):
        self._check_finite_gates()
        probabilities = self.gate_probabilities(self.config.motion_steps)
        return dict(refine=[value.detach() >= self.config.gate_hard_threshold
                            for value in probabilities])

    def refinement_masks(self, step):
        if self.stage(step) == 'dense_fit':
            opened = step > self.config.coarse_fit_steps
            return [torch.full_like(value, opened, dtype=torch.bool)
                    for value in self.refine_logits]
        if self.hard_masks is not None and self.stage(step) in {'gate_refine', 'incremental'}:
            return self.hard_masks['refine']
        return self._hard_gate_masks()['refine']

    def _paths(self, gates):
        paths = [self.base_xyz.new_ones(self.counts[0])]
        for level, parent in enumerate(self.parents):
            node_gate = self.base_xyz.new_zeros(self.counts[level])
            node_gate = node_gate.index_copy(0, self.active_layout['refine_rows'][level],
                                             gates[level].to(self.base_xyz))
            paths.append(paths[level][parent] * node_gate[parent])
        return paths

    def _region_motion(self, integer=False):
        quantize = integer_symbols if integer else quantized_symbols
        symbols = [quantize(torch.cat((translation, rotation), dim=-1),
                            self.motion_steps_vector)
                   for translation, rotation in zip(self.region_translation,
                                                    self.region_rotation)]
        return symbols if integer else [value * self.motion_steps_vector for value in symbols]

    def forward(self, step, training=False):
        if training:
            self.prepare_step(step)
        phase = self.stage(step)
        if phase == 'dense_fit':
            gates = [torch.full_like(value, float(step > self.config.coarse_fit_steps))
                     for value in self.refine_logits]
        elif phase == 'gate_compress':
            probabilities = self.gate_probabilities(step)
            gates = [(value >= self.config.gate_hard_threshold).to(value)
                     + (value - value.detach()) for value in probabilities]
        else:
            gates = self.refinement_masks(step)
        return _conditional_transform(self.base_xyz, self.base_rotation,
            self.active_layout, self._region_motion(), self._paths(gates))

    def pose(self, step=None, training=False):
        return self(self.last_step if step is None else step, training=training)

    @torch.no_grad()
    def _check_finite_gates(self, *, gradients=False):
        for level, value in enumerate(self.refine_logits):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Nonfinite refinement gate at level={level}, step={self.last_step}')
            if gradients and value.grad is not None and not bool(torch.isfinite(value.grad).all()):
                raise FloatingPointError(f'Nonfinite refinement gradient at level={level}, step={self.last_step}')

    def optimizer_step(self):
        phase = self.stage(self.last_step)
        if phase in {'gate_refine', 'incremental'}:
            paths = self._paths(self.refinement_masks(self.last_step))
            for level in range(len(self.counts)):
                for parameter in (self.region_translation[level], self.region_rotation[level]):
                    self._mask_optimizer_rows(parameter, paths[level].bool())
        self.motion_optimizer.step()
        if phase == 'gate_compress':
            self._check_finite_gates(gradients=True)
            self.gate_optimizer.step()
            self._check_finite_gates()
        self.optimizer_zero_grad()

    def optimizer_zero_grad(self):
        self.motion_optimizer.zero_grad(set_to_none=True)
        self.gate_optimizer.zero_grad(set_to_none=True)

    def _hard_masks_for_step(self, step):
        gates = self.refinement_masks(step)
        paths = self._paths(gates)
        xyz, rotation = _conditional_transform(self.base_xyz, self.base_rotation,
            self.active_layout, self._region_motion(), paths)
        changed = (xyz != self.base_xyz).any(-1) | (rotation != self.base_rotation).any(-1)
        return dict(refine=gates, m=changed.detach())

    def _symbol_support(self, differentiable=False):
        result = []
        for translation, rotation in zip(self.region_translation, self.region_rotation):
            raw = torch.cat((translation, rotation), dim=-1)
            # Range/finiteness checks are performed by integer_symbols at
            # export. Training needs a detached zero-bucket decision only.
            symbols = (raw.detach() / self.motion_steps_vector).round()
            hard = (symbols != 0).any(-1).to(raw)
            if differentiable:
                soft = -torch.expm1(-(raw / self.motion_steps_vector).abs().sum(-1))
                hard = hard + (soft - soft.detach())
            result.append(hard)
        return result

    def _participation(self, gates, support):
        """Probability that at least one reachable motion record affects each GS."""
        # Backward recursion respects parent/child dependence. A child is
        # reached only after its parent's gate; this is not an independent OR.
        conditional = self.base_xyz.new_zeros(len(self.base_xyz))
        for level in reversed(range(len(self.counts))):
            owner = self.index[level]
            valid = owner >= 0
            if not self.counts[level]:
                continue
            own = support[level][owner.clamp_min(0)] * valid.to(self.base_xyz)
            if level < len(self.counts) - 1:
                node_gate = self.base_xyz.new_zeros(self.counts[level])
                node_gate = node_gate.index_copy(0, self.active_layout['refine_rows'][level],
                                                 gates[level].to(self.base_xyz))
                child = conditional * node_gate[owner.clamp_min(0)] * valid.to(self.base_xyz)
            else:
                child = conditional
            conditional = own + (1 - own) * child
        return conditional

    def storage_parts(self, step, hard=False):
        fixed = hard or self.stage(step) != 'gate_compress'
        gates = self.refinement_masks(step) if fixed else self.gate_probabilities(step)
        paths = self._paths(gates)
        support = self._symbol_support(differentiable=not fixed)
        node_count = sum(value.sum() for value in paths)
        child_count = sum((value.sum() for value in paths[1:]), self.base_xyz.new_zeros(()))
        gate_count = sum((paths[level][rows].sum()
                          for level, rows in enumerate(self.active_layout['refine_rows'])),
                         self.base_xyz.new_zeros(()))
        motion_count = sum((path * active).sum() for path, active in zip(paths, support))
        gs_count = self._participation(gates, support).sum()
        return dict(motion_region_bytes=motion_count * 24.,
                    motion_index_bytes=motion_count * 4.,
                    pivot_bytes=motion_count * 12.,
                    topology_bytes=node_count * 4. + child_count * 4.
                        + gate_count * 4. + gs_count * 5.,
                    refinement_mask_bytes=gate_count,
                    gs_index_bytes=gs_count * 4.,
                    motion_metadata_bytes=self.base_xyz.new_tensor(
                        float(4 + 4 * len(self.counts) + 6 * 4)))

    def storage_bit_counts(self, step, hard=False):
        fixed = hard or self.stage(step) != 'gate_compress'
        gates = self.refinement_masks(step) if fixed else self.gate_probabilities(step)
        paths = self._paths(gates)
        count = sum((paths[level][rows].sum()
                     for level, rows in enumerate(self.active_layout['refine_rows'])),
                    self.base_xyz.new_zeros(()))
        return dict(m_bits=0, z_bits=int(count) * 8 if fixed else count * 8,
                    refinement_bits=int(count) * 8 if fixed else count * 8)

    def gs_activity_fraction(self, step, hard=False):
        fixed = hard or self.stage(step) != 'gate_compress'
        gates = self.refinement_masks(step) if fixed else self.gate_probabilities(step)
        return self._participation(gates, self._symbol_support()).mean()

    @torch.no_grad()
    def register_materialized_layout(self, materialized_xyz, birth_sources):
        sources = torch.as_tensor(birth_sources, device=self.base_xyz.device, dtype=torch.long)
        if (sources.ndim != 1 or len(materialized_xyz) != len(self.base_xyz) + len(sources)
                or (sources.numel() and bool(((sources < 0) | (sources >= len(self.base_xyz))).any()))):
            raise ValueError('Binary birth sources must map each retained newborn to an old GS')
        self.materialized_layout = _copy_layout(stable_binary_forest(materialized_xyz,
            self.config.coarse_region_gs, self.config.min_child_gs,
            self.config.max_motion_levels))
        return self.materialized_layout

    @torch.no_grad()
    def compensation_deletions(self, opacity, previous_new_count):
        if not self.committed:
            raise ValueError('Binary KEEP protection requires a committed prior frame')
        opacity = opacity.detach().flatten()
        if (isinstance(previous_new_count, bool) or not isinstance(previous_new_count, int)
                or previous_new_count < 0 or len(opacity) != len(self.base_xyz) + previous_new_count):
            raise ValueError('Binary deletion candidates disagree with old/new counts')
        protected = torch.zeros_like(opacity, dtype=torch.bool)
        protected[:len(self.base_xyz)] = ~self._hard_masks_for_step(
            self.config.motion_steps)['m'].to(opacity.device)
        candidates = torch.where(~protected)[0]
        deleted = candidates[torch.argsort(opacity[candidates], stable=True)[:previous_new_count]]
        return deleted, dict(enabled=True, protected_keep_old_count=int(protected.sum()),
            eligible_count=len(candidates), requested_count=previous_new_count,
            deleted_count=len(deleted), deleted_keep_count=int(protected[deleted].sum()),
            semantics='Exact unchanged old poses are protected; all newborns remain eligible.')

    @torch.no_grad()
    def statistics(self):
        step = max(self.last_step, 1)
        gates = self.refinement_masks(step)
        paths = self._paths(gates)
        support = self._symbol_support()
        levels = []
        for level, count in enumerate(self.counts):
            reachable = paths[level].bool()
            entry = dict(level=level, nodes=count, reachable_nodes=int(reachable.sum()),
                         nonzero_motion_records=int((reachable & support[level].bool()).sum()))
            if level < len(gates):
                rows = self.active_layout['refine_rows'][level]
                visited = reachable[rows]
                entry.update(internal_nodes=len(rows), reachable_refine_gates=int(visited.sum()),
                             refine_open=int((visited & gates[level]).sum()),
                             refine_closed=int((visited & ~gates[level]).sum()))
            levels.append(entry)
        changed = self._hard_masks_for_step(step)['m']
        terminal_depth = torch.zeros(len(changed), device=self.base_xyz.device,
                                     dtype=torch.long)
        for level in range(1, len(self.counts)):
            if not self.counts[level]:
                continue
            owner = self.index[level]
            reached = (owner >= 0) & paths[level][owner.clamp_min(0)].bool()
            terminal_depth[reached] = level
        depth_counts = torch.bincount(terminal_depth, minlength=len(self.counts))
        return dict(region_counts=list(self.counts), levels=levels,
            gs=dict(denominator=len(changed), moved=int(changed.sum()), keep=int((~changed).sum()),
                    terminal_depth_counts=depth_counts.cpu().tolist(),
                    terminal_depth_semantics='Selected terminal path depth over all old GS; '
                        'this is independent of whether the selected motion quantizes to zero.'),
            semantics='Root quantization zero is KEEP unless an enabled descendant adds motion. '
                'Refine gates enable conditional child residuals; closing an ancestor closes '
                'every descendant. Terminal motion is shared region motion without per-GS tensors.')

    @torch.no_grad()
    def _sparse_parameter_arrays(self):
        gates = self.refinement_masks(self.config.motion_steps)
        paths = self._paths(gates)
        symbols = self._region_motion(integer=True)
        support = [(value != 0).any(-1) for value in symbols]
        participating = self._participation(gates, [value.to(self.base_xyz) for value in support]).bool()
        gs_rows = torch.where(participating)[0]
        terminal_level = torch.zeros(len(gs_rows), device=self.base_xyz.device, dtype=torch.uint8)
        terminal_node = self.index[0, gs_rows].clone()

        def array(value, dtype):
            return value.detach().cpu().numpy().astype(dtype)

        arrays = dict(representation=np.array(self.layout_version),
            motion_export_format=np.array('sparse_parameters'),
            base_digest=np.array(self._base_digest()),
            source_row_count=np.array(len(self.base_xyz), dtype=np.uint32),
            counts=np.asarray(self.counts, dtype=np.int32),
            motion_steps=array(self.motion_steps_vector, np.float32),
            gs_rows=array(gs_rows, np.uint32))
        for level in range(len(self.counts)):
            reachable = paths[level].bool()
            nodes = torch.where(reachable)[0]
            motion_rows = torch.where(reachable & support[level])[0]
            arrays[f'node_rows_l{level}'] = array(nodes, np.uint32)
            arrays[f'motion_rows_l{level}'] = array(motion_rows, np.uint32)
            arrays[f'motion_symbols_l{level}'] = array(symbols[level][motion_rows], np.int32)
            arrays[f'motion_pivots_l{level}'] = array(self.active_layout['pivots'][level][motion_rows], np.float32)
            if level:
                arrays[f'parents_l{level - 1}'] = array(self.parents[level - 1][nodes], np.uint32)
                owner = self.index[level, gs_rows]
                open_row = owner >= 0
                if self.counts[level]:
                    open_row = open_row & reachable[owner.clamp_min(0)]
                if bool(open_row.any()):
                    terminal_level[open_row] = level
                    terminal_node[open_row] = owner[open_row]
            if level < len(gates):
                rows = self.active_layout['refine_rows'][level]
                selected = reachable[rows]
                arrays[f'refine_rows_l{level}'] = array(rows[selected], np.uint32)
                arrays[f'refine_mask_l{level}'] = array(gates[level][selected], np.bool_)
        arrays['terminal_level'] = array(terminal_level, np.uint8)
        arrays['terminal_node'] = array(terminal_node, np.uint32)
        return arrays

    @torch.no_grad()
    def export_motion(self, path):
        if not self.committed and self.stage(self.last_step) != 'gate_refine':
            raise RuntimeError('Binary export requires fixed refinement gates')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._sparse_parameter_arrays()
        np.savez(path, **arrays)
        numerical = sum(value.nbytes for key, value in arrays.items()
                        if key.startswith('motion_symbols_') or key.startswith('motion_pivots_')
                        or key == 'motion_steps')
        parameter_bytes = self.raw_parameter_bytes(arrays)
        return dict(path=str(path), archive_bytes=path.stat().st_size,
            motion_export_format='sparse_parameters', layout_encoding='binary_enabled_path_arrays',
            layout_dependency=None, parameter_bytes=parameter_bytes,
            numerical_payload_bytes=numerical,
            structure_payload_bytes=parameter_bytes - numerical,
            parameter_fields=[key for key in arrays if key not in {
                'representation', 'motion_export_format', 'base_digest'}],
            semantics='Ordinary sparse arrays: reachable node/parent IDs and internal bool refine '
                'decisions; int32 symbols and float32 centers only for reachable nonzero regional '
                'motion; GS row IDs and terminal-node ownership only for nonzero motion paths. '
                'Steps, counts and source row count are charged. Closed descendants have no records. '
                'Archive bytes additionally include NPZ headers and verification metadata.')

    @classmethod
    def decode_export_motion(cls, path, base_xyz, base_rotation):
        with np.load(path, allow_pickle=False) as packet:
            if str(packet['representation'].item()) != cls.layout_version:
                raise ValueError('Binary motion representation mismatch')
            xyz = torch.as_tensor(base_xyz).detach()
            rotation = torch.as_tensor(base_rotation, device=xyz.device, dtype=xyz.dtype).detach()
            digest = hashlib.sha256()
            for value in (xyz, rotation):
                data = value.cpu().contiguous().numpy()
                digest.update(str(tuple(data.shape)).encode())
                digest.update(data.tobytes())
            if digest.hexdigest() != str(packet['base_digest'].item()):
                raise ValueError('Binary motion base digest mismatch')
            count = packet['source_row_count']
            counts = packet['counts']
            if (count.dtype != np.uint32 or count.shape != () or int(count) != len(xyz)
                    or counts.dtype != np.int32 or counts.ndim != 1 or not 1 <= len(counts) <= 3
                    or counts[0] < 1 or (counts < 0).any()):
                raise ValueError('Invalid binary motion dimensions')
            counts = counts.tolist()
            depth = len(counts)

            def ids(name, limit, expected=None):
                value = packet[name]
                if (value.dtype != np.uint32 or value.ndim != 1
                        or (expected is not None and value.shape != (expected,))
                        or (value.size and (value >= limit).any())):
                    raise ValueError(f'Invalid binary indices: {name}')
                return value.astype(np.int64)

            def unique_ids(name, limit):
                value = ids(name, limit)
                if value.size and (value[1:] <= value[:-1]).any():
                    raise ValueError(f'Binary indices must be ordered and unique: {name}')
                return value

            gs_rows = unique_ids('gs_rows', len(xyz))
            terminal = packet['terminal_level']
            if (terminal.dtype != np.uint8 or terminal.shape != (len(gs_rows),)
                    or (terminal >= depth).any()):
                raise ValueError('Invalid binary terminal levels')
            terminal_node = ids('terminal_node', max(counts), len(gs_rows))
            steps_np = packet['motion_steps']
            if (steps_np.dtype != np.float32 or steps_np.shape != (6,)
                    or not np.isfinite(steps_np).all() or (steps_np <= 0).any()):
                raise ValueError('Invalid binary quantization steps')
            steps = torch.as_tensor(steps_np, device=xyz.device, dtype=xyz.dtype)
            parents, nodes, regions, pivots, opened_nodes = [], [], [], [], []
            for level, number in enumerate(counts):
                node = unique_ids(f'node_rows_l{level}', number)
                nodes.append(node)
                if level == 0 and not np.array_equal(node, np.arange(number)):
                    raise ValueError('Binary packet must include every root node')
                if level:
                    link = ids(f'parents_l{level - 1}', counts[level - 1], len(node))
                    if not np.isin(link, nodes[level - 1]).all():
                        raise ValueError('Binary child references an unreachable parent')
                    if not np.isin(link, opened_nodes[level - 1]).all():
                        raise ValueError('Binary child references a closed refinement gate')
                    actual_parents, child_counts = np.unique(link, return_counts=True)
                    if (not np.array_equal(actual_parents, opened_nodes[level - 1])
                            or (child_counts != 2).any()):
                        raise ValueError('Each open binary refinement must contain exactly two children')
                    full_parent = np.full(number, -1, dtype=np.int64)
                    full_parent[node] = link
                    parents.append(full_parent)
                motion_rows = unique_ids(f'motion_rows_l{level}', number)
                if not np.isin(motion_rows, node).all():
                    raise ValueError('Binary motion row must be reachable')
                symbols = packet[f'motion_symbols_l{level}']
                centers = packet[f'motion_pivots_l{level}']
                if (symbols.dtype != np.int32 or symbols.shape != (len(motion_rows), 6)
                        or (len(symbols) and not (symbols != 0).any(-1).all())
                        or centers.dtype != np.float32 or centers.shape != (len(motion_rows), 3)
                        or not np.isfinite(centers).all()):
                    raise ValueError('Invalid binary nonzero motion record')
                region = xyz.new_zeros(number, 6)
                center = xyz.new_zeros(number, 3)
                row_tensor = torch.as_tensor(motion_rows, device=xyz.device)
                region[row_tensor] = torch.as_tensor(symbols, device=xyz.device).to(xyz.dtype) * steps
                center[row_tensor] = torch.as_tensor(centers, device=xyz.device, dtype=xyz.dtype)
                regions.append(region)
                pivots.append(center)
                if level < depth - 1:
                    refine_rows = unique_ids(f'refine_rows_l{level}', number)
                    flags = packet[f'refine_mask_l{level}']
                    if (flags.dtype != np.bool_ or flags.shape != refine_rows.shape
                            or not np.isin(refine_rows, node).all()):
                        raise ValueError('Invalid binary refinement metadata')
                    opened_nodes.append(refine_rows[flags])
            index_np = np.full((depth, len(xyz)), -1, dtype=np.int64)
            for level in reversed(range(depth)):
                selected = terminal == level
                own = terminal_node[selected]
                if (own >= counts[level]).any() or not np.isin(own, nodes[level]).all():
                    raise ValueError('Binary terminal node must be reachable')
                if level < depth - 1 and np.isin(own, opened_nodes[level]).any():
                    raise ValueError('Binary GS ownership cannot stop at an open refinement gate')
                rows = gs_rows[selected]
                index_np[level, rows] = own
                for ancestor in reversed(range(level)):
                    own = parents[ancestor][own]
                    if (own < 0).any():
                        raise ValueError('Binary terminal path has no parent')
                    index_np[ancestor, rows] = own
            index = torch.as_tensor(index_np, device=xyz.device)
            layout = dict(index=index, counts=counts, parents=parents, pivots=pivots)
            paths = [xyz.new_ones(number) for number in counts]
            moved_xyz, moved_rotation = _conditional_transform(xyz, rotation, layout, regions, paths)
            return moved_xyz.cpu().numpy(), moved_rotation.cpu().numpy()

    def capture(self):
        if self.index is None:
            return None
        return dict(layout_version=self.layout_version, config=asdict(self.config),
            state={name: value.detach().clone() for name, value in self.state_dict().items()},
            extent=self.extent, last_step=self.last_step, committed=self.committed,
            hard_masks=({'refine': [value.detach().cpu().clone()
                                  for value in self.hard_masks['refine']]}
                        if self.hard_masks is not None else None),
            active_layout=_copy_layout(self.active_layout),
            previous_layout=_copy_layout(self.previous_layout),
            materialized_layout=_copy_layout(self.materialized_layout),
            tree_setup=self.tree_setup, last_metrics=dict(self.last_metrics),
            stage_seconds=dict(self.stage_seconds),
            motion_optimizer=self.motion_optimizer.state_dict(),
            gate_optimizer=self.gate_optimizer.state_dict())

    def restore(self, packet):
        if packet is None:
            return
        if packet.get('layout_version') != self.layout_version:
            raise ValueError('Binary refinement checkpoint version mismatch')
        saved = self.config_class(**packet['config'])
        saved.validate()
        if asdict(saved) != asdict(self.config):
            raise ValueError('Binary refinement resume requires identical settings')
        state = packet['state']
        xyz, rotation = state['base_xyz'], state['base_rotation']
        self.setup(xyz, rotation, packet['extent'], warm_start=False,
            frame_index=(packet.get('tree_setup') or {}).get('frame'),
            layout_override=packet['active_layout'])
        self.load_state_dict(state)
        self.motion_optimizer.load_state_dict(packet['motion_optimizer'])
        self.gate_optimizer.load_state_dict(packet['gate_optimizer'])
        self.previous_layout = _copy_layout(packet.get('previous_layout'))
        self.materialized_layout = _copy_layout(packet.get('materialized_layout'))
        hard = packet.get('hard_masks')
        self.hard_masks = (dict(refine=[value.to(xyz.device) for value in hard['refine']])
                           if hard is not None else None)
        self.last_step = packet['last_step']
        self.committed = packet['committed']
        self.tree_setup = packet.get('tree_setup')
        self.last_metrics = dict(packet.get('last_metrics', {}))
        self.stage_seconds = dict(packet.get('stage_seconds', {}))
        self.current_phase = None
        self._configure_phase(max(self.last_step, 1), restoring=True)
        if self.committed:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
