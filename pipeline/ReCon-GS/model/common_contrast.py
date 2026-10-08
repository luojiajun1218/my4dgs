"""Trainable common-motion/contrast tree with an exact fixed-width codec.

Each occupied root stores one world rigid transform.  Each effective binary
split stores one six-vector contrast.  For child member weights ``wl, wr`` the
local translations and world-tangent rotations are derived as
``left=-wr*contrast`` and ``right=wl*contrast``.  Unary branches inherit the
parent transform and consume neither a gate nor a contrast record.
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import struct
import time

import torch
from torch import nn
from torch.nn import functional as F
from libgs.utils.general import quaternion_multiply

from .binary_refinement import (BinaryRefinementConfig, BinaryRefinementMotion,
                                _binary_layout, _copy_layout,
                                stable_binary_forest)
from .node_motion import node_conditional_transform, terminal_node_rows, rotation_delta
from .region_rigid import rotation_vector_quaternion
from .residual_motion import (RAW_FIELDS, inverse_unit_quaternion,
                              reference_digest, survivor_rows)
from .storage_aware_binary import (_BitReader, _BitWriter, _children,
                                   _delete_rows, _nearest_source_rows,
                                   _occupancy, _signed_code, _signed_value,
                                   _load_birth_ply, layout_digest)


CODEC_VERSION = 'common_contrast_preorder_v1'
CODEC_MAGIC = b'CCRB'
CODEC_HEADER = struct.Struct('<4sBBH6If32s32s32s')
CODEC_HEADER_BYTES = CODEC_HEADER.size


@dataclass
class CommonContrastConfig(BinaryRefinementConfig):
    """Configuration for the fixed-precision common/contrast backend."""

    fixed_motion_bits: int = 16
    translation_step_extent: float = 1e-5
    rotation_step_rad: float = 1e-4
    rate_support_temperature: float = 0.1
    cache_static_layout: bool = True
    cache_frozen_storage: bool = True
    force_all_gates_open: bool = False

    def validate(self):
        self._validate_binary_config(max_levels=4, rebuild_interval=5)
        if (isinstance(self.fixed_motion_bits, bool) or
                not isinstance(self.fixed_motion_bits, int) or
                not 2 <= self.fixed_motion_bits <= 32):
            raise ValueError('fixed_motion_bits must be an integer in [2, 32]')
        for name in ('translation_step_extent', 'rotation_step_rad',
                     'rate_support_temperature'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        for name in ('cache_static_layout', 'cache_frozen_storage',
                     'force_all_gates_open'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be bool')


def _member_counts(layout):
    result = []
    for level, count in enumerate(layout['counts']):
        owner = layout['index'][level]
        valid = owner >= 0
        result.append(torch.bincount(owner[valid], minlength=count))
    return result


def _effective_splits(layout):
    """Return binary masks, unary-child maps and fixed member weights."""
    occupancy = _occupancy(layout)
    children = _children(layout)
    members = _member_counts(layout)
    gate_positions = [{int(node): i for i, node in enumerate(rows.tolist())}
                      for rows in layout['refine_rows']]
    binary, unary, weights = [], [], []
    for level in range(len(layout['counts']) - 1):
        binary_level = torch.zeros(layout['counts'][level], dtype=torch.bool,
                                   device=layout['index'].device)
        unary_level = torch.full((layout['counts'][level],), -1, dtype=torch.long,
                                 device=layout['index'].device)
        weight_level = layout['pivots'][level].new_zeros(
            (len(layout['refine_rows'][level]), 2))
        for parent, candidates in enumerate(children[level]):
            occupied = [child for child in candidates if bool(occupancy[level + 1][child])]
            if len(occupied) == 1:
                unary_level[parent] = occupied[0]
            elif len(occupied) == 2:
                binary_level[parent] = True
                position = gate_positions[level].get(parent)
                if position is None:
                    raise ValueError('Effective binary split has no gate row')
                left, right = occupied
                total = members[level + 1][left] + members[level + 1][right]
                if int(total) <= 0:
                    raise ValueError('Effective split has no occupied members')
                weight_level[position, 0] = members[level + 1][left] / total
                weight_level[position, 1] = members[level + 1][right] / total
            elif len(occupied) > 2:
                raise ValueError('Common/contrast backend requires a binary forest')
        binary.append(binary_level)
        unary.append(unary_level)
        weights.append(weight_level)
    return occupancy, children, members, gate_positions, binary, unary, weights


def _canonical_quaternion(value):
    """Normalize q/-q with a deterministic tie rule for exact pi rotations."""
    value = F.normalize(value, dim=-1)
    scalar = value[..., :1]
    vector = value[..., 1:]
    # At w=0 choose the sign of the largest-magnitude vector component.
    largest = vector.abs().argmax(-1, keepdim=True)
    tie_component = vector.gather(-1, largest)
    flip = (scalar < 0) | ((scalar == 0) & (tie_component < 0))
    return torch.where(flip, -value, value)


def _principal_rotation_vector(value):
    value = _canonical_quaternion(value)
    vector = value[..., 1:]
    length = vector.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, value[..., :1])
    scale = torch.where(length > 0,
        angle / length.clamp_min(torch.finfo(length.dtype).tiny),
        torch.full_like(length, 2.))
    return vector * scale


def decompose_terminal_transforms(layout, terminal_rotation, terminal_delta):
    """Exactly factor per-GS terminal transforms on the principal SO(3) branch.

    Every GS belonging to one terminal node must supply the same cumulative
    transform.  Relative child rotations must be strictly below pi for a
    unique principal-log contrast.  Pivots use the general displacement
    formula; no centroid identity is assumed.
    """
    index, counts = layout['index'], layout['counts']
    if terminal_rotation.shape != (index.shape[1], 4) or terminal_delta.shape != (index.shape[1], 3):
        raise ValueError('Terminal transforms must contain one cumulative pose per GS')
    occupancy, children, _, positions, binary, unary, weights = _effective_splits(layout)
    paths = [terminal_delta.new_ones(count) for count in counts]
    terminal = terminal_node_rows(layout, paths, selected_only=False)
    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)
    rotations = [terminal_rotation.new_zeros(count, 4) for count in counts]
    deltas = [terminal_delta.new_zeros(count, 3) for count in counts]
    known = [torch.zeros(count, device=terminal.device, dtype=torch.bool) for count in counts]
    for level, count in enumerate(counts):
        for node in range(count):
            rows = torch.where(terminal == offsets[level] + node)[0]
            if not len(rows):
                continue
            q = _canonical_quaternion(terminal_rotation[rows])
            reference = q[0]
            equivalent = torch.minimum((q - reference).abs().amax(-1),
                                       (q + reference).abs().amax(-1))
            if bool((equivalent > 2e-6).any()) or bool(
                    ((terminal_delta[rows] - terminal_delta[rows[0]]).abs().amax(-1) > 2e-6).any()):
                raise ValueError('One terminal node cannot represent differing GS transforms')
            rotations[level][node] = reference
            deltas[level][node] = terminal_delta[rows[0]]
            known[level][node] = True
    contrasts = [terminal_delta.new_zeros(len(rows), 6) for rows in layout['refine_rows']]
    for level in reversed(range(len(counts) - 1)):
        for node in torch.where(occupancy[level])[0].tolist():
            child = [value for value in children[level][node]
                     if bool(occupancy[level + 1][value])]
            if not child:
                if not bool(known[level][node]):
                    raise ValueError('Occupied terminal node has no supplied transform')
                continue
            if not all(bool(known[level + 1][value]) for value in child):
                raise ValueError('Child cumulative transforms are incomplete')
            if len(child) == 1:
                child_node = child[0]
                rotations[level][node] = rotations[level + 1][child_node]
                offset = (layout['pivots'][level + 1][child_node] -
                          layout['pivots'][level][node])
                deltas[level][node] = (deltas[level + 1][child_node] -
                    rotation_delta(rotations[level][node][None], offset[None])[0])
                known[level][node] = True
                continue
            if len(child) != 2 or not bool(binary[level][node]):
                raise ValueError('Decomposition requires effective binary or unary branches')
            left, right = child
            ql, qr = rotations[level + 1][left], rotations[level + 1][right]
            relative = quaternion_multiply(qr[None], inverse_unit_quaternion(ql[None]))
            omega = _principal_rotation_vector(relative)[0]
            if float(omega.norm()) >= math.pi - 1e-6:
                raise ValueError('Relative child rotation is on the non-unique pi branch')
            position = positions[level][node]
            wl, wr = weights[level][position]
            qp = quaternion_multiply(rotation_vector_quaternion((wr * omega)[None]), ql[None])[0]
            pp = layout['pivots'][level][node]
            child_pivots = layout['pivots'][level + 1]
            al = deltas[level + 1][left] - rotation_delta(
                qp[None], (child_pivots[left] - pp)[None])[0]
            ar = deltas[level + 1][right] - rotation_delta(
                qp[None], (child_pivots[right] - pp)[None])[0]
            rotations[level][node] = qp
            deltas[level][node] = wl * al + wr * ar
            contrasts[level][position, :3] = ar - al
            contrasts[level][position, 3:] = omega
            known[level][node] = True
    roots = torch.cat((deltas[0], _principal_rotation_vector(rotations[0])), dim=-1)
    return roots, contrasts


def reconstruct_cumulative_transforms(layout, roots, contrasts):
    """Reconstruct all occupied node transforms from root/common contrasts."""
    counts = layout['counts']
    if roots.shape != (counts[0], 6) or len(contrasts) != len(counts) - 1:
        raise ValueError('Root/contrast shapes disagree with the layout')
    occupancy, children, _, positions, binary, unary, weights = _effective_splits(layout)
    rotations = [rotation_vector_quaternion(roots[:, 3:])]
    deltas = [roots[:, :3]]
    for level in range(len(counts) - 1):
        q = roots.new_zeros(counts[level + 1], 4)
        q[:, 0] = 1
        delta = roots.new_zeros(counts[level + 1], 3)
        for node in torch.where(occupancy[level])[0].tolist():
            child = [value for value in children[level][node]
                     if bool(occupancy[level + 1][value])]
            if not child:
                continue
            if len(child) == 1:
                local = roots.new_zeros(6)
                child_values = [(child[0], local)]
            else:
                position = positions[level][node]
                wl, wr = weights[level][position]
                child_values = [(child[0], -wr * contrasts[level][position]),
                                (child[1], wl * contrasts[level][position])]
            for child_node, local in child_values:
                child_q = quaternion_multiply(
                    rotation_vector_quaternion(local[None, 3:]),
                    rotations[level][node][None])[0]
                offset = (layout['pivots'][level + 1][child_node] -
                          layout['pivots'][level][node])
                child_delta = (rotation_delta(rotations[level][node][None], offset[None])[0] +
                               deltas[level][node] + local[:3])
                q[child_node], delta[child_node] = child_q, child_delta
        rotations.append(q)
        deltas.append(delta)
    return rotations, deltas


class CommonContrastMotion(BinaryRefinementMotion):
    """One root transform plus one contrast transform per effective split."""

    layout_version = CODEC_VERSION
    config_class = CommonContrastConfig

    def __init__(self, config):
        super().__init__(config)
        self.tree_rebuilt = False
        self.previous_frame_index = None

    def setup(self, xyz, rotation, extent, warm_start=True, previous_rows=None,
              previous_new_count=None, frame_index=None, layout_override=None):
        if (not len(xyz) or rotation.shape != (len(xyz), 4) or
                not math.isfinite(float(extent)) or extent <= 0):
            raise ValueError('Common/contrast motion requires valid base poses and extent')
        old_layout = self.materialized_layout
        if layout_override is None:
            if (isinstance(frame_index, bool) or not isinstance(frame_index, int)
                    or frame_index < 1):
                raise ValueError('Common/contrast setup requires a positive frame_index')
            rebuild = old_layout is None or (frame_index - 1) % self.config.tree_rebuild_interval == 0
            if rebuild:
                layout = stable_binary_forest(xyz, self.config.coarse_region_gs,
                    self.config.min_child_gs, self.config.max_motion_levels)
            else:
                old_count = old_layout['index'].shape[1]
                rows = (torch.arange(len(xyz), device=xyz.device) if previous_rows is None else
                        torch.as_tensor(previous_rows, device=xyz.device, dtype=torch.long))
                if rows.shape != (len(xyz),) or bool(((rows < 0) | (rows >= old_count)).any()):
                    raise ValueError('Cached topology reuse requires valid survivor rows')
                if previous_new_count is not None and old_count != len(self.base_xyz) + previous_new_count:
                    raise ValueError('Cached tree and prior birth count disagree')
                layout = _binary_layout(old_layout['index'].to(xyz.device)[:, rows],
                    old_layout['counts'], old_layout['parents'], xyz, old_layout['pivots'])
        else:
            rebuild = old_layout is None or (frame_index is not None and
                (frame_index - 1) % self.config.tree_rebuild_interval == 0)
            layout = _binary_layout(layout_override['index'], layout_override['counts'],
                layout_override['parents'], xyz, layout_override.get('pivots'))
        if len(layout['counts']) != self.config.max_motion_levels:
            raise ValueError('Common/contrast layout depth differs from configured depth')

        started = time.perf_counter()
        self.previous_layout = _copy_layout(old_layout)
        self.active_layout = layout
        self.materialized_layout = _copy_layout(layout)
        self.index, self.counts, self.parents = layout['index'], layout['counts'], layout['parents']
        self.extent = float(extent)
        for name, value in (('base_xyz', xyz.detach().clone()),
                            ('base_rotation', rotation.detach().clone())):
            if name in self._buffers:
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)
        self.root_translation = nn.Parameter(xyz.new_zeros(self.counts[0], 3))
        self.root_rotation = nn.Parameter(xyz.new_zeros(self.counts[0], 3))
        self.contrast_translation = nn.ParameterList([
            nn.Parameter(xyz.new_zeros(len(rows), 3)) for rows in layout['refine_rows']])
        self.contrast_rotation = nn.ParameterList([
            nn.Parameter(xyz.new_zeros(len(rows), 3)) for rows in layout['refine_rows']])
        self.refine_logits = nn.ParameterList([
            nn.Parameter(xyz.new_full((len(rows),), float(self.config.initial_gate_logit)))
            for rows in layout['refine_rows']])
        (self._cached_occupancy, self._cached_children, self._cached_members,
         self._cached_gate_positions, self._cached_binary, self._cached_unary,
         self._cached_weights) = _effective_splits(layout)
        self._cached_split_positions = []
        self._cached_split_children = []
        self._cached_split_weights = []
        for level, rows in enumerate(layout['refine_rows']):
            positions = torch.where(self._cached_binary[level][rows])[0]
            child_pairs = []
            for position in positions.tolist():
                node = int(rows[position])
                occupied = [child for child in self._cached_children[level][node]
                            if bool(self._cached_occupancy[level + 1][child])]
                if len(occupied) != 2:
                    raise AssertionError('Cached effective split is not binary')
                child_pairs.append(occupied)
            self._cached_split_positions.append(positions)
            self._cached_split_children.append(torch.as_tensor(
                child_pairs, device=xyz.device, dtype=torch.long).reshape(-1, 2))
            self._cached_split_weights.append(self._cached_weights[level][positions])
        self._steps = xyz.new_tensor([self.extent * self.config.translation_step_extent] * 3 +
                                     [self.config.rotation_step_rad] * 3)
        self._make_optimizers()
        self.hard_masks = None
        self.hard_storage_cache = None
        self._frozen_storage_parts = {}
        self.last_step = 0
        self.current_phase = None
        self.committed = False
        self.tree_probe_complete = True
        self.probe_seen_views = {}
        self.last_metrics = {}
        elapsed = time.perf_counter() - started
        self.tree_rebuilt = bool(rebuild)
        self.previous_frame_index = frame_index
        self.stage_seconds = dict(tree_topology=elapsed,
                                  tree_rebuild=elapsed if rebuild else 0.,
                                  tree_setup=elapsed)
        self.tree_setup = dict(frame=frame_index, builder='spatial', probe_steps=0,
            action='initial' if old_layout is None else ('rebuild' if rebuild else 'reuse'),
            tree_rebuild_interval=self.config.tree_rebuild_interval,
            tree_rebuilt=bool(rebuild), initialization_decay=0.,
            old_region_counts=list(old_layout['counts']) if old_layout else None,
            region_counts=list(self.counts),
            effective_binary_splits=[int(value.sum()) for value in self._cached_binary],
            unary_branches=[int((value >= 0).sum()) for value in self._cached_unary],
            semantics='Post-deletion member counts define fixed split weights. Binary splits use one '
                'learnable contrast and one subtree gate; unary branches inherit without a parameter.')
        return layout

    def motion_parameters(self):
        return [self.root_translation, self.root_rotation,
                *self.contrast_translation, *self.contrast_rotation]

    def gate_parameters(self):
        return list(self.refine_logits)

    def _make_motion_optimizer(self, scale=1.):
        return torch.optim.Adam([
            dict(params=[self.root_translation, *self.contrast_translation],
                 lr=self.config.region_translation_lr * scale),
            dict(params=[self.root_rotation, *self.contrast_rotation],
                 lr=self.config.region_rotation_lr * scale)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)

    def _make_optimizers(self):
        self.motion_optimizer = self._make_motion_optimizer()
        self.gate_optimizer = torch.optim.Adam([
            dict(params=self.refine_logits, lr=self.config.gate_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.gate_adam_eps)
        self.optimizer = self.motion_optimizer

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
            self.root_translation.requires_grad_(True)
            self.root_rotation.requires_grad_(True)
            if not coarse:
                for parameter in [*self.contrast_translation, *self.contrast_rotation]:
                    parameter.requires_grad_(True)
        if phase == 'gate_compress' and not self.config.force_all_gates_open:
            for parameter in self.refine_logits:
                parameter.requires_grad_(True)
        self.optimizer = self.motion_optimizer
        self.current_phase = signature
        return phase

    def _hard_gate_masks(self):
        if self.config.force_all_gates_open:
            return dict(refine=[torch.ones_like(value, dtype=torch.bool)
                                for value in self.refine_logits])
        return super()._hard_gate_masks()

    def refinement_masks(self, step):
        if self.config.force_all_gates_open:
            return [torch.ones_like(value, dtype=torch.bool) for value in self.refine_logits]
        return super().refinement_masks(step)

    def _quantize(self, raw):
        bits = self.config.fixed_motion_bits
        lower, upper = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        scaled = raw / self._steps
        symbols = torch.round(scaled).clamp(lower, upper)
        decoded = symbols * self._steps
        return decoded + (raw - raw.detach()), symbols.to(torch.int64), scaled

    def _quantized_parameters(self):
        root_raw = torch.cat((self.root_translation, self.root_rotation), dim=-1)
        root, root_symbols, root_scaled = self._quantize(root_raw)
        contrasts, symbols, scaled = [], [], []
        for translation, rotation in zip(self.contrast_translation, self.contrast_rotation):
            value, code, raw_scaled = self._quantize(torch.cat((translation, rotation), dim=-1))
            contrasts.append(value)
            symbols.append(code)
            scaled.append(raw_scaled)
        return root, contrasts, root_symbols, symbols, root_scaled, scaled

    def _regions(self, root, contrasts):
        regions = [root]
        for level in range(len(self.parents)):
            child_motion = root.new_zeros(self.counts[level + 1], 6)
            positions = self._cached_split_positions[level]
            if len(positions):
                selected = contrasts[level][positions]
                weights = self._cached_split_weights[level]
                derived = torch.stack((-weights[:, 1, None] * selected,
                                        weights[:, 0, None] * selected), dim=1)
                child_motion = child_motion.index_copy(
                    0, self._cached_split_children[level].reshape(-1),
                    derived.reshape(-1, 6))
            regions.append(child_motion)
        return regions

    def _gate_values(self, step):
        phase = self.stage(step)
        if phase == 'dense_fit':
            opened = step > self.config.coarse_fit_steps
            return [torch.full_like(value, float(opened)) for value in self.refine_logits]
        if phase == 'gate_compress' and not self.config.force_all_gates_open:
            result = []
            for probability in self.gate_probabilities(step):
                hard = (probability >= self.config.gate_hard_threshold).to(probability)
                result.append(hard + (probability - probability.detach()))
            return result
        return [value.to(self.base_xyz) for value in self.refinement_masks(step)]

    def _paths(self, gates):
        paths = [self._cached_occupancy[0].to(self.base_xyz)]
        for level, parent in enumerate(self.parents):
            factor = self.base_xyz.new_zeros(self.counts[level])
            unary = self._cached_unary[level] >= 0
            factor[unary] = 1.
            rows = self.active_layout['refine_rows'][level]
            binary_positions = self._cached_binary[level][rows]
            if bool(binary_positions.any()):
                factor = factor.index_copy(0, rows[binary_positions],
                    gates[level][binary_positions].to(self.base_xyz))
            paths.append(paths[level][parent] * factor[parent] *
                         self._cached_occupancy[level + 1].to(self.base_xyz))
        return paths

    def forward(self, step, training=False):
        if training:
            self.prepare_step(step)
        root, contrasts, *_ = self._quantized_parameters()
        paths = self._paths(self._gate_values(step))
        selected = self.stage(step) in {'gate_refine', 'incremental'}
        terminal = terminal_node_rows(self.active_layout, paths, selected_only=selected)
        return node_conditional_transform(self.base_xyz, self.base_rotation,
            self.active_layout, self._regions(root, contrasts), paths,
            selected_only=selected, terminal=terminal)

    def optimizer_step(self):
        phase = self.stage(self.last_step)
        if phase in {'gate_refine', 'incremental'}:
            gates = self.refinement_masks(self.last_step)
            paths = self._paths(gates)
            root_active = paths[0].bool()
            for parameter in (self.root_translation, self.root_rotation):
                self._mask_optimizer_rows(parameter, root_active)
            for level, rows in enumerate(self.active_layout['refine_rows']):
                active = paths[level][rows].bool() & self._cached_binary[level][rows] & gates[level].bool()
                for parameter in (self.contrast_translation[level], self.contrast_rotation[level]):
                    self._mask_optimizer_rows(parameter, active)
        self.motion_optimizer.step()
        if phase == 'gate_compress' and not self.config.force_all_gates_open:
            self._check_finite_gates(gradients=True)
            self.gate_optimizer.step()
            self._check_finite_gates()
        self.optimizer_zero_grad()

    def _hard_masks_for_step(self, step):
        xyz, rotation = self(step)
        changed = ((xyz != self.base_xyz).any(-1) |
                   (rotation != self.base_rotation).any(-1))
        return dict(refine=self.refinement_masks(step), m=changed.detach())

    def _support(self, scaled, symbols, differentiable):
        hard = symbols.ne(0).any(-1).to(scaled)
        if not differentiable:
            return hard
        smooth_zero = torch.sigmoid((0.5 - scaled.abs()) /
                                    self.config.rate_support_temperature).prod(-1)
        soft = 1 - smooth_zero
        return hard + (soft - soft.detach())

    def storage_parts(self, step, hard=False):
        phase = self.stage(step)
        frozen = self.config.cache_frozen_storage and phase == 'incremental'
        if frozen and hard in self._frozen_storage_parts:
            return dict(self._frozen_storage_parts[hard])
        gate_fixed = hard or phase != 'gate_compress' or self.config.force_all_gates_open
        support_differentiable = not hard and phase == 'gate_compress'
        gates = (self.refinement_masks(step) if gate_fixed else self.gate_probabilities(step))
        paths = self._paths([value.to(self.base_xyz) for value in gates])
        _, _, root_symbols, split_symbols, root_scaled, split_scaled = self._quantized_parameters()
        root_support = self._support(root_scaled, root_symbols, support_differentiable)
        bits = self.config.fixed_motion_bits
        presence = paths[0].sum()
        payload = (paths[0] * root_support * (6 * bits)).sum()
        flags = self.base_xyz.new_zeros(())
        for level, rows in enumerate(self.active_layout['refine_rows']):
            binary = self._cached_binary[level][rows].to(self.base_xyz)
            reach = paths[level][rows] * binary
            flags = flags + reach.sum()
            opened = gates[level].to(self.base_xyz)
            contrast_reach = reach * opened
            presence = presence + contrast_reach.sum()
            support = self._support(split_scaled[level], split_symbols[level],
                                    support_differentiable)
            payload = payload + (contrast_reach * support * (6 * bits)).sum()
        total_bits = presence + flags + payload
        padding = (self.base_xyz.new_tensor(float((-int(total_bits)) % 8))
                   if hard else self.base_xyz.new_zeros(()))
        result = dict(codec_header_bytes=self.base_xyz.new_tensor(float(CODEC_HEADER_BYTES)),
            payload_padding_bytes=padding / 8,
            stop_refine_flag_bytes=flags / 8,
            motion_presence_flag_bytes=presence / 8,
            fixed_symbol_bytes=payload / 8)
        if frozen:
            self._frozen_storage_parts[hard] = {key: value.detach() for key, value in result.items()}
            return dict(self._frozen_storage_parts[hard])
        return result

    def storage_bit_counts(self, step, hard=False):
        parts = self.storage_parts(step, hard=hard)
        return dict(m_bits=0, z_bits=sum(parts.values()) * 8,
                    refinement_bits=parts['stop_refine_flag_bytes'] * 8)

    def _hard_symbols(self):
        _, _, root, splits, _, _ = self._quantized_parameters()
        return root.detach().cpu().tolist(), [value.detach().cpu().tolist() for value in splits]

    def _encode_payload(self, step):
        writer = _BitWriter()
        roots, splits = self._hard_symbols()
        gates = [value.detach().cpu().tolist() for value in self.refinement_masks(step)]
        occupancy = [value.detach().cpu().tolist() for value in self._cached_occupancy]
        binary = [value.detach().cpu().tolist() for value in self._cached_binary]
        unary = [value.detach().cpu().tolist() for value in self._cached_unary]
        bits = self.config.fixed_motion_bits

        def motion(values):
            present = any(value != 0 for value in values)
            writer.write(int(present))
            if present:
                for value in values:
                    writer.write(_signed_code(value, bits), bits)

        def visit(level, node):
            if level >= len(self.refine_logits):
                return
            if binary[level][node]:
                position = self._cached_gate_positions[level][node]
                opened = bool(gates[level][position])
                writer.write(int(opened))
                if opened:
                    motion(splits[level][position])
                    for child in self._cached_children[level][node]:
                        if occupancy[level + 1][child]:
                            visit(level + 1, child)
            elif unary[level][node] >= 0:
                visit(level + 1, unary[level][node])

        for root, occupied in enumerate(occupancy[0]):
            if occupied:
                motion(roots[root])
                visit(0, root)
        count = writer.bit_count
        return writer.finish(), count

    def export_motion(self, path, *, frame_index, previous_state_digest,
                      post_motion_digest, previous_count, deleted_count, birth_count):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload, bit_count = self._encode_payload(self.config.motion_steps)
        header = CODEC_HEADER.pack(CODEC_MAGIC, 1, 0, CODEC_HEADER_BYTES,
            int(frame_index), int(previous_count), len(self.base_xyz), int(deleted_count),
            int(birth_count), int(bit_count), float(self.extent), bytes.fromhex(previous_state_digest),
            bytes.fromhex(post_motion_digest), layout_digest(self.active_layout))
        path.write_bytes(header + payload)
        if path.stat().st_size != CODEC_HEADER_BYTES + math.ceil(bit_count / 8):
            raise AssertionError('Common/contrast stream size disagrees with encoded bit length')
        return dict(path=str(path), archive_bytes=path.stat().st_size,
            parameter_bytes=path.stat().st_size, payload_bits=bit_count,
            payload_padding_bits=(-bit_count) % 8, codec_header_bytes=CODEC_HEADER_BYTES,
            frame=frame_index, layout_encoding=CODEC_VERSION,
            fixed_motion_bits=self.config.fixed_motion_bits,
            semantics='Each occupied root has one presence bit and optional fixed-width six-vector. '
                'Each reached effective binary split has one refine bit; an open split has one '
                'contrast presence bit and optional six-vector. Unary branches consume no fields.')

    @staticmethod
    def read_header(path):
        packet = Path(path).read_bytes()
        if len(packet) < CODEC_HEADER_BYTES:
            raise ValueError('Truncated common/contrast header')
        values = CODEC_HEADER.unpack(packet[:CODEC_HEADER_BYTES])
        (magic, version, flags, header_size, frame, previous_count, base_count,
         deleted_count, birth_count, bit_count, extent, previous_digest,
         base_digest, topology) = values
        if magic != CODEC_MAGIC or version != 1 or flags != 0 or header_size != CODEC_HEADER_BYTES:
            raise ValueError('Unsupported common/contrast stream')
        if not math.isfinite(extent) or extent <= 0:
            raise ValueError('Invalid scene extent in common/contrast stream')
        if len(packet) != CODEC_HEADER_BYTES + math.ceil(bit_count / 8):
            raise ValueError('Common/contrast stream length disagrees with bit count')
        return dict(codec=CODEC_VERSION, frame=frame, previous_count=previous_count,
            base_count=base_count, deleted_count=deleted_count, birth_count=birth_count,
            payload_bits=bit_count, extent=extent, previous_digest=previous_digest.hex(),
            post_motion_digest=base_digest.hex(), topology_digest=topology.hex(),
            payload=packet[CODEC_HEADER_BYTES:])

    @classmethod
    def decode_motion(cls, path, base_xyz, base_rotation, config, layout,
                      *, coding_state=None):
        header = cls.read_header(path)
        if len(base_xyz) != header['base_count']:
            raise ValueError('Post-deletion base count disagrees with stream')
        if layout_digest(layout).hex() != header['topology_digest']:
            raise ValueError('Deterministic tree digest disagrees with stream')
        occupancy, children, members, positions, binary, unary, weights = _effective_splits(layout)
        reader = _BitReader(header['payload'], header['payload_bits'])
        counts = layout['counts']
        root = base_xyz.new_zeros(counts[0], 6)
        contrasts = [base_xyz.new_zeros(len(rows), 6) for rows in layout['refine_rows']]
        gates = [torch.zeros(len(rows), device=base_xyz.device, dtype=torch.bool)
                 for rows in layout['refine_rows']]
        bits = config.fixed_motion_bits
        steps = base_xyz.new_tensor([float(header['extent']) * config.translation_step_extent] * 3 +
                                    [config.rotation_step_rad] * 3)

        def motion():
            if not reader.read():
                return base_xyz.new_zeros(6), [0] * 6
            values = [_signed_value(reader.read(bits), bits) for _ in range(6)]
            if not any(values):
                raise ValueError('A zero motion must omit its fixed-width payload')
            return base_xyz.new_tensor(values) * steps, values

        coded_root = torch.zeros(counts[0], 6, device=base_xyz.device, dtype=torch.long)
        coded_splits = [torch.zeros_like(value, dtype=torch.long) for value in contrasts]

        def visit(level, node):
            if level >= len(gates):
                return
            if bool(binary[level][node]):
                position = positions[level][node]
                opened = bool(reader.read())
                gates[level][position] = opened
                if opened:
                    value, code = motion()
                    contrasts[level][position] = value
                    coded_splits[level][position] = torch.as_tensor(code, device=base_xyz.device)
                    for child in children[level][node]:
                        if bool(occupancy[level + 1][child]):
                            visit(level + 1, child)
            elif int(unary[level][node]) >= 0:
                visit(level + 1, int(unary[level][node]))

        for node in torch.where(occupancy[0])[0].tolist():
            value, code = motion()
            root[node] = value
            coded_root[node] = torch.as_tensor(code, device=base_xyz.device)
            visit(0, node)
        if reader.cursor != header['payload_bits']:
            raise ValueError('Common/contrast stream has unread bits')

        # Reconstruct child locals from the single stored split contrast.
        regions = [root]
        for level in range(len(gates)):
            child_motion = base_xyz.new_zeros(counts[level + 1], 6)
            for position, node in enumerate(layout['refine_rows'][level].tolist()):
                if not bool(binary[level][node]):
                    continue
                occupied = [child for child in children[level][node]
                            if bool(occupancy[level + 1][child])]
                wl, wr = weights[level][position]
                child_motion[occupied[0]] = -wr * contrasts[level][position]
                child_motion[occupied[1]] = wl * contrasts[level][position]
            regions.append(child_motion)
        # Unary branches pass automatically; binary branches use decoded gates.
        paths = [occupancy[0].to(base_xyz)]
        for level, parent in enumerate(layout['parents']):
            factor = base_xyz.new_zeros(counts[level])
            factor[unary[level] >= 0] = 1.
            rows = layout['refine_rows'][level]
            active = binary[level][rows]
            factor[rows[active]] = gates[level][active].to(base_xyz)
            paths.append(paths[level][parent] * factor[parent] * occupancy[level + 1].to(base_xyz))
        terminal = terminal_node_rows(layout, paths, selected_only=True)
        xyz, rotation = node_conditional_transform(base_xyz, base_rotation, layout,
            regions, paths, selected_only=True, terminal=terminal)
        if coding_state is not None:
            coding_state.update(root_symbols=coded_root, split_symbols=coded_splits,
                                contrasts=contrasts, paths=paths)
        return xyz, rotation, gates, regions, header

    @torch.no_grad()
    def register_materialized_layout(self, materialized_xyz, birth_sources):
        source = torch.as_tensor(birth_sources, device=self.base_xyz.device, dtype=torch.long)
        old_count = len(self.base_xyz)
        if (source.ndim != 1 or len(materialized_xyz) != old_count + len(source) or
                bool(((source < 0) | (source >= old_count)).any())):
            raise ValueError('Materialized newborn count does not match source rows')
        computed = _nearest_source_rows(materialized_xyz[:old_count], materialized_xyz[old_count:])
        full_index = torch.cat((self.index, self.index[:, computed]), dim=1)
        self.materialized_layout = _copy_layout(_binary_layout(full_index, self.counts,
            self.parents, materialized_xyz, self.active_layout['pivots']))
        return self.materialized_layout

    def compensation_deletions(self, opacity, previous_new_count):
        if not self.committed:
            raise ValueError('Common/contrast deletion selection needs a committed prior frame')
        if len(opacity) != len(self.base_xyz) + previous_new_count:
            raise ValueError('Deletion count disagrees with the previous common/contrast tree')
        if previous_new_count == 0:
            return torch.empty(0, dtype=torch.long, device=opacity.device), dict(
                enabled=True, protected_keep_old_count=0, eligible_count=len(opacity),
                requested_count=0, deleted_count=0,
                semantics='No prior births require synchronized deletion.')
        xyz, rotation = self(self.config.motion_steps)
        changed = ((xyz != self.base_xyz).any(-1) | (rotation != self.base_rotation).any(-1))
        protected = torch.zeros(len(opacity), device=opacity.device, dtype=torch.bool)
        protected[:len(self.base_xyz)] = ~changed.to(opacity.device)
        candidates = torch.where(~protected)[0]
        deleted = candidates[torch.argsort(opacity.detach().flatten()[candidates], stable=True)
                             [:previous_new_count]]
        return deleted, dict(enabled=True, protected_keep_old_count=int(protected.sum()),
            eligible_count=len(candidates), requested_count=previous_new_count,
            deleted_count=len(deleted), deleted_keep_count=0,
            semantics='Decoder repeats stable opacity deletion after protecting exact unchanged rows.')

    def gs_activity_fraction(self, step, hard=False):
        return self._hard_masks_for_step(step)['m'].to(self.base_xyz).mean()

    def statistics(self):
        step = (self.config.motion_steps if self.hard_masks is not None
                else max(self.last_step, 1))
        gates = self.refinement_masks(step)
        paths = self._paths(gates)
        _, _, root_symbols, split_symbols, _, _ = self._quantized_parameters()
        levels = []
        for level, count in enumerate(self.counts):
            reached = paths[level].bool() & self._cached_occupancy[level]
            entry = dict(level=level, nodes=count, reachable_nodes=int(reached.sum()))
            if level == 0:
                entry['nonzero_root_records'] = int((reached & root_symbols.ne(0).any(-1)).sum())
            if level < len(gates):
                rows = self.active_layout['refine_rows'][level]
                valid = self._cached_binary[level][rows]
                visited = reached[rows] & valid
                entry.update(effective_binary_splits=int(valid.sum()),
                    unary_branches=int((self._cached_unary[level] >= 0).sum()),
                    reachable_refine_gates=int(visited.sum()),
                    refine_open=int((visited & gates[level]).sum()),
                    refine_closed=int((visited & ~gates[level]).sum()),
                    nonzero_contrast_records=int((visited & gates[level] &
                                                  split_symbols[level].ne(0).any(-1)).sum()))
            levels.append(entry)
        changed = self._hard_masks_for_step(step)['m']
        payload, bits = self._encode_payload(step)
        return dict(region_counts=list(self.counts), levels=levels,
            gs=dict(denominator=len(changed), changed=int(changed.sum()),
                    unchanged=len(changed) - int(changed.sum())),
            fixed_motion_bits=self.config.fixed_motion_bits,
            hard_coded_payload_bits=bits, payload_bytes=len(payload),
            tree_rebuilt=self.tree_rebuilt, tree_setup=self.tree_setup)

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
            raise ValueError('Common/contrast checkpoint version mismatch')
        saved = self.config_class(**packet['config'])
        saved.validate()
        if asdict(saved) != asdict(self.config):
            raise ValueError('Common/contrast resume requires identical settings')
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


class CommonContrastDecoder:
    """Sequential decoder for the common/contrast stream."""

    def __init__(self, config, *, retain_frame_state=False):
        config.validate()
        self.config = config
        self.materialized_layout = None
        self.previous_changed = None
        self.previous_birth_count = 0
        self.last_frame = 0
        self.last_pose_xyz = None
        self.last_pose_rotation = None
        self.retain_frame_state = retain_frame_state
        self.last_frame_state = None

    def advance(self, motion_path, birth_path, raw):
        header = CommonContrastMotion.read_header(motion_path)
        frame = header['frame']
        if frame != self.last_frame + 1:
            raise ValueError(f'Expected frame {self.last_frame + 1}, received {frame}')
        if len(raw['xyz']) != header['previous_count'] or reference_digest(raw) != header['previous_digest']:
            raise ValueError('Previous Gaussian state disagrees with common/contrast stream')
        if header['deleted_count'] != self.previous_birth_count:
            raise ValueError('Deletion count does not match prior birth count')
        deleted = _delete_rows(raw, self.previous_birth_count, self.previous_changed)
        rows = survivor_rows(len(raw['xyz']), deleted)
        base = {name: raw[name][rows].clone() for name in RAW_FIELDS}
        if len(base['xyz']) != header['base_count']:
            raise ValueError('Deterministic deletion disagrees with stream base count')
        rebuild = (frame - 1) % self.config.tree_rebuild_interval == 0
        if rebuild:
            layout = stable_binary_forest(base['xyz'], self.config.coarse_region_gs,
                self.config.min_child_gs, self.config.max_motion_levels)
        else:
            if self.materialized_layout is None:
                raise ValueError('Intermediate frame has no cached common/contrast tree')
            old = self.materialized_layout
            layout = _binary_layout(old['index'].to(base['xyz'].device)[:, rows],
                old['counts'], old['parents'], base['xyz'], old['pivots'])
        coding = {} if self.retain_frame_state else None
        xyz, rotation, gates, regions, decoded = CommonContrastMotion.decode_motion(
            motion_path, base['xyz'], base['rotation'], self.config, layout,
            coding_state=coding)
        if self.retain_frame_state:
            self.last_frame_state = dict(base_xyz=base['xyz'], base_rotation=base['rotation'],
                layout=layout, previous_layout=self.materialized_layout, gates=gates,
                regions=regions, coding=coding, header=decoded)
        changed = ((xyz != base['xyz']).any(-1) | (rotation != base['rotation']).any(-1)).detach()
        base['xyz'], base['rotation'] = xyz, rotation
        if reference_digest(base) != header['post_motion_digest']:
            raise ValueError('Decoded post-motion state digest mismatch')
        births = (_load_birth_ply(birth_path, base['xyz'].device) if birth_path else
                  {name: base[name].new_empty((0,) + base[name].shape[1:]) for name in RAW_FIELDS})
        if len(births['xyz']) != header['birth_count']:
            raise ValueError('Birth PLY count disagrees with stream')
        if len(births['xyz']):
            sources = _nearest_source_rows(base['xyz'], births['xyz'])
            full_index = torch.cat((layout['index'], layout['index'][:, sources]), dim=1)
            materialized_xyz = torch.cat((base['xyz'], births['xyz']), dim=0)
        else:
            full_index, materialized_xyz = layout['index'], base['xyz']
        for name in RAW_FIELDS:
            base[name] = torch.cat((base[name], births[name].to(base[name])), dim=0)
        self.materialized_layout = _copy_layout(_binary_layout(full_index,
            layout['counts'], layout['parents'], materialized_xyz, layout['pivots']))
        self.previous_changed = changed
        self.previous_birth_count = len(births['xyz'])
        self.last_frame = frame
        self.last_pose_xyz = torch.cat((xyz, births['xyz'].to(xyz)), dim=0)
        self.last_pose_rotation = torch.cat((rotation, births['rotation'].to(rotation)), dim=0)
        return base, dict(frame=frame, deleted_rows=deleted.detach().cpu().tolist(),
            birth_count=self.previous_birth_count, changed_old_count=int(changed.sum()),
            tree_rebuilt=rebuild, topology_digest=header['topology_digest'],
            coded_payload_bits=header['payload_bits'], motion_path=str(motion_path),
            birth_path=str(birth_path) if birth_path else None)
