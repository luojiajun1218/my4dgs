"""Rate-aware binary motion hierarchy with a deterministic frame bitstream."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import time

import numpy as np
import torch
from torch import nn

from .binary_refinement import (BinaryRefinementConfig, BinaryRefinementMotion,
                                _binary_layout, _conditional_transform,
                                _copy_layout, stable_binary_forest)
from .residual_motion import RAW_FIELDS, reference_digest, survivor_rows
from .node_motion import node_conditional_transform, terminal_node_rows


CODEC_VERSION = 'storage_aware_binary_preorder_v1'
CODEC_MAGIC = b'SARB'
CODEC_HEADER = struct.Struct('<4sBBH6If32s32s32s')
CODEC_HEADER_BYTES = CODEC_HEADER.size


@dataclass
class StorageAwareBinaryConfig(BinaryRefinementConfig):
    precision_bits: list = None
    translation_steps_extent: list = None
    rotation_steps_rad: list = None
    precision_lr: float = 0.02
    precision_initial_logits: list = None
    rate_support_temperature: float = 0.1
    motion_compute: str = 'reference'
    cache_static_layout: bool = False
    selected_precision_only: bool = False
    cache_frozen_storage: bool = False

    def validate(self):
        self._validate_binary_config(max_levels=4, rebuild_interval=5)
        if self.motion_compute not in {'reference', 'node'}:
            raise ValueError('motion_compute must be reference or node')
        for name in ('cache_static_layout', 'selected_precision_only', 'cache_frozen_storage'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be bool')
        if not self.precision_bits or len(self.precision_bits) < 2:
            raise ValueError('precision_bits must contain at least two coding choices')
        bits = list(self.precision_bits)
        if any(isinstance(value, bool) or not isinstance(value, int) or
               value < 2 or value > 32 for value in bits):
            raise ValueError('precision_bits must contain integers in [2, 32]')
        if bits != sorted(set(bits)):
            raise ValueError('precision_bits must be unique and ascending')
        for name in ('translation_steps_extent', 'rotation_steps_rad'):
            values = list(getattr(self, name) or [])
            if len(values) != len(bits) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f'{name} must have one finite positive value per precision')
            setattr(self, name, [float(value) for value in values])
        if (isinstance(self.precision_lr, bool) or
                not math.isfinite(self.precision_lr) or self.precision_lr <= 0):
            raise ValueError('precision_lr must be finite and positive')
        if (self.precision_initial_logits is None or
                len(self.precision_initial_logits) != len(bits) or
                any(not math.isfinite(float(value)) for value in self.precision_initial_logits)):
            raise ValueError('precision_initial_logits must match precision_bits')
        self.precision_initial_logits = [float(value) for value in self.precision_initial_logits]
        if max(range(len(bits)), key=self.precision_initial_logits.__getitem__) != len(bits) - 1:
            raise ValueError('The initial hard precision must be the highest precision')
        if (isinstance(self.rate_support_temperature, bool) or
                not math.isfinite(self.rate_support_temperature) or
                self.rate_support_temperature <= 0):
            raise ValueError('rate_support_temperature must be finite and positive')


def layout_digest(layout):
    digest = hashlib.sha256()
    digest.update(np.asarray(layout['counts'], dtype='<u4').tobytes())
    digest.update(np.ascontiguousarray(
        layout['index'].detach().cpu().numpy(), dtype='<i4').tobytes())
    for value in layout['parents']:
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype='<u4').tobytes())
    return digest.digest()


def _nearest_source_rows(reference_xyz, query_xyz):
    """Use the same single-worker nearest-source rule as newborn training."""
    if len(query_xyz) == 0:
        return torch.empty(0, dtype=torch.long, device=reference_xyz.device)
    from scipy.spatial import cKDTree
    reference = reference_xyz.detach().cpu().numpy()
    query = query_xyz.detach().cpu().numpy()
    _, source = cKDTree(reference).query(query, k=1, workers=1)
    return torch.as_tensor(source, dtype=torch.long, device=reference_xyz.device)


class _BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current = 0
        self.used = 0
        self.bit_count = 0

    def write(self, value, width=1):
        value, width = int(value), int(width)
        if width < 1 or value < 0 or value >= (1 << width):
            raise ValueError('Bit field does not fit its declared width')
        for offset in range(width):
            self.current |= ((value >> offset) & 1) << self.used
            self.used += 1
            self.bit_count += 1
            if self.used == 8:
                self.data.append(self.current)
                self.current, self.used = 0, 0

    def finish(self):
        if self.used:
            self.data.append(self.current)
            self.current, self.used = 0, 0
        return bytes(self.data)


class _BitReader:
    def __init__(self, payload, bit_count):
        self.payload = memoryview(payload)
        self.bit_count = int(bit_count)
        self.cursor = 0
        if self.bit_count < 0 or len(self.payload) != (self.bit_count + 7) // 8:
            raise ValueError('Invalid binary motion payload length')

    def read(self, width=1):
        width = int(width)
        if width < 1 or self.cursor + width > self.bit_count:
            raise ValueError('Truncated binary motion payload')
        value = 0
        for offset in range(width):
            byte = self.payload[self.cursor >> 3]
            bit = (byte >> (self.cursor & 7)) & 1
            value |= bit << offset
            self.cursor += 1
        return value


def _signed_code(value, bits):
    value, bits = int(value), int(bits)
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if value < lower or value > upper:
        raise ValueError('Motion symbol exceeds selected precision')
    return value & ((1 << bits) - 1)


def _signed_value(value, bits):
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def _occupancy(layout):
    result = []
    index = layout['index']
    for level, count in enumerate(layout['counts']):
        owner = index[level]
        valid = owner >= 0
        result.append(torch.bincount(owner[valid], minlength=count) > 0)
    return result


def _children(layout):
    result = []
    for level, parent in enumerate(layout['parents']):
        groups = [[] for _ in range(layout['counts'][level])]
        for child, owner in enumerate(parent.detach().cpu().tolist()):
            groups[owner].append(child)
        result.append(groups)
    return result


def birth_ply_size(count, sh_degree):
    """Exact binary PLY size produced by GaussianModel.save_ply for this schema."""
    count = int(count)
    if count < 0 or sh_degree < 0:
        raise ValueError('Invalid PLY vertex count or SH degree')
    if count == 0:
        return 0
    coefficients = (sh_degree + 1) ** 2
    properties = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    properties += [f'f_dc_{i}' for i in range(3)]
    properties += [f'f_rest_{i}' for i in range(3 * (coefficients - 1))]
    properties += ['opacity']
    properties += [f'scale_{i}' for i in range(3)]
    properties += [f'rot_{i}' for i in range(4)]
    header = ('ply\nformat binary_little_endian 1.0\n' +
              f'element vertex {count}\n' +
              ''.join(f'property float {name}\n' for name in properties) +
              'end_header\n').encode('ascii')
    return len(header) + count * len(properties) * 4


def _delete_rows(raw, previous_birth_count, previous_changed):
    """Recreate the encoder's opacity pruning without transmitting row IDs."""
    total = len(raw['xyz'])
    if (isinstance(previous_birth_count, bool) or not isinstance(previous_birth_count, int)
            or previous_birth_count < 0 or previous_birth_count >= total):
        if previous_birth_count == 0 and total > 0:
            return torch.empty(0, dtype=torch.long, device=raw['xyz'].device)
        raise ValueError('Invalid previous birth count for synchronized deletion')
    if previous_birth_count == 0:
        return torch.empty(0, dtype=torch.long, device=raw['xyz'].device)
    old_count = total - previous_birth_count
    if previous_changed is None or previous_changed.shape != (old_count,):
        raise ValueError('Previous decoded pose changes do not match old GS rows')
    protected = torch.zeros(total, device=raw['xyz'].device, dtype=torch.bool)
    protected[:old_count] = ~previous_changed.to(device=protected.device, dtype=torch.bool)
    candidates = torch.where(~protected)[0]
    opacity = raw['opacity'].detach().reshape(total, -1)[:, 0]
    deleted = candidates[torch.argsort(opacity[candidates], stable=True)[:previous_birth_count]]
    return deleted


class StorageAwareBinaryMotion(BinaryRefinementMotion):
    """Jointly choose binary refinement, per-node precision and coded bit length."""

    layout_version = CODEC_VERSION
    config_class = StorageAwareBinaryConfig

    def __init__(self, config):
        super().__init__(config)
        self.selector_bits = math.ceil(math.log2(len(config.precision_bits)))
        self.tree_rebuilt = False
        self.previous_frame_index = None

    def setup(self, xyz, rotation, extent, warm_start=True, previous_rows=None,
              previous_new_count=None, frame_index=None, layout_override=None):
        old_layout = self.materialized_layout
        if layout_override is None:
            if (isinstance(frame_index, bool) or not isinstance(frame_index, int)
                    or frame_index < 1):
                raise ValueError('Storage-aware binary setup requires a positive frame_index')
            rebuild = old_layout is None or (frame_index - 1) % self.config.tree_rebuild_interval == 0
            if rebuild:
                layout_override = stable_binary_forest(xyz, self.config.coarse_region_gs,
                    self.config.min_child_gs, self.config.max_motion_levels)
            else:
                old_count = old_layout['index'].shape[1]
                rows = (torch.arange(len(xyz), device=xyz.device) if previous_rows is None else
                        torch.as_tensor(previous_rows, device=xyz.device, dtype=torch.long))
                if (rows.shape != (len(xyz),) or
                        bool(((rows < 0) | (rows >= old_count)).any())):
                    raise ValueError('Cached topology reuse requires valid survivor rows')
                if previous_new_count is not None and old_count != len(self.base_xyz) + previous_new_count:
                    raise ValueError('Cached tree and previous birth count disagree')
                layout_override = _binary_layout(
                    old_layout['index'].to(xyz.device)[:, rows], old_layout['counts'],
                    old_layout['parents'], xyz, old_layout['pivots'])
        else:
            rebuild = old_layout is None or (frame_index is not None and
                (frame_index - 1) % self.config.tree_rebuild_interval == 0)
        start = time.perf_counter()
        super().setup(xyz, rotation, extent, warm_start=False,
            previous_rows=previous_rows, previous_new_count=previous_new_count,
            frame_index=frame_index, layout_override=layout_override)
        self.tree_rebuilt = bool(rebuild)
        self.previous_frame_index = frame_index
        self.tree_setup.update(action='initial' if old_layout is None else
            ('rebuild' if rebuild else 'reuse'), tree_rebuild_interval=self.config.tree_rebuild_interval,
            rebuild_anchor='frame 1, then every fifth frame', tree_rebuilt=bool(rebuild),
            semantics='The deterministic binary tree is rebuilt from the post-deletion base on frames '
                '1, 6, 11, ... . Intermediate frames remap surviving rows through the cached tree. '
                'Newborn rows inherit the deterministic nearest-source leaf and are cached for the next frame.')
        self.stage_seconds['tree_rebuild'] = self.stage_seconds.get('tree_topology', 0.) if rebuild else 0.
        self.stage_seconds['tree_setup'] = time.perf_counter() - start

        initial = xyz.new_tensor(self.config.precision_initial_logits)
        self.precision_logits = nn.ParameterList([
            nn.Parameter(initial[None, :].expand(count, -1).clone())
            for count in self.counts])
        optimizer_options = dict(betas=tuple(self.config.adam_betas), eps=self.config.gate_adam_eps)
        self.gate_optimizer = torch.optim.Adam([
            dict(params=self.refine_logits, lr=self.config.gate_lr),
            dict(params=self.precision_logits, lr=self.config.precision_lr)], **optimizer_options)
        self.optimizer = self.motion_optimizer
        self._cached_occupancy = _occupancy(self.active_layout)
        self._cached_children = _children(self.active_layout)
        self._cached_gate_positions = [dict((int(node), i) for i, node in enumerate(rows.tolist()))
                                      for rows in self.active_layout['refine_rows']]
        self._cached_active_gates = []
        for level, parent in enumerate(self.parents):
            counts = torch.zeros(self.counts[level], device=xyz.device, dtype=torch.int32)
            counts.index_add_(0, parent, self._cached_occupancy[level + 1].to(torch.int32))
            self._cached_active_gates.append(counts > 0)
        self._cached_steps = self._make_candidate_steps()
        self._cached_bits = xyz.new_tensor(self.config.precision_bits)
        self._static_terminal = terminal_node_rows(self.active_layout,
            [xyz.new_ones(count) for count in self.counts], selected_only=False)
        self._fixed_paths = self._fixed_rows = self._fixed_terminal = None
        self._frozen_storage_parts = {}
        return self.active_layout

    def _occupancy_values(self):
        return self._cached_occupancy if self.config.cache_static_layout else _occupancy(self.active_layout)

    def _children_values(self):
        return self._cached_children if self.config.cache_static_layout else _children(self.active_layout)

    def _configure_phase(self, step, restoring=False):
        phase = super()._configure_phase(step, restoring=restoring)
        for parameter in self.precision_logits:
            parameter.requires_grad_(phase == 'gate_compress')
        return phase

    def _precision_probabilities(self, step):
        temperature = self._temperature(step)
        return [torch.softmax(value / temperature, dim=-1) for value in self.precision_logits]

    def _hard_gate_masks(self):
        base = super()._hard_gate_masks()
        base['precision'] = [value.argmax(-1) for value in self._precision_probabilities(
            self.config.motion_steps)]
        return base

    def _precision_weights(self, step, *, differentiable=True):
        phase = self.stage(step)
        result = []
        for level, logits in enumerate(self.precision_logits):
            count = len(logits)
            if phase == 'dense_fit':
                index = torch.full((count,), len(self.config.precision_bits) - 1,
                                   dtype=torch.long, device=logits.device)
                result.append(torch.nn.functional.one_hot(index, len(self.config.precision_bits)).to(logits))
            elif phase == 'gate_compress' and differentiable:
                probability = self._precision_probabilities(step)[level]
                index = probability.detach().argmax(-1)
                hard = torch.nn.functional.one_hot(index, probability.shape[-1]).to(probability)
                result.append(hard + (probability - probability.detach()))
            else:
                if self.hard_masks is not None and phase in {'gate_refine', 'incremental'}:
                    index = self.hard_masks['precision'][level]
                else:
                    index = self._precision_probabilities(step)[level].detach().argmax(-1)
                result.append(torch.nn.functional.one_hot(index, len(self.config.precision_bits)).to(logits))
        return result

    def _make_candidate_steps(self):
        translation = self.base_xyz.new_tensor(self.config.translation_steps_extent) * self.extent
        rotation = self.base_xyz.new_tensor(self.config.rotation_steps_rad)
        return torch.cat((translation[:, None].expand(-1, 3),
                          rotation[:, None].expand(-1, 3)), dim=1)

    def _candidate_steps(self):
        return self._cached_steps if self.config.cache_static_layout else self._make_candidate_steps()

    def _precision_indices(self, step):
        phase = self.stage(step)
        if phase == 'dense_fit':
            return [torch.full((count,), len(self.config.precision_bits) - 1,
                               device=self.base_xyz.device, dtype=torch.long) for count in self.counts]
        if self.hard_masks is not None and phase in {'gate_refine', 'incremental'}:
            return self.hard_masks['precision']
        return [value.detach().argmax(-1) for value in self.precision_logits]

    def _quantized_candidates(self, raw):
        steps = self._candidate_steps()
        bits = torch.as_tensor(self.config.precision_bits, device=raw.device, dtype=raw.dtype)
        lower = -(2. ** (bits - 1))[:, None, None]
        upper = (2. ** (bits - 1) - 1)[:, None, None]
        scaled = raw[None] / steps[:, None, :]
        rounded = torch.round(scaled).clamp(lower, upper)
        decoded = rounded * steps[:, None, :]
        quantized = decoded + (raw - raw.detach())[None]
        return quantized, rounded, scaled

    def _region_motion(self, step=None, *, hard=False, active_rows=None):
        if step is None:
            step = self.config.motion_steps
        output = []
        if self.config.selected_precision_only and (hard or self.stage(step) != 'gate_compress'):
            choices = self._precision_indices(step)
            steps = self._candidate_steps()
            limit_levels = (1 if self.stage(step) == 'dense_fit' and
                            step <= self.config.coarse_fit_steps else len(self.counts))
            for level in range(limit_levels):
                rows = (active_rows[level] if active_rows is not None else
                        torch.arange(self.counts[level], device=self.base_xyz.device))
                raw = torch.cat((self.region_translation[level][rows],
                                 self.region_rotation[level][rows]), dim=-1)
                selected_steps = steps[choices[level][rows]]
                bits = self._cached_bits[choices[level][rows]]
                limit = 2. ** (bits - 1)
                symbols = torch.minimum(torch.maximum(torch.round(raw / selected_steps),
                    -limit[:, None]), limit[:, None] - 1)
                quantized = symbols * selected_steps + (raw - raw.detach())
                output.append(raw.new_zeros((self.counts[level], 6)).index_copy(0, rows, quantized))
            return output
        weights = self._precision_weights(step, differentiable=not hard)
        for level, (translation, rotation) in enumerate(zip(self.region_translation,
                                                            self.region_rotation)):
            raw = torch.cat((translation, rotation), dim=-1)
            candidates, _, _ = self._quantized_candidates(raw)
            output.append(torch.einsum('nk,knc->nc', weights[level], candidates))
        return output

    def _selected_symbols(self, step):
        selections = ([value.argmax(-1) for value in self._precision_weights(
            step, differentiable=False)] if self.config.motion_compute == 'reference'
            else self._precision_indices(step))
        steps = self._candidate_steps()
        symbols = []
        saturation = []
        for level, (translation, rotation) in enumerate(zip(self.region_translation,
                                                            self.region_rotation)):
            raw = torch.cat((translation, rotation), dim=-1).detach()
            choice = selections[level]
            level_steps = steps[choice]
            bits = torch.as_tensor(self.config.precision_bits, device=raw.device)[choice]
            limit = torch.pow(2., bits.to(raw) - 1)
            signed = torch.round(raw / level_steps)
            clipped = torch.minimum(torch.maximum(signed, -limit[:, None]), limit[:, None] - 1)
            saturation.append((signed != clipped).any(-1))
            symbols.append(clipped.to(torch.int64))
        return selections, symbols, saturation

    def forward(self, step, training=False):
        if training:
            self.prepare_step(step)
        phase = self.stage(step)
        if phase == 'dense_fit':
            opened = step > self.config.coarse_fit_steps
            gates = [torch.full_like(value, float(opened)) for value in self.refine_logits]
        elif phase == 'gate_compress':
            gates = []
            for probability in self.gate_probabilities(step):
                hard = (probability >= self.config.gate_hard_threshold).to(probability)
                gates.append(hard + (probability - probability.detach()))
        else:
            gates = self.refinement_masks(step)
        if self.config.motion_compute == 'reference':
            # Build the region and path graphs in their original order.
            return _conditional_transform(self.base_xyz, self.base_rotation,
                self.active_layout, self._region_motion(step), self._paths(gates))
        paths = self._paths(gates)
        selected = phase in {'gate_refine', 'incremental'}
        rows = terminal = None
        if self.config.motion_compute == 'node' and selected:
            if self._fixed_paths is None:
                self._fixed_paths = [value.detach() for value in paths]
                self._fixed_rows = [torch.where(value > 0)[0] for value in paths]
                self._fixed_terminal = terminal_node_rows(self.active_layout, paths, selected_only=True)
            paths, rows, terminal = self._fixed_paths, self._fixed_rows, self._fixed_terminal
        regions = self._region_motion(step, active_rows=rows)
        if not selected:
            terminal = self.active_layout['index'][0] if len(regions) == 1 else self._static_terminal
        return node_conditional_transform(self.base_xyz, self.base_rotation,
            self.active_layout, regions, paths[:len(regions)], selected_only=selected,
            terminal=terminal, active_rows=rows)

    def shadow_frontier_plan(self, step, gate_level, gate_positions=None):
        """Temporarily open one reachable closed frontier without changing logits.

        All harder descendant choices retain their current hard values. Target
        motion rows are exactly those that become reachable because this
        frontier was opened; the already reachable parent rows are excluded.
        """
        if self.stage(step) != 'gate_compress' or self.hard_masks is not None:
            raise ValueError('Shadow exploration requires a soft gate-compress step')
        if (isinstance(gate_level, bool) or not isinstance(gate_level, int)
                or gate_level < 0 or gate_level >= len(self.refine_logits)):
            raise ValueError('Shadow gate_level is outside the refinement hierarchy')
        gates = [value.detach().clone() for value in self.refinement_masks(step)]
        paths = self._paths(gates)
        occupancy = self._occupancy_values()
        internal = self.active_layout['refine_rows'][gate_level]
        internal_occupied = occupancy[gate_level][internal].bool()
        reachable = paths[gate_level][internal].bool() & internal_occupied
        closed = ~gates[gate_level].bool()
        candidate_positions = torch.where(reachable & closed)[0]
        if gate_positions is None:
            positions = candidate_positions
        else:
            positions = torch.as_tensor(
                gate_positions, device=candidate_positions.device, dtype=torch.long)
            if positions.ndim != 1 or len(positions) != len(torch.unique(positions)):
                raise ValueError('Shadow gate positions must be a unique 1-D selection')
            if (len(positions) and (int(positions.min()) < 0
                                    or int(positions.max()) >= len(gates[gate_level]))):
                raise ValueError('Shadow gate position is outside the selected level')
            candidate_mask = torch.zeros_like(gates[gate_level], dtype=torch.bool)
            candidate_mask[candidate_positions] = True
            if len(positions) and not bool(candidate_mask[positions].all()):
                raise ValueError('Shadow selection contains a non-frontier gate')
        nodes = internal[positions]
        shadow_gates = [value.clone() for value in gates]
        shadow_gates[gate_level][positions] = True
        shadow_paths = self._paths(shadow_gates)
        target_rows = [(shadow.bool() & ~base.bool() & occupied.bool())
                       for base, shadow, occupied in zip(paths, shadow_paths, occupancy)]
        if any(bool(mask.any()) for mask in target_rows[:gate_level + 1]):
            raise AssertionError('Shadow frontier unexpectedly targets an ancestor row')
        return dict(step=int(step), gate_level=gate_level,
                    candidate_gate_positions=candidate_positions,
                    candidate_gate_nodes=internal[candidate_positions],
                    gate_positions=positions, gate_nodes=nodes,
                    source_gates=gates, shadow_gates=shadow_gates,
                    source_paths=paths, shadow_paths=shadow_paths,
                    target_rows=target_rows)

    def shadow_frontier_pose(self, step, plan):
        """Render a planned frontier with detached hard gate/precision choices."""
        if plan['gate_level'] >= len(self.refine_logits):
            raise ValueError('Shadow plan does not belong to this hierarchy')
        if len(plan['shadow_paths']) != len(self.counts):
            raise ValueError('Shadow plan depth differs from this hierarchy')
        regions = self._region_motion(step, hard=True)
        return node_conditional_transform(
            self.base_xyz, self.base_rotation, self.active_layout, regions,
            plan['shadow_paths'][:len(regions)], selected_only=False,
            terminal=self._static_terminal)

    def _hard_masks_for_step(self, step):
        if self.config.motion_compute == 'reference':
            return super()._hard_masks_for_step(step)
        xyz, rotation = self(step)
        changed = ((xyz != self.base_xyz).any(-1) | (rotation != self.base_rotation).any(-1))
        return dict(refine=self.refinement_masks(step), m=changed.detach())

    def optimizer_step(self):
        if self.stage(self.last_step) == 'gate_compress':
            for level, value in enumerate(self.precision_logits):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f'Nonfinite precision logits at level={level}')
                if value.grad is not None and not bool(torch.isfinite(value.grad).all()):
                    raise FloatingPointError(f'Nonfinite precision gradients at level={level}')
        super().optimizer_step()

    def _support_by_precision(self, raw):
        candidates, rounded, scaled = self._quantized_candidates(raw)
        hard = (rounded != 0).any(-1).to(raw)
        distance = scaled.abs()
        smooth_zero = torch.sigmoid((0.5 - distance) / self.config.rate_support_temperature)
        soft = 1 - smooth_zero.prod(-1)
        return (hard + (soft - soft.detach())).transpose(0, 1), hard.transpose(0, 1).bool()

    def _hard_node_choices(self, step):
        selections, symbols, saturation = self._selected_symbols(step)
        supports = [value.ne(0).any(-1) for value in symbols]
        return selections, symbols, supports, saturation

    def _hard_rate_bits(self, step):
        gates = self.refinement_masks(step)
        occupancy = self._occupancy_values()
        children = self._children_values()
        _, symbols, _, _ = self._hard_node_choices(step)
        precision = ([value.argmax(-1) for value in self._precision_weights(
            step, differentiable=False)] if self.config.motion_compute == 'reference'
            else self._precision_indices(step))
        if self.config.cache_static_layout:
            paths = self._paths(gates)
            total = self.base_xyz.new_zeros(())
            for level in range(len(self.counts)):
                reached = paths[level].bool() & occupancy[level]
                nonzero = symbols[level].ne(0).any(-1)
                costs = 1 + nonzero.to(self.base_xyz) * (
                    self.selector_bits + 6 * self._cached_bits[precision[level]])
                if level < len(gates):
                    costs = costs + self._cached_active_gates[level].to(costs)
                total = total + (reached.to(costs) * costs).sum()
            return int(total.item())
        return sum(self._hard_rate_bits_node(0, root, None, occupancy, children,
            gates, symbols, precision) for root in torch.where(occupancy[0])[0].tolist())

    def _refine_position(self, level, node):
        if self.config.cache_static_layout:
            return self._cached_gate_positions[level][node]
        rows = self.active_layout['refine_rows'][level]
        matches = torch.where(rows == node)[0]
        if len(matches) != 1:
            raise ValueError('Internal node has no unique refinement gate')
        return int(matches[0])

    def _hard_rate_bits_node(self, level, node, paths, occupancy, children,
                             gates, symbols, precision):
        if not bool(occupancy[level][node]):
            return 0
        result = 1
        selected = symbols[level][node]
        if bool((selected != 0).any()):
            result += self.selector_bits + 6 * self.config.precision_bits[int(precision[level][node])]
        if level < len(self.refine_logits):
            child_nodes = [child for child in children[level][node]
                           if bool(occupancy[level + 1][child])]
            if child_nodes:
                result += 1
                if bool(gates[level][self._refine_position(level, node)] >= self.config.gate_hard_threshold):
                    for child in child_nodes:
                        result += self._hard_rate_bits_node(level + 1, child, paths,
                            occupancy, children, gates, symbols, precision)
        return result

    def storage_parts(self, step, hard=False):
        phase = self.stage(step)
        frozen = self.config.cache_frozen_storage and phase == 'incremental'
        if frozen and hard in self._frozen_storage_parts:
            return dict(self._frozen_storage_parts[hard])
        occupancy = self._occupancy_values()
        gates = (self.refinement_masks(step) if hard or phase != 'gate_compress'
                 else self.gate_probabilities(step))
        gate_prob = [value.to(self.base_xyz) for value in gates]
        reach = [occupancy[0].to(self.base_xyz)]
        active_gate_masks = []
        for level, parent in enumerate(self.parents):
            child_has_members = occupancy[level + 1]
            if self.config.cache_static_layout:
                mask = self._cached_active_gates[level]
            else:
                mask = torch.zeros(self.counts[level], device=self.base_xyz.device, dtype=torch.int32)
                if len(parent):
                    mask.index_add_(0, parent, child_has_members.to(torch.int32))
                mask = mask > 0
            active_gate_masks.append(mask)
            node_gate = self.base_xyz.new_zeros(self.counts[level])
            node_gate = node_gate.index_copy(0, self.active_layout['refine_rows'][level], gate_prob[level])
            reach.append(reach[level][parent] * node_gate[parent] * child_has_members.to(self.base_xyz))

        weights = self._precision_weights(step, differentiable=(not hard and phase == 'gate_compress'))
        expected_presence = self.base_xyz.new_zeros(())
        expected_payload = self.base_xyz.new_zeros(())
        expected_gates = self.base_xyz.new_zeros(())
        for level, (translation, rotation) in enumerate(zip(self.region_translation,
                                                             self.region_rotation)):
            expected_presence = expected_presence + reach[level].sum()
            if level < len(active_gate_masks):
                expected_gates = expected_gates + (reach[level] * active_gate_masks[level].to(reach[level])).sum()
            raw = torch.cat((translation, rotation), dim=-1)
            support, _ = self._support_by_precision(raw)
            costs = support * self.base_xyz.new_tensor(
                [self.selector_bits + 6 * bits for bits in self.config.precision_bits])[None, :]
            expected_payload = expected_payload + (reach[level][:, None] * weights[level] * costs).sum()

        soft_bits = expected_presence + expected_gates + expected_payload
        if hard:
            hard_bits = self._hard_rate_bits(step)
            padding = self.base_xyz.new_tensor(float((-hard_bits) % 8))
        else:
            padding = self.base_xyz.new_zeros(())
        result = dict(codec_header_bytes=self.base_xyz.new_tensor(float(CODEC_HEADER_BYTES)),
                    payload_padding_bytes=padding / 8,
                    stop_refine_flag_bytes=expected_gates / 8,
                    motion_presence_flag_bytes=expected_presence / 8,
                    precision_and_symbol_bytes=expected_payload / 8)
        if frozen:
            self._frozen_storage_parts[hard] = {key: value.detach() for key, value in result.items()}
            return dict(self._frozen_storage_parts[hard])
        return result

    def _write_node(self, writer, level, node, occupancy, children, gates,
                    precision, symbols):
        if not bool(occupancy[level][node]):
            return
        symbol = symbols[level][node]
        present = any(value != 0 for value in symbol)
        writer.write(int(present))
        if present:
            precision_index = int(precision[level][node])
            bits = self.config.precision_bits[precision_index]
            writer.write(precision_index, self.selector_bits)
            for value in symbol:
                writer.write(_signed_code(value, bits), bits)
        if level < len(self.refine_logits):
            child_nodes = [child for child in children[level][node]
                           if bool(occupancy[level + 1][child])]
            if child_nodes:
                row = self._cached_gate_positions[level][node]
                opened = bool(gates[level][row])
                writer.write(int(opened))
                if opened:
                    for child in child_nodes:
                        self._write_node(writer, level + 1, child, occupancy,
                                         children, gates, precision, symbols)

    def _encode_payload(self, step):
        writer = _BitWriter()
        occupancy = self._occupancy_values()
        children = self._children_values()
        gates = self.refinement_masks(step)
        precision, symbols, _ = self._selected_symbols(step)
        # One transfer per tensor, rather than thousands of scalar CUDA reads
        # while traversing nodes. The serialized bit order is unchanged.
        cpu_occupancy = [value.detach().cpu().tolist() for value in occupancy]
        cpu_gates = [value.detach().cpu().tolist() for value in gates]
        cpu_precision = [value.detach().cpu().tolist() for value in precision]
        cpu_symbols = [value.detach().cpu().tolist() for value in symbols]
        for root, present in enumerate(cpu_occupancy[0]):
            if present:
                self._write_node(writer, 0, root, cpu_occupancy, children,
                                 cpu_gates, cpu_precision, cpu_symbols)
        bit_count = writer.bit_count
        payload = writer.finish()
        return payload, bit_count, precision, symbols

    def export_motion(self, path, *, frame_index, previous_state_digest,
                      post_motion_digest, previous_count, deleted_count, birth_count):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload, bit_count, precision, symbols = self._encode_payload(self.config.motion_steps)
        header = CODEC_HEADER.pack(CODEC_MAGIC, 1, 0, CODEC_HEADER_BYTES,
            int(frame_index), int(previous_count), len(self.base_xyz), int(deleted_count),
            int(birth_count), int(bit_count), float(self.extent), bytes.fromhex(previous_state_digest),
            bytes.fromhex(post_motion_digest), layout_digest(self.active_layout))
        path.write_bytes(header + payload)
        if len(path.read_bytes()) != CODEC_HEADER_BYTES + math.ceil(bit_count / 8):
            raise AssertionError('Bitstream size disagrees with exact encoded length')
        saturation = sum(int(value.sum()) for value in self._selected_symbols(self.config.motion_steps)[2])
        return dict(path=str(path), archive_bytes=path.stat().st_size,
            parameter_bytes=path.stat().st_size, payload_bits=bit_count,
            payload_padding_bits=(-bit_count) % 8, codec_header_bytes=CODEC_HEADER_BYTES,
            frame=frame_index, layout_encoding='binary_preorder_bitstream_v1',
            precision_selector_bits=self.selector_bits,
            precision_bits=list(self.config.precision_bits),
            saturated_motion_nodes=saturation,
            semantics='One preorder presence bit per reached node; a refinement bit only for '
                'reached internal nodes with surviving children; nonzero nodes carry a fixed '
                'precision selector followed by six signed symbols. No GS ownership or node IDs are stored.')

    @staticmethod
    def read_header(path):
        packet = Path(path).read_bytes()
        if len(packet) < CODEC_HEADER_BYTES:
            raise ValueError('Truncated storage-aware binary header')
        values = CODEC_HEADER.unpack(packet[:CODEC_HEADER_BYTES])
        (magic, version, flags, header_size, frame, previous_count, base_count,
         deleted_count, birth_count, bit_count, extent, previous_digest,
         base_digest, topology) = values
        if magic != CODEC_MAGIC or version != 1 or flags != 0 or header_size != CODEC_HEADER_BYTES:
            raise ValueError('Unsupported storage-aware binary stream')
        if not math.isfinite(extent) or extent <= 0:
            raise ValueError('Invalid scene extent in storage-aware binary stream')
        expected_size = CODEC_HEADER_BYTES + math.ceil(bit_count / 8)
        if len(packet) != expected_size:
            raise ValueError('Binary stream size does not match its encoded bit count')
        return dict(frame=frame, previous_count=previous_count, base_count=base_count,
            deleted_count=deleted_count, birth_count=birth_count, payload_bits=bit_count,
            extent=extent,
            previous_digest=previous_digest.hex(), post_motion_digest=base_digest.hex(),
            topology_digest=topology.hex(), payload=packet[CODEC_HEADER_BYTES:])

    @classmethod
    def decode_motion(cls, path, base_xyz, base_rotation, config, layout, *, coding_state=None):
        header = cls.read_header(path)
        if len(base_xyz) != header['base_count']:
            raise ValueError('Post-deletion base count disagrees with stream header')
        if layout_digest(layout).hex() != header['topology_digest']:
            raise ValueError('Deterministic tree digest disagrees with motion stream')
        bits = list(config.precision_bits)
        selector_bits = math.ceil(math.log2(len(bits)))
        reader = _BitReader(header['payload'], header['payload_bits'])
        occupancy = _occupancy(layout)
        children = _children(layout)
        counts = layout['counts']
        device = base_xyz.device
        regions = [base_xyz.new_zeros(count, 6) for count in counts]
        if coding_state is not None:
            precision_choices = [torch.full((count,), len(bits) - 1, device=device,
                                             dtype=torch.long) for count in counts]
            coded_symbols = [torch.zeros(count, 6, device=device, dtype=torch.long) for count in counts]
        gates = [base_xyz.new_zeros(len(rows), dtype=torch.bool)
                 for rows in layout['refine_rows']]
        gate_positions = [{int(node): i for i, node in enumerate(rows.tolist())}
                          for rows in layout['refine_rows']]
        steps = torch.cat((base_xyz.new_tensor(config.translation_steps_extent)[:, None].expand(-1, 3)
                           * float(header['extent']),
                           base_xyz.new_tensor(config.rotation_steps_rad)[:, None].expand(-1, 3)), dim=1)

        def visit(level, node):
            if not bool(occupancy[level][node]):
                return
            present = reader.read()
            if present:
                precision = reader.read(selector_bits)
                if precision >= len(bits):
                    raise ValueError('Invalid precision selector')
                width = bits[precision]
                values = [_signed_value(reader.read(width), width) for _ in range(6)]
                if not any(values):
                    raise ValueError('A zero motion must omit its precision payload')
                regions[level][node] = base_xyz.new_tensor(values) * steps[precision]
                if coding_state is not None:
                    precision_choices[level][node] = precision
                    coded_symbols[level][node] = torch.tensor(values, device=device, dtype=torch.long)
            if level < len(gates):
                child_nodes = [child for child in children[level][node]
                               if bool(occupancy[level + 1][child])]
                if child_nodes:
                    opened = reader.read()
                    position = gate_positions[level].get(node)
                    if position is None:
                        raise ValueError('Stream refines a node without a binary gate')
                    gates[level][position] = bool(opened)
                    if opened:
                        for child in child_nodes:
                            visit(level + 1, child)

        for root in torch.where(occupancy[0])[0].tolist():
            visit(0, root)
        if reader.cursor != header['payload_bits']:
            raise ValueError('Binary motion has unread payload bits')
        paths = [base_xyz.new_ones(counts[0])]
        for level, parent in enumerate(layout['parents']):
            node_gate = base_xyz.new_zeros(counts[level])
            node_gate = node_gate.index_copy(0, layout['refine_rows'][level], gates[level].to(base_xyz))
            paths.append(paths[level][parent] * node_gate[parent])
        transform = (node_conditional_transform if config.motion_compute == 'node'
                     else _conditional_transform)
        transform_options = dict(selected_only=True) if config.motion_compute == 'node' else {}
        xyz, rotation = transform(base_xyz, base_rotation, layout, regions, paths, **transform_options)
        if coding_state is not None:
            coding_state.update(precision=precision_choices, symbols=coded_symbols)
        return xyz, rotation, gates, regions, header

    @torch.no_grad()
    def register_materialized_layout(self, materialized_xyz, birth_sources):
        """Route full-PLY newborns by their decoded positions, matching the decoder.

        birth_sources are the attribute predictors selected before quantization.
        Quantization can move a newborn across their nearest-neighbor boundary.
        Those predictors are not transmitted and must not determine cached ownership.
        """
        source = torch.as_tensor(birth_sources, device=self.base_xyz.device, dtype=torch.long)
        old_count = len(self.base_xyz)
        if (source.ndim != 1 or len(materialized_xyz) != old_count + len(source) or
                bool(((source < 0) | (source >= old_count)).any())):
            raise ValueError('Materialized newborn count does not match source rows')
        computed = _nearest_source_rows(materialized_xyz[:old_count], materialized_xyz[old_count:])
        full_index = torch.cat((self.index, self.index[:, computed]), dim=1)
        self.materialized_layout = _copy_layout(_binary_layout(
            full_index, self.counts, self.parents, materialized_xyz,
            self.active_layout['pivots']))
        return self.materialized_layout

    def compensation_deletions(self, opacity, previous_new_count):
        if not self.committed:
            raise ValueError('Storage-aware deletion selection needs the committed prior frame')
        if len(opacity) != len(self.base_xyz) + previous_new_count:
            raise ValueError('Storage-aware deletion count disagrees with the previous tree')
        if previous_new_count == 0:
            return torch.empty(0, dtype=torch.long, device=opacity.device), dict(
                enabled=True, protected_keep_old_count=0, eligible_count=len(opacity),
                requested_count=0, deleted_count=0,
                semantics='No row IDs are transmitted; the next decoder repeats this deterministic policy.')
        xyz, rotation = self(self.config.motion_steps)
        changed = ((xyz != self.base_xyz).any(-1) |
                   (rotation != self.base_rotation).any(-1))
        previous_changed = changed
        total = len(opacity)
        old_count = len(self.base_xyz)
        protected = torch.zeros(total, device=opacity.device, dtype=torch.bool)
        protected[:old_count] = ~previous_changed.to(protected.device)
        candidates = torch.where(~protected)[0]
        deleted = candidates[torch.argsort(opacity.detach().flatten()[candidates], stable=True)[:previous_new_count]]
        return deleted, dict(enabled=True, protected_keep_old_count=int(protected.sum()),
            eligible_count=len(candidates), requested_count=previous_new_count,
            deleted_count=len(deleted), deleted_keep_count=0,
            semantics='Decoder recomputes the previous exact changed-pose mask, then removes the '
                'lowest-opacity rows with stable row-order ties. Deletion identities are not transmitted.')

    def statistics(self):
        step = max(self.last_step, 1)
        gates = self.refinement_masks(step)
        paths = self._paths(gates)
        occupancy = _occupancy(self.active_layout)
        precision, symbols, saturation = self._selected_symbols(step)
        levels = []
        terminals = torch.zeros(len(self.base_xyz), device=self.base_xyz.device, dtype=torch.long)
        for level, count in enumerate(self.counts):
            reached = (paths[level] > 0.5) & occupancy[level]
            nonzero = symbols[level].ne(0).any(-1)
            histogram = {str(bits): int(((precision[level] == i) & reached & nonzero).sum())
                         for i, bits in enumerate(self.config.precision_bits)}
            entry = dict(level=level, nodes=count, reachable_nodes=int(reached.sum()),
                         nonzero_motion_records=int((reached & nonzero).sum()),
                         precision_bit_histogram=histogram,
                         saturated_motion_nodes=int(saturation[level].sum()))
            if level < len(gates):
                rows = self.active_layout['refine_rows'][level]
                visited = reached[rows]
                entry.update(internal_nodes=len(rows), reachable_refine_gates=int(visited.sum()),
                    refine_open=int((visited & gates[level]).sum()),
                    refine_closed=int((visited & ~gates[level]).sum()))
            levels.append(entry)
        terminal_depth = torch.zeros(len(self.base_xyz), device=self.base_xyz.device, dtype=torch.long)
        for level in range(len(self.counts)):
            owner = self.index[level]
            valid = owner >= 0
            reached_rows = (valid & (paths[level][owner.clamp_min(0)] > 0.5)
                            if self.counts[level] else torch.zeros_like(valid))
            terminal_depth[reached_rows] = level
        deepest = terminal_depth
        depth_counts = torch.bincount(deepest, minlength=len(self.counts)).tolist()
        changed = self._hard_masks_for_step(step)['m']
        hard_bits = self._hard_rate_bits(step)
        return dict(region_counts=self.counts, levels=levels,
            gs=dict(denominator=len(self.base_xyz), changed=int(changed.sum()),
                    unchanged=len(self.base_xyz) - int(changed.sum()),
                    terminal_depth_counts=depth_counts),
            precision_bits=list(self.config.precision_bits),
            hard_coded_payload_bits=hard_bits,
            tree_rebuilt=self.tree_rebuilt, tree_setup=self.tree_setup)

    def gs_activity_fraction(self, step, hard=False):
        return self._hard_masks_for_step(step)['m'].to(self.base_xyz).mean()

    def capture(self):
        packet = super().capture()
        if packet is not None and self.hard_masks is not None:
            packet['hard_masks'] = dict(
                refine=[value.detach().cpu().clone() for value in self.hard_masks['refine']],
                precision=[value.detach().cpu().clone() for value in self.hard_masks['precision']])
        return packet

    def restore(self, packet):
        super().restore(packet)
        if packet and packet.get('hard_masks') is not None:
            hard = packet['hard_masks']
            if 'precision' not in hard:
                raise ValueError('Storage-aware checkpoint lacks hard precision choices')
            self.hard_masks['precision'] = [value.to(self.base_xyz.device)
                                            for value in hard['precision']]


class StorageAwareBinaryDecoder:
    """Sequential decoder that mirrors deletion, cached trees and newborn routing."""

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
        header = StorageAwareBinaryMotion.read_header(motion_path)
        frame = header['frame']
        if frame != self.last_frame + 1:
            raise ValueError(f'Expected frame {self.last_frame + 1}, received {frame}')
        if len(raw['xyz']) != header['previous_count']:
            raise ValueError('Previous Gaussian count disagrees with motion stream')
        if reference_digest(raw) != header['previous_digest']:
            raise ValueError('Previous complete Gaussian state digest mismatch')
        if header['deleted_count'] != self.previous_birth_count:
            raise ValueError('Deletion count does not match the previous frame birth count')
        deleted = _delete_rows(raw, self.previous_birth_count, self.previous_changed)
        rows = survivor_rows(len(raw['xyz']), deleted)
        base = {name: raw[name][rows].clone() for name in RAW_FIELDS}
        if len(base['xyz']) != header['base_count']:
            raise ValueError('Deterministic deletion result disagrees with stream base count')
        rebuild = (frame - 1) % self.config.tree_rebuild_interval == 0
        if rebuild:
            layout = stable_binary_forest(base['xyz'], self.config.coarse_region_gs,
                self.config.min_child_gs, self.config.max_motion_levels)
        else:
            if self.materialized_layout is None:
                raise ValueError('Intermediate frame has no cached deterministic tree')
            old = self.materialized_layout
            layout = _binary_layout(old['index'].to(base['xyz'].device)[:, rows],
                old['counts'], old['parents'], base['xyz'], old['pivots'])
        if layout_digest(layout).hex() != header['topology_digest']:
            raise ValueError('Reconstructed cached tree digest mismatch')
        coding_state = {} if self.retain_frame_state else None
        xyz, rotation, gates, regions, decoded_header = StorageAwareBinaryMotion.decode_motion(
            motion_path, base['xyz'], base['rotation'], self.config, layout, coding_state=coding_state)
        if self.retain_frame_state:
            self.last_frame_state = dict(base_xyz=base['xyz'], base_rotation=base['rotation'],
                layout=layout, previous_layout=self.materialized_layout, gates=gates, regions=regions,
                coding=coding_state, header=decoded_header)
        changed = ((xyz != base['xyz']).any(-1) |
                   (rotation != base['rotation']).any(-1)).detach()
        base['xyz'], base['rotation'] = xyz, rotation
        if reference_digest(base) != header['post_motion_digest']:
            raise ValueError('Decoded post-motion Gaussian state digest mismatch')

        births = _load_birth_ply(birth_path, base['xyz'].device) if birth_path else {
            name: base[name].new_empty((0,) + base[name].shape[1:]) for name in RAW_FIELDS}
        if len(births['xyz']) != header['birth_count']:
            raise ValueError('Birth PLY count disagrees with motion stream')
        if len(births['xyz']):
            sources = _nearest_source_rows(base['xyz'], births['xyz'])
            full_index = torch.cat((layout['index'], layout['index'][:, sources]), dim=1)
            materialized_xyz = torch.cat((base['xyz'], births['xyz']), dim=0)
        else:
            sources = torch.empty(0, device=base['xyz'].device, dtype=torch.long)
            full_index = layout['index']
            materialized_xyz = base['xyz']
        for name in RAW_FIELDS:
            base[name] = torch.cat((base[name], births[name].to(base[name])), dim=0)
        self.materialized_layout = _copy_layout(_binary_layout(
            full_index, layout['counts'], layout['parents'], materialized_xyz,
            layout['pivots']))
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


def decoder_for_binary_stream(path, config, *, retain_frame_state=False):
    """Dispatch a sequential decoder from the stream magic without ambiguity."""
    magic = Path(path).read_bytes()[:4]
    if magic == CODEC_MAGIC:
        if not isinstance(config, StorageAwareBinaryConfig):
            raise TypeError('SARB stream requires StorageAwareBinaryConfig')
        return StorageAwareBinaryDecoder(config, retain_frame_state=retain_frame_state)
    # Lazy import avoids a module cycle: common_contrast reuses the bit IO.
    from .common_contrast import (CODEC_MAGIC as COMMON_CONTRAST_MAGIC,
                                  CommonContrastConfig, CommonContrastDecoder)
    if magic == COMMON_CONTRAST_MAGIC:
        if not isinstance(config, CommonContrastConfig):
            raise TypeError('CCRB stream requires CommonContrastConfig')
        return CommonContrastDecoder(config, retain_frame_state=retain_frame_state)
    raise ValueError('Unknown binary motion codec identifier')


def _load_birth_ply(path, device):
    if path is None:
        return {}
    from plyfile import PlyData
    vertex = PlyData.read(str(path))['vertex']
    names = vertex.data.dtype.names

    def fields(prefix):
        ordered = sorted((name for name in names if name.startswith(prefix)),
                         key=lambda name: int(name.rsplit('_', 1)[1]))
        return np.stack([vertex[name] for name in ordered], axis=1).astype(np.float32)

    arrays = dict(
        xyz=np.stack([vertex[name] for name in ('x', 'y', 'z')], axis=1).astype(np.float32),
        rotation=fields('rot_'), logscale=fields('scale_'),
        opacity=np.asarray(vertex['opacity'], dtype=np.float32)[:, None],
        sh_dc=fields('f_dc_').reshape(-1, 3, 1).transpose(0, 2, 1),
        sh_rest=fields('f_rest_').reshape(len(vertex), 3, -1).transpose(0, 2, 1))
    return {name: torch.from_numpy(value.copy()).to(device) for name, value in arrays.items()}
