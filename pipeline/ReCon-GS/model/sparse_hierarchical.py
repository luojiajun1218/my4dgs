"""Independent binary motion gates for hierarchical residual motion.

This mode is intentionally separate from the legacy KEEP/SHARE/SPLIT model.
Region gates are independent at every depth; a child can move when its parent
region gate is closed. A final per-GS gate restores the exact previous pose.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import math
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libgs.utils.general import quaternion_multiply

from .gated_residual import _layout, _layout_cpu, _occupancy
from .region_rigid import apply_region_rigid, rotation_vector_quaternion
from .residual_motion import (QuantizedResidualConfig, integer_symbols,
                              quantized_symbols, stable_layout,
                              inverse_unit_quaternion, shortest_rotation_vector)
from .sparse_layout_codec import decode_layout, encode_layout, file_digest, numpy_layout


LAYOUT_VERSION = 'independent_binary_hierarchical_motion_v1'


@dataclass
class SparseHierarchicalConfig(QuantizedResidualConfig):
    """Shared quantization/birth settings plus explicit three-stage budgets.

    ``motion_steps`` remains the public boundary consumed by the trainer and
    the existing incremental-GS path. It must equal the sum of the three
    independently configured motion stages.
    """

    dense_fit_steps: Optional[int] = None
    gate_compress_steps: Optional[int] = None
    initial_gate_logit: Optional[float] = None
    m_initial_gate_logit: Optional[float] = None
    m_fine_sharing: bool = False
    m_fine_gate_lr: Optional[float] = None
    m_hard_forward: bool = False
    gate_hard_threshold: Optional[float] = None
    refine_lr_scale: float = 1.0
    rebase_on_hardening: bool = False

    def validate(self):
        # Reuse validation for the unchanged quantization and birth interface.
        # The legacy three-state gates and their storage selector are not part
        # of this model, so validate those inherited fields separately below.
        base_names = {value.name for value in fields(QuantizedResidualConfig)}
        values = {name: getattr(self, name) for name in base_names}
        values.update(enable_learned_gates=False, storage_loss_weight=0.0,
                      gs_activity_loss_weight=0.0, tree_rebuild_interval=0,
                      motion_gradient_mode='all_split', motion_gradient_warmup_steps=0,
                      fit_fine_split_candidates=False, reset_keep_gate_priors=False)
        QuantizedResidualConfig(**values).validate()

        for name in ('dense_fit_steps', 'gate_compress_steps', 'gate_refine_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.motion_steps != (self.dense_fit_steps + self.gate_compress_steps +
                                 self.gate_refine_steps):
            raise ValueError('motion_steps must equal dense_fit_steps + gate_compress_steps + gate_refine_steps')
        if not isinstance(self.rebase_on_hardening, bool):
            raise ValueError('rebase_on_hardening must be bool')
        if not isinstance(self.m_fine_sharing, bool):
            raise ValueError('m_fine_sharing must be bool')
        if not isinstance(self.m_hard_forward, bool):
            raise ValueError('m_hard_forward must be bool')
        if self.m_fine_sharing and (
                self.m_fine_gate_lr is None or isinstance(self.m_fine_gate_lr, bool) or
                not isinstance(self.m_fine_gate_lr, (int, float)) or
                not math.isfinite(self.m_fine_gate_lr) or self.m_fine_gate_lr <= 0):
            raise ValueError('Shared fine m gates require a finite positive m_fine_gate_lr')
        if (isinstance(self.refine_lr_scale, bool) or
                not isinstance(self.refine_lr_scale, (int, float)) or
                not math.isfinite(self.refine_lr_scale) or self.refine_lr_scale <= 0):
            raise ValueError('refine_lr_scale must be finite and positive')
        if isinstance(self.tree_rebuild_interval, bool) or not isinstance(self.tree_rebuild_interval, int) or self.tree_rebuild_interval < 0:
            raise ValueError('tree_rebuild_interval must be a nonnegative integer')
        if (self.initial_gate_logit is None or isinstance(self.initial_gate_logit, bool) or
                not isinstance(self.initial_gate_logit, (int, float)) or
                not math.isfinite(self.initial_gate_logit)):
            raise ValueError('initial_gate_logit must be finite')
        if self.m_initial_gate_logit is not None and (
                isinstance(self.m_initial_gate_logit, bool) or
                not isinstance(self.m_initial_gate_logit, (int, float)) or
                not math.isfinite(self.m_initial_gate_logit)):
            raise ValueError('m_initial_gate_logit must be finite or None')
        if (self.gate_hard_threshold is None or isinstance(self.gate_hard_threshold, bool) or
                not isinstance(self.gate_hard_threshold, (int, float)) or
                not math.isfinite(self.gate_hard_threshold) or
                not 0 < self.gate_hard_threshold < 1):
            raise ValueError('gate_hard_threshold must be finite and in (0, 1)')
        if (self.storage_loss_weight is None or isinstance(self.storage_loss_weight, bool) or
                not isinstance(self.storage_loss_weight, (int, float)) or
                not math.isfinite(self.storage_loss_weight) or self.storage_loss_weight < 0):
            raise ValueError('storage_loss_weight must be finite and nonnegative')
        if self.gs_activity_loss_weight != 0:
            raise ValueError('sparse_hierarchical v1 does not use gs_activity_loss_weight')
        if (self.gate_lr is None or isinstance(self.gate_lr, bool) or
                not isinstance(self.gate_lr, (int, float)) or
                not math.isfinite(self.gate_lr) or self.gate_lr <= 0):
            raise ValueError('gate_lr must be finite and positive')
        if (self.gate_temperature_start is None or isinstance(self.gate_temperature_start, bool) or
                not isinstance(self.gate_temperature_start, (int, float)) or
                not math.isfinite(self.gate_temperature_start) or self.gate_temperature_start <= 0):
            raise ValueError('gate_temperature_start must be finite and positive')
        if (self.gate_temperature_end is None or isinstance(self.gate_temperature_end, bool) or
                not isinstance(self.gate_temperature_end, (int, float)) or
                not math.isfinite(self.gate_temperature_end) or self.gate_temperature_end <= 0):
            raise ValueError('gate_temperature_end must be finite and positive')
        if self.gate_policy != 'learned':
            raise ValueError('sparse_hierarchical requires gate_policy: learned')
        if not self.enable_learned_gates:
            raise ValueError('sparse_hierarchical requires enable_learned_gates: true')
        if (self.motion_gradient_mode != 'all_split' or self.motion_gradient_warmup_steps != 0 or
                self.fit_fine_split_candidates or self.reset_keep_gate_priors or
                self.gs_activity_loss_weight != 0):
            raise ValueError('Legacy candidate, gradient-probe and activity options are disabled in sparse_hierarchical')

    def keep_candidate_fit_due(self, step):
        return False


def _identity_quaternion(reference):
    value = reference.new_zeros((len(reference), 4))
    value[:, 0] = 1
    return value


def _quantized_motion(motion, integer=False):
    quantize = integer_symbols if integer else quantized_symbols
    steps = motion.motion_steps_vector
    regions = [quantize(torch.cat((translation, rotation), -1), steps)
               for translation, rotation in zip(motion.region_translation,
                                                motion.region_rotation)]
    local = quantize(torch.cat((motion.gs_translation, motion.gs_rotation), -1), steps)
    if integer:
        return regions, local
    return [value * steps for value in regions], local * steps


def _transform(base_xyz, base_rotation, layout, regions, local, z_regions,
               z_local, m_mask, *, soft_m=False):
    """Apply independent region residuals, then the final per-GS pose gate."""
    index, pivots = layout['index'], layout['pivots']
    n = len(base_xyz)
    xyz = base_xyz
    rotation = base_rotation
    identity = _identity_quaternion(base_xyz)
    advected_pivots = [pivots[level][index[level]] for level in range(3)]

    for level in range(3):
        owner = index[level]
        weight = z_regions[level][owner]
        transform = regions[level][owner]
        translation = transform[:, :3] * weight[:, None]
        rotation_vector = transform[:, 3:] * weight[:, None]
        old_xyz, old_rotation = xyz, rotation
        xyz, rotation = apply_region_rigid(
            xyz, rotation, advected_pivots[level], translation, rotation_vector)
        closed = weight == 0
        xyz = torch.where(closed[:, None], old_xyz, xyz)
        rotation = torch.where(closed[:, None], old_rotation, rotation)

        # A parent rotation carries every child pivot with it, even when the
        # child has its own independent gate.
        for child_level in range(level + 1, 3):
            old_pivot = advected_pivots[child_level]
            moved_pivot, _ = apply_region_rigid(
                old_pivot, identity, advected_pivots[level], translation,
                rotation_vector)
            advected_pivots[child_level] = torch.where(
                closed[:, None], old_pivot, moved_pivot)

    local_weight = z_local
    local_xyz = xyz + local[:, :3] * local_weight[:, None]
    local_rotation = quaternion_multiply(
        rotation_vector_quaternion(local[:, 3:] * local_weight[:, None]), rotation)
    xyz = torch.where((local_weight == 0)[:, None], xyz, local_xyz)
    rotation = torch.where((local_weight == 0)[:, None], rotation, local_rotation)

    if soft_m:
        gate = m_mask.clamp(0, 1)[:, None]
        xyz = base_xyz + gate * (xyz - base_xyz)
        base_q = F.normalize(base_rotation, dim=-1)
        candidate_q = F.normalize(rotation, dim=-1)
        candidate_q = torch.where((candidate_q * base_q).sum(-1, keepdim=True) < 0,
                                  -candidate_q, candidate_q)
        rotation = F.normalize((1 - gate) * base_q + gate * candidate_q, dim=-1)
    else:
        open_mask = m_mask.bool()
        xyz = torch.where(open_mask[:, None], xyz, base_xyz)
        rotation = torch.where(open_mask[:, None], rotation, base_rotation)
    return xyz, rotation


def _pack_bits(bits):
    """Pack bits most-significant first; one stream has one final byte pad."""
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    packed = np.zeros((len(values) + 7) // 8, dtype=np.uint8)
    for offset, value in enumerate(values):
        packed[offset // 8] |= int(value & 1) << (7 - offset % 8)
    return packed, int(len(values))


def _unpack_bit(data, cursor):
    if cursor >= len(data) * 8:
        raise ValueError('Sparse gate bitstream ended early')
    return (int(data[cursor // 8]) >> (7 - cursor % 8)) & 1, cursor + 1


def _children(layout):
    result = []
    occupancy = _occupancy(layout)
    for level in range(2):
        rows = [[] for _ in range(layout['counts'][level])]
        for child, parent in enumerate(layout['parents'][level].detach().cpu().tolist()):
            if bool(occupancy[level + 1][child]):
                rows[parent].append(child)
        result.append(rows)
    return result


def _members_by_node(layout):
    """Build sorted point-row lists once instead of rescanning all GS per node."""
    owners = layout['index'].detach().cpu()
    groups = []
    for level, count in enumerate(layout['counts']):
        owner = owners[level]
        order = torch.argsort(owner, stable=True)
        sizes = torch.bincount(owner, minlength=count).tolist()
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        groups.append([order[offsets[node]:offsets[node + 1]]
                       for node in range(count)])
    return groups


def encode_m_tree(mask, layout):
    """Encode m with 2-bit uniform-off/on/mixed tags and mixed leaf bitmaps."""
    mask = torch.as_tensor(mask, dtype=torch.bool).detach().cpu()
    if mask.ndim != 1 or len(mask) != layout['index'].shape[1]:
        raise ValueError('m mask must contain one binary decision per GS')
    index = layout['index'].detach().cpu()
    occupied = [value.detach().cpu() for value in _occupancy(layout)]
    children = _children(layout)
    members_by_node = _members_by_node(layout)
    bits = []

    def visit(level, node):
        members = members_by_node[level][node]
        values = mask[members]
        if bool(values.all()):
            bits.extend((0, 1))  # uniform on
        elif not bool(values.any()):
            bits.extend((0, 0))  # uniform off
        else:
            bits.extend((1, 0))  # mixed
            if level == 2:
                bits.extend(values.to(torch.uint8).tolist())
            else:
                for child in children[level][node]:
                    visit(level + 1, child)

    for root in torch.where(occupied[0])[0].tolist():
        visit(0, root)
    return _pack_bits(bits)


def decode_m_tree(data, bit_count, layout):
    data = np.asarray(data, dtype=np.uint8).reshape(-1)
    if bit_count < 0 or bit_count > len(data) * 8 or len(data) != (bit_count + 7) // 8:
        raise ValueError('Invalid m mask bit count')
    index = layout['index'].detach().cpu()
    occupied = [value.detach().cpu() for value in _occupancy(layout)]
    children = _children(layout)
    members_by_node = _members_by_node(layout)
    result = torch.zeros(index.shape[1], dtype=torch.bool)
    cursor = 0

    def visit(level, node):
        nonlocal cursor
        first, cursor = _unpack_bit(data, cursor)
        second, cursor = _unpack_bit(data, cursor)
        members = members_by_node[level][node]
        if (first, second) == (0, 0):
            return
        if (first, second) == (0, 1):
            result[members] = True
            return
        if (first, second) != (1, 0):
            raise ValueError('Unsupported m mask node tag')
        if level == 2:
            for member in members.tolist():
                value, cursor = _unpack_bit(data, cursor)
                result[member] = bool(value)
            return
        for child in children[level][node]:
            visit(level + 1, child)

    for root in torch.where(occupied[0])[0].tolist():
        visit(0, root)
    if cursor != bit_count:
        raise ValueError('m mask bit count does not match decoded tree')
    return result


def _expected_m_tree_bits(probability, layout):
    """Exact expected tag/bitmap length for independent Bernoulli m gates."""
    eps = torch.finfo(probability.dtype).eps
    p_on = probability.clamp(eps, 1 - eps)
    index = layout['index']
    occupancy = _occupancy(layout)
    child_owner = layout['parents']

    def segment_log_sum(log_values, owner, count):
        output = log_values.new_zeros(count)
        output.index_add_(0, owner, log_values)
        return output

    # Carry log probabilities between levels. An exp result may underflow to
    # zero; taking log of that result at its parent makes backward nonfinite.
    log_all_on = segment_log_sum(p_on.log(), index[2], layout['counts'][2])
    log_all_off = segment_log_sum(torch.log1p(-p_on), index[2], layout['counts'][2])
    all_on, all_off = log_all_on.exp(), log_all_off.exp()
    fine_sizes = torch.bincount(index[2], minlength=layout['counts'][2]).to(probability)
    expected = 2 + (1 - all_on - all_off).clamp_min(0) * fine_sizes
    expected = expected * occupancy[2].to(probability)
    for level in (1, 0):
        parent = child_owner[level]
        log_all_on = segment_log_sum(log_all_on[occupancy[level + 1]],
                                    parent[occupancy[level + 1]], layout['counts'][level])
        log_all_off = segment_log_sum(log_all_off[occupancy[level + 1]],
                                     parent[occupancy[level + 1]], layout['counts'][level])
        all_on, all_off = log_all_on.exp(), log_all_off.exp()
        child_bits = probability.new_zeros(layout['counts'][level])
        child_bits.index_add_(0, parent[occupancy[level + 1]],
                              expected[occupancy[level + 1]])
        child_count = torch.bincount(parent[occupancy[level + 1]],
                                     minlength=layout['counts'][level]).to(probability)
        expected = 2 + child_bits - 2 * child_count * (all_on + all_off)
        expected = expected.clamp_min(2) * occupancy[level].to(probability)
    return expected[occupancy[0]].sum()


class SparseHierarchicalMotion(nn.Module):
    """Three-stage, independent binary m/z hierarchical motion model."""

    layout_version = LAYOUT_VERSION

    def __init__(self, config: SparseHierarchicalConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.index = None
        self.active_layout = None
        self.previous_layout = None
        self.materialized_layout = None
        self.committed = False
        self.optimizer = None
        self.motion_optimizer = None
        self.gate_optimizer = None
        self.last_step = 0
        self.current_phase = None
        self.hard_masks = None
        self.hard_storage_cache = None
        self.last_metrics = {}
        self.stage_seconds = {}
        self.tree_setup = None
        # Disk-only state. A restored checkpoint starts a self-contained packet
        # in its new output directory, without referring to historical outputs.
        self._export_layout_record = None
        self._export_survivor_rows = None

    @property
    def motion_steps_vector(self):
        return self.base_xyz.new_tensor(
            [self.config.motion_xyz_step_extent * self.extent] * 3 +
            [self.config.motion_rotation_step_rad] * 3)

    def _make_optimizers(self):
        opts = dict(betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)
        self.motion_optimizer = torch.optim.Adam([
            dict(params=self.region_translation, lr=self.config.region_translation_lr),
            dict(params=self.region_rotation, lr=self.config.region_rotation_lr),
            dict(params=[self.gs_translation], lr=self.config.gs_translation_lr),
            dict(params=[self.gs_rotation], lr=self.config.gs_rotation_lr)], **opts)
        gates = [self.m_logits, *self.z_region_logits, self.z_local_logits]
        groups = [dict(params=gates, lr=self.config.gate_lr)]
        if self.config.m_fine_sharing:
            groups.append(dict(params=[self.m_fine_logits], lr=self.config.m_fine_gate_lr))
        self.gate_optimizer = torch.optim.Adam(groups, **opts)
        self.optimizer = self.motion_optimizer

    def setup(self, xyz, rotation, extent, warm_start=True,
              previous_rows=None, previous_new_count=None, frame_index=None,
              layout_override=None):
        if not len(xyz):
            raise ValueError('Sparse hierarchical motion needs at least one previous GS')
        if self.config.tree_rebuild_interval and (isinstance(frame_index, bool) or
                not isinstance(frame_index, int) or frame_index < 1):
            raise ValueError('Periodic sparse tree rebuild requires a positive frame_index')
        layout_start = time.perf_counter()
        old_layout = self.materialized_layout
        previous_motion = None
        if warm_start and self.config.motion_init_decay and self.index is not None:
            previous_motion = self._aligned_previous_motion(
                xyz, rotation, previous_rows, previous_new_count)
        rebuild = (old_layout is not None and self.config.tree_rebuild_interval > 0 and
                   frame_index % self.config.tree_rebuild_interval == 0)
        rows = None
        if layout_override is not None:
            layout = _layout(layout_override['index'].to(xyz.device),
                             layout_override['counts'], layout_override['parents'],
                             xyz, layout_override.get('pivots'))
        elif old_layout is not None:
            old_count = old_layout['index'].shape[1]
            if previous_rows is None:
                if old_count != len(xyz):
                    raise ValueError('Sparse topology reuse needs previous_rows after deletion or births')
                rows = torch.arange(len(xyz), device=xyz.device)
            else:
                rows = torch.as_tensor(previous_rows, device=xyz.device, dtype=torch.long)
            if rows.ndim != 1 or len(rows) != len(xyz) or (rows.numel() and
                    bool(((rows < 0) | (rows >= old_count)).any())):
                raise ValueError('Sparse previous_rows must map each survivor to its materialized row')
            if previous_new_count is not None and old_count != len(self.base_xyz) + previous_new_count:
                raise ValueError('Sparse previous layout and newborn count disagree')
            if rebuild:
                index, counts, parents = stable_layout(xyz, self.config.fine_region_gs)
                layout = _layout(index, counts, parents, xyz)
            else:
                old_index = old_layout['index'].to(xyz.device)[:, rows]
                layout = _layout(old_index, old_layout['counts'], old_layout['parents'], xyz,
                                 old_layout['pivots'])
        else:
            rows = None
            index, counts, parents = stable_layout(xyz, self.config.fine_region_gs)
            layout = _layout(index, counts, parents, xyz)
        topology_seconds = time.perf_counter() - layout_start
        self.previous_layout = _layout_cpu(old_layout)
        self._export_survivor_rows = (
            rows.detach().cpu().numpy().copy() if rows is not None and not rebuild
            and layout_override is None else None)
        self.active_layout = layout
        self.materialized_layout = _layout_cpu(layout)
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
        self.gs_translation = nn.Parameter(torch.zeros_like(xyz))
        self.gs_rotation = nn.Parameter(torch.zeros_like(xyz))
        initial = float(self.config.initial_gate_logit)
        m_initial = (initial if self.config.m_initial_gate_logit is None
                     else float(self.config.m_initial_gate_logit))
        self.m_logits = nn.Parameter(xyz.new_full((len(xyz),), m_initial))
        # Shared fine evidence reaches members without a direct image gradient;
        # the existing per-GS logit can still override its region's decision.
        self.m_fine_logits = nn.Parameter(xyz.new_zeros(self.counts[2]))
        self.z_region_logits = nn.ParameterList(
            [nn.Parameter(xyz.new_full((count,), initial)) for count in self.counts])
        self.z_local_logits = nn.Parameter(xyz.new_full((len(xyz),), initial))
        if previous_motion is not None:
            self._initialize_previous_motion(*previous_motion)
        self._make_optimizers()
        self.hard_masks = None
        self.hard_storage_cache = None
        self.current_phase = None
        self.last_step = 0
        self.committed = False
        self.last_metrics = {}
        self.stage_seconds = {'tree_topology': topology_seconds, 'tree_rebuild': 0.}
        self.tree_setup = dict(frame=frame_index,
            initialization_decay=self.config.motion_init_decay if previous_motion is not None else 0.,
            action='rebuild' if rebuild else ('initial' if old_layout is None else 'reuse'),
            old_region_counts=list(old_layout['counts']) if old_layout is not None else None,
            region_counts=list(self.counts), topology_seconds=topology_seconds,
            semantics='Stable three-level spatial tree. Survivor row identities are reused between rebuilds; '
                'periodic rebuild recomputes stable_layout from the surviving base poses. Gates and motion '
                'gates reset each frame. Optional inherited total motion follows survivor identities; '
                'newborns have zero inherited motion. A fresh optimizer is always used.')

    @torch.no_grad()
    def _aligned_previous_motion(self, xyz, rotation, previous_rows, previous_new_count):
        """Map committed world-space displacement to surviving old rows only."""
        if not self.committed:
            raise ValueError('Sparse motion warm start requires a committed previous frame')
        old_count = len(self.base_xyz)
        if previous_rows is None:
            if len(xyz) != old_count:
                raise ValueError('Sparse warm start needs survivor identities after births or deletion')
            rows = torch.arange(old_count, device=xyz.device)
            previous_new_count = 0
        else:
            rows = torch.as_tensor(previous_rows, device=xyz.device)
        if (isinstance(previous_new_count, bool) or not isinstance(previous_new_count, int) or
                previous_new_count < 0):
            raise ValueError('Sparse warm start requires a nonnegative previous_new_count')
        if (rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64) or
                len(rows) != len(xyz)):
            raise ValueError('Sparse warm start requires integer survivor identities')
        rows = rows.long()
        if (bool(((rows < 0) | (rows >= old_count + previous_new_count)).any()) or
                bool((rows[1:] <= rows[:-1]).any())):
            raise ValueError('Sparse warm start survivor identities must be ordered and unique')
        is_old = rows < old_count
        prior_xyz, prior_rotation = self(self.config.motion_steps)
        aligned_q = (F.normalize(prior_rotation[rows[is_old]], dim=-1) *
                     F.normalize(rotation[is_old], dim=-1)).sum(-1).abs()
        if (not torch.allclose(prior_xyz[rows[is_old]], xyz[is_old], rtol=1e-5, atol=1e-6) or
                not torch.allclose(aligned_q, torch.ones_like(aligned_q), rtol=1e-5, atol=1e-6)):
            raise ValueError('Sparse warm start survivor identities disagree with committed poses')
        displacement, relative_rotation = self.final_quantized_total_motion()
        translation = torch.zeros_like(xyz)
        inherited_rotation = _identity_quaternion(xyz)
        translation[is_old] = displacement[rows[is_old]]
        inherited_rotation[is_old] = relative_rotation[rows[is_old]]
        return translation, inherited_rotation

    @torch.no_grad()
    def _initialize_previous_motion(self, translation, rotation):
        """Project decayed displacement into the hierarchy without pivot drift.

        Regional rotations start at zero. Translation means are quantized at
        each depth; the local remainder preserves the inherited total pose.
        World rotation is initialized locally so spatial tree rebuilds cannot
        change the warm prediction through different rigid pivots.
        """
        target = translation.to(self.base_xyz) * self.config.motion_init_decay
        steps = self.motion_steps_vector
        remainder = target.clone()
        for level, parameter in enumerate(self.region_translation):
            owner = self.index[level]
            count = torch.bincount(owner, minlength=self.counts[level]).to(target)
            total = torch.zeros_like(parameter)
            total.index_add_(0, owner, remainder)
            parameter.copy_(torch.round(total / count.clamp_min(1)[:, None] / steps[:3]) * steps[:3])
            remainder -= parameter[owner]
        region_sum = sum(value[self.index[level]]
                         for level, value in enumerate(self.region_translation))
        self.gs_translation.copy_(torch.round(target / steps[:3]) * steps[:3] - region_sum)
        vector = shortest_rotation_vector(rotation.to(self.base_xyz)) * self.config.motion_init_decay
        self.gs_rotation.copy_(torch.round(vector / steps[3:]) * steps[3:])

    def install_materialized_layout(self, layout):
        self.materialized_layout = _layout_cpu(layout)

    def register_materialized_layout(self, materialized_xyz, birth_sources):
        if self.active_layout is None:
            raise ValueError('Materialized sparse layout requires an active tree')
        sources = torch.as_tensor(birth_sources, device=self.index.device, dtype=torch.long)
        if sources.ndim != 1 or (sources.numel() and
                bool(((sources < 0) | (sources >= len(self.base_xyz))).any())):
            raise ValueError('Sparse birth sources must be valid old-GS rows')
        full_index = torch.cat((self.index, self.index[:, sources]), dim=1)
        full_layout = _layout(full_index, self.counts, self.parents,
                              materialized_xyz, self.active_layout['pivots'])
        self.install_materialized_layout(full_layout)
        return self.materialized_layout

    def stage(self, step):
        if step <= self.config.dense_fit_steps:
            return 'dense_fit'
        if step <= self.config.dense_fit_steps + self.config.gate_compress_steps:
            return 'gate_compress'
        if step <= self.config.motion_steps:
            return 'gate_refine'
        return 'incremental'

    def _hard_gate_masks(self):
        self._check_finite_gates()
        threshold = self.config.gate_hard_threshold
        temperature = self.config.gate_temperature_end
        return dict(m=torch.sigmoid(self._m_gate_logits().detach() / temperature) >= threshold,
                    z_regions=[torch.sigmoid(value.detach() / temperature) >= threshold
                               for value in self.z_region_logits],
                    z_local=torch.sigmoid(self.z_local_logits.detach() / temperature) >= threshold)

    @torch.no_grad()
    def _check_finite_gates(self, *, gradients=False):
        values = [('m_logits', self.m_logits), ('z_local_logits', self.z_local_logits)]
        if self.config.m_fine_sharing:
            values.append(('m_fine_logits', self.m_fine_logits))
        values.extend((f'z_region_logits.{level}', value)
                      for level, value in enumerate(self.z_region_logits))
        if gradients:
            values += [(f'{name}.grad', value.grad) for name, value in list(values)
                       if value.grad is not None]
        finite = torch.stack([torch.isfinite(value).all() for _, value in values])
        if not bool(finite.all()):
            bad = [name for (name, _), valid in zip(values, finite.cpu().tolist())
                   if not valid]
            frame = (self.tree_setup or {}).get('frame')
            raise FloatingPointError(
                f'Nonfinite sparse gates at frame={frame}, step={self.last_step}: '
                f'{", ".join(bad)}. Refusing to update or harden invalid gates.')

    def _clear_gradients(self):
        for parameter in self.parameters():
            parameter.grad = None

    @torch.no_grad()
    def _rebase_hardened_motion(self):
        """Fit the retained hard representation to the final soft pose.

        Binary decisions stay fixed. Only an already-open local residual may
        absorb the difference; m=0 still returns the exact previous pose.
        """
        step = self.config.dense_fit_steps + self.config.gate_compress_steps
        m, z_regions, z_local = self.gate_probabilities(step)
        m = self._render_m_probability(m)
        regions, local = _quantized_motion(self)
        target_xyz, target_rotation = _transform(
            self.base_xyz, self.base_rotation, self.active_layout, regions, local,
            z_regions, z_local, m, soft_m=True)
        for level in range(3):
            self.region_translation[level].mul_(z_regions[level][:, None])
            self.region_rotation[level].mul_(z_regions[level][:, None])
        regions, _ = _quantized_motion(self)
        regional_xyz, regional_rotation = _transform(
            self.base_xyz, self.base_rotation, self.active_layout, regions,
            self.base_xyz.new_zeros(len(self.base_xyz), 6),
            self.hard_masks['z_regions'], torch.zeros_like(self.hard_masks['z_local']),
            torch.ones_like(self.hard_masks['m']))
        active = self.hard_masks['m'] & self.hard_masks['z_local']
        relative_rotation = quaternion_multiply(
            target_rotation, inverse_unit_quaternion(F.normalize(regional_rotation, dim=-1)))
        self.gs_translation.copy_(torch.where(
            active[:, None], target_xyz - regional_xyz, torch.zeros_like(self.gs_translation)))
        self.gs_rotation.copy_(torch.where(
            active[:, None], shortest_rotation_vector(relative_rotation),
            torch.zeros_like(self.gs_rotation)))
        self.last_metrics['hardening/rebased_points'] = int(active.sum())

    def _configure_phase(self, step, restoring=False):
        phase = self.stage(step)
        if phase == self.current_phase:
            return phase
        self._clear_gradients()
        if phase == 'gate_refine' and self.current_phase != 'gate_refine':
            if self.hard_masks is None:
                self.hard_masks = self._hard_gate_masks()
                if self.config.rebase_on_hardening and not restoring:
                    self._rebase_hardened_motion()
            if self.hard_storage_cache is None:
                self.hard_storage_cache = self._encode_hard_gates(self.hard_masks)
            # A fresh optimizer drops dense-fit momentum; closed motion rows
            # also have their rowwise state cleared by optimizer_step below.
            if not restoring:
                opts = dict(betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)
                self.motion_optimizer = torch.optim.Adam([
                    dict(params=self.region_translation,
                         lr=self.config.region_translation_lr * self.config.refine_lr_scale),
                    dict(params=self.region_rotation,
                         lr=self.config.region_rotation_lr * self.config.refine_lr_scale),
                    dict(params=[self.gs_translation],
                         lr=self.config.gs_translation_lr * self.config.refine_lr_scale),
                    dict(params=[self.gs_rotation],
                         lr=self.config.gs_rotation_lr * self.config.refine_lr_scale)], **opts)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if phase == 'dense_fit':
            for parameter in self.motion_parameters():
                parameter.requires_grad_(True)
            self.optimizer = self.motion_optimizer
        elif phase == 'gate_compress':
            for parameter in self.gate_parameters():
                parameter.requires_grad_(True)
            self.optimizer = self.gate_optimizer
        elif phase == 'gate_refine':
            for parameter in self.motion_parameters():
                parameter.requires_grad_(True)
            self.optimizer = self.motion_optimizer
        else:
            self.optimizer = self.motion_optimizer
        self.current_phase = phase
        return phase

    def motion_parameters(self):
        return [*self.region_translation, *self.region_rotation,
                self.gs_translation, self.gs_rotation]

    def gate_parameters(self):
        values = [self.m_logits, *self.z_region_logits, self.z_local_logits]
        return values + ([self.m_fine_logits] if self.config.m_fine_sharing else [])

    def _m_gate_logits(self):
        if self.config.m_fine_sharing:
            return self.m_logits + self.m_fine_logits[self.index[2]]
        return self.m_logits

    def _render_m_probability(self, probability):
        """Binary pose selection in forward, sigmoid derivative in backward."""
        if not self.config.m_hard_forward:
            return probability
        hard = (probability >= self.config.gate_hard_threshold).to(probability)
        return hard + (probability - probability.detach())

    def _temperature(self, step):
        start = self.config.dense_fit_steps + 1
        end = start + self.config.gate_compress_steps - 1
        progress = min(max((step - start) / max(end - start, 1), 0.), 1.)
        return (self.config.gate_temperature_start * (1 - progress) +
                self.config.gate_temperature_end * progress)

    def gate_probabilities(self, step, *, use_temperature=True):
        temperature = self._temperature(step) if use_temperature else 1.
        return (torch.sigmoid(self._m_gate_logits() / temperature),
                [torch.sigmoid(value / temperature) for value in self.z_region_logits],
                torch.sigmoid(self.z_local_logits / temperature))

    def forward(self, step, training=False):
        if training:
            phase = self._configure_phase(step)
            self.last_step = int(step)
        else:
            phase = self.stage(step)
        regions, local = _quantized_motion(self)
        if phase == 'dense_fit':
            m = torch.ones_like(self.m_logits)
            z_regions = [torch.ones_like(value) for value in self.z_region_logits]
            z_local = torch.ones_like(self.z_local_logits)
            return _transform(self.base_xyz, self.base_rotation, self.active_layout,
                              regions, local, z_regions, z_local, m)
        if phase == 'gate_compress':
            m, z_regions, z_local = self.gate_probabilities(step)
            m = self._render_m_probability(m)
            return _transform(self.base_xyz, self.base_rotation, self.active_layout,
                              regions, local, z_regions, z_local, m, soft_m=True)
        masks = (self.hard_masks if self.hard_masks is not None
                 else self._hard_gate_masks())
        return _transform(self.base_xyz, self.base_rotation, self.active_layout,
                          regions, local, masks['z_regions'], masks['z_local'], masks['m'])

    def optimizer_step(self):
        phase = self.stage(self.last_step)
        if phase == 'gate_compress':
            self._check_finite_gates(gradients=True)
            self.gate_optimizer.step()
            self._check_finite_gates()
            self.gate_optimizer.zero_grad(set_to_none=True)
            return
        if phase == 'gate_refine' and self.hard_masks is not None:
            m = self.hard_masks['m']
            for level, _ in enumerate(self.region_translation):
                owner = self.index[level]
                active = self.hard_masks['z_regions'][level]
                row_active = torch.zeros_like(active, dtype=torch.int32)
                row_active.index_add_(0, owner, m.to(torch.int32))
                active = active & (row_active > 0)
                for value in (self.region_translation[level], self.region_rotation[level]):
                    self._mask_optimizer_rows(value, active)
            local_active = m & self.hard_masks['z_local']
            self._mask_optimizer_rows(self.gs_translation, local_active)
            self._mask_optimizer_rows(self.gs_rotation, local_active)
        self.motion_optimizer.step()
        self.motion_optimizer.zero_grad(set_to_none=True)

    def _mask_optimizer_rows(self, parameter, active):
        if parameter.grad is not None:
            parameter.grad.mul_(active.reshape((-1,) + (1,) * (parameter.ndim - 1)))
        state = self.motion_optimizer.state.get(parameter, {})
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = state.get(name)
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value.mul_(active.reshape((-1,) + (1,) * (parameter.ndim - 1)))

    def _hard_masks_for_step(self, step):
        phase = self.stage(step)
        if phase == 'dense_fit':
            return dict(m=torch.ones_like(self.m_logits, dtype=torch.bool),
                        z_regions=[torch.ones_like(value, dtype=torch.bool)
                                   for value in self.z_region_logits],
                        z_local=torch.ones_like(self.z_local_logits, dtype=torch.bool))
        if phase in {'gate_refine', 'incremental'} and self.hard_masks is not None:
            return self.hard_masks
        return self._hard_gate_masks()

    def _region_has_open_m(self, m_mask, level):
        owners = self.index[level]
        counts = torch.zeros(self.counts[level], device=owners.device, dtype=torch.int32)
        counts.index_add_(0, owners, m_mask.to(torch.int32))
        return counts > 0

    def _encode_hard_gates(self, masks):
        m = masks['m'].detach().bool()
        m_data, m_bits = encode_m_tree(m, self.active_layout)
        active = [self._region_has_open_m(m, level) for level in range(3)]
        z_values = []
        active_rows, payload_rows = [], []
        region_slots = 0
        for level in range(3):
            rows = torch.where(active[level])[0]
            selected = rows[masks['z_regions'][level].detach()[rows]]
            active_rows.append(rows.detach().cpu())
            payload_rows.append(selected.detach().cpu())
            z_values.extend(masks['z_regions'][level].detach()[rows].cpu().to(torch.uint8).tolist())
            region_slots += int(len(selected))
        local_rows = torch.where(m & masks['z_local'].detach().bool())[0]
        z_values.extend(masks['z_local'].detach()[m].cpu().to(torch.uint8).tolist())
        z_data, z_bits = _pack_bits(z_values)
        symbol_bytes = 6 * 4  # six quantized int32 symbols per open transform.
        parts = dict(motion_region_bytes=region_slots * symbol_bytes,
                     motion_local_bytes=int(len(local_rows)) * symbol_bytes,
                     m_mask_bytes=len(m_data), z_mask_bytes=len(z_data))
        return dict(m_data=m_data, z_data=z_data, m_bits=m_bits, z_bits=z_bits,
                    active_rows=active_rows, payload_rows=payload_rows,
                    local_rows=local_rows.detach().cpu(), parts=parts)

    def _gate_probabilities_for_storage(self, step):
        return self.gate_probabilities(step, use_temperature=True)

    def storage_parts(self, step, hard=False):
        """Expected mean-field bytes or byte-exact hard payload estimate.

        The estimate counts int32 motion symbols and two separately packed bit
        streams. It excludes deterministic tree membership and archive headers.
        """
        if not hard and self.stage(step) in {'gate_refine', 'incremental'}:
            return self.storage_parts(step, hard=True)
        if hard:
            cache = (self.hard_storage_cache if self.hard_storage_cache is not None
                     and self.hard_masks is not None else
                     self._encode_hard_gates(self._hard_masks_for_step(step)))
            return {name: self.base_xyz.new_tensor(value)
                    for name, value in cache['parts'].items()}

        m, z_regions, z_local = self._gate_probabilities_for_storage(step)
        expected_m_bits = _expected_m_tree_bits(m, self.active_layout)
        region_bytes = self.base_xyz.new_zeros(())
        z_region_bits = self.base_xyz.new_zeros(())
        for level in range(3):
            p_any = 1 - self._segment_product(
                1 - m, self.index[level], self.counts[level])
            region_bytes = region_bytes + (z_regions[level] * p_any).sum() * 24.
            z_region_bits = z_region_bits + p_any.sum()
        local_bytes = (m * z_local).sum() * 24.
        z_local_bits = m.sum()
        return dict(motion_region_bytes=region_bytes,
                    motion_local_bytes=local_bytes,
                    m_mask_bytes=expected_m_bits / 8,
                    z_mask_bytes=(z_region_bits + z_local_bits) / 8)

    @staticmethod
    def _segment_product(values, owner, count):
        eps = torch.finfo(values.dtype).tiny
        result = values.new_zeros(count)
        result.index_add_(0, owner, values.clamp_min(eps).log())
        return result.exp()

    def storage_bit_counts(self, step, hard=False):
        """Gate-stream lengths kept separate from byte-valued storage classes."""
        if hard:
            if self.hard_storage_cache is not None and self.hard_masks is not None:
                return dict(m_bits=self.hard_storage_cache['m_bits'],
                            z_bits=self.hard_storage_cache['z_bits'])
            masks = self._hard_masks_for_step(step)
            encoded = self._encode_hard_gates(masks)
            return dict(m_bits=encoded['m_bits'], z_bits=encoded['z_bits'])
        m, _, _ = self._gate_probabilities_for_storage(step)
        m_bits = _expected_m_tree_bits(m, self.active_layout)
        z_bits = m.new_zeros(())
        for level in range(3):
            p_any = 1 - self._segment_product(1 - m, self.index[level], self.counts[level])
            z_bits = z_bits + p_any.sum()
        return dict(m_bits=m_bits, z_bits=z_bits + m.sum())

    def storage_scale(self, step):
        return min(max((step - self.config.storage_warmup_steps) /
                       self.config.storage_ramp_steps, 0.), 1.)

    def reference_storage_bytes(self, first_frame_gs_count):
        if isinstance(first_frame_gs_count, bool) or not isinstance(first_frame_gs_count, int) or first_frame_gs_count < 1:
            raise ValueError('first_frame_gs_count must be a positive integer')
        return first_frame_gs_count * 24

    def gs_activity_fraction(self, step, hard=False):
        masks = self._hard_masks_for_step(step) if hard else self.gate_probabilities(step)
        return (masks['m'].to(self.base_xyz) if hard else masks[0]).mean()

    def compensation_deletions(self, opacity, previous_new_count):
        if not self.committed:
            raise ValueError('Previous m=0 rows can be protected only after commit')
        if isinstance(previous_new_count, bool) or not isinstance(previous_new_count, int) or previous_new_count < 0:
            raise ValueError('previous_new_count must be nonnegative')
        opacity = opacity.detach().flatten()
        if len(opacity) != len(self.base_xyz) + previous_new_count:
            raise ValueError('Sparse deletion candidates disagree with old/new row counts')
        protected = torch.zeros_like(opacity, dtype=torch.bool)
        protected[:len(self.base_xyz)] = ~self.hard_masks['m'].to(opacity.device)
        candidates = torch.where(~protected)[0]
        order = torch.argsort(opacity[candidates], stable=True)
        deleted = candidates[order[:previous_new_count]]
        return deleted, dict(enabled=True, protected_keep_old_count=int(protected.sum()),
            eligible_count=len(candidates), requested_count=previous_new_count,
            deleted_count=len(deleted), deleted_keep_count=int(protected[deleted].sum()),
            semantics='Committed m=0 old rows are protected; newly born rows remain eligible.')

    @torch.no_grad()
    def final_quantized_total_motion(self):
        xyz, rotation = self(self.config.motion_steps)
        relative = quaternion_multiply(rotation, inverse_unit_quaternion(self.base_rotation))
        return xyz - self.base_xyz, F.normalize(relative, dim=-1)

    def statistics(self):
        masks = self._hard_masks_for_step(max(self.last_step, 1))
        m = masks['m']
        levels = []
        for level in range(3):
            active = self._region_has_open_m(m, level)
            z = masks['z_regions'][level]
            levels.append(dict(level=level, denominator=int(active.sum()),
                z_on_relevant_regions=int((active & z).sum()),
                z_off_relevant_regions=int((active & ~z).sum())))
        return dict(region_counts=list(self.counts), levels=levels,
            gs=dict(denominator=len(m), m_open=int(m.sum()), m_closed=int((~m).sum()),
                    local_z_on=int((m & masks['z_local']).sum()),
                    local_z_off=int((m & ~masks['z_local']).sum())),
            semantics='Independent binary hierarchy: z gates are evaluated independently at every level; '
                'm=0 returns exact base pose. Region rows without an m-open descendant need no z state. '
                'Counts describe gate decisions, not semantic foreground/background labels.')

    @torch.no_grad()
    def _make_export_arrays(self, include_layout=True):
        masks = self._hard_masks_for_step(self.config.motion_steps)
        cache = (self.hard_storage_cache if self.hard_storage_cache is not None
                 and self.hard_masks is not None else self._encode_hard_gates(masks))
        m = masks['m'].detach().cpu()
        region_symbols, local_symbols = _quantized_motion(self, integer=True)
        arrays = dict(
            representation=np.array(LAYOUT_VERSION),
            base_digest=np.array(self._base_digest()),
            extent=np.array(self.extent, dtype=np.float32),
            motion_steps=np.asarray(self.motion_steps_vector.detach().cpu(), dtype=np.float32),
            m_mask=cache['m_data'], m_bits=np.array(cache['m_bits'], dtype=np.int64),
            z_mask=cache['z_data'], z_bits=np.array(cache['z_bits'], dtype=np.int64))
        if include_layout:
            arrays['counts'] = np.asarray(self.counts, dtype=np.int32)
            arrays['index'] = self.index.detach().cpu().numpy().astype(np.int32)
            for level, value in enumerate(self.parents):
                arrays[f'parents_l{level}'] = value.detach().cpu().numpy().astype(np.int32)
            for level, value in enumerate(self.active_layout['pivots']):
                arrays[f'pivots_l{level}'] = value.detach().cpu().numpy().astype(np.float32)
        for level in range(3):
            rows = cache['active_rows'][level]
            selected = cache['payload_rows'][level]
            arrays[f'z_region_rows_l{level}'] = rows.numpy().astype(np.int32)
            arrays[f'region_rows_l{level}'] = selected.numpy().astype(np.int32)
            arrays[f'region_payload_l{level}'] = region_symbols[level][selected.to(region_symbols[level].device)].detach().cpu().numpy().astype(np.int32)
        local_rows = cache['local_rows']
        arrays['local_rows'] = local_rows.numpy().astype(np.int32)
        arrays['local_payload'] = local_symbols[local_rows.to(local_symbols.device)].detach().cpu().numpy().astype(np.int32)
        return arrays

    def _base_digest(self):
        digest = hashlib.sha256()
        for value in (self.base_xyz, self.base_rotation):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(tuple(array.shape)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def export_motion(self, path):
        if not self.committed and self.stage(self.last_step) != 'gate_refine':
            raise RuntimeError('Sparse hierarchical export requires a fixed refine gate mask')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        layout = numpy_layout(self.active_layout)
        previous_layout = (numpy_layout(self.previous_layout)
                           if self.previous_layout is not None else None)
        arrays = self._make_export_arrays(include_layout=False)
        layout_arrays = encode_layout(
            layout, path, previous=self._export_layout_record,
            previous_layout=previous_layout, survivor_rows=self._export_survivor_rows)
        arrays.update(layout_arrays)
        np.savez_compressed(path, **arrays)
        self._export_layout_record = dict(
            path=path.resolve(), layout=layout, sha256=file_digest(path))
        hard_parts = self.storage_parts(self.config.motion_steps, hard=True)
        return dict(path=str(path), archive_bytes=path.stat().st_size,
                    layout_encoding=str(layout_arrays['layout_encoding'].item()),
                    layout_dependency=(str(layout_arrays['layout_source'].item())
                                       if 'layout_source' in layout_arrays else None),
                    m_mask_bytes=int(hard_parts['m_mask_bytes']),
                    z_mask_bytes=int(hard_parts['z_mask_bytes']),
                    motion_region_bytes=int(hard_parts['motion_region_bytes']),
                    motion_local_bytes=int(hard_parts['motion_local_bytes']),
                    payload_estimate_bytes=sum(int(hard_parts[name]) for name in (
                        'm_mask_bytes', 'z_mask_bytes', 'motion_region_bytes',
                        'motion_local_bytes')),
                    m_bits=int(self.storage_bit_counts(self.config.motion_steps, hard=True)['m_bits']),
                    z_bits=int(self.storage_bit_counts(self.config.motion_steps, hard=True)['z_bits']),
                    semantics='Payload estimate counts packed gate streams and int32 motion symbols. '
                        'Layout keyframes store grouped fine ownership and parent links; reuse frames '
                        'require the named predecessor packet in this directory. Original GS order is preserved. '
                        'archive_bytes counts the entire compressed packet, including layout and dependency metadata.')

    @staticmethod
    def decode_export_motion(path, base_xyz, base_rotation):
        """Decode absolute poses, resolving same-directory layout dependencies.

        Legacy self-contained packets remain readable. New reuse packets need
        their predecessor files, but never a training checkpoint or images.
        """
        with np.load(path, allow_pickle=False) as packet:
            representation = str(packet['representation'].item())
            if representation != LAYOUT_VERSION:
                raise ValueError(f'Unsupported sparse hierarchical export: {representation}')
            xyz = torch.as_tensor(base_xyz).detach()
            rotation = torch.as_tensor(base_rotation, device=xyz.device, dtype=xyz.dtype).detach()
            digest = hashlib.sha256()
            for value in (xyz, rotation):
                array = value.cpu().contiguous().numpy()
                digest.update(str(tuple(array.shape)).encode())
                digest.update(array.tobytes())
            if digest.hexdigest() != str(packet['base_digest'].item()):
                raise ValueError('Sparse hierarchical export base digest mismatch')
            decoded_layout = decode_layout(path, packet)
            counts = decoded_layout['counts'].tolist()
            if decoded_layout['index'].shape[1] != len(xyz):
                raise ValueError('Sparse layout GS count differs from the motion base')
            index = torch.as_tensor(decoded_layout['index'], device=xyz.device)
            parents = [torch.as_tensor(value, device=xyz.device)
                       for value in decoded_layout['parents']]
            pivots = [torch.as_tensor(value, device=xyz.device, dtype=xyz.dtype)
                      for value in decoded_layout['pivots']]
            layout = dict(index=index, counts=counts, parents=parents, pivots=pivots)
            m = decode_m_tree(packet['m_mask'], int(packet['m_bits'].item()), layout).to(xyz.device)
            z_bits = np.unpackbits(packet['z_mask'], bitorder='big')[:int(packet['z_bits'].item())]
            cursor = 0
            z_regions = [torch.zeros(count, device=xyz.device, dtype=torch.bool) for count in counts]
            for level in range(3):
                active = torch.as_tensor(packet[f'z_region_rows_l{level}'].astype(np.int64), device=xyz.device)
                count = len(active)
                z_regions[level][active] = torch.as_tensor(z_bits[cursor:cursor + count],
                                                            device=xyz.device, dtype=torch.bool)
                cursor += count
            z_local = torch.zeros(len(xyz), device=xyz.device, dtype=torch.bool)
            local_rows = torch.where(m)[0]
            z_local[local_rows] = torch.as_tensor(z_bits[cursor:cursor + len(local_rows)],
                                                   device=xyz.device, dtype=torch.bool)
            cursor += len(local_rows)
            if cursor != int(packet['z_bits'].item()):
                raise ValueError('Sparse hierarchical z bit count mismatch')
            steps = torch.as_tensor(packet['motion_steps'], device=xyz.device, dtype=xyz.dtype)
            regions = [torch.zeros(count, 6, device=xyz.device, dtype=xyz.dtype) for count in counts]
            for level in range(3):
                rows = torch.as_tensor(packet[f'region_rows_l{level}'].astype(np.int64), device=xyz.device)
                payload = torch.as_tensor(packet[f'region_payload_l{level}'].astype(np.int32),
                                          device=xyz.device).to(xyz.dtype)
                regions[level][rows] = payload * steps
            local = torch.zeros(len(xyz), 6, device=xyz.device, dtype=xyz.dtype)
            local_rows = torch.as_tensor(packet['local_rows'].astype(np.int64), device=xyz.device)
            local_payload = torch.as_tensor(packet['local_payload'].astype(np.int32),
                                            device=xyz.device).to(xyz.dtype)
            local[local_rows] = local_payload * steps
            moved_xyz, moved_rotation = _transform(
                xyz, rotation, layout, regions, local, z_regions, z_local, m)
            return (moved_xyz.detach().cpu().numpy(),
                    F.normalize(moved_rotation, dim=-1).detach().cpu().numpy())

    def capture(self):
        if self.index is None:
            return None
        return dict(layout_version=LAYOUT_VERSION, config=asdict(self.config),
            state=self.state_dict(), extent=self.extent, last_step=self.last_step,
            current_phase=self.current_phase, committed=self.committed,
            hard_masks=({name: ([value.detach().cpu().clone() for value in item]
                                if isinstance(item, list) else item.detach().cpu().clone())
                         for name, item in self.hard_masks.items()} if self.hard_masks is not None else None),
            active_layout=_layout_cpu(self.active_layout),
            previous_layout=_layout_cpu(self.previous_layout),
            materialized_layout=_layout_cpu(self.materialized_layout),
            tree_setup=self.tree_setup, last_metrics=self.last_metrics,
            stage_seconds=self.stage_seconds,
            # This cache contains CPU row-index tensors. Rebuild it from the
            # saved hard masks after restore so torch.load(map_location=...)
            # cannot silently move those indices to CUDA.
            hard_storage_cache=None,
            motion_optimizer=self.motion_optimizer.state_dict(),
            gate_optimizer=self.gate_optimizer.state_dict())

    def restore(self, packet):
        if packet is None:
            return
        if packet.get('layout_version') != LAYOUT_VERSION:
            raise ValueError('Sparse hierarchical checkpoint version mismatch')
        self._export_layout_record = None
        saved = SparseHierarchicalConfig(**packet['config'])
        saved.validate()
        if asdict(saved) != asdict(self.config):
            raise ValueError('Sparse hierarchical resume requires identical settings')
        state = dict(packet['state'])
        xyz, rotation = state['base_xyz'], state['base_rotation']
        layout_packet = packet['active_layout']
        layout = _layout(layout_packet['index'].to(xyz.device), layout_packet['counts'],
                         layout_packet['parents'], xyz, layout_packet['pivots'])
        self.setup(xyz, rotation, packet['extent'], warm_start=False,
                   frame_index=(packet.get('tree_setup') or {}).get('frame'),
                   layout_override=layout)
        self.active_layout = layout
        self.index, self.counts, self.parents = layout['index'], layout['counts'], layout['parents']
        state.setdefault('m_fine_logits', self.m_fine_logits.detach().clone())
        self.load_state_dict(state)
        self.motion_optimizer.load_state_dict(packet['motion_optimizer'])
        self.gate_optimizer.load_state_dict(packet['gate_optimizer'])
        self.previous_layout = _layout_cpu(packet.get('previous_layout'))
        self.materialized_layout = _layout_cpu(packet.get('materialized_layout'))
        hard_masks = packet.get('hard_masks')
        self.hard_masks = ({name: ([value.to(xyz.device) for value in item]
                                   if isinstance(item, list) else item.to(xyz.device))
                            for name, item in hard_masks.items()} if hard_masks is not None else None)
        self.hard_storage_cache = None
        self.last_step = packet['last_step']
        self.committed = packet['committed']
        self.last_metrics = packet['last_metrics']
        self.stage_seconds = packet['stage_seconds']
        self.tree_setup = packet.get('tree_setup')
        self.current_phase = None
        self._configure_phase(max(self.last_step, 1), restoring=True)
        if self.hard_masks is not None:
            self.hard_storage_cache = self._encode_hard_gates(self.hard_masks)
        if self.committed:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
