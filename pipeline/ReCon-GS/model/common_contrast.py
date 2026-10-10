"""Trainable common-motion/contrast tree with an exact fixed-width codec.

Each occupied root stores one world rigid transform.  Each effective binary
split stores one six-vector contrast.  For child member weights ``wl, wr`` the
local translations and world-tangent rotations are derived as
``left=-wr*contrast`` and ``right=wl*contrast``.  Unary branches inherit the
parent transform and consume neither a gate nor a contrast record.
"""

from dataclasses import asdict, dataclass
import copy
import math
from pathlib import Path
import struct
import time
import zlib
from typing import List, Optional

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
ENTROPY_MAGIC = b'CCZ1'
ENTROPY_HEADER = struct.Struct('<4sII')


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
    component_modes: bool = False  # v2: omit quantized-zero translation/rotation blocks
    direct_tree_enabled: bool = False
    direct_update_interval: int = 5
    direct_candidate_windows: Optional[int] = None
    direct_neighbors: int = 3
    direct_max_trial_splits: Optional[int] = None
    direct_initial_levels: int = 3
    direct_node_budget: int = 1000000
    direct_gradient_split_enabled: bool = False
    direct_split_score_mode: str = 'gradient_norm'  # Or rigid_residual for shared-motion conflict
    direct_preserve_split_motion: bool = False
    direct_save_trial_artifacts: bool = True
    direct_reset_on_noop: bool = True  # Legacy periodic reset even with no edit candidates
    direct_enable_swaps: bool = True
    direct_preserve_optimizer_moments: bool = False
    motion_prior_decay: float = 0.0
    motion_entropy_compression: bool = False
    birth_export_format: str = 'ply'  # Or raw_zlib: lossless attributes + tree sources
    birth_residual_enabled: bool = False
    birth_residual_camera_names: Optional[List[str]] = None
    birth_residual_max_candidates: int = 100
    birth_residual_proposals_per_view: int = 64
    birth_residual_cell_size: int = 32
    birth_residual_min_error: float = .08
    birth_residual_min_alpha: float = .05
    birth_residual_min_views: int = 2
    birth_residual_depth_ratios: Optional[List[float]] = None
    birth_residual_radius_pixels: float = 2.
    birth_residual_separation: float = 1.5
    birth_residual_occlusion_tolerance: float = .03
    birth_residual_initial_opacity: float = .1
    birth_lifetime_enabled: bool = False
    birth_lifetime_opacity_lr: float = .1
    birth_lifetime_opacity_regularization: float = .001
    birth_lifetime_expire_opacity: float = .0039
    rotation_moves_positions: bool = True  # False follows ReCon-GS covariance-only rotation
    direct_split_camera_names: Optional[List[str]] = None
    direct_split_min_visible_views: Optional[int] = None
    direct_split_top_k: Optional[int] = None
    direct_split_candidate_fraction: float = 1.0
    direct_split_cluster_iterations: Optional[int] = None
    direct_split_position_weight: Optional[float] = None
    direct_split_gradient_weight: Optional[float] = None
    direct_edit_encoding: str = 'row_json'
    direct_proposal_step: Optional[int] = None  # None preserves the late legacy proposal
    direct_split_min_child_gs: Optional[int] = None  # None reuses initial-tree minimum
    direct_enable_merges: bool = True
    direct_merge_max_disagreement: Optional[float] = None
    direct_prune_inactive_splits: bool = False
    direct_candidate_gate_logit: float = 0.0  # Early-proposal prior on edited split paths

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
                     'force_all_gates_open', 'component_modes', 'direct_tree_enabled',
                     'direct_gradient_split_enabled', 'direct_enable_merges',
                     'direct_prune_inactive_splits', 'direct_preserve_split_motion',
                     'direct_save_trial_artifacts'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be bool')
        if not isinstance(self.direct_reset_on_noop, bool):
            raise ValueError('direct_reset_on_noop must be bool')
        for name in ('direct_enable_swaps', 'direct_preserve_optimizer_moments', 'motion_entropy_compression'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be bool')
        if (isinstance(self.motion_prior_decay, bool) or not math.isfinite(self.motion_prior_decay)
                or not 0 <= self.motion_prior_decay <= 1):
            raise ValueError('motion_prior_decay must be in [0, 1]')

        if self.direct_tree_enabled and not self.component_modes:
            raise ValueError('direct tree v3 requires component_modes')
        if self.birth_appearance_action == 'color_update' and not self.component_modes:
            raise ValueError('Sparse color v5 requires component_modes')
        if self.birth_export_format not in ('ply', 'raw_zlib'):
            raise ValueError('Unknown common/contrast birth export format')
        if not isinstance(self.birth_residual_enabled, bool):
            raise ValueError('birth_residual_enabled must be bool')
        if self.birth_residual_enabled:
            names = self.birth_residual_camera_names
            if not names or len(set(names)) != len(names):
                raise ValueError('Residual births require distinct training cameras')
            for name in ('birth_residual_max_candidates', 'birth_residual_proposals_per_view',
                         'birth_residual_cell_size', 'birth_residual_min_views'):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{name} must be a positive integer')
            if self.birth_residual_min_views > len(names):
                raise ValueError('Residual minimum support exceeds camera count')
            for name in ('birth_residual_min_error', 'birth_residual_min_alpha',
                         'birth_residual_radius_pixels', 'birth_residual_separation'):
                value = getattr(self, name)
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
            if not 0 < self.birth_residual_initial_opacity < 1:
                raise ValueError('Residual initial opacity must be in (0, 1)')
            if not math.isfinite(self.birth_residual_occlusion_tolerance) or self.birth_residual_occlusion_tolerance < 0:
                raise ValueError('Residual occlusion tolerance must be finite and nonnegative')
            if (not self.birth_residual_depth_ratios or any(isinstance(value, bool) or
                    not math.isfinite(value) or value <= 0 for value in self.birth_residual_depth_ratios)):
                raise ValueError('Residual depth ratios must be positive')
        if not isinstance(self.birth_lifetime_enabled, bool):
            raise ValueError('birth_lifetime_enabled must be bool')
        if self.birth_lifetime_enabled:
            if not self.enable_incremental or not self.component_modes:
                raise ValueError('Birth lifetime fitting requires incremental steps')
            if (not math.isfinite(self.birth_lifetime_opacity_lr) or self.birth_lifetime_opacity_lr <= 0 or
                    not math.isfinite(self.birth_lifetime_opacity_regularization) or self.birth_lifetime_opacity_regularization < 0 or
                    not 0 < self.birth_lifetime_expire_opacity < 1):
                raise ValueError('Invalid birth lifetime settings')
        if not isinstance(self.rotation_moves_positions, bool):
            raise ValueError('rotation_moves_positions must be bool')
        if not self.rotation_moves_positions:
            if not self.component_modes:
                raise ValueError('Covariance-only rotation requires component modes')
            if self.direct_tree_enabled and (self.direct_enable_swaps or self.direct_enable_merges):
                raise ValueError('Covariance-only motion currently supports motion-preserving splits only')
        if self.direct_edit_encoding not in ('row_json', 'node_ops'):
            raise ValueError('Unknown direct tree edit encoding')
        if self.direct_split_score_mode not in ('gradient_norm', 'rigid_residual'):
            raise ValueError('Unknown direct split score mode')
        if self.direct_proposal_step is not None:
            value = self.direct_proposal_step
            if (isinstance(value, bool) or not isinstance(value, int) or
                    not self.dense_fit_steps <= value <= self.dense_fit_steps + self.gate_compress_steps):
                raise ValueError('Proposal step must follow dense fit and precede motion refinement')
        if self.direct_split_min_child_gs is not None:
            value = self.direct_split_min_child_gs
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Split child minimum must be a positive integer')
        if self.direct_merge_max_disagreement is not None:
            value = self.direct_merge_max_disagreement
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Merge disagreement must be finite and nonnegative')
        if self.direct_prune_inactive_splits and self.direct_edit_encoding != 'node_ops':
            raise ValueError('Split pruning requires replayable node operations')
        if not math.isfinite(self.direct_candidate_gate_logit):
            raise ValueError('Candidate gate logit must be finite')
        for name in ('direct_update_interval', 'direct_neighbors', 'direct_initial_levels', 'direct_node_budget'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, minimum in (('direct_candidate_windows', 0), ('direct_max_trial_splits', 1)):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
                raise ValueError(f'{name} must be None or an integer >= {minimum}')
        if self.direct_tree_enabled and not 1 <= self.direct_initial_levels <= self.max_motion_levels:
            raise ValueError('Initial tree depth exceeds the motion-level budget')
        if self.direct_gradient_split_enabled:
            if not self.direct_tree_enabled:
                raise ValueError('Gradient split selection requires direct_tree_enabled')
            names = self.direct_split_camera_names
            if not names or len(set(names)) != len(names):
                raise ValueError('Split scoring requires distinct training camera names')
            self.direct_split_camera_names = list(names)
            for name in ('direct_split_min_visible_views', 'direct_split_cluster_iterations'):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f'{name} must be a positive integer')
            if self.direct_split_min_visible_views > len(names):
                raise ValueError('Split observation requirement exceeds scoring view count')
            top_k = self.direct_split_top_k
            if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
                raise ValueError('direct_split_top_k must be None or a positive integer')
            fraction = self.direct_split_candidate_fraction
            if (isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1):
                raise ValueError('direct_split_candidate_fraction must be in (0, 1]')
            if (top_k is not None and self.direct_max_trial_splits is not None and
                    top_k > self.direct_max_trial_splits):
                raise ValueError('Split top_k exceeds the local split budget')
            for name in ('direct_split_position_weight', 'direct_split_gradient_weight'):
                value = getattr(self, name)
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                        not math.isfinite(value) or value <= 0):
                    raise ValueError(f'{name} must be finite and positive')


def _member_counts(layout):
    result = []
    for level, count in enumerate(layout['counts']):
        owner = layout['index'][level]
        valid = owner >= 0
        result.append(torch.bincount(owner[valid], minlength=count))
    return result


def _effective_splits(layout):
    """Return binary masks, unary-child maps and fixed member weights."""
    children = _children(layout)
    members = _member_counts(layout)
    occupancy = [value > 0 for value in members]
    member_lists = [value.detach().cpu().tolist() for value in members]
    gate_positions = [{int(node): i for i, node in enumerate(rows.tolist())}
                      for rows in layout['refine_rows']]
    binary, unary, weights = [], [], []
    for level in range(len(layout['counts']) - 1):
        binary_values = [False] * layout['counts'][level]
        unary_values = [-1] * layout['counts'][level]
        weight_level = layout['pivots'][level].new_zeros(
            (len(layout['refine_rows'][level]), 2))
        positions, left_rows, right_rows = [], [], []
        for parent, candidates in enumerate(children[level]):
            occupied = [child for child in candidates if member_lists[level + 1][child] > 0]
            if len(occupied) == 1:
                unary_values[parent] = occupied[0]
            elif len(occupied) == 2:
                binary_values[parent] = True
                position = gate_positions[level].get(parent)
                if position is None:
                    raise ValueError('Effective binary split has no gate row')
                left, right = occupied
                positions.append(position)
                left_rows.append(left)
                right_rows.append(right)
            elif len(occupied) > 2:
                raise ValueError('Common/contrast backend requires a binary forest')
        device = layout['index'].device
        binary_level = torch.tensor(binary_values, dtype=torch.bool, device=device)
        unary_level = torch.tensor(unary_values, dtype=torch.long, device=device)
        if positions:
            left = members[level + 1][torch.tensor(left_rows, device=device)]
            right = members[level + 1][torch.tensor(right_rows, device=device)]
            total = left + right
            # Keep integer true-division on this device, as in the scalar codec.
            values = torch.stack((left / total, right / total), dim=-1)
            weight_level[torch.tensor(positions, device=device)] = values.to(weight_level)
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


def decompose_terminal_transforms(layout, terminal_rotation, terminal_delta, affected_roots=None):
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
    root_owner = [torch.arange(counts[0], device=index.device)]
    for parent in layout['parents']:
        root_owner.append(root_owner[-1][parent])
    selected = [set(range(count)) if affected_roots is None else
                {node for node, root in enumerate(owner.cpu().tolist()) if root in affected_roots}
                for count, owner in zip(counts, root_owner)]
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
            if node not in selected[level]:
                continue
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
            if node not in selected[level]:
                continue
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
        prior = None
        if self.config.motion_prior_decay and self.index is not None and self.committed:
            root, contrasts, *_ = self._quantized_parameters()
            gates = self.refinement_masks(self.config.motion_steps)
            paths = self._paths(gates)
            prior = (root.detach().clone(),
                [(value * (paths[level][self.active_layout['refine_rows'][level]] * gates[level])[:, None]).detach().clone()
                 for level, value in enumerate(contrasts)],
                [rows.detach().cpu().tolist() for rows in self.active_layout['refine_rows']])
        if layout_override is None:
            if (isinstance(frame_index, bool) or not isinstance(frame_index, int)
                    or frame_index < 1):
                raise ValueError('Common/contrast setup requires a positive frame_index')
            rebuild = old_layout is None or (not self.config.direct_tree_enabled and
                (frame_index - 1) % self.config.tree_rebuild_interval == 0)
            if rebuild:
                layout = stable_binary_forest(xyz, self.config.coarse_region_gs,
                    self.config.min_child_gs,
                    self.config.direct_initial_levels if self.config.direct_tree_enabled else self.config.max_motion_levels)
                if self.config.direct_tree_enabled:
                    while len(layout['counts']) < self.config.max_motion_levels:
                        layout['index'] = torch.cat((layout['index'], torch.full_like(layout['index'][:1], -1)))
                        layout['counts'].append(0)
                        layout['parents'].append(torch.empty(0, device=xyz.device, dtype=torch.long))
                    layout = _binary_layout(layout['index'], layout['counts'], layout['parents'], xyz)
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
        self._frame_coding_layout = _copy_layout(layout)
        self._frame_edit_payload = None
        self._pending_edit_operations = None
        self._node_edit_cost_cache = None
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
        inherited = prior is not None and not rebuild and len(prior[0]) == self.counts[0]
        if inherited:
            decay = self.config.motion_prior_decay
            with torch.no_grad():
                self.root_translation.copy_(prior[0][:, :3] * decay)
                self.root_rotation.copy_(prior[0][:, 3:] * decay)
                for level, rows in enumerate(layout['refine_rows']):
                    previous = {node: i for i, node in enumerate(prior[2][level])}
                    for position, node in enumerate(rows.detach().cpu().tolist()):
                        if node in previous:
                            value = prior[1][level][previous[node]] * decay
                            self.contrast_translation[level][position].copy_(value[:3])
                            self.contrast_rotation[level][position].copy_(value[3:])
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
            tree_rebuilt=bool(rebuild), initialization_decay=self.config.motion_prior_decay if inherited else 0.,
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

    def motion_quantization_metadata(self):
        """Report this backend's effective grid and zero-bin semantics."""
        return dict(
            translation_step_extent=self.config.translation_step_extent,
            translation_step_world=self.extent * self.config.translation_step_extent,
            rotation_step_rad=self.config.rotation_step_rad,
            guaranteed_zero_translation_below_extent=self.config.translation_step_extent / 2,
            guaranteed_zero_rotation_below_rad=self.config.rotation_step_rad / 2,
            fixed_motion_bits=self.config.fixed_motion_bits,
            quantized_records='Root shared transforms and split contrasts',
            unused_legacy_motion_fields=['motion_xyz_step_extent', 'motion_rotation_step_rad'],
            semantics='Each record component rounds to the nearest signed integer on this grid. '
                      'A zero contrast inherits its shared ancestor motion. Member weights are '
                      'applied after contrast quantization; the composed GS pose has no additional '
                      'per-GS quantization. Exact nonzero GS changes are not counts of independent '
                      'stored GS motion records.')

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
            selected_only=selected, terminal=terminal,
            rotate_positions=self.config.rotation_moves_positions)

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
        def component_cost(scaled, symbols):
            if not self.config.component_modes:
                return self._support(scaled, symbols, support_differentiable) * 6
            return 3 * (self._support(scaled[..., :3], symbols[..., :3], support_differentiable) +
                        self._support(scaled[..., 3:], symbols[..., 3:], support_differentiable))

        root_support = component_cost(root_scaled, root_symbols)
        bits = self.config.fixed_motion_bits
        presence = paths[0].sum() * (2 if self.config.component_modes else 1)
        payload = (paths[0] * root_support * bits).sum()
        flags = self.base_xyz.new_zeros(())
        for level, rows in enumerate(self.active_layout['refine_rows']):
            binary = self._cached_binary[level][rows].to(self.base_xyz)
            reach = paths[level][rows] * binary
            flags = flags + reach.sum()
            opened = gates[level].to(self.base_xyz)
            contrast_reach = reach * opened
            presence = presence + contrast_reach.sum() * (2 if self.config.component_modes else 1)
            support = component_cost(split_scaled[level], split_symbols[level])
            payload = payload + (contrast_reach * support * bits).sum()
        total_bits = presence + flags + payload
        padding = (self.base_xyz.new_tensor(float((-int(total_bits)) % 8))
                   if hard else self.base_xyz.new_zeros(()))
        edits = self._structural_payload()
        structural_bytes = 4 + len(edits) if (self.config.direct_tree_enabled or
            self.config.birth_appearance_action == 'color_update') else 0
        if (support_differentiable and self.config.direct_prune_inactive_splits and
                self._pending_edit_operations):
            structural_bytes = self._expected_node_edit_bytes(paths, gates)
        structure_cost = (structural_bytes if torch.is_tensor(structural_bytes) else
                          self.base_xyz.new_tensor(float(structural_bytes)))
        result = dict(structure_edit_bytes=structure_cost,
            codec_header_bytes=self.base_xyz.new_tensor(float(CODEC_HEADER_BYTES)),
            payload_padding_bytes=padding / 8,
            stop_refine_flag_bytes=flags / 8,
            motion_presence_flag_bytes=presence / 8,
            fixed_symbol_bytes=payload / 8)
        if frozen:
            self._frozen_storage_parts[hard] = {key: value.detach() for key, value in result.items()}
            return dict(self._frozen_storage_parts[hard])
        return result

    def _expected_node_edit_bytes(self, paths, gates):
        """Differentiable expected split-body cost; hard writer remains exact."""
        from .tree_edit_codec import encode_node_edits, encode_operation
        cached = self._node_edit_cost_cache
        if cached is None:
            operations = self._pending_edit_operations
            bodies = [len(encode_operation(op)) for op in operations]
            prefix = len(encode_node_edits(operations)) - sum(bodies)
            constant = sum(size for op, size in zip(operations, bodies) if op['kind'] != 'split')
            groups = {}
            for op, size in zip(operations, bodies):
                if op['kind'] != 'split':
                    continue
                level, node = op['level'], op['node']
                groups.setdefault(level, []).append((node, self._cached_gate_positions[level][node], size))
            groups = {level: (torch.tensor([r[0] for r in rows], device=self.base_xyz.device),
                              torch.tensor([r[1] for r in rows], device=self.base_xyz.device),
                              self.base_xyz.new_tensor([r[2] for r in rows]))
                      for level, rows in groups.items()}
            cached = self._node_edit_cost_cache = (prefix, constant, groups)
        prefix, constant, groups = cached
        probabilities, costs = [], self.base_xyz.new_zeros(())
        for level, (nodes, positions, size) in groups.items():
            probability = paths[level][nodes] * gates[level][positions]
            probabilities.append(probability)
            costs = costs + (size * probability).sum()
        present = (self.base_xyz.new_ones(()) if constant or not probabilities else
                   1 - (1 - torch.cat(probabilities)).prod())
        return 4 + constant + prefix * present + costs

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
            if self.config.component_modes:
                mode = int(any(values[:3])) + 2 * int(any(values[3:]))
                writer.write(mode, 2)
                for offset, enabled in ((0, mode & 1), (3, mode & 2)):
                    if enabled:
                        for value in values[offset:offset + 3]:
                            writer.write(_signed_code(value, bits), bits)
            else:
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

    def _structural_payload(self):
        if not self.config.direct_tree_enabled:
            return b''
        if self._frame_edit_payload is not None:
            return self._frame_edit_payload
        if layout_digest(self._frame_coding_layout) == layout_digest(self.active_layout):
            self._frame_edit_payload = b''
        elif self.config.direct_edit_encoding == 'node_ops':
            from .tree_edit_codec import encode_node_edits
            if self._pending_edit_operations is None:
                raise ValueError('Changed node-op layout lacks replay operations')
            self._frame_edit_payload = encode_node_edits(self._pending_edit_operations)
        else:
            from .direct_tree import layout_edits
            self._frame_edit_payload = layout_edits(self._frame_coding_layout, self.active_layout)
        return self._frame_edit_payload

    def encode_motion(self, *, frame_index, previous_state_digest,
                      post_motion_digest, previous_count, deleted_count, birth_count,
                      appearance_present=False, opacity_present=False):
        """Return the exact deployment packet and size without filesystem I/O."""
        payload, bit_count = self._encode_payload(self.config.motion_steps)
        version = ((4 if self.config.direct_edit_encoding == 'node_ops' else 3)
                   if self.config.direct_tree_enabled else 2 if self.config.component_modes else 1)
        if self.config.birth_appearance_action == 'color_update':
            version = 5
        if self.config.birth_lifetime_enabled:
            version = 6
        if not self.config.rotation_moves_positions:
            version = 7
        if appearance_present and version < 5:
            raise ValueError('Appearance stream requires common/contrast v5')
        if opacity_present and not self.config.birth_lifetime_enabled:
            raise ValueError('Opacity stream requires common/contrast v6')
        edits = self._structural_payload()
        structure = struct.pack('<I', len(edits)) + edits if version >= 3 else b''
        flags = int(appearance_present) | (2 * int(opacity_present)) | (4 * int(self.config.birth_lifetime_enabled))
        flags |= 8 * int(not self.config.rotation_moves_positions)
        header = CODEC_HEADER.pack(CODEC_MAGIC, version, flags, CODEC_HEADER_BYTES,
            int(frame_index), int(previous_count), len(self.base_xyz), int(deleted_count),
            int(birth_count), int(bit_count), float(self.extent), bytes.fromhex(previous_state_digest),
            bytes.fromhex(post_motion_digest), layout_digest(self.active_layout))
        packet = header + structure + payload
        if len(packet) != CODEC_HEADER_BYTES + len(structure) + math.ceil(bit_count / 8):
            raise AssertionError('Common/contrast stream size disagrees with encoded bit length')
        parameter_bytes = len(packet)
        if self.config.motion_entropy_compression:
            packet = ENTROPY_HEADER.pack(ENTROPY_MAGIC, len(packet), zlib.crc32(packet)) + zlib.compress(packet, 3)
        return packet, dict(archive_bytes=len(packet), entropy_compressed=self.config.motion_entropy_compression,
            parameter_bytes=parameter_bytes, payload_bits=bit_count,
            payload_padding_bits=(-bit_count) % 8, codec_header_bytes=CODEC_HEADER_BYTES,
            frame=frame_index, layout_encoding=('common_contrast_covariance_rotation_v7' if version == 7 else
                'common_contrast_birth_lifetime_v6' if version == 6 else
                'common_contrast_sparse_color_v5' if version == 5 else
                'common_contrast_node_edits_v4' if version == 4 else
                'common_contrast_direct_tree_v3' if version == 3 else
                'common_contrast_component_v2' if self.config.component_modes else CODEC_VERSION),
            structure_edit_bytes=len(structure),
            component_modes=self.config.component_modes,
            fixed_motion_bits=self.config.fixed_motion_bits,
            semantics=('v2 uses a two-bit identity/translation/rotation/full mode and omits quantized-zero blocks. '
                'Centers retain existing deterministic pivot semantics. ' if self.config.component_modes else '') +
                'Each occupied root has a motion record. '
                'Each reached effective binary split has one refine bit; an open split has one '
                'contrast presence bit and optional six-vector. Unary branches consume no fields.')

    def export_motion(self, path, **kwargs):
        packet, report = self.encode_motion(**kwargs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packet)
        if path.stat().st_size != len(packet):
            raise AssertionError('Common/contrast file size disagrees with encoded packet')
        return dict(report, path=str(path))

    @staticmethod
    def read_header(path):
        """Read a deployment file or an already encoded in-memory packet."""
        packet = bytes(path) if isinstance(path, (bytes, bytearray, memoryview)) else Path(path).read_bytes()
        if packet.startswith(ENTROPY_MAGIC):
            if len(packet) < ENTROPY_HEADER.size:
                raise ValueError('Truncated entropy envelope')
            _, length, checksum = ENTROPY_HEADER.unpack_from(packet)
            if length > 256 * 1024 * 1024:
                raise ValueError('Oversized entropy envelope')
            decoder = zlib.decompressobj()
            body = decoder.decompress(packet[ENTROPY_HEADER.size:], length + 1)
            if (len(body) != length or not decoder.eof or decoder.unused_data or
                    decoder.unconsumed_tail or zlib.crc32(body) != checksum):
                raise ValueError('Entropy envelope failed validation')
            packet = body
        if len(packet) < CODEC_HEADER_BYTES:
            raise ValueError('Truncated common/contrast header')
        values = CODEC_HEADER.unpack(packet[:CODEC_HEADER_BYTES])
        (magic, version, flags, header_size, frame, previous_count, base_count,
         deleted_count, birth_count, bit_count, extent, previous_digest,
         base_digest, topology) = values
        invalid_flags = (flags != 0 if version < 5 else flags not in (0, 1) if version == 5 else
                         bool(flags & ~7) or not bool(flags & 4) if version == 6 else
                         bool(flags & ~15) or not bool(flags & 8) or (bool(flags & 2) and not bool(flags & 4)))
        if (magic != CODEC_MAGIC or version not in (1, 2, 3, 4, 5, 6, 7) or
                invalid_flags or header_size != CODEC_HEADER_BYTES):
            raise ValueError('Unsupported common/contrast stream')
        if not math.isfinite(extent) or extent <= 0:
            raise ValueError('Invalid scene extent in common/contrast stream')
        cursor = CODEC_HEADER_BYTES
        edits = b''
        if version >= 3:
            if len(packet) < cursor + 4:
                raise ValueError('Truncated structure size')
            size = struct.unpack_from('<I', packet, cursor)[0]
            cursor += 4
            if size > len(packet) - cursor:
                raise ValueError('Truncated structure edits')
            edits = packet[cursor:cursor + size]
            cursor += size
        if len(packet) != cursor + math.ceil(bit_count / 8):
            raise ValueError('Common/contrast stream length disagrees with bit count')
        return dict(codec=('common_contrast_covariance_rotation_v7' if version == 7 else
                'common_contrast_birth_lifetime_v6' if version == 6 else
                'common_contrast_sparse_color_v5' if version == 5 else
                'common_contrast_node_edits_v4' if version == 4 else
                'common_contrast_direct_tree_v3' if version == 3 else
                'common_contrast_component_v2' if version == 2 else CODEC_VERSION),
            component_modes=(version >= 2), structure_edits=edits, appearance_present=bool(flags & 1),
            opacity_present=bool(flags & 2), lifetime_enabled=bool(flags & 4),
            rotation_moves_positions=not bool(flags & 8),
            structure_edit_bytes=cursor - CODEC_HEADER_BYTES,
            frame=frame, previous_count=previous_count,
            base_count=base_count, deleted_count=deleted_count, birth_count=birth_count,
            payload_bits=bit_count, extent=extent, previous_digest=previous_digest.hex(),
            post_motion_digest=base_digest.hex(), topology_digest=topology.hex(),
            payload=packet[cursor:])

    @classmethod
    def decode_motion(cls, path, base_xyz, base_rotation, config, layout,
                      *, coding_state=None):
        header = cls.read_header(path)
        if len(base_xyz) != header['base_count']:
            raise ValueError('Post-deletion base count disagrees with stream')
        if layout_digest(layout).hex() != header['topology_digest']:
            raise ValueError('Deterministic tree digest disagrees with stream')
        occupancy, children, members, positions, binary, unary, weights = _effective_splits(layout)
        occupied_values = [value.detach().cpu().tolist() for value in occupancy]
        binary_values = [value.detach().cpu().tolist() for value in binary]
        unary_values = [value.detach().cpu().tolist() for value in unary]
        reader = _BitReader(header['payload'], header['payload_bits'])
        counts = layout['counts']
        root_codes = [[0] * 6 for _ in range(counts[0])]
        split_codes = [[[0] * 6 for _ in range(len(rows))] for rows in layout['refine_rows']]
        gate_values = [[False] * len(rows) for rows in layout['refine_rows']]
        bits = config.fixed_motion_bits
        steps = base_xyz.new_tensor([float(header['extent']) * config.translation_step_extent] * 3 +
                                    [config.rotation_step_rad] * 3)

        def motion():
            if header['component_modes']:
                mode = reader.read(2)
                values = [0] * 6
                for offset, enabled in ((0, mode & 1), (3, mode & 2)):
                    if enabled:
                        block = [_signed_value(reader.read(bits), bits) for _ in range(3)]
                        if not any(block):
                            raise ValueError('A zero component must omit its payload')
                        values[offset:offset + 3] = block
            else:
                if not reader.read():
                    return [0] * 6
                values = [_signed_value(reader.read(bits), bits) for _ in range(6)]
                if not any(values):
                    raise ValueError('A zero motion must omit its fixed-width payload')
            return values

        def visit(level, node):
            if level >= len(gate_values):
                return
            if binary_values[level][node]:
                position = positions[level][node]
                opened = bool(reader.read())
                gate_values[level][position] = opened
                if opened:
                    split_codes[level][position] = motion()
                    for child in children[level][node]:
                        if occupied_values[level + 1][child]:
                            visit(level + 1, child)
            elif unary_values[level][node] >= 0:
                visit(level + 1, unary_values[level][node])

        for node, occupied in enumerate(occupied_values[0]):
            if occupied:
                root_codes[node] = motion()
                visit(0, node)
        if reader.cursor != header['payload_bits']:
            raise ValueError('Common/contrast stream has unread bits')
        coded_root = torch.tensor(root_codes, device=base_xyz.device, dtype=torch.long).reshape(-1, 6)
        coded_splits = [torch.tensor(value, device=base_xyz.device, dtype=torch.long).reshape(-1, 6)
                        for value in split_codes]
        root = coded_root.to(base_xyz) * steps
        contrasts = [value.to(base_xyz) * steps for value in coded_splits]
        gates = [torch.tensor(value, device=base_xyz.device, dtype=torch.bool) for value in gate_values]

        # Reconstruct child locals from the single stored split contrast.
        regions = [root]
        for level in range(len(gates)):
            child_motion = base_xyz.new_zeros(counts[level + 1], 6)
            pairs = [(position, *[child for child in children[level][node]
                                 if occupied_values[level + 1][child]])
                     for position, node in enumerate(layout['refine_rows'][level].tolist())
                     if binary_values[level][node]]
            if pairs:
                pair_rows = torch.tensor(pairs, device=base_xyz.device, dtype=torch.long)
                position, left, right = pair_rows.unbind(-1)
                values = contrasts[level][position]
                child_motion[left] = -weights[level][position, 1, None] * values
                child_motion[right] = weights[level][position, 0, None] * values
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
            regions, paths, selected_only=True, terminal=terminal,
            rotate_positions=header['rotation_moves_positions'])
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
        self.birth_layout_sources = computed
        full_index = torch.cat((self.index, self.index[:, computed]), dim=1)
        self.materialized_layout = _copy_layout(_binary_layout(full_index, self.counts,
            self.parents, materialized_xyz, self.active_layout['pivots']))
        return self.materialized_layout

    def compensation_deletions(self, opacity, previous_new_count, birth_frames=None):
        if not self.committed:
            raise ValueError('Common/contrast deletion selection needs a committed prior frame')
        if len(opacity) != len(self.base_xyz) + previous_new_count:
            raise ValueError('Deletion count disagrees with the previous common/contrast tree')
        if self.config.birth_lifetime_enabled:
            from .birth_lifetime import expiration_deletions
            xyz, rotation = self(self.config.motion_steps)
            changed = ((xyz != self.base_xyz).any(-1) | (rotation != self.base_rotation).any(-1))
            if birth_frames is None:
                raise ValueError('Birth lifetime deletion requires point origins')
            deleted, report = expiration_deletions(opacity, previous_new_count, changed,
                birth_frames, self.config.birth_lifetime_expire_opacity)
            return deleted, dict(report, enabled=True)
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
            frame_coding_layout=_copy_layout(self._frame_coding_layout),
            edit_operations=copy.deepcopy(self._pending_edit_operations),
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
        self._frame_coding_layout = _copy_layout(packet.get('frame_coding_layout', packet['active_layout']))
        self._pending_edit_operations = copy.deepcopy(packet.get('edit_operations'))
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
        self.birth_frames = None

    def advance(self, motion_path, birth_path, raw, appearance_path=None, opacity_path=None):
        header = CommonContrastMotion.read_header(motion_path)
        frame = header['frame']
        if frame != self.last_frame + 1:
            raise ValueError(f'Expected frame {self.last_frame + 1}, received {frame}')
        if len(raw['xyz']) != header['previous_count'] or reference_digest(raw) != header['previous_digest']:
            raise ValueError('Previous Gaussian state disagrees with common/contrast stream')
        if not header['lifetime_enabled'] and header['deleted_count'] != self.previous_birth_count:
            raise ValueError('Deletion count does not match prior birth count')
        if self.birth_frames is None:
            self.birth_frames = torch.zeros(len(raw['xyz']), device=raw['xyz'].device, dtype=torch.long)
        if header['lifetime_enabled']:
            from .birth_lifetime import expiration_deletions
            if not self.config.birth_lifetime_enabled:
                raise ValueError('Birth lifetime stream requires matching configuration')
            changed = self.previous_changed
            if changed is None and frame == 1:
                changed = torch.zeros(len(raw['xyz']), device=raw['xyz'].device, dtype=torch.bool)
            deleted, _ = expiration_deletions(raw['opacity'], self.previous_birth_count, changed,
                self.birth_frames, self.config.birth_lifetime_expire_opacity)
        else:
            deleted = _delete_rows(raw, self.previous_birth_count, self.previous_changed)
        if len(deleted) != header['deleted_count']:
            raise ValueError('Deterministic deleted count disagrees with stream')
        rows = survivor_rows(len(raw['xyz']), deleted)
        base = {name: raw[name][rows].clone() for name in RAW_FIELDS}
        if len(base['xyz']) != header['base_count']:
            raise ValueError('Deterministic deletion disagrees with stream base count')
        rebuild = self.materialized_layout is None if self.config.direct_tree_enabled else (frame - 1) % self.config.tree_rebuild_interval == 0
        if rebuild:
            layout = stable_binary_forest(base['xyz'], self.config.coarse_region_gs,
                self.config.min_child_gs,
                self.config.direct_initial_levels if self.config.direct_tree_enabled else self.config.max_motion_levels)
            if self.config.direct_tree_enabled:
                while len(layout['counts']) < self.config.max_motion_levels:
                    layout['index'] = torch.cat((layout['index'], torch.full_like(layout['index'][:1], -1)))
                    layout['counts'].append(0)
                    layout['parents'].append(torch.empty(0, device=base['xyz'].device, dtype=torch.long))
                layout = _binary_layout(layout['index'], layout['counts'], layout['parents'], base['xyz'])
        else:
            if self.materialized_layout is None:
                raise ValueError('Intermediate frame has no cached common/contrast tree')
            old = self.materialized_layout
            layout = _binary_layout(old['index'].to(base['xyz'].device)[:, rows],
                old['counts'], old['parents'], base['xyz'], old['pivots'])
        if header['structure_edits']:
            from .direct_tree import apply_layout_edits
            layout = apply_layout_edits(layout, header['structure_edits'], base['xyz'])
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
        if header['appearance_present']:
            if appearance_path is None:
                raise ValueError('Sparse color stream is required by motion header')
            from .sparse_color import apply_sparse_color_packet
            base = apply_sparse_color_packet(appearance_path, base)
        elif appearance_path is not None:
            raise ValueError('Unexpected sparse color stream')
        if header['opacity_present']:
            if opacity_path is None:
                raise ValueError('Historical birth opacity stream is required')
            from .birth_lifetime import apply_opacity_packet
            base = apply_opacity_packet(opacity_path, base)
        elif opacity_path is not None:
            raise ValueError('Unexpected historical birth opacity stream')
        if reference_digest(base) != header['post_motion_digest']:
            raise ValueError('Decoded post-motion state digest mismatch')
        sources = None
        if birth_path:
            from .birth_codec import MAGIC as BIRTH_MAGIC, decode_birth_packet
            birth_packet = Path(birth_path).read_bytes()
            if birth_packet.startswith(BIRTH_MAGIC):
                births, sources = decode_birth_packet(birth_packet, base)
            else:
                births = _load_birth_ply(birth_path, base['xyz'].device)
        else:
            births = {name: base[name].new_empty((0,) + base[name].shape[1:]) for name in RAW_FIELDS}
        if len(births['xyz']) != header['birth_count']:
            raise ValueError('Birth PLY count disagrees with stream')
        if len(births['xyz']):
            if sources is None:
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
        self.birth_frames = torch.cat((self.birth_frames[rows.to(self.birth_frames.device)],
            torch.full((self.previous_birth_count,), frame, device=self.birth_frames.device, dtype=torch.long)))
        self.last_frame = frame
        self.last_pose_xyz = torch.cat((xyz, births['xyz'].to(xyz)), dim=0)
        self.last_pose_rotation = torch.cat((rotation, births['rotation'].to(rotation)), dim=0)
        return base, dict(frame=frame, deleted_rows=deleted.detach().cpu().tolist(),
            birth_count=self.previous_birth_count, changed_old_count=int(changed.sum()),
            tree_rebuilt=rebuild, topology_digest=header['topology_digest'],
            coded_payload_bits=header['payload_bits'], structure_edit_bytes=header['structure_edit_bytes'], motion_path=str(motion_path),
            birth_path=str(birth_path) if birth_path else None)
