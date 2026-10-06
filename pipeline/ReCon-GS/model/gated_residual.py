"""Persistent three-level rigid tree with learned KEEP/SHARE/SPLIT gates."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from libgs.utils.general import quaternion_multiply

from .region_rigid import apply_region_rigid, rotation_vector_quaternion
from .residual_motion import (QuantizedResidualMotion, QuantizedResidualConfig,
                              integer_symbols, quantized_symbols,
                              inverse_unit_quaternion, stable_layout)


from .gs_activity_cost import candidate_activity_support, expected_activity_fraction


LAYOUT_VERSION = 'persistent_gated_rigid_storage_v2'
KEEP, SHARE, SPLIT = 0, 1, 2


def _layout(index, counts, parents, xyz, previous_pivots=None):
    """The index records identity; pivots are recomputed from surviving base rows."""
    index = index.long()
    depth = len(counts)
    if depth not in (1, 3) or index.shape != (depth, len(xyz)) or not len(xyz):
        raise ValueError('Persistent layout needs one or three owner rows and nonempty xyz')
    # Pivots are reference geometry. Ordered CPU float64 sums keep
    # persistent layout setup deterministic across devices and checkpoint resume.
    points = xyz.detach().cpu().to(torch.float64).numpy()
    pivots = []
    for level, count in enumerate(counts):
        owner = index[level].detach().cpu().numpy()
        if count < 1 or np.any((owner < 0) | (owner >= count)):
            raise ValueError('Persistent region owner is out of range')
        sums = np.zeros((count, 3), dtype=np.float64)
        np.add.at(sums, owner, points)
        sizes = np.bincount(owner, minlength=count)[:, None]
        means = sums / np.maximum(sizes, 1)
        if previous_pivots is not None:
            fallback = previous_pivots[level].detach().cpu().to(torch.float64).numpy()
            means = np.where(sizes > 0, means, fallback)
        pivots.append(torch.from_numpy(means).to(xyz))
    parent = [torch.as_tensor(row, device=index.device, dtype=torch.long) for row in parents]
    if len(parent) != depth - 1 or any(len(parent[n]) != counts[n + 1]
                                     for n in range(depth - 1)):
        raise ValueError('Persistent parent array length mismatch')
    for level in range(depth - 1):
        if bool(((parent[level] < 0) | (parent[level] >= counts[level])).any()):
            raise ValueError('Persistent parent index out of range')
        if bool((parent[level][index[level + 1]] != index[level]).any()):
            raise ValueError('Persistent parent and owner disagree')
    return dict(index=index, counts=list(map(int, counts)), parents=parent, pivots=pivots)


def _layout_cpu(layout):
    if layout is None:
        return None
    return dict(index=layout['index'].detach().cpu().clone(),
                counts=list(layout['counts']),
                parents=[row.detach().cpu().clone() for row in layout['parents']],
                pivots=[row.detach().cpu().clone() for row in layout['pivots']])


def _occupancy(layout):
    return [torch.bincount(layout['index'][n], minlength=layout['counts'][n]) > 0
            for n in range(len(layout['counts']))]


@torch.no_grad()
def _inherit_gate_logits(old_owner, new_owner, old_logits, new_count, initial_logits):
    """Match same-level regions by surviving GS membership, not spatial distance.

    A strict majority source supplies its exact logits. Without a majority,
    agreeing source states supply member-weighted logits; conflicting states
    start from the configured initial logits. No dense region-pair matrix.
    """
    old = old_owner.detach().cpu().numpy()
    new = new_owner.detach().cpu().numpy()
    logits = old_logits.detach().cpu().numpy()
    result = np.broadcast_to(np.asarray(initial_logits, dtype=logits.dtype),
                             (new_count, 3)).copy()
    order = np.argsort(new, kind='stable')
    boundaries = np.concatenate(([0], np.cumsum(np.bincount(new, minlength=new_count))))
    for region in range(new_count):
        members = old[order[boundaries[region]:boundaries[region + 1]]]
        if not len(members):
            continue
        sources, counts = np.unique(members, return_counts=True)
        winner = counts.argmax()
        if 2 * counts[winner] > len(members):
            result[region] = logits[sources[winner]]
        elif np.unique(logits[sources].argmax(axis=1)).size == 1:
            result[region] = np.average(logits[sources], axis=0, weights=counts)
    return torch.from_numpy(result).to(old_logits)


@torch.no_grad()
def _inherit_region_priors(old_centers, new_centers, values, neighbors):
    """Same-level KNN inheritance, with Markley quaternion averaging as in ReCon-GS.

    Centers: [regions, 3]; values: [old regions, 6] (translation, rotation vector).
    K is capped when a level has fewer old regions. Gates are inherited separately.
    Chunking bounds temporary distance memory and does not change the model.
    """
    k = min(neighbors, len(old_centers))
    inherited = []
    for centers in new_centers.split(256):
        nearest = torch.cdist(centers, old_centers).argsort(dim=1, stable=True)[:, :k]
        samples = values[nearest]
        quaternions = rotation_vector_quaternion(samples[..., 3:])
        covariance = torch.einsum('nki,nkj->nij', quaternions, quaternions)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        mean = eigenvectors[..., -1]
        # An ambiguous mean (e.g. equal orthogonal rotations) uses the nearest
        # source, avoiding an arbitrary eigenspace basis as an initialization.
        ambiguous = ((eigenvalues[:, -1] - eigenvalues[:, -2]) <=
                     torch.finfo(values.dtype).eps * k)
        mean = torch.where(ambiguous[:, None], quaternions[:, 0], mean)
        mean = torch.where(mean[:, :1] < 0, -mean, mean)
        angle = 2 * torch.atan2(mean[:, 1:].norm(dim=-1, keepdim=True), mean[:, :1])
        vector = 2 * mean[:, 1:] / torch.sinc(angle / (2 * torch.pi))
        inherited.append(torch.cat((samples[..., :3].mean(dim=1), vector), dim=-1))
    return torch.cat(inherited)


def _selection(layout, states):
    occupancy = _occupancy(layout)
    visited = [occupancy[0]]
    for level in range(1, 3):
        parent = layout['parents'][level - 1]
        visited.append(occupancy[level] & visited[level - 1][parent] &
                       (states[level - 1][parent] == SPLIT))
    owner = layout['index']
    active = torch.ones(owner.shape[1], device=owner.device, dtype=torch.bool)
    terminal = torch.full((owner.shape[1],), SPLIT, device=owner.device, dtype=torch.long)
    for level in range(3):
        row_state = states[level][owner[level]]
        terminal = torch.where(active & (row_state != SPLIT), row_state, terminal)
        active = active & (row_state == SPLIT)
    return visited, active, terminal


def fixed_budget_states(layout, level_counts, seed):
    """Content-blind cut with exact reachable KEEP/SHARE/SPLIT counts.

    Uses a private CPU RNG; never consumes training RNG or examines images,
    coordinates, losses or learned node identities. Counts are [3 levels, 3
    actions]. An infeasible budget raises instead of silently changing rate.
    """
    if len(level_counts) != 3:
        raise ValueError('Fixed budget needs three levels')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    occupancy = _occupancy(layout)
    states = []
    visited = occupancy[0]
    for level, counts in enumerate(level_counts):
        if (len(counts) != 3 or any(isinstance(n, bool) or not isinstance(n, int)
                                    or n < 0 for n in counts)):
            raise ValueError('Fixed budget counts must be three nonnegative integers')
        rows = torch.where(visited)[0]
        if sum(counts) != len(rows):
            raise ValueError(f'Infeasible fixed budget at level {level}: '
                             f'{sum(counts)} requested, {len(rows)} reachable')
        order = torch.randperm(len(rows), generator=generator).to(rows.device)
        labels = torch.repeat_interleave(torch.arange(3, device=rows.device),
                                        torch.tensor(counts, device=rows.device))
        state = torch.full_like(occupancy[level], KEEP, dtype=torch.long)
        state[rows[order]] = labels
        states.append(state)
        if level < 2:
            parent = layout['parents'][level]
            visited = occupancy[level + 1] & visited[parent] & (state[parent] == SPLIT)
    return states


@torch.no_grad()
def actual_motion_statistics(base_xyz, base_rotation, xyz, rotation, extent,
                             translation_thresholds, rotation_thresholds, terminal):
    """Old surviving GS pose changes, with sign-invariant rotation angles."""
    translation = (xyz - base_xyz).norm(dim=-1) / extent
    old = F.normalize(base_rotation, dim=-1)
    new = F.normalize(rotation, dim=-1)
    new = torch.where((old * new).sum(-1, keepdim=True) < 0, -new, new)
    # Chord atan2 avoids acos rounding tiny rotations to zero.
    angle = 4 * torch.atan2((new - old).norm(dim=-1), (new + old).norm(dim=-1))
    keep = terminal == KEEP
    xyz_changed = (xyz != base_xyz).any(-1)
    rotation_changed = (new != old).any(-1)

    def distribution(values, thresholds):
        quantiles = torch.quantile(values.float(), values.new_tensor([0., .5, .9, .99, 1.]))
        return dict(quantiles=dict(zip(('min', 'p50', 'p90', 'p99', 'max'),
                                      quantiles.cpu().tolist())),
                    above_threshold=[dict(threshold=float(t), count=int((values > t).sum()),
                                          fraction=float((values > t).float().mean()))
                                     for t in thresholds])

    return dict(old_surviving_count=len(xyz), keep_count=int(keep.sum()),
                position_changed_count=int(xyz_changed.sum()),
                rotation_changed_count=int(rotation_changed.sum()),
                any_pose_changed_count=int((xyz_changed | rotation_changed).sum()),
                keep_position_changed_count=int((keep & xyz_changed).sum()),
                keep_rotation_changed_count=int((keep & rotation_changed).sum()),
                translation_extent=distribution(translation, translation_thresholds),
                rotation_radians=distribution(angle, rotation_thresholds),
                semantics='Previous surviving GS only; excludes births/deletions. '
                          'KEEP is a pose decision, not a background label. '
                          'Exact change counts compare stored positions and normalized '
                          'sign-aligned orientations; thresholds measure motion magnitude.')


def _motion(base_xyz, base_rotation, layout, regions, local, states, *, soft_probs=None,
            probe_keep_states=None):
    """Rigid centers and pivots travel together down each SPLIT path."""
    if probe_keep_states is not None and soft_probs is not None:
        raise ValueError('KEEP probe isolation must not alter the gate proxy')
    index, pivots = layout['index'], layout['pivots']
    n = len(base_xyz)
    identity = base_xyz.new_zeros(n, 4)
    identity[:, 0] = 1
    initial_pivots = [pivots[level][index[level]] for level in range(3)]

    def descend(level, xyz, orientation, current_pivots):
        if level == 3:
            moved = xyz + local[:, :3]
            rotated = quaternion_multiply(rotation_vector_quaternion(local[:, 3:]), orientation)
            return moved, rotated
        if probe_keep_states is not None and level > 0:
            # Only this KEEP terminal's SHARE candidate is isolated. Its own
            # motion stays differentiable; ancestor inputs keep their values
            # but receive no gradient from this candidate. The mask comes from
            # hard gate choices, not scene labels or a foreground oracle.
            kept_here = (probe_keep_states[level][index[level]] == KEEP)[:, None]
            xyz = torch.where(kept_here, xyz.detach(), xyz)
            orientation = torch.where(kept_here, orientation.detach(), orientation)
            current_pivots = [torch.where(kept_here, value.detach(), value)
                              for value in current_pivots]
        selected = regions[level][index[level]]
        moved, rotated = apply_region_rigid(
            xyz, orientation, current_pivots[level], selected[:, :3], selected[:, 3:])
        if level < 2:
            next_pivots = list(current_pivots)
            for child in range(level + 1, 3):
                next_pivots[child], _ = apply_region_rigid(
                    current_pivots[child], identity, current_pivots[level],
                    selected[:, :3], selected[:, 3:])
        else:
            next_pivots = current_pivots
        child_xyz, child_rotation = descend(level + 1, moved, rotated, next_pivots)
        if soft_probs is None:
            state = states[level][index[level]]
            is_keep = state == KEEP
            is_split = state == SPLIT
            xyz_result = torch.where(is_split[:, None], child_xyz, moved)
            xyz_result = torch.where(is_keep[:, None], base_xyz, xyz_result)
            rot_result = torch.where(is_split[:, None], child_rotation, rotated)
            rot_result = torch.where(is_keep[:, None], base_rotation, rot_result)
            return xyz_result, rot_result
        probabilities = soft_probs[level][index[level]]
        xyz_result = (probabilities[:, 0:1] * base_xyz +
                      probabilities[:, 1:2] * moved +
                      probabilities[:, 2:3] * child_xyz)
        # All candidates share the previous orientation's quaternion hemisphere.
        candidates = [base_rotation, rotated, child_rotation]
        aligned = [torch.where((value * base_rotation).sum(-1, keepdim=True) < 0,
                               -value, value) for value in candidates]
        rot_result = F.normalize(sum(probabilities[:, i:i + 1] * aligned[i]
                                     for i in range(3)), dim=-1)
        return xyz_result, rot_result

    return descend(0, base_xyz, base_rotation, initial_pivots)


class GatedResidualMotion(QuantizedResidualMotion):
    layout_version = LAYOUT_VERSION

    def __init__(self, config):
        if not config.enable_learned_gates:
            raise ValueError('GatedResidualMotion requires enable_learned_gates')
        super().__init__(config)
        self.active_layout = None
        self.previous_layout = None
        self.materialized_layout = None
        self.frozen_states = None
        self.tree_setup = None

    def _initialize_parameters(self, xyz, rotation, extent, layout):
        self.extent = float(extent)
        self.active_layout = layout
        self.index, self.counts, self.parents = (layout['index'], layout['counts'], layout['parents'])
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
        init = xyz.new_tensor(self.config.gate_initial_logits)
        self.gate_logits = nn.ParameterList(
            [nn.Parameter(init.expand(count, 3).clone()) for count in self.counts])
        self.optimizer = torch.optim.Adam([
            dict(params=self.region_translation, lr=self.config.region_translation_lr),
            dict(params=self.region_rotation, lr=self.config.region_rotation_lr),
            dict(params=[self.gs_translation], lr=self.config.gs_translation_lr),
            dict(params=[self.gs_rotation], lr=self.config.gs_rotation_lr),
            dict(params=self.gate_logits, lr=self.config.gate_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)

    def setup(self, xyz, rotation, extent, warm_start=True,
              previous_rows=None, previous_new_count=None, frame_index=None):
        interval = self.config.tree_rebuild_interval
        if interval and (isinstance(frame_index, bool) or
                         not isinstance(frame_index, int) or frame_index < 1):
            raise ValueError('Periodic tree rebuild requires a positive frame_index')
        old_layout = self.materialized_layout if self.materialized_layout is not None else None
        rebuild = old_layout is not None and interval > 0 and frame_index % interval == 0
        if xyz.is_cuda:
            torch.cuda.synchronize(xyz.device)
        tree_started = time.perf_counter()
        old_symbols = None
        old_states = None
        old_logits = None
        old_steps = None
        old_count = 0
        if old_layout is not None:
            self.previous_layout = _layout_cpu(old_layout)
            old_count = len(self.base_xyz)
            if previous_rows is None:
                previous_rows = torch.arange(len(xyz), device=xyz.device)
                previous_new_count = 0
            if (isinstance(previous_new_count, bool) or
                    not isinstance(previous_new_count, int) or
                    previous_new_count < 0):
                raise ValueError('Persistent warm start requires previous_new_count')
            rows = torch.as_tensor(previous_rows, device=xyz.device)
            if rows.dtype not in (torch.int32, torch.int64):
                raise ValueError('Persistent row mapping requires integer indices')
            rows = rows.long()
            if (rows.ndim != 1 or len(rows) != len(xyz) or
                    bool(((rows < 0) | (rows >= len(old_layout['index'][0]))).any()) or
                    bool((rows[1:] <= rows[:-1]).any())):
                raise ValueError('Persistent row mapping must be ordered and in range')
            if len(old_layout['index'][0]) != old_count + previous_new_count:
                raise ValueError('Persistent reference and newborn count mismatch')
            selected_index = old_layout['index'].to(xyz.device)[:, rows]
            layout = _layout(selected_index, old_layout['counts'], old_layout['parents'],
                             xyz, old_layout['pivots'])
            if warm_start:
                if not self.committed:
                    raise ValueError('Motion warm start requires committed previous frame')
                old_symbols = self._dense_symbols(integer=True)
                old_steps = self.motion_steps_vector.detach().clone()
                old_states = self.gate_state_arrays()
                old_logits = [value.detach().clone() for value in self.gate_logits]
                if not all(bool(torch.isfinite(value).all()) for value in old_logits):
                    raise ValueError('Previous gated logits must be finite')
                old_local_rows = rows < old_count
                with torch.no_grad():
                    previous_xyz, previous_rotation = _motion(
                        self.base_xyz, self.base_rotation, self.active_layout,
                        [old_symbols[f'region_{level}'].to(self.base_xyz) * old_steps
                         for level in range(3)],
                        old_symbols['gs_local'].to(self.base_xyz) * old_steps,
                        old_states['states'])
                    expected_rows = rows[old_local_rows]
                    aligned = torch.abs((F.normalize(previous_rotation[expected_rows], dim=-1) *
                                         F.normalize(rotation[old_local_rows], dim=-1)).sum(-1))
                    if (not torch.allclose(previous_xyz[expected_rows], xyz[old_local_rows],
                                           rtol=1e-5, atol=1e-6) or
                            not torch.allclose(aligned, torch.ones_like(aligned),
                                               rtol=1e-5, atol=1e-6)):
                        raise ValueError('Persistent row mapping does not match committed motion pose')
        else:
            self.previous_layout = None
            index, counts, parents = stable_layout(xyz, self.config.fine_region_gs)
            layout = _layout(index, counts, parents, xyz)
        if rebuild:
            previous_surviving_layout = layout
            index, counts, parents = stable_layout(xyz, self.config.fine_region_gs)
            layout = _layout(index, counts, parents, xyz)
        if xyz.is_cuda:
            torch.cuda.synchronize(xyz.device)
        tree_seconds = time.perf_counter() - tree_started
        initialization_started = time.perf_counter()
        self._initialize_parameters(xyz, rotation, extent, layout)
        self.frozen_states = None
        if old_symbols is not None:
            decay = self.config.motion_init_decay
            with torch.no_grad():
                for level in range(3):
                    selected = old_states['visited'][level] & (
                        old_states['states'][level] != KEEP)
                    values = old_symbols[f'region_{level}'].to(xyz) * old_steps.to(xyz)
                    if rebuild:
                        # Closed candidates were not applied last frame. They
                        # contribute identity motion, not speculative parameters.
                        values = torch.where(selected[:, None], values, torch.zeros_like(values))
                        logits = torch.where(old_states['visited'][level][:, None],
                            old_logits[level].to(xyz), xyz.new_tensor(self.config.gate_initial_logits))
                        occupied = _occupancy(previous_surviving_layout)[level]
                        values = _inherit_region_priors(
                            previous_surviving_layout['pivots'][level][occupied],
                            layout['pivots'][level], values[occupied],
                            self.config.tree_rebuild_neighbors)
                        logits = _inherit_gate_logits(
                            previous_surviving_layout['index'][level], layout['index'][level],
                            logits, self.counts[level], self.config.gate_initial_logits)
                        self.region_translation[level].copy_(values[:, :3] * decay)
                        self.region_rotation[level].copy_(values[:, 3:] * decay)
                        self.gate_logits[level].copy_(logits)
                    else:
                        self.region_translation[level][selected] = values[selected, :3] * decay
                        self.region_rotation[level][selected] = values[selected, 3:] * decay
                        self.gate_logits[level].copy_(old_logits[level].to(xyz))
                old_rows = rows[old_local_rows]
                selected = old_states['gs_leaf_open'][old_rows]
                local = old_symbols['gs_local'][old_rows].to(xyz) * old_steps.to(xyz)
                surviving_indices = torch.where(old_local_rows)[0]
                self.gs_translation[surviving_indices[selected]] = local[selected, :3] * decay
                self.gs_rotation[surviving_indices[selected]] = local[selected, 3:] * decay
        reset_counts = [0, 0, 0]
        self.reset_keep_masks = None
        if self.config.reset_keep_gate_priors:
            # An inherited KEEP + zero candidate + hard-path Jacobian cannot
            # relearn motion from image gradients. Reset only this gate prior;
            # motion warm starts, layout, optimizer and step budget stay intact.
            with torch.no_grad():
                self.reset_keep_masks = []
                for level, logits in enumerate(self.gate_logits):
                    keep = logits.argmax(-1) == KEEP
                    self.reset_keep_masks.append(keep.clone())
                    reset_counts[level] = int(keep.sum())
                    logits[keep] = xyz.new_tensor(self.config.gate_initial_logits)
        if self.config.gate_policy == 'fixed_budget_random':
            if not isinstance(frame_index, int) or frame_index < 1:
                raise ValueError('Fixed budget policy requires positive frame_index')
            manifest = json.loads(Path(self.config.fixed_gate_budget_path).read_text())
            counts = manifest['frames'][str(frame_index)]['level_counts']
            self.frozen_states = fixed_budget_states(
                layout, counts, self.config.fixed_gate_seed + frame_index)
            for parameter in self.gate_logits:
                parameter.requires_grad_(False)
        self.committed = False
        self.last_step = 0
        self.last_metrics = {}
        self.stage_seconds = {}
        if xyz.is_cuda:
            torch.cuda.synchronize(xyz.device)
        initialization_seconds = time.perf_counter() - initialization_started
        self.tree_setup = dict(
            frame=frame_index, interval=interval,
            action='rebuild' if rebuild else ('initial' if old_layout is None else 'reuse'),
            initialization=(('motion_knn_gate_membership' if rebuild else 'inherit')
                            if old_symbols is not None else 'reset'),
            neighbors=self.config.tree_rebuild_neighbors,
            initialization_decay=self.config.motion_init_decay if old_symbols is not None else 0.,
            reset_keep_gate_priors=self.config.reset_keep_gate_priors,
            reset_keep_gate_nodes=reset_counts,
            old_region_counts=list(old_layout['counts']) if old_layout is not None else None,
            region_counts=list(self.counts),
            topology_seconds=tree_seconds, initialization_seconds=initialization_seconds,
            semantics='Frame 1 initializes; existing trees rebuild at positive frame multiples of interval; '
                      '0 disables rebuild. Deletion protection is applied before setup. '
                      'Rebuild inherits same-level nearest occupied region priors: mean translation, '
                      'Markley mean rotation converted to rotation vector, then motion_init_decay; '
                      'Gate logits follow same-level surviving GS membership: strict-majority source, '
                      'otherwise member-weighted agreeing states, otherwise configured initial logits; '
                      'unvisited sources use configured initial logits. Gates remain trainable. '
                      'Closed sources contribute identity motion. GS residuals retain row identity. '
                      'Base poses are unchanged; inherited motion is a next-frame prediction. '
                      'Topology time includes row/pose validation and pivot construction. '
                      'Initialization includes parameter/optimizer creation and optional prior inheritance.')
        self.stage_seconds['tree_topology'] = tree_seconds
        self.stage_seconds['tree_initialization'] = initialization_seconds
        self.stage_seconds['tree_rebuild'] = (tree_seconds + initialization_seconds if rebuild else 0.)
        self.initial_gate_states = [value.clone() for value in self.gate_state_arrays()['states']]

    def install_materialized_layout(self, layout):
        self.materialized_layout = _layout_cpu(layout)


    def register_materialized_layout(self, materialized_xyz, birth_sources):
        """Extend persistent owners from the materialized surviving source rows."""
        if self.active_layout is None:
            raise ValueError('Materialized layout requires an active motion layout')
        source_raw = torch.as_tensor(birth_sources, device=self.index.device)
        if source_raw.ndim != 1 or source_raw.dtype not in (torch.int32, torch.int64):
            raise ValueError('Materialized birth sources must be one-dimensional integers')
        sources = source_raw.long()
        base_count = self.index.shape[1]
        if (len(materialized_xyz) != base_count + len(sources) or
                (len(sources) and bool(((sources < 0) | (sources >= base_count)).any()))):
            raise ValueError('Materialized birth sources do not match materialized raw')
        full_index = torch.cat((self.index, self.index[:, sources]), dim=1)
        full_layout = _layout(full_index, self.counts, self.parents,
                              materialized_xyz, self.active_layout['pivots'])
        self.install_materialized_layout(full_layout)
        return self.materialized_layout


    def _dense_symbols(self, integer=False):
        quantize = integer_symbols if integer else quantized_symbols
        steps = self.motion_steps_vector
        result = {f'region_{level}': quantize(
            torch.cat((self.region_translation[level], self.region_rotation[level]), -1),
            steps) for level in range(3)}
        result['gs_local'] = quantize(torch.cat((self.gs_translation, self.gs_rotation), -1), steps)
        return result

    def gate_state_arrays(self):
        if self.config.gate_policy == 'fixed_coarse':
            states = [torch.full((count,), SHARE, device=self.base_xyz.device, dtype=torch.long)
                      for count in self.counts]
        else:
            states = (self.frozen_states if self.frozen_states is not None else
                      [value.detach().argmax(-1) for value in self.gate_logits])
        visited, leaf_open, terminal = _selection(self.active_layout, states)
        return dict(states=states, visited=visited,
                    gs_leaf_open=leaf_open, gs_terminal=terminal)

    @torch.no_grad()
    def compensation_deletions(self, opacity, previous_new_count):
        """Protect the previous frame's terminal KEEP old rows before refitting.

        Newborn rows were not assigned a motion gate in their birth frame and
        remain eligible. Thus the requested compensation budget can be met
        without deleting a protected old row, even when every old row is KEEP.
        """
        if not self.committed:
            raise ValueError('KEEP deletion protection requires committed gates')
        if (isinstance(previous_new_count, bool) or
                not isinstance(previous_new_count, int) or previous_new_count < 0):
            raise ValueError('previous_new_count must be a nonnegative integer')
        opacity = opacity.detach().flatten()
        old_count = len(self.base_xyz)
        if len(opacity) != old_count + previous_new_count:
            raise ValueError('Deletion candidates disagree with old/new row counts')
        protected = torch.zeros_like(opacity, dtype=torch.bool)
        protected[:old_count] = self.gate_state_arrays()['gs_terminal'].to(opacity.device) == KEEP
        candidates = torch.where(~protected)[0]
        order = torch.argsort(opacity[candidates], stable=True)
        deleted = candidates[order[:previous_new_count]]
        return deleted, dict(
            enabled=True, protected_keep_old_count=int(protected.sum()),
            eligible_count=len(candidates), requested_count=previous_new_count,
            deleted_count=len(deleted), deleted_keep_count=int(protected[deleted].sum()),
            semantics='Previous-frame terminal KEEP old rows are protected; '
                      'newborn rows have no motion KEEP decision yet. '
                      'Eligible rows are ranked by activated opacity with stable ties.')

    def _temperature(self, step):
        joint = max(self.config.motion_steps - self.config.gate_refine_steps, 1)
        fraction = min(max((step - 1) / max(joint - 1, 1), 0.), 1.)
        return (self.config.gate_temperature_start * (1 - fraction) +
                self.config.gate_temperature_end * fraction)

    def _probabilities(self, step):
        if self.config.gate_policy == 'fixed_coarse':
            states = self.gate_state_arrays()['states']
            return [F.one_hot(state, 3).to(self.base_xyz) for state in states]
        if self.frozen_states is not None:
            return [F.one_hot(state, 3).to(self.base_xyz) for state in self.frozen_states]
        return [F.softmax(value / self._temperature(step), dim=-1)
                for value in self.gate_logits]


    def forward(self, step, training=False):
        if training:
            self.last_step = step
        if (self.config.gate_policy == 'learned' and self.frozen_states is None and
                step > self.config.motion_steps - self.config.gate_refine_steps):
            self.frozen_states = [value.detach().argmax(-1) for value in self.gate_logits]
            for value in self.gate_logits:
                value.requires_grad_(False)
        selection = self.gate_state_arrays()
        dense = self._dense_symbols()
        steps = self.motion_steps_vector
        regions = [dense[f'region_{level}'] * steps for level in range(3)]
        local = dense['gs_local'] * steps
        if (self.config.gate_policy == 'fixed_coarse' or self.frozen_states is not None or
                not torch.is_grad_enabled()):
            return _motion(self.base_xyz, self.base_rotation, self.active_layout,
                           regions, local, selection['states'])
        with torch.no_grad():
            hard_xyz, hard_rotation = _motion(
                self.base_xyz, self.base_rotation, self.active_layout,
                regions, local, selection['states'])
        # Gate logits compare detached candidates. Their image gradient and the
        # expected reachable storage gradient still describe the tree choices.
        gate_xyz, gate_rotation = _motion(
            self.base_xyz, self.base_rotation, self.active_layout,
            [value.detach() for value in regions], local.detach(), selection['states'],
            soft_probs=self._probabilities(step))
        # All modes keep the hard rendered pose and the same gate proxy above.
        # all_split uses the full candidate-chain Jacobian, including closed paths.
        # hard_path differentiates only the selected hard path; it never multiplies
        # motion gradients by soft reach probabilities. Shared gradients sum GS.
        gradient_mode = self.config.motion_gradient_mode
        if (self.config.motion_gradient_warmup_steps > 0 and
                step <= self.config.motion_gradient_warmup_steps):
            gradient_mode = 'all_split'
        if gradient_mode in {'keep_probe', 'keep_local_probe'}:
            # Fit a shared candidate at a reachable KEEP terminal without
            # forgetting its learned gate prior or changing the rendered pose.
            # KEEP and SHARE both stop descent, so deeper unvisited paths stay
            # closed. The next gate proxy can compare this fitted candidate.
            motion_states = [torch.where(state == KEEP, SHARE, state)
                             for state in selection['states']]
        elif gradient_mode in {'hard_path', 'keep_candidate_loss'}:
            motion_states = selection['states']
        else:
            motion_states = [torch.full_like(state, SPLIT) for state in selection['states']]
        candidate_xyz, candidate_rotation = _motion(
            self.base_xyz, self.base_rotation, self.active_layout,
            regions, local, motion_states,
            probe_keep_states=(selection['states']
                               if gradient_mode == 'keep_local_probe' else None))
        return (hard_xyz.detach() + (gate_xyz - gate_xyz.detach()) +
                (candidate_xyz - candidate_xyz.detach()),
                hard_rotation.detach() + (gate_rotation - gate_rotation.detach()) +
                (candidate_rotation - candidate_rotation.detach()))

    def keep_share_candidate_pose(self):
        """Fit disjoint closed candidates in one auxiliary pose/render.

        All reachable KEEP terminals are disjoint by tree construction. Their
        SHARE candidates are rendered together; already active motion, local
        residuals, SPLIT ancestors and gate probabilities remain detached.
        Optionally fit local SPLIT alternatives at reachable fine SHARE nodes;
        they are disjoint from all KEEP terminals. The hard pose is untouched.
        """
        selection = self.gate_state_arrays()
        masks = [visited & (state == KEEP) for visited, state in
                 zip(selection['visited'], selection['states'])]
        fine_mask = selection['visited'][2] & (selection['states'][2] == SHARE)
        if not self.config.fit_fine_split_candidates:
            fine_mask = torch.zeros_like(fine_mask)
        count = sum(int(mask.sum()) for mask in masks) + int(fine_mask.sum())
        if not count:
            return None
        dense = self._dense_symbols()
        steps = self.motion_steps_vector
        regions = []
        for level, mask in enumerate(masks):
            value = dense[f'region_{level}'] * steps
            regions.append(torch.where(mask[:, None], value, value.detach()))
        states = [torch.where(state == KEEP, SHARE, state) for state in selection['states']]
        states[2] = torch.where(fine_mask, SPLIT, states[2])
        local = dense['gs_local'] * steps
        if bool(fine_mask.any()):
            local_mask = fine_mask[self.active_layout['index'][2]]
            local = torch.where(local_mask[:, None], local, local.detach())
        else:
            local = local.detach()
        xyz, rotation = _motion(self.base_xyz, self.base_rotation, self.active_layout,
                                regions, local, states)
        return xyz, rotation, count

    @torch.no_grad()
    def final_quantized_total_motion(self):
        """Effective per-GS world displacement and relative orientation."""
        xyz, rotation = self.forward(self.config.motion_steps)
        relative = quaternion_multiply(rotation,
                                        inverse_unit_quaternion(self.base_rotation))
        return xyz - self.base_xyz, F.normalize(relative, dim=-1)

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def stage(self, step):
        return ('gate_refine' if step > self.config.motion_steps - self.config.gate_refine_steps
                else 'gate_joint')

    def _reach_probabilities(self, step):
        probabilities = self._probabilities(step)
        occupancy = _occupancy(self.active_layout)
        reach = [occupancy[0].to(self.base_xyz)]
        for level in range(1, 3):
            parent = self.parents[level - 1]
            reach.append(occupancy[level].to(self.base_xyz) * reach[level - 1][parent] *
                         probabilities[level - 1][parent, SPLIT])
        return probabilities, reach

    def gs_activity_fraction(self, step, hard=False):
        dense = self._dense_symbols()
        steps = self.motion_steps_vector
        support = candidate_activity_support(
            self.base_xyz, self.base_rotation, self.active_layout,
            [dense[f'region_{level}'] * steps for level in range(3)],
            dense['gs_local'] * steps,
            self.extent * self.config.motion_xyz_step_extent / 2,
            self.config.motion_rotation_step_rad / 2)
        probabilities = ([F.one_hot(state, 3).to(self.base_xyz)
                          for state in self.gate_state_arrays()['states']]
                         if hard else self._probabilities(step))
        return expected_activity_fraction(probabilities, self.index, support)

    def storage_parts(self, step, hard=False):
        """Bytes for reachable gate decisions and selected motion tensors."""
        zero = self.base_xyz.new_zeros(())
        gate_bytes = zero
        region_bytes = zero
        gs_width = (3 * self.gs_translation.element_size() +
                    3 * self.gs_rotation.element_size())
        if hard:
            selected = self.gate_state_arrays()
            for level in range(3):
                visited = selected['visited'][level]
                gate_bytes = gate_bytes + visited.sum().to(self.base_xyz)
                width = (3 * self.region_translation[level].element_size() +
                         3 * self.region_rotation[level].element_size())
                active = visited & (selected['states'][level] != KEEP)
                region_bytes = region_bytes + active.sum().to(self.base_xyz) * width
            gs_bytes = selected['gs_leaf_open'].sum().to(self.base_xyz) * gs_width
        else:
            probabilities, reach = self._reach_probabilities(step)
            for level in range(3):
                gate_bytes = gate_bytes + reach[level].sum()
                width = (3 * self.region_translation[level].element_size() +
                         3 * self.region_rotation[level].element_size())
                active_probability = (probabilities[level][:, SHARE] +
                                      probabilities[level][:, SPLIT])
                region_bytes = region_bytes + (reach[level] * active_probability).sum() * width
            leaf_probability = (reach[2][self.index[2]] *
                                probabilities[2][self.index[2], SPLIT])
            gs_bytes = leaf_probability.sum() * gs_width
        if self.config.gate_policy == 'fixed_coarse':
            # The fixed cut is a constant policy, so no per-node gate state is stored.
            gate_bytes = zero
        return dict(gate_bytes=gate_bytes, region_motion_bytes=region_bytes,
                    gs_motion_bytes=gs_bytes)

    def storage_scale(self, step):
        if not self.config.storage_only:
            raise ValueError('storage_scale requires storage_only configuration')
        return min(max((step - self.config.storage_warmup_steps) /
                       self.config.storage_ramp_steps, 0.), 1.)

    def reference_storage_bytes(self, first_frame_gs_count):
        if (isinstance(first_frame_gs_count, bool) or
                not isinstance(first_frame_gs_count, int) or
                first_frame_gs_count < 1):
            raise ValueError('first_frame_gs_count must be a positive integer')
        return first_frame_gs_count * (
            3 * self.gs_translation.element_size() +
            3 * self.gs_rotation.element_size())


    def statistics(self):
        selected = self.gate_state_arrays()
        levels = []
        for level in range(3):
            states, visited = selected['states'][level], selected['visited'][level]
            counts = {name: int(((states == number) & visited).sum())
                      for number, name in enumerate(('KEEP', 'SHARED', 'SPLIT'))}
            denominator = int(visited.sum())
            initial = getattr(self, 'initial_gate_states', None)
            transitions = (torch.bincount(
                initial[level][visited] * 3 + states[visited], minlength=9).reshape(3, 3)
                .cpu().tolist() if initial is not None else None)
            refreshed = getattr(self, 'reset_keep_masks', None)
            refreshed_counts = ([int(((states == action) & visited & refreshed[level]).sum())
                                 for action in range(3)] if refreshed is not None else None)
            levels.append(dict(level=level, denominator=denominator, counts=counts,
                               initial_to_final_actions=transitions,
                               refreshed_keep_final_counts=refreshed_counts,
                               fractions={key: value / denominator if denominator else 0.
                                          for key, value in counts.items()},
                               visited_count=denominator))
        terminal = selected['gs_terminal']
        gs_counts = {name: int((terminal == number).sum())
                     for number, name in enumerate(('KEEP', 'SHARED', 'SPLIT'))}
        n = len(terminal)
        return dict(region_counts=self.counts, levels=levels,
                    gs=dict(denominator=n, counts=gs_counts,
                            fractions={name: value / n for name, value in gs_counts.items()}),
                    gs_local_active_fraction=float(selected['gs_leaf_open'].float().mean()),
                    semantics='Hard gate decisions, not a measured nonzero-motion count. '
                              'Each level counts reachable occupied nodes. GS counts include old motion '
                              'rows only, excluding this frame newborns. KEEP is absolute previous pose.')

    def capture(self):
        packet = super().capture()
        if packet is not None:
            packet.update(layout_version=LAYOUT_VERSION,
                          active_layout=_layout_cpu(self.active_layout),
                          previous_layout=_layout_cpu(self.previous_layout),
                          materialized_layout=_layout_cpu(self.materialized_layout),
                          tree_setup=self.tree_setup,
                          frozen_states=([state.detach().cpu().clone() for state in self.frozen_states]
                                         if self.frozen_states is not None else None))
        return packet

    def restore(self, packet):
        if packet is None:
            return
        if packet.get('layout_version') != LAYOUT_VERSION:
            raise ValueError('Gated checkpoint version mismatch; start this storage/gradient '
                             'version from a shared first frame')
        saved_config = dict(packet['config'])
        saved_config.setdefault('motion_init_decay', 0.0)
        saved_config.setdefault('densification_stats_steps', None)
        saved_config.setdefault('max_new_gaussians', None)
        saved_config.setdefault('motion_gradient_warmup_steps', 0)
        try:
            saved = QuantizedResidualConfig(**saved_config)
        except TypeError as exc:
            raise ValueError('Legacy rate/codec motion checkpoints are incompatible; start from a shared first frame') from exc
        saved.validate()
        if asdict(saved) != asdict(self.config):
            raise ValueError('Gated residual resume requires identical settings')
        state = packet['state']
        xyz, rotation = state['base_xyz'], state['base_rotation']
        layout = packet['active_layout']
        layout = _layout(layout['index'].to(xyz.device), layout['counts'],
                         layout['parents'], xyz, layout['pivots'])
        self._initialize_parameters(xyz, rotation, packet['extent'], layout)
        self.load_state_dict(state)
        self.optimizer.load_state_dict(packet['optimizer'])
        self.previous_layout = _layout_cpu(packet['previous_layout'])
        self.materialized_layout = _layout_cpu(packet['materialized_layout'])
        self.frozen_states = ([value.to(xyz.device) for value in packet['frozen_states']]
                              if packet['frozen_states'] is not None else None)
        self.last_step, self.committed = packet['last_step'], packet['committed']
        if self.frozen_states is not None:
            for parameter in self.gate_logits:
                parameter.requires_grad_(False)
        if self.committed:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self.last_metrics = packet['last_metrics']
        self.stage_seconds = packet['stage_seconds']
        self.tree_setup = packet.get('tree_setup')
