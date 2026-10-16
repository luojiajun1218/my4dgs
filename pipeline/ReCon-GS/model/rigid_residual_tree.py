"""Shared motion optimization utilities and historical rigid-stream pose math.

The active trainer uses SpatialGatedMotion. Parent/pivot composition below is
retained only to decode historical packets; it is not a training method.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from .spatial_gate import SpatialGateConfig


@dataclass
class RigidTreeConfig:
    """Settings for spatial gate motion and quantized births."""
    region_count: int = 9
    motion_steps: int = 100
    birth_steps: int = 100
    motion_xyz_step_extent: float = 5e-6
    motion_rotation_step_rad: float = 5e-5
    gs_xyz_step_extent: float = 5e-6
    gs_rotation_step_rad: float = 5e-5
    region_translation_lr: float = 1e-3
    region_rotation_lr: float = 1e-3
    gs_translation_lr: float = 1e-3
    gs_rotation_lr: float = 1e-3
    motion_lr_final_ratio: float = 1.0
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    rate_loss_weight: float = 0.01
    rate_normalization: str = 'frame'
    relative_motion_loss_weight: float = 0.001
    relative_motion_units: str = 'extent_radians'
    rate_warmup_steps: int = 0
    rate_ramp_steps: int = 1
    motion_fit_window_steps: Optional[int] = None
    spatial_gate: SpatialGateConfig = field(default_factory=SpatialGateConfig)

    def motion_steps_for_frame(self, frame):
        return self.motion_steps

    def total_steps_for_frame(self, frame):
        return self.motion_steps + self.birth_steps

    def validate(self):
        self.spatial_gate.validate(self.motion_steps)
        if not self.spatial_gate.enabled or self.spatial_gate.learn_region_motion:
            raise ValueError('Use statistically initialized motion and learnable subdivision gates')
        for name in ('region_count', 'motion_steps', 'birth_steps', 'rate_ramp_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('rate_warmup_steps',):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.region_count != 9 or self.total_steps_for_frame(1) > 300:
            raise ValueError('Use eight initial regions and at most 300 steps per frame')
        if self.rate_warmup_steps >= self.motion_steps:
            raise ValueError('Rate warmup must fit the motion phase')
        if self.rate_loss_weight and self.spatial_gate.start_step <= self.rate_warmup_steps:
            raise ValueError('Fit complete GS motion proposals before region initialization')
        window = self.motion_fit_window_steps
        if window is not None and (isinstance(window, bool) or
                not isinstance(window, int) or not 1 <= window <= self.motion_steps):
            raise ValueError('Motion fit window must be a positive integer within the motion phase')
        if self.rate_normalization not in ('frame', 'reference'):
            raise ValueError('rate_normalization must be frame or reference')
        if self.relative_motion_units != 'extent_radians':
            raise ValueError('Active motion fitting uses extent-normalized translation and radians')
        for name in ('motion_xyz_step_extent', 'motion_rotation_step_rad',
                     'region_translation_lr', 'region_rotation_lr',
                     'gs_translation_lr', 'gs_rotation_lr', 'adam_eps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('rate_loss_weight', 'relative_motion_loss_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if (isinstance(self.motion_lr_final_ratio, bool) or
                not math.isfinite(self.motion_lr_final_ratio) or not 0 < self.motion_lr_final_ratio <= 1):
            raise ValueError('motion_lr_final_ratio must be in (0, 1]')
        if (self.gs_xyz_step_extent != self.motion_xyz_step_extent or
                self.gs_rotation_step_rad != self.motion_rotation_step_rad):
            raise ValueError('Spatial recoding requires the same region and GS quantization grid')
        if len(self.adam_betas) != 2 or any(not 0 <= value < 1 for value in self.adam_betas):
            raise ValueError('adam_betas must contain two values in [0, 1)')


# Older spatial checkpoints contain defaults for retired training paths.
# Strip only these known metadata fields when comparing/loading their config.
RETIRED_TREE_CONFIG_FIELDS = frozenset(('post_birth_motion_steps', 'fixed_support_refinement', 'tree_edit_interval', 'tree_edit_step', 'tree_edit_neighbors', 'tree_edit_region_trials', 'parent_max_depth', 'shared_steps', 'rate_support_temperature', 'adaptive_clustering', 'bootstrap_tree_step', 'motion_warm_start', 'causal_translation_init', 'persistent_shared_from_start', 'clustering', 'residual_hierarchy', 'tree_refine_steps', 'hdbscan', 'residual_clustering_method', 'gmm', 'shared_fit_min_reduction', 'heir', 'online_flat_clustering', 'online_partition_policy', 'post_partition_steps', 'online_clustering', 'persistent_regions', 'motion_tree', 'tree_child_relative_loss_weight', 'root_motion_discovery'))


def saved_config_with_default_motion_lr(saved_config):
    if not isinstance(saved_config, dict):
        return saved_config
    normalized = {**asdict(RigidTreeConfig()), **{
        name: value for name, value in saved_config.items()
        if name not in RETIRED_TREE_CONFIG_FIELDS}}
    normalized['spatial_gate'] = {**asdict(SpatialGateConfig()), **saved_config.get('spatial_gate', {})}
    normalized['spatial_gate']['image_fit_during_gate'] = saved_config.get('spatial_gate', {}).get('image_fit_during_gate', False)
    normalized['spatial_gate']['learn_region_motion'] = saved_config.get('spatial_gate', {}).get('learn_region_motion', True)
    normalized['relative_motion_units'] = saved_config.get('relative_motion_units', 'extent_radians')
    return normalized


def rotation_vector_quaternion(vector):
    """Exponential map with finite, nonzero derivative at the identity."""
    angle = vector.norm(dim=-1, keepdim=True)
    xyz = vector * (torch.sinc(angle / (2 * torch.pi)) / 2)
    return torch.cat((torch.cos(angle / 2), xyz), dim=-1)


def quaternion_multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), dim=-1)


def quaternion_inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quaternion_rotate(q, vector):
    return vector + rotation_delta(q, vector)


def rotation_delta(q, vector):
    t = 2 * torch.cross(q[..., 1:], vector, dim=-1)
    return q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def quaternion_angle_error(a, b):
    """Shortest orientation difference in radians, stable near identity.

    The tree edit tolerance is below float32 dot-product resolution. Promote
    before normalization so even a quaternion compared with itself passes.
    """
    a = F.normalize(a.double(), dim=-1)
    b = F.normalize(b.double(), dim=-1)
    relative = quaternion_multiply(a, quaternion_inverse(b))
    return 2 * torch.atan2(relative[..., 1:].norm(dim=-1),
                           relative[..., 0].abs())


def _identity(count, ref):
    q = ref.new_zeros((count, 4))
    q[:, 0] = 1
    return q


def _topological_order(parents):
    values = (parents.detach().cpu().tolist() if torch.is_tensor(parents)
              else list(parents))
    if not values or values[0] != -1:
        raise ValueError('Region zero must be the identity root')
    order = []
    status = [0] * len(values)
    status[0] = 2
    for start in range(1, len(values)):
        node, path = start, []
        while status[node] != 2:
            if status[node] == 1:
                raise ValueError('Region parent cycle')
            status[node] = 1
            path.append(node)
            parent = values[node]
            if parent < 0 or parent >= len(values) or parent == node:
                raise ValueError('Invalid region parent')
            node = parent
        for node in reversed(path):
            status[node] = 2
            order.append(node)
    return order


def _depth_levels(parents):
    values = (parents.detach().cpu().tolist() if torch.is_tensor(parents)
              else list(parents))
    order = _topological_order(parents)
    depth = [0] * len(parents)
    for node in order:
        depth[node] = depth[values[node]] + 1
    return [[node for node in order if depth[node] == level]
            for level in range(1, max(depth) + 1)]


def region_pivots(xyz, owner, parents):
    """Centroid of previous decoded subtree points, parent pivot if empty."""
    # Pivots are codec state. Compute them on CPU in a fixed row order so CUDA
    # scatter kernels cannot choose a different reduction order at export.
    output_device, output_dtype = xyz.device, xyz.dtype
    xyz = xyz.detach().cpu().contiguous()
    owner = owner.detach().cpu().long().contiguous()
    parents = parents.detach().cpu().long().contiguous()
    count = len(parents)
    order = _topological_order(parents)
    parent_values = parents.detach().cpu().tolist()
    sums = xyz.new_zeros(count, 3)
    counts = xyz.new_zeros(count, 1)
    sums.index_add_(0, owner, xyz)
    counts.index_add_(0, owner, xyz.new_ones(len(xyz), 1))
    for node in reversed(order):
        parent = parent_values[node]
        sums[parent] += sums[node]
        counts[parent] += counts[node]
    pivots = [None] * count
    pivots[0] = sums[0] / counts[0].clamp_min(1)
    for node in order:
        pivots[node] = (sums[node] / counts[node] if counts[node].item() > 0
                        else pivots[parent_values[node]])
    return torch.stack(pivots).to(device=output_device, dtype=output_dtype)


def _quantized(value, step):
    scaled = value / step
    hard = scaled.round().clamp(-32768, 32767) * step
    return hard.detach() + (value - value.detach())


def _integer(value, step):
    scaled = (value.detach() / step).round()
    if not bool(torch.isfinite(scaled).all()):
        raise ValueError('Motion symbols must be finite')
    return scaled.clamp(-32768, 32767).to(torch.int16)


def _varint_length(value):
    value = int(value)
    length = 1
    while value >= 128:
        value >>= 7
        length += 1
    return length


def _id_lengths(count, reference):
    index = torch.arange(count, device=reference.device)
    length = torch.ones(count, device=reference.device, dtype=reference.dtype)
    for threshold in (128, 16384, 2097152, 268435456):
        length = length + (index >= threshold).to(reference.dtype)
    return length


def compose_pose(base_xyz, base_rotation, owner, parents, pivots,
                 region_translation, region_rotvec, gs_translation, gs_rotvec):
    """Apply noncommuting parent-to-child SE(3), then GS pose residual."""
    q_global, b_global = region_transforms(base_xyz, parents, pivots,
                                           region_translation, region_rotvec)
    region_q = q_global[owner]
    xyz = quaternion_rotate(region_q, base_xyz) + b_global[owner] + gs_translation
    rotation = quaternion_multiply(
        rotation_vector_quaternion(gs_rotvec),
        quaternion_multiply(region_q, base_rotation))
    # The reference path is bit-exact, while the STE gradients above survive.
    static = ((owner == 0) & (gs_translation.detach() == 0).all(-1) &
              (gs_rotvec.detach() == 0).all(-1))
    xyz = xyz + (torch.where(static[:, None], base_xyz, xyz) - xyz).detach()
    rotation = rotation + (torch.where(static[:, None], base_rotation, rotation) - rotation).detach()
    return xyz, rotation


def region_transforms(base_xyz, parents, pivots, translation, rotvec):
    """Global affine transforms; child rotates around inherited world pivot."""
    count = len(parents)
    q_global = torch.cat((_identity(1, base_xyz), base_xyz.new_zeros(count - 1, 4)))
    b_global = base_xyz.new_zeros(count, 3)
    q_local = rotation_vector_quaternion(rotvec)
    for nodes in _depth_levels(parents):
        index = torch.as_tensor(nodes, dtype=torch.long, device=base_xyz.device)
        parent = parents[index].long()
        parent_q = q_global[parent]
        parent_b = b_global[parent]
        inherited_pivot = quaternion_rotate(parent_q, pivots[index])
        q = q_local[index]
        result_q = quaternion_multiply(q, parent_q)
        result_b = parent_b + rotation_delta(q, -inherited_pivot) + translation[index]
        q_global = q_global.index_copy(0, index, result_q)
        b_global = b_global.index_copy(0, index, result_b)
    return q_global, b_global


class MotionBase(nn.Module):
    """Optimizer, quantization, and checkpoint lifecycle shared by spatial motion."""
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.optimizer = None
        self.owner = self.parents = None
        self.codec_previous_state = self.frame_reference_state = None
        self.rate_reference_count = None
        self.pending_birth_sources = None
        self.topology_edits, self.last_metrics = [], {}
        self.committed, self.last_step = False, 0
        self.observations = None
        self.flat_region_diagnostics = {}

    @property
    def steps(self):
        """Regional translation/rotation steps (legacy public property)."""
        return self.base_xyz.new_tensor(
            [self.config.motion_xyz_step_extent * self.extent] * 3 +
            [self.config.motion_rotation_step_rad] * 3)


    @property
    def gs_steps(self):
        return self.base_xyz.new_tensor([self.config.gs_xyz_step_extent * self.extent] * 3 +
                                        [self.config.gs_rotation_step_rad] * 3)


    @property
    def motion_steps(self):
        return self.config.motion_steps_for_frame(self.frame_index)


    @property
    def rate_normalization_count(self):
        """Keep a byte's price fixed when newborn GS enlarge the scene."""
        if self.config.rate_normalization == 'reference':
            if self.rate_reference_count is None:
                raise ValueError('Set up the shared reference before pricing motion')
            return self.rate_reference_count
        return len(self.base_xyz)


    @property
    def rate_normalization_bytes(self):
        return max(1, self.rate_normalization_count * 24)


    def materialized_pose(self, step=None):
        return self.forward(self.motion_steps if step is None else step)


    def _motion_lr_factor(self, step):
        """Decay during the second half of motion fitting."""
        start = self.motion_steps // 2
        progress = (step - start) / max(1, self.motion_steps - start)
        progress = min(max(progress, 0.), 1.)
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return (self.config.motion_lr_final_ratio +
                (1 - self.config.motion_lr_final_ratio) * cosine)

    def _motion_fit_window_due(self):
        """Finish local fit windows, including an incomplete phase-end window."""
        window = self.config.motion_fit_window_steps
        if window is None:
            return True
        return self.last_step == self.motion_steps or self.last_step % window == 0


    @torch.no_grad()
    def _proximal_blocks(self, translation, rotation, step_t, step_r,
                         id_lengths, lr_t, lr_r, first_row):
        """Choose both, translation, rotation, or zero by Adam-metric L0 cost.

        The local quadratic uses Adam's bias-corrected second moment as its
        diagonal metric. Candidates pay actual sparse block and row-ID bytes,
        normalized by raw six-float motion bytes. A region starts at row one
        because the identity root never carries a stored motion block.
        """
        if self.config.rate_loss_weight == 0:
            return 0, 0
        ramp = min(max((self.last_step - self.config.rate_warmup_steps) /
                       self.config.rate_ramp_steps, 0.), 1.)
        if ramp == 0:
            return 0, 0
        state_t = self.optimizer.state.get(translation)
        state_r = self.optimizer.state.get(rotation)
        values_t, values_r = translation[first_row:], rotation[first_row:]
        beta2 = self.config.adam_betas[1]

        def zero_gain(parameter, state, lr):
            if not state or 'exp_avg_sq' not in state:
                return parameter.new_zeros(len(parameter))
            step = int(state['step'])
            variance = state['exp_avg_sq'][first_row:] / (1 - beta2 ** step)
            metric = (variance.sqrt() + self.config.adam_eps) / lr
            return 0.5 * (metric * parameter.square()).sum(-1)

        gain_t = zero_gain(values_t, state_t, lr_t)
        gain_r = zero_gain(values_r, state_r, lr_r)
        active_t = _integer(values_t, step_t).ne(0).any(-1)
        active_r = _integer(values_r, step_r).ne(0).any(-1)
        index_cost = id_lengths + 1
        price = self.config.rate_loss_weight * ramp / self.rate_normalization_bytes
        cost_both = price * ((active_t | active_r).to(index_cost) * index_cost +
                             6 * (active_t.to(index_cost) + active_r.to(index_cost)))
        cost_t = gain_r + price * active_t.to(index_cost) * (index_cost + 6)
        cost_r = gain_t + price * active_r.to(index_cost) * (index_cost + 6)
        cost_none = gain_t + gain_r
        choice = torch.stack((cost_both, cost_t, cost_r, cost_none)).argmin(0)
        prune_t = (choice == 2) | (choice == 3)
        prune_r = (choice == 1) | (choice == 3)
        values_t[prune_t] = 0
        values_r[prune_r] = 0
        return int(prune_t.sum()), int(prune_r.sum())


    @torch.no_grad()
    def _proximal_gs_blocks(self):
        groups = self.optimizer.param_groups
        pruned_t, pruned_r = self._proximal_blocks(
            self.gs_translation, self.gs_rotation,
            self.gs_steps[:3], self.gs_steps[3:], self._gs_id_lengths,
            groups[2]['lr'], groups[3]['lr'], 0)
        self.last_metrics['motion/prox_zero_translation_blocks'] = pruned_t
        self.last_metrics['motion/prox_zero_rotation_blocks'] = pruned_r


    def _hard_motion_bytes(self):
        rt, rr, gt, gr = self._values(hard=True)
        r_t = rt[1:].ne(0).any(-1)
        r_r = rr[1:].ne(0).any(-1)
        g_t = gt.ne(0).any(-1)
        g_r = gr.ne(0).any(-1)
        def sparse_bytes(active_t, active_r):
            rows = torch.where(active_t | active_r)[0]
            # Each sparse section restarts at -1: first gap=row+1, then
            # positive differences between consecutive active row IDs.
            gaps = torch.diff(torch.cat((rows.new_tensor([-1]), rows)))
            id_bytes = torch.ones_like(gaps)
            for shift in range(7, 63, 7):
                id_bytes += (gaps >= (1 << shift)).to(id_bytes)
            return (id_bytes.sum() + rows.numel() +
                    6 * (active_t.sum() + active_r.sum()) +
                    _varint_length(rows.numel()))
        return sparse_bytes(r_t, r_r) + sparse_bytes(g_t, g_r)


    def _make_optimizer(self):
        self.optimizer = torch.optim.Adam([
            dict(params=[self.region_translation], lr=self.config.region_translation_lr),
            dict(params=[self.region_rotation], lr=self.config.region_rotation_lr),
            dict(params=[self.gs_translation], lr=self.config.gs_translation_lr),
            dict(params=[self.gs_rotation], lr=self.config.gs_rotation_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)


    @torch.no_grad()
    def mark_observed(self, visible):
        if self.observations is not None:
            visible = torch.as_tensor(visible, device=self.observations.device, dtype=torch.bool)
            if visible.shape != self.observations.shape:
                raise ValueError('Motion visibility must match old GS rows')
            self.observations.add_(visible.long())


    @torch.no_grad()
    def register_materialized_layout(self, xyz, rotation, birth_source_rows=None):
        """Carry ownership into the next frame, including each new GS source."""
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rotation.shape != (len(xyz), 4):
            raise ValueError('Materialized pose shape mismatch')
        old_count = len(self.owner)
        if len(xyz) < old_count:
            raise ValueError('Materialized layout cannot discard motion survivors')
        sources = torch.as_tensor([] if birth_source_rows is None else birth_source_rows,
                                  device=self.owner.device, dtype=torch.long)
        if len(sources) != len(xyz) - old_count or bool(((sources < 0) | (sources >= old_count)).any()):
            raise ValueError('Birth sources must index previous surviving GS rows')
        expected_xyz, expected_q = self.materialized_pose()
        dot = (F.normalize(expected_q, dim=-1) *
               F.normalize(rotation[:old_count], dim=-1)).sum(-1).abs()
        if not torch.allclose(expected_xyz, xyz[:old_count], atol=1e-6, rtol=1e-5) or not bool((dot > 1 - 1e-5).all()):
            raise ValueError('Materialized old-GS pose differs from quantized renderer pose')
        self.pending_birth_sources = sources.detach().cpu().clone()
        self.materialized_xyz = xyz.detach().clone()
        self.materialized_rotation = rotation.detach().clone()
        self.committed = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)


    def capture(self):
        if self.owner is None:
            return None
        names = ('owner', 'parents', 'base_xyz', 'base_rotation',
                 'previous_rows', 'materialized_xyz', 'materialized_rotation',
                 'pending_birth_sources', 'observations')
        tensors = {name: getattr(self, name, None) for name in names}
        return dict(config=asdict(self.config), tensors=tensors,
                    rate_reference_count=self.rate_reference_count,
                    frame_index=self.frame_index, extent=self.extent,
                    parameters={name: parameter.detach().clone() for name, parameter in
                                (('region_translation', self.region_translation),
                                 ('region_rotation', self.region_rotation),
                                 ('gs_translation', self.gs_translation),
                                 ('gs_rotation', self.gs_rotation))},
                    optimizer=self.optimizer.state_dict(), committed=self.committed,
                    last_step=self.last_step, topology_edits=list(self.topology_edits),
                    last_metrics=deepcopy(self.last_metrics),
                    flat_region_diagnostics=deepcopy(self.flat_region_diagnostics),
                    codec_previous_state=(None if self.codec_previous_state is None else
                                          self.codec_previous_state.capture()),
                    frame_reference_state=(None if self.frame_reference_state is None else
                                           self.frame_reference_state.capture()))


    def restore(self, packet):
        if packet is None:
            return
        gate = packet['config'].get('spatial_gate', {})
        if not gate.get('enabled') or gate.get('learn_region_motion', True):
            raise ValueError('Retired motion checkpoints can be replayed, but cannot resume spatial training')
        if saved_config_with_default_motion_lr(packet['config']) != asdict(self.config):
            raise ValueError('Spatial motion checkpoint configuration differs')
        from .rigid_tree_codec import DecodedState
        state = packet['codec_previous_state']
        self.codec_previous_state = None if state is None else DecodedState.restore(state)
        reference = packet.get('frame_reference_state')
        tensors = packet['tensors']
        # Restore the saved variable-sized layout directly. Re-running setup
        # would spatially repartition it and lose appended/merged node rows.
        state_names = {'owner', 'parents', 'base_xyz', 'base_rotation', 'previous_rows',
                       'materialized_xyz', 'materialized_rotation', 'pending_birth_sources',
                       'observations', 'spatial_paths', 'opened', 'node_counts', 'split_eligible'}
        for name, value in tensors.items():
            if name in state_names:
                setattr(self, name, value)
        reference_count = packet.get('rate_reference_count')
        if reference_count is None and self.config.rate_normalization == 'frame':
            reference_count = len(self.base_xyz)
        if (isinstance(reference_count, bool) or not isinstance(reference_count, int) or
                reference_count <= 0):
            raise ValueError('Checkpoint needs a positive shared-reference GS count')
        self.rate_reference_count = reference_count
        self.extent = packet['extent']
        for name, value in packet['parameters'].items():
            setattr(self, name, nn.Parameter(value.detach().clone()))
        self._make_optimizer()
        self.optimizer.load_state_dict(packet['optimizer'])
        self.frame_index = packet['frame_index']
        self.committed = packet['committed']
        self.last_step = packet['last_step']
        self.topology_edits = list(packet['topology_edits'])
        self.last_metrics = deepcopy(packet.get('last_metrics', {}))
        self.flat_region_diagnostics = deepcopy(packet.get('flat_region_diagnostics', {}))
        if self.observations is None:
            self.observations = torch.zeros(len(self.base_xyz), dtype=torch.long, device=self.base_xyz.device)
        self._region_id_lengths = _id_lengths(len(self.parents) - 1, self.base_xyz)
        self._gs_id_lengths = _id_lengths(len(self.base_xyz), self.base_xyz)
        self.codec_previous_state = None if state is None else DecodedState.restore(state)
        self.frame_reference_state = (None if reference is None else
                                      DecodedState.restore(reference))
        if self.committed:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
