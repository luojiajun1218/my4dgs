"""Fixed-size rigid residual forest for motion-only dynamic Gaussians.

Quaternions are scalar-first. Region zero is the identity reference; a point
has one hard owner. Every region transform acts on the *previous decoded*
world-space pose, after its parent's transform. The GS residual acts last.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RigidTreeConfig:
    region_count: int = 32
    motion_steps: int = 100
    birth_steps: int = 100
    tree_edit_interval: int = 10
    tree_edit_step: int = 60
    tree_edit_neighbors: int = 8
    tree_edit_region_trials: int = 4
    parent_max_depth: int = 8
    shared_steps: int = 0
    motion_xyz_step_extent: float = 5e-6
    motion_rotation_step_rad: float = 5e-5
    gs_xyz_step_extent: float | None = None
    gs_rotation_step_rad: float | None = None
    region_translation_lr: float = 1e-3
    region_rotation_lr: float = 1e-3
    gs_translation_lr: float = 1e-3
    gs_rotation_lr: float = 1e-3
    motion_lr_final_ratio: float = 1.0
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    rate_loss_weight: float = 0.01
    relative_motion_loss_weight: float = 0.001
    rate_support_temperature: float = 1.0
    rate_warmup_steps: int = 0
    rate_ramp_steps: int = 1

    def validate(self):
        for name in ('region_count', 'motion_steps', 'birth_steps',
                     'tree_edit_interval', 'tree_edit_step',
                     'tree_edit_neighbors', 'tree_edit_region_trials',
                     'parent_max_depth'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.region_count < 2 or self.tree_edit_step > self.motion_steps:
            raise ValueError('Need at least two regions and an edit step within motion fitting')
        if (isinstance(self.shared_steps, bool) or not isinstance(self.shared_steps, int) or
                self.shared_steps < 0 or self.shared_steps >= self.motion_steps or
                self.tree_edit_step <= self.shared_steps):
            raise ValueError('shared_steps must leave joint fitting and precede tree edits')
        for name in ('motion_xyz_step_extent', 'motion_rotation_step_rad',
                     'region_translation_lr', 'region_rotation_lr',
                     'gs_translation_lr', 'gs_rotation_lr', 'adam_eps',
                     'rate_support_temperature'):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('rate_loss_weight', 'relative_motion_loss_weight'):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if (isinstance(self.motion_lr_final_ratio, bool) or
                not isinstance(self.motion_lr_final_ratio, (float, int)) or
                not math.isfinite(self.motion_lr_final_ratio) or
                not 0 < self.motion_lr_final_ratio <= 1):
            raise ValueError('motion_lr_final_ratio must be in (0, 1]')
        for name in ('gs_xyz_step_extent', 'gs_rotation_step_rad'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (float, int)) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive when set')
        if ((self.gs_xyz_step_extent is None) !=
                (self.gs_rotation_step_rad is None)):
            raise ValueError('Both GS quantization steps must be set together')
        if (self.rate_warmup_steps < 0 or self.rate_ramp_steps < 1 or
                self.rate_warmup_steps >= self.motion_steps):
            raise ValueError('Invalid rate warmup/ramp schedule')
        if len(self.adam_betas) != 2 or any(not 0 <= v < 1 for v in self.adam_betas):
            raise ValueError('adam_betas must contain two values in [0, 1)')


def saved_config_with_default_motion_lr(saved_config):
    """Compare pre-schedule checkpoints without changing their saved payload."""
    if isinstance(saved_config, dict) and 'motion_lr_final_ratio' not in saved_config:
        return {**saved_config, 'motion_lr_final_ratio': 1.0}
    return saved_config


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


def quaternion_rotvec(q):
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    norm = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, q[..., :1])
    return v * torch.where(norm > 0, angle / norm.clamp_min(1e-30),
                           torch.full_like(norm, 2))


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


def initial_layout(xyz, region_count):
    """Stable spatial bisection; keeps all allocated empty regions valid."""
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError('Initial layout needs a nonempty [N, 3] point cloud')
    if region_count < 2:
        raise ValueError('Region count must include root and one moving region')
    points = xyz.detach().cpu()
    owner = torch.ones(len(points), dtype=torch.long)
    parents = torch.zeros(region_count, dtype=torch.long)
    parents[0] = -1
    groups = {1: torch.arange(len(points), dtype=torch.long)}
    for child in range(2, min(region_count, len(points) + 1)):
        largest = max(groups, key=lambda node: (len(groups[node]), -node))
        rows = groups[largest]
        if len(rows) < 2:
            break
        axis = int((points[rows].amax(0) - points[rows].amin(0)).argmax())
        ordered = sorted(rows.tolist(), key=lambda row: (float(points[row, axis]), row))
        half = len(ordered) // 2
        moved = torch.tensor(ordered[half:], dtype=torch.long)
        kept = torch.tensor(ordered[:half], dtype=torch.long)
        owner[moved] = child
        parents[child] = 0
        groups[largest], groups[child] = kept, moved
    return owner.to(xyz.device), parents.to(xyz.device)


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


def relative_motion_magnitude(translation, rotvec, extent):
    """L1 residual SE(3) magnitude per region, with translation scale-free."""
    return (translation / extent).abs().sum(-1) + rotvec.abs().sum(-1)


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


class RigidResidualTree(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.optimizer = None
        self.owner = None
        self.parents = None
        self.pivots = None
        self.codec_previous_state = None
        self.frame_reference_state = None
        self.materialized_owner = None
        self.pending_birth_sources = None
        self.topology_edits = []
        self.last_metrics = {}
        self.stage_seconds = {}
        self.committed = False
        self.last_step = 0
        self._topology_updated = False
        self._metadata_cache = None

    @property
    def steps(self):
        """Regional translation/rotation steps (legacy public property)."""
        return self.base_xyz.new_tensor(
            [self.config.motion_xyz_step_extent * self.extent] * 3 +
            [self.config.motion_rotation_step_rad] * 3)

    @property
    def gs_steps(self):
        xyz_step = (self.config.gs_xyz_step_extent
                    if self.config.gs_xyz_step_extent is not None
                    else self.config.motion_xyz_step_extent)
        rotation_step = (self.config.gs_rotation_step_rad
                         if self.config.gs_rotation_step_rad is not None
                         else self.config.motion_rotation_step_rad)
        return self.base_xyz.new_tensor([xyz_step * self.extent] * 3 +
                                        [rotation_step] * 3)

    def setup(self, xyz, rotation, extent, frame_index=0, previous_rows=None):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rotation.shape != (len(xyz), 4):
            raise ValueError('Expected [N,3] centers and [N,4] raw rotations')
        if not len(xyz) or not math.isfinite(float(extent)) or extent <= 0:
            raise ValueError('Motion reference must be nonempty with positive extent')
        has_previous = self.materialized_owner is not None
        if not has_previous:
            owner, parents = initial_layout(xyz, self.config.region_count)
        else:
            if previous_rows is None:
                previous_rows = torch.arange(len(xyz), device=xyz.device)
            rows = torch.as_tensor(previous_rows, device=xyz.device, dtype=torch.long)
            if (rows.shape != (len(xyz),) or
                    bool(((rows < 0) | (rows >= len(self.materialized_owner))).any()) or
                    rows.unique().numel() != len(rows)):
                raise ValueError('Previous rows must map each surviving GS once')
            if self.codec_previous_state is not None:
                prior_xyz = self.codec_previous_state.xyz.to(xyz)
                prior_rotation = self.codec_previous_state.rotation.to(rotation)
                dot = (F.normalize(rotation, dim=-1) *
                       F.normalize(prior_rotation[rows], dim=-1)).sum(-1).abs()
                if (not torch.allclose(xyz, prior_xyz[rows], atol=1e-6, rtol=1e-5) or
                        not bool((dot > 1 - 1e-5).all())):
                    raise ValueError('Frame reference differs from previous decoded state')
            owner = self.materialized_owner.to(xyz.device)[rows].clone()
            parents = self.parents.clone()
        self.owner, self.parents = owner, parents
        self.pivots = region_pivots(xyz.detach(), owner, parents)
        self.base_xyz = xyz.detach().clone()
        self.base_rotation = rotation.detach().clone()
        self.extent = float(extent)
        self.frame_index = int(frame_index)
        self.previous_rows = (torch.arange(len(xyz), device=xyz.device)
                              if previous_rows is None else
                              torch.as_tensor(previous_rows, device=xyz.device, dtype=torch.long).clone())
        if self.codec_previous_state is not None and has_previous:
            # The stream, rather than a slightly different CUDA computation,
            # defines the reference used by every subsequent frame.
            self.base_xyz = self.codec_previous_state.xyz[self.previous_rows.cpu()].to(xyz)
            self.base_rotation = self.codec_previous_state.rotation[self.previous_rows.cpu()].to(rotation)
            self.pivots = region_pivots(self.base_xyz, owner, parents)
        if self.codec_previous_state is None:
            from .rigid_tree_codec import DecodedState
            self.codec_previous_state = DecodedState(
                self.base_xyz.detach().cpu(), self.base_rotation.detach().cpu(),
                owner.detach().cpu(), parents.detach().cpu(), self.frame_index - 1)
        self.frame_reference_state = self.codec_previous_state
        count = self.config.region_count
        self.region_translation = nn.Parameter(xyz.new_zeros(count, 3))
        self.region_rotation = nn.Parameter(xyz.new_zeros(count, 3))
        self.gs_translation = nn.Parameter(torch.zeros_like(xyz))
        self.gs_rotation = nn.Parameter(torch.zeros_like(xyz))
        self.optimizer = torch.optim.Adam([
            dict(params=[self.region_translation], lr=self.config.region_translation_lr),
            dict(params=[self.region_rotation], lr=self.config.region_rotation_lr),
            dict(params=[self.gs_translation], lr=self.config.gs_translation_lr),
            dict(params=[self.gs_rotation], lr=self.config.gs_rotation_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)
        self.materialized_owner = None
        self.pending_birth_sources = None
        self.topology_edits = []
        self.last_metrics = {}
        self.committed = False
        self.last_step = 0
        self._topology_updated = False
        self._metadata_cache = None
        self._region_id_lengths = _id_lengths(count - 1, xyz)
        self._gs_id_lengths = _id_lengths(len(xyz), xyz)

    def _values(self, hard=False, step=None):
        region_step, gs_step = self.steps, self.gs_steps
        quantize = _integer if hard else _quantized
        rt = quantize(self.region_translation, region_step[:3])
        rr = quantize(self.region_rotation, region_step[3:])
        gt = quantize(self.gs_translation, gs_step[:3])
        gr = quantize(self.gs_rotation, gs_step[3:])
        if step is not None and step <= self.config.shared_steps:
            gt, gr = torch.zeros_like(gt), torch.zeros_like(gr)
        # Root is always the identity and has no transmitted symbols.
        if hard:
            rt = rt.clone(); rr = rr.clone()
            rt[0] = 0; rr[0] = 0
        else:
            zero = torch.zeros_like(rt[:1])
            rt = torch.cat((zero, rt[1:]))
            rr = torch.cat((zero, rr[1:]))
        return rt, rr, gt, gr

    def forward(self, step, training=False):
        if training:
            self.last_step = int(step)
        rt, rr, gt, gr = self._values(step=step)
        return compose_pose(self.base_xyz, self.base_rotation, self.owner,
                            self.parents, self.pivots, rt, rr, gt, gr)

    def materialized_pose(self, step=None):
        return self.forward(self.config.motion_steps if step is None else step)

    @torch.no_grad()
    def optimizer_step(self):
        if self.last_step <= self.config.shared_steps:
            self.gs_translation.grad = None
            self.gs_rotation.grad = None
        factor = self._motion_lr_factor(self.last_step)
        base_lrs = (self.config.region_translation_lr,
                    self.config.region_rotation_lr,
                    self.config.gs_translation_lr,
                    self.config.gs_rotation_lr)
        for group, base_lr in zip(self.optimizer.param_groups, base_lrs):
            group['lr'] = base_lr * factor
        self.optimizer.step()
        projected = 0
        for parameter, scale in ((self.region_translation, self.steps[:3]),
                                 (self.region_rotation, self.steps[3:]),
                                 (self.gs_translation, self.gs_steps[:3]),
                                 (self.gs_rotation, self.gs_steps[3:])):
            limit = 32767 * scale
            projected += int((parameter.abs() > limit).any(-1).sum())
            parameter.clamp_(min=-limit, max=limit)
        self._proximal_region_blocks()
        if self.last_step > self.config.shared_steps:
            self._proximal_gs_blocks()
        self.last_metrics['motion/int16_projected_blocks'] = projected
        self.optimizer.zero_grad(set_to_none=True)

    def _motion_lr_factor(self, step):
        """One cosine decay over the second half of motion fitting."""
        start = self.config.motion_steps // 2
        progress = min(max((step - start) /
                           max(1, self.config.motion_steps - start), 0.), 1.)
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return (self.config.motion_lr_final_ratio +
                (1 - self.config.motion_lr_final_ratio) * cosine)

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
        price = self.config.rate_loss_weight * ramp / max(1, len(self.base_xyz) * 24)
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
    def _proximal_region_blocks(self):
        groups = self.optimizer.param_groups
        pruned_t, pruned_r = self._proximal_blocks(
            self.region_translation, self.region_rotation,
            self.steps[:3], self.steps[3:], self._region_id_lengths,
            groups[0]['lr'], groups[1]['lr'], 1)
        self.last_metrics['motion/prox_zero_region_translation_blocks'] = pruned_t
        self.last_metrics['motion/prox_zero_region_rotation_blocks'] = pruned_r

    @torch.no_grad()
    def _proximal_gs_blocks(self):
        groups = self.optimizer.param_groups
        pruned_t, pruned_r = self._proximal_blocks(
            self.gs_translation, self.gs_rotation,
            self.gs_steps[:3], self.gs_steps[3:], self._gs_id_lengths,
            groups[2]['lr'], groups[3]['lr'], 0)
        self.last_metrics['motion/prox_zero_translation_blocks'] = pruned_t
        self.last_metrics['motion/prox_zero_rotation_blocks'] = pruned_r

    def stage(self, step):
        return ('tree_edit' if (self.frame_index % self.config.tree_edit_interval == 0 and
                                step == self.config.tree_edit_step and
                                not self._topology_updated) else 'motion_fit')

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

    def _metadata_bytes(self):
        if self._metadata_cache is not None:
            return self._metadata_cache
        prior = self.frame_reference_state
        rows = self.previous_rows.detach().cpu().tolist()
        birth = (self.pending_birth_sources.tolist()
                 if self.pending_birth_sources is not None else [])
        quantization_bytes = (40 if self.config.gs_xyz_step_extent is not None
                              else 24)
        size = (4 + 32 + quantization_bytes + 16 +
                _varint_length(self.frame_index) +
                _varint_length(len(self.parents)) + _varint_length(len(rows)) +
                _varint_length(len(birth)) + 1)
        if len(rows) < len(prior.xyz):
            deleted = sorted(set(range(len(prior.xyz))) - set(rows))
            size += _varint_length(len(deleted))
            last = -1
            for row in deleted:
                size += _varint_length(row - last - 1)
                last = row
        parent_edits = torch.where(self.parents.detach().cpu() != prior.parents)[0].tolist()
        owner_edits = torch.where(self.owner.detach().cpu() != prior.owner[rows])[0].tolist()
        size += _varint_length(len(parent_edits)) + _varint_length(len(owner_edits))
        for node in parent_edits:
            size += _varint_length(node) + _varint_length(int(self.parents[node]))
        for row in owner_edits:
            size += _varint_length(row) + _varint_length(int(self.owner[row]))
        size += sum(_varint_length(source) + 28 for source in birth)
        self._metadata_cache = size
        return size

    def loss_terms(self, step, diagnostics=True):
        region_scale_t = self.steps[:3]
        region_scale_r = self.steps[3:]
        gs_scale_t = self.gs_steps[:3]
        gs_scale_r = self.gs_steps[3:]
        temperature = self.config.rate_support_temperature
        def support(value, scale):
            magnitude = (value / scale).abs().sum(-1)
            return magnitude / (magnitude + temperature)
        rt = support(self.region_translation[1:], region_scale_t)
        rr = support(self.region_rotation[1:], region_scale_r)
        gt = support(self.gs_translation, gs_scale_t)
        gr = support(self.gs_rotation, gs_scale_r)
        # Smooth/proximal and owner-edit decisions retain the absolute-ID
        # proxy. Only _hard_motion_bytes is the exact gap stream.
        rid = self._region_id_lengths
        gid = self._gs_id_lengths
        region_soft_bytes = (((rt + rr - rt * rr) * (rid + 1)).sum() +
                             (rt.sum() + rr.sum()) * 6)
        gs_soft_bytes = (((gt + gr - gt * gr) * (gid + 1)).sum() +
                         (gt.sum() + gr.sum()) * 6)
        # Both region and GS blocks use the discrete Adam-metric proximal
        # choice after the optimizer step. Their soft byte count is a
        # diagnostic only, detached to avoid charging either block twice.
        soft_bytes = (region_soft_bytes + gs_soft_bytes).detach()
        metadata = self._metadata_bytes()
        soft_bytes = soft_bytes + metadata
        raw_bytes = max(1, len(self.base_xyz) * 6 * 4)
        rate = soft_bytes / raw_bytes
        ramp = min(max((step - self.config.rate_warmup_steps) /
                       self.config.rate_ramp_steps, 0.), 1.)
        # Relative transform is the only regional smoothness prior. Distinct
        # branches can move differently without a same-world-motion penalty.
        relative = self._relative_motion()
        result = dict(rate_loss=rate * self.config.rate_loss_weight * ramp,
                      relative_motion_loss=relative * self.config.relative_motion_loss_weight,
                      rate_bytes_surrogate=soft_bytes,
                      rate_fraction=rate,
                      relative_motion=relative)
        if diagnostics:
            # This is the exact size of the uncompressed delta-row stream.
            # The actual stored size is known only after export_motion.
            result['rate_bytes_hard'] = self._hard_motion_bytes() + metadata
            result['rate_bytes_hard_uncompressed'] = result['rate_bytes_hard']
            result['int16_overflow_blocks'] = sum(
                (parameter.detach() / scale).abs().gt(32767.5).any(-1).sum()
                for parameter, scale in ((self.region_translation[1:], region_scale_t),
                                         (self.region_rotation[1:], region_scale_r),
                                         (self.gs_translation, gs_scale_t),
                                         (self.gs_rotation, gs_scale_r)))
            region_t, region_r, gs_t, gs_r = self._values(hard=True)
            region_t = region_t[1:].ne(0).any(-1)
            region_r = region_r[1:].ne(0).any(-1)
            gs_t = gs_t.ne(0).any(-1)
            gs_r = gs_r.ne(0).any(-1)
            result.update(region_translation_active_blocks=region_t.sum(),
                          region_rotation_active_blocks=region_r.sum(),
                          region_active_nodes=(region_t | region_r).sum(),
                          gs_translation_active_blocks=gs_t.sum(),
                          gs_rotation_active_blocks=gs_r.sum(),
                          gs_active_rows=(gs_t | gs_r).sum())
        return result

    @torch.no_grad()
    def _global_regions(self):
        rt, rr, _, _ = self._values()
        return region_transforms(self.base_xyz, self.parents, self.pivots, rt, rr)

    def _relative_motion(self):
        return relative_motion_magnitude(
            self.region_translation[1:], self.region_rotation[1:],
            self.extent).mean()

    @torch.no_grad()
    def _residual_for_target(self, target_xyz, target_rotation, candidate_owner,
                             q_global=None, b_global=None):
        if q_global is None:
            q_global, b_global = self._global_regions()
        q = q_global[candidate_owner]
        predicted_xyz = quaternion_rotate(q, self.base_xyz) + b_global[candidate_owner]
        predicted_rotation = F.normalize(quaternion_multiply(q, self.base_rotation), dim=-1)
        t = target_xyz - predicted_xyz
        r = quaternion_rotvec(quaternion_multiply(F.normalize(target_rotation, dim=-1),
                                                   quaternion_inverse(predicted_rotation)))
        ts = _integer(t, self.gs_steps[:3])
        rs = _integer(r, self.gs_steps[3:])
        return ts, rs

    @torch.no_grad()
    def update_topology(self, step):
        """Greedy sparse owner edits using full rigid-pose fit and edit bytes.

        Tree reparenting is considered after owner edits. Every proposal is
        checked against the current decoded target and a cycle-free hierarchy.
        """
        if self.stage(step) != 'tree_edit':
            return []
        self._topology_updated = True
        before_bytes = float(self._hard_motion_bytes()) + self._metadata_bytes()
        target_xyz, target_q = (value.detach() for value in self.forward(step))
        q_global, b_global = self._global_regions()
        _, _, current_t, current_r = self._values(hard=True)
        current_t_active = current_t.ne(0).any(-1)
        current_r_active = current_r.ne(0).any(-1)
        current_cost = (6 * (current_t_active.to(self.base_xyz) +
                             current_r_active.to(self.base_xyz)) +
                        (self._gs_id_lengths + 1) *
                        (current_t_active | current_r_active).to(self.base_xyz))
        count = len(self.base_xyz)
        edits = []
        nearest = min(self.config.tree_edit_neighbors, len(self.parents) - 1)
        for start in range(0, count, 4096):
            end = min(count, start + 4096)
            distance = torch.cdist(self.base_xyz[start:end], self.pivots[1:])
            near = distance.topk(nearest, largest=False).indices + 1
            # Do not materialize a full N x R pose tensor.
            best_owner = self.owner[start:end].clone()
            best_t, best_r = current_t[start:end].clone(), current_r[start:end].clone()
            best_cost = current_cost[start:end].clone()
            for col in range(nearest + 1):
                owner = (torch.zeros_like(best_owner) if col == 0 else near[:, col - 1])
                region_q = q_global[owner]
                predicted_xyz = quaternion_rotate(region_q, self.base_xyz[start:end]) + b_global[owner]
                predicted_q = F.normalize(quaternion_multiply(
                    region_q, self.base_rotation[start:end]), dim=-1)
                t = _integer(target_xyz[start:end] - predicted_xyz, self.gs_steps[:3])
                r = _integer(quaternion_rotvec(quaternion_multiply(
                    F.normalize(target_q[start:end], dim=-1), quaternion_inverse(predicted_q))), self.gs_steps[3:])
                overflow = (((target_xyz[start:end] - predicted_xyz) / self.gs_steps[:3]).abs() > 32767.5).any(-1)
                overflow |= (quaternion_rotvec(quaternion_multiply(
                    F.normalize(target_q[start:end], dim=-1), quaternion_inverse(predicted_q))) /
                             self.gs_steps[3:]).abs().gt(32767.5).any(-1)
                recovered_xyz = predicted_xyz + t.to(predicted_xyz) * self.gs_steps[:3]
                recovered_q = quaternion_multiply(
                    rotation_vector_quaternion(r.to(predicted_xyz) * self.gs_steps[3:]),
                    quaternion_multiply(region_q, self.base_rotation[start:end]))
                pose_ok = ((recovered_xyz - target_xyz[start:end]).norm(dim=-1) <=
                           math.sqrt(3) * float(self.gs_steps[0]) / 2 + 1e-6)
                angle_error = quaternion_angle_error(recovered_q, target_q[start:end])
                pose_ok &= angle_error <= math.sqrt(3) * float(self.gs_steps[3]) / 2
                active_t, active_r = t.ne(0).any(-1), r.ne(0).any(-1)
                row_id_bytes = self._gs_id_lengths[start:end]
                cost = (6 * (active_t.to(self.base_xyz) +
                             active_r.to(self.base_xyz)) +
                        (row_id_bytes + 1) * (active_t | active_r).to(self.base_xyz))
                # The edit stores an absolute GS row ID and new region ID.
                owner_id_bytes = torch.ones_like(row_id_bytes)
                for threshold in (128, 16384, 2097152):
                    owner_id_bytes += (owner >= threshold).to(owner_id_bytes)
                cost += (owner != self.owner[start:end]).to(self.base_xyz) * (
                    row_id_bytes + owner_id_bytes)
                better = (cost < best_cost) & ~overflow & pose_ok
                best_cost = torch.where(better, cost, best_cost)
                best_owner = torch.where(better, owner, best_owner)
                best_t = torch.where(better[:, None], t, best_t)
                best_r = torch.where(better[:, None], r, best_r)
            changed = best_owner != self.owner[start:end]
            rows = torch.arange(start, end, device=self.owner.device)[changed]
            for row in rows.tolist():
                edits.append(dict(kind='owner', row=row, owner=int(best_owner[row - start])))
            self.owner[start:end] = best_owner
            self.gs_translation[start:end] = best_t.to(self.gs_translation) * self.gs_steps[:3]
            self.gs_rotation[start:end] = best_r.to(self.gs_rotation) * self.gs_steps[3:]
        # Quantization after rerouting may differ by at most half a step.
        self.optimizer.state.pop(self.gs_translation, None)
        self.optimizer.state.pop(self.gs_rotation, None)
        edits.extend(self._try_reparent(target_xyz, target_q))
        self.topology_edits.extend(edits)
        self._metadata_cache = None
        after_bytes = float(self._hard_motion_bytes()) + self._metadata_bytes()
        final_xyz, final_q = self.forward(step)
        position_error = float((final_xyz - target_xyz).norm(dim=-1).amax())
        rotation_error = float(quaternion_angle_error(final_q, target_q).amax())
        self.last_metrics.update({
            'tree_edit/uncompressed_bytes_before': before_bytes,
            'tree_edit/uncompressed_bytes_after': after_bytes,
            'tree_edit/uncompressed_bytes_delta': after_bytes - before_bytes,
            'tree_edit/owner_edits': sum(edit['kind'] == 'owner' for edit in edits),
            'tree_edit/parent_edits': sum(edit['kind'] == 'parent' for edit in edits),
            'tree_edit/pose_max_xyz_error_world': position_error,
            'tree_edit/pose_max_rotation_error_rad': rotation_error})
        return edits

    @torch.no_grad()
    def _try_reparent(self, target_xyz, target_q):
        """Bounded full-pose proposals; a valid edit must save weighted cost."""
        active = torch.where(self.owner.bincount(minlength=len(self.parents))[1:] > 0)[0] + 1
        if not len(active):
            return []
        rt, rr, _, _ = self._values()
        strength = relative_motion_magnitude(rt[active], rr[active], self.extent)
        selected = active[strength.topk(min(self.config.tree_edit_region_trials,
                                             len(active))).indices]
        edits = []
        raw_bytes = max(1, len(self.owner) * 24)
        for node_tensor in selected:
            node = int(node_tensor)
            original_parent = int(self.parents[node])
            q_global, b_global = self._global_regions()
            q_target, b_target = q_global[node].clone(), b_global[node].clone()
            original_rt = self.region_translation[node].clone()
            original_rr = self.region_rotation[node].clone()
            original_gt = self.gs_translation.detach().clone()
            original_gr = self.gs_rotation.detach().clone()
            baseline = (self.config.rate_loss_weight * float(self._hard_motion_bytes()) / raw_bytes +
                        self.config.relative_motion_loss_weight *
                        float(self._relative_motion()))
            best = None
            distances = (self.pivots - self.pivots[node]).norm(dim=-1)
            near = distances.topk(min(len(self.parents), self.config.tree_edit_neighbors + 2),
                                  largest=False).indices.tolist()
            for parent in [0] + near:
                if parent == node or parent == original_parent:
                    continue
                trial = self.parents.clone()
                trial[node] = parent
                try:
                    order = _topological_order(trial)
                except ValueError:
                    continue
                depth = [0] * len(trial)
                for child in order:
                    depth[child] = depth[int(trial[child])] + 1
                if max(depth) > self.config.parent_max_depth:
                    continue
                q_parent, b_parent = q_global[parent], b_global[parent]
                local_q = quaternion_multiply(q_target, quaternion_inverse(q_parent))
                local_r = quaternion_rotvec(local_q[None])[0]
                inherited_pivot = quaternion_rotate(q_parent, self.pivots[node])
                local_t = (b_target - b_parent -
                           rotation_delta(local_q, -inherited_pivot))
                if bool((local_t / self.steps[:3]).abs().gt(32767.5).any()) or bool((local_r / self.steps[3:]).abs().gt(32767.5).any()):
                    continue
                self.parents[node] = parent
                self.region_translation[node] = _integer(local_t, self.steps[:3]).to(local_t) * self.steps[:3]
                self.region_rotation[node] = _integer(local_r, self.steps[3:]).to(local_r) * self.steps[3:]
                candidate_gt, candidate_gr = self._residual_for_target(
                    target_xyz, target_q, self.owner)
                self.gs_translation.copy_(candidate_gt.to(self.gs_translation) * self.gs_steps[:3])
                self.gs_rotation.copy_(candidate_gr.to(self.gs_rotation) * self.gs_steps[3:])
                decoded_xyz, decoded_q = self.forward(self.config.motion_steps)
                xyz_error = (decoded_xyz - target_xyz).norm(dim=-1).amax()
                angle_error = quaternion_angle_error(decoded_q, target_q)
                if (xyz_error <= math.sqrt(3) * float(self.gs_steps[0]) / 2 + 1e-6 and
                        bool((angle_error <= math.sqrt(3) * float(self.gs_steps[3]) / 2).all())):
                    edit_bytes = _varint_length(node) + _varint_length(parent)
                    relative = float(self._relative_motion())
                    score = (self.config.rate_loss_weight *
                             (float(self._hard_motion_bytes()) + edit_bytes) / raw_bytes +
                             self.config.relative_motion_loss_weight * relative)
                    if score < baseline and (best is None or score < best[0]):
                        best = (score, parent, self.region_translation[node].clone(),
                                self.region_rotation[node].clone(),
                                self.gs_translation.detach().clone(),
                                self.gs_rotation.detach().clone())
                self.parents[node] = original_parent
                self.region_translation[node] = original_rt
                self.region_rotation[node] = original_rr
                self.gs_translation.copy_(original_gt)
                self.gs_rotation.copy_(original_gr)
            if best is not None:
                _, parent, new_rt, new_rr, new_gt, new_gr = best
                self.parents[node] = parent
                self.region_translation[node] = new_rt
                self.region_rotation[node] = new_rr
                self.gs_translation.copy_(new_gt)
                self.gs_rotation.copy_(new_gr)
                self.optimizer.state.pop(self.region_translation, None)
                self.optimizer.state.pop(self.region_rotation, None)
                self.optimizer.state.pop(self.gs_translation, None)
                self.optimizer.state.pop(self.gs_rotation, None)
                edits.append(dict(kind='parent', node=node, parent=parent))
        return edits

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
        self._metadata_cache = None
        self.materialized_owner = torch.cat((self.owner, self.owner[sources])).detach().clone()
        self.materialized_xyz = xyz.detach().clone()
        self.materialized_rotation = rotation.detach().clone()
        self.committed = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def export_motion(self, path):
        from .rigid_tree_codec import (compress_motion_packet, decode_motion,
                                       encode_motion, motion_packet_info)
        if not self.committed:
            raise ValueError('Materialize the frame before exporting motion')
        rt, rr, gt, gr = self._values(hard=True)
        raw_packet = encode_motion(
            self.frame_reference_state, self.frame_index, self.extent,
            self.previous_rows.detach().cpu(), self.owner.detach().cpu(),
            self.parents.detach().cpu(), rt.detach().cpu(), rr.detach().cpu(),
            gt.detach().cpu(), gr.detach().cpu(),
            self.pending_birth_sources,
            self.materialized_xyz[len(self.owner):].detach().cpu(),
            self.materialized_rotation[len(self.owner):].detach().cpu(),
            self.config.motion_xyz_step_extent, self.config.motion_rotation_step_rad,
            gs_translation_step_extent=self.config.gs_xyz_step_extent,
            gs_rotation_step_rad=self.config.gs_rotation_step_rad)
        packet = compress_motion_packet(raw_packet, mode='auto')
        packet_info = motion_packet_info(packet)
        decoded = decode_motion(packet, self.frame_reference_state)
        if not torch.allclose(decoded.xyz, self.materialized_xyz.detach().cpu(), atol=1e-6, rtol=1e-5):
            raise ValueError('Motion stream does not reconstruct materialized centers')
        q_dot = (F.normalize(decoded.rotation, dim=-1) *
                 F.normalize(self.materialized_rotation.detach().cpu(), dim=-1)).sum(-1).abs()
        if not bool((q_dot > 1 - 1e-5).all()):
            raise ValueError('Motion stream does not reconstruct materialized orientations')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packet)
        self.codec_previous_state = decoded
        return dict(path=str(path), bytes=path.stat().st_size,
                    payload_bytes=path.stat().st_size, **packet_info,
                    frame_index=self.frame_index, gs_count=len(decoded.xyz),
                    region_count=len(self.parents),
                    region_translation_step_world=float(self.steps[0]),
                    region_rotation_step_rad=float(self.steps[3]),
                    gs_translation_step_world=float(self.gs_steps[0]),
                    gs_rotation_step_rad=float(self.gs_steps[3]))

    def capture(self):
        if self.owner is None:
            return None
        names = ('owner', 'parents', 'pivots', 'base_xyz', 'base_rotation',
                 'previous_rows', 'materialized_owner', 'materialized_xyz',
                 'materialized_rotation', 'pending_birth_sources')
        tensors = {name: getattr(self, name, None) for name in names}
        return dict(config=asdict(self.config), tensors=tensors,
                    frame_index=self.frame_index, extent=self.extent,
                    parameters={name: parameter.detach().clone() for name, parameter in
                                (('region_translation', self.region_translation),
                                 ('region_rotation', self.region_rotation),
                                 ('gs_translation', self.gs_translation),
                                 ('gs_rotation', self.gs_rotation))},
                    optimizer=self.optimizer.state_dict(), committed=self.committed,
                    last_step=self.last_step, topology_edits=list(self.topology_edits),
                    last_metrics=deepcopy(self.last_metrics),
                    topology_updated=self._topology_updated,
                    codec_previous_state=(None if self.codec_previous_state is None else
                                          self.codec_previous_state.capture()),
                    frame_reference_state=(None if self.frame_reference_state is None else
                                           self.frame_reference_state.capture()))

    def restore(self, packet):
        if packet is None:
            return
        if saved_config_with_default_motion_lr(packet['config']) != asdict(self.config):
            raise ValueError('Rigid tree checkpoint configuration differs')
        from .rigid_tree_codec import DecodedState
        state = packet['codec_previous_state']
        self.codec_previous_state = None if state is None else DecodedState.restore(state)
        reference = packet.get('frame_reference_state')
        tensors = packet['tensors']
        self.materialized_owner = None
        self.setup(tensors['base_xyz'], tensors['base_rotation'], packet['extent'],
                   packet['frame_index'])
        for name, value in tensors.items():
            setattr(self, name, value)
        with torch.no_grad():
            for name, value in packet['parameters'].items():
                getattr(self, name).copy_(value)
        self.optimizer.load_state_dict(packet['optimizer'])
        self.frame_index = packet['frame_index']
        self.committed = packet['committed']
        self.last_step = packet['last_step']
        self.topology_edits = list(packet['topology_edits'])
        self.last_metrics = deepcopy(packet.get('last_metrics', {}))
        self._topology_updated = bool(packet.get('topology_updated', False))
        self._metadata_cache = None
        self.codec_previous_state = None if state is None else DecodedState.restore(state)
        self.frame_reference_state = (None if reference is None else
                                      DecodedState.restore(reference))
        if self.committed:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
