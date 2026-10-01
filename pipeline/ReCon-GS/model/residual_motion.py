"""Quantized cumulative regional offsets and per-Gaussian residuals."""
from dataclasses import asdict, dataclass, fields
from typing import List, Optional
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from libgs.utils.general import quaternion_multiply
from .region_rigid import rotation_vector_quaternion

RAW_FIELDS = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')
RAW_ATTRIBUTES = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest')
LAYOUT_VERSION = 'stable_cpu_ternary_v1'


@dataclass
class QuantizedResidualConfig:
    fine_region_gs: Optional[int] = None
    motion_steps: Optional[int] = None
    shared_warmup_steps: Optional[int] = None
    incremental_steps: Optional[int] = None
    enable_incremental: Optional[bool] = None
    motion_init_decay: float = 0.0
    region_translation_lr: Optional[float] = None
    region_rotation_lr: Optional[float] = None
    gs_translation_lr: Optional[float] = None
    gs_rotation_lr: Optional[float] = None
    adam_betas: Optional[List[float]] = None
    adam_eps: Optional[float] = None
    motion_xyz_step_extent: Optional[float] = None
    motion_rotation_step_rad: Optional[float] = None
    new_xyz_step_extent: Optional[float] = None
    new_rotation_step: Optional[float] = None
    new_logscale_step: Optional[float] = None
    new_opacity_step: Optional[float] = None
    new_sh_step: Optional[float] = None
    opacity_keep_threshold: Optional[float] = None
    opacity_support_temperature: Optional[float] = None
    clone_sigma_scale: Optional[float] = None
    new_max_screen_size: Optional[float] = None
    densification_stats_steps: Optional[int] = None
    max_new_gaussians: Optional[int] = None
    enable_learned_gates: bool = False
    tree_rebuild_interval: int = 0
    tree_rebuild_neighbors: int = 3
    gate_lr: Optional[float] = None
    gate_initial_logits: Optional[List[float]] = None
    gate_temperature_start: Optional[float] = None
    gate_temperature_end: Optional[float] = None
    gate_refine_steps: Optional[int] = None
    storage_only: bool = True
    storage_loss_weight: Optional[float] = None
    storage_warmup_steps: Optional[int] = None
    storage_ramp_steps: Optional[int] = None

    def validate(self):
        # libgs structured configs may contain OmegaConf containers.
        if self.storage_only is not True:
            raise ValueError('quantized_residual now requires storage_only: true; rate/codec training is unsupported')
        if self.adam_betas is not None:
            self.adam_betas = [float(value) for value in self.adam_betas]
        gated = {'gate_lr', 'gate_initial_logits', 'gate_temperature_start',
                 'gate_temperature_end', 'gate_refine_steps'}
        if self.gate_initial_logits is not None:
            self.gate_initial_logits = [float(value) for value in self.gate_initial_logits]
        optional = {'densification_stats_steps', 'max_new_gaussians'} | (
            set() if self.enable_learned_gates else gated)
        missing = [f.name for f in fields(self) if f.name not in optional and getattr(self, f.name) is None]
        if missing:
            raise ValueError('quantized_residual requires explicit YAML values: ' + ', '.join(missing))
        positive_ints = ['fine_region_gs', 'motion_steps', 'storage_ramp_steps', 'tree_rebuild_neighbors']
        for name in positive_ints:
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        nonnegative_ints = ['shared_warmup_steps', 'incremental_steps', 'tree_rebuild_interval']
        nonnegative_ints += ['storage_warmup_steps']
        for name in nonnegative_ints:
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.tree_rebuild_interval and not self.enable_learned_gates:
            raise ValueError('tree_rebuild_interval requires enable_learned_gates')
        if self.shared_warmup_steps >= self.motion_steps:
            raise ValueError('shared_warmup_steps must leave joint residual fitting steps')
        for name in ('densification_stats_steps', 'max_new_gaussians'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f'{name} must be a positive integer or None')
        if self.densification_stats_steps is not None and self.densification_stats_steps > self.motion_steps:
            raise ValueError('densification_stats_steps must not exceed motion_steps')
        if not isinstance(self.enable_incremental, bool) or self.enable_incremental != (self.incremental_steps > 0):
            raise ValueError('enable_incremental must agree with incremental_steps')
        if (isinstance(self.motion_init_decay, bool) or
                not math.isfinite(self.motion_init_decay) or not 0 <= self.motion_init_decay <= 1):
            raise ValueError('motion_init_decay must be finite and in [0, 1]')
        if len(self.adam_betas) != 2 or not all(0 <= v < 1 for v in self.adam_betas):
            raise ValueError('adam_betas must contain two values in [0, 1)')
        for name in ('region_translation_lr', 'region_rotation_lr', 'gs_translation_lr', 'gs_rotation_lr',
                     'adam_eps', 'motion_xyz_step_extent', 'motion_rotation_step_rad',
                     'new_xyz_step_extent', 'new_rotation_step', 'new_logscale_step', 'new_opacity_step',
                     'new_sh_step', 'opacity_support_temperature', 'clone_sigma_scale', 'new_max_screen_size'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.opacity_keep_threshold < 1:
            raise ValueError('opacity_keep_threshold must be in (0, 1)')
        if not math.isfinite(self.storage_loss_weight) or self.storage_loss_weight < 0:
            raise ValueError('storage_loss_weight must be finite and nonnegative')
        if not self.enable_learned_gates and self.storage_loss_weight != 0:
            raise ValueError('nongated motion has no differentiable storage selector; storage_loss_weight must be zero')
        if not isinstance(self.enable_learned_gates, bool):
            raise ValueError('enable_learned_gates must be bool')
        if self.enable_learned_gates:
            if (len(self.gate_initial_logits) != 3 or
                    not all(math.isfinite(v) for v in self.gate_initial_logits)):
                raise ValueError('gate_initial_logits needs three finite values')
            for name in ('gate_lr', 'gate_temperature_start', 'gate_temperature_end'):
                value = getattr(self, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{name} must be finite and positive')
            for name in ('gate_refine_steps',):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f'{name} must be a nonnegative integer')
            if self.gate_refine_steps >= self.motion_steps:
                raise ValueError('gate_refine_steps must leave joint gate training steps')

    @property
    def total_steps(self):
        return self.motion_steps + self.incremental_steps

    @property
    def densification_stats_start(self):
        return (self.motion_steps - self.densification_stats_steps + 1
                if self.densification_stats_steps is not None else 1)


def quantized_symbols(value, step):
    continuous = value / torch.as_tensor(step, device=value.device, dtype=value.dtype)
    return continuous + (continuous.round() - continuous).detach()


def integer_symbols(value, step):
    symbols = (value.detach() / torch.as_tensor(step, device=value.device, dtype=value.dtype)).round()
    if not bool(torch.isfinite(symbols).all()) or (symbols.numel() and bool(((symbols < -(2 ** 31)) | (symbols >= 2 ** 31)).any())):
        raise ValueError('Quantized values must fit finite int32 symbols')
    return symbols.to(torch.int32)


def stable_layout(xyz, fine_region_gs):
    """Three levels; CPU stable point order resolves coordinate ties deterministically."""
    points = xyz.detach().cpu().float().numpy()
    if not len(points):
        raise ValueError('Residual layout requires at least one previous Gaussian')
    def divide(indices, parts):
        axis = np.ptp(points[indices], axis=0).argmax()
        order = np.lexsort((indices, points[indices, axis]))
        return list(np.array_split(indices[order], min(parts, len(indices))))
    coarse_count = max(1, math.ceil(len(points) / (fine_region_gs * 9)))
    coarse = [np.arange(len(points), dtype=np.int64)]
    while len(coarse) < coarse_count:
        largest = max(range(len(coarse)), key=lambda i: len(coarse[i]))
        coarse[largest:largest + 1] = divide(coarse[largest], 2)
    groups = [coarse]
    parent = []
    for _ in range(2):
        next_groups, parents = [], []
        for n, indices in enumerate(groups[-1]):
            parts = divide(indices, 3)
            next_groups.extend(parts)
            parents.extend([n] * len(parts))
        groups.append(next_groups)
        parent.append(torch.tensor(parents, dtype=torch.long))
    index = torch.empty((3, len(points)), dtype=torch.long)
    for level, rows in enumerate(groups):
        for n, indices in enumerate(rows):
            index[level, torch.from_numpy(indices)] = n
    return index.to(xyz.device), [len(rows) for rows in groups], parent


def compose_motion(base_xyz, base_rotation, index, region_values, gs_values):
    """Translations are offsets; world rotation order is GS*fine*middle*coarse*old."""
    offset = gs_values[:, :3]
    rotation = base_rotation
    path_active = torch.zeros(len(base_xyz), device=base_xyz.device, dtype=torch.bool)
    for level, values in enumerate(region_values):
        selected = values[index[level]]
        offset = offset + selected[:, :3]
        rotation = quaternion_multiply(rotation_vector_quaternion(selected[:, 3:]), rotation)
        path_active |= selected.detach().ne(0).any(-1)
    rotation = quaternion_multiply(rotation_vector_quaternion(gs_values[:, 3:]), rotation)
    path_active |= gs_values.detach().ne(0).any(-1)
    moved = base_xyz + offset
    # Exact zero-path reuse in the forward, preserving STE image gradients.
    exact_xyz = torch.where(path_active[:, None], moved, base_xyz)
    exact_rotation = torch.where(path_active[:, None], rotation, base_rotation)
    return (exact_xyz.detach() + (moved - moved.detach()),
            exact_rotation.detach() + (rotation - rotation.detach()))


def shortest_rotation_vector(quaternion):
    """Scalar-first quaternion to the shortest relative rotation vector."""
    quaternion = F.normalize(quaternion, dim=-1)
    quaternion = torch.where(quaternion[:, :1] < 0, -quaternion, quaternion)
    imaginary = quaternion[:, 1:]
    length = imaginary.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, quaternion[:, :1])
    scale = torch.where(length > 0, angle / length.clamp_min(torch.finfo(length.dtype).tiny),
                        torch.full_like(length, 2.))
    return imaginary * scale


def inverse_unit_quaternion(quaternion):
    return torch.cat((quaternion[:, :1], -quaternion[:, 1:]), dim=-1)


def model_raw(model):
    return {name: getattr(model, attr).detach().clone() for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}


def survivor_rows(reference_count, deleted_indices):
    """Previous materialized row numbers, in the order retained by Gaussian pruning."""
    deleted = torch.as_tensor(deleted_indices, dtype=torch.long).reshape(-1)
    if (reference_count < 0 or deleted.numel() > reference_count or
            bool(((deleted < 0) | (deleted >= reference_count)).any()) or
            deleted.unique().numel() != deleted.numel()):
        raise ValueError('Deleted previous-frame indices must be unique and in range')
    keep = torch.ones(reference_count, dtype=torch.bool, device=deleted.device)
    keep[deleted] = False
    return torch.arange(reference_count, device=deleted.device)[keep]


def reference_digest(raw):
    digest = hashlib.sha256()
    for name in RAW_FIELDS:
        tensor = raw[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def attribute_steps(config, extent):
    return dict(xyz=config.new_xyz_step_extent * extent, rotation=config.new_rotation_step,
                logscale=config.new_logscale_step, opacity=config.new_opacity_step,
                sh_dc=config.new_sh_step, sh_rest=config.new_sh_step)


class QuantizedResidualMotion(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.optimizer = None
        self.committed = False
        self.last_metrics = {}
        self.stage_seconds = {}
        self.index = None

    def setup(self, xyz, rotation, extent, warm_start=True,
              previous_rows=None, previous_new_count=None):
        previous = None
        if warm_start and self.config.motion_init_decay and self.index is not None:
            if not self.committed:
                raise ValueError('Motion warm start requires a committed previous frame')
            if previous_rows is None:
                if len(xyz) != len(self.base_xyz):
                    raise ValueError('Motion warm start requires unchanged Gaussian count and order')
                previous_rows = torch.arange(len(xyz), device=xyz.device)
                previous_new_count = 0
            elif previous_new_count is None:
                raise ValueError('Motion warm start mapping requires previous_new_count')
            if (isinstance(previous_new_count, bool) or not isinstance(previous_new_count, int) or
                    previous_new_count < 0):
                raise ValueError('previous_new_count must be a nonnegative integer')
            rows = torch.as_tensor(previous_rows, device=xyz.device)
            if rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64) or len(rows) != len(xyz):
                raise ValueError('Motion warm start mapping requires one integer row per surviving Gaussian')
            rows = rows.to(torch.long)
            previous_count = len(self.base_xyz) + previous_new_count
            if (bool(((rows < 0) | (rows >= previous_count)).any()) or
                    bool((rows[1:] <= rows[:-1]).any())):
                raise ValueError('Motion warm start mapping must contain unique, ordered in-range rows')
            old_mask = rows < len(self.base_xyz)
            expected_xyz, expected_rotation = self.forward(self.config.motion_steps)
            aligned_rotation = torch.abs((F.normalize(expected_rotation[rows[old_mask]], dim=-1) *
                                          F.normalize(rotation[old_mask], dim=-1)).sum(-1))
            if (not torch.allclose(expected_xyz[rows[old_mask]], xyz[old_mask], rtol=1e-5, atol=1e-6) or
                    not torch.allclose(aligned_rotation, torch.ones_like(aligned_rotation), rtol=1e-5, atol=1e-6)):
                raise ValueError('Motion warm start mapping does not match committed ordered previous-frame Gaussians')
            old_translation, old_rotation = self.final_quantized_total_motion()
            translation = torch.zeros_like(xyz)
            relative_rotation = xyz.new_zeros(len(xyz), 4)
            relative_rotation[:, 0] = 1
            translation[old_mask] = old_translation[rows[old_mask]]
            relative_rotation[old_mask] = old_rotation[rows[old_mask]]
            # Newly added Gaussians have no previous motion history.
            previous = translation, relative_rotation
        self.extent = float(extent)
        self.index, self.counts, self.parents = stable_layout(xyz, self.config.fine_region_gs)
        for name, value in (('base_xyz', xyz.detach().clone()), ('base_rotation', rotation.detach().clone())):
            if name in self._buffers:
                setattr(self, name, value)
            else:
                self.register_buffer(name, value)
        self.region_translation = nn.ParameterList([nn.Parameter(xyz.new_zeros(n, 3)) for n in self.counts])
        self.region_rotation = nn.ParameterList([nn.Parameter(xyz.new_zeros(n, 3)) for n in self.counts])
        self.gs_translation = nn.Parameter(torch.zeros_like(xyz))
        self.gs_rotation = nn.Parameter(torch.zeros_like(xyz))
        if previous is not None:
            self.initialize_from_previous(*previous)
        self.make_optimizer()
        self.committed = False
        self.last_step = 0
        self.last_metrics = {}
        self.stage_seconds = {}

    @torch.no_grad()
    def final_quantized_total_motion(self):
        """Per-GS final quantized translation and relative world rotation."""
        symbols = self.symbol_tensors(integer=True)
        steps = self.motion_steps_vector
        offset = symbols['gs_local'][:, :3].to(steps) * steps[:3]
        identity = self.base_xyz.new_zeros(len(self.base_xyz), 4)
        identity[:, 0] = 1
        quaternion = identity
        for level in range(3):
            selected = symbols[f'region_{level}'][self.index[level]].to(steps) * steps
            offset = offset + selected[:, :3]
            quaternion = quaternion_multiply(rotation_vector_quaternion(selected[:, 3:]), quaternion)
        local = symbols['gs_local'].to(steps) * steps
        quaternion = quaternion_multiply(rotation_vector_quaternion(local[:, 3:]), quaternion)
        return offset.detach().clone(), F.normalize(quaternion.detach(), dim=-1)

    @torch.no_grad()
    def initialize_from_previous(self, old_translation, old_rotation):
        """Project identity-aligned total motion onto this frame's new layout."""
        decay = self.config.motion_init_decay
        target_translation = old_translation.to(self.base_xyz) * decay
        target_rotation = rotation_vector_quaternion(
            shortest_rotation_vector(old_rotation.to(self.base_xyz)) * decay)
        steps = self.motion_steps_vector
        translation_left = target_translation.clone()
        rotation_left = target_rotation.clone()
        for level in range(3):
            owner = self.index[level]
            counts = self.base_xyz.new_zeros(self.counts[level], 1)
            counts.index_add_(0, owner, self.base_xyz.new_ones(len(owner), 1))
            translation_sum = self.base_xyz.new_zeros(self.counts[level], 3)
            translation_sum.index_add_(0, owner, translation_left)
            translation_symbols = torch.round(translation_sum / counts / steps[:3])
            translation = translation_symbols * steps[:3]
            self.region_translation[level].copy_(translation)
            translation_left -= translation[owner]
            rotation_sum = self.base_xyz.new_zeros(self.counts[level], 3)
            rotation_sum.index_add_(0, owner, shortest_rotation_vector(rotation_left))
            rotation_symbols = torch.round(rotation_sum / counts / steps[3:])
            rotation = rotation_symbols * steps[3:]
            self.region_rotation[level].copy_(rotation)
            # q_left = q_remainder * q_new_region in compose_motion order.
            region_quaternion = rotation_vector_quaternion(rotation[owner])
            rotation_left = quaternion_multiply(rotation_left, inverse_unit_quaternion(region_quaternion))
        target_translation_symbols = torch.round(target_translation / steps[:3])
        region_translation_symbols = sum(
            torch.round(value[self.index[level]] / steps[:3])
            for level, value in enumerate(self.region_translation))
        self.gs_translation.copy_((target_translation_symbols - region_translation_symbols) * steps[:3])
        self.gs_rotation.copy_(torch.round(shortest_rotation_vector(rotation_left) / steps[3:]) * steps[3:])

    def make_optimizer(self):
        self.optimizer = torch.optim.Adam([
            dict(params=self.region_translation, lr=self.config.region_translation_lr),
            dict(params=self.region_rotation, lr=self.config.region_rotation_lr),
            dict(params=[self.gs_translation], lr=self.config.gs_translation_lr),
            dict(params=[self.gs_rotation], lr=self.config.gs_rotation_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)

    def stage(self, step):
        return 'shared_warmup' if step <= self.config.shared_warmup_steps else 'joint_residual'

    @property
    def motion_steps_vector(self):
        return self.base_xyz.new_tensor([self.config.motion_xyz_step_extent * self.extent] * 3 +
                                        [self.config.motion_rotation_step_rad] * 3)

    def symbol_tensors(self, step=None, integer=False):
        quantize = integer_symbols if integer else quantized_symbols
        steps = self.motion_steps_vector
        result = {f'region_{level}': quantize(torch.cat((translation, rotation), -1), steps)
                  for level, (translation, rotation) in enumerate(zip(self.region_translation, self.region_rotation))}
        local = torch.cat((self.gs_translation, self.gs_rotation), -1)
        if step is not None and step <= self.config.shared_warmup_steps:
            local = local.detach() if self.config.motion_init_decay else local.detach() * 0
        result['gs_local'] = quantize(local, steps)
        return result

    def forward(self, step, training=False):
        symbols = self.symbol_tensors(step)
        if training:
            self.last_step = step
        steps = self.motion_steps_vector
        return compose_motion(self.base_xyz, self.base_rotation, self.index,
                              [symbols[f'region_{n}'] * steps for n in range(3)], symbols['gs_local'] * steps)

    @torch.no_grad()
    def optimizer_step(self):
        if self.last_step <= self.config.shared_warmup_steps:
            for parameter in (self.gs_translation, self.gs_rotation):
                parameter.grad = None
                self.optimizer.state.pop(parameter, None)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def storage_parts(self, step, hard=False):
        """Dense raw motion tensors; a nongated model has no stored selection map."""
        bytes_per_tuple = 6 * self.base_xyz.element_size()
        zero = self.base_xyz.new_zeros(())
        region_count = sum(self.counts)
        return dict(gate_bytes=zero, region_motion_bytes=zero + region_count * bytes_per_tuple,
                    gs_motion_bytes=zero + len(self.base_xyz) * bytes_per_tuple)

    def storage_scale(self, step):
        return min(max((step - self.config.storage_warmup_steps) /
                       self.config.storage_ramp_steps, 0.), 1.)

    def reference_storage_bytes(self, first_frame_gs_count):
        return first_frame_gs_count * 6 * self.base_xyz.element_size()

    def statistics(self):
        symbols = self.symbol_tensors(integer=True)
        active = [symbols[f'region_{n}'].ne(0).any(-1) for n in range(3)]
        local = symbols['gs_local'].ne(0).any(-1)
        inherited = torch.zeros(len(self.base_xyz), device=self.base_xyz.device, dtype=torch.bool)
        result = dict(region_counts=self.counts, levels=[])
        for n in range(3):
            inherited |= active[n][self.index[n]]
            descendants = local.clone()
            for child_level in range(n + 1, 3):
                descendants |= active[child_level][self.index[child_level]]
            # CUDA torch 2.5 amax scatter does not dispatch Boolean tensors.
            split_count = torch.zeros(self.counts[n], device=self.base_xyz.device, dtype=torch.int32)
            moving_count = torch.zeros_like(split_count)
            split_count.scatter_reduce_(0, self.index[n], descendants.to(torch.int32), reduce='amax', include_self=True)
            moving_count.scatter_reduce_(0, self.index[n], inherited.to(torch.int32), reduce='amax', include_self=True)
            split, moving_path = split_count.bool(), moving_count.bool()
            shared = ~split & moving_path
            keep = ~split & ~moving_path
            row = dict(level=n, denominator=self.counts[n],
                       counts=dict(KEEP=int(keep.sum()), SHARED=int(shared.sum()), SPLIT=int(split.sum())),
                       nonzero_motion_fraction=float(active[n].float().mean()),
                       inherited_gs_fraction=float(inherited.float().mean()))
            row['fractions'] = {name: count / row['denominator'] for name, count in row['counts'].items()}
            result['levels'].append(row)
        gs_counts = dict(KEEP=int((~local & ~inherited).sum()),
                         SHARED=int((~local & inherited).sum()), SPLIT=int(local.sum()))
        result['gs'] = dict(denominator=len(local), counts=gs_counts,
                            fractions={name: count / len(local) for name, count in gs_counts.items()})
        result.update(gs_local_active_fraction=float(local.float().mean()),
                      gs_shared_inherited_fraction=float((~local & inherited).float().mean()),
                      gs_keep_fraction=float((~local & ~inherited).float().mean()),
                      semantics='Each level has its own region denominator; levels are not one mutually exclusive tree cut. '
                      'A region SPLIT means any lower regional or local-GS residual is nonzero and retains its own motion. '
                      'Without active descendants, a moving ancestor/self path is SHARED; an entirely zero path is KEEP. '
                      'Old-GS SPLIT means local residual nonzero; SHARED inherits regional motion with local zero; KEEP has all four residuals zero.')
        return result

    def capture(self):
        if self.index is None:
            return None
        return dict(config=asdict(self.config), state=self.state_dict(), extent=self.extent,
                    last_step=self.last_step, committed=self.committed, last_metrics=self.last_metrics,
                    stage_seconds=self.stage_seconds, optimizer=self.optimizer.state_dict())

    def restore(self, packet):
        if packet is None:
            return
        saved_config = dict(packet['config'])
        saved_config.setdefault('motion_init_decay', 0.0)
        saved_config.setdefault('densification_stats_steps', None)
        saved_config.setdefault('max_new_gaussians', None)
        saved_config.setdefault('enable_learned_gates', False)
        if not saved_config.get('storage_only', False):
            raise ValueError('Legacy rate/codec dynamic-frame checkpoints cannot resume with storage-only motion')
        removed = set(saved_config) - {field.name for field in fields(self.config)}
        if removed:
            raise ValueError('Dynamic-frame checkpoint uses removed rate/codec settings: '
                             + ', '.join(sorted(removed)))
        for name in ('gate_lr', 'gate_initial_logits', 'gate_temperature_start',
                     'gate_temperature_end', 'gate_refine_steps'):
            saved_config.setdefault(name, None)
        if saved_config != asdict(self.config):
            raise ValueError('Quantized residual resume requires identical layout, quantization and storage settings')
        self.setup(packet['state']['base_xyz'], packet['state']['base_rotation'], packet['extent'], warm_start=False)
        self.load_state_dict(packet['state'])
        self.optimizer.load_state_dict(packet['optimizer'])
        self.last_step, self.committed = packet['last_step'], packet['committed']
        if self.committed:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self.last_metrics, self.stage_seconds = packet['last_metrics'], packet['stage_seconds']


class QuantizedNewGaussianView:
    """Render quantized raw attributes; opacity determines retention without another gate."""
    def __init__(self, model, config, extent, reference, source_tree=None):
        self.model, self.config = model, config
        self.steps = attribute_steps(config, extent)
        if not len(reference['xyz']):
            raise ValueError('New Gaussian prediction needs a nonempty motion-adjusted reference')
        if source_tree is None:
            from scipy.spatial import cKDTree
            source_tree = cKDTree(reference['xyz'].detach().cpu().numpy())
        positions = model._xyz.detach().cpu().numpy()
        _, source = source_tree.query(positions, k=1, workers=1)
        self.source = torch.as_tensor(source, device=model._xyz.device, dtype=torch.long)
        self.predictor = {name: value[self.source].detach() for name, value in reference.items()}
        self.predictor['rotation'] = F.normalize(self.predictor['rotation'], dim=-1)
        self.symbols = {}
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            value = getattr(model, attr)
            if name == 'rotation':
                value = F.normalize(value, dim=-1)
                value = torch.where((value * self.predictor[name]).sum(-1, keepdim=True) < 0, -value, value)
            self.symbols[name] = quantized_symbols(value - self.predictor[name], self.steps[name])
        opacity = torch.sigmoid(self.raw_value('opacity'))
        self.keep = opacity.detach().flatten() >= config.opacity_keep_threshold
        probability = torch.sigmoid((opacity.flatten() - config.opacity_keep_threshold) /
                                    config.opacity_support_temperature)
        self.keep_st = self.keep.to(probability) + (probability - probability.detach())

    def raw_value(self, name):
        return self.predictor[name] + self.symbols[name] * self.steps[name]

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def get_xyz(self):
        return self.raw_value('xyz')

    @property
    def get_rotation(self):
        return F.normalize(self.raw_value('rotation'), dim=-1)

    @property
    def get_scaling(self):
        return torch.exp(self.raw_value('logscale'))

    @property
    def get_opacity(self):
        return torch.sigmoid(self.raw_value('opacity')) * self.keep_st[:, None]

    @property
    def get_features(self):
        return torch.cat((self.raw_value('sh_dc'), self.raw_value('sh_rest')), dim=1)

    def get_covariance(self, scaling_modifier=1):
        return self.model.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

def materialize_quantized_gaussians(base_raw, new_view=None):
    """Append retained quantized births to already moved/deleted old raw tensors."""
    raw = {name: base_raw[name].detach().clone() for name in RAW_FIELDS}
    base_count = len(raw['xyz'])
    if any(len(value) != base_count for value in raw.values()):
        raise ValueError('Materialized base attributes disagree on Gaussian count')
    if new_view is None:
        return raw, torch.empty(0, dtype=torch.long, device=raw['xyz'].device)
    keep = new_view.keep
    if (keep.ndim != 1 or keep.dtype != torch.bool or
            new_view.source.ndim != 1 or
            new_view.source.dtype not in (torch.int32, torch.int64) or
            len(keep) != len(new_view.source)):
        raise ValueError('Quantized birth mask and sources must be aligned')
    sources = new_view.source[keep].detach().to(
        device=raw['xyz'].device, dtype=torch.long).clone()
    if len(sources) and bool(((sources < 0) | (sources >= base_count)).any()):
        raise ValueError('Retained birth source index is out of bounds')
    for name in RAW_FIELDS:
        birth = new_view.raw_value(name)[keep].detach()
        if name == 'rotation':
            birth = F.normalize(birth, dim=-1)
        if birth.shape[1:] != raw[name].shape[1:] or len(birth) != len(sources):
            raise ValueError(f'Retained birth {name} shape disagrees with base')
        raw[name] = torch.cat((raw[name], birth.to(raw[name])), dim=0)
    return raw, sources
