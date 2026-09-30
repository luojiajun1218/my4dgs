"""Quantized cumulative regional offsets and per-Gaussian residuals."""
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional
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
    lambda_rate: Optional[float] = None
    rate_warmup_steps: Optional[int] = None
    rate_ramp_steps: Optional[int] = None
    motion_entropy_scales: Optional[List[float]] = None
    new_entropy_scales: Optional[Dict[str, float]] = None
    tuple_mask_probability: Optional[float] = None
    support_temperature: Optional[float] = None
    entropy_radius: Optional[int] = None
    entropy_chunk_rows: Optional[int] = None
    opacity_keep_threshold: Optional[float] = None
    opacity_support_temperature: Optional[float] = None
    clone_sigma_scale: Optional[float] = None
    new_max_screen_size: Optional[float] = None

    def validate(self):
        # libgs structured configs may contain OmegaConf containers.
        if self.adam_betas is not None:
            self.adam_betas = [float(value) for value in self.adam_betas]
        if self.motion_entropy_scales is not None:
            self.motion_entropy_scales = [float(value) for value in self.motion_entropy_scales]
        if self.new_entropy_scales is not None:
            self.new_entropy_scales = {str(name): float(value) for name, value in self.new_entropy_scales.items()}
        missing = [f.name for f in fields(self) if getattr(self, f.name) is None]
        if missing:
            raise ValueError('quantized_residual requires explicit YAML values: ' + ', '.join(missing))
        for name in ('fine_region_gs', 'motion_steps', 'rate_ramp_steps', 'entropy_radius', 'entropy_chunk_rows'):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('shared_warmup_steps', 'incremental_steps', 'rate_warmup_steps'):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.shared_warmup_steps >= self.motion_steps:
            raise ValueError('shared_warmup_steps must leave joint residual fitting steps')
        if not isinstance(self.enable_incremental, bool) or self.enable_incremental != (self.incremental_steps > 0):
            raise ValueError('enable_incremental must agree with incremental_steps')
        if (isinstance(self.motion_init_decay, bool) or
                not math.isfinite(self.motion_init_decay) or not 0 <= self.motion_init_decay <= 1):
            raise ValueError('motion_init_decay must be finite and in [0, 1]')
        if self.motion_init_decay and self.enable_incremental:
            raise ValueError('motion_init_decay requires enable_incremental: false')
        if len(self.adam_betas) != 2 or not all(0 <= v < 1 for v in self.adam_betas):
            raise ValueError('adam_betas must contain two values in [0, 1)')
        for name in ('region_translation_lr', 'region_rotation_lr', 'gs_translation_lr', 'gs_rotation_lr',
                     'adam_eps', 'motion_xyz_step_extent', 'motion_rotation_step_rad',
                     'new_xyz_step_extent', 'new_rotation_step', 'new_logscale_step', 'new_opacity_step',
                     'new_sh_step', 'support_temperature', 'opacity_support_temperature', 'clone_sigma_scale', 'new_max_screen_size'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.lambda_rate) or self.lambda_rate < 0:
            raise ValueError('lambda_rate must be finite and nonnegative')
        if not 0 < self.tuple_mask_probability < 1 or not 0 < self.opacity_keep_threshold < 1:
            raise ValueError('Mask probability and opacity threshold must be in (0, 1)')
        if len(self.motion_entropy_scales) != 6 or not all(math.isfinite(v) and v > 0 for v in self.motion_entropy_scales):
            raise ValueError('motion_entropy_scales requires six positive symbol-domain Laplace scales')
        if set(self.new_entropy_scales) != set(RAW_FIELDS) or not all(math.isfinite(v) and v > 0 for v in self.new_entropy_scales.values()):
            raise ValueError('new_entropy_scales must provide a positive symbol-domain scale for every raw field')
        if self.entropy_radius > 16382:
            raise ValueError('Entropy alphabet must fit nonzero 16-bit CDF frequencies')

    @property
    def total_steps(self):
        return self.motion_steps + self.incremental_steps


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

    def setup(self, xyz, rotation, extent, warm_start=True):
        previous = None
        if warm_start and self.config.motion_init_decay and self.index is not None:
            if not self.committed:
                raise ValueError('Motion warm start requires a committed previous frame')
            if len(xyz) != len(self.base_xyz):
                raise ValueError('Motion warm start requires unchanged Gaussian count and order')
            expected_xyz, expected_rotation = self.forward(self.config.motion_steps)
            aligned_rotation = torch.abs((F.normalize(expected_rotation, dim=-1) *
                                          F.normalize(rotation, dim=-1)).sum(-1))
            if (not torch.allclose(expected_xyz, xyz, rtol=1e-5, atol=1e-6) or
                    not torch.allclose(aligned_rotation, torch.ones_like(aligned_rotation), rtol=1e-5, atol=1e-6)):
                raise ValueError('Motion warm start requires the same ordered previous-frame Gaussians')
            previous = self.final_quantized_total_motion()
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
        """Per-GS final encoded translation and relative world rotation."""
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

    def rate_scale(self, step):
        return min(max((step - self.config.rate_warmup_steps) / self.config.rate_ramp_steps, 0.), 1.)

    def entropy_models(self):
        return {name: dict(scale=list(self.config.motion_entropy_scales), sparse=True,
                           mask_probability=self.config.tuple_mask_probability)
                for name in ('region_0', 'region_1', 'region_2', 'gs_local')}

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
                       encoded_active_fraction=float(active[n].float().mean()),
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
        if saved_config != asdict(self.config):
            raise ValueError('Quantized residual resume requires identical layout, quantization and rate settings')
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

    def entropy_models(self):
        return {f'new_{name}': dict(scale=[self.config.new_entropy_scales[name]] * math.prod(value.shape[1:]),
                                    sparse=False)
                for name, value in self.symbols.items()}

    def integer_packet(self):
        packet = {f'new_{name}': integer_symbols(value[self.keep].flatten(1), 1)
                  for name, value in self.symbols.items()}
        packet['new_source_old_index'] = self.source[self.keep].to(torch.int32)
        return packet


def replay_frame(previous, tensors, metadata):
    """Independent CPU replay: delete -> rebuild layout -> motion -> append retained births."""
    if metadata['layout_version'] != LAYOUT_VERSION:
        raise ValueError('Unsupported residual layout version')
    previous = {name: value.detach().cpu() for name, value in previous.items()}
    if reference_digest(previous) != metadata['previous_reference_sha256']:
        raise ValueError('Incremental packet previous-reference digest mismatch')
    deleted = tensors['deleted_prev_indices'].cpu().long()
    if len(deleted) and (deleted.min() < 0 or deleted.max() >= len(previous['xyz']) or deleted.unique().numel() != deleted.numel()):
        raise ValueError('Invalid previous-frame deletion indices')
    keep = torch.ones(len(previous['xyz']), dtype=torch.bool)
    keep[deleted] = False
    remaining = {name: value[keep] for name, value in previous.items()}
    cfg = QuantizedResidualConfig(**metadata['config'])
    cfg.validate()
    if not cfg.enable_incremental and (len(deleted) or len(tensors['new_source_old_index'])):
        raise ValueError('Incremental-disabled residual packet cannot delete or add Gaussians')
    index, counts, _ = stable_layout(remaining['xyz'], cfg.fine_region_gs)
    region = [tensors[f'region_{n}'].cpu().float() for n in range(3)]
    if [len(row) for row in region] != counts or tensors['gs_local'].shape != (len(remaining['xyz']), 6):
        raise ValueError('Encoded motion dimensions do not match reconstructed layout')
    extent = metadata['extent']
    steps = remaining['xyz'].new_tensor([cfg.motion_xyz_step_extent * extent] * 3 + [cfg.motion_rotation_step_rad] * 3)
    xyz, rotation = compose_motion(remaining['xyz'], remaining['rotation'], index,
                                  [row * steps for row in region], tensors['gs_local'].cpu().float() * steps)
    remaining.update(xyz=xyz, rotation=rotation)
    attr_steps = attribute_steps(cfg, extent)
    sources = tensors['new_source_old_index'].cpu().long()
    if len(sources) and (sources.min() < 0 or sources.max() >= len(remaining['xyz'])):
        raise ValueError('New Gaussian prediction source index is out of bounds')
    births = {}
    for name in RAW_FIELDS:
        predictor = remaining[name][sources]
        if name == 'rotation':
            predictor = F.normalize(predictor, dim=-1)
        residual = tensors[f'new_{name}'].cpu().to(predictor).reshape_as(predictor)
        births[name] = predictor + residual * attr_steps[name]
    births['rotation'] = F.normalize(births['rotation'], dim=-1)
    count = len(births['xyz'])
    if any(len(value) != count or value.shape[1:] != remaining[name].shape[1:] for name, value in births.items()):
        raise ValueError('Encoded new Gaussian attributes have inconsistent dimensions')
    return {name: torch.cat((remaining[name], births[name]), dim=0) for name in RAW_FIELDS}
