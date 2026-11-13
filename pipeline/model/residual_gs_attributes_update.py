"""逐个 GS 的残差属性更新。

负责逐点位置、旋转、透明度、缩放和颜色增量的参数及相关更新；
区域共享属性放在 regional_attributes_update.py，码流组织交给 frame_stream 编解码模块。
"""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn

from .frame_stream_format import (FIELDS, PLANAR_MAGIC, HEADER, MAX_BODY, _check_reference,
    _materialize, validate_sparse_color_selection, _step, _opacity_varint as _varint,
    opacity_reference_hash, MODES, canonical_alpha, canonical_alpha_logits,
    OPACITY_MAGICS, _rows)
from .frame_stream_encode import (encode_full_attribute_delta, encode_opacity_delta,
    pack_layout, encode_opacity_layout, encode_logscale_delta, encode_feature_dc_packet)
from .frame_stream_decode import (decode_full_attribute_delta, decode_opacity_delta,
    decode_logscale_delta, decode_feature_dc_delta)
from .rigid_residual_tree import MotionBase, _integer, _quantized
from .storage_rate import raw_sparse_row_bytes, sparse_block_support



@dataclass
class FullUpdateConfig:
    enabled: bool = False
    audit_camera_sequence: bool = False
    gradient_diagnostics: bool = True
    logscale_step: float = .005
    sh_dc_step: float = .001
    sh_rest_step: float = .001
    storage_loss_enabled: bool = True
    per_attribute_l1_weight: float = .005
    point_dc_rate_scale: float = 1.0
    point_logscale_rate_multiplier: float = 1.0
    point_sh_rest_rate_multiplier: float = 1.0
    attribute_aware_gate: bool = False
    regional_opacity_l1_weight: float | None = None
    repair_temporal_weight: float = 0.0
    repair_temporal_scale: float = 0.1
    repair_temporal_lag: int = 1
    repair_rmse_mix: float = 0.0
    motion_rmse_mix: float = 0.0
    early_joint_repair: bool = False
    dense_color_200: bool = False
    regional_full_200: bool = False
    shared_region_orientation: bool = False

    # Read-only compatibility defaults for generic code and historical decoders.
    # These are not dataclass fields or selectable training options.
    train_attributes = True
    color_only = False
    resume_from_finalized_200 = False
    train_point_dc = True
    point_training_schedule = 'standard'
    color_point_budget = 0
    regional_full_point_budget = 0
    sparse_color_stream_v2 = False
    sparse_color_logscale_v3 = False
    shared_dc_rate_scale = 1.0

    @property
    def regional_repair(self):
        return self.enabled

    @property
    def stop_after_motion(self):
        return self.dense_color_200

    def validate(self):
        for name in ('enabled', 'audit_camera_sequence', 'gradient_diagnostics', 'storage_loss_enabled', 'early_joint_repair', 'dense_color_200', 'regional_full_200', 'attribute_aware_gate', 'shared_region_orientation'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'full_update.{name} must be boolean')
        for name in ('logscale_step', 'sh_dc_step', 'sh_rest_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'full_update.{name} must be finite and positive')
        for name in ('per_attribute_l1_weight', 'point_dc_rate_scale', 'point_logscale_rate_multiplier', 'point_sh_rest_rate_multiplier'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'full_update.{name} must be finite and nonnegative')
        opacity_l1 = self.regional_opacity_l1_weight
        if opacity_l1 is not None and (isinstance(opacity_l1, bool) or
                not isinstance(opacity_l1, (float, int)) or not math.isfinite(opacity_l1) or opacity_l1 < 0):
            raise ValueError('full_update.regional_opacity_l1_weight must be finite and nonnegative')
        for name in ('repair_temporal_weight', 'repair_temporal_scale', 'repair_rmse_mix', 'motion_rmse_mix'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'full_update.{name} must be finite')
        if not 0 <= self.repair_temporal_weight <= 2 or self.repair_temporal_scale <= 0:
            raise ValueError('Invalid temporal repair weight or scale')
        if type(self.repair_temporal_lag) is not int or self.repair_temporal_lag < 1:
            raise ValueError('full_update.repair_temporal_lag must be a positive integer')
        if not 0 <= self.repair_rmse_mix <= 1:
            raise ValueError('full_update.repair_rmse_mix must be in [0, 1]')
        if not 0 <= self.motion_rmse_mix <= 1:
            raise ValueError('full_update.motion_rmse_mix must be in [0, 1]')
        if self.dense_color_200 and (not self.enabled or not self.early_joint_repair):
            raise ValueError('Dense 200-step color requires early joint full updates')
        if self.regional_full_200 and not self.dense_color_200:
            raise ValueError('Regional full attributes require the dense 200-step plan')
        if self.attribute_aware_gate and not self.regional_full_200:
            raise ValueError('Attribute-aware gate requires regional full attributes')


def normalize_full_update_config(saved):
    """Resume new checkpoints; old full-update training checkpoints are replay-only."""
    saved = dict(saved)
    if (saved.get('learnable_gate', False) or saved.get('point_gate_enabled', False) or
            saved.get('gate_probability_weighted_proximal', False) or
            saved.get('temporal_focus_strength', 0.) != 0 or saved.get('tail_rmse_mix', 0.) != 0):
        raise ValueError('Learnable attribute-repair checkpoints are replay-only')
    for name in ('learnable_gate', 'gate_learning_rate', 'gate_temperature', 'gate_initial_logit',
                 'gate_rate_scale', 'gate_rate_model', 'gate_probability_weighted_proximal',
                 'point_gate_enabled', 'point_gate_learning_rate', 'point_gate_temperature',
                 'point_gate_initial_logit', 'temporal_focus_strength', 'temporal_focus_scale',
                 'temporal_focus_mode', 'tail_rmse_mix'):
        saved.pop(name, None)
    defaults = asdict(FullUpdateConfig())
    obsolete = set(saved) - set(defaults)
    if saved.get('enabled', False) and obsolete:
        raise ValueError('Historical full-update training checkpoints are replay-only')
    return {**defaults, **{key: value for key, value in saved.items() if key in defaults}}










class _QuantizedFullAttribute(torch.autograd.Function):
    """Preserve zero-reference bits while training both additive layers."""
    @staticmethod
    def forward(ctx, reference, delta, step):
        symbols = torch.round(delta / step)
        return torch.where(symbols.ne(0), reference + symbols * step, reference)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient, None












class ResidualAppearanceAttributes:
    """Three row-aligned delta tensors; optimizer state belongs to motion Adam."""
    def __init__(self, config):
        self.config = config
        self.steps = tuple(float(getattr(config, name + '_step')) for name in FIELDS)
        self.parameters = {}
        self._initial_zero_versions = None

    def setup(self, reference, optimizer, rates):
        _check_reference(reference)
        self.parameters = {name: nn.Parameter(torch.zeros_like(reference[name])) for name in FIELDS}
        self._initial_zero_versions = {name: value._version for name, value in self.parameters.items()}
        for name, rate in zip(FIELDS, rates):
            optimizer.add_param_group(dict(params=[self.parameters[name]], name='early_full_' + name,
                                           lr=rate, betas=(.9, .999), eps=1e-8))

    def apply(self, reference):
        result = {}
        for name, step in zip(FIELDS, self.steps):
            delta = self.parameters[name]
            result[name] = _QuantizedFullAttribute.apply(reference[name], delta, step)
        return result

    def initial_zero_frozen(self):
        versions = self._initial_zero_versions
        return (versions is not None and set(versions) == set(self.parameters) and
                all(not value.requires_grad and value._version == versions[name]
                    for name, value in self.parameters.items()))

    def canonical(self, reference):
        symbols = self.symbols()
        return _materialize(reference, symbols, self.steps)

    def point_rate_multiplier(self, name):
        if name == 'logscale':
            return self.config.point_logscale_rate_multiplier
        if name == 'sh_rest':
            return self.config.point_sh_rest_rate_multiplier
        return 1.0

    def symbols(self):
        # Match the quantizer used by the training forward on the parameter's
        # device. CPU and CUDA division can round differently at half bins.
        symbols = {name: torch.round(self.parameters[name].detach() / step).cpu()
                   for name, step in zip(FIELDS, self.steps)}
        return symbols

    @torch.no_grad()
    def rate_terms(self, byte_price, *, include_scalar_counts=True, include_symbols=False):
        from .storage_rate import point_full_attribute_rate_terms
        return point_full_attribute_rate_terms(self, byte_price, include_scalar_counts=include_scalar_counts, include_symbols=include_symbols)

    @torch.no_grad()
    def after_step(self, optimizer, byte_price, *, proximal):
        for name, step in zip(FIELDS, self.steps):
            parameter = self.parameters[name]
            parameter.clamp_(-32767 * step, 32767 * step)
            if not self.config.storage_loss_enabled or not parameter.numel():
                continue
            state = optimizer.state.get(parameter, {})
            if 'exp_avg_sq' not in state:
                continue
            group = next(g for g in optimizer.param_groups if g.get('name') == 'early_full_' + name)
            lr = group['lr']
            if lr <= 0:
                continue
            beta2 = group['betas'][1]
            updates = int(state['step'])
            metric = (state['exp_avg_sq'] / (1 - beta2 ** updates)).sqrt().add(group['eps'])
            if self.config.per_attribute_l1_weight:
                threshold = lr * self.config.per_attribute_l1_weight / parameter.numel() / metric
                parameter.copy_(parameter.sign() * (parameter.abs() - threshold).clamp_min(0))
            if proximal and byte_price > 0:
                matrix = parameter.reshape(len(parameter), -1)
                gain = (.5 * metric * parameter.square() / lr).reshape_as(matrix).sum(-1)
                active = torch.round(matrix / step).ne(0).any(-1)
                multiplier = self.point_rate_multiplier(name)
                price = raw_sparse_row_bytes(matrix.shape[1]) * byte_price * multiplier
                prune = active & gain.lt(price)
                matrix[prune] = 0

    @torch.no_grad()
    def retain(self, rows, optimizer):
        for name, previous in list(self.parameters.items()):
            parameter = nn.Parameter(previous.detach()[rows].clone(), requires_grad=previous.requires_grad)
            state = optimizer.state.pop(previous, {})
            if state:
                optimizer.state[parameter] = {key: value[rows].clone()
                    if torch.is_tensor(value) and value.shape == previous.shape else value
                    for key, value in state.items()}
            group = next(g for g in optimizer.param_groups if g.get('name') == 'early_full_' + name)
            group['params'] = [parameter]
            self.parameters[name] = parameter

    def capture(self, optimizer):
        groups = {}
        for name, parameter in self.parameters.items():
            group = next(g for g in optimizer.param_groups if g.get('name') == 'early_full_' + name)
            groups[name] = dict(parameter=parameter.detach().cpu().clone(),
                hyper={k: v for k, v in group.items() if k != 'params'},
                state={k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                       for k, v in optimizer.state.get(parameter, {}).items()})
        return dict(config=asdict(self.config), groups=groups)

    def restore(self, packet, optimizer, device, rows):
        # Restored deltas need not be zero, even when temporarily frozen.
        self._initial_zero_versions = None
        normalized = None if packet is None else normalize_full_update_config(packet.get('config', {}))
        if packet is None or normalized != asdict(self.config) or set(packet['groups']) != set(FIELDS):
            raise ValueError('Full attribute checkpoint is missing or incompatible')
        for name in FIELDS:
            saved = packet['groups'][name]
            value = saved['parameter'].to(device)
            if len(value) != rows:
                raise ValueError('Full attribute checkpoint survivor rows differ')
            param = nn.Parameter(value.detach().clone())
            self.parameters[name] = param
            optimizer.add_param_group(dict(params=[param], **saved['hyper']))
            if saved['state']:
                optimizer.state[param] = {k: v.to('cpu' if k == 'step' else device) if torch.is_tensor(v) else v
                                          for k, v in saved['state'].items()}

    def export(self, path, reference, frame_index, *, sparse_only=False):
        symbols = self.symbols()
        packet = encode_full_attribute_delta(reference, symbols, frame_index=frame_index, steps=self.steps, sparse_only=sparse_only)
        if packet is None:
            return None
        actual = decode_full_attribute_delta(packet, reference, frame_index=frame_index)
        expected = self.canonical(reference)
        if any(not torch.equal(actual[k], expected[k]) for k in FIELDS):
            raise ValueError('Full attributes differ from their packet')
        path = Path(path)
        path.write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), format=packet[:4].decode(),
                    fields=list(FIELDS), timing='after_regional_dc_before_births',
                    nonzero_symbols={k: int(v.ne(0).sum()) for k, v in symbols.items()})




@dataclass
class OpacityUpdateConfig:
    enabled: bool = False
    step: float = .05
    learning_rate: float = .05
    prune_threshold: float = .01
    proximal_regularization: bool = False
    mode: str = 'logit'
    l1_weight: float = 0.
    decay_with_motion: bool = False
    amplitude_refit_steps: int = 0
    joint_with_birth: bool = False

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('opacity_update.enabled must be boolean')
        if not isinstance(self.proximal_regularization, bool):
            raise ValueError('opacity_update.proximal_regularization must be boolean')
        if not isinstance(self.decay_with_motion, bool):
            raise ValueError('opacity_update.decay_with_motion must be boolean')
        if not isinstance(self.joint_with_birth, bool):
            raise ValueError('opacity_update.joint_with_birth must be boolean')
        if self.joint_with_birth and not self.enabled:
            raise ValueError('Birth opacity fitting requires opacity updates')
        if (isinstance(self.amplitude_refit_steps, bool) or
                not isinstance(self.amplitude_refit_steps, int) or self.amplitude_refit_steps < 0):
            raise ValueError('opacity_update.amplitude_refit_steps must be a nonnegative integer')
        if self.amplitude_refit_steps and not self.enabled:
            raise ValueError('Opacity amplitude refitting requires opacity updates')
        if (isinstance(self.l1_weight, bool) or not math.isfinite(self.l1_weight) or self.l1_weight < 0):
            raise ValueError('opacity_update.l1_weight must be finite and nonnegative')
        _step(self.step)
        if self.mode not in ('logit', 'alpha'):
            raise ValueError('Unknown opacity update mode')
        if (isinstance(self.learning_rate, bool) or not math.isfinite(self.learning_rate) or
                self.learning_rate <= 0):
            raise ValueError('opacity_update.learning_rate must be positive')
        if (isinstance(self.prune_threshold, bool) or not math.isfinite(self.prune_threshold) or
                not 0 <= self.prune_threshold < 1):
            raise ValueError('opacity_update.prune_threshold must be in [0, 1)')


class _QuantizedLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, delta, step):
        symbols = torch.round(delta / step)
        # Keep raw reference bits for zero symbols, including signed zero.
        return torch.where(symbols.ne(0), reference + symbols * step, reference)

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient, None


class _QuantizedAlpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, canonical_reference, render_reference, delta, step):
        symbols = torch.round(delta / step)
        target = (canonical_reference + symbols * step).clamp(0.,1.)
        return torch.where(symbols.ne(0),target,render_reference)

    @staticmethod
    def backward(ctx, gradient):
        return None,None,gradient,None


class _PreserveZeroLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, logits, zero):
        return torch.where(zero,reference,logits)

    @staticmethod
    def backward(ctx, gradient):
        return None,gradient,None




class QuantizedOpacityUpdate(nn.Module):
    """Fit old logits relative to the preceding decoded frame; freeze at birth."""
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.step = _step(config.step)
        self.reference = None
        self.delta = None
        self.rows = None
        self.optimizer = None
        self.committed = False
        self.refit_support = None

    def setup(self, reference):
        self.reference = reference.detach().clone()
        self.alpha_reference = (canonical_alpha(self.reference) if self.config.mode=='alpha' else None)
        self.render_alpha_reference = (self.reference.sigmoid() if self.config.mode=='alpha' else None)
        self.delta = nn.Parameter(torch.zeros_like(reference))
        self.rows = torch.arange(len(reference), device=reference.device)
        self.optimizer = torch.optim.Adam([self.delta], lr=self.config.learning_rate, eps=1e-8)
        self.committed = False
        self.refit_support = None

    @torch.no_grad()
    def prepare_step(self, step, motion_steps):
        """Finish selection before fitting amplitudes on the same sparse rows.

        The mask is a training constraint, not a transmitted list. Final nonzero
        symbols still use the ordinary opacity codec and may lose more rows.
        """
        count = self.config.amplitude_refit_steps
        if count >= motion_steps:
            raise ValueError('Opacity amplitude refitting must leave selection steps')
        if (not count or self.committed or self.refit_support is not None or
                step <= motion_steps - count or step > motion_steps):
            return
        self.refit_support = self.symbols().ne(0)
        self.delta.masked_fill_(~self.refit_support, 0)
        # Old sparse gradients/momentum must not affect the photo-only refit.
        self.optimizer.state.pop(self.delta, None)

    def forward(self, delta=None):
        delta = self.delta if delta is None else delta
        if self.config.mode=='alpha':
            alpha=self.alphas(delta)
            logits=torch.logit(alpha.clamp(1e-6,1-1e-6))
            logits=torch.where(alpha.le(0),logits.new_full((),-100.),
                torch.where(alpha.ge(1),logits.new_full((),100.),logits))
            zero=torch.round(delta.detach()/self.step)[self.rows].eq(0)
            return _PreserveZeroLogits.apply(self.reference[self.rows],logits,zero)
        return _QuantizedLogits.apply(self.reference, delta, self.step)[self.rows]

    def alphas(self, delta=None):
        delta = self.delta if delta is None else delta
        if self.config.mode=='alpha':
            return _QuantizedAlpha.apply(self.alpha_reference,self.render_alpha_reference,
                delta,self.step)[self.rows]
        return self(delta).sigmoid()

    def rate_terms(self, byte_price, width=1.):
        from .storage_rate import opacity_rate_terms
        return opacity_rate_terms(self, byte_price, width=width)

    @torch.no_grad()
    def optimizer_step(self, byte_price, proximal=False, lr_factor=1.,perform_step=True,external_state=None):
        learning_rate = self.config.learning_rate * lr_factor
        self.optimizer.param_groups[0]['lr'] = learning_rate
        if self.refit_support is not None and self.delta.grad is not None:
            self.delta.grad.mul_(self.refit_support)
        if perform_step:self.optimizer.step()
        self.delta.clamp_(min=-32767 * self.step, max=32767 * self.step)
        if self.config.mode=='alpha':
            self.delta.copy_(torch.minimum(torch.maximum(self.delta,-self.alpha_reference),1-self.alpha_reference))
        if self.refit_support is not None:
            self.delta.masked_fill_(~self.refit_support, 0)
        elif ((proximal and byte_price > 0) or
                (self.config.proximal_regularization and self.config.l1_weight > 0)):
            state = external_state if external_state is not None else self.optimizer.state.get(self.delta, {})
            if 'exp_avg_sq' in state:
                t = int(state['step'])
                metric = (state['exp_avg_sq'] / (1 - .999 ** t)).sqrt().add(1e-8)
                if self.config.proximal_regularization and self.config.l1_weight:
                    threshold = learning_rate * self.config.l1_weight / self.delta.numel() / metric
                    self.delta.copy_(self.delta.sign() * (self.delta.abs() - threshold).clamp_min(0))
                if proximal and byte_price > 0:
                    gain = .5 * metric * self.delta.square() / learning_rate
                    active = torch.round(self.delta / self.step).ne(0)
                    from .storage_rate import raw_sparse_row_bytes
                    prune = gain.lt(raw_sparse_row_bytes(1) * byte_price)
                    if not self.config.proximal_regularization:
                        prune &= active
                    self.delta[prune] = 0
        if perform_step:self.optimizer.zero_grad(set_to_none=True)

    def symbols(self):
        return torch.round(self.delta.detach() / self.step)

    @torch.no_grad()
    def canonical_logits(self, full=False):
        reference = self.reference.detach().cpu()
        symbols = self.symbols().cpu()
        if self.config.mode=='alpha':
            result=canonical_alpha_logits(reference,symbols,self.step)
            return result if full else result[self.rows.cpu()]
        result = reference.clone()
        active = symbols[:, 0].ne(0)
        result[active, 0] = reference[active, 0] + symbols[active, 0] * self.step
        return result if full else result[self.rows.cpu()]

    def commit(self, rows):
        self.rows = rows.detach().to(self.reference.device, dtype=torch.long).clone()
        self.committed = True
        self.delta.requires_grad_(self.config.joint_with_birth)

    def finish(self):
        self.delta.requires_grad_(False)

    @torch.no_grad()
    def export(self, path, frame_index):
        reference = self.reference[self.rows].cpu()
        symbols = self.symbols()[self.rows].cpu()
        packet = encode_opacity_delta(reference, symbols, frame_index=frame_index, step=self.step, mode=self.config.mode)
        if packet is None:
            return None
        decoded = decode_opacity_delta(packet, reference, frame_index=frame_index)
        if not torch.equal(decoded.logits, self.canonical_logits()):
            raise ValueError('Opacity packet differs from materialized quantized logits')
        path = Path(path)
        path.write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), rows=len(decoded.rows),
                    step=self.step, mode=self.config.mode, timing='before_births',
                    format=packet[:4].decode('ascii'),
                    raw_body_bytes=sum(len(_varint(int(gap))) + 2 for gap in
                        torch.diff(torch.cat((decoded.rows.new_tensor([-1]), decoded.rows))).tolist()))

    def capture(self):
        if self.reference is None:
            return None
        return dict(config=asdict(self.config), reference=self.reference,
                    delta=self.delta.detach(), rows=self.rows,
                    committed=self.committed, optimizer=self.optimizer.state_dict(),
                    refit_support=self.refit_support, trainable=self.delta.requires_grad)

    def restore(self, packet, device):
        normalized = None if packet is None else {**asdict(OpacityUpdateConfig()), **packet['config']}
        if packet is None or normalized != asdict(self.config):
            raise ValueError('Opacity checkpoint configuration differs or is missing')
        self.setup(packet['reference'].to(device))
        self.delta.data.copy_(packet['delta'].to(device))
        self.rows = packet['rows'].to(device)
        self.committed = bool(packet['committed'])
        self.delta.requires_grad_(packet.get('trainable', not self.committed))
        self.optimizer.load_state_dict(packet['optimizer'])
        support = packet.get('refit_support')
        if support is not None:
            if (not self.config.amplitude_refit_steps or support.dtype != torch.bool or
                    support.shape != self.delta.shape):
                raise ValueError('Invalid opacity amplitude-refit support')
            self.refit_support = support.to(device).clone()
            if bool(self.delta.detach()[~self.refit_support].ne(0).any()):
                raise ValueError('Opacity refit changed an excluded row')


class ResidualGSMotionAttributes(nn.Module):
    """Own and update each GS's translation and rotation residuals."""

    def __init__(self, reference):
        super().__init__()
        self.translation = nn.Parameter(torch.zeros_like(reference))
        self.rotation = nn.Parameter(torch.zeros_like(reference))

    @classmethod
    def from_tensors(cls, translation, rotation):
        if translation.shape != rotation.shape or translation.ndim != 2 or translation.shape[1] != 3:
            raise ValueError('GS residual motion checkpoint has invalid shapes')
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.translation = nn.Parameter(translation.detach().clone())
        result.rotation = nn.Parameter(rotation.detach().clone())
        return result

    @staticmethod
    def quantization_steps(reference, config, extent):
        return reference.new_tensor([config.gs_xyz_step_extent * extent] * 3 +
                                    [config.gs_rotation_step_rad] * 3)

    def optimizer_groups(self, config):
        return [dict(params=[self.translation], lr=config.gs_translation_lr),
                dict(params=[self.rotation], lr=config.gs_rotation_lr)]

    @torch.no_grad()
    def retain(self, rows, optimizer):
        for name, group_index in (('translation', 2), ('rotation', 3)):
            previous = getattr(self, name)
            parameter = nn.Parameter(previous.detach()[rows].clone(),
                                     requires_grad=previous.requires_grad)
            state = optimizer.state.pop(previous, None)
            optimizer.param_groups[group_index]['params'] = [parameter]
            if state is not None:
                optimizer.state[parameter] = {
                    key: value[rows].clone() if torch.is_tensor(value) and value.shape == previous.shape else value
                    for key, value in state.items()}
            setattr(self, name, parameter)

    def _private_residuals(self, motion, detach_proposal=False):
        """Coding error of an independent complete-motion proposal."""
        leaf = motion._private_nodes()
        values = (motion.gs_translation, motion.gs_rotation)
        return tuple(_quantized(value.detach() if detach_proposal else value, grid) - prediction[leaf]
                     for value, grid, prediction in zip(values,
                         (motion.gs_steps[:3], motion.gs_steps[3:]), motion._regional_predictions()))

    def _private_symbols(self, motion):
        leaf, support = motion._private_nodes(), motion._private_support()[:, None]
        result = []
        for proposal, prediction, grid in zip((motion.gs_translation, motion.gs_rotation),
                motion._regional_predictions(hard=True), (motion.gs_steps[:3], motion.gs_steps[3:])):
            symbols = (_integer(proposal, grid).to(torch.int32) - prediction[leaf].to(torch.int32)) * support
            if bool(((symbols < -32768) | (symbols > 32767)).any()):
                raise ValueError('Private coding residual exceeds int16 range')
            result.append(symbols.to(torch.int16))
        return tuple(result)

    def _gs_motion(self, motion, hard=False, relaxed=False):
        from .regional_attributes_update import _hard_value_soft_gradient
        if motion.private_full_motion:
            symbols = motion._private_symbols()
            if hard:
                return symbols
            values = (_quantized(motion.gs_translation, motion.gs_steps[:3]),
                      _quantized(motion.gs_rotation, motion.gs_steps[3:]))
            probability = motion._private_probabilities()[:, None]
            soft = tuple(value * probability for value in values)
            if relaxed:
                return soft
            return tuple(_hard_value_soft_gradient(symbol.to(value) * grid, value)
                         for symbol, value, grid in zip(symbols, soft,
                             (motion.gs_steps[:3], motion.gs_steps[3:])))
        quantize = _integer if hard else _quantized
        values = (quantize(motion.gs_translation, motion.gs_steps[:3]),
                  quantize(motion.gs_rotation, motion.gs_steps[3:]))
        if not motion.hierarchical_private:
            return values
        mask = motion._private_support()[:, None]
        if hard or motion._hard_gate_frozen:
            return tuple(value * mask for value in values)
        probability = motion._private_probabilities()[:, None]
        if relaxed:
            return tuple(value * probability for value in values)
        return tuple(_hard_value_soft_gradient(value * mask, value * probability) for value in values)

    @torch.no_grad()
    def _proximal_relative_gs(self, motion):
        """Adam-metric soft threshold for extent/radian-normalized residual L1.

        Unlike feeding sign gradients into Adam, this proximal map only moves
        a value towards zero and cannot jump across zero due to the penalty.
        The periodic block-L0 step uses the same photometric-only moments.
        """
        if (motion.shared_discovery_active or not motion.gs_sparsity_active or not motion.proximal_gs_regularization or
                not motion.config.relative_motion_loss_weight):
            return
        ramp = min(max((motion.last_step - motion.config.rate_warmup_steps) /
                       motion.config.rate_ramp_steps, 0.), 1.)
        coefficient = motion.config.relative_motion_loss_weight * ramp / len(motion.owner)
        if motion.hierarchical_private:
            coefficient = coefficient * motion._private_probabilities()[:, None]
        origins = motion._private_origins() if motion.private_full_motion else (None, None)
        for (parameter, group, scale), origin in zip((
                (motion.gs_translation, motion.optimizer.param_groups[2], motion.extent),
                (motion.gs_rotation, motion.optimizer.param_groups[3], 1.)), origins):
            state = motion.optimizer.state.get(parameter, {})
            if 'exp_avg_sq' not in state:
                continue
            t = int(state['step'])
            variance = state['exp_avg_sq'] / (1 - motion.config.adam_betas[1] ** t)
            threshold = (group['lr'] * coefficient / scale /
                         (variance.sqrt() + motion.config.adam_eps))
            residual = parameter if origin is None else parameter - origin
            shrunk = residual.sign() * (residual.abs() - threshold).clamp_min(0)
            parameter.copy_(shrunk if origin is None else origin + shrunk)
        if motion._motion_fit_window_due():
            motion.last_metrics['motion/prox_l1_zero_translation_components'] = int(motion.gs_translation.eq(0).sum())
            motion.last_metrics['motion/prox_l1_zero_rotation_components'] = int(motion.gs_rotation.eq(0).sum())

    @torch.no_grad()
    def _proximal_gs_blocks(self, motion):
        if motion.shared_discovery_active or not motion.gs_sparsity_active:
            return
        weight = motion._private_probabilities().detach() if motion.hierarchical_private else None
        origins = motion._private_origins() if motion.private_full_motion else None
        residuals = ((motion.gs_translation, motion.gs_rotation) if origins is None else
                     tuple(value - origin for value, origin in
                           zip((motion.gs_translation, motion.gs_rotation), origins)))
        if motion.hierarchical_private and 'exp_avg_sq' not in motion.optimizer.state.get(motion.gs_translation, {}):
            # Fresh photo moments are built on the first fixed-gate step.
            return
        if motion.config.rate_model != 'codec':
            if not motion.hierarchical_private and not motion.private_full_motion:
                return MotionBase._proximal_gs_blocks(motion)
            t, r = residuals
            at = _integer(t, motion.gs_steps[:3]).ne(0).any(-1).to(t)
            ar = _integer(r, motion.gs_steps[3:]).ne(0).any(-1).to(r)
            ids = motion._gs_id_lengths + 1
            from .storage_rate import raw_motion_block_bytes
            prices = [raw_motion_block_bytes(at, ar, ids),
                      raw_motion_block_bytes(at, ar * 0, ids),
                      raw_motion_block_bytes(at * 0, ar, ids)]
            groups = motion.optimizer.param_groups
            pt, pr = motion._proximal_blocks(motion.gs_translation, motion.gs_rotation,
                motion.gs_steps[:3], motion.gs_steps[3:],
                motion._gs_id_lengths, groups[2]['lr'], groups[3]['lr'], 0,
                block_bytes=([value * weight for value in prices] if weight is not None else prices),
                origins=origins)
            motion.last_metrics.update({'motion/prox_zero_translation_blocks': pt,
                                     'motion/prox_zero_rotation_blocks': pr})
            return
        motion._measure_codec_rate(motion.last_step, force=True)
        t, r = residuals[0] / motion.gs_steps[:3], residuals[1] / motion.gs_steps[3:]
        st, sr = t.round().ne(0).any(-1).to(t), r.round().ne(0).any(-1).to(r)
        zero_t, zero_r = torch.zeros_like(t), torch.zeros_like(r)
        rate = motion._codec_rate
        prices = [rate.costs(t, r, st, sr, 1),
                  rate.costs(t, zero_r, st, sr * 0, 1),
                  rate.costs(zero_t, r, st * 0, sr, 1)]
        prices = [value * rate.last_scale for value in prices]
        if weight is not None:
            prices = [value * weight for value in prices]
        groups = motion.optimizer.param_groups
        pt, pr = motion._proximal_blocks(motion.gs_translation, motion.gs_rotation,
            motion.gs_steps[:3], motion.gs_steps[3:], motion._gs_id_lengths,
            groups[2]['lr'], groups[3]['lr'], 0, block_bytes=prices, origins=origins)
        motion.last_metrics.update({'motion/prox_zero_translation_blocks': pt,
                                 'motion/prox_zero_rotation_blocks': pr})


@dataclass
class FeatureDCUpdateConfig:
    enabled: bool = False
    step: float = .01
    learning_rate: float = .0025
    l1_weight: float = .005
    decay_with_motion: bool = True
    joint_with_birth: bool = True
    sharing: str = 'per_gs'
    gate_joint: bool = False
    post_motion_only: bool = False
    tail_learning_rate_scale: float = 1.

    def scalar_config(self):
        return OpacityUpdateConfig(enabled=self.enabled, step=self.step,
            learning_rate=self.learning_rate, l1_weight=self.l1_weight,
            proximal_regularization=True, decay_with_motion=self.decay_with_motion,
            joint_with_birth=self.enabled and self.joint_with_birth, mode='logit', prune_threshold=0.)

    def validate(self):
        if not isinstance(self.post_motion_only, bool):
            raise ValueError('feature_dc_update.post_motion_only must be boolean')
        if not isinstance(self.gate_joint, bool):
            raise ValueError('feature_dc_update.gate_joint must be boolean')
        scale = self.tail_learning_rate_scale
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 < scale <= 1:
            raise ValueError('feature_dc_update.tail_learning_rate_scale must be in (0, 1]')
        if self.gate_joint and self.sharing != 'spatial_region':
            raise ValueError('Joint gate colors require spatial region sharing')
        if self.sharing not in ('per_gs', 'spatial_region'):
            raise ValueError('Unknown DC feature sharing mode')
        if self.post_motion_only and (not self.enabled or self.sharing != 'spatial_region' or
                                     self.gate_joint or not self.joint_with_birth):
            raise ValueError('Post-motion regional DC requires enabled spatial sharing, frozen gates and tail fitting')
        if not isinstance(self.joint_with_birth, bool):
            raise ValueError('feature_dc_update.joint_with_birth must be boolean')
        self.scalar_config().validate()


class QuantizedFeatureDCUpdate(nn.Module):
    """Per-GS DC increments. SDR1 regional packets are supported for decode only."""
    def __init__(self, config, motion=None):
        super().__init__()
        config.validate()
        if config.sharing != 'per_gs' or config.gate_joint:
            raise ValueError('Regional shared-colour training is retired')
        self.config = config
        self.scalar = QuantizedOpacityUpdate(config.scalar_config())

    @property
    def regional(self):
        return False

    def setup(self, reference):
        if reference.ndim != 3 or reference.shape[1:] != (1, 3):
            raise ValueError('DC features must have shape [N, 1, 3]')
        self.scalar.setup(reference.reshape(-1, 1))

    def forward(self, training=False):
        return self.scalar().reshape(-1, 1, 3)

    def canonical_features(self):
        return self.scalar.canonical_logits().reshape(-1, 1, 3)

    def rate_terms(self, byte_price, width=1.):
        from .storage_rate import point_dc_rate_terms
        return point_dc_rate_terms(self, byte_price, width=width)

    def optimizer_step(self, *args, **kwargs):
        self.scalar.optimizer_step(*args, **kwargs)

    def commit(self, rows):
        channels = torch.arange(3, device=rows.device)
        self.scalar.commit((rows[:, None] * 3 + channels).flatten())

    def finish(self):
        self.scalar.finish()

    def capture(self):
        if self.scalar.reference is None:
            return None
        return dict(config=asdict(self.config), scalar=self.scalar.capture())

    def restore(self, packet, device):
        normalized = None if packet is None else {**asdict(FeatureDCUpdateConfig()), **packet['config']}
        if packet is None or normalized != asdict(self.config):
            raise ValueError('DC feature checkpoint configuration differs or is missing')
        self.scalar.restore(packet['scalar'], device)
        if len(self.scalar.rows) % 3:
            raise ValueError('Incomplete RGB channel rows')
        rows = self.scalar.rows.reshape(-1, 3)
        if not torch.equal(rows, (rows[:, :1] // 3) * 3 + torch.arange(3, device=device)):
            raise ValueError('DC feature rows lost GS/channel alignment')

    @torch.no_grad()
    def export(self, path, frame_index):
        path = Path(path)
        info = self.scalar.export(path, frame_index)
        if info is None:
            return None
        packet = encode_feature_dc_packet(path.read_bytes())
        reference = self.scalar.reference[self.scalar.rows].reshape(-1, 1, 3)
        decoded = decode_feature_dc_delta(packet, reference, frame_index=frame_index)
        if not torch.equal(decoded, self.canonical_features()):
            raise ValueError('DC feature packet differs from materialized features')
        path.write_bytes(packet)
        return {**info, 'archive_bytes': len(packet), 'format': packet[:4].decode(),
                'channels': 3, 'sharing': 'per_gs', 'timing': 'before_births'}


@dataclass
class LogScaleUpdateConfig:
    enabled: bool = False
    step: float = .005
    learning_rate: float = .01
    l1_weight: float = .005
    max_rows: int = 1024
    selection_steps: int = 80
    refit_steps: int = 20
    proximal_interval: int = 20
    components: str = 'isotropic'

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('logscale_update.enabled must be boolean')
        for name, value, allow_zero in (
                ('step', self.step, False),
                ('learning_rate', self.learning_rate, False),
                ('l1_weight', self.l1_weight, True)):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
                qualifier = 'nonnegative' if allow_zero else 'positive'
                raise ValueError(f'logscale_update.{name} must be finite and {qualifier}')
        for name, value in (
                ('max_rows', self.max_rows),
                ('selection_steps', self.selection_steps),
                ('refit_steps', self.refit_steps),
                ('proximal_interval', self.proximal_interval)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'logscale_update.{name} must be a positive integer')
        if self.components not in ('isotropic', 'anisotropic'):
            raise ValueError('logscale_update.components must be isotropic or anisotropic')














class MotionSupportLogScaleUpdate(nn.Module):
    """Fit sparse int8 scale symbols on final point-motion support."""

    group_name = 'early_old_logscale'

    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.delta = None
        self.candidates = None
        self.survivor_rows = None
        self.support = None
        self.start_step = None
        self.finished = False

    @property
    def ready(self):
        return self.delta is not None

    @property
    def channels(self):
        return 1 if self.config.components == 'isotropic' else 3

    def reset(self):
        self.delta = None
        self.candidates = None
        self.survivor_rows = None
        self.support = None
        self.start_step = None
        self.finished = False

    def _group(self, optimizer):
        groups = [group for group in optimizer.param_groups
                  if group.get('name') == self.group_name]
        if len(groups) != 1:
            raise ValueError('Log-scale optimizer group is missing or duplicated')
        return groups[0]

    def setup(self, count, candidate_rows, optimizer, *, start_step):
        if self.ready:
            raise ValueError('Log-scale update is already initialized')
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('Log-scale reference count must be positive')
        rows = _rows(candidate_rows, count, name='Motion-supported log-scale rows')
        if any(group.get('name') == self.group_name for group in optimizer.param_groups):
            raise ValueError('Log-scale optimizer group already exists')
        device = optimizer.param_groups[0]['params'][0].device
        self.delta = nn.Parameter(torch.zeros(count, self.channels, device=device))
        self.candidates = torch.zeros(count, dtype=torch.bool, device=device)
        self.candidates[rows.to(device)] = True
        self.survivor_rows = torch.arange(count, device=device)
        self.support = None
        self.start_step = int(start_step)
        self.finished = False
        optimizer.add_param_group(dict(params=[self.delta], name=self.group_name,
            lr=self.config.learning_rate, betas=(.9, .999), eps=1e-8))

    def _active_mask(self):
        if not self.ready:
            raise ValueError('Log-scale update has not been initialized')
        return self.candidates if self.support is None else self.support

    def quantized_delta(self):
        mask = self._active_mask()[:, None]
        value = self.delta * mask
        symbols = torch.round(value / self.config.step).clamp(-127, 127)
        return value + (symbols * self.config.step - value).detach()

    def current_logscale(self, reference):
        if not self.ready:
            return reference
        if tuple(reference.shape) != (len(self.survivor_rows), 3):
            raise ValueError('Current log-scale reference differs from survivor layout')
        delta = self.quantized_delta()[self.survivor_rows]
        return reference + (delta.expand(-1, 3) if self.channels == 1 else delta)

    @torch.no_grad()
    def canonical_logscale(self, reference):
        value = self.current_logscale(reference)
        return value.detach()

    def rate_terms(self, byte_price):
        from .storage_rate import point_logscale_rate_terms
        return point_logscale_rate_terms(self, byte_price)

    @torch.no_grad()
    def before_optimizer_step(self):
        if self.delta is None or self.delta.grad is None:
            return
        self.delta.grad.mul_(self._active_mask()[:, None])

    @torch.no_grad()
    def after_optimizer_step(self, step, optimizer, byte_price):
        if not self.ready or self.finished:
            return
        local_step = int(step) - self.start_step
        if not 1 <= local_step <= self.config.selection_steps + self.config.refit_steps:
            raise ValueError('Log-scale optimizer step is outside its configured fit interval')
        self.delta.clamp_(-127 * self.config.step, 127 * self.config.step)
        self.delta.mul_(self._active_mask()[:, None])
        state = optimizer.state.get(self.delta, {})
        group = self._group(optimizer)
        if 'exp_avg_sq' in state:
            updates = int(state['step'])
            metric = (state['exp_avg_sq'] / (1 - .999 ** updates)).sqrt().add(1e-8)
            if self.config.l1_weight:
                threshold = (group['lr'] * self.config.l1_weight /
                             self.delta.numel() / metric)
                self.delta.copy_(self.delta.sign() *
                                 (self.delta.abs() - threshold).clamp_min(0))
            if (self.support is None and local_step % self.config.proximal_interval == 0 and
                    byte_price > 0):
                gain = .5 * metric * self.delta.square() / group['lr']
                prune = gain.lt(byte_price) & self.candidates[:, None]
                self.delta[prune] = 0
        if local_step == self.config.selection_steps:
            if 'exp_avg_sq' not in state:
                raise ValueError('Log-scale selection requires Adam second moments')
            live = torch.zeros_like(self.candidates)
            live[self.survivor_rows] = True
            eligible = self.candidates & live
            count = min(self.config.max_rows, int(eligible.sum()))
            support = torch.zeros_like(self.candidates)
            if count:
                updates = int(state['step'])
                metric = (state['exp_avg_sq'] / (1 - .999 ** updates)).sqrt().add(1e-8)
                gain = (.5 * metric * self.delta.square() / group['lr']).sum(-1)
                gain = gain.masked_fill(~eligible, -torch.inf)
                support[torch.topk(gain, count, sorted=False).indices] = True
            self.support = support
            self.delta.mul_(support[:, None])
            optimizer.state.pop(self.delta, None)
        elif self.support is not None:
            self.delta.mul_(self.support[:, None])

    @torch.no_grad()
    def retain_survivors(self, rows):
        if not self.ready:
            return
        rows = _rows(rows, len(self.survivor_rows), name='Log-scale survivor rows')
        if not len(rows):
            raise ValueError('Log-scale survivors must be nonempty')
        self.survivor_rows = self.survivor_rows[rows.to(self.survivor_rows.device)].clone()

    @torch.no_grad()
    def finish(self):
        if not self.ready:
            return
        if self.support is None:
            raise ValueError('Log-scale update finished before hard support selection')
        self.delta.requires_grad_(False)
        self.finished = True

    def current_symbols(self):
        if not self.ready:
            return None
        symbols = torch.round(self.delta.detach() / self.config.step).clamp(-127, 127)
        symbols = symbols.to(torch.int8) * self._active_mask()[:, None].to(torch.int8)
        symbols = symbols[self.survivor_rows]
        return symbols[:, 0] if self.channels == 1 else symbols

    def current_candidates(self):
        if not self.ready:
            return None
        return self.candidates[self.survivor_rows].nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def export(self, path, reference, *, frame_index, allowed_rows):
        if not self.ready:
            return None
        allowed_rows = _rows(allowed_rows, len(reference), name='Final motion support')
        candidates = self.current_candidates().cpu()
        if not torch.equal(candidates, allowed_rows):
            raise ValueError('Log-scale candidates differ from final point-motion support')
        symbols = self.current_symbols().cpu()
        packet = encode_logscale_delta(reference, symbols, allowed_rows=allowed_rows,
                                       frame_index=frame_index, step=self.config.step)
        if packet is None:
            return None
        decoded = decode_logscale_delta(packet, reference, allowed_rows=allowed_rows,
                                        frame_index=frame_index)
        if not torch.equal(decoded.logscale, self.canonical_logscale(reference).cpu()):
            raise ValueError('Log-scale packet differs from materialized old scales')
        path = Path(path)
        path.write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), rows=len(decoded.rows),
                    candidate_rows=len(allowed_rows), step=self.config.step,
                    format=packet[:4].decode('ascii'), timing='before_births',
                    isotropic=self.channels == 1, components=self.config.components,
                    symbol_dtype='int8')

    def capture(self, optimizer):
        if not self.ready:
            return None
        state = optimizer.state.get(self.delta, {})
        return dict(config=asdict(self.config), delta=self.delta.detach(),
                    candidates=self.candidates, survivor_rows=self.survivor_rows,
                    support=self.support, start_step=self.start_step,
                    finished=self.finished, trainable=self.delta.requires_grad,
                    optimizer_state={name: (value.detach().clone() if torch.is_tensor(value) else value)
                                     for name, value in state.items()})

    def restore(self, packet, device, optimizer):
        if packet is None:
            self.reset()
            return
        normalized = {**asdict(LogScaleUpdateConfig()), **packet.get('config', {})}
        if normalized != asdict(self.config):
            raise ValueError('Log-scale checkpoint configuration differs')
        delta = packet.get('delta')
        candidates = packet.get('candidates')
        survivors = packet.get('survivor_rows')
        support = packet.get('support')
        if (not torch.is_tensor(delta) or delta.ndim != 2 or delta.shape[1:] != (self.channels,) or
                not torch.is_tensor(candidates) or candidates.dtype != torch.bool or
                candidates.shape != delta.shape[:1] or not torch.is_tensor(survivors) or
                survivors.dtype != torch.long or survivors.ndim != 1 or
                (support is not None and (not torch.is_tensor(support) or
                 support.dtype != torch.bool or support.shape != candidates.shape))):
            raise ValueError('Invalid log-scale checkpoint layout')
        if any(group.get('name') == self.group_name for group in optimizer.param_groups):
            raise ValueError('Log-scale optimizer group already exists during restore')
        self.delta = nn.Parameter(delta.to(device).clone(),
                                  requires_grad=bool(packet.get('trainable', True)))
        self.candidates = candidates.to(device).clone()
        self.survivor_rows = survivors.to(device).clone()
        self.support = None if support is None else support.to(device).clone()
        self.start_step = int(packet['start_step'])
        self.finished = bool(packet.get('finished', False))
        optimizer.add_param_group(dict(params=[self.delta], name=self.group_name,
            lr=self.config.learning_rate, betas=(.9, .999), eps=1e-8))
        state = packet.get('optimizer_state', {})
        if state:
            optimizer.state[self.delta] = {
                name: (value.to(device).clone() if torch.is_tensor(value) else value)
                for name, value in state.items()}
        if (bool((self.survivor_rows < 0).any()) or
                bool((self.survivor_rows >= len(self.delta)).any()) or
                (len(self.survivor_rows) > 1 and
                 bool((self.survivor_rows[1:] <= self.survivor_rows[:-1]).any())) or
                (self.support is not None and bool((self.support & ~self.candidates).any())) or
                bool((self.delta.detach()[~self._active_mask()] != 0).any())):
            raise ValueError('Invalid log-scale checkpoint support state')
