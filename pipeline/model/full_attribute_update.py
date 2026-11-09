"""Keyframe old-GS attributes on the shared Adam and a checked real packet."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import struct
import zlib

import numpy as np
import torch
from torch import nn

from .sparse_integer_codec import _Reader, _read_unsigned_values, encode_unsigned_values
from .spatial_gated_motion import SpatialGatedMotion

FIELDS = ('logscale', 'sh_dc', 'sh_rest')
MAGIC = b'FUA1'
SPARSE_MAGIC = b'FUA2'
PLANAR_MAGIC = b'FUA3'
HEADER = struct.Struct('<4sIII3dI32sI')
MAX_BODY = 512 * 1024 * 1024


@dataclass
class FullUpdateConfig:
    enabled: bool = False
    gradient_position_lr_scale: float = 0.25
    audit_camera_sequence: bool = False
    gradient_diagnostics: bool = True
    logscale_step: float = .005
    sh_dc_step: float = .001
    sh_rest_step: float = .001
    storage_loss_enabled: bool = True
    per_attribute_l1_weight: float = .005
    point_dc_rate_scale: float = 1.0
    probe_views_per_round: int = 2
    probe_opacity_weight: bool = True
    point_budget_fraction: float | None = None
    repair_temporal_weight: float = 0.0
    repair_temporal_scale: float = 0.1
    repair_temporal_lag: int = 1
    repair_rmse_mix: float = 0.0
    motion_rmse_mix: float = 0.0
    early_joint_repair: bool = False
    early_probe_attribute_only: bool = False
    shared_region_orientation: bool = False

    # Read-only compatibility defaults for generic code and historical decoders.
    # These are not dataclass fields or selectable training options.
    interval = 10
    repair_only_after_motion = False
    train_attributes = True
    stop_after_motion = False
    color_only = False
    resume_from_finalized_200 = False
    train_point_dc = True
    point_training_schedule = 'standard'
    color_point_budget = 0
    regional_full_point_budget = 0
    sparse_color_stream_v2 = False
    sparse_color_logscale_v3 = False
    shared_dc_rate_scale = 1.0
    repair_selection = 'all_points'
    image_repair_priority = 'mean_error'
    image_repair_error_metric = 'l1'
    image_repair_error_threshold = .03
    image_repair_min_support = 8.
    image_repair_max_group_points = 64
    image_repair_max_points = 4096

    @property
    def regional_repair(self):
        return self.enabled

    def validate(self):
        for name in ('enabled', 'audit_camera_sequence', 'gradient_diagnostics', 'storage_loss_enabled', 'early_joint_repair', 'early_probe_attribute_only', 'probe_opacity_weight', 'shared_region_orientation'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'full_update.{name} must be boolean')
        for name in ('gradient_position_lr_scale', 'logscale_step', 'sh_dc_step', 'sh_rest_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'full_update.{name} must be finite and positive')
        if self.gradient_position_lr_scale > 1:
            raise ValueError('full_update.gradient_position_lr_scale must not exceed 1')
        for name in ('per_attribute_l1_weight', 'point_dc_rate_scale'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'full_update.{name} must be finite and nonnegative')
        if type(self.probe_views_per_round) is not int or self.probe_views_per_round not in (2, 4):
            raise ValueError('full_update.probe_views_per_round must be 2 or 4')
        if self.point_budget_fraction is not None and (
                isinstance(self.point_budget_fraction, bool) or
                not isinstance(self.point_budget_fraction, (float, int)) or
                not math.isfinite(self.point_budget_fraction) or
                not 0 < self.point_budget_fraction <= .01):
            raise ValueError('full_update.point_budget_fraction must be in (0, 0.01]')
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
        if self.early_probe_attribute_only and not self.early_joint_repair:
            raise ValueError('Attribute-only early probes require early joint repair')


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


def _reference_hash(reference):
    digest = hashlib.sha256()
    for name in FIELDS:
        value = reference[name].detach().cpu().contiguous()
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Full attribute references must be finite float32')
        digest.update(name.encode())
        digest.update(struct.pack('<I', value.ndim))
        digest.update(struct.pack('<' + 'I' * value.ndim, *value.shape))
        digest.update(value.numpy().tobytes())
    return digest.digest()


def _check_reference(reference):
    if set(reference) != set(FIELDS):
        raise ValueError('Full attribute reference fields differ')
    n = len(reference['logscale'])
    if (n < 1 or reference['logscale'].shape != (n, 3) or
            reference['sh_dc'].shape != (n, 1, 3) or
            reference['sh_rest'].ndim != 3 or reference['sh_rest'].shape[0] != n or
            reference['sh_rest'].shape[2] != 3):
        raise ValueError('Full attribute reference dimensions differ')
    return n, reference['sh_rest'].shape[1]


def _materialize(reference, symbols, steps):
    result = {}
    for name, step in zip(FIELDS, steps):
        value = reference[name].detach().cpu().contiguous()
        integer = torch.as_tensor(symbols[name]).to(dtype=value.dtype, device='cpu')
        result[name] = torch.where(integer.ne(0), value + integer * step, value)
    return result


def validate_sparse_color_selection(symbols, selected, budget, allowed_fields=('sh_dc',)):
    """Encoder-only check; a receiver cannot reconstruct the training selection."""
    selected = torch.as_tensor(selected).detach().cpu()
    if type(budget) is not int or budget < 1 or selected.dtype != torch.bool or selected.ndim != 1:
        raise ValueError('Invalid sparse color training selection')
    if set(symbols) != set(FIELDS) or any(len(v) != len(selected) for v in symbols.values()):
        raise ValueError('Sparse color selection row count differs')
    if allowed_fields not in (('sh_dc',), ('sh_dc', 'logscale')):
        raise ValueError('Invalid sparse attribute authorization')
    if int(selected.sum()) > budget:
        raise ValueError('Color training selection exceeds budget')
    for name, value in symbols.items():
        value = value.detach().cpu()
        if (name not in allowed_fields and value.ne(0).any()) or value[~selected].ne(0).any():
            raise ValueError('Sparse color symbols violate training selection or field authorization')


class _QuantizedFullAttribute(torch.autograd.Function):
    """Preserve zero-reference bits while training both additive layers."""
    @staticmethod
    def forward(ctx, reference, delta, step):
        symbols = torch.round(delta / step)
        return torch.where(symbols.ne(0), reference + symbols * step, reference)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient, None


def _sparse_body(arrays):
    # Three independent, sorted row-gap lists and their full signed-int16
    # attribute blocks. Gaps are stored separately for vectorized decoding.
    chunks = []
    for array in arrays:
        matrix = array.reshape(len(array), -1)
        rows = np.flatnonzero(np.any(matrix != 0, axis=1))
        gaps = np.diff(np.r_[-1, rows]).astype(np.uint32)
        chunks.append(struct.pack('<I', len(rows)))
        chunks.append(encode_unsigned_values(gaps))
        chunks.append(matrix[rows].astype('<i2', copy=False).tobytes())
    return b''.join(chunks)


def _sparse_zigzag_byteplane(body, count, rest, *, inverse=False):
    """Reversibly transform FUA2 int16 values, retaining canonical row gaps.

    FUA3 keeps each field's count and varint gaps unchanged. Its values are
    zigzagged uint16, with all low bytes followed by all high bytes. The body
    is exactly the same length as FUA2; the regular header and CRC protect it.
    """
    reader = _Reader(body)
    chunks = []
    for width in (3, 3, 3 * rest):
        prefix = bytes(reader.take(4))
        active = struct.unpack('<I', prefix)[0]
        if active > count or (width == 0 and active):
            raise ValueError('Invalid full attribute sparse row count')
        start = reader.cursor
        gaps = _read_unsigned_values(reader, active).astype(np.uint64)
        if bool((gaps == 0).any()) or (active and int(gaps.sum()) > count):
            raise ValueError('Invalid full attribute row gaps')
        chunks.extend((prefix, bytes(reader.data[start:reader.cursor])))
        components = active * width
        source = bytes(reader.take(components * 2))
        if inverse:
            plane = np.frombuffer(source, dtype=np.uint8)
            interleaved = np.empty((components, 2), dtype=np.uint8)
            interleaved[:, 0], interleaved[:, 1] = plane[:components], plane[components:]
            words = np.frombuffer(interleaved.tobytes(), dtype='<u2').astype(np.int32)
            values = ((words >> 1) ^ -(words & 1)).astype('<i2')
            converted = values.tobytes()
        else:
            values = np.frombuffer(source, dtype='<i2')
            signed = values.astype(np.int32)
            words = ((signed << 1) ^ (signed >> 15)).astype('<u2')
            plane = np.frombuffer(words.tobytes(), dtype=np.uint8).reshape(-1, 2)
            converted = plane[:, 0].tobytes() + plane[:, 1].tobytes()
        matrix = values.reshape(active, width)
        if active and (bool((matrix == -32768).any()) or bool((~np.any(matrix != 0, axis=1)).any())):
            raise ValueError('Invalid full attribute sparse symbols')
        chunks.append(converted)
    if reader.remaining:
        raise ValueError('Trailing full attribute sparse data')
    result = b''.join(chunks)
    if len(result) != len(body):
        raise ValueError('Invalid full attribute planar length')
    return result


def _encode_body(magic, body, frame_index, count, rest, steps, prior):
    prefix = HEADER.pack(magic, frame_index, count, rest, *steps, len(body), prior, 0)
    checksum = zlib.crc32(prefix + body)
    return HEADER.pack(magic, frame_index, count, rest, *steps, len(body), prior, checksum) + zlib.compress(body, 6)


def encode_full_attribute_delta(reference, symbols, *, frame_index, steps, sparse_only=False):
    reference = {k: v.detach().cpu().contiguous() for k, v in reference.items()}
    count, rest = _check_reference(reference)
    prior = _reference_hash(reference)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 1 <= frame_index <= 0xffffffff:
        raise ValueError('Invalid full attribute frame')
    if len(steps) != 3 or any(not math.isfinite(x) or x <= 0 for x in steps):
        raise ValueError('Invalid full attribute steps')
    chunks, arrays = [], []
    for name in FIELDS:
        value = torch.as_tensor(symbols[name]).detach().cpu()
        if (value.shape != reference[name].shape or not bool(torch.isfinite(value).all()) or
                not torch.equal(value, value.round()) or bool(value.abs().gt(32767).any())):
            raise ValueError(f'Invalid {name} integer symbols')
        array = value.to(torch.int16).contiguous().numpy().astype('<i2', copy=False)
        arrays.append(array)
        chunks.append(array.tobytes())
    body = b''.join(chunks)
    if len(body) > MAX_BODY:
        raise ValueError('Full attribute packet too large')
    if not any(any(chunk) for chunk in chunks):
        return None
    dense = _encode_body(MAGIC, body, frame_index, count, rest, steps, prior)
    sparse_body = _sparse_body(arrays)
    if len(sparse_body) > MAX_BODY and sparse_only:
        raise ValueError('Sparse full attribute packet too large')
    if len(sparse_body) > MAX_BODY:
        return dense
    sparse = _encode_body(SPARSE_MAGIC, sparse_body, frame_index, count, rest, steps, prior)
    planar_body = _sparse_zigzag_byteplane(sparse_body, count, rest)
    planar = _encode_body(PLANAR_MAGIC, planar_body, frame_index, count, rest, steps, prior)
    return min((sparse, planar) if sparse_only else (dense, sparse, planar), key=len)


def decode_full_attribute_delta(packet, reference, *, frame_index, color_only_budget=None,
                                allowed_fields=('sh_dc',)):
    reference = {k: v.detach().cpu().contiguous() for k, v in reference.items()}
    count, rest = _check_reference(reference)
    if len(packet) < HEADER.size:
        raise ValueError('Truncated full attribute packet')
    magic, frame, rows, coeff, *tail = HEADER.unpack_from(packet)
    if color_only_budget is not None:
        if isinstance(color_only_budget, bool) or not isinstance(color_only_budget, int) or color_only_budget < 0:
            raise ValueError('Invalid sparse color budget')
        if magic not in (SPARSE_MAGIC, PLANAR_MAGIC):
            raise ValueError('Sparse color stream requires FUA2 or FUA3')
        if allowed_fields not in (('sh_dc',), ('sh_dc', 'logscale')):
            raise ValueError('Invalid sparse attribute authorization')
    elif allowed_fields != ('sh_dc',):
        raise ValueError('Sparse field authorization requires a budget')
    steps, length, prior, checksum = tuple(tail[:3]), tail[3], tail[4], tail[5]
    expected = count * (3 + 3 + 3 * rest) * 2
    sparse_max = 12 + count * (15 + 2 * (6 + 3 * rest))
    if (magic not in (MAGIC, SPARSE_MAGIC, PLANAR_MAGIC) or frame != frame_index or rows != count or coeff != rest or
            (magic == MAGIC and length != expected) or
            (magic in (SPARSE_MAGIC, PLANAR_MAGIC) and not 12 <= length <= sparse_max) or length > MAX_BODY or
            any(not math.isfinite(v) or v <= 0 for v in steps) or prior != _reference_hash(reference)):
        raise ValueError('Full attribute packet metadata or reference differs')
    decoder = zlib.decompressobj()
    try:
        body = decoder.decompress(packet[HEADER.size:], length + 1)
    except zlib.error as error:
        raise ValueError('Damaged full attribute packet') from error
    if len(body) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Invalid full attribute compressed length')
    prefix = HEADER.pack(magic, frame, rows, coeff, *steps, length, prior, 0)
    if zlib.crc32(prefix + body) != checksum:
        raise ValueError('Full attribute checksum mismatch')
    if magic == PLANAR_MAGIC:
        body = _sparse_zigzag_byteplane(body, count, rest, inverse=True)
    if magic == MAGIC:
        symbols = {}
        offset = 0
        for name in FIELDS:
            shape = reference[name].shape
            size = math.prod(shape)
            symbols[name] = torch.from_numpy(np.frombuffer(body, dtype='<i2', count=size, offset=offset).copy().reshape(shape))
            offset += size * 2
        return _materialize(reference, symbols, steps)

    # Sparse streams contain complete rows only for changed points. Apply
    # their symbols directly, avoiding dense integer and float intermediates.
    # Keep torch.where so zero components preserve reference bits, including -0.
    reader = _Reader(body)
    result = {}
    active_union = np.zeros(count, dtype=np.bool_) if color_only_budget is not None else None
    for name, step in zip(FIELDS, steps):
        value = reference[name]
        shape = value.shape
        width = math.prod(shape[1:])
        active = struct.unpack('<I', reader.take(4))[0]
        if color_only_budget is not None and ((name not in allowed_fields and active) or active > color_only_budget):
            raise ValueError('Sparse packet exceeds budget or changes unauthorized fields')
        if active > count or (width == 0 and active):
            raise ValueError('Invalid full attribute sparse row count')
        gaps = _read_unsigned_values(reader, active).astype(np.uint64)
        if bool((gaps == 0).any()) or (active and int(gaps.sum()) > count):
            raise ValueError('Invalid full attribute row gaps')
        indices = np.cumsum(gaps, dtype=np.uint64).astype(np.int64) - 1
        if active_union is not None:
            active_union[indices] = True
            if int(active_union.sum()) > color_only_budget:
                raise ValueError('Sparse attribute union exceeds point budget')
        values = np.frombuffer(reader.take(active * width * 2), dtype='<i2').copy().reshape(active, width)
        if active and (bool((values == -32768).any()) or bool((~np.any(values != 0, axis=1)).any())):
            raise ValueError('Invalid full attribute sparse symbols')
        decoded = value.clone()
        if active:
            rows = torch.from_numpy(indices)
            integer = torch.from_numpy(values.reshape((active, *shape[1:]))).to(dtype=value.dtype)
            selected = value.index_select(0, rows)
            decoded.index_copy_(0, rows, torch.where(integer.ne(0), selected + integer * step, selected))
        result[name] = decoded
    if reader.remaining:
        raise ValueError('Trailing full attribute sparse data')
    if active_union is not None and not active_union.any():
        raise ValueError('Empty sparse attribute packet must be omitted')
    return result


class FullAttributeUpdate:
    """Three row-aligned delta tensors; optimizer state belongs to motion Adam."""
    def __init__(self, config):
        self.config = config
        self.steps = tuple(float(getattr(config, name + '_step')) for name in FIELDS)
        self.parameters = {}
        self._initial_zero_versions = None
        self.regional_scale_symbols = None

    def setup(self, reference, optimizer, rates):
        _check_reference(reference)
        self.parameters = {name: nn.Parameter(torch.zeros_like(reference[name])) for name in FIELDS}
        self._initial_zero_versions = {name: value._version for name, value in self.parameters.items()}
        for name, rate in zip(FIELDS, rates):
            optimizer.add_param_group(dict(params=[self.parameters[name]], name='early_full_' + name,
                                           lr=rate, betas=(.9, .999), eps=1e-8))

    def _scale_opposed(self):
        if self.regional_scale_symbols is None:
            return None
        shared = self.regional_scale_symbols()
        if shared is None:
            return None
        point = torch.round(self.parameters['logscale'].detach() / self.steps[0])
        return point * shared.detach().to(point) < 0

    def apply(self, reference):
        result = {}
        opposed = self._scale_opposed()
        for name, step in zip(FIELDS, self.steps):
            delta = self.parameters[name]
            if name == 'logscale' and opposed is not None:
                # Forward is zero for opposed symbols; keep an STE gradient.
                delta = torch.where(opposed, delta - delta.detach(), delta)
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

    def symbols(self):
        symbols = {name: torch.round(self.parameters[name].detach().cpu() / step)
                   for name, step in zip(FIELDS, self.steps)}
        opposed = self._scale_opposed()
        if opposed is not None:
            symbols['logscale'].masked_fill_(opposed.cpu(), 0)
        return symbols

    @torch.no_grad()
    def rate_terms(self, byte_price):
        """Report raw sparse-block price; Adam proximal performs its gradient step."""
        fields = {}
        for name, step in zip(FIELDS, self.steps):
            value = self.parameters[name].detach()
            matrix = value.reshape(len(value), -1)
            integers = torch.round(matrix / step)
            active = integers.ne(0).any(-1)
            rows = active.sum()
            # One-byte typical gap plus the full int16 attribute row. Actual
            # varint gaps, compression and FUA1/FUA2/FUA3 choice are measured.
            raw = rows.to(value) * (1 + 2 * matrix.shape[1])
            l1 = (self.config.per_attribute_l1_weight * value.abs().mean()
                  if self.config.storage_loss_enabled and value.numel() else value.sum() * 0)
            fields[name] = dict(active_rows=rows, active_scalars=integers.ne(0).sum(),
                                raw_bytes_surrogate=raw, l1_loss=l1)
        first = self.parameters[FIELDS[0]]
        raw = sum((item['raw_bytes_surrogate'] for item in fields.values()), first.new_zeros(()))
        raw = raw + raw.gt(0).to(raw) * (HEADER.size + 12)
        l1 = sum((item['l1_loss'] for item in fields.values()), first.new_zeros(()))
        loss = raw * byte_price + l1 if self.config.storage_loss_enabled else raw * 0
        return dict(loss=loss, raw_bytes_surrogate=raw, l1_loss=l1, fields=fields)

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
                price = (1 + 2 * matrix.shape[1]) * byte_price
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


class FullUpdateSpatialGatedMotion(SpatialGatedMotion):
    """Default motion behavior unless the current frame requests full updates."""
    regional_repair = False
    original_rotation_updates = False
    shared_region_orientation = False
    color_only = False

    @torch.no_grad()
    def _enforce_rotation_policy(self):
        if self.regional_repair and not self.original_rotation_updates:
            # Optionally fit only shared region orientation in motion fitting;
            # independent point rotations remain reserved for selected repair.
            if self.last_step <= 200:
                frozen = ((self.gs_rotation,) if self.shared_region_orientation else
                          (self.region_rotation, self.gs_rotation))
                for parameter in frozen:
                    parameter.zero_()
                    parameter.grad = None
                    parameter.requires_grad_(False)
                    self.optimizer.state.pop(parameter, None)
            return
        super()._enforce_rotation_policy()

    @torch.no_grad()
    def prepare_plan_step(self, step):
        if self.regional_repair and step > 200:
            self.last_step = int(step)
            self._hard_gate_frozen = True
            for parameter in (self.gate_logits, self.region_translation, self.region_rotation):
                parameter.requires_grad_(False)
                parameter.grad = None
            for parameter in (self.gs_translation, self.gs_rotation):
                parameter.requires_grad_(not self.color_only)
                if self.color_only:
                    parameter.grad = None
            return
        super().prepare_plan_step(step)

    @torch.no_grad()
    def optimizer_step(self):
        if self.regional_repair:
            if self.last_step <= 200:
                return super().optimizer_step()
            self.optimizer.step()
            if self.color_only:
                self.optimizer.zero_grad(set_to_none=True)
                return []
            for parameter, grid in ((self.gs_translation, self.gs_steps[:3]),
                                    (self.gs_rotation, self.gs_steps[3:])):
                parameter.clamp_(min=-32767 * grid, max=32767 * grid)
            self._proximal_relative_gs()
            if self._motion_fit_window_due() or self.last_step == 300:
                self._proximal_gs_blocks()
            self.optimizer.zero_grad(set_to_none=True)
            return []
        return super().optimizer_step()
