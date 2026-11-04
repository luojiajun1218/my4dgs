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
    interval: int = 10
    repair_only_after_motion: bool = False
    regional_repair: bool = False
    logscale_step: float = .005
    sh_dc_step: float = .001
    sh_rest_step: float = .001
    storage_loss_enabled: bool = True
    per_attribute_l1_weight: float = .005

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('full_update.enabled must be boolean')
        if not isinstance(self.repair_only_after_motion, bool):
            raise ValueError('full_update.repair_only_after_motion must be boolean')
        if not isinstance(self.regional_repair, bool):
            raise ValueError('full_update.regional_repair must be boolean')
        if self.regional_repair and (not self.enabled or self.repair_only_after_motion):
            raise ValueError('Regional repair requires full_update.enabled and no legacy repair-only mode')
        if not isinstance(self.storage_loss_enabled, bool):
            raise ValueError('full_update.storage_loss_enabled must be boolean')
        weight = self.per_attribute_l1_weight
        if isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('full_update.per_attribute_l1_weight must be finite and nonnegative')
        if isinstance(self.interval, bool) or not isinstance(self.interval, int) or self.interval < 1:
            raise ValueError('full_update.interval must be a positive integer')
        for name in ('logscale_step', 'sh_dc_step', 'sh_rest_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'full_update.{name} must be finite and positive')


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


def encode_full_attribute_delta(reference, symbols, *, frame_index, steps):
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
    if len(sparse_body) > MAX_BODY:
        return dense
    sparse = _encode_body(SPARSE_MAGIC, sparse_body, frame_index, count, rest, steps, prior)
    planar_body = _sparse_zigzag_byteplane(sparse_body, count, rest)
    planar = _encode_body(PLANAR_MAGIC, planar_body, frame_index, count, rest, steps, prior)
    return min((dense, sparse, planar), key=len)


def decode_full_attribute_delta(packet, reference, *, frame_index):
    reference = {k: v.detach().cpu().contiguous() for k, v in reference.items()}
    count, rest = _check_reference(reference)
    if len(packet) < HEADER.size:
        raise ValueError('Truncated full attribute packet')
    magic, frame, rows, coeff, *tail = HEADER.unpack_from(packet)
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
    symbols = {}
    if magic == MAGIC:
        offset = 0
        for name in FIELDS:
            shape = reference[name].shape
            size = math.prod(shape)
            symbols[name] = torch.from_numpy(np.frombuffer(body, dtype='<i2', count=size, offset=offset).copy().reshape(shape))
            offset += size * 2
    else:
        reader = _Reader(body)
        for name in FIELDS:
            shape = reference[name].shape
            width = math.prod(shape[1:])
            active = struct.unpack('<I', reader.take(4))[0]
            if active > count or (width == 0 and active):
                raise ValueError('Invalid full attribute sparse row count')
            gaps = _read_unsigned_values(reader, active).astype(np.uint64)
            if bool((gaps == 0).any()) or (active and int(gaps.sum()) > count):
                raise ValueError('Invalid full attribute row gaps')
            indices = np.cumsum(gaps, dtype=np.uint64).astype(np.int64) - 1
            values = np.frombuffer(reader.take(active * width * 2), dtype='<i2').copy().reshape(active, width)
            if active and (bool((values == -32768).any()) or bool((~np.any(values != 0, axis=1)).any())):
                raise ValueError('Invalid full attribute sparse symbols')
            matrix = np.zeros((count, width), dtype='<i2')
            matrix[indices] = values
            symbols[name] = torch.from_numpy(matrix.reshape(shape))
        if reader.remaining:
            raise ValueError('Trailing full attribute sparse data')
    return _materialize(reference, symbols, steps)


class FullAttributeUpdate:
    """Three row-aligned delta tensors; optimizer state belongs to motion Adam."""
    def __init__(self, config):
        self.config = config
        self.steps = tuple(float(getattr(config, name + '_step')) for name in FIELDS)
        self.parameters = {}

    def setup(self, reference, optimizer, rates):
        _check_reference(reference)
        self.parameters = {name: nn.Parameter(torch.zeros_like(reference[name])) for name in FIELDS}
        for name, rate in zip(FIELDS, rates):
            optimizer.add_param_group(dict(params=[self.parameters[name]], name='early_full_' + name,
                                           lr=rate, betas=(.9, .999), eps=1e-8))

    def apply(self, reference):
        return {name: _QuantizedFullAttribute.apply(reference[name], self.parameters[name], step)
                for name, step in zip(FIELDS, self.steps)}

    def canonical(self, reference):
        symbols = self.symbols()
        return _materialize(reference, symbols, self.steps)

    def symbols(self):
        return {name: torch.round(self.parameters[name].detach().cpu() / step)
                for name, step in zip(FIELDS, self.steps)}

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
                prune = active & gain.lt((1 + 2 * matrix.shape[1]) * byte_price)
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
        normalized = None if packet is None else {**asdict(FullUpdateConfig()), **packet.get('config', {})}
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

    def export(self, path, reference, frame_index):
        symbols = self.symbols()
        packet = encode_full_attribute_delta(reference, symbols, frame_index=frame_index, steps=self.steps)
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

    @torch.no_grad()
    def _enforce_rotation_policy(self):
        if self.regional_repair and not self.original_rotation_updates:
            # Rotation is enabled in the codec, but before step 201 the
            # original translation-only motion policy must still apply.
            if self.last_step <= 200:
                for parameter in (self.region_rotation, self.gs_rotation):
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
                parameter.requires_grad_(True)
            return
        super().prepare_plan_step(step)

    @torch.no_grad()
    def optimizer_step(self):
        if self.regional_repair:
            if self.last_step <= 200:
                return super().optimizer_step()
            self.optimizer.step()
            for parameter, grid in ((self.gs_translation, self.gs_steps[:3]),
                                    (self.gs_rotation, self.gs_steps[3:])):
                parameter.clamp_(min=-32767 * grid, max=32767 * grid)
            self._proximal_relative_gs()
            if self._motion_fit_window_due() or self.last_step == 300:
                self._proximal_gs_blocks()
            self.optimizer.zero_grad(set_to_none=True)
            return []
        return super().optimizer_step()
