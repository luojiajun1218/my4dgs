"""Motion-supported sparse log-scale residuals and their exact packet."""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import struct
import zlib

import torch
from torch import nn

from .rigid_tree_codec import _uvar


RAW_MAGIC = b'SLS1'
ZLIB_MAGIC = b'SLZ1'
ANISOTROPIC_RAW_MAGIC = b'SLA1'
ANISOTROPIC_ZLIB_MAGIC = b'ZLA1'
HEADER = struct.Struct('<4sIIId32sII')
MAX_BODY_BYTES = 16 * 1024 * 1024


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


def _rows(rows, count, *, name):
    rows = torch.as_tensor(rows).detach().cpu().contiguous().long()
    if (rows.ndim != 1 or bool(((rows < 0) | (rows >= count)).any()) or
            (len(rows) > 1 and bool((rows[1:] <= rows[:-1]).any()))):
        raise ValueError(f'{name} must be ordered unique in-range row IDs')
    return rows


def _reference_digest(reference, allowed_rows):
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    allowed_rows = _rows(allowed_rows, len(reference), name='Allowed log-scale rows')
    digest = hashlib.sha256(struct.pack('<II', len(reference), len(allowed_rows)))
    digest.update(reference.numpy().tobytes())
    digest.update(allowed_rows.numpy().tobytes())
    return digest.digest()


def _read_varint(body, cursor):
    value = 0
    for shift in range(0, 70, 7):
        if cursor >= len(body):
            raise ValueError('Truncated log-scale row gap')
        byte = body[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, cursor
    raise ValueError('Oversized log-scale row gap')


@dataclass(frozen=True)
class DecodedLogScaleUpdate:
    logscale: torch.Tensor
    rows: torch.Tensor
    symbols: torch.Tensor
    step: float
    prior: bytes


def encode_logscale_delta(reference, symbols, *, allowed_rows, frame_index, step):
    """Encode signed-int8 residuals on motion-supported old rows.

    A one-dimensional symbol tensor keeps the original isotropic stream
    bit-compatible.  A ``[N, 3]`` tensor stores one quantized residual per
    log-scale axis while retaining the same row support and reference binding.
    """
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    symbols = torch.as_tensor(symbols).detach().cpu().contiguous()
    if (reference.dtype != torch.float32 or reference.ndim != 2 or
            reference.shape[1:] != (3,) or not bool(torch.isfinite(reference).all())):
        raise ValueError('Log-scale reference must be finite float32 [N, 3]')
    if symbols.shape == (len(reference),):
        channels = 1
    elif symbols.shape == (len(reference), 3):
        channels = 3
    else:
        raise ValueError('Log-scale symbols must be aligned isotropic [N] or anisotropic [N, 3] integers')
    if (not bool(torch.isfinite(symbols).all()) or
            not torch.equal(symbols, symbols.round()) or bool(symbols.abs().gt(127).any())):
        raise ValueError('Log-scale symbols must be aligned signed-int8 integers')
    if (isinstance(frame_index, bool) or not isinstance(frame_index, int) or
            not 1 <= frame_index <= 0xffffffff):
        raise ValueError('Log-scale frame must be a positive uint32')
    if (isinstance(step, bool) or not isinstance(step, (int, float)) or
            not math.isfinite(step) or step <= 0):
        raise ValueError('Log-scale quantization step must be positive')
    allowed_rows = _rows(allowed_rows, len(reference), name='Allowed log-scale rows')
    active = (symbols.ne(0) if channels == 1 else symbols.ne(0).any(-1)).nonzero(as_tuple=True)[0]
    if len(active) and not bool(torch.isin(active, allowed_rows).all()):
        raise ValueError('Log-scale symbols extend outside transmitted motion support')
    if not len(active):
        return None
    body = bytearray()
    previous = -1
    for row in active.tolist():
        body.extend(_uvar(row - previous))
        values = symbols[row].reshape(-1).tolist()
        body.extend(struct.pack(f'<{channels}b', *(int(value) for value in values)))
        previous = row
    if len(body) > MAX_BODY_BYTES:
        raise ValueError('Log-scale packet body is too large')
    prior = _reference_digest(reference, allowed_rows)
    prefix = struct.pack('<IIId32sI', frame_index, len(reference), len(active),
                         float(step), prior, len(body))
    raw_magic = RAW_MAGIC if channels == 1 else ANISOTROPIC_RAW_MAGIC
    zlib_magic = ZLIB_MAGIC if channels == 1 else ANISOTROPIC_ZLIB_MAGIC
    # Keep legacy isotropic packets bit-compatible.  Bind the new component
    # count explicitly so changing only the four-byte magic cannot reinterpret
    # an isotropic body as an anisotropic one (or vice versa).
    checksum = zlib.crc32(prefix + (b'\x03' if channels == 3 else b'') + body)
    raw = HEADER.pack(raw_magic, frame_index, len(reference), len(active),
                      float(step), prior, len(body), checksum) + body
    compressed = HEADER.pack(zlib_magic, frame_index, len(reference), len(active),
                             float(step), prior, len(body), checksum) + zlib.compress(body, 9)
    packet = min((raw, compressed), key=len)
    decoded = decode_logscale_delta(packet, reference, allowed_rows=allowed_rows,
                                    frame_index=frame_index)
    expected = reference.clone()
    delta = symbols[active].to(reference.dtype) * float(step)
    expected[active] = reference[active] + (delta[:, None] if channels == 1 else delta)
    if not torch.equal(decoded.logscale, expected):
        raise ValueError('Log-scale packet differs from canonical quantized values')
    return packet


def decode_logscale_delta(packet, reference, *, allowed_rows, frame_index):
    packet = bytes(packet)
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    if (reference.dtype != torch.float32 or reference.ndim != 2 or
            reference.shape[1:] != (3,) or not bool(torch.isfinite(reference).all())):
        raise ValueError('Log-scale reference must be finite float32 [N, 3]')
    if len(packet) < HEADER.size:
        raise ValueError('Truncated log-scale packet')
    magic, index, count, nonzero, step, prior, length, checksum = HEADER.unpack_from(packet)
    magics = (RAW_MAGIC, ZLIB_MAGIC, ANISOTROPIC_RAW_MAGIC, ANISOTROPIC_ZLIB_MAGIC)
    channels = 1 if magic in (RAW_MAGIC, ZLIB_MAGIC) else 3
    if (magic not in magics or index != frame_index or count != len(reference) or
            not 1 <= nonzero <= count or not math.isfinite(step) or step <= 0 or
            length < (1 + channels) * nonzero or length > MAX_BODY_BYTES):
        raise ValueError('Log-scale metadata differs from the motion frame')
    encoded = packet[HEADER.size:]
    if magic in (RAW_MAGIC, ANISOTROPIC_RAW_MAGIC):
        body = encoded
    else:
        decoder = zlib.decompressobj()
        try:
            body = decoder.decompress(encoded, length + 1)
        except zlib.error as error:
            raise ValueError('Damaged compressed log-scale packet') from error
        if (len(body) != length or not decoder.eof or decoder.unused_data or
                decoder.unconsumed_tail):
            raise ValueError('Invalid compressed log-scale length or trailer')
    if len(body) != length:
        raise ValueError('Log-scale body length differs from its header')
    allowed_rows = _rows(allowed_rows, len(reference), name='Allowed log-scale rows')
    if prior != _reference_digest(reference, allowed_rows):
        raise ValueError('Log-scale reference or motion support differs')
    prefix = struct.pack('<IIId32sI', index, count, nonzero, step, prior, length)
    if zlib.crc32(prefix + (b'\x03' if channels == 3 else b'') + body) != checksum:
        raise ValueError('Log-scale checksum mismatch')
    rows, values, cursor, previous = [], [], 0, -1
    for _ in range(nonzero):
        gap, cursor = _read_varint(body, cursor)
        previous += gap
        if gap < 1 or previous >= count or cursor + channels > len(body):
            raise ValueError('Invalid log-scale update row')
        value = struct.unpack_from(f'<{channels}b', body, cursor)
        cursor += channels
        if not any(value):
            raise ValueError('Redundant zero log-scale update')
        rows.append(previous)
        values.append(value[0] if channels == 1 else value)
    if cursor != len(body):
        raise ValueError('Unexpected log-scale packet trailer')
    rows = torch.tensor(rows, dtype=torch.long)
    if not bool(torch.isin(rows, allowed_rows).all()):
        raise ValueError('Decoded log-scale rows extend outside motion support')
    symbols = torch.tensor(values, dtype=torch.int8)
    result = reference.clone()
    delta = symbols.to(reference.dtype) * step
    result[rows] = reference[rows] + (delta[:, None] if channels == 1 else delta)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Decoded log-scale update is nonfinite')
    return DecodedLogScaleUpdate(result, rows, symbols, float(step), prior)


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
        if not self.ready:
            zero = torch.zeros(())
            return dict(loss=zero, active_rows=zero, value_bytes_surrogate=zero)
        current = self.quantized_delta()[self.survivor_rows]
        active = current.ne(0).any(-1).sum()
        value_bytes = active.to(current) * self.channels
        # Selection is handled by proximal Adam and a hard row budget.  Keep
        # this literal forward price detached, matching the diagnostic fit.
        loss = value_bytes.detach() * byte_price
        return dict(loss=loss, active_rows=active, value_bytes_surrogate=value_bytes)

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
