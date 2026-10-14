"""Optional sparse old-GS opacity deltas in logit or activated-alpha coordinates.

Birth prediction must use the unchanged preceding appearance. Replay births
first, then apply this packet to the old rows only. No SH/scale field changes.
"""

from dataclasses import dataclass
import hashlib
import math
import struct
import zlib

import numpy as np
import torch
from torch import nn


MAGIC = b'OPD1'
ALPHA_MAGIC = b'OPD2'
SATURATED_ALPHA_MAGIC = b'OPD3'
MODES = {'logit': MAGIC, 'alpha': ALPHA_MAGIC, 'saturated_alpha': SATURATED_ALPHA_MAGIC}
# OPD3 uses the preceding raw logit to select coordinates, without a mask payload.
# Ordinary rows keep OPD1 units; saturated rows use alpha units scaled by 0.25.
# The fixed float32 raw threshold makes CPU/GPU eligibility independent of sigmoid rounding.
SATURATION_LOGIT_THRESHOLD = struct.unpack('<f', struct.pack('<f', math.log(.99 / .01)))[0]
SATURATED_ALPHA_SCALE = .25
# magic, frame, old count, nonzero count, float32 step, body length, CRC, prior hash
HEADER = struct.Struct('<4sIIIfII32s')
MAX_BODY_BYTES = 64 * 1024 * 1024


def _step(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Opacity step must be a positive finite float32')
    try:
        result = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    except (OverflowError, ValueError, TypeError, struct.error) as error:
        raise ValueError('Opacity step must be a positive finite float32') from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Opacity step must be a positive finite float32')
    return result


def _reference(value):
    value = torch.as_tensor(value).detach()
    if (value.dtype != torch.float32 or value.ndim != 2 or value.shape[1] != 1 or
            not bool(torch.isfinite(value).all())):
        raise ValueError('Opacity reference must be finite float32 [N, 1] logits')
    return value


def opacity_reference_hash(reference):
    value = _reference(reference).cpu().contiguous()
    return hashlib.sha256(struct.pack('<I', len(value)) +
                          value.numpy().astype('<f4', copy=False).tobytes()).digest()


def _frame(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 0xffffffff:
        raise ValueError('Opacity frame must be a positive uint32 integer')
    return value


def _varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _read_varint(body, offset):
    start, value = offset, 0
    for shift in range(0, 35, 7):
        if offset >= len(body):
            raise ValueError('Truncated opacity row index')
        byte = body[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            if value > 0xffffffff or _varint(value) != body[start:offset]:
                raise ValueError('Invalid opacity row index encoding')
            return value, offset
    raise ValueError('Invalid opacity row index encoding')


def encode_opacity_delta(symbols, reference, *, step, frame_index, mode='logit'):
    """Return actual sparse compressed bytes, or None for exactly zero updates."""
    reference, step, frame_index = _reference(reference), _step(step), _frame(frame_index)
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError('Opacity mode must be logit, alpha or saturated_alpha')
    magic = MODES[mode]
    if len(reference) > 0xffffffff:
        raise ValueError('Opacity reference has too many rows')
    symbols = torch.as_tensor(symbols).detach().cpu()
    if (symbols.dtype == torch.bool or symbols.is_complex() or
            tuple(symbols.shape) != tuple(reference.shape) or
            not bool(torch.isfinite(symbols).all()) or
            not torch.equal(symbols, symbols.round()) or
            bool(((symbols.double() < -32768) | (symbols.double() > 32767)).any())):
        raise ValueError('Opacity symbols must be aligned signed int16 integers')
    symbols = symbols.to(torch.int16).flatten()
    rows = symbols.ne(0).nonzero(as_tuple=True)[0].tolist()
    if not rows:
        return None
    body, previous = bytearray(), -1
    for row in rows:
        body.extend(_varint(row - previous))
        body.extend(struct.pack('<h', int(symbols[row])))
        previous = row
    if len(body) > MAX_BODY_BYTES:
        raise ValueError('Opacity body exceeds decoder size limit')
    prior = opacity_reference_hash(reference)
    prefix = HEADER.pack(magic, frame_index, len(reference), len(rows), step,
                         len(body), 0, prior)
    checksum = zlib.crc32(prefix + body)
    header = HEADER.pack(magic, frame_index, len(reference), len(rows), step,
                         len(body), checksum, prior)
    return header + zlib.compress(body, 3)


@dataclass
class DecodedOpacityDelta:
    logits: torch.Tensor
    rows: torch.Tensor
    symbols: torch.Tensor
    frame_index: int
    step: float
    mode: str = 'logit'


def decode_opacity_delta(packet, reference, *, frame_index=None):
    """Check the exact preceding logits before applying signed integer deltas."""
    reference = _reference(reference)
    packet = bytes(packet)
    if len(packet) < HEADER.size:
        raise ValueError('Truncated opacity header')
    magic, frame, count, nonzero, step, length, checksum, prior = HEADER.unpack_from(packet)
    if (magic not in MODES.values() or frame == 0 or count != len(reference) or
            not 1 <= nonzero <= count or not nonzero * 3 <= length <= nonzero * 7 or
            length > MAX_BODY_BYTES or not math.isfinite(step) or step <= 0):
        raise ValueError('Invalid opacity packet header')
    if frame_index is not None and _frame(frame_index) != frame:
        raise ValueError('Opacity frame does not match motion frame')
    if prior != opacity_reference_hash(reference):
        raise ValueError('Opacity reference hash mismatch')
    try:
        decoder = zlib.decompressobj()
        body = decoder.decompress(packet[HEADER.size:], length + 1)
    except zlib.error as error:
        raise ValueError('Damaged opacity compressed body') from error
    if (len(body) != length or not decoder.eof or decoder.unused_data or
            decoder.unconsumed_tail):
        raise ValueError('Opacity body length or trailer mismatch')
    prefix = HEADER.pack(magic, frame, count, nonzero, step, length, 0, prior)
    if zlib.crc32(prefix + body) != checksum:
        raise ValueError('Opacity checksum mismatch')
    rows, values, offset, previous = [], [], 0, -1
    for _ in range(nonzero):
        gap, offset = _read_varint(body, offset)
        row = previous + gap
        if gap < 1 or row >= count or offset + 2 > len(body):
            raise ValueError('Invalid, duplicate or truncated opacity row')
        value = struct.unpack_from('<h', body, offset)[0]
        if value == 0:
            raise ValueError('Sparse opacity packet contains a zero update')
        rows.append(row)
        values.append(value)
        offset += 2
        previous = row
    if offset != len(body):
        raise ValueError('Unexpected opacity body trailer')
    rows = torch.tensor(rows, dtype=torch.long, device=reference.device)
    values = torch.tensor(values, dtype=torch.int16, device=reference.device)
    logits = reference.clone()
    mode = next(mode for mode, identifier in MODES.items() if identifier == magic)
    if mode == 'alpha':
        logits[rows, 0] = _canonical_alpha_logits(reference[rows], values[:, None], step)[:, 0]
    elif mode == 'saturated_alpha':
        logits[rows, 0] = _canonical_saturated_alpha_logits(reference[rows], values[:, None], step)[:, 0]
    else:
        logits[rows, 0] = reference[rows, 0] + values.float() * step
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Decoded opacity logits are nonfinite')
    return DecodedOpacityDelta(logits, rows, values, int(frame), float(step), mode)


def apply_optional_opacity_delta(packet, reference, *, frame_index=None):
    """Legacy/zero-update frames carry no opacity packet and keep every logit."""
    if packet is None:
        return _reference(reference).clone()
    return decode_opacity_delta(packet, reference, frame_index=frame_index).logits


class SparseOpacityDelta(nn.Module):
    """Bidirectional quantized deltas relative to the actual preceding decode."""

    def __init__(self, reference, step=.01, mode='logit'):
        super().__init__()
        reference = _reference(reference)
        if not isinstance(mode, str) or mode not in MODES:
            raise ValueError('Opacity mode must be logit, alpha or saturated_alpha')
        self.mode = mode
        self.register_buffer('reference', reference.clone())
        self.register_buffer('step', reference.new_tensor(_step(step)))
        self.delta = nn.Parameter(torch.zeros_like(reference))
        # Derived buffers are rebuilt from the actual decoded reference on restore.
        self.register_buffer('_alpha', (_canonical_alpha(reference)
            if mode != 'logit' else None), persistent=False)
        self.register_buffer('_saturated', (reference.ge(SATURATION_LOGIT_THRESHOLD)
            if mode == 'saturated_alpha' else None), persistent=False)

    def symbols(self, training=True):
        continuous = self.delta / self.step
        hard = continuous.round()
        if self.mode != 'logit':
            alpha_step = self.step * (SATURATED_ALPHA_SCALE if self.mode == 'saturated_alpha' else 1.)
            bounded = hard.clamp(min=-32768, max=32767)
            bounded = torch.minimum(torch.maximum(bounded, (-self._alpha / alpha_step).floor()),
                                    ((1 - self._alpha) / alpha_step).ceil())
            hard = torch.where(self._saturated, bounded, hard) if self.mode == 'saturated_alpha' else bounded
        return hard.detach() + (continuous - continuous.detach()) if training else hard.detach()

    def forward(self, training=True):
        if self.mode == 'alpha':
            logits = _canonical_alpha_logits(self.reference, self.symbols(False), float(self.step))
            if training:
                return _AlphaLogitsSTE.apply(logits, self.delta)
            return logits
        if self.mode == 'saturated_alpha':
            logits = _canonical_saturated_alpha_logits(self.reference, self.symbols(False), float(self.step))
            return _SaturatedAlphaLogitsSTE.apply(logits, self.delta, self._saturated) if training else logits
        # Do not clip the preceding base; every change is represented by a symbol.
        if training:
            return _OpacityQuantizedLogits.apply(self.reference, self.delta, self.step)
        symbols = self.symbols(False)
        return torch.where(symbols.eq(0), self.reference,
                           self.reference + symbols * self.step)

    def activated(self, training=True):
        if self.mode == 'logit':
            return self(training).sigmoid()
        # Training uses exactly the alpha rendered from the CPU-canonical raw
        # logits that export installs. Its gradient is in alpha coordinates.
        alpha = self(False).sigmoid()
        if self.mode == 'saturated_alpha':
            return _SaturatedAlphaSTE.apply(alpha, self.delta, self._saturated) if training else alpha
        return _AlphaSTE.apply(alpha, self.delta) if training else alpha

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        if self.mode != 'logit':
            self._alpha.copy_(_canonical_alpha(self.reference))
        if self.mode == 'saturated_alpha':
            self._saturated.copy_(self.reference.ge(SATURATION_LOGIT_THRESHOLD))

    def packet(self, frame_index):
        return encode_opacity_delta(self.symbols(False), self.reference,
                                    step=float(self.step), frame_index=frame_index, mode=self.mode)

    @torch.no_grad()
    def proximal_zero(self, optimizer, byte_price):
        """Use the motion model's per-byte price and Adam-metric L0 decision.

        Absolute row-ID lengths are the same conservative proxy as motion;
        the exported packet uses exact positive gaps and measures real bytes.
        """
        if not math.isfinite(float(byte_price)) or byte_price < 0:
            raise ValueError('Opacity byte price must be finite and nonnegative')
        group = optimizer.param_groups[0]
        if (len(optimizer.param_groups) != 1 or len(group['params']) != 1 or
                group['params'][0] is not self.delta):
            raise ValueError('Opacity proximal requires its dedicated Adam parameter')
        limit = 32767 * self.step
        projected = int(self.delta.abs().gt(limit).sum())
        self.delta.clamp_(min=-limit, max=limit)
        if self.mode != 'logit':
            alpha_step = self.step * (SATURATED_ALPHA_SCALE if self.mode == 'saturated_alpha' else 1.)
            bounded = torch.minimum(torch.maximum(self.delta,
                (-self._alpha / alpha_step).floor() * self.step),
                ((1 - self._alpha) / alpha_step).ceil() * self.step)
            self.delta.copy_(torch.where(self._saturated, bounded, self.delta)
                             if self.mode == 'saturated_alpha' else bounded)
        state = optimizer.state.get(self.delta)
        if byte_price == 0 or not state or 'exp_avg_sq' not in state:
            return dict(projected_rows=projected, proximal_zeroed_rows=0,
                        byte_price=float(byte_price))
        variance = state['exp_avg_sq'] / (1 - group['betas'][1] ** int(state['step']))
        gain = .5 * ((variance.sqrt() + group['eps']) / group['lr']) * self.delta.square()
        rows = torch.arange(len(self.delta), device=self.delta.device, dtype=torch.long)
        id_bytes = torch.ones_like(rows)
        for shift in range(7, 35, 7):
            id_bytes += rows.ge(1 << shift)
        active = self.symbols(False).ne(0).flatten()
        drop = active & (gain.flatten() <= float(byte_price) * (id_bytes + 2))
        self.delta[drop] = 0
        return dict(projected_rows=projected, proximal_zeroed_rows=int(drop.sum()),
                    byte_price=float(byte_price))



def _canonical_alpha(reference):
    """One CPU definition avoids device-dependent inverse-logit state drift."""
    return reference.detach().cpu().sigmoid().to(reference.device)


def _canonical_alpha_logits(reference, symbols, step):
    # Zero symbols retain exact preceding raw bits, including signed zero.
    active = symbols.detach().ne(0).flatten().nonzero(as_tuple=True)[0]
    result = reference.detach().clone()
    if not len(active):
        return result
    selected = reference[active].detach().cpu()
    alpha = _canonical_alpha(selected).numpy()
    delta = symbols[active].detach().cpu().numpy().astype(np.float32) * np.float32(step)
    target = np.clip(alpha + delta, 0., 1.)
    logits = np.empty_like(target)
    low, high = target <= 0, target >= 1
    logits[low], logits[high] = -100., 100.
    middle = ~(low | high)
    mid = target[middle].astype(np.float64)
    logits[middle] = (np.log(mid) - np.log1p(-mid)).astype(np.float32)
    result[active] = torch.from_numpy(logits).to(reference.device)
    return result


def _canonical_saturated_alpha_logits(reference, symbols, step):
    active = symbols.detach().ne(0).flatten().nonzero(as_tuple=True)[0]
    result = reference.detach().clone()
    if not len(active):
        return result
    selected = reference[active].detach().cpu()
    values = symbols[active].detach().cpu()
    saturated = selected.ge(SATURATION_LOGIT_THRESHOLD).flatten()
    logits = selected + values.float() * _step(step)
    if saturated.any():
        logits[saturated] = _canonical_alpha_logits(selected[saturated], values[saturated],
            _step(_step(step) * SATURATED_ALPHA_SCALE))
    result[active] = logits.to(reference.device)
    return result


class _SaturatedAlphaSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, delta, saturated):
        ctx.save_for_backward(alpha, saturated)
        return alpha

    @staticmethod
    def backward(ctx, gradient):
        alpha, saturated = ctx.saved_tensors
        ordinary = torch.ops.aten.sigmoid_backward(gradient, alpha)
        return None, torch.where(saturated, gradient * SATURATED_ALPHA_SCALE, ordinary), None


class _SaturatedAlphaLogitsSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, delta, saturated):
        ctx.save_for_backward(logits.sigmoid(), saturated)
        return logits

    @staticmethod
    def backward(ctx, gradient):
        alpha, saturated = ctx.saved_tensors
        slope = torch.where(saturated, SATURATED_ALPHA_SCALE /
                            (alpha * (1 - alpha)).clamp_min(1e-8), 1.)
        return None, gradient * slope, None


class _AlphaSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, delta):
        return alpha

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient


class _AlphaLogitsSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, delta):
        ctx.save_for_backward(logits.sigmoid())
        return logits

    @staticmethod
    def backward(ctx, gradient):
        alpha, = ctx.saved_tensors
        return None, gradient / (alpha * (1 - alpha)).clamp_min(1e-8)


class _OpacityQuantizedLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, delta, step):
        symbols = (delta / step).round()
        # Inactive rows preserve their raw bits, including negative zero.
        return torch.where(symbols.eq(0), reference, reference + symbols * step)

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient, None


class QuantizedOldOpacityView:
    """Delegate frozen attributes while keeping opacity gradients out of caches."""

    def __init__(self, model, delta):
        if len(model.get_xyz) != len(delta.reference):
            raise ValueError('Old opacity view rows disagree with the frozen model')
        self.model, self.delta = model, delta

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def get_opacity(self):
        return self.delta.activated(torch.is_grad_enabled())
