"""Exact opacity packet decoder, including historical quantization modes."""

from dataclasses import dataclass
import hashlib
import math
import struct
import zlib

import numpy as np
import torch
from .sparse_integer_codec import (OPACITY_MAGICS, unpack_layout, decode_opacity_layout)

MAGIC = b'OPD1'
ALPHA_MAGIC = b'OPD2'
SATURATED_ALPHA_MAGIC = b'OPD3'
CANONICAL_ALPHA_MAGIC = b'OPD4'
MODES = {'logit': MAGIC, 'alpha': ALPHA_MAGIC, 'saturated_alpha': SATURATED_ALPHA_MAGIC,
         'alpha_canonical': CANONICAL_ALPHA_MAGIC}
SATURATION_LOGIT_THRESHOLD = struct.unpack('<f', struct.pack('<f', math.log(.99 / .01)))[0]
SATURATED_ALPHA_SCALE = .25
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


def _varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


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
    layout_body = None
    parsed = None
    if packet[:4] in OPACITY_MAGICS:
        layout, raw_length = unpack_layout(packet, OPACITY_MAGICS,
                                           max_raw_bytes=HEADER.size + MAX_BODY_BYTES)
        # The planar decoder already validates and materializes its rows and
        # symbols. Reuse them only after the original packet CRC is checked.
        parsed = {}
        raw = decode_opacity_layout(layout, raw_length, parsed=parsed)
        packet, layout_body = raw[:HEADER.size], raw[HEADER.size:]
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
    if layout_body is None:
        try:
            decoder = zlib.decompressobj()
            body = decoder.decompress(packet[HEADER.size:], length + 1)
        except zlib.error as error:
            raise ValueError('Damaged opacity compressed body') from error
        if (len(body) != length or not decoder.eof or decoder.unused_data or
                decoder.unconsumed_tail):
            raise ValueError('Opacity body length or trailer mismatch')
    else:
        body = layout_body
    prefix = HEADER.pack(magic, frame, count, nonzero, step, length, 0, prior)
    if zlib.crc32(prefix + body) != checksum:
        raise ValueError('Opacity checksum mismatch')
    if parsed is None:
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
    else:
        rows = torch.from_numpy(parsed['rows']).to(device=reference.device)
        values = torch.from_numpy(parsed['symbols'].reshape(-1)).to(device=reference.device)
    logits = reference.clone()
    mode = next(mode for mode, identifier in MODES.items() if identifier == magic)
    if mode == 'alpha_canonical':
        logits[rows, 0] = canonical_alpha_logits(reference[rows], values[:, None], step)[:, 0]
    elif mode == 'alpha':
        logits[rows, 0] = _canonical_alpha_logits(reference[rows], values[:, None], step)[:, 0]
    elif mode == 'saturated_alpha':
        logits[rows, 0] = _canonical_saturated_alpha_logits(reference[rows], values[:, None], step)[:, 0]
    else:
        logits[rows, 0] = reference[rows, 0] + values.float() * step
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Decoded opacity logits are nonfinite')
    return DecodedOpacityDelta(logits, rows, values, int(frame), float(step), mode)


def _canonical_alpha(reference):
    """One CPU definition avoids device-dependent inverse-logit state drift."""
    return reference.detach().cpu().sigmoid().to(reference.device)


def canonical_alpha(reference):
    """OPD4 alpha: float64 sigmoid rounded once, independent of SIMD tails.

    Historical OPD2/3 keep their original torch CPU sigmoid definition above.
    Their vector and scalar tails can differ by an ULP after row selection.
    """
    value = reference.detach().cpu().numpy().astype(np.float64)
    exponential = np.exp(-np.abs(value))
    result = np.where(value >= 0, 1 / (1 + exponential),
                      exponential / (1 + exponential)).astype(np.float32)
    return torch.from_numpy(result).to(reference.device)


def canonical_alpha_logits(reference, symbols, step):
    active = symbols.detach().ne(0).flatten().nonzero(as_tuple=True)[0]
    result = reference.detach().clone()
    if not len(active):
        return result
    alpha = canonical_alpha(reference[active]).cpu().numpy()
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
