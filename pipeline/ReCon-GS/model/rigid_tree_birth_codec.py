"""Birth appearance stream; motion stream already carries pose and source row."""

from pathlib import Path
import math
import struct
import zlib

import numpy as np
import torch


MAGIC = b'RTB1'  # Legacy absolute float32 appearance.
HEADER = struct.Struct('<4sIIIII')
SYMBOL_MAGIC = b'RTB2'
# Magic, old/birth/rest counts, four symbol widths, four float32 steps,
# uncompressed body length, CRC32 of the header prefix and body.
SYMBOL_HEADER = struct.Struct('<4sIIH4B4fII')
MAX_BODY_BYTES = 256 * 1024 * 1024
ATTRIBUTES = ('logscale', 'opacity', 'sh_dc', 'sh_rest')


def _shapes(count, rest):
    return dict(logscale=(count, 3), opacity=(count, 1),
                sh_dc=(count, 1, 3), sh_rest=(count, rest, 3))


def _check_raw(raw, count, rest):
    for name, shape in _shapes(count, rest).items():
        value = raw[name]
        if (value.dtype != torch.float32 or tuple(value.shape) != shape or
                not bool(torch.isfinite(value).all())):
            raise ValueError(f'Birth {name} needs finite float32 raw values')


def _decompress(packet, offset, length):
    try:
        decoder = zlib.decompressobj()
        body = decoder.decompress(packet[offset:], length + 1)
    except zlib.error as error:
        raise ValueError('Damaged birth appearance stream') from error
    if (len(body) != length or not decoder.eof or
            decoder.unused_data or decoder.unconsumed_tail):
        raise ValueError('Birth appearance length or trailer mismatch')
    return body


def _sources(source_rows, count, base_count):
    if source_rows is None:
        raise ValueError('RTB2 requires motion birth source rows')
    rows = torch.as_tensor(source_rows).detach().cpu().contiguous()
    if (rows.ndim != 1 or len(rows) != count or
            rows.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) or
            bool(((rows < 0) | (rows >= base_count)).any())):
        raise ValueError('Invalid motion birth source rows')
    return rows.long()


def encode_birth_attributes(raw, base_count):
    """Write the original RTB1 format for old streams and exactness fallback."""
    count = len(raw['logscale'])
    if count < 1 or base_count < 1:
        raise ValueError('Birth appearance needs newborn rows and a base')
    rest = raw['sh_rest'].shape[1]
    _check_raw(raw, count, rest)
    body = b''.join(raw[name].detach().cpu().contiguous().numpy().astype(
        '<f4', copy=False).tobytes() for name in ATTRIBUTES)
    if len(body) > MAX_BODY_BYTES:
        raise ValueError('Birth appearance exceeds decoder size limit')
    return HEADER.pack(MAGIC, int(base_count), count, rest, len(body),
                       zlib.crc32(body)) + zlib.compress(body, 3)


def encode_birth_symbols(raw, base_count, *, symbols, steps, predictor,
                         source_rows):
    """Write actual quantizer symbols; return None if bytes cannot reconstruct.

    Only newborn tensors are copied to CPU. The caller retains the preceding
    GS attributes on the training device and supplies their gathered predictors.
    """
    count = len(raw['logscale'])
    if count < 1 or base_count < 1 or count > 0xffffffff or base_count > 0xffffffff:
        raise ValueError('Birth appearance needs valid newborn and base counts')
    rest = raw['sh_rest'].shape[1]
    if rest > 0xffff:
        raise ValueError('Too many SH coefficients for RTB2')
    _check_raw(raw, count, rest)
    _sources(source_rows, count, base_count)
    pieces, widths, packed_steps = [], [], []
    for name, shape in _shapes(count, rest).items():
        step = float(steps[name])
        if not math.isfinite(step) or step <= 0:
            raise ValueError(f'Invalid birth {name} quantization step')
        try:
            packed_step = struct.unpack('<f', struct.pack('<f', step))[0]
        except OverflowError as error:
            raise ValueError(f'Invalid float32 birth {name} step') from error
        if not math.isfinite(packed_step) or packed_step <= 0:
            raise ValueError(f'Invalid float32 birth {name} step')
        packed_steps.append(packed_step)
        symbol = torch.as_tensor(symbols[name]).detach().cpu().contiguous()
        pred = torch.as_tensor(predictor[name]).detach().cpu().contiguous()
        target = raw[name].detach().cpu().contiguous()
        if (tuple(symbol.shape) != shape or tuple(pred.shape) != shape or
                symbol.dtype != torch.float32 or pred.dtype != torch.float32 or
                not bool(torch.isfinite(symbol).all()) or
                not bool(torch.isfinite(pred).all())):
            raise ValueError(f'Invalid birth {name} symbols or predictor')
        if not torch.equal(symbol, symbol.round()):
            return None
        if symbol.numel() and bool(((symbol.double() < -(2 ** 31)) |
                                     (symbol.double() >= 2 ** 31)).any()):
            return None
        reconstructed = pred + symbol * packed_step
        if reconstructed.numpy().tobytes() != target.numpy().tobytes():
            return None
        width = (2 if not symbol.numel() or
                 bool(((symbol >= -(2 ** 15)) & (symbol < 2 ** 15)).all()) else 4)
        widths.append(width)
        dtype = '<i2' if width == 2 else '<i4'
        pieces.append(symbol.numpy().astype(dtype).tobytes())
    body = b''.join(pieces)
    if len(body) > MAX_BODY_BYTES:
        raise ValueError('Birth symbols exceed decoder size limit')
    prefix = SYMBOL_HEADER.pack(SYMBOL_MAGIC, base_count, count, rest,
                                *widths, *packed_steps, len(body), 0)[:-4]
    checksum = zlib.crc32(prefix + body)
    return prefix + struct.pack('<I', checksum) + zlib.compress(body, 3)


def export_birth_attributes(path, raw, base_count, *, symbols=None, steps=None,
                            predictor=None, source_rows=None):
    if any(value is not None for value in (symbols, steps, predictor, source_rows)):
        if any(value is None for value in (symbols, steps, predictor, source_rows)):
            raise ValueError('RTB2 requires symbols, steps, predictor and sources')
        packet = encode_birth_symbols(raw, base_count, symbols=symbols,
            steps=steps, predictor=predictor, source_rows=source_rows)
    else:
        packet = None
    fallback = packet is None
    if fallback:
        packet = encode_birth_attributes(raw, base_count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packet)
    return dict(path=str(path), archive_bytes=len(packet),
                rows=len(raw['logscale']),
                format=('rigid_birth_appearance_zlib_v1' if fallback else
                        'rigid_birth_source_symbols_zlib_v2'),
                exactness_fallback=fallback)


def decode_birth_attributes(packet, base, decoded_xyz, decoded_rotation,
                            birth_sources=None):
    packet = bytes(packet)
    magic = packet[:4]
    if magic not in (MAGIC, SYMBOL_MAGIC):
        raise ValueError('Unsupported birth appearance stream')
    if magic == MAGIC:
        if len(packet) < HEADER.size:
            raise ValueError('Truncated birth appearance stream')
        _, base_count, count, rest, length, checksum = HEADER.unpack_from(packet)
        expected = count * (7 + 3 * rest) * 4
        offset = HEADER.size
        if (base_count != len(base['xyz']) or rest != base['sh_rest'].shape[1] or
                count < 1 or length != expected or length > MAX_BODY_BYTES):
            raise ValueError('Invalid birth appearance header')
        widths = (4,) * 4
        steps = None
        source = None
    else:
        if len(packet) < SYMBOL_HEADER.size:
            raise ValueError('Truncated birth symbol stream')
        fields = SYMBOL_HEADER.unpack_from(packet)
        _, base_count, count, rest = fields[:4]
        widths, steps, length, checksum = fields[4:8], fields[8:12], fields[12], fields[13]
        offset = SYMBOL_HEADER.size
        expected = count * (3 * widths[0] + widths[1] +
                            3 * widths[2] + 3 * rest * widths[3])
        if (base_count != len(base['xyz']) or rest != base['sh_rest'].shape[1] or
                count < 1 or any(width not in (2, 4) for width in widths) or
                any(not math.isfinite(step) or step <= 0 for step in steps) or
                length != expected or length > MAX_BODY_BYTES):
            raise ValueError('Invalid birth symbol header')
        source = _sources(birth_sources, count, base_count)
    if (tuple(decoded_xyz.shape) != (count, 3) or
            tuple(decoded_rotation.shape) != (count, 4)):
        raise ValueError('Motion and birth appearance counts disagree')
    for name, shape in _shapes(base_count, rest).items():
        base_value = base[name]
        if (base_value.dtype != torch.float32 or tuple(base_value.shape) != shape or
                not bool(torch.isfinite(base_value).all())):
            raise ValueError(f'Invalid base {name} for birth appearance')
    body = _decompress(packet, offset, length)
    checked = zlib.crc32((packet[:offset - 4] if magic == SYMBOL_MAGIC else b'') + body)
    if checked != checksum:
        raise ValueError('Birth appearance checksum mismatch')
    shapes = _shapes(count, rest)
    position = 0
    raw = dict(xyz=decoded_xyz.clone(), rotation=decoded_rotation.clone())
    for index, name in enumerate(ATTRIBUTES):
        shape = shapes[name]
        elements = int(np.prod(shape))
        dtype = '<i2' if widths[index] == 2 else '<i4'
        if magic == MAGIC:
            dtype = '<f4'
        array = np.frombuffer(body, dtype=dtype, count=elements,
                              offset=position).reshape(shape).copy()
        value = torch.as_tensor(array, device=base[name].device)
        if magic == SYMBOL_MAGIC:
            value = base[name][source.to(base[name].device)] + value.float() * steps[index]
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite reconstructed birth {name}')
        elif not np.isfinite(array).all():
            raise ValueError(f'Nonfinite birth {name}')
        raw[name] = value.to(dtype=base[name].dtype)
        position += elements * widths[index]
    return raw
