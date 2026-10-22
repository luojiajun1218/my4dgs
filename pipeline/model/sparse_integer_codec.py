"""Reversible planar/zigzag layouts for existing spatial and opacity bytes.

Quantizers and decoded float32 state are unchanged. New envelopes carry the
original byte count and transform byte count; original SHA/CRC checks remain
the authority after undoing the layout.
"""
import lzma
import struct
import zlib

import numpy as np

MOTION_MAGICS = (b'SVI1', b'SVZ1', b'SVX1')
GATE_DELTA_MAGICS = (b'SPG5', b'SPG6', b'SPG7', b'SPG8')
TRANSLATION_ONLY_SPATIAL_MAGICS = (b'SPG3', b'SPG4', b'SPG7', b'SPG8')
OPACITY_MAGICS = (b'OVI1', b'OVZ1', b'OVX1')
OPACITY_HEADER = struct.Struct('<4sIIIfII32s')
MAX_LZMA_MEMORY = 128 * 1024 * 1024


def _varint(value):
    value = int(value)
    if value < 0:
        raise ValueError('Sparse integers must be nonnegative')
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def encode_unsigned_values(values):
    """Canonical uint32 varints, vectorized without per-symbol Python calls."""
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    if not len(values):
        return b''
    maximum = int(values.max())
    if maximum > 0xffffffff:
        raise ValueError('Sparse symbol exceeds uint32')
    width = max(1, (maximum.bit_length() + 6) // 7)
    shifts = np.arange(width, dtype=np.uint64) * 7
    used = values[:, None] >= (np.uint64(1) << shifts[None])
    used[:, 0] = True
    continuation = values[:, None] >= (np.uint64(1) << (shifts[None] + 7))
    words = (((values[:, None] >> shifts[None]) & 127) | (continuation.astype(np.uint64) << 7)).astype(np.uint8)
    return words[used].tobytes()


class _Reader:
    def __init__(self, data):
        self.data = memoryview(data)
        self.cursor = 0

    @property
    def remaining(self):
        return len(self.data) - self.cursor

    def take(self, size):
        if size < 0 or size > self.remaining:
            raise ValueError('Truncated sparse integer layout')
        result = self.data[self.cursor:self.cursor + size]
        self.cursor += size
        return result

    def integer(self):
        start, value = self.cursor, 0
        for shift in range(0, 35, 7):
            byte = self.take(1)[0]
            value |= (byte & 127) << shift
            if byte < 128:
                if value > 0xffffffff or bytes(self.data[start:self.cursor]) != _varint(value):
                    raise ValueError('Noncanonical sparse integer')
                return value
        raise ValueError('Oversized sparse integer')

    def gaps(self, count, limit, *, deletion=False):
        if count > self.remaining or count > limit:
            raise ValueError('Too many sparse rows')
        start, previous = self.cursor, -1
        for _ in range(count):
            gap = self.integer()
            row = previous + gap + int(deletion)
            if (not deletion and gap == 0) or row >= limit:
                raise ValueError('Invalid sparse row gap')
            previous = row
        return bytes(self.data[start:self.cursor])


def pack_layout(layout, raw_length, magics):
    """Choose actual shortest transformed raw/zlib/XZ envelope."""
    size = _varint(raw_length) + _varint(len(layout))
    candidates = [magics[0] + size + layout,
                  magics[1] + size + zlib.compress(layout, 6),
                  magics[2] + size + lzma.compress(layout, format=lzma.FORMAT_XZ, preset=9)]
    return min(candidates, key=len)


def unpack_layout(packet, magics, *, max_raw_bytes):
    reader = _Reader(packet)
    magic = bytes(reader.take(4))
    if magic not in magics:
        raise ValueError('Unsupported sparse integer envelope')
    raw_length, length = reader.integer(), reader.integer()
    # A signed int16 becomes at most three zigzag bytes. Two times the raw
    # bound also leaves room for the prefix-length field.
    if not 1 <= raw_length <= max_raw_bytes or not 1 <= length <= 2 * raw_length + 8:
        raise ValueError('Invalid sparse integer layout length')
    payload = bytes(reader.take(reader.remaining))
    if magic == magics[0]:
        decoded = payload
    else:
        try:
            if magic == magics[1]:
                decoder = zlib.decompressobj()
                decoded = decoder.decompress(payload, length + 1)
                extra = decoder.unconsumed_tail
            else:
                decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ,
                                               memlimit=MAX_LZMA_MEMORY)
                decoded = decoder.decompress(payload, max_length=length + 1)
                extra = b''
        except (zlib.error, lzma.LZMAError, EOFError) as error:
            raise ValueError('Damaged sparse integer compression') from error
        if not decoder.eof or decoder.unused_data or extra:
            raise ValueError('Sparse integer compression trailer mismatch')
    if len(decoded) != length:
        raise ValueError('Sparse integer layout length mismatch')
    return decoded, raw_length


def _write_values(values, components):
    values = np.asarray(values, dtype='<i2').reshape(-1, components).T.ravel().astype(np.int64)
    return encode_unsigned_values(np.where(values >= 0, 2 * values, -2 * values - 1))


def _read_values(reader, count, components):
    total = count * components
    if total > reader.remaining:
        raise ValueError('Truncated sparse integer components')
    values = []
    for _ in range(total):
        symbol = reader.integer()
        if symbol > 65535:
            raise ValueError('Sparse symbol exceeds signed int16')
        values.append((symbol >> 1) ^ -(symbol & 1))
    return np.array(values, dtype='<i2').reshape(components, count).T


def _spatial_prefix(reader):
    magic = bytes(reader.take(4))
    if magic not in (b'SPG1', b'SPG2', b'SPG3', b'SPG4', *GATE_DELTA_MAGICS):
        raise ValueError('Planar motion layout requires a spatial motion packet')
    reader.take(32)
    frame, regions, count, births = [reader.integer() for _ in range(4)]
    if not frame or not regions or not count:
        raise ValueError('Invalid spatial layout dimensions')
    reader.take(40)
    survivor_mode = reader.take(1)[0]
    if survivor_mode == 1:
        deleted = reader.integer()
        reader.gaps(deleted, count + deleted, deletion=True)
    elif survivor_mode != 0:
        raise ValueError('Invalid survivor layout')
    if reader.integer() != 0:
        raise ValueError('Spatial layout requires a fixed parent tree')
    gate_start = reader.cursor
    if magic in GATE_DELTA_MAGICS:
        reader.gaps(reader.integer(), regions)
    else:
        reader.take((regions + 7) // 8)
    return regions, count, births, gate_start


def _encode_sparse(reader, limit):
    start = reader.cursor
    count = reader.integer()
    count_bytes = bytes(reader.data[start:reader.cursor])
    if count > limit or count > reader.remaining // 8:
        raise ValueError('Invalid sparse motion row count')
    gaps, flags, translations, rotations = bytearray(), bytearray(), [], []
    previous = -1
    for _ in range(count):
        start = reader.cursor
        gap = reader.integer()
        previous += gap
        if not gap or previous >= limit:
            raise ValueError('Invalid sparse motion row')
        gaps.extend(reader.data[start:reader.cursor])
        flag = reader.take(1)[0]
        if flag not in (1, 2, 3):
            raise ValueError('Invalid sparse motion flags')
        flags.append(flag)
        if flag & 1:
            translations.append(struct.unpack('<3h', reader.take(6)))
        if flag & 2:
            rotations.append(struct.unpack('<3h', reader.take(6)))
    return (count_bytes + gaps + flags + _write_values(translations, 3)
            + _write_values(rotations, 3))


def _decode_sparse(reader, limit):
    start = reader.cursor
    count = reader.integer()
    result = bytearray(reader.data[start:reader.cursor])
    gaps = reader.gaps(count, limit)
    flags = bytes(reader.take(count))
    if any(flag not in (1, 2, 3) for flag in flags):
        raise ValueError('Invalid planar motion flags')
    t = _read_values(reader, sum(bool(x & 1) for x in flags), 3)
    r = _read_values(reader, sum(bool(x & 2) for x in flags), 3)
    gaps_reader = _Reader(gaps)
    ti = ri = 0
    for flag in flags:
        start = gaps_reader.cursor
        gaps_reader.integer()
        result.extend(gaps_reader.data[start:gaps_reader.cursor])
        result.append(flag)
        if flag & 1:
            result.extend(t[ti].tobytes())
            ti += 1
        if flag & 2:
            result.extend(r[ri].tobytes())
            ri += 1
    return bytes(result)


def _check_motion_tail(tail, births, count, pose_bytes=28):
    reader = _Reader(tail)
    if births > reader.remaining // (pose_bytes + 1):
        raise ValueError('Truncated sparse birth layout')
    for _ in range(births):
        if reader.integer() >= count:
            raise ValueError('Invalid sparse birth source')
    reader.take(births * pose_bytes + 16)
    if reader.remaining:
        raise ValueError('Unexpected sparse motion trailer')


def encode_motion_layout(raw):
    reader = _Reader(raw)
    regions, count, births, _ = _spatial_prefix(reader)
    prefix = raw[:reader.cursor]
    region = _encode_sparse(reader, regions - 1)
    point = _encode_sparse(reader, count)
    tail = bytes(reader.take(reader.remaining))
    _check_motion_tail(tail, births, count, 12 if raw[:4] in TRANSLATION_ONLY_SPATIAL_MAGICS else 28)
    return _varint(len(prefix)) + prefix + region + point + tail


def decode_motion_layout(layout, raw_length):
    reader = _Reader(layout)
    prefix = bytes(reader.take(reader.integer()))
    prefix_reader = _Reader(prefix)
    regions, count, births, _ = _spatial_prefix(prefix_reader)
    if prefix_reader.remaining:
        raise ValueError('Unexpected spatial prefix trailer')
    region = _decode_sparse(reader, regions - 1)
    point = _decode_sparse(reader, count)
    tail = bytes(reader.take(reader.remaining))
    _check_motion_tail(tail, births, count, 12 if prefix[:4] in TRANSLATION_ONLY_SPATIAL_MAGICS else 28)
    raw = prefix + region + point + tail
    if len(raw) != raw_length:
        raise ValueError('Reconstructed motion length mismatch')
    return raw


def _opacity_header(reader):
    header = bytes(reader.take(OPACITY_HEADER.size))
    magic, frame, count, nonzero, step, size, checksum, prior = OPACITY_HEADER.unpack(header)
    if magic not in (b'OPD1', b'OPD2', b'OPD3', b'OPD4') or not 1 <= nonzero <= count:
        raise ValueError('Invalid sparse opacity dimensions')
    if not nonzero * 3 <= size <= nonzero * 7:
        raise ValueError('Invalid sparse opacity body size')
    return header, count, nonzero, size


def encode_opacity_layout(raw):
    reader = _Reader(raw)
    header, count, nonzero, size = _opacity_header(reader)
    if reader.remaining != size:
        raise ValueError('Opacity layout body length mismatch')
    gaps, values, previous = bytearray(), [], -1
    for _ in range(nonzero):
        start = reader.cursor
        gap = reader.integer()
        previous += gap
        if not gap or previous >= count:
            raise ValueError('Invalid opacity layout row')
        gaps.extend(reader.data[start:reader.cursor])
        values.append(struct.unpack('<h', reader.take(2)))
    if reader.remaining:
        raise ValueError('Unexpected opacity layout trailer')
    return header + gaps + _write_values(values, 1)


def decode_opacity_layout(layout, raw_length):
    reader = _Reader(layout)
    header, count, nonzero, size = _opacity_header(reader)
    gaps = reader.gaps(nonzero, count)
    values = _read_values(reader, nonzero, 1)
    if reader.remaining:
        raise ValueError('Unexpected planar opacity trailer')
    result = bytearray(header)
    gaps_reader = _Reader(gaps)
    for value in values:
        start = gaps_reader.cursor
        gaps_reader.integer()
        result.extend(gaps_reader.data[start:gaps_reader.cursor])
        result.extend(value.tobytes())
    if len(result) != raw_length or len(result) != len(header) + size:
        raise ValueError('Reconstructed opacity length mismatch')
    return bytes(result)
