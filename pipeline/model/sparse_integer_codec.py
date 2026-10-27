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
        start = self.cursor
        values = _read_unsigned_values(self, count)
        if ((not deletion and bool((values == 0).any())) or
                (count and int(values.sum(dtype=np.uint64)) + count * int(deletion) > limit)):
            raise ValueError('Invalid sparse row gap')
        return bytes(self.data[start:self.cursor])


def _read_unsigned_values(reader, count):
    """Read canonical uint32 varints in bulk, preserving the exact byte cursor."""
    if count < 0 or count > reader.remaining:
        raise ValueError('Truncated sparse integer components')
    if not count:
        return np.empty(0, dtype=np.uint32)
    section = reader.data[reader.cursor:reader.cursor + min(reader.remaining, count * 5)]
    try:
        data = np.frombuffer(section, dtype=np.uint8)
    except BufferError:
        data = np.frombuffer(bytes(section), dtype=np.uint8)
    ends = np.flatnonzero(data < 128)[:count]
    if len(ends) != count:
        raise ValueError('Truncated or oversized sparse integer')
    starts = np.r_[0, ends[:-1] + 1]
    lengths = ends - starts + 1
    if bool((lengths > 5).any()):
        raise ValueError('Oversized sparse integer')
    terminal = data[ends]
    if bool(((lengths > 1) & (terminal == 0)).any()) or bool(((lengths == 5) & (terminal > 15)).any()):
        raise ValueError('Noncanonical sparse integer')
    values = np.zeros(count, dtype=np.uint32)
    for byte_index in range(int(lengths.max())):
        used = lengths > byte_index
        values[used] |= (data[starts[used] + byte_index].astype(np.uint32) & 127) << (7 * byte_index)
    reader.cursor += int(ends[-1]) + 1
    return values


def _interleaved_rows(gaps, suffix_widths):
    """Place validated variable-length gap bytes before each row's fixed tail."""
    source = np.frombuffer(gaps, dtype=np.uint8)
    ends = np.flatnonzero(source < 128)
    lengths = np.diff(np.r_[-1, ends])
    if len(lengths) != len(suffix_widths):
        raise ValueError('Sparse gap count differs from row count')
    widths = lengths + np.asarray(suffix_widths, dtype=np.int64)
    starts = np.cumsum(widths, dtype=np.int64) - widths
    result = np.empty(int(widths.sum()), dtype=np.uint8)
    gap_starts = np.r_[0, ends[:-1] + 1] if len(ends) else np.empty(0, dtype=np.int64)
    destinations = np.arange(len(source)) + np.repeat(starts - gap_starts, lengths)
    result[destinations] = source
    return result, starts + lengths


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
    symbols = _read_unsigned_values(reader, total)
    if bool((symbols > 65535).any()):
        raise ValueError('Sparse symbol exceeds signed int16')
    symbols = symbols.astype(np.int32)
    values = (symbols >> 1) ^ -(symbols & 1)
    return values.astype('<i2').reshape(components, count).T


def _spatial_prefix(reader):
    if bytes(reader.take(4)) not in (b'SPG1', b'SPG2', b'SPG3', b'SPG4', b'SPG5', b'SPG6', b'SPG7', b'SPG8'):
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
    reader.take((regions + 7) // 8)
    return regions, count, births


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


def _decode_sparse(reader, limit, *, decoded=None):
    start = reader.cursor
    count = reader.integer()
    result = bytearray(reader.data[start:reader.cursor])
    gaps = reader.gaps(count, limit)
    flags = np.frombuffer(reader.take(count), dtype=np.uint8)
    if bool(((flags < 1) | (flags > 3)).any()):
        raise ValueError('Invalid planar motion flags')
    active_t, active_r = (flags & 1).astype(bool), (flags & 2).astype(bool)
    t = _read_values(reader, int(active_t.sum()), 3)
    r = _read_values(reader, int(active_r.sum()), 3)
    records, suffixes = _interleaved_rows(gaps,
        1 + 6 * (active_t.astype(np.int64) + active_r.astype(np.int64)))
    records[suffixes] = flags
    offsets = np.arange(6)
    records[(suffixes[active_t, None] + 1 + offsets).ravel()] = np.frombuffer(t.tobytes(), dtype=np.uint8)
    rotation_starts = suffixes[active_r] + 1 + 6 * active_t[active_r]
    records[(rotation_starts[:, None] + offsets).ravel()] = np.frombuffer(r.tobytes(), dtype=np.uint8)
    result.extend(records.tobytes())
    if decoded is not None:
        # Reuse these already-parsed symbols after the reconstructed packet's
        # checksum is checked. The scalar raw decoder rejects zero blocks too.
        if bool((~t.any(-1)).any()):
            raise ValueError('Redundant zero translation block')
        if bool((~r.any(-1)).any()):
            raise ValueError('Redundant zero rotation block')
        rows = _read_unsigned_values(_Reader(gaps), count).astype(np.int64).cumsum() - 1
        # Keep sparse arrays until the raw decoder has checked these declared
        # dimensions against its preceding state. Untrusted headers must not
        # allocate dense arrays before that check.
        decoded.append((rows[active_t], t, rows[active_r], r))
    return bytes(result)


def _check_motion_tail(tail, births, count, pose_bytes=28, offset=False):
    reader = _Reader(tail)
    if births > reader.remaining // (pose_bytes + 1):
        raise ValueError('Truncated sparse birth layout')
    if bool((_read_unsigned_values(reader, births) >= count).any()):
        raise ValueError('Invalid sparse birth source')
    if offset:
        steps=struct.unpack('<2d',reader.take(16))
        if not all(np.isfinite(v) and v>0 for v in steps):raise ValueError('Invalid birth pose steps')
        _read_unsigned_values(reader,births*pose_bytes)
        reader.take(16)
    else:reader.take(births * pose_bytes + 16)
    if reader.remaining:
        raise ValueError('Unexpected sparse motion trailer')


def encode_motion_layout(raw):
    reader = _Reader(raw)
    regions, count, births = _spatial_prefix(reader)
    prefix = raw[:reader.cursor]
    region = _encode_sparse(reader, regions - 1)
    point = _encode_sparse(reader, count)
    tail = bytes(reader.take(reader.remaining))
    _check_motion_tail(tail, births, count, 12 if raw[:4] in (b'SPG3', b'SPG4') else 3 if raw[:4] in (b'SPG7',b'SPG8') else 7 if raw[:4] in (b'SPG5',b'SPG6') else 28, offset=raw[:4] in (b'SPG5',b'SPG6',b'SPG7',b'SPG8'))
    return _varint(len(prefix)) + prefix + region + point + tail


def decode_motion_layout(layout, raw_length, *, sparse_sections=None):
    """Restore original bytes, optionally retaining symbols for the raw decoder.

    Section offsets refer to the restored raw bytes. The caller must verify
    their existing checksum before consuming any retained arrays.
    """
    reader = _Reader(layout)
    prefix = bytes(reader.take(reader.integer()))
    prefix_reader = _Reader(prefix)
    regions, count, births = _spatial_prefix(prefix_reader)
    if prefix_reader.remaining:
        raise ValueError('Unexpected spatial prefix trailer')
    decoded = [] if sparse_sections is not None else None
    region = _decode_sparse(reader, regions - 1, decoded=decoded)
    point = _decode_sparse(reader, count, decoded=decoded)
    tail = bytes(reader.take(reader.remaining))
    _check_motion_tail(tail, births, count, 12 if prefix[:4] in (b'SPG3', b'SPG4') else 3 if prefix[:4] in (b'SPG7',b'SPG8') else 7 if prefix[:4] in (b'SPG5',b'SPG6') else 28, offset=prefix[:4] in (b'SPG5',b'SPG6',b'SPG7',b'SPG8'))
    raw = prefix + region + point + tail
    if len(raw) != raw_length:
        raise ValueError('Reconstructed motion length mismatch')
    if sparse_sections is not None:
        sparse_sections[len(prefix)] = (len(region), regions - 1, *decoded[0])
        sparse_sections[len(prefix) + len(region)] = (len(point), count, *decoded[1])
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


def decode_opacity_layout(layout, raw_length, *, parsed=None):
    reader = _Reader(layout)
    header, count, nonzero, size = _opacity_header(reader)
    gaps = reader.gaps(nonzero, count)
    values = _read_values(reader, nonzero, 1)
    if reader.remaining:
        raise ValueError('Unexpected planar opacity trailer')
    records, suffixes = _interleaved_rows(gaps, np.full(nonzero, 2, dtype=np.int64))
    records[(suffixes[:, None] + np.arange(2)).ravel()] = np.frombuffer(values.tobytes(), dtype=np.uint8)
    result = bytearray(header)
    result.extend(records.tobytes())
    if len(result) != raw_length or len(result) != len(header) + size:
        raise ValueError('Reconstructed opacity length mismatch')
    if parsed is not None:
        if bool((values == 0).any()):
            raise ValueError('Sparse opacity packet contains a zero update')
        parsed['rows'] = _read_unsigned_values(_Reader(gaps), nonzero).astype(np.int64).cumsum() - 1
        parsed['symbols'] = values
    return bytes(result)
