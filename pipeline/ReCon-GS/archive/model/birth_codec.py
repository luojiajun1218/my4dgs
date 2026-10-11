"""Lossless birth attributes and explicit tree attachment rows."""

from pathlib import Path
import struct
import zlib

import numpy as np
import torch

from .residual_motion import RAW_FIELDS


MAGIC = b'GSB1'
HEADER = struct.Struct('<4sIIIII')


def raw_birth_size(count, sh_degree):
    """Pre-entropy cost: header, source IDs and six float32 attributes."""
    count, sh_degree = int(count), int(sh_degree)
    if count < 0 or sh_degree < 0:
        raise ValueError('Invalid birth count or SH degree')
    return HEADER.size + count * (4 + 4 * (11 + 3 * (sh_degree + 1) ** 2)) if count else 0


def encode_birth_packet(raw, sources, base_count):
    count = len(raw['xyz'])
    sources = torch.as_tensor(sources)
    if (not count or base_count < 1 or sources.shape != (count,) or
            sources.dtype not in (torch.int32, torch.int64) or
            bool(((sources < 0) | (sources >= base_count)).any())):
        raise ValueError('Birth packet needs valid tree attachment rows')
    rest = raw['sh_rest'].shape[1]
    shapes = dict(xyz=(3,), sh_dc=(1, 3), sh_rest=(rest, 3),
                  logscale=(3,), rotation=(4,), opacity=(1,))
    pieces = [sources.detach().cpu().numpy().astype('<u4').tobytes()]
    for name in RAW_FIELDS:
        value = raw[name]
        if value.dtype != torch.float32 or value.shape != (count,) + shapes[name] or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Birth {name} requires finite float32 values with its raw shape')
        pieces.append(value.detach().cpu().numpy().astype('<f4').tobytes())
    body = b''.join(pieces)
    return HEADER.pack(MAGIC, int(base_count), count, rest, len(body), zlib.crc32(body)) + zlib.compress(body, 3)


def export_birth_packet(path, raw, sources, base_count):
    packet = encode_birth_packet(raw, sources, base_count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packet)
    return dict(path=str(path), archive_bytes=len(packet), rows=len(sources),
                format='birth_raw_float32_zlib_v1', attachment_rows_stored=True)


def decode_birth_packet(packet, base):
    if len(packet) < HEADER.size:
        raise ValueError('Truncated birth packet')
    magic, base_count, count, rest, length, checksum = HEADER.unpack_from(packet)
    expected_length = count * (4 + 4 * (14 + 3 * rest))
    if (magic != MAGIC or base_count != len(base['xyz']) or not count or
            rest != base['sh_rest'].shape[1] or length != expected_length or length > 256 * 1024 * 1024):
        raise ValueError('Invalid birth header')
    decoder = zlib.decompressobj()
    body = decoder.decompress(packet[HEADER.size:], length + 1)
    if (len(body) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or
            zlib.crc32(body) != checksum):
        raise ValueError('Birth payload failed validation')
    sources = np.frombuffer(body[:4 * count], dtype='<u4').astype(np.int64)
    if np.any(sources >= base_count):
        raise ValueError('Birth attachment row out of bounds')
    shapes = dict(xyz=(3,), sh_dc=(1, 3), sh_rest=(rest, 3),
                  logscale=(3,), rotation=(4,), opacity=(1,))
    offset, raw = 4 * count, {}
    for name in RAW_FIELDS:
        shape = (count,) + shapes[name]
        elements = int(np.prod(shape))
        value = np.frombuffer(body, dtype='<f4', count=elements, offset=offset).reshape(shape).copy()
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite birth {name}')
        raw[name] = torch.as_tensor(value, device=base[name].device, dtype=base[name].dtype)
        offset += elements * 4
    return raw, torch.as_tensor(sources.copy(), device=base['xyz'].device, dtype=torch.long)
