"""Birth appearance stream; motion stream already carries pose and source row."""

from pathlib import Path
import struct
import zlib

import numpy as np
import torch


MAGIC = b'RTB1'
HEADER = struct.Struct('<4sIIIII')
ATTRIBUTES = ('logscale', 'opacity', 'sh_dc', 'sh_rest')


def encode_birth_attributes(raw, base_count):
    count = len(raw['logscale'])
    if count < 1 or base_count < 1:
        raise ValueError('Birth appearance needs newborn rows and a base')
    rest = raw['sh_rest'].shape[1]
    shapes = dict(logscale=(3,), opacity=(1,), sh_dc=(1, 3),
                  sh_rest=(rest, 3))
    pieces = []
    for name in ATTRIBUTES:
        value = raw[name]
        if (value.dtype != torch.float32 or
                value.shape != (count,) + shapes[name] or
                not bool(torch.isfinite(value).all())):
            raise ValueError(f'Birth {name} needs finite float32 raw values')
        pieces.append(value.detach().cpu().numpy().astype('<f4').tobytes())
    body = b''.join(pieces)
    return HEADER.pack(MAGIC, int(base_count), count, rest, len(body),
                       zlib.crc32(body)) + zlib.compress(body, 3)


def export_birth_attributes(path, raw, base_count):
    packet = encode_birth_attributes(raw, base_count)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(packet)
    return dict(path=str(path), archive_bytes=len(packet),
                rows=len(raw['logscale']), format='rigid_birth_appearance_zlib_v1')


def decode_birth_attributes(packet, base, decoded_xyz, decoded_rotation):
    if len(packet) < HEADER.size:
        raise ValueError('Truncated birth appearance stream')
    magic, base_count, count, rest, length, checksum = HEADER.unpack_from(packet)
    expected = count * (7 + 3 * rest) * 4
    if (magic != MAGIC or base_count != len(base['xyz']) or
            rest != base['sh_rest'].shape[1] or count < 1 or
            length != expected or length > 256 * 1024 * 1024):
        raise ValueError('Invalid birth appearance header')
    if (decoded_xyz.shape != (count, 3) or
            decoded_rotation.shape != (count, 4)):
        raise ValueError('Motion and birth appearance counts disagree')
    decoder = zlib.decompressobj()
    body = decoder.decompress(packet[HEADER.size:], length + 1)
    if (len(body) != length or not decoder.eof or
            decoder.unused_data or decoder.unconsumed_tail or
            zlib.crc32(body) != checksum):
        raise ValueError('Birth appearance checksum or length mismatch')
    shapes = dict(logscale=(count, 3), opacity=(count, 1),
                  sh_dc=(count, 1, 3), sh_rest=(count, rest, 3))
    offset = 0
    raw = dict(xyz=decoded_xyz.clone(), rotation=decoded_rotation.clone())
    for name in ATTRIBUTES:
        shape = shapes[name]
        elements = int(np.prod(shape))
        array = np.frombuffer(body, dtype='<f4', count=elements,
                              offset=offset).reshape(shape).copy()
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite birth {name}')
        raw[name] = torch.as_tensor(array, device=base[name].device,
                                    dtype=base[name].dtype)
        offset += elements * 4
    return raw
