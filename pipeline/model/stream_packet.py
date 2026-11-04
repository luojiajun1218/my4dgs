"""Checked packet parsing independent of dense geometry materialization.

The accelerated player specializes the active SPG5/6 translation stream.
Packet checksums and metadata are checked here. A production GPU player checks
declared geometry-state links; independent verification must additionally
download and hash every decoded state before its playback FPS is reported.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import zlib

import numpy as np

from . import rigid_tree_codec as motion
from . import rigid_tree_birth_codec as birth
from . import legacy_opacity_codec as opacity
from . import sparse_integer_codec as integer


@dataclass(frozen=True)
class BirthAttributes:
    values: dict
    steps: tuple | None


@dataclass(frozen=True)
class OpacitySymbols:
    rows: np.ndarray
    values: np.ndarray
    step: float
    prior: bytes
    count: int
    mode: str


@dataclass(frozen=True)
class StreamStep:
    frame: int
    prior_digest: bytes
    digest: bytes
    raw_digest: str
    previous_count: int
    old_count: int
    birth_count: int
    deleted: np.ndarray
    opened: np.ndarray
    regional_symbols: np.ndarray
    relative: bool
    gs_rows: np.ndarray
    gs_symbols: np.ndarray
    region_step: float
    gs_step: float
    sources: np.ndarray
    pose_symbols: np.ndarray
    pose_steps: tuple
    attributes: BirthAttributes | None
    opacity: OpacitySymbols | None
    opacity_before_births: bool
    stored_bytes: int
    feature_dc: bytes | None = None
    inherit_birth_rotation: bool = False
    logscale_update: bytes | None = None


def _birth_attributes(packet, old_count, count, rest):
    packet = bytes(packet)
    magic = packet[:4]
    if magic == birth.SYMBOL_MAGIC:
        if len(packet) < birth.SYMBOL_HEADER.size:
            raise ValueError('Truncated birth symbol header')
        fields = birth.SYMBOL_HEADER.unpack_from(packet)
        _, base, rows, coefficients = fields[:4]
        widths, steps, length, checksum = fields[4:8], fields[8:12], fields[12], fields[13]
        offset = birth.SYMBOL_HEADER.size
        if (any(width not in (2,4) for width in widths) or
                any(not math.isfinite(step) or step <= 0 for step in steps)):
            raise ValueError('Invalid birth quantization metadata')
        dtype = lambda width: '<i2' if width == 2 else '<i4'
        prefix = packet[:offset-4]
    elif magic == birth.MAGIC:
        if len(packet) < birth.HEADER.size:
            raise ValueError('Truncated birth absolute header')
        _, base, rows, coefficients, length, checksum = birth.HEADER.unpack_from(packet)
        widths, steps, offset, prefix = (4,)*4, None, birth.HEADER.size, b''
        dtype = lambda width: '<f4'
    else:
        raise ValueError('Unsupported birth appearance packet')
    expected = count*(3*widths[0]+widths[1]+3*widths[2]+3*rest*widths[3])
    if (base != old_count or rows != count or coefficients != rest or
            count < 1 or length != expected or length > birth.MAX_BODY_BYTES):
        raise ValueError('Birth dimensions differ from the motion packet')
    body = birth._decompress(packet, offset, length)
    if zlib.crc32(prefix+body) != checksum:
        raise ValueError('Birth appearance checksum mismatch')
    values, cursor = {}, 0
    for name, shape, width in zip(birth.ATTRIBUTES, birth._shapes(count,rest).values(), widths):
        array = np.frombuffer(body,dtype=dtype(width),count=int(np.prod(shape)),offset=cursor).reshape(shape)
        if steps is None and not bool(np.isfinite(array).all()):
            raise ValueError('Nonfinite absolute newborn appearance')
        values[name] = array
        cursor += array.size*width
    return BirthAttributes(values, steps)


def _opacity_symbols(packet, frame, count):
    packet = bytes(packet)
    parsed = {}
    if packet[:4] in integer.OPACITY_MAGICS:
        layout, length = integer.unpack_layout(packet,integer.OPACITY_MAGICS,
            max_raw_bytes=opacity.HEADER.size+opacity.MAX_BODY_BYTES)
        raw = integer.decode_opacity_layout(layout,length,parsed=parsed)
        header, body = raw[:opacity.HEADER.size], raw[opacity.HEADER.size:]
    else:
        if len(packet) < opacity.HEADER.size:
            raise ValueError('Truncated opacity header')
        header = packet[:opacity.HEADER.size]
        length = opacity.HEADER.unpack(header)[5]
        if length > opacity.MAX_BODY_BYTES:
            raise ValueError('Oversized opacity body')
        decoder = zlib.decompressobj()
        try:
            body = decoder.decompress(packet[opacity.HEADER.size:],length+1)
        except zlib.error as error:
            raise ValueError('Damaged opacity packet') from error
        if (len(body) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError('Invalid opacity compressed length or trailer')
    magic, index, rows, nonzero, step, length, checksum, prior = opacity.HEADER.unpack(header)
    if (magic not in opacity.MODES.values() or index != frame or rows != count or
            not 1 <= nonzero <= count or not nonzero*3 <= length <= nonzero*7 or
            len(body) != length or not math.isfinite(step) or step <= 0):
        raise ValueError('Opacity metadata differs from the motion frame')
    prefix = opacity.HEADER.pack(magic,index,rows,nonzero,step,length,0,prior)
    if zlib.crc32(prefix+body) != checksum:
        raise ValueError('Opacity checksum mismatch')
    if not parsed:
        ids, symbols, cursor, previous = [], [], 0, -1
        for _ in range(nonzero):
            gap, cursor = opacity._read_varint(body,cursor)
            previous += gap
            if gap < 1 or previous >= count or cursor+2 > len(body):
                raise ValueError('Invalid opacity row')
            symbol = struct.unpack_from('<h',body,cursor)[0]
            if symbol == 0:
                raise ValueError('Redundant zero opacity update')
            ids.append(previous);symbols.append(symbol);cursor += 2
        if cursor != len(body):
            raise ValueError('Unexpected opacity trailer')
        parsed = dict(rows=np.asarray(ids,dtype=np.int64),symbols=np.asarray(symbols,dtype=np.int16)[:,None])
    mode = next(name for name,value in opacity.MODES.items() if value == magic)
    return OpacitySymbols(parsed['rows'],parsed['symbols'],float(step),prior,count,mode)


def read_stream_step(directory, frame, *, region_count, depth, rest, max_points=1_000_000):
    """Read every required file and validate the existing transmitted bytes."""
    directory = Path(directory)
    manifest_file = directory/f'frame_{frame:03d}.json'
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_names = dict(motion=f'motion_{frame:03d}.bin',birth=f'birth_{frame:03d}.bin',
                          opacity_update=f'opacity_{frame:03d}.bin')
    if manifest.get('frame') != frame:
        raise ValueError('Unsupported frame manifest')
    # This specialized fast player cannot apply general per-GS rotation or
    # full appearance packets. Call the complete CPU replay path instead.
    if manifest.get('full_update') or manifest.get('full_attribute_update') is not None:
        raise ValueError('Full-update frames require the complete CPU decoder')
    if (directory / f'fullattr_{frame:03d}.bin').exists():
        raise ValueError('Unexpected full attribute packet absent from frame manifest')
    for key, name in expected_names.items():
        if manifest.get(key) not in (name,None) or (key == 'motion' and manifest.get(key) != name):
            raise ValueError('Unexpected stream filename')
    feature = manifest.get('feature_dc_update')
    feature_packet = None
    if feature is not None:
        if (feature != f'shdc_{frame:03d}.bin' or
                manifest.get('feature_dc_update_timing', 'before_births') != 'before_births'):
            raise ValueError('Shared color packet must precede newborn attributes')
        feature_packet = (directory / feature).read_bytes()
        if feature_packet[:4] not in (b'SDR1', b'SDC1'):
            raise ValueError('Unknown shared color packet')
    elif (directory/f'shdc_{frame:03d}.bin').exists():
        raise ValueError('Unexpected feature DC packet')
    logscale = manifest.get('logscale_update')
    logscale_packet = None
    if logscale is not None:
        if (logscale != f'logscale_{frame:03d}.bin' or
                manifest.get('logscale_update_timing', 'before_births') != 'before_births'):
            raise ValueError('Log-scale packet must precede newborn attributes')
        logscale_packet = (directory / logscale).read_bytes()
        if logscale_packet[:4] not in (b'SLS1', b'SLZ1', b'SLA1', b'ZLA1'):
            raise ValueError('Unknown log-scale packet')
    elif (directory/f'logscale_{frame:03d}.bin').exists():
        raise ValueError('Unexpected log-scale packet')
    packet = (directory/expected_names['motion']).read_bytes()
    stored = len(manifest_bytes)+len(packet)
    if feature_packet is not None:stored += len(feature_packet)
    if logscale_packet is not None:stored += len(logscale_packet)
    sections = {}
    raw = motion.uncompressed_motion_packet(packet,sparse_sections=sections)
    reader = integer._Reader(raw[:-motion.CHECKSUM_BYTES])
    magic = bytes(reader.take(4))
    if magic not in motion.OFFSET_SPATIAL_MAGICS:
        raise ValueError('GPU streaming requires a spatial offset-pose format')
    inherit_rotation = magic in motion.TRANSLATION_ONLY_MAGICS
    prior = bytes(reader.take(32))
    index, regions, count, births = [reader.integer() for _ in range(4)]
    if (index != frame or regions != region_count or count < 1 or
            count+births > max_points or births > 10_000_000):
        raise ValueError('Invalid motion frame dimensions')
    extent, region_step, rotation_step, gs_step, gs_rotation_step = struct.unpack('<5d',reader.take(40))
    if not all(math.isfinite(value) and value > 0 for value in
               (extent,region_step,rotation_step,gs_step,gs_rotation_step)):
        raise ValueError('Invalid motion quantization metadata')
    mode = reader.take(1)[0]
    if mode == 0:
        deleted = np.empty(0,dtype=np.int64)
    elif mode == 1:
        number = reader.integer()
        gaps = integer._read_unsigned_values(reader,number).astype(np.int64)
        deleted = gaps.cumsum()+np.arange(number,dtype=np.int64)
        if number and int(deleted[-1]) >= count+number:
            raise ValueError('Deletion row outside the preceding population')
    else:
        raise ValueError('Invalid survivor mapping mode')
    if reader.integer() != 0:
        raise ValueError('Spatial tree cannot be reparented')
    opened = np.unpackbits(np.frombuffer(reader.take((regions+7)//8),dtype=np.uint8),bitorder='little')[:regions].astype(bool)
    first_leaf = sum(8**level for level in range(depth))
    if not opened[0] or bool(opened[first_leaf:].any()):
        raise ValueError('Invalid spatial gate frontier')
    def sparse(limit):
        section = sections.get(reader.cursor)
        if section is not None:
            span, expected, rows, values, rotation_rows, rotation_values = section
            if expected != limit:
                raise ValueError('Sparse motion dimension mismatch')
            reader.take(span)
            if len(rotation_rows):
                raise ValueError('GPU streaming requires zero old rotation updates')
            return rows,values
        start = reader.cursor
        old = motion._Reader(reader.data)
        old.cursor = start
        t,r = motion._sparse_read(old,limit,delta_rows=True)
        reader.cursor = old.cursor
        if bool(r.ne(0).any()):
            raise ValueError('GPU streaming requires zero old rotation updates')
        ids = t.ne(0).any(-1).nonzero().flatten().numpy()
        return ids,t.numpy()[ids]
    regional_rows, regional_values = sparse(regions-1)
    gs_rows, gs_values = sparse(count)
    regional = np.zeros((regions,3),dtype=np.int32)
    regional[regional_rows+1] = regional_values
    sources = integer._read_unsigned_values(reader,births).astype(np.int64)
    if bool((sources >= count).any()):
        raise ValueError('Newborn source outside surviving rows')
    pose_steps = struct.unpack('<2d',reader.take(16))
    if not all(math.isfinite(value) and value > 0 for value in pose_steps):
        raise ValueError('Invalid newborn pose quantization steps')
    width = 3 if inherit_rotation else 7
    packed = integer._read_unsigned_values(reader,births*width).astype(np.int64)
    pose_symbols = ((packed>>1)^-(packed&1)).astype(np.float32).reshape(births,width)
    if inherit_rotation:
        pose_symbols = np.pad(pose_symbols, ((0,0),(0,4)))
    if reader.remaining:
        raise ValueError('Unexpected motion trailer')
    if (manifest['old_count'] != count or manifest['birth_count'] != births or
            manifest['previous_count'] != count+len(deleted) or manifest['total_count'] != count+births):
        raise ValueError('Manifest population differs from motion packet')
    attribute = None
    if births:
        if manifest.get('birth') != expected_names['birth']:
            raise ValueError('Newborn attributes are missing')
        packet = (directory/expected_names['birth']).read_bytes();stored += len(packet)
        attribute = _birth_attributes(packet,count,births,rest)
    elif manifest.get('birth') is not None or (directory/expected_names['birth']).exists():
        raise ValueError('Unexpected birth packet for an empty frame')
    update = None
    if manifest.get('opacity_update') is not None:
        packet = (directory/expected_names['opacity_update']).read_bytes();stored += len(packet)
        update = _opacity_symbols(packet,frame,count)
    elif (directory/expected_names['opacity_update']).exists():
        raise ValueError('Opacity packet absent from manifest')
    timing = manifest.get('opacity_update_timing','after_births')
    if timing not in ('before_births','after_births'):
        raise ValueError('Unsupported opacity update timing')
    return StreamStep(frame,prior,bytes.fromhex(manifest['decoded_state_sha256']),manifest['raw_state_sha256'],
        count+len(deleted),count,births,deleted,opened,regional,magic in (b'SPG6',b'SPG8'),gs_rows,gs_values,
        extent*region_step,extent*gs_step,sources,pose_symbols,pose_steps,attribute,update,
        timing == 'before_births',stored,feature_packet,inherit_rotation,logscale_packet)
