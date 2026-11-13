"""码流格式及编解码共用的数据定义和基础操作。

包含数据包常量、字段、解码状态、参考摘要及共享量化/校验工具；
整帧编码与解码入口分别放在 frame_stream_encode.py 和 frame_stream_decode.py。
"""

from dataclasses import dataclass
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
from zipfile import ZIP_LZMA, ZipFile
import zlib

import numpy as np
import torch

from .rigid_residual_tree import compose_pose, region_pivots, _topological_order
from .spatial_region_gate import (compose_spatial_pose, spatial_owner, spatial_route,
                           spatial_state_route, SPATIAL_ROUTING_FIELDS,
                           validate_spatial_routing, spatial_accumulate)

RAW_FIELDS = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')
RAW_ATTRIBUTES = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest')


def reference_digest(raw):
    """Fingerprint the ordered raw GS attributes for stream verification."""
    digest = hashlib.sha256()
    for name in RAW_FIELDS:
        tensor = raw[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(memoryview(tensor.numpy()))
    return digest.hexdigest()


MOTION_MAGICS = (b'SVI1', b'SVZ1', b'SVX1')


OPACITY_MAGICS = (b'OVI1', b'OVZ1', b'OVX1')


OPACITY_HEADER = struct.Struct('<4sIIIfII32s')


MAX_LZMA_MEMORY = 128 * 1024 * 1024


def _integer_varint(value):
    value = int(value)
    if value < 0:
        raise ValueError('Sparse integers must be nonnegative')
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


class _IntegerReader:
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
                if value > 0xffffffff or bytes(self.data[start:self.cursor]) != _integer_varint(value):
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


def _check_motion_tail(tail, births, count, pose_bytes=28, offset=False):
    reader = _IntegerReader(tail)
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


def _opacity_header(reader):
    header = bytes(reader.take(OPACITY_HEADER.size))
    magic, frame, count, nonzero, step, size, checksum, prior = OPACITY_HEADER.unpack(header)
    if magic not in (b'OPD1', b'OPD2', b'OPD3', b'OPD4') or not 1 <= nonzero <= count:
        raise ValueError('Invalid sparse opacity dimensions')
    if not nonzero * 3 <= size <= nonzero * 7:
        raise ValueError('Invalid sparse opacity body size')
    return header, count, nonzero, size


MAGIC = b'RRT1'


DUAL_STEP_MAGIC = b'RRT2'


COMPRESSED_MAGIC = b'RRC1'


DUAL_STEP_COMPRESSED_MAGIC = b'RRC2'


LZMA_COMPRESSED_MAGIC = b'RRL1'


DUAL_STEP_LZMA_COMPRESSED_MAGIC = b'RRL2'


DELTA_MAGIC = b'RSD1'


DUAL_STEP_DELTA_MAGIC = b'RSD2'


DELTA_COMPRESSED_MAGIC = b'RSC1'


DUAL_STEP_DELTA_COMPRESSED_MAGIC = b'RSC2'


DELTA_LZMA_COMPRESSED_MAGIC = b'RSL1'


DUAL_STEP_DELTA_LZMA_COMPRESSED_MAGIC = b'RSL2'


DYNAMIC_MAGIC = b'RSD3'


DYNAMIC_COMPRESSED_MAGIC = b'RSC3'


DYNAMIC_LZMA_COMPRESSED_MAGIC = b'RSL3'


SPATIAL_MAGIC = b'SPG1'


SPATIAL_COMPRESSED_MAGIC = b'SPC1'


SPATIAL_LZMA_COMPRESSED_MAGIC = b'SPL1'


RELATIVE_SPATIAL_MAGIC = b'SPG2'


TRANSLATION_ONLY_SPATIAL_MAGIC = b'SPG3'


RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC = b'SPG4'


TRANSLATION_ONLY_MAGICS = (TRANSLATION_ONLY_SPATIAL_MAGIC,
                           RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC, b'SPG7', b'SPG8')


OFFSET_SPATIAL_MAGICS=(b'SPG5',b'SPG6',b'SPG7',b'SPG8')


SPATIAL_MAGICS = tuple(dict.fromkeys((SPATIAL_MAGIC, RELATIVE_SPATIAL_MAGIC,
                                     *TRANSLATION_ONLY_MAGICS, *OFFSET_SPATIAL_MAGICS)))


CHECKSUM_BYTES = 16


MAX_RAW_PACKET_BYTES = 512 * 1024 * 1024


MAX_LZMA_MEMORY_BYTES = 128 * 1024 * 1024


RAW_MAGICS = (MAGIC, DUAL_STEP_MAGIC, DELTA_MAGIC, DUAL_STEP_DELTA_MAGIC,
              DYNAMIC_MAGIC, *SPATIAL_MAGICS)


WRAPPERS = {
    b'SPG5': (b'SPC5',b'SPL5'), b'SPG6': (b'SPC6',b'SPL6'),
    b'SPG7': (b'SPC7',b'SPL7'), b'SPG8': (b'SPC8',b'SPL8'),
    MAGIC: (COMPRESSED_MAGIC, LZMA_COMPRESSED_MAGIC),
    DUAL_STEP_MAGIC: (DUAL_STEP_COMPRESSED_MAGIC, DUAL_STEP_LZMA_COMPRESSED_MAGIC),
    DELTA_MAGIC: (DELTA_COMPRESSED_MAGIC, DELTA_LZMA_COMPRESSED_MAGIC),
    DUAL_STEP_DELTA_MAGIC: (DUAL_STEP_DELTA_COMPRESSED_MAGIC,
                            DUAL_STEP_DELTA_LZMA_COMPRESSED_MAGIC),
    DYNAMIC_MAGIC: (DYNAMIC_COMPRESSED_MAGIC, DYNAMIC_LZMA_COMPRESSED_MAGIC),
    SPATIAL_MAGIC: (SPATIAL_COMPRESSED_MAGIC, SPATIAL_LZMA_COMPRESSED_MAGIC),
    RELATIVE_SPATIAL_MAGIC: (b'SPC2', b'SPL2'),
    TRANSLATION_ONLY_SPATIAL_MAGIC: (b'SPC3', b'SPL3'),
    RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC: (b'SPC4', b'SPL4'),
}


INNER_MAGIC = {wrapper: raw for raw, pair in WRAPPERS.items()
               for wrapper in pair}


@dataclass
class DecodedState:
    xyz: torch.Tensor
    rotation: torch.Tensor
    owner: torch.Tensor
    parents: torch.Tensor
    frame_index: int
    birth_sources: torch.Tensor | None = None
    spatial_paths: torch.Tensor | None = None
    spatial_open: torch.Tensor | None = None
    spatial_centers: torch.Tensor | None = None
    spatial_support_ball: torch.Tensor | None = None
    spatial_leaf_centers: torch.Tensor | None = None
    spatial_leaf_nodes: torch.Tensor | None = None
    # Packet-local previous row IDs, before appending births. Not part of the
    # materialized state digest: the state alone determines the next frame.
    survivor_rows: torch.Tensor | None = None
    # Packet-local current old-row IDs carrying a transmitted point-translation
    # residual. Attribute side streams may reuse this support without changing
    # the geometry-state digest used to link consecutive motion packets.
    gs_translation_rows: torch.Tensor | None = None

    def capture(self):
        return dict(xyz=self.xyz, rotation=self.rotation, owner=self.owner,
                    parents=self.parents, frame_index=self.frame_index,
                    birth_sources=self.birth_sources,
                    spatial_paths=self.spatial_paths, spatial_open=self.spatial_open,
                    spatial_centers=self.spatial_centers,
                    survivor_rows=self.survivor_rows,
                    gs_translation_rows=self.gs_translation_rows,
                    **{name: getattr(self, name) for name in SPATIAL_ROUTING_FIELDS})

    @classmethod
    def restore(cls, packet):
        # torch.load(map_location='cuda') moves checkpoint tensors, but decoded
        # states are CPU codec data. Preserve every tensor's saved dtype.
        fields = ('xyz', 'rotation', 'owner', 'parents', 'birth_sources',
                  'spatial_paths', 'spatial_open', 'spatial_centers',
                  'survivor_rows', 'gs_translation_rows', *SPATIAL_ROUTING_FIELDS)
        restored = {name: (None if packet.get(name) is None else
                           torch.as_tensor(packet[name]).detach().cpu().contiguous())
                    for name in fields}
        return cls(**restored, frame_index=int(packet['frame_index']))

    def digest(self):
        digest = hashlib.sha256()
        digest.update(struct.pack('<q', int(self.frame_index)))
        for tensor in (self.xyz, self.rotation, self.owner, self.parents):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(memoryview(value.numpy()))
        if self.spatial_paths is not None:
            for tensor in (self.spatial_paths, self.spatial_open):
                value = tensor.detach().cpu().contiguous()
                digest.update(str(tuple(value.shape)).encode('ascii'))
                digest.update(memoryview(value.numpy()))
        if self.spatial_centers is not None:
            value = self.spatial_centers.detach().cpu().contiguous()
            digest.update(b'previous_decoded_position_routing_v1')
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(memoryview(value.numpy()))
        if self.spatial_support_ball is not None:
            digest.update(b'nearest_supported_leaf_routing_v1')
            for name in SPATIAL_ROUTING_FIELDS:
                value = getattr(self, name).detach().cpu().contiguous()
                digest.update(str(tuple(value.shape)).encode('ascii'))
                digest.update(memoryview(value.numpy()))
        return digest.digest()

    @classmethod
    def from_reference(cls, packet):
        def read(name):
            return torch.from_numpy(packet[name].copy()) if name in packet else None
        xyz, rotation, owner = read('xyz'), read('rotation'), read('owner')
        parents, paths = read('parents').long(), read('spatial_paths')
        opened, centers = read('spatial_open'), read('spatial_centers')
        routing = {name: read(name) for name in SPATIAL_ROUTING_FIELDS}
        if 'spatial_derived_depth' in packet:
            value = np.asarray(packet['spatial_derived_depth'])
            if (value.ndim != 0 or value.dtype.kind not in 'iu' or
                    owner is not None or paths is not None):
                raise ValueError('Invalid derived spatial reference metadata')
            depth = int(value)
            if (not 1 <= depth <= 5 or centers is None or opened is None or
                    opened.dtype != torch.bool or opened.shape != parents.shape or
                    len(parents) != sum(8 ** level for level in range(depth + 1))):
                raise ValueError('Invalid derived spatial reference layout')
            paths = spatial_route(xyz, centers, depth, routing=routing)
            owner = spatial_owner(paths, opened)
        if owner is None:
            raise ValueError('Reference has no Gaussian ownership')
        return cls(xyz, rotation, owner.long(), parents, int(packet['frame_index']),
                   spatial_paths=paths, spatial_open=opened, spatial_centers=centers,
                   **routing)


def compact_reference_fields(fields):
    """Omit routes and owners only when the stored reference reproduces them exactly."""
    result = dict(fields)
    if any(name not in fields for name in
           ('owner', 'spatial_paths', 'spatial_open', 'spatial_centers')):
        return result
    state = DecodedState.from_reference(fields)
    _validate_state(state)
    depth = state.spatial_paths.shape[1]
    if not 1 <= depth <= 5:
        return result
    paths = spatial_state_route(state.xyz, state)
    if (torch.equal(paths, state.spatial_paths) and
            torch.equal(spatial_owner(paths, state.spatial_open), state.owner)):
        result.pop('owner')
        result.pop('spatial_paths')
        result['spatial_derived_depth'] = np.array(depth, dtype=np.uint8)
    return result


def write_reference_npz(path, fields):
    """Write a NumPy-compatible archive using lossless ZIP LZMA members."""
    with ZipFile(path, mode='w', compression=ZIP_LZMA, allowZip64=True) as archive:
        for name, value in fields.items():
            with archive.open(f'{name}.npy', mode='w', force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)


def _uvar(value):
    value = int(value)
    if value < 0:
        raise ValueError('Stream integers must be nonnegative')
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _verify_raw_checksum(raw):
    if (len(raw) < len(MAGIC) + 32 + CHECKSUM_BYTES or
            hashlib.sha256(raw[:-CHECKSUM_BYTES]).digest()[:CHECKSUM_BYTES] !=
            raw[-CHECKSUM_BYTES:]):
        raise ValueError('Rigid motion stream checksum mismatch')


class _Reader:
    def __init__(self, payload):
        self.payload = memoryview(payload)
        self.cursor = 0

    def take(self, size):
        if size < 0 or self.cursor + size > len(self.payload):
            raise ValueError('Truncated rigid motion stream')
        start = self.cursor
        self.cursor += size
        return self.payload[start:self.cursor]

    def integer(self):
        value = 0
        for shift in range(0, 70, 7):
            byte = self.take(1)[0]
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError('Oversized rigid stream integer')


def _cpu(tensor, dtype, shape=None):
    tensor = torch.as_tensor(tensor).detach().cpu().contiguous().to(dtype)
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f'Expected shape {shape}, got {tuple(tensor.shape)}')
    return tensor


def _validate_state(state):
    if not isinstance(state, DecodedState):
        raise ValueError('Decoder requires the previous decoded state')
    if (state.xyz.ndim != 2 or state.xyz.shape[1] != 3 or
            state.rotation.shape != (len(state.xyz), 4) or
            state.owner.shape != (len(state.xyz),) or
            state.parents.ndim != 1 or len(state.parents) < 1):
        raise ValueError('Malformed previous decoded state')
    if (state.xyz.dtype != torch.float32 or state.rotation.dtype != torch.float32 or
            state.owner.dtype != torch.long or state.parents.dtype != torch.long):
        raise ValueError('Previous state requires float32 poses and int64 topology')
    _topological_order(state.parents)
    if bool(((state.owner < 0) | (state.owner >= len(state.parents))).any()):
        raise ValueError('Previous state has an invalid owner')
    if (not bool(torch.isfinite(state.xyz).all()) or
            not bool(torch.isfinite(state.rotation).all()) or
            bool(state.rotation.norm(dim=-1).le(0).any())):
        raise ValueError('Previous decoded pose is nonfinite or has zero rotation')
    if state.spatial_paths is not None or state.spatial_open is not None:
        paths, opened = state.spatial_paths, state.spatial_open
        if (paths is None or opened is None or paths.dtype != torch.long or
                opened.dtype != torch.bool or paths.ndim != 2 or
                paths.shape[0] != len(state.xyz) or paths.shape[1] < 1 or
                opened.shape != state.parents.shape or not bool(opened[0]) or
                bool(((paths < 1) | (paths >= len(state.parents))).any())):
            raise ValueError('Invalid spatial reference paths or gates')
        if (bool(state.parents[paths[:, 0]].ne(0).any()) or
                (paths.shape[1] > 1 and
                 not torch.equal(state.parents[paths[:, 1:]], paths[:, :-1])) or
                bool(opened[paths[:, -1]].any()) or
                not torch.equal(state.owner, spatial_owner(paths, opened))):
            raise ValueError('Spatial paths or gates disagree with the selected frontier')
    if state.spatial_centers is not None:
        centers = state.spatial_centers
        if (state.spatial_paths is None or centers.dtype != torch.float32 or
                centers.shape != (sum(8 ** level for level in
                    range(state.spatial_paths.shape[1])), 3) or
                len(state.parents) != sum(8 ** level for level in
                    range(state.spatial_paths.shape[1] + 1)) or
                not bool(torch.isfinite(centers).all())):
            raise ValueError('Invalid static spatial split planes')
    routing = {name: getattr(state, name) for name in SPATIAL_ROUTING_FIELDS}
    if any(value is not None for value in routing.values()):
        if state.spatial_centers is None:
            raise ValueError('Nearest-region routing requires static split planes')
        validate_spatial_routing(routing, state.spatial_paths.shape[1])


def _compose_decoded_pose(base_xyz, base_rotation, owner, parents, pivots,
                          rt, rr, gt, gr, extent, step_extent, step_rotation,
                          gs_step_extent, gs_step_rotation, *, spatial,
                          spatial_relative=False, inherit_birth_rotation=False):
    """Shared arithmetic for in-memory composition and actual packet replay."""
    # Header scalars are binary64, and symbols/decoded reference are evaluated
    # as CPU float32 exactly as in the wire decoder. In particular, do not
    # compute tensor steps in float64 then cast the final pose.
    extent, step_extent, step_rotation, gs_step_extent, gs_step_rotation = map(
        float, (extent, step_extent, step_rotation, gs_step_extent, gs_step_rotation))
    if spatial_relative:
        rt = spatial_accumulate(rt.to(torch.int32), parents)
        rr = spatial_accumulate(rr.to(torch.int32), parents)
    values = (rt.float() * (extent * step_extent), rr.float() * step_rotation,
              gt.float() * (extent * gs_step_extent), gr.float() * gs_step_rotation)
    if spatial:
        xyz, rotation = compose_spatial_pose(base_xyz, base_rotation, owner, *values)
        if inherit_birth_rotation:
            rotation = base_rotation.clone()
        return xyz, rotation
    return compose_pose(base_xyz, base_rotation, owner, parents, pivots, *values)


@torch.no_grad()
def compose_decoded_survivors(previous_state, extent, previous_rows, owner, parents,
                              rt, rr, gt, gr, translation_step_extent,
                              rotation_step_rad, *, gs_translation_step_extent=None,
                              gs_rotation_step_rad=None, spatial_open=None,
                              spatial_relative=False, inherit_birth_rotation=False,
                              region_map=None, pivots=None):
    """Return CPU float32 survivor poses identical to encode/decode, without I/O.

    Inputs are the same quantized motion symbols and survivor layout passed to
    encode_motion. This does not encode, compress, hash, or mutate codec state.
    The returned quaternion is the raw decoder quaternion: no extra normalize.
    """
    rows = _cpu(previous_rows, torch.long)
    count = len(rows)
    base_xyz = previous_state.xyz[rows].detach().cpu().float()
    base_rotation = previous_state.rotation[rows].detach().cpu().float()
    prior_owner = previous_state.owner[rows].detach().cpu().long()
    prior_parents = previous_state.parents.detach().cpu().long()
    parents = _cpu(parents, torch.long)
    owner = _cpu(owner, torch.long, (count,))
    regions = len(parents)
    rt, rr = (_cpu(v, torch.int16, (regions, 3)) for v in (rt, rr))
    gt, gr = (_cpu(v, torch.int16, (count, 3)) for v in (gt, gr))
    if spatial_open is not None:
        opened = _cpu(spatial_open, torch.bool, (regions,))
        paths = previous_state.spatial_paths[rows]
        if previous_state.spatial_centers is not None:
            paths = spatial_state_route(base_xyz, previous_state)
        decoded_owner = spatial_owner(paths, opened)
        if not torch.equal(owner, decoded_owner) or not torch.equal(parents, prior_parents):
            raise ValueError('Survivor composition needs the encoded spatial owner/tree')
        decoded_pivots = None
    elif region_map is not None:
        # Dynamic pivots are serialized as raw float32; copy that exact dtype.
        decoded_pivots = _cpu(pivots, torch.float32, (regions, 3))
    else:
        # Parent edits change this frame topology, never its prior-frame pivot.
        decoded_pivots = region_pivots(base_xyz, prior_owner, prior_parents)
    return _compose_decoded_pose(base_xyz, base_rotation, owner, parents, decoded_pivots,
        rt, rr, gt, gr, extent, translation_step_extent, rotation_step_rad,
        translation_step_extent if gs_translation_step_extent is None else gs_translation_step_extent,
        rotation_step_rad if gs_rotation_step_rad is None else gs_rotation_step_rad,
        spatial=spatial_open is not None, spatial_relative=spatial_relative,
        inherit_birth_rotation=inherit_birth_rotation)


OPACITY_MAGIC = b'OPD1'


ALPHA_MAGIC = b'OPD2'


SATURATED_ALPHA_MAGIC = b'OPD3'


CANONICAL_ALPHA_MAGIC = b'OPD4'


MODES = {'logit': OPACITY_MAGIC, 'alpha': ALPHA_MAGIC, 'saturated_alpha': SATURATED_ALPHA_MAGIC,
         'alpha_canonical': CANONICAL_ALPHA_MAGIC}


SATURATION_LOGIT_THRESHOLD = struct.unpack('<f', struct.pack('<f', math.log(.99 / .01)))[0]


SATURATED_ALPHA_SCALE = .25


OPACITY_HEADER = struct.Struct('<4sIIIfII32s')


MAX_OPACITY_BODY_BYTES = 64 * 1024 * 1024


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


def _opacity_varint(value):
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
    digest = hashlib.sha256(struct.pack('<I', len(value)))
    digest.update(memoryview(value.numpy().astype('<f4', copy=False)))
    return digest.digest()


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
            if value > 0xffffffff or _opacity_varint(value) != body[start:offset]:
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


BIRTH_MAGIC = b'RTB1'  # Legacy absolute float32 appearance.


BIRTH_HEADER = struct.Struct('<4sIIIII')


SYMBOL_MAGIC = b'RTB2'


SYMBOL_HEADER = struct.Struct('<4sIIH4B4fII')


MAX_BIRTH_BODY_BYTES = 256 * 1024 * 1024


ATTRIBUTES = ('logscale', 'opacity', 'sh_dc', 'sh_rest')


def _shapes(count, rest):
    return dict(logscale=(count, 3), opacity=(count, 1),
                sh_dc=(count, 1, 3), sh_rest=(count, rest, 3))


FIELDS = ('logscale', 'sh_dc', 'sh_rest')


PLANAR_MAGIC = b'FUA3'


HEADER = struct.Struct('<4sIII3dI32sI')


MAX_BODY = 512 * 1024 * 1024


def _reference_hash(reference):
    digest = hashlib.sha256()
    for name in FIELDS:
        value = reference[name].detach().cpu().contiguous()
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError('Full attribute references must be finite float32')
        digest.update(name.encode())
        digest.update(struct.pack('<I', value.ndim))
        digest.update(struct.pack('<' + 'I' * value.ndim, *value.shape))
        digest.update(memoryview(value.numpy()))
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


def validate_sparse_color_selection(symbols, selected, budget, allowed_fields=('sh_dc',)):
    """Encoder-only check; a receiver cannot reconstruct the training selection."""
    selected = torch.as_tensor(selected).detach().cpu()
    if type(budget) is not int or budget < 1 or selected.dtype != torch.bool or selected.ndim != 1:
        raise ValueError('Invalid sparse color training selection')
    if set(symbols) != set(FIELDS) or any(len(v) != len(selected) for v in symbols.values()):
        raise ValueError('Sparse color selection row count differs')
    if allowed_fields not in (('sh_dc',), ('sh_dc', 'logscale')):
        raise ValueError('Invalid sparse attribute authorization')
    if int(selected.sum()) > budget:
        raise ValueError('Color training selection exceeds budget')
    for name, value in symbols.items():
        value = value.detach().cpu()
        if (name not in allowed_fields and value.ne(0).any()) or value[~selected].ne(0).any():
            raise ValueError('Sparse color symbols violate training selection or field authorization')


def _sparse_zigzag_byteplane(body, count, rest, *, inverse=False):
    """Reversibly transform sparse int16 values, retaining row gaps.

    Values are zigzagged uint16, with low bytes followed by high bytes.
    """
    reader = _IntegerReader(body)
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


COLOR_MASK_HEADER=struct.Struct('<4sIII')


FRAME_FIELDS = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')


RAW_MAGIC = b'SLS1'


ZLIB_MAGIC = b'SLZ1'


ANISOTROPIC_RAW_MAGIC = b'SLA1'


ANISOTROPIC_ZLIB_MAGIC = b'ZLA1'


LOGSCALE_HEADER = struct.Struct('<4sIIId32sII')


MAX_LOGSCALE_BODY_BYTES = 16 * 1024 * 1024


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
    digest.update(memoryview(reference.numpy()))
    digest.update(memoryview(allowed_rows.numpy()))
    return digest.digest()


@dataclass(frozen=True)
class DecodedLogScaleUpdate:
    logscale: torch.Tensor
    rows: torch.Tensor
    symbols: torch.Tensor
    step: float
    prior: bytes


FEATURE_MAGIC = b'SDC1'


REGIONAL_FEATURE_MAGIC = b'SDR1'


def _regional_owner(owner, node_count, rows):
    if (isinstance(node_count, bool) or not isinstance(node_count, int) or node_count <= 0 or
            owner is None):
        raise ValueError('Regional DC updates require the decoded motion owner and node count')
    owner = torch.as_tensor(owner).detach().cpu().contiguous()
    if (owner.dtype != torch.long or owner.shape != (rows,) or
            bool(((owner < 0) | (owner >= node_count)).any())):
        raise ValueError('Regional DC ownership differs from the surviving GS layout')
    return owner


def _regional_digest(reference, owner, node_count):
    digest = hashlib.sha256(struct.pack('<QQ', node_count, len(reference)))
    digest.update(memoryview(reference.detach().cpu().contiguous().numpy()))
    digest.update(memoryview(owner.detach().cpu().contiguous().numpy()))
    return digest.digest()


def _merge_regional_dc(reference, delta):
    # A zero update must preserve signed zero and all other reference bits.
    return torch.where(delta.ne(0), reference + delta, reference)


REGIONAL_ATTRIBUTE_MAGIC = b'RFA1'


REGIONAL_ATTRIBUTE_HEADER = struct.Struct('<4sIII')


def _regional_attribute_zeros(nodes, rest, device='cpu'):
    return dict(logscale=torch.zeros(nodes, 3, device=device),
                sh_dc=torch.zeros(nodes, 1, 3, device=device),
                sh_rest=torch.zeros(nodes, rest, 3, device=device))


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
