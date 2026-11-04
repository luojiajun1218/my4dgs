"""Independent sequential frame stream for the rigid residual tree.

The decoder needs only the previous decoded state and this packet. It derives
each pivot from the previous state before applying owner/parent edits, then
reconstructs the current pose from sparse signed-int16 motion symbols.
"""

from dataclasses import dataclass
import hashlib
import lzma
import math
import struct
import zlib

import numpy as np
import torch

from .rigid_residual_tree import (compose_pose,
                                  region_pivots, _topological_order)
from .spatial_gate import (compose_spatial_pose, spatial_owner, spatial_state_route,
                           SPATIAL_ROUTING_FIELDS, validate_spatial_routing,
                           spatial_accumulate)
from .sparse_integer_codec import (MOTION_MAGICS, pack_layout, unpack_layout,
                                   encode_motion_layout, decode_motion_layout)


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
            digest.update(value.numpy().tobytes())
        if self.spatial_paths is not None:
            for tensor in (self.spatial_paths, self.spatial_open):
                value = tensor.detach().cpu().contiguous()
                digest.update(str(tuple(value.shape)).encode('ascii'))
                digest.update(value.numpy().tobytes())
        if self.spatial_centers is not None:
            value = self.spatial_centers.detach().cpu().contiguous()
            digest.update(b'previous_decoded_position_routing_v1')
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(value.numpy().tobytes())
        if self.spatial_support_ball is not None:
            digest.update(b'nearest_supported_leaf_routing_v1')
            for name in SPATIAL_ROUTING_FIELDS:
                value = getattr(self, name).detach().cpu().contiguous()
                digest.update(str(tuple(value.shape)).encode('ascii'))
                digest.update(value.numpy().tobytes())
        return digest.digest()

    @classmethod
    def from_reference(cls, packet):
        def read(name):
            return torch.from_numpy(packet[name].copy()) if name in packet else None
        return cls(read('xyz'), read('rotation'), read('owner').long(),
                   read('parents').long(), int(packet['frame_index']),
                   spatial_paths=read('spatial_paths'), spatial_open=read('spatial_open'),
                   spatial_centers=read('spatial_centers'),
                   **{name: read(name) for name in SPATIAL_ROUTING_FIELDS})


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


def compress_motion_packet(raw_packet, mode='auto'):
    """Store the smallest exact raw, zlib, or XZ packet in automatic mode."""
    if mode not in ('none', 'auto', 'zlib'):
        raise ValueError('Motion compression must be none, auto, or zlib')
    raw_packet = bytes(raw_packet)
    if raw_packet[:4] not in RAW_MAGICS:
        raise ValueError('Compression requires an uncompressed rigid motion packet')
    if len(raw_packet) > MAX_RAW_PACKET_BYTES:
        raise ValueError('Rigid motion packet exceeds decoder size limit')
    _verify_raw_checksum(raw_packet)
    if mode == 'none':
        return raw_packet
    compressed_magic, lzma_magic = WRAPPERS[raw_packet[:4]]
    zlib_packet = (compressed_magic + _uvar(len(raw_packet)) +
                   zlib.compress(raw_packet, level=6))
    if mode == 'zlib':
        return zlib_packet
    lzma_packet = (lzma_magic + _uvar(len(raw_packet)) +
                   lzma.compress(raw_packet, format=lzma.FORMAT_XZ, preset=9))
    candidates = [raw_packet, zlib_packet, lzma_packet]
    if raw_packet[:4] in SPATIAL_MAGICS:
        layout = encode_motion_layout(raw_packet)
        candidates.append(pack_layout(layout, len(raw_packet), MOTION_MAGICS))
    return min(candidates, key=len)


def uncompressed_motion_packet(packet, *, sparse_sections=None):
    """Return verified raw packet bytes from current or legacy storage."""
    packet = bytes(packet)
    if packet[:4] in MOTION_MAGICS:
        layout, raw_length = unpack_layout(packet, MOTION_MAGICS,
                                           max_raw_bytes=MAX_RAW_PACKET_BYTES)
        raw = decode_motion_layout(layout, raw_length, sparse_sections=sparse_sections)
        _verify_raw_checksum(raw)
        return raw
    if packet[:4] in RAW_MAGICS:
        if len(packet) > MAX_RAW_PACKET_BYTES:
            raise ValueError('Rigid motion packet exceeds decoder size limit')
        _verify_raw_checksum(packet)
        return packet
    if packet[:4] not in INNER_MAGIC:
        raise ValueError('Unsupported rigid motion stream')
    reader = _Reader(packet)
    reader.take(len(COMPRESSED_MAGIC))
    raw_length = reader.integer()
    if raw_length < len(MAGIC) + 32 + CHECKSUM_BYTES or raw_length > MAX_RAW_PACKET_BYTES:
        raise ValueError('Invalid uncompressed rigid motion size')
    compressed = bytes(reader.take(len(packet) - reader.cursor))
    try:
        if packet[:4] in (pair[0] for pair in WRAPPERS.values()):
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed, raw_length + 1)
            extra = decoder.unconsumed_tail
        else:
            decoder = lzma.LZMADecompressor(
                format=lzma.FORMAT_XZ, memlimit=MAX_LZMA_MEMORY_BYTES)
            raw = decoder.decompress(compressed, max_length=raw_length + 1)
            extra = b''
    except (zlib.error, lzma.LZMAError, EOFError) as error:
        raise ValueError('Damaged compressed rigid motion stream') from error
    if (len(raw) != raw_length or not decoder.eof or
            decoder.unused_data or extra or
            raw[:4] != INNER_MAGIC[packet[:4]]):
        raise ValueError('Compressed rigid motion length or trailer mismatch')
    _verify_raw_checksum(raw)
    return raw


def _verify_raw_checksum(raw):
    if (len(raw) < len(MAGIC) + 32 + CHECKSUM_BYTES or
            hashlib.sha256(raw[:-CHECKSUM_BYTES]).digest()[:CHECKSUM_BYTES] !=
            raw[-CHECKSUM_BYTES:]):
        raise ValueError('Rigid motion stream checksum mismatch')


def motion_packet_info(packet):
    """Distinguish true stored bytes from the uncompressed stream size."""
    raw = uncompressed_motion_packet(packet)
    magic = bytes(packet[:4])
    formats = {**{magic:magic.decode()+'-parent-offset' for magic in (b'SPG5',b'SPC5',b'SPL5',b'SPG6',b'SPC6',b'SPL6',b'SPG7',b'SPC7',b'SPL7',b'SPG8',b'SPC8',b'SPL8')},MAGIC: 'RRT1-raw', DUAL_STEP_MAGIC: 'RRT2-raw',
               COMPRESSED_MAGIC: 'RRC1-zlib',
               DUAL_STEP_COMPRESSED_MAGIC: 'RRC2-zlib',
               LZMA_COMPRESSED_MAGIC: 'RRL1-lzma',
               DUAL_STEP_LZMA_COMPRESSED_MAGIC: 'RRL2-lzma',
               DELTA_MAGIC: 'RSD1-raw',
               DUAL_STEP_DELTA_MAGIC: 'RSD2-raw',
               DELTA_COMPRESSED_MAGIC: 'RSC1-zlib',
               DUAL_STEP_DELTA_COMPRESSED_MAGIC: 'RSC2-zlib',
               DELTA_LZMA_COMPRESSED_MAGIC: 'RSL1-lzma',
               DUAL_STEP_DELTA_LZMA_COMPRESSED_MAGIC: 'RSL2-lzma',
               DYNAMIC_MAGIC: 'RSD3-raw', DYNAMIC_COMPRESSED_MAGIC: 'RSC3-zlib',
               DYNAMIC_LZMA_COMPRESSED_MAGIC: 'RSL3-lzma',
               SPATIAL_MAGIC: 'SPG1-raw', SPATIAL_COMPRESSED_MAGIC: 'SPC1-zlib',
               SPATIAL_LZMA_COMPRESSED_MAGIC: 'SPL1-lzma',
               RELATIVE_SPATIAL_MAGIC: 'SPG2-relative-raw',
               b'SPC2': 'SPC2-relative-zlib', b'SPL2': 'SPL2-relative-lzma',
               b'SPG3': 'SPG3-translation-only-raw', b'SPC3': 'SPC3-translation-only-zlib',
               b'SPL3': 'SPL3-translation-only-lzma',
               b'SPG4': 'SPG4-relative-translation-only-raw', b'SPC4': 'SPC4-relative-translation-only-zlib',
               b'SPL4': 'SPL4-relative-translation-only-lzma',
               MOTION_MAGICS[0]: 'SVI1-planar-varint',
               MOTION_MAGICS[1]: 'SVZ1-planar-varint-zlib',
               MOTION_MAGICS[2]: 'SVX1-planar-varint-lzma'}
    return dict(format=formats[magic], stored_bytes=len(packet),
                raw_stream_bytes=len(raw), compression_savings_bytes=len(raw) - len(packet))


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


def _sparse_write(result, t, r):
    # Inputs have already been canonicalized on CPU. Avoid a tensor scalar
    # conversion and two tiny tensor slices for every transmitted point.
    t, r = np.asarray(t, dtype='<i2'), np.asarray(r, dtype='<i2')
    active_t, active_r = t.any(-1), r.any(-1)
    active = np.flatnonzero(active_t | active_r)
    result.extend(_uvar(len(active)))
    last = -1
    flags = active_t[active].astype(np.uint8) | (active_r[active].astype(np.uint8) << 1)
    for row, flags, translation, rotation in zip(active.tolist(), flags.tolist(),
                                                t[active].tolist(), r[active].tolist()):
        result.extend(_uvar(row - last))
        last = row
        result.append(flags)
        if flags & 1:
            result.extend(struct.pack('<3h', *translation))
        if flags & 2:
            result.extend(struct.pack('<3h', *rotation))


def _sparse_read(reader, count, *, delta_rows=False, sparse_sections=None):
    if sparse_sections is not None and reader.cursor in sparse_sections:
        span, expected_count, t_rows, t_values, r_rows, r_values = sparse_sections.pop(reader.cursor)
        if not delta_rows or count != expected_count:
            raise ValueError('Planar motion section differs from raw decoder dimensions')
        reader.take(span)
        t = np.zeros((count, 3), dtype='<i2')
        r = np.zeros((count, 3), dtype='<i2')
        t[t_rows] = t_values
        r[r_rows] = r_values
        return torch.from_numpy(t), torch.from_numpy(r)
    t = np.zeros((count, 3), dtype='<i2')
    r = np.zeros((count, 3), dtype='<i2')
    blocks = reader.integer()
    if blocks > count:
        raise ValueError('More motion blocks than IDs')
    last = -1
    for _ in range(blocks):
        value = reader.integer()
        if delta_rows and value < 1:
            raise ValueError('Sparse row gap must be positive')
        row = last + value if delta_rows else value
        flags = reader.take(1)[0]
        if row <= last or row >= count or flags not in (1, 2, 3):
            raise ValueError('Invalid sparse motion ID or block flags')
        last = row
        if flags & 1:
            values = struct.unpack('<3h', reader.take(6))
            if not any(values):
                raise ValueError('Redundant zero translation block')
            t[row] = values
        if flags & 2:
            values = struct.unpack('<3h', reader.take(6))
            if not any(values):
                raise ValueError('Redundant zero rotation block')
            r[row] = values
    return torch.from_numpy(t), torch.from_numpy(r)


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


def dynamic_layout_bytes(previous_state, previous_rows, parents, pivots, region_map):
    """Encode node replacement/growth and pivots; return remapped prior owners.

    A merge redirects the removed node's members with one map entry, instead of
    transmitting a new owner for every member. Unchanged frames carry one byte.
    Explicit pivots prevent new nodes or changed subtrees silently changing the
    rotation centers used by the fitted frame.
    """
    rows = _cpu(previous_rows, torch.long)
    parents = _cpu(parents, torch.long)
    regions = len(parents)
    _topological_order(parents)
    pivots = _cpu(pivots, torch.float32, (regions, 3))
    mapping = _cpu(region_map, torch.long, (len(previous_state.parents),))
    if (regions < 1 or int(mapping[0]) != 0 or
            bool(((mapping < 0) | (mapping >= regions)).any()) or
            not bool(torch.isfinite(pivots).all())):
        raise ValueError('Invalid dynamic region mapping or pivots')
    prior_owner = previous_state.owner[rows]
    unchanged = (regions == len(previous_state.parents) and
                 torch.equal(mapping, torch.arange(regions)) and
                 torch.equal(parents, previous_state.parents) and
                 torch.equal(pivots, region_pivots(
                     previous_state.xyz[rows], prior_owner, previous_state.parents)))
    result = bytearray([0 if unchanged else 1])
    if not unchanged:
        for node in mapping.tolist():
            result.extend(_uvar(node))
        for parent in parents[1:].tolist():
            result.extend(_uvar(parent))
        result.extend(pivots.numpy().astype('<f4', copy=False).tobytes())
    return result, mapping[prior_owner]


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


def encode_motion(previous_state, frame_index, extent, previous_rows, owner,
                  parents, region_t_i16, region_r_i16, gs_t_i16, gs_r_i16,
                  birth_source_rows, birth_xyz, birth_rotation,
                  translation_step_extent, rotation_step_rad, compression='none',
                  gs_translation_step_extent=None, gs_rotation_step_rad=None,
                  region_map=None, pivots=None, spatial_open=None,
                  spatial_relative=False, inherit_birth_rotation=False, birth_pose_steps=None):
    """Encode actual sparse translation/rotation blocks and topology deltas."""
    if birth_pose_steps is not None and spatial_open is None:
        raise ValueError('Parent-offset poses require a spatial stream')
    _validate_state(previous_state)
    if frame_index != previous_state.frame_index + 1:
        raise ValueError('Frame index must follow the previous decoded frame')
    previous_rows = _cpu(previous_rows, torch.long)
    count = len(previous_rows)
    dynamic = region_map is not None or pivots is not None
    spatial = spatial_open is not None
    if spatial_relative and not spatial:
        raise ValueError('Relative spatial motion requires spatial gates')
    if inherit_birth_rotation and not spatial:
        raise ValueError('Inherited birth rotation requires a spatial stream')
    if spatial and (dynamic or previous_state.spatial_paths is None):
        raise ValueError('Spatial stream needs a spatial reference and a fixed candidate tree')
    if previous_state.spatial_paths is not None and not spatial:
        raise ValueError('Spatial reference requires spatial motion semantics')
    if dynamic and (region_map is None or pivots is None):
        raise ValueError('Dynamic layout requires both region mapping and pivots')
    regions = len(parents) if dynamic else len(previous_state.parents)
    owner = _cpu(owner, torch.long, (count,))
    parents = _cpu(parents, torch.long, (regions,))
    rt = _cpu(region_t_i16, torch.int16, (regions, 3))
    rr = _cpu(region_r_i16, torch.int16, (regions, 3))
    gt = _cpu(gs_t_i16, torch.int16, (count, 3))
    gr = _cpu(gs_r_i16, torch.int16, (count, 3))
    sources = _cpu(birth_source_rows, torch.long)
    birth_xyz = _cpu(birth_xyz, torch.float32, (len(sources), 3))
    birth_rotation = _cpu(birth_rotation, torch.float32, (len(sources), 4))
    if (previous_rows.ndim != 1 or sources.ndim != 1 or not count or
            bool(((previous_rows < 0) | (previous_rows >= len(previous_state.xyz))).any()) or
            bool((previous_rows[1:] <= previous_rows[:-1]).any()) or
            bool(((sources < 0) | (sources >= count)).any()) or
            bool(((owner < 0) | (owner >= regions)).any())):
        raise ValueError('Invalid survivor, birth source, or owner ID')
    _topological_order(parents)
    if spatial:
        spatial_open = _cpu(spatial_open, torch.bool, (regions,))
        paths = previous_state.spatial_paths[previous_rows]
        if previous_state.spatial_centers is not None:
            paths = spatial_state_route(previous_state.xyz[previous_rows], previous_state)
        if (not torch.equal(parents, previous_state.parents) or
                not bool(spatial_open[0]) or bool(spatial_open[paths[:, -1]].any()) or
                not torch.equal(owner, spatial_owner(paths, spatial_open))):
            raise ValueError('Spatial owners must follow the encoded gates and reference routing')
    if bool(rt[0].ne(0).any()) or bool(rr[0].ne(0).any()):
        raise ValueError('Identity region cannot carry motion')
    if inherit_birth_rotation:
        expected = previous_state.rotation[previous_rows][sources]
        if bool(rr.ne(0).any()) or bool(gr.ne(0).any()) or not torch.equal(birth_rotation, expected):
            raise ValueError('Translation-only stream must preserve all old and inherited orientations')
    if (not math.isfinite(extent) or extent <= 0 or
            not math.isfinite(translation_step_extent) or translation_step_extent <= 0 or
            not math.isfinite(rotation_step_rad) or rotation_step_rad <= 0):
        raise ValueError('Invalid quantization metadata')
    dual_step = (gs_translation_step_extent is not None or
                 gs_rotation_step_rad is not None)
    if dual_step and (gs_translation_step_extent is None or
                      gs_rotation_step_rad is None or
                      not math.isfinite(gs_translation_step_extent) or
                      gs_translation_step_extent <= 0 or
                      not math.isfinite(gs_rotation_step_rad) or
                      gs_rotation_step_rad <= 0):
        raise ValueError('Dual-step stream needs positive GS translation and rotation steps')
    if dynamic and not dual_step:
        raise ValueError('Dynamic motion requires explicit region and GS steps')
    if spatial and not dual_step:
        raise ValueError('Spatial stream requires explicit region and GS steps')
    spatial_magic = ((RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC if spatial_relative else TRANSLATION_ONLY_SPATIAL_MAGIC)
                     if inherit_birth_rotation else RELATIVE_SPATIAL_MAGIC if spatial_relative else SPATIAL_MAGIC)
    if birth_pose_steps is not None:
        spatial_magic = ((b'SPG8' if spatial_relative else b'SPG7') if inherit_birth_rotation
                         else (b'SPG6' if spatial_relative else b'SPG5'))
    result = bytearray(spatial_magic if spatial else DYNAMIC_MAGIC if dynamic else
                       DUAL_STEP_DELTA_MAGIC if dual_step else DELTA_MAGIC)
    result.extend(previous_state.digest())
    for value in (frame_index, regions, count, len(sources)):
        result.extend(_uvar(value))
    result.extend(struct.pack('<5d' if dual_step else '<3d',
                              *((extent, translation_step_extent, rotation_step_rad,
                                 gs_translation_step_extent, gs_rotation_step_rad)
                                if dual_step else
                                (extent, translation_step_extent, rotation_step_rad))))
    if count == len(previous_state.xyz):
        if not torch.equal(previous_rows, torch.arange(count)):
            raise ValueError('Equal-count survivor mapping must be identity')
        result.append(0)
    else:
        result.append(1)
        keep = torch.ones(len(previous_state.xyz), dtype=torch.bool)
        keep[previous_rows] = False
        deleted = torch.where(keep)[0].tolist()
        result.extend(_uvar(len(deleted)))
        last = -1
        for row in deleted:
            result.extend(_uvar(row - last - 1))
            last = row
    if dynamic:
        layout, prior_owner = dynamic_layout_bytes(
            previous_state, previous_rows, parents, pivots, region_map)
        result.extend(layout)
    else:
        prior_owner = previous_state.owner[previous_rows]
    owner_edits = torch.where(owner != prior_owner)[0].tolist()
    if not dynamic:
        parent_edits = torch.where(parents != previous_state.parents)[0].tolist()
        result.extend(_uvar(len(parent_edits)))
        for node in parent_edits:
            if node == 0:
                raise ValueError('Cannot reparent identity region')
            result.extend(_uvar(node))
            result.extend(_uvar(int(parents[node])))
    if spatial:
        # A split changes thousands of owners, but only one region gate bit.
        gates = bytearray((regions + 7) // 8)
        for node in torch.where(spatial_open)[0].tolist():
            gates[node // 8] |= 1 << (node % 8)
        result.extend(gates)
    else:
        result.extend(_uvar(len(owner_edits)))
        for row in owner_edits:
            result.extend(_uvar(row))
            result.extend(_uvar(int(owner[row])))
    _sparse_write(result, rt[1:], rr[1:])
    _sparse_write(result, gt, gr)
    for source in sources.tolist():
        result.extend(_uvar(source))
    if birth_pose_steps is not None:
        steps=tuple(float(v) for v in birth_pose_steps)
        if len(steps)!=2 or not all(math.isfinite(v) and v>0 for v in steps):raise ValueError('Invalid birth pose steps')
        parent_xyz,parent_rotation=compose_decoded_survivors(previous_state,extent,previous_rows,
            owner,parents,rt,rr,gt,gr,translation_step_extent,rotation_step_rad,
            gs_translation_step_extent=gs_translation_step_extent,
            gs_rotation_step_rad=gs_rotation_step_rad,spatial_open=spatial_open,
            spatial_relative=spatial_relative, inherit_birth_rotation=inherit_birth_rotation)
        predictor=torch.cat((parent_xyz[sources],torch.nn.functional.normalize(parent_rotation[sources],dim=-1)),1)
        values=torch.cat((birth_xyz,birth_rotation),1)
        scale=values.new_tensor([steps[0]]*3+[steps[1]]*4)
        if inherit_birth_rotation:
            predictor, values, scale = predictor[:, :3], values[:, :3], scale[:3]
        symbols=torch.round((values-predictor)/scale).to(torch.int64)
        if bool((symbols.abs()>2147483647).any()):raise ValueError('Birth pose symbol exceeds signed32')
        result.extend(struct.pack('<2d',*steps))
        for value in symbols.flatten().tolist():result.extend(_uvar(2*value if value>=0 else -2*value-1))
    else:
        for xyz, rotation in zip(birth_xyz.tolist(), birth_rotation.tolist()):
            result.extend(struct.pack('<3f', *xyz) if inherit_birth_rotation else struct.pack('<3f4f', *xyz, *rotation))
    result.extend(hashlib.sha256(result).digest()[:CHECKSUM_BYTES])
    return compress_motion_packet(result, mode=compression)


def decode_motion(payload, prior_state):
    """Replay one frame and return its complete state for the next packet."""
    _validate_state(prior_state)
    sparse_sections = {}
    payload = uncompressed_motion_packet(payload, sparse_sections=sparse_sections)
    if len(payload) < 4 + 32 + CHECKSUM_BYTES:
        raise ValueError('Truncated rigid motion stream')
    if hashlib.sha256(payload[:-CHECKSUM_BYTES]).digest()[:CHECKSUM_BYTES] != payload[-CHECKSUM_BYTES:]:
        raise ValueError('Rigid motion stream checksum mismatch')
    reader = _Reader(payload[:-CHECKSUM_BYTES])
    stream_magic = bytes(reader.take(4))
    if stream_magic not in RAW_MAGICS:
        raise ValueError('Unsupported rigid motion stream')
    if bytes(reader.take(32)) != prior_state.digest():
        raise ValueError('Previous decoded state digest mismatch')
    frame_index, regions, count, birth_count = [reader.integer() for _ in range(4)]
    dynamic = stream_magic == DYNAMIC_MAGIC
    spatial = stream_magic in SPATIAL_MAGICS
    inherit_birth_rotation = stream_magic in TRANSLATION_ONLY_MAGICS
    if spatial != (prior_state.spatial_paths is not None):
        raise ValueError('Motion semantics differ from the decoder reference')
    if (frame_index != prior_state.frame_index + 1 or regions < 1 or
            (not dynamic and regions != len(prior_state.parents)) or count == 0 or
            count > len(prior_state.xyz) or birth_count > 10_000_000):
        raise ValueError('Invalid frame index, region count, or Gaussian count')
    dual_step = stream_magic in (DUAL_STEP_MAGIC, DUAL_STEP_DELTA_MAGIC, DYNAMIC_MAGIC, *SPATIAL_MAGICS)
    delta_rows = stream_magic in (DELTA_MAGIC, DUAL_STEP_DELTA_MAGIC, DYNAMIC_MAGIC, *SPATIAL_MAGICS)
    quantization = struct.unpack('<5d' if dual_step else '<3d',
                                 reader.take(40 if dual_step else 24))
    if not all(math.isfinite(v) and v > 0 for v in quantization):
        raise ValueError('Invalid motion quantization metadata')
    extent, step_extent, step_rotation = quantization[:3]
    gs_step_extent, gs_step_rotation = (quantization[3:]
                                        if dual_step else quantization[1:3])
    mode = reader.take(1)[0]
    if mode == 0:
        if count != len(prior_state.xyz):
            raise ValueError('Identity mapping has wrong count')
        rows = list(range(count))
    elif mode == 1:
        deleted_count = reader.integer()
        if deleted_count != len(prior_state.xyz) - count:
            raise ValueError('Deletion count differs from survivors')
        deleted = set()
        previous = -1
        for _ in range(deleted_count):
            previous += reader.integer() + 1
            if previous >= len(prior_state.xyz):
                raise ValueError('Deleted row outside previous state')
            deleted.add(previous)
        if len(deleted) != deleted_count:
            raise ValueError('Duplicate deleted row')
        rows = [row for row in range(len(prior_state.xyz)) if row not in deleted]
    else:
        raise ValueError('Unknown survivor mapping mode')
    rows = torch.tensor(rows, dtype=torch.long)
    base_xyz = prior_state.xyz[rows].detach().cpu().float()
    base_rotation = prior_state.rotation[rows].detach().cpu().float()
    prior_owner = prior_state.owner[rows].detach().cpu().long()
    parents = prior_state.parents.detach().cpu().long().clone()
    # Pivots are based on prior topology and prior decoded points, so edits
    # never silently alter the origin about which this frame's motion rotates.
    pivots = None if spatial else region_pivots(base_xyz, prior_owner, parents)
    if dynamic:
        layout_mode = reader.take(1)[0]
        if layout_mode == 0:
            if regions != len(parents):
                raise ValueError('Unchanged layout has a different node count')
        elif layout_mode == 1:
            if 13 * regions - 1 + len(parents) > len(reader.payload) - reader.cursor:
                raise ValueError('Dynamic layout exceeds remaining packet bytes')
            mapping = torch.tensor([reader.integer() for _ in range(len(parents))])
            if int(mapping[0]) != 0 or bool(((mapping < 0) | (mapping >= regions)).any()):
                raise ValueError('Invalid decoded region mapping')
            prior_owner = mapping[prior_owner]
            parents = torch.tensor([-1] + [reader.integer() for _ in range(regions - 1)])
            if bool(((parents[1:] < 0) | (parents[1:] >= regions)).any()):
                raise ValueError('Dynamic parent outside region range')
            pivots = torch.frombuffer(bytearray(reader.take(regions * 12)),
                                      dtype=torch.float32).reshape(regions, 3).clone()
            if not bool(torch.isfinite(pivots).all()):
                raise ValueError('Nonfinite dynamic pivots')
        else:
            raise ValueError('Unknown dynamic layout mode')
    else:
        parent_count = reader.integer()
        if parent_count > regions - 1:
            raise ValueError('Too many parent edits')
        last = 0
        for _ in range(parent_count):
            node = reader.integer()
            parent = reader.integer()
            if node <= last or node >= regions or parent >= regions or parent == int(parents[node]):
                raise ValueError('Invalid parent edit')
            parents[node] = parent
            last = node
    _topological_order(parents)
    owner = prior_owner.clone()
    paths, opened = None, None
    if spatial:
        if not torch.equal(parents, prior_state.parents):
            raise ValueError('Spatial candidate tree cannot be reparented')
        bits = reader.take((regions + 7) // 8)
        opened = torch.tensor([bool(bits[node // 8] & (1 << (node % 8)))
                               for node in range(regions)])
        paths = prior_state.spatial_paths[rows].clone()
        if prior_state.spatial_centers is not None:
            paths = spatial_state_route(prior_state.xyz[rows], prior_state)
        if not bool(opened[0]) or bool(opened[paths[:, -1]].any()):
            raise ValueError('Invalid spatial gate frontier')
        owner = spatial_owner(paths, opened)
    edit_count = 0 if spatial else reader.integer()
    if edit_count > count:
        raise ValueError('Too many owner edits')
    last = -1
    for _ in range(edit_count):
        row = reader.integer()
        new_owner = reader.integer()
        if (row <= last or row >= count or new_owner >= regions or
                new_owner == int(owner[row])):
            raise ValueError('Invalid owner edit')
        owner[row] = new_owner
        last = row
    rt, rr = _sparse_read(reader, regions - 1, delta_rows=delta_rows,
                          sparse_sections=sparse_sections)
    rt = torch.cat((torch.zeros(1, 3, dtype=torch.int16), rt))
    rr = torch.cat((torch.zeros(1, 3, dtype=torch.int16), rr))
    gt, gr = _sparse_read(reader, count, delta_rows=delta_rows,
                          sparse_sections=sparse_sections)
    if inherit_birth_rotation and (bool(rr.ne(0).any()) or bool(gr.ne(0).any())):
        raise ValueError('Translation-only stream contains rotation updates')
    offset_pose=stream_magic in OFFSET_SPATIAL_MAGICS
    pose_bytes = (3 if inherit_birth_rotation else 7) if offset_pose else 12 if inherit_birth_rotation else 28
    if birth_count > (len(reader.payload) - reader.cursor) // (pose_bytes + 1):
        raise ValueError('Birth count exceeds remaining stream bytes')
    sources = [reader.integer() for _ in range(birth_count)]
    if any(source >= count for source in sources):
        raise ValueError('Birth source outside surviving GS rows')
    if offset_pose:
        birth_steps=struct.unpack('<2d',reader.take(16))
        if not all(math.isfinite(v) and v>0 for v in birth_steps):raise ValueError('Invalid birth pose steps')
        packed=[reader.integer() for _ in range(birth_count*pose_bytes)]
        if any(value>4294967295 for value in packed):raise ValueError('Birth pose symbol exceeds signed32')
        births=[[-(v//2)-1 if v&1 else v//2 for v in packed[i:i+pose_bytes]] for i in range(0,len(packed),pose_bytes)]
    else:
        births = [struct.unpack('<3f' if inherit_birth_rotation else '<3f4f', reader.take(pose_bytes)) for _ in range(birth_count)]
    if reader.cursor != len(reader.payload):
        raise ValueError('Trailing rigid motion bytes')
    xyz, rotation = _compose_decoded_pose(base_xyz, base_rotation, owner, parents, pivots,
        rt, rr, gt, gr, extent, step_extent, step_rotation, gs_step_extent, gs_step_rotation,
        spatial=spatial,
        spatial_relative=stream_magic in (RELATIVE_SPATIAL_MAGIC, RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC,b'SPG6',b'SPG8'),
        inherit_birth_rotation=inherit_birth_rotation)
    if births:
        birth_xyz = torch.tensor([entry[:3] for entry in births], dtype=torch.float32)
        birth_rotation = (rotation[torch.tensor(sources, dtype=torch.long)].clone() if inherit_birth_rotation else
                          torch.tensor([entry[3:] for entry in births], dtype=torch.float32))
        if offset_pose:
            source=torch.tensor(sources,dtype=torch.long)
            birth_xyz=xyz[source]+birth_xyz*birth_xyz.new_tensor(birth_steps[0])
            if not inherit_birth_rotation:
                birth_rotation=torch.nn.functional.normalize(rotation[source],dim=-1)+birth_rotation*birth_rotation.new_tensor(birth_steps[1])
        xyz = torch.cat((xyz, birth_xyz))
        rotation = torch.cat((rotation, birth_rotation))
        owner = torch.cat((owner, owner[torch.tensor(sources, dtype=torch.long)]))
        if spatial:
            birth_paths = paths[torch.tensor(sources, dtype=torch.long)]
            if prior_state.spatial_centers is not None:
                birth_paths = spatial_state_route(birth_xyz, prior_state)
                owner[-birth_count:] = spatial_owner(birth_paths, opened)
            paths = torch.cat((paths, birth_paths))
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(rotation).all()):
        raise ValueError('Decoded pose contains nonfinite values')
    return DecodedState(xyz, rotation, owner, parents, frame_index,
                        torch.tensor(sources, dtype=torch.long), paths, opened,
                        prior_state.spatial_centers,
                        survivor_rows=rows,
                        gs_translation_rows=gt.ne(0).any(-1).nonzero(as_tuple=True)[0],
                        **{name: getattr(prior_state, name) for name in SPATIAL_ROUTING_FIELDS})
