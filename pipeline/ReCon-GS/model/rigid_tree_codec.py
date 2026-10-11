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

import torch

from .rigid_residual_tree import (compose_pose, initial_layout,
                                  region_pivots, _topological_order)


MAGIC = b'RRT1'
DUAL_STEP_MAGIC = b'RRT2'
COMPRESSED_MAGIC = b'RRC1'
DUAL_STEP_COMPRESSED_MAGIC = b'RRC2'
LZMA_COMPRESSED_MAGIC = b'RRL1'
DUAL_STEP_LZMA_COMPRESSED_MAGIC = b'RRL2'
CHECKSUM_BYTES = 16
MAX_RAW_PACKET_BYTES = 512 * 1024 * 1024
MAX_LZMA_MEMORY_BYTES = 128 * 1024 * 1024


@dataclass
class DecodedState:
    xyz: torch.Tensor
    rotation: torch.Tensor
    owner: torch.Tensor
    parents: torch.Tensor
    frame_index: int
    birth_sources: torch.Tensor | None = None

    def capture(self):
        return dict(xyz=self.xyz, rotation=self.rotation, owner=self.owner,
                    parents=self.parents, frame_index=self.frame_index,
                    birth_sources=self.birth_sources)

    @classmethod
    def restore(cls, packet):
        # torch.load(map_location='cuda') moves checkpoint tensors, but decoded
        # states are CPU codec data. Preserve every tensor's saved dtype.
        fields = ('xyz', 'rotation', 'owner', 'parents', 'birth_sources')
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
        return digest.digest()


def bootstrap_state(xyz, rotation, region_count, frame_index=0):
    """Rebuild the deterministic tree from the shared first-frame raw pose."""
    xyz = _cpu(xyz, torch.float32)
    rotation = _cpu(rotation, torch.float32, (len(xyz), 4))
    owner, parents = initial_layout(xyz, region_count)
    state = DecodedState(xyz, rotation, owner, parents, int(frame_index))
    _validate_state(state)
    return state


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
    if not raw_packet.startswith((MAGIC, DUAL_STEP_MAGIC)):
        raise ValueError('Compression requires an uncompressed rigid motion packet')
    if len(raw_packet) > MAX_RAW_PACKET_BYTES:
        raise ValueError('Rigid motion packet exceeds decoder size limit')
    _verify_raw_checksum(raw_packet)
    if mode == 'none':
        return raw_packet
    compressed_magic = (DUAL_STEP_COMPRESSED_MAGIC
                        if raw_packet.startswith(DUAL_STEP_MAGIC) else COMPRESSED_MAGIC)
    zlib_packet = (compressed_magic + _uvar(len(raw_packet)) +
                   zlib.compress(raw_packet, level=6))
    if mode == 'zlib':
        return zlib_packet
    lzma_magic = (DUAL_STEP_LZMA_COMPRESSED_MAGIC
                  if raw_packet.startswith(DUAL_STEP_MAGIC) else LZMA_COMPRESSED_MAGIC)
    lzma_packet = (lzma_magic + _uvar(len(raw_packet)) +
                   lzma.compress(raw_packet, format=lzma.FORMAT_XZ, preset=9))
    return min((raw_packet, zlib_packet, lzma_packet), key=len)


def uncompressed_motion_packet(packet):
    """Return verified raw packet bytes from current or legacy storage."""
    packet = bytes(packet)
    if packet.startswith((MAGIC, DUAL_STEP_MAGIC)):
        if len(packet) > MAX_RAW_PACKET_BYTES:
            raise ValueError('Rigid motion packet exceeds decoder size limit')
        _verify_raw_checksum(packet)
        return packet
    zlib_magic = (COMPRESSED_MAGIC, DUAL_STEP_COMPRESSED_MAGIC)
    lzma_magic = (LZMA_COMPRESSED_MAGIC, DUAL_STEP_LZMA_COMPRESSED_MAGIC)
    if not packet.startswith(zlib_magic + lzma_magic):
        raise ValueError('Unsupported rigid motion stream')
    reader = _Reader(packet)
    reader.take(len(COMPRESSED_MAGIC))
    raw_length = reader.integer()
    if raw_length < len(MAGIC) + 32 + CHECKSUM_BYTES or raw_length > MAX_RAW_PACKET_BYTES:
        raise ValueError('Invalid uncompressed rigid motion size')
    compressed = bytes(reader.take(len(packet) - reader.cursor))
    try:
        if packet[:4] in zlib_magic:
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
            not raw.startswith(DUAL_STEP_MAGIC if packet[:4] in (
                DUAL_STEP_COMPRESSED_MAGIC, DUAL_STEP_LZMA_COMPRESSED_MAGIC)
                else MAGIC)):
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
    formats = {MAGIC: 'RRT1-raw', DUAL_STEP_MAGIC: 'RRT2-raw',
               COMPRESSED_MAGIC: 'RRC1-zlib',
               DUAL_STEP_COMPRESSED_MAGIC: 'RRC2-zlib',
               LZMA_COMPRESSED_MAGIC: 'RRL1-lzma',
               DUAL_STEP_LZMA_COMPRESSED_MAGIC: 'RRL2-lzma'}
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
    active_t = t.ne(0).any(-1)
    active_r = r.ne(0).any(-1)
    active = torch.where(active_t | active_r)[0].tolist()
    result.extend(_uvar(len(active)))
    for row in active:
        flags = int(active_t[row]) | (int(active_r[row]) << 1)
        result.extend(_uvar(row))
        result.append(flags)
        if flags & 1:
            result.extend(struct.pack('<3h', *t[row].tolist()))
        if flags & 2:
            result.extend(struct.pack('<3h', *r[row].tolist()))


def _sparse_read(reader, count):
    t = torch.zeros(count, 3, dtype=torch.int16)
    r = torch.zeros(count, 3, dtype=torch.int16)
    blocks = reader.integer()
    if blocks > count:
        raise ValueError('More motion blocks than IDs')
    last = -1
    for _ in range(blocks):
        row = reader.integer()
        flags = reader.take(1)[0]
        if row <= last or row >= count or flags not in (1, 2, 3):
            raise ValueError('Invalid sparse motion ID or block flags')
        last = row
        if flags & 1:
            values = struct.unpack('<3h', reader.take(6))
            if not any(values):
                raise ValueError('Redundant zero translation block')
            t[row] = torch.tensor(values, dtype=torch.int16)
        if flags & 2:
            values = struct.unpack('<3h', reader.take(6))
            if not any(values):
                raise ValueError('Redundant zero rotation block')
            r[row] = torch.tensor(values, dtype=torch.int16)
    return t, r


def _validate_state(state):
    if not isinstance(state, DecodedState):
        raise ValueError('Decoder requires the previous decoded state')
    if (state.xyz.ndim != 2 or state.xyz.shape[1] != 3 or
            state.rotation.shape != (len(state.xyz), 4) or
            state.owner.shape != (len(state.xyz),) or
            state.parents.ndim != 1 or len(state.parents) < 2):
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


def encode_motion(previous_state, frame_index, extent, previous_rows, owner,
                  parents, region_t_i16, region_r_i16, gs_t_i16, gs_r_i16,
                  birth_source_rows, birth_xyz, birth_rotation,
                  translation_step_extent, rotation_step_rad, compression='none',
                  gs_translation_step_extent=None, gs_rotation_step_rad=None):
    """Encode actual sparse translation/rotation blocks and topology deltas."""
    _validate_state(previous_state)
    if frame_index != previous_state.frame_index + 1:
        raise ValueError('Frame index must follow the previous decoded frame')
    previous_rows = _cpu(previous_rows, torch.long)
    count = len(previous_rows)
    regions = len(previous_state.parents)
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
    if bool(rt[0].ne(0).any()) or bool(rr[0].ne(0).any()):
        raise ValueError('Identity region cannot carry motion')
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
    result = bytearray(DUAL_STEP_MAGIC if dual_step else MAGIC)
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
    prior_owner = previous_state.owner[previous_rows]
    owner_edits = torch.where(owner != prior_owner)[0].tolist()
    parent_edits = torch.where(parents != previous_state.parents)[0].tolist()
    result.extend(_uvar(len(parent_edits)))
    for node in parent_edits:
        if node == 0:
            raise ValueError('Cannot reparent identity region')
        result.extend(_uvar(node))
        result.extend(_uvar(int(parents[node])))
    result.extend(_uvar(len(owner_edits)))
    for row in owner_edits:
        result.extend(_uvar(row))
        result.extend(_uvar(int(owner[row])))
    _sparse_write(result, rt[1:], rr[1:])
    _sparse_write(result, gt, gr)
    for source in sources.tolist():
        result.extend(_uvar(source))
    for xyz, rotation in zip(birth_xyz.tolist(), birth_rotation.tolist()):
        result.extend(struct.pack('<3f4f', *xyz, *rotation))
    result.extend(hashlib.sha256(result).digest()[:CHECKSUM_BYTES])
    return compress_motion_packet(result, mode=compression)


def decode_motion(payload, prior_state):
    """Replay one frame and return its complete state for the next packet."""
    _validate_state(prior_state)
    payload = uncompressed_motion_packet(payload)
    if len(payload) < 4 + 32 + CHECKSUM_BYTES:
        raise ValueError('Truncated rigid motion stream')
    if hashlib.sha256(payload[:-CHECKSUM_BYTES]).digest()[:CHECKSUM_BYTES] != payload[-CHECKSUM_BYTES:]:
        raise ValueError('Rigid motion stream checksum mismatch')
    reader = _Reader(payload[:-CHECKSUM_BYTES])
    stream_magic = bytes(reader.take(4))
    if stream_magic not in (MAGIC, DUAL_STEP_MAGIC):
        raise ValueError('Unsupported rigid motion stream')
    if bytes(reader.take(32)) != prior_state.digest():
        raise ValueError('Previous decoded state digest mismatch')
    frame_index, regions, count, birth_count = [reader.integer() for _ in range(4)]
    if (frame_index != prior_state.frame_index + 1 or
            regions != len(prior_state.parents) or count == 0 or
            count > len(prior_state.xyz) or birth_count > 10_000_000):
        raise ValueError('Invalid frame index, region count, or Gaussian count')
    dual_step = stream_magic == DUAL_STEP_MAGIC
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
    pivots = region_pivots(base_xyz, prior_owner, parents)
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
    edit_count = reader.integer()
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
    rt, rr = _sparse_read(reader, regions - 1)
    rt = torch.cat((torch.zeros(1, 3, dtype=torch.int16), rt))
    rr = torch.cat((torch.zeros(1, 3, dtype=torch.int16), rr))
    gt, gr = _sparse_read(reader, count)
    if birth_count > (len(reader.payload) - reader.cursor) // 29:
        raise ValueError('Birth count exceeds remaining stream bytes')
    sources = [reader.integer() for _ in range(birth_count)]
    if any(source >= count for source in sources):
        raise ValueError('Birth source outside surviving GS rows')
    births = [struct.unpack('<3f4f', reader.take(28)) for _ in range(birth_count)]
    if reader.cursor != len(reader.payload):
        raise ValueError('Trailing rigid motion bytes')
    step_t = extent * step_extent
    xyz, rotation = compose_pose(
        base_xyz, base_rotation, owner, parents, pivots,
        rt.float() * step_t, rr.float() * step_rotation,
        gt.float() * (extent * gs_step_extent), gr.float() * gs_step_rotation)
    if births:
        birth_xyz = torch.tensor([entry[:3] for entry in births], dtype=torch.float32)
        birth_rotation = torch.tensor([entry[3:] for entry in births], dtype=torch.float32)
        xyz = torch.cat((xyz, birth_xyz))
        rotation = torch.cat((rotation, birth_rotation))
        owner = torch.cat((owner, owner[torch.tensor(sources, dtype=torch.long)]))
    if not bool(torch.isfinite(xyz).all()) or not bool(torch.isfinite(rotation).all()):
        raise ValueError('Decoded pose contains nonfinite values')
    return DecodedState(xyz, rotation, owner, parents, frame_index,
                        torch.tensor(sources, dtype=torch.long))
