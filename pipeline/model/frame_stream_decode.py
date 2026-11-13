"""逐帧码流的解码与重放入口。

负责读取并校验运动、区域和逐点属性数据包，按顺序还原帧状态；
支持现有 CPU 路径及部分 GPU 属性还原，码流定义和编码分别由对应文件维护。
"""

from dataclasses import dataclass
from functools import lru_cache
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

from .frame_stream_format import reference_digest
from .rigid_residual_tree import compose_pose, region_pivots, _topological_order
from .spatial_region_gate import (compose_spatial_pose, spatial_owner, spatial_route,
                           spatial_state_route, SPATIAL_ROUTING_FIELDS,
                           validate_spatial_routing, spatial_accumulate)

from .frame_stream_format import (
    ANISOTROPIC_RAW_MAGIC,
    ANISOTROPIC_ZLIB_MAGIC,
    ATTRIBUTES,
    BIRTH_HEADER,
    BIRTH_MAGIC,
    BirthAttributes,
    CHECKSUM_BYTES,
    COLOR_MASK_HEADER,
    COMPRESSED_MAGIC,
    DELTA_COMPRESSED_MAGIC,
    DELTA_LZMA_COMPRESSED_MAGIC,
    DELTA_MAGIC,
    DUAL_STEP_COMPRESSED_MAGIC,
    DUAL_STEP_DELTA_COMPRESSED_MAGIC,
    DUAL_STEP_DELTA_LZMA_COMPRESSED_MAGIC,
    DUAL_STEP_DELTA_MAGIC,
    DUAL_STEP_LZMA_COMPRESSED_MAGIC,
    DUAL_STEP_MAGIC,
    DYNAMIC_COMPRESSED_MAGIC,
    DYNAMIC_LZMA_COMPRESSED_MAGIC,
    DYNAMIC_MAGIC,
    DecodedLogScaleUpdate,
    DecodedOpacityDelta,
    DecodedState,
    FEATURE_MAGIC,
    FIELDS,
    FRAME_FIELDS,
    HEADER,
    INNER_MAGIC,
    LOGSCALE_HEADER,
    LZMA_COMPRESSED_MAGIC,
    MAGIC,
    MAX_BIRTH_BODY_BYTES,
    MAX_BODY,
    MAX_LOGSCALE_BODY_BYTES,
    MAX_LZMA_MEMORY,
    MAX_LZMA_MEMORY_BYTES,
    MAX_OPACITY_BODY_BYTES,
    MAX_RAW_PACKET_BYTES,
    MODES,
    MOTION_MAGICS,
    OFFSET_SPATIAL_MAGICS,
    OPACITY_HEADER,
    OPACITY_MAGICS,
    OpacitySymbols,
    PLANAR_MAGIC,
    RAW_MAGIC,
    RAW_MAGICS,
    REGIONAL_ATTRIBUTE_HEADER,
    REGIONAL_ATTRIBUTE_MAGIC,
    REGIONAL_FEATURE_MAGIC,
    RELATIVE_SPATIAL_MAGIC,
    RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC,
    SPATIAL_COMPRESSED_MAGIC,
    SPATIAL_LZMA_COMPRESSED_MAGIC,
    SPATIAL_MAGIC,
    SPATIAL_MAGICS,
    SYMBOL_HEADER,
    SYMBOL_MAGIC,
    StreamStep,
    TRANSLATION_ONLY_MAGICS,
    WRAPPERS,
    ZLIB_MAGIC,
    _IntegerReader,
    _Reader,
    _canonical_alpha_logits,
    _canonical_saturated_alpha_logits,
    _check_motion_tail,
    _check_reference,
    _compose_decoded_pose,
    _frame,
    _interleaved_rows,
    _merge_regional_dc,
    _opacity_header,
    _read_unsigned_values,
    _read_values,
    _read_varint,
    _reference,
    _reference_digest,
    _reference_hash,
    _regional_attribute_zeros,
    _regional_digest,
    _regional_owner,
    _rows,
    _shapes,
    _sparse_zigzag_byteplane,
    _spatial_prefix,
    _validate_state,
    _verify_raw_checksum,
    canonical_alpha_logits,
    opacity_reference_hash,
)


def unpack_layout(packet, magics, *, max_raw_bytes):
    reader = _IntegerReader(packet)
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
        rows = _read_unsigned_values(_IntegerReader(gaps), count).astype(np.int64).cumsum() - 1
        # Keep sparse arrays until the raw decoder has checked these declared
        # dimensions against its preceding state. Untrusted headers must not
        # allocate dense arrays before that check.
        decoded.append((rows[active_t], t, rows[active_r], r))
    return bytes(result)


def decode_motion_layout(layout, raw_length, *, sparse_sections=None):
    """Restore original bytes, optionally retaining symbols for the raw decoder.

    Section offsets refer to the restored raw bytes. The caller must verify
    their existing checksum before consuming any retained arrays.
    """
    reader = _IntegerReader(layout)
    prefix = bytes(reader.take(reader.integer()))
    prefix_reader = _IntegerReader(prefix)
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


def decode_opacity_layout(layout, raw_length, *, parsed=None):
    reader = _IntegerReader(layout)
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
        parsed['rows'] = _read_unsigned_values(_IntegerReader(gaps), nonzero).astype(np.int64).cumsum() - 1
        parsed['symbols'] = values
    return bytes(result)


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


def decode_motion(payload, prior_state, *, gpu_reference=None, gpu_capture=None,
                  verified_prior_digest=None):
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
    prior_digest = prior_state.digest() if verified_prior_digest is None else verified_prior_digest
    if not isinstance(prior_digest, bytes) or len(prior_digest) != 32:
        raise ValueError('Invalid verified previous-state digest')
    if bytes(reader.take(32)) != prior_digest:
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
        rows = torch.arange(count, dtype=torch.long)
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
        rows = torch.tensor([row for row in range(len(prior_state.xyz)) if row not in deleted],
                            dtype=torch.long)
    else:
        raise ValueError('Unknown survivor mapping mode')
    # The survivor mapping is ordered. When every point survives it is the
    # identity, so reuse the read-only reference instead of gathering it.
    identity = len(rows) == len(prior_state.xyz)
    selection = slice(None) if identity else rows
    base_xyz = prior_state.xyz[selection].detach().cpu().float()
    base_rotation = prior_state.rotation[selection].detach().cpu().float()
    prior_owner = prior_state.owner[selection].detach().cpu().long()
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
    # The prior state was already validated. A spatial packet without parent
    # edits keeps an exact clone of that topology, so no second walk is needed.
    if not (spatial and parent_count == 0):
        _topological_order(parents)
    owner = prior_owner.clone()
    paths, opened = None, None
    if spatial:
        if not torch.equal(parents, prior_state.parents):
            raise ValueError('Spatial candidate tree cannot be reparented')
        bits = reader.take((regions + 7) // 8)
        opened = torch.from_numpy(np.unpackbits(np.frombuffer(bits, dtype=np.uint8),
                                               bitorder='little')[:regions].astype(np.bool_))
        if prior_state.spatial_centers is None:
            paths = prior_state.spatial_paths[rows].clone()
        else:
            # base_xyz already is the exact survivor selection used for pose
            # composition; avoid gathering those positions and paths twice.
            paths = spatial_state_route(base_xyz, prior_state)
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
    if gpu_reference is not None and gpu_capture is not None:
        device = gpu_reference['xyz'].device
        gpu_rows = None if identity else rows.to(device)
        gpu_args = (gpu_reference['xyz'] if identity else gpu_reference['xyz'].index_select(0, gpu_rows),
                    gpu_reference['rotation'] if identity else gpu_reference['rotation'].index_select(0, gpu_rows),
                    owner.to(device), parents.to(device),
                    None if pivots is None else pivots.to(device),
                    rt.to(device), rr.to(device), gt.to(device), gr.to(device),
                    extent, step_extent, step_rotation, gs_step_extent, gs_step_rotation)
        gpu_xyz, gpu_rotation = _compose_decoded_pose(*gpu_args, spatial=spatial,
            spatial_relative=stream_magic in (RELATIVE_SPATIAL_MAGIC, RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC,b'SPG6',b'SPG8'),
            inherit_birth_rotation=inherit_birth_rotation)
        if births:
            gpu_sources = torch.tensor(sources, dtype=torch.long, device=device)
            gpu_birth_xyz = torch.tensor([entry[:3] for entry in births], dtype=torch.float32, device=device)
            gpu_birth_rotation = (gpu_rotation[gpu_sources].clone() if inherit_birth_rotation else
                torch.tensor([entry[3:] for entry in births], dtype=torch.float32, device=device))
            if offset_pose:
                gpu_birth_xyz = gpu_xyz[gpu_sources] + gpu_birth_xyz * birth_steps[0]
                if not inherit_birth_rotation:
                    gpu_birth_rotation = torch.nn.functional.normalize(gpu_rotation[gpu_sources], dim=-1) + gpu_birth_rotation * birth_steps[1]
            gpu_xyz = torch.cat((gpu_xyz, gpu_birth_xyz))
            gpu_rotation = torch.cat((gpu_rotation, gpu_birth_rotation))
        gpu_capture['xyz'], gpu_capture['rotation'] = gpu_xyz, gpu_rotation
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


def decode_opacity_delta(packet, reference, *, frame_index=None, gpu_reference=None, gpu_capture=None):
    """Check the exact preceding logits before applying signed integer deltas."""
    reference = _reference(reference)
    packet = bytes(packet)
    layout_body = None
    parsed = None
    if packet[:4] in OPACITY_MAGICS:
        layout, raw_length = unpack_layout(packet, OPACITY_MAGICS,
                                           max_raw_bytes=OPACITY_HEADER.size + MAX_OPACITY_BODY_BYTES)
        # The planar decoder already validates and materializes its rows and
        # symbols. Reuse them only after the original packet CRC is checked.
        parsed = {}
        raw = decode_opacity_layout(layout, raw_length, parsed=parsed)
        packet, layout_body = raw[:OPACITY_HEADER.size], raw[OPACITY_HEADER.size:]
    if len(packet) < OPACITY_HEADER.size:
        raise ValueError('Truncated opacity header')
    magic, frame, count, nonzero, step, length, checksum, prior = OPACITY_HEADER.unpack_from(packet)
    if (magic not in MODES.values() or frame == 0 or count != len(reference) or
            not 1 <= nonzero <= count or not nonzero * 3 <= length <= nonzero * 7 or
            length > MAX_OPACITY_BODY_BYTES or not math.isfinite(step) or step <= 0):
        raise ValueError('Invalid opacity packet header')
    if frame_index is not None and _frame(frame_index) != frame:
        raise ValueError('Opacity frame does not match motion frame')
    if prior != opacity_reference_hash(reference):
        raise ValueError('Opacity reference hash mismatch')
    if layout_body is None:
        try:
            decoder = zlib.decompressobj()
            body = decoder.decompress(packet[OPACITY_HEADER.size:], length + 1)
        except zlib.error as error:
            raise ValueError('Damaged opacity compressed body') from error
        if (len(body) != length or not decoder.eof or decoder.unused_data or
                decoder.unconsumed_tail):
            raise ValueError('Opacity body length or trailer mismatch')
    else:
        body = layout_body
    prefix = OPACITY_HEADER.pack(magic, frame, count, nonzero, step, length, 0, prior)
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
    if gpu_reference is not None and gpu_capture is not None:
        device = gpu_reference.device
        gpu_rows = rows.to(device)
        gpu_values = values.to(device).float().reshape(-1, 1)
        selected = gpu_reference.index_select(0, gpu_rows)

        def alpha_update(value, symbols, amount, *, canonical=False):
            alpha = value.double().sigmoid().float() if canonical else value.sigmoid()
            target = (alpha + symbols * amount).clamp(0., 1.)
            precise = target.double()
            middle = (precise.log() - torch.log1p(-precise)).float()
            return torch.where(target <= 0, -100., torch.where(target >= 1, 100., middle))

        if mode == 'alpha_canonical':
            updated = alpha_update(selected, gpu_values, step, canonical=True)
        elif mode == 'alpha':
            updated = alpha_update(selected, gpu_values, step)
        elif mode == 'saturated_alpha':
            from .frame_stream_format import SATURATION_LOGIT_THRESHOLD, SATURATED_ALPHA_SCALE, _step
            alpha_value = alpha_update(selected, gpu_values, _step(_step(step) * SATURATED_ALPHA_SCALE))
            updated = torch.where(selected >= SATURATION_LOGIT_THRESHOLD,
                                  alpha_value, selected + gpu_values * step)
        else:
            updated = selected + gpu_values * step
        gpu_logits = gpu_reference.clone()
        gpu_logits.index_copy_(0, gpu_rows, updated)
        gpu_capture['opacity'] = gpu_logits
    return DecodedOpacityDelta(logits, rows, values, int(frame), float(step), mode)


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


def decode_birth_attributes(packet, base, decoded_xyz, decoded_rotation,
                            birth_sources=None):
    packet = bytes(packet)
    magic = packet[:4]
    if magic not in (BIRTH_MAGIC, SYMBOL_MAGIC):
        raise ValueError('Unsupported birth appearance stream')
    if magic == BIRTH_MAGIC:
        if len(packet) < BIRTH_HEADER.size:
            raise ValueError('Truncated birth appearance stream')
        _, base_count, count, rest, length, checksum = BIRTH_HEADER.unpack_from(packet)
        expected = count * (7 + 3 * rest) * 4
        offset = BIRTH_HEADER.size
        if (base_count != len(base['xyz']) or rest != base['sh_rest'].shape[1] or
                count < 1 or length != expected or length > MAX_BIRTH_BODY_BYTES):
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
                length != expected or length > MAX_BIRTH_BODY_BYTES):
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
        if magic == BIRTH_MAGIC:
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


def apply_stream_opacity_update(directory, frame, current, old_count, manifest=None, *,
                                gpu_reference=None, gpu_capture=None):
    """Apply optional old-row logits at the timing specified by the manifest."""
    if manifest is None:
        import json
        path = directory / f'frame_{frame:03d}.json'
        manifest = json.loads(path.read_text()) if path.exists() else {}
    name = manifest.get('opacity_update')
    expected = f'opacity_{frame:03d}.bin'
    if name not in (None, expected):
        raise ValueError('Opacity manifest filename differs from expected frame packet')
    path = directory / expected
    if name is None:
        if path.exists():
            raise ValueError('Unexpected opacity packet absent from frame manifest')
        return current
    current['opacity'][:old_count] = decode_opacity_delta(
        path.read_bytes(), current['opacity'][:old_count], frame_index=frame,
        gpu_reference=gpu_reference, gpu_capture=gpu_capture).logits
    return current


def decode_frame(directory: Path, frame: int, state, raw, *, attribute_device='cpu',
                 gpu_reference=None, gpu_capture=None, verified_prior_digest=None,
                 manifest=None):
    """Materialize survivors first, then decode births using surviving sources."""
    manifest_path = directory / f'frame_{frame:03d}.json'
    if (directory / 'population_window.json').exists() and not manifest_path.is_file():
        raise ValueError('Population-window frame has not been committed by a manifest')
    motion_file = directory / f'motion_{frame:03d}.bin'
    decoded = decode_motion(motion_file.read_bytes(), state,
                            gpu_reference=gpu_reference, gpu_capture=gpu_capture,
                            verified_prior_digest=verified_prior_digest)
    rows = decoded.survivor_rows
    if rows is None:
        raise ValueError('Motion decoder lost previous-row survivor mapping')
    old_count = len(rows)
    if gpu_reference is not None and old_count != len(state.xyz):
        gpu_reference = {name: value.index_select(0, rows.to(value.device))
                         for name, value in gpu_reference.items()}
    birth_count = len(decoded.xyz) - old_count
    if any(len(value) != len(state.xyz) for value in raw.values()):
        raise ValueError('Previous raw attributes differ from decoded GS count')
    if manifest is None:
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Decoder survivor IDs are strictly ordered. If every old point survives,
    # the mapping is identity, so do not gather and clone each full tensor.
    # The decoded pose replaces xyz/rotation. A full-attribute packet creates
    # independent output tensors for these three fields; defer their copies
    # until that decoder rather than cloning each field twice.
    all_survive = old_count == len(state.xyz)
    full_will_copy = bool(manifest.get('full_attribute_update'))
    current = {}
    for name, value in raw.items():
        if name == 'xyz':
            current[name] = decoded.xyz[:old_count]
        elif name == 'rotation':
            current[name] = decoded.rotation[:old_count]
        elif all_survive and full_will_copy and name in ('logscale', 'sh_dc', 'sh_rest'):
            current[name] = value
        else:
            current[name] = value.clone() if all_survive else value.index_select(0, rows)
    timing = manifest.get('opacity_update_timing', 'after_births')
    if timing not in ('before_births', 'after_births'):
        raise ValueError('Unknown opacity update timing')
    if timing == 'before_births':
        current = apply_stream_opacity_update(directory, frame, current, old_count, manifest,
            gpu_reference=None if gpu_reference is None else gpu_reference.get('opacity'), gpu_capture=gpu_capture)
    feature_name = manifest.get('feature_dc_update')
    if feature_name is not None or manifest.get('logscale_update') is not None:
        # These optional earlier packets change the reference before FUA3.
        gpu_reference = None
    if feature_name is not None:
        if (feature_name != f'shdc_{frame:03d}.bin' or
                manifest.get('feature_dc_update_timing', 'before_births') != 'before_births'):
            raise ValueError('Shared color packet must precede newborn attribute prediction')
        current['sh_dc'] = decode_feature_dc_delta(
            (directory / feature_name).read_bytes(), current['sh_dc'], frame_index=frame,
            owner=decoded.owner[:old_count], node_count=len(decoded.parents))
    elif (directory / f'shdc_{frame:03d}.bin').exists():
        raise ValueError('Unexpected DC feature packet absent from frame manifest')
    color_version = manifest.get('color_stream_version')
    if color_version is not None and (type(color_version) is not int or color_version not in (2, 3)):
        raise ValueError('Unknown color stream version')
    if color_version == 3 and manifest.get('logscale_update') is not None:
        raise ValueError('Sparse attribute v3 forbids separate logscale packets')
    logscale_name = manifest.get('logscale_update')
    if logscale_name is not None:
        if (logscale_name != f'logscale_{frame:03d}.bin' or
                manifest.get('logscale_update_timing', 'before_births') != 'before_births'):
            raise ValueError('Log-scale packet must precede newborn attribute prediction')
        if decoded.gs_translation_rows is None:
            raise ValueError('Motion decoder lost point-translation support')
        current['logscale'] = decode_logscale_delta(
            (directory / logscale_name).read_bytes(), current['logscale'],
            allowed_rows=decoded.gs_translation_rows, frame_index=frame).logscale
    elif (directory / f'logscale_{frame:03d}.bin').exists():
        raise ValueError('Unexpected log-scale packet absent from frame manifest')
    if color_version in (2, 3):
        budget = manifest.get('color_point_budget')
        if (type(budget) is not int or budget < 1 or
                'color_point_mask' in manifest or 'color_point_mask_count' in manifest or
                manifest.get('full_update') is not True or
                'full_attribute_update' not in manifest or
                manifest.get('full_attribute_update_timing') != 'after_regional_dc_before_births' or
                (color_version == 2 and 'color_attribute_fields' in manifest) or
                (color_version == 3 and manifest.get('color_attribute_fields') != ['sh_dc', 'logscale'])):
            raise ValueError('Invalid sparse color stream manifest')
    point_mask_name = manifest.get('color_point_mask')
    point_mask = None
    expected_mask = f'pointmask_{frame:03d}.bin'
    if point_mask_name is not None:
        if point_mask_name != expected_mask:
            raise ValueError('Color point mask filename differs from frame')
        point_mask = torch.from_numpy(decode_color_point_mask((directory / point_mask_name).read_bytes(), frame, old_count))
        if (int(point_mask.sum()) != manifest.get('color_point_mask_count') or
                int(point_mask.sum()) > manifest.get('color_point_budget', -1)):
            raise ValueError('Color point mask metadata differs from packet')
    elif (directory / expected_mask).exists():
        raise ValueError('Unexpected color point mask packet absent from manifest')
    elif manifest.get('color_point_mask_count', 0):
        raise ValueError('Nonempty color point mask is missing')
    elif 'color_point_budget' in manifest and color_version not in (2, 3):
        point_mask = torch.zeros(old_count, dtype=torch.bool)
    full_name = manifest.get('full_attribute_update')
    if full_name is not None:
        if (manifest.get('full_update') is not True or
                full_name != f'fullattr_{frame:03d}.bin' or
                manifest.get('full_attribute_update_timing') != 'after_regional_dc_before_births'):
            raise ValueError('Full attributes must follow regional DC and precede newborn prediction')
        reference = {name: current[name] for name in ('logscale', 'sh_dc', 'sh_rest')}
        current.update(decode_full_attribute_delta((directory / full_name).read_bytes(),
                                                   reference, frame_index=frame,
                                                   color_only_budget=budget if color_version in (2, 3) else None,
                                                   allowed_fields=('sh_dc', 'logscale') if color_version == 3 else ('sh_dc',),
                                                   device=attribute_device, gpu_reference=gpu_reference,
                                                   gpu_capture=gpu_capture))
        if point_mask is not None and not torch.equal(current['sh_dc'][~point_mask], reference['sh_dc'][~point_mask]):
            raise ValueError('Decoded point DC changed outside its transmitted mask')
        if 'color_point_budget' in manifest and ((color_version != 3 and not torch.equal(current['logscale'], reference['logscale'])) or
                not torch.equal(current['sh_rest'], reference['sh_rest'])):
            raise ValueError('Color-only residual packet changed non-DC attributes')
    elif (directory / f'fullattr_{frame:03d}.bin').exists():
        raise ValueError('Unexpected full attribute packet absent from frame manifest')
    regional_name = manifest.get('regional_full_attributes')
    if regional_name is not None:
        if (regional_name != f'regfull_{frame:03d}.bin' or
                manifest.get('regional_full_timing') != 'after_point_attributes_before_births'):
            raise ValueError('Invalid regional full attribute packet timing')
        regional_gpu_reference = None if gpu_reference is None else {**gpu_reference, **(gpu_capture or {})}
        current.update(decode_regional_full((directory / regional_name).read_bytes(), current,
            decoded.owner[:old_count], len(decoded.parents), frame, device=attribute_device,
            gpu_reference=regional_gpu_reference,
            gpu_capture=gpu_capture))
    elif (directory / f'regfull_{frame:03d}.bin').exists():
        raise ValueError('Unexpected regional full attribute packet')
    birth_file = directory / f'birth_{frame:03d}.bin'
    if birth_count:
        newborn = decode_birth_attributes(
            birth_file.read_bytes(), current, decoded.xyz[old_count:],
            decoded.rotation[old_count:], decoded.birth_sources)
        sources = decoded.birth_sources
        if sources is None or len(sources) != birth_count:
            raise ValueError('Motion stream lost birth attachment rows')
        expected_owner = decoded.owner[:old_count][sources]
        if decoded.spatial_centers is not None:
            from importlib import import_module
            spatial = import_module('pipeline.model.spatial_region_gate')
            expected_paths = spatial.spatial_state_route(decoded.xyz[old_count:], decoded)
            expected_owner = spatial.spatial_owner(expected_paths, decoded.spatial_open)
        if not torch.equal(decoded.owner[old_count:], expected_owner):
            raise ValueError('Birth owner differs from encoded source owner')
        for name in ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest'):
            current[name] = torch.cat((current[name], newborn[name]))
        if (not torch.allclose(current['xyz'], decoded.xyz, rtol=1e-5, atol=1e-6) or
                not torch.allclose(current['rotation'], decoded.rotation, rtol=1e-5, atol=1e-6)):
            raise ValueError('Birth attributes disagree with decoded motion')
    elif birth_file.exists():
        raise ValueError('Unexpected birth stream for empty birth frame')
    if timing == 'after_births':
        current = apply_stream_opacity_update(directory, frame, current, old_count, manifest,
            gpu_reference=(None if gpu_reference is None else
                           (gpu_capture or {}).get('opacity', gpu_reference.get('opacity'))),
            gpu_capture=gpu_capture)
    return decoded, current


def decode_full_attribute_delta(packet, reference, *, frame_index, color_only_budget=None,
                                allowed_fields=('sh_dc',), device='cpu',
                                gpu_reference=None, gpu_capture=None):
    return _decode_full_attribute_delta(packet, reference, frame_index=frame_index,
        color_only_budget=color_only_budget, allowed_fields=allowed_fields, device=device,
        gpu_reference=gpu_reference, gpu_capture=gpu_capture)


def _decode_full_attribute_delta(packet, reference, *, frame_index, color_only_budget=None,
                                 allowed_fields=('sh_dc',), device='cpu',
                                 gpu_reference=None, gpu_capture=None,
                                 zero_reference_hash=None):
    reference = {k: v.detach().cpu().contiguous() for k, v in reference.items()}
    count, rest = _check_reference(reference)
    if len(packet) < HEADER.size:
        raise ValueError('Truncated full attribute packet')
    magic, frame, rows, coeff, *tail = HEADER.unpack_from(packet)
    if color_only_budget is not None:
        if isinstance(color_only_budget, bool) or not isinstance(color_only_budget, int) or color_only_budget < 0:
            raise ValueError('Invalid sparse color budget')
        if magic != PLANAR_MAGIC:
            raise ValueError('Sparse color stream requires FUA3')
        if allowed_fields not in (('sh_dc',), ('sh_dc', 'logscale')):
            raise ValueError('Invalid sparse attribute authorization')
    elif allowed_fields != ('sh_dc',):
        raise ValueError('Sparse field authorization requires a budget')
    steps, length, prior, checksum = tuple(tail[:3]), tail[3], tail[4], tail[5]
    sparse_max = 12 + count * (15 + 2 * (6 + 3 * rest))
    if (magic != PLANAR_MAGIC or frame != frame_index or rows != count or coeff != rest or
            not 12 <= length <= sparse_max or length > MAX_BODY or
            any(not math.isfinite(v) or v <= 0 for v in steps)):
        raise ValueError('Full attribute packet metadata or reference differs')
    expected_reference_hash = (_reference_hash(reference) if zero_reference_hash is None
                               else zero_reference_hash)
    if prior != expected_reference_hash:
        raise ValueError('Full attribute packet metadata or reference differs')
    decoder = zlib.decompressobj()
    try:
        body = decoder.decompress(packet[HEADER.size:], length + 1)
    except zlib.error as error:
        raise ValueError('Damaged full attribute packet') from error
    if len(body) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Invalid full attribute compressed length')
    prefix = HEADER.pack(magic, frame, rows, coeff, *steps, length, prior, 0)
    if zlib.crc32(prefix + body) != checksum:
        raise ValueError('Full attribute checksum mismatch')
    body = _sparse_zigzag_byteplane(body, count, rest, inverse=True)

    # Sparse streams contain complete rows only for changed points. Apply
    # their symbols directly, avoiding dense integer and float intermediates.
    # Keep torch.where so zero components preserve reference bits, including -0.
    reader = _IntegerReader(body)
    result = {}
    active_union = np.zeros(count, dtype=np.bool_) if color_only_budget is not None else None
    for name, step in zip(FIELDS, steps):
        value = reference[name]
        shape = value.shape
        width = math.prod(shape[1:])
        active = struct.unpack('<I', reader.take(4))[0]
        if color_only_budget is not None and ((name not in allowed_fields and active) or active > color_only_budget):
            raise ValueError('Sparse packet exceeds budget or changes unauthorized fields')
        if active > count or (width == 0 and active):
            raise ValueError('Invalid full attribute sparse row count')
        gaps = _read_unsigned_values(reader, active).astype(np.uint64)
        if bool((gaps == 0).any()) or (active and int(gaps.sum()) > count):
            raise ValueError('Invalid full attribute row gaps')
        indices = np.cumsum(gaps, dtype=np.uint64).astype(np.int64) - 1
        if active_union is not None:
            active_union[indices] = True
            if int(active_union.sum()) > color_only_budget:
                raise ValueError('Sparse attribute union exceeds point budget')
        values = np.frombuffer(reader.take(active * width * 2), dtype='<i2').copy().reshape(active, width)
        if active and (bool((values == -32768).any()) or bool((~np.any(values != 0, axis=1)).any())):
            raise ValueError('Invalid full attribute sparse symbols')
        cached = None if gpu_reference is None else gpu_reference.get(name)
        requested_device = torch.device(device)
        if cached is not None and (cached.shape != value.shape or
                cached.device.type != requested_device.type or
                (requested_device.index is not None and cached.device.index != requested_device.index)):
            raise ValueError('GPU attribute reference does not match the decoded layout')
        if gpu_capture is not None:
            # Keep the CPU reference for exact packet checks while applying
            # the same symbols independently to the resident GPU state.
            decoded = value if zero_reference_hash is not None else value.clone()
            if cached is not None:
                gpu_decoded = cached.clone()
            elif zero_reference_hash is not None:
                gpu_decoded = torch.zeros_like(value, device=device)
            else:
                gpu_decoded = value.to(device=device, copy=True)
            if active:
                rows = torch.from_numpy(indices)
                integer = torch.from_numpy(values.reshape((active, *shape[1:]))).to(dtype=value.dtype)
                selected = decoded.index_select(0, rows)
                decoded.index_copy_(0, rows, torch.where(integer.ne(0), selected + integer * step, selected))
                gpu_rows, gpu_integer = rows.to(device), integer.to(device)
                selected = gpu_decoded.index_select(0, gpu_rows)
                gpu_decoded.index_copy_(0, gpu_rows, torch.where(gpu_integer.ne(0), selected + gpu_integer * step, selected))
            gpu_capture[name] = gpu_decoded
            result[name] = decoded
        else:
            decoded = (value if zero_reference_hash is not None and torch.device(device).type == 'cpu'
                       else value.to(device=device, copy=True))
            if active:
                rows = torch.from_numpy(indices).to(device=device)
                integer = torch.from_numpy(values.reshape((active, *shape[1:]))).to(device=device, dtype=value.dtype)
                selected = decoded.index_select(0, rows)
                decoded.index_copy_(0, rows, torch.where(integer.ne(0), selected + integer * step, selected))
            result[name] = decoded.cpu()
    if reader.remaining:
        raise ValueError('Trailing full attribute sparse data')
    if active_union is not None and not active_union.any():
        raise ValueError('Empty sparse attribute packet must be omitted')
    return result


def decode_color_point_mask(packet,frame,point_count):
    if len(packet)<COLOR_MASK_HEADER.size:raise ValueError('Truncated point mask')
    magic,got_frame,n,count=COLOR_MASK_HEADER.unpack(packet[:COLOR_MASK_HEADER.size])
    if magic!=b'CPM1' or got_frame!=frame or n!=point_count or count<1 or count>n:
        raise ValueError('Point mask header mismatch')
    obj=zlib.decompressobj()
    try:
        data=obj.decompress(packet[COLOR_MASK_HEADER.size:],count*4+1)
        if obj.unconsumed_tail: raise ValueError('Oversized point mask')
        data+=obj.flush()
    except zlib.error as exc:raise ValueError('Invalid point mask compression') from exc
    if not obj.eof or obj.unused_data or len(data)!=count*4:raise ValueError('Invalid point mask length')
    rows=np.frombuffer(data,dtype='<u4').astype(np.int64)
    if np.any(rows>=n) or np.any(rows[1:]<=rows[:-1]):raise ValueError('Invalid point mask indices')
    mask=np.zeros(n,dtype=np.bool_);mask[rows]=True
    return mask


def replay(directory: Path, reference: Path, through: int, *, attribute_device='cpu'):
    with np.load(reference, allow_pickle=False) as packet:
        raw = {key: torch.from_numpy(packet[key].copy()) for key in FRAME_FIELDS}
        state = DecodedState.from_reference(packet)
    if through < state.frame_index:
        raise ValueError('Requested frame precedes the decoder reference')
    for frame in range(state.frame_index + 1, through + 1):
        manifest_path = directory / f'frame_{frame:03d}.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        decoded, current = decode_frame(directory, frame, state, raw,
                                        attribute_device=attribute_device, manifest=manifest)
        if manifest is not None:
            if (manifest['frame'] != frame or
                    manifest['decoded_state_sha256'] != decoded.digest().hex() or
                    manifest['raw_state_sha256'] != reference_digest(current)):
                raise ValueError('Frame manifest state digest mismatch')
        state, raw = decoded, current
    return state, raw


def _logscale_read_varint(body, cursor):
    value = 0
    for shift in range(0, 70, 7):
        if cursor >= len(body):
            raise ValueError('Truncated log-scale row gap')
        byte = body[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, cursor
    raise ValueError('Oversized log-scale row gap')


def decode_logscale_delta(packet, reference, *, allowed_rows, frame_index):
    packet = bytes(packet)
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    if (reference.dtype != torch.float32 or reference.ndim != 2 or
            reference.shape[1:] != (3,) or not bool(torch.isfinite(reference).all())):
        raise ValueError('Log-scale reference must be finite float32 [N, 3]')
    if len(packet) < LOGSCALE_HEADER.size:
        raise ValueError('Truncated log-scale packet')
    magic, index, count, nonzero, step, prior, length, checksum = LOGSCALE_HEADER.unpack_from(packet)
    magics = (RAW_MAGIC, ZLIB_MAGIC, ANISOTROPIC_RAW_MAGIC, ANISOTROPIC_ZLIB_MAGIC)
    channels = 1 if magic in (RAW_MAGIC, ZLIB_MAGIC) else 3
    if (magic not in magics or index != frame_index or count != len(reference) or
            not 1 <= nonzero <= count or not math.isfinite(step) or step <= 0 or
            length < (1 + channels) * nonzero or length > MAX_LOGSCALE_BODY_BYTES):
        raise ValueError('Log-scale metadata differs from the motion frame')
    encoded = packet[LOGSCALE_HEADER.size:]
    if magic in (RAW_MAGIC, ANISOTROPIC_RAW_MAGIC):
        body = encoded
    else:
        decoder = zlib.decompressobj()
        try:
            body = decoder.decompress(encoded, length + 1)
        except zlib.error as error:
            raise ValueError('Damaged compressed log-scale packet') from error
        if (len(body) != length or not decoder.eof or decoder.unused_data or
                decoder.unconsumed_tail):
            raise ValueError('Invalid compressed log-scale length or trailer')
    if len(body) != length:
        raise ValueError('Log-scale body length differs from its header')
    allowed_rows = _rows(allowed_rows, len(reference), name='Allowed log-scale rows')
    if prior != _reference_digest(reference, allowed_rows):
        raise ValueError('Log-scale reference or motion support differs')
    prefix = struct.pack('<IIId32sI', index, count, nonzero, step, prior, length)
    if zlib.crc32(prefix + (b'\x03' if channels == 3 else b'') + body) != checksum:
        raise ValueError('Log-scale checksum mismatch')
    rows, values, cursor, previous = [], [], 0, -1
    for _ in range(nonzero):
        gap, cursor = _logscale_read_varint(body, cursor)
        previous += gap
        if gap < 1 or previous >= count or cursor + channels > len(body):
            raise ValueError('Invalid log-scale update row')
        value = struct.unpack_from(f'<{channels}b', body, cursor)
        cursor += channels
        if not any(value):
            raise ValueError('Redundant zero log-scale update')
        rows.append(previous)
        values.append(value[0] if channels == 1 else value)
    if cursor != len(body):
        raise ValueError('Unexpected log-scale packet trailer')
    rows = torch.tensor(rows, dtype=torch.long)
    if not bool(torch.isin(rows, allowed_rows).all()):
        raise ValueError('Decoded log-scale rows extend outside motion support')
    symbols = torch.tensor(values, dtype=torch.int8)
    result = reference.clone()
    delta = symbols.to(reference.dtype) * step
    result[rows] = reference[rows] + (delta[:, None] if channels == 1 else delta)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Decoded log-scale update is nonfinite')
    return DecodedLogScaleUpdate(result, rows, symbols, float(step), prior)


def decode_feature_dc_delta(packet, reference, *, frame_index, owner=None, node_count=None):
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    if reference.ndim != 3 or reference.shape[1:] != (1, 3):
        raise ValueError('DC features must have shape [N, 1, 3]')
    if packet[:4] == REGIONAL_FEATURE_MAGIC:
        owner = _regional_owner(owner, node_count, len(reference))
        if len(packet) < 36 or packet[4:36] != _regional_digest(reference, owner, node_count):
            raise ValueError('Regional DC reference or motion ownership differs')
        zeros = reference.new_zeros(node_count * 3, 1)
        decoded = decode_opacity_delta(packet[36:], zeros, frame_index=frame_index)
        if decoded.mode != 'logit':
            raise ValueError('Regional DC updates require additive linear symbols')
        delta = decoded.logits.reshape(node_count, 1, 3)[owner]
        return _merge_regional_dc(reference, delta)
    if packet[:4] != FEATURE_MAGIC:
        raise ValueError('Unknown DC feature update stream')
    decoded = decode_opacity_delta(packet[4:], reference.reshape(-1, 1), frame_index=frame_index)
    if decoded.mode != 'logit':
        raise ValueError('DC updates require additive linear symbols')
    return decoded.logits.reshape_as(reference)


@lru_cache(maxsize=16)
def _regional_zero_reference_hash(nodes, rest):
    """The regional predictor is always a newly allocated all-zero tensor."""
    return _reference_hash(_regional_attribute_zeros(nodes, rest))


def decode_regional_full(packet, raw, owner, nodes, frame, *, device='cpu',
                         gpu_reference=None, gpu_capture=None):
    packet = bytes(packet)
    if len(packet) < REGIONAL_ATTRIBUTE_HEADER.size:
        raise ValueError('Truncated regional attributes')
    magic, encoded_frame, n_attr, n_alpha = REGIONAL_ATTRIBUTE_HEADER.unpack_from(packet)
    if (magic != REGIONAL_ATTRIBUTE_MAGIC or encoded_frame != frame or
            len(packet) != REGIONAL_ATTRIBUTE_HEADER.size + n_attr + n_alpha or not (n_attr or n_alpha)):
        raise ValueError('Invalid regional attribute packet')
    owner = torch.as_tensor(owner, dtype=torch.long, device='cpu')
    if owner.shape != (len(raw['opacity']),) or bool(((owner < 0) | (owner >= nodes)).any()):
        raise ValueError('Invalid regional attribute ownership')
    # A decoded update below produces a new tensor, so cloning its input
    # first would copy the same full attribute buffer twice. Keep independent
    # outputs for fields that the packet leaves untouched.
    result = {k: (raw[k].detach().cpu() if (n_attr and k in FIELDS) or (n_alpha and k == 'opacity')
                  else raw[k].detach().cpu().clone()) for k in (*FIELDS, 'opacity')}
    if n_attr:
        rest = raw['sh_rest'].shape[1]
        reference = _regional_attribute_zeros(nodes, rest)
        node_capture = {} if gpu_capture is not None else None
        delta = _decode_full_attribute_delta(packet[REGIONAL_ATTRIBUTE_HEADER.size:REGIONAL_ATTRIBUTE_HEADER.size+n_attr], reference,
            frame_index=frame, device=device, gpu_capture=node_capture,
            zero_reference_hash=_regional_zero_reference_hash(nodes, rest))
        device_owner = owner.to(device)
        for key in FIELDS:
            node_value = node_capture[key] if node_capture is not None else delta[key].to(device)
            value = node_value[device_owner]
            cached = None if gpu_reference is None else gpu_reference.get(key)
            reference = cached if cached is not None else result[key].to(device)
            updated = torch.where(value.ne(0), reference + value, reference)
            if gpu_capture is not None:
                gpu_capture[key] = updated
                cpu_value = delta[key][owner]
                result[key] = torch.where(cpu_value.ne(0), result[key] + cpu_value, result[key])
            else:
                result[key] = updated.cpu()
    if n_alpha:
        reference = torch.zeros(nodes, 1)
        cpu_offset = decode_opacity_delta(packet[REGIONAL_ATTRIBUTE_HEADER.size+n_attr:], reference,
                                          frame_index=frame).logits[owner]
        offset = cpu_offset.to(device)
        cached = None if gpu_reference is None else gpu_reference.get('opacity')
        reference = cached if cached is not None else result['opacity'].to(device)
        updated = torch.where(offset.ne(0), reference + offset, reference)
        if gpu_capture is not None:
            gpu_capture['opacity'] = updated
            result['opacity'] = torch.where(cpu_offset.ne(0), result['opacity'] + cpu_offset, result['opacity'])
        else:
            result['opacity'] = updated.cpu()
    return result


def _birth_attributes(packet, old_count, count, rest):
    packet = bytes(packet)
    magic = packet[:4]
    if magic == SYMBOL_MAGIC:
        if len(packet) < SYMBOL_HEADER.size:
            raise ValueError('Truncated birth symbol header')
        fields = SYMBOL_HEADER.unpack_from(packet)
        _, base, rows, coefficients = fields[:4]
        widths, steps, length, checksum = fields[4:8], fields[8:12], fields[12], fields[13]
        offset = SYMBOL_HEADER.size
        if (any(width not in (2,4) for width in widths) or
                any(not math.isfinite(step) or step <= 0 for step in steps)):
            raise ValueError('Invalid birth quantization metadata')
        dtype = lambda width: '<i2' if width == 2 else '<i4'
        prefix = packet[:offset-4]
    elif magic == BIRTH_MAGIC:
        if len(packet) < BIRTH_HEADER.size:
            raise ValueError('Truncated birth absolute header')
        _, base, rows, coefficients, length, checksum = BIRTH_HEADER.unpack_from(packet)
        widths, steps, offset, prefix = (4,)*4, None, BIRTH_HEADER.size, b''
        dtype = lambda width: '<f4'
    else:
        raise ValueError('Unsupported birth appearance packet')
    expected = count*(3*widths[0]+widths[1]+3*widths[2]+3*rest*widths[3])
    if (base != old_count or rows != count or coefficients != rest or
            count < 1 or length != expected or length > MAX_BIRTH_BODY_BYTES):
        raise ValueError('Birth dimensions differ from the motion packet')
    body = _decompress(packet, offset, length)
    if zlib.crc32(prefix+body) != checksum:
        raise ValueError('Birth appearance checksum mismatch')
    values, cursor = {}, 0
    for name, shape, width in zip(ATTRIBUTES, _shapes(count,rest).values(), widths):
        array = np.frombuffer(body,dtype=dtype(width),count=int(np.prod(shape)),offset=cursor).reshape(shape)
        if steps is None and not bool(np.isfinite(array).all()):
            raise ValueError('Nonfinite absolute newborn appearance')
        values[name] = array
        cursor += array.size*width
    return BirthAttributes(values, steps)


def _opacity_symbols(packet, frame, count):
    packet = bytes(packet)
    parsed = {}
    if packet[:4] in OPACITY_MAGICS:
        layout, length = unpack_layout(packet,OPACITY_MAGICS,
            max_raw_bytes=OPACITY_HEADER.size+MAX_OPACITY_BODY_BYTES)
        raw = decode_opacity_layout(layout,length,parsed=parsed)
        header, body = raw[:OPACITY_HEADER.size], raw[OPACITY_HEADER.size:]
    else:
        if len(packet) < OPACITY_HEADER.size:
            raise ValueError('Truncated opacity header')
        header = packet[:OPACITY_HEADER.size]
        length = OPACITY_HEADER.unpack(header)[5]
        if length > MAX_OPACITY_BODY_BYTES:
            raise ValueError('Oversized opacity body')
        decoder = zlib.decompressobj()
        try:
            body = decoder.decompress(packet[OPACITY_HEADER.size:],length+1)
        except zlib.error as error:
            raise ValueError('Damaged opacity packet') from error
        if (len(body) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError('Invalid opacity compressed length or trailer')
    magic, index, rows, nonzero, step, length, checksum, prior = OPACITY_HEADER.unpack(header)
    if (magic not in MODES.values() or index != frame or rows != count or
            not 1 <= nonzero <= count or not nonzero*3 <= length <= nonzero*7 or
            len(body) != length or not math.isfinite(step) or step <= 0):
        raise ValueError('Opacity metadata differs from the motion frame')
    prefix = OPACITY_HEADER.pack(magic,index,rows,nonzero,step,length,0,prior)
    if zlib.crc32(prefix+body) != checksum:
        raise ValueError('Opacity checksum mismatch')
    if not parsed:
        ids, symbols, cursor, previous = [], [], 0, -1
        for _ in range(nonzero):
            gap, cursor = _read_varint(body,cursor)
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
    mode = next(name for name,value in MODES.items() if value == magic)
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
    raw = uncompressed_motion_packet(packet,sparse_sections=sections)
    reader = _IntegerReader(raw[:-CHECKSUM_BYTES])
    magic = bytes(reader.take(4))
    if magic not in OFFSET_SPATIAL_MAGICS:
        raise ValueError('GPU streaming requires a spatial offset-pose format')
    inherit_rotation = magic in TRANSLATION_ONLY_MAGICS
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
        gaps = _read_unsigned_values(reader,number).astype(np.int64)
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
        old = _Reader(reader.data)
        old.cursor = start
        t,r = _sparse_read(old,limit,delta_rows=True)
        reader.cursor = old.cursor
        if bool(r.ne(0).any()):
            raise ValueError('GPU streaming requires zero old rotation updates')
        ids = t.ne(0).any(-1).nonzero().flatten().numpy()
        return ids,t.numpy()[ids]
    regional_rows, regional_values = sparse(regions-1)
    gs_rows, gs_values = sparse(count)
    regional = np.zeros((regions,3),dtype=np.int32)
    regional[regional_rows+1] = regional_values
    sources = _read_unsigned_values(reader,births).astype(np.int64)
    if bool((sources >= count).any()):
        raise ValueError('Newborn source outside surviving rows')
    pose_steps = struct.unpack('<2d',reader.take(16))
    if not all(math.isfinite(value) and value > 0 for value in pose_steps):
        raise ValueError('Invalid newborn pose quantization steps')
    width = 3 if inherit_rotation else 7
    packed = _read_unsigned_values(reader,births*width).astype(np.int64)
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
