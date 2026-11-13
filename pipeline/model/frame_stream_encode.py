"""逐帧码流的编码与导出入口。

负责把运动及属性增量编码为数据包、组织整帧导出和记录清单；
格式定义放在 frame_stream_format.py，对应还原逻辑放在 frame_stream_decode.py。
"""

from dataclasses import dataclass
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
from typing import NamedTuple
from zipfile import ZIP_LZMA, ZipFile
import zlib

import numpy as np
import torch

from .rigid_residual_tree import compose_pose, region_pivots, _topological_order
from .spatial_region_gate import (compose_spatial_pose, spatial_owner, spatial_route,
                           spatial_state_route, SPATIAL_ROUTING_FIELDS,
                           validate_spatial_routing, spatial_accumulate)

from .frame_stream_format import (
    ANISOTROPIC_RAW_MAGIC,
    ANISOTROPIC_ZLIB_MAGIC,
    CHECKSUM_BYTES,
    COLOR_MASK_HEADER,
    DELTA_MAGIC,
    DUAL_STEP_DELTA_MAGIC,
    DYNAMIC_MAGIC,
    FEATURE_MAGIC,
    FIELDS,
    HEADER,
    LOGSCALE_HEADER,
    MAX_BODY,
    MAX_LOGSCALE_BODY_BYTES,
    MAX_RAW_PACKET_BYTES,
    MODES,
    MOTION_MAGICS,
    OPACITY_HEADER,
    OPACITY_MAGICS,
    PLANAR_MAGIC,
    RAW_MAGIC,
    RAW_MAGICS,
    REGIONAL_ATTRIBUTE_HEADER,
    REGIONAL_ATTRIBUTE_MAGIC,
    RELATIVE_SPATIAL_MAGIC,
    RELATIVE_TRANSLATION_ONLY_SPATIAL_MAGIC,
    SPATIAL_MAGIC,
    SPATIAL_MAGICS,
    TRANSLATION_ONLY_SPATIAL_MAGIC,
    WRAPPERS,
    ZLIB_MAGIC,
    _IntegerReader,
    _check_motion_tail,
    _check_reference,
    _cpu,
    _integer_varint,
    _opacity_header,
    _opacity_varint,
    _reference_digest,
    _reference_hash,
    _regional_attribute_zeros,
    _regional_digest,
    _rows,
    _sparse_zigzag_byteplane,
    _spatial_prefix,
    _step,
    _uvar,
    _validate_state,
    _verify_raw_checksum,
    compose_decoded_survivors,
    opacity_reference_hash,
    validate_sparse_color_selection,
)

from .frame_stream_decode import decode_opacity_delta, decode_logscale_delta


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


def pack_layout(layout, raw_length, magics):
    """Choose actual shortest transformed raw/zlib/XZ envelope."""
    size = _integer_varint(raw_length) + _integer_varint(len(layout))
    candidates = [magics[0] + size + layout,
                  magics[1] + size + zlib.compress(layout, 6),
                  magics[2] + size + lzma.compress(layout, format=lzma.FORMAT_XZ, preset=9)]
    return min(candidates, key=len)


def _write_values(values, components):
    values = np.asarray(values, dtype='<i2').reshape(-1, components).T.ravel().astype(np.int64)
    return encode_unsigned_values(np.where(values >= 0, 2 * values, -2 * values - 1))


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


def encode_motion_layout(raw):
    reader = _IntegerReader(raw)
    regions, count, births = _spatial_prefix(reader)
    prefix = raw[:reader.cursor]
    region = _encode_sparse(reader, regions - 1)
    point = _encode_sparse(reader, count)
    tail = bytes(reader.take(reader.remaining))
    _check_motion_tail(tail, births, count, 12 if raw[:4] in (b'SPG3', b'SPG4') else 3 if raw[:4] in (b'SPG7',b'SPG8') else 7 if raw[:4] in (b'SPG5',b'SPG6') else 28, offset=raw[:4] in (b'SPG5',b'SPG6',b'SPG7',b'SPG8'))
    return _integer_varint(len(prefix)) + prefix + region + point + tail


def encode_opacity_layout(raw):
    reader = _IntegerReader(raw)
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


def _sparse_body(arrays):
    # Three independent, sorted row-gap lists and their full signed-int16
    # attribute blocks. Gaps are stored separately for vectorized decoding.
    chunks = []
    for array in arrays:
        matrix = array.reshape(len(array), -1)
        rows = np.flatnonzero(np.any(matrix != 0, axis=1))
        gaps = np.diff(np.r_[-1, rows]).astype(np.uint32)
        chunks.append(struct.pack('<I', len(rows)))
        chunks.append(encode_unsigned_values(gaps))
        chunks.append(matrix[rows].astype('<i2', copy=False).tobytes())
    return b''.join(chunks)


def _encode_body(magic, body, frame_index, count, rest, steps, prior):
    prefix = HEADER.pack(magic, frame_index, count, rest, *steps, len(body), prior, 0)
    checksum = zlib.crc32(prefix + body)
    return HEADER.pack(magic, frame_index, count, rest, *steps, len(body), prior, checksum) + zlib.compress(body, 6)


def encode_full_attribute_delta(reference, symbols, *, frame_index, steps, sparse_only=False):
    reference = {k: v.detach().cpu().contiguous() for k, v in reference.items()}
    count, rest = _check_reference(reference)
    prior = _reference_hash(reference)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 1 <= frame_index <= 0xffffffff:
        raise ValueError('Invalid full attribute frame')
    if len(steps) != 3 or any(not math.isfinite(x) or x <= 0 for x in steps):
        raise ValueError('Invalid full attribute steps')
    arrays = []
    for name in FIELDS:
        value = torch.as_tensor(symbols[name]).detach().cpu()
        if (value.shape != reference[name].shape or not bool(torch.isfinite(value).all()) or
                not torch.equal(value, value.round()) or bool(value.abs().gt(32767).any())):
            raise ValueError(f'Invalid {name} integer symbols')
        array = value.to(torch.int16).contiguous().numpy().astype('<i2', copy=False)
        arrays.append(array)
    if sum(array.nbytes for array in arrays) > MAX_BODY:
        raise ValueError('Full attribute packet too large')
    if not any(np.any(array) for array in arrays):
        return None
    sparse_body = _sparse_body(arrays)
    if len(sparse_body) > MAX_BODY:
        raise ValueError('Sparse full attribute packet too large')
    planar_body = _sparse_zigzag_byteplane(sparse_body, count, rest)
    return _encode_body(PLANAR_MAGIC, planar_body, frame_index, count, rest, steps, prior)


def encode_opacity_delta(reference, symbols, *, frame_index, step, mode='logit'):
    """Write exact quantized opacity deltas using the shortest lossless layout."""
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    step = _step(step)
    values = torch.as_tensor(symbols).detach().cpu()
    if (values.shape != reference.shape or not bool(torch.isfinite(values).all()) or
            not torch.equal(values, values.round()) or bool(values.abs().gt(32767).any())):
        raise ValueError('Opacity symbols must be aligned signed-int16 integers')
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or not 1 <= frame_index <= 0xffffffff:
        raise ValueError('Opacity frame must be a positive uint32')
    prior = opacity_reference_hash(reference)
    if mode not in ('logit','alpha'):
        raise ValueError('Unknown opacity update mode')
    magic = MODES['alpha_canonical' if mode == 'alpha' else mode]
    rows = values[:, 0].ne(0).nonzero(as_tuple=True)[0]
    if not len(rows):
        return None
    body = bytearray()
    previous = -1
    for row in rows.tolist():
        body.extend(_opacity_varint(row - previous))
        body.extend(struct.pack('<h', int(values[row, 0])))
        previous = row
    prefix = OPACITY_HEADER.pack(magic, frame_index, len(reference), len(rows), step,
                         len(body), 0, prior)
    checksum = zlib.crc32(prefix + body)
    header = OPACITY_HEADER.pack(magic, frame_index, len(reference), len(rows), step,
                         len(body), checksum, prior)
    packet = header + zlib.compress(body, 6)
    layout = encode_opacity_layout(header + body)
    packet = min((packet, pack_layout(layout, OPACITY_HEADER.size + len(body), OPACITY_MAGICS)), key=len)
    # The existing decoder is also the byte-exact definition of materialization.
    decode_opacity_delta(packet, reference, frame_index=frame_index)
    return packet


def encode_color_point_mask(mask,frame):
    mask=np.asarray(mask)
    if mask.ndim!=1 or mask.dtype!=np.bool_ or isinstance(frame,bool) or not isinstance(frame,int) or frame<1:
        raise ValueError('Invalid point mask')
    rows=np.flatnonzero(mask).astype('<u4')
    if not len(rows):return None
    return COLOR_MASK_HEADER.pack(b'CPM1',frame,len(mask),len(rows))+zlib.compress(rows.tobytes(),9)


def encode_logscale_delta(reference, symbols, *, allowed_rows, frame_index, step):
    """Encode signed-int8 residuals on motion-supported old rows.

    A one-dimensional symbol tensor keeps the original isotropic stream
    bit-compatible.  A ``[N, 3]`` tensor stores one quantized residual per
    log-scale axis while retaining the same row support and reference binding.
    """
    reference = torch.as_tensor(reference).detach().cpu().contiguous()
    symbols = torch.as_tensor(symbols).detach().cpu().contiguous()
    if (reference.dtype != torch.float32 or reference.ndim != 2 or
            reference.shape[1:] != (3,) or not bool(torch.isfinite(reference).all())):
        raise ValueError('Log-scale reference must be finite float32 [N, 3]')
    if symbols.shape == (len(reference),):
        channels = 1
    elif symbols.shape == (len(reference), 3):
        channels = 3
    else:
        raise ValueError('Log-scale symbols must be aligned isotropic [N] or anisotropic [N, 3] integers')
    if (not bool(torch.isfinite(symbols).all()) or
            not torch.equal(symbols, symbols.round()) or bool(symbols.abs().gt(127).any())):
        raise ValueError('Log-scale symbols must be aligned signed-int8 integers')
    if (isinstance(frame_index, bool) or not isinstance(frame_index, int) or
            not 1 <= frame_index <= 0xffffffff):
        raise ValueError('Log-scale frame must be a positive uint32')
    if (isinstance(step, bool) or not isinstance(step, (int, float)) or
            not math.isfinite(step) or step <= 0):
        raise ValueError('Log-scale quantization step must be positive')
    allowed_rows = _rows(allowed_rows, len(reference), name='Allowed log-scale rows')
    active = (symbols.ne(0) if channels == 1 else symbols.ne(0).any(-1)).nonzero(as_tuple=True)[0]
    if len(active) and not bool(torch.isin(active, allowed_rows).all()):
        raise ValueError('Log-scale symbols extend outside transmitted motion support')
    if not len(active):
        return None
    body = bytearray()
    previous = -1
    for row in active.tolist():
        body.extend(_uvar(row - previous))
        values = symbols[row].reshape(-1).tolist()
        body.extend(struct.pack(f'<{channels}b', *(int(value) for value in values)))
        previous = row
    if len(body) > MAX_LOGSCALE_BODY_BYTES:
        raise ValueError('Log-scale packet body is too large')
    prior = _reference_digest(reference, allowed_rows)
    prefix = struct.pack('<IIId32sI', frame_index, len(reference), len(active),
                         float(step), prior, len(body))
    raw_magic = RAW_MAGIC if channels == 1 else ANISOTROPIC_RAW_MAGIC
    zlib_magic = ZLIB_MAGIC if channels == 1 else ANISOTROPIC_ZLIB_MAGIC
    # Keep legacy isotropic packets bit-compatible.  Bind the new component
    # count explicitly so changing only the four-byte magic cannot reinterpret
    # an isotropic body as an anisotropic one (or vice versa).
    checksum = zlib.crc32(prefix + (b'\x03' if channels == 3 else b'') + body)
    raw = LOGSCALE_HEADER.pack(raw_magic, frame_index, len(reference), len(active),
                      float(step), prior, len(body), checksum) + body
    compressed = LOGSCALE_HEADER.pack(zlib_magic, frame_index, len(reference), len(active),
                             float(step), prior, len(body), checksum) + zlib.compress(body, 9)
    packet = min((raw, compressed), key=len)
    decoded = decode_logscale_delta(packet, reference, allowed_rows=allowed_rows,
                                    frame_index=frame_index)
    expected = reference.clone()
    delta = symbols[active].to(reference.dtype) * float(step)
    expected[active] = reference[active] + (delta[:, None] if channels == 1 else delta)
    if not torch.equal(decoded.logscale, expected):
        raise ValueError('Log-scale packet differs from canonical quantized values')
    return packet


def encode_regional_scalar_packet(magic, reference, symbols, owner, node_count, *, frame_index, step):
    """Encode a regional DC update using the shared scalar stream."""
    reference = reference.detach().cpu()
    owner = owner.cpu()
    zeros = reference.new_zeros(node_count * 3, 1)
    body = encode_opacity_delta(zeros, symbols.cpu().reshape(-1, 1),
                                frame_index=frame_index, step=step, mode='logit')
    if body is None:
        return None
    return magic + _regional_digest(reference, owner, node_count) + body


def encode_feature_dc_packet(scalar_packet):
    return FEATURE_MAGIC + scalar_packet


def encode_regional_full(symbols, steps, nodes, frame):
    """Pack the region-shared appearance and opacity updates for one frame."""
    appearance = encode_full_attribute_delta(
        _regional_attribute_zeros(nodes, 3), {name: symbols[name] for name in FIELDS},
        frame_index=frame, steps=tuple(steps[name] for name in FIELDS))
    opacity = encode_opacity_delta(torch.zeros(nodes, 1), symbols['opacity'],
                                   frame_index=frame, step=steps['opacity'], mode='logit')
    if not appearance and not opacity:
        return None
    return (REGIONAL_ATTRIBUTE_HEADER.pack(REGIONAL_ATTRIBUTE_MAGIC, frame,
                                           len(appearance or b''), len(opacity or b'')) +
            (appearance or b'') + (opacity or b''))

class EncodedFrame(NamedTuple):
    """Materialized frame and packet metadata for the training report."""
    stream_dir: Path
    regional_full_info: object
    regional_full: object
    raw: dict
    sources: torch.Tensor
    motion_path: Path
    motion_info: object
    activity: dict
    rate_terms: object
    logscale_info: object
    opacity_info: object
    feature_info: object
    full_info: object
    point_mask_info: object

    @property
    def payload_bytes(self):
        packets = (self.regional_full_info, self.opacity_info, self.feature_info,
                   self.logscale_info, self.full_info,
                   self.point_mask_info)
        return self.motion_path.stat().st_size + sum(
            packet['archive_bytes'] for packet in packets if packet is not None)


def encode_frame(self, old_raw, full_reference, logscale_reference, logscale_update, full):
    """Export all packets for one trained frame in decoder order.

    The module owns the trained values; this routine owns packet organization.
    Returns the materialized state and packet metadata used for the manifest.
    """
    from .frame_stream_decode import decode_regional_full, decode_color_point_mask

    stream_dir = self.output_dir / 'frame_storage' / 'rigid_tree'
    stream_dir.mkdir(parents=True, exist_ok=True)
    regional_full_info = None
    regional_full = getattr(self, 'regional_full_attributes', None)
    point_full_raw = {name: old_raw[name] for name in (*FIELDS, 'opacity')}
    if regional_full is not None:
        regional_full_info = regional_full.export(stream_dir / f'regfull_{self.current_frame:03d}.bin', self.current_frame)
        if regional_full_info is not None:
            regional_values = decode_regional_full(Path(regional_full_info['path']).read_bytes(), old_raw,
                self.motion.owner.detach().cpu(), len(self.motion.parents), self.current_frame)
            old_raw.update({name: value.to(old_raw[name]) for name,value in regional_values.items()})
    raw = {name: value.detach().clone() for name, value in old_raw.items()}
    sources = torch.empty(0, dtype=torch.long, device=self.device)
    motion_path = stream_dir / f'motion_{self.current_frame:03d}.bin'
    region_xyz, region_rot, gs_xyz, gs_rot = self.motion._values(hard=True)
    activity = dict(
        moving_regions=int((region_xyz[1:].ne(0).any(-1) |
                            region_rot[1:].ne(0).any(-1)).sum()),
        old_gs_with_local_residual=int((gs_xyz.ne(0).any(-1) |
                                        gs_rot.ne(0).any(-1)).sum()),
        old_gs=len(gs_xyz), topology_edits=self.birth_control.get('tree_edit_count', 0))
    self.motion.register_materialized_layout(raw['xyz'], raw['rotation'],
                                             birth_source_rows=sources)
    rate_terms = self.motion.loss_terms(self.motion_steps, diagnostics=True)
    motion_info = self.motion.export_motion(motion_path,
        inherit_birth_rotation=(not self.config.birth.rotation_updates and
                                not self.config.rigid_tree.rotation_updates),
        birth_pose_steps=((self.config.birth.new_xyz_step_extent*self.datamodule.cameras_extent,self.config.birth.new_rotation_step)
            if self.config.birth.radiance_mode=='adaptive_residual_birth' else None))
    logscale_info = None
    if logscale_update is not None and logscale_update.ready:
        decoded_motion = self.motion.codec_previous_state
        if decoded_motion.gs_translation_rows is None:
            raise ValueError('Exported motion lost point-translation support')
        logscale_info = logscale_update.export(
            stream_dir / f'logscale_{self.current_frame:03d}.bin',
            logscale_reference, frame_index=self.current_frame,
            allowed_rows=decoded_motion.gs_translation_rows)
        expected_logscale = logscale_update.canonical_logscale(
            logscale_reference).to(raw['logscale'])
        if not torch.equal(raw['logscale'][:len(expected_logscale)], expected_logscale):
            raise ValueError('Materialized old log-scale differs from its sparse packet')
    opacity_info = None
    if self.opacity_update is not None:
        opacity_info = self.opacity_update.export(
            stream_dir / f'opacity_{self.current_frame:03d}.bin', self.current_frame)
        expected_logits = self.opacity_update.canonical_logits().to(raw['opacity'])
        actual_point_opacity = point_full_raw['opacity'] if regional_full is not None else raw['opacity']
        if not torch.equal(actual_point_opacity[:len(self.motion.owner)], expected_logits):
            raise ValueError('Materialized old opacity differs from its sparse packet')
    feature_info = None
    if getattr(self, 'feature_dc_update', None) is not None:
        feature_info = self.feature_dc_update.export(
            stream_dir / f'shdc_{self.current_frame:03d}.bin', self.current_frame)
        expected_features = self.feature_dc_update.canonical_features().to(raw['sh_dc'])
        if full_reference is None and not torch.equal(raw['sh_dc'][:len(expected_features)], expected_features):
            raise ValueError('Materialized old DC differs from its shared color packet')
    full_info = None
    if full_reference is not None:
        full_info = full.export(stream_dir / f'fullattr_{self.current_frame:03d}.bin',
                                full_reference, self.current_frame, sparse_only=False)
        expected_full = full.canonical(full_reference)
        for name in FIELDS:
            actual_point = point_full_raw[name] if regional_full is not None else raw[name]
            if not torch.equal(actual_point[:len(expected_full[name])].cpu(), expected_full[name]):
                raise ValueError(f'Materialized old {name} differs from its full attribute packet')
    point_mask_info = None
    return EncodedFrame(stream_dir, regional_full_info, regional_full, raw, sources,
                        motion_path, motion_info, activity, rate_terms, logscale_info,
                        opacity_info, feature_info, full_info, point_mask_info)


def add_frame_packet_manifest(manifest, encoded, self, full_reference, logscale_update):
    """Record the frame's packet names and decoding order in its manifest."""
    opacity_info, feature_info = encoded.opacity_info, encoded.feature_info
    logscale_info = encoded.logscale_info
    point_mask_info, full_info = encoded.point_mask_info, encoded.full_info
    if self.opacity_update is not None:
        manifest.update(opacity_update=Path(opacity_info['path']).name if opacity_info else None,
                        opacity_update_timing='before_births')
    if getattr(self, 'feature_dc_update', None) is not None:
        manifest.update(feature_dc_update=Path(feature_info['path']).name if feature_info else None,
                        feature_dc_update_timing='before_births')
    if logscale_update is not None:
        manifest.update(logscale_update=(Path(logscale_info['path']).name
                                         if logscale_info else None),
                        logscale_update_timing='before_births')
    if full_reference is not None:
        manifest.update(full_update=True,
            full_attribute_update=Path(full_info['path']).name if full_info else None,
            full_attribute_update_timing='after_regional_dc_before_births')
