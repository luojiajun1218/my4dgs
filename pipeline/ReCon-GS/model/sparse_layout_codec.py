"""Lossless layout packets: grouped fine memberships and frame-to-frame reuse.

Training keeps its original GS order. Only disk memberships are grouped by
fine region; parent links recover the two coarser ownership arrays. Reuse
packets refer to an earlier packet in the same directory, never a checkpoint.
"""
import hashlib
from pathlib import Path

import numpy as np


GROUPED = 'grouped_fine_v1'
REUSE = 'reuse_fine_v1'


def compact_unsigned(values):
    values = np.asarray(values)
    if values.dtype.kind not in 'iu' or (values.size and values.min() < 0):
        raise ValueError('Layout identifiers must be nonnegative integers')
    maximum = int(values.max()) if values.size else 0
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return values.astype(dtype)
    raise ValueError('Layout identifier overflow')


def encode_gaps(values):
    """Canonical unsigned LEB128, with one to five bytes per uint32 value."""
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError('Row gaps must be an integer vector')
    if values.size and (values.min() < 0 or values.max() > np.iinfo(np.uint32).max):
        raise ValueError('Row gap is outside uint32 range')
    values = values.astype(np.uint64)
    lengths = np.ones(len(values), dtype=np.int64)
    for shift in (7, 14, 21, 28):
        lengths += values >= (1 << shift)
    starts = np.cumsum(lengths) - lengths
    data = np.empty(int(lengths.sum()), dtype=np.uint8)
    for offset in range(5):
        selected = lengths > offset
        data[starts[selected] + offset] = (
            ((values[selected] >> (7 * offset)) & 127).astype(np.uint8)
            | np.where(lengths[selected] > offset + 1, 128, 0).astype(np.uint8))
    return data


def decode_gaps(data):
    data = np.asarray(data)
    if data.ndim != 1 or data.dtype != np.uint8:
        raise ValueError('Packed row gaps must be a uint8 vector')
    if not len(data):
        return np.empty(0, dtype=np.int64)
    ends = np.flatnonzero(data < 128)
    if not len(ends) or ends[-1] != len(data) - 1:
        raise ValueError('Truncated packed row gaps')
    starts = np.concatenate(([0], ends[:-1] + 1))
    lengths = ends - starts + 1
    if (lengths > 5).any() or ((lengths > 1) & (data[ends] == 0)).any():
        raise ValueError('Invalid packed row gaps')
    shifts = 7 * (np.arange(len(data)) - np.repeat(starts, lengths))
    values = np.add.reduceat((data.astype(np.uint64) & 127) << shifts.astype(np.uint64), starts)
    if (values > np.iinfo(np.uint32).max).any():
        raise ValueError('Packed row gap overflow')
    return values.astype(np.int64)


def numpy_layout(layout):
    def array(value, dtype):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        # Export records must not follow later in-place changes to CPU tensors.
        return np.array(value, dtype=dtype, copy=True)
    result = dict(index=array(layout['index'], np.int64),
                  counts=array(layout['counts'], np.int64),
                  parents=[array(v, np.int64) for v in layout['parents']],
                  pivots=[array(v, np.float32) for v in layout['pivots']])
    validate_layout(result)
    return result


def validate_layout(layout):
    counts, index = layout['counts'], layout['index']
    depth = len(counts)
    if (counts.shape != (depth,) or depth not in (1, 3) or (counts < 1).any()
            or index.ndim != 2 or index.shape[0] != depth
            or len(layout['parents']) != depth - 1 or len(layout['pivots']) != depth):
        raise ValueError('Invalid motion layout dimensions')
    for level in range(depth):
        if (index[level] < 0).any() or (index[level] >= counts[level]).any():
            raise ValueError('Layout owner outside its region range')
        pivots = layout['pivots'][level]
        if pivots.shape != (counts[level], 3) or not np.isfinite(pivots).all():
            raise ValueError('Invalid layout pivots')
    for level in range(depth - 1):
        parent = layout['parents'][level]
        if (parent.shape != (counts[level + 1],) or (parent < 0).any()
                or (parent >= counts[level]).any()):
            raise ValueError('Invalid layout parents')
        if not np.array_equal(parent[index[level + 1]], index[level]):
            raise ValueError('Fine ownership does not determine its ancestors')


def layout_digest(layout):
    digest = hashlib.sha256()
    values = [(layout['counts'], '<u8'), (layout['index'][-1], '<u8')]
    values.extend((value, '<u8') for value in layout['parents'])
    values.extend((value, '<f4') for value in layout['pivots'])
    for value, dtype in values:
        value = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _same_structure(left, right):
    return (np.array_equal(left['counts'], right['counts'])
            and all(np.array_equal(a, b) for a, b in zip(left['parents'], right['parents'])))


def encode_layout(layout, path, previous=None, previous_layout=None, survivor_rows=None):
    """Reuse only when the predecessor and exact survivor identities are known."""
    if len(layout['counts']) != 3:
        raise ValueError('Legacy grouped layout codec requires three levels')
    path = Path(path).resolve()
    header = dict(layout_row_count=np.array(layout['index'].shape[1], dtype=np.int64),
                  layout_digest=np.array(layout_digest(layout)))
    if previous is not None and previous_layout is not None and survivor_rows is not None:
        source = previous['path']
        old = previous['layout']
        old_fine, materialized_fine = old['index'][2], previous_layout['index'][2]
        rows = np.asarray(survivor_rows, dtype=np.int64)
        reusable = (source.parent == path.parent and source != path and source.is_file()
                    and _same_structure(old, layout) and _same_structure(old, previous_layout)
                    and len(materialized_fine) >= len(old_fine)
                    and np.array_equal(materialized_fine[:len(old_fine)], old_fine)
                    and rows.shape == (layout['index'].shape[1],)
                    and (rows >= 0).all() and (rows < len(materialized_fine)).all()
                    and (np.diff(rows) > 0).all())
        if reusable and np.array_equal(materialized_fine[rows], layout['index'][2]):
            if file_digest(source) != previous['sha256']:
                raise ValueError('Previous layout packet changed after export')
            retained = np.zeros(len(materialized_fine), dtype=bool)
            retained[rows] = True
            deleted = np.flatnonzero(~retained)
            header.update(layout_encoding=np.array(REUSE),
                          layout_source=np.array(source.name),
                          layout_source_sha256=np.array(previous['sha256']),
                          layout_appended_fine=compact_unsigned(materialized_fine[len(old_fine):]),
                          layout_deleted_gaps=encode_gaps(np.diff(np.concatenate(([-1], deleted)))))
            # Region centres are recomputed even when memberships are reused.
            # Bitwise deltas preserve float32 exactly, without changing pivots
            # or assuming they stayed fixed as the scene moved.
            for level in range(3):
                left = np.ascontiguousarray(old['pivots'][level], dtype='<f4').view('<u4')
                right = np.ascontiguousarray(layout['pivots'][level], dtype='<f4').view('<u4')
                difference = np.bitwise_xor(left, right)
                if difference.any():
                    header[f'pivot_xor_l{level}'] = difference
            return header
    # Sorted member rows within each region become positive gaps. Decode puts
    # them back in the original GS order; no rendering/model permutation occurs.
    fine = layout['index'][2]
    sizes = np.bincount(fine, minlength=int(layout['counts'][2]))
    order = np.argsort(fine, kind='stable')
    gaps = np.diff(np.concatenate(([-1], order)))
    starts = (np.cumsum(sizes) - sizes)[sizes > 0]
    gaps[starts] = order[starts] + 1
    header.update(layout_encoding=np.array(GROUPED), counts=compact_unsigned(layout['counts']),
                  fine_group_sizes=compact_unsigned(sizes), fine_group_gaps=encode_gaps(gaps))
    for level, parent in enumerate(layout['parents']):
        header[f'parents_l{level}'] = compact_unsigned(parent)
    for level, pivot in enumerate(layout['pivots']):
        header[f'pivots_l{level}'] = pivot
    return header


def _restore_index(fine, counts, parents, pivots):
    if (fine < 0).any() or (fine >= counts[2]).any():
        raise ValueError('Fine owner outside its region range')
    # Validate links before indexing them, including unoccupied regions.
    for level in range(2):
        if (parents[level].shape != (counts[level + 1],)
                or (parents[level] < 0).any() or (parents[level] >= counts[level]).any()):
            raise ValueError('Invalid layout parents')
    middle = parents[1][fine]
    result = dict(index=np.stack((parents[0][middle], middle, fine)),
                  counts=counts, parents=parents, pivots=pivots)
    validate_layout(result)
    return result


def _read_layout_fields(packet):
    names = ('index', 'counts', 'fine_group_sizes', 'fine_group_gaps')
    return {name: packet[name] for name in packet.files
            if name in names or name.startswith(('layout_', 'parents_l', 'pivots_l', 'pivot_xor_l'))}


def decode_layout(path, packet):
    """Resolve and verify a layout dependency chain without loading motion payloads."""
    current = Path(path).resolve()
    fields = _read_layout_fields(packet)
    pending, visited = [], {current}
    while str(fields.get('layout_encoding', GROUPED)) == REUSE:
        pending.append(fields)
        name = str(fields['layout_source'].item())
        if not name or name in {'.', '..'} or Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Layout dependency must be a filename in the same directory')
        source = current.parent / name
        if source in visited:
            raise ValueError('Cyclic layout dependency')
        visited.add(source)
        if file_digest(source) != str(fields['layout_source_sha256'].item()):
            raise ValueError('Layout dependency hash mismatch')
        with np.load(source, allow_pickle=False) as previous:
            fields = _read_layout_fields(previous)
        current = source
    counts = fields['counts'].astype(np.int64)
    if counts.shape != (3,) or (counts < 1).any():
        raise ValueError('Invalid layout counts')
    parents = [fields[f'parents_l{level}'].astype(np.int64) for level in range(2)]
    pivots = [fields[f'pivots_l{level}'].astype(np.float32) for level in range(3)]
    if 'index' in fields:  # Original per-row int32 packets remain readable.
        layout = dict(index=fields['index'].astype(np.int64), counts=counts,
                      parents=parents, pivots=pivots)
        validate_layout(layout)
    else:
        if str(fields['layout_encoding'].item()) != GROUPED:
            raise ValueError('Unsupported layout encoding')
        sizes = fields['fine_group_sizes'].astype(np.int64)
        gaps = decode_gaps(fields['fine_group_gaps'])
        rows = int(fields['layout_row_count'].item())
        if (rows < 0 or sizes.shape != (counts[2],) or (sizes < 0).any()
                or int(sizes.sum()) != rows or len(gaps) != rows or (gaps < 1).any()):
            raise ValueError('Invalid grouped layout membership sizes')
        starts = (np.cumsum(sizes) - sizes)[sizes > 0]
        sums = np.cumsum(gaps)
        offsets = sums[starts] - gaps[starts]
        members = sums - np.repeat(offsets, sizes[sizes > 0]) - 1
        if not np.array_equal(np.sort(members), np.arange(rows)):
            raise ValueError('Grouped layout must contain each GS exactly once')
        fine = np.empty(rows, dtype=np.int64)
        fine[members] = np.repeat(np.arange(counts[2]), sizes)
        layout = _restore_index(fine, counts, parents, pivots)
    if 'layout_digest' in fields and layout_digest(layout) != str(fields['layout_digest'].item()):
        raise ValueError('Layout digest mismatch')
    for fields in reversed(pending):
        appended = fields['layout_appended_fine'].astype(np.int64)
        if appended.ndim != 1:
            raise ValueError('Invalid appended fine owners')
        fine = np.concatenate((layout['index'][2], appended))
        gaps = decode_gaps(fields['layout_deleted_gaps'])
        deleted = np.cumsum(gaps) - 1
        if (gaps < 1).any() or (deleted < 0).any() or (deleted >= len(fine)).any():
            raise ValueError('Invalid deleted layout rows')
        fine = np.delete(fine, deleted)
        if len(fine) != int(fields['layout_row_count'].item()):
            raise ValueError('Layout survivor count mismatch')
        pivots = []
        for level, pivot in enumerate(layout['pivots']):
            name = f'pivot_xor_l{level}'
            if name in fields:
                difference = fields[name]
                if difference.shape != pivot.shape or difference.dtype != np.dtype('<u4'):
                    raise ValueError('Invalid pivot bitwise delta')
                bits = np.ascontiguousarray(pivot, dtype='<f4').view('<u4')
                pivot = np.bitwise_xor(bits, difference).view('<f4')
            pivots.append(pivot)
        layout = _restore_index(fine, layout['counts'], layout['parents'], pivots)
        if layout_digest(layout) != str(fields['layout_digest'].item()):
            raise ValueError('Reused layout digest mismatch')
    return layout
