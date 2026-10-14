"""Tree-aware persistent-region diagnostics with exact quantized GS counts."""

import numpy as np

from .online_motion_diagnostics import _array, _mask, summarize_flat_regions


def _integer_rows(value, rows, name):
    result = _array(value, dtype=None)
    if result.shape != (rows, 3) or result.dtype.kind not in 'iu':
        raise ValueError(f'{name} must contain integer symbols [{rows},3]')
    return result


def _native(value):
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return _native(value.item())
    if hasattr(value, 'detach'):
        return _native(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    raise TypeError(f'Unsupported/nonfinite diagnostic value: {type(value).__name__}')


def summarize_persistent_regions(source_xyz, target_xyz, source_rotation,
                                 target_rotation, owner, parents,
                                 world_quaternion, world_translation, *,
                                 region_ids=None, surviving_count=None,
                                 gs_translation_i16=None, gs_rotation_i16=None,
                                 region_translation_i16=None, region_rotation_i16=None,
                                 observed_mask=None, static_mask=None, noise_mask=None,
                                 unknown_mask=None, previous_owner=None, candidate_owner=None,
                                 extent=1., edit_metrics=None, packet_metrics=None):
    """Summarize direct membership and inherited motion without fake birth history.

    owner/parents are codec slot indices. region_ids are stable logical IDs;
    both identities are exposed, so compaction never appears as an identity
    change. Input poses include current newborn rows; those rows are always
    excluded from transition residual statistics. Exact GS nonzero counts use
    hard integer symbols for surviving rows only and ignore floating noise.
    Internal nodes except root must own no GS. World affine transforms use
    R*x+t; local integer shared blocks and inherited paths are distinguished.
    """
    owners = _array(owner, np.int64)
    parent = _array(parents, np.int64)
    count, nodes = len(owners), len(parent)
    if (owners.shape != (count,) or parent.shape != (nodes,) or nodes == 0 or
            parent[0] != -1 or np.any(parent[1:] < 0) or np.any(parent[1:] >= nodes) or
            np.any(owners < 0) or np.any(owners >= nodes)):
        raise ValueError('Persistent diagnostic layout must use valid rooted codec slots')
    logical = np.arange(nodes, dtype=np.int64) if region_ids is None else _array(region_ids, np.int64)
    if (logical.shape != (nodes,) or logical[0] != 0 or np.any(logical < 0) or
            len(np.unique(logical)) != nodes):
        raise ValueError('Persistent logical IDs must be unique with identity root 0')
    depth = np.zeros(nodes, dtype=np.int64)
    paths = []
    for slot in range(nodes):
        cursor, path, visited = slot, [], set()
        while cursor != -1:
            if cursor in visited:
                raise ValueError('Persistent diagnostic tree contains a cycle')
            visited.add(cursor)
            path.append(cursor)
            cursor = int(parent[cursor])
        paths.append(path)
        depth[slot] = len(path) - 1
    children = np.bincount(parent[1:], minlength=nodes)
    direct = np.bincount(owners, minlength=nodes)
    if np.any((children[1:] > 0) & (direct[1:] > 0)):
        raise ValueError('Internal persistent nodes cannot directly own GS')
    has_gs_symbols = gs_translation_i16 is not None or gs_rotation_i16 is not None
    if has_gs_symbols and (gs_translation_i16 is None or gs_rotation_i16 is None):
        raise ValueError('Supply both independent translation and rotation symbols')
    if surviving_count is None:
        surviving_count = len(gs_translation_i16) if has_gs_symbols else count
    if (isinstance(surviving_count, bool) or int(surviving_count) != surviving_count or
            not 0 <= surviving_count <= count):
        raise ValueError('Invalid surviving GS count')
    survivors = int(surviving_count)
    observed = _mask(observed_mask, count, default=True).copy()
    observed[survivors:] = False
    report = summarize_flat_regions(
        source_xyz, target_xyz, source_rotation, target_rotation, owners,
        world_quaternion, world_translation, observed_mask=observed,
        static_mask=static_mask, noise_mask=noise_mask, unknown_mask=unknown_mask,
        previous_owner=previous_owner, candidate_owner=candidate_owner, extent=extent)
    gt_nonzero = gr_nonzero = residual_nonzero = None
    if has_gs_symbols:
        gt = _integer_rows(gs_translation_i16, survivors, 'GS translation')
        gr = _integer_rows(gs_rotation_i16, survivors, 'GS rotation')
        gt_nonzero, gr_nonzero = np.any(gt != 0, axis=1), np.any(gr != 0, axis=1)
        residual_nonzero = gt_nonzero | gr_nonzero
    local_active = None
    if region_translation_i16 is not None or region_rotation_i16 is not None:
        if region_translation_i16 is None or region_rotation_i16 is None:
            raise ValueError('Supply both local shared symbol arrays')
        rt = _integer_rows(region_translation_i16, nodes, 'Regional translation')
        rr = _integer_rows(region_rotation_i16, nodes, 'Regional rotation')
        local_active = np.any(rt != 0, axis=1) | np.any(rr != 0, axis=1)
        if local_active[0]:
            raise ValueError('Identity root cannot contain shared motion symbols')
    subtree = direct.copy()
    for slot in np.argsort(-depth):
        if slot != 0:
            subtree[parent[slot]] += subtree[slot]
    for slot, row in enumerate(report['regions']):
        members = owners[:survivors] == slot
        survivor_members = int(members.sum())
        row.update(
            region_slot=slot, logical_region_id=int(logical[slot]),
            parent_region_slot=int(parent[slot]),
            parent_id=-1 if slot == 0 else int(logical[parent[slot]]),
            depth=int(depth[slot]), is_internal=bool(children[slot]),
            child_count=int(children[slot]), subtree_gs_count=int(subtree[slot]),
            surviving_gs_count=survivor_members,
            newborn_gs_count=int(direct[slot]) - survivor_members,
            encoded_local_shared_nonzero=(None if local_active is None else bool(local_active[slot])),
            encoded_shared_path_nonzero=(None if local_active is None else
                                         bool(local_active[paths[slot]].any())))
        row['independent_quantized'] = dict(
            available=has_gs_symbols, surviving_gs=survivor_members,
            translation_nonzero=None if not has_gs_symbols else int((members & gt_nonzero).sum()),
            rotation_nonzero=None if not has_gs_symbols else int((members & gr_nonzero).sum()),
            residual_nonzero=None if not has_gs_symbols else int((members & residual_nonzero).sum()),
            residual_zero=None if not has_gs_symbols else int((members & ~residual_nonzero).sum()),
            residual_zero_fraction=(None if not has_gs_symbols or not survivor_members else
                                    float((members & ~residual_nonzero).sum() / survivor_members)))
    report.update(
        region_ids=logical.tolist(), surviving_gs=survivors,
        newborn_gs=count - survivors, internal_region_count=int(children[1:].astype(bool).sum()),
        max_depth=int(depth.max(initial=0)),
        residual_scope='observed surviving GS only; integer support counts all surviving GS',
        birth_residual_semantics='newborn GS have no prior-frame residual (N/A)',
        edits=_native(edit_metrics) if edit_metrics is not None else None,
        packet_metrics=_native(packet_metrics) if packet_metrics is not None else None)
    return report
