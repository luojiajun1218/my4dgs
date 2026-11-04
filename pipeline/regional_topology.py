"""Gate-independent, world-space regional colour evidence for frame-plan births.

All regions form a disjoint partition of the occupied D5 leaves.  A weak leaf
is pooled with other unresolved descendants at its closest reliable ancestor;
leaves already resolved below that ancestor are never counted a second time.
Depth comes from the immutable frame-zero candidate layout, not the active gate.
"""
from pathlib import Path

import numpy as np
import torch

from .model.spatial_gate import SPATIAL_ROUTING_FIELDS, spatial_state_route


@torch.no_grad()
def measurement_layout(module):
    """Load the fixed D5 frame-zero tree once; do not infer it from this frame."""
    cached = getattr(module, '_regional_color_layout', None)
    if cached is not None:
        return cached
    cfg = module.config.birth
    depth = int(cfg.regional_color_depth)
    path = cfg.regional_color_reference_layout
    if path:
        from .model.rigid_tree_codec import DecodedState, _validate_state
        with np.load(path, allow_pickle=False) as data:
            layout = DecodedState.from_reference(data)
            extent = float(data['extent'])
        _validate_state(layout)
        if layout.frame_index != 0 or not np.isclose(
                extent, float(module.datamodule.cameras_extent), rtol=1e-6, atol=1e-8):
            raise ValueError('Regional measurement layout must be frame 0 in the same world coordinates')
        label = str(Path(path))
    else:
        layout = module.motion.frame_reference_state
        label = 'immutable frame-zero motion candidate layout'
    if (depth != 5 or layout is None or layout.spatial_paths is None or
            layout.spatial_paths.shape[1] != depth or layout.spatial_centers is None):
        raise ValueError('Regional colour measurement requires a fixed D5 reference layout')
    module._regional_color_layout = (layout, label)
    return layout, label


def assign_measurement_units(paths, numerator, denominator, threshold=.03,
                             min_support=32., min_views=2):
    """Aggregate per-GS evidence to mutually exclusive reliable regions.

    `paths` is [N,5] and evidence is [V,N], with raw Talpha-weighted L1
    numerators and Talpha pixel weights. A unit may be the unresolved
    remainder of an ancestor after reliable descendants have been removed.
    Scores are equal-weight means of independently normalized reliable views.
    The returned NumPy arrays are also suitable for a compressed audit sidecar.
    """
    paths = np.asarray(paths, dtype=np.int64)
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if (paths.ndim != 2 or paths.shape[1] != 5 or numerator.ndim != 2 or
            numerator.shape != denominator.shape or numerator.shape[1] != len(paths) or
            not np.isfinite(numerator).all() or not np.isfinite(denominator).all() or
            (numerator < 0).any() or (denominator < 0).any() or
            not np.isfinite(threshold) or threshold < 0 or
            not np.isfinite(min_support) or min_support <= 0 or
            isinstance(min_views, bool) or not isinstance(min_views, int) or min_views < 1):
        raise ValueError('Invalid regional colour evidence or reliability configuration')
    views = numerator.shape[0]
    if min_views > views:
        raise ValueError('Regional colour measurement needs enough training views')
    # Verify every path really is a breadth-first octree path, including its root.
    for level in range(5):
        lo = sum(8 ** i for i in range(level + 1))
        hi = lo + 8 ** (level + 1)
        if ((paths[:, level] < lo) | (paths[:, level] >= hi)).any():
            raise ValueError('Regional colour paths are outside the fixed D5 tree')
        if level:
            parent_lo = sum(8 ** i for i in range(level))
            if not np.array_equal(parent_lo + (paths[:, level] - lo) // 8, paths[:, level-1]):
                raise ValueError('Regional colour path has inconsistent parent')
    leaf, first, inverse = np.unique(paths[:, -1], return_inverse=True, return_index=True)
    leaf_paths = paths[first]
    leaf_num = np.zeros((len(leaf), views), dtype=np.float64)
    leaf_den = np.zeros_like(leaf_num)
    for view in range(views):
        np.add.at(leaf_num[:, view], inverse, numerator[view])
        np.add.at(leaf_den[:, view], inverse, denominator[view])
    node_count = sum(8 ** i for i in range(6))
    pending_num = np.zeros((node_count, views), dtype=np.float64)
    pending_den = np.zeros_like(pending_num)
    pending_count = np.zeros(node_count, dtype=np.int64)
    pending_num[leaf] = leaf_num
    pending_den[leaf] = leaf_den
    pending_count[leaf] = 1
    leaf_unit = np.full(len(leaf), -1, dtype=np.int64)
    units_node, units_depth, units_num, units_den, units_score, units_reason = [], [], [], [], [], []
    for depth in range(5, -1, -1):
        lo = sum(8 ** i for i in range(depth)) if depth else 0
        hi = lo + 8 ** depth if depth else 1
        nodes = np.flatnonzero(pending_count[lo:hi]) + lo
        if not len(nodes):
            continue
        reliable = pending_den[nodes] >= min_support
        counts = reliable.sum(axis=1)
        ready = counts >= min_views
        finalize = ready | (depth == 0)
        selected_nodes = nodes[finalize]
        node_to_unit = np.full(node_count, -1, dtype=np.int64)
        for node in selected_nodes:
            node = int(node)
            support = pending_den[node] >= min_support
            score = (float((pending_num[node, support] / pending_den[node, support]).mean())
                     if int(support.sum()) >= min_views else np.nan)
            # 0 = insufficient evidence, 1 = below/equal threshold, 2 = triggered.
            reason = 0 if np.isnan(score) else (2 if score > threshold else 1)
            unit = len(units_node)
            node_to_unit[node] = unit
            units_node.append(node)
            units_depth.append(depth)
            units_num.append(pending_num[node].copy())
            units_den.append(pending_den[node].copy())
            units_score.append(score)
            units_reason.append(reason)
        current_nodes = (np.zeros(len(leaf), dtype=np.int64) if depth == 0 else
                         leaf_paths[:, depth-1])
        assignments = node_to_unit[current_nodes]
        take = (leaf_unit < 0) & (assignments >= 0)
        leaf_unit[take] = assignments[take]
        unready = nodes[~finalize]
        if len(unready):
            if depth == 1:
                parents = np.zeros(len(unready), dtype=np.int64)
            else:
                parent_lo = sum(8 ** i for i in range(depth-1))
                parents = parent_lo + (unready - lo) // 8
            np.add.at(pending_num, parents, pending_num[unready])
            np.add.at(pending_den, parents, pending_den[unready])
            np.add.at(pending_count, parents, pending_count[unready])
    if (leaf_unit < 0).any():
        raise RuntimeError('Regional evidence lost occupied leaves')
    def matrix(rows):
        return np.asarray(rows, dtype=np.float64).reshape(-1, views)
    result = dict(leaf_id=leaf, leaf_numerator=leaf_num, leaf_denominator=leaf_den,
                  leaf_unit=leaf_unit, point_unit=leaf_unit[inverse],
                  unit_node=np.asarray(units_node, dtype=np.int64),
                  unit_depth=np.asarray(units_depth, dtype=np.int8),
                  unit_numerator=matrix(units_num), unit_denominator=matrix(units_den),
                  unit_score=np.asarray(units_score, dtype=np.float64),
                  unit_reason=np.asarray(units_reason, dtype=np.int8))
    for prefix in ('leaf', 'unit'):
        num, den = result[f'{prefix}_numerator'], result[f'{prefix}_denominator']
        score = np.full_like(num, np.nan)
        np.divide(num, den, out=score, where=den > 0)
        result[f'{prefix}_view_score'] = score
        result[f'{prefix}_view_reliable'] = den >= min_support
    return result


def native_colour_evidence(module, cache):
    """Measure the actual full-scene Talpha response, retaining no model grads."""
    renderer = module.renderer
    old = module.gaussians
    xyz = old.get_xyz
    previous_model = renderer.gaussians
    previous_collect = getattr(renderer, 'collect_contributions', False)
    has_last = hasattr(renderer, 'last_contributions')
    previous_last = getattr(renderer, 'last_contributions', None)
    numerators, denominators, uncovered = [], [], []
    try:
        renderer.gaussians = old
        renderer.collect_contributions = False
        for item in cache:
            view = item['view']
            if Path(str(view.path)).name == 'cam00.png' or bool(getattr(view, 'is_test', False)):
                raise ValueError('Regional colour evidence cannot use external test cameras')
            l1 = item['residual'].detach().abs().mean(0)
            with torch.enable_grad():
                probe = torch.ones_like(xyz, requires_grad=True)
                coverage_rgb = renderer(view, torch.zeros_like(module.background),
                                        override_color=probe)['render']
                if coverage_rgb.shape != (3, *l1.shape):
                    raise ValueError('Regional colour render has an unexpected pixel shape')
                gradient = torch.autograd.grad((coverage_rgb * l1[None]).sum() / 3,
                                               probe, only_inputs=True)[0]
            coverage = coverage_rgb.detach().mean(0)
            numerator = gradient.detach().sum(-1).clamp_min(0)
            denominator = item['native_mass'].detach().clamp_min(0)
            if (not bool(torch.isfinite(numerator).all() & torch.isfinite(denominator).all()
                         & torch.isfinite(coverage).all() & torch.isfinite(l1).all()) or
                    bool((coverage < -1e-4).any()) or bool((coverage > 1 + 1e-4).any())):
                raise ValueError('Regional colour native contribution is nonfinite or outside [0,1]')
            missing = (1 - coverage.clamp(0, 1))
            numerators.append(numerator.cpu().double().numpy())
            denominators.append(denominator.cpu().double().numpy())
            uncovered.append(dict(camera=Path(str(view.path)).name,
                total_l1=float(l1.double().sum()), attributed_l1=float(numerator.double().sum()),
                unassigned_l1=float((missing.double() * l1.double()).sum()),
                unassigned_effective_pixels=float(missing.double().sum()),
                full_pixel_count=int(l1.numel()),
                semantics='transmittance-weighted L1 without existing GS contribution; no depth inferred'))
    finally:
        renderer.gaussians = previous_model
        renderer.collect_contributions = previous_collect
        if has_last:
            renderer.last_contributions = previous_last
        elif hasattr(renderer, 'last_contributions'):
            delattr(renderer, 'last_contributions')
    return np.stack(numerators), np.stack(denominators), uncovered


def select_region_candidates(units, scores, observed_view_count, cap):
    """Rank Talpha-weighted L1 mass only among points in triggered units."""
    scores = np.asarray(scores, dtype=np.float64)
    observed_view_count = np.asarray(observed_view_count)
    point_unit = units['point_unit']
    if scores.shape != point_unit.shape or observed_view_count.shape != scores.shape or cap < 0:
        raise ValueError('Regional candidate evidence does not match GS rows')
    eligible = (scores > 0) & (observed_view_count >= 2) & (units['unit_reason'][point_unit] == 2)
    rows = np.flatnonzero(eligible)
    if not len(rows) or not cap:
        return np.empty(0, dtype=np.int64)
    # Partitioning never changes ordering for the same triggered evidence.
    # Greater total explainable error may legitimately earn more candidates;
    # there is no quota per region. Stable row IDs resolve exact ties.
    order = np.lexsort((rows, -scores[rows]))
    return rows[order[:cap]]


def route_seed_units(layout, xyz, units):
    """Resolve seed positions to the same occupied measurement partition."""
    paths = spatial_state_route(xyz, layout, 5).numpy()
    leaves = units['leaf_id']
    positions = np.searchsorted(leaves, paths[:, -1])
    valid = positions < len(leaves)
    valid[valid] &= leaves[positions[valid]] == paths[valid, -1]
    result = np.full(len(xyz), -1, dtype=np.int64)
    result[valid] = units['leaf_unit'][positions[valid]]
    return paths[:, -1], result


def save_region_diagnostics(module, step, units, metadata, **extra):
    directory = Path(module.output_dir) / 'frame_diagnostics' / 'regional_color'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'frame_{int(module.current_frame):03d}_step_{int(step):03d}.npz'
    payload = {key: value for key, value in units.items() if key != 'point_unit'}
    payload.update(extra)
    payload['camera_names'] = np.asarray(metadata['camera_names'])
    payload['threshold'] = np.asarray(metadata['threshold'])
    payload['min_support'] = np.asarray(metadata['min_support'])
    payload['min_views'] = np.asarray(metadata['min_views'])
    # Write through a same-directory temporary file so checkpoints never point
    # at a partially written audit file after interruption.
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    return str(path.relative_to(module.output_dir))
