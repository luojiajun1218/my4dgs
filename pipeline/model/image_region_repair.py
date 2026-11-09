"""Choose bounded frame-local attribute repairs from two rendered views."""

import math

import numpy as np


def select_image_error_regions(leaf, xyz, numerator, denominator, *,
                               threshold, min_support, max_group_points, max_points,
                               priority='mean_error', error_metric='l1'):
    """Select high-error candidate leaves, splitting overfull leaves spatially.

    ``numerator`` and ``denominator`` are the two views' per-GS accumulated
    T-alpha-weighted L1 or squared RGB error and visibility. For squared
    error the threshold applies to the root mean squared error, while the
    excess-per-point priority uses squared-error mass. A selected group makes its
    GS eligible for all post-motion point-attribute updates. It does not change
    the motion tree or introduce a transmitted group mask.
    """
    leaf = np.asarray(leaf)
    xyz = np.asarray(xyz)
    numerator = np.asarray(numerator)
    denominator = np.asarray(denominator)
    count = len(leaf)
    if (leaf.ndim != 1 or leaf.dtype.kind not in 'iu' or count == 0 or
            np.any(leaf < 0) or xyz.shape != (count, 3) or
            numerator.shape != (2, count) or denominator.shape != (2, count) or
            not np.isfinite(xyz).all() or not np.isfinite(numerator).all() or
            not np.isfinite(denominator).all() or np.any(numerator < 0) or
            np.any(denominator < 0)):
        raise ValueError('Image repair needs finite two-view evidence and aligned GS leaves')
    for name, value in (('threshold', threshold), ('min_support', min_support)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name, value in (('max_group_points', max_group_points), ('max_points', max_points)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if priority not in ('mean_error', 'excess_per_point', 'mean_then_excess'):
        raise ValueError('Unknown image repair priority')
    if error_metric not in ('l1', 'mse'):
        raise ValueError('Unknown image repair error metric')

    # The motion cut is untouched. Groups begin as prebuilt deepest leaves;
    # only oversized leaves are split into deterministic spatial slabs.
    order = np.argsort(leaf, kind='stable')
    ordered_leaf = leaf[order]
    starts = np.r_[0, np.flatnonzero(ordered_leaf[1:] != ordered_leaf[:-1]) + 1]
    leaf_sizes = np.diff(np.r_[starts, count])
    leaves = ordered_leaf[starts]
    for start, size in zip(starts[leaf_sizes > max_group_points],
                           leaf_sizes[leaf_sizes > max_group_points]):
        rows = order[start:start + size]
        axis = int(np.argmax(np.ptp(xyz[rows], axis=0)))
        order[start:start + size] = rows[np.argsort(xyz[rows, axis], kind='stable')]

    groups_per_leaf = (leaf_sizes + max_group_points - 1) // max_group_points
    group_start = np.cumsum(np.r_[0, groups_per_leaf[:-1]])
    local_index = np.arange(count) - np.repeat(starts, leaf_sizes)
    ordered_groups = np.repeat(group_start, leaf_sizes) + local_index // max_group_points
    group_for_point = np.empty(count, dtype=np.int64)
    group_for_point[order] = ordered_groups
    group_leaf = np.repeat(leaves, groups_per_leaf)
    groups = len(group_leaf)
    group_count = np.bincount(group_for_point, minlength=groups)
    # Pool evidence from both random views. A zero-support group is unknown,
    # not a measured zero-error group.
    num = np.bincount(group_for_point, weights=numerator.sum(0, dtype=np.float64), minlength=groups)
    den = np.bincount(group_for_point, weights=denominator.sum(0, dtype=np.float64), minlength=groups)
    mean_error = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    score = np.sqrt(mean_error) if error_metric == 'mse' else mean_error
    reliable = den >= min_support
    candidates = np.flatnonzero(reliable & (score > threshold))
    # The optional priority values excess visible error per eligible point;
    # point count is a fixed budget proxy, not a prediction of coded bytes.
    threshold_error = threshold * threshold if error_metric == 'mse' else threshold
    excess = np.maximum(mean_error - threshold_error, 0.) * den / group_count
    # Stable ordering makes equal-priority decisions reproducible. Skip groups
    # that cannot fit; later smaller groups can still use the remaining budget.
    # The mixed option reserves half the point budget for mean error and then
    # uses the rest for visible excess per point, without rendering more views.
    selected = np.zeros(groups, dtype=bool)
    def fill(rank, remaining):
        ordered = candidates[np.argsort(-rank[candidates], kind='stable')]
        for group in ordered:
            size = int(group_count[group])
            if not selected[group] and size <= remaining:
                selected[group] = True
                remaining -= size
                if not remaining:
                    break
        return remaining
    if priority == 'mean_then_excess':
        fill(score, max(1, max_points // 2))
        fill(excess, max_points - int(group_count[selected].sum()))
    else:
        fill(score if priority == 'mean_error' else excess, max_points)
    return dict(leaf=group_leaf, point_count=group_count, support=den, score=score,
                reliable=reliable, selected=selected, point_group=group_for_point,
                eligible=selected[group_for_point], candidate_count=len(candidates))
