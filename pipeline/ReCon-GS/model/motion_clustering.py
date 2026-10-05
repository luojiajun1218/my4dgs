"""Local, hard motion clustering with a shared SE(3) fitting criterion.

Only current-frame training probes are consumed. NumPy float64 reductions and
stable ownership order make the layout reproducible without a CUDA renderer.
"""

import numpy as np
from scipy.spatial import cKDTree


def quaternion_matrices(quaternions):
    """Convert wxyz quaternions [N, 4] to rotation matrices [N, 3, 3]."""
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if values.ndim != 2 or values.shape[1] != 4 or np.any(norms <= 0):
        raise ValueError('Expected nonzero wxyz quaternions [N, 4]')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite probe quaternions')
    w, x, y, z = (values / norms).T
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * z + w * y), 2 * (x * y + w * z),
                     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x),
                     1 - 2 * (x * x + y * y)), axis=-1).reshape(-1, 3, 3)


def relative_matrices(base_rotation, target_rotation):
    base = quaternion_matrices(base_rotation)
    target = quaternion_matrices(target_rotation)
    return target @ base.transpose(0, 2, 1)


def fit_rigid_groups(xyz, target_xyz, relative_rotation, owner, count,
                     weights, orientation_length):
    """Weighted rigid fit, with orientation frames resolving small-group ambiguity.

    Minimize position squared error plus orientation_length**2 times the
    Frobenius squared error between the shared and per-GS rotations. Groups
    without observations return identity motion; their spatial centers still
    follow their members for candidate lookup.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    relative_rotation = np.asarray(relative_rotation, dtype=np.float64)
    owner = np.asarray(owner, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or target_xyz.shape != xyz.shape or
            relative_rotation.shape != (len(xyz), 3, 3) or
            owner.shape != (len(xyz),) or weights.shape != (len(xyz),) or
            count < 1 or np.any(owner < 0) or np.any(owner >= count)):
        raise ValueError('Invalid rigid fitting shapes or owners')
    if (not all(np.isfinite(value).all() for value in
                (xyz, target_xyz, relative_rotation, weights)) or
            np.any(weights < 0) or not np.isfinite(orientation_length) or
            orientation_length < 0):
        raise ValueError('Invalid rigid fitting values')
    mass = np.bincount(owner, weights=weights, minlength=count)
    source_sum = np.zeros((count, 3), dtype=np.float64)
    target_sum = np.zeros_like(source_sum)
    np.add.at(source_sum, owner, xyz * weights[:, None])
    np.add.at(target_sum, owner, target_xyz * weights[:, None])
    denominator = np.maximum(mass, np.finfo(np.float64).tiny)[:, None]
    source_mean = source_sum / denominator
    target_mean = target_sum / denominator
    source_centered = xyz - source_mean[owner]
    target_centered = target_xyz - target_mean[owner]
    covariance = np.zeros((count, 3, 3), dtype=np.float64)
    values = (target_centered[:, :, None] * source_centered[:, None, :] +
              orientation_length ** 2 * relative_rotation) * weights[:, None, None]
    np.add.at(covariance, owner, values)
    left, _, right = np.linalg.svd(covariance)
    correction = np.ones((count, 3), dtype=np.float64)
    correction[:, -1] = np.where(np.linalg.det(left @ right) < 0, -1., 1.)
    rotation = (left * correction[:, None, :]) @ right
    translation = target_mean - np.einsum('nij,nj->ni', rotation, source_mean)
    empty = mass == 0
    rotation[empty] = np.eye(3)
    translation[empty] = 0
    centers = source_mean.copy()
    sizes = np.bincount(owner, minlength=count)
    fallback = np.zeros_like(centers)
    np.add.at(fallback, owner, xyz)
    centers[empty] = (fallback / np.maximum(sizes, 1)[:, None])[empty]
    return rotation, translation, centers, mass


def rigid_fit_metrics(xyz, target_xyz, relative_rotation, owner, count,
                      weights, extent, orientation_extent, spatial_weight=0.):
    rotation, translation, centers, mass = fit_rigid_groups(
        xyz, target_xyz, relative_rotation, owner, count, weights,
        extent * orientation_extent)
    predicted = np.einsum('nij,nj->ni', rotation[owner], xyz) + translation[owner]
    position = np.square(predicted - target_xyz).sum(-1) / extent ** 2
    orientation = np.square(rotation[owner] - relative_rotation).sum((1, 2))
    spatial = np.square(xyz - centers[owner]).sum(-1) / extent ** 2
    denominator = max(float(weights.sum()), np.finfo(np.float64).tiny)
    objective = (position + orientation_extent ** 2 * orientation +
                 spatial_weight * spatial)
    metrics = dict(position_rmse_extent=float(np.sqrt(np.dot(weights, position) / denominator)),
                   rotation_chord_rmse=float(np.sqrt(np.dot(weights, orientation) / denominator)),
                   objective=float(np.dot(weights, objective) / denominator),
                   occupied_regions=int(np.count_nonzero(np.bincount(owner, minlength=count))),
                   observed_regions=int(np.count_nonzero(mass)))
    return (rotation, translation, centers, mass), metrics


def _seed_children(xyz, parent_owner, child_parents):
    """Spatially initialize each parent's existing child slots, keeping the budget."""
    parent_count = int(child_parents.max()) + 1
    order = np.argsort(parent_owner, kind='stable')
    boundaries = np.cumsum(np.bincount(parent_owner, minlength=parent_count))[:-1]
    members = np.split(order, boundaries)
    child_order = np.argsort(child_parents, kind='stable')
    child_boundaries = np.cumsum(np.bincount(child_parents, minlength=parent_count))[:-1]
    slots = np.split(child_order, child_boundaries)
    owner = np.empty(len(xyz), dtype=np.int64)
    for rows, children in zip(members, slots):
        if not len(rows):
            continue
        if not len(children):
            raise ValueError('An occupied parent has no child slots')
        axis = np.ptp(xyz[rows], axis=0).argmax()
        ordered = rows[np.lexsort((rows, xyz[rows, axis]))]
        for child, part in zip(children, np.array_split(ordered, len(children))):
            owner[part] = child
    return owner


def _child_candidates(parent_owner, child_parents):
    parent_count = int(child_parents.max()) + 1
    sizes = np.bincount(child_parents, minlength=parent_count)
    lookup = np.full((parent_count, int(sizes.max())), -1, dtype=np.int64)
    offset = np.zeros(parent_count, dtype=np.int64)
    for child, parent in enumerate(child_parents):
        lookup[parent, offset[parent]] = child
        offset[parent] += 1
    return lookup[parent_owner]


def _refine(xyz, target_xyz, relative_rotation, owner, count, weights, extent,
            neighbors, iterations, spatial_weight, orientation_extent,
            parent_owner=None, child_parents=None):
    owner = owner.copy()
    _, initial = rigid_fit_metrics(xyz, target_xyz, relative_rotation, owner,
        count, weights, extent, orientation_extent, spatial_weight)
    changes = []
    for _ in range(iterations):
        (rotation, translation, centers, mass), _ = rigid_fit_metrics(
            xyz, target_xyz, relative_rotation, owner, count, weights, extent,
            orientation_extent, spatial_weight)
        if parent_owner is None:
            occupied = np.flatnonzero(np.bincount(owner, minlength=count))
            tree = cKDTree(centers[occupied])
            _, near = tree.query(xyz, k=min(neighbors, len(occupied)), workers=1)
            candidates = occupied[np.asarray(near).reshape(len(xyz), -1)]
        else:
            candidates = _child_candidates(parent_owner, child_parents)
        # Current ownership is first, so equal scores keep the current group.
        candidates = np.column_stack((owner, candidates))
        valid = candidates >= 0
        candidates = np.maximum(candidates, 0)
        predicted = np.einsum('nkij,nj->nki', rotation[candidates], xyz)
        predicted += translation[candidates]
        score = np.square(predicted - target_xyz[:, None]).sum(-1) / extent ** 2
        score += orientation_extent ** 2 * np.square(
            rotation[candidates] - relative_rotation[:, None]).sum((2, 3))
        score += spatial_weight * np.square(
            xyz[:, None] - centers[candidates]).sum(-1) / extent ** 2
        valid &= (mass[candidates] > 0)
        valid[:, 0] = True
        score[~valid] = np.inf
        proposed = candidates[np.arange(len(xyz)), score.argmin(-1)]
        proposed[weights == 0] = owner[weights == 0]
        changed = int(np.count_nonzero(proposed != owner))
        changes.append(changed)
        owner = proposed
        if not changed:
            break
    _, final = rigid_fit_metrics(xyz, target_xyz, relative_rotation, owner,
        count, weights, extent, orientation_extent, spatial_weight)
    return owner, dict(before=initial, after=final, changed_per_iteration=changes)


def cluster_motion_layout(xyz, base_rotation, target_xyz, target_rotation,
                          seed_index, counts, parents, observations, *, extent,
                          neighbors, iterations, spatial_weight,
                          orientation_extent, min_observations):
    """Return a nested three-level layout with the seed's exact region budget."""
    xyz = np.asarray(xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    seed_index = np.asarray(seed_index, dtype=np.int64)
    parents = [np.asarray(value, dtype=np.int64) for value in parents]
    if (not np.isfinite(extent) or extent <= 0 or neighbors < 1 or
            iterations < 1 or min_observations < 1 or
            spatial_weight < 0 or not np.isfinite(spatial_weight) or
            orientation_extent <= 0 or not np.isfinite(orientation_extent) or
            seed_index.shape != (3, len(xyz)) or len(counts) != 3 or
            len(parents) != 2 or observations.shape != (len(xyz),) or
            not np.isfinite(observations).all() or np.any(observations < 0)):
        raise ValueError('Invalid motion clustering parameters')
    qualified = observations >= min_observations
    weights = np.where(qualified, observations, 0.)
    weights /= max(float(weights.max()), 1.)
    relative_rotation = relative_matrices(base_rotation, target_rotation)
    index, levels = [], []
    for level in range(3):
        parent_owner = None if level == 0 else index[level - 1]
        child_parents = None if level == 0 else parents[level - 1]
        seed = (seed_index[0].copy() if level == 0 else
                _seed_children(xyz, parent_owner, child_parents))
        owner, metrics = _refine(xyz, target_xyz, relative_rotation, seed,
            int(counts[level]), weights, extent, neighbors, iterations,
            spatial_weight, orientation_extent, parent_owner, child_parents)
        index.append(owner)
        levels.append(dict(level=level, **metrics))
    return np.stack(index), dict(qualified_rows=int(qualified.sum()),
        unqualified_rows=int((~qualified).sum()), levels=levels,
        semantics='Current training-view probe; local shared-SE3 fitting with '
            'orientation consistency. Unobserved rows keep their spatial seed; '
            'each child remains inside its selected parent. Region budgets are unchanged.')
