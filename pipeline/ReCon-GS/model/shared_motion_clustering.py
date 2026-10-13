"""Deterministic shared-SE(3) groups, without a prescribed cluster count.

Inputs are poses already learned by the ordinary motion optimizer. No extra
rendering, optimization, future frames, or spatial partition is performed here.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class MotionClusteringConfig:
    translation_tolerance_extent: float = 4e-4
    rotation_tolerance_rad: float = 4e-3
    orientation_length_extent: float = 0.01
    min_cluster_gs: int = 8
    min_observations: int = 1
    motion_neighbors: int = 8
    motion_neighbor_radius: float = 2.0
    split_noise_steps: float = 0.5

    def validate(self):
        for name in ('translation_tolerance_extent', 'rotation_tolerance_rad',
                     'orientation_length_extent', 'motion_neighbor_radius', 'split_noise_steps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'clustering.{name} must be finite and positive')
        for name in ('min_cluster_gs', 'min_observations', 'motion_neighbors'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'clustering.{name} must be a positive integer')


def quaternion_matrices(quaternions):
    values = np.asarray(quaternions, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 4 or
            not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0)):
        raise ValueError('Expected finite nonzero scalar-first quaternions [N,4]')
    return Rotation.from_quat(values[:, [1, 2, 3, 0]]).as_matrix()


def matrix_quaternion(matrix):
    quaternion = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    return quaternion if quaternion[0] >= 0 else -quaternion


def fit_shared_motion(source, target, relative_rotation, weights,
                      orientation_length):
    """Weighted rigid fit of positions and orientation frames, in float64.

    Returns a scalar-first quaternion and a world affine offset. Rotation is
    fitted jointly with positions: a rotating rigid object is not mistaken for
    many translations merely because its points have different displacements.
    """
    source, target = (np.asarray(value, dtype=np.float64)
                      for value in (source, target))
    rotations = np.asarray(relative_rotation, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or
            rotations.shape != (len(source), 3, 3) or weights.shape != (len(source),) or
            not len(source) or not all(np.isfinite(value).all() for value in
                                      (source, target, rotations, weights)) or
            np.any(weights < 0) or weights.sum() <= 0 or
            not math.isfinite(orientation_length) or orientation_length <= 0):
        raise ValueError('Invalid shared motion fitting inputs')
    weights = weights / weights.sum()
    source_center = (source * weights[:, None]).sum(0)
    target_center = (target * weights[:, None]).sum(0)
    covariance = ((target - target_center)[:, :, None] *
                  (source - source_center)[:, None, :] +
                  orientation_length ** 2 * rotations)
    covariance = (covariance * weights[:, None, None]).sum(0)
    left, _, right = np.linalg.svd(covariance)
    correction = np.ones(3)
    correction[-1] = 1 if np.linalg.det(left @ right) >= 0 else -1
    matrix = (left * correction[None, :]) @ right
    offset = target_center - matrix @ source_center
    position_error = np.linalg.norm(source @ matrix.T + offset - target, axis=1)
    rotation_error = Rotation.from_matrix(
        rotations @ matrix.T).magnitude()
    return dict(quaternion=matrix_quaternion(matrix), offset=offset,
                position_error=position_error, rotation_error=rotation_error)


def motion_features(source, target, relative_quaternion,
                    translation_tolerance, rotation_tolerance):
    """SE(3) affine/rotation features, not XYZ coordinates or point velocities."""
    matrices = quaternion_matrices(relative_quaternion)
    affine = target - np.einsum('nij,nj->ni', matrices, source)
    rotvec = Rotation.from_matrix(matrices).as_rotvec()
    return np.concatenate((affine / translation_tolerance,
                           rotvec / rotation_tolerance), axis=1), matrices


def shared_motion_groups(source, target, relative_quaternion, weights,
                         extent, config):
    """Group similar motion bins and verify each group with a shared rigid fit.

    Motion bins seed the search; nearby motion bins may merge if the union can
    share an SE(3). The number of bins/groups is data-dependent. Spatially
    separated points can share a group. Incompatible/undersupported points
    retain their existing GS residual rather than forcing singleton regions.
    """
    config.validate()
    source, target, quaternion, weights = (
        np.asarray(value, dtype=np.float64)
        for value in (source, target, relative_quaternion, weights))
    if (source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or
            quaternion.shape != (len(source), 4) or weights.shape != (len(source),) or
            not math.isfinite(extent) or extent <= 0 or
            not all(np.isfinite(value).all() for value in
                    (source, target, quaternion, weights)) or np.any(weights <= 0)):
        raise ValueError('Invalid motion clustering inputs')
    if len(source) < config.min_cluster_gs:
        return []
    tolerance_t = config.translation_tolerance_extent * extent
    tolerance_r = config.rotation_tolerance_rad
    features, rotations = motion_features(source, target, quaternion,
                                         tolerance_t, tolerance_r)
    cells = np.rint(features)
    if np.abs(cells).max(initial=0) >= np.iinfo(np.int64).max:
        raise ValueError('Motion features exceed integer bin range')
    _, inverse = np.unique(cells.astype(np.int64), axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    order = np.argsort(inverse, kind='stable')
    bins = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    centers = np.stack([features[rows].mean(0) for rows in bins])
    k = min(count, config.motion_neighbors + 1)
    _, neighbors = cKDTree(centers).query(centers, k=k, workers=1)
    neighbors = np.asarray(neighbors).reshape(count, k)

    def fit(rows):
        return fit_shared_motion(source[rows], target[rows], rotations[rows],
                                 weights[rows], config.orientation_length_extent * extent)

    def coherent(result):
        return (result['position_error'].max(initial=0) <= tolerance_t and
                result['rotation_error'].max(initial=0) <= tolerance_r)

    consumed = np.zeros(count, dtype=bool)
    groups = []
    for index in range(count):
        if consumed[index]:
            continue
        rows = bins[index]
        result = fit(rows)
        if not coherent(result):
            # Motion-feature bisection resolves incompatible seeds without a
            # spatial split or a target number of regions.
            pending = [rows]
            while pending:
                part = pending.pop()
                if len(part) < config.min_cluster_gs:
                    continue
                fitted = fit(part)
                if coherent(fitted):
                    groups.append(dict(rows=part, **fitted))
                    continue
                axis = int(np.ptp(features[part], axis=0).argmax())
                ranked = part[np.argsort(features[part, axis], kind='stable')]
                half = len(ranked) // 2
                pending.extend((ranked[half:], ranked[:half]))
            consumed[index] = True
            continue
        consumed[index] = True
        for candidate in sorted(set(int(value) for value in neighbors[index])):
            if (consumed[candidate] or np.max(np.abs(centers[index] - centers[candidate])) >
                    config.motion_neighbor_radius):
                continue
            proposed = np.sort(np.concatenate((rows, bins[candidate])))
            fitted = fit(proposed)
            if coherent(fitted):
                rows, result = proposed, fitted
                consumed[candidate] = True
        if len(rows) >= config.min_cluster_gs:
            groups.append(dict(rows=rows, **result))
    return sorted(groups, key=lambda group: int(group['rows'].min()))


def residual_exceeds_parent(child_translation, child_rotvec,
                            parent_translation, parent_rotvec,
                            translation_noise, rotation_noise):
    """User's split rule: compare translation/rotation amplitudes separately."""
    return bool(
        np.linalg.norm(child_translation) > np.linalg.norm(parent_translation) + translation_noise or
        np.linalg.norm(child_rotvec) > np.linalg.norm(parent_rotvec) + rotation_noise)
