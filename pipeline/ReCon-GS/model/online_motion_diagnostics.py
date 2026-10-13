"""CPU diagnostics for flat shared motions; independent of training/rendering.

Quaternions are scalar-first. Shared translations are world affine offsets:
``predicted_xyz = R(shared_quaternion) @ source_xyz + shared_translation``.
Residuals are measured against this composed transform, not quaternion deltas.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


def _array(value, dtype=np.float64):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def ownership_changed_fraction(previous_owner, current_owner):
    """Minimum reassigned fraction after matching arbitrary region labels.

    Compare common rows only: callers must align survivor rows first. Additional
    current rows (births) have no previous ownership and are excluded. Negative
    labels denote semantic categories and are matched only to themselves.
    """
    before = _array(previous_owner, np.int64).reshape(-1)
    after = _array(current_owner, np.int64).reshape(-1)
    count = min(len(before), len(after))
    if count == 0:
        return 0.0
    before, after = before[:count], after[:count]
    matches = int(np.count_nonzero((before < 0) & (before == after)))
    valid = (before >= 0) & (after >= 0)
    if valid.any():
        _, left = np.unique(before[valid], return_inverse=True)
        _, right = np.unique(after[valid], return_inverse=True)
        overlap = np.zeros((int(left.max()) + 1, int(right.max()) + 1), np.int64)
        np.add.at(overlap, (left, right), 1)
        rows, columns = linear_sum_assignment(overlap, maximize=True)
        matches += int(overlap[rows, columns].sum())
    return float((count - matches) / count)


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return dict(count=0, mean=None, p50=None, p90=None, max=None)
    return dict(count=int(len(values)), mean=float(values.mean()),
                p50=float(np.quantile(values, .5)),
                p90=float(np.quantile(values, .9)), max=float(values.max()))


def _mask(value, count, default=False):
    if value is None:
        return np.full(count, default, dtype=bool)
    result = _array(value, bool)
    if result.shape != (count,):
        raise ValueError('Diagnostic masks must match all current GS rows')
    return result


def quantized_motion_counts(region_translation, region_rotation,
                            gs_translation, gs_rotation):
    """JSON-safe nonzero int16 block counts; identity root is excluded.

    Input arrays must be already quantized integer symbols [rows,3]. GS counts
    include every surviving row, not just moving or observed rows. This makes
    partition/refinement snapshots comparable with identical memberships.
    """
    rt, rr, gt, gr = (_array(value, dtype=None) for value in (
        region_translation, region_rotation, gs_translation, gs_rotation))
    if (any(value.ndim != 2 or value.shape[1] != 3 or value.dtype.kind not in 'iu'
            for value in (rt, rr, gt, gr)) or rt.shape != rr.shape or gt.shape != gr.shape):
        raise ValueError('Quantized motion counts need aligned integer [rows,3] arrays')
    region_t, region_r = (np.any(value[1:] != 0, axis=1) for value in (rt, rr))
    gs_t, gs_r = (np.any(value != 0, axis=1) for value in (gt, gr))
    active = gs_t | gs_r
    count = len(active)
    return dict(region_translation_active_blocks=int(region_t.sum()),
                region_rotation_active_blocks=int(region_r.sum()),
                region_active_nodes=int((region_t | region_r).sum()),
                gs_translation_active_blocks=int(gs_t.sum()),
                gs_rotation_active_blocks=int(gs_r.sum()), gs_active_rows=int(active.sum()),
                gs_zero_residual_rows=int((~active).sum()), total_surviving_gs=count,
                gs_zero_residual_fraction=float((~active).sum() / count) if count else 0.,
                gs_translation_zero_fraction=float((~gs_t).sum() / count) if count else 0.,
                gs_rotation_zero_fraction=float((~gs_r).sum() / count) if count else 0.)


def _rotations(quaternions):
    norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(~np.isfinite(quaternions)) or np.any(norm <= 0):
        raise ValueError('Diagnostic rotations must be finite nonzero quaternions')
    return Rotation.from_quat((quaternions / norm)[:, [1, 2, 3, 0]])


def summarize_flat_regions(source_xyz, target_xyz, source_rotation,
                           target_rotation, owner, shared_quaternion,
                           shared_translation, *, observed_mask=None,
                           static_mask=None, noise_mask=None, unknown_mask=None,
                           previous_owner=None, candidate_owner=None, extent=1.0):
    """Return JSON-safe per-region shared motion and independent GS residuals.

    Counts and proportions include *all* current GS, including unobserved GS.
    Residual statistics include observed finite rows only. A missing observation
    mask means all finite rows are observed; an explicit all-false mask yields
    null statistics, never artificial zero residuals. Births may be masked out.
    Unknown/noise/static masks refer to proposal status, independent of owner IDs.
    """
    source, target, source_q, target_q = map(_array, (
        source_xyz, target_xyz, source_rotation, target_rotation))
    owners = _array(owner, np.int64)
    shared_q, shared_t = map(_array, (shared_quaternion, shared_translation))
    count = len(owners)
    regions = len(shared_q)
    if (owners.shape != (count,) or source.shape != (count, 3) or
            target.shape != (count, 3) or source_q.shape != (count, 4) or
            target_q.shape != (count, 4) or shared_q.shape != (regions, 4) or
            shared_t.shape != (regions, 3)):
        raise ValueError('Diagnostic motion arrays have incompatible shapes')
    if np.any(owners >= regions) or not np.isfinite(shared_t).all():
        raise ValueError('Diagnostic shared transforms/owners are invalid')
    extent = float(extent)
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('Diagnostic extent must be finite and positive')
    shared = _rotations(shared_q) if regions else None
    observed = _mask(observed_mask, count, default=True)
    finite = (np.isfinite(source).all(1) & np.isfinite(target).all(1) &
              np.isfinite(source_q).all(1) & np.isfinite(target_q).all(1) &
              (np.linalg.norm(source_q, axis=1) > 0) &
              (np.linalg.norm(target_q, axis=1) > 0))
    usable = observed & finite & (owners >= 0)
    rows = np.flatnonzero(usable)
    translation_error = np.full(count, np.nan)
    rotation_error = np.full(count, np.nan)
    if len(rows):
        predicted = shared[owners[rows]].apply(source[rows]) + shared_t[owners[rows]]
        translation_error[rows] = np.linalg.norm(target[rows] - predicted, axis=1)
        predicted_q = shared[owners[rows]] * _rotations(source_q[rows])
        rotation_error[rows] = (_rotations(target_q[rows]) * predicted_q.inv()).magnitude()
    denominator = max(count, 1)
    result = dict(total_gs=count, region_count=regions,
                  populated_region_count=int(len(np.unique(owners[owners >= 0]))),
                  observed_gs=int(observed.sum()), residual_valid_gs=int(usable.sum()),
                  unknown_fraction=float(_mask(unknown_mask, count).sum() / denominator)
                  if unknown_mask is not None else float((owners < 0).sum() / denominator),
                  noise_fraction=float(_mask(noise_mask, count).sum() / denominator),
                  static_fraction=float(_mask(static_mask, count).sum() / denominator),
                  units=dict(translation='scene world units',
                             translation_extent='fraction of scene extent',
                             rotation='radians', rotation_degrees='degrees'),
                  shared_translation_convention='affine offset: R*x + t', regions=[])
    for region in range(regions):
        members = owners == region
        valid = members & usable
        rotvec = shared[region].as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        quaternion = shared[region].as_quat()[[3, 0, 1, 2]]
        if quaternion[0] < 0:
            quaternion = -quaternion
        result['regions'].append(dict(
            region_id=region, gs_count=int(members.sum()),
            gs_proportion_all=float(members.sum() / denominator),
            observed_gs_count=int((members & observed).sum()),
            residual_valid_gs_count=int(valid.sum()),
            shared_translation_xyz=shared_t[region].tolist(),
            shared_translation_norm=float(np.linalg.norm(shared_t[region])),
            shared_rotation_quaternion_wxyz=quaternion.tolist(),
            shared_rotation_rotvec_rad=rotvec.tolist(),
            shared_rotation_axis=(rotvec / angle).tolist() if angle > 1e-12 else [0., 0., 0.],
            shared_rotation_angle_rad=angle,
            shared_rotation_angle_degrees=float(np.degrees(angle)),
            independent_translation_residual=_summary(translation_error[valid]),
            independent_translation_residual_extent=_summary(translation_error[valid] / extent),
            independent_rotation_residual_rad=_summary(rotation_error[valid]),
            independent_rotation_residual_degrees=_summary(np.degrees(rotation_error[valid]))))
    if previous_owner is not None:
        previous = _array(previous_owner, np.int64).reshape(-1)
        result['ownership_changed_fraction'] = ownership_changed_fraction(previous, owners)
        result['ownership_compared_gs'] = min(len(previous), count)
        result['new_gs_without_prior_owner'] = max(0, count - len(previous))
    if candidate_owner is not None:
        candidate = _array(candidate_owner, np.int64).reshape(-1)
        result['candidate_region_count'] = int(len(np.unique(candidate[candidate >= 0])))
        if previous_owner is not None:
            result['candidate_ownership_changed_fraction'] = ownership_changed_fraction(previous_owner, candidate)
    return result
