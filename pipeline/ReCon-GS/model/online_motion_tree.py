"""Causal motion trees over persistent leaf regions, with no GS-owning parents.

Logical region IDs are separate from dense codec slots. Parent admission uses
an explicitly named quantized-magnitude objective, not a claim of byte savings.
All leaf world transforms remain exact before quantization. The integration
must quantize region parameters and retarget independent GS residuals against
the same target pose before checking the actual complete motion packet.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class MotionTreeConfig:
    max_depth: int = 3
    update_interval: int = 10
    neighbors: int = 8
    max_children: int = 16
    max_new_parents: int = 32
    min_observed_gs: int = 3
    max_fit_points: int = 2048
    history_frames: int = 5
    fit_tolerance_extent: float = 4e-4
    rotation_tolerance_rad: float = .004
    min_quantized_reduction: float = .25

    def validate(self):
        for name in ('max_depth', 'update_interval', 'neighbors', 'max_children',
                     'max_new_parents', 'min_observed_gs', 'max_fit_points', 'history_frames'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'motion_tree.{name} must be a positive integer')
        if (self.max_depth > 3 or self.max_children < 2 or self.min_observed_gs < 3 or
                self.max_fit_points < self.min_observed_gs * self.max_children or
                self.history_frames > 5):
            raise ValueError('motion tree supports depth<=3, 2+ children and observed positional SE(3) fits')
        for name in ('fit_tolerance_extent', 'rotation_tolerance_rad'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'motion_tree.{name} must be finite and positive')
        if (isinstance(self.min_quantized_reduction, bool) or
                not math.isfinite(self.min_quantized_reduction) or
                not 0 < self.min_quantized_reduction < 1):
            raise ValueError('motion_tree.min_quantized_reduction must lie in (0,1)')


@dataclass
class MotionTreeProposal:
    region_ids: np.ndarray
    parents: np.ndarray
    pivots: np.ndarray
    world_quaternion: np.ndarray
    world_offset: np.ndarray
    local_translation: np.ndarray
    local_rotvec: np.ndarray
    next_region_id: int
    metrics: dict


def validate_motion_tree(parents, owner=None, max_depth=3):
    """Return topological order/depth; reject cycles and GS-owned internals."""
    parents = np.asarray(parents)
    if (parents.ndim != 1 or not len(parents) or parents.dtype.kind not in 'iu' or
            int(parents[0]) != -1 or np.any(parents[1:] < 0) or
            np.any(parents[1:] >= len(parents))):
        raise ValueError('motion tree needs identity root and valid parent slots')
    depth = np.full(len(parents), -1, np.int64)
    depth[0] = 0
    order = []
    for start in range(1, len(parents)):
        path, visiting, node = [], set(), start
        while depth[node] < 0:
            if node in visiting:
                raise ValueError('motion tree contains a cycle')
            visiting.add(node)
            path.append(node)
            node = int(parents[node])
        for node in reversed(path):
            depth[node] = depth[int(parents[node])] + 1
            order.append(node)
    if depth.max(initial=0) > max_depth:
        raise ValueError('motion tree exceeds maximum depth')
    if owner is not None:
        owner = np.asarray(owner)
        if (owner.ndim != 1 or owner.dtype.kind not in 'iu' or
                np.any(owner < 0) or np.any(owner >= len(parents))):
            raise ValueError('invalid GS owner slots')
        internal = np.zeros(len(parents), bool)
        internal[parents[1:]] = True
        internal[0] = False  # Root may own static/independent GS.
        if np.any(internal[owner]):
            raise ValueError('internal motion nodes must not directly own GS')
    return np.asarray(order, np.int64), depth


def _world_inputs(parents, pivots, quaternion, offset):
    order, _ = validate_motion_tree(parents, max_depth=max(3, len(parents)))
    n = len(parents)
    pivots, quaternion, offset = (np.asarray(v, np.float64) for v in (pivots, quaternion, offset))
    if (pivots.shape != (n, 3) or quaternion.shape != (n, 4) or offset.shape != (n, 3) or
            not all(np.isfinite(v).all() for v in (pivots, quaternion, offset)) or
            np.any(np.linalg.norm(quaternion, axis=1) == 0)):
        raise ValueError('invalid motion tree world transforms')
    matrices = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
    if not np.allclose(matrices[0], np.eye(3), atol=1e-10) or not np.allclose(offset[0], 0., atol=1e-10):
        raise ValueError('root motion must be identity')
    return order, pivots, matrices, offset


def world_to_local_motion(parents, pivots, world_quaternion, world_offset):
    """Exact inverse of the repository's inherited-pivot composition.

    R_child = R_local R_parent, and
    b_child = b_parent + (I-R_local) R_parent pivot + t_local.
    This is deliberately the codec convention, not naive quaternion subtraction
    or a generic parent-frame affine offset.
    """
    parents = np.asarray(parents)
    order, pivots, matrix, offset = _world_inputs(parents, pivots, world_quaternion, world_offset)
    translation, rotvec = np.zeros_like(offset), np.zeros_like(offset)
    for node in order:
        parent = int(parents[node])
        relative = matrix[node] @ matrix[parent].T
        inherited_pivot = matrix[parent] @ pivots[node]
        translation[node] = offset[node] - offset[parent] - inherited_pivot + relative @ inherited_pivot
        rotvec[node] = Rotation.from_matrix(relative).as_rotvec()
    return translation, rotvec


def local_to_world_motion(parents, pivots, local_translation, local_rotvec):
    parents = np.asarray(parents)
    order, _ = validate_motion_tree(parents, max_depth=max(3, len(parents)))
    n = len(parents)
    pivots, translation, rotvec = (np.asarray(v, np.float64) for v in
                                  (pivots, local_translation, local_rotvec))
    if (pivots.shape != (n, 3) or translation.shape != (n, 3) or rotvec.shape != (n, 3) or
            not all(np.isfinite(v).all() for v in (pivots, translation, rotvec)) or
            not np.allclose(translation[0], 0.) or not np.allclose(rotvec[0], 0.)):
        raise ValueError('invalid local motion tree transforms')
    local = Rotation.from_rotvec(rotvec).as_matrix()
    matrix = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    offset = np.zeros((n, 3))
    for node in order:
        parent = int(parents[node])
        inherited_pivot = matrix[parent] @ pivots[node]
        matrix[node] = local[node] @ matrix[parent]
        offset[node] = offset[parent] + inherited_pivot - local[node] @ inherited_pivot + translation[node]
    q = Rotation.from_matrix(matrix).as_quat()[:, [3, 0, 1, 2]]
    q[q[:, 0] < 0] *= -1
    return q, offset


def _step(value):
    value = np.broadcast_to(np.asarray(value, np.float64), (3,)).copy()
    if not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError('regional motion steps must be finite and positive')
    return value


def quantized_regional_cost(translation, rotvec, translation_step, rotation_step):
    """Integer L1 and exact raw sparse regional section bytes, excluding root.

    Integer magnitude is the admission/refinement objective; these two values
    must never be confused. Complete compressed packet costs belong to caller.
    """
    t = np.rint(np.asarray(translation, np.float64) / _step(translation_step))
    r = np.rint(np.asarray(rotvec, np.float64) / _step(rotation_step))
    if t.shape != r.shape or t.ndim != 2 or t.shape[1] != 3 or not np.isfinite(t).all() or not np.isfinite(r).all():
        raise ValueError('invalid regional quantized motion')
    if np.any(np.abs(t) > 32767) or np.any(np.abs(r) > 32767):
        raise ValueError('regional motion exceeds int16 support')

    def varint_size(value):
        return max(1, (int(value).bit_length() + 6) // 7)

    active_t, active_r = np.any(t[1:] != 0, axis=1), np.any(r[1:] != 0, axis=1)
    rows = np.flatnonzero(active_t | active_r)
    gaps = np.diff(np.concatenate(([-1], rows)))
    size = varint_size(len(rows)) + sum(varint_size(gap) for gap in gaps)
    size += len(rows) + 6 * int(active_t.sum() + active_r.sum())
    return dict(integer_l1=float(np.abs(t[1:]).sum() + np.abs(r[1:]).sum()),
                sparse_bytes=int(size), active_translation_blocks=int(active_t.sum()),
                active_rotation_blocks=int(active_r.sum()))


def _fit_positions(source, target):
    center_x, center_y = source.mean(0), target.mean(0)
    x, y = source - center_x, target - center_y
    left, singular, right = np.linalg.svd(y.T @ x)
    # Collinear/coincident samples cannot support a physical 3D rotation.
    if singular[1] <= max(singular[0] * 1e-10, 1e-20):
        return None
    correction = np.ones(3)
    correction[-1] = 1. if np.linalg.det(left @ right) >= 0 else -1.
    matrix = (left * correction[None]) @ right
    offset = center_y - matrix @ center_x
    error = np.linalg.norm(source @ matrix.T + offset - target, axis=1)
    q = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    if q[0] < 0:
        q *= -1
    return q, offset, float(np.sqrt(np.square(error).mean()))


def propose_motion_tree(source_xyz, target_xyz, owner, region_ids, world_quaternion,
                        world_offset, extent, translation_step, rotation_step, config,
                        parents=None, pivots=None, observed=None, history=None,
                        next_region_id=None, frame_index=None):
    """Insert coherent parents among siblings, preserving existing valid trees.

    ``history`` accepts OnlineMotionHistory.aligned(): source/target [N,T,3],
    observed [N,T], chronological frames. Current observations are explicit;
    an already-present current slot is excluded, and future evidence rejected.
    Missing historical support is skipped, never padded into stationary motion.
    Each new parent must reduce the affected children's integer L1 by >=25%
    including its own motion. Unrelated nodes are not part of that local gate.
    """
    config.validate()
    source, target = (np.asarray(v, np.float64) for v in (source_xyz, target_xyz))
    owner, ids = np.asarray(owner), np.asarray(region_ids)
    n = len(ids)
    if (source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape or
            owner.shape != (len(source),) or ids.ndim != 1 or ids.dtype.kind not in 'iu' or
            not n or int(ids[0]) != 0 or np.any(ids < 0) or len(np.unique(ids)) != n or
            not math.isfinite(extent) or extent <= 0):
        raise ValueError('invalid persistent-region tree inputs')
    ids = ids.astype(np.int64, copy=True)
    parents = np.concatenate(([-1], np.zeros(n - 1, np.int64))) if parents is None else np.asarray(parents).copy()
    if parents.shape != (n,):
        raise ValueError('logical region IDs must align every tree slot')
    order, _ = validate_motion_tree(parents, owner, config.max_depth)
    observed = np.ones(len(source), bool) if observed is None else np.asarray(observed)
    if (observed.shape != (len(source),) or observed.dtype.kind != 'b' or
            not np.isfinite(source).all() or not np.isfinite(target[observed]).all()):
        raise ValueError('motion tree needs finite reference positions and observed targets')
    if pivots is None:
        sums, counts = np.zeros((n, 3)), np.zeros(n)
        np.add.at(sums, owner, source)
        np.add.at(counts, owner, 1.)
        for node in order[::-1]:
            sums[parents[node]] += sums[node]
            counts[parents[node]] += counts[node]
        pivots = np.zeros((n, 3))
        if counts[0]:
            pivots[0] = sums[0] / counts[0]
        for node in order:
            pivots[node] = sums[node] / counts[node] if counts[node] else pivots[parents[node]]
    else:
        pivots = np.asarray(pivots, np.float64).copy()
    quaternion, offset = (np.asarray(v, np.float64).copy() for v in (world_quaternion, world_offset))
    _world_inputs(parents, pivots, quaternion, offset)
    step_t, step_r = _step(translation_step), _step(rotation_step)
    translation, rotvec = world_to_local_motion(parents, pivots, quaternion, offset)
    before = quantized_regional_cost(translation, rotvec, step_t, step_r)
    if next_region_id is None:
        next_region_id = int(ids.max()) + 1
    if (isinstance(next_region_id, bool) or not isinstance(next_region_id, (int, np.integer))
            or next_region_id <= int(ids.max())):
        raise ValueError('new logical region IDs must be monotonic and unused')
    metrics = dict(admission_objective='affected_quantized_integer_l1',
                   required_reduction=config.min_quantized_reduction, before=before,
                   inserted_parents=[], candidate_evaluations=0,
                   current_observed_gs=int(observed.sum()), history_transitions_checked=0,
                   update_skipped=False)
    metrics['rejection_counts'] = dict(parent_mismatch=0, depth=0, current_support=0,
        current_degenerate_fit=0, current_position_fit=0, child_rotation=0,
        history_degenerate_fit=0, history_position_fit=0,
        quantization_overflow=0, quantized_gain=0)
    metrics['historical_missing_support_checks'] = 0
    history_slots = []
    if history is not None:
        hs, ht, hm = (np.asarray(history[key]) for key in ('source', 'target', 'observed'))
        if (hs.ndim != 3 or hs.shape[:1] != source.shape[:1] or hs.shape[2] != 3 or
                ht.shape != hs.shape or hm.shape != hs.shape[:2] or hm.dtype.kind != 'b'):
            raise ValueError('history must align stable current GS IDs and observation masks')
        frames = history.get('frames')
        if frames is not None and (len(frames) != hs.shape[1] or list(frames) != sorted(set(frames))):
            raise ValueError('tree history must be chronological actual transitions')
        for slot in range(hs.shape[1]):
            if frames is not None and frame_index is not None:
                if frames[slot] > frame_index:
                    raise ValueError('motion tree cannot consume future evidence')
                if frames[slot] == frame_index:
                    continue
            if not np.isfinite(hs[:, slot][hm[:, slot]]).all() or not np.isfinite(ht[:, slot][hm[:, slot]]).all():
                raise ValueError('observed historical motion must be finite')
            history_slots.append((hs[:, slot], ht[:, slot], hm[:, slot]))
        history_slots = history_slots[-max(0, config.history_frames - 1):] if config.history_frames > 1 else []
    metrics['history_transitions_checked'] = len(history_slots)
    if frame_index is not None:
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or frame_index < 1:
            raise ValueError('tree updates need a positive current frame index')
        metrics['update_skipped'] = frame_index != 1 and frame_index % config.update_interval != 0

    def subtree_rows():
        tree_order, _ = validate_motion_tree(parents, owner, config.max_depth)
        rows = [np.flatnonzero(owner == node) for node in range(len(parents))]
        for node in tree_order[::-1]:
            parent = int(parents[node])
            rows[parent] = np.concatenate((rows[parent], rows[node]))
        return rows

    def sample_support(branches, mask):
        parts = [rows[node][mask[rows[node]]] for node in branches]
        if any(len(part) < config.min_observed_gs for part in parts):
            return None
        # Equal per-branch samples prevent a large child hiding a small one.
        budget = config.max_fit_points // len(parts)
        return np.concatenate([part[np.linspace(0, len(part) - 1, min(len(part), budget), dtype=np.int64)]
                               for part in parts])

    def evaluate(branches):
        metrics['candidate_evaluations'] += 1
        parent = int(parents[branches[0]])
        if any(parents[node] != parent for node in branches):
            metrics['rejection_counts']['parent_mismatch'] += 1
            return None
        _, depth = validate_motion_tree(parents, max_depth=config.max_depth)
        affected = np.concatenate([rows[node] for node in branches])
        # All descendants of the grouped branches gain one level.
        branch_set = set(branches)
        for node in range(1, len(parents)):
            ancestor = node
            while ancestor > 0 and ancestor not in branch_set:
                ancestor = int(parents[ancestor])
            if ancestor in branch_set and depth[node] >= config.max_depth:
                metrics['rejection_counts']['depth'] += 1
                return None
        sample = sample_support(branches, observed)
        if sample is None:
            metrics['rejection_counts']['current_support'] += 1
            return None
        fit = _fit_positions(source[sample], target[sample])
        if fit is None or fit[2] > config.fit_tolerance_extent * extent:
            metrics['rejection_counts']['current_degenerate_fit' if fit is None else 'current_position_fit'] += 1
            return None
        q, b, error = fit
        parent_matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        branch_matrix = Rotation.from_quat(quaternion[branches][:, [1, 2, 3, 0]]).as_matrix()
        if np.any(Rotation.from_matrix(branch_matrix @ parent_matrix.T).magnitude() > config.rotation_tolerance_rad):
            metrics['rejection_counts']['child_rotation'] += 1
            return None
        historical_errors = []
        for hs, ht, hm in history_slots:
            support = sample_support(branches, hm)
            if support is None:
                metrics['historical_missing_support_checks'] += 1
                continue
            historical = _fit_positions(hs[support].astype(np.float64), ht[support].astype(np.float64))
            if historical is None or historical[2] > config.fit_tolerance_extent * extent:
                metrics['rejection_counts']['history_degenerate_fit' if historical is None else 'history_position_fit'] += 1
                return None
            historical_errors.append(historical[2])
        new_parents = np.concatenate((parents, [parent]))
        new_parents[branches] = len(parents)
        new_pivots = np.concatenate((pivots, source[affected].mean(0, keepdims=True)))
        new_q, new_b = np.concatenate((quaternion, q[None])), np.concatenate((offset, b[None]))
        nt, nr = world_to_local_motion(new_parents, new_pivots, new_q, new_b)
        # Affected-section root placeholder allows the same quantization/cost
        # function to count all child locals and the new parent exactly once.
        selected = np.asarray(branches + [len(parents)])
        old_selected = np.asarray(branches)
        old_t = np.concatenate((np.zeros((1, 3)), translation[old_selected]))
        old_r = np.concatenate((np.zeros((1, 3)), rotvec[old_selected]))
        new_t = np.concatenate((np.zeros((1, 3)), nt[selected]))
        new_r = np.concatenate((np.zeros((1, 3)), nr[selected]))
        try:
            old_cost = quantized_regional_cost(old_t, old_r, step_t, step_r)
            new_cost = quantized_regional_cost(new_t, new_r, step_t, step_r)
        except ValueError:
            metrics['rejection_counts']['quantization_overflow'] += 1
            return None
        gain = old_cost['integer_l1'] - new_cost['integer_l1']
        if old_cost['integer_l1'] <= 0 or gain / old_cost['integer_l1'] < config.min_quantized_reduction:
            metrics['rejection_counts']['quantized_gain'] += 1
            return None
        return dict(children=branches, parents=new_parents, pivots=new_pivots,
                    quaternion=new_q, offset=new_b, translation=nt, rotvec=nr,
                    gain=gain, before=old_cost, after=new_cost, current_fit_rms=error,
                    historical_fit_rms=historical_errors)

    if not metrics['update_skipped']:
        for _ in range(config.max_new_parents):
            rows = subtree_rows()
            best = None
            for parent in range(len(parents)):
                siblings = np.asarray([node for node in range(1, len(parents))
                                       if parents[node] == parent and len(rows[node])])
                if len(siblings) < 2:
                    continue
                q = Rotation.from_quat(quaternion[siblings][:, [1, 2, 3, 0]]).as_rotvec()
                features = np.concatenate((offset[siblings] / (config.fit_tolerance_extent * extent),
                                           q / config.rotation_tolerance_rad), axis=1)
                k = min(len(siblings), config.neighbors + 1)
                _, neighbors = cKDTree(features).query(features, k=k, workers=1)
                neighbors = np.asarray(neighbors).reshape(len(siblings), k)
                edges = sorted({tuple(sorted((int(siblings[i]), int(siblings[j]))))
                                for i in range(len(siblings)) for j in neighbors[i] if i != j})
                lookup = {int(node): i for i, node in enumerate(siblings)}
                for a, b in edges:
                    candidate = evaluate([a, b])
                    if candidate is None:
                        continue
                    pool = set(siblings[neighbors[lookup[a]]].tolist() + siblings[neighbors[lookup[b]]].tolist())
                    pool.difference_update((a, b))
                    for child in sorted(pool, key=lambda node: int(ids[node])):
                        if len(candidate['children']) >= config.max_children:
                            break
                        grown = evaluate(candidate['children'] + [child])
                        if grown is not None and grown['gain'] > candidate['gain']:
                            candidate = grown
                    key = (-candidate['gain'], tuple(int(ids[node]) for node in candidate['children']))
                    if best is None or key < best[0]:
                        best = key, candidate
            if best is None:
                break
            candidate = best[1]
            children = candidate['children']
            parent_slot, logical_id = len(parents), int(next_region_id)
            metrics['inserted_parents'].append(dict(region_id=logical_id, slot=parent_slot,
                parent_region_id=int(ids[parents[children[0]]]),
                child_region_ids=[int(ids[node]) for node in children],
                integer_l1_before=candidate['before']['integer_l1'],
                integer_l1_after=candidate['after']['integer_l1'],
                quantized_reduction=candidate['gain'] / candidate['before']['integer_l1'],
                current_fit_rms=candidate['current_fit_rms'],
                historical_fit_rms=candidate['historical_fit_rms']))
            parents, pivots = candidate['parents'], candidate['pivots']
            quaternion, offset = candidate['quaternion'], candidate['offset']
            translation, rotvec = candidate['translation'], candidate['rotvec']
            ids = np.concatenate((ids, [logical_id]))
            next_region_id += 1
    validate_motion_tree(parents, owner, config.max_depth)
    metrics['after'] = quantized_regional_cost(translation, rotvec, step_t, step_r)
    metrics['added_node_pivot_payload_bytes'] = 12 * len(metrics['inserted_parents'])
    metrics['complete_packet_bytes_scope'] = 'caller measures exact codec packet after regional quantization and GS retargeting'
    metrics['shared_fit'] = 'observed positional SE(3); covariance quaternion evidence is not assumed'
    return MotionTreeProposal(ids, parents, pivots, quaternion, offset, translation,
                              rotvec, int(next_region_id), metrics)
