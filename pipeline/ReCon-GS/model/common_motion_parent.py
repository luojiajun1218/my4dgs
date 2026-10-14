"""Unactivated common-component parents derived from child WORLD sharing.

Parents explain the shared regional transforms, not the per-GS corrections.
Virtual targets are R_child*x+b_child. Different child rotations/translations
remain exact relative motions; virtual RMS is diagnostic, never a rigid-object
or equal-rotation gate. Current and historical admission use the same actual
quantized parent-plus-child magnitude objective, separately from byte counts.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .online_motion_tree import (MotionTreeProposal, _fit_positions, _step,
    _world_inputs, quantized_regional_cost, validate_motion_tree, world_to_local_motion)


@dataclass
class CommonMotionParentConfig:
    max_depth: int = 3
    update_interval: int = 10
    neighbors: int = 8
    max_children: int = 16
    max_new_parents: int = 32
    min_observed_gs: int = 3
    max_fit_points: int = 2048
    history_frames: int = 5
    min_quantized_reduction: float = .25

    def validate(self):
        for name in ('max_depth', 'update_interval', 'neighbors', 'max_children',
                     'max_new_parents', 'min_observed_gs', 'max_fit_points', 'history_frames'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'common parent {name} must be a positive integer')
        if (self.max_depth > 3 or self.max_children < 2 or self.min_observed_gs < 3 or
                self.max_fit_points < self.max_children * self.min_observed_gs or self.history_frames > 5):
            raise ValueError('common parents require depth<=3 and bounded supported positional fits')
        if (isinstance(self.min_quantized_reduction, bool) or
                not math.isfinite(self.min_quantized_reduction) or
                not 0 < self.min_quantized_reduction < 1):
            raise ValueError('common parent reduction must lie in (0,1)')


def _rows_and_pivots(source, owner, parents, observed):
    order, _ = validate_motion_tree(parents, owner)
    rows = [np.flatnonzero(owner == slot) for slot in range(len(parents))]
    for slot in order[::-1]:
        parent = int(parents[slot])
        rows[parent] = np.sort(np.concatenate((rows[parent], rows[slot])))
    pivots = np.zeros((len(parents), 3))
    for slot in np.concatenate(([0], order)):
        supported = rows[slot][observed[rows[slot]]]
        if len(supported):
            pivots[slot] = source[supported].mean(0)
        elif slot:
            pivots[slot] = pivots[int(parents[slot])]
    return rows, pivots, order


def _observations(source, owner, parents, observed):
    source, owner, parents = np.asarray(source, np.float64), np.asarray(owner), np.asarray(parents)
    observed = np.ones(len(source), bool) if observed is None else np.asarray(observed)
    if (source.ndim != 2 or source.shape[1] != 3 or owner.shape != (len(source),) or
            observed.shape != owner.shape or observed.dtype.kind != 'b' or
            not np.isfinite(source[observed]).all()):
        raise ValueError('common parents need observed finite source positions and aligned GS ownership')
    validate_motion_tree(parents, owner)
    return source, owner, parents, observed


def _virtual_common_fit(source, rows, branches, quaternion, offset, observed, valid, config):
    branches = [int(slot) for slot in branches if valid[int(slot)] and
                int(observed[rows[int(slot)]].sum()) >= config.min_observed_gs]
    if len(branches) < 2:
        return None, 'insufficient_supported_children'
    budget = config.max_fit_points // len(branches)
    x, y, weights = [], [], []
    for slot in branches:
        support = rows[slot][observed[rows[slot]]]
        sample = support[np.linspace(0, len(support) - 1, min(len(support), budget), dtype=np.int64)]
        points = source[sample]
        matrix = Rotation.from_quat(quaternion[slot, [1, 2, 3, 0]]).as_matrix()
        x.append(points)
        y.append(points @ matrix.T + offset[slot])
        # Each logical child contributes equally, irrespective of GS population.
        weights.append(np.full(len(sample), 1. / (len(branches) * len(sample))))
    x, y, weights = np.concatenate(x), np.concatenate(y), np.concatenate(weights)
    cx, cy = ((value * weights[:, None]).sum(0) for value in (x, y))
    covariance = np.einsum('ni,nj,n->ij', y - cy, x - cx, weights)
    left, singular, right = np.linalg.svd(covariance)
    if singular[1] <= max(singular[0] * 1e-10, 1e-20):
        return None, 'degenerate_virtual_geometry'
    correction = np.ones(3)
    correction[-1] = 1. if np.linalg.det(left @ right) >= 0 else -1.
    matrix = (left * correction[None]) @ right
    affine = cy - matrix @ cx
    rms = float(np.sqrt((np.square(np.linalg.norm(x @ matrix.T + affine - y, axis=1)) * weights).sum()))
    q = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    if q[0] < 0:
        q *= -1
    return dict(quaternion=q, offset=affine, virtual_rms=rms,
                supported_children=branches, fit_points=len(x)), None


def refit_parent_world_from_children(source_xyz, owner, parents, pivots,
                                     world_quaternion, world_offset, observed=None,
                                     world_valid=None, config=None):
    """Bottom-up internal refit, with observed leaf WORLD supplied by caller.

    Unsupported nodes retain their input WORLD and are marked invalid, rather
    than treating an invented identity as stationary evidence. A parent requires
    at least two valid directly attached children. Slot order need not be depth
    order. This function returns WORLD values; the caller derives local values,
    quantizes them and retargets GS against the same current target.
    """
    config = CommonMotionParentConfig() if config is None else config
    config.validate()
    source, owner, parents, observed = _observations(source_xyz, owner, parents, observed)
    _, pivots, _, _ = _world_inputs(parents, pivots, world_quaternion, world_offset)
    q, b = np.asarray(world_quaternion, np.float64).copy(), np.asarray(world_offset, np.float64).copy()
    rows, _, order = _rows_and_pivots(source, owner, parents, observed)
    if world_valid is None:
        valid = np.array([int(observed[part].sum()) >= config.min_observed_gs for part in rows])
    else:
        valid = np.asarray(world_valid)
        if valid.shape != (len(parents),) or valid.dtype.kind != 'b':
            raise ValueError('world_valid must be a boolean per-slot evidence mask')
        valid = valid.copy()
        valid &= np.array([int(observed[part].sum()) >= config.min_observed_gs for part in rows])
    valid[0] = True
    metrics = dict(fit_semantics='direct supported child WORLD virtual targets',
                   internal_nodes=[], unsupported_internal_nodes=[], failed_internal_fits=[])
    for slot in order[::-1]:
        children = np.flatnonzero(parents == slot).tolist()
        if not children:
            continue
        fit, reason = _virtual_common_fit(source, rows, children, q, b, observed, valid, config)
        if fit is None:
            valid[slot] = False
            destination = ('unsupported_internal_nodes' if reason == 'insufficient_supported_children'
                           else 'failed_internal_fits')
            metrics[destination].append(dict(slot=int(slot), reason=reason))
        else:
            q[slot], b[slot], valid[slot] = fit['quaternion'], fit['offset'], True
            metrics['internal_nodes'].append(dict(slot=int(slot), child_slots=fit['supported_children'],
                                                 virtual_rms=fit['virtual_rms'], fit_points=fit['fit_points']))
    return dict(world_quaternion=q, world_offset=b, valid_slot=valid, metrics=metrics)


def _quantized_world_moves(source, rows, branches, q, b, observed, step_t, step_r):
    for slot in branches:
        matrix = Rotation.from_quat(q[slot, [1, 2, 3, 0]]).as_matrix()
        if np.any(np.rint(Rotation.from_matrix(matrix).as_rotvec() / step_r) != 0):
            return True
        support = rows[slot][observed[rows[slot]]]
        if len(support) and np.any(np.rint((source[support] @ matrix.T + b[slot] - source[support]) / step_t) != 0):
            return True
    return False


def _affected_cost(t, r, slots, step_t, step_r):
    slots = np.asarray(slots, np.int64)
    return quantized_regional_cost(np.concatenate((np.zeros((1, 3)), t[slots])),
                                   np.concatenate((np.zeros((1, 3)), r[slots])), step_t, step_r)


def propose_common_motion_parents(source_xyz, owner, region_ids, world_quaternion,
                                  world_offset, extent, translation_step, rotation_step,
                                  config, parents=None, pivots=None, observed=None,
                                  world_valid=None, history=None, next_region_id=None,
                                  frame_index=None):
    """Sibling parent proposals with current/historical common-component gains.

    Current targets come only from supplied child WORLD sharing. Historical
    raw observations first fit leaf WORLD values; parent extraction then uses
    the same virtual-target semantics. Each historical pivot uses that slot's
    actual source and supported observed GS, excluding missing/born placeholders.
    Missing support and all-quantized-zero WORLD transitions are reported as
    unavailable evidence and never improve the historical gain denominator.
    """
    config.validate()
    ids = np.asarray(region_ids)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or int(ids[0]) != 0 or
            np.any(ids < 0) or len(np.unique(ids)) != len(ids) or
            not math.isfinite(extent) or extent <= 0):
        raise ValueError('common parents need stable unique logical IDs and positive extent')
    ids = ids.astype(np.int64, copy=True)
    parents = np.concatenate(([-1], np.zeros(len(ids) - 1, np.int64))) if parents is None else np.asarray(parents).copy()
    if parents.shape != ids.shape:
        raise ValueError('logical IDs and parent slots must align')
    source, owner, parents, observed = _observations(source_xyz, owner, parents, observed)
    validate_motion_tree(parents, owner, config.max_depth)
    rows, derived_pivots, _ = _rows_and_pivots(source, owner, parents, observed)
    pivots = derived_pivots if pivots is None else np.asarray(pivots, np.float64).copy()
    q, b = np.asarray(world_quaternion, np.float64).copy(), np.asarray(world_offset, np.float64).copy()
    _world_inputs(parents, pivots, q, b)
    valid = (np.array([int(observed[part].sum()) >= config.min_observed_gs for part in rows])
             if world_valid is None else np.asarray(world_valid).copy())
    if valid.shape != ids.shape or valid.dtype.kind != 'b':
        raise ValueError('world_valid must align the original region slots')
    valid &= np.array([int(observed[part].sum()) >= config.min_observed_gs for part in rows])
    valid[0] = True
    step_t, step_r = _step(translation_step), _step(rotation_step)
    t, r = world_to_local_motion(parents, pivots, q, b)
    before = quantized_regional_cost(t, r, step_t, step_r)
    next_id = int(ids.max()) + 1 if next_region_id is None else next_region_id
    if isinstance(next_id, bool) or not isinstance(next_id, (int, np.integer)) or next_id <= int(ids.max()):
        raise ValueError('new parent logical IDs must be monotonic and unused')
    if frame_index is not None and (isinstance(frame_index, bool) or
            not isinstance(frame_index, (int, np.integer)) or frame_index < 1):
        raise ValueError('current frame must be a positive integer')
    metrics = dict(admission_objective='affected parent-plus-child quantized integer L1',
        fit_semantics='child WORLD virtual targets; GS residuals excluded from parent fit',
        virtual_rms_gate=False, child_equal_rotation_gate=False,
        required_reduction=config.min_quantized_reduction, before=before,
        candidate_evaluations=0, inserted_parents=[], history_frames=[],
        history_pivot_scope='actual per-transition source of supported observed leaf GS; missing/birth omitted',
        update_skipped=bool(frame_index is not None and frame_index != 1 and frame_index % config.update_interval != 0),
        rejection_counts=dict(depth=0, current_support=0, current_zero_world=0,
            current_virtual_fit=0, current_quantized_gain=0, history_quantized_gain=0, overflow=0))

    # Historical leaf fits are independent of the proposed parent topology and
    # cached once. New internal slots are appended, so old leaf slots stay valid.
    historical = []
    if history is not None:
        hs, ht, hm = (np.asarray(history[key]) for key in ('source', 'target', 'observed'))
        frames = history.get('frames')
        if (hs.ndim != 3 or hs.shape[0] != len(source) or hs.shape[2] != 3 or ht.shape != hs.shape or
                hm.shape != hs.shape[:2] or hm.dtype.kind != 'b' or frames is None or frame_index is None or
                len(frames) != hs.shape[1] or list(frames) != sorted(set(frames))):
            raise ValueError('history needs aligned actual chronological frames and current frame index')
        slots = []
        for slot, frame in enumerate(frames):
            if frame > frame_index:
                raise ValueError('common parents cannot consume future motion evidence')
            if frame < frame_index:
                slots.append(slot)
        slots = slots[-(config.history_frames - 1):] if config.history_frames > 1 else []
        direct = [np.flatnonzero(owner == slot) for slot in range(len(ids))]
        for slot in slots:
            mask = hm[:, slot].copy()
            x, y = hs[:, slot].astype(np.float64), ht[:, slot].astype(np.float64)
            if not np.isfinite(x[mask]).all() or not np.isfinite(y[mask]).all():
                raise ValueError('observed historical motion must be finite')
            hq, hb, hv = np.zeros((len(ids), 4)), np.zeros((len(ids), 3)), np.zeros(len(ids), bool)
            hq[:, 0], hv[0] = 1., True
            supported_mask = np.zeros(len(owner), bool)
            for leaf in np.unique(owner[owner > 0]):
                support = direct[int(leaf)][mask[direct[int(leaf)]]]
                if len(support) < config.min_observed_gs:
                    continue
                sample = support[np.linspace(0, len(support) - 1, min(len(support), config.max_fit_points), dtype=np.int64)]
                fitted = _fit_positions(x[sample], y[sample])
                if fitted is not None:
                    hq[leaf], hb[leaf], hv[leaf] = fitted[0], fitted[1], True
                    supported_mask[support] = True
            historical.append(dict(frame=int(frames[slot]), source=x, observed=supported_mask,
                                   leaf_q=hq, leaf_b=hb, leaf_valid=hv))
        metrics['history_frames'] = [item['frame'] for item in historical]

    def historical_states():
        result = []
        for item in historical:
            size = len(parents)
            hq, hb, hv = np.zeros((size, 4)), np.zeros((size, 3)), np.zeros(size, bool)
            hq[:, 0] = 1
            old = len(item['leaf_q'])
            hq[:old], hb[:old], hv[:old] = item['leaf_q'], item['leaf_b'], item['leaf_valid']
            hrows, hp, _ = _rows_and_pivots(item['source'], owner, parents, item['observed'])
            fitted = refit_parent_world_from_children(item['source'], owner, parents, hp,
                hq, hb, item['observed'], hv, config)
            result.append(dict(**item, rows=hrows, pivots=hp, q=fitted['world_quaternion'],
                               b=fitted['world_offset'], valid=fitted['valid_slot']))
        return result

    def candidate_motion(branches, fit, source, rows, mask, q, b, pivots):
        parent = int(parents[branches[0]])
        affected = np.concatenate([rows[node][mask[rows[node]]] for node in branches])
        np_parent = np.concatenate((parents, [parent]))
        np_parent[branches] = len(parents)
        np_pivot = np.concatenate((pivots, source[affected].mean(0, keepdims=True)))
        nq, nb = np.concatenate((q, fit['quaternion'][None])), np.concatenate((b, fit['offset'][None]))
        nt, nr = world_to_local_motion(np_parent, np_pivot, nq, nb)
        return dict(parents=np_parent, pivots=np_pivot, q=nq, b=nb, t=nt, r=nr)

    def evaluate(branches, hstates):
        metrics['candidate_evaluations'] += 1
        branch_set = set(branches)
        _, depth = validate_motion_tree(parents, max_depth=config.max_depth)
        for slot in range(1, len(parents)):
            ancestor = slot
            while ancestor > 0 and ancestor not in branch_set:
                ancestor = int(parents[ancestor])
            if ancestor in branch_set and depth[slot] >= config.max_depth:
                metrics['rejection_counts']['depth'] += 1
                return None
        if any(not valid[node] or int(observed[rows[node]].sum()) < config.min_observed_gs for node in branches):
            metrics['rejection_counts']['current_support'] += 1
            return None
        if not _quantized_world_moves(source, rows, branches, q, b, observed, step_t, step_r):
            metrics['rejection_counts']['current_zero_world'] += 1
            return None
        fit, _ = _virtual_common_fit(source, rows, branches, q, b, observed, valid, config)
        if fit is None:
            metrics['rejection_counts']['current_virtual_fit'] += 1
            return None
        motion = candidate_motion(branches, fit, source, rows, observed, q, b, pivots)
        try:
            old_cost = _affected_cost(t, r, branches, step_t, step_r)
            new_cost = _affected_cost(motion['t'], motion['r'], branches + [len(parents)], step_t, step_r)
        except ValueError:
            metrics['rejection_counts']['overflow'] += 1
            return None
        gain = old_cost['integer_l1'] - new_cost['integer_l1']
        if old_cost['integer_l1'] <= 0 or gain / old_cost['integer_l1'] < config.min_quantized_reduction:
            metrics['rejection_counts']['current_quantized_gain'] += 1
            return None
        history_before, history_after = 0., 0.
        history_details = []
        for item in hstates:
            supported = [node for node in branches if item['valid'][node]]
            if len(supported) < 2:
                history_details.append(dict(frame=item['frame'], evidence='missing_supported_children'))
                continue
            if not _quantized_world_moves(item['source'], item['rows'], supported, item['q'], item['b'],
                                          item['observed'], step_t, step_r):
                history_details.append(dict(frame=item['frame'], evidence='observed_zero_world_motion'))
                continue
            hfit, reason = _virtual_common_fit(item['source'], item['rows'], supported,
                item['q'], item['b'], item['observed'], item['valid'], config)
            if hfit is None:
                history_details.append(dict(frame=item['frame'], evidence=reason))
                continue
            hm = candidate_motion(supported, hfit, item['source'], item['rows'], item['observed'],
                                  item['q'], item['b'], item['pivots'])
            ot, orot = world_to_local_motion(parents, item['pivots'], item['q'], item['b'])
            try:
                oc = _affected_cost(ot, orot, supported, step_t, step_r)
                nc = _affected_cost(hm['t'], hm['r'], supported + [len(parents)], step_t, step_r)
            except ValueError:
                metrics['rejection_counts']['overflow'] += 1
                return None
            if oc['integer_l1'] <= 0:
                history_details.append(dict(frame=item['frame'], evidence='zero_relative_cost'))
                continue
            history_before += oc['integer_l1']
            history_after += nc['integer_l1']
            history_details.append(dict(frame=item['frame'], evidence='supported_nonzero_common_component',
                child_slots=[int(node) for node in supported], integer_l1_before=oc['integer_l1'],
                integer_l1_after=nc['integer_l1'], virtual_rms=hfit['virtual_rms']))
        if history_before and (history_before - history_after) / history_before < config.min_quantized_reduction:
            metrics['rejection_counts']['history_quantized_gain'] += 1
            return None
        return dict(**motion, children=branches, fit=fit, gain=gain, before=old_cost, after=new_cost,
                    history=history_details, history_integer_l1_before=history_before,
                    history_integer_l1_after=history_after)

    if not metrics['update_skipped']:
        for _ in range(config.max_new_parents):
            rows, _, _ = _rows_and_pivots(source, owner, parents, observed)
            hstates = historical_states()
            best = None
            for parent in range(len(parents)):
                siblings = np.array([node for node in range(1, len(parents))
                                     if parents[node] == parent and len(rows[node])], np.int64)
                if len(siblings) < 2:
                    continue
                # Affine offsets alone depend on the arbitrary world origin.
                # Compare every sibling's displacement at the SAME observed
                # source anchor, which is invariant under a coordinate shift.
                parts = [rows[node][observed[rows[node]]] for node in siblings if valid[node]]
                support = np.concatenate(parts) if parts else np.empty(0, np.int64)
                anchor = source[support].mean(0) if len(support) else np.zeros(3)
                rotations = Rotation.from_quat(q[siblings][:, [1, 2, 3, 0]])
                displacement = np.einsum('nij,j->ni', rotations.as_matrix(), anchor) + b[siblings] - anchor
                features = np.concatenate((displacement / step_t, rotations.as_rotvec() / step_r), axis=1)
                k = min(len(siblings), config.neighbors + 1)
                _, nearest = cKDTree(features).query(features, k=k, workers=1)
                nearest = np.asarray(nearest).reshape(len(siblings), k)
                edges = sorted({tuple(sorted((int(siblings[i]), int(siblings[j]))))
                                for i in range(len(siblings)) for j in nearest[i] if i != j})
                lookup = {int(slot): i for i, slot in enumerate(siblings)}
                for a, child in edges:
                    candidate = evaluate([a, child], hstates)
                    if candidate is None:
                        continue
                    pool = set(siblings[nearest[lookup[a]]].tolist() + siblings[nearest[lookup[child]]].tolist())
                    pool.difference_update((a, child))
                    for extra in sorted(pool, key=lambda slot: int(ids[slot])):
                        if len(candidate['children']) >= config.max_children:
                            break
                        grown = evaluate(candidate['children'] + [extra], hstates)
                        if grown is not None and grown['gain'] > candidate['gain']:
                            candidate = grown
                    key = (-candidate['gain'], tuple(int(ids[slot]) for slot in candidate['children']))
                    if best is None or key < best[0]:
                        best = key, candidate
            if best is None:
                break
            chosen = best[1]
            children = chosen['children']
            metrics['inserted_parents'].append(dict(region_id=int(next_id), slot=len(parents),
                parent_region_id=int(ids[int(parents[children[0]])]),
                child_region_ids=[int(ids[node]) for node in children],
                integer_l1_before=chosen['before']['integer_l1'], integer_l1_after=chosen['after']['integer_l1'],
                quantized_reduction=chosen['gain'] / chosen['before']['integer_l1'],
                virtual_rms=chosen['fit']['virtual_rms'], fit_points=chosen['fit']['fit_points'],
                history=chosen['history'], history_integer_l1_before=chosen['history_integer_l1_before'],
                history_integer_l1_after=chosen['history_integer_l1_after']))
            parents, pivots, q, b, t, r = (chosen[key] for key in ('parents', 'pivots', 'q', 'b', 't', 'r'))
            ids, valid = np.concatenate((ids, [int(next_id)])), np.concatenate((valid, [True]))
            next_id += 1
    validate_motion_tree(parents, owner, config.max_depth)
    metrics['after'] = quantized_regional_cost(t, r, step_t, step_r)
    metrics['whole_packet_bytes_scope'] = 'caller quantizes, retargets GS and measures complete codec packet'
    return MotionTreeProposal(ids, parents, pivots, q, b, t, r, int(next_id), metrics)
