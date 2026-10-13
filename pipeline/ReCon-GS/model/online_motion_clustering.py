"""Causal, flat motion partitions. Proposals never change GS motion or bytes.

Stable GS IDs, not historical region IDs, align the last observed transitions.
Statistical methods use the same displacement histories. Different observation
patterns are fitted separately, so an absent transition is never zero motion.
Covariance quaternion changes are optional evidence, not physical rotations by
default. The caller must apply its common quantization/storage acceptance gate.
"""

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class OnlineMotionClusteringConfig:
    method: str = 'greedy'
    history_frames: int = 5
    update_interval: int = 1
    min_region_size: int = 8
    min_observations: int = 1
    translation_scale_extent: float = 4e-4
    static_translation_extent: float = 4e-4
    rotation_scale_rad: float = .004
    rotation_feature_weight: float = 0.
    orientation_fit_weight: float = 0.
    orientation_length_extent: float = .01
    component_counts: list = field(default_factory=lambda: [1, 2, 4, 8])
    max_fit_points: int = 8192
    max_iter: int = 50
    assignment_chunk_size: int = 4096
    max_feature_distance: float = 4.
    random_seed: int = 0
    gmm_min_probability: float = .55
    gmm_reg_covar: float = 1e-4
    hdbscan_min_samples: int = 8
    hdbscan_allow_single_cluster: bool = True
    greedy_seed_size: int = 32
    greedy_neighbors: int = 8
    greedy_max_regions: int = 64
    greedy_fit_tolerance_extent: float = 4e-4

    def validate(self):
        if self.method not in ('kmeans', 'gmm', 'hdbscan', 'greedy'):
            raise ValueError('online clustering method must be kmeans/gmm/hdbscan/greedy')
        for name in ('history_frames', 'update_interval', 'min_region_size',
                     'min_observations', 'max_fit_points', 'max_iter',
                     'assignment_chunk_size', 'hdbscan_min_samples',
                     'greedy_seed_size', 'greedy_neighbors', 'greedy_max_regions'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.history_frames > 5 or self.min_observations > self.history_frames:
            raise ValueError('history uses at most five actual transitions')
        if self.min_region_size < 2 or self.max_fit_points < 2:
            raise ValueError('region size and fit sample size must be at least two')
        if (not self.component_counts or len(self.component_counts) > 8 or
                list(self.component_counts) != sorted(set(self.component_counts)) or
                any(isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 64
                    for k in self.component_counts)):
            raise ValueError('component_counts must have at most eight increasing values in [1,64]')
        for name in ('translation_scale_extent', 'static_translation_extent',
                     'rotation_scale_rad', 'orientation_length_extent',
                     'max_feature_distance', 'gmm_reg_covar',
                     'greedy_fit_tolerance_extent'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('rotation_feature_weight', 'orientation_fit_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 <= self.gmm_min_probability <= 1:
            raise ValueError('gmm_min_probability must lie in [0,1]')
        if (isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int)
                or self.random_seed < 0 or
                not isinstance(self.hdbscan_allow_single_cluster, bool)):
            raise ValueError('invalid random seed or HDBSCAN single-cluster flag')


@dataclass
class CandidatePartition:
    groups: list
    labels: np.ndarray
    metrics: dict


def _ids(values):
    result = np.asarray(values)
    if (result.ndim != 1 or result.dtype.kind not in 'iu' or
            len(np.unique(result)) != len(result)):
        raise ValueError('GS IDs must be a unique integer vector')
    return result


class OnlineMotionHistory:
    """Stores only completed transitions; append once after each free fit.

    IDs must not be reused after deletion. New GS have no evidence before birth.
    ``observed`` is a boolean visibility/support mask, not an observation count.
    ``relative_quaternion`` is scalar first and describes the complete update.
    Inputs are copied, so later optimizer mutations cannot change past evidence.
    """

    def __init__(self, max_transitions=5):
        if (isinstance(max_transitions, bool) or not isinstance(max_transitions, int)
                or not 1 <= max_transitions <= 5):
            raise ValueError('max_transitions must lie in [1,5]')
        self.max_transitions = max_transitions
        self.records = []

    def append(self, frame_index, ids, source_xyz, target_xyz,
               relative_quaternion=None, observed=None):
        if (isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer))
                or frame_index < 1 or
                (self.records and frame_index <= self.records[-1]['frame_index'])):
            raise ValueError('transition target frames must be strictly increasing and positive')
        ids = _ids(ids)
        n = len(ids)
        source, target = (np.asarray(v, dtype=np.float64) for v in (source_xyz, target_xyz))
        mask = np.ones(n, dtype=bool) if observed is None else np.asarray(observed)
        quaternion = np.zeros((n, 4)) if relative_quaternion is None else np.asarray(
            relative_quaternion, dtype=np.float64)
        if relative_quaternion is None:
            quaternion[:, 0] = 1
        if (source.shape != (n, 3) or target.shape != source.shape or
                quaternion.shape != (n, 4) or mask.shape != (n,) or mask.dtype.kind != 'b' or
                not all(np.isfinite(v[mask]).all() for v in (source, target, quaternion)) or
                np.any(np.linalg.norm(quaternion[mask], axis=1) == 0)):
            raise ValueError('invalid observed complete-motion transition')
        order = np.argsort(ids, kind='stable')
        # Invalid/missing data are masked; placeholders are never consumed by fits.
        source, target, quaternion = (np.where(mask[:, None], v, 0.)
                                      for v in (source, target, quaternion))
        record = dict(frame_index=int(frame_index), ids=ids[order].copy(),
                      source=source[order].astype(np.float32),
                      target=target[order].astype(np.float32),
                      quaternion=quaternion[order].astype(np.float32),
                      observed=mask[order].copy())
        self.records.append(record)
        self.records = self.records[-self.max_transitions:]

    def aligned(self, current_ids, window=None):
        ids = _ids(current_ids)
        records = self.records if window is None else self.records[-window:]
        n, t = len(ids), len(records)
        source, target = np.zeros((n, t, 3), np.float32), np.zeros((n, t, 3), np.float32)
        quaternion = np.zeros((n, t, 4), np.float32)
        quaternion[..., 0] = 1
        observed = np.zeros((n, t), dtype=bool)
        for slot, record in enumerate(records):
            stored = record['ids']
            if not len(stored):
                continue
            locations = np.searchsorted(stored, ids)
            valid = locations < len(stored)
            valid[valid] &= stored[locations[valid]] == ids[valid]
            rows, locations = np.flatnonzero(valid), locations[valid]
            source[rows, slot] = record['source'][locations]
            target[rows, slot] = record['target'][locations]
            quaternion[rows, slot] = record['quaternion'][locations]
            observed[rows, slot] = record['observed'][locations]
        return dict(source=source, target=target, quaternion=quaternion,
                    observed=observed,
                    frames=[r['frame_index'] for r in records])

    def state_dict(self):
        return dict(max_transitions=self.max_transitions, records=[{
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in record.items()} for record in self.records])

    def load_state_dict(self, state):
        records = state.get('records', [])
        if len(records) > self.max_transitions:
            raise ValueError('checkpoint history exceeds configured transition window')
        replacement = OnlineMotionHistory(self.max_transitions)
        for record in records:
            replacement.append(record['frame_index'], record['ids'], record['source'],
                               record['target'], record['quaternion'], record['observed'])
        self.records = replacement.records


def _features(window, rows, slots, extent, config):
    displacement = (window['target'][rows][:, slots].astype(np.float64) -
                    window['source'][rows][:, slots].astype(np.float64))
    features = displacement / (config.translation_scale_extent * extent)
    if config.rotation_feature_weight > 0:
        q = window['quaternion'][rows][:, slots].reshape(-1, 4)
        rotvec = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(rows), -1, 3)
        features = np.concatenate((features, rotvec * config.rotation_feature_weight /
                                   config.rotation_scale_rad), axis=2)
    return features.reshape(len(rows), -1) / math.sqrt(len(slots))


def _statistical_partition(values, config, fit_budget):
    """Bounded deterministic model selection followed by chunked rejection."""
    from threadpoolctl import threadpool_limits

    random = np.random.default_rng(config.random_seed)
    fit = values[np.sort(random.choice(len(values), min(len(values), fit_budget), replace=False))]
    counts = [k for k in config.component_counts if
              k <= min(len(fit), len(values) // config.min_region_size,
                       len(np.unique(fit, axis=0)))]
    if not counts or len(fit) < 2:
        return np.full(len(values), -1, np.int64), dict(fit_points=len(fit), components=0)
    best, best_score = None, math.inf
    with threadpool_limits(limits=1):
        for k in counts:
            if config.method == 'kmeans':
                from sklearn.cluster import KMeans

                model = KMeans(n_clusters=k, random_state=config.random_seed,
                               n_init=1, max_iter=config.max_iter).fit(fit)
                # Isotropic Gaussian BIC, with a noise floor in normalized units.
                variance = max(model.inertia_ / fit.size, config.gmm_reg_covar)
                score = fit.size * math.log(variance) + k * (fit.shape[1] + 1) * math.log(len(fit))
            else:
                from sklearn.mixture import GaussianMixture

                model = GaussianMixture(n_components=k, covariance_type='diag',
                    random_state=config.random_seed, n_init=1, max_iter=config.max_iter,
                    reg_covar=config.gmm_reg_covar).fit(fit)
                score = model.bic(fit)
            if math.isfinite(score) and score < best_score:
                best, best_score = model, float(score)
        predicted = np.full(len(values), -1, np.int64)
        if best is None:
            return predicted, dict(fit_points=len(fit), components=0)
        centers = best.cluster_centers_ if config.method == 'kmeans' else best.means_
        for start in range(0, len(values), config.assignment_chunk_size):
            chunk = values[start:start + config.assignment_chunk_size]
            if config.method == 'kmeans':
                label = best.predict(chunk)
                accepted = np.ones(len(chunk), dtype=bool)
            else:
                probability = best.predict_proba(chunk)
                label = probability.argmax(1)
                accepted = probability.max(1) >= config.gmm_min_probability
            accepted &= np.linalg.norm(chunk - centers[label], axis=1) <= config.max_feature_distance
            predicted[start:start + len(chunk)] = np.where(accepted, label, -1)
    return predicted, dict(fit_points=len(fit), components=len(centers), selection_score=best_score)


def _rigid_statistics(source, target, rotations=None):
    # Per-transition sufficient statistics make each seed-union fit O(window),
    # including a giant coherent region. No GS pairwise cost matrix is built.
    result = dict(n=len(source), x=source.sum(0), y=target.sum(0),
                  xy=np.einsum('nti,ntj->tij', target, source),
                  xx=np.einsum('nti,nti->t', source, source),
                  yy=np.einsum('nti,nti->t', target, target))
    if rotations is not None:
        result['rotation'] = rotations.sum(0)
    return result


def _union_statistics(left, right):
    return {key: left[key] + right[key] for key in left}


def _rigid_transform(stats, orientation_regularizer=0.):
    n = stats['n']
    covariance = stats['xy'] - np.einsum('ti,tj->tij', stats['y'], stats['x']) / n
    fit_covariance = covariance.copy()
    if orientation_regularizer:
        fit_covariance += orientation_regularizer * stats['rotation']
    left, _, right = np.linalg.svd(fit_covariance)
    correction = np.ones((len(covariance), 3))
    correction[:, -1] = np.where(np.linalg.det(left @ right) >= 0, 1., -1.)
    matrix = (left * correction[:, None, :]) @ right
    offset = stats['y'] / n - np.einsum('tij,tj->ti', matrix, stats['x'] / n)
    return matrix, offset, covariance


def _rigid_error(stats, orientation_regularizer=0.):
    n = stats['n']
    matrix, _, covariance = _rigid_transform(stats, orientation_regularizer)
    energy = stats['xx'] + stats['yy'] - (
        np.square(stats['x']).sum(1) + np.square(stats['y']).sum(1)) / n
    sse = energy - 2 * np.einsum('tij,tij->t', matrix, covariance)
    return float(np.sqrt(np.maximum(sse, 0.) / n).max(initial=0.))


def _greedy_partition(window, rows, slots, values, extent, config):
    source = window['source'][rows][:, slots].astype(np.float64)
    target = window['target'][rows][:, slots].astype(np.float64)
    rotations = None
    regularizer = config.orientation_fit_weight * (config.orientation_length_extent * extent) ** 2
    if regularizer:
        q = window['quaternion'][rows][:, slots].reshape(-1, 4)
        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix().reshape(len(rows), len(slots), 3, 3)
    tolerance = config.greedy_fit_tolerance_extent * extent
    cells = np.rint(values)
    # Float cells avoid overflow for exceptionally large normalized motion.
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    bins = np.split(order, np.cumsum(np.bincount(inverse))[:-1])
    seeds, statistics = [], []
    for bucket in bins:
        pending = [bucket[start:start + config.greedy_seed_size]
                   for start in range(0, len(bucket), config.greedy_seed_size)]
        while pending:
            part = pending.pop()
            stats = _rigid_statistics(source[part], target[part],
                                      None if rotations is None else rotations[part])
            if len(part) > 1 and _rigid_error(stats, regularizer) > tolerance:
                axis = int(np.ptp(values[part], axis=0).argmax())
                ranked = part[np.argsort(values[part, axis], kind='stable')]
                half = len(ranked) // 2
                pending.extend((ranked[half:], ranked[:half]))
            else:
                seeds.append(part)
                statistics.append(stats)
    centers = np.stack([values[part].mean(0) for part in seeds])
    count = len(seeds)
    k = min(count, config.greedy_neighbors + 1)
    _, nearest = cKDTree(centers).query(centers, k=k, workers=1)
    nearest = np.asarray(nearest).reshape(count, k)
    edges = sorted({(min(i, int(j)), max(i, int(j))) for i in range(count)
                    for j in nearest[i] if i != j})
    ranked_edges = sorted((_rigid_error(_union_statistics(statistics[a], statistics[b]), regularizer),
                           a, b) for a, b in edges)
    parent = np.arange(count)

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    merges = 0
    for _, a, b in ranked_edges:
        a, b = root(a), root(b)
        if a == b:
            continue
        union = _union_statistics(statistics[a], statistics[b])
        if _rigid_error(union, regularizer) <= tolerance:
            a, b = min(a, b), max(a, b)
            parent[b], statistics[a] = a, union
            merges += 1
    labels = np.full(len(rows), -1, np.int64)
    for seed, part in enumerate(seeds):
        labels[part] = root(seed)
    return labels, dict(seed_count=count, neighbor_edges=len(edges), merges=merges)


def _bounded_greedy_partition(window, rows, slots, values, extent, config, fit_budget):
    """Small-seed graph on a bounded sample; chunked full-motion assignment.

    Rebuilds from scratch every proposal. Unsupported/sample-missed motions keep
    independent GS residuals. Assignment checks actual observed displacement
    under fitted shared rotations, rather than nearest displacement centroid.
    """
    random = np.random.default_rng(config.random_seed)
    sampled = np.sort(random.choice(len(rows), min(len(rows), fit_budget), replace=False))
    seed_labels, detail = _greedy_partition(window, rows[sampled], slots, values[sampled], extent, config)
    valid = seed_labels >= 0
    order = np.flatnonzero(valid)[np.argsort(seed_labels[valid], kind='stable')]
    parts = (np.split(order, np.flatnonzero(np.diff(seed_labels[order])) + 1) if len(order) else [])
    parts = [sampled[part] for part in parts if len(part) >= config.min_region_size]
    parts.sort(key=lambda part: (-len(part), int(part.min())))
    parts = parts[:config.greedy_max_regions]
    output = np.full(len(rows), -1, np.int64)
    best_error = np.full(len(rows), math.inf)
    regularizer = config.orientation_fit_weight * (config.orientation_length_extent * extent) ** 2
    for index, part in enumerate(parts):
        source = window['source'][rows[part]][:, slots].astype(np.float64)
        target = window['target'][rows[part]][:, slots].astype(np.float64)
        rotations = None
        if regularizer:
            q = window['quaternion'][rows[part]][:, slots].reshape(-1, 4)
            rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix().reshape(len(part), len(slots), 3, 3)
        matrix, offset, _ = _rigid_transform(_rigid_statistics(source, target, rotations), regularizer)
        for start in range(0, len(rows), config.assignment_chunk_size):
            end = min(start + config.assignment_chunk_size, len(rows))
            source = window['source'][rows[start:end]][:, slots].astype(np.float64)
            target = window['target'][rows[start:end]][:, slots].astype(np.float64)
            error = np.linalg.norm(np.einsum('tij,ntj->nti', matrix, source) + offset - target,
                                   axis=2).max(1)
            accepted = ((error <= config.greedy_fit_tolerance_extent * extent) &
                        (error < best_error[start:end]))
            output[start:end][accepted] = index
            best_error[start:end][accepted] = error[accepted]
    detail.update(fit_points=len(sampled), sampled_regions=len(parts))
    return output, detail


def propose_motion_regions(history, current_ids, extent, config):
    """Fresh partition from completed motions; previous groups cannot lock splits.

    Labels are -3 unknown, -2 near-static centers, -1 unsupported/noise and
    nonnegative proposed regions. Noise/unobserved are never forced into groups.
    Output rows index current_ids. No shared-motion subtraction happens here.
    """
    config.validate()
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError('extent must be finite and positive')
    ids = _ids(current_ids)
    canonical = np.argsort(ids, kind='stable')
    window = history.aligned(ids[canonical], config.history_frames)
    mask = window['observed']
    labels = np.full(len(current_ids), -3, np.int64)
    metrics = dict(method=config.method, history_transitions=mask.shape[1],
                   history_frames=window['frames'], patterns=[])
    if not mask.shape[1] or not len(labels):
        metrics.update(static_gs=0, unknown_gs=len(labels), noise_gs=0, candidate_regions=0)
        return CandidatePartition([], labels, metrics)
    eligible = (mask.sum(1) >= config.min_observations) & mask[:, -1]
    displacement = window['target'].astype(np.float64) - window['source'].astype(np.float64)
    moving = np.linalg.norm(displacement, axis=2) > config.static_translation_extent * extent
    if config.rotation_feature_weight > 0:
        selected = np.flatnonzero(mask.reshape(-1))
        q = window['quaternion'].reshape(-1, 4)[selected]
        angle = np.zeros(mask.size)
        angle[selected] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).magnitude()
        moving |= angle.reshape(mask.shape) * config.rotation_feature_weight > config.rotation_scale_rad
    static = eligible & ~np.any(moving & mask, axis=1)
    labels[static] = -2
    selected = np.flatnonzero(eligible & ~static)
    labels[selected] = -1
    if len(selected):
        patterns, assignment = np.unique(mask[selected], axis=0, return_inverse=True)
        fit_budget = max(2, config.max_fit_points // len(patterns))
        groups = []
        for pattern_index, pattern in enumerate(patterns):
            rows = selected[assignment == pattern_index]
            if len(rows) < config.min_region_size:
                continue
            slots = np.flatnonzero(pattern)
            values = _features(window, rows, slots, extent, config)
            if config.method in ('kmeans', 'gmm'):
                proposed, detail = _statistical_partition(values, config, fit_budget)
            elif config.method == 'hdbscan':
                from sklearn.cluster import HDBSCAN

                if len(rows) < max(config.min_region_size, config.hdbscan_min_samples):
                    continue
                proposed = HDBSCAN(min_cluster_size=config.min_region_size,
                    min_samples=config.hdbscan_min_samples,
                    allow_single_cluster=config.hdbscan_allow_single_cluster,
                    algorithm='kd_tree', n_jobs=1, copy=True).fit_predict(values)
                detail = dict(fit_points=len(values))
            else:
                proposed, detail = _bounded_greedy_partition(window, rows, slots, values, extent, config, fit_budget)
            metrics['patterns'].append(dict(observed_transitions=slots.tolist(), gs=len(rows), **detail))
            valid = proposed >= 0
            if np.any(valid):
                order = np.flatnonzero(valid)[np.argsort(proposed[valid], kind='stable')]
                boundaries = np.flatnonzero(np.diff(proposed[order])) + 1
                groups.extend(rows[part] for part in np.split(order, boundaries)
                              if len(part) >= config.min_region_size)
        groups.sort(key=lambda part: int(part.min()))
        for index, rows in enumerate(groups):
            labels[rows] = index
    else:
        groups = []
    metrics.update(static_gs=int(np.sum(labels == -2)), unknown_gs=int(np.sum(labels == -3)),
                   noise_gs=int(np.sum(labels == -1)), candidate_regions=len(groups))
    restored = np.empty_like(labels)
    restored[canonical] = labels
    return CandidatePartition([np.sort(canonical[part]) for part in groups], restored, metrics)
