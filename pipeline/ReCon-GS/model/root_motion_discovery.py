"""Causal discovery of sustained local motion among currently independent GS.

This helper only proposes new leaves from the root. It never changes existing
leaf ownership, optimizers, motion parameters, history, or codec state. The
caller fits sharing, retargets the identical target, validates decoding and
allocates stable logical IDs before joint refinement.
"""

from dataclasses import dataclass, replace
import math

import numpy as np

from .online_motion_clustering import (CandidatePartition, _features, _ids,
                                       _statistical_partition)


@dataclass
class RootMotionDiscoveryConfig:
    enabled: bool = False
    min_motion_observations: int = 2
    min_region_size: int = 8
    max_candidates: int = 2
    spatial_scale_extent: float = .05

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('root discovery enabled must be boolean')
        for name in ('min_motion_observations', 'min_region_size', 'max_candidates'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'root discovery {name} must be a positive integer')
        if not 2 <= self.min_motion_observations <= 5 or self.min_region_size < 8 or self.max_candidates > 64:
            raise ValueError('root discovery requires 2..5 observations, 8+ GS and bounded candidates')
        if (isinstance(self.spatial_scale_extent, bool) or
                not math.isfinite(self.spatial_scale_extent) or self.spatial_scale_extent <= 0):
            raise ValueError('root discovery spatial_scale_extent must be finite and positive')


def propose_root_motion_regions(history, current_ids, root_mask, extent,
                                translation_step, clustering_config, config,
                                available_leaves=64, priorities=None):
    """Propose at most two local Kmeans groups per observed-history pattern.

    The latest transition must be observed and moving. Earlier missing
    observations hold evidence, whereas the most recent observed stationary
    transition resets it. Births never receive fabricated prior observations.
    Motion evidence uses rounded GS translation symbols and the established
    static_translation_extent threshold. Features retain complete observed
    displacement histories, plus scaled current source positions for locality.
    """
    config.validate()
    clustering_config.validate()
    ids = _ids(current_ids)
    root = np.asarray(root_mask)
    step = np.asarray(translation_step, np.float64)
    if (root.shape != ids.shape or root.dtype.kind != 'b' or step.shape != (3,) or
            not np.isfinite(step).all() or np.any(step <= 0) or
            not math.isfinite(extent) or extent <= 0):
        raise ValueError('root discovery needs aligned root masks, extent and GS translation steps')
    if isinstance(available_leaves, bool) or not isinstance(available_leaves, (int, np.integer)) or available_leaves < 0:
        raise ValueError('available_leaves must be a nonnegative integer')
    if priorities is not None:
        priorities = np.asarray(priorities, np.float64)
        if priorities.shape != ids.shape or not np.isfinite(priorities).all() or np.any(priorities < 0):
            raise ValueError('root discovery priorities must be finite nonnegative per-GS demand')
    if clustering_config.history_frames < config.min_motion_observations:
        raise ValueError('motion history window must support root discovery evidence length')
    canonical = np.argsort(ids, kind='stable')
    root = root[canonical]
    window = history.aligned(ids[canonical], clustering_config.history_frames)
    observed = window['observed']
    labels = np.full(len(ids), -3, np.int64)
    maximum = min(config.max_candidates, int(available_leaves))
    metrics = dict(enabled=config.enabled, method='root_local_kmeans',
        label_scope='currently root-owned GS only; other leaf ownership is untouched',
        history_frames=window['frames'], history_transitions=observed.shape[1],
        current_root_gs=int(root.sum()), min_motion_observations=config.min_motion_observations,
        motion_threshold_world=float(clustering_config.static_translation_extent * extent),
        motion_evidence='GS-step-rounded displacement; consecutive valid observations',
        missing_observations='hold evidence; observed stationary resets',
        available_leaves=int(available_leaves), max_candidates=maximum, patterns=[],
        current_observed_root_gs=0, current_moving_root_gs=0, sustained_moving_root_gs=0,
        insufficient_motion_evidence_gs=0, stationary_reset_gs=0,
        candidate_regions=0, proposed_gs=0, candidates_before_cap=0,
        rejected_small_groups=0, rejected_small_gs=0, capacity_dropped_groups=0,
        capacity_dropped_gs=0, fit_points=0, rejection_reason=None)
    metrics.update(generic_feature_distance_rejection_applied=False,
                   generic_feature_distance_would_reject_gs=0)

    def result(groups):
        restored = np.empty_like(labels)
        restored[canonical] = labels
        return CandidatePartition([np.sort(canonical[group]) for group in groups], restored, metrics)

    if not config.enabled:
        metrics['rejection_reason'] = 'discovery_disabled'
        return result([])
    if not observed.shape[1]:
        metrics['rejection_reason'] = 'no_observed_transitions'
        return result([])
    displacement = window['target'].astype(np.float64) - window['source'].astype(np.float64)
    integer = np.rint(displacement / step)
    moving = np.linalg.norm(integer * step, axis=2) > metrics['motion_threshold_world']
    current = root & observed[:, -1]
    latest_moving = current & moving[:, -1]
    labels[current & ~moving[:, -1]] = -2
    streak = latest_moving.astype(np.int64)
    stopped = ~latest_moving
    stationary_reset = np.zeros(len(ids), bool)
    for slot in range(observed.shape[1] - 2, -1, -1):
        valid = observed[:, slot] & ~stopped
        stationary_reset |= valid & ~moving[:, slot] & (streak < config.min_motion_observations)
        streak += valid & moving[:, slot]
        stopped |= valid & ~moving[:, slot]
    eligible = latest_moving & (streak >= config.min_motion_observations)
    labels[eligible] = -1
    metrics.update(current_observed_root_gs=int(current.sum()),
        current_moving_root_gs=int(latest_moving.sum()), sustained_moving_root_gs=int(eligible.sum()),
        insufficient_motion_evidence_gs=int((latest_moving & ~eligible).sum()),
        stationary_reset_gs=int((latest_moving & ~eligible & stationary_reset).sum()))
    selected = np.flatnonzero(eligible)
    if not len(selected):
        metrics['rejection_reason'] = 'no_sustained_root_motion'
        return result([])
    if maximum == 0:
        metrics['rejection_reason'] = 'no_available_leaves'
        return result([])
    patterns, assignments = np.unique(observed[selected], axis=0, return_inverse=True)
    fit_budget = max(2, clustering_config.max_fit_points // len(patterns))
    # Counts are bounded initialization choices, not separate training runs.
    fit_config = replace(clustering_config, method='kmeans', min_region_size=config.min_region_size,
                         component_counts=[1, 2] if maximum >= 2 else [1])
    demand = (np.abs(integer[:, -1]).sum(1) if priorities is None else priorities[canonical])
    proposals = []
    for pattern_index, pattern in enumerate(patterns):
        rows = selected[assignments == pattern_index]
        if len(rows) < config.min_region_size:
            metrics['rejected_small_groups'] += 1
            metrics['rejected_small_gs'] += len(rows)
            continue
        slots = np.flatnonzero(pattern)
        motion = _features(window, rows, slots, extent, fit_config)
        position = window['source'][rows, -1].astype(np.float64) / (config.spatial_scale_extent * extent)
        features = np.concatenate((motion, position), axis=1)
        # Sustained real motion is the eligibility gate. A generic Euclidean
        # distance in motion+position feature space is not shared-SE(3) evidence;
        # retain assignments and let the caller validate fitted motion/poses.
        proposed, detail = _statistical_partition(features, fit_config, fit_budget,
                                                   reject_feature_distance=False)
        metrics['patterns'].append(dict(observed_transitions=slots.tolist(), gs=len(rows),
            motion_feature_dimensions=int(motion.shape[1]), spatial_feature_dimensions=3,
            total_feature_dimensions=int(features.shape[1]), **detail))
        metrics['fit_points'] += detail['fit_points']
        metrics['generic_feature_distance_would_reject_gs'] += detail['feature_distance_rejected_points']
        for label in np.unique(proposed[proposed >= 0]):
            group = rows[proposed == label]
            if len(group) < config.min_region_size:
                metrics['rejected_small_groups'] += 1
                metrics['rejected_small_gs'] += len(group)
            else:
                proposals.append((float(demand[group].sum()), int(ids[canonical[group]].min()), group))
    proposals.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    metrics['candidates_before_cap'] = len(proposals)
    dropped = proposals[maximum:]
    metrics['capacity_dropped_groups'] = len(dropped)
    metrics['capacity_dropped_gs'] = sum(len(group) for _, _, group in dropped)
    kept = proposals[:maximum]
    groups = [group for _, _, group in kept]
    for label, group in enumerate(groups):
        labels[group] = label
    metrics.update(candidate_regions=len(groups), proposed_gs=sum(len(group) for group in groups),
                   candidate_quantized_demand=[float(demand) for demand, _, _ in kept])
    if not groups:
        metrics['rejection_reason'] = 'no_supported_local_candidates'
    return result(groups)
