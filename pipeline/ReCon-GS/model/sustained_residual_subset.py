"""Unactivated causal evidence for high-residual subsets within stable leaves.

This module only identifies supported member rows. It does not cluster, change
ownership, recycle regions, refit poses or claim an integer cost is byte cost.
The caller supplies its existing per-GS quantized residual threshold.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class SustainedResidualSubsetConfig:
    required_observations: int = 3
    min_region_size: int = 8

    def validate(self):
        for name in ('required_observations', 'min_region_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass
class SustainedResidualSubsetProposal:
    # One current-row mask; groups split those rows by stable logical leaf ID.
    candidate_mask: np.ndarray
    groups: list
    streaks: np.ndarray
    metrics: dict


def _identity_rows(gs_ids, leaf_ids):
    ids, leaves = np.asarray(gs_ids), np.asarray(leaf_ids)
    if (ids.ndim != 1 or leaves.shape != ids.shape or ids.dtype.kind not in 'iu' or
            leaves.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(leaves < 0) or
            len(np.unique(ids)) != len(ids) or
            np.any(ids > np.iinfo(np.int64).max) or np.any(leaves > np.iinfo(np.int64).max)):
        raise ValueError('GS IDs must be unique nonnegative integers and logical leaves must align')
    return ids.astype(np.int64), leaves.astype(np.int64)


def _motion_inputs(gs_ids, leaf_ids, quantized_residuals, observed, high_units):
    if isinstance(high_units, bool) or not math.isfinite(high_units) or high_units <= 0:
        raise ValueError('the high residual threshold must be finite and positive')
    ids, leaves = _identity_rows(gs_ids, leaf_ids)
    symbols, mask = np.asarray(quantized_residuals), np.asarray(observed)
    if (symbols.shape != (len(ids), 6) or symbols.dtype.kind != 'i' or
            mask.shape != ids.shape or mask.dtype.kind != 'b' or
            np.any(symbols < -32768) or np.any(symbols > 32767)):
        raise ValueError('residuals must be signed int16-range symbols with a boolean observed mask')
    wide = symbols.astype(np.int64)
    block_norm_squared = np.maximum(np.square(wide[:, :3]).sum(-1), np.square(wide[:, 3:]).sum(-1))
    high = mask & (leaves > 0) & (block_norm_squared > high_units ** 2)
    return ids, leaves, symbols, mask, high


class SustainedResidualSubsetEvidence:
    """Tracks consecutive valid high evidence for each (GS ID, logical leaf).

    Missing observations hold evidence, while observed low values reset it.
    Migration resets evidence even when the migrating row is currently hidden.
    Removed GS disappear from state, so a newly introduced row cannot inherit
    fabricated observations. Current missing rows are never proposed.
    """

    def __init__(self, config=None):
        self.config = SustainedResidualSubsetConfig() if config is None else config
        self.config.validate()
        self.last_frame = None
        self.ids = np.empty(0, np.int64)
        self.leaves = np.empty(0, np.int64)
        self.counts = np.empty(0, np.int64)

    def observe(self, frame_index, gs_ids, leaf_ids, quantized_residuals, observed,
                high_units):
        """Consume one actual frame and return high subsets in current row order.

        ``quantized_residuals`` contains signed integer translation XYZ and
        rotation XYZ symbols, each already in its own quantization units.
        ``high_units`` tests max(L2(translation symbols), L2(rotation symbols)).
        This detects local mismatch, while integer L1 remains a cost diagnostic.
        No frame padding, float-motion nonzero check or future history is used.
        """
        self.config.validate()
        if (isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or
                frame_index < 1 or (self.last_frame is not None and frame_index <= self.last_frame)):
            raise ValueError('frame indices must be actual strictly increasing positive frames')
        ids, leaves, symbols, mask, high = _motion_inputs(gs_ids, leaf_ids, quantized_residuals, observed, high_units)
        locations = np.searchsorted(self.ids, ids)
        matched = locations < len(self.ids)
        matched[matched] &= self.ids[locations[matched]] == ids[matched]
        same_leaf = np.zeros(len(ids), bool)
        same_leaf[matched] = self.leaves[locations[matched]] == leaves[matched]
        streaks = np.zeros(len(ids), np.int64)
        streaks[same_leaf] = self.counts[locations[same_leaf]]
        # Observed evidence is explicit. Missing symbols, including placeholder
        # zeros, are not stationary evidence and do not advance the streak.
        streaks[mask & ~high] = 0
        streaks[high] += 1
        streaks[leaves == 0] = 0
        candidate_mask, groups, rejected, eligible = self._groups(leaves, symbols, mask, high, streaks)
        order = np.argsort(ids, kind='stable')
        self.ids, self.leaves, self.counts = (value[order].copy() for value in (ids, leaves, streaks))
        self.last_frame = int(frame_index)
        metrics = dict(frame=int(frame_index), gs_count=len(ids), observed_gs=int(mask.sum()),
            current_high_gs=int(high.sum()), sustained_high_gs=int(eligible.sum()),
            candidate_gs=int(candidate_mask.sum()), candidate_regions=len(groups),
            new_gs=int((~matched).sum()), migrated_gs=int((matched & ~same_leaf).sum()),
            missing_held_gs=int((~mask & same_leaf & (leaves > 0) & (streaks > 0)).sum()),
            required_valid_observations=self.config.required_observations,
            high_units=float(high_units), cost_scope='quantized integer L1 and active blocks, not encoded bytes',
            threshold_scope='max translation/rotation block L2 in their own GS quantization units',
            evidence_scope='stable GS ID and stable logical leaf ID, observed high only; missing holds',
            region_rejections=rejected, current_frame_counted=True,
            spatial_coherence_checked=False, ownership_or_pose_modified=False)
        return SustainedResidualSubsetProposal(candidate_mask, groups, streaks, metrics)

    def _groups(self, leaves, symbols, mask, high, streaks):
        integer_cost = np.abs(symbols.astype(np.int64)).sum(-1)
        eligible = high & (streaks >= self.config.required_observations)
        candidate_mask = np.zeros(len(leaves), bool)
        groups, rejected = [], []
        for leaf in np.unique(leaves[eligible]):
            members = np.flatnonzero(leaves == leaf)
            selected = members[eligible[members]]
            retained = len(members) - len(selected)
            if len(selected) < self.config.min_region_size or retained < self.config.min_region_size:
                rejected.append(dict(region_id=int(leaf), sustained_gs=len(selected), retained_gs=retained,
                    reason='insufficient_subset_support' if len(selected) < self.config.min_region_size
                    else 'insufficient_retained_members'))
                continue
            candidate_mask[selected] = True
            groups.append(dict(region_id=int(leaf), rows=selected.copy(),
                integer_l1=int(integer_cost[selected].sum()),
                observed_region_integer_l1=int(integer_cost[members[mask[members]]].sum()),
                active_translation_blocks=int(np.any(symbols[selected, :3] != 0, axis=1).sum()),
                active_rotation_blocks=int(np.any(symbols[selected, 3:] != 0, axis=1).sum()),
                member_count=len(members), retained_count=retained))
        return candidate_mask, groups, rejected, eligible

    def peek(self, frame_index, gs_ids, leaf_ids, quantized_residuals, observed, high_units):
        """Read past evidence and recheck current demand without consuming a frame."""
        self.config.validate()
        if (isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or
                frame_index < 1 or (self.last_frame is not None and frame_index <= self.last_frame)):
            raise ValueError('peek must inspect a frame later than the stored final evidence')
        ids, leaves, symbols, mask, high = _motion_inputs(gs_ids, leaf_ids, quantized_residuals, observed, high_units)
        locations = np.searchsorted(self.ids, ids)
        matched = locations < len(self.ids)
        matched[matched] &= self.ids[locations[matched]] == ids[matched]
        same_leaf = np.zeros(len(ids), bool)
        same_leaf[matched] = self.leaves[locations[matched]] == leaves[matched]
        streaks = np.zeros(len(ids), np.int64)
        streaks[same_leaf] = self.counts[locations[same_leaf]]
        candidate_mask, groups, rejected, eligible = self._groups(leaves, symbols, mask, high, streaks)
        metrics = dict(frame=int(frame_index), evidence_last_frame=self.last_frame, current_frame_counted=False,
            gs_count=len(ids), observed_gs=int(mask.sum()), current_high_gs=int(high.sum()),
            sustained_high_gs=int(eligible.sum()), candidate_gs=int(candidate_mask.sum()),
            candidate_regions=len(groups), required_valid_observations=self.config.required_observations,
            unmatched_or_changed_leaf_gs=int((~same_leaf).sum()), high_units=float(high_units),
            threshold_scope='max translation/rotation block L2 in their own GS quantization units',
            cost_scope='current quantized integer L1 and active blocks; past streak only, not bytes',
            region_rejections=rejected, spatial_coherence_checked=False, ownership_or_pose_modified=False)
        return SustainedResidualSubsetProposal(candidate_mask, groups, streaks, metrics)

    def state_dict(self):
        return dict(last_frame=self.last_frame, ids=self.ids.copy(),
                    leaves=self.leaves.copy(), counts=self.counts.copy())

    def load_state_dict(self, state):
        ids, leaves = _identity_rows(state['ids'], state['leaves'])
        counts = np.asarray(state['counts'])
        frame = state['last_frame']
        if (counts.shape != ids.shape or counts.dtype.kind not in 'iu' or np.any(counts < 0) or
                np.any(counts > np.iinfo(np.int64).max) or
                (frame is not None and (isinstance(frame, bool) or not isinstance(frame, int) or frame < 1)) or
                (frame is None and len(ids)) or np.any(counts[leaves == 0] != 0)):
            raise ValueError('invalid causal residual subset checkpoint')
        order = np.argsort(ids, kind='stable')
        self.ids, self.leaves, self.counts = ids[order].copy(), leaves[order].copy(), counts[order].astype(np.int64).copy()
        self.last_frame = frame


def propose_local_residual_groups(history, ids, source_xyz, selected_rows, extent,
                                  clustering_config, spatial_scale_extent,
                                  min_region_size, max_groups=2):
    """Bounded local Kmeans proposals inside an already qualified high subset.

    The caller has validated stable leaf ownership and sustained evidence. This
    function adds no motion threshold or generic feature-distance rejection.
    Missing-history masks are kept as separate buckets and small buckets remain
    in the original leaf, rather than padding motion evidence with zero values.
    """
    from dataclasses import replace
    from .online_motion_clustering import _features, _statistical_partition

    ids, _ = _identity_rows(ids, np.zeros(len(ids), np.int64))
    source = np.asarray(source_xyz)
    rows = np.asarray(selected_rows)
    if (source.shape != (len(ids), 3) or rows.ndim != 1 or rows.dtype.kind not in 'iu' or
            len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(ids)) or
            not np.isfinite(source[rows]).all() or not math.isfinite(extent) or extent <= 0 or
            not math.isfinite(spatial_scale_extent) or spatial_scale_extent <= 0 or
            isinstance(min_region_size, bool) or not isinstance(min_region_size, int) or min_region_size < 1 or
            isinstance(max_groups, bool) or not isinstance(max_groups, int) or not 1 <= max_groups <= 2):
        raise ValueError('local grouping needs aligned positions, unique selected rows and bounded group support')
    window = history.aligned(ids)
    masks = window['observed'][rows]
    if masks.shape[1] == 0:
        return [], dict(selected_gs=len(rows), proposed_gs=0, groups=0, reason='no_observed_history')
    order = np.argsort(ids[rows], kind='stable')
    rows, masks = rows[order], masks[order]
    cfg = replace(clustering_config, method='kmeans', component_counts=list(range(1, max_groups + 1)),
                  min_region_size=min_region_size)
    pattern, inverse = np.unique(masks, axis=0, return_inverse=True)
    candidates, buckets = [], []
    for slot, observed in enumerate(pattern):
        part = rows[inverse == slot]
        positions = np.flatnonzero(observed)
        if not len(positions) or len(part) < min_region_size:
            buckets.append(dict(gs_count=len(part), observed_transitions=len(positions), proposed_gs=0,
                                reason='missing_or_small_support'))
            continue
        values = _features(window, part, positions, extent, cfg)
        spatial = source[part].astype(np.float64) / (spatial_scale_extent * extent)
        spatial -= spatial.mean(0)  # Coordinate origin does not affect grouping.
        values = np.concatenate((values, spatial), axis=1)
        labels, details = _statistical_partition(values, cfg, cfg.max_fit_points,
                                                reject_feature_distance=False)
        local = [part[labels == label] for label in np.unique(labels) if label >= 0]
        candidates.extend(group for group in local if len(group) >= min_region_size)
        buckets.append(dict(gs_count=len(part), observed_transitions=len(positions),
            feature_dimensions=values.shape[1], proposed_gs=sum(len(group) for group in local), fit=details))
    # Bounded global admission across mask buckets. No point outside the supplied
    # subset can enter a group, and excluded groups stay with their existing leaf.
    candidates.sort(key=lambda group: (-len(group), int(ids[group].min())))
    groups = candidates[:max_groups]
    return groups, dict(selected_gs=len(rows), proposed_gs=sum(len(group) for group in groups),
        groups=len(groups), mask_buckets=buckets, max_groups=max_groups,
        small_or_capacity_excluded_gs=len(rows) - sum(len(group) for group in groups),
        grouping_scope='selected rows only, full observed position displacement history plus current source position',
        generic_feature_distance_gate=False)
