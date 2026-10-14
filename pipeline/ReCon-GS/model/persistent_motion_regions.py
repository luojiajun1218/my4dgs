"""Persistent flat regions with stable logical IDs and causal edit evidence.

Packet slots remain dense. Logical IDs and GS IDs are never reused, and every
representation change is fitted against one unchanged current-frame target.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .sustained_residual_subset import (SustainedResidualSubsetEvidence,
                                       SustainedResidualSubsetConfig)


@dataclass
class PersistentRegionConfig:
    enabled: bool = False
    tree_enabled: bool = False
    structure_interval: int = 10
    max_leaves: int = 64
    min_region_size: int = 8
    migration_gain: float = .25
    migration_frames: int = 2
    migration_neighbors: int = 8
    split_p90_units: float = 4.
    split_frames: int = 3
    idle_frames: int = 3
    merge_neighbors: int = 8
    max_merge_trials: int = 8
    max_split_trials: int = 4
    local_maintenance_enabled: bool = False
    local_subset_high_units: float = 4.
    local_subset_frames: int = 3
    local_subset_max_new_leaves: int = 2
    local_subset_spatial_scale_extent: float = .05
    world_zero_frames: int = 3

    def validate(self):
        if (not isinstance(self.enabled, bool) or not isinstance(self.tree_enabled, bool) or
                not isinstance(self.local_maintenance_enabled, bool)):
            raise ValueError('Persistent enable flags must be boolean')
        if self.tree_enabled and not self.enabled:
            raise ValueError('Motion trees require persistent regions')
        if self.local_maintenance_enabled and not self.enabled:
            raise ValueError('Local region maintenance requires persistent regions')
        for name in ('structure_interval', 'max_leaves', 'min_region_size',
                     'migration_frames', 'migration_neighbors', 'split_frames',
                     'idle_frames', 'merge_neighbors', 'max_merge_trials', 'max_split_trials',
                     'local_subset_frames', 'local_subset_max_new_leaves', 'world_zero_frames'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'persistent_regions.{name} must be a positive integer')
        if self.max_leaves > 64 or self.min_region_size < 8:
            raise ValueError('Persistent regions require at most 64 leaves and at least 8 GS per new region')
        if self.local_subset_max_new_leaves > 2:
            raise ValueError('Local maintenance proposes at most two new leaves per event')
        if not 0 < self.migration_gain < 1 or not math.isfinite(self.migration_gain):
            raise ValueError('migration_gain must lie in (0,1)')
        if not math.isfinite(self.split_p90_units) or self.split_p90_units <= 0:
            raise ValueError('split_p90_units must be positive')
        for name in ('local_subset_high_units', 'local_subset_spatial_scale_extent'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'persistent_regions.{name} must be positive')


class PersistentRegionState:
    def __init__(self):
        self.initialized = False
        self.region_ids = torch.tensor([0], dtype=torch.long)
        self.reference_region_ids = self.region_ids.clone()
        self.next_region_id = 1
        self.migration_pending = {}
        self.region_evidence = {}
        self.last_evidence_frame = -1
        self.redirects = {}
        self.parent_overrides = {}
        self.last_discovery_frame = -1
        self.last_discovery_result = None
        self.subset_evidence = SustainedResidualSubsetEvidence()
        self.subset_last_observation = None
        self.world_zero_evidence = {}
        self.last_local_frame = -1
        self.last_local_result = None

    def capture(self):
        return dict(initialized=self.initialized, region_ids=self.region_ids.clone(),
            reference_region_ids=self.reference_region_ids.clone(),
            next_region_id=self.next_region_id,
            migration_pending=deepcopy(self.migration_pending),
            region_evidence=deepcopy(self.region_evidence),
            last_evidence_frame=self.last_evidence_frame,
            redirects=deepcopy(self.redirects), parent_overrides=deepcopy(self.parent_overrides),
            last_discovery_frame=self.last_discovery_frame,
            last_discovery_result=deepcopy(self.last_discovery_result),
            subset_evidence=self.subset_evidence.state_dict(),
            subset_last_observation=deepcopy(self.subset_last_observation),
            world_zero_evidence=deepcopy(self.world_zero_evidence),
            last_local_frame=self.last_local_frame,
            last_local_result=deepcopy(self.last_local_result))

    def restore(self, state):
        self.initialized = bool(state.get('initialized', False))
        self.region_ids = torch.as_tensor(state.get('region_ids', [0])).detach().cpu().long().clone()
        self.reference_region_ids = torch.as_tensor(state.get('reference_region_ids', [0])).detach().cpu().long().clone()
        self.next_region_id = int(state.get('next_region_id', 1))
        self.migration_pending = deepcopy(state.get('migration_pending', {}))
        self.region_evidence = deepcopy(state.get('region_evidence', {}))
        self.last_evidence_frame = int(state.get('last_evidence_frame', -1))
        self.redirects = deepcopy(state.get('redirects', {}))
        self.parent_overrides = deepcopy(state.get('parent_overrides', {}))
        self.last_discovery_frame = int(state.get('last_discovery_frame', -1))
        self.last_discovery_result = deepcopy(state.get('last_discovery_result'))
        self.subset_evidence = SustainedResidualSubsetEvidence()
        if state.get('subset_evidence') is not None:
            self.subset_evidence.load_state_dict(state['subset_evidence'])
        self.subset_last_observation = deepcopy(state.get('subset_last_observation'))
        self.world_zero_evidence = deepcopy(state.get('world_zero_evidence', {}))
        self.last_local_frame = int(state.get('last_local_frame', -1))
        self.last_local_result = deepcopy(state.get('last_local_result'))


def ancestry_closure(owner, parents):
    """Keep owner-empty internal nodes whenever a live leaf descends from them."""
    owner = torch.as_tensor(owner).detach().cpu().long()
    parents = torch.as_tensor(parents).detach().cpu().long()
    if parents.ndim != 1 or not len(parents) or int(parents[0]) != -1:
        raise ValueError('Persistent ancestry requires an identity root')
    keep = {0}
    for slot in torch.unique(owner).tolist():
        path = set()
        while slot not in keep:
            if slot in path or slot < 0 or slot >= len(parents):
                raise ValueError('Invalid persistent region ancestry')
            path.add(slot)
            slot = int(parents[slot])
        keep.update(path)
    return sorted(keep)


class PersistentMotionMixin:
    @property
    def _persistent_redirects(self):
        return self.persistent_state.redirects

    def _persistent_start_frame(self):
        state = self.persistent_state
        if len(state.region_ids) != len(self.parents):
            raise ValueError('Persistent logical IDs do not match the decoded slot layout')
        state.reference_region_ids = state.region_ids.clone()
        state.redirects = {}
        state.parent_overrides = {}
        live_ids = set(self.online_ids.cpu().tolist())
        state.migration_pending = {key: value for key, value in state.migration_pending.items()
                                   if key in live_ids}

    def _persistent_owner_ids(self):
        return self.persistent_state.region_ids.to(self.owner.device)[self.owner]

    @staticmethod
    def _follow_region_redirect(identifier, redirects):
        seen = set()
        while identifier in redirects:
            if identifier in seen:
                raise ValueError('Persistent region redirect cycle')
            seen.add(identifier)
            identifier = int(redirects[identifier])
        return identifier

    @torch.no_grad()
    def _compact_persistent_layout(self, owner_ids, redirects=None):
        """Return dense slots while preserving the ancestry of surviving leaves."""
        state = self.persistent_state
        redirects = {**state.redirects, **({} if redirects is None else redirects)}
        previous_ids = state.region_ids.tolist()
        previous_slots = {int(identifier): slot for slot, identifier in enumerate(previous_ids)}
        live_ids = set(torch.unique(owner_ids).detach().cpu().tolist())
        live_known = [previous_slots[identifier] for identifier in live_ids if identifier in previous_slots]
        keep = ancestry_closure(live_known, self.parents)
        logical = sorted({0, *live_ids, *(previous_ids[slot] for slot in keep)})
        slots = {identifier: slot for slot, identifier in enumerate(logical)}
        owner = torch.zeros_like(owner_ids)
        for identifier, slot in slots.items():
            owner[owner_ids == identifier] = slot
        parents = owner.new_zeros(len(logical)); parents[0] = -1
        for identifier in logical[1:]:
            if identifier in previous_slots:
                old_parent = int(self.parents[previous_slots[identifier]])
                parent_id = previous_ids[old_parent]
                parents[slots[identifier]] = slots[parent_id]
            elif identifier in state.parent_overrides:
                parents[slots[identifier]] = slots[state.parent_overrides[identifier]]
        mapping = [slots.get(self._follow_region_redirect(int(identifier), redirects), 0)
                   for identifier in state.reference_region_ids.tolist()]
        return dict(owner=owner, parents=parents, region_ids=torch.tensor(logical, dtype=torch.long),
                    region_map=owner.new_tensor(mapping))

    @torch.no_grad()
    def _fit_persistent_partition(self, owner_ids, target_xyz, target_q, redirects=None):
        """Compact live leaves, fit their current sharing, and preserve GS poses."""
        from .rigid_residual_tree import region_pivots
        layout = self._compact_persistent_layout(owner_ids, redirects)
        self.owner, self.parents, self.region_map = layout['owner'], layout['parents'], layout['region_map']
        owner = self.owner
        logical = layout['region_ids'].tolist()
        self.pivots = region_pivots(self.base_xyz, owner, self.parents)
        state = self.persistent_state
        state.region_ids = layout['region_ids'].clone()
        if self.config.persistent_regions.tree_enabled:
            return self._refit_persistent_tree(target_xyz, target_q)
        self.region_translation = nn.Parameter(self.base_xyz.new_zeros(len(logical), 3))
        self.region_rotation = nn.Parameter(self.base_xyz.new_zeros(len(logical), 3))
        self.gs_translation = nn.Parameter(torch.zeros_like(self.base_xyz))
        self.gs_rotation = nn.Parameter(torch.zeros_like(self.base_xyz))
        observed = self.observations >= self.config.online_clustering.min_observations
        root_q = self.base_xyz.new_tensor([1., 0., 0., 0.])
        for slot in range(len(logical)):
            rows = torch.where(owner == slot)[0]
            if not len(rows):
                continue
            q, b = root_q, self.base_xyz.new_zeros(3)
            valid = rows[observed[rows]]
            if slot and len(valid) >= 3:
                t, r = self._fit_online_local(valid, target_xyz, target_q, self.pivots[slot])
                fitted = self._proposed_global(0, self.pivots[slot], t, r)
                if fitted is None:
                    return False
                self.region_translation[slot], self.region_rotation[slot] = fitted[:2]
                q, b = fitted[2:]
            residual = self._retarget_rows(rows, q, b, target_xyz, target_q)
            if residual is None:
                return False
            self.gs_translation[rows], self.gs_rotation[rows] = residual
        state.region_ids = torch.tensor(logical, dtype=torch.long)
        self._metadata_cache = None
        return True

    def _persistent_quantized_cost(self):
        # Legacy whole-leaf split/merge proxy: each block uses its own grid.
        # New local subset edits explicitly normalize regional symbols to GS
        # units instead; preserve the historical whole-leaf heuristic here.
        return sum(int(value.long().abs().sum()) for value in self._values(hard=True))

    def _persistent_layout_snapshot(self):
        return dict(partition=self._online_partition_snapshot(), state=self.persistent_state.capture())

    def _restore_persistent_layout(self, snapshot):
        self._install_online_partition(snapshot['partition'])
        self.persistent_state.restore(snapshot['state'])

    @torch.no_grad()
    def _try_persistent_owner_ids(self, owner_ids, target_xyz, target_q, redirects=None,
                                  require_reduction=False):
        original = self._persistent_layout_snapshot()
        original_cost = self._persistent_quantized_cost()
        valid = self._fit_persistent_partition(owner_ids, target_xyz, target_q, redirects)
        # Quantized retargeting validates every GS pose. The final complete
        # packet is separately decoded once after all accepted local edits.
        if not valid or (require_reduction and self._persistent_quantized_cost() >= original_cost):
            self._restore_persistent_layout(original)
            return False
        self._make_optimizer()
        if redirects:
            self.persistent_state.redirects.update(redirects)
        from .rigid_residual_tree import _id_lengths
        self._region_id_lengths = _id_lengths(len(self.parents) - 1, self.base_xyz)
        return True

    @torch.no_grad()
    def _persistent_cost_for_owners(self, rows, candidates, target_xyz, target_q):
        from .rigid_residual_tree import (quaternion_rotate, quaternion_multiply,
            quaternion_inverse, quaternion_rotvec, rotation_vector_quaternion,
            quaternion_angle_error)
        q_all, b_all = self._global_regions()
        q, b = q_all[candidates], b_all[candidates]
        predicted = quaternion_rotate(q, self.base_xyz[rows]) + b
        predicted_q = F.normalize(quaternion_multiply(q, self.base_rotation[rows]), dim=-1)
        t = target_xyz[rows] - predicted
        r = quaternion_rotvec(quaternion_multiply(F.normalize(target_q[rows], dim=-1),
                                                  quaternion_inverse(predicted_q)))
        scaled_t, scaled_r = t / self.gs_steps[:3], r / self.gs_steps[3:]
        integer_t, integer_r = scaled_t.round(), scaled_r.round()
        valid = (integer_t.abs().le(32767).all(-1) & integer_r.abs().le(32767).all(-1))
        recovered = predicted + integer_t * self.gs_steps[:3]
        recovered_q = quaternion_multiply(rotation_vector_quaternion(integer_r * self.gs_steps[3:]), predicted_q)
        epsilon = torch.finfo(self.base_xyz.dtype).eps * max(1., self.extent) * 8
        valid &= (recovered - target_xyz[rows]).norm(dim=-1).le(math.sqrt(3) * float(self.gs_steps[0]) / 2 + epsilon)
        valid &= quaternion_angle_error(recovered_q, target_q[rows]).le(math.sqrt(3) * float(self.gs_steps[3]) / 2 + epsilon)
        cost = (integer_t.abs().sum(-1) + integer_r.abs().sum(-1)) / 6
        return torch.where(valid, cost, torch.full_like(cost, math.inf))

    @torch.no_grad()
    def _migrate_persistent_gs(self, target_xyz, target_q):
        cfg, state = self.config.persistent_regions, self.persistent_state
        observed = self.observations >= self.config.online_clustering.min_observations
        observed_rows = torch.where(observed)[0]
        owner_ids = self._persistent_owner_ids()
        pending_rows = {}
        pending_baseline = {}
        moved = 0
        _, _, gt, gr = self._values(hard=True)
        stored_cost = (gt.long().abs().sum(-1) + gr.long().abs().sum(-1)).float() / 6
        internal = torch.zeros(len(self.parents), dtype=torch.bool, device=self.owner.device)
        internal[self.parents[1:]] = True
        leaf_slots = torch.where(~internal & (torch.arange(len(self.parents), device=self.owner.device) > 0))[0]
        near_count = min(cfg.migration_neighbors, len(leaf_slots))
        for start in range(0, len(observed_rows), self.config.online_clustering.assignment_chunk_size):
            rows = observed_rows[start:start + self.config.online_clustering.assignment_chunk_size]
            current = self.owner[rows]
            baseline = stored_cost[rows]
            best, cost = current.clone(), baseline.clone()
            near = (leaf_slots[torch.cdist(self.base_xyz[rows], self.pivots[leaf_slots]).topk(near_count, largest=False).indices]
                    if near_count else rows.new_empty((len(rows), 0)))
            for column in range(near_count + 1):
                candidate = torch.zeros_like(current) if column == 0 else near[:, column - 1]
                trial = self._persistent_cost_for_owners(rows, candidate, target_xyz, target_q)
                better = trial < cost
                best[better], cost[better] = candidate[better], trial[better]
            improves = (best != current) & (baseline > 0) & (cost <= baseline * (1 - cfg.migration_gain))
            ids = self.online_ids[rows.cpu()].tolist()
            target_ids = state.region_ids[best.cpu()].tolist()
            for row, identifier, target, improvement, original_cost in zip(
                    rows.cpu().tolist(), ids, target_ids, improves.cpu().tolist(), baseline.cpu().tolist()):
                if not improvement:
                    state.migration_pending.pop(identifier, None)
                    continue
                previous = state.migration_pending.get(identifier)
                streak = previous['streak'] + 1 if previous and previous['target'] == target else 1
                if streak >= cfg.migration_frames:
                    pending_rows[row] = target
                    pending_baseline[row] = original_cost
                else:
                    state.migration_pending[identifier] = dict(target=int(target), streak=int(streak))
        original = self._persistent_layout_snapshot()
        for row, target in pending_rows.items():
            owner_ids[row] = target
        if pending_rows and self._try_persistent_owner_ids(owner_ids, target_xyz, target_q):
            _, _, gt, gr = self._values(hard=True)
            after = ((gt.long().abs().sum(-1) + gr.long().abs().sum(-1)).float() / 6).cpu().numpy()
            good = {row: target for row, target in pending_rows.items()
                    if float(after[row]) <= pending_baseline[row] * (1 - cfg.migration_gain)}
            if len(good) != len(pending_rows):
                self._restore_persistent_layout(original)
                owner_ids = self._persistent_owner_ids()
                for row, target in good.items():
                    owner_ids[row] = target
                if not good or not self._try_persistent_owner_ids(owner_ids, target_xyz, target_q):
                    return 0
                _, _, gt, gr = self._values(hard=True)
                after = ((gt.long().abs().sum(-1) + gr.long().abs().sum(-1)).float() / 6).cpu().numpy()
                if any(float(after[row]) > pending_baseline[row] * (1 - cfg.migration_gain) for row in good):
                    self._restore_persistent_layout(original)
                    return 0
                pending_rows = good
            for row in pending_rows:
                state.migration_pending.pop(int(self.online_ids[row]), None)
            moved = len(pending_rows)
        return moved

    def _persistent_region_measurements(self):
        cfg, state = self.config.persistent_regions, self.persistent_state
        rt, rr, gt, gr = self._values(hard=True)
        observed = self.observations >= self.config.online_clustering.min_observations
        units = torch.cat((gt.long().abs(), gr.long().abs()), -1).float().mean(-1)
        result = {}
        for slot, identifier in enumerate(state.region_ids.tolist()):
            if slot == 0:
                continue
            members = self.owner == slot
            valid = members & observed
            count = int(valid.sum())
            supported = count >= cfg.min_region_size
            result[identifier] = dict(observed_gs=count, valid=supported,
                p90_units=float(torch.quantile(units[valid], .9)) if supported else None,
                idle=bool(rt[slot].eq(0).all() and rr[slot].eq(0).all() and
                          gt[members].eq(0).all() and gr[members].eq(0).all()))
        return result

    @torch.no_grad()
    def _discover_persistent_root_regions(self, target_xyz, target_q, history):
        """Create a few local leaves from persistent, actually observed root motion.

        Existing leaves retain their owners and logical IDs. This initializer
        admits useful nonidentity sharing by pose validity, rather than charging
        all new ownership bytes as an immediate rejection before joint fitting.
        """
        from .root_motion_discovery import propose_root_motion_regions
        cfg, state = self.config.root_motion_discovery, self.persistent_state
        if not cfg.enabled:
            return dict(enabled=False, accepted_regions=[], trials=[])
        if state.last_discovery_frame == self.frame_index:
            result = deepcopy(state.last_discovery_result or {})
            result['same_frame_skipped'] = True
            return result
        state.last_discovery_frame = self.frame_index
        _, _, gt, gr = self._values(hard=True)
        priorities = (gt.long().abs().sum(-1) + gr.long().abs().sum(-1)).cpu().numpy()
        leaves = int(torch.unique(self.owner[self.owner > 0]).numel())
        available = self.config.persistent_regions.max_leaves - leaves
        result = dict(enabled=True, frame=self.frame_index, same_frame_skipped=False,
            available_leaves=available, accepted_regions=[], trials=[],
            acceptance='nonidentity sharing and decoded pose validity; no immediate packet byte veto',
            evidence='consecutive valid observed translations in GS quantization units',
            spatial_scale_extent=cfg.spatial_scale_extent)
        if available <= 0:
            result['rejection_reason'] = 'leaf_capacity_reached'
            state.last_discovery_result = deepcopy(result)
            return result
        root_mask = self.owner.detach().cpu().eq(0).numpy()
        proposal = propose_root_motion_regions(history, self.online_ids.numpy(), root_mask,
            self.extent, self.gs_steps[:3].cpu().numpy(), self.config.online_clustering,
            cfg, available_leaves=available, priorities=priorities)
        result['proposal'] = proposal.metrics
        for group in proposal.groups:
            rows = torch.as_tensor(group, device=self.owner.device, dtype=torch.long)
            if len(rows) < cfg.min_region_size or not bool(self.owner[rows].eq(0).all()):
                result['trials'].append(dict(gs_count=len(rows), accepted=False, reason='unsupported_or_no_longer_root'))
                continue
            original = self._persistent_layout_snapshot()
            before = self._online_packet_score(target_xyz, target_q)
            identifier = state.next_region_id
            state.parent_overrides[identifier] = 0
            owner_ids = self._persistent_owner_ids(); owner_ids[rows] = identifier
            trial = dict(region_id=identifier, gs_count=len(rows), accepted=False,
                before_packet=before, after_packet=None)
            if not self._try_persistent_owner_ids(owner_ids, target_xyz, target_q):
                self._restore_persistent_layout(original)
                trial['reason'] = 'sharing_fit_or_quantized_retarget_failed'
                result['trials'].append(trial)
                continue
            slot = int(torch.where(state.region_ids == identifier)[0][0])
            rt, rr, new_gt, new_gr = self._values(hard=True)
            if not bool(rt[slot].ne(0).any() or rr[slot].ne(0).any()):
                self._restore_persistent_layout(original)
                trial['reason'] = 'identity_shared_fit'
                result['trials'].append(trial)
                continue
            after = self._online_packet_score(target_xyz, target_q)
            trial['after_packet'] = after
            if not after['valid']:
                self._restore_persistent_layout(original)
                trial['reason'] = 'decoded_pose_invalid'
                result['trials'].append(trial)
                continue
            state.next_region_id += 1
            state.initialized = True
            trial.update(accepted=True, reason='valid_root_shared_candidate',
                packet_bytes_delta=after['bytes'] - before['bytes'],
                before_gs_integer_l1=int(priorities[rows.cpu().numpy()].sum()),
                after_gs_integer_l1=int(new_gt[rows].long().abs().sum() + new_gr[rows].long().abs().sum()))
            result['accepted_regions'].append(dict(region_id=identifier, gs_count=len(rows)))
            result['trials'].append(trial)
        state.last_discovery_result = deepcopy(result)
        return result

    @torch.no_grad()
    def _finalize_persistent_evidence(self):
        state = self.persistent_state
        if state.last_evidence_frame == self.frame_index:
            return
        original = state.capture()
        try:
            self._record_persistent_evidence()
        except Exception:
            # A partial sidecar/world observation must not consume a frame
            # before all evidence fields and the common marker are committed.
            state.restore(original)
            raise

    @torch.no_grad()
    def _record_persistent_evidence(self):
        state, cfg = self.persistent_state, self.config.persistent_regions
        if state.last_evidence_frame == self.frame_index:
            return
        for identifier, measure in self._persistent_region_measurements().items():
            previous = state.region_evidence.setdefault(identifier,
                dict(split_streak=0, idle_streak=0, last_valid_frame=-1))
            if not measure['valid']:
                continue
            previous.update(observed_gs=measure['observed_gs'], p90_units=measure['p90_units'],
                            last_valid_frame=self.frame_index)
            previous['split_streak'] = previous['split_streak'] + 1 if measure['p90_units'] > cfg.split_p90_units else 0
            previous['idle_streak'] = previous['idle_streak'] + 1 if measure['idle'] else 0
        if cfg.local_maintenance_enabled:
            state.subset_evidence.config = SustainedResidualSubsetConfig(
                required_observations=cfg.local_subset_frames, min_region_size=cfg.min_region_size)
            _, _, gt, gr = self._values(hard=True)
            symbols = torch.cat((gt, gr), -1).short().cpu().numpy()
            observed = (self.observations >= self.config.online_clustering.min_observations).cpu().numpy()
            evidence = state.subset_evidence.observe(self.frame_index, self.online_ids.numpy(),
                self._persistent_owner_ids().cpu().numpy(), symbols, observed,
                high_units=cfg.local_subset_high_units)
            state.subset_last_observation = dict(evidence.metrics,
                stage='final_joint_motion', step=self.motion_steps)
            for identifier, measure in self._persistent_world_zero_measurements().items():
                previous = state.world_zero_evidence.setdefault(identifier,
                    dict(streak=0, last_valid_frame=-1))
                if measure['valid']:
                    previous.update(streak=previous['streak'] + 1 if measure['world_shared_zero'] else 0,
                                    last_valid_frame=self.frame_index, observed_gs=measure['observed_gs'])
        state.last_evidence_frame = self.frame_index

    @torch.no_grad()
    def _split_persistent_region(self, identifier, target_xyz, target_q, history):
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
        from .online_motion_clustering import _features
        cfg, state = self.config.persistent_regions, self.persistent_state
        slots = {int(value): slot for slot, value in enumerate(state.region_ids.tolist())}
        if identifier not in slots or torch.unique(self.owner[self.owner > 0]).numel() >= cfg.max_leaves:
            return False
        observed = self.observations >= self.config.online_clustering.min_observations
        rows = torch.where((self.owner == slots[identifier]) & observed)[0]
        if len(rows) < 2 * cfg.min_region_size:
            return False
        window = history.aligned(self.online_ids.numpy())
        indices = rows.cpu().numpy()
        common = np.flatnonzero(np.all(window['observed'][indices], axis=0))
        if not len(common):
            return False
        features = _features(window, indices, common, self.extent, self.config.online_clustering)
        if len(np.unique(features, axis=0)) < 2:
            return False
        rng = np.random.default_rng(self.config.online_clustering.random_seed + self.frame_index + identifier)
        sample = np.sort(rng.choice(len(rows), min(len(rows), self.config.online_clustering.max_fit_points), replace=False))
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=2, n_init=1, max_iter=self.config.online_clustering.max_iter,
                           random_state=self.config.online_clustering.random_seed).fit(features[sample])
            labels = model.predict(features)
        groups = [rows[torch.as_tensor(labels == label, device=rows.device)] for label in range(2)]
        if min(len(group) for group in groups) < cfg.min_region_size:
            return False
        groups.sort(key=lambda group: (-len(group), int(self.online_ids[group.cpu()].min())))
        owner_ids = self._persistent_owner_ids()
        original = self._persistent_layout_snapshot()
        new_id = state.next_region_id
        source_parent = int(self.parents[slots[identifier]])
        state.parent_overrides[new_id] = int(state.region_ids[source_parent])
        owner_ids[groups[1]] = new_id
        if not self._try_persistent_owner_ids(owner_ids, target_xyz, target_q, require_reduction=True):
            self._restore_persistent_layout(original)
            return False
        state.next_region_id += 1
        state.region_evidence.pop(identifier, None)
        state.region_evidence.pop(new_id, None)
        return True

    @torch.no_grad()
    def _edit_persistent_structure(self, target_xyz, target_q, history,
                                   skip_idle=False, skip_split=False, skip_merge=False):
        cfg, state = self.config.persistent_regions, self.persistent_state
        edits, redirects = [], {}
        measures = self._persistent_region_measurements()
        # Final refined evidence supplies the streak; current observed motion
        # must still support the condition before an edit is considered.
        for identifier, measure in measures.items():
            evidence = state.region_evidence.get(identifier, {})
            if (not skip_idle and measure['valid'] and measure['idle'] and
                    evidence.get('idle_streak', 0) >= cfg.idle_frames):
                owner_ids = self._persistent_owner_ids(); owner_ids[owner_ids == identifier] = 0
                proposed = {**redirects, identifier: 0}
                if self._try_persistent_owner_ids(owner_ids, target_xyz, target_q, proposed):
                    redirects = proposed
                    edits.append(dict(kind='persistent_idle_to_root', region_id=identifier))
        eligible = sorted((identifier for identifier, measure in measures.items()
            if measure['valid'] and measure['p90_units'] > cfg.split_p90_units and
            state.region_evidence.get(identifier, {}).get('split_streak', 0) >= cfg.split_frames),
            key=lambda identifier: (-measures[identifier]['p90_units'], identifier))
        for identifier in ([] if skip_split else eligible[:cfg.max_split_trials]):
            if self._split_persistent_region(identifier, target_xyz, target_q, history):
                edits.append(dict(kind='persistent_split', region_id=identifier,
                                  child_region_id=state.next_region_id - 1))
        pairs = []
        merge_support = self._persistent_region_measurements()
        if not skip_merge and len(state.region_ids) > 2:
            distance = torch.cdist(self.pivots[1:], self.pivots[1:]).cpu().numpy()
            world_q, world_b = self._global_regions()
            from .rigid_residual_tree import quaternion_angle_error
            # Prioritize similarity of shared motion; spatial proximity only
            # breaks ties and limits otherwise equivalent local proposals.
            motion_distance = torch.cdist(world_b[1:] / self.steps[0], world_b[1:] / self.steps[0])
            count = len(world_q) - 1
            motion_distance += quaternion_angle_error(
                world_q[1:, None, :].expand(count, count, 4),
                world_q[None, 1:, :].expand(count, count, 4)).to(motion_distance) / self.steps[3]
            motion_distance = motion_distance.cpu().numpy()
            for left in range(len(distance)):
                if not merge_support.get(int(state.region_ids[left + 1]), {}).get('valid', False):
                    continue
                neighbors = [right for right in np.lexsort((distance[left], motion_distance[left]))
                    if right != left and merge_support.get(int(state.region_ids[right + 1]), {}).get('valid', False)]
                for right in neighbors[:cfg.merge_neighbors]:
                    if left < right:
                        pairs.append((float(motion_distance[left, right]), float(distance[left, right]), int(state.region_ids[left + 1]), int(state.region_ids[right + 1])))
        for _, _, left, right in sorted(pairs)[:cfg.max_merge_trials]:
            if left not in state.region_ids.tolist() or right not in state.region_ids.tolist():
                continue
            if not (merge_support.get(left, {}).get('valid', False) and merge_support.get(right, {}).get('valid', False)):
                continue
            owner_ids = self._persistent_owner_ids(); owner_ids[owner_ids == right] = left
            proposed = {**redirects, right: left}
            if self._try_persistent_owner_ids(owner_ids, target_xyz, target_q, proposed, require_reduction=True):
                redirects = proposed
                state.region_evidence.pop(right, None)
                edits.append(dict(kind='persistent_merge', source_region_id=right, destination_region_id=left))
        return edits

    @torch.no_grad()
    def _edit_local_persistent_structure(self, target_xyz, target_q, history):
        """Recycle/merge first, then expose sustained member motion locally."""
        from .sustained_residual_subset import propose_local_residual_groups
        cfg, state = self.config.persistent_regions, self.persistent_state
        if state.last_local_frame == self.frame_index:
            result = deepcopy(state.last_local_result or {})
            result['same_frame_skipped'] = True
            return [], result
        state.last_local_frame = self.frame_index
        edits = []
        result = dict(frame=self.frame_index, enabled=True, same_frame_skipped=False,
            order=['world_zero_recycle', 'legacy_merge', 'sustained_subset',
                   'legacy_whole_leaf_split', 'root_discovery', 'tree'],
            gain_scope='New regional symbols converted to GS grid units plus affected GS integer L1',
            legacy_gain_scope='Whole-leaf merge/split unchanged: each block integer L1 in its own grid',
            recycle_trials=[], subset_trials=[], accepted_regions=[], clustering=[])
        for identifier, measure in self._persistent_world_zero_measurements().items():
            if (measure['valid'] and measure['world_shared_zero'] and
                    state.world_zero_evidence.get(identifier, {}).get('streak', 0) >= cfg.world_zero_frames):
                trial = self._apply_persistent_local_edit(target_xyz, target_q,
                                                          redirects={identifier: 0})
                trial['region_id'] = identifier
                result['recycle_trials'].append(trial)
                if trial['accepted']:
                    edits.append(dict(kind='persistent_world_zero_to_root', region_id=identifier))
                    state.world_zero_evidence.pop(identifier, None)
                    state.region_evidence.pop(identifier, None)
        merge_edits = self._edit_persistent_structure(target_xyz, target_q, history,
                                                     skip_idle=True, skip_split=True)
        edits.extend(merge_edits)
        result['legacy_merge_edits'] = deepcopy(merge_edits)
        state.subset_evidence.config = SustainedResidualSubsetConfig(
            required_observations=cfg.local_subset_frames, min_region_size=cfg.min_region_size)
        _, _, gt, gr = self._values(hard=True)
        observed = (self.observations >= self.config.online_clustering.min_observations).cpu().numpy()
        symbols = torch.cat((gt, gr), -1).short().cpu().numpy()
        proposal = state.subset_evidence.peek(self.frame_index, self.online_ids.numpy(),
            self._persistent_owner_ids().cpu().numpy(), symbols, observed,
            high_units=cfg.local_subset_high_units)
        result['evidence'] = proposal.metrics
        result['evidence_stage'] = 'Previous completed final_joint_motion only; current100 checks high without advancing'
        candidates = []
        # Bound work per maintenance event, while prioritizing actual encoded
        # residual demand rather than floating point near-zero values.
        sources = sorted(proposal.groups, key=lambda group: (-group['integer_l1'], group['region_id']))
        for group in sources[:cfg.max_split_trials]:
            groups, metrics = propose_local_residual_groups(history, self.online_ids.numpy(),
                self.base_xyz.cpu().numpy(), group['rows'], self.extent,
                self.config.online_clustering, cfg.local_subset_spatial_scale_extent,
                cfg.min_region_size, max_groups=cfg.local_subset_max_new_leaves)
            result['clustering'].append(dict(source_region_id=int(group['region_id']), **metrics))
            for rows in groups:
                demand = int(np.abs(symbols[np.asarray(rows)].astype(np.int64)).sum())
                candidates.append((demand, int(group['region_id']), np.asarray(rows)))
        for demand, identifier, indices in sorted(candidates,
                key=lambda value: (-value[0], value[1], int(self.online_ids[value[2]].min()))):
            if len(result['accepted_regions']) >= cfg.local_subset_max_new_leaves:
                break
            available = cfg.max_leaves - int(torch.unique(self.owner[self.owner > 0]).numel())
            if available <= 0:
                result['subset_trials'].append(dict(source_region_id=identifier,
                    gs_count=len(indices), accepted=False, reason='leaf_capacity_reached'))
                break
            rows = torch.as_tensor(indices, device=self.owner.device, dtype=torch.long)
            owners = self._persistent_owner_ids()
            if (not bool(owners[rows].eq(identifier).all()) or
                    int(owners.eq(identifier).sum()) - len(rows) < cfg.min_region_size):
                result['subset_trials'].append(dict(source_region_id=identifier,
                    gs_count=len(rows), accepted=False, reason='changed_membership_or_insufficient_retained_support'))
                continue
            new_id = state.next_region_id
            trial = self._apply_persistent_local_edit(target_xyz, target_q,
                new_leaves=[dict(region_id=new_id, source_region_id=identifier, rows=rows)])
            trial.update(source_region_id=identifier, region_id=new_id, gs_count=len(rows),
                         proposed_gs_integer_l1=demand)
            result['subset_trials'].append(trial)
            if trial['accepted']:
                result['accepted_regions'].append(dict(region_id=new_id,
                    source_region_id=identifier, gs_count=len(rows)))
                edits.append(dict(kind='persistent_residual_subset', source_region_id=identifier,
                                  child_region_id=new_id, gs_count=len(rows)))
        split_edits = self._edit_persistent_structure(target_xyz, target_q, history,
                                                     skip_idle=True, skip_merge=True)
        result['legacy_split_edits'] = deepcopy(split_edits)
        edits.extend(split_edits)
        result['remaining_leaf_capacity'] = cfg.max_leaves - int(torch.unique(self.owner[self.owner > 0]).numel())
        state.last_local_result = deepcopy(result)
        return edits, result

    @torch.no_grad()
    def _update_persistent_regions(self, step):
        from .online_motion_clustering import OnlineMotionHistory
        from .rigid_residual_tree import quaternion_multiply, quaternion_inverse
        state = self.persistent_state
        if not state.initialized:
            target_xyz, target_q = (value.detach().clone() for value in self.forward(step))
            edits = self._update_online_flat_partition(step)
            state.region_ids = torch.arange(len(self.parents), dtype=torch.long)
            state.next_region_id = len(self.parents)
            state.initialized = bool(self.owner.any())
            self.flat_region_diagnostics['persistent'] = dict(initialization=True,
                region_ids=state.region_ids.tolist(), migrated_gs=0, structural_edits=[])
            if self.config.root_motion_discovery.enabled:
                history = self._persistent_current_history(target_xyz, target_q)
                self.flat_region_diagnostics['root_discovery'] = self._discover_persistent_root_regions(target_xyz, target_q, history)
            if self.config.persistent_regions.tree_enabled and state.initialized:
                history = self._persistent_current_history(target_xyz, target_q)
                self.flat_region_diagnostics['tree'] = self._propose_persistent_tree(target_xyz, target_q, history)
                self.flat_region_diagnostics['persistent']['region_ids'] = state.region_ids.tolist()
                self._update_online_region_diagnostics(target_xyz, target_q)
                score = self._online_packet_score(target_xyz, target_q)
                self.flat_region_diagnostics['refinement']['before'] = self._online_refinement_snapshot(step, score)
                self.flat_region_diagnostics['refinement']['objective'].update(
                    child_relative_motion_target='Non-root-parent child corrections only; root-level sharing excluded',
                    child_relative_motion_weight=self.config.tree_child_relative_loss_weight)
            return edits
        started = time.perf_counter()
        target_xyz, target_q = (value.detach().clone() for value in self.forward(step))
        original = self._persistent_layout_snapshot()
        original_score = self._online_packet_score(target_xyz, target_q)
        old_owner = self.owner.clone()
        owner_ids = self._persistent_owner_ids()
        if not self._try_persistent_owner_ids(owner_ids, target_xyz, target_q):
            self._restore_persistent_layout(original)
            self._topology_updated = True
            return []
        retained_score = self._online_packet_score(target_xyz, target_q)
        moved = self._migrate_persistent_gs(target_xyz, target_q)
        structural = self.frame_index % self.config.persistent_regions.structure_interval == 0
        tree_due = (self.config.persistent_regions.tree_enabled and
            (self.frame_index == 1 or self.frame_index % self.config.motion_tree.update_interval == 0))
        history = self._persistent_current_history(target_xyz, target_q) if structural or tree_due or self.config.root_motion_discovery.enabled else None
        local_metrics = None
        if structural and self.config.persistent_regions.local_maintenance_enabled:
            edits, local_metrics = self._edit_local_persistent_structure(target_xyz, target_q, history)
            discovery = self._discover_persistent_root_regions(target_xyz, target_q, history)
        else:
            discovery = self._discover_persistent_root_regions(target_xyz, target_q, history)
            edits = self._edit_persistent_structure(target_xyz, target_q, history) if structural else []
        tree_metrics = self._propose_persistent_tree(target_xyz, target_q, history) if self.config.persistent_regions.tree_enabled else None
        final_score = self._online_packet_score(target_xyz, target_q)
        adjustment_valid = bool(final_score['valid'])
        failed_score = None
        if not adjustment_valid:
            failed_score = deepcopy(final_score)
            self._restore_persistent_layout(original)
            final_score, moved, edits = original_score, 0, []
            discovery['rolled_back'] = True
            discovery['proposed_accepted_regions'] = deepcopy(discovery.get('accepted_regions', []))
            discovery['accepted_regions'] = []
            for trial in discovery.get('trials', []):
                if trial.get('accepted', False):
                    trial['proposed_accepted'] = True
                    trial['accepted'] = False
                    trial['reason'] = 'whole_adjustment_decoded_pose_invalid'
            if self.config.root_motion_discovery.enabled:
                state.last_discovery_frame = self.frame_index
                state.last_discovery_result = deepcopy(discovery)
            if tree_metrics is not None:
                tree_metrics['proposed_accepted'] = bool(tree_metrics.get('accepted', False))
                tree_metrics['accepted'] = False
                tree_metrics['rolled_back'] = True
                tree_metrics['rejection_reason'] = 'whole_adjustment_decoded_pose_invalid'
            if local_metrics is not None:
                local_metrics['rolled_back'] = True
                local_metrics['proposed_accepted_regions'] = deepcopy(local_metrics.get('accepted_regions', []))
                local_metrics['accepted_regions'] = []
                for trial in local_metrics.get('subset_trials', []) + local_metrics.get('recycle_trials', []):
                    if trial.get('accepted', False):
                        trial.update(proposed_accepted=True, accepted=False,
                                     reason='whole_adjustment_decoded_pose_invalid')
                local_metrics['proposed_legacy_merge_edits'] = local_metrics.get('legacy_merge_edits', [])
                local_metrics['proposed_legacy_split_edits'] = local_metrics.get('legacy_split_edits', [])
                local_metrics['legacy_merge_edits'] = []
                local_metrics['legacy_split_edits'] = []
                state.last_local_frame = self.frame_index
                state.last_local_result = deepcopy(local_metrics)
        selected = self._online_partition_snapshot(); self._install_online_partition(selected)
        self._topology_updated = True
        self.bootstrap_finalized = True
        self.cluster_labels = None
        self.flat_region_diagnostics = dict(frame=self.frame_index, proposal_skipped=False,
            proposal=dict(method='persistent', history_transitions=min(
                self.config.online_clustering.history_frames, len(self.online_history.records) + 1)),
            gate=dict(acceptance_policy='persistent_motion',
                decision_reason='pose_valid_persistent_refit' if adjustment_valid else 'whole_adjustment_decoded_pose_invalid',
                accepted=adjustment_valid, old=retained_score, candidate=final_score,
                failed_candidate=failed_score,
                rejection_reason=None if adjustment_valid else 'whole_adjustment_decoded_pose_invalid',
                packet_improves=final_score['bytes'] < retained_score['bytes'],
                packet_bytes_delta=final_score['bytes'] - retained_score['bytes']),
            persistent=dict(initialization=False, region_ids=state.region_ids.tolist(),
                migrated_gs=moved, pending_migrations=len(state.migration_pending),
                structural_check=structural, structural_edits=edits,
                evidence=deepcopy(state.region_evidence), seconds=time.perf_counter() - started))
        if tree_metrics is not None:
            self.flat_region_diagnostics['tree'] = tree_metrics
        self.flat_region_diagnostics['root_discovery'] = discovery
        if self.config.persistent_regions.local_maintenance_enabled:
            self.flat_region_diagnostics['local_maintenance'] = (local_metrics if local_metrics is not None else
                dict(enabled=True, frame=self.frame_index, performed=False,
                     reason='structure_interval_not_due', accepted_regions=[]))
        self._update_online_region_diagnostics(target_xyz, target_q, old_owner)
        self.flat_region_diagnostics['refinement'] = dict(free_motion_steps=self.config.motion_steps,
            joint_refinement_steps=self.config.post_partition_steps,
            start_step=self.config.motion_steps + 1, end_step=self.motion_steps,
            membership_fixed=True, finalized=False,
            objective=dict(image_loss=True,
                storage='Adam-metric sparse block proximal; detached rate diagnostic',
                relative_motion_target='GS residual translation / extent and rotation radians',
                relative_motion_weight=self.config.relative_motion_loss_weight,
                child_relative_motion_target='Non-root-parent child corrections only; root-level sharing excluded',
                child_relative_motion_weight=(self.config.tree_child_relative_loss_weight
                    if self.config.persistent_regions.tree_enabled else 0.)),
            before=self._online_refinement_snapshot(step, final_score))
        edit = dict(kind='persistent_adjust', frame=self.frame_index, migrated_gs=moved,
                    structural_edits=edits)
        self.topology_edits.append(edit)
        return [edit]

    def _persistent_current_history(self, target_xyz, target_q):
        from .online_motion_clustering import OnlineMotionHistory
        from .rigid_residual_tree import quaternion_multiply, quaternion_inverse
        history = OnlineMotionHistory(self.config.online_clustering.history_frames)
        history.load_state_dict(self.online_history.state_dict())
        relative = quaternion_multiply(F.normalize(target_q, dim=-1),
            quaternion_inverse(F.normalize(self.base_rotation, dim=-1)))
        observed = (self.observations >= self.config.online_clustering.min_observations).cpu().numpy()
        history.append(self.frame_index, self.online_ids.numpy(), self.base_xyz.cpu().numpy(),
                       target_xyz.cpu().numpy(), relative.cpu().numpy(), observed)
        return history
