"""Causal flat partitions with configurable acceptance and exact packet costs.

No hierarchy is learned here. Existing tree/HEIR experiments keep their own
implementation; this mixin is enabled only by online_flat_clustering.
"""

import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .online_motion_clustering import OnlineMotionHistory, propose_motion_regions


class OnlineFlatMotionMixin:
    def _online_partition_snapshot(self):
        names = ('owner', 'parents', 'pivots', 'region_map', 'region_translation',
                 'region_rotation', 'gs_translation', 'gs_rotation')
        return {name: getattr(self, name).detach().clone() for name in names}

    def _install_online_partition(self, snapshot):
        from .rigid_residual_tree import _id_lengths
        for name, value in snapshot.items():
            value = value.detach().clone()
            setattr(self, name, nn.Parameter(value) if name.endswith(('translation', 'rotation')) else value)
        self._make_optimizer()
        self._region_id_lengths = _id_lengths(len(self.parents) - 1, self.base_xyz)
        self._metadata_cache = None

    def _online_candidate_owner(self, groups, old_owner):
        """Match by GS overlap so arbitrary cluster label order costs no churn.

        Existing node IDs stay allocated. Splits receive fresh IDs; disappeared
        nodes remain empty and zero. No per-GS pose or history uses region IDs.
        """
        owner = torch.zeros_like(old_owner)
        group_rows = [torch.as_tensor(group, device=owner.device, dtype=torch.long) for group in groups]
        pairs = []
        for group_id, rows in enumerate(group_rows):
            nodes, counts = torch.unique(old_owner[rows], return_counts=True)
            pairs.extend((int(count), int(node), group_id) for node, count in zip(nodes, counts) if int(node) > 0)
        chosen, used = {}, set()
        for count, node, group_id in sorted(pairs, key=lambda item: (-item[0], item[1], item[2])):
            if group_id not in chosen and node not in used:
                chosen[group_id] = node
                used.add(node)
        next_node = len(self.frame_reference_state.parents)
        # Group order is deterministic upstream, and unmatched groups are new.
        for group_id, rows in enumerate(group_rows):
            if group_id not in chosen:
                chosen[group_id] = next_node
                next_node += 1
            owner[rows] = chosen[group_id]
        return owner, next_node

    def _fit_online_partition(self, owner, nodes, target_xyz, target_q):
        from .rigid_residual_tree import region_pivots
        self.owner = owner.clone()
        self.parents = owner.new_zeros(nodes)
        self.parents[0] = -1
        self.pivots = region_pivots(self.base_xyz, owner, self.parents)
        # Stable overlap matching preserves every previous node's identifier.
        self.region_map = torch.arange(len(self.frame_reference_state.parents), device=owner.device)
        self.region_translation = nn.Parameter(self.base_xyz.new_zeros(nodes, 3))
        self.region_rotation = nn.Parameter(self.base_xyz.new_zeros(nodes, 3))
        rows = torch.where(owner == 0)[0]
        root = self.base_xyz.new_tensor([1., 0., 0., 0.])
        values = self._retarget_rows(rows, root, self.base_xyz.new_zeros(3), target_xyz, target_q)
        if values is None:
            return False
        self.gs_translation = nn.Parameter(torch.zeros_like(self.base_xyz))
        self.gs_rotation = nn.Parameter(torch.zeros_like(self.base_xyz))
        self.gs_translation[rows], self.gs_rotation[rows] = values
        for node in torch.unique(owner).tolist():
            if node == 0:
                continue
            members = torch.where(owner == node)[0]
            t, r = self._fit_online_local(members, target_xyz, target_q, self.pivots[node])
            shared = self._proposed_global(0, self.pivots[node], t, r)
            residual = (None if shared is None else self._retarget_rows(
                members, shared[2], shared[3], target_xyz, target_q))
            if residual is None:
                return False
            self.region_translation[node], self.region_rotation[node] = shared[:2]
            self.gs_translation[members], self.gs_rotation[members] = residual
        self._metadata_cache = None
        return True

    def _fit_online_local(self, members, target_xyz, target_q, pivot):
        from .shared_motion_clustering import fit_shared_motion, quaternion_matrices
        from .rigid_residual_tree import (quaternion_multiply, quaternion_inverse,
            quaternion_rotvec, rotation_delta)
        cfg = self.config.online_clustering
        relative = quaternion_multiply(F.normalize(target_q[members], dim=-1),
                                      quaternion_inverse(F.normalize(self.base_rotation[members], dim=-1)))
        # Covariance orientations are optional evidence; the default shared fit
        # is positional Kabsch. The existing robust solver needs positive length.
        length = max(np.sqrt(cfg.orientation_fit_weight) * cfg.orientation_length_extent * self.extent,
                     np.finfo(np.float64).eps)
        fit = fit_shared_motion(self.base_xyz[members].double().cpu().numpy(),
            target_xyz[members].double().cpu().numpy(),
            quaternion_matrices(relative.double().cpu().numpy()),
            self.observations[members].clamp_min(1).double().cpu().numpy(), length)
        q = torch.as_tensor(fit['quaternion'], device=self.base_xyz.device, dtype=self.base_xyz.dtype)
        b = torch.as_tensor(fit['offset'], device=self.base_xyz.device, dtype=self.base_xyz.dtype)
        return b - rotation_delta(q, -pivot), quaternion_rotvec(q[None])[0]

    def _online_packet_score(self, target_xyz, target_q):
        """Count header, shared/residual blocks, layout, owner updates, checksum.

        Both competitors contain the same survivors, and no births yet. Birth
        selection happens subsequently; those unknown future rows cannot be
        honestly credited as expected future savings at this gate.
        """
        from .rigid_tree_codec import (encode_motion, compress_motion_packet, decode_motion,
                                       dynamic_layout_bytes, _sparse_write, _uvar)
        from .rigid_residual_tree import quaternion_angle_error
        rt, rr, gt, gr = (value.detach().cpu() for value in self._values(hard=True))
        raw = encode_motion(self.frame_reference_state, self.frame_index, self.extent,
            self.previous_rows.cpu(), self.owner.cpu(), self.parents.cpu(), rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            self.config.motion_xyz_step_extent, self.config.motion_rotation_step_rad,
            gs_translation_step_extent=self.config.gs_xyz_step_extent,
            gs_rotation_step_rad=self.config.gs_rotation_step_rad,
            region_map=self.region_map.cpu(), pivots=self.pivots.cpu())
        packet = compress_motion_packet(raw, mode='auto')
        layout, mapped_owner = dynamic_layout_bytes(self.frame_reference_state,
            self.previous_rows.cpu(), self.parents.cpu(), self.pivots.cpu(), self.region_map.cpu())
        owner_edits = torch.where(self.owner.cpu() != mapped_owner)[0].tolist()
        ownership_bytes = len(_uvar(len(owner_edits))) + sum(
            len(_uvar(row)) + len(_uvar(int(self.owner[row]))) for row in owner_edits)
        shared, residual = bytearray(), bytearray()
        _sparse_write(shared, rt[1:], rr[1:])
        _sparse_write(residual, gt, gr)
        decoded = decode_motion(packet, self.frame_reference_state)
        epsilon = torch.finfo(self.base_xyz.dtype).eps * max(1., self.extent) * 8
        xyz_error = float((decoded.xyz - target_xyz.cpu()).norm(dim=-1).max())
        rotation_error = float(quaternion_angle_error(decoded.rotation, target_q.cpu()).max())
        # NumPy comparisons return np.bool_, which json.dumps cannot encode.
        # This score is persisted inside the nested gate diagnostics, so expose
        # a native boolean at its producer rather than stringifying the report.
        valid = bool(xyz_error <= np.sqrt(3) * float(self.gs_steps[0]) / 2 + epsilon and
                     rotation_error <= np.sqrt(3) * float(self.gs_steps[3]) / 2 + epsilon)
        return dict(bytes=len(packet), raw_bytes=len(raw), valid=valid,
                    raw_layout_bytes=len(layout), raw_ownership_bytes=ownership_bytes,
                    ownership_edit_count=len(owner_edits), raw_shared_motion_bytes=len(shared),
                    raw_residual_motion_bytes=len(residual),
                    raw_header_survivor_checksum_bytes=len(raw) - len(layout) - ownership_bytes - len(shared) - len(residual),
                    max_translation_error=xyz_error, max_rotation_error=rotation_error)

    @torch.no_grad()
    def _update_online_flat_partition(self, step):
        from .rigid_residual_tree import quaternion_multiply, quaternion_inverse
        started = time.perf_counter()
        target_xyz, target_q = (value.detach().clone() for value in self.forward(step))
        old_owner = self.owner.clone()
        original = self._online_partition_snapshot()
        original_score = self._online_packet_score(target_xyz, target_q)
        history = OnlineMotionHistory(self.config.online_clustering.history_frames)
        history.load_state_dict(self.online_history.state_dict())
        relative = quaternion_multiply(F.normalize(target_q, dim=-1),
                                      quaternion_inverse(F.normalize(self.base_rotation, dim=-1)))
        history.append(self.frame_index, self.online_ids.cpu().numpy(), self.base_xyz.cpu().numpy(),
                       target_xyz.cpu().numpy(), relative.cpu().numpy(),
                       (self.observations >= self.config.online_clustering.min_observations).cpu().numpy())
        candidate = propose_motion_regions(history, self.online_ids, self.extent, self.config.online_clustering)
        candidate_owner, nodes = self._online_candidate_owner(candidate.groups, old_owner)
        # Refit the previous partition against precisely the same target, rather
        # than comparing a freshly fitted candidate with stale shared motion.
        old_valid = self._fit_online_partition(old_owner, len(self.frame_reference_state.parents), target_xyz, target_q)
        old_score = self._online_packet_score(target_xyz, target_q) if old_valid else original_score
        if not old_valid or not old_score['valid']:
            self._install_online_partition(original)
            old_score = original_score
        baseline = self._online_partition_snapshot()
        new_valid = self._fit_online_partition(candidate_owner, nodes, target_xyz, target_q)
        candidate_score = self._online_packet_score(target_xyz, target_q) if new_valid else None
        policy = self.config.online_partition_policy
        candidate_valid = bool(candidate_score and candidate_score['valid'])
        packet_improves = bool(candidate_score and candidate_score['bytes'] < old_score['bytes'])
        if not candidate_valid:
            accepted, decision_reason = False, 'quantization_error'
        elif not any(len(group) for group in candidate.groups):
            # Missing/static/noise observations do not constitute a new region
            # partition. Preserve an existing meaningful grouping in this case.
            accepted, decision_reason = False, 'empty_proposal_keep_previous'
        elif policy == 'accept_valid':
            accepted, decision_reason = True, 'valid_candidate'
        elif packet_improves:
            accepted, decision_reason = True, 'smaller_packet'
        else:
            accepted, decision_reason = False, 'complete_packet_not_smaller'
        selected = self._online_partition_snapshot() if accepted else baseline
        self._install_online_partition(selected)
        self._topology_updated = True
        self.bootstrap_finalized = True
        elapsed = time.perf_counter() - started
        if self.frame_index == 1:
            self.bootstrap_seconds = elapsed
        self.last_metrics.update({'flat_partition/accepted': int(accepted),
            'flat_partition/old_packet_bytes': old_score['bytes'],
            'flat_partition/candidate_packet_bytes': candidate_score['bytes'] if candidate_score else -1,
            'flat_partition/candidate_pose_valid': int(bool(candidate_score and candidate_score['valid'])),
            'flat_partition/seconds': elapsed})
        self.cluster_labels = torch.as_tensor(candidate.labels).cpu()
        self.flat_region_diagnostics = dict(proposal=candidate.metrics,
            frame=self.frame_index, proposal_skipped=False,
            gate=dict(old=old_score, candidate=candidate_score, accepted=accepted,
                acceptance_policy=policy, decision_reason=decision_reason,
                packet_improves=packet_improves,
                packet_bytes_delta=candidate_score['bytes'] - old_score['bytes'] if candidate_score else None,
                rejection_reason=None if accepted else decision_reason),
            history_transition_count=len(history.records),
            proposal_frame=self.frame_index)
        self._update_online_region_diagnostics(target_xyz, target_q, old_owner, candidate_owner)
        if self.config.post_partition_steps:
            self.flat_region_diagnostics['refinement'] = dict(
                free_motion_steps=self.config.motion_steps,
                joint_refinement_steps=self.config.post_partition_steps,
                start_step=self.config.motion_steps + 1, end_step=self.motion_steps,
                membership_fixed=True, finalized=False,
                objective=dict(image_loss=True,
                    storage='Adam-metric sparse block proximal; detached rate diagnostic',
                    relative_motion_target='GS residual translation / extent and rotation radians',
                    relative_motion_weight=self.config.relative_motion_loss_weight),
                before=self._online_refinement_snapshot(step,
                    candidate_score if accepted else old_score))
        edit = dict(kind='flat_partition', frame=self.frame_index, accepted=accepted,
                    acceptance_policy=policy, decision_reason=decision_reason,
                    old_packet_bytes=old_score['bytes'],
                    candidate_packet_bytes=candidate_score['bytes'] if candidate_score else None)
        self.topology_edits.append(edit)
        return [edit]

    def _online_refinement_snapshot(self, step, packet_score):
        from copy import deepcopy
        from .online_motion_diagnostics import quantized_motion_counts
        rt, rr, gt, gr = self._values(hard=True)
        return dict(step=int(step), packet_score=packet_score,
            regions=deepcopy(self.flat_region_diagnostics.get('regions', {})),
            quantized_motion=quantized_motion_counts(rt, rr, gt, gr))

    @torch.no_grad()
    def finalize_online_refinement(self):
        """Record the actual final fitted pose before freezing motion for births."""
        if not self.config.online_flat_clustering or not self.config.post_partition_steps:
            return
        if self.last_step != self.motion_steps:
            raise ValueError('Post-partition refinement can only finalize after the last motion step')
        refinement = self.flat_region_diagnostics.setdefault('refinement', dict(
            free_motion_steps=self.config.motion_steps,
            joint_refinement_steps=self.config.post_partition_steps,
            start_step=self.config.motion_steps + 1, end_step=self.motion_steps,
            membership_fixed=True, before=None))
        if refinement.get('finalized', False):
            return
        target_xyz, target_q = (value.detach() for value in self.materialized_pose())
        self._update_online_region_diagnostics(target_xyz, target_q)
        score = self._online_packet_score(target_xyz, target_q)
        if not score['valid']:
            raise ValueError('Refined motion does not match its decoded packet within quantization tolerance')
        refinement['after'] = self._online_refinement_snapshot(self.motion_steps, score)
        refinement['finalized'] = True


    def _update_online_region_diagnostics(self, target_xyz, target_q, old_owner=None, candidate_owner=None, decoded=None):
        from .online_motion_diagnostics import summarize_flat_regions
        shared_q, shared_b = self._global_regions()
        labels = self.cluster_labels
        source_xyz, source_q, owner = self.base_xyz, self.base_rotation, self.owner
        observed = self.observations >= self.config.online_clustering.min_observations
        if decoded is not None and len(decoded.xyz) > len(owner):
            births = len(decoded.xyz) - len(owner)
            # Newborn GS have no transition source. Placeholder source poses are
            # masked out of residual statistics; proportions include all GS.
            source_xyz = torch.cat((source_xyz, decoded.xyz[len(owner):].to(source_xyz)))
            source_q = torch.cat((source_q, decoded.rotation[len(owner):].to(source_q)))
            target_xyz, target_q = decoded.xyz.to(source_xyz), decoded.rotation.to(source_q)
            owner = decoded.owner.to(owner.device)
            observed = torch.cat((observed, observed.new_zeros(births)))
            if labels is not None:
                labels = torch.cat((labels, labels.new_full((births,), -3)))
        previous = self.flat_region_diagnostics.get('regions', {})
        summary = summarize_flat_regions(
            source_xyz, target_xyz, source_q, target_q,
            owner, shared_q, shared_b, observed_mask=observed,
            static_mask=None if labels is None else labels == -2,
            noise_mask=None if labels is None else labels == -1,
            unknown_mask=None if labels is None else labels == -3,
            previous_owner=old_owner, candidate_owner=candidate_owner, extent=self.extent)
        if old_owner is None:
            for name in ('ownership_changed_fraction', 'ownership_compared_gs',
                         'candidate_region_count', 'candidate_ownership_changed_fraction'):
                if name in previous:
                    summary[name] = previous[name]
        summary['new_gs_without_motion_history'] = len(owner) - len(self.owner)
        summary['residual_scope'] = 'observed surviving GS; all-GS proportions include newborn GS'
        self.flat_region_diagnostics['regions'] = summary

    def _record_online_decoded_motion(self, decoded):
        from .rigid_residual_tree import quaternion_multiply, quaternion_inverse
        count = len(self.owner)
        target_xyz = decoded.xyz[:count].to(self.base_xyz)
        target_q = decoded.rotation[:count].to(self.base_rotation)
        relative = quaternion_multiply(F.normalize(target_q, dim=-1),
                                      quaternion_inverse(F.normalize(self.base_rotation, dim=-1)))
        self.online_history.append(self.frame_index, self.online_ids.cpu().numpy(),
            self.base_xyz.cpu().numpy(), target_xyz.cpu().numpy(), relative.cpu().numpy(),
            (self.observations >= self.config.online_clustering.min_observations).cpu().numpy())
        self.flat_region_diagnostics['accepted_history_transition_count'] = len(self.online_history.records)
        self._update_online_region_diagnostics(target_xyz, target_q, decoded=decoded)
