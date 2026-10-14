"""Pose-preserving tree fitting and admission for persistent region layouts.

The persistent owner/ID lifecycle is owned by PersistentMotionMixin. This
module fits existing ancestors every frame and appends coherent new parents
without assigning GS to them. Regional quantization is always followed by GS
retargeting, and a complete codec decode validates the final candidate pose.
"""

import numpy as np
import torch
from torch import nn

from .online_motion_tree import (propose_motion_tree, validate_motion_tree,
                                 world_to_local_motion)


class PersistentMotionTreeMixin:
    @staticmethod
    def _tree_descendant_rows(owner, parents):
        """Stable row order, including owner-empty parents with live children."""
        owner = owner.detach().cpu().numpy()
        parents = parents.detach().cpu().numpy()
        order, _ = validate_motion_tree(parents, owner)
        rows = [np.flatnonzero(owner == slot) for slot in range(len(parents))]
        for slot in order[::-1]:
            parent = int(parents[slot])
            rows[parent] = np.sort(np.concatenate((rows[parent], rows[slot])))
        return rows

    @torch.no_grad()
    def _install_tree_motion_and_retarget(self, translation, rotvec, target_xyz, target_q):
        """Install quantized locals; retain every GS target through its residual."""
        t = torch.as_tensor(translation, device=self.base_xyz.device, dtype=self.base_xyz.dtype)
        r = torch.as_tensor(rotvec, device=self.base_xyz.device, dtype=self.base_xyz.dtype)
        if t.shape != (len(self.parents), 3) or r.shape != t.shape:
            return False
        integer_t, integer_r = (t / self.steps[:3]).round(), (r / self.steps[3:]).round()
        if not all(bool(torch.isfinite(value).all() and value.ge(-32768).all() and
                        value.le(32767).all()) for value in (integer_t, integer_r)):
            return False
        if bool(integer_t[0].ne(0).any() or integer_r[0].ne(0).any()):
            return False
        self.region_translation = nn.Parameter(integer_t * self.steps[:3])
        self.region_rotation = nn.Parameter(integer_r * self.steps[3:])
        self.gs_translation = nn.Parameter(torch.zeros_like(self.base_xyz))
        self.gs_rotation = nn.Parameter(torch.zeros_like(self.base_xyz))
        world_q, world_b = self._global_regions()
        for slot in torch.unique(self.owner).tolist():
            members = torch.where(self.owner == slot)[0]
            values = self._retarget_rows(members, world_q[slot], world_b[slot], target_xyz, target_q)
            if values is None:
                return False
            self.gs_translation[members], self.gs_rotation[members] = values
        self._metadata_cache = None
        return True

    @torch.no_grad()
    def _refit_persistent_tree(self, target_xyz, target_q):
        """Fit world sharing for every live leaf/ancestor before deriving locals.

        Parents, owner slots, pivots and logical IDs have already been compacted
        by the caller. Unsupported nodes conservatively carry identity world
        motion; independent residuals preserve their fitted GS targets. The
        caller restores its snapshot if this method returns False and creates
        an optimizer only once the complete layout is accepted.
        """
        from .rigid_residual_tree import rotation_delta, rotation_vector_quaternion

        try:
            validate_motion_tree(self.parents.detach().cpu().numpy(),
                self.owner.detach().cpu().numpy(), self.config.motion_tree.max_depth)
            rows = self._tree_descendant_rows(self.owner, self.parents)
            observed = (self.observations >= self.config.online_clustering.min_observations)
            observed = observed.detach().cpu().numpy()
            count = len(self.parents)
            q_world = self.base_xyz.new_zeros(count, 4)
            q_world[:, 0] = 1
            b_world = self.base_xyz.new_zeros(count, 3)
            for slot in range(1, count):
                supported = rows[slot][observed[rows[slot]]]
                if len(supported) < self.config.motion_tree.min_observed_gs:
                    continue
                members = torch.as_tensor(supported, device=self.owner.device, dtype=torch.long)
                t, r = self._fit_online_local(members, target_xyz, target_q, self.pivots[slot])
                q = rotation_vector_quaternion(r[None])[0]
                q_world[slot] = q
                b_world[slot] = rotation_delta(q, -self.pivots[slot]) + t
            local_t, local_r = world_to_local_motion(self.parents.detach().cpu().numpy(),
                self.pivots.detach().cpu().double().numpy(), q_world.detach().cpu().double().numpy(),
                b_world.detach().cpu().double().numpy())
            return self._install_tree_motion_and_retarget(local_t, local_r, target_xyz, target_q)
        except (ValueError, np.linalg.LinAlgError):
            return False

    @torch.no_grad()
    def _propose_persistent_tree(self, target_xyz, target_q, history):
        """Apply an observed quantized-magnitude improvement, then decode-check.

        Admission does not veto a useful initialized parent merely because the
        first complete packet contains its topology cost. Both actual packet
        costs are exposed, and the eventual experiment must satisfy the full
        storage requirement. No promised future byte savings are counted here.
        """
        if not self.config.persistent_regions.tree_enabled:
            return dict(enabled=False, accepted=False, rejection_reason='tree_disabled')
        from .rigid_residual_tree import _id_lengths

        state = self.persistent_state
        original = self._persistent_layout_snapshot()
        before = self._online_packet_score(target_xyz, target_q)
        result = dict(enabled=True, accepted=False, before_packet=before,
                      after_packet=None, rejection_reason=None, helper=None,
                      admission_objective='affected_quantized_integer_l1',
                      whole_packet_byte_veto=False)
        try:
            world_q, world_b = self._global_regions()
            observed = self.observations >= self.config.online_clustering.min_observations
            proposal = propose_motion_tree(self.base_xyz.detach().cpu().double().numpy(),
                target_xyz.detach().cpu().double().numpy(), self.owner.detach().cpu().numpy(),
                state.region_ids.numpy(), world_q.detach().cpu().double().numpy(),
                world_b.detach().cpu().double().numpy(), self.extent,
                self.steps[:3].detach().cpu().double().numpy(),
                self.steps[3:].detach().cpu().double().numpy(), self.config.motion_tree,
                parents=self.parents.detach().cpu().numpy(), pivots=self.pivots.detach().cpu().double().numpy(),
                observed=observed.detach().cpu().numpy(),
                history=None if history is None else history.aligned(self.online_ids.numpy()),
                next_region_id=state.next_region_id, frame_index=self.frame_index)
            result['helper'] = proposal.metrics
            if not proposal.metrics['inserted_parents']:
                result['rejection_reason'] = ('tree_update_not_due' if proposal.metrics['update_skipped']
                                              else 'no_supported_quantized_gain')
                return result
            self.parents = self.owner.new_tensor(proposal.parents)
            self.pivots = torch.as_tensor(proposal.pivots, device=self.base_xyz.device,
                                         dtype=self.base_xyz.dtype)
            state.region_ids = torch.as_tensor(proposal.region_ids, dtype=torch.long).clone()
            slots = {int(identifier): slot for slot, identifier in enumerate(state.region_ids.tolist())}
            redirects = getattr(self, '_persistent_redirects', {})
            self.region_map = self.owner.new_tensor([slots.get(
                self._follow_region_redirect(int(identifier), redirects), 0)
                for identifier in state.reference_region_ids.tolist()])
            validate_motion_tree(proposal.parents, self.owner.detach().cpu().numpy(),
                                 self.config.motion_tree.max_depth)
            if not self._install_tree_motion_and_retarget(proposal.local_translation,
                    proposal.local_rotvec, target_xyz, target_q):
                result['rejection_reason'] = 'tree_quantization_or_gs_retarget_failed'
                self._restore_persistent_layout(original)
                return result
            score = self._online_packet_score(target_xyz, target_q)
            result['after_packet'] = score
            if not score['valid']:
                result['rejection_reason'] = 'tree_packet_pose_invalid'
                self._restore_persistent_layout(original)
                return result
            state.next_region_id = proposal.next_region_id
            self._make_optimizer()
            self._region_id_lengths = _id_lengths(len(self.parents) - 1, self.base_xyz)
            result.update(accepted=True, packet_bytes_delta=score['bytes'] - before['bytes'],
                          packet_improves=score['bytes'] < before['bytes'])
            return result
        except (ValueError, np.linalg.LinAlgError) as error:
            self._restore_persistent_layout(original)
            result['rejection_reason'] = 'tree_candidate_invalid'
            result['error'] = str(error)
            return result
