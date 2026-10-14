"""Local persistent leaf edits that preserve untouched pivots and parameters.

Admission compares actual quantized motion in the GS quantization reference.
This magnitude proxy is distinct from sparse-block and compressed packet bytes.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .online_motion_tree import validate_motion_tree


def unified_gs_motion_cost(region_t, region_r, gs_t, gs_r, regional_steps,
                           gs_steps, affected_rows=None):
    """Weight regional integers by regional/GS step, avoiding precision bias."""
    rt, rr, gt, gr = [torch.as_tensor(value).detach().cpu().long() for value in
                      (region_t, region_r, gs_t, gs_r)]
    rs, gs = [torch.as_tensor(value, dtype=torch.float64).detach().cpu() for value in (regional_steps, gs_steps)]
    if (rt.shape != rr.shape or gt.shape != gr.shape or rt.ndim != 2 or gt.ndim != 2 or
            rt.shape[1] != 3 or gt.shape[1] != 3 or rs.shape != (6,) or gs.shape != (6,) or
            not bool(torch.isfinite(rs).all() and torch.isfinite(gs).all() and rs.gt(0).all() and gs.gt(0).all())):
        raise ValueError('unified motion cost needs integer XYZ blocks and positive six-channel steps')
    if affected_rows is not None:
        rows = torch.as_tensor(affected_rows).detach().cpu().long()
        gt, gr = gt[rows], gr[rows]
    ratio = rs / gs
    regional_cost = float((rt[1:].abs().double() * ratio[:3]).sum() +
                          (rr[1:].abs().double() * ratio[3:]).sum())
    gs_cost = int(gt.abs().sum() + gr.abs().sum())
    return dict(unified_gs_units_l1=regional_cost + gs_cost,
        regional_in_gs_units_l1=regional_cost, affected_gs_integer_l1=gs_cost,
        regional_raw_integer_l1=int(rt[1:].abs().sum() + rr[1:].abs().sum()),
        regional_to_gs_step_ratio=ratio.tolist(),
        regional_translation_blocks=int(rt[1:].ne(0).any(-1).sum()),
        regional_rotation_blocks=int(rr[1:].ne(0).any(-1).sum()),
        affected_gs_translation_blocks=int(gt.ne(0).any(-1).sum()),
        affected_gs_rotation_blocks=int(gr.ne(0).any(-1).sum()),
        affected_gs_rows=int((gt.ne(0).any(-1) | gr.ne(0).any(-1)).sum()),
        scope='all regional symbols weighted into GS units plus affected GS integers; not encoded bytes')


class _LocalEditRejected(Exception):
    pass


def _raw_sparse_motion_bytes(values):
    from .rigid_tree_codec import _sparse_write
    rt, rr, gt, gr = [value.detach().cpu() for value in values]
    shared, residual = bytearray(), bytearray()
    _sparse_write(shared, rt[1:], rr[1:])
    _sparse_write(residual, gt, gr)
    return len(shared), len(residual)


class PersistentLocalMotionMixin:
    @torch.no_grad()
    def _persistent_world_zero_measurements(self):
        """Composed WORLD identity evidence; independent GS residuals may move."""
        from .rigid_residual_tree import quaternion_rotate, quaternion_rotvec

        q, b = self._global_regions()
        observed = self.observations >= self.config.online_clustering.min_observations
        result = {}
        for slot in torch.unique(self.owner[self.owner > 0]).tolist():
            rows = torch.where((self.owner == slot) & observed)[0]
            valid = len(rows) >= self.config.persistent_regions.min_region_size
            zero = False
            if valid:
                rotation_zero = bool((quaternion_rotvec(q[slot:slot + 1]) / self.steps[3:]).round().eq(0).all())
                displacement = quaternion_rotate(q[slot].expand(len(rows), -1), self.base_xyz[rows]) + b[slot] - self.base_xyz[rows]
                zero = rotation_zero and bool((displacement / self.steps[:3]).round().eq(0).all())
            result[int(self.persistent_state.region_ids[slot])] = dict(
                observed_gs=len(rows), valid=valid, world_shared_zero=bool(zero),
                evidence='composed WORLD rotation and observed shared displacement round to zero in regional steps')
        return result

    @torch.no_grad()
    def _apply_persistent_local_edit(self, target_xyz, target_q, new_leaves=None, redirects=None):
        """Append sibling leaves/recycle zero-WORLD leaves, then codec validate.

        ``new_leaves`` contains region_id, source_region_id and current GS rows.
        Only changed-owner rows are retargeted; all unaffected local parameters,
        GS residuals and explicit pivots are copied exactly. Recycle evidence
        streaks are checked by the caller; current WORLD zero is checked here.
        """
        from .rigid_residual_tree import (quaternion_inverse, quaternion_multiply,
            quaternion_rotvec, rotation_vector_quaternion, rotation_delta, _id_lengths)

        new_leaves = [] if new_leaves is None else list(new_leaves)
        redirects = {} if redirects is None else dict(redirects)
        original = self._persistent_layout_snapshot()
        state = self.persistent_state
        original_ids = state.region_ids.tolist()
        old_slots = {identifier: slot for slot, identifier in enumerate(original_ids)}
        owner_ids = self._persistent_owner_ids().clone()
        before_values = self._values(hard=True)
        report = dict(accepted=False, reason=None, new_region_ids=[], recycled_region_ids=[],
            affected_gs=0, before_motion_cost=None, after_motion_cost=None,
            before_packet=None, after_packet=None,
            admission_policy=('strict_decrease_in_unified_GS_quantized_units' if new_leaves
                              else 'supported_current_WORLD_zero_and_pose_valid_capacity_recycle'),
            whole_packet_byte_veto=False, unaffected_parameters_and_pivots_preserved=False)
        try:
            if not new_leaves and not redirects:
                raise _LocalEditRejected('empty_edit')
            validate_motion_tree(self.parents.detach().cpu().numpy(), self.owner.detach().cpu().numpy(),
                                 self.config.motion_tree.max_depth)
            if (target_xyz.shape != self.base_xyz.shape or target_q.shape != self.base_rotation.shape or
                    not bool(torch.isfinite(target_xyz).all() and torch.isfinite(target_q).all())):
                raise _LocalEditRejected('invalid_target_pose')
            observed = self.observations >= self.config.online_clustering.min_observations
            changed = torch.zeros(len(self.owner), dtype=torch.bool, device=self.owner.device)
            zero_evidence = self._persistent_world_zero_measurements() if redirects else {}
            for source, destination in redirects.items():
                if (isinstance(source, bool) or not isinstance(source, (int, np.integer)) or
                        source not in old_slots or source == 0 or destination != 0):
                    raise _LocalEditRejected('invalid_recycle_redirect')
                if not zero_evidence.get(int(source), {}).get('world_shared_zero', False):
                    raise _LocalEditRejected('current_WORLD_shared_not_zero_or_unsupported')
                rows = owner_ids == source
                changed |= rows
                owner_ids[rows] = 0
                report['recycled_region_ids'].append(int(source))
            selected_by_source, leaves = {}, []
            for leaf in new_leaves:
                identifier, source = leaf['region_id'], leaf['source_region_id']
                if (isinstance(identifier, bool) or not isinstance(identifier, (int, np.integer)) or
                        identifier < state.next_region_id or identifier in old_slots or
                        identifier in report['new_region_ids'] or source not in old_slots or source in redirects):
                    raise _LocalEditRejected('invalid_new_region_identity')
                rows = torch.as_tensor(leaf['rows'], device=self.owner.device)
                if (rows.ndim != 1 or rows.dtype not in (torch.int32, torch.int64) or
                        len(rows) < self.config.persistent_regions.min_region_size or
                        bool((rows < 0).any() or (rows >= len(self.owner)).any()) or len(rows.unique()) != len(rows)):
                    raise _LocalEditRejected('invalid_selected_rows_or_support')
                rows = rows.long()
                if (not bool(owner_ids[rows].eq(source).all() and observed[rows].all()) or bool(changed[rows].any())):
                    raise _LocalEditRejected('selected_membership_changed_or_unobserved')
                source_slot = old_slots[source]
                parent_id = (0 if source == 0 else original_ids[int(self.parents[source_slot])])
                state.parent_overrides[int(identifier)] = int(parent_id)
                owner_ids[rows] = int(identifier)
                changed[rows] = True
                selected_by_source[source] = selected_by_source.get(source, 0) + len(rows)
                leaves.append((int(identifier), rows))
                report['new_region_ids'].append(int(identifier))
            for source, count in selected_by_source.items():
                total = int(self._persistent_owner_ids().eq(source).sum())
                if source and total - count < self.config.persistent_regions.min_region_size:
                    raise _LocalEditRejected('insufficient_retained_members')
            count = int(torch.unique(owner_ids[owner_ids > 0]).numel())
            if count > self.config.persistent_regions.max_leaves:
                raise _LocalEditRejected('leaf_capacity_reached')
            rows = torch.where(changed)[0]
            report['affected_gs'] = len(rows)
            report['before_motion_cost'] = unified_gs_motion_cost(*before_values, self.steps, self.gs_steps, rows)
            report['raw_shared_motion_bytes_before'], report['raw_residual_motion_bytes_before'] = _raw_sparse_motion_bytes(before_values)
            report['before_packet'] = self._online_packet_score(target_xyz, target_q)
            layout = self._compact_persistent_layout(owner_ids, redirects)
            logical = layout['region_ids'].tolist()
            new_slots = {identifier: slot for slot, identifier in enumerate(logical)}
            rt, rr = self.base_xyz.new_zeros(len(logical), 3), self.base_xyz.new_zeros(len(logical), 3)
            pivots = self.base_xyz.new_zeros(len(logical), 3)
            for identifier, slot in new_slots.items():
                if identifier in old_slots:
                    old = old_slots[identifier]
                    rt[slot] = original['partition']['region_translation'][old]
                    rr[slot] = original['partition']['region_rotation'][old]
                    pivots[slot] = original['partition']['pivots'][old]
            self.owner, self.parents, self.region_map = layout['owner'], layout['parents'], layout['region_map']
            self.pivots = pivots
            self.region_translation, self.region_rotation = nn.Parameter(rt), nn.Parameter(rr)
            self.gs_translation = nn.Parameter(original['partition']['gs_translation'].clone())
            self.gs_rotation = nn.Parameter(original['partition']['gs_rotation'].clone())
            state.region_ids = layout['region_ids'].clone()
            parent_q, parent_b = self._global_regions()
            for identifier, members in leaves:
                slot = new_slots[identifier]
                pivot = self.base_xyz[members].mean(0)
                self.pivots[slot] = pivot
                world_t, world_r = self._fit_online_local(members, target_xyz, target_q, pivot)
                world_q = rotation_vector_quaternion(world_r[None])[0]
                world_b = rotation_delta(world_q, -pivot) + world_t
                parent = int(self.parents[slot])
                relative = F.normalize(quaternion_multiply(world_q, quaternion_inverse(parent_q[parent])), dim=-1)
                local_r = quaternion_rotvec(relative[None])[0]
                inherited_pivot = rigid_rotate(parent_q[parent], pivot)
                local_t = world_b - parent_b[parent] - rotation_delta(relative, -inherited_pivot)
                integer_t, integer_r = (local_t / self.steps[:3]).round(), (local_r / self.steps[3:]).round()
                if not all(bool(torch.isfinite(value).all() and value.ge(-32768).all() and value.le(32767).all())
                           for value in (integer_t, integer_r)):
                    raise _LocalEditRejected('shared_fit_quantization_overflow')
                self.region_translation[slot] = integer_t * self.steps[:3]
                self.region_rotation[slot] = integer_r * self.steps[3:]
            validate_motion_tree(self.parents.detach().cpu().numpy(), self.owner.detach().cpu().numpy(),
                                 self.config.motion_tree.max_depth)
            world_q, world_b = self._global_regions()
            for slot in torch.unique(self.owner[rows]).tolist():
                members = rows[self.owner[rows] == slot]
                values = self._retarget_rows(members, world_q[slot], world_b[slot], target_xyz, target_q)
                if values is None:
                    raise _LocalEditRejected('quantized_retarget_pose_invalid')
                self.gs_translation[members], self.gs_rotation[members] = values
            self._metadata_cache = None
            self._region_id_lengths = _id_lengths(len(self.parents) - 1, self.base_xyz)
            after_values = self._values(hard=True)
            report['after_motion_cost'] = unified_gs_motion_cost(*after_values, self.steps, self.gs_steps, rows)
            report['raw_shared_motion_bytes_after'], report['raw_residual_motion_bytes_after'] = _raw_sparse_motion_bytes(after_values)
            report['unified_cost_before'] = report['before_motion_cost']['unified_gs_units_l1']
            report['unified_cost_after'] = report['after_motion_cost']['unified_gs_units_l1']
            report['unified_quantized_gain'] = report['unified_cost_before'] - report['unified_cost_after']
            # Values have actually been quantized and GS retargeted. Do not gate
            # a zero-WORLD capacity recycle on a strict magnitude improvement.
            if new_leaves and not report['unified_cost_after'] < report['unified_cost_before']:
                raise _LocalEditRejected('no_unified_quantized_gain')
            report['after_packet'] = self._online_packet_score(target_xyz, target_q)
            if not report['after_packet']['valid']:
                raise _LocalEditRejected('full_codec_pose_guard_failed')
            state.redirects.update({int(key): int(value) for key, value in redirects.items()})
            if leaves:
                state.next_region_id = max(identifier for identifier, _ in leaves) + 1
            self._make_optimizer()
            report.update(accepted=True, reason='valid_local_quantized_gain' if leaves else 'valid_WORLD_zero_recycle',
                unaffected_parameters_and_pivots_preserved=True,
                packet_bytes_delta=report['after_packet']['bytes'] - report['before_packet']['bytes'],
                raw_shared_motion_bytes_before=report['before_packet']['raw_shared_motion_bytes'],
                raw_shared_motion_bytes_after=report['after_packet']['raw_shared_motion_bytes'],
                raw_residual_motion_bytes_before=report['before_packet']['raw_residual_motion_bytes'],
                raw_residual_motion_bytes_after=report['after_packet']['raw_residual_motion_bytes'])
        except (_LocalEditRejected, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            self._restore_persistent_layout(original)
            report['reason'] = str(error) if isinstance(error, _LocalEditRejected) else 'local_fit_or_codec_error'
            if not isinstance(error, _LocalEditRejected):
                report['error'] = str(error)
            report['rolled_back'] = True
        return report


def rigid_rotate(quaternion, vector):
    # A lazy import avoids the RigidResidualTree/MRO module import cycle.
    from .rigid_residual_tree import quaternion_rotate
    return quaternion_rotate(quaternion, vector)
