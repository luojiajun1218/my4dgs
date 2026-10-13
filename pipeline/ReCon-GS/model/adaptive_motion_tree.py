"""Pose-preserving edits of regions grouped by their learned shared motion."""

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree

from . import rigid_residual_tree as rigid
from .shared_motion_clustering import (fit_shared_motion,
    quaternion_matrices, residual_exceeds_parent, shared_motion_groups)
from .residual_region_clustering import gmm_motion_regions, hdbscan_motion_regions
from .heir_motion_hierarchy import HeirMotionHierarchy, se3_parent_costs


class AdaptiveMotionTreeMixin:
    @torch.no_grad()
    def _apply_changed_gs_residuals(self, rows, values):
        changed = ((rigid._integer(self.gs_translation[rows], self.gs_steps[:3]) !=
                    rigid._integer(values[0], self.gs_steps[:3])).any(-1) |
                   (rigid._integer(self.gs_rotation[rows], self.gs_steps[3:]) !=
                    rigid._integer(values[1], self.gs_steps[3:])).any(-1))
        if bool(changed.any()):
            # A pose-preserving decomposition changes the residual origin,
            # not the point's world translation Jacobian. Keep its accumulated
            # Adam metric rather than treating the necessary correction as a
            # freshly unsupported motion block.
            self.gs_translation[rows[changed]] = values[0][changed]
            self.gs_rotation[rows[changed]] = values[1][changed]

    @torch.no_grad()
    def _heir_motion_features(self):
        q, offset = self._global_regions()
        config = self.config.hdbscan
        translation = offset
        if self.config.heir.geometry == 'se3':
            translation = rigid.quaternion_rotate(q, self.pivots) + offset - self.pivots
        features = torch.cat((translation / (config.translation_scale_extent * self.extent),
                              rigid.quaternion_rotvec(q) / config.rotation_scale_rad), -1)
        features[0] = 0
        return features.detach()

    @torch.no_grad()
    def _prepare_heir_history(self):
        past = self.motion_history.pool(self.owner, len(self.parents),
            self.config.clustering.orientation_length_extent * self.extent,
            self.config.hdbscan.min_observations)
        self.heir_past = {}
        for name, width in (('quaternion', 4), ('offset', 3), ('pivots', 3), ('valid', None)):
            shape = (0, len(self.parents)) + (() if width is None else (width,))
            self.heir_past[name] = (torch.as_tensor(np.asarray(past[name]), device=self.base_xyz.device,
                dtype=torch.bool if width is None else self.base_xyz.dtype) if past[name] else
                torch.empty(shape, device=self.base_xyz.device,
                            dtype=torch.bool if width is None else self.base_xyz.dtype))

    @torch.no_grad()
    def _se3_heir_context(self, current_only=False):
        q, b = self._global_regions()
        current = dict(quaternion=q[None], offset=b[None], pivots=self.pivots[None],
                       valid=torch.ones((1, len(q)), device=q.device, dtype=torch.bool))
        if not current_only:
            if self.heir_past is None:
                self._prepare_heir_history()
            current = {name: torch.cat((self.heir_past[name], value)) for name, value in current.items()}
        position = (rigid.quaternion_rotate(current['quaternion'], current['pivots']) +
                    current['offset'] - current['pivots'])
        features = torch.cat((position / (self.config.hdbscan.translation_scale_extent * self.extent),
            rigid.quaternion_rotvec(current['quaternion']) / self.config.hdbscan.rotation_scale_rad), -1)
        features[:, 0] = 0
        costs, allowed = se3_parent_costs(current['quaternion'], current['offset'],
            current['pivots'], current['valid'], self.extent, self.steps)
        return features, current['valid'], costs, allowed

    def heir_loss_terms(self, step):
        if (self.heir_hierarchy is None or self.heir_finalized or
                not self.config.motion_steps < step <= self.motion_steps):
            return {}
        progress = (step - self.config.motion_steps) / self.config.tree_refine_steps
        if self.config.heir.geometry == 'se3':
            self.heir_features, valid, costs, allowed = self._se3_heir_context()
            terms = self.heir_hierarchy.loss_terms(self.heir_features, progress, valid, allowed, costs)
            self.last_metrics.update({'heir/time_samples': len(self.heir_features),
                'heir/missing_node_samples': int((~valid).sum()),
                'heir/eligible_parent_edges': int(allowed.sum())})
        else:
            self.heir_features = self._heir_motion_features()
            terms = self.heir_hierarchy.loss_terms(self.heir_features, progress)
        self.last_metrics.update({f'heir/{name}': float(value.detach())
                                  for name, value in terms.items()})
        return terms

    @torch.no_grad()
    def _build_residual_regions(self, step):
        """Replace residual groups using standard clustering, then fit each SE(3).

        All point motions are expressed against the previous decoded frame.
        Near-identity, unobserved and noise points use the identity shared
        transform while retaining their fitted individual motion corrections.
        """
        started = time.perf_counter()
        target_xyz, target_q = (value.detach().clone() for value in self.forward(step))
        relative_q = rigid.quaternion_multiply(F.normalize(target_q, dim=-1),
            rigid.quaternion_inverse(F.normalize(self.base_rotation, dim=-1)))
        inputs = (self.base_xyz.double().cpu().numpy(), target_xyz.double().cpu().numpy(),
                  relative_q.double().cpu().numpy(), self.observations.cpu().numpy(),
                  self.extent, self.config.hdbscan)
        clustering_metrics = {}
        method = self.config.residual_clustering_method
        if method == 'gmm':
            groups, labels, clustering_metrics = gmm_motion_regions(*inputs, self.config.gmm)
        else:
            groups, labels = hdbscan_motion_regions(*inputs)
        before = len(self.parents)
        if self.config.heir.geometry == 'se3' and self.frame_index > 1 and before > 1:
            self.previous_partition = self.owner.detach().clone()
        root_q = self.base_xyz.new_tensor([1., 0., 0., 0.])
        rows = torch.arange(len(self.base_xyz), device=self.owner.device)
        root_values = self._retarget_rows(rows, root_q, self.base_xyz.new_zeros(3),
                                          target_xyz, target_q)
        if root_values is None:
            raise ValueError('Residual clustering cannot represent independent GS motion')
        count = len(groups) + 1
        source_rows = rows.new_full((count,), -1)
        source_rows[0] = 0
        self._replace_regions(self.base_xyz.new_zeros(count, 3),
                              self.base_xyz.new_zeros(count, 3), source_rows)
        self.owner = torch.zeros_like(self.owner)
        self.parents = rows.new_zeros(count)
        self.parents[0] = -1
        self._protected_regions = set()
        self.pivots = self.base_xyz.new_zeros(count, 3)
        self.pivots[0] = self.base_xyz.mean(0)
        self._apply_changed_gs_residuals(rows, root_values)
        self.cluster_labels = torch.as_tensor(labels, device=rows.device)
        raw_noise_gs = int(np.count_nonzero(labels == -1))
        rejected = 0
        rejected_gs = 0
        near_identity_groups = 0
        nonsparse_groups = 0
        deferred_sparse_groups = 0
        for node, selected in enumerate(groups, 1):
            members = torch.as_tensor(selected, device=rows.device, dtype=torch.long)
            pivot = self.base_xyz[members].detach().cpu().mean(0).to(self.base_xyz)
            translation, vector, _ = self._fit_local(members, 0, target_xyz, target_q, pivot)
            proposal = self._proposed_global(0, pivot, translation, vector)
            values = (None if proposal is None else
                      self._retarget_rows(members, proposal[2], proposal[3], target_xyz, target_q))
            if values is None:
                self.cluster_labels[members] = -1
                rejected += 1
                rejected_gs += len(members)
                continue
            predicted = rigid.quaternion_rotate(
                proposal[2].expand(len(members), -1), self.base_xyz[members]) + proposal[3]
            near_identity = (
                bool((predicted - self.base_xyz[members]).norm(dim=-1).le(
                    self.config.hdbscan.static_translation_extent * self.extent).all()) and
                float(proposal[1].norm()) <= self.config.hdbscan.static_rotation_rad)
            # Translation and rotation have independent six-byte payloads.
            # Saving translation is useful even if a GS still needs rotation.
            before_active = (root_values[0][members].ne(0).any(-1).sum() +
                             root_values[1][members].ne(0).any(-1).sum())
            after_active = (values[0].ne(0).any(-1).sum() +
                            values[1].ne(0).any(-1).sum())
            length = self.config.clustering.orientation_length_extent * self.extent
            predicted_q = rigid.quaternion_multiply(
                proposal[2].expand(len(members), -1), self.base_rotation[members])
            root_error = (target_xyz[members] - self.base_xyz[members]).square().sum(-1)
            root_error += (length * rigid.quaternion_angle_error(
                target_q[members], self.base_rotation[members])).square()
            shared_error = (target_xyz[members] - predicted).square().sum(-1)
            shared_error += (length * rigid.quaternion_angle_error(target_q[members], predicted_q)).square()
            weights = self.observations[members].clamp_min(1).to(root_error)
            root_error = (root_error * weights).sum()
            shared_error = (shared_error * weights).sum()
            reduction = float((root_error - shared_error) /
                              root_error.clamp_min(torch.finfo(root_error.dtype).tiny))
            immediate_savings = after_active < before_active
            useful_fit = reduction >= self.config.shared_fit_min_reduction
            if near_identity or not (immediate_savings or useful_fit):
                # Density clusters need not admit useful shared rigid motion.
                # Keep individual fitted poses on the identity root when their
                # mean motion cancels, or sharing neither saves payload blocks
                # nor explains a meaningful part of the fitted motion.
                # These are noise, not a claim that the members are stationary.
                self.cluster_labels[members] = -1
                near_identity_groups += int(near_identity)
                nonsparse_groups += int(not near_identity)
                rejected += 1
                rejected_gs += len(members)
                continue
            deferred_sparse_groups += int(not immediate_savings)
            self.pivots[node] = pivot
            self.region_translation[node], self.region_rotation[node] = proposal[:2]
            self.owner[members] = node
            self._apply_changed_gs_residuals(members, values)
            self._protected_regions.add(node)
        prior = self.frame_reference_state
        prior_owner = prior.owner[self.previous_rows.detach().cpu()].to(self.owner.device)
        self.region_map = rows.new_zeros(len(prior.parents))
        for node in range(1, len(prior.parents)):
            members = prior_owner == node
            if bool(members.any()):
                self.region_map[node] = torch.bincount(self.owner[members], minlength=count).argmax()
        self._compact_motion_regions()
        self.heir_features = self._heir_motion_features()
        retained = self.config.heir.geometry == 'se3' and self.heir_hierarchy is not None
        if not retained:
            self.heir_hierarchy = HeirMotionHierarchy(6, self.config.heir).to(self.base_xyz)
            self.heir_optimizer = self.heir_hierarchy.make_optimizer()
        for parameter in self.heir_hierarchy.parameters():
            parameter.requires_grad_(True)
        if self.config.heir.geometry == 'se3':
            self.heir_past = None
            self._prepare_heir_history()
            self.heir_features = self._se3_heir_context()[0]
        self.last_metrics['heir/retained_network'] = int(retained)
        self.heir_finalized = False
        self._topology_updated = True
        self._metadata_cache = None
        elapsed = time.perf_counter() - started
        self.last_metrics.update({
            'tree_edit/region_count_before': before,
            'tree_edit/region_count_after': len(self.parents),
            'tree_edit/static_candidates': int((self.cluster_labels == -2).sum()),
            'tree_edit/noise_gs': int((self.cluster_labels == -1).sum()),
            'tree_edit/raw_cluster_noise_gs': raw_noise_gs,
            'tree_edit/unobserved_gs': int((self.cluster_labels == -3).sum()),
            'tree_edit/rejected_motion_groups': rejected,
            'tree_edit/rejected_motion_gs': rejected_gs,
            'tree_edit/near_identity_motion_groups': near_identity_groups,
            'tree_edit/nonsparse_motion_groups': nonsparse_groups,
            'tree_edit/deferred_sparse_motion_groups': deferred_sparse_groups,
            'tree_edit/cluster_seconds': elapsed,
            'tree_edit/pose_max_xyz_error_world': float(
                (self.forward(step)[0] - target_xyz).norm(dim=-1).max()),
            'tree_edit/pose_max_rotation_error_rad': float(
                rigid.quaternion_angle_error(self.forward(step)[1], target_q).max()),
        })
        self.last_metrics.update({f'tree_edit/{name}': value
                                  for name, value in clustering_metrics.items()})
        if method == 'hdbscan':
            self.last_metrics['tree_edit/hdbscan_noise_gs'] = raw_noise_gs
        if self.frame_index == 1:
            self.bootstrap_seconds += elapsed
        edits = [dict(kind='cluster', method=method, regions=len(self.parents) - 1)]
        self.topology_edits.extend(edits)
        return edits

    @torch.no_grad()
    def _finalize_heir_tree(self, step, proposed=None):
        """Apply learned parents as quantized SE(3), compensating point poses."""
        started = time.perf_counter()
        if self.config.heir.geometry == 'se3' and proposed is None:
            self._select_region_partition(step)
        target_xyz, target_q = (value.detach().clone() for value in self.forward(step))
        if proposed is None:
            if self.config.heir.geometry == 'se3':
                self.heir_features, valid, costs, allowed = self._se3_heir_context()
                proposed = self.heir_hierarchy.infer_parents(self.heir_features,
                    self.config.parent_max_depth, valid, allowed)
                rows = torch.arange(1, len(proposed), device=proposed.device)
                self.last_metrics['tree_edit/selected_parent_cost_sum'] = float(costs[rows, proposed[1:]].sum())
                self.last_metrics['tree_edit/identity_parent_cost_sum'] = float(costs[rows, 0].sum())
            else:
                self.heir_features = self._heir_motion_features()
                proposed = self.heir_hierarchy.infer_parents(self.heir_features, self.config.parent_max_depth)
        old_parents = self.parents.clone()
        old_t, old_r = self.region_translation.detach().clone(), self.region_rotation.detach().clone()
        q, offset = self._global_regions()
        translation, rotation = torch.zeros_like(old_t), torch.zeros_like(old_r)
        valid = True
        for node in rigid._topological_order(proposed):
            t, r = self._local_from_global(node, int(proposed[node]), self.pivots[node], q, offset)
            if (bool((t / self.steps[:3]).abs().gt(32767.5).any()) or
                    bool((r / self.steps[3:]).abs().gt(32767.5).any())):
                valid = False
                break
            translation[node] = rigid._integer(t, self.steps[:3]).to(t) * self.steps[:3]
            rotation[node] = rigid._integer(r, self.steps[3:]).to(r) * self.steps[3:]
        transfers = []
        if valid:
            self.parents = proposed
            self.region_translation.copy_(translation)
            self.region_rotation.copy_(rotation)
            new_q, new_b = self._global_regions()
            for node in self.owner.unique().tolist():
                rows = torch.where(self.owner == node)[0]
                values = self._retarget_rows(rows, new_q[node], new_b[node], target_xyz, target_q)
                if values is None:
                    valid = False
                    break
                transfers.append((rows, values))
        if not valid:
            self.parents = old_parents
            self.region_translation.copy_(old_t)
            self.region_rotation.copy_(old_r)
        else:
            for rows, values in transfers:
                self._apply_changed_gs_residuals(rows, values)
            changed = torch.where((old_t != translation).any(-1) | (old_r != rotation).any(-1))[0]
            self._reset_adam_rows(self.region_translation, changed)
            self._reset_adam_rows(self.region_rotation, changed)
            self._protected_regions.update(changed.tolist())
        changed_parents = torch.where(self.parents != old_parents)[0].tolist()
        edits = [dict(kind='parent', node=node, parent=int(self.parents[node]), method='heir')
                 for node in changed_parents]
        collapsed = self._collapse_identity_leaves()
        self._compact_motion_regions()
        self.heir_features = self._heir_motion_features()
        if self.frame_index == 1:
            self._rebase_bootstrap(target_xyz, target_q)
            self.bootstrap_seconds += time.perf_counter() - started
        if self.config.heir.geometry == 'se3':
            self.heir_past = None
            if self.config.tree_frame(self.frame_index):
                self._prepare_heir_history()
                self.heir_features = self._se3_heir_context()[0]
            else:
                self.heir_features = self._heir_motion_features()[None]
        self.heir_finalized = True
        if self.heir_hierarchy is not None:
            for parameter in self.heir_hierarchy.parameters():
                parameter.requires_grad_(False)
        self._metadata_cache = None
        edits.extend(collapsed)
        self.topology_edits.extend(edits)
        self.last_metrics.update({'tree_edit/heir_parent_changes': len(changed_parents),
            'tree_edit/identity_leaf_merges': len(collapsed),
            'tree_edit/region_count_after': len(self.parents),
            'tree_edit/heir_projection_applied': int(valid),
            'tree_edit/heir_seconds': time.perf_counter() - started,
            'tree_edit/pose_max_xyz_error_world': float((self.forward(step)[0] - target_xyz).norm(dim=-1).max()),
            'tree_edit/pose_max_rotation_error_rad': float(rigid.quaternion_angle_error(self.forward(step)[1], target_q).max())})
        return edits

    @torch.no_grad()
    def _validate_se3_parents(self, step):
        _, _, _, allowed = self._se3_heir_context(current_only=True)
        rows = torch.arange(1, len(self.parents), device=self.parents.device)
        bad = rows[~allowed[rows, self.parents[1:]]]
        self.last_metrics['tree_edit/counteracting_parents_removed'] = len(bad)
        if not len(bad):
            self.heir_finalized = True
            return []
        proposed = self.parents.clone()
        proposed[bad] = 0
        return self._finalize_heir_tree(step, proposed)

    @torch.no_grad()
    def _partition_packet_bytes(self):
        """Price a candidate with the real compressed motion/owner encoding."""
        from .rigid_tree_codec import encode_motion, compress_motion_packet

        rt, rr, gt, gr = (value.detach().cpu() for value in self._values(hard=True))
        raw = encode_motion(self.frame_reference_state, self.frame_index, self.extent,
            self.previous_rows.cpu(), self.owner.cpu(), self.parents.cpu(), rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            self.config.motion_xyz_step_extent, self.config.motion_rotation_step_rad,
            gs_translation_step_extent=self.config.gs_xyz_step_extent,
            gs_rotation_step_rad=self.config.gs_rotation_step_rad,
            region_map=self.region_map.cpu(), pivots=self.pivots.cpu())
        return len(compress_motion_packet(raw, mode='auto'))

    @torch.no_grad()
    def _refit_previous_partition(self, owner, target_xyz, target_q):
        """Re-express the same fitted poses under the previous membership."""
        nodes = len(self.frame_reference_state.parents)
        rows = torch.arange(len(owner), device=owner.device)
        root = self.base_xyz.new_tensor([1., 0., 0., 0.])
        values = self._retarget_rows(rows, root, self.base_xyz.new_zeros(3), target_xyz, target_q)
        if values is None:
            return False
        self._replace_regions(self.base_xyz.new_zeros(nodes, 3), self.base_xyz.new_zeros(nodes, 3),
                              rows.new_full((nodes,), -1))
        self.owner = owner.clone()
        self.parents = rows.new_zeros(nodes)
        self.parents[0] = -1
        self.pivots = rigid.region_pivots(self.base_xyz, self.owner, self.parents)
        self.region_map = torch.arange(nodes, device=owner.device)
        self._protected_regions = set()
        self._apply_changed_gs_residuals(rows, values)
        for node in range(1, nodes):
            members = torch.where(owner == node)[0]
            if not len(members):
                continue
            t, r, _ = self._fit_local(members, 0, target_xyz, target_q, self.pivots[node])
            proposal = self._proposed_global(0, self.pivots[node], t, r)
            values = (None if proposal is None else self._retarget_rows(
                members, proposal[2], proposal[3], target_xyz, target_q))
            if values is None:
                return False
            self.region_translation[node], self.region_rotation[node] = proposal[:2]
            self._apply_changed_gs_residuals(members, values)
        self._compact_motion_regions()
        self._metadata_cache = None
        return True

    @torch.no_grad()
    def _select_region_partition(self, step):
        if self.previous_partition is None:
            return
        xyz, rotation = (value.detach().clone() for value in self.forward(step))
        proposal_bytes = self._partition_packet_bytes()
        snapshot = self.capture()
        previous = self.previous_partition.clone()
        valid = self._refit_previous_partition(previous, xyz, rotation)
        retained_bytes = self._partition_packet_bytes() if valid else None
        retained = valid and retained_bytes <= proposal_bytes
        if not retained:
            # Constructor initialization during restore must not perturb the
            # next training sample's RNG merely because a partition lost.
            rng = torch.get_rng_state()
            self.restore(snapshot)
            torch.set_rng_state(rng)
        else:
            self.heir_past = None
            self._prepare_heir_history()
        self.previous_partition = None
        self.last_metrics.update({'tree_edit/proposed_partition_bytes': proposal_bytes,
            'tree_edit/retained_partition_bytes': retained_bytes if valid else -1,
            'tree_edit/previous_partition_retained': int(retained)})

    def _make_optimizer(self):
        self.optimizer = torch.optim.Adam([
            dict(params=[self.region_translation], lr=self.config.region_translation_lr),
            dict(params=[self.region_rotation], lr=self.config.region_rotation_lr),
            dict(params=[self.gs_translation], lr=self.config.gs_translation_lr),
            dict(params=[self.gs_rotation], lr=self.config.gs_rotation_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)

    @torch.no_grad()
    def mark_observed(self, visible):
        if self.observations is not None:
            visible = torch.as_tensor(visible, device=self.observations.device, dtype=torch.bool)
            if visible.shape != self.observations.shape:
                raise ValueError('Motion visibility must match old GS rows')
            self.observations.add_(visible.long())

    @torch.no_grad()
    def _reset_adam_rows(self, parameter, rows):
        rows = torch.as_tensor(rows, device=parameter.device, dtype=torch.long)
        if not rows.numel():
            return
        state = self.optimizer.state.get(parameter, {})
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if name in state:
                state[name][rows] = 0

    @torch.no_grad()
    def _replace_regions(self, translation, rotation, source_rows):
        """Resize only region parameters, preserving Adam rows for surviving IDs."""
        for index, name, value in ((0, 'region_translation', translation),
                                    (1, 'region_rotation', rotation)):
            old = getattr(self, name)
            parameter = nn.Parameter(value.detach().clone())
            source = torch.as_tensor(source_rows, device=value.device, dtype=torch.long)
            state = self.optimizer.state.pop(old, None)
            self.optimizer.param_groups[index]['params'] = [parameter]
            if state:
                replaced = {}
                keep = source >= 0
                for key, item in state.items():
                    if torch.is_tensor(item) and item.shape == old.shape:
                        padded = item.new_zeros(value.shape)
                        padded[keep] = item[source[keep]]
                        replaced[key] = padded
                    else:
                        replaced[key] = item.clone() if torch.is_tensor(item) else item
                self.optimizer.state[parameter] = replaced
            setattr(self, name, parameter)
        self._region_id_lengths = rigid._id_lengths(len(translation) - 1, self.base_xyz)
        self._metadata_cache = None

    @torch.no_grad()
    def _local_from_global(self, node, parent, pivot, global_q, global_b):
        local_q = rigid.quaternion_multiply(global_q[node], rigid.quaternion_inverse(global_q[parent]))
        vector = rigid.quaternion_rotvec(local_q[None])[0]
        inherited_pivot = rigid.quaternion_rotate(global_q[parent], pivot)
        translation = global_b[node] - global_b[parent] - rigid.rotation_delta(local_q, -inherited_pivot)
        return translation, vector

    @torch.no_grad()
    def _fit_local(self, rows, parent, target_xyz, target_q, pivot):
        q_global, b_global = self._global_regions()
        source = rigid.quaternion_rotate(q_global[parent].expand(len(rows), -1),
                                         self.base_xyz[rows]) + b_global[parent]
        source_q = rigid.quaternion_multiply(q_global[parent].expand(len(rows), -1),
                                             self.base_rotation[rows])
        relative = rigid.quaternion_multiply(target_q[rows],
                                              rigid.quaternion_inverse(source_q))
        weights = self.observations[rows].clamp_min(1).double().cpu().numpy()
        result = fit_shared_motion(source.double().cpu().numpy(),
            target_xyz[rows].double().cpu().numpy(),
            quaternion_matrices(relative.double().cpu().numpy()), weights,
            self.config.clustering.orientation_length_extent * self.extent)
        child_q = torch.as_tensor(result['quaternion'], device=self.base_xyz.device,
                                  dtype=self.base_xyz.dtype)
        offset = torch.as_tensor(result['offset'], device=self.base_xyz.device,
                                 dtype=self.base_xyz.dtype)
        inherited_pivot = rigid.quaternion_rotate(q_global[parent], pivot) + b_global[parent]
        translation = offset - rigid.rotation_delta(child_q, -inherited_pivot)
        vector = rigid.quaternion_rotvec(child_q[None])[0]
        return translation, vector, result

    @torch.no_grad()
    def _proposed_global(self, parent, pivot, translation, vector):
        if (bool((translation / self.steps[:3]).abs().gt(32767.5).any()) or
                bool((vector / self.steps[3:]).abs().gt(32767.5).any())):
            return None
        translation = rigid._integer(translation, self.steps[:3]).to(translation) * self.steps[:3]
        vector = rigid._integer(vector, self.steps[3:]).to(vector) * self.steps[3:]
        q, b = self._global_regions()
        local = rigid.rotation_vector_quaternion(vector)
        global_q = rigid.quaternion_multiply(local, q[parent])
        global_b = b[parent] + rigid.rotation_delta(local, -rigid.quaternion_rotate(q[parent], pivot)) + translation
        return translation, vector, global_q, global_b

    @torch.no_grad()
    def _retarget_rows(self, rows, global_q, global_b, target_xyz, target_q):
        predicted = rigid.quaternion_rotate(global_q.expand(len(rows), -1),
                                            self.base_xyz[rows]) + global_b
        predicted_q = rigid.quaternion_multiply(global_q.expand(len(rows), -1),
                                                self.base_rotation[rows])
        t = target_xyz[rows] - predicted
        r = rigid.quaternion_rotvec(rigid.quaternion_multiply(
            target_q[rows], rigid.quaternion_inverse(predicted_q)))
        if (bool((t / self.gs_steps[:3]).abs().gt(32767.5).any()) or
                bool((r / self.gs_steps[3:]).abs().gt(32767.5).any())):
            return None
        t = rigid._integer(t, self.gs_steps[:3]).to(t) * self.gs_steps[:3]
        r = rigid._integer(r, self.gs_steps[3:]).to(r) * self.gs_steps[3:]
        recovered_q = rigid.quaternion_multiply(rigid.rotation_vector_quaternion(r), predicted_q)
        epsilon = torch.finfo(self.base_xyz.dtype).eps * max(1., self.extent) * 8
        if (bool((predicted + t - target_xyz[rows]).norm(dim=-1).gt(
                np.sqrt(3) * float(self.gs_steps[0]) / 2 + epsilon).any()) or
                bool(rigid.quaternion_angle_error(recovered_q, target_q[rows]).gt(
                    np.sqrt(3) * float(self.gs_steps[3]) / 2 + epsilon).any())):
            return None
        return t, r

    @torch.no_grad()
    def _apply_gs_residuals(self, rows, values):
        for parameter, value in zip((self.gs_translation, self.gs_rotation), values):
            parameter[rows] = value
            self._reset_adam_rows(parameter, rows)

    @torch.no_grad()
    def _split_motion_regions(self, target_xyz, target_q):
        cfg = self.config.clustering
        edits = []
        original_nodes = sorted(self.owner.unique().tolist())
        depth = [0] * len(self.parents)
        for node in rigid._topological_order(self.parents):
            depth[node] = depth[int(self.parents[node])] + 1
        for parent in original_nodes:
            if depth[parent] >= self.config.parent_max_depth:
                continue
            rows = torch.where((self.owner == parent) &
                               (self.observations >= cfg.min_observations))[0]
            if len(rows) < cfg.min_cluster_gs:
                continue
            q, b = self._global_regions()
            source = rigid.quaternion_rotate(q[parent].expand(len(rows), -1), self.base_xyz[rows]) + b[parent]
            source_q = rigid.quaternion_multiply(q[parent].expand(len(rows), -1), self.base_rotation[rows])
            relative = rigid.quaternion_multiply(target_q[rows], rigid.quaternion_inverse(source_q))
            groups = shared_motion_groups(source.double().cpu().numpy(),
                target_xyz[rows].double().cpu().numpy(), relative.double().cpu().numpy(),
                self.observations[rows].double().cpu().numpy(), self.extent, cfg)
            for group in groups:
                selected = rows[torch.as_tensor(group['rows'], device=rows.device)]
                pivot = self.base_xyz[selected].detach().cpu().mean(0).to(self.base_xyz)
                translation, vector, _ = self._fit_local(selected, parent, target_xyz, target_q, pivot)
                rt, rr, _, _ = self._values()
                if not residual_exceeds_parent(
                        translation.cpu().numpy(), vector.cpu().numpy(),
                        rt[parent].cpu().numpy(), rr[parent].cpu().numpy(),
                        float(self.gs_steps[0]) * cfg.split_noise_steps,
                        float(self.gs_steps[3]) * cfg.split_noise_steps):
                    continue
                proposal = self._proposed_global(parent, pivot, translation, vector)
                if proposal is None:
                    continue
                translation, vector, global_q, global_b = proposal
                residual = self._retarget_rows(selected, global_q, global_b, target_xyz, target_q)
                if residual is None:
                    continue
                node = len(self.parents)
                source_rows = torch.cat((torch.arange(node, device=rows.device), rows.new_tensor([-1])))
                self._replace_regions(torch.cat((self.region_translation.detach(), translation[None])),
                                      torch.cat((self.region_rotation.detach(), vector[None])), source_rows)
                self.parents = torch.cat((self.parents, self.parents.new_tensor([parent])))
                self.pivots = torch.cat((self.pivots, pivot[None]))
                self.owner[selected] = node
                self._apply_gs_residuals(selected, residual)
                self._protected_regions.add(node)
                edits.append(dict(kind='split', node=node, parent=parent, members=len(selected)))
        return edits

    @torch.no_grad()
    def _merge_motion_regions(self, target_xyz, target_q):
        cfg = self.config.clustering
        children = set(self.parents[1:].tolist())
        leaves = [node for node in range(1, len(self.parents))
                  if node not in children and bool((self.owner == node).any())]
        if len(leaves) < 2:
            return []
        q, b = self._global_regions()
        features = torch.cat((b[leaves] / (cfg.translation_tolerance_extent * self.extent),
                               rigid.quaternion_rotvec(q[leaves]) / cfg.rotation_tolerance_rad), -1)
        features = features.double().cpu().numpy()
        k = min(len(leaves), cfg.motion_neighbors + 1)
        _, near = cKDTree(features).query(features, k=k, workers=1)
        pairs = sorted({tuple(sorted((leaves[index], leaves[int(candidate)])))
                        for index, candidates in enumerate(np.asarray(near).reshape(len(leaves), k))
                        for candidate in candidates if int(candidate) != index and
                        np.max(np.abs(features[index] - features[int(candidate)])) <= cfg.motion_neighbor_radius})
        removed = set()
        edits = []
        for leader, other in pairs:
            if leader in removed or other in removed or int(self.parents[leader]) != int(self.parents[other]):
                continue
            rows = torch.where((self.owner == leader) | (self.owner == other))[0]
            pivot = self.base_xyz[rows].detach().cpu().mean(0).to(self.base_xyz)
            translation, vector, fit = self._fit_local(rows, int(self.parents[leader]), target_xyz, target_q, pivot)
            if (fit['position_error'].max() > cfg.translation_tolerance_extent * self.extent or
                    fit['rotation_error'].max() > cfg.rotation_tolerance_rad):
                continue
            proposal = self._proposed_global(int(self.parents[leader]), pivot, translation, vector)
            if proposal is None:
                continue
            translation, vector, global_q, global_b = proposal
            residual = self._retarget_rows(rows, global_q, global_b, target_xyz, target_q)
            if residual is None:
                continue
            self.pivots[leader] = pivot
            self.region_translation[leader] = translation
            self.region_rotation[leader] = vector
            self.owner[rows] = leader
            self._apply_gs_residuals(rows, residual)
            self._reset_adam_rows(self.region_translation, [leader])
            self._reset_adam_rows(self.region_rotation, [leader])
            self._protected_regions.add(leader)
            self.region_map[self.region_map == other] = leader
            removed.add(other)
            edits.append(dict(kind='merge', node=other, into=leader, members=len(rows)))
        return edits

    @torch.no_grad()
    def _compact_motion_regions(self):
        used = set(self.owner.tolist()) | ({0} if self.config.residual_hierarchy else {0, 1})
        for node in list(used):
            while node != 0:
                node = int(self.parents[node])
                used.add(node)
        keep = torch.tensor(sorted(used), device=self.owner.device, dtype=torch.long)
        if len(keep) == len(self.parents):
            return
        remap = torch.zeros(len(self.parents), device=self.owner.device, dtype=torch.long)
        remap[keep] = torch.arange(len(keep), device=keep.device)
        self._replace_regions(self.region_translation.detach()[keep],
                              self.region_rotation.detach()[keep], keep)
        self.owner = remap[self.owner]
        self.region_map = remap[self.region_map]
        parents = self.parents[keep].clone()
        parents[1:] = remap[parents[1:]]
        self.parents = parents
        self.pivots = self.pivots[keep]
        self._protected_regions = {int(remap[node]) for node in self._protected_regions if node in used}

    @torch.no_grad()
    def _collapse_identity_leaves(self):
        """Fold a fully inherited leaf into its parent without altering GS state."""
        rt, rr, _, _ = self._values(hard=True)
        removed = set()
        edits = []
        while True:
            parents_with_children = {int(self.parents[node]) for node in range(1, len(self.parents))
                                     if node not in removed}
            first_node = 1 if self.config.residual_hierarchy else 2
            candidates = [node for node in range(first_node, len(self.parents))
                          if node not in removed and node not in parents_with_children and
                          not bool(rt[node].any()) and not bool(rr[node].any())]
            if not candidates:
                break
            for node in candidates:
                parent = int(self.parents[node])
                count = int((self.owner == node).sum())
                self.owner[self.owner == node] = parent
                self.region_map[self.region_map == node] = parent
                removed.add(node)
                if count:
                    edits.append(dict(kind='merge', node=node, into=parent, members=count))
        return edits

    @torch.no_grad()
    def _reparent_motion_regions(self, target_xyz, target_q):
        """Search parents by motion similarity, with a pose-preserving transfer."""
        cfg = self.config.clustering
        nodes = [node for node in range(1, len(self.parents))
                 if bool((self.owner == node).any())]
        if len(nodes) < 2:
            return []
        q, b = self._global_regions()
        feature = torch.cat((b[nodes] / (cfg.translation_tolerance_extent * self.extent),
                             rigid.quaternion_rotvec(q[nodes]) / cfg.rotation_tolerance_rad), -1)
        feature = feature.double().cpu().numpy()
        k = min(len(nodes), cfg.motion_neighbors + 1)
        _, nearby = cKDTree(feature).query(feature, k=k, workers=1)
        edits = []
        for index, node in enumerate(nodes):
            original_parent = int(self.parents[node])
            old_t = self.region_translation[node].detach().clone()
            old_r = self.region_rotation[node].detach().clone()
            old_score = float(old_t.norm()) / self.extent + float(old_r.norm())
            best = None
            for parent in [0] + [nodes[int(value)] for value in np.asarray(nearby).reshape(len(nodes), k)[index]]:
                if parent in (node, original_parent):
                    continue
                trial = self.parents.clone()
                trial[node] = parent
                try:
                    order = rigid._topological_order(trial)
                except ValueError:
                    continue
                depth = [0] * len(trial)
                for child in order:
                    depth[child] = depth[int(trial[child])] + 1
                if max(depth) > self.config.parent_max_depth:
                    continue
                global_q, global_b = self._global_regions()
                t, r = self._local_from_global(node, parent, self.pivots[node], global_q, global_b)
                score = float(t.norm()) / self.extent + float(r.norm())
                if (score >= old_score or
                        float(t.norm()) > float(old_t.norm()) + cfg.translation_tolerance_extent * self.extent or
                        float(r.norm()) > float(old_r.norm()) + cfg.rotation_tolerance_rad):
                    continue
                if (bool((t / self.steps[:3]).abs().gt(32767.5).any()) or
                        bool((r / self.steps[3:]).abs().gt(32767.5).any())):
                    continue
                descendants = {node}
                for child in order:
                    if int(trial[child]) in descendants:
                        descendants.add(child)
                self.parents[node] = parent
                self.region_translation[node] = rigid._integer(t, self.steps[:3]).to(t) * self.steps[:3]
                self.region_rotation[node] = rigid._integer(r, self.steps[3:]).to(r) * self.steps[3:]
                proposed_q, proposed_b = self._global_regions()
                transfers = []
                for child in sorted(descendants):
                    rows = torch.where(self.owner == child)[0]
                    if not len(rows):
                        continue
                    values = self._retarget_rows(rows, proposed_q[child], proposed_b[child], target_xyz, target_q)
                    if values is None:
                        transfers = None
                        break
                    # Reparenting must not create a dense correction side stream.
                    active_before = ((rigid._integer(self.gs_translation[rows], self.gs_steps[:3]) != 0).any(-1) |
                                     (rigid._integer(self.gs_rotation[rows], self.gs_steps[3:]) != 0).any(-1)).sum()
                    active_after = (values[0].ne(0).any(-1) | values[1].ne(0).any(-1)).sum()
                    if active_after > active_before:
                        transfers = None
                        break
                    transfers.append((rows, values))
                chosen_t = self.region_translation[node].detach().clone()
                chosen_r = self.region_rotation[node].detach().clone()
                self.parents[node] = original_parent
                self.region_translation[node] = old_t
                self.region_rotation[node] = old_r
                if transfers is not None and (best is None or score < best[0]):
                    best = (score, parent, chosen_t, chosen_r, transfers)
            if best is None:
                continue
            _, parent, t, r, transfers = best
            self.parents[node] = parent
            self.region_translation[node] = t
            self.region_rotation[node] = r
            self._reset_adam_rows(self.region_translation, [node])
            self._reset_adam_rows(self.region_rotation, [node])
            self._protected_regions.add(node)
            for rows, values in transfers:
                changed = ((rigid._integer(self.gs_translation[rows], self.gs_steps[:3]) !=
                            rigid._integer(values[0], self.gs_steps[:3])).any(-1) |
                           (rigid._integer(self.gs_rotation[rows], self.gs_steps[3:]) !=
                            rigid._integer(values[1], self.gs_steps[3:])).any(-1))
                if bool(changed.any()):
                    self._apply_gs_residuals(rows[changed], (values[0][changed], values[1][changed]))
            edits.append(dict(kind='parent', node=node, parent=parent))
        return edits

    @torch.no_grad()
    def _rebase_bootstrap(self, target_xyz=None, target_q=None):
        """Freeze the learned frame0 layout without installing frame1 GS poses."""
        from .rigid_tree_codec import DecodedState
        if target_xyz is None:
            target_xyz, target_q = (value.detach().clone() for value in self.forward(self.last_step))
        q, b = self._global_regions()
        old_pivots = self.pivots.clone()
        self.pivots = rigid.region_pivots(self.base_xyz, self.owner, self.parents)
        for node in rigid._topological_order(self.parents):
            translation, vector = self._local_from_global(node, int(self.parents[node]), self.pivots[node], q, b)
            translation = rigid._integer(translation, self.steps[:3]).to(translation) * self.steps[:3]
            vector = rigid._integer(vector, self.steps[3:]).to(vector) * self.steps[3:]
            if (not torch.equal(old_pivots[node], self.pivots[node]) or
                    not torch.equal(rigid._integer(self.region_translation[node], self.steps[:3]),
                                    rigid._integer(translation, self.steps[:3])) or
                    not torch.equal(rigid._integer(self.region_rotation[node], self.steps[3:]),
                                    rigid._integer(vector, self.steps[3:]))):
                self.region_translation[node] = translation
                self.region_rotation[node] = vector
                self._reset_adam_rows(self.region_translation, [node])
                self._reset_adam_rows(self.region_rotation, [node])
                self._protected_regions.add(node)
        # Compensate pivot/rounding changes only on GS whose quantized residual
        # changed; other GS Adam rows and sub-grid accumulations remain intact.
        new_q, new_b = self._global_regions()
        for node in self.owner.unique().tolist():
            rows = torch.where(self.owner == node)[0]
            values = self._retarget_rows(rows, new_q[node], new_b[node], target_xyz, target_q)
            if values is None:
                raise ValueError('Initial tree rebase exceeds quantized pose tolerance')
            changed = ((rigid._integer(self.gs_translation[rows], self.gs_steps[:3]) !=
                        rigid._integer(values[0], self.gs_steps[:3])).any(-1) |
                       (rigid._integer(self.gs_rotation[rows], self.gs_steps[3:]) !=
                        rigid._integer(values[1], self.gs_steps[3:])).any(-1))
            if bool(changed.any()):
                self._apply_gs_residuals(rows[changed], (values[0][changed], values[1][changed]))
        reference = DecodedState(self.base_xyz.detach().cpu().float().clone(),
            self.base_rotation.detach().cpu().float().clone(), self.owner.detach().cpu().clone(),
            self.parents.detach().cpu().clone(), 0)
        self.codec_previous_state = reference
        self.frame_reference_state = reference
        self.region_map = torch.arange(len(self.parents), device=self.owner.device)
        self.bootstrap_finalized = True
        self._metadata_cache = None

    @torch.no_grad()
    def _update_motion_clusters(self, step):
        started = time.perf_counter()
        bootstrap = self.frame_index == 1 and not self.bootstrap_finalized
        self._topology_updated = True
        target_xyz, target_q = (value.detach().clone() for value in self.forward(step))
        before = len(self.parents)
        edits = self._split_motion_regions(target_xyz, target_q)
        edits.extend(self._merge_motion_regions(target_xyz, target_q))
        edits.extend(self._reparent_motion_regions(target_xyz, target_q))
        edits.extend(self._collapse_identity_leaves())
        self._compact_motion_regions()
        if self.frame_index == 1 and not self.bootstrap_finalized:
            self._rebase_bootstrap(target_xyz, target_q)
        self.topology_edits.extend(edits)
        self._metadata_cache = None
        after_xyz, after_q = self.forward(step)
        self.last_metrics.update({
            'tree_edit/region_count_before': before,
            'tree_edit/region_count_after': len(self.parents),
            'tree_edit/splits': sum(edit['kind'] == 'split' for edit in edits),
            'tree_edit/merges': sum(edit['kind'] == 'merge' for edit in edits),
            'tree_edit/parent_edits': sum(edit['kind'] == 'parent' for edit in edits),
            'tree_edit/pose_max_xyz_error_world': float((after_xyz - target_xyz).norm(dim=-1).max()),
            'tree_edit/pose_max_rotation_error_rad': float(rigid.quaternion_angle_error(after_q, target_q).max()),
        })
        seconds = time.perf_counter() - started
        self.last_metrics['tree_edit/seconds'] = seconds
        if bootstrap:
            self.bootstrap_seconds = seconds
        return edits

    @torch.no_grad()
    def _warm_motion(self):
        q, b = self._global_regions()
        _, _, translation, rotation = self._values()
        if self.pending_birth_sources is not None and len(self.pending_birth_sources):
            source = self.pending_birth_sources.to(translation.device)
            translation = torch.cat((translation, translation[source]))
            rotation = torch.cat((rotation, rotation[source]))
        return q.detach().clone(), b.detach().clone(), translation.detach().clone(), rotation.detach().clone()

    @torch.no_grad()
    def _initialize_warm_motion(self, warm):
        q, b, translation, rotation = warm
        for node in rigid._topological_order(self.parents):
            t, r = self._local_from_global(node, int(self.parents[node]), self.pivots[node], q, b)
            self.region_translation[node] = t
            self.region_rotation[node] = r
        self.gs_translation.copy_(translation[self.previous_rows])
        self.gs_rotation.copy_(rotation[self.previous_rows])
