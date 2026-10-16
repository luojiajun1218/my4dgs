"""Learn the spatial resolution at which Gaussian motion should be shared.

Each active region predicts its full frame increment. Its GS retain residual
increments. A learned binary gate replaces a region by its octree children;
there is no physical pivot rotation or parent-motion accumulation.
"""

from pathlib import Path
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .rigid_residual_tree import (MotionBase, _id_lengths, _integer,
    _quantized, quaternion_multiply,
    rotation_vector_quaternion)
from .spatial_gate import spatial_layout, spatial_owner, spatial_route


def _rate_relative_representative(symbols, row_costs, region_row_cost,
                                  byte_price, magnitude_scales):
    """Minimize the same block-byte and residual-L1 objective used by gates.

    Byte savings separate into T matches, R matches, and exact joint matches.
    Away from observed block vectors, L1 is minimized by a coordinate median.
    Thus observed joint pairs and four independent zero/nonzero block choices
    suffice; there is no search over all six-dimensional integer vectors.
    """
    symbols = np.asarray(symbols, dtype=np.int32)
    row_costs = np.asarray(row_costs, dtype=np.float64)
    scales = np.asarray(magnitude_scales, dtype=np.float64)
    lower = np.maximum(symbols.max(0) - 32767, -32767)
    upper = np.minimum(symbols.min(0) + 32767, 32767)
    if np.any(lower > upper):
        raise ValueError('No int16 representative and residual exist')

    def block(start):
        values = symbols[:, start:start + 3]
        median = np.clip(np.rint(np.median(values, axis=0)),
                         lower[start:start + 3], upper[start:start + 3]).astype(np.int32)
        candidates, inverse = np.unique(np.concatenate((values, median[None],
            np.zeros((1, 3), dtype=np.int32))), axis=0, return_inverse=True)
        membership = inverse[:len(values)]
        counts = np.bincount(membership, minlength=len(candidates))
        nonzero = candidates.any(1)
        cost = byte_price * (6 * nonzero - 6 * counts)
        for axis in range(3):
            ordered = np.sort(values[:, axis]).astype(np.float64)
            prefix = np.r_[0., np.cumsum(ordered)]
            query = candidates[:, axis].astype(np.float64)
            split = np.searchsorted(ordered, query)
            distance = (query * split - prefix[split] + prefix[-1] - prefix[split] -
                        query * (len(values) - split))
            cost = cost + scales[start + axis] * distance
        valid = ((candidates >= lower[start:start + 3]) &
                 (candidates <= upper[start:start + 3])).all(1)
        cost[~valid] = np.inf
        return candidates, membership, nonzero, cost

    t, ti, tn, tc = block(0)
    r, ri, rn, rc = block(3)
    pairs, inverse = np.unique(np.stack((ti, ri), axis=1), axis=0, return_inverse=True)
    joint_reward = np.bincount(inverse, weights=row_costs, minlength=len(pairs))
    scores = tc[pairs[:, 0]] + rc[pairs[:, 1]] + byte_price * (
        region_row_cost * (tn[pairs[:, 0]] | rn[pairs[:, 1]]) - joint_reward)
    candidates = [np.concatenate((t[pairs[:, 0]], r[pairs[:, 1]]), axis=1)]
    all_scores = [scores]
    for use_t in (False, True):
        for use_r in (False, True):
            it = np.flatnonzero((tn == use_t) & np.isfinite(tc))
            ir = np.flatnonzero((rn == use_r) & np.isfinite(rc))
            if not len(it) or not len(ir):
                continue
            bt = it[np.lexsort((np.abs(t[it]).sum(1), tc[it]))[0]]
            br = ir[np.lexsort((np.abs(r[ir]).sum(1), rc[ir]))[0]]
            candidates.append(np.r_[t[bt], r[br]][None])
            all_scores.append(np.array([tc[bt] + rc[br] +
                byte_price * region_row_cost * (use_t or use_r)]))
    candidates, scores = np.concatenate(candidates), np.concatenate(all_scores)
    best = np.lexsort(((np.abs(candidates) * scales).sum(1),
                       np.abs(candidates).sum(1), scores))[0]
    constant = byte_price * (row_costs.sum() + 12 * len(symbols))
    return candidates[best], float(scores[best] + constant)


@torch.no_grad()
def _retain_bin_remainder(symbols, previous, step):
    """Change integer origins without erasing pending sub-bin optimization.

    At exact half bins, round-to-even depends on the new origin's parity.
    Nudge such ties inward so the fitted integer motion stays unchanged.
    """
    remainder = previous - torch.round(previous / step) * step
    center = symbols * step
    result = center + remainder
    for _ in range(2):
        mismatch = torch.round(result / step).ne(symbols)
        result = torch.where(mismatch, torch.nextafter(result, center), result)
    result = torch.where(torch.round(result / step).ne(symbols), center, result)
    return result.clamp(min=-32767 * step, max=32767 * step)


class SpatialGatedMotion(MotionBase):
    """Learn region increments and GS residuals; gates select spatial resolution."""
    def initialize_reference(self, xyz, rotation, extent, frame_index=0):
        from .rigid_tree_codec import DecodedState
        paths, parents, opened, centers = spatial_layout(
            xyz.detach(), self.config.spatial_gate.max_depth, return_centers=True)
        owner = spatial_owner(paths, opened)
        self.codec_previous_state = DecodedState(
            xyz.detach().cpu().float().clone(), rotation.detach().cpu().float().clone(),
            owner.cpu(), parents.cpu(), frame_index,
            spatial_paths=paths.cpu(), spatial_open=opened.cpu(), spatial_centers=centers)
        return self.codec_previous_state

    def setup(self, xyz, rotation, extent, frame_index=0, previous_rows=None):
        if self.codec_previous_state is None:
            self.initialize_reference(xyz, rotation, extent, frame_index - 1)
        prior = self.codec_previous_state
        if prior.spatial_paths is None or frame_index != prior.frame_index + 1:
            raise ValueError('Spatial motion requires the immediately preceding spatial state')
        rows = (torch.arange(len(xyz)) if previous_rows is None else
                torch.as_tensor(previous_rows).cpu().long())
        if (rows.shape != (len(xyz),) or len(rows.unique()) != len(rows) or
                bool(((rows < 0) | (rows >= len(prior.xyz))).any())):
            raise ValueError('Invalid spatial survivor rows')
        if not torch.allclose(xyz.detach().cpu(), prior.xyz[rows], atol=1e-6, rtol=1e-5):
            raise ValueError('Spatial motion reference differs from the previous decoded frame')
        # Reuse the last applied increment only as the next free-fit starting
        # point. Sharing, gates and residual sparsification are still fitted
        # to the current images. No additional motion field is transmitted.
        proposal = None
        if self.committed and self.frame_index == frame_index - 1:
            with torch.no_grad():
                t, r = self._full_motion_symbols()
                sources = self.pending_birth_sources.to(t.device)
                t = torch.cat((t, t[sources]), dim=0)
                r = torch.cat((r, r[sources]), dim=0)
                if len(t) != len(prior.xyz):
                    raise ValueError('Previous motion proposal lost GS lineage')
                indices = rows.to(t.device)
                # Native ReCon-GS retains .6 of translation; its normalized
                # quaternion update retains the previous orientation increment.
                proposal = (torch.round(.6 * t[indices]).clamp(-32767, 32767),
                            r[indices].clamp(-32767, 32767))
        self.base_xyz, self.base_rotation = prior.xyz[rows].to(xyz), prior.rotation[rows].to(rotation)
        self.extent, self.frame_index = float(extent), int(frame_index)
        self.previous_rows = rows.to(xyz.device)
        self.frame_reference_state = prior
        if prior.spatial_centers is None:
            raise ValueError('Fixed-membership spatial references are replay-only')
        self.spatial_paths = spatial_route(prior.xyz[rows], prior.spatial_centers,
            self.config.spatial_gate.max_depth).to(xyz.device)
        self.parents = prior.parents.to(xyz.device)
        self.opened = prior.spatial_open.to(xyz.device).clone()
        self.owner = spatial_owner(self.spatial_paths, self.opened)
        count = len(self.parents)
        self.node_counts = torch.bincount(self.spatial_paths.flatten(), minlength=count)
        child_counts = torch.zeros(count, dtype=torch.long, device=xyz.device)
        populated = torch.where(self.node_counts > 0)[0]
        child_counts.scatter_add_(0, self.parents[populated], torch.ones_like(populated))
        self.split_eligible = ((self.node_counts >= self.config.spatial_gate.min_region_size) &
                               (child_counts >= 2))
        self.split_eligible[0] = False
        self.opened[1:] &= self.split_eligible[1:]
        logits = xyz.new_full((count,), self.config.spatial_gate.initial_logit)
        if frame_index > 1:
            # Each frame fits new motion with a fresh optimizer. Preserve the
            # previous decisions as a weak prior, not saturated old logits
            # which cannot recover within the current frame's short fit.
            reachable = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
            used_open = torch.zeros_like(self.opened)
            for level in range(self.spatial_paths.shape[1]):
                nodes = self.spatial_paths[:, level]
                used_open[nodes[reachable & self.opened[nodes]]] = True
                reachable &= self.opened[nodes]
            bias = abs(self.config.spatial_gate.initial_logit)
            logits.copy_(torch.where(used_open, logits.new_full((), bias), logits.new_full((), -bias)))
        self.gate_logits = nn.Parameter(logits)
        self.region_translation = nn.Parameter(xyz.new_zeros(count, 3))
        self.region_rotation = nn.Parameter(xyz.new_zeros(count, 3))
        self.gs_translation = nn.Parameter(torch.zeros_like(xyz))
        self.gs_rotation = nn.Parameter(torch.zeros_like(xyz))
        if proposal is not None:
            with torch.no_grad():
                self.gs_translation.copy_(proposal[0].to(xyz) * self.gs_steps[:3])
                self.gs_rotation.copy_(proposal[1].to(xyz) * self.gs_steps[3:])
        self._make_optimizer()
        self.rate_reference_count = self.rate_reference_count or len(xyz)
        self._region_id_lengths = _id_lengths(count - 1, xyz)
        self._gs_id_lengths = _id_lengths(len(xyz), xyz)
        self.observations = torch.zeros(len(xyz), dtype=torch.long, device=xyz.device)
        self.pending_birth_sources = None
        self.topology_edits, self.last_metrics = [], {}
        self.committed = False
        self.last_step = 0
        self._candidate_initialized = False
        self._shared_field_active = False
        self._regional_fit_support = None
        self._coding_initialized = False
        self._hard_gate_frozen = False
        self.flat_region_diagnostics = {'method': 'learned_spatial_gate', 'frame': frame_index}
        self.owner = spatial_owner(self.spatial_paths, self.opened)

    def _make_optimizer(self):
        super()._make_optimizer()
        self.optimizer.add_param_group(dict(params=[self.gate_logits],
            lr=self.config.spatial_gate.learning_rate, eps=1e-12))

    @property
    def shared_prestructure_active(self):
        return True


    @torch.no_grad()
    def _mask_regional_fit_updates(self):
        """Fit observed candidates on the currently selected spatial frontier.

        Observations identify support, including zero motion estimates.
        Image fitting can change a supported region's complete motion. Unselected dictionaries stay
        fixed after hard coding begins, including their Adam momentum.
        """
        support = self._regional_fit_support
        if not self._candidate_initialized or support is None:
            self.region_translation.grad = None
            self.region_rotation.grad = None
            return
        support = support.clone()
        if self._coding_initialized or self._hard_gate_frozen:
            support &= torch.bincount(self.owner, minlength=len(self.parents)).gt(0)
        for parameter in (self.region_translation, self.region_rotation):
            if parameter.grad is not None:
                parameter.grad.mul_(support[:, None])
            state = self.optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key].mul_(support[:, None])

    def _gate_probabilities(self):
        prob = torch.sigmoid(self.gate_logits / self.config.spatial_gate.temperature)
        return prob * self.split_eligible.to(prob)

    def _frontier_probabilities(self):
        return self.opened.to(self.gate_logits) if self._hard_gate_frozen else self._gate_probabilities()

    def _region_selection_probabilities(self, probabilities=None):
        """Chance a node is selected, including all ancestor split decisions."""
        if probabilities is None:
            if not self._candidate_initialized:
                return torch.bincount(self.owner, minlength=len(self.parents)).gt(0).to(self.gate_logits)
            probabilities = self._frontier_probabilities()
        selected = [probabilities.new_zeros(1)]
        start, previous_start, reach = 1, 0, None
        for level in range(self.spatial_paths.shape[1]):
            end = start + 8 ** (level + 1)
            if level == 0:
                reach = probabilities.new_ones(end - start)
            else:
                parents = self.parents[start:end]
                reach = reach[parents - previous_start] * probabilities[parents]
            selected.append(reach * (1 - probabilities[start:end]) * self.node_counts[start:end].gt(0))
            previous_start, start = start, end
        return torch.cat(selected)

    def _path_weights(self, probabilities):
        reach, weights = torch.ones_like(self.gs_translation[:, 0]), []
        for level in range(self.spatial_paths.shape[1]):
            if level == self.spatial_paths.shape[1] - 1:
                weights.append(reach)
            else:
                split = probabilities[self.spatial_paths[:, level]]
                weights.append(reach * (1 - split))
                reach = reach * split
        return torch.stack(weights, -1)

    def _values(self, hard=False, step=None):
        quantize = _integer if hard else _quantized
        rt, rr = quantize(self.region_translation, self.steps[:3]), quantize(self.region_rotation, self.steps[3:])
        gt = quantize(self.gs_translation, self.gs_steps[:3])
        gr = quantize(self.gs_rotation, self.gs_steps[3:])
        active = torch.bincount(self.owner, minlength=len(self.parents)).gt(0)
        active[0] = False
        return rt * active[:, None], rr * active[:, None], gt, gr

    def _shared_motion(self):
        rt = _quantized(self.region_translation, self.steps[:3])
        rr = _quantized(self.region_rotation, self.steps[3:])
        if self._shared_field_active:
            probabilities = self._gate_probabilities()
            soft = self._path_weights(probabilities)
            hard = self._path_weights(probabilities.detach().ge(.5).to(probabilities))
            # Render one actual frontier; differentiate its alternatives.
            # No point residual may compensate for the subdivision decision.
            weights = hard.detach() + (soft - soft.detach())
            return ((weights[..., None] * rt[self.spatial_paths]).sum(1),
                    (weights[..., None] * rr[self.spatial_paths]).sum(1))
        return rt[self.owner], rr[self.owner]

    def forward(self, step, training=False):
        if training:
            self.last_step = int(step)
        shared_t, shared_r = self._shared_motion()
        gt = _quantized(self.gs_translation, self.gs_steps[:3])
        gr = _quantized(self.gs_rotation, self.gs_steps[3:])
        xyz = self.base_xyz + shared_t + gt
        rotation = quaternion_multiply(rotation_vector_quaternion(shared_r + gr), self.base_rotation)
        return xyz, rotation

    def _full_motion_symbols(self):
        shared_t, shared_r = self._shared_motion()
        return (torch.round(shared_t / self.steps[:3]) + torch.round(self.gs_translation / self.gs_steps[:3]),
                torch.round(shared_r / self.steps[3:]) + torch.round(self.gs_rotation / self.gs_steps[3:]))

    @torch.no_grad()
    def _rebase_full_motion(self, full_motion, preserve_remainder=False):
        shared_t, shared_r = self._shared_motion()
        t = full_motion[0] - torch.round(shared_t / self.steps[:3])
        r = full_motion[1] - torch.round(shared_r / self.steps[3:])
        if t.abs().max() > 32767 or r.abs().max() > 32767:
            raise ValueError('Spatial recoding exceeds int16 residual range')
        if preserve_remainder:
            self.gs_translation.copy_(_retain_bin_remainder(t, self.gs_translation, self.gs_steps[:3]))
            self.gs_rotation.copy_(_retain_bin_remainder(r, self.gs_rotation, self.gs_steps[3:]))
        else:
            self.gs_translation.copy_(t * self.gs_steps[:3])
            self.gs_rotation.copy_(r * self.gs_steps[3:])

    def _metadata_bytes(self):
        births = [] if self.pending_birth_sources is None else self.pending_birth_sources.tolist()
        from .rigid_tree_codec import _uvar
        return (4 + 32 + 40 + 16 + 2 + (len(self.parents) + 7) // 8 +
                sum(len(_uvar(value)) for value in
                    (self.frame_index, len(self.parents), len(self.owner), len(births))) +
                sum(len(_uvar(source)) + 28 for source in births))

    def loss_terms(self, step, diagnostics=True):
        rt = _quantized(self.region_translation, self.steps[:3])
        rr = _quantized(self.region_rotation, self.steps[3:])
        # All candidate regions see the currently needed full GS increment.
        # It is temporary training data, never an additional stored GS field.
        shared_t, shared_r = self._shared_motion()
        point_t = _quantized(self.gs_translation, self.gs_steps[:3])
        point_r = _quantized(self.gs_rotation, self.gs_steps[3:])
        total_t = (shared_t + point_t).detach()
        total_r = (shared_r + point_r).detach()
        probabilities = self._frontier_probabilities()
        if step <= self.config.spatial_gate.gate_start_step or step > self.config.spatial_gate.freeze_step:
            probabilities = probabilities.detach()
        weights = self._path_weights(probabilities)
        def support(value, scale):
            # The gate needs the true nonzero block count. It is differentiable
            # through the frontier probabilities, even though symbols are hard.
            return torch.round(value / scale).ne(0).any(-1).to(value.dtype)

        residual_bytes = self.base_xyz.new_zeros(())
        relative = self.base_xyz.new_zeros(())
        for level in range(0 if self._shared_field_active else self.spatial_paths.shape[1]):
            node = self.spatial_paths[:, level]
            st = support(total_t - rt[node], self.gs_steps[:3])
            sr = support(total_r - rr[node], self.gs_steps[3:])
            cost = (st + sr - st * sr) * (self._gs_id_lengths + 1) + (st + sr) * 6
            residual_bytes = residual_bytes + (weights[:, level] * cost).sum()
            # Each candidate must represent the *full* required motion, not
            # an arbitrary component of a soft mixture. Condition on the
            # current shared prediction when fitting point residuals. In a
            # hard frontier this is the actual GS residual. Representatives
            # are initialized statistically before shared-field fitting.
            candidate_t = point_t + shared_t.detach() - rt[node]
            candidate_r = point_r + shared_r.detach() - rr[node]
            magnitude = (candidate_t / self.extent).abs().sum(-1) + candidate_r.abs().sum(-1)
            relative = relative + (weights[:, level] * magnitude).mean()
        # The probability a candidate is selected is the same for every point
        # it covers. Recover it by averaging their path weights, not by charging
        # a region once for each GS.
        regional_weight = self._region_selection_probabilities(probabilities)
        st, sr = support(rt, self.steps[:3]), support(rr, self.steps[3:])
        rid = torch.cat((self._region_id_lengths.new_zeros(1), self._region_id_lengths))
        region_bytes = (regional_weight * ((st + sr - st * sr) * (rid + 1) + (st + sr) * 6)).sum()
        soft_bytes = residual_bytes + region_bytes + self._metadata_bytes()
        if not self._candidate_initialized:
            # Before sharing exists, this is complete motion, not a residual.
            # These proposals are not transmitted or sparsified yet.
            relative = self.base_xyz.new_zeros(())
        ramp = min(max((step - self.config.rate_warmup_steps) / self.config.rate_ramp_steps, 0.), 1.)
        if not self._candidate_initialized:
            # Free point motion supplies proposals, not transmitted parameters.
            ramp = 0.
        rate = soft_bytes / self.rate_normalization_bytes
        result = dict(rate_loss=rate * self.config.rate_loss_weight * ramp,
            relative_motion_loss=relative * self.config.relative_motion_loss_weight * ramp,
            rate_bytes_surrogate=soft_bytes, rate_fraction=rate, relative_motion=relative)
        if diagnostics:
            region_t, region_r, gs_t, gs_r = self._values(hard=True)
            gt, gr = gs_t.ne(0).any(-1), gs_r.ne(0).any(-1)
            rt_active, rr_active = region_t.ne(0).any(-1), region_r.ne(0).any(-1)
            result.update(rate_bytes_hard=self._hard_motion_bytes() + self._metadata_bytes(),
                rate_normalization_gs_count=self.base_xyz.new_tensor(self.rate_normalization_count),
                rate_byte_price=self.base_xyz.new_tensor(self.config.rate_loss_weight * ramp / self.rate_normalization_bytes),
                gs_active_rows=(gt | gr).sum(), gs_translation_active_blocks=gt.sum(), gs_rotation_active_blocks=gr.sum(),
                region_active_nodes=(rt_active | rr_active).sum(),
                spatial_active_regions=self.owner.unique().numel() * self.base_xyz.new_ones(()),
                spatial_open_gates=self.opened[1:].sum(),
                spatial_gate_mean_probability=probabilities[self.split_eligible].mean() if self.split_eligible.any() else self.base_xyz.new_zeros(()))
        return result

    @torch.no_grad()
    def _statistical_representatives(self, groups, symbols, observed):
        if not observed.any():
            return np.empty((0, 7), dtype=np.int32)
        row_costs = self._gs_id_lengths.detach().cpu().numpy() + 1
        # Both gate losses have the same ramp, which cancels in this argmin.
        price = self.config.rate_loss_weight / self.rate_normalization_bytes
        scales = (self.gs_steps.detach().cpu().numpy().astype(np.float64) /
                  np.array([self.extent] * 3 + [1.] * 3) *
                  self.config.relative_motion_loss_weight / len(self.owner))
        order = np.argsort(groups, kind='stable')
        nodes, starts = np.unique(groups[order], return_index=True)
        result = []
        for node, rows in zip(nodes, np.split(order, starts[1:])):
            if node == 0 or not observed[rows].any():
                continue
            rid = 2 + sum(node - 1 >= threshold for threshold in
                          (128, 16384, 2097152, 268435456))
            try:
                best, _ = _rate_relative_representative(
                    symbols[rows], row_costs[rows], rid, price, scales)
            except ValueError:
                continue
            result.append(np.r_[node, best])
        return np.asarray(result, dtype=np.int32).reshape(-1, 7)

    @torch.no_grad()
    def _initialize_candidates(self, full_motion=None):
        target = (torch.cat(full_motion, -1).to(torch.int32) if full_motion is not None else
                  torch.cat((_integer(self.gs_translation, self.gs_steps[:3]),
                             _integer(self.gs_rotation, self.gs_steps[3:])), -1))
        observed = self.observations.gt(0)
        symbols = target.cpu().numpy()
        paths = self.spatial_paths.cpu().numpy()
        observed = observed.cpu().numpy()
        # Initialization is a motion estimate, not a coding decision. A zero
        # estimate must remain learnable; gates decide whether finer regions
        # are useful after their motion is allowed to fit image evidence.
        self._regional_fit_support = torch.zeros(len(self.parents), dtype=torch.bool,
                                                 device=self.base_xyz.device)
        for level in range(self.spatial_paths.shape[1]):
            groups = paths[:, level]
            order = np.argsort(groups, kind='stable')
            nodes, starts = np.unique(groups[order], return_index=True)
            estimates = []
            for node, rows in zip(nodes, np.split(order, starts[1:])):
                if observed[rows].any():
                    median = np.rint(np.median(symbols[rows], axis=0)).clip(-32767, 32767)
                    estimates.append(np.r_[node, median])
            if not estimates:
                continue
            estimates = np.asarray(estimates)
            index = torch.as_tensor(estimates[:, 0], device=self.base_xyz.device,
                                    dtype=torch.long)
            values = torch.as_tensor(estimates[:, 1:], device=self.base_xyz.device,
                                     dtype=self.base_xyz.dtype)
            self.region_translation[index] = values[:, :3] * self.steps[:3]
            self.region_rotation[index] = values[:, 3:] * self.steps[3:]
            self._regional_fit_support[index] = True
        self._candidate_initialized = True
        self._regional_fit_support[0] = False

    @torch.no_grad()
    def _refresh_active_representatives(self):
        """Absorb common fitted residuals into the current cut, preserving pose.

        The cut and all Adam moments stay fixed. Refresh only after it is hard:
        soft estimation predictors do not yet represent one selected region.
        """
        if not self._hard_gate_frozen:
            raise ValueError('Region mode refresh requires a fixed hard frontier')
        full = self._full_motion_symbols()
        symbols = torch.cat(full, -1).to(torch.int32).cpu().numpy()
        owners = self.owner.cpu().numpy()
        observed = self.observations.gt(0).cpu().numpy()
        modes = self._statistical_representatives(owners, symbols, observed)
        if not len(modes):
            return 0
        # A modal value must fit the regional stream and every member residual,
        # including unobserved members. Otherwise keep that region's old origin.
        lower = np.full((len(self.parents), 6), np.iinfo(np.int32).max, dtype=np.int32)
        upper = np.full_like(lower, np.iinfo(np.int32).min)
        np.minimum.at(lower, owners, symbols)
        np.maximum.at(upper, owners, symbols)
        nodes, best = modes[:, 0], modes[:, 1:]
        valid = ((np.abs(best) <= 32767).all(1) &
                 (lower[nodes] - best >= -32767).all(1) &
                 (upper[nodes] - best <= 32767).all(1) & (nodes != 0))
        nodes = torch.from_numpy(nodes[valid].astype(np.int64)).to(self.owner.device)
        best = torch.from_numpy(best[valid].astype(np.float32)).to(self.base_xyz)
        if not len(nodes):
            return 0
        self.region_translation[nodes] = _retain_bin_remainder(
            best[:, :3], self.region_translation[nodes], self.steps[:3])
        self.region_rotation[nodes] = _retain_bin_remainder(
            best[:, 3:], self.region_rotation[nodes], self.steps[3:])
        self._rebase_full_motion(full, preserve_remainder=True)
        return len(nodes)

    @torch.no_grad()
    def _recode_owner(self, new_owner, full_motion=None, preserve_remainder=False):
        """Change the sharing representation without changing a rendered pose."""
        changed = new_owner.ne(self.owner)
        if full_motion is None and not changed.any():
            return
        if full_motion is None:
            shared_t, shared_r = self._shared_motion()
            total_t = torch.round(shared_t / self.steps[:3]) + torch.round(self.gs_translation / self.gs_steps[:3])
            total_r = torch.round(shared_r / self.steps[3:]) + torch.round(self.gs_rotation / self.gs_steps[3:])
        else:
            total_t, total_r = full_motion
            changed = torch.ones_like(changed)
        rt = _integer(self.region_translation, self.steps[:3]).float()
        rr = _integer(self.region_rotation, self.steps[3:]).float()
        residual_t, residual_r = total_t[changed] - rt[new_owner[changed]], total_r[changed] - rr[new_owner[changed]]
        if residual_t.abs().max() > 32767 or residual_r.abs().max() > 32767:
            raise ValueError('Spatial recoding exceeds int16 residual range')
        if preserve_remainder:
            self.gs_translation[changed] = _retain_bin_remainder(
                residual_t, self.gs_translation[changed], self.gs_steps[:3])
            self.gs_rotation[changed] = _retain_bin_remainder(
                residual_r, self.gs_rotation[changed], self.gs_steps[3:])
        else:
            self.gs_translation[changed] = residual_t * self.gs_steps[:3]
            self.gs_rotation[changed] = residual_r * self.gs_steps[3:]
        # Additive recoding leaves the pose and its GS image Jacobian intact;
        # its optimizer history remains valid in the new relative coordinates.
        self.owner = new_owner

    @torch.no_grad()
    def _select_frontier(self, freeze=True, full_motion=None, preserve_remainder=False):
        if (not self._shared_field_active and full_motion is None and
                self._candidate_initialized and not self._coding_initialized and not self._hard_gate_frozen):
            full_motion = self._full_motion_symbols()
        self._coding_initialized = True
        new_open = self._gate_probabilities().ge(0.5)
        new_open[0] = True
        new_owner = spatial_owner(self.spatial_paths, new_open)
        gate_changes = torch.where(new_open != self.opened)[0]
        if not self._shared_field_active:
            self._recode_owner(new_owner, full_motion=full_motion,
                               preserve_remainder=preserve_remainder)
        self._hard_gate_frozen = freeze
        edits = [dict(kind='spatial_gate', node=int(node), opened=bool(new_open[node]),
                      step=self.last_step) for node in gate_changes]
        self.opened, self.owner = new_open, new_owner
        self.topology_edits.extend(edits)
        return edits

    @torch.no_grad()
    def _begin_shared_field_fit(self):
        full = self._full_motion_symbols()
        self._initialize_candidates(full_motion=full)
        for parameter in (self.gs_translation, self.gs_rotation):
            parameter.zero_()
            parameter.grad = None
            parameter.requires_grad_(False)
        self._shared_field_active = True
        self._coding_initialized = True
        return self._select_frontier(freeze=False)

    @torch.no_grad()
    def _begin_point_residual_fit(self):
        if not self._hard_gate_frozen or not self._shared_field_active:
            raise ValueError('Point residuals require the fitted shared field and selected tree')
        self._shared_field_active = False
        # Keep the learned shared field. Reintroducing absolute warm proposals
        # here would overwrite it and create compensating point residuals.
        for parameter in (self.gs_translation, self.gs_rotation):
            parameter.zero_()
            parameter.requires_grad_(True)
            self.optimizer.state.pop(parameter, None)

    @torch.no_grad()
    def optimizer_step(self):
        cfg = self.config.spatial_gate
        self._mask_regional_fit_updates()
        if self.last_step <= cfg.gate_start_step or self._hard_gate_frozen:
            self.gate_logits.grad = None
        factor = self._motion_lr_factor(self.last_step)
        for group, lr in zip(self.optimizer.param_groups, (self.config.region_translation_lr,
                self.config.region_rotation_lr, self.config.gs_translation_lr, self.config.gs_rotation_lr)):
            group['lr'] = lr * factor
        self.optimizer.step()
        for parameter, scale in ((self.region_translation, self.steps[:3]),
                (self.region_rotation, self.steps[3:]), (self.gs_translation, self.gs_steps[:3]),
                (self.gs_rotation, self.gs_steps[3:])):
            parameter.clamp_(min=-32767 * scale, max=32767 * scale)
        self.gate_logits.clamp_(-6, 6)
        initializing = self.last_step == cfg.start_step and not self._candidate_initialized
        edits = []
        if initializing:
            edits = self._begin_shared_field_fit()
        elif self._shared_field_active:
            if self.last_step >= cfg.gate_start_step:
                edits = self._select_frontier(freeze=self.last_step >= cfg.freeze_step)
                if self._hard_gate_frozen:
                    self._begin_point_residual_fit()
        elif self._candidate_initialized and self._motion_fit_window_due():
            if self._hard_gate_frozen:
                self.last_metrics['motion/refreshed_region_modes'] = self._refresh_active_representatives()
            self.last_metrics['motion/prox_zero_region_translation_blocks'] = 0
            self.last_metrics['motion/prox_zero_region_rotation_blocks'] = 0
            super()._proximal_gs_blocks()
        if self.last_step in (cfg.start_step, cfg.gate_start_step, (cfg.start_step + cfg.freeze_step) // 2,
                              cfg.freeze_step, self.motion_steps):
            self._record_gate_learning()
        self.optimizer.zero_grad(set_to_none=True)
        return edits

    @torch.no_grad()
    def _record_gate_learning(self):
        selection = self._region_selection_probabilities()
        rt = _integer(self.region_translation, self.steps[:3]).ne(0).any(-1)
        rr = _integer(self.region_rotation, self.steps[3:]).ne(0).any(-1)
        layers = []
        for level in range(self.spatial_paths.shape[1]):
            nodes = self.spatial_paths[:, level].unique()
            grad = self.region_translation.grad
            layers.append(dict(depth=level + 1, candidates=len(nodes),
                nonzero_motion=int((rt[nodes] | rr[nodes]).sum()),
                mean_selection_probability=float(selection[nodes].mean()),
                translation_gradient_rms=(float(grad[nodes].square().mean().sqrt()) if grad is not None else 0.)))
        self.flat_region_diagnostics.setdefault('learning_trace', []).append(dict(
            step=self.last_step, hard_frozen=self._hard_gate_frozen,
            shared_field_active=self._shared_field_active,
            gate_gradient_norm=(float(self.gate_logits.grad.norm()) if self.gate_logits.grad is not None else 0.),
            coding_initialized=self._coding_initialized,
            top_gate_probabilities=self._gate_probabilities()[1:9].tolist(), layers=layers))

    @torch.no_grad()
    def record_final_diagnostics(self):
        counts = torch.bincount(self.owner, minlength=len(self.parents))
        depth = {int(node): level + 1 for level in range(self.spatial_paths.shape[1])
                 for node in self.spatial_paths[:, level].unique().tolist()}
        active = torch.where(counts > 0)[0]
        # A small residual count alone does not establish correct motion. Keep
        # its distribution beside support and region size for image-error audits.
        gt = _integer(self.gs_translation, self.gs_steps[:3])
        gr = _integer(self.gs_rotation, self.gs_steps[3:])
        residual = gt.ne(0).any(-1) | gr.ne(0).any(-1)
        residual_counts = torch.bincount(self.owner[residual], minlength=len(self.parents))
        observed_counts = torch.bincount(self.owner[self.observations.gt(0)],
                                        minlength=len(self.parents))
        leaf = self.spatial_paths[:, -1]
        leaf_counts = torch.bincount(leaf, minlength=len(self.parents))
        leaf_residual_counts = torch.bincount(leaf[residual], minlength=len(self.parents))
        largest_leaf = int(leaf_counts.argmax())
        deepest = self.spatial_paths.shape[1]
        deepest_nodes = active.new_tensor([int(node) for node in active
                                           if depth[int(node)] == deepest])
        tree_health = dict(
            scope='surviving GS before this frame birth; counts do not measure image correctness',
            point_weighted_depth=sum(depth[int(node)] * int(counts[node]) for node in active) / len(self.owner),
            points_on_deepest_frontier=int(counts[deepest_nodes].sum()),
            residual_points_on_deepest_frontier=int(residual_counts[deepest_nodes].sum()),
            largest_candidate_leaf=dict(node=largest_leaf, count=int(leaf_counts[largest_leaf]),
                local_residual_count=int(leaf_residual_counts[largest_leaf])),
        )
        self.flat_region_diagnostics.update(
            semantics='absolute region increments; orientation-only rotations; GS residuals',
            spatial_assignment='previous_decoded_position_v1',
            region_representative='quantized median initialization; shared field fit; point residual fit',
            region_motion_learning=(self._regional_fit_support is not None and
                                    bool(self._regional_fit_support.any())),
            candidate_nodes=len(self.parents), active_regions=len(active),
            open_gates=int(self.opened[1:].sum()), eligible_gates=int(self.split_eligible.sum()),
            gate_start_step=self.config.spatial_gate.start_step,
            gate_learning_start_step=self.config.spatial_gate.gate_start_step,
            relative_motion_units='extent/radians; expected full-motion candidate residual',
            gate_freeze_step=self.config.spatial_gate.freeze_step,
            active_regions_by_depth={str(level): sum(depth[int(node)] == level for node in active)
                                     for level in range(1, self.spatial_paths.shape[1] + 1)},
            tree_health=tree_health,
            regions=[dict(node=int(node), depth=depth[int(node)], count=int(counts[node]),
                observed_count=int(observed_counts[node]), local_residual_count=int(residual_counts[node]),
                local_residual_fraction=float(residual_counts[node] / counts[node]),
                translation=self.region_translation[node].tolist(), rotation=self.region_rotation[node].tolist())
                for node in active],
            gates=[dict(node=int(node), probability=float(self._gate_probabilities()[node]),
                        opened=bool(self.opened[node])) for node in torch.where(self.split_eligible)[0]])

    def export_motion(self, path):
        from .rigid_tree_codec import encode_motion, decode_motion, motion_packet_info
        if not self.committed:
            raise ValueError('Materialize spatial motion before exporting')
        rt, rr, gt, gr = [value.detach().cpu() for value in self._values(hard=True)]
        packet = encode_motion(self.frame_reference_state, self.frame_index, self.extent,
            self.previous_rows.cpu(), self.owner.cpu(), self.parents.cpu(), rt, rr, gt, gr,
            self.pending_birth_sources, self.materialized_xyz[len(self.owner):].cpu(),
            self.materialized_rotation[len(self.owner):].cpu(),
            self.config.motion_xyz_step_extent, self.config.motion_rotation_step_rad,
            compression='auto', gs_translation_step_extent=self.config.gs_xyz_step_extent,
            gs_rotation_step_rad=self.config.gs_rotation_step_rad, spatial_open=self.opened.cpu())
        decoded = decode_motion(packet, self.frame_reference_state)
        if not torch.allclose(decoded.xyz, self.materialized_xyz.cpu(), atol=1e-6, rtol=1e-5):
            raise ValueError('Spatial stream differs from the quantized training pose')
        dot = (F.normalize(decoded.rotation, dim=-1) * F.normalize(self.materialized_rotation.cpu(), dim=-1)).sum(-1).abs()
        if not bool((dot > 1 - 1e-5).all()):
            raise ValueError('Spatial stream differs from the quantized training orientation')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packet)
        self.codec_previous_state = decoded
        return dict(path=str(path), bytes=len(packet), payload_bytes=len(packet),
            **motion_packet_info(packet), frame_index=self.frame_index, gs_count=len(decoded.xyz),
            region_count=len(self.parents), active_region_count=int(self.owner.unique().numel()),
            region_translation_step_world=float(self.steps[0]), region_rotation_step_rad=float(self.steps[3]),
            gs_translation_step_world=float(self.gs_steps[0]), gs_rotation_step_rad=float(self.gs_steps[3]),
            semantics='spatial gates; absolute region increments; no rotation of GS centers')

    def capture(self):
        packet = super().capture()
        if packet is not None:
            packet['representative_objective'] = 'rate_relative_v1'
            packet['gate_sparsification'] = 'fit_windows'
            packet['regional_image_fitting'] = 'observed_candidates_v2'
            packet['candidate_initialization'] = 'quantized_median_v1'
            packet['motion_schedule'] = 'shared_field_zero_residual_v2'
            packet['motion_initialization'] = 'previous_increment_v1'
            packet['spatial_assignment'] = 'previous_decoded_position_v1'
            packet['shared_field_active'] = self._shared_field_active
            packet['regional_fit_support'] = self._regional_fit_support
            packet['parameters']['gate_logits'] = self.gate_logits.detach().clone()
            packet['tensors'].update({name: getattr(self, name) for name in
                ('spatial_paths', 'opened', 'node_counts', 'split_eligible')})
            packet['candidate_initialized'] = self._candidate_initialized
            packet['coding_initialized'] = self._coding_initialized
            packet['hard_gate_frozen'] = self._hard_gate_frozen
        return packet

    def restore(self, packet):
        if packet is not None and packet.get('spatial_assignment') != 'previous_decoded_position_v1':
            raise ValueError('Fixed-membership spatial checkpoints are replay-only')
        if packet is not None and packet.get('motion_initialization') != 'previous_increment_v1':
            raise ValueError('Earlier zero-initialized motion checkpoints are replay-only')
        if packet is not None and packet.get('motion_schedule') != 'shared_field_zero_residual_v2':
            raise ValueError('Earlier point-first motion schedules are replay-only')
        if packet is not None and packet.get('regional_image_fitting') != 'observed_candidates_v2':
            raise ValueError('Earlier regional image-fitting schedules are replay-only')
        if packet is not None and packet.get('gate_sparsification') != 'fit_windows':
            raise ValueError('Earlier gate sparsification schedules are replay-only')
        if packet is not None and (packet.get('fixed_refit_support') is not None or
                packet['config'].get('post_birth_motion_steps', 0)):
            raise ValueError('Post-birth refinement checkpoints are replay-only')
        if packet is not None and packet.get('representative_objective') != 'rate_relative_v1':
            raise ValueError('Earlier region representative objectives are replay-only')
        if packet is not None and packet.get('candidate_initialization') != 'quantized_median_v1':
            raise ValueError('Earlier candidate initializations are replay-only')
        super().restore(packet)
        self._candidate_initialized = bool(packet and packet.get('candidate_initialized', False))
        self._shared_field_active = bool(packet and packet.get('shared_field_active', False))
        self._regional_fit_support = None if packet is None else packet['regional_fit_support']
        self._coding_initialized = bool(packet and packet.get('coding_initialized', packet.get('hard_gate_frozen', False)))
        self._hard_gate_frozen = bool(packet and packet.get('hard_gate_frozen', False))
        if self._shared_field_active:
            for parameter in (self.gs_translation, self.gs_rotation):
                parameter.requires_grad_(False)
