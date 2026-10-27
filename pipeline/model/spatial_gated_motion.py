"""Learn the spatial resolution at which Gaussian motion should be shared.

By default each selected region predicts its full frame increment. The optional
relative representation retains ancestor increments and stores child corrections.
A learned binary gate selects the spatial frontier; GS retain residual motion.
Rotations change Gaussian orientation, never rotate centers around a pivot.
"""

from pathlib import Path
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .rigid_residual_tree import (MotionBase, _id_lengths, _integer,
    _quantized, quaternion_multiply,
    rotation_vector_quaternion)
from .spatial_gate import spatial_layout, spatial_owner, spatial_state_route, spatial_accumulate


def _hard_value_soft_gradient(hard, soft):
    """Keep discrete forward values while differentiating the soft objective."""
    return hard.detach() + (soft - soft.detach())


def _rate_relative_representative(symbols, row_costs, region_row_cost,
                                  byte_price, magnitude_scales, region_origin=None):
    """Minimize the same block-byte and residual-L1 objective used by gates.

    Byte savings separate into T matches, R matches, and exact joint matches.
    Away from observed block vectors, L1 is minimized by a coordinate median.
    Thus observed joint pairs and four independent zero/nonzero block choices
    suffice; there is no search over all six-dimensional integer vectors.
    """
    symbols = np.asarray(symbols, dtype=np.int32)
    row_costs = np.asarray(row_costs, dtype=np.float64)
    scales = np.asarray(magnitude_scales, dtype=np.float64)
    origin = np.zeros(6, dtype=np.int32) if region_origin is None else np.asarray(region_origin, dtype=np.int32)
    lower = np.maximum(symbols.max(0) - 32767, origin - 32767)
    upper = np.minimum(symbols.min(0) + 32767, origin + 32767)
    if np.any(lower > upper):
        raise ValueError('No int16 representative and residual exist')

    def block(start):
        values = symbols[:, start:start + 3]
        median = np.clip(np.rint(np.median(values, axis=0)),
                         lower[start:start + 3], upper[start:start + 3]).astype(np.int32)
        candidates, inverse = np.unique(np.concatenate((values, median[None],
            origin[None, start:start + 3])), axis=0, return_inverse=True)
        membership = inverse[:len(values)]
        counts = np.bincount(membership, minlength=len(candidates))
        nonzero = (candidates != origin[start:start + 3]).any(1)
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
    @property
    def joint_training(self):
        return self.config.spatial_gate.training_schedule == 'joint'

    @property
    def proximal_gs_regularization(self):
        return self.joint_training and self.config.proximal_gs_regularization

    @property
    def joint_image_gradients(self):
        return self.joint_training and self.config.spatial_gate.joint_image_gradients

    @property
    def relative_regions(self):
        return self.config.spatial_gate.region_motion == 'relative'

    @property
    def hierarchical_private(self):
        return getattr(self, '_frame_hierarchical_private', self.config.spatial_gate.hierarchical_private)

    @property
    def private_full_motion(self):
        return getattr(self, '_frame_private_full_motion', self.config.spatial_gate.private_full_motion)

    @property
    def all_region_proposals(self):
        return self.private_full_motion and not self.hierarchical_private

    @property
    def shared_discovery_active(self):
        return self.config.spatial_gate.shared_discovery and (
            self.last_step <= 100 if self.config.spatial_gate.frame_training_plan else not self._hard_gate_frozen)

    @torch.no_grad()
    def prepare_plan_step(self, step):
        if not self.config.spatial_gate.frame_training_plan:
            return
        self.last_step = int(step)
        gate = step <= 100 or 151 <= step <= 180
        self._hard_gate_frozen = not gate
        for parameter, enabled in ((self.gate_logits, gate),
                (self.gs_translation, step >= 101), (self.gs_rotation, step >= 101 and self.config.rotation_updates)):
            parameter.requires_grad_(enabled)
            if not enabled:
                parameter.grad = None
        self._enforce_rotation_policy()

    def _private_nodes(self):
        return self.spatial_paths[:, -1] if self.hierarchical_private else self.owner

    def _private_origins(self):
        leaf = self._private_nodes()
        return tuple(value[leaf].detach() for value in self._regional_predictions())

    def _private_residuals(self, detach_proposal=False):
        """Coding error of an independent complete-motion proposal."""
        leaf = self._private_nodes()
        values = (self.gs_translation, self.gs_rotation)
        return tuple(_quantized(value.detach() if detach_proposal else value, grid) - prediction[leaf]
                     for value, grid, prediction in zip(values,
                         (self.gs_steps[:3], self.gs_steps[3:]), self._regional_predictions()))

    def _private_symbols(self):
        leaf, support = self._private_nodes(), self._private_support()[:, None]
        result = []
        for proposal, prediction, grid in zip((self.gs_translation, self.gs_rotation),
                self._regional_predictions(hard=True), (self.gs_steps[:3], self.gs_steps[3:])):
            symbols = (_integer(proposal, grid).to(torch.int32) - prediction[leaf].to(torch.int32)) * support
            if bool(((symbols < -32768) | (symbols > 32767)).any()):
                raise ValueError('Private coding residual exceeds int16 range')
            result.append(symbols.to(torch.int16))
        return tuple(result)

    @property
    def soft_image_gates(self):
        return self.config.spatial_gate.image_relaxation == 'soft_anneal'

    @property
    def stochastic_image_gates(self):
        return self.config.spatial_gate.image_relaxation == 'stochastic_ste'

    @property
    def binary_concrete_gates(self):
        return self.stochastic_image_gates and self.shared_image_recode and self.joint_image_gradients

    def _gate_estimator(self):
        if self.binary_concrete_gates:
            return 'recoded_binary_concrete_v2'
        return ('stochastic_binary_path_v1' if self.stochastic_image_gates else
                'soft_field_anneal_v1' if self.soft_image_gates else 'hard_value_soft_path_v1')

    def _initialize_gate_sampler(self):
        # An independent CPU generator keeps data sampling/birth RNG unchanged,
        # makes the state portable, and transfers only one small noise vector.
        self._gate_rng = torch.Generator(device='cpu')
        self._gate_rng.manual_seed((torch.initial_seed() + self.frame_index * 1000003) % (2 ** 63 - 1))
        self._sampled_gate_step, self._sampled_gate_noise = 0, None

    def _sampled_path_weights(self, *, sample_if_needed=True):
        """Reuse one discrete path for every attribute and view of a step."""
        if self._sampled_gate_noise is None or self._sampled_gate_step != self.last_step:
            if not sample_if_needed:
                raise ValueError('Render sampled motion before its joint regional attributes')
            uniform = torch.rand(self.gate_logits.shape, generator=self._gate_rng).clamp_(1e-6, 1 - 1e-6)
            self._sampled_gate_noise = (uniform.log() - torch.log1p(-uniform)).to(self.gate_logits)
            self._sampled_gate_step = self.last_step
        draw = ((self.gate_logits + self._sampled_gate_noise) / self._gate_temperature()
                if self.binary_concrete_gates else
                self.gate_logits / self._gate_temperature() + self._sampled_gate_noise)
        soft = self._path_weights(draw.sigmoid() * self.split_eligible)
        hard = self._path_weights((draw.ge(0) & self.split_eligible).to(self.gate_logits))
        return hard.detach() + (soft - soft.detach())

    def _stochastic_image_motion(self):
        weights = self._sampled_path_weights()
        result = []
        for region, proposal, grid in zip(self._regional_predictions(),
                (self.gs_translation, self.gs_rotation), (self.gs_steps[:3], self.gs_steps[3:])):
            private = _quantized(proposal, grid)
            experts = (torch.cat((region[self.spatial_paths[:, :-1]], private[:, None]), 1)
                       if self.private_full_motion else region[self.spatial_paths])
            # Parameter gradients use the actually sampled expert; only the
            # discrete selection uses the relaxed path derivative.
            field = (weights[..., None] * experts).sum(1)
            if not self.private_full_motion:
                field = field + (weights[:, -1:] * private if self.hierarchical_private else private)
            result.append(field)
        return tuple(result)

    @property
    def soft_image_active(self):
        return self.soft_image_gates and not self._hard_gate_frozen

    def _private_support(self):
        """All GS may correct motion unless hierarchical permission is enabled."""
        if not self.hierarchical_private:
            return torch.ones_like(self.owner, dtype=torch.bool)
        return self.owner.eq(self.spatial_paths[:, -1])

    def _private_probabilities(self):
        if not self.hierarchical_private:
            return torch.ones_like(self.gs_translation[:, 0])
        if self._hard_gate_frozen:
            return self._private_support().to(self.gs_translation)
        return self._gate_probabilities()[self.spatial_paths[:, :-1]].prod(-1)

    def _gs_motion(self, hard=False, relaxed=False):
        if self.private_full_motion:
            symbols = self._private_symbols()
            if hard:
                return symbols
            values = (_quantized(self.gs_translation, self.gs_steps[:3]),
                      _quantized(self.gs_rotation, self.gs_steps[3:]))
            probability = self._private_probabilities()[:, None]
            soft = tuple(value * probability for value in values)
            if relaxed:
                return soft
            return tuple(_hard_value_soft_gradient(symbol.to(value) * grid, value)
                         for symbol, value, grid in zip(symbols, soft,
                             (self.gs_steps[:3], self.gs_steps[3:])))
        quantize = _integer if hard else _quantized
        values = (quantize(self.gs_translation, self.gs_steps[:3]),
                  quantize(self.gs_rotation, self.gs_steps[3:]))
        if not self.hierarchical_private:
            return values
        mask = self._private_support()[:, None]
        if hard or self._hard_gate_frozen:
            return tuple(value * mask for value in values)
        probability = self._private_probabilities()[:, None]
        if relaxed:
            return tuple(value * probability for value in values)
        return tuple(_hard_value_soft_gradient(value * mask, value * probability) for value in values)

    @torch.no_grad()
    def _finalize_private_support(self):
        """Freeze physical capacity and start moments for its new image Jacobian."""
        support = self._private_support()
        for parameter in (self.gs_translation, self.gs_rotation):
            parameter[~support] = 0
        # Before freezing, GS/candidate image gradients had soft path weights.
        # Afterwards selected parameters receive the full physical derivative.
        for parameter in (self.region_translation, self.region_rotation,
                          self.gs_translation, self.gs_rotation):
            self.optimizer.state.pop(parameter, None)

    @torch.no_grad()
    def _begin_fixed_gate_residuals(self):
        """Retain encoded motion while making shared leaf fields physical again."""
        if not self._hard_gate_frozen or not self.private_full_motion:
            raise ValueError('Residual coordinates require newly frozen full-motion proposals')
        support = self._private_support()
        symbols = self._private_symbols()
        for parameter, integer, grid in zip((self.gs_translation, self.gs_rotation),
                symbols, (self.gs_steps[:3], self.gs_steps[3:])):
            parameter.copy_(_retain_bin_remainder(integer.to(parameter), parameter, grid))
            parameter[~support] = 0
        self._frame_private_full_motion = False
        self.flat_region_diagnostics['post_gate_parameterization'] = dict(
            step=self.last_step, mode='shared_plus_private_residual', preserves_encoded_motion=True)

    @property
    def shared_image_recode(self):
        return self.joint_training and self.config.spatial_gate.shared_image_recode

    @property
    def gs_sparsity_active(self):
        return (not self.config.defer_gs_sparsity_until_gate_freeze or
                self.last_step > self.config.spatial_gate.freeze_step)

    @property
    def amplitude_refit_active(self):
        return getattr(self, '_amplitude_refit_support', None) is not None

    @property
    def regional_image_updates(self):
        return (self.amplitude_refit_active or self.shared_image_recode or
                (self.config.spatial_gate.predictor_objective == 'proximal_rd' and self._hard_gate_frozen))

    @torch.no_grad()
    def _prepare_amplitude_refit(self, step):
        count = self.config.amplitude_refit_steps
        if not count or self.amplitude_refit_active or step <= self.motion_steps - count:
            return
        if not self._hard_gate_frozen:
            raise ValueError('Freeze learned gates before motion amplitude refitting')
        self._amplitude_refit_support = {}
        stored = self._stored_region_support()
        for name, grid in (('region_translation', self.steps[:3]),
                ('region_rotation', self.steps[3:]), ('gs_translation', self.gs_steps[:3]),
                ('gs_rotation', self.gs_steps[3:])):
            parameter = getattr(self, name)
            support = torch.round(parameter / grid).ne(0).any(-1)
            if name.startswith('region_'):
                support &= stored
                # Unselected dictionary entries remain available next frame.
                parameter[stored & ~support] = 0
                self.optimizer.state.pop(parameter, None)
            else:
                parameter[~support] = 0
                for key in ('exp_avg', 'exp_avg_sq'):
                    state = self.optimizer.state.get(parameter, {})
                    if key in state:
                        state[key].mul_(support[:, None])
            self._amplitude_refit_support[name] = support

    @torch.no_grad()
    def _mask_amplitude_refit_updates(self):
        for name, support in self._amplitude_refit_support.items():
            parameter = getattr(self, name)
            if parameter.grad is not None:
                parameter.grad.mul_(support[:, None])
            for key in ('exp_avg', 'exp_avg_sq'):
                state = self.optimizer.state.get(parameter, {})
                if key in state:
                    state[key].mul_(support[:, None])

    def _regional_predictions(self, hard=False):
        if not self.relative_regions:
            quantize = _integer if hard else _quantized
            return (quantize(self.region_translation, self.steps[:3]),
                    quantize(self.region_rotation, self.steps[3:]))
        result = []
        for parameter, step in ((self.region_translation, self.steps[:3]),
                                (self.region_rotation, self.steps[3:])):
            symbols = (_integer(parameter, step).to(torch.int32) if hard else
                       _quantized(parameter / step, 1.))
            # The codec's sentinel root is an identity, never an unpriced
            # global parameter receiving every descendant's image gradient.
            symbols = torch.cat((torch.zeros_like(symbols[:1]), symbols[1:]))
            total = spatial_accumulate(symbols, self.parents)
            result.append(total if hard else total * step)
        return tuple(result)

    def _stored_region_support(self):
        """Relative packets retain the selected frontier and its ancestors."""
        active = torch.bincount(self.owner, minlength=len(self.parents)).gt(0)
        if self.relative_regions:
            # The histogram already identifies every occupied node. Sorting
            # all GS owners again is unnecessary, especially at a fixed cut.
            nodes = active.nonzero(as_tuple=True)[0]
            for _ in range(self.spatial_paths.shape[1]):
                active[nodes] = True
                nodes = self.parents[nodes].clamp_min(0)
        active[0] = False
        return active

    @torch.no_grad()
    def enable_relative_regions(self, *, inherit_unused=False):
        """Preserve applied motion; optionally initialize unused children by inheritance."""
        if self.relative_regions:
            return
        image_refinement = (self.stochastic_image_gates and
                            ((self.config.spatial_gate.private_full_motion and
                              self.config.spatial_gate.reparameterize_after_gate_freeze) or
                             not self.config.spatial_gate.hierarchical_private))
        if (not self.committed or not self.joint_training or self.shared_image_recode or
                (self.joint_image_gradients and not image_refinement)):
            raise ValueError('Migrate a finalized joint full-motion checkpoint only')
        full = tuple(value.clone() for value in self._full_motion_symbols())
        used = torch.bincount(self.owner, minlength=len(self.parents)).gt(0)
        for _ in range(self.spatial_paths.shape[1]):
            used[self.parents[used].clamp_min(0)] = True
        converted = []
        for parameter, step in ((self.region_translation, self.steps[:3]),
                                (self.region_rotation, self.steps[3:])):
            absolute = _integer(parameter, step).to(torch.int32)
            local = absolute - absolute[self.parents]
            local[0] = 0  # The identity root has sentinel parent -1.
            if inherit_unused:
                # Unused children start at their parent's fitted field. A
                # stale absolute zero must not become its negative correction.
                local[~used] = 0
            if bool(local.abs().gt(32767).any()):
                raise ValueError('Relative conversion exceeds the int16 correction range')
            converted.append(_retain_bin_remainder(local, parameter - parameter[self.parents], step))
            converted[-1][0] = 0
        self.region_translation.copy_(converted[0])
        self.region_rotation.copy_(converted[1])
        self.config.spatial_gate.region_motion = 'relative'
        self._codec_rate = None
        assert all(torch.equal(a, b) for a, b in zip(full, self._full_motion_symbols()))

    def initialize_reference(self, xyz, rotation, extent, frame_index=0):
        from .rigid_tree_codec import DecodedState
        paths, parents, opened, centers, routing = spatial_layout(
            xyz.detach(), self.config.spatial_gate.max_depth, return_centers=True,
            initialization=self.config.spatial_gate.initialization,
            distance_quantile=self.config.spatial_gate.initialization_distance_quantile,
            outlier_assignment=self.config.spatial_gate.outlier_assignment,
            return_routing=True)
        owner = spatial_owner(paths, opened)
        self.codec_previous_state = DecodedState(
            xyz.detach().cpu().float().clone(), rotation.detach().cpu().float().clone(),
            owner.cpu(), parents.cpu(), frame_index,
            spatial_paths=paths.cpu(), spatial_open=opened.cpu(), spatial_centers=centers,
            **routing)
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
        previous_regions = previous_gates = previous_residuals = None
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
                if self.joint_training:
                    proposal = (t[indices], r[indices])
                    previous_regions = (self.region_translation.detach().clone(),
                                        self.region_rotation.detach().clone())
                    previous_gates = self.gate_logits.detach().clone()
                    previous_residuals = tuple(torch.cat((value.detach(), value.detach()[sources]))[indices]
                        for value in (self.gs_translation, self.gs_rotation))
                else:
                    proposal = (torch.round(.6 * t[indices]).clamp(-32767, 32767),
                                r[indices].clamp(-32767, 32767))
        # A frame-boundary migration must read the preceding frame's applied
        # motion under its original policy before enabling the new capacity.
        self._frame_hierarchical_private = self.config.spatial_gate.hierarchical_private
        self._frame_private_full_motion = self.config.spatial_gate.private_full_motion
        self.base_xyz, self.base_rotation = prior.xyz[rows].to(xyz), prior.rotation[rows].to(rotation)
        self.extent, self.frame_index = float(extent), int(frame_index)
        self.previous_rows = rows.to(xyz.device)
        self.frame_reference_state = prior
        self._survivor_mapping_bytes = self._count_survivor_mapping_bytes()
        if prior.spatial_centers is None:
            raise ValueError('Fixed-membership spatial references are replay-only')
        self.spatial_paths = spatial_state_route(prior.xyz[rows], prior,
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
        if self.joint_training and previous_gates is not None:
            logits.copy_(previous_gates.to(logits) * self.config.spatial_gate.temporal_logit_decay)
        elif frame_index > 1:
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
        self._codec_rate = None
        self.committed = False
        self.last_step = 0
        if self.stochastic_image_gates:
            self._initialize_gate_sampler()
        self._candidate_initialized = False
        self._shared_field_active = False
        self._regional_fit_support = None
        self._coding_initialized = False
        self._hard_gate_frozen = False
        self._amplitude_refit_support = None
        self._predictor_fit_state = None
        self.flat_region_diagnostics = {'method': 'learned_spatial_gate', 'frame': frame_index}
        self.owner = spatial_owner(self.spatial_paths, self.opened)
        if self.joint_training:
            self._begin_joint_frame(proposal, previous_regions, previous_residuals)
        self._enforce_rotation_policy()

    @torch.no_grad()
    def _enforce_rotation_policy(self):
        """Freeze frame-to-frame orientation increments, keeping the base pose.

        Applied after warm starts so an inherited increment cannot leak into
        a translation-only frame. Newborn attributes retain their normal fit.
        """
        if self.config.rotation_updates:
            return
        for parameter in (self.region_rotation, self.gs_rotation):
            parameter.zero_()
            parameter.grad = None
            parameter.requires_grad_(False)
            self.optimizer.state.pop(parameter, None)

    @torch.no_grad()
    def _begin_joint_frame(self, proposal, previous_regions, previous_residuals):
        """Fit from step one, retaining the previous frame's actual increments."""
        if self.frame_index == 1:
            # The initial logit's sign selects the first hard frontier. A
            # negative value starts with the eight coarse regions; later
            # frames keep their inherited decisions and logits.
            self.opened = self._gate_probabilities().ge(.5)
            self.opened[0] = True
        self.owner = spatial_owner(self.spatial_paths, self.opened)
        if previous_regions is not None:
            for parameter, previous in zip((self.region_translation, self.region_rotation), previous_regions):
                if parameter.shape != previous.shape:
                    raise ValueError('Warm-started joint motion requires the same candidate tree')
                parameter.copy_(previous.to(parameter))
        if proposal is not None:
            rt, rr = (value.to(self.gs_translation) for value in self._regional_predictions(hard=True))
            residuals = (proposal if self.private_full_motion else
                         (proposal[0].to(rt) - rt[self.owner], proposal[1].to(rr) - rr[self.owner]))
            for parameter, symbols, remainder, step in zip(
                    (self.gs_translation, self.gs_rotation), residuals,
                    previous_residuals, (self.gs_steps[:3], self.gs_steps[3:])):
                if bool(symbols.abs().gt(32767).any()):
                    raise ValueError('Joint warm-start exceeds the int16 residual range')
                parameter.copy_(_retain_bin_remainder(symbols, remainder.to(parameter), step))
        self._candidate_initialized = self._coding_initialized = True
        self._shared_field_active = True
        self._regional_fit_support = self.node_counts.gt(0)
        self._regional_fit_support[0] = False
        self.flat_region_diagnostics.update(training_schedule='joint',
            temporal_initialization='previous_full_increment' if proposal is not None else 'zero',
            initial_gates=(('all_eligible_open' if self.opened[1:].any() else 'all_eligible_closed')
                           if self.frame_index == 1 else 'previous_frame'))
        if self.config.spatial_gate.frame_training_plan:
            # Only previously transmitted relative corrections are inherited.
            # Newly available children start at their parent, once per frame.
            prior = self.frame_reference_state
            used = torch.bincount(prior.owner.to(self.owner.device), minlength=len(self.parents)).gt(0)
            nodes = used.nonzero(as_tuple=True)[0]
            for _ in range(self.spatial_paths.shape[1]):
                used[nodes] = True
                nodes = self.parents[nodes].clamp_min(0)
            self.region_translation[~used] = 0
            self.region_rotation[~used] = 0
        if self.config.spatial_gate.shared_discovery:
            # Start from inherited regional predictions. Point moments learn
            # image sensitivity while zero learning rates hold E at zero.
            self.gs_translation.zero_()
            self.gs_rotation.zero_()
        if self.config.spatial_gate.frame_training_plan:
            self.prepare_plan_step(1)

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
        if self._hard_gate_frozen or (self._coding_initialized and not self.joint_training):
            support &= self._stored_region_support()
        for parameter in (self.region_translation, self.region_rotation):
            if parameter.grad is not None:
                parameter.grad.mul_(support[:, None])
            state = self.optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key].mul_(support[:, None])
            if (self.shared_image_recode and self.joint_image_gradients and
                    not self._hard_gate_frozen and parameter.grad is not None and
                    'exp_avg' in state):
                # These predictors fit only actual main/auxiliary images.
                # A sampled-out row must not move on its old velocity; retain
                # its variance estimate for its next observed image gradient.
                participating = parameter.grad.ne(0).any(-1)
                state['exp_avg'].mul_(participating[:, None])

    def _gate_temperature(self):
        cfg = self.config.spatial_gate
        if not (self.soft_image_gates or self.stochastic_image_gates):
            return cfg.temperature
        progress = min(max((self.last_step - 1) / max(cfg.freeze_step - 1, 1), 0.), 1.)
        return cfg.temperature * (cfg.final_temperature / cfg.temperature) ** progress

    def _gate_probabilities(self):
        # Temperature controls the relaxed derivative, while the actual
        # Bernoulli probability remains sigmoid(logit) in the recoding mode.
        prob = torch.sigmoid(self.gate_logits if self.binary_concrete_gates else
                             self.gate_logits / self._gate_temperature())
        return prob * self.split_eligible.to(prob)

    def _frontier_probabilities(self):
        return self.opened.to(self.gate_logits) if self._hard_gate_frozen else self._gate_probabilities()

    def _region_selection_probabilities(self, probabilities=None, include_ancestors=False):
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
            stop = 1. if include_ancestors else 1 - probabilities[start:end]
            selected.append(reach * stop * self.node_counts[start:end].gt(0))
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
        gt, gr = self._gs_motion(hard=hard)
        active = self._stored_region_support()
        return rt * active[:, None], rr * active[:, None], gt, gr

    def _shared_motion(self, relaxed=False):
        rt, rr = self._regional_predictions()
        if self.all_region_proposals:
            # U is the image variable. E=Q(U)-P is its actual coding residual;
            # the predictor has a separate real image objective below.
            return rt[self.owner].detach(), rr[self.owner].detach()
        if self.regional_image_updates:
            return rt[self.owner], rr[self.owner]
        if self.joint_image_gradients:
            if self._hard_gate_frozen:
                if self.private_full_motion:
                    coarse = ~self._private_support()[:, None]
                    return tuple(_hard_value_soft_gradient(value[self.owner], value[self.owner] * coarse)
                                 for value in (rt, rr))
                return rt[self.owner], rr[self.owner]
            soft = self._path_weights(self._gate_probabilities())
            if self.private_full_motion:
                # The last alternative is the independent full GS proposal.
                # Leaf region vectors fit coding residuals, not image error.
                soft = torch.cat((soft[:, :-1], torch.zeros_like(soft[:, -1:])), -1)
            if relaxed:
                # Each candidate now affects the image used to train it.
                # Materialization/encoding always use the discrete field.
                return tuple((soft[..., None] * value[self.spatial_paths]).sum(1)
                             for value in (rt, rr))
            hard = self._path_weights(self.opened.to(soft))
            # Relax the complete field. Applying STE only to its weights
            # leaves every hard-closed candidate's image Jacobian at zero.
            # The rendered cut remains binary; all candidate fields now fit
            # the same image evidence through their soft path probabilities.
            return tuple(_hard_value_soft_gradient(
                (hard[..., None] * value[self.spatial_paths]).sum(1),
                (soft[..., None] * value[self.spatial_paths]).sum(1)) for value in (rt, rr))
        if self.joint_training:
            # Joint coding uses full-motion optimization coordinates. Region
            # changes are compensated in the GS residual after Adam, just as
            # gate switches are. Images therefore train the full GS increment;
            # every candidate and its gate learn from the residual/rate loss.
            # A fictitious image derivative through the uncompensated shared
            # field would reward the current representation's initialization.
            return rt[self.owner].detach(), rr[self.owner].detach()
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

    @property
    def proposal_image_weight(self):
        return (self.config.spatial_gate.proposal_image_weight
                if self.private_full_motion and not self._hard_gate_frozen else 0.)

    def proposal_pose(self):
        """Fit complete proposals even when the sampled gate chooses sharing.

        This training-only field has no regional or gate dependence. The
        final pose and wire packet always use the learned discrete frontier.
        """
        if not self.private_full_motion or self._hard_gate_frozen:
            raise ValueError('Proposal images require unfrozen complete-motion coordinates')
        xyz = self.base_xyz + _quantized(self.gs_translation, self.gs_steps[:3])
        if not self.config.rotation_updates:
            return xyz, self.base_rotation
        vector = _quantized(self.gs_rotation, self.gs_steps[3:])
        return xyz, quaternion_multiply(rotation_vector_quaternion(vector), self.base_rotation)

    @property
    def predictor_image_weight(self):
        return (self.config.spatial_gate.predictor_image_weight
                if (self.all_region_proposals or self.shared_image_recode) and not self._hard_gate_frozen else 0.)

    def predictor_pose(self):
        """Render one actual sampled shared predictor, without GS corrections."""
        if not self.predictor_image_weight:
            raise ValueError('Predictor images require unfrozen all-region proposals')
        weights = self._sampled_path_weights(sample_if_needed=False)
        t, r = ((weights[..., None] * value[self.spatial_paths]).sum(1)
                for value in self._regional_predictions())
        rotation = (quaternion_multiply(rotation_vector_quaternion(r), self.base_rotation)
                    if self.config.rotation_updates else self.base_rotation)
        return self.base_xyz + t, rotation

    def forward(self, step, training=False):
        if training:
            self.last_step = int(step)
            self.prepare_plan_step(step)
            self._prepare_amplitude_refit(step)
        if training and self.shared_discovery_active:
            translation, rotation_vector = self._stochastic_image_motion()
            xyz = self.base_xyz + translation
        elif self.all_region_proposals or self.predictor_image_weight and self.shared_image_recode:
            if training:
                self._sampled_path_weights()
            shared_t, shared_r = self._shared_motion()
            gt, gr = self._gs_motion()
            xyz = self.base_xyz + shared_t + gt
            rotation_vector = shared_r + gr
        elif (training and self.stochastic_image_gates and not self._hard_gate_frozen
              and not self.config.spatial_gate.frame_training_plan):
            translation, rotation_vector = self._stochastic_image_motion()
            xyz = self.base_xyz + translation
        else:
            relaxed = training and self.soft_image_active
            shared_t, shared_r = self._shared_motion(relaxed=relaxed)
            gt, gr = self._gs_motion(relaxed=relaxed)
            xyz = self.base_xyz + shared_t + gt
            rotation_vector = shared_r + gr
        if not self.config.rotation_updates:
            return xyz, self.base_rotation
        rotation = quaternion_multiply(rotation_vector_quaternion(rotation_vector), self.base_rotation)
        return xyz, rotation

    def _full_motion_symbols(self):
        shared_t, shared_r = self._shared_motion()
        if self.private_full_motion:
            gt, gr = self._private_symbols()
            return (torch.round(shared_t / self.steps[:3]) + gt.to(torch.int32),
                    torch.round(shared_r / self.steps[3:]) + gr.to(torch.int32))
        gt = torch.round(self.gs_translation / self.gs_steps[:3])
        gr = torch.round(self.gs_rotation / self.gs_steps[3:])
        if self.hierarchical_private:
            mask = self._private_support()[:, None]
            gt, gr = gt * mask, gr * mask
        return (torch.round(shared_t / self.steps[:3]) + gt,
                torch.round(shared_r / self.steps[3:]) + gr)

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

    @torch.no_grad()
    def retain_survivors(self, rows):
        """Retire fitted low-alpha GS before births, preserving previous IDs."""
        if self.committed:
            raise ValueError('Cannot retire GS after motion packet finalization')
        rows = torch.as_tensor(rows, device=self.base_xyz.device, dtype=torch.long)
        count = len(self.owner)
        if (rows.ndim != 1 or not len(rows) or
                bool(((rows < 0) | (rows >= count)).any()) or
                bool((rows[1:] <= rows[:-1]).any())):
            raise ValueError('Motion survivors must be nonempty ordered row IDs')
        for name in ('owner', 'base_xyz', 'base_rotation', 'previous_rows',
                     'observations', 'spatial_paths'):
            setattr(self, name, getattr(self, name)[rows].clone())
        for name, group_index in (('gs_translation', 2), ('gs_rotation', 3)):
            previous = getattr(self, name)
            parameter = nn.Parameter(previous.detach()[rows].clone(),
                                     requires_grad=previous.requires_grad)
            state = self.optimizer.state.pop(previous, None)
            self.optimizer.param_groups[group_index]['params'] = [parameter]
            if state is not None:
                self.optimizer.state[parameter] = {
                    key: value[rows].clone() if torch.is_tensor(value) and value.shape == previous.shape else value
                    for key, value in state.items()}
            setattr(self, name, parameter)
        if self.amplitude_refit_active:
            for name in ('gs_translation', 'gs_rotation'):
                self._amplitude_refit_support[name] = self._amplitude_refit_support[name][rows].clone()
        if getattr(self, '_predictor_fit_state', None) is not None:
            self._predictor_fit_state = {name: value[rows].clone()
                                        for name, value in self._predictor_fit_state.items()}
        self.node_counts = torch.bincount(self.spatial_paths.flatten(), minlength=len(self.parents))
        self._gs_id_lengths = _id_lengths(len(rows), self.base_xyz)
        self._survivor_mapping_bytes = self._count_survivor_mapping_bytes()

    def _count_survivor_mapping_bytes(self):
        """Cache the actual deletion-gap section once per frame or restore."""
        from .rigid_tree_codec import _uvar
        count = len(self.frame_reference_state.xyz)
        rows = self.previous_rows.detach().cpu().long()
        if len(rows) == count:
            return 1
        deleted = torch.ones(count, dtype=torch.bool)
        deleted[rows] = False
        ids = deleted.nonzero(as_tuple=True)[0].tolist()
        previous = -1
        size = 1 + len(_uvar(len(ids)))
        for row in ids:
            size += len(_uvar(row - previous - 1))
            previous = row
        return size

    def _metadata_bytes(self):
        births = [] if self.pending_birth_sources is None else self.pending_birth_sources.tolist()
        from .rigid_tree_codec import _uvar
        return (4 + 32 + 40 + 16 + 1 + self._survivor_mapping_bytes +
                (len(self.parents) + 7) // 8 +
                sum(len(_uvar(value)) for value in
                    (self.frame_index, len(self.parents), len(self.owner), len(births))) +
                sum(len(_uvar(source)) + 28 for source in births))

    def regularization_loss(self, step):
        """Camera-independent training penalty; free proposals have zero price."""
        if not self._candidate_initialized or self.amplitude_refit_active:
            return self.base_xyz.new_zeros(())
        terms = self.loss_terms(step, diagnostics=False)
        return terms['rate_loss'] + terms['relative_motion_loss'] + terms.get('candidate_motion_loss', 0.)

    @torch.no_grad()
    def _measure_codec_rate(self, step, force=False):
        """Encode the current old-point update, without unknown future births."""
        from .motion_rate import MotionCodecRate
        from .rigid_tree_codec import encode_motion
        rate = getattr(self, '_codec_rate', None)
        rebuild = rate is None or rate.points != len(self.owner)
        due = (force or rebuild or step <= self.config.codec_rate_initial_steps
               or step in (1, self.config.spatial_gate.freeze_step, self.motion_steps)
               or step % self.config.codec_rate_interval == 0)
        if not due and rate.tables is not None:
            return None, rate.last_scale
        rt, rr, gt, gr = [value.cpu() for value in self._values(hard=True)]
        if rebuild:
            empty = encode_motion(self.frame_reference_state, self.frame_index, self.extent,
                self.previous_rows.cpu(), self.owner.cpu(), self.parents.cpu(),
                torch.zeros_like(rt), torch.zeros_like(rr), torch.zeros_like(gt), torch.zeros_like(gr),
                torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
                self.config.motion_xyz_step_extent, self.config.motion_rotation_step_rad,
                compression='none', gs_translation_step_extent=self.config.gs_xyz_step_extent,
                gs_rotation_step_rad=self.config.gs_rotation_step_rad, spatial_open=self.opened.cpu(),
                spatial_relative=self.relative_regions)
            rate = self._codec_rate = MotionCodecRate(empty, len(self.parents), len(self.owner),
                self.config.motion_fit_window_steps or 20)
        actual = rate.measure(rt, rr, gt, gr, self.opened.cpu(), int(step))
        scale = actual / max(rate.last_entropy_bytes + self._metadata_bytes(), 1.)
        rate.last_scale = scale
        return actual, scale

    def loss_terms(self, step, diagnostics=True):
        if self.config.spatial_gate.predictor_objective == 'proximal_rd':
            return self._predictor_rd_terms(step, diagnostics)
        steps, gs_steps = self.steps, self.gs_steps
        local_rt = _quantized(self.region_translation, steps[:3])
        local_rr = _quantized(self.region_rotation, steps[3:])
        rt, rr = self._regional_predictions() if self.relative_regions else (local_rt, local_rr)
        codec = self.config.rate_model == 'codec'
        codec_bytes, codec_scale = self._measure_codec_rate(step, force=diagnostics) if codec else (None, None)
        probabilities = self._frontier_probabilities()
        if step <= self.config.spatial_gate.gate_start_step or step > self.config.spatial_gate.freeze_step:
            probabilities = probabilities.detach()
        from .sparse_updates import sparse_block_support
        def support(value, scale, continuous=None):
            return sparse_block_support(value, scale, continuous=continuous,
                differentiable=self.config.differentiable_rate_support,
                width=self.config.sparse_rate_width_bins)

        residual_bytes = self.base_xyz.new_zeros(())
        relative = self.base_xyz.new_zeros(())
        hard_residual_bytes = self.base_xyz.new_zeros(())
        hard_relative = self.base_xyz.new_zeros(())
        candidate_relative = self.base_xyz.new_zeros(())
        discrete_joint = self.joint_training and not self._hard_gate_frozen
        gate_only_soft_cost = discrete_joint and self.config.differentiable_rate_support
        # Shared-field fitting has no point residual objective. Once the gate
        # is frozen, every point has exactly one selected node; all other
        # level terms have zero weight and need not be constructed.
        if not self.shared_discovery_active and (not self._shared_field_active or self.joint_training):
            shared_t, shared_r = self._shared_motion()
            point_t = _quantized(self.gs_translation, gs_steps[:3])
            point_r = _quantized(self.gs_rotation, gs_steps[3:])
            total_t = (shared_t + point_t).detach()
            total_r = (shared_r + point_r).detach()
            weights = None if self._hard_gate_frozen else self._path_weights(probabilities)
            nodes = ([self.owner] if self._hard_gate_frozen else
                     self.spatial_paths.unbind(1))
        else:
            nodes = ()
        if self.shared_discovery_active:
            # E is constrained to zero for every cut in this phase. The gate
            # therefore trades actual shared-field images against region cost.
            nodes = ()
        elif self.all_region_proposals:
            # A candidate changes the residual that is literally encoded,
            # while U remains the fitted total motion. Price every level.
            for level, node in enumerate(nodes):
                ct = self.gs_translation.detach() - rt[node]
                cr = self.gs_rotation.detach() - rr[node]
                pt, pr = point_t.detach() - rt[node], point_r.detach() - rr[node]
                st, sr = support(pt, gs_steps[:3], continuous=ct), support(pr, gs_steps[3:], continuous=cr)
                cost = (self._codec_rate.costs(pt / gs_steps[:3], pr / gs_steps[3:], st, sr, 1)
                        if codec else (st + sr - st * sr) * (self._gs_id_lengths + 1) + (st + sr) * 6)
                magnitude = (ct / self.extent).abs().sum(-1) + cr.abs().sum(-1)
                weight = weights[:, level]
                residual_bytes = residual_bytes + (weight * cost).sum()
                relative = relative + (weight * magnitude).mean()
                selected = self.owner.eq(node).to(cost)
                hard_residual_bytes = hard_residual_bytes + (selected * cost).sum()
                hard_relative = hard_relative + (selected * magnitude).mean()
            residual_bytes = _hard_value_soft_gradient(hard_residual_bytes, residual_bytes)
            relative = _hard_value_soft_gradient(hard_relative, relative)
            nodes = ()
        elif self.joint_image_gradients and not self.shared_image_recode:
            # Point residuals are actual jointly optimized parameters, not
            # hypothetically rebased residuals for every alternative gate.
            # Their current transmitted rate is independent of a gate switch.
            if self.private_full_motion:
                pt, pr = self._private_residuals(detach_proposal=True)
                leaf_t, leaf_r = self._regional_predictions()
                leaf = self.spatial_paths[:, -1]
                ct, cr = self.gs_translation.detach() - leaf_t[leaf], self.gs_rotation.detach() - leaf_r[leaf]
            else:
                pt = point_t.detach() if self.proximal_gs_regularization else point_t
                pr = point_r.detach() if self.proximal_gs_regularization else point_r
                ct = self.gs_translation.detach() if self.proximal_gs_regularization else self.gs_translation
                cr = self.gs_rotation.detach() if self.proximal_gs_regularization else self.gs_rotation
            st, sr = support(pt, gs_steps[:3], continuous=ct), support(pr, gs_steps[3:], continuous=cr)
            cost = (self._codec_rate.costs(pt / gs_steps[:3], pr / gs_steps[3:], st, sr, 1)
                    if codec else (st + sr - st * sr) * (self._gs_id_lengths + 1) + (st + sr) * 6)
            residual_bytes = cost.sum()
            relative = ((ct / self.extent).abs().sum(-1) + cr.abs().sum(-1)).mean()
            if self.hierarchical_private:
                magnitude = (ct / self.extent).abs().sum(-1) + cr.abs().sum(-1)
                if self.private_full_motion:
                    probability = self._private_probabilities()
                    residual_bytes, relative = (probability * cost).sum(), (probability * magnitude).mean()
                    if not self.soft_image_active:
                        permission = self._private_support().to(cost)
                        residual_bytes = _hard_value_soft_gradient((permission * cost).sum(), residual_bytes)
                        relative = _hard_value_soft_gradient((permission * magnitude).mean(), relative)
                else:
                    permission = (self._private_probabilities() if self.soft_image_active else
                        _hard_value_soft_gradient(self._private_support().to(cost), self._private_probabilities()))
                    residual_bytes = (permission * cost).sum()
                    relative = (permission * magnitude).mean()
            if self._hard_gate_frozen:
                nodes = ()
        for level, node in enumerate(nodes):
            if self.joint_image_gradients and not self.shared_image_recode:
                # Dense field STE fits closed candidates from actual image
                # gradients. A second target copied from the current coarse
                # motion would instead train away their alternative detail.
                continue
            if self.shared_image_recode and not self.joint_image_gradients and not self._hard_gate_frozen:
                # A closed candidate needs a target independent of its ancestors'
                # gate probabilities. Balance its observed members per region.
                mask = self.owner.ne(node) & self.observations.gt(0)
                counts = torch.bincount(node[mask], minlength=len(self.parents))
                weight = mask.to(rt) / counts[node].clamp_min(1)
                error = ((total_t - rt[node]) / self.extent).abs().sum(-1) + (total_r - rr[node]).abs().sum(-1)
                candidate_relative = candidate_relative + (weight * error).sum() / (
                    counts.gt(0).sum().clamp_min(1) * self.spatial_paths.shape[1])
            # Proximal mode keeps GS Adam moments photometric. Region/gate
            # derivatives still fit the same fixed full-motion targets; GS
            # sparsity is applied after Adam, not mixed into its variance.
            regularized_t = point_t.detach() if self.proximal_gs_regularization else point_t
            regularized_r = point_r.detach() if self.proximal_gs_regularization else point_r
            continuous_t = (self.gs_translation.detach() if self.proximal_gs_regularization
                            else self.gs_translation)
            continuous_r = (self.gs_rotation.detach() if self.proximal_gs_regularization
                            else self.gs_rotation)
            # A gate switch rebases the residual, so its rate target can hold
            # full motion fixed. Physical shared-image updates must not receive
            # this fictitious compensating derivative; they fit image evidence.
            predictor_t = rt[node].detach() if self.shared_image_recode else rt[node]
            predictor_r = rr[node].detach() if self.shared_image_recode else rr[node]
            candidate_t = regularized_t + shared_t.detach() - predictor_t
            candidate_r = regularized_r + shared_r.detach() - predictor_r
            st = support(candidate_t if self.joint_training else total_t - rt[node], gs_steps[:3],
                continuous=(continuous_t + shared_t.detach() - predictor_t) if self.joint_training else None)
            sr = support(candidate_r if self.joint_training else total_r - rr[node], gs_steps[3:],
                continuous=(continuous_r + shared_r.detach() - predictor_r) if self.joint_training else None)
            cost = (self._codec_rate.costs(candidate_t / gs_steps[:3], candidate_r / gs_steps[3:], st, sr, 1)
                    if codec else (st + sr - st * sr) * (self._gs_id_lengths + 1) + (st + sr) * 6)
            weight = 1. if weights is None else weights[:, level]
            residual_bytes = residual_bytes + (weight * (cost.detach() if gate_only_soft_cost or self.shared_image_recode else cost)).sum()
            # Each candidate must represent the *full* required motion, not
            # an arbitrary component of a soft mixture. Condition on the
            # current shared prediction when fitting point residuals. In a
            # hard frontier this is the actual GS residual. Representatives
            # are initialized statistically before shared-field fitting.
            if self.proximal_gs_regularization:
                # Match the continuous L1 objective handled by soft-threshold
                # updates. In particular, sub-bin proposals no longer drift
                # for free until they cross the half-bin boundary.
                relative_t = continuous_t + shared_t.detach() - predictor_t
                relative_r = continuous_r + shared_r.detach() - predictor_r
            else:
                relative_t, relative_r = candidate_t, candidate_r
            magnitude = (relative_t / self.extent).abs().sum(-1) + relative_r.abs().sum(-1)
            relative = relative + (weight * magnitude).mean()
            if discrete_joint:
                selected = self.owner.eq(node).to(magnitude)
                hard_residual_bytes = hard_residual_bytes + (selected * cost).sum()
                hard_relative = hard_relative + (selected * magnitude).mean()
        # The probability a candidate is selected is the same for every point
        # it covers. Recover it by averaging their path weights, not by charging
        # a region once for each GS.
        refine_closed = (self.config.spatial_gate.refine_unselected_candidates and
                         self.config.spatial_gate.frame_training_plan and
                         not self.shared_discovery_active)
        if (self.relative_regions and (not self.joint_image_gradients or refine_closed)
                and not self._hard_gate_frozen):
            # Closed candidates fit the required correction even when an
            # ancestor is hard closed. This loss cannot train the gate, GS,
            # or parent target; only the candidate's own correction changes.
            used = self._stored_region_support()
            target_t = self.gs_translation.detach() + shared_t.detach()
            target_r = self.gs_rotation.detach() + shared_r.detach()
            for node in self.spatial_paths.unbind(1):
                parent = self.parents[node]
                error_t = target_t - rt[parent].detach() - local_rt[node]
                error_r = target_r - rr[parent].detach() - local_rr[node]
                mask = (~used[node] & self.observations.gt(0)).to(rt)
                candidate_relative = candidate_relative + (mask * (
                    (error_t / self.extent).abs().sum(-1) + error_r.abs().sum(-1))).mean() / self.spatial_paths.shape[1]
        regional_weight = self._region_selection_probabilities(probabilities,
            include_ancestors=self.relative_regions)
        if self.joint_image_gradients and not self.soft_image_active and not self.private_full_motion:
            selected = self._stored_region_support().to(rt)
            regional_weight = _hard_value_soft_gradient(selected, regional_weight)
        st = support(local_rt, steps[:3], continuous=self.region_translation)
        sr = support(local_rr, steps[3:], continuous=self.region_rotation)
        rid = torch.cat((self._region_id_lengths.new_zeros(1), self._region_id_lengths))
        regional_cost = (self._codec_rate.costs(local_rt / steps[:3], local_rr / steps[3:], st, sr, 0)
                         if codec else (st + sr - st * sr) * (rid + 1) + (st + sr) * 6)
        region_bytes = (regional_weight * (regional_cost.detach() if self.shared_image_recode or
                        (gate_only_soft_cost and not self.joint_image_gradients) else regional_cost)).sum()
        if self.private_full_motion and not self.soft_image_active and not self._hard_gate_frozen:
            selected = self._stored_region_support().to(rt)
            region_bytes = _hard_value_soft_gradient((selected * regional_cost).sum(), region_bytes)
        soft_bytes = residual_bytes + region_bytes + self._metadata_bytes()
        if discrete_joint and (not self.joint_image_gradients or self.shared_image_recode):
            selected = self._stored_region_support().to(rt)
            hard_bytes = hard_residual_bytes + (selected * regional_cost).sum() + self._metadata_bytes()
            # Apply STE to the complete path objective, rather than each gate
            # product, so closed ancestors do not erase descendant gradients.
            soft_bytes = (hard_bytes + (soft_bytes - soft_bytes.detach()) if gate_only_soft_cost else
                          _hard_value_soft_gradient(hard_bytes, soft_bytes))
            relative = _hard_value_soft_gradient(hard_relative, relative)
        if codec:
            # Measure literal bytes at fit windows/boundaries. Between them,
            # predict the current hard streams using the calibrated model;
            # full compression cannot be differentiated and need not block
            # every optimizer update just to refresh a scalar loss value.
            if codec_bytes is None:
                selected = self._stored_region_support()
                hard_region = self._codec_rate.costs(local_rt / steps[:3], local_rr / steps[3:], st, sr, 0, allowed=selected).sum()
                if self.joint_image_gradients:
                    hard_points = residual_bytes.detach()
                else:
                    ht, hr = point_t.detach(), point_r.detach()
                    hst = torch.round(ht / gs_steps[:3]).ne(0).any(-1).to(ht)
                    hsr = torch.round(hr / gs_steps[3:]).ne(0).any(-1).to(hr)
                    hard_points = self._codec_rate.costs(ht / gs_steps[:3], hr / gs_steps[3:], hst, hsr, 1).sum()
                value = codec_scale * (hard_points + hard_region + self._metadata_bytes()).detach()
            else:
                value = soft_bytes.new_tensor(codec_bytes)
            soft_bytes = value + codec_scale * (soft_bytes - soft_bytes.detach())
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
        if self.joint_image_gradients or self.relative_regions or self.shared_image_recode:
            result['candidate_motion_loss'] = candidate_relative * self.config.relative_motion_loss_weight * ramp
        if codec:
            result['rate_codec_measured'] = self.base_xyz.new_tensor(codec_bytes is not None)
        if diagnostics:
            result.update(self._support_diagnostics(probabilities))
            result.update(rate_bytes_hard=self._hard_motion_bytes() + self._metadata_bytes(),
                rate_normalization_gs_count=self.base_xyz.new_tensor(self.rate_normalization_count),
                rate_byte_price=self.base_xyz.new_tensor(self.config.rate_loss_weight * ramp / self.rate_normalization_bytes))
            if codec:
                result.update(rate_bytes_raw=result['rate_bytes_hard'],
                              rate_bytes_hard=self.base_xyz.new_tensor(codec_bytes),
                              rate_entropy_bytes=self.base_xyz.new_tensor(self._codec_rate.last_entropy_bytes),
                              rate_codec_gradient_scale=self.base_xyz.new_tensor(codec_scale))
        return result

    def _support_diagnostics(self, probabilities):
        region_t, region_r, gs_t, gs_r = self._values(hard=True)
        gt, gr = gs_t.ne(0).any(-1), gs_r.ne(0).any(-1)
        rt_active, rr_active = region_t.ne(0).any(-1), region_r.ne(0).any(-1)
        return dict(gs_active_rows=(gt | gr).sum(), gs_translation_active_blocks=gt.sum(),
            gs_rotation_active_blocks=gr.sum(), region_active_nodes=(rt_active | rr_active).sum(),
            spatial_active_regions=self.owner.unique().numel() * self.base_xyz.new_ones(()),
            spatial_open_gates=self.opened[1:].sum(), spatial_gate_mean_probability=(
                probabilities[self.split_eligible].mean() if self.split_eligible.any() else self.base_xyz.new_zeros(())))

    @torch.no_grad()
    def _record_predictor_proposal(self):
        """Save photo-only proposals before L1/L0 can erase branch evidence."""
        shared = self._shared_motion()
        target = torch.cat((shared[0] + self.gs_translation,
                            shared[1] + self.gs_rotation), -1).detach().clone()
        metrics = []
        for parameter, group in ((self.gs_translation, self.optimizer.param_groups[2]),
                                  (self.gs_rotation, self.optimizer.param_groups[3])):
            state = self.optimizer.state.get(parameter, {})
            if 'exp_avg_sq' not in state:
                metrics.append(torch.zeros_like(parameter))
                continue
            variance = state['exp_avg_sq'] / (1 - self.config.adam_betas[1] ** int(state['step']))
            metrics.append((variance.sqrt() + self.config.adam_eps) / group['lr'])
        self._predictor_fit_state = dict(target=target, metric=torch.cat(metrics, -1))

    def _predictor_rd_terms(self, step, diagnostics):
        """Learn cuts using the cost of sparsifying each candidate's proposal.

        All candidates fit the same unshrunk photometric target, including
        branches whose ancestors are closed. Conditional fitting is separate
        from the expected path cost so ancestors cannot starve their training.
        """
        from .predictor_rd import predictor_cost
        zero = self.base_xyz.new_zeros(())
        result = dict(rate_loss=zero, relative_motion_loss=zero, candidate_motion_loss=zero)
        state = getattr(self, '_predictor_fit_state', None)
        ramp = min(max((step - self.config.rate_warmup_steps) / self.config.rate_ramp_steps, 0.), 1.)
        price = self.config.rate_loss_weight * ramp / self.rate_normalization_bytes
        if state is not None:
            rt, rr = self._regional_predictions()
            prediction = torch.cat((rt, rr), -1)
            l1 = self.gs_steps.new_tensor([1 / self.extent] * 3 + [1.] * 3)
            l1 *= self.config.relative_motion_loss_weight * ramp / len(self.owner)
            probabilities = self._frontier_probabilities()
            if self._hard_gate_frozen:
                probabilities = probabilities.detach()
            weights = None if self._hard_gate_frozen else self._path_weights(probabilities)
            paths = [self.owner] if self._hard_gate_frozen else self.spatial_paths.unbind(1)
            expected, fitted, expected_bytes = zero, zero, zero
            observed = self.observations.gt(0).to(rt)
            for level, node in enumerate(paths):
                cost, _, byte_cost = predictor_cost(state['target'], prediction[node],
                    state['metric'], self.gs_steps, l1, price, self._gs_id_lengths + 1)
                cost = cost * observed
                weight = 1. if weights is None else weights[:, level]
                expected = expected + (weight * cost.detach()).sum()
                expected_bytes = expected_bytes + (weight * byte_cost * observed).sum()
                fitted = fitted + cost.sum() / len(paths)
            active_t, active_r = rt.ne(0).any(-1).to(rt), rr.ne(0).any(-1).to(rr)
            ids = torch.cat((self._region_id_lengths.new_zeros(1), self._region_id_lengths)) + 1
            region_cost = (active_t + active_r - active_t * active_r) * ids + 6 * (active_t + active_r)
            selection = self._region_selection_probabilities(probabilities)
            region_bytes = (selection * region_cost).sum()
            # Expected costs train only the gate. The envelope fits predictors
            # independently; photo-only GS Adam receives neither derivative.
            result['relative_motion_loss'] = expected
            result['rate_loss'] = price * region_bytes
            result['candidate_motion_loss'] = fitted.detach() if self._hard_gate_frozen else fitted
            if diagnostics:
                result['predictor_expected_residual_bytes'] = expected_bytes.detach()
                result['predictor_expected_region_bytes'] = region_bytes.detach()
        if diagnostics:
            hard_bytes = self._hard_motion_bytes() + self._metadata_bytes()
            result.update(self._support_diagnostics(self._frontier_probabilities()))
            result.update(rate_bytes_surrogate=hard_bytes, rate_bytes_hard=hard_bytes,
                rate_byte_price=self.base_xyz.new_tensor(price),
                rate_normalization_gs_count=self.base_xyz.new_tensor(self.rate_normalization_count),
                rate_normalization_bytes=self.base_xyz.new_tensor(self.rate_normalization_bytes))
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
        origins = (torch.cat(self._regional_predictions(hard=True), -1).cpu().numpy()
                   if self.relative_regions else None)
        for node, rows in zip(nodes, np.split(order, starts[1:])):
            if node == 0 or not observed[rows].any():
                continue
            rid = 2 + sum(node - 1 >= threshold for threshold in
                          (128, 16384, 2097152, 268435456))
            try:
                best, _ = _rate_relative_representative(
                    symbols[rows], row_costs[rows], rid, price, scales,
                    region_origin=None if origins is None else origins[int(self.parents[node])])
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
    def _refresh_unselected_candidates(self):
        """Train closed branches before their learned gates judge their utility.

        Only dictionary entries outside the current hard frontier are seeded.
        The pose, selected predictors, GS residuals and gate values stay exact.
        """
        if self._hard_gate_frozen:
            raise ValueError('Candidate refresh belongs to the learnable-gate phase')
        full = torch.cat(self._full_motion_symbols(), -1).to(torch.int32)
        selected = self._stored_region_support()
        previous = (self.region_translation.detach().clone(), self.region_rotation.detach().clone())
        old_support = self._regional_fit_support.clone()
        # Whole-vector matches create zero-coded residual blocks. A coordinate
        # median alone can combine axes into a vector no GS actually uses.
        symbols = full.cpu().numpy()
        paths = self.spatial_paths.cpu().numpy()
        observed = self.observations.gt(0).cpu().numpy()
        estimates = np.concatenate([self._statistical_representatives(paths[:, level], symbols, observed)
            for level in range(paths.shape[1])], axis=0)
        estimated = torch.zeros_like(selected)
        if len(estimates):
            index = torch.as_tensor(estimates[:, 0], device=full.device, dtype=torch.long)
            values = torch.as_tensor(estimates[:, 1:], device=full.device, dtype=self.base_xyz.dtype)
            self.region_translation[index] = values[:, :3] * self.steps[:3]
            self.region_rotation[index] = values[:, 3:] * self.steps[3:]
            estimated[index] = True
        candidate = torch.cat(self._regional_predictions(hard=True), -1).to(torch.int32)
        # Seeding a child must never create an unencodable member residual.
        shape = (len(self.parents), 6)
        lower = torch.full(shape, torch.iinfo(torch.int32).max, device=full.device, dtype=torch.int32)
        upper = torch.full(shape, torch.iinfo(torch.int32).min, device=full.device, dtype=torch.int32)
        for nodes in self.spatial_paths.unbind(1):
            index = nodes[:, None].expand(-1, 6)
            lower.scatter_reduce_(0, index, full, reduce='amin', include_self=True)
            upper.scatter_reduce_(0, index, full, reduce='amax', include_self=True)
        valid = (~selected & estimated & self.node_counts.gt(0) &
                 (lower - candidate >= -32767).all(1) & (upper - candidate <= 32767).all(1))
        valid[0] = False
        self._regional_fit_support = old_support | valid
        for parameter, before in zip((self.region_translation, self.region_rotation), previous):
            parameter[~valid] = before[~valid]
            state = self.optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key][valid] = 0
        return int(valid.sum())

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
        origins = (torch.cat(self._regional_predictions(hard=True), -1).cpu().numpy()[
            self.parents.cpu().numpy()[nodes]] if self.relative_regions else np.zeros_like(best))
        valid = ((np.abs(best - origins) <= 32767).all(1) &
                 (lower[nodes] - best >= -32767).all(1) &
                 (upper[nodes] - best <= 32767).all(1) & (nodes != 0))
        nodes = torch.from_numpy(nodes[valid].astype(np.int64)).to(self.owner.device)
        best = torch.from_numpy((best - origins)[valid].astype(np.float32)).to(self.base_xyz)
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
        rt, rr = (value.float() for value in self._regional_predictions(hard=True))
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
        if self.config.spatial_gate.frame_training_plan:
            before = tuple(value.clone() for value in self._full_motion_symbols())
            new_open = self._gate_probabilities().ge(.5)
            new_open[0] = True
            new_owner = spatial_owner(self.spatial_paths, new_open)
            count = int(new_owner.ne(self.owner).sum())
            changes = torch.where(new_open.ne(self.opened))[0]
            if not self.shared_discovery_active and count:
                self._recode_owner(new_owner, preserve_remainder=True)
                after = self._full_motion_symbols()
                if not all(torch.equal(a, b) for a, b in zip(before, after)):
                    raise ValueError('Gate recoding changed quantized total motion')
            self.opened, self.owner = new_open, new_owner
            self._hard_gate_frozen = freeze
            self.flat_region_diagnostics.setdefault('reference_changes', []).append(
                dict(step=self.last_step, changed_gaussians=count,
                     preserved_total_motion=not self.shared_discovery_active))
            edits = [dict(kind='spatial_gate', node=int(n), opened=bool(new_open[n]), step=self.last_step)
                     for n in changes]
            self.topology_edits.extend(edits)
            return edits
        was_frozen = self._hard_gate_frozen
        if (not self._shared_field_active and full_motion is None and
                self._candidate_initialized and not self._coding_initialized and not self._hard_gate_frozen):
            full_motion = self._full_motion_symbols()
        self._coding_initialized = True
        new_open = self._gate_probabilities().ge(0.5)
        new_open[0] = True
        new_owner = spatial_owner(self.spatial_paths, new_open)
        gate_changes = torch.where(new_open != self.opened)[0]
        if self.shared_image_recode and self.joint_image_gradients:
            # Repartition the current fitted motion. Regional parameter
            # updates above remain physical; only a gate change rebases E.
            if not self.shared_discovery_active and bool(new_owner.ne(self.owner).any()):
                self._recode_owner(new_owner, preserve_remainder=True)
        elif self.joint_image_gradients:
            # The forward model and its STE must agree: a hard split changes
            # the selected shared field. Do not silently undo it in point
            # residuals; both fields continue fitting after the decision.
            pass
        elif self.joint_training and self._shared_field_active:
            # Switching representation must retain the pose just fitted by
            # the old frontier, even if gate logits crossed their threshold.
            if bool(new_owner.ne(self.owner).any()):
                rt, rr = (value.float() for value in self._regional_predictions(hard=True))
                full_motion = (rt[self.owner] + _integer(self.gs_translation, self.gs_steps[:3]),
                               rr[self.owner] + _integer(self.gs_rotation, self.gs_steps[3:]))
                self._recode_owner(new_owner, full_motion=full_motion, preserve_remainder=True)
        elif not self._shared_field_active:
            self._recode_owner(new_owner, full_motion=full_motion,
                               preserve_remainder=preserve_remainder)
        if freeze and not was_frozen and self.soft_image_gates:
            soft_t = self._shared_motion(relaxed=True)[0] + self._gs_motion(relaxed=True)[0]
            rt = self._regional_predictions()[0]
            gt = _quantized(self.gs_translation, self.gs_steps[:3])
            if self.private_full_motion:
                gt = gt - rt[self.spatial_paths[:, -1]]
            if self.hierarchical_private:
                gt = gt * new_owner.eq(self.spatial_paths[:, -1])[:, None]
            jump = (rt[new_owner] + gt - soft_t).norm(dim=-1)
            self.flat_region_diagnostics['gate_hardening'] = dict(step=self.last_step,
                mean_translation_jump=float(jump.mean()),
                p95_translation_jump=float(torch.quantile(jump, .95)),
                maximum_translation_jump=float(jump.max()),
                gs_quantization_step=float(self.gs_steps[0]))
        self._hard_gate_frozen = freeze
        edits = [dict(kind='spatial_gate', node=int(node), opened=bool(new_open[node]),
                      step=self.last_step) for node in gate_changes]
        self.opened, self.owner = new_open, new_owner
        if freeze and not was_frozen and self.config.spatial_gate.shared_discovery:
            factor = self._motion_lr_factor(self.last_step)
            self.optimizer.param_groups[2]['lr'] = self.config.gs_translation_lr * factor
            self.optimizer.param_groups[3]['lr'] = self.config.gs_rotation_lr * factor
            for parameter in (self.gs_translation, self.gs_rotation):
                state = self.optimizer.state.get(parameter, {})
                if 'exp_avg' in state:
                    # Keep sensitivity accumulated from real images, and
                    # fit a fresh direction under the newly fixed predictor.
                    state['exp_avg'].zero_()
        if freeze and not was_frozen and self.hierarchical_private:
            self._finalize_private_support()
            if self.config.spatial_gate.reparameterize_after_gate_freeze:
                self._begin_fixed_gate_residuals()
        elif freeze and not was_frozen and (self.soft_image_gates or self.stochastic_image_gates):
            # The last 30 steps fit the actual discrete field. Its physical
            # Jacobian differs from the soft mixture used during exploration.
            parameters = (self.region_translation, self.region_rotation)
            if not (self.shared_image_recode and self.joint_image_gradients):
                parameters += (self.gs_translation, self.gs_rotation)
            # Physical recoding uses the same GS image derivative before and
            # after freezing; only the auxiliary predictor objective stops.
            for parameter in parameters:
                self.optimizer.state.pop(parameter, None)
            if self.config.spatial_gate.reparameterize_after_gate_freeze:
                self._begin_fixed_gate_residuals()
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
        self._enforce_rotation_policy()

    @torch.no_grad()
    def _proximal_relative_gs(self):
        """Adam-metric soft threshold for extent/radian-normalized residual L1.

        Unlike feeding sign gradients into Adam, this proximal map only moves
        a value towards zero and cannot jump across zero due to the penalty.
        The periodic block-L0 step uses the same photometric-only moments.
        """
        if (self.shared_discovery_active or not self.gs_sparsity_active or not self.proximal_gs_regularization or
                not self.config.relative_motion_loss_weight):
            return
        ramp = min(max((self.last_step - self.config.rate_warmup_steps) /
                       self.config.rate_ramp_steps, 0.), 1.)
        coefficient = self.config.relative_motion_loss_weight * ramp / len(self.owner)
        if self.hierarchical_private:
            coefficient = coefficient * self._private_probabilities()[:, None]
        origins = self._private_origins() if self.private_full_motion else (None, None)
        for (parameter, group, scale), origin in zip((
                (self.gs_translation, self.optimizer.param_groups[2], self.extent),
                (self.gs_rotation, self.optimizer.param_groups[3], 1.)), origins):
            state = self.optimizer.state.get(parameter, {})
            if 'exp_avg_sq' not in state:
                continue
            t = int(state['step'])
            variance = state['exp_avg_sq'] / (1 - self.config.adam_betas[1] ** t)
            threshold = (group['lr'] * coefficient / scale /
                         (variance.sqrt() + self.config.adam_eps))
            residual = parameter if origin is None else parameter - origin
            shrunk = residual.sign() * (residual.abs() - threshold).clamp_min(0)
            parameter.copy_(shrunk if origin is None else origin + shrunk)
        if self._motion_fit_window_due():
            self.last_metrics['motion/prox_l1_zero_translation_components'] = int(self.gs_translation.eq(0).sum())
            self.last_metrics['motion/prox_l1_zero_rotation_components'] = int(self.gs_rotation.eq(0).sum())

    @torch.no_grad()
    def optimizer_step(self):
        cfg = self.config.spatial_gate
        self._mask_regional_fit_updates()
        if self.amplitude_refit_active:
            self._mask_amplitude_refit_updates()
        if self.last_step <= cfg.gate_start_step or self._hard_gate_frozen:
            self.gate_logits.grad = None
        factor = self._motion_lr_factor(self.last_step)
        for index, (group, lr) in enumerate(zip(self.optimizer.param_groups, (self.config.region_translation_lr,
                self.config.region_rotation_lr, self.config.gs_translation_lr, self.config.gs_rotation_lr))):
            group['lr'] = 0. if self.shared_discovery_active and index >= 2 else lr * factor
        previous_shared = None
        if self.joint_training and not self.joint_image_gradients and not self.regional_image_updates:
            previous_shared = tuple(value.detach() for value in self._shared_motion())
        self.optimizer.step()
        for parameter, scale in ((self.region_translation, self.steps[:3]),
                (self.region_rotation, self.steps[3:]), (self.gs_translation, self.gs_steps[:3]),
                (self.gs_rotation, self.gs_steps[3:])):
            parameter.clamp_(min=-32767 * scale, max=32767 * scale)
        self.gate_logits.clamp_(-6, 6)
        if self.joint_training:
            # Coding-only coordinates compensate dictionary changes. Shared
            # image mode keeps the physical region update; only a hard gate
            # switch below rebases residuals without changing the fitted pose.
            if not self.joint_image_gradients and not self.regional_image_updates:
                full_motion = (torch.round(previous_shared[0] / self.steps[:3]) +
                               _integer(self.gs_translation, self.gs_steps[:3]),
                               torch.round(previous_shared[1] / self.steps[3:]) +
                               _integer(self.gs_rotation, self.gs_steps[3:]))
                self._rebase_full_motion(full_motion, preserve_remainder=True)
            if not self.amplitude_refit_active:
                if self.config.spatial_gate.predictor_objective == 'proximal_rd':
                    self._record_predictor_proposal()
                self._proximal_relative_gs()
            return self._finish_joint_step()
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
            if self._hard_gate_frozen and not self.joint_image_gradients:
                self.last_metrics['motion/refreshed_region_modes'] = self._refresh_active_representatives()
            self.last_metrics['motion/prox_zero_region_translation_blocks'] = 0
            self.last_metrics['motion/prox_zero_region_rotation_blocks'] = 0
            self._proximal_gs_blocks()
        if self.last_step in (cfg.start_step, cfg.gate_start_step, (cfg.start_step + cfg.freeze_step) // 2,
                              cfg.freeze_step, self.motion_steps):
            self._record_gate_learning()
        self.optimizer.zero_grad(set_to_none=True)
        return edits

    @torch.no_grad()
    def _finish_joint_step(self):
        cfg = self.config.spatial_gate
        edits = []
        if cfg.frame_training_plan:
            if self.last_step <= 100 or 151 <= self.last_step <= 180:
                edits = self._select_frontier(freeze=self.last_step in (100, 180))
            if self.last_step >= 100:
                self._shared_field_active = False
            if self._motion_fit_window_due():
                self._proximal_gs_blocks()
            self.optimizer.zero_grad(set_to_none=True)
            return edits
        if (cfg.candidate_refresh_interval and not self._hard_gate_frozen and
                self.last_step < cfg.freeze_step and self.last_step % cfg.candidate_refresh_interval == 0):
            count = self._refresh_unselected_candidates()
            self.flat_region_diagnostics.setdefault('candidate_refresh_trace', []).append(
                dict(step=self.last_step, refreshed_nodes=count, representative='rate_relative_v1'))
        if not self._hard_gate_frozen:
            edits = self._select_frontier(freeze=self.last_step >= cfg.freeze_step)
            if self._hard_gate_frozen:
                # Keep point residuals and optimizer history across step 70.
                self._shared_field_active = False
                if cfg.predictor_objective == 'proximal_rd':
                    # The selected predictors now fit true image error. Their
                    # old moments belonged to a local proposal objective.
                    for parameter in (self.region_translation, self.region_rotation):
                        self.optimizer.state.pop(parameter, None)
        if self._motion_fit_window_due() and not self.amplitude_refit_active:
            if (self._hard_gate_frozen and not self.joint_image_gradients and
                    cfg.predictor_objective != 'proximal_rd'):
                self.last_metrics['motion/refreshed_region_modes'] = self._refresh_active_representatives()
            self._proximal_gs_blocks()
        if self.last_step in (1, cfg.freeze_step // 2, cfg.freeze_step, self.motion_steps):
            self._record_gate_learning()
        self.optimizer.zero_grad(set_to_none=True)
        return edits

    @torch.no_grad()
    def _proximal_gs_blocks(self):
        if self.shared_discovery_active or not self.gs_sparsity_active:
            return
        weight = self._private_probabilities().detach() if self.hierarchical_private else None
        origins = self._private_origins() if self.private_full_motion else None
        residuals = ((self.gs_translation, self.gs_rotation) if origins is None else
                     tuple(value - origin for value, origin in
                           zip((self.gs_translation, self.gs_rotation), origins)))
        if self.hierarchical_private and 'exp_avg_sq' not in self.optimizer.state.get(self.gs_translation, {}):
            # Fresh photo moments are built on the first fixed-gate step.
            return
        if self.config.rate_model != 'codec':
            if not self.hierarchical_private and not self.private_full_motion:
                return super()._proximal_gs_blocks()
            t, r = residuals
            at = _integer(t, self.gs_steps[:3]).ne(0).any(-1).to(t)
            ar = _integer(r, self.gs_steps[3:]).ne(0).any(-1).to(r)
            ids = self._gs_id_lengths + 1
            prices = [(at + ar - at * ar) * ids + 6 * (at + ar),
                      at * (ids + 6), ar * (ids + 6)]
            groups = self.optimizer.param_groups
            pt, pr = self._proximal_blocks(self.gs_translation, self.gs_rotation,
                self.gs_steps[:3], self.gs_steps[3:],
                self._gs_id_lengths, groups[2]['lr'], groups[3]['lr'], 0,
                block_bytes=([value * weight for value in prices] if weight is not None else prices),
                origins=origins)
            self.last_metrics.update({'motion/prox_zero_translation_blocks': pt,
                                     'motion/prox_zero_rotation_blocks': pr})
            return
        self._measure_codec_rate(self.last_step, force=True)
        t, r = residuals[0] / self.gs_steps[:3], residuals[1] / self.gs_steps[3:]
        st, sr = t.round().ne(0).any(-1).to(t), r.round().ne(0).any(-1).to(r)
        zero_t, zero_r = torch.zeros_like(t), torch.zeros_like(r)
        rate = self._codec_rate
        prices = [rate.costs(t, r, st, sr, 1),
                  rate.costs(t, zero_r, st, sr * 0, 1),
                  rate.costs(zero_t, r, st * 0, sr, 1)]
        prices = [value * rate.last_scale for value in prices]
        if weight is not None:
            prices = [value * weight for value in prices]
        groups = self.optimizer.param_groups
        pt, pr = self._proximal_blocks(self.gs_translation, self.gs_rotation,
            self.gs_steps[:3], self.gs_steps[3:], self._gs_id_lengths,
            groups[2]['lr'], groups[3]['lr'], 0, block_bytes=prices, origins=origins)
        self.last_metrics.update({'motion/prox_zero_translation_blocks': pt,
                                 'motion/prox_zero_rotation_blocks': pr})

    @torch.no_grad()
    def _record_gate_learning(self):
        selection = self._region_selection_probabilities()
        rt = _integer(self.region_translation, self.steps[:3]).ne(0).any(-1)
        rr = _integer(self.region_rotation, self.steps[3:]).ne(0).any(-1)
        layers, active = [], self.owner.unique()
        for level in range(self.spatial_paths.shape[1]):
            nodes = self.spatial_paths[:, level].unique()
            grad = self.region_translation.grad
            layers.append(dict(depth=level + 1, candidates=len(nodes),
                active_regions=int(torch.isin(active, nodes).sum()),
                nonzero_motion=int((rt[nodes] | rr[nodes]).sum()),
                mean_selection_probability=float(selection[nodes].mean()),
                translation_gradient_rms=(float(grad[nodes].square().mean().sqrt()) if grad is not None else 0.)))
        self.flat_region_diagnostics.setdefault('learning_trace', []).append(dict(
            step=self.last_step, hard_frozen=self._hard_gate_frozen,
            gate_temperature=float(self._gate_temperature()),
            image_relaxation=self.config.spatial_gate.image_relaxation,
            shared_field_active=self._shared_field_active,
            active_regions=len(active),
            active_regions_by_depth={str(layer['depth']): layer['active_regions'] for layer in layers},
            gate_gradient_norm=(float(self.gate_logits.grad.norm()) if self.gate_logits.grad is not None else 0.),
            coding_initialized=self._coding_initialized,
            top_gate_probabilities=self._gate_probabilities()[1:9].tolist(), layers=layers,
            private_allowed_gs=(int(self._private_support().sum()) if self.hierarchical_private else len(self.owner))))

    @torch.no_grad()
    def record_final_diagnostics(self):
        counts = torch.bincount(self.owner, minlength=len(self.parents))
        depth = {int(node): level + 1 for level in range(self.spatial_paths.shape[1])
                 for node in self.spatial_paths[:, level].unique().tolist()}
        active = torch.where(counts > 0)[0]
        # A small residual count alone does not establish correct motion. Keep
        # its distribution beside support and region size for image-error audits.
        gt, gr = (self._private_symbols() if self.private_full_motion else
                  (_integer(self.gs_translation, self.gs_steps[:3]), _integer(self.gs_rotation, self.gs_steps[3:])))
        residual = gt.ne(0).any(-1) | gr.ne(0).any(-1)
        residual_counts = torch.bincount(self.owner[residual], minlength=len(self.parents))
        observed_counts = torch.bincount(self.owner[self.observations.gt(0)],
                                        minlength=len(self.parents))
        leaf = self.spatial_paths[:, -1]
        leaf_counts = torch.bincount(leaf, minlength=len(self.parents))
        leaf_residual_counts = torch.bincount(leaf[residual], minlength=len(self.parents))
        # Materialize small diagnostic tables once. Reading CUDA scalars in
        # the region/gate loops otherwise synchronizes thousands of times.
        residual_fractions = (residual_counts / counts.clamp_min(1)).cpu()
        counts, residual_counts, observed_counts, leaf_counts, leaf_residual_counts = (
            table.cpu() for table in (counts, residual_counts, observed_counts,
                                     leaf_counts, leaf_residual_counts))
        active = active.cpu().tolist()
        translations = self.region_translation.cpu().tolist()
        rotations = self.region_rotation.cpu().tolist()
        gate_probabilities = self._gate_probabilities().cpu().tolist()
        gate_nodes = torch.where(self.split_eligible)[0].cpu().tolist()
        opened = self.opened.cpu()
        largest_leaf = int(leaf_counts.argmax())
        deepest = self.spatial_paths.shape[1]
        deepest_nodes = torch.tensor([node for node in active if depth[node] == deepest],
                                     dtype=torch.long)
        tree_health = dict(
            scope='surviving GS before this frame birth; counts do not measure image correctness',
            point_weighted_depth=sum(depth[int(node)] * int(counts[node]) for node in active) / len(self.owner),
            points_on_deepest_frontier=int(counts[deepest_nodes].sum()),
            residual_points_on_deepest_frontier=int(residual_counts[deepest_nodes].sum()),
            largest_candidate_leaf=dict(node=largest_leaf, count=int(leaf_counts[largest_leaf]),
                local_residual_count=int(leaf_residual_counts[largest_leaf])),
        )
        self.flat_region_diagnostics.update(
            semantics=('parent-relative region corrections' if self.relative_regions else 'absolute region increments') + '; orientation-only rotations; GS residuals',
            spatial_assignment='previous_decoded_position_v1',
            spatial_initialization=self.config.spatial_gate.initialization,
            spatial_initialization_distance_quantile=self.config.spatial_gate.initialization_distance_quantile,
            spatial_outlier_assignment=self.config.spatial_gate.outlier_assignment,
            training_schedule=self.config.spatial_gate.training_schedule,
            region_representative=('previous region field; joint region and GS fit' if self.joint_training
                                   else 'quantized median initialization; shared field fit; point residual fit'),
            region_motion_learning=(self._regional_fit_support is not None and
                                    bool(self._regional_fit_support.any())),
            joint_image_gradients=self.joint_image_gradients,
            image_relaxation=self.config.spatial_gate.image_relaxation,
            hierarchical_private=self.hierarchical_private,
            proposal_image_weight=self.config.spatial_gate.proposal_image_weight,
            private_full_motion=self.private_full_motion,
            private_full_motion_configured=self.config.spatial_gate.private_full_motion,
            reparameterize_after_gate_freeze=self.config.spatial_gate.reparameterize_after_gate_freeze,
            predictor_objective=self.config.spatial_gate.predictor_objective,
            shared_image_recode=self.shared_image_recode,
            rotation_updates=self.config.rotation_updates,
            rate_model=self.config.rate_model,
            region_motion=self.config.spatial_gate.region_motion,
            stored_region_nodes=int(self._stored_region_support().sum()),
            candidate_nodes=len(self.parents), active_regions=len(active),
            open_gates=int(opened[1:].sum()), eligible_gates=len(gate_nodes),
            gate_start_step=self.config.spatial_gate.start_step,
            gate_learning_start_step=self.config.spatial_gate.gate_start_step,
            relative_motion_units='extent/radians; expected full-motion candidate residual',
            gate_freeze_step=self.config.spatial_gate.freeze_step,
            active_regions_by_depth={str(level): sum(depth[int(node)] == level for node in active)
                                     for level in range(1, self.spatial_paths.shape[1] + 1)},
            tree_health=tree_health,
            regions=[dict(node=int(node), depth=depth[int(node)], count=int(counts[node]),
                observed_count=int(observed_counts[node]), local_residual_count=int(residual_counts[node]),
                local_residual_fraction=float(residual_fractions[node]),
                translation=translations[node], rotation=rotations[node])
                for node in active],
            gates=[dict(node=node, probability=gate_probabilities[node],
                        opened=bool(opened[node])) for node in gate_nodes])

    @torch.no_grad()
    def decoded_survivor_pose(self):
        """CPU decoder-exact old pose without encoding or advancing state."""
        from .rigid_tree_codec import compose_decoded_survivors
        rt, rr, gt, gr = self._values(hard=True)
        return compose_decoded_survivors(self.frame_reference_state,self.extent,
            self.previous_rows,self.owner,self.parents,rt,rr,gt,gr,
            self.config.motion_xyz_step_extent,self.config.motion_rotation_step_rad,
            gs_translation_step_extent=self.config.gs_xyz_step_extent,
            gs_rotation_step_rad=self.config.gs_rotation_step_rad,
            spatial_open=self.opened,spatial_relative=self.relative_regions)

    def export_motion(self, path, inherit_birth_rotation=False, birth_pose_steps=None):
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
            gs_rotation_step_rad=self.config.gs_rotation_step_rad, spatial_open=self.opened.cpu(),
            spatial_relative=self.relative_regions,
            inherit_birth_rotation=inherit_birth_rotation,birth_pose_steps=birth_pose_steps)
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
            semantics='spatial gates; ' + ('parent-relative region corrections' if self.relative_regions else 'absolute region increments') + '; no rotation of GS centers')

    def capture(self):
        packet = super().capture()
        if packet is not None:
            packet['representative_objective'] = 'rate_relative_v1'
            packet['gate_sparsification'] = 'fit_windows'
            packet['regional_image_fitting'] = ('stochastic_proposals_fixed_residuals_v1'
                                                if self.config.spatial_gate.reparameterize_after_gate_freeze else
                                                'joint_image_residual_stochastic_v1'
                                                if self.stochastic_image_gates and not self.config.spatial_gate.private_full_motion else
                                                'joint_image_private_proposal_stochastic_v1'
                                                if self.stochastic_image_gates else
                                                'joint_image_private_proposal_soft_v1'
                                                if self.private_full_motion and self.soft_image_gates else
                                                'joint_image_private_proposal_v1' if self.private_full_motion else
                                                'joint_image_soft_anneal_private_v1'
                                                if self.soft_image_gates and self.hierarchical_private else
                                                'joint_image_soft_anneal_v1' if self.soft_image_gates else
                                                'joint_image_hierarchical_private_v1' if self.hierarchical_private else
                                                'predictor_proposal_then_image_v1'
                                                if self.config.spatial_gate.predictor_objective == 'proximal_rd' else
                                                'shared_image_recode_v1' if self.shared_image_recode else
                                                'joint_image_dense_candidates_v2' if self.joint_image_gradients else
                                                'joint_motion_coordinates_v1' if self.joint_training
                                                else 'observed_candidates_v2')
            packet['candidate_initialization'] = ('previous_region_field_v1' if self.joint_training
                                                   else 'quantized_median_v1')
            packet['motion_schedule'] = ('joint_motion_v2' if self.joint_training
                                          else 'shared_field_zero_residual_v2')
            if self.joint_training:
                packet['gate_estimator'] = self._gate_estimator()
            if self.stochastic_image_gates:
                packet['image_gate_sampler'] = dict(rng_state=self._gate_rng.get_state().clone(),
                    step=self._sampled_gate_step,
                    noise=(None if self._sampled_gate_noise is None else self._sampled_gate_noise.detach().cpu().clone()))
            if self.config.spatial_gate.reparameterize_after_gate_freeze:
                packet['private_full_motion_active'] = self.private_full_motion
            packet['motion_initialization'] = ('previous_full_increment_v2' if self.joint_training
                                                else 'previous_increment_v1')
            packet['spatial_assignment'] = 'previous_decoded_position_v1'
            packet['shared_field_active'] = self._shared_field_active
            packet['regional_fit_support'] = self._regional_fit_support
            packet['parameters']['gate_logits'] = self.gate_logits.detach().clone()
            packet['tensors'].update({name: getattr(self, name) for name in
                ('spatial_paths', 'opened', 'node_counts', 'split_eligible')})
            packet['candidate_initialized'] = self._candidate_initialized
            packet['coding_initialized'] = self._coding_initialized
            packet['hard_gate_frozen'] = self._hard_gate_frozen
            packet['amplitude_refit_support'] = getattr(self, '_amplitude_refit_support', None)
            packet['codec_rate'] = (None if getattr(self, '_codec_rate', None) is None
                                    else self._codec_rate.capture())
            if self.config.spatial_gate.predictor_objective == 'proximal_rd':
                packet['predictor_fit_state'] = self._predictor_fit_state
        return packet

    def restore(self, packet):
        self._frame_hierarchical_private = self.config.spatial_gate.hierarchical_private
        self._frame_private_full_motion = self.config.spatial_gate.private_full_motion
        if packet is not None and packet.get('spatial_assignment') != 'previous_decoded_position_v1':
            raise ValueError('Fixed-membership spatial checkpoints are replay-only')
        initialization = 'previous_full_increment_v2' if self.joint_training else 'previous_increment_v1'
        schedule = 'joint_motion_v2' if self.joint_training else 'shared_field_zero_residual_v2'
        candidate = 'previous_region_field_v1' if self.joint_training else 'quantized_median_v1'
        if packet is not None and packet.get('motion_initialization') != initialization:
            raise ValueError('Earlier zero-initialized motion checkpoints are replay-only')
        if packet is not None and packet.get('motion_schedule') != schedule:
            raise ValueError('Earlier point-first motion schedules are replay-only')
        estimator = self._gate_estimator()
        if packet is not None and self.joint_training and packet.get('gate_estimator') != estimator:
            raise ValueError('Earlier joint gate estimators are replay-only')
        fitting = ('stochastic_proposals_fixed_residuals_v1'
                   if self.config.spatial_gate.reparameterize_after_gate_freeze else
                   'joint_image_residual_stochastic_v1'
                   if self.stochastic_image_gates and not self.config.spatial_gate.private_full_motion else
                   'joint_image_private_proposal_stochastic_v1' if self.stochastic_image_gates else
                   'joint_image_private_proposal_soft_v1' if self.private_full_motion and self.soft_image_gates else
                   'joint_image_private_proposal_v1' if self.private_full_motion else
                   'joint_image_soft_anneal_private_v1' if self.soft_image_gates and self.hierarchical_private else
                   'joint_image_soft_anneal_v1' if self.soft_image_gates else
                   'joint_image_hierarchical_private_v1' if self.hierarchical_private else
                   'predictor_proposal_then_image_v1'
                   if self.config.spatial_gate.predictor_objective == 'proximal_rd' else
                   'shared_image_recode_v1' if self.shared_image_recode else
                   'joint_image_dense_candidates_v2' if self.joint_image_gradients else
                   'joint_motion_coordinates_v1' if self.joint_training else 'observed_candidates_v2')
        if packet is not None and packet.get('regional_image_fitting') != fitting:
            raise ValueError('Earlier regional image-fitting schedules are replay-only')
        if packet is not None and packet.get('gate_sparsification') != 'fit_windows':
            raise ValueError('Earlier gate sparsification schedules are replay-only')
        if packet is not None and (packet.get('fixed_refit_support') is not None or
                packet['config'].get('post_birth_motion_steps', 0)):
            raise ValueError('Post-birth refinement checkpoints are replay-only')
        if packet is not None and packet.get('representative_objective') != 'rate_relative_v1':
            raise ValueError('Earlier region representative objectives are replay-only')
        if packet is not None and packet.get('candidate_initialization') != candidate:
            raise ValueError('Earlier candidate initializations are replay-only')
        super().restore(packet)
        if packet is not None:
            self._survivor_mapping_bytes = self._count_survivor_mapping_bytes()
        self._candidate_initialized = bool(packet and packet.get('candidate_initialized', False))
        self._shared_field_active = bool(packet and packet.get('shared_field_active', False))
        self._regional_fit_support = None if packet is None else packet['regional_fit_support']
        self._coding_initialized = bool(packet and packet.get('coding_initialized', packet.get('hard_gate_frozen', False)))
        self._hard_gate_frozen = bool(packet and packet.get('hard_gate_frozen', False))
        if packet is not None and self.config.spatial_gate.reparameterize_after_gate_freeze:
            active = packet.get('private_full_motion_active')
            if not isinstance(active, bool) or active == self._hard_gate_frozen:
                raise ValueError('Checkpoint private coordinates disagree with its gate stage')
            self._frame_private_full_motion = active
        self._amplitude_refit_support = None if packet is None else packet.get('amplitude_refit_support')
        self._predictor_fit_state = None if packet is None else packet.get('predictor_fit_state')
        if self._predictor_fit_state is not None:
            if (set(self._predictor_fit_state) != {'target', 'metric'} or any(
                    value.shape != (len(self.owner), 6) or not bool(torch.isfinite(value).all())
                    for value in self._predictor_fit_state.values())):
                raise ValueError('Invalid shared predictor fitting state')
            self._predictor_fit_state = {name: value.to(self.base_xyz).clone()
                                        for name, value in self._predictor_fit_state.items()}
        if self.amplitude_refit_active:
            if not self.config.amplitude_refit_steps or not self._hard_gate_frozen:
                raise ValueError('Invalid motion amplitude-refit checkpoint')
            expected = {'region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'}
            if set(self._amplitude_refit_support) != expected:
                raise ValueError('Incomplete motion amplitude-refit support')
            for name, support in self._amplitude_refit_support.items():
                parameter = getattr(self, name)
                if support.dtype != torch.bool or support.shape != parameter.shape[:1]:
                    raise ValueError('Invalid motion amplitude-refit support shape')
                self._amplitude_refit_support[name] = support.to(parameter.device).clone()
        self._codec_rate = None
        if packet is not None and packet.get('codec_rate') is not None:
            from .motion_rate import MotionCodecRate
            self._codec_rate = MotionCodecRate.restore(packet['codec_rate'])
        if self._shared_field_active and not self.joint_training:
            for parameter in (self.gs_translation, self.gs_rotation):
                parameter.requires_grad_(False)
        if packet is not None:
            self._enforce_rotation_policy()
            if self.stochastic_image_gates:
                self._initialize_gate_sampler()
                sampler = packet.get('image_gate_sampler')
                if not isinstance(sampler, dict) or set(sampler) != {'rng_state', 'step', 'noise'}:
                    raise ValueError('Stochastic gate checkpoint lacks its sampler state')
                step, noise, rng = sampler['step'], sampler['noise'], sampler['rng_state']
                if (isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= self.last_step or
                        not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1 or
                        (noise is None and step != 0) or
                        (noise is not None and (not torch.is_tensor(noise) or
                         noise.shape != self.gate_logits.shape or not noise.is_floating_point() or
                         not bool(torch.isfinite(noise).all())))):
                    raise ValueError('Invalid stochastic gate sampler state')
                try:
                    self._gate_rng.set_state(rng.cpu())
                except RuntimeError as error:
                    raise ValueError('Invalid stochastic gate RNG state') from error
                self._sampled_gate_step = step
                self._sampled_gate_noise = None if noise is None else noise.to(self.gate_logits).clone()
