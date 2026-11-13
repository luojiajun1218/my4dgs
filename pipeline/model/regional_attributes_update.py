"""区域共享的运动和外观属性更新。

负责区域位置、旋转及外观属性增量的参数、量化和更新，并配合当前区域归属；
区域划分与 gate 选择在 spatial_region_gate.py，逐点残差在 residual_gs_attributes_update.py。
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .storage_rate import raw_sparse_row_bytes
from .frame_stream_format import (FIELDS, _regional_attribute_zeros as _zeros, REGIONAL_ATTRIBUTE_MAGIC as MAGIC, REGIONAL_ATTRIBUTE_HEADER as HEADER)
from .frame_stream_encode import (encode_regional_full)
from .frame_stream_decode import (decode_full_attribute_delta, decode_regional_full)
from .frame_stream_format import (_step)
from .rigid_residual_tree import _integer, _quantized
from .spatial_region_gate import spatial_accumulate
class _HardForwardGateProxy(torch.autograd.Function):
    """Preserve the hard tensor bit-for-bit and add only the proxy's gradient."""
    @staticmethod
    def forward(ctx, hard, proxy):
        return hard

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


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


class RegionalMotionAttributes(nn.Module):
    """Own regional translation and rotation, independently of point residuals."""
    def __init__(self, reference, count):
        super().__init__()
        self.translation = nn.Parameter(reference.new_zeros(count, 3))
        self.rotation = nn.Parameter(reference.new_zeros(count, 3))

    @classmethod
    def from_tensors(cls, translation, rotation):
        if translation.shape != rotation.shape or translation.ndim != 2 or translation.shape[1] != 3:
            raise ValueError('Regional motion checkpoint has invalid shapes')
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.translation = nn.Parameter(translation.detach().clone())
        result.rotation = nn.Parameter(rotation.detach().clone())
        return result

    def optimizer_groups(self, config):
        return [dict(params=[self.translation], lr=config.region_translation_lr),
                dict(params=[self.rotation], lr=config.region_rotation_lr)]

    @staticmethod
    def quantization_steps(reference, config, extent):
        return reference.new_tensor(
            [config.motion_xyz_step_extent * extent] * 3 +
            [config.motion_rotation_step_rad] * 3)

    def values(self, motion, hard=False):
        quantize = _integer if hard else _quantized
        rt = quantize(self.translation, motion.steps[:3])
        rr = quantize(self.rotation, motion.steps[3:])
        active = motion._stored_region_support()
        return rt * active[:, None], rr * active[:, None]

    def predictions(self, motion, hard=False):
        if not motion.relative_regions:
            quantize = _integer if hard else _quantized
            return (quantize(motion.region_translation, motion.steps[:3]),
                    quantize(motion.region_rotation, motion.steps[3:]))
        result = []
        for parameter, step in ((motion.region_translation, motion.steps[:3]),
                                (motion.region_rotation, motion.steps[3:])):
            symbols = (_integer(parameter, step).to(torch.int32) if hard else
                       _quantized(parameter / step, 1.))
            # The codec's sentinel root is an identity, never an unpriced
            # global parameter receiving every descendant's image gradient.
            symbols = torch.cat((torch.zeros_like(symbols[:1]), symbols[1:]))
            total = spatial_accumulate(symbols, motion.parents)
            result.append(total if hard else total * step)
        return tuple(result)

    def stored_support(self, motion):
        """Relative packets retain the selected frontier and its ancestors."""
        active = torch.bincount(motion.owner, minlength=len(motion.parents)).gt(0)
        if motion.relative_regions:
            # The histogram already identifies every occupied node. Sorting
            # all GS owners again is unnecessary, especially at a fixed cut.
            nodes = active.nonzero(as_tuple=True)[0]
            for _ in range(motion.spatial_paths.shape[1]):
                active[nodes] = True
                nodes = motion.parents[nodes].clamp_min(0)
        active[0] = False
        return active

    @torch.no_grad()
    def enable_relative(self, motion, *, inherit_unused=False):
        """Preserve applied motion; optionally initialize unused children by inheritance."""
        if motion.relative_regions:
            return
        image_refinement = (motion.stochastic_image_gates and
                            ((motion.config.spatial_gate.private_full_motion and
                              motion.config.spatial_gate.reparameterize_after_gate_freeze) or
                             not motion.config.spatial_gate.hierarchical_private))
        if (not motion.committed or not motion.joint_training or motion.shared_image_recode or
                (motion.joint_image_gradients and not image_refinement)):
            raise ValueError('Migrate a finalized joint full-motion checkpoint only')
        full = tuple(value.clone() for value in motion._full_motion_symbols())
        used = torch.bincount(motion.owner, minlength=len(motion.parents)).gt(0)
        for _ in range(motion.spatial_paths.shape[1]):
            used[motion.parents[used].clamp_min(0)] = True
        converted = []
        for parameter, step in ((motion.region_translation, motion.steps[:3]),
                                (motion.region_rotation, motion.steps[3:])):
            absolute = _integer(parameter, step).to(torch.int32)
            local = absolute - absolute[motion.parents]
            local[0] = 0  # The identity root has sentinel parent -1.
            if inherit_unused:
                # Unused children start at their parent's fitted field. A
                # stale absolute zero must not become its negative correction.
                local[~used] = 0
            if bool(local.abs().gt(32767).any()):
                raise ValueError('Relative conversion exceeds the int16 correction range')
            converted.append(_retain_bin_remainder(local, parameter - parameter[motion.parents], step))
            converted[-1][0] = 0
        motion.region_translation.copy_(converted[0])
        motion.region_rotation.copy_(converted[1])
        motion.config.spatial_gate.region_motion = 'relative'
        motion._codec_rate = None
        assert all(torch.equal(a, b) for a, b in zip(full, motion._full_motion_symbols()))

    @torch.no_grad()
    def mask_fit_updates(self, motion):
        """Fit observed candidates on the currently selected spatial frontier.

        Observations identify support, including zero motion estimates.
        Image fitting can change a supported region's complete motion. Unselected dictionaries stay
        fixed after hard coding begins, including their Adam momentum.
        """
        support = motion._regional_fit_support
        if not motion._candidate_initialized or support is None:
            motion.region_translation.grad = None
            motion.region_rotation.grad = None
            return
        support = support.clone()
        if ((motion._hard_gate_frozen and not motion.refining_unselected_candidates) or
                (motion._coding_initialized and not motion.joint_training)):
            support &= motion._stored_region_support()
        for parameter in (motion.region_translation, motion.region_rotation):
            if parameter.grad is not None:
                parameter.grad.mul_(support[:, None])
            state = motion.optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key].mul_(support[:, None])
            if (motion.shared_image_recode and motion.joint_image_gradients and
                    (not motion._hard_gate_frozen or motion.refining_unselected_candidates) and parameter.grad is not None and
                    'exp_avg' in state):
                # These predictors fit only actual main/auxiliary images.
                # A sampled-out row must not move on its old velocity; retain
                # its variance estimate for its next observed image gradient.
                participating = parameter.grad.ne(0).any(-1)
                state['exp_avg'].mul_(participating[:, None])

    def shared_motion(self, motion, relaxed=False):
        rt, rr = motion._regional_predictions()
        if motion.all_region_proposals:
            # U is the image variable. E=Q(U)-P is its actual coding residual;
            # the predictor has a separate real image objective below.
            return rt[motion.owner].detach(), rr[motion.owner].detach()
        if motion.regional_image_updates:
            return rt[motion.owner], rr[motion.owner]
        if motion.joint_image_gradients:
            if motion._hard_gate_frozen:
                if motion.private_full_motion:
                    coarse = ~motion._private_support()[:, None]
                    return tuple(_hard_value_soft_gradient(value[motion.owner], value[motion.owner] * coarse)
                                 for value in (rt, rr))
                return rt[motion.owner], rr[motion.owner]
            soft = motion._path_weights(motion._gate_probabilities())
            if motion.private_full_motion:
                # The last alternative is the independent full GS proposal.
                # Leaf region vectors fit coding residuals, not image error.
                soft = torch.cat((soft[:, :-1], torch.zeros_like(soft[:, -1:])), -1)
            if relaxed:
                # Each candidate now affects the image used to train it.
                # Materialization/encoding always use the discrete field.
                return tuple((soft[..., None] * value[motion.spatial_paths]).sum(1)
                             for value in (rt, rr))
            hard = motion._path_weights(motion.opened.to(soft))
            # Relax the complete field. Applying STE only to its weights
            # leaves every hard-closed candidate's image Jacobian at zero.
            # The rendered cut remains binary; all candidate fields now fit
            # the same image evidence through their soft path probabilities.
            return tuple(_hard_value_soft_gradient(
                (hard[..., None] * value[motion.spatial_paths]).sum(1),
                (soft[..., None] * value[motion.spatial_paths]).sum(1)) for value in (rt, rr))
        if motion.joint_training:
            # Joint coding uses full-motion optimization coordinates. Region
            # changes are compensated in the GS residual after Adam, just as
            # gate switches are. Images therefore train the full GS increment;
            # every candidate and its gate learn from the residual/rate loss.
            # A fictitious image derivative through the uncompensated shared
            # field would reward the current representation's initialization.
            return rt[motion.owner].detach(), rr[motion.owner].detach()
        if motion._shared_field_active:
            probabilities = motion._gate_probabilities()
            soft = motion._path_weights(probabilities)
            hard = motion._path_weights(probabilities.detach().ge(.5).to(probabilities))
            # Render one actual frontier; differentiate its alternatives.
            # No point residual may compensate for the subdivision decision.
            weights = hard.detach() + (soft - soft.detach())
            return ((weights[..., None] * rt[motion.spatial_paths]).sum(1),
                    (weights[..., None] * rr[motion.spatial_paths]).sum(1))
        return rt[motion.owner], rr[motion.owner]

    @torch.no_grad()
    def statistical_representatives(self, motion, groups, symbols, observed):
        if not observed.any():
            return np.empty((0, 7), dtype=np.int32)
        row_costs = motion._gs_id_lengths.detach().cpu().numpy() + 1
        # Both gate losses have the same ramp, which cancels in this argmin.
        price = motion.effective_rate_loss_weight / motion.rate_normalization_bytes
        scales = (motion.gs_steps.detach().cpu().numpy().astype(np.float64) /
                  np.array([motion.extent] * 3 + [1.] * 3) *
                  motion.config.relative_motion_loss_weight / len(motion.owner))
        order = np.argsort(groups, kind='stable')
        nodes, starts = np.unique(groups[order], return_index=True)
        result = []
        origins = (torch.cat(motion._regional_predictions(hard=True), -1).cpu().numpy()
                   if motion.relative_regions else None)
        for node, rows in zip(nodes, np.split(order, starts[1:])):
            if node == 0 or not observed[rows].any():
                continue
            rid = 2 + sum(node - 1 >= threshold for threshold in
                          (128, 16384, 2097152, 268435456))
            try:
                best, _ = _rate_relative_representative(
                    symbols[rows], row_costs[rows], rid, price, scales,
                    region_origin=None if origins is None else origins[int(motion.parents[node])])
            except ValueError:
                continue
            result.append(np.r_[node, best])
        return np.asarray(result, dtype=np.int32).reshape(-1, 7)

    @torch.no_grad()
    def initialize_candidates(self, motion, full_motion=None):
        target = (torch.cat(full_motion, -1).to(torch.int32) if full_motion is not None else
                  torch.cat((_integer(motion.gs_translation, motion.gs_steps[:3]),
                             _integer(motion.gs_rotation, motion.gs_steps[3:])), -1))
        observed = motion.observations.gt(0)
        symbols = target.cpu().numpy()
        paths = motion.spatial_paths.cpu().numpy()
        observed = observed.cpu().numpy()
        # Initialization is a motion estimate, not a coding decision. A zero
        # estimate must remain learnable; gates decide whether finer regions
        # are useful after their motion is allowed to fit image evidence.
        motion._regional_fit_support = torch.zeros(len(motion.parents), dtype=torch.bool,
                                                 device=motion.base_xyz.device)
        for level in range(motion.spatial_paths.shape[1]):
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
            index = torch.as_tensor(estimates[:, 0], device=motion.base_xyz.device,
                                    dtype=torch.long)
            values = torch.as_tensor(estimates[:, 1:], device=motion.base_xyz.device,
                                     dtype=motion.base_xyz.dtype)
            motion.region_translation[index] = values[:, :3] * motion.steps[:3]
            motion.region_rotation[index] = values[:, 3:] * motion.steps[3:]
            motion._regional_fit_support[index] = True
        motion._candidate_initialized = True
        motion._regional_fit_support[0] = False

    @torch.no_grad()
    def refresh_unselected_candidates(self, motion):
        """Train closed branches before their learned gates judge their utility.

        Only dictionary entries outside the current hard frontier are seeded.
        The pose, selected predictors, GS residuals and gate values stay exact.
        """
        if motion._hard_gate_frozen:
            raise ValueError('Candidate refresh belongs to the learnable-gate phase')
        full = torch.cat(motion._full_motion_symbols(), -1).to(torch.int32)
        selected = motion._stored_region_support()
        previous = (motion.region_translation.detach().clone(), motion.region_rotation.detach().clone())
        old_support = motion._regional_fit_support.clone()
        # Whole-vector matches create zero-coded residual blocks. A coordinate
        # median alone can combine axes into a vector no GS actually uses.
        symbols = full.cpu().numpy()
        paths = motion.spatial_paths.cpu().numpy()
        observed = motion.observations.gt(0).cpu().numpy()
        estimates = np.concatenate([motion._statistical_representatives(paths[:, level], symbols, observed)
            for level in range(paths.shape[1])], axis=0)
        estimated = torch.zeros_like(selected)
        if len(estimates):
            index = torch.as_tensor(estimates[:, 0], device=full.device, dtype=torch.long)
            values = torch.as_tensor(estimates[:, 1:], device=full.device, dtype=motion.base_xyz.dtype)
            motion.region_translation[index] = values[:, :3] * motion.steps[:3]
            motion.region_rotation[index] = values[:, 3:] * motion.steps[3:]
            estimated[index] = True
        candidate = torch.cat(motion._regional_predictions(hard=True), -1).to(torch.int32)
        # Seeding a child must never create an unencodable member residual.
        shape = (len(motion.parents), 6)
        lower = torch.full(shape, torch.iinfo(torch.int32).max, device=full.device, dtype=torch.int32)
        upper = torch.full(shape, torch.iinfo(torch.int32).min, device=full.device, dtype=torch.int32)
        for nodes in motion.spatial_paths.unbind(1):
            index = nodes[:, None].expand(-1, 6)
            lower.scatter_reduce_(0, index, full, reduce='amin', include_self=True)
            upper.scatter_reduce_(0, index, full, reduce='amax', include_self=True)
        valid = (~selected & estimated & motion.node_counts.gt(0) &
                 (lower - candidate >= -32767).all(1) & (upper - candidate <= 32767).all(1))
        valid[0] = False
        motion._regional_fit_support = old_support | valid
        for parameter, before in zip((motion.region_translation, motion.region_rotation), previous):
            parameter[~valid] = before[~valid]
            state = motion.optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key][valid] = 0
        return int(valid.sum())

    @torch.no_grad()
    def refresh_active_representatives(self, motion):
        """Absorb common fitted residuals into the current cut, preserving pose.

        The cut and all Adam moments stay fixed. Refresh only after it is hard:
        soft estimation predictors do not yet represent one selected region.
        """
        if not motion._hard_gate_frozen:
            raise ValueError('Region mode refresh requires a fixed hard frontier')
        full = motion._full_motion_symbols()
        symbols = torch.cat(full, -1).to(torch.int32).cpu().numpy()
        owners = motion.owner.cpu().numpy()
        observed = motion.observations.gt(0).cpu().numpy()
        modes = motion._statistical_representatives(owners, symbols, observed)
        if not len(modes):
            return 0
        # A modal value must fit the regional stream and every member residual,
        # including unobserved members. Otherwise keep that region's old origin.
        lower = np.full((len(motion.parents), 6), np.iinfo(np.int32).max, dtype=np.int32)
        upper = np.full_like(lower, np.iinfo(np.int32).min)
        np.minimum.at(lower, owners, symbols)
        np.maximum.at(upper, owners, symbols)
        nodes, best = modes[:, 0], modes[:, 1:]
        origins = (torch.cat(motion._regional_predictions(hard=True), -1).cpu().numpy()[
            motion.parents.cpu().numpy()[nodes]] if motion.relative_regions else np.zeros_like(best))
        valid = ((np.abs(best - origins) <= 32767).all(1) &
                 (lower[nodes] - best >= -32767).all(1) &
                 (upper[nodes] - best <= 32767).all(1) & (nodes != 0))
        nodes = torch.from_numpy(nodes[valid].astype(np.int64)).to(motion.owner.device)
        best = torch.from_numpy((best - origins)[valid].astype(np.float32)).to(motion.base_xyz)
        if not len(nodes):
            return 0
        motion.region_translation[nodes] = _retain_bin_remainder(
            best[:, :3], motion.region_translation[nodes], motion.steps[:3])
        motion.region_rotation[nodes] = _retain_bin_remainder(
            best[:, 3:], motion.region_rotation[nodes], motion.steps[3:])
        motion._rebase_full_motion(full, preserve_remainder=True)
        return len(nodes)


class RegionalAttributesUpdate:
    """Regional log-scale, SH and opacity deltas alongside regional motion."""
    def __init__(self, motion, config, gaussian, opacity):
        self.motion, self.config = motion, config
        self.opacity_l1_weight = (opacity.l1_weight if config.regional_opacity_l1_weight is None
                                  else config.regional_opacity_l1_weight)
        self.steps = dict(logscale=config.logscale_step, sh_dc=config.sh_dc_step,
                          sh_rest=config.sh_rest_step, opacity=_step(opacity.step))
        nodes = len(motion.parents)
        template = _zeros(nodes, 3, motion.base_xyz.device)
        template['opacity'] = motion.base_xyz.new_zeros(nodes, 1)
        self.parameters = {k: nn.Parameter(v) for k, v in template.items()}
        self.rates = dict(logscale=gaussian.scaling_lr, sh_dc=gaussian.feature_lr,
                          sh_rest=gaussian.feature_lr/20, opacity=opacity.learning_rate)
        for name, value in self.parameters.items():
            motion.optimizer.add_param_group(dict(params=[value], name='early_regional_full_'+name,
                lr=self.rates[name], betas=(.9, .999), eps=1e-8))

    def prepare(self, step, ratio):
        factor = ratio ** min(max((step-101)/199, 0.), 1.)
        for name, value in self.parameters.items():
            value.requires_grad_(step >= 101)
            if step < 101:
                value.grad = None
            self.group(name)['lr'] = self.rates[name]*factor

    def group(self, name):
        return next(x for x in self.motion.optimizer.param_groups if x.get('name') == 'early_regional_full_'+name)

    def reachable_gate_mask(self):
        paths = self.motion.spatial_paths
        reachable = torch.zeros_like(self.motion.opened, dtype=torch.bool)
        points = torch.ones(len(paths), dtype=torch.bool, device=paths.device)
        for nodes in paths.unbind(1)[:-1]:
            reachable[nodes[points]] = True
            points = points & self.motion.opened[nodes]
        return reachable & self.motion.split_eligible

    def capture(self):
        return {name: dict(value=value.detach().cpu().clone(),
                hyper={k:v for k,v in self.group(name).items() if k != 'params'},
                state={k:(v.detach().cpu().clone() if torch.is_tensor(v) else v)
                       for k,v in self.motion.optimizer.state.get(value, {}).items()})
                for name,value in self.parameters.items()}

    def restore(self, saved):
        if not isinstance(saved, dict) or set(saved) != set(self.parameters):
            raise ValueError('Regional full attribute checkpoint is missing')
        for name,value in self.parameters.items():
            item = saved[name]
            if item['value'].shape != value.shape or not bool(torch.isfinite(item['value']).all()):
                raise ValueError('Regional full attribute checkpoint has invalid values')
            with torch.no_grad():
                value.copy_(item['value'].to(value))
            self.group(name).update(item['hyper'])
            self.motion.optimizer.state[value] = {k:(v.to('cpu' if k == 'step' else value.device)
                if torch.is_tensor(v) else v) for k,v in item['state'].items()}

    def gate_image_weights(self):
        """Build one image-gate graph per render for all regional attributes."""
        motion = self.motion
        if (self.config.attribute_aware_gate and getattr(motion, 'last_step', 0) > 100 and
                not motion._hard_gate_frozen and torch.is_grad_enabled()):
            return motion._path_weights(motion._gate_probabilities())
        return None

    def apply(self, name, reference, gate_weights=None):
        from .residual_gs_attributes_update import _QuantizedFullAttribute
        parameter = self.parameters[name]
        hard = _QuantizedFullAttribute.apply(reference, parameter[self.motion.owner], self.steps[name])
        motion = self.motion
        if (self.config.attribute_aware_gate and getattr(motion, 'last_step', 0) > 100 and
                not motion._hard_gate_frozen and torch.is_grad_enabled()):
            weights = gate_weights if gate_weights is not None else self.gate_image_weights()
            candidate = torch.round(parameter.detach() / self.steps[name]) * self.steps[name]
            dimensions = (1,) * (parameter.ndim - 1)
            soft = (weights.reshape(*weights.shape, *dimensions) * candidate[motion.spatial_paths]).sum(1)
            # The actual image keeps the hard cut. Only the gate gets the
            # alternative-path image derivative; attribute Jacobians stay hard.
            return _HardForwardGateProxy.apply(hard, soft)
        return hard

    def alpha(self, alpha, committed=False, point_logits=None, gate_weights=None):
        offset = self.parameters['opacity'][self.motion.owner]
        logits = torch.logit(alpha.clamp(1e-6, 1-1e-6))
        surrogate = self.apply('opacity', logits, gate_weights=gate_weights).sigmoid()
        if committed:
            base = point_logits.detach().cpu()
            symbols = torch.round(self.parameters['opacity'].detach()/self.steps['opacity']).cpu()
            delta = symbols[self.motion.owner.detach().cpu()] * self.steps['opacity']
            exact = torch.where(delta.ne(0), base + delta, base).to(alpha.device).sigmoid()
        else:
            integer = torch.round(offset.detach()/self.steps['opacity'])
            exact = torch.where(integer.ne(0), surrogate, alpha)
        return exact.detach() + (surrogate - surrogate.detach())

    @torch.no_grad()
    def rate_terms(self, price):
        from .storage_rate import regional_full_attribute_rate_terms
        return regional_full_attribute_rate_terms(self, price)

    def gate_rate_terms(self, full, opacity, price, *, point_symbols=None):
        from .storage_rate import regional_full_attribute_gate_rate_terms
        return regional_full_attribute_gate_rate_terms(self, full, opacity, price, point_symbols=point_symbols)

    @torch.no_grad()
    def rebase_points(self, previous_owner, full, opacity):
        """Keep point attributes stable when the motion cut changes ownership."""
        owner = self.motion.owner
        changed = previous_owner.ne(owner)
        # If the hard cut is unchanged, rebasing would write the original
        # values back to every row.
        if not bool(changed.any()):
            return
        # A cut must not silently change attributes by saturating the point
        # int16 stream. Validate every field before mutating any of them.
        targets = {}
        for name in FIELDS:
            parameter = full.parameters[name]
            region_q = torch.round(self.parameters[name].detach() / self.steps[name])
            target_q = (torch.round(parameter.detach() / self.steps[name]) +
                        region_q[previous_owner] - region_q[owner])
            mask = changed.reshape((-1,) + (1,)*(parameter.ndim-1))
            if bool((mask & target_q.abs().gt(32767)).any()):
                raise ValueError(f'Attribute gate recoding exceeds int16 range for {name}')
            targets[name] = target_q
        for name in FIELDS:
            parameter = full.parameters[name]
            rebased = _retain_bin_remainder(targets[name], parameter, self.steps[name])
            mask = changed.reshape((-1,) + (1,)*(parameter.ndim-1))
            parameter.copy_(torch.where(mask, rebased, parameter))
        region_q = torch.round(self.parameters['opacity'].detach() / self.steps['opacity'])
        old_offset = region_q[previous_owner] * self.steps['opacity']
        new_offset = region_q[owner] * self.steps['opacity']
        reference, parameter, step = opacity.alpha_reference, opacity.delta, opacity.step
        old_q = torch.round(parameter / step)
        alpha = (reference + old_q*step).clamp(0., 1.)
        logits = torch.logit(alpha.clamp(1e-6, 1-1e-6))
        new_alpha = torch.sigmoid(logits + old_offset - new_offset)
        new_q = torch.round((new_alpha-reference) / step).clamp(-32767, 32767)
        candidate = _retain_bin_remainder(new_q, parameter, step).clamp(-reference, 1-reference)
        # Identical offsets need no conversion, preserving exact zero/saturation bits.
        mask = changed[:, None] & old_offset.ne(new_offset)
        parameter.copy_(torch.where(mask, candidate, parameter))

    @torch.no_grad()
    def after_step(self, step, price):
        if step < 101:
            return
        active_nodes = torch.bincount(self.motion.owner, minlength=len(self.motion.parents)).gt(0)
        for name, value in self.parameters.items():
            value.clamp_(-32767*self.steps[name], 32767*self.steps[name])
            if name != 'opacity' and not self.config.storage_loss_enabled:
                continue
            state = self.motion.optimizer.state.get(value, {})
            if 'exp_avg_sq' not in state:
                continue
            group = self.group(name); lr = group['lr']
            metric = (state['exp_avg_sq']/(1-group['betas'][1]**int(state['step']))).sqrt().add(group['eps'])
            width = value.numel() // len(value)
            count = active_nodes.sum().clamp_min(1).to(value)*width
            l1_weight = self.opacity_l1_weight if name == 'opacity' else self.config.per_attribute_l1_weight
            threshold = lr*l1_weight/count/metric
            shrunk = value.sign()*(value.abs()-threshold).clamp_min(0)
            mask = active_nodes.reshape((-1,)+(1,)*(value.ndim-1))
            value.copy_(torch.where(mask, shrunk, value))
            if step % 20 == 0 or step == 200:
                mat = value.reshape(len(value), -1)
                gain = (.5*metric*value.square()/lr).reshape(len(value), -1).sum(-1)
                row_cost = raw_sparse_row_bytes(mat.shape[1])
                multiplier = 1. if name == 'opacity' else self.config.point_dc_rate_scale
                drop = active_nodes & torch.round(mat/self.steps[name]).ne(0).any(-1) & gain.lt(price*multiplier*row_cost)
                mat[drop] = 0

    @torch.no_grad()
    def symbols(self):
        occupied = torch.bincount(self.motion.owner, minlength=len(self.motion.parents)).gt(0)
        return {k: torch.where(occupied.reshape((-1,)+(1,)*(v.ndim-1)),
                torch.round(v.detach()/self.steps[k]), torch.zeros_like(v)).cpu() for k,v in self.parameters.items()}

    @torch.no_grad()
    def export(self, path, frame):
        symbols = self.symbols(); nodes = len(self.motion.parents)
        packet = encode_regional_full(symbols, self.steps, nodes, frame)
        if packet is None:
            return None
        Path(path).write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), format='RFA1',
            active_regions={k:int(v.reshape(nodes,-1).ne(0).any(-1).sum()) for k,v in symbols.items()})
