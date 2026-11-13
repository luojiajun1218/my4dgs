"""训练阶段的运动和属性存储开销估算。

集中计算字节代理、稀疏代价及相关损失，供区域和逐点更新使用；
这些数值不等于实际压缩包大小，真实码流及字节数由 frame_stream 编码导出。
"""

import hashlib

import numpy as np
import torch

from .frame_stream_format import (CHECKSUM_BYTES, FIELDS, HEADER as FULL_ATTRIBUTE_HEADER,
    OPACITY_HEADER, FEATURE_MAGIC)
from .rigid_residual_tree import _quantized, _varint_length
from .frame_stream_encode import (_sparse_write, compress_motion_packet)
from .frame_stream_encode import (encode_unsigned_values)


def _bytes(values):
    return np.frombuffer(encode_unsigned_values(values), dtype=np.uint8)


def _streams(t, r):
    t, r = np.asarray(t, dtype=np.int64), np.asarray(r, dtype=np.int64)
    at, ar = t.any(1), r.any(1)
    rows = np.flatnonzero(at | ar)
    streams = [_bytes(np.diff(np.r_[-1, rows])),
               (at[rows].astype(np.uint8) | (ar[rows].astype(np.uint8) << 1))]
    for values, active in ((t, at), (r, ar)):
        zigzag = np.where(values >= 0, 2 * values, -2 * values - 1)
        streams.extend(_bytes(zigzag[active, axis]) for axis in range(3))
    return streams


class MotionCodecRate:
    def __init__(self, empty_packet, regions, points, refresh_steps=20):
        # The validated empty packet ends in gates, two zero sparse counts,
        # and a checksum. Keep the immutable prefix, including survivor edits.
        self.regions, self.points = int(regions), int(points)
        self.gate_bytes = (regions + 7) // 8
        self.prefix = bytes(empty_packet[:-(self.gate_bytes + 2 + CHECKSUM_BYTES)])
        self.refresh_steps = int(refresh_steps)
        self.last_refresh = -1
        self.tables = None
        self._device_tables = {}
        self.last_scale = 1.
        self.last_packet_bytes = 0
        self.last_entropy_bytes = 0.
        self.last_measure_step = -1

    def packet(self, rt, rr, gt, gr, opened):
        arrays = [np.asarray(value, dtype='<i2') for value in (rt, rr, gt, gr)]
        gates = np.packbits(np.asarray(opened, dtype=np.uint8), bitorder='little').tobytes()
        body = bytearray(self.prefix + gates)
        _sparse_write(body, arrays[0][1:], arrays[1][1:])
        _sparse_write(body, arrays[2], arrays[3])
        body.extend(hashlib.sha256(body).digest()[:CHECKSUM_BYTES])
        return compress_motion_packet(body, mode='auto')

    def measure(self, rt, rr, gt, gr, opened, step):
        arrays = [np.asarray(value, dtype='<i2') for value in (rt, rr, gt, gr)]
        packet = self.packet(*arrays, opened)
        self.last_packet_bytes = len(packet)
        self.last_measure_step = int(step)
        streams = [_streams(arrays[0][1:], arrays[1][1:]), _streams(arrays[2], arrays[3])]
        if self.tables is None or (step != self.last_refresh and step % self.refresh_steps == 0):
            tables = []
            for section_streams in streams:
                section = []
                for stream in section_streams:
                    previous = np.r_[256, stream[:-1]] if len(stream) else np.empty(0, dtype=int)
                    counts = np.bincount(previous * 256 + stream,
                        minlength=257 * 256).reshape(257, 256).astype(np.float64)
                    marginal = (np.bincount(stream, minlength=256) + .5) / (len(stream) + 128.)
                    probability = (counts + 8. * marginal[None]) / (counts.sum(1, keepdims=True) + 8.)
                    section.append(-np.log2(probability).astype(np.float32))
                tables.append(np.stack(section))
            self.tables = torch.from_numpy(np.stack(tables))
            self._device_tables.clear()
            self.last_refresh = int(step)
        self.last_entropy_bytes = 0.
        for section, section_streams in enumerate(streams):
            for index, stream in enumerate(section_streams):
                if len(stream):
                    previous = np.r_[256, stream[:-1]]
                    self.last_entropy_bytes += float(self.tables[section, index].numpy()[previous, stream].sum()) / 8.
        return len(packet)

    def _book(self, section, like):
        key = (str(like.device), like.dtype)
        if key not in self._device_tables:
            self._device_tables[key] = self.tables.to(device=like.device, dtype=like.dtype)
        return self._device_tables[key][section]

    @staticmethod
    def _previous(active):
        rows = torch.arange(len(active), device=active.device)
        seen = torch.where(active, rows, -1).cummax(0).values
        previous = torch.cat((seen.new_tensor([-1]), seen[:-1]))
        return previous

    @staticmethod
    def _words(values, previous, tables, max_bytes=3):
        """Per-row byte-context cost, including within-varint transitions."""
        lengths = torch.ones_like(values)
        for shift in range(7, 7 * max_bytes, 7):
            lengths += values.ge(1 << shift)
        prior = values[previous.clamp_min(0)]
        prior_lengths = lengths[previous.clamp_min(0)]
        last = (prior >> (7 * (prior_lengths - 1))) & 127
        context = torch.where(previous[:, None].ge(0), last, 256)
        axes = torch.arange(values.shape[1], device=values.device)[None]
        result = tables.new_zeros(values.shape)
        for index in range(max_bytes):
            byte = ((values >> (7 * index)) & 127) | (values.ge(1 << (7 * (index + 1))).long() << 7)
            price = tables[axes, context, byte]
            result += torch.where(lengths.gt(index), price, 0.)
            context = byte
        return result.sum(-1)

    def costs(self, t, r, st, sr, section, allowed=None):
        """Estimated bytes per sparse row; hard symbols, soft block support."""
        if section == 0 and len(t) == self.regions:
            # Region row zero is the identity and is absent from the section.
            allowed = None if allowed is None else allowed[1:]
            return torch.cat((t.new_zeros(1), self.costs(t[1:], r[1:], st[1:], sr[1:], section, allowed)))
        book = self._book(section, t)
        it, ir = t.detach().round().long(), r.detach().round().long()
        at, ar = it.ne(0).any(-1), ir.ne(0).any(-1)
        if allowed is not None:
            at, ar = at & allowed, ar & allowed
            st, sr = st * allowed, sr * allowed
        active = at | ar
        previous = self._previous(active)
        rows = torch.arange(len(active), device=t.device)
        gaps = (rows - previous)[:, None]
        gap_cost = self._words(gaps, previous, book[:1], max_bytes=max(1, (len(active).bit_length() + 6) // 7))
        flags = at.long() + 2 * ar.long()
        prior_flags = torch.where(previous.ge(0), flags[previous.clamp_min(0)], 256)
        flag_cost = book[1, prior_flags, flags]
        result = (st + sr - st * sr) * (gap_cost + flag_cost)
        for values, hard, soft, tables in ((it, at, st, book[2:5]), (ir, ar, sr, book[5:8])):
            zigzag = torch.where(values.ge(0), 2 * values, -2 * values - 1)
            result = result + soft * self._words(zigzag, self._previous(hard), tables)
        return result / 8.

    def capture(self):
        return dict(prefix=self.prefix, regions=self.regions, points=self.points,
                    refresh_steps=self.refresh_steps, last_refresh=self.last_refresh,
                    tables=self.tables, last_scale=self.last_scale,
                    last_packet_bytes=self.last_packet_bytes, last_entropy_bytes=self.last_entropy_bytes,
                    last_measure_step=self.last_measure_step)

    @classmethod
    def restore(cls, state):
        result = cls.__new__(cls)
        result.__dict__.update(state)
        result.last_measure_step = state.get('last_measure_step', -1)
        if result.tables is not None:
            result.tables = result.tables.detach().cpu()
        result.gate_bytes = (result.regions + 7) // 8
        result._device_tables = {}
        return result


def raw_sparse_row_bytes(width):
    """Estimated gap plus int16 values in a nonzero attribute row."""
    return 1 + 2 * width


def raw_motion_block_bytes(translation_support, rotation_support, index_and_flag_bytes):
    """Estimated sparse-row bytes for quantized translation and rotation."""
    t, r = translation_support, rotation_support
    return (t + r - t * r) * index_and_flag_bytes + 6 * (t + r)


def frame_byte_price(module, step):
    """Normalized, scheduled price used by the frame's storage terms."""
    cfg = module.config.rigid_tree
    ramp = min(max((step - cfg.rate_warmup_steps) / cfg.rate_ramp_steps, 0.), 1.)
    weight = getattr(module.motion, 'effective_rate_loss_weight', cfg.rate_loss_weight)
    return weight * ramp / module.motion.rate_normalization_bytes


def opacity_byte_price(module, step):
    if getattr(module.config.birth, 'population_rate_exempt', False):
        return 0.
    return module._rate_byte_price(step)


def sparse_block_support(value, step, *, continuous=None, differentiable=False, width=1.):
    """True nonzero block in forward; bin-scaled smooth L0 gradient backward."""
    symbols = torch.round(value / step)
    hard = symbols.ne(0).any(-1).to(value.dtype)
    if not differentiable:
        return hard
    continuous = value if continuous is None else continuous
    bins = continuous / step
    # Evaluate the sparse derivative on the transmitted bins. In a zero bin
    # abs has zero subgradient: a large Adam step must not turn an already-free
    # sub-bin update into an opposite-sign nonzero update solely due to rate.
    bins = bins + (symbols - bins).detach()
    soft = -torch.expm1(-bins.abs().sum(-1) / width)
    return hard + (soft - soft.detach())


def opacity_rate_terms(self, byte_price, width=1.):
    value = self.delta[self.rows]
    if self.config.proximal_regularization or self.refit_support is not None:
        value = value.detach()
    support = sparse_block_support(value, self.step, differentiable=True, width=width)
    nonzero = torch.round(value / self.step).ne(0).sum()
    # Gap + int16 is three raw bytes for a typical row. Actual gap lengths
    # and compressed bytes are checked at export, not assumed to be equal.
    raw_bytes = raw_sparse_row_bytes(1) * support.sum() + OPACITY_HEADER.size * nonzero.gt(0).to(value)
    l1 = self.config.l1_weight * value.abs().mean()
    return dict(loss=raw_bytes * byte_price + l1, l1_loss=l1,
                raw_bytes_surrogate=raw_bytes,
                active_rows=nonzero)


def point_logscale_rate_terms(self, byte_price):
    if not self.ready:
        zero = torch.zeros(())
        return dict(loss=zero, active_rows=zero, value_bytes_surrogate=zero)
    current = self.quantized_delta()[self.survivor_rows]
    active = current.ne(0).any(-1).sum()
    value_bytes = active.to(current) * self.channels
    # Selection is handled by proximal Adam and a hard row budget.  Keep
    # this literal forward price detached, matching the diagnostic fit.
    loss = value_bytes.detach() * byte_price
    return dict(loss=loss, active_rows=active, value_bytes_surrogate=value_bytes)


def point_dc_rate_terms(self, byte_price, width=1.):
    terms = self.scalar.rate_terms(byte_price, width)
    tag = terms['active_rows'].gt(0).to(terms['loss']) * len(FEATURE_MAGIC)
    return {**terms, 'loss': terms['loss'] + tag * byte_price,
            'raw_bytes_surrogate': terms['raw_bytes_surrogate'] + tag}


@torch.no_grad()
def point_full_attribute_rate_terms(self, byte_price, *, include_scalar_counts=True, include_symbols=False):
    """Report raw sparse-block price; Adam proximal performs its gradient step."""
    fields = {}
    symbols = {} if include_symbols else None
    for name, step in zip(FIELDS, self.steps):
        value = self.parameters[name].detach()
        matrix = value.reshape(len(value), -1)
        integers = torch.round(matrix / step)
        if symbols is not None:
            symbols[name] = integers.reshape_as(value)
        nonzero = integers.ne(0)
        active = nonzero.any(-1)
        rows = active.sum()
        # One-byte typical gap plus the full int16 attribute row. Actual
        # varint gaps and FUA3 compression are measured on export.
        raw = rows.to(value) * raw_sparse_row_bytes(matrix.shape[1])
        l1 = (self.config.per_attribute_l1_weight * value.abs().mean()
              if self.config.storage_loss_enabled and value.numel() else value.sum() * 0)
        multiplier = self.point_rate_multiplier(name)
        fields[name] = dict(active_rows=rows, raw_bytes_surrogate=raw,
                            l1_loss=l1, rate_multiplier=value.new_tensor(multiplier))
        if include_scalar_counts:
            fields[name]['active_scalars'] = nonzero.sum()
    first = self.parameters[FIELDS[0]]
    raw = sum((item['raw_bytes_surrogate'] for item in fields.values()), first.new_zeros(()))
    raw = raw + raw.gt(0).to(raw) * (FULL_ATTRIBUTE_HEADER.size + 12)
    l1 = sum((item['l1_loss'] for item in fields.values()), first.new_zeros(()))
    weighted_raw = sum((item['raw_bytes_surrogate'] * item['rate_multiplier']
                        for item in fields.values()), first.new_zeros(()))
    weighted_raw = weighted_raw + raw.gt(0).to(raw) * (FULL_ATTRIBUTE_HEADER.size + 12)
    loss = weighted_raw * byte_price + l1 if self.config.storage_loss_enabled else raw * 0
    result = dict(loss=loss, raw_bytes_surrogate=raw, l1_loss=l1, fields=fields)
    if symbols is not None:
        result['symbols'] = symbols
    return result


@torch.no_grad()
def regional_full_attribute_rate_terms(self, price):
    """Detached accounting for the same regional penalties used after Adam."""
    occupied = torch.bincount(self.motion.owner, minlength=len(self.motion.parents)).gt(0)
    fields = {}
    zero = self.motion.base_xyz.new_zeros(())
    for name, value in self.parameters.items():
        matrix = value.detach().reshape(len(value), -1)
        symbols = torch.round(matrix / self.steps[name])
        active = occupied & symbols.ne(0).any(-1)
        raw = active.sum().to(value) * raw_sparse_row_bytes(matrix.shape[1])
        count = occupied.sum().clamp_min(1).to(value) * matrix.shape[1]
        weight = self.opacity_l1_weight if name == 'opacity' else self.config.per_attribute_l1_weight
        multiplier = 1. if name == 'opacity' else self.config.point_dc_rate_scale
        enabled = name == 'opacity' or self.config.storage_loss_enabled
        l1 = (matrix.abs() * occupied[:, None]).sum() * weight / count if enabled else zero
        fields[name] = dict(active_regions=active.sum(), raw_bytes_surrogate=raw,
            byte_price=value.new_tensor(price * multiplier if enabled else 0.), l1_loss=l1)
    raw = sum((item['raw_bytes_surrogate'] for item in fields.values()), zero)
    l1 = sum((item['l1_loss'] for item in fields.values()), zero)
    penalty = sum((item['raw_bytes_surrogate'] * item['byte_price'] for item in fields.values()), zero) + l1
    return dict(loss=penalty, raw_bytes_surrogate=raw, l1_loss=l1, fields=fields)


def regional_full_attribute_gate_rate_terms(self, full, opacity, price, *, point_symbols=None):
    """Gate-only proxy for regional bytes plus appearance-preserving point rebasing.

    Costs use raw sparse rows, not compressed packet bytes.  Candidate point
    residuals are held at the current reconstructed attributes; opacity's
    alpha/logit conversion incurs point-grid rounding near saturation.
    """
    motion = self.motion
    if motion._hard_gate_frozen or not motion.gate_logits.requires_grad:
        zero = motion.base_xyz.new_zeros(())
        return dict(proxy=zero, hard_weighted_bytes=zero, expected_weighted_bytes=zero)
    probabilities = motion._rate_gate_probabilities(motion._frontier_probabilities())
    weights = motion._path_weights(probabilities)
    selections = motion._region_selection_probabilities(probabilities, include_ancestors=False)
    paths, owner = motion.spatial_paths, motion.owner
    with torch.no_grad():
        nodes = len(motion.parents)
        node_cost = motion.base_xyz.new_zeros(nodes)
        point_symbols = {} if point_symbols is None else dict(point_symbols)
        regional_symbols = {}
        for name in (*FIELDS, 'opacity'):
            value = self.parameters[name].detach()
            q = torch.round(value / self.steps[name])
            regional_symbols[name] = q
            matrix = q.reshape(nodes, -1)
            multiplier = 1. if name == 'opacity' else self.config.point_dc_rate_scale
            enabled = name == 'opacity' or self.config.storage_loss_enabled
            if enabled:
                node_cost += matrix.ne(0).any(-1).to(node_cost) * raw_sparse_row_bytes(matrix.shape[1]) * multiplier
            if name != 'opacity' and name not in point_symbols:
                point_symbols[name] = torch.round(full.parameters[name].detach() / self.steps[name])
        alpha_step = opacity.step
        alpha_ref = opacity.alpha_reference.detach()
        alpha_q = torch.round(opacity.delta.detach() / alpha_step)
        point_symbols['opacity'] = alpha_q
        point_alpha = (alpha_ref + alpha_q * alpha_step).clamp(0., 1.)
        current_logit = torch.logit(point_alpha.clamp(1e-6, 1-1e-6))
        current_logit = current_logit + regional_symbols['opacity'][owner] * self.steps['opacity']
        # The current point plus its owner is identical for every
        # candidate path; only the proposed region differs.
        point_plus_owner = {name: point_symbols[name] + regional_symbols[name][owner]
                            for name in FIELDS} if self.config.storage_loss_enabled else {}
        candidates = []
        for node in paths.unbind(1):
            cost = motion.base_xyz.new_zeros(len(owner))
            for name in FIELDS:
                if not self.config.storage_loss_enabled:
                    continue
                q = (point_plus_owner[name] - regional_symbols[name][node]).clamp(-32767, 32767)
                width = q.numel() // len(q)
                multiplier = self.config.point_dc_rate_scale * full.point_rate_multiplier(name)
                cost += q.reshape(len(q), -1).ne(0).any(-1).to(cost) * raw_sparse_row_bytes(width) * multiplier
            alpha_candidate = torch.sigmoid(current_logit - regional_symbols['opacity'][node] * self.steps['opacity'])
            candidate_q = torch.round((alpha_candidate - alpha_ref) / alpha_step).clamp(-32767, 32767)
            candidate_q = torch.where(node[:, None] == owner[:, None], alpha_q, candidate_q)
            cost += candidate_q.reshape(len(owner), -1).ne(0).any(-1).to(cost) * 3
            candidates.append(cost)
        point_costs = torch.stack(candidates, dim=1)
        current = motion.base_xyz.new_zeros(len(owner))
        for name in FIELDS:
            q = point_symbols[name]
            width = q.numel() // len(q)
            multiplier = self.config.point_dc_rate_scale * full.point_rate_multiplier(name)
            if self.config.storage_loss_enabled:
                current += q.reshape(len(q), -1).ne(0).any(-1).to(current) * raw_sparse_row_bytes(width) * multiplier
        current += alpha_q.reshape(len(owner), -1).ne(0).any(-1).to(current) * 3
        occupied = torch.bincount(owner, minlength=nodes).gt(0)
        hard = current.sum() + node_cost[occupied].sum()
    expected = (weights * point_costs).sum() + (selections * node_cost).sum()
    # Point and regional hard penalties are already accounted for elsewhere.
    # This zero-forward term adds only the gate derivative, avoiding a second charge.
    proxy = price * (expected - expected.detach())
    return dict(proxy=proxy, hard_weighted_bytes=hard, expected_weighted_bytes=expected.detach())


def survivor_mapping_bytes(self):
    """Cache the actual deletion-gap section once per frame or restore."""
    from .frame_stream_format import (_uvar)
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


def motion_metadata_bytes(self):
    births = [] if self.pending_birth_sources is None else self.pending_birth_sources.tolist()
    from .frame_stream_format import (_uvar)
    return (4 + 32 + 40 + 16 + 1 + self._survivor_mapping_bytes +
            (len(self.parents) + 7) // 8 +
            sum(len(_uvar(value)) for value in
                (self.frame_index, len(self.parents), len(self.owner), len(births))) +
            sum(len(_uvar(source)) + 28 for source in births))


def motion_rate_terms(self, step, diagnostics=True):
    from .spatial_motion_training import _hard_value_soft_gradient
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
    rate_probabilities = self._rate_gate_probabilities(probabilities)
    def support(value, scale, continuous=None):
        return sparse_block_support(value, scale, continuous=continuous,
            differentiable=self.config.differentiable_rate_support,
            width=self.config.sparse_rate_width_bins)

    residual_bytes = self.base_xyz.new_zeros(())
    relative = self.base_xyz.new_zeros(())
    hard_residual_bytes = self.base_xyz.new_zeros(())
    hard_relative = self.base_xyz.new_zeros(())
    candidate_relative = self.base_xyz.new_zeros(())
    candidate_bytes = self.base_xyz.new_zeros(())
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
        rate_weights = None if self._hard_gate_frozen else self._path_weights(rate_probabilities)
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
                    if codec else raw_motion_block_bytes(st, sr, self._gs_id_lengths + 1))
            magnitude = (ct / self.extent).abs().sum(-1) + cr.abs().sum(-1)
            weight = weights[:, level]
            rate_weight = rate_weights[:, level]
            residual_bytes = residual_bytes + (rate_weight * cost).sum()
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
                if codec else raw_motion_block_bytes(st, sr, self._gs_id_lengths + 1))
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
                if codec else raw_motion_block_bytes(st, sr, self._gs_id_lengths + 1))
        weight = 1. if weights is None else weights[:, level]
        rate_weight = 1. if rate_weights is None else rate_weights[:, level]
        residual_bytes = residual_bytes + (rate_weight * (
            cost.detach() if gate_only_soft_cost or self.shared_image_recode else cost)).sum()
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
    refine_closed = self.refining_unselected_candidates
    if (self.relative_regions and (not self.joint_image_gradients or refine_closed)
            and (not self._hard_gate_frozen or refine_closed)):
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
            if refine_closed:
                # Fit unopened alternatives to the same residual byte objective
                # that selects the frontier, rather than L1 alone. Targets and
                # ancestors stay detached: this cannot move the rendered field.
                st, sr = support(error_t, gs_steps[:3]), support(error_r, gs_steps[3:])
                cost = (self._codec_rate.costs(error_t / gs_steps[:3],
                            error_r / gs_steps[3:], st, sr, 1) if codec else
                        raw_motion_block_bytes(st, sr, self._gs_id_lengths + 1))
                candidate_bytes = candidate_bytes + (mask * cost).sum() / self.spatial_paths.shape[1]
        if refine_closed:
            # Charge each candidate block once, not once per member. Opening
            # a region must save enough point bytes to pay for its predictor.
            active = (~used & self.node_counts.gt(0)).clone()
            active[0] = False
            st, sr = support(local_rt, steps[:3]), support(local_rr, steps[3:])
            rid = torch.cat((self._region_id_lengths.new_zeros(1), self._region_id_lengths))
            cost = (self._codec_rate.costs(local_rt / steps[:3], local_rr / steps[3:], st, sr, 0)
                    if codec else raw_motion_block_bytes(st, sr, rid + 1))
            candidate_bytes = candidate_bytes + (active.to(cost) * cost).sum() / self.spatial_paths.shape[1]
    regional_weight = self._region_selection_probabilities(rate_probabilities,
        include_ancestors=self.relative_regions)
    if self.joint_image_gradients and not self.soft_image_active and not self.private_full_motion:
        selected = self._stored_region_support().to(rt)
        regional_weight = _hard_value_soft_gradient(selected, regional_weight)
    # A high frame-level multiplier should still prune expensive point
    # residuals without forcing the spatial predictor itself to collapse to
    # a coarse cut. The same discounted gate derivative is used above for
    # residual rows and here for region blocks. Forward bytes and gradients
    # into the transmitted values themselves remain unchanged.
    selection_scale = self.config.spatial_gate.region_selection_rate_gradient_scale
    st = support(local_rt, steps[:3], continuous=self.region_translation)
    sr = support(local_rr, steps[3:], continuous=self.region_rotation)
    rid = torch.cat((self._region_id_lengths.new_zeros(1), self._region_id_lengths))
    regional_cost = (self._codec_rate.costs(local_rt / steps[:3], local_rr / steps[3:], st, sr, 0)
                     if codec else raw_motion_block_bytes(st, sr, rid + 1))
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
    result = dict(rate_loss=rate * self.effective_rate_loss_weight * ramp,
        relative_motion_loss=relative * self.config.relative_motion_loss_weight * ramp,
        rate_bytes_surrogate=soft_bytes, rate_fraction=rate, relative_motion=relative)
    if self.joint_image_gradients or self.relative_regions or self.shared_image_recode:
        result['candidate_motion_loss'] = ramp * (
            candidate_relative * self.config.relative_motion_loss_weight +
            candidate_bytes / self.rate_normalization_bytes * self.effective_rate_loss_weight *
            (codec_scale if codec else 1.))
        result['candidate_rate_bytes_surrogate'] = candidate_bytes.detach()
    if codec:
        result['rate_codec_measured'] = self.base_xyz.new_tensor(codec_bytes is not None)
    if diagnostics:
        result.update(self._support_diagnostics(probabilities))
        result.update(rate_bytes_hard=self._hard_motion_bytes() + self._metadata_bytes(),
            rate_normalization_gs_count=self.base_xyz.new_tensor(self.rate_normalization_count),
            rate_byte_price=self.base_xyz.new_tensor(self.effective_rate_loss_weight * ramp / self.rate_normalization_bytes),
            region_selection_rate_gradient_scale=self.base_xyz.new_tensor(selection_scale))
        if codec:
            result.update(rate_bytes_raw=result['rate_bytes_hard'],
                          rate_bytes_hard=self.base_xyz.new_tensor(codec_bytes),
                          rate_entropy_bytes=self.base_xyz.new_tensor(self._codec_rate.last_entropy_bytes),
                          rate_codec_gradient_scale=self.base_xyz.new_tensor(codec_scale))
    return result


def predictor_rd_rate_terms(self, step, diagnostics):
    """Learn cuts using the cost of sparsifying each candidate's proposal.

    All candidates fit the same unshrunk photometric target, including
    branches whose ancestors are closed. Conditional fitting is separate
    from the expected path cost so ancestors cannot starve their training.
    """
    from .spatial_motion_training import predictor_cost
    zero = self.base_xyz.new_zeros(())
    result = dict(rate_loss=zero, relative_motion_loss=zero, candidate_motion_loss=zero)
    state = getattr(self, '_predictor_fit_state', None)
    ramp = min(max((step - self.config.rate_warmup_steps) / self.config.rate_ramp_steps, 0.), 1.)
    price = self.effective_rate_loss_weight * ramp / self.rate_normalization_bytes
    if state is not None:
        rt, rr = self._regional_predictions()
        prediction = torch.cat((rt, rr), -1)
        l1 = self.gs_steps.new_tensor([1 / self.extent] * 3 + [1.] * 3)
        l1 *= self.config.relative_motion_loss_weight * ramp / len(self.owner)
        probabilities = self._frontier_probabilities()
        if self._hard_gate_frozen:
            probabilities = probabilities.detach()
        weights = None if self._hard_gate_frozen else self._path_weights(probabilities)
        rate_weights = (None if self._hard_gate_frozen else
                        self._path_weights(self._rate_gate_probabilities(probabilities)))
        paths = [self.owner] if self._hard_gate_frozen else self.spatial_paths.unbind(1)
        expected, fitted, expected_bytes = zero, zero, zero
        observed = self.observations.gt(0).to(rt)
        for level, node in enumerate(paths):
            cost, _, byte_cost = predictor_cost(state['target'], prediction[node],
                state['metric'], self.gs_steps, l1, price, self._gs_id_lengths + 1)
            cost = cost * observed
            weight = 1. if weights is None else weights[:, level]
            rate_weight = 1. if rate_weights is None else rate_weights[:, level]
            priced_bytes = byte_cost * observed
            nonrate = cost - price * priced_bytes
            expected = expected + (weight * nonrate.detach()).sum()
            expected = expected + (rate_weight * (price * priced_bytes).detach()).sum()
            expected_bytes = expected_bytes + (weight * priced_bytes).sum()
            fitted = fitted + cost.sum() / len(paths)
        active_t, active_r = rt.ne(0).any(-1).to(rt), rr.ne(0).any(-1).to(rr)
        ids = torch.cat((self._region_id_lengths.new_zeros(1), self._region_id_lengths)) + 1
        region_cost = (active_t + active_r - active_t * active_r) * ids + 6 * (active_t + active_r)
        selection = self._region_selection_probabilities(
            self._rate_gate_probabilities(probabilities))
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
            region_selection_rate_gradient_scale=self.base_xyz.new_tensor(
                self.config.spatial_gate.region_selection_rate_gradient_scale),
            rate_normalization_gs_count=self.base_xyz.new_tensor(self.rate_normalization_count),
            rate_normalization_bytes=self.base_xyz.new_tensor(self.rate_normalization_bytes))
    return result


def hard_motion_bytes(self):
    rt, rr, gt, gr = self._values(hard=True)
    r_t = rt[1:].ne(0).any(-1)
    r_r = rr[1:].ne(0).any(-1)
    g_t = gt.ne(0).any(-1)
    g_r = gr.ne(0).any(-1)
    def sparse_bytes(active_t, active_r):
        rows = torch.where(active_t | active_r)[0]
        # Each sparse section restarts at -1: first gap=row+1, then
        # positive differences between consecutive active row IDs.
        gaps = torch.diff(torch.cat((rows.new_tensor([-1]), rows)))
        id_bytes = torch.ones_like(gaps)
        for shift in range(7, 63, 7):
            id_bytes += (gaps >= (1 << shift)).to(id_bytes)
        return (id_bytes.sum() + rows.numel() +
                6 * (active_t.sum() + active_r.sum()) +
                _varint_length(rows.numel()))
    return sparse_bytes(r_t, r_r) + sparse_bytes(g_t, g_r)
