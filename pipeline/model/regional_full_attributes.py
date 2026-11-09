"""Quantized node attributes on the current motion cut, with a real packet."""
import struct
from pathlib import Path

import torch
from torch import nn

from .full_attribute_update import (FIELDS, _QuantizedFullAttribute,
    encode_full_attribute_delta, decode_full_attribute_delta)
from .sparse_updates import encode_opacity_delta
from .legacy_opacity_codec import decode_opacity_delta, _step
from .spatial_gated_motion import _HardForwardGateProxy, _retain_bin_remainder

MAGIC = b'RFA1'
HEADER = struct.Struct('<4sIII')


def _zeros(nodes, rest, device='cpu'):
    return dict(logscale=torch.zeros(nodes, 3, device=device),
                sh_dc=torch.zeros(nodes, 1, 3, device=device),
                sh_rest=torch.zeros(nodes, rest, 3, device=device))


def decode_regional_full(packet, raw, owner, nodes, frame):
    packet = bytes(packet)
    if len(packet) < HEADER.size:
        raise ValueError('Truncated regional attributes')
    magic, encoded_frame, n_attr, n_alpha = HEADER.unpack_from(packet)
    if (magic != MAGIC or encoded_frame != frame or
            len(packet) != HEADER.size + n_attr + n_alpha or not (n_attr or n_alpha)):
        raise ValueError('Invalid regional attribute packet')
    owner = torch.as_tensor(owner, dtype=torch.long, device='cpu')
    if owner.shape != (len(raw['opacity']),) or bool(((owner < 0) | (owner >= nodes)).any()):
        raise ValueError('Invalid regional attribute ownership')
    result = {k: raw[k].detach().cpu().clone() for k in (*FIELDS, 'opacity')}
    if n_attr:
        reference = _zeros(nodes, raw['sh_rest'].shape[1])
        delta = decode_full_attribute_delta(packet[HEADER.size:HEADER.size+n_attr], reference,
                                            frame_index=frame)
        for key in FIELDS:
            value = delta[key][owner]
            result[key] = torch.where(value.ne(0), result[key] + value, result[key])
    if n_alpha:
        reference = torch.zeros(nodes, 1)
        offset = decode_opacity_delta(packet[HEADER.size+n_attr:], reference,
                                      frame_index=frame).logits[owner]
        result['opacity'] = torch.where(offset.ne(0), result['opacity'] + offset, result['opacity'])
    return result


class RegionalFullAttributes:
    """Regional log-scale, SH and opacity-logit deltas; motion owns pose."""
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
        """Detached accounting for the same regional penalties used after Adam."""
        occupied = torch.bincount(self.motion.owner, minlength=len(self.motion.parents)).gt(0)
        fields = {}
        zero = self.motion.base_xyz.new_zeros(())
        for name, value in self.parameters.items():
            matrix = value.detach().reshape(len(value), -1)
            symbols = torch.round(matrix / self.steps[name])
            active = occupied & symbols.ne(0).any(-1)
            raw = active.sum().to(value) * (1 + 2 * matrix.shape[1])
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

    def gate_rate_terms(self, full, opacity, price, *, point_symbols=None):
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
                    node_cost += matrix.ne(0).any(-1).to(node_cost) * (1 + 2*matrix.shape[1]) * multiplier
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
                    cost += q.reshape(len(q), -1).ne(0).any(-1).to(cost) * (1 + 2*width) * multiplier
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
                    current += q.reshape(len(q), -1).ne(0).any(-1).to(current) * (1 + 2*width) * multiplier
            current += alpha_q.reshape(len(owner), -1).ne(0).any(-1).to(current) * 3
            occupied = torch.bincount(owner, minlength=nodes).gt(0)
            hard = current.sum() + node_cost[occupied].sum()
        expected = (weights * point_costs).sum() + (selections * node_cost).sum()
        # Point and regional hard penalties are already accounted for elsewhere.
        # This zero-forward term adds only the gate derivative, avoiding a second charge.
        proxy = price * (expected - expected.detach())
        return dict(proxy=proxy, hard_weighted_bytes=hard, expected_weighted_bytes=expected.detach())

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
                row_cost = 1 + 2*mat.shape[1]
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
        a = encode_full_attribute_delta(_zeros(nodes, 3), {k:symbols[k] for k in FIELDS},
            frame_index=frame, steps=tuple(self.steps[k] for k in FIELDS))
        b = encode_opacity_delta(torch.zeros(nodes,1), symbols['opacity'], frame_index=frame,
                                 step=self.steps['opacity'], mode='logit')
        if not a and not b:
            return None
        packet = HEADER.pack(MAGIC, frame, len(a or b''), len(b or b''))+(a or b'')+(b or b'')
        Path(path).write_bytes(packet)
        return dict(path=str(path), archive_bytes=len(packet), format='RFA1',
            active_regions={k:int(v.reshape(nodes,-1).ne(0).any(-1).sum()) for k,v in symbols.items()})
