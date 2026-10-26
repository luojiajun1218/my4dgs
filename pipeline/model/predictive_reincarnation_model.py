"""Independent predictive colour branches with an intact native parent support.

Unlike the V9 mass proxy, details are ordinary additional Gaussians.  The parent
is never narrowed or reduced to initialize them.  Phase 0 fits parent material,
phase 1 fits independent details/novel points, and phase 2 permits alpha
competition while locking novel geometry.  The controller owns the A/B split.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES, attribute_steps
from .conservative_family import _clone_tree
from .reincarnation_model import (ReincarnationModel, C0, C1, _empty_raw, _ste,
                                  _sh1_vector, project_sh1_sphere, sh1_sphere_range)


ENDPOINT_SCOPE = ('P mothers and all their details use immutable original-source '
                  'SH1 priors; N material is held at its current mature value')


def _zero_opacity(value):
    """Finite wire logits whose sigmoid underflows to exactly zero in this dtype."""
    return torch.full_like(value, 2*math.log(torch.finfo(value.dtype).tiny))


class PredictiveReincarnationModel(ReincarnationModel):
    """V10 model, retaining the established V9 factory/state/subset protocol."""

    def __init__(self, sh_degree):
        super().__init__(sh_degree)
        self.predictive_phase = 0
        self._predictive_endpoints = None
        self.predictive_weights = None

    @property
    def detail_rows(self):
        parts = [rows[1:] for rows in self.reincarnation_groups if len(rows) > 1]
        return torch.cat(parts) if parts else self._xyz.new_empty(0, dtype=torch.long)

    @property
    def detail_mask(self):
        result = torch.zeros(len(self._xyz), device=self._xyz.device, dtype=torch.bool)
        result[self.detail_rows] = True
        return result

    def install_reincarnation_state(self, state):
        super().install_reincarnation_state(state)
        self.predictive_phase = int(state.get('predictive_phase', 0))
        self._predictive_endpoints = _clone_tree(state.get('predictive_endpoints'), self._xyz.device)
        self.predictive_weights = _clone_tree(state.get('predictive_weights'), self._xyz.device)
        self.set_training_stage(self.predictive_phase)
        return self

    def reincarnation_state(self, device='cpu'):
        state = super().reincarnation_state(device)
        state.update(predictive_phase=self.predictive_phase,
            predictive_endpoints=_clone_tree(self._predictive_endpoints, device),
            predictive_weights=_clone_tree(self.predictive_weights, device),
            predictive_version='dual_predictive_support_v10', endpoint_scope=ENDPOINT_SCOPE)
        return state

    def set_training_stage(self, phase):
        phase = int(phase)
        if phase not in (0, 1, 2):
            raise ValueError('Predictive reincarnation phase must be 0, 1 or 2')
        self.predictive_phase = phase
        self._reincarnation_terminal = phase == 2
        for parameter in (self.p_common_dc, self.p_common_sh1):
            parameter.requires_grad_(phase == 0)
            parameter.grad = None
        # The inherited family latents remain empty: V10 never evaluates Li2,
        # subtracts mother mass, or derives a paired child covariance.
        for parameter in (self.family_tangent_offset, self.family_gate_logit, self.family_contrast):
            parameter.requires_grad_(False)
            parameter.grad = None
        for attr in RAW_ATTRIBUTES:
            getattr(self, attr).grad = None
        # A detached row shares its Parameter/Adam step with live details.
        # Zero gradients alone do not stop previously accumulated momentum.
        if phase == 2 and self.optimizer is not None:
            for attr in ('_xyz', '_rotation', '_scaling'):
                parameter = getattr(self, attr)
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    value = self.optimizer.state.get(parameter, {}).get(key)
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value[:self.novel_count].zero_()
        self._clear_reincarnation_cache()
        return self

    def _prior_features(self):
        n = self.novel_count
        root = self._point_root
        dc = torch.cat((self.novel_initial_raw['sh_dc'], self.support_anchor_raw['sh_dc'][root]), 0)
        rest = torch.cat((self.novel_initial_raw['sh_rest'], self.support_anchor_raw['sh_rest'][root]), 0)
        if n:
            ndc, nrest, _ = project_sh1_sphere(dc[:n], rest[:n],
                dc.new_zeros(n, 3), dc.new_ones(n, 3))
            dc = torch.cat((ndc, dc[n:]), 0)
            rest = torch.cat((nrest, rest[n:]), 0)
        return dc, rest

    def _radiance_bounds(self):
        dc, rest = self._prior_features()
        low, high = sh1_sphere_range(dc, rest)
        low, high = low.clamp_max(0.), high.clamp_min(1.)
        if self.novel_count:
            low[:self.novel_count] = 0.
            high[:self.novel_count] = 1.
        return low, high

    def _bounded_features(self, dc, rest, config=None, predictor=None):
        original_dc, original_rest = dc, rest
        prior_dc, prior_rest = self._prior_features()
        if config is not None:
            step = dc.new_tensor(config.new_sh_step)
            dc = predictor['sh_dc']+((dc-predictor['sh_dc'])/step).round()*step
            rest = predictor['sh_rest']+((rest-predictor['sh_rest'])/step).round()*step
        low, high = self._radiance_bounds()
        dc, rest, _ = project_sh1_sphere(dc, rest, low, high)
        pristine = ((original_dc.detach() == prior_dc).all((1, 2)) &
                    (original_rest.detach() == prior_rest).all((1, 2)))
        dc = torch.where(pristine[:, None, None], prior_dc, dc)
        rest = torch.where(pristine[:, None, None], prior_rest, rest)
        return _ste(original_dc, dc), _ste(original_rest, rest)

    def _maturity_opacity_floor(self):
        rows = [torch.arange(self.novel_count, device=self._xyz.device)]
        floors = [self.novel_initial_raw['opacity']]
        for root, group in enumerate(self.reincarnation_groups):
            if len(group) == 3:
                alpha = float(self.reincarnation_specs[root].get('detail_birth_alpha', 0.))
                value = self._opacity.new_full((2, 1), alpha)
                floor = value.log()-torch.log1p(-value) if alpha else _zero_opacity(value)
                rows.append(group[1:]); floors.append(floor)
        return torch.cat(rows), torch.cat(floors)

    def _phenotype(self, config=None, extent=None, reference=None, source_rows=None):
        self._check_degree()
        n, phase = self.novel_count, self.predictive_phase
        raw = dict(zip(RAW_FIELDS, (getattr(self, attr) for attr in RAW_ATTRIBUTES)))
        result = {name: value.detach().clone() if phase == 0 else value.clone() for name, value in raw.items()}
        mothers = self.reincarnation_rows
        # Parent pose/covariance always comes from the untouched native source.
        for name in ('xyz', 'rotation', 'logscale'):
            result[name] = result[name].index_copy(0, mothers, self.support_anchor_raw[name])
        parent_opacity = self.support_anchor_raw['opacity'] if phase < 2 else raw['opacity'][mothers]
        result['opacity'] = result['opacity'].index_copy(0, mothers, parent_opacity)
        result['sh_dc'] = result['sh_dc'].index_copy(0, mothers, self.p_common_dc)
        result['sh_rest'] = result['sh_rest'].index_copy(0, mothers, self.p_common_sh1)
        if phase == 0:
            inactive = torch.ones(len(self._xyz), device=self._xyz.device, dtype=torch.bool)
            inactive[mothers] = False
            for name, value in result.items():
                mask = inactive.reshape((-1,)+(1,)*(value.ndim-1))
                result[name] = torch.where(mask, value.detach(), value)
            result['opacity'] = torch.where(inactive[:, None], _zero_opacity(result['opacity']), result['opacity'])
        elif phase == 2 and n:
            for name in ('xyz', 'rotation', 'logscale'):
                result[name] = torch.cat((result[name][:n].detach(), result[name][n:]), 0)
        if phase == 1:
            protected, floor = self._maturity_opacity_floor()
            live = result['opacity'][protected]
            result['opacity'] = result['opacity'].index_copy(0, protected, _ste(live, torch.maximum(live, floor)))
        predictor = None
        if config is not None:
            sources = torch.as_tensor(source_rows, device=self._xyz.device, dtype=torch.long)
            if (sources.shape != (len(self._xyz),) or
                    bool(((sources < 0) | (sources >= len(reference['xyz']))).any())):
                raise ValueError('Predictive codec sources must address the actual reference')
            predictor = {name: value[sources].detach() for name, value in reference.items()}
            # N keeps the existing ordinary quantizer.  Native parents and
            # detail support are explicit ATTR when their physical raw values
            # are not exactly representable by the source grid.
            steps = attribute_steps(config, extent)
            for name in ('xyz', 'rotation', 'logscale', 'opacity'):
                live, p = result[name][:n], predictor[name][:n]
                if name == 'rotation':
                    if not getattr(config, 'rotation_updates', True):
                        target = p
                    else:
                        live, p = F.normalize(live, dim=-1), F.normalize(p, dim=-1)
                        live = torch.where((live*p).sum(-1, keepdim=True) < 0, -live, live)
                        target = p+((live-p)/steps[name]).round()*steps[name]
                else:
                    target = p+((live-p)/steps[name]).round()*steps[name]
                result[name] = torch.cat((_ste(live, target), result[name][n:]), 0)
            if phase == 0 and n:
                result['opacity'] = torch.cat((_zero_opacity(result['opacity'][:n]), result['opacity'][n:]), 0)
            elif phase == 1:
                # N's ordinary opacity quantizer must not round below its
                # physical birth floor either; exact ATTR is the fallback.
                protected, floor = self._maturity_opacity_floor()
                live = result['opacity'][protected]
                result['opacity'] = result['opacity'].index_copy(0, protected, _ste(live, torch.maximum(live, floor)))
        dc, rest = self._bounded_features(result['sh_dc'], result['sh_rest'], config, predictor)
        result['sh_dc'], result['sh_rest'] = dc, rest
        return result

    @torch.no_grad()
    def project_parameters(self):
        dc, rest = self._features_dc.clone(), self._features_rest.clone()
        dc[self.reincarnation_rows] = self.p_common_dc
        rest[self.reincarnation_rows] = self.p_common_sh1
        low, high = self._radiance_bounds()
        dc, rest, _ = project_sh1_sphere(dc, rest, low, high)
        self._features_dc.copy_(dc)
        self._features_rest.copy_(rest)
        self.p_common_dc.copy_(dc[self.reincarnation_rows])
        self.p_common_sh1.copy_(rest[self.reincarnation_rows])
        if self.predictive_phase == 1:
            protected, floor = self._maturity_opacity_floor()
            self._opacity[protected] = torch.maximum(self._opacity[protected], floor)
        self._clear_reincarnation_cache()
        return self

    project_ = project_parameters

    def install_details_from_coarse(self, root_indices=None, detail_alpha=.1):
        """Return an independent model: retain each parent, append two details.

        Details start at the original center/orientation.  Each halves a
        different one of the two largest principal scales; there is no world
        displacement and no compensating reduction of the parent opacity.
        """
        roots = len(self.reincarnation_groups)
        chosen = torch.arange(roots, device=self._xyz.device) if root_indices is None else torch.as_tensor(
            root_indices, device=self._xyz.device, dtype=torch.long).flatten()
        if len(chosen.unique()) != len(chosen) or bool(((chosen < 0) | (chosen >= roots)).any()):
            raise ValueError('Detail roots must be valid and unique')
        if not math.isfinite(float(detail_alpha)) or not 0 <= float(detail_alpha) < 1:
            raise ValueError('Detail seed alpha must lie in [0, 1)')
        selected = set(chosen.tolist())
        n, groups, row_map, parts = self.novel_count, [], list(range(self.novel_count)), {k:[] for k in RAW_FIELDS}
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            parts[name].append(getattr(self, attr)[:n].detach().clone())
        next_row = n
        for root, old in enumerate(self.reincarnation_groups):
            add = root in selected and len(old) == 1
            count = 3 if add or len(old) == 3 else 1
            rows = torch.arange(next_row, next_row+count, device=self._xyz.device)
            groups.append(rows)
            next_row += count
            row_map.extend(old.tolist())
            if add:
                row_map.extend([-1, -1])
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                value = getattr(self, attr)[old].detach().clone()
                if name in ('xyz', 'rotation', 'logscale'):
                    value[0] = self.support_anchor_raw[name][root]
                elif name == 'sh_dc':
                    value[0] = self.p_common_dc[root].detach()
                elif name == 'sh_rest':
                    value[0] = self.p_common_sh1[root].detach()
                if add:
                    child = value[:1].repeat((2,)+(1,)*(value.ndim-1))
                    if name == 'logscale':
                        axes = torch.argsort(self.support_anchor_raw['logscale'][root], descending=True, stable=True)[:2]
                        child[torch.arange(2, device=axes.device), axes] -= math.log(2.)
                    elif name == 'opacity':
                        alpha = child.new_full(child.shape, float(detail_alpha))
                        child = (alpha.log()-torch.log1p(-alpha)) if detail_alpha else _zero_opacity(child)
                    value = torch.cat((value, child), 0)
                parts[name].append(value)
        result = type(self)(self.max_sh_degree)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(result, attr, nn.Parameter(torch.cat(parts[name], 0)))
        result.active_sh_degree, result.spatial_lr_scale = self.active_sh_degree, self.spatial_lr_scale
        result.max_radii2D = result._xyz.new_zeros(next_row)
        result.xyz_gradient_accum = result._xyz.new_zeros(next_row, 1)
        result.denom = result._xyz.new_zeros(next_row, 1)
        source_tail = [self.source_rows[old[0]].expand(len(rows))
                       for old, rows in zip(self.reincarnation_groups, groups)]
        result.source_rows = torch.cat((self.source_rows[:n], *source_tail)).clone()
        state = self.reincarnation_state(device=self._xyz.device)
        state.update(groups=groups, predictive_phase=1, predictive_endpoints=None, predictive_weights=None)
        for root in selected:
            if len(self.reincarnation_groups[root]) == 1:
                state['specs'][root]['detail_birth_alpha'] = float(detail_alpha)
        result.install_reincarnation_state(state)
        self._copy_expanded_optimizer(result, torch.tensor(row_map, device=self._xyz.device))
        return result

    def install_families_from_coarse(self, root_indices, tangent_offsets=None, gate_logits=None):
        if tangent_offsets is not None or gate_logits is not None:
            raise ValueError('V10 details use intact parents, without V9 offset/mass-gate initialization')
        return self.install_details_from_coarse(root_indices)

    def _copy_expanded_optimizer(self, result, row_map):
        if self.optimizer is None:
            return
        result.training_setup(self._training_args)
        old_groups = {group['name']: group for group in self.optimizer.param_groups}
        for group in result.optimizer.param_groups:
            old = old_groups[group['name']]
            for key, value in old.items():
                if key != 'params':
                    group[key] = _clone_tree(value)
            old_parameter, parameter = old['params'][0], group['params'][0]
            previous = self.optimizer.state.get(old_parameter)
            if previous:
                copied = {}
                for key, value in previous.items():
                    if torch.is_tensor(value) and value.shape == old_parameter.shape and group['name'] in (
                            'xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation'):
                        target = torch.zeros_like(parameter)
                        valid = row_map >= 0
                        target[valid] = value[row_map[valid]]
                        copied[key] = target
                    else:
                        copied[key] = _clone_tree(value)
                result.optimizer.state[parameter] = copied

    @torch.no_grad()
    def zero_details_raw(self):
        raw = _clone_tree(self.reincarnation_phenotype())
        raw['opacity'][self.detail_rows] = _zero_opacity(raw['opacity'][self.detail_rows])
        return raw

    @torch.no_grad()
    def support_identity_raw(self):
        """Native parent plus zero details/N, preserving original six P fields."""
        raw = self.zero_details_raw()
        for name in RAW_FIELDS:
            raw[name][self.reincarnation_rows] = self.support_anchor_raw[name]
        raw['opacity'][:self.novel_count] = _zero_opacity(raw['opacity'][:self.novel_count])
        return raw

    @torch.no_grad()
    def freeze_predictive_endpoints(self, force=False):
        if self._predictive_endpoints is not None and not force:
            return self
        full = _clone_tree(self.reincarnation_phenotype())
        prior = _clone_tree(full)
        pdc, prest = self._prior_features()
        n = self.novel_count
        prior['sh_dc'][n:], prior['sh_rest'][n:] = pdc[n:], prest[n:]
        low, high = self._radiance_bounds()
        dc_only, sh_only = _clone_tree(prior), _clone_tree(prior)
        # Fixed-rest DC interval and fixed-DC SH ball keep the endpoints truly
        # DC-only/SH-only.  A joint cone projection could silently mix them.
        norm = torch.linalg.vector_norm(_sh1_vector(prior['sh_rest']), dim=-1)
        a = .5+C0*full['sh_dc'][..., 0, :]
        bounded_a = torch.maximum(low+norm, torch.minimum(a, high-norm))
        dc_only['sh_dc'][n:] = ((bounded_a-.5)/C0)[n:, None, :]
        center = .5+C0*prior['sh_dc'][..., 0, :]
        allowance = torch.minimum(center-low, high-center).clamp_min(0)
        current_norm = torch.linalg.vector_norm(_sh1_vector(full['sh_rest']), dim=-1)
        factor = (allowance/current_norm.clamp_min(torch.finfo(full['sh_dc'].dtype).tiny)).clamp_max(1)
        sh_only['sh_rest'][n:] = (full['sh_rest']*factor[:, None, :])[n:]
        self._predictive_endpoints = dict(prior=prior, dc=dc_only, sh=sh_only, full=full)
        self.predictive_weights = None
        return self

    def endpoint_raw(self, kind):
        aliases = {'dc_only':'dc', 'DC-only':'dc', 'sh_only':'sh', 'SH-only':'sh'}
        kind = aliases.get(kind, kind)
        if kind not in ('prior', 'dc', 'sh', 'full'):
            raise ValueError('Predictive endpoint must be prior, dc, sh or full')
        self.freeze_predictive_endpoints()
        return _clone_tree(self._predictive_endpoints[kind])

    @torch.no_grad()
    def apply_predictive_weights(self, weights):
        self.freeze_predictive_endpoints()
        w = torch.as_tensor(weights, dtype=torch.float64, device=self._xyz.device).flatten()
        if (w.shape != (3,) or not bool(torch.isfinite(w).all()) or
                bool((w < -1.e-10).any()) or float(w.sum()) > 1+1.e-10):
            raise ValueError('Three predictive weights must be nonnegative with sum <=1')
        w = w.clamp_min(0)
        if float(w.sum()) > 1:
            w = w/w.sum()
        p = self._predictive_endpoints['prior']
        for name, attr in (('sh_dc', '_features_dc'), ('sh_rest', '_features_rest')):
            value = p[name].clone()
            for weight, kind in zip(w.to(value), ('dc', 'sh', 'full')):
                value.add_(self._predictive_endpoints[kind][name]-p[name], alpha=float(weight))
            getattr(self, attr).copy_(value)
        self.p_common_dc.copy_(self._features_dc[self.reincarnation_rows])
        self.p_common_sh1.copy_(self._features_rest[self.reincarnation_rows])
        self.predictive_weights = w.detach().clone()
        self.project_parameters()
        return self

    def reincarnation_subset(self, selection):
        choice = torch.as_tensor(selection, device=self._xyz.device)
        ids = choice.nonzero().flatten() if choice.dtype == torch.bool else choice.long().flatten()
        # Let the reliable V9 implementation preserve root atomicity and Adam.
        # Row-shaped endpoint packets are sliced separately, after that remap.
        endpoints = self._predictive_endpoints
        self._predictive_endpoints = None
        try:
            result = super().reincarnation_subset(selection)
        finally:
            self._predictive_endpoints = endpoints
        if endpoints is not None:
            result._predictive_endpoints = {kind:{name:value[ids].detach().clone() for name,value in raw.items()}
                                            for kind,raw in endpoints.items()}
        result.predictive_weights = _clone_tree(self.predictive_weights)
        return result

    def capture(self):
        packet = super().capture()
        packet['predictive_reincarnation'] = packet.pop('reincarnation')
        return packet

    def restore(self, packet, training_args, device=None):
        packet = dict(packet)
        packet['reincarnation'] = packet['predictive_reincarnation']
        super().restore(packet, training_args, device)
        # The optimizer is loaded after install_reincarnation_state.
        return self.set_training_stage(self.predictive_phase)
