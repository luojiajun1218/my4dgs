"""SH1 support reincarnation and independent, bounded novel Gaussians.

P supports retain the original native geometry/alpha. A shared DC/SH1 material
may learn, and fine supports derive a symmetric equal-mass pair from the same
original anchor. N rows stay a free prefix for controller-owned cross fitting.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES, attribute_steps
from .conservative_family import ConservativeFamilyModel, _clone_tree


C0 = .28209479177387814
C1 = .4886025119029199


class _ExactForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameter, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def _ste(parameter, value):
    """Identity gradient with exactly the requested forward bits, even -0."""
    return _ExactForward.apply(parameter, value.detach())


def _sh1_vector(sh1):
    if sh1.shape[-2] == 0:
        return sh1.new_zeros((*sh1.shape[:-2], 3, 3))
    if sh1.shape[-2:] != (3, 3):
        raise ValueError('Only active SH0/SH1 materials are supported')
    # eval_sh: -C1*y*rest0 + C1*z*rest1 - C1*x*rest2.
    return C1*torch.stack((-sh1[..., 2, :], -sh1[..., 0, :], sh1[..., 1, :]), -1)


def sh1_sphere_range(dc, sh1, contrast=None):
    a = .5+C0*dc[..., 0, :]
    radius = torch.linalg.vector_norm(_sh1_vector(sh1), dim=-1)
    if contrast is not None:
        radius = radius+C0*contrast.abs()
    return a-radius, a+radius


def _simplex_magnitudes(values, radius):
    """Euclidean projection of nonnegative group magnitudes onto sum<=radius."""
    ordered = values.sort(-1, descending=True).values
    ranks = torch.arange(1, values.shape[-1]+1, device=values.device, dtype=values.dtype)
    excess = ordered.cumsum(-1)-radius[..., None]
    active = ordered-excess/ranks > 0
    count = active.sum(-1).clamp_min(1)
    threshold = excess.gather(-1, (count-1)[..., None])[..., 0]/count
    projected = (values-threshold[..., None]).clamp_min(0)
    return torch.where((values.sum(-1) <= radius)[..., None], values, projected)


def project_sh1_sphere(dc, sh1, low, high, contrast=None):
    """Project |a-mid|+||b||+C0|contrast|<=radius per RGB, in RGB units.

    This is the exact two/three-group nonnegative simplex projection. It bounds
    both fine children over the entire sphere, rather than only observed views.
    """
    if dc.shape[-2:] != (1, 3) or low.shape != dc.shape[:-2]+(3,) or high.shape != low.shape:
        raise ValueError('DC and full-sphere envelope shapes disagree')
    if sh1.shape[:-2] != dc.shape[:-2] or sh1.shape[-2] not in (0, 3) or sh1.shape[-1] != 3:
        raise ValueError('SH1 must have zero or three coefficients per RGB')
    midpoint, radius = (low+high)*.5, (high-low)*.5
    if not bool(torch.isfinite(low).all() & torch.isfinite(high).all() & (radius >= 0).all()):
        raise ValueError('Full-sphere radiance bounds must be finite and nonempty')
    centered = .5+C0*dc[..., 0, :]-midpoint
    vector = _sh1_vector(sh1)
    norm = torch.linalg.vector_norm(vector, dim=-1)
    detail = torch.zeros_like(centered) if contrast is None else C0*contrast
    magnitude = torch.stack((centered.abs(), norm, detail.abs()), -1)
    projected = _simplex_magnitudes(magnitude, radius)
    inside = magnitude.sum(-1) <= radius
    target_dc = ((midpoint+centered.sign()*projected[..., 0]-.5)/C0)[..., None, :]
    scale = projected[..., 1]/norm.clamp_min(torch.finfo(norm.dtype).tiny)
    target_vector = vector*scale[..., None]
    target_rest = torch.stack((-target_vector[..., 1], target_vector[..., 2], -target_vector[..., 0]), -2)/C1
    target_rest = target_rest[..., :sh1.shape[-2], :]
    target_detail = detail.sign()*projected[..., 2]/C0
    # Feasible inputs are exact identity, avoiding needless inverse-affine ULPs.
    target_dc = torch.where(inside[..., None, :], dc, target_dc)
    target_rest = torch.where(inside[..., None, :], sh1, target_rest)
    if contrast is not None:
        target_detail = torch.where(inside, contrast, target_detail)
    return target_dc, target_rest, target_detail


def _empty_raw(count, degree, device, dtype):
    return dict(xyz=torch.zeros(count, 3, device=device, dtype=dtype),
        rotation=torch.tensor([[1., 0., 0., 0.]], device=device, dtype=dtype).repeat(count, 1),
        logscale=torch.zeros(count, 3, device=device, dtype=dtype),
        opacity=torch.zeros(count, 1, device=device, dtype=dtype),
        sh_dc=torch.zeros(count, 1, 3, device=device, dtype=dtype),
        sh_rest=torch.zeros(count, 3 if degree else 0, 3, device=device, dtype=dtype))


class ReincarnationModel(ConservativeFamilyModel):
    """N prefix plus atomically selectable P groups with immutable native support."""
    def __init__(self, sh_degree):
        if sh_degree not in (0, 1):
            raise NotImplementedError('Support reincarnation currently supports active SH degree <=1 only')
        super().__init__(sh_degree)
        self.novel_count = 0
        self.reincarnation_groups = ()
        self.reincarnation_rows = torch.empty(0, dtype=torch.long)
        self.reincarnation_specs = ()
        self.support_anchor_raw = {}
        self.novel_initial_raw = {}
        self.p_common_dc = nn.Parameter(torch.empty(0, 1, 3))
        self.p_common_sh1 = nn.Parameter(torch.empty(0, 3 if sh_degree else 0, 3))
        self._family_roots = torch.empty(0, dtype=torch.long)
        self._reincarnation_cache = None
        self._reincarnation_terminal = False

    def _clear_reincarnation_cache(self, gradient=None):
        self._reincarnation_cache = None
        self._family_cache = None
        return gradient

    def _check_degree(self):
        if self.active_sh_degree > 1 or self._features_rest.shape[1] not in (0, 3):
            raise NotImplementedError('Full-sphere support reincarnation requires active SH0/SH1, without higher SH')

    @property
    def novel_mask(self):
        return torch.arange(len(self._xyz), device=self._xyz.device) < self.novel_count

    def install_reincarnations(self, specs, novel_raw=None, views=None):
        """N rows first, then exact native 1:1 supports. Views never define bounds."""
        specs = list(specs)
        if novel_raw is not None:
            template = novel_raw['xyz']
        elif specs:
            template = torch.as_tensor(specs[0]['raw']['xyz'])
        else:
            template = torch.empty(0, 3)
        device, dtype = template.device, template.dtype
        if not template.is_floating_point():
            dtype = torch.float32
        count = len(novel_raw['xyz']) if novel_raw is not None else 0
        novel = (_empty_raw(0, self.max_sh_degree, device, dtype) if novel_raw is None else
                 {name: torch.as_tensor(novel_raw[name], device=device, dtype=dtype).clone() for name in RAW_FIELDS})
        anchors = ({name: torch.stack([torch.as_tensor(spec['raw'][name], device=device, dtype=dtype)
                                      for spec in specs]) for name in RAW_FIELDS} if specs else
                   _empty_raw(0, self.max_sh_degree, device, dtype))
        rest_count = 3 if self.max_sh_degree else 0
        for raw, rows in ((novel, count), (anchors, len(specs))):
            expected = _empty_raw(rows, self.max_sh_degree, device, dtype)
            if any(raw[name].shape != expected[name].shape or not bool(torch.isfinite(raw[name]).all()) for name in RAW_FIELDS):
                raise ValueError('Native support/novel six raw fields have incompatible shapes or nonfinite values')
        old_rows = [int(spec['old_row']) for spec in specs]
        if len(set(old_rows)) != len(old_rows) or any(row < 0 for row in old_rows):
            raise ValueError('Each reincarnated support must own a distinct original old row')
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self, attr, nn.Parameter(torch.cat((novel[name], anchors[name]), 0)))
        self.source_rows = torch.tensor([0]*count+[int(spec.get('source_row', spec['old_row'])) for spec in specs], device=device, dtype=torch.long)
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = self._xyz.new_zeros(count+len(specs))
        self.xyz_gradient_accum = self._xyz.new_zeros(count+len(specs), 1)
        self.denom = self._xyz.new_zeros(count+len(specs), 1)
        empty_family = dict(rows=torch.empty(0, 3, device=device, dtype=torch.long),
            anchor_raw={name: value[:0] for name, value in anchors.items()},
            tangent_offset=self._xyz.new_empty(0, 2), gate_logit=self._xyz.new_empty(0, 1),
            contrast_dc=self._xyz.new_empty(0, 3))
        state = dict(novel_count=count, groups=[torch.tensor([count+index], device=device) for index in range(len(specs))],
            anchor_raw=anchors, common_dc=anchors['sh_dc'].clone(), common_sh1=anchors['sh_rest'].clone(),
            novel_initial_raw=novel, specs=[{key: value for key, value in spec.items() if key != 'raw'} for spec in specs],
            family=empty_family, family_roots=torch.empty(0, device=device, dtype=torch.long), terminal=False)
        self.install_reincarnation_state(state)
        return self

    def install_reincarnation_state(self, state):
        device, dtype, total = self._xyz.device, self._xyz.dtype, len(self._xyz)
        self._check_degree()
        self.novel_count = int(state['novel_count'])
        self.reincarnation_groups = tuple(torch.as_tensor(rows, device=device, dtype=torch.long).clone() for rows in state['groups'])
        roots = len(self.reincarnation_groups)
        flat = torch.cat(self.reincarnation_groups) if roots else torch.empty(0, device=device, dtype=torch.long)
        if (not 0 <= self.novel_count <= total or any(len(rows) not in (1, 3) for rows in self.reincarnation_groups) or
                not torch.equal(flat.sort().values, torch.arange(self.novel_count, total, device=device))):
            raise ValueError('Reincarnation groups must partition P rows after the independent novel prefix')
        self.reincarnation_rows = torch.tensor([int(rows[0]) for rows in self.reincarnation_groups], device=device, dtype=torch.long)
        self.support_anchor_raw = _clone_tree(state['anchor_raw'], device)
        expected = _empty_raw(roots, self.max_sh_degree, device, dtype)
        if any(self.support_anchor_raw[name].shape != expected[name].shape for name in RAW_FIELDS):
            raise ValueError('Immutable original supports must align with root groups')
        self.reincarnation_specs = tuple(_clone_tree(item, device) for item in state['specs'])
        self.novel_initial_raw = _clone_tree(state.get('novel_initial_raw',
            {name: getattr(self, attr)[:self.novel_count] for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}), device)
        self._family_roots = torch.as_tensor(state['family_roots'], device=device, dtype=torch.long).clone()
        self._point_root = torch.empty(total-self.novel_count, device=device, dtype=torch.long)
        self._point_sign = self._xyz.new_zeros(total-self.novel_count)
        for root, rows in enumerate(self.reincarnation_groups):
            self._point_root[rows-self.novel_count] = root
            if len(rows) == 3:
                self._point_sign[rows[1:]-self.novel_count] = self._xyz.new_tensor([1., -1.])
        self.point_kind = torch.zeros(total, device=device, dtype=torch.long)
        self.point_kind[self.novel_count:] = 1
        for rows in self.reincarnation_groups:
            if len(rows) == 3:
                self.point_kind[rows[1]], self.point_kind[rows[2]] = 2, 3
        for key, shape in (('common_dc', (roots, 1, 3)), ('common_sh1', (roots, 3 if self.max_sh_degree else 0, 3))):
            value = torch.as_tensor(state[key], device=device, dtype=dtype)
            if value.shape != shape:
                raise ValueError('Common reincarnation radiance shape disagrees with supports')
            parameter = nn.Parameter(value.detach().clone())
            parameter.register_hook(self._clear_reincarnation_cache)
            setattr(self, 'p_'+key, parameter)
        self.install_family_state(_clone_tree(state['family'], device))
        if len(self.family_rows) != len(self._family_roots):
            raise ValueError('Fine family rows and original root mapping disagree')
        for root, rows in zip(self._family_roots.tolist(), self.family_rows):
            if not torch.equal(rows, self.reincarnation_groups[root]):
                raise ValueError('Fine family must retain its complete original support group')
        self.family_contrast.requires_grad_(True)
        for parameter in (self.family_contrast, self.family_tangent_offset, self.family_gate_logit):
            flag = parameter.requires_grad
            parameter.requires_grad_(True)
            parameter.register_hook(self._clear_reincarnation_cache)
            parameter.requires_grad_(flag)
        self.set_training_stage(state.get('terminal', False))
        self._clear_reincarnation_cache()
        return self

    def reincarnation_state(self, device='cpu'):
        return _clone_tree(dict(novel_count=self.novel_count, groups=self.reincarnation_groups,
            anchor_raw=self.support_anchor_raw, common_dc=self.p_common_dc, common_sh1=self.p_common_sh1,
            novel_initial_raw=self.novel_initial_raw, specs=self.reincarnation_specs,
            family=self.family_state(device=self._xyz.device), family_roots=self._family_roots,
            terminal=self._reincarnation_terminal), device)

    def set_training_stage(self, terminal):
        self._reincarnation_terminal = bool(terminal)
        self.family_gate_logit.requires_grad_(bool(terminal))
        self.family_gate_logit.grad = None
        self.family_contrast.requires_grad_(True)
        self._clear_reincarnation_cache()
        return self

    def _support_radiance(self, quantized=False, config=None, predictor=None):
        common_dc, common_sh1 = self.p_common_dc, self.p_common_sh1
        contrast = common_dc.new_zeros(len(common_dc), 3).index_copy(0, self._family_roots, self.family_contrast)
        initial_low, initial_high = sh1_sphere_range(self.support_anchor_raw['sh_dc'], self.support_anchor_raw['sh_rest'])
        low, high = initial_low.clamp_max(0.), initial_high.clamp_min(1.)
        dc, rest, detail = common_dc, common_sh1, contrast
        if quantized:
            step = common_dc.new_tensor(config.new_sh_step)
            pdc, prest = predictor['sh_dc'][self.reincarnation_rows], predictor['sh_rest'][self.reincarnation_rows]
            dc = pdc+((common_dc-pdc)/step).round()*step
            rest = prest+((common_sh1-prest)/step).round()*step
            detail = (contrast/step).round()*step
        dc, rest, detail = project_sh1_sphere(dc, rest, low, high, detail)
        # Exact native initial support radiance, independent of source lattice.
        pristine = ((common_dc.detach() == self.support_anchor_raw['sh_dc']).all((1, 2)) &
                    (common_sh1.detach() == self.support_anchor_raw['sh_rest']).all((1, 2)) &
                    (contrast.detach() == 0).all(-1))
        dc = torch.where(pristine[:, None, None], self.support_anchor_raw['sh_dc'], dc)
        rest = torch.where(pristine[:, None, None], self.support_anchor_raw['sh_rest'], rest)
        dc, rest, detail = _ste(common_dc, dc), _ste(common_sh1, rest), _ste(contrast, detail)
        return dc, rest, detail

    def _phenotype(self, config=None, extent=None, reference=None, source_rows=None):
        self._check_degree()
        quantized = config is not None
        raw = dict(zip(RAW_FIELDS, (getattr(self, attr) for attr in RAW_ATTRIBUTES)))
        n = self.novel_count
        if quantized:
            sources = torch.as_tensor(source_rows, device=self._xyz.device, dtype=torch.long)
            if sources.shape != (len(self._xyz),) or bool(((sources < 0) | (sources >= len(reference['xyz']))).any()):
                raise ValueError('Reincarnation codec sources must address the actual reference')
            predictor = {name: value[sources].detach() for name, value in reference.items()}
            if len(self.family_rows) and not bool((sources[self.family_rows] == sources[self.family_rows[:, :1]]).all()):
                raise ValueError('A fine reincarnation family requires one shared surviving predictor')
            steps = attribute_steps(config, extent)
            result = {}
            for name, value in raw.items():
                value = value[:n]
                p = predictor[name][:n]
                if name == 'rotation':
                    if not getattr(config, 'rotation_updates', True):
                        result[name] = p
                        continue
                    value, p = F.normalize(value, dim=-1), F.normalize(p, dim=-1)
                    value = torch.where((value*p).sum(-1, keepdim=True) < 0, -value, value)
                result[name] = _ste(value, p+((value-p)/steps[name]).round()*steps[name])
        else:
            predictor = None
            result = {name: value[:n] for name, value in raw.items()}
        novel_dc, novel_rest, _ = project_sh1_sphere(result['sh_dc'], result['sh_rest'],
            self._xyz.new_zeros(n, 3), self._xyz.new_ones(n, 3))
        result['sh_dc'] = _ste(raw['sh_dc'][:n], novel_dc)
        result['sh_rest'] = _ste(raw['sh_rest'][:n], novel_rest)
        geometry = {name: value[self._point_root] for name, value in self.support_anchor_raw.items()}
        if len(self.family_rows):
            offset = self.family_tangent_offset
            if quantized:
                step = offset.new_tensor(config.new_xyz_step_extent*extent)
                offset = _ste(offset, (offset/step).round()*step)
            values = self._family_values(offset, torch.zeros_like(self.family_contrast))
            local_rows = self.family_rows.flatten()-n
            for name in ('xyz', 'rotation', 'logscale', 'opacity'):
                geometry[name] = geometry[name].index_copy(0, local_rows, values[name].flatten(0, 1))
        common_dc, common_rest, contrast = self._support_radiance(quantized, config, predictor)
        root = self._point_root
        signed = self._point_sign[:, None, None]
        shifted_dc = common_dc[root]+signed*contrast[root, None]
        geometry['sh_dc'] = torch.where(signed == 0, common_dc[root], shifted_dc)
        geometry['sh_rest'] = common_rest[root]
        return {name: torch.cat((result[name], geometry[name]), 0) for name in RAW_FIELDS}

    def reincarnation_phenotype(self):
        leaves = [getattr(self, attr) for attr in RAW_ATTRIBUTES]+[self.p_common_dc, self.p_common_sh1,
            self.family_tangent_offset, self.family_gate_logit, self.family_contrast]
        key = (torch.is_grad_enabled(), tuple((id(value), value._version, value.requires_grad) for value in leaves))
        if self._reincarnation_cache is not None and self._reincarnation_cache[0] == key:
            return self._reincarnation_cache[1]
        result = self._phenotype()
        for value in result.values():
            if value.requires_grad:
                value.register_hook(self._clear_reincarnation_cache)
        self._reincarnation_cache = key, result
        return result

    def birth_parameter_value(self, name):
        return self.reincarnation_phenotype()[name]

    def reincarnation_quantized_raw(self, config, extent, reference, source_rows):
        return self._phenotype(config, extent, reference, source_rows)

    @torch.no_grad()
    def project_parameters(self):
        n = self.novel_count
        dc, rest, _ = project_sh1_sphere(self._features_dc[:n], self._features_rest[:n],
            self._xyz.new_zeros(n, 3), self._xyz.new_ones(n, 3))
        self._features_dc[:n].copy_(dc); self._features_rest[:n].copy_(rest)
        contrast = self._xyz.new_zeros(len(self.reincarnation_groups), 3).index_copy(0, self._family_roots, self.family_contrast)
        dc, rest, detail = self._support_radiance()
        self.p_common_dc.copy_(dc); self.p_common_sh1.copy_(rest)
        self.family_contrast.copy_(detail[self._family_roots])
        self._clear_reincarnation_cache()

    project_ = project_parameters

    def install_families_from_coarse(self, root_indices, tangent_offsets=None, gate_logits=None):
        """Independent fine cohort; original geometry/prior and mature material.

        N is not split, and its current/immutable-initial states are kept apart.
        Controller may reset current N to that initial state for independent B.
        A new topology starts with no Adam state; capture/subset do preserve it.
        """
        chosen = torch.as_tensor(root_indices, device=self._xyz.device, dtype=torch.long).flatten()
        roots = len(self.reincarnation_groups)
        if len(chosen.unique()) != len(chosen) or bool(((chosen < 0) | (chosen >= roots)).any()):
            raise ValueError('Fine support root indices must be valid and unique')
        if any(len(self.reincarnation_groups[root]) != 1 for root in chosen.tolist()):
            raise ValueError('Only coarse 1:1 supports can start a new fine family')
        if tangent_offsets is not None:
            tangent_offsets = torch.as_tensor(tangent_offsets, device=self._xyz.device, dtype=self._xyz.dtype)
            if tangent_offsets.shape != (len(chosen), 2):
                raise ValueError('Fine tangent offsets must align with selected roots')
        if gate_logits is not None:
            gate_logits = torch.as_tensor(gate_logits, device=self._xyz.device, dtype=self._xyz.dtype)
            if gate_logits.shape != (len(chosen), 1):
                raise ValueError('Fine gate logits must align with selected roots')
        selected = {root: index for index, root in enumerate(chosen.tolist())}
        existing = {root: index for index, root in enumerate(self._family_roots.tolist())}
        groups, family_roots, family_rows, offsets, gates, details = [], [], [], [], [], []
        n, next_row = self.novel_count, self.novel_count
        for root in range(roots):
            fine = root in selected or root in existing
            rows = torch.arange(next_row, next_row+(3 if fine else 1), device=self._xyz.device)
            groups.append(rows); next_row += len(rows)
            if fine:
                family_roots.append(root); family_rows.append(rows)
                if root in existing:
                    index = existing[root]
                    offsets.append(self.family_tangent_offset[index].detach().clone())
                    gates.append(self.family_gate_logit[index].detach().clone())
                    details.append(self.family_contrast[index].detach().clone())
                else:
                    index = selected[root]
                    seed = self.reincarnation_specs[root].get('tangent_offset', self._xyz.new_zeros(2))
                    displacement = (torch.as_tensor(seed, device=self._xyz.device, dtype=self._xyz.dtype).clone()
                                    if tangent_offsets is None else tangent_offsets[index].clone())
                    parent_scale = self.support_anchor_raw['logscale'][root].exp()
                    axes = torch.arange(3, device=self._xyz.device)
                    tangent = axes[axes != parent_scale.argmin()]
                    radius_squared = (displacement/parent_scale[tangent]).square().sum()
                    if bool(radius_squared >= 1):
                        raise ValueError('Fine half-displacement must lie inside the original one-sigma tangent ellipse')
                    offsets.append(displacement/(1-radius_squared).sqrt())
                    gates.append(self._xyz.new_zeros(1) if gate_logits is None else gate_logits[index].clone())
                    details.append(self._xyz.new_zeros(3))
        result = type(self)(self.max_sh_degree)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            tails = [self.support_anchor_raw[name][root:root+1].expand(len(rows), *self.support_anchor_raw[name].shape[1:])
                     for root, rows in enumerate(groups)]
            value = torch.cat((getattr(self, attr)[:n].detach(), *tails), 0)
            setattr(result, attr, nn.Parameter(value.clone()))
        result.active_sh_degree, result.spatial_lr_scale = self.active_sh_degree, self.spatial_lr_scale
        source_tails = [self.source_rows[rows[0]].expand(len(new_rows))
                        for rows, new_rows in zip(self.reincarnation_groups, groups)]
        result.source_rows = torch.cat((self.source_rows[:n], *source_tails)).clone()
        result.max_radii2D = result._xyz.new_zeros(next_row)
        result.xyz_gradient_accum = result._xyz.new_zeros(next_row, 1)
        result.denom = result._xyz.new_zeros(next_row, 1)
        state = self.reincarnation_state(device=self._xyz.device)
        f = torch.tensor(family_roots, device=self._xyz.device, dtype=torch.long)
        state.update(groups=groups, family_roots=f, terminal=False,
            family=dict(rows=torch.stack(family_rows) if family_rows else f.new_empty((0, 3)),
                anchor_raw={name: value[f] for name, value in self.support_anchor_raw.items()},
                tangent_offset=torch.stack(offsets) if offsets else self._xyz.new_empty((0, 2)),
                gate_logit=torch.stack(gates) if gates else self._xyz.new_empty((0, 1)),
                contrast_dc=torch.stack(details) if details else self._xyz.new_empty((0, 3))))
        result.install_reincarnation_state(state)
        return result

    def training_setup(self, args):
        self._training_args = args
        self.percent_dense = args.percent_dense
        self.xyz_gradient_accum = self._xyz.new_zeros(len(self._xyz), 1)
        self.denom = self._xyz.new_zeros(len(self._xyz), 1)
        groups = ((self._xyz, args.position_lr_init*self.spatial_lr_scale, 'xyz'),
            (self._features_dc, args.feature_lr, 'f_dc'),
            (self._features_rest, args.feature_lr/20, 'f_rest'),
            (self._opacity, args.opacity_lr, 'opacity'), (self._scaling, args.scaling_lr, 'scaling'),
            (self._rotation, args.rotation_lr, 'rotation'),
            (self.p_common_dc, args.feature_lr, 'p_common_dc'),
            (self.p_common_sh1, args.feature_lr/20, 'p_common_sh1'),
            (self.family_contrast, args.feature_lr, 'family_contrast'),
            (self.family_tangent_offset, args.position_lr_init*self.spatial_lr_scale, 'family_tangent_offset'),
            (self.family_gate_logit, args.opacity_lr, 'family_gate_logit'))
        self.optimizer = torch.optim.Adam([dict(params=[parameter], lr=rate, name=name)
                                          for parameter, rate, name in groups], lr=0., eps=1.e-15)

    def capture(self):
        base = (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
            self._scaling, self._rotation, self._opacity, self.max_radii2D,
            self.xyz_gradient_accum, self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None, self.spatial_lr_scale)
        return _clone_tree(dict(base=base, reincarnation=self.reincarnation_state(), source_rows=self.source_rows), 'cpu')

    def restore(self, packet, training_args, device=None):
        base = packet['base']
        device = device or base[1].device
        self.active_sh_degree, self.spatial_lr_scale = base[0], base[11]
        for attr, index in (('_xyz', 1), ('_features_dc', 2), ('_features_rest', 3),
                            ('_scaling', 4), ('_rotation', 5), ('_opacity', 6)):
            setattr(self, attr, nn.Parameter(_clone_tree(base[index], device)))
        self.max_radii2D = _clone_tree(base[7], device)
        self.install_reincarnation_state(_clone_tree(packet['reincarnation'], device))
        self.training_setup(training_args)
        self.xyz_gradient_accum, self.denom = _clone_tree(base[8], device), _clone_tree(base[9], device)
        if base[10] is not None:
            self.optimizer.load_state_dict(_clone_tree(base[10], device))
        self.source_rows = _clone_tree(packet.get('source_rows'), device)
        return self

    def reincarnation_subset(self, selection):
        """N rows independently; P groups atomically, including shared Adam."""
        choice = torch.as_tensor(selection, device=self._xyz.device)
        if choice.dtype == torch.bool:
            if choice.shape != (len(self._xyz),):
                raise ValueError('Reincarnation subset mask must match model population')
            ids = choice.nonzero().flatten()
        else:
            ids = choice.long().flatten()
        if len(ids.unique()) != len(ids) or bool(((ids < 0) | (ids >= len(self._xyz))).any()):
            raise ValueError('Reincarnation subset rows must be valid and unique')
        novel = ids < self.novel_count
        new_n = int(novel.sum())
        if not bool(novel[:new_n].all()) or bool(novel[new_n:].any()):
            raise ValueError('Reincarnation selection must retain a novel prefix')
        inverse = torch.full((len(self._xyz),), -1, device=ids.device, dtype=torch.long)
        inverse[ids] = torch.arange(len(ids), device=ids.device)
        roots = []
        for root, rows in enumerate(self.reincarnation_groups):
            keep = inverse[rows] >= 0
            if bool(keep.any()) != bool(keep.all()):
                raise ValueError('A reincarnated support/family must be selected atomically')
            if bool(keep.all()): roots.append(root)
        roots = torch.tensor(roots, device=ids.device, dtype=torch.long)
        root_inverse = torch.full((len(self.reincarnation_groups),), -1, device=ids.device, dtype=torch.long)
        root_inverse[roots] = torch.arange(len(roots), device=ids.device)
        fine = (root_inverse[self._family_roots] >= 0).nonzero().flatten()
        result = type(self)(self.max_sh_degree)
        for attr in RAW_ATTRIBUTES:
            setattr(result, attr, nn.Parameter(_clone_tree(getattr(self, attr)[ids])))
        result.active_sh_degree, result.spatial_lr_scale = self.active_sh_degree, self.spatial_lr_scale
        result.source_rows = _clone_tree(self.source_rows[ids]) if self.source_rows is not None else None
        result.max_radii2D = _clone_tree(self.max_radii2D[ids])
        result.xyz_gradient_accum = (_clone_tree(self.xyz_gradient_accum[ids]) if len(self.xyz_gradient_accum) else self._xyz.new_zeros(len(ids), 1))
        result.denom = _clone_tree(self.denom[ids]) if len(self.denom) else self._xyz.new_zeros(len(ids), 1)
        state = self.reincarnation_state(device=ids.device)
        state.update(novel_count=new_n, groups=[inverse[self.reincarnation_groups[root]] for root in roots.tolist()],
            anchor_raw={name: value[roots] for name, value in state['anchor_raw'].items()},
            common_dc=state['common_dc'][roots], common_sh1=state['common_sh1'][roots],
            novel_initial_raw={name: value[ids[:new_n]] for name, value in self.novel_initial_raw.items()},
            specs=[state['specs'][root] for root in roots.tolist()], family_roots=root_inverse[self._family_roots[fine]])
        state['family'] = dict(rows=inverse[self.family_rows[fine]],
            anchor_raw={name: value[fine] for name, value in self.family_anchor_raw.items()},
            tangent_offset=self.family_tangent_offset[fine], gate_logit=self.family_gate_logit[fine],
            contrast_dc=self.family_contrast[fine])
        result.install_reincarnation_state(state)
        if self.optimizer is not None:
            result.training_setup(self._training_args)
            old_groups = {group['name']: group for group in self.optimizer.param_groups}
            for group in result.optimizer.param_groups:
                old = old_groups[group['name']]
                for key, value in old.items():
                    if key != 'params': group[key] = _clone_tree(value)
                old_parameter, parameter = old['params'][0], group['params'][0]
                state = self.optimizer.state.get(old_parameter)
                if state:
                    subset = (roots if group['name'].startswith('p_common_') else fine
                              if group['name'].startswith('family_') else ids)
                    result.optimizer.state[parameter] = {key:_clone_tree(value[subset] if torch.is_tensor(value) and value.shape == old_parameter.shape else value)
                                                         for key, value in state.items()}
        return result

    def prune_points(self, remove):
        remove = torch.as_tensor(remove, device=self._xyz.device)
        mask = remove if remove.dtype == torch.bool else torch.zeros(len(self._xyz), device=self._xyz.device, dtype=torch.bool)
        if remove.dtype != torch.bool: mask[remove.long()] = True
        subset = self.reincarnation_subset(~mask)
        self.__dict__.clear(); self.__dict__.update(subset.__dict__)
