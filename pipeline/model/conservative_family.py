"""A retained coarse surface kernel plus a symmetric signed detail pair.

The conserved quantity is a thin-surface optical-depth integral proxy, not
per-camera alpha compositing or post-clamp rendered brightness. World poses are
derived from one tangent latent; independently quantizing child xyz/q would
break the shared-plane constraint and is deliberately avoided here.
"""
import math
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from libgs.utils.general import get_expon_lr_func
from .gaussian import BirthGaussianModel
from .birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES


_CAP = .99
_CAP_LOG = -math.log1p(-_CAP)
_NODES = 4096
_TABLES = {}


def _li2_scalar(a):
    if a > .5:
        return math.pi**2/6-math.log(a)*math.log1p(-a)-_li2_scalar(1-a)
    power, result = a, 0.
    for k in range(1, 49):
        result += power/(k*k)
        power *= a
    return result


_F_CAP = _li2_scalar(_CAP)
OPTICAL_MASS_MAX = _F_CAP+_CAP_LOG*math.log(1/_CAP)


@lru_cache(maxsize=1)
def _cpu_tables():
    alpha = [-math.expm1(-_CAP_LOG*i/_NODES) for i in range(_NODES+1)]
    alpha[0], alpha[-1] = 0., _CAP
    value = [_li2_scalar(a) for a in alpha]
    derivative = [(-math.log1p(-a)/a if a else 1.) for a in alpha]
    return tuple(torch.tensor(x, dtype=torch.float64) for x in (alpha, value, derivative))


def _tables(value):
    key = (value.device, value.dtype)
    if key not in _TABLES:
        _TABLES[key] = tuple(x.to(device=value.device, dtype=value.dtype) for x in _cpu_tables())
    return _TABLES[key]


def _hermite(x, x0, x1, y0, y1, slope0, slope1):
    h = x1-x0
    t = (x-x0)/h
    # Cubic Hermite interpolation, with exact endpoint slopes.
    return y0+t*(h*slope0+t*(3*(y1-y0)-h*(2*slope0+slope1)+
                            t*(2*(y0-y1)+h*(slope0+slope1))))


def _mass_derivative(alpha):
    clipped = alpha.clamp(0, _CAP)
    return torch.where(clipped > 0,
        -torch.log1p(-clipped)/alpha.clamp_min(torch.finfo(alpha.dtype).tiny),
        torch.ones_like(alpha))


class _OpticalMass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha):
        a = alpha.clamp(0, 1)
        grid, values, slopes = _tables(a)
        pos = (-torch.log1p(-a.clamp_max(_CAP))/_CAP_LOG*_NODES).long().clamp(0, _NODES-1)
        y = _hermite(a.clamp_max(_CAP), grid[pos], grid[pos+1],
                     values[pos], values[pos+1], slopes[pos], slopes[pos+1])
        high = _F_CAP+_CAP_LOG*torch.log(a.clamp_min(_CAP)/_CAP)
        result = torch.where(a > _CAP, high, y)
        ctx.save_for_backward(a, alpha)
        return result

    @staticmethod
    def backward(ctx, grad):
        a, original = ctx.saved_tensors
        derivative = _mass_derivative(a)
        return grad*derivative*((original >= 0) & (original <= 1))


class _InverseOpticalMass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, target):
        y = target.clamp(0, OPTICAL_MASS_MAX)
        grid, values, slopes = _tables(y)
        pos = (torch.searchsorted(values, y.contiguous())-1).clamp(0, _NODES-1)
        alpha = _hermite(y.clamp_max(_F_CAP), values[pos], values[pos+1],
                         grid[pos], grid[pos+1], 1/slopes[pos], 1/slopes[pos+1])
        high = _CAP*torch.exp((y-_F_CAP)/_CAP_LOG)
        alpha = torch.where(y > _F_CAP, high, alpha).clamp(0, 1)
        alpha = torch.where(y <= 0, torch.zeros_like(alpha), alpha)
        alpha = torch.where(y >= OPTICAL_MASS_MAX, torch.ones_like(alpha), alpha)
        ctx.save_for_backward(alpha, target)
        return alpha

    @staticmethod
    def backward(ctx, grad):
        alpha, target = ctx.saved_tensors
        return grad/_mass_derivative(alpha)*((target >= 0) & (target <= OPTICAL_MASS_MAX))


def optical_depth_integral(alpha):
    """Li2(alpha), continued above native alpha cap .99; differentiable lookup.

    A cached nonuniform Hermite table avoids iterative solves or long series in
    each render. Gradients use the analytic derivative, not lookup-bin slopes.
    """
    return _OpticalMass.apply(alpha)


def inverse_optical_depth_integral(mass):
    """Monotone inverse on [0,F(1)], with analytic implicit backward."""
    return _InverseOpticalMass.apply(mass)


def _clone_tree(value, device=None):
    if torch.is_tensor(value):
        return value.detach().to(device=device or value.device).clone()
    if isinstance(value, dict):
        return {k: _clone_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(v, device) for v in value)
    if isinstance(value, list):
        return [_clone_tree(v, device) for v in value]
    return value


def _quat_product(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def _rotation(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def _canonical_logit(alpha):
    # Finite wire values also represent exact lifecycle endpoints. Their tiny
    # alpha error is deliberate; no renderer-only hidden alpha state exists.
    tiny = max(math.exp(-100), torch.finfo(alpha.dtype).tiny)
    bounded = alpha.clamp(tiny, 1-torch.finfo(alpha.dtype).eps)
    value = bounded.log()-torch.log1p(-bounded)
    return torch.where(alpha <= 0, alpha.new_full((), -100),
                       torch.where(alpha >= 1, alpha.new_full((), 100), value))


def _ste_quantize(value, step):
    return value+(torch.round(value/step)*step-value).detach()


class ConservativeFamilyModel(BirthGaussianModel):
    """Raw free GS rows plus atomically selectable mother/plus/minus families."""
    def __init__(self, sh_degree):
        super().__init__(sh_degree)
        self.family_rows = torch.empty((0, 3), dtype=torch.long)
        self.family_anchor_raw = {}
        self.family_tangent_offset = nn.Parameter(torch.empty((0, 2)))
        self.family_gate_logit = nn.Parameter(torch.empty((0, 1)))
        self.family_contrast = nn.Parameter(torch.empty((0, 3)), requires_grad=False)
        self._family_cache = None
        self._training_args = None

    def _clear_family_cache(self, gradient=None):
        self._family_cache = None
        return gradient

    def install_families(self, rows, anchor_raw, tangent_offset=None,
                         gate_logit=None, contrast_dc=None):
        device, dtype = self._xyz.device, self._xyz.dtype
        rows = torch.as_tensor(rows, device=device, dtype=torch.long).reshape(-1, 3).clone()
        count = len(rows)
        if rows.numel() and (bool((rows < 0).any()) or bool((rows >= len(self._xyz)).any()) or
                            len(rows.flatten().unique()) != rows.numel()):
            raise ValueError('Family rows must be distinct valid mother/plus/minus rows')
        anchors = {}
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            supplied = anchor_raw.get(name, getattr(self, attr)[:0]) if count == 0 else anchor_raw[name]
            value = torch.as_tensor(supplied, device=device, dtype=dtype)
            if value.shape != (count, *getattr(self, attr).shape[1:]):
                raise ValueError(f'Family anchor {name} has the wrong shape')
            anchors[name] = value.detach().clone()
        self.family_rows, self.family_anchor_raw = rows, anchors
        def latent(value, shape, train=True):
            value = torch.zeros(shape, device=device, dtype=dtype) if value is None else torch.as_tensor(value, device=device, dtype=dtype)
            if value.shape != shape:
                raise ValueError('Family latent shape disagrees with family count')
            parameter = nn.Parameter(value.detach().clone(), requires_grad=train)
            if train:
                parameter.register_hook(self._clear_family_cache)
            return parameter
        self.family_tangent_offset = latent(tangent_offset, (count, 2))
        self.family_gate_logit = latent(gate_logit, (count, 1))
        self.family_contrast = latent(contrast_dc, (count, 3), False)
        if count:
            sigma = anchors['logscale'].exp()
            normal = sigma.argmin(-1)
            axes = torch.arange(3, device=device).expand(count, 3)
            tangent = axes[axes != normal[:, None]].reshape(count, 2)
            self._family_normal_axis, self._family_tangent_axes = normal, tangent
            self._family_frame = _rotation(anchors['rotation'])
            self._family_tangent_sigma = sigma.gather(1, tangent)
            self._family_parent_area = self._family_tangent_sigma.prod(-1)
            self._family_parent_integral = optical_depth_integral(anchors['opacity'].sigmoid().flatten()).detach()
        self._clear_family_cache()
        return self

    def install_family_state(self, state):
        self.install_families(state['rows'], state['anchor_raw'],
            state.get('tangent_offset'), state.get('gate_logit'), state.get('contrast_dc'))
        trainable = state.get('trainable', {})
        for attr, key in (('family_tangent_offset', 'tangent_offset'),
                          ('family_gate_logit', 'gate_logit'),
                          ('family_contrast', 'contrast_dc')):
            if key in trainable:
                getattr(self, attr).requires_grad_(trainable[key])
        return self

    def family_state(self, device='cpu'):
        return _clone_tree(dict(rows=self.family_rows, anchor_raw=self.family_anchor_raw,
            tangent_offset=self.family_tangent_offset, gate_logit=self.family_gate_logit,
            contrast_dc=self.family_contrast,
            trainable=dict(tangent_offset=self.family_tangent_offset.requires_grad,
                gate_logit=self.family_gate_logit.requires_grad,
                contrast_dc=self.family_contrast.requires_grad)), device)

    def _family_values(self, offset=None, contrast=None):
        anchor = self.family_anchor_raw
        offset = self.family_tangent_offset if offset is None else offset
        contrast = self.family_contrast if contrast is None else contrast
        sigma = self._family_tangent_sigma
        v = offset/sigma
        denominator = 1+v.square().sum(-1)
        local_d = offset/denominator.sqrt()[:, None]
        tangent_frame = self._family_frame.gather(2, self._family_tangent_axes[:, None].expand(-1, 3, -1))
        d = (tangent_frame@local_d[..., None])[..., 0]
        # Stable tangent covariance and eigenvalues. det/large-eigenvalue avoids
        # catastrophic cancellation for an almost rank-one tangent ellipse.
        a = sigma[:, 0].square()*(1+v[:, 1].square())/denominator
        c = sigma[:, 1].square()*(1+v[:, 0].square())/denominator
        b = -sigma.prod(-1)*v[:, 0]*v[:, 1]/denominator
        discriminant = (a-c).square()+4*b.square()
        degenerate = discriminant == 0
        root = torch.where(degenerate, torch.zeros_like(discriminant),
            torch.where(degenerate, torch.ones_like(discriminant), discriminant).sqrt())
        high = (a+c+root)*.5
        determinant = sigma.prod(-1).square()/denominator
        low = determinant/high.clamp_min(torch.finfo(high.dtype).tiny)
        angle = .5*torch.atan2(torch.where(degenerate, torch.zeros_like(b), 2*b),
                               torch.where(degenerate, torch.ones_like(a), a-c))
        axes = F.one_hot(self._family_tangent_axes, 3).to(offset)
        rotation_axis = torch.cross(axes[:, 0], axes[:, 1], dim=-1)
        local_q = torch.cat(((angle*.5).cos()[:, None],
                             rotation_axis*(angle*.5).sin()[:, None]), -1)
        child_q = _quat_product(F.normalize(anchor['rotation'], dim=-1), local_q)
        child_logscale = anchor['logscale'].scatter(1, self._family_tangent_axes,
                                                   torch.stack((high, low), -1).sqrt().log())
        child_area = self._family_parent_area/denominator.sqrt()
        target_mass = self._family_parent_area*self._family_parent_integral
        capacity = 2*child_area*OPTICAL_MASS_MAX/target_mass.clamp_min(torch.finfo(offset.dtype).tiny)
        gate = torch.minimum(self.family_gate_logit.sigmoid().flatten(), capacity)
        mother_mass = (1-gate)*self._family_parent_integral
        child_mass = gate*target_mass/(2*child_area).clamp_min(torch.finfo(offset.dtype).tiny)
        # One vectorized inverse call for both mother and equal child alpha.
        alphas = inverse_optical_depth_integral(torch.stack((mother_mass, child_mass), -1))
        logits = _canonical_logit(alphas)
        dc = anchor['sh_dc']
        return dict(xyz=torch.stack((anchor['xyz'], anchor['xyz']+d, anchor['xyz']-d), 1),
            rotation=torch.stack((anchor['rotation'], child_q, child_q), 1),
            logscale=torch.stack((anchor['logscale'], child_logscale, child_logscale), 1),
            opacity=torch.stack((logits[:, 0], logits[:, 1], logits[:, 1]), 1)[..., None],
            sh_dc=torch.stack((dc, dc+contrast[:, None], dc-contrast[:, None]), 1),
            sh_rest=anchor['sh_rest'][:, None].expand(-1, 3, -1, -1))

    def family_phenotype(self):
        leaves = [getattr(self, attr) for attr in RAW_ATTRIBUTES]+[
            self.family_tangent_offset, self.family_gate_logit, self.family_contrast]
        key = (torch.is_grad_enabled(), tuple((id(x), x._version, x.requires_grad) for x in leaves))
        if self._family_cache is not None and self._family_cache[0] == key:
            return self._family_cache[1]
        result = dict(zip(RAW_FIELDS, (getattr(self, attr) for attr in RAW_ATTRIBUTES)))
        if len(self.family_rows):
            values = self._family_values()
            ids = self.family_rows.flatten()
            result = {name:value.index_copy(0, ids, values[name].flatten(0, 1))
                      for name,value in result.items()}
            for value in result.values():
                if value.requires_grad:
                    value.register_hook(self._clear_family_cache)
        self._family_cache = key, result
        return result

    def birth_parameter_value(self, name):
        return self.family_phenotype()[name]

    def family_quantized_raw(self, config, extent, reference, source_rows):
        """Source-coded scalar fields; latent-coded tangent poses and contrast.

        Caller must use these family rows directly, without another independent
        xyz/quaternion rounding. Ordinary free rows are untouched by this hook.
        """
        result = dict(self.family_phenotype())
        if not len(self.family_rows):
            return result
        sources = torch.as_tensor(source_rows, device=self._xyz.device, dtype=torch.long)[self.family_rows]
        if not bool((sources == sources[:, :1]).all()):
            raise ValueError('A family requires one shared surviving source predictor')
        offset = _ste_quantize(self.family_tangent_offset, config.new_xyz_step_extent*extent)
        contrast = _ste_quantize(self.family_contrast, config.new_sh_step)
        values = self._family_values(offset, contrast)
        predictor = {name:reference[name][sources[:, 0]].detach() for name in RAW_FIELDS}
        for name, step in (('logscale', config.new_logscale_step), ('opacity', config.new_opacity_step)):
            p = predictor[name][:, None]
            values[name] = p+_ste_quantize(values[name]-p, step)
        dc_base = predictor['sh_dc']+_ste_quantize(self.family_anchor_raw['sh_dc']-predictor['sh_dc'], config.new_sh_step)
        values['sh_dc'] = torch.stack((dc_base, dc_base+contrast[:, None], dc_base-contrast[:, None]), 1)
        rest = predictor['sh_rest']+_ste_quantize(self.family_anchor_raw['sh_rest']-predictor['sh_rest'], config.new_sh_step)
        values['sh_rest'] = rest[:, None].expand(-1, 3, -1, -1)
        ids = self.family_rows.flatten()
        return {name:value.index_copy(0, ids, values[name].flatten(0, 1)) for name,value in result.items()}

    @property
    def get_xyz(self): return self.birth_parameter_value('xyz')
    @property
    def get_rotation(self): return F.normalize(self.birth_parameter_value('rotation'), dim=-1)
    @property
    def get_scaling(self): return self.birth_parameter_value('logscale').exp()
    @property
    def get_opacity(self): return self.birth_parameter_value('opacity').sigmoid()
    @property
    def get_features(self): return torch.cat((self.birth_parameter_value('sh_dc'), self.birth_parameter_value('sh_rest')), 1)

    def get_covariance(self, scaling_modifier=1.):
        # The base implementation uses _rotation directly; family poses are
        # derived, so both the CUDA and precomputed-covariance paths need this.
        basis = _rotation(self.get_rotation)
        matrix = basis@torch.diag_embed((self.get_scaling*scaling_modifier).square())@basis.transpose(-1, -2)
        return torch.stack((matrix[:,0,0],matrix[:,0,1],matrix[:,0,2],
                            matrix[:,1,1],matrix[:,1,2],matrix[:,2,2]), -1)

    def training_setup(self, args):
        self._training_args = args
        self.percent_dense = args.percent_dense
        self.xyz_gradient_accum = self._xyz.new_zeros((len(self._xyz), 1))
        self.denom = self._xyz.new_zeros((len(self._xyz), 1))
        names = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')
        rates = (args.position_lr_init*self.spatial_lr_scale, args.feature_lr,
                 args.feature_lr/20, args.opacity_lr, args.scaling_lr, args.rotation_lr)
        # RAW_ATTRIBUTES order differs from the usual six optimizer group order.
        mapping = dict(xyz=self._xyz, f_dc=self._features_dc, f_rest=self._features_rest,
                       scaling=self._scaling, rotation=self._rotation, opacity=self._opacity)
        groups = [dict(params=[mapping[name]], lr=rate, name=name) for name,rate in zip(names, rates)]
        if len(self.family_rows):
            groups.extend((dict(params=[self.family_tangent_offset], lr=rates[0], name='family_tangent_offset'),
                           dict(params=[self.family_gate_logit], lr=args.opacity_lr, name='family_gate_logit')))
        self.optimizer = torch.optim.Adam(groups, lr=0., eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=rates[0],
            lr_final=args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult, max_steps=args.position_lr_max_steps)

    def capture(self):
        base = (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
            self._scaling, self._rotation, self._opacity, self.max_radii2D,
            self.xyz_gradient_accum, self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None, self.spatial_lr_scale)
        return dict(base=_clone_tree(base, 'cpu'), family=self.family_state(),
                    source_rows=_clone_tree(self.source_rows, 'cpu'))

    def restore(self, packet, training_args, device=None):
        family = packet.get('family') if isinstance(packet, dict) else None
        base = packet['base'] if isinstance(packet, dict) else packet
        device = device or base[1].device
        self.active_sh_degree, self.spatial_lr_scale = base[0], base[11]
        for attr,index in (('_xyz', 1), ('_features_dc', 2), ('_features_rest', 3),
                           ('_scaling', 4), ('_rotation', 5), ('_opacity', 6)):
            setattr(self, attr, nn.Parameter(base[index].detach().to(device).clone()))
        self.max_radii2D = base[7].detach().to(device).clone()
        if family is None:
            anchors = {name:getattr(self, attr)[:0] for name,attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}
            self.install_families(torch.empty((0, 3), dtype=torch.long, device=device), anchors)
        else:
            self.install_family_state(family)
        self.training_setup(training_args)
        self.xyz_gradient_accum, self.denom = [base[i].detach().to(device).clone() for i in (8, 9)]
        if base[10] is not None:
            self.optimizer.load_state_dict(_clone_tree(base[10], device))
        self.source_rows = (_clone_tree(packet.get('source_rows'), device) if isinstance(packet, dict) else None)
        self._clear_family_cache()
        return self

    def family_subset(self, selection):
        """Independent subset factory; every family is retained or removed whole."""
        choice = torch.as_tensor(selection, device=self._xyz.device)
        if choice.dtype == torch.bool:
            if choice.shape != (len(self._xyz),):
                raise ValueError('Family subset mask must match model rows')
            ids = choice.nonzero().flatten()
        else:
            ids = choice.long().flatten()
        if len(ids.unique()) != len(ids) or bool(((ids < 0) | (ids >= len(self._xyz))).any()):
            raise ValueError('Subset rows must be valid and unique')
        inverse = torch.full((len(self._xyz),), -1, device=ids.device, dtype=torch.long)
        inverse[ids] = torch.arange(len(ids), device=ids.device)
        mapped = inverse[self.family_rows]
        selected = mapped >= 0
        if bool((selected.any(-1) != selected.all(-1)).any()):
            raise ValueError('A conservative family must be selected atomically')
        families = selected.all(-1).nonzero().flatten()
        result = type(self)(self.max_sh_degree)
        for attr in RAW_ATTRIBUTES:
            setattr(result, attr, nn.Parameter(getattr(self, attr).detach()[ids].clone()))
        result.active_sh_degree, result.spatial_lr_scale = self.active_sh_degree, self.spatial_lr_scale
        result.max_radii2D = self.max_radii2D[ids].clone() if len(self.max_radii2D) else self._xyz.new_zeros(len(ids))
        result.source_rows = self.source_rows[ids].clone() if self.source_rows is not None else None
        state = self.family_state(device=self._xyz.device)
        state['rows'] = mapped[families]
        state['anchor_raw'] = {k:v[families] for k,v in state['anchor_raw'].items()}
        for name in ('tangent_offset', 'gate_logit', 'contrast_dc'):
            state[name] = state[name][families]
        result.install_family_state(state)
        if self.optimizer is not None:
            result.training_setup(self._training_args)
            old_groups = {g['name']:g for g in self.optimizer.param_groups}
            for group in result.optimizer.param_groups:
                old = old_groups[group['name']]
                for key,value in old.items():
                    if key != 'params': group[key] = value
                oldp, newp = old['params'][0], group['params'][0]
                state = self.optimizer.state.get(oldp)
                if state:
                    rows = families if group['name'].startswith('family_') else ids
                    result.optimizer.state[newp] = {k:(_clone_tree(v[rows]) if torch.is_tensor(v) and v.shape == oldp.shape else _clone_tree(v)) for k,v in state.items()}
        return result

    def prune_points(self, remove):
        mask = torch.zeros(len(self._xyz), device=self._xyz.device, dtype=torch.bool)
        if remove.dtype == torch.bool: mask = remove
        else: mask[remove] = True
        subset = self.family_subset(~mask)
        self.__dict__.clear()
        self.__dict__.update(subset.__dict__)


def install_conservative_families(model, state):
    """Install family traits on an already populated raw newborn model."""
    if not isinstance(model, ConservativeFamilyModel):
        model.__class__ = ConservativeFamilyModel
        model.family_rows = torch.empty((0, 3), dtype=torch.long, device=model._xyz.device)
        model.family_anchor_raw = {}
        model._family_cache = None
        model._training_args = None
    return model.install_family_state(state)
