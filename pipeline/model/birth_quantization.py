"""Raw Gaussian fields and quantized newborn attributes for the active rigid tree."""

import hashlib

import torch
from torch.nn import functional as F

RAW_FIELDS = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')
RAW_ATTRIBUTES = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest')

def quantized_symbols(value, step):
    continuous = value / torch.as_tensor(step, device=value.device, dtype=value.dtype)
    return continuous + (continuous.round() - continuous).detach()


def model_raw(model):
    bulk_hook = getattr(model, 'birth_parameter_values', None)
    if bulk_hook is not None:
        values = bulk_hook()
        return {name: values[name].detach().clone() for name in RAW_FIELDS}
    hook = getattr(model, 'birth_parameter_value', None)
    return {name: (hook(name) if hook is not None else getattr(model, attr)).detach().clone()
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}


def reference_digest(raw):
    digest = hashlib.sha256()
    for name in RAW_FIELDS:
        tensor = raw[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def attribute_steps(config, extent):
    return dict(xyz=config.new_xyz_step_extent * extent, rotation=config.new_rotation_step,
                logscale=config.new_logscale_step, opacity=config.new_opacity_step,
                sh_dc=config.new_sh_step, sh_rest=config.new_sh_step)


def physical_covariance(scale,rotation,modifier=1.):
    w,x,y,z=F.normalize(rotation,dim=-1).unbind(-1)
    matrix=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
    basis=matrix*(scale*modifier)[:,None,:]
    covariance=basis@basis.transpose(1,2)
    return covariance[:,(0,0,0,1,1,2),(0,1,2,1,2,2)]


class LiveReplacementView:
    """Replace native rows in place; keep camera-depth tie/evidence ordering."""
    def __init__(self,old,newborn):
        self.old,self.newborn=old,newborn
        self.rows=newborn.parent_rows
    def __getattr__(self,name):return getattr(self.old,name)
    def _replace(self,value,new):return value.index_copy(0,self.rows,new)
    @property
    def get_xyz(self):return self._replace(self.old.get_xyz,self.newborn.get_xyz)
    @property
    def get_rotation(self):return self._replace(self.old.get_rotation,self.newborn.get_rotation)
    @property
    def get_scaling(self):return self._replace(self.old.get_scaling,self.newborn.get_scaling)
    @property
    def get_opacity(self):return self._replace(self.old.get_opacity,self.newborn.get_opacity)
    @property
    def get_features(self):return self._replace(self.old.get_features,self.newborn.get_features)
    def get_covariance(self,scaling_modifier=1.):
        return physical_covariance(self.get_scaling,self.get_rotation,scaling_modifier)


class QuantizedNewGaussianView:
    """Quantized attributes for candidates that survived periodic pruning."""

    def __init__(self, model, config, extent, reference, source_rows):
        self.model, self.config = model, config
        self.steps = attribute_steps(config, extent)
        if not len(reference['xyz']):
            raise ValueError('New Gaussian prediction needs a nonempty motion-adjusted reference')
        self.source = torch.as_tensor(source_rows, device=model._xyz.device,
                                      dtype=torch.long)
        if (self.source.shape != (len(model._xyz),) or
                bool(((self.source < 0) | (self.source >= len(reference['xyz']))).any())):
            raise ValueError('Birth source rows must match the actual clone/split lineage')
        self.predictor = {name: value[self.source].detach() for name, value in reference.items()}
        self.rotation_updates = getattr(config, 'rotation_updates', True)
        if self.rotation_updates:
            self.predictor['rotation'] = F.normalize(self.predictor['rotation'], dim=-1)
        self.symbols = {}
        bulk_hook = getattr(model, 'birth_parameter_values', None)
        parameter_values = bulk_hook() if bulk_hook is not None else None
        parameter_value = getattr(model, 'birth_parameter_value', None)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            value = (parameter_values[name] if parameter_values is not None else
                     parameter_value(name) if parameter_value is not None else getattr(model, attr))
            if name == 'rotation':
                if not self.rotation_updates:
                    self.symbols[name] = torch.zeros_like(value, requires_grad=False)
                    continue
                value = F.normalize(value, dim=-1)
                value = torch.where((value * self.predictor[name]).sum(-1, keepdim=True) < 0, -value, value)
            self.symbols[name] = quantized_symbols(value - self.predictor[name], self.steps[name])
        self.family_rows = getattr(model, 'family_rows', self.source.new_empty((0, 3))).flatten()
        self.family_raw = None
        self.patch_raw = None
        self.reincarnation_raw = None
        self.persistent_raw = None
        self.early_raw = None
        early_hook=getattr(model,'early_quantized_raw',None)
        if early_hook is not None:
            self.early_raw=early_hook(config,extent,reference,self.source)
            for name in ('logscale','opacity','sh_dc','sh_rest'):
                self.symbols[name]=quantized_symbols(self.early_raw[name]-self.predictor[name],self.steps[name])
        persistent_hook = getattr(model,'persistent_quantized_raw',None)
        if persistent_hook is not None:
            self.persistent_raw = persistent_hook(config,extent,reference,self.source)
            for name in ('logscale','opacity','sh_dc','sh_rest'):
                self.symbols[name]=quantized_symbols(self.persistent_raw[name]-self.predictor[name],self.steps[name])
        reincarnation_hook = getattr(model, 'reincarnation_quantized_raw', None)
        if reincarnation_hook is not None:
            self.reincarnation_raw = reincarnation_hook(config, extent, reference, self.source)
            for name in ('logscale', 'opacity', 'sh_dc', 'sh_rest'):
                self.symbols[name] = quantized_symbols(self.reincarnation_raw[name]-self.predictor[name], self.steps[name])
        patch_hook = getattr(model, 'patch_quantized_raw', None)
        if patch_hook is not None:
            self.patch_raw = patch_hook(config, extent, reference, self.source)
            for name in ('logscale', 'opacity', 'sh_dc', 'sh_rest'):
                self.symbols[name] = quantized_symbols(self.patch_raw[name]-self.predictor[name], self.steps[name])
        if len(self.family_rows) and self.reincarnation_raw is None:
            self.family_raw = model.family_quantized_raw(config, extent, reference, self.source)
            # The flat appearance codec still receives ordinary integer
            # symbols. Pose comes directly from quantized local family latents,
            # never independent world xyz/quaternion rounding.
            for name in ('logscale', 'opacity', 'sh_dc', 'sh_rest'):
                value = quantized_symbols(self.family_raw[name]-self.predictor[name], self.steps[name])
                self.symbols[name] = self.symbols[name].index_copy(0, self.family_rows, value[self.family_rows])
            rows = model.family_rows
            base = quantized_symbols(model.family_anchor_raw['sh_dc']-self.predictor['sh_dc'][rows[:, 0]],
                                     self.steps['sh_dc'])
            contrast = quantized_symbols(model.family_contrast, self.steps['sh_dc'])[:, None]
            paired = torch.stack((base, base+contrast, base-contrast), 1).flatten(0, 1)
            self.symbols['sh_dc'] = self.symbols['sh_dc'].index_copy(0, self.family_rows, paired)

    def raw_value(self, name):
        if self.early_raw is not None:
            value=self.early_raw[name]
        elif self.persistent_raw is not None:
            value = self.persistent_raw[name]
        elif self.reincarnation_raw is not None:
            value = self.reincarnation_raw[name]
        elif self.patch_raw is not None:
            value = self.patch_raw[name]
        elif name == 'rotation' and not self.rotation_updates:
            value = self.predictor[name]
        else:
            value = self.predictor[name] + self.symbols[name] * self.steps[name]
            if self.family_raw is not None and name in ('xyz', 'rotation'):
                value = value.index_copy(0, self.family_rows, self.family_raw[name][self.family_rows])
        # Cross-fitting excludes every geometric AND appearance gradient from
        # the complementary views. The rendered forward scene is unchanged.
        scale = getattr(self, 'row_gradient_scale', None)
        if scale is not None:
            value = value.detach()+(value-value.detach())*scale.reshape((-1,)+(1,)*(value.ndim-1))
        locked = getattr(self, 'locked_geometry_rows', None)
        if locked is not None and name in ('xyz', 'rotation', 'logscale'):
            value = torch.where(locked.reshape((-1,)+(1,)*(value.ndim-1)), value.detach(), value)
        return value


    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def get_xyz(self):
        return self.raw_value('xyz')

    @property
    def get_rotation(self):
        return F.normalize(self.raw_value('rotation'), dim=-1)

    @property
    def get_scaling(self):
        return torch.exp(self.raw_value('logscale'))

    @property
    def get_opacity(self):
        if self.early_raw is not None:
            return self.model.get_opacity
        alpha = torch.sigmoid(self.raw_value('opacity'))
        if getattr(self.model, 'probability_alpha', False):
            carrier = self.model._opacity.clamp(0, 1)
            return alpha.detach()+(carrier-carrier.detach())
        return alpha

    @property
    def get_features(self):
        return torch.cat((self.raw_value('sh_dc'), self.raw_value('sh_rest')), dim=1)

    def get_covariance(self, scaling_modifier=1):
        if self.early_raw is not None:return self.model.get_covariance(scaling_modifier)
        return self.model.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)


def patch_old_state(module, model=None, transaction=None):
    """Exact OPD4 scene plus straight-through carrier gradients, without commit."""
    from .sparse_updates import canonical_alpha_logits
    model = module.births if model is None else model
    transaction = module.birth_transaction if transaction is None else transaction
    update = module.opacity_update
    baseline_symbols = transaction.old_base_symbols
    baseline_logits = transaction.old_base_opacity
    if baseline_symbols is None or baseline_logits is None:
        raise ValueError('Patch takeover lost its immutable motion-stage opacity baseline')
    desired = model.raw_old_alpha(baseline_logits)
    owned = model.owned_old_rows
    if transaction.fixed_keep is not None:
        owned = owned[~transaction.provisional_deleted[owned]]
    symbols = baseline_symbols.detach().clone()
    if len(owned):
        ref = update.alpha_reference[owned].detach()
        target = desired[owned]
        step = ref.new_tensor(update.step)
        ratio = (target.detach().double()-ref.double())/step.double()
        selected = ratio.round()
        selected = torch.where(target.detach() <= 0, -torch.ceil(ref.double()/step.double()), selected)
        selected = torch.where(target.detach() >= 1, torch.ceil((1-ref.double())/step.double()), selected)
        symbols[owned] = selected.to(symbols)
    pinned = getattr(transaction, 'fixed_old_symbols', None)
    if transaction.fixed_keep is not None and pinned is not None:
        # Fixed selected population: a retired carrier cannot reappear during
        # final refit, though all original data remain available for rollback.
        symbols[transaction.provisional_deleted] = pinned[transaction.provisional_deleted]
    logits = canonical_alpha_logits(update.reference, symbols, update.step).to(baseline_logits)
    if len(owned):
        zero = desired[owned].detach() <= 0
        one = desired[owned].detach() >= 1
        fix_zero = zero & (logits[owned] > -100)
        fix_one = one & (logits[owned] < 100)
        if bool(fix_zero.any() or fix_one.any()):
            symbols[owned] += fix_one.to(symbols)-fix_zero.to(symbols)
            logits = canonical_alpha_logits(update.reference, symbols, update.step).to(baseline_logits)
    alpha = logits.sigmoid()
    if torch.is_grad_enabled() and len(owned):
        live = alpha[owned]+(desired[owned]-desired[owned].detach())
        alpha = alpha.index_copy(0, owned, live)
    return logits, alpha, symbols


def materialize_quantized_gaussians(base_raw, new_view=None):
    """Append quantized births to already moved/deleted old raw tensors."""
    raw = {name: base_raw[name].detach().clone() for name in RAW_FIELDS}
    base_count = len(raw['xyz'])
    if any(len(value) != base_count for value in raw.values()):
        raise ValueError('Materialized base attributes disagree on Gaussian count')
    if new_view is None:
        return raw, torch.empty(0, dtype=torch.long, device=raw['xyz'].device)
    if (new_view.source.ndim != 1 or
            new_view.source.dtype not in (torch.int32, torch.int64) or
            len(new_view.source) != len(new_view.model._xyz)):
        raise ValueError('Quantized birth sources must be aligned')
    sources = new_view.source.detach().to(
        device=raw['xyz'].device, dtype=torch.long).clone()
    if len(sources) and bool(((sources < 0) | (sources >= base_count)).any()):
        raise ValueError('Retained birth source index is out of bounds')
    for name in RAW_FIELDS:
        birth = new_view.raw_value(name).detach()
        if (name == 'rotation' and getattr(new_view, 'rotation_updates', True) and
                getattr(new_view,'persistent_raw',None) is None and getattr(new_view,'early_raw',None) is None):
            birth = F.normalize(birth, dim=-1)
        if birth.shape[1:] != raw[name].shape[1:] or len(birth) != len(sources):
            raise ValueError(f'Retained birth {name} shape disagrees with base')
        raw[name] = torch.cat((raw[name], birth.to(raw[name])), dim=0)
    return raw, sources


@torch.no_grad()
def family_quantization_diagnostics(model, view):
    rows = getattr(model, 'family_rows', None)
    if rows is None or not len(rows):
        return dict(families=0)
    from .conservative_family import optical_depth_integral
    from .rigid_residual_tree import quaternion_rotate
    anchor = model.family_anchor_raw
    q = F.normalize(anchor['rotation'], dim=-1)
    identity = torch.eye(3, device=q.device, dtype=q.dtype)
    frame = quaternion_rotate(q[:, None].expand(-1, 3, -1), identity[None].expand(len(q), -1, -1)).transpose(1, 2)
    order = anchor['logscale'].argsort(-1)
    tangent = frame.gather(2, order[:, None, 1:].expand(-1, 3, -1))
    normal = frame.gather(2, order[:, None, :1].expand(-1, 3, -1))[..., 0]
    scale = anchor['logscale'].exp()
    pose = view.raw_value('xyz')[rows]
    displacement = pose[:, 1]-pose[:, 0]
    local = (frame.transpose(1, 2)@displacement[..., None])[..., 0]
    rho = (local/scale).square().sum(-1).sqrt()
    family_q = F.normalize(view.raw_value('rotation')[rows], dim=-1)
    basis = quaternion_rotate(family_q[..., None, :].expand(-1, -1, 3, -1),
        identity[None, None].expand(len(rows), 3, -1, -1)).transpose(-1, -2)
    sigma = view.raw_value('logscale')[rows].exp()
    projection = tangent[:, None].transpose(-1, -2)@(basis*sigma[..., None, :])
    cov2 = projection@projection.transpose(-1, -2)
    area = torch.linalg.det(cov2).clamp_min(0).sqrt()
    target = scale.gather(1, order[:, 1:]).prod(-1)*optical_depth_integral(anchor['opacity'].sigmoid().flatten())
    actual = (area*optical_depth_integral(view.raw_value('opacity')[rows].sigmoid()[..., 0])).sum(-1)
    ratio = actual/target.clamp_min(torch.finfo(target.dtype).tiny)
    dc = view.symbols['sh_dc'][rows]
    cancel = (dc[:, 1]-dc[:, 0])+(dc[:, 2]-dc[:, 0])
    normal_component = (displacement*normal).sum(-1).abs()
    return dict(families=len(rows), mass_ratio_mean=float(ratio.mean()),
        mass_ratio_min=float(ratio.min()), mass_ratio_max=float(ratio.max()),
        max_tangent_mahalanobis=float(rho.max()), max_normal_offset=float(normal_component.max()),
        contrast_symbol_cancellation_max=float(cancel.abs().max()),
        shared_source=bool((view.source[rows] == view.source[rows[:, :1]]).all()),
        semantics='native-capped optical profile integral times covariance area in the fixed parent tangent plane; scalar codec grid/float-pose error, not occluded pixel coverage')
