"""Raw Gaussian fields and quantized newborn attributes for the active rigid tree."""

import hashlib

import torch
from torch.nn import functional as F

RAW_FIELDS = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')
RAW_ATTRIBUTES = ('_xyz', '_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest')

def quantized_symbols(value, step):
    continuous = value / torch.as_tensor(step, device=value.device, dtype=value.dtype)
    return continuous + (continuous.round() - continuous).detach()


@torch.no_grad()
def reanchor_reconstruction_tolerance(value, predictor, step):
    """Half a quantization bin plus bounded floating-point roundoff.

    Reanchoring subtracts a reference, divides/rounds, multiplies, then adds
    that reference back. At large world coordinates a float32 ULP can exceed
    half the spatial bin. Account for those four arithmetic roundings without
    changing the symbols, codec, or training forward operation.
    """
    if (not value.is_floating_point() or value.dtype != predictor.dtype or
            value.shape != predictor.shape or value.device != predictor.device):
        raise ValueError('Reanchor values and predictors must be aligned floating tensors')
    unit = torch.as_tensor(step, device=value.device, dtype=value.dtype)
    if (unit.numel() != 1 or not bool(torch.isfinite(unit).all()) or
            not bool(unit.gt(0).all()) or
            not bool(torch.isfinite(value).all() & torch.isfinite(predictor).all())):
        raise ValueError('Reanchor values, predictors and positive step must be finite')
    difference = value.detach() - predictor.detach()
    scaled = difference / unit
    offset = scaled.round() * unit
    reconstructed = predictor.detach() + offset
    if not all(bool(torch.isfinite(x).all()) for x in
               (difference, scaled, offset, reconstructed)):
        raise ValueError('Reanchor arithmetic overflowed')

    def spacing(x):
        upper = torch.nextafter(x, torch.full_like(x, torch.inf))
        lower = torch.nextafter(x, torch.full_like(x, -torch.inf))
        return torch.maximum((upper.double()-x.double()).abs(),
                             (x.double()-lower.double()).abs())

    roundoff = .5 * (spacing(difference) + spacing(scaled)*unit.double() +
                     spacing(offset) + spacing(reconstructed))
    if not bool(torch.isfinite(roundoff).all()):
        raise ValueError('Reanchor arithmetic has no finite rounding bound')
    # Retain the former guard's small margin; only the coordinate-dependent
    # IEEE rounding allowance is new. The caller still checks every element.
    return unit.double()*.501 + 1e-6 + roundoff


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
        self.early_raw = None
        early_hook=getattr(model,'early_quantized_raw',None)
        if early_hook is not None:
            self.early_raw=early_hook(config,extent,reference,self.source)
            for name in ('logscale','opacity','sh_dc','sh_rest'):
                self.symbols[name]=quantized_symbols(self.early_raw[name]-self.predictor[name],self.steps[name])
    def raw_value(self, name):
        if self.early_raw is not None:
            value=self.early_raw[name]
        elif name == 'rotation' and not self.rotation_updates:
            value = self.predictor[name]
        else:
            value = self.predictor[name] + self.symbols[name] * self.steps[name]
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
                getattr(new_view,'early_raw',None) is None):
            birth = F.normalize(birth, dim=-1)
        if birth.shape[1:] != raw[name].shape[1:] or len(birth) != len(sources):
            raise ValueError(f'Retained birth {name} shape disagrees with base')
        raw[name] = torch.cat((raw[name], birth.to(raw[name])), dim=0)
    return raw, sources
