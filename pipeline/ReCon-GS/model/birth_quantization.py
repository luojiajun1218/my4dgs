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
    return {name: getattr(model, attr).detach().clone() for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}


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
        self.predictor['rotation'] = F.normalize(self.predictor['rotation'], dim=-1)
        self.symbols = {}
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            value = getattr(model, attr)
            if name == 'rotation':
                value = F.normalize(value, dim=-1)
                value = torch.where((value * self.predictor[name]).sum(-1, keepdim=True) < 0, -value, value)
            self.symbols[name] = quantized_symbols(value - self.predictor[name], self.steps[name])

    def raw_value(self, name):
        return self.predictor[name] + self.symbols[name] * self.steps[name]

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
        return torch.sigmoid(self.raw_value('opacity'))

    @property
    def get_features(self):
        return torch.cat((self.raw_value('sh_dc'), self.raw_value('sh_rest')), dim=1)

    def get_covariance(self, scaling_modifier=1):
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
        if name == 'rotation':
            birth = F.normalize(birth, dim=-1)
        if birth.shape[1:] != raw[name].shape[1:] or len(birth) != len(sources):
            raise ValueError(f'Retained birth {name} shape disagrees with base')
        raw[name] = torch.cat((raw[name], birth.to(raw[name])), dim=0)
    return raw, sources
