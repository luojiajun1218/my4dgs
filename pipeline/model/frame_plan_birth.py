"""Two independent birth cohorts, with absolute training carriers and delta coding.

Each cohort owns Adam parameters/state and a birth-age learning-rate clock.
Codec references are detached surviving old points; initialization lineage never
constrains reference selection or protects a parent from deletion.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES
from .early_replacement_model import _logits, _Exact
from .conservative_family import _clone_tree


class FramePlanBirthModel:
    independent_birth = True
    probability_alpha = True
    early_quantized_raw = None

    def __init__(self, sh_degree=1, device='cpu'):
        self.max_sh_degree = self.active_sh_degree = sh_degree
        self.spatial_lr_scale = 1.
        self.device = torch.device(device)
        self.cohorts = []
        self.source_rows = torch.empty(0, dtype=torch.long, device=device)
        self.parent_rows = self.source_rows.clone()  # initialization lineage only
        self.max_radii2D = torch.empty(0, device=device)
        self.optimizer = None

    def _cat(self, attr):
        if self.cohorts:
            return torch.cat([batch['parameters'][attr] for batch in self.cohorts])
        shape = dict(_xyz=(3,), _rotation=(4,), _scaling=(3,), _opacity=(1,),
                     _features_dc=(1, 3), _features_rest=((self.max_sh_degree+1)**2-1, 3))[attr]
        return torch.empty((0, *shape), device=self.device)

    def __getattr__(self, name):
        if name in RAW_ATTRIBUTES:
            return self._cat(name)
        raise AttributeError(name)

    @staticmethod
    def covariance_activation(scale, modifier, rotation):
        from .birth_quantization import physical_covariance
        return physical_covariance(scale, rotation, modifier)

    def birth_parameter_values(self):
        opacity=[]
        for batch in self.cohorts:
            alpha=batch['parameters']['_opacity'].clamp(0,1)
            logits=_logits(alpha)
            # sigmoid(quantized logit) may round to one in float32. Preserve
            # its finite coded logit while the probability carrier is pristine;
            # converting one back to +100 would silently reinitialize alpha.
            pristine=alpha.detach().eq(batch['alpha_reference'])
            opacity.append(_Exact.apply(logits,torch.where(pristine,batch['opacity_reference'],logits)))
        raw_opacity=torch.cat(opacity) if opacity else self._opacity
        return dict(xyz=self._xyz, rotation=F.normalize(self._rotation, dim=-1),
                    logscale=self._scaling, opacity=raw_opacity,
                    sh_dc=self._features_dc, sh_rest=self._features_rest)

    def add(self, raw, parents, references, step, optimizer, args):
        if not len(parents):
            return
        count=len(parents)
        shapes=dict(xyz=(count,3),rotation=(count,4),logscale=(count,3),opacity=(count,1),
                    sh_dc=(count,1,3),sh_rest=(count,(self.max_sh_degree+1)**2-1,3))
        if (len(references)!=count or any(raw[name].shape!=shape or not bool(torch.isfinite(raw[name]).all())
                                         for name,shape in shapes.items())):
            raise ValueError('Frame-plan births require aligned finite raw attributes and references')
        self.optimizer = optimizer
        cohort = dict(birth_step=step, parameters={attr: nn.Parameter(
            (raw[name].sigmoid() if name == 'opacity' else raw[name]).detach().clone())
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)})
        cohort['opacity_reference']=raw['opacity'].detach().clone()
        cohort['alpha_reference']=raw['opacity'].sigmoid().detach().clone()
        self.cohorts.append(cohort)
        self.parent_rows = torch.cat((self.parent_rows, parents))
        self.source_rows = torch.cat((self.source_rows, references))
        self.max_radii2D = torch.cat((self.max_radii2D, raw['xyz'].new_zeros(len(parents))))
        self._attach_cohort(len(self.cohorts)-1, optimizer, args)

    def _attach_cohort(self, index, optimizer, args):
        batch = self.cohorts[index]
        for attr, param in batch['parameters'].items():
            optimizer.add_param_group(dict(params=[param], name=f'early_plan_{index}_{attr}',
                lr=1., betas=(.9, .999), eps=1e-15))

    def set_rates(self, step, args, ratio):
        rates = dict(_xyz=args.position_lr_init*self.spatial_lr_scale,
                     _rotation=args.rotation_lr, _scaling=args.scaling_lr,
                     _opacity=args.opacity_lr, _features_dc=args.feature_lr,
                     _features_rest=args.feature_lr/20)
        for index, batch in enumerate(self.cohorts):
            start = batch['birth_step']
            age = max(0, step-start-1)
            factor = ratio**min(1., age/max(1, 300-start-1))
            for group in self.optimizer.param_groups:
                for attr in rates:
                    if group.get('name') == f'early_plan_{index}_{attr}':
                        group['lr'] = rates[attr]*factor

    @torch.no_grad()
    def select_rows_inplace(self, selected, optimizer=None):
        optimizer = optimizer or self.optimizer
        offset = 0
        for index, batch in enumerate(self.cohorts):
            count = len(batch['parameters']['_xyz'])
            local = selected[(selected >= offset) & (selected < offset+count)]-offset
            for attr, previous in list(batch['parameters'].items()):
                parameter = nn.Parameter(previous.detach()[local].clone())
                state = optimizer.state.pop(previous, {})
                if state:
                    optimizer.state[parameter] = {key: value[local].clone()
                        if torch.is_tensor(value) and value.shape == previous.shape else value
                        for key, value in state.items()}
                for group in optimizer.param_groups:
                    if group.get('name') == f'early_plan_{index}_{attr}':
                        group['params'] = [parameter]
                batch['parameters'][attr] = parameter
            batch['opacity_reference']=batch['opacity_reference'][local]
            batch['alpha_reference']=batch['alpha_reference'][local]
            offset += count
        self.source_rows = self.source_rows[selected]
        self.parent_rows = self.parent_rows[selected]
        self.max_radii2D = self.max_radii2D[selected]

    @torch.no_grad()
    def set_actual(self, raw):
        offset = 0
        for batch in self.cohorts:
            count = len(batch['parameters']['_xyz'])
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                value = raw[name][offset:offset+count]
                batch['parameters'][attr].copy_(value.sigmoid() if name == 'opacity' else value)
            batch['opacity_reference']=raw['opacity'][offset:offset+count].detach().clone()
            batch['alpha_reference']=raw['opacity'][offset:offset+count].sigmoid().detach().clone()
            offset += count

    @torch.no_grad()
    def project_parameters(self):
        for batch in self.cohorts:
            batch['parameters']['_opacity'].clamp_(0, 1)

    def capture(self):
        groups = {}
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                if group.get('name', '').startswith('early_plan_'):
                    param = group['params'][0]
                    groups[group['name']] = dict(hyper={k: v for k, v in group.items() if k != 'params'},
                                                state=self.optimizer.state.get(param, {}))
        return dict(frame_plan_birth=_clone_tree(dict(cohorts=self.cohorts, source_rows=self.source_rows,
            parent_rows=self.parent_rows, groups=groups, spatial_lr_scale=self.spatial_lr_scale), 'cpu'))

    def restore(self, packet, optimizer, args, device):
        state = _clone_tree(packet['frame_plan_birth'], device)
        self.device = torch.device(device)
        self.optimizer = optimizer
        self.spatial_lr_scale = state['spatial_lr_scale']
        self.source_rows, self.parent_rows = state['source_rows'], state['parent_rows']
        self.cohorts = state['cohorts']
        for index, batch in enumerate(self.cohorts):
            batch['parameters'] = {attr: nn.Parameter(value) for attr, value in batch['parameters'].items()}
            self._attach_cohort(index, optimizer, args)
        for group in optimizer.param_groups:
            saved = state['groups'].get(group.get('name'))
            if saved is not None:
                group.update(saved['hyper'])
                param = group['params'][0]
                optimizer.state[param] = {k: v.cpu() if k == 'step' and torch.is_tensor(v) else v
                                          for k, v in saved['state'].items()}
        self.max_radii2D = self._xyz.new_zeros(len(self._xyz))
        return self


@torch.no_grad()
def nearest_old_references(raw, reference, extent, position_weight=.5, colour_weight=.5):
    """Exhaustive deterministic nearest match; ties choose lowest old row.

    Squared world distance is normalized by scene extent squared; squared
    RGB DC distance by three channels (RGB range 0..1). Equal weights.
    References always address surviving old GS, so cycles are impossible.
    """
    if not len(reference['xyz']):
        raise ValueError('Frame plan cannot encode/recover a scene with zero surviving old anchors')
    if not len(raw['xyz']):
        return torch.empty(0, dtype=torch.long, device=reference['xyz'].device)
    xyz = reference['xyz'].double()/max(float(extent), 1e-12)
    colour = reference['sh_dc'][:, 0].double()*.28209479177387814+.5
    result = []
    # Bound the candidate matrix without relying on nondeterministic top-k.
    for begin in range(0, len(raw['xyz']), 32):
        points = raw['xyz'][begin:begin+32].double()/max(float(extent), 1e-12)
        rgb = raw['sh_dc'][begin:begin+32, 0].double()*.28209479177387814+.5
        distance = position_weight*((points[:, None]-xyz[None])**2).sum(-1)
        distance += colour_weight*((rgb[:, None]-colour[None])**2).mean(-1)
        result.append(distance.argmin(-1))
    return torch.cat(result)
