"""Fixed evidence planes with trainable tangent kernels and old coverage.

Only UV position, tangent scale and in-plane angle can move. Material priors
belong to the measured surface and are independent of entropy-code predictors.
The quantized phenotype is still six ordinary GS fields, so existing exact
appearance/float-pose fallbacks can decode it without hidden patch state.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS, RAW_ATTRIBUTES
from .gaussian import BirthGaussianModel


C0 = .28209479177387814


def _clone(value, device=None):
    if torch.is_tensor(value):
        return value.detach().to(device=device or value.device).clone()
    if isinstance(value, dict):
        return {key: _clone(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item, device) for item in value)
    return value


def _frame_quaternion(frame):
    """Scalar-first quaternion of fixed right-handed frame columns."""
    if not len(frame):
        return frame.new_empty((0, 4))
    r = frame
    roots = torch.stack((1+r[:, 0, 0]+r[:, 1, 1]+r[:, 2, 2],
                         1+r[:, 0, 0]-r[:, 1, 1]-r[:, 2, 2],
                         1-r[:, 0, 0]+r[:, 1, 1]-r[:, 2, 2],
                         1-r[:, 0, 0]-r[:, 1, 1]+r[:, 2, 2]), -1).clamp_min(0).sqrt()
    candidates = torch.stack((
        torch.stack((roots[:, 0]**2, r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1]), -1),
        torch.stack((r[:, 2, 1]-r[:, 1, 2], roots[:, 1]**2, r[:, 1, 0]+r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0]), -1),
        torch.stack((r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]+r[:, 0, 1], roots[:, 2]**2, r[:, 2, 1]+r[:, 1, 2]), -1),
        torch.stack((r[:, 1, 0]-r[:, 0, 1], r[:, 2, 0]+r[:, 0, 2], r[:, 2, 1]+r[:, 1, 2], roots[:, 3]**2), -1)), 1)
    index = roots.argmax(-1)
    return F.normalize(candidates[torch.arange(len(frame), device=frame.device), index], dim=-1)


def _quat_product(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)), -1)


def _covariance(scales, modifier, quaternion):
    """Device-independent covariance of the supplied derived pose."""
    w, x, y, z = F.normalize(quaternion, dim=-1).unbind(-1)
    basis = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    matrix = basis@torch.diag_embed((scales*modifier).square())@basis.transpose(-1, -2)
    return torch.stack((matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2],
                        matrix[:, 1, 1], matrix[:, 1, 2], matrix[:, 2, 2]), -1)


def _ste_value(value, forward):
    return value+(forward-value).detach()


def _bounded_grid(value, lower, upper, step):
    """A local lattice centered inside the support always has a feasible zero."""
    step = torch.as_tensor(step, device=value.device, dtype=value.dtype)
    origin = (lower+upper)*.5
    symbols = ((value-origin)/step).round()
    low, high = torch.ceil((lower-origin).double()/step.double()), torch.floor((upper-origin).double()/step.double())
    symbols = symbols.double().maximum(low).minimum(high).to(value)
    forward = (origin+symbols*step).maximum(lower).minimum(upper)
    return _ste_value(value, forward)


def _bounded_source_dc(value, predictor, lower, upper, step):
    """Keep decoded source-grid DC physically bounded; exact fallback if needed."""
    step = torch.as_tensor(step, device=value.device, dtype=value.dtype)
    low = torch.ceil((lower.double()-predictor.double())/step.double())
    high = torch.floor((upper.double()-predictor.double())/step.double())
    symbol = ((value-predictor)/step).round().double().maximum(low).minimum(high).to(value)
    forward = predictor+symbol*step
    # A narrow trusted HDR/SH box may contain no predictor lattice point.
    # Preserve the physical box; existing RTB1 exactATTR then handles the wire.
    forward = torch.where(low <= high, forward, value)
    forward = forward.maximum(lower).minimum(upper)
    return _ste_value(value, forward)


class SurfacePatchModel(BirthGaussianModel):
    """Every newborn row belongs to one atomic, fixed-plane surface patch."""
    def __init__(self, sh_degree):
        super().__init__(sh_degree)
        self.covariance_activation = _covariance
        self.patch_rows = ()
        self.patch_old_rows = ()
        self.patch_metadata = ()
        self.patch_center = torch.empty(0, 3)
        self.patch_frame = torch.empty(0, 3, 3)
        self.patch_uv_bounds = torch.empty(0, 2, 2)
        self.patch_scale_bounds = torch.empty(0, 2, 2)
        self.patch_thickness = torch.empty(0, 1)
        self.patch_prior_dc = torch.empty(0, 1, 3)
        self.patch_prior_sh_rest = torch.empty(0, (sh_degree+1)**2-1, 3)
        self.patch_uv = nn.Parameter(torch.empty(0, 2))
        self.patch_logscale2d = nn.Parameter(torch.empty(0, 2))
        self.patch_angle = nn.Parameter(torch.empty(0, 1))
        self.patch_retention = nn.Parameter(torch.empty(0, 1))
        self.patch_retention_cap = 1.
        self._radiance_centers = torch.empty(0, 3)
        self._patch_cache = None
        self._training_args = None

    @property
    def owned_old_rows(self):
        return (torch.cat(self.patch_old_rows) if len(self.patch_old_rows) else
                torch.empty(0, dtype=torch.long, device=self._xyz.device))

    def _clear_patch_cache(self, gradient=None):
        self._patch_cache = None
        return gradient

    def install_patches(self, specs, old_count=None, views=None):
        """Build a model from C's evidence specs; no source predictor sets priors.

        Required keys: center, frame (columns t0,t1,n), uv_bounds (lower,upper),
        thickness, kernel_uv, kernel_scale, and kernel_color or kernel_dc.
        Optional: kernel_sh_rest, prior_dc/prior_sh_rest, kernel_alpha/angle,
        scale_bounds, old_rows, source_rows, retention (default exactly one).
        """
        specs = list(specs)
        if specs:
            first = torch.as_tensor(specs[0]['kernel_uv'])
            device, dtype = first.device, first.dtype
            if not first.is_floating_point():
                dtype = torch.float32
        else:
            device, dtype = self._xyz.device, self._xyz.dtype
        def tensor(value):
            return torch.as_tensor(value, device=device, dtype=dtype)
        fields = {key: [] for key in ('center', 'frame', 'uv_bounds', 'scale_bounds', 'thickness',
                                      'uv', 'logscale2d', 'angle', 'retention', 'prior_dc', 'prior_sh_rest')}
        dc, rest, opacity, sources, rows, old_rows, metadata = [], [], [], [], [], [], []
        offset, rest_count = 0, (self.max_sh_degree+1)**2-1
        for spec in specs:
            uv = tensor(spec['kernel_uv'])
            count = len(uv)
            if uv.shape != (count, 2) or count == 0:
                raise ValueError('Every surface patch needs nonempty kernel_uv[K,2]')
            center, frame = tensor(spec['center']), tensor(spec['frame'])
            bounds, thickness = tensor(spec['uv_bounds']), tensor(spec['thickness']).reshape(1)
            scales = tensor(spec['kernel_scale'])
            scale_bounds = tensor(spec.get('scale_bounds', torch.stack((thickness.expand(2), bounds[1]-bounds[0]))))
            # An explicit mature DC wins on refinement; for new evidence use
            # its angular-aware material prior before a bare RGB conversion.
            color = spec.get('kernel_dc', spec.get('prior_dc'))
            if color is None:
                rgb = tensor(spec['kernel_color'])
                if rgb.shape != (count, 3) or not bool(((rgb >= 0) & (rgb <= 1)).all()):
                    raise ValueError('Evidence kernel_color must be LDR RGB; use explicit trusted kernel_dc for HDR')
                color = ((rgb-.5)/C0)[:, None]
            color = tensor(color).reshape(count, 1, 3)
            sh_rest = tensor(spec.get('kernel_sh_rest', spec.get('sh_rest', torch.zeros(count, rest_count, 3))))
            if sh_rest.shape == (rest_count, 3):
                sh_rest = sh_rest[None].expand(count, -1, -1)
            prior_dc = tensor(spec.get('prior_dc', color)).reshape(count, 1, 3)
            prior_rest = tensor(spec.get('prior_sh_rest', sh_rest))
            if prior_rest.shape == (rest_count, 3):
                prior_rest = prior_rest[None].expand(count, -1, -1)
            alpha = tensor(spec.get('kernel_alpha', .1)).reshape(-1, 1).expand(count, 1)
            angle = tensor(spec.get('kernel_angle', 0.)).reshape(-1, 1).expand(count, 1)
            retention = tensor(spec.get('retention', 1.)).reshape(1)
            if (center.shape != (3,) or frame.shape != (3, 3) or bounds.shape != (2, 2) or
                    scales.shape != (count, 2) or scale_bounds.shape != (2, 2) or
                    sh_rest.shape != (count, rest_count, 3) or prior_rest.shape != sh_rest.shape):
                raise ValueError('Surface patch spec has inconsistent geometry or SH shapes')
            if (not bool((bounds[0] < bounds[1]).all()) or not bool((thickness > 0).all()) or
                    not bool((scale_bounds[0] > 0).all() & (scale_bounds[0] <= scale_bounds[1]).all()) or
                    not bool(((uv >= bounds[0]) & (uv <= bounds[1])).all()) or
                    not bool(((scales >= scale_bounds[0]) & (scales <= scale_bounds[1])).all()) or
                    not bool(((alpha > 0) & (alpha < 1)).all()) or
                    not bool(((retention >= 0) & (retention <= 1)).all())):
                raise ValueError('Surface patch support, scales, initial alpha or retention is invalid')
            if not torch.equal(sh_rest, prior_rest):
                raise ValueError('Fixed kernel SH-rest must equal its trusted surface material prior')
            for key, value in dict(center=center, frame=frame, uv_bounds=bounds, scale_bounds=scale_bounds,
                    thickness=thickness, uv=uv, logscale2d=scales.log(), angle=angle, retention=retention,
                    prior_dc=prior_dc, prior_sh_rest=prior_rest).items():
                fields[key].append(value)
            dc.append(color); rest.append(sh_rest); opacity.append(torch.logit(alpha))
            source = torch.as_tensor(spec.get('source_rows', torch.zeros(count)), device=device, dtype=torch.long)
            if source.shape != (count,) or bool((source < 0).any()):
                raise ValueError('Surface patch codec sources must be nonnegative kernel-aligned rows')
            sources.append(source)
            rows.append(torch.arange(offset, offset+count, device=device)); offset += count
            old_rows.append(torch.as_tensor(spec.get('old_rows', []), device=device, dtype=torch.long))
            canonical = {'center', 'frame', 'uv_bounds', 'scale_bounds', 'thickness', 'kernel_uv',
                         'kernel_scale', 'kernel_angle', 'kernel_alpha', 'kernel_color', 'kernel_dc',
                         'kernel_sh_rest', 'sh_rest', 'prior_dc', 'prior_sh_rest', 'retention', 'old_rows', 'source_rows'}
            metadata.append(_clone({key: value for key, value in spec.items() if key not in canonical}))
        point_shapes = dict(uv=(0, 2), logscale2d=(0, 2), angle=(0, 1), prior_dc=(0, 1, 3),
                            prior_sh_rest=(0, rest_count, 3))
        patch_shapes = dict(center=(0, 3), frame=(0, 3, 3), uv_bounds=(0, 2, 2),
                            scale_bounds=(0, 2, 2), thickness=(0, 1), retention=(0, 1))
        state = {key: (torch.cat(value) if key in point_shapes else torch.stack(value)) if value else
                 torch.empty((point_shapes | patch_shapes)[key], device=device, dtype=dtype)
                 for key, value in fields.items()}
        state.update(rows=rows, old_rows=old_rows, metadata=metadata,
                     radiance_centers=torch.empty(0, 3, device=device, dtype=dtype))
        self._xyz = nn.Parameter(torch.zeros(offset, 3, device=device, dtype=dtype), requires_grad=False)
        self._rotation = nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device=device, dtype=dtype).repeat(offset, 1), requires_grad=False)
        self._scaling = nn.Parameter(torch.zeros(offset, 3, device=device, dtype=dtype), requires_grad=False)
        self._features_dc = nn.Parameter(torch.cat(dc) if dc else torch.empty(0, 1, 3, device=device, dtype=dtype))
        self._features_rest = nn.Parameter(torch.cat(rest) if rest else torch.empty(0, rest_count, 3, device=device, dtype=dtype), requires_grad=False)
        self._opacity = nn.Parameter(torch.cat(opacity) if opacity else torch.empty(0, 1, device=device, dtype=dtype))
        self.source_rows = torch.cat(sources) if sources else torch.empty(0, device=device, dtype=torch.long)
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = self._xyz.new_zeros(offset)
        self.xyz_gradient_accum = self._xyz.new_zeros(offset, 1)
        self.denom = self._xyz.new_zeros(offset, 1)
        self.install_patch_state(state)
        if views is not None:
            self.set_radiance_views(views)
        if old_count is not None and (bool((self.owned_old_rows >= old_count).any()) or
                                      bool((self.source_rows >= old_count).any())):
            raise ValueError('Surface patch ownership/source rows exceed the original old population')
        # Raw placeholders aid existing capture code; phenotype always derives
        # them from local trainable coordinates and the fixed plane.
        with torch.no_grad():
            geometry = self._geometry(self.patch_uv, self.patch_logscale2d, self.patch_angle)
            for name, attr in (('xyz', '_xyz'), ('rotation', '_rotation'), ('logscale', '_scaling')):
                getattr(self, attr).copy_(geometry[name])
        return self

    def export_patch_specs(self):
        """Current mature local kernels for a topology-only next generation.

        The immutable material prior is exported separately from fitted DC and
        from codec source_rows. Fine-layout proposals can edit kernel arrays
        while retaining this same measured plane and old ownership.
        """
        raw = self.patch_phenotype()
        result = []
        for patch, rows in enumerate(self.patch_rows):
            spec = _clone(self.patch_metadata[patch])
            spec.update(_clone(dict(center=self.patch_center[patch], frame=self.patch_frame[patch],
                uv_bounds=self.patch_uv_bounds[patch], scale_bounds=self.patch_scale_bounds[patch],
                thickness=self.patch_thickness[patch], kernel_uv=self.patch_uv[rows],
                kernel_scale=self.patch_logscale2d[rows].exp(), kernel_angle=self.patch_angle[rows],
                kernel_dc=raw['sh_dc'][rows], kernel_sh_rest=self._features_rest[rows],
                kernel_alpha=self._opacity[rows].sigmoid(), prior_dc=self.patch_prior_dc[rows],
                prior_sh_rest=self.patch_prior_sh_rest[rows], retention=self.patch_retention[patch].clamp(0, self.patch_retention_cap),
                old_rows=self.patch_old_rows[patch], source_rows=self.source_rows[rows])))
            result.append(spec)
        return result

    def install_patch_state(self, state):
        device, dtype, count = self._xyz.device, self._xyz.dtype, len(self._xyz)
        def value(name):
            return torch.as_tensor(state[name], device=device, dtype=dtype).detach().clone()
        self.patch_rows = tuple(torch.as_tensor(row, device=device, dtype=torch.long).clone() for row in state['rows'])
        self.patch_old_rows = tuple(torch.as_tensor(row, device=device, dtype=torch.long).clone() for row in state['old_rows'])
        patches = len(self.patch_rows)
        self.patch_metadata = tuple(_clone(item, device) for item in state.get('metadata', [{} for _ in range(patches)]))
        if len(self.patch_metadata) != patches:
            raise ValueError('Surface patch metadata must align with complete patches')
        ids = torch.cat(self.patch_rows) if patches else torch.empty(0, device=device, dtype=torch.long)
        if len(self.patch_old_rows) != patches or not torch.equal(ids.sort().values, torch.arange(count, device=device)):
            raise ValueError('Surface patch rows must partition every model row exactly once')
        old = self.owned_old_rows
        if len(old.unique()) != len(old) or bool((old < 0).any()):
            raise ValueError('Old carrier ownership must be nonnegative and unique across patches')
        shapes = dict(center=(patches, 3), frame=(patches, 3, 3), uv_bounds=(patches, 2, 2),
            scale_bounds=(patches, 2, 2), thickness=(patches, 1), prior_dc=(count, 1, 3),
            prior_sh_rest=self._features_rest.shape)
        for key, shape in shapes.items():
            tensor = value(key)
            if tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Surface patch {key} has invalid shape or nonfinite values')
            setattr(self, 'patch_'+key, tensor)
        if (not bool((self.patch_uv_bounds[:, 0] < self.patch_uv_bounds[:, 1]).all()) or
                not bool((self.patch_scale_bounds[:, 0] > 0).all() &
                         (self.patch_scale_bounds[:, 0] <= self.patch_scale_bounds[:, 1]).all()) or
                not bool((self.patch_thickness > 0).all()) or
                not torch.equal(self.patch_prior_sh_rest, self._features_rest.detach())):
            raise ValueError('Surface patch state must retain valid support and its fixed material SH-rest')
        identity = torch.eye(3, device=device, dtype=dtype).expand(patches, -1, -1)
        if (not torch.allclose(self.patch_frame.transpose(1, 2)@self.patch_frame, identity, atol=2.e-5, rtol=2.e-5) or
                bool((torch.det(self.patch_frame) <= 0).any())):
            raise ValueError('Surface patch frames must be right-handed orthonormal columns')
        self._point_patch = torch.empty(count, device=device, dtype=torch.long)
        for index, row in enumerate(self.patch_rows):
            self._point_patch[row] = index
        self._frame_q = _frame_quaternion(self.patch_frame)
        for key, shape in dict(uv=(count, 2), logscale2d=(count, 2), angle=(count, 1), retention=(patches, 1)).items():
            tensor = value(key)
            if tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Surface patch latent {key} has invalid shape or nonfinite values')
            parameter = nn.Parameter(tensor)
            parameter.register_hook(self._clear_patch_cache)
            setattr(self, 'patch_'+key, parameter)
        self.set_retention_cap(state.get('retention_cap', 1.))
        self._radiance_centers = torch.as_tensor(state.get('radiance_centers', []), device=device, dtype=dtype).reshape(-1, 3).clone()
        self._features_dc.requires_grad_(True)
        self._opacity.requires_grad_(True)
        for attr in ('_xyz', '_rotation', '_scaling', '_features_rest'):
            getattr(self, attr).requires_grad_(False)
        self._clear_patch_cache()
        return self

    def patch_state(self, device='cpu'):
        state = dict(rows=self.patch_rows, old_rows=self.patch_old_rows, metadata=self.patch_metadata,
                     radiance_centers=self._radiance_centers, retention_cap=self.patch_retention_cap)
        for key in ('center', 'frame', 'uv_bounds', 'scale_bounds', 'thickness', 'uv',
                    'logscale2d', 'angle', 'retention', 'prior_dc', 'prior_sh_rest'):
            state[key] = getattr(self, 'patch_'+key)
        return _clone(state, device)

    def set_radiance_views(self, views):
        views = list(views.values()) if isinstance(views, dict) else views
        if torch.is_tensor(views):
            centers = views.to(self._xyz)
        else:
            centers = []
            for view in views:
                center = getattr(view, 'camera_center', None)
                if center is None:
                    center = torch.linalg.inv(view.world_view_transform)[3, :3]
                centers.append(center.to(self._xyz))
            centers = torch.stack(centers) if len(centers) else self._xyz.new_empty((0, 3))
        if centers.ndim != 2 or centers.shape[1] != 3 or not bool(torch.isfinite(centers).all()):
            raise ValueError('Surface radiance bounds require finite training camera centers[V,3]')
        self._radiance_centers = centers.detach().clone()
        self._clear_patch_cache()
        return self

    def _geometry(self, uv, logscale, angle):
        p = self._point_patch
        frame = self.patch_frame[p]
        xyz = self.patch_center[p]+(frame[:, :, :2]@uv[:, :, None])[:, :, 0]
        half = angle*.5
        qz = torch.cat((half.cos(), torch.zeros_like(half).expand(-1, 2), half.sin()), -1)
        rotation = _quat_product(self._frame_q[p], qz)
        scaling = torch.cat((logscale, self.patch_thickness[p].log()), -1)
        return dict(xyz=xyz, rotation=rotation, logscale=scaling)

    def radiance_dc_bounds(self, xyz=None):
        """Absolute coefficient box from trusted material, never codec source HDR."""
        from libgs.utils.sh import eval_sh
        xyz = self._geometry(self.patch_uv, self.patch_logscale2d, self.patch_angle)['xyz'] if xyz is None else xyz
        if not len(xyz):
            return self.patch_prior_dc.clone(), self.patch_prior_dc.clone()
        if not len(self._radiance_centers):
            if bool((self.patch_prior_sh_rest != 0).any()):
                raise ValueError('Set standard training radiance views before fitting inherited directional SH')
            angular = xyz.new_full((1, len(xyz), 3), .5)
        else:
            direction = F.normalize(xyz[None]-self._radiance_centers[:, None], dim=-1)
            coefficients = torch.cat((torch.zeros_like(self.patch_prior_dc), self.patch_prior_sh_rest), 1)
            angular = eval_sh(self.active_sh_degree, coefficients.transpose(1, 2)[None], direction)+.5
        prior = self.patch_prior_dc[:, 0][None]*C0+angular
        low, high = prior.amin(0).clamp_max(0.), prior.amax(0).clamp_min(1.)
        lower = ((low[None]-angular)/C0).amax(0)[:, None]
        upper = ((high[None]-angular)/C0).amin(0)[:, None]
        # The material prior is analytically feasible; retain it exactly when
        # subtracting large SH terms introduces a few roundoff bits.
        return torch.minimum(lower, self.patch_prior_dc), torch.maximum(upper, self.patch_prior_dc)

    def patch_phenotype(self):
        leaves = [self.patch_uv, self.patch_logscale2d, self.patch_angle, self.patch_retention,
                  self._features_dc, self._features_rest, self._opacity]
        key = (torch.is_grad_enabled(), self.active_sh_degree,
               tuple((id(x), x._version, x.requires_grad) for x in leaves))
        if self._patch_cache is not None and self._patch_cache[0] == key:
            return self._patch_cache[1]
        result = self._geometry(self.patch_uv, self.patch_logscale2d, self.patch_angle)
        lower, upper = self.radiance_dc_bounds(result['xyz'])
        dc = _ste_value(self._features_dc, self._features_dc.maximum(lower).minimum(upper))
        result.update(opacity=self._opacity, sh_dc=dc, sh_rest=self._features_rest)
        for value in result.values():
            if value.requires_grad:
                value.register_hook(self._clear_patch_cache)
        self._patch_cache = key, result
        return result

    def birth_parameter_value(self, name):
        return self.patch_phenotype()[name]

    def patch_quantized_raw(self, config, extent, reference, source_rows):
        """Quantize local coordinates first; preserve plane/thickness/material."""
        p = self._point_patch
        uv_bounds, scale_bounds = self.patch_uv_bounds[p], self.patch_scale_bounds[p].log()
        uv = _bounded_grid(self.patch_uv, uv_bounds[:, 0], uv_bounds[:, 1], config.new_xyz_step_extent*extent)
        scale = _bounded_grid(self.patch_logscale2d, scale_bounds[:, 0], scale_bounds[:, 1], config.new_logscale_step)
        angle = _ste_value(self.patch_angle, (self.patch_angle/config.new_rotation_step).round()*config.new_rotation_step)
        result = self._geometry(uv, scale, angle)
        source = torch.as_tensor(source_rows, device=self._xyz.device, dtype=torch.long)
        if source.shape != (len(self._xyz),) or bool(((source < 0) | (source >= len(reference['xyz']))).any()):
            raise ValueError('Surface patch source predictors must align with the actual reference')
        lower, upper = self.radiance_dc_bounds(result['xyz'])
        dc = _bounded_source_dc(self._features_dc, reference['sh_dc'][source].detach(), lower, upper, config.new_sh_step)
        pred_opacity = reference['opacity'][source].detach()
        opacity = _ste_value(self._opacity, pred_opacity+((self._opacity-pred_opacity)/config.new_opacity_step).round()*config.new_opacity_step)
        result.update(opacity=opacity, sh_dc=dc, sh_rest=self._features_rest)
        return result

    @torch.no_grad()
    def set_retention_cap(self, cap):
        cap = float(cap)
        if not math.isfinite(cap) or not 0 <= cap <= 1:
            raise ValueError('Surface patch retention cap must be in [0,1]')
        self.patch_retention_cap = cap
        self.patch_retention.clamp_(0, cap)
        self._clear_patch_cache()
        return self

    @torch.no_grad()
    def project_parameters(self, views=None):
        if views is not None:
            self.set_radiance_views(views)
        p = self._point_patch
        self.patch_uv.copy_(self.patch_uv.maximum(self.patch_uv_bounds[p, 0]).minimum(self.patch_uv_bounds[p, 1]))
        bounds = self.patch_scale_bounds[p].log()
        self.patch_logscale2d.copy_(self.patch_logscale2d.maximum(bounds[:, 0]).minimum(bounds[:, 1]))
        self.patch_angle.copy_((self.patch_angle+math.pi).remainder(2*math.pi)-math.pi)
        self.patch_retention.clamp_(0, self.patch_retention_cap)
        lower, upper = self.radiance_dc_bounds()
        self._features_dc.copy_(self._features_dc.maximum(lower).minimum(upper))
        self._clear_patch_cache()

    project_ = project_parameters

    def raw_old_alpha(self, reference_raw_opacity):
        """Continuous ownership takeover; A quantizes its OPD4 increments."""
        alpha = reference_raw_opacity.detach().sigmoid()
        rows = self.owned_old_rows
        if len(rows) == 0:
            return alpha
        if reference_raw_opacity.shape[1:] != (1,) or bool((rows >= len(alpha)).any()):
            raise ValueError('Old patch ownership must address the full original old-opacity layout')
        owner = torch.cat([torch.full((len(row),), index, device=rows.device, dtype=torch.long)
                           for index, row in enumerate(self.patch_old_rows)])
        target = alpha[rows]*self.patch_retention.clamp(0, self.patch_retention_cap)[owner]
        return alpha.index_copy(0, rows, target)

    retained_old_alpha = raw_old_alpha

    @property
    def get_xyz(self): return self.birth_parameter_value('xyz')
    @property
    def get_rotation(self): return F.normalize(self.birth_parameter_value('rotation'), dim=-1)
    @property
    def get_scaling(self): return self.birth_parameter_value('logscale').exp()
    @property
    def get_opacity(self): return self.birth_parameter_value('opacity').sigmoid()
    @property
    def get_features(self): return torch.cat((self.birth_parameter_value('sh_dc'), self._features_rest), 1)

    def get_covariance(self, scaling_modifier=1.):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def training_setup(self, args):
        self._training_args = args
        self.percent_dense = args.percent_dense
        self.xyz_gradient_accum = self._xyz.new_zeros(len(self._xyz), 1)
        self.denom = self._xyz.new_zeros(len(self._xyz), 1)
        groups = ((self.patch_uv, args.position_lr_init*self.spatial_lr_scale, 'xyz'),
                  (self.patch_logscale2d, args.scaling_lr, 'scaling'),
                  (self.patch_angle, args.rotation_lr, 'rotation'),
                  (self._features_dc, args.feature_lr, 'f_dc'),
                  (self._opacity, args.opacity_lr, 'opacity'),
                  (self.patch_retention, args.opacity_lr, 'patch_retention'))
        self.optimizer = torch.optim.Adam([dict(params=[parameter], lr=lr, name=name)
                                          for parameter, lr, name in groups], lr=0., eps=1.e-15)

    def capture(self):
        base = (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
            self._scaling, self._rotation, self._opacity, self.max_radii2D,
            self.xyz_gradient_accum, self.denom,
            self.optimizer.state_dict() if self.optimizer is not None else None, self.spatial_lr_scale)
        return _clone(dict(base=base, patch=self.patch_state(), source_rows=self.source_rows), 'cpu')

    def restore(self, packet, training_args, device=None):
        base, state = packet['base'], packet['patch']
        device = device or base[1].device
        self.active_sh_degree, self.spatial_lr_scale = base[0], base[11]
        for attr, index in (('_xyz', 1), ('_features_dc', 2), ('_features_rest', 3),
                            ('_scaling', 4), ('_rotation', 5), ('_opacity', 6)):
            setattr(self, attr, nn.Parameter(_clone(base[index], device)))
        self.max_radii2D = _clone(base[7], device)
        self.install_patch_state(_clone(state, device))
        self.training_setup(training_args)
        self.xyz_gradient_accum, self.denom = _clone(base[8], device), _clone(base[9], device)
        if base[10] is not None:
            self.optimizer.load_state_dict(_clone(base[10], device))
        self.source_rows = _clone(packet.get('source_rows'), device)
        return self

    def patch_subset(self, selection):
        """Independent subset with entire ownership/latents/Adam state retained."""
        choice = torch.as_tensor(selection, device=self._xyz.device)
        if choice.dtype == torch.bool:
            if choice.shape != (len(self._xyz),):
                raise ValueError('Surface patch subset mask must align with model rows')
            ids = choice.nonzero().flatten()
        else:
            ids = choice.long().flatten()
        if len(ids.unique()) != len(ids) or bool(((ids < 0) | (ids >= len(self._xyz))).any()):
            raise ValueError('Surface patch subset rows must be valid and unique')
        inverse = torch.full((len(self._xyz),), -1, device=ids.device, dtype=torch.long)
        inverse[ids] = torch.arange(len(ids), device=ids.device)
        selected = []
        for index, rows in enumerate(self.patch_rows):
            keep = inverse[rows] >= 0
            if bool(keep.any()) != bool(keep.all()):
                raise ValueError('A surface patch must be selected atomically with its complete ownership')
            if bool(keep.all()):
                selected.append(index)
        patches = torch.tensor(selected, device=ids.device, dtype=torch.long)
        result = type(self)(self.max_sh_degree)
        for attr in RAW_ATTRIBUTES:
            setattr(result, attr, nn.Parameter(_clone(getattr(self, attr)[ids])))
        result.active_sh_degree, result.spatial_lr_scale = self.active_sh_degree, self.spatial_lr_scale
        result.max_radii2D = _clone(self.max_radii2D[ids]) if len(self.max_radii2D) else self._xyz.new_zeros(len(ids))
        result.xyz_gradient_accum = (_clone(self.xyz_gradient_accum[ids]) if len(self.xyz_gradient_accum) else
                                     self._xyz.new_zeros(len(ids), 1))
        result.denom = _clone(self.denom[ids]) if len(self.denom) else self._xyz.new_zeros(len(ids), 1)
        result.source_rows = _clone(self.source_rows[ids]) if self.source_rows is not None else None
        state = self.patch_state(device=self._xyz.device)
        state['rows'] = [inverse[self.patch_rows[index]] for index in selected]
        state['old_rows'] = [state['old_rows'][index] for index in selected]
        state['metadata'] = [state['metadata'][index] for index in selected]
        for key in ('center', 'frame', 'uv_bounds', 'scale_bounds', 'thickness', 'retention'):
            state[key] = state[key][patches]
        for key in ('uv', 'logscale2d', 'angle', 'prior_dc', 'prior_sh_rest'):
            state[key] = state[key][ids]
        result.install_patch_state(state)
        if self.optimizer is not None:
            result.training_setup(self._training_args)
            old_groups = {group['name']: group for group in self.optimizer.param_groups}
            for group in result.optimizer.param_groups:
                old = old_groups[group['name']]
                for key, value in old.items():
                    if key != 'params': group[key] = _clone(value)
                oldp, newp = old['params'][0], group['params'][0]
                state = self.optimizer.state.get(oldp)
                if state:
                    subset = patches if group['name'] == 'patch_retention' else ids
                    result.optimizer.state[newp] = {key: _clone(value[subset] if torch.is_tensor(value) and value.shape == oldp.shape else value)
                                                   for key, value in state.items()}
        return result

    def prune_points(self, remove):
        remove = torch.as_tensor(remove, device=self._xyz.device)
        mask = remove if remove.dtype == torch.bool else torch.zeros(len(self._xyz), device=self._xyz.device, dtype=torch.bool)
        if remove.dtype != torch.bool: mask[remove.long()] = True
        subset = self.patch_subset(~mask)
        self.__dict__.clear()
        self.__dict__.update(subset.__dict__)


def install_surface_patches(model, state):
    """Install fixed-plane traits on an already populated newborn raw model."""
    if not isinstance(model, SurfacePatchModel):
        model.__class__ = SurfacePatchModel
        model.covariance_activation = _covariance
        model._patch_cache = None
        model._training_args = None
    return model.install_patch_state(state)
