"""Opt-in, no-grad component evidence; no production renderer route changes.

Dominance is native accepted alpha*T ownership, not nearest depth, photometric
ownership or ground-truth surface confidence. Conditional geometry describes
one projected Gaussian; other-component mass remains an explicit diagnostic.
"""

import functools
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


NATIVE_AA_VARIANCE = .3


@functools.lru_cache(maxsize=1)
def _surface_backend():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[2] / 'extensions' / 'observed_rasterizer'
    return load(name='my4dgs_observed_surface_aux', sources=[str(root / name) for name in (
        'cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/forward.cu',
        'cuda_rasterizer/backward.cu', 'rasterize_points.cu', 'surface_aux.cu', 'surface_aux_ext.cpp')],
        extra_include_paths=[str(root / 'third_party/glm')],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)


def _ownership_maps(dominant_id, max_talpha, full_T):
    if dominant_id.shape != max_talpha.shape or dominant_id.shape != full_T.shape or dominant_id.ndim != 2:
        raise ValueError('Surface ownership ID, weight and full_T must have identical HxW shapes')
    coverage = 1-full_T
    positive = coverage > 0
    share = torch.where(positive, max_talpha/coverage.clamp_min(torch.finfo(coverage.dtype).tiny), 0.)
    return dict(dominant_id=dominant_id, max_talpha=max_talpha, full_T=full_T,
                coverage=coverage, dominant_share=share,
                other_component_weight=coverage-max_talpha)


class SurfaceComponentRasterizer(nn.Module):
    """Independent forward-only counterpart of the observed rasterizer.

    The old Python/C++ entry points and their tuples remain unchanged. Returned
    IDs index this exact input population; codec predictor/source_rows are not
    geometric identity maps. P=0 preserves legacy RGB behavior and returns
    ID=-1, weight=0, final transmittance=1.
    """
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    @torch.no_grad()
    def forward(self, means3D, means2D=None, opacities=None, shs=None,
                colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        if not means3D.is_cuda or means3D.dtype != torch.float32 or means3D.ndim != 2 or means3D.shape[1] != 3:
            raise ValueError('Surface aux requires float32 CUDA means3D[N,3]')
        if (shs is None) == (colors_precomp is None):
            raise ValueError('Provide exactly one of SH coefficients or precomputed colors')
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
                (scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise ValueError('Provide scale/rotation together or a precomputed covariance')
        count = len(means3D)
        if opacities is None or opacities.shape != (count, 1):
            raise ValueError('Surface aux opacities must have shape (N,1)')
        settings = self.raster_settings
        def tensor(value):
            if value is None:
                return means3D.new_empty(0)
            if value.device != means3D.device or value.dtype != means3D.dtype:
                raise ValueError('Surface aux floating inputs must share one CUDA device and float32 dtype')
            return value.contiguous()
        color, radii, observed, point_id, weight, final_T = _surface_backend().rasterize_surface_aux(
            tensor(settings.bg), tensor(means3D), tensor(colors_precomp), tensor(opacities),
            tensor(scales), tensor(rotations), float(settings.scale_modifier), tensor(cov3D_precomp),
            tensor(settings.viewmatrix), tensor(settings.projmatrix), float(settings.tanfovx),
            float(settings.tanfovy), int(settings.image_height), int(settings.image_width),
            tensor(shs), int(settings.sh_degree), tensor(settings.campos),
            bool(settings.prefiltered), bool(settings.debug))
        result = _ownership_maps(point_id, weight, final_T)
        result.update(render=color, radii=radii, observed=observed)
        return result


@torch.no_grad()
def render_surface_components(renderer, viewpoint, background, scaling_modifier=1.):
    """Read an existing renderer/model without mutating its flags or diagnostics."""
    gaussians = renderer.gaussians
    settings = SimpleNamespace(image_height=int(viewpoint.image.shape[1]),
        image_width=int(viewpoint.image.shape[2]), tanfovx=math.tan(float(viewpoint.fovx)*.5),
        tanfovy=math.tan(float(viewpoint.fovy)*.5), bg=background,
        scale_modifier=scaling_modifier, viewmatrix=viewpoint.world_view_transform,
        projmatrix=viewpoint.full_proj_transform, sh_degree=gaussians.active_sh_degree,
        campos=viewpoint.camera_center, prefiltered=False, debug=renderer.config.debug)
    kwargs = dict(means3D=gaussians.get_xyz, opacities=gaussians.get_opacity,
                  shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None)
    if renderer.config.compute_cov3D_python:
        kwargs['cov3D_precomp'] = gaussians.get_covariance(scaling_modifier)
    else:
        kwargs.update(scales=gaussians.get_scaling, rotations=gaussians.get_rotation)
    kwargs = renderer.apply_color(kwargs, viewpoint)
    return SurfaceComponentRasterizer(settings)(**kwargs)


def unpack_covariance(covariance):
    if covariance.ndim == 3 and covariance.shape[1:] == (3, 3):
        return covariance
    if covariance.ndim != 2 or covariance.shape[1] != 6:
        raise ValueError('World covariance must have shape (N,3,3) or packed (N,6)')
    a, b, c, d, e, f = covariance.unbind(-1)
    return torch.stack((a, b, c, b, d, e, c, e, f), -1).reshape(-1, 3, 3)


def covariance_from_scale_rotation(scales, rotations, scale_modifier=1.):
    """Native quaternion formula, with the exact rotation supplied to rendering.

    Native preprocessing does not normalize its quaternion. Callers should pass
    their actual normalized get_rotation output, not a different raw preimage.
    """
    if scales.ndim != 2 or scales.shape[1] != 3 or rotations.shape != (len(scales), 4):
        raise ValueError('Component scale/rotation shapes must be (N,3) and (N,4)')
    w, x, y, z = rotations.unbind(-1)
    basis = torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)
    return basis@torch.diag_embed((scales*scale_modifier).square())@basis.transpose(-1, -2)


@torch.no_grad()
def component_projection_statistics(means3D, covariance, viewmatrix, projmatrix,
                                    tan_fovx, tan_fovy, width, height):
    """Conditional-Gaussian coefficients matching native EWA linearization.

    Camera coordinates use the existing row-vector matrices. Sxy includes the
    native .3 pixel^2 filter and J its +/-1.3 FOV clamp. Geometry is a proxy of
    one fitted Gaussian, not a posterior over true surfaces or mixture labels.
    """
    if means3D.ndim != 2 or means3D.shape[1] != 3 or not means3D.is_floating_point():
        raise ValueError('Component means must be floating (N,3)')
    covariance = unpack_covariance(covariance).to(means3D)
    if covariance.shape[0] != len(means3D) or viewmatrix.shape != (4, 4) or projmatrix.shape != (4, 4):
        raise ValueError('Component covariance/camera dimensions disagree')
    if width <= 0 or height <= 0 or tan_fovx <= 0 or tan_fovy <= 0:
        raise ValueError('Surface component image/fov parameters must be positive')
    viewmatrix, projmatrix = viewmatrix.to(means3D), projmatrix.to(means3D)
    count, dtype = len(means3D), means3D.dtype
    identity = torch.eye(3, device=means3D.device, dtype=dtype).expand(count, -1, -1)
    finite = torch.isfinite(means3D).all(-1) & torch.isfinite(covariance).all((-1, -2))
    sym = (covariance+covariance.transpose(-1, -2))*.5
    sym = torch.where(finite[:, None, None], sym, identity)
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    scale = eigenvalues.abs().amax(-1).clamp_min(torch.finfo(dtype).tiny)
    tolerance = 32*torch.finfo(dtype).eps*scale
    psd = eigenvalues[:, 0] >= -tolerance
    safe_covariance = torch.where(psd[:, None, None], sym, identity)
    safe_means = torch.where(finite[:, None], means3D, 0.)
    camera = safe_means@viewmatrix[:3, :3]+viewmatrix[3, :3]
    rotation = viewmatrix[:3, :3].T
    camera_covariance = rotation[None]@safe_covariance@rotation.T[None]
    z = camera[:, 2]
    safe_z = torch.where(z != 0, z, torch.ones_like(z))
    ratio = camera[:, :2]/safe_z[:, None]
    limits = means3D.new_tensor([1.3*float(tan_fovx), 1.3*float(tan_fovy)])
    clipped = ratio.maximum(-limits).minimum(limits)
    focal = means3D.new_tensor([float(width)/(2*float(tan_fovx)), float(height)/(2*float(tan_fovy))])
    jacobian = means3D.new_zeros(count, 2, 3)
    jacobian[:, 0, 0], jacobian[:, 1, 1] = focal[0]/safe_z, focal[1]/safe_z
    jacobian[:, :, 2] = -focal[None]*clipped/safe_z[:, None]
    screen_covariance = jacobian@camera_covariance@jacobian.transpose(-1, -2)
    screen_covariance = screen_covariance+NATIVE_AA_VARIANCE*torch.eye(2, device=means3D.device, dtype=dtype)[None]
    cross = (camera_covariance[:, 2:3]@jacobian.transpose(-1, -2))[:, 0]
    gain = torch.linalg.solve(screen_covariance, cross[:, :, None])[:, :, 0]
    raw_schur = camera_covariance[:, 2, 2]-(cross*gain).sum(-1)
    # Positive residual form of the same Schur complement avoids subtracting
    # two large almost-equal depths for thin tilted splats.
    residual = means3D.new_tensor([0., 0., 1.])[None]-(jacobian.transpose(-1, -2)@gain[:, :, None])[:, :, 0]
    variance = (residual*(camera_covariance@residual[:, :, None])[:, :, 0]).sum(-1)+NATIVE_AA_VARIANCE*gain.square().sum(-1)
    variance_scale = camera_covariance.abs().amax((-1, -2))
    numerical_tolerance = 32*torch.finfo(dtype).eps*variance_scale
    homogeneous = torch.cat((safe_means, torch.ones_like(z[:, None])), -1)@projmatrix
    ndc = homogeneous[:, :2]/(homogeneous[:, 3:4]+1.e-7)
    projected = ((ndc+1)*means3D.new_tensor([width, height])[None]-1)*.5
    valid = finite & psd & (z > 0) & (variance >= -numerical_tolerance)
    valid &= torch.isfinite(projected).all(-1) & torch.isfinite(gain).all(-1) & torch.isfinite(variance)
    center = torch.linalg.inv(viewmatrix)[3, :3]
    normal = eigenvectors[:, :, 0]
    facing = (normal*(center[None]-safe_means)).sum(-1)
    normal = normal*torch.where(facing >= 0, 1., -1.)[:, None]
    gap = eigenvalues[:, 1]-eigenvalues[:, 0]
    normal_unique = gap > tolerance
    anisotropy = gap/eigenvalues[:, 1].abs().clamp_min(torch.finfo(dtype).tiny)
    nan = means3D.new_tensor(float('nan'))
    return dict(camera_mean=torch.where(valid[:, None], camera, nan),
        projected_mean=torch.where(valid[:, None], projected, nan),
        depth_gain=torch.where(valid[:, None], gain, nan),
        depth_variance=torch.where(valid, variance.clamp_min(0), nan), raw_schur=raw_schur,
        world_normal=torch.where(valid[:, None], normal, nan), covariance_eigenvalues=eigenvalues,
        normal_unique=normal_unique & valid, normal_anisotropy=anisotropy,
        valid=valid, projection_clamped=(ratio != clipped).any(-1),
        screen_covariance=screen_covariance, camera_covariance=camera_covariance, projection_jacobian=jacobian)


@torch.no_grad()
def conditional_component_evidence(statistics, dominant_id, max_talpha, full_T):
    """Gather per-pixel within-component geometry while retaining association mass.

    Different labels/depths must not be pooled back into one uncertainty before
    geometric/photo certification. A small conditional variance says nothing
    about the unknown geometry represented by other_component_weight.
    """
    result = _ownership_maps(dominant_id, max_talpha, full_T)
    height, width = dominant_id.shape
    count = len(statistics['valid'])
    dtype, device = max_talpha.dtype, dominant_id.device
    depth = torch.full((height, width), float('nan'), dtype=dtype, device=device)
    variance = depth.clone()
    normal = torch.full((3, height, width), float('nan'), dtype=dtype, device=device)
    valid = torch.zeros_like(dominant_id, dtype=torch.bool)
    normal_unique = valid.clone()
    normal_anisotropy = depth.clone()
    if count:
        indices = dominant_id.long().clamp(0, count-1)
        valid = (dominant_id >= 0) & (dominant_id < count) & statistics['valid'][indices]
        yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                               torch.arange(width, device=device, dtype=dtype), indexing='ij')
        pixel = torch.stack((xx, yy), -1)
        offset = pixel-statistics['projected_mean'][indices]
        values = statistics['camera_mean'][indices, 2]+(offset*statistics['depth_gain'][indices]).sum(-1)
        valid &= torch.isfinite(values) & (values > 0)
        depth = torch.where(valid, values.to(dtype), depth)
        variance = torch.where(valid, statistics['depth_variance'][indices].to(dtype), variance)
        gathered_normal = statistics['world_normal'][indices].permute(2, 0, 1).to(dtype)
        normal = torch.where(valid[None], gathered_normal, normal)
        normal_unique = valid & statistics['normal_unique'][indices]
        normal_anisotropy = torch.where(valid, statistics['normal_anisotropy'][indices].to(dtype), normal_anisotropy)
    result.update(depth=depth, depth_variance=variance, normal=normal, component_valid=valid,
                  normal_unique=normal_unique, normal_anisotropy=normal_anisotropy,
                  component_statistics=statistics)
    return result
