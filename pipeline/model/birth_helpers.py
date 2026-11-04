"""Shared tensor-state and SH math for adaptive births."""

import torch

C0 = .28209479177387814
C1 = .4886025119029199


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
