"""Image objectives and a local removal-utility estimate for fitted newborns."""
import math

import torch
from libgs.metric import ssim


def birth_image_loss(image, target, ssim_weight=0.):
    loss = (image-target).square().mean()
    if ssim_weight:
        loss = loss + ssim_weight * (1-ssim(image, target))
    return loss


def useful_birth_subsets(utility, fractions):
    """Keep the full batch as an option, then smaller positive-utility pools."""
    if utility.ndim != 1 or not bool(torch.isfinite(utility).all()):
        raise ValueError('Newborn image utility must be a finite vector')
    all_rows = torch.arange(len(utility), device=utility.device)
    pools = [all_rows]
    ranked = torch.argsort(utility, descending=True, stable=True)
    ranked = ranked[utility[ranked] > 0]
    counts = {len(all_rows)}
    for fraction in fractions:
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Useful birth fractions must lie in (0, 1]')
        count = math.ceil(len(ranked)*fraction)
        if count and count not in counts:
            pools.append(ranked[:count].sort().values)
            counts.add(count)
    return pools


def midpoint_image_utility(renderer, cameras, background, logits, full_alpha,
                           baseline_mse, ssim_weight=0.):
    """Estimate removal benefit at half alpha, including occlusion/compositing.

    A single splat's rendered image is linear in its alpha while other splats
    are fixed (apart from rasterizer cutoffs). MSE's midpoint derivative thus
    estimates its finite removal cost. Joint interactions and the SSIM term
    remain approximate; final admission always renders the actual subsets.
    """
    if len(cameras) != len(baseline_mse) or not cameras:
        raise ValueError('Utility requires aligned baseline camera errors')
    score = torch.zeros_like(logits)
    with torch.enable_grad():
        for camera, error in zip(cameras, baseline_mse):
            image = renderer(camera, background)['render'].clamp(0, 1)
            loss = birth_image_loss(image, camera.image, ssim_weight) / max(float(error), 1e-8)
            gradient, = torch.autograd.grad(loss, logits)
            score -= gradient.detach() / len(cameras)
    half_alpha = logits.detach().sigmoid()
    score *= full_alpha / (half_alpha * (1-half_alpha)).clamp_min(1e-8)
    score = score.flatten()
    if not bool(torch.isfinite(score).all()):
        raise ValueError('Rasterizer produced nonfinite newborn image utility')
    return score
