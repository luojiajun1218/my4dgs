"""Retire weak old Gaussians while keeping birth predictors decodable."""

import math

import torch


def birth_retirement_rows(old_alpha, source_rows, current_contribution,
                          previous_contribution, *, previous_birth_count=0,
                          count_tolerance=0.2, opacity_threshold=0.01,
                          protect_previous_births=True):
    """Choose D old rows from actual two-frame contribution and the K band.

    Newborn source rows and (optionally) the previous frame's appended births
    stay in the reference pool so every delta has a stable predictor. Among
    the remaining rows, low-alpha points set the natural retirement count;
    the count is then clamped to the configured integer band and ordered by
    maximum contribution in the current and previous frame.
    """
    alpha = torch.as_tensor(old_alpha).detach().flatten()
    current = torch.as_tensor(current_contribution, device=alpha.device).detach().flatten()
    previous = torch.as_tensor(previous_contribution, device=alpha.device).detach().flatten()
    sources = torch.as_tensor(source_rows, device=alpha.device)
    count, births = len(alpha), len(sources)
    if (count < 1 or current.shape != (count,) or previous.shape != (count,) or
            sources.ndim != 1 or sources.dtype not in (torch.int32, torch.int64) or
            bool(((sources < 0) | (sources >= count)).any()) or
            not bool(torch.isfinite(alpha).all()) or
            bool(((alpha < 0) | (alpha > 1)).any()) or
            not bool(torch.isfinite(current).all()) or
            not bool(torch.isfinite(previous).all()) or
            bool((current < 0).any() or (previous < 0).any())):
        raise ValueError('Retirement inputs must be finite, aligned alpha and contribution values')
    if (isinstance(count_tolerance, bool) or not isinstance(count_tolerance, (int, float)) or
            not math.isfinite(count_tolerance) or not 0 <= count_tolerance < 1 or
            isinstance(opacity_threshold, bool) or
            not isinstance(opacity_threshold, (int, float)) or
            not math.isfinite(opacity_threshold) or not 0 <= opacity_threshold < 1):
        raise ValueError('Invalid retirement tolerance or opacity threshold')
    if (isinstance(previous_birth_count, bool) or
            not isinstance(previous_birth_count, int) or
            not 0 <= previous_birth_count <= count or
            not isinstance(protect_previous_births, bool)):
        raise ValueError('Invalid previous-frame birth protection count')

    protected = torch.zeros(count, device=alpha.device, dtype=torch.bool)
    if births:
        protected[sources.long()] = True
    if protect_previous_births and previous_birth_count:
        protected[count - previous_birth_count:] = True
    eligible = (~protected).nonzero(as_tuple=True)[0]

    if births == 0:
        retired_count = 0
        natural_count = 0
        lower = upper = 0
    else:
        lower = math.ceil((1 - count_tolerance) * births - 1e-9)
        upper = math.floor((1 + count_tolerance) * births + 1e-9)
        capacity = len(eligible)
        if lower > capacity:
            raise ValueError(
                f'Cannot satisfy retirement band [{lower}, {upper}] for K={births}: '
                f'only {capacity} unprotected old rows can be removed. '
                'No births have been dropped.')
        natural_count = int(alpha[eligible].lt(opacity_threshold).sum())
        retired_count = min(max(natural_count, lower), upper, capacity)

    score = torch.maximum(current, previous)
    ranked = eligible[torch.argsort(score[eligible], stable=True)]
    retired_rows = ranked[:retired_count]
    remove = torch.zeros(count, device=alpha.device, dtype=torch.bool)
    remove[retired_rows] = True
    survivors = (~remove).nonzero(as_tuple=True)[0]
    return survivors, dict(
        birth_count=births,
        natural_retirements=natural_count,
        retired_count=retired_count,
        min_retirements=lower,
        max_retirements=upper,
        count_difference=births - retired_count,
        eligible_old_count=len(eligible),
        protected_source_count=int(torch.unique(sources).numel()),
        protected_previous_birth_count=(previous_birth_count if protect_previous_births else 0),
        opacity_threshold=float(opacity_threshold),
        contribution_rule='max_current_previous_training_view',
    )


def compositing_contributions(renderer, cameras, background):
    """Return max-camera sum_pixel(alpha*T), measured by color derivatives."""
    cameras = list(cameras)
    count = len(renderer.gaussians.get_xyz)
    scores = background.new_zeros(count)
    if not cameras:
        raise ValueError('Contribution ranking requires at least one training camera')
    with torch.enable_grad():
        for camera in cameras:
            colors = background.new_zeros(count, 3, requires_grad=True)
            image = renderer(camera, background, override_color=colors)['render']
            weight, = torch.autograd.grad(image.sum(), colors)
            weight = weight.detach().mean(dim=1)
            if not bool(torch.isfinite(weight).all()) or bool((weight < -1e-6).any()):
                raise ValueError('Rasterizer produced invalid contribution weights')
            scores = torch.maximum(scores, weight.clamp_min(0))
    return scores
