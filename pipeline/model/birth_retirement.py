"""Balanced retirement by observed compositing and validation image quality."""

import math

import torch

from .multiview_birth import validation_edit_gain


def retirement_count_trials(birth_count, tolerance):
    if not 0 <= tolerance < 1 or birth_count < 0:
        raise ValueError('Invalid retirement count band')
    return sorted({math.ceil((1-tolerance)*birth_count-1e-9), birth_count,
                   math.floor((1+tolerance)*birth_count+1e-9)})


def rank_retirement_candidates(current, previous, sources, previous_birth_count=0):
    """Keep recently visible points and give last-frame births one grace frame."""
    if current.ndim != 1 or previous.shape != current.shape:
        raise ValueError('Current/previous contribution arrays must be aligned')
    if (not bool(torch.isfinite(current).all() and torch.isfinite(previous).all()) or
            bool((current < 0).any() or (previous < 0).any())):
        raise ValueError('Contribution weights must be finite and nonnegative')
    count = len(current)
    if not 0 <= previous_birth_count <= count:
        raise ValueError('Previous births do not fit the old point layout')
    sources = torch.as_tensor(sources, device=current.device, dtype=torch.long)
    if sources.ndim != 1 or bool(((sources < 0) | (sources >= count)).any()):
        raise ValueError('Birth predictors must be valid old rows')
    protected = torch.zeros(count, device=current.device, dtype=torch.bool)
    protected[sources] = True
    if previous_birth_count:
        protected[-previous_birth_count:] = True
    score = torch.maximum(current, previous)
    eligible = (~protected).nonzero(as_tuple=True)[0]
    return eligible[torch.argsort(score[eligible], stable=True)], protected


def choose_validation_trial(before, trials, min_gain_db, max_ssim_drop):
    """Select a passing trial, so an unsafe PSNR maximum cannot hide a safe one."""
    passing = [i for i, metrics in enumerate(trials)
               if validation_edit_gain(before, metrics, min_gain_db=min_gain_db,
                                       max_ssim_drop=max_ssim_drop)[0]]
    if not passing:
        return None
    return max(passing, key=lambda i: (trials[i]['psnr'], -i))


def compositing_contributions(renderer, cameras, background):
    """max_camera sum_pixel(alpha*T), obtained exactly by color derivatives."""
    count = len(renderer.gaussians.get_xyz)
    scores = background.new_zeros(count)
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
