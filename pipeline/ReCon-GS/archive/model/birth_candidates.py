"""Read-only appearance evidence for births missed by position gradients."""

import math
import time

import torch
from torch.nn.functional import l1_loss
from libgs.metric import ssim


def select_appearance_birth_sources(scores, eligible, fraction):
    """Select up to ceil(N * fraction) positive-scored eligible GS, stably.

    Scores [N] are mean visible-view DC-color gradient norms. The budget is
    independent of the number of geometric candidates, so a stationary
    appearance change can propose births even with zero position gradients.
    """
    if (scores.ndim != 1 or not scores.is_floating_point() or
            eligible.shape != scores.shape or eligible.dtype != torch.bool or
            eligible.device != scores.device or not bool(torch.isfinite(scores).all())):
        raise ValueError('Appearance scores require finite floating [N] and bool eligibility')
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or
            not math.isfinite(fraction) or not 0 <= fraction <= 1):
        raise ValueError('Appearance birth fraction must be finite and in [0, 1]')
    rows = torch.where(eligible & (scores > 0))[0]
    count = math.ceil(len(scores) * fraction)
    return rows[torch.argsort(scores[rows], descending=True, stable=True)[:count]]


@torch.enable_grad()
def collect_appearance_birth_sources(module):
    """Probe the committed old-GS state using optimization training cameras.

    A temporary DC tensor receives image-only derivatives; the original GS
    parameters, optimizer, CPU sampling RNG and renderer binding are restored.
    No held-out camera, storage term or optimizer step selects a source.
    """
    model, cfg = module.gaussians, module.residual_motion.config
    names = list(cfg.birth_appearance_camera_names)
    dataset = module.datamodule.get_train_dataset()
    indices = module.datamodule._camera_indices(dataset, names)
    configured = getattr(module.datamodule.config, 'training_camera_names', None)
    validation = set(getattr(module.datamodule.config, 'validation_camera_names', None) or [])
    allowed = set(configured) if configured is not None else {item.path.name for item in dataset.items}
    if not set(names) <= allowed or set(names) & validation:
        raise ValueError('Appearance birth cameras must belong to the optimization training split')
    original_renderer = module.renderer.gaussians
    original_dc = model._features_dc
    leaf = torch.nn.Parameter(original_dc.detach(), requires_grad=True)
    scores = original_dc.new_zeros(len(model.get_xyz))
    observations = torch.zeros(len(scores), dtype=torch.long, device=scores.device)
    observed = []
    module.trainer.synchronize()
    started = time.perf_counter()
    try:
        module.renderer.gaussians = model
        model._features_dc = leaf
        with torch.random.fork_rng(devices=[]):
            for viewpoint in module.datamodule.train_dataloader(random=False, indices=indices):
                viewpoint = viewpoint.to(module.device)
                observed.append(viewpoint.path.name)
                package = module.renderer(viewpoint, module.background)
                weight = module.config.lambda_dssim
                loss = ((1 - weight) * l1_loss(package['render'], viewpoint.image) +
                        weight * (1 - ssim(package['render'], viewpoint.image)))
                gradient, = torch.autograd.grad(loss, leaf)
                gradient = gradient.detach().flatten(1)
                valid = package['visibility_filter'] & torch.isfinite(gradient).all(-1)
                scores += torch.where(valid, gradient.norm(dim=-1), 0)
                observations += valid.long()
    finally:
        model._features_dc = original_dc
        module.renderer.gaussians = original_renderer
    if observed != names:
        raise ValueError('Appearance probe did not return exactly the requested cameras')
    scores /= observations.clamp_min(1)
    eligible = ((observations > 0) &
                (model.get_scaling.amax(-1) <= model.percent_dense * module.datamodule.cameras_extent))
    rows = select_appearance_birth_sources(scores, eligible, cfg.birth_appearance_fraction)
    module.trainer.synchronize()
    return rows, dict(enabled=True, cameras=observed, extra_renders=len(observed),
        eligible_gs=int(eligible.sum()), supported_gs=int((observations > 0).sum()),
        requested_sources=len(rows), fraction=cfg.birth_appearance_fraction,
        seconds=time.perf_counter() - started,
        semantics='Image-only DC-color gradient norms at one fixed post-motion state. '
                  'Only small, observed GS qualify. Budget uses old-GS count; no old attributes update.')
