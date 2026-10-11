"""New-space birth proposals from multi-view training residuals."""

import math
import time

import numpy as np
import torch
from torch.nn import functional as F


def backproject_pixels(pixels, depth, view):
    """Map pixel centers [N,2] and eye-Z [N] to world points [N,3]."""
    height, width = view.image.shape[-2:]
    ndc = (pixels + .5) * pixels.new_tensor([2 / width, 2 / height]) - 1
    eye = torch.stack((ndc[:, 0] * depth * math.tan(view.fovx / 2),
                       ndc[:, 1] * depth * math.tan(view.fovy / 2),
                       depth, torch.ones_like(depth)), dim=-1)
    return (eye @ torch.linalg.inv(view.world_view_transform))[:, :3]


def project_points(points, view):
    """Return rasterizer NDC coordinates and eye-Z, using row matrices."""
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=-1)
    clip = homogeneous @ view.full_proj_transform
    ndc = clip[:, :2] / clip[:, 3:].clamp_min(torch.finfo(points.dtype).eps)
    eye_z = (homogeneous @ view.world_view_transform)[:, 2]
    return ndc, eye_z


def grid_error_peaks(error, cell_size, limit, threshold):
    """One residual maximum per image cell; deterministic score/row ties."""
    values, indices = F.max_pool2d(error[None, None], cell_size, stride=cell_size,
                                  ceil_mode=True, return_indices=True)
    values, indices = values.flatten(), indices.flatten()
    rows = torch.where(values >= threshold)[0]
    order = torch.argsort(values[rows], descending=True, stable=True)[:limit]
    selected = indices[rows[order]]
    return torch.stack((selected % error.shape[1], selected // error.shape[1]), dim=-1).to(error)


def multiview_birth_evidence(points, packages, min_error, min_alpha, occlusion_tolerance):
    """Score hypotheses without reading held-out views or updating old GS."""
    scores = points.new_zeros(len(points))
    colors = points.new_zeros(len(points), 3)
    support = torch.zeros(len(points), device=points.device, dtype=torch.long)
    if not len(points):
        return scores, colors, support
    for view, error, depth, alpha in packages:
        grid, eye_z = project_points(points, view)
        texture = torch.cat((error[None], depth[None], alpha[None], view.image), dim=0)
        sampled = F.grid_sample(texture[None], grid[None, :, None],
                                mode='bilinear', padding_mode='zeros', align_corners=False)[0, :, :, 0]
        visible = ((grid.abs() <= 1).all(-1) & (eye_z > 0) &
                   ((sampled[2] < min_alpha) | (eye_z <= sampled[1] * (1 + occlusion_tolerance))))
        matched = visible & (sampled[0] >= min_error)
        scores += sampled[0] * matched
        colors += sampled[3:].T * matched[:, None]
        support += matched.long()
    scores /= support.clamp_min(1)
    colors /= support.clamp_min(1)[:, None]
    return scores, colors, support


def select_spaced_births(points, scales, scores, support, minimum_views, limit, separation):
    """Greedy radius separation over a small CPU proposal pool."""
    eligible = torch.where((support >= minimum_views) & torch.isfinite(points).all(-1) &
                           torch.isfinite(scales) & (scales > 0) & (scores > 0))[0]
    rank = scores[eligible] * support[eligible].to(scores).sqrt()
    order = eligible[torch.argsort(rank, descending=True, stable=True)].cpu().tolist()
    xyz, radii = points.detach().cpu().numpy(), scales.detach().cpu().numpy()
    chosen = []
    for row in order:
        if chosen:
            distance = np.linalg.norm(xyz[chosen] - xyz[row], axis=-1)
            if np.any(distance < separation * np.maximum(radii[chosen], radii[row])):
                continue
        chosen.append(row)
        if len(chosen) == limit:
            break
    return torch.tensor(chosen, device=points.device, dtype=torch.long)


@torch.no_grad()
def collect_residual_births(module):
    cfg, model = module.residual_motion.config, module.gaussians
    names = list(cfg.birth_residual_camera_names)
    dataset = module.datamodule.get_train_dataset()
    indices = module.datamodule._camera_indices(dataset, names)
    configured = getattr(module.datamodule.config, 'training_camera_names', None)
    validation = set(getattr(module.datamodule.config, 'validation_camera_names', None) or [])
    allowed = set(configured) if configured is not None else {item.path.name for item in dataset.items}
    if not set(names) <= allowed or set(names) & validation:
        raise ValueError('Residual birth cameras must be optimization training cameras')
    original = module.renderer.gaussians
    module.trainer.synchronize()
    started = time.perf_counter()
    packages, positions, scales, observed = [], [], [], []
    try:
        module.renderer.gaussians = model
        xyz = model.get_xyz
        homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
        with torch.random.fork_rng(devices=[]):
            for view in module.datamodule.train_dataloader(random=False, indices=indices):
                view = view.to(module.device)
                observed.append(view.path.name)
                image = module.renderer(view, module.background)['render'].clamp(0., 1.)
                eye_z = (homogeneous @ view.world_view_transform)[:, 2]
                depth_colors = torch.stack((eye_z.clamp_min(0), torch.ones_like(eye_z),
                                            torch.zeros_like(eye_z)), dim=-1)
                depth_image = module.renderer(view, torch.zeros_like(module.background),
                                               override_color=depth_colors)['render']
                alpha = depth_image[1]
                depth = depth_image[0] / alpha.clamp_min(1e-6)
                error = (view.image - image).abs().mean(0)
                packages.append((view, error, depth, alpha))
                pixels = grid_error_peaks(error, cfg.birth_residual_cell_size,
                    cfg.birth_residual_proposals_per_view, cfg.birth_residual_min_error)
                y, x = pixels.long().unbind(-1)[::-1]
                seed_depth = depth[y, x]
                valid = (alpha[y, x] >= cfg.birth_residual_min_alpha) & torch.isfinite(seed_depth) & (seed_depth > 0)
                pixels, seed_depth = pixels[valid], seed_depth[valid]
                for ratio in cfg.birth_residual_depth_ratios:
                    hypothesis_depth = seed_depth * ratio
                    positions.append(backproject_pixels(pixels, hypothesis_depth, view))
                    focal_x = view.image.shape[-1] / (2 * math.tan(view.fovx / 2))
                    scales.append(hypothesis_depth / focal_x * cfg.birth_residual_radius_pixels)
    finally:
        module.renderer.gaussians = original
    if observed != names:
        raise ValueError('Residual birth probe camera order differs')
    points, radii = torch.cat(positions), torch.cat(scales)
    scores, colors, support = multiview_birth_evidence(points, packages,
        cfg.birth_residual_min_error, cfg.birth_residual_min_alpha, cfg.birth_residual_occlusion_tolerance)
    chosen = select_spaced_births(points, radii, scores, support, cfg.birth_residual_min_views,
        cfg.birth_residual_max_candidates, cfg.birth_residual_separation)
    points, radii, colors = points[chosen], radii[chosen], colors[chosen]
    count = len(chosen)
    opacity_logit = math.log(cfg.birth_residual_initial_opacity / (1 - cfg.birth_residual_initial_opacity))
    rotation = points.new_zeros(count, 4)
    rotation[:, 0] = 1
    raw = dict(xyz=points, sh_dc=((colors - .5) / 0.28209479177387814)[:, None],
               sh_rest=model._features_rest.new_zeros((count,) + model._features_rest.shape[1:]),
               opacity=points.new_full((count, 1), opacity_logit),
               logscale=radii.clamp_min(torch.finfo(points.dtype).tiny).log()[:, None].expand(-1, 3).clone(),
               rotation=rotation)
    module.trainer.synchronize()
    return raw, dict(enabled=True, cameras=observed, extra_renders=2 * len(observed),
        hypothesis_count=len(scores), multi_view_supported=int((support >= cfg.birth_residual_min_views).sum()),
        added=count, minimum_views=cfg.birth_residual_min_views, minimum_error=cfg.birth_residual_min_error,
        support_counts=support[chosen].cpu().tolist(), seconds=time.perf_counter() - started,
        semantics='Pixel residual peaks use rendered eye-depth hypotheses. Other optimization views '
                  'must support the position with residual and visibility evidence; no test camera is used. '
                  'Render-depth is an approximation, not measured scene depth or triangulation.')
