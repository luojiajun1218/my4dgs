"""Propose Gaussian births from residuals supported by several training views.

Camera tensors follow the row-vector convention used by libgs. These helpers do
not read held-out images and do not decide which points the codec retires.
"""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS


C0 = 0.28209479177387814
C1 = 0.4886025119029199


@dataclass
class MultiViewBirthConfig:
    candidates_per_view: int = 96
    depth_samples: int = 9
    depth_span_extent: float = .04
    min_support_views: int = 3
    min_support_fraction: float = .3
    error_quantile: float = .9
    min_error: float = 1e-5
    color_sigma: float = .2
    occlusion_tolerance_extent: float = .002
    initial_sigma_pixels: float = 2.
    initial_alpha: float = .1
    dedup_extent: float = .00075

    def validate(self):
        for name in ('candidates_per_view', 'depth_samples', 'min_support_views'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('depth_span_extent', 'min_error', 'color_sigma',
                     'occlusion_tolerance_extent', 'initial_sigma_pixels', 'dedup_extent'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.min_support_fraction <= 1:
            raise ValueError('min_support_fraction must be in (0, 1]')
        if not 0 < self.error_quantile < 1 or not 0 < self.initial_alpha < 1:
            raise ValueError('error_quantile and initial_alpha must be in (0, 1)')


@dataclass
class ViewEvidence:
    camera: object
    image: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    error: torch.Tensor
    threshold: torch.Tensor


def project_points(xyz, camera):
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=1)
    clip = homogeneous @ camera.full_proj_transform
    grid = clip[:, :2] / clip[:, 3:4].clamp_min(1e-8)
    depth = (homogeneous @ camera.world_view_transform)[:, 2]
    inside = (depth > 0) & (grid.abs() < 1).all(dim=1)
    return grid, depth, inside


def backproject_pixels(pixel, depth, camera):
    height, width = camera.image.shape[-2:]
    x, y = pixel.unbind(-1)
    x = ((x + .5) * (2 / width) - 1) * math.tan(camera.fovx / 2) * depth
    y = ((y + .5) * (2 / height) - 1) * math.tan(camera.fovy / 2) * depth
    local = torch.stack((x, y, depth, torch.ones_like(depth)), dim=-1)
    return (local @ torch.linalg.inv(camera.world_view_transform))[..., :3]


def sample_map(value, grid):
    if value.ndim == 2:
        value = value.unsqueeze(0)
    result = F.grid_sample(value.unsqueeze(0), grid[None, :, None],
                           mode='bilinear', padding_mode='zeros', align_corners=False)
    return result[0, :, :, 0].T


@torch.no_grad()
def collect_view_evidence(renderer, cameras, raw, background, config):
    config.validate()
    evidence = []
    homogeneous = torch.cat((raw['xyz'], torch.ones_like(raw['xyz'][:, :1])), dim=1)
    for camera in cameras:
        prediction = renderer(camera, background)['render']
        z = (homogeneous @ camera.world_view_transform)[:, 2]
        labels = torch.stack((z, torch.ones_like(z), torch.zeros_like(z)), dim=1)
        # These channels carry depth and alpha, so background contributes zero.
        accumulated = renderer(camera, torch.zeros_like(background), override_color=labels)['render']
        alpha = accumulated[1]
        depth = accumulated[0] / alpha.clamp_min(1e-8)
        error = (prediction - camera.image).square().mean(dim=0)
        error = F.avg_pool2d(error[None, None], 5, 1, 2)[0, 0]
        valid = (alpha > .5) & (depth > 0)
        threshold = (torch.quantile(error[valid], config.error_quantile)
                     if valid.any() else error.new_tensor(config.min_error))
        evidence.append(ViewEvidence(camera, camera.image, depth, alpha, error,
                                     threshold.clamp_min(config.min_error)))
    return evidence


@torch.no_grad()
def score_depth_candidates(xyz, seed_rgb, evidence, extent, config):
    """Return geometric/color support, including visibility and contradiction."""
    count = len(xyz)
    device = xyz.device
    visible_count = torch.zeros(count, device=device)
    supported_count = torch.zeros_like(visible_count)
    signal = torch.zeros_like(visible_count)
    support_weights, colors, scales, directions = [], [], [], []
    for item in evidence:
        grid, z, inside = project_points(xyz, item.camera)
        old_depth = sample_map(item.depth, grid)[:, 0]
        old_alpha = sample_map(item.alpha, grid)[:, 0]
        error = sample_map(item.error, grid)[:, 0]
        rgb = sample_map(item.image, grid)
        visible = inside & (old_alpha > .5) & (old_depth > 0)
        visible &= z <= old_depth + config.occlusion_tolerance_extent * extent
        difference = (rgb - seed_rgb).square().mean(dim=1)
        affinity = torch.exp(-difference / config.color_sigma ** 2)
        support = visible & (error >= item.threshold) & (affinity >= .2)
        weight = support.float() * affinity
        visible_count += visible.float()
        supported_count += support.float()
        signal += weight * (error / item.threshold).clamp_max(20)
        support_weights.append(weight)
        colors.append(rgb)
        width = item.image.shape[-1]
        scales.append(z * (2 * math.tan(item.camera.fovx / 2) / width))
        directions.append(F.normalize(xyz - item.camera.camera_center, dim=1))
    enough = (supported_count >= config.min_support_views)
    enough &= supported_count >= config.min_support_fraction * visible_count
    # Prefer agreement across cameras. Penalize projections into well-fit areas.
    scores = supported_count + signal / visible_count.clamp_min(1)
    scores -= .25 * (visible_count - supported_count)
    scores.masked_fill_(~enough, -torch.inf)
    return scores, dict(support_count=supported_count, visible_count=visible_count,
        weights=torch.stack(support_weights), colors=torch.stack(colors),
        scales=torch.stack(scales), directions=torch.stack(directions))


def sh1_basis(direction):
    x, y, z = direction.unbind(-1)
    return torch.stack((torch.full_like(x, C0), -C1*y, C1*z, -C1*x), dim=-1)


@torch.no_grad()
def fit_initial_sh1(colors, directions, weights):
    """Weighted multi-view fit with a ridge on directional appearance terms."""
    design = sh1_basis(directions).permute(1, 0, 2)
    color = colors.permute(1, 0, 2) - .5
    weight = weights.T
    gram = torch.einsum('nvi,nv,nvj->nij', design, weight, design)
    rhs = torch.einsum('nvi,nv,nvc->nic', design, weight, color)
    prior = torch.zeros_like(rhs)
    prior[:, 0] = (color * weight[..., None]).sum(1) / weight.sum(1).clamp_min(1e-8)[:, None] / C0
    ridge = design.new_tensor((1e-4, .03, .03, .03))
    gram += torch.diag(ridge)[None]
    rhs += ridge[None, :, None] * prior
    return torch.linalg.solve(gram, rhs)


@torch.no_grad()
def propose_multiview_births(renderer, cameras, raw, background, extent, config):
    """Choose a depth per residual ray, deduplicate, and initialize appearance."""
    config.validate()
    if len(cameras) < config.min_support_views:
        raise ValueError('Too few fitting cameras for multi-view birth support')
    evidence = collect_view_evidence(renderer, cameras, raw, background, config)
    proposals, seed_colors, offsets = [], [], []
    for item in evidence:
        height, width = item.image.shape[-2:]
        valid = (item.alpha > .5) & (item.depth > 0)
        error = item.error.masked_fill(~valid, 0)
        values, pixels = F.max_pool2d(error[None, None], 16, 16, return_indices=True)
        top = torch.topk(values.flatten(), min(config.candidates_per_view, values.numel())).indices
        top = top[values.flatten()[top] >= item.threshold]
        pixel = pixels.flatten()[top]
        if not len(pixel):
            continue
        y = torch.div(pixel, width, rounding_mode='floor')
        x = pixel % width
        offset = torch.linspace(-1, 1, config.depth_samples, device=error.device)
        z = item.depth[y, x, None] + offset * config.depth_span_extent * extent
        z = z.clamp_min(1e-5 * extent)
        xy = torch.stack((x, y), dim=1).float()[:, None].expand(-1, config.depth_samples, -1)
        proposals.append(backproject_pixels(xy, z, item.camera).reshape(-1, 3))
        seed_colors.append(item.image[:, y, x].T[:, None].expand(-1, config.depth_samples, -1).reshape(-1, 3))
        offsets.append(offset.expand(len(pixel), -1).reshape(-1))
    empty = {name: value[:0].clone() for name, value in raw.items()}
    if not proposals:
        return empty, torch.empty(0, device=raw['xyz'].device, dtype=torch.long), dict(proposed_rays=0, accepted=0)
    xyz, rgb = torch.cat(proposals), torch.cat(seed_colors)
    score, details = score_depth_candidates(xyz, rgb, evidence, extent, config)
    score -= .01 * torch.cat(offsets).abs()
    ray_scores = score.reshape(-1, config.depth_samples)
    best_score, best_depth = ray_scores.max(1)
    rows = torch.arange(len(best_depth), device=xyz.device) * config.depth_samples + best_depth
    rows = rows[torch.isfinite(best_score)]
    # Keep strongest evidence when more than one ray proposes the same cell.
    rows = rows[torch.argsort(score[rows], descending=True, stable=True)]
    cells = torch.round(xyz[rows] / (config.dedup_extent * extent)).long().cpu().tolist()
    unique, keep = set(), []
    for index, cell in enumerate(cells):
        cell = tuple(cell)
        if cell not in unique:
            unique.add(cell)
            keep.append(index)
    rows = rows[torch.as_tensor(keep, device=rows.device, dtype=torch.long)]
    report = dict(proposed_rays=len(ray_scores), tested_depths=len(xyz),
                  supported_rays=int(torch.isfinite(best_score).sum()), accepted=len(rows))
    if not len(rows):
        return empty, torch.empty(0, device=xyz.device, dtype=torch.long), report
    selected = xyz[rows]
    from scipy.spatial import cKDTree
    _, nearest = cKDTree(raw['xyz'].cpu().numpy()).query(selected.cpu().numpy())
    sources = torch.as_tensor(nearest, device=xyz.device, dtype=torch.long)
    born = {name: value[sources].clone() for name, value in raw.items()}
    born['xyz'] = selected
    weights = details['weights'][:, rows]
    coefficients = fit_initial_sh1(details['colors'][:, rows], details['directions'][:, rows], weights)
    born['sh_dc'], born['sh_rest'] = coefficients[:, :1], coefficients[:, 1:]
    sigma = (details['scales'][:, rows] * weights).sum(0) / weights.sum(0).clamp_min(1e-8)
    sigma = (sigma * config.initial_sigma_pixels).clamp(5e-5 * extent, .005 * extent)
    born['logscale'] = sigma[:, None].expand(-1, 3).log().contiguous()
    born['opacity'].fill_(math.log(config.initial_alpha / (1 - config.initial_alpha)))
    report.update(mean_support=float(details['support_count'][rows].mean()),
                  mean_visible=float(details['visible_count'][rows].mean()),
                  mean_depth_offset_extent=float(torch.cat(offsets)[rows].abs().mean()) * config.depth_span_extent)
    return born, sources, report


def validation_edit_gain(before, after, *, min_gain_db=0., max_ssim_drop=1e-4):
    """PSNR-aligned admission, using fitting-set validation cameras only."""
    gain = after['psnr'] - before['psnr']
    ssim_ok = ('ssim' not in before or 'ssim' not in after or
               after['ssim'] >= before['ssim'] - max_ssim_drop)
    return bool(math.isfinite(gain) and gain > min_gain_db and ssim_ok), gain
