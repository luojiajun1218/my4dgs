"""Read-only position-gradient evidence for ranking possible tree splits.

Gradients are image-loss derivatives at one unchanged rendered GS state, in
world coordinates. They rank proposals; they are not estimates of motion.
All selection budgets and support thresholds are supplied by the caller.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class PositionGradientEvidence:
    mean: torch.Tensor  # [N, 3], same floating dtype/device as input
    visible_views: torch.Tensor  # [N], int64
    supported: torch.Tensor  # [N], bool


@dataclass
class SplitRegionScores:
    score: torch.Tensor  # [R], mean of supported per-GS gradient norms
    total_gs: torch.Tensor  # [R], int64
    supported_gs: torch.Tensor  # [R], int64


def average_position_gradients(
    gradients: torch.Tensor,
    visible: torch.Tensor,
    *,
    min_visible_views: int,
) -> PositionGradientEvidence:
    """Average visible finite gradients before taking any vector norms.

    Inputs: gradients [V,N,3], visible [V,N]. Missing observations do not
    count as zero-gradient evidence. The caller must keep model state fixed
    across views and exclude rate/storage terms from the image loss.
    """
    if (gradients.ndim != 3 or gradients.shape[-1] != 3 or
            not gradients.is_floating_point() or visible.shape != gradients.shape[:2]):
        raise ValueError('Expected floating gradients [V,N,3] and visibility [V,N]')
    if visible.dtype != torch.bool or visible.device != gradients.device:
        raise ValueError('Visibility must be bool on the gradient device')
    if (isinstance(min_visible_views, bool) or
            not isinstance(min_visible_views, int) or min_visible_views < 1):
        raise ValueError('min_visible_views must be a positive integer')
    gradients = gradients.detach()
    valid = visible & torch.isfinite(gradients).all(-1)
    counts = valid.sum(0)
    total = torch.where(valid[..., None], gradients, 0).sum(0)
    mean = total / counts.clamp_min(1)[:, None]
    return PositionGradientEvidence(mean, counts, counts >= min_visible_views)


def score_split_regions(
    evidence: PositionGradientEvidence,
    owners: torch.Tensor,
    *,
    region_count: int,
) -> SplitRegionScores:
    """Mean of GS norms per region, never norm of the region mean.

    owners [N] are local region IDs; -1 denotes GS outside this candidate
    level. Empty/unsupported regions have score zero and retain support
    counts so the caller can exclude them from candidate selection.
    """
    mean, support = evidence.mean, evidence.supported
    if mean.ndim != 2 or mean.shape[-1] != 3 or support.shape != mean.shape[:1]:
        raise ValueError('Expected mean gradients [N,3] and support [N]')
    if support.dtype != torch.bool or support.device != mean.device:
        raise ValueError('Support must be bool on the gradient device')
    if (owners.shape != mean.shape[:1] or owners.dtype != torch.long or
            owners.device != mean.device):
        raise ValueError('Owners must be int64 [N] on the gradient device')
    if (isinstance(region_count, bool) or not isinstance(region_count, int) or
            region_count < 0):
        raise ValueError('region_count must be a nonnegative integer')
    if bool(((owners < -1) | (owners >= region_count)).any()):
        raise ValueError('Region owner is out of range')
    if not bool(torch.isfinite(mean[support]).all()):
        raise ValueError('Supported gradients must be finite')
    assigned = owners >= 0
    qualified = assigned & support
    total = torch.bincount(owners[assigned], minlength=region_count)
    count = torch.bincount(owners[qualified], minlength=region_count)
    summed = mean.new_zeros(region_count)
    summed.index_add_(0, owners[qualified], mean[qualified].norm(dim=-1))
    return SplitRegionScores(summed / count.clamp_min(1), total, count)


def select_split_regions(
    scores: SplitRegionScores,
    legal: torch.Tensor,
    *,
    top_k: int | None,
    candidate_fraction: float = 1.0,
) -> torch.Tensor:
    """Rank independently of neighboring merge/reparent proposals.

    legal [R] is supplied by the tree caller after checking depth, node
    budget, minimum child sizes and observation support. Stable ties use
    the lower region ID. Regions with no evidence or no signal are omitted.
    None removes the count cap; candidate_fraction scales with eligible size.
    """
    if (legal.shape != scores.score.shape or legal.dtype != torch.bool or
            legal.device != scores.score.device):
        raise ValueError('Legal-region mask must be bool [R] on the score device')
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
        raise ValueError('top_k must be None or a nonnegative integer')
    if (isinstance(candidate_fraction, bool) or not math.isfinite(candidate_fraction) or
            not 0 < candidate_fraction <= 1):
        raise ValueError('candidate_fraction must be in (0, 1]')
    eligible = (legal & (scores.supported_gs > 0) &
                torch.isfinite(scores.score) & (scores.score > 0))
    identifiers = torch.where(eligible)[0]
    order = torch.argsort(scores.score[identifiers], descending=True, stable=True)
    count = math.ceil(len(identifiers) * candidate_fraction)
    if top_k is not None:
        count = min(count, top_k)
    return identifiers[order[:count]]


def cluster_split_region(
    xyz: torch.Tensor,
    evidence: PositionGradientEvidence,
    *,
    min_child_gs: int,
    iterations: int,
    position_weight: float,
    gradient_weight: float,
) -> torch.Tensor | None:
    """Deterministic two-means on separately RMS-scaled position/gradient.

    Only supported observations fit the centroids. Unsupported GS are then
    assigned by spatial distance alone, never by invented zero gradients.
    Both clusters must have min_child_gs supported members. Labels [N] are
    canonicalized so the first supported GS belongs to child zero.
    """
    if (xyz.shape != evidence.mean.shape or xyz.ndim != 2 or xyz.shape[-1] != 3 or
            xyz.device != evidence.mean.device or not xyz.is_floating_point()):
        raise ValueError('Clustering expects floating XYZ [N,3] on the gradient device')
    for name, value in (('min_child_gs', min_child_gs), ('iterations', iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name, value in (('position_weight', position_weight), ('gradient_weight', gradient_weight)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError('Clustering positions must be finite')
    known = evidence.supported
    if int(known.sum()) < 2 * min_child_gs:
        return None
    points = xyz[known].detach().double()
    gradients = evidence.mean[known].detach().double()
    if not bool(torch.isfinite(gradients).all()):
        raise ValueError('Supported clustering gradients must be finite')
    def standardize(values):
        centered = values - values.mean(0)
        rms = centered.square().sum(-1).mean().sqrt()
        return centered / rms if float(rms) > 0 else torch.zeros_like(centered)
    features = torch.cat((position_weight * standardize(points),
                          gradient_weight * standardize(gradients)), -1)
    first = int(features.square().sum(-1).argmax())
    distances = (features - features[first]).square().sum(-1)
    second = int(distances.argmax())
    if float(distances[second]) == 0:
        return None
    centers = features[[first, second]].clone()
    previous = None
    for _ in range(iterations):
        labels = (features[:, None] - centers).square().sum(-1).argmin(-1)
        if not bool(torch.bincount(labels, minlength=2).all()):
            return None
        if previous is not None and torch.equal(labels, previous):
            break
        centers = torch.stack([features[labels == child].mean(0) for child in (0, 1)])
        previous = labels
    if int(torch.bincount(labels, minlength=2).min()) < min_child_gs:
        return None
    if int(labels[0]) == 1:
        labels = 1 - labels
    result = torch.empty(len(xyz), device=xyz.device, dtype=torch.long)
    result[known] = labels
    if bool((~known).any()):
        spatial_centers = torch.stack([points[labels == child].mean(0) for child in (0, 1)])
        result[~known] = (xyz[~known].double()[:, None] - spatial_centers).square().sum(-1).argmin(-1)
    return result
