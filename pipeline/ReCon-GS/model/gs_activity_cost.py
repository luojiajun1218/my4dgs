"""Geometry-grounded GS activity cost; integration remains opt-in."""
import torch
from torch.nn import functional as F

from libgs.utils.general import quaternion_multiply

from .region_rigid import apply_region_rigid, rotation_vector_quaternion
from .residual_motion import inverse_unit_quaternion, shortest_rotation_vector


@torch.no_grad()
def candidate_activity_support(base_xyz, base_rotation, layout, regions, local,
                               translation_threshold, rotation_threshold):
    """Return bool [GS, 4] for three SHARE endpoints and the GS-local endpoint.

    Each endpoint contains its full ancestor SPLIT chain. Descendant KEEP
    returns the absolute previous pose, so it contributes zero support. Motion
    tensors are detached: this support cost selects gates, not motion magnitude.
    Thresholds are world distance and radians, respectively.
    """
    index = layout['index']
    pivots = [layout['pivots'][level][index[level]] for level in range(3)]
    identity = base_xyz.new_zeros(len(base_xyz), 4)
    identity[:, 0] = 1
    base_unit = F.normalize(base_rotation, dim=-1)
    base_inverse = inverse_unit_quaternion(base_unit)
    xyz, rotation = base_xyz, base_rotation
    support = []

    def changed(candidate_xyz, candidate_rotation):
        distance = (candidate_xyz - base_xyz).norm(dim=-1)
        relative = quaternion_multiply(F.normalize(candidate_rotation, dim=-1), base_inverse)
        angle = shortest_rotation_vector(relative).norm(dim=-1)
        return (distance > translation_threshold) | (angle > rotation_threshold)

    for level in range(3):
        selected = regions[level][index[level]]
        xyz, rotation = apply_region_rigid(xyz, rotation, pivots[level],
                                          selected[:, :3], selected[:, 3:])
        support.append(changed(xyz, rotation))
        for child in range(level + 1, 3):
            pivots[child], _ = apply_region_rigid(pivots[child], identity, pivots[level],
                                                selected[:, :3], selected[:, 3:])
    xyz = xyz + local[:, :3]
    rotation = quaternion_multiply(rotation_vector_quaternion(local[:, 3:]), rotation)
    support.append(changed(xyz, rotation))
    return torch.stack(support, dim=-1)


def expected_activity_fraction(probabilities, index, support):
    """Expected changed-GS fraction over terminal choices; gradient to gates only.

    probabilities has three [regions, KEEP/SHARE/SPLIT] tensors; index is
    integer [3, GS]; support is detached bool [GS, 4]. Every GS is counted once
    at its terminal choice, including ancestor cancellation and local residuals.
    """
    if index.ndim != 2 or index.shape[0] != 3 or index.shape[1] < 1:
        raise ValueError('Activity cost requires nonempty [3, GS] owner indices')
    if support.shape != (index.shape[1], 4):
        raise ValueError('Activity support must have shape [GS, 4]')
    reach = probabilities[0].new_ones(index.shape[1])
    expected = torch.zeros_like(reach)
    for level in range(3):
        p = probabilities[level][index[level]]
        expected = expected + reach * p[:, 1] * support[:, level].detach().to(p)
        reach = reach * p[:, 2]
    expected = expected + reach * support[:, 3].detach().to(reach)
    return expected.mean()
