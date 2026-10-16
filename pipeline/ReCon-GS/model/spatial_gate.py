"""Stable spatial candidates and differentiable decisions to subdivide them."""

from dataclasses import dataclass
import math

import torch
import numpy as np


@dataclass
class SpatialGateConfig:
    enabled: bool = True
    max_depth: int = 3
    min_region_size: int = 16
    learning_rate: float = 0.15
    initial_logit: float = -0.2
    start_step: int = 20
    gate_start_step: int = 40
    freeze_step: int = 60
    temperature: float = 1.0
    learn_region_motion: bool = False  # No unrestricted image updates during bootstrap.
    image_fit_during_gate: bool = True

    def validate(self, motion_steps):
        if not isinstance(self.enabled, bool):
            raise ValueError('spatial gate enabled must be boolean')
        if not isinstance(self.learn_region_motion, bool):
            raise ValueError('learn_region_motion must be boolean')
        if self.image_fit_during_gate is not True:
            raise ValueError('Frozen-motion gate checkpoints are replay-only')
        if not 2 <= self.max_depth <= 5 or self.min_region_size < 2:
            raise ValueError('Invalid spatial depth or minimum region size')
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0 and
                math.isfinite(self.temperature) and self.temperature > 0 and
                math.isfinite(self.initial_logit)):
            raise ValueError('Invalid spatial gate optimization settings')
        if self.enabled and not 1 <= self.start_step <= self.gate_start_step < self.freeze_step < motion_steps:
            raise ValueError('Gate fitting must leave time for motion fitting after hard selection')


def spatial_layout(xyz, depth, return_centers=False):
    """Create an octree in reference coordinates, with the first eight cells active.

    Candidate descendants exist only as a training dictionary. The frame's
    hard gates choose a disjoint frontier; motion never accumulates along paths.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError('Spatial layout requires nonempty [N,3] positions')
    # Extreme background points must not collapse almost the whole scene into
    # one octant. Split occupied cells at their coordinate medians, once in the
    # reference frame. The split planes remain fixed; GS can be routed again
    # from the previous decoded positions without transmitting owner edits.
    points = xyz.detach().cpu().numpy()
    split_count = sum(8 ** level for level in range(depth))
    centers = np.zeros((split_count, 3), dtype=np.float32)
    lower, upper = np.zeros_like(centers), np.zeros_like(centers)
    lower[0], upper[0] = points.min(0), points.max(0)
    prior_codes = np.zeros(len(points), dtype=np.int64)
    paths, parent_parts = [], [torch.tensor([-1], device=xyz.device)]
    offset, previous_offset = 1, 0
    for level in range(1, depth + 1):
        octants = np.zeros(len(points), dtype=np.int64)
        occupied = set()
        for parent in np.unique(prior_codes):
            rows = np.flatnonzero(prior_codes == parent)
            center = np.median(points[rows], axis=0)
            centers[previous_offset + parent] = center
            occupied.add(int(parent))
            side = points[rows] >= center
            octants[rows] = side[:, 0] + 2 * side[:, 1] + 4 * side[:, 2]
        for parent in range(8 ** (level - 1)):
            node = previous_offset + parent
            if parent not in occupied:
                centers[node] = .5 * lower[node] + .5 * upper[node]
            if level < depth:
                for octant in range(8):
                    child = offset + 8 * parent + octant
                    lower[child], upper[child] = lower[node], upper[node]
                    for axis in range(3):
                        if (octant >> axis) & 1:
                            lower[child, axis] = centers[node, axis]
                        else:
                            upper[child, axis] = centers[node, axis]
        codes_for_points = 8 * prior_codes + octants
        paths.append(torch.from_numpy(codes_for_points.copy()).to(xyz.device) + offset)
        codes = torch.arange(8 ** level, device=xyz.device)
        if level == 1:
            parents = torch.zeros_like(codes)
        else:
            parents = previous_offset + codes // 8
        parent_parts.append(parents)
        previous_offset, offset = offset, offset + 8 ** level
        prior_codes = codes_for_points
    parents = torch.cat(parent_parts).long()
    paths = torch.stack(paths, -1)
    opened = torch.zeros(len(parents), dtype=torch.bool, device=xyz.device)
    opened[0] = True
    result = paths, parents, opened
    return (*result, torch.from_numpy(centers)) if return_centers else result


def spatial_route(xyz, centers, depth):
    """CPU routing from shared float32 decoded positions and static planes.

    Equality belongs to the upper cell, matching reference construction.
    CPU evaluation on both sides avoids GPU boundary-rounding differences.
    """
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or
            centers.shape != (sum(8 ** level for level in range(depth)), 3) or
            not bool(torch.isfinite(xyz).all()) or
            not bool(torch.isfinite(centers).all())):
        raise ValueError('Invalid spatial routing positions or split planes')
    points = xyz.detach().cpu().float().numpy()
    planes = centers.detach().cpu().float().numpy()
    parent = np.zeros(len(points), dtype=np.int64)
    codes, offset, paths = parent.copy(), 1, []
    for level in range(depth):
        side = points >= planes[parent]
        octant = side[:, 0].astype(np.int64) + 2 * side[:, 1] + 4 * side[:, 2]
        codes = 8 * codes + octant
        parent = offset + codes
        paths.append(parent.copy())
        offset += 8 ** (level + 1)
    return torch.from_numpy(np.stack(paths, axis=1))


def spatial_owner(paths, opened):
    """Select exactly one node on each GS path, stopping at the first closed gate."""
    owner = paths[:, 0].clone()
    reach = torch.ones(len(paths), dtype=torch.bool, device=paths.device)
    for level in range(1, paths.shape[1]):
        reach = reach & opened[paths[:, level - 1]]
        owner = torch.where(reach, paths[:, level], owner)
    return owner


def compose_spatial_pose(xyz, rotation, owner, region_t, region_r, gs_t, gs_r):
    """Shared absolute increments and GS residuals; rotations affect orientation only."""
    from .rigid_residual_tree import rotation_vector_quaternion, quaternion_multiply
    increment = rotation_vector_quaternion(region_r[owner] + gs_r)
    return xyz + region_t[owner] + gs_t, quaternion_multiply(increment, rotation)
