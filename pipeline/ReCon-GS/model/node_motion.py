"""Compose hierarchy motion on nodes, then apply one transform per Gaussian."""

import torch
from libgs.utils.general import quaternion_multiply

from .region_rigid import rotation_vector_quaternion


def rotation_delta(quaternion, vector):
    cross = torch.cross(quaternion[:, 1:], vector, dim=-1)
    return 2 * (quaternion[:, :1] * cross +
                torch.cross(quaternion[:, 1:], cross, dim=-1))


def terminal_node_rows(layout, paths, *, selected_only, levels=None):
    """Return flattened node IDs of the last existing/reachable node per GS."""
    levels = len(paths) if levels is None else levels
    index, counts = layout['index'], layout['counts']
    terminal = index[0].clone()
    offset = counts[0]
    for level in range(1, levels):
        owner = index[level]
        valid = owner >= 0
        if selected_only and counts[level]:
            valid = valid & (paths[level][owner.clamp_min(0)] > 0)
        terminal = torch.where(valid, owner + offset, terminal)
        offset += counts[level]
    return terminal


def node_conditional_transform(base_xyz, base_rotation, layout, regions, paths,
                               *, selected_only=False, terminal=None,
                               active_rows=None):
    """Compose world rotations and pivot displacements; inputs are float32/64.

    Joint STE training evaluates every node, including closed candidates. Fixed
    choices may omit unreachable nodes. The final exact-zero correction retains
    raw XYZ/quaternions while the differentiable expression keeps opening grads.
    """
    levels = len(regions)
    rotations, displacements, active = [], [], []
    for level, motion in enumerate(regions):
        count = len(motion)
        rows = (active_rows[level] if active_rows is not None else
                torch.where(paths[level] > 0)[0] if selected_only else
                torch.arange(count, device=base_xyz.device))
        values = motion[rows] * paths[level][rows, None]
        local_q = rotation_vector_quaternion(values[:, 3:])
        own_active = values.detach().ne(0).any(-1)
        if level == 0:
            q, delta, changed = local_q, values[:, :3], own_active
        else:
            parent = layout['parents'][level - 1][rows]
            parent_q = rotations[level - 1][parent]
            relative_center = (layout['pivots'][level][rows] -
                               layout['pivots'][level - 1][parent])
            delta = (rotation_delta(parent_q, relative_center) +
                     displacements[level - 1][parent] + values[:, :3])
            q = quaternion_multiply(local_q, parent_q)
            changed = active[level - 1][parent] | own_active
        full_q = base_xyz.new_zeros((count, 4))
        full_q[:, 0] = 1
        full_delta = base_xyz.new_zeros((count, 3))
        full_active = torch.zeros(count, device=base_xyz.device, dtype=torch.bool)
        rotations.append(full_q.index_copy(0, rows, q))
        displacements.append(full_delta.index_copy(0, rows, delta))
        active.append(full_active.index_copy(0, rows, changed))
    if terminal is None:
        terminal = terminal_node_rows(layout, paths, selected_only=selected_only,
                                      levels=levels)
    q = torch.cat(rotations, dim=0)[terminal]
    center = torch.cat(layout['pivots'][:levels], dim=0)[terminal]
    delta = torch.cat(displacements, dim=0)[terminal]
    changed = torch.cat(active, dim=0)[terminal]
    xyz = base_xyz + rotation_delta(q, base_xyz - center) + delta
    rotation = quaternion_multiply(q, base_rotation)
    exact_xyz = torch.where(changed[:, None], xyz, base_xyz)
    exact_rotation = torch.where(changed[:, None], rotation, base_rotation)
    return (exact_xyz.detach() + (xyz - xyz.detach()),
            exact_rotation.detach() + (rotation - rotation.detach()))
