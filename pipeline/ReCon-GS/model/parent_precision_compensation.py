"""Offline geometry helpers for parent-precision compensation audits.

This module is not called by the training or export pipeline.  It rewrites the
immediate-child residuals of an already-open binary node so that changing the
parent precision preserves the source decoded child cumulative poses before the
child residuals are quantized again.
"""

import torch

from libgs.utils.general import quaternion_multiply

from .node_motion import rotation_delta
from .region_rigid import rotation_vector_quaternion
from .residual_motion import inverse_unit_quaternion, shortest_rotation_vector


def freeze_hard_choices(motion, step):
    """Freeze exactly the gate and precision choices visible at ``step``."""
    gates = [value.detach().bool().clone() for value in motion.refinement_masks(step)]
    precision = [value.detach().argmax(-1).clone()
                 for value in motion._precision_weights(step, differentiable=False)]
    motion.hard_masks = dict(refine=gates, precision=precision)
    return motion.hard_masks


def set_frozen_node_action(motion, level, node, *, opened, parent_precision_index=None):
    """Set one frozen internal-node gate and, optionally, its precision."""
    if motion.hard_masks is None or 'precision' not in motion.hard_masks:
        raise ValueError('Node action requires explicitly frozen hard choices')
    if not isinstance(opened, bool):
        raise ValueError('opened must be bool')
    if level < 0 or level >= len(motion.refine_logits):
        raise ValueError('Node action requires an internal parent')
    if node < 0 or node >= motion.counts[level]:
        raise ValueError('Node action parent is outside the hierarchy')
    position = motion._refine_position(level, node)
    motion.hard_masks['refine'][level][position] = opened
    if parent_precision_index is not None:
        choices = motion.hard_masks['precision'][level]
        if (isinstance(parent_precision_index, bool) or
                not isinstance(parent_precision_index, int) or
                parent_precision_index < 0 or
                parent_precision_index >= len(motion.config.precision_bits)):
            raise ValueError('Parent precision index is out of range')
        choices[node] = parent_precision_index
    return position


def cumulative_node_poses(layout, regions, paths):
    """Return cumulative scalar-first quaternions and pivot displacements."""
    rotations, displacements = [], []
    for level, motion in enumerate(regions):
        values = motion * paths[level][:, None]
        local_rotation = rotation_vector_quaternion(values[:, 3:])
        if level == 0:
            rotation = local_rotation
            displacement = values[:, :3]
        else:
            parent = layout['parents'][level - 1]
            parent_rotation = rotations[level - 1][parent]
            offset = layout['pivots'][level] - layout['pivots'][level - 1][parent]
            displacement = (rotation_delta(parent_rotation, offset) +
                            displacements[level - 1][parent] + values[:, :3])
            rotation = quaternion_multiply(local_rotation, parent_rotation)
        rotations.append(rotation)
        displacements.append(displacement)
    return rotations, displacements


def immediate_child_coverage(layout, level, node):
    """Check that immediate children partition every GS owned by the parent."""
    if level < 0 or level + 1 >= len(layout['counts']):
        raise ValueError('Parent has no child level')
    if node < 0 or node >= layout['counts'][level]:
        raise ValueError('Parent node is outside the layout')
    children = torch.where(layout['parents'][level] == node)[0]
    if len(children) != 2:
        raise ValueError('Compensation requires one binary parent with two children')
    parent_members = layout['index'][level] == node
    child_members = torch.zeros_like(parent_members)
    for child in children:
        child_members |= layout['index'][level + 1] == child
    if not torch.equal(parent_members, child_members):
        raise ValueError('Immediate children do not exactly cover the parent GS rows')
    return children, parent_members


def quantize_rows(raw, steps, bits):
    """Apply the production round-and-clamp rule to a batch of six-vectors."""
    if raw.ndim != 2 or raw.shape[-1] != 6 or steps.shape != raw.shape:
        raise ValueError('raw and steps must be matching [N, 6] tensors')
    bits = torch.as_tensor(bits, device=raw.device, dtype=raw.dtype).reshape(-1)
    if len(bits) != len(raw):
        raise ValueError('bits must contain one width per row')
    limit = torch.pow(raw.new_tensor(2.), bits - 1)
    rounded = torch.round(raw / steps)
    symbols = torch.minimum(torch.maximum(rounded, -limit[:, None]),
                            limit[:, None] - 1)
    return symbols * steps, symbols.to(torch.int64), (rounded != symbols).any(-1)


def quaternion_angle_error(left, right):
    """Shortest angular difference in radians, invariant to quaternion sign."""
    relative = quaternion_multiply(
        left.detach().to(torch.float64),
        inverse_unit_quaternion(right.detach().to(torch.float64)))
    return 2 * torch.atan2(relative[:, 1:].norm(dim=-1), relative[:, :1].abs().flatten())


@torch.no_grad()
def rebase_immediate_children(motion, step, level, node, parent_precision_index):
    """Change one open parent's bit choice and analytically rebase its children.

    The returned continuous check is evaluated before the rebased child values
    are quantized.  The motion parameters receive those continuous values; the
    normal production quantizer subsequently rounds them at each child's
    unchanged hard precision.
    """
    if motion.hard_masks is None or 'precision' not in motion.hard_masks:
        raise ValueError('Compensation requires explicitly frozen hard choices')
    if motion.stage(step) not in {'gate_refine', 'incremental'}:
        raise ValueError('Compensation step must consume frozen hard choices')
    if level < 0 or level >= len(motion.refine_logits):
        raise ValueError('Compensation requires an internal parent node')
    if node < 0 or node >= motion.counts[level]:
        raise ValueError('Parent node is outside the hierarchy')
    gates = motion.hard_masks['refine']
    paths = motion._paths(gates)
    occupancy = motion._occupancy_values()
    if not bool(paths[level][node]) or not bool(occupancy[level][node]):
        raise ValueError('Compensation requires a reachable occupied parent')
    gate_position = motion._refine_position(level, node)
    if not bool(gates[level][gate_position]):
        raise ValueError('STOP parent cannot be compensated through unreachable children')
    structural_children, parent_members = immediate_child_coverage(
        motion.active_layout, level, node)
    children = structural_children[occupancy[level + 1][structural_children]]
    if not len(children):
        raise ValueError('Reachable occupied parent has no occupied immediate child')

    source_regions = [value.detach().clone()
                      for value in motion._region_motion(step, hard=True)]
    source_rotation, source_displacement = cumulative_node_poses(
        motion.active_layout, source_regions, paths)
    original_precision_index = int(motion.hard_masks['precision'][level][node])
    if parent_precision_index == original_precision_index:
        raise ValueError('Alternate parent precision must differ from the source')
    if parent_precision_index < 0 or parent_precision_index >= len(motion.config.precision_bits):
        raise ValueError('Alternate parent precision index is out of range')

    motion.hard_masks['precision'][level][node] = int(parent_precision_index)
    naive_regions = [value.detach().clone()
                     for value in motion._region_motion(step, hard=True)]
    naive_rotation, naive_displacement = cumulative_node_poses(
        motion.active_layout, naive_regions, paths)
    new_parent_rotation = naive_rotation[level][node].expand(len(children), -1)
    new_parent_displacement = naive_displacement[level][node].expand(len(children), -1)
    child_offset = (motion.active_layout['pivots'][level + 1][children] -
                    motion.active_layout['pivots'][level][node])
    target_local_rotation = quaternion_multiply(
        source_rotation[level + 1][children],
        inverse_unit_quaternion(new_parent_rotation))
    target_rotation_vector = shortest_rotation_vector(target_local_rotation)
    target_translation = (source_displacement[level + 1][children] -
                          new_parent_displacement -
                          rotation_delta(new_parent_rotation, child_offset))

    continuous_regions = [value.clone() for value in naive_regions]
    continuous_regions[level + 1][children, :3] = target_translation
    continuous_regions[level + 1][children, 3:] = target_rotation_vector
    continuous_rotation, continuous_displacement = cumulative_node_poses(
        motion.active_layout, continuous_regions, paths)
    continuous_position_error = (
        continuous_displacement[level + 1][children] -
        source_displacement[level + 1][children]).norm(dim=-1)
    continuous_rotation_error = quaternion_angle_error(
        continuous_rotation[level + 1][children],
        source_rotation[level + 1][children])

    motion.region_translation[level + 1][children] = target_translation
    motion.region_rotation[level + 1][children] = target_rotation_vector
    return dict(children=children.detach().clone(),
                structural_children=structural_children.detach().clone(),
                parent_members=parent_members.detach().clone(),
                original_precision_index=original_precision_index,
                alternate_precision_index=int(parent_precision_index),
                target_translation=target_translation.detach().clone(),
                target_rotation_vector=target_rotation_vector.detach().clone(),
                continuous_position_error=continuous_position_error.detach().clone(),
                continuous_rotation_error=continuous_rotation_error.detach().clone())
