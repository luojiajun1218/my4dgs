"""World-space rigid motion; quaternions use scalar-first (w, x, y, z)."""
import torch
from torch import Tensor
from libgs.utils.general import quaternion_multiply


def rotation_vector_quaternion(vector: Tensor) -> Tensor:
    angle = vector.norm(dim=-1, keepdim=True)
    # sinc has a finite derivative at identity; no tunable small-angle cutoff.
    xyz = vector * (torch.sinc(angle / (2 * torch.pi)) / 2)
    return torch.cat([torch.cos(angle / 2), xyz], dim=-1)


def apply_region_rigid(xyz: Tensor, orientation: Tensor, center: Tensor,
                       translation: Tensor, rotation_vector: Tensor):
    """All inputs are per-GS; return centers and world-rotated orientations."""
    q = rotation_vector_quaternion(rotation_vector)
    local = xyz - center
    cross = torch.cross(q[:, 1:], local, dim=-1)
    rotated = local + 2 * (q[:, :1] * cross + torch.cross(q[:, 1:], cross, dim=-1))
    return rotated + center + translation, quaternion_multiply(q, orientation)
