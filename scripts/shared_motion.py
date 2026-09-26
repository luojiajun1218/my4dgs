"""Coarse-to-fine, region-shared motion for subsequent video frames.

The spatial groups are built from the first-frame octree coordinates.  They are
material groups: their membership does not change when the Gaussians move.
"""

import math

import torch
from torch import nn


def quaternion_multiply(left, right):
    """Hamilton product for the wxyz quaternion convention used by S²GS."""
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def rotation_vector_to_quaternion(vector):
    """Exponential map, stable at zero rotation."""
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scalar = torch.cos(0.5 * angle)
    imaginary = 0.5 * torch.sinc(angle / (2.0 * math.pi)) * vector
    return torch.cat((scalar, imaginary), dim=-1)


class SharedMotion(nn.Module):
    """One shared translation/rotation per region at each octree resolution.

    The coarsest level is always available.  Finer levels have differentiable
    gates and a sparsity penalty, so they can be activated by image gradients
    during the first training stage without a discrete trial/rollback search.
    """

    def __init__(self, anchors, root_indices, init_pos, voxel_size, fork,
                 levels=6, spatial_levels=None, lr=5e-4, gate_lr=5e-2,
                 lambda_deform=1.0, lambda_sparse=1e-3):
        super().__init__()
        if levels < 1:
            raise ValueError("SharedMotion requires at least one level")
        self.levels = levels
        self.lambda_deform = lambda_deform
        self.lambda_sparse = lambda_sparse
        self.translations = nn.ParameterList()
        self.rotations = nn.ParameterList()
        self.gate_logits = nn.ParameterList()
        self.node_counts = []
        # A shorter hierarchy uses the coarsest levels through the S²GS root.
        # The four-level prefix [0, 1, 2, 3] was stronger and smaller than
        # the spread-out [0, 2, 3, 5] in the 60-frame Flame Steak preflight.
        source_levels = spatial_levels or levels
        reference_levels = max(6, levels, source_levels)
        root_level = max(0, reference_levels - source_levels)
        self.spatial_level_indices = list(range(levels))

        for level in range(levels):
            spatial_level = self.spatial_level_indices[level]
            if spatial_level == root_level:
                coords, point_to_node = torch.unique(
                    root_indices.detach().reshape(-1), sorted=True,
                    return_inverse=True,
                )
            else:
                cell_size = voxel_size * (
                    float(fork) ** (root_level - spatial_level))
                cell = torch.floor((anchors.detach() - init_pos) / cell_size).long()
                coords, point_to_node = torch.unique(
                    cell, dim=0, sorted=True, return_inverse=True,
                )
            self.register_buffer(f"coords_{level}", coords)
            self.register_buffer(f"point_to_node_{level}", point_to_node)
            n_nodes = coords.shape[0]
            self.node_counts.append(n_nodes)
            self.translations.append(nn.Parameter(anchors.new_zeros(n_nodes, 3)))
            self.rotations.append(nn.Parameter(anchors.new_zeros(n_nodes, 3)))
            if level > 0:
                self.gate_logits.append(nn.Parameter(anchors.new_full((n_nodes,), 1.0)))

        self.optimizer = torch.optim.Adam([
            {"params": list(self.translations) + list(self.rotations), "lr": lr},
            {"params": list(self.gate_logits), "lr": gate_lr},
        ], eps=1e-15)

    def _effective_node_motion(self, level):
        translation = self.translations[level]
        rotation = self.rotations[level]
        if level > 0:
            gate = self._node_gate(level).unsqueeze(-1)
            translation = gate * translation
            rotation = gate * rotation
        return translation, rotation

    def _node_gate(self, level):
        probability = torch.sigmoid(self.gate_logits[level - 1])
        forward_value = torch.clamp(2.0 * probability - 1.0, 0.0, 1.0)
        # Exact zero for closed nodes, but a straight-through gradient lets a
        # currently closed region reopen if image evidence becomes strong.
        return (forward_value - probability).detach() + probability

    def forward(self):
        n_points = self.point_to_node_0.numel()
        translation = self.translations[0].new_zeros(n_points, 3)
        quaternion = self.translations[0].new_zeros(n_points, 4)
        quaternion[:, 0] = 1.0
        for level in range(self.levels):
            node_translation, node_rotation = self._effective_node_motion(level)
            index = getattr(self, f"point_to_node_{level}")
            translation = translation + node_translation[index]
            local_quaternion = rotation_vector_to_quaternion(node_rotation[index])
            quaternion = quaternion_multiply(local_quaternion, quaternion)
        return translation, quaternion

    def regularization(self):
        """ReCon-style deformation penalty plus sparse fine-level activation."""
        penalty = self.translations[0].abs().mean() + self.rotations[0].abs().mean()
        for level in range(1, self.levels):
            gate = self._node_gate(level)
            probability = torch.sigmoid(self.gate_logits[level - 1])
            magnitude = (self.translations[level].abs().mean(dim=-1)
                         + self.rotations[level].abs().mean(dim=-1))
            penalty = penalty + (gate * magnitude).mean()
            penalty = penalty + self.lambda_sparse * probability.mean()
        return self.lambda_deform * penalty

    @torch.no_grad()
    def storage_bits(self, threshold=0.0):
        """Conservative per-frame estimate for active float32 node transforms."""
        bits = 0
        for level in range(self.levels):
            translation, rotation = self._effective_node_motion(level)
            active = ((torch.linalg.vector_norm(translation, dim=-1) > threshold)
                      | (torch.linalg.vector_norm(rotation, dim=-1) > threshold))
            n_active = int(active.sum().item())
            index_bits = max(1, math.ceil(math.log2(max(2, self.node_counts[level]))))
            bits += n_active * (6 * 32 + index_bits)
            bits += self.node_counts[level]  # active-node mask
        return bits

    @torch.no_grad()
    def frame_payload(self, threshold=0.0):
        levels = []
        for level in range(self.levels):
            translation, rotation = self._effective_node_motion(level)
            active = ((torch.linalg.vector_norm(translation, dim=-1) > threshold)
                      | (torch.linalg.vector_norm(rotation, dim=-1) > threshold))
            levels.append({
                "level": level,
                "spatial_level": self.spatial_level_indices[level],
                "coordinates": getattr(self, f"coords_{level}")[active].cpu(),
                "translation": translation[active].cpu(),
                "rotation_vector": rotation[active].cpu(),
            })
        return {"levels": levels, "estimated_bits": self.storage_bits(threshold)}
