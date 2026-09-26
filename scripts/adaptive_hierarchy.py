"""Adaptive OFF/UPDATE/SPLIT hierarchy for frame residuals."""

import math

import torch
from torch import nn
from torch.nn import functional as F

OFF = 0
UPDATE = 1
SPLIT = 2


class AdaptiveOffsetGate(nn.Module):
    """Sparse per-Gaussian offset residuals inside terminal UPDATE regions.

    The hierarchy decides which regions may move.  This gate only decides
    whether a Gaussian still needs its own offset after shared motion.  Gate
    parameters are allocated for UPDATE Gaussians, never for OFF regions.
    """

    def __init__(self, update_mask, initial_probs, lr, temperature,
                 lambda_l0, lambda_l2):
        super().__init__()
        update_mask = update_mask.flatten().bool()
        initial_probs = initial_probs.flatten()
        if update_mask.shape != initial_probs.shape:
            raise ValueError("Offset gate probabilities must match the GS mask")
        if temperature <= 0:
            raise ValueError("Offset gate temperature must be positive")
        self.register_buffer("eligible_indices", torch.nonzero(
            update_mask, as_tuple=False).flatten())
        self.initial_count = update_mask.numel()
        self.newborn_count = 0
        self.temperature = temperature
        self.lambda_l0 = lambda_l0
        self.lambda_l2 = lambda_l2
        probabilities = initial_probs[self.eligible_indices].clamp(1e-4, 1 - 1e-4)
        # Initial deterministic gate values match the supplied probabilities.
        self.logits = nn.Parameter(temperature * torch.logit(probabilities))
        self.optimizer = torch.optim.Adam([self.logits], lr=lr, eps=1e-15)

    @property
    def num_points(self):
        return self.initial_count + self.newborn_count

    def append_new_points(self, count):
        """Newborn Gaussians are active by construction and have no gate bit."""
        if count < 0:
            raise ValueError("Cannot remove points through gate append")
        self.newborn_count += count

    def _values(self, hard=True):
        soft = torch.sigmoid(self.logits / self.temperature)
        selected = torch.round(soft * 10.0) / 10.0 if hard else soft
        values = soft.new_zeros(self.initial_count).index_copy(
            0, self.eligible_indices, selected)
        if self.newborn_count:
            values = torch.cat((values, values.new_ones(self.newborn_count)))
        return values

    def active_mask(self):
        return self._values(hard=True) != 0

    def forward(self, residual):
        if residual.shape[0] != self.num_points:
            raise RuntimeError("Offset gate does not match Gaussian count")
        view_shape = [self.num_points] + [1] * (residual.dim() - 1)
        hard = self._values(hard=True).reshape(view_shape)
        if not self.training:
            return hard * residual
        soft = self._values(hard=False).reshape(view_shape)
        # Use the discrete gate for rendering, but retain a soft gradient for
        # the residual even when its current discrete gate is zero.  Otherwise
        # zero-initialized offset and gate would be unable to turn on together.
        soft_result = soft * residual
        return (hard * residual).detach() - soft_result.detach() + soft_result

    def regularization(self, residual):
        if residual.shape[0] != self.num_points:
            raise RuntimeError("Offset regularization does not match Gaussian count")
        selected = residual[self.eligible_indices]
        if selected.numel() == 0:
            return residual.new_zeros(())
        probability = torch.sigmoid(self.logits / self.temperature)
        magnitude = selected.reshape(selected.shape[0], -1).square().mean(dim=1)
        penalty = self.lambda_l0 + 0.5 * self.lambda_l2 * magnitude
        return (probability * penalty).mean()

    @torch.no_grad()
    def size_bits(self):
        """Estimate a binary pattern plus four bits per nonzero 0.1 gate."""
        gate = self._values(hard=True)[self.eligible_indices]
        total = gate.numel()
        if total == 0:
            return 0.0
        active = int((gate != 0).sum().item())
        inactive = total - active
        pattern = 0.0
        for count in (active, inactive):
            if count:
                pattern -= count * math.log2(count / total)
        return pattern + active * 4

    @torch.no_grad()
    def frame_payload(self):
        values = self._values(hard=True)
        active = torch.nonzero(values != 0, as_tuple=False).flatten()
        return {
            "active_indices": active.cpu(),
            "gate_values": values[active].cpu(),
            "eligible_gaussians": int(self.eligible_indices.numel()),
            "active_gaussians": int(active.numel()),
            "estimated_bits": self.size_bits(),
        }


def sample_pixel_map(pixel_xy, value_map, visible=None):
    """Sample a 3x3-smoothed image map at projected Gaussian centers."""
    height, width = value_map.shape[-2:]
    xy = pixel_xy.detach().round().long()
    valid = ((xy[:, 0] >= 0) & (xy[:, 0] < width)
             & (xy[:, 1] >= 0) & (xy[:, 1] < height))
    if visible is not None:
        valid &= visible.flatten().bool()
    smoothed = F.avg_pool2d(value_map[None, None], 3, stride=1,
                            padding=1)[0, 0]
    values = value_map.new_zeros(xy.shape[0])
    values[valid] = smoothed[xy[valid, 1], xy[valid, 0]]
    return values, valid


def project_gaussians(xyz, camera):
    homogeneous = torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
    clip = homogeneous @ camera.full_proj_transform
    ndc = clip[:, :2] / clip[:, 3:].clamp_min(1e-8)
    xy = torch.stack(((ndc[:, 0] + 1.0) * camera.image_width * 0.5,
                      (ndc[:, 1] + 1.0) * camera.image_height * 0.5), dim=-1)
    return xy, clip[:, 3] > 0


def quaternion_multiply(left, right):
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack((
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ), dim=-1)


def rotation_vector_to_quaternion(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    scalar = torch.cos(0.5 * angle)
    imaginary = 0.5 * torch.sinc(angle / (2.0 * math.pi)) * vector
    return torch.cat((scalar, imaginary), dim=-1)


def quaternion_to_rotation_vector(quaternion):
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1)
    quaternion = torch.where(
        quaternion[..., :1] < 0, -quaternion, quaternion)
    imaginary = quaternion[..., 1:]
    imaginary_norm = torch.linalg.vector_norm(
        imaginary, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(
        imaginary_norm, quaternion[..., :1].clamp_min(1e-8))
    scale = torch.where(
        imaginary_norm > 1e-8,
        angle / imaginary_norm.clamp_min(1e-8),
        torch.full_like(imaginary_norm, 2.0))
    return imaginary * scale


class AdaptiveHierarchyMotion(nn.Module):
    """Implicit spatial hierarchy with one structural decision per region.

    Parent/child relationships are inferred from per-level point-to-cell maps.
    During hierarchy fitting, children learn corrections on top of fitted
    parent motion.  Finalization folds each complete path into its terminal
    UPDATE region; rendering and storage therefore need no parent table or
    parent-to-child motion path.
    """

    def __init__(self, anchors, init_pos, voxel_size, fork, dynamic_scores,
                 dynamic_threshold, image_change_scores=None,
                 image_change_threshold=0.005, levels=4,
                 root_scale=2.0,
                 lr=5e-4, lambda_deform=1.0, off_ratio=0.0,
                 update_ratio=0.60, split_residual_ratio=0.008,
                 split_heterogeneity=0.35, split_min_gain=0.0003,
                 split_relative_gain=0.02):
        super().__init__()
        if levels < 1:
            raise ValueError("Adaptive hierarchy requires at least one level")
        if dynamic_scores.numel() != anchors.shape[0]:
            raise ValueError("Dynamic scores must contain one value per Gaussian")
        if not 0.0 <= off_ratio < update_ratio <= 1.0:
            raise ValueError("Require 0 <= off_ratio < update_ratio <= 1")
        if root_scale <= 0.0:
            raise ValueError("Hierarchy root scale must be positive")

        self.levels = levels
        self.lambda_deform = lambda_deform
        self.dynamic_threshold = dynamic_threshold
        self.image_change_threshold = image_change_threshold
        self.off_ratio = off_ratio
        self.update_ratio = update_ratio
        self.split_residual_ratio = split_residual_ratio
        self.split_heterogeneity = split_heterogeneity
        self.split_min_gain = split_min_gain
        self.split_relative_gain = split_relative_gain
        self.node_counts = []
        self.translations = nn.ParameterList()
        self.rotations = nn.ParameterList()

        self.spatial_level_indices = list(range(levels))
        for level in range(levels):
            spatial_level = self.spatial_level_indices[level]
            # S²GS's root cells use voxel_size.  Start only one octree step
            # coarser, then refine below that scale instead of opening an
            # eight-times-wider region for each dynamic cue.
            cell_size = voxel_size * root_scale / (float(fork) ** spatial_level)
            cell = torch.floor((anchors.detach() - init_pos) / cell_size).long()
            coordinates, point_to_node = torch.unique(
                cell, dim=0, sorted=True, return_inverse=True)
            self.register_buffer(f"coords_{level}", coordinates)
            self.register_buffer(f"point_to_node_{level}", point_to_node)
            node_count = coordinates.shape[0]
            self.node_counts.append(node_count)
            self.translations.append(nn.Parameter(anchors.new_zeros(node_count, 3)))
            self.rotations.append(nn.Parameter(anchors.new_zeros(node_count, 3)))
            self.register_buffer(
                f"states_{level}", torch.full(
                    (node_count,), OFF, dtype=torch.int8, device=anchors.device))
            self.register_buffer(
                f"reached_{level}", torch.zeros(
                    node_count, dtype=torch.bool, device=anchors.device))

        scores = torch.nan_to_num(dynamic_scores.detach().flatten(), nan=0.0)
        self.register_buffer("dynamic_scores", scores)
        if image_change_scores is None:
            image_change_scores = torch.zeros_like(scores)
        if image_change_scores.numel() != scores.numel():
            raise ValueError("Image change scores must contain one value per Gaussian")
        self.register_buffer("image_change_scores", torch.nan_to_num(
            image_change_scores.detach().flatten(), nan=0.0))
        self.register_buffer(
            "update_point_mask", torch.zeros(
                anchors.shape[0], dtype=torch.bool, device=anchors.device))
        self.register_buffer(
            "point_depth", torch.zeros(
                anchors.shape[0], dtype=torch.long, device=anchors.device))
        self.register_buffer(
            "residual_sum", anchors.new_zeros(anchors.shape[0]))
        self.register_buffer(
            "residual_count", anchors.new_zeros(anchors.shape[0]))
        self.register_buffer(
            "terminal_translation", anchors.new_zeros(anchors.shape[0], 3))
        terminal_quaternion = anchors.new_zeros(anchors.shape[0], 4)
        terminal_quaternion[:, 0] = 1.0
        self.register_buffer("terminal_quaternion", terminal_quaternion)
        self.terminal_only = False
        self.trial_nodes = None
        self.finished = False
        self.last_trial_candidates = 0
        self.last_trial_accepted = 0
        self.current_level = 0
        self._frontier_nodes = torch.zeros(
            self.node_counts[0], dtype=torch.bool, device=anchors.device)
        self._initialize_first_level()

        self.optimizer = torch.optim.Adam(
            list(self.translations) + list(self.rotations), lr=lr, eps=1e-15)

    def _node_mean(self, values, level):
        mapping = getattr(self, f"point_to_node_{level}")
        count = values.new_zeros(self.node_counts[level])
        total = values.new_zeros(self.node_counts[level])
        count.scatter_add_(0, mapping, torch.ones_like(values))
        total.scatter_add_(0, mapping, values)
        return total / count.clamp_min(1.0)

    def _node_observed_mean(self, values, observed, level):
        """Aggregate only observed points, keeping the two tensors' units."""
        mapping = getattr(self, f"point_to_node_{level}")
        total = values.new_zeros(self.node_counts[level])
        count = values.new_zeros(self.node_counts[level])
        weights = observed.to(values.dtype)
        total.scatter_add_(0, mapping, values * weights)
        count.scatter_add_(0, mapping, weights)
        return total / count.clamp_min(1.0), count

    def _dynamic_ratio(self, level):
        active = self._point_evidence().float()
        return self._node_mean(active, level)

    def _point_evidence(self):
        return ((self.dynamic_scores > self.dynamic_threshold)
                | (self.image_change_scores > self.image_change_threshold))

    @torch.no_grad()
    def _initialize_first_level(self):
        ratio = self._dynamic_ratio(0)
        reached = getattr(self, "reached_0")
        states = getattr(self, "states_0")
        reached.fill_(True)
        candidates = ratio > self.off_ratio
        states[candidates] = UPDATE
        self._frontier_nodes = candidates
        mapping = getattr(self, "point_to_node_0")
        self.update_point_mask.copy_(candidates[mapping])
        self.point_depth.zero_()

    @torch.no_grad()
    def accumulate_residual(self, visible_mask, residual_values):
        visible_mask = visible_mask.flatten().bool()
        if visible_mask.numel() != self.update_point_mask.numel():
            raise ValueError("Visible mask size does not match hierarchy")
        if residual_values.numel() != int(visible_mask.sum().item()):
            raise ValueError("Residual values do not match visible points")
        self.residual_sum[visible_mask] += residual_values.detach().flatten()
        self.residual_count[visible_mask] += 1.0

    def _child_heterogeneity(self, residual_scores, observed, level):
        if level + 1 >= self.levels:
            return residual_scores.new_zeros(self.node_counts[level])
        parent_map = getattr(self, f"point_to_node_{level}")
        child_map = getattr(self, f"point_to_node_{level + 1}")
        child_mean, child_count = self._node_observed_mean(
            residual_scores, observed, level + 1)
        point_child_mean = child_mean[child_map]
        point_child_observed = child_count[child_map] > 0
        maximum = residual_scores.new_full(
            (self.node_counts[level],), -torch.inf)
        minimum = residual_scores.new_full(
            (self.node_counts[level],), torch.inf)
        maximum.scatter_reduce_(0, parent_map, torch.where(
            point_child_observed, point_child_mean,
            torch.full_like(point_child_mean, -torch.inf)),
                                reduce="amax", include_self=True)
        minimum.scatter_reduce_(0, parent_map, torch.where(
            point_child_observed, point_child_mean,
            torch.full_like(point_child_mean, torch.inf)),
                                reduce="amin", include_self=True)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        minimum = torch.nan_to_num(minimum, posinf=0.0)
        return (maximum - minimum) / maximum.abs().clamp_min(1e-8)

    @torch.no_grad()
    def begin_split_trial(self):
        """Temporarily expose children; image-loss validation decides later."""
        level = self.current_level
        if self.finished or level + 1 >= self.levels or not torch.any(self._frontier_nodes):
            self.residual_sum.zero_()
            self.residual_count.zero_()
            return False

        observed = self.residual_count > 0
        residual_scores = torch.where(
            observed,
            self.residual_sum / self.residual_count.clamp_min(1.0),
            torch.zeros_like(self.residual_sum),
        )
        node_residual, observations = self._node_observed_mean(
            residual_scores, observed, level)
        dynamic_ratio = self._dynamic_ratio(level)
        heterogeneity = self._child_heterogeneity(
            residual_scores, observed, level)
        candidate = (self._frontier_nodes & (observations > 0)
                     & (node_residual >= self.split_residual_ratio)
                     & ((dynamic_ratio < self.update_ratio)
                        | (heterogeneity >= self.split_heterogeneity)))
        self.last_trial_candidates = int(candidate.sum().item())
        if not torch.any(candidate):
            self.finished = True
            self.residual_sum.zero_()
            self.residual_count.zero_()
            return False

        parent_map = getattr(self, f"point_to_node_{level}")
        child_map = getattr(self, f"point_to_node_{level + 1}")
        trial_points = candidate[parent_map]
        child_reached = torch.zeros(
            self.node_counts[level + 1], dtype=torch.int32,
            device=self.update_point_mask.device)
        child_reached.scatter_add_(0, child_map, trial_points.to(torch.int32))
        child_reached = child_reached > 0
        child_candidates = child_reached & (
            self._dynamic_ratio(level + 1) > self.off_ratio)

        self.trial_nodes = candidate.clone()
        getattr(self, f"states_{level}")[candidate] = SPLIT
        getattr(self, f"reached_{level + 1}").copy_(child_reached)
        getattr(self, f"states_{level + 1}")[child_candidates] = UPDATE
        self.update_point_mask[trial_points] = child_candidates[child_map[trial_points]]
        self.point_depth[trial_points] = level + 1
        self.current_level = level + 1
        self._frontier_nodes = child_candidates

        self.residual_sum.zero_()
        self.residual_count.zero_()
        return True

    @torch.no_grad()
    def finish_split_trial(self, before_error, after_error):
        """Keep only splits that measurably lower held-out local image error."""
        if self.trial_nodes is None:
            return self.state_summary()
        parent_level = self.current_level - 1
        candidate = self.trial_nodes
        gain = before_error - after_error
        accepted = (candidate & torch.isfinite(gain)
                    & (gain > self.split_min_gain)
                    & (gain > before_error * self.split_relative_gain))
        self.last_trial_accepted = int(accepted.sum().item())
        rejected = candidate & ~accepted
        parent_map = getattr(self, f"point_to_node_{parent_level}")
        child_map = getattr(self, f"point_to_node_{self.current_level}")
        rejected_points = rejected[parent_map]
        self.update_point_mask[rejected_points] = True
        self.point_depth[rejected_points] = parent_level
        getattr(self, f"states_{parent_level}")[rejected] = UPDATE

        accepted_children = torch.zeros(
            self.node_counts[self.current_level], dtype=torch.int32,
            device=parent_map.device)
        accepted_children.scatter_add_(
            0, child_map, accepted[parent_map].to(torch.int32))
        accepted_children = accepted_children > 0
        reached = getattr(self, f"reached_{self.current_level}")
        reached &= accepted_children
        states = getattr(self, f"states_{self.current_level}")
        states[~reached] = OFF
        self._frontier_nodes &= accepted_children
        self.finished = not bool(torch.any(self._frontier_nodes))
        self.trial_nodes = None
        return self.state_summary()

    @torch.no_grad()
    def finalize(self):
        # Keep the hierarchical residual parameterization while it is useful
        # for optimization, then bake every path into its terminal region.
        translation, quaternion = self._forward_path()
        for level in range(self.levels):
            mapping = getattr(self, f"point_to_node_{level}")
            terminal_points = self.update_point_mask & (self.point_depth == level)
            if not torch.any(terminal_points):
                continue
            weights = terminal_points.to(translation.dtype)
            counts = translation.new_zeros(self.node_counts[level])
            counts.scatter_add_(0, mapping, weights)
            terminal_nodes = counts > 0
            translation_sum = translation.new_zeros(
                self.node_counts[level], 3)
            quaternion_sum = quaternion.new_zeros(
                self.node_counts[level], 4)
            translation_sum.scatter_add_(
                0, mapping[:, None].expand(-1, 3),
                translation * weights[:, None])
            quaternion_sum.scatter_add_(
                0, mapping[:, None].expand(-1, 4),
                quaternion * weights[:, None])
            self.translations[level].data[terminal_nodes] = (
                translation_sum[terminal_nodes]
                / counts[terminal_nodes, None])
            terminal_quaternions = torch.nn.functional.normalize(
                quaternion_sum[terminal_nodes]
                / counts[terminal_nodes, None], dim=-1)
            self.rotations[level].data[terminal_nodes] = \
                quaternion_to_rotation_vector(terminal_quaternions)

        self.terminal_translation.copy_(translation)
        self.terminal_quaternion.copy_(quaternion)
        self.terminal_only = True
        self.residual_sum.zero_()
        self.residual_count.zero_()
        self._frontier_nodes.zero_()
        return self.state_summary()

    def _forward_path(self):
        point_count = self.update_point_mask.numel()
        translation = self.translations[0].new_zeros(point_count, 3)
        quaternion = self.translations[0].new_zeros(point_count, 4)
        quaternion[:, 0] = 1.0
        for level in range(self.levels):
            use_level = self.update_point_mask & (self.point_depth >= level)
            point_indices = torch.nonzero(use_level, as_tuple=False).flatten()
            mapping = getattr(self, f"point_to_node_{level}")
            node_indices = mapping[point_indices]
            translation[point_indices] += self.translations[level][node_indices]
            local_quaternion = rotation_vector_to_quaternion(
                self.rotations[level][node_indices])
            quaternion[point_indices] = quaternion_multiply(
                local_quaternion, quaternion[point_indices])
        return translation, quaternion

    def forward(self):
        if self.terminal_only:
            return self.terminal_translation, self.terminal_quaternion
        return self._forward_path()

    def regularization(self):
        penalty = self.translations[0].new_zeros(())
        for level in range(self.levels):
            node_mask = (self._terminal_node_mask(level)
                         if self.terminal_only else self._path_node_mask(level))
            active_nodes = node_mask.to(
                dtype=self.translations[level].dtype)
            magnitude = (self.translations[level].abs().mean(dim=-1)
                         + self.rotations[level].abs().mean(dim=-1))
            penalty = penalty + (magnitude * active_nodes).sum() / active_nodes.sum().clamp_min(1.0)
        return self.lambda_deform * penalty

    def _terminal_node_mask(self, level):
        mapping = getattr(self, f"point_to_node_{level}")
        point_mask = self.update_point_mask & (self.point_depth == level)
        node_count = torch.zeros(
            self.node_counts[level], dtype=torch.int32, device=mapping.device)
        node_count.scatter_add_(0, mapping, point_mask.to(torch.int32))
        return node_count > 0

    def _path_node_mask(self, level):
        mapping = getattr(self, f"point_to_node_{level}")
        point_mask = self.update_point_mask & (self.point_depth >= level)
        node_count = torch.zeros(
            self.node_counts[level], dtype=torch.int32, device=mapping.device)
        node_count.scatter_add_(0, mapping, point_mask.to(torch.int32))
        return node_count > 0

    @torch.no_grad()
    def storage_bits(self, threshold=0.0):
        bits = 0
        for level in range(self.levels):
            reached = getattr(self, f"reached_{level}")
            bits += int(reached.sum().item()) * 2
            terminal_nodes = self._terminal_node_mask(level)
            magnitude = (
                torch.linalg.vector_norm(self.translations[level], dim=-1)
                + torch.linalg.vector_norm(self.rotations[level], dim=-1))
            active = terminal_nodes & (magnitude > threshold)
            index_bits = max(
                1, math.ceil(math.log2(max(2, self.node_counts[level]))))
            bits += int(active.sum().item()) * (6 * 32 + index_bits)
        return bits

    @torch.no_grad()
    def frame_payload(self, threshold=0.0):
        levels = []
        for level in range(self.levels):
            reached = getattr(self, f"reached_{level}")
            states = getattr(self, f"states_{level}")
            terminal_nodes = self._terminal_node_mask(level)
            magnitude = (
                torch.linalg.vector_norm(self.translations[level], dim=-1)
                + torch.linalg.vector_norm(self.rotations[level], dim=-1))
            motion_nodes = terminal_nodes & (magnitude > threshold)
            coordinates = getattr(self, f"coords_{level}")
            levels.append({
                "level": level,
                "spatial_level": self.spatial_level_indices[level],
                "region_coordinates": coordinates[reached].cpu(),
                "region_states": states[reached].cpu(),
                "motion_coordinates": coordinates[motion_nodes].cpu(),
                "translation": self.translations[level][motion_nodes].cpu(),
                "rotation_vector": self.rotations[level][motion_nodes].cpu(),
            })
        return {
            "levels": levels,
            "estimated_bits": self.storage_bits(threshold),
            "updated_gaussians": int(self.update_point_mask.sum().item()),
        }

    @torch.no_grad()
    def state_summary(self):
        summary = []
        for level in range(self.levels):
            reached = getattr(self, f"reached_{level}")
            states = getattr(self, f"states_{level}")
            summary.append({
                "off": int(((states == OFF) & reached).sum().item()),
                "update": int(((states == UPDATE) & reached).sum().item()),
                "split": int(((states == SPLIT) & reached).sum().item()),
            })
        return summary
