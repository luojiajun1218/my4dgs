"""Stable spatial candidates and differentiable decisions to subdivide them."""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import numpy as np


SPATIAL_ROUTING_FIELDS = ('spatial_support_ball', 'spatial_leaf_centers',
                          'spatial_leaf_nodes')


@dataclass
class SpatialGateConfig:
    enabled: bool = True
    max_depth: int = 4
    initialization: str = 'median'
    initialization_distance_quantile: float = 0.999
    reference_layout: Optional[str] = None
    outlier_assignment: str = 'spatial'
    training_schedule: str = 'staged'
    min_region_size: int = 16
    learning_rate: float = 0.15
    initial_logit: float = -0.2
    start_step: int = 20
    gate_start_step: int = 40
    freeze_step: int = 60
    temperature: float = 1.0
    learn_region_motion: bool = False  # No unrestricted image updates during bootstrap.
    image_fit_during_gate: bool = True
    joint_image_gradients: bool = False
    region_motion: str = 'absolute'
    candidate_refresh_interval: int = 0
    temporal_logit_decay: float = 1.0
    shared_image_recode: bool = False
    predictor_objective: str = 'residual_l1'
    hierarchical_private: bool = False
    private_full_motion: bool = False
    reparameterize_after_gate_freeze: bool = False
    image_relaxation: str = 'hard_ste'
    final_temperature: float = .3
    proposal_image_weight: float = 0.
    predictor_image_weight: float = 0.
    shared_discovery: bool = False
    frame_training_plan: bool = False
    final_joint_refine_steps: int = 0
    refine_unselected_candidates: bool = False
    candidate_refinement_start_step: int = 151
    reset_gate_optimizer_on_refine: bool = False

    def validate(self, motion_steps):
        if self.reference_layout is not None and (
                not isinstance(self.reference_layout,str) or not self.reference_layout.strip() or
                not self.enabled):
            raise ValueError('reference_layout requires an enabled spatial gate and a reference NPZ path')
        if (isinstance(self.final_joint_refine_steps, bool) or
                not isinstance(self.final_joint_refine_steps, int) or
                not 0 <= self.final_joint_refine_steps <= 50 or
                (self.final_joint_refine_steps and not self.frame_training_plan)):
            raise ValueError('Final joint refinement requires frame plan and 0..50 steps after birth250')
        if not isinstance(self.refine_unselected_candidates, bool):
            raise ValueError('refine_unselected_candidates must be boolean')
        if self.refine_unselected_candidates and not self.frame_training_plan:
            raise ValueError('Unselected-candidate refinement requires the frame training plan')
        if (isinstance(self.candidate_refinement_start_step, bool) or
                not isinstance(self.candidate_refinement_start_step,int) or
                self.candidate_refinement_start_step not in (101,151)):
            raise ValueError('Candidate refinement starts at step101 or151')
        if (not isinstance(self.reset_gate_optimizer_on_refine, bool) or
                (self.reset_gate_optimizer_on_refine and not self.frame_training_plan)):
            raise ValueError('Gate optimizer restart requires the frame training plan')
        if self.frame_training_plan and (self.training_schedule != 'joint' or
                not self.shared_discovery or self.freeze_step != 180 or
                self.gate_start_step != 0 or motion_steps != 200):
            raise ValueError('Frame plan requires joint discovery, gate1..100/151..180, motion200')
        relative_image = (self.region_motion == 'relative' and
                          self.image_relaxation == 'stochastic_ste' and
                          ((self.private_full_motion and
                            self.reparameterize_after_gate_freeze) or
                           (not self.hierarchical_private and not self.private_full_motion and
                            not self.reparameterize_after_gate_freeze)))
        if (isinstance(self.predictor_image_weight, bool) or
                not math.isfinite(self.predictor_image_weight) or self.predictor_image_weight < 0):
            raise ValueError('Predictor image weight must be finite and nonnegative')
        global_proposals = self.private_full_motion and not self.hierarchical_private
        image_recode = (self.shared_image_recode and self.joint_image_gradients and
                        not self.hierarchical_private and not self.private_full_motion)
        if not isinstance(self.shared_discovery, bool) or (self.shared_discovery and not image_recode):
            raise ValueError('Shared discovery requires physical predictor recoding without private permission')
        if self.predictor_image_weight and (not (global_proposals or image_recode) or
                self.image_relaxation != 'stochastic_ste' or
                (global_proposals and not self.reparameterize_after_gate_freeze) or self.proposal_image_weight):
            raise ValueError('Predictor images require stochastic proposals or physical gate recoding')
        if (isinstance(self.proposal_image_weight, bool) or
                not math.isfinite(self.proposal_image_weight) or self.proposal_image_weight < 0):
            raise ValueError('Proposal image weight must be finite and nonnegative')
        if self.proposal_image_weight and (
                not self.private_full_motion or self.image_relaxation != 'stochastic_ste'):
            raise ValueError('Proposal image supervision requires stochastic full-motion proposals')
        if not isinstance(self.reparameterize_after_gate_freeze, bool):
            raise ValueError('reparameterize_after_gate_freeze must be boolean')
        if self.reparameterize_after_gate_freeze and (
                not self.private_full_motion or self.image_relaxation != 'stochastic_ste'):
            raise ValueError('Post-gate residual coordinates require stochastic full-motion proposals')
        if not isinstance(self.private_full_motion, bool):
            raise ValueError('private_full_motion must be boolean')
        if self.private_full_motion and (self.candidate_refresh_interval or
                (not self.hierarchical_private and not self.predictor_image_weight)):
            raise ValueError('Full-motion proposals require hierarchical permission or predictor image fitting')
        if self.image_relaxation not in ('hard_ste', 'soft_anneal', 'stochastic_ste'):
            raise ValueError('Unknown image gate relaxation')
        if (self.image_relaxation == 'stochastic_ste' and self.hierarchical_private and
                not self.private_full_motion):
            raise ValueError('Stochastic hierarchical capacity requires private full-motion proposals')
        if self.image_relaxation != 'hard_ste':
            if (self.training_schedule != 'joint' or not self.joint_image_gradients or
                    (self.region_motion != 'absolute' and not relative_image)):
                raise ValueError('Image relaxation requires supported joint image coordinates')
            if (isinstance(self.final_temperature, bool) or
                    not math.isfinite(self.final_temperature) or
                    not 0 < self.final_temperature <= self.temperature):
                raise ValueError('Final gate temperature must be positive and no larger than its start')
        if not isinstance(self.hierarchical_private, bool):
            raise ValueError('hierarchical_private must be boolean')
        if self.hierarchical_private and (not self.joint_image_gradients or
                self.training_schedule != 'joint' or
                (self.region_motion != 'absolute' and not relative_image)):
            raise ValueError('Hierarchical private motion requires supported joint image fitting')
        if self.predictor_objective not in ('residual_l1', 'proximal_rd'):
            raise ValueError('Unknown shared predictor objective')
        if self.predictor_objective == 'proximal_rd' and (
                self.training_schedule != 'joint' or self.region_motion != 'absolute' or
                not self.learn_region_motion or self.joint_image_gradients or self.shared_image_recode):
            raise ValueError('Local proximal RD requires absolute joint full-motion coordinates')
        if self.region_motion not in ('absolute', 'relative'):
            raise ValueError('Unknown spatial region motion representation')
        if not isinstance(self.shared_image_recode, bool):
            raise ValueError('shared_image_recode must be boolean')
        if self.shared_image_recode and not image_recode and (self.training_schedule != 'joint' or
                not self.learn_region_motion or self.joint_image_gradients or self.region_motion != 'absolute'):
            raise ValueError('Shared image recoding requires absolute joint region learning')
        if image_recode and not self.predictor_image_weight and not self.shared_discovery:
            raise ValueError('Physical gate recoding requires direct predictor image supervision')
        if (isinstance(self.candidate_refresh_interval, bool) or
                not isinstance(self.candidate_refresh_interval, int) or self.candidate_refresh_interval < 0):
            raise ValueError('candidate_refresh_interval must be a nonnegative integer')
        if self.candidate_refresh_interval and (self.training_schedule != 'joint' or self.region_motion != 'absolute'):
            raise ValueError('Candidate refresh requires absolute joint motion')
        if (isinstance(self.temporal_logit_decay, bool) or not math.isfinite(self.temporal_logit_decay) or
                not 0 < self.temporal_logit_decay <= 1):
            raise ValueError('temporal_logit_decay must be in (0, 1]')
        if self.region_motion == 'relative' and (self.training_schedule != 'joint' or
                (self.joint_image_gradients and not relative_image)):
            raise ValueError('Relative regions require coding coordinates or stochastic residual refinement')
        if not isinstance(self.enabled, bool):
            raise ValueError('spatial gate enabled must be boolean')
        if not isinstance(self.learn_region_motion, bool):
            raise ValueError('learn_region_motion must be boolean')
        if not isinstance(self.joint_image_gradients, bool):
            raise ValueError('joint_image_gradients must be boolean')
        if self.joint_image_gradients and (self.training_schedule != 'joint' or not self.learn_region_motion):
            raise ValueError('Joint image gradients require joint training and learn_region_motion')
        if self.image_fit_during_gate is not True:
            raise ValueError('Frozen-motion gate checkpoints are replay-only')
        if not 2 <= self.max_depth <= 5 or self.min_region_size < 2:
            raise ValueError('Invalid spatial depth or minimum region size')
        _validate_initialization(self.initialization, self.initialization_distance_quantile)
        if self.outlier_assignment not in ('spatial', 'nearest_region'):
            raise ValueError('Unknown outlier assignment')
        if self.outlier_assignment == 'nearest_region' and self.initialization != 'robust_median':
            raise ValueError('Nearest-region assignment requires robust_median initialization')
        if self.training_schedule not in ('staged', 'joint'):
            raise ValueError('Unknown spatial motion training schedule')
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0 and
                math.isfinite(self.temperature) and self.temperature > 0 and
                math.isfinite(self.initial_logit)):
            raise ValueError('Invalid spatial gate optimization settings')
        if self.enabled:
            if self.training_schedule == 'joint':
                delayed_gate = image_recode and 0 <= self.gate_start_step < self.freeze_step
                if not (self.start_step == 0 and (self.gate_start_step == 0 or delayed_gate) and
                        0 < self.freeze_step < motion_steps):
                    raise ValueError('Joint fitting starts immediately and must leave a fixed-gate phase')
            elif not 1 <= self.start_step <= self.gate_start_step < self.freeze_step < motion_steps:
                raise ValueError('Gate fitting must leave time for motion fitting after hard selection')


def _validate_initialization(initialization, distance_quantile):
    if initialization not in ('median', 'robust_median'):
        raise ValueError('Spatial initialization must be median or robust_median')
    if (isinstance(distance_quantile, bool) or
            not isinstance(distance_quantile, (float, int)) or
            not math.isfinite(distance_quantile) or not .5 < distance_quantile < 1.):
        raise ValueError('Spatial initialization distance quantile must be in (.5, 1)')


def spatial_initialization_support(xyz, initialization='median', distance_quantile=.999):
    """Select statistics support, never a GS pruning or rendering mask.

    S2GS uses distance quantiles for its LoD scale estimate. Here the same
    robust-statistics idea selects a subset of the shared reference used only
    to estimate split planes. Distances are measured from the coordinate
    median, rather than S2GS's camera centers. All GS are still routed.
    """
    _validate_initialization(initialization, distance_quantile)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or
            not xyz.is_floating_point() or not bool(torch.isfinite(xyz).all())):
        raise ValueError('Spatial layout requires nonempty finite floating-point [N,3] positions')
    if initialization == 'median':
        return torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
    points = xyz.detach().cpu().double().numpy()
    radius = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    support = radius <= np.quantile(radius, distance_quantile)
    return torch.from_numpy(support).to(xyz.device)


def spatial_layout(xyz, depth, return_centers=False, *, initialization='median',
                   distance_quantile=.999, outlier_assignment='spatial',
                   return_routing=False):
    """Create an octree in reference coordinates, with the first eight cells active.

    Candidate descendants exist only as a training dictionary. The frame's
    hard gates choose a disjoint frontier; motion never accumulates along paths.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or not 2 <= depth <= 5:
        raise ValueError('Spatial depth must be an integer in [2,5]')
    support = spatial_initialization_support(xyz, initialization, distance_quantile).cpu().numpy()
    if outlier_assignment not in ('spatial', 'nearest_region'):
        raise ValueError('Unknown outlier assignment')
    if outlier_assignment == 'nearest_region' and initialization != 'robust_median':
        raise ValueError('Nearest-region assignment requires robust_median initialization')
    # Extreme background points must not collapse almost the whole scene into
    # one octant. Split occupied cells at their coordinate medians, once in the
    # reference frame. The split planes remain fixed; GS can be routed again
    # from the previous decoded positions without transmitting owner edits.
    points = xyz.detach().cpu().numpy()
    split_count = sum(8 ** level for level in range(depth))
    centers = np.zeros((split_count, 3), dtype=np.float32)
    lower, upper = np.zeros_like(centers), np.zeros_like(centers)
    lower[0], upper[0] = points[support].min(0), points[support].max(0)
    prior_codes = np.zeros(len(points), dtype=np.int64)
    paths, parent_parts = [], [torch.tensor([-1], device=xyz.device)]
    offset, previous_offset = 1, 0
    for level in range(1, depth + 1):
        octants = np.zeros(len(points), dtype=np.int64)
        occupied = set()
        for parent in np.unique(prior_codes):
            rows = np.flatnonzero(prior_codes == parent)
            statistics_rows = rows[support[rows]]
            node = previous_offset + parent
            center = (np.median(points[statistics_rows], axis=0) if len(statistics_rows)
                      else .5 * lower[node] + .5 * upper[node])
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
    routing = {}
    if outlier_assignment == 'nearest_region':
        precise = points.astype(np.float64)
        center = np.median(precise, axis=0)
        radius = float(np.quantile(np.linalg.norm(precise - center, axis=1), distance_quantile))
        supported_leaves = paths.detach().cpu().numpy()[support, -1]
        nodes, inverse = np.unique(supported_leaves, return_inverse=True)
        population = np.bincount(inverse)
        representatives = np.stack([
            np.bincount(inverse, weights=precise[support, axis]) / population
            for axis in range(3)], axis=1)
        routing = dict(spatial_support_ball=torch.from_numpy(np.r_[center, radius]),
                       spatial_leaf_centers=torch.from_numpy(representatives.astype(np.float32)),
                       spatial_leaf_nodes=torch.from_numpy(nodes.astype(np.int64)))
        paths = spatial_route(xyz, torch.from_numpy(centers), depth, routing=routing).to(xyz.device)
    result = paths, parents, opened
    if return_centers:
        result = (*result, torch.from_numpy(centers))
    return (*result, routing) if return_routing else result


def validate_spatial_routing(routing, depth):
    """Validate the static nearest-leaf dictionary shared by encoder and decoder."""
    values = [routing.get(name) for name in SPATIAL_ROUTING_FIELDS]
    if all(value is None for value in values):
        return False
    if any(value is None for value in values):
        raise ValueError('Incomplete nearest-region reference')
    ball, centers, nodes = values
    first_leaf = sum(8 ** level for level in range(depth))
    if (ball.dtype != torch.float64 or ball.shape != (4,) or
            centers.dtype != torch.float32 or centers.ndim != 2 or centers.shape[1] != 3 or
            not len(centers) or nodes.dtype != torch.long or nodes.shape != (len(centers),) or
            not bool(torch.isfinite(ball).all()) or float(ball[3]) < 0 or
            not bool(torch.isfinite(centers).all()) or
            bool(((nodes < first_leaf) | (nodes >= first_leaf + 8 ** depth)).any()) or
            bool((nodes[1:] <= nodes[:-1]).any())):
        raise ValueError('Invalid nearest-region reference')
    return True


def spatial_route(xyz, centers, depth, *, routing=None):
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
    paths = np.stack(paths, axis=1)
    if routing and validate_spatial_routing(routing, depth):
        ball = routing['spatial_support_ball'].detach().cpu().numpy()
        representatives = routing['spatial_leaf_centers'].detach().cpu().double().numpy()
        leaf_nodes = routing['spatial_leaf_nodes'].detach().cpu().numpy()
        precise = points.astype(np.float64)
        far = np.flatnonzero(np.linalg.norm(precise - ball[:3], axis=1) > ball[3])
        # Bounded-memory float64 distances. Sorted leaf IDs break exact ties
        # deterministically; no GPU nearest-neighbor kernel is used by the codec.
        first_leaf = sum(8 ** level for level in range(depth))
        offsets = np.array([sum(8 ** item for item in range(level))
                            for level in range(1, depth + 1)], dtype=np.int64)
        shifts = 3 * np.arange(depth - 1, -1, -1)
        for start in range(0, len(far), 32):
            rows = far[start:start + 32]
            distances = np.zeros((len(rows), len(representatives)), dtype=np.float64)
            for axis in range(3):
                delta = precise[rows, axis, None] - representatives[None, :, axis]
                distances += delta * delta
            nearest = leaf_nodes[distances.argmin(axis=1)] - first_leaf
            paths[rows] = offsets + np.right_shift(nearest[:, None], shifts)
    return torch.from_numpy(paths)


def spatial_state_route(xyz, state, depth=None):
    """Route from the immutable reference, including optional far-point policy."""
    depth = state.spatial_paths.shape[1] if depth is None else depth
    routing = {name: getattr(state, name, None) for name in SPATIAL_ROUTING_FIELDS}
    return spatial_route(xyz, state.spatial_centers, depth, routing=routing)


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


def spatial_accumulate(values, parents):
    """Sum corrections in the fixed breadth-first octree, retaining gradients.

    Integer-valued inputs stay integer-valued: quantize corrections once,
    sum their symbols, and multiply by the quantization step only afterwards.
    """
    parts = [values[:1]]
    start, width = 1, 8
    while start < len(values):
        end = min(start + width, len(values))
        previous = torch.cat(parts)
        parts.append(values[start:end] + previous[parents[start:end]])
        start, width = end, width * 8
    return torch.cat(parts)
