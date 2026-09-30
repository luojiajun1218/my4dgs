"""Independent leaf probing followed by bounded, rendered bottom-up selection."""
from dataclasses import asdict, dataclass, fields
import math
from typing import List, Optional

import torch
from torch import nn

from .region_rigid import apply_region_rigid

KEEP, SHARED, SPLIT = 0, 1, 2
UNKNOWN, STATIC, CHANGED = 0, 1, 2


@dataclass
class SelectiveMotionConfig:
    # None means not discussed/configured. The legacy path does not use these.
    max_depth: Optional[int] = None
    coarse_regions: Optional[int] = None
    max_leaf_regions: Optional[int] = None
    min_region_gs: Optional[int] = None
    probe_steps: Optional[int] = None
    shared_fit_steps_per_depth: Optional[int] = None
    refine_steps: Optional[int] = None
    incremental_steps: Optional[int] = None
    enable_incremental: Optional[bool] = None
    comparison_view_names: Optional[List[str]] = None
    max_leaf_comparisons: Optional[int] = None
    max_internal_comparisons: Optional[int] = None
    loss_tolerance: Optional[float] = None
    min_visible_fraction: Optional[float] = None
    min_visible_views: Optional[int] = None
    min_change_views: Optional[int] = None
    min_changed_fraction: Optional[float] = None
    image_change_threshold: Optional[float] = None
    evidence_patch_radius: Optional[int] = None
    translation_lr: Optional[float] = None
    rotation_lr: Optional[float] = None
    adam_betas: Optional[List[float]] = None
    adam_eps: Optional[float] = None
    motion_regularization: Optional[float] = None
    camera_match_atol: Optional[float] = None
    clone_sigma_scale: Optional[float] = None
    roi_padding_pixels: Optional[int] = None
    final_loss_tolerance: Optional[float] = None

    def validate(self):
        missing = [f.name for f in fields(self) if getattr(self, f.name) is None]
        if missing:
            raise ValueError('selective_motion requires explicit YAML values: ' + ', '.join(missing))
        for name in ('max_depth', 'coarse_regions', 'max_leaf_regions', 'min_region_gs', 'probe_steps',
                     'shared_fit_steps_per_depth', 'refine_steps', 'max_leaf_comparisons',
                     'max_internal_comparisons', 'min_visible_views', 'min_change_views'):
            if getattr(self, name) < 1:
                raise ValueError(f'selective_motion.{name} must be positive')
        if self.incremental_steps < 0 or self.evidence_patch_radius < 0:
            raise ValueError('Incremental steps and patch radius must be nonnegative')
        if self.max_leaf_regions < self.coarse_regions or self.roi_padding_pixels < 0:
            raise ValueError('Leaf budget must cover coarse regions; ROI padding must be nonnegative')
        if self.enable_incremental != (self.incremental_steps > 0):
            raise ValueError('enable_incremental must agree with incremental_steps > 0')
        if not self.comparison_view_names or len(set(self.comparison_view_names)) != len(self.comparison_view_names):
            raise ValueError('comparison_view_names must contain unique training camera filenames')
        if not 0 < self.min_visible_fraction <= 1 or self.min_visible_views > len(self.comparison_view_names):
            raise ValueError('Invalid selective visibility thresholds')
        if not 0 < self.min_changed_fraction <= 1 or self.min_change_views > self.min_visible_views:
            raise ValueError('Invalid cross-view image-change thresholds')
        if len(self.adam_betas) != 2 or not all(0 <= b < 1 for b in self.adam_betas):
            raise ValueError('adam_betas must contain two values in [0, 1)')
        for name in ('translation_lr', 'rotation_lr', 'adam_eps', 'clone_sigma_scale'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('loss_tolerance', 'final_loss_tolerance', 'image_change_threshold', 'motion_regularization', 'camera_match_atol'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')

    @property
    def motion_steps(self):
        return self.probe_steps + self.max_depth * self.shared_fit_steps_per_depth + self.refine_steps

    @property
    def total_steps(self):
        return self.motion_steps + self.incremental_steps


class SelectiveMotion(nn.Module):
    def __init__(self, config: SelectiveMotionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.children = []
        self.members = []
        self.depths = []
        self.selected = False
        self.committed = False
        self.comparison_records = []
        self.topology_events = []
        self.optimizer = None
        self.final_review = None
        self.stage_seconds = {}

    def setup(self, xyz, raw_rotation):
        """Build a balanced spatial binary tree without copying Gaussian attributes."""
        self.children, self.members, self.depths = [], [], []
        for name in ('accepted_translation', 'accepted_rotation_vector'):
            if name in self._buffers:
                delattr(self, name)
        if not len(xyz):
            raise ValueError('Selective motion needs a nonempty previous-frame GS model')

        def partition(indices):
            points = xyz[indices]
            axis = (points.max(0).values - points.min(0).values).argmax()
            order = torch.argsort(points[:, axis], stable=True)
            half = indices.numel() // 2
            return indices[order[:half]], indices[order[half:]]

        coarse = [torch.arange(len(xyz), device=xyz.device)]
        while len(coarse) < self.config.coarse_regions:
            largest = max(range(len(coarse)), key=lambda i: len(coarse[i]))
            if len(coarse[largest]) < 2 * self.config.min_region_gs:
                break
            coarse[largest:largest + 1] = partition(coarse[largest])
        leaf_count = len(coarse)

        def build(indices, depth):
            nonlocal leaf_count
            node = len(self.members)
            self.members.append(indices)
            self.depths.append(depth)
            self.children.append([])
            if (depth < self.config.max_depth and indices.numel() >= 2 * self.config.min_region_gs
                    and leaf_count < self.config.max_leaf_regions):
                leaf_count += 1
                left, right = partition(indices)
                self.children[node] = [build(left, depth + 1), build(right, depth + 1)]
            return node

        for indices in coarse:
            build(indices, 0)
        centers = torch.stack([xyz[m].mean(0) for m in self.members])
        self.register_or_set('base_xyz', xyz.detach().clone())
        self.register_or_set('base_rotation', raw_rotation.detach().clone())
        self.register_or_set('centers', centers)
        self.register_or_set('states', torch.full((len(centers),), SPLIT, device=xyz.device, dtype=torch.long))
        self.register_or_set('evidence', torch.zeros(len(centers), device=xyz.device, dtype=torch.long))
        self.register_or_set('visible_views', torch.zeros(len(centers), device=xyz.device, dtype=torch.long))
        self.register_or_set('change_votes', torch.zeros_like(self.visible_views))
        self.register_or_set('old_error', torch.zeros_like(centers[:, 0]))
        self.register_or_set('new_error', torch.zeros_like(centers[:, 0]))
        self.register_or_set('reference_radii', torch.zeros(len(xyz), device=xyz.device))
        self.translation = nn.Parameter(torch.zeros_like(centers))
        self.rotation_vector = nn.Parameter(torch.zeros_like(centers))
        owner = torch.empty(len(xyz), device=xyz.device, dtype=torch.long)
        for node, children in enumerate(self.children):
            if not children:
                owner[self.members[node]] = node
                self.states[node] = SHARED
        self.register_or_set('owner', owner)
        self.selected, self.committed = False, False
        self.comparison_records = []
        self.topology_events = []
        self.final_review = None
        self.stage_seconds = {}
        self.make_optimizer()

    def register_or_set(self, name, value):
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value)

    def make_optimizer(self):
        self.optimizer = torch.optim.Adam([
            dict(params=[self.translation], lr=self.config.translation_lr),
            dict(params=[self.rotation_vector], lr=self.config.rotation_lr)],
            betas=tuple(self.config.adam_betas), eps=self.config.adam_eps)

    def owners_at_depth(self, depth):
        owner = self.owner.clone()
        for node, node_depth in enumerate(self.depths):
            if node_depth == depth or (node_depth < depth and not self.children[node]):
                owner[self.members[node]] = node
        return owner

    def stage(self, step):
        if step <= self.config.probe_steps:
            return 'probe', self.owner
        fit_step = step - self.config.probe_steps - 1
        fit_budget = self.config.max_depth * self.config.shared_fit_steps_per_depth
        if fit_step < fit_budget:
            depth = self.config.max_depth - 1 - fit_step // self.config.shared_fit_steps_per_depth
            return 'shared_fit', self.owners_at_depth(depth)
        return 'refine', self.owner

    def transform(self, owner=None, keep=None):
        owner = self.owner if owner is None else owner
        keep = self.states[owner] == KEEP if keep is None else keep
        # Unknown and observed static are distinct evidence, but both reuse old GS.
        keep = keep | (self.evidence[owner] != CHANGED)
        xyz, rotation = apply_region_rigid(
            self.base_xyz, self.base_rotation, self.centers[owner],
            self.translation[owner], self.rotation_vector[owner])
        return torch.where(keep[:, None], self.base_xyz, xyz), torch.where(
            keep[:, None], self.base_rotation, rotation)

    def regularization(self, owner):
        nodes = owner.unique()
        nodes = nodes[self.evidence[nodes] == CHANGED]
        if not nodes.numel():
            return self.translation.sum() * 0
        return self.config.motion_regularization * (
            self.translation[nodes].square().mean() + self.rotation_vector[nodes].square().mean())

    @torch.no_grad()
    def optimizer_step(self, owner):
        active = torch.zeros(len(self.members), device=owner.device, dtype=torch.bool)
        nodes = owner.unique()
        active[nodes] = (self.evidence[nodes] == CHANGED) & (self.states[nodes] != KEEP)
        frozen = ~active
        for parameter in (self.translation, self.rotation_vector):
            if parameter.grad is not None:
                parameter.grad[frozen] = 0
            for value in self.optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value[frozen] = 0
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # A frozen row has zero momentum and cannot drift in future steps.
        # Keep fitted inactive candidates intact for later comparisons.

    @torch.no_grad()
    def gather_evidence(self, pairs, render_reference):
        """Sample image changes around visible old centers in matched training cameras."""
        radius = self.config.evidence_patch_radius
        offset = torch.arange(-radius, radius + 1, device=self.base_xyz.device)
        oy, ox = torch.meshgrid(offset, offset, indexing='ij')
        homogeneous = torch.cat([self.base_xyz, torch.ones_like(self.base_xyz[:, :1])], -1)
        for current, previous in pairs:
            reference = render_reference(current)
            visibility = reference['visibility_filter'][:len(self.base_xyz)]
            self.reference_radii = torch.maximum(self.reference_radii, reference['radii'][:len(self.base_xyz)].float())
            clip = homogeneous @ current.full_proj_transform
            valid = clip[:, 3] > 0
            ndc = clip[:, :2] / torch.where(valid[:, None], clip[:, 3:4], torch.ones_like(clip[:, 3:4]))
            height, width = current.image.shape[-2:]
            pixel = ((ndc + 1) * self.base_xyz.new_tensor([width, height]) / 2 - .5).round().long()
            px = pixel[:, 0, None] + ox.flatten()
            py = pixel[:, 1, None] + oy.flatten()
            inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            visible = visibility & valid & inside.any(-1)
            difference = (current.image - previous.image).abs().mean(0)
            sampled = difference[py.clamp(0, height - 1), px.clamp(0, width - 1)]
            change = (sampled * inside).sum(-1) / inside.sum(-1).clamp_min(1)
            old_residual = (reference['render'] - previous.image).abs().mean(0)
            new_residual = (reference['render'] - current.image).abs().mean(0)
            old_sample = old_residual[py.clamp(0, height - 1), px.clamp(0, width - 1)]
            new_sample = new_residual[py.clamp(0, height - 1), px.clamp(0, width - 1)]
            old_error = (old_sample * inside).sum(-1) / inside.sum(-1).clamp_min(1)
            new_error = (new_sample * inside).sum(-1) / inside.sum(-1).clamp_min(1)
            for node, indices in enumerate(self.members):
                observed = visible[indices]
                if observed.float().mean() >= self.config.min_visible_fraction:
                    self.visible_views[node] += 1
                    fraction = (change[indices][observed] > self.config.image_change_threshold).float().mean()
                    self.change_votes[node] += fraction >= self.config.min_changed_fraction
                    self.old_error[node] += old_error[indices][observed].mean()
                    self.new_error[node] += new_error[indices][observed].mean()
        observed = self.visible_views >= self.config.min_visible_views
        changed = self.change_votes >= self.config.min_change_views
        self.old_error /= self.visible_views.clamp_min(1)
        self.new_error /= self.visible_views.clamp_min(1)
        self.evidence[:] = UNKNOWN
        self.evidence[observed & (self.change_votes == 0)] = STATIC
        self.evidence[observed & changed] = CHANGED
        self.states[self.evidence != CHANGED] = KEEP

    @torch.no_grad()
    def select(self, render_loss):
        """Compare common-ROI rendered losses; other selected owners stay fixed."""
        compared = {'leaf': 0, 'internal': 0}
        # Reserve independent shortlists before the bottom-up walk; rank by new
        # reference error above old fit error, then cross-view change evidence.
        allowed = set()
        for internal, budget in ((False, self.config.max_leaf_comparisons),
                                 (True, self.config.max_internal_comparisons)):
            nodes = [n for n, children in enumerate(self.children)
                     if bool(children) == internal and self.evidence[n] == CHANGED]
            nodes.sort(key=lambda n: (float(self.new_error[n] - self.old_error[n]),
                                     float(self.change_votes[n] / self.visible_views[n].clamp_min(1))), reverse=True)
            allowed.update(nodes[:budget])
        for node in reversed(range(len(self.members))):
            indices = self.members[node]
            if self.evidence[node] != CHANGED:
                # Uncertain coarse visibility must not erase observed child motion.
                descendants = self.owner[indices].unique()
                if (self.evidence[descendants] == CHANGED).any():
                    self.states[node] = SPLIT
                    continue
                self.owner[indices] = node
                self.states[node] = KEEP
                continue
            kind = 'internal' if self.children[node] else 'leaf'
            if node not in allowed:
                continue  # Retain independently probed descendants; never invent KEEP.
            current_owner = self.owner.clone()
            child_keep = self.states[current_owner] == KEEP
            alternative = current_owner.clone()
            alternative[indices] = node
            keep_mask = child_keep.clone()
            keep_mask[indices] = True
            shared_keep = child_keep.clone()
            shared_keep[indices] = False
            baseline, keep_loss, shared_loss = render_loss(node, [
                (current_owner, child_keep), (alternative, keep_mask), (alternative, shared_keep)])
            if not all(math.isfinite(value) for value in (baseline, keep_loss, shared_loss)):
                raise ValueError('Selective candidate rendering produced a nonfinite loss')
            # Protect unseen/static leaves and descendants already accepted as KEEP.
            shared_blocked = bool((self.evidence[current_owner[indices].unique()] != CHANGED).any()
                                  or child_keep[indices].any())
            descendant_evidence = self.evidence[current_owner[indices].unique()]
            blocked_reasons = []
            if bool((descendant_evidence == UNKNOWN).any()):
                blocked_reasons.append('unknown_evidence')
            if bool((descendant_evidence == STATIC).any()):
                blocked_reasons.append('patch_static_evidence')
            if bool(child_keep[indices].any()):
                blocked_reasons.append('selected_keep')
            if shared_blocked:
                shared_loss = float('inf')
            best = min(baseline, keep_loss, shared_loss)
            tolerance = self.config.loss_tolerance
            if keep_loss <= best + tolerance:
                state = KEEP
            elif shared_loss <= best + tolerance:
                state = SHARED
            else:
                state = SPLIT if self.children[node] else SHARED
            if state != SPLIT:
                self.owner[indices] = node
            self.states[node] = state
            self.comparison_records.append(dict(node=node, keep=keep_loss,
                                                shared=None if shared_blocked else shared_loss,
                                                shared_blocked=shared_blocked, shared_blocked_reasons=blocked_reasons,
                                                children=baseline, state=state))
            compared[kind] += 1
        self.selected = True
        self.register_or_set('accepted_translation', self.translation.detach().clone())
        self.register_or_set('accepted_rotation_vector', self.rotation_vector.detach().clone())
        # Clear all old momentum; selected owners start a new refinement phase.
        self.make_optimizer()

    def capture(self):
        if not self.members:
            return None
        return dict(config=asdict(self.config), state=self.state_dict(), children=self.children, members=self.members,
                    depths=self.depths, selected=self.selected, committed=self.committed,
                    comparisons=self.comparison_records, topology_events=self.topology_events,
                    final_review=self.final_review,
                    stage_seconds=self.stage_seconds,
                    optimizer=self.optimizer.state_dict())

    def restore(self, packet):
        if packet is None:
            return
        if packet['config'] != asdict(self.config):
            raise ValueError('Selective resume requires the saved structure, budgets and selection parameters')
        state = packet['state']
        self.setup(state['base_xyz'], state['base_rotation'])
        self.children, self.members, self.depths = packet['children'], packet['members'], packet['depths']
        for name in ('accepted_translation', 'accepted_rotation_vector'):
            if name in state:
                self.register_or_set(name, state[name].clone())
        self.load_state_dict(state)
        self.selected, self.committed = packet['selected'], packet['committed']
        self.comparison_records = packet['comparisons']
        self.topology_events = packet['topology_events']
        self.final_review = packet['final_review']
        self.stage_seconds = packet['stage_seconds']
        self.optimizer.load_state_dict(packet['optimizer'])
