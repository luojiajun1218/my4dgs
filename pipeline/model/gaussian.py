"""Gaussian storage and newborn fitting for the active rigid residual tree."""

from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from libgs.model.gaussian import GaussianModel as BaseGaussianModel
from libgs.utils.general import build_rotation
from .birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS


def select_budget_mask(mask: Tensor, scores: Tensor, limit: Optional[int]) -> Tensor:
    """Keep highest scoring eligible rows, preserving row order on score ties."""
    if limit is None or int(mask.sum().item()) <= limit:
        return mask
    indices = mask.nonzero(as_tuple=True)[0]
    ranked = indices[torch.argsort(scores[indices], descending=True, stable=True)]
    selected = torch.zeros_like(mask)
    selected[ranked[:limit]] = True
    return selected


class GaussianModel(BaseGaussianModel):
    # Frozen canonical attributes are reused across full inference renders.
    # _render_cached bypasses autograd sources and invalidates on tensor change.
    render_attribute_cache = True

    def _render_cached(self, name, tensors, function):
        if (not self.render_attribute_cache or torch.is_inference_mode_enabled() or
                (torch.is_grad_enabled() and any(value.requires_grad for value in tensors))):
            return function()
        try:
            signature = tuple(value._version for value in tensors)
        except RuntimeError:  # Inference tensors have no mutation version counter.
            return function()
        cache = getattr(self, '_render_cache', None)
        if cache is None:
            cache = self._render_cache = {}
        cached = cache.get(name)
        if (cached is None or cached[0] != signature or
                any(previous is not current for previous, current in zip(cached[1], tensors))):
            # Holding the sources prevents allocator/address or Python-id reuse
            # from producing a false match after temporary motion or restore.
            cache[name] = (signature, tuple(tensors), function())
        return cache[name][2]

    @property
    def get_features(self):
        dc = self._features_dc
        return self._render_cached('features', (dc, self._features_rest),
            lambda: torch.cat((dc, self._features_rest), dim=1))

    @property
    def get_scaling(self):
        return self._render_cached('scaling', (self._scaling,),
                                   lambda: self.scaling_activation(self._scaling))

    @property
    def get_opacity(self):
        alpha = getattr(self, 'rigid_alpha', None)
        if alpha is not None:
            return alpha
        override = getattr(self, 'rigid_opacity', None)
        if override is not None:
            return self._render_cached('opacity_override', (override,),
                                      lambda: self.opacity_activation(override))
        return self._render_cached('opacity', (self._opacity,),
                                   lambda: self.opacity_activation(self._opacity))

    def initialize_incremental_optimizer(self, training_args) -> None:
        """Start newborn optimization with stage-two rates and fresh Adam moments.

        Call after restoring the clone snapshot and pruning it to newborn rows.
        Restored optimizer groups carry the old stage's learning rates and state.
        """
        self.training_setup(training_args)

    @property
    def get_xyz(self) -> Float[Tensor, "n 3"]:
        override = getattr(self, 'rigid_xyz', None)
        return super().get_xyz if override is None else override

    @property
    def get_rotation(self) -> Float[Tensor, "n 4"]:
        override = getattr(self, 'rigid_rotation', None)
        if override is not None:
            return self._render_cached('rotation_override', (override,),
                                       lambda: self.rotation_activation(override))
        return self._render_cached('rotation', (self._rotation,),
                                   lambda: self.rotation_activation(self._rotation))

    def get_covariance(self, scaling_modifier=1):
        if getattr(self, 'rigid_rotation', None) is not None:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
        return super().get_covariance(scaling_modifier)

    def add_densification_stats(
        self,
        viewspace_point_tensor: Float[Tensor, "n 2"],
        update_filter: Bool[Tensor, "n"],
        align: Literal["left", "right"] = "left",
    ):
        grad = viewspace_point_tensor.grad
        if grad.shape[0] != update_filter.shape[0]:
            if align == "left":
                grad = grad[: update_filter.shape[0]]
            else:
                grad = grad[-update_filter.shape[0] :]

        self.xyz_gradient_accum[update_filter] += torch.norm(
            grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def adaptive_densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        scene_extent: float,
        max_screen_size: Optional[float] = None,
        *,
        max_points: Optional[int] = None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        split_stats = self.adaptive_densify_and_split(
            grads, max_grad, scene_extent, max_points=max_points
        )

        before_prune = len(self._xyz)
        low_opacity = (self.get_opacity < min_opacity).reshape(-1)
        big_points_vs = torch.zeros_like(low_opacity)
        big_points_ws = torch.zeros_like(low_opacity)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * scene_extent
        prune_mask = low_opacity | big_points_vs | big_points_ws
        reasons = dict(low_opacity=int(low_opacity.sum()),
                       screen_size=int(big_points_vs.sum()),
                       world_size=int(big_points_ws.sum()),
                       multiple=int(((low_opacity.int() + big_points_vs.int() +
                                      big_points_ws.int()) > 1).sum()))
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        return dict(**split_stats, pruned=before_prune - len(self._xyz),
                    prune_reasons=reasons,
                    peak_before_prune=before_prune)

    def adaptive_densify_and_clone(
        self,
        grad_threshold: float,
        scene_extent: float,
        *,
        max_new_gaussians: Optional[int] = None,
        return_source_rows: bool = False,
    ):
        """Append exact source copies for small, observed high-gradient points."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Extract points that satisfy the gradient condition
        scores = torch.norm(grads, dim=-1)
        scaling = self.get_scaling
        selected_pts_mask = scores >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        eligible = int(selected_pts_mask.sum().item())
        if max_new_gaussians is not None:
            selected_pts_mask = select_budget_mask(
                selected_pts_mask, scores, max_new_gaussians)
        added = int(selected_pts_mask.sum().item())
        source_rows = (selected_pts_mask.nonzero(as_tuple=True)[0]
                       if return_source_rows else None)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
        report = dict(eligible=eligible, added=added, blocked=eligible - added)
        return (report, source_rows) if return_source_rows else report

    def adaptive_densify_and_split(
        self,
        grads: Float[Tensor, "n 2"],
        grad_threshold: float,
        scene_extent: float,
        N: int = 2,
        *,
        max_points: Optional[int] = None,
    ):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, device="cuda")
        padded_grad[: grads.shape[0]] = grads.reshape(-1)

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent,
        )
        eligible = int(selected_pts_mask.sum().item())
        if max_points is not None:
            if isinstance(N, bool) or not isinstance(N, int) or N < 2:
                raise ValueError('Budgeted split requires integer N >= 2')
            available = max(0, (max_points - n_init_points) // (N - 1))
            selected_pts_mask = select_budget_mask(selected_pts_mask, padded_grad, available)
        selected = int(selected_pts_mask.sum().item())
        self._split_selected_points(selected_pts_mask, N, selected)
        return dict(eligible=eligible, added=selected * (N - 1),
                    blocked=eligible - selected)

    def _split_selected_points(self, selected_pts_mask, N, selected):
        """Materialize a single selection shared by geometry and birth lineage."""
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            stds / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        options = dict(device="cuda", dtype=bool)
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected, **options),
            )
        )
        self.prune_points(prune_filter)

    @torch.no_grad()
    def apply_selected_motion(self, xyz: Tensor, raw_rotation: Tensor):
        """Copy accepted values exactly; the base optimizer is never stepped."""
        self._xyz.copy_(xyz)
        self._rotation.copy_(raw_rotation)
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in (self._xyz, self._rotation):
            self.optimizer.state.pop(parameter, None)

    def prune_points(self, mask_or_indices: Tensor):
        if mask_or_indices.dtype is torch.bool:
            return super().prune_points(mask_or_indices)
        mask = torch.zeros_like(self._xyz[:, 0], dtype=torch.bool)
        mask[mask_or_indices] = True
        return super().prune_points(mask)


class BirthGaussianModel(GaussianModel):
    """Keep predictor lineage and learning age aligned with candidate topology."""

    source_rows: Optional[Tensor] = None
    created_steps: Optional[Tensor] = None

    @torch.no_grad()
    def constrain_position_step(self, previous_xyz, previous_logscale,
                                previous_rotation, sigma_cap, minimum_step):
        """Limit Adam's move along each Gaussian axis, down to the codec grid."""
        from .rigid_residual_tree import quaternion_rotate
        rotation = torch.nn.functional.normalize(previous_rotation, dim=-1)
        inverse = rotation.clone()
        inverse[:, 1:] *= -1
        local = quaternion_rotate(inverse, self._xyz - previous_xyz)
        bound = (previous_logscale.exp() * sigma_cap).clamp_min(minimum_step)
        local = torch.minimum(torch.maximum(local, -bound), bound)
        self._xyz.copy_(previous_xyz + quaternion_rotate(rotation, local))

    @torch.no_grad()
    def initialize_covariance_sources(self, source_model, source_rows, created_step,
                                      training_args, sigma_scale=0.5,
                                      initial_alpha=0.1, split_scale_threshold=None,
                                      max_screen_size=None):
        """Seed nearby, low-alpha points without copying the whole scene.

        Source rows are attribute-codec predictors, not spatial region IDs.
        The configured opacity is a true alpha; store its logit so its
        quantized render starts with the same intended weak coverage.
        """
        from .rigid_residual_tree import quaternion_rotate
        rows = torch.as_tensor(source_rows, device=source_model._xyz.device,
                               dtype=torch.long).flatten()
        born = {name: getattr(source_model, attr)[rows].detach().clone()
                for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}
        scaling = born['logscale'].exp()
        factor = torch.ones_like(scaling[:, :1])
        if split_scale_threshold is not None:
            large = scaling.amax(dim=-1) > split_scale_threshold
            factor[large] = 1 / 1.6
        if max_screen_size is not None:
            # Use only the training views already observed during motion.
            # Leave room for fitting before the unchanged screen-size prune.
            radius = source_model.max_radii2D[rows, None].clamp_min(1)
            factor = torch.minimum(factor, (0.5 * max_screen_size / radius).clamp_max(1))
        scaling = scaling * factor
        born['logscale'] += factor.log()
        samples = torch.normal(torch.zeros_like(born['xyz']), scaling * sigma_scale)
        rotation = torch.nn.functional.normalize(born['rotation'], dim=-1)
        born['xyz'] += quaternion_rotate(rotation, samples)
        born['opacity'] = torch.logit(torch.full_like(born['opacity'], initial_alpha))
        self.initialize_from_sources(source_model, None, created_step,
                                     training_args, proposal=(born, rows))

    @staticmethod
    def _cap_source_opacity(raw, max_alpha):
        """Cap old-source clone alpha in alpha space, preserving logit semantics."""
        if max_alpha is None:
            return raw
        if not 0 < float(max_alpha) < 1:
            raise ValueError('Old-source clone alpha cap must be in (0, 1)')
        alpha = raw['_opacity'].sigmoid().clamp(max=float(max_alpha))
        alpha = alpha.clamp_min(torch.finfo(alpha.dtype).eps)
        raw['_opacity'] = torch.logit(alpha)
        return raw

    @torch.no_grad()
    def initialize_from_sources(self, source_model, source_rows, created_step,
                                training_args, opacity_alpha_cap=None, proposal=None):
        """Create the first candidate pool from a residual proposal or source rows."""
        if proposal is not None:
            born, inherited = proposal
            raw = {attr: born[name].detach().clone()
                   for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}
            inherited = torch.as_tensor(inherited, dtype=torch.long).flatten()
            if source_model is not None:
                self.active_sh_degree = source_model.active_sh_degree
                self.spatial_lr_scale = source_model.spatial_lr_scale
        else:
            source_rows = torch.as_tensor(source_rows, device=source_model._xyz.device,
                                          dtype=torch.long).flatten()
            self.active_sh_degree = source_model.active_sh_degree
            self.spatial_lr_scale = source_model.spatial_lr_scale
            inherited = (source_model.source_rows[source_rows]
                         if isinstance(source_model, BirthGaussianModel) else source_rows)
            raw = {attr: getattr(source_model, attr)[source_rows].detach().clone()
                   for attr in RAW_ATTRIBUTES}
            self._cap_source_opacity(raw, opacity_alpha_cap)
        for attr in RAW_ATTRIBUTES:
            value = raw[attr]
            setattr(self, attr, torch.nn.Parameter(value))
        count = len(inherited)
        device = self._xyz.device
        self.source_rows = inherited.detach().clone().long()
        self.created_steps = torch.full((count,), int(created_step),
                                        device=device, dtype=torch.long)
        self.max_radii2D = self._xyz.new_zeros(count)
        self.xyz_gradient_accum = self._xyz.new_zeros((count, 1))
        self.denom = self._xyz.new_zeros((count, 1))
        self.optimizer = None
        if count:
            self.training_setup(training_args)
            self.max_radii2D = self._xyz.new_zeros(count)

    @torch.no_grad()
    def append_clones_from(self, source_model, source_rows, created_step,
                           training_args=None, opacity_alpha_cap=None):
        """Append exact copies and map each child to its old reference source."""
        source_rows = torch.as_tensor(source_rows, device=source_model._xyz.device,
                                      dtype=torch.long).flatten()
        if not len(source_rows):
            return 0
        inherited = (source_model.source_rows[source_rows]
                     if isinstance(source_model, BirthGaussianModel) else source_rows)
        raw = {attr: getattr(source_model, attr)[source_rows].detach().clone()
               for attr in RAW_ATTRIBUTES}
        self._cap_source_opacity(raw, opacity_alpha_cap)
        if self.optimizer is None:
            if training_args is None:
                raise ValueError('A birth optimizer configuration is required for first clones')
            self.initialize_from_sources(source_model, source_rows, created_step,
                                          training_args, opacity_alpha_cap)
            return len(source_rows)
        previous_radii = self.max_radii2D.detach().clone()
        self.densification_postfix(
            raw['_xyz'], raw['_features_dc'], raw['_features_rest'],
            raw['_opacity'], raw['_scaling'], raw['_rotation'])
        # libgs resets all densification buffers in densification_postfix.
        # Keep the already observed screen-size evidence for existing rows;
        # newly appended clones have no radius evidence until rendered.
        self.max_radii2D = torch.cat((previous_radii,
            previous_radii.new_zeros(len(source_rows))))
        self.source_rows = torch.cat((self.source_rows, inherited.detach().clone().long()))
        self.created_steps = torch.cat((self.created_steps,
            self.created_steps.new_full((len(source_rows),), int(created_step))))
        return len(source_rows)

    @torch.no_grad()
    def append_proposal(self, proposal, created_step, training_args=None):
        """Append residual-proposal points with image-fitted positions and appearance."""
        born, inherited = proposal
        inherited = torch.as_tensor(inherited, dtype=torch.long).flatten()
        if not len(inherited):
            return 0
        if self.optimizer is None:
            if training_args is None:
                raise ValueError('A birth optimizer configuration is required for first proposals')
            self.initialize_from_sources(None, None, created_step, training_args,
                                          proposal=proposal)
            return len(inherited)
        previous_radii = self.max_radii2D.detach().clone()
        self.densification_postfix(
            born['xyz'], born['sh_dc'], born['sh_rest'],
            born['opacity'], born['logscale'], born['rotation'])
        self.max_radii2D = torch.cat((previous_radii,
            previous_radii.new_zeros(len(inherited))))
        self.source_rows = torch.cat((self.source_rows,
            inherited.to(self.source_rows.device)))
        self.created_steps = torch.cat((self.created_steps,
            self.created_steps.new_full((len(inherited),), int(created_step))))
        return len(inherited)

    @torch.no_grad()
    def split_rows(self, source_rows, created_step, N=2):
        """Replace large newborns with covariance-sampled children."""
        source_rows = torch.as_tensor(source_rows, device=self._xyz.device,
                                      dtype=torch.long).flatten()
        if not len(source_rows):
            return 0
        if (N < 2 or self.optimizer is None or self.source_rows is None or
                self.created_steps is None):
            raise ValueError('Birth splitting requires an initialized pool and N >= 2')
        inherited = self.source_rows[source_rows].repeat(N).clone()
        n_selected = len(source_rows)
        stds = self.get_scaling[source_rows].repeat(N, 1)
        samples = torch.normal(torch.zeros_like(stds), stds)
        rotations = build_rotation(self._rotation[source_rows]).repeat(N, 1, 1)
        xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1)
        xyz = xyz + self._xyz[source_rows].repeat(N, 1)
        scaling = self.scaling_inverse_activation(stds / (0.8 * N))
        rotation = self._rotation[source_rows].repeat(N, 1)
        dc = self._features_dc[source_rows].repeat(N, 1, 1)
        rest = self._features_rest[source_rows].repeat(N, 1, 1)
        opacity = self._opacity[source_rows].repeat(N, 1)
        previous_radii = self.max_radii2D.detach().clone()
        self.densification_postfix(xyz, dc, rest, opacity, scaling, rotation)
        # Retain observed radii for unsplit rows. Split children start without
        # a screen-size observation and their parents are pruned immediately.
        self.max_radii2D = torch.cat((previous_radii,
            previous_radii.new_zeros(N * n_selected)))
        self.source_rows = torch.cat((self.source_rows, inherited))
        self.created_steps = torch.cat((self.created_steps,
            self.created_steps.new_full((N * n_selected,), int(created_step))))
        remove = torch.zeros(len(self._xyz), dtype=torch.bool, device=self._xyz.device)
        remove[source_rows] = True
        self.prune_points(remove)
        return N * n_selected

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold, min_opacity, scene_extent,
                          max_screen_size, created_step, max_points=None):
        """Split observed large newborns and prune, following 0902bc1."""
        if not len(self._xyz):
            return dict(clone_added=0, split_added=0, pruned=0,
                        split_children=0, split_parents_replaced=0,
                        prune_reasons={})
        grads = self.xyz_gradient_accum / self.denom.clamp_min(1)
        scores = torch.linalg.vector_norm(grads, dim=-1)
        selected = self.denom.flatten().gt(0) & scores.ge(grad_threshold)
        scaling = self.get_scaling
        large = selected & scaling.max(dim=1).values.gt(self.percent_dense * scene_extent)
        # Each two-child split replaces its parent, adding one net row.
        if max_points is not None:
            large = select_budget_mask(large, scores, max(0, max_points - len(self._xyz)))
        split_rows = large.nonzero(as_tuple=True)[0]
        splits = self.split_rows(split_rows, created_step, N=2)

        before_prune = len(self._xyz)
        low_opacity = self.get_opacity.flatten().lt(min_opacity)
        big_screen = self.max_radii2D.gt(max_screen_size)
        big_world = self.get_scaling.max(dim=1).values.gt(0.1 * scene_extent)
        remove = low_opacity | big_screen | big_world
        reasons = dict(low_opacity=int(low_opacity.sum()),
                       screen_size=int(big_screen.sum()),
                       world_size=int(big_world.sum()),
                       multiple=int(((low_opacity.int() + big_screen.int() +
                                      big_world.int()) > 1).sum()))
        if bool(remove.any()):
            self.prune_points(remove)
        return dict(clone_added=0,
                    split_added=splits - len(split_rows),
                    split_children=splits,
                    split_parents_replaced=len(split_rows),
                    pruned=before_prune - len(self._xyz),
                    prune_reasons=reasons, eligible=int(selected.sum()))

    def prune_points(self, mask_or_indices: Tensor):
        if mask_or_indices.dtype is torch.bool:
            mask = mask_or_indices
        else:
            mask = torch.zeros(len(self._xyz), dtype=torch.bool,
                               device=self._xyz.device)
            mask[mask_or_indices] = True
        if self.source_rows is not None:
            if len(self.source_rows) != len(mask):
                raise ValueError('Birth source rows disagree with candidate count')
            self.source_rows = self.source_rows[~mask]
        if self.created_steps is not None:
            if len(self.created_steps) != len(mask):
                raise ValueError('Birth ages disagree with candidate count')
            self.created_steps = self.created_steps[~mask]
        return super().prune_points(mask)
