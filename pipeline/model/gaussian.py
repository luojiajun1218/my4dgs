"""Gaussian storage and newborn fitting for the active rigid residual tree."""

from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from libgs.model.gaussian import GaussianModel as BaseGaussianModel
from libgs.utils.general import build_rotation


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
        dc = getattr(self, 'rigid_sh_dc', None)
        dc = self._features_dc if dc is None else dc
        rest = getattr(self, 'rigid_sh_rest', None)
        rest = self._features_rest if rest is None else rest
        return self._render_cached('features', (dc, rest),
            lambda: torch.cat((dc, rest), dim=1))

    @property
    def get_scaling(self):
        override = getattr(self, 'rigid_logscale', None)
        source = self._scaling if override is None else override
        return self._render_cached('scaling_override' if override is not None else 'scaling',
                                   (source,), lambda: self.scaling_activation(source))

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
        sigma_scale: float = 2.0,
        *,
        max_new_gaussians: Optional[int] = None,
        source_covariance_init: bool = False,
        return_source_rows: bool = False,
        appearance_scores: Optional[Tensor] = None,
        max_appearance_gaussians: int = 0,
    ):
        """Clone selected sources, optionally returning their exact row indices."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Extract points that satisfy the gradient condition
        scores = torch.norm(grads, dim=-1)
        scaling = self.get_scaling
        appearance_enabled = (
            appearance_scores is not None and max_appearance_gaussians > 0)
        if appearance_enabled:
            if (isinstance(max_appearance_gaussians, bool)
                    or not isinstance(max_appearance_gaussians, int)
                    or max_appearance_gaussians < 0):
                raise ValueError(
                    'Appearance clone budget requires a nonnegative int')
            if appearance_scores.shape != (self.get_xyz.shape[0],):
                raise ValueError('appearance_scores must have shape (n,)')
            if not bool(torch.isfinite(appearance_scores).all().item()):
                raise ValueError('appearance_scores must be finite')
        positional_mask = scores >= grad_threshold
        positional_mask = torch.logical_and(
            positional_mask,
            torch.max(scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        if appearance_enabled:
            appearance_candidate_mask = appearance_scores > 0
            appearance_limit = max_appearance_gaussians
            if max_new_gaussians is not None:
                appearance_limit = min(appearance_limit, max_new_gaussians)
            appearance_mask = select_budget_mask(
                appearance_candidate_mask, appearance_scores, appearance_limit)
            appearance_count = int(appearance_mask.sum().item())
            positional_mask = torch.logical_and(positional_mask, ~appearance_mask)
            eligible = int(torch.logical_or(
                positional_mask, appearance_candidate_mask).sum().item())
            if max_new_gaussians is not None:
                positional_mask = select_budget_mask(
                    positional_mask, scores,
                    max(0, max_new_gaussians - appearance_count))
            selected_pts_mask = torch.logical_or(positional_mask, appearance_mask)
        else:
            selected_pts_mask = positional_mask
            eligible = int(selected_pts_mask.sum().item())
            if max_new_gaussians is not None:
                selected_pts_mask = select_budget_mask(
                    selected_pts_mask, scores, max_new_gaussians)
        added = int(selected_pts_mask.sum().item())
        source_rows = (selected_pts_mask.nonzero(as_tuple=True)[0]
                       if return_source_rows else None)

        means = self._xyz[selected_pts_mask]
        stds = scaling[selected_pts_mask]
        source_rotation = self._rotation[selected_pts_mask]
        if source_covariance_init:
            from .rigid_residual_tree import quaternion_rotate
            samples = torch.normal(torch.zeros_like(means), stds * sigma_scale)
            q = torch.nn.functional.normalize(source_rotation, dim=-1)
            new_xyz = means + quaternion_rotate(q, samples)
        else:
            new_xyz = torch.normal(means, stds * sigma_scale)
        if appearance_enabled:
            new_xyz = torch.where(
                appearance_mask[selected_pts_mask].unsqueeze(-1), means, new_xyz)
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        # Preserve ReCon-GS's raw opacity-logit initialization.
        new_opacities = torch.zeros_like(self._opacity[selected_pts_mask]) + 0.1
        new_scaling = self._scaling[selected_pts_mask]
        if source_covariance_init:
            new_rotation = source_rotation.clone()
        else:
            new_rotation = torch.zeros_like(source_rotation)
            new_rotation[:, 0] = 1

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )
        report = dict(eligible=eligible, added=added, blocked=eligible - added)
        if appearance_enabled:
            report["appearance_added"] = int(appearance_mask.sum().item())
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
    """Track the exact source row through newborn splits and pruning."""

    source_rows: Optional[Tensor] = None

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
        return super().prune_points(mask)

    def adaptive_densify_and_split(self, grads, grad_threshold, scene_extent,
                                   N=2, *, max_points=None):
        if self.source_rows is None:
            raise ValueError('Birth lineage must be set before candidate splitting')
        try:
            return super().adaptive_densify_and_split(
                grads, grad_threshold, scene_extent, N, max_points=max_points)
        except Exception:
            # The caller must abandon this candidate model on a failed split.
            self.source_rows = None
            raise

    def _split_selected_points(self, selected_pts_mask, N, selected):
        # Append N children in exactly the order used by the geometry below.
        self.source_rows = torch.cat((self.source_rows,
                                      self.source_rows[selected_pts_mask].repeat(N)))
        super()._split_selected_points(selected_pts_mask, N, selected)
