"""GS 模型的属性存储与渲染接口。

负责基础属性、当前帧属性覆盖、渲染激活与缓存，以及模型状态操作；
区域和逐点属性如何更新分别由对应 attributes_update 文件负责。
"""

from typing import Literal, Optional

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from libgs.model.gaussian import GaussianModel as BaseGaussianModel
from libgs.utils.general import build_rotation

from .frame_stream_format import RAW_ATTRIBUTES, RAW_FIELDS


def model_raw(model):
    """Copy the model's stored, unactivated Gaussian attributes."""
    return {name: getattr(model, attr).detach().clone()
            for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES)}


def clone_tensor_tree(value, device):
    """Copy checkpoint tensors to the requested device, preserving containers."""
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {key: clone_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tensor_tree(item, device) for item in value)
    return value


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
