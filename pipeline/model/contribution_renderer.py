"""Collect actual motion evidence during training; inference uses the native backend."""

from dataclasses import dataclass

import torch
from libgs.renderer import Renderer, RendererConfig as BaseRendererConfig

from .contribution_rasterizer import GaussianRasterizer


@dataclass
class RendererConfig(BaseRendererConfig):
    stable_covariance_gradients: bool = False


class _CollectingRasterizer:
    def __init__(self, settings, renderer, stable=False):
        self.rasterizer = GaussianRasterizer(settings, stable_covariance_gradients=stable)
        self.renderer = renderer

    def __call__(self, **kwargs):
        image, radii, observed = self.rasterizer(**kwargs)
        if self.renderer.collect_contributions:
            self.renderer.last_contributions = observed.gt(0)
        return image, radii


class ContributionRenderer(Renderer):
    collect_contributions = False
    last_contributions = None

    def create_rasterizer(self, viewpoint, bg_color, scaling_modifier):
        native = super().create_rasterizer(viewpoint, bg_color, scaling_modifier)
        stable = getattr(self.config, 'stable_covariance_gradients', False)
        if not isinstance(stable, bool):
            raise ValueError('stable_covariance_gradients must be boolean')
        if self.collect_contributions or (stable and torch.is_grad_enabled()):
            return _CollectingRasterizer(native.raster_settings, self, stable)
        return native

    def forward(self, *args, **kwargs):
        self.last_contributions = None
        result = super().forward(*args, **kwargs)
        if self.collect_contributions:
            if self.last_contributions is None:
                raise RuntimeError('Contribution collection did not return a mask')
            result['motion_contribution_filter'] = self.last_contributions
        return result
