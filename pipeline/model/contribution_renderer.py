"""Collect actual motion evidence during training; inference uses the native backend."""

import torch
from libgs.renderer import Renderer

from .contribution_rasterizer import GaussianRasterizer


class _CollectingRasterizer:
    def __init__(self, settings, renderer):
        self.rasterizer = GaussianRasterizer(settings)
        self.renderer = renderer

    def __call__(self, **kwargs):
        image, radii, observed = self.rasterizer(**kwargs)
        self.renderer.last_contributions = observed.gt(0)
        return image, radii


class ContributionRenderer(Renderer):
    collect_contributions = False
    last_contributions = None

    def create_screenspace_points(self):
        xyz = self.gaussians.get_xyz
        if torch.is_grad_enabled():
            # A fresh leaf already retains its gradient. The native '+ 0'
            # creates an unnecessary tensor/kernel before every training view.
            return torch.zeros_like(xyz, requires_grad=True)
        signature = (tuple(xyz.shape), xyz.device, xyz.dtype)
        if getattr(self, '_screenspace_signature', None) != signature:
            self._screenspace_zero = torch.zeros_like(xyz)
            self._screenspace_signature = signature
        return self._screenspace_zero

    def create_rasterizer(self, viewpoint, bg_color, scaling_modifier):
        native = super().create_rasterizer(viewpoint, bg_color, scaling_modifier)
        if self.collect_contributions:
            return _CollectingRasterizer(native.raster_settings, self)
        return native

    def forward(self, *args, **kwargs):
        self.last_contributions = None
        result = super().forward(*args, **kwargs)
        if self.collect_contributions:
            if self.last_contributions is None:
                raise RuntimeError('Contribution collection did not return a mask')
            result['motion_contribution_filter'] = self.last_contributions
        return result
