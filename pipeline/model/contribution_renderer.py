"""Collect actual motion evidence during training; inference uses the native backend."""

from libgs.renderer import Renderer

from .contribution_rasterizer import GaussianRasterizer, RegionalErrorStats


class _CollectingRasterizer:
    def __init__(self, settings, renderer):
        self.rasterizer = GaussianRasterizer(settings)
        self.renderer = renderer

    def __call__(self, **kwargs):
        sink = RegionalErrorStats() if self.renderer.collect_error_stats else None
        image, radii, observed = self.rasterizer(**kwargs, stats_sink=sink)
        if self.renderer.collect_contributions:
            self.renderer.last_contributions = observed.gt(0)
        if sink is not None:
            self.renderer.last_error_stats = sink
        return image, radii


class ContributionRenderer(Renderer):
    collect_contributions = False
    last_contributions = None
    collect_error_stats = False
    last_error_stats = None

    def create_rasterizer(self, viewpoint, bg_color, scaling_modifier):
        native = super().create_rasterizer(viewpoint, bg_color, scaling_modifier)
        if self.collect_contributions or self.collect_error_stats:
            return _CollectingRasterizer(native.raster_settings, self)
        return native

    def forward(self, *args, **kwargs):
        self.last_contributions = None
        self.last_error_stats = None
        result = super().forward(*args, **kwargs)
        if self.collect_contributions:
            if self.last_contributions is None:
                raise RuntimeError('Contribution collection did not return a mask')
            result['motion_contribution_filter'] = self.last_contributions
        if self.collect_error_stats:
            if self.last_error_stats is None:
                raise RuntimeError('Regional error collection did not return forward buffers')
            result['regional_error_stats'] = self.last_error_stats
        return result
