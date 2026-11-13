"""训练时的贡献感知渲染器适配层。

按需调用 contribution_rasterizer 收集实际参与渲染的 GS 掩码，
并将结果交给训练流程；普通渲染继续使用 libgs 的原生接口。
"""

from libgs.renderer import Renderer

from .contribution_rasterizer import GaussianRasterizer


class _CollectingRasterizer:
    def __init__(self, settings, renderer):
        self.rasterizer = (GaussianRasterizer(settings, stable_backward=True)
                           if renderer.stable_backward else GaussianRasterizer(settings))
        self.renderer = renderer

    def __call__(self, **kwargs):
        image, radii, observed = self.rasterizer(**kwargs)
        if self.renderer.collect_contributions:
            self.renderer.last_contributions = observed.gt(0)
        return image, radii


class ContributionRenderer(Renderer):
    stable_backward = False
    collect_contributions = False
    last_contributions = None

    def create_rasterizer(self, viewpoint, bg_color, scaling_modifier):
        native = super().create_rasterizer(viewpoint, bg_color, scaling_modifier)
        if self.collect_contributions or self.stable_backward:
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
