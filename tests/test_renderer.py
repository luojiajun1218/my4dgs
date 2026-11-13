"""检查渲染结果、贡献统计和渲染缓存。

覆盖贡献掩码、透明度覆盖、图像及属性梯度，以及缓存启用前后的一致性；
图像损失本身由 test_loss.py 检查。
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.model import contribution_rasterizer as raster
from pipeline.model import contribution_renderer as renderer_module
from pipeline.model.frame_stream_format import RAW_ATTRIBUTES
from pipeline.model.gaussian import GaussianModel

class ContributionRendererTests(unittest.TestCase):
    def test_renderer_packages_contribution_mask(self):
        renderer = renderer_module.ContributionRenderer.__new__(renderer_module.ContributionRenderer)
        torch.nn.Module.__init__(renderer)
        renderer.collect_contributions = True
        native = SimpleNamespace(raster_settings='settings')

        class FakeRasterizer:
            def __init__(self, settings):
                self.settings = settings

            def __call__(self, **kwargs):
                return torch.ones(3, 2, 3), torch.ones(2), torch.tensor([1, 0])

        def base_forward(obj, *args, **kwargs):
            image, radii = obj.create_rasterizer(None, None, 1.)(means3D=None)
            return {'render': image, 'radii': radii}

        with patch.object(renderer_module.Renderer, 'create_rasterizer', return_value=native), \
             patch.object(renderer_module.Renderer, 'forward', base_forward), \
             patch.object(renderer_module, 'GaussianRasterizer', FakeRasterizer):
            result = renderer(None, None)
            self.assertEqual(result['motion_contribution_filter'].tolist(), [True, False])

class ActiveRenderCacheTests(unittest.TestCase):
    @staticmethod
    def model(trainable):
        result = GaussianModel(0)
        raw = dict(_xyz=torch.tensor([[-.25, 0., 0.], [.25, 0., 0.]]),
                   _rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                   _scaling=torch.full((2, 3), -.4),
                   _opacity=torch.tensor([[.1], [-.2]]),
                   _features_dc=torch.tensor([[[.2, .4, .6]], [[.8, .3, .1]]]),
                   _features_rest=torch.zeros(2, 0, 3))
        for name, value in raw.items():
            setattr(result, name, torch.nn.Parameter(value, requires_grad=trainable))
        return result

    @staticmethod
    def image(model):
        # A small differentiable image assembled from the same renderer inputs.
        pixels = torch.linspace(-1., 1., 7)
        centers = model.get_xyz[:, :1]
        width = model.get_scaling[:, :1]
        alpha = model.get_opacity
        rotation = model.get_rotation
        color = model.get_features[:, 0, :] * (rotation[:, :1] + rotation[:, 1:2])
        coverage = torch.exp(-.5 * ((centers - pixels) / width).square()) * alpha
        return coverage.T @ color

    def test_cache_preserves_image_and_attribute_gradients(self):
        cached, uncached = self.model(True), self.model(True)
        uncached.render_attribute_cache = False
        left, right = self.image(cached), self.image(uncached)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().sum().backward()
        right.square().sum().backward()
        for name in RAW_ATTRIBUTES:
            torch.testing.assert_close(getattr(cached, name).grad,
                                       getattr(uncached, name).grad,
                                       rtol=0, atol=0)

    def test_frozen_cache_preserves_motion_gradient_and_invalidates_on_change(self):
        cached, uncached = self.model(False), self.model(False)
        uncached.render_attribute_cache = False
        cached.rigid_xyz = cached._xyz.detach().clone().requires_grad_()
        uncached.rigid_xyz = uncached._xyz.detach().clone().requires_grad_()
        cached.rigid_rotation = cached._rotation.detach().clone().requires_grad_()
        uncached.rigid_rotation = uncached._rotation.detach().clone().requires_grad_()
        with torch.no_grad():
            self.assertIs(cached.get_features, cached.get_features)
            self.assertIs(cached.get_scaling, cached.get_scaling)
            self.assertIs(cached.get_opacity, cached.get_opacity)
        left, right = self.image(cached), self.image(uncached)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().sum().backward()
        right.square().sum().backward()
        torch.testing.assert_close(cached.rigid_xyz.grad, uncached.rigid_xyz.grad,
                                   rtol=0, atol=0)
        torch.testing.assert_close(cached.rigid_rotation.grad,
                                   uncached.rigid_rotation.grad, rtol=0, atol=0)
        self.assertGreater(float(cached.rigid_rotation.grad.abs().sum()), 0.)
        with torch.no_grad():
            old = cached.get_opacity
            cached._opacity.add_(.25)
            self.assertIsNot(cached.get_opacity, old)
            torch.testing.assert_close(cached.get_opacity,
                                       torch.sigmoid(cached._opacity), rtol=0, atol=0)

class OpacityRenderTests(unittest.TestCase):
    def test_alpha_override_uses_quantized_values_with_gradient(self):
        model = GaussianModel(1)
        model._opacity = torch.nn.Parameter(torch.tensor([[2.]]), requires_grad=False)
        override = torch.tensor([[-1.]], requires_grad=True)
        model.rigid_opacity = override
        model.get_opacity.sum().backward()
        self.assertGreater(float(override.grad), 0.)
        self.assertIsNone(model._opacity.grad)
        model.rigid_opacity = None
        torch.testing.assert_close(model.get_opacity, model._opacity.sigmoid())

@unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA contribution tests')
class MotionContributionGPUChecks(unittest.TestCase):
    def settings(self):
        from diff_gaussian_rasterization import GaussianRasterizationSettings
        from libgs.data.utils.graphics import getProjectionMatrix
        return GaussianRasterizationSettings(image_height=64, image_width=64,
            tanfovx=math.tan(.5), tanfovy=math.tan(.5), bg=torch.zeros(3, device='cuda'),
            scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'),
            projmatrix=getProjectionMatrix(.01,100,1.,1.).transpose(0,1).cuda(),
            sh_degree=1, campos=torch.zeros(3,device='cuda'), prefiltered=False, debug=False)

    def arguments(self, count=12):
        generator = torch.Generator(device='cuda').manual_seed(916)
        xyz = torch.randn(count,3,generator=generator,device='cuda')*.15
        xyz[:,2] += 2
        quaternion = torch.randn(count,4,generator=generator,device='cuda')
        quaternion /= quaternion.norm(dim=1,keepdim=True)
        return dict(means3D=xyz, means2D=torch.zeros_like(xyz),
            shs=torch.randn(count,4,3,generator=generator,device='cuda')*.1,
            opacities=torch.full((count,1),.7,device='cuda'),
            scales=torch.full((count,3),.06,device='cuda'), rotations=quaternion,
            cov3D_precomp=None, colors_precomp=None)

    def test_image_exact_and_all_image_gradients_match_native(self):
        from diff_gaussian_rasterization import GaussianRasterizer
        settings = self.settings()
        original = self.arguments()
        native = {k: None if v is None else v.detach().clone().requires_grad_(True)
                  for k,v in original.items()}
        observed = {k: None if v is None else v.detach().clone().requires_grad_(True)
                    for k,v in original.items()}
        native_image, native_radii = GaussianRasterizer(settings)(**native)
        image, radii, mask = raster.GaussianRasterizer(settings)(**observed)
        self.assertTrue(torch.equal(image, native_image))
        self.assertTrue(torch.equal(radii, native_radii))
        self.assertFalse(bool((mask.gt(0) & radii.eq(0)).any()))
        weights=torch.linspace(.1,1,image.numel(),device='cuda').reshape_as(image)
        (native_image.square()*weights).sum().backward()
        (image.square()*weights).sum().backward()
        for name in ('means3D','means2D','shs','opacities','scales','rotations'):
            torch.testing.assert_close(observed[name].grad,native[name].grad,rtol=2e-5,atol=2e-5)

    def test_transparent_occluded_out_of_view_and_visible(self):
        args=self.arguments(7)
        args.update(shs=None,colors_precomp=torch.ones(7,3,device='cuda'),
                    opacities=torch.ones(7,1,device='cuda'),
                    rotations=torch.tensor([[1.,0.,0.,0.]]*7,device='cuda'))
        args['means3D']=torch.tensor([[0.,0.,2.]]*4+[[0.,0.,3.],[.8,0.,2.],[100.,0.,2.]],device='cuda')
        args['scales']=torch.tensor([[.25]*3]*4+[[.0001]*3,[.1]*3,[.1]*3],device='cuda')
        args['opacities'][5]=0
        image,radii,mask=raster.GaussianRasterizer(self.settings())(**args)
        self.assertGreater(int(radii[4]),0)
        self.assertGreater(int(radii[5]),0)
        self.assertFalse(bool(mask[4]))  # Small rear point hidden by the front layers.
        self.assertFalse(bool(mask[5]))  # In-frustum transparent point.
        self.assertFalse(bool(mask[6]))  # Outside image bounds.
        self.assertTrue(bool(mask[:4].any()))





if __name__ == '__main__':
    unittest.main()
