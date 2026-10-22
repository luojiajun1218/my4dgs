"""Rendering evidence and gradient-preserving tensor caches, with optional native CUDA checks."""
from types import SimpleNamespace
import unittest
import torch
import importlib
import math
from unittest.mock import Mock
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.model.contribution_renderer import ContributionRenderer
from pipeline.module import materialize_motion_evidence_counts
from _motion_fixtures import staged_motion


OBSERVED = importlib.import_module('pipeline.model.contribution_rasterizer')


MODULE = importlib.import_module('pipeline.module')


GAUSSIAN = importlib.import_module('pipeline.model.gaussian')


class RendererScratchTests(unittest.TestCase):
    def renderer(self):
        return SimpleNamespace(gaussians=SimpleNamespace(get_xyz=torch.ones(4, 3)))

    def test_training_views_have_distinct_leaf_gradients(self):
        renderer = self.renderer()
        first = ContributionRenderer.create_screenspace_points(renderer)
        second = ContributionRenderer.create_screenspace_points(renderer)
        self.assertTrue(first.is_leaf and second.is_leaf)
        self.assertIsNot(first, second)
        (first.sum() * 2 + second.sum() * 3).backward()
        torch.testing.assert_close(first.grad, torch.full_like(first, 2))
        torch.testing.assert_close(second.grad, torch.full_like(second, 3))

    def test_inference_cache_invalidates_on_shape_or_dtype_and_never_enters_training(self):
        renderer = self.renderer()
        with torch.no_grad():
            first = ContributionRenderer.create_screenspace_points(renderer)
            self.assertIs(first, ContributionRenderer.create_screenspace_points(renderer))
            renderer.gaussians.get_xyz = torch.ones(5, 3)
            changed = ContributionRenderer.create_screenspace_points(renderer)
            self.assertIsNot(first, changed)
            renderer.gaussians.get_xyz = torch.ones(5, 3, dtype=torch.float64)
            self.assertIsNot(changed, ContributionRenderer.create_screenspace_points(renderer))
        training = ContributionRenderer.create_screenspace_points(renderer)
        self.assertTrue(training.requires_grad)
        self.assertIsNot(training, renderer._screenspace_zero)

    def test_diagnostic_counts_serialize_as_integers_after_accumulating_views(self):
        control = dict(motion_frustum_samples=3, motion_observed_samples=1)
        for counts in (torch.tensor([5, 2]), torch.tensor([7, 4])):
            for key, value in zip(control, counts):
                control[key] = control[key] + value
        self.assertTrue(all(torch.is_tensor(value) for value in control.values()))
        materialize_motion_evidence_counts(control)
        self.assertEqual(control, dict(motion_frustum_samples=15, motion_observed_samples=7))
        self.assertTrue(all(type(value) is int for value in control.values()))
        materialize_motion_evidence_counts(control)


class MotionEvidenceHookTests(unittest.TestCase):


    def test_actual_motion_counter_uses_compositing_not_frustum(self):
        source = importlib.import_module('pipeline.module')
        motion = staged_motion()
        motion.observations.zero_()
        config = source.ModuleConfig(effective_motion_observations=True)
        adapter = SimpleNamespace(mark_observed=motion.mark_observed, stage=lambda step: 'free',
                                  optimizer_step=lambda **kwargs: [])
        module = SimpleNamespace(trainer=SimpleNamespace(global_step=1), motion_committed=False,
            motion=adapter, config=config, birth_control={}, motion_steps=200)
        frustum = torch.ones_like(motion.observations, dtype=torch.bool)
        used = torch.zeros_like(frustum)
        used[1:3] = True
        source.Module.post_training_step(module, dict(visibility_filter=frustum,
                                                    motion_contribution_filter=used))
        self.assertTrue(torch.equal(motion.observations.gt(0), used))
        self.assertEqual(module.birth_control['motion_observed_samples'], 2)
        self.assertEqual(module.birth_control['motion_frustum_samples'], len(frustum))
        before = motion.observations.clone()
        with self.assertRaisesRegex(ValueError, 'compositing mask'):
            source.Module.post_training_step(module, dict(visibility_filter=frustum))
        self.assertTrue(torch.equal(before, motion.observations))

    def test_old_configuration_still_uses_frustum(self):
        source = importlib.import_module('pipeline.module')
        config = source.ModuleConfig()
        mark = Mock()
        module = SimpleNamespace(trainer=SimpleNamespace(global_step=1), motion_committed=False,
            motion=SimpleNamespace(mark_observed=mark, stage=lambda step: 'free',
                                   optimizer_step=lambda **kwargs: []),
            config=config, birth_control={}, motion_steps=200)
        visible = torch.tensor([True, False])
        source.Module.post_training_step(module, dict(visibility_filter=visible))
        self.assertIs(mark.call_args.args[0], visible)


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
        image, radii, mask = OBSERVED.GaussianRasterizer(settings)(**observed)
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
        image,radii,mask=OBSERVED.GaussianRasterizer(self.settings())(**args)
        self.assertGreater(int(radii[4]),0)
        self.assertGreater(int(radii[5]),0)
        self.assertFalse(bool(mask[4]))  # Small rear point hidden by the front layers.
        self.assertFalse(bool(mask[5]))  # In-frustum transparent point.
        self.assertFalse(bool(mask[6]))  # Outside image bounds.
        self.assertTrue(bool(mask[:4].any()))


class ActiveRenderCacheTests(unittest.TestCase):
    @staticmethod
    def model(trainable):
        result = GAUSSIAN.GaussianModel(0)
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
        for name in MODULE.RAW_ATTRIBUTES:
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


if __name__ == '__main__':
    unittest.main()
