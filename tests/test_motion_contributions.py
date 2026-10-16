"""Actual compositing evidence, backend equivalence and motion-hook wiring."""
import importlib
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

import test_spatial_gated_motion as spatial_tests
OBSERVED = importlib.import_module('pipeline.ReCon-GS.model.contribution_rasterizer')


class MotionEvidenceHookTests(unittest.TestCase):


    def test_actual_motion_counter_uses_compositing_not_frustum(self):
        source = importlib.import_module('pipeline.ReCon-GS.module')
        motion = spatial_tests.SpatialGatedMotionTests().model()
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
        source = importlib.import_module('pipeline.ReCon-GS.module')
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


if __name__ == '__main__':
    unittest.main()
