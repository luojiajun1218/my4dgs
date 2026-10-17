"""Motion capture range and full-resolution phase boundaries."""

import importlib
import math
import unittest

import torch
from torch.nn.functional import l1_loss

MODULE = importlib.import_module('pipeline.module')


class MotionImageScheduleTests(unittest.TestCase):
    def test_coarse_tracking_ends_before_coding_and_birth_statistics(self):
        for step, factor in ((1, 4), (20, 4), (21, 2), (40, 2), (41, 1),
                             (60, 1), (71, 1), (100, 1)):
            self.assertEqual(MODULE.motion_image_factor(step, 100, True, True), factor)
        for step in (1, 20, 100, 150, 201, 300):
            self.assertEqual(MODULE.motion_image_factor(step, 100, True, False), 1)
            self.assertEqual(MODULE.motion_image_factor(step, 100, False, True), 1)

    def test_full_resolution_loss_and_gradients_are_unchanged(self):
        generator = torch.Generator().manual_seed(81)
        image = torch.rand(3, 24, 25, generator=generator, requires_grad=True)
        target = torch.rand(3, 24, 25, generator=generator)
        old = .8 * l1_loss(image, target) + .2 * (1 - MODULE.ssim(image, target))
        new = MODULE.motion_photometric_loss(image, target, .2)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(new, image, retain_graph=True)[0],
                                   torch.autograd.grad(old, image)[0], rtol=0, atol=0)

    def test_extra_motion_budget_keeps_gate_and_residual_fitting_at_full_resolution(self):
        for budget in (100, 200):
            for step, factor in ((1, 4), (20, 4), (21, 2), (40, 2),
                                 (41, 1), (60, 1), (80, 1), (100, 1), (200, 1)):
                self.assertEqual(MODULE.motion_image_factor(step, budget, True, True,
                                                            bootstrap_steps=40), factor)
        # A different bootstrap boundary remains tied to that phase, too.
        self.assertEqual(MODULE.motion_image_factor(35, 200, True, True, 35), 2)
        self.assertEqual(MODULE.motion_image_factor(36, 200, True, True, 35), 1)

    def test_resume_rejects_a_different_image_schedule_but_accepts_equivalent_legacy(self):
        compatible = MODULE.motion_image_schedule_compatible
        self.assertTrue(compatible(None, True, 100, 40))
        self.assertFalse(compatible(None, True, 200, 40))
        self.assertTrue(compatible(MODULE.MOTION_IMAGE_SCHEDULE, True, 200, 40))
        self.assertTrue(compatible(None, False, 200, 40))
        self.assertFalse(compatible('unknown_schedule', True, 200, 40))

    def test_coarse_loss_keeps_gradient_to_original_full_image(self):
        image = torch.rand(3, 25, 27, requires_grad=True)
        target = torch.rand_like(image)
        MODULE.motion_photometric_loss(image, target, .2, 4).backward()
        self.assertEqual(image.grad.shape, image.shape)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.abs().sum()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'Native rasterizer capture-range check')
    def test_coarse_image_provides_direction_for_disjoint_gaussian_motion(self):
        from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
        from libgs.data.utils.graphics import getProjectionMatrix
        settings = GaussianRasterizationSettings(image_height=64, image_width=64,
            tanfovx=math.tan(.5), tanfovy=math.tan(.5), bg=torch.zeros(3, device='cuda'),
            scale_modifier=1., viewmatrix=torch.eye(4, device='cuda'),
            projmatrix=getProjectionMatrix(.01, 100, 1., 1.).T.cuda(), sh_degree=0,
            campos=torch.zeros(3, device='cuda'), prefiltered=False, debug=False)
        def render(x):
            xyz = torch.tensor([[x, 0., 2.]], device='cuda', requires_grad=True)
            image, _ = GaussianRasterizer(settings)(means3D=xyz, means2D=torch.zeros_like(xyz),
                colors_precomp=torch.ones(1, 3, device='cuda'), shs=None,
                opacities=torch.tensor([[.7]], device='cuda'),
                scales=torch.tensor([[.05]*3], device='cuda'),
                rotations=torch.tensor([[1., 0., 0., 0.]], device='cuda'), cov3D_precomp=None)
            return xyz, image
        _, target = render(.75)
        target = target.detach()
        gradients = []
        for factor in (1, 4):
            xyz, image = render(0.)
            MODULE.motion_photometric_loss(image, target, .2, factor).backward()
            gradients.append(float(xyz.grad[0, 0]))
        self.assertLess(gradients[1], -1e-4)
        self.assertGreater(abs(gradients[1]), 100 * abs(gradients[0]))


if __name__ == '__main__':
    unittest.main()
