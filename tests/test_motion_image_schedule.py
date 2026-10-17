"""Motion capture range and full-resolution phase boundaries."""

import importlib
import math
from types import SimpleNamespace
import unittest

import torch
from torch.nn.functional import l1_loss
import test_proposal_image_supervision as proposal_fixtures

MODULE = importlib.import_module('pipeline.module')


class MotionImageScheduleTests(unittest.TestCase):
    def test_rmse_follows_psnr_error_order_when_l1_prefers_a_large_outlier(self):
        target = torch.zeros(3, 24, 24)
        outlier = target.clone()
        outlier[:, 12, 12] = .3
        diffuse = target + .005
        self.assertLess(float(l1_loss(outlier, target)), float(l1_loss(diffuse, target)))
        self.assertGreater(float((outlier - target).square().mean()),
                           float((diffuse - target).square().mean()))
        self.assertGreater(float(MODULE.motion_photometric_loss(outlier, target, 0., metric='rmse')),
                           float(MODULE.motion_photometric_loss(diffuse, target, 0., metric='rmse')))

    def test_rmse_gives_weak_pixel_errors_less_gradient_than_large_errors(self):
        image = torch.full((3, 24, 24), .001)
        image[:, 0, 0] = .1
        image.requires_grad_()
        target = torch.zeros_like(image)
        loss = MODULE.motion_photometric_loss(image, target, 0., metric='rmse')
        grad = torch.autograd.grad(loss, image)[0]
        self.assertAlmostEqual(float(grad[0, 0, 0] / grad[0, 1, 1]), 100., places=4)
        l1_grad = torch.autograd.grad(l1_loss(image, target), image)[0]
        self.assertEqual(float(l1_grad[0, 0, 0]), float(l1_grad[0, 1, 1]))

    def test_rmse_exact_match_has_zero_finite_gradient_and_rejects_unknown_metric(self):
        image = torch.full((3, 24, 24), .37, requires_grad=True)
        loss = MODULE.motion_photometric_loss(image, image.detach(), 0., metric='rmse')
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(image.grad).all()))
        self.assertTrue(bool(image.grad.eq(0).all()))
        with self.assertRaises(ValueError):
            MODULE.motion_photometric_loss(image, image.detach(), .2, metric='unknown')

    def test_module_uses_rmse_for_both_main_and_auxiliary_proposal_images(self):
        helper = proposal_fixtures.ProposalImageSupervisionTests()
        module, _, _ = helper.module(helper.model())
        module.config.photometric_loss = 'rmse'
        module.config.lambda_dssim = 0.
        target = torch.linspace(0., 1., 3 * 20 * 20).reshape(3, 20, 20)
        loss, metrics, package = MODULE.Module.training_step(module,
            SimpleNamespace(image=target), 1, regularization_weight=0.)
        expected = ((package['render'].double() - target.double()).square().mean() + 1e-8).sqrt() - 1e-4
        torch.testing.assert_close(metrics['image_loss'].double(), expected.detach(), rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(loss.double(), expected * 2, rtol=1e-6, atol=1e-8)
        self.assertEqual(MODULE.ModuleConfig().photometric_loss, 'l1')

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
