"""Geometry, occlusion and PSNR-based validation for residual births."""
from pathlib import Path
import math
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.model.multiview_birth import (
    MultiViewBirthConfig, ViewEvidence, project_points, backproject_pixels,
    score_depth_candidates, fit_initial_sh1, sh1_basis, validation_edit_gain,
    collect_view_evidence,
)


def camera(center_x=0., width=128, height=96):
    view = torch.eye(4)
    view[3, 0] = -center_x
    projection = torch.zeros(4, 4)
    projection[0, 0] = 2.
    projection[1, 1] = 2.
    projection[2, 2] = projection[2, 3] = 1.
    return SimpleNamespace(image=torch.zeros(3, height, width),
        world_view_transform=view, full_proj_transform=view @ projection,
        camera_center=torch.tensor([center_x, 0., 0.]),
        fovx=2*math.atan(.5), fovy=2*math.atan(.5))


class MultiViewBirthTests(unittest.TestCase):
    def test_pixel_roundtrip_uses_pixel_centers_and_camera_translation(self):
        cam = camera(.7)
        pixel = torch.tensor([[0., 0.], [71., 34.], [127., 95.]])
        xyz = backproject_pixels(pixel, torch.tensor([2., 4., 3.]), cam)
        grid, depth, inside = project_points(xyz, cam)
        reconstructed = (grid + 1) * torch.tensor([128., 96.]) / 2 - .5
        torch.testing.assert_close(reconstructed, pixel, atol=1e-5, rtol=0)
        torch.testing.assert_close(depth, torch.tensor([2., 4., 3.]))
        self.assertTrue(bool(inside.all()))

    def evidence(self):
        result = []
        truth = torch.tensor([[0., 0., 3.]])
        for x in (-1., 0., 1.):
            cam = camera(x)
            cam.image[:] = torch.tensor([.8, .5, .2])[:, None, None]
            grid, _, _ = project_points(truth, cam)
            uv = (grid[0]+1)*torch.tensor([128., 96.])/2-.5
            yy, xx = torch.meshgrid(torch.arange(96), torch.arange(128), indexing='ij')
            error = torch.exp(-((xx-uv[0]).square()+(yy-uv[1]).square())/2)
            result.append(ViewEvidence(cam, cam.image, torch.full((96,128),4.),
                torch.ones(96,128), error, torch.tensor(.45)))
        return result

    def test_other_cameras_select_depth_and_reject_single_view_agreement(self):
        xyz = torch.tensor([[0., 0., 2.5], [0., 0., 3.], [0., 0., 3.5]])
        score, details = score_depth_candidates(xyz,
            torch.tensor([[.8,.5,.2]]).expand(3,-1), self.evidence(), 1., MultiViewBirthConfig())
        self.assertEqual(torch.isfinite(score).tolist(), [False, True, False])
        self.assertEqual(int(details['support_count'][1]), 3)

    def test_occlusion_excludes_a_high_error_match_behind_old_surface(self):
        evidence = self.evidence()
        for item in evidence:
            item.depth.fill_(2.)
        score, details = score_depth_candidates(torch.tensor([[0.,0.,3.]]),
            torch.tensor([[.8,.5,.2]]), evidence, 1., MultiViewBirthConfig())
        self.assertFalse(bool(torch.isfinite(score).any()))
        self.assertEqual(float(details['visible_count'][0]), 0.)

    def test_depth_and_alpha_labels_exclude_the_scene_background(self):
        cam = camera()
        backgrounds = []

        def renderer(camera, background, override_color=None):
            backgrounds.append(background.clone())
            image = (camera.image if override_color is None else
                     override_color[0, :, None, None].expand_as(camera.image)*.8
                     + background[:, None, None]*.2)
            return {'render': image}

        item, = collect_view_evidence(renderer, [cam], {'xyz': torch.tensor([[0.,0.,3.]])},
                                     torch.ones(3), MultiViewBirthConfig())
        torch.testing.assert_close(item.depth, torch.full((96,128), 3.))
        torch.testing.assert_close(item.alpha, torch.full((96,128), .8))
        self.assertTrue(torch.equal(backgrounds[1], torch.zeros(3)))

    def test_multiview_color_initialization_reproduces_constant_appearance(self):
        generator = torch.Generator().manual_seed(3)
        directions = torch.nn.functional.normalize(torch.randn(12,2,3,generator=generator),dim=-1)
        colors = torch.tensor([.8,.5,.2]).expand(12,2,3)
        coefficients = fit_initial_sh1(colors,directions,torch.ones(12,2))
        rendered = torch.einsum('vni,nic->vnc',sh1_basis(directions),coefficients)+.5
        torch.testing.assert_close(rendered,colors,atol=1e-5,rtol=0)

    def test_psnr_admission_is_not_rejected_by_unrelated_photo_loss(self):
        before = dict(psnr=34.,ssim=.96,photo=.013)
        after = dict(psnr=34.03,ssim=.96,photo=.0131)
        accepted,gain = validation_edit_gain(before,after,min_gain_db=.005)
        self.assertTrue(accepted)
        self.assertAlmostEqual(gain,.03)
        self.assertFalse(validation_edit_gain(before,{**after,'ssim':.95})[0])
        self.assertFalse(validation_edit_gain(before,{**after,'psnr':float('nan')})[0])

    def test_invalid_geometry_settings_are_rejected(self):
        for config in (MultiViewBirthConfig(depth_samples=0),
                       MultiViewBirthConfig(color_sigma=float('nan')),
                       MultiViewBirthConfig(initial_alpha=1.)):
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == '__main__':
    unittest.main()
