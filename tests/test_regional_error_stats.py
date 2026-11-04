"""CPU contracts and a small, separately invoked real-CUDA correctness fixture."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.model import contribution_rasterizer as raster
from pipeline.model import contribution_renderer as renderer_module



class _ObservedCPU:
    def rasterize_gaussians(self, *args):
        xyz = args[1]
        self.geom = torch.arange(8, dtype=torch.uint8)
        self.binning = torch.arange(4, dtype=torch.uint8)
        self.image = torch.arange(6, dtype=torch.uint8)
        return (3, torch.ones((3, 2, 3)), torch.ones(len(xyz), dtype=torch.int32),
                self.geom, self.binning, self.image, torch.ones(len(xyz), dtype=torch.int32))

    def rasterize_gaussians_backward(self, *args):
        _, xyz, _, colors, scales, rotations = args[:6]
        return (torch.ones_like(xyz), torch.ones_like(colors),
                torch.ones((len(xyz), 1)), torch.ones_like(xyz),
                torch.empty(0), torch.empty(0), torch.ones_like(scales),
                torch.ones_like(rotations))


class RegionalStatsCPU(unittest.TestCase):
    @staticmethod
    def inputs():
        n = 2
        args = {
            'means3D':torch.zeros(n, 3, requires_grad=True),
            'means2D':torch.zeros(n, 3, requires_grad=True),
            'opacities':torch.ones(n, 1, requires_grad=True),
            'colors_precomp':torch.ones(n, 3, requires_grad=True),
            'scales':torch.ones(n, 3, requires_grad=True),
            'rotations':torch.ones(n, 4, requires_grad=True),
        }
        settings = raster.GaussianRasterizationSettings(
            2, 3, 1., 1., torch.zeros(3), 1., torch.eye(4), torch.eye(4),
            0, torch.zeros(3), False, False)
        return raster.GaussianRasterizer(settings), args

    def test_sink_is_per_forward_detached_and_preserves_three_outputs_backward(self):
        backend = _ObservedCPU()
        renderer, args = self.inputs()
        sink = raster.RegionalErrorStats()

        class FakeStats:
            def accumulate(self, geom, binning, image, errors, points, instances):
                self.received = (geom, binning, image, errors, points, instances)
                return torch.tensor([4., 0.], requires_grad=True), torch.tensor([2., 0.], requires_grad=True)

        stats = FakeStats()
        with patch.object(raster, '_backend', return_value=backend), \
             patch.object(raster, '_regional_error_backend', return_value=stats):
            result = renderer(**args, stats_sink=sink)
            self.assertEqual(len(result), 3)
            error = torch.arange(6., requires_grad=True).reshape(3, 2).t()
            num, den = sink.accumulate(error)
            self.assertIs(stats.received[0], backend.geom)
            self.assertIs(stats.received[1], backend.binning)
            self.assertIs(stats.received[2], backend.image)
            self.assertTrue(stats.received[3].is_contiguous())
            self.assertFalse(stats.received[3].requires_grad)
            self.assertEqual(stats.received[4:], (2, 3))
            self.assertFalse(num.requires_grad)
            self.assertFalse(den.requires_grad)
            self.assertIsNone(sink._state)
            with self.assertRaisesRegex(RuntimeError, 'unconsumed'):
                sink.accumulate(error)
            with self.assertRaisesRegex(RuntimeError, 'already contain'):
                sink._capture(backend.geom, backend.binning, backend.image, 2, 3, 2, 3)
            result[0].sum().backward()
        for tensor in args.values():
            torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))

    def test_old_path_never_loads_stats_and_bad_shape_does_not_consume(self):
        backend = _ObservedCPU()
        renderer, args = self.inputs()
        sink = raster.RegionalErrorStats()
        with patch.object(raster, '_backend', return_value=backend), \
             patch.object(raster, '_regional_error_backend', side_effect=AssertionError('unexpected load')):
            output = renderer(**args)
            self.assertEqual(len(output), 3)
            output[0].sum().backward()
            renderer(**args, stats_sink=sink)
            with self.assertRaisesRegex(ValueError, 'shape'):
                sink.accumulate(torch.ones(3, 2))
            self.assertIsNotNone(sink._state)

    def test_renderer_packages_separate_per_view_sinks_and_clears_flags(self):
        renderer = renderer_module.ContributionRenderer.__new__(renderer_module.ContributionRenderer)
        torch.nn.Module.__init__(renderer)
        renderer.collect_contributions = True
        renderer.collect_error_stats = True
        native = SimpleNamespace(raster_settings='settings')

        class FakeRasterizer:
            def __init__(self, settings):
                self.settings = settings

            def __call__(self, **kwargs):
                sink = kwargs['stats_sink']
                if sink is not None:
                    buffer = torch.empty(0, dtype=torch.uint8)
                    sink._capture(buffer, buffer, buffer, 2, 0, 2, 3)
                return torch.ones(3,2,3), torch.ones(2), torch.tensor([1,0])

        def base_forward(obj, *args, **kwargs):
            image, radii = obj.create_rasterizer(None, None, 1.)(means3D=None)
            return {'render':image, 'radii':radii}

        with patch.object(renderer_module.Renderer, 'create_rasterizer', return_value=native), \
             patch.object(renderer_module.Renderer, 'forward', base_forward), \
             patch.object(renderer_module, 'GaussianRasterizer', FakeRasterizer):
            first = renderer(None, None)
            second = renderer(None, None)
            self.assertIsNot(first['regional_error_stats'], second['regional_error_stats'])
            self.assertEqual(first['motion_contribution_filter'].tolist(), [True, False])
            self.assertEqual(second['motion_contribution_filter'].tolist(), [True, False])
            renderer.collect_error_stats = False
            third = renderer(None, None)
            self.assertNotIn('regional_error_stats', third)
            self.assertIsNone(renderer.last_error_stats)


@unittest.skipUnless(torch.cuda.is_available(), 'Requires explicitly scheduled CUDA fixture')
class RegionalStatsCUDA(unittest.TestCase):
    """Compare against actual observed-rasterizer one-hot colour weights.

    Extra reference renders are confined to this fixture, not training.
    """
    def setUp(self):
        from libgs.data.utils.graphics import getProjectionMatrix
        self.settings = raster.GaussianRasterizationSettings(
            33, 33, math.tan(.5), math.tan(.5), torch.zeros(3, device='cuda'),
            1., torch.eye(4, device='cuda'),
            getProjectionMatrix(.01, 100., 1., 1.).transpose(0, 1).cuda(),
            0, torch.zeros(3, device='cuda'), False, False)
        self.rasterizer = raster.GaussianRasterizer(self.settings)

    @staticmethod
    def scene():
        device = 'cuda'
        xyz = torch.tensor([[0.,0.,1.8], [0.,0.,1.9], [0.,0.,2.0],
            [0.,0.,2.1], [0.,0.,2.2], [.25,0.,2.0], [.25,.12,2.0],
            [100.,0.,2.]], device=device)
        n = len(xyz)
        return dict(means3D=xyz, means2D=torch.zeros_like(xyz),
            opacities=torch.tensor([.99,.99,.99,.99,.7,.003,.6,.8], device=device)[:, None],
            colors_precomp=torch.linspace(.1,.9,n*3,device=device).reshape(n,3),
            scales=torch.tensor([[.22]*3]*4 + [[.0001]*3, [.1]*3, [.12]*3, [.1]*3],device=device),
            rotations=torch.tensor([[1.,0.,0.,0.]]*n,device=device))

    def test_real_forward_weights_occlusion_cutoffs_and_backward(self):
        original = self.scene()
        a = {k:v.detach().clone().requires_grad_(True) for k,v in original.items()}
        b = {k:v.detach().clone().requires_grad_(True) for k,v in original.items()}
        sink = raster.RegionalErrorStats()
        image, radii, mask = self.rasterizer(**a, stats_sink=sink)
        ordinary_image, ordinary_radii, ordinary_mask = self.rasterizer(**b)
        self.assertTrue(torch.equal(image, ordinary_image))
        self.assertTrue(torch.equal(radii, ordinary_radii))
        self.assertTrue(torch.equal(mask, ordinary_mask))
        error = torch.linspace(.05, 1., 33*33, device='cuda').reshape(33,33)
        num, den = sink.accumulate(error)
        self.assertEqual(tuple(num.shape), (len(a['means3D']),))
        self.assertFalse(num.requires_grad)
        self.assertFalse(den.requires_grad)

        # Colour is linear in each visible T*alpha. A one-hot colour render
        # independently exposes this exact per-pixel coefficient.
        weights = []
        for index in range(len(a['means3D'])):
            colors = torch.zeros_like(original['colors_precomp'])
            colors[index, 0] = 1.
            kwargs = {**original, 'colors_precomp':colors}
            weights.append(self.rasterizer(**kwargs)[0][0].detach())
        weights = torch.stack(weights)
        reference_den = weights.sum((1,2))
        reference_num = (weights * error).sum((1,2))
        torch.testing.assert_close(den, reference_den, rtol=5e-5, atol=3e-5)
        torch.testing.assert_close(num, reference_num, rtol=5e-5, atol=3e-5)
        # Behind the highly opaque front, the threshold-causing entry itself
        # is not blended. Other pixels still exercise overlapping contributors.
        self.assertGreater(float(weights[0,16,16]), .98)
        self.assertEqual(float(weights[1,16,16]), 0.)
        self.assertGreater(int(radii[4]), 0)
        self.assertEqual(float(den[4]), 0.)  # fully hidden small rear splat
        self.assertEqual(float(den[5]), 0.)  # alpha < 1/255
        self.assertEqual(float(den[7]), 0.)  # outside view

        grad_weight = torch.linspace(.1, 1., image.numel(), device='cuda').reshape_as(image)
        (image.square()*grad_weight).sum().backward()
        (ordinary_image.square()*grad_weight).sum().backward()
        for name in a:
            torch.testing.assert_close(a[name].grad, b[name].grad, rtol=2e-5, atol=2e-5)

    def test_empty_and_fully_culled_views(self):
        original = self.scene()
        for args in ({k:v[:0].contiguous() for k,v in original.items()},
                     {k:v[-1:].contiguous() for k,v in original.items()}):
            sink = raster.RegionalErrorStats()
            self.rasterizer(**args, stats_sink=sink)
            num, den = sink.accumulate(torch.ones(33,33,device='cuda'))
            self.assertEqual(num.numel(), len(args['means3D']))
            self.assertEqual(float(num.sum()), 0.)
            self.assertEqual(float(den.sum()), 0.)


class CompatibilityTests(unittest.TestCase):
    def test_legacy_three_outputs_observed_semantics_and_backward_arity_stay_unchanged(self):
        count = 2
        xyz = torch.zeros(count, 3, requires_grad=True)
        screen = torch.zeros(count, 3, requires_grad=True)
        color = torch.full((count, 3), .5, requires_grad=True)
        alpha = torch.full((count, 1), .3, requires_grad=True)
        scale = torch.ones(count, 3, requires_grad=True)
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1).requires_grad_()
        settings = raster.GaussianRasterizationSettings(1, 1, 1., 1., torch.zeros(3),
            1., torch.eye(4), torch.eye(4), 0, torch.zeros(3), False, False)
        class Backend:
            def rasterize_gaussians(self, *args):
                return (2, torch.full((3, 1, 1), .5), torch.tensor([2, 0], dtype=torch.int32),
                        torch.empty(0), torch.empty(0), torch.empty(0), torch.tensor([1, 0], dtype=torch.int32))
            def rasterize_gaussians_backward(self, *args):
                return (torch.ones_like(screen), torch.ones_like(color), torch.ones_like(alpha),
                        torch.ones_like(xyz), torch.empty(0), torch.empty(0), torch.ones_like(scale), torch.ones_like(q))
        with patch.object(raster, '_backend', return_value=Backend()):
            output = raster.GaussianRasterizer(settings)(xyz, screen, alpha, colors_precomp=color, scales=scale, rotations=q)
            self.assertEqual(len(output), 3)
            self.assertEqual(output[2].tolist(), [1, 0])
            output[0].sum().backward()
        for parameter in (xyz, screen, color, alpha, scale, q):
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


if __name__ == "__main__":
    unittest.main()
