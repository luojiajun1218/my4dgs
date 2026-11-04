"""Clone birth geometry and checkpoint compatibility."""
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch

GAUSSIAN = importlib.import_module('pipeline.model.gaussian')


class BirthSourceCovarianceTests(unittest.TestCase):
    def test_clone_returns_exact_sources_with_budget_ties_and_rejected_rows(self):
        for budget, expected in ((None, [1, 2, 5]), (0, []), (1, [1]),
                                 (2, [1, 2]), (5, [1, 2, 5])):
            with self.subTest(budget=budget):
                rows = torch.arange(6, dtype=torch.float32)
                scale = torch.tensor([[.1]] * 3 + [[1.], [.1], [.1]]).expand(-1, 3)
                source = SimpleNamespace(_xyz=rows[:, None].expand(-1, 3),
                    xyz_gradient_accum=torch.tensor([[.2], [.8], [.8], [.9],
                                                    [float('nan')], [.7]]),
                    denom=torch.ones(6, 1), get_scaling=scale, percent_dense=.5,
                    _rotation=torch.tensor([[1., 0., 0., 0.]] * 6),
                    _scaling=scale.log(), _opacity=rows[:, None],
                    _features_dc=rows[:, None, None].expand(-1, 1, 3),
                    _features_rest=rows[:, None, None].expand(-1, 3, 3))
                captured = []
                source.densification_postfix = lambda *values: captured.append(values)
                with patch('torch.normal', side_effect=lambda mean, std: mean.clone()):
                    report, selected = GAUSSIAN.GaussianModel.adaptive_densify_and_clone(
                        source, .5, 1., max_new_gaussians=budget, return_source_rows=True)
                torch.testing.assert_close(selected, torch.tensor(expected, dtype=torch.long))
                self.assertEqual(report, dict(eligible=3, added=len(expected),
                                              blocked=3 - len(expected)))
                # Geometry and appearance rows must agree with the returned lineage.
                torch.testing.assert_close(captured[0][0], source._xyz[selected])
                torch.testing.assert_close(captured[0][1], source._features_dc[selected])
                torch.testing.assert_close(captured[0][2], source._features_rest[selected])

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA newborn splits')
    def test_split_budget_preserves_child_order_and_source_lineage(self):
        fields = ('_xyz', '_features_dc', '_features_rest', '_opacity',
                  '_scaling', '_rotation')
        def prune(model, remove):
            for field in fields:
                setattr(model, field, getattr(model, field)[~remove])

        cases = ((2, None, [1, 2, 3, 4]), (2, 6, []), (2, 7, [1]),
                 (2, 9, [1, 2, 3]), (3, 8, [1]))
        for children, budget, selected_rows in cases:
            with self.subTest(children=children, budget=budget):
                model = GAUSSIAN.BirthGaussianModel(1)
                ids = torch.tensor([11, 22, 33, 44, 55, 66], device='cuda')
                model._xyz = ids[:, None].expand(-1, 3).float()
                model._features_dc = ids[:, None, None].expand(-1, 1, 3).float()
                model._features_rest = ids[:, None, None].expand(-1, 3, 3).float()
                model._opacity = ids[:, None].float()
                model._rotation = model._xyz.new_tensor([[1., 0., 0., 0.]] * 6)
                model._scaling = model._xyz.new_tensor(
                    [[.1], [.5], [.6], [.3], [.4], [.5]]).expand(-1, 3).log()
                model.percent_dense = .2
                model.source_rows = ids.clone()
                def postfix(*values):
                    for field, value in zip(fields, values):
                        setattr(model, field, torch.cat((getattr(model, field), value)))
                model.densification_postfix = postfix
                grads = model._xyz.new_tensor([[.9], [.8], [.8], [.7], [.6], [.4]])
                with patch.object(GAUSSIAN.BaseGaussianModel, 'prune_points', new=prune), \
                        patch('torch.normal', side_effect=lambda mean, std: torch.zeros_like(mean)):
                    report = model.adaptive_densify_and_split(
                        grads, .5, 1., children, max_points=budget)
                selected = torch.tensor(selected_rows, device='cuda', dtype=torch.long)
                survivors = torch.tensor([i for i in range(6) if i not in selected_rows],
                                         device='cuda', dtype=torch.long)
                expected = torch.cat((ids[survivors], ids[selected].repeat(children)))
                torch.testing.assert_close(model.source_rows, expected)
                torch.testing.assert_close(model._xyz[:, 0], expected.float())
                torch.testing.assert_close(model._features_dc[:, 0, 0], expected.float())
                self.assertEqual(report, dict(eligible=4,
                    added=len(selected_rows) * (children - 1), blocked=4 - len(selected_rows)))

    def test_frozen_newborn_orientation_inherits_raw_source_without_gradient(self):
        from pipeline.module import BirthConfig
        from pipeline.model.gaussian import GaussianModel
        from pipeline.model.birth_quantization import (
            QuantizedNewGaussianView, RAW_ATTRIBUTES, materialize_quantized_gaussians)
        import test_codec as replacement
        reference = replacement.reference_raw()
        reference['rotation'][1] = torch.tensor([2., .3, -.2, .1])
        source = torch.tensor([1, 3])
        model = GaussianModel(1)
        for name, attr in zip(replacement.RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(reference[name][source].clone()))
        with torch.no_grad():
            model._rotation.fill_(.2)
        view = QuantizedNewGaussianView(model, BirthConfig(rotation_updates=False),
                                        2., reference, source)
        (view.get_xyz.sum() + view.get_rotation.sum()).backward()
        self.assertIsNone(model._rotation.grad)
        self.assertIsNotNone(model._xyz.grad)
        self.assertTrue(bool(view.symbols['rotation'].eq(0).all()))
        raw, _ = materialize_quantized_gaussians(reference, view)
        torch.testing.assert_close(raw['rotation'][-2:].view(torch.int32),
                                   reference['rotation'][source].view(torch.int32), rtol=0, atol=0)

    def test_rotated_source_controls_sampling_and_newborn_orientation(self):
        captured = {}
        q = torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5]])
        scale = torch.tensor([[.1, .01, .001]])
        source = SimpleNamespace(_xyz=torch.zeros(1, 3), xyz_gradient_accum=torch.ones(1, 1),
            denom=torch.ones(1, 1), get_scaling=scale, percent_dense=.5,
            _rotation=q, _scaling=scale.log(), _opacity=torch.zeros(1, 1),
            _features_dc=torch.ones(1, 1, 3), _features_rest=torch.zeros(1, 3, 3))

        def postfix(*values):
            captured['values'] = values

        source.densification_postfix = postfix
        with patch('torch.normal', return_value=torch.tensor([[.2, 0., 0.]])):
            result = GAUSSIAN.GaussianModel.adaptive_densify_and_clone(
                source, .1, 1., sigma_scale=2., source_covariance_init=True)
        self.assertEqual(result['added'], 1)
        xyz, dc, rest, opacity, logscale, rotation = captured['values']
        torch.testing.assert_close(xyz, torch.tensor([[0., .2, 0.]]), rtol=0, atol=1e-7)
        torch.testing.assert_close(rotation, q, rtol=0, atol=0)
        torch.testing.assert_close(logscale, scale.log(), rtol=0, atol=0)
        torch.testing.assert_close(source._xyz, torch.zeros(1, 3), rtol=0, atol=0)
        torch.testing.assert_close(source._opacity, torch.zeros(1, 1), rtol=0, atol=0)

    def test_refinement_birth_allocation_checkpoint_cannot_silently_resume(self):
        from unittest.mock import Mock
        module = importlib.import_module('pipeline.module')
        harness = SimpleNamespace(gaussians=SimpleNamespace(restore=Mock()),
            renderer=SimpleNamespace(), config=SimpleNamespace(gaussian=object(),
                rigid_tree=SimpleNamespace(target_update_bytes=0),
                birth=module.BirthConfig()))
        with self.assertRaisesRegex(ValueError, 'newborn allocation.*replay-only'):
            module.Module.on_load_checkpoint(harness, dict(gaussians=object(),
                rigid_tree_version=1, current_frame=1,
                birth_allocation='clone_small_refine_large_v1'))


from pipeline.model.birth_quantization import (
    quantized_symbols, reanchor_reconstruction_tolerance,
)


class ReanchorPrecisionTest(unittest.TestCase):
    step = 6.646713829040529e-5

    def check_real_coordinate_pair(self, device):
        # Coordinates sampled from the independently decoded Coffee frame 26.
        value = torch.tensor([2.652822256088257], device=device)
        predictor = torch.tensor([-31.709793090820312], device=device)
        decoded = predictor + quantized_symbols(value-predictor, self.step)*self.step
        error = (decoded.double()-value.double()).abs()
        self.assertGreater(float(error.max()), self.step*.501+1e-6)
        bound = reanchor_reconstruction_tolerance(value, predictor, self.step)
        self.assertTrue(bool((error <= bound).all()))
        # A real misplaced row remains a failure; the check is not disabled.
        corrupted = decoded + 32*self.step
        self.assertTrue(bool(((corrupted.double()-value.double()).abs() > bound).all()))

    def test_real_coordinate_pair_cpu(self):
        self.check_real_coordinate_pair('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_real_coordinate_pair_cuda(self):
        self.check_real_coordinate_pair('cuda')

    def test_distant_and_nearby_references_remain_bounded(self):
        generator = torch.Generator().manual_seed(27)
        for dtype in (torch.float32, torch.float64):
            values = torch.randn(20000, 3, generator=generator, dtype=dtype)*500
            references = torch.randn(20000, 3, generator=generator, dtype=dtype)*500
            decoded = references+quantized_symbols(values-references, self.step)*self.step
            bound = reanchor_reconstruction_tolerance(values, references, self.step)
            self.assertTrue(bool(((decoded.double()-values.double()).abs() <= bound).all()))

    def test_near_origin_still_rejects_more_than_half_bin(self):
        value = torch.tensor([0., .005, -.005])
        reference = torch.zeros_like(value)
        bound = reanchor_reconstruction_tolerance(value, reference, .001)
        self.assertTrue(bool((bound < .000503).all()))
        self.assertTrue(bool(((value.double()+.00075-value.double()).abs() > bound).all()))

    def test_invalid_values_and_steps_are_rejected(self):
        good = torch.tensor([1.])
        for step in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                reanchor_reconstruction_tolerance(good, good, step)
        for bad in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                reanchor_reconstruction_tolerance(torch.tensor([bad]), good, .001)
        with self.assertRaises(ValueError):
            reanchor_reconstruction_tolerance(good, torch.zeros(2), .001)
