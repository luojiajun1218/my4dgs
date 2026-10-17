"""Newborn geometry should respect the oriented ellipsoid of its source GS."""

import importlib
from types import SimpleNamespace
import unittest
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
        import test_gaussian_replacement as replacement
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
            renderer=SimpleNamespace(), config=SimpleNamespace(gaussian=object()))
        with self.assertRaisesRegex(ValueError, 'newborn allocation.*replay-only'):
            module.Module.on_load_checkpoint(harness, dict(gaussians=object(),
                rigid_tree_version=1, current_frame=1,
                birth_allocation='clone_small_refine_large_v1'))


if __name__ == '__main__':
    unittest.main()
