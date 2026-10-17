"""Newborn geometry should respect the oriented ellipsoid of its source GS."""

import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


GAUSSIAN = importlib.import_module('pipeline.model.gaussian')


class BirthSourceCovarianceTests(unittest.TestCase):
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
