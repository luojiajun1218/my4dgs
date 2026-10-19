"""Constant grids and borrowed birth inputs must preserve gradients and ownership."""
from types import SimpleNamespace
import unittest

import torch

from pipeline.model.birth_quantization import (
    RAW_ATTRIBUTES, RAW_FIELDS, QuantizedNewGaussianView,
    materialize_quantized_gaussians, model_raw)
from pipeline.model.rigid_residual_tree import MotionBase, RigidTreeConfig
from pipeline.module import BirthConfig, Module


class QuantizationGridReuseTests(unittest.TestCase):
    def model(self):
        model = MotionBase(RigidTreeConfig())
        model.base_xyz = torch.zeros(4, 3)
        model.extent = 6.5
        return model

    def test_repeated_grids_preserve_quantized_values_and_separate_backward_passes(self):
        model = self.model()
        expected = model.base_xyz.new_tensor(
            [model.config.motion_xyz_step_extent * model.extent] * 3 +
            [model.config.motion_rotation_step_rad] * 3)
        first = model.steps
        self.assertIs(first, model.steps)
        self.assertIs(first, model.gs_steps)
        for _ in range(2):
            value = torch.tensor([.004, -.002, .008, .01, -.02, .03], requires_grad=True)
            normalized = value / model.steps
            legacy = value / expected
            torch.testing.assert_close(normalized, legacy, rtol=0, atol=0)
            normalized.sum().backward()
            torch.testing.assert_close(value.grad, 1 / expected, rtol=0, atol=0)

    def test_config_extent_dtype_and_mutation_cannot_leave_stale_constants(self):
        model = self.model()
        model.steps.fill_(0)
        self.assertTrue(bool(model.steps.gt(0).all()))
        model.config.motion_xyz_step_extent *= 2
        torch.testing.assert_close(model.steps[:3], 2 * model.gs_steps[:3], rtol=0, atol=0)
        model.extent *= 2
        self.assertAlmostEqual(float(model.steps[0]),
            model.config.motion_xyz_step_extent * model.extent, places=9)
        model.base_xyz = model.base_xyz.double()
        self.assertEqual(model.steps.dtype, torch.float64)

    def test_inference_first_does_not_leak_an_inference_tensor_into_training(self):
        model = self.model()
        with torch.inference_mode():
            model.steps
        grid = model.steps
        value = torch.ones(6, requires_grad=True)
        (value * grid).sum().backward()
        torch.testing.assert_close(value.grad, grid, rtol=0, atol=0)


class BorrowedBirthReferenceTests(unittest.TestCase):
    @staticmethod
    def model(count=32):
        generator = torch.Generator().manual_seed(928)
        shapes = ((3,), (4,), (3,), (1,), (1, 3), (3, 3))
        return SimpleNamespace(**{
            name: torch.nn.Parameter(torch.randn(count, *shape, generator=generator))
            for name, shape in zip(RAW_ATTRIBUTES, shapes)})

    def test_default_snapshot_owns_data_and_materialization_still_owns_borrowed_input(self):
        base = self.model()
        snapshot, borrowed = model_raw(base), model_raw(base, clone=False)
        result, sources = materialize_quantized_gaussians(borrowed)
        self.assertEqual(len(sources), 0)
        for name, attribute in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            original = getattr(base, attribute)
            self.assertFalse(borrowed[name].requires_grad)
            self.assertEqual(borrowed[name].data_ptr(), original.data_ptr())
            self.assertNotEqual(snapshot[name].data_ptr(), original.data_ptr())
            self.assertNotEqual(result[name].data_ptr(), original.data_ptr())
            with torch.no_grad():
                original.add_(1)
            torch.testing.assert_close(result[name], snapshot[name], rtol=0, atol=0)

    def test_source_gather_preserves_symbols_outputs_gradients_and_predictor_lifetime(self):
        for rotation_updates in (False, True):
            base = self.model()
            newborn = self.model(4)
            source = torch.tensor([1, 7, 7, 29])
            config = BirthConfig(rotation_updates=rotation_updates)
            owned = QuantizedNewGaussianView(newborn, config, 6.5, model_raw(base), source)
            borrowed = QuantizedNewGaussianView(newborn, config, 6.5,
                                               model_raw(base, clone=False), source)
            for name in RAW_FIELDS:
                torch.testing.assert_close(borrowed.symbols[name], owned.symbols[name], rtol=0, atol=0)
                torch.testing.assert_close(borrowed.raw_value(name), owned.raw_value(name), rtol=0, atol=0)
            for view in (owned, borrowed):
                sum(view.raw_value(name).square().sum() for name in RAW_FIELDS).backward()
                gradients = [getattr(newborn, key).grad for key in RAW_ATTRIBUTES]
                if view is owned:
                    expected = [None if g is None else g.clone() for g in gradients]
                else:
                    for actual, previous in zip(gradients, expected):
                        if previous is None:
                            self.assertIsNone(actual)
                        else:
                            torch.testing.assert_close(actual, previous, rtol=0, atol=0)
                for key in RAW_ATTRIBUTES:
                    getattr(newborn, key).grad = None
            before = {name: borrowed.raw_value(name).detach().clone() for name in RAW_FIELDS}
            with torch.no_grad():
                for key in RAW_ATTRIBUTES:
                    getattr(base, key).add_(1)
            for name in RAW_FIELDS:
                torch.testing.assert_close(borrowed.raw_value(name), before[name], rtol=0, atol=0)
            self.assertTrue(all(getattr(base, key).grad is None for key in RAW_ATTRIBUTES))


class BirthUpdateOrderingTests(unittest.TestCase):
    def test_densification_precedes_step_and_terminal_pruning_follows_last_step(self):
        for step, expected in (
                (250, ['stats', 'densify', 'step', 'zero']),
                (251, ['stats', 'step', 'zero']),
                (300, ['stats', 'step', 'zero', 'prune', 'finalize'])):
            events = []
            def densify(*args, **kwargs):
                events.append('densify')
                return dict(added=1, pruned=2)
            worker = SimpleNamespace(
                trainer=SimpleNamespace(global_step=step), motion_committed=True,
                motion_steps=200, total_steps=300,
                config=SimpleNamespace(birth=BirthConfig(densify_interval=50)),
                datamodule=SimpleNamespace(cameras_extent=6.5),
                gaussians=SimpleNamespace(_xyz=torch.zeros(3, 3)),
                births=SimpleNamespace(_xyz=torch.zeros(2, 3), max_radii2D=torch.zeros(2),
                    add_densification_stats=lambda *args, **kw: events.append('stats'),
                    adaptive_densify_and_prune=densify,
                    optimizer=SimpleNamespace(step=lambda: events.append('step'),
                        zero_grad=lambda **kw: events.append('zero'))),
                birth_control=dict(split_added=0, pruned=0),
                opacity_update=None, feature_dc_update=None,
                _prune_terminal_births=lambda: events.append('prune') or 3,
                finalize_frame=lambda: events.append('finalize'))
            package=dict(visibility_filter=torch.ones(5, dtype=torch.bool),
                radii=torch.ones(5), viewspace_points=torch.zeros(5, 3))
            Module.post_training_step(worker, package)
            self.assertEqual(events, expected)
            self.assertEqual(worker.birth_control['pruned'], 3 if step == 300 else
                             (2 if step == 250 else 0))


if __name__ == '__main__':
    unittest.main()
