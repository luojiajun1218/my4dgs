"""First-frame references, Gaussian birth lineage, quantized attributes, and terminal optimizer ordering."""
import importlib
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.model.birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS, QuantizedNewGaussianView, materialize_quantized_gaussians, model_raw
from pipeline.module import BirthConfig, Module


MODULE = importlib.import_module('pipeline.module')


TRAINER = importlib.import_module('pipeline.trainer')


GAUSSIAN = importlib.import_module('pipeline.model.gaussian')


REPLAY = importlib.import_module('scripts.replay_rigid_tree')


class FakeNativeGaussians:
    def __init__(self):
        self.prune_calls = 0
        self.restore(self.raw(), None)

    @staticmethod
    def raw():
        centers = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        return dict(_xyz=torch.cat((centers, torch.zeros(4, 2)), dim=1),
                    _rotation=torch.tensor([[1., 0., 0., 0.]] * 4),
                    _scaling=torch.zeros(4, 3),
                    _opacity=torch.logit(torch.tensor([[.01], [.2], [.005], [.5]])),
                    _features_dc=centers.reshape(4, 1, 1).expand(4, 1, 3).clone(),
                    _features_rest=torch.zeros(4, 0, 3))

    def capture(self):
        return {name: getattr(self, name).detach().clone()
                for name in MODULE.RAW_ATTRIBUTES}

    def restore(self, packet, config):
        for name in MODULE.RAW_ATTRIBUTES:
            setattr(self, name, torch.nn.Parameter(packet[name].detach().clone()))

    def prune_points(self, mask):
        self.prune_calls += 1
        for name in MODULE.RAW_ATTRIBUTES:
            value = getattr(self, name)
            setattr(self, name, torch.nn.Parameter(value.detach()[~mask].clone()))


def fake_module(config, gaussians, frame):
    holder = SimpleNamespace(config=config, gaussians=gaussians,
        renderer=SimpleNamespace(gaussians=None), current_frame=frame,
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=15000)),
        _freeze_base=lambda: [getattr(gaussians, name).requires_grad_(False)
                              for name in MODULE.RAW_ATTRIBUTES])
    return holder


class NativeReferencePruningTests(unittest.TestCase):
    def test_removed_color_training_checkpoint_is_replay_only(self):
        config = MODULE.ModuleConfig()
        saved = asdict(config)
        saved['feature_dc_update'] = dict(enabled=True, sharing='spatial_region')
        gaussians = FakeNativeGaussians()
        active = fake_module(config, gaussians, frame=1)
        with self.assertRaisesRegex(ValueError, 'DC color-update checkpoints are replay-only'):
            MODULE.Module.on_load_checkpoint(active, dict(
                gaussians=gaussians.capture(), rigid_tree_version=1,
                rigid_tree_config=asdict(config.rigid_tree),
                rigid_tree_module_config=saved))

    def test_threshold_validation_and_full_delete_guard(self):
        for bad in (-.01, 1., float('nan'), float('inf'), True, '0.015'):
            with self.subTest(threshold=bad), self.assertRaisesRegex(
                    ValueError, 'initial_opacity_threshold'):
                MODULE.validate_initial_opacity_threshold(bad)
        original = FakeNativeGaussians()
        with self.assertRaisesRegex(ValueError, 'every Gaussian'):
            MODULE.prune_native_initial_gaussians(original, .9)
        self.assertEqual(original.prune_calls, 0)
        self.assertEqual(len(original._xyz), 4)
        unpruned = MODULE.prune_native_initial_gaussians(original, 0.)
        self.assertEqual((unpruned['original_count'], unpruned['retained_count'],
                          unpruned['removed_count']), (4, 4, 0))
        self.assertEqual(original.prune_calls, 0)

    def test_native_prune_keeps_deterministic_rows_and_own_resume_does_not_prune(self):
        config = MODULE.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        MODULE.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        torch.testing.assert_close(native._xyz[:, 0], torch.tensor([1., 3.]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(native._features_dc[:, 0, 0],
                                   torch.tensor([1., 3.]), rtol=0, atol=0)
        self.assertEqual(native.prune_calls, 1)
        self.assertEqual(active.reference_count, 2)
        self.assertEqual(active.initial_reference, dict(
            source='native_frame0_checkpoint', threshold=.015,
            original_count=4, retained_count=2, removed_count=2))
        self.assertTrue(all(not getattr(native, name).requires_grad
                            for name in MODULE.RAW_ATTRIBUTES))

        own = dict(gaussians=native.capture(), rigid_tree_version=1,
            rigid_tree_config=asdict(config.rigid_tree),
            rigid_tree_module_config=asdict(config),
            rigid_tree_initial_reference=active.initial_reference,
            rigid_tree_frame=dict(prepared=False, finalized=True,
                motion_committed=False, reference_count=2, report=None,
                birth_control={}), births=None)
        restored = FakeNativeGaussians()
        resume = fake_module(config, restored, frame=1)
        MODULE.Module.on_load_checkpoint(resume, own)
        self.assertEqual(restored.prune_calls, 0)
        self.assertEqual(len(restored._xyz), 2)
        self.assertEqual(resume.initial_reference, active.initial_reference)
        self.assertEqual(resume.reference_count, 2)

    def test_pruned_reference_layout_and_frame_zero_replay_have_same_count(self):
        config = MODULE.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        MODULE.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        with tempfile.TemporaryDirectory() as directory:
            worker = SimpleNamespace(output_dir=Path(directory), module=active,
                datamodule=SimpleNamespace(cameras_extent=1.))
            active.motion = importlib.import_module('pipeline.model.spatial_gated_motion').SpatialGatedMotion(config.rigid_tree)
            TRAINER.Trainer._write_reference(worker, frame_index=0)
            with np.load(worker.decoder_reference_path, allow_pickle=False) as packet:
                self.assertEqual(int(packet['initial_original_count']), 4)
                self.assertEqual(int(packet['initial_retained_count']), 2)
                self.assertEqual(int(packet['initial_removed_count']), 2)
                self.assertEqual(float(packet['initial_opacity_threshold']), .015)
                self.assertEqual(len(packet['owner']), 2)
            state, raw = REPLAY.replay(worker.decoder_reference_path.parent,
                                       worker.decoder_reference_path, through=0)
            self.assertEqual(len(state.xyz), active.reference_count)
            torch.testing.assert_close(raw['xyz'][:, 0], torch.tensor([1., 3.]),
                                       rtol=0, atol=0)
            rebuilt = active.motion.codec_previous_state
            torch.testing.assert_close(state.owner, rebuilt.owner, rtol=0, atol=0)
            torch.testing.assert_close(state.parents, rebuilt.parents, rtol=0, atol=0)


class TerminalBirthTests(unittest.TestCase):
    def test_final_optimizer_update_precedes_prune_without_new_split(self):
        quantization = importlib.import_module('pipeline.model.birth_quantization')
        events = []
        births = GAUSSIAN.BirthGaussianModel(0)
        fields = dict(_xyz=torch.zeros(2, 3),
                      _rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                      _scaling=torch.zeros(2, 3),
                      _opacity=torch.logit(torch.tensor([[.005], [.002]])),
                      _features_dc=torch.zeros(2, 1, 3),
                      _features_rest=torch.zeros(2, 0, 3))
        for name, value in fields.items():
            setattr(births, name, torch.nn.Parameter(value))
        births.max_radii2D = torch.zeros(2)
        births.xyz_gradient_accum = torch.zeros(2, 1)
        births.denom = torch.zeros(2, 1)
        births.source_rows = torch.tensor([0, 1])
        groups = [(key, attr) for key, attr in (
            ('xyz', '_xyz'), ('f_dc', '_features_dc'),
            ('f_rest', '_features_rest'), ('opacity', '_opacity'),
            ('scaling', '_scaling'), ('rotation', '_rotation'))]
        births.optimizer = torch.optim.Adam([
            dict(name=key, params=[getattr(births, attr)])
            for key, attr in groups], lr=1.)
        births._opacity.grad = torch.tensor([[-1.], [0.]])
        original_step = births.optimizer.step
        original_zero = births.optimizer.zero_grad
        original_prune = births.prune_points
        def step():
            events.append('step')
            original_step()
        def zero_grad(**kwargs):
            events.append('zero')
            original_zero(**kwargs)
        def prune_points(remove):
            events.append('prune')
            original_prune(remove)
        births.optimizer.step = step
        births.optimizer.zero_grad = zero_grad
        births.prune_points = prune_points
        births.add_densification_stats = lambda *args, **kwargs: None
        def fail_densify(*args, **kwargs):
            raise AssertionError('Terminal step must not split')
        births.adaptive_densify_and_prune = fail_densify
        config = MODULE.ModuleConfig()
        config.rigid_tree.motion_steps = 100
        config.rigid_tree.birth_steps = 200
        config.birth.densify_interval = 40  # Final step is a former split point.
        active = SimpleNamespace(config=config, motion_committed=True,
            births=births, gaussians=SimpleNamespace(_xyz=torch.zeros(2, 3)),
            trainer=SimpleNamespace(global_step=300), motion_steps=100,
            total_steps=300, datamodule=SimpleNamespace(cameras_extent=20.),
            birth_control=dict(pruned=0, split_added=0),
            finalize_frame=lambda: events.append('finalize'))
        active._prune_terminal_births = lambda: MODULE.Module._prune_terminal_births(active)
        active.make_birth_view = lambda: SimpleNamespace(get_opacity=births.get_opacity)
        package = dict(visibility_filter=torch.tensor([False, False, True, True]),
                       radii=torch.zeros(4), viewspace_points=None)
        MODULE.Module.post_training_step(active, package)
        self.assertEqual(events, ['step', 'zero', 'prune', 'finalize'])
        self.assertEqual(active.birth_control['pruned'], 1)
        self.assertEqual(active.birth_control['split_added'], 0)
        torch.testing.assert_close(births.source_rows, torch.tensor([0]),
                                   rtol=0, atol=0)
        self.assertEqual(len(births._xyz), 1)
        reference = dict(xyz=torch.zeros(2, 3),
            rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
            logscale=torch.zeros(2, 3), opacity=torch.zeros(2, 1),
            sh_dc=torch.zeros(2, 1, 3), sh_rest=torch.zeros(2, 0, 3))
        view = quantization.QuantizedNewGaussianView(
            births, config.birth, 20., reference, source_rows=births.source_rows)
        materialized, sources = quantization.materialize_quantized_gaussians(
            reference, view)
        self.assertEqual(len(materialized['xyz']), 3)
        torch.testing.assert_close(sources, torch.tensor([0]), rtol=0, atol=0)


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
        from _stream_fixtures import reference_raw
        from pipeline.model.birth_quantization import RAW_FIELDS
        reference = reference_raw()
        reference['rotation'][1] = torch.tensor([2., .3, -.2, .1])
        source = torch.tensor([1, 3])
        model = GaussianModel(1)
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
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
                opacity_update=None,
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
