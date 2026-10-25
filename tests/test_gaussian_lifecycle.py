"""Gaussian birth lineage, decoder references, and quantized attribute lifecycle."""
import importlib
import math
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.model.birth_quantization import (
    RAW_ATTRIBUTES, RAW_FIELDS, QuantizedNewGaussianView,
    materialize_quantized_gaussians, model_raw,
)
from pipeline.module import BirthConfig, GaussianConfig, Module, ModuleConfig
from pipeline.model.gaussian import BirthGaussianModel, GaussianModel
from pipeline.model.rigid_residual_tree import quaternion_rotate

MODULE = importlib.import_module('pipeline.module')
TRAINER = importlib.import_module('pipeline.trainer')
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
    return SimpleNamespace(config=config, gaussians=gaussians,
        renderer=SimpleNamespace(gaussians=None), current_frame=frame,
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=15000)),
        _freeze_base=lambda: [getattr(gaussians, name).requires_grad_(False)
                              for name in MODULE.RAW_ATTRIBUTES])


class BirthPositionStepTests(unittest.TestCase):
    def test_adam_birth_step_respects_rotated_axes_and_codec_grid_floor(self):
        for cap in (None, .5):
            with self.subTest(cap=cap):
                births = BirthGaussianModel(1)
                births._xyz = torch.nn.Parameter(torch.zeros(2, 3))
                births._scaling = torch.nn.Parameter(torch.tensor([
                    [.1, .001, 1e-8], [.2, .01, .0001]]).log(), requires_grad=False)
                births._rotation = torch.nn.Parameter(torch.tensor([
                    [math.cos(math.pi/12), 0., 0., math.sin(math.pi/12)],
                    [1., 0., 0., 0.]]), requires_grad=False)
                births.optimizer = torch.optim.Adam([
                    {'params': [births._xyz], 'name': 'xyz'},
                    {'params': [births._scaling], 'name': 'scaling'},
                    {'params': [births._rotation], 'name': 'rotation'}], lr=1.)
                before = births._xyz.detach().clone()
                old_scale = births._scaling.detach().clone()
                old_rotation = births._rotation.detach().clone()
                births._xyz.grad = torch.ones_like(births._xyz)
                active = SimpleNamespace(births=births,
                    config=SimpleNamespace(birth=BirthConfig(position_step_sigma_cap=cap)),
                    datamodule=SimpleNamespace(cameras_extent=2.),
                    _configure_birth_phase=lambda step: None,
                    _birth_learning_rate_scale=lambda step: torch.ones(2))
                Module._birth_optimizer_step(active, 181)
                if cap is None:
                    torch.testing.assert_close(births._xyz, -torch.ones(2, 3))
                else:
                    inverse = old_rotation.clone(); inverse[:, 1:] *= -1
                    local = quaternion_rotate(inverse, births._xyz - before)
                    bound = (old_scale.exp() * cap).clamp_min(2e-5)
                    self.assertTrue(bool((local.abs() <= bound + 1e-7).all()))
                    self.assertAlmostEqual(float(local[0, 2].abs()), 2e-5, places=8)
                torch.testing.assert_close(births._scaling, old_scale, rtol=0, atol=0)
                torch.testing.assert_close(births._rotation, old_rotation, rtol=0, atol=0)


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

    def test_native_prune_keeps_deterministic_rows_and_resume_does_not_prune_again(self):
        config = MODULE.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        MODULE.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        torch.testing.assert_close(native._xyz[:, 0], torch.tensor([1., 3.]), rtol=0, atol=0)
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
            active.motion = importlib.import_module(
                'pipeline.model.spatial_gated_motion').SpatialGatedMotion(config.rigid_tree)
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
            torch.testing.assert_close(raw['xyz'][:, 0], torch.tensor([1., 3.]), rtol=0, atol=0)
            rebuilt = active.motion.codec_previous_state
            torch.testing.assert_close(state.owner, rebuilt.owner, rtol=0, atol=0)
            torch.testing.assert_close(state.parents, rebuilt.parents, rtol=0, atol=0)


class BirthQuantizedAttributeTests(unittest.TestCase):
    def test_frozen_newborn_orientation_inherits_raw_source_without_gradient(self):
        from _stream_fixtures import reference_raw
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

    def test_checkpoint_from_retired_birth_allocation_is_replay_only(self):
        from unittest.mock import Mock
        harness = SimpleNamespace(gaussians=SimpleNamespace(restore=Mock()),
            renderer=SimpleNamespace(), config=SimpleNamespace(gaussian=object()),
            birth_allocation='iterative_gradient_clone_v1')
        with self.assertRaisesRegex(ValueError, 'newborn allocation.*replay-only'):
            Module.on_load_checkpoint(harness, dict(gaussians=object(),
                rigid_tree_version=1, current_frame=1,
                birth_allocation='clone_small_refine_large_v1'))

    @staticmethod
    def model(count, rows, scales, config):
        xyz = torch.zeros(count, 3, device='cuda')
        xyz[:, 0] = torch.arange(count, device='cuda', dtype=torch.float32)
        rotation = torch.zeros(count, 4, device='cuda')
        rotation[:, 0] = 1
        logscale = torch.log(torch.tensor(scales, dtype=torch.float32,
                                          device='cuda')[:, None]).expand(-1, 3).clone()
        opacity = torch.logit(torch.linspace(.4, .7, count, device='cuda')[:, None])
        sh_dc = torch.arange(count, device='cuda', dtype=torch.float32)[:, None, None].expand(-1, 1, 3).clone()
        sh_rest = torch.arange(count, device='cuda', dtype=torch.float32)[:, None, None].expand(-1, 3, 3).clone()
        raw = dict(xyz=xyz, rotation=rotation, logscale=logscale,
                   opacity=opacity, sh_dc=sh_dc, sh_rest=sh_rest)
        model = GaussianModel(1)
        for field, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(raw[field].clone()))
        model.active_sh_degree = 1
        model.spatial_lr_scale = 1.
        model.training_setup(config)
        return model

    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA newborn topology operations')
    def test_window_splits_large_newborns_without_cloning_small_or_old_rows(self):
        stage2 = GaussianConfig()
        source = self.model(4, None, [.005, .15, .005, .005], stage2)
        births = BirthGaussianModel(1)
        births.initialize_from_sources(source, torch.tensor([0, 1, 2], device='cuda'),
                                       180, stage2)
        # The second row is large and splits; row 2 is not gradient eligible,
        # but its observed radius must survive the postfix call and prune it.
        births.xyz_gradient_accum[:] = torch.tensor([[1.], [1.], [0.]], device='cuda')
        births.denom[:] = 1
        births.max_radii2D[:] = torch.tensor([0., 30., 30.], device='cuda')
        source.xyz_gradient_accum[3] = 1
        source.denom[3] = 1

        config = ModuleConfig(gaussian_stage2=stage2,
            birth=BirthConfig(densify_grad_threshold=.5, densify_interval=30,
                              new_max_screen_size=20., refit_steps=30))
        module = SimpleNamespace(config=config, gaussians=source, births=births,
            datamodule=SimpleNamespace(cameras_extent=1.),
            birth_control=dict(clone_added=3, split_added=0, pruned=0),
            device=torch.device('cuda'),
            _record_birth_pool_diagnostic=lambda *args, **kwargs: None)
        def reject_old_sources():
            raise AssertionError('Old points must seed births only once')
        module._old_birth_sources = reject_old_sources
        module._reset_densification_stats = lambda: Module._reset_densification_stats(module)
        module._configure_birth_phase = lambda step: Module._configure_birth_phase(module, step)
        Module._densify_births(module, 210)

        self.assertEqual(len(births._xyz), 3)
        self.assertEqual(births.source_rows.tolist(), [0, 1, 1])
        self.assertEqual(births.created_steps.tolist(), [180, 210, 210])
        self.assertEqual(module.birth_control['clone_added'], 3)
        self.assertEqual(module.birth_control['split_added'], 1)
        self.assertEqual(module.birth_control['split_children'], 2)
        self.assertEqual(module.birth_control['split_parents_replaced'], 1)
        self.assertEqual(module.birth_control['pruned'], 1)
        self.assertEqual(module.birth_control['clone_added'] +
                         module.birth_control['split_added'] -
                         module.birth_control['pruned'], len(births._xyz))
        self.assertEqual(module.birth_control['densification_events'][0]['step'], 210)
        self.assertNotIn('proposal_added', module.birth_control['densification_events'][0])
        self.assertEqual(module.birth_control['densification_events'][0]['newborn_clones'], 0)
        self.assertEqual(module.birth_control['densification_events'][0]['newborn_split_children_count'], 2)

        for attr in RAW_ATTRIBUTES:
            self.assertEqual(len(getattr(births, attr)), len(births.source_rows))
            torch.testing.assert_close(getattr(births, attr)[0], getattr(source, attr)[0],
                                       rtol=0, atol=0)
        for attr in ('_rotation', '_opacity', '_features_dc', '_features_rest'):
            for child in (1, 2):
                torch.testing.assert_close(getattr(births, attr)[child], getattr(source, attr)[1],
                                           rtol=0, atol=0)
        # The oversized, non-selected candidate was removed by retained screen
        # evidence; split parents are not counted as retirement/pruning.
        self.assertNotIn(2, births.source_rows.tolist())


    @unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA split budget')
    def test_newborn_splits_stay_inside_pool_budget_and_choose_strong_gradients(self):
        cfg=GaussianConfig()
        source=self.model(3,None,[.02,.02,.02],cfg)
        births=BirthGaussianModel(1)
        births.initialize_from_sources(source,torch.arange(3,device='cuda'),180,cfg)
        births.xyz_gradient_accum[:]=torch.tensor([[1.],[2.],[3.]],device='cuda')
        births.denom[:]=1
        first=births.densify_and_prune(.5,.01,1.,20.,220,max_points=3)
        self.assertEqual(first['split_added'],0)
        self.assertEqual(len(births._xyz),3)
        second=births.densify_and_prune(.5,.01,1.,20.,260,max_points=4)
        self.assertEqual(second['split_added'],1)
        self.assertEqual(len(births._xyz),4)
        self.assertEqual(births.source_rows.tolist(),[0,1,2,2])


class TerminalBirthTests(unittest.TestCase):
    def test_step_270_optimizer_update_precedes_prune_and_no_new_split(self):
        events = []
        births = BirthGaussianModel(0)
        fields = dict(
            _xyz=torch.zeros(2, 3),
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
        births.source_rows = torch.tensor([4, 5])
        births.created_steps = torch.tensor([180, 180])
        births.spatial_lr_scale = 1.
        parameter_groups = [('xyz', '_xyz'), ('f_dc', '_features_dc'),
            ('f_rest', '_features_rest'), ('opacity', '_opacity'),
            ('scaling', '_scaling'), ('rotation', '_rotation')]
        births.optimizer = torch.optim.Adam([
            dict(name=group, params=[getattr(births, attr)])
            for group, attr in parameter_groups], lr=1.)
        births._opacity.grad = torch.tensor([[-1.], [0.]])

        original_step = births.optimizer.step
        original_zero = births.optimizer.zero_grad
        original_prune = births.prune_points

        def optimizer_step(*args, **kwargs):
            events.append('optimizer_step')
            return original_step(*args, **kwargs)

        def optimizer_zero(*args, **kwargs):
            events.append('optimizer_zero')
            return original_zero(*args, **kwargs)

        def prune_points(mask):
            events.append('terminal_prune')
            return original_prune(mask)

        births.optimizer.step = optimizer_step
        births.optimizer.zero_grad = optimizer_zero
        births.prune_points = prune_points
        births.add_densification_stats = lambda *args, **kwargs: None

        class BirthView:
            @property
            def get_opacity(self):
                return torch.sigmoid(births._opacity)

            @property
            def get_scaling(self):
                return torch.exp(births._scaling)

        config = ModuleConfig()
        config.gaussian_stage2.opacity_lr = 1.
        config.birth.refit_steps = 30
        active = SimpleNamespace(
            config=config, trainer=SimpleNamespace(global_step=270),
            births=births, gaussians=SimpleNamespace(
                _xyz=torch.zeros(2, 3),
                add_densification_stats=lambda *args, **kwargs: None),
            motion_steps=180, total_steps=300, birth_commit_step=270,
            motion_committed=True, birth_refit_active=False, frame_finalized=False,
            datamodule=SimpleNamespace(cameras_extent=20.),
            birth_control=dict(clone_added=2, split_added=0, pruned=0,
                               replacement_committed=False),
            make_birth_view=lambda: BirthView(),
            _configure_birth_phase=lambda step: Module._configure_birth_phase(active, step),
            _birth_learning_rate_scale=lambda step: torch.ones(2),
            _birth_optimizer_step=lambda step: Module._birth_optimizer_step(active, step),
            _update_birth_opacity=lambda step: None,
            _prune_terminal_births=lambda: Module._prune_terminal_births(active),
            _record_birth_pool_diagnostic=lambda *args, **kwargs: None,
            commit_birth_replacement=lambda: (events.append('commit'),
                                               active.birth_control.update(
                                                   replacement_committed=True)),
            _begin_birth_refit=lambda: (events.append('refit'),
                                        setattr(active, 'birth_refit_active', True)),
            _record_birth_learning=lambda *args, **kwargs: None)

        def fail_densify(*args, **kwargs):
            raise AssertionError('Step 270 must commit the pool without densifying')

        active._densify_births = fail_densify
        viewspace = torch.zeros(4, 3, requires_grad=True)
        viewspace.grad = torch.zeros_like(viewspace)
        Module.post_training_step(active, dict(
            visibility_filter=torch.ones(4, dtype=torch.bool),
            radii=torch.zeros(4), viewspace_points=viewspace))

        self.assertEqual(events, ['optimizer_step', 'optimizer_zero',
                                  'terminal_prune', 'commit', 'refit'])
        self.assertEqual(active.birth_control['pruned'], 1)
        self.assertTrue(active.birth_control['replacement_committed'])
        torch.testing.assert_close(births.source_rows, torch.tensor([4]), rtol=0, atol=0)
        self.assertEqual(len(births._xyz), 1)
        self.assertGreater(float(births.get_opacity[0]), .01)


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

    def test_source_gather_preserves_quantized_values_gradients_and_predictor_lifetime(self):
        for rotation_updates in (False, True):
            base, newborn = self.model(), self.model(4)
            source = torch.tensor([1, 7, 7, 29])
            config = BirthConfig(rotation_updates=rotation_updates)
            owned = QuantizedNewGaussianView(newborn, config, 6.5, model_raw(base), source)
            borrowed = QuantizedNewGaussianView(newborn, config, 6.5,
                                                model_raw(base, clone=False), source)
            for name in RAW_FIELDS:
                torch.testing.assert_close(borrowed.symbols[name], owned.symbols[name], rtol=0, atol=0)
                torch.testing.assert_close(borrowed.raw_value(name), owned.raw_value(name), rtol=0, atol=0)
            expected = None
            for view in (owned, borrowed):
                sum(view.raw_value(name).square().sum() for name in RAW_FIELDS).backward()
                gradients = [getattr(newborn, key).grad for key in RAW_ATTRIBUTES]
                if expected is None:
                    expected = [None if grad is None else grad.clone() for grad in gradients]
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


if __name__ == '__main__':
    unittest.main()
