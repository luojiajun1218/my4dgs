"""CPU checks for the native first-frame bootstrap and frozen render cache."""

import importlib
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if 'pipeline.ReCon-GS' not in sys.modules:
    pipeline_package = ModuleType('pipeline')
    pipeline_package.__path__ = [str(ROOT / 'pipeline')]
    recon_package = ModuleType('pipeline.ReCon-GS')
    recon_package.__path__ = [str(ROOT / 'pipeline' / 'ReCon-GS')]
    pipeline_package.__dict__['ReCon-GS'] = recon_package
    sys.modules['pipeline'] = pipeline_package
    sys.modules['pipeline.ReCon-GS'] = recon_package

module = importlib.import_module('pipeline.ReCon-GS.module')
trainer = importlib.import_module('pipeline.ReCon-GS.trainer')
gaussian = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
codec = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
replay = importlib.import_module('scripts.replay_rigid_tree')


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
                for name in module.RAW_ATTRIBUTES}

    def restore(self, packet, config):
        for name in module.RAW_ATTRIBUTES:
            setattr(self, name, torch.nn.Parameter(packet[name].detach().clone()))

    def prune_points(self, mask):
        self.prune_calls += 1
        for name in module.RAW_ATTRIBUTES:
            value = getattr(self, name)
            setattr(self, name, torch.nn.Parameter(value.detach()[~mask].clone()))


def fake_module(config, gaussians, frame):
    holder = SimpleNamespace(config=config, gaussians=gaussians,
        renderer=SimpleNamespace(gaussians=None), current_frame=frame,
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=15000)),
        _freeze_base=lambda: [getattr(gaussians, name).requires_grad_(False)
                              for name in module.RAW_ATTRIBUTES])
    return holder


class NativeReferencePruningTests(unittest.TestCase):
    def test_threshold_validation_and_full_delete_guard(self):
        for bad in (-.01, 1., float('nan'), float('inf'), True, '0.015'):
            with self.subTest(threshold=bad), self.assertRaisesRegex(
                    ValueError, 'initial_opacity_threshold'):
                module.validate_initial_opacity_threshold(bad)
        original = FakeNativeGaussians()
        with self.assertRaisesRegex(ValueError, 'every Gaussian'):
            module.prune_native_initial_gaussians(original, .9)
        self.assertEqual(original.prune_calls, 0)
        self.assertEqual(len(original._xyz), 4)
        unpruned = module.prune_native_initial_gaussians(original, 0.)
        self.assertEqual((unpruned['original_count'], unpruned['retained_count'],
                          unpruned['removed_count']), (4, 4, 0))
        self.assertEqual(original.prune_calls, 0)

    def test_native_prune_keeps_deterministic_rows_and_own_resume_does_not_prune(self):
        config = module.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        module.Module.on_load_checkpoint(active,
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
                            for name in module.RAW_ATTRIBUTES))

        own = dict(gaussians=native.capture(), rigid_tree_version=1,
            rigid_tree_config=asdict(config.rigid_tree),
            rigid_tree_module_config=asdict(config),
            rigid_tree_initial_reference=active.initial_reference,
            rigid_tree_frame=dict(prepared=False, finalized=True,
                motion_committed=False, reference_count=2, report=None,
                birth_control={}), births=None)
        restored = FakeNativeGaussians()
        resume = fake_module(config, restored, frame=1)
        module.Module.on_load_checkpoint(resume, own)
        self.assertEqual(restored.prune_calls, 0)
        self.assertEqual(len(restored._xyz), 2)
        self.assertEqual(resume.initial_reference, active.initial_reference)
        self.assertEqual(resume.reference_count, 2)

    def test_pruned_reference_layout_and_frame_zero_replay_have_same_count(self):
        config = module.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        module.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        with tempfile.TemporaryDirectory() as directory:
            worker = SimpleNamespace(output_dir=Path(directory), module=active,
                datamodule=SimpleNamespace(cameras_extent=1.))
            trainer.Trainer._write_reference(worker, frame_index=0)
            with np.load(worker.decoder_reference_path, allow_pickle=False) as packet:
                self.assertEqual(int(packet['initial_original_count']), 4)
                self.assertEqual(int(packet['initial_retained_count']), 2)
                self.assertEqual(int(packet['initial_removed_count']), 2)
                self.assertEqual(float(packet['initial_opacity_threshold']), .015)
                self.assertEqual(len(packet['owner']), 2)
            state, raw = replay.replay(worker.decoder_reference_path.parent,
                                       worker.decoder_reference_path, through=0)
            self.assertEqual(len(state.xyz), active.reference_count)
            torch.testing.assert_close(raw['xyz'][:, 0], torch.tensor([1., 3.]),
                                       rtol=0, atol=0)
            rebuilt = codec.bootstrap_state(raw['xyz'], raw['rotation'],
                                            config.rigid_tree.region_count)
            torch.testing.assert_close(state.owner, rebuilt.owner, rtol=0, atol=0)
            torch.testing.assert_close(state.parents, rebuilt.parents, rtol=0, atol=0)


class ActiveRenderCacheTests(unittest.TestCase):
    @staticmethod
    def model(trainable):
        result = gaussian.GaussianModel(0)
        raw = dict(_xyz=torch.tensor([[-.25, 0., 0.], [.25, 0., 0.]]),
                   _rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                   _scaling=torch.full((2, 3), -.4),
                   _opacity=torch.tensor([[.1], [-.2]]),
                   _features_dc=torch.tensor([[[.2, .4, .6]], [[.8, .3, .1]]]),
                   _features_rest=torch.zeros(2, 0, 3))
        for name, value in raw.items():
            setattr(result, name, torch.nn.Parameter(value, requires_grad=trainable))
        return result

    @staticmethod
    def image(model):
        # A small differentiable image assembled from the same renderer inputs.
        pixels = torch.linspace(-1., 1., 7)
        centers = model.get_xyz[:, :1]
        width = model.get_scaling[:, :1]
        alpha = model.get_opacity
        rotation = model.get_rotation
        color = model.get_features[:, 0, :] * (rotation[:, :1] + rotation[:, 1:2])
        coverage = torch.exp(-.5 * ((centers - pixels) / width).square()) * alpha
        return coverage.T @ color

    def test_cache_preserves_image_and_attribute_gradients(self):
        cached, uncached = self.model(True), self.model(True)
        uncached.render_attribute_cache = False
        left, right = self.image(cached), self.image(uncached)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().sum().backward()
        right.square().sum().backward()
        for name in module.RAW_ATTRIBUTES:
            torch.testing.assert_close(getattr(cached, name).grad,
                                       getattr(uncached, name).grad,
                                       rtol=0, atol=0)

    def test_frozen_cache_preserves_motion_gradient_and_invalidates_on_change(self):
        cached, uncached = self.model(False), self.model(False)
        uncached.render_attribute_cache = False
        cached.rigid_xyz = cached._xyz.detach().clone().requires_grad_()
        uncached.rigid_xyz = uncached._xyz.detach().clone().requires_grad_()
        cached.rigid_rotation = cached._rotation.detach().clone().requires_grad_()
        uncached.rigid_rotation = uncached._rotation.detach().clone().requires_grad_()
        with torch.no_grad():
            self.assertIs(cached.get_features, cached.get_features)
            self.assertIs(cached.get_scaling, cached.get_scaling)
            self.assertIs(cached.get_opacity, cached.get_opacity)
        left, right = self.image(cached), self.image(uncached)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().sum().backward()
        right.square().sum().backward()
        torch.testing.assert_close(cached.rigid_xyz.grad, uncached.rigid_xyz.grad,
                                   rtol=0, atol=0)
        torch.testing.assert_close(cached.rigid_rotation.grad,
                                   uncached.rigid_rotation.grad, rtol=0, atol=0)
        self.assertGreater(float(cached.rigid_rotation.grad.abs().sum()), 0.)
        with torch.no_grad():
            old = cached.get_opacity
            cached._opacity.add_(.25)
            self.assertIsNot(cached.get_opacity, old)
            torch.testing.assert_close(cached.get_opacity,
                                       torch.sigmoid(cached._opacity), rtol=0, atol=0)


class TerminalBirthTests(unittest.TestCase):
    def test_final_optimizer_update_precedes_prune_without_new_split(self):
        quantization = importlib.import_module('pipeline.ReCon-GS.model.birth_quantization')
        events = []
        births = gaussian.BirthGaussianModel(0)
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
        config = module.ModuleConfig()
        config.rigid_tree.motion_steps = 100
        config.rigid_tree.birth_steps = 200
        config.birth.densify_interval = 40  # Final step is a former split point.
        active = SimpleNamespace(config=config, motion_committed=True,
            births=births, gaussians=SimpleNamespace(_xyz=torch.zeros(2, 3)),
            trainer=SimpleNamespace(global_step=300), motion_steps=100,
            total_steps=300, datamodule=SimpleNamespace(cameras_extent=20.),
            birth_control=dict(pruned=0, split_added=0),
            finalize_frame=lambda: events.append('finalize'))
        active._prune_terminal_births = lambda: module.Module._prune_terminal_births(active)
        package = dict(visibility_filter=torch.tensor([False, False, True, True]),
                       radii=torch.zeros(4), viewspace_points=None)
        module.Module.post_training_step(active, package)
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


if __name__ == '__main__':
    unittest.main()
