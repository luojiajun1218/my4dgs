"""Actual module hooks/checkpoints/finalization with CPU image proxies only."""

import copy
import importlib
import random
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

import test_persistent_motion_integration as lifecycle_tests


class WorkerStageView:
    def __init__(self, path, worker_id):
        self.path, self.worker_id = Path(path), worker_id
        self.image = torch.full((3, 16, 16), .4)

    def to(self, device):
        self.image = self.image.to(device)
        return self


class WorkerStageDataset(torch.utils.data.Dataset):
    items = [SimpleNamespace(path=Path('cam01.png')), SimpleNamespace(path=Path('cam00.png'))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        info = torch.utils.data.get_worker_info()
        if info is None:
            raise AssertionError('This dataset image path requires a DataLoader worker')
        # Exercise the actual installed library warning that crashed direct
        # main-process dataset indexing, alongside real worker view transport.
        from libgs.data.dataset.base import warning
        warning()
        return WorkerStageView(self.items[index].path, info.id)


class SparseOpacityModuleTests(unittest.TestCase):
    def fixture(self, directory, with_birth=True):
        lifecycle = lifecycle_tests.PersistentLifecycleTests()
        rigid, _, _, _, _ = lifecycle.modules()
        source = importlib.import_module('pipeline.ReCon-GS.module')
        motion = lifecycle.fixture()
        lifecycle.initialize(motion)
        motion.forward(200, training=True)
        motion.finalize_online_refinement()

        class CPUSetup:
            # The installed Gaussian training_setup allocates CUDA statistics;
            # retain real Gaussian capture/restore/fields with CPU Adam/statistics.
            def training_setup(self, config):
                self.percent_dense = config.percent_dense
                self.xyz_gradient_accum = self._xyz.new_zeros(len(self._xyz), 1)
                self.denom = self._xyz.new_ones(len(self._xyz), 1)
                names = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')
                attrs = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
                rates = (config.position_lr_init, config.feature_lr, config.feature_lr / 20,
                         config.opacity_lr, config.scaling_lr, config.rotation_lr)
                self.optimizer = torch.optim.Adam([dict(params=[getattr(self, attr)], name=name, lr=rate)
                    for attr, name, rate in zip(attrs, names, rates)], eps=1e-15)

        class CPUGaussian(CPUSetup, source.GaussianModel):
            pass

        class CPUBirth(CPUSetup, source.BirthGaussianModel):
            pass

        config = source.ModuleConfig(rigid_tree=motion.config,
            old_opacity=source.OldOpacityConfig(enabled=True))
        config.birth.densify_interval = 1000  # Exclude GPU densification from this numerical fixture.
        gaussians = CPUGaussian(1)
        count = len(motion.base_xyz)
        raw = dict(_xyz=motion.base_xyz.clone(), _rotation=motion.base_rotation.clone(),
            _scaling=torch.full((count, 3), -5.), _opacity=torch.full((count, 1), -1.),
            _features_dc=torch.full((count, 1, 3), .2), _features_rest=torch.zeros(count, 3, 3))
        raw['_opacity'][0] = 2.
        for name, value in raw.items():
            setattr(gaussians, name, nn.Parameter(value))
        gaussians.spatial_lr_scale = 1.
        gaussians.max_radii2D = torch.zeros(count)
        gaussians.training_setup(config.gaussian)
        gaussians.adaptive_densify_and_clone = Mock(return_value=dict(added=0))
        births = CPUBirth(1)
        module = SimpleNamespace(config=config, gaussians=gaussians, motion=motion, births=births,
            device=torch.device('cpu'), output_dir=Path(directory), current_frame=1,
            motion_steps=200, total_steps=300, frame_prepared=True, motion_committed=False,
            frame_finalized=False, reference_count=count, initial_reference=dict(retained_count=count),
            frame_report=None, birth_control=dict(clone_added=0, split_added=0, pruned=0),
            tree_statistics=[], stage_image_metrics=[], old_opacity_delta=None,
            old_opacity_optimizer=None, old_opacity_metrics={}, background=torch.zeros(3),
            datamodule=SimpleNamespace(current_frame=1, cameras_extent=1.))
        for name in ('_freeze_base', '_initialize_old_opacity', '_step_old_opacity', '_begin_births',
                     'make_birth_view', 'forward', 'post_training_step', 'finalize_frame',
                     '_prune_terminal_births', 'on_save_checkpoint', 'on_load_checkpoint'):
            setattr(module, name, MethodType(getattr(source.Module, name), module))

        class Renderer:
            def __init__(self):
                self.gaussians = gaussians
                self.seen = []

            def __call__(self, view, background):
                # Per-row alpha contributes to pixels; renderer/GPU is replaced,
                # while actual module view selection and autograd remain intact.
                model = self.gaussians
                opacity = model.get_opacity[:, 0]
                self.seen.append((type(model).__name__, len(opacity)))
                indices = torch.arange(256).remainder(len(opacity))
                points = torch.zeros(len(opacity), 3, requires_grad=True)
                image = opacity[indices].reshape(1, 16, 16).expand(3, -1, -1) + points.sum() * 0
                torch.rand(1); np.random.rand(); random.random()
                return dict(render=image, viewspace_points=points,
                    visibility_filter=torch.ones(len(opacity), dtype=torch.bool), radii=torch.zeros(len(opacity)))

        module.renderer = Renderer()
        stream = module.output_dir / 'frame_storage' / 'rigid_tree'
        stream.mkdir(parents=True)
        reference = stream / 'reference_frame0.npz'
        raw_fields = source.model_raw(gaussians)
        reference_state = motion.frame_reference_state
        np.savez_compressed(reference, **{name: value.detach().numpy() for name, value in raw_fields.items()},
            owner=reference_state.owner.numpy(), parents=reference_state.parents.numpy(), frame_index=0)
        module.trainer = SimpleNamespace(global_step=200, decoder_reference_path=reference,
                                        bootstrap_pending=False)
        module._begin_births()
        self.assertIsNotNone(module.old_opacity_delta)
        if with_birth:
            for name in raw:
                setattr(births, name, nn.Parameter(getattr(gaussians, name)[:1].detach().clone()))
            births.spatial_lr_scale = 1.
            births.source_rows = torch.tensor([0])
            births.max_radii2D = torch.zeros(1)
            births.training_setup(config.gaussian_stage2)
            module.birth_control['clone_added'] = 1
        view = SimpleNamespace(image=torch.full((3, 16, 16), .4), path=Path('cam00.png'))
        return module, source, CPUBirth, view

    def train_opacity(self, module, view, start, end):
        for step in range(start, end + 1):
            module.trainer.global_step = step
            package = module.forward(view, training=True)
            image = package['render']
            # Per-row target lowers old row0, raises row1 and fits a newborn.
            desired_alpha = module.old_opacity_delta.reference.sigmoid().flatten().clone()
            desired_alpha[0] -= .15
            desired_alpha[1] += .15
            if len(module.births._xyz):
                desired_alpha = torch.cat((desired_alpha, torch.tensor([.65])))
            target = desired_alpha[torch.arange(256).remainder(len(desired_alpha))].reshape(1, 16, 16).expand_as(image)
            (image - target).square().sum().backward()
            module.post_training_step(package)

    def test_pending_birth_checkpoint_adam_resume_finalization_and_next_reference(self):
        from scripts import replay_rigid_tree
        with tempfile.TemporaryDirectory() as directory:
            module, source, _, view = self.fixture(directory)
            self.train_opacity(module, view, 201, 250)
            checkpoint = dict(global_step=250)
            module.on_save_checkpoint(checkpoint)
            self.assertIsNotNone(checkpoint['old_opacity_delta'])
            self.assertIsNotNone(checkpoint['births'])
            resumed, _, _, _ = self.fixture(Path(directory) / 'resume')
            resumed.on_load_checkpoint(copy.deepcopy(checkpoint))
            resumed.trainer.global_step = 250
            torch.testing.assert_close(module.old_opacity_delta.delta, resumed.old_opacity_delta.delta, rtol=0, atol=0)
            self.assertEqual(module.old_opacity_optimizer.state_dict()['state'][0]['step'],
                             resumed.old_opacity_optimizer.state_dict()['state'][0]['step'])
            self.assertIs(resumed.old_opacity_optimizer.param_groups[0]['params'][0], resumed.old_opacity_delta.delta)
            for name in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(module.old_opacity_optimizer.state_dict()['state'][0][name],
                    resumed.old_opacity_optimizer.state_dict()['state'][0][name], rtol=0, atol=0)
            for step in range(251, 301):
                self.train_opacity(module, view, step, step)
                self.train_opacity(resumed, view, step, step)
            self.assertTrue(module.frame_finalized)
            self.assertIsNone(module.old_opacity_delta)
            for attr in source.RAW_ATTRIBUTES:
                torch.testing.assert_close(getattr(module.gaussians, attr), getattr(resumed.gaussians, attr), rtol=0, atol=0)
            self.assertGreater(module.old_opacity_metrics['increased_rows'], 0)
            self.assertGreater(module.old_opacity_metrics['decreased_rows'], 0)
            stream = Path(directory) / 'frame_storage' / 'rigid_tree'
            _, raw = replay_rigid_tree.replay(stream, module.trainer.decoder_reference_path, 1)
            self.assertEqual(source.reference_digest(raw), source.reference_digest(source.model_raw(module.gaussians)))
            final = dict(global_step=300); module.on_save_checkpoint(final)
            self.assertIsNone(final['old_opacity_delta'])
            logits = module.gaussians._opacity.detach().clone()
            finalized, _, _, _ = self.fixture(Path(directory) / 'finalized', with_birth=False)
            finalized.on_load_checkpoint(copy.deepcopy(final))
            self.assertTrue(finalized.frame_finalized)
            self.assertIsNone(finalized.old_opacity_delta)
            torch.testing.assert_close(finalized.gaussians._opacity, logits, rtol=0, atol=0)
            finalized.current_frame = finalized.datamodule.current_frame = 2
            finalized.frame_prepared = False
            source.Module.prepare_frame(finalized)
            finalized.motion.observations.fill_(10)
            finalized.motion.forward(100)
            finalized.motion.update_topology(100)
            finalized.motion.forward(200, training=True)
            finalized.motion.finalize_online_refinement()
            finalized._begin_births()
            torch.testing.assert_close(finalized.old_opacity_delta.reference, logits, rtol=0, atol=0)
            self.assertEqual(int(finalized.old_opacity_delta.symbols(False).ne(0).sum()), 0)
            # Real cache keys invalidate because finalize installed new raw tensors.
            with torch.no_grad():
                torch.testing.assert_close(module.gaussians.get_opacity, logits.sigmoid(), rtol=0, atol=0)

    def test_no_births_still_updates_opacity_and_exports_real_packet(self):
        from scripts import replay_rigid_tree
        with tempfile.TemporaryDirectory() as directory:
            module, source, _, view = self.fixture(directory, with_birth=False)
            self.train_opacity(module, view, 201, 300)
            self.assertTrue(module.frame_finalized)
            self.assertEqual(module.frame_report['birth_count'], 0)
            self.assertIsNone(module.frame_report['birth'])
            self.assertGreater(module.frame_report['opacity_update']['archive_bytes'], 0)
            self.assertGreater(module.old_opacity_metrics['increased_rows'], 0)
            self.assertGreater(module.old_opacity_metrics['decreased_rows'], 0)
            stream = Path(directory) / 'frame_storage' / 'rigid_tree'
            _, raw = replay_rigid_tree.replay(stream, module.trainer.decoder_reference_path, 1)
            self.assertEqual(source.reference_digest(raw), module.frame_report['materialized_sha256'])

    def test_step200_old_only_excludes_clone_opacity_and_stage_diagnostics_restore_rng(self):
        trainer_source = importlib.import_module('pipeline.ReCon-GS.trainer')
        with tempfile.TemporaryDirectory() as directory:
            module, _, _, view = self.fixture(directory)
            with torch.no_grad():
                module.old_opacity_delta.delta[0] = -.5
            model = module.renderer.gaussians
            old_image = module.forward(view, old_only=True)['render']
            self.assertEqual(module.renderer.seen[-1][1], 32)
            self.assertEqual(module.renderer.seen[-1][0], 'CPUGaussian')
            indices = torch.arange(256).remainder(32)
            expected = module.gaussians._opacity.detach().sigmoid()[indices, 0].reshape(1, 16, 16).expand_as(old_image)
            torch.testing.assert_close(old_image, expected, rtol=0, atol=0)
            self.assertIs(module.renderer.gaussians, model)
            module.forward(view)
            self.assertEqual(module.renderer.seen[-1][1], 33)
            harness = SimpleNamespace(module=module, current_frame=1,
                config=SimpleNamespace(stage_image_camera='cam00.png'))
            harness._capture_rng = MethodType(trainer_source.Trainer._capture_rng, harness)
            harness._restore_rng = trainer_source.Trainer._restore_rng
            with patch('torch.cuda.is_available', return_value=False):
                rng = harness._capture_rng()
                trainer_source.Trainer._record_stage_image(harness, 200, view)
                after = harness._capture_rng()
            torch.testing.assert_close(rng['torch'], after['torch'], rtol=0, atol=0)
            np.testing.assert_equal(rng['numpy'], after['numpy'])
            self.assertEqual(rng['python'], after['python'])
            self.assertEqual(module.stage_image_metrics[-1]['phase'], 'post_joint_motion_old_only')
            self.assertTrue(module.stage_image_metrics[-1]['old_only'])
            self.assertEqual(module.renderer.seen[-1][1], 32)

    def test_real_worker_loader_stage_lookup_and_rng_restore(self):
        trainer_source = importlib.import_module('pipeline.ReCon-GS.trainer')
        data_source = importlib.import_module('pipeline.ReCon-GS.data')
        with tempfile.TemporaryDirectory() as directory:
            module, _, _, _ = self.fixture(directory)
            dataset = WorkerStageDataset()
            with self.assertRaises(AssertionError):
                dataset[1]
            datamodule = SimpleNamespace(config=SimpleNamespace(
                extra_dataloader_kwargs=dict(num_workers=1, pin_memory=False), expected_image_size=[16, 16]),
                get_test_dataset=lambda scale=1.: dataset)
            datamodule._view_collate = MethodType(data_source.DataModule._view_collate, datamodule)
            datamodule.test_dataloader = MethodType(data_source.DataModule.test_dataloader, datamodule)
            harness = SimpleNamespace(module=module, current_frame=1, device=torch.device('cpu'),
                datamodule=datamodule, config=SimpleNamespace(stage_image_camera='cam00.png'))
            harness._capture_rng = MethodType(trainer_source.Trainer._capture_rng, harness)
            harness._restore_rng = trainer_source.Trainer._restore_rng
            with patch('torch.cuda.is_available', return_value=False):
                rng = harness._capture_rng()
                selected = trainer_source.Trainer._record_stage_image(harness, 200)
                after = harness._capture_rng()
            self.assertEqual(selected.worker_id, 0)
            self.assertEqual(selected.path.name, 'cam00.png')
            self.assertEqual(module.renderer.seen[-1], ('CPUGaussian', 32))
            torch.testing.assert_close(rng['torch'], after['torch'], rtol=0, atol=0)
            np.testing.assert_equal(rng['numpy'], after['numpy'])
            self.assertEqual(rng['python'], after['python'])
            self.assertEqual(module.stage_image_metrics[-1]['camera'], 'cam00.png')
            self.assertTrue(module.stage_image_metrics[-1]['old_only'])


if __name__ == '__main__':
    unittest.main()
