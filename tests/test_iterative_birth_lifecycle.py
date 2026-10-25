"""Fixed-set motion refitting and exact stream replay across checkpoint resume."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.module import BirthConfig, Module, ModuleConfig
from pipeline.model.birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS, model_raw
from pipeline.model.gaussian import BirthGaussianModel, GaussianModel
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import decode_frame
from _motion_fixtures import current_motion
from _stream_fixtures import reference_raw


class PoseRenderer:
    """Differentiable position-only renderer exercising the merged model path."""
    def __init__(self, model):
        self.gaussians = model

    def __call__(self, viewpoint, background):
        return dict(render=self.gaussians.get_xyz)


@unittest.skipUnless(torch.cuda.is_available(), 'Native Gaussian optimizer uses CUDA buffers')
class IterativeBirthLifecycleCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @staticmethod
    def gaussian(raw, config, *, births=False, sources=None, created_step=None):
        model = BirthGaussianModel(1) if births else GaussianModel(1)
        for field, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(raw[field].cuda().clone()))
        model.spatial_lr_scale = 1.
        model.active_sh_degree = 1
        model.training_setup(config)
        model.max_radii2D = torch.zeros(len(model._xyz), device='cuda')
        if births:
            model.source_rows = sources.cuda().clone()
            model.created_steps = torch.full((len(sources),), created_step,
                                             dtype=torch.long, device='cuda')
        return model

    def active(self, directory):
        motion = copy.deepcopy(current_motion(points_per_axis=2, max_depth=2,
                                               amplitude_refit_steps=30))
        motion_cfg = motion.config
        config = ModuleConfig(rigid_tree=motion_cfg,
            birth=BirthConfig(refit_steps=30, replacement_count_tolerance=.2,
                              rotation_updates=False))
        raw = reference_raw()
        raw = {name: value.repeat((4,) + (1,) * (value.ndim - 1))
               for name, value in raw.items()}
        count = len(raw['xyz'])
        raw['logscale'].fill_(-3.)
        raw['opacity'] = torch.logit(torch.linspace(.1, .8, count)[:, None])
        data = SimpleNamespace(current_frame=1, cameras_extent=2.)
        module = Module(config, data)
        module.trainer = SimpleNamespace(global_step=270, device=torch.device('cuda'),
            output_dir=Path(directory), config=SimpleNamespace(num_init_steps=15000),
            datamodule=data, module=module)
        module.gaussians = self.gaussian(raw, config.gaussian)
        module.renderer = PoseRenderer(module.gaussians)
        module.background = torch.zeros(3, device='cuda')
        module.motion = SpatialGatedMotion(motion_cfg)
        module.motion.setup(raw['xyz'].cuda(), raw['rotation'].cuda(), 2., frame_index=1)
        initial = copy.deepcopy(module.motion.frame_reference_state)
        with torch.no_grad():
            module.motion._hard_gate_frozen = True
            module.motion.region_translation[module.motion.owner[0], 0] = 2 * module.motion.steps[0]
            module.motion.gs_translation[3, 0] = 3 * module.motion.gs_steps[0]
            module.motion(motion_cfg.motion_steps - motion_cfg.amplitude_refit_steps + 1,
                          training=True)
            xyz, rotation = module.motion(motion_cfg.motion_steps)
            module.gaussians.apply_selected_motion(xyz, rotation)
        for parameter in module.motion.parameters():
            parameter.requires_grad_(False)
        module.motion.optimizer.zero_grad(set_to_none=True)

        module.frame_prepared = module.motion_committed = True
        module.frame_finalized = False
        module.opacity_update = None
        module.reference_count = count
        module.initial_reference = dict(original_count=count, retained_count=count,
            removed_count=0, threshold=0.)
        module.birth_control = dict(clone_added=3, split_added=0, split_children=0,
            split_parents_replaced=0, pruned=0, previous_birth_count=0,
            source_tracked=True, replacement_committed=False)
        module.retired_reference = None
        module.frame_report = None
        module.stage_image_metrics = []

        sources = torch.tensor([0, 5, 6], device='cuda')
        newborn = {name: value[sources.to(value.device)].clone()
                   for name, value in model_raw(module.gaussians).items()}
        newborn['xyz'][:, 0] += .003
        newborn['opacity'] = torch.logit(torch.tensor([[.95], [.98], [.02]], device='cuda'))
        module.births = self.gaussian(newborn, config.gaussian_stage2,
            births=True, sources=sources, created_step=module.motion_steps)
        module._freeze_base()
        module._configure_birth_phase(module.motion_steps + 1)
        Trainer._write_reference(module.trainer, 0, initial)
        return module, raw, initial

    @staticmethod
    def train_pose_step(module, step):
        module.trainer.global_step = step
        pose = module.forward(None, training=True)['render']
        old_count = len(module.gaussians._xyz)
        if module.birth_refit_active:
            expected, _ = module.motion(step)
            torch.testing.assert_close(pose[:old_count], expected, rtol=0, atol=0)
        loss = (pose[:, 0] - .04).square().mean()
        if loss.requires_grad:
            loss.backward()
        points = torch.zeros(len(pose), 3, device='cuda', requires_grad=True)
        points.grad = torch.zeros_like(points)
        package = dict(visibility_filter=torch.ones(len(pose), dtype=torch.bool, device='cuda'),
                       radii=torch.zeros(len(pose), device='cuda'), viewspace_points=points)
        module.post_training_step(package)

    def test_step270_freezes_pool_then_refits_fixed_support_and_replays_after_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_dir = Path(temporary) / 'original'
            module, raw, initial = self.active(first_dir)
            old_attributes = {name: value.detach().clone()
                              for name, value in model_raw(module.gaussians).items()}

            # No proposal or pruning happens after the final 270-step optimizer update.
            with patch.object(module, '_retirement_contributions', return_value=(
                    torch.zeros(len(module.gaussians._xyz), device='cuda'),
                    torch.zeros(len(module.gaussians._xyz), device='cuda'))):
                self.train_pose_step(module, 270)
            self.assertEqual(len(module.births._xyz), 3)
            self.assertEqual(len(module.gaussians._xyz), 29)
            self.assertEqual(len(module.gaussians._xyz) + len(module.births._xyz), 32)
            self.assertEqual(module.birth_control['retirement']['retired_count'], 3)
            self.assertTrue(module.birth_refit_active)
            support = copy.deepcopy(module.motion._amplitude_refit_support)
            sources = module.births.source_rows.clone()
            gates = module.motion.opened.clone()
            old_after_retirement = model_raw(module.gaussians)
            for name in ('rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest'):
                torch.testing.assert_close(old_after_retirement[name],
                    old_attributes[name][torch.tensor([0, 4, 5, *range(6, 32)], device='cuda')],
                    rtol=0, atol=0)

            checkpoint = dict(global_step=270, current_frame=1)
            module.on_save_checkpoint(checkpoint)
            checkpoint = copy.deepcopy(checkpoint)
            resumed, _, _ = self.active(Path(temporary) / 'resumed')
            resumed.on_load_checkpoint(checkpoint)
            self.assertTrue(resumed.birth_refit_active)
            Trainer._write_reference(resumed.trainer, 0, initial)

            for step in range(271, 301):
                for worker in (module, resumed):
                    self.train_pose_step(worker, step)
                    self.assertEqual(len(worker.gaussians._xyz) + len(worker.births._xyz), 32)
                    self.assertTrue(torch.equal(worker.motion.opened, gates))
                    if step < 300:
                        self.assertTrue(torch.equal(worker.births.source_rows, sources))
                        for name, mask in support.items():
                            self.assertFalse(bool((worker.motion._amplitude_refit_support[name] &
                                                   ~mask).any()))
                    if step == 300:
                        self.assertTrue(worker.frame_finalized)

            for worker in (module, resumed):
                stream = worker.output_dir / 'frame_storage' / 'rigid_tree'
                _, decoded = decode_frame(stream, 1, initial, raw)
                for name, value in model_raw(worker.gaussians).items():
                    torch.testing.assert_close(value.cpu(), decoded[name], rtol=0, atol=0)
                self.assertEqual(worker.frame_report['birth_count'], 3)
                self.assertEqual(worker.frame_report['retired_count'], 3)
            for suffix in ('motion_001.bin', 'birth_001.bin', 'frame_001.json'):
                self.assertEqual((first_dir / 'frame_storage/rigid_tree' / suffix).read_bytes(),
                    (resumed.output_dir / 'frame_storage/rigid_tree' / suffix).read_bytes())

    def test_newborn_only_tail_keeps_old_motion_frozen_and_replays_identically_after_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            module, raw, initial = self.active(Path(temporary) / 'original')
            module.config.birth.refit_steps = 0
            checkpoint = dict(global_step=270, current_frame=1)
            module.on_save_checkpoint(checkpoint)
            resumed, _, _ = self.active(Path(temporary) / 'resumed')
            resumed.config.birth.refit_steps = 0
            resumed.on_load_checkpoint(copy.deepcopy(checkpoint))
            old_pose = model_raw(module.gaussians)['xyz']
            gate = module.motion.opened.clone()
            for step in range(271, 301):
                for worker in (module, resumed):
                    with patch.object(worker, '_retirement_contributions', return_value=(
                            torch.zeros(len(old_pose), device='cuda'),
                            torch.zeros(len(old_pose), device='cuda'))):
                        self.train_pose_step(worker, step)
                    self.assertFalse(worker.birth_refit_active)
                    self.assertTrue(torch.equal(worker.motion.opened, gate))
                    if step < 300:
                        torch.testing.assert_close(worker.gaussians._xyz, old_pose, rtol=0, atol=0)
                        self.assertFalse(any(value.requires_grad for value in worker.motion.parameters()))
            for worker in (module, resumed):
                self.assertTrue(worker.frame_finalized)
                self.assertEqual(worker.frame_report['retired_count'], 3)
                _, decoded = decode_frame(worker.output_dir / 'frame_storage/rigid_tree',
                                           1, initial, raw)
                for name, value in model_raw(worker.gaussians).items():
                    torch.testing.assert_close(value.cpu(), decoded[name], rtol=0, atol=0)
            for suffix in ('motion_001.bin', 'birth_001.bin', 'frame_001.json'):
                self.assertEqual((module.output_dir / 'frame_storage/rigid_tree' / suffix).read_bytes(),
                                 (resumed.output_dir / 'frame_storage/rigid_tree' / suffix).read_bytes())


if __name__ == '__main__':
    unittest.main()
