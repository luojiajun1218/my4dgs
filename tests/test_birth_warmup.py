"""Newborn fitting, joint opacity selection, fixed refit and boundary restores."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.module import BirthConfig, Module, ModuleConfig
from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
from pipeline.trainer import Trainer, stage_image_steps
from scripts.replay_rigid_tree import decode_frame
import test_balanced_replacement as fixtures


class BirthWarmupConfigTests(unittest.TestCase):
    def test_explicit_alpha_and_valid_warmup_are_required(self):
        for steps in (True, -1, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                BirthConfig(warmup_steps=steps, initial_opacity=.1).validate(180)
        for alpha in (True, 0., 1., float('nan'), float('inf')):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                BirthConfig(warmup_steps=20, initial_opacity=alpha).validate(180)
        with self.assertRaisesRegex(ValueError, 'explicit initial_opacity'):
            BirthConfig(warmup_steps=20).validate(180)
        BirthConfig(warmup_steps=20, initial_opacity=.1).validate(180)
        BirthConfig().validate(180)  # Historical recipes keep their behavior.

    def test_warmup_must_leave_selection_steps(self):
        config=ModuleConfig(birth=BirthConfig(warmup_steps=60, initial_opacity=.1, refit_steps=40))
        with self.assertRaisesRegex(ValueError, 'leave candidate-selection'):
            Module(config, None).setup()


class AttributeRenderer:
    """Differentiable objective including every attribute used by the new schedule."""
    def __init__(self, model):
        self.gaussians = model

    def __call__(self, viewpoint, background):
        g = self.gaussians
        return dict(render=torch.cat((g.get_xyz, g.get_scaling,
                    g.get_features[:, 0], g.get_opacity), dim=-1))


@unittest.skipUnless(torch.cuda.is_available(), 'Native Gaussian optimizers require CUDA')
class BirthWarmupLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def active(self, path, *, births=10):
        module, raw, initial = fixtures.BalancedLifecycleCudaTests().active(path,
            policy='birth_driven_retirement', repeats=4, birth_count=births,
            warmup_steps=20, joint_opacity=True)
        module.renderer = AttributeRenderer(module.gaussians)
        module.trainer.global_step = 180
        module._record_birth_learning(180)
        return module, raw, initial

    @staticmethod
    def train(module, step):
        module.trainer.global_step = step
        module.pre_training_step()
        image = module.forward(None, training=True)['render']
        target = torch.tensor([.04, .02, .01, .08, .08, .08, .2, .2, .2, .5], device='cuda')
        loss=(image-target).square().mean()
        if loss.requires_grad:  # Same handling as Trainer._fit_views for empty warmup.
            loss.backward()
        points = torch.zeros(len(image), 3, device='cuda', requires_grad=True)
        points.grad = torch.zeros_like(points)
        module.post_training_step(dict(viewspace_points=points,
            visibility_filter=torch.ones(len(image), device='cuda', dtype=torch.bool),
            radii=torch.zeros(len(image), device='cuda')))

    def test_warmup_joint_competition_and_refit_obey_attribute_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            module, _, _ = self.active(Path(temp))
            self.assertEqual(stage_image_steps(module), (180, 200, 260, 300))
            alpha, scale = module.births._opacity.clone(), module.births._scaling.clone()
            xyz, dc = module.births._xyz.clone(), module.births._features_dc.clone()
            old_delta = module.opacity_update.delta.clone()
            old = model_raw(module.gaussians)
            for step in range(181, 201):
                self.train(module, step)
                torch.testing.assert_close(module.births._opacity, alpha, rtol=0, atol=0)
                torch.testing.assert_close(module.births._scaling, scale, rtol=0, atol=0)
                torch.testing.assert_close(module.opacity_update.delta, old_delta, rtol=0, atol=0)
                for name, value in model_raw(module.gaussians).items():
                    torch.testing.assert_close(value, old[name], rtol=0, atol=0)
            self.assertFalse(torch.equal(module.births._xyz, xyz))
            self.assertFalse(torch.equal(module.births._features_dc, dc))
            self.assertNotIn(module.births._opacity, module.births.optimizer.state)
            self.train(module, 201)
            self.assertTrue(module.opacity_update.delta.requires_grad)
            self.assertFalse(torch.equal(module.opacity_update.delta, old_delta))
            self.assertFalse(torch.equal(module.births._opacity, alpha))
            self.assertFalse(torch.equal(module.births._scaling, scale))
            for step in range(202, 261):
                self.train(module, step)
            frozen = module.opacity_update.delta.clone()
            rows = module.motion.previous_rows.clone()
            self.assertTrue(module.birth_refit_active)
            self.assertFalse(module.opacity_update.delta.requires_grad)
            for step in range(261, 301):
                self.train(module, step)
                torch.testing.assert_close(module.opacity_update.delta, frozen, rtol=0, atol=0)
                torch.testing.assert_close(module.motion.previous_rows, rows, rtol=0, atol=0)
            trace = module.frame_report['birth_control']['learning_trace']
            self.assertEqual([r['step'] for r in trace], [180,185,190,200,201,220,260,300])
            self.assertTrue(all(not r['new_opacity_trainable'] for r in trace if r['step']<=200))
            self.assertTrue(all(not r['old_opacity_trainable'] for r in trace if r['step']>=260))

    def test_boundary_checkpoint_resume_and_export_are_exact_for_joint_opacity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            module, raw, initial = self.active(root/'original')
            snapshots = {}
            for step in range(181, 301):
                self.train(module, step)
                if step in (190,200,201,220,260,280):
                    checkpoint = dict(global_step=step, current_frame=1)
                    module.on_save_checkpoint(checkpoint)
                    snapshots[step] = copy.deepcopy(checkpoint)
            final = model_raw(module.gaussians)
            for step, checkpoint in snapshots.items():
                with self.subTest(resume_step=step):
                    restored, _, _ = self.active(root/f'resume_{step}')
                    restored.trainer.global_step=step
                    restored.on_load_checkpoint(copy.deepcopy(checkpoint))
                    self.assertEqual(restored.births._opacity.requires_grad, step>200)
                    self.assertEqual(restored.opacity_update.delta.requires_grad, 200<step<260)
                    Trainer._write_reference(restored.trainer, 0, initial)
                    for after in range(step+1,301):
                        self.train(restored,after)
                    for name,value in model_raw(restored.gaussians).items():
                        torch.testing.assert_close(value,final[name],rtol=0,atol=0)
                    stream=restored.output_dir/'frame_storage/rigid_tree'
                    _,decoded=decode_frame(stream,1,initial,raw)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(decoded[name],final[name].cpu(),rtol=0,atol=0)
                    original=module.output_dir/'frame_storage/rigid_tree'
                    for name in ('motion_001.bin','birth_001.bin','opacity_001.bin','frame_001.json'):
                        self.assertEqual((stream/name).read_bytes(),(original/name).read_bytes())

    def test_empty_newborn_pool_can_finish_joint_selection_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            module,raw,initial=self.active(Path(temp),births=0)
            for step in range(181,301):
                self.train(module,step)
            self.assertEqual(module.frame_report['birth_count'],0)
            self.assertEqual(module.frame_report['retired_count'],0)
            _,decoded=decode_frame(module.output_dir/'frame_storage/rigid_tree',1,initial,raw)
            for name,value in model_raw(module.gaussians).items():
                torch.testing.assert_close(decoded[name],value.cpu(),rtol=0,atol=0)

    def test_historical_checkpoint_defaults_restore_without_enabling_warmup(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture=fixtures.BalancedLifecycleCudaTests()
            module,_,_=fixture.active(Path(temp)/'before')
            checkpoint=dict(global_step=260,current_frame=1)
            module.on_save_checkpoint(checkpoint)
            for key in ('warmup_steps','initial_opacity'):
                checkpoint['rigid_tree_module_config']['birth'].pop(key)
            restored,_,_=fixture.active(Path(temp)/'after')
            restored.on_load_checkpoint(copy.deepcopy(checkpoint))
            self.assertEqual(restored.config.birth.warmup_steps,0)
            self.assertTrue(restored.births._opacity.requires_grad)


if __name__=='__main__':
    unittest.main()
