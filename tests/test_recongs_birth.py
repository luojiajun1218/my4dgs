"""Official spawn and next-frame global replacement, with exact stream replay."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.module import BirthConfig, GaussianConfig, Module, ModuleConfig, restore_retired_reference
from pipeline.model.birth_quantization import RAW_ATTRIBUTES, RAW_FIELDS, model_raw
from pipeline.model.gaussian import GaussianModel
from pipeline.model.recongs_birth import ReConGSBirthGaussianModel, recongs_survivor_rows
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import replay
from _stream_fixtures import reference_raw, motion_model


def recon_config():
    return BirthConfig(method='recongs', rotation_updates=True, max_new_gaussians=None)


def seed_native_optimizer(model):
    """Represent a trained native first-frame optimizer without changing its GS."""
    for group in model.optimizer.param_groups:
        parameter = group['params'][0]
        model.optimizer.state[parameter] = dict(step=torch.tensor(17.),
            exp_avg=torch.full_like(parameter, .2), exp_avg_sq=torch.full_like(parameter, .3))


class ReConGSBirthTests(unittest.TestCase):
    def test_spawn_matches_upstream_world_axes_identity_and_logit(self):
        raw = reference_raw()
        source = GaussianModel(1)
        for field, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(source, attr, torch.nn.Parameter(raw[field].clone()))
        source.active_sh_degree = 1; source.spatial_lr_scale = 2.
        source.percent_dense = .01
        source._scaling.data.fill_(-6.)
        source._rotation.data[:] = torch.tensor([.7071068, 0., 0., .7071068])
        source.denom = torch.ones(8, 1)
        source.xyz_gradient_accum = torch.zeros(8, 1)
        source.xyz_gradient_accum[2] = .01
        before = model_raw(source)
        born = ReConGSBirthGaussianModel(1)
        with patch.object(born, 'training_setup'), patch('torch.normal',
                side_effect=lambda mean, std: mean + std) as sample:
            born.initialize_recongs_sources(source, 180, GaussianConfig(), .00015, 2.)
        self.assertEqual(born.source_rows.tolist(), [2])
        torch.testing.assert_close(sample.call_args.args[1], source.get_scaling[[2]] * 2.)
        torch.testing.assert_close(born._xyz, source._xyz[[2]] + source.get_scaling[[2]] * 2.)
        torch.testing.assert_close(born._rotation, torch.tensor([[1., 0., 0., 0.]]))
        torch.testing.assert_close(born._opacity, torch.tensor([[.1]]))
        self.assertAlmostEqual(float(born.get_opacity), .5249792, places=6)
        for field in ('logscale', 'sh_dc', 'sh_rest'):
            torch.testing.assert_close(model_raw(born)[field], before[field][[2]])
        for field, value in before.items():
            torch.testing.assert_close(model_raw(source)[field], value, rtol=0, atol=0)

    def test_global_merge_can_evict_new_and_old_points_including_a_predictor(self):
        # Old predictor 0 and the weak new point 4 lose; strong new point 3 stays.
        alpha = torch.tensor([.002, .8, .7, .95, .001])
        rows = recongs_survivor_rows(alpha, previous_birth_count=2)
        self.assertEqual(rows.tolist(), [1, 2, 3])
        self.assertEqual(recongs_survivor_rows(alpha, 0).tolist(), list(range(5)))
        with self.assertRaises(ValueError):
            recongs_survivor_rows(alpha, 5)

    def test_recongs_rejects_legacy_caps_and_refit(self):
        recon_config().validate(180)
        for field, value in [('max_new_gaussians', 400), ('refit_steps', 20),
                             ('position_step_sigma_cap', 1.), ('rotation_updates', False)]:
            cfg = recon_config(); setattr(cfg, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'ReCon-GS'):
                cfg.validate(180)

    @unittest.skipUnless(torch.cuda.is_available(), 'Native optimizer uses CUDA buffers')
    def test_real_cuda_render_backpropagates_to_newborn_attributes_only(self):
        from libgs.data.utils.graphics import getProjectionMatrix
        from libgs.renderer import Renderer

        torch.set_num_threads(2)
        cfg = ModuleConfig(birth=recon_config())
        module = Module(cfg, SimpleNamespace(current_frame=1, cameras_extent=10.))
        module.trainer = SimpleNamespace(device=torch.device('cuda'),
            global_step=module.motion_steps + 1)
        raw = {k: v.cuda() for k, v in reference_raw().items()}
        raw['xyz'] *= .12; raw['xyz'][:, 2] += 2.
        raw['logscale'][:] = torch.tensor([.03, .05, .07], device='cuda').log()
        module.gaussians = GaussianModel(1)
        for field, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(module.gaussians, attr, torch.nn.Parameter(raw[field].clone()))
        module.gaussians.spatial_lr_scale = 1.; module.gaussians.active_sh_degree = 1
        module.gaussians.training_setup(cfg.gaussian)
        seed_native_optimizer(module.gaussians)
        module._freeze_base()
        module.gaussians.xyz_gradient_accum.fill_(.01)
        module.gaussians.denom.fill_(1.)
        module.births = module._new_birth_model()
        module.births.initialize_recongs_sources(module.gaussians, module.motion_steps,
            cfg.gaussian_stage2, .00015, 10.)
        base_groups = {g['name']: g for g in module.gaussians.optimizer.param_groups}
        for group in module.births.optimizer.param_groups:
            self.assertEqual(group['lr'], base_groups[group['name']]['lr'])
            state = module.births.optimizer.state[group['params'][0]]
            self.assertEqual(float(state['step']), 17.)
            self.assertEqual(int(state['exp_avg'].count_nonzero()), 0)
            self.assertEqual(int(state['exp_avg_sq'].count_nonzero()), 0)
        module._configure_birth_phase(module.trainer.global_step)
        feature_group = next(g for g in module.births.optimizer.param_groups if g['name']=='f_dc')
        self.assertEqual(feature_group['lr'], cfg.gaussian.feature_lr)
        module.renderer = Renderer(cfg.renderer, module.gaussians)
        module.background = torch.zeros(3, device='cuda'); module.opacity_update = None
        module.frame_prepared = module.motion_committed = True; module.frame_finalized = False
        module.birth_control = dict(clone_added=8, split_added=0, pruned=0)
        camera = SimpleNamespace(image=torch.zeros(3, 64, 64, device='cuda'),
            fovx=1., fovy=1., world_view_transform=torch.eye(4, device='cuda'),
            full_proj_transform=getProjectionMatrix(.01, 100., 1., 1.).transpose(0, 1).cuda(),
            camera_center=torch.zeros(3, device='cuda'))
        loss, _, package = module.training_step(camera, module.trainer.global_step)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(package['visibility_filter'][8:].any()))
        for attr in RAW_ATTRIBUTES:
            self.assertIsNone(getattr(module.gaussians, attr).grad)
            self.assertIsNotNone(getattr(module.births, attr).grad)
            self.assertTrue(bool(torch.isfinite(getattr(module.births, attr).grad).all()))
        before = module.births._xyz.detach().clone()
        module.post_training_step(package)
        self.assertFalse(torch.equal(module.births._xyz, before))
        for field, value in raw.items():
            torch.testing.assert_close(model_raw(module.gaussians)[field], value, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Native optimizer uses CUDA buffers')
    def test_native_fit_freezes_base_and_densifies_before_optimizer(self):
        class PoseRenderer:
            def __init__(self, gaussians): self.gaussians = gaussians
            def __call__(self, viewpoint, background):
                xyz = self.gaussians.get_xyz
                points = torch.zeros_like(xyz, requires_grad=True)
                return dict(render=xyz + points * 0, viewspace_points=points,
                    radii=torch.full((len(xyz),), 99., device=xyz.device),
                    visibility_filter=torch.ones(len(xyz), dtype=torch.bool, device=xyz.device))

        cfg = ModuleConfig(birth=recon_config())
        module = Module(cfg, SimpleNamespace(current_frame=1, cameras_extent=2.))
        module.trainer = SimpleNamespace(device=torch.device('cuda'), global_step=0)
        raw = {k: v.cuda() for k, v in reference_raw().items()}
        module.gaussians = GaussianModel(1)
        for field, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(module.gaussians, attr, torch.nn.Parameter(raw[field].clone()))
        module.gaussians.spatial_lr_scale = 1.; module.gaussians.active_sh_degree = 1
        module.gaussians.training_setup(cfg.gaussian)
        module._freeze_base()
        module.renderer = PoseRenderer(module.gaussians)
        module.background = torch.zeros(3, device='cuda'); module.opacity_update = None
        module.frame_prepared = module.motion_committed = True; module.frame_finalized = False
        module.birth_control = dict(clone_added=2, split_added=0, pruned=0)
        module._record_birth_pool_diagnostic = lambda *args: None
        module.births = module._new_birth_model()
        sources = torch.tensor([0, 1], device='cuda')
        born = {k: v[sources].clone() for k, v in raw.items()}
        born['logscale'].fill_(-6.); born['opacity'].fill_(.1)
        module.births.initialize_from_sources(module.gaussians, None, module.motion_steps,
            cfg.gaussian_stage2, proposal=(born, sources))
        for offset in (1, 40):
            module.trainer.global_step = module.motion_steps + offset
            module._configure_birth_phase(module.trainer.global_step)
            before = module.births._xyz.detach().clone()
            package = module.forward(None, training=True)
            package['render'].square().mean().backward()
            module.post_training_step(package)
            if offset == 1:
                self.assertFalse(torch.equal(before, module.births._xyz))
            else:
                # Upstream topology surgery clears parameter gradients before Adam.
                torch.testing.assert_close(module.births._xyz, before, rtol=0, atol=0)
                self.assertEqual(len(module.births._xyz), 2)
                self.assertEqual(len(module.birth_control['densification_events']), 1)
            for field, value in raw.items():
                torch.testing.assert_close(model_raw(module.gaussians)[field], value, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Native optimizer uses CUDA buffers')
    def test_two_frames_deferred_merge_resume_reference_and_exact_replay(self):
        torch.set_num_threads(2)
        raw = {k: v.cuda() for k, v in reference_raw().items()}
        motion = motion_model(raw)
        cfg = ModuleConfig(rigid_tree=motion.config, birth=recon_config(),
                           opacity_update=OpacityUpdateConfig(enabled=True))
        with tempfile.TemporaryDirectory() as directory:
            data = SimpleNamespace(current_frame=1, cameras_extent=2.)
            module = Module(cfg, data)
            worker = SimpleNamespace(device=torch.device('cuda'), output_dir=Path(directory),
                global_step=module.total_steps, config=SimpleNamespace(num_init_steps=15000),
                datamodule=data, module=module)
            module.trainer = worker; module.motion = motion
            module.gaussians = GaussianModel(1)
            for field, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
                setattr(module.gaussians, attr, torch.nn.Parameter(raw[field].clone()))
            module.gaussians.spatial_lr_scale = 1.; module.gaussians.active_sh_degree = 1
            module.gaussians.training_setup(cfg.gaussian)
            seed_native_optimizer(module.gaussians)
            module.renderer = SimpleNamespace(gaussians=module.gaussians)
            module.background = torch.zeros(3, device='cuda')
            module.opacity_update = QuantizedOpacityUpdate(cfg.opacity_update)
            module.opacity_update.setup(raw['opacity'])
            module.opacity_update.commit(torch.arange(8, device='cuda'))
            module.frame_prepared = module.motion_committed = True
            module.frame_finalized = False; module.reference_count = 8
            module.initial_reference = dict(original_count=8, retained_count=8,
                removed_count=0, threshold=0.)
            module.birth_control = dict(clone_added=2, split_added=0, pruned=0,
                retirement=dict(previous_count=8, retired_count=0), previous_birth_count=0)
            module.stage_image_metrics = []; module.retired_reference = None
            module._birth_diagnostic_views = {}
            module.births = module._new_birth_model()
            sources = torch.tensor([0, 1], device='cuda')
            born = {k: v[sources].clone() for k, v in raw.items()}
            born['opacity'] = torch.logit(torch.tensor([[.95], [.001]], device='cuda'))
            born['rotation'][0] = torch.tensor([.7071068, 0., 0., .7071068], device='cuda')
            module.births.initialize_from_sources(module.gaussians, None, 180,
                cfg.gaussian_stage2, proposal=(born, sources))
            self.assertIs(module.make_birth_view(), module.births)
            self.assertIsNot(module.make_birth_view(quantized=True), module.births)
            initial = copy.deepcopy(motion.frame_reference_state)
            motion.record_final_diagnostics()
            Trainer._write_reference(worker, 0, initial)
            module.finalize_frame()
            first = model_raw(module.gaussians)
            for group in module.gaussians.optimizer.param_groups:
                state = module.gaussians.optimizer.state[group['params'][0]]
                self.assertEqual(float(state['step']), 17.)
                torch.testing.assert_close(state['exp_avg'][:8],
                    torch.full_like(state['exp_avg'][:8], .2))
                self.assertEqual(int(state['exp_avg'][8:].count_nonzero()), 0)
            self.assertEqual(len(first['xyz']), 10)
            self.assertEqual(module.frame_report['retired_count'], 0)
            # Current-frame weak birth remains; upstream has no extra terminal prune.
            self.assertLess(float(first['opacity'][-1].sigmoid()), .01)

            module.begin_next_frame(); data.current_frame = 2
            module.prepare_frame()
            self.assertEqual(len(module.gaussians._xyz), 8)
            self.assertEqual(module.motion.previous_rows.tolist(), list(range(1, 9)))
            self.assertEqual(len(module.opacity_update.reference), 10)
            restored = restore_retired_reference(model_raw(module.gaussians),
                module.motion.frame_reference_state, module.motion.previous_rows,
                module.retired_reference)
            for field in RAW_FIELDS:
                torch.testing.assert_close(restored[field], first[field].cpu(), rtol=0, atol=0)
            # Native checkpoint restore retains the deferred merge and full reference.
            checkpoint = dict(current_frame=2, global_step=1)
            module.on_save_checkpoint(checkpoint)
            resumed = Module(copy.deepcopy(cfg), data); resumed.trainer = worker
            resumed.gaussians = GaussianModel(1); resumed.births = resumed._new_birth_model()
            resumed.renderer = SimpleNamespace(gaussians=resumed.gaussians)
            resumed.motion = type(motion)(cfg.rigid_tree)
            resumed.opacity_update = QuantizedOpacityUpdate(cfg.opacity_update)
            resumed.on_load_checkpoint(copy.deepcopy(checkpoint))
            self.assertEqual(resumed.motion.previous_rows.tolist(), list(range(1, 9)))
            module = resumed; worker.module = module; worker.global_step = module.total_steps
            module._commit_opacity_and_retire()
            module.motion_committed = True
            module.motion.record_final_diagnostics()
            module.finalize_frame()
            for state in module.gaussians.optimizer.state.values():
                self.assertEqual(float(state['step']), 17.)
            self.assertEqual(module.frame_report['retired_count'], 2)
            self.assertEqual(module.frame_report['birth_count'], 0)
            stream = Path(directory)/'frame_storage/rigid_tree'
            _, decoded = replay(stream, stream/'reference_frame0.npz', 2)
            for field, value in model_raw(module.gaussians).items():
                torch.testing.assert_close(decoded[field], value.cpu(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
