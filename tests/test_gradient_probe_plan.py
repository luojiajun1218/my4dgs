import unittest
import copy
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.frame_plan import FramePlanController, gradient_probe_step
from pipeline.model.full_attribute_update import FullUpdateConfig


class ProbePlanTests(unittest.TestCase):
    def test_full_attribute_rates_survive_step_scheduler(self):
        from pipeline.module import Module, _full_attribute_learning_rates
        from pipeline.model.full_attribute_update import FullAttributeUpdate, FIELDS
        args = SimpleNamespace(scaling_lr=.005, feature_lr=.0025)
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3),
                         sh_rest=torch.zeros(4, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = FullAttributeUpdate(FullUpdateConfig())
        rates = _full_attribute_learning_rates(args)
        update.setup(reference, optimizer, tuple(rates[name] for name in FIELDS))
        module = SimpleNamespace(full_update_active=True, full_attribute_update=update,
            motion=SimpleNamespace(optimizer=optimizer), motion_steps=200,
            config=SimpleNamespace(gaussian=args, full_update=SimpleNamespace(early_joint_repair=True),
                                   rigid_tree=SimpleNamespace(motion_lr_final_ratio=.1)))
        for step in (101, 200, 300):
            Module.full_update_prepare_step(module, step)
            groups = {g.get('name'): g for g in optimizer.param_groups}
            self.assertAlmostEqual(groups['early_full_sh_rest']['lr'] /
                                   groups['early_full_sh_dc']['lr'], .05)

    def make_controller(self, count, *, views=2, fraction=None):
        class Harness(FramePlanController):
            def _regional_attribute_parameters(self):
                return self.parameters

        obj = Harness()
        obj.device = 'cpu'
        obj.config = SimpleNamespace(full_update=FullUpdateConfig(
            enabled=True, probe_views_per_round=views, point_budget_fraction=fraction))
        obj.parameters = {name: torch.nn.Parameter(torch.zeros(count, 1))
                          for name in ('xyz', 'rotation', 'opacity', 'logscale', 'sh_dc', 'sh_rest')}
        obj.motion = SimpleNamespace(optimizer=torch.optim.Adam(list(obj.parameters.values())))
        obj._plan_regional_attribute_state = {'eligible': torch.zeros(count, dtype=torch.bool)}
        obj.birth_control = {'regional_attribute_repair': {}}
        obj._plan_step_views = [SimpleNamespace(path='train_camera.png')]
        return obj

    def test_four_view_selection_uses_opacity_without_visibility_gating(self):
        obj = self.make_controller(10000, views=4, fraction=.01)
        alpha = torch.ones(10000)
        alpha[0] = .0001
        obj._render_opacity_alpha = lambda: alpha
        for step in (201, 202, 203, 204, 251, 252, 253, 254):
            for parameter in obj.parameters.values():
                parameter.grad = torch.ones_like(parameter)
            visible = torch.ones(10000, dtype=torch.bool)
            visible[1] = False
            obj._plan_gradient_probe(step, {'visibility_filter': visible})
            if step in (201, 202, 203, 251, 252, 253):
                self.assertEqual(len(obj.birth_control['regional_attribute_repair'].get('selection_rounds', [])),
                                 0 if step < 251 else 1)
        selected = obj._plan_regional_attribute_state['eligible']
        self.assertEqual(int(selected.sum()), 100)
        self.assertFalse(bool(selected[0]))
        # An explicit frustum mask must not suppress an otherwise scored point.
        self.assertTrue(bool(selected[1]))
        self.assertEqual([r['added'] for r in obj.birth_control['regional_attribute_repair']['selection_rounds']],
                         [50, 50])
        self.assertEqual(len(obj.birth_control['regional_attribute_repair']['probes']), 8)
        self.assertEqual(obj.birth_control['regional_attribute_repair']['selection_step'], 254)
        self.assertEqual(obj.birth_control['regional_attribute_repair']['selection_only_views'], 8)

    def test_four_probe_cameras_are_distinct_per_round(self):
        obj = self.make_controller(1000, views=4, fraction=.01)
        obj.full_update_active = True
        obj.config.birth = SimpleNamespace(regional_postfit_admission=False, regional_color_birth=False)
        obj.datamodule = SimpleNamespace(config=SimpleNamespace(training_view_seed=7))
        obj.current_frame = 3
        obj.transaction_camera_views = lambda: {
            f'cam{i:02}.png': SimpleNamespace(path=f'cam{i:02}.png') for i in range(1, 6)}
        for start in (201, 251):
            names = [obj.frame_plan_training_views(step, 1)[0].path for step in range(start, start + 4)]
            self.assertEqual(len(set(names)), 4)
            self.assertNotIn('cam00.png', names)

    def test_probe_can_use_image_gradients_without_extra_opacity_weight(self):
        obj = self.make_controller(1000, views=4, fraction=.01)
        obj.config.full_update.probe_opacity_weight = False
        alpha = torch.ones(1000)
        alpha[0] = .0001
        obj._render_opacity_alpha = lambda: alpha
        for step in (201, 202, 203, 204):
            for parameter in obj.parameters.values():
                parameter.grad = torch.ones_like(parameter)
                parameter.grad[0] = 5
            obj._plan_gradient_probe(step, {})
        self.assertTrue(bool(obj._plan_regional_attribute_state['eligible'][0]))
        self.assertEqual(int(obj._plan_regional_attribute_state['eligible'].sum()), 5)

    def test_early_probe_can_score_only_restricted_attributes(self):
        obj = self.make_controller(1000, views=4, fraction=.01)
        obj.config.full_update.early_joint_repair = True
        obj.config.full_update.early_probe_attribute_only = True
        obj._render_opacity_alpha = lambda: torch.ones(1000)
        for step in (101, 102, 103, 104):
            for name, parameter in obj.parameters.items():
                parameter.grad = torch.ones_like(parameter)
                if name in ('logscale', 'sh_dc', 'sh_rest'):
                    parameter.grad[10:20] = 20
                else:
                    parameter.grad[1] = 1000
            obj._plan_gradient_probe(step, {})
        selected = obj._plan_regional_attribute_state['eligible']
        self.assertEqual(int(selected.sum()), 10)
        self.assertTrue(bool(selected[10:20].all()))
        self.assertFalse(bool(selected[1]))
        self.assertEqual(set(obj.birth_control['regional_attribute_repair']['probes'][0]['median_gradient']),
                         {'logscale', 'sh_dc', 'sh_rest'})

    def test_attribute_only_probes_require_early_joint_repair(self):
        with self.assertRaises(ValueError):
            FullUpdateConfig(enabled=True, early_probe_attribute_only=True).validate()

    def test_shared_orientation_keeps_point_rotation_frozen_during_motion(self):
        from pipeline.model.gaussian import BirthGaussianModel
        from tests import test_full_attribute_update as old_fixture

        def config_adapter(**kwargs):
            kwargs.pop('regional_repair', None)
            return FullUpdateConfig(**kwargs, shared_region_orientation=True)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', old_fixture.cpu_gaussian_setup), \
                patch.object(old_fixture, 'FullUpdateConfig', side_effect=config_adapter):
            module, _ = old_fixture.fixture(directory)
            motion = module.motion
            motion.shared_region_orientation = True
            motion.prepare_plan_step(120)
            self.assertTrue(motion.region_rotation.requires_grad)
            self.assertFalse(motion.gs_rotation.requires_grad)
            with torch.no_grad():
                motion.region_rotation[1, 0] = .02
                motion.gs_rotation[0, 0] = .02
            motion._enforce_rotation_policy()
            self.assertAlmostEqual(float(motion.region_rotation[1, 0]), .02, places=6)
            self.assertEqual(float(motion.gs_rotation[0, 0]), 0.)
            motion.prepare_plan_step(201)
            self.assertFalse(motion.region_rotation.requires_grad)
            self.assertTrue(motion.gs_rotation.requires_grad)

    def test_shared_orientation_survives_real_frame_encoding(self):
        from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
        from pipeline.model.gaussian import BirthGaussianModel
        from pipeline.module import Module
        from pipeline.trainer import Trainer
        from scripts.replay_rigid_tree import decode_frame
        from tests import test_full_attribute_update as old_fixture

        torch.set_num_threads(2)
        torch.manual_seed(9102)
        def config_adapter(**kwargs):
            kwargs.pop('regional_repair', None)
            return FullUpdateConfig(**kwargs, early_joint_repair=True,
                                    probe_views_per_round=4, shared_region_orientation=True)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', old_fixture.cpu_gaussian_setup), \
                patch.object(old_fixture, 'FullUpdateConfig', side_effect=config_adapter):
            module, runner = old_fixture.fixture(directory)
            module.motion.shared_region_orientation = True
            module._render_opacity_alpha = MethodType(Module._render_opacity_alpha, module)
            runner._views_per_step = lambda _: 1
            prior = copy.deepcopy(module.motion.frame_reference_state)
            initial = model_raw(module.gaussians)
            module._plan_diagnostic_reference = {name: value.detach().cpu().clone()
                                                 for name, value in initial.items()}
            worker = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(worker, 0, prior)
            module.trainer.decoder_reference_path = worker.decoder_reference_path
            for step in range(1, 301):
                old_fixture.fit_step(module, runner, step)
                if step == 180:
                    owner = int(module.motion.owner[0])
                    self.assertGreater(owner, 0)
                    with torch.no_grad():
                        module.motion.region_rotation[owner, 0] = .02
                if step == 200:
                    self.assertEqual(int(module.motion.gs_rotation.count_nonzero()), 0)
            directory = Path(directory) / 'frame_storage' / 'rigid_tree'
            self.assertGreater(int(module.motion._values(hard=True)[1].count_nonzero()), 0)
            decoded, raw = decode_frame(directory, 1, prior, initial)
            self.assertEqual(decoded.digest(), module.motion.codec_previous_state.digest())
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], model_raw(module.gaussians)[name].cpu()), name)

    def test_strong_single_view_gradient_is_not_penalized_by_visibility_count(self):
        obj = self.make_controller(1000, views=4, fraction=.01)
        obj._render_opacity_alpha = lambda: torch.ones(1000)
        for step in (201, 202, 203, 204):
            for parameter in obj.parameters.values():
                parameter.grad = torch.ones_like(parameter)
                parameter.grad[0] = 10 if step == 201 else 0
            visible = torch.ones(1000, dtype=torch.bool)
            visible[0] = step == 201
            obj._plan_gradient_probe(step, {'visibility_filter': visible})
        self.assertTrue(bool(obj._plan_regional_attribute_state['eligible'][0]))

    def test_four_view_probe_can_resume_mid_round(self):
        obj = self.make_controller(10000, views=4, fraction=.01)
        obj._render_opacity_alpha = lambda: torch.ones(10000)
        package = {'visibility_filter': torch.ones(10000, dtype=torch.bool)}
        def probe(target, step):
            for parameter in target.parameters.values():
                parameter.grad = torch.arange(10000, dtype=torch.float32).reshape(-1, 1)
            target._plan_gradient_probe(step, package)
        for step in (201, 202):
            probe(obj, step)
        resumed = self.make_controller(10000, views=4, fraction=.01)
        resumed._render_opacity_alpha = lambda: torch.ones(10000)
        resumed._plan_regional_attribute_state = copy.deepcopy(obj._plan_regional_attribute_state)
        for step in (203, 204):
            probe(obj, step)
            probe(resumed, step)
        self.assertTrue(torch.equal(obj._plan_regional_attribute_state['eligible'],
                                    resumed._plan_regional_attribute_state['eligible']))

    def test_temporal_repair_loss_uses_soft_training_image_weights(self):
        from pipeline.module import temporal_repair_weights, weighted_repair_rmse
        target = torch.zeros(3, 11, 11)
        previous = target.clone()
        previous[:, 3:8, 3:8] = .2
        weights = temporal_repair_weights(target, previous, strength=.5, scale=.1)
        self.assertEqual(tuple(weights.shape), (1, 11, 11))
        self.assertGreater(float(weights[0, 5, 5]), float(weights[0, 0, 0]))
        self.assertGreaterEqual(float(weights.min()), 1.)
        self.assertLessEqual(float(weights.max()), 1.5)
        image = torch.full_like(target, .1, requires_grad=True)
        loss = weighted_repair_rmse(image, target, weights)
        loss.backward()
        self.assertGreater(float(image.grad[:, 5, 5].abs().mean()),
                           float(image.grad[:, 0, 0].abs().mean()))
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_two_rounds_only_probe_and_respect_union_budget(self):
        obj = self.make_controller(15000)
        for step in (201, 202, 251, 252):
            for parameter in obj.parameters.values():
                parameter.grad = torch.ones_like(parameter)
            obj._plan_gradient_probe(step, {})
            self.assertTrue(all(parameter.grad is None for parameter in obj.parameters.values()))
            self.assertTrue(all(torch.count_nonzero(parameter) == 0 for parameter in obj.parameters.values()))
            self.assertEqual(len(obj.motion.optimizer.state), 0)
        selected = obj._plan_regional_attribute_state['eligible']
        self.assertEqual(int(selected.sum()), 1000)
        self.assertEqual([r['added'] for r in obj.birth_control['regional_attribute_repair']['selection_rounds']],
                         [500, 500])
        self.assertTrue(bool(selected[:1000].all()))
        self.assertEqual(len(obj.birth_control['regional_attribute_repair']['probes']), 4)

    def test_zero_gradients_do_not_fill_budget(self):
        obj = self.make_controller(1000)
        for step in (201, 202):
            obj._plan_gradient_probe(step, {})
        self.assertEqual(int(obj._plan_regional_attribute_state['eligible'].sum()), 0)

    def test_repair_alpha_matches_decoded_logits_and_keeps_gradient(self):
        from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
        from pipeline.module import Module

        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        update.setup(torch.tensor([[-3.1], [-1.7], [0.4], [1.2]], dtype=torch.float32))
        with torch.no_grad():
            update.delta.copy_(torch.tensor([[.01], [-.02], [.03], [0.]]))
        module = SimpleNamespace(full_update_active=True,
            config=SimpleNamespace(full_update=FullUpdateConfig(enabled=True)),
            trainer=SimpleNamespace(global_step=201), motion_steps=200, opacity_update=update)
        alpha = Module._render_opacity_alpha(module)
        self.assertTrue(torch.equal(alpha, update.canonical_logits().sigmoid()))
        alpha.sum().backward()
        self.assertTrue(torch.equal(update.delta.grad, torch.ones_like(update.delta)))

    def test_shared_orientation_uses_decoder_values_and_keeps_gradient(self):
        from pipeline.module import decoder_exact_rotation
        rotation = torch.tensor([[.7, .2, .1, .6]], requires_grad=True)
        decoded = torch.tensor([[.7000001, .2, .1, .6]])
        motion = SimpleNamespace(decoded_survivor_pose=lambda: (torch.zeros(1, 3), decoded))
        actual = decoder_exact_rotation(motion, rotation)
        self.assertTrue(torch.equal(actual, decoded))
        actual.sum().backward()
        self.assertTrue(torch.equal(rotation.grad, torch.ones_like(rotation)))

    def test_only_the_four_probe_steps_are_selected(self):
        obj = SimpleNamespace(full_update_active=True,
                              config=SimpleNamespace(full_update=SimpleNamespace(regional_repair=True)))
        self.assertEqual([step for step in range(1, 301) if gradient_probe_step(obj, step)],
                         [201, 202, 251, 252])

    def test_other_full_update_training_modes_are_rejected(self):
        FullUpdateConfig(enabled=True).validate()
        with self.assertRaises(ValueError):
            FullUpdateConfig(enabled=True, probe_views_per_round=4,
                             point_budget_fraction=.011).validate()
        for options in ({'repair_selection': 'motion_owner'}, {'color_only': True},
                        {'stop_after_motion': True}, {'train_attributes': False}):
            with self.assertRaises(TypeError):
                FullUpdateConfig(enabled=True, **options)

    def test_cpu_frame_uses_single_views_and_replays_encoded_attributes(self):
        from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
        from pipeline.model.gaussian import BirthGaussianModel
        from pipeline.module import Module
        from pipeline.trainer import Trainer
        from scripts.replay_rigid_tree import decode_frame
        from tests import test_full_attribute_update as old_fixture

        torch.set_num_threads(2)
        torch.manual_seed(9102)
        def config_adapter(**kwargs):
            kwargs.pop('regional_repair', None)
            return FullUpdateConfig(**kwargs)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', old_fixture.cpu_gaussian_setup), \
                patch.object(old_fixture, 'FullUpdateConfig', side_effect=config_adapter):
            module, runner = old_fixture.fixture(directory)
            runner._views_per_step = lambda step: 1
            prior = copy.deepcopy(module.motion.frame_reference_state)
            initial = model_raw(module.gaussians)
            module._plan_diagnostic_reference = {name: value.detach().cpu().clone()
                                                 for name, value in initial.items()}
            worker = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(worker, 0, prior)
            module.trainer.decoder_reference_path = worker.decoder_reference_path
            for step in range(1, 301):
                if step in (202, 252):
                    continue  # The preceding iteration verified the restored second probe.
                old_fixture.fit_step(module, runner, step)
                self.assertEqual(len(module._plan_step_views), 1)
                if step == 200:
                    self.assertEqual(int(module._plan_regional_attribute_state['eligible'].sum()), 0)
                if step in (201, 251):
                    checkpoint = dict(global_step=step, current_frame=1)
                    Module.on_save_checkpoint(module, checkpoint)
                    clone, clone_runner = old_fixture.fixture(directory)
                    clone_runner._views_per_step = lambda _: 1
                    clone.trainer.decoder_reference_path = worker.decoder_reference_path
                    Module.on_load_checkpoint(clone, copy.deepcopy(checkpoint))
                    old_fixture.fit_step(clone, clone_runner, step + 1)
                    old_fixture.fit_step(module, runner, step + 1)
                    self.assertTrue(torch.equal(clone._plan_regional_attribute_state['eligible'],
                                                module._plan_regional_attribute_state['eligible']))
            info = module.birth_control['regional_attribute_repair']
            self.assertIsNotNone(module.frame_report['forward_decode_check'])
            self.assertLessEqual(module.frame_report['forward_decode_check']['max_abs'], 1e-5)
            self.assertEqual([x['step'] for x in info['probes']], [201, 202, 251, 252])
            self.assertLessEqual(info['selected_count'], len(initial['xyz']) // 10)
            decoded, raw = decode_frame(Path(directory) / 'frame_storage' / 'rigid_tree', 1, prior, initial)
            self.assertEqual(decoded.digest(), module.motion.codec_previous_state.digest())
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], model_raw(module.gaussians)[name].cpu()), name)

    def test_early_joint_repair_resumes_and_replays(self):
        from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
        from pipeline.model.gaussian import BirthGaussianModel
        from pipeline.module import Module
        from pipeline.trainer import Trainer
        from scripts.replay_rigid_tree import decode_frame
        from tests import test_full_attribute_update as old_fixture

        torch.set_num_threads(2)
        torch.manual_seed(9102)
        def config_adapter(**kwargs):
            kwargs.pop('regional_repair', None)
            return FullUpdateConfig(**kwargs, early_joint_repair=True, probe_views_per_round=4)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', old_fixture.cpu_gaussian_setup), \
                patch.object(old_fixture, 'FullUpdateConfig', side_effect=config_adapter):
            module, runner = old_fixture.fixture(directory)
            module._render_opacity_alpha = MethodType(Module._render_opacity_alpha, module)
            runner._views_per_step = lambda _: 1
            prior = copy.deepcopy(module.motion.frame_reference_state)
            initial = model_raw(module.gaussians)
            module._plan_diagnostic_reference = {name: value.detach().cpu().clone()
                                                 for name, value in initial.items()}
            worker = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(worker, 0, prior)
            module.trainer.decoder_reference_path = worker.decoder_reference_path
            for step in range(1, 301):
                if step in (103, 151):
                    continue
                old_fixture.fit_step(module, runner, step)
                if step in (102, 150):
                    checkpoint = dict(global_step=step, current_frame=1)
                    Module.on_save_checkpoint(module, checkpoint)
                    clone, clone_runner = old_fixture.fixture(directory)
                    clone._render_opacity_alpha = MethodType(Module._render_opacity_alpha, clone)
                    clone_runner._views_per_step = lambda _: 1
                    clone.trainer.decoder_reference_path = worker.decoder_reference_path
                    Module.on_load_checkpoint(clone, copy.deepcopy(checkpoint))
                    old_fixture.fit_step(clone, clone_runner, step + 1)
                    old_fixture.fit_step(module, runner, step + 1)
                    self.assertTrue(torch.equal(clone._plan_regional_attribute_state['eligible'],
                                                module._plan_regional_attribute_state['eligible']))
            info = module.birth_control['regional_attribute_repair']
            self.assertEqual([x['step'] for x in info['probes']], [101, 102, 103, 104])
            self.assertLessEqual(info['selected_count'], len(initial['xyz']) // 10)
            self.assertLessEqual(module.frame_report['forward_decode_check']['max_abs'], 1e-5)
            decoded, raw = decode_frame(Path(directory) / 'frame_storage' / 'rigid_tree', 1, prior, initial)
            self.assertEqual(decoded.digest(), module.motion.codec_previous_state.digest())
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], model_raw(module.gaussians)[name].cpu()), name)

    def test_early_joint_and_post_motion_shared_dc_replay(self):
        from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
        from pipeline.model.gaussian import BirthGaussianModel
        from pipeline.model.sparse_features import FeatureDCUpdateConfig, RegionalFeatureDCUpdate
        from pipeline.module import Module
        from pipeline.trainer import Trainer
        from scripts.replay_rigid_tree import decode_frame
        from tests import test_full_attribute_update as old_fixture

        torch.set_num_threads(2)
        torch.manual_seed(9102)
        def config_adapter(**kwargs):
            kwargs.pop('regional_repair', None)
            return FullUpdateConfig(**kwargs, early_joint_repair=True, probe_views_per_round=4)
        def create(directory):
            module, runner = old_fixture.fixture(directory)
            module._render_opacity_alpha = MethodType(Module._render_opacity_alpha, module)
            module.config.feature_dc_update = FeatureDCUpdateConfig(enabled=True,
                sharing='spatial_region', post_motion_only=True, joint_with_birth=True)
            module.feature_dc_update = RegionalFeatureDCUpdate(module.config.feature_dc_update, module.motion)
            module.feature_dc_update.setup(module.gaussians._features_dc)
            runner._views_per_step = lambda _: 1
            return module, runner
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(BirthGaussianModel, 'training_setup', old_fixture.cpu_gaussian_setup), \
                patch.object(old_fixture, 'FullUpdateConfig', side_effect=config_adapter):
            module, runner = create(directory)
            prior = copy.deepcopy(module.motion.frame_reference_state)
            initial = model_raw(module.gaussians)
            module._plan_diagnostic_reference = {name: value.detach().cpu().clone()
                                                 for name, value in initial.items()}
            worker = SimpleNamespace(output_dir=Path(directory), module=module, datamodule=module.datamodule)
            Trainer._write_reference(worker, 0, prior)
            module.trainer.decoder_reference_path = worker.decoder_reference_path
            for step in range(1, 301):
                if step in (151, 221):
                    continue
                old_fixture.fit_step(module, runner, step)
                if step == 150:
                    self.assertFalse(module.feature_dc_update.ready)
                if step == 200:
                    self.assertTrue(module.feature_dc_update.ready)
                    self.assertTrue(torch.equal(module.feature_dc_update.owner, module.motion.owner))
                if step in (150, 220):
                    checkpoint = dict(global_step=step, current_frame=1)
                    Module.on_save_checkpoint(module, checkpoint)
                    clone, clone_runner = create(directory)
                    clone.trainer.decoder_reference_path = worker.decoder_reference_path
                    Module.on_load_checkpoint(clone, copy.deepcopy(checkpoint))
                    old_fixture.fit_step(clone, clone_runner, step + 1)
                    old_fixture.fit_step(module, runner, step + 1)
                    if step == 220:
                        self.assertTrue(torch.equal(clone.feature_dc_update.delta, module.feature_dc_update.delta))
            self.assertGreater(int(module.feature_dc_update.symbols().count_nonzero()), 0)
            self.assertLessEqual(module.frame_report['forward_decode_check']['max_abs'], 1e-5)
            decoded, raw = decode_frame(Path(directory) / 'frame_storage' / 'rigid_tree', 1, prior, initial)
            self.assertEqual(decoded.digest(), module.motion.codec_previous_state.digest())
            for name in RAW_FIELDS:
                self.assertTrue(torch.equal(raw[name], model_raw(module.gaussians)[name].cpu()), name)


if __name__ == '__main__':
    unittest.main()
