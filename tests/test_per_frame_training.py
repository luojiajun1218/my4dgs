"""检查当前 200 步逐帧训练的调度与视角划分。

覆盖阶段切换、优化器更新次数，以及训练视角采样与测试视角隔离；
损失公式和具体属性更新分别由 test_loss.py 与属性更新测试负责。
"""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from pipeline.data import DataModule
from pipeline.model.regional_attributes_update import RegionalAttributesUpdate
from pipeline.model.residual_gs_attributes_update import (
    FullUpdateConfig, OpacityUpdateConfig, QuantizedOpacityUpdate,
    ResidualAppearanceAttributes,
)
from pipeline.model.spatial_motion_training import SpatialMotionTraining
from pipeline.module import GaussianConfig, Module
from pipeline.per_frame_training import PerFrameTraining
from pipeline.trainer import Trainer
from tests import test_regional_attributes_update as spatial_tests


def motion_fixture():
    """为逐帧调度和区域属性检查构造相同的 CPU 运动模型。"""
    template = spatial_tests.JointSpatialMotionTests().model()
    config = copy.deepcopy(template.config)
    config.motion_steps = 200
    config.rotation_updates = False
    config.amplitude_refit_steps = 0
    config.proximal_gs_regularization = True
    config.differentiable_rate_support = True
    config.shared_discovery_steps = 100
    gate = config.spatial_gate
    gate.joint_image_gradients = gate.shared_image_recode = gate.shared_discovery = True
    gate.region_motion = 'relative'
    gate.image_relaxation = 'stochastic_ste'
    gate.frame_training_plan = True
    gate.joint_gate_start_step = 151
    gate.freeze_step = 180
    config.validate()
    model = SpatialMotionTraining(config)
    model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
    model.observations.fill_(1)
    return model


class PerFrameTrainingTests(unittest.TestCase):
    def test_one_view_produces_one_optimizer_update(self):
        parameter = torch.tensor(.5, requires_grad=True)
        optimizer = torch.optim.SGD([parameter], lr=.1)
        choose_view = Mock(return_value=[-3.])

        def train(view, step, regularization_weight=1.):
            loss = (parameter - view).square()
            return loss, {'image_loss': loss.detach(), 'psnr': torch.tensor(1.), 'gs': 1}, {}

        def update(package):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        module = SimpleNamespace(
            per_frame_training=SimpleNamespace(frame_plan_training_views=choose_view),
            training_step=Mock(side_effect=train), post_training_step=Mock(side_effect=update),
        )
        trainer = SimpleNamespace(module=module)
        row = Trainer._fit_views(trainer, None, 1, defer_metrics=True)
        loss, metrics = Trainer._materialize_step_metrics([row])[0]

        choose_view.assert_called_once_with(1, 1)
        module.training_step.assert_called_once()
        module.post_training_step.assert_called_once()
        self.assertAlmostEqual(loss, 12.25)
        self.assertAlmostEqual(metrics['image_loss'], 12.25)
        self.assertAlmostEqual(float(parameter), -.2, places=6)

    def test_dataset_rejects_overlapping_training_and_test_cameras(self):
        def dataset(names):
            return SimpleNamespace(items=[SimpleNamespace(path=Path(name)) for name in names],
                                   setup=lambda: None)

        context = SimpleNamespace(
            scene=SimpleNamespace(train_dataset=dataset(['cam01.png', 'cam02.png']),
                                  test_dataset=dataset(['cam00.png'])),
            config=SimpleNamespace(shuffle=False),
        )
        DataModule.setup(context)
        context.scene.test_dataset = dataset(['cam02.png'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            DataModule.setup(context)

    def test_200_step_motion_and_attribute_boundaries(self):
        motion = motion_fixture()
        full_config = FullUpdateConfig(enabled=True, early_joint_repair=True,
                                       dense_color_200=True, regional_full_200=True)
        opacity_config = OpacityUpdateConfig(enabled=True, mode='alpha')
        gaussian_config = GaussianConfig()
        count = len(motion.owner)
        reference = dict(logscale=torch.zeros(count, 3), sh_dc=torch.zeros(count, 1, 3),
                         sh_rest=torch.zeros(count, 3, 3))
        point_attributes = ResidualAppearanceAttributes(full_config)
        point_attributes.setup(reference, motion.optimizer,
                               (gaussian_config.scaling_lr, gaussian_config.feature_lr,
                                gaussian_config.feature_lr / 20))
        opacity = QuantizedOpacityUpdate(opacity_config)
        opacity.setup(torch.zeros(count, 1))
        regional = RegionalAttributesUpdate(motion, full_config, gaussian_config, opacity_config)
        context = SimpleNamespace(
            device='cpu', birth_control={}, motion=motion, opacity_update=opacity,
            full_update_active=True, full_attribute_update=point_attributes,
            regional_full_attributes=regional,
            config=SimpleNamespace(rigid_tree=motion.config, full_update=full_config,
                                   opacity_update=opacity_config, gaussian=gaussian_config),
        )
        context.full_update_prepare_step = lambda step: Module.full_update_prepare_step(context, step)
        controller = PerFrameTraining(context)

        # The gate can reopen during full-attribute fitting; the parameters
        # themselves remain trainable throughout steps 101..200.
        for step, gate_enabled, point_enabled in (
                (1, True, False), (100, True, False), (101, False, True),
                (150, False, True), (151, True, True), (180, True, True),
                (181, False, True), (200, False, True)):
            with self.subTest(step=step):
                controller._early_prepare_rates(step)
                self.assertEqual(motion.gate_logits.requires_grad, gate_enabled)
                self.assertTrue(motion.region_translation.requires_grad)
                self.assertEqual(motion.gs_translation.requires_grad, point_enabled)
                self.assertEqual(opacity.delta.requires_grad, point_enabled)
                for parameter in (*point_attributes.parameters.values(),
                                  *regional.parameters.values()):
                    self.assertEqual(parameter.requires_grad, point_enabled)
        with self.assertRaises(ValueError):
            controller._early_prepare_rates(201)

    def test_training_views_cycle_without_test_camera_or_global_rng_changes(self):
        views = {f'cam{i:02}.png': SimpleNamespace(path=Path(f'cam{i:02}.png'))
                 for i in range(1, 21)}
        context = SimpleNamespace(
            transaction_camera_views=lambda: views,
            config=SimpleNamespace(full_update=SimpleNamespace(audit_camera_sequence=False)),
            datamodule=SimpleNamespace(config=SimpleNamespace(training_view_seed=37)),
            current_frame=1,
        )
        controller = PerFrameTraining(context)
        rng = torch.random.get_rng_state()
        selected = [controller.frame_plan_training_views(step, 1)[0].path.name
                    for step in range(1, 21)]
        self.assertEqual(set(selected), set(views))
        self.assertNotIn('cam00.png', selected)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(selected, [controller.frame_plan_training_views(step, 1)[0].path.name
                                    for step in range(1, 21)])

        # A mistakenly included held-out camera must not silently enter training.
        views.clear()
        views.update({name: SimpleNamespace(path=Path(name))
                      for name in ('cam00.png', 'cam01.png')})
        with self.assertRaisesRegex(ValueError, 'held-out'):
            for step in (1, 2):
                controller.frame_plan_training_views(step, 1)

    def test_training_view_cache_reads_only_training_dataset(self):
        class View:
            def __init__(self, name):
                self.path, self.image = Path(name), torch.zeros(3, 2, 3)

            def to(self, device):
                return self

        class Dataset:
            def __init__(self, names):
                self.items = [View(name) for name in names]

            def __getitem__(self, index):
                return self.items[index]

        train = Dataset(['cam01.png', 'cam02.png', 'cam03.png'])
        test = Dataset(['cam00.png'])
        data = SimpleNamespace(config=SimpleNamespace(training_view_seed=0, expected_image_size=[3, 2]),
                               get_train_dataset=lambda: train, get_test_dataset=lambda: test)
        context = SimpleNamespace(
            _transaction_views=None, datamodule=data, device='cpu',
            trainer=SimpleNamespace(config=SimpleNamespace(cache_data_on_device=False)),
            config=SimpleNamespace(full_update=SimpleNamespace(audit_camera_sequence=False)),
            current_frame=1,
        )
        context.transaction_camera_views = lambda: Module.transaction_camera_views(context)
        controller = PerFrameTraining(context)
        selected = [controller.frame_plan_training_views(step, 1)[0].path.name
                    for step in range(1, 4)]
        self.assertEqual(set(selected), {'cam01.png', 'cam02.png', 'cam03.png'})
        self.assertNotIn('cam00.png', selected)


if __name__ == '__main__':
    unittest.main()
