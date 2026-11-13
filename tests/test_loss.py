"""检查动态帧训练损失的计算与组合。

覆盖图像损失、RMSE 和 SSIM 缓存，检查不同训练阶段的损失分项、存储权重和梯度。
区域及逐点属性更新的具体行为由各自测试文件负责。
"""
import gc
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from pipeline import loss as training_loss
from pipeline.loss import compute_training_loss, mixed_image_loss
from pipeline.model.storage_rate import frame_byte_price


class TrainingLossTests(unittest.TestCase):
    def fixture(self, *, point_frozen=False, rmse_mix=0.):
        value = torch.tensor(2., requires_grad=True)
        gate = torch.nn.Parameter(torch.tensor(.1))
        motion = SimpleNamespace(
            gate_logits=gate, _hard_gate_frozen=False,
            regularization_loss=Mock(side_effect=lambda step: 2 * value),
        )
        opacity = SimpleNamespace(rate_terms=Mock(side_effect=lambda *args, **kwargs: {'loss': 3 * value}))
        full = SimpleNamespace(
            config=SimpleNamespace(storage_loss_enabled=True),
            initial_zero_frozen=lambda: point_frozen,
            rate_terms=Mock(side_effect=lambda *args, **kwargs: {'loss': 5 * value, 'symbols': {}}),
        )
        regional = SimpleNamespace(
            rate_terms=Mock(side_effect=lambda price: {'loss': 7 * value,
                                                       'raw_bytes_surrogate': value.new_tensor(10.)}),
            gate_rate_terms=Mock(side_effect=lambda *args, **kwargs: {
                'proxy': 11 * value, 'hard_weighted_bytes': value.new_tensor(1.),
                'expected_weighted_bytes': value.new_tensor(2.),
            }),
        )
        full_config = SimpleNamespace(
            motion_rmse_mix=rmse_mix, repair_rmse_mix=0., repair_temporal_weight=0.,
            regional_repair=True, attribute_aware_gate=True, point_dc_rate_scale=3.,
        )
        context = SimpleNamespace(
            config=SimpleNamespace(
                motion_coarse_to_fine=False, photometric_loss='l1', lambda_dssim=.2,
                birth=SimpleNamespace(photometric_loss=None, frame_training_plan=True,
                                      joint_old_opacity=True, population_loss_weight=0.),
                opacity_update=SimpleNamespace(joint_with_birth=True),
                rigid_tree=SimpleNamespace(sparse_rate_width_bins=1.,
                    spatial_gate=SimpleNamespace(start_step=0, joint_gate_start_step=151)),
                full_update=full_config,
            ),
            motion=motion, motion_steps=200, motion_committed=False,
            frame_prepared=True, frame_finalized=False, early_initialized=False,
            full_update_active=True, full_attribute_update=full,
            regional_full_attributes=regional, opacity_update=opacity,
            gaussians=SimpleNamespace(_xyz=torch.zeros(2, 3)),
            _rate_byte_price=Mock(return_value=.25), _opacity_rate_price=Mock(return_value=.25),
            forward=Mock(side_effect=lambda view, training: {'render': value * torch.ones(3, 2, 2)}),
        )
        return value, context, SimpleNamespace(image=torch.zeros(3, 2, 2), path='cam01.png')

    def test_stage_and_weighted_storage_components_preserve_gradients(self):
        for step, frozen, expected, gradient, names in (
                (100, True, 4., 2., {'image', 'motion_regularization'}),
                (152, False, 30., 15., {'image', 'motion_regularization',
                                        'opacity_storage', 'point_storage',
                                        'regional_storage', 'attribute_gate'})):
            with self.subTest(step=step):
                value, context, view = self.fixture(point_frozen=frozen)
                components = {'stale': value.detach()}
                with patch('pipeline.loss.motion_photometric_loss', side_effect=lambda image, *args, **kwargs: image.mean()):
                    loss, _, _ = compute_training_loss(context, view, step, .5, components=components)
                self.assertEqual(set(components), names)
                self.assertAlmostEqual(float(loss.detach()), expected)
                self.assertAlmostEqual(sum(float(term) for term in components.values()), expected)
                self.assertTrue(all(not term.requires_grad for term in components.values()))
                loss.backward()
                self.assertAlmostEqual(float(value.grad), gradient)
                if frozen:
                    context.opacity_update.rate_terms.assert_not_called()
                    context.full_attribute_update.rate_terms.assert_not_called()
                    context.regional_full_attributes.rate_terms.assert_not_called()
                else:
                    context.full_attribute_update.rate_terms.assert_called_once_with(
                        .75, include_scalar_counts=False, include_symbols=True)

    def test_zero_regularization_excludes_all_storage_terms(self):
        value, context, view = self.fixture()
        components = {}
        with patch('pipeline.loss.motion_photometric_loss', side_effect=lambda image, *args, **kwargs: image.mean()):
            loss, _, _ = compute_training_loss(context, view, 152, 0., components=components)
        self.assertEqual(set(components), {'image'})
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 2.)
        self.assertAlmostEqual(float(value.grad), 1.)
        context.motion.regularization_loss.assert_not_called()
        context.opacity_update.rate_terms.assert_not_called()
        context.full_attribute_update.rate_terms.assert_not_called()
        context.regional_full_attributes.rate_terms.assert_not_called()

    def test_image_rmse_mix_and_weighted_gradient(self):
        image = torch.tensor([[[1., 3.]]], requires_grad=True)
        target = torch.zeros_like(image)
        weights = torch.tensor([[[1., 3.]]])
        photo = image.abs().mean()
        actual, rmse, _ = mixed_image_loss(image, target, photo, .8, weights)
        mse = (1. + 3. * 9.) / 4.
        expected = .2 * photo + .8 * (mse / ((mse + 1e-8) ** .5 + 1e-4))
        torch.testing.assert_close(actual, expected)
        self.assertGreater(float(rmse), 0.)
        gradient = torch.autograd.grad(actual, image)[0]
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient[0, 0, 1]), float(gradient[0, 0, 0]))

    def test_storage_price_ramp_and_normalization(self):
        config = SimpleNamespace(rate_warmup_steps=10, rate_ramp_steps=4, rate_loss_weight=2.)
        module = SimpleNamespace(config=SimpleNamespace(rigid_tree=config),
                                 motion=SimpleNamespace(rate_normalization_bytes=100.))
        self.assertEqual(frame_byte_price(module, 10), 0.)
        self.assertAlmostEqual(frame_byte_price(module, 12), .01)
        self.assertAlmostEqual(frame_byte_price(module, 14), .02)
        self.assertAlmostEqual(frame_byte_price(module, 200), .02)


class ImageLossTests(unittest.TestCase):
    def test_cached_ssim_matches_loss_and_gradients_across_resolutions(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                image = torch.rand(3, 25, 27, device=device, requires_grad=True)
                target = torch.rand_like(image)
                cache = training_loss._TrainingSSIMCache()
                for factor in (1, 2, 4):
                    expected = training_loss.motion_photometric_loss(image, target, .2, factor)
                    actual = training_loss.motion_photometric_loss(image, target, .2, factor, cache=cache)
                    expected_grad = torch.autograd.grad(expected, image)[0]
                    actual_grad = torch.autograd.grad(actual, image)[0]
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)
                first = cache.prepare(image, target, image.shape[-2:])
                second = cache.prepare(image, target, image.shape[-2:])
                self.assertIs(first[1], second[1])
                target.add_(.1)
                changed = cache.prepare(image, target, image.shape[-2:])
                self.assertIsNot(first[1], changed[1])
                torch.testing.assert_close(
                    training_loss.motion_photometric_loss(image, target, .2, cache=cache),
                    training_loss.motion_photometric_loss(image, target, .2), rtol=1e-6, atol=1e-7)
                cache.clear()
                self.assertEqual(cache._target_bytes, 0)
                self.assertFalse(cache._targets)

    def test_cached_ssim_keeps_target_gradients_and_honours_budget(self):
        image = torch.rand(3, 17, 19, requires_grad=True)
        target = torch.rand_like(image, requires_grad=True)
        cache = training_loss._TrainingSSIMCache(max_bytes=1)
        expected = training_loss.motion_photometric_loss(image, target, .2)
        actual = training_loss.motion_photometric_loss(image, target, .2, cache=cache)
        expected_grads = torch.autograd.grad(expected, (image, target))
        actual_grads = torch.autograd.grad(actual, (image, target))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        training_loss.motion_photometric_loss(image, target.detach(), .2, cache=cache)
        self.assertFalse(cache._targets)
        if torch.cuda.is_available():
            gpu_image = image.detach().cuda()
            gpu_target = target.detach().cuda()
            limited = training_loss._TrainingSSIMCache()
            with patch.object(torch.cuda, 'mem_get_info', return_value=(0, 1024)):
                self.assertIsNone(limited.prepare(gpu_image, gpu_target, gpu_image.shape[-2:]))
            self.assertFalse(limited._targets)

    def test_reused_pixel_squares_preserve_rmse_gradient_and_psnr(self):
        from libgs.metric import psnr
        from pipeline.loss import stabilized_rmse, _stabilized_rmse_from_mse
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            torch.manual_seed(81)
            image = torch.rand(3, 19, 23, device=device, requires_grad=True)
            target = torch.rand_like(image)
            original_loss = stabilized_rmse(image, target)
            original_grad = torch.autograd.grad(original_loss, image)[0]
            squares = (image - target).square()
            reused_loss = _stabilized_rmse_from_mse(squares.mean())
            reused_grad = torch.autograd.grad(reused_loss, image)[0]
            channel_mse = squares.detach().view(image.shape[0], -1).mean(1, keepdim=True)
            reused_psnr = (20 * torch.log10(1. / torch.sqrt(channel_mse))).mean()
            self.assertTrue(torch.equal(original_loss, reused_loss))
            self.assertTrue(torch.equal(original_grad, reused_grad))
            self.assertTrue(torch.equal(psnr(image.detach(), target), reused_psnr))

    def test_rmse_has_finite_zero_and_emphasizes_larger_pixel_errors(self):
        from pipeline.loss import stabilized_rmse
        target = torch.zeros(1, 1, 2)
        exact = target.clone().requires_grad_()
        loss = stabilized_rmse(exact, target)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(exact.grad).all())
        self.assertTrue(torch.equal(exact.grad, torch.zeros_like(exact)))
        image = torch.tensor([[[.4, .02]]], requires_grad=True)
        loss = .75 * (image - target).abs().mean() + .25 * stabilized_rmse(image, target)
        loss.backward()
        self.assertGreater(float(image.grad[0, 0, 0]), float(image.grad[0, 0, 1]))
        self.assertGreater(float(image.grad[0, 0, 1]), 0.)

    def test_ssim_cache_releases_moments_for_expired_targets(self):
        cache = training_loss._TrainingSSIMCache()
        image = torch.zeros(3, 12, 12)
        target = torch.ones_like(image)
        cache.prepare(image, target, (12, 12))
        self.assertEqual(len(cache._targets), 1)
        del target
        gc.collect()
        another_target = torch.zeros_like(image)
        cache.prepare(image, another_target, (12, 12))
        self.assertEqual(len(cache._targets), 1)

if __name__ == '__main__':
    unittest.main()
