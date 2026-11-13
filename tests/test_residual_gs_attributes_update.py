"""检查逐个 GS 的残差运动与属性更新。

覆盖逐点属性和透明度的量化、存储正则与优化器状态，
以及逐点运动和坐标重编码；区域共享运动由区域属性更新测试负责。
"""

import copy
import unittest

import torch

from pipeline.model.residual_gs_attributes_update import (
    FIELDS, HEADER, FullUpdateConfig, OpacityUpdateConfig,
    QuantizedOpacityUpdate, ResidualAppearanceAttributes,
)
from pipeline.model.storage_rate import sparse_block_support


class ResidualGSAttributeUpdateTests(unittest.TestCase):

    def test_sh_rest_rate_multiplier_only_changes_its_proximal_pruning(self):
        cfg = FullUpdateConfig(per_attribute_l1_weight=0., point_sh_rest_rate_multiplier=.1)
        ref = dict(logscale=torch.zeros(1, 3), sh_dc=torch.zeros(1, 1, 3),
                   sh_rest=torch.zeros(1, 3, 3))
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = ResidualAppearanceAttributes(cfg)
        update.setup(ref, opt, (.01, .01, .01))
        for name in ('sh_dc', 'sh_rest'):
            param = update.parameters[name]
            with torch.no_grad():
                param.reshape(-1)[0] = .01
            opt.state[param] = {'step': torch.tensor(1.), 'exp_avg': torch.zeros_like(param),
                                'exp_avg_sq': torch.full_like(param, .001)}
        diagnostic = update.rate_terms(.001)
        self.assertAlmostEqual(float(diagnostic['fields']['sh_rest']['rate_multiplier']), .1)
        self.assertEqual(float(diagnostic['fields']['sh_dc']['rate_multiplier']), 1.)
        update.after_step(opt, .001, proximal=True)
        self.assertEqual(float(update.parameters['sh_dc'][0, 0, 0]), 0.)
        self.assertNotEqual(float(update.parameters['sh_rest'][0, 0, 0]), 0.)

    def test_fast_rate_accounting_preserves_loss(self):
        reference = dict(logscale=torch.randn(4, 3), sh_dc=torch.randn(4, 1, 3),
                         sh_rest=torch.randn(4, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = ResidualAppearanceAttributes(FullUpdateConfig(point_logscale_rate_multiplier=2.))
        update.setup(reference, optimizer, (.005, .001, .001))
        with torch.no_grad():
            for parameter in update.parameters.values():
                parameter.normal_(0, .003)
        full = update.rate_terms(.008)
        fast = update.rate_terms(.008, include_scalar_counts=False)
        self.assertTrue(torch.equal(full['loss'], fast['loss']))
        self.assertTrue(torch.equal(full['raw_bytes_surrogate'], fast['raw_bytes_surrogate']))
        self.assertTrue(all('active_scalars' not in item for item in fast['fields'].values()))


    def test_scale_rate_multiplier_only_changes_point_scale_pruning(self):
        cfg = FullUpdateConfig(per_attribute_l1_weight=0., point_logscale_rate_multiplier=2.)
        reference = dict(logscale=torch.zeros(1, 3), sh_dc=torch.zeros(1, 1, 3),
                         sh_rest=torch.zeros(1, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = ResidualAppearanceAttributes(cfg)
        update.setup(reference, optimizer, (.01, .01, .01))
        for name in ('logscale', 'sh_dc'):
            parameter = update.parameters[name]
            with torch.no_grad():
                parameter.reshape(-1)[0] = .01
            optimizer.state[parameter] = {'step': torch.tensor(1.),
                                          'exp_avg': torch.zeros_like(parameter),
                                          'exp_avg_sq': torch.full_like(parameter, .001)}
        diagnostic = update.rate_terms(.0006)
        self.assertEqual(float(diagnostic['fields']['logscale']['rate_multiplier']), 2.)
        self.assertEqual(float(diagnostic['fields']['sh_dc']['rate_multiplier']), 1.)
        update.after_step(optimizer, .0006, proximal=True)
        self.assertEqual(float(update.parameters['logscale'][0, 0]), 0.)
        self.assertNotEqual(float(update.parameters['sh_dc'][0, 0, 0]), 0.)

    def test_zero_full_update_can_only_be_skipped_while_frozen_and_unchanged(self):
        reference = dict(logscale=torch.randn(4, 3), sh_dc=torch.randn(4, 1, 3),
                         sh_rest=torch.randn(4, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = ResidualAppearanceAttributes(FullUpdateConfig())
        update.setup(reference, optimizer, (.01, .01, .01))
        self.assertFalse(update.initial_zero_frozen())
        for parameter in update.parameters.values():
            parameter.requires_grad_(False)
        self.assertTrue(update.initial_zero_frozen())
        for name, result in update.apply(reference).items():
            self.assertTrue(torch.equal(result, reference[name]))
        with torch.no_grad():
            update.parameters['sh_dc'][0, 0, 0] = .01
        self.assertFalse(update.initial_zero_frozen())


    def test_shared_adam_retirement_and_checkpoint(self):
        ref = dict(logscale=torch.zeros(5, 3), sh_dc=torch.zeros(5, 1, 3),
                   sh_rest=torch.zeros(5, 3, 3))
        anchor = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([anchor])
        update = ResidualAppearanceAttributes(FullUpdateConfig(enabled=True))
        update.setup(ref, optimizer, (.01, .02, .003))
        loss = sum(value.sum() for value in update.apply(ref).values())
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        keep = torch.tensor([4, 1, 3])
        previous = {k: v.detach()[keep].clone() for k, v in update.parameters.items()}
        oldstate = {k: optimizer.state[v]['exp_avg'][keep].clone() for k, v in update.parameters.items()}
        update.retain(keep, optimizer)
        for name in FIELDS:
            self.assertTrue(torch.equal(previous[name], update.parameters[name]))
            self.assertTrue(torch.equal(oldstate[name], optimizer.state[update.parameters[name]]['exp_avg']))
        packet = update.capture(optimizer)
        restored_opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        restored = ResidualAppearanceAttributes(update.config)
        restored.restore(packet, restored_opt, 'cpu', 3)
        for name in FIELDS:
            self.assertTrue(torch.equal(restored.parameters[name], update.parameters[name]))
            self.assertEqual(int(restored_opt.state[restored.parameters[name]]['step']), 1)
            for key in ('exp_avg', 'exp_avg_sq'):
                self.assertTrue(torch.equal(restored_opt.state[restored.parameters[name]][key],
                                            optimizer.state[update.parameters[name]][key]))
            old_group = next(g for g in optimizer.param_groups if g.get('name') == 'early_full_' + name)
            new_group = next(g for g in restored_opt.param_groups if g.get('name') == 'early_full_' + name)
            self.assertEqual({k: v for k, v in old_group.items() if k != 'params'},
                             {k: v for k, v in new_group.items() if k != 'params'})


    def test_rate_and_prox_use_attribute_rows_keep_free_bins_and_allow_regrowth(self):
        ref = dict(logscale=torch.zeros(3, 3), sh_dc=torch.zeros(3, 1, 3),
                   sh_rest=torch.zeros(3, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = ResidualAppearanceAttributes(FullUpdateConfig(enabled=True, per_attribute_l1_weight=0.))
        update.setup(ref, optimizer, (.01, .01, .01))
        sum(v.sum() for v in update.apply(ref).values()).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for value in update.parameters.values(): value.zero_()
            update.parameters['logscale'][0, 0] = .002
            update.parameters['logscale'][1, :2] = .01
            update.parameters['sh_dc'][2, 0, 0] = .002
            update.parameters['sh_rest'][1, 0, 0] = -.002
        terms = update.rate_terms(1.)
        self.assertEqual(int(terms['fields']['logscale']['active_rows']), 1)
        self.assertEqual(int(terms['fields']['logscale']['active_scalars']), 2)
        self.assertEqual(int(terms['fields']['sh_dc']['active_rows']), 1)
        self.assertEqual(int(terms['fields']['sh_rest']['active_rows']), 1)
        self.assertEqual(int(terms['raw_bytes_surrogate']), HEADER.size + 12 + 7 + 7 + 19)
        self.assertEqual(float(terms['loss']), float(terms['raw_bytes_surrogate']))
        free = update.parameters['logscale'][0, 0].detach().clone()
        update.after_step(optimizer, 1., proximal=True)
        self.assertEqual(float(update.parameters['logscale'][0, 0]), float(free))
        self.assertFalse(bool(update.parameters['logscale'][1].any()))
        self.assertFalse(bool(update.parameters['sh_dc'][2].any()))
        self.assertFalse(bool(update.parameters['sh_rest'][1].any()))
        # No candidate is frozen; the next image gradient can grow pruned rows.
        update.apply(ref)['logscale'][1].sum().backward()
        self.assertTrue(bool(update.parameters['logscale'].grad[1].ne(0).all()))


    def test_l1_prox_matches_adam_metric_each_step(self):
        ref = dict(logscale=torch.zeros(2, 3), sh_dc=torch.zeros(2, 1, 3),
                   sh_rest=torch.zeros(2, 3, 3))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = ResidualAppearanceAttributes(FullUpdateConfig(enabled=True, per_attribute_l1_weight=.005))
        update.setup(ref, optimizer, (.01, .01, .01))
        sum(v.sum() for v in update.apply(ref).values()).backward()
        optimizer.step()
        before = {name: value.detach().clone() for name, value in update.parameters.items()}
        expected = {}
        for name, value in update.parameters.items():
            state = optimizer.state[value]
            group = next(g for g in optimizer.param_groups if g.get('name') == 'early_full_' + name)
            metric = (state['exp_avg_sq'] / (1 - group['betas'][1] ** int(state['step']))).sqrt() + group['eps']
            threshold = group['lr'] * .005 / value.numel() / metric
            expected[name] = before[name].sign() * (before[name].abs() - threshold).clamp_min(0)
        update.after_step(optimizer, 0., proximal=False)
        for name in FIELDS:
            torch.testing.assert_close(update.parameters[name], expected[name], rtol=0, atol=0)


class PointOpacityUpdateTests(unittest.TestCase):
    def test_hard_block_values_and_zero_bin_has_no_sparse_gradient(self):
        value = torch.tensor([[.02, 0., 0.], [.07, -.08, 0.]], requires_grad=True)
        support = sparse_block_support(value, .1, differentiable=True)
        torch.testing.assert_close(support, torch.tensor([0., 1.]), rtol=0, atol=0)
        support.sum().backward()
        self.assertEqual(float(value.grad[0, 0]), 0.)
        self.assertGreater(float(value.grad[1, 0]), 0.)
        self.assertLess(float(value.grad[1, 1]), 0.)
        self.assertFalse(sparse_block_support(value.detach(), .1).requires_grad)

    def test_zero_opacity_symbols_keep_bits_and_image_gradient(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.1))
        reference = torch.tensor([[-0.], [1.], [-2.]])
        update.setup(reference)
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.02], [.13], [-.08]])
        logits = update()
        self.assertEqual(int(logits.detach().view(torch.int32)[0]),
                         int(reference.view(torch.int32)[0]))
        logits.sigmoid().sum().backward()
        self.assertTrue(bool(update.delta.grad.gt(0).all()))
        self.assertTrue(torch.isfinite(update.delta.grad).all())
        torch.testing.assert_close(update.canonical_logits(), logits.detach(), rtol=0, atol=0)

    def test_opacity_rate_only_pushes_nonzero_quantized_updates(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.05))
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.01], [.03], [-.1]])
        terms = update.rate_terms(1.)
        self.assertEqual(int(terms['active_rows']), 2)
        terms['loss'].backward()
        self.assertEqual(float(update.delta.grad[0]), 0.)
        self.assertGreater(float(update.delta.grad[1]), 0.)
        self.assertLess(float(update.delta.grad[2]), 0.)

    def test_proximal_opacity_adam_keeps_only_image_second_moments(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, proximal_regularization=True,
                                                            l1_weight=.005))
        update.setup(torch.tensor([[0.], [1.], [-1.]]))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.03], [-.1], [.2]])
        image_loss = (update().sigmoid() - .6).square().mean()
        gradient, = torch.autograd.grad(image_loss, [update.delta], retain_graph=True)
        rate = update.rate_terms(.01)['loss']
        self.assertFalse(rate.requires_grad)
        (image_loss + rate).backward()
        update.optimizer_step(0., proximal=False)
        torch.testing.assert_close(update.optimizer.state[update.delta]['exp_avg_sq'],
            gradient.square() * .001, atol=1e-15, rtol=1e-6)

    def test_proximal_opacity_clears_weak_latents_and_retains_large_update(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, proximal_regularization=True))
        update.setup(torch.zeros(3,1))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.01],[.03],[2.]])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(update.delta), exp_avg_sq=torch.full_like(update.delta, 1e-11))
        update.delta.grad = torch.zeros_like(update.delta)
        update.optimizer_step(1e-6, proximal=True)
        self.assertEqual(float(update.delta[0]),0.)
        self.assertEqual(float(update.delta[1]),0.)
        self.assertEqual(float(update.delta[2]),2.)

    def test_continuous_opacity_penalty_cannot_create_opposite_updates(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha',
            proximal_regularization=True, l1_weight=.005, learning_rate=.01))
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:, 0] = torch.tensor([.1, -.1, 1e-5])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(update.delta), exp_avg_sq=torch.full_like(update.delta, 1e-8))
        update.delta.grad = torch.zeros_like(update.delta)
        update.optimizer_step(0., lr_factor=.2)
        self.assertGreater(float(update.delta[0]), 0.)
        self.assertLess(float(update.delta[0]), .1)
        self.assertLess(float(update.delta[1]), 0.)
        self.assertGreater(float(update.delta[1]), -.1)
        self.assertEqual(float(update.delta[2]), 0.)
        self.assertAlmostEqual(update.optimizer.param_groups[0]['lr'], .002)

    def test_alpha_mode_has_unsaturated_image_gradient_and_zero_bits(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01,learning_rate=.01)
        update=QuantizedOpacityUpdate(cfg)
        update.setup(torch.tensor([[-0.],[7.],[-7.]]))
        output=update.alphas()
        output.sum().backward()
        torch.testing.assert_close(update.delta.grad,torch.ones(3,1),atol=0,rtol=0)
        self.assertEqual(int(update().detach().view(torch.int32)[0]),int(update.reference.view(torch.int32)[0]))

    def test_alpha_mode_bounds_and_checkpoint_state(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01)
        update=QuantizedOpacityUpdate(cfg);update.setup(torch.tensor([[7.],[-7.],[0.]]))
        with torch.no_grad():update.delta[:,0]=torch.tensor([20.,-20.,.11])
        update.delta.grad=torch.zeros_like(update.delta)
        update.optimizer_step(0.)
        self.assertTrue(bool(update.delta.ge(-update.alpha_reference).all()))
        self.assertTrue(bool(update.delta.le(1-update.alpha_reference).all()))
        restored=QuantizedOpacityUpdate(copy.deepcopy(cfg));restored.restore(copy.deepcopy(update.capture()),torch.device('cpu'))
        torch.testing.assert_close(update.alphas(),restored.alphas(),atol=0,rtol=0)
        torch.testing.assert_close(update.canonical_logits(),restored.canonical_logits(),atol=0,rtol=0)


import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_regional_attributes_update as regional_fixtures
from pipeline.model import rigid_residual_tree as RIGID, spatial_motion_training as SPATIAL
from pipeline.model.rigid_residual_tree import _quantized
from pipeline.model.rigid_residual_tree import rotation_vector_quaternion
from pipeline.model.spatial_motion_training import SpatialMotionTraining
from pipeline.model.frame_stream_decode import decode_motion
from pipeline.model.frame_stream_encode import encode_motion


class PointSpatialMotionTests(unittest.TestCase):
    model = regional_fixtures.SpatialMotionTrainingTests.model


    def test_rotations_change_orientation_without_rotating_centers(self):
        model = self.model(1)
        with torch.no_grad():
            model.region_rotation[1:, 2] = .3
            model.gs_rotation[:, 2] = .3
        xyz, q = model(1)
        torch.testing.assert_close(xyz, model.base_xyz, rtol=0, atol=0)
        self.assertGreater(float((q - model.base_rotation).abs().max()), .1)

    def test_coordinate_recoding_preserves_gs_optimizer_moments(self):
        model = self.model()
        model._candidate_initialized = model._coding_initialized = True
        model.gs_translation.grad = torch.ones_like(model.gs_translation) * .001
        model.gs_rotation.grad = torch.ones_like(model.gs_rotation) * .001
        model.optimizer.step()
        saved = {name: {key: model.optimizer.state[getattr(model, name)][key].clone()
                       for key in ('exp_avg', 'exp_avg_sq')}
                 for name in ('gs_translation', 'gs_rotation')}
        model._recode_owner(model.spatial_paths[:, 1])
        for name, moments in saved.items():
            for key, before in moments.items():
                torch.testing.assert_close(model.optimizer.state[getattr(model, name)][key], before, rtol=0, atol=0)


class TranslationOnlyMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        source = regional_fixtures.JointSpatialMotionTests().model()
        config = copy.deepcopy(source.config)
        config.rotation_updates = False
        config.proximal_gs_regularization = True
        model = SpatialMotionTraining(config)
        rotation = rotation_vector_quaternion(torch.full_like(source.base_xyz, .2))
        model.setup(source.base_xyz, rotation, source.extent, frame_index=1)
        return model

    def test_translation_learns_with_no_rotation_parameters_or_adam_moments(self):
        model = self.model()
        xyz, orientation = model(1, training=True)
        image = (xyz - model.base_xyz - .01).square().mean()
        terms = model.loss_terms(1, False)
        (image + terms['rate_loss'] + terms['relative_motion_loss']).backward()
        model.optimizer_step()
        self.assertIsNotNone(model.optimizer.state.get(model.gs_translation))
        for parameter in (model.region_rotation, model.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertNotIn(parameter, model.optimizer.state)
            self.assertTrue(bool(parameter.eq(0).all()))
        torch.testing.assert_close(orientation, model.base_rotation, rtol=0, atol=0)


    def test_restore_and_survivor_pruning_keep_motion_rotations_disabled(self):
        model = self.model()
        restored = SpatialMotionTraining(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        rows = torch.arange(0, len(model.owner), 2)
        restored.retain_survivors(rows)
        for parameter in (restored.region_rotation, restored.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertTrue(bool(parameter.eq(0).all()))
        replayed = regional_fixtures.JointSpatialMotionTests().commit(restored)
        torch.testing.assert_close(replayed.rotation, model.base_rotation[rows], rtol=0, atol=0)


class ProximalGSRegularizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = regional_fixtures.JointSpatialMotionTests().model()
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        return model

    def test_point_block_pricing_does_not_depend_on_absolute_id(self):
        model = SpatialMotionTraining(copy.deepcopy(self.model().config))
        count = 16385
        generator = torch.Generator().manual_seed(9)
        xyz = torch.rand((count, 3), generator=generator) * 2 - 1
        rotation = torch.zeros(count, 4)
        rotation[:, 0] = 1
        model.setup(xyz, rotation, 2., frame_index=1)
        self.assertEqual(model._gs_id_lengths[[127, 128, 16383, 16384]].tolist(),
                         [1., 1., 1., 1.])

        # Put two identical blocks on opposite sides of the old 16384 index
        # boundary. Their gain is between the former 8- and 10-byte prices.
        translation, rotation = model.gs_translation, model.gs_rotation
        with torch.no_grad():
            translation.zero_()
            translation[[127, 16384], 0] = .01
        model.last_step = 20
        price = model.effective_rate_loss_weight / model.rate_normalization_bytes
        metric = 2 * (9 * price) / .01**2
        variance = (metric - model.config.adam_eps)**2 * (1 - model.config.adam_betas[1])
        model.optimizer.state[translation] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(translation), exp_avg_sq=torch.full_like(translation, variance))
        model._proximal_blocks(translation, rotation, model.gs_steps[:3], model.gs_steps[3:],
                               model._gs_id_lengths, 1., 1., 0)
        self.assertTrue(bool(translation[[127, 16384], 0].ne(0).all()))

    def test_regularizers_train_regions_and_gates_but_not_gs_adam_moments(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:, 0] = .6 * model.steps[0]
            model.region_rotation[1:, 1] = .6 * model.steps[3]
            model.gs_translation[:, 0] = .7 * model.gs_steps[0]
            model.gs_rotation[:, 1] = .8 * model.gs_steps[3]
        terms = model.loss_terms(5, False)
        gradients = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
            [model.gs_translation, model.gs_rotation, model.region_translation,
             model.region_rotation, model.gate_logits], allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        for gradient in gradients[2:]:
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_soft_threshold_is_exact_metric_prox_and_cannot_cross_zero(self):
        model = self.model()
        model.config.relative_motion_loss_weight = 1.
        model.last_step = 1
        for p, group, scale in [(model.gs_translation, model.optimizer.param_groups[2], model.extent),
                                (model.gs_rotation, model.optimizer.param_groups[3], 1.)]:
            group['lr'] = .01
            model.optimizer.state[p] = dict(step=torch.tensor(1.),
                exp_avg=torch.zeros_like(p), exp_avg_sq=torch.full_like(p, 1-model.config.adam_betas[1]))
            with torch.no_grad():
                p.zero_()
                p[0] = torch.tensor([1e-9, -1e-9, 0.])
                p[1] = torch.tensor([.000501, -.000501, .2])
        original_t, original_r = model.gs_translation.detach().clone(), model.gs_rotation.detach().clone()
        model._proximal_relative_gs()
        for original, p, scale in [(original_t, model.gs_translation, model.extent),
                                   (original_r, model.gs_rotation, 1.)]:
            threshold = .01 / (len(model.owner) * scale * (1 + model.config.adam_eps))
            expected = original.sign() * (original.abs() - threshold).clamp_min(0)
            torch.testing.assert_close(p, expected, rtol=1e-6, atol=1e-10)
            self.assertTrue(bool((p * original >= 0).all()))
            self.assertTrue(bool(p.abs().le(original.abs()).all()))
            self.assertTrue(bool(p[0].eq(0).all()))
        self.assertEqual(float(_quantized(model.gs_rotation, model.gs_steps[3:])[1,0]), 0.)

    def test_one_real_update_stores_only_photometric_gs_second_moment(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = torch.linspace(-.003,.003,len(model.owner))
            model.gs_rotation[:, 1] = torch.linspace(-.002,.002,len(model.owner))
        xyz, rotation = model(1, training=True)
        image_proxy = (xyz-model.base_xyz-.0001).square().mean() + rotation[:,1:].square().mean()
        expected = torch.autograd.grad(image_proxy,[model.gs_translation,model.gs_rotation],retain_graph=True)
        terms = model.loss_terms(1,False)
        (image_proxy+terms['rate_loss']+terms['relative_motion_loss']).backward()
        model.optimizer_step()
        for p,gradient in zip((model.gs_translation,model.gs_rotation),expected):
            torch.testing.assert_close(model.optimizer.state[p]['exp_avg_sq'],
                gradient.square()*(1-model.config.adam_betas[1]),rtol=1e-6,atol=1e-20)

    def test_metric_and_shrink_survive_checkpoint_restore(self):
        model = self.model()
        with torch.no_grad():
            model.gs_rotation[:,0]=.000501
        model.last_step=1
        model.gs_rotation.grad=torch.full_like(model.gs_rotation,1e-6)
        model.optimizer_step()
        captured=copy.deepcopy(model.capture())
        restored=SpatialMotionTraining(copy.deepcopy(model.config))
        restored.restore(captured)
        self.assertTrue(restored.proximal_gs_regularization)
        for a,b in [(model.gs_translation,restored.gs_translation),(model.gs_rotation,restored.gs_rotation)]:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for current in [model,restored]:
            current.last_step=2
            current.gs_rotation.grad=torch.full_like(current.gs_rotation,2e-6)
            current.optimizer_step()
        torch.testing.assert_close(model.gs_rotation,restored.gs_rotation,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
