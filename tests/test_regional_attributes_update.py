"""检查区域属性更新与区域共享运动。

覆盖空间树划分与路由、gate 选择及梯度、区域运动和属性更新，
以及区域变化时与逐点残差的衔接。逐个 GS 的属性优化由对应测试文件负责。
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.model.residual_gs_attributes_update import FullUpdateConfig
from pipeline.model.residual_gs_attributes_update import ResidualAppearanceAttributes
from pipeline.model.regional_attributes_update import RegionalAttributesUpdate, decode_regional_full
from pipeline.model.residual_gs_attributes_update import OpacityUpdateConfig, QuantizedOpacityUpdate


class RegionalAttributesUpdateTest(unittest.TestCase):
    def setUp(self):
        self.motion = SimpleNamespace(parents=torch.arange(4), base_xyz=torch.zeros(4, 3),
                                      owner=torch.tensor([0, 1, 1, 3]))
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.motion.optimizer = torch.optim.Adam([self.anchor])
        self.region = RegionalAttributesUpdate(self.motion, FullUpdateConfig(),
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig())

    def _point_updates(self):
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3),
                         sh_rest=torch.zeros(4, 3, 3))
        full = ResidualAppearanceAttributes(self.region.config)
        full.setup(reference, self.motion.optimizer, (.01, .01, .01))
        opacity = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        opacity.setup(torch.zeros(4, 1))
        return reference, full, opacity

    def _enable_attribute_gate(self, logit=0.):
        self.region.config.attribute_aware_gate = True
        self.motion.gate_logits = torch.nn.Parameter(torch.tensor(logit))
        self.motion.last_step, self.motion._hard_gate_frozen = 151, False
        self.motion.spatial_paths = torch.tensor([[0, 1]] * 4)
        self.motion.owner = torch.zeros(4, dtype=torch.long)
        self.motion._gate_probabilities = lambda: self.motion.gate_logits.sigmoid()
        self.motion._path_weights = lambda p: torch.stack(((1-p).expand(4), p.expand(4)), -1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for cross-device quantization')
    def test_committed_opacity_uses_encoded_cuda_half_bin(self):
        motion = SimpleNamespace(parents=torch.arange(1, device='cuda'),
                                 base_xyz=torch.zeros(1, 3, device='cuda'),
                                 owner=torch.zeros(1, dtype=torch.long, device='cuda'))
        motion.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, device='cuda'))])
        region = RegionalAttributesUpdate(motion, FullUpdateConfig(),
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig(step=.001))
        with torch.no_grad():
            region.parameters['opacity'][0, 0] = -0.07750000059604645
        point_logits = torch.zeros(1, 1)
        actual = region.alpha(torch.full((1, 1), .5, device='cuda'), True, point_logits)
        symbol = region.symbols()['opacity'][0, 0]
        expected = (point_logits + symbol * region.steps['opacity']).cuda().sigmoid()
        self.assertTrue(torch.equal(actual.detach(), expected))

    def test_packet_round_trip_and_unoccupied_nodes(self):
        with torch.no_grad():
            self.region.parameters['sh_dc'][1, 0] = torch.tensor([.003, -.002, .001])
            self.region.parameters['logscale'][3] = torch.tensor([.005, 0., -.005])
            self.region.parameters['opacity'][1] = .03
            self.region.parameters['sh_rest'][2, 0] = 1.  # unoccupied, never transmitted
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3),
                         sh_rest=torch.zeros(4, 3, 3), opacity=torch.zeros(4, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'regional.bin'
            info = self.region.export(path, 1)
            actual = decode_regional_full(path.read_bytes(), reference, self.motion.owner, 4, 1)
            self.assertEqual(info['active_regions']['sh_rest'], 0)
            for name in ('logscale', 'sh_dc', 'sh_rest', 'opacity'):
                expected = self.region.apply(name, reference[name]).detach()
                torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                decode_regional_full(path.read_bytes()[:-1], reference, self.motion.owner, 4, 1)

    def test_zero_quantized_value_retains_image_gradient(self):
        self.region.prepare(101, .2)
        output = self.region.apply('sh_dc', torch.zeros(4, 1, 3))
        output.sum().backward()
        self.assertTrue(self.region.parameters['sh_dc'].grad[1].ne(0).all())

    def test_optimizer_checkpoint_round_trip(self):
        self.region.prepare(101, .2)
        self.region.apply('sh_dc', torch.zeros(4, 1, 3)).sum().backward()
        self.motion.optimizer.step()
        saved = self.region.capture()
        motion = SimpleNamespace(parents=torch.arange(4), base_xyz=torch.zeros(4, 3),
                                 owner=torch.tensor([0, 1, 1, 3]))
        motion.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        restored = RegionalAttributesUpdate(motion, FullUpdateConfig(),
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig())
        restored.restore(saved)
        torch.testing.assert_close(restored.parameters['sh_dc'], self.region.parameters['sh_dc'])
        torch.testing.assert_close(motion.optimizer.state[restored.parameters['sh_dc']]['exp_avg'],
                                   self.motion.optimizer.state[self.region.parameters['sh_dc']]['exp_avg'])

    def test_gate_prices_region_and_rebased_point_rows_without_extra_gradients(self):
        self.motion.gate_logits = torch.nn.Parameter(torch.tensor(0.))
        self.motion._hard_gate_frozen = False
        self.motion.spatial_paths = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.motion.owner = torch.zeros(4, dtype=torch.long)
        self.motion._frontier_probabilities = lambda: self.motion.gate_logits.sigmoid()
        self.motion._rate_gate_probabilities = lambda p: p
        self.motion._path_weights = lambda p: torch.stack(((1-p).expand(4), p.expand(4)), -1)
        self.motion._region_selection_probabilities = lambda p, include_ancestors=False: torch.stack((1-p, p, p*0, p*0))
        _, full, opacity = self._point_updates()
        for name, width in (('logscale', 3), ('sh_dc', 3), ('sh_rest', 9), ('opacity', 1)):
            with self.subTest(name=name):
                with torch.no_grad():
                    for parameter in self.region.parameters.values():
                        parameter.zero_()
                    amount = .2 if name == 'opacity' else self.region.steps[name] * 3
                    self.region.parameters[name].reshape(4, -1)[0, 0] = amount
                self.motion.gate_logits.grad = None
                terms = self.region.gate_rate_terms(full, opacity, 1.)
                cached = full.rate_terms(1., include_symbols=True)['symbols']
                reused = self.region.gate_rate_terms(full, opacity, 1., point_symbols=cached)
                for key in ('proxy', 'hard_weighted_bytes', 'expected_weighted_bytes'):
                    self.assertTrue(torch.equal(terms[key], reused[key]))
                direct_grad = torch.autograd.grad(terms['proxy'], self.motion.gate_logits,
                                                  retain_graph=True)[0]
                reused_grad = torch.autograd.grad(reused['proxy'], self.motion.gate_logits)[0]
                self.assertTrue(torch.equal(direct_grad, reused_grad))
                self.assertEqual(float(terms['proxy']), 0.)
                self.assertEqual(float(terms['hard_weighted_bytes']), 1. + 2.*width)
                terms['proxy'].backward()
                self.assertGreater(float(self.motion.gate_logits.grad), 0.)
                self.assertIsNone(self.region.parameters[name].grad)
                if name != 'opacity':
                    self.assertIsNone(full.parameters[name].grad)
        with torch.no_grad():
            self.region.parameters['opacity'].zero_()
            self.region.parameters['logscale'][0, 0] = .015
        self.motion.gate_logits.grad = None
        self.region.gate_rate_terms(full, opacity, 1.)['proxy'].backward()
        base_gradient = self.motion.gate_logits.grad.clone()
        self.region.config.point_logscale_rate_multiplier = 2.
        self.motion.gate_logits.grad = None
        self.region.gate_rate_terms(full, opacity, 1.)['proxy'].backward()
        self.assertGreater(float(self.motion.gate_logits.grad), float(base_gradient))
        with torch.no_grad():
            for index, (name, value) in enumerate(full.parameters.items()):
                value.reshape(4, -1)[index, 0] = self.region.steps[name] * (index + 2)
        symbols = full.rate_terms(1., include_symbols=True)['symbols']
        direct = self.region.gate_rate_terms(full, opacity, 1.)
        reused = self.region.gate_rate_terms(full, opacity, 1., point_symbols=symbols)
        for key in ('proxy', 'hard_weighted_bytes', 'expected_weighted_bytes'):
            self.assertTrue(torch.equal(direct[key], reused[key]))
        direct_grad = torch.autograd.grad(direct['proxy'], self.motion.gate_logits)[0]
        reused_grad = torch.autograd.grad(reused['proxy'], self.motion.gate_logits)[0]
        self.assertTrue(torch.equal(direct_grad, reused_grad))
        with torch.no_grad():
            for value in self.region.parameters.values():
                value.zero_()
            for value in full.parameters.values():
                value.zero_()
            full.parameters['sh_rest'][0, 0, 0] = .003
        self.region.config.point_sh_rest_rate_multiplier = .1
        discounted = self.region.gate_rate_terms(full, opacity, 1.)
        self.assertAlmostEqual(float(discounted['hard_weighted_bytes']), 1.9, places=6)

    def test_regional_opacity_has_independent_l1_and_diagnostics(self):
        config = FullUpdateConfig(regional_opacity_l1_weight=.005)
        region = RegionalAttributesUpdate(self.motion, config,
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig(l1_weight=.05))
        with torch.no_grad():
            region.parameters['opacity'][1] = .1
        result = region.rate_terms(0.)
        self.assertEqual(region.opacity_l1_weight, .005)
        self.assertEqual(int(result['fields']['opacity']['active_regions']), 1)
        self.assertAlmostEqual(float(result['fields']['opacity']['l1_loss']), .005*.1/3, places=8)

    def test_attribute_gate_gradient_on_real_spatial_tree(self):
        from tests.test_per_frame_training import motion_fixture
        motion = motion_fixture()
        motion.prepare_plan_step(151)
        config = FullUpdateConfig(point_dc_rate_scale=4., point_logscale_rate_multiplier=2.)
        region = RegionalAttributesUpdate(motion, config,
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig(enabled=True, mode='alpha'))
        count = len(motion.owner)
        reference = dict(logscale=torch.zeros(count, 3), sh_dc=torch.zeros(count, 1, 3),
                         sh_rest=torch.zeros(count, 3, 3))
        full = ResidualAppearanceAttributes(config)
        full.setup(reference, motion.optimizer, (.005, .0025, .000125))
        opacity = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha'))
        opacity.setup(torch.zeros(count, 1))
        with torch.no_grad():
            region.parameters['sh_dc'][motion.owner.unique(), 0, 0] = .01
        terms = region.gate_rate_terms(full, opacity, 1.)
        terms['proxy'].backward()
        self.assertTrue(motion.gate_logits.grad.ne(0).any())
        self.assertIsNone(region.parameters['sh_dc'].grad)
        self.assertIsNone(full.parameters['sh_dc'].grad)

    def test_regional_image_gradient_reaches_gate_with_exact_hard_forward(self):
        self._enable_attribute_gate()
        with torch.no_grad():
            self.region.parameters['sh_dc'][0, 0, 0] = .003
            self.region.parameters['sh_dc'][1, 0, 0] = -.002
        reference = torch.zeros(4, 1, 3)
        result = self.region.apply('sh_dc', reference)
        expected = reference + (torch.round(self.region.parameters['sh_dc'].detach()[self.motion.owner] /
                                             self.region.steps['sh_dc']) * self.region.steps['sh_dc'])
        self.assertTrue(torch.equal(result, expected))
        result.sum().backward()
        self.assertNotEqual(float(self.motion.gate_logits.grad), 0.)
        self.assertTrue(self.region.parameters['sh_dc'].grad[0].ne(0).all())
        self.assertFalse(self.region.parameters['sh_dc'].grad[1].ne(0).any())

    def test_regional_opacity_image_gradient_reaches_gate(self):
        self._enable_attribute_gate()
        with torch.no_grad():
            self.region.parameters['opacity'][0, 0] = .10
            self.region.parameters['opacity'][1, 0] = -.05
        alpha = torch.full((4, 1), .5)
        result = self.region.alpha(alpha)
        torch.testing.assert_close(result, torch.sigmoid(torch.full_like(alpha, .10)))
        result.sum().backward()
        self.assertNotEqual(float(self.motion.gate_logits.grad), 0.)
        self.assertTrue(self.region.parameters['opacity'].grad[0].ne(0).all())
        self.assertFalse(self.region.parameters['opacity'].grad[1].ne(0).any())

    def test_shared_image_gate_weights_preserve_outputs_and_gradients(self):
        self._enable_attribute_gate(.4)
        with torch.no_grad():
            for i, (name, value) in enumerate(self.region.parameters.items()):
                value[0].fill_(self.region.steps[name] * (i+2))
                value[1].fill_(-self.region.steps[name] * (i+3))

        def evaluate(shared):
            self.motion.optimizer.zero_grad(set_to_none=True)
            self.motion.gate_logits.grad = None
            weights = self.region.gate_image_weights() if shared else None
            outputs = [self.region.apply(name, torch.zeros_like(value)[self.motion.owner], gate_weights=weights)
                       for name, value in self.region.parameters.items() if name != 'opacity']
            outputs.append(self.region.alpha(torch.full((4, 1), .5), gate_weights=weights))
            sum((i+1)*out.sum() for i, out in enumerate(outputs)).backward()
            gradients = [self.motion.gate_logits.grad.clone()] + [value.grad.clone()
                          for value in self.region.parameters.values()]
            return [out.detach().clone() for out in outputs], gradients

        before, before_grad = evaluate(False)
        after, after_grad = evaluate(True)
        for a, b in zip(before, after):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(before_grad, after_grad):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_unchanged_owner_skips_all_attribute_rebasing(self):
        _, full, opacity = self._point_updates()
        versions = [p._version for p in (*full.parameters.values(), opacity.delta)]
        with patch('pipeline.model.regional_attributes_update._retain_bin_remainder',
                   side_effect=AssertionError('unchanged owners should not be rebased')):
            self.region.rebase_points(self.motion.owner.clone(), full, opacity)
        self.assertEqual(versions, [p._version for p in (*full.parameters.values(), opacity.delta)])

    def test_rebase_preserves_quantized_attributes_on_owner_change(self):
        reference, full, opacity = self._point_updates()
        with torch.no_grad():
            self.region.parameters['sh_dc'][1, 0, 0] = .003
            self.region.parameters['logscale'][1, 1] = .01
            self.region.parameters['sh_rest'][1, 1, 1] = -.003
            self.region.parameters['opacity'][1] = .20
        before = {name:self.region.apply(name, full.apply(reference)[name]).detach()
                  for name in ('logscale', 'sh_dc', 'sh_rest')}
        before_alpha = self.region.alpha(opacity.alphas()).detach()
        previous_owner = self.motion.owner.clone()
        self.motion.owner = torch.tensor([0, 3, 3, 3])
        self.region.rebase_points(previous_owner, full, opacity)
        after = {name:self.region.apply(name, full.apply(reference)[name]).detach()
                 for name in ('logscale', 'sh_dc', 'sh_rest')}
        after_alpha = self.region.alpha(opacity.alphas()).detach()
        for name in before:
            torch.testing.assert_close(after[name], before[name], atol=1e-7, rtol=0.)
        self.assertLess(float((after_alpha-before_alpha).abs().max()), .0051)

    def test_rebase_rejects_unrepresentable_point_residual_before_mutation(self):
        _, full, opacity = self._point_updates()
        previous_owner = self.motion.owner.clone()
        with torch.no_grad():
            full.parameters['sh_dc'][1, 0, 0] = 32767 * self.region.steps['sh_dc']
            self.region.parameters['sh_dc'][1, 0, 0] = self.region.steps['sh_dc']
        before = {k: v.detach().clone() for k, v in full.parameters.items()}
        self.motion.owner = torch.tensor([0, 3, 3, 3])
        with self.assertRaisesRegex(ValueError, 'int16 range'):
            self.region.rebase_points(previous_owner, full, opacity)
        for name, value in full.parameters.items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_rebase_preserves_quantized_field_at_half_grid_boundary(self):
        _, full, opacity = self._point_updates()
        previous = self.motion.owner.clone()
        with torch.no_grad():
            full.parameters['logscale'][1, 0] = .5 * self.region.steps['logscale']
            self.region.parameters['logscale'][1, 0] = self.region.steps['logscale']
        old_symbol = (torch.round(full.parameters['logscale'][1, 0] / self.region.steps['logscale']) +
                      torch.round(self.region.parameters['logscale'][1, 0] / self.region.steps['logscale']))
        self.motion.owner = torch.tensor([0, 3, 3, 3])
        self.region.rebase_points(previous, full, opacity)
        new_symbol = torch.round(full.parameters['logscale'][1, 0] / self.region.steps['logscale'])
        self.assertEqual(float(new_symbol), float(old_symbol))


import copy
import importlib
import itertools
import json
import math
import random
import numpy as np
from pipeline.model import frame_stream_format as _stream_fmt, frame_stream_encode as _stream_enc, frame_stream_decode as _stream_dec
from pipeline.model.frame_stream_format import reference_digest, DecodedState, _validate_state, compose_decoded_survivors, MOTION_MAGICS
from pipeline.model.rigid_residual_tree import RigidTreeConfig, _quantized, rotation_vector_quaternion, _topological_order
from pipeline.model.frame_stream_encode import compress_motion_packet, encode_motion, encode_motion_layout, pack_layout
from pipeline.model.frame_stream_decode import decode_motion, uncompressed_motion_packet, read_stream_step
from pipeline.model.spatial_region_gate import (SPATIAL_ROUTING_FIELDS, spatial_accumulate, spatial_layout,
    spatial_owner, spatial_route, spatial_state_route)
from pipeline.model.spatial_motion_training import SpatialMotionTraining
from scripts.extend_tree_depth import extend_tree_depth

RIGID = importlib.import_module('pipeline.model.rigid_residual_tree')
SPATIAL = importlib.import_module('pipeline.model.spatial_motion_training')
LAYOUT = importlib.import_module('pipeline.model.spatial_region_gate')
MOTION = SPATIAL


class SpatialMotionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, repetitions=16, extent=1.):
        coords = torch.cartesian_prod(torch.tensor([.1, .3, .6, .9]),
                                     torch.tensor([.1, .3, .6, .9]),
                                     torch.tensor([.1, .3, .6, .9]))
        xyz = coords.repeat_interleave(repetitions, 0)
        q = torch.zeros(len(xyz), 4); q[:, 0] = 1
        cfg = RIGID.RigidTreeConfig(region_count=9,
            motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
            motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
            rate_loss_weight=.015, relative_motion_loss_weight=0.)
        cfg.spatial_gate.enabled = True
        cfg.motion_steps = 200
        cfg.spatial_gate.max_depth = 3
        cfg.spatial_gate.min_region_size = 8
        cfg.spatial_gate.training_schedule = 'joint'
        cfg.spatial_gate.learn_region_motion = True
        cfg.spatial_gate.start_step = cfg.spatial_gate.gate_start_step = 0
        cfg.spatial_gate.freeze_step = 180
        model = SPATIAL.SpatialMotionTraining(cfg)
        model.setup(xyz, q, extent, frame_index=1)
        return model

    @staticmethod
    def representative_cost(symbols, candidate, rows, regional_row, price, scales):
        residual = symbols - candidate
        t, r = residual[:, :3].any(1), residual[:, 3:].any(1)
        rt, rr = candidate[:3].any(), candidate[3:].any()
        byte_count = ((t | r) * rows + 6 * (t.astype(int) + r)).sum()
        byte_count += regional_row * bool(rt or rr) + 6 * (int(rt) + int(rr))
        return price * byte_count + (np.abs(residual) * scales).sum()

    def test_statistical_representative_matches_exhaustive_byte_and_l1_objective(self):
        rng = np.random.default_rng(7)
        candidates = np.array(list(itertools.product(range(-1,2), repeat=6)))
        for repeat in range(12):
            symbols = rng.integers(-1,2,size=(8,6),dtype=np.int32)
            rows = rng.integers(2,5,size=8)
            price = 0. if repeat == 0 else float(rng.uniform(0,1))
            scales = np.zeros(6) if repeat == 1 else rng.uniform(0,1,size=6)
            chosen, actual = SPATIAL._rate_relative_representative(symbols,rows,3,price,scales)
            expected = min(self.representative_cost(symbols,c,rows,3,price,scales)
                           for c in candidates)
            self.assertAlmostEqual(actual,expected,places=10)
            self.assertAlmostEqual(actual,self.representative_cost(
                symbols,chosen,rows,3,price,scales),places=10)

    def test_shared_translation_is_not_lost_when_rotations_have_different_symbols(self):
        symbols = np.zeros((10,6),dtype=np.int32)
        symbols[:8,0] = 20
        symbols[:8,3] = np.arange(1,9)
        rows = np.full(10,3)
        chosen, cost = SPATIAL._rate_relative_representative(
            symbols,rows,3,1.,np.zeros(6))
        self.assertEqual(chosen[0],20)
        old = self.representative_cost(symbols,np.zeros(6,int),rows,3,1.,np.zeros(6))
        self.assertLess(cost,old)

    def test_representative_respects_both_region_and_residual_integer_ranges(self):
        symbols = np.zeros((5,6),dtype=np.int32)
        symbols[:,0] = [45000,45001,45002,45003,45004]
        chosen,_ = SPATIAL._rate_relative_representative(
            symbols,np.full(5,3),3,.1,np.ones(6))
        self.assertTrue((np.abs(chosen)<=32767).all())
        self.assertTrue((np.abs(symbols-chosen)<=32767).all())


        # Finite-step quantized fitting may finish one bin from the static
        # majority optimum; a zero initialization is no longer a hard freeze.


    def test_static_planes_reproduce_initial_paths_and_route_empty_cells(self):
        model = self.model(1)
        reference = model.frame_reference_state
        paths = LAYOUT.spatial_route(reference.xyz, reference.spatial_centers,
                                     model.config.spatial_gate.max_depth)
        torch.testing.assert_close(paths, reference.spatial_paths, rtol=0, atol=0)
        # Even a previously empty cell has deterministic descendants.
        outside = torch.tensor([[-10., -10., -10.], [10., 10., 10.]])
        routed = LAYOUT.spatial_route(outside, reference.spatial_centers,
                                      model.config.spatial_gate.max_depth)
        torch.testing.assert_close(reference.parents[routed[:, 1:]], routed[:, :-1])

    def test_crossing_split_changes_next_frame_membership_and_replays_exactly(self):
        model = self.model(1)
        original = model.spatial_paths[0].clone()
        with torch.no_grad():
            model.gs_translation[0] = torch.tensor([.8, .8, .8])
        xyz, q = model.materialized_pose()
        model.register_materialized_layout(xyz, q, torch.empty(0, dtype=torch.long))
        reference = model.frame_reference_state
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'first.bin'
            model.export_motion(path)
            decoded = _stream_dec.decode_motion(path.read_bytes(), reference)
            self.assertTrue(torch.equal(decoded.spatial_paths[0], original))
            resumed = SPATIAL.SpatialMotionTraining(model.config)
            resumed.restore(copy.deepcopy(model.capture()))
            for current in (model, resumed):
                current.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
                self.assertFalse(torch.equal(current.spatial_paths[0], original))
            torch.testing.assert_close(model.spatial_paths, resumed.spatial_paths)
            with torch.no_grad():
                model.region_translation[1:9, 0] = torch.arange(1, 9) * .001
            xyz, q = model.materialized_pose()
            model.register_materialized_layout(xyz, q)
            model.export_motion(Path(directory) / 'second.bin')
            second = _stream_dec.decode_motion((Path(directory) / 'second.bin').read_bytes(), decoded)
            torch.testing.assert_close(second.xyz, xyz, rtol=0, atol=1e-7)
            torch.testing.assert_close(second.spatial_paths, model.spatial_paths)


class JointSpatialMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, initial_logit=-2.):
        axis = torch.linspace(-1., 1., 8)
        xyz = torch.cat((torch.cartesian_prod(axis, axis, axis),
                         torch.tensor([[-100., -100., -100.], [100., 100., 100.]])))
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        config = RigidTreeConfig(motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
            motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
            rate_loss_weight=.002, relative_motion_loss_weight=.01,
            rate_warmup_steps=0, rate_ramp_steps=1, motion_fit_window_steps=20)
        gate = config.spatial_gate
        gate.max_depth = 3
        gate.initialization = 'robust_median'
        gate.initialization_distance_quantile = .99
        gate.outlier_assignment = 'nearest_region'
        gate.training_schedule = 'joint'
        gate.learn_region_motion = True
        gate.min_region_size = 2
        gate.initial_logit = initial_logit
        gate.start_step = gate.gate_start_step = 0
        config.motion_steps = 200
        gate.freeze_step = 180
        config.validate()
        model = SpatialMotionTraining(config)
        model.setup(xyz, q, 2., frame_index=1)
        return model

    def commit(self, model):
        reference = copy.deepcopy(model.frame_reference_state)
        xyz, q = model.materialized_pose()
        sources = torch.empty(0, dtype=torch.long)
        model.register_materialized_layout(xyz, q, sources)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.bin'
            model.export_motion(path)
            decoded = decode_motion(path.read_bytes(), reference)
        torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)
        torch.testing.assert_close(decoded.rotation, q, rtol=0, atol=1e-6)
        return decoded

    def test_far_points_use_nearest_supported_leaf_without_changing_geometry(self):
        model = self.model()
        reference = model.frame_reference_state
        query = torch.tensor([[100., 100., 100.], [-100., -100., -100.],
                              [200., -200., 100.]])
        distances = (query.double()[:, None] - reference.spatial_leaf_centers.double()[None]).square().sum(-1)
        expected = reference.spatial_leaf_nodes[distances.argmin(1)]
        paths = spatial_state_route(query, reference)
        torch.testing.assert_close(paths[:, -1], expected, rtol=0, atol=0)
        torch.testing.assert_close(reference.parents[paths[:, 1:]], paths[:, :-1])
        torch.testing.assert_close(model(1)[0], model.base_xyz, rtol=0, atol=0)
        self.assertEqual(len(model.owner), 514)
        inside = torch.tensor([[0., 0., 0.], [.1, -.2, .3]])
        torch.testing.assert_close(spatial_state_route(inside, reference),
            spatial_route(inside, reference.spatial_centers, 3), rtol=0, atol=0)

    def test_fast_nearest_supported_leaves_preserves_ties_and_fallback(self):
        from pipeline.model import spatial_region_gate as spatial_gate
        if spatial_gate._scipy_cdist is None:
            self.skipTest('SciPy fast distance path unavailable')
        random = np.random.default_rng(19)
        points = random.uniform(-4, 4, size=(73, 3)).astype(np.float32).astype(np.float64)
        points[0] = (0., 0., 0.)
        centers = random.uniform(10, 20, size=(139, 3)).astype(np.float32).astype(np.float64)
        centers[:3] = [[-1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]]
        nodes = np.arange(len(centers), dtype=np.int64) + 100
        actual = spatial_gate._nearest_supported_leaves(points, centers, nodes)
        with patch.object(spatial_gate, '_scipy_cdist', None):
            fallback = spatial_gate._nearest_supported_leaves(points, centers, nodes)
        np.testing.assert_array_equal(actual, fallback)
        self.assertEqual(actual[0], nodes[0])
        cdist = spatial_gate._scipy_cdist
        def rounded_distance(query, representatives, metric):
            result = cdist(query, representatives, metric=metric)
            result[0, 0] = np.nextafter(result[0, 0], np.inf)
            return result
        with patch.object(spatial_gate, '_scipy_cdist', rounded_distance):
            rounded = spatial_gate._nearest_supported_leaves(points[:1], centers, nodes)
        self.assertEqual(rounded[0], nodes[0])

    def test_indexed_nearest_reuses_index_and_detects_tensor_and_numpy_mutation(self):
        from pipeline.model import spatial_region_gate as spatial_gate
        if spatial_gate._scipy_ckdtree is None:
            self.skipTest('SciPy KD-tree unavailable')
        centers = torch.tensor([[-1., 0., 0.], [1., 0., 0.],
                                [-1., 0., 0.], [10., 0., 0.]], dtype=torch.float32)
        points = np.array([[0., 0., 0.], [-1., 0., 0.], [4., 0., 0.]], dtype=np.float64)
        nodes = np.arange(100, 104, dtype=np.int64)
        spatial_gate._NEAREST_INDEX_CACHE.clear()
        original_tree = spatial_gate._scipy_ckdtree
        try:
            with patch.object(spatial_gate, '_scipy_ckdtree', wraps=original_tree) as build:
                def check():
                    expected = spatial_gate._nearest_supported_leaves(points, centers.double().numpy(), nodes)
                    actual = spatial_gate._indexed_nearest_supported_leaves(points, centers, nodes)
                    np.testing.assert_array_equal(actual, expected)
                check()
                check()
                self.assertEqual(build.call_count, 1)
                centers[0, 0] = 100.
                check()
                self.assertEqual(build.call_count, 2)
                # NumPy aliases can mutate storage without changing _version.
                centers.numpy()[1, 0] = 30.
                check()
                self.assertEqual(build.call_count, 3)
            with patch.object(spatial_gate, '_scipy_ckdtree', None):
                check()
        finally:
            spatial_gate._NEAREST_INDEX_CACHE.clear()

    def test_routing_reference_is_shared_once_restored_and_digest_bound(self):
        reference = self.model().frame_reference_state
        packet = {key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
                  for key, value in reference.capture().items() if value is not None}
        restored = DecodedState.from_reference(packet)
        self.assertEqual(restored.digest(), reference.digest())
        restored = DecodedState.restore(reference.capture())
        for name in SPATIAL_ROUTING_FIELDS:
            torch.testing.assert_close(getattr(restored, name), getattr(reference, name), rtol=0, atol=0)
        changed = copy.deepcopy(reference)
        changed.spatial_leaf_centers[0, 0] += .01
        self.assertNotEqual(changed.digest(), reference.digest())

    def test_derived_reference_roundtrips_far_points_and_falls_back_when_routes_differ(self):
        reference = self.model().frame_reference_state
        packet = {key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
                  for key, value in reference.capture().items() if value is not None}
        compact = _stream_fmt.compact_reference_fields(packet)
        self.assertNotIn('owner', compact)
        self.assertNotIn('spatial_paths', compact)
        self.assertEqual(DecodedState.from_reference(compact).digest(), reference.digest())
        for bad_depth in (np.array(0), np.array(6), np.array(3.0)):
            with self.subTest(depth=bad_depth), self.assertRaises(ValueError):
                DecodedState.from_reference({**compact, 'spatial_derived_depth': bad_depth})
        with self.assertRaises(ValueError):
            DecodedState.from_reference({**compact, 'owner': packet['owner']})

        other = ((reference.spatial_paths != reference.spatial_paths[0]).any(1)
                 .nonzero(as_tuple=True)[0][0])
        paths = reference.spatial_paths.clone()
        paths[0] = paths[other]
        changed = {**packet, 'spatial_paths': paths.numpy(),
                   'owner': spatial_owner(paths, reference.spatial_open).numpy()}
        fallback = _stream_fmt.compact_reference_fields(changed)
        self.assertNotIn('spatial_derived_depth', fallback)
        self.assertIn('spatial_paths', fallback)
        self.assertIn('owner', fallback)

    def test_first_dynamic_frame_closes_gates_and_both_motion_fields_learn(self):
        model = self.model()
        self.assertFalse(bool(model.opened[1:].any()))
        self.assertTrue(bool(model.opened[0]))
        self.assertEqual(len(model.owner.unique()), 8)
        self.assertTrue(bool(model.gate_logits.eq(-2.).all()))
        self.assertEqual(model.flat_region_diagnostics['initial_gates'], 'all_eligible_closed')
        self.assertFalse(model._hard_gate_frozen)
        xyz, q = model(1, training=True)
        loss = (xyz[:, 0] - model.base_xyz[:, 0] - .1).square().mean()
        loss = loss + (q[:, 1] - .01).square().mean()
        loss.backward()
        self.assertIsNone(model.gate_logits.grad)
        self.assertIsNone(model.region_translation.grad)
        self.assertIsNone(model.region_rotation.grad)
        model.optimizer_step()
        for step in (2, 3):
            xyz, q = model(step, training=True)
            loss = (xyz[:, 0] - model.base_xyz[:, 0] - .1).square().mean()
            loss = loss + (q[:, 1] - .01).square().mean() + model.regularization_loss(step)
            loss.backward()
            if step == 2:
                model.optimizer_step()
        for parameter in (model.region_translation, model.region_rotation,
                          model.gs_translation, model.gs_rotation):
            self.assertTrue(parameter.requires_grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(model._candidate_initialized)

    def test_first_gate_frontier_uses_configured_initial_logit(self):
        model = self.model(initial_logit=2.)
        self.assertTrue(bool(model.opened[model.split_eligible].all()))
        self.assertTrue(bool(model.gate_logits.eq(2.).all()))
        self.assertEqual(model.flat_region_diagnostics['initial_gates'], 'all_eligible_open')

    def test_region_updates_change_residual_origin_without_changing_full_pose(self):
        model = self.model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.region_translation_lr = .004
        with torch.no_grad():
            model.gs_translation[:, 0] = .03
        before = tuple(value.detach().clone() for value in model(3, training=True))
        model.regularization_loss(3).backward()
        model.optimizer_step()
        self.assertGreater(float(model.region_translation.abs().sum()), 0.)
        self.assertLess(float(model.gs_translation[:, 0].mean()), .03)
        for actual, expected in zip(model(3), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_joint_loss_learns_merges_and_splits_from_either_initialization(self):
        for varying_children in (False, True):
            for initial_logit in (-2., 2.):
                with self.subTest(varying_children=varying_children, initial_logit=initial_logit):
                    model = self.model(initial_logit=initial_logit)
                    parent = int(model.spatial_paths[0, 0])
                    target = model.base_xyz.clone()
                    if varying_children:
                        children = torch.where(model.parents.eq(parent) & model.node_counts.gt(0))[0]
                        for index, child in enumerate(children):
                            target[model.spatial_paths[:, 1].eq(child), 0] += .02 if index % 2 else -.02
                    else:
                        target[:, 0] += .02
                    for step in range(1, 71):
                        model.observations.fill_(1)
                        xyz, _ = model(step, training=True)
                        loss = .5 * ((xyz - target) / .01).square().mean()
                        loss = loss + model.regularization_loss(step)
                        loss.backward()
                        model.optimizer_step()
                    self.assertEqual(bool(model.opened[parent]), varying_children)
                    self.assertLess(float((model(70)[0] - target).square().mean()), 1e-6)

    def test_live_joint_gate_cost_includes_point_residuals(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.linspace(0., .01, len(model.parents))
        zero = model.loss_terms(1)
        with torch.no_grad():
            model.gs_translation[:, 0] = .1
        nonzero = model.loss_terms(1)
        self.assertGreater(float(nonzero['relative_motion']), float(zero['relative_motion']))
        self.assertGreater(float(nonzero['rate_bytes_surrogate']), float(zero['rate_bytes_surrogate']))
        model.regularization_loss(1).backward()
        self.assertGreater(float(model.gate_logits.grad.abs().sum()), 0.)
        self.assertGreater(float(model.gs_translation.grad.abs().sum()), 0.)
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)

    def test_region_selection_rate_scale_changes_only_gate_gradient(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where(model.parents.eq(parent) & model.node_counts.gt(0))[0]
        with torch.no_grad():
            # Parent and children predict the same full motion, so point
            # residual rate is identical on either side of this gate.  Its
            # remaining rate derivative is solely the selected region blocks.
            model.region_translation[parent, 0] = .004
            model.region_translation[children, 0] = .004
            model.gate_logits[parent] = -.2
        values, gradients = [], []
        # Zero is used only as an internal gradient-attribution probe; public
        # configuration validation intentionally rejects it.
        for scale in (1., 0., .25):
            model.config.spatial_gate.region_selection_rate_gradient_scale = scale
            terms = model.loss_terms(10, False)
            values.append(float(terms['rate_bytes_surrogate']))
            gradients.append(torch.autograd.grad(
                terms['rate_loss'], model.gate_logits, retain_graph=False)[0][parent])
        self.assertEqual(values, [values[0]] * 3)
        region_gradient = gradients[0] - gradients[1]
        self.assertGreater(abs(float(region_gradient)), 0.)
        torch.testing.assert_close(gradients[2] - gradients[1], region_gradient * .25,
                                   rtol=1e-6, atol=1e-10)

    def test_region_selection_rate_scale_also_prices_point_residual_choice(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where(model.parents.eq(parent) & model.node_counts.gt(0))[0]
        members = model.owner.eq(parent)
        with torch.no_grad():
            model.region_translation[parent, 0] = .004
            model.region_translation[children, 0] = .024
            model.gs_translation[members, 0] = .02
            model.gate_logits[parent] = -.2
        values, gradients = [], []
        for scale in (1., 0., .25):
            model.config.spatial_gate.region_selection_rate_gradient_scale = scale
            terms = model.loss_terms(10, False)
            values.append(float(terms['rate_bytes_surrogate']))
            gradients.append(torch.autograd.grad(
                terms['rate_loss'], model.gate_logits, retain_graph=False)[0])
        self.assertEqual(values, [values[0]] * 3)
        self.assertGreater(float(gradients[0].abs().sum()), 0.)
        torch.testing.assert_close(gradients[1], torch.zeros_like(gradients[1]),
                                   rtol=0, atol=1e-12)
        torch.testing.assert_close(gradients[2], gradients[0] * .25,
                                   rtol=1e-6, atol=1e-10)

    def test_region_selection_rate_scale_rejects_nonpositive_or_nonfinite_values(self):
        for value in (0., -1., float('inf'), float('nan'), True, 1.01):
            config = copy.deepcopy(self.model().config)
            config.spatial_gate.region_selection_rate_gradient_scale = value
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, 'Region-selection rate gradient scale'):
                config.validate()

    def test_gate_ste_has_discrete_cost_and_the_soft_binary_derivative(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where(model.parents.eq(parent))[0]
        with torch.no_grad():
            model.region_translation[parent, 0] = .004
            model.region_translation[children, 0] = .024
            model.gs_translation[model.owner.eq(parent), 0] = .02
            model.gate_logits[parent] = -.2
        captured = copy.deepcopy(model.capture())
        model._hard_gate_frozen = True
        terms = model.loss_terms(10, False)
        closed_cost = float(terms['rate_loss'] + terms['relative_motion_loss'])
        model._hard_gate_frozen = False
        with torch.no_grad():
            model.gate_logits[parent] = 2.
        model._select_frontier(freeze=True)
        terms = model.loss_terms(10, False)
        open_cost = float(terms['rate_loss'] + terms['relative_motion_loss'])
        model.restore(captured)
        probability = torch.sigmoid(model.gate_logits[parent] / model.config.spatial_gate.temperature)
        probabilities = torch.zeros_like(model.gate_logits)
        probabilities[parent] = probability
        with patch.object(model, '_frontier_probabilities', return_value=probabilities):
            terms = model.loss_terms(10, False)
            loss = terms['rate_loss'] + terms['relative_motion_loss']
            self.assertAlmostEqual(float(loss), closed_cost, places=9)
            actual = torch.autograd.grad(loss, model.gate_logits)[0][parent]
        expected = (open_cost - closed_cost) * probability * (1 - probability) / model.config.spatial_gate.temperature
        self.assertAlmostEqual(float(actual), float(expected), places=9)
        self.assertLess(float(actual), 0.)

    def test_closed_ancestor_does_not_block_descendant_gate_gradient(self):
        model = self.model()
        node = int(model.spatial_paths[0, 1])
        parent = int(model.parents[node])
        self.assertFalse(bool(model.opened[parent]))
        children = torch.where(model.parents.eq(node) & model.node_counts.gt(0))[0]
        with torch.no_grad():
            for index, child in enumerate(children):
                rows = model.spatial_paths[:, -1].eq(child)
                value = .02 if index % 2 else -.02
                model.region_translation[child, 0] = value
                model.gs_translation[rows, 0] = value
        terms = model.loss_terms(10, False)
        (terms['rate_loss'] + terms['relative_motion_loss']).backward()
        self.assertLess(float(model.gate_logits.grad[node]), 0.)

    def test_gate_switch_recodes_residuals_without_pose_jump(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)) % 7 * .002
            model.region_rotation[:, 2] = torch.arange(len(model.parents)) % 5 * .001
            model.gs_translation[:, 1] = .0073
        before = tuple(value.detach().clone() for value in model(5))
        for logit in (2., -2.):
            old_owner = model.owner.clone()
            with torch.no_grad():
                model.gate_logits[1] = logit
            model._select_frontier(freeze=False)
            self.assertTrue(bool(model.owner.ne(old_owner).any()))
            for actual, expected in zip(model(5), before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertGreater(float(model.gs_translation.abs().sum()), 0.)


class JointImageRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = JointSpatialMotionTests().model()
        model.config.spatial_gate.joint_image_gradients = True
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        model.config.rate_model = 'codec'
        model.config.codec_rate_interval = 1
        model.config.validate()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)).remainder(7) * .002
            model.region_rotation[:, 1] = torch.arange(len(model.parents)).remainder(5) * .001
            model.region_translation[0].zero_()
            model.region_rotation[0].zero_()
            model.gs_translation[:, 1] = .004
            model.gs_rotation[:, 2] = .002
        return model

    @staticmethod
    def packet(model):
        rt, rr, gt, gr = model._values(hard=True)
        return encode_motion(model.frame_reference_state, model.frame_index, model.extent,
            model.previous_rows, model.owner, model.parents, rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            model.config.motion_xyz_step_extent, model.config.motion_rotation_step_rad,
            compression='auto', gs_translation_step_extent=model.config.gs_xyz_step_extent,
            gs_rotation_step_rad=model.config.gs_rotation_step_rad, spatial_open=model.opened)

    @staticmethod
    def image_loss(model, step):
        xyz, q = model(step, training=True)
        return (xyz - model.base_xyz - .02).square().mean() + (q[:, 1:] - .01).square().mean()

    def test_image_error_reaches_region_gate_and_point_motion_without_regularizers(self):
        model = self.model()
        gradients = torch.autograd.grad(self.image_loss(model, 1),
            [model.region_translation, model.region_rotation, model.gate_logits,
             model.gs_translation, model.gs_rotation])
        for gradient in gradients:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_regional_image_update_is_not_cancelled_in_point_residuals(self):
        model = self.model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.region_translation_lr = .004
        model.config.relative_motion_loss_weight = model.config.rate_loss_weight = 0.
        model.optimizer.param_groups[-1]['lr'] = 0.
        before = float(self.image_loss(model, 1))
        old_points = model.gs_translation.detach().clone()
        self.image_loss(model, 1).backward()
        model.optimizer_step()
        self.assertLess(float(self.image_loss(model, 1)), before)
        torch.testing.assert_close(model.gs_translation, old_points, rtol=0, atol=0)

    def test_hard_gate_changes_shared_field_without_hidden_point_compensation(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where(model.parents.eq(parent))[0]
        with torch.no_grad():
            model.region_translation[parent, 0] = .002
            model.region_translation[children, 0] = .01
        before = model(1)[0].detach().clone()
        before_symbols = model._full_motion_symbols()[0].detach().clone()
        point = model.gs_translation.detach().clone()
        members = model.owner.eq(parent)
        with torch.no_grad():
            model.gate_logits[parent] = 2.
        model._select_frontier(freeze=False)
        torch.testing.assert_close(model.gs_translation, point, rtol=0, atol=0)
        difference = model._full_motion_symbols()[0] - before_symbols
        self.assertTrue(bool(difference[members, 0].eq(4).all()))
        torch.testing.assert_close(model(1)[0][members, 0] - before[members, 0],
                                   before.new_full((int(members.sum()),), .008), atol=8e-6, rtol=0)

    def test_freezing_gates_keeps_regional_and_point_image_training(self):
        model = self.model()
        model._select_frontier(freeze=True)
        logits = model.gate_logits.detach().clone()
        region = model.region_translation.detach().clone()
        point = model.gs_translation.detach().clone()
        self.image_loss(model, 71).backward()
        self.assertIsNone(model.gate_logits.grad)
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)
        self.assertGreater(float(model.gs_translation.grad.abs().sum()), 0.)
        model.optimizer_step()
        torch.testing.assert_close(model.gate_logits, logits, rtol=0, atol=0)
        self.assertTrue(bool(model.region_translation.ne(region).any()))
        self.assertTrue(bool(model.gs_translation.ne(point).any()))

    def test_forward_rate_matches_production_packet_and_independent_decoder(self):
        model = self.model()
        for logit in (-2., 2.):
            with torch.no_grad():
                model.gate_logits[model.split_eligible] = logit
            model._select_frontier(freeze=False)
            terms = model.loss_terms(2)
            packet = self.packet(model)
            self.assertEqual(float(terms['rate_bytes_surrogate']), len(packet))
            rt, rr, gt, gr = model._values(hard=True)
            self.assertEqual(model._codec_rate.packet(rt, rr, gt, gr, model.opened), packet)
            decoded = decode_motion(packet, model.frame_reference_state)
            torch.testing.assert_close(decoded.xyz, model(2)[0], rtol=0, atol=1e-6)

    def test_rate_rebuilds_survivor_prefix_after_retirement(self):
        model = self.model()
        model.loss_terms(1, False)
        model.retain_survivors(torch.arange(0, len(model.owner), 2))
        terms = model.loss_terms(2, False)
        self.assertEqual(float(terms['rate_bytes_surrogate']), len(self.packet(model)))

    def test_codec_regularizer_keeps_data_only_point_adam_and_finite_gate_gradient(self):
        model = self.model()
        terms = model.loss_terms(3, False)
        gradients = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
            [model.gs_translation, model.gs_rotation, model.region_translation,
             model.region_rotation, model.gate_logits], allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        for gradient in gradients[2:]:
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_closed_candidates_fit_real_images_with_a_binary_forward_cut(self):
        model = self.model()
        terms = model.loss_terms(3, False)
        self.assertFalse(terms['relative_motion_loss'].requires_grad)
        self.assertEqual(float(terms['candidate_motion_loss']), 0.)
        self.assertFalse(terms['candidate_motion_loss'].requires_grad)
        xyz, _ = model(3, training=True)
        rt, _ = model._regional_predictions()
        expected = model.base_xyz + rt[model.owner] + torch.round(
            model.gs_translation / model.gs_steps[:3]) * model.gs_steps[:3]
        torch.testing.assert_close(xyz, expected, rtol=0, atol=0)
        region, = torch.autograd.grad(self.image_loss(model, 3), [model.region_translation])
        closed = model.node_counts.gt(0) & ~model._stored_region_support()
        self.assertGreater(int(closed.sum()), 0)
        self.assertGreater(float(region[closed].abs().sum()), 0.)

    def test_byte_context_prices_match_transmitted_streams_at_varint_boundaries(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation.zero_()
            model.gs_rotation.zero_()
            rows = torch.tensor([0, 1, 129, 260, 513])
            model.gs_translation[rows] = torch.tensor([[0, 1, -1], [63, 64, -65],
                [127, 128, -129], [8191, 8192, -8193], [32767, -32767, 0.]]) * model.gs_steps[:3]
            model.gs_rotation[rows[::2], 1] = 128 * model.gs_steps[4]
        model._measure_codec_rate(1)
        rt, rr, gt, gr = model._values(hard=True)
        total = 0.
        for section, t, r in ((0, rt, rr), (1, gt, gr)):
            st, sr = t.ne(0).any(-1).float(), r.ne(0).any(-1).float()
            total += float(model._codec_rate.costs(t.float(), r.float(), st, sr, section).sum())
        self.assertAlmostEqual(total, model._codec_rate.last_entropy_bytes, places=4)

    def test_rate_calibration_reuses_model_and_diagnostics_force_real_encoding(self):
        model = self.model()
        model.config.codec_rate_interval = 20
        model.config.codec_rate_initial_steps = 0
        first = model.loss_terms(1, False)
        self.assertEqual(float(first['rate_codec_measured']), 1.)
        with patch.object(model._codec_rate, 'measure', wraps=model._codec_rate.measure) as measure:
            estimated = model.loss_terms(2, False)
            measure.assert_not_called()
            self.assertEqual(float(estimated['rate_codec_measured']), 0.)
            self.assertAlmostEqual(float(estimated['rate_bytes_surrogate']),
                                   float(first['rate_bytes_surrogate']), delta=.02)
            actual = model.loss_terms(2, True)
            measure.assert_called_once()
            self.assertEqual(float(actual['rate_codec_measured']), 1.)
            self.assertEqual(float(actual['rate_bytes_surrogate']), len(self.packet(model)))

    def test_rapid_initial_motion_uses_real_packet_lengths_before_periodic_calibration(self):
        model = self.model()
        model.config.codec_rate_interval = 20
        model.config.codec_rate_initial_steps = 20
        for step in (1, 5, 10, 19, 20):
            with torch.no_grad():
                model.gs_translation[:, 0] = step * model.gs_steps[0]
            terms = model.loss_terms(step, False)
            self.assertEqual(float(terms['rate_codec_measured']), 1.)
            self.assertEqual(float(terms['rate_bytes_surrogate']), len(self.packet(model)))
        self.assertEqual(float(model.loss_terms(21, False)['rate_codec_measured']), 0.)

    def test_rate_context_and_joint_updates_survive_resume(self):
        model = self.model()
        model.config.codec_rate_interval = 20
        model.config.codec_rate_initial_steps = 0
        (self.image_loss(model, 1) + model.regularization_loss(1)).backward()
        model.optimizer_step()
        saved = copy.deepcopy(model.capture())
        restored = SpatialMotionTraining(copy.deepcopy(model.config))
        restored.restore(saved)
        for current in (model, restored):
            (self.image_loss(current, 2) + current.regularization_loss(2)).backward()
            current.optimizer_step()
        for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation', 'gate_logits'):
            torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
        self.assertEqual(float(model.loss_terms(3, False)['rate_bytes_surrogate']),
                         float(restored.loss_terms(3, False)['rate_bytes_surrogate']))


class RelativeSpatialMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        m = JointSpatialMotionTests().model()
        m.config.spatial_gate.region_motion = 'relative'
        m.config.proximal_gs_regularization = True
        m.config.differentiable_rate_support = True
        m.observations.fill_(1)
        return m

    def packet(self, m, compression='auto'):
        rt, rr, gt, gr = m._values(hard=True)
        return encode_motion(m.frame_reference_state, m.frame_index, m.extent,
            m.previous_rows, m.owner, m.parents, rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            m.config.motion_xyz_step_extent, m.config.motion_rotation_step_rad,
            compression=compression, gs_translation_step_extent=m.config.gs_xyz_step_extent,
            gs_rotation_step_rad=m.config.gs_rotation_step_rad, spatial_open=m.opened,
            spatial_relative=True)

    def test_integer_corrections_accumulate_without_int16_overflow(self):
        m = self.model()
        with torch.no_grad():
            m.region_translation[1:, 0] = 20000 * m.steps[0]
            m.gate_logits[m.split_eligible] = 4.
        # Construct a large valid pose directly. Recoding a coarse pose into
        # this field would itself need a residual outside signed-int16 range.
        m.opened = m.split_eligible.clone()
        m.opened[0] = True
        m.owner = spatial_owner(m.spatial_paths, m.opened)
        m._hard_gate_frozen = True
        pred = m._regional_predictions(hard=True)[0]
        self.assertTrue(bool(pred[m.owner, 0].eq(60000).all()))
        for compression in ('none', 'auto', 'zlib'):
            data = self.packet(m, compression)
            self.assertEqual(uncompressed_motion_packet(data)[:4], b'SPG2')
            decoded = decode_motion(data, m.frame_reference_state)
            torch.testing.assert_close(decoded.xyz, m(100)[0], atol=1e-6, rtol=0)

    def test_selected_path_keeps_parent_corrections_and_discards_hidden_nodes(self):
        m = self.model()
        parent = int(m.owner[0])
        child = int(m.spatial_paths[0, 1])
        with torch.no_grad():
            m.region_translation[parent, 0] = .004
            m.region_translation[child, 0] = .006
            m.gate_logits[parent] = 4.
        m._select_frontier(freeze=True)
        rt, _, _, _ = m._values(hard=True)
        self.assertEqual(int(rt[parent, 0]), 2)
        self.assertEqual(int(rt[child, 0]), 3)
        self.assertFalse(bool(m.owner.eq(parent).any()))
        self.assertGreater(int(m._stored_region_support().sum()), m.owner.unique().numel())
        decoded = decode_motion(self.packet(m), m.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, m(100)[0], atol=1e-6, rtol=0)

    def test_ancestor_support_stops_at_identity_in_mixed_depth_frontiers(self):
        m = self.model()
        for logit in (-4., 4.):
            with torch.no_grad():
                m.gate_logits[m.split_eligible] = logit
                # Retain a coarse region while other branches can be deep.
                m.gate_logits[1] = -4.
            m._select_frontier(freeze=False)
            expected = torch.zeros_like(m.opened)
            reach = torch.ones(len(m.owner), dtype=torch.bool)
            for nodes in m.spatial_paths.unbind(1):
                expected[nodes[reach]] = True
                reach &= m.opened[nodes]
            torch.testing.assert_close(m._stored_region_support(), expected, atol=0, rtol=0)
            self.assertEqual(int(expected.sum()), len(m.owner.unique()) +
                sum(bool(expected[n] and m.opened[n]) for n in range(1,len(expected))))

    def test_closed_candidate_fits_without_training_gate_target_or_parent(self):
        m = self.model()
        with torch.no_grad():
            m.gs_translation[:, 0] = .012
            m.gate_logits.fill_(-6)
        m._select_frontier(freeze=False)
        loss = m.loss_terms(5, False)['candidate_motion_loss']
        grad = torch.autograd.grad(loss, [m.region_translation, m.gs_translation,
            m.gate_logits], allow_unused=True)
        self.assertIsNone(grad[1])
        self.assertIsNone(grad[2])
        self.assertEqual(float(grad[0][1:9].abs().sum()), 0.)
        self.assertGreater(float(grad[0][9:].abs().sum()), 0.)

    def test_gate_gradient_opens_shared_correction_without_rule_edits(self):
        m = self.model()
        parent = int(m.owner[0])
        children = torch.where(m.parents.eq(parent) & m.node_counts.gt(0))[0]
        with torch.no_grad():
            m.region_translation[parent, 0] = .02
            m.gs_translation[m.owner.eq(parent), 0] = .01
            for k, child in enumerate(children):
                rows = m.spatial_paths[:, 1].eq(child)
                correction = -.01 if k % 2 else .01
                m.gs_translation[rows, 0] = correction
                m.region_translation[child, 0] = correction
            m.gate_logits.fill_(-4.)
            m.gate_logits[parent] = -.2
        full = tuple(x.detach().clone() for x in m._full_motion_symbols())
        optimizer = torch.optim.Adam([m.gate_logits], lr=.15, eps=1e-12)
        initial = float(m.loss_terms(5, False)['rate_bytes_surrogate'])
        for step in range(1, 31):
            optimizer.zero_grad(set_to_none=True)
            m.loss_terms(step, False)['rate_loss'].backward()
            optimizer.step()
            m._select_frontier(freeze=False)
        self.assertTrue(bool(m.opened[parent]))
        self.assertLess(float(m.loss_terms(31, False)['rate_bytes_surrogate']), initial)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))


    def test_explicit_migration_preserves_all_predictions_and_next_frame_warm_start(self):
        m = JointSpatialMotionTests().model()
        with torch.no_grad():
            m.region_translation[:, 0] = torch.arange(len(m.parents)).remainder(5) * .002
            m.region_rotation[:, 1] = torch.arange(len(m.parents)).remainder(7) * .001
            m.gs_translation[:, 2] = .0024
        before = tuple(x.detach().clone() for x in m._regional_predictions(hard=True))
        full = tuple(x.clone() for x in m._full_motion_symbols())
        JointSpatialMotionTests().commit(m)
        m.enable_relative_regions()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, m._regional_predictions(hard=True))))
        xyz, q = m.codec_previous_state.xyz, m.codec_previous_state.rotation
        m.setup(xyz, q, m.extent, frame_index=2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))

    def test_relative_checkpoint_resumes_the_same_update(self):
        m = self.model()
        with torch.no_grad():
            m.gs_translation[:, 0] = .003
        m(1, training=True)
        (m(1)[0].square().mean() + m.regularization_loss(1)).backward()
        m.optimizer_step()
        saved = copy.deepcopy(m.capture())
        restored = SpatialMotionTraining(copy.deepcopy(m.config))
        restored.restore(saved)
        for model in (m, restored):
            (model(2, training=True)[0].square().mean() + model.regularization_loss(2)).backward()
            model.optimizer_step()
        for key in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation', 'gate_logits'):
            torch.testing.assert_close(getattr(m, key), getattr(restored, key), atol=0, rtol=0)


class GateCandidateRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = JointSpatialMotionTests().model(initial_logit=-3.)
        model.config.spatial_gate.candidate_refresh_interval = 20
        model.config.spatial_gate.temporal_logit_decay = .5
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        model.observations.fill_(1)
        parent = int(model.owner[0])
        body = model.owner.eq(parent)
        with torch.no_grad():
            direction = model.spatial_paths[body, 1].remainder(2).float() * 2 - 1
            model.gs_translation[body, 0] = direction * 20 * model.gs_steps[0]
        return model, parent, body

    def test_refresh_preserves_selected_predictors_pose_and_gates(self):
        model, _, _ = self.model()
        before = model(20)[0].detach().clone()
        selected = model._stored_region_support()
        rt, gt = model.region_translation.detach().clone(), model.gs_translation.detach().clone()
        gates = model.gate_logits.detach().clone()
        opened = model.opened.clone()
        p = model.region_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
            exp_avg_sq=torch.ones_like(p))
        self.assertGreater(model._refresh_unselected_candidates(), 0)
        torch.testing.assert_close(model(20)[0], before, rtol=0, atol=0)
        torch.testing.assert_close(model.region_translation[selected], rt[selected], rtol=0, atol=0)
        torch.testing.assert_close(model.gs_translation, gt, rtol=0, atol=0)
        torch.testing.assert_close(model.gate_logits, gates, rtol=0, atol=0)
        self.assertTrue(torch.equal(model.opened, opened))
        self.assertTrue(bool(model.optimizer.state[p]['exp_avg'][selected].eq(1).all()))
        refreshed = ~selected & model.node_counts.gt(0)
        self.assertTrue(bool(model.optimizer.state[p]['exp_avg'][refreshed].eq(0).all()))

    def test_gate_learns_shared_child_motion_and_reduces_coded_blocks(self):
        model, parent, body = self.model()
        before = model(20)[0].detach().clone()
        initial_bytes = float(model.loss_terms(20, False)['rate_bytes_surrogate'])
        model._refresh_unselected_candidates()
        optimizer = torch.optim.Adam([model.gate_logits], lr=.15)
        for step in range(21, 71):
            optimizer.zero_grad(set_to_none=True)
            model.regularization_loss(step).backward()
            optimizer.step()
            with torch.no_grad():
                model.gate_logits.clamp_(-6, 6)
                model._select_frontier(freeze=False, preserve_remainder=True)
        self.assertGreater(float(model.gate_logits[parent]), 0.)
        self.assertTrue(bool(model.owner[body].ne(parent).all()))
        self.assertTrue(bool(torch.round(model.gs_translation[body] / model.gs_steps[:3]).eq(0).all()))
        self.assertLess(float(model.loss_terms(70, False)['rate_bytes_surrogate']), initial_bytes)
        torch.testing.assert_close(model(70)[0], before, rtol=0, atol=1e-6)

    def test_refresh_stops_after_gate_freeze_and_validates_settings(self):
        model, _, _ = self.model()
        model._hard_gate_frozen = True
        with self.assertRaises(ValueError):
            model._refresh_unselected_candidates()
        for value in (True, -1, .5):
            model.config.spatial_gate.candidate_refresh_interval = value
            with self.assertRaises(ValueError):
                model.config.validate()

    def test_temporal_decay_preserves_decisions_with_less_confidence(self):
        helper = JointSpatialMotionTests()
        model = helper.model()
        model.config.spatial_gate.temporal_logit_decay = .5
        with torch.no_grad():
            model.gate_logits.fill_(-6.)
        decoded = helper.commit(model)
        opened = decoded.spatial_open.clone()
        model.setup(decoded.xyz, decoded.rotation, 2., frame_index=2)
        torch.testing.assert_close(model.gate_logits, torch.full_like(model.gate_logits, -3.), rtol=0, atol=0)
        self.assertTrue(torch.equal(model.opened, opened))


class SpatialInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cloud(self):
        axis = torch.linspace(-1., 1., 16)
        core = torch.cartesian_prod(axis, axis, axis)
        return torch.cat((core, torch.tensor([[-100., -100., -100.],
                                             [100., 100., 100.]])))

    def model(self, xyz):
        config = RIGID.RigidTreeConfig()
        config.spatial_gate.max_depth = 3
        config.spatial_gate.initialization = 'robust_median'
        config.validate()
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        model = MOTION.SpatialMotionTraining(config)
        model.setup(xyz, q, extent=2., frame_index=1)
        return model

    def test_routing_matches_scalar_oracle_at_and_around_split_planes(self):
        rng = np.random.default_rng(20261105)
        depth = 5
        centers = rng.normal(size=(sum(8 ** i for i in range(depth)), 3)).astype(np.float32)
        boundary = centers[:64].copy()
        points = np.concatenate((rng.normal(size=(256, 3)).astype(np.float32), boundary,
            np.nextafter(boundary, np.float32(np.inf)),
            np.nextafter(boundary, np.float32(-np.inf))))
        expected = []
        for point in points:
            parent, code, offset, path = 0, 0, 1, []
            for level in range(depth):
                octant = sum((1 << axis) for axis in range(3)
                             if point[axis] >= centers[parent, axis])
                code = code * 8 + octant
                parent = offset + code
                path.append(parent)
                offset += 8 ** (level + 1)
            expected.append(path)
        actual = LAYOUT.spatial_route(torch.from_numpy(points), torch.from_numpy(centers), depth)
        np.testing.assert_array_equal(actual.numpy(), np.asarray(expected, dtype=np.int64))
        self.assertEqual(tuple(LAYOUT.spatial_route(torch.empty((0, 3)),
            torch.from_numpy(centers), depth).shape), (0, depth))
        with self.assertRaises(ValueError):
            LAYOUT.spatial_route(torch.empty((0, 3)), torch.empty((0, 3)), 0)

    def test_farther_outliers_cannot_move_supported_split_planes(self):
        xyz = self.cloud()
        pushed = xyz.clone()
        pushed[-2:] *= 10000.
        support = LAYOUT.spatial_initialization_support(xyz, 'robust_median')
        self.assertFalse(bool(support[-2:].any()))
        first = LAYOUT.spatial_layout(xyz, 5, True, initialization='robust_median')
        second = LAYOUT.spatial_layout(pushed, 5, True, initialization='robust_median')
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # All points, including the farthest two, retain a valid full path.
        self.assertEqual(first[0].shape, (len(xyz), 5))
        routed = LAYOUT.spatial_route(pushed, first[3], 5)
        torch.testing.assert_close(routed, first[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1][routed[:, 1:]], routed[:, :-1])

    def test_support_never_prunes_or_clamps_model_geometry(self):
        xyz = self.cloud()
        original = xyz.clone()
        model = self.model(xyz)
        torch.testing.assert_close(model.base_xyz, original, rtol=0, atol=0)
        torch.testing.assert_close(model(1)[0], original, rtol=0, atol=0)
        self.assertEqual(len(model.owner), len(original))
        self.assertEqual(len(model.parents), 585)

    def test_degenerate_and_invalid_clouds_have_defined_behavior(self):
        xyz = torch.ones(32, 3)
        result = LAYOUT.spatial_layout(xyz, 3, True, initialization='robust_median')
        self.assertTrue(bool(torch.isfinite(result[3]).all()))
        torch.testing.assert_close(LAYOUT.spatial_route(xyz, result[3], 3), result[0])
        for q in (0., .5, 1., float('nan'), True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                LAYOUT.spatial_layout(xyz, 3, initialization='robust_median', distance_quantile=q)
        for bad in (torch.zeros(0, 3), torch.tensor([[float('nan'), 0., 0.]]),
                    torch.tensor([[float('inf'), 0., 0.]])):
            with self.assertRaises(ValueError):
                LAYOUT.spatial_layout(bad, 3)


def reference_fixture():
    generator = torch.Generator().manual_seed(20261102)
    xyz = torch.rand((512, 3), generator=generator) * 2 - 1
    xyz = torch.cat((xyz, torch.tensor([[-30., -30., -30.], [30., 30., 30.]])))
    rotation = torch.zeros((len(xyz), 4)); rotation[:, 0] = 1
    paths, parents, opened, centers, routing = spatial_layout(
        xyz, 4, return_centers=True, initialization='robust_median',
        distance_quantile=.99, outlier_assignment='nearest_region', return_routing=True)
    raw = dict(xyz=xyz, rotation=rotation, logscale=torch.zeros(len(xyz), 3),
               opacity=torch.zeros(len(xyz), 1), sh_dc=torch.zeros(len(xyz), 1, 3),
               sh_rest=torch.zeros(len(xyz), 3, 3))
    packet = {name: value.numpy() for name, value in raw.items()}
    packet.update(owner=spatial_owner(paths, opened).numpy(), parents=parents.numpy(),
                  frame_index=np.array(0), extent=np.array(2.), region_count=np.array(len(parents)),
                  spatial_paths=paths.numpy(), spatial_open=opened.numpy(),
                  spatial_centers=centers.numpy(),
                  **{key: value.numpy() for key, value in routing.items()})
    return packet, raw


def motion_payload(prior, frame, opened, *, inherit_rotation=False):
    paths = spatial_state_route(prior.xyz, prior)
    owner = spatial_owner(paths, opened)
    rows = torch.arange(len(prior.xyz))
    region_t = torch.zeros((len(prior.parents), 3), dtype=torch.int16)
    region_r = torch.zeros_like(region_t)
    deepest = int(paths[:, -1].max())
    assert deepest > 32767
    region_t[deepest] = torch.tensor([5, -3, 2], dtype=torch.int16)
    gs_t = torch.zeros((len(rows), 3), dtype=torch.int16)
    gs_r = torch.zeros_like(gs_t)
    gs_t[1] = torch.tensor([-2, 1, 4], dtype=torch.int16)
    sources = torch.empty(0, dtype=torch.long)
    xyz, rotation = compose_decoded_survivors(
        prior, 2., rows, owner, prior.parents, region_t, region_r, gs_t, gs_r,
        .001, .001, gs_translation_step_extent=.001, gs_rotation_step_rad=.001,
        spatial_open=opened, spatial_relative=True, inherit_birth_rotation=inherit_rotation)
    empty_xyz = torch.empty(0, 3)
    arguments = (prior, frame, 2., rows, owner, prior.parents, region_t, region_r,
                 gs_t, gs_r, sources, empty_xyz, rotation[sources], .001, .001)
    options = dict(gs_translation_step_extent=.001, gs_rotation_step_rad=.001,
                   spatial_open=opened, spatial_relative=True,
                   inherit_birth_rotation=inherit_rotation, birth_pose_steps=(.002, .001))
    return encode_motion(*arguments, compression='none', **options), deepest


class DepthFiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.old, cls.raw = reference_fixture()
        cls.new = extend_tree_depth(cls.old)
        cls.prior = DecodedState.from_reference(cls.new)
        _validate_state(cls.prior)

    def frontier(self):
        paths = self.prior.spatial_paths
        row = int(paths[:, -1].argmax())
        opened = self.prior.spatial_open.clone()
        opened[paths[row, :-1]] = True
        return opened

    def test_extension_preserves_previous_routes_and_all_non_layout_fields(self):
        self.assertEqual(self.prior.spatial_paths.shape[1], 5)
        self.assertEqual(len(self.prior.parents), 37449)
        np.testing.assert_array_equal(self.old['spatial_paths'], self.new['spatial_paths'][:, :4])
        np.testing.assert_array_equal(self.old['parents'], self.new['parents'][:4681])
        changed = {'parents', 'region_count', 'spatial_paths', 'spatial_open',
                   'spatial_centers', 'spatial_leaf_nodes'}
        for key in self.old.keys() - changed:
            np.testing.assert_array_equal(self.old[key], self.new[key], err_msg=key)

    def test_extension_accepts_compact_reference_with_derived_routes(self):
        compact = _stream_fmt.compact_reference_fields(self.old)
        self.assertIn('spatial_derived_depth', compact)
        self.assertNotIn('owner', compact)
        extended = extend_tree_depth(compact)
        self.assertNotIn('spatial_derived_depth', extended)
        state = DecodedState.from_reference(extended)
        _validate_state(state)
        np.testing.assert_array_equal(state.spatial_paths.numpy()[:, :4], self.old['spatial_paths'])
        np.testing.assert_array_equal(state.owner.numpy(), self.old['owner'])
        for key in ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest'):
            np.testing.assert_array_equal(extended[key], self.old[key], err_msg=key)

    def test_reference_template_loads_and_raw_gate_cost_is_counted(self):
        metadata = []
        with tempfile.TemporaryDirectory() as folder:
            for depth, reference in ((4, self.old), (5, self.new)):
                path = Path(folder) / f'reference{depth}.npz'
                np.savez_compressed(path, **reference)
                cfg = RigidTreeConfig()
                cfg.spatial_gate.max_depth = depth
                cfg.spatial_gate.initialization = 'robust_median'
                cfg.spatial_gate.outlier_assignment = 'nearest_region'
                cfg.spatial_gate.reference_layout = str(path)
                cfg.spatial_gate.training_schedule = 'joint'
                cfg.spatial_gate.start_step = cfg.spatial_gate.gate_start_step = 0
                cfg.validate()
                model = SpatialMotionTraining(cfg)
                model.setup(self.raw['xyz'], self.raw['rotation'], 2., frame_index=1)
                np.testing.assert_array_equal(model.spatial_paths.cpu().numpy(), reference['spatial_paths'])
                metadata.append(model._metadata_bytes())
        self.assertEqual(metadata[1] - metadata[0], 4097)  # bitmap + region-count varint

    def test_quantized_motion_codec_and_stream_parser_support_large_node_ids(self):
        from pipeline.model.frame_stream_encode import (compress_motion_packet)
        opened = self.frontier()
        raw_packet, deepest = motion_payload(self.prior, 1, opened)
        self.assertEqual(raw_packet[:4], b'SPG6')
        packed = compress_motion_packet(raw_packet)
        self.assertEqual(uncompressed_motion_packet(packed), raw_packet)
        decoded = decode_motion(packed, self.prior)
        self.assertTrue(torch.equal(decoded.spatial_open, opened))
        self.assertTrue(torch.equal(decoded.owner, spatial_owner(self.prior.spatial_paths, opened)))
        expected_raw = {**self.raw, 'xyz': decoded.xyz, 'rotation': decoded.rotation}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'motion_001.bin').write_bytes(packed)
            manifest = dict(frame=1, motion='motion_001.bin', birth=None, opacity_update=None,
                            previous_count=len(self.prior.xyz), old_count=len(self.prior.xyz),
                            birth_count=0, total_count=len(decoded.xyz),
                            decoded_state_sha256=decoded.digest().hex(),
                            raw_state_sha256=reference_digest(expected_raw))
            (directory / 'frame_001.json').write_text(json.dumps(manifest))
            step = read_stream_step(directory, 1, region_count=37449, depth=5, rest=3)
            np.testing.assert_array_equal(step.opened, opened.numpy())
            np.testing.assert_array_equal(step.regional_symbols[deepest], [5, -3, 2])
            self.assertEqual(step.stored_bytes, sum(p.stat().st_size for p in directory.iterdir()))
        inherited, _ = motion_payload(self.prior, 1, opened, inherit_rotation=True)
        self.assertEqual(inherited[:4], b'SPG8')
        self.assertEqual(decode_motion(inherited, self.prior).digest(), decoded.digest())


class TopologicalOrderTests(unittest.TestCase):
    def test_breadth_first_tree_returns_exact_existing_order(self):
        for dtype in (torch.int16, torch.int32, torch.int64):
            parents=torch.tensor([-1,0,0,1,1,2,2],dtype=dtype)
            self.assertEqual(_topological_order(parents),list(range(1,7)))
        self.assertEqual(_topological_order(torch.tensor([-1])),[])

    def test_arbitrary_valid_numbering_keeps_depth_first_order(self):
        parents=torch.tensor([-1,2,0,4,2])
        self.assertEqual(_topological_order(parents),[2,1,4,3])
        self.assertEqual(_topological_order(parents.tolist()),[2,1,4,3])

    def test_invalid_roots_parents_and_cycles_are_still_rejected(self):
        for values in ([],[0],[-1,-1],[-1,1],[-1,2,1],[-1,99]):
            with self.subTest(values=values),self.assertRaises(ValueError):
                _topological_order(torch.tensor(values,dtype=torch.long))

    def test_random_permuted_trees_preserve_parent_before_child(self):
        generator=random.Random(20261105)
        for _ in range(50):
            count=generator.randrange(2,100)
            original=[-1]+[generator.randrange(node) for node in range(1,count)]
            permutation=[0]+generator.sample(range(1,count),count-1)
            parents=[-1]*count
            for old in range(1,count):
                parents[permutation[old]]=permutation[original[old]]
            order=_topological_order(torch.tensor(parents,dtype=torch.long))
            self.assertEqual(set(order),set(range(1,count)))
            index={node:i for i,node in enumerate([0,*order])}
            self.assertTrue(all(index[parents[node]]<index[node] for node in range(1,count)))


if __name__ == '__main__':
    unittest.main()
