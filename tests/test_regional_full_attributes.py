import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pipeline.model.full_attribute_update import FullUpdateConfig
from pipeline.model.full_attribute_update import FullAttributeUpdate
from pipeline.model.regional_full_attributes import RegionalFullAttributes, decode_regional_full
from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate


class RegionalFullAttributesTest(unittest.TestCase):
    def setUp(self):
        self.motion = SimpleNamespace(parents=torch.arange(4), base_xyz=torch.zeros(4, 3),
                                      owner=torch.tensor([0, 1, 1, 3]))
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.motion.optimizer = torch.optim.Adam([self.anchor])
        self.region = RegionalFullAttributes(self.motion, FullUpdateConfig(),
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for cross-device quantization')
    def test_committed_opacity_uses_encoded_cuda_half_bin(self):
        motion = SimpleNamespace(parents=torch.arange(1, device='cuda'),
                                 base_xyz=torch.zeros(1, 3, device='cuda'),
                                 owner=torch.zeros(1, dtype=torch.long, device='cuda'))
        motion.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, device='cuda'))])
        region = RegionalFullAttributes(motion, FullUpdateConfig(),
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
        restored = RegionalFullAttributes(motion, FullUpdateConfig(),
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
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3), sh_rest=torch.zeros(4, 3, 3))
        full = FullAttributeUpdate(self.region.config)
        full.setup(reference, self.motion.optimizer, (.01, .01, .01))
        opacity = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        opacity.setup(torch.zeros(4, 1))
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
        region = RegionalFullAttributes(self.motion, config,
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig(l1_weight=.05))
        with torch.no_grad():
            region.parameters['opacity'][1] = .1
        result = region.rate_terms(0.)
        self.assertEqual(region.opacity_l1_weight, .005)
        self.assertEqual(int(result['fields']['opacity']['active_regions']), 1)
        self.assertAlmostEqual(float(result['fields']['opacity']['l1_loss']), .005*.1/3, places=8)

    def test_attribute_gate_gradient_on_real_spatial_tree(self):
        from tests.test_frame_plan import motion_fixture
        motion = motion_fixture()
        motion.prepare_plan_step(151)
        config = FullUpdateConfig(point_dc_rate_scale=4., point_logscale_rate_multiplier=2.)
        region = RegionalFullAttributes(motion, config,
            SimpleNamespace(scaling_lr=.005, feature_lr=.0025), OpacityUpdateConfig(enabled=True, mode='alpha'))
        count = len(motion.owner)
        reference = dict(logscale=torch.zeros(count, 3), sh_dc=torch.zeros(count, 1, 3),
                         sh_rest=torch.zeros(count, 3, 3))
        full = FullAttributeUpdate(config)
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
        self.region.config.attribute_aware_gate = True
        self.motion.gate_logits = torch.nn.Parameter(torch.tensor(0.))
        self.motion.last_step, self.motion._hard_gate_frozen = 151, False
        self.motion.spatial_paths = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.motion.owner = torch.zeros(4, dtype=torch.long)
        self.motion._gate_probabilities = lambda: self.motion.gate_logits.sigmoid()
        self.motion._path_weights = lambda p: torch.stack(((1-p).expand(4), p.expand(4)), -1)
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
        self.region.config.attribute_aware_gate = True
        self.motion.gate_logits = torch.nn.Parameter(torch.tensor(0.))
        self.motion.last_step, self.motion._hard_gate_frozen = 151, False
        self.motion.spatial_paths = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.motion.owner = torch.zeros(4, dtype=torch.long)
        self.motion._gate_probabilities = lambda: self.motion.gate_logits.sigmoid()
        self.motion._path_weights = lambda p: torch.stack(((1-p).expand(4), p.expand(4)), -1)
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
        self.region.config.attribute_aware_gate = True
        self.motion.gate_logits = torch.nn.Parameter(torch.tensor(.4))
        self.motion.last_step, self.motion._hard_gate_frozen = 151, False
        self.motion.spatial_paths = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
        self.motion.owner = torch.zeros(4, dtype=torch.long)
        self.motion._gate_probabilities = lambda: self.motion.gate_logits.sigmoid()
        self.motion._path_weights = lambda p: torch.stack(((1-p).expand(4), p.expand(4)), -1)
        with torch.no_grad():
            for i, value in enumerate(self.region.parameters.values()):
                value[0].fill_(self.region.steps[list(self.region.parameters)[i]] * (i+2))
                value[1].fill_(-self.region.steps[list(self.region.parameters)[i]] * (i+3))

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
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3), sh_rest=torch.zeros(4, 3, 3))
        full = FullAttributeUpdate(self.region.config)
        full.setup(reference, self.motion.optimizer, (.01, .01, .01))
        opacity = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        opacity.setup(torch.zeros(4, 1))
        versions = [p._version for p in (*full.parameters.values(), opacity.delta)]
        with patch('pipeline.model.regional_full_attributes._retain_bin_remainder',
                   side_effect=AssertionError('unchanged owners should not be rebased')):
            self.region.rebase_points(self.motion.owner.clone(), full, opacity)
        self.assertEqual(versions, [p._version for p in (*full.parameters.values(), opacity.delta)])

    def test_rebase_preserves_quantized_attributes_on_owner_change(self):
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3), sh_rest=torch.zeros(4, 3, 3))
        full = FullAttributeUpdate(self.region.config)
        full.setup(reference, self.motion.optimizer, (.01, .01, .01))
        opacity = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        opacity.setup(torch.zeros(4, 1))
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
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3), sh_rest=torch.zeros(4, 3, 3))
        full = FullAttributeUpdate(self.region.config)
        full.setup(reference, self.motion.optimizer, (.01, .01, .01))
        opacity = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        opacity.setup(torch.zeros(4, 1))
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
        reference = dict(logscale=torch.zeros(4, 3), sh_dc=torch.zeros(4, 1, 3), sh_rest=torch.zeros(4, 3, 3))
        full = FullAttributeUpdate(self.region.config)
        full.setup(reference, self.motion.optimizer, (.01, .01, .01))
        opacity = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        opacity.setup(torch.zeros(4, 1))
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


if __name__ == '__main__':
    unittest.main()
