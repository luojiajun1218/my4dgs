import tempfile
from pathlib import Path
import unittest
import torch
from pipeline.model.regional_logscale import RegionalLogScaleConfig, RegionalLogScaleUpdate, decode_regional_logscale
from pipeline.model.full_attribute_update import FullUpdateConfig, FullAttributeUpdate

class TestRegionalLogScale(unittest.TestCase):
    def test_default_disabled_config_validates(self):
        config = RegionalLogScaleConfig()
        self.assertFalse(config.enabled)
        config.validate()

    def test_nonzero_two_frame_symbols_and_reference_binding(self):
        reference = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        owner = torch.tensor([1, 1, 2])
        for frame in (1, 2):
            opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
            shared = RegionalLogScaleUpdate(RegionalLogScaleConfig(enabled=True), None)
            shared.setup(reference)
            shared.bind(owner, 3, torch.tensor([1, 2]), opt)
            with torch.no_grad():
                shared.delta[1] = torch.tensor([.01, -.005, 0.])
                shared.delta[2] = torch.tensor([-.015, 0., .005])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'scale.bin'
                self.assertIsNotNone(shared.export(path, frame))
                result = decode_regional_logscale(path.read_bytes(), reference, frame_index=frame, owner=owner, node_count=3)
                self.assertTrue(torch.equal(result, shared.canonical_features()))
                with self.assertRaises(ValueError):
                    decode_regional_logscale(path.read_bytes(), reference, frame_index=frame, owner=torch.tensor([1, 2, 2]), node_count=3)
                reference = result
    def test_point_symbol_constraint_and_gradient(self):
        ref = dict(logscale=torch.zeros(2,3), sh_dc=torch.zeros(2,1,3), sh_rest=torch.zeros(2,0,3))
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        full = FullAttributeUpdate(FullUpdateConfig())
        full.setup(ref, opt, (.01,.01,.01))
        full.regional_scale_symbols = lambda: torch.tensor([[1.,-1.,0.],[-1.,1.,0.]])
        with torch.no_grad(): full.parameters['logscale'].copy_(torch.tensor([[-.01,.01,-.01],[.01,-.01,.01]]))
        expected = torch.tensor([[0.,0.,-.01],[0.,0.,.01]])
        actual = full.apply(ref)['logscale']
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(full.canonical(ref)['logscale'], expected))
        actual.sum().backward()
        self.assertTrue(torch.equal(full.parameters['logscale'].grad, torch.ones(2,3)))
    def test_checkpoint_restore_distinct_optimizer_group(self):
        shared = RegionalLogScaleUpdate(RegionalLogScaleConfig(enabled=True), None)
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        shared.setup(torch.zeros(2,3)); shared.bind(torch.tensor([0,1]),2,torch.tensor([0,1]),opt)
        packet=shared.capture()
        other=RegionalLogScaleUpdate(RegionalLogScaleConfig(enabled=True), None)
        opt2=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        other.restore(packet,'cpu',opt2)
        self.assertEqual(other._group()['name'],'early_regional_logscale')
        self.assertEqual(other.forward().shape,(2,3))

class TestRegionalScaleResume(unittest.TestCase):
    def test_only_verified_finalized_exact_default_enable(self):
        from dataclasses import asdict
        from pipeline.module import regional_scale_resume_allowed
        saved = {'regional_logscale': asdict(RegionalLogScaleConfig()), 'other': {'rate': 1}}
        requested = {**saved, 'regional_logscale': {**saved['regional_logscale'], 'enabled': True}}
        state = {'prepared': True, 'finalized': True, 'motion_committed': True}
        self.assertTrue(regional_scale_resume_allowed(saved, requested, state, object()))
        self.assertFalse(regional_scale_resume_allowed(saved, requested, state, None))
        self.assertFalse(regional_scale_resume_allowed(saved, requested, {**state, 'finalized': False}, object()))
        self.assertFalse(regional_scale_resume_allowed(saved, {**requested, 'other': {'rate': 2}}, state, object()))
        self.assertFalse(regional_scale_resume_allowed(saved, {**requested, 'regional_logscale': {**requested['regional_logscale'], 'step': .01}}, state, object()))
