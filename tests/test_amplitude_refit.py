"""Photo-only refitting may change amplitudes, never grow transmitted support."""
import copy
import unittest

import torch

from pipeline.model.spatial_gated_motion import SpatialGatedMotion
import test_joint_spatial_motion as joint_tests


class AmplitudeRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_tests.JointSpatialMotionTests().model()
        model.config.amplitude_refit_steps = 20
        model.config.proximal_gs_regularization = True
        model._hard_gate_frozen = True
        with torch.no_grad():
            model.region_translation[model.owner[0], 0] = 3 * model.steps[0]
            model.gs_translation[0, 0] = 2 * model.gs_steps[0]
            model.gs_translation[1, 0] = .2 * model.gs_steps[0]
        return model

    def test_refit_preserves_forward_then_updates_selected_blocks_from_images(self):
        model = self.model()
        before = model(80)[0].detach().clone()
        xyz, _ = model(81, training=True)
        torch.testing.assert_close(xyz, before, rtol=0, atol=0)
        self.assertTrue(model.amplitude_refit_active)
        self.assertEqual(float(model.regularization_loss(81)), 0.)
        support = model._amplitude_refit_support
        region_before = model.region_translation.detach().clone()
        gs_before = model.gs_translation.detach().clone()
        gates = model.opened.clone()
        ((xyz - model.base_xyz - .03).square().mean()).backward()
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)
        model.optimizer_step()
        self.assertTrue(torch.equal(model.opened, gates))
        self.assertGreater(float((model.region_translation - region_before).abs().sum()), 0.)
        self.assertGreater(float((model.gs_translation - gs_before).abs().sum()), 0.)
        self.assertFalse(bool(model.gs_translation[~support['gs_translation']].ne(0).any()))
        torch.testing.assert_close(model.region_translation[~support['region_translation']],
            region_before[~support['region_translation']], rtol=0, atol=0)

    def test_refit_masks_adam_momentum_and_survives_resume_and_retirement(self):
        model = self.model()
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
            exp_avg_sq=torch.ones_like(p))
        model(81, training=True)
        self.assertFalse(bool(model.optimizer.state[p]['exp_avg'][1:].ne(0).any()))
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        for current in (model, restored):
            xyz, _ = current(82, training=True)
            ((xyz - current.base_xyz - .03).square().mean()).backward()
            current.optimizer_step()
            current.retain_survivors(torch.arange(len(current.owner))[::2])
            self.assertEqual(len(current._amplitude_refit_support['gs_translation']), len(current.owner))
        torch.testing.assert_close(model.gs_translation, restored.gs_translation, rtol=0, atol=0)
        torch.testing.assert_close(model.region_translation, restored.region_translation, rtol=0, atol=0)

    def test_refit_requires_frozen_gates_and_valid_schedule(self):
        model = self.model()
        model._hard_gate_frozen = False
        with self.assertRaises(ValueError):
            model(81, training=True)
        for count in (True, -1, 31, 100):
            model.config.amplitude_refit_steps = count
            with self.assertRaises(ValueError):
                model.config.validate()


if __name__ == '__main__':
    unittest.main()
