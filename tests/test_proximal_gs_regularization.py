"""GS sparse updates must shrink values without contaminating data moments."""
import copy
import unittest

import torch

from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_residual_tree import _quantized
import test_joint_spatial_motion as joint_tests


class ProximalGSRegularizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_tests.JointSpatialMotionTests().model()
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        return model

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
        restored=SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(captured)
        self.assertTrue(restored.proximal_gs_regularization)
        for a,b in [(model.gs_translation,restored.gs_translation),(model.gs_rotation,restored.gs_rotation)]:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for current in [model,restored]:
            current.last_step=2
            current.gs_rotation.grad=torch.full_like(current.gs_rotation,2e-6)
            current.optimizer_step()
        torch.testing.assert_close(model.gs_rotation,restored.gs_rotation,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
