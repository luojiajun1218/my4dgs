"""Candidate self-feedback, discrete replay, and annealed-gate restoration."""
import copy
import unittest

import torch

import test_joint_spatial_motion as fixtures
import test_joint_image_rate as codec_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class SoftGateFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, relaxation='soft_anneal', private=False):
        template = fixtures.JointSpatialMotionTests().model(initial_logit=-.2)
        cfg = copy.deepcopy(template.config)
        cfg.rotation_updates = False
        cfg.proximal_gs_regularization = True
        cfg.differentiable_rate_support = True
        gate = cfg.spatial_gate
        gate.joint_image_gradients = True
        gate.hierarchical_private = private
        gate.image_relaxation = relaxation
        gate.temperature = 3.
        gate.final_temperature = .3
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    def test_closed_candidate_gets_feedback_when_its_prediction_overshoots(self):
        gradients = {}
        for relaxation in ('hard_ste', 'soft_anneal'):
            model = self.model(relaxation)
            node = int(model.spatial_paths[0, 1])
            members = model.spatial_paths[:, 1].eq(node)
            target = model.base_xyz.clone()
            target[members, 0] += .02
            gradients[relaxation] = []
            for displacement in (.02, .2):
                with torch.no_grad():
                    model.region_translation.zero_()
                    model.region_translation[node, 0] = displacement
                xyz, _ = model(1, training=True)
                loss = (xyz - target).square().mean()
                grad = torch.autograd.grad(loss, model.region_translation)[0][node, 0]
                gradients[relaxation].append(float(grad))
        self.assertEqual(*gradients['hard_ste'])
        self.assertLess(gradients['soft_anneal'][0], 0.)
        self.assertGreater(gradients['soft_anneal'][1], 0.)

    def test_private_candidate_also_receives_its_own_prediction_error(self):
        model = self.model(private=True)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        gradients = []
        for displacement in (.02, .2):
            with torch.no_grad():
                model.gs_translation[:, 0] = displacement
            xyz, _ = model(1, training=True)
            grad = torch.autograd.grad((xyz - target).square().mean(), model.gs_translation)[0]
            gradients.append(float(grad[:, 0].sum()))
        self.assertLess(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)

    def test_evaluation_and_packet_stay_discrete_while_training_is_relaxed(self):
        for private in (False, True):
            with self.subTest(private=private):
                model = self.model(private=private)
                with torch.no_grad():
                    model.region_translation[9:, 0] = .02
                    model.gs_translation[:, 1] = .02
                soft, _ = model(10, training=True)
                hard, _ = model(10, training=False)
                self.assertGreater(float((soft - hard).abs().max()), 0.)
                state = decode_motion(codec_fixtures.JointImageRateTests.packet(model),
                                      model.frame_reference_state)
                torch.testing.assert_close(state.xyz, hard, rtol=0, atol=0)
                model._select_frontier(freeze=True)
                fitted, _ = model(71, training=True)
                state = decode_motion(codec_fixtures.JointImageRateTests.packet(model),
                                      model.frame_reference_state)
                torch.testing.assert_close(state.xyz, fitted, rtol=0, atol=0)

    def test_temperature_anneals_without_changing_discrete_threshold(self):
        model = self.model()
        temperatures = []
        owners = []
        for step in (1, 20, 40, 70):
            model(step, training=True)
            temperatures.append(model._gate_temperature())
            model._select_frontier(freeze=False)
            owners.append(model.owner.clone())
        self.assertEqual(temperatures[0], 3.)
        self.assertAlmostEqual(temperatures[-1], .3)
        self.assertTrue(all(a > b for a, b in zip(temperatures, temperatures[1:])))
        self.assertTrue(all(torch.equal(owners[0], owner) for owner in owners[1:]))

    def test_resume_preserves_soft_stage_and_fixed_stage_moments(self):
        for step in (20, 70):
            with self.subTest(step=step):
                model = self.model()
                xyz, _ = model(step, training=True)
                loss = (xyz - model.base_xyz - .02).square().mean()
                (loss + model.regularization_loss(step)).backward()
                model.optimizer_step()
                self.assertEqual(model._hard_gate_frozen, step == 70)
                if step == 70:
                    self.assertNotIn(model.region_translation, model.optimizer.state)
                    self.assertNotIn(model.gs_translation, model.optimizer.state)
                packet = copy.deepcopy(model.capture())
                restored = SpatialGatedMotion(copy.deepcopy(model.config))
                restored.restore(packet)
                self.assertEqual(model._gate_temperature(), restored._gate_temperature())
                for current in (model, restored):
                    xyz, _ = current(step + 1, training=True)
                    loss = (xyz - current.base_xyz - .02).square().mean()
                    (loss + current.regularization_loss(step + 1)).backward()
                    current.optimizer_step()
                for name in ('region_translation', 'gs_translation', 'gate_logits'):
                    torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
                incompatible = copy.deepcopy(model.config)
                incompatible.spatial_gate.image_relaxation = 'hard_ste'
                with self.assertRaises(ValueError):
                    SpatialGatedMotion(incompatible).restore(packet)

    def test_relaxation_configuration_rejects_incompatible_modes(self):
        model = self.model()
        for temperature in (0., -1., float('nan'), 4., True):
            cfg = copy.deepcopy(model.config)
            cfg.spatial_gate.final_temperature = temperature
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                cfg.validate()
        cfg = copy.deepcopy(model.config)
        cfg.spatial_gate.joint_image_gradients = False
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == '__main__':
    unittest.main()
