"""Sample discrete experts, learn probabilities, and restore private gate RNG."""
import copy
import unittest

import torch

import test_private_motion_proposals as fixtures
import test_joint_image_rate as codec_fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion


class StochasticGateFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        template = fixtures.PrivateMotionProposalTests().model()
        cfg = copy.deepcopy(template.config)
        cfg.spatial_gate.image_relaxation = 'stochastic_ste'
        cfg.spatial_gate.temperature = 3.
        cfg.spatial_gate.final_temperature = .3
        cfg.validate()
        model = SpatialGatedMotion(cfg)
        model.setup(template.base_xyz, template.base_rotation, template.extent, frame_index=1)
        model.observations.fill_(1)
        return model

    def test_render_uses_one_actual_expert_and_reuses_the_sample_for_two_views(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)).remainder(7) * .002
            model.gs_translation[:, 0] = .054
        global_rng = torch.get_rng_state().clone()
        xyz, _ = model(1, training=True)
        noise = model._sampled_gate_noise.clone()
        rng = model._gate_rng.get_state().clone()
        second, _ = model(1, training=True)
        torch.testing.assert_close(xyz, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, model._gate_rng.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        regions = model._regional_predictions()[0]
        proposal = torch.round(model.gs_translation / model.gs_steps[:3]) * model.gs_steps[:3]
        choices = torch.cat((regions[model.spatial_paths[:, :-1]], proposal[:, None]), 1)
        choices = model.base_xyz[:, None] + choices
        distance = (xyz[:, None] - choices).abs().amax(-1).amin(-1)
        self.assertEqual(float(distance.max()), 0.)
        model(2, training=True)
        self.assertFalse(torch.equal(noise, model._sampled_gate_noise))

    def test_sampled_expert_gets_its_own_error_and_unselected_experts_have_no_image_gradient(self):
        model = self.model()
        model(1, training=True)
        draw = model.gate_logits / model._gate_temperature() + model._sampled_gate_noise
        weights = model._path_weights((draw.ge(0) & model.split_eligible).to(model.gate_logits))
        row, level = weights[:, :-1].nonzero()[0].tolist()
        node = int(model.spatial_paths[row, level])
        members = model.spatial_paths[:, level].eq(node)
        target = model.base_xyz.clone()
        target[members, 0] += .02
        gradients = []
        for displacement in (.01, .2):
            with torch.no_grad():
                model.region_translation.zero_()
                model.region_translation[node, 0] = displacement
            xyz, _ = model(1, training=True)
            gradient = torch.autograd.grad((xyz - target).square().mean(), model.region_translation)[0]
            gradients.append(float(gradient[node, 0]))
            selected = torch.zeros(len(model.parents), dtype=torch.bool)
            for layer in range(model.spatial_paths.shape[1] - 1):
                selected[model.spatial_paths[weights[:, layer].bool(), layer]] = True
            self.assertEqual(float(gradient[~selected].abs().sum()), 0.)
        self.assertLess(gradients[0], 0.)
        self.assertGreater(gradients[1], 0.)

    def test_private_gradients_use_sampled_capacity_and_gate_probabilities_are_learnable(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = .01
        xyz, _ = model(1, training=True)
        target = model.base_xyz.clone()
        target[:, 0] += .02
        point, gate = torch.autograd.grad((xyz - target).square().mean(),
            [model.gs_translation, model.gate_logits])
        draw = model.gate_logits / model._gate_temperature() + model._sampled_gate_noise
        allowed = (draw.ge(0) & model.split_eligible)[model.spatial_paths[:, :-1]].all(-1)
        self.assertTrue(torch.equal(point.ne(0).any(-1), allowed))
        self.assertGreater(float(gate.abs().sum()), 0.)

        with torch.no_grad():
            model.gs_translation[:, 0] = .02
        optimizer = torch.optim.Adam([model.gate_logits], lr=.3)
        initial = float((model(1)[0] - target).square().mean())
        for step in range(1, 61):
            optimizer.zero_grad()
            xyz, _ = model(step, training=True)
            (xyz - target).square().mean().backward()
            optimizer.step()
            model._select_frontier(freeze=False)
        final = float((model(60)[0] - target).square().mean())
        self.assertLess(final, initial * .1)

    def test_sampling_never_changes_the_deterministic_stream_and_final_fit_is_exact(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[9:, 0] = .02
            model.gs_translation[:, 1] = .03
        packet = codec_fixtures.JointImageRateTests.packet(model)
        owner = model.owner.clone()
        for step in (1, 2, 3):
            model(step, training=True)
        self.assertTrue(torch.equal(owner, model.owner))
        self.assertEqual(packet, codec_fixtures.JointImageRateTests.packet(model))
        state = decode_motion(packet, model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(3)[0], rtol=0, atol=0)
        model._select_frontier(freeze=True)
        state = decode_motion(codec_fixtures.JointImageRateTests.packet(model), model.frame_reference_state)
        torch.testing.assert_close(state.xyz, model(71, training=True)[0], rtol=0, atol=0)

    def test_resume_preserves_current_and_future_gate_samples_and_updates(self):
        for step in (20, 70):
            model = self.model()
            target = model.base_xyz.clone()
            target[:, 0] += .02
            xyz, _ = model(step, training=True)
            ((xyz - target).square().mean() + model.regularization_loss(step)).backward()
            model.optimizer_step()
            restored = SpatialGatedMotion(copy.deepcopy(model.config))
            packet = copy.deepcopy(model.capture())
            restored.restore(packet)
            if step == 20:
                torch.testing.assert_close(model(step, training=True)[0],
                                           restored(step, training=True)[0], rtol=0, atol=0)
            for current in (model, restored):
                xyz, _ = current(step + 1, training=True)
                ((xyz - target).square().mean() + current.regularization_loss(step + 1)).backward()
                current.optimizer_step()
            self.assertTrue(torch.equal(model._sampled_gate_noise, restored._sampled_gate_noise))
            self.assertTrue(torch.equal(model._gate_rng.get_state(), restored._gate_rng.get_state()))
            for name in ('region_translation', 'gs_translation', 'gate_logits'):
                torch.testing.assert_close(getattr(model, name), getattr(restored, name), rtol=0, atol=0)
            self.assertEqual(codec_fixtures.JointImageRateTests.packet(model),
                             codec_fixtures.JointImageRateTests.packet(restored))
            broken = copy.deepcopy(packet)
            broken.pop('image_gate_sampler')
            with self.assertRaises(ValueError):
                SpatialGatedMotion(copy.deepcopy(model.config)).restore(broken)

    def test_stochastic_mode_rejects_an_incompatible_private_parameterization(self):
        model = self.model()
        cfg = copy.deepcopy(model.config)
        cfg.spatial_gate.private_full_motion = False
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == '__main__':
    unittest.main()
