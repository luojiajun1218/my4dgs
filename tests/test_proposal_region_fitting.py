"""Unselected shared candidates fit detached, image-trained motion proposals."""
import copy
from types import SimpleNamespace
import unittest

import torch

import test_proposal_image_supervision as fixtures
from pipeline.model.spatial_gated_motion import SpatialGatedMotion


class ProposalRegionFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, weight=.1):
        model = fixtures.ProposalImageSupervisionTests().model()
        model.config.spatial_gate.proposal_region_fit_weight = weight
        model.config.validate()
        return model

    def test_closed_middle_candidates_receive_target_gradients_independent_of_gate_and_proposal(self):
        motion = self.model()
        with torch.no_grad():
            motion.gs_translation[:, 0] = .02
        motion._sampled_gate_step = 1
        motion._sampled_gate_noise = torch.full_like(motion.gate_logits, -30.)
        xyz, _ = motion(1, training=True)
        target = motion.base_xyz.clone()
        target[:, 0] += .02
        image_gradient = torch.autograd.grad((xyz - target).square().mean(),
                                             motion.region_translation)[0]
        middle = motion.spatial_paths[:, 1:-1].unique()
        self.assertGreater(len(middle), 0)
        self.assertEqual(float(image_gradient[middle].abs().sum()), 0.)
        gradients = torch.autograd.grad(motion.proposal_region_fit_loss(),
            [motion.region_translation, motion.gs_translation, motion.gate_logits], allow_unused=True)
        self.assertTrue(bool(gradients[0][motion.spatial_paths.unique(), 0].lt(0).all()))
        self.assertIsNone(gradients[1])
        self.assertIsNone(gradients[2])

    def test_large_background_regions_do_not_change_other_regions_gradient_weight(self):
        def fit(repeats):
            paths = torch.tensor([[1, 3]] * repeats + [[2, 4]])
            regions = torch.zeros(5, 3, requires_grad=True)
            proposal = torch.tensor([[.02, 0., 0.]] * len(paths), requires_grad=True)
            model = SimpleNamespace(proposal_image_weight=1.,
                config=SimpleNamespace(spatial_gate=SimpleNamespace(proposal_region_fit_weight=.1),
                                       rotation_updates=False),
                base_xyz=torch.zeros_like(proposal), gs_translation=proposal,
                gs_rotation=torch.zeros_like(proposal), gs_steps=torch.full((6,), .001),
                observations=torch.ones(len(paths)), spatial_paths=paths,
                parents=torch.tensor([-1, 0, 0, 1, 2]), extent=1.,
                _regional_predictions=lambda: (regions, torch.zeros_like(regions)))
            loss = SpatialGatedMotion.proposal_region_fit_loss(model)
            return loss, torch.autograd.grad(loss, regions)[0]
        first, first_grad = fit(1)
        repeated, repeated_grad = fit(100)
        torch.testing.assert_close(first, repeated, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(first_grad, repeated_grad, rtol=1e-6, atol=1e-8)
        self.assertEqual(float(first_grad[1, 0]), float(first_grad[2, 0]))

    def test_unobserved_outlier_proposals_do_not_pull_shared_candidates(self):
        motion = self.model()
        with torch.no_grad():
            motion.gs_translation.zero_()
            motion.gs_translation[0] = .2
            motion.observations[0] = 0
        loss = motion.proposal_region_fit_loss()
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(motion.region_translation.grad.abs().sum()), 0.)
        self.assertIsNone(motion.gs_translation.grad)
        motion.observations[0] = 1
        self.assertGreater(float(motion.proposal_region_fit_loss()), 0.)

    def test_training_objective_includes_the_term_only_before_gate_freeze(self):
        motion = self.model()
        with torch.no_grad():
            motion.gs_translation[:, 0] = .02
        terms = motion.loss_terms(1, diagnostics=False)
        expected = terms['rate_loss'] + terms['relative_motion_loss'] + terms['candidate_motion_loss']
        expected += .1 * motion.proposal_region_fit_loss()
        torch.testing.assert_close(motion.regularization_loss(1), expected, rtol=0, atol=0)
        motion._select_frontier(freeze=True)
        loss = motion.proposal_region_fit_loss()
        self.assertEqual(float(loss), 0.)
        self.assertFalse(loss.requires_grad)
        self.assertEqual(float(motion.loss_terms(71, diagnostics=False)['proposal_region_fit_loss']), 0.)

    def test_checkpoint_restore_and_zero_default_compatibility(self):
        for weight in (0., .1):
            motion = self.model(weight)
            with torch.no_grad():
                motion.gs_translation[:, 0] = .02
            packet = copy.deepcopy(motion.capture())
            if weight == 0:
                packet['config']['spatial_gate'].pop('proposal_region_fit_weight')
            restored = SpatialGatedMotion(copy.deepcopy(motion.config))
            restored.restore(packet)
            torch.testing.assert_close(motion.proposal_region_fit_loss(),
                                       restored.proposal_region_fit_loss(), rtol=0, atol=0)
            if not weight:
                self.assertNotIn('proposal_region_fit_loss', restored.loss_terms(1, diagnostics=False))

    def test_invalid_weights_or_untrained_targets_are_rejected(self):
        template = self.model().config
        for weight in (True, -.1, float('inf'), float('nan')):
            cfg = copy.deepcopy(template)
            cfg.spatial_gate.proposal_region_fit_weight = weight
            with self.assertRaises(ValueError):
                cfg.validate()
        cfg = copy.deepcopy(template)
        cfg.spatial_gate.proposal_image_weight = 0.
        with self.assertRaises(ValueError):
            cfg.validate()


if __name__ == '__main__':
    unittest.main()
