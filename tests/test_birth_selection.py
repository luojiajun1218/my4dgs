"""Newborn utility estimates, subset lineage and structural image fitting."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.model.birth_selection import (birth_image_loss, midpoint_image_utility,
                                           useful_birth_subsets)
from pipeline.module import BirthConfig


class BirthSelectionTests(unittest.TestCase):
    def test_midpoint_retains_an_optimal_splat_and_rejects_a_ghost(self):
        alpha = torch.full((3, 1), .4)
        logits = torch.nn.Parameter(torch.logit(alpha*.5))
        camera = SimpleNamespace(image=torch.tensor([.4, 0., .2]).expand(3, 1, 3))

        def renderer(_, background):
            return {'render': logits.sigmoid().T.expand(3, 1, 3)}

        with torch.no_grad():
            score = midpoint_image_utility(renderer, [camera], torch.zeros(3), logits, alpha, [1.])
        # At full alpha the first point's gradient is already zero; its finite
        # removal still hurts. The midpoint recovers that useful contribution.
        torch.testing.assert_close(score, torch.tensor([.16/3, -.16/3, 0.]), atol=1e-7, rtol=0)
        self.assertIsNone(logits.grad)

    def test_smaller_pools_remove_harmful_points_and_keep_original_row_order(self):
        score = torch.tensor([-2., .1, .8, -1., .9, .3, .2, 0.])
        pools = useful_birth_subsets(score, [1., .5, .25])
        self.assertEqual([rows.tolist() for rows in pools],
                         [list(range(8)), [1, 2, 4, 5, 6], [2, 4, 5], [2, 4]])
        self.assertEqual([len(rows) for rows in useful_birth_subsets(torch.ones(2), [1., .5, .25])], [2, 1])
        self.assertEqual(len(useful_birth_subsets(torch.zeros(3), [1., .5])), 1)

    def test_structural_objective_has_zero_loss_at_target_and_finite_gradients(self):
        torch.manual_seed(4)
        target = torch.rand(3, 16, 16)
        image = (target*.8+.1).requires_grad_(True)
        mse = birth_image_loss(image, target)
        structural = birth_image_loss(image, target, .01)
        self.assertGreater(float(structural), float(mse))
        gradient, = torch.autograd.grad(structural, image)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertAlmostEqual(float(birth_image_loss(target, target, .01)), 0., places=7)

    def test_invalid_scores_and_settings_fail_before_model_pruning(self):
        with self.assertRaises(ValueError):
            useful_birth_subsets(torch.tensor([float('nan')]), [1.])
        for values in (dict(fit_loss='mse_ssim'), dict(fit_ssim_weight=-.1),
                       dict(admission_useful_fractions=[1., 1.]),
                       dict(admission_useful_fractions=[.5]),
                       dict(refit_all_training_views=1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BirthConfig(**values).validate(180)


if __name__ == '__main__':
    unittest.main()
