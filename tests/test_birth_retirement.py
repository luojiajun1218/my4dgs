"""Contribution history, balanced trial counts and safe replacement admission."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.model.birth_retirement import (
    retirement_count_trials, rank_retirement_candidates,
    choose_validation_trial, compositing_contributions,
)


class ContributionRetirementTests(unittest.TestCase):
    def test_history_protects_previously_visible_points_and_recent_births(self):
        current = torch.tensor([0., .4, .1, .2, 0., 0.])
        previous = torch.tensor([.9, 0., .1, .2, 0., 0.])
        ranked, protected = rank_retirement_candidates(
            current, previous, torch.tensor([1]), previous_birth_count=2)
        self.assertEqual(ranked.tolist(), [2, 3, 0])
        self.assertEqual(protected.tolist(), [False, True, False, False, True, True])
        # With no grace period the invisible suffix is eligible again.
        ranked, _ = rank_retirement_candidates(current, previous, torch.tensor([1]))
        self.assertEqual(ranked[:2].tolist(), [4, 5])

    def test_integer_trial_band_allows_both_more_and_fewer_retirements(self):
        for births, expected in ((0, [0]), (1, [1]), (9, [9]),
                                 (10, [9, 10, 11]), (947, [853, 947, 1041])):
            self.assertEqual(retirement_count_trials(births, .1), expected)

    def test_bad_ssim_maximum_cannot_hide_a_safe_psnr_improvement(self):
        baseline = dict(psnr=34., ssim=.97)
        trials = [dict(psnr=34.1, ssim=.97001),
                  dict(psnr=34.2, ssim=.96),
                  dict(psnr=34.08, ssim=.97001)]
        self.assertEqual(choose_validation_trial(baseline, trials, .005, 1e-4), 0)
        self.assertIsNone(choose_validation_trial(baseline, trials, .3, 1e-4))
        self.assertIsNone(choose_validation_trial(baseline, [], .005, 1e-4))

    def test_color_derivatives_measure_compositing_and_take_max_over_views(self):
        class Renderer:
            gaussians = SimpleNamespace(get_xyz=torch.zeros(3, 3))

            def __call__(self, weights, background, override_color):
                return {'render': (override_color * weights[:, None]).sum(0)[:, None, None]}

        with torch.no_grad():
            scores = compositing_contributions(Renderer(),
                [torch.tensor([.1, .9, 0.]), torch.tensor([.8, .2, .3])], torch.zeros(3))
        torch.testing.assert_close(scores, torch.tensor([.8, .9, .3]), rtol=0, atol=1e-7)
        self.assertFalse(scores.requires_grad)

    def test_misaligned_or_invalid_evidence_cannot_drive_deletion(self):
        for current, previous, sources, recent in (
                (torch.ones(2), torch.ones(1), [], 0),
                (torch.tensor([float('nan')]), torch.ones(1), [], 0),
                (torch.tensor([-.1]), torch.ones(1), [], 0),
                (torch.ones(2), torch.ones(2), [2], 0),
                (torch.ones(2), torch.ones(2), [], 3)):
            with self.subTest(sources=sources, recent=recent), self.assertRaises(ValueError):
                rank_retirement_candidates(current, previous, sources, recent)


if __name__ == '__main__':
    unittest.main()
