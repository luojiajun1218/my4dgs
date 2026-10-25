"""Numerical regressions for the fixed 20% birth-retirement band."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.model.birth_retirement import (
    birth_retirement_rows, compositing_contributions,
)


class BirthRetirementTests(unittest.TestCase):
    def test_natural_low_alpha_count_is_clamped_to_twenty_percent_band(self):
        # K=100 gives [80, 120]; sources and the previous-frame tail do not
        # contribute to D0 and cannot be retired.
        for natural, expected in ((0, 80), (97, 97), (100, 100), (150, 120)):
            with self.subTest(natural=natural):
                alpha = torch.full((500,), .5)
                alpha[100:100 + natural] = .001
                source_rows = torch.arange(100)
                current = torch.arange(500, dtype=torch.float32)
                previous = torch.flip(current, dims=(0,))
                survivors, info = birth_retirement_rows(alpha, source_rows,
                    current, previous, previous_birth_count=20,
                    count_tolerance=.2, opacity_threshold=.01)
                self.assertEqual(500 - len(survivors), expected)
                self.assertEqual(info['natural_retirements'], natural)
                self.assertEqual((info['min_retirements'], info['max_retirements']), (80, 120))
                self.assertTrue(set(source_rows.tolist()).issubset(survivors.tolist()))
                self.assertTrue(set(range(480, 500)).issubset(survivors.tolist()))

    def test_deletion_orders_by_maximum_actual_current_and_previous_contribution(self):
        alpha = torch.tensor([.5] * 12)
        # K=5 -> band [4, 6], natural count=0 -> exactly four weakest rows.
        current = torch.tensor([.8, .3, .2, .7, .6, .5, .4, .1, .9, .2, .1, .1])
        previous = torch.tensor([.1, .4, .3, .1, .2, .6, .5, .2, .1, .1, .2, .8])
        survivors, info = birth_retirement_rows(alpha, torch.tensor([0, 1, 2, 3, 4]),
            current, previous, count_tolerance=.2)
        self.assertEqual(info['retired_count'], 4)
        # Source rows 0..4 are protected. Among eligible rows, max contribution
        # ranks rows 7, 9, 10, 6 first (stable order resolves ties).
        self.assertEqual(survivors.tolist(), [0, 1, 2, 3, 4, 5, 8, 11])

    def test_zero_and_small_pools_have_exact_integer_bounds(self):
        alpha = torch.full((20,), .5)
        scores = torch.zeros(20)
        survivors, zero = birth_retirement_rows(alpha, torch.empty(0, dtype=torch.long),
                                                scores, scores)
        self.assertEqual(len(survivors), 20)
        self.assertEqual((zero['retired_count'], zero['min_retirements'],
                          zero['max_retirements']), (0, 0, 0))
        for count in (1, 4, 5, 9, 10):
            _, info = birth_retirement_rows(alpha, torch.zeros(count, dtype=torch.long),
                scores, scores, count_tolerance=.2)
            self.assertEqual(info['retired_count'], (int(.8 * count + .999999)))
            self.assertEqual(info['min_retirements'], (int(.8 * count + .999999)))

    def test_insufficient_unprotected_capacity_fails_without_dropping_births(self):
        source_rows = torch.arange(10)
        with self.assertRaisesRegex(ValueError, 'No births have been dropped'):
            birth_retirement_rows(torch.full((12,), .5), source_rows,
                torch.zeros(12), torch.zeros(12), count_tolerance=.2)
        self.assertEqual(source_rows.tolist(), list(range(10)))

    def test_color_derivatives_measure_compositing_and_take_max_over_views(self):
        class Renderer:
            gaussians = SimpleNamespace(get_xyz=torch.zeros(3, 3))

            def __call__(self, weights, background, override_color):
                return {'render': (override_color * weights[:, None]).sum(0)[:, None, None]}

        scores = compositing_contributions(Renderer(),
            [torch.tensor([.1, .9, 0.]), torch.tensor([.8, .2, .3])], torch.zeros(3))
        torch.testing.assert_close(scores, torch.tensor([.8, .9, .3]), rtol=0, atol=1e-7)
        self.assertFalse(scores.requires_grad)


if __name__ == '__main__':
    unittest.main()
