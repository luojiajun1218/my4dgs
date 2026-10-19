"""Training gradients remain per-view; evaluation reuses zero scratch storage."""
from types import SimpleNamespace
import unittest

import torch

from pipeline.model.contribution_renderer import ContributionRenderer
from pipeline.module import materialize_motion_evidence_counts


class RendererScratchTests(unittest.TestCase):
    def renderer(self):
        return SimpleNamespace(gaussians=SimpleNamespace(get_xyz=torch.ones(4, 3)))

    def test_training_views_have_distinct_leaf_gradients(self):
        renderer = self.renderer()
        first = ContributionRenderer.create_screenspace_points(renderer)
        second = ContributionRenderer.create_screenspace_points(renderer)
        self.assertTrue(first.is_leaf and second.is_leaf)
        self.assertIsNot(first, second)
        (first.sum() * 2 + second.sum() * 3).backward()
        torch.testing.assert_close(first.grad, torch.full_like(first, 2))
        torch.testing.assert_close(second.grad, torch.full_like(second, 3))

    def test_inference_cache_invalidates_on_shape_or_dtype_and_never_enters_training(self):
        renderer = self.renderer()
        with torch.no_grad():
            first = ContributionRenderer.create_screenspace_points(renderer)
            self.assertIs(first, ContributionRenderer.create_screenspace_points(renderer))
            renderer.gaussians.get_xyz = torch.ones(5, 3)
            changed = ContributionRenderer.create_screenspace_points(renderer)
            self.assertIsNot(first, changed)
            renderer.gaussians.get_xyz = torch.ones(5, 3, dtype=torch.float64)
            self.assertIsNot(changed, ContributionRenderer.create_screenspace_points(renderer))
        training = ContributionRenderer.create_screenspace_points(renderer)
        self.assertTrue(training.requires_grad)
        self.assertIsNot(training, renderer._screenspace_zero)

    def test_diagnostic_counts_serialize_as_integers_after_accumulating_views(self):
        control = dict(motion_frustum_samples=3, motion_observed_samples=1)
        for counts in (torch.tensor([5, 2]), torch.tensor([7, 4])):
            for key, value in zip(control, counts):
                control[key] = control[key] + value
        self.assertTrue(all(torch.is_tensor(value) for value in control.values()))
        materialize_motion_evidence_counts(control)
        self.assertEqual(control, dict(motion_frustum_samples=15, motion_observed_samples=7))
        self.assertTrue(all(type(value) is int for value in control.values()))
        materialize_motion_evidence_counts(control)


if __name__ == '__main__':
    unittest.main()
