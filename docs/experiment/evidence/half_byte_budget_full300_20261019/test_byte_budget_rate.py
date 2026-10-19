"""Byte prices must stay fixed across differently sized initial references."""
import copy
import unittest

import torch

from pipeline.model.spatial_gated_motion import SpatialGatedMotion
import test_joint_spatial_motion as joint_tests


class ByteBudgetRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_tests.JointSpatialMotionTests().model()
        model.config.rate_normalization = 'budget'
        model.config.rate_budget_bytes = 50000
        model.config.rate_loss_weight = 3e-5
        with torch.no_grad():
            model.region_translation[1:, 0] = model.steps[0]
            model.gs_translation[0, 0] = 2 * model.gs_steps[0]
        model.config.validate()
        return model

    def test_identical_packet_has_identical_loss_at_different_reference_counts(self):
        model = self.model()
        model.rate_reference_count = 10000
        first = model.loss_terms(30)
        model.rate_reference_count = 1000000
        second = model.loss_terms(30)
        torch.testing.assert_close(first['rate_loss'], second['rate_loss'], rtol=0, atol=0)
        self.assertAlmostEqual(float(second['rate_byte_price']), 6e-10, places=16)
        self.assertEqual(float(second['rate_normalization_bytes']), 50000.)
        # Legacy reference normalization remains explicitly density dependent.
        model.config.rate_normalization = 'reference'
        model.config.rate_budget_bytes = None
        before = model.loss_terms(30)['rate_loss']
        model.rate_reference_count //= 10
        after = model.loss_terms(30)['rate_loss']
        torch.testing.assert_close(after, 10 * before)

    def test_budget_and_gradients_survive_checkpoint_restore(self):
        model = self.model()
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        self.assertEqual(restored.rate_normalization_bytes, 50000)
        terms = restored.loss_terms(30)
        terms['rate_loss'].backward()
        self.assertIsNotNone(restored.gate_logits.grad)
        self.assertGreater(float(restored.gate_logits.grad.abs().sum()), 0.)
        torch.testing.assert_close(terms['rate_loss'], model.loss_terms(30)['rate_loss'])

    def test_budget_is_explicit_and_validated(self):
        model = self.model()
        for invalid in (None, 0, -1, True, 1.5, float('nan')):
            with self.subTest(invalid=invalid):
                model.config.rate_budget_bytes = invalid
                with self.assertRaisesRegex(ValueError, 'rate_budget_bytes'):
                    model.config.validate()
        model.config.rate_budget_bytes = 50000
        model.config.rate_normalization = 'reference'
        with self.assertRaisesRegex(ValueError, 'budget normalization'):
            model.config.validate()


if __name__ == '__main__':
    unittest.main()
