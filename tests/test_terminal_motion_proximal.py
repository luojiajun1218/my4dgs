"""A terminal coding check changes residual support, never the learned gate."""
import copy
import unittest
from unittest.mock import Mock

import torch

from pipeline.model.spatial_gated_motion import SpatialGatedMotion
import test_amplitude_refit as fixtures


class TerminalMotionProximalTests(unittest.TestCase):
    def model(self):
        model = fixtures.AmplitudeRefitTests().model()
        model.config.terminal_motion_proximal = True
        return model

    def test_only_terminal_refit_step_runs_cleanup_and_restore_preserves_switch(self):
        model = self.model()
        model(81, training=True)
        model._proximal_gs_blocks = Mock()
        model._record_gate_learning = Mock()
        for step in (81, 99):
            model.last_step = step
            model._finish_joint_step()
        model._proximal_gs_blocks.assert_not_called()
        model.last_step = 100
        model._finish_joint_step()
        model._proximal_gs_blocks.assert_called_once()
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        self.assertTrue(restored.config.terminal_motion_proximal)
        model.config.terminal_motion_proximal = False
        model._proximal_gs_blocks.reset_mock()
        model._finish_joint_step()
        model._proximal_gs_blocks.assert_not_called()

    def test_weak_block_removed_but_high_image_sensitivity_and_gate_retained(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation.zero_()
            model.gs_translation[:2, 0] = 2 * model.gs_steps[0]
        model(81, training=True)
        variance = torch.full_like(model.gs_translation, 1e-20)
        variance[1] = 1e4
        model.optimizer.state[model.gs_translation] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(variance), exp_avg_sq=variance)
        gates, regions = model.opened.clone(), model.region_translation.detach().clone()
        model._record_gate_learning = Mock()
        model.last_step = 100
        model._finish_joint_step()
        self.assertEqual(float(model.gs_translation[0].abs().sum()), 0.)
        self.assertGreater(float(model.gs_translation[1].abs().sum()), 0.)
        torch.testing.assert_close(gates, model.opened, rtol=0, atol=0)
        torch.testing.assert_close(regions, model.region_translation, rtol=0, atol=0)
        self.assertEqual(model.last_metrics['motion/terminal_removed_translation_blocks'], 1)

    def test_configuration_rejects_wrong_types_and_unmatched_schedule(self):
        model = self.model()
        model.config.validate()
        for value in (1, None, 'true'):
            model.config.terminal_motion_proximal = value
            with self.assertRaisesRegex(ValueError, 'terminal_motion_proximal'):
                model.config.validate()
        model.config.terminal_motion_proximal = True
        model.config.amplitude_refit_steps = 0
        with self.assertRaisesRegex(ValueError, 'Terminal motion proximal'):
            model.config.validate()


if __name__ == '__main__':
    unittest.main()
