"""Multi-view motion gradients, evidence, birth scores and resume ordering."""
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

TRAINER = importlib.import_module('pipeline.ReCon-GS.trainer')
MODULE = importlib.import_module('pipeline.ReCon-GS.module')


class MotionViewUpdates(unittest.TestCase):
    def trainer(self, views=4):
        cfg = SimpleNamespace(birth_steps=100,
            spatial_gate=SimpleNamespace(gate_start_step=60, freeze_step=90))
        trainer = SimpleNamespace(config=SimpleNamespace(motion_views_per_step=views),
            module=SimpleNamespace(config=SimpleNamespace(rigid_tree=cfg),
                                   motion_steps=100, total_steps=200))
        trainer._views_per_step = lambda step: TRAINER.Trainer._views_per_step(trainer, step)
        return trainer

    def test_one_update_averages_all_views_and_charges_regularization_once(self):
        parameter = torch.tensor(.5, requires_grad=True)
        trainer = self.trainer()
        optimizer = torch.optim.SGD([parameter], lr=.1)
        received = []
        targets = [-3., 1., 2., 4.]
        fetcher = SimpleNamespace(next=iter(targets).__next__)
        def train(target, step):
            loss = (parameter - target).square() + .3 * parameter.square()
            return loss, dict(image_loss=float(loss.detach()), psnr=1., gs=2), {
                'render': torch.zeros(5), 'visibility_filter': torch.tensor([True, False]),
                'motion_contribution_filter': torch.tensor([False, True]),
                'viewspace_points': torch.zeros(2, 3)}
        def post(packages):
            received.append((float(parameter.grad), packages))
            optimizer.step()
        trainer.module.training_step = train
        trainer.module.post_training_step = post
        loss, metrics = TRAINER.Trainer._fit_views(trainer, fetcher, 1)
        expected = sum((.5-t)**2 + .3*.5**2 for t in targets)/4
        self.assertAlmostEqual(loss, expected, places=5)
        self.assertAlmostEqual(received[0][0], -.7, places=5)
        self.assertAlmostEqual(float(parameter), .57, places=5)
        self.assertEqual(len(received), 1)
        self.assertEqual(len(received[0][1]), 4)
        self.assertNotIn('render', received[0][1][0])
        self.assertEqual(metrics['gs'], 2)

    def test_schedule_and_resume_replay_exact_view_consumption(self):
        trainer = self.trainer()
        for step, wanted in [(1,4),(60,4),(61,4),(90,4),(91,4),(100,4),
                             (101,1),(200,1)]:
            self.assertEqual(trainer._views_per_step(step), wanted)
        self.assertEqual(sum(trainer._views_per_step(s) for s in range(1,201)), 500)
        trainer.global_step = 95
        trainer._resume_rng = {'marker': True}
        trainer._restore_rng = Mock()
        fetcher = SimpleNamespace(next=Mock())
        TRAINER.Trainer._resume_fetcher(trainer, fetcher)
        self.assertEqual(fetcher.next.call_count, 380)
        trainer._restore_rng.assert_called_once_with({'marker': True})
        self.assertIsNone(trainer._resume_rng)
        old = self.trainer(views=1)
        self.assertEqual(sum(old._views_per_step(s) for s in range(1,201)), 200)

    def test_all_camera_evidence_and_unscaled_birth_scores_reach_one_update(self):
        captured = []
        model = SimpleNamespace(add_densification_stats=lambda point, mask:
                                captured.append(point.grad.clone()))
        motion = SimpleNamespace(mark_observed=Mock(), optimizer_step=Mock(return_value=[]))
        module = SimpleNamespace(trainer=SimpleNamespace(global_step=99),
            motion_committed=False, motion_steps=100,
            config=SimpleNamespace(effective_motion_observations=True,
                birth=SimpleNamespace(densification_stats_steps=30)),
            gaussians=model, motion=motion, birth_control={})
        packages = []
        for i in range(4):
            points = torch.zeros(2,3,requires_grad=True)
            points.grad = torch.full_like(points, .25)
            packages.append(dict(viewspace_points=points,
                visibility_filter=torch.tensor([True,True]),
                motion_contribution_filter=torch.tensor([i%2==0,i%2==1])))
        MODULE.Module.post_training_step(module, packages)
        self.assertEqual(motion.mark_observed.call_count, 4)
        motion.optimizer_step.assert_called_once()
        self.assertEqual(module.birth_control['motion_observed_samples'], 4)
        self.assertEqual(module.birth_control['motion_frustum_samples'], 8)
        self.assertEqual(len(captured), 4)
        for value in captured:
            torch.testing.assert_close(value, torch.ones_like(value))


if __name__ == '__main__':
    unittest.main()
