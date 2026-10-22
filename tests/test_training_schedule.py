"""Multi-view optimization, photometric gradients, birth evidence, and deterministic view consumption."""
import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch
from torch.nn.functional import l1_loss
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


TRAINER = importlib.import_module('pipeline.trainer')


MODULE = importlib.import_module('pipeline.module')


class MotionViewUpdates(unittest.TestCase):
    def trainer(self, views=4):
        cfg = SimpleNamespace(birth_steps=100,
            spatial_gate=SimpleNamespace(gate_start_step=60, freeze_step=90))
        trainer = SimpleNamespace(config=SimpleNamespace(motion_views_per_step=views),
            module=SimpleNamespace(config=SimpleNamespace(rigid_tree=cfg),
                                   motion_steps=100, total_steps=200))
        trainer._views_per_step = lambda step: TRAINER.Trainer._views_per_step(trainer, step)
        return trainer

    def test_replacement_and_refit_have_distinct_boundaries_and_legacy_schedule_is_unchanged(self):
        module = SimpleNamespace(motion_steps=180, birth_selection_end_step=260, total_steps=300)
        for step, expected in ((1, 'motion'), (180, 'motion'), (181, 'birth'),
                               (260, 'birth'), (261, 'refit'), (300, 'refit')):
            with self.subTest(step=step):
                self.assertEqual(TRAINER.frame_training_phase(module, step), expected)
        self.assertEqual(TRAINER.stage_image_steps(module), (180, 260, 300))
        del module.birth_selection_end_step
        self.assertEqual(TRAINER.frame_training_phase(module, 261), 'birth')
        self.assertEqual(TRAINER.stage_image_steps(module), (180, 300))

    def test_progress_counts_points_after_replacement_instead_of_the_candidate_pool(self):
        trainer = self.trainer(views=1)
        trainer.module.gaussians = SimpleNamespace(_xyz=torch.zeros(4, 3))
        trainer.module.births = SimpleNamespace(_xyz=torch.zeros(2, 3))
        trainer.module.motion_committed = True
        trainer.module.training_step = lambda *args, **kwargs: (
            torch.tensor(0.), dict(image_loss=0., psnr=30., gs=6), {})
        def replace(_):
            trainer.module.gaussians._xyz = torch.zeros(3, 3)
            trainer.module.births._xyz = torch.zeros(1, 3)
        trainer.module.post_training_step = replace
        _, metrics = TRAINER.Trainer._fit_views(trainer,
            SimpleNamespace(next=lambda: None), 260)
        self.assertEqual(metrics['gs'], 4)

    def test_stage_diagnostics_preserve_rng_and_distinguish_replacement_from_refit(self):
        view = SimpleNamespace(image=torch.ones(3, 4, 4), path=Path('cam00.png'))
        calls = []
        def render(_, old_only=False):
            calls.append(old_only)
            torch.rand(1)
            return {'render': torch.full_like(view.image, .5)}
        module = SimpleNamespace(motion_steps=180, birth_selection_end_step=260,
            total_steps=300, forward=render, stage_image_metrics=[],
            gaussians=SimpleNamespace(_xyz=torch.zeros(5, 3)),
            births=SimpleNamespace(_xyz=torch.zeros(2, 3)))
        trainer = SimpleNamespace(module=module, current_frame=1)
        trainer._capture_rng = lambda: TRAINER.Trainer._capture_rng(trainer)
        trainer._restore_rng = TRAINER.Trainer._restore_rng
        before = torch.get_rng_state().clone()
        with patch('torch.cuda.is_available', return_value=False):
            for step in (180, 260, 260, 300):
                if step == 300:
                    module.gaussians._xyz = torch.zeros(7, 3)
                    module.births._xyz = torch.zeros(0, 3)
                result = TRAINER.Trainer._record_stage_image(trainer, step, view)
                self.assertIs(result, view)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(calls, [True, False, False, False])
        self.assertEqual([row['step'] for row in module.stage_image_metrics], [180, 260, 300])
        self.assertEqual([row['gaussian_count'] for row in module.stage_image_metrics], [5, 7, 7])
        self.assertEqual(module.stage_image_metrics[1]['phase'],
                         'after_balanced_replacement_before_refit')
        self.assertFalse(any(row['participates_in_training_or_selection']
                             for row in module.stage_image_metrics))

    def test_one_update_averages_all_views_and_charges_regularization_once(self):
        parameter = torch.tensor(.5, requires_grad=True)
        trainer = self.trainer()
        optimizer = torch.optim.SGD([parameter], lr=.1)
        received = []
        targets = [-3., 1., 2., 4.]
        fetcher = SimpleNamespace(next=iter(targets).__next__)
        regularizer_calls = []
        def train(target, step, regularization_weight=1.):
            regularizer_calls.append(regularization_weight)
            loss = (parameter - target).square() + regularization_weight * .3 * parameter.square()
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
        self.assertEqual(regularizer_calls, [0., 0., 0., 4])

    def test_device_metrics_are_returned_as_python_values_after_all_views(self):
        trainer = self.trainer(views=2)
        targets = iter([1., 3.])
        trainer.module.training_step = lambda target, step, **kwargs: (
            torch.tensor(target), dict(image_loss=torch.tensor(target),
                psnr=torch.tensor(target * 10), gs=7), {})
        trainer.module.post_training_step = Mock()
        loss, metrics = TRAINER.Trainer._fit_views(trainer,
            SimpleNamespace(next=targets.__next__), 1)
        self.assertEqual(loss, 2.)
        self.assertEqual(metrics, dict(image_loss=2., psnr=20., gs=7))
        self.assertIsInstance(metrics['image_loss'], float)

    def test_deferred_metrics_preserve_gradients_and_materialized_values(self):
        results = []
        for deferred in (False, True):
            parameter = torch.tensor(.5, requires_grad=True)
            trainer = self.trainer(views=2)
            optimizer = torch.optim.SGD([parameter], lr=.1)
            targets = iter([1., 3., -2., 4., .25, .75])
            def train(target, step, **kwargs):
                loss = (parameter - target).square()
                metrics = dict(image_loss=loss.detach(), psnr=loss.detach() * 2,
                               gs=7, motion_image_factor=1)
                if step == 2:
                    metrics['optional_loss'] = loss.detach() * .1
                return loss, metrics, {}
            def update(_):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            trainer.module.training_step = train
            trainer.module.post_training_step = update
            rows = [TRAINER.Trainer._fit_views(trainer,
                    SimpleNamespace(next=targets.__next__), step, defer_metrics=deferred)
                    for step in range(1, 4)]
            if deferred:
                self.assertTrue(all(torch.is_tensor(loss) and not loss.requires_grad
                                    for loss, _ in rows))
                rows = TRAINER.Trainer._materialize_step_metrics(rows)
            results.append((parameter.detach().clone(), rows))
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        self.assertEqual(results[0][1], results[1][1])

    def test_frame_batches_diagnostics_across_resume_tail_without_changing_reports(self):
        class Progress:
            def __init__(self, **kwargs):
                self.n = kwargs['initial']
                self.postfix = []
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def set_postfix(self, values, refresh=False):
                self.postfix.append(values)
            def update(self, count):
                self.n += count

        with TemporaryDirectory() as directory:
            fetcher = object()
            def fit(_, step, *, defer_metrics=False):
                self.assertTrue(defer_metrics)
                return (torch.tensor(step / 7),
                        dict(image_loss=torch.tensor(step / 11),
                             psnr=torch.tensor(step / 13), gs=7))
            module = SimpleNamespace(pre_training_step=Mock(), motion_steps=20,
                total_steps=23, frame_report={}, stage_image_metrics=[])
            trainer = SimpleNamespace(current_frame=1, global_step=4, num_steps=23,
                config=SimpleNamespace(stage_image_camera=None), module=module,
                output_dir=Path(directory), _make_fetcher=Mock(return_value=fetcher),
                _resume_fetcher=Mock(), _fit_views=fit,
                _materialize_step_metrics=TRAINER.Trainer._materialize_step_metrics,
                validation_loop=Mock(return_value={'test': {'psnr': 34.2}}))
            trainer._checkpoint_due = lambda: trainer.global_step % 7 == 0 or trainer.global_step == 23
            trainer.ckpt_save_path = Path(directory) / 'checkpoint.pth'
            saved_steps = []
            trainer.save_checkpoint = lambda: saved_steps.append(trainer.global_step)
            progress = []
            def make_progress(**kwargs):
                bar = Progress(**kwargs)
                progress.append(bar)
                return bar
            transfers = []
            original_tolist = torch.Tensor.tolist
            def tolist(tensor):
                transfers.append(tensor.numel())
                return original_tolist(tensor)
            with patch.object(TRAINER, 'tqdm', make_progress), \
                    patch.object(torch.Tensor, 'tolist', tolist):
                report = TRAINER.Trainer.frame_training_loop(trainer)
            # Resume at step 5: six rows to the first refresh, ten, then a
            # three-step tail. Each row has the loss and two tensor metrics.
            self.assertEqual(transfers, [18, 30, 9])
            self.assertEqual(saved_steps, [7, 14, 21, 23])
            self.assertEqual(report['train_loss_steps'], 19)
            expected_loss, image_ema, psnr_ema = 0., 0., 0.
            for step in range(5, 24):
                expected_loss += float(torch.tensor(step / 7))
                image_ema = .6 * image_ema + .4 * float(torch.tensor(step / 11))
                psnr_ema = .6 * psnr_ema + .4 * float(torch.tensor(step / 13))
            self.assertEqual(report['mean_train_loss'], expected_loss / 19)
            self.assertEqual(report['final_train_metrics'],
                dict(image_loss=float(torch.tensor(23 / 11)),
                     psnr=float(torch.tensor(23 / 13)), gs=7))
            self.assertEqual(progress[0].postfix[-1],
                dict(phase='birth', image=f'{image_ema:.5f}', psnr=f'{psnr_ema:.2f}', gs=7))
            self.assertEqual(json.loads((Path(directory) / 'frame_summary/frame_001.json').read_text()), report)

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


    def test_full_resolution_loss_and_gradients_are_unchanged(self):
        generator = torch.Generator().manual_seed(81)
        image = torch.rand(3, 24, 25, generator=generator, requires_grad=True)
        target = torch.rand(3, 24, 25, generator=generator)
        old = .8 * l1_loss(image, target) + .2 * (1 - MODULE.ssim(image, target))
        new = MODULE.motion_photometric_loss(image, target, .2)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(new, image, retain_graph=True)[0],
                                   torch.autograd.grad(old, image)[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
