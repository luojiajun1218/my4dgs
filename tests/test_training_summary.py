"""Final log coverage, actual packet accounting, FPS weighting, and RNG isolation."""

import json
import hashlib
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.run_summary import build_run_summary, format_run_summary, time_render
from pipeline.trainer import Trainer


class TrainingSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.stream = self.run / 'frame_storage/rigid_tree'
        self.stream.mkdir(parents=True)
        self.reference = self.stream / 'reference_frame0.npz'
        self.reference.write_bytes(b'reference')
        (self.run / 'frame_summary').mkdir()
        self.initial = dict(frame=0, total_count=10,
            evaluation={'test': dict(psnr=30., ssim=.9, lpips=.2)},
            render_timing=dict(timed_renders=50, render_seconds=.3))
        self.payload_sizes, self.index_sizes = [], []

    def write_frame(self, frame, psnr):
        manifest = dict(frame=frame)
        for key, size in [('motion', 4), ('birth', 6), ('opacity_update', 2),
                          ('feature_dc_update', 3)]:
            name = f'{key}_{frame:03d}.bin'
            (self.stream / name).write_bytes(b'x' * size)
            manifest[key] = name
        path = self.stream / f'frame_{frame:03d}.json'
        path.write_text(json.dumps(manifest, indent=2))
        self.payload_sizes.append(15)
        self.index_sizes.append(path.stat().st_size)
        row = dict(frame=frame, evaluation={'test': dict(psnr=psnr, ssim=.96, lpips=.1)},
            frame_manifest=str(path.relative_to(self.run)), total_count=10 + frame,
            initial_reference=dict(retained_count=10), elapsed_seconds=999.,
            timing=dict(version=2, training_seconds=2., optimization_seconds=2.,
                evaluation_seconds=1., checkpoint_seconds=0., bookkeeping_seconds=0.,
                frame_wall_seconds=3.),
            render_timing=dict(timed_renders=50, render_seconds=.1 * frame))
        (self.run / 'frame_summary' / f'frame_{frame:03d}.json').write_text(json.dumps(row))
        return row

    def summary(self, **kwargs):
        return build_run_summary(self.run, 3, initial_report=self.initial, **kwargs)

    def test_includes_reference_and_counts_payloads_indices_and_shared_reference_once(self):
        self.write_frame(1, 33.)
        self.write_frame(2, 36.)
        result = self.summary()
        self.assertTrue(result['sequence_complete'])
        self.assertEqual(result['quality']['test']['metrics']['psnr'], 33.)
        self.assertEqual(result['quality']['test']['metric_frame_counts']['psnr'], 3)
        storage = result['storage']
        complete_bytes = sum(self.payload_sizes) + sum(self.index_sizes)
        self.assertEqual(storage['complete_update_kb_mean'], complete_bytes / 2 / 1000)
        self.assertEqual(storage['total_mb'], (complete_bytes + 9) / 1e6)
        self.assertEqual(storage['reference_amortized_kb_per_frame'], (complete_bytes + 9) / 3000)
        self.assertAlmostEqual(result['pure_render_fps'], 150 / .6)
        self.assertEqual(result['fps_notice'], 'FPS仅供参考')
        self.assertIn('FPS仅供参考', '\n'.join(format_run_summary(result)))
        self.assertEqual(result['model_training_seconds_mean'], 2.)
        self.assertEqual((result['initial_gaussians'], result['final_gaussians']), (10, 12))

    def test_partial_or_missing_measurements_are_labelled_with_coverage(self):
        self.write_frame(1, 33.)
        result = build_run_summary(self.run, 3)
        self.assertFalse(result['sequence_complete'])
        self.assertEqual(result['quality']['test']['metric_frame_counts']['psnr'], 1)
        self.assertIsNone(result['storage']['reference_amortized_kb_per_frame'])
        text = '\n'.join(format_run_summary(result))
        self.assertIn('partial coverage', text)
        self.assertIn('PSNR=33.0000 dB (1/3 frames)', text)

    def test_missing_packet_does_not_become_a_zero_byte_update(self):
        self.write_frame(1, 33.)
        self.write_frame(2, 36.)
        (self.stream / 'birth_001.bin').unlink()
        result = self.summary()
        self.assertEqual(result['storage']['measured_dynamic_frames'], 1)
        self.assertIsNone(result['storage']['complete_update_kb_mean'])
        self.assertIsNone(result['storage']['total_mb'])

    def test_legacy_elapsed_is_not_reported_as_model_training_time(self):
        row = self.write_frame(1, 33.)
        row['timing']['version'] = 1
        (self.run / 'frame_summary/frame_001.json').write_text(json.dumps(row))
        result = self.summary()
        self.assertIsNone(result['model_training_seconds_mean'])
        self.assertIn('model training=N/A', '\n'.join(format_run_summary(result)))

    def test_different_metric_coverage_is_not_filled_with_zeroes(self):
        self.write_frame(1, 33.)
        self.initial['evaluation']['test'].pop('lpips')
        result = self.summary()['quality']['test']
        self.assertEqual(result['metrics']['lpips'], .1)
        self.assertEqual(result['metric_frame_counts'], dict(psnr=2, ssim=2, lpips=1))

    def test_timing_waits_for_gpu_and_excludes_warmups_and_prior_work(self):
        state = dict(now=0., pending=100.)
        def synchronize():
            state['now'] += state.pop('pending', 0.)
        def render():
            state['pending'] = state.get('pending', 0.) + 2.
        result = time_render(render, synchronize, warmups=2, repeats=3,
                             clock=lambda: state['now'])
        self.assertEqual(result['render_seconds'], 6.)
        self.assertEqual(result['render_fps'], .5)
        self.assertEqual(result['fps_notice'], 'FPS仅供参考')

    def test_render_sampling_preserves_rng(self):
        trainer = object.__new__(Trainer)
        trainer._dummy = torch.empty(0)
        trainer.module = SimpleNamespace(
            datamodule=SimpleNamespace(current_frame=1),
            forward=lambda view: {'render': torch.rand(3, 2, 4)})
        view = SimpleNamespace(path=Path('cam00.png'), image=torch.zeros(3, 2, 4))
        before = torch.get_rng_state().clone()
        with patch('torch.cuda.is_available', return_value=False):
            result = trainer._measure_render(view)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(result['image_size'], [4, 2])
        self.assertEqual(result['timed_renders'], 50)

    def test_validation_samples_only_the_selected_heldout_camera(self):
        trainer = object.__new__(Trainer)
        trainer._dummy = torch.empty(0)
        trainer.config = SimpleNamespace(stage_image_camera='cam00.png')
        def view(name):
            result = SimpleNamespace(path=Path(name))
            result.to = lambda device: result
            return result
        train_view, other_test, selected_test = view('cam01.png'), view('cam02.png'), view('cam00.png')
        data = SimpleNamespace(eval_names=['train', 'test'],
            val_dataloader=lambda: [[train_view], [other_test, selected_test]])
        trainer.module = SimpleNamespace(datamodule=data,
            validation_step=Mock(return_value=dict(psnr=34.)),
            validation_end=Mock(return_value={'test': {'psnr': 34.}}))
        trainer._measure_render = Mock(return_value=dict(timed_renders=50, render_seconds=.2))
        result = trainer.validation_loop()
        self.assertEqual(trainer.module.validation_step.call_count, 3)
        trainer._measure_render.assert_called_once_with(selected_test)
        self.assertEqual(result, {'test': {'psnr': 34.}})

    def test_prefix_resume_reads_saved_frames_once_and_keeps_reference_metrics(self):
        self.write_frame(1, 33.)
        source = self.run / 'source'
        source.mkdir()
        initial = dict(self.initial, reference_sha256=hashlib.sha256(b'reference').hexdigest())
        (source / 'reference_evaluation.json').write_text(json.dumps(initial))
        trainer = object.__new__(Trainer)
        trainer._dummy = torch.empty(0)
        trainer.module = SimpleNamespace(datamodule=SimpleNamespace(current_frame=1, num_frames=3))
        trainer.output_dir, trainer.resume_prefix_dir = self.run, source
        trainer.decoder_reference_path = self.reference
        trainer.setup, trainer.validation_loop = Mock(), Mock()
        trainer.training_loop = lambda: [self.write_frame(2, 36.)]
        with patch('pipeline.trainer.logging.info'):
            trainer.fit(trainer.module)
        trainer.validation_loop.assert_not_called()
        saved = json.loads((self.run / 'run_summary.json').read_text())
        self.assertEqual(saved['dynamic_frames'], 2)
        self.assertEqual(saved['quality']['test']['metrics']['psnr'], 33.)
        self.assertEqual(saved['render_measured_frames'], 3)
        self.assertEqual(json.loads((self.run / 'reference_evaluation.json').read_text()), initial)

    def test_fit_prints_and_saves_summary_without_consuming_rng_at_reference_eval(self):
        trainer = object.__new__(Trainer)
        trainer._dummy = torch.empty(0)
        trainer.config = SimpleNamespace()
        trainer.module = SimpleNamespace(gaussians=SimpleNamespace(_xyz=torch.zeros(10, 3)),
            datamodule=SimpleNamespace(current_frame=0, num_frames=3))
        trainer.output_dir = self.run
        trainer.decoder_reference_path = self.reference
        trainer.setup = Mock()
        trainer._validation_render_timing = self.initial['render_timing']
        before = torch.get_rng_state().clone()
        py_before, np_before = random.getstate(), np.random.get_state()
        def validate():
            torch.rand(2)
            random.random()
            np.random.rand()
            return self.initial['evaluation']
        def train():
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(py_before, random.getstate())
            np.testing.assert_equal(np_before, np.random.get_state())
            return [self.write_frame(1, 33.), self.write_frame(2, 36.)]
        trainer.validation_loop, trainer.training_loop = validate, train
        with patch('torch.cuda.is_available', return_value=False), \
                patch('pipeline.trainer.logging.info') as log:
            result = trainer.fit(trainer.module)
        self.assertEqual(len(result), 2)
        saved = json.loads((self.run / 'run_summary.json').read_text())
        self.assertEqual(saved['quality']['test']['metrics']['psnr'], 33.)
        self.assertEqual(saved['render_measured_frames'], 3)
        self.assertIn('Training summary', log.call_args.args[1])
        self.assertTrue((self.run / 'reference_evaluation.json').is_file())


if __name__ == '__main__':
    unittest.main()
