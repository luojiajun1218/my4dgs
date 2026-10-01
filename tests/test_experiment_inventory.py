"""Checks for completion, source precision, and actual-byte report accounting."""
import ast
import importlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Union
from unittest.mock import Mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPORT = importlib.import_module('scripts.summarize_experiments')


def load_production_method(source_name, class_name, method_name, namespace=None):
    """Execute one production method without importing the renderer or GUI package."""
    source = Path(__file__).resolve().parents[1] / 'pipeline' / 'ReCon-GS' / source_name
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    scope = dict(namespace or {})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope[method_name]


class ExperimentInventoryTests(unittest.TestCase):
    @staticmethod
    def storage_fixture():
        parts = dict(gate_bytes=2., region_motion_bytes=24., gs_motion_bytes=0.,
                     new_bytes=96., deletion_bytes=4.)
        return dict(storage_only=True, actual=None,
                    actual_bytes_by_class=None, total_stream_bytes=None, bootstrap_bytes=None,
                    estimated_total_bits=None, estimated_bits_by_class=None,
                    hard_storage_bytes=126., expected_storage_bytes=126.,
                    storage_bytes_by_class=parts, expected_storage_bytes_by_class=dict(parts),
                    reference_storage_bytes=240., new_count=1, deleted_count=1,
                    stage_seconds=dict(state_update=.1),
                    activities=dict(gs=dict(denominator=10,
                                            counts=dict(KEEP=5, SHARED=5, SPLIT=0),
                                            fractions=dict(KEEP=.5, SHARED=.5, SPLIT=0.))))

    def test_storage_reports_parameter_bytes_without_packets_or_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = REPORT.stream_frame(Path(tmp), dict(frame=1, stream=self.storage_fixture()))
            self.assertEqual(row['storage_measurement'], 'parameter_bytes_estimate')
            self.assertEqual(row['hard_storage_bytes'], 126.)
            self.assertEqual(row['storage_region_motion_bytes'], 24.)
            self.assertEqual(row['storage_state_update_seconds'], .1)
            self.assertNotIn('codec_state_update_seconds', row)
            self.assertNotIn('estimated_bits', row)
            self.assertNotIn('stream_bytes', row)
            self.assertFalse((Path(tmp) / 'incremental_stream').exists())

    def test_storage_rejects_wrong_totals_and_encoded_byte_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = self.storage_fixture()
            stream['hard_storage_bytes'] += 1
            with self.assertRaisesRegex(ValueError, 'class totals'):
                REPORT.stream_frame(Path(tmp), dict(frame=1, stream=stream))
            stream = self.storage_fixture()
            stream['total_stream_bytes'] = 0
            with self.assertRaisesRegex(ValueError, 'must not claim'):
                REPORT.stream_frame(Path(tmp), dict(frame=1, stream=stream))
            stream = self.storage_fixture()
            stream['expected_storage_bytes'] = float('nan')
            with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                REPORT.stream_frame(Path(tmp), dict(frame=1, stream=stream))
            stream = self.storage_fixture()
            stream['activities']['gs']['fractions']['KEEP'] = .6
            with self.assertRaisesRegex(ValueError, 'fractions do not match'):
                REPORT.stream_frame(Path(tmp), dict(frame=1, stream=stream))

    def test_storage_run_keeps_quality_and_parameter_estimates_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'fixture_run'
            run.mkdir()
            config = dict(trainer=dict(num_incr_steps=300, num_init_steps=15000),
                data=dict(root='fixture', resolution=-1, extra_dataset_kwargs=dict(num_frames=2)),
                module=dict(sh_degree=1, motion_mode='quantized_residual', quantized_codec_enabled=False))
            (run / 'config.yaml').write_text(yaml.safe_dump(config))
            (run / 'logging.INFO').write_text('')
            directory = run / 'frame_summary'
            directory.mkdir()
            report = dict(frame=1, quantized_evaluation=dict(test=dict(psnr=35., ssim=.98, l1=.01)),
                          timing=dict(training=1., wall=2.), stream=self.storage_fixture())
            (directory / 'frame_001.json').write_text(json.dumps(report))
            entry, rows, _, _ = REPORT.summarize_run(run)
            self.assertEqual(entry['mean']['psnr'], 35.)
            self.assertEqual(entry['mean']['hard_storage_bytes'], 126.)
            self.assertEqual(entry['storage']['mean_hard_bytes'], 126.)
            self.assertEqual(rows[0]['storage_measurement'], 'parameter_bytes_estimate')
            self.assertNotIn('stream', entry)
            self.assertNotIn('estimated_bits', rows[0])

    def test_storage_console_uses_final_birth_count_and_explicit_denominators(self):
        info = Mock()
        finish_summary = load_production_method('trainer.py', 'Trainer',
            'finish_quantized_frame_summary',
            dict(json=json, time=time, logging=SimpleNamespace(info=info)))
        stream = self.storage_fixture()
        stream.update(state_update_seconds=.1, birth_control=dict(
            enabled=True, stats_window_start=1, stats_window_end=2,
            stats_steps_observed=2, clone_eligible=3, clone_added=3,
            clone_blocked=0, split_added=0, split_blocked=0,
            candidate_pool_peak=3, final_candidate_count=3, final_retained=3,
            max_new_gaussians=200, budget_touched=False, candidate_pruned=0,
            opacity_pruned=0, previous_compensation_deleted=1,
            motion_history=dict(old_motion_rows=8, newborn_rows=2,
                                first_frame_rows=0, initialization_decay=.6),
            deletion_protection=dict(protected_keep_old_count=4,
                                     deleted_keep_count=0, eligible_count=6)))
        stream['new_count'] = 3
        stream['tree_setup'] = dict(action='rebuild', interval=5,
            old_region_counts=[1, 3, 9], region_counts=[1, 3, 9],
            initialization='motion_knn_gate_membership', topology_seconds=.02, initialization_seconds=.03)
        stream['stage_seconds']['tree_rebuild'] = .05
        stream['activities']['levels'] = [dict(level=0, denominator=2,
            fractions=dict(KEEP=.5, SHARED=.5, SPLIT=0.))]
        with tempfile.TemporaryDirectory() as tmp:
            fake = SimpleNamespace(
                output_dir=Path(tmp), current_frame=2,
                module=SimpleNamespace(quant_report=stream,
                    config=SimpleNamespace(quantized_residual=SimpleNamespace(enable_learned_gates=True))),
                frame_eval_metrics=dict(test=dict(psnr=35.)),
                frame_times=dict(training=2., evaluation=1., drawing=0., saving=0.),
                synchronize=lambda: None, frame_storage=lambda: dict(total_bytes=1024),
                record_io=lambda *args: None, routine_info=lambda *args: None)
            finish_summary(fake, 0.)
            rendered = '\n'.join(call.args[0] % call.args[1:] for call in info.call_args_list)
            self.assertIn('最终保留=3', rendered)
            self.assertIn('旧GS(n=10，不含新增)', rendered)
            self.assertIn('到达节点', rendered)
            self.assertIn('历史对应(非实际非零运动)', rendered)
            self.assertIn('误删KEEP=0', rendered)
            self.assertIn('action=rebuild 间隔=5帧', rendered)
            self.assertIn('初始化=motion_knn_gate_membership', rendered)
            self.assertIn('重建合计=0.0500s', rendered)

    def test_validation_fps_uses_total_synchronized_forward_time(self):
        import torch
        validation_end = load_production_method('module.py', 'Module', 'validation_end',
                                                dict(List=List, Union=Union))
        fake = SimpleNamespace(
            datamodule=SimpleNamespace(eval_names=['test']), global_step=2,
            trainer=SimpleNamespace(frame_eval_metrics={}, routine_info=lambda *args: None),
            gaussians=SimpleNamespace(get_opacity=torch.ones(2, 1), get_xyz=torch.zeros(2, 3)),
            log_dict=lambda *args: None, log_histogram=lambda *args: None,
            log=lambda *args: None)
        views = [dict(l1=torch.tensor(.1), psnr=torch.tensor(30.),
                      ssim=torch.tensor(.9), forward_seconds=.1),
                 dict(l1=torch.tensor(.1), psnr=torch.tensor(30.),
                      ssim=torch.tensor(.9), forward_seconds=.2)]
        validation_end(fake, views)
        self.assertAlmostEqual(fake.trainer.frame_eval_metrics['test']['fps'], 2 / .3)
        self.assertAlmostEqual(fake.trainer.frame_eval_metrics['test']['mean_forward_seconds'], .15)

    def test_step_console_reads_retained_births_after_incremental_buffer_reset(self):
        import torch
        post_step = load_production_method('module.py', 'Module', 'post_training_step',
            dict(torch=torch, time=time, Optional=Optional, Callable=Callable))
        info = Mock()
        fake = SimpleNamespace(
            timer=Mock(), global_step=300,
            trainer=SimpleNamespace(is_first_frame=False, is_last_step=True,
                synchronize=Mock(), record_io=Mock(), routine_info=info),
            config=SimpleNamespace(motion_mode='quantized_residual'),
            gaussians_incr=SimpleNamespace(get_xyz=torch.empty(0, 3)),
            quant_report=dict(new_count=3),
            post_training_step_incr=Mock(), save_point_cloud=Mock())
        post_step(fake, {})
        info.assert_any_call('Number of incremental gaussians: 3')

    def test_incomplete_tqdm_frame_is_not_a_final_result(self):
        log = (
            'Training progress: 5%|x| 14/300 [00:00, Frame index=15, PSNR (Test)=34.8, Frame time=9.75]\r'
            'Training progress: 5%|x| 15/300 [00:00, Frame index=15, PSNR (Test)=34.8, Frame time=9.75]\r'
            'Training progress: 5%|x| 15/300 [00:00, Frame index=15, PSNR (Test)=34.8, Frame time=9.75]\r'
            'Training progress: 5%|x| 15/300 [00:00, Frame index=16, PSNR (Test)=34.1, Frame time=10.1]'
        )
        rows = REPORT.parse_s2gs_progress(log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['frame'], 14)
        self.assertEqual(rows[0]['original_frame_index'], 15)
        self.assertEqual(rows[0]['psnr'], 34.8)

    def test_per_frame_step_reset_keeps_frame_identity(self):
        log = (
            '==> Start learning frame 1 ...\nEvaluate test dataset at step 200:\n'
            '\tL1: 0.0100 | PSNR: 34.0000 | SSIM: 0.9700 | FPS: 300.0000\n'
            '==> Start learning frame 2 ...\nEvaluate test dataset at step 200:\n'
            '\tL1: 0.0120 | PSNR: 32.0000 | SSIM: 0.9600 | FPS: 300.0000\n'
        )
        rows, starts, _ = REPORT.parse_recongs_log(log)
        self.assertEqual(starts, {1, 2})
        self.assertEqual([(row['frame'], row['step']) for row in rows], [(1, 200), (2, 200)])

    def test_shared_frame_zero_and_partial_windows_are_excluded(self):
        entries = [dict(experiment='reference', family='recongs', external_reference=True),
                   dict(experiment='candidate', family='recongs')]
        rows = [dict(experiment=name, frame=frame, psnr=(99. if frame == 0 else value), ssim=.9)
                for name, value in [('reference', 32.), ('candidate', 31.)] for frame in range(6)]
        comparisons = REPORT.paired_comparisons(entries, rows)
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]['n'], 5)
        self.assertEqual(comparisons[0]['psnr'], 31.)
        self.assertEqual(comparisons[0]['delta_psnr'], -1.)

    def test_bootstrap_is_in_total_but_not_bin_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            stream_dir = run / 'incremental_stream'
            stream_dir.mkdir()
            packet = stream_dir / 'frame_001.bin'
            packet.write_bytes(b'1234567890')
            stream = dict(frame=1, total_stream_bytes=13, bootstrap_bytes=3,
                estimated_total_bits=100., decoded_reference_sha256='fixture',
                actual=dict(total_bytes=10), actual_bytes_by_class=dict(motion_bytes=10, header_bytes=3),
                new_count=0, deleted_count=0, stage_seconds={},
                activities=dict(gs=dict(denominator=2, fractions=dict(KEEP=.5, SHARED=.5, SPLIT=0.))))
            packet.with_suffix('.json').write_text(json.dumps(stream))
            row = REPORT.stream_frame(run, dict(frame=1, stream=stream))
            self.assertEqual(row['stream_bytes'], 13)
            stream['total_stream_bytes'] = 16
            with self.assertRaisesRegex(ValueError, 'bootstrap accounting mismatch'):
                REPORT.stream_frame(run, dict(frame=1, stream=stream))

    def test_namespace_is_never_executed(self):
        self.assertEqual(REPORT.parse_namespace('Namespace(seed=0, sizes=[1, 2])'),
                         dict(seed=0, sizes=[1, 2]))
        with self.assertRaises(ValueError):
            REPORT.parse_namespace("Namespace(seed=__import__('os').getpid())")

    def test_disabled_codec_reports_estimates_without_reading_a_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = dict(codec_enabled=False, total_stream_bytes=None, bootstrap_bytes=None,
                actual=None, actual_bytes_by_class=None, estimated_total_bits=100.,
                new_count=1, deleted_count=2, stage_seconds=dict(state_update=.1),
                activities=dict(gs=dict(denominator=2, fractions=dict(KEEP=.5, SHARED=.5, SPLIT=0.))))
            row = REPORT.stream_frame(Path(tmp), dict(frame=1, stream=stream))
            self.assertEqual(row['estimated_bits'], 100.)
            self.assertEqual(row['rate_measurement'], 'estimated_only')
            self.assertEqual(row['births'], 1)
            self.assertNotIn('stream_bytes', row)
            self.assertNotIn('header_bytes', row)
            self.assertFalse((Path(tmp) / 'incremental_stream').exists())
            stream['total_stream_bytes'] = 0
            with self.assertRaisesRegex(ValueError, 'must not claim actual encoded bytes'):
                REPORT.stream_frame(Path(tmp), dict(frame=1, stream=stream))

    def test_disabled_codec_run_keeps_quality_and_has_no_actual_stream_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'fixture_run'
            run.mkdir()
            config = dict(trainer=dict(num_incr_steps=300, num_init_steps=15000),
                data=dict(root='fixture', resolution=-1, extra_dataset_kwargs=dict(num_frames=2)),
                module=dict(sh_degree=1, motion_mode='quantized_residual', quantized_codec_enabled=False))
            (run / 'config.yaml').write_text(yaml.safe_dump(config))
            (run / 'logging.INFO').write_text('')
            directory = run / 'frame_summary'
            directory.mkdir()
            stream = dict(codec_enabled=False, total_stream_bytes=None, bootstrap_bytes=None,
                actual=None, actual_bytes_by_class=None, estimated_total_bits=100.,
                new_count=1, deleted_count=0, stage_seconds=dict(state_update=.1),
                activities=dict(gs=dict(denominator=2, fractions=dict(KEEP=.5, SHARED=.5, SPLIT=0.))))
            report = dict(frame=1, quantized_evaluation=dict(test=dict(psnr=35., ssim=.98, l1=.01)),
                timing=dict(training=1., wall=2.), stream=stream)
            (directory / 'frame_001.json').write_text(json.dumps(report))
            entry, rows, _, _ = REPORT.summarize_run(run)
            self.assertEqual(entry['status'], 'complete')
            self.assertEqual(entry['mean']['psnr'], 35.)
            self.assertEqual(rows[0]['rate_measurement'], 'estimated_only')
            self.assertNotIn('stream', entry)
            self.assertNotIn('stream_bytes', rows[0])


if __name__ == '__main__':
    unittest.main()
