"""Checks for completion, source precision, and actual-byte report accounting."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPORT = importlib.import_module('scripts.summarize_experiments')


class ExperimentInventoryTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
