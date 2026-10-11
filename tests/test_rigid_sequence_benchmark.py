"""CPU checks for honest full-sequence render-FPS accounting."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_rigid_residual_tree as benchmark


def frame_row(frame, seconds, psnr, decode_seconds=0., motion_bytes=0,
              birth_bytes=0):
    return dict(frame=frame, timed_renders=50, render_seconds=seconds,
                file_decode_verify_seconds=decode_seconds,
                upload_seconds=decode_seconds,
                motion_bytes=motion_bytes, birth_bytes=birth_bytes,
                encoded_bytes=motion_bytes + birth_bytes,
                quality=dict(psnr=psnr))


class SequenceAggregationTests(unittest.TestCase):
    def test_render_fps_uses_total_renders_and_only_render_seconds(self):
        rows = [frame_row(0, 1., 30., 100.),
                frame_row(1, 2., 31., 200.),
                frame_row(2, 4., 32., 300.)]
        result = benchmark._sequence_groups(rows, segment_size=2)
        overall = result['all_frames']
        dynamic = result['dynamic_frames']
        self.assertEqual(overall['total_timed_renders'], 150)
        self.assertEqual(overall['total_render_seconds'], 7.)
        self.assertAlmostEqual(overall['aggregate_fps'], 150 / 7)
        self.assertEqual(overall['frame_fps_min'], 12.5)
        self.assertAlmostEqual(overall['frame_fps_p05'], 13.75)
        self.assertEqual(overall['psnr_mean'], 31.)
        self.assertEqual(dynamic['total_timed_renders'], 100)
        self.assertEqual(dynamic['total_render_seconds'], 6.)
        self.assertAlmostEqual(dynamic['aggregate_fps'], 100 / 6)
        self.assertEqual(dynamic['psnr_mean'], 31.5)
        self.assertEqual(result['bootstrap_frame0']['frame_count'], 1)
        self.assertEqual([(part['first_frame'], part['last_frame'])
                          for part in result['all_segments']], [(0, 1), (2, 2)])
        self.assertEqual([(part['first_frame'], part['last_frame'])
                          for part in result['dynamic_segments']], [(1, 2)])

    def test_all_300_and_dynamic_299_ranges_and_segment_weights(self):
        rows = [frame_row(frame, 1. if frame < 150 else 2.,
                          30. + frame / 1000., decode_seconds=999.,
                          motion_bytes=frame, birth_bytes=2 * frame)
                for frame in range(300)]
        result = benchmark._sequence_groups(rows, segment_size=50)
        self.assertEqual(result['all_frames']['frame_count'], 300)
        self.assertEqual(result['dynamic_frames']['frame_count'], 299)
        self.assertEqual(result['all_frames']['total_timed_renders'], 15000)
        self.assertEqual(result['all_frames']['total_render_seconds'], 450.)
        self.assertAlmostEqual(result['all_frames']['aggregate_fps'], 15000 / 450)
        self.assertEqual([(x['first_frame'], x['last_frame'])
                          for x in result['all_segments']],
                         [(0, 49), (50, 99), (100, 149),
                          (150, 199), (200, 249), (250, 299)])
        self.assertEqual([(x['first_frame'], x['last_frame'])
                          for x in result['dynamic_segments']],
                         [(1, 50), (51, 100), (101, 150),
                          (151, 200), (201, 250), (251, 299)])
        storage = result['dynamic_frames']['storage']
        self.assertEqual(storage['dynamic_frame_count'], 299)
        self.assertEqual(storage['motion_bytes_total'], sum(range(1, 300)))
        self.assertEqual(storage['birth_bytes_total'], 2 * sum(range(1, 300)))
        self.assertEqual(storage['encoded_bytes_total'], 3 * sum(range(1, 300)))
        self.assertEqual(storage['encoded_bytes_mean'], 450.)
        self.assertAlmostEqual(storage['encoded_bytes_p95'], 3 * 284.1)
        self.assertEqual(storage['encoded_bytes_max'], 897)
        self.assertEqual(result['all_segments'][0]['storage']['encoded_bytes_total'],
                         3 * sum(range(1, 50)))
        self.assertEqual(result['bootstrap_frame0']['storage']['dynamic_frame_count'], 0)

    def test_cache_selection_honors_explicit_flags(self):
        module = SimpleNamespace(gaussians=SimpleNamespace(render_attribute_cache=True))
        for requested, expected in ((None, True), (True, True), (False, False)):
            with self.subTest(requested=requested):
                args = SimpleNamespace(render_attribute_cache=requested)
                self.assertIs(benchmark._selected_render_cache(module, args), expected)

    def test_actual_packet_sizes_must_match_saved_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = Path(directory)
            (stream / 'motion_001.bin').write_bytes(b'12345')
            (stream / 'birth_001.bin').write_bytes(b'ab')
            manifest = dict(motion='motion_001.bin', birth='birth_001.bin')
            summary = dict(motion=dict(bytes=5), birth=dict(archive_bytes=2),
                           encoded_bytes=7, shared_reference_bytes=11,
                           decoder_reference='reference_frame0.npz')
            measured = benchmark._actual_frame_storage(
                stream, 1, manifest, summary, 11, 'reference_frame0.npz')
            self.assertEqual(measured, dict(motion_bytes=5, birth_bytes=2,
                                            encoded_bytes=7))
            summary['encoded_bytes'] = 8
            with self.assertRaisesRegex(ValueError, 'storage summary'):
                benchmark._actual_frame_storage(
                    stream, 1, manifest, summary, 11, 'reference_frame0.npz')
            summary['encoded_bytes'] = 7
            (stream / 'birth_001.bin').write_bytes(b'abcd')
            with self.assertRaisesRegex(ValueError, 'storage summary'):
                benchmark._actual_frame_storage(
                    stream, 1, manifest, summary, 11, 'reference_frame0.npz')

    def test_missing_duplicate_and_nonpositive_samples_fail(self):
        for rows in ([frame_row(0, 1., 30.), frame_row(2, 1., 30.)],
                     [frame_row(0, 1., 30.), frame_row(0, 1., 30.)]):
            with self.subTest(rows=[x['frame'] for x in rows]):
                with self.assertRaises(ValueError):
                    benchmark._sequence_groups(rows, segment_size=50)
        with self.assertRaises(ValueError):
            benchmark.summarize_render_frames([frame_row(0, 0., 30.)])
        with self.assertRaises(ValueError):
            benchmark._sequence_groups([frame_row(0, 1., 30.)], 0)

    def test_manifest_checks_decoded_and_full_raw_digests(self):
        from importlib import import_module
        digest = import_module('pipeline.ReCon-GS.model.birth_quantization').reference_digest
        raw = dict(xyz=torch.zeros(1, 3), rotation=torch.tensor([[1., 0., 0., 0.]]),
                   logscale=torch.zeros(1, 3), opacity=torch.zeros(1, 1),
                   sh_dc=torch.zeros(1, 1, 3), sh_rest=torch.zeros(1, 0, 3))
        decoded = type('Decoded', (), {'digest': lambda self: b'abc'})()
        manifest = dict(frame=1, decoded_state_sha256=b'abc'.hex(),
                        raw_state_sha256=digest(raw), total_count=1,
                        reference='frame_storage/rigid_tree/reference_frame0.npz')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frame_001.json'
            path.write_text(json.dumps(manifest))
            benchmark._validate_stream_manifest(Path(directory), 1, decoded,
                                                 raw, manifest['reference'])
            manifest['raw_state_sha256'] = 'wrong'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'manifest'):
                benchmark._validate_stream_manifest(Path(directory), 1,
                                                     decoded, raw,
                                                     manifest['reference'])


if __name__ == '__main__':
    unittest.main()
