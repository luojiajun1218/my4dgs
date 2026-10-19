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
              birth_bytes=0, manifest_index_bytes=None):
    row = dict(frame=frame, timed_renders=50, render_seconds=seconds,
                file_decode_verify_seconds=decode_seconds,
                upload_seconds=decode_seconds,
                motion_bytes=motion_bytes, birth_bytes=birth_bytes,
                encoded_bytes=motion_bytes + birth_bytes,
                quality=dict(psnr=psnr))
    if manifest_index_bytes is not None:
        row.update(payload_bytes=motion_bytes + birth_bytes,
                   manifest_index_bytes=manifest_index_bytes,
                   complete_update_bytes=motion_bytes + birth_bytes + manifest_index_bytes)
    return row


class SequenceAggregationTests(unittest.TestCase):
    def test_prefix_measurement_never_changes_the_planned_dataset(self):
        self.assertEqual(benchmark._sequence_frame_count(300, None), 300)
        self.assertEqual(benchmark._sequence_frame_count(300, 29), 30)
        self.assertEqual(benchmark._sequence_frame_count(300, 299), 300)
        for invalid in (-1, 0, 300, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'through-frame'):
                benchmark._sequence_frame_count(300, invalid)
        with self.assertRaises(ValueError):
            benchmark._sequence_frame_count(1, None)

    def test_quality_validation_uses_configured_resolution(self):
        half = SimpleNamespace(expected_image_size=[1345, 1008])
        full = SimpleNamespace(expected_image_size=[1600, 1199])
        image = torch.zeros(3, 1008, 1345)
        shape = benchmark._configured_image_shape(half)
        self.assertEqual(shape, (3, 1008, 1345))
        benchmark._validate_quality_input(image, shape)
        with self.assertRaisesRegex(ValueError, 'cam00 input'):
            benchmark._validate_quality_input(image, benchmark._configured_image_shape(full))
        for value in (float('nan'), -0.1, 1.1):
            image[0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite RGB'):
                benchmark._validate_quality_input(image, shape)
        for invalid in (None, [0, 1008], [1345], [True, 1008]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'expected_image_size'):
                benchmark._configured_image_shape(SimpleNamespace(expected_image_size=invalid))

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
                          motion_bytes=frame, birth_bytes=2 * frame,
                          manifest_index_bytes=0 if frame == 0 else 400 + len(str(frame)))
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
        expected_index_total = sum(400 + len(str(frame)) for frame in range(1, 300))
        self.assertEqual(storage['manifest_index_bytes_total'], expected_index_total)
        self.assertEqual(storage['payload_bytes_mean'], storage['encoded_bytes_mean'])
        self.assertAlmostEqual(storage['complete_update_bytes_mean'],
                               (3 * sum(range(1, 300)) + expected_index_total) / 299)
        self.assertEqual(storage['complete_update_bytes_max'], 897 + 403)
        whole_storage = benchmark._summarize_sequence_storage(rows, reference_bytes=12345)
        self.assertEqual(whole_storage['total_reference_and_dynamic_bytes'],
                         12345 + 3 * sum(range(1, 300)))
        self.assertEqual(whole_storage['total_reference_and_complete_update_bytes'],
                         whole_storage['total_reference_and_dynamic_bytes'] + expected_index_total)
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
            manifest_path = stream / 'frame_001.json'
            manifest_path.write_text(json.dumps(manifest))
            summary = dict(motion=dict(bytes=5), birth=dict(archive_bytes=2),
                           encoded_bytes=7, shared_reference_bytes=11,
                           decoder_reference='reference_frame0.npz')
            measured = benchmark._actual_frame_storage(
                stream, 1, manifest, summary, 11, 'reference_frame0.npz')
            self.assertEqual(measured, dict(motion_bytes=5, birth_bytes=2,
                encoded_bytes=7, payload_bytes=7,
                manifest_index_bytes=manifest_path.stat().st_size,
                complete_update_bytes=7 + manifest_path.stat().st_size))
            summary['encoded_bytes'] = 8
            with self.assertRaisesRegex(ValueError, 'storage summary'):
                benchmark._actual_frame_storage(
                    stream, 1, manifest, summary, 11, 'reference_frame0.npz')
            summary['encoded_bytes'] = 7
            (stream / 'birth_001.bin').write_bytes(b'abcd')
            with self.assertRaisesRegex(ValueError, 'storage summary'):
                benchmark._actual_frame_storage(
                    stream, 1, manifest, summary, 11, 'reference_frame0.npz')

    def test_manifest_actual_bytes_change_complete_mean_without_changing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = Path(directory)
            (stream / 'motion_001.bin').write_bytes(b'12345')
            manifest = dict(motion='motion_001.bin', birth=None)
            path = stream / 'frame_001.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            summary = dict(motion=dict(bytes=5), birth=None, encoded_bytes=5,
                           shared_reference_bytes=11, decoder_reference='reference_frame0.npz')
            first = benchmark._actual_frame_storage(stream, 1, manifest, summary,
                                                     11, 'reference_frame0.npz')
            # UTF-8 byte length and formatting can differ from character count.
            path.write_text(json.dumps({**manifest, 'note': '实际索引'},
                                       ensure_ascii=False, indent=2), encoding='utf-8')
            second = benchmark._actual_frame_storage(stream, 1, manifest, summary,
                                                      11, 'reference_frame0.npz')
            self.assertEqual(first['encoded_bytes'], second['encoded_bytes'])
            self.assertEqual(second['manifest_index_bytes'], path.stat().st_size)
            self.assertGreater(second['manifest_index_bytes'], first['manifest_index_bytes'])
            self.assertGreater(second['manifest_index_bytes'], len(path.read_text(encoding='utf-8')))
            rows = [frame_row(0, 1., 30., manifest_index_bytes=0),
                    dict(frame=1, **first), dict(frame=2, **second)]
            result = benchmark._summarize_sequence_storage(rows, reference_bytes=11)
            dynamic = result['dynamic_packets']
            self.assertEqual(dynamic['encoded_bytes_mean'], 5.)
            self.assertEqual(dynamic['complete_update_bytes_mean'],
                             (first['complete_update_bytes'] + second['complete_update_bytes']) / 2)
            self.assertEqual(result['total_reference_and_complete_update_bytes'],
                             11 + first['complete_update_bytes'] + second['complete_update_bytes'])
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                benchmark._actual_frame_storage(stream, 1, manifest, summary,
                                                 11, 'reference_frame0.npz')

    def test_historical_rows_cannot_silently_claim_zero_index_bytes(self):
        row = frame_row(1, 1., 34., motion_bytes=50000)
        summary = benchmark._summarize_sequence_storage([row], reference_bytes=11)
        self.assertEqual(summary['dynamic_packets']['encoded_bytes_mean'], 50000.)
        self.assertFalse(summary['dynamic_packets']['manifest_index_measured'])
        self.assertIsNone(summary['dynamic_packets']['complete_update_bytes_mean'])
        self.assertIsNone(summary['total_reference_and_complete_update_bytes'])
        measured = frame_row(1, 1., 34., motion_bytes=50000, manifest_index_bytes=436)
        measured['complete_update_bytes'] = 50000
        with self.assertRaisesRegex(ValueError, 'include payload and manifest'):
            benchmark._summarize_storage([measured])

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
        digest = import_module('pipeline.model.birth_quantization').reference_digest
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
