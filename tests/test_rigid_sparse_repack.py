"""CPU-only lossless stream repack and independent-run metadata regression."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from tests.test_rigid_tree_compression import CODEC, legacy_packet, packet_fixture
from scripts import repack_rigid_tree_sparse_ids as REPACK
from scripts import replay_rigid_tree as REPLAY


class SparseRepackTests(unittest.TestCase):
    def test_complete_one_frame_run_repackages_without_changing_state(self):
        prior, args = packet_fixture()
        old_packet = legacy_packet(args)
        decoded = CODEC.decode_motion(old_packet, prior)
        n = len(prior.xyz)
        raw = dict(xyz=prior.xyz, rotation=prior.rotation,
                   logscale=torch.zeros(n, 3), opacity=torch.zeros(n, 1),
                   sh_dc=torch.zeros(n, 1, 3), sh_rest=torch.zeros(n, 0, 3))
        newborn = dict(logscale=torch.full((1, 3), .125),
                       opacity=torch.full((1, 1), .25),
                       sh_dc=torch.zeros(1, 1, 3),
                       sh_rest=torch.zeros(1, 0, 3))
        birth_packet = REPACK.birth_codec.encode_birth_attributes(newborn, n)
        raw1 = {name: torch.cat((raw[name], newborn[name]))
                for name in REPACK.birth_codec.ATTRIBUTES}
        raw1.update(xyz=decoded.xyz, rotation=decoded.rotation)
        digest = REPACK.quantization.reference_digest(raw1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            target = root / 'repacked'
            stream = source / 'frame_storage' / 'rigid_tree'
            summary_dir = source / 'frame_summary'
            stream.mkdir(parents=True)
            summary_dir.mkdir()
            config = dict(experiment_name='original_training',
                          data=dict(extra_dataset_kwargs=dict(num_frames=2)))
            (source / 'config.yaml').write_text(yaml.safe_dump(config))
            reference = stream / 'reference_frame0.npz'
            np.savez_compressed(reference,
                **{name: value.numpy() for name, value in raw.items()},
                owner=prior.owner.numpy(), parents=prior.parents.numpy(),
                frame_index=np.array(0, dtype=np.int32),
                region_count=np.array(len(prior.parents), dtype=np.int32))
            (stream / 'motion_001.bin').write_bytes(old_packet)
            (stream / 'birth_001.bin').write_bytes(birth_packet)
            manifest = dict(frame=1,
                reference='frame_storage/rigid_tree/reference_frame0.npz',
                motion='motion_001.bin', birth='birth_001.bin',
                old_count=n, birth_count=1, total_count=n + 1,
                decoded_state_sha256=decoded.digest().hex(),
                raw_state_sha256=digest)
            (stream / 'frame_001.json').write_text(json.dumps(manifest))
            quality = dict(psnr=34., ssim=.97, l1=.015)
            summary = dict(frame=1,
                motion=dict(path=str(stream / 'motion_001.bin'),
                            **CODEC.motion_packet_info(old_packet),
                            bytes=len(old_packet)),
                birth=dict(path=str(stream / 'birth_001.bin'),
                           archive_bytes=len(birth_packet)),
                encoded_bytes=len(old_packet) + len(birth_packet),
                shared_reference_bytes=reference.stat().st_size,
                decoder_reference=manifest['reference'],
                materialized_sha256=digest,
                motion_rate_diagnostics=dict(rate_bytes_hard=float(len(old_packet))),
                evaluation=dict(test=quality),
                checkpoint=str(source / 'large_training_checkpoint.pth'))
            (summary_dir / 'frame_001.json').write_text(json.dumps(summary))
            before = hashlib.sha256(old_packet).hexdigest()
            report = REPACK.repack(source, target)
            self.assertTrue(report['complete'])
            self.assertFalse(report['partial'])
            self.assertEqual(before, hashlib.sha256(
                (stream / 'motion_001.bin').read_bytes()).hexdigest())
            self.assertFalse((target / 'large_training_checkpoint.pth').exists())
            target_stream = target / 'frame_storage' / 'rigid_tree'
            new_packet = (target_stream / 'motion_001.bin').read_bytes()
            self.assertTrue(CODEC.uncompressed_motion_packet(new_packet).startswith(
                CODEC.DELTA_MAGIC))
            actual_state, actual_raw = REPLAY.replay(
                target_stream, target_stream / reference.name, 1)
            self.assertEqual(actual_state.digest(), decoded.digest())
            self.assertEqual(REPACK.quantization.reference_digest(actual_raw),
                             digest)
            saved = json.loads((target / 'frame_summary' /
                                'frame_001.json').read_text())
            self.assertEqual(saved['evaluation']['test'], quality)
            self.assertEqual(saved['motion']['bytes'], len(new_packet))
            self.assertEqual(saved['encoded_bytes'],
                             len(new_packet) + len(birth_packet))
            self.assertEqual(saved['source_training_motion_rate_diagnostics'],
                             summary['motion_rate_diagnostics'])
            self.assertIsNone(saved['checkpoint'])
            self.assertEqual(saved['source_training_checkpoint'],
                             summary['checkpoint'])
            self.assertEqual(len(report['codec_file_sha256']), 64)
            with self.assertRaises(FileExistsError):
                REPACK.repack(source, target)
            config['data']['extra_dataset_kwargs']['num_frames'] = 3
            (source / 'config.yaml').write_text(yaml.safe_dump(config))
            partial = REPACK.repack(source, root / 'explicit_partial', through=1)
            self.assertTrue(partial['partial'])
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['configured_dynamic_frames'], 2)
            self.assertEqual(partial['repacked_dynamic_frames'], 1)

    def test_default_requires_full_sequence_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            (source / 'config.yaml').write_text(yaml.safe_dump(dict(
                experiment_name='incomplete',
                data=dict(extra_dataset_kwargs=dict(num_frames=3)))))
            target = Path(directory) / 'target'
            with self.assertRaises(FileNotFoundError):
                REPACK.repack(source, target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
