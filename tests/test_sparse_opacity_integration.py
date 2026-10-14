"""CPU stream/birth ordering, opacity lineage, digests and actual byte accounting."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_online_flat_regions import CODEC
from scripts import benchmark_rigid_residual_tree as benchmark
from scripts import replay_rigid_tree as replay

BIRTH = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_birth_codec')
OPACITY = importlib.import_module('pipeline.ReCon-GS.model.sparse_opacity')
QUANT = importlib.import_module('pipeline.ReCon-GS.model.birth_quantization')


class SparseOpacityStreamTests(unittest.TestCase):
    def fixture(self, directory, opacity=True):
        stream = Path(directory)
        xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        q = torch.tensor([[1., 0., 0., 0.]]).repeat(3, 1)
        raw = dict(xyz=xyz, rotation=q, opacity=torch.tensor([[2.], [-1.], [.5]]),
                   logscale=torch.full((3, 3), -2.), sh_dc=torch.full((3, 1, 3), .2),
                   sh_rest=torch.zeros(3, 3, 3))
        state = CODEC.DecodedState(xyz, q, torch.zeros(3, dtype=torch.long), torch.tensor([-1]), 0)
        reference = stream / 'reference_frame0.npz'
        np.savez_compressed(reference, **{name: value.numpy() for name, value in raw.items()},
                            owner=state.owner.numpy(), parents=state.parents.numpy(), frame_index=0)
        states, raws, manifests, summaries = {0: state}, {0: raw}, {}, {}
        for frame in (1, 2):
            count = len(state.xyz)
            birth_xyz = xyz[:1] + .02 if frame == 1 else torch.empty(0, 3)
            birth_q = q[:1] if frame == 1 else torch.empty(0, 4)
            sources = torch.tensor([0]) if frame == 1 else torch.empty(0, dtype=torch.long)
            z = torch.zeros(count, 3, dtype=torch.int16); zr = torch.zeros(1, 3, dtype=torch.int16)
            motion = CODEC.encode_motion(state, frame, 1., torch.arange(count), state.owner,
                state.parents, zr, zr, z, z, sources, birth_xyz, birth_q, .001, .001, compression='auto')
            (stream / f'motion_{frame:03d}.bin').write_bytes(motion)
            decoded = CODEC.decode_motion(motion, state)
            current = {name: value.clone() for name, value in raw.items()}
            current.update(xyz=decoded.xyz, rotation=decoded.rotation)
            birth_packet = None
            if frame == 1:
                predictor = {name: raw[name][:1] for name in BIRTH.ATTRIBUTES}
                symbols = {name: torch.zeros_like(value) for name, value in predictor.items()}
                symbols['opacity'].fill_(5)
                steps = {name: .1 for name in BIRTH.ATTRIBUTES}
                newborn = {name: predictor[name] + symbols[name] * torch.tensor(steps[name], dtype=torch.float32)
                           for name in BIRTH.ATTRIBUTES}
                birth_packet = BIRTH.encode_birth_symbols(newborn, count, symbols=symbols,
                    steps=steps, predictor=predictor, source_rows=sources)
                self.assertIsNotNone(birth_packet)
                (stream / 'birth_001.bin').write_bytes(birth_packet)
                for name in BIRTH.ATTRIBUTES:
                    current[name] = torch.cat((raw[name], newborn[name]))
            update = torch.zeros_like(raw['opacity'], dtype=torch.int16)
            if opacity:
                update[0] = -25 if frame == 1 else 10
                if frame == 2:
                    update[3] = -12  # The earlier newborn is now an old GS.
            packet = OPACITY.encode_opacity_delta(update, raw['opacity'], step=.01, frame_index=frame)
            if packet is not None:
                (stream / f'opacity_{frame:03d}.bin').write_bytes(packet)
                current['opacity'][:count] = OPACITY.decode_opacity_delta(packet, raw['opacity'], frame_index=frame).logits
            manifest = dict(frame=frame, motion=f'motion_{frame:03d}.bin',
                birth=f'birth_{frame:03d}.bin' if birth_packet else None,
                reference=reference.name, decoded_state_sha256=decoded.digest().hex(),
                raw_state_sha256=QUANT.reference_digest(current), total_count=len(decoded.xyz))
            if opacity:
                manifest['opacity_update'] = f'opacity_{frame:03d}.bin' if packet else None
            (stream / f'frame_{frame:03d}.json').write_text(json.dumps(manifest))
            summary = dict(motion=dict(bytes=len(motion)), birth=dict(archive_bytes=len(birth_packet)) if birth_packet else None,
                opacity_update=dict(archive_bytes=len(packet)) if packet else None,
                encoded_bytes=len(motion) + (len(birth_packet) if birth_packet else 0) + (len(packet) if packet else 0),
                shared_reference_bytes=reference.stat().st_size, decoder_reference=reference.name)
            states[frame], raws[frame], manifests[frame], summaries[frame] = decoded, current, manifest, summary
            state, raw = decoded, current
        return stream, reference, states, raws, manifests, summaries

    def test_birth_predictor_before_old_update_and_next_frame_newborn_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, reference, states, raws, _, _ = self.fixture(directory)
            state1, raw1 = replay.replay(stream, reference, 1)
            self.assertEqual(float(raw1['opacity'][0]), 1.75)
            self.assertEqual(float(raw1['opacity'][-1]), 2.5, 'Birth must predict from preceding 2.0, not updated 1.75')
            state2, raw2 = replay.replay(stream, reference, 2)
            self.assertEqual(state2.digest(), states[2].digest())
            self.assertEqual(QUANT.reference_digest(raw2), QUANT.reference_digest(raws[2]))
            torch.testing.assert_close(raw2['opacity'][-1], torch.tensor([2.38]), rtol=0, atol=0)
            bench1, braw1 = benchmark._decode_frame(stream, 1, states[0], raws[0])
            bench2, braw2 = benchmark._decode_frame(stream, 2, bench1, braw1)
            self.assertEqual(bench2.digest(), state2.digest())
            self.assertEqual(QUANT.reference_digest(braw2), QUANT.reference_digest(raw2))

    def test_legacy_stream_without_opacity_field_or_file_remains_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, reference, states, raws, _, _ = self.fixture(directory, opacity=False)
            state, raw = replay.replay(stream, reference, 2)
            self.assertEqual(state.digest(), states[2].digest())
            self.assertEqual(QUANT.reference_digest(raw), QUANT.reference_digest(raws[2]))
            self.assertEqual(float(raw['opacity'][0]), 2.)
            self.assertEqual(list(stream.glob('opacity_*.bin')), [])

    def test_actual_storage_and_sequence_totals_include_opacity_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, reference, states, raws, manifests, summaries = self.fixture(directory)
            rows = []
            for frame in (1, 2):
                benchmark._validate_stream_manifest(stream, frame, states[frame], raws[frame], Path(reference.name))
                storage = benchmark._actual_frame_storage(stream, frame, manifests[frame], summaries[frame],
                                                         reference.stat().st_size, reference.name)
                self.assertEqual(storage['opacity_update_bytes'], (stream / f'opacity_{frame:03d}.bin').stat().st_size)
                self.assertEqual(storage['complete_update_bytes'], sum((stream / name).stat().st_size for name in (
                    f'motion_{frame:03d}.bin', f'opacity_{frame:03d}.bin', f'frame_{frame:03d}.json')) +
                    ((stream / 'birth_001.bin').stat().st_size if frame == 1 else 0))
                rows.append(dict(frame=frame, **storage))
            total = benchmark._summarize_storage(rows)
            self.assertEqual(total['opacity_update_bytes_total'], sum(row['opacity_update_bytes'] for row in rows))
            self.assertEqual(total['complete_update_bytes_total'], sum(row['complete_update_bytes'] for row in rows))
            broken = {**summaries[1], 'encoded_bytes': summaries[1]['encoded_bytes'] - rows[0]['opacity_update_bytes']}
            with self.assertRaises(ValueError):
                benchmark._actual_frame_storage(stream, 1, manifests[1], broken, reference.stat().st_size, reference.name)

    def test_enabled_zero_update_has_no_file_and_explicit_zero_byte_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            stream, reference, states, raws, manifests, summaries = self.fixture(directory, opacity=False)
            for frame in (1, 2):
                manifests[frame]['opacity_update'] = None
                (stream / f'frame_{frame:03d}.json').write_text(json.dumps(manifests[frame]))
                storage = benchmark._actual_frame_storage(stream, frame, manifests[frame], summaries[frame],
                                                         reference.stat().st_size, reference.name)
                self.assertEqual(storage['opacity_update_bytes'], 0)
            self.assertEqual(list(stream.glob('opacity_*.bin')), [])
            state, raw = replay.replay(stream, reference, 2)
            self.assertEqual(state.digest(), states[2].digest())
            self.assertEqual(QUANT.reference_digest(raw), QUANT.reference_digest(raws[2]))

    def test_missing_corrupt_wrong_frame_or_reference_packets_are_rejected(self):
        for condition in ('missing', 'corrupt', 'frame', 'reference'):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                stream, reference, states, raws, _, _ = self.fixture(directory)
                path = stream / 'opacity_001.bin'
                if condition == 'missing':
                    path.unlink()
                elif condition == 'corrupt':
                    packet = bytearray(path.read_bytes()); packet[-1] ^= 1; path.write_bytes(packet)
                else:
                    ref = raws[0]['opacity'].clone()
                    if condition == 'reference':
                        ref[0] += .01
                    path.write_bytes(OPACITY.encode_opacity_delta(torch.ones_like(ref), ref,
                        step=.01, frame_index=2 if condition == 'frame' else 1))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    replay.replay(stream, reference, 1)

    def test_orphan_packet_unsafe_manifest_name_and_stale_raw_digest_rejected(self):
        for condition in ('orphan', 'name', 'digest'):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as directory:
                stream, reference, _, _, manifests, _ = self.fixture(directory)
                manifest = manifests[1]
                if condition == 'orphan':
                    manifest.pop('opacity_update')
                elif condition == 'name':
                    manifest['opacity_update'] = '../opacity_001.bin'
                else:
                    manifest['raw_state_sha256'] = '0' * 64
                (stream / 'frame_001.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    replay.replay(stream, reference, 1)


if __name__ == '__main__':
    unittest.main()
