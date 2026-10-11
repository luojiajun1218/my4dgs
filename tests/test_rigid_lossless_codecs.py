"""CPU-only regressions for current and historical rigid stream envelopes."""

import hashlib
import importlib
import json
import lzma
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if 'pipeline.ReCon-GS' not in sys.modules:
    pipeline = ModuleType('pipeline')
    pipeline.__path__ = [str(ROOT / 'pipeline')]
    recon = ModuleType('pipeline.ReCon-GS')
    recon.__path__ = [str(ROOT / 'pipeline' / 'ReCon-GS')]
    sys.modules['pipeline'] = pipeline
    sys.modules['pipeline.ReCon-GS'] = recon

BIRTH = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_birth_codec')
MOTION = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
QUANT = importlib.import_module('pipeline.ReCon-GS.model.birth_quantization')
REPLAY = importlib.import_module('scripts.replay_rigid_tree')


def appearance_fixture(rest=2, symbols_override=None):
    count = 3
    base = dict(
        xyz=torch.zeros(count, 3),
        rotation=torch.tensor([[1., 0., 0., 0.]] * count),
        logscale=torch.arange(9, dtype=torch.float32).reshape(3, 3) / 8,
        opacity=torch.arange(3, dtype=torch.float32).reshape(3, 1) / 8,
        sh_dc=torch.arange(9, dtype=torch.float32).reshape(3, 1, 3) / 8,
        sh_rest=torch.arange(9 * rest, dtype=torch.float32).reshape(3, rest, 3) / 8)
    source = torch.tensor([2, 0])
    symbols = {name: torch.zeros_like(base[name][source]) for name in BIRTH.ATTRIBUTES}
    symbols['logscale'][0] = torch.tensor([-32768., 32767., 32768.])
    symbols['opacity'][:, 0] = torch.tensor([-1., 0.])
    if rest:
        symbols['sh_rest'][1, 0, 0] = -32769.
    if symbols_override is not None:
        symbols_override(symbols)
    predictor = {name: base[name][source] for name in symbols}
    steps = {name: .125 for name in symbols}
    raw = {name: predictor[name] + symbols[name] * steps[name]
           for name in symbols}
    xyz = torch.ones(2, 3)
    rotation = base['rotation'][:2]
    return base, raw, symbols, steps, predictor, source, xyz, rotation


class BirthSymbolCodecTests(unittest.TestCase):
    def test_exact_float32_bytes_and_field_widths(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        packet = BIRTH.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        header = BIRTH.SYMBOL_HEADER.unpack_from(packet)
        self.assertEqual(header[:4], (BIRTH.SYMBOL_MAGIC, 3, 2, 2))
        self.assertEqual(header[4:8], (4, 2, 2, 4))
        decoded = BIRTH.decode_birth_attributes(packet, base, xyz, rotation, source)
        for name in BIRTH.ATTRIBUTES:
            torch.testing.assert_close(decoded[name], raw[name], rtol=0, atol=0)
            self.assertEqual(decoded[name].numpy().tobytes(),
                             raw[name].numpy().tobytes())
        self.assertEqual(decoded['xyz'].numpy().tobytes(), xyz.numpy().tobytes())

    def test_zero_rest_and_legacy_read(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = (
            appearance_fixture(rest=0))
        packet = BIRTH.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        decoded = BIRTH.decode_birth_attributes(packet, base, xyz, rotation, source)
        self.assertEqual(decoded['sh_rest'].shape, (2, 0, 3))
        legacy = BIRTH.encode_birth_attributes(raw, 3)
        old = BIRTH.decode_birth_attributes(legacy, base, xyz, rotation)
        for name in BIRTH.ATTRIBUTES:
            self.assertEqual(old[name].numpy().tobytes(), raw[name].numpy().tobytes())

    def test_export_uses_symbols_or_exactness_fallback(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'birth.bin'
            info = BIRTH.export_birth_attributes(
                path, raw, 3, symbols=symbols, steps=steps,
                predictor=predictor, source_rows=source)
            self.assertFalse(info['exactness_fallback'])
            self.assertEqual(info['archive_bytes'], path.stat().st_size)
            self.assertEqual(path.read_bytes()[:4], BIRTH.SYMBOL_MAGIC)
            changed = dict(raw)
            changed['opacity'] = raw['opacity'].clone()
            changed['opacity'][0, 0] += .01
            info = BIRTH.export_birth_attributes(
                path, changed, 3, symbols=symbols, steps=steps,
                predictor=predictor, source_rows=source)
            self.assertTrue(info['exactness_fallback'])
            self.assertEqual(path.read_bytes()[:4], BIRTH.MAGIC)
            decoded = BIRTH.decode_birth_attributes(
                path.read_bytes(), base, xyz, rotation, source)
            self.assertEqual(decoded['opacity'].numpy().tobytes(),
                             changed['opacity'].numpy().tobytes())

    def test_invalid_sources_steps_shapes_and_noninteger_symbols(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        packet = BIRTH.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        for bad_source in (None, torch.tensor([-1, 0]), torch.tensor([0, 3]),
                           torch.tensor([0]), torch.tensor([0., 1.])):
            with self.assertRaises(ValueError):
                BIRTH.decode_birth_attributes(packet, base, xyz, rotation,
                                              bad_source)
        with self.assertRaises(ValueError):
            BIRTH.decode_birth_attributes(packet, base, xyz[:1], rotation, source)
        for invalid in (0., -1., float('nan'), float('inf'), 1e100):
            bad_steps = dict(steps, opacity=invalid)
            with self.assertRaises(ValueError):
                BIRTH.encode_birth_symbols(
                    raw, 3, symbols=symbols, steps=bad_steps,
                    predictor=predictor, source_rows=source)
        malformed = dict(symbols, opacity=symbols['opacity'][:1])
        with self.assertRaises(ValueError):
            BIRTH.encode_birth_symbols(
                raw, 3, symbols=malformed, steps=steps, predictor=predictor,
                source_rows=source)
        fractional = dict(symbols, opacity=symbols['opacity'] + .25)
        self.assertIsNone(BIRTH.encode_birth_symbols(
            raw, 3, symbols=fractional, steps=steps, predictor=predictor,
            source_rows=source))
        excessive = dict(symbols, opacity=torch.full_like(symbols['opacity'], 2. ** 31))
        self.assertIsNone(BIRTH.encode_birth_symbols(
            raw, 3, symbols=excessive, steps=steps, predictor=predictor,
            source_rows=source))

    def test_header_body_corruption_truncation_and_trailer(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        packet = BIRTH.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        altered_step = bytearray(packet)
        altered_step[19] ^= 1
        altered_width = bytearray(packet)
        altered_width[14] = 2
        altered_crc = bytearray(packet)
        altered_crc[BIRTH.SYMBOL_HEADER.size - 1] ^= 1
        for bad in (packet[:-1], packet + b'X', bytes(altered_step),
                    bytes(altered_width), bytes(altered_crc)):
            with self.assertRaises(ValueError):
                BIRTH.decode_birth_attributes(bad, base, xyz, rotation, source)

    def test_valid_crc_cannot_hide_nonfinite_reconstruction_or_base(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        packet = BIRTH.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        body = zlib.decompress(packet[BIRTH.SYMBOL_HEADER.size:])
        bad = bytearray(packet)
        struct.pack_into('<f', bad, 18, 3e38)
        checksum = zlib.crc32(bad[:BIRTH.SYMBOL_HEADER.size - 4] + body)
        struct.pack_into('<I', bad, BIRTH.SYMBOL_HEADER.size - 4, checksum)
        with self.assertRaisesRegex(ValueError, 'Nonfinite reconstructed'):
            BIRTH.decode_birth_attributes(bytes(bad), base, xyz, rotation, source)
        invalid_base = dict(base, opacity=base['opacity'].clone())
        invalid_base['opacity'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid base opacity'):
            BIRTH.decode_birth_attributes(packet, invalid_base, xyz, rotation, source)


class MotionEnvelopeTests(unittest.TestCase):
    @staticmethod
    def raw_packet(body):
        raw = MOTION.DUAL_STEP_MAGIC + body
        return raw + hashlib.sha256(raw).digest()[:MOTION.CHECKSUM_BYTES]

    def test_auto_compares_actual_packet_lengths(self):
        for body in (b'A' * 10000, bytes(range(256)) * 20):
            raw = self.raw_packet(body)
            zlib_packet = MOTION.compress_motion_packet(raw, mode='zlib')
            lzma_packet = (MOTION.DUAL_STEP_LZMA_COMPRESSED_MAGIC +
                           MOTION._uvar(len(raw)) +
                           lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9))
            chosen = MOTION.compress_motion_packet(raw, mode='auto')
            self.assertEqual(len(chosen), min(map(len, (raw, zlib_packet, lzma_packet))))
            self.assertEqual(MOTION.uncompressed_motion_packet(chosen), raw)
            self.assertEqual(MOTION.uncompressed_motion_packet(lzma_packet), raw)
            self.assertEqual(MOTION.motion_packet_info(lzma_packet)['stored_bytes'],
                             len(lzma_packet))

    def test_lzma_rejects_damage_trailer_and_expansion(self):
        raw = self.raw_packet(b'A' * 10000)
        prefix = MOTION.DUAL_STEP_LZMA_COMPRESSED_MAGIC
        lz = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9)
        good = prefix + MOTION._uvar(len(raw)) + lz
        for bad in (good[:-1], good + b'X', prefix + MOTION._uvar(len(raw) - 1) + lz,
                    prefix + MOTION._uvar(MOTION.MAX_RAW_PACKET_BYTES + 1) + lz):
            with self.assertRaises(ValueError):
                MOTION.uncompressed_motion_packet(bad)


class FullReplayTests(unittest.TestCase):
    def test_rtb2_two_frame_owner_edit_and_birth_replay(self):
        base, _, _, steps, _, _, _, _ = appearance_fixture()
        prior = MOTION.DecodedState(base['xyz'], base['rotation'],
                                    torch.tensor([1, 2, 2]),
                                    torch.tensor([-1, 0, 0]), 0)
        zeros_region = torch.zeros(3, 3, dtype=torch.int16)
        zeros_gs = torch.zeros(3, 3, dtype=torch.int16)
        edited_owner = prior.owner.clone()
        edited_owner[2] = 1
        birth_xyz = torch.tensor([[1., 2., 3.]])
        birth_q = torch.tensor([[1., 0., 0., 0.]])
        motion1 = MOTION.encode_motion(
            prior, 1, 1., torch.arange(3), edited_owner, prior.parents,
            zeros_region, zeros_region, zeros_gs, zeros_gs,
            torch.tensor([2]), birth_xyz, birth_q, .01, .01,
            compression='auto')
        state1 = MOTION.decode_motion(motion1, prior)
        self.assertEqual(int(state1.owner[-1]), 1)
        self.assertNotEqual(int(prior.owner[2]), int(state1.owner[-1]))
        symbols = {name: torch.ones_like(base[name][2:3])
                   for name in BIRTH.ATTRIBUTES}
        predictor = {name: base[name][2:3] for name in BIRTH.ATTRIBUTES}
        newborn = {name: predictor[name] + symbols[name] * steps[name]
                   for name in BIRTH.ATTRIBUTES}
        birth1 = BIRTH.encode_birth_symbols(
            newborn, 3, symbols=symbols, steps=steps,
            predictor=predictor, source_rows=torch.tensor([2]))
        self.assertTrue(birth1.startswith(BIRTH.SYMBOL_MAGIC))
        raw1 = {name: torch.cat((base[name], newborn[name]))
                for name in BIRTH.ATTRIBUTES}
        raw1.update(xyz=state1.xyz, rotation=state1.rotation)
        zeros_next = torch.zeros(4, 3, dtype=torch.int16)
        motion2 = MOTION.encode_motion(
            state1, 2, 1., torch.arange(4), state1.owner, state1.parents,
            zeros_region, zeros_region, zeros_next, zeros_next,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3),
            torch.empty(0, 4), .01, .01, compression='auto')
        state2 = MOTION.decode_motion(motion2, state1)
        with tempfile.TemporaryDirectory() as directory:
            stream = Path(directory)
            reference = stream / 'reference_frame0.npz'
            np.savez_compressed(reference,
                **{name: value.numpy() for name, value in base.items()},
                owner=prior.owner.numpy(), parents=prior.parents.numpy(),
                frame_index=np.array(0, dtype=np.int32))
            (stream / 'motion_001.bin').write_bytes(motion1)
            (stream / 'birth_001.bin').write_bytes(birth1)
            (stream / 'motion_002.bin').write_bytes(motion2)
            (stream / 'frame_001.json').write_text(json.dumps(dict(
                frame=1, decoded_state_sha256=state1.digest().hex(),
                raw_state_sha256=QUANT.reference_digest(raw1))))
            (stream / 'frame_002.json').write_text(json.dumps(dict(
                frame=2, decoded_state_sha256=state2.digest().hex(),
                raw_state_sha256=QUANT.reference_digest(raw1))))
            actual_state, actual_raw = REPLAY.replay(stream, reference, 2)
            self.assertEqual(actual_state.digest(), state2.digest())
            self.assertEqual(QUANT.reference_digest(actual_raw),
                             QUANT.reference_digest(raw1))


if __name__ == '__main__':
    unittest.main()
