"""Lossless motion and attribute streams: envelopes, integer bounds, legacy readers, and state restore."""
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
import numpy as np
import torch
from unittest import mock
from io import BytesIO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.model import sparse_integer_codec as INTEGER_CODEC
from pipeline.model import rigid_tree_codec as MOTION_CODEC
from pipeline.model.legacy_opacity_codec import HEADER, MAGIC, decode_opacity_delta, opacity_reference_hash
from pipeline.model.sparse_updates import encode_opacity_delta


BIRTH_CODEC = importlib.import_module('pipeline.model.rigid_tree_birth_codec')


MOTION_CODEC = importlib.import_module('pipeline.model.rigid_tree_codec')


BIRTH_QUANTIZATION = importlib.import_module('pipeline.model.birth_quantization')


REPLAY = importlib.import_module('scripts.replay_rigid_tree')


DecodedState = importlib.import_module(
    'pipeline.model.rigid_tree_codec').DecodedState


OPACITY_CODEC = importlib.import_module('pipeline.model.legacy_opacity_codec')


FIXTURES = [('logit', '4f5044310200000004000000030000000ad7233c0900000094861248bea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '0ad7233c0000803f295c9f407b147ec0'), ('alpha', '4f5044320200000004000000030000000ad7233c09000000b5712892bea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '98dc233d0000803fdf296640eb353fc0'), ('saturated_alpha', '4f5044330200000004000000030000000ad7233c09000000aadcc1dbbea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '0ad7233c0000803f0efc8d407b147ec0')]


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
    symbols = {name: torch.zeros_like(base[name][source]) for name in BIRTH_CODEC.ATTRIBUTES}
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
        packet = BIRTH_CODEC.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        header = BIRTH_CODEC.SYMBOL_HEADER.unpack_from(packet)
        self.assertEqual(header[:4], (BIRTH_CODEC.SYMBOL_MAGIC, 3, 2, 2))
        self.assertEqual(header[4:8], (4, 2, 2, 4))
        decoded = BIRTH_CODEC.decode_birth_attributes(packet, base, xyz, rotation, source)
        for name in BIRTH_CODEC.ATTRIBUTES:
            torch.testing.assert_close(decoded[name], raw[name], rtol=0, atol=0)
            self.assertEqual(decoded[name].numpy().tobytes(),
                             raw[name].numpy().tobytes())
        self.assertEqual(decoded['xyz'].numpy().tobytes(), xyz.numpy().tobytes())

    def test_zero_rest_and_legacy_read(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = (
            appearance_fixture(rest=0))
        packet = BIRTH_CODEC.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        decoded = BIRTH_CODEC.decode_birth_attributes(packet, base, xyz, rotation, source)
        self.assertEqual(decoded['sh_rest'].shape, (2, 0, 3))
        legacy = BIRTH_CODEC.encode_birth_attributes(raw, 3)
        old = BIRTH_CODEC.decode_birth_attributes(legacy, base, xyz, rotation)
        for name in BIRTH_CODEC.ATTRIBUTES:
            self.assertEqual(old[name].numpy().tobytes(), raw[name].numpy().tobytes())

    def test_export_uses_symbols_or_exactness_fallback(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'birth.bin'
            info = BIRTH_CODEC.export_birth_attributes(
                path, raw, 3, symbols=symbols, steps=steps,
                predictor=predictor, source_rows=source)
            self.assertFalse(info['exactness_fallback'])
            self.assertEqual(info['archive_bytes'], path.stat().st_size)
            self.assertEqual(path.read_bytes()[:4], BIRTH_CODEC.SYMBOL_MAGIC)
            changed = dict(raw)
            changed['opacity'] = raw['opacity'].clone()
            changed['opacity'][0, 0] += .01
            info = BIRTH_CODEC.export_birth_attributes(
                path, changed, 3, symbols=symbols, steps=steps,
                predictor=predictor, source_rows=source)
            self.assertTrue(info['exactness_fallback'])
            self.assertEqual(path.read_bytes()[:4], BIRTH_CODEC.MAGIC)
            decoded = BIRTH_CODEC.decode_birth_attributes(
                path.read_bytes(), base, xyz, rotation, source)
            self.assertEqual(decoded['opacity'].numpy().tobytes(),
                             changed['opacity'].numpy().tobytes())

    def test_invalid_sources_steps_shapes_and_noninteger_symbols(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        packet = BIRTH_CODEC.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        for bad_source in (None, torch.tensor([-1, 0]), torch.tensor([0, 3]),
                           torch.tensor([0]), torch.tensor([0., 1.])):
            with self.assertRaises(ValueError):
                BIRTH_CODEC.decode_birth_attributes(packet, base, xyz, rotation,
                                              bad_source)
        with self.assertRaises(ValueError):
            BIRTH_CODEC.decode_birth_attributes(packet, base, xyz[:1], rotation, source)
        for invalid in (0., -1., float('nan'), float('inf'), 1e100):
            bad_steps = dict(steps, opacity=invalid)
            with self.assertRaises(ValueError):
                BIRTH_CODEC.encode_birth_symbols(
                    raw, 3, symbols=symbols, steps=bad_steps,
                    predictor=predictor, source_rows=source)
        malformed = dict(symbols, opacity=symbols['opacity'][:1])
        with self.assertRaises(ValueError):
            BIRTH_CODEC.encode_birth_symbols(
                raw, 3, symbols=malformed, steps=steps, predictor=predictor,
                source_rows=source)
        fractional = dict(symbols, opacity=symbols['opacity'] + .25)
        self.assertIsNone(BIRTH_CODEC.encode_birth_symbols(
            raw, 3, symbols=fractional, steps=steps, predictor=predictor,
            source_rows=source))
        excessive = dict(symbols, opacity=torch.full_like(symbols['opacity'], 2. ** 31))
        self.assertIsNone(BIRTH_CODEC.encode_birth_symbols(
            raw, 3, symbols=excessive, steps=steps, predictor=predictor,
            source_rows=source))

    def test_header_body_corruption_truncation_and_trailer(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        packet = BIRTH_CODEC.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        altered_step = bytearray(packet)
        altered_step[19] ^= 1
        altered_width = bytearray(packet)
        altered_width[14] = 2
        altered_crc = bytearray(packet)
        altered_crc[BIRTH_CODEC.SYMBOL_HEADER.size - 1] ^= 1
        for bad in (packet[:-1], packet + b'X', bytes(altered_step),
                    bytes(altered_width), bytes(altered_crc)):
            with self.assertRaises(ValueError):
                BIRTH_CODEC.decode_birth_attributes(bad, base, xyz, rotation, source)

    def test_valid_crc_cannot_hide_nonfinite_reconstruction_or_base(self):
        base, raw, symbols, steps, predictor, source, xyz, rotation = appearance_fixture()
        packet = BIRTH_CODEC.encode_birth_symbols(
            raw, 3, symbols=symbols, steps=steps, predictor=predictor,
            source_rows=source)
        body = zlib.decompress(packet[BIRTH_CODEC.SYMBOL_HEADER.size:])
        bad = bytearray(packet)
        struct.pack_into('<f', bad, 18, 3e38)
        checksum = zlib.crc32(bad[:BIRTH_CODEC.SYMBOL_HEADER.size - 4] + body)
        struct.pack_into('<I', bad, BIRTH_CODEC.SYMBOL_HEADER.size - 4, checksum)
        with self.assertRaisesRegex(ValueError, 'Nonfinite reconstructed'):
            BIRTH_CODEC.decode_birth_attributes(bytes(bad), base, xyz, rotation, source)
        invalid_base = dict(base, opacity=base['opacity'].clone())
        invalid_base['opacity'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid base opacity'):
            BIRTH_CODEC.decode_birth_attributes(packet, invalid_base, xyz, rotation, source)


class MotionEnvelopeTests(unittest.TestCase):
    @staticmethod
    def raw_packet(body):
        raw = MOTION_CODEC.DUAL_STEP_MAGIC + body
        return raw + hashlib.sha256(raw).digest()[:MOTION_CODEC.CHECKSUM_BYTES]

    def test_auto_compares_actual_packet_lengths(self):
        for body in (b'A' * 10000, bytes(range(256)) * 20):
            raw = self.raw_packet(body)
            zlib_packet = MOTION_CODEC.compress_motion_packet(raw, mode='zlib')
            lzma_packet = (MOTION_CODEC.DUAL_STEP_LZMA_COMPRESSED_MAGIC +
                           MOTION_CODEC._uvar(len(raw)) +
                           lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9))
            chosen = MOTION_CODEC.compress_motion_packet(raw, mode='auto')
            self.assertEqual(len(chosen), min(map(len, (raw, zlib_packet, lzma_packet))))
            self.assertEqual(MOTION_CODEC.uncompressed_motion_packet(chosen), raw)
            self.assertEqual(MOTION_CODEC.uncompressed_motion_packet(lzma_packet), raw)
            self.assertEqual(MOTION_CODEC.motion_packet_info(lzma_packet)['stored_bytes'],
                             len(lzma_packet))

    def test_lzma_rejects_damage_trailer_and_expansion(self):
        raw = self.raw_packet(b'A' * 10000)
        prefix = MOTION_CODEC.DUAL_STEP_LZMA_COMPRESSED_MAGIC
        lz = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9)
        good = prefix + MOTION_CODEC._uvar(len(raw)) + lz
        for bad in (good[:-1], good + b'X', prefix + MOTION_CODEC._uvar(len(raw) - 1) + lz,
                    prefix + MOTION_CODEC._uvar(MOTION_CODEC.MAX_RAW_PACKET_BYTES + 1) + lz):
            with self.assertRaises(ValueError):
                MOTION_CODEC.uncompressed_motion_packet(bad)

    def test_delta_sparse_rows_reject_zero_gap_overflow_and_truncation(self):
        for payload in (
            MOTION_CODEC._uvar(1) + MOTION_CODEC._uvar(0) + b'\x01' + struct.pack('<3h', 1, 0, 0),
            MOTION_CODEC._uvar(1) + MOTION_CODEC._uvar(3) + b'\x01' + struct.pack('<3h', 1, 0, 0),
            MOTION_CODEC._uvar(1) + MOTION_CODEC._uvar(1) + b'\x01' + b'\x01\x00',
        ):
            with self.assertRaises(ValueError):
                MOTION_CODEC._sparse_read(MOTION_CODEC._Reader(payload), 2, delta_rows=True)
        valid = (MOTION_CODEC._uvar(2) + MOTION_CODEC._uvar(1) + b'\x01' +
                 struct.pack('<3h', 1, 0, 0) + MOTION_CODEC._uvar(1) + b'\x02' +
                 struct.pack('<3h', 0, 1, 0))
        translation, rotation = MOTION_CODEC._sparse_read(
            MOTION_CODEC._Reader(valid), 2, delta_rows=True)
        torch.testing.assert_close(translation[0], torch.tensor([1, 0, 0], dtype=torch.int16))
        torch.testing.assert_close(rotation[1], torch.tensor([0, 1, 0], dtype=torch.int16))


class FullReplayTests(unittest.TestCase):
    def test_rtb2_two_frame_owner_edit_and_birth_replay(self):
        base, _, _, steps, _, _, _, _ = appearance_fixture()
        prior = MOTION_CODEC.DecodedState(base['xyz'], base['rotation'],
                                    torch.tensor([1, 2, 2]),
                                    torch.tensor([-1, 0, 0]), 0)
        zeros_region = torch.zeros(3, 3, dtype=torch.int16)
        zeros_gs = torch.zeros(3, 3, dtype=torch.int16)
        edited_owner = prior.owner.clone()
        edited_owner[2] = 1
        birth_xyz = torch.tensor([[1., 2., 3.]])
        birth_q = torch.tensor([[1., 0., 0., 0.]])
        motion1 = MOTION_CODEC.encode_motion(
            prior, 1, 1., torch.arange(3), edited_owner, prior.parents,
            zeros_region, zeros_region, zeros_gs, zeros_gs,
            torch.tensor([2]), birth_xyz, birth_q, .01, .01,
            compression='auto')
        state1 = MOTION_CODEC.decode_motion(motion1, prior)
        self.assertEqual(int(state1.owner[-1]), 1)
        self.assertNotEqual(int(prior.owner[2]), int(state1.owner[-1]))
        symbols = {name: torch.ones_like(base[name][2:3])
                   for name in BIRTH_CODEC.ATTRIBUTES}
        predictor = {name: base[name][2:3] for name in BIRTH_CODEC.ATTRIBUTES}
        newborn = {name: predictor[name] + symbols[name] * steps[name]
                   for name in BIRTH_CODEC.ATTRIBUTES}
        birth1 = BIRTH_CODEC.encode_birth_symbols(
            newborn, 3, symbols=symbols, steps=steps,
            predictor=predictor, source_rows=torch.tensor([2]))
        self.assertTrue(birth1.startswith(BIRTH_CODEC.SYMBOL_MAGIC))
        raw1 = {name: torch.cat((base[name], newborn[name]))
                for name in BIRTH_CODEC.ATTRIBUTES}
        raw1.update(xyz=state1.xyz, rotation=state1.rotation)
        zeros_next = torch.zeros(4, 3, dtype=torch.int16)
        motion2 = MOTION_CODEC.encode_motion(
            state1, 2, 1., torch.arange(4), state1.owner, state1.parents,
            zeros_region, zeros_region, zeros_next, zeros_next,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3),
            torch.empty(0, 4), .01, .01, compression='auto')
        state2 = MOTION_CODEC.decode_motion(motion2, state1)
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
                raw_state_sha256=BIRTH_QUANTIZATION.reference_digest(raw1))))
            (stream / 'frame_002.json').write_text(json.dumps(dict(
                frame=2, decoded_state_sha256=state2.digest().hex(),
                raw_state_sha256=BIRTH_QUANTIZATION.reference_digest(raw1))))
            actual_state, actual_raw = REPLAY.replay(stream, reference, 2)
            self.assertEqual(actual_state.digest(), state2.digest())
            self.assertEqual(BIRTH_QUANTIZATION.reference_digest(actual_raw),
                             BIRTH_QUANTIZATION.reference_digest(raw1))


def packet_fixture(count=256):
    xyz = torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 3).clone()
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1
    owner = torch.ones(count, dtype=torch.long)
    parents = torch.tensor([-1, 0], dtype=torch.long)
    prior = MOTION_CODEC.DecodedState(xyz, rotation, owner, parents, 0)
    region = torch.zeros(2, 3, dtype=torch.int16)
    gs_translation = torch.full((count, 3), 42, dtype=torch.int16)
    gs_rotation = torch.full((count, 3), 7, dtype=torch.int16)
    args = (prior, 1, 1., torch.arange(count), owner, parents,
            region, region, gs_translation, gs_rotation,
            torch.tensor([0]), torch.tensor([[999., 0., 0.]]),
            torch.tensor([[1., 0., 0., 0.]]), .01, .01)
    return prior, args


def legacy_packet(args, **kwargs):
    """Build a pre-delta wire fixture without adding a legacy writer mode."""
    def absolute_sparse_write(result, translation, rotation):
        active_t = translation.ne(0).any(-1)
        active_r = rotation.ne(0).any(-1)
        active = torch.where(active_t | active_r)[0].tolist()
        result.extend(MOTION_CODEC._uvar(len(active)))
        for row in active:
            flags = int(active_t[row]) | (int(active_r[row]) << 1)
            result.extend(MOTION_CODEC._uvar(row))
            result.append(flags)
            if flags & 1:
                result.extend(struct.pack('<3h', *translation[row].tolist()))
            if flags & 2:
                result.extend(struct.pack('<3h', *rotation[row].tolist()))

    with mock.patch.object(MOTION_CODEC, '_sparse_write', absolute_sparse_write):
        new_raw = MOTION_CODEC.encode_motion(*args, **kwargs)
    old_magic = (MOTION_CODEC.DUAL_STEP_MAGIC if new_raw.startswith(
        MOTION_CODEC.DUAL_STEP_DELTA_MAGIC) else MOTION_CODEC.MAGIC)
    content = old_magic + new_raw[4:-MOTION_CODEC.CHECKSUM_BYTES]
    return content + hashlib.sha256(content).digest()[:MOTION_CODEC.CHECKSUM_BYTES]


class MotionPacketCompressionTests(unittest.TestCase):
    def test_compressed_and_legacy_packets_replay_identically(self):
        prior, args = packet_fixture()
        delta_raw = MOTION_CODEC.encode_motion(*args)
        legacy = legacy_packet(args)
        packed = MOTION_CODEC.encode_motion(*args, compression='auto')
        assert legacy.startswith(MOTION_CODEC.MAGIC)
        assert delta_raw.startswith(MOTION_CODEC.DELTA_MAGIC)
        assert packed.startswith((MOTION_CODEC.DELTA_COMPRESSED_MAGIC,
                                  MOTION_CODEC.DELTA_LZMA_COMPRESSED_MAGIC))
        assert len(packed) < len(delta_raw)
        old_state = MOTION_CODEC.decode_motion(legacy, prior)
        new_state = MOTION_CODEC.decode_motion(packed, prior)
        assert MOTION_CODEC.decode_motion(delta_raw, prior).digest() == old_state.digest()
        for name in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources'):
            torch.testing.assert_close(getattr(new_state, name), getattr(old_state, name),
                                       rtol=0, atol=0)
        assert new_state.digest() == old_state.digest()
        info = MOTION_CODEC.motion_packet_info(packed)
        assert info == dict(format=('RSC1-zlib' if packed.startswith(
                                MOTION_CODEC.DELTA_COMPRESSED_MAGIC) else 'RSL1-lzma'),
                            stored_bytes=len(packed),
                            raw_stream_bytes=len(delta_raw),
                            compression_savings_bytes=len(delta_raw) - len(packed))
        for wrapper in (MOTION_CODEC.compress_motion_packet(legacy, mode='zlib'),
                        MOTION_CODEC.LZMA_COMPRESSED_MAGIC + MOTION_CODEC._uvar(len(legacy)) +
                        lzma.compress(legacy, format=lzma.FORMAT_XZ, preset=9)):
            assert MOTION_CODEC.decode_motion(wrapper, prior).digest() == old_state.digest()

    def test_auto_compression_keeps_incompressible_legacy_packet(self):
        seed = b'rigid-tree-incompressible-fixture'
        body = b''.join(hashlib.sha256(seed + i.to_bytes(4, 'little')).digest()
                        for i in range(16))
        raw = MOTION_CODEC.MAGIC + body
        raw += hashlib.sha256(raw).digest()[:MOTION_CODEC.CHECKSUM_BYTES]
        assert MOTION_CODEC.compress_motion_packet(raw, mode='auto') == raw

    def test_compressed_packet_rejects_mutation_truncation_trailing_and_bomb(self):
        prior, args = packet_fixture()
        packet = MOTION_CODEC.encode_motion(*args, compression='zlib')
        damaged = bytearray(packet)
        damaged[len(damaged) // 2] ^= 1
        for candidate in (bytes(damaged), packet[:-1], packet + b'X',
                          MOTION_CODEC.DELTA_COMPRESSED_MAGIC + MOTION_CODEC._uvar(
                              MOTION_CODEC.MAX_RAW_PACKET_BYTES + 1) + packet[5:]):
            try:
                MOTION_CODEC.decode_motion(candidate, prior)
            except ValueError:
                pass
            else:
                raise AssertionError('Damaged compressed stream was accepted')

    def test_lzma_packet_roundtrips_and_rejects_damage_trailers_and_expansion(self):
        prior, args = packet_fixture()
        raw = MOTION_CODEC.encode_motion(*args)
        packet = (MOTION_CODEC.DELTA_LZMA_COMPRESSED_MAGIC + MOTION_CODEC._uvar(len(raw)) +
                  lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9))
        assert MOTION_CODEC.uncompressed_motion_packet(packet) == raw
        assert MOTION_CODEC.decode_motion(packet, prior).digest() == (
            MOTION_CODEC.decode_motion(raw, prior).digest())
        assert MOTION_CODEC.motion_packet_info(packet)['stored_bytes'] == len(packet)
        for bad in (packet[:-1], packet + b'X',
                    packet[:5] + MOTION_CODEC._uvar(len(raw) - 1) + packet[6:],
                    MOTION_CODEC.DELTA_LZMA_COMPRESSED_MAGIC + MOTION_CODEC._uvar(
                        MOTION_CODEC.MAX_RAW_PACKET_BYTES + 1) + packet[6:]):
            try:
                MOTION_CODEC.uncompressed_motion_packet(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Damaged LZMA stream was accepted')


def wrapped(layout, raw_length, magics):
    prefix = INTEGER_CODEC._varint(raw_length) + INTEGER_CODEC._varint(len(layout))
    return [magics[0] + prefix + layout,
            magics[1] + prefix + zlib.compress(layout, 6),
            magics[2] + prefix + lzma.compress(layout, preset=9)]


def spatial_fixture():
    from _motion_fixtures import staged_motion
    model = staged_motion(repetitions=32)
    prior = model.codec_previous_state
    rows = torch.arange(len(prior.xyz))
    rows = rows[(rows % 11) != 0]
    owner = prior.owner[rows]
    generator = torch.Generator().manual_seed(73)
    rt = torch.zeros(len(prior.parents), 3, dtype=torch.int16)
    rr = torch.zeros_like(rt)
    rt[owner.unique()] = torch.tensor([3, 0, -7], dtype=torch.int16)
    gt = torch.randint(-8, 9, (len(rows), 3), generator=generator, dtype=torch.int16)
    gt[0] = torch.tensor([-32768, 32767, 0], dtype=torch.int16)
    gr = torch.zeros_like(gt)
    gr[torch.arange(len(rows)), torch.randint(3, (len(rows),), generator=generator)] = 1
    gr[1] = torch.tensor([0, -32768, 32767], dtype=torch.int16)
    sources = torch.tensor([0, len(rows) - 1])
    args = (prior, 1, 1., rows, owner, prior.parents, rt, rr, gt, gr,
            sources, prior.xyz[rows[sources]] + .001, prior.rotation[rows[sources]], .001, .001)
    raw = MOTION_CODEC.encode_motion(*args, gs_translation_step_extent=.001,
                          gs_rotation_step_rad=.001, spatial_open=prior.spatial_open)
    return prior, raw


def opacity_fixture():
    generator = torch.Generator().manual_seed(27)
    reference = torch.randn(2048, 1, generator=generator)
    reference[0] = -0.
    symbols = torch.randint(-4, 5, reference.shape, generator=generator)
    symbols[1, 0], symbols[-1, 0] = -32767, 32767
    rows = symbols[:, 0].ne(0).nonzero(as_tuple=True)[0]
    body = bytearray()
    previous = -1
    for row in rows.tolist():
        body.extend(INTEGER_CODEC._varint(row - previous))
        body.extend(struct.pack('<h', int(symbols[row, 0])))
        previous = row
    prefix = HEADER.pack(MAGIC, 1, len(reference), len(rows), .125,
                         len(body), 0, opacity_reference_hash(reference))
    header = HEADER.pack(MAGIC, 1, len(reference), len(rows), .125,
                         len(body), zlib.crc32(prefix + body), opacity_reference_hash(reference))
    return reference, symbols, header + body


class SparseIntegerCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_vectorized_varints_preserve_legacy_bytes_at_boundaries(self):
        import numpy as np
        values = np.r_[np.array([0, 1, 127, 128, 16383, 16384, 2097151, 2097152,
                                268435455, 268435456, 0xffffffff], dtype=np.uint64),
                       np.random.default_rng(29).integers(0, 2**32, 2048, dtype=np.uint64)]
        expected = b''.join(INTEGER_CODEC._varint(value) for value in values)
        self.assertEqual(INTEGER_CODEC.encode_unsigned_values(values), expected)
        self.assertEqual(INTEGER_CODEC.encode_unsigned_values([]), b'')

    def test_motion_all_envelopes_restore_exact_raw_and_state_with_deletions_births(self):
        prior, raw = spatial_fixture()
        layout = INTEGER_CODEC.encode_motion_layout(raw)
        expected = MOTION_CODEC.decode_motion(raw, prior)
        for packet in wrapped(layout, len(raw), INTEGER_CODEC.MOTION_MAGICS):
            with self.subTest(magic=packet[:4]):
                self.assertEqual(MOTION_CODEC.uncompressed_motion_packet(packet), raw)
                actual = MOTION_CODEC.decode_motion(packet, prior)
                self.assertEqual(actual.digest(), expected.digest())
                for name in ('xyz', 'rotation', 'owner', 'parents', 'survivor_rows', 'birth_sources'):
                    self.assertEqual(getattr(actual, name).numpy().tobytes(),
                                     getattr(expected, name).numpy().tobytes())
                self.assertEqual(MOTION_CODEC.motion_packet_info(packet)['stored_bytes'], len(packet))
        chosen = MOTION_CODEC.compress_motion_packet(raw)
        old = [raw, MOTION_CODEC.compress_motion_packet(raw, 'zlib'),
               MOTION_CODEC.SPATIAL_LZMA_COMPRESSED_MAGIC + MOTION_CODEC._uvar(len(raw)) + lzma.compress(raw, preset=9)]
        self.assertLessEqual(len(chosen), min(map(len, old)))
        self.assertEqual(MOTION_CODEC.uncompressed_motion_packet(chosen), raw)

    def test_opacity_auto_and_all_envelopes_keep_logits_reference_bits_and_symbols(self):
        reference, symbols, raw = opacity_fixture()
        layout = INTEGER_CODEC.encode_opacity_layout(raw)
        legacy = raw[:HEADER.size] + zlib.compress(raw[HEADER.size:], 6)
        expected = decode_opacity_delta(legacy, reference, frame_index=1)
        auto = encode_opacity_delta(reference, symbols, frame_index=1, step=.125)
        self.assertLessEqual(len(auto), len(legacy))
        for packet in [auto, *wrapped(layout, len(raw), INTEGER_CODEC.OPACITY_MAGICS)]:
            decoded = decode_opacity_delta(packet, reference, frame_index=1)
            self.assertEqual(decoded.logits.numpy().tobytes(), expected.logits.numpy().tobytes())
            self.assertEqual(decoded.symbols.numpy().tobytes(), expected.symbols.numpy().tobytes())
            self.assertEqual(int(decoded.logits.view(torch.int32)[0]),
                             int(expected.logits.view(torch.int32)[0]))
            with self.assertRaises(ValueError):
                decode_opacity_delta(packet, reference + .01, frame_index=1)
            with self.assertRaises(ValueError):
                decode_opacity_delta(packet, reference, frame_index=2)

    def test_motion_and_opacity_reject_damage_truncation_trailers_and_length_limits(self):
        prior, raw_motion = spatial_fixture()
        reference, _, raw_opacity = opacity_fixture()
        for raw, magics, transform, decode in [
                (raw_motion, INTEGER_CODEC.MOTION_MAGICS, INTEGER_CODEC.encode_motion_layout,
                 lambda p: MOTION_CODEC.decode_motion(p, prior)),
                (raw_opacity, INTEGER_CODEC.OPACITY_MAGICS, INTEGER_CODEC.encode_opacity_layout,
                 lambda p: decode_opacity_delta(p, reference))]:
            layout = transform(raw)
            for packet in wrapped(layout, len(raw), magics):
                mutated = bytearray(packet)
                mutated[-1] ^= 1
                bad_length = magics[0] + INTEGER_CODEC._varint(0xffffffff) + INTEGER_CODEC._varint(len(layout)) + layout
                for bad in (packet[:-1], packet + b'X', bytes(mutated), bad_length):
                    with self.subTest(magic=packet[:4]), self.assertRaises(ValueError):
                        decode(bad)

    def test_planar_rejects_noncanonical_gaps_and_out_of_range_int16(self):
        for payload in (b'\x80\x00', INTEGER_CODEC._varint(65536)):
            with self.assertRaises(ValueError):
                INTEGER_CODEC._read_values(INTEGER_CODEC._Reader(payload), 1, 1)
        for gap in (b'\0', INTEGER_CODEC._varint(4)):
            with self.assertRaises(ValueError):
                INTEGER_CODEC._Reader(gap).gaps(1, 3)


def _state(frame):
    return DecodedState(
        xyz=torch.tensor([[0.25, -1., 2.]], dtype=torch.float32),
        rotation=torch.tensor([[1., 0., 0., 0.]], dtype=torch.float32),
        owner=torch.tensor([1], dtype=torch.int64),
        parents=torch.tensor([-1, 0], dtype=torch.int64),
        frame_index=frame,
        birth_sources=torch.tensor([0], dtype=torch.int32) if frame else None,
        survivor_rows=torch.tensor([2], dtype=torch.int64) if frame else None)


class DecodedStateRestoreTests(unittest.TestCase):
    def test_cpu_restore_preserves_values_dtypes_and_optional_birth_sources(self):
        for frame in (0, 49):
            with self.subTest(frame=frame):
                original = _state(frame)
                restored = DecodedState.restore(original.capture())
                self.assertEqual(restored.frame_index, frame)
                for field in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources', 'survivor_rows'):
                    before, after = getattr(original, field), getattr(restored, field)
                    if before is None:
                        self.assertIsNone(after)
                    else:
                        self.assertEqual(after.device.type, 'cpu')
                        self.assertEqual(after.dtype, before.dtype)
                        torch.testing.assert_close(after, before, rtol=0, atol=0)
                self.assertEqual(restored.digest(), original.digest())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_map_location_restores_previous_and_reference_to_cpu(self):
        states = dict(codec_previous_state=_state(49).capture(),
                      frame_reference_state=_state(48).capture())
        buffer = BytesIO()
        torch.save(states, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location='cuda:0', weights_only=False)
        for name, packet in loaded.items():
            self.assertEqual(packet['parents'].device.type, 'cuda')
            restored = DecodedState.restore(packet)
            self.assertEqual(restored.frame_index,
                             49 if name == 'codec_previous_state' else 48)
            for field in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources', 'survivor_rows'):
                value = getattr(restored, field)
                if value is not None:
                    self.assertEqual(value.device.type, 'cpu')
                    self.assertEqual(value.dtype, packet[field].dtype)
                    torch.testing.assert_close(value, packet[field].cpu(),
                                               rtol=0, atol=0)


class HistoricalOpacityReplayTests(unittest.TestCase):
    def reference(self):
        return torch.tensor([[0.], [1.], [5.], [-4.]], dtype=torch.float32)

    def test_all_historical_coordinate_modes_reconstruct_exact_logits(self):
        for mode, encoded, logits in FIXTURES:
            with self.subTest(mode=mode):
                actual = OPACITY_CODEC.decode_opacity_delta(bytes.fromhex(encoded), self.reference(), frame_index=2)
                expected = torch.from_numpy(np.frombuffer(bytes.fromhex(logits), dtype='<f4').copy()).reshape(-1, 1)
                torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
                self.assertEqual(actual.mode, mode)

    def test_corrupt_packets_are_rejected(self):
        for mode, encoded, _ in FIXTURES:
            with self.subTest(mode=mode):
                damaged = bytearray.fromhex(encoded)
                damaged[-1] ^= 1
                with self.assertRaises(ValueError):
                    OPACITY_CODEC.decode_opacity_delta(bytes(damaged), self.reference(), frame_index=2)

    def test_reference_and_frame_mismatches_are_rejected(self):
        encoded = bytes.fromhex(FIXTURES[0][1])
        with self.assertRaises(ValueError):
            OPACITY_CODEC.decode_opacity_delta(encoded, self.reference() + .01, frame_index=2)
        with self.assertRaises(ValueError):
            OPACITY_CODEC.decode_opacity_delta(encoded, self.reference(), frame_index=3)


if __name__ == '__main__':
    unittest.main()
