"""Codec tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import copy
import hashlib
import importlib
import json
import lzma
import struct
import tempfile
import unittest
import zlib
from dataclasses import asdict
from io import BytesIO
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from pipeline.model import rigid_tree_codec as M, sparse_integer_codec as S
from pipeline.model.birth_quantization import (
    QuantizedNewGaussianView,
    RAW_ATTRIBUTES,
    RAW_FIELDS,
    materialize_quantized_gaussians,
    model_raw,
)
from pipeline.model.gaussian import GaussianModel
from pipeline.model.legacy_opacity_codec import (
    HEADER,
    MAGIC,
    decode_opacity_delta,
    opacity_reference_hash,
)
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.rigid_tree_birth_codec import export_birth_attributes
from pipeline.model.rigid_tree_codec import DecodedState, decode_motion
from pipeline.model.sparse_updates import (
    OpacityUpdateConfig,
    QuantizedOpacityUpdate,
    encode_opacity_delta,
    sparse_block_support,
)
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.module import (
    BirthConfig,
    Module,
    replace_low_opacity_gaussians,
    restore_retired_reference,
)
from pipeline.trainer import Trainer
from scripts import benchmark_rigid_residual_tree as benchmark
from scripts.benchmark_rigid_residual_tree import _decode_frame
from scripts.replay_rigid_tree import decode_frame, replay

# ----- test_rigid_tree_compression -----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CODEC = importlib.import_module('pipeline.model.rigid_tree_codec')
RIGID = importlib.import_module('pipeline.model.rigid_residual_tree')


def packet_fixture(count=256):
    xyz = torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 3).clone()
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1
    owner = torch.ones(count, dtype=torch.long)
    parents = torch.tensor([-1, 0], dtype=torch.long)
    prior = CODEC.DecodedState(xyz, rotation, owner, parents, 0)
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
        result.extend(CODEC._uvar(len(active)))
        for row in active:
            flags = int(active_t[row]) | (int(active_r[row]) << 1)
            result.extend(CODEC._uvar(row))
            result.append(flags)
            if flags & 1:
                result.extend(struct.pack('<3h', *translation[row].tolist()))
            if flags & 2:
                result.extend(struct.pack('<3h', *rotation[row].tolist()))

    with mock.patch.object(CODEC, '_sparse_write', absolute_sparse_write):
        new_raw = CODEC.encode_motion(*args, **kwargs)
    old_magic = (CODEC.DUAL_STEP_MAGIC if new_raw.startswith(
        CODEC.DUAL_STEP_DELTA_MAGIC) else CODEC.MAGIC)
    content = old_magic + new_raw[4:-CODEC.CHECKSUM_BYTES]
    return content + hashlib.sha256(content).digest()[:CODEC.CHECKSUM_BYTES]


class MotionPacketCompressionTests(unittest.TestCase):
    def test_compressed_and_legacy_packets_replay_identically(self):
        prior, args = packet_fixture()
        delta_raw = CODEC.encode_motion(*args)
        legacy = legacy_packet(args)
        packed = CODEC.encode_motion(*args, compression='auto')
        assert legacy.startswith(CODEC.MAGIC)
        assert delta_raw.startswith(CODEC.DELTA_MAGIC)
        assert packed.startswith((CODEC.DELTA_COMPRESSED_MAGIC,
                                  CODEC.DELTA_LZMA_COMPRESSED_MAGIC))
        assert len(packed) < len(delta_raw)
        old_state = CODEC.decode_motion(legacy, prior)
        new_state = CODEC.decode_motion(packed, prior)
        assert CODEC.decode_motion(delta_raw, prior).digest() == old_state.digest()
        for name in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources'):
            torch.testing.assert_close(getattr(new_state, name), getattr(old_state, name),
                                       rtol=0, atol=0)
        assert new_state.digest() == old_state.digest()
        info = CODEC.motion_packet_info(packed)
        assert info == dict(format=('RSC1-zlib' if packed.startswith(
                                CODEC.DELTA_COMPRESSED_MAGIC) else 'RSL1-lzma'),
                            stored_bytes=len(packed),
                            raw_stream_bytes=len(delta_raw),
                            compression_savings_bytes=len(delta_raw) - len(packed))
        for wrapper in (CODEC.compress_motion_packet(legacy, mode='zlib'),
                        CODEC.LZMA_COMPRESSED_MAGIC + CODEC._uvar(len(legacy)) +
                        lzma.compress(legacy, format=lzma.FORMAT_XZ, preset=9)):
            assert CODEC.decode_motion(wrapper, prior).digest() == old_state.digest()


    def test_auto_compression_keeps_incompressible_legacy_packet(self):
        seed = b'rigid-tree-incompressible-fixture'
        body = b''.join(hashlib.sha256(seed + i.to_bytes(4, 'little')).digest()
                        for i in range(16))
        raw = CODEC.MAGIC + body
        raw += hashlib.sha256(raw).digest()[:CODEC.CHECKSUM_BYTES]
        assert CODEC.compress_motion_packet(raw, mode='auto') == raw


    def test_compressed_packet_rejects_mutation_truncation_trailing_and_bomb(self):
        prior, args = packet_fixture()
        packet = CODEC.encode_motion(*args, compression='zlib')
        damaged = bytearray(packet)
        damaged[len(damaged) // 2] ^= 1
        for candidate in (bytes(damaged), packet[:-1], packet + b'X',
                          CODEC.DELTA_COMPRESSED_MAGIC + CODEC._uvar(
                              CODEC.MAX_RAW_PACKET_BYTES + 1) + packet[5:]):
            try:
                CODEC.decode_motion(candidate, prior)
            except ValueError:
                pass
            else:
                raise AssertionError('Damaged compressed stream was accepted')


    def test_lzma_packet_roundtrips_and_rejects_damage_trailers_and_expansion(self):
        prior, args = packet_fixture()
        raw = CODEC.encode_motion(*args)
        packet = (CODEC.DELTA_LZMA_COMPRESSED_MAGIC + CODEC._uvar(len(raw)) +
                  lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9))
        assert CODEC.uncompressed_motion_packet(packet) == raw
        assert CODEC.decode_motion(packet, prior).digest() == (
            CODEC.decode_motion(raw, prior).digest())
        assert CODEC.motion_packet_info(packet)['stored_bytes'] == len(packet)
        for bad in (packet[:-1], packet + b'X',
                    packet[:5] + CODEC._uvar(len(raw) - 1) + packet[6:],
                    CODEC.DELTA_LZMA_COMPRESSED_MAGIC + CODEC._uvar(
                        CODEC.MAX_RAW_PACKET_BYTES + 1) + packet[6:]):
            try:
                CODEC.uncompressed_motion_packet(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Damaged LZMA stream was accepted')


    def test_proximal_gs_keeps_large_signal_and_zeroes_small_signal(self):
        from test_spatial_motion import SpatialGatedMotionTests
        model = SpatialGatedMotionTests().model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = .01
        model.last_step = 2
        translation_grad = torch.zeros_like(model.gs_translation)
        rotation_grad = torch.zeros_like(model.gs_rotation)
        translation_grad[0, 0] = rotation_grad[0, 0] = 1e-8
        translation_grad[1, 0] = rotation_grad[1, 0] = 1e-2
        model.gs_translation.grad = translation_grad
        model.gs_rotation.grad = rotation_grad
        model.optimizer_step()
        model._proximal_gs_blocks()
        assert bool((model.gs_translation[0] == 0).all())
        assert bool((model.gs_rotation[0] == 0).all())
        assert bool(model.gs_translation[1].ne(0).any())
        assert bool(model.gs_rotation[1].ne(0).any())
        assert model.last_metrics['motion/prox_zero_translation_blocks'] >= 1
        diagnostics = model.loss_terms(2, diagnostics=True)
        assert int(diagnostics['gs_translation_active_blocks']) == 1
        assert int(diagnostics['gs_rotation_active_blocks']) == 1


    def test_proximal_gs_handles_one_optimizer_block_without_moments(self):
        from test_spatial_motion import SpatialGatedMotionTests
        model = SpatialGatedMotionTests().model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = .01
        model.last_step = 2
        gradient = torch.zeros_like(model.gs_translation)
        gradient[0, 0] = 1e-8
        gradient[1, 0] = 1e-2
        model.gs_translation.grad = gradient
        model.optimizer_step()
        model._proximal_gs_blocks()
        assert model.gs_rotation not in model.optimizer.state
        assert bool((model.gs_translation[0] == 0).all())
        assert bool(model.gs_translation[1].ne(0).any())


# ----- test_rigid_lossless_codecs -----
sys.path.insert(0, str(ROOT))
BIRTH = importlib.import_module('pipeline.model.rigid_tree_birth_codec')
MOTION = importlib.import_module('pipeline.model.rigid_tree_codec')
QUANT = importlib.import_module('pipeline.model.birth_quantization')
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

    def test_delta_sparse_rows_reject_zero_gap_overflow_and_truncation(self):
        for payload in (
            MOTION._uvar(1) + MOTION._uvar(0) + b'\x01' + struct.pack('<3h', 1, 0, 0),
            MOTION._uvar(1) + MOTION._uvar(3) + b'\x01' + struct.pack('<3h', 1, 0, 0),
            MOTION._uvar(1) + MOTION._uvar(1) + b'\x01' + b'\x01\x00',
        ):
            with self.assertRaises(ValueError):
                MOTION._sparse_read(MOTION._Reader(payload), 2, delta_rows=True)
        valid = (MOTION._uvar(2) + MOTION._uvar(1) + b'\x01' +
                 struct.pack('<3h', 1, 0, 0) + MOTION._uvar(1) + b'\x02' +
                 struct.pack('<3h', 0, 1, 0))
        translation, rotation = MOTION._sparse_read(
            MOTION._Reader(valid), 2, delta_rows=True)
        torch.testing.assert_close(translation[0], torch.tensor([1, 0, 0], dtype=torch.int16))
        torch.testing.assert_close(rotation[1], torch.tensor([0, 1, 0], dtype=torch.int16))


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


# ----- test_rigid_reference_pruning -----
sys.path.insert(0, str(ROOT))
module = importlib.import_module('pipeline.module')
trainer = importlib.import_module('pipeline.trainer')
gaussian = importlib.import_module('pipeline.model.gaussian')
codec = importlib.import_module('pipeline.model.rigid_tree_codec')
replay__rigid_reference_pruning = importlib.import_module('scripts.replay_rigid_tree')


class FakeNativeGaussians:
    def __init__(self):
        self.prune_calls = 0
        self.restore(self.raw(), None)

    @staticmethod
    def raw():
        centers = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        return dict(_xyz=torch.cat((centers, torch.zeros(4, 2)), dim=1),
                    _rotation=torch.tensor([[1., 0., 0., 0.]] * 4),
                    _scaling=torch.zeros(4, 3),
                    _opacity=torch.logit(torch.tensor([[.01], [.2], [.005], [.5]])),
                    _features_dc=centers.reshape(4, 1, 1).expand(4, 1, 3).clone(),
                    _features_rest=torch.zeros(4, 0, 3))

    def capture(self):
        return {name: getattr(self, name).detach().clone()
                for name in module.RAW_ATTRIBUTES}

    def restore(self, packet, config):
        for name in module.RAW_ATTRIBUTES:
            setattr(self, name, torch.nn.Parameter(packet[name].detach().clone()))

    def prune_points(self, mask):
        self.prune_calls += 1
        for name in module.RAW_ATTRIBUTES:
            value = getattr(self, name)
            setattr(self, name, torch.nn.Parameter(value.detach()[~mask].clone()))


def fake_module(config, gaussians, frame):
    holder = SimpleNamespace(config=config, gaussians=gaussians,
        renderer=SimpleNamespace(gaussians=None), current_frame=frame,
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=15000)),
        _freeze_base=lambda: [getattr(gaussians, name).requires_grad_(False)
                              for name in module.RAW_ATTRIBUTES])
    return holder


class NativeReferencePruningTests(unittest.TestCase):
    def test_threshold_validation_and_full_delete_guard(self):
        for bad in (-.01, 1., float('nan'), float('inf'), True, '0.015'):
            with self.subTest(threshold=bad), self.assertRaisesRegex(
                    ValueError, 'initial_opacity_threshold'):
                module.validate_initial_opacity_threshold(bad)
        original = FakeNativeGaussians()
        with self.assertRaisesRegex(ValueError, 'every Gaussian'):
            module.prune_native_initial_gaussians(original, .9)
        self.assertEqual(original.prune_calls, 0)
        self.assertEqual(len(original._xyz), 4)
        unpruned = module.prune_native_initial_gaussians(original, 0.)
        self.assertEqual((unpruned['original_count'], unpruned['retained_count'],
                          unpruned['removed_count']), (4, 4, 0))
        self.assertEqual(original.prune_calls, 0)

    def test_native_prune_keeps_deterministic_rows_and_own_resume_does_not_prune(self):
        config = module.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        module.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        torch.testing.assert_close(native._xyz[:, 0], torch.tensor([1., 3.]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(native._features_dc[:, 0, 0],
                                   torch.tensor([1., 3.]), rtol=0, atol=0)
        self.assertEqual(native.prune_calls, 1)
        self.assertEqual(active.reference_count, 2)
        self.assertEqual(active.initial_reference, dict(
            source='native_frame0_checkpoint', threshold=.015,
            original_count=4, retained_count=2, removed_count=2))
        self.assertTrue(all(not getattr(native, name).requires_grad
                            for name in module.RAW_ATTRIBUTES))

        own = dict(gaussians=native.capture(), rigid_tree_version=1,
            rigid_tree_config=asdict(config.rigid_tree),
            rigid_tree_module_config=asdict(config),
            rigid_tree_initial_reference=active.initial_reference,
            rigid_tree_frame=dict(prepared=False, finalized=True,
                motion_committed=False, reference_count=2, report=None,
                birth_control={}), births=None)
        restored = FakeNativeGaussians()
        resume = fake_module(config, restored, frame=1)
        module.Module.on_load_checkpoint(resume, own)
        self.assertEqual(restored.prune_calls, 0)
        self.assertEqual(len(restored._xyz), 2)
        self.assertEqual(resume.initial_reference, active.initial_reference)
        self.assertEqual(resume.reference_count, 2)

    def test_pruned_reference_layout_and_frame_zero_replay_have_same_count(self):
        config = module.ModuleConfig(initial_opacity_threshold=.015)
        native = FakeNativeGaussians()
        active = fake_module(config, native, frame=0)
        module.Module.on_load_checkpoint(active,
            dict(global_step=15000, gaussians=native.capture()))
        with tempfile.TemporaryDirectory() as directory:
            worker = SimpleNamespace(output_dir=Path(directory), module=active,
                datamodule=SimpleNamespace(cameras_extent=1.))
            active.motion = importlib.import_module('pipeline.model.spatial_gated_motion').SpatialGatedMotion(config.rigid_tree)
            trainer.Trainer._write_reference(worker, frame_index=0)
            with np.load(worker.decoder_reference_path, allow_pickle=False) as packet:
                self.assertEqual(int(packet['initial_original_count']), 4)
                self.assertEqual(int(packet['initial_retained_count']), 2)
                self.assertEqual(int(packet['initial_removed_count']), 2)
                self.assertEqual(float(packet['initial_opacity_threshold']), .015)
                self.assertEqual(len(packet['owner']), 2)
            state, raw = replay__rigid_reference_pruning.replay(worker.decoder_reference_path.parent,
                                       worker.decoder_reference_path, through=0)
            self.assertEqual(len(state.xyz), active.reference_count)
            torch.testing.assert_close(raw['xyz'][:, 0], torch.tensor([1., 3.]),
                                       rtol=0, atol=0)
            rebuilt = active.motion.codec_previous_state
            torch.testing.assert_close(state.owner, rebuilt.owner, rtol=0, atol=0)
            torch.testing.assert_close(state.parents, rebuilt.parents, rtol=0, atol=0)


class ActiveRenderCacheTests(unittest.TestCase):
    @staticmethod
    def model(trainable):
        result = gaussian.GaussianModel(0)
        raw = dict(_xyz=torch.tensor([[-.25, 0., 0.], [.25, 0., 0.]]),
                   _rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                   _scaling=torch.full((2, 3), -.4),
                   _opacity=torch.tensor([[.1], [-.2]]),
                   _features_dc=torch.tensor([[[.2, .4, .6]], [[.8, .3, .1]]]),
                   _features_rest=torch.zeros(2, 0, 3))
        for name, value in raw.items():
            setattr(result, name, torch.nn.Parameter(value, requires_grad=trainable))
        return result

    @staticmethod
    def image(model):
        # A small differentiable image assembled from the same renderer inputs.
        pixels = torch.linspace(-1., 1., 7)
        centers = model.get_xyz[:, :1]
        width = model.get_scaling[:, :1]
        alpha = model.get_opacity
        rotation = model.get_rotation
        color = model.get_features[:, 0, :] * (rotation[:, :1] + rotation[:, 1:2])
        coverage = torch.exp(-.5 * ((centers - pixels) / width).square()) * alpha
        return coverage.T @ color

    def test_cache_preserves_image_and_attribute_gradients(self):
        cached, uncached = self.model(True), self.model(True)
        uncached.render_attribute_cache = False
        left, right = self.image(cached), self.image(uncached)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().sum().backward()
        right.square().sum().backward()
        for name in module.RAW_ATTRIBUTES:
            torch.testing.assert_close(getattr(cached, name).grad,
                                       getattr(uncached, name).grad,
                                       rtol=0, atol=0)

    def test_frozen_cache_preserves_motion_gradient_and_invalidates_on_change(self):
        cached, uncached = self.model(False), self.model(False)
        uncached.render_attribute_cache = False
        cached.rigid_xyz = cached._xyz.detach().clone().requires_grad_()
        uncached.rigid_xyz = uncached._xyz.detach().clone().requires_grad_()
        cached.rigid_rotation = cached._rotation.detach().clone().requires_grad_()
        uncached.rigid_rotation = uncached._rotation.detach().clone().requires_grad_()
        with torch.no_grad():
            self.assertIs(cached.get_features, cached.get_features)
            self.assertIs(cached.get_scaling, cached.get_scaling)
            self.assertIs(cached.get_opacity, cached.get_opacity)
        left, right = self.image(cached), self.image(uncached)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.square().sum().backward()
        right.square().sum().backward()
        torch.testing.assert_close(cached.rigid_xyz.grad, uncached.rigid_xyz.grad,
                                   rtol=0, atol=0)
        torch.testing.assert_close(cached.rigid_rotation.grad,
                                   uncached.rigid_rotation.grad, rtol=0, atol=0)
        self.assertGreater(float(cached.rigid_rotation.grad.abs().sum()), 0.)
        with torch.no_grad():
            old = cached.get_opacity
            cached._opacity.add_(.25)
            self.assertIsNot(cached.get_opacity, old)
            torch.testing.assert_close(cached.get_opacity,
                                       torch.sigmoid(cached._opacity), rtol=0, atol=0)


class TerminalBirthTests(unittest.TestCase):
    def test_final_optimizer_update_precedes_prune_without_new_split(self):
        quantization = importlib.import_module('pipeline.model.birth_quantization')
        events = []
        births = gaussian.BirthGaussianModel(0)
        fields = dict(_xyz=torch.zeros(2, 3),
                      _rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                      _scaling=torch.zeros(2, 3),
                      _opacity=torch.logit(torch.tensor([[.005], [.002]])),
                      _features_dc=torch.zeros(2, 1, 3),
                      _features_rest=torch.zeros(2, 0, 3))
        for name, value in fields.items():
            setattr(births, name, torch.nn.Parameter(value))
        births.max_radii2D = torch.zeros(2)
        births.xyz_gradient_accum = torch.zeros(2, 1)
        births.denom = torch.zeros(2, 1)
        births.source_rows = torch.tensor([0, 1])
        groups = [(key, attr) for key, attr in (
            ('xyz', '_xyz'), ('f_dc', '_features_dc'),
            ('f_rest', '_features_rest'), ('opacity', '_opacity'),
            ('scaling', '_scaling'), ('rotation', '_rotation'))]
        births.optimizer = torch.optim.Adam([
            dict(name=key, params=[getattr(births, attr)])
            for key, attr in groups], lr=1.)
        births._opacity.grad = torch.tensor([[-1.], [0.]])
        original_step = births.optimizer.step
        original_zero = births.optimizer.zero_grad
        original_prune = births.prune_points
        def step():
            events.append('step')
            original_step()
        def zero_grad(**kwargs):
            events.append('zero')
            original_zero(**kwargs)
        def prune_points(remove):
            events.append('prune')
            original_prune(remove)
        births.optimizer.step = step
        births.optimizer.zero_grad = zero_grad
        births.prune_points = prune_points
        births.add_densification_stats = lambda *args, **kwargs: None
        def fail_densify(*args, **kwargs):
            raise AssertionError('Terminal step must not split')
        births.adaptive_densify_and_prune = fail_densify
        config = module.ModuleConfig()
        config.rigid_tree.motion_steps = 100
        config.rigid_tree.birth_steps = 200
        config.birth.densify_interval = 40  # Final step is a former split point.
        active = SimpleNamespace(config=config, motion_committed=True,
            births=births, gaussians=SimpleNamespace(_xyz=torch.zeros(2, 3)),
            trainer=SimpleNamespace(global_step=300), motion_steps=100,
            total_steps=300, datamodule=SimpleNamespace(cameras_extent=20.),
            birth_control=dict(pruned=0, split_added=0),
            finalize_frame=lambda: events.append('finalize'))
        active._prune_terminal_births = lambda: module.Module._prune_terminal_births(active)
        package = dict(visibility_filter=torch.tensor([False, False, True, True]),
                       radii=torch.zeros(4), viewspace_points=None)
        module.Module.post_training_step(active, package)
        self.assertEqual(events, ['step', 'zero', 'prune', 'finalize'])
        self.assertEqual(active.birth_control['pruned'], 1)
        self.assertEqual(active.birth_control['split_added'], 0)
        torch.testing.assert_close(births.source_rows, torch.tensor([0]),
                                   rtol=0, atol=0)
        self.assertEqual(len(births._xyz), 1)
        reference = dict(xyz=torch.zeros(2, 3),
            rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
            logscale=torch.zeros(2, 3), opacity=torch.zeros(2, 1),
            sh_dc=torch.zeros(2, 1, 3), sh_rest=torch.zeros(2, 0, 3))
        view = quantization.QuantizedNewGaussianView(
            births, config.birth, 20., reference, source_rows=births.source_rows)
        materialized, sources = quantization.materialize_quantized_gaussians(
            reference, view)
        self.assertEqual(len(materialized['xyz']), 3)
        torch.testing.assert_close(sources, torch.tensor([0]), rtol=0, atol=0)


# ----- test_sparse_integer_codec -----
sys.path.insert(0, str(ROOT))


def wrapped(layout, raw_length, magics):
    prefix = S._varint(raw_length) + S._varint(len(layout))
    return [magics[0] + prefix + layout,
            magics[1] + prefix + zlib.compress(layout, 6),
            magics[2] + prefix + lzma.compress(layout, preset=9)]


def spatial_fixture():
    from test_spatial_motion import SpatialGatedMotionTests
    model = SpatialGatedMotionTests().model(repetitions=32)
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
    raw = M.encode_motion(*args, gs_translation_step_extent=.001,
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
        body.extend(S._varint(row - previous))
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
        expected = b''.join(S._varint(value) for value in values)
        self.assertEqual(S.encode_unsigned_values(values), expected)
        self.assertEqual(S.encode_unsigned_values([]), b'')

    def test_motion_all_envelopes_restore_exact_raw_and_state_with_deletions_births(self):
        prior, raw = spatial_fixture()
        layout = S.encode_motion_layout(raw)
        expected = M.decode_motion(raw, prior)
        for packet in wrapped(layout, len(raw), S.MOTION_MAGICS):
            with self.subTest(magic=packet[:4]):
                self.assertEqual(M.uncompressed_motion_packet(packet), raw)
                actual = M.decode_motion(packet, prior)
                self.assertEqual(actual.digest(), expected.digest())
                for name in ('xyz', 'rotation', 'owner', 'parents', 'survivor_rows', 'birth_sources'):
                    self.assertEqual(getattr(actual, name).numpy().tobytes(),
                                     getattr(expected, name).numpy().tobytes())
                self.assertEqual(M.motion_packet_info(packet)['stored_bytes'], len(packet))
        chosen = M.compress_motion_packet(raw)
        old = [raw, M.compress_motion_packet(raw, 'zlib'),
               M.SPATIAL_LZMA_COMPRESSED_MAGIC + M._uvar(len(raw)) + lzma.compress(raw, preset=9)]
        self.assertLessEqual(len(chosen), min(map(len, old)))
        self.assertEqual(M.uncompressed_motion_packet(chosen), raw)

    def test_opacity_auto_and_all_envelopes_keep_logits_reference_bits_and_symbols(self):
        reference, symbols, raw = opacity_fixture()
        layout = S.encode_opacity_layout(raw)
        legacy = raw[:HEADER.size] + zlib.compress(raw[HEADER.size:], 6)
        expected = decode_opacity_delta(legacy, reference, frame_index=1)
        auto = encode_opacity_delta(reference, symbols, frame_index=1, step=.125)
        self.assertLessEqual(len(auto), len(legacy))
        for packet in [auto, *wrapped(layout, len(raw), S.OPACITY_MAGICS)]:
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
                (raw_motion, S.MOTION_MAGICS, S.encode_motion_layout,
                 lambda p: M.decode_motion(p, prior)),
                (raw_opacity, S.OPACITY_MAGICS, S.encode_opacity_layout,
                 lambda p: decode_opacity_delta(p, reference))]:
            layout = transform(raw)
            for packet in wrapped(layout, len(raw), magics):
                mutated = bytearray(packet)
                mutated[-1] ^= 1
                bad_length = magics[0] + S._varint(0xffffffff) + S._varint(len(layout)) + layout
                for bad in (packet[:-1], packet + b'X', bytes(mutated), bad_length):
                    with self.subTest(magic=packet[:4]), self.assertRaises(ValueError):
                        decode(bad)

    def test_planar_rejects_noncanonical_gaps_and_out_of_range_int16(self):
        for payload in (b'\x80\x00', S._varint(65536)):
            with self.assertRaises(ValueError):
                S._read_values(S._Reader(payload), 1, 1)
        for gap in (b'\0', S._varint(4)):
            with self.assertRaises(ValueError):
                S._Reader(gap).gaps(1, 3)


# ----- test_gaussian_replacement -----
class Rows:
    def __init__(self, raw):
        self.prune_calls = 0
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self, attr, raw[name].detach().clone())

    def prune_points(self, remove):
        self.prune_calls += 1
        for attr in RAW_ATTRIBUTES:
            setattr(self, attr, getattr(self, attr)[~remove].clone())


def reference_raw():
    xyz = torch.cartesian_prod(*[torch.tensor([-1., 1.])] * 3)
    ids = torch.arange(len(xyz), dtype=torch.float32)
    return dict(xyz=xyz, rotation=torch.tensor([[1., 0., 0., 0.]] * len(xyz)),
        logscale=ids[:, None].expand(-1, 3).clone() * .03,
        opacity=torch.logit(torch.tensor([[.005]] + [[.7]] * (len(xyz) - 1))),
        sh_dc=ids[:, None, None].expand(-1, 1, 3).clone() * .1,
        sh_rest=ids[:, None, None].expand(-1, 3, 3).clone() * .01)


def motion_model(raw):
    cfg = RigidTreeConfig()
    cfg.spatial_gate.max_depth = 2
    cfg.spatial_gate.min_region_size = 2
    cfg.spatial_gate.training_schedule = 'joint'
    cfg.spatial_gate.start_step = cfg.spatial_gate.gate_start_step = 0
    cfg.spatial_gate.initial_logit = -2.
    cfg.validate()
    motion = SpatialGatedMotion(cfg)
    motion.setup(raw['xyz'], raw['rotation'], 2., frame_index=1)
    return motion


def export_frame(directory, motion, raw, sources):
    xyz, rotation = motion.materialized_pose()
    base = {**raw, 'xyz': xyz, 'rotation': rotation}
    newborn = {name: value[sources].clone() for name, value in base.items()}
    if len(sources):
        newborn['xyz'][:, 0] += .003
        newborn['opacity'][:] = torch.logit(torch.tensor(.8))
        if len(sources) > 1:
            newborn['opacity'][-1] = torch.logit(torch.tensor(.001))
        newborn['sh_dc'] += .01
    view = (QuantizedNewGaussianView(Rows(newborn), BirthConfig(), 2., base, sources)
            if len(sources) else None)
    expected, _ = materialize_quantized_gaussians(base, view)
    motion.register_materialized_layout(expected['xyz'], expected['rotation'], sources)
    frame = motion.frame_index
    motion.export_motion(directory / f'motion_{frame:03d}.bin')
    expected['xyz'] = motion.codec_previous_state.xyz
    expected['rotation'] = motion.codec_previous_state.rotation
    if len(sources):
        export_birth_attributes(directory / f'birth_{frame:03d}.bin',
            {name: expected[name][-len(sources):] for name in RAW_FIELDS[2:]},
            len(raw['xyz']),
            symbols={name: view.symbols[name] for name in RAW_FIELDS[2:]},
            steps={name: view.steps[name] for name in RAW_FIELDS[2:]},
            predictor={name: view.predictor[name] for name in RAW_FIELDS[2:]},
            source_rows=sources)
    return expected


class GaussianReplacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_replacement_retires_weak_old_and_new_rows_without_reordering(self):
        raw = reference_raw()
        model = Rows(raw)
        model._opacity[-1] = torch.logit(torch.tensor(.001))
        rows, retired, report = replace_low_opacity_gaussians(model, 2)
        torch.testing.assert_close(rows, torch.arange(1, 7))
        self.assertEqual(report['retired_older_count'], 1)
        self.assertEqual(report['retired_previous_birth_count'], 1)
        torch.testing.assert_close(retired['rows'], torch.tensor([0, 7]))
        torch.testing.assert_close(model._features_dc, raw['sh_dc'][rows])

    def test_empty_births_and_invalid_budgets_do_not_delete_points(self):
        model = Rows(reference_raw())
        for bad in (-1, 8, 9, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace_low_opacity_gaussians(model, bad)
        rows, retired, report = replace_low_opacity_gaussians(model, 0)
        self.assertEqual(model.prune_calls, 0)
        self.assertIsNone(retired)
        self.assertEqual(report['retired_count'], 0)
        torch.testing.assert_close(rows, torch.arange(8))
        self.assertFalse(BirthConfig().replace_low_opacity)
        with self.assertRaisesRegex(ValueError, 'replace_low_opacity'):
            BirthConfig(replace_low_opacity=1).validate(100)

    def test_opacity_ties_have_deterministic_retirement_order(self):
        model = Rows(reference_raw())
        model._opacity.fill_(0.)
        rows, retired, _ = replace_low_opacity_gaussians(model, 2)
        torch.testing.assert_close(retired['rows'], torch.tensor([0, 1]))
        torch.testing.assert_close(rows, torch.arange(2, 8))

    def test_cross_frame_births_deletions_replay_and_midframe_bootstrap(self):
        # More births than deletions must still retire old appearance rows;
        # inferring birth count from the net change would silently be wrong.
        for new_count in (0, 1, 3):
            with self.subTest(new_count=new_count), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                raw = reference_raw()
                motion = motion_model(raw)
                initial = copy.deepcopy(motion.frame_reference_state)
                with torch.no_grad():
                    motion.gs_translation[:, 0] = torch.arange(8) * .002
                    motion.gs_rotation[:, 1] = torch.arange(8) * .001
                first = export_frame(directory, motion, raw, torch.tensor([0, 6]))
                prior, decoded_first = decode_frame(directory, 1, initial, raw)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(decoded_first[name], first[name], rtol=0, atol=0)
                full = tuple(value.clone() for value in motion._full_motion_symbols())
                model = Rows(first)
                rows, retired, report = replace_low_opacity_gaussians(model, 2)
                self.assertEqual(report['retired_older_count'], 1)
                self.assertEqual(report['retired_previous_birth_count'], 1)
                kept = model_raw(model)
                motion.setup(kept['xyz'], kept['rotation'], 2., frame_index=2, previous_rows=rows)
                for actual, previous in zip(motion._full_motion_symbols(), full):
                    expected = torch.cat((previous, previous[torch.tensor([0, 6])]))[rows]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                restored = SpatialGatedMotion(copy.deepcopy(motion.config))
                restored.restore(copy.deepcopy(motion.capture()))
                self.assertEqual(restored._survivor_mapping_bytes, motion._survivor_mapping_bytes)
                rebuilt = restore_retired_reference(kept, prior, rows, retired)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(rebuilt[name], first[name], rtol=0, atol=0)
                bootstrap_module = SimpleNamespace(gaussians=model, motion=restored,
                    frame_finalized=False, retired_reference=retired,
                    initial_reference=dict(original_count=8, retained_count=8,
                                           removed_count=0, threshold=0.))
                worker = SimpleNamespace(output_dir=directory / 'resume', module=bootstrap_module,
                    datamodule=SimpleNamespace(cameras_extent=2.))
                Trainer._write_reference(worker, frame_index=1, state=prior)
                state_from_bootstrap, raw_from_bootstrap = replay(
                    worker.decoder_reference_path.parent, worker.decoder_reference_path, 1)
                self.assertEqual(state_from_bootstrap.digest(), prior.digest())
                for name in RAW_FIELDS:
                    torch.testing.assert_close(raw_from_bootstrap[name], first[name], rtol=0, atol=0)
                sources = torch.arange(new_count, dtype=torch.long)
                second = export_frame(directory, restored, kept, sources)
                for decode in (decode_frame, _decode_frame):
                    current, decoded_second = decode(directory, 2, prior, first)
                    torch.testing.assert_close(current.survivor_rows, rows)
                    self.assertEqual(len(current.xyz), 8 + new_count)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(decoded_second[name], second[name], rtol=0, atol=0)
                if new_count:
                    model = Rows(second)
                    rows, _, _ = replace_low_opacity_gaussians(model, new_count)
                    third_base = model_raw(model)
                    restored.setup(third_base['xyz'], third_base['rotation'], 2.,
                                   frame_index=3, previous_rows=rows)
                    third = export_frame(directory, restored, third_base,
                                         torch.empty(0, dtype=torch.long))
                    last, decoded_third = decode_frame(directory, 3, current, second)
                    self.assertEqual(len(last.xyz), 8)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(decoded_third[name], third[name], rtol=0, atol=0)


# ----- test_sparse_updates -----
class SparseUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_hard_block_values_and_zero_bin_has_no_sparse_gradient(self):
        value = torch.tensor([[.02, 0., 0.], [.07, -.08, 0.]], requires_grad=True)
        support = sparse_block_support(value, .1, differentiable=True)
        torch.testing.assert_close(support, torch.tensor([0., 1.]), rtol=0, atol=0)
        support.sum().backward()
        self.assertEqual(float(value.grad[0, 0]), 0.)
        self.assertGreater(float(value.grad[1, 0]), 0.)
        self.assertLess(float(value.grad[1, 1]), 0.)
        self.assertFalse(sparse_block_support(value.detach(), .1).requires_grad)

    def test_rate_loss_reaches_both_motion_fields_and_keeps_discrete_value(self):
        # Multiple GS per region make sharing cheaper than separate residuals.
        # One identical GS per region has a valid symmetric zero derivative.
        raw = {name: value.repeat_interleave(2, dim=0) for name, value in reference_raw().items()}
        motion = motion_model(raw)
        motion.config.differentiable_rate_support = True
        with torch.no_grad():
            motion.region_translation[1:, 0] = .6 * motion.steps[0]
            motion.region_rotation[1:, 1] = .6 * motion.steps[3]
            motion.gs_translation[:, 0] = .7 * motion.gs_steps[0]
            motion.gs_rotation[:, 1] = .8 * motion.gs_steps[3]
        loss = motion.loss_terms(5, False)['rate_loss']
        parameters = [motion.region_translation, motion.region_rotation,
                      motion.gs_translation, motion.gs_rotation]
        gradients = torch.autograd.grad(loss, parameters)
        for gradient in gradients:
            self.assertGreater(float(gradient.abs().sum()), 0.)
        motion.config.differentiable_rate_support = False
        self.assertEqual(float(loss), float(motion.loss_terms(5, False)['rate_loss']))

    def test_zero_opacity_symbols_keep_bits_and_image_gradient(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.1))
        reference = torch.tensor([[-0.], [1.], [-2.]])
        update.setup(reference)
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.02], [.13], [-.08]])
        logits = update()
        self.assertEqual(int(logits.detach().view(torch.int32)[0]),
                         int(reference.view(torch.int32)[0]))
        logits.sigmoid().sum().backward()
        self.assertTrue(bool(update.delta.grad.gt(0).all()))
        self.assertTrue(torch.isfinite(update.delta.grad).all())
        torch.testing.assert_close(update.canonical_logits(), logits.detach(), rtol=0, atol=0)

    def test_opacity_rate_only_pushes_nonzero_quantized_updates(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.05))
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.01], [.03], [-.1]])
        terms = update.rate_terms(1.)
        self.assertEqual(int(terms['active_rows']), 2)
        terms['loss'].backward()
        self.assertEqual(float(update.delta.grad[0]), 0.)
        self.assertGreater(float(update.delta.grad[1]), 0.)
        self.assertLess(float(update.delta.grad[2]), 0.)

    def test_proximal_opacity_adam_keeps_only_image_second_moments(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, proximal_regularization=True,
                                                            l1_weight=.005))
        update.setup(torch.tensor([[0.], [1.], [-1.]]))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.03], [-.1], [.2]])
        image_loss = (update().sigmoid() - .6).square().mean()
        gradient, = torch.autograd.grad(image_loss, [update.delta], retain_graph=True)
        rate = update.rate_terms(.01)['loss']
        self.assertFalse(rate.requires_grad)
        (image_loss + rate).backward()
        update.optimizer_step(0., proximal=False)
        torch.testing.assert_close(update.optimizer.state[update.delta]['exp_avg_sq'],
            gradient.square() * .001, atol=1e-15, rtol=1e-6)

    def test_proximal_opacity_clears_weak_latents_and_retains_large_update(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, proximal_regularization=True))
        update.setup(torch.zeros(3,1))
        with torch.no_grad():
            update.delta[:] = torch.tensor([[.01],[.03],[2.]])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(update.delta), exp_avg_sq=torch.full_like(update.delta, 1e-11))
        update.delta.grad = torch.zeros_like(update.delta)
        update.optimizer_step(1e-6, proximal=True)
        self.assertEqual(float(update.delta[0]),0.)
        self.assertEqual(float(update.delta[1]),0.)
        self.assertEqual(float(update.delta[2]),2.)

    def test_continuous_opacity_penalty_cannot_create_opposite_updates(self):
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha',
            proximal_regularization=True, l1_weight=.005, learning_rate=.01))
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:, 0] = torch.tensor([.1, -.1, 1e-5])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(update.delta), exp_avg_sq=torch.full_like(update.delta, 1e-8))
        update.delta.grad = torch.zeros_like(update.delta)
        update.optimizer_step(0., lr_factor=.2)
        self.assertGreater(float(update.delta[0]), 0.)
        self.assertLess(float(update.delta[0]), .1)
        self.assertLess(float(update.delta[1]), 0.)
        self.assertGreater(float(update.delta[1]), -.1)
        self.assertEqual(float(update.delta[2]), 0.)
        self.assertAlmostEqual(update.optimizer.param_groups[0]['lr'], .002)

    def test_legacy_opacity_config_defaults_and_proximal_resume(self):
        old = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True))
        old.setup(torch.zeros(3,1))
        saved = copy.deepcopy(old.capture())
        saved['config'].pop('proximal_regularization')
        restored = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True))
        restored.restore(saved,torch.device('cpu'))
        torch.testing.assert_close(old(),restored(),atol=0,rtol=0)
        cfg = OpacityUpdateConfig(enabled=True,proximal_regularization=True)
        one = QuantizedOpacityUpdate(cfg); one.setup(torch.zeros(3,1))
        one().sigmoid().sum().backward(); one.optimizer_step(0.)
        two = QuantizedOpacityUpdate(copy.deepcopy(cfg));two.restore(copy.deepcopy(one.capture()),torch.device('cpu'))
        for current in (one,two):
            current().sigmoid().sum().backward();current.optimizer_step(1e-8,proximal=True)
        torch.testing.assert_close(one.delta,two.delta,atol=0,rtol=0)

    def test_opacity_refit_keeps_zero_rows_frozen_and_removes_amplitude_bias(self):
        cfg = OpacityUpdateConfig(enabled=True, mode='alpha', step=.01,
            proximal_regularization=True, l1_weight=100., amplitude_refit_steps=20)
        update = QuantizedOpacityUpdate(cfg)
        update.setup(torch.zeros(3, 1))
        with torch.no_grad():
            update.delta[:, 0] = torch.tensor([.004, .1, -.1])
        update.optimizer.state[update.delta] = dict(step=torch.tensor(1.),
            exp_avg=torch.ones_like(update.delta), exp_avg_sq=torch.ones_like(update.delta))
        update.prepare_step(80, 100)
        self.assertIsNone(update.refit_support)
        before = update.symbols().ne(0)
        update.prepare_step(81, 100)
        self.assertEqual(float(update.delta[0]), 0.)
        self.assertNotIn(update.delta, update.optimizer.state)
        self.assertFalse(update.rate_terms(100.)['loss'].requires_grad)
        for _ in range(3):
            # Huge rate/L1 penalties would erase all updates without refitting.
            (-update.alphas().sum()).backward()
            update.optimizer_step(100., proximal=True)
            self.assertEqual(float(update.delta[0]), 0.)
            self.assertFalse(bool((update.symbols().ne(0) & ~before).any()))
        self.assertGreater(float(update.delta[1]), .1)

    def test_opacity_refit_resume_and_survivor_selection_are_exact(self):
        cfg = OpacityUpdateConfig(enabled=True, mode='alpha', step=.01, amplitude_refit_steps=20)
        one = QuantizedOpacityUpdate(cfg)
        one.setup(torch.zeros(3, 1))
        with torch.no_grad():
            one.delta[:, 0] = torch.tensor([0., .1, -.1])
        one.prepare_step(81, 100)
        one.alphas().sum().backward(); one.optimizer_step(0.)
        two = QuantizedOpacityUpdate(copy.deepcopy(cfg))
        two.restore(copy.deepcopy(one.capture()), torch.device('cpu'))
        for update in (one, two):
            update.prepare_step(82, 100)
            update.alphas().sum().backward(); update.optimizer_step(1., proximal=True)
            update.commit(torch.tensor([2, 0]))
        torch.testing.assert_close(one.delta, two.delta, atol=0, rtol=0)
        torch.testing.assert_close(one.canonical_logits(), two.canonical_logits(), atol=0, rtol=0)
        self.assertEqual(float(two.delta[0]), 0.)
        with tempfile.TemporaryDirectory() as directory:
            packet = two.export(Path(directory)/'opacity_002.bin', 2)
            decoded = decode_opacity_delta(Path(packet['path']).read_bytes(), two.reference[two.rows], frame_index=2)
            torch.testing.assert_close(decoded.logits, two.canonical_logits(), atol=0, rtol=0)

    def test_opacity_refit_validation_and_legacy_default(self):
        for value in (True, -1, .5):
            with self.assertRaises(ValueError):
                OpacityUpdateConfig(enabled=True, amplitude_refit_steps=value).validate()
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True))
        update.setup(torch.zeros(2, 1))
        saved = copy.deepcopy(update.capture())
        saved.pop('refit_support')
        saved['config'].pop('amplitude_refit_steps')
        update.restore(saved, torch.device('cpu'))
        update.prepare_step(100, 100)
        self.assertIsNone(update.refit_support)
        bad = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, amplitude_refit_steps=100))
        bad.setup(torch.zeros(2, 1))
        with self.assertRaises(ValueError):
            bad.prepare_step(1, 100)

    def test_birth_opacity_joint_fit_retains_previous_frame_reference_and_resumes(self):
        cfg = OpacityUpdateConfig(enabled=True, mode='alpha', step=.01,
            joint_with_birth=True, proximal_regularization=True)
        one = QuantizedOpacityUpdate(cfg)
        one.setup(torch.tensor([[1.], [-1.], [0.]]))
        one.commit(torch.tensor([2, 0]))
        self.assertTrue(one.delta.requires_grad)
        reference = one.reference.clone()
        one.alphas().sum().backward(); one.optimizer_step(0.)
        self.assertEqual(float(one.delta[1]), 0.)  # Deleted before birth.
        two = QuantizedOpacityUpdate(copy.deepcopy(cfg))
        two.restore(copy.deepcopy(one.capture()), torch.device('cpu'))
        for current in (one, two):
            self.assertTrue(current.delta.requires_grad)
            current.alphas().sum().backward(); current.optimizer_step(0.)
            current.finish()
        torch.testing.assert_close(one.delta, two.delta, rtol=0, atol=0)
        torch.testing.assert_close(two.reference, reference, rtol=0, atol=0)
        finished = QuantizedOpacityUpdate(copy.deepcopy(cfg))
        finished.restore(copy.deepcopy(two.capture()), torch.device('cpu'))
        self.assertFalse(finished.delta.requires_grad)
        with tempfile.TemporaryDirectory() as directory:
            info = two.export(Path(directory)/'opacity_002.bin', 2)
            decoded = decode_opacity_delta(Path(info['path']).read_bytes(), reference[two.rows], frame_index=2)
            torch.testing.assert_close(decoded.logits, two.canonical_logits(), atol=0, rtol=0)

    def test_alpha_mode_has_unsaturated_image_gradient_and_zero_bits(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01,learning_rate=.01)
        update=QuantizedOpacityUpdate(cfg)
        update.setup(torch.tensor([[-0.],[7.],[-7.]]))
        output=update.alphas()
        output.sum().backward()
        torch.testing.assert_close(update.delta.grad,torch.ones(3,1),atol=0,rtol=0)
        self.assertEqual(int(update().detach().view(torch.int32)[0]),int(update.reference.view(torch.int32)[0]))

    def test_alpha_mode_render_grid_and_exact_canonical_packet(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01)
        update=QuantizedOpacityUpdate(cfg)
        update.setup(torch.tensor([[0.],[5.],[-5.],[1.]]))
        with torch.no_grad():update.delta[:,0]=torch.tensor([.024,-.995,.025,0.])
        expected=(update.alpha_reference+update.symbols()*update.step).clamp(0,1)
        torch.testing.assert_close(update.alphas(),expected,atol=1e-7,rtol=0)
        packet=encode_opacity_delta(update.reference,update.symbols(),frame_index=2,step=cfg.step,mode='alpha')
        decoded=decode_opacity_delta(packet,update.reference)
        self.assertEqual(decoded.mode,'alpha_canonical')
        torch.testing.assert_close(decoded.logits,update.canonical_logits(),atol=0,rtol=0)
        self.assertLess(float(decoded.logits[1].sigmoid()),.01)

    def test_alpha_mode_bounds_and_checkpoint_state(self):
        cfg=OpacityUpdateConfig(enabled=True,mode='alpha',step=.01)
        update=QuantizedOpacityUpdate(cfg);update.setup(torch.tensor([[7.],[-7.],[0.]]))
        with torch.no_grad():update.delta[:,0]=torch.tensor([20.,-20.,.11])
        update.delta.grad=torch.zeros_like(update.delta)
        update.optimizer_step(0.)
        self.assertTrue(bool(update.delta.ge(-update.alpha_reference).all()))
        self.assertTrue(bool(update.delta.le(1-update.alpha_reference).all()))
        restored=QuantizedOpacityUpdate(copy.deepcopy(cfg));restored.restore(copy.deepcopy(update.capture()),torch.device('cpu'))
        torch.testing.assert_close(update.alphas(),restored.alphas(),atol=0,rtol=0)
        torch.testing.assert_close(update.canonical_logits(),restored.canonical_logits(),atol=0,rtol=0)

    def test_alpha_packet_is_invariant_to_survivor_selection_and_simd_tails(self):
        generator = torch.Generator().manual_seed(295)
        reference = torch.randn(4099, 1, generator=generator) * 4
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='alpha', step=.01))
        update.setup(reference)
        with torch.no_grad():
            update.delta.copy_(torch.randn(4099, 1, generator=generator) * .08)
        full = update.canonical_logits(full=True)
        for count in (1, 7, 31, 32, 101, 3001, 4099):
            rows = torch.randperm(4099, generator=generator)[:count]
            selected_ref = reference[rows]
            packet = encode_opacity_delta(selected_ref, update.symbols()[rows],
                                           frame_index=2, step=.01, mode='alpha')
            if packet is None:
                decoded = selected_ref
            else:
                decoded = decode_opacity_delta(packet, selected_ref).logits
            torch.testing.assert_close(decoded.view(torch.int32),
                                       full[rows].view(torch.int32), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_gpu_opacity_materialization_matches_cpu_packet_bits(self):
        generator = torch.Generator().manual_seed(83)
        reference = torch.randn(1024, 1, generator=generator)
        reference[0] = -0.
        update = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, step=.05))
        update.setup(reference.cuda())
        with torch.no_grad():
            update.delta.copy_(torch.randn(1024, 1, generator=generator).cuda() * .2)
            update.delta[0] = .01
        rendered = update().detach().cpu()
        torch.testing.assert_close(rendered.view(torch.int32),
                                   update.canonical_logits().view(torch.int32), rtol=0, atol=0)

    def test_gap_packet_exact_roundtrip_checksums_and_invalid_symbols(self):
        reference = torch.linspace(-4., 4., 300)[:, None]
        symbols = torch.zeros_like(reference)
        symbols[[0, 150, 299], 0] = torch.tensor([-200., 1., 32000.])
        packet = encode_opacity_delta(reference, symbols, frame_index=2, step=.05)
        decoded = decode_opacity_delta(packet, reference, frame_index=2)
        expected = reference.clone()
        expected[decoded.rows, 0] += decoded.symbols.float() * decoded.step
        torch.testing.assert_close(decoded.logits, expected, rtol=0, atol=0)
        damaged = bytearray(packet)
        damaged[-1] ^= 1
        with self.assertRaises(ValueError):
            decode_opacity_delta(damaged, reference, frame_index=2)
        self.assertIsNone(encode_opacity_delta(reference, torch.zeros_like(symbols), frame_index=2, step=.05))
        for bad in [.5, 32768., float('nan')]:
            invalid = symbols.clone()
            invalid[0] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                encode_opacity_delta(reference, invalid, frame_index=2, step=.05)

    def test_current_opacity_retires_previously_opaque_point_and_replays_births(self):
        raw = reference_raw()
        raw['opacity'][0] = torch.logit(torch.tensor(.99))
        motion = motion_model(raw)
        initial = copy.deepcopy(motion.frame_reference_state)
        opacity_cfg = OpacityUpdateConfig(enabled=True, step=.05)
        update = QuantizedOpacityUpdate(opacity_cfg)
        update.setup(raw['opacity'])
        with torch.no_grad():
            update.delta[0] = -12.
            update.delta[1] = .1
        restored = QuantizedOpacityUpdate(copy.deepcopy(opacity_cfg))
        restored.restore(copy.deepcopy(update.capture()), torch.device('cpu'))
        torch.testing.assert_close(restored(), update(), rtol=0, atol=0)
        holder = SimpleNamespace(opacity_update=restored, gaussians=Rows(raw), motion=motion,
            config=SimpleNamespace(opacity_update=opacity_cfg), device=torch.device('cpu'),
            birth_control=dict(previous_birth_count=3), retired_reference=None)
        Module._commit_opacity_and_retire(holder)
        self.assertEqual(holder.birth_control['retirement']['retired_count'], 1)
        self.assertEqual(len(holder.gaussians._xyz), 7)
        torch.testing.assert_close(motion.previous_rows, torch.arange(1, 8))
        loaded_motion = SpatialGatedMotion(copy.deepcopy(motion.config))
        loaded_motion.restore(copy.deepcopy(motion.capture()))
        self.assertEqual(len(loaded_motion.gs_translation), 7)
        holder.frame_finalized = False
        holder.initial_reference = dict(original_count=8, retained_count=8,
                                        removed_count=0, threshold=0.)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            worker = SimpleNamespace(output_dir=directory/'resume', module=holder,
                datamodule=SimpleNamespace(cameras_extent=2.))
            Trainer._write_reference(worker, 0, initial)
            with np.load(worker.decoder_reference_path) as reference:
                for name in RAW_FIELDS:
                    torch.testing.assert_close(torch.from_numpy(reference[name]), raw[name], rtol=0, atol=0)
            current_raw = model_raw(holder.gaussians)
            expected = export_frame(directory, loaded_motion, current_raw, torch.tensor([0]))
            info = restored.export(directory/'opacity_001.bin', 1)
            self.assertEqual(info['rows'], 1)  # Deleted point's large alpha delta is not transmitted.
            manifest = dict(opacity_update='opacity_001.bin', opacity_update_timing='before_births')
            (directory/'frame_001.json').write_text(json.dumps(manifest))
            decoded, replayed = decode_frame(directory, 1, initial, raw)
            self.assertEqual(len(decoded.xyz), 8)
            for name in RAW_FIELDS:
                torch.testing.assert_close(replayed[name], expected[name], rtol=0, atol=0)
            manifest['opacity_update_timing'] = 'after_births'
            (directory/'frame_001.json').write_text(json.dumps(manifest))
            _, wrong = decode_frame(directory, 1, initial, raw)
            self.assertFalse(torch.equal(wrong['opacity'], expected['opacity']))

    def test_alpha_override_uses_quantized_values_with_gradient(self):
        model = GaussianModel(1)
        model._opacity = torch.nn.Parameter(torch.tensor([[2.]]), requires_grad=False)
        override = torch.tensor([[-1.]], requires_grad=True)
        model.rigid_opacity = override
        model.get_opacity.sum().backward()
        self.assertGreater(float(override.grad), 0.)
        self.assertIsNone(model._opacity.grad)
        model.rigid_opacity = None
        torch.testing.assert_close(model.get_opacity, model._opacity.sigmoid())

    def test_materialized_opacity_invalidates_previous_render_cache(self):
        raw = reference_raw()
        raw['opacity'][0] = 0.
        model = GaussianModel(1)
        from pipeline.model.birth_quantization import RAW_ATTRIBUTES
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(model, attr, torch.nn.Parameter(raw[name].clone(), requires_grad=False))
        before = model.get_opacity.clone()
        config = OpacityUpdateConfig(enabled=True)
        update = QuantizedOpacityUpdate(config)
        update.setup(raw['opacity'])
        with torch.no_grad():
            update.delta[0] = .1
        holder = SimpleNamespace(opacity_update=update, gaussians=model,
            motion=motion_model(raw), config=SimpleNamespace(opacity_update=config),
            device=torch.device('cpu'), birth_control={}, retired_reference=None)
        Module._commit_opacity_and_retire(holder)
        self.assertFalse(torch.equal(model.get_opacity, before))
        torch.testing.assert_close(model.get_opacity, update.canonical_logits().sigmoid(), rtol=0, atol=0)


# ----- test_rigid_decoder_state_restore -----
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DecodedState__rigid_decoder_state_restore = importlib.import_module(
    'pipeline.model.rigid_tree_codec').DecodedState


def _state(frame):
    return DecodedState__rigid_decoder_state_restore(
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
                restored = DecodedState__rigid_decoder_state_restore.restore(original.capture())
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
            restored = DecodedState__rigid_decoder_state_restore.restore(packet)
            self.assertEqual(restored.frame_index,
                             49 if name == 'codec_previous_state' else 48)
            for field in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources', 'survivor_rows'):
                value = getattr(restored, field)
                if value is not None:
                    self.assertEqual(value.device.type, 'cpu')
                    self.assertEqual(value.dtype, packet[field].dtype)
                    torch.testing.assert_close(value, packet[field].cpu(),
                                               rtol=0, atol=0)


# ----- test_legacy_opacity_replay -----
LEGACY_CODEC = importlib.import_module('pipeline.model.legacy_opacity_codec')
# Frozen OPD1/2/3 packets and their exact float32 logits from the prior codec.
FIXTURES = [('logit', '4f5044310200000004000000030000000ad7233c0900000094861248bea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '0ad7233c0000803f295c9f407b147ec0'), ('alpha', '4f5044320200000004000000030000000ad7233c09000000b5712892bea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '98dc233d0000803fdf296640eb353fc0'), ('saturated_alpha', '4f5044330200000004000000030000000ad7233c09000000aadcc1dbbea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '0ad7233c0000803f0efc8d407b147ec0')]


class HistoricalOpacityReplayTests(unittest.TestCase):
    def reference(self):
        return torch.tensor([[0.], [1.], [5.], [-4.]], dtype=torch.float32)

    def test_all_historical_coordinate_modes_reconstruct_exact_logits(self):
        for mode, encoded, logits in FIXTURES:
            with self.subTest(mode=mode):
                actual = LEGACY_CODEC.decode_opacity_delta(bytes.fromhex(encoded), self.reference(), frame_index=2)
                expected = torch.from_numpy(np.frombuffer(bytes.fromhex(logits), dtype='<f4').copy()).reshape(-1, 1)
                torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
                self.assertEqual(actual.mode, mode)

    def test_corrupt_packets_are_rejected(self):
        for mode, encoded, _ in FIXTURES:
            with self.subTest(mode=mode):
                damaged = bytearray.fromhex(encoded)
                damaged[-1] ^= 1
                with self.assertRaises(ValueError):
                    LEGACY_CODEC.decode_opacity_delta(bytes(damaged), self.reference(), frame_index=2)

    def test_reference_and_frame_mismatches_are_rejected(self):
        encoded = bytes.fromhex(FIXTURES[0][1])
        with self.assertRaises(ValueError):
            LEGACY_CODEC.decode_opacity_delta(encoded, self.reference() + .01, frame_index=2)
        with self.assertRaises(ValueError):
            LEGACY_CODEC.decode_opacity_delta(encoded, self.reference(), frame_index=3)


# ----- test_rigid_sequence_benchmark -----
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
