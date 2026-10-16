"""Focused lossless motion packet and actual-byte regressions."""

import hashlib
import importlib
import lzma
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib
from pathlib import Path
from types import ModuleType

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

CODEC = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
RIGID = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')


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
        from test_spatial_gated_motion import SpatialGatedMotionTests
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
        from test_spatial_gated_motion import SpatialGatedMotionTests
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



if __name__ == '__main__':
    unittest.main()
