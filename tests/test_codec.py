"""检查逐帧码流的往返和播放一致性。

覆盖运动、透明度及全属性的编码解码，检查码流损坏、边界值和参考状态校验，
以及解码后多帧状态与原始属性的一致性。属性优化和训练调度由各自测试文件负责。
"""

from pipeline.model import frame_stream_format as _stream_fmt, frame_stream_encode as _stream_enc, frame_stream_decode as _stream_dec

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hashlib
import json
import lzma
import struct
import tempfile
import unittest
import zlib
from unittest.mock import patch

import numpy as np
import torch

from pipeline.model.frame_stream_format import RAW_FIELDS
from pipeline.model.frame_stream_format import (OPACITY_HEADER as HEADER, OPACITY_MAGIC as MAGIC, opacity_reference_hash)
from pipeline.model.frame_stream_decode import (decode_opacity_delta)
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.frame_stream_decode import (decode_motion)
from pipeline.model import residual_gs_attributes_update as fua
from pipeline.model.frame_stream_format import FIELDS, PLANAR_MAGIC, DecodedState, _reference_hash
from pipeline.model.frame_stream_encode import _encode_body, encode_motion
from pipeline.model.frame_stream_decode import decode_frame, decode_full_attribute_delta
from pipeline.model.residual_gs_attributes_update import (
    ResidualAppearanceAttributes, FullUpdateConfig, encode_full_attribute_delta,
    OpacityUpdateConfig,
    QuantizedOpacityUpdate,
    encode_opacity_delta,
)
from pipeline.model.spatial_motion_training import SpatialMotionTraining
from tests import test_regional_attributes_update as spatial_tests


# ----- test_rigid_tree_compression -----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FULL_ATTRIBUTE_STEPS = (.005, .001, .001)


def packet_fixture(count=256):
    xyz = torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 3).clone()
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1
    owner = torch.ones(count, dtype=torch.long)
    parents = torch.tensor([-1, 0], dtype=torch.long)
    prior = _stream_fmt.DecodedState(xyz, rotation, owner, parents, 0)
    region = torch.zeros(2, 3, dtype=torch.int16)
    gs_translation = torch.full((count, 3), 42, dtype=torch.int16)
    gs_rotation = torch.full((count, 3), 7, dtype=torch.int16)
    args = (prior, 1, 1., torch.arange(count), owner, parents,
            region, region, gs_translation, gs_rotation,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3),
            torch.empty(0, 4), .01, .01)
    return prior, args


class MotionPacketCompressionTests(unittest.TestCase):


    def test_compressed_packet_rejects_mutation_truncation_trailing_and_bomb(self):
        prior, args = packet_fixture()
        packet = _stream_enc.encode_motion(*args, compression='zlib')
        damaged = bytearray(packet)
        damaged[len(damaged) // 2] ^= 1
        for candidate in (bytes(damaged), packet[:-1], packet + b'X',
                          _stream_fmt.DELTA_COMPRESSED_MAGIC + _stream_fmt._uvar(
                              _stream_fmt.MAX_RAW_PACKET_BYTES + 1) + packet[5:]):
            try:
                _stream_dec.decode_motion(candidate, prior)
            except ValueError:
                pass
            else:
                raise AssertionError('Damaged compressed stream was accepted')


    def test_lzma_packet_roundtrips_and_rejects_damage_trailers_and_expansion(self):
        prior, args = packet_fixture()
        raw = _stream_enc.encode_motion(*args)
        packet = (_stream_fmt.DELTA_LZMA_COMPRESSED_MAGIC + _stream_fmt._uvar(len(raw)) +
                  lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9))
        assert _stream_dec.uncompressed_motion_packet(packet) == raw
        assert _stream_dec.decode_motion(packet, prior).digest() == (
            _stream_dec.decode_motion(raw, prior).digest())
        assert _stream_dec.motion_packet_info(packet)['stored_bytes'] == len(packet)
        for bad in (packet[:-1], packet + b'X',
                    packet[:5] + _stream_fmt._uvar(len(raw) - 1) + packet[6:],
                    _stream_fmt.DELTA_LZMA_COMPRESSED_MAGIC + _stream_fmt._uvar(
                        _stream_fmt.MAX_RAW_PACKET_BYTES + 1) + packet[6:]):
            try:
                _stream_dec.uncompressed_motion_packet(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('Damaged LZMA stream was accepted')


# ----- test_rigid_lossless_codecs -----
sys.path.insert(0, str(ROOT))


class MotionEnvelopeTests(unittest.TestCase):
    @staticmethod
    def raw_packet(body):
        raw = _stream_fmt.DUAL_STEP_MAGIC + body
        return raw + hashlib.sha256(raw).digest()[:_stream_fmt.CHECKSUM_BYTES]

    def test_auto_compares_actual_packet_lengths(self):
        for body in (b'A' * 10000, bytes(range(256)) * 20):
            raw = self.raw_packet(body)
            zlib_packet = _stream_enc.compress_motion_packet(raw, mode='zlib')
            lzma_packet = (_stream_fmt.DUAL_STEP_LZMA_COMPRESSED_MAGIC +
                           _stream_fmt._uvar(len(raw)) +
                           lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9))
            chosen = _stream_enc.compress_motion_packet(raw, mode='auto')
            self.assertEqual(len(chosen), min(map(len, (raw, zlib_packet, lzma_packet))))
            self.assertEqual(_stream_dec.uncompressed_motion_packet(chosen), raw)
            self.assertEqual(_stream_dec.uncompressed_motion_packet(lzma_packet), raw)
            self.assertEqual(_stream_dec.motion_packet_info(lzma_packet)['stored_bytes'],
                             len(lzma_packet))

    def test_lzma_rejects_damage_trailer_and_expansion(self):
        raw = self.raw_packet(b'A' * 10000)
        prefix = _stream_fmt.DUAL_STEP_LZMA_COMPRESSED_MAGIC
        lz = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9)
        good = prefix + _stream_fmt._uvar(len(raw)) + lz
        for bad in (good[:-1], good + b'X', prefix + _stream_fmt._uvar(len(raw) - 1) + lz,
                    prefix + _stream_fmt._uvar(_stream_fmt.MAX_RAW_PACKET_BYTES + 1) + lz):
            with self.assertRaises(ValueError):
                _stream_dec.uncompressed_motion_packet(bad)

    def test_delta_sparse_rows_reject_zero_gap_overflow_and_truncation(self):
        for payload in (
            _stream_fmt._uvar(1) + _stream_fmt._uvar(0) + b'\x01' + struct.pack('<3h', 1, 0, 0),
            _stream_fmt._uvar(1) + _stream_fmt._uvar(3) + b'\x01' + struct.pack('<3h', 1, 0, 0),
            _stream_fmt._uvar(1) + _stream_fmt._uvar(1) + b'\x01' + b'\x01\x00',
        ):
            with self.assertRaises(ValueError):
                _stream_dec._sparse_read(_stream_fmt._Reader(payload), 2, delta_rows=True)
        valid = (_stream_fmt._uvar(2) + _stream_fmt._uvar(1) + b'\x01' +
                 struct.pack('<3h', 1, 0, 0) + _stream_fmt._uvar(1) + b'\x02' +
                 struct.pack('<3h', 0, 1, 0))
        translation, rotation = _stream_dec._sparse_read(
            _stream_fmt._Reader(valid), 2, delta_rows=True)
        torch.testing.assert_close(translation[0], torch.tensor([1, 0, 0], dtype=torch.int16))
        torch.testing.assert_close(rotation[1], torch.tensor([0, 1, 0], dtype=torch.int16))


sys.path.insert(0, str(ROOT))


# ----- test_sparse_integer_codec -----
sys.path.insert(0, str(ROOT))


def wrapped(layout, raw_length, magics):
    prefix = _stream_fmt._integer_varint(raw_length) + _stream_fmt._integer_varint(len(layout))
    return [magics[0] + prefix + layout,
            magics[1] + prefix + zlib.compress(layout, 6),
            magics[2] + prefix + lzma.compress(layout, preset=9)]


def spatial_fixture():
    from tests.test_regional_attributes_update import SpatialMotionTrainingTests
    model = SpatialMotionTrainingTests().model(repetitions=32)
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
    sources = torch.empty(0, dtype=torch.long)
    args = (prior, 1, 1., rows, owner, prior.parents, rt, rr, gt, gr,
            sources, prior.xyz[rows[sources]] + .001, prior.rotation[rows[sources]], .001, .001)
    raw = _stream_enc.encode_motion(*args, gs_translation_step_extent=.001,
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
        body.extend(_stream_fmt._integer_varint(row - previous))
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

    def test_vectorized_varints_match_canonical_bytes_at_boundaries(self):
        import numpy as np
        values = np.r_[np.array([0, 1, 127, 128, 16383, 16384, 2097151, 2097152,
                                268435455, 268435456, 0xffffffff], dtype=np.uint64),
                       np.random.default_rng(29).integers(0, 2**32, 2048, dtype=np.uint64)]
        expected = b''.join(_stream_fmt._integer_varint(value) for value in values)
        self.assertEqual(_stream_enc.encode_unsigned_values(values), expected)
        self.assertEqual(_stream_enc.encode_unsigned_values([]), b'')

    def test_motion_all_envelopes_restore_exact_raw_and_state(self):
        prior, raw = spatial_fixture()
        layout = _stream_enc.encode_motion_layout(raw)
        expected = _stream_dec.decode_motion(raw, prior)
        for packet in wrapped(layout, len(raw), _stream_fmt.MOTION_MAGICS):
            with self.subTest(magic=packet[:4]):
                self.assertEqual(_stream_dec.uncompressed_motion_packet(packet), raw)
                actual = _stream_dec.decode_motion(packet, prior)
                self.assertEqual(actual.digest(), expected.digest())
                for name in ('xyz', 'rotation', 'owner', 'parents', 'survivor_rows', 'birth_sources'):
                    self.assertEqual(getattr(actual, name).numpy().tobytes(),
                                     getattr(expected, name).numpy().tobytes())
                self.assertEqual(_stream_dec.motion_packet_info(packet)['stored_bytes'], len(packet))
        chosen = _stream_enc.compress_motion_packet(raw)
        old = [raw, _stream_enc.compress_motion_packet(raw, 'zlib'),
               _stream_fmt.SPATIAL_LZMA_COMPRESSED_MAGIC + _stream_fmt._uvar(len(raw)) + lzma.compress(raw, preset=9)]
        self.assertLessEqual(len(chosen), min(map(len, old)))
        self.assertEqual(_stream_dec.uncompressed_motion_packet(chosen), raw)

    def test_opacity_auto_and_all_envelopes_keep_logits_reference_bits_and_symbols(self):
        reference, symbols, raw = opacity_fixture()
        layout = _stream_enc.encode_opacity_layout(raw)
        auto = encode_opacity_delta(reference, symbols, frame_index=1, step=.125)
        expected = decode_opacity_delta(auto, reference, frame_index=1)
        for packet in [auto, *wrapped(layout, len(raw), _stream_fmt.OPACITY_MAGICS)]:
            decoded = decode_opacity_delta(packet, reference, frame_index=1)
            self.assertEqual(decoded.logits.numpy().tobytes(), expected.logits.numpy().tobytes())
            self.assertEqual(decoded.symbols.numpy().tobytes(), expected.symbols.numpy().tobytes())
            self.assertEqual(int(decoded.logits.view(torch.int32)[0]),
                             int(expected.logits.view(torch.int32)[0]))
            with self.assertRaises(ValueError):
                decode_opacity_delta(packet, reference + .01, frame_index=1)
            with self.assertRaises(ValueError):
                decode_opacity_delta(packet, reference, frame_index=2)

    def test_planar_opacity_rejects_valid_layout_with_wrong_raw_checksum(self):
        reference, _, raw = opacity_fixture()
        header = list(HEADER.unpack(raw[:HEADER.size]))
        header[6] ^= 1
        damaged = HEADER.pack(*header) + raw[HEADER.size:]
        layout = _stream_enc.encode_opacity_layout(damaged)
        for packet in wrapped(layout, len(damaged), _stream_fmt.OPACITY_MAGICS):
            with self.subTest(magic=packet[:4]), self.assertRaisesRegex(ValueError, 'checksum'):
                decode_opacity_delta(packet, reference, frame_index=1)

    def test_motion_and_opacity_reject_damage_truncation_trailers_and_length_limits(self):
        prior, raw_motion = spatial_fixture()
        reference, _, raw_opacity = opacity_fixture()
        for raw, magics, transform, decode in [
                (raw_motion, _stream_fmt.MOTION_MAGICS, _stream_enc.encode_motion_layout,
                 lambda p: _stream_dec.decode_motion(p, prior)),
                (raw_opacity, _stream_fmt.OPACITY_MAGICS, _stream_enc.encode_opacity_layout,
                 lambda p: decode_opacity_delta(p, reference))]:
            layout = transform(raw)
            for packet in wrapped(layout, len(raw), magics):
                mutated = bytearray(packet)
                mutated[-1] ^= 1
                bad_length = magics[0] + _stream_fmt._integer_varint(0xffffffff) + _stream_fmt._integer_varint(len(layout)) + layout
                for bad in (packet[:-1], packet + b'X', bytes(mutated), bad_length):
                    with self.subTest(magic=packet[:4]), self.assertRaises(ValueError):
                        decode(bad)

    def test_planar_rejects_noncanonical_gaps_and_out_of_range_int16(self):
        for payload in (b'\x80\x00', _stream_fmt._integer_varint(65536)):
            with self.assertRaises(ValueError):
                _stream_fmt._read_values(_stream_fmt._IntegerReader(payload), 1, 1)
        for gap in (b'\0', _stream_fmt._integer_varint(4)):
            with self.assertRaises(ValueError):
                _stream_fmt._IntegerReader(gap).gaps(1, 3)


# ----- test_gaussian_replacement -----


def reference_raw(rest=3):
    xyz = torch.cartesian_prod(*[torch.tensor([-1., 1.])] * 3)
    ids = torch.arange(len(xyz), dtype=torch.float32)
    return dict(xyz=xyz, rotation=torch.tensor([[1., 0., 0., 0.]] * len(xyz)),
        logscale=ids[:, None].expand(-1, 3).clone() * .03,
        opacity=torch.logit(torch.tensor([[.005]] + [[.7]] * (len(xyz) - 1))),
        sh_dc=ids[:, None, None].expand(-1, 1, 3).clone() * .1,
        sh_rest=ids[:, None, None].expand(-1, rest, 3).clone() * .01)


def motion_model(raw):
    cfg = RigidTreeConfig()
    cfg.spatial_gate.max_depth = 2
    cfg.spatial_gate.min_region_size = 2
    cfg.spatial_gate.training_schedule = 'joint'
    cfg.spatial_gate.start_step = cfg.spatial_gate.gate_start_step = 0
    cfg.spatial_gate.initial_logit = -2.
    cfg.validate()
    motion = SpatialMotionTraining(cfg)
    motion.setup(raw['xyz'], raw['rotation'], 2., frame_index=1)
    return motion


# ----- test_sparse_updates -----
class OpacityStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


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


# ----- test_rigid_decoder_state_restore -----
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DecodedFrameConsistencyTests(unittest.TestCase):
    def test_regional_updates_keep_inputs_and_outputs_independent(self):
        nodes = 4
        steps = dict(logscale=.005, sh_dc=.001, sh_rest=.001, opacity=.01)
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for appearance, opacity in ((True, False), (False, True), (True, True)):
            symbols = dict(logscale=torch.zeros(nodes, 3, dtype=torch.int16),
                           sh_dc=torch.zeros(nodes, 1, 3, dtype=torch.int16),
                           sh_rest=torch.zeros(nodes, 3, 3, dtype=torch.int16),
                           opacity=torch.zeros(nodes, 1, dtype=torch.int16))
            if appearance:
                symbols['logscale'][2, 0] = 2
            if opacity:
                symbols['opacity'][1, 0] = 1
            packet = _stream_enc.encode_regional_full(symbols, steps, nodes, 1)
            for device in devices:
                with self.subTest(appearance=appearance, opacity=opacity, device=device):
                    reference = dict(logscale=torch.full((nodes, 3), -.25),
                                     sh_dc=torch.full((nodes, 1, 3), .5),
                                     sh_rest=torch.full((nodes, 3, 3), -.75),
                                     opacity=torch.full((nodes, 1), .25))
                    snapshot = {name: value.clone() for name, value in reference.items()}
                    capture = {} if device == 'cuda' else None
                    gpu_reference = ({name: value.to(device) for name, value in reference.items()}
                                     if capture is not None else None)
                    result = _stream_dec.decode_regional_full(
                        packet, reference, torch.arange(nodes), nodes, 1, device=device,
                        gpu_reference=gpu_reference, gpu_capture=capture)
                    if capture is not None:
                        for name, value in capture.items():
                            torch.testing.assert_close(value.cpu(), result[name], rtol=0, atol=1e-6)
                    for name, value in result.items():
                        torch.testing.assert_close(reference[name], snapshot[name], rtol=0, atol=0)
                        value.reshape(-1)[0] += 1
                        torch.testing.assert_close(reference[name], snapshot[name], rtol=0, atol=0)

    def test_regional_zero_reference_hash_cache_keeps_reference_validation(self):
        nodes = 4
        symbols = dict(logscale=torch.zeros(nodes, 3, dtype=torch.int16),
                       sh_dc=torch.zeros(nodes, 1, 3, dtype=torch.int16),
                       sh_rest=torch.zeros(nodes, 3, 3, dtype=torch.int16),
                       opacity=torch.zeros(nodes, 1, dtype=torch.int16))
        symbols['logscale'][2, 0] = 2
        steps = dict(logscale=.005, sh_dc=.001, sh_rest=.001, opacity=.01)
        packet = _stream_enc.encode_regional_full(symbols, steps, nodes, 1)
        reference = dict(logscale=torch.zeros(nodes, 3), sh_dc=torch.zeros(nodes, 1, 3),
                         sh_rest=torch.zeros(nodes, 3, 3), opacity=torch.zeros(nodes, 1))
        owner = torch.arange(nodes)
        _stream_dec._regional_zero_reference_hash.cache_clear()
        first = _stream_dec.decode_regional_full(packet, reference, owner, nodes, 1)
        second = _stream_dec.decode_regional_full(packet, reference, owner, nodes, 1)
        self.assertGreaterEqual(_stream_dec._regional_zero_reference_hash.cache_info().hits, 1)
        torch.testing.assert_close(first['logscale'], second['logscale'], rtol=0, atol=0)
        damaged = bytearray(packet)
        prior_offset = _stream_fmt.REGIONAL_ATTRIBUTE_HEADER.size + struct.calcsize('<4sIII3dI')
        damaged[prior_offset] ^= 1
        with self.assertRaisesRegex(ValueError, 'reference'):
            _stream_dec.decode_regional_full(bytes(damaged), reference, owner, nodes, 1)

    def test_raw_and_attribute_hashes_keep_exact_bytes_for_noncontiguous_and_empty_fields(self):
        from pipeline.model.frame_stream_format import RAW_FIELDS, reference_digest
        raw = reference_raw(rest=0)
        raw['xyz'] = raw['xyz'].t().contiguous().t()
        raw['xyz'][0, 0] = -0.
        expected = hashlib.sha256()
        for name in RAW_FIELDS:
            value = raw[name].detach().cpu().contiguous()
            expected.update(name.encode())
            expected.update(str(tuple(value.shape)).encode())
            expected.update(value.numpy().tobytes())
        self.assertEqual(reference_digest(raw), expected.hexdigest())
        expected = hashlib.sha256()
        for name in _stream_fmt.FIELDS:
            value = raw[name].detach().cpu().contiguous()
            expected.update(name.encode())
            expected.update(struct.pack('<I', value.ndim))
            expected.update(struct.pack('<' + 'I' * value.ndim, *value.shape))
            expected.update(value.numpy().tobytes())
        self.assertEqual(_stream_fmt._reference_hash({name: raw[name] for name in _stream_fmt.FIELDS}),
                         expected.digest())


# ----- test_rigid_sequence_benchmark -----
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SurvivorCompositionTests(unittest.TestCase):


    def test_nonspatial_matches_wire_with_survivors_and_owner_edits(self):
        prior,args=packet_fixture(16)
        rows=torch.tensor([0,2,4,6,8,10,12,14])
        owner=prior.owner[rows].clone();owner[0]=0
        parents=prior.parents
        rt,rr=args[6:8];gt,gr=args[8:10]
        metadata=(1.234567890123,.000123456789,.00123456789)
        steps=dict(gs_translation_step_extent=.0000123456789,gs_rotation_step_rad=.000987654321)
        inputs=(prior,1,metadata[0],rows,owner,parents,rt,rr,gt[rows],gr[rows],
            torch.empty(0,dtype=torch.long),torch.empty(0,3),torch.empty(0,4),*metadata[1:])
        decoded=_stream_dec.decode_motion(_stream_enc.encode_motion(*inputs,**steps),prior)
        xyz,q=_stream_fmt.compose_decoded_survivors(prior,metadata[0],rows,owner,parents,
            rt,rr,gt[rows],gr[rows],*metadata[1:],**steps)
        self.assertTrue(torch.equal(xyz,decoded.xyz));self.assertTrue(torch.equal(q,decoded.rotation))

    def test_spatial_relative_offsets_match_wire_and_never_recurse(self):
        for relative in (False,True):
            with self.subTest(relative=relative):
                model=spatial_tests.SpatialMotionTrainingTests().model(repetitions=2)
                prior=model.codec_previous_state
                # Nonunit raw quaternions detect unwanted normalization.
                prior.rotation=prior.rotation*1.00123
                rows=torch.arange(len(prior.xyz))[1:]
                owner=prior.owner[rows];parents=prior.parents;opened=prior.spatial_open
                g=torch.Generator().manual_seed(37)
                rt=torch.randint(-20,21,(len(parents),3),generator=g,dtype=torch.int16);rt[0]=0
                rr=torch.randint(-20,21,rt.shape,generator=g,dtype=torch.int16);rr[0]=0
                gt=torch.randint(-30,31,(len(rows),3),generator=g,dtype=torch.int16)
                gr=torch.randint(-30,31,gt.shape,generator=g,dtype=torch.int16)
                extent=1.234567890123;step=.000123456789;rstep=.00123456789
                options=dict(gs_translation_step_extent=.0000123456789,gs_rotation_step_rad=.000987654321,
                    spatial_open=opened,spatial_relative=relative)
                xyz,q=_stream_fmt.compose_decoded_survivors(prior,extent,rows,owner,parents,rt,rr,gt,gr,step,rstep,**options)
                sources=torch.empty(0,dtype=torch.long);birth_xyz=torch.empty(0,3)
                birth_q=torch.empty(0,4)
                inputs=(prior,1,extent,rows,owner,parents,rt,rr,gt,gr,sources,birth_xyz,birth_q,step,rstep)
                with patch.object(_stream_dec,'decode_motion',side_effect=AssertionError('encoder must not decode')):
                    packet=_stream_enc.encode_motion(*inputs,birth_pose_steps=(.00001,.001),compression='auto',**options)
                decoded=_stream_dec.decode_motion(packet,prior)
                self.assertTrue(torch.equal(xyz,decoded.xyz[:len(rows)]))
                self.assertTrue(torch.equal(q,decoded.rotation[:len(rows)]))

    def test_motion_convenience_method_does_not_mutate_or_encode(self):
        model=spatial_tests.SpatialMotionTrainingTests().model(repetitions=2)
        before=model.codec_previous_state.digest()
        with patch.object(_stream_enc,'encode_motion',side_effect=AssertionError('pure composition')):
            xyz,q=model.decoded_survivor_pose()
        self.assertEqual(before,model.codec_previous_state.digest())
        rt,rr,gt,gr=model._values(hard=True)
        empty=torch.empty(0,dtype=torch.long)
        packet=_stream_enc.encode_motion(model.frame_reference_state,model.frame_index,model.extent,
            model.previous_rows,model.owner,model.parents,rt,rr,gt,gr,empty,torch.empty(0,3),torch.empty(0,4),
            model.config.motion_xyz_step_extent,model.config.motion_rotation_step_rad,
            gs_translation_step_extent=model.config.gs_xyz_step_extent,gs_rotation_step_rad=model.config.gs_rotation_step_rad,
            spatial_open=model.opened,spatial_relative=model.relative_regions)
        decoded=_stream_dec.decode_motion(packet,model.frame_reference_state)
        self.assertTrue(torch.equal(xyz,decoded.xyz));self.assertTrue(torch.equal(q,decoded.rotation))


class BulkSparseDecodeTests(unittest.TestCase):
    def test_bulk_varints_match_scalar_canonical_reader_and_leave_trailers(self):
        rng=np.random.default_rng(728)
        values=np.r_[np.array([0,1,127,128,16383,16384,65535,65536,2**28,2**32-1],dtype=np.uint64),
                     rng.integers(0,2**32,size=10001,dtype=np.uint64)]
        encoded=_stream_enc.encode_unsigned_values(values)
        for count in (0,1,9,10,len(values)):
            with self.subTest(count=count):
                scalar=_stream_fmt._IntegerReader(encoded+b'trailer');bulk=_stream_fmt._IntegerReader(encoded+b'trailer')
                expected=[scalar.integer() for _ in range(count)]
                actual=_stream_fmt._read_unsigned_values(bulk,count)
                self.assertEqual(actual.tolist(),expected)
                self.assertEqual(bulk.cursor,scalar.cursor)
                self.assertEqual(bytes(bulk.take(bulk.remaining)),bytes(scalar.take(scalar.remaining)))

    def test_bulk_rejects_the_scalar_readers_malformed_and_noncanonical_words(self):
        examples=[(b'\x80\x00',1),(b'\x81\x00',1),(b'\x80'*5+b'\x01',1),
                  (b'\xff'*4+b'\x10',1),(b'\x80',1),(b'\x00',2),(b'\x00\x81\x00',2)]
        for data,count in examples:
            with self.subTest(data=data,count=count):
                scalar=_stream_fmt._IntegerReader(data)
                with self.assertRaises(ValueError):
                    for _ in range(count):scalar.integer()
                with self.assertRaises(ValueError):
                    _stream_fmt._read_unsigned_values(_stream_fmt._IntegerReader(data),count)

    def test_large_mixed_sparse_blocks_reconstruct_exact_bytes_and_integer_arrays(self):
        rng=np.random.default_rng(7421)
        for count in (0,1,10003):
            t=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            r=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            t[rng.random(count)<.7]=0;r[rng.random(count)<.85]=0
            raw=bytearray();_stream_enc._sparse_write(raw,t,r)
            reader=_stream_fmt._IntegerReader(raw)
            planar=_stream_enc._encode_sparse(reader,count)
            reconstructed=_stream_dec._decode_sparse(_stream_fmt._IntegerReader(planar),count)
            self.assertEqual(bytes(raw),reconstructed)
            decoded_t,decoded_r=_stream_dec._sparse_read(_stream_fmt._Reader(reconstructed),count,delta_rows=True)
            torch.testing.assert_close(decoded_t,torch.from_numpy(t),rtol=0,atol=0)
            torch.testing.assert_close(decoded_r,torch.from_numpy(r),rtol=0,atol=0)

    def test_gap_validation_preserves_survivor_and_deletion_boundaries(self):
        for values,limit,deletion,valid in [([1,1,1],3,False,True),([0,0,0],3,True,True),
                ([1,2,1],3,False,False),([0],3,False,False),([0,0,1],3,True,False)]:
            data=_stream_enc.encode_unsigned_values(values);reader=_stream_fmt._IntegerReader(data)
            if valid:self.assertEqual(reader.gaps(len(values),limit,deletion=deletion),data)
            else:
                with self.assertRaises(ValueError):reader.gaps(len(values),limit,deletion=deletion)

    def test_spatial_decode_does_not_compute_unused_rigid_pivots(self):
        model=spatial_tests.JointImageRateTests().model()
        payload=spatial_tests.JointImageRateTests.packet(model)
        with patch.object(_stream_dec,'region_pivots',side_effect=AssertionError('unused spatial pivots')):
            decoded=_stream_dec.decode_motion(payload,model.frame_reference_state)
        xyz,q=model.materialized_pose()
        torch.testing.assert_close(decoded.xyz,xyz,rtol=0,atol=0)
        torch.testing.assert_close(decoded.rotation,q,rtol=0,atol=0)

    def test_retained_planar_arrays_match_raw_parser_and_preserve_section_cursor(self):
        rng=np.random.default_rng(8102)
        for count in (0,1,12003):
            t=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            r=rng.integers(-32768,32768,size=(count,3),dtype=np.int16)
            t[rng.random(count)<.6]=0;r[rng.random(count)<.9]=0
            raw=bytearray();_stream_enc._sparse_write(raw,t,r)
            planar=_stream_enc._encode_sparse(_stream_fmt._IntegerReader(raw),count)
            retained=[]
            self.assertEqual(bytes(raw),_stream_dec._decode_sparse(_stream_fmt._IntegerReader(planar),count,decoded=retained))
            sections={0:(len(raw),count,*retained[0])}
            reader=_stream_fmt._Reader(raw+b'trailer')
            actual=_stream_dec._sparse_read(reader,count,delta_rows=True,sparse_sections=sections)
            self.assertFalse(sections)
            self.assertEqual(bytes(reader.take(7)),b'trailer')
            for value,expected in zip(actual,(t,r)):
                torch.testing.assert_close(value,torch.from_numpy(expected),rtol=0,atol=0)

    def test_retained_sections_cannot_bypass_zero_block_validation(self):
        for flag in (1,2,3):
            raw=b'\x01\x01'+bytes([flag])+b'\0'*(6*(bool(flag&1)+bool(flag&2)))
            planar=_stream_enc._encode_sparse(_stream_fmt._IntegerReader(raw),1)
            with self.assertRaisesRegex(ValueError,'Redundant zero'):
                _stream_dec._sparse_read(_stream_fmt._Reader(raw),1,delta_rows=True)
            with self.assertRaisesRegex(ValueError,'Redundant zero'):
                _stream_dec._decode_sparse(_stream_fmt._IntegerReader(planar),1,decoded=[])

    def test_retaining_an_empty_section_does_not_allocate_untrusted_population_size(self):
        retained=[]
        with patch.object(_stream_fmt.np,'zeros',side_effect=AssertionError('dense allocation before state check')):
            self.assertEqual(_stream_dec._decode_sparse(_stream_fmt._IntegerReader(b'\0'),2**32-1,decoded=retained),b'\0')
        self.assertTrue(all(value.size==0 for value in retained[0]))

    def test_whole_packet_uses_retained_sections_and_checks_original_checksum(self):
        model=spatial_tests.JointImageRateTests().model()
        raw=_stream_dec.uncompressed_motion_packet(spatial_tests.JointImageRateTests.packet(model))
        planar=_stream_enc.pack_layout(_stream_enc.encode_motion_layout(raw),len(raw),_stream_fmt.MOTION_MAGICS)
        expected=_stream_dec.decode_motion(raw,model.frame_reference_state)
        original=_stream_dec._sparse_read
        def checked(reader,count,**kwargs):
            self.assertIn(reader.cursor,kwargs['sparse_sections'])
            return original(reader,count,**kwargs)
        with patch.object(_stream_dec,'_sparse_read',side_effect=checked) as read:
            actual=_stream_dec.decode_motion(planar,model.frame_reference_state)
        self.assertEqual(read.call_count,2)
        self.assertEqual(actual.digest(),expected.digest())
        damaged=raw[:-1]+bytes([raw[-1]^1])
        packet=_stream_enc.pack_layout(_stream_enc.encode_motion_layout(damaged),len(damaged),_stream_fmt.MOTION_MAGICS)
        with patch.object(_stream_dec,'_compose_decoded_pose',side_effect=AssertionError('must validate first')), \
                self.assertRaisesRegex(ValueError,'checksum'):
            _stream_dec.decode_motion(packet,model.frame_reference_state)


# ----- Stream packet and GPU decoder parity -----


from pipeline.model.frame_stream_format import RAW_FIELDS,reference_digest
from pipeline.model.frame_stream_decode import (_opacity_symbols, read_stream_step)


def make_stream(directory, births=0, inherit_rotation=False, preserve=False):
    if births:
        raise ValueError('Current test fixture only supports the fixed GS population')
    raw=reference_raw()
    model=motion_model(raw);prior=model.frame_reference_state
    rows=torch.arange(len(prior.xyz)) if preserve else torch.tensor([0,2,3,4,5,6,7]);n=len(rows)
    rt=torch.zeros(len(prior.parents),3,dtype=torch.int16);rr=torch.zeros_like(rt)
    gt=torch.zeros(n,3,dtype=torch.int16);gr=torch.zeros_like(gt)
    gt[2]=torch.tensor([1,-2,0],dtype=torch.int16)
    sources=torch.tensor([0,4][:births],dtype=torch.long)
    xyz,q=_stream_fmt.compose_decoded_survivors(prior,2.,rows,prior.owner[rows],prior.parents,
        rt,rr,gt,gr,.001,.001,spatial_open=prior.spatial_open,spatial_relative=True,
        inherit_birth_rotation=inherit_rotation)
    newborn_xyz=xyz[sources]+torch.tensor([.002,0.,-.002])
    payload=_stream_enc.encode_motion(prior,1,2.,rows,prior.owner[rows],prior.parents,rt,rr,gt,gr,
        sources,newborn_xyz,q[sources],.001,.001,spatial_open=prior.spatial_open,
        spatial_relative=True,birth_pose_steps=(.002,.001),
        inherit_birth_rotation=inherit_rotation,
        gs_translation_step_extent=.001,gs_rotation_step_rad=.001)
    payload=_stream_enc.compress_motion_packet(payload)
    decoded=_stream_dec.decode_motion(payload,prior)
    current={name:value[rows].clone() for name,value in raw.items()}
    current['xyz']=decoded.xyz[:n];current['rotation']=decoded.rotation[:n]
    (directory/'motion_001.bin').write_bytes(payload)
    current['xyz'],current['rotation']=decoded.xyz,decoded.rotation
    manifest=dict(frame=1,motion='motion_001.bin',birth='birth_001.bin' if births else None,
        opacity_update=None,old_count=n,birth_count=births,total_count=n+births,
        previous_count=len(prior.xyz),decoded_state_sha256=decoded.digest().hex(),
        raw_state_sha256=reference_digest(current))
    (directory/'frame_001.json').write_text(json.dumps(manifest))
    return prior,raw,decoded,current,manifest


class StreamPacketTests(unittest.TestCase):


    def test_opacity_parsing_keeps_original_rows_symbols_and_all_envelope_checks(self):
        from pipeline.model.frame_stream_decode import (decode_opacity_delta)
        from pipeline.model.frame_stream_format import (OPACITY_HEADER as HEADER)
        import zlib
        reference,symbols,raw=opacity_fixture()
        planar=_stream_enc.encode_opacity_layout(raw)
        for packet in wrapped(planar,len(raw),_stream_fmt.OPACITY_MAGICS):
            parsed=_opacity_symbols(packet,1,len(reference))
            original=decode_opacity_delta(packet,reference,frame_index=1)
            np.testing.assert_array_equal(parsed.rows,original.rows.numpy())
            np.testing.assert_array_equal(parsed.values[:,0],original.symbols.numpy())
            with self.assertRaises(ValueError):_opacity_symbols(packet,2,len(reference))
            with self.assertRaises(ValueError):_opacity_symbols(packet+b'x',1,len(reference))


@unittest.skipUnless(torch.cuda.is_available(),'GPU incremental stream checks')
class GPUStreamTests(unittest.TestCase):
    def test_general_playback_keeps_dense_and_deleted_ids_across_frames(self):
        from scripts.frame_stream_player import GPUStream
        for preserve in (True, False):
            for births in (0,):
                with self.subTest(preserve=preserve, births=births), tempfile.TemporaryDirectory() as folder:
                    directory = Path(folder)
                    prior, raw, first, expected, _ = make_stream(
                        directory, births, inherit_rotation=True, preserve=preserve)
                    player = GPUStream(prior, raw, capacity=32)
                    player.advance_frame(directory, 1)
                    self.assertEqual(player._dense_ids, preserve)
                    actual = player.capture_raw()
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=1e-6)
                    player.verify_render_inputs(expected, atol=1e-6)

                    # Follow with an identity survivor mapping, including
                    # when the stable IDs already contain a deletion gap.
                    count = len(first.xyz)
                    rows = torch.arange(count)
                    region = torch.zeros(len(first.parents), 3, dtype=torch.int16)
                    point = torch.zeros(count, 3, dtype=torch.int16)
                    sources = torch.empty(0, dtype=torch.long)
                    payload = _stream_enc.encode_motion(
                        first, 2, 2., rows, first.owner, first.parents,
                        region, region, point, point, sources,
                        torch.empty(0, 3), torch.empty(0, 4), .001, .001,
                        spatial_open=first.spatial_open, spatial_relative=True,
                        inherit_birth_rotation=True, gs_translation_step_extent=.001,
                        gs_rotation_step_rad=.001)
                    payload = _stream_enc.compress_motion_packet(payload)
                    second = decode_motion(payload, first)
                    expected = {**expected, 'xyz': second.xyz, 'rotation': second.rotation}
                    (directory / 'motion_002.bin').write_bytes(payload)
                    (directory / 'frame_002.json').write_text(json.dumps(dict(
                        frame=2, motion='motion_002.bin', birth=None, opacity_update=None,
                        previous_count=count, old_count=count, birth_count=0, total_count=count,
                        decoded_state_sha256=second.digest().hex(),
                        raw_state_sha256=reference_digest(expected))))
                    player.advance_frame(directory, 2)
                    self.assertEqual(player._dense_ids, preserve)
                    actual = player.capture_raw()
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=1e-6)
                    player.verify_render_inputs(expected, atol=1e-6)


class FullAttributeStreamTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for cross-device quantization')
    def test_cuda_half_bin_forward_matches_encoded_symbols(self):
        ref = dict(logscale=torch.zeros(1, 3, device='cuda'),
                   sh_dc=torch.zeros(1, 1, 3, device='cuda'),
                   sh_rest=torch.zeros(1, 3, 3, device='cuda'))
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1, device='cuda'))])
        update = ResidualAppearanceAttributes(FullUpdateConfig())
        update.setup(ref, optimizer, (.001, .001, .001))
        with torch.no_grad():
            update.parameters['sh_dc'][0, 0, 0] = -0.07750000059604645
        forward = update.apply(ref)['sh_dc'].detach().cpu()
        symbols = update.symbols()
        cpu_ref = {name: value.cpu() for name, value in ref.items()}
        packet = encode_full_attribute_delta(cpu_ref, symbols, frame_index=1, steps=(.001, .001, .001))
        decoded = decode_full_attribute_delta(packet, cpu_ref, frame_index=1)
        self.assertTrue(torch.equal(forward, decoded['sh_dc']))

    def test_roundtrip_zero_preservation_reference_and_corruption(self):
        ref = dict(logscale=torch.tensor([[-0., 1., 2.], [3., 4., 5.]]),
                   sh_dc=torch.zeros(2, 1, 3), sh_rest=torch.zeros(2, 3, 3))
        symbols = {k: torch.zeros_like(v) for k, v in ref.items()}
        steps = (.005, .001, .001)
        self.assertIsNone(encode_full_attribute_delta(ref, symbols, frame_index=10, steps=steps))
        symbols['logscale'][1, 0] = -13
        symbols['sh_dc'][0, 0, 0] = 12
        symbols['sh_rest'][1, 2, 1] = 7
        packet = encode_full_attribute_delta(ref, symbols, frame_index=10, steps=steps)
        result = decode_full_attribute_delta(packet, ref, frame_index=10)
        self.assertTrue(torch.signbit(result['logscale'][0, 0]))
        for name, step in zip(FIELDS, steps):
            self.assertTrue(torch.equal(result[name], torch.where(symbols[name].ne(0),
                                           ref[name] + symbols[name] * step, ref[name])))
        changed = {k: v.clone() for k, v in ref.items()}
        changed['sh_dc'][0, 0, 1] = 1
        with self.assertRaises(ValueError):
            decode_full_attribute_delta(packet, changed, frame_index=10)
        damaged = bytearray(packet); damaged[-1] ^= 1
        with self.assertRaises(ValueError):
            decode_full_attribute_delta(damaged, ref, frame_index=10)

    def test_sparse_and_dense_use_fua3(self):
        torch.manual_seed(9)
        ref = dict(logscale=torch.zeros(3000, 3), sh_dc=torch.zeros(3000, 1, 3),
                   sh_rest=torch.zeros(3000, 3, 3))
        ref['logscale'][0, 0] = -0.
        steps = (.005, .001, .001)
        sparse = {k: torch.zeros_like(v) for k, v in ref.items()}
        for index, name in enumerate(FIELDS):
            sparse[name][1000 * index + 3].flatten()[0] = 25 + index
        sparse['logscale'][0, 1] = 17
        sparse['sh_rest'][-1, -1, -1] = -27
        packet = encode_full_attribute_delta(ref, sparse, frame_index=10, steps=steps)
        self.assertEqual(packet[:4], PLANAR_MAGIC)
        decoded = decode_full_attribute_delta(packet, ref, frame_index=10)
        self.assertTrue(torch.signbit(decoded['logscale'][0, 0]))
        for name, step in zip(FIELDS, steps):
            self.assertTrue(torch.equal(decoded[name], torch.where(sparse[name].ne(0),
                ref[name] + sparse[name] * step, ref[name])))
        dense = {k: torch.randint(-32767, 32768, v.shape).float() for k, v in ref.items()}
        packet = encode_full_attribute_delta(ref, dense, frame_index=10, steps=steps)
        self.assertEqual(packet[:4], PLANAR_MAGIC)
        decoded = decode_full_attribute_delta(packet, ref, frame_index=10)
        for name, step in zip(FIELDS, steps):
            self.assertTrue(torch.equal(decoded[name], torch.where(dense[name].ne(0),
                ref[name] + dense[name] * step, ref[name])))

    def test_sparse_formats_materialize_exact_bits_without_mutating_reference(self):
        torch.manual_seed(4242)
        steps = (.005, .001, .002)
        for rest in (0, 3, 15):
            ref = dict(logscale=torch.randn(257, 3), sh_dc=torch.randn(257, 1, 3),
                       sh_rest=torch.randn(257, rest, 3))
            symbols = {name: torch.zeros_like(value, dtype=torch.int16) for name, value in ref.items()}
            for name in FIELDS:
                if not symbols[name][1].numel():
                    continue
                # Zero components in a changed row must retain signed-zero bits.
                ref[name][1].view(-1)[0] = -0.
                symbols[name][1].view(-1)[1] = 32767
                symbols[name][127].view(-1)[0] = -32767
                symbols[name][-1].view(-1)[-1] = 7
            original = {name: value.clone() for name, value in ref.items()}
            body = _stream_enc._sparse_body([symbols[name].numpy() for name in FIELDS])
            planar = _stream_fmt._sparse_zigzag_byteplane(body, len(ref['logscale']), rest)
            for magic, payload in ((fua.PLANAR_MAGIC, planar),):
                with self.subTest(rest=rest, magic=magic):
                    packet = _encode_body(magic, payload, 7, 257, rest, steps, _reference_hash(ref))
                    decoded = decode_full_attribute_delta(packet, ref, frame_index=7)
                    for name, step in zip(FIELDS, steps):
                        integer = symbols[name].float()
                        expected = torch.where(integer.ne(0), ref[name] + integer * step, ref[name])
                        self.assertTrue(torch.equal(decoded[name].view(torch.int32), expected.view(torch.int32)))
                        self.assertTrue(torch.equal(ref[name].view(torch.int32), original[name].view(torch.int32)))
                    decoded['logscale'][0, 0] = 123.
                    self.assertTrue(torch.equal(ref['logscale'].view(torch.int32), original['logscale'].view(torch.int32)))

    def test_reordered_survivors_bind_reference_and_roundtrip(self):
        ref = dict(logscale=torch.arange(15).reshape(5, 3).float(),
                   sh_dc=torch.arange(15).reshape(5, 1, 3).float(),
                   sh_rest=torch.arange(45).reshape(5, 3, 3).float())
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        update = ResidualAppearanceAttributes(FullUpdateConfig(enabled=True))
        update.setup(ref, optimizer, (.01, .01, .01))
        with torch.no_grad():
            update.parameters['sh_dc'][4, 0, 0] = .003
            update.parameters['sh_rest'][1, 2, 2] = -.004
        rows = torch.tensor([4, 1, 3])
        update.retain(rows, optimizer)
        kept = {name: value[rows] for name, value in ref.items()}
        packet = encode_full_attribute_delta(kept, update.symbols(), frame_index=10, steps=update.steps)
        decoded = decode_full_attribute_delta(packet, kept, frame_index=10)
        expected = update.canonical(kept)
        for name in FIELDS:
            self.assertTrue(torch.equal(decoded[name], expected[name]))
        swapped = {name: value[torch.tensor([1, 0, 2])] for name, value in kept.items()}
        with self.assertRaises(ValueError):
            decode_full_attribute_delta(packet, swapped, frame_index=10)


def make_full_attribute_stream(directory, *, magic):
    count = 512
    xyz = torch.zeros(count, 3)
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1
    owner = torch.ones(count, dtype=torch.long)
    parents = torch.tensor([-1, 0])
    prior = DecodedState(xyz, rotation, owner, parents, 0)
    raw = dict(xyz=xyz, rotation=rotation, logscale=torch.zeros(count, 3),
               opacity=torch.zeros(count, 1), sh_dc=torch.zeros(count, 1, 3),
               sh_rest=torch.zeros(count, 3, 3))
    region = torch.zeros(2, 3, dtype=torch.int16)
    points = torch.zeros(count, 3, dtype=torch.int16)
    motion = encode_motion(prior, 1, 1., torch.arange(count), owner, parents,
        region, region, points, points, torch.empty(0, dtype=torch.long),
        torch.empty(0, 3), torch.empty(0, 4), .001, .001)
    full_attribute_reference = {name: raw[name] for name in fua.FIELDS}
    symbols = {name: torch.zeros_like(value) for name, value in full_attribute_reference.items()}
    for value in symbols.values():
        value[-1].flatten()[0] = 7
    arrays = [symbols[name].to(torch.int16).numpy().astype('<i2', copy=False)
              for name in fua.FIELDS]
    body = _stream_fmt._sparse_zigzag_byteplane(_stream_enc._sparse_body(arrays), count, 3)
    packet = _stream_enc._encode_body(magic, body, 1, count, 3, (.005, .001, .001),
                              _stream_fmt._reference_hash(full_attribute_reference))
    assert packet[:4] == magic
    directory = Path(directory)
    (directory / 'motion_001.bin').write_bytes(motion)
    (directory / 'fullattr_001.bin').write_bytes(packet)
    manifest = dict(frame=1, motion='motion_001.bin', birth=None, full_update=True,
                    full_attribute_update='fullattr_001.bin',
                    full_attribute_update_timing='after_regional_dc_before_births',
                    opacity_update_timing='before_births')
    index = directory / 'frame_001.json'
    index.write_text(json.dumps(manifest))
    payload = len(motion) + len(packet)
    summary = dict(motion=dict(bytes=len(motion)), birth=None,
                   full_attribute_update=dict(archive_bytes=len(packet)),
                   packet_bytes=payload, manifest_bytes=index.stat().st_size,
                   encoded_bytes=payload + index.stat().st_size,
                   complete_dynamic_update_bytes=payload + index.stat().st_size,
                   shared_reference_bytes=11, decoder_reference='reference_frame0.npz')
    return prior, raw, manifest, summary, packet


class FullAttributeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_replay_full_attributes_leave_previous_state_independent(self):
        for magic in (fua.PLANAR_MAGIC,):
            with self.subTest(magic=magic), tempfile.TemporaryDirectory() as directory:
                stream = Path(directory)
                prior, raw, _, _, packet = make_full_attribute_stream(stream, magic=magic)
                before = {name: value.clone() for name, value in raw.items()}
                _, current = decode_frame(stream, 1, prior, raw)
                for name in RAW_FIELDS:
                    self.assertTrue(torch.equal(raw[name].view(torch.int32), before[name].view(torch.int32)), name)
                    self.assertNotEqual(current[name].data_ptr(), raw[name].data_ptr(), name)
                    current[name].fill_(123)
                    self.assertTrue(torch.equal(raw[name].view(torch.int32), before[name].view(torch.int32)), name)

                damaged = bytearray(packet)
                damaged[-1] ^= 1
                (stream / 'fullattr_001.bin').write_bytes(damaged)
                with self.assertRaises(ValueError):
                    decode_frame(stream, 1, prior, raw)
                for name in RAW_FIELDS:
                    self.assertTrue(torch.equal(raw[name].view(torch.int32), before[name].view(torch.int32)), name)


def full_attribute_reference(count=8, rest=3):
    result = dict(logscale=torch.zeros(count, 3), sh_dc=torch.zeros(count, 1, 3),
                  sh_rest=torch.zeros(count, rest, 3))
    result['logscale'][0, 0] = -0.
    return result


def full_attribute_reference_packet(ref, symbols, frame=10):
    arrays = [symbols[name].to(torch.int16).numpy().astype('<i2', copy=False) for name in fua.FIELDS]
    sparse = _stream_enc._sparse_body(arrays)
    planar = _stream_fmt._sparse_zigzag_byteplane(sparse, len(ref['logscale']), ref['sh_rest'].shape[1])
    args = (frame, len(ref['logscale']), ref['sh_rest'].shape[1], FULL_ATTRIBUTE_STEPS, _stream_fmt._reference_hash(ref))
    return _stream_enc._encode_body(fua.PLANAR_MAGIC, planar, *args)


def full_attribute_bits(value):
    return value.detach().cpu().contiguous().numpy().tobytes()


class FUA3CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fua3_full_packet_matches_reference_layout(self):
        gen = torch.Generator().manual_seed(20261104)
        for count, active_mod, magnitude in ((8, 1, 32767), (4096, 127, 30),
                                             (300, 3, 20), (500, 1, 4)):
            ref = full_attribute_reference(count)
            symbols = {name: torch.randint(-magnitude, magnitude + 1, value.shape,
                                            generator=gen).float() for name, value in ref.items()}
            for value in symbols.values():
                value[torch.arange(count) % active_mod != 0] = 0
            for name in fua.FIELDS:
                symbols[name][0].flatten()[0] = -32767
                symbols[name][-1].flatten()[-1] = 32767
            new = full_attribute_reference_packet(ref, symbols)
            packet = fua.encode_full_attribute_delta(ref, symbols, frame_index=10, steps=FULL_ATTRIBUTE_STEPS)
            self.assertEqual(packet, new)
            results = [fua.decode_full_attribute_delta(p, ref, frame_index=10)
                       for p in (new, packet)]
            for name, step in zip(fua.FIELDS, FULL_ATTRIBUTE_STEPS):
                expected = torch.where(symbols[name].ne(0), ref[name] + symbols[name] * step, ref[name])
                for result in results:
                    self.assertEqual(full_attribute_bits(result[name]), full_attribute_bits(expected))

    def test_empty_fields_extremes_and_signed_range(self):
        ref = full_attribute_reference(4, rest=0)
        zero = {name: torch.zeros_like(value) for name, value in ref.items()}
        self.assertIsNone(fua.encode_full_attribute_delta(ref, zero, frame_index=10, steps=FULL_ATTRIBUTE_STEPS))
        empty_body = struct.pack('<III', 0, 0, 0)
        empty = _stream_enc._encode_body(fua.PLANAR_MAGIC, empty_body, 10, 4, 0, FULL_ATTRIBUTE_STEPS, _stream_fmt._reference_hash(ref))
        decoded = fua.decode_full_attribute_delta(empty, ref, frame_index=10)
        for name in fua.FIELDS:
            self.assertEqual(full_attribute_bits(decoded[name]), full_attribute_bits(ref[name]))
        symbols = {name: value.clone() for name, value in zero.items()}
        symbols['logscale'][0, :2] = torch.tensor([-32767., 32767.])
        for packet in (full_attribute_reference_packet(ref, symbols),):
            out = fua.decode_full_attribute_delta(packet, ref, frame_index=10)
            self.assertEqual(full_attribute_bits(out['logscale']), full_attribute_bits(torch.where(
                symbols['logscale'].ne(0), ref['logscale'] + symbols['logscale'] * FULL_ATTRIBUTE_STEPS[0], ref['logscale'])))
        symbols['logscale'][0, 0] = -32768
        with self.assertRaises(ValueError):
            fua.encode_full_attribute_delta(ref, symbols, frame_index=10, steps=FULL_ATTRIBUTE_STEPS)
        # FUA3 must reject the unsupported -32768 symbol.
        planar = struct.pack('<I', 1) + b'\x01' + bytes([255, 0, 0, 255, 0, 0]) + struct.pack('<II', 0, 0)
        packet = _stream_enc._encode_body(fua.PLANAR_MAGIC, planar, 10, 4, 0, FULL_ATTRIBUTE_STEPS, _stream_fmt._reference_hash(ref))
        with self.assertRaises(ValueError):
            fua.decode_full_attribute_delta(packet, ref, frame_index=10)

    def test_corrupt_headers_streams_gaps_and_rows_are_rejected(self):
        ref = full_attribute_reference(3)
        symbols = {name: torch.zeros_like(value) for name, value in ref.items()}
        symbols['logscale'][1, 1] = -7
        good = full_attribute_reference_packet(ref, symbols)
        for broken in (good[:3], good[:-1], good + b'\0', b'BAD!' + good[4:],
                       good[:fua.HEADER.size] + b'not-zlib'):
            with self.subTest(length=len(broken)), self.assertRaises(ValueError):
                fua.decode_full_attribute_delta(broken, ref, frame_index=10)
        altered = {name: value.clone() for name, value in ref.items()}
        altered['sh_dc'][0, 0, 0] = 1
        with self.assertRaises(ValueError):
            fua.decode_full_attribute_delta(good, altered, frame_index=10)
        with self.assertRaises(ValueError):
            fua.decode_full_attribute_delta(good, ref, frame_index=11)
        empty = struct.pack('<I', 0)
        row = bytes([2, 0, 0, 0, 0, 0])  # zigzag(1), three low then high bytes
        bad_bodies = [struct.pack('<I', 4) + b'\1' * 4 + row * 4 + empty * 2,
                      struct.pack('<I', 1) + b'\0' + row + empty * 2,
                      struct.pack('<I', 1) + b'\4' + row + empty * 2,
                      struct.pack('<I', 1) + b'\x81\0' + row + empty * 2,
                      struct.pack('<I', 1) + b'\1' + bytes(6) + empty * 2,
                      struct.pack('<I', 1) + b'\1' + row[:-1],
                      struct.pack('<I', 1) + b'\1' + row + empty * 2 + b'\0']
        for body in bad_bodies:
            packet = _stream_enc._encode_body(fua.PLANAR_MAGIC, body, 10, 3, 3, FULL_ATTRIBUTE_STEPS, _stream_fmt._reference_hash(ref))
            with self.subTest(body=body), self.assertRaises(ValueError):
                fua.decode_full_attribute_delta(packet, ref, frame_index=10)



if __name__ == "__main__":
    unittest.main()
