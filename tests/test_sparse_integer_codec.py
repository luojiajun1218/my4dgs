"""New sparse layouts preserve exact packets and sequential decoded state."""
import hashlib
import lzma
from pathlib import Path
import struct
import sys
import unittest
import zlib

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model import sparse_integer_codec as S
from pipeline.model import rigid_tree_codec as M
from pipeline.model.legacy_opacity_codec import (HEADER, MAGIC, decode_opacity_delta,
                                                opacity_reference_hash)
from pipeline.model.sparse_updates import encode_opacity_delta


def wrapped(layout, raw_length, magics):
    prefix = S._varint(raw_length) + S._varint(len(layout))
    return [magics[0] + prefix + layout,
            magics[1] + prefix + zlib.compress(layout, 6),
            magics[2] + prefix + lzma.compress(layout, preset=9)]


def spatial_fixture():
    from test_spatial_gated_motion import SpatialGatedMotionTests
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


if __name__ == '__main__':
    unittest.main()
