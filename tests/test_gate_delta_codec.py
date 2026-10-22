"""Causal gate updates must preserve the full decoded state and legacy streams."""
import copy
import hashlib
import unittest

import torch

from pipeline.model import rigid_tree_codec as codec
from pipeline.model import sparse_integer_codec as integer
from pipeline.model.motion_rate import MotionCodecRate
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gate import spatial_owner, spatial_state_route
from pipeline.model.spatial_gated_motion import SpatialGatedMotion


class GateDeltaCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        cfg = RigidTreeConfig()
        gate = cfg.spatial_gate
        gate.max_depth, gate.min_region_size = 4, 2
        gate.training_schedule = 'joint'
        gate.start_step = gate.gate_start_step = 0
        gate.initial_logit = -2.
        self.model = SpatialGatedMotion(cfg)
        xyz = torch.cartesian_prod(*[torch.linspace(-1., 1., 4)] * 3).repeat_interleave(2, 0)
        rotation = torch.tensor([[1., 0., 0., 0.]]).repeat(len(xyz), 1)
        self.model.setup(xyz, rotation, 1., frame_index=1)
        self.prior = copy.deepcopy(self.model.frame_reference_state)

    def arguments(self, prior=None, opened=None, *, births=False, retire=False,
                  relative=False, translation_only=False):
        prior = self.prior if prior is None else prior
        opened = prior.spatial_open.clone() if opened is None else opened
        rows = torch.arange(len(prior.xyz))
        if retire:
            rows = rows[rows % 7 != 0]
        paths = spatial_state_route(prior.xyz[rows], prior)
        owner = spatial_owner(paths, opened)
        rt = torch.zeros(len(prior.parents), 3, dtype=torch.int16)
        rr = torch.zeros_like(rt)
        rt[1, 0], rt[9, 1] = 3, -2
        gt = torch.zeros(len(rows), 3, dtype=torch.int16)
        gr = torch.zeros_like(gt)
        gt[::5, 2] = 2
        if not translation_only:
            rr[1, 1], gr[::11, 0] = 1, -1
        sources = torch.tensor([0, len(rows) - 1]) if births else torch.empty(0, dtype=torch.long)
        args = (prior, prior.frame_index + 1, 1., rows, owner, prior.parents,
                rt, rr, gt, gr, sources, prior.xyz[rows[sources]] + .003,
                prior.rotation[rows[sources]], .001, .001)
        kwargs = dict(gs_translation_step_extent=.001, gs_rotation_step_rad=.001,
                      spatial_open=opened, spatial_relative=relative,
                      inherit_birth_rotation=translation_only)
        return args, kwargs

    def assert_same_state(self, left, right):
        self.assertEqual(left.digest(), right.digest())
        for name in ('xyz', 'rotation', 'owner', 'parents', 'spatial_paths',
                     'spatial_open', 'birth_sources', 'survivor_rows'):
            a, b = getattr(left, name), getattr(right, name)
            if a is None:
                self.assertIsNone(b)
            else:
                self.assertEqual(a.numpy().tobytes(), b.numpy().tobytes(), name)

    def test_unchanged_gates_encode_one_zero_count(self):
        args, kwargs = self.arguments()
        old = codec.encode_motion(*args, **kwargs, gate_delta=False)
        new = codec.encode_motion(*args, **kwargs)
        self.assertEqual(new[:4], b'SPG5')
        reader = integer._Reader(new)
        _, _, _, start = integer._spatial_prefix(reader)
        self.assertEqual(new[start:reader.cursor], b'\x00')
        self.assertEqual(len(old) - len(new), (len(self.prior.parents) + 7) // 8 - 1)
        self.assert_same_state(codec.decode_motion(old, self.prior),
                               codec.decode_motion(new, self.prior))

    def test_sparse_changes_births_retirement_and_all_pose_variants(self):
        opened = self.prior.spatial_open.clone()
        opened[1] = True
        before = self.prior.digest()
        for relative in (False, True):
            for translation_only in (False, True):
                args, kwargs = self.arguments(opened=opened, births=True, retire=True,
                    relative=relative, translation_only=translation_only)
                legacy = codec.encode_motion(*args, **kwargs, gate_delta=False)
                expected = codec.decode_motion(legacy, self.prior)
                for compression in ('none', 'zlib', 'auto'):
                    packet = codec.encode_motion(*args, **kwargs, compression=compression)
                    raw = codec.uncompressed_motion_packet(packet)
                    self.assertIn(raw[:4], codec.SPATIAL_MAGICS)
                    if compression == 'none':
                        self.assertIn(raw[:4], integer.GATE_DELTA_MAGICS)
                    layout = integer.encode_motion_layout(raw)
                    planar = integer.pack_layout(layout, len(raw), integer.MOTION_MAGICS)
                    self.assertEqual(integer.decode_motion_layout(layout, len(raw)), raw)
                    for encoded in (packet, planar):
                        self.assert_same_state(codec.decode_motion(encoded, self.prior), expected)
        self.assertEqual(self.prior.digest(), before)

    def test_automatic_encoding_never_exceeds_full_bitmap_encoding(self):
        for index in range(6):
            opened = self.prior.spatial_open.clone()
            opened[1:585:index + 1] = True
            args, kwargs = self.arguments(opened=opened, births=True, relative=bool(index % 2))
            full = codec.encode_motion(*args, **kwargs, gate_delta=False, compression='auto')
            delta = codec.encode_motion(*args, **kwargs, compression='auto')
            self.assertLessEqual(len(delta), len(full))
            self.assert_same_state(codec.decode_motion(delta, self.prior),
                                   codec.decode_motion(full, self.prior))

    def test_open_then_close_in_next_frame_preserves_birth_lineage(self):
        opened = self.prior.spatial_open.clone()
        opened[1] = True
        args, kwargs = self.arguments(opened=opened, births=True)
        first = codec.decode_motion(codec.encode_motion(*args, **kwargs), self.prior)
        opened = first.spatial_open.clone()
        opened[1] = False
        args, kwargs = self.arguments(prior=first, opened=opened, births=True, retire=True)
        new = codec.encode_motion(*args, **kwargs)
        old = codec.encode_motion(*args, **kwargs, gate_delta=False)
        final = codec.decode_motion(new, first)
        self.assertFalse(bool(final.spatial_open[1]))
        self.assert_same_state(final, codec.decode_motion(old, first))
        with self.assertRaises(ValueError):
            codec.decode_motion(new, self.prior)

    def test_unreachable_descendant_bit_is_not_lost(self):
        opened = self.prior.spatial_open.clone()
        opened[9] = True
        args, kwargs = self.arguments(opened=opened)
        self.assertTrue(torch.equal(args[4], self.prior.owner))
        decoded = codec.decode_motion(codec.encode_motion(*args, **kwargs), self.prior)
        self.assertTrue(bool(decoded.spatial_open[9]))
        self.assertFalse(bool(decoded.spatial_open[1]))

    def test_dense_change_falls_back_to_legacy_bitmap(self):
        opened = self.prior.spatial_open.clone()
        opened[:585] = True
        args, kwargs = self.arguments(opened=opened)
        packet = codec.encode_motion(*args, **kwargs)
        self.assertEqual(packet[:4], b'SPG1')
        self.assertTrue(torch.equal(codec.decode_motion(packet, self.prior).spatial_open, opened))

    def test_corrupt_changed_ids_are_rejected_after_valid_checksum(self):
        args, kwargs = self.arguments()
        raw = codec.encode_motion(*args, **kwargs)
        reader = integer._Reader(raw)
        _, _, _, start = integer._spatial_prefix(reader)
        end = reader.cursor
        for changes in (b'\x01\x00', b'\x01\x01',
                        b'\x01' + codec._uvar(len(self.prior.parents) + 1),
                        b'\x02\x02\x00', codec._uvar(len(self.prior.parents))):
            body = raw[:start] + changes + raw[end:-codec.CHECKSUM_BYTES]
            bad = body + hashlib.sha256(body).digest()[:codec.CHECKSUM_BYTES]
            with self.assertRaises(ValueError):
                codec.decode_motion(bad, self.prior)

    def test_codec_rate_matches_production_packet_and_restore(self):
        args, kwargs = self.arguments()
        zeros = list(args)
        for index in (6, 7, 8, 9):
            zeros[index] = torch.zeros_like(args[index])
        empty = codec.encode_motion(*zeros, **kwargs)
        rate = MotionCodecRate(empty, len(self.prior.parents), len(args[3]),
                               previous_opened=self.prior.spatial_open)
        for node in (None, 1):
            opened = self.prior.spatial_open.clone()
            if node is not None:
                opened[node] = True
            kwargs['spatial_open'] = opened
            altered = list(args)
            altered[4] = spatial_owner(spatial_state_route(self.prior.xyz, self.prior), opened)
            expected = codec.encode_motion(*altered, **kwargs, compression='auto')
            actual = rate.packet(*(value.numpy() for value in args[6:10]), opened)
            self.assertEqual(actual, expected)
            restored = MotionCodecRate.restore(copy.deepcopy(rate.capture()))
            self.assertEqual(restored.packet(*(value.numpy() for value in args[6:10]), opened), expected)
        old_rate = rate.capture()
        old_rate.pop('previous_opened')
        restored = MotionCodecRate.restore(old_rate)
        self.assertIsNone(restored.previous_opened)
        self.model._codec_rate = restored
        self.model._measure_codec_rate(1, force=True)
        self.assertTrue(torch.equal(self.model._codec_rate.previous_opened,
                                    self.prior.spatial_open))


if __name__ == '__main__':
    unittest.main()
