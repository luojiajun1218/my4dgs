"""Historical DC packets remain readable after color-training removal."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pipeline.model.legacy_feature_codec import (
    MAGIC, REGIONAL_MAGIC, _regional_digest, decode_feature_dc_delta)
from pipeline.model.sparse_updates import encode_opacity_delta
from scripts.replay_rigid_tree import decode_frame
from _stream_fixtures import reference_raw, motion_model, export_frame


class LegacyFeatureCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_old_per_gaussian_packet_channel_order_frame_and_reference(self):
        reference = torch.arange(12).float().reshape(4, 1, 3) * .1
        symbols = torch.arange(12).reshape(-1, 1)
        packet = MAGIC + encode_opacity_delta(
            reference.reshape(-1, 1), symbols, frame_index=2, step=.01)
        restored = decode_feature_dc_delta(packet, reference, frame_index=2)
        expected = reference + symbols.reshape_as(reference) * .01
        torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        for ref, frame in ((reference.flip(0), 2), (reference, 3)):
            with self.assertRaises(ValueError):
                decode_feature_dc_delta(packet, ref, frame_index=frame)

    def test_old_region_packet_binds_motion_owner_and_preserves_signed_zero(self):
        reference = torch.tensor([[[-0., .2, -.3]], [[.4, -.5, .6]]])
        owner = torch.tensor([1, 2])
        symbols = torch.tensor([[0, 0, 0], [0, 2, -1], [1, 0, -2]])
        packet = (REGIONAL_MAGIC + _regional_digest(reference, owner, 3) +
            encode_opacity_delta(torch.zeros(9, 1), symbols.reshape(-1, 1),
                                 frame_index=1, step=.01))
        restored = decode_feature_dc_delta(packet, reference, frame_index=1,
                                          owner=owner, node_count=3)
        delta = symbols.float()[owner, None] * .01
        expected = torch.where(delta.ne(0), reference + delta, reference)
        torch.testing.assert_close(restored.view(torch.int32), expected.view(torch.int32), rtol=0, atol=0)
        self.assertEqual(int(restored.view(torch.int32)[0, 0, 0]), -2147483648)
        for ref, own, frame in ((reference, owner.flip(0), 1),
                                (reference + .001, owner, 1), (reference, owner, 2)):
            with self.assertRaises(ValueError):
                decode_feature_dc_delta(packet, ref, frame_index=frame, owner=own, node_count=3)
        with self.assertRaises(ValueError):
            decode_feature_dc_delta(packet, reference, frame_index=1)

    def test_historical_retirement_birth_packet_uses_updated_source_colors(self):
        raw = reference_raw()
        motion = motion_model(raw)
        preceding = copy.deepcopy(motion.frame_reference_state)
        rows = torch.tensor([1, 3, 4, 6, 7])
        motion.retain_survivors(rows)
        reference = raw['sh_dc'][rows]
        symbols = torch.arange(reference.numel()).reshape(-1, 1)
        packet = MAGIC + encode_opacity_delta(reference.reshape(-1, 1), symbols,
                                              frame_index=1, step=.01)
        surviving = {name: value[rows].clone() for name, value in raw.items()}
        surviving['sh_dc'] = decode_feature_dc_delta(packet, reference, frame_index=1)
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            expected = export_frame(directory, motion, surviving, torch.tensor([0, 2]))
            (directory / 'shdc_001.bin').write_bytes(packet)
            (directory / 'frame_001.json').write_text(json.dumps(dict(
                feature_dc_update='shdc_001.bin', feature_dc_update_timing='before_births')))
            _, restored = decode_frame(directory, 1, preceding, raw)
        for key in expected:
            torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
