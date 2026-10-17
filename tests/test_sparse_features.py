"""DC deltas must preserve untouched bits, survivor rows and birth predictors."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pipeline.model.gaussian import GaussianModel
from pipeline.model.sparse_features import (FeatureDCUpdateConfig, QuantizedFeatureDCUpdate,
    decode_feature_dc_delta)
from scripts.replay_rigid_tree import decode_frame
from test_gaussian_replacement import reference_raw, motion_model, export_frame


class SparseFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_bits_rgb_gradient_and_feature_override(self):
        update = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True))
        reference = torch.tensor([[[-0., .2, -.3]], [[.4, -.5, .6]]])
        update.setup(reference)
        with torch.no_grad():
            update.scalar.delta[:, 0] = torch.tensor([.004, .02, -.01, 0., .03, 0.])
        values = update()
        self.assertEqual(int(values.detach().view(torch.int32)[0, 0, 0]), -2147483648)
        values.sum().backward()
        torch.testing.assert_close(update.scalar.delta.grad, torch.ones(6, 1), rtol=0, atol=0)
        self.assertFalse(update.rate_terms(1.)['loss'].requires_grad)
        model = GaussianModel(1)
        model._features_dc = reference
        model._features_rest = torch.zeros(2, 3, 3)
        model.rigid_sh_dc = values
        torch.testing.assert_close(model.get_features[:, :1], values, rtol=0, atol=0)
        model.rigid_sh_dc = None
        torch.testing.assert_close(model.get_features[:, :1], reference, rtol=0, atol=0)

    def test_channel_order_survivors_resume_and_exact_packet(self):
        cfg = FeatureDCUpdateConfig(enabled=True)
        one = QuantizedFeatureDCUpdate(cfg)
        reference = torch.arange(12).float().reshape(4, 1, 3) * .1
        one.setup(reference)
        with torch.no_grad():
            one.scalar.delta[:, 0] = torch.arange(12).float() * .01
        one.commit(torch.tensor([1, 3]))
        two = QuantizedFeatureDCUpdate(copy.deepcopy(cfg))
        two.restore(copy.deepcopy(one.capture()), torch.device('cpu'))
        for update in (one, two):
            update().sum().backward(); update.optimizer_step(0.)
            update.finish()
        torch.testing.assert_close(one.canonical_features(), two.canonical_features(), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'shdc_002.bin'
            info = two.export(path, 2)
            self.assertEqual(path.stat().st_size, info['archive_bytes'])
            decoded = decode_feature_dc_delta(path.read_bytes(), reference[[1, 3]], frame_index=2)
            torch.testing.assert_close(decoded, two.canonical_features(), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                decode_feature_dc_delta(path.read_bytes(), reference[[3, 1]], frame_index=2)
            with self.assertRaises(ValueError):
                decode_feature_dc_delta(path.read_bytes(), reference[[1, 3]], frame_index=3)

    def test_empty_feature_packet_is_omitted(self):
        update = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True))
        update.setup(torch.zeros(3, 1, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'shdc_001.bin'
            self.assertIsNone(update.export(path, 1))
            self.assertFalse(path.exists())

    def test_deleted_rows_and_births_replay_with_updated_source_colors(self):
        raw = reference_raw()
        motion = motion_model(raw)
        preceding = copy.deepcopy(motion.frame_reference_state)
        rows = torch.tensor([1, 3, 4, 6, 7])
        motion.retain_survivors(rows)
        update = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True))
        update.setup(raw['sh_dc'])
        with torch.no_grad():
            update.scalar.delta[:, 0] = torch.arange(24).float() * .01
        update.commit(rows)
        surviving = {name: value[rows].clone() for name, value in raw.items()}
        surviving['sh_dc'] = update.canonical_features()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            expected = export_frame(directory, motion, surviving, torch.tensor([0, 2]))
            update.export(directory/'shdc_001.bin', 1)
            (directory/'frame_001.json').write_text(json.dumps(dict(
                feature_dc_update='shdc_001.bin', feature_dc_update_timing='before_births')))
            _, decoded = decode_frame(directory, 1, preceding, raw)
            for name in raw:
                torch.testing.assert_close(decoded[name], expected[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
