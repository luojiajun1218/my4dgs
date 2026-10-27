"""Shared DC updates reuse motion owners and preserve exact frame replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from pipeline.model.sparse_features import (FeatureDCUpdateConfig, QuantizedFeatureDCUpdate,
                                           decode_feature_dc_delta)
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from scripts.replay_rigid_tree import decode_frame
import test_spatial_motion as motion_fixtures
import scripts.benchmark_rigid_residual_tree as benchmark
from test_codec import reference_raw, motion_model, export_frame


class RegionalFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, frozen=True):
        motion = motion_fixtures.JointSpatialMotionTests().model()
        if frozen:
            motion._select_frontier(freeze=True)
        cfg = FeatureDCUpdateConfig(enabled=True, sharing='spatial_region', l1_weight=0.)
        update = QuantizedFeatureDCUpdate(cfg, motion=motion)
        reference = torch.arange(len(motion.owner) * 3).float().reshape(-1, 1, 3) * .001
        reference[0, 0, 0] = -0.
        update.setup(reference)
        return update, motion, reference

    def test_zero_bits_and_region_aggregated_gradients(self):
        update, motion, reference = self.model()
        node = int(motion.owner[0])
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[node] = torch.tensor([.004, .02, -.01])
        value = update()
        self.assertEqual(int(value.detach().view(torch.int32)[0, 0, 0]), -2147483648)
        value.sum().backward()
        counts = torch.bincount(motion.owner, minlength=len(motion.parents)).float()
        expected = counts[:, None].expand(-1, 3).reshape(-1, 1)
        torch.testing.assert_close(update.scalar.delta.grad, expected, rtol=0, atol=0)
        self.assertFalse(update.rate_terms(1.)['loss'].requires_grad)
        torch.testing.assert_close(value.detach(), update.canonical_features(), rtol=0, atol=0)

    def test_color_fitting_is_inactive_before_gate_freeze(self):
        update, motion, reference = self.model(frozen=False)
        with torch.no_grad():
            update.scalar.delta.fill_(.02)
        torch.testing.assert_close(update().view(torch.int32), reference.view(torch.int32), rtol=0, atol=0)
        self.assertFalse(update().requires_grad)
        self.assertEqual(float(update.rate_terms(1.)['loss']), 0.)
        before = update.scalar.delta.detach().clone()
        update.optimizer_step(1., proximal=True)
        torch.testing.assert_close(update.scalar.delta, before, rtol=0, atol=0)
        self.assertFalse(update.scalar.optimizer.state)
        motion._select_frontier(freeze=True)
        update().sum().backward()
        self.assertGreater(float(update.scalar.delta.grad.abs().sum()), 0.)

    def test_packet_uses_region_rows_and_binds_previous_features_and_motion_owners(self):
        update, motion, reference = self.model()
        used = motion.owner.unique()
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[used] = torch.tensor([.01, -.02, .03])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shdc_001.bin'
            info = update.export(path, 1)
            packet = path.read_bytes()
            self.assertEqual(packet[:4], b'SDR1')
            self.assertEqual(info['archive_bytes'], len(packet))
            self.assertLessEqual(info['rows'], len(used) * 3)
            decoded = decode_feature_dc_delta(packet, reference, frame_index=1,
                owner=motion.owner, node_count=len(motion.parents))
            torch.testing.assert_close(decoded, update.canonical_features(), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                decode_feature_dc_delta(packet, reference, frame_index=1)
            wrong = motion.owner.clone()
            wrong[0] = int(used[-1])
            for ref, owner, frame in ((reference, wrong, 1), (reference + .001, motion.owner, 1),
                                       (reference, motion.owner, 2)):
                with self.assertRaises(ValueError):
                    decode_feature_dc_delta(packet, ref, frame_index=frame,
                        owner=owner, node_count=len(motion.parents))

    def test_retirement_drops_unused_fields_and_keeps_node_indices(self):
        update, motion, reference = self.model()
        update().sum().backward()
        update.optimizer_step(0.)
        removed = int(motion.owner[0])
        rows = motion.owner.ne(removed).nonzero(as_tuple=True)[0]
        motion.retain_survivors(rows)
        update.commit(rows)
        self.assertEqual(update.scalar.delta.numel(), len(motion.parents) * 3)
        self.assertEqual(len(update()), len(rows))
        self.assertTrue(bool(update.scalar.delta.reshape(-1, 3)[removed].eq(0).all()))
        state = update.scalar.optimizer.state[update.scalar.delta]
        self.assertTrue(bool(state['exp_avg'].reshape(-1, 3)[removed].eq(0).all()))
        self.assertTrue(torch.equal(update.scalar.rows, torch.arange(len(motion.parents) * 3)))

    def test_committed_reference_and_optimizer_restore_exactly(self):
        update, motion, reference = self.model()
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[motion.owner.unique()] = .04
        update().sum().backward()
        update.optimizer_step(0.)
        rows = torch.arange(0, len(motion.owner), 2)
        motion.retain_survivors(rows)
        update.commit(rows)
        restored_motion = SpatialGatedMotion(copy.deepcopy(motion.config))
        restored_motion.restore(copy.deepcopy(motion.capture()))
        restored = QuantizedFeatureDCUpdate(copy.deepcopy(update.config), motion=restored_motion)
        restored.restore(copy.deepcopy(update.capture()), torch.device('cpu'))
        for current in (update, restored):
            current().square().mean().backward()
            current.optimizer_step(0.)
            current.finish()
        torch.testing.assert_close(update.canonical_features().view(torch.int32),
                                   restored.canonical_features().view(torch.int32), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.bin', Path(directory) / 'b.bin'
            update.export(a, 1)
            restored.export(b, 1)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_feature_payload_is_included_in_actual_complete_storage(self):
        update, motion, reference = self.model()
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[motion.owner.unique()] = .02
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            feature = directory / 'shdc_001.bin'
            info = update.export(feature, 1)
            (directory / 'motion_001.bin').write_bytes(b'motion')
            manifest = dict(motion='motion_001.bin', birth=None, feature_dc_update=feature.name)
            path = directory / 'frame_001.json'
            path.write_text(json.dumps(manifest))
            payload = 6 + feature.stat().st_size
            summary = dict(motion=dict(bytes=6), birth=None, feature_dc_update=info,
                encoded_bytes=payload, shared_reference_bytes=11, decoder_reference='reference_frame0.npz')
            storage = benchmark._actual_frame_storage(directory, 1, manifest, summary,
                                                       11, 'reference_frame0.npz')
            self.assertEqual(storage['feature_dc_update_bytes'], feature.stat().st_size)
            self.assertEqual(storage['complete_update_bytes'], payload + path.stat().st_size)
            feature.write_bytes(feature.read_bytes() + b'x')
            with self.assertRaises(ValueError):
                benchmark._actual_frame_storage(directory, 1, manifest, summary, 11, 'reference_frame0.npz')

    def test_empty_update_omits_packet_and_old_per_gs_packets_still_restore(self):
        update, motion, reference = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'shdc_001.bin'
            self.assertIsNone(update.export(path, 1))
            self.assertFalse(path.exists())
        old = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True))
        old.setup(reference)
        packet = copy.deepcopy(old.capture())
        packet['config'].pop('sharing')
        restored = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True))
        restored.restore(packet, torch.device('cpu'))
        torch.testing.assert_close(old().view(torch.int32), restored().view(torch.int32), rtol=0, atol=0)

    def test_survivors_and_births_replay_using_updated_shared_source_colors(self):
        raw = reference_raw()
        motion = motion_model(raw)
        preceding = copy.deepcopy(motion.frame_reference_state)
        motion._select_frontier(freeze=True)
        update = QuantizedFeatureDCUpdate(
            FeatureDCUpdateConfig(enabled=True, sharing='spatial_region'), motion=motion)
        update.setup(raw['sh_dc'])
        with torch.no_grad():
            update.scalar.delta.reshape(-1, 3)[motion.owner.unique()] = torch.tensor([.02, -.01, .03])
        rows = torch.tensor([1, 3, 4, 6, 7])
        motion.retain_survivors(rows)
        update.commit(rows)
        surviving = {name: value[rows].clone() for name, value in raw.items()}
        surviving['sh_dc'] = update.canonical_features()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            expected = export_frame(directory, motion, surviving, torch.tensor([0, 2]))
            update.export(directory / 'shdc_001.bin', 1)
            (directory / 'frame_001.json').write_text(json.dumps(dict(
                feature_dc_update='shdc_001.bin', feature_dc_update_timing='before_births')))
            _, decoded = decode_frame(directory, 1, preceding, raw)
            for name in raw:
                torch.testing.assert_close(decoded[name], expected[name], rtol=0, atol=0)

    def test_regional_updates_require_a_prepared_motion_binding(self):
        update = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True, sharing='spatial_region'))
        with self.assertRaises(ValueError):
            update.setup(torch.zeros(3, 1, 3))


if __name__ == '__main__':
    unittest.main()
