"""Sparse DC and log-scale updates preserve their codec and predictor contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pipeline.model.gaussian import GaussianModel
from pipeline.model.sparse_features import (FeatureDCUpdateConfig, QuantizedFeatureDCUpdate,
    REGIONAL_MAGIC, _regional_digest, decode_feature_dc_delta)
from pipeline.model.sparse_logscale import (LogScaleUpdateConfig, MotionSupportLogScaleUpdate,
    decode_logscale_delta, encode_logscale_delta)
from scripts.replay_rigid_tree import decode_frame
from tests.test_codec import reference_raw, motion_model, export_frame

from pipeline.model.sparse_updates import OpacityUpdateConfig, QuantizedOpacityUpdate
from scripts.benchmark_rigid_residual_tree import _actual_frame_storage


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


class SparseLogScaleCodecTests(unittest.TestCase):
    def test_packet_is_exact_and_bound_to_reference_and_motion_support(self):
        reference = torch.tensor([
            [-3., -2., -1.], [-2., -2., -2.], [-1., -1., -1.],
            [0., 0., 0.], [1., 1., 1.]], dtype=torch.float32)
        symbols = torch.tensor([0, -4, 0, 7, 0])
        allowed = torch.tensor([1, 3, 4])
        packet = encode_logscale_delta(reference, symbols, allowed_rows=allowed,
                                       frame_index=3, step=.005)
        decoded = decode_logscale_delta(packet, reference, allowed_rows=allowed,
                                        frame_index=3)
        expected = reference.clone()
        expected[1] += -.02
        expected[3] += .035
        torch.testing.assert_close(decoded.logscale, expected, rtol=0, atol=0)
        self.assertEqual(decoded.rows.tolist(), [1, 3])
        self.assertEqual(decoded.symbols.tolist(), [-4, 7])
        with self.assertRaisesRegex(ValueError, 'reference or motion support'):
            decode_logscale_delta(packet, reference, allowed_rows=torch.tensor([1, 2, 3, 4]),
                                  frame_index=3)
        changed = reference.clone()
        changed[0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'reference or motion support'):
            decode_logscale_delta(packet, changed, allowed_rows=allowed, frame_index=3)
        with self.assertRaisesRegex(ValueError, 'outside transmitted motion support'):
            encode_logscale_delta(reference, torch.tensor([1, 0, 0, 0, 0]),
                                  allowed_rows=allowed, frame_index=3, step=.005)

    def test_zero_update_has_no_packet(self):
        reference = torch.zeros(3, 3)
        self.assertIsNone(encode_logscale_delta(
            reference, torch.zeros(3), allowed_rows=torch.tensor([0, 2]),
            frame_index=1, step=.005))

    def test_anisotropic_packet_is_exact_and_preserves_axis_symbols(self):
        reference = torch.tensor([
            [-3., -2., -1.], [-2., -2., -2.], [-1., -1., -1.],
            [0., 0., 0.]], dtype=torch.float32)
        symbols = torch.tensor([[0, 0, 0], [-4, 0, 2], [0, 0, 0], [7, -3, 1]])
        allowed = torch.tensor([1, 2, 3])
        packet = encode_logscale_delta(reference, symbols, allowed_rows=allowed,
                                       frame_index=4, step=.005)
        self.assertIn(packet[:4], (b'SLA1', b'ZLA1'))
        decoded = decode_logscale_delta(packet, reference, allowed_rows=allowed,
                                        frame_index=4)
        expected = reference.clone()
        expected[1] += torch.tensor([-.02, 0., .01])
        expected[3] += torch.tensor([.035, -.015, .005])
        torch.testing.assert_close(decoded.logscale, expected, rtol=0, atol=0)
        self.assertEqual(decoded.rows.tolist(), [1, 3])
        self.assertEqual(decoded.symbols.tolist(), [[-4, 0, 2], [7, -3, 1]])
        changed_magic = ((b'SLS1' if packet[:4] == b'SLA1' else b'SLZ1') + packet[4:])
        with self.assertRaises(ValueError):
            decode_logscale_delta(changed_magic, reference, allowed_rows=allowed,
                                  frame_index=4)


class MotionSupportLogScaleUpdateTests(unittest.TestCase):
    def config(self):
        return LogScaleUpdateConfig(enabled=True, step=.01, learning_rate=.1,
            l1_weight=0., max_rows=1, selection_steps=1, refit_steps=1,
            proximal_interval=1)

    def optimizer(self):
        anchor = torch.nn.Parameter(torch.zeros(()))
        return anchor, torch.optim.Adam([dict(params=[anchor], name='anchor')], lr=.01)

    def test_hard_budget_survivors_checkpoint_and_export(self):
        anchor, optimizer = self.optimizer()
        update = MotionSupportLogScaleUpdate(self.config())
        update.setup(5, torch.tensor([1, 2, 4]), optimizer, start_step=10)
        reference = torch.zeros(5, 3)
        weights = torch.tensor([0., 1., 4., 0., 2.])[:, None]
        loss = (update.current_logscale(reference) * weights).sum()
        loss.backward()
        update.before_optimizer_step()
        optimizer.step()
        update.after_optimizer_step(11, optimizer, byte_price=0.)
        optimizer.zero_grad(set_to_none=True)
        self.assertEqual(int(update.support.sum()), 1)
        selected = int(update.support.nonzero()[0])
        self.assertEqual(selected, 2)
        loss = update.current_logscale(reference).sum()
        loss.backward()
        update.before_optimizer_step()
        optimizer.step()
        update.after_optimizer_step(12, optimizer, byte_price=0.)
        optimizer.zero_grad(set_to_none=True)
        update.retain_survivors(torch.tensor([0, 2, 3, 4]))
        current_reference = reference[[0, 2, 3, 4]]
        packet = update.capture(optimizer)

        other_anchor, other_optimizer = self.optimizer()
        restored = MotionSupportLogScaleUpdate(copy.deepcopy(self.config()))
        restored.restore(copy.deepcopy(packet), torch.device('cpu'), other_optimizer)
        torch.testing.assert_close(restored.current_logscale(current_reference),
                                   update.current_logscale(current_reference),
                                   rtol=0, atol=0)
        self.assertEqual(restored.current_candidates().tolist(), [1, 3])
        restored.finish()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'logscale_001.bin'
            info = restored.export(path, current_reference, frame_index=1,
                                   allowed_rows=torch.tensor([1, 3]))
            self.assertIsNotNone(info)
            decoded = decode_logscale_delta(path.read_bytes(), current_reference,
                allowed_rows=torch.tensor([1, 3]), frame_index=1)
            torch.testing.assert_close(decoded.logscale,
                restored.canonical_logscale(current_reference), rtol=0, atol=0)

    def test_config_validation(self):
        for field, value in (('step', 0.), ('learning_rate', float('inf')),
                             ('l1_weight', -1.), ('max_rows', 0),
                             ('selection_steps', True), ('refit_steps', 0),
                             ('proximal_interval', 0)):
            config = self.config()
            setattr(config, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                config.validate()
        config = self.config()
        config.components = 'vector'
        with self.assertRaises(ValueError):
            config.validate()

    def test_anisotropic_update_selects_rows_and_roundtrips_checkpoint(self):
        config = self.config()
        config.components = 'anisotropic'
        config.max_rows = 2
        anchor, optimizer = self.optimizer()
        update = MotionSupportLogScaleUpdate(config)
        update.setup(4, torch.tensor([0, 1, 3]), optimizer, start_step=10)
        self.assertEqual(tuple(update.delta.shape), (4, 3))
        reference = torch.zeros(4, 3)
        weights = torch.tensor([[1., 0., 0.], [0., 4., 0.],
                                [0., 0., 0.], [0., 0., 2.]])
        (update.current_logscale(reference) * weights).sum().backward()
        update.before_optimizer_step()
        optimizer.step()
        update.after_optimizer_step(11, optimizer, byte_price=0.)
        optimizer.zero_grad(set_to_none=True)
        self.assertEqual(int(update.support.sum()), 2)
        update.current_logscale(reference).sum().backward()
        update.before_optimizer_step()
        optimizer.step()
        update.after_optimizer_step(12, optimizer, byte_price=0.)
        optimizer.zero_grad(set_to_none=True)
        captured = update.capture(optimizer)
        other_anchor, other_optimizer = self.optimizer()
        restored = MotionSupportLogScaleUpdate(copy.deepcopy(config))
        restored.restore(copy.deepcopy(captured), torch.device('cpu'), other_optimizer)
        torch.testing.assert_close(restored.current_logscale(reference),
                                   update.current_logscale(reference), rtol=0, atol=0)
        restored.finish()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'logscale_001.bin'
            info = restored.export(path, reference, frame_index=1,
                                   allowed_rows=torch.tensor([0, 1, 3]))
            self.assertFalse(info['isotropic'])
            self.assertEqual(info['components'], 'anisotropic')
            decoded = decode_logscale_delta(path.read_bytes(), reference,
                allowed_rows=torch.tensor([0, 1, 3]), frame_index=1)
            torch.testing.assert_close(decoded.logscale,
                restored.canonical_logscale(reference), rtol=0, atol=0)


def historical_packet(path, reference, owner, node_count, *, frame=1):
    """Construct the old tagged format without reviving its training class."""
    scalar = QuantizedOpacityUpdate(OpacityUpdateConfig(enabled=True, mode='logit', step=.01))
    scalar.setup(torch.zeros(node_count * 3, 1))
    with torch.no_grad():
        first = int(owner[0])
        scalar.delta.reshape(-1, 3)[first] = torch.tensor([.01, -.02, .03])
    scalar.export(path, frame)
    return REGIONAL_MAGIC + _regional_digest(reference, owner, node_count) + path.read_bytes()


class HistoricalSharedDCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_packet_binds_owner_reference_and_frame_preserving_zero_bits(self):
        reference = torch.tensor([[[.1, .2, .3]], [[-0., .4, .5]], [[.6, .7, .8]]])
        owner = torch.tensor([1, 2, 1])
        with tempfile.TemporaryDirectory() as directory:
            packet = historical_packet(Path(directory)/'scalar.bin', reference, owner, 3)
        decoded = decode_feature_dc_delta(packet, reference, frame_index=1, owner=owner, node_count=3)
        delta = torch.tensor([[[.01, -.02, .03]], [[0., 0., 0.]], [[.01, -.02, .03]]])
        expected = torch.where(delta.ne(0), reference + delta, reference)
        self.assertTrue(torch.equal(decoded.view(torch.int32), expected.view(torch.int32)))
        self.assertTrue(torch.signbit(decoded[1, 0, 0]))
        wrong_owner = owner.clone(); wrong_owner[0] = 2
        for old, owners, frame, nodes in ((reference, None, 1, 3),
                (reference, wrong_owner, 1, 3), (reference + .001, owner, 1, 3),
                (reference, owner, 2, 3), (reference, owner, 1, 4)):
            with self.subTest(frame=frame, nodes=nodes), self.assertRaises(ValueError):
                decode_feature_dc_delta(packet, old, frame_index=frame, owner=owners, node_count=nodes)

    def test_survivors_and_births_replay_from_updated_shared_source_colours(self):
        raw = reference_raw()
        motion = motion_model(raw)
        prior = copy.deepcopy(motion.frame_reference_state)
        motion._select_frontier(freeze=True)
        rows = torch.tensor([1, 3, 4, 6, 7])
        motion.retain_survivors(rows)
        surviving = {name: value[rows].clone() for name, value in raw.items()}
        reference, owner = surviving['sh_dc'], motion.owner.detach().cpu()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packet = historical_packet(path/'scalar.bin', reference, owner, len(motion.parents))
            surviving['sh_dc'] = decode_feature_dc_delta(packet, reference, frame_index=1,
                owner=owner, node_count=len(motion.parents))
            expected = export_frame(path, motion, surviving, torch.tensor([0, 2]))
            (path/'shdc_001.bin').write_bytes(packet)
            (path/'frame_001.json').write_text(json.dumps(dict(
                feature_dc_update='shdc_001.bin', feature_dc_update_timing='before_births')))
            _, decoded = decode_frame(path, 1, prior, raw)
            for name in raw:
                self.assertTrue(torch.equal(decoded[name], expected[name].cpu()), name)

    def test_per_gs_checkpoint_without_legacy_sharing_field_restores(self):
        reference = torch.arange(12).float().reshape(4, 1, 3) * .1
        original = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True))
        original.setup(reference)
        packet = copy.deepcopy(original.capture())
        packet['config'].pop('sharing')
        restored = QuantizedFeatureDCUpdate(FeatureDCUpdateConfig(enabled=True))
        restored.restore(packet, torch.device('cpu'))
        self.assertTrue(torch.equal(original().view(torch.int32), restored().view(torch.int32)))

    def test_historical_sdr_bytes_are_included_in_complete_storage(self):
        reference = torch.zeros(2, 1, 3)
        owner = torch.tensor([1, 2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packet = historical_packet(path/'scalar.bin', reference, owner, 3)
            feature = path/'shdc_001.bin'
            feature.write_bytes(packet)
            (path/'motion_001.bin').write_bytes(b'motion')
            manifest = dict(motion='motion_001.bin', birth=None, feature_dc_update=feature.name)
            index = path/'frame_001.json'
            index.write_text(json.dumps(manifest))
            payload = 6 + len(packet)
            summary = dict(motion=dict(bytes=6), birth=None,
                feature_dc_update=dict(archive_bytes=len(packet)), encoded_bytes=payload,
                shared_reference_bytes=11, decoder_reference='reference_frame0.npz')
            measured = _actual_frame_storage(path, 1, manifest, summary, 11, 'reference_frame0.npz')
            self.assertEqual(measured['feature_dc_update_bytes'], len(packet))
            self.assertEqual(measured['complete_update_bytes'], payload + index.stat().st_size)
            feature.write_bytes(packet + b'x')
            with self.assertRaises(ValueError):
                _actual_frame_storage(path, 1, manifest, summary, 11, 'reference_frame0.npz')


if __name__ == "__main__":
    unittest.main()
