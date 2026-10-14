"""Independent CPU codec/view preparation; production imports remain untouched."""

import importlib
from pathlib import Path
import struct
import sys
from types import ModuleType
import unittest
import zlib

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline' / 'ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
OPACITY = importlib.import_module('pipeline.ReCon-GS.model.sparse_opacity')


def forge(packet, body=None, **updates):
    fields = list(OPACITY.HEADER.unpack_from(packet))
    if body is None:
        body = zlib.decompress(packet[OPACITY.HEADER.size:])
    for index, value in updates.items():
        fields[int(index)] = value
    fields[5], fields[6] = len(body), 0
    fields[6] = zlib.crc32(OPACITY.HEADER.pack(*fields) + body)
    return OPACITY.HEADER.pack(*fields) + zlib.compress(body, 3)


class FrozenModel:
    def __init__(self, reference):
        self.get_xyz = torch.zeros(len(reference), 3)
        self.cached_opacity = torch.sigmoid(reference)

    @property
    def get_opacity(self):
        return self.cached_opacity


class SparseOpacityTests(unittest.TestCase):
    def test_positive_negative_extreme_symbols_and_zero_update(self):
        reference = torch.tensor([[-2.], [1.], [0.], [10.], [-10.]], dtype=torch.float32)
        symbols = torch.tensor([[32767], [-32768], [0], [3], [-4]], dtype=torch.int16)
        packet = OPACITY.encode_opacity_delta(symbols, reference, step=.01, frame_index=12)
        decoded = OPACITY.decode_opacity_delta(packet, reference, frame_index=12)
        expected = reference + symbols.float() * OPACITY._step(.01)
        self.assertTrue(torch.equal(decoded.logits, expected))
        self.assertEqual(decoded.rows.tolist(), [0, 1, 3, 4])
        self.assertEqual(decoded.symbols.tolist(), [32767, -32768, 3, -4])
        self.assertEqual(decoded.frame_index, 12)
        self.assertIsNone(OPACITY.encode_opacity_delta(torch.zeros_like(reference), reference,
                                                      step=.01, frame_index=12))
        untouched = OPACITY.apply_optional_opacity_delta(None, reference)
        self.assertTrue(torch.equal(untouched, reference))
        self.assertNotEqual(untouched.data_ptr(), reference.data_ptr())

    def test_exact_ste_forward_random_and_large_delta_with_identity_gradient(self):
        generator = torch.Generator().manual_seed(129)
        model = OPACITY.SparseOpacityDelta(torch.zeros(10000, 1))
        with torch.no_grad():
            model.delta.copy_(torch.randn(10000, 1, generator=generator) * 1000)
        symbols = model.symbols(True)
        self.assertTrue(torch.equal(symbols, model.symbols(False)))
        self.assertTrue(torch.equal(symbols, symbols.round()))
        model().sum().backward()
        self.assertTrue(torch.equal(model.delta.grad, torch.ones_like(model.delta)))

    def test_view_gradients_adam_updates_and_checkpoint_are_exact(self):
        reference = torch.tensor([[2.], [-2.], [0.]], dtype=torch.float32)
        model = OPACITY.SparseOpacityDelta(reference)
        base = FrozenModel(reference)
        view = OPACITY.QuantizedOldOpacityView(base, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.05)
        targets = torch.tensor([[.2], [.8], [.5]])
        initial_loss = float((view.get_opacity.detach() - targets).square().mean())
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = (view.get_opacity - targets).square().mean()
            loss.backward()
            optimizer.step()
        self.assertLess(float(loss), initial_loss / 100)
        self.assertLess(float(model.delta[0]), 0)
        self.assertGreater(float(model.delta[1]), 0)
        self.assertTrue(torch.equal(base.get_opacity, torch.sigmoid(reference)))
        with torch.no_grad():
            actual = model(False)
            self.assertTrue(torch.equal(view.get_opacity, torch.sigmoid(actual)))
        packet = model.packet(8)
        decoded = OPACITY.decode_opacity_delta(packet, reference, frame_index=8)
        self.assertTrue(torch.equal(decoded.logits, actual))
        restored = OPACITY.SparseOpacityDelta(reference + 1, step=.2)
        restored.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(restored(False), actual))

    def test_signed_zero_reference_bits_and_hash_match_for_empty_and_mixed_packets(self):
        reference = torch.tensor([[-0.], [0.], [-0.], [-3.]])
        model = OPACITY.SparseOpacityDelta(reference)
        self.assertEqual(model().detach().numpy().tobytes(), reference.numpy().tobytes())
        self.assertEqual(model(False).numpy().tobytes(), reference.numpy().tobytes())
        model().sum().backward()
        self.assertTrue(torch.equal(model.delta.grad, torch.ones_like(model.delta)))
        with torch.no_grad():
            model.delta[1] = .2
            model.delta[3] = -.3
        decoded = OPACITY.decode_opacity_delta(model.packet(1), reference)
        self.assertEqual(decoded.logits.numpy().tobytes(), model(False).numpy().tobytes())
        self.assertEqual(OPACITY.opacity_reference_hash(decoded.logits),
                         OPACITY.opacity_reference_hash(model(False)))

    def test_same_motion_byte_price_prox_keeps_needed_change_and_zeros_weak_change(self):
        model = OPACITY.SparseOpacityDelta(torch.zeros(2, 1))
        optimizer = torch.optim.Adam([model.delta], lr=.05)
        model.delta.grad = torch.tensor([[1.], [.000001]])
        optimizer.step()
        with torch.no_grad():
            model.delta[:] = torch.tensor([[.1], [.01]])
        before = model.delta.detach().clone()
        result = model.proximal_zero(optimizer, byte_price=1e-6)
        self.assertEqual(result['proximal_zeroed_rows'], 1)
        self.assertEqual(float(model.delta[1]), 0.)
        self.assertEqual(float(model.delta[0]), float(before[0]))

    def test_wrong_prior_frame_count_corruption_and_trailers_fail(self):
        reference = torch.zeros(3, 1)
        packet = OPACITY.encode_opacity_delta(torch.ones(3, 1), reference,
                                             step=.01, frame_index=2)
        for bad_reference in (reference + .01, reference[:2]):
            with self.assertRaises(ValueError):
                OPACITY.decode_opacity_delta(packet, bad_reference)
        with self.assertRaises(ValueError):
            OPACITY.decode_opacity_delta(packet, reference, frame_index=3)
        for broken in (packet[:3], packet[:-1], packet + b'x',
                       packet[:OPACITY.HEADER.size] + b'bad'):
            with self.assertRaises(ValueError):
                OPACITY.decode_opacity_delta(broken, reference)
        fields = list(OPACITY.HEADER.unpack_from(packet))
        fields[6] ^= 1
        with self.assertRaises(ValueError):
            OPACITY.decode_opacity_delta(OPACITY.HEADER.pack(*fields) +
                                         packet[OPACITY.HEADER.size:], reference)

    def test_invalid_sparse_rows_with_valid_crc_are_rejected(self):
        reference = torch.zeros(3, 1)
        packet = OPACITY.encode_opacity_delta(torch.ones(3, 1), reference,
                                             step=.01, frame_index=2)
        # Three row gaps and int16 values: zero gap, out-of-range row, zero symbol.
        for body in (b'\x01\x01\x00\x00\x01\x00\x01\x01\x00',
                     b'\x01\x01\x00\x01\x01\x00\x02\x01\x00',
                     b'\x01\x00\x00\x01\x01\x00\x01\x01\x00'):
            with self.assertRaises(ValueError):
                OPACITY.decode_opacity_delta(forge(packet, body), reference)

    def test_born_predictor_keeps_preupdate_appearance_and_only_old_rows_change(self):
        birth = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_birth_codec')
        reference = dict(xyz=torch.zeros(2, 3), rotation=torch.tensor([[1., 0, 0, 0]] * 2),
                         logscale=torch.zeros(2, 3), opacity=torch.tensor([[1.], [-1.]]),
                         sh_dc=torch.zeros(2, 1, 3), sh_rest=torch.zeros(2, 3, 3))
        sources = torch.tensor([0])
        predictor = {key: reference[key][sources] for key in birth.ATTRIBUTES}
        symbols = {key: torch.zeros_like(value) for key, value in predictor.items()}
        symbols['opacity'][0, 0] = 2
        steps = {key: .01 for key in predictor}
        raw = {key: predictor[key] + symbols[key] * OPACITY._step(.01) for key in predictor}
        birth_packet = birth.encode_birth_symbols(raw, 2, symbols=symbols, steps=steps,
                                                  predictor=predictor, source_rows=sources)
        newborn = birth.decode_birth_attributes(birth_packet, reference,
                     torch.zeros(1, 3), torch.tensor([[1., 0, 0, 0]]), sources)
        opacity_packet = OPACITY.encode_opacity_delta(torch.tensor([[-100], [25]]),
                             reference['opacity'], step=.01, frame_index=2)
        combined = torch.cat([reference['opacity'], newborn['opacity']])
        combined[:2] = OPACITY.decode_opacity_delta(opacity_packet, combined[:2], frame_index=2).logits
        self.assertTrue(torch.equal(combined[2:], raw['opacity']))
        self.assertTrue(torch.equal(combined[:2], torch.tensor([[0.], [-.75]])))
        wrong_predictor = dict(reference, opacity=combined[:2])
        wrong_newborn = birth.decode_birth_attributes(birth_packet, wrong_predictor,
                    torch.zeros(1, 3), torch.tensor([[1., 0, 0, 0]]), sources)
        self.assertFalse(torch.equal(wrong_newborn['opacity'], raw['opacity']))

    def test_encoder_rejects_fractional_overflow_nonfinite_and_bad_types(self):
        reference = torch.zeros(1, 1)
        for symbols in (torch.tensor([[.5]]), torch.tensor([[32768.]]),
                        torch.tensor([[float('nan')]]), torch.tensor([[True]])):
            with self.assertRaises(ValueError):
                OPACITY.encode_opacity_delta(symbols, reference, step=.01, frame_index=1)
        for step in (True, 0., -1., float('inf'), 1e100, 1e-100):
            with self.assertRaises(ValueError):
                OPACITY.SparseOpacityDelta(reference, step)
        with self.assertRaises(ValueError):
            OPACITY.SparseOpacityDelta(reference.double())
        with self.assertRaises(ValueError):
            OPACITY.QuantizedOldOpacityView(FrozenModel(torch.zeros(2, 1)),
                                          OPACITY.SparseOpacityDelta(reference))


if __name__ == '__main__':
    unittest.main()
