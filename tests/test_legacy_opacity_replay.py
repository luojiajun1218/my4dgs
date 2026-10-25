"""Read historical opacity packets without retaining an opacity training path."""
import importlib
import unittest
import numpy as np
import torch

CODEC = importlib.import_module('pipeline.model.legacy_opacity_codec')
# Frozen OPD1/2/3 packets and their exact float32 logits from the prior codec.
FIXTURES = [('logit', '4f5044310200000004000000030000000ad7233c0900000094861248bea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '0ad7233c0000803f295c9f407b147ec0'), ('alpha', '4f5044320200000004000000030000000ad7233c09000000b5712892bea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '98dc233d0000803fdf296640eb353fc0'), ('saturated_alpha', '4f5044330200000004000000030000000ad7233c09000000aadcc1dbbea0a68b72d66d0206c86a4ccebab4bc903ef00efc8aba755b3fea852a2cd91e785e63646460faf79f9199010009210206', '0ad7233c0000803f0efc8d407b147ec0')]


class HistoricalOpacityReplayTests(unittest.TestCase):
    def reference(self):
        return torch.tensor([[0.], [1.], [5.], [-4.]], dtype=torch.float32)

    def test_all_historical_coordinate_modes_reconstruct_exact_logits(self):
        for mode, encoded, logits in FIXTURES:
            with self.subTest(mode=mode):
                actual = CODEC.decode_opacity_delta(bytes.fromhex(encoded), self.reference(), frame_index=2)
                expected = torch.from_numpy(np.frombuffer(bytes.fromhex(logits), dtype='<f4').copy()).reshape(-1, 1)
                torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
                self.assertEqual(actual.mode, mode)

    def test_corrupt_packets_are_rejected(self):
        for mode, encoded, _ in FIXTURES:
            with self.subTest(mode=mode):
                damaged = bytearray.fromhex(encoded)
                damaged[-1] ^= 1
                with self.assertRaises(ValueError):
                    CODEC.decode_opacity_delta(bytes(damaged), self.reference(), frame_index=2)

    def test_reference_and_frame_mismatches_are_rejected(self):
        encoded = bytes.fromhex(FIXTURES[0][1])
        with self.assertRaises(ValueError):
            CODEC.decode_opacity_delta(encoded, self.reference() + .01, frame_index=2)
        with self.assertRaises(ValueError):
            CODEC.decode_opacity_delta(encoded, self.reference(), frame_index=3)


if __name__ == '__main__':
    unittest.main()
