"""Checkpoint map_location must not move decoded codec state off CPU."""

import importlib
from io import BytesIO
from pathlib import Path
import sys
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DecodedState = importlib.import_module(
    'pipeline.ReCon-GS.model.rigid_tree_codec').DecodedState


def _state(frame):
    return DecodedState(
        xyz=torch.tensor([[0.25, -1., 2.]], dtype=torch.float32),
        rotation=torch.tensor([[1., 0., 0., 0.]], dtype=torch.float32),
        owner=torch.tensor([1], dtype=torch.int64),
        parents=torch.tensor([-1, 0], dtype=torch.int64),
        frame_index=frame,
        birth_sources=torch.tensor([0], dtype=torch.int32) if frame else None)


class DecodedStateRestoreTests(unittest.TestCase):
    def test_cpu_restore_preserves_values_dtypes_and_optional_birth_sources(self):
        for frame in (0, 49):
            with self.subTest(frame=frame):
                original = _state(frame)
                restored = DecodedState.restore(original.capture())
                self.assertEqual(restored.frame_index, frame)
                for field in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources'):
                    before, after = getattr(original, field), getattr(restored, field)
                    if before is None:
                        self.assertIsNone(after)
                    else:
                        self.assertEqual(after.device.type, 'cpu')
                        self.assertEqual(after.dtype, before.dtype)
                        torch.testing.assert_close(after, before, rtol=0, atol=0)
                self.assertEqual(restored.digest(), original.digest())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_map_location_restores_previous_and_reference_to_cpu(self):
        states = dict(codec_previous_state=_state(49).capture(),
                      frame_reference_state=_state(48).capture())
        buffer = BytesIO()
        torch.save(states, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location='cuda:0', weights_only=False)
        for name, packet in loaded.items():
            self.assertEqual(packet['parents'].device.type, 'cuda')
            restored = DecodedState.restore(packet)
            self.assertEqual(restored.frame_index,
                             49 if name == 'codec_previous_state' else 48)
            for field in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources'):
                value = getattr(restored, field)
                if value is not None:
                    self.assertEqual(value.device.type, 'cpu')
                    self.assertEqual(value.dtype, packet[field].dtype)
                    torch.testing.assert_close(value, packet[field].cpu(),
                                               rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
