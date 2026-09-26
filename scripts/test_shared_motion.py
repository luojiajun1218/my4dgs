import unittest

import torch

from scripts.shared_motion import SharedMotion, rotation_vector_to_quaternion


class SharedMotionTest(unittest.TestCase):
    def make_motion(self):
        anchors = torch.tensor([
            [0.1, 0.1, 0.1], [0.2, 0.1, 0.1],
            [1.1, 0.1, 0.1], [1.2, 0.1, 0.1],
        ])
        roots = torch.tensor([0, 0, 1, 1])
        return SharedMotion(
            anchors, roots, torch.zeros(3), 1.0, 2,
            levels=6, spatial_levels=3,
            lambda_deform=1.0, lambda_sparse=1e-3,
        )

    def test_six_levels_start_at_identity(self):
        motion = self.make_motion()
        translation, rotation = motion()
        self.assertEqual(len(motion.node_counts), 6)
        torch.testing.assert_close(translation, torch.zeros(4, 3))
        torch.testing.assert_close(rotation[:, 0], torch.ones(4))
        torch.testing.assert_close(rotation[:, 1:], torch.zeros(4, 3))
        translation.sum().backward()
        for level in range(6):
            self.assertTrue(torch.any(motion.translations[level].grad != 0))

    def test_four_levels_keep_coarse_prefix_through_root(self):
        six = self.make_motion()
        four = SharedMotion(
            torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1],
                          [1.1, 0.1, 0.1], [1.2, 0.1, 0.1]]),
            torch.tensor([0, 0, 1, 1]), torch.zeros(3), 1.0, 2,
            levels=4, spatial_levels=3)
        self.assertEqual(four.spatial_level_indices, [0, 1, 2, 3])
        self.assertEqual(four.node_counts, six.node_counts[:4])
        for level, spatial_level in enumerate(four.spatial_level_indices):
            torch.testing.assert_close(
                getattr(four, f"point_to_node_{level}"),
                getattr(six, f"point_to_node_{spatial_level}"))

    def test_shared_root_motion_and_gradients(self):
        motion = self.make_motion()
        with torch.no_grad():
            motion.translations[3][0, 0] = 0.25
            motion.gate_logits[2].fill_(20.0)
        translation, _ = motion()
        torch.testing.assert_close(translation[:2, 0], torch.full((2,), 0.25))
        torch.testing.assert_close(translation[2:, 0], torch.zeros(2))
        loss = translation[0, 0] + motion.regularization()
        loss.backward()
        self.assertIsNotNone(motion.translations[0].grad)

    def test_rotation_and_payload(self):
        motion = self.make_motion()
        with torch.no_grad():
            motion.rotations[0][0, 2] = 0.2
        _, rotation = motion()
        self.assertGreater(rotation[0, 3].item(), 0)
        self.assertAlmostEqual(torch.linalg.vector_norm(rotation[0]).item(), 1.0, places=5)
        self.assertGreater(motion.storage_bits(), 0)
        self.assertEqual(len(motion.frame_payload()["levels"]), 6)
        torch.testing.assert_close(
            rotation_vector_to_quaternion(torch.zeros(1, 3)),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )


if __name__ == "__main__":
    unittest.main()
