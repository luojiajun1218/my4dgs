import random
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from scripts.first_frame_snapshot import (
    load_frame1_snapshot,
    save_frame1_snapshot,
)


class _SyntheticGaussianModel:
    """Small CPU model exposing the frame-one state used by the snapshot API."""

    def __init__(self, count=4, device="cpu"):
        self.param_names = ("anchor", "f_rest")
        self.max_sh_degree = 2
        self.frame_idx = 1
        self.active_sh_degree = 2
        self.spatial_lr_scale = 4.0
        self._latents = OrderedDict([
            ("anchor", nn.Parameter(
                torch.arange(count * 3, dtype=torch.float32, device=device)
                .reshape(count, 3), requires_grad=True)),
            ("f_rest", nn.Parameter(
                torch.arange(count * 2, dtype=torch.float32, device=device)
                .reshape(count, 2), requires_grad=False)),
        ])
        self.latent_decoders = OrderedDict([
            ("anchor", nn.Identity()),
            ("f_rest", nn.Linear(2, 3, bias=True).to(device)),
        ])
        self.prev_atts = OrderedDict([
            ("anchor", torch.full((count, 3), 0.25, device=device)),
            ("f_rest", torch.full((count, 3), 0.5, device=device)),
        ])
        self.prev_latents = OrderedDict([
            ("anchor", torch.full((count, 3), 0.75, device=device)),
            ("f_rest", torch.full((count, 2), 1.25, device=device)),
        ])
        self.prev_atts_initial = OrderedDict([
            ("anchor", torch.full((count, 3), 1.5, device=device)),
            ("f_rest", torch.full((count, 3), 2.0, device=device)),
        ])
        self.mask_anchor = torch.ones((count, 1), dtype=torch.bool, device=device)
        self.mask_offset = torch.zeros((count, 1), dtype=torch.bool, device=device)
        self.root_indices = torch.arange(count, device=device)
        self.positions = torch.arange(count * 3, dtype=torch.float32, device=device).reshape(count, 3)
        self.root_coord_to_idx = {(0, 1, 2): 0, (3, 4, 5): 1}
        self.coarse_intervals = np.asarray([3, 6, 9], dtype=np.int32)
        self.gate_atts = None
        self.shared_motion = None
        self.hierarchy_update_mask = None
        self.hierarchy_residual_gate = None
        self.current_frame_shared_bits = 0
        self.split_generator = torch.Generator(device=device).manual_seed(70)
        self.optimizer = object()  # Must not be serialized; frame two recreates it.
        self.latent_args = SimpleNamespace(config="construction only")
        self.model_args = SimpleNamespace(config="construction only")
        self.scaling_activation = torch.exp


class FirstFrameSnapshotTest(unittest.TestCase):
    def setUp(self):
        random.seed(101)
        np.random.seed(202)
        torch.manual_seed(303)

    def test_round_trip_restores_model_and_all_rng_streams(self):
        source = _SyntheticGaussianModel(count=4)
        source.latent_decoders["anchor"].eval()
        source.latent_decoders["f_rest"].weight.requires_grad_(False)
        source_camera_rng = random.Random(404)
        source_camera_rng.random()
        source.split_generator.manual_seed(505)
        torch.rand(2, generator=source.split_generator)

        expected_latents = {
            key: value.detach().clone() for key, value in source._latents.items()
        }
        expected_decoder = {
            key: {name: tensor.detach().clone()
                  for name, tensor in decoder.state_dict().items()}
            for key, decoder in source.latent_decoders.items()
        }
        expected_previous = {
            field: {key: value.detach().clone()
                    for key, value in getattr(source, field).items()}
            for field in ("prev_atts", "prev_latents", "prev_atts_initial")
        }
        expected_split_state = source.split_generator.get_state().clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "frame1.pt"
            info = save_frame1_snapshot(checkpoint, source, source_camera_rng)
            self.assertEqual(info["num_gaussians"], 4)

            expected_next = (
                random.random(),
                source_camera_rng.random(),
                float(np.random.random()),
                torch.rand(3),
                torch.rand(3, generator=source.split_generator),
            )

            # Use a different initial Gaussian count to ensure dynamic tensors
            # and latents are resized from the snapshot, not left at init size.
            destination = _SyntheticGaussianModel(count=1)
            destination.active_sh_degree = 0
            destination.mask_offset.fill_(True)
            destination_camera_rng = random.Random(999)
            result = load_frame1_snapshot(
                checkpoint, destination, destination_camera_rng)

        self.assertEqual(result["num_gaussians"], 4)
        self.assertTrue(result["rng_restored"])
        self.assertIsNone(destination.optimizer)
        self.assertEqual(destination._latents["anchor"].requires_grad, True)
        self.assertEqual(destination._latents["f_rest"].requires_grad, False)
        self.assertFalse(destination.latent_decoders["anchor"].training)
        self.assertFalse(destination.latent_decoders["f_rest"].weight.requires_grad)
        self.assertEqual(destination.active_sh_degree, 2)
        for key, expected in expected_latents.items():
            torch.testing.assert_close(destination._latents[key], expected)
        for key, decoder_state in expected_decoder.items():
            restored_state = destination.latent_decoders[key].state_dict()
            self.assertEqual(tuple(restored_state), tuple(decoder_state))
            for name, expected in decoder_state.items():
                torch.testing.assert_close(restored_state[name], expected)
        for field, expected_values in expected_previous.items():
            for key, expected in expected_values.items():
                torch.testing.assert_close(getattr(destination, field)[key], expected)
        self.assertEqual(destination.mask_offset.shape, (4, 1))
        self.assertFalse(torch.any(destination.mask_offset))
        torch.testing.assert_close(destination.positions, source.positions)
        self.assertEqual(destination.root_coord_to_idx, source.root_coord_to_idx)
        np.testing.assert_array_equal(destination.coarse_intervals, source.coarse_intervals)
        self.assertTrue(torch.equal(
            destination.split_generator.get_state(), expected_split_state))

        actual_next = (
            random.random(),
            destination_camera_rng.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.rand(3, generator=destination.split_generator),
        )
        self.assertEqual(actual_next[:3], expected_next[:3])
        torch.testing.assert_close(actual_next[3], expected_next[3])
        torch.testing.assert_close(actual_next[4], expected_next[4])

    def test_save_rejects_incomplete_or_non_frame_one_state(self):
        model = _SyntheticGaussianModel()
        model.prev_atts["anchor"] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.pt"
            with self.assertRaisesRegex(ValueError, r"prev_atts\['anchor'\]"):
                save_frame1_snapshot(path, model, random.Random(0))

            model.prev_atts["anchor"] = torch.zeros((4, 3))
            model.frame_idx = 2
            with self.assertRaisesRegex(ValueError, "frame_idx=2"):
                save_frame1_snapshot(path, model, random.Random(0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional")
    def test_cuda_rng_is_restored_when_available(self):
        torch.cuda.manual_seed_all(606)
        source = _SyntheticGaussianModel(device="cuda:0")
        source_camera_rng = random.Random(707)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "frame1_cuda.pt"
            save_frame1_snapshot(checkpoint, source, source_camera_rng)
            expected = torch.rand(4, device="cuda:0")

            destination = _SyntheticGaussianModel(count=1, device="cuda:0")
            load_frame1_snapshot(
                checkpoint, destination, random.Random(999))
            actual = torch.rand(4, device="cuda:0")
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
