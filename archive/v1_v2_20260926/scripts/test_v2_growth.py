import unittest
from types import SimpleNamespace

import torch

from scripts.v2_growth import (eligible_birth_mask,
                               freeze_existing_gaussians,
                               mask_existing_gradients,
                               restore_shared_decoders)


class V2GrowthTest(unittest.TestCase):
    def test_birth_requires_observations_and_level_gradient(self):
        config = SimpleNamespace(
            update_interval=100, success_threshold=0.8,
            densify_grad_threshold=0.0002, update_ratio=0.2)
        scores = torch.tensor([0.001, 0.0001, 0.001, float("nan")])
        observations = torch.tensor([41.0, 41.0, 40.0, 41.0])
        levels = torch.tensor([1, 1, 2, 1])
        torch.testing.assert_close(
            eligible_birth_mask(scores, observations, levels, config, 2),
            torch.tensor([True, False, False, False]))

    def test_next_frame_restores_shared_decoder_gradients(self):
        decoder = torch.nn.Linear(2, 2)
        model = SimpleNamespace(
            latent_decoders={"offset": decoder}, gate_atts=None)
        freeze_existing_gaussians(model)
        self.assertFalse(decoder.weight.requires_grad)
        restore_shared_decoders(model)
        self.assertTrue(decoder.weight.requires_grad)

    def test_stage_three_keeps_existing_latents_fixed(self):
        parameter = torch.nn.Parameter(torch.zeros(4, 2))
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        old_values = parameter[:2].detach().clone()

        parameter.grad = torch.ones_like(parameter)
        model = SimpleNamespace(
            _latents={"anchor": parameter}, optimizer=optimizer)
        mask_existing_gradients(model, old_count=2)
        optimizer.step()

        torch.testing.assert_close(parameter[:2], old_values)
        self.assertFalse(torch.equal(parameter[2:], old_values))


if __name__ == "__main__":
    unittest.main()
