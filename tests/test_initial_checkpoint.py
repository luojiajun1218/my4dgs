"""Static pruning seeds must remain trainable after CPU serialization."""
import io
import unittest

import torch

from scripts.compact_initial_model import cpu_tree


class InitialCheckpointTest(unittest.TestCase):
    def test_saved_parameter_keeps_adam_resume_behavior(self):
        parameter = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)

        def step(value, opt):
            (value - torch.tensor([0.7, 0.1])).square().sum().backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        step(parameter, optimizer)
        buffer = io.BytesIO()
        torch.save(cpu_tree(dict(parameter=parameter, optimizer=optimizer.state_dict())), buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        resumed = restored['parameter']
        self.assertIsInstance(resumed, torch.nn.Parameter)
        self.assertTrue(resumed.requires_grad)
        self.assertTrue(resumed.is_leaf)
        self.assertTrue(torch.equal(parameter, resumed))
        resumed_optimizer = torch.optim.Adam([resumed], lr=0.01)
        resumed_optimizer.load_state_dict(restored['optimizer'])
        step(parameter, optimizer)
        step(resumed, resumed_optimizer)
        self.assertTrue(torch.equal(parameter, resumed))

    def test_frozen_parameter_and_state_tensors_stay_frozen(self):
        frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        state = torch.tensor([2.0], requires_grad=True)
        copied = cpu_tree((frozen, [state]))
        self.assertIsInstance(copied[0], torch.nn.Parameter)
        self.assertFalse(copied[0].requires_grad)
        self.assertFalse(copied[1][0].requires_grad)
        self.assertTrue(torch.equal(copied[0], frozen))
        self.assertTrue(torch.equal(copied[1][0], state))


if __name__ == '__main__':
    unittest.main()
