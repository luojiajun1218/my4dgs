"""Deterministic CPU checks for the optional hierarchical step-size optimizer."""
import copy
import importlib.util
from pathlib import Path

import torch

path = Path(__file__).resolve().parents[1] / 'pipeline/ReCon-GS/model/row_scaled_adam.py'
spec = importlib.util.spec_from_file_location('row_scaled_adam', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RowScaledAdam = module.RowScaledAdam


def test_scaled_updates_and_restore():
    torch.manual_seed(81)
    value = torch.nn.Parameter(torch.randn(8, 6, dtype=torch.float64))
    optimizer = RowScaledAdam(value, 3, .2, lr=.001, eps=1e-15)
    for _ in range(4):
        previous = value.detach().clone()
        reference = torch.nn.Parameter(previous.clone())
        ordinary = torch.optim.Adam([reference], lr=.001, eps=1e-15)
        ordinary.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        gradient = torch.randn_like(value)
        value.grad, reference.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        ordinary.step()
        torch.testing.assert_close(value[:3], reference[:3], rtol=0, atol=0)
        torch.testing.assert_close(value[3:], previous[3:].lerp(reference[3:], .2),
                                   rtol=0, atol=0)
    saved = copy.deepcopy(optimizer.state_dict())
    clone = torch.nn.Parameter(value.detach().clone())
    restored = RowScaledAdam(clone, 3, .2, lr=.001, eps=1e-15)
    restored.load_state_dict(saved)
    gradient = torch.randn_like(value)
    value.grad, clone.grad = gradient.clone(), gradient.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(value, clone, rtol=0, atol=0)


def test_unit_scale_is_adam():
    torch.manual_seed(19)
    a = torch.nn.Parameter(torch.randn(5, 2))
    b = torch.nn.Parameter(a.detach().clone())
    optimizer = RowScaledAdam(a, 2, 1., lr=.003)
    ordinary = torch.optim.Adam([b], lr=.003)
    for _ in range(5):
        gradient = torch.randn_like(a)
        a.grad, b.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        ordinary.step()
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_closure():
    value = torch.nn.Parameter(torch.ones(4, 2))
    optimizer = RowScaledAdam(value, 2, .25, lr=.01)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = value.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert float(loss) == 8
    torch.testing.assert_close(value[:2], torch.full((2, 2), .99))
    torch.testing.assert_close(value[2:], torch.full((2, 2), .9975))


def test_old_adam_checkpoint():
    value = torch.nn.Parameter(torch.ones(4, 2))
    old = torch.optim.Adam([value], lr=.01)
    value.grad = torch.ones_like(value)
    old.step()
    state = copy.deepcopy(old.state_dict())
    new = RowScaledAdam(value, 2, .25, lr=.01)
    new.load_state_dict(state)
    assert 'first_scaled_row' not in state['param_groups'][0]
    before = value.detach().clone()
    new.step()
    torch.testing.assert_close(before[:2] - value[:2], torch.full((2, 2), .01))
    torch.testing.assert_close(before[2:] - value[2:], torch.full((2, 2), .0025))


if __name__ == '__main__':
    test_scaled_updates_and_restore()
    test_unit_scale_is_adam()
    test_closure()
    test_old_adam_checkpoint()
    print('RowScaledAdam deterministic tests passed')
