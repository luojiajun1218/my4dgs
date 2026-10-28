"""Static pruning seeds must remain trainable after CPU serialization."""
import io
from types import SimpleNamespace
import unittest

import torch

from scripts.compact_initial_model import cpu_tree, coverage_selection
from scripts.run_first_frame import (COLOUR_FIELDS, STATIC_FIELDS,
    configure_static_parameters, static_field_digests, verify_colour_only)


class InitialCheckpointTest(unittest.TestCase):
    def test_coverage_keeps_best_available_points_in_every_region(self):
        scores = torch.tensor([9., 8., 7., 6., 5., 4., .1, .0])
        groups = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        selected = coverage_selection(scores, groups, 6)
        self.assertEqual(selected.tolist(), [0, 1, 4, 5, 6, 7])
        self.assertEqual(torch.bincount(groups[selected]).tolist(), [2, 2, 2])

    def test_coverage_ties_are_stable_and_remaining_budget_uses_global_rank(self):
        scores = torch.ones(7)
        groups = torch.tensor([2, 0, 1, 0, 1, 0, 1])
        selected = coverage_selection(scores, groups, 6)
        self.assertEqual(selected.tolist(), [0, 1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, 'requires'):
            coverage_selection(scores, groups, 4)

    def make_model(self, *, parameters=True):
        model = SimpleNamespace()
        for name in STATIC_FIELDS:
            value = torch.tensor([0.2, -0.4])
            setattr(model, name, torch.nn.Parameter(value) if parameters else value)
        model.optimizer = torch.optim.Adam([getattr(model, name) for name in STATIC_FIELDS], lr=0.01)
        return model

    def test_colour_only_freezes_geometry_even_with_existing_adam_moments(self):
        model = self.make_model()
        sum(getattr(model, name).square().sum() for name in STATIC_FIELDS).backward()
        model.optimizer.step()
        before = static_field_digests(model)
        configure_static_parameters(model, colour_only=True)
        self.assertTrue(all(getattr(model, name).grad is None for name in STATIC_FIELDS))
        sum((getattr(model, name)-1).square().sum() for name in STATIC_FIELDS).backward()
        model.optimizer.step()
        after = static_field_digests(model)
        self.assertEqual(set(verify_colour_only(before, after)), set(COLOUR_FIELDS))

    def test_old_detached_seed_can_resume_native_full_fitting(self):
        model = self.make_model(parameters=False)
        configure_static_parameters(model, colour_only=False)
        before = static_field_digests(model)
        sum(getattr(model, name).square().sum() for name in STATIC_FIELDS).backward()
        model.optimizer.step()
        after = static_field_digests(model)
        self.assertTrue(all(before[name] != after[name] for name in STATIC_FIELDS))

    def test_colour_only_verifier_rejects_geometry_change_and_no_fit(self):
        model = self.make_model()
        before = static_field_digests(model)
        with self.assertRaisesRegex(ValueError, 'did not change'):
            verify_colour_only(before, before)
        changed = dict(before, _xyz='changed')
        with self.assertRaisesRegex(ValueError, 'frozen attributes'):
            verify_colour_only(before, changed)

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
