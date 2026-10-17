"""Joint motion fitting, causal warm starts, and deterministic far-point routing."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import DecodedState, decode_motion
from pipeline.model.spatial_gate import (SPATIAL_ROUTING_FIELDS, spatial_owner,
    spatial_route, spatial_state_route)


class JointSpatialMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, initial_logit=-2.):
        axis = torch.linspace(-1., 1., 8)
        xyz = torch.cat((torch.cartesian_prod(axis, axis, axis),
                         torch.tensor([[-100., -100., -100.], [100., 100., 100.]])))
        q = torch.zeros(len(xyz), 4)
        q[:, 0] = 1.
        config = RigidTreeConfig(motion_xyz_step_extent=.001, gs_xyz_step_extent=.001,
            motion_rotation_step_rad=.001, gs_rotation_step_rad=.001,
            rate_loss_weight=.002, relative_motion_loss_weight=.01,
            rate_warmup_steps=0, rate_ramp_steps=1, motion_fit_window_steps=20)
        gate = config.spatial_gate
        gate.max_depth = 3
        gate.initialization = 'robust_median'
        gate.initialization_distance_quantile = .99
        gate.outlier_assignment = 'nearest_region'
        gate.training_schedule = 'joint'
        gate.learn_region_motion = True
        gate.min_region_size = 2
        gate.initial_logit = initial_logit
        gate.start_step = gate.gate_start_step = 0
        gate.freeze_step = 70
        config.validate()
        model = SpatialGatedMotion(config)
        model.setup(xyz, q, 2., frame_index=1)
        return model

    def commit(self, model, births=None, sources=None):
        reference = copy.deepcopy(model.frame_reference_state)
        xyz, q = model.materialized_pose()
        sources = torch.empty(0, dtype=torch.long) if sources is None else sources
        if births is not None:
            xyz = torch.cat((xyz, births))
            q = torch.cat((q, q[sources]))
        model.register_materialized_layout(xyz, q, sources)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.bin'
            model.export_motion(path)
            decoded = decode_motion(path.read_bytes(), reference)
        torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)
        torch.testing.assert_close(decoded.rotation, q, rtol=0, atol=1e-6)
        return decoded

    def test_far_points_use_nearest_supported_leaf_without_changing_geometry(self):
        model = self.model()
        reference = model.frame_reference_state
        query = torch.tensor([[100., 100., 100.], [-100., -100., -100.],
                              [200., -200., 100.]])
        distances = (query.double()[:, None] - reference.spatial_leaf_centers.double()[None]).square().sum(-1)
        expected = reference.spatial_leaf_nodes[distances.argmin(1)]
        paths = spatial_state_route(query, reference)
        torch.testing.assert_close(paths[:, -1], expected, rtol=0, atol=0)
        torch.testing.assert_close(reference.parents[paths[:, 1:]], paths[:, :-1])
        torch.testing.assert_close(model(1)[0], model.base_xyz, rtol=0, atol=0)
        self.assertEqual(len(model.owner), 514)
        inside = torch.tensor([[0., 0., 0.], [.1, -.2, .3]])
        torch.testing.assert_close(spatial_state_route(inside, reference),
            spatial_route(inside, reference.spatial_centers, 3), rtol=0, atol=0)

    def test_routing_reference_is_shared_once_restored_and_digest_bound(self):
        reference = self.model().frame_reference_state
        packet = {key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
                  for key, value in reference.capture().items() if value is not None}
        restored = DecodedState.from_reference(packet)
        self.assertEqual(restored.digest(), reference.digest())
        restored = DecodedState.restore(reference.capture())
        for name in SPATIAL_ROUTING_FIELDS:
            torch.testing.assert_close(getattr(restored, name), getattr(reference, name), rtol=0, atol=0)
        changed = copy.deepcopy(reference)
        changed.spatial_leaf_centers[0, 0] += .01
        self.assertNotEqual(changed.digest(), reference.digest())

    def test_first_dynamic_frame_closes_gates_and_both_motion_fields_learn(self):
        model = self.model()
        self.assertFalse(bool(model.opened[1:].any()))
        self.assertTrue(bool(model.opened[0]))
        self.assertEqual(len(model.owner.unique()), 8)
        self.assertTrue(bool(model.gate_logits.eq(-2.).all()))
        self.assertEqual(model.flat_region_diagnostics['initial_gates'], 'all_eligible_closed')
        self.assertFalse(model._hard_gate_frozen)
        xyz, q = model(1, training=True)
        loss = (xyz[:, 0] - model.base_xyz[:, 0] - .1).square().mean()
        loss = loss + (q[:, 1] - .01).square().mean()
        loss.backward()
        self.assertIsNone(model.gate_logits.grad)
        self.assertIsNone(model.region_translation.grad)
        self.assertIsNone(model.region_rotation.grad)
        model.optimizer_step()
        for step in (2, 3):
            xyz, q = model(step, training=True)
            loss = (xyz[:, 0] - model.base_xyz[:, 0] - .1).square().mean()
            loss = loss + (q[:, 1] - .01).square().mean() + model.regularization_loss(step)
            loss.backward()
            if step == 2:
                model.optimizer_step()
        for parameter in (model.region_translation, model.region_rotation,
                          model.gs_translation, model.gs_rotation):
            self.assertTrue(parameter.requires_grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(model._candidate_initialized)

    def test_first_gate_frontier_uses_configured_initial_logit(self):
        model = self.model(initial_logit=2.)
        self.assertTrue(bool(model.opened[model.split_eligible].all()))
        self.assertTrue(bool(model.gate_logits.eq(2.).all()))
        self.assertEqual(model.flat_region_diagnostics['initial_gates'], 'all_eligible_open')

    def test_region_updates_change_residual_origin_without_changing_full_pose(self):
        model = self.model()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.region_translation_lr = .004
        with torch.no_grad():
            model.gs_translation[:, 0] = .03
        before = tuple(value.detach().clone() for value in model(3, training=True))
        model.regularization_loss(3).backward()
        model.optimizer_step()
        self.assertGreater(float(model.region_translation.abs().sum()), 0.)
        self.assertLess(float(model.gs_translation[:, 0].mean()), .03)
        for actual, expected in zip(model(3), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_joint_loss_learns_merges_and_splits_from_either_initialization(self):
        for varying_children in (False, True):
            for initial_logit in (-2., 2.):
                with self.subTest(varying_children=varying_children, initial_logit=initial_logit):
                    model = self.model(initial_logit=initial_logit)
                    parent = int(model.spatial_paths[0, 0])
                    target = model.base_xyz.clone()
                    if varying_children:
                        children = torch.where(model.parents.eq(parent) & model.node_counts.gt(0))[0]
                        for index, child in enumerate(children):
                            target[model.spatial_paths[:, 1].eq(child), 0] += .02 if index % 2 else -.02
                    else:
                        target[:, 0] += .02
                    for step in range(1, 71):
                        model.observations.fill_(1)
                        xyz, _ = model(step, training=True)
                        loss = .5 * ((xyz - target) / .01).square().mean()
                        loss = loss + model.regularization_loss(step)
                        loss.backward()
                        model.optimizer_step()
                    self.assertEqual(bool(model.opened[parent]), varying_children)
                    self.assertLess(float((model(70)[0] - target).square().mean()), 1e-6)

    def test_live_joint_gate_cost_includes_point_residuals(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.linspace(0., .01, len(model.parents))
        zero = model.loss_terms(1)
        with torch.no_grad():
            model.gs_translation[:, 0] = .1
        nonzero = model.loss_terms(1)
        self.assertGreater(float(nonzero['relative_motion']), float(zero['relative_motion']))
        self.assertGreater(float(nonzero['rate_bytes_surrogate']), float(zero['rate_bytes_surrogate']))
        model.regularization_loss(1).backward()
        self.assertGreater(float(model.gate_logits.grad.abs().sum()), 0.)
        self.assertGreater(float(model.gs_translation.grad.abs().sum()), 0.)
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)

    def test_gate_ste_has_discrete_cost_and_the_soft_binary_derivative(self):
        model = self.model()
        parent = int(model.owner[0])
        children = torch.where(model.parents.eq(parent))[0]
        with torch.no_grad():
            model.region_translation[parent, 0] = .004
            model.region_translation[children, 0] = .024
            model.gs_translation[model.owner.eq(parent), 0] = .02
            model.gate_logits[parent] = -.2
        captured = copy.deepcopy(model.capture())
        model._hard_gate_frozen = True
        terms = model.loss_terms(10, False)
        closed_cost = float(terms['rate_loss'] + terms['relative_motion_loss'])
        model._hard_gate_frozen = False
        with torch.no_grad():
            model.gate_logits[parent] = 2.
        model._select_frontier(freeze=True)
        terms = model.loss_terms(10, False)
        open_cost = float(terms['rate_loss'] + terms['relative_motion_loss'])
        model.restore(captured)
        probability = torch.sigmoid(model.gate_logits[parent] / model.config.spatial_gate.temperature)
        probabilities = torch.zeros_like(model.gate_logits)
        probabilities[parent] = probability
        with patch.object(model, '_frontier_probabilities', return_value=probabilities):
            terms = model.loss_terms(10, False)
            loss = terms['rate_loss'] + terms['relative_motion_loss']
            self.assertAlmostEqual(float(loss), closed_cost, places=9)
            actual = torch.autograd.grad(loss, model.gate_logits)[0][parent]
        expected = (open_cost - closed_cost) * probability * (1 - probability) / model.config.spatial_gate.temperature
        self.assertAlmostEqual(float(actual), float(expected), places=9)
        self.assertLess(float(actual), 0.)

    def test_closed_ancestor_does_not_block_descendant_gate_gradient(self):
        model = self.model()
        node = int(model.spatial_paths[0, 1])
        parent = int(model.parents[node])
        self.assertFalse(bool(model.opened[parent]))
        children = torch.where(model.parents.eq(node) & model.node_counts.gt(0))[0]
        with torch.no_grad():
            for index, child in enumerate(children):
                rows = model.spatial_paths[:, -1].eq(child)
                value = .02 if index % 2 else -.02
                model.region_translation[child, 0] = value
                model.gs_translation[rows, 0] = value
        terms = model.loss_terms(10, False)
        (terms['rate_loss'] + terms['relative_motion_loss']).backward()
        self.assertLess(float(model.gate_logits.grad[node]), 0.)

    def test_gate_switch_recodes_residuals_without_pose_jump(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)) % 7 * .002
            model.region_rotation[:, 2] = torch.arange(len(model.parents)) % 5 * .001
            model.gs_translation[:, 1] = .0073
        before = tuple(value.detach().clone() for value in model(5))
        for logit in (2., -2.):
            old_owner = model.owner.clone()
            with torch.no_grad():
                model.gate_logits[1] = logit
            model._select_frontier(freeze=False)
            self.assertTrue(bool(model.owner.ne(old_owner).any()))
            for actual, expected in zip(model(5), before):
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertGreater(float(model.gs_translation.abs().sum()), 0.)

    def test_step70_freezes_only_gates_and_keeps_point_values_and_optimizer_history(self):
        model = self.model()
        model.config.region_translation_lr = model.config.region_rotation_lr = 0.
        model.config.gs_translation_lr = model.config.gs_rotation_lr = 0.
        model.config.spatial_gate.learning_rate = 0.
        model.optimizer.param_groups[-1]['lr'] = 0.
        with torch.no_grad():
            model.gs_translation[:, 0] = .0373
            model.gs_rotation[:, 1] = .0023
        before = (model.gs_translation.detach().clone(), model.gs_rotation.detach().clone())
        xyz, q = model(70, training=True)
        (xyz.sum() + q.sum()).backward()
        model.optimizer_step()
        self.assertTrue(model._hard_gate_frozen)
        self.assertFalse(model._shared_field_active)
        for parameter, expected in zip((model.gs_translation, model.gs_rotation), before):
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
            self.assertTrue(parameter.requires_grad)
            self.assertIn('exp_avg', model.optimizer.state[parameter])
        gates = model.gate_logits.detach().clone()
        model.optimizer.param_groups[-1]['lr'] = .15
        xyz, _ = model(71, training=True)
        xyz.sum().backward()
        model.gate_logits.grad = torch.ones_like(model.gate_logits)
        model.optimizer_step()
        torch.testing.assert_close(model.gate_logits, gates, rtol=0, atol=0)

    def test_next_frame_inherits_full_motion_with_births_and_recomputed_owners(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[:, 0] = .01
            model.region_rotation[:, 2] = .002
            model.gs_translation[0, 1] = .0063
            model.gs_rotation[0, 0] = .0032
            model.gate_logits[1] = 3.25
            model.gate_logits[2] = -3.25
        model._select_frontier(freeze=False)
        previous_gates = model.gate_logits.detach().clone()
        total = tuple(value.detach().clone() for value in model._full_motion_symbols())
        source = torch.tensor([0], dtype=torch.long)
        decoded = self.commit(model, torch.tensor([[200., -200., 100.]]), source)
        captured = copy.deepcopy(model.capture())
        resumed = SpatialGatedMotion(copy.deepcopy(model.config))
        resumed.restore(captured)
        for current in (model, resumed):
            current.setup(decoded.xyz, decoded.rotation, 2., frame_index=2)
            for actual, previous in zip(current._full_motion_symbols(), total):
                expected = torch.cat((previous, previous[source]))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(current.spatial_paths,
                spatial_state_route(decoded.xyz, decoded), rtol=0, atol=0)
            torch.testing.assert_close(current.gate_logits, previous_gates, rtol=0, atol=0)
            self.assertTrue(bool(current.opened[1]))
            self.assertFalse(bool(current.opened[2]))
            self.assertEqual(current.flat_region_diagnostics['initial_gates'], 'previous_frame')
        torch.testing.assert_close(model.gate_logits, resumed.gate_logits, rtol=0, atol=0)

    def test_mid_joint_checkpoint_restores_trainable_residuals_and_rejects_old_schedule(self):
        model = self.model()
        captured = copy.deepcopy(model.capture())
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(captured)
        self.assertTrue(restored.gs_translation.requires_grad)
        self.assertTrue(restored.gs_rotation.requires_grad)
        self.assertTrue(restored._shared_field_active)
        old = copy.deepcopy(captured)
        old['motion_schedule'] = 'shared_field_zero_residual_v2'
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            restored.restore(old)
        old['motion_schedule'] = 'joint_motion_v1'
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            restored.restore(old)


if __name__ == '__main__':
    unittest.main()
