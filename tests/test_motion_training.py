"""Current shared discovery, sparse motion regularization, amplitude refits, and rotation policy."""
import copy
import unittest
import torch
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _motion_fixtures import current_motion, joint_motion, commit_motion, motion_packet
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.rigid_tree_codec import decode_motion, uncompressed_motion_packet, compress_motion_packet
from pipeline.model.sparse_integer_codec import encode_motion_layout, pack_layout, MOTION_MAGICS
from pipeline.model.rigid_residual_tree import MotionBase, RigidTreeConfig, _quantized, rotation_vector_quaternion
from pipeline.model.sparse_updates import sparse_block_support
from _stream_fixtures import reference_raw, motion_model


class SharedDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_same_model(self, left, right):
        for name in ('region_translation', 'region_rotation', 'gs_translation',
                     'gs_rotation', 'gate_logits', 'owner', 'opened'):
            torch.testing.assert_close(getattr(left, name), getattr(right, name), rtol=0, atol=0)
        self.assertTrue(torch.equal(left._gate_rng.get_state(), right._gate_rng.get_state()))
        self.assertEqual(left._hard_gate_frozen, right._hard_gate_frozen)
        self.assertEqual(left.amplitude_refit_active, right.amplitude_refit_active)
        self.assertEqual(motion_packet(left), motion_packet(right))

    def test_formal_recipe_uses_shared_discovery_without_private_or_auxiliary_proposals(self):
        model = current_motion()
        config, gate = model.config, model.config.spatial_gate
        self.assertEqual((config.motion_steps, config.birth_steps, config.amplitude_refit_steps),
                         (180, 120, 20))
        self.assertEqual((gate.gate_start_step, gate.freeze_step), (20, 70))
        self.assertTrue(gate.shared_discovery and gate.shared_image_recode)
        self.assertEqual((gate.region_motion, gate.image_relaxation), ('relative', 'stochastic_ste'))
        self.assertFalse(gate.hierarchical_private or gate.private_full_motion or
                         gate.reparameterize_after_gate_freeze or config.rotation_updates)
        self.assertEqual((gate.proposal_image_weight, gate.predictor_image_weight,
                          gate.candidate_refresh_interval), (0., 0., 0))
        self.assertEqual(config.rate_model, 'raw_blocks')
        for field in ('shared_image_recode', 'joint_image_gradients'):
            invalid = copy.deepcopy(config)
            setattr(invalid.spatial_gate, field, False)
            with self.subTest(field=field), self.assertRaises(ValueError):
                invalid.validate()

    def test_one_discrete_path_is_reused_across_views_without_changing_the_stream_or_global_rng(self):
        model = current_motion(8, max_depth=3, amplitude_refit_steps=0)
        with torch.no_grad():
            model.region_translation[:, 0] = torch.arange(len(model.parents)).remainder(7) * .002
        packet = motion_packet(model)
        global_rng = torch.get_rng_state().clone()
        xyz, rotation = model(21, training=True)
        noise, state = model._sampled_gate_noise.clone(), model._gate_rng.get_state().clone()
        second, _ = model(21, training=True)
        torch.testing.assert_close(xyz, second, rtol=0, atol=0)
        choices = model.base_xyz[:, None] + model._regional_predictions()[0][model.spatial_paths]
        self.assertEqual(float((xyz[:, None] - choices).abs().amax(-1).amin(-1).max()), 0.)
        torch.testing.assert_close(rotation, model.base_rotation, rtol=0, atol=0)
        self.assertTrue(torch.equal(state, model._gate_rng.get_state()))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        model(22, training=True)
        self.assertFalse(torch.equal(noise, model._sampled_gate_noise))
        self.assertEqual(motion_packet(model), packet)
        decoded = decode_motion(packet, model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model.materialized_pose()[0], rtol=0, atol=0)

    def test_image_gradients_fit_parent_and_child_while_point_values_wait_for_freezing(self):
        model = current_motion(8, max_depth=3, amplitude_refit_steps=0)
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.region_translation[9:73, 0] = .004
        model._sampled_gate_step = 21
        model._sampled_gate_noise = torch.full_like(model.gate_logits, -.1)
        xyz, _ = model(21, training=True)
        loss = (xyz - model.base_xyz - .02).square().mean()
        point, region, gate = torch.autograd.grad(loss,
            [model.gs_translation, model.region_translation, model.gate_logits], retain_graph=True)
        self.assertTrue(bool(point.ne(0).all()))
        self.assertGreater(float(region[1:9].abs().sum()), 0.)
        self.assertEqual(float(region[9:].abs().sum()), 0.)
        self.assertTrue(bool(gate[1:9].lt(0).all()))
        regularizers = model.regularization_loss(21)
        self.assertIsNone(torch.autograd.grad(regularizers, model.gs_translation, allow_unused=True)[0])
        (loss + model.regularization_loss(21)).backward()
        model.optimizer_step()
        self.assertTrue(bool(model.gs_translation.eq(0).all()))
        self.assertGreater(float(model.optimizer.state[model.gs_translation]['exp_avg_sq'].sum()), 0.)
        self.assertFalse(model.hierarchical_private or model.private_full_motion)

    def test_all_motion_steps_and_boundary_checkpoints_resume_identical_updates_and_decoded_pose(self):
        model = current_motion()
        target = model.base_xyz.clone()
        target[:, 0] += torch.where(target[:, 1] > 0, .006, -.003)
        target[:, 2] += torch.where(target[:, 0] > 0, .002, -.001)
        twin = None
        frozen_gates = None
        support = None
        refit_start = model.config.motion_steps - model.config.amplitude_refit_steps + 1
        for step in range(1, model.motion_steps + 1):
            for current in (model,) if twin is None else (model, twin):
                xyz, rotation = current(step, training=True)
                loss = (xyz - target).square().mean() + current.regularization_loss(step)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                current.optimizer_step()
                torch.testing.assert_close(rotation, current.base_rotation, rtol=0, atol=0)
            if step < 70:
                self.assertTrue(model.shared_discovery_active)
                self.assertTrue(bool(model.gs_translation.eq(0).all()))
                if step <= 20:
                    self.assertTrue(bool(model.gate_logits.eq(model.config.spatial_gate.initial_logit).all()))
            elif step == 70:
                self.assertFalse(model.shared_discovery_active)
                self.assertTrue(model._hard_gate_frozen)
                frozen_gates = model.gate_logits.detach().clone()
                self.assertGreater(model.optimizer.param_groups[2]['lr'], 0.)
                moments = model.optimizer.state[model.gs_translation]
                self.assertTrue(bool(moments['exp_avg'].eq(0).all()))
                self.assertGreater(float(moments['exp_avg_sq'].sum()), 0.)
            if step >= 70:
                torch.testing.assert_close(model.gate_logits, frozen_gates, rtol=0, atol=0)
            self.assertEqual(model.amplitude_refit_active, step >= refit_start)
            if step == refit_start:
                support = {name: value.clone() for name, value in model._amplitude_refit_support.items()}
            elif step > refit_start:
                for name, expected in support.items():
                    self.assertTrue(torch.equal(model._amplitude_refit_support[name], expected))
            if twin is not None:
                self.assert_same_model(model, twin)
            if step in (20, 69, 70, refit_start - 1, refit_start,
                        model.motion_steps - 1, model.motion_steps):
                twin = SpatialGatedMotion(copy.deepcopy(model.config))
                twin.restore(copy.deepcopy(model.capture()))
                self.assert_same_model(model, twin)
        packet = motion_packet(model)
        decoded = decode_motion(packet, model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model.materialized_pose()[0], rtol=0, atol=0)
        torch.testing.assert_close(decoded.rotation.view(torch.int32),
                                   model.base_rotation.view(torch.int32), rtol=0, atol=0)
        self.assertGreater(float(model.region_translation.abs().sum()), 0.)

    def test_image_error_opens_a_useful_shared_split_without_any_private_capacity(self):
        model = current_motion(8, max_depth=3, amplitude_refit_steps=0)
        with torch.no_grad():
            model.region_translation[9:73, 0] = .02
        target = model.base_xyz.clone()
        target[:, 0] += .02
        initial = float((model.materialized_pose()[0] - target).square().mean())
        optimizer = torch.optim.Adam([model.gate_logits], lr=.3)
        for step in range(21, 70):
            xyz, _ = model(step, training=True)
            loss = ((xyz - target) / .01).square().mean() + model.regularization_loss(step)
            optimizer.zero_grad(set_to_none=True)
            model.gate_logits.grad = torch.autograd.grad(loss, model.gate_logits)[0]
            optimizer.step()
            model._select_frontier(freeze=False)
        self.assertTrue(bool(model.opened[1:9].all()))
        self.assertLess(float((model.materialized_pose()[0] - target).square().mean()), initial * .1)
        self.assertTrue(bool(model.gs_translation.eq(0).all()))

    def test_frozen_relative_frontier_keeps_point_exceptions_and_refit_cannot_grow_support(self):
        model = current_motion(8, max_depth=3)
        model.last_step = 70
        model._select_frontier(freeze=True)
        with torch.no_grad():
            model.region_translation[1:9, 0] = .006
            model.gs_translation[0, 0] = .004
        before = motion_packet(model)
        model(181, training=True)
        self.assertEqual(motion_packet(model), before)
        support = model._amplitude_refit_support['gs_translation'].clone()
        self.assertTrue(bool(support[0]))
        self.assertFalse(bool(support[1:].any()))
        xyz, _ = model(182, training=True)
        ((xyz - model.base_xyz - .02).square().mean()).backward()
        model.optimizer_step()
        self.assertTrue(bool(model.gs_translation[~support].eq(0).all()))
        self.assertEqual(float(model.regularization_loss(182)), 0.)
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        self.assert_same_model(model, restored)
        decoded = decode_motion(motion_packet(model), model.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, model.materialized_pose()[0], rtol=0, atol=0)

    def test_next_frame_inherits_decisions_with_decay_after_births_and_rejects_missing_sampler(self):
        model = current_motion(8, max_depth=3, amplitude_refit_steps=0)
        with torch.no_grad():
            model.gate_logits[1:9] = 2.
            model.region_translation[1:9, 0] = .006
        model.last_step = 70
        model._select_frontier(freeze=True)
        prior_gates = model.gate_logits.detach().clone()
        decoded = commit_motion(model, births=model.base_xyz[:1], sources=torch.tensor([0]))
        model.setup(decoded.xyz, decoded.rotation, model.extent, frame_index=2)
        torch.testing.assert_close(model.gate_logits,
            prior_gates * model.config.spatial_gate.temporal_logit_decay, rtol=0, atol=0)
        self.assertTrue(torch.equal(model.opened, decoded.spatial_open))
        self.assertTrue(bool(model.gs_translation.eq(0).all()))
        model(21, training=True)
        saved = copy.deepcopy(model.capture())
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(saved)
        self.assert_same_model(model, restored)
        saved.pop('image_gate_sampler')
        with self.assertRaises(ValueError):
            restored.restore(saved)


class AmplitudeRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_motion()
        model.config.amplitude_refit_steps = 20
        model.config.proximal_gs_regularization = True
        model._hard_gate_frozen = True
        with torch.no_grad():
            model.region_translation[model.owner[0], 0] = 3 * model.steps[0]
            model.gs_translation[0, 0] = 2 * model.gs_steps[0]
            model.gs_translation[1, 0] = .2 * model.gs_steps[0]
        return model

    def test_refit_preserves_forward_then_updates_selected_blocks_from_images(self):
        model = self.model()
        before = model(80)[0].detach().clone()
        xyz, _ = model(81, training=True)
        torch.testing.assert_close(xyz, before, rtol=0, atol=0)
        self.assertTrue(model.amplitude_refit_active)
        self.assertEqual(float(model.regularization_loss(81)), 0.)
        support = model._amplitude_refit_support
        region_before = model.region_translation.detach().clone()
        gs_before = model.gs_translation.detach().clone()
        gates = model.opened.clone()
        ((xyz - model.base_xyz - .03).square().mean()).backward()
        self.assertGreater(float(model.region_translation.grad.abs().sum()), 0.)
        model.optimizer_step()
        self.assertTrue(torch.equal(model.opened, gates))
        self.assertGreater(float((model.region_translation - region_before).abs().sum()), 0.)
        self.assertGreater(float((model.gs_translation - gs_before).abs().sum()), 0.)
        self.assertFalse(bool(model.gs_translation[~support['gs_translation']].ne(0).any()))
        torch.testing.assert_close(model.region_translation[~support['region_translation']],
            region_before[~support['region_translation']], rtol=0, atol=0)

    def test_refit_masks_adam_momentum_and_survives_resume_and_retirement(self):
        model = self.model()
        p = model.gs_translation
        model.optimizer.state[p] = dict(step=torch.tensor(1.), exp_avg=torch.ones_like(p),
            exp_avg_sq=torch.ones_like(p))
        model(81, training=True)
        self.assertFalse(bool(model.optimizer.state[p]['exp_avg'][1:].ne(0).any()))
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        for current in (model, restored):
            xyz, _ = current(82, training=True)
            ((xyz - current.base_xyz - .03).square().mean()).backward()
            current.optimizer_step()
            current.retain_survivors(torch.arange(len(current.owner))[::2])
            self.assertEqual(len(current._amplitude_refit_support['gs_translation']), len(current.owner))
        torch.testing.assert_close(model.gs_translation, restored.gs_translation, rtol=0, atol=0)
        torch.testing.assert_close(model.region_translation, restored.region_translation, rtol=0, atol=0)

    def test_refit_requires_frozen_gates_and_valid_schedule(self):
        model = self.model()
        model._hard_gate_frozen = False
        with self.assertRaises(ValueError):
            model(81, training=True)
        for count in (True, -1, 31, 100):
            model.config.amplitude_refit_steps = count
            with self.assertRaises(ValueError):
                model.config.validate()


class ProximalGSRegularizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_motion()
        model.config.proximal_gs_regularization = True
        model.config.differentiable_rate_support = True
        return model

    def test_regularizers_train_regions_and_gates_but_not_gs_adam_moments(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:, 0] = .6 * model.steps[0]
            model.region_rotation[1:, 1] = .6 * model.steps[3]
            model.gs_translation[:, 0] = .7 * model.gs_steps[0]
            model.gs_rotation[:, 1] = .8 * model.gs_steps[3]
        terms = model.loss_terms(5, False)
        gradients = torch.autograd.grad(terms['rate_loss'] + terms['relative_motion_loss'],
            [model.gs_translation, model.gs_rotation, model.region_translation,
             model.region_rotation, model.gate_logits], allow_unused=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[1])
        for gradient in gradients[2:]:
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_soft_threshold_is_exact_metric_prox_and_cannot_cross_zero(self):
        model = self.model()
        model.config.relative_motion_loss_weight = 1.
        model.last_step = 1
        for p, group, scale in [(model.gs_translation, model.optimizer.param_groups[2], model.extent),
                                (model.gs_rotation, model.optimizer.param_groups[3], 1.)]:
            group['lr'] = .01
            model.optimizer.state[p] = dict(step=torch.tensor(1.),
                exp_avg=torch.zeros_like(p), exp_avg_sq=torch.full_like(p, 1-model.config.adam_betas[1]))
            with torch.no_grad():
                p.zero_()
                p[0] = torch.tensor([1e-9, -1e-9, 0.])
                p[1] = torch.tensor([.000501, -.000501, .2])
        original_t, original_r = model.gs_translation.detach().clone(), model.gs_rotation.detach().clone()
        model._proximal_relative_gs()
        for original, p, scale in [(original_t, model.gs_translation, model.extent),
                                   (original_r, model.gs_rotation, 1.)]:
            threshold = .01 / (len(model.owner) * scale * (1 + model.config.adam_eps))
            expected = original.sign() * (original.abs() - threshold).clamp_min(0)
            torch.testing.assert_close(p, expected, rtol=1e-6, atol=1e-10)
            self.assertTrue(bool((p * original >= 0).all()))
            self.assertTrue(bool(p.abs().le(original.abs()).all()))
            self.assertTrue(bool(p[0].eq(0).all()))
        self.assertEqual(float(_quantized(model.gs_rotation, model.gs_steps[3:])[1,0]), 0.)

    def test_one_real_update_stores_only_photometric_gs_second_moment(self):
        model = self.model()
        with torch.no_grad():
            model.gs_translation[:, 0] = torch.linspace(-.003,.003,len(model.owner))
            model.gs_rotation[:, 1] = torch.linspace(-.002,.002,len(model.owner))
        xyz, rotation = model(1, training=True)
        image_proxy = (xyz-model.base_xyz-.0001).square().mean() + rotation[:,1:].square().mean()
        expected = torch.autograd.grad(image_proxy,[model.gs_translation,model.gs_rotation],retain_graph=True)
        terms = model.loss_terms(1,False)
        (image_proxy+terms['rate_loss']+terms['relative_motion_loss']).backward()
        model.optimizer_step()
        for p,gradient in zip((model.gs_translation,model.gs_rotation),expected):
            torch.testing.assert_close(model.optimizer.state[p]['exp_avg_sq'],
                gradient.square()*(1-model.config.adam_betas[1]),rtol=1e-6,atol=1e-20)

    def test_metric_and_shrink_survive_checkpoint_restore(self):
        model = self.model()
        with torch.no_grad():
            model.gs_rotation[:,0]=.000501
        model.last_step=1
        model.gs_rotation.grad=torch.full_like(model.gs_rotation,1e-6)
        model.optimizer_step()
        captured=copy.deepcopy(model.capture())
        restored=SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(captured)
        self.assertTrue(restored.proximal_gs_regularization)
        for a,b in [(model.gs_translation,restored.gs_translation),(model.gs_rotation,restored.gs_rotation)]:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        for current in [model,restored]:
            current.last_step=2
            current.gs_rotation.grad=torch.full_like(current.gs_rotation,2e-6)
            current.optimizer_step()
        torch.testing.assert_close(model.gs_rotation,restored.gs_rotation,rtol=0,atol=0)


class ByteBudgetRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        model = joint_motion()
        model.config.rate_normalization = 'budget'
        model.config.rate_budget_bytes = 50000
        model.config.rate_loss_weight = 3e-5
        with torch.no_grad():
            model.region_translation[1:, 0] = model.steps[0]
            model.gs_translation[0, 0] = 2 * model.gs_steps[0]
        model.config.validate()
        return model

    def test_identical_packet_has_identical_loss_at_different_reference_counts(self):
        model = self.model()
        model.rate_reference_count = 10000
        first = model.loss_terms(30)
        model.rate_reference_count = 1000000
        second = model.loss_terms(30)
        torch.testing.assert_close(first['rate_loss'], second['rate_loss'], rtol=0, atol=0)
        self.assertAlmostEqual(float(second['rate_byte_price']), 6e-10, places=16)
        self.assertEqual(float(second['rate_normalization_bytes']), 50000.)
        # Legacy reference normalization remains explicitly density dependent.
        model.config.rate_normalization = 'reference'
        model.config.rate_budget_bytes = None
        before = model.loss_terms(30)['rate_loss']
        model.rate_reference_count //= 10
        after = model.loss_terms(30)['rate_loss']
        torch.testing.assert_close(after, 10 * before)

    def test_budget_and_gradients_survive_checkpoint_restore(self):
        model = self.model()
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        self.assertEqual(restored.rate_normalization_bytes, 50000)
        terms = restored.loss_terms(30)
        terms['rate_loss'].backward()
        self.assertIsNotNone(restored.gate_logits.grad)
        self.assertGreater(float(restored.gate_logits.grad.abs().sum()), 0.)
        torch.testing.assert_close(terms['rate_loss'], model.loss_terms(30)['rate_loss'])

    def test_budget_is_explicit_and_validated(self):
        model = self.model()
        for invalid in (None, 0, -1, True, 1.5, float('nan')):
            with self.subTest(invalid=invalid):
                model.config.rate_budget_bytes = invalid
                with self.assertRaisesRegex(ValueError, 'rate_budget_bytes'):
                    model.config.validate()
        model.config.rate_budget_bytes = 50000
        model.config.rate_normalization = 'reference'
        with self.assertRaisesRegex(ValueError, 'budget normalization'):
            model.config.validate()


class TranslationOnlyMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        source = joint_motion()
        config = copy.deepcopy(source.config)
        config.rotation_updates = False
        config.proximal_gs_regularization = True
        model = SpatialGatedMotion(config)
        rotation = rotation_vector_quaternion(torch.full_like(source.base_xyz, .2))
        model.setup(source.base_xyz, rotation, source.extent, frame_index=1)
        return model

    def test_translation_learns_with_no_rotation_parameters_or_adam_moments(self):
        model = self.model()
        xyz, orientation = model(1, training=True)
        image = (xyz - model.base_xyz - .01).square().mean()
        terms = model.loss_terms(1, False)
        (image + terms['rate_loss'] + terms['relative_motion_loss']).backward()
        model.optimizer_step()
        self.assertIsNotNone(model.optimizer.state.get(model.gs_translation))
        for parameter in (model.region_rotation, model.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertNotIn(parameter, model.optimizer.state)
            self.assertTrue(bool(parameter.eq(0).all()))
        torch.testing.assert_close(orientation, model.base_rotation, rtol=0, atol=0)

    def test_causal_transition_drops_old_increment_and_replays_initial_orientation(self):
        source = joint_motion()
        with torch.no_grad():
            source.region_rotation[1:, 1] = .02
            source.gs_rotation[:, 0] = .01
        decoded = commit_motion(source)
        source.config.rotation_updates = False
        source.setup(decoded.xyz, decoded.rotation, source.extent, frame_index=2)
        self.assertTrue(bool(source.gs_rotation.eq(0).all()))
        self.assertTrue(bool(source.region_rotation.eq(0).all()))
        torch.testing.assert_close(source(1)[1], decoded.rotation, rtol=0, atol=0)
        replayed = commit_motion(source)
        torch.testing.assert_close(replayed.rotation, decoded.rotation, rtol=0, atol=0)

    def test_restore_and_survivor_pruning_keep_motion_rotations_disabled(self):
        model = self.model()
        restored = SpatialGatedMotion(copy.deepcopy(model.config))
        restored.restore(copy.deepcopy(model.capture()))
        rows = torch.arange(0, len(model.owner), 2)
        restored.retain_survivors(rows)
        for parameter in (restored.region_rotation, restored.gs_rotation):
            self.assertFalse(parameter.requires_grad)
            self.assertTrue(bool(parameter.eq(0).all()))
        replayed = commit_motion(restored)
        torch.testing.assert_close(replayed.rotation, model.base_rotation[rows], rtol=0, atol=0)

    def test_all_rotation_off_replays_births_without_transmitting_quaternions(self):
        for relative in (False, True):
            model = self.model()
            if relative:
                model.config.spatial_gate.region_motion = 'relative'
            prior = copy.deepcopy(model.frame_reference_state)
            xyz, orientation = model.materialized_pose()
            sources = torch.tensor([1, 17, 100])
            xyz = torch.cat((xyz, xyz[sources] + .003))
            orientation = torch.cat((orientation, orientation[sources]))
            model.register_materialized_layout(xyz, orientation, sources)
            with tempfile.TemporaryDirectory() as directory:
                inherited, explicit = Path(directory)/'inherited.bin', Path(directory)/'explicit.bin'
                model.export_motion(inherited, inherit_birth_rotation=True)
                model.export_motion(explicit)
                packet = inherited.read_bytes()
                raw = uncompressed_motion_packet(packet)
                self.assertIn(raw[:4], (b'SPG4', b'SPG8') if relative else (b'SPG3', b'SPG7'))
                self.assertEqual(len(uncompressed_motion_packet(explicit.read_bytes())) - len(raw),
                                 16 * len(sources))
                formats = (packet, raw, compress_motion_packet(raw, 'zlib'),
                           pack_layout(encode_motion_layout(raw), len(raw), MOTION_MAGICS))
                for encoded in formats:
                    decoded = decode_motion(encoded, prior)
                    torch.testing.assert_close(decoded.rotation.view(torch.int32),
                                               orientation.view(torch.int32), rtol=0, atol=0)
                    torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)

    def test_inherited_rotation_stream_rejects_hidden_rotation_updates(self):
        model = self.model()
        xyz, orientation = model.materialized_pose()
        model.register_materialized_layout(xyz, orientation, torch.empty(0, dtype=torch.long))
        with torch.no_grad():
            model.region_rotation[1, 0] = .02
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'preserve all'):
            model.export_motion(Path(directory)/'bad.bin', inherit_birth_rotation=True)


class QuantizationGridReuseTests(unittest.TestCase):
    def model(self):
        model = MotionBase(RigidTreeConfig())
        model.base_xyz = torch.zeros(4, 3)
        model.extent = 6.5
        return model

    def test_repeated_grids_preserve_quantized_values_and_separate_backward_passes(self):
        model = self.model()
        expected = model.base_xyz.new_tensor(
            [model.config.motion_xyz_step_extent * model.extent] * 3 +
            [model.config.motion_rotation_step_rad] * 3)
        first = model.steps
        self.assertIs(first, model.steps)
        self.assertIs(first, model.gs_steps)
        for _ in range(2):
            value = torch.tensor([.004, -.002, .008, .01, -.02, .03], requires_grad=True)
            normalized = value / model.steps
            legacy = value / expected
            torch.testing.assert_close(normalized, legacy, rtol=0, atol=0)
            normalized.sum().backward()
            torch.testing.assert_close(value.grad, 1 / expected, rtol=0, atol=0)

    def test_config_extent_dtype_and_mutation_cannot_leave_stale_constants(self):
        model = self.model()
        model.steps.fill_(0)
        self.assertTrue(bool(model.steps.gt(0).all()))
        model.config.motion_xyz_step_extent *= 2
        torch.testing.assert_close(model.steps[:3], 2 * model.gs_steps[:3], rtol=0, atol=0)
        model.extent *= 2
        self.assertAlmostEqual(float(model.steps[0]),
            model.config.motion_xyz_step_extent * model.extent, places=9)
        model.base_xyz = model.base_xyz.double()
        self.assertEqual(model.steps.dtype, torch.float64)

    def test_inference_first_does_not_leak_an_inference_tensor_into_training(self):
        model = self.model()
        with torch.inference_mode():
            model.steps
        grid = model.steps
        value = torch.ones(6, requires_grad=True)
        (value * grid).sum().backward()
        torch.testing.assert_close(value.grad, grid, rtol=0, atol=0)


class QuantizedMotionPruningTests(unittest.TestCase):
    def test_proximal_gs_keeps_large_signal_and_zeroes_small_signal(self):
        from _motion_fixtures import staged_motion
        model = staged_motion()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = .01
        model.last_step = 2
        translation_grad = torch.zeros_like(model.gs_translation)
        rotation_grad = torch.zeros_like(model.gs_rotation)
        translation_grad[0, 0] = rotation_grad[0, 0] = 1e-8
        translation_grad[1, 0] = rotation_grad[1, 0] = 1e-2
        model.gs_translation.grad = translation_grad
        model.gs_rotation.grad = rotation_grad
        model.optimizer_step()
        model._proximal_gs_blocks()
        assert bool((model.gs_translation[0] == 0).all())
        assert bool((model.gs_rotation[0] == 0).all())
        assert bool(model.gs_translation[1].ne(0).any())
        assert bool(model.gs_rotation[1].ne(0).any())
        assert model.last_metrics['motion/prox_zero_translation_blocks'] >= 1
        diagnostics = model.loss_terms(2, diagnostics=True)
        assert int(diagnostics['gs_translation_active_blocks']) == 1
        assert int(diagnostics['gs_rotation_active_blocks']) == 1

    def test_proximal_gs_handles_one_optimizer_block_without_moments(self):
        from _motion_fixtures import staged_motion
        model = staged_motion()
        model.config.gs_translation_lr = model.config.gs_rotation_lr = .01
        model.last_step = 2
        gradient = torch.zeros_like(model.gs_translation)
        gradient[0, 0] = 1e-8
        gradient[1, 0] = 1e-2
        model.gs_translation.grad = gradient
        model.optimizer_step()
        model._proximal_gs_blocks()
        assert model.gs_rotation not in model.optimizer.state
        assert bool((model.gs_translation[0] == 0).all())
        assert bool(model.gs_translation[1].ne(0).any())


class SparseMotionSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_hard_block_values_and_zero_bin_has_no_sparse_gradient(self):
        value = torch.tensor([[.02, 0., 0.], [.07, -.08, 0.]], requires_grad=True)
        support = sparse_block_support(value, .1, differentiable=True)
        torch.testing.assert_close(support, torch.tensor([0., 1.]), rtol=0, atol=0)
        support.sum().backward()
        self.assertEqual(float(value.grad[0, 0]), 0.)
        self.assertGreater(float(value.grad[1, 0]), 0.)
        self.assertLess(float(value.grad[1, 1]), 0.)
        self.assertFalse(sparse_block_support(value.detach(), .1).requires_grad)

    def test_rate_loss_reaches_both_motion_fields_and_keeps_discrete_value(self):
        # Multiple GS per region make sharing cheaper than separate residuals.
        # One identical GS per region has a valid symmetric zero derivative.
        raw = {name: value.repeat_interleave(2, dim=0) for name, value in reference_raw().items()}
        motion = motion_model(raw)
        motion.config.differentiable_rate_support = True
        with torch.no_grad():
            motion.region_translation[1:, 0] = .6 * motion.steps[0]
            motion.region_rotation[1:, 1] = .6 * motion.steps[3]
            motion.gs_translation[:, 0] = .7 * motion.gs_steps[0]
            motion.gs_rotation[:, 1] = .8 * motion.gs_steps[3]
        loss = motion.loss_terms(5, False)['rate_loss']
        parameters = [motion.region_translation, motion.region_rotation,
                      motion.gs_translation, motion.gs_rotation]
        gradients = torch.autograd.grad(loss, parameters)
        for gradient in gradients:
            self.assertGreater(float(gradient.abs().sum()), 0.)
        motion.config.differentiable_rate_support = False
        self.assertEqual(float(loss), float(motion.loss_terms(5, False)['rate_loss']))


if __name__ == '__main__':
    unittest.main()
