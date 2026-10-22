"""Hierarchical motion composition, exact coordinate recoding, causal streams, and checkpoint compatibility."""
import copy
import importlib
import itertools
from pathlib import Path
import tempfile
import unittest
import torch
import numpy as np
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from _motion_fixtures import staged_motion
from pipeline.model.rigid_residual_tree import saved_config_with_default_motion_lr
from _motion_fixtures import joint_motion, commit_motion
from pipeline.model.spatial_gated_motion import SpatialGatedMotion
from pipeline.model.spatial_gate import spatial_state_route
from pipeline.model.rigid_tree_codec import encode_motion, decode_motion, uncompressed_motion_packet
from pipeline.model.spatial_gate import spatial_owner
from pipeline.module import restore_retired_reference
from pipeline.model.birth_quantization import RAW_FIELDS, model_raw
from pipeline.trainer import Trainer
from scripts.replay_rigid_tree import decode_frame, replay
from scripts.benchmark_rigid_residual_tree import _decode_frame
from _stream_fixtures import Rows, reference_raw, motion_model, export_frame


SPATIAL = importlib.import_module('pipeline.model.spatial_gated_motion')


CODEC = importlib.import_module('pipeline.model.rigid_tree_codec')


LAYOUT = importlib.import_module('pipeline.model.spatial_gate')


class SpatialGatedMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, repetitions=16, extent=1.):
        return staged_motion(repetitions=repetitions, extent=extent)

    @staticmethod
    def representative_cost(symbols, candidate, rows, regional_row, price, scales):
        residual = symbols - candidate
        t, r = residual[:, :3].any(1), residual[:, 3:].any(1)
        rt, rr = candidate[:3].any(), candidate[3:].any()
        byte_count = ((t | r) * rows + 6 * (t.astype(int) + r)).sum()
        byte_count += regional_row * bool(rt or rr) + 6 * (int(rt) + int(rr))
        return price * byte_count + (np.abs(residual) * scales).sum()

    def test_statistical_representative_matches_exhaustive_byte_and_l1_objective(self):
        rng = np.random.default_rng(7)
        candidates = np.array(list(itertools.product(range(-1,2), repeat=6)))
        for repeat in range(12):
            symbols = rng.integers(-1,2,size=(8,6),dtype=np.int32)
            rows = rng.integers(2,5,size=8)
            price = 0. if repeat == 0 else float(rng.uniform(0,1))
            scales = np.zeros(6) if repeat == 1 else rng.uniform(0,1,size=6)
            chosen, actual = SPATIAL._rate_relative_representative(symbols,rows,3,price,scales)
            expected = min(self.representative_cost(symbols,c,rows,3,price,scales)
                           for c in candidates)
            self.assertAlmostEqual(actual,expected,places=10)
            self.assertAlmostEqual(actual,self.representative_cost(
                symbols,chosen,rows,3,price,scales),places=10)

    def test_shared_translation_is_not_lost_when_rotations_have_different_symbols(self):
        symbols = np.zeros((10,6),dtype=np.int32)
        symbols[:8,0] = 20
        symbols[:8,3] = np.arange(1,9)
        rows = np.full(10,3)
        chosen, cost = SPATIAL._rate_relative_representative(
            symbols,rows,3,1.,np.zeros(6))
        self.assertEqual(chosen[0],20)
        old = self.representative_cost(symbols,np.zeros(6,int),rows,3,1.,np.zeros(6))
        self.assertLess(cost,old)

    def test_representative_respects_both_region_and_residual_integer_ranges(self):
        symbols = np.zeros((5,6),dtype=np.int32)
        symbols[:,0] = [45000,45001,45002,45003,45004]
        chosen,_ = SPATIAL._rate_relative_representative(
            symbols,np.full(5,3),3,.1,np.ones(6))
        self.assertTrue((np.abs(chosen)<=32767).all())
        self.assertTrue((np.abs(symbols-chosen)<=32767).all())

    def test_previous_joint_mode_objective_checkpoint_cannot_silently_resume(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet.pop('representative_objective')
        with self.assertRaisesRegex(ValueError,'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

        # Finite-step quantized fitting may finish one bin from the static
        # majority optimum; a zero initialization is no longer a hard freeze.

    def test_old_late_only_regional_fitting_checkpoint_is_replay_only(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet.pop('regional_image_fitting')
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_retired_refinement_is_readable_as_metadata_but_cannot_resume(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet['config'].update(post_birth_motion_steps=100, fixed_support_refinement=True)
        normalized = saved_config_with_default_motion_lr(packet['config'])
        self.assertNotIn('post_birth_motion_steps', normalized)
        self.assertNotIn('fixed_support_refinement', normalized)
        with self.assertRaisesRegex(ValueError, 'refinement checkpoints are replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_old_gate_sparse_pause_schedule_cannot_silently_resume(self):
        model=self.model();packet=copy.deepcopy(model.capture())
        packet.pop('gate_sparsification')
        with self.assertRaisesRegex(ValueError,'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_gradient_refined_region_checkpoint_is_replay_only(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet['config']['spatial_gate']['learn_region_motion'] = True
        with self.assertRaisesRegex(ValueError, 'replayed'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_gate_recoding_preserves_half_bin_ties(self):
        model = self.model()
        model.config.motion_xyz_step_extent = model.config.gs_xyz_step_extent = .125
        parent = int(model.owner[0])
        rows = model.owner == parent
        with torch.no_grad():
            model.gs_translation[rows, 0] = .0625
            model.region_translation[model.parents == parent, 0] = -.125
        before = model.materialized_pose()
        with torch.no_grad():
            model.gate_logits[parent] = 4
        model._select_frontier()
        for old, new in zip(before, model.materialized_pose()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.assertTrue(model._values(hard=True)[2][rows, 0].eq(1).all())


    def test_live_gate_recoding_preserves_sub_bin_image_updates(self):
        model = self.model()
        model._candidate_initialized = model._coding_initialized = True
        with torch.no_grad():
            model.gs_translation[:,0] = .0102
            model.region_translation[9:73,0] = .003
        full = model._full_motion_symbols()
        before = model.materialized_pose()
        model._recode_owner(model.spatial_paths[:,1], full, preserve_remainder=True)
        for left, right in zip(before, model.materialized_pose()):
            torch.testing.assert_close(left,right,rtol=0,atol=1e-7)
        remainder = model.gs_translation[:,0] - torch.round(
            model.gs_translation[:,0]/model.gs_steps[0])*model.gs_steps[0]
        torch.testing.assert_close(remainder,torch.full_like(remainder,.0002),rtol=0,atol=1e-8)

    def test_old_frozen_motion_schedule_cannot_silently_resume_as_live_fitting(self):
        model = self.model()
        packet = copy.deepcopy(model.capture())
        packet['config']['spatial_gate'].pop('image_fit_during_gate')
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)

    def test_coordinate_recoding_preserves_gs_optimizer_moments(self):
        model = self.model()
        model._candidate_initialized = model._coding_initialized = True
        model.gs_translation.grad = torch.ones_like(model.gs_translation) * .001
        model.gs_rotation.grad = torch.ones_like(model.gs_rotation) * .001
        model.optimizer.step()
        saved = {name: {key: model.optimizer.state[getattr(model, name)][key].clone()
                       for key in ('exp_avg', 'exp_avg_sq')}
                 for name in ('gs_translation', 'gs_rotation')}
        model._recode_owner(model.spatial_paths[:, 1])
        for name, moments in saved.items():
            for key, before in moments.items():
                torch.testing.assert_close(model.optimizer.state[getattr(model, name)][key], before, rtol=0, atol=0)


    def test_sequential_codec_birth_paths_and_checkpoint_resume(self):
        model = self.model()
        with torch.no_grad():
            model.region_translation[1:, 0] = .005
            model.region_rotation[1:, 2] = .01
            model.gate_logits[model.split_eligible] = 2
        model.last_step = 40
        # Populate the optimizer to check that it restores the gate's moments.
        model.loss_terms(50, False)['rate_loss'].backward()
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        model._select_frontier()
        captured = copy.deepcopy(model.capture())
        resumed = SPATIAL.SpatialGatedMotion(model.config)
        resumed.restore(captured)
        torch.testing.assert_close(resumed.gate_logits, model.gate_logits)
        self.assertEqual(len(resumed.optimizer.param_groups), 5)
        self.assertIn(resumed.gate_logits, resumed.optimizer.state)
        model.record_final_diagnostics()
        xyz, q = model.materialized_pose()
        source = torch.tensor([0, 42])
        model.register_materialized_layout(torch.cat((xyz, xyz[source])),
                                           torch.cat((q, q[source])), source)
        with tempfile.TemporaryDirectory() as directory:
            info = model.export_motion(Path(directory) / 'motion.bin')
            self.assertTrue(info['format'].startswith(('SPG', 'SPC', 'SPL')))
            decoded = model.codec_previous_state
            expected_paths = LAYOUT.spatial_route(decoded.xyz[-2:], decoded.spatial_centers,
                                                  model.config.spatial_gate.max_depth)
            torch.testing.assert_close(decoded.spatial_paths[-2:], expected_paths)
            t, r = model._full_motion_symbols()
            expected = torch.round(.6 * torch.cat((t, t[source])))
            model.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
            expected_paths = LAYOUT.spatial_route(decoded.xyz, decoded.spatial_centers,
                                                  model.config.spatial_gate.max_depth)
            torch.testing.assert_close(model.spatial_paths, expected_paths)
            torch.testing.assert_close(model.owner, LAYOUT.spatial_owner(expected_paths, model.opened))
            torch.testing.assert_close(model._full_motion_symbols()[0], expected)
            self.assertEqual(model.frame_reference_state.digest(), decoded.digest())

    def test_static_planes_reproduce_initial_paths_and_route_empty_cells(self):
        model = self.model(1)
        reference = model.frame_reference_state
        paths = LAYOUT.spatial_route(reference.xyz, reference.spatial_centers,
                                     model.config.spatial_gate.max_depth)
        torch.testing.assert_close(paths, reference.spatial_paths, rtol=0, atol=0)
        # Even a previously empty cell has deterministic descendants.
        outside = torch.tensor([[-10., -10., -10.], [10., 10., 10.]])
        routed = LAYOUT.spatial_route(outside, reference.spatial_centers,
                                      model.config.spatial_gate.max_depth)
        torch.testing.assert_close(reference.parents[routed[:, 1:]], routed[:, :-1])

    def test_crossing_split_changes_next_frame_membership_and_replays_exactly(self):
        model = self.model(1)
        original = model.spatial_paths[0].clone()
        with torch.no_grad():
            model.gs_translation[0] = torch.tensor([.8, .8, .8])
        xyz, q = model.materialized_pose()
        # A newborn can move away from its source's region during birth fitting.
        newborn = xyz[0:1] - 1.5
        model.register_materialized_layout(torch.cat((xyz, newborn)),
                                           torch.cat((q, q[0:1])), torch.tensor([0]))
        reference = model.frame_reference_state
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'first.bin'
            model.export_motion(path)
            decoded = CODEC.decode_motion(path.read_bytes(), reference)
            self.assertTrue(torch.equal(decoded.spatial_paths[0], original))
            self.assertFalse(torch.equal(decoded.spatial_paths[0], decoded.spatial_paths[-1]))
            resumed = SPATIAL.SpatialGatedMotion(model.config)
            resumed.restore(copy.deepcopy(model.capture()))
            for current in (model, resumed):
                current.setup(decoded.xyz, decoded.rotation, 1., frame_index=2)
                self.assertFalse(torch.equal(current.spatial_paths[0], original))
            torch.testing.assert_close(model.spatial_paths, resumed.spatial_paths)
            with torch.no_grad():
                model.region_translation[1:9, 0] = torch.arange(1, 9) * .001
            xyz, q = model.materialized_pose()
            model.register_materialized_layout(xyz, q)
            model.export_motion(Path(directory) / 'second.bin')
            second = CODEC.decode_motion((Path(directory) / 'second.bin').read_bytes(), decoded)
            torch.testing.assert_close(second.xyz, xyz, rtol=0, atol=1e-7)
            torch.testing.assert_close(second.spatial_paths, model.spatial_paths)

    def test_legacy_streams_keep_fixed_membership_and_original_reference_hash(self):
        model = self.model(1)
        prior = copy.deepcopy(model.frame_reference_state)
        prior.spatial_centers = None
        legacy = prior.capture()
        legacy.pop('spatial_centers')
        restored = CODEC.DecodedState.restore(legacy)
        self.assertEqual(prior.digest(), restored.digest())
        rt, rr, gt, gr = [x.cpu() for x in model._values(hard=True)]
        gt[0] = 800
        packet = CODEC.encode_motion(prior, 1, 1., model.previous_rows, model.owner,
            model.parents, rt, rr, gt, gr, torch.empty(0, dtype=torch.long),
            torch.empty(0, 3), torch.empty(0, 4), .001, .001,
            gs_translation_step_extent=.001, gs_rotation_step_rad=.001,
            spatial_open=model.opened)
        decoded = CODEC.decode_motion(packet, restored)
        self.assertIsNone(decoded.spatial_centers)
        torch.testing.assert_close(decoded.spatial_paths, prior.spatial_paths)
        self.assertGreater(float(decoded.xyz[0, 0]), .8)
        changed = copy.deepcopy(model.frame_reference_state)
        changed.spatial_centers[0, 0] += .01
        self.assertNotEqual(changed.digest(), model.frame_reference_state.digest())

    def test_fixed_membership_training_checkpoint_is_replay_only(self):
        model = self.model(1)
        packet = copy.deepcopy(model.capture())
        packet.pop('spatial_assignment')
        with self.assertRaisesRegex(ValueError, 'replay-only'):
            SPATIAL.SpatialGatedMotion(model.config).restore(packet)


class JointMotionStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_next_frame_inherits_full_motion_with_births_and_recomputed_owners(self):
        model = joint_motion()
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
        decoded = commit_motion(model, torch.tensor([[200., -200., 100.]]), source)
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
        model = joint_motion()
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


class RelativeSpatialMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        m = joint_motion()
        m.config.spatial_gate.region_motion = 'relative'
        m.config.proximal_gs_regularization = True
        m.config.differentiable_rate_support = True
        m.observations.fill_(1)
        return m

    def packet(self, m, compression='auto'):
        rt, rr, gt, gr = m._values(hard=True)
        return encode_motion(m.frame_reference_state, m.frame_index, m.extent,
            m.previous_rows, m.owner, m.parents, rt, rr, gt, gr,
            torch.empty(0, dtype=torch.long), torch.empty(0, 3), torch.empty(0, 4),
            m.config.motion_xyz_step_extent, m.config.motion_rotation_step_rad,
            compression=compression, gs_translation_step_extent=m.config.gs_xyz_step_extent,
            gs_rotation_step_rad=m.config.gs_rotation_step_rad, spatial_open=m.opened,
            spatial_relative=True)

    def test_integer_corrections_accumulate_without_int16_overflow(self):
        m = self.model()
        with torch.no_grad():
            m.region_translation[1:, 0] = 20000 * m.steps[0]
            m.gate_logits[m.split_eligible] = 4.
        # Construct a large valid pose directly. Recoding a coarse pose into
        # this field would itself need a residual outside signed-int16 range.
        m.opened = m.split_eligible.clone()
        m.opened[0] = True
        m.owner = spatial_owner(m.spatial_paths, m.opened)
        m._hard_gate_frozen = True
        pred = m._regional_predictions(hard=True)[0]
        self.assertTrue(bool(pred[m.owner, 0].eq(60000).all()))
        for compression in ('none', 'auto', 'zlib'):
            data = self.packet(m, compression)
            self.assertIn(uncompressed_motion_packet(data)[:4], (b'SPG2', b'SPG6'))
            decoded = decode_motion(data, m.frame_reference_state)
            torch.testing.assert_close(decoded.xyz, m(100)[0], atol=1e-6, rtol=0)

    def test_selected_path_keeps_parent_corrections_and_discards_hidden_nodes(self):
        m = self.model()
        parent = int(m.owner[0])
        child = int(m.spatial_paths[0, 1])
        with torch.no_grad():
            m.region_translation[parent, 0] = .004
            m.region_translation[child, 0] = .006
            m.gate_logits[parent] = 4.
        m._select_frontier(freeze=True)
        rt, _, _, _ = m._values(hard=True)
        self.assertEqual(int(rt[parent, 0]), 2)
        self.assertEqual(int(rt[child, 0]), 3)
        self.assertFalse(bool(m.owner.eq(parent).any()))
        self.assertGreater(int(m._stored_region_support().sum()), m.owner.unique().numel())
        decoded = decode_motion(self.packet(m), m.frame_reference_state)
        torch.testing.assert_close(decoded.xyz, m(100)[0], atol=1e-6, rtol=0)

    def test_ancestor_support_stops_at_identity_in_mixed_depth_frontiers(self):
        m = self.model()
        for logit in (-4., 4.):
            with torch.no_grad():
                m.gate_logits[m.split_eligible] = logit
                # Retain a coarse region while other branches can be deep.
                m.gate_logits[1] = -4.
            m._select_frontier(freeze=False)
            expected = torch.zeros_like(m.opened)
            reach = torch.ones(len(m.owner), dtype=torch.bool)
            for nodes in m.spatial_paths.unbind(1):
                expected[nodes[reach]] = True
                reach &= m.opened[nodes]
            torch.testing.assert_close(m._stored_region_support(), expected, atol=0, rtol=0)
            self.assertEqual(int(expected.sum()), len(m.owner.unique()) +
                sum(bool(expected[n] and m.opened[n]) for n in range(1,len(expected))))

    def test_closed_candidate_fits_without_training_gate_target_or_parent(self):
        m = self.model()
        with torch.no_grad():
            m.gs_translation[:, 0] = .012
            m.gate_logits.fill_(-6)
        m._select_frontier(freeze=False)
        loss = m.loss_terms(5, False)['candidate_motion_loss']
        grad = torch.autograd.grad(loss, [m.region_translation, m.gs_translation,
            m.gate_logits], allow_unused=True)
        self.assertIsNone(grad[1])
        self.assertIsNone(grad[2])
        self.assertEqual(float(grad[0][1:9].abs().sum()), 0.)
        self.assertGreater(float(grad[0][9:].abs().sum()), 0.)

    def test_gate_gradient_opens_shared_correction_without_rule_edits(self):
        m = self.model()
        parent = int(m.owner[0])
        children = torch.where(m.parents.eq(parent) & m.node_counts.gt(0))[0]
        with torch.no_grad():
            m.region_translation[parent, 0] = .02
            m.gs_translation[m.owner.eq(parent), 0] = .01
            for k, child in enumerate(children):
                rows = m.spatial_paths[:, 1].eq(child)
                correction = -.01 if k % 2 else .01
                m.gs_translation[rows, 0] = correction
                m.region_translation[child, 0] = correction
            m.gate_logits.fill_(-4.)
            m.gate_logits[parent] = -.2
        full = tuple(x.detach().clone() for x in m._full_motion_symbols())
        optimizer = torch.optim.Adam([m.gate_logits], lr=.15, eps=1e-12)
        initial = float(m.loss_terms(5, False)['rate_bytes_surrogate'])
        for step in range(1, 31):
            optimizer.zero_grad(set_to_none=True)
            m.loss_terms(step, False)['rate_loss'].backward()
            optimizer.step()
            m._select_frontier(freeze=False)
        self.assertTrue(bool(m.opened[parent]))
        self.assertLess(float(m.loss_terms(31, False)['rate_bytes_surrogate']), initial)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))

    def test_fixed_frontier_learns_ancestors_and_refits_without_pose_change(self):
        m = self.model()
        with torch.no_grad():
            m.gate_logits[m.split_eligible] = 4.
            m.gs_translation[:, 0] = .006
        m._select_frontier(freeze=True)
        loss = m.regularization_loss(80)
        loss.backward()
        self.assertGreater(float(m.region_translation.grad[1:9].abs().sum()), 0.)
        self.assertIsNone(m.gate_logits.grad)
        full = tuple(x.clone() for x in m._full_motion_symbols())
        m._refresh_active_representatives()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))

    def test_explicit_migration_preserves_all_predictions_and_next_frame_warm_start(self):
        m = joint_motion()
        with torch.no_grad():
            m.region_translation[:, 0] = torch.arange(len(m.parents)).remainder(5) * .002
            m.region_rotation[:, 1] = torch.arange(len(m.parents)).remainder(7) * .001
            m.gs_translation[:, 2] = .0024
        before = tuple(x.detach().clone() for x in m._regional_predictions(hard=True))
        full = tuple(x.clone() for x in m._full_motion_symbols())
        commit_motion(m)
        m.enable_relative_regions()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, m._regional_predictions(hard=True))))
        xyz, q = m.codec_previous_state.xyz, m.codec_previous_state.rotation
        m.setup(xyz, q, m.extent, frame_index=2)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full, m._full_motion_symbols())))

    def test_relative_checkpoint_resumes_the_same_update(self):
        m = self.model()
        with torch.no_grad():
            m.gs_translation[:, 0] = .003
        m(1, training=True)
        (m(1)[0].square().mean() + m.regularization_loss(1)).backward()
        m.optimizer_step()
        saved = copy.deepcopy(m.capture())
        restored = SpatialGatedMotion(copy.deepcopy(m.config))
        restored.restore(saved)
        for model in (m, restored):
            (model(2, training=True)[0].square().mean() + model.regularization_loss(2)).backward()
            model.optimizer_step()
        for key in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation', 'gate_logits'):
            torch.testing.assert_close(getattr(m, key), getattr(restored, key), atol=0, rtol=0)

    def test_relative_chain_with_births_and_survivor_retirement(self):
        m = self.model()
        with torch.no_grad():
            m.region_translation[1:9, 0] = .004
            m.region_rotation[1:9, 1] = .002
        first = commit_motion(m,
            births=m.base_xyz[:1] + .01, sources=torch.tensor([0]))
        m.setup(first.xyz, first.rotation, m.extent, frame_index=2)
        m.retain_survivors(torch.arange(0, len(m.owner), 2))
        second = commit_motion(m)
        self.assertEqual(len(second.xyz), (len(first.xyz) + 1) // 2)


class StreamSurvivorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cross_frame_births_deletions_replay_and_midframe_bootstrap(self):
        # More births than deletions must still retire old appearance rows;
        # inferring birth count from the net change would silently be wrong.
        for new_count in (0, 1, 3):
            with self.subTest(new_count=new_count), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                raw = reference_raw()
                motion = motion_model(raw)
                initial = copy.deepcopy(motion.frame_reference_state)
                with torch.no_grad():
                    motion.gs_translation[:, 0] = torch.arange(8) * .002
                    motion.gs_rotation[:, 1] = torch.arange(8) * .001
                first = export_frame(directory, motion, raw, torch.tensor([0, 6]))
                prior, decoded_first = decode_frame(directory, 1, initial, raw)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(decoded_first[name], first[name], rtol=0, atol=0)
                full = tuple(value.clone() for value in motion._full_motion_symbols())
                model = Rows(first)
                remove = torch.zeros(len(first['xyz']), dtype=torch.bool)
                remove[[0, len(remove) - 1]] = True
                rows = (~remove).nonzero(as_tuple=True)[0]
                retired = dict(rows=remove.nonzero(as_tuple=True)[0],
                    attributes={name: first[name][remove].clone() for name in RAW_FIELDS[2:]})
                model.prune_points(remove)
                kept = model_raw(model)
                motion.setup(kept['xyz'], kept['rotation'], 2., frame_index=2, previous_rows=rows)
                for actual, previous in zip(motion._full_motion_symbols(), full):
                    expected = torch.cat((previous, previous[torch.tensor([0, 6])]))[rows]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                restored = SpatialGatedMotion(copy.deepcopy(motion.config))
                restored.restore(copy.deepcopy(motion.capture()))
                self.assertEqual(restored._survivor_mapping_bytes, motion._survivor_mapping_bytes)
                rebuilt = restore_retired_reference(kept, prior, rows, retired)
                for name in RAW_FIELDS:
                    torch.testing.assert_close(rebuilt[name], first[name], rtol=0, atol=0)
                bootstrap_module = SimpleNamespace(gaussians=model, motion=restored,
                    frame_finalized=False, retired_reference=retired,
                    initial_reference=dict(original_count=8, retained_count=8,
                                           removed_count=0, threshold=0.))
                worker = SimpleNamespace(output_dir=directory / 'resume', module=bootstrap_module,
                    datamodule=SimpleNamespace(cameras_extent=2.))
                Trainer._write_reference(worker, frame_index=1, state=prior)
                state_from_bootstrap, raw_from_bootstrap = replay(
                    worker.decoder_reference_path.parent, worker.decoder_reference_path, 1)
                self.assertEqual(state_from_bootstrap.digest(), prior.digest())
                for name in RAW_FIELDS:
                    torch.testing.assert_close(raw_from_bootstrap[name], first[name], rtol=0, atol=0)
                sources = torch.arange(new_count, dtype=torch.long)
                second = export_frame(directory, restored, kept, sources)
                for decode in (decode_frame, _decode_frame):
                    current, decoded_second = decode(directory, 2, prior, first)
                    torch.testing.assert_close(current.survivor_rows, rows)
                    self.assertEqual(len(current.xyz), 8 + new_count)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(decoded_second[name], second[name], rtol=0, atol=0)
                if new_count:
                    model = Rows(second)
                    remove = torch.arange(len(second['xyz'])) >= len(second['xyz']) - new_count
                    model.prune_points(remove)
                    rows = (~remove).nonzero(as_tuple=True)[0]
                    third_base = model_raw(model)
                    restored.setup(third_base['xyz'], third_base['rotation'], 2.,
                                   frame_index=3, previous_rows=rows)
                    third = export_frame(directory, restored, third_base,
                                         torch.empty(0, dtype=torch.long))
                    last, decoded_third = decode_frame(directory, 3, current, second)
                    self.assertEqual(len(last.xyz), 8)
                    for name in RAW_FIELDS:
                        torch.testing.assert_close(decoded_third[name], third[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
