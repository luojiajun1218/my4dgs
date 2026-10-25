"""Relative region composition, learned gates, and independent stream replay."""
import copy
import unittest

import torch

import test_joint_spatial_motion as joint_tests
from pipeline.model.rigid_tree_codec import encode_motion, decode_motion, uncompressed_motion_packet
from pipeline.model.spatial_gate import spatial_accumulate, spatial_owner
from pipeline.model.spatial_gated_motion import SpatialGatedMotion


class RelativeSpatialMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        m = joint_tests.JointSpatialMotionTests().model()
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
            self.assertEqual(uncompressed_motion_packet(data)[:4], b'SPG2')
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
        m = joint_tests.JointSpatialMotionTests().model()
        with torch.no_grad():
            m.region_translation[:, 0] = torch.arange(len(m.parents)).remainder(5) * .002
            m.region_rotation[:, 1] = torch.arange(len(m.parents)).remainder(7) * .001
            m.gs_translation[:, 2] = .0024
        before = tuple(x.detach().clone() for x in m._regional_predictions(hard=True))
        full = tuple(x.clone() for x in m._full_motion_symbols())
        joint_tests.JointSpatialMotionTests().commit(m)
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
        first = joint_tests.JointSpatialMotionTests().commit(m,
            births=m.base_xyz[:1] + .01, sources=torch.tensor([0]))
        m.setup(first.xyz, first.rotation, m.extent, frame_index=2)
        m.retain_survivors(torch.arange(0, len(m.owner), 2))
        second = joint_tests.JointSpatialMotionTests().commit(m)
        self.assertEqual(len(second.xyz), (len(first.xyz) + 1) // 2)


if __name__ == '__main__':
    unittest.main()
