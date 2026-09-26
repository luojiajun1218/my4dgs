import unittest
from types import SimpleNamespace

import torch

from scripts.adaptive_hierarchy import (
    AdaptiveHierarchyMotion, AdaptiveOffsetGate, OFF, SPLIT, UPDATE,
    sample_pixel_map)
from scene.decoders import DecoderIdentity
from scene.gaussian_model import GaussianModel


class AdaptiveHierarchyMotionTest(unittest.TestCase):
    def make_hierarchy(self, scores=None, image_scores=None, root_scale=8.0):
        anchors = torch.tensor([
            [0.1, 0.1, 0.1], [0.2, 0.1, 0.1],
            [4.1, 0.1, 0.1], [4.2, 0.1, 0.1],
            [8.1, 0.1, 0.1], [8.2, 0.1, 0.1],
        ])
        if scores is None:
            scores = torch.tensor([1., 1., 0., 0., 0., 0.])
        return AdaptiveHierarchyMotion(
            anchors, torch.zeros(3), 1.0, 2, scores,
            dynamic_threshold=0.5, image_change_scores=image_scores,
            image_change_threshold=0.01, levels=3,
            root_scale=root_scale,
            off_ratio=0.0, update_ratio=0.75,
            split_residual_ratio=0.015, split_heterogeneity=0.2,
            split_min_gain=0.002, split_relative_gain=0.02)

    def propose(self, hierarchy):
        hierarchy.accumulate_residual(
            torch.ones(6, dtype=torch.bool),
            torch.tensor([0.1, 0.1, 0.02, 0.02, 0., 0.]))
        return hierarchy.begin_split_trial()

    def test_static_regions_stop_at_off(self):
        hierarchy = self.make_hierarchy()
        static_node = hierarchy.point_to_node_0[4]
        self.assertEqual(hierarchy.states_0[static_node].item(), OFF)
        self.assertFalse(torch.any(hierarchy.update_point_mask[4:]))

    def test_finer_root_separates_previous_coarse_region(self):
        coarse = self.make_hierarchy(root_scale=8.0)
        finer = self.make_hierarchy(root_scale=2.0)
        self.assertEqual(coarse.point_to_node_0[0], coarse.point_to_node_0[2])
        self.assertNotEqual(finer.point_to_node_0[0], finer.point_to_node_0[2])
        self.assertEqual(finer.node_counts[0], 3)

    def test_image_evidence_reopens_off_each_frame(self):
        hierarchy = self.make_hierarchy(
            torch.zeros(6), torch.tensor([0., 0., 0., 0., 0.02, 0.02]))
        self.assertFalse(torch.any(hierarchy.update_point_mask[:4]))
        self.assertTrue(torch.all(hierarchy.update_point_mask[4:]))

    def test_split_children_inherit_parent_motion(self):
        hierarchy = self.make_hierarchy()
        with torch.no_grad():
            hierarchy.translations[0][:, 0] = 0.25
        self.assertTrue(self.propose(hierarchy))
        self.assertTrue(torch.any(hierarchy.states_0 == SPLIT))
        with torch.no_grad():
            hierarchy.translations[1][:, 0] = 0.10
        translation, _ = hierarchy()
        torch.testing.assert_close(
            translation[:2, 0], torch.full((2,), 0.35))
        torch.testing.assert_close(
            translation[2:], torch.zeros_like(translation[2:]))

    def test_trial_rejects_split_without_image_gain(self):
        hierarchy = self.make_hierarchy()
        self.assertTrue(self.propose(hierarchy))
        before = torch.full((hierarchy.node_counts[0],), 0.1)
        hierarchy.finish_split_trial(before, before.clone())
        self.assertFalse(torch.any(hierarchy.states_0 == SPLIT))
        self.assertTrue(torch.all(hierarchy.update_point_mask[:4]))

    def test_trial_keeps_split_with_image_gain(self):
        hierarchy = self.make_hierarchy()
        self.assertTrue(self.propose(hierarchy))
        before = torch.full((hierarchy.node_counts[0],), 0.1)
        hierarchy.finish_split_trial(before, before - 0.02)
        self.assertTrue(torch.any(hierarchy.states_0 == SPLIT))
        self.assertTrue(torch.all(hierarchy.update_point_mask[:2]))
        self.assertFalse(torch.any(hierarchy.update_point_mask[2:]))

    def test_finalize_folds_path_into_terminal_regions(self):
        hierarchy = self.make_hierarchy()
        with torch.no_grad():
            hierarchy.translations[0][:, 0] = 0.25
        self.propose(hierarchy)
        before_error = torch.full((hierarchy.node_counts[0],), 0.1)
        hierarchy.finish_split_trial(before_error, before_error - 0.02)
        with torch.no_grad():
            hierarchy.translations[1][:, 0] = 0.10
        before_translation, before_quaternion = hierarchy()
        hierarchy.finalize()
        after_translation, after_quaternion = hierarchy()
        torch.testing.assert_close(after_translation, before_translation)
        torch.testing.assert_close(after_quaternion, before_quaternion)
        self.assertTrue(hierarchy.terminal_only)

    def test_payload_has_states_without_parent_table(self):
        hierarchy = self.make_hierarchy()
        payload = hierarchy.frame_payload()
        self.assertIn("region_states", payload["levels"][0])
        self.assertNotIn("parents", payload["levels"][0])
        self.assertIn(UPDATE, payload["levels"][0]["region_states"].tolist())

    def test_pixel_sampling_checks_visibility(self):
        values, valid = sample_pixel_map(
            torch.tensor([[1., 1.], [10., 10.]]),
            torch.ones(3, 3), torch.tensor([True, True]))
        self.assertEqual(valid.tolist(), [True, False])
        self.assertGreater(values[0].item(), 0.0)

    def test_hierarchy_gate_only_masks_offset(self):
        previous = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        current = previous + 1.0
        model = SimpleNamespace(
            hierarchy_update_mask=torch.tensor([False, True]),
            hierarchy_residual_gate=None,
            prev_atts={"offset": previous},
        )
        actual = GaussianModel._apply_hierarchy_update(model, "offset", current)
        torch.testing.assert_close(actual[0], previous[0])
        torch.testing.assert_close(actual[1], current[1])
        appearance = torch.ones(2, 4)
        self.assertIs(
            GaussianModel._apply_hierarchy_update(model, "f_dc", appearance),
            appearance)

    def test_setting_hierarchy_gate_preserves_appearance_masks(self):
        model = SimpleNamespace(
            get_atts={"anchor": torch.zeros(2, 3)},
            mask_offset=torch.ones(2, 1, dtype=torch.bool),
            mask_features_dc=torch.ones(2, 1, dtype=torch.bool),
            hierarchy_update_mask=None,
        )
        GaussianModel.set_hierarchy_update_mask(
            model, torch.tensor([False, True]))
        self.assertEqual(model.mask_offset.flatten().tolist(), [False, True])
        self.assertEqual(model.mask_features_dc.flatten().tolist(), [True, True])

    def test_offset_gate_is_sparse_only_inside_update_regions(self):
        gate = AdaptiveOffsetGate(
            torch.tensor([True, True, False]),
            torch.tensor([0.8, 0.04, 0.9]),
            lr=0.1, temperature=0.3, lambda_l0=0.01,
            lambda_l2=0.01)
        self.assertEqual(gate.active_mask().tolist(), [True, False, False])
        previous = torch.ones(3, 3)
        model = SimpleNamespace(
            hierarchy_update_mask=torch.tensor([True, True, False]),
            hierarchy_residual_gate=gate,
            prev_atts={"offset": previous},
        )
        actual = GaussianModel._apply_hierarchy_update(
            model, "offset", previous + 1.0)
        torch.testing.assert_close(actual[0], previous[0] + 0.8)
        torch.testing.assert_close(actual[1:], previous[1:])

    def test_closed_offset_gate_retains_surrogate_gradient(self):
        gate = AdaptiveOffsetGate(
            torch.tensor([True]), torch.tensor([0.04]),
            lr=0.1, temperature=0.3, lambda_l0=0.01,
            lambda_l2=0.01)
        residual = torch.zeros(1, 3, requires_grad=True)
        self.assertFalse(gate.active_mask()[0].item())
        gate(residual).sum().backward()
        self.assertGreater(residual.grad.abs().sum().item(), 0.0)

    def test_new_gaussians_start_active_without_extra_gate_parameters(self):
        gate = AdaptiveOffsetGate(
            torch.tensor([True, False]), torch.tensor([0.04, 0.9]),
            lr=0.1, temperature=0.3, lambda_l0=0.01,
            lambda_l2=0.01)
        gate.append_new_points(2)
        self.assertEqual(gate.active_mask().tolist(),
                         [False, False, True, True])
        self.assertEqual(gate.logits.numel(), 1)
        self.assertEqual(gate.frame_payload()["active_gaussians"], 2)

    def test_next_frame_rebases_unused_raw_offset(self):
        rendered = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        raw = torch.tensor([[9., 9., 9.], [8., 8., 8.]])
        model = SimpleNamespace(
            prev_atts={"offset": rendered},
            latent_decoders={"offset": DecoderIdentity()},
            _latents={"offset": torch.nn.Parameter(raw)},
            hierarchy_residual_gate=object(),
            hierarchy_update_mask=torch.tensor([True, False]),
        )
        GaussianModel.prepare_hierarchy_frame(model)
        torch.testing.assert_close(model._latents["offset"], rendered)
        self.assertIsNone(model.hierarchy_residual_gate)
        self.assertIsNone(model.hierarchy_update_mask)


if __name__ == "__main__":
    unittest.main()
