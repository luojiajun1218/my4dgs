import io
import os
from pathlib import Path
import unittest
import zlib

import torch

from scripts.adaptive_gate_partition import (
    OFF, SPLIT, UPDATE, build_adaptive_gate_partition,
    build_compact_gate_payload, decode_compact_gate_payload,
    decode_sqres_reopen_mask, encode_sqres_reopen_mask,
    finalize_gate_partition, reconstruct_gate_groups,
    refine_adaptive_gate_partition, inherit_gate_parameters,
)
from scene.decoders import DecoderIdentity, Gate, LatentDecoderRes
from scene.gaussian_model import (
    GaussianModel, adaptive_gate_offset_size_bits,
    residual_decoder_gate_for_attribute,
)


class AdaptiveGatePartitionTest(unittest.TestCase):
    def test_static_roots_remain_off_and_map_every_point(self):
        roots = torch.tensor([4, 4, 9])
        anchors = torch.tensor([
            [0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [2.1, 0.1, 0.1],
        ])
        partition = build_adaptive_gate_partition(
            roots, anchors, torch.zeros(3, dtype=torch.bool),
            voxel_size=1.0, max_levels=4)
        self.assertEqual(partition["split_count"], 0)
        self.assertEqual(partition["terminal_group_count"], 2)
        self.assertEqual([node["state"] for node in partition["nodes"]],
                         [OFF, OFF])
        self.assertEqual(partition["point_group_ids"].numel(), 3)
        self.assertEqual(partition["root_occupancy_bins"]["2"], 1)

    def test_mixed_root_splits_when_offset_savings_pay_compact_topology(self):
        roots = torch.full((8,), 17)
        anchors = torch.tensor([
            [0.05, 0.1, 0.1], [0.10, 0.1, 0.1],
            [0.15, 0.1, 0.1], [0.20, 0.1, 0.1],
            [0.55, 0.1, 0.1], [0.60, 0.1, 0.1],
            [0.65, 0.1, 0.1], [0.70, 0.1, 0.1],
        ])
        evidence = torch.tensor([True, True, True, True,
                                 False, False, False, False])
        partition = build_adaptive_gate_partition(
            roots, anchors, evidence, voxel_size=torch.tensor(1.0),
            max_levels=4, min_points=2, offset_bits_per_point=96)
        self.assertEqual(partition["split_count"], 1)
        self.assertEqual(partition["terminal_group_count"], 2)
        self.assertEqual(partition["nodes"][0]["state"], SPLIT)
        self.assertGreater(partition["net_predicted_saved_bits"], 0)
        self.assertEqual(partition["topology_bits"], 3)
        self.assertEqual(partition["group_gate_overhead_bits"], 1)
        self.assertEqual(partition["point_group_ids"].numel(), 8)

        group_gates = torch.tensor([0.9, 0.01])
        payload = finalize_gate_partition(partition, group_gates)
        self.assertEqual(payload["terminal_regions"][0]["state"], UPDATE)
        self.assertEqual(payload["terminal_regions"][1]["state"], OFF)
        self.assertEqual(payload["nodes"][0]["state"], SPLIT)
        self.assertEqual(payload["state_codes"], {
            "OFF": OFF, "UPDATE": UPDATE, "SPLIT": SPLIT,
        })

    def test_weak_offset_savings_decline_split(self):
        roots = torch.full((8,), 3)
        anchors = torch.tensor([
            [0.05, 0.1, 0.1], [0.10, 0.1, 0.1],
            [0.15, 0.1, 0.1], [0.20, 0.1, 0.1],
            [0.55, 0.1, 0.1], [0.60, 0.1, 0.1],
            [0.65, 0.1, 0.1], [0.70, 0.1, 0.1],
        ])
        evidence = torch.tensor([True, True, True, True,
                                 False, False, False, False])
        partition = build_adaptive_gate_partition(
            roots, anchors, evidence, voxel_size=1.0,
            min_points=2, offset_bits_per_point=1)
        self.assertEqual(partition["split_count"], 0)
        self.assertEqual(partition["terminal_group_count"], 1)
        self.assertEqual(partition["net_predicted_saved_bits"], 0)

    def test_recursion_is_limited_to_four_levels(self):
        roots = torch.zeros(16, dtype=torch.long)
        anchors = torch.tensor([
            [0.01, 0.1, 0.1], [0.02, 0.1, 0.1],
            [0.03, 0.1, 0.1], [0.04, 0.1, 0.1],
            [0.15, 0.1, 0.1], [0.16, 0.1, 0.1],
            [0.17, 0.1, 0.1], [0.18, 0.1, 0.1],
            [0.30, 0.1, 0.1], [0.31, 0.1, 0.1],
            [0.32, 0.1, 0.1], [0.33, 0.1, 0.1],
            [0.60, 0.1, 0.1], [0.61, 0.1, 0.1],
            [0.62, 0.1, 0.1], [0.63, 0.1, 0.1],
        ])
        evidence = torch.tensor([
            True, True, False, False,
            False, False, False, False,
            False, False, False, False,
            False, False, False, False,
        ])
        partition = build_adaptive_gate_partition(
            roots, anchors, evidence, voxel_size=1.0,
            max_levels=4, min_points=2, offset_bits_per_point=96)
        self.assertEqual(partition["max_level"], 3)
        self.assertEqual(partition["split_count"], 3)
        with self.assertRaises(ValueError):
            build_adaptive_gate_partition(
                roots, anchors, evidence, voxel_size=1.0, max_levels=5)

    def test_split_paths_reconstruct_exact_group_mapping(self):
        roots = torch.tensor([8] * 8 + [11] * 4)
        anchors = torch.tensor([
            [0.05, 0.1, 0.1], [0.10, 0.1, 0.1],
            [0.15, 0.1, 0.1], [0.20, 0.1, 0.1],
            [0.55, 0.1, 0.1], [0.60, 0.1, 0.1],
            [0.65, 0.1, 0.1], [0.70, 0.1, 0.1],
            [2.10, 0.1, 0.1], [2.20, 0.1, 0.1],
            [2.30, 0.1, 0.1], [2.40, 0.1, 0.1],
        ])
        evidence = torch.tensor([True] * 4 + [False] * 8)
        partition = build_adaptive_gate_partition(
            roots, anchors, evidence, voxel_size=1.0,
            min_points=2, offset_bits_per_point=96)
        reconstructed = reconstruct_gate_groups(
            roots, anchors, voxel_size=torch.tensor(1.0),
            split_paths=partition["split_paths"], max_levels=4)
        torch.testing.assert_close(
            reconstructed, partition["point_group_ids"])
        self.assertEqual(torch.unique(reconstructed).numel(),
                         partition["terminal_group_count"])

    @staticmethod
    def _root_only_partition(anchors, evidence):
        roots = torch.full((anchors.shape[0],), 7, dtype=torch.long)
        partition = build_adaptive_gate_partition(
            roots, anchors, evidence, voxel_size=1.0, max_levels=1,
            min_points=2, offset_bits_per_point=96)
        partition["max_levels"] = 4
        partition["refinement_stats"] = []
        return roots, partition

    def test_v7b_stages_require_active_parent_and_residual_heterogeneity(self):
        frozen_anchors = torch.tensor([
            [0.05, 0.1, 0.1], [0.30, 0.1, 0.1], [0.60, 0.1, 0.1],
        ])
        evidence = torch.tensor([True, False, False])
        residual = torch.tensor([0.50, 0.05, 0.05])
        roots, partition = self._root_only_partition(
            frozen_anchors, evidence)

        # An OFF fitted root has no offset payload to save: do not split it.
        off_partition, _, off_stats = refine_adaptive_gate_partition(
            partition, roots, frozen_anchors, evidence, residual,
            torch.tensor([False]), 1.0, target_level=0,
            residual_ratio=0.8)
        self.assertEqual(off_stats["rate_accepted_groups"], 0)
        self.assertEqual(off_partition["topology_bits"], 0)

        # A mixed root with no local residual separation is also kept coarse.
        roots, partition = self._root_only_partition(frozen_anchors, evidence)
        _, _, flat_stats = refine_adaptive_gate_partition(
            partition, roots, frozen_anchors, evidence,
            torch.tensor([0.1, 0.1, 0.1]), torch.tensor([True]),
            1.0, target_level=0, residual_ratio=0.8)
        self.assertEqual(flat_stats["residual_heterogeneous_groups"], 0)
        self.assertEqual(flat_stats["rate_accepted_groups"], 0)

        roots, partition = self._root_only_partition(frozen_anchors, evidence)
        partition, parent_ids, first_stats = refine_adaptive_gate_partition(
            partition, roots, frozen_anchors, evidence, residual,
            torch.tensor([True]), 1.0, target_level=0,
            residual_ratio=0.8)
        self.assertEqual(first_stats["rate_accepted_groups"], 1)
        self.assertEqual(parent_ids.tolist(), [0, 0])
        self.assertEqual(first_stats["topology_bits"], 3)
        self.assertEqual(first_stats["group_gate_overhead_bits"], 1)

        # The next scheduled stage can split only the mixed level-one child.
        partition, parent_ids, second_stats = refine_adaptive_gate_partition(
            partition, roots, frozen_anchors, evidence, residual,
            torch.tensor([True, True]), 1.0, target_level=1,
            residual_ratio=0.8)
        self.assertEqual(second_stats["rate_accepted_groups"], 1)
        self.assertEqual(parent_ids.tolist(), [0, 0, 1])
        self.assertEqual(second_stats["topology_bits"], 4)
        self.assertEqual(second_stats["group_gate_overhead_bits"], 2)
        self.assertEqual(partition["terminal_group_count"], 3)
        self.assertTrue(torch.all(partition["point_group_ids"] >= 0))
        self.assertEqual(
            partition["topology_bits"],
            sum(node["topology_bits"] for node in partition["nodes"]))

    def test_v7b_wide_ancestor_path_uses_actual_branch_bits(self):
        anchors = torch.tensor(
            [[0.05, 0.1, 0.1], [0.30, 0.1, 0.1]]
            + [[0.55 + 0.5 * index, 0.1, 0.1]
               for index in range(8)])
        evidence = torch.tensor([True] + [False] * 9)
        residual = torch.tensor([0.5] + [0.01] * 9)
        roots, partition = self._root_only_partition(anchors, evidence)
        partition, _, first_stats = refine_adaptive_gate_partition(
            partition, roots, anchors, evidence, residual,
            torch.tensor([True]), 1.0, target_level=0,
            residual_ratio=0.8)
        self.assertEqual(first_stats["oversized_spatial_groups"], 1)
        self.assertEqual(first_stats["rate_accepted_groups"], 1)
        self.assertEqual(partition["terminal_group_count"], 9)

        partition, _, second_stats = refine_adaptive_gate_partition(
            partition, roots, anchors, evidence, residual,
            torch.tensor([True] + [False] * 8), 1.0,
            target_level=1, residual_ratio=0.8)
        # The split parent is root child 0 of a 9-way occupied partition, so
        # its path ordinal costs ceil(log2(9))=4 bits, not a fixed 3 bits.
        self.assertEqual(second_stats["rate_accepted_groups"], 1)
        self.assertEqual(second_stats["topology_bits"], 7)
        self.assertEqual(partition["split_count"], 2)
        group_gates = torch.tensor([
            0.1 * index
            for index in range(partition["terminal_group_count"])
        ])
        compact = build_compact_gate_payload(
            partition, group_gates, frame_idx=3,
            init_pos=torch.zeros(3), voxel_size=1.0)
        groups, decoded_gates, _ = decode_compact_gate_payload(
            compact, roots, anchors)
        self.assertTrue(torch.equal(groups, partition["point_group_ids"]))
        self.assertTrue(torch.equal(decoded_gates, group_gates))

    def test_compact_sidecar_round_trips_group_mapping_and_gate_levels(self):
        roots = torch.tensor([20] * 8 + [40] * 3)
        anchors = torch.tensor([
            [0.05, 0.1, 0.1], [0.10, 0.1, 0.1],
            [0.15, 0.1, 0.1], [0.20, 0.1, 0.1],
            [0.55, 0.1, 0.1], [0.60, 0.1, 0.1],
            [0.65, 0.1, 0.1], [0.70, 0.1, 0.1],
            [2.10, 0.1, 0.1], [2.20, 0.1, 0.1], [2.30, 0.1, 0.1],
        ])
        evidence = torch.tensor([True] * 4 + [False] * 7)
        partition = build_adaptive_gate_partition(
            roots, anchors, evidence, voxel_size=1.0,
            max_levels=4, offset_bits_per_point=96)
        gate_values = torch.tensor([
            0.1 * index for index in range(partition["terminal_group_count"])
        ])
        compact = build_compact_gate_payload(
            partition, gate_values, frame_idx=2,
            init_pos=torch.zeros(3), voxel_size=1.0)
        groups, decoded_group_gates, point_gates = decode_compact_gate_payload(
            compact, roots, anchors)
        self.assertTrue(torch.equal(groups, partition["point_group_ids"]))
        self.assertTrue(torch.equal(decoded_group_gates, gate_values))
        self.assertTrue(torch.equal(
            point_gates, gate_values[partition["point_group_ids"]]))
        self.assertIn("split_paths", compact)
        self.assertIn("gate_levels", compact)
        self.assertNotIn("point_group_ids", compact)
        self.assertNotIn("nodes", compact)

    def test_compact_sidecar_round_trips_real_v7b_smoke_payload(self):
        repo_root = Path(__file__).resolve().parents[1]
        smoke_payload_path = Path(os.environ.get(
            "S2GS_V7B_SMOKE_PAYLOAD",
            repo_root / "output/v7b_smoke_20260925_01"
            "/adaptive_gate_partition/frame_0002.pt"))
        snapshot_path = Path(os.environ.get(
            "S2GS_FRAME1_SNAPSHOT",
            repo_root / "output/paired_smoke_20260925_01_frame1.pt"))
        if not smoke_payload_path.is_file() or not snapshot_path.is_file():
            self.skipTest(
                "real-smoke round trip requires S2GS_V7B_SMOKE_PAYLOAD and "
                "S2GS_FRAME1_SNAPSHOT artifacts")

        old_payload = torch.load(
            smoke_payload_path, map_location="cpu", weights_only=False)
        snapshot = torch.load(
            snapshot_path, map_location="cpu", weights_only=True)
        roots = snapshot["model_fields"]["root_indices"]["value"]
        anchors = snapshot["latents"]["anchor"]["value"]
        group_gates = torch.tensor([
            region["gate_value"]
            for region in old_payload["terminal_regions"]], dtype=torch.float32)
        partition = {
            "root_values": old_payload["root_values"],
            "split_paths": old_payload["compact_topology"]["split_paths"],
            "terminal_group_count": old_payload["terminal_group_count"],
            "max_levels": old_payload["max_levels"],
        }
        compact = build_compact_gate_payload(
            partition, group_gates, old_payload["frame_idx"],
            old_payload["grid_origin"], old_payload["root_cell_side"])
        groups, decoded_gates, point_gates = decode_compact_gate_payload(
            compact, roots, anchors)
        reference_groups = old_payload["point_group_ids"]
        reference_point_gates = group_gates.index_select(0, reference_groups)
        self.assertTrue(torch.equal(groups, reference_groups))
        self.assertTrue(torch.equal(decoded_gates, group_gates))
        self.assertTrue(torch.equal(point_gates, reference_point_gates))

        compact_stream = io.BytesIO()
        torch.save(compact, compact_stream)
        verbose_bytes = smoke_payload_path.stat().st_size
        self.assertLess(compact_stream.tell(), verbose_bytes // 10)

    def test_v7b_reconstruction_uses_frozen_previous_frame_anchors(self):
        frozen_anchors = torch.tensor([
            [0.05, 0.1, 0.1], [0.30, 0.1, 0.1], [0.60, 0.1, 0.1],
        ])
        evidence = torch.tensor([True, False, False])
        residual = torch.tensor([0.50, 0.05, 0.05])
        roots, partition = self._root_only_partition(
            frozen_anchors, evidence)
        partition, _, _ = refine_adaptive_gate_partition(
            partition, roots, frozen_anchors, evidence, residual,
            torch.tensor([True]), 1.0, target_level=0,
            residual_ratio=0.8)
        live_anchors_after_training = frozen_anchors.clone()
        live_anchors_after_training[1, 0] = 0.55
        self.assertFalse(torch.equal(frozen_anchors,
                                     live_anchors_after_training))
        reconstructed = reconstruct_gate_groups(
            roots, frozen_anchors, 1.0,
            split_paths=partition["split_paths"], max_levels=4)
        torch.testing.assert_close(reconstructed, partition["point_group_ids"])
        live_reconstruction = reconstruct_gate_groups(
            roots, live_anchors_after_training, 1.0,
            split_paths=partition["split_paths"], max_levels=4)
        self.assertFalse(torch.equal(live_reconstruction,
                                    partition["point_group_ids"]))

    def test_active_offset_storage_uses_per_point_mask(self):
        offsets = torch.zeros((5, 3), dtype=torch.float32)
        active = torch.tensor([True, False, True, False, False])
        self.assertEqual(adaptive_gate_offset_size_bits(offsets, active), 192)

    def test_child_gate_logits_and_adam_state_inherit_parent_exactly(self):
        old_inverse = torch.tensor([0, 0, 1])
        old_gate = Gate(2, device="cpu", inverse_indices=old_inverse)
        with torch.no_grad():
            old_gate.log_alphas.copy_(torch.tensor([0.7, -0.4]))
        old_gate.optimizer.state[old_gate.log_alphas] = {
            "step": torch.tensor(5.0),
            "exp_avg": torch.tensor([0.2, -0.3]),
            "exp_avg_sq": torch.tensor([0.04, 0.09]),
        }

        new_inverse = torch.tensor([0, 1, 2])
        new_gate = Gate(3, device="cpu", inverse_indices=new_inverse)
        inherit_gate_parameters(old_gate, new_gate, torch.tensor([0, 0, 1]))
        self.assertTrue(torch.equal(
            new_gate.log_alphas, torch.tensor([0.7, 0.7, -0.4])))
        old_point_gates = old_gate.sample_gate_Gumbel_Softmax(
            stochastic=False).index_select(0, old_inverse)
        new_point_gates = new_gate.sample_gate_Gumbel_Softmax(
            stochastic=False).index_select(0, new_inverse)
        self.assertTrue(torch.equal(old_point_gates, new_point_gates))
        state = new_gate.optimizer.state[new_gate.log_alphas]
        self.assertTrue(torch.equal(state["exp_avg"],
                                    torch.tensor([0.2, 0.2, -0.3])))
        self.assertTrue(torch.equal(state["exp_avg_sq"],
                                    torch.tensor([0.04, 0.04, 0.09])))

    def test_v7b_sq_res_attribute_can_reactivate_after_previous_off_gate(self):
        previous_off_gate = torch.zeros(2)
        v7b_mask = residual_decoder_gate_for_attribute(
            previous_off_gate, adaptive_v7b=True,
            attribute_gate_enabled=False, attribute_name="f_dc")
        legacy_mask = residual_decoder_gate_for_attribute(
            previous_off_gate, adaptive_v7b=False,
            attribute_gate_enabled=False, sqres_floor=0.1,
            attribute_name="f_dc")
        self.assertIsNone(v7b_mask)
        self.assertIs(legacy_mask, previous_off_gate)

        decoder = LatentDecoderRes(
            latent_dim=1, feature_dim=1, ldecode_matrix="learnable",
            use_shift=False, latent_norm="none", num_layers_dec=0,
            hidden_dim_dec=1, activation="none", final_activation="none",
            ldec_std=0.1, gates=v7b_mask)
        decoder.identity = False
        decoder.frame_idx = 2
        decoder.decoded_att = torch.zeros((2, 1))
        decoder.layers = torch.nn.Identity()
        decoder.div.data.fill_(1.0)
        residual = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
        decoder(residual).sum().backward()
        self.assertTrue(torch.equal(residual.grad, torch.ones((2, 1))))

    def test_v7c_sqres_floor_clamps_only_unconfigured_target_attributes(self):
        previous_gate = torch.tensor([0.0, 0.05, 0.4, 1.0])
        expected = torch.tensor([0.1, 0.1, 0.4, 1.0])
        for attribute in ("f_dc", "sc", "rot", "op"):
            with self.subTest(attribute=attribute):
                actual = residual_decoder_gate_for_attribute(
                    previous_gate, adaptive_v7b=True,
                    attribute_gate_enabled=False, sqres_floor=0.1,
                    attribute_name=attribute)
                self.assertTrue(torch.equal(actual, expected))

    def test_v7c_floor_does_not_change_configured_gate_or_other_attributes(self):
        previous_gate = torch.tensor([0.0, 0.5])
        configured = residual_decoder_gate_for_attribute(
            previous_gate, adaptive_v7b=True, attribute_gate_enabled=True,
            sqres_floor=0.1, attribute_name="f_dc")
        unsupported_attribute = residual_decoder_gate_for_attribute(
            previous_gate, adaptive_v7b=True, attribute_gate_enabled=False,
            sqres_floor=0.1, attribute_name="f_rest")
        self.assertIs(configured, previous_gate)
        self.assertIsNone(unsupported_attribute)

    def test_v7c_floor_with_no_previous_gate_stays_none(self):
        self.assertIsNone(residual_decoder_gate_for_attribute(
            None, adaptive_v7b=True, attribute_gate_enabled=False,
            sqres_floor=0.1, attribute_name="f_dc"))

    def test_v7c_rejects_invalid_floor(self):
        with self.assertRaises(ValueError):
            residual_decoder_gate_for_attribute(
                torch.zeros(1), adaptive_v7b=True,
                attribute_gate_enabled=False, sqres_floor=1.1,
                attribute_name="f_dc")

    def test_v7d_cue_floor_reopens_only_old_off_cue_positive_points(self):
        previous_gate = torch.tensor([0.0, 0.05, 0.051, 0.7, 0.0])
        evidence = torch.tensor([True, True, True, True, False])
        expected = torch.tensor([0.1, 0.1, 0.051, 0.7, 0.0])
        for attribute in ("f_dc", "sc", "rot", "op"):
            with self.subTest(attribute=attribute):
                actual = residual_decoder_gate_for_attribute(
                    previous_gate, adaptive_v7b=True,
                    attribute_gate_enabled=False, sqres_cue_floor=0.1,
                    point_evidence=evidence, attribute_name=attribute)
                self.assertTrue(torch.equal(actual, expected))

    def test_v7d_reopened_mask_allows_sq_res_gradient_only_at_cued_points(self):
        previous_gate = torch.tensor([0.0, 0.4, 0.0])
        evidence = torch.tensor([True, False, False])
        gate = residual_decoder_gate_for_attribute(
            previous_gate, adaptive_v7b=True,
            attribute_gate_enabled=False, sqres_cue_floor=0.1,
            point_evidence=evidence, attribute_name="f_dc")
        decoder = LatentDecoderRes(
            latent_dim=1, feature_dim=1, ldecode_matrix="learnable",
            use_shift=False, latent_norm="none", num_layers_dec=0,
            hidden_dim_dec=1, activation="none", final_activation="none",
            ldec_std=0.1, gates=gate)
        decoder.identity = False
        decoder.frame_idx = 2
        decoder.decoded_att = torch.zeros((3, 1))
        decoder.layers = torch.nn.Identity()
        decoder.div.data.fill_(1.0)
        residual = torch.nn.Parameter(torch.ones((3, 1)))
        decoder(residual).sum().backward()
        self.assertTrue(torch.equal(residual.grad, torch.tensor([[0.1], [0.4], [0.0]])))

    def test_v7d_does_not_change_defaults_configured_attributes_or_baseline(self):
        previous_gate = torch.tensor([0.0, 0.4])
        evidence = torch.tensor([True, False])
        # Default V7b behavior remains fully decoupled.
        self.assertIsNone(residual_decoder_gate_for_attribute(
            previous_gate, adaptive_v7b=True,
            attribute_gate_enabled=False, point_evidence=evidence,
            attribute_name="f_dc"))
        # Explicit attribute gates continue to use the configured old gate.
        self.assertIs(previous_gate, residual_decoder_gate_for_attribute(
            previous_gate, adaptive_v7b=True,
            attribute_gate_enabled=True, sqres_cue_floor=0.1,
            point_evidence=evidence, attribute_name="f_dc"))
        # Baseline path still applies its old gate, independent of new options.
        self.assertIs(previous_gate, residual_decoder_gate_for_attribute(
            previous_gate, adaptive_v7b=False,
            attribute_gate_enabled=False, sqres_cue_floor=0.1,
            point_evidence=evidence, attribute_name="f_dc"))
        # Frame 2 with no prior Gate remains unchanged.
        self.assertIsNone(residual_decoder_gate_for_attribute(
            None, adaptive_v7b=True, attribute_gate_enabled=False,
            sqres_cue_floor=0.1, point_evidence=None,
            attribute_name="f_dc"))

    def test_v7d_validates_evidence_and_disallows_both_floor_modes(self):
        with self.assertRaises(ValueError):
            residual_decoder_gate_for_attribute(
                torch.zeros(2), adaptive_v7b=True,
                attribute_gate_enabled=False, sqres_cue_floor=0.1,
                point_evidence=torch.ones(1, dtype=torch.bool),
                attribute_name="f_dc")

    def test_v7d_reopen_mask_roundtrips_raw_zlib_and_empty_encodings(self):
        sparse = torch.zeros(59_839, dtype=torch.bool)
        sparse[::7] = True
        encoded_sparse = encode_sqres_reopen_mask(sparse)
        self.assertEqual(encoded_sparse["encoding"],
                         "bitset_zlib_lsb0_v1")
        decoded_sparse = decode_sqres_reopen_mask(
            encoded_sparse, expected_point_count=sparse.numel())
        self.assertTrue(torch.equal(decoded_sparse, sparse))
        self.assertEqual(encoded_sparse["popcount"], int(sparse.sum()))
        self.assertLessEqual(len(encoded_sparse["blob"]),
                             encoded_sparse["raw_bytes"])

        tiny = torch.tensor([True, False, True, False, True])
        encoded_tiny = encode_sqres_reopen_mask(tiny)
        self.assertEqual(encoded_tiny["encoding"], "bitset_raw_lsb0_v1")
        self.assertTrue(torch.equal(
            decode_sqres_reopen_mask(encoded_tiny), tiny))

        empty = encode_sqres_reopen_mask(torch.zeros(13, dtype=torch.bool))
        self.assertEqual(empty["encoding"], "empty")
        self.assertEqual(empty["blob"], b"")
        self.assertTrue(torch.equal(
            decode_sqres_reopen_mask(empty), torch.zeros(13, dtype=torch.bool)))

    def test_v7d_reopen_mask_rejects_bad_count_padding_and_popcount(self):
        payload = encode_sqres_reopen_mask(torch.tensor([True, False, False]))
        with self.assertRaises(ValueError):
            decode_sqres_reopen_mask(payload, expected_point_count=4)
        bad_padding = dict(payload)
        bad_padding["blob"] = bytes([bad_padding["blob"][0] | 0x80])
        with self.assertRaises(ValueError):
            decode_sqres_reopen_mask(bad_padding)
        bad_count = dict(payload, popcount=2)
        with self.assertRaises(ValueError):
            decode_sqres_reopen_mask(bad_count)
        oversized = {
            "encoding": "bitset_zlib_lsb0_v1",
            "point_count": 8,
            "popcount": 0,
            "raw_bytes": 1,
            "blob": zlib.compress(bytes(100_000)),
        }
        with self.assertRaises(ValueError):
            decode_sqres_reopen_mask(oversized)

    def test_v7d_compact_sidecar_preserves_partition_and_carries_reopen_mask(self):
        roots = torch.tensor([5, 5, 9, 9])
        anchors = torch.tensor([
            [0.05, 0.1, 0.1], [0.65, 0.1, 0.1],
            [2.05, 0.1, 0.1], [2.65, 0.1, 0.1],
        ])
        evidence = torch.tensor([True, False, False, True])
        partition = build_adaptive_gate_partition(
            roots, anchors, evidence, voxel_size=1.0, max_levels=4,
            min_points=2, offset_bits_per_point=96)
        groups = torch.tensor([0.9, 0.1, 0.0, 0.8])[:
            partition["terminal_group_count"]]
        # A real mask is independent of the existing offset Gate stream.
        reopen_mask = torch.tensor([True, False, False, True])
        mask_payload = encode_sqres_reopen_mask(reopen_mask)
        sidecar = build_compact_gate_payload(
            partition, groups, 3, torch.zeros(3), 1.0,
            sqres_reopen_mask_payload=mask_payload)
        self.assertEqual(sidecar["format"],
                         "adaptive_gate_partition_v7d_compact_v1")
        rebuilt_groups, rebuilt_gate_groups, rebuilt_point_gates = (
            decode_compact_gate_payload(sidecar, roots, anchors))
        self.assertTrue(torch.equal(
            rebuilt_groups, partition["point_group_ids"]))
        self.assertTrue(torch.equal(rebuilt_gate_groups, groups))
        self.assertTrue(torch.equal(
            rebuilt_point_gates,
            groups.index_select(0, partition["point_group_ids"])))
        rebuilt_mask = decode_sqres_reopen_mask(
            sidecar["sq_res_reopen_mask"],
            expected_point_count=roots.numel())
        self.assertTrue(torch.equal(rebuilt_mask, reopen_mask))

    def test_v7d_size_charges_encoded_reopen_mask_once(self):
        model = GaussianModel.__new__(GaussianModel)
        model.param_names = ["offset", "f_dc"]
        model._latents = {
            "offset": torch.nn.Parameter(torch.zeros((2, 3))),
            "f_dc": torch.nn.Parameter(torch.zeros((2, 3))),
        }
        model.latent_decoders = {
            "offset": DecoderIdentity(), "f_dc": DecoderIdentity(),
        }
        model.frz_anchor = model.frz_offset = "all"
        model.frz_features_dc = model.frz_features_rest = "all"
        model.frz_scaling = model.frz_rotation = model.frz_opacity = "all"
        for name in ("anchor", "offset", "features_dc", "features_rest",
                     "scaling", "rotation", "opacity"):
            setattr(model, f"mask_{name}", torch.ones((2, 1), dtype=torch.bool))
        model.gate_atts = None
        model.hierarchy_update_mask = None
        model.hierarchy_residual_gate = None
        model.adaptive_gate_partition = None
        model.adaptive_gate_topology_bits = 0
        model.adaptive_gate_sqres_reopen_mask_bits = 8 * 9
        model.gate_params = {"offset": False, "f_dc": False}
        model.frame_idx = 3
        model.shared_motion = None
        model.current_frame_shared_bits = 0
        self.assertEqual(GaussianModel.size(model), 72)
        with self.assertRaises(ValueError):
            residual_decoder_gate_for_attribute(
                torch.zeros(1), adaptive_v7b=True,
                attribute_gate_enabled=False, sqres_floor=0.1,
                sqres_cue_floor=0.1,
                point_evidence=torch.ones(1, dtype=torch.bool),
                attribute_name="f_dc")


if __name__ == "__main__":
    unittest.main()
