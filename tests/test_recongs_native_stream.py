import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_recongs_native_stream.py"
SPEC = importlib.util.spec_from_file_location("recongs_native_stream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeStreamSerializationTests(unittest.TestCase):
    def test_native_index_encoding_defaults_to_legacy_and_env_overrides_config(self):
        self.assertEqual(MODULE.resolve_native_index_encoding(), "legacy")
        self.assertEqual(MODULE.resolve_native_index_encoding(None), "legacy")
        self.assertEqual(
            MODULE.resolve_native_index_encoding("compressed_full"), "compressed_full"
        )
        self.assertEqual(
            MODULE.resolve_native_index_encoding("legacy", "exact_index_delta"),
            "exact_index_delta",
        )
        with self.assertRaisesRegex(ValueError, "native_index_encoding"):
            MODULE.resolve_native_index_encoding("unknown")

    def test_exact_index_delta_first_frame_is_a_decodable_full_packet(self):
        delta = np.arange(21, dtype=np.float32).reshape(3, 7)
        index = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
        transition = MODULE.normalize_transition(3, 0, np.empty(0, dtype=np.int64))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "first.npz"
            info = MODULE.write_exact_index_delta_packet(
                path, delta, index, transition, previous_index=None
            )
            self.assertEqual(info["index_encoding"], "full")
            self.assertEqual(info["index_encoding_code"], 0)
            self.assertIsNone(info["exact_delta_candidate_bytes"])
            self.assertGreater(info["full_compress_wall_s"], 0.0)
            self.assertEqual(info["delta_compress_wall_s"], 0.0)
            with np.load(path, allow_pickle=False) as packet:
                self.assertIn("index", packet.files)
                self.assertNotIn("index_birth_values", packet.files)
                decoded = MODULE.decode_topology_index(packet, None, transition)
        np.testing.assert_array_equal(decoded, index.astype(np.int32))

    def test_exact_index_delta_rejects_malformed_snapshot_before_writing(self):
        transition = MODULE.normalize_transition(2, 0, np.empty(0, dtype=np.int64))
        valid_delta = np.zeros((3, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with self.assertRaisesRegex(ValueError, r"shape \[anchors, 7\]"):
                MODULE.write_exact_index_delta_packet(
                    folder / "bad-delta.npz", np.zeros((3, 6), dtype=np.float32),
                    np.zeros((2, 2), dtype=np.int64), transition, None
                )
            with self.assertRaisesRegex(ValueError, "transition survivors"):
                MODULE.write_exact_index_delta_packet(
                    folder / "bad-rows.npz", valid_delta,
                    np.zeros((2, 1), dtype=np.int64), transition, None
                )
            with self.assertRaisesRegex(ValueError, "outside delta"):
                MODULE.write_exact_index_delta_packet(
                    folder / "bad-index.npz", valid_delta,
                    np.array([[0, 3], [1, 2]], dtype=np.int64), transition, None
                )
            self.assertFalse((folder / "bad-delta.npz").exists())
            self.assertFalse((folder / "bad-rows.npz").exists())
            self.assertFalse((folder / "bad-index.npz").exists())

    def test_transition_preserves_append_then_prune_row_order(self):
        result = MODULE.normalize_transition(5, 2, np.array([6, 0]))
        np.testing.assert_array_equal(result["deleted_source_indices"], [0, 6])
        self.assertEqual(result["source_rows"], 7)
        self.assertEqual(result["survivor_rows"], 5)
        keep = np.ones(result["source_rows"], dtype=bool)
        keep[result["deleted_source_indices"]] = False
        np.testing.assert_array_equal(np.flatnonzero(keep), [1, 2, 3, 4, 5])

    def test_transition_rejects_duplicate_and_out_of_range_deletions(self):
        with self.assertRaises(ValueError):
            MODULE.normalize_transition(4, 1, np.array([2, 2]))
        with self.assertRaises(ValueError):
            MODULE.normalize_transition(4, 1, np.array([5]))

    def test_packet_keeps_native_dtypes_and_stores_without_compression(self):
        delta = np.arange(21, dtype=np.float32).reshape(3, 7)
        index = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        transition = MODULE.normalize_transition(3, 1, np.array([1]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "motion.npz"
            MODULE.write_native_packet(path, delta, index, transition)
            self.assertGreater(path.stat().st_size, 0)
            with zipfile.ZipFile(path) as archive:
                self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED
                                    for item in archive.infolist()))
            with np.load(path, allow_pickle=False) as packet:
                self.assertEqual(packet["delta"].dtype, np.float32)
                self.assertEqual(packet["index"].dtype, np.int64)
                np.testing.assert_array_equal(packet["delta"], delta)
                np.testing.assert_array_equal(packet["index"], index)
                np.testing.assert_array_equal(packet["deleted_source_indices"], [1])
                self.assertNotIn("survivor_source_indices", packet.files)

    def test_inherited_topology_packet_roundtrips_and_saves_actual_index_bytes(self):
        rng = np.random.default_rng(19)
        previous_index = rng.integers(0, 256, size=(3, 1000), dtype=np.int64)
        transition = MODULE.normalize_transition(
            1000, 5, np.array([4, 28, 80, 1000, 1002, 1003, 1004])
        )
        keep = np.ones(transition["source_rows"], dtype=bool)
        keep[transition["deleted_source_indices"]] = False
        survivors = np.flatnonzero(keep)
        old_rows = survivors[survivors < 1000]
        birth_values = np.array([[8], [9], [10]], dtype=np.int64)
        full_index = np.concatenate((previous_index[:, old_rows], birth_values), axis=1)
        delta = np.zeros((256, 7), dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            full_path = folder / "full.npz"
            reuse_path = folder / "reuse.npz"
            full_bytes = MODULE.write_native_packet(
                full_path, delta, full_index, transition
            )
            reuse_bytes = MODULE.write_native_packet(
                reuse_path, delta, full_index, transition,
                previous_index=previous_index,
            )
            self.assertLess(reuse_bytes, full_bytes)
            with np.load(reuse_path, allow_pickle=False) as packet:
                self.assertNotIn("index", packet.files)
                np.testing.assert_array_equal(packet["index_birth_values"], birth_values)
                decoded = MODULE.decode_topology_index(
                    packet, previous_index, transition
                )
            np.testing.assert_array_equal(decoded, full_index)

    def test_native_field_reconstructs_translation_and_normalized_rotation(self):
        delta = np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 2.0],
        ], dtype=np.float32)
        index = np.array([[0, 1], [1, 0]], dtype=np.int64)
        decoded = MODULE.decode_native_field(delta, index)
        np.testing.assert_allclose(decoded[:, :3], [[5, 7, 9], [5, 7, 9]])
        np.testing.assert_allclose(decoded[:, 3:], [[0, 0, 0, 3], [0, 0, 0, 3]])
        error = MODULE.verify_native_application(
            delta,
            index,
            np.array([[5, 7, 9], [5, 7, 9]], dtype=np.float32),
            np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32),
        )
        self.assertEqual(error, 0.0)

    def test_frozen_native_snapshot_survives_live_optimizer_mutation(self):
        import torch

        live_delta = torch.tensor([[1.0, 2, 3, 0, 0, 0, 1]], dtype=torch.float32)
        live_index = torch.tensor([[0, 0]], dtype=torch.int64)
        saved_delta = live_delta.detach().clone()
        saved_index = live_index.detach().clone()
        with torch.no_grad():
            live_delta.add_(10)
            live_index.zero_()
        np.testing.assert_array_equal(saved_delta.numpy()[0, :3], [1, 2, 3])
        np.testing.assert_array_equal(saved_index.numpy(), [[0, 0]])

    def test_exact_index_delta_is_lossless_and_selected_by_actual_npz_size(self):
        rng = np.random.default_rng(90210)
        previous_index = rng.integers(
            0, 4096, size=(3, 5000), dtype=np.int64
        )
        original_previous = previous_index.copy()
        transition = MODULE.normalize_transition(
            5000, 3, np.array([0, 4999, 5001], dtype=np.int64)
        )
        keep = np.ones(transition["source_rows"], dtype=bool)
        keep[transition["deleted_source_indices"]] = False
        survivors = np.flatnonzero(keep)
        old_rows = survivors[survivors < 5000]
        inherited = previous_index[:, old_rows]
        current_old = inherited.copy()
        current_old[0, [5, 81]] = [3001, 3500]
        fresh_rows = np.array([[61, 62], [71, 72], [81, 82]], dtype=np.int64)
        current_index = np.concatenate((current_old, fresh_rows), axis=1)
        delta = rng.standard_normal((4096, 7), dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            full_path = folder / "full.npz"
            delta_path = folder / "exact-delta.npz"
            full_info = MODULE.write_compressed_full_packet(
                full_path, delta, current_index, transition
            )
            delta_info = MODULE.write_exact_index_delta_packet(
                delta_path, delta, current_index, transition, previous_index
            )
            deleted = MODULE.checked_int32(
                transition["deleted_source_indices"], "deleted_source_indices"
            )
            changed = current_old != inherited
            changed_cells = np.flatnonzero(changed.reshape(-1)).astype(np.int32)
            changed_values = current_old.reshape(-1)[changed_cells].astype(np.int32)
            fresh_values = current_index[:, old_rows.size:]
            full_candidate, _ = MODULE.compressed_npz_bytes({
                "delta": delta,
                "index": MODULE.checked_int32(current_index, "index"),
                "deleted_source_indices": deleted,
                "index_encoding": np.asarray(0, dtype=np.uint8),
            })
            exact_candidate, _ = MODULE.compressed_npz_bytes({
                "delta": delta,
                "deleted_source_indices": deleted,
                "index_changed_cells": changed_cells,
                "index_changed_values": changed_values,
                "index_birth_values": fresh_values.astype(np.int32),
                "index_encoding": np.asarray(2, dtype=np.uint8),
            })
            full_probe, _ = MODULE.compressed_npz_bytes({
                "index": MODULE.checked_int32(current_index, "index"),
                "index_encoding": np.asarray(0, dtype=np.uint8),
            })
            exact_probe, _ = MODULE.compressed_npz_bytes({
                "index_changed_cells": changed_cells,
                "index_changed_values": changed_values,
                "index_birth_values": fresh_values.astype(np.int32),
                "index_encoding": np.asarray(2, dtype=np.uint8),
            })
            self.assertEqual(delta_info["index_encoding"], "exact_index_delta")
            self.assertEqual(delta_info["index_encoding_code"], 2)
            self.assertLess(delta_path.stat().st_size, full_path.stat().st_size)
            self.assertEqual(delta_info["bytes"], delta_path.stat().st_size)
            self.assertEqual(full_info["index_encoding_code"], 0)
            self.assertGreater(full_info["full_compress_wall_s"], 0.0)
            self.assertGreater(delta_info["full_compress_wall_s"], 0.0)
            self.assertGreater(delta_info["delta_compress_wall_s"], 0.0)
            self.assertEqual(delta_info["full_candidate_bytes"], len(full_candidate))
            self.assertEqual(delta_info["exact_delta_candidate_bytes"], len(exact_candidate))
            self.assertEqual(full_info["bytes"], len(full_candidate))
            self.assertEqual(
                len(full_candidate) - len(exact_candidate),
                len(full_probe) - len(exact_probe),
            )
            with zipfile.ZipFile(delta_path) as archive:
                self.assertTrue(all(
                    item.compress_type == zipfile.ZIP_DEFLATED
                    for item in archive.infolist()
                ))
            with np.load(delta_path, allow_pickle=False) as packet:
                self.assertEqual(packet["delta"].dtype, np.float32)
                self.assertEqual(packet["deleted_source_indices"].dtype, np.int32)
                self.assertEqual(packet["index_changed_cells"].dtype, np.int32)
                self.assertEqual(packet["index_changed_values"].dtype, np.int32)
                self.assertEqual(packet["index_birth_values"].dtype, np.int32)
                decoded = MODULE.decode_topology_index(
                    packet, previous_index, transition
                )
                np.testing.assert_array_equal(packet["delta"], delta)
            np.testing.assert_array_equal(decoded, current_index)
            np.testing.assert_array_equal(previous_index, original_previous)

    def test_exact_index_delta_falls_back_to_full_when_diff_file_is_larger(self):
        rng = np.random.default_rng(73)
        previous_index = rng.integers(0, 2048, size=(3, 2000), dtype=np.int64)
        current_index = rng.integers(0, 2048, size=(3, 2000), dtype=np.int64)
        transition = MODULE.normalize_transition(2000, 0, np.empty(0, dtype=np.int64))
        delta = rng.standard_normal((2048, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "selected.npz"
            info = MODULE.write_exact_index_delta_packet(
                path, delta, current_index, transition, previous_index
            )
            self.assertEqual(info["index_encoding"], "full")
            self.assertEqual(info["index_encoding_code"], 0)
            self.assertLessEqual(
                info["full_candidate_bytes"], info["exact_delta_candidate_bytes"]
            )
            deleted = MODULE.checked_int32(
                transition["deleted_source_indices"], "deleted_source_indices"
            )
            changed_cells = np.flatnonzero(previous_index != current_index).astype(np.int32)
            exact_candidate, _ = MODULE.compressed_npz_bytes({
                "delta": delta,
                "deleted_source_indices": deleted,
                "index_changed_cells": changed_cells,
                "index_changed_values": current_index.reshape(-1)[changed_cells].astype(np.int32),
                "index_birth_values": np.empty((3, 0), dtype=np.int32),
                "index_encoding": np.asarray(2, dtype=np.uint8),
            })
            full_candidate, _ = MODULE.compressed_npz_bytes({
                "delta": delta,
                "index": MODULE.checked_int32(current_index, "index"),
                "deleted_source_indices": deleted,
                "index_encoding": np.asarray(0, dtype=np.uint8),
            })
            full_probe, _ = MODULE.compressed_npz_bytes({
                "index": MODULE.checked_int32(current_index, "index"),
                "index_encoding": np.asarray(0, dtype=np.uint8),
            })
            exact_probe, _ = MODULE.compressed_npz_bytes({
                "index_changed_cells": changed_cells,
                "index_changed_values": current_index.reshape(-1)[changed_cells].astype(np.int32),
                "index_birth_values": np.empty((3, 0), dtype=np.int32),
                "index_encoding": np.asarray(2, dtype=np.uint8),
            })
            self.assertEqual(info["full_candidate_bytes"], len(full_candidate))
            self.assertEqual(info["exact_delta_candidate_bytes"], len(exact_candidate))
            self.assertEqual(
                len(full_candidate) - len(exact_candidate),
                len(full_probe) - len(exact_probe),
            )
            self.assertEqual(path.stat().st_size, len(full_candidate))
            with np.load(path, allow_pickle=False) as packet:
                decoded = MODULE.decode_topology_index(packet, None, transition)
                np.testing.assert_array_equal(decoded, current_index.astype(np.int32))

    def test_exact_index_delta_rejects_ids_outside_lossless_int32_range(self):
        transition = MODULE.normalize_transition(2, 0, np.empty(0, dtype=np.int64))
        delta = np.zeros((2, 7), dtype=np.float32)
        index = np.array([[0, np.iinfo(np.int32).max + 1]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "index"):
                MODULE.write_exact_index_delta_packet(
                    Path(folder) / "bad.npz", delta, index, transition, None
                )

    def test_disk_packet_decodes_using_transition_json_without_deleted_row_ids(self):
        rng = np.random.default_rng(811)
        previous_index = rng.integers(0, 4096, size=(3, 1000), dtype=np.int64)
        transition = MODULE.normalize_transition(1000, 3, np.array([0, 999, 1001]))
        keep = np.ones(transition["source_rows"], dtype=bool)
        keep[transition["deleted_source_indices"]] = False
        survivor_source_rows = np.flatnonzero(keep)
        old_survivors = survivor_source_rows[survivor_source_rows < 1000]
        fresh = np.array([[21, 22], [31, 32], [41, 42]], dtype=np.int64)
        current_index = np.concatenate((previous_index[:, old_survivors], fresh), axis=1)
        delta = rng.standard_normal((4096, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            packet_path = folder / "native_motion_002.npz"
            MODULE.write_exact_index_delta_packet(
                packet_path, delta, current_index, transition, previous_index
            )
            # The JSON only needs the row counts; delete IDs live in the NPZ.
            transition_path = folder / "transition_002.json"
            transition_path.write_text(json.dumps({
                "old_base_rows": 1000,
                "previous_birth_rows": 3,
                "source_rows": 1003,
                "survivor_rows": 1000,
                "deleted_rows": 3,
            }))
            metadata = json.loads(transition_path.read_text())
            with np.load(packet_path, allow_pickle=False) as packet:
                decoded = MODULE.decode_topology_index(
                    packet, previous_index, metadata
                )
                self.assertEqual(packet["index_changed_cells"].dtype, np.int32)
        np.testing.assert_array_equal(decoded, current_index.astype(np.int32))

    def test_exact_index_delta_empty_diff_and_legacy_code_zero(self):
        previous_index = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        transition = MODULE.normalize_transition(3, 0, np.empty(0, dtype=np.int64))
        delta = np.zeros((8, 7), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            full_path = folder / "legacy.npz"
            np.savez(full_path, delta=delta, index=previous_index)
            with np.load(full_path, allow_pickle=False) as legacy:
                np.testing.assert_array_equal(
                    MODULE.decode_topology_index(legacy, None, transition),
                    previous_index,
                )

            empty_delta_path = folder / "empty_diff_code2.npz"
            np.savez_compressed(
                empty_delta_path,
                delta=delta,
                deleted_source_indices=np.empty(0, dtype=np.int32),
                index_changed_cells=np.empty(0, dtype=np.int32),
                index_changed_values=np.empty(0, dtype=np.int32),
                index_birth_values=np.empty((2, 0), dtype=np.int32),
                index_encoding=np.asarray(2, dtype=np.uint8),
            )
            with np.load(empty_delta_path, allow_pickle=False) as packet:
                decoded_empty = MODULE.decode_topology_index(
                    packet, previous_index.astype(np.int32), transition
                )
            np.testing.assert_array_equal(decoded_empty, previous_index.astype(np.int32))

            exact_path = folder / "empty_diff.npz"
            info = MODULE.write_exact_index_delta_packet(
                exact_path, delta, previous_index, transition, previous_index
            )
            with np.load(exact_path, allow_pickle=False) as packet:
                if info["index_encoding_code"] == 2:
                    self.assertEqual(packet["index_changed_cells"].size, 0)
                decoded = MODULE.decode_topology_index(packet, previous_index, transition)
        np.testing.assert_array_equal(decoded, previous_index.astype(np.int32))

    def test_exact_index_delta_decoder_rejects_duplicate_or_invalid_change_cells(self):
        previous_index = np.array([[0, 1], [2, 3]], dtype=np.int32)
        transition = MODULE.normalize_transition(2, 0, np.empty(0, dtype=np.int64))
        base_packet = {
            "delta": np.zeros((4, 7), dtype=np.float32),
            "deleted_source_indices": np.empty(0, dtype=np.int32),
            "index_encoding": np.asarray(2, dtype=np.uint8),
            "index_birth_values": np.empty((2, 0), dtype=np.int32),
            "index_changed_values": np.asarray([1, 2], dtype=np.int32),
        }
        duplicate = dict(base_packet,
                         index_changed_cells=np.asarray([1, 1], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            MODULE.decode_topology_index(duplicate, previous_index, transition)
        out_of_range = dict(base_packet,
                            index_changed_cells=np.asarray([4, 5], dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "outside inherited rows"):
            MODULE.decode_topology_index(out_of_range, previous_index, transition)

    def test_checked_int32_rejects_negative_ids(self):
        with self.assertRaisesRegex(ValueError, "non-negative int32"):
            MODULE.checked_int32(np.array([-1], dtype=np.int64), "negative")


if __name__ == "__main__":
    unittest.main()
