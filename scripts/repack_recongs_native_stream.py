#!/usr/bin/env python3
"""Repack a completed full-index ReCon-GS stream without retraining it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_recongs_native_stream import (  # noqa: E402
    decode_topology_index,
    normalize_transition,
    sha256_file,
    write_compressed_full_packet,
    write_exact_index_delta_packet,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)
    return path.stat().st_size


def repack(
    source_run: Path,
    output_run: Path,
    index_mode: str = "exact_index_delta",
) -> Path:
    if index_mode not in {"exact_index_delta", "compressed_full"}:
        raise ValueError(f"unsupported index mode: {index_mode}")
    source_run = source_run.resolve()
    output_run = output_run.resolve()
    source_metrics = source_run / "stream_metrics"
    source_stream = source_run / "frame_storage"
    metadata_path = source_metrics / "run_metadata.json"
    summary_path = source_metrics / "experiment_summary.json"
    records_path = source_metrics / "frames.jsonl"
    manifest_path = source_stream / "sequence_manifest.json"
    for required in (metadata_path, summary_path, records_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_metadata = load_json(metadata_path)
    source_summary = load_json(summary_path)
    source_manifest = load_json(manifest_path)
    records = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    expected_frames = int(source_summary.get("frames_expected", 0))
    if source_metadata.get("status") != "completed" or source_summary.get("status") != "completed":
        raise ValueError("source ReCon-GS run is not completed")
    if expected_frames < 1 or len(records) != expected_frames:
        raise ValueError(
            f"source has {len(records)} frame records; expected {expected_frames}"
        )
    if [int(row["frame"]) for row in records] != list(range(1, expected_frames + 1)):
        raise ValueError("source frame records must be contiguous and ordered from frame 1")
    if source_manifest.get("frames_completed") != expected_frames:
        raise ValueError("source sequence manifest frame count is incomplete")
    if output_run.exists() and any(output_run.iterdir()):
        raise FileExistsError(f"refusing non-empty output path: {output_run}")
    output_run.mkdir(parents=True, exist_ok=True)
    stream_dir = output_run / "frame_storage"
    point_cloud_dir = output_run / "point_cloud"
    metrics_dir = output_run / "stream_metrics"
    for folder in (stream_dir, point_cloud_dir, metrics_dir):
        folder.mkdir(parents=True, exist_ok=True)

    shared_base_source = Path(source_metadata["shared_base"]["path"]).resolve()
    shared_base_dest = output_run / "shared_frame0.ply"
    if not shared_base_source.is_file():
        raise FileNotFoundError(shared_base_source)
    expected_base_hash = source_metadata["shared_base"].get("sha256")
    if expected_base_hash and sha256_file(shared_base_source) != expected_base_hash:
        raise ValueError("source shared frame-0 base hash does not match metadata")
    shutil.copy2(shared_base_source, shared_base_dest)
    shared_base = {
        **source_metadata["shared_base"],
        "path": "shared_frame0.ply",
        "bytes": shared_base_dest.stat().st_size,
        "sha256": sha256_file(shared_base_dest),
        "meaning": "shared trained frame-0 Gaussian base; counted once",
    }

    output_records = []
    frame_entries = []
    source_previous_index = None
    candidate_previous_index = None
    packet_bytes = birth_bytes = transition_bytes = 0
    encode_wall_total = 0.0
    encoding_counts = {"full": 0, "exact_index_delta": 0}
    encoder_started = time.perf_counter()

    for source_record in records:
        frame = int(source_record["frame"])
        source_packet_path = source_run / source_record["packet"]["path"]
        source_transition_path = source_run / source_record["transition_metadata"]["path"]
        source_birth_path = source_run / source_record["birth_ply"]["path"]
        for path in (source_packet_path, source_transition_path, source_birth_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(source_packet_path, allow_pickle=False) as source_packet:
            if "index" not in source_packet.files:
                if source_previous_index is None:
                    raise ValueError("first source frame must contain a full index")
                source_transition = load_json(source_transition_path)
                transition_state = normalize_transition(
                    int(source_transition["old_base_rows"]),
                    int(source_transition["previous_birth_rows"]),
                    source_packet["deleted_source_indices"],
                )
                full_index = decode_topology_index(
                    source_packet, source_previous_index, transition_state
                )
            else:
                full_index = np.asarray(source_packet["index"])
            delta = np.asarray(source_packet["delta"])
            deleted = np.asarray(source_packet["deleted_source_indices"])
        if full_index.dtype.kind not in "iu" or full_index.ndim != 2:
            raise ValueError(f"frame {frame}: invalid native index array")
        if delta.dtype != np.float32 or delta.ndim != 2 or delta.shape[1] != 7:
            raise ValueError(f"frame {frame}: invalid float32 motion array")
        source_transition = load_json(source_transition_path)
        transition = normalize_transition(
            int(source_transition["old_base_rows"]),
            int(source_transition["previous_birth_rows"]),
            deleted,
        )
        if transition["source_rows"] != int(source_transition["concatenated_source_rows"]):
            raise ValueError(f"frame {frame}: source row count changed")
        if transition["survivor_rows"] != int(source_transition["survivor_rows"]):
            raise ValueError(f"frame {frame}: survivor row count changed")
        if full_index.shape[1] != transition["survivor_rows"]:
            raise ValueError(f"frame {frame}: full index row count differs from transition")
        source_previous_index = np.asarray(full_index).copy()

        packet_path = stream_dir / f"native_motion_{frame:03d}.npz"
        if index_mode == "exact_index_delta":
            serialization = write_exact_index_delta_packet(
                packet_path,
                delta,
                full_index,
                transition,
                previous_index=candidate_previous_index,
            )
        else:
            serialization = write_compressed_full_packet(
                packet_path, delta, full_index, transition
            )
        with np.load(packet_path, allow_pickle=False) as candidate_packet:
            decoded_candidate_index = decode_topology_index(
                candidate_packet, candidate_previous_index, transition
            )
            serialization["stored_array_dtypes"] = {
                key: str(candidate_packet[key].dtype)
                for key in candidate_packet.files
            }
        if not np.array_equal(decoded_candidate_index, full_index):
            raise RuntimeError(
                f"frame {frame}: written candidate did not decode to the source index"
            )
        candidate_previous_index = decoded_candidate_index.copy()
        encode_wall_total += float(serialization["packet_encode_and_write_wall_s"])
        encoding_counts[serialization["index_encoding"]] += 1
        packet_bytes += packet_path.stat().st_size

        birth_path = point_cloud_dir / f"incremental_{frame:03d}.ply"
        shutil.copyfile(source_birth_path, birth_path)
        birth_hash = sha256_file(birth_path)
        birth_bytes += birth_path.stat().st_size

        transition_rel = f"frame_storage/transition_{frame:03d}.json"
        packet_rel = f"frame_storage/native_motion_{frame:03d}.npz"
        birth_rel = f"point_cloud/{birth_path.name}"
        out_transition = {
            "schema": "recongs_exact_index_delta_transition_v1",
            "frame": frame,
            "source_frame": frame - 1,
            "source_row_order": transition["row_order"],
            "old_base_rows": transition["old_base_rows"],
            "previous_birth_rows": transition["previous_birth_rows"],
            "source_rows": transition["source_rows"],
            "deleted_rows": int(transition["deleted_source_indices"].size),
            "survivor_rows": transition["survivor_rows"],
            "birth_row_order": source_transition["birth_ply"]["row_order"],
            "motion": {
                "path": packet_rel,
                "bytes": packet_path.stat().st_size,
                "sha256": sha256_file(packet_path),
                "delta_shape": list(delta.shape),
                "delta_dtype": "float32",
                "index_shape": list(full_index.shape),
                "index_dtype": "int32 after range-checked lossless cast",
                "index_encoding": serialization["index_encoding"],
                "index_encoding_code": serialization["index_encoding_code"],
                "deletion_array": "deleted_source_indices",
                "compression": "np.savez_compressed ZIP_DEFLATED",
                "delta_semantics": source_transition["motion"].get("delta_semantics"),
            },
            "birth_ply": {
                "path": birth_rel,
                "bytes": birth_path.stat().st_size,
                "sha256": birth_hash,
                "rows": int(source_record["birth_ply"]["rows"]),
            },
        }
        transition_path = stream_dir / f"transition_{frame:03d}.json"
        transition_bytes += atomic_json(transition_path, out_transition)

        record = json.loads(json.dumps(source_record))
        record["packet"] = {
            **record["packet"],
            "path": packet_rel,
            "bytes": packet_path.stat().st_size,
            "index_dtype": "int32",
            "index_encoding": serialization["index_encoding"],
            "index_encoding_code": serialization["index_encoding_code"],
            "compression": "np.savez_compressed ZIP_DEFLATED",
            "serializer_candidates": {
                key: value for key, value in serialization.items()
                if key not in {"bytes", "index_encoding", "index_encoding_code", "compression"}
            },
        }
        record["birth_ply"]["path"] = birth_rel
        record["birth_ply"]["bytes"] = birth_path.stat().st_size
        record["transition_metadata"] = {
            "path": transition_rel,
            "bytes": transition_path.stat().st_size,
        }
        record.setdefault("timing", {})["offline_exact_packet_encode_write_wall_s"] = float(
            serialization["packet_encode_and_write_wall_s"]
        )
        record["timing"]["offline_repack_not_in_training_or_frame_wall"] = True
        output_records.append(record)
        frame_entries.append({
            "frame": frame,
            "motion": packet_rel,
            "transition": transition_rel,
            "birth": birth_rel,
        })

    elapsed = time.perf_counter() - encoder_started
    manifest = {
        "schema": f"recongs_native_{index_mode}_stream_v1",
        "status": "completed",
        "frame_range": [1, expected_frames],
        "frame_count_expected": expected_frames,
        "frames_completed": expected_frames,
        "shared_base": shared_base,
        "index_decoder": {
            "packet_code_0": "full int32 index [grid_level, surviving_gaussian_row]",
            "packet_code_2": (
                "carry prior decoded assignments for surviving old-base rows; apply exact "
                "changed-cell values in C-order; append index_birth_values for rows without "
                "a carried prior assignment"
            ),
            "transition_inputs": "old_base_rows, previous_birth_rows, source_rows, deleted_source_indices",
            "decode_order": "frames are decoded sequentially; every result is the full exact index for that frame",
            "serializer": "np.savez_compressed ZIP_DEFLATED; no quantization; float32 motion unchanged",
        },
        "advance_rule": "append prior birth PLY after prior base; remove deleted_source_indices in source order; retain survivors in ascending source row order; apply this frame motion; keep this frame birth PLY as the next transition input",
        "frames": frame_entries,
    }
    manifest_bytes = atomic_json(stream_dir / "sequence_manifest.json", manifest)
    frame_metrics = metrics_dir / "frames.jsonl"
    frame_metrics.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output_records))
    transition_manifest = frame_metrics.stat().st_size
    run_metadata = {
        "status": "completed",
        "operation": f"offline {index_mode} repack; no model retraining",
        "source_run_dir": str(source_run),
        "source_experiment_summary": source_summary,
        "native_index_encoding": index_mode,
        "index_values_exact": True,
        "training_algorithm_changed": False,
        "shared_base": shared_base,
        "run_dir": str(output_run),
    }
    atomic_json(metrics_dir / "run_metadata.json", run_metadata)
    total_stream = packet_bytes + birth_bytes + transition_bytes + manifest_bytes + shared_base["bytes"]
    summary = {
        "status": "completed",
        "operation": f"offline {index_mode} repack; training values and metrics inherited from source run",
        "source_run_dir": str(source_run),
        "frames_completed": expected_frames,
        "frames_expected": expected_frames,
        "index_packet_encodings": encoding_counts,
        "offline_encoder": {
            "total_packet_encode_write_s": encode_wall_total,
            "repack_loop_wall_s_including_copy_hash_json": elapsed,
            "mean_packet_encode_write_ms_per_frame": 1000.0 * encode_wall_total / expected_frames,
            "note": "post-training CPU serialization cost; not included in source training-step or frame-loop timing",
        },
        "actual_stream_bytes": {
            "native_motion_npz_bytes": packet_bytes,
            "incremental_birth_ply_bytes": birth_bytes,
            "required_transition_json_bytes": transition_bytes,
            "required_sequence_manifest_bytes": manifest_bytes,
            "dynamic_stream_bytes": packet_bytes + birth_bytes + transition_bytes + manifest_bytes,
            "shared_frame0_ply_bytes": shared_base["bytes"],
            "total_with_shared_base_bytes": total_stream,
        },
        "source_training_quality_timing_and_renderer_metrics": {
            "mean_quality": source_summary.get("quality_mean"),
            "training_step_wall": source_summary.get("training_steps_wall_synchronized_s"),
            "source_frame_loop_wall": source_summary.get(
                "frame_loop_wall_synchronized_s_including_validation_and_io"
            ),
            "source_renderer_fps": source_summary.get("cam00_synchronized_renderer_fps"),
            "note": "source measurements are unchanged; this offline repack does not measure an integrated live writer or playback FPS",
        },
        "run_dir": str(output_run),
    }
    summary["actual_stream_bytes"]["total_with_shared_base_bytes"] = total_stream
    atomic_json(metrics_dir / "experiment_summary.json", summary)
    return output_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output_run", type=Path)
    parser.add_argument(
        "--index-mode",
        choices=("exact_index_delta", "compressed_full"),
        default="exact_index_delta",
    )
    args = parser.parse_args()
    print(repack(args.source_run, args.output_run, args.index_mode))


if __name__ == "__main__":
    main()
