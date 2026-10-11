#!/usr/bin/env python3
"""Verify a repacked ReCon-GS stream against its full-index source run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.archive.run_recongs_native_stream import decode_topology_index, normalize_transition


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "stream_metrics/frames.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def packet_state(run_dir: Path, record: dict[str, Any], previous: np.ndarray | None):
    packet_path = run_dir / record["packet"]["path"]
    with np.load(packet_path, allow_pickle=False) as packet:
        delta = np.asarray(packet["delta"])
        deleted = np.asarray(packet["deleted_source_indices"])
        transition = read_json(run_dir / record["transition_metadata"]["path"])
        normalized = normalize_transition(
            int(transition["old_base_rows"]),
            int(transition["previous_birth_rows"]),
            deleted,
        )
        advertised_source_rows = transition.get(
            "source_rows", transition.get("concatenated_source_rows")
        )
        if advertised_source_rows is not None and int(advertised_source_rows) != normalized["source_rows"]:
            raise ValueError(f"frame {record['frame']}: JSON source row count disagrees with packet")
        advertised_survivors = transition.get("survivor_rows")
        if advertised_survivors is not None and int(advertised_survivors) != normalized["survivor_rows"]:
            raise ValueError(f"frame {record['frame']}: JSON survivor count disagrees with packet")
        advertised_deleted = transition.get("deleted_rows")
        if advertised_deleted is not None and int(advertised_deleted) != len(deleted):
            raise ValueError(f"frame {record['frame']}: JSON deletion count disagrees with packet")
        code = int(np.asarray(packet["index_encoding"]).item()) if "index_encoding" in packet.files else 0
        record_code = int(record["packet"].get("index_encoding_code", 0))
        transition_code = transition.get("motion", {}).get("index_encoding_code")
        if code != record_code or (transition_code is not None and code != int(transition_code)):
            raise ValueError(f"frame {record['frame']}: encoded packet metadata disagrees")
        decoded = decode_topology_index(packet, previous, normalized)
    if delta.dtype != np.float32 or delta.ndim != 2 or delta.shape[1] != 7:
        raise ValueError(f"frame {record['frame']}: motion is not float32 [anchors,7]")
    if decoded.ndim != 2 or decoded.shape[1] != normalized["survivor_rows"]:
        raise ValueError(f"frame {record['frame']}: decoded index shape is invalid")
    if decoded.size and (decoded.min() < 0 or decoded.max() >= len(delta)):
        raise ValueError(f"frame {record['frame']}: decoded anchor ID is out of range")
    return delta.copy(), deleted.copy(), normalized, decoded.copy()


def verify(reference_run: Path, candidate_run: Path) -> dict[str, Any]:
    reference_run, candidate_run = reference_run.resolve(), candidate_run.resolve()
    ref_meta = read_json(reference_run / "stream_metrics/run_metadata.json")
    cand_meta = read_json(candidate_run / "stream_metrics/run_metadata.json")
    ref_manifest = read_json(reference_run / "frame_storage/sequence_manifest.json")
    cand_manifest = read_json(candidate_run / "frame_storage/sequence_manifest.json")
    if ref_meta.get("status") != "completed" or ref_manifest.get("status") != "completed":
        raise ValueError("reference run is incomplete")
    if cand_meta.get("status") != "completed" or cand_manifest.get("status") != "completed":
        raise ValueError("candidate run is incomplete")
    ref_records, cand_records = read_records(reference_run), read_records(candidate_run)
    if len(ref_records) != len(cand_records) or not ref_records:
        raise ValueError("reference and candidate frame counts differ or are empty")
    frame_ids = list(range(1, len(ref_records) + 1))
    if [int(row["frame"]) for row in ref_records] != frame_ids:
        raise ValueError("reference frames are not contiguous from frame 1")
    if [int(row["frame"]) for row in cand_records] != frame_ids:
        raise ValueError("candidate frames are not contiguous from frame 1")
    if ref_manifest.get("frames_completed") != len(frame_ids):
        raise ValueError("reference sequence manifest frame count is incorrect")
    if cand_manifest.get("frames_completed") != len(frame_ids):
        raise ValueError("candidate sequence manifest frame count is incorrect")

    ref_base = Path(ref_meta["shared_base"]["path"])
    cand_base = Path(cand_meta["shared_base"]["path"])
    if not ref_base.is_absolute():
        ref_base = reference_run / ref_base
    if not cand_base.is_absolute():
        cand_base = candidate_run / cand_base
    ref_base, cand_base = ref_base.resolve(), cand_base.resolve()
    if sha256_file(ref_base) != sha256_file(cand_base):
        raise ValueError("shared trained frame-0 base differs between streams")
    if ref_meta["shared_base"]["sha256"] != sha256_file(ref_base):
        raise ValueError("reference frame-0 base hash does not match metadata")
    if cand_meta["shared_base"]["sha256"] != sha256_file(cand_base):
        raise ValueError("candidate frame-0 base hash does not match metadata")

    reference_previous = None
    candidate_previous = None
    packet_bytes = birth_bytes = transition_bytes = 0
    full_count = delta_count = 0
    decoded_frames = []
    for ref_record, cand_record in zip(ref_records, cand_records):
        frame = int(ref_record["frame"])
        for key in ("evaluation", "renderer_benchmark", "steps", "complete_training_steps"):
            if ref_record.get(key) != cand_record.get(key):
                raise ValueError(f"frame {frame}: source {key} metrics changed")
        for key in (
            "training_steps_wall_synchronized_s",
            "algorithm_frame_estimate_s",
            "frame_loop_wall_synchronized_s_including_validation_and_io",
        ):
            if ref_record.get("timing", {}).get(key) != cand_record.get("timing", {}).get(key):
                raise ValueError(f"frame {frame}: source timing {key} changed")
        ref_delta, ref_deleted, ref_transition, ref_index = packet_state(
            reference_run, ref_record, reference_previous
        )
        cand_delta, cand_deleted, cand_transition, cand_index = packet_state(
            candidate_run, cand_record, candidate_previous
        )
        if not np.array_equal(cand_delta, ref_delta):
            raise ValueError(f"frame {frame}: float32 motion values changed")
        if not np.array_equal(cand_deleted, ref_deleted):
            raise ValueError(f"frame {frame}: deletion row IDs changed")
        if not np.array_equal(cand_index, ref_index):
            raise ValueError(f"frame {frame}: decoded index differs from REF")
        for key in ("old_base_rows", "previous_birth_rows", "source_rows", "survivor_rows"):
            if int(cand_transition[key]) != int(ref_transition[key]):
                raise ValueError(f"frame {frame}: transition {key} changed")
        ref_birth = reference_run / ref_record["birth_ply"]["path"]
        cand_birth = candidate_run / cand_record["birth_ply"]["path"]
        if sha256_file(ref_birth) != sha256_file(cand_birth):
            raise ValueError(f"frame {frame}: birth PLY differs from REF")
        if int(cand_record["birth_ply"]["rows"]) != int(ref_record["birth_ply"]["rows"]):
            raise ValueError(f"frame {frame}: birth PLY row count changed")
        cand_transition_path = candidate_run / cand_record["transition_metadata"]["path"]
        cand_packet_path = candidate_run / cand_record["packet"]["path"]
        packet_bytes += cand_packet_path.stat().st_size
        birth_bytes += cand_birth.stat().st_size
        transition_bytes += cand_transition_path.stat().st_size
        encoding = int(cand_record["packet"]["index_encoding_code"])
        if encoding == 2:
            delta_count += 1
        elif encoding == 0:
            full_count += 1
        else:
            raise ValueError(f"frame {frame}: unexpected candidate index code {encoding}")
        if ref_delta.shape != cand_delta.shape or ref_index.shape != cand_index.shape:
            raise ValueError(f"frame {frame}: source and candidate state shapes differ")
        decoded_frames.append(frame)
        reference_previous, candidate_previous = ref_index, cand_index

    ref_manifest_path = reference_run / "frame_storage/sequence_manifest.json"
    cand_manifest_path = candidate_run / "frame_storage/sequence_manifest.json"
    manifest_bytes = cand_manifest_path.stat().st_size
    base_bytes = cand_base.stat().st_size
    dynamic_bytes = packet_bytes + birth_bytes + transition_bytes + manifest_bytes
    total_bytes = dynamic_bytes + base_bytes

    ref_summary = read_json(reference_run / "stream_metrics/experiment_summary.json")
    cand_summary = read_json(candidate_run / "stream_metrics/experiment_summary.json")
    # Exact repacking must retain every source quality, training and render observation.
    for key in ("quality_mean", "training_steps_wall_synchronized_s", "cam00_synchronized_renderer_fps"):
        if key in ref_summary and key in cand_summary and ref_summary[key] != cand_summary[key]:
            raise ValueError(f"candidate changed source metric summary field {key}")

    candidate_accounting = cand_summary.get("actual_stream_bytes", {})
    recomputed_accounting = {
        "native_motion_npz_bytes": packet_bytes,
        "incremental_birth_ply_bytes": birth_bytes,
        "required_transition_json_bytes": transition_bytes,
        "required_sequence_manifest_bytes": manifest_bytes,
        "dynamic_stream_bytes": dynamic_bytes,
        "shared_frame0_ply_bytes": base_bytes,
        "total_with_shared_base_bytes": total_bytes,
    }
    for key, value in recomputed_accounting.items():
        if key in candidate_accounting and int(candidate_accounting[key]) != value:
            raise ValueError(f"candidate summary byte count mismatch for {key}")

    return {
        "status": "passed",
        "reference_run": str(reference_run),
        "candidate_run": str(candidate_run),
        "frames_compared": len(decoded_frames),
        "all_motion_delta_arrays_bitwise_equal": True,
        "all_decoded_indices_equal_to_REF": True,
        "all_deletion_ids_and_row_transitions_equal": True,
        "all_incremental_birth_ply_sha256_equal": True,
        "shared_trained_frame0_base_sha256_equal": True,
        "source_quality_training_and_renderer_metrics_unchanged": True,
        "candidate_index_packets": {"full": full_count, "exact_index_delta": delta_count},
        "candidate_actual_bytes": {
            "native_motion_npz": packet_bytes,
            "incremental_birth_ply": birth_bytes,
            "transition_json": transition_bytes,
            "sequence_manifest": manifest_bytes,
            "dynamic_stream": dynamic_bytes,
            "shared_frame0_base_once": base_bytes,
            "total_with_shared_base": total_bytes,
        },
        "reference_actual_total_with_shared_base_bytes": (
            int(ref_summary["actual_stream_bytes"]["total_with_shared_base_bytes"])
            if "actual_stream_bytes" in ref_summary else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_run", type=Path)
    parser.add_argument("candidate_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.reference_run, args.candidate_run)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
