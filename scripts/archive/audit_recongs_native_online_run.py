#!/usr/bin/env python3
"""Independently decode and account for a completed native ReCon-GS stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.archive.verify_recongs_native_stream import packet_state, read_json, read_records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_series(records: list[dict[str, Any]], getter) -> dict[str, Any]:
    values = [getter(record) for record in records]
    values = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def audit(run_dir: Path, reference_run: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    metrics_dir = run_dir / "stream_metrics"
    meta = read_json(metrics_dir / "run_metadata.json")
    manifest = read_json(run_dir / "frame_storage/sequence_manifest.json")
    summary = read_json(metrics_dir / "experiment_summary.json")
    records = read_records(run_dir)
    frame_count = len(records)
    if meta.get("status") != "completed" or manifest.get("status") != "completed":
        raise ValueError("run metadata and sequence manifest must both be completed")
    if frame_count == 0 or manifest.get("frames_completed") != frame_count:
        raise ValueError("frame records and manifest count disagree or are empty")
    if summary.get("frames_completed") != frame_count:
        raise ValueError("frame record and experiment summary counts disagree")
    if [int(record["frame"]) for record in records] != list(range(1, frame_count + 1)):
        raise ValueError("frame records are not contiguous from frame 1")
    if meta.get("dynamic_frame_range") != [1, frame_count]:
        raise ValueError("metadata dynamic frame range disagrees with records")
    if meta.get("max_dynamic_frames_override") is not None:
        raise ValueError("run used a dynamic-frame-count override")
    if int(summary.get("frames_expected", -1)) != frame_count:
        raise ValueError("experiment summary expected-frame count disagrees")

    previous_index = None
    encodings: Counter[str] = Counter()
    byte_totals: Counter[str] = Counter()
    full_index_npz_bytes = 0
    exact_candidate_npz_bytes = 0
    all_translation_reconstruction_zero = True
    for record in records:
        frame = int(record["frame"])
        if not record.get("complete_training_steps"):
            raise ValueError(f"frame {frame}: training steps are incomplete")
        packet = record["packet"]
        delta, _, _, decoded_index = packet_state(run_dir, record, previous_index)
        code = int(packet["index_encoding_code"])
        encoding = str(packet["index_encoding"])
        if code == 0 and encoding != "full":
            raise ValueError(f"frame {frame}: encoding label/code disagree")
        if code == 2 and encoding != "exact_index_delta":
            raise ValueError(f"frame {frame}: encoding label/code disagree")
        if code not in (0, 2):
            raise ValueError(f"frame {frame}: unsupported index encoding code {code}")
        encodings[encoding] += 1
        if decoded_index.size and (int(decoded_index.min()) < 0 or int(decoded_index.max()) >= len(delta)):
            raise ValueError(f"frame {frame}: decoded index contains an invalid anchor ID")

        candidates = packet.get("serializer_candidates", {})
        full_bytes = int(candidates.get("full_candidate_bytes", 0))
        exact_value = candidates.get("exact_delta_candidate_bytes")
        if full_bytes <= 0:
            raise ValueError(f"frame {frame}: missing full-index candidate size")
        full_index_npz_bytes += full_bytes
        exact_bytes = int(exact_value) if exact_value is not None else None
        if exact_bytes is not None:
            exact_candidate_npz_bytes += exact_bytes
        if code == 2 and (exact_bytes is None or exact_bytes >= full_bytes):
            raise ValueError(f"frame {frame}: exact candidate was not smaller than full")
        selected_bytes = exact_bytes if code == 2 else full_bytes
        packet_path = run_dir / packet["path"]
        actual_packet_bytes = packet_path.stat().st_size
        if actual_packet_bytes != selected_bytes:
            raise ValueError(f"frame {frame}: selected archive size differs from disk")
        byte_totals["motion_npz"] += actual_packet_bytes
        all_translation_reconstruction_zero &= packet.get("max_translation_reconstruction_abs_difference") == 0.0

        birth_path = run_dir / record["birth_ply"]["path"]
        transition_path = run_dir / record["transition_metadata"]["path"]
        if not birth_path.is_file() or not transition_path.is_file():
            raise ValueError(f"frame {frame}: missing birth PLY or transition JSON")
        if birth_path.stat().st_size != int(record["birth_ply"]["bytes"]):
            raise ValueError(f"frame {frame}: birth PLY size disagrees with record")
        byte_totals["birth_ply"] += birth_path.stat().st_size
        byte_totals["transition_json"] += transition_path.stat().st_size
        previous_index = decoded_index

    if not all_translation_reconstruction_zero:
        raise ValueError("one or more frames report nonzero translation reconstruction error")
    manifest_path = run_dir / "frame_storage/sequence_manifest.json"
    byte_totals["manifest"] = manifest_path.stat().st_size
    base_path = Path(meta["shared_base"]["path"])
    if not base_path.is_absolute():
        base_path = run_dir / base_path
    base_bytes = base_path.stat().st_size
    base_hash = sha256_file(base_path)
    if base_hash != meta["shared_base"]["sha256"]:
        raise ValueError("shared frame-0 base SHA-256 disagrees with run metadata")
    byte_totals["shared_base"] = base_bytes
    byte_totals["dynamic_stream"] = (
        byte_totals["motion_npz"]
        + byte_totals["birth_ply"]
        + byte_totals["transition_json"]
        + byte_totals["manifest"]
    )
    byte_totals["actual_total"] = byte_totals["dynamic_stream"] + base_bytes

    full_index_total = (
        full_index_npz_bytes
        + byte_totals["birth_ply"]
        + byte_totals["transition_json"]
        + byte_totals["manifest"]
        + base_bytes
    )
    saved_bytes = full_index_total - byte_totals["actual_total"]
    summary_bytes = summary.get("actual_stream_bytes", {})
    for key, expected in (
        ("native_motion_and_topology_npz_bytes", byte_totals["motion_npz"]),
        ("incremental_birth_ply_bytes", byte_totals["birth_ply"]),
        ("required_transition_json_bytes", byte_totals["transition_json"]),
        ("required_sequence_manifest_bytes", byte_totals["manifest"]),
        ("shared_frame0_ply_bytes", base_bytes),
        ("total_with_shared_base_bytes", byte_totals["actual_total"]),
    ):
        if key in summary_bytes and int(summary_bytes[key]) != expected:
            raise ValueError(f"experiment summary actual byte count mismatch for {key}")

    result: dict[str, Any] = {
        "status": "passed",
        "run_dir": str(run_dir),
        "frames": frame_count,
        "steps_per_frame": sorted({int(record["steps"]) for record in records}),
        "index_encodings": dict(encodings),
        "all_packets_independently_decoded": True,
        "all_translation_reconstruction_max_abs_differences_zero": True,
        "shared_base_sha256": base_hash,
        "actual_bytes": dict(byte_totals),
        "same_run_counterfactual_full_index_total_bytes": full_index_total,
        "same_run_counterfactual_full_index_npz_bytes": full_index_npz_bytes,
        "actual_savings_bytes": saved_bytes,
        "actual_savings_fraction": saved_bytes / full_index_total,
        "timings": {
            "training_steps_s": metric_series(records, lambda r: r.get("timing", {}).get("training_steps_wall_synchronized_s")),
            "algorithm_frame_estimate_s": metric_series(records, lambda r: r.get("timing", {}).get("algorithm_frame_estimate_s")),
            "frame_loop_s": metric_series(records, lambda r: r.get("timing", {}).get("frame_loop_wall_synchronized_s_including_validation_and_io")),
            "packet_encode_and_write_s": metric_series(records, lambda r: r.get("timing", {}).get("packet_encode_and_write_wall_s")),
            "renderer_fps": metric_series(records, lambda r: r.get("renderer_benchmark", {}).get("fps")),
        },
        "quality_test": {
            key: metric_series(records, lambda r, metric=key: r.get("evaluation", {}).get("test", {}).get(metric))
            for key in ("psnr", "ssim", "l1")
        },
    }

    if reference_run is not None:
        reference_run = reference_run.resolve()
        ref_meta = read_json(reference_run / "stream_metrics/run_metadata.json")
        ref_summary = read_json(reference_run / "stream_metrics/experiment_summary.json")
        if meta["shared_base"]["sha256"] != ref_meta["shared_base"]["sha256"]:
            raise ValueError("live run and reference use different frame-0 bases")
        result["reference_run"] = str(reference_run)
        result["reference_comparison"] = {
            "quality_mean": ref_summary.get("quality_mean"),
            "training_steps_mean_s": ref_summary.get("training_steps_wall_synchronized_s", {}).get("mean"),
            "algorithm_frame_estimate_mean_s": ref_summary.get("algorithm_frame_estimate_s", {}).get("mean"),
            "frame_loop_mean_s": ref_summary.get("frame_loop_wall_synchronized_s_including_validation_and_io", {}).get("mean"),
            "renderer_fps_mean": ref_summary.get("cam00_synchronized_renderer_fps", {}).get("mean"),
        }
        own_q = summary.get("quality_mean", {})
        ref_q = result["reference_comparison"].get("quality_mean", {}) or {}
        result["quality_mean_delta_vs_reference"] = {
            key: float(own_q[key]) - float(ref_q[key])
            for key in ("psnr", "ssim", "l1")
            if key in own_q and key in ref_q
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir, args.reference_run)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
