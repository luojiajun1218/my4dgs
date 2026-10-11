#!/usr/bin/env python3
"""Paired on-disk encoder benchmark over real ReCon-GS frame snapshots."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.archive.run_recongs_native_stream import (  # noqa: E402
    decode_topology_index,
    normalize_transition,
    write_compressed_full_packet,
    write_exact_index_delta_packet,
)


def read_records(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "stream_metrics/frames.jsonl").read_text().splitlines()
        if line.strip()
    ]


def load_selected_samples(
    source_run: Path, selected_frames: set[int]
) -> list[dict[str, Any]]:
    records = read_records(source_run)
    if [int(row["frame"]) for row in records] != list(range(1, len(records) + 1)):
        raise ValueError("source frames must be contiguous and begin at frame 1")
    if not selected_frames or min(selected_frames) < 1 or max(selected_frames) > len(records):
        raise ValueError(f"selected frames must be in [1, {len(records)}]")

    samples = []
    previous_index = None
    for record in records[:max(selected_frames)]:
        frame = int(record["frame"])
        packet_path = source_run / record["packet"]["path"]
        transition_path = source_run / record["transition_metadata"]["path"]
        metadata = json.loads(transition_path.read_text())
        with np.load(packet_path, allow_pickle=False) as packet:
            delta = np.asarray(packet["delta"]).copy()
            deleted = np.asarray(packet["deleted_source_indices"]).copy()
            transition = normalize_transition(
                int(metadata["old_base_rows"]),
                int(metadata["previous_birth_rows"]),
                deleted,
            )
            current_index = decode_topology_index(packet, previous_index, transition).copy()
        if frame in selected_frames:
            samples.append(dict(
                frame=frame,
                delta=delta,
                index=current_index,
                transition=transition,
                previous_index=(None if previous_index is None else previous_index.copy()),
            ))
        previous_index = current_index
    return samples


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
    }


def benchmark(source_run: Path, frames: list[int], repeats: int, seed: int) -> dict[str, Any]:
    source_run = source_run.resolve()
    samples = load_selected_samples(source_run, set(frames))
    observations = {"compressed_full": [], "exact_index_delta": []}
    rng = random.Random(seed)
    writer_call_order = 0

    with tempfile.TemporaryDirectory(prefix="recongs_index_delta_io_") as folder:
        output_dir = Path(folder)
        for repeat in range(repeats):
            frame_order = list(samples)
            rng.shuffle(frame_order)
            for sample in frame_order:
                mode_order = ["compressed_full", "exact_index_delta"]
                rng.shuffle(mode_order)
                for mode in mode_order:
                    writer_call_order += 1
                    path = output_dir / f"r{repeat:02d}_f{sample['frame']:03d}_{mode}.npz"
                    started = time.perf_counter()
                    if mode == "compressed_full":
                        metrics = write_compressed_full_packet(
                            path, sample["delta"], sample["index"], sample["transition"]
                        )
                    else:
                        metrics = write_exact_index_delta_packet(
                            path, sample["delta"], sample["index"], sample["transition"],
                            sample["previous_index"],
                        )
                    elapsed = time.perf_counter() - started
                    with np.load(path, allow_pickle=False) as packet:
                        decoded = decode_topology_index(
                            packet, sample["previous_index"], sample["transition"]
                        )
                    if not np.array_equal(decoded, sample["index"]):
                        raise RuntimeError(
                            f"frame {sample['frame']}: {mode} disk packet failed exact recovery"
                        )
                    observations[mode].append({
                        "repeat": repeat + 1,
                        "writer_call_order": writer_call_order,
                        "frame": sample["frame"],
                        "pair_key": f"{repeat + 1}:{sample['frame']}",
                        "bytes": int(path.stat().st_size),
                        "wall_ms": elapsed * 1000.0,
                        "reported_encode_write_ms": (
                            float(metrics["packet_encode_and_write_wall_s"]) * 1000.0
                        ),
                        "prepare_ms": float(metrics.get("index_delta_prepare_wall_s", 0.0)) * 1000.0,
                        "full_candidate_compress_ms": float(metrics.get("full_compress_wall_s", 0.0)) * 1000.0,
                        "exact_delta_candidate_compress_ms": float(metrics.get("delta_compress_wall_s", 0.0)) * 1000.0,
                        "write_ms": float(metrics["packet_write_wall_s"]) * 1000.0,
                        "index_encoding": metrics["index_encoding"],
                        "full_candidate_bytes": metrics.get("full_candidate_bytes"),
                        "exact_delta_candidate_bytes": metrics.get("exact_delta_candidate_bytes"),
                    })

    full_observations = observations["compressed_full"]
    exact_observations = observations["exact_index_delta"]
    full_times = [row["wall_ms"] for row in full_observations]
    exact_times = [row["wall_ms"] for row in exact_observations]
    full_by_key = {row["pair_key"]: row for row in full_observations}
    exact_by_key = {row["pair_key"]: row for row in exact_observations}
    paired_saving = [
        full_by_key[key]["bytes"] - exact_by_key[key]["bytes"]
        for key in full_by_key
    ]
    choice_breakdown = {}
    for choice in ("exact_index_delta", "full"):
        keys = [
            key for key, row in exact_by_key.items()
            if row["index_encoding"] == choice
        ]
        if not keys:
            continue
        full_ms = statistics.fmean(full_by_key[key]["wall_ms"] for key in keys)
        exact_ms = statistics.fmean(exact_by_key[key]["wall_ms"] for key in keys)
        choice_breakdown[choice] = {
            "pairs": len(keys),
            "mean_full_ms": full_ms,
            "mean_exact_writer_ms": exact_ms,
            "mean_added_ms_per_packet": exact_ms - full_ms,
            "mean_bytes_saved_per_packet": statistics.fmean(
                full_by_key[key]["bytes"] - exact_by_key[key]["bytes"]
                for key in keys
            ),
        }
    repeat_summaries = []
    for repeat in range(repeats):
        repeat_full = [row for row in full_observations if row["repeat"] == repeat + 1]
        repeat_exact = [row for row in exact_observations if row["repeat"] == repeat + 1]
        repeat_full_by_frame = {row["frame"]: row for row in repeat_full}
        repeat_exact_by_frame = {row["frame"]: row for row in repeat_exact}
        exact_order = sorted(
            (row for row in observations["exact_index_delta"] if row["repeat"] == repeat + 1),
            key=lambda row: row["writer_call_order"],
        )
        repeat_summaries.append({
            "repeat": repeat + 1,
            "compressed_full_mean_ms": statistics.fmean(row["wall_ms"] for row in repeat_full),
            "exact_writer_mean_ms": statistics.fmean(row["wall_ms"] for row in repeat_exact),
            "added_ms_per_packet": (
                statistics.fmean(row["wall_ms"] for row in repeat_exact)
                - statistics.fmean(row["wall_ms"] for row in repeat_full)
            ),
            "mean_bytes_saved_per_packet": statistics.fmean(
                repeat_full_by_frame[frame]["bytes"] - repeat_exact_by_frame[frame]["bytes"]
                for frame in repeat_full_by_frame
            ),
            "exact_choice_counts": {
                choice: sum(row["index_encoding"] == choice for row in repeat_exact)
                for choice in ("exact_index_delta", "full")
            },
            "exact_writer_mean_phase_ms": {
                key: statistics.fmean(row[key] for row in repeat_exact)
                for key in (
                    "prepare_ms", "full_candidate_compress_ms",
                    "exact_delta_candidate_compress_ms", "write_ms",
                )
            },
            "first_exact_writer_call_order": (
                exact_order[0]["writer_call_order"] if exact_order else None
            ),
            "last_exact_writer_call_order": (
                exact_order[-1]["writer_call_order"] if exact_order else None
            ),
        })
    return {
        "status": "passed",
        "source_run": str(source_run),
        "selected_frames": frames,
        "repeats": repeats,
        "pairs": len(paired_saving),
        "measurement_scope": (
            "CPU packet preparation, ZIP_DEFLATED serialization and fsync-backed atomic disk write; "
            "real REF delta/index/deletion arrays; excludes source read, training, PLY/JSON/manifest IO, "
            "GPU work and directory cleanup. Both packet decoders are checked against the full index."
        ),
        "cache_and_order_note": (
            "REF arrays are loaded once before timing. Each repeat interleaves randomized frame order "
            "and randomized full/exact writer order, writing to one temporary directory with file fsync. "
            "The OS page cache is not cleared, so later passes may be warm; per-repeat results expose drift."
        ),
        "paired_compressed_full": {
            **summarize(full_times),
            "mean_bytes": statistics.fmean(row["bytes"] for row in full_observations),
            "observations": full_observations,
        },
        "paired_exact_index_delta": {
            **summarize(exact_times),
            "mean_bytes": statistics.fmean(row["bytes"] for row in exact_observations),
            "observations": exact_observations,
        },
        "paired_summary": {
            "mean_full_ms": statistics.fmean(full_times),
            "mean_exact_ms": statistics.fmean(exact_times),
            "mean_added_ms_per_packet": statistics.fmean(exact_times) - statistics.fmean(full_times),
            "relative_added_encode_write_percent": (
                (statistics.fmean(exact_times) / statistics.fmean(full_times) - 1.0) * 100.0
            ),
            "mean_bytes_saved_per_packet": statistics.fmean(paired_saving),
        },
        "paired_summary_by_exact_writer_choice": choice_breakdown,
        "per_repeat_summary": repeat_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path, help="completed full-index ReCon-GS run")
    parser.add_argument(
        "--frames", default="1,2,150,151,299",
        help="comma-separated 1-based frame IDs, or 'all' for every source frame",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames.strip().lower() == "all":
        frame_count = len(read_records(args.source_run.resolve()))
        frames = list(range(1, frame_count + 1))
    else:
        frames = sorted(int(value) for value in args.frames.split(",") if value)
    if len(frames) != len(set(frames)) or args.repeats < 1:
        raise ValueError("frames must be unique and repeats must be positive")
    result = benchmark(args.source_run, frames, args.repeats, args.seed)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark report: {args.output}")
    args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
