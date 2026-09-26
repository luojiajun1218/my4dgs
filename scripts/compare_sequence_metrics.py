#!/usr/bin/env python3
"""Compare completed per-frame metrics from S²GS sequence runs.

Inputs may be output directories, ``training_metrics.json`` files, or
``train.log`` files. Directories prefer a complete JSON history (including
frame-2 onward snapshot continuations) and fall back to finished tqdm postfix
records when JSON is missing or incomplete.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_FRAME_RE = re.compile(r"\bFrame index\s*=\s*(\d+)")
_FIELD_PATTERNS = {
    "PSNR (Test)": re.compile(rf"\bPSNR\s*\(Test\)\s*=\s*({_NUMBER})"),
    "Size (MB)": re.compile(rf"\bSize\s*\(MB\)\s*=\s*({_NUMBER})"),
    "FPS": re.compile(rf"\bFPS\s*=\s*({_NUMBER})"),
    "Frame time": re.compile(rf"\bFrame time\s*=\s*({_NUMBER})"),
}


@dataclass(frozen=True)
class RunMetrics:
    name: str
    path: Path
    frames: dict[int, dict[str, float]]
    source: str
    expected_frames: int | None
    complete: bool
    expected_start_frame: int = 1


def parse_training_metrics(data: Any) -> dict[int, dict[str, float]]:
    """Read the per-frame rows written by train.py, skipping unusable rows."""
    if not isinstance(data, list):
        raise ValueError("training_metrics.json must contain a list of frame rows")

    rows: dict[int, dict[str, float]] = {}
    for item in data:
        if not isinstance(item, Mapping):
            continue
        try:
            frame_index = int(item["Frame index"])
        except (KeyError, TypeError, ValueError):
            continue
        if frame_index < 1:
            continue

        metrics: dict[str, float] = {}
        for key in _FIELD_PATTERNS:
            value = item.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                metrics[key] = number
        rows[frame_index] = metrics
    return rows


def _line_is_complete(line: str, original_ending: str) -> bool:
    """A postfix is complete only when tqdm closed it and terminated the line."""
    if original_ending not in ("\n", "\r", "\r\n"):
        return False
    stripped = line.rstrip(" \t\r\n")
    # train.py's tqdm postfix is enclosed in brackets. This also excludes a
    # final write that was cut off after all four fields but before closure.
    return stripped.endswith("]")


def parse_train_log(text: str) -> dict[int, dict[str, float]]:
    """Recover the last complete tqdm metrics record for each frame.

    Logs can contain multiple tqdm redraws for one frame. The latest complete
    record wins; an unterminated final line is always ignored.
    """
    clean = _ANSI_ESCAPE.sub("", text)
    rows: dict[int, dict[str, float]] = {}

    # Keep endings so an unterminated final progress update can be rejected.
    for raw_line in clean.splitlines(keepends=True):
        body = raw_line.rstrip("\r\n")
        ending = raw_line[len(body):]
        if not _line_is_complete(body, ending):
            continue
        frame_match = _FRAME_RE.search(body)
        if not frame_match:
            continue

        metrics: dict[str, float] = {}
        for key, pattern in _FIELD_PATTERNS.items():
            match = pattern.search(body, frame_match.end())
            if not match:
                break
            value = float(match.group(1))
            if not math.isfinite(value):
                break
            metrics[key] = value
        else:
            frame_index = int(frame_match.group(1))
            if frame_index > 0:
                rows[frame_index] = metrics
    return rows


def _expected_frame_count(root: Path) -> int | None:
    # The log reports the number of frames actually loaded. ``max_frames`` in
    # cfg_args is only an upper bound when the source sequence is shorter.
    train_log = root / "train.log"
    if train_log.is_file():
        match = re.search(
            r"training\(\): running with\s+(\d+)\s+frames",
            train_log.read_text(errors="replace"),
        )
        if match:
            return int(match.group(1))

    cfg_args = root / "cfg_args"
    if cfg_args.is_file():
        match = re.search(r"\bmax_frames\s*=\s*(\d+)", cfg_args.read_text(errors="replace"))
        if match:
            return int(match.group(1))
    return None


def _expected_frame_range(root: Path) -> tuple[int, int] | None:
    """Return the frame range a run was expected to write, when knowable.

    A frame-one snapshot continuation intentionally writes frames 2..N, so a
    missing frame 1 is not evidence that its metrics JSON is incomplete.
    """
    end = _expected_frame_count(root)
    if end is None:
        return None

    start = 1
    cfg_args = root / "cfg_args"
    cfg_text = cfg_args.read_text(errors="replace") if cfg_args.is_file() else ""
    resume_match = re.search(
        r"\bresume_frame1_snapshot\s*=\s*([^,)]*)", cfg_text)
    resume_value = resume_match.group(1).strip() if resume_match else "None"
    if resume_value not in {"None", "''", '""'}:
        start = 2
    else:
        train_log = root / "train.log"
        if train_log.is_file() and re.search(
                r"Resuming at frame\s+2\s+from frame-one snapshot",
                train_log.read_text(errors="replace")):
            start = 2

    if re.search(r"\bstop_after_frame1_snapshot\s*=\s*True\b", cfg_text):
        return (1, 1)
    return (start, end)


def _has_expected_frames(
    frames: Mapping[int, Any], expected_range: tuple[int, int] | None
) -> bool:
    if not frames:
        return False
    if expected_range is None:
        start, end = min(frames), max(frames)
    else:
        start, end = expected_range
    required_fields = set(_FIELD_PATTERNS)
    return all(
        index in frames and required_fields.issubset(frames[index])
        for index in range(start, end + 1)
    )


def _json_rows(path: Path) -> dict[int, dict[str, float]] | None:
    try:
        return parse_training_metrics(json.loads(path.read_text()))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def load_run(path_value: str | Path, name: str | None = None) -> RunMetrics:
    """Load a complete JSON history or recover complete rows from train.log."""
    supplied = Path(path_value).expanduser()
    if supplied.is_dir():
        root = supplied
        requested_kind = "directory"
    elif supplied.is_file():
        root = supplied.parent
        requested_kind = supplied.name
    else:
        raise FileNotFoundError(f"No such run input: {supplied}")

    expected_range = _expected_frame_range(root)
    expected = expected_range[1] if expected_range is not None else None
    expected_start = expected_range[0] if expected_range is not None else 1
    json_path = supplied if requested_kind == "training_metrics.json" else root / "training_metrics.json"
    log_path = supplied if requested_kind == "train.log" else root / "train.log"

    json_rows = _json_rows(json_path) if json_path.is_file() else None
    if json_rows:
        # Validate against this run's intended coverage, not an assumption
        # that every run contains frame 1. Resumed runs intentionally begin at
        # frame 2 and their JSON values retain more precision than tqdm logs.
        json_complete = _has_expected_frames(json_rows, expected_range)
        if json_complete:
            return RunMetrics(
                name=name or root.name,
                path=root,
                frames=json_rows,
                source=str(json_path),
                expected_frames=expected,
                complete=True,
                expected_start_frame=expected_start,
            )

    if log_path.is_file():
        log_rows = parse_train_log(log_path.read_text(errors="replace"))
        if not log_rows:
            raise ValueError(f"No complete per-frame tqdm records found in {log_path}")
        complete = _has_expected_frames(log_rows, expected_range)
        return RunMetrics(
            name=name or root.name,
            path=root,
            frames=log_rows,
            source=str(log_path),
            expected_frames=expected,
            complete=complete,
            expected_start_frame=expected_start,
        )

    if json_rows:
        # A valid but shorter JSON file may be all that was preserved.
        return RunMetrics(
            name=name or root.name,
            path=root,
            frames=json_rows,
            source=str(json_path),
            expected_frames=expected,
            complete=False,
            expected_start_frame=expected_start,
        )

    raise FileNotFoundError(
        f"Could not find usable training_metrics.json or train.log under {root}"
    )


def common_frame_indices(candidate: RunMetrics, baseline: RunMetrics) -> list[int]:
    """Return sorted frame IDs available in both runs."""
    return sorted(candidate.frames.keys() & baseline.frames.keys())


def temporal_windows(common_indices: Iterable[int]) -> list[tuple[str, list[int]]]:
    """Split aligned dynamic frames into early, middle, and late windows."""
    indices = sorted(index for index in common_indices if index >= 2)
    if not indices:
        return []
    last = indices[-1]
    boundaries = ((2, min(30, last), "2-30"),
                  (31, min(120, last), "31-120"),
                  (121, last, f"121-{last}"))
    windows = []
    for start, end, label in boundaries:
        selected = [index for index in indices if start <= index <= end]
        if selected:
            windows.append((label, selected))
    return windows


def _mean(run: RunMetrics, indices: Iterable[int], key: str) -> float | None:
    values = [run.frames[index][key] for index in indices if key in run.frames[index]]
    return sum(values) / len(values) if values else None


def _format(value: float | None, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _format_frame_ranges(indices: Iterable[int]) -> str:
    """Compactly format sorted frame IDs, e.g. 1-3,5,8-10."""
    values = sorted(set(indices))
    if not values:
        return "none"
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous))
        start = previous = value
    ranges.append((start, previous))
    return ",".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )


def _coverage_description(run: RunMetrics) -> str:
    actual = _format_frame_ranges(run.frames)
    if run.expected_frames is None:
        return f"coverage {actual}; expected range unknown"

    expected_start = run.expected_start_frame
    expected_end = run.expected_frames
    expected_indices = range(expected_start, expected_end + 1)
    missing = _format_frame_ranges(
        index for index in expected_indices if index not in run.frames
    )
    expected = _format_frame_ranges(expected_indices)
    return (
        f"coverage {actual}; expected {expected}; "
        f"missing expected frames {missing}"
    )


def _row(
    label: str,
    candidate: RunMetrics,
    baseline: RunMetrics,
    indices: list[int],
) -> str:
    definitions = (
        ("PSNR, held-out view (dB)", "PSNR (Test)", 3),
        ("Estimated storage (MB/frame)", "Size (MB)", 4),
        ("Held-out-view render FPS", "FPS", 2),
        ("Reported frame time (s)", "Frame time", 2),
    )
    lines = [f"{label} (n={len(indices)})"]
    for display, key, digits in definitions:
        candidate_value = _mean(candidate, indices, key)
        baseline_value = _mean(baseline, indices, key)
        delta = (
            candidate_value - baseline_value
            if candidate_value is not None and baseline_value is not None
            else None
        )
        lines.append(
            f"  {display:<32} {_format(candidate_value, digits):>10}"
            f"  {baseline.name}: {_format(baseline_value, digits):>10}"
            f"  Δ: {_format(delta, digits):>10}"
        )
    return "\n".join(lines)


def render_comparison(candidate: RunMetrics, baseline: RunMetrics) -> str:
    common = common_frame_indices(candidate, baseline)
    if not common:
        raise ValueError("The runs have no common frame indices")

    last_common = common[-1]
    candidate_status = "complete" if candidate.complete else f"partial through {max(candidate.frames)}"
    baseline_status = "complete" if baseline.complete else f"partial through {max(baseline.frames)}"
    lines = [
        f"Candidate: {candidate.name} ({candidate_status}; {len(candidate.frames)} metric frames)",
        f"  {_coverage_description(candidate)}",
        f"  source: {candidate.source}",
        f"Baseline:  {baseline.name} ({baseline_status}; {len(baseline.frames)} metric frames)",
        f"  {_coverage_description(baseline)}",
        f"  source: {baseline.source}",
        f"Aligned frame IDs: {_format_frame_ranges(common)} ({len(common)} common frames)",
        "",
        "Metric labels: storage is the repository's ideal estimated MB/frame, not encoded bytes;",
        "FPS is the synchronized held-out test-camera render metric, not the all-view fps.py benchmark.",
        "Frame time is train.py's reported timer; with timed=False, per-step CUDA sync is not forced.",
        "'Active gates' is omitted because v6 counts UPDATE regions there while S²GS counts active",
        "per-Gaussian gates.",
        "",
    ]

    if 1 in candidate.frames and 1 in baseline.frames:
        lines.append(_row("Frame 1 (static initialization)", candidate, baseline, [1]))
        lines.append("")

    dynamic = [index for index in common if index >= 2]
    if dynamic:
        lines.append(_row(f"Dynamic frames 2-{last_common}", candidate, baseline, dynamic))
        for label, indices in temporal_windows(common):
            lines.append("")
            lines.append(_row(f"Temporal window {label}", candidate, baseline, indices))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="run directory, training_metrics.json, or train.log")
    parser.add_argument("baseline", help="run directory, training_metrics.json, or train.log")
    parser.add_argument("--candidate-name", help="display label for the candidate")
    parser.add_argument("--baseline-name", help="display label for the baseline")
    args = parser.parse_args(argv)

    try:
        candidate = load_run(args.candidate, args.candidate_name)
        baseline = load_run(args.baseline, args.baseline_name)
        print(render_comparison(candidate, baseline))
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
