#!/usr/bin/env python3
"""Run the legacy ReCon-GS baseline and export its native per-frame motion stream.

The runner loads the external ReCon-GS checkout, reuses its trained frame-0
checkpoint, and instruments only serialization and measurement. It does not
change the training forward pass, losses, optimizer, or sampling.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path
from statistics import mean
from types import MethodType
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RECON_GS_ROOT = Path("/home/gamer-02/Documents/DeepLearning/ReCon-GS")
RECON_GS_ROOT = Path(os.environ.get("RECON_GS_ROOT", ORIGINAL_RECON_GS_ROOT)).resolve()
BASE_RUN = Path(os.environ.get(
    "RECON_GS_BASE_RUN",
    str(ORIGINAL_RECON_GS_ROOT / "output/flame_steak_fullres/20260926T225223_dc5baf"),
)).resolve()
SHARED_BASE_PLY = BASE_RUN / "point_cloud/gaussians-0-15000.ply"
CONFIG_PATH = PROJECT_ROOT / "config/recongs_flame_steak_native_stream_20261002.yaml"


def normalize_transition(
    old_base_rows: int,
    previous_birth_rows: int,
    deleted_source_indices: np.ndarray,
) -> dict[str, Any]:
    """Describe exact append-then-prune row order used at a frame boundary."""
    if old_base_rows < 0 or previous_birth_rows < 0:
        raise ValueError("row counts must be non-negative")
    source_rows = old_base_rows + previous_birth_rows
    deleted = np.asarray(deleted_source_indices, dtype=np.int64).reshape(-1)
    if np.any(deleted < 0) or np.any(deleted >= source_rows):
        raise ValueError("deleted row index is outside the concatenated source rows")
    if np.unique(deleted).size != deleted.size:
        raise ValueError("deleted row indices must be unique")
    deleted = np.sort(deleted)
    keep = np.ones(source_rows, dtype=bool)
    keep[deleted] = False
    survivors = np.flatnonzero(keep).astype(np.int64, copy=False)
    return {
        "old_base_rows": old_base_rows,
        "previous_birth_rows": previous_birth_rows,
        "source_rows": source_rows,
        "deleted_source_indices": deleted,
        "survivor_rows": int(survivors.size),
        "row_order": "previous base rows, then previous frame birth PLY rows; prune preserves source order",
    }


def resolve_native_index_encoding(value: Any = None, env_value: str | None = None) -> str:
    """Select the packet writer mode while keeping old configs on legacy output."""
    selected = env_value if env_value is not None else value
    if selected is None:
        selected = "legacy"
    selected = str(selected)
    if selected not in {"legacy", "compressed_full", "exact_index_delta"}:
        raise ValueError(
            "native_index_encoding must be legacy, compressed_full, or exact_index_delta"
        )
    return selected


def decode_native_field(delta: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Reconstruct per-Gaussian translation and quaternion from native arrays."""
    delta = np.asarray(delta)
    index = np.asarray(index)
    if delta.ndim != 2 or delta.shape[1] != 7:
        raise ValueError(f"delta must have shape [anchors, 7], got {delta.shape}")
    if index.ndim != 2:
        raise ValueError(f"index must have shape [levels, rows], got {index.shape}")
    if index.size and (index.min() < 0 or index.max() >= len(delta)):
        raise ValueError("native index points outside delta array")
    return delta[index].sum(axis=0)


def verify_native_application(
    delta: np.ndarray,
    index: np.ndarray,
    applied_xyz: np.ndarray,
    applied_rotation: np.ndarray,
) -> float:
    """Check that a serialized packet recreates the exact stage-1 application."""
    field = decode_native_field(delta, index)
    rotation = field[:, 3:]
    norms = np.linalg.norm(rotation, axis=-1, keepdims=True)
    rotation = rotation / np.maximum(norms, 1e-12)
    if not np.allclose(field[:, :3], applied_xyz, rtol=1e-5, atol=1e-6):
        raise RuntimeError("serialized native delta/index do not reproduce applied translation")
    if not np.allclose(rotation, applied_rotation, rtol=1e-5, atol=1e-6):
        raise RuntimeError("serialized native delta/index do not reproduce applied rotation")
    return float(np.max(np.abs(field[:, :3] - applied_xyz))) if field.size else 0.0


def write_native_packet(
    path: Path,
    delta: np.ndarray,
    index: np.ndarray,
    transition: dict[str, Any],
    previous_index: np.ndarray | None = None,
) -> int:
    """Store native motion and either full or inherited topology without compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload: dict[str, np.ndarray] = {
        "delta": np.asarray(delta),
        "deleted_source_indices": transition["deleted_source_indices"],
    }
    if previous_index is None:
        payload["index"] = np.asarray(index)
        payload["index_encoding"] = np.asarray(0, dtype=np.uint8)
    else:
        previous_index = np.asarray(previous_index)
        old_base_rows = int(transition["old_base_rows"])
        if previous_index.ndim != 2 or previous_index.shape[1] != old_base_rows:
            raise ValueError("previous index rows do not match the old base row count")
        keep = np.ones(int(transition["source_rows"]), dtype=bool)
        keep[transition["deleted_source_indices"]] = False
        survivor_source_rows = np.flatnonzero(keep)
        old_survivor_source_rows = survivor_source_rows[
            survivor_source_rows < old_base_rows
        ]
        inherited = previous_index[:, old_survivor_source_rows]
        birth_assignments = np.asarray(index)[:, inherited.shape[1]:]
        reconstructed = np.concatenate((inherited, birth_assignments), axis=1)
        if reconstructed.shape != np.asarray(index).shape or not np.array_equal(
            reconstructed, index
        ):
            raise ValueError("inherited survivor topology does not reproduce the current index")
        payload["index_birth_values"] = birth_assignments
        payload["index_encoding"] = np.asarray(1, dtype=np.uint8)
    with partial.open("wb") as stream:
        np.savez(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    return path.stat().st_size


def checked_int32(values: np.ndarray, name: str) -> np.ndarray:
    """Cast integer row/anchor IDs without changing their represented values."""
    values = np.asarray(values)
    if values.dtype.kind not in "iu":
        raise TypeError(f"{name} must use an integer dtype, got {values.dtype}")
    if values.size:
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < 0 or maximum > np.iinfo(np.int32).max:
            raise ValueError(f"{name} cannot be represented as non-negative int32")
    return values.astype(np.int32, copy=False)


def compressed_npz_bytes(payload: dict[str, np.ndarray]) -> tuple[bytes, float]:
    """Serialize one packet with the existing ZIP_DEFLATED NumPy container."""
    started = time.perf_counter()
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue(), time.perf_counter() - started


def write_packet_bytes(path: Path, serialized: bytes) -> int:
    """Atomically persist already-serialized bytes and return the on-disk size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    return path.stat().st_size


def write_compressed_full_packet(
    path: Path,
    delta: np.ndarray,
    index: np.ndarray,
    transition: dict[str, Any],
) -> dict[str, Any]:
    """Write a full index packet under ZIP_DEFLATED with lossless int32 IDs."""
    started = time.perf_counter()
    delta = np.asarray(delta)
    if delta.dtype != np.float32:
        raise TypeError(f"motion delta must remain float32, got {delta.dtype}")
    if delta.ndim != 2 or delta.shape[1] != 7:
        raise ValueError("motion delta must have shape [anchors, 7]")
    payload = {
        "delta": delta,
        "index": checked_int32(index, "index"),
        "deleted_source_indices": checked_int32(
            transition["deleted_source_indices"], "deleted_source_indices"
        ),
        "index_encoding": np.asarray(0, dtype=np.uint8),
    }
    _validate_packet_arrays(payload["delta"], payload["index"], transition)
    encoded, encode_seconds = compressed_npz_bytes(payload)
    write_started = time.perf_counter()
    file_bytes = write_packet_bytes(path, encoded)
    write_seconds = time.perf_counter() - write_started
    return {
        "bytes": file_bytes,
        "index_encoding": "full",
        "index_encoding_code": 0,
        "full_candidate_bytes": file_bytes,
        "exact_delta_candidate_bytes": None,
        "index_delta_prepare_wall_s": 0.0,
        "full_compress_wall_s": encode_seconds,
        "delta_compress_wall_s": 0.0,
        "packet_write_wall_s": write_seconds,
        # Include validation, casts, payload construction, compression and disk IO.
        "packet_encode_and_write_wall_s": time.perf_counter() - started,
        "source_index_dtype": str(np.asarray(index).dtype),
        "stored_array_dtypes": {key: str(value.dtype) for key, value in payload.items()},
        "compression": "np.savez_compressed ZIP_DEFLATED",
    }


def write_exact_index_delta_packet(
    path: Path,
    delta: np.ndarray,
    index: np.ndarray,
    transition: dict[str, Any],
    previous_index: np.ndarray | None,
) -> dict[str, Any]:
    """Write the smaller lossless full or delta packet using actual ZIP bytes.

    The delta carries unchanged assignments for surviving old-base rows from the
    previous decoded full index. It stores exact changed cells and all rows for
    which the previous packet has no index assignment. Training arrays are not
    mutated; the caller continues using ``index`` for its in-memory training
    state and verification.
    """
    started = time.perf_counter()
    delta = np.asarray(delta)
    if delta.dtype != np.float32:
        raise TypeError(f"motion delta must remain float32, got {delta.dtype}")
    if delta.ndim != 2 or delta.shape[1] != 7:
        raise ValueError("motion delta must have shape [anchors, 7]")
    full_index = checked_int32(index, "index")
    deleted = checked_int32(
        transition["deleted_source_indices"], "deleted_source_indices"
    )
    _validate_packet_arrays(delta, full_index, transition)
    if deleted.shape != np.asarray(transition["deleted_source_indices"]).reshape(-1).shape:
        raise ValueError("deleted source IDs must be a one-dimensional vector")
    full_payload = {
        "delta": delta,
        "index": full_index,
        "deleted_source_indices": deleted,
        "index_encoding": np.asarray(0, dtype=np.uint8),
    }

    if previous_index is None:
        full_serialized, full_compress_seconds = compressed_npz_bytes(full_payload)
        write_started = time.perf_counter()
        file_bytes = write_packet_bytes(path, full_serialized)
        write_seconds = time.perf_counter() - write_started
        return {
            "bytes": file_bytes,
            "index_encoding": "full",
            "index_encoding_code": 0,
            "full_candidate_bytes": len(full_serialized),
            "exact_delta_candidate_bytes": None,
            "index_delta_prepare_wall_s": 0.0,
            "full_compress_wall_s": full_compress_seconds,
            "delta_compress_wall_s": 0.0,
            "packet_write_wall_s": write_seconds,
            "packet_encode_and_write_wall_s": time.perf_counter() - started,
            "source_index_dtype": str(np.asarray(index).dtype),
            "stored_array_dtypes": {
                key: str(value.dtype) for key, value in full_payload.items()
            },
            "compression": "np.savez_compressed ZIP_DEFLATED",
        }

    prepare_started = time.perf_counter()
    previous = checked_int32(previous_index, "previous_index")
    old_base_rows = int(transition["old_base_rows"])
    source_rows = int(transition["source_rows"])
    if previous.ndim != 2 or previous.shape[1] != old_base_rows:
        raise ValueError("previous index rows do not match the old base row count")
    if full_index.ndim != 2 or full_index.shape[0] != previous.shape[0]:
        raise ValueError("current and previous index layer counts differ")
    if deleted.size and int(deleted.max()) >= source_rows:
        raise ValueError("deleted source row is outside the transition source")
    keep = np.ones(source_rows, dtype=bool)
    keep[deleted] = False
    survivor_source_rows = np.flatnonzero(keep)
    old_survivor_rows = survivor_source_rows[survivor_source_rows < old_base_rows]
    if old_survivor_rows.size > full_index.shape[1]:
        raise ValueError("current index omits surviving old-base rows")
    if full_index.shape[1] != int(transition["survivor_rows"]):
        raise ValueError("current index rows do not match transition survivors")

    inherited = previous[:, old_survivor_rows]
    current_old_values = full_index[:, :old_survivor_rows.size]
    mismatch = inherited != current_old_values
    changed_cells_64 = np.flatnonzero(mismatch.reshape(-1))
    if changed_cells_64.size and int(changed_cells_64.max()) > np.iinfo(np.int32).max:
        raise ValueError("changed cell ID cannot be represented as int32")
    changed_cells = changed_cells_64.astype(np.int32, copy=False)
    changed_values = current_old_values.reshape(-1)[changed_cells]
    fresh_values = full_index[:, old_survivor_rows.size:]
    expected_fresh_rows = int(survivor_source_rows.size - old_survivor_rows.size)
    if fresh_values.shape[1] != expected_fresh_rows:
        raise ValueError("untracked survivor index rows do not match transition row order")
    delta_prepare_seconds = time.perf_counter() - prepare_started
    delta_payload = {
        "delta": delta,
        "deleted_source_indices": deleted,
        "index_changed_cells": changed_cells,
        "index_changed_values": changed_values,
        "index_birth_values": fresh_values,
        "index_encoding": np.asarray(2, dtype=np.uint8),
    }

    # Compare the actual complete NPZ candidates. This duplicates compression
    # of the shared motion array, but avoids approximating archive byte costs;
    # a lower-cost index-only probe was measured slower on the full REF stream.
    full_serialized, full_compress_seconds = compressed_npz_bytes(full_payload)
    delta_serialized, delta_compress_seconds = compressed_npz_bytes(delta_payload)
    full_candidate_bytes = len(full_serialized)
    exact_delta_candidate_bytes = len(delta_serialized)
    if exact_delta_candidate_bytes < full_candidate_bytes:
        selected, encoding, encoding_code = delta_serialized, "exact_index_delta", 2
        del full_serialized
    else:
        selected, encoding, encoding_code = full_serialized, "full", 0
        del delta_serialized
    write_started = time.perf_counter()
    file_bytes = write_packet_bytes(path, selected)
    write_seconds = time.perf_counter() - write_started
    return {
        "bytes": file_bytes,
        "index_encoding": encoding,
        "index_encoding_code": encoding_code,
        "full_candidate_bytes": int(full_candidate_bytes),
        "exact_delta_candidate_bytes": int(exact_delta_candidate_bytes),
        "index_delta_prepare_wall_s": delta_prepare_seconds,
        "full_compress_wall_s": full_compress_seconds,
        "delta_compress_wall_s": delta_compress_seconds,
        "packet_write_wall_s": write_seconds,
        "packet_encode_and_write_wall_s": time.perf_counter() - started,
        "source_index_dtype": str(np.asarray(index).dtype),
        "stored_array_dtypes": {
            key: str(value.dtype)
            for key, value in (delta_payload if encoding_code == 2 else full_payload).items()
        },
        "changed_index_cells": int(changed_cells.size),
        "fresh_index_rows": int(fresh_values.shape[1]),
        "compression": (
            "np.savez_compressed ZIP_DEFLATED; compare both complete candidate archives "
            "and write the smaller actual archive"
        ),
    }


def decode_topology_index(
    packet: Any,
    previous_index: np.ndarray | None,
    transition: dict[str, Any],
) -> np.ndarray:
    """Rebuild inherited survivor assignments from a prior full index packet."""
    packet_keys = set(packet.files) if hasattr(packet, "files") else set(packet)
    encoding = (
        int(np.asarray(packet["index_encoding"]).item())
        if "index_encoding" in packet_keys else 0
    )
    if "index" in packet_keys:
        if encoding != 0:
            raise ValueError("full index packet has a conflicting index encoding")
        index = np.asarray(packet["index"])
        if index.ndim != 2:
            raise ValueError("full topology index must have shape [levels, rows]")
        if index.dtype.kind not in "iu":
            raise TypeError("full topology index must use an integer dtype")
        if index.size and index.min() < 0:
            raise ValueError("full topology index must contain non-negative IDs")
        expected_rows = transition.get("survivor_rows")
        if expected_rows is not None and index.shape[1] != int(expected_rows):
            raise ValueError("full topology row count does not match transition survivors")
        if "delta" in packet:
            delta = np.asarray(packet["delta"])
            if delta.ndim != 2 or delta.shape[1] != 7:
                raise ValueError("motion delta must have shape [anchors, 7]")
            if index.size and (index.min() < 0 or index.max() >= len(delta)):
                raise ValueError("full topology index points outside delta")
        return index
    if "index_birth_values" not in packet_keys or "index_encoding" not in packet_keys:
        raise ValueError("packet has neither a full index nor an inherited-index payload")
    if encoding not in (1, 2):
        raise ValueError(f"unsupported topology index encoding {encoding}")
    if previous_index is None:
        raise ValueError("inherited topology requires the preceding frame index")
    if "deleted_source_indices" in packet_keys:
        deleted = np.asarray(packet["deleted_source_indices"])
    elif "deleted_source_indices" in transition:
        deleted = np.asarray(transition["deleted_source_indices"])
    else:
        raise ValueError("inherited topology packet lacks its deletion row IDs")
    normalized_transition = normalize_transition(
        int(transition["old_base_rows"]),
        int(transition["previous_birth_rows"]),
        deleted,
    )
    for key in ("source_rows", "survivor_rows"):
        if key in transition and int(transition[key]) != int(normalized_transition[key]):
            raise ValueError(f"transition {key} disagrees with packet deletion IDs")
    transition = normalized_transition
    previous_index = np.asarray(previous_index)
    old_base_rows = int(transition["old_base_rows"])
    if previous_index.ndim != 2 or previous_index.shape[1] != old_base_rows:
        raise ValueError("previous index rows do not match the old base row count")
    keep = np.ones(int(transition["source_rows"]), dtype=bool)
    keep[transition["deleted_source_indices"]] = False
    survivor_source_rows = np.flatnonzero(keep)
    old_survivor_source_rows = survivor_source_rows[
        survivor_source_rows < old_base_rows
    ]
    inherited = previous_index[:, old_survivor_source_rows]
    if encoding == 2:
        if not {"index_changed_cells", "index_changed_values"}.issubset(packet_keys):
            raise ValueError("exact index-delta packet is missing changed-cell arrays")
        cells = np.asarray(packet["index_changed_cells"])
        values = np.asarray(packet["index_changed_values"])
        if cells.dtype != np.int32 or values.dtype != np.int32:
            raise ValueError("exact index-delta cells and values must be int32")
        if cells.ndim != 1 or values.shape != cells.shape:
            raise ValueError("exact index-delta cells and values must be matching vectors")
        if cells.size > 1 and np.any(cells[1:] <= cells[:-1]):
            raise ValueError("exact index-delta cells must be sorted and unique")
        flat = inherited.reshape(-1).copy()
        if cells.size and (cells.min() < 0 or cells.max() >= flat.size):
            raise ValueError("exact index-delta cell ID is outside inherited rows")
        flat[cells] = values
        inherited = flat.reshape(inherited.shape)
    birth_assignments = np.asarray(packet["index_birth_values"])
    if birth_assignments.ndim != 2 or birth_assignments.shape[0] != previous_index.shape[0]:
        raise ValueError("fresh topology rows have an incompatible layer count")
    if birth_assignments.dtype.kind not in "iu":
        raise TypeError("fresh topology assignments must use an integer dtype")
    if encoding == 2 and birth_assignments.dtype != np.int32:
        raise ValueError("exact index-delta birth assignments must be int32")
    index = np.concatenate((inherited, birth_assignments), axis=1)
    if index.shape[1] != int(transition["survivor_rows"]):
        raise ValueError("decoded topology row count does not match transition survivors")
    if index.size and (index.min() < 0 or index.max() > np.iinfo(np.int32).max):
        raise ValueError("decoded topology index is outside the non-negative int32 range")
    if "delta" in packet:
        delta = np.asarray(packet["delta"])
        if delta.ndim != 2 or delta.shape[1] != 7:
            raise ValueError("motion delta must have shape [anchors, 7]")
        if index.size and index.max() >= len(delta):
            raise ValueError("decoded topology index points outside delta")
    return index


def _validate_packet_arrays(
    delta: np.ndarray, index: np.ndarray, transition: dict[str, Any]
) -> None:
    """Reject malformed snapshots before writing bytes or selecting a codec."""
    if index.ndim != 2:
        raise ValueError("topology index must have shape [levels, rows]")
    expected_rows = int(transition["survivor_rows"])
    if index.shape[1] != expected_rows:
        raise ValueError("topology row count does not match transition survivors")
    if index.size and int(index.max()) >= len(delta):
        raise ValueError("topology index points outside delta")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == "baseline_source_snapshot.json":
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        if path.is_symlink():
            digest.update(b"L\0" + os.readlink(path).encode("utf-8") + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + bytes.fromhex(sha256_file(path)))
        elif path.is_dir():
            digest.update(b"D\0")
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"could not find PLY vertex count in {path}")


def scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def synchronized_fps(call, torch, device, warmup: int = 5, repeats: int = 30) -> dict[str, Any]:
    """Use the same repeated, synchronized timing method as the playback comparison."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    return {
        "warmup_renders": warmup,
        "timed_renders": repeats,
        "seconds": seconds,
        "fps": repeats / seconds,
        "ms_per_render": seconds * 1000 / repeats,
        "scope": "module forward only, no clamp; cam00 test view; synchronized batch wall time",
    }


def run(config_path: Path = CONFIG_PATH) -> Path:
    config_path = Path(config_path).resolve()
    if not RECON_GS_ROOT.is_dir():
        raise FileNotFoundError(RECON_GS_ROOT)
    if not SHARED_BASE_PLY.is_file():
        raise FileNotFoundError(SHARED_BASE_PLY)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    source_manifest_path = RECON_GS_ROOT / "baseline_source_snapshot.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            "ReCon-GS must be run from a frozen source snapshot with "
            f"{source_manifest_path.name}; set RECON_GS_ROOT to that snapshot"
        )
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_tree_sha256(RECON_GS_ROOT) != source_manifest.get(
        "source_tree_sha256_excluding_this_manifest"
    ):
        raise RuntimeError("frozen ReCon-GS source tree no longer matches its manifest")
    source_archive_path = Path(os.environ.get(
        "RECON_GS_SOURCE_ARCHIVE", "/tmp/recongs_native_source_snapshot_20261002.tar.gz"
    ))
    source_archive_record_path = Path(os.environ.get(
        "RECON_GS_SOURCE_ARCHIVE_RECORD",
        str(source_archive_path) + ".json",
    ))
    if not source_archive_path.is_file() or not source_archive_record_path.is_file():
        raise FileNotFoundError("frozen ReCon-GS source archive and its hash record are required")
    source_archive_record = json.loads(source_archive_record_path.read_text())
    if source_archive_record.get("snapshot_tree_sha256") != source_manifest.get(
        "source_tree_sha256_excluding_this_manifest"
    ):
        raise RuntimeError("source archive tree hash and active snapshot manifest differ")
    if sha256_file(source_archive_path) != source_archive_record["archive_sha256"]:
        raise RuntimeError("frozen ReCon-GS source archive hash does not match its record")

    sys.path.insert(0, str(RECON_GS_ROOT))
    os.chdir(RECON_GS_ROOT)

    import torch
    from omegaconf import OmegaConf
    from libgs.metric import lpips, psnr, ssim
    from libgs.pipeline.loader import load_pipeline
    from libgs.utils.config import to_structured
    from torch.nn.functional import l1_loss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; baseline training was not started")

    # The upstream pipeline opens a local GUI socket at import time even when
    # no GUI is used. Keep this headless baseline inside the restricted local
    # runner by substituting a no-op socket while the pipeline modules import.
    import socket

    real_socket_factory = socket.socket

    class DisabledGuiSocket:
        def bind(self, *args, **kwargs):
            return None

        def listen(self, *args, **kwargs):
            return None

        def settimeout(self, *args, **kwargs):
            return None

        def accept(self, *args, **kwargs):
            raise OSError("ReCon-GS GUI socket is disabled for this headless experiment")

    try:
        socket.socket = lambda *args, **kwargs: DisabledGuiSocket()
        Config, Pipeline = load_pipeline("ReCon-GS")
    finally:
        socket.socket = real_socket_factory
    from libgs.renderer import network_gui
    network_gui.init = lambda *args, **kwargs: None
    network_gui.try_connect = lambda *args, **kwargs: None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            module=r"libgs\.utils\.config",
            message=r"(?:Try|Failed) to cast '.*' to typing\.Optional\[(?:pathlib\.Path|str)\]$",
        )
        raw_config = OmegaConf.load(config_path)
        runner_options = raw_config.pop("runner_options", None)
        if runner_options is None:
            runner_options = OmegaConf.create({})
        config, remaining = to_structured(raw_config, Config)
    if remaining:
        raise ValueError(f"unrecognized config fields: {OmegaConf.to_container(remaining)}")
    if config.ckpt_path is not None:
        config.ckpt_path = Path(config.ckpt_path)

    reuse_topology_env = os.environ.get("RECON_GS_REUSE_TOPOLOGY")
    if reuse_topology_env is None:
        reuse_topology_enabled = bool(
            OmegaConf.select(
                runner_options,
                "reuse_topology_on_nonreset_frames",
                default=False,
            )
        )
    elif reuse_topology_env.lower() in {"1", "true", "yes"}:
        reuse_topology_enabled = True
    elif reuse_topology_env.lower() in {"0", "false", "no"}:
        reuse_topology_enabled = False
    else:
        raise ValueError("RECON_GS_REUSE_TOPOLOGY must be a boolean value")

    native_index_env = os.environ.get("RECON_GS_NATIVE_INDEX_ENCODING")
    native_index_encoding = resolve_native_index_encoding(
        OmegaConf.select(runner_options, "native_index_encoding", default="legacy"),
        native_index_env,
    )
    if reuse_topology_enabled and native_index_encoding == "exact_index_delta":
        raise ValueError(
            "exact_index_delta requires the original full ReCon-GS topology; "
            "disable reuse_topology_on_nonreset_frames"
        )

    experiment_root = Path(config.output_dir) / config.experiment_name
    if experiment_root.exists() and any(experiment_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output path: {experiment_root}")

    pipeline = Pipeline(config)
    module = pipeline.module
    trainer = pipeline.trainer
    run_dir = Path(trainer.output_dir)
    stream_dir = run_dir / "frame_storage"
    metric_dir = run_dir / "stream_metrics"
    stream_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    archived_snapshot = provenance_dir / source_archive_path.name
    shutil.copy2(source_archive_path, archived_snapshot)
    atomic_json(provenance_dir / "source_snapshot_manifest.json", source_manifest)
    atomic_json(provenance_dir / "source_snapshot_archive.json", source_archive_record)
    if sha256_file(archived_snapshot) != source_archive_record["archive_sha256"]:
        raise RuntimeError("copied source snapshot archive failed hash verification")

    def synchronize() -> None:
        torch.cuda.synchronize(trainer.device)

    base_ply_info = {
        "path": str(SHARED_BASE_PLY),
        "bytes": SHARED_BASE_PLY.stat().st_size,
        "sha256": sha256_file(SHARED_BASE_PLY),
        "rows": ply_vertex_count(SHARED_BASE_PLY),
        "meaning": "shared trained frame-0 Gaussian base; counted once in sequence total",
    }
    state: dict[str, Any] = {
        "frames": {}, "current": None, "validation": {}, "renderer_benchmarks": {},
        "last_full_index": None,
    }

    def refresh_manifests(status: str = "running") -> None:
        rows = [state["frames"][f] for f in sorted(state["frames"])]
        sequence = {
            "schema": "recongs_native_delta_topology_birth_ply_v2",
            "status": status,
            "frame_range": [1, int(module.datamodule.num_frames) - 1],
            "frame_count_expected": int(module.datamodule.num_frames) - 1,
            "frames_completed": len(rows),
            "shared_base": base_ply_info,
            "motion_packet": {
                "legacy": (
                    "uncompressed np.savez; full index on rebuild frames and inherited survivor "
                    "index plus new-birth assignments on topology-reuse frames"
                ),
                "compressed_full": "np.savez_compressed ZIP_DEFLATED with a lossless full int32 index",
                "exact_index_delta": (
                    "np.savez_compressed ZIP_DEFLATED; exact changed index cells and fresh rows, "
                    "falling back to a full index packet when its actual file is smaller"
                ),
            }[native_index_encoding],
            "transition": "append prior birth PLY after prior base, then remove exact source rows; survivors derive by retaining ascending source row order",
            "frames": [
                {
                    "frame": row["frame"],
                    "packet": row["packet"]["path"],
                    "birth_ply": row["birth_ply"]["path"],
                    "transition_metadata": row["transition_metadata"]["path"],
                }
                for row in rows
            ],
        }
        atomic_json(stream_dir / "sequence_manifest.json", sequence)
        lines = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        partial = metric_dir / "frames.jsonl.partial"
        partial.write_text(lines)
        partial.replace(metric_dir / "frames.jsonl")

    orig_pre = module.pre_training_step
    orig_post = module.post_training_step
    orig_save_ply = module.save_point_cloud
    orig_frame_loop = trainer.frame_training_loop
    orig_validation = module.validation_step
    orig_checkpoint = trainer.save_checkpoint

    def wrapped_pre(self, *args, **kwargs):
        frame = int(self.current_frame)
        first_dynamic_step = frame > 0 and self.trainer.is_first_step
        if not first_dynamic_step:
            return orig_pre(*args, **kwargs)

        synchronize()
        started = time.perf_counter()
        old_base_rows = int(self.gaussians.get_xyz.shape[0])
        previous_birth_rows = int(self.gaussians_incr.get_xyz.shape[0])
        deleted_traces = []
        deleted_trace_events = []
        reuse_trace: dict[str, int] = {}
        original_prune = self.gaussians.prune_points

        def traced_prune(mask_or_indices):
            is_mask = mask_or_indices.dtype is torch.bool
            source = mask_or_indices if is_mask else mask_or_indices.reshape(-1)
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            deleted_traces.append((is_mask, source.detach().clone()))
            event_end.record()
            deleted_trace_events.append((event_start, event_end))
            return original_prune(mask_or_indices)

        self.gaussians.prune_points = traced_prune
        reuse_this_frame = (
            reuse_topology_enabled
            and frame > 1
            and bool(self.config.re_hierarchization)
            and frame % int(self.deform.config.grid_reset_interval) != 0
        )
        original_deformation_reset = self.deform.reset
        if reuse_this_frame:
            def reuse_deformation_reset(deformation, xyz):
                source_rows = old_base_rows + previous_birth_rows
                deleted_mask = torch.zeros(source_rows, dtype=torch.bool, device=xyz.device)
                for is_mask, source in deleted_traces:
                    source = source.to(device=xyz.device)
                    if is_mask:
                        if source.numel() != source_rows:
                            raise RuntimeError(
                                "topology reuse received a prune mask with the wrong row count"
                            )
                        deleted_mask |= source.reshape(-1)
                    else:
                        deleted_mask[source.reshape(-1).long()] = True
                survivor_source_rows = torch.nonzero(
                    ~deleted_mask, as_tuple=False
                ).flatten()
                if survivor_source_rows.numel() != xyz.shape[0]:
                    raise RuntimeError(
                        "topology reuse survivor rows do not match the pruned model"
                    )
                old_survivor_source_rows = survivor_source_rows[
                    survivor_source_rows < old_base_rows
                ]
                birth_output_rows = torch.nonzero(
                    survivor_source_rows >= old_base_rows, as_tuple=False
                ).flatten()
                for grid in deformation.grids:
                    previous_p2v = grid.p2v
                    if previous_p2v.shape[0] != old_base_rows:
                        raise RuntimeError(
                            "topology reuse grid rows do not match the previous base"
                        )
                    inherited_p2v = previous_p2v.index_select(
                        0, old_survivor_source_rows
                    )
                    if birth_output_rows.numel():
                        grid.reset_p2v(xyz.index_select(0, birth_output_rows))
                        birth_p2v = grid.p2v
                        grid.p2v = torch.cat((inherited_p2v, birth_p2v), dim=0)
                    else:
                        grid.p2v = inherited_p2v

                index_parts, offset = [], 0
                for grid in deformation.grids:
                    index_parts.append(grid.p2v + offset)
                    offset += grid.vertex_num
                deformation.index = torch.stack(index_parts).squeeze(dim=-1)
                if deformation.config.momentum is not None:
                    deformation.delta.data *= deformation.config.momentum
                deformation.grids_num = offset
                reuse_trace["reused_survivor_rows"] = int(
                    old_survivor_source_rows.numel()
                )
                reuse_trace["recomputed_birth_rows"] = int(birth_output_rows.numel())
                reuse_trace["layers"] = len(deformation.grids)

            self.deform.reset = MethodType(reuse_deformation_reset, self.deform)
        try:
            result = orig_pre(*args, **kwargs)
        finally:
            del self.gaussians.prune_points
            if reuse_this_frame:
                self.deform.reset = original_deformation_reset

        state["current"] = {
            "frame": frame,
            "old_base_rows": old_base_rows,
            "previous_birth_rows": previous_birth_rows,
            "deleted_trace_tensors": deleted_traces,
            "deleted_trace_events": deleted_trace_events,
            "step_started": started,
            "ply_save_wall_s": 0.0,
            "suppress_final_ply": False,
            "baked_delta_tensor": None,
            "baked_index_tensor": None,
            "applied_xyz_tensor": None,
            "applied_rotation_tensor": None,
            "capture_event_start": None,
            "capture_event_end": None,
            "original_apply": None,
            "topology_reused": reuse_this_frame,
            "topology_reuse": reuse_trace,
        }

        # The native deformation applied at stage-1 step 100 is consumed
        # before its optimizer update. Freeze these source arrays at the exact
        # apply_deformation call; the live delta changes afterward.
        current = state["current"]
        original_apply = self.gaussians.apply_deformation
        current["original_apply"] = original_apply

        def traced_apply(model, delta_xyz, delta_rotation):
            if (current["baked_delta_tensor"] is None
                    and int(self.global_step) == int(self.deform.config.num_stage1_steps)):
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
                current["baked_delta_tensor"] = self.deform.delta.detach().clone()
                current["baked_index_tensor"] = self.deform.index.detach().clone()
                current["applied_xyz_tensor"] = delta_xyz.detach()
                current["applied_rotation_tensor"] = delta_rotation.detach()
                event_end.record()
                current["capture_event_start"] = event_start
                current["capture_event_end"] = event_end
            return original_apply(delta_xyz, delta_rotation)

        self.gaussians.apply_deformation = MethodType(traced_apply, self.gaussians)
        return result

    def wrapped_save_point_cloud(self):
        current = state.get("current")
        if current is not None and current.get("suppress_final_ply"):
            return None
        timed = (current is not None and current["frame"] == int(self.current_frame)
                 and self.trainer.is_last_step)
        if not timed:
            return orig_save_ply()
        synchronize()
        started = time.perf_counter()
        try:
            return orig_save_ply()
        finally:
            synchronize()
            current["ply_save_wall_s"] += time.perf_counter() - started

    def wrapped_post(self, render_pkg, *args, **kwargs):
        current = state.get("current")
        final_dynamic_step = (
            current is not None
            and current["frame"] == int(self.current_frame)
            and self.trainer.is_last_step
        )
        if final_dynamic_step:
            current["suppress_final_ply"] = True
        try:
            result = orig_post(render_pkg, *args, **kwargs)
        finally:
            if final_dynamic_step:
                current["suppress_final_ply"] = False
        if not final_dynamic_step:
            return result

        frame = current["frame"]
        synchronize()
        training_steps_wall = time.perf_counter() - current["step_started"]

        # Save the frame's birth PLY only after ending the train timer.
        ply_started = time.perf_counter()
        orig_save_ply()
        synchronize()
        current["ply_save_wall_s"] += time.perf_counter() - ply_started

        export_started = time.perf_counter()
        deletion_started = time.perf_counter()
        deleted_parts = []
        for is_mask, source in current["deleted_trace_tensors"]:
            source_cpu = source.detach().to(device="cpu")
            values = source_cpu.numpy()
            deleted_parts.append(
                np.flatnonzero(values).astype(np.int64, copy=False)
                if is_mask else values.astype(np.int64, copy=False)
            )
        all_deleted = (np.concatenate(deleted_parts) if deleted_parts
                       else np.empty(0, dtype=np.int64))
        transition = normalize_transition(
            current["old_base_rows"], current["previous_birth_rows"], all_deleted
        )
        if (self.config.merge_to_base and current["previous_birth_rows"]
                and all_deleted.size != current["previous_birth_rows"]):
            raise RuntimeError(
                f"frame {frame}: expected {current['previous_birth_rows']} pruned rows, "
                f"observed {all_deleted.size}"
            )
        current_base_rows = int(self.gaussians.get_xyz.shape[0])
        if current_base_rows != transition["survivor_rows"]:
            raise RuntimeError(
                f"frame {frame}: transition keeps {transition['survivor_rows']} rows, "
                f"model has {current_base_rows}"
            )
        deletion_diagnostic_wall = time.perf_counter() - deletion_started

        if current["baked_delta_tensor"] is None or current["baked_index_tensor"] is None:
            raise RuntimeError(f"frame {frame}: failed to capture stage-1 applied deformation")
        delta = current["baked_delta_tensor"].to(device="cpu").numpy()
        index = current["baked_index_tensor"].to(device="cpu").numpy()
        applied_xyz = current["applied_xyz_tensor"].to(device="cpu").numpy()
        applied_rotation = current["applied_rotation_tensor"].to(device="cpu").numpy()
        if index.ndim != 2 or index.shape[1] != transition["survivor_rows"]:
            raise RuntimeError(
                f"frame {frame}: native index shape {index.shape} does not match survivor rows "
                f"{transition['survivor_rows']}"
            )
        max_translation_difference = verify_native_application(
            delta, index, applied_xyz, applied_rotation
        )
        packet_path = stream_dir / f"native_motion_{frame:03d}.npz"
        previous_decoded_index = state["last_full_index"]
        if native_index_encoding == "exact_index_delta":
            packet_serialization = write_exact_index_delta_packet(
                packet_path,
                delta,
                index,
                transition,
                previous_index=state["last_full_index"],
            )
        elif native_index_encoding == "compressed_full":
            packet_serialization = write_compressed_full_packet(
                packet_path, delta, index, transition
            )
        else:
            previous_index = (
                state["last_full_index"] if current["topology_reused"] else None
            )
            packet_bytes = write_native_packet(
                packet_path, delta, index, transition, previous_index=previous_index
            )
            packet_serialization = {
                "bytes": packet_bytes,
                "index_encoding": (
                    "inherit_survivor_rows_plus_birth_assignments"
                    if current["topology_reused"] else "full"
                ),
                "index_encoding_code": 1 if current["topology_reused"] else 0,
                "full_candidate_bytes": None,
                "exact_delta_candidate_bytes": None,
                "packet_encode_and_write_wall_s": None,
                "compression": "none (ZIP_STORED np.savez arrays)",
            }
        packet_bytes = int(packet_serialization["bytes"])
        # Close the serialization loop before advancing the next frame's state.
        # This catches corrupt/mis-shaped packets in the live writer itself.
        with np.load(packet_path, allow_pickle=False) as written_packet:
            decoded_index = decode_topology_index(
                written_packet, previous_decoded_index, transition
            )
            packet_serialization["stored_array_dtypes"] = {
                key: str(written_packet[key].dtype) for key in written_packet.files
            }
        if not np.array_equal(decoded_index, index):
            raise RuntimeError(
                f"frame {frame}: disk packet did not decode to the frozen source index"
            )
        packet_serialization["decoded_index_dtype"] = str(decoded_index.dtype)
        state["last_full_index"] = decoded_index.copy()

        birth_path = run_dir / "point_cloud" / f"gaussians-{frame}-{self.num_steps}.ply"
        if not birth_path.is_file():
            raise FileNotFoundError(f"frame {frame}: expected native incremental PLY at {birth_path}")
        birth_rows = ply_vertex_count(birth_path)
        expected_birth_rows = int(self.gaussians_incr.get_xyz.shape[0])
        if birth_rows != expected_birth_rows:
            raise RuntimeError(
                f"frame {frame}: PLY has {birth_rows} rows but incremental model has {expected_birth_rows}"
            )

        capture_cuda_s = None
        if current["capture_event_start"] is not None:
            capture_cuda_s = (
                current["capture_event_start"].elapsed_time(current["capture_event_end"])
                / 1000.0
            )
        trace_capture_cuda_s = sum(
            start.elapsed_time(end) for start, end in current["deleted_trace_events"]
        ) / 1000.0
        transition_meta = {
            "schema": "recongs_frame_transition_v1",
            "frame": frame,
            "source_frame": frame - 1,
            "source_row_order": transition["row_order"],
            "old_base_rows": current["old_base_rows"],
            "previous_birth_rows": current["previous_birth_rows"],
            "concatenated_source_rows": transition["source_rows"],
            "deleted_rows": int(transition["deleted_source_indices"].size),
            "survivor_rows": transition["survivor_rows"],
            "motion": {
                "path": str(packet_path.relative_to(run_dir)),
                "bytes": packet_bytes,
                "sha256": sha256_file(packet_path),
                "delta_shape": list(delta.shape),
                "delta_dtype": str(delta.dtype),
                "index_shape": list(index.shape),
                "source_index_dtype": str(index.dtype),
                "index_dtype": str(decoded_index.dtype),
                "stored_array_dtypes": packet_serialization["stored_array_dtypes"],
                "index_encoding": packet_serialization["index_encoding"],
                "index_encoding_code": packet_serialization["index_encoding_code"],
                "index_axis_semantics": "[grid_level, surviving_gaussian_row] after decode",
                "deletion_array": "deleted_source_indices",
                "survivors": "derived by retaining source rows in ascending original row order",
                "compression": packet_serialization["compression"],
                "delta_semantics": "snapshot at stage-1 apply_deformation call, before optimizer update",
                "application_reconstruction_check": "native delta/index reproduces stage-1 per-row translation and normalized quaternion",
            },
            "birth_ply": {
                "path": str(birth_path.relative_to(run_dir)),
                "bytes": birth_path.stat().st_size,
                "sha256": sha256_file(birth_path),
                "rows": birth_rows,
                "row_order": "native GaussianModel incremental order",
            },
        }
        transition_path = stream_dir / f"transition_{frame:03d}.json"
        atomic_json(transition_path, transition_meta)
        export_wall = time.perf_counter() - export_started

        record = {
            "frame": frame,
            "steps": int(self.num_steps),
            "transition": {
                "old_base_rows": current["old_base_rows"],
                "previous_birth_rows": current["previous_birth_rows"],
                "source_rows": transition["source_rows"],
                "deleted_rows": int(transition["deleted_source_indices"].size),
                "survivor_rows": transition["survivor_rows"],
            },
            "packet": {
                "path": str(packet_path.relative_to(run_dir)),
                "bytes": packet_bytes,
                "delta_shape": list(delta.shape),
                "delta_dtype": str(delta.dtype),
                "index_shape": list(index.shape),
                "source_index_dtype": str(index.dtype),
                "index_dtype": str(decoded_index.dtype),
                "stored_array_dtypes": packet_serialization["stored_array_dtypes"],
                "index_encoding": packet_serialization["index_encoding"],
                "index_encoding_code": packet_serialization["index_encoding_code"],
                "max_translation_reconstruction_abs_difference": max_translation_difference,
                "serializer_candidates": {
                    key: value for key, value in packet_serialization.items()
                    if key not in {"bytes", "index_encoding", "index_encoding_code", "compression"}
                },
                "compression": packet_serialization["compression"],
            },
            "topology_reuse": current["topology_reuse"],
            "birth_ply": {
                "path": str(birth_path.relative_to(run_dir)),
                "bytes": birth_path.stat().st_size,
                "rows": birth_rows,
            },
            "transition_metadata": {
                "path": str(transition_path.relative_to(run_dir)),
                "bytes": transition_path.stat().st_size,
            },
            "timing": {
                "training_steps_wall_synchronized_s": training_steps_wall,
                "algorithm_frame_estimate_s": (
                    max(0.0, training_steps_wall - (capture_cuda_s or 0.0)
                        - trace_capture_cuda_s)
                ),
                "deformation_snapshot_cuda_s": capture_cuda_s,
                "transition_trace_cuda_s": trace_capture_cuda_s,
                "incremental_ply_save_wall_s": current["ply_save_wall_s"],
                "transition_capture_diagnostic_wall_s": deletion_diagnostic_wall,
                "native_packet_and_metadata_wall_s": export_wall,
                "packet_encode_and_write_wall_s": packet_serialization.get(
                    "packet_encode_and_write_wall_s"
                ),
                "index_delta_prepare_wall_s": packet_serialization.get(
                    "index_delta_prepare_wall_s"
                ),
                "full_candidate_compress_wall_s": packet_serialization.get(
                    "full_compress_wall_s"
                ),
                "exact_delta_candidate_compress_wall_s": packet_serialization.get(
                    "delta_compress_wall_s"
                ),
                "packet_write_wall_s": packet_serialization.get("packet_write_wall_s"),
                "semantics": "first pre-step through final post-step; synchronized; includes frame setup, data fetch, 200 training steps, backward/optimizer and logging; excludes final PLY, host serialization/metadata IO, checkpoint and validation; estimate subtracts measured CUDA snapshot-copy time but retains Python hook/event overhead and is not an uninstrumented measurement",
            },
            "evaluation": {},
            "complete_training_steps": True,
        }
        self.gaussians.apply_deformation = current["original_apply"]
        state["frames"][frame] = record
        state["current"] = None
        refresh_manifests()
        return result

    def wrapped_validation(self, viewpoint, idx: int, loader_idx: int):
        synchronize()
        started = time.perf_counter()
        image = self.forward(viewpoint)["render"].clamp(0.0, 1.0)
        synchronize()
        elapsed = time.perf_counter() - started
        gt_image = viewpoint.image
        metrics = {"l1": l1_loss(image, gt_image), "psnr": psnr(image, gt_image)}
        eval_name = self.datamodule.eval_names[loader_idx]
        frame = int(self.current_frame)
        camera = Path(viewpoint.path).name
        if self.config.full_eval:
            metrics["ssim"] = ssim(image, gt_image)
            if self.config.lpips_required:
                metrics["lpips"] = lpips(image, gt_image, net_type="vgg").mean()
            metrics["fps"] = 1.0 / elapsed if elapsed > 0 else float("inf")
        views = state["validation"].setdefault(frame, {}).setdefault(eval_name, [])
        views.append({
            "view": camera,
            "single_call_synchronized_seconds": elapsed,
            **{name: scalar(value) for name, value in metrics.items()},
        })
        if frame > 0 and eval_name == "test" and "cam00" in camera.lower():
            call = lambda: self.forward(viewpoint)
            state["renderer_benchmarks"][frame] = {
                "camera": camera,
                **synchronized_fps(call, torch, trainer.device),
            }
        return metrics

    def wrapped_frame_loop(self):
        frame = int(self.current_frame)
        should_time = frame > 0 and self.global_step < self.num_steps
        if not should_time:
            return orig_frame_loop()
        synchronize()
        started = time.perf_counter()
        result = orig_frame_loop()
        synchronize()
        elapsed = time.perf_counter() - started
        record = state["frames"].get(frame)
        if record is None:
            raise RuntimeError(f"frame {frame} finished without exporting a native stream record")
        record["timing"]["frame_loop_wall_synchronized_s_including_validation_and_io"] = elapsed
        record["timing"]["frame_total_wall_s_including_training_export_validation_io"] = elapsed
        record["evaluation"] = {}
        for name, views in state["validation"].get(frame, {}).items():
            row = {"views": views}
            for key in ("l1", "psnr", "ssim", "lpips"):
                values = [view[key] for view in views if key in view]
                if values:
                    row[key] = mean(values)
            record["evaluation"][name] = row
        if frame not in state["renderer_benchmarks"]:
            raise RuntimeError(f"frame {frame} has no cam00 synchronized renderer benchmark")
        record["renderer_benchmark"] = state["renderer_benchmarks"][frame]
        refresh_manifests()
        return result

    def wrapped_checkpoint(self):
        frame = int(self.current_frame)
        interval = int(self.config.saving_ckpt_every_n_frames)
        should_save = self.is_last_frame or frame % interval == 0
        if frame <= 0 or not self.is_last_step or not should_save:
            return orig_checkpoint()
        synchronize()
        started = time.perf_counter()
        result = orig_checkpoint()
        synchronize()
        record = state["frames"].get(frame)
        if record is not None:
            record["timing"]["checkpoint_wall_synchronized_s"] = time.perf_counter() - started
        return result

    module.pre_training_step = MethodType(wrapped_pre, module)
    module.post_training_step = MethodType(wrapped_post, module)
    module.save_point_cloud = MethodType(wrapped_save_point_cloud, module)
    module.validation_step = MethodType(wrapped_validation, module)
    trainer.frame_training_loop = MethodType(wrapped_frame_loop, trainer)
    trainer.save_checkpoint = MethodType(wrapped_checkpoint, trainer)

    max_dynamic_frames_raw = os.environ.get("RECON_GS_MAX_DYNAMIC_FRAMES")
    max_dynamic_frames = int(max_dynamic_frames_raw) if max_dynamic_frames_raw else None
    if max_dynamic_frames is not None:
        if max_dynamic_frames < 1:
            raise ValueError("RECON_GS_MAX_DYNAMIC_FRAMES must be at least 1")
        orig_datamodule_setup = module.datamodule.setup

        def wrapped_datamodule_setup(self, *args, **kwargs):
            result = orig_datamodule_setup(*args, **kwargs)
            available = int(self.scene.train_dataset.num_frames)
            if available < 2:
                raise RuntimeError("smoke run requires frame 0 plus one dynamic frame")
            limited = min(available, max_dynamic_frames + 1)
            self.scene.train_dataset.num_frames = limited
            if self.scene.test_dataset is not None:
                self.scene.test_dataset.num_frames = min(
                    int(self.scene.test_dataset.num_frames), limited
                )
            run_metadata["num_frames"] = limited
            run_metadata["dynamic_frame_range"] = [1, limited - 1]
            run_metadata["max_dynamic_frames_override"] = max_dynamic_frames
            atomic_json(run_dir / "stream_metrics" / "run_metadata.json", run_metadata)
            refresh_manifests()
            return result

        module.datamodule.setup = MethodType(wrapped_datamodule_setup, module.datamodule)

    run_metadata = {
        "status": "running",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "recongs_root": str(RECON_GS_ROOT),
        "recongs_source_snapshot": source_manifest,
        "recongs_source_snapshot_archive": {
            **source_archive_record,
            "copied_path": str(archived_snapshot),
        },
        "shared_frame0_checkpoint": str(config.ckpt_path),
        "shared_frame0_checkpoint_bytes": Path(config.ckpt_path).stat().st_size,
        "shared_frame0_checkpoint_sha256": sha256_file(Path(config.ckpt_path)),
        "shared_base": base_ply_info,
        "num_frames": int(module.datamodule.num_frames),
        "dynamic_frame_range": [1, int(module.datamodule.num_frames) - 1],
        "max_dynamic_frames_override": max_dynamic_frames,
        "reuse_topology_on_nonreset_frames": reuse_topology_enabled,
        "native_index_encoding": native_index_encoding,
        "run_dir": str(run_dir),
        "render_timing": "per dynamic frame: five warmups then 30 CUDA-synchronized cam00 module-forward calls without clamp (matching the existing playback comparison); quality metrics are separately computed over all configured test views",
        "training_timing": "CUDA-synchronized first pre-step through final post-step per frame; reported separately from final PLY, packet encoding/metadata IO, checkpoint and validation. frame_total_wall_s_including_training_export_validation_io includes those costs",
    }
    atomic_json(run_dir / "stream_metrics" / "run_metadata.json", run_metadata)
    refresh_manifests()
    try:
        pipeline.run()
    except BaseException as error:
        run_metadata["status"] = "interrupted"
        run_metadata["error"] = f"{type(error).__name__}: {error}"
        atomic_json(run_dir / "stream_metrics" / "run_metadata.json", run_metadata)
        refresh_manifests("interrupted")
        raise

    rows = [state["frames"][frame] for frame in sorted(state["frames"])]
    completed_status = (
        "completed" if len(rows) == int(module.datamodule.num_frames) - 1 else "incomplete"
    )
    refresh_manifests(completed_status)
    sequence_manifest_bytes = (stream_dir / "sequence_manifest.json").stat().st_size
    sizes = {
        "native_motion_and_topology_npz_bytes": sum(row["packet"]["bytes"] for row in rows),
        "incremental_birth_ply_bytes": sum(row["birth_ply"]["bytes"] for row in rows),
        "required_transition_json_bytes": sum(row["transition_metadata"]["bytes"] for row in rows),
        "required_sequence_manifest_bytes": sequence_manifest_bytes,
    }
    sizes["dynamic_stream_bytes"] = sum(sizes.values())
    sizes["shared_frame0_ply_bytes"] = base_ply_info["bytes"]
    sizes["total_with_shared_base_bytes"] = sizes["dynamic_stream_bytes"] + base_ply_info["bytes"]
    test_rows = [row["evaluation"]["test"] for row in rows if "test" in row["evaluation"]]
    train_wall = [row["timing"]["training_steps_wall_synchronized_s"] for row in rows]
    frame_wall = [
        row["timing"].get("frame_loop_wall_synchronized_s_including_validation_and_io", 0.0)
        for row in rows
    ]
    fps_values = [row["renderer_benchmark"]["fps"] for row in rows]
    algorithm_estimates = [row["timing"]["algorithm_frame_estimate_s"] for row in rows]
    summary = {
        "status": completed_status,
        "frames_completed": len(rows),
        "frames_expected": int(module.datamodule.num_frames) - 1,
        "quality_mean": {
            key: mean([row[key] for row in test_rows if key in row])
            for key in ("psnr", "ssim", "l1")
            if any(key in row for row in test_rows)
        },
        "cam00_synchronized_renderer_fps": {
            "mean": mean(fps_values) if fps_values else None,
            "median": float(np.median(fps_values)) if fps_values else None,
            "frames": len(fps_values),
            "scope": "per dynamic frame, same cam00 test view, 5 warmups + 30 synchronized module-forward calls, no clamp",
        },
        "algorithm_frame_estimate_s": {
            "mean": mean(algorithm_estimates) if algorithm_estimates else None,
            "median": float(np.median(algorithm_estimates)) if algorithm_estimates else None,
            "total": sum(algorithm_estimates),
            "note": "estimate subtracts measured CUDA tensor-snapshot copy time only; Python instrumentation overhead remains, so use raw synchronized wall time alongside it",
        },
        "training_steps_wall_synchronized_s": {
            "mean": mean(train_wall) if train_wall else None,
            "median": float(np.median(train_wall)) if train_wall else None,
            "total": sum(train_wall),
        },
        "frame_loop_wall_synchronized_s_including_validation_and_io": {
            "mean": mean(frame_wall) if frame_wall else None,
            "median": float(np.median(frame_wall)) if frame_wall else None,
            "total": sum(frame_wall),
        },
        "actual_stream_bytes": sizes,
        "native_index_encoding": native_index_encoding,
        "shared_base": base_ply_info,
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "stream_metrics" / "experiment_summary.json", summary)
    run_metadata["status"] = summary["status"]
    atomic_json(run_dir / "stream_metrics" / "run_metadata.json", run_metadata)
    refresh_manifests(summary["status"])
    print(json.dumps(summary, indent=2))
    return run_dir


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG_PATH
    run(config_arg)
