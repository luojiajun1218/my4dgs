#!/usr/bin/env python3
"""Build a dataset overlay with a denser sampled MVS initialization.

Only the derived point cloud and small manifests are written.  Images, frame
directories, camera files, poses and split metadata remain linked to the
source dataset, so the source tree and its symlink targets are never changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np


_POINT_ENTRIES = {"points3D.ply", "points3D_downsample2.ply"}
_SPECIAL_ENTRIES = _POINT_ENTRIES | {"sparse", "adapter.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_cloud(cloud: Any, expected_points: int | None = None) -> int:
    arrays = [np.asarray(cloud.points), np.asarray(cloud.colors), np.asarray(cloud.normals)]
    count = len(arrays[0])
    if expected_points is not None and count != expected_points:
        raise ValueError(f"Expected {expected_points} points, found {count}")
    if count < 1 or not cloud.has_colors() or not cloud.has_normals():
        raise ValueError("Point cloud must contain points, colors and normals")
    if any(array.shape != (count, 3) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Point cloud contains malformed or non-finite attributes")
    return count


def _link_absolute(link: Path, target: Path, *, directory: bool | None = None) -> None:
    if directory is None:
        directory = target.is_dir()
    link.symlink_to(target.resolve(), target_is_directory=directory)


def derive_overlay(
    source: Path,
    fused_ply: Path,
    output: Path,
    repo_link: Path,
    initial_points: int,
) -> dict[str, Any]:
    """Create and atomically publish a read-only-data overlay."""
    source = source.resolve(strict=True)
    fused_ply = fused_ply.resolve(strict=True)
    output = output.absolute()
    repo_link = repo_link.absolute()

    if initial_points < 1:
        raise ValueError("--initial-points must be positive")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if repo_link.exists() or repo_link.is_symlink():
        raise FileExistsError(repo_link)
    if output == source or source in output.parents:
        raise ValueError("Output must not be the source dataset or live inside it")
    if not fused_ply.is_file():
        raise FileNotFoundError(fused_ply)

    required = [
        source / "frames",
        source / "sparse/0/cameras.bin",
        source / "sparse/0/images.bin",
        source / "sparse/0/points3D.ply",
        source / "test_split.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Source dataset is incomplete: {missing}")

    import open3d as o3d

    fused = o3d.io.read_point_cloud(str(fused_ply))
    fused_points = _validate_cloud(fused)
    if fused_points < initial_points:
        raise ValueError(
            f"Fused cloud has {fused_points} points, fewer than requested {initial_points}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    repo_link.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    published = False
    try:
        sampled = fused.farthest_point_down_sample(initial_points)
        _validate_cloud(sampled, initial_points)
        cloud_name = f"points3D_mvs_fps_{initial_points}.ply"
        cloud_path = staging / cloud_name
        if not o3d.io.write_point_cloud(
            str(cloud_path), sampled, write_ascii=False, compressed=False, print_progress=True
        ):
            raise RuntimeError("Open3D failed to write the sampled point cloud")
        reloaded = o3d.io.read_point_cloud(str(cloud_path))
        _validate_cloud(reloaded, initial_points)

        linked_entries: list[str] = []
        for entry in sorted(source.iterdir(), key=lambda item: item.name):
            if entry.name in _SPECIAL_ENTRIES:
                continue
            _link_absolute(staging / entry.name, entry)
            linked_entries.append(entry.name)

        (staging / "points3D.ply").symlink_to(cloud_name)
        (staging / "points3D_downsample2.ply").symlink_to("points3D.ply")
        sparse = staging / "sparse/0"
        sparse.mkdir(parents=True)
        _link_absolute(sparse / "cameras.bin", source / "sparse/0/cameras.bin", directory=False)
        _link_absolute(sparse / "images.bin", source / "sparse/0/images.bin", directory=False)
        (sparse / "points3D.ply").symlink_to("../../points3D.ply")

        source_point = (source / "sparse/0/points3D.ply").resolve(strict=True)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "created_local_time": datetime.now().astimezone().isoformat(),
            "source_dataset": str(source),
            "source_dataset_unchanged": True,
            "source_images_copied": False,
            "linked_entries": linked_entries,
            "source_initial_cloud": str(source_point),
            "source_initial_cloud_sha256": sha256(source_point),
            "fused_cloud": str(fused_ply),
            "fused_cloud_sha256": sha256(fused_ply),
            "fused_points": fused_points,
            "sample_method": "open3d.farthest_point_down_sample",
            "sample_determinism": "deterministic first-point seed from source ordering",
            "initial_points": initial_points,
            "initial_cloud_file": cloud_name,
            "initial_cloud_sha256": sha256(cloud_path),
            "open3d_version": o3d.__version__,
        }
        (staging / "initial_cloud_overlay.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        adapter: dict[str, Any] = {}
        source_adapter = source / "adapter.json"
        if source_adapter.is_file():
            adapter = json.loads(source_adapter.read_text())
        adapter["initial_cloud_overlay"] = manifest
        (staging / "adapter.json").write_text(
            json.dumps(adapter, indent=2, sort_keys=True) + "\n"
        )

        staging.rename(output)
        published = True
        repo_link.symlink_to(output.resolve(strict=True), target_is_directory=True)
        manifest["output_dataset"] = str(output)
        manifest["repo_link"] = str(repo_link)
        return manifest
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fused-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-link", type=Path, required=True)
    parser.add_argument("--initial-points", type=int, required=True)
    args = parser.parse_args()
    manifest = derive_overlay(
        args.source, args.fused_ply, args.output, args.repo_link, args.initial_points
    )
    print("INITIAL_CLOUD_OVERLAY_COMPLETE")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
