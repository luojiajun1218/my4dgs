#!/usr/bin/env python3
"""Freeze the current ReCon-GS source without copying its data or outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE = Path("/home/gamer-02/Documents/DeepLearning/ReCon-GS")
DEFAULT_TARGET = Path("/tmp/recongs_native_source_snapshot_20261002")
DEFAULT_ARCHIVE = Path("/tmp/recongs_native_source_snapshot_20261002.tar.gz")
EXPECTED_DIRTY_FILES = ("main.py", "pipeline/ReCon-GS/module.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == "baseline_source_snapshot.json":
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        if path.is_symlink():
            digest.update(b"L\0" + os.readlink(path).encode("utf-8") + b"\0")
        elif path.is_file():
            digest.update(b"F\0" + bytes.fromhex(sha256(path)))
        elif path.is_dir():
            digest.update(b"D\0")
    return digest.hexdigest()


def run_git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args], check=True,
        capture_output=True, text=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    archive = args.archive.resolve()
    if target.exists() or archive.exists():
        raise FileExistsError(f"refusing to overwrite snapshot or archive: {target}, {archive}")

    status = run_git(source, "status", "--porcelain", "--untracked-files=no").splitlines()
    dirty_paths = [line[3:] for line in status]
    if sorted(dirty_paths) != sorted(EXPECTED_DIRTY_FILES):
        raise RuntimeError(f"expected only {EXPECTED_DIRTY_FILES}, found tracked changes {status}")
    commit = run_git(source, "rev-parse", "HEAD").strip()
    dataset = (source / "data/colmap4d/flame_steak").resolve(strict=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    archive_tmp = target.with_suffix(".source.tar")
    with archive_tmp.open("wb") as stream:
        subprocess.run(
            ["git", "-C", str(source), "archive", "--format=tar", "HEAD"],
            check=True, stdout=stream,
        )
    target.mkdir(parents=True)
    try:
        with tarfile.open(archive_tmp, "r") as source_tar:
            source_tar.extractall(target, filter="data")
    finally:
        archive_tmp.unlink(missing_ok=True)

    file_hashes = {}
    for relative in EXPECTED_DIRTY_FILES:
        source_file = source / relative
        target_file = target / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        file_hashes[relative] = sha256(target_file)

    dataset_link = target / "data/colmap4d/flame_steak"
    dataset_link.parent.mkdir(parents=True, exist_ok=True)
    if dataset_link.exists() or dataset_link.is_symlink():
        if dataset_link.is_dir() and not dataset_link.is_symlink():
            shutil.rmtree(dataset_link)
        else:
            dataset_link.unlink()
    dataset_link.symlink_to(dataset, target_is_directory=True)

    # Minimal git metadata lets libgs report the pinned upstream revision.
    git_head = target / ".git/HEAD"
    git_head.parent.mkdir(parents=True, exist_ok=True)
    git_head.write_text(commit + "\n")
    tree_hash = tree_sha256(target)
    snapshot_manifest = {
        "schema": "recongs_frozen_source_snapshot_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkout": str(source),
        "source_commit": commit,
        "tracked_worktree_status": status,
        "overlaid_dirty_source_files_sha256": file_hashes,
        "source_tree_sha256_excluding_this_manifest": tree_hash,
        "dataset_symlink": str(dataset_link),
        "dataset_target": str(dataset),
        "data_copied": False,
        "outputs_copied": False,
    }
    (target / "baseline_source_snapshot.json").write_text(
        json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n"
    )

    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz", compresslevel=6) as output_tar:
        output_tar.add(target, arcname=target.name, recursive=True)
    archive_record = {
        "archive_path": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "snapshot_tree_sha256": tree_hash,
    }
    archive_record_path = archive.with_suffix(archive.suffix + ".json")
    archive_record_path.write_text(json.dumps(archive_record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**snapshot_manifest, **archive_record,
                      "archive_record_path": str(archive_record_path)}, indent=2))


if __name__ == "__main__":
    main()
