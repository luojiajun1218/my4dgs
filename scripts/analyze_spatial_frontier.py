"""Counterfactual byte audit for a completed spatial-motion checkpoint.

The decoded total motion, survivor set and births are held fixed.  The script
re-encodes that same state with the learned frontier and with uniform-depth
frontiers whose regional representatives are coordinate medians.  The latter
are defined diagnostics, not globally optimal byte lower bounds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.model.rigid_tree_codec import DecodedState, decode_motion, encode_motion


def _absolute_regions(parameters, parents, translation_step, rotation_step):
    local = torch.cat((
        torch.round(parameters["region_translation"] / translation_step),
        torch.round(parameters["region_rotation"] / rotation_step),
    ), dim=1).to(torch.int32)
    local[0] = 0
    absolute = local.clone()
    for node in range(1, len(absolute)):
        absolute[node] += absolute[parents[node]]
    return absolute


def _stored_nodes(owner, parents, depth):
    active = torch.bincount(owner, minlength=len(parents)).gt(0)
    nodes = active.nonzero(as_tuple=True)[0]
    for _ in range(depth):
        active[nodes] = True
        nodes = parents[nodes].clamp_min(0)
    active[0] = False
    return active


def _uniform_medians(total, paths, initial):
    representatives = initial.clone()
    values = total.numpy()
    routes = paths.numpy()
    for level in range(paths.shape[1]):
        groups = routes[:, level]
        order = np.argsort(groups, kind="stable")
        nodes, starts = np.unique(groups[order], return_index=True)
        for node, rows in zip(nodes, np.split(order, starts[1:])):
            median = np.rint(np.median(values[rows], axis=0)).astype(np.int32)
            representatives[int(node)] = torch.from_numpy(median)
    return representatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--motion-packet", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    capture = checkpoint["rigid_tree_motion"]
    parameters = capture["parameters"]
    tensors = capture["tensors"]
    config = capture["config"]
    frame = int(capture["frame_index"])
    if frame < 1 or frame != int(checkpoint["current_frame"]):
        raise ValueError("Expected a completed dynamic-frame checkpoint")

    extent = float(capture["extent"])
    translation_step = extent * float(config["motion_xyz_step_extent"])
    rotation_step = float(config["motion_rotation_step_rad"])
    gs_translation_step = extent * float(config["gs_xyz_step_extent"])
    gs_rotation_step = float(config["gs_rotation_step_rad"])
    parents = tensors["parents"].long()
    paths = tensors["spatial_paths"].long()
    learned_owner = tensors["owner"].long()
    learned_open = tensors["opened"].bool()
    depth = paths.shape[1]

    absolute = _absolute_regions(parameters, parents, translation_step, rotation_step)
    point = torch.cat((
        torch.round(parameters["gs_translation"] / gs_translation_step),
        torch.round(parameters["gs_rotation"] / gs_rotation_step),
    ), dim=1).to(torch.int32)
    total = absolute[learned_owner] + point
    medians = _uniform_medians(total, paths, absolute)

    prior = DecodedState.restore(capture["frame_reference_state"])
    sources = tensors["pending_birth_sources"].long()
    old_count = len(learned_owner)
    birth_xyz = tensors["materialized_xyz"][old_count:]
    birth_rotation = tensors["materialized_rotation"][old_count:]
    module_config = checkpoint["rigid_tree_module_config"]
    birth_config = module_config["birth"]
    birth_pose_steps = None
    if birth_config["radiance_mode"] == "adaptive_residual_birth":
        birth_pose_steps = (
            extent * float(birth_config["new_xyz_step_extent"]),
            float(birth_config["new_rotation_step"]),
        )
    inherit_birth_rotation = not (
        bool(birth_config["rotation_updates"]) or bool(config["rotation_updates"])
    )
    spatial_relative = config["spatial_gate"]["region_motion"] == "relative"

    def encode(owner, opened, representatives):
        local = representatives.clone()
        local[1:] -= representatives[parents[1:]]
        local[0] = 0
        stored = _stored_nodes(owner, parents, depth)
        local *= stored[:, None]
        residual = total - representatives[owner]
        if bool(residual.abs().gt(32767).any()) or bool(local.abs().gt(32767).any()):
            raise ValueError("Counterfactual representation exceeds int16 range")
        packet = encode_motion(
            prior, frame, extent, tensors["previous_rows"], owner, parents,
            local[:, :3].to(torch.int16), local[:, 3:].to(torch.int16),
            residual[:, :3].to(torch.int16), residual[:, 3:].to(torch.int16),
            sources, birth_xyz, birth_rotation,
            config["motion_xyz_step_extent"], config["motion_rotation_step_rad"],
            compression="auto",
            gs_translation_step_extent=config["gs_xyz_step_extent"],
            gs_rotation_step_rad=config["gs_rotation_step_rad"],
            spatial_open=opened, spatial_relative=spatial_relative,
            inherit_birth_rotation=inherit_birth_rotation,
            birth_pose_steps=birth_pose_steps,
        )
        decoded = decode_motion(packet, prior)
        expected_xyz = tensors["materialized_xyz"]
        expected_rotation = tensors["materialized_rotation"]
        if not torch.allclose(decoded.xyz, expected_xyz, atol=1e-6, rtol=1e-5):
            raise ValueError("Counterfactual changed the decoded positions")
        dot = (torch.nn.functional.normalize(decoded.rotation, dim=-1) *
               torch.nn.functional.normalize(expected_rotation, dim=-1)).sum(-1).abs()
        if not bool(dot.gt(1 - 1e-5).all()):
            raise ValueError("Counterfactual changed the decoded rotations")
        return packet, {
            "bytes": len(packet),
            "active_regions": int(owner.unique().numel()),
            "point_residual_rows": int(residual.ne(0).any(-1).sum()),
            "stored_region_rows": int(local.ne(0).any(-1).sum()),
        }

    learned_packet, learned = encode(learned_owner, learned_open, absolute)
    saved_packet = args.motion_packet.read_bytes()
    if learned_packet != saved_packet:
        raise ValueError("Checkpoint does not reproduce the supplied motion packet exactly")
    learned["sha256"] = hashlib.sha256(learned_packet).hexdigest()

    uniform = []
    for level in range(depth):
        owner = paths[:, level].clone()
        opened = torch.zeros(len(parents), dtype=torch.bool)
        opened[0] = True
        for ancestor_level in range(level):
            opened[paths[:, ancestor_level].unique()] = True
        _, result = encode(owner, opened, medians)
        result.update(depth=level + 1, delta_bytes=result["bytes"] - learned["bytes"])
        uniform.append(result)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "motion_packet": str(args.motion_packet.resolve()),
        "frame": frame,
        "semantics": (
            "Total quantized point motion, survivors and births are fixed. Uniform-depth "
            "regional representatives are coordinate medians, not optimal byte lower bounds."
        ),
        "learned_frontier": learned,
        "uniform_depth_coordinate_median": uniform,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
