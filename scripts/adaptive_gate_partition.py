"""Conservative, cue-driven gate grouping for sequential S²GS frames.

The base S²GS root index remains the first grouping level.  Only a mixed root
can be split, and only when turning cue-inactive child regions off is predicted
to save more offset bits than the topology and additional gate groups cost.
Child cells are deterministic from the Gaussian positions and the S²GS root
voxel size, so the payload only needs to describe the split tree and terminal
region states.
"""

import math
import zlib

import numpy as np
import torch

OFF = 0
UPDATE = 1
SPLIT = 2


def _as_root_ids(root_indices):
    root_indices = torch.as_tensor(root_indices)
    if root_indices.ndim == 1:
        return root_indices.long()
    if root_indices.ndim == 2 and root_indices.shape[1] == 1:
        return root_indices[:, 0].long()
    raise ValueError("root_indices must contain one scalar root ID per Gaussian")


def _as_voxel_size(value):
    scalar = torch.as_tensor(value)
    if scalar.numel() != 1:
        raise ValueError("voxel_size must be a scalar")
    scalar = float(scalar.detach().cpu().item())
    if not math.isfinite(scalar) or scalar <= 0:
        raise ValueError("voxel_size must be finite and positive")
    return scalar


def build_adaptive_gate_partition(
        root_indices, anchors, evidence, voxel_size, init_pos=None,
        max_levels=4, min_points=2, offset_bits_per_point=96):
    """Build a terminal point-to-gate map and an OFF/UPDATE/SPLIT tree.

    Args:
        root_indices: S²GS root ID for each Gaussian, shape [N] or [N, 1].
        anchors: decoded Gaussian anchor positions, shape [N, 3].
        evidence: Boolean dynamic cue per Gaussian, shape [N].
        voxel_size: Side length of an S²GS root cell.
        init_pos: S²GS grid origin. Defaults to zero for standalone use.
        max_levels: Maximum number of region levels including the root level;
            the implementation and training CLI cap this at four.
        min_points: Minimum points in a mixed region before refinement.
        offset_bits_per_point: Conservative estimated offset payload avoided
            when one Gaussian is assigned to an OFF terminal group.

    Returns a dictionary containing a contiguous point-to-terminal-group map,
    terminal metadata, explicit hierarchy nodes, and conservative rate
    estimates. Group indices are ordered deterministically by sorted root and
    child coordinates.
    """
    roots = _as_root_ids(root_indices)
    anchors = torch.as_tensor(anchors, device=roots.device)
    evidence = torch.as_tensor(evidence, device=roots.device).flatten().bool()
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError("anchors must have shape [N, 3]")
    if anchors.shape[0] != roots.numel() or evidence.numel() != roots.numel():
        raise ValueError("root IDs, anchors, and evidence must have matching N")
    if not torch.isfinite(anchors).all():
        raise ValueError("anchors must be finite")
    voxel_size = _as_voxel_size(voxel_size)
    if not 1 <= max_levels <= 4:
        raise ValueError("max_levels must be between one and four")
    if min_points < 2:
        raise ValueError("min_points must be at least two")
    if offset_bits_per_point < 0:
        raise ValueError("offset_bits_per_point cannot be negative")

    point_count = roots.numel()
    if init_pos is None:
        origin = anchors.new_zeros(3)
    else:
        origin = torch.as_tensor(init_pos, device=anchors.device,
                                 dtype=anchors.dtype).flatten()
        if origin.numel() != 3:
            raise ValueError("init_pos must contain three coordinates")

    # Do a single device-to-host transfer and keep the sparse tree walk on CPU.
    # Most S²GS roots contain one or two points; per-root torch.nonzero calls
    # would launch tens of thousands of tiny GPU kernels each frame.
    roots_cpu = roots.detach().cpu().numpy()
    anchors_cpu = anchors.detach().cpu().numpy()
    evidence_cpu = evidence.detach().cpu().numpy()
    origin_cpu = origin.detach().cpu().numpy()
    root_values_np, root_inverse_np, root_counts_np = np.unique(
        roots_cpu, return_inverse=True, return_counts=True)
    root_count = int(root_values_np.size)
    root_id_bits = max(1, math.ceil(math.log2(max(2, root_count))))
    # Compact codes carry split-parent IDs and branch/state symbols. Child
    # coordinates are deterministic from the prior-frame anchors and grid
    # rule, so the verbose debug payload coordinates are not charged here.
    branch_bits_per_level = 3  # Octant within an eight-child 3D split.

    # Mutable records are internal only. Their IDs are assigned in traversal
    # order, which is stable because both roots and child cells are sorted.
    nodes = []
    terminal_nodes = []
    point_group_ids_cpu = np.full(point_count, -1, dtype=np.int64)
    predicted_saved_bits = 0
    topology_bits = 0
    group_gate_overhead_bits = 0
    split_paths = []
    root_cue_counts_np = np.bincount(
        root_inverse_np, weights=evidence_cpu.astype(np.int64),
        minlength=root_count).astype(np.int64)
    mixed_roots = int(np.count_nonzero(
        (root_cue_counts_np > 0) & (root_cue_counts_np < root_counts_np)))
    mixed_roots_eligible = int(np.count_nonzero(
        (root_cue_counts_np > 0) & (root_cue_counts_np < root_counts_np)
        & (root_counts_np >= min_points)))

    root_order = np.argsort(root_inverse_np, kind="stable")
    root_offsets = np.concatenate((
        np.zeros(1, dtype=np.int64), np.cumsum(root_counts_np, dtype=np.int64)))

    def add_node(root_id, level, indices, parent_id, cell, root_group_id,
                 evidence_count):
        node_id = len(nodes)
        record = {
            "node_id": node_id,
            "root_id": int(root_id),
            "root_group_id": int(root_group_id),
            "level": int(level),
            "parent_id": parent_id,
            "cell": None if cell is None else tuple(int(v) for v in cell),
            "point_count": int(indices.size),
            "evidence_count": int(evidence_count),
            "state": OFF,
            "predicted_saved_bits": 0,
            "topology_bits": 0,
            "group_gate_overhead_bits": 0,
            "group_id": None,
        }
        nodes.append(record)
        return node_id

    def visit(root_id, root_group_id, indices, level, parent_id, cell, path):
        nonlocal predicted_saved_bits, topology_bits
        nonlocal group_gate_overhead_bits
        cue_count = int(np.count_nonzero(evidence_cpu[indices]))
        node_id = add_node(root_id, level, indices, parent_id, cell,
                           root_group_id, cue_count)
        mixed = 0 < cue_count < int(indices.size)

        if mixed and indices.size >= min_points and level + 1 < max_levels:
            child_size = voxel_size / (2.0 ** (level + 1))
            child_coords = np.floor(
                (anchors_cpu[indices] - origin_cpu) / child_size).astype(np.int64)
            child_groups = {}
            for point_id, coordinate in zip(indices.tolist(), child_coords.tolist()):
                child_groups.setdefault(tuple(coordinate), []).append(point_id)
            child_cells = sorted(child_groups)
            child_count = len(child_cells)
            if child_count > 1:
                inactive_child_points = 0
                for child_cell in child_cells:
                    child_indices = np.asarray(child_groups[child_cell],
                                               dtype=np.int64)
                    if not np.any(evidence_cpu[child_indices]):
                        inactive_child_points += int(child_indices.size)

                saved_bits = inactive_child_points * offset_bits_per_point
                # Compact topology only carries the split-parent ID/path and
                # its SPLIT marker.  OFF/UPDATE child states are already
                # represented by S²GS Gate entropy and are not charged again.
                parent_id_bits = root_id_bits + branch_bits_per_level * level
                split_topology_bits = parent_id_bits + 2
                # Splitting one group into K terminal gates can add at most
                # one bit of binary entropy per new group.  This is a
                # conservative rate guard, separate from transmitted split
                # topology and already-counted terminal state entropy.
                extra_gate_bits = max(0, child_count - 1)
                rate_overhead = split_topology_bits + extra_gate_bits
                if saved_bits > rate_overhead:
                    nodes[node_id]["state"] = SPLIT
                    nodes[node_id]["predicted_saved_bits"] = saved_bits
                    nodes[node_id]["topology_bits"] = split_topology_bits
                    nodes[node_id]["group_gate_overhead_bits"] = extra_gate_bits
                    nodes[node_id]["path"] = tuple(path)
                    predicted_saved_bits += saved_bits
                    topology_bits += split_topology_bits
                    group_gate_overhead_bits += extra_gate_bits
                    split_paths.append((int(root_id), tuple(path)))
                    for child_id, child_cell in enumerate(child_cells):
                        child_indices = np.asarray(child_groups[child_cell],
                                                   dtype=np.int64)
                        visit(root_id, root_group_id, child_indices,
                              level + 1, node_id, child_cell,
                              path + (child_id,))
                    return

        terminal_id = len(terminal_nodes)
        initial_state = UPDATE if cue_count > 0 else OFF
        nodes[node_id]["state"] = initial_state
        nodes[node_id]["group_id"] = terminal_id
        nodes[node_id]["path"] = tuple(path)
        point_group_ids_cpu[indices] = terminal_id
        terminal_nodes.append({
            "group_id": terminal_id,
            "node_id": node_id,
            "root_id": int(root_id),
            "root_group_id": int(root_group_id),
            "level": int(level),
            "cell": None if cell is None else tuple(int(v) for v in cell),
            "path": tuple(path),
            "point_count": int(indices.size),
            "evidence_count": cue_count,
            "initial_state": initial_state,
        })

    for root_group_id, root_id in enumerate(root_values_np.tolist()):
        start = int(root_offsets[root_group_id])
        stop = int(root_offsets[root_group_id + 1])
        indices = root_order[start:stop]
        visit(root_id, root_group_id, indices, 0, None, None, ())

    if np.any(point_group_ids_cpu < 0):
        raise RuntimeError("Adaptive gate partition left Gaussians unmapped")
    terminal_count = len(terminal_nodes)
    group_sizes = np.bincount(point_group_ids_cpu, minlength=terminal_count)
    if np.any(group_sizes <= 0):
        raise RuntimeError("Adaptive gate partition produced an empty group")

    return {
        "point_group_ids": torch.as_tensor(
            point_group_ids_cpu, dtype=torch.long, device=roots.device),
        "root_values": torch.as_tensor(
            root_values_np.copy(), dtype=torch.long, device=roots.device),
        "nodes": nodes,
        "terminal_nodes": terminal_nodes,
        "root_count": root_count,
        "root_occupancy_bins": {
            "1": int(np.count_nonzero(root_counts_np == 1)),
            "2": int(np.count_nonzero(root_counts_np == 2)),
            "3": int(np.count_nonzero(root_counts_np == 3)),
            "4": int(np.count_nonzero(root_counts_np == 4)),
            "5-7": int(np.count_nonzero(
                (root_counts_np >= 5) & (root_counts_np <= 7))),
            "8+": int(np.count_nonzero(root_counts_np >= 8)),
        },
        "roots_ge_2": int(np.count_nonzero(root_counts_np >= 2)),
        "points_in_roots_ge_2": int(
            root_counts_np[root_counts_np >= 2].sum()),
        "mixed_roots": mixed_roots,
        "mixed_roots_eligible": mixed_roots_eligible,
        "terminal_group_count": terminal_count,
        "split_count": sum(node["state"] == SPLIT for node in nodes),
        "max_level": max(node["level"] for node in nodes) if nodes else 0,
        "predicted_saved_bits": int(predicted_saved_bits),
        "topology_bits": int(topology_bits),
        "group_gate_overhead_bits": int(group_gate_overhead_bits),
        "net_predicted_saved_bits": int(
            predicted_saved_bits - topology_bits - group_gate_overhead_bits),
        "max_levels": int(max_levels),
        "min_points": int(min_points),
        "offset_bits_per_point": int(offset_bits_per_point),
        "split_paths": split_paths,
    }


def refine_adaptive_gate_partition(
        partition, root_indices, anchors, evidence, residual_scores,
        active_group_mask, voxel_size, init_pos=None, target_level=0,
        min_points=2, offset_bits_per_point=96, residual_ratio=0.8,
        active_threshold=0.05):
    """Split eligible active terminal regions by one octree level.

    Refinement is intended to run after the current Gate has been fitted for a
    warm-up interval.  Offset savings are credited only beneath a deterministically
    active parent Gate, and only if cue-negative spatial children have lower
    ungated residual magnitude than cue-positive children.  New terminal groups
    inherit their parent's prior Gate parameter; the caller must use
    ``parent_group_ids`` to copy the learned parameters and optimizer state.
    """
    roots = _as_root_ids(root_indices)
    anchors = torch.as_tensor(anchors, device=roots.device)
    evidence = torch.as_tensor(evidence, device=roots.device).flatten().bool()
    residual_scores = torch.as_tensor(
        residual_scores, device=roots.device).flatten().float()
    active_group_mask = torch.as_tensor(
        active_group_mask, device=roots.device).flatten().bool()
    voxel_size = _as_voxel_size(voxel_size)
    if anchors.ndim != 2 or anchors.shape != (roots.numel(), 3):
        raise ValueError("anchors must have shape [N, 3] matching root IDs")
    if (evidence.numel() != roots.numel()
            or residual_scores.numel() != roots.numel()):
        raise ValueError("cue and residual scores must have one value per point")
    if not torch.isfinite(residual_scores).all():
        raise ValueError("residual scores must be finite")
    if not 0.0 <= residual_ratio < 1.0:
        raise ValueError("residual_ratio must be in [0, 1)")
    if min_points < 2 or offset_bits_per_point < 0:
        raise ValueError("invalid minimum point count or offset bit cost")
    if not 0.0 <= active_threshold <= 1.0:
        raise ValueError("active_threshold must be in [0, 1]")

    point_group_ids = partition["point_group_ids"]
    old_group_count = int(partition["terminal_group_count"])
    if point_group_ids.numel() != roots.numel():
        raise ValueError("partition map no longer matches the Gaussian count")
    if active_group_mask.numel() != old_group_count:
        raise ValueError("active mask must have one value per current Gate group")

    if init_pos is None:
        origin = anchors.new_zeros(3)
    else:
        origin = torch.as_tensor(init_pos, device=anchors.device,
                                 dtype=anchors.dtype).flatten()
        if origin.numel() != 3:
            raise ValueError("init_pos must contain three coordinates")

    # Refine sparse candidate regions on CPU after one transfer, as in the
    # initial builder; avoid per-region CUDA scans/synchronization.
    roots_cpu = roots.detach().cpu().numpy()
    anchors_cpu = anchors.detach().cpu().numpy()
    evidence_cpu = evidence.detach().cpu().numpy()
    residual_cpu = residual_scores.detach().cpu().numpy()
    origin_cpu = origin.detach().cpu().numpy()
    group_ids_cpu = point_group_ids.detach().cpu().numpy()
    active_cpu = active_group_mask.detach().cpu().numpy()
    root_count = int(partition["root_count"])
    root_id_bits = max(1, math.ceil(math.log2(max(2, root_count))))

    group_counts = np.bincount(group_ids_cpu, minlength=old_group_count)
    group_order = np.argsort(group_ids_cpu, kind="stable")
    group_offsets = np.concatenate((
        np.zeros(1, dtype=np.int64),
        np.cumsum(group_counts, dtype=np.int64)))
    terminals_by_id = sorted(
        partition["terminal_nodes"], key=lambda item: int(item["group_id"]))
    if len(terminals_by_id) != old_group_count:
        raise RuntimeError("terminal metadata does not match Gate group count")
    nodes_by_id = {int(node["node_id"]): node
                   for node in partition["nodes"]}
    children_per_parent = {}
    for node in partition["nodes"]:
        parent_id = node["parent_id"]
        if parent_id is not None:
            parent_id = int(parent_id)
            children_per_parent[parent_id] = (
                children_per_parent.get(parent_id, 0) + 1)

    def path_bits_to_terminal(terminal):
        """Charge each ancestor child ordinal at its actual alphabet width."""
        bits = 0
        node = nodes_by_id[int(terminal["node_id"])]
        while node["parent_id"] is not None:
            parent_id = int(node["parent_id"])
            parent = nodes_by_id[parent_id]
            child_count = children_per_parent.get(parent_id, 0)
            if parent["state"] != SPLIT or child_count < 2:
                raise RuntimeError("Terminal path has an invalid split parent")
            bits += max(1, math.ceil(math.log2(child_count)))
            node = parent
        return bits

    proposals = {}
    projected_group_count = old_group_count
    stats = {
        "target_level": int(target_level),
        "eligible_groups": 0,
        "inactive_parent_groups": 0,
        "cue_mixed_groups": 0,
        "spatial_groups": 0,
        "oversized_spatial_groups": 0,
        "residual_heterogeneous_groups": 0,
        "rate_accepted_groups": 0,
        "active_parent_points": 0,
        "static_child_points": 0,
        "predicted_saved_bits": 0,
        "topology_bits": 0,
        "group_gate_overhead_bits": 0,
    }

    for terminal in terminals_by_id:
        old_group_id = int(terminal["group_id"])
        if int(terminal["level"]) != int(target_level):
            continue
        stats["eligible_groups"] += 1
        start = int(group_offsets[old_group_id])
        stop = int(group_offsets[old_group_id + 1])
        indices = group_order[start:stop]
        if indices.size < min_points:
            continue
        if not active_cpu[old_group_id]:
            stats["inactive_parent_groups"] += 1
            continue

        cue_count = int(np.count_nonzero(evidence_cpu[indices]))
        if not 0 < cue_count < int(indices.size):
            continue
        stats["cue_mixed_groups"] += 1

        level = int(terminal["level"])
        child_size = voxel_size / (2.0 ** (level + 1))
        child_coords = np.floor(
            (anchors_cpu[indices] - origin_cpu) / child_size).astype(np.int64)
        child_groups = {}
        for point_id, coordinate in zip(indices.tolist(), child_coords.tolist()):
            child_groups.setdefault(tuple(coordinate), []).append(point_id)
        child_cells = sorted(child_groups)
        if len(child_cells) < 2:
            continue
        if len(child_cells) > 8:
            stats["oversized_spatial_groups"] += 1

        static_parts = []
        dynamic_parts = []
        inactive_child_points = 0
        for child_cell in child_cells:
            child_indices = np.asarray(child_groups[child_cell], dtype=np.int64)
            child_evidence = evidence_cpu[child_indices]
            if not np.any(child_evidence):
                static_parts.append(child_indices)
                inactive_child_points += int(child_indices.size)
            else:
                # A mixed child is not predicted OFF and does not contribute
                # to either side of the residual heterogeneity comparison.
                dynamic_parts.append(child_indices[child_evidence])
        if inactive_child_points <= 0 or not dynamic_parts:
            continue
        stats["spatial_groups"] += 1

        static_indices = np.concatenate(static_parts) if static_parts else np.empty(0, dtype=np.int64)
        dynamic_indices = np.concatenate(dynamic_parts) if dynamic_parts else np.empty(0, dtype=np.int64)
        if static_indices.size == 0 or dynamic_indices.size == 0:
            continue
        static_residual = float(np.mean(residual_cpu[static_indices]))
        dynamic_residual = float(np.mean(residual_cpu[dynamic_indices]))
        if not (dynamic_residual > 0.0
                and static_residual <= residual_ratio * dynamic_residual):
            continue
        stats["residual_heterogeneous_groups"] += 1

        child_count = len(child_cells)
        saved_bits = inactive_child_points * int(offset_bits_per_point)
        parent_id_bits = root_id_bits + path_bits_to_terminal(terminal)
        split_topology_bits = parent_id_bits + 2
        added_groups = max(0, child_count - 1)
        candidate_group_count = projected_group_count + added_groups
        # Gate.size uses zero-order binary-state entropy over all terminal
        # groups. A rare state among ~38k groups can cost ~log2(N) bits, so one
        # bit per added group is not conservative. Budget each new group at
        # ceil(log2(projected total groups)).
        extra_gate_bits = added_groups * max(
            1, math.ceil(math.log2(max(2, candidate_group_count))))
        if saved_bits <= split_topology_bits + extra_gate_bits:
            continue
        stats["rate_accepted_groups"] += 1
        stats["active_parent_points"] += int(indices.size)
        stats["static_child_points"] += inactive_child_points
        stats["predicted_saved_bits"] += saved_bits
        stats["topology_bits"] += split_topology_bits
        stats["group_gate_overhead_bits"] += extra_gate_bits
        projected_group_count = candidate_group_count
        proposals[old_group_id] = {
            "terminal": terminal,
            "indices": indices,
            "child_cells": child_cells,
            "child_groups": child_groups,
            "cue_count": cue_count,
            "saved_bits": saved_bits,
            "topology_bits": split_topology_bits,
            "group_overhead_bits": extra_gate_bits,
            "static_residual": static_residual,
            "dynamic_residual": dynamic_residual,
        }

    # No mutation is required when a stage found no worthwhile split.
    if not proposals:
        partition.setdefault("refinement_stats", []).append(stats)
        return partition, torch.arange(
            old_group_count, dtype=torch.long, device=roots.device), stats

    old_nodes = partition["nodes"]
    for old_group_id, proposal in proposals.items():
        terminal = proposal["terminal"]
        node_id = int(terminal["node_id"])
        parent_node = old_nodes[node_id]
        parent_node["state"] = SPLIT
        parent_node["group_id"] = None
        parent_node["predicted_saved_bits"] = int(proposal["saved_bits"])
        parent_node["topology_bits"] = int(proposal["topology_bits"])
        parent_node["group_gate_overhead_bits"] = int(
            proposal["group_overhead_bits"])
        parent_node["static_residual_mean"] = proposal["static_residual"]
        parent_node["dynamic_residual_mean"] = proposal["dynamic_residual"]
        parent_node["path"] = tuple(terminal["path"])
        split_path = (int(terminal["root_id"]), tuple(terminal["path"]))
        if split_path not in partition["split_paths"]:
            partition["split_paths"].append(split_path)

    # Preserve depth-first terminal order by replacing each split parent in
    # place with its sorted children.  Group IDs are then compacted and the
    # point map is rebuilt, so split-path reconstruction stays deterministic.
    new_terminals = []
    parent_group_ids = []
    point_group_ids_cpu = np.full(roots.numel(), -1, dtype=np.int64)
    for terminal in terminals_by_id:
        old_group_id = int(terminal["group_id"])
        proposal = proposals.get(old_group_id)
        if proposal is None:
            new_group_id = len(new_terminals)
            updated = dict(terminal)
            updated["group_id"] = new_group_id
            old_nodes[int(updated["node_id"])]["group_id"] = new_group_id
            new_terminals.append(updated)
            parent_group_ids.append(old_group_id)
            point_indices = group_order[
                int(group_offsets[old_group_id]):int(group_offsets[old_group_id + 1])]
            point_group_ids_cpu[point_indices] = new_group_id
            continue

        parent_node_id = int(terminal["node_id"])
        root_group_id = int(terminal["root_group_id"])
        root_id = int(terminal["root_id"])
        level = int(terminal["level"])
        parent_path = tuple(terminal["path"])
        for child_id, child_cell in enumerate(proposal["child_cells"]):
            child_indices = np.asarray(
                proposal["child_groups"][child_cell], dtype=np.int64)
            child_group_id = len(new_terminals)
            child_path = parent_path + (child_id,)
            cue_count = int(np.count_nonzero(evidence_cpu[child_indices]))
            child_node_id = len(old_nodes)
            child_record = {
                "node_id": child_node_id,
                "root_id": root_id,
                "root_group_id": root_group_id,
                "level": level + 1,
                "parent_id": parent_node_id,
                "cell": tuple(int(v) for v in child_cell),
                "point_count": int(child_indices.size),
                "evidence_count": cue_count,
                "state": UPDATE if cue_count > 0 else OFF,
                "predicted_saved_bits": 0,
                "topology_bits": 0,
                "group_gate_overhead_bits": 0,
                "group_id": child_group_id,
                "path": child_path,
            }
            old_nodes.append(child_record)
            new_terminals.append({
                "group_id": child_group_id,
                "node_id": child_node_id,
                "root_id": root_id,
                "root_group_id": root_group_id,
                "level": level + 1,
                "cell": tuple(int(v) for v in child_cell),
                "path": child_path,
                "point_count": int(child_indices.size),
                "evidence_count": cue_count,
                "initial_state": UPDATE if cue_count > 0 else OFF,
            })
            parent_group_ids.append(old_group_id)
            point_group_ids_cpu[child_indices] = child_group_id

    if np.any(point_group_ids_cpu < 0):
        raise RuntimeError("Refinement left Gaussians without a Gate group")
    if len(parent_group_ids) != len(new_terminals):
        raise RuntimeError("Refinement parent-Gate map is inconsistent")

    partition["terminal_nodes"] = new_terminals
    partition["point_group_ids"] = torch.as_tensor(
        point_group_ids_cpu, dtype=torch.long, device=roots.device)
    partition["terminal_group_count"] = len(new_terminals)
    partition["split_count"] = sum(node["state"] == SPLIT for node in old_nodes)
    partition["max_level"] = max(
        (int(node["level"]) for node in new_terminals), default=0)
    partition["predicted_saved_bits"] += int(stats["predicted_saved_bits"])
    partition["topology_bits"] += int(stats["topology_bits"])
    partition["group_gate_overhead_bits"] += int(
        stats["group_gate_overhead_bits"])
    partition["net_predicted_saved_bits"] = int(
        partition["predicted_saved_bits"] - partition["topology_bits"]
        - partition["group_gate_overhead_bits"])
    partition["refinement_stats"] = partition.get("refinement_stats", []) + [stats]
    # Stage-time parent IDs are also useful for parameter/state inheritance.
    parent_group_ids = torch.as_tensor(
        parent_group_ids, dtype=torch.long, device=roots.device)
    partition["last_parent_group_ids"] = parent_group_ids
    return partition, parent_group_ids, stats


def inherit_gate_parameters(old_gate, new_gate, parent_group_ids):
    """Copy learned Gate logits and Adam state from parent groups to children."""
    parent_group_ids = torch.as_tensor(
        parent_group_ids, dtype=torch.long,
        device=old_gate.log_alphas.device).flatten()
    if parent_group_ids.numel() != new_gate.log_alphas.numel():
        raise ValueError("one old parent group is required per new Gate group")
    if parent_group_ids.numel() and (
            int(parent_group_ids.min()) < 0
            or int(parent_group_ids.max()) >= old_gate.log_alphas.numel()):
        raise ValueError("parent group index is outside the old Gate")

    with torch.no_grad():
        new_gate.log_alphas.copy_(old_gate.log_alphas.index_select(
            0, parent_group_ids))
        old_group_gate = old_gate.sample_gate_Gumbel_Softmax(stochastic=False)
        new_group_gate = old_group_gate.index_select(0, parent_group_ids)
        inverse_indices = new_gate.inverse_indices
        if inverse_indices is not None:
            new_gate.gate = new_group_gate.index_select(0, inverse_indices)
        else:
            new_gate.gate = new_group_gate
        new_gate.iter_counter = old_gate.iter_counter

    old_state = old_gate.optimizer.state.get(old_gate.log_alphas)
    if old_state:
        copied_state = {}
        for key, value in old_state.items():
            if torch.is_tensor(value) and value.shape == old_gate.log_alphas.shape:
                copied_state[key] = value.index_select(0, parent_group_ids).clone()
            elif torch.is_tensor(value):
                copied_state[key] = value.clone()
            else:
                copied_state[key] = value
        new_gate.optimizer.state[new_gate.log_alphas] = copied_state
    return new_gate


def reconstruct_gate_groups(root_indices, anchors, voxel_size, init_pos=None,
                            split_paths=(), max_levels=4):
    """Recreate terminal gate IDs using only roots, anchors, and split paths."""
    roots = _as_root_ids(root_indices)
    anchors = torch.as_tensor(anchors, device=roots.device)
    if anchors.ndim != 2 or anchors.shape != (roots.numel(), 3):
        raise ValueError("anchors must have shape [N, 3] matching root IDs")
    voxel_size = _as_voxel_size(voxel_size)
    if not 1 <= max_levels <= 4:
        raise ValueError("max_levels must be between one and four")
    if init_pos is None:
        origin = anchors.new_zeros(3)
    else:
        origin = torch.as_tensor(init_pos, device=anchors.device,
                                 dtype=anchors.dtype).flatten()
        if origin.numel() != 3:
            raise ValueError("init_pos must contain three coordinates")
    roots_cpu = roots.detach().cpu().numpy()
    anchors_cpu = anchors.detach().cpu().numpy()
    origin_cpu = origin.detach().cpu().numpy()
    root_values_np, root_inverse_np, root_counts_np = np.unique(
        roots_cpu, return_inverse=True, return_counts=True)
    root_order = np.argsort(root_inverse_np, kind="stable")
    root_offsets = np.concatenate((
        np.zeros(1, dtype=np.int64), np.cumsum(root_counts_np, dtype=np.int64)))
    split_key_set = {(int(root), tuple(path))
                     for root, path in split_paths}
    consumed_split_keys = set()
    point_group_ids_cpu = np.full(roots.numel(), -1, dtype=np.int64)
    terminal_group = 0

    def visit(root_id, indices, level, path):
        nonlocal terminal_group
        if (int(root_id), tuple(path)) in split_key_set:
            consumed_split_keys.add((int(root_id), tuple(path)))
            if level + 1 >= max_levels:
                raise ValueError("Split path exceeds configured maximum depth")
            child_size = voxel_size / (2.0 ** (level + 1))
            child_coords = np.floor(
                (anchors_cpu[indices] - origin_cpu) / child_size).astype(np.int64)
            child_groups = {}
            for point_id, coordinate in zip(indices.tolist(), child_coords.tolist()):
                child_groups.setdefault(tuple(coordinate), []).append(point_id)
            child_cells = sorted(child_groups)
            if len(child_cells) < 2:
                raise ValueError("Split path does not produce spatial children")
            for child_id, child_cell in enumerate(child_cells):
                child_indices = np.asarray(child_groups[child_cell],
                                           dtype=np.int64)
                visit(root_id, child_indices, level + 1,
                      path + (child_id,))
            return
        point_group_ids_cpu[indices] = terminal_group
        terminal_group += 1

    for root_group_id, root_id in enumerate(root_values_np.tolist()):
        start = int(root_offsets[root_group_id])
        stop = int(root_offsets[root_group_id + 1])
        indices = root_order[start:stop]
        visit(root_id, indices, 0, ())
    if consumed_split_keys != split_key_set:
        raise ValueError("Split paths reference nonexistent regions")
    if np.any(point_group_ids_cpu < 0):
        raise RuntimeError("Reconstruction left Gaussians unmapped")
    return torch.as_tensor(
        point_group_ids_cpu, dtype=torch.long, device=roots.device)


def _pack_gate_levels(group_gates):
    """Pack S²GS Gate values (rounded to tenths) as two nibbles per byte."""
    values = torch.as_tensor(group_gates).detach().cpu().flatten().float()
    if not torch.isfinite(values).all():
        raise ValueError("Gate values must be finite")
    levels = torch.round(values * 10).to(torch.int64)
    if torch.any((levels < 0) | (levels > 10)):
        raise ValueError("Gate values must lie in [0, 1]")
    if not torch.allclose(values, levels.float() / 10.0, atol=1e-6, rtol=0):
        raise ValueError("Gate values must use S²GS one-decimal quantization")
    packed = bytearray((levels.numel() + 1) // 2)
    for index, level in enumerate(levels.tolist()):
        byte_index = index // 2
        if index % 2 == 0:
            packed[byte_index] = int(level)
        else:
            packed[byte_index] |= int(level) << 4
    return bytes(packed), int(levels.numel())


def _unpack_gate_levels(data, count, device="cpu"):
    """Decode packed 0..10 Gate levels into the exact quantized float values."""
    count = int(count)
    if count < 0 or len(data) != (count + 1) // 2:
        raise ValueError("Packed Gate-level byte count is inconsistent")
    levels = []
    for index in range(count):
        packed = data[index // 2]
        levels.append((packed & 0x0F) if index % 2 == 0 else (packed >> 4))
    if any(level > 10 for level in levels):
        raise ValueError("Packed Gate level exceeds ten")
    # Construct from Python decimal literals so CUDA division does not turn
    # the 0.9 level into the adjacent float32 value (one ULP higher than the
    # value produced by StraightThroughFloat's decimal rounding).
    level_values = [level / 10.0 for level in levels]
    return torch.tensor(level_values, dtype=torch.float32, device=device)


def encode_sqres_reopen_mask(reopen_mask):
    """Bit-pack a V7d sq_res reopen mask, choosing raw or zlib per frame."""
    mask = torch.as_tensor(reopen_mask).detach().flatten().bool().cpu().numpy()
    point_count = int(mask.size)
    if point_count <= 0:
        raise ValueError("sq_res reopen mask must contain at least one point")
    popcount = int(np.count_nonzero(mask))
    raw = np.packbits(mask.astype(np.uint8), bitorder="little").tobytes()
    if popcount == 0:
        encoding, blob = "empty", b""
    else:
        compressed = zlib.compress(raw)
        if len(compressed) < len(raw):
            encoding, blob = "bitset_zlib_lsb0_v1", compressed
        else:
            encoding, blob = "bitset_raw_lsb0_v1", raw
    return {
        "encoding": encoding,
        "point_count": point_count,
        "popcount": popcount,
        "raw_bytes": len(raw),
        "blob": blob,
    }


def decode_sqres_reopen_mask(payload, expected_point_count=None, device=None):
    """Decode and strictly validate a V7d packed point-order bitset."""
    if not isinstance(payload, dict):
        raise ValueError("sq_res reopen mask payload must be a dictionary")
    point_count = int(payload.get("point_count", -1))
    popcount = int(payload.get("popcount", -1))
    if point_count <= 0 or not 0 <= popcount <= point_count:
        raise ValueError("Invalid sq_res reopen mask counts")
    if (expected_point_count is not None
            and point_count != int(expected_point_count)):
        raise ValueError("sq_res reopen mask point count mismatch")
    raw_bytes = (point_count + 7) // 8
    if int(payload.get("raw_bytes", -1)) != raw_bytes:
        raise ValueError("sq_res reopen mask raw byte count mismatch")
    blob = payload.get("blob")
    if not isinstance(blob, bytes):
        raise ValueError("sq_res reopen mask blob must be bytes")
    encoding = payload.get("encoding")
    if encoding == "empty":
        if popcount != 0 or blob:
            raise ValueError("Invalid empty sq_res reopen mask")
        raw = bytes(raw_bytes)
    elif encoding == "bitset_raw_lsb0_v1":
        if len(blob) != raw_bytes:
            raise ValueError("Truncated or oversized raw sq_res reopen mask")
        raw = blob
    elif encoding == "bitset_zlib_lsb0_v1":
        decoder = zlib.decompressobj()
        try:
            # Cap expansion before flushing so a malformed sidecar cannot
            # inflate arbitrarily large data before its declared N is checked.
            raw = decoder.decompress(blob, raw_bytes + 1)
            if len(raw) > raw_bytes or decoder.unconsumed_tail:
                raise ValueError("Compressed sq_res reopen mask exceeds N")
            raw += decoder.flush(raw_bytes + 1)
        except zlib.error as exc:
            raise ValueError("Invalid compressed sq_res reopen mask") from exc
        if (not decoder.eof or decoder.unused_data
                or decoder.unconsumed_tail or len(raw) != raw_bytes):
            raise ValueError("Invalid compressed sq_res reopen mask stream")
    else:
        raise ValueError(f"Unknown sq_res reopen mask encoding: {encoding!r}")

    remainder = point_count % 8
    if remainder and raw[-1] & (0xFF ^ ((1 << remainder) - 1)):
        raise ValueError("Non-zero padding bits in sq_res reopen mask")
    bits = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:point_count]
    if int(np.count_nonzero(bits)) != popcount:
        raise ValueError("sq_res reopen mask popcount mismatch")
    return torch.as_tensor(bits.copy(), device=device, dtype=torch.bool)


def _pack_split_paths(partition):
    """Pack split parents as root ordinals and byte-sized child ordinals.

    Root IDs themselves can be large sparse integers.  Their ordinal in the
    sorted unique root list is sufficient and deterministic. A split path
    stores one uint16 child ordinal per level; this remains compact and
    supports more than eight occupied cells when anchors have drifted.
    """
    root_values = torch.as_tensor(
        partition["root_values"]).detach().cpu().flatten().tolist()
    root_to_ordinal = {int(root): index
                       for index, root in enumerate(root_values)}
    root_width = 2 if len(root_values) <= 0xFFFF else 4
    packed = bytearray()
    paths = partition["split_paths"]
    if len(root_values) > 0xFFFFFFFF:
        raise ValueError("Too many roots for compact split-path encoding")
    for root_id, path in paths:
        root_ordinal = root_to_ordinal.get(int(root_id))
        if root_ordinal is None:
            raise ValueError(f"Split path references unknown root {root_id}")
        path = tuple(int(child) for child in path)
        if len(path) > 255 or any(child < 0 or child > 0xFFFF for child in path):
            raise ValueError("Split path exceeds compact child ordinal limits")
        packed.extend(root_ordinal.to_bytes(root_width, "little"))
        packed.append(len(path))
        for child in path:
            packed.extend(child.to_bytes(2, "little"))
    return bytes(packed), root_width, len(paths)


def _unpack_split_paths(data, root_values, root_width, split_count):
    root_values = torch.as_tensor(root_values).detach().cpu().flatten().tolist()
    root_width = int(root_width)
    split_count = int(split_count)
    if root_width not in (2, 4) or split_count < 0:
        raise ValueError("Invalid compact split-path header")
    paths = []
    cursor = 0
    for _ in range(split_count):
        if cursor + root_width + 1 > len(data):
            raise ValueError("Truncated compact split-path stream")
        root_ordinal = int.from_bytes(data[cursor:cursor + root_width], "little")
        cursor += root_width
        if root_ordinal >= len(root_values):
            raise ValueError("Compact split path has an invalid root ordinal")
        path_length = data[cursor]
        cursor += 1
        path_end = cursor + path_length * 2
        if path_end > len(data):
            raise ValueError("Truncated compact split path")
        path = tuple(int.from_bytes(data[offset:offset + 2], "little")
                     for offset in range(cursor, path_end, 2))
        cursor = path_end
        paths.append((int(root_values[root_ordinal]), path))
    if cursor != len(data):
        raise ValueError("Compact split-path stream has trailing bytes")
    return paths


def build_compact_gate_payload(partition, group_gates, frame_idx,
                               init_pos, voxel_size, active_threshold=0.05,
                               sqres_reopen_mask_payload=None):
    """Create a compact V7b sidecar, optionally with the V7d sq_res mask.

    Point-to-terminal-group IDs and verbose per-node records are omitted:
    they can be reconstructed from the previous-frame root IDs/anchors and
    the split-path stream. Gate values are stored as exact S²GS tenths, not
    only binary OFF/UPDATE states. The optional V7d field only codes the
    attribute decoder's cue-derived reopen mask; offset Gate semantics stay
    identical to the V7b compact stream.
    """
    group_gates = torch.as_tensor(group_gates).detach().flatten()
    if group_gates.numel() != int(partition["terminal_group_count"]):
        raise ValueError("Gate count does not match terminal groups")
    root_values = torch.as_tensor(
        partition["root_values"]).detach().cpu().flatten()
    split_blob, root_width, split_count = _pack_split_paths(partition)
    gate_blob, gate_count = _pack_gate_levels(group_gates)
    origin = torch.as_tensor(init_pos).detach().cpu().flatten()
    if origin.numel() != 3:
        raise ValueError("Grid origin must contain three coordinates")
    voxel_size = _as_voxel_size(voxel_size)
    payload = {
        "format": "adaptive_gate_partition_v7b_compact_v1",
        "payload_kind": "compact_decoder_sidecar",
        "frame_idx": int(frame_idx),
        "coordinate_rule": "previous_frame_decoded_anchor_octree",
        "topology_anchor_source": "frozen_previous_frame_decoded_anchors",
        "root_cell_side": voxel_size,
        "grid_origin": tuple(float(value) for value in origin.tolist()),
        "root_count": int(root_values.numel()),
        "terminal_group_count": int(gate_count),
        "split_count": int(split_count),
        "max_levels": int(partition["max_levels"]),
        "active_threshold": float(active_threshold),
        "state_codes": {"OFF": OFF, "UPDATE": UPDATE, "SPLIT": SPLIT},
        "terminal_state_rule": "gate_level_0_is_OFF_else_UPDATE",
        "split_path_encoding": "root_ordinal_u16_or_u32_depth_u8_child_u16_v1",
        "split_root_ordinal_bytes": int(root_width),
        "split_paths": split_blob,
        "gate_level_encoding": "two_uint4_levels_per_byte_tenths_v1",
        "gate_level_count": int(gate_count),
        "gate_levels": gate_blob,
    }
    if sqres_reopen_mask_payload is not None:
        expected_points = int(partition["point_group_ids"].numel())
        decode_sqres_reopen_mask(
            sqres_reopen_mask_payload,
            expected_point_count=expected_points)
        payload["format"] = "adaptive_gate_partition_v7d_compact_v1"
        payload["sq_res_reopen_mask"] = dict(sqres_reopen_mask_payload)
    return payload


def decode_compact_gate_payload(payload, root_indices, anchors):
    """Reconstruct exact group IDs and per-point quantized Gate values."""
    roots = _as_root_ids(root_indices)
    root_values = torch.unique(roots, sorted=True)
    if int(root_values.numel()) != int(payload["root_count"]):
        raise ValueError("Compact sidecar root count does not match source model")
    split_paths = _unpack_split_paths(
        payload["split_paths"], root_values,
        payload["split_root_ordinal_bytes"], payload["split_count"])
    point_group_ids = reconstruct_gate_groups(
        roots, anchors, payload["root_cell_side"],
        init_pos=payload["grid_origin"], split_paths=split_paths,
        max_levels=payload["max_levels"])
    if (point_group_ids.numel() == 0
            or int(point_group_ids.max().item()) + 1
            != int(payload["terminal_group_count"])):
        raise ValueError("Reconstructed group count does not match sidecar")
    group_levels = _unpack_gate_levels(
        payload["gate_levels"], payload["gate_level_count"],
        device=point_group_ids.device)
    if group_levels.numel() != int(payload["terminal_group_count"]):
        raise ValueError("Packed Gate count does not match terminal groups")
    point_gates = group_levels.index_select(0, point_group_ids)
    return point_group_ids, group_levels, point_gates


def finalize_gate_partition(partition, group_gates, active_threshold=0.05):
    """Create a serializable per-frame OFF/UPDATE/SPLIT topology payload."""
    group_gates = torch.as_tensor(group_gates).flatten().detach().cpu()
    terminals = partition["terminal_nodes"]
    if group_gates.numel() != len(terminals):
        raise ValueError("One deterministic gate value is required per terminal group")
    active = group_gates > active_threshold
    terminal_payload = []
    for group, active_group in zip(terminals, active.tolist()):
        terminal_payload.append({
            **group,
            "state": UPDATE if active_group else OFF,
            "gate_value": float(group_gates[len(terminal_payload)]),
        })
    node_payload = []
    for node in partition["nodes"]:
        record = dict(node)
        if record["state"] != SPLIT:
            group_id = record["group_id"]
            record["state"] = UPDATE if bool(active[group_id]) else OFF
        node_payload.append(record)
    point_group_ids = partition["point_group_ids"].detach().cpu()
    return {
        "format": "adaptive_gate_partition_v1",
        "payload_kind": "verbose_debug_uncompressed",
        "state_codes": {"OFF": OFF, "UPDATE": UPDATE, "SPLIT": SPLIT},
        "max_levels": partition["max_levels"],
        "min_points": partition["min_points"],
        "root_count": partition["root_count"],
        "terminal_group_count": partition["terminal_group_count"],
        "split_count": partition["split_count"],
        "max_level": partition["max_level"],
        "predicted_saved_bits": partition["predicted_saved_bits"],
        "topology_bits": partition["topology_bits"],
        "group_gate_overhead_bits": partition[
            "group_gate_overhead_bits"],
        "net_predicted_saved_bits": partition["net_predicted_saved_bits"],
        "root_occupancy_bins": partition["root_occupancy_bins"],
        "mixed_roots": partition["mixed_roots"],
        "mixed_roots_eligible": partition["mixed_roots_eligible"],
        "active_groups": int(active.sum()),
        "active_gaussians": int(active[point_group_ids].sum()),
        "compact_topology": {
            "split_paths": partition["split_paths"],
            "terminal_states": [
                int(UPDATE if active_group else OFF)
                for active_group in active.tolist()],
        },
        "nodes": node_payload,
        "terminal_regions": terminal_payload,
        "point_group_ids": point_group_ids,
    }
