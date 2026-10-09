"""One bounded local topology batch, followed by equal-budget continuation.

Hints use fitted complete world transforms only. They never accept an edit.
The renderer-side transaction makes the sole decoded rate-distortion decision.
"""

import copy
import json
import zlib

import torch

from .binary_refinement import _binary_layout, _copy_layout
from .common_contrast import (CommonContrastMotion, _principal_rotation_vector,
                              decompose_terminal_transforms,
                              reconstruct_cumulative_transforms)
from .node_motion import rotation_delta, terminal_node_rows
from .region_rigid import rotation_vector_quaternion
from .split_region_scores import (PositionGradientEvidence, cluster_split_region,
                                  score_split_regions, select_split_regions)
from .tree_edit_codec import MAGIC, decode_node_edits, replay_node_edits
from libgs.utils.general import quaternion_multiply


def cumulative_transforms(motion, roots, contrasts):
    """Vectorized existing world-rotation/pivot-displacement semantics."""
    regions = motion._regions(roots, contrasts)
    rotations = [rotation_vector_quaternion(roots[:, 3:])]
    deltas = [roots[:, :3]]
    for level in range(1, len(regions)):
        parent = motion.parents[level - 1]
        parent_q = rotations[-1][parent]
        local = regions[level]
        rotations.append(quaternion_multiply(rotation_vector_quaternion(local[:, 3:]), parent_q))
        offset = motion.active_layout['pivots'][level] - motion.active_layout['pivots'][level - 1][parent]
        deltas.append(rotation_delta(parent_q, offset) + deltas[-1][parent] + local[:, :3])
    return rotations, deltas


def layout_edits(before, after):
    """Explicit membership/parent changes; current decoded XYZ recreates pivots."""
    records = dict(counts=after['counts'], index=[], parents=[])
    before_index = before['index'].detach().cpu()
    after_index = after['index'].detach().cpu()
    for level in range(len(after['counts'])):
        changed = torch.where(before_index[level] != after_index[level])[0]
        records['index'].extend([[level, int(row), int(after_index[level, row])]
                                 for row in changed.cpu().tolist()])
    for level, parent in enumerate(after['parents']):
        old = before['parents'][level]
        records['parents'].append(parent.cpu().tolist() if not torch.equal(parent.cpu(), old.cpu()) else None)
    return zlib.compress(json.dumps(records, separators=(',', ':')).encode(), level=9)


def apply_layout_edits(layout, payload, xyz):
    if not payload:
        return _binary_layout(layout['index'], layout['counts'], layout['parents'],
                              xyz, layout['pivots'])
    if payload.startswith(MAGIC):
        index, counts, parents = replay_node_edits(layout, decode_node_edits(payload))
        fallback = []
        for count, old in zip(counts, layout['pivots']):
            value = xyz.new_zeros(count, 3)
            value[:min(count, len(old))] = old[:min(count, len(old))].to(xyz)
            fallback.append(value)
        return _binary_layout(index, counts, parents, xyz, fallback)
    records = json.loads(zlib.decompress(payload))
    index = layout['index'].clone()
    if len(records['counts']) != len(layout['counts']) or len(records['parents']) != len(layout['parents']):
        raise ValueError('Unsupported edit depth')
    seen = set()
    for level, row, owner in records['index']:
        if (level, row) in seen or not 0 <= level < len(index) or not 0 <= row < index.shape[1]:
            raise ValueError('Invalid or duplicate member edit')
        seen.add((level, row))
        index[level, row] = owner
    parents = [old if value is None else torch.tensor(value, device=xyz.device, dtype=torch.long)
               for old, value in zip(layout['parents'], records['parents'])]
    # Empty newly appended nodes need no inherited pivots. Nonempty means are
    # exactly those used by the encoder; no image evidence is used here.
    fallback = []
    for count, old in zip(records['counts'], layout['pivots']):
        value = xyz.new_zeros(count, 3)
        value[:min(count, len(old))] = old[:min(count, len(old))].to(xyz)
        fallback.append(value)
    return _binary_layout(index, records['counts'], parents, xyz, fallback)


def gradient_split_proposals(motion, xyz, evidence):
    """Rank geometric terminal regions independently of neighboring pairs."""
    cfg, layout = motion.config, motion.active_layout
    child_minimum = cfg.direct_split_min_child_gs or cfg.min_child_gs
    counts = layout['counts']
    paths = [layout['index'].new_ones(count) for count in counts]
    owners = terminal_node_rows(layout, paths, selected_only=False)
    scores = score_split_regions(evidence, owners, region_count=sum(counts))
    legal_flags, locations = [], []
    for level, count in enumerate(counts):
        for node in range(count):
            locations.append((level, node))
            legal_flags.append(level < len(counts) - 1 and
                               not motion._cached_children[level][node])
    legal = torch.tensor(legal_flags, device=xyz.device, dtype=torch.bool)
    legal &= scores.supported_gs >= 2 * child_minimum
    if sum(counts) + 2 > cfg.direct_node_budget:
        legal.zero_()
    chosen = select_split_regions(scores, legal, top_k=cfg.direct_split_top_k,
                                  candidate_fraction=cfg.direct_split_candidate_fraction)
    if cfg.direct_max_trial_splits is not None:
        chosen = chosen[:cfg.direct_max_trial_splits]
    proposals, records = [], []
    for identifier in chosen.cpu().tolist():
        level, node = locations[identifier]
        rows = torch.where(owners == identifier)[0]
        local = PositionGradientEvidence(evidence.mean[rows], evidence.visible_views[rows],
                                         evidence.supported[rows])
        labels = cluster_split_region(xyz[rows], local, min_child_gs=child_minimum,
            iterations=cfg.direct_split_cluster_iterations,
            position_weight=cfg.direct_split_position_weight,
            gradient_weight=cfg.direct_split_gradient_weight)
        record = dict(level=level, node=node, score=float(scores.score[identifier]),
                      total_gs=int(scores.total_gs[identifier]),
                      supported_gs=int(scores.supported_gs[identifier]),
                      clustered=labels is not None)
        if labels is not None:
            record['child_gs'] = torch.bincount(labels, minlength=2).cpu().tolist()
            proposals.append((level, node, rows, labels, record))
        records.append(record)
    eligible = legal & torch.isfinite(scores.score) & (scores.score > 0)
    return proposals, dict(eligible_regions=int(eligible.sum()), legal_regions=int(legal.sum()),
                           selected_regions=records,
                           candidate_fraction=cfg.direct_split_candidate_fraction,
                           count_cap=cfg.direct_split_top_k,
                           supported_gs=int(evidence.supported.sum()),
                           total_gs=len(xyz), score='mean of per-GS mean-visible-world-gradient norms')


def propose_local_batch(motion, saliency=None, *, split_xyz=None, split_evidence=None):
    """One deterministic batch of configured disjoint edits.

    Each window can exchange coarse children, merge two initial three-level
    roots beneath one common root, or add spatial children to terminal nodes.
    No candidate gets a separate fitting run or acceptance test.
    None removes global count caps; gradient splits can select a fraction
    of all eligible regions, independently of neighboring windows.
    """
    cfg, old = motion.config, motion.active_layout
    index = old['index'].clone()
    counts = list(old['counts'])
    parents = [value.clone() for value in old['parents']]
    raw_root_values = torch.cat((motion.root_translation, motion.root_rotation), -1).detach()
    root_values, contrasts, *_ = motion._quantized_parameters()
    root_values = root_values.detach()
    contrasts = [value.detach() for value in contrasts]
    gates = motion.refinement_masks(motion.last_step)
    paths = motion._paths(gates)
    contrasts = [value * (paths[level][old['refine_rows'][level]] * gates[level])[:, None]
                 for level, value in enumerate(contrasts)]
    rotations, deltas = cumulative_transforms(motion, root_values, contrasts)
    full_q = torch.cat(rotations)
    full_delta = torch.cat(deltas)
    old_terminal = terminal_node_rows(old, [index.new_ones(n) for n in counts], selected_only=False)
    old_pivots = torch.cat(old['pivots'])
    q = full_q[old_terminal]
    world_b = full_delta[old_terminal] - rotation_delta(q, old_pivots[old_terminal])
    descriptor = torch.cat((world_b / motion.extent, _principal_rotation_vector(q)), -1)
    valid = old['index'][1] >= 0
    member_count = motion._cached_members[1]
    child_descriptor = descriptor.new_zeros(counts[1], 6)
    child_descriptor.index_add_(0, old['index'][1, valid], descriptor[valid])
    child_descriptor /= member_count.clamp_min(1)[:, None]
    hints = child_descriptor.cpu().numpy()
    child_sizes = member_count.cpu().tolist()
    roots = torch.where(motion._cached_occupancy[0])[0].cpu().tolist()
    if cfg.direct_candidate_windows == 0 and not cfg.direct_gradient_split_enabled:
        return None, dict(windows=[], reason='no_local_budget')
    from scipy.spatial import cKDTree
    pairs = set()
    if len(roots) >= 2 and cfg.direct_candidate_windows != 0:
        centers = old['pivots'][0][roots].cpu().numpy()
        _, neighbor = cKDTree(centers).query(centers, k=min(cfg.direct_neighbors + 1, len(roots)), workers=1)
        for row in range(len(roots)):
            for other in neighbor[row, 1:]:
                pairs.add(tuple(sorted((roots[row], roots[int(other)]))))
    records, used = [], set()
    operations = []
    merged_roots = set()
    split_count = 0
    split_proposals, split_report = [], None
    if cfg.direct_gradient_split_enabled:
        if split_xyz is None or split_evidence is None:
            raise ValueError('Gradient split proposals require fixed-state position evidence')
        split_proposals, split_report = gradient_split_proposals(motion, split_xyz, split_evidence)
    # Hint: fitted child disagreement, in a common world reference. All
    # neighboring pairs are scanned, including low-error / equal-motion pairs.
    scored = []
    for a, b in sorted(pairs):
        ca = motion._cached_children[0][a]
        cb = motion._cached_children[0][b]
        if len(ca) != 2 or len(cb) != 2:
            continue
        child = ca + cb
        if any(child_sizes[node] == 0 for node in child):
            continue
        def cost(groups):
            return sum(float(((hints[child[x]] - hints[child[y]]) ** 2).sum()) for x, y in groups)
        options = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        values = [cost(option) for option in options]
        chosen = min(range(3), key=values.__getitem__)
        if values[0] - values[chosen] <= 64 * torch.finfo(descriptor.dtype).eps ** 2:
            chosen = 0
        scored.append((values[0] - values[chosen], a, b, child, chosen))
    scored.sort(key=lambda value: (-value[0], value[1], value[2]))
    for gain, a, b, child, chosen in scored:
        if a in used or b in used:
            continue
        changed = []
        if chosen:
            # Swap one whole subtree; its unique GS descendants move together.
            left = child[1]
            right = child[2 if chosen == 1 else 3]
            parents[0][left], parents[0][right] = b, a
            index[0, old['index'][1] == left] = b
            index[0, old['index'][1] == right] = a
            changed.append('cross_boundary_regroup')
            operations.append(dict(kind='swap', left=left, right=right))
        elif (cfg.direct_enable_merges and len(counts) == 4 and
              not bool((index[3, (index[0] == a) | (index[0] == b)] >= 0).any())):
            if cfg.direct_merge_max_disagreement is not None:
                left_mean = descriptor[old['index'][0] == a].mean(0)
                right_mean = descriptor[old['index'][0] == b].mean(0)
                if float((left_mean - right_mean).norm()) > cfg.direct_merge_max_disagreement:
                    continue
            # Direct merge entry independent of a high-error trigger. The two
            # old three-level roots become child branches under one common
            # factor; original terminal motion classes survive at depth four.
            window = (old['index'][0] == a) | (old['index'][0] == b)
            fine_a = torch.where((old['index'][2] >= 0) & (old['index'][0] == a))[0]
            fine_b = torch.where((old['index'][2] >= 0) & (old['index'][0] == b))[0]
            coarse = child
            slots = sorted(set(old['index'][2, fine_a].tolist()))
            old_fine = sorted(set(old['index'][2, window].tolist()) - {-1})
            if len(slots) == 4 and len(old_fine) == 8:
                index[0, window] = a
                index[1, fine_a], index[1, fine_b] = child[0], child[1]
                for position, node in enumerate(coarse):
                    index[2, old['index'][1] == node] = slots[position]
                for position, node in enumerate(old_fine):
                    rows = (old['index'][2] == node) & window
                    index[3, rows] = counts[3] + position
                    parent = int(index[2, torch.where(rows)[0][0]])
                    parents[2] = torch.cat((parents[2], parents[2].new_tensor([parent])))
                counts[3] += len(old_fine)
                changed.append('common_parent_merge_preserving_children')
                operations.append(dict(kind='merge', a=a, b=b))
                merged_roots.update((a, b))
        if (not cfg.direct_gradient_split_enabled and
                'common_parent_merge_preserving_children' not in changed and
                (cfg.direct_max_trial_splits is None or split_count < cfg.direct_max_trial_splits)):
            # Rank only small terminal regions in this already selected window.
            # Screen-gradient magnitude is a hint; it never accepts the edit.
            terminal_nodes = sorted(set(index[2, (index[0] == a) | (index[0] == b)].tolist()) - {-1})
            ranked = []
            for node in terminal_nodes:
                rows = torch.where(index[2] == node)[0]
                if len(rows) < 2 * cfg.min_child_gs or bool((index[3, rows] >= 0).any()):
                    continue
                hint = float(saliency[rows].mean()) if saliency is not None else 0.
                ranked.append((-hint, node, rows))
            for _, node, rows in sorted(ranked, key=lambda value: (value[0], value[1]))[:1]:
                points = motion.base_xyz[rows]
                axis = int((points.amax(0) - points.amin(0)).argmax())
                order = rows[torch.argsort(points[:, axis], stable=True)]
                midpoint = len(order) // 2
                index[3, order[:midpoint]], index[3, order[midpoint:]] = counts[3], counts[3] + 1
                counts[3] += 2
                split_count += 1
                parents[2] = torch.cat((parents[2], parents[2].new_tensor([node, node])))
                changed.append('selected_terminal_trial_split')
                labels = torch.zeros(len(rows), dtype=torch.long, device=rows.device)
                labels[torch.argsort(points[:, axis], stable=True)[midpoint:]] = 1
                operations.append(dict(kind='split', level=2, node=node, labels=labels.cpu().tolist()))
        if changed:
            records.append(dict(roots=[a, b], actions=changed, fitted_hint_gain=gain))
            used.update((a, b))
        if cfg.direct_candidate_windows is not None and len(records) >= cfg.direct_candidate_windows:
            break
    for level, node, rows, labels, record in split_proposals:
        if set(old['index'][0, rows].cpu().tolist()) & merged_roots:
            record['edit_skipped'] = 'conflicting_common_parent_merge'
            continue
        if sum(counts) + 2 > cfg.direct_node_budget:
            record['edit_skipped'] = 'node_budget'
            continue
        index[level + 1, rows] = counts[level + 1] + labels
        counts[level + 1] += 2
        parents[level] = torch.cat((parents[level], parents[level].new_tensor([node, node])))
        changed_roots = sorted(set(index[0, rows].cpu().tolist()))
        used.update(changed_roots)
        records.append(dict(roots=changed_roots, actions=['gradient_cluster_terminal_split'],
                            level=level, node=node, score=record['score']))
        record['edit_applied'] = True
        operations.append(dict(kind='split', level=level, node=node, labels=labels.cpu().tolist()))
    if not records:
        return None, dict(windows=[], reason='no_legal_edits', split_selection=split_report)
    if sum(counts) > cfg.direct_node_budget:
        return None, dict(windows=records, reason='node_budget')
    fallback = []
    for count, old_pivot in zip(counts, old['pivots']):
        pivot = motion.base_xyz.new_zeros(count, 3)
        pivot[:len(old_pivot)] = old_pivot
        fallback.append(pivot)
    layout = _binary_layout(index, counts, parents, motion.base_xyz, fallback)
    terminal = terminal_node_rows(layout, [index.new_ones(n) for n in counts], selected_only=False)
    pivot = torch.cat(layout['pivots'])[terminal]
    terminal_delta = world_b + rotation_delta(q, pivot)
    roots_value, new_contrasts = decompose_terminal_transforms(layout, q, terminal_delta, affected_roots=used)
    # Preserve every unrelated raw parameter exactly, avoiding quantizer
    # boundary changes from a mathematically equivalent global factorization.
    untouched_roots = torch.tensor([node not in used for node in range(counts[0])],
                                   device=index.device, dtype=torch.bool)
    roots_value[untouched_roots] = raw_root_values[untouched_roots]
    root_owner = [torch.arange(counts[0], device=index.device)]
    for parent in parents:
        root_owner.append(root_owner[-1][parent])
    for level, rows in enumerate(layout['refine_rows']):
        old_positions = {int(node): pos for pos, node in enumerate(old['refine_rows'][level])}
        old_raw = torch.cat((motion.contrast_translation[level], motion.contrast_rotation[level]), -1).detach()
        owners = root_owner[level].cpu().tolist()
        for position, node in enumerate(rows.cpu().tolist()):
            if owners[node] not in used and node in old_positions:
                new_contrasts[level][position] = old_raw[old_positions[node]]
    candidate = CommonContrastMotion(copy.deepcopy(cfg))
    candidate.setup(motion.base_xyz, motion.base_rotation, motion.extent,
                    frame_index=motion.previous_frame_index, layout_override=layout)
    candidate._frame_coding_layout = _copy_layout(motion._frame_coding_layout)
    candidate._pending_edit_operations = operations
    with torch.no_grad():
        candidate.root_translation.copy_(roots_value[:, :3])
        candidate.root_rotation.copy_(roots_value[:, 3:])
        for t, r, value in zip(candidate.contrast_translation, candidate.contrast_rotation, new_contrasts):
            t.copy_(value[:, :3]); r.copy_(value[:, 3:])
        for level, logits in enumerate(candidate.refine_logits):
            old_rows = {int(node): pos for pos, node in enumerate(old['refine_rows'][level])}
            for pos, node in enumerate(layout['refine_rows'][level]):
                if int(node) in old_rows:
                    logits[pos] = motion.refine_logits[level][old_rows[int(node)]]
    # Keep unrelated gate choices. Only chosen local windows open residuals
    # for the reserved normal refinement steps; identity children stay zero.
    candidate.hard_masks = dict(refine=[])
    window_rows = torch.zeros(index.shape[1], dtype=torch.bool, device=index.device)
    for record in records:
        for root in record['roots']:
            window_rows |= old['index'][0] == root
    for level, rows in enumerate(layout['refine_rows']):
        mask = torch.zeros(len(rows), dtype=torch.bool, device=index.device)
        old_rows = {int(node): pos for pos, node in enumerate(old['refine_rows'][level])}
        for pos, node in enumerate(rows):
            if int(node) in old_rows:
                mask[pos] = gates[level][old_rows[int(node)]]
            if bool(((layout['index'][level] == node) & window_rows).any()):
                mask[pos] = True
        candidate.hard_masks['refine'].append(mask)
    if motion.last_step < cfg.dense_fit_steps + cfg.gate_compress_steps:
        candidate.hard_masks = None
        # Give new details time to fit before multiple ancestors can close.
        with torch.no_grad():
            for operation in operations:
                if operation['kind'] != 'split':
                    continue
                level, node = operation['level'], operation['node']
                while level >= 0:
                    if level < len(candidate.refine_logits):
                        position = candidate._cached_gate_positions[level].get(node)
                        if position is not None:
                            candidate.refine_logits[level][position].clamp_(min=cfg.direct_candidate_gate_logit)
                    if level == 0:
                        break
                    node = int(layout['parents'][level - 1][node])
                    level -= 1
    candidate.last_step = motion.last_step
    candidate.stage_seconds = dict(motion.stage_seconds)
    # The normal gate_refine transition resets motion Adam in both branches.
    candidate._make_optimizers()
    motion._make_optimizers()
    candidate.current_phase = motion.current_phase = None
    compare_step = (cfg.dense_fit_steps + cfg.gate_compress_steps + 1
                    if motion.last_step >= cfg.dense_fit_steps + cfg.gate_compress_steps else motion.last_step)
    old_xyz, old_q = motion(compare_step)
    new_xyz, new_q = candidate(compare_step)
    report = dict(windows=records, reason='proposed', old_counts=old['counts'], new_counts=counts,
                  split_selection=split_report,
                  changed_owners=int((old['index'] != layout['index']).any(0).sum()),
                  initialization_xyz_max_abs=float((old_xyz - new_xyz).abs().max()),
                  initialization_quaternion_max_abs=float(torch.minimum((old_q - new_q).abs().amax(-1),
                                                                            (old_q + new_q).abs().amax(-1)).max()))
    return candidate, report


@torch.no_grad()
def prune_inactive_split_edits(motion):
    """Remove new split routes whose parent/path gates ended closed.

    Existing node parameters and active split parent IDs are preserved.
    Only unused, newly appended descendants are removed; no pose refit occurs.
    """
    operations = motion._pending_edit_operations
    if not operations or not motion.config.direct_prune_inactive_splits:
        return motion, dict(removed_splits=0)
    gates = motion.refinement_masks(motion.config.motion_steps)
    paths = motion._paths(gates)
    keep, removed = [], []
    for operation in operations:
        if operation['kind'] == 'split':
            level, node = operation['level'], operation['node']
            position = motion._cached_gate_positions[level][node]
            if not bool(paths[level][node]) or not bool(gates[level][position]):
                removed.append((level, node))
                continue
        keep.append(operation)
    if not removed:
        return motion, dict(removed_splits=0)
    index, counts, parents = replay_node_edits(motion._frame_coding_layout, keep)
    fallback = []
    for count, old in zip(counts, motion.active_layout['pivots']):
        value = motion.base_xyz.new_zeros(count, 3)
        value[:min(count, len(old))] = old[:min(count, len(old))]
        fallback.append(value)
    layout = _binary_layout(index, counts, parents, motion.base_xyz, fallback)
    pruned = CommonContrastMotion(copy.deepcopy(motion.config))
    pruned.setup(motion.base_xyz, motion.base_rotation, motion.extent,
        frame_index=motion.previous_frame_index, layout_override=layout)
    pruned._frame_coding_layout = _copy_layout(motion._frame_coding_layout)
    pruned._pending_edit_operations = keep
    pruned.root_translation.copy_(motion.root_translation)
    pruned.root_rotation.copy_(motion.root_rotation)
    pruned.hard_masks = dict(refine=[])
    for level, rows in enumerate(layout['refine_rows']):
        old_positions = motion._cached_gate_positions[level]
        indices = torch.tensor([old_positions[int(node)] for node in rows],
                               device=motion.base_xyz.device, dtype=torch.long)
        pruned.contrast_translation[level].copy_(motion.contrast_translation[level][indices])
        pruned.contrast_rotation[level].copy_(motion.contrast_rotation[level][indices])
        pruned.refine_logits[level].copy_(motion.refine_logits[level][indices])
        pruned.hard_masks['refine'].append(gates[level][indices].clone())
    pruned.last_step = motion.last_step
    pruned.stage_seconds = dict(motion.stage_seconds)
    pruned.last_metrics = dict(motion.last_metrics)
    before_xyz, before_rotation = motion(motion.config.motion_steps)
    after_xyz, after_rotation = pruned(pruned.config.motion_steps)
    if not torch.equal(before_xyz, after_xyz) or not torch.equal(before_rotation, after_rotation):
        raise AssertionError('Removing unreachable split edits changed the materialized pose')
    return pruned, dict(removed_splits=len(removed), retained_operations=len(keep),
                       remaining_counts=counts)
