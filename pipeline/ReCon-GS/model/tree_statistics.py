"""Small, human-readable snapshots of direct GS ownership after tree edits."""

import csv
from pathlib import Path

import torch

from .rigid_residual_tree import _topological_order


@torch.no_grad()
def write_tree_statistics(path, motion, original_count, step):
    """Export the post-edit tree, excluding births of the current frame.

    Each GS has one direct owner, so each percentage column sums to 100.
    Frame0 columns exclude all births; old-GS columns include prior births.
    A zero local region residual can still inherit nonzero parent motion.
    """
    owner = motion.owner.detach().cpu().long()
    parents = motion.parents.detach().cpu().long()
    if (isinstance(original_count, bool) or not isinstance(original_count, int)
            or not 0 < original_count <= len(owner)):
        raise ValueError('Tree statistics need the retained frame0 GS count')
    nodes = len(parents)
    if bool(((owner < 0) | (owner >= nodes)).any()):
        raise ValueError('Tree statistics encountered an invalid owner')
    order = _topological_order(parents)
    depth = [0] * nodes
    for node in order:
        depth[node] = depth[int(parents[node])] + 1
    original = torch.bincount(owner[:original_count], minlength=nodes)
    old = torch.bincount(owner, minlength=nodes)
    rt, rr, gt, gr = (value.detach().cpu() for value in motion._values(hard=True))
    local_active = rt.ne(0).any(-1) | rr.ne(0).any(-1)
    gs_active = gt.ne(0).any(-1) | gr.ne(0).any(-1)
    q, offset = (value.detach().cpu() for value in motion._global_regions())
    inherited_active = q[:, 1:].ne(0).any(-1) | offset.ne(0).any(-1)
    residual_count = torch.bincount(owner[gs_active], minlength=nodes)
    translation_step = motion.steps[:3].detach().cpu()
    rotation_step = motion.steps[3:].detach().cpu()
    translation = (rt * translation_step).norm(dim=-1)
    rotation = (rr * rotation_step).norm(dim=-1)
    rows = [dict(
        frame=motion.frame_index, step=step, node_id=node,
        parent_id=int(parents[node]), depth=depth[node],
        original_frame0_gs_count=int(original[node]),
        original_frame0_gs_percent=100 * int(original[node]) / original_count,
        old_gs_count=int(old[node]), old_gs_percent=100 * int(old[node]) / len(owner),
        old_gs_with_local_residual=int(residual_count[node]),
        local_translation_norm_world=float(translation[node]),
        local_rotation_norm_rad=float(rotation[node]),
        local_motion_nonzero=int(local_active[node]),
        inherited_motion_nonzero=int(inherited_active[node]),
    ) for node in range(nodes)]
    labels = getattr(motion, 'cluster_labels', None)
    if labels is not None:
        labels = labels.detach().cpu()
        for node, row in enumerate(rows):
            members = owner == node
            row.update(near_identity_candidates_at_clustering=int((members & labels.eq(-2)).sum()),
                       noise_at_clustering=int((members & labels.eq(-1)).sum()),
                       unobserved_at_clustering=int((members & labels.eq(-3)).sum()))
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Tree statistics must not overwrite a symlink target')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(path=str(path), frame=motion.frame_index, step=step,
                nodes=nodes, original_frame0_gs=original_count, old_gs=len(owner),
                scope='Direct ownership after tree update and proximal projection; '
                      'excludes current-frame births. Local zero is not necessarily static.')
