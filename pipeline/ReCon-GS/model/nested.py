"""Parent-restricted spatial refinement and visible-gradient statistics."""
import torch
from torch import nn


class NestedFineGrid(nn.Module):
    def __init__(self, xyz, parent, num_middle, children_per_parent, overlap_ratio=0.15):
        super().__init__()
        if xyz.shape[0] == 0 or children_per_parent < 1:
            raise ValueError('Nested sampling requires points and at least one candidate child')
        # Batched farthest-point sampling: all parents are processed together.
        count = torch.bincount(parent, minlength=num_middle)
        point_ids = torch.arange(xyz.shape[0], device=xyz.device)
        total = xyz.new_zeros(num_middle, 3)
        total.index_add_(0, parent, xyz)
        centroid = total / count.clamp_min(1)[:, None]
        center_distance = ((xyz - centroid[parent]) ** 2).sum(-1)
        minimum = xyz.new_full((num_middle,), float('inf'))
        minimum.scatter_reduce_(0, parent, center_distance, reduce='amin', include_self=True)
        first = torch.full((num_middle,), xyz.shape[0], device=xyz.device, dtype=torch.long)
        first.scatter_reduce_(0, parent, torch.where(center_distance == minimum[parent], point_ids,
                                                    xyz.shape[0]), reduce='amin', include_self=True)
        nearest = torch.full((xyz.shape[0],), float('inf'), device=xyz.device)
        assignment = torch.zeros_like(parent)
        vertices, parents, sampled_ids = [], [], []
        table = torch.full((num_middle, children_per_parent), -1, device=xyz.device, dtype=torch.long)
        selected = first
        valid = count > 0
        offset = 0
        for slot in range(children_per_parent):
            current_parents = torch.where(valid)[0]
            if current_parents.numel() == 0:
                break
            current_ids = selected[current_parents]
            child_ids = torch.zeros(num_middle, device=xyz.device, dtype=torch.long)
            child_ids[current_parents] = torch.arange(len(current_parents), device=xyz.device) + offset
            table[current_parents, slot] = child_ids[current_parents]
            offset += current_parents.numel()
            # Empty parents have no points; their safe placeholder is never used.
            centers = xyz[selected.clamp_max(xyz.shape[0] - 1)]
            distance = ((xyz - centers[parent]) ** 2).sum(-1)
            update = valid[parent] & (distance < nearest)
            nearest = torch.where(update, distance, nearest)
            assignment[update] = child_ids[parent[update]]
            vertices.append(xyz[current_ids])
            parents.append(current_parents)
            sampled_ids.append(current_ids)
            maximum = xyz.new_full((num_middle,), -1.)
            maximum.scatter_reduce_(0, parent, nearest, reduce='amax', include_self=True)
            # Zero remaining distance means no distinct spatial child is left.
            valid = (maximum > 0) & (count > slot + 1)
            candidate = torch.where(nearest == maximum[parent], point_ids, xyz.shape[0])
            selected = torch.full_like(first, xyz.shape[0])
            selected.scatter_reduce_(0, parent, candidate, reduce='amin', include_self=True)
        self.register_buffer('vertex', torch.cat(vertices))
        self.register_buffer('parent', torch.cat(parents))
        self.register_buffer('vertex_idx', torch.cat(sampled_ids))
        self.register_buffer('p2v', assignment[:, None])
        self.register_buffer('child_table', table)
        self.vertex_num = self.vertex.shape[0]
        assert torch.equal(self.parent[assignment], parent)
        # Missing slots have zero weight; index 0 is only a safe gather placeholder.
        self.register_buffer('blend_indices', table[parent].clamp_min(0))
        self.register_buffer('blend_weights', boundary_blend_weights(
            xyz, parent, self.vertex, table, assignment, overlap_ratio))

    @torch.no_grad()
    def overlap_statistics(self):
        support = self.blend_weights > 0
        shared = support.sum(-1) > 1
        count = torch.bincount(self.blend_indices[support], minlength=self.vertex_num)
        shared_count = torch.bincount(self.blend_indices[support & shared[:, None]],
                                      minlength=self.vertex_num)
        return dict(shared_gs=shared.sum(), overlap_gs_ratio=shared.float().mean(),
                    fine_support_count=count, fine_shared_count=shared_count,
                    fine_shared_ratio=shared_count.float() / count.clamp_min(1))


@torch.no_grad()
def boundary_blend_weights(xyz, parent, vertex, children, assignment, overlap_ratio):
    """Smooth compact support across expanded Voronoi cells, within each parent.

    For anchors separated by D, the full transition band is overlap_ratio * D.
    Products of pairwise tapers also handle three-way junctions continuously.
    """
    if not 0 <= overlap_ratio <= 1:
        raise ValueError('overlap_ratio must be in [0, 1]')
    slots = children[parent]
    if overlap_ratio == 0:
        return ((slots == assignment[:, None]) & (slots >= 0)).to(xyz.dtype)
    centers = vertex[children.clamp_min(0)]
    distance_sq = ((xyz[:, None] - centers[parent]) ** 2).sum(-1)
    scores = (slots >= 0).to(xyz.dtype)
    for c in range(children.shape[1]):
        for other in range(children.shape[1]):
            if c == other:
                continue
            separation_sq = ((centers[:, c] - centers[:, other]) ** 2).sum(-1)
            # Signed distance to the bisector, mapped from [-band/2, band/2]
            # to [0, 1]. Distinct sampled anchors make valid denominators positive.
            denominator = (2 * overlap_ratio * separation_sq[parent]).clamp_min(
                torch.finfo(xyz.dtype).tiny)
            t = (0.5 + (distance_sq[:, other] - distance_sq[:, c]) / denominator).clamp(0, 1)
            taper = t.square() * (3 - 2 * t)
            pair_present = (slots[:, c] >= 0) & (slots[:, other] >= 0)
            scores[:, c] *= torch.where(pair_present, taper, 1.)
    return scores / scores.sum(-1, keepdim=True)


@torch.no_grad()
def visible_gradient_variance(gradient, parent, visible, num_middle):
    """Population variance of 3D vectors; fewer than two visible GS is missing."""
    valid = visible.bool() & torch.isfinite(gradient).all(-1)
    p, g = parent[valid], gradient[valid]
    count = torch.bincount(p, minlength=num_middle)
    total = gradient.new_zeros(num_middle, 3)
    total.index_add_(0, p, g)
    mean = total / count.clamp_min(1)[:, None]
    squared = ((g - mean[p]) ** 2).sum(-1)
    variance = gradient.new_zeros(num_middle)
    variance.index_add_(0, p, squared)
    return variance / count.clamp_min(1), count


@torch.no_grad()
def weighted_gradient_diagnostics(gradient, parent, visible, num_middle, fine):
    """Compare parent variance with weighted within-child variance on the same GS."""
    before, count = visible_gradient_variance(gradient, parent, visible, num_middle)
    valid = visible.bool() & torch.isfinite(gradient).all(-1)
    g, p = gradient[valid], parent[valid]
    indices, weights = fine.blend_indices[valid], fine.blend_weights[valid]
    supported = weights > 0
    ids, w = indices[supported], weights[supported]
    mass = gradient.new_zeros(fine.vertex_num)
    total = gradient.new_zeros(fine.vertex_num, 3)
    mass.index_add_(0, ids, w)
    total.index_add_(0, ids, (g[:, None] * weights[:, :, None])[supported])
    mean = total / mass.clamp_min(torch.finfo(gradient.dtype).tiny)[:, None]
    within = (weights * ((g[:, None] - mean[indices]) ** 2).sum(-1)).sum(-1)
    after = gradient.new_zeros(num_middle)
    after.index_add_(0, p, within)
    return before, after / count.clamp_min(1), count
