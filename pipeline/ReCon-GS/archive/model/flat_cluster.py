"""Single-level shared rigid motion with sparse per-GS residuals."""

from dataclasses import dataclass
import heapq
import math
from typing import Optional

import numpy as np
import torch

from .motion_clustering import cluster_flat_motion_layout
from .sparse_hierarchical import SparseHierarchicalConfig, SparseHierarchicalMotion


@dataclass
class FlatClusterConfig(SparseHierarchicalConfig):
    # Approximate initial members per cluster; there are no parent budgets.
    cluster_gs: Optional[int] = None
    tree_builder: str = 'rigid_cluster'

    def validate(self):
        if (isinstance(self.cluster_gs, bool) or not isinstance(self.cluster_gs, int)
                or self.cluster_gs < 1):
            raise ValueError('cluster_gs must be a positive integer')
        # The shared birth/optimizer validator still carries the legacy name.
        # Flat seeding reads cluster_gs only; no extra hierarchy budget exists.
        self.fine_region_gs = self.cluster_gs
        super().validate()
        if self.tree_builder != 'rigid_cluster':
            raise ValueError('flat_cluster requires current-frame rigid clustering')
        if self.tree_cluster_spatial_weight != 0:
            raise ValueError('Flat clustering uses spatial neighbours without a spatial score')
        if self.motion_export_format not in {'raw_parameters', 'sparse_parameters'}:
            raise ValueError('Flat clusters require ordinary parameter arrays')


def stable_flat_layout(xyz, cluster_gs):
    """Deterministic spatial seeds, returned as one owner row and no parents.

    Splits only initialize the membership vector. No split tree is retained,
    trained, or exported. CPU coordinate ties use the original GS row index.
    """
    if (isinstance(cluster_gs, bool) or not isinstance(cluster_gs, int)
            or cluster_gs < 1 or xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz)):
        raise ValueError('Flat seeds require nonempty xyz and positive cluster_gs')
    points = xyz.detach().cpu().double().numpy()
    if not np.isfinite(points).all():
        raise ValueError('Flat seed coordinates must be finite')
    count = math.ceil(len(points) / cluster_gs)
    groups = [(-len(points), 0, np.arange(len(points), dtype=np.int64))]
    serial = 1
    while len(groups) < count:
        _, _, rows = heapq.heappop(groups)
        axis = np.ptp(points[rows], axis=0).argmax()
        ordered = rows[np.lexsort((rows, points[rows, axis]))]
        for part in np.array_split(ordered, 2):
            heapq.heappush(groups, (-len(part), serial, part))
            serial += 1
    owner = np.empty(len(points), dtype=np.int64)
    for label, (_, _, rows) in enumerate(sorted(groups, key=lambda item: item[1])):
        owner[rows] = label
    return torch.as_tensor(owner[None, :], device=xyz.device), [count], []


class FlatClusterMotion(SparseHierarchicalMotion):
    """Reuse phase/gate training, with exactly one group transform per GS."""

    layout_version = 'single_level_shared_cluster_motion_v1'
    num_motion_levels = 1
    config_class = FlatClusterConfig

    def _seed_layout(self, xyz):
        return stable_flat_layout(xyz, self.config.cluster_gs)

    def _cluster_layout(self, source, source_q, target, target_q, observations):
        return cluster_flat_motion_layout(source, source_q, target, target_q,
            self.index[0].detach().cpu().numpy(), self.counts[0], observations,
            extent=self.extent, neighbors=self.config.tree_cluster_neighbors,
            iterations=self.config.tree_cluster_iterations,
            orientation_extent=self.config.tree_cluster_orientation_extent,
            min_observations=self.config.tree_probe_min_observations)

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.tree_setup['semantics'] = (
            'One motion group per GS; no parents or octree. Spatial partitions '
            'only seed the groups. Current training-view probes refine membership '
            'with shared-SE3 fitting error; spatial neighbours limit candidates '
            'without a spatial penalty. Membership follows survivor identities '
            'between periodic refreshes. Each group has one rigid transform; '
            'local residuals and final m gates handle exceptions and exact reuse.')

    def statistics(self):
        result = super().statistics()
        result['semantics'] = (
            'Single-level shared motion. Group z enables the shared transform; '
            'local z enables a per-GS residual. Final m=0 returns the exact '
            'previous pose. There are no depth gates or parent transforms.')
        return result
