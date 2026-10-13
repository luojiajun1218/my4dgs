"""Causal GS motion observations, pooled using the current region membership."""

import numpy as np
import torch

from .shared_motion_clustering import fit_shared_motion, quaternion_matrices


class MotionHistory:
    def __init__(self, window):
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError('Motion history window must be a nonnegative integer')
        self.window = window
        self.records = []

    @torch.no_grad()
    def record(self, frame, source, target, relative_rotation, observations):
        if self.window == 0:
            return
        if (target.shape != source.shape or source.ndim != 2 or source.shape[1] != 3 or
                relative_rotation.shape != (len(source), 4) or
                observations.shape != (len(source),) or
                bool(relative_rotation.norm(dim=-1).eq(0).any()) or
                not all(bool(torch.isfinite(x).all()) for x in (source, target, relative_rotation))):
            raise ValueError('Motion history needs finite, aligned surviving GS observations')
        if self.records and frame < self.records[-1]['frame']:
            raise ValueError('Motion history cannot observe a future frame before the past')
        record = dict(frame=int(frame), source=source.detach().cpu().float().clone(),
                      displacement=(target - source).detach().cpu().float().clone(),
                      rotation=relative_rotation.detach().cpu().float().clone(),
                      observations=observations.detach().cpu().long().clone())
        if self.records and frame == self.records[-1]['frame']:
            self.records[-1] = record
        else:
            self.records.append(record)
        self.records = self.records[-self.window:]

    @torch.no_grad()
    def pool(self, owner, nodes, orientation_length, min_observations):
        """Fit each current group's past SE(3), excluding unborn/unobserved GS.

        GS row IDs persist because the active method does not delete old GS.
        Historical regional IDs are never used to align the observations.
        """
        owner = owner.detach().cpu().numpy()
        quaternions, offsets, pivots, validity = [], [], [], []
        for record in self.records:
            count = len(record['source'])
            if count > len(owner):
                raise ValueError('Motion history does not support deletion of old GS')
            source = record['source'].numpy().astype(np.float64)
            target = source + record['displacement'].numpy()
            observed = record['observations'].numpy()
            q = np.zeros((nodes, 4), dtype=np.float64)
            q[:, 0] = 1
            b, p = np.zeros((nodes, 3)), np.zeros((nodes, 3))
            valid = np.zeros(nodes, dtype=bool)
            valid[0] = True
            for node in range(1, nodes):
                rows = np.flatnonzero((owner[:count] == node) & (observed >= min_observations))
                # Three observed centers are the minimum positional rigid fit.
                if len(rows) < 3:
                    continue
                fit = fit_shared_motion(source[rows], target[rows],
                    quaternion_matrices(record['rotation'][rows].numpy()),
                    observed[rows], orientation_length)
                q[node], b[node], p[node] = fit['quaternion'], fit['offset'], source[rows].mean(0)
                valid[node] = True
            quaternions.append(q)
            offsets.append(b)
            pivots.append(p)
            validity.append(valid)
        return dict(quaternion=quaternions, offset=offsets, pivots=pivots, valid=validity)

    def capture(self):
        return [dict(record) for record in self.records]

    def restore(self, records):
        if len(records) > self.window:
            raise ValueError('Checkpoint motion history exceeds its configured window')
        frames = [record['frame'] for record in records]
        if frames != sorted(set(frames)):
            raise ValueError('Checkpoint motion history must be chronological')
        self.records = [dict(frame=record['frame'], **{
            name: record[name].detach().cpu().clone()
            for name in ('source', 'displacement', 'rotation', 'observations')}) for record in records]
