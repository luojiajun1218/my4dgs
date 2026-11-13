"""为首帧空间参考文件的候选区域树增加一层。

读取已有首帧参考，构造下一层节点及路由，检查旧层路径不变后写入新文件；
保留已有 GS 属性与划分。只处理 2 至 4 层的首帧参考，不训练或修改源文件。
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.model.spatial_region_gate import spatial_owner, spatial_route, SPATIAL_ROUTING_FIELDS
from pipeline.model.frame_stream_format import (DecodedState, _validate_state)


def extend_tree_depth(packet):
    state = DecodedState.from_reference(packet)
    _validate_state(state)
    if state.spatial_paths is None or state.spatial_centers is None:
        raise ValueError('Expected a spatial reference with split planes')
    depth = state.spatial_paths.shape[1]
    if state.frame_index != 0 or not 2 <= depth < 5:
        raise ValueError('Expected a frame-zero spatial reference of depth 2..4')
    xyz = state.xyz.numpy()
    old_start = sum(8 ** k for k in range(depth))
    new_start = len(state.parents)
    centers = np.zeros((new_start, 3), dtype=np.float32)
    centers[:old_start] = state.spatial_centers.numpy()
    # Empty leaves have no eligible split. Their finite split planes still
    # need to be deterministic for subsequent decoded-position routing.
    centers[old_start:] = centers[state.parents[old_start:].numpy()]
    supported = np.ones(len(xyz), dtype=bool)
    if state.spatial_support_ball is not None:
        ball = state.spatial_support_ball.numpy()
        supported = np.linalg.norm(xyz.astype(np.float64) - ball[:3], axis=1) <= ball[3]
    leaves = state.spatial_paths[:, -1].numpy()
    # Group supported points once instead of scanning all points for every leaf.
    supported_xyz, supported_leaves = xyz[supported], leaves[supported]
    order = np.argsort(supported_leaves, kind='stable')
    leaf_ids, starts = np.unique(supported_leaves[order], return_index=True)
    for leaf, rows in zip(leaf_ids, np.split(order, starts[1:])):
        centers[leaf] = np.median(supported_xyz[rows], axis=0)
    old_routing = {key: getattr(state, key) for key in SPATIAL_ROUTING_FIELDS}
    routing = dict(old_routing)
    if state.spatial_support_ball is not None:
        # Retain the exact old nearest-representative dictionary. Map each
        # representative to a child of its old leaf: outliers cannot jump to
        # a different coarse region just because depth was increased.
        nodes = state.spatial_leaf_nodes.numpy()
        side = state.spatial_leaf_centers.numpy() >= centers[nodes]
        octant = side[:, 0] + 2 * side[:, 1] + 4 * side[:, 2]
        routing['spatial_leaf_nodes'] = torch.from_numpy(
            new_start + 8 * (nodes - old_start) + octant)
    center_tensor = torch.from_numpy(centers)
    paths = spatial_route(state.xyz, center_tensor, depth + 1, routing=routing)
    if not torch.equal(paths[:, :-1], state.spatial_paths):
        raise ValueError('Extension changed an existing reference route')
    parents = np.r_[state.parents.numpy(),
                    np.repeat(np.arange(old_start, new_start, dtype=np.int64), 8)]
    opened = np.r_[state.spatial_open.numpy(), np.zeros(8 ** (depth + 1), dtype=bool)]
    result = {k: np.array(packet[k], copy=True) for k in packet}
    # Compact references omit paths and owners and carry their old depth in
    # metadata.  Materialize both for the extended tree instead of retaining
    # metadata that describes the previous depth.
    result.pop('spatial_derived_depth', None)
    result.update(parents=parents, spatial_paths=paths.numpy(), spatial_open=opened,
                  spatial_centers=centers, region_count=np.array(len(parents)),
                  owner=spatial_owner(paths, torch.from_numpy(opened)).numpy())
    for key, value in routing.items():
        if value is not None:
            result[key] = value.numpy()
    _validate_state(DecodedState.from_reference(result))
    # Also check off-reference positions, including far outliers.
    generator = torch.Generator().manual_seed(20261102)
    probes = torch.randn(2000, 3, generator=generator) * state.xyz.std(0).clamp_min(.01) * 3 + state.xyz.mean(0)
    old_routes = spatial_route(probes, state.spatial_centers, depth, routing=old_routing)
    new_routes = spatial_route(probes, center_tensor, depth + 1, routing=routing)
    if not torch.equal(old_routes, new_routes[:, :-1]):
        raise ValueError('Extension changed coarse routing for probe positions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.source, allow_pickle=False) as packet:
        result = extend_tree_depth(packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as target:
        np.savez_compressed(target, **result)
    print(f"depth={result['spatial_paths'].shape[1]} nodes={len(result['parents'])}; old routes and GS attributes preserved")


if __name__ == '__main__':
    main()
