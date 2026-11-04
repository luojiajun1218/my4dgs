"""Add one candidate level without changing existing planes or routing prefixes."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.model.spatial_gate import spatial_route, SPATIAL_ROUTING_FIELDS
from pipeline.model.rigid_tree_codec import DecodedState, _validate_state


def extend(packet):
    state = DecodedState.from_reference(packet)
    _validate_state(state)
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
    for leaf in np.unique(leaves[supported]):
        centers[leaf] = np.median(xyz[supported & (leaves == leaf)], axis=0)
    routing = {k: getattr(state, k) for k in SPATIAL_ROUTING_FIELDS}
    if state.spatial_support_ball is not None:
        # Retain the exact old nearest-representative dictionary. Map each
        # representative to a child of its old leaf: outliers cannot jump to
        # a different coarse region just because depth was increased.
        nodes = state.spatial_leaf_nodes.numpy()
        side = state.spatial_leaf_centers.numpy() >= centers[nodes]
        octant = side[:, 0] + 2 * side[:, 1] + 4 * side[:, 2]
        routing['spatial_leaf_nodes'] = torch.from_numpy(
            new_start + 8 * (nodes - old_start) + octant)
    paths = spatial_route(state.xyz, torch.from_numpy(centers), depth + 1, routing=routing)
    if not torch.equal(paths[:, :-1], state.spatial_paths):
        raise ValueError('Extension changed an existing reference route')
    parents = np.r_[state.parents.numpy(),
                    np.repeat(np.arange(old_start, new_start, dtype=np.int64), 8)]
    opened = np.r_[state.spatial_open.numpy(), np.zeros(8 ** (depth + 1), dtype=bool)]
    result = {k: np.array(packet[k], copy=True) for k in packet}
    result.update(parents=parents, spatial_paths=paths.numpy(), spatial_open=opened,
                  spatial_centers=centers, region_count=np.array(len(parents)))
    for key, value in routing.items():
        if value is not None:
            result[key] = value.numpy()
    _validate_state(DecodedState.from_reference(result))
    # Also check off-reference positions, including far outliers.
    generator = torch.Generator().manual_seed(20261102)
    probes = torch.randn(2000, 3, generator=generator) * state.xyz.std(0).clamp_min(.01) * 3 + state.xyz.mean(0)
    old_routes = spatial_route(probes, state.spatial_centers, depth,
                               routing={k: getattr(state, k) for k in SPATIAL_ROUTING_FIELDS})
    new_routes = spatial_route(probes, torch.from_numpy(centers), depth + 1, routing=routing)
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
        result = extend(packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as target:
        np.savez_compressed(target, **result)
    print(f"depth={result['spatial_paths'].shape[1]} nodes={len(result['parents'])}; old routes and GS attributes preserved")


if __name__ == '__main__':
    main()
