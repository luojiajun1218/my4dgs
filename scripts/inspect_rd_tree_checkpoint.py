"""Read-only encoder-state diagnostic of the actual selected local tree depths."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_motion')


def norm_summary(value):
    """Euclidean per-point norms; rotations are rotation-vector radians."""
    value = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError('Nonfinite checkpoint motion values')
    levels = [0., .5, .9, .99, .999, 1.]
    return {'mean': float(norms.mean()),
            'quantiles': dict(zip(map(str, levels), np.quantile(norms, levels).tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['deformation']
    motion = M.RDTreeMotion(M.RDTreeConfig(**state['config']))
    motion.restore(state)  # Checks recomputed packet against saved packet.
    projection = motion.project()
    splits = set(projection.split_nodes)
    leaf_hist, point_hist, split_hist = Counter(), Counter(), Counter()
    leaf_sizes = []
    stack = [(0, 0)]
    while stack:
        index, depth = stack.pop()
        node = motion.tree.nodes[index]
        if index in splits:
            assert node.left >= 0 and node.right >= 0
            split_hist[depth] += 1
            stack.extend(((node.left, depth + 1), (node.right, depth + 1)))
        else:
            leaf_hist[depth] += 1
            point_hist[depth] += len(node.indices)
            leaf_sizes.append(len(node.indices))
    assert sum(point_hist.values()) == motion.tree.point_count
    assert len(leaf_sizes) == len(splits) + 1
    latent = motion.latent.detach().numpy()
    decoded = projection.field
    norms = {}
    for name, value in [('latent', latent), ('decoded', decoded), ('residual', latent - decoded)]:
        norms[name + '_translation'] = norm_summary(value[:, :3])
        norms[name + '_rotation_radians'] = norm_summary(value[:, 3:])
    result = dict(checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  config=state['config'], candidate_leaves=sum(n.left < 0 for n in motion.tree.nodes),
                  selected_leaves=len(leaf_sizes), leaf_depth_histogram=dict(sorted(leaf_hist.items())),
                  point_depth_histogram=dict(sorted(point_hist.items())),
                  split_depth_histogram=dict(sorted(split_hist.items())),
                  leaf_size_min=min(leaf_sizes), leaf_size_max=max(leaf_sizes),
                  payload_bits=projection.payload_bits,
                  motion_norms=norms,
                  quadratic_commitment=projection.distortion,
                  weighted_payload_rate=motion.rate * projection.payload_bits,
                  scope='Encoder-state diagnostic, not an independent stream decoder or boundary-quality proof.')
    output = ROOT / 'output' / ('idea_rd_tree_structure_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT / 'pipeline/ReCon-GS/model/rd_tree_motion.py',
               ROOT / 'pipeline/ReCon-GS/model/rd_tree_projection.py']
    result['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print('Saved:', output)


if __name__ == '__main__':
    main()
