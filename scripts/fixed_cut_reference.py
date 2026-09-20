"""Independent slow fixed-cut oracle for the planned hierarchy ablation.

Not imported by training. Keep the full candidate tree/codec: a depth limit
forces every eligible ancestor to split, even if its contrast quantizes to zero.
"""
import importlib
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')


def project_fixed_cut(target, tree, step, rate, rho, depth):
    if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    target = np.asarray(target, dtype=np.float64)
    step, rho = np.asarray(step, dtype=np.float64), np.asarray(rho, dtype=np.float64)
    if target.ndim != 2 or target.shape[0] != tree.point_count or not target.shape[1]:
        raise ValueError('Expected target [N,C]')
    channels = target.shape[1]
    if step.shape != (channels,) or rho.shape != (channels,):
        raise ValueError('Expected channel-wise quantization and curvature')
    if (not all(np.isfinite(v).all() for v in (target, step, rho))
            or (step <= 0).any() or (rho <= 0).any() or not np.isfinite(rate) or rate < 0):
        raise ValueError('Invalid metric/rate')
    means = [target[n.indices].mean(0) for n in tree.nodes]

    def symbols(mean, weight):
        return np.array([P.choose_symbol(float(mean[c]), float(step[c]),
                                         float(weight * rho[c]), rate)[0]
                         for c in range(channels)], dtype=np.int64)

    root = symbols(means[0], tree.point_count)
    details = {}
    bits = sum(P.signed_symbol_bits(s) for s in root)

    def visit(index, level):
        nonlocal bits
        node = tree.nodes[index]
        if node.left < 0:
            return
        bits += 1
        if level >= depth:
            return
        nl, nr = len(tree.nodes[node.left].indices), len(tree.nodes[node.right].indices)
        details[index] = symbols(means[node.left] - means[node.right], nl * nr / (nl + nr))
        bits += sum(P.signed_symbol_bits(s) for s in details[index])
        visit(node.left, level + 1)
        visit(node.right, level + 1)

    visit(0, 0)
    field = P.reconstruct(tree, root, details, step)
    distortion = .5 * float(np.sum(rho * (field - target) ** 2))
    result = P.Projection(field, root, details, tuple(details), step.copy(), bits,
                          distortion, distortion + rate * bits)
    packet = P.encode_projection(result, tree)
    if not np.array_equal(P.decode_projection(packet, tree), field):
        raise AssertionError('Fixed-cut codec mismatch')
    return result
