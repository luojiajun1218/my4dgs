"""CPU research reference: codec-aware projection onto a pruned motion tree.

Used by RDTreeMotion training. The optimum is for a diagonal quadratic surrogate,
hard nested regions, and a fixed *candidate* tree, not for rendered image loss or
arbitrary partitions. All retained subtree depths compete in one bottom-up pass.
The decoder needs the identical candidate tree from shared geometry/order.
"""
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
import struct

import numpy as np


@dataclass
class Node:
    indices: np.ndarray
    left: int = -1
    right: int = -1


@dataclass
class CandidateTree:
    nodes: list
    point_count: int

    def fingerprint(self):
        digest = hashlib.sha256()
        for node in self.nodes:
            digest.update(struct.pack('<iii', len(node.indices), node.left, node.right))
            digest.update(np.asarray(node.indices, dtype='<i8').tobytes())
        return digest.digest()[:16]


def build_candidate_tree(xyz, max_depth=4, min_leaf_size=1, partition='median',
                         clustering_iterations=4):
    """Deterministic median splits or parent-restricted geometric two-means.

    This is a bounded geometry dictionary, NOT learned motion boundaries. Exact
    reconstruction across platforms still requires a shared geometry convention.
    Median splits use the longest extent with index tie breaking. Two-means
    starts at the two median-part means, with bounded Lloyd iterations and a
    median fallback for undersized clusters. Neither uses renderer gradients.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or not len(xyz) or not xyz.shape[1] or not np.isfinite(xyz).all():
        raise ValueError('Expected finite nonempty [N,D] coordinates')
    if max_depth < 0 or min_leaf_size < 1:
        raise ValueError('Invalid candidate limits')
    if partition not in ('median', 'geometry_2means', 'geometry_2means_batched') or clustering_iterations < 1:
        raise ValueError('Invalid candidate partition settings')
    if partition == 'geometry_2means_batched':
        return _build_batched_clusters(xyz, max_depth, min_leaf_size, clustering_iterations)
    nodes = []

    def visit(indices, depth):
        index = len(nodes)
        nodes.append(Node(indices))
        if depth < max_depth and len(indices) >= 2 * min_leaf_size:
            axis = int(np.argmax(np.ptp(xyz[indices], axis=0)))
            order = np.lexsort((indices, xyz[indices, axis]))
            ordered = indices[order]
            middle = len(ordered) // 2
            left_indices, right_indices = ordered[:middle], ordered[middle:]
            if partition == 'geometry_2means':
                # Parent-restricted geometry clustering. Initialization and tie
                # rules are deterministic, but this is NOT motion supervision.
                # Keep median fallback if a cluster violates the leaf minimum.
                centers = np.stack((xyz[left_indices].mean(0), xyz[right_indices].mean(0)))
                local = xyz[indices]
                assignment = None
                for _ in range(clustering_iterations):
                    distance = ((local[:, None, :] - centers[None, :, :]) ** 2).sum(2)
                    updated = distance[:, 1] < distance[:, 0]
                    n_right = int(updated.sum())
                    if min(n_right, len(indices) - n_right) < min_leaf_size:
                        assignment = None
                        break
                    if assignment is not None and np.array_equal(assignment, updated):
                        break
                    assignment = updated
                    centers = np.stack((local[~assignment].mean(0), local[assignment].mean(0)))
                if assignment is not None:
                    left_indices, right_indices = indices[~assignment], indices[assignment]
            nodes[index].left = visit(left_indices, depth + 1)
            nodes[index].right = visit(right_indices, depth + 1)
        return index

    visit(np.arange(len(xyz), dtype=np.int64), 0)
    return CandidateTree(nodes, len(xyz))


def _build_batched_clusters(xyz, max_depth, min_leaf_size, iterations):
    """Same parent-local Lloyd construction, batched over a depth frontier.

    Uses grouped sums instead of independent means; floating-point reductions
    may differ at decision ties. A distinct protocol name makes this explicit.
    Output numbering remains depth-first, as in the recursive reference.
    """
    groups = [np.arange(len(xyz), dtype=np.int64)]
    children = {}
    frontier = [0]
    for _ in range(max_depth):
        active = [i for i in frontier if len(groups[i]) >= 2 * min_leaf_size]
        if not active:
            break
        fallback, centers = [], []
        for index in active:
            ids = groups[index]
            axis = int(np.argmax(np.ptp(xyz[ids], axis=0)))
            ordered = ids[np.lexsort((ids, xyz[ids, axis]))]
            middle = len(ids) // 2
            a, b = ordered[:middle], ordered[middle:]
            fallback.append((a, b))
            centers.append([xyz[a].mean(0), xyz[b].mean(0)])
        centers = np.asarray(centers)
        sizes = np.array([len(groups[i]) for i in active])
        starts = np.concatenate(([0], np.cumsum(sizes)))
        ids = np.concatenate([groups[i] for i in active])
        parent = np.repeat(np.arange(len(active)), sizes)
        local = xyz[ids]
        valid = np.ones(len(active), dtype=bool)
        labels = np.zeros(len(ids), dtype=bool)
        for _ in range(iterations):
            distance_left = ((local - centers[parent, 0]) ** 2).sum(1)
            distance_right = ((local - centers[parent, 1]) ** 2).sum(1)
            labels = distance_right < distance_left
            code = parent * 2 + labels
            counts = np.bincount(code, minlength=len(active) * 2).reshape(-1, 2)
            valid &= counts.min(1) >= min_leaf_size
            sums = np.stack([np.bincount(code, weights=local[:, channel], minlength=len(active) * 2)
                             for channel in range(xyz.shape[1])], axis=1).reshape(centers.shape)
            means = sums / np.maximum(counts[:, :, None], 1)
            centers[valid] = means[valid]
        frontier = []
        for offset, index in enumerate(active):
            if valid[offset]:
                selected = labels[starts[offset]:starts[offset + 1]]
                a, b = groups[index][~selected], groups[index][selected]
            else:
                a, b = fallback[offset]
            left = len(groups)
            groups.extend((a, b))
            children[index] = (left, left + 1)
            frontier.extend((left, left + 1))
    nodes = []

    def emit(index):
        output_index = len(nodes)
        nodes.append(Node(groups[index]))
        if index in children:
            a, b = children[index]
            nodes[output_index].left = emit(a)
            nodes[output_index].right = emit(b)
        return output_index

    emit(0)
    return CandidateTree(nodes, len(xyz))


def signed_symbol_bits(value):
    """Exact signed order-0 Exp-Golomb length; zero costs one bit."""
    value = int(value)
    mapped = 2 * value - 1 if value > 0 else -2 * value
    return 2 * (mapped + 1).bit_length() - 1


def choose_symbol(value, step, quadratic_weight, rate_weight):
    """Exact scalar minimizer of .5*w*(value-Q*k)^2 + lambda*code_bits(k).

    Code length is constant on power-of-two magnitude intervals. Check the
    nearest integer in each interval and zero, instead of a huge codebook.
    Ties favor shorter codes, then smaller magnitude. Inputs are finite scalars.
    """
    if not all(math.isfinite(x) for x in (value, step, quadratic_weight, rate_weight)):
        raise ValueError('Nonfinite scalar input')
    if step <= 0 or quadratic_weight <= 0 or rate_weight < 0:
        raise ValueError('Invalid scalar objective')
    magnitude = abs(value / step)
    if not math.isfinite(magnitude) or magnitude > 2 ** 50:
        raise ValueError('Symbol range too large for this research reference')
    limit = int(math.ceil(magnitude))
    candidates = [0]
    sign = 1 if value >= 0 else -1
    low = 1
    while low <= limit:
        high = min(2 * low - 1, limit)
        nearest = min(high, max(low, int(round(magnitude))))
        candidates.append(sign * nearest)
        low *= 2

    def key(symbol):
        bits = signed_symbol_bits(symbol)
        cost = .5 * quadratic_weight * (value - step * symbol) ** 2 + rate_weight * bits
        return cost, bits, abs(symbol)

    symbol = min(candidates, key=key)
    return symbol, key(symbol)[0]


@dataclass
class Projection:
    field: np.ndarray
    root_symbols: np.ndarray
    detail_symbols: dict
    split_nodes: tuple
    quant_step: np.ndarray
    payload_bits: int
    distortion: float
    objective: float


def choose_symbols_batch(values, steps, weights, rate_weight):
    """Vectorized exact scalar quantizers; ascending bands preserve tie rules."""
    values, steps, weights = np.broadcast_arrays(np.asarray(values, dtype=np.float64),
                                                 np.asarray(steps, dtype=np.float64),
                                                 np.asarray(weights, dtype=np.float64))
    if (not all(np.isfinite(a).all() for a in (values, steps, weights))
            or (steps <= 0).any() or (weights <= 0).any()
            or not math.isfinite(rate_weight) or rate_weight < 0):
        raise ValueError('Invalid batched quantization input')
    magnitude = np.abs(values / steps)
    if not np.isfinite(magnitude).all() or (magnitude > 2 ** 50).any():
        raise ValueError('Symbol range too large for this research reference')
    limit = np.ceil(magnitude).astype(np.int64)
    symbols = np.zeros(values.shape, dtype=np.int64)
    costs = .5 * weights * values ** 2 + rate_weight
    low, band = 1, 0
    while low <= limit.max(initial=0):
        high = np.minimum(2 * low - 1, limit)
        nearest = np.minimum(high, np.maximum(low, np.rint(magnitude))).astype(np.int64)
        candidate = np.where(values >= 0, nearest, -nearest)
        trial = .5 * weights * (values - steps * candidate) ** 2 + rate_weight * (2 * band + 3)
        improve = (low <= limit) & (trial < costs)
        symbols = np.where(improve, candidate, symbols)
        costs = np.where(improve, trial, costs)
        low *= 2
        band += 1
    return symbols, costs


def reconstruct(tree, root_symbols, detail_symbols, quant_step):
    step = np.asarray(quant_step, dtype=np.float64)
    field = np.broadcast_to(np.asarray(root_symbols) * step,
                            (tree.point_count, len(step))).copy()
    for index, symbols in detail_symbols.items():
        node = tree.nodes[index]
        if node.left < 0:
            raise ValueError('Cannot split an atomic leaf')
        left, right = tree.nodes[node.left], tree.nodes[node.right]
        n_left, n_right = len(left.indices), len(right.indices)
        delta = np.asarray(symbols) * step
        field[left.indices] += n_right / (n_left + n_right) * delta
        field[right.indices] -= n_left / (n_left + n_right) * delta
    return field


def _tree_array_layout(tree):
    left = np.array([node.left for node in tree.nodes])
    right = np.array([node.right for node in tree.nodes])
    count = np.array([len(node.indices) for node in tree.nodes])
    depth = np.zeros(len(tree.nodes), dtype=np.int64)
    for index in np.flatnonzero(left >= 0):
        depth[left[index]] = depth[right[index]] = depth[index] + 1
    levels = [np.flatnonzero((depth == d) & (left >= 0)) for d in range(depth.max() + 1)]
    leaves = np.flatnonzero(left < 0)
    order = np.concatenate([tree.nodes[i].indices for i in leaves])
    starts = np.concatenate(([0], np.cumsum(count[leaves])[:-1]))
    packed_leaf = np.repeat(leaves, count[leaves])
    point_leaf = np.empty(tree.point_count, dtype=np.int64)
    point_leaf[order] = packed_leaf
    return left, right, count, levels, leaves, order, starts, packed_leaf, point_leaf


def _tree_statistics(target, rho, layout):
    """Centered leaf moments plus parallel variance decomposition by depth."""
    left, right, count, levels, leaves, order, starts, packed_leaf, _ = layout
    packed = target[order]
    means = np.zeros((len(left), target.shape[1]))
    variances = np.zeros(len(left))
    means[leaves] = np.add.reduceat(packed, starts, axis=0) / count[leaves, None]
    residual = packed - means[packed_leaf]
    variances[leaves] = np.add.reduceat(.5 * np.sum(rho * residual ** 2, axis=1), starts)
    for nodes in reversed(levels):
        a, b = left[nodes], right[nodes]
        means[nodes] = (count[a, None] * means[a] + count[b, None] * means[b]) / count[nodes, None]
        weight = count[a] * count[b] / count[nodes]
        variances[nodes] = (variances[a] + variances[b]
                           + .5 * weight * np.sum(rho * (means[a] - means[b]) ** 2, axis=1))
    return means, variances


class ProjectionWorkspace:
    """Cached indices for one immutable candidate tree; rebuild after any edit."""
    def __init__(self, tree):
        self.tree = tree
        self.layout = _tree_array_layout(tree)


def _reconstruct_by_depth(root_symbols, retained, step, layout):
    left, right, count, levels, _, _, _, _, point_leaf = layout
    values = np.zeros((len(left), len(step)))
    details = np.zeros_like(values)
    values[0] = root_symbols * step
    if retained:
        indices = np.fromiter(retained.keys(), dtype=np.int64, count=len(retained))
        details[indices] = np.stack(list(retained.values())) * step
    for nodes in levels:
        a, b = left[nodes], right[nodes]
        values[a] = values[nodes] + (count[b] / count[nodes])[:, None] * details[nodes]
        values[b] = values[nodes] - (count[a] / count[nodes])[:, None] * details[nodes]
    return values[point_leaf]


def project_rd_tree(target, tree, quant_step, rate_weight, curvature=None, *,
                    vectorized_symbols=False, vectorized_statistics=False, workspace=None,
                    fixed_cut_depth=None):
    """Select quantized root/details AND a prefix-closed emitted tree.

    target [N,C], curvature [C] is constant across points. The exact objective
    is .5*sum(curvature*(field-target)^2) + rate_weight*payload_bits. Packet
    header is constant for this call; byte padding (0..7 bits) is not optimized.
    fixed_cut_depth disables structural optimization: every eligible node above
    that depth splits, including zero-valued details. The candidate tree remains
    unchanged, so stopping flags at the cut are still encoded.
    """
    if fixed_cut_depth is not None and (isinstance(fixed_cut_depth, bool)
            or not isinstance(fixed_cut_depth, Integral) or fixed_cut_depth < 0):
        raise ValueError('fixed_cut_depth must be a nonnegative integer or None')
    target = np.asarray(target, dtype=np.float64)
    step = np.asarray(quant_step, dtype=np.float64)
    if target.ndim != 2 or target.shape[0] != tree.point_count or not target.shape[1]:
        raise ValueError('Expected target [tree.point_count,C]')
    channels = target.shape[1]
    rho = np.ones(channels) if curvature is None else np.asarray(curvature, dtype=np.float64)
    if step.shape != (channels,) or rho.shape != (channels,):
        raise ValueError('Step and curvature must have shape [C]')
    if not all(np.isfinite(x).all() for x in (target, step, rho)):
        raise ValueError('Nonfinite input')
    if (step <= 0).any() or (rho <= 0).any() or not math.isfinite(rate_weight) or rate_weight < 0:
        raise ValueError('Invalid rate, step, or curvature')
    if workspace is not None and (workspace.tree is not tree or not vectorized_statistics):
        raise ValueError('Workspace must belong to the same immutable tree and vectorized statistics')
    layout = (workspace.layout if workspace is not None else _tree_array_layout(tree)) if vectorized_statistics else None
    if vectorized_statistics:
        means, variances = _tree_statistics(target, rho, layout)
    else:
        means = [target[node.indices].mean(0) for node in tree.nodes]
        variances = [.5 * float(np.sum(rho * (target[node.indices] - mean) ** 2))
                     for node, mean in zip(tree.nodes, means)]
    best = np.zeros(len(tree.nodes))
    depths = np.zeros(len(tree.nodes), dtype=np.int64)
    if fixed_cut_depth is not None:
        for index, node in enumerate(tree.nodes):
            if node.left >= 0:
                depths[node.left] = depths[node.right] = depths[index] + 1
    take_split = np.zeros(len(tree.nodes), dtype=bool)
    detail_symbols = {}
    detail_bits = {}
    batched_costs = {}
    levelwise = vectorized_symbols and vectorized_statistics
    if vectorized_symbols:
        internal = [i for i, n in enumerate(tree.nodes) if n.left >= 0]
        if internal:
            if levelwise:
                left_ids, right_ids, counts = layout[:3]
                a, b = left_ids[internal], right_ids[internal]
                details = means[a] - means[b]
                weights = counts[a] * counts[b] / counts[internal]
            else:
                details = np.stack([means[tree.nodes[i].left] - means[tree.nodes[i].right]
                                    for i in internal])
                weights = np.array([len(tree.nodes[tree.nodes[i].left].indices)
                                    * len(tree.nodes[tree.nodes[i].right].indices)
                                    / len(tree.nodes[i].indices) for i in internal])
            symbols, costs = choose_symbols_batch(details, step, weights[:, None] * rho, rate_weight)
            detail_symbols = dict(zip(internal, symbols))
            # frexp gives exact integer bit bands for the supported <=2**50
            # range, unlike log2 which can round just-below-power-of-two values.
            bit_counts = (2 * np.frexp(np.abs(symbols))[1] + 1).sum(1)
            detail_bits = dict(zip(internal, bit_counts.tolist()))
            batched_costs = dict(zip(internal, costs.sum(1)))
    if levelwise:
        best[:] = variances
        local_cost = np.zeros(len(tree.nodes))
        if internal:
            local_cost[internal] = costs.sum(1)
        for nodes in reversed(layout[3]):
            stop = variances[nodes] + rate_weight
            split = (local_cost[nodes] + rate_weight
                     + best[layout[0][nodes]] + best[layout[1][nodes]])
            take_split[nodes] = (split < stop if fixed_cut_depth is None
                                 else depths[nodes] < fixed_cut_depth)
            best[nodes] = np.where(take_split[nodes], split, stop)
    for index in (() if levelwise else reversed(range(len(tree.nodes)))):
        node = tree.nodes[index]
        if node.left < 0:
            best[index] = variances[index]
            continue
        left, right = tree.nodes[node.left], tree.nodes[node.right]
        weight = len(left.indices) * len(right.indices) / len(node.indices)
        detail = means[node.left] - means[node.right]
        if vectorized_symbols:
            fitted_cost = batched_costs[index]
        else:
            fitted = [choose_symbol(float(detail[c]), float(step[c]), float(weight * rho[c]),
                                    rate_weight) for c in range(channels)]
            detail_symbols[index] = np.array([x[0] for x in fitted], dtype=np.int64)
            fitted_cost = sum(x[1] for x in fitted)
        # One stop/split bit at each visited, splittable candidate node.
        stop_cost = variances[index] + rate_weight
        split_cost = fitted_cost + rate_weight + best[node.left] + best[node.right]
        take_split[index] = (split_cost < stop_cost if fixed_cut_depth is None
                             else depths[index] < fixed_cut_depth)
        best[index] = split_cost if take_split[index] else stop_cost

    root_fit = [choose_symbol(float(means[0][c]), float(step[c]),
                              float(tree.point_count * rho[c]), rate_weight)
                for c in range(channels)]
    root_symbols = np.array([x[0] for x in root_fit], dtype=np.int64)
    retained = {}
    bits = sum(signed_symbol_bits(x) for x in root_symbols)

    def collect(index):
        nonlocal bits
        node = tree.nodes[index]
        if node.left < 0:
            return
        bits += 1
        if take_split[index]:
            retained[index] = detail_symbols[index]
            bits += (detail_bits[index] if vectorized_symbols
                     else sum(signed_symbol_bits(x) for x in retained[index]))
            collect(node.left)
            collect(node.right)

    collect(0)
    field = (_reconstruct_by_depth(root_symbols, retained, step, layout)
             if vectorized_statistics else reconstruct(tree, root_symbols, retained, step))
    distortion = .5 * float(np.sum(rho * (field - target) ** 2))
    objective = distortion + rate_weight * bits
    predicted = sum(x[1] for x in root_fit) + best[0]
    if not np.isclose(objective, predicted, rtol=1e-10, atol=1e-10):
        raise AssertionError('Tree recurrence disagrees with reconstructed objective')
    return Projection(field, root_symbols, retained, tuple(retained), step.copy(),
                      bits, distortion, objective)


def encode_projection(projection, tree):
    """Real prefix code; 32+8*C header bytes bind the externally shared tree.

    This is not a self-contained scene codec. The tree digest detects mismatch;
    it does not magically transmit or reconstruct a motion-learned partition.
    """
    bits = []

    def emit_symbol(value):
        value = int(value)
        mapped = 2 * value - 1 if value > 0 else -2 * value
        word = bin(mapped + 1)[2:]
        bits.extend([0] * (len(word) - 1))
        bits.extend(int(x) for x in word)

    for value in projection.root_symbols:
        emit_symbol(value)

    def visit(index):
        node = tree.nodes[index]
        if node.left < 0:
            return
        split = index in projection.detail_symbols
        bits.append(int(split))
        if split:
            for value in projection.detail_symbols[index]:
                emit_symbol(value)
            visit(node.left)
            visit(node.right)

    visit(0)
    if len(bits) != projection.payload_bits:
        raise AssertionError('Estimated and actual meaningful bit counts differ')
    channels = len(projection.quant_step)
    header = b'RDP1' + struct.pack('<III', tree.point_count, channels, len(bits))
    header += tree.fingerprint() + np.asarray(projection.quant_step, dtype='<f8').tobytes()
    return header + np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def decode_projection(packet, tree):
    if len(packet) < 32 or packet[:4] != b'RDP1':
        raise ValueError('Invalid packet')
    points, channels, count = struct.unpack('<III', packet[4:16])
    header_size = 32 + 8 * channels
    if points != tree.point_count or packet[16:32] != tree.fingerprint():
        raise ValueError('Shared candidate tree/order mismatch')
    if not channels or len(packet) != header_size + (count + 7) // 8:
        raise ValueError('Invalid packet length')
    step = np.frombuffer(packet[32:header_size], dtype='<f8').copy()
    if not np.isfinite(step).all() or (step <= 0).any():
        raise ValueError('Invalid quantization step')
    packed = np.frombuffer(packet[header_size:], dtype=np.uint8)
    unpacked = np.unpackbits(packed)
    if unpacked[count:].any():
        raise ValueError('Nonzero padding')
    bits = unpacked[:count]
    position = 0

    def read_bit():
        nonlocal position
        if position >= count:
            raise ValueError('Truncated code')
        bit = int(bits[position])
        position += 1
        return bit

    def read_symbol():
        zeros = 0
        while read_bit() == 0:
            zeros += 1
            if zeros > 52:
                raise ValueError('Unsupported integer range')
        number = 1
        for _ in range(zeros):
            number = 2 * number + read_bit()
        mapped = number - 1
        return (mapped + 1) // 2 if mapped % 2 else -(mapped // 2)

    root = np.array([read_symbol() for _ in range(channels)], dtype=np.int64)
    details = {}

    def visit(index):
        node = tree.nodes[index]
        if node.left >= 0 and read_bit():
            details[index] = np.array([read_symbol() for _ in range(channels)], dtype=np.int64)
            visit(node.left)
            visit(node.right)

    visit(0)
    if position != count:
        raise ValueError('Unconsumed payload')
    return reconstruct(tree, root, details, step)


@dataclass
class ProposalStep:
    projection: Projection
    distortion: float
    objective: float
    accepted: bool
    distortion_evaluations: int
    projection_calls: int
    chosen_scale: float | None
    trials: list


def rd_proposal_step(state, gradient, tree, curvature, rate_weight, scales,
                     distortion_fn, current_distortion=None):
    """One gradient, bounded structural proposals, true-objective acceptance.

    The callback evaluates distortion only, on identical observations for all
    proposals. This guarantees nonincrease of D + lambda*meaningful_payload_bits
    for the *evaluated* observations, not unseen views or future frames. There is
    no global optimality claim. All new callback evaluations are counted.
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    rho = np.asarray(curvature, dtype=np.float64)
    scales = tuple(float(scale) for scale in scales)
    if gradient.shape != state.field.shape or rho.shape != state.quant_step.shape:
        raise ValueError('Gradient/curvature shape mismatch')
    if not np.isfinite(gradient).all() or not np.isfinite(rho).all() or (rho <= 0).any():
        raise ValueError('Invalid gradient/curvature')
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError('Expected positive finite proposal scales')
    if not math.isfinite(rate_weight) or rate_weight < 0:
        raise ValueError('Invalid rate weight')
    evaluations = int(current_distortion is None)
    base_distortion = float(distortion_fn(state.field) if current_distortion is None
                            else current_distortion)
    if not math.isfinite(base_distortion):
        raise ValueError('Nonfinite starting distortion')
    best, best_distortion = state, base_distortion
    best_objective = base_distortion + rate_weight * state.payload_bits
    chosen_scale, trials = None, []
    cache = {state.field.tobytes(): base_distortion}
    for scale in scales:
        proposal_rho = rho * scale
        target = state.field - gradient / proposal_rho
        candidate = project_rd_tree(target, tree, state.quant_step, rate_weight, proposal_rho)
        key = candidate.field.tobytes()
        cached = key in cache
        if not cached:
            cache[key] = float(distortion_fn(candidate.field))
            evaluations += 1
        distortion = cache[key]
        objective = distortion + rate_weight * candidate.payload_bits
        trials.append({'scale': scale, 'payload_bits': candidate.payload_bits,
                       'splits': len(candidate.split_nodes), 'distortion': distortion,
                       'objective': objective, 'cached_distortion': cached})
        if math.isfinite(objective) and objective < best_objective:
            best, best_distortion, best_objective = candidate, distortion, objective
            chosen_scale = scale
    return ProposalStep(best, best_distortion, best_objective, best is not state,
                        evaluations, len(scales), chosen_scale, trials)


def tree_rate_envelope(value, tree, quant_step, rate_weight, curvature, *, optimized=False,
                       workspace=None):
    """Differentiable a.e. tree-code envelope, NOT a renderer-loss guarantee.

    E(x)=min_{T,q} .5*||x-decode(T,q)||_rho^2+lambda*bits(T,q).
    Away from ties, dE/dx=rho*(x-decode(T*,q*)). No derivative of a discrete
    topology or a trainable amplitude gate is needed. CPU projection is solely
    a research reference; this helper is not a scalable GPU implementation.
    """
    import torch

    result = project_rd_tree(value.detach().cpu().numpy(), tree, quant_step,
                             rate_weight, curvature, vectorized_symbols=optimized,
                             vectorized_statistics=optimized, workspace=workspace)
    decoded = torch.as_tensor(result.field, dtype=value.dtype, device=value.device)
    rho = torch.as_tensor(curvature, dtype=value.dtype, device=value.device)
    loss = .5 * torch.sum(rho * (value - decoded) ** 2) + rate_weight * result.payload_bits
    return loss, result
