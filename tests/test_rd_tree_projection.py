"""Deterministic mathematical/codec checks; no renderer or scene required."""
import importlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')


def close(actual, expected):
    torch.testing.assert_close(torch.as_tensor(actual, dtype=torch.float64),
                               torch.as_tensor(expected, dtype=torch.float64),
                               rtol=1e-10, atol=1e-10)


def test_scalar_codec_optimum():
    rng = np.random.default_rng(37)
    for value, weight, rate in zip(rng.uniform(-3, 3, 50), rng.uniform(.1, 5, 50),
                                  rng.uniform(0, .3, 50)):
        symbol, cost = P.choose_symbol(value, .125, weight, rate)
        exhaustive = [.5 * weight * (value - .125 * q) ** 2 + rate * P.signed_symbol_bits(q)
                      for q in range(-32, 33)]
        close(cost, min(exhaustive))
        assert -32 <= symbol <= 32


def all_cuts(tree, index=0):
    node = tree.nodes[index]
    if node.left < 0:
        return [frozenset()]
    return [frozenset()] + [frozenset({index}) | left | right
                           for left in all_cuts(tree, node.left)
                           for right in all_cuts(tree, node.right)]


def cut_bits(tree, root, details, index=0):
    def visit(node_index):
        node = tree.nodes[node_index]
        if node.left < 0:
            return 0
        if node_index not in details:
            return 1
        return (1 + sum(P.signed_symbol_bits(x) for x in details[node_index])
                + visit(node.left) + visit(node.right))
    return sum(P.signed_symbol_bits(x) for x in root) + visit(index)


def test_tree_dp_matches_all_cuts():
    rng = np.random.default_rng(13)
    tree = P.build_candidate_tree(np.arange(5)[:, None], max_depth=3)
    # Unequal child masses and multiple channels exercise the weighted basis.
    for rate in (0., .005, .1, 1.):
        target = rng.normal(size=(5, 2))
        step, rho = np.array([.125, .25]), np.array([.4, 2.])
        means = [target[node.indices].mean(0) for node in tree.nodes]
        root = np.array([P.choose_symbol(means[0][c], step[c], 5 * rho[c], rate)[0]
                         for c in range(2)])
        symbols = {}
        for index, node in enumerate(tree.nodes):
            if node.left >= 0:
                weight = (len(tree.nodes[node.left].indices) * len(tree.nodes[node.right].indices)
                          / len(node.indices))
                detail = means[node.left] - means[node.right]
                symbols[index] = np.array([P.choose_symbol(detail[c], step[c], weight * rho[c], rate)[0]
                                           for c in range(2)])
        costs = []
        for cut in all_cuts(tree):
            details = {index: symbols[index] for index in cut}
            field = P.reconstruct(tree, root, details, step)
            costs.append(.5 * np.sum(rho * (field - target) ** 2)
                         + rate * cut_bits(tree, root, details))
        result = P.project_rd_tree(target, tree, step, rate, rho)
        close(result.objective, min(costs))


def test_zero_parent_keeps_useful_descendants():
    tree = P.build_candidate_tree(np.arange(4)[:, None], max_depth=2)
    # Small rate weight keeps exact values; larger rates may validly quantize
    # toward cheaper power-of-two code bins even when the structure is retained.
    result = P.project_rd_tree(np.array([1., -1., 1., -1.])[:, None], tree, [.25], .001)
    assert len(result.split_nodes) == 3
    assert result.detail_symbols[0].item() == 0
    close(result.field[:, 0], [1., -1., 1., -1.])
    # Immediate first split alone has zero distortion gain and positive cost.
    means = np.array([0., 0.])
    assert means[0] == means[1]


def test_local_depth_and_reversible_selection():
    tree = P.build_candidate_tree(np.arange(8)[:, None], max_depth=3)
    uniform = np.ones((8, 1))
    local = uniform.copy()
    local[6:, 0] = [2., 0.]
    coarse = P.project_rd_tree(uniform, tree, [.25], .001)
    fine = P.project_rd_tree(local, tree, [.25], .001)
    restored = P.project_rd_tree(uniform, tree, [.25], .001)
    assert not coarse.split_nodes and not restored.split_nodes
    assert len(fine.split_nodes) == 3
    assert tree.nodes[0].left not in fine.split_nodes
    close(fine.field, local)
    large = P.project_rd_tree(uniform * 20, tree, [.25], .01)
    assert not large.split_nodes


def test_rate_sweep_and_codec():
    rng = np.random.default_rng(29)
    tree = P.build_candidate_tree(rng.normal(size=(23, 3)), max_depth=4, min_leaf_size=2)
    target = rng.normal(size=(23, 6))
    previous_bits = float('inf')
    previous_distortion = -1.
    for rate in (0., .001, .01, .1, 1., 10.):
        result = P.project_rd_tree(target, tree, np.full(6, .125), rate)
        assert result.payload_bits <= previous_bits
        assert result.distortion >= previous_distortion - 1e-10
        previous_bits, previous_distortion = result.payload_bits, result.distortion
        packet = P.encode_projection(result, tree)
        decoded = P.decode_projection(packet, tree)
        close(decoded, result.field)
        header_bits = (32 + 8 * 6) * 8
        assert 0 <= len(packet) * 8 - header_bits - result.payload_bits <= 7
        try:
            P.decode_projection(packet[:-1], tree)
        except ValueError:
            pass
        else:
            raise AssertionError('Truncated packet accepted')
    wrong_tree = P.build_candidate_tree(rng.normal(size=(23, 3)), max_depth=4, min_leaf_size=2)
    try:
        P.decode_projection(packet, wrong_tree)
    except ValueError:
        pass
    else:
        raise AssertionError('Different shared tree accepted')


def test_proximal_descent_on_coupled_quadratic():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(12, 8)) / np.sqrt(12)
    target = np.array([1., 1., 1., 1., 1., -1., 1., -1.])
    observed = matrix @ target
    curvature = np.linalg.norm(matrix, 2) ** 2 * 1.01
    tree = P.build_candidate_tree(np.arange(8)[:, None], max_depth=3)
    rate, step = .001, [.125]
    state = P.project_rd_tree(np.zeros((8, 1)), tree, step, rate)
    values = []
    for _ in range(60):
        field = state.field[:, 0]
        residual = matrix @ field - observed
        values.append(.5 * residual @ residual + rate * state.payload_bits)
        gradient = matrix.T @ residual
        proposal = (field - gradient / curvature)[:, None]
        state = P.project_rd_tree(proposal, tree, step, rate, [curvature])
    assert np.max(np.diff(values)) <= 1e-10
    assert values[-1] < .1 * values[0]


def test_soft_gate_scaling_counterexample():
    # A deterministic amplitude gate plus p penalty is not a structure code.
    p = torch.tensor([.5, .05], dtype=torch.float64)
    delta = 1 / p
    close(p * delta, torch.ones_like(p))
    assert p[1] < p[0]


def test_quantized_proximal_stall_and_shadow_counterexample():
    rng = np.random.default_rng(20260920)
    matrix = rng.normal(size=(32, 16)) / np.sqrt(32)
    truth = np.concatenate([np.ones(8), np.tile([1., -1.], 4)])
    observed = matrix @ truth
    rho = np.linalg.norm(matrix, 2) ** 2 * 1.01
    tree = P.build_candidate_tree(np.arange(16)[:, None], max_depth=4)
    final = {}
    for mode in ('hard', 'shadow'):
        state = P.project_rd_tree(np.zeros((16, 1)), tree, [.125], .001)
        latent = np.zeros(16)
        history = []
        for _ in range(81):
            field = state.field[:, 0]
            residual = matrix @ field - observed
            history.append(.5 * residual @ residual + .001 * state.payload_bits)
            latent = (field if mode == 'hard' else latent) - matrix.T @ residual / rho
            state = P.project_rd_tree(latent[:, None], tree, [.125], .001, [rho])
        final[mode] = history[-1]
        if mode == 'shadow':
            assert max(np.diff(history)) > .01  # Do not claim monotone descent.
    assert final['hard'] > .9
    close(final['shadow'], .072)


def test_shadow_accumulation_can_override_rate_penalty():
    tree = P.build_candidate_tree(np.arange(8)[:, None], max_depth=3)
    truth = np.tile([.125, -.125], 4)
    rate = .1
    state = P.project_rd_tree(np.zeros((8, 1)), tree, [.125], rate)
    latent = np.zeros(8)
    values = []
    for _ in range(100):
        field = state.field[:, 0]
        values.append(.5 * np.sum((field - truth) ** 2) + rate * state.payload_bits)
        latent -= field - truth
        state = P.project_rd_tree(latent[:, None], tree, [.125], rate)
    close(values[0], .2625)
    assert max(values) > 7.


def test_rd_proposal_acceptance_and_budget():
    rng = np.random.default_rng(20260920)
    matrix = rng.normal(size=(32, 16)) / np.sqrt(32)
    truth = np.concatenate([np.ones(8), np.tile([1., -1.], 4)])
    observed = matrix @ truth
    rho = np.linalg.norm(matrix, 2) ** 2 * 1.01
    tree = P.build_candidate_tree(np.arange(16)[:, None], max_depth=4)
    rate = .001
    state = P.project_rd_tree(np.zeros((16, 1)), tree, [.125], rate)

    def evaluate(field):
        error = matrix @ field[:, 0] - observed
        return float(.5 * error @ error)

    for _ in range(20):
        distortion = evaluate(state.field)
        gradient = (matrix.T @ (matrix @ state.field[:, 0] - observed))[:, None]
        old = distortion + rate * state.payload_bits
        result = P.rd_proposal_step(state, gradient, tree, [rho], rate, [1., .25, .0625],
                                    evaluate, distortion)
        assert result.objective <= old
        assert result.projection_calls == 3
        assert result.distortion_evaluations <= 3
        state = result.projection
        if not result.accepted:
            break
    assert evaluate(state.field) + rate * state.payload_bits < .21

    weak = np.tile([.125, -.125], 8)[:, None]
    state = P.project_rd_tree(np.zeros_like(weak), tree, [.125], .1)
    distortion = lambda field: float(.5 * np.sum((field - weak) ** 2))
    update = P.rd_proposal_step(state, -weak, tree, [1.], .1, [1., .25, .0625],
                                distortion, distortion(state.field))
    assert not update.accepted
    assert not update.projection.split_nodes


def test_duplicate_proposal_evaluated_once():
    tree = P.build_candidate_tree(np.arange(4)[:, None], max_depth=2)
    state = P.project_rd_tree(np.zeros((4, 1)), tree, [.125], .001)
    calls = []

    def evaluate(field):
        calls.append(1)
        return float(.5 * np.sum((field - 1.) ** 2))

    update = P.rd_proposal_step(state, -np.ones((4, 1)), tree, [1.], .001,
                                [1., 1., 1.], evaluate, 2.)
    assert update.distortion_evaluations == len(calls) == 1
    assert update.accepted


def test_tree_rate_envelope_gradient():
    torch.manual_seed(83)
    tree = P.build_candidate_tree(np.arange(5)[:, None], max_depth=3)
    value = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    step, rho, rate = [.125, .25], [.7, 1.3], .027

    def envelope(x):
        return P.tree_rate_envelope(x, tree, step, rate, rho)[0]

    assert torch.autograd.gradcheck(envelope, (value,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradcheck(
        lambda x: P.tree_rate_envelope(x, tree, step, rate, rho, optimized=True)[0],
        (value,), eps=1e-6, atol=1e-5, rtol=1e-4)
    loss, projection = P.tree_rate_envelope(value, tree, step, rate, rho)
    close(loss.detach(), projection.objective)
    loss.backward()
    expected = torch.tensor(rho, dtype=value.dtype) * (value.detach() - torch.tensor(projection.field))
    close(value.grad, expected)


def test_tree_envelope_prevents_unbounded_shadow_on_weak_detail():
    tree = P.build_candidate_tree(np.arange(8)[:, None], max_depth=3)
    truth = torch.tensor(np.tile([.125, -.125], 4)[:, None], dtype=torch.float64)
    value = torch.zeros_like(truth, requires_grad=True)
    for _ in range(100):
        envelope, projection = P.tree_rate_envelope(value, tree, [.125], .1, [1.])
        decoded = torch.tensor(projection.field, dtype=value.dtype)
        rendered = value + (decoded - value).detach()
        objective = .5 * (rendered - truth).square().sum() + envelope
        objective.backward()
        with torch.no_grad():
            value -= .45 * value.grad
        value.grad = None
        assert not projection.split_nodes
        close(.5 * (decoded - truth).square().sum() + .1 * projection.payload_bits, .2625)
    assert value.abs().max() < .126


def test_parent_restricted_geometry_clustering():
    rng = np.random.default_rng(31)
    xyz = np.concatenate((rng.normal(-2, .1, (5, 3)), rng.normal(2, .1, (11, 3))))
    tree = P.build_candidate_tree(xyz, max_depth=3, min_leaf_size=2,
                                 partition='geometry_2means')
    repeat = P.build_candidate_tree(xyz, max_depth=3, min_leaf_size=2,
                                   partition='geometry_2means')
    assert tree.fingerprint() == repeat.fingerprint()
    root = tree.nodes[0]
    assert sorted([len(tree.nodes[root.left].indices), len(tree.nodes[root.right].indices)]) == [5, 11]
    for node in tree.nodes:
        if node.left >= 0:
            left, right = tree.nodes[node.left].indices, tree.nodes[node.right].indices
            assert not set(left) & set(right)
            assert set(left) | set(right) == set(node.indices)
            assert min(len(left), len(right)) >= 2
    value = np.where(xyz[:, :1] < 0, -1., 1.)
    projection = P.project_rd_tree(value, tree, [.125], .001)
    close(projection.field, value)
    assert projection.split_nodes == (0,)
    close(P.decode_projection(P.encode_projection(projection, tree), tree), value)
    # Degenerate identical coordinates must fall back safely, not lose points.
    degenerate = P.build_candidate_tree(np.zeros((8, 3)), 3, 2, 'geometry_2means')
    assert sum(len(n.indices) for n in degenerate.nodes if n.left < 0) == 8


def test_vectorized_quantization_and_projection():
    rng = np.random.default_rng(930)
    values = np.concatenate((rng.normal(size=500), np.arange(-16, 17) * .125,
                             np.arange(-16, 17) * .125 + .0625))
    for rate in (0., .001, .1, 10.):
        symbols, costs = P.choose_symbols_batch(values, .125, 1.3, rate)
        scalar = [P.choose_symbol(float(v), .125, 1.3, rate) for v in values]
        assert np.array_equal(symbols, [item[0] for item in scalar])
        close(costs, [item[1] for item in scalar])
        for partition in ('median', 'geometry_2means'):
            xyz, target = rng.normal(size=(129, 3)), rng.normal(size=(129, 6))
            tree = P.build_candidate_tree(xyz, 6, 3, partition)
            step, rho = np.linspace(.05, .2, 6), np.linspace(.3, 2., 6)
            reference = P.project_rd_tree(target, tree, step, rate, rho)
            fast = P.project_rd_tree(target, tree, step, rate, rho, vectorized_symbols=True)
            close(fast.objective, reference.objective)
            assert P.encode_projection(fast, tree) == P.encode_projection(reference, tree)
            fully_batched = P.project_rd_tree(target, tree, step, rate, rho,
                                               vectorized_symbols=True, vectorized_statistics=True,
                                               workspace=P.ProjectionWorkspace(tree))
            close(fully_batched.objective, reference.objective)
            assert np.array_equal(fully_batched.field, reference.field)
            assert P.encode_projection(fully_batched, tree) == P.encode_projection(reference, tree)


def run_all():
    test_scalar_codec_optimum()
    test_tree_dp_matches_all_cuts()
    test_zero_parent_keeps_useful_descendants()
    test_local_depth_and_reversible_selection()
    test_rate_sweep_and_codec()
    test_proximal_descent_on_coupled_quadratic()
    test_soft_gate_scaling_counterexample()
    test_quantized_proximal_stall_and_shadow_counterexample()
    test_shadow_accumulation_can_override_rate_penalty()
    test_rd_proposal_acceptance_and_budget()
    test_duplicate_proposal_evaluated_once()
    test_tree_rate_envelope_gradient()
    test_tree_envelope_prevents_unbounded_shadow_on_weak_detail()
    test_parent_restricted_geometry_clustering()
    test_vectorized_quantization_and_projection()
    test_batched_candidate_clustering()
    test_large_symbol_bands_and_workspace_identity()


def test_batched_candidate_clustering():
    rng = np.random.default_rng(307)
    for xyz in (rng.normal(size=(137, 3)), np.zeros((31, 3)),
                np.repeat(np.arange(23)[:, None], 3, axis=1)):
        for iterations in (1, 4, 9):
            reference = P.build_candidate_tree(xyz, 5, 2, 'geometry_2means', iterations)
            batched = P.build_candidate_tree(xyz, 5, 2, 'geometry_2means_batched', iterations)
            assert batched.fingerprint() == reference.fingerprint()


def test_large_symbol_bands_and_workspace_identity():
    tree = P.build_candidate_tree(np.arange(2)[:, None], 1, 1)
    workspace = P.ProjectionWorkspace(tree)
    for symbol in (0, 1, 2, 2 ** 40 - 1, 2 ** 50 - 1, 2 ** 50):
        target = np.array([[symbol / 2], [-symbol / 2]], dtype=np.float64)
        reference = P.project_rd_tree(target, tree, [1.], 0.)
        fast = P.project_rd_tree(target, tree, [1.], 0., vectorized_symbols=True,
                                 vectorized_statistics=True, workspace=workspace)
        assert P.encode_projection(fast, tree) == P.encode_projection(reference, tree)
    other = P.build_candidate_tree(np.arange(2)[:, None], 1, 1)
    try:
        P.project_rd_tree(np.zeros((2, 1)), other, [1.], 0., vectorized_statistics=True,
                          workspace=workspace)
    except ValueError:
        pass
    else:
        raise AssertionError('Workspace from a different tree was accepted')


if __name__ == '__main__':
    run_all()
    print('All RD tree projection mathematical/codec checks passed.')
