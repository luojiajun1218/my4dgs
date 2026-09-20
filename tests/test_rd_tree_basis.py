"""Deterministic coordinate, weighted orthogonality, and gradient checks."""
import importlib
import numpy as np
import torch

P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')
B = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_basis')
F = importlib.import_module('scripts.gradient_ema')


def test_tree_motion_basis_coordinates_and_gradient():
    torch.manual_seed(1749)
    xyz = torch.randn(23, 3, dtype=torch.float64).numpy()
    for depth in (0, 4):
        tree = P.build_candidate_tree(xyz, max_depth=depth, min_leaf_size=2)
        basis = B.TreeMotionBasis(tree)
        coefficients = torch.randn(basis.coefficient_count, 3, dtype=torch.float64, requires_grad=True)
        field = basis(coefficients)
        detached = coefficients.detach().numpy()
        reference = P.reconstruct(tree, detached[0],
                                  {node: detached[i + 1] for i, node in enumerate(basis.internal_nodes)},
                                  np.ones(3))
        torch.testing.assert_close(field, torch.as_tensor(reference), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(field.mean(0), coefficients[0], rtol=1e-12, atol=1e-12)
        matrix = basis(torch.eye(basis.coefficient_count, dtype=torch.float64))
        torch.testing.assert_close(matrix.T @ matrix, torch.diag(basis.energy_weights),
                                   rtol=1e-12, atol=1e-12)
        assert torch.autograd.gradcheck(basis, (coefficients,), eps=1e-6, atol=1e-6, rtol=1e-5)
        gradient = torch.randn_like(field)
        actual = torch.autograd.grad((field * gradient).sum(), coefficients, retain_graph=True)[0]
        torch.testing.assert_close(actual, matrix.T @ gradient, rtol=1e-12, atol=1e-12)
        projected_gradient = matrix @ (actual / basis.energy_weights[:, None])
        expected_leaf_means = torch.empty_like(gradient)
        for node in tree.nodes:
            if node.left < 0:
                expected_leaf_means[node.indices] = gradient[node.indices].mean(0)
        torch.testing.assert_close(projected_gradient, expected_leaf_means, rtol=1e-12, atol=1e-12)
        target = torch.randn_like(field)
        rho = torch.tensor([.2, .7, 1.3], dtype=torch.float64)
        envelope, projection = P.tree_rate_envelope(field, tree, np.full(3, .01), .003, rho.numpy())
        derivative = torch.autograd.grad(envelope, coefficients)[0]
        expected = matrix.T @ (rho * (field.detach() - torch.as_tensor(projection.field)))
        torch.testing.assert_close(derivative, expected, rtol=1e-12, atol=1e-12)
        # Energy-normalized tree SGD and dense SGD have identical encodable
        # atomic means; within-atom variance cannot change a tree-cut decision.
        dense_step = -.03 * gradient / rho
        tree_step = basis(-.03 * actual / (basis.energy_weights[:, None] * rho))
        dense_projection = P.project_rd_tree(dense_step.numpy(), tree, np.full(3, .01), .003, rho.numpy())
        tree_projection = P.project_rd_tree(tree_step.detach().numpy(), tree, np.full(3, .01), .003,
                                            rho.numpy())
        assert P.encode_projection(dense_projection, tree) == P.encode_projection(tree_projection, tree)
        assert basis.coefficient_count == sum(n.left < 0 for n in tree.nodes)


def test_atomic_nullspace_cannot_change_sgd_stream():
    """Large within-atom latent residuals do not imply large coded motion.

    This equivalence assumes fixed geometry, shared per-channel curvature and
    SGD step size. Nonlinear per-point clipping/Adam need not commute with the
    atomic averaging operator and are intentionally outside the claim.
    """
    generator = torch.Generator().manual_seed(62109)
    xyz = torch.randn(47, 3, generator=generator, dtype=torch.float64).numpy()
    tree = P.build_candidate_tree(xyz, max_depth=4, min_leaf_size=3)
    basis = B.TreeMotionBasis(tree)
    matrix = basis(torch.eye(basis.coefficient_count, dtype=torch.float64))
    weights = basis.energy_weights[:, None]

    def atomic_mean(value):
        return matrix @ ((matrix.T @ value) / weights)

    rho = torch.tensor([.2, .3, .4, 1., 2., 3.], dtype=torch.float64)
    quant = np.full(6, .01)
    for fixed_cut in (None, 4):
        compact = atomic_mean(torch.randn(47, 6, generator=generator, dtype=torch.float64))
        noise = torch.randn(47, 6, generator=generator, dtype=torch.float64)
        dense = compact + 100. * (noise - atomic_mean(noise))
        for _ in range(12):
            kwargs = dict(fixed_cut_depth=fixed_cut)
            left = P.project_rd_tree(dense.numpy(), tree, quant, .003, rho.numpy(), **kwargs)
            right = P.project_rd_tree(compact.numpy(), tree, quant, .003, rho.numpy(), **kwargs)
            assert P.encode_projection(left, tree) == P.encode_projection(right, tree)
            torch.testing.assert_close(atomic_mean(dense), compact, rtol=1e-10, atol=1e-10)
            # A deterministic changing renderer-gradient stand-in: identical
            # decoded states receive identical full per-point gradient arrays.
            decoded = torch.as_tensor(left.field)
            gradient = torch.randn(47, 6, generator=generator, dtype=torch.float64)
            gradient = .03 * gradient + .01 * decoded.sin()
            dense -= .5 / rho * (gradient + rho * (dense - decoded))
            compact -= .5 / rho * (atomic_mean(gradient) + rho * (compact - decoded))


def run_all():
    test_tree_motion_basis_coordinates_and_gradient()
    test_atomic_nullspace_cannot_change_sgd_stream()
    test_renderer_gradient_ema()


def test_renderer_gradient_ema():
    rng = np.random.default_rng(582)
    gradients = rng.normal(size=(9, 7, 6))
    for beta in (0., .5, .75):
        filt = F.GradientEMA(beta)
        for index, gradient in enumerate(gradients):
            before = gradient.copy()
            actual = filt.update(gradient)
            weights = beta ** np.arange(index, -1, -1)
            expected = np.einsum('i,ijk->jk', weights / weights.sum(), gradients[:index + 1])
            torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected),
                                       rtol=1e-12, atol=1e-12)
            assert np.array_equal(before, gradient)
            if beta == 0.:
                assert np.array_equal(actual, gradient)
    for invalid in (-.1, 1., float('nan')):
        try:
            F.GradientEMA(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid EMA beta accepted')
    # Shared beta preserves atomic averaging: unlike pointwise clipping, this
    # filter cannot mix within-atom zero-mean signals into encodable means.
    point_filter, mean_filter = F.GradientEMA(.75), F.GradientEMA(.75)
    for gradient in gradients:
        filtered_points = point_filter.update(gradient)
        filtered_mean = mean_filter.update(gradient.mean(axis=0))
        torch.testing.assert_close(torch.from_numpy(filtered_points.mean(axis=0)),
                                   torch.from_numpy(filtered_mean), rtol=1e-12, atol=1e-12)
    for invalid_gradient in (np.zeros((8, 6)), np.full((7, 6), np.nan)):
        count_before = point_filter.updates
        state_before = point_filter.mean.copy()
        try:
            point_filter.update(invalid_gradient)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid gradient accepted')
        assert point_filter.updates == count_before
        assert np.array_equal(point_filter.mean, state_before)


if __name__ == '__main__':
    run_all()
    print('Tree motion basis checks passed')
