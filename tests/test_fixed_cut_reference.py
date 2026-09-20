"""CPU-only oracle checks, run separately while the frozen full run executes."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fixed_cut_reference import P, project_fixed_cut


def test_zero_details_do_not_prune():
    tree = P.build_candidate_tree(np.arange(5)[:, None], max_depth=3)
    target = np.zeros((5, 2))
    counts = []
    for depth in range(5):
        result = project_fixed_cut(target, tree, [.1, .2], 1., [1., 2.], depth)
        counts.append(len(result.split_nodes))
        assert all(not s.any() for s in result.detail_symbols.values())
        torch.testing.assert_close(torch.from_numpy(result.field), torch.from_numpy(target),
                                   rtol=0, atol=0)
    assert counts == [0, 1, 3, 4, 4]
    assert not P.project_rd_tree(target, tree, [.1, .2], 1., [1., 2.]).split_nodes


def test_adaptive_objective_bounds_fixed_cuts():
    rng = np.random.default_rng(719)
    tree = P.build_candidate_tree(np.arange(9)[:, None], max_depth=4)
    for rate in (0., .001, .1, 10.):
        target = rng.normal(size=(9, 3))
        step, rho = [.1, .2, .3], [.2, 1., 3.]
        adaptive = P.project_rd_tree(target, tree, step, rate, rho)
        for depth in range(6):
            fixed = project_fixed_cut(target, tree, step, rate, rho, depth)
            assert adaptive.objective <= fixed.objective + 1e-10
        if rate == 0:
            torch.testing.assert_close(torch.tensor(adaptive.objective),
                                       torch.tensor(fixed.objective), rtol=1e-6, atol=1e-8)


def test_production_fixed_cut_matches_oracle():
    rng = np.random.default_rng(937)
    tree = P.build_candidate_tree(np.arange(11)[:, None], max_depth=4)
    workspace = P.ProjectionWorkspace(tree)
    for rate in (0., .01, 1.):
        for target in (np.zeros((11, 2)), rng.normal(size=(11, 2))):
            for depth in range(6):
                oracle = project_fixed_cut(target, tree, [.125, .25], rate, [.4, 2.], depth)
                for vectorized_symbols, vectorized_statistics in ((False, False), (True, False),
                                                                  (False, True), (True, True)):
                    result = P.project_rd_tree(
                        target, tree, [.125, .25], rate, [.4, 2.], fixed_cut_depth=depth,
                        vectorized_symbols=vectorized_symbols,
                        vectorized_statistics=vectorized_statistics,
                        workspace=workspace if vectorized_statistics else None)
                    assert result.split_nodes == oracle.split_nodes
                    assert P.encode_projection(result, tree) == P.encode_projection(oracle, tree)
                    torch.testing.assert_close(torch.tensor(result.objective),
                                               torch.tensor(oracle.objective), rtol=1e-6, atol=1e-8)
    for invalid in (-1, 1.5, True, '2'):
        try:
            P.project_rd_tree(np.zeros((11, 2)), tree, [.1, .1], 0., fixed_cut_depth=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid fixed cut accepted')


if __name__ == '__main__':
    test_zero_details_do_not_prune()
    test_adaptive_objective_bounds_fixed_cuts()
    test_production_fixed_cut_matches_oracle()
    print('Fixed-cut reference checks passed.')
