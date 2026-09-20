"""Deterministic rollback, cache, real-rate and evaluation-count checks."""
import importlib
import numpy as np

P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')
A = importlib.import_module('pipeline.ReCon-GS.model.rd_latent_acceptance')


def test_acceptance_and_rollback():
    tree = P.build_candidate_tree(np.array([[0., 0.], [1., 0.]]), max_depth=1)
    latent = np.zeros((2, 1))
    rate = .001
    project = lambda x: P.project_rd_tree(x, tree, np.array([.01]), rate, np.ones(1))
    state = project(latent)
    calls = []
    def loss(x):
        calls.append(x.copy())
        return float(np.square(x - 1).mean())
    good = A.accept_latent_step(latent, np.ones_like(latent), state, tree, rate,
                               [1., .5], project, loss, 1.)
    assert good.accepted and good.scale == 1. and good.projection_calls == 1
    assert good.distortion_evaluations == len(calls) == 1
    assert good.objective <= 1. + rate * state.payload_bits
    np.testing.assert_array_equal(latent, np.zeros_like(latent))
    calls.clear()
    bad = A.accept_latent_step(latent, -np.ones_like(latent), state, tree, rate,
                              [1., .5], project, loss, 1.)
    assert not bad.accepted and bad.projection is state
    assert bad.projection_calls == bad.distortion_evaluations == len(calls) == 2
    np.testing.assert_array_equal(bad.latent, latent)
    assert not np.shares_memory(bad.latent, latent)
    calls.clear()
    smaller = A.accept_latent_step(latent, np.full_like(latent, 3.), state, tree, rate,
                                  [1., .25], project, loss, 1.)
    assert smaller.accepted and smaller.scale == .25
    assert smaller.projection_calls == smaller.distortion_evaluations == 2


def test_same_packet_ties_cache_and_actual_rate():
    tree = P.build_candidate_tree(np.array([[0., 0.], [1., 0.]]), max_depth=1)
    latent = np.zeros((2, 1))
    project = lambda x: P.project_rd_tree(x, tree, np.array([1.]), 0., np.ones(1))
    state = project(latent)
    def forbidden(x):
        raise AssertionError('Same packet must not trigger a render')
    result = A.accept_latent_step(latent, np.full_like(latent, .1), state, tree, 0.,
                                 [1.], project, forbidden, 1.)
    assert result.accepted and result.distortion_evaluations == 0
    np.testing.assert_array_equal(result.latent, latent + .1)
    # A decrease in distortion is insufficient when actual extra bits cost more.
    result = A.accept_latent_step(latent, np.ones_like(latent), state, tree, 1.,
                                 [1.], project, lambda x: .9, 1.)
    assert not result.accepted and result.distortion_evaluations == 1


def test_observation_cache_and_nonfinite_trials():
    cache = A.ObservationLossCache()
    assert cache.get(b'a', (1, 2)) is None
    cache.put(b'a', (1, 2), .3)
    assert cache.get(b'a', (1, 2)) == .3
    assert cache.get(b'b', (1, 2)) is None
    assert cache.get(b'a', (2, 1)) is None
    try:
        cache.put(b'a', (1, 2), float('nan'))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite cached loss was accepted')
    tree = P.build_candidate_tree(np.array([[0., 0.], [1., 0.]]), max_depth=1)
    latent = np.zeros((2, 1))
    project = lambda x: P.project_rd_tree(x, tree, np.array([1.]), 0., np.ones(1))
    result = A.accept_latent_step(latent, np.ones_like(latent), project(latent), tree, 0.,
                                 [1., 1.], project, lambda x: float('nan'), 1.)
    assert not result.accepted and result.distortion == 1.
    assert result.projection_calls == 2 and result.distortion_evaluations == 1
    assert result.trials[1]['cached_distortion'] and result.trials[1]['objective'] is None


def run_all():
    test_acceptance_and_rollback()
    test_same_packet_ties_cache_and_actual_rate()
    test_observation_cache_and_nonfinite_trials()
