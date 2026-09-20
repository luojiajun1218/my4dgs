"""Numerical checks without loading the renderer or a scene."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
D = importlib.import_module('pipeline.ReCon-GS.model.deformation')
N = importlib.import_module('pipeline.ReCon-GS.model.nested')
B = importlib.import_module('pipeline.ReCon-GS.model.binary_motion')
C = importlib.import_module('pipeline.ReCon-GS.model.motion_codec')


def test_variance():
    g = torch.tensor([[1.,0,0],[-1.,0,0],[2.,0,0],[2.,0,0],[99.,0,0]])
    parent = torch.tensor([0,0,1,1,2])
    h, count = N.visible_gradient_variance(g, parent, torch.tensor([1,1,1,1,0]), 4)
    torch.testing.assert_close(h, torch.tensor([1.,0,0,0]))
    assert count.tolist() == [2,2,0,0]


def test_nested_sampling():
    xyz = torch.tensor([[0.,0,0],[1.,0,0],[2.,0,0],[10.,0,0],[11.,0,0],[20.,0,0]])
    parent = torch.tensor([0,0,0,1,1,2])
    fine = N.NestedFineGrid(xyz, parent, 4, 3)
    assert fine.vertex_num == 6
    assert torch.equal(fine.parent[fine.p2v.flatten()], parent)
    assert torch.unique(fine.vertex, dim=0).shape[0] == 6
    assert fine.vertex_idx[0].item() == 1  # First parent starts nearest its centroid.
    supported = fine.blend_weights > 0
    assert torch.equal(fine.parent[fine.blend_indices[supported]],
                       parent[:, None].expand_as(supported)[supported])
    torch.testing.assert_close(fine.blend_weights.sum(-1), torch.ones(len(xyz)))
    # Coincident coordinates produce one spatial child per nonempty parent.
    degenerate = N.NestedFineGrid(torch.zeros_like(xyz), parent, 4, 3)
    assert torch.equal(degenerate.parent[degenerate.p2v.flatten()], parent)
    assert degenerate.vertex_num == 3
    assert not (degenerate.blend_weights > 0).sum(-1).gt(1).any()


def test_overlap_width_and_junction():
    vertex = torch.tensor([[0., 0, 0], [1., 0, 0]], dtype=torch.float64)
    children = torch.tensor([[0, 1]])
    xyz = torch.zeros(7, 3, dtype=torch.float64)
    xyz[:, 0] = torch.tensor([.3, .425, .4625, .5, .5375, .575, .7], dtype=torch.float64)
    parent = torch.zeros(7, dtype=torch.long)
    assignment = (xyz[:, 0] > .5).long()
    weights = N.boundary_blend_weights(xyz, parent, vertex, children, assignment, .15)
    expected = torch.tensor([[1., 0], [1, 0], [.84375, .15625], [.5, .5],
                             [.15625, .84375], [0, 1], [0, 1]], dtype=torch.float64)
    torch.testing.assert_close(weights, expected)
    hard = N.boundary_blend_weights(xyz, parent, vertex, children, assignment, 0)
    torch.testing.assert_close(hard.sum(-1), torch.ones(7, dtype=torch.float64))
    assert (hard > 0).sum(-1).eq(1).all()
    # All three siblings participate smoothly at a three-way junction.
    root3 = 3 ** .5
    vertex = torch.tensor([[-1., 0, 0], [1., 0, 0], [0, root3, 0]], dtype=torch.float64)
    xyz = torch.tensor([[-1e-6, root3 / 3, 0], [0, root3 / 3, 0],
                        [1e-6, root3 / 3, 0]], dtype=torch.float64)
    weights = N.boundary_blend_weights(xyz, torch.zeros(3, dtype=torch.long), vertex,
                                       torch.tensor([[0, 1, 2]]), torch.zeros(3, dtype=torch.long), .15)
    torch.testing.assert_close(weights[1], torch.full((3,), 1 / 3, dtype=torch.float64))
    assert (weights[0] - weights[2]).abs().max() < 1e-3


def test_weighted_diagnostics():
    fine = SimpleNamespace(vertex_num=2, blend_indices=torch.tensor([[0, 1]] * 4),
                           blend_weights=torch.tensor([[1., 0], [.75, .25], [.25, .75], [0, 1]]))
    gradient = torch.tensor([[1., 0, 0], [1., 0, 0], [-1., 0, 0], [-1., 0, 0]])
    parent = torch.zeros(4, dtype=torch.long)
    before, after, count = N.weighted_gradient_diagnostics(
        gradient, parent, torch.ones(4, dtype=torch.bool), 2, fine)
    torch.testing.assert_close(before, torch.tensor([1., 0]))
    torch.testing.assert_close(after, torch.tensor([.4375, 0]))
    assert count.tolist() == [4, 0]
    # Identical gradients remain zero variance even with shared memberships.
    before, after, _ = N.weighted_gradient_diagnostics(
        torch.ones_like(gradient), parent, torch.ones(4, dtype=torch.bool), 2, fine)
    assert not before.any() and not after.any()
    gradient[0] = float('nan')
    before, after, count = N.weighted_gradient_diagnostics(
        gradient, parent, torch.tensor([True, False, False, True]), 2, fine)
    assert count.tolist() == [1, 0]
    assert not before.any() and not after.any()


def test_schedule_and_reset():
    torch.manual_seed(0)
    xyz = torch.randn(80, 3)
    deform = D.Deformation(D.DeformConfig(hierarchy_mode='nested', max_gs_per_grid=8,
                                         grid_level_ratio=3, momentum=None))
    deform.setup(xyz)
    n_fine = deform.grids[0].vertex_num
    identity = deform.delta[:n_fine].detach().clone()
    for step in range(1, 51):
        delta, _ = deform(xyz)
        if deform.collecting_gradients(step):
            delta.retain_grad()
        loss = (delta - xyz * .1).square().mean() + deform.reg_loss()
        loss.backward()
        if deform.collecting_gradients(step):
            deform.accumulate_gradient_score(delta.grad, torch.ones(80, dtype=torch.bool), step=step)
        deform.optimizer.step()
        deform.optimizer.zero_grad(set_to_none=True)
    torch.testing.assert_close(deform.delta[:n_fine], identity)
    assert not deform.fine_active_mask.any()
    assert deform.score_steps == list(range(31, 51))
    assert (deform.score_after_sum <= deform.score_sum + 1e-12).all()
    with torch.no_grad():
        # Nonidentity parent rotation exposes accidental addition of Fine identity.
        deform.delta[n_fine:, 4] += .2
    before_xyz, before_rot = deform(xyz)
    before_reg = deform.reg_loss().detach().clone()
    stats = deform.activate_refinement()
    after_xyz, after_rot = deform(xyz)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=0)
    torch.testing.assert_close(after_rot, before_rot, rtol=0, atol=0)
    torch.testing.assert_close(deform.reg_loss(), before_reg, rtol=0, atol=0)
    assert {'H_before', 'H_after', 'H_reduction', 'fine_shared_ratio'} <= stats.keys()
    assert deform.middle_active_mask.sum().item() == int((torch.bincount(deform.middle_parent_of_gs)>0).sum()*.5)
    delta, _ = deform(xyz)
    (delta - xyz * .1).square().mean().backward()
    assert not deform.delta.grad[:n_fine][~deform.fine_active_mask].any()
    assert deform.delta.grad[:n_fine][deform.fine_active_mask].abs().sum() > 0
    deform.optimizer.step()
    # Mid-frame state restores both output and optimizer.
    restored = D.Deformation(deform.config)
    restored.restore(deform.capture())
    torch.testing.assert_close(restored(xyz)[0], deform(xyz)[0])
    torch.testing.assert_close(restored(xyz)[1], deform(xyz)[1])
    torch.testing.assert_close(restored.grids[0].blend_weights, deform.grids[0].blend_weights)
    # New GS and moving memberships remain restricted on the next frame.
    deform.reset(torch.cat([xyz, torch.randn(10,3)]))
    assert deform.index.shape == (3,90)
    assert torch.equal(deform.fine_to_mid[deform.index[0]], deform.middle_parent_of_gs)
    assert not deform.fine_active_mask.any()
    deform.reset_grid(torch.randn(120,3))
    assert deform.index.shape == (3,120)
    assert torch.equal(deform.fine_to_mid[deform.index[0]], deform.middle_parent_of_gs)


def test_gradhet_does_not_fill_missing_or_zero_scores():
    torch.manual_seed(1)
    xyz = torch.randn(80, 3)
    deform = D.Deformation(D.DeformConfig(hierarchy_mode='nested', max_gs_per_grid=8,
                                         grid_level_ratio=3, fine_keep_ratio=1, momentum=None))
    deform.setup(xyz)
    deform.activate_refinement()
    assert not deform.middle_active_mask.any()  # No observations.
    deform.score_observations.fill_(1)
    deform.activate_refinement()
    assert not deform.middle_active_mask.any()  # Observed zero variance.
    candidate = torch.where(torch.bincount(deform.fine_to_mid,
                                           minlength=deform.score_sum.numel()) >= 2)[0][0]
    deform.score_sum[candidate] = 1
    deform.activate_refinement()
    assert deform.middle_active_mask.sum() == 1
    assert deform.middle_active_mask[candidate]


def test_binary_quantization_and_rotation():
    x = torch.tensor([[.011, -.013, 0., 0., 0., 0.]], requires_grad=True)
    q = torch.full((6,), .01)
    y = C.quantize_ste(x, q)
    torch.testing.assert_close(y, torch.tensor([[.01, -.01, 0., 0., 0., 0.]]), rtol=0, atol=1e-8)
    rotation = C.rotation_vector_to_quaternion(x[:, 3:])
    torch.testing.assert_close(rotation, torch.tensor([[1., 0., 0., 0.]]), rtol=0, atol=0)
    (y.sum() + rotation.sum()).backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad[:, :3], torch.ones(1, 3), rtol=0, atol=0)
    bits = C.laplace_bits(torch.tensor([0., 1., 2., 3.]), torch.ones(4))
    assert torch.isfinite(bits).all() and (bits[1:] > bits[:-1]).all()
    symbols = torch.arange(-100., 101.)
    mass = (2 ** -C.laplace_bits(symbols, torch.ones_like(symbols))).sum()
    torch.testing.assert_close(mass, torch.tensor(1.), rtol=0, atol=3e-7)


def test_gradient_assisted_binary_clusters():
    xyz = torch.zeros(16, 3)
    xyz[:, 0] = torch.arange(16.)
    parent = torch.zeros(16, dtype=torch.long)
    gradient = torch.zeros_like(xyz)
    gradient[:, 0] = torch.tensor([-1., 1.] * 8)
    observed = torch.ones(16, dtype=torch.bool)
    spatial = B.binary_clusters(xyz, parent, gradient, observed, 1, 0.)
    assisted = B.binary_clusters(xyz, parent, gradient, observed, 1, 10.)
    assert spatial[:8].unique().numel() == spatial[8:].unique().numel() == 1
    assert assisted[::2].unique().numel() == assisted[1::2].unique().numel() == 1
    assert assisted[0] != assisted[1]
    degenerate = B.binary_clusters(torch.zeros_like(xyz), parent, gradient,
                                   torch.zeros_like(observed), 1, 0.)
    assert degenerate.unique().numel() == 1


def make_binary_test_model():
    torch.manual_seed(42)
    xyz = torch.randn(128, 3)
    config = D.DeformConfig(hierarchy_mode='binary', binary_root_size=64,
                           binary_min_child=2, binary_max_depth=3,
                           binary_max_splits=30, binary_candidate_batch=20,
                           binary_gradient_weight=.5, momentum=None)
    model = B.BinaryMotion(config)
    model.setup(xyz)
    return model, xyz


def test_binary_continuity_hierarchy_and_restore():
    model, xyz = make_binary_test_model()
    model.accumulate_gradient_score(xyz, torch.ones(len(xyz), dtype=torch.bool))
    proposal = model.propose()
    assert len(proposal['nodes']) > 0
    proposal['nodes'] = proposal['nodes'][:1]
    before = model(xyz)
    before_reg = model.reg_loss().detach().clone()
    old_parent = model.leaf_index.clone()
    selected_parent = proposal['nodes'][0]
    model.activate(proposal)
    for a, b in zip(before, model(xyz)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(model.reg_loss(), before_reg, rtol=0, atol=0)
    selected_points = old_parent == selected_parent
    assert torch.equal(model.leaf_index[~selected_points], old_parent[~selected_points])
    assert model.leaf_index[selected_points].unique().numel() == 2
    torch.testing.assert_close(model.path_weight[:, 0].sum(), torch.tensor(0.), rtol=0, atol=1e-5)
    # Both an unsplit root and an existing child can compete in the next round.
    model.clear_scores()
    model.accumulate_gradient_score(torch.randn_like(xyz), torch.ones(len(xyz), dtype=torch.bool))
    next_proposal = model.propose()
    assert {0, 1} <= set(next_proposal['depths'].tolist())
    coefficients = C.quantize_ste(model.delta[:model.get_grid_num], model.quant_step)
    dense = coefficients[model.root_index]
    for depth in range(model.active_depth):
        dense = dense + coefficients[model.path_index[:, depth].clamp_min(0)] * model.path_weight[:, depth, None]
    sparse = model.field()
    torch.testing.assert_close(sparse, dense, rtol=0, atol=0)
    dense_grad = torch.autograd.grad(dense.sum(), model.delta)[0]
    sparse_grad = torch.autograd.grad(sparse.sum(), model.delta)[0]
    torch.testing.assert_close(sparse_grad, dense_grad, rtol=0, atol=1e-5)
    loss = (model(xyz)[0] - xyz * .01).square().mean() + 1e-7 * model.rate_bits()
    loss.backward()
    assert torch.isfinite(model.delta.grad).all()
    assert model.delta.grad[model.root_count].abs().sum() > 0
    assert not model.delta.grad[model.get_grid_num:].any()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    checkpoint = model.capture()
    restored = B.BinaryMotion(model.config)
    restored.restore(checkpoint)
    for a, b in zip(model(xyz), restored(xyz)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with torch.no_grad():
        model.delta.add_(1.)
    assert not torch.equal(model.delta, checkpoint['state']['delta'])
    assert torch.equal(restored.delta, checkpoint['state']['delta'])
    restored.reset(torch.cat([xyz, xyz[:8] + .01]))
    assert len(restored.root_index) == 136 and restored.split_count == 0


def test_binary_motion_codec():
    import tempfile
    import numpy as np
    model, xyz = make_binary_test_model()
    model.accumulate_gradient_score(xyz, torch.ones(len(xyz), dtype=torch.bool))
    model.activate(model.propose())
    model.clear_scores()
    model.accumulate_gradient_score(torch.randn_like(xyz), torch.ones(len(xyz), dtype=torch.bool))
    model.activate(model.propose())
    with torch.no_grad():
        model.delta[:model.get_grid_num].normal_(0, .005)
    with tempfile.TemporaryDirectory(prefix='binary-motion-test-') as directory:
        stats = model.export(Path(directory) / 'motion.npz')
        assert stats['motion_bytes'] > 0 and stats['decode_xyz_max_error'] < 1e-7
        symbols = np.array([[0, 1, -2, 0, 0, 100000], [0, -1, 2, 0, 0, -100000]], dtype=np.int32)
        payload = C.encode_coefficients(symbols, np.array([0, 0]), np.ones((1, 6)))
        np.testing.assert_array_equal(C.decode_coefficients(payload), symbols)


def _check_binary_trial(all_views):
    import contextlib
    import tempfile
    from types import MethodType
    from unittest.mock import patch
    M = importlib.import_module('pipeline.ReCon-GS.module')
    model, xyz = make_binary_test_model()
    model.config.binary_probe_steps = 2
    model.config.binary_eval_views = 2
    model.config.binary_decision_all_views = all_views
    model.accumulate_gradient_score(xyz, torch.ones(len(xyz), dtype=torch.bool))
    module = M.Module(M.ModuleConfig(deformation=model.config, lambda_deform=.1),
                      datamodule=SimpleNamespace(current_frame=1))
    module.deform = model
    module.gaussians = SimpleNamespace(get_xyz_ori=xyz)
    image = .5 + (xyz * .01).T.reshape(3, 16, 8)
    module.binary_views = {str(i): SimpleNamespace(path=Path(f'train_cam_{i}.png'), image=image)
                           for i in range(6)}

    def forward(self, viewpoint, training=False):
        displacement, _ = self.deform(xyz)
        return {'render': .5 + displacement.T.reshape(3, 16, 8)}

    module.forward = MethodType(forward, module)
    with tempfile.TemporaryDirectory(prefix='binary-trial-test-') as directory:
        module.trainer = SimpleNamespace(output_dir=Path(directory), device=torch.device('cpu'),
                                         global_step=20, num_steps=200)
        with patch('torch.cuda.synchronize'), patch('torch.random.fork_rng',
                                                    return_value=contextlib.nullcontext()):
            module.binary_refinement_trial()
        assert module.deform.probe_backward_steps == 4
        assert len(module.deform.events) == 1
        event = module.deform.events[0]
        assert event['decision_uses_all_training_views'] == all_views
        if all_views:
            assert set(event['train_views']) <= set(event['decision_views'])
        else:
            assert set(event['train_views']).isdisjoint(event['decision_views'])
        assert event['accepted'] == (event['objective_after'] < event['objective_before'])
        assert not module.deform.gradient_observations.any()


def test_binary_trial_integration_without_renderer():
    _check_binary_trial(False)
    _check_binary_trial(True)


def test_hierarchical_step_size_optimizer():
    path = Path(__file__).with_name('test_row_scaled_adam.py')
    spec = importlib.util.spec_from_file_location('row_optimizer_tests', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.test_scaled_updates_and_restore()
    checks.test_unit_scale_is_adam()
    checks.test_closure()
    checks.test_old_adam_checkpoint()


def test_rd_tree_projection_reference():
    path = Path(__file__).with_name('test_rd_tree_projection.py')
    spec = importlib.util.spec_from_file_location('rd_tree_projection_checks', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.run_all()


def test_rd_tree_motion_training_state():
    path = Path(__file__).with_name('test_rd_tree_motion.py')
    spec = importlib.util.spec_from_file_location('rd_tree_motion_checks', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.run_all()


def test_rd_tree_basis_coordinates():
    path = Path(__file__).with_name('test_rd_tree_basis.py')
    spec = importlib.util.spec_from_file_location('rd_tree_basis_checks', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.run_all()


def test_rd_latent_acceptance_checks():
    path = Path(__file__).with_name('test_rd_latent_acceptance.py')
    spec = importlib.util.spec_from_file_location('rd_latent_acceptance_checks', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.run_all()


def test_fixed_cut_oracle():
    path = Path(__file__).with_name('test_fixed_cut_reference.py')
    spec = importlib.util.spec_from_file_location('fixed_cut_checks', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.test_zero_details_do_not_prune()
    checks.test_adaptive_objective_bounds_fixed_cuts()
    checks.test_production_fixed_cut_matches_oracle()


def test_lossless_zero_subtree_control():
    path = Path(__file__).with_name('test_zero_subtree_compaction.py')
    spec = importlib.util.spec_from_file_location('zero_compaction_checks', path)
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.test_zero_subtree_compaction()


if __name__ == '__main__':
    test_variance()
    test_nested_sampling()
    test_overlap_width_and_junction()
    test_weighted_diagnostics()
    test_schedule_and_reset()
    test_gradhet_does_not_fill_missing_or_zero_scores()
    test_binary_quantization_and_rotation()
    test_gradient_assisted_binary_clusters()
    test_binary_continuity_hierarchy_and_restore()
    test_binary_motion_codec()
    test_binary_trial_integration_without_renderer()
    test_hierarchical_step_size_optimizer()
    test_rd_tree_projection_reference()
    test_rd_tree_motion_training_state()
    test_rd_tree_basis_coordinates()
    test_rd_latent_acceptance_checks()
    test_fixed_cut_oracle()
    test_lossless_zero_subtree_control()
    print('All nested and binary motion numerical/codec checks passed.')
