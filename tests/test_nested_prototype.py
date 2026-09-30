"""Numerical checks without loading the renderer or a scene."""
import importlib
import copy
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
D = importlib.import_module('pipeline.ReCon-GS.model.deformation')
N = importlib.import_module('pipeline.ReCon-GS.model.nested')
G = importlib.import_module('pipeline.ReCon-GS.model.gated_four')
R = importlib.import_module('pipeline.ReCon-GS.model.region_rigid')
S = importlib.import_module('pipeline.ReCon-GS.model.selective_motion')
DATA = importlib.import_module('pipeline.ReCon-GS.data')
Q = importlib.import_module('pipeline.ReCon-GS.model.residual_motion')
C = importlib.import_module('pipeline.ReCon-GS.model.residual_codec')


class WorkerOnlyPairView(SimpleNamespace):
    def to(self, device):
        self.image = self.image.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        return self


class WorkerOnlyPairDataset(torch.utils.data.Dataset):
    """A fixture that reproduces libgs' requirement to read inside a worker."""
    def __init__(self, frame):
        self.frame = frame
        self.items = [SimpleNamespace(path=Path(name)) for name in ('b.png', 'a.png')]

    def __getitem__(self, index):
        assert torch.utils.data.get_worker_info() is not None
        return WorkerOnlyPairView(image=torch.full((3, 2, 2), float(self.frame)),
                                  full_proj_transform=torch.eye(4),
                                  name=self.items[index].path.name)

    def __len__(self):
        return len(self.items)

    def get_frame_dataset(self, frame):
        return WorkerOnlyPairDataset(frame)


def test_selective_pairs_use_workers_and_frame_cache():
    data = DATA.DataModule.__new__(DATA.DataModule)
    data.config = SimpleNamespace(extra_dataloader_kwargs=dict(num_workers=1, pin_memory=False))
    data.scene = SimpleNamespace(train_dataset=WorkerOnlyPairDataset(0))
    data.current_frame = 1
    names = ['a.png', 'b.png']
    pairs = data.selective_training_pairs(names, 'cpu', 0.0)
    assert [now.name for now, _ in pairs] == names
    assert [old.name for _, old in pairs] == names
    assert all(now.image[0, 0, 0] == 1 and old.image[0, 0, 0] == 0 for now, old in pairs)
    assert data.selective_training_pairs(names, 'cpu', 0.0) is pairs
    data.current_frame = 2
    next_pairs = data.selective_training_pairs(names, 'cpu', 0.0)
    assert next_pairs is not pairs
    assert all(now.image[0, 0, 0] == 2 and old.image[0, 0, 0] == 1 for now, old in next_pairs)


def selective_test_config(**overrides):
    # Deliberate deterministic fixtures; these are not experimental defaults.
    values = dict(max_depth=2, coarse_regions=1, max_leaf_regions=4,
                  min_region_gs=1, probe_steps=2, shared_fit_steps_per_depth=1,
                  refine_steps=2, incremental_steps=0, enable_incremental=False,
                  comparison_view_names=['a.png', 'b.png'], max_leaf_comparisons=4,
                  max_internal_comparisons=4, loss_tolerance=0.0,
                  min_visible_fraction=1.0, min_visible_views=2, min_change_views=2,
                  min_changed_fraction=1.0, image_change_threshold=0.1,
                  evidence_patch_radius=0, translation_lr=0.01, rotation_lr=0.01,
                  adam_betas=[0.9, 0.999], adam_eps=1e-8, motion_regularization=0.0,
                  camera_match_atol=0.0, clone_sigma_scale=2.0, roi_padding_pixels=0,
                  final_loss_tolerance=0.0)
    values.update(overrides)
    return S.SelectiveMotionConfig(**values)


def test_region_rigid_world_rotation():
    xyz = torch.tensor([[2., 0, 0]], dtype=torch.float64)
    center = torch.tensor([[1., 0, 0]], dtype=torch.float64)
    translation = torch.tensor([[0., 0, 3]], dtype=torch.float64)
    rotation = torch.tensor([[0., 0, torch.pi / 2]], dtype=torch.float64)
    original = R.rotation_vector_quaternion(torch.tensor([[torch.pi / 2, 0., 0.]], dtype=torch.float64))
    moved, orientation = R.apply_region_rigid(xyz, original, center, translation, rotation)
    torch.testing.assert_close(moved, torch.tensor([[1., 1, 3]], dtype=torch.float64), rtol=0, atol=1e-12)
    # World Z rotation followed by original X orientation; multiplication is not commutative.
    expected = torch.tensor([[.5, .5, .5, .5]], dtype=torch.float64)
    torch.testing.assert_close(orientation, expected, rtol=0, atol=1e-12)


def test_selective_partition_keep_and_checkpoint():
    xyz = torch.tensor([[-3., 0, 0], [-1., 0, 0], [1., 0, 0], [3., 0, 0]])
    raw_rotation = torch.tensor([[2., 0, 0, 0]] * 4)
    motion = S.SelectiveMotion(selective_test_config())
    motion.setup(xyz, raw_rotation)
    assert sorted(torch.cat([motion.members[n] for n, c in enumerate(motion.children) if not c]).tolist()) == [0, 1, 2, 3]
    motion.evidence[:] = S.CHANGED
    with torch.no_grad():
        motion.translation[0, 0] = 100  # SPLIT root is never added to child motion.
        motion.translation[motion.owner, 0] = torch.tensor([-1., -1., 1., 1.])
    moved, _ = motion.transform()
    torch.testing.assert_close(moved[:, 0], torch.tensor([-4., -2., 2., 4.]), rtol=0, atol=0)
    motion.states[motion.owner] = S.KEEP
    kept_xyz, kept_rotation = motion.transform()
    assert torch.equal(kept_xyz, xyz) and torch.equal(kept_rotation, raw_rotation)
    packet = copy.deepcopy(motion.capture())
    restored = S.SelectiveMotion(selective_test_config())
    restored.restore(packet)
    assert torch.equal(restored.owner, motion.owner)
    assert torch.equal(restored.transform()[1], raw_rotation)
    assert [motion.stage(i)[0] for i in range(1, 7)] == ['probe', 'probe', 'shared_fit', 'shared_fit', 'refine', 'refine']


def test_selective_keep_clears_optimizer_momentum():
    motion = S.SelectiveMotion(selective_test_config())
    motion.setup(torch.zeros(1, 3), torch.tensor([[2., 0, 0, 0]]))
    motion.evidence[:] = S.CHANGED
    motion.translation.sum().backward()
    motion.optimizer_step(motion.owner)
    before = motion.translation.detach().clone()
    motion.states[:] = S.KEEP
    motion.translation.sum().backward()
    motion.optimizer_step(motion.owner)
    assert torch.equal(motion.translation, before)
    assert not motion.optimizer.state[motion.translation]['exp_avg'].any()


def test_selective_evidence_separates_old_fit_error():
    motion = S.SelectiveMotion(selective_test_config())
    motion.setup(torch.zeros(1, 3), torch.tensor([[1., 0, 0, 0]]))
    old = SimpleNamespace(image=torch.zeros(3, 8, 8), full_proj_transform=torch.eye(4))
    now = SimpleNamespace(image=torch.zeros(3, 8, 8), full_proj_transform=torch.eye(4))
    reference = lambda view: dict(render=torch.ones_like(view.image),
                                  visibility_filter=torch.ones(1, dtype=torch.bool), radii=torch.ones(1))
    motion.gather_evidence([(now, old), (now, old)], reference)
    assert motion.evidence.tolist() == [S.STATIC]
    torch.testing.assert_close(motion.old_error, torch.ones(1), rtol=0, atol=0)
    motion.setup(torch.zeros(1, 3), torch.tensor([[1., 0, 0, 0]]))
    now.image.fill_(1)
    motion.gather_evidence([(now, old), (now, old)], reference)
    assert motion.evidence.tolist() == [S.CHANGED]
    assert motion.change_votes.tolist() == [2]


def test_selective_unknown_parent_does_not_erase_children():
    motion = S.SelectiveMotion(selective_test_config(max_leaf_comparisons=1, max_internal_comparisons=1))
    motion.setup(torch.tensor([[-1., 0, 0], [1., 0, 0]]), torch.tensor([[1., 0, 0, 0]] * 2))
    motion.evidence[:] = S.CHANGED
    motion.evidence[0] = S.UNKNOWN
    calls = []
    def compare(node, candidates):
        calls.append(node)
        return [0.0, 1.0, 0.0]
    motion.select(compare)
    assert not (motion.owner == 0).any()
    assert len(calls) == 1
    packet = copy.deepcopy(motion.capture())
    restored = S.SelectiveMotion(motion.config)
    restored.restore(packet)
    assert restored.selected
    torch.testing.assert_close(restored.accepted_translation, motion.accepted_translation, rtol=0, atol=0)


def test_selective_forest_and_internal_budget():
    config = selective_test_config(coarse_regions=2, max_leaf_regions=8,
                                   max_leaf_comparisons=1, max_internal_comparisons=2)
    motion = S.SelectiveMotion(config)
    xyz = torch.zeros(8, 3)
    xyz[:, 0] = torch.arange(8)
    motion.setup(xyz, torch.tensor([[1., 0, 0, 0]] * 8))
    leaves = [node for node, children in enumerate(motion.children) if not children]
    assert len(leaves) == 8
    assert motion.depths.count(0) == 2
    motion.evidence[:] = S.CHANGED
    calls = []
    def compare(node, candidates):
        calls.append(node)
        return [0.0, 1.0, 0.0]
    motion.select(compare)
    assert sum(not motion.children[node] for node in calls) == 1
    assert sum(bool(motion.children[node]) for node in calls) == 2
    # Selected terminal owners are disjoint: every GS has exactly one owner.
    for node in motion.owner.unique().tolist():
        assert torch.equal(motion.owner[motion.members[node]], torch.full_like(motion.members[node], node))


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


def test_four_level_gates_and_checkpoint():
    torch.manual_seed(7)
    xyz = torch.randn(180, 3)
    config = D.DeformConfig(hierarchy_mode='nested', num_grid_levels=4,
                            max_gs_per_grid=5, grid_level_ratio=3,
                            gate_keep_ratio=.5, momentum=None)
    deform = G.GatedFourDeformation(config)
    deform.setup(xyz)
    assert len(deform.grids) == 4
    assert deform.index.shape == (4, len(xyz))
    assert not any(deform._gate_mask(level).any() for level in (2, 3, 4))
    with torch.no_grad():
        deform.delta[:deform.grids[0].vertex_num, :3].normal_(0, .01)
        deform.delta[:deform.grids[0].vertex_num, 4] = .1

    visible = torch.ones(len(xyz), dtype=torch.bool)
    for step in range(1, 61):
        if deform.collecting_gradients(step):
            deform.accumulate_gradient_score(xyz, visible, step=step)
        if step == 35:
            midframe = G.GatedFourDeformation(config)
            midframe.restore(copy.deepcopy(deform.capture()))
            assert midframe.active_gate_count == 1
            torch.testing.assert_close(midframe.score_sum, deform.score_sum, rtol=0, atol=0)
            torch.testing.assert_close(midframe(xyz)[0], deform(xyz)[0], rtol=0, atol=0)
        if deform.gate_due(step):
            before = deform(xyz)
            before_reg = deform.reg_loss()
            stats = deform.activate_refinement(step)
            after = deform(xyz)
            for old, new in zip(before, after):
                torch.testing.assert_close(new, old, rtol=0, atol=0)
            torch.testing.assert_close(deform.reg_loss(), before_reg, rtol=0, atol=0)
            assert stats['score_steps'] == list(range(step - 9, step + 1))
            assert 0 < stats['selected'] <= stats['budget']
    assert deform.active_gate_count == 3
    assert not deform.collecting_gradients(61)
    assert torch.all(deform.gate_mask_l2[deform.grids[1].parent[deform.gate_mask_l3]])
    assert torch.all(deform.gate_mask_l3[deform.grids[2].parent[deform.gate_mask_l4]])

    loss = deform(xyz)[0].square().mean() + deform.reg_loss()
    loss.backward()
    for level in (2, 3, 4):
        grid = deform.grids[level - 1]
        offset = deform.offsets[level - 1]
        inactive = ~deform._gate_mask(level)[grid.parent]
        assert not deform.delta.grad[offset:offset + grid.vertex_num][inactive].any()
    deform.optimizer.step()
    deform.optimizer.zero_grad(set_to_none=True)
    checkpoint = copy.deepcopy(deform.capture())
    restored = G.GatedFourDeformation(config)
    restored.restore(checkpoint)
    for left, right in zip(deform(xyz), restored(xyz)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert restored.active_gate_count == 3
    assert restored.optimizer.state_dict()['state']
    restored.reset(torch.cat([xyz, xyz[:5]]))
    assert restored.index.shape == (4, len(xyz) + 5)
    assert restored.active_gate_count == 0
    assert not any(restored._gate_mask(level).any() for level in (2, 3, 4))
    restored.reset_grid(torch.randn(190, 3))
    assert restored.index.shape == (4, 190)
    assert restored.active_gate_count == 0


def test_four_level_export_reconstructs_motion():
    torch.manual_seed(8)
    xyz = torch.randn(180, 3)
    config = D.DeformConfig(hierarchy_mode='nested', num_grid_levels=4,
                            max_gs_per_grid=5, grid_level_ratio=3, momentum=None)
    deform = G.GatedFourDeformation(config)
    deform.setup(xyz)
    visible = torch.ones(len(xyz), dtype=torch.bool)
    for step in range(1, 61):
        if deform.collecting_gradients(step):
            deform.accumulate_gradient_score(xyz, visible, step=step)
        if deform.gate_due(step):
            deform.activate_refinement(step)
    with torch.no_grad():
        deform.delta[:, :3].normal_(0, .01)
        deform.delta[:, 4:].normal_(0, .01)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.npz'
        stats = deform.export(path)
        assert stats['active_anchors'] < stats['candidate_anchors']
        with np.load(path) as packet:
            dense = packet['delta'][packet['base_index']].copy()
            for level in (2, 3, 4):
                indices = packet[f'blend_indices_l{level}']
                weights = packet[f'blend_weights_l{level}']
                gate = packet[f'gate_l{level}']
                blended = (packet['delta'][indices.clip(min=0)] * weights[:, :, None]).sum(1)
                dense += blended * gate[:, None]
        expected_xyz, expected_rot = deform(xyz)
        torch.testing.assert_close(torch.from_numpy(dense[:, :3]), expected_xyz,
                                   rtol=0, atol=1e-6)
        decoded_rot = torch.nn.functional.normalize(torch.from_numpy(dense[:, 3:]), dim=-1)
        torch.testing.assert_close(decoded_rot, expected_rot, rtol=0, atol=1e-6)



def residual_test_config(**overrides):
    # Synthetic fixture units, not selected experiment hyperparameters.
    values = dict(fine_region_gs=2, motion_steps=4, shared_warmup_steps=1,
        incremental_steps=2, enable_incremental=True,
        region_translation_lr=.01, region_rotation_lr=.01,
        gs_translation_lr=.01, gs_rotation_lr=.01, adam_betas=[.9, .999], adam_eps=1e-8,
        motion_xyz_step_extent=.1, motion_rotation_step_rad=.1,
        new_xyz_step_extent=.1, new_rotation_step=.1, new_logscale_step=.1,
        new_opacity_step=.1, new_sh_step=.1, lambda_rate=.01,
        rate_warmup_steps=1, rate_ramp_steps=2,
        motion_entropy_scales=[1.] * 6, new_entropy_scales={name: 1. for name in Q.RAW_FIELDS},
        tuple_mask_probability=.1, support_temperature=.5, entropy_radius=3, entropy_chunk_rows=2,
        opacity_keep_threshold=.01, opacity_support_temperature=.1, clone_sigma_scale=2., new_max_screen_size=20.)
    values.update(overrides)
    return Q.QuantizedResidualConfig(**values)


def residual_raw_fixture(count=18):
    xyz = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) / 10
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1
    return dict(xyz=xyz, rotation=rotation, logscale=torch.zeros(count, 3),
                opacity=torch.zeros(count, 1), sh_dc=torch.zeros(count, 1, 3),
                sh_rest=torch.zeros(count, 3, 3))


def test_residual_layout_accumulation_and_zero_inheritance():
    raw = residual_raw_fixture()
    cfg = residual_test_config()
    motion = Q.QuantizedResidualMotion(cfg)
    motion.setup(raw['xyz'], raw['rotation'], 1.)
    duplicate_index, counts, _ = Q.stable_layout(raw['xyz'].clone(), cfg.fine_region_gs)
    torch.testing.assert_close(motion.index, duplicate_index, rtol=0, atol=0)
    assert motion.counts == counts
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .2
        motion.region_translation[1][:, 1] = .3
        motion.region_translation[2][:, 2] = .4
        motion.gs_translation[:, 0] = .1
        motion.region_rotation[0][:, 2] = .2
        motion.region_rotation[1][:, 0] = .1
    xyz, rotation = motion(4)
    torch.testing.assert_close(xyz, raw['xyz'] + torch.tensor([.3, .3, .4]), rtol=0, atol=2e-6)
    expected = R.rotation_vector_quaternion(torch.tensor([[0., 0., .2]])).expand(len(xyz), -1)
    expected = R.quaternion_multiply(R.rotation_vector_quaternion(torch.tensor([[.1, 0., 0.]])).expand(len(xyz), -1), expected)
    torch.testing.assert_close(rotation, expected, rtol=0, atol=2e-6)
    with torch.no_grad():
        motion.region_translation[1].zero_()
        motion.region_translation[2].zero_()
        motion.region_rotation[1].zero_()
        motion.gs_translation.zero_()
    inherited_xyz, _ = motion(4)
    torch.testing.assert_close(inherited_xyz, raw['xyz'] + torch.tensor([.2, 0., 0.]), rtol=0, atol=2e-6)
    stats = motion.statistics()
    assert stats['gs']['counts']['KEEP'] == 0
    assert stats['gs']['counts']['SHARED'] == len(xyz)
    assert all(row['counts']['SPLIT'] == 0 for row in stats['levels'])
    with torch.no_grad():
        motion.region_translation[0].zero_()
        motion.region_rotation[0].zero_()
    kept_xyz, kept_rot = motion(4)
    torch.testing.assert_close(kept_xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(kept_rot, raw['rotation'], rtol=0, atol=0)
    with torch.no_grad():
        motion.gs_translation[0, 0] = .1
    descendants = motion.statistics()
    assert descendants['gs']['counts']['SPLIT'] == 1
    assert descendants['levels'][0]['counts']['SPLIT'] == 1
    assert descendants['levels'][0]['encoded_active_fraction'] == 0


def test_residual_quantized_zero_has_image_proxy_gradient():
    raw = residual_raw_fixture()
    motion = Q.QuantizedResidualMotion(residual_test_config())
    motion.setup(raw['xyz'], raw['rotation'], 1.)
    with torch.no_grad():
        motion.gs_translation[:, 0] = .04  # Round(.04/.1)=0, but STE remains trainable.
    xyz, rotation = motion(4, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    image_proxy = xyz[:, 0].sum() + rotation[:, 1].sum()
    image_proxy.backward()
    torch.testing.assert_close(motion.gs_translation.grad[:, 0], torch.ones(len(xyz)), rtol=0, atol=0)
    assert torch.isfinite(motion.gs_rotation.grad).all()
    assert motion.gs_rotation.grad[:, 0].abs().sum() > 0
    motion.optimizer.zero_grad(set_to_none=True)
    motion(1, training=True)[0].sum().backward()
    motion.optimizer_step()
    assert not motion.optimizer.state.get(motion.gs_translation)


def test_residual_integer_quantization_checks_int32_bounds():
    try:
        Q.integer_symbols(torch.tensor([float(2 ** 31)], dtype=torch.float32), 1.)
    except ValueError:
        pass
    else:
        raise AssertionError('Positive int32 overflow must be rejected before conversion')
    minimum = Q.integer_symbols(torch.tensor([-float(2 ** 31)], dtype=torch.float32), 1.)
    torch.testing.assert_close(minimum, torch.tensor([-(2 ** 31)], dtype=torch.int32), rtol=0, atol=0)


def test_residual_sparse_rate_cdf_and_gradients():
    cdf, probabilities = C.laplace_integer_cdf(1., 3)
    unsigned = cdf.to(torch.int64).remainder(65536)
    unsigned[-1] = 65536
    assert unsigned[0] == 0 and unsigned[-1] == 65536
    assert (unsigned.diff() > 0).all()
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1., dtype=torch.float64), rtol=0, atol=0)
    raw = torch.tensor([[0., 0.], [1.1, 0.], [4.1, -4.1]], requires_grad=True)
    symbols = Q.quantized_symbols(raw, 1.)
    bits = C.expected_torchac_bits(symbols, [1., 1.], entropy_radius=3,
                                  mask_probability=.1, support_temperature=.5, sparse=True)
    mask_probability = C._mask_cdf(.1)[1]
    expected = -mask_probability[0].log2() - 2 * mask_probability[1].log2()
    expected += -probabilities[4].log2() - probabilities[3].log2()
    expected += 2 * (-probabilities[-1].log2() + 32)
    torch.testing.assert_close(bits.detach().double(), expected, rtol=1e-6, atol=1e-5)
    bits.backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad[1].abs().sum() > 0
    zero = torch.zeros(3, 0, requires_grad=True)
    per_row = C.expected_torchac_bits(zero, 1., entropy_radius=3, reduction='none')
    assert per_row.shape == (3,)
    per_row.sum().backward()
    assert zero.grad is not None


def test_residual_entropy_codec_sparse_overflow_and_empty():
    packet = dict(motion=torch.tensor([[0, 0], [1, 0], [-3, 3], [4, -4],
                                      [2 ** 30, -(2 ** 30)], [0, 0]], dtype=torch.int32),
                  birth=torch.tensor([[0, 0, 0], [1, 0, -1]], dtype=torch.int32),
                  deleted=torch.tensor([1, 5], dtype=torch.int32),
                  empty=torch.empty(0, 6, dtype=torch.int32),
                  sh_empty=torch.empty(3, 0, dtype=torch.int32))
    models = dict(motion=dict(scale=[1., 2.], sparse=True, mask_probability=.1),
                  birth=dict(scale=1., sparse=False), empty=dict(scale=1., sparse=True, mask_probability=.1),
                  sh_empty=dict(scale=[], sparse=False))
    metadata = dict(layout_version=Q.LAYOUT_VERSION, fixture='sparse/one-component/overflow/empty')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'frame.bin'
        counts = C.encode_entropy_frame(path, packet, metadata, entropy_radius=3, chunk_rows=2, entropy_models=models)
        decoded = C.decode_entropy_frame(path)
        assert decoded['metadata'] == metadata
        assert path.stat().st_size == counts['total_bytes']
        assert counts['total_bytes'] == counts['header_bytes'] + sum(counts[key] for key in
            ('mask_bytes', 'value_bytes', 'escape_bytes', 'raw_bytes', 'chunk_header_bytes'))
        for name, value in packet.items():
            assert decoded['tensors'][name].shape == value.shape
            assert decoded['tensors'][name].dtype == value.dtype
            torch.testing.assert_close(decoded['tensors'][name], value, rtol=0, atol=0)
        zero_packet = dict(motion=torch.zeros(4, 6, dtype=torch.int32))
        zero_models = dict(motion=dict(scale=[1.] * 6, sparse=True, mask_probability=.1))
        zero_stats = C.encode_entropy_frame(path, zero_packet, {}, entropy_radius=3, chunk_rows=2, entropy_models=zero_models)
        assert zero_stats['value_bytes'] == 0 and zero_stats['escape_bytes'] == 0


def residual_replay_packet(previous, deleted, cfg, new_sources):
    keep = torch.ones(len(previous['xyz']), dtype=torch.bool)
    keep[deleted.long()] = False
    _, counts, _ = Q.stable_layout(previous['xyz'][keep], cfg.fine_region_gs)
    packet = {f'region_{n}': torch.zeros(count, 6, dtype=torch.int32) for n, count in enumerate(counts)}
    packet.update(gs_local=torch.zeros(int(keep.sum()), 6, dtype=torch.int32),
                  deleted_prev_indices=deleted, new_source_old_index=new_sources)
    for name, raw in previous.items():
        packet['new_' + name] = torch.zeros(len(new_sources), int(np.prod(raw.shape[1:])), dtype=torch.int32)
    metadata = dict(config=Q.asdict(cfg), extent=1., layout_version=Q.LAYOUT_VERSION,
                    previous_reference_sha256=Q.reference_digest(previous))
    models = {name: dict(scale=[1.] * 6, sparse=True, mask_probability=.1)
              for name in ('region_0', 'region_1', 'region_2', 'gs_local')}
    for name, raw in previous.items():
        models['new_' + name] = dict(scale=[1.] * int(np.prod(raw.shape[1:])), sparse=False)
    return packet, metadata, models




def test_residual_new_quantization_retention_gradient_and_empty_graph():
    cfg = residual_test_config()
    reference = residual_raw_fixture()
    model = SimpleNamespace()
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(reference[name][:1].clone()))
    with torch.no_grad():
        model._xyz[:, 0] += .04
        model._opacity.fill_(-10.)
    view = Q.QuantizedNewGaussianView(model, cfg, 1., reference)
    torch.testing.assert_close(view.get_xyz, reference['xyz'][:1], rtol=0, atol=0)
    assert not view.keep.any()
    view.get_xyz.sum().backward(retain_graph=True)
    torch.testing.assert_close(model._xyz.grad, torch.ones_like(model._xyz), rtol=0, atol=0)
    model._opacity.grad = None
    costs = view.keep_st * 32
    for name, symbols in view.symbols.items():
        costs = costs + view.keep_st * C.expected_torchac_bits(symbols.flatten(1), 1.,
            entropy_radius=3, reduction='none')
    costs.sum().backward()
    assert torch.isfinite(model._opacity.grad).all()
    assert model._opacity.grad.abs().sum() > 0
    empty = SimpleNamespace()
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(empty, attr, torch.nn.Parameter(reference[name][:0].clone()))
    empty_view = Q.QuantizedNewGaussianView(empty, cfg, 1., reference)
    differentiable_zero = empty_view.get_opacity.sum() + empty_view.keep_st.sum() * 32
    assert differentiable_zero.requires_grad
    differentiable_zero.backward()
    assert empty._opacity.grad is not None

def test_residual_complete_two_frame_replay_and_checkpoint():
    torch.manual_seed(19)
    cfg = residual_test_config()
    previous = residual_raw_fixture(18)
    with tempfile.TemporaryDirectory() as directory:
        for frame, deleted in enumerate((torch.tensor([2], dtype=torch.int32),
                                         torch.tensor([0, 5], dtype=torch.int32)), start=1):
            packet, metadata, models = residual_replay_packet(previous, deleted, cfg,
                                                               torch.tensor([0, 4], dtype=torch.int32))
            packet['region_0'][:, 0] = frame
            packet['gs_local'][-1, 1] = 1
            packet['new_xyz'][1, 2] = 1
            packet['new_opacity'][0, 0] = -1
            # Independent manual expectation for this translation-only replay fixture.
            keep = torch.ones(len(previous['xyz']), dtype=torch.bool)
            keep[deleted.long()] = False
            remaining = {name: value[keep].clone() for name, value in previous.items()}
            remaining['xyz'][:, 0] += frame * .1
            remaining['xyz'][-1, 1] += .1
            born = {name: value[torch.tensor([0, 4])].clone() for name, value in remaining.items()}
            born['xyz'][1, 2] += .1
            born['opacity'][0, 0] -= .1
            expected = {name: torch.cat((remaining[name], born[name])) for name in Q.RAW_FIELDS}
            path = Path(directory) / f'frame_{frame}.bin'
            C.encode_entropy_frame(path, packet, metadata, entropy_radius=3, chunk_rows=2, entropy_models=models)
            decoded = C.decode_entropy_frame(path)
            bad_metadata = dict(decoded['metadata'], previous_reference_sha256='wrong-reference')
            try:
                Q.replay_frame(previous, decoded['tensors'], bad_metadata)
            except ValueError:
                pass
            else:
                raise AssertionError('Replay must reject the wrong previous decoded reference')
            reconstructed = Q.replay_frame(previous, decoded['tensors'], decoded['metadata'])
            for name in Q.RAW_FIELDS:
                torch.testing.assert_close(reconstructed[name], expected[name], rtol=0, atol=0)
            assert len(reconstructed['xyz']) == len(previous['xyz']) - len(deleted) + 2
            previous = reconstructed
    motion = Q.QuantizedResidualMotion(cfg)
    motion.setup(previous['xyz'], previous['rotation'], 1.)
    motion(4, training=True)[0].sum().backward()
    motion.optimizer_step()
    before = motion(4)
    packet = copy.deepcopy(motion.capture())
    restored = Q.QuantizedResidualMotion(cfg)
    restored.restore(packet)
    for left, right in zip(before, restored(4)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert restored.optimizer.state_dict()['state']
    motion.committed = True
    frozen = Q.QuantizedResidualMotion(cfg)
    frozen.restore(copy.deepcopy(motion.capture()))
    assert all(not value.requires_grad for value in frozen.parameters())
    frozen.setup(previous['xyz'], previous['rotation'], 1.)
    assert all(value.requires_grad for value in frozen.parameters())


def test_residual_motion_warm_start_repartitions_total_motion_and_restores():
    cfg = residual_test_config(incremental_steps=0, enable_incremental=False,
                               motion_init_decay=.6, motion_rotation_step_rad=.001)
    raw = residual_raw_fixture()
    old = Q.QuantizedResidualMotion(cfg)
    old.setup(raw['xyz'], raw['rotation'], 1.)
    with torch.no_grad():
        old.region_translation[0][:, 1] = .2
        old.region_translation[1][:, 2] = .1
        old.region_rotation[0][:, 2] = .2
        old.gs_translation[0, 0] = 1.2  # Move one GS across the next spatial split.
        old.gs_translation[1, 0] = -.4
        old.gs_rotation[0, 0] = .3
        old.gs_rotation[2, 2] = 3.5  # The old total exceeds pi; decay must use its shortest rotation.
    old.committed = True
    old_xyz, old_rotation = old(cfg.motion_steps)
    old_index = old.index.clone()
    old_total_translation, old_total_rotation = old.final_quantized_total_motion()
    packet = copy.deepcopy(old.capture())
    next_motion = Q.QuantizedResidualMotion(cfg)
    next_motion.restore(packet)
    next_motion.setup(old_xyz.detach(), old_rotation.detach(), 1.)
    assert not torch.equal(next_motion.index, old_index)
    assert next_motion.gs_translation.abs().sum() > 0
    assert next_motion.gs_rotation.abs().sum() > 0
    initial_xyz, initial_rotation = next_motion(1)
    joint_xyz, joint_rotation = next_motion(cfg.shared_warmup_steps + 1)
    torch.testing.assert_close(initial_xyz, joint_xyz, rtol=0, atol=0)
    torch.testing.assert_close(initial_rotation, joint_rotation, rtol=0, atol=0)
    expected_offset = (old_total_translation * cfg.motion_init_decay / .1).round() * .1
    torch.testing.assert_close(initial_xyz - old_xyz, expected_offset, rtol=0, atol=2e-6)
    _, initialized_rotation = next_motion.final_quantized_total_motion()
    target_rotation = R.rotation_vector_quaternion(
        Q.shortest_rotation_vector(old_total_rotation) * cfg.motion_init_decay)
    rotation_delta = R.quaternion_multiply(initialized_rotation, Q.inverse_unit_quaternion(target_rotation))
    rotation_error = Q.shortest_rotation_vector(rotation_delta).norm(dim=-1)
    assert float(rotation_error.max()) < 3 ** .5 * cfg.motion_rotation_step_rad / 2 + 1e-5
    assert Q.shortest_rotation_vector(old_total_rotation)[2, 2] < 0
    assert Q.shortest_rotation_vector(target_rotation)[2, 2] < 0
    local_translation_before = next_motion.gs_translation.detach().clone()
    local_rotation_before = next_motion.gs_rotation.detach().clone()
    next_motion(1, training=True)[0].sum().backward()
    assert next_motion.gs_translation.grad is None
    assert next_motion.gs_rotation.grad is None
    next_motion.optimizer_step()
    torch.testing.assert_close(next_motion.gs_translation, local_translation_before, rtol=0, atol=0)
    torch.testing.assert_close(next_motion.gs_rotation, local_rotation_before, rtol=0, atol=0)
    assert not next_motion.optimizer.state.get(next_motion.gs_translation)
    midframe = Q.QuantizedResidualMotion(cfg)
    midframe.restore(copy.deepcopy(next_motion.capture()))
    for left, right in zip(next_motion(1), midframe(1)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    # Restoring into a used object must replace state without applying another warm start.
    next_motion.restore(packet)
    for left, right in zip(old(cfg.motion_steps), next_motion(cfg.motion_steps)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    next_motion.setup(old_xyz.detach(), old_rotation.detach(), 1.)
    for left, right in zip((initial_xyz, initial_rotation), next_motion(1)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    bad_xyz = old_xyz.detach().clone()
    bad_xyz[[0, 2]] = bad_xyz[[2, 0]]
    try:
        old.setup(bad_xyz, old_rotation.detach(), 1.)
    except ValueError as exc:
        assert 'ordered' in str(exc)
    else:
        raise AssertionError('Warm start must reject Gaussian reordering')


def test_residual_motion_decay_zero_and_no_topology_replay():
    cfg = residual_test_config(incremental_steps=0, enable_incremental=False)
    raw = residual_raw_fixture()
    motion = Q.QuantizedResidualMotion(cfg)
    motion.setup(raw['xyz'], raw['rotation'], 1.)
    with torch.no_grad():
        motion.gs_translation[:, 0] = .2
    torch.testing.assert_close(motion(1)[0], raw['xyz'], rtol=0, atol=0)
    motion.committed = True
    packet = copy.deepcopy(motion.capture())
    del packet['config']['motion_init_decay']  # Checkpoints from before the field existed.
    restored = Q.QuantizedResidualMotion(cfg)
    restored.restore(packet)
    restored.setup(raw['xyz'], raw['rotation'], 1.)
    assert restored.gs_translation.count_nonzero() == 0
    previous = raw
    for _ in range(2):
        frame, metadata, _ = residual_replay_packet(previous, torch.empty(0, dtype=torch.int32), cfg,
                                                    torch.empty(0, dtype=torch.int32))
        frame['region_0'][:, 0] = 1
        replayed = Q.replay_frame(previous, frame, metadata)
        assert len(replayed['xyz']) == len(raw['xyz'])
        previous = replayed
    frame['deleted_prev_indices'] = torch.tensor([0], dtype=torch.int32)
    try:
        Q.replay_frame(previous, frame, dict(metadata, previous_reference_sha256=Q.reference_digest(previous)))
    except ValueError as exc:
        assert 'cannot delete' in str(exc)
    else:
        raise AssertionError('Incremental-disabled replay must reject old-GS deletion')
    frame['deleted_prev_indices'] = torch.empty(0, dtype=torch.int32)
    frame['new_source_old_index'] = torch.tensor([0], dtype=torch.int32)
    try:
        Q.replay_frame(previous, frame, dict(metadata, previous_reference_sha256=Q.reference_digest(previous)))
    except ValueError as exc:
        assert 'cannot delete or add' in str(exc)
    else:
        raise AssertionError('Incremental-disabled replay must reject new Gaussians')
    try:
        residual_test_config(motion_init_decay=.6).validate()
    except ValueError as exc:
        assert 'enable_incremental' in str(exc)
    else:
        raise AssertionError('Warm start must reject birth/deletion mode')

if __name__ == '__main__':
    test_selective_pairs_use_workers_and_frame_cache()
    test_region_rigid_world_rotation()
    test_selective_partition_keep_and_checkpoint()
    test_selective_keep_clears_optimizer_momentum()
    test_selective_evidence_separates_old_fit_error()
    test_selective_unknown_parent_does_not_erase_children()
    test_selective_forest_and_internal_budget()
    test_variance()
    test_nested_sampling()
    test_overlap_width_and_junction()
    test_weighted_diagnostics()
    test_schedule_and_reset()
    test_gradhet_does_not_fill_missing_or_zero_scores()
    test_four_level_gates_and_checkpoint()
    test_four_level_export_reconstructs_motion()
    test_residual_layout_accumulation_and_zero_inheritance()
    test_residual_quantized_zero_has_image_proxy_gradient()
    test_residual_integer_quantization_checks_int32_bounds()
    test_residual_sparse_rate_cdf_and_gradients()
    test_residual_entropy_codec_sparse_overflow_and_empty()
    test_residual_new_quantization_retention_gradient_and_empty_graph()
    test_residual_complete_two_frame_replay_and_checkpoint()
    test_residual_motion_warm_start_repartitions_total_motion_and_restores()
    test_residual_motion_decay_zero_and_no_topology_replay()
    print('All nested prototype numerical checks passed.')
