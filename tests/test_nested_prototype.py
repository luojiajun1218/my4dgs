"""Numerical checks without loading the renderer or a scene."""
import importlib
import ast
import copy
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
# Import numerical modules without executing the production pipeline initializer,
# which creates a GUI socket at import time.
pipeline_package = ModuleType('pipeline')
pipeline_package.__path__ = [str(project_root / 'pipeline')]
recon_package = ModuleType('pipeline.ReCon-GS')
recon_package.__path__ = [str(project_root / 'pipeline' / 'ReCon-GS')]
pipeline_package.__dict__['ReCon-GS'] = recon_package
sys.modules['pipeline'] = pipeline_package
sys.modules['pipeline.ReCon-GS'] = recon_package
D = importlib.import_module('pipeline.ReCon-GS.model.deformation')
N = importlib.import_module('pipeline.ReCon-GS.model.nested')
G = importlib.import_module('pipeline.ReCon-GS.model.gated_four')
R = importlib.import_module('pipeline.ReCon-GS.model.region_rigid')
S = importlib.import_module('pipeline.ReCon-GS.model.selective_motion')
DATA = importlib.import_module('pipeline.ReCon-GS.data')
Q = importlib.import_module('pipeline.ReCon-GS.model.residual_motion')
GR = importlib.import_module('pipeline.ReCon-GS.model.gated_residual')
GAUSS = importlib.import_module('pipeline.ReCon-GS.model.gaussian')


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


def test_region_rigid_zero_motion_is_bitwise_identity_with_gradients():
    xyz = torch.tensor([[.1, -.2, .3], [1.7, .9, -2.3]], dtype=torch.float32)
    center = torch.tensor([[.4, .5, -.6], [-1.1, .7, .8]], dtype=torch.float32)
    original = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]],
                            dtype=torch.float32)
    translation = torch.zeros(2, 3, dtype=torch.float32, requires_grad=True)
    rotation = torch.zeros(2, 3, dtype=torch.float32, requires_grad=True)
    moved, orientation = R.apply_region_rigid(xyz, original, center,
                                               translation, rotation)
    assert torch.equal(moved, xyz)
    assert torch.equal(orientation, original)
    (moved[:, 1].sum() + moved[:, 2].sum() + orientation[:, 2].sum()).backward()
    assert translation.grad is not None and torch.isfinite(translation.grad).all()
    assert rotation.grad is not None and torch.isfinite(rotation.grad).all()
    assert translation.grad.abs().sum() > 0
    assert rotation.grad.abs().sum() > 0


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
        new_opacity_step=.1, new_sh_step=.1,
        storage_only=True, storage_loss_weight=0., storage_warmup_steps=1, storage_ramp_steps=2,
        opacity_keep_threshold=.01, opacity_support_temperature=.1,
        clone_sigma_scale=2., new_max_screen_size=20.)
    values.update(overrides)
    return Q.QuantizedResidualConfig(**values)


def gated_residual_test_config(**overrides):
    # Test-only schedule and units; these are not experiment hyperparameters.
    values = dict(enable_learned_gates=True, gate_lr=.01,
                  gate_initial_logits=[-2., 2., -2.],
                  gate_temperature_start=1., gate_temperature_end=.5,
                  gate_refine_steps=1, storage_loss_weight=.001)
    values.update(overrides)
    return residual_test_config(**values)


def storage_test_config(**overrides):
    return gated_residual_test_config(**overrides)


def test_storage_config_and_legacy_codec_checkpoint_rejection():
    storage_test_config().validate()
    residual_test_config().validate()
    for name, value in (('storage_loss_weight', -1.),
                        ('storage_loss_weight', float('nan')),
                        ('storage_warmup_steps', -1), ('storage_ramp_steps', 0)):
        try:
            storage_test_config(**{name: value}).validate()
        except ValueError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f'{name}={value!r} must be rejected')
    try:
        storage_test_config(enable_learned_gates=False).validate()
    except ValueError:
        pass
    else:
        raise AssertionError('A positive storage loss requires differentiable learned gates')
    raw, motion = gated_fixture()
    packet = copy.deepcopy(motion.capture())
    packet['config']['lambda_rate'] = .001
    restored = GR.GatedResidualMotion(motion.config)
    try:
        restored.restore(packet)
    except ValueError as exc:
        assert 'Legacy' in str(exc)
    else:
        raise AssertionError('Old codec checkpoints must not silently change training semantics')


def test_gated_residual_config_requires_explicit_valid_gate_parameters():
    residual_test_config().validate()
    gated_residual_test_config().validate()
    gated_residual_test_config(gate_initial_logits=[1., 0., 0.]).validate()
    gated_residual_test_config(gate_initial_logits=[0., 0., 1.]).validate()
    try:
        residual_test_config(enable_learned_gates=True).validate()
    except ValueError as exc:
        assert 'gate_' in str(exc)
    else:
        raise AssertionError('Learned gates require explicit gate parameters')
    invalid = dict(gate_lr=0., gate_initial_logits=[0., 0.],
                   gate_temperature_start=0., gate_temperature_end=float('nan'),
                   gate_refine_steps=4)
    for name, value in invalid.items():
        try:
            gated_residual_test_config(**{name: value}).validate()
        except ValueError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f'{name}={value!r} must be rejected')
    try:
        gated_residual_test_config(gate_initial_logits=[0., float('inf'), 0.]).validate()
    except ValueError as exc:
        assert 'gate_initial_logits' in str(exc)
    else:
        raise AssertionError('Nonfinite gate logits must be rejected')


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
    assert descendants['levels'][0]['nonzero_motion_fraction'] == 0


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
    bytes_per_birth = 4 + sum(value[0].numel() * value.element_size()
                              for value in reference.values())
    costs = view.keep_st * bytes_per_birth
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


def test_residual_motion_decay_zero_and_no_birth_materialization():
    cfg = residual_test_config(incremental_steps=0, enable_incremental=False)
    raw = residual_raw_fixture()
    motion = Q.QuantizedResidualMotion(cfg)
    motion.setup(raw['xyz'], raw['rotation'], 1.)
    with torch.no_grad():
        motion.gs_translation[:, 0] = .2
    torch.testing.assert_close(motion(1)[0], raw['xyz'], rtol=0, atol=0)
    motion.committed = True
    restored = Q.QuantizedResidualMotion(cfg)
    restored.restore(copy.deepcopy(motion.capture()))
    restored.setup(raw['xyz'], raw['rotation'], 1.)
    assert restored.gs_translation.count_nonzero() == 0
    for _ in range(2):
        moved_xyz, moved_rotation = restored(cfg.motion_steps)
        raw, sources = Q.materialize_quantized_gaussians(
            dict(raw, xyz=moved_xyz, rotation=moved_rotation))
        assert sources.numel() == 0 and len(raw['xyz']) == 18
        restored.committed = True
        restored.setup(raw['xyz'], raw['rotation'], 1.)


def test_residual_motion_warm_start_maps_survivors_and_newborns():
    cfg = residual_test_config(motion_init_decay=.6, motion_rotation_step_rad=.001)
    raw = residual_raw_fixture(count=6)
    old = Q.QuantizedResidualMotion(cfg)
    old.setup(raw['xyz'], raw['rotation'], 1.)
    with torch.no_grad():
        old.gs_translation[:, 0] = torch.tensor([.1, .2, .3, .4, .5, .6])
        old.gs_rotation[:, 2] = torch.tensor([.1, .2, .3, .4, .5, .6])
    old.committed = True
    old_xyz, old_rotation = old(cfg.motion_steps)
    newborn_xyz = torch.tensor([[10., 0., 0.], [11., 0., 0.]])
    newborn_rotation = torch.tensor([[1., 0., 0., 0.]] * 2)
    previous_xyz = torch.cat((old_xyz.detach(), newborn_xyz))
    previous_rotation = torch.cat((old_rotation.detach(), newborn_rotation))
    deleted = torch.tensor([1, 6], dtype=torch.int32)
    rows = Q.survivor_rows(len(previous_xyz), deleted)
    assert rows.tolist() == [0, 2, 3, 4, 5, 7]
    next_motion = Q.QuantizedResidualMotion(cfg)
    next_motion.restore(copy.deepcopy(old.capture()))
    next_motion.setup(previous_xyz[rows], previous_rotation[rows], 1.,
                      previous_rows=rows, previous_new_count=2)
    old_translation, old_relative_rotation = old.final_quantized_total_motion()
    initialized_xyz, _ = next_motion(cfg.motion_steps)
    expected_translation = torch.zeros_like(previous_xyz[rows])
    expected_translation[:5] = old_translation[rows[:5]] * cfg.motion_init_decay
    expected_translation = (expected_translation / cfg.motion_xyz_step_extent).round() * cfg.motion_xyz_step_extent
    torch.testing.assert_close(initialized_xyz - previous_xyz[rows], expected_translation, rtol=0, atol=2e-6)
    _, initialized_rotation = next_motion.final_quantized_total_motion()
    target_rotation = R.rotation_vector_quaternion(
        Q.shortest_rotation_vector(old_relative_rotation[rows[:5]]) * cfg.motion_init_decay)
    delta = R.quaternion_multiply(initialized_rotation[:5], Q.inverse_unit_quaternion(target_rotation))
    assert float(Q.shortest_rotation_vector(delta).norm(dim=-1).max()) < .002
    assert float(Q.shortest_rotation_vector(initialized_rotation[5:]).norm()) < .002
    restored = Q.QuantizedResidualMotion(cfg)
    restored.restore(copy.deepcopy(next_motion.capture()))
    for left, right in zip(next_motion(cfg.motion_steps), restored(cfg.motion_steps)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for retained_rows, newborn_count in ((torch.arange(6), 2), (torch.arange(6), 0)):
        candidate = Q.QuantizedResidualMotion(cfg)
        candidate.restore(copy.deepcopy(old.capture()))
        candidate.setup(old_xyz.detach(), old_rotation.detach(), 1.,
                        previous_rows=retained_rows, previous_new_count=newborn_count)
        torch.testing.assert_close(candidate(cfg.motion_steps)[0] - old_xyz.detach(),
                                   (old_translation * cfg.motion_init_decay / .1).round() * .1,
                                   rtol=0, atol=2e-6)
    for bad_rows, bad_count, bad_xyz in (
            (torch.tensor([0, 0, 3, 4, 5, 7]), 2, previous_xyz[rows]),
            (rows, 1, previous_xyz[rows]),
            (rows, 2, previous_xyz[rows].roll(1, 0))):
        candidate = Q.QuantizedResidualMotion(cfg)
        candidate.restore(copy.deepcopy(old.capture()))
        try:
            candidate.setup(bad_xyz, previous_rotation[rows], 1.,
                            previous_rows=bad_rows, previous_new_count=bad_count)
        except ValueError as exc:
            assert 'mapping' in str(exc)
        else:
            raise AssertionError('Invalid survivor mapping must be rejected')


def test_residual_birth_budget_window_and_stable_selection():
    cfg = residual_test_config(densification_stats_steps=2, max_new_gaussians=2)
    cfg.validate()
    assert cfg.densification_stats_start == 3
    assert residual_test_config().densification_stats_start == 1
    eligible = torch.tensor([True, True, False, True, True])
    scores = torch.tensor([2., 5., 100., 5., 1.])
    assert GAUSS.select_budget_mask(eligible, scores, 2).tolist() == [False, True, False, True, False]
    assert GAUSS.select_budget_mask(eligible, scores, 1).tolist() == [False, True, False, False, False]
    assert GAUSS.select_budget_mask(eligible, scores, 0).count_nonzero() == 0
    assert torch.equal(GAUSS.select_budget_mask(eligible, scores, None), eligible)
    for name, bad in (('densification_stats_steps', 0), ('densification_stats_steps', 5),
                      ('densification_stats_steps', True), ('densification_stats_steps', 1.5),
                      ('max_new_gaussians', 0), ('max_new_gaussians', False),
                      ('max_new_gaussians', 2.0)):
        try:
            residual_test_config(**{name: bad}).validate()
        except ValueError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f'{name}={bad!r} must be rejected')


def gated_fixture(count=18, **config_values):
    torch.manual_seed(23)
    raw = residual_raw_fixture(count)
    motion = GR.GatedResidualMotion(gated_residual_test_config(**config_values))
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    return raw, motion


def set_gates(motion, choices):
    """Set every node explicitly; unspecified levels use KEEP."""
    with torch.no_grad():
        for level, logits in enumerate(motion.gate_logits):
            logits.fill_(-20.)
            logits[:, choices.get(level, GR.KEEP)] = 20.


def test_gated_keep_cancels_ancestor_and_closed_descendant_motion():
    raw, motion = gated_fixture()
    set_gates(motion, {0: GR.SPLIT, 1: GR.SHARE, 2: GR.SPLIT})
    with torch.no_grad():
        motion.gate_logits[1][0] = torch.tensor([20., -20., -20.])
        motion.gate_logits[1][1] = torch.tensor([-20., -20., 20.])
        motion.gate_logits[2][motion.parents[1] == 1] = torch.tensor([20., -20., -20.])
        motion.region_translation[0][:, 0] = .3
        motion.region_rotation[0][:, 2] = .4
        motion.region_translation[1][:, 1] = 10.
        motion.region_rotation[1][:, 1] = 2.
        motion.region_translation[2][:, 0] = 20.
        motion.gs_translation[:, 0] = 30.
    xyz, rotation = motion(2)
    kept = (motion.index[1] == 0) | (motion.index[1] == 1)
    torch.testing.assert_close(xyz[kept], raw['xyz'][kept], rtol=0, atol=0)
    torch.testing.assert_close(rotation[kept], raw['rotation'][kept], rtol=0, atol=0)
    assert not torch.equal(xyz[~kept], raw['xyz'][~kept])
    selected = motion.gate_state_arrays()
    assert not selected['visited'][2][motion.parents[1] == 0].any()
    assert not selected['gs_leaf_open'][kept].any()
    assert motion.storage_parts(2, hard=True)['gs_motion_bytes'] == 0


def test_gated_share_rigid_rotation_about_advected_pivot():
    raw, motion = gated_fixture()
    set_gates(motion, {0: GR.SHARE})
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .2
        motion.region_rotation[0][:, 2] = 1.6
        motion.region_translation[1][:, 0] = 100.
        motion.region_translation[2][:, 0] = 100.
        motion.gs_translation[:, 0] = 100.
    xyz, rotation = motion(2)
    pivot = raw['xyz'].mean(0, keepdim=True).expand_as(raw['xyz'])
    expected_xyz, expected_rotation = R.apply_region_rigid(
        raw['xyz'], raw['rotation'], pivot,
        torch.tensor([[.2, 0., 0.]]).expand_as(raw['xyz']),
        torch.tensor([[0., 0., 1.6]]).expand_as(raw['xyz']))
    torch.testing.assert_close(xyz, expected_xyz, rtol=0, atol=2e-6)
    torch.testing.assert_close(rotation, expected_rotation, rtol=0, atol=2e-6)
    torch.testing.assert_close(torch.pdist(xyz), torch.pdist(raw['xyz']), rtol=1e-5, atol=2e-6)
    assert not motion.gate_state_arrays()['visited'][1].any()
    assert motion.storage_parts(2, hard=True)['gs_motion_bytes'] == 0


def test_gated_split_inherits_rigid_motion_then_adds_local_correction():
    raw, motion = gated_fixture()
    set_gates(motion, {0: GR.SPLIT, 1: GR.SHARE, 2: GR.SHARE})
    child = 1
    leaf = int(torch.where(motion.parents[1] == child)[0][0])
    point = int(torch.where(motion.index[2] == leaf)[0][0])
    with torch.no_grad():
        motion.gate_logits[1][child] = torch.tensor([-20., -20., 20.])
        motion.gate_logits[2][leaf] = torch.tensor([-20., -20., 20.])
        motion.region_translation[0][:, 0] = .2
        motion.region_rotation[0][:, 2] = .2
        motion.region_translation[1][child, 1] = .3
        motion.region_rotation[1][child, 2] = .1
        motion.region_translation[2][leaf, 2] = .4
        motion.gs_translation[point, 0] = .1
    xyz, rotation = motion(2)
    base = raw['xyz'][point:point + 1]
    orientation = raw['rotation'][point:point + 1]
    root_pivot = motion.active_layout['pivots'][0][motion.index[0, point]:motion.index[0, point] + 1]
    child_pivot = motion.active_layout['pivots'][1][child:child + 1]
    leaf_pivot = motion.active_layout['pivots'][2][leaf:leaf + 1]
    trans0 = torch.tensor([[.2, 0., 0.]])
    rot0 = torch.tensor([[0., 0., .2]])
    p1, _ = R.apply_region_rigid(child_pivot, orientation, root_pivot, trans0, rot0)
    p2, _ = R.apply_region_rigid(leaf_pivot, orientation, root_pivot, trans0, rot0)
    expected, orientation = R.apply_region_rigid(base, orientation, root_pivot, trans0, rot0)
    expected, orientation = R.apply_region_rigid(expected, orientation, p1,
                                                  torch.tensor([[0., .3, 0.]]),
                                                  torch.tensor([[0., 0., .1]]))
    p2, _ = R.apply_region_rigid(p2, raw['rotation'][point:point + 1], p1,
                                 torch.tensor([[0., .3, 0.]]),
                                 torch.tensor([[0., 0., .1]]))
    expected, orientation = R.apply_region_rigid(expected, orientation, p2,
                                                  torch.tensor([[0., 0., .4]]),
                                                  torch.zeros(1, 3))
    expected[:, 0] += .1
    torch.testing.assert_close(xyz[point:point + 1], expected, rtol=0, atol=2e-6)
    torch.testing.assert_close(rotation[point:point + 1], orientation, rtol=0, atol=2e-6)
    other = int(torch.where(motion.index[1] != child)[0][0])
    root_only, _ = R.apply_region_rigid(raw['xyz'][other:other + 1],
                                         raw['rotation'][other:other + 1], root_pivot,
                                         trans0, rot0)
    torch.testing.assert_close(xyz[other:other + 1], root_only, rtol=0, atol=2e-6)


def test_gated_hard_forward_soft_gradients_and_refinement_freeze():
    raw, motion = gated_fixture()
    set_gates(motion, {0: GR.KEEP, 1: GR.KEEP, 2: GR.KEEP})
    with torch.no_grad():
        for logits in motion.gate_logits:
            logits[:, :] = torch.tensor([.2, .1, 0.])
        motion.region_translation[0][:, 0] = .2
        motion.region_translation[1][:, 1] = .3
        motion.region_translation[2][:, 2] = .4
        motion.gs_translation[:, 0] = .5
    xyz, rotation = motion(2, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)
    (xyz[:, 0].sum() + xyz[:, 1].sum() + xyz[:, 2].sum()).backward()
    for logits in motion.gate_logits:
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0
    assert motion.gs_translation.grad is not None
    assert torch.isfinite(motion.gs_translation.grad).all()
    assert motion.gs_translation.grad.abs().sum() > 0
    motion.optimizer.zero_grad(set_to_none=True)
    motion(4, training=True)
    assert all(not value.requires_grad for value in motion.gate_logits)
    frozen = [value.clone() for value in motion.gate_state_arrays()['states']]
    for before, after in zip(frozen, motion.gate_state_arrays()['states']):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def storage_fixture(dtype=torch.float32):
    raw = residual_raw_fixture(18)
    xyz = raw['xyz'].to(dtype)
    rotation = raw['rotation'].to(dtype)
    motion = GR.GatedResidualMotion(storage_test_config())
    motion.setup(xyz, rotation, 1.)
    return motion


def assert_storage_parts(parts, gate, region, gs, dtype=torch.float32):
    expected = dict(gate_bytes=gate, region_motion_bytes=region,
                    gs_motion_bytes=gs)
    assert set(parts) == set(expected)
    for name, value in expected.items():
        torch.testing.assert_close(parts[name], torch.tensor(float(value), dtype=dtype),
                                   rtol=0, atol=0)


def test_gated_storage_counts_reachable_bytes_and_dtype():
    motion = storage_fixture()
    n = len(motion.base_xyz)
    assert motion.counts == [1, 3, 9]
    assert motion.reference_storage_bytes(n) == n * 6 * 4
    set_gates(motion, {0: GR.KEEP, 1: GR.SPLIT, 2: GR.SPLIT})
    assert_storage_parts(motion.storage_parts(2, hard=True), 1, 0, 0)
    set_gates(motion, {0: GR.SHARE, 1: GR.SPLIT, 2: GR.SPLIT})
    assert_storage_parts(motion.storage_parts(2, hard=True), 1, 24, 0)
    set_gates(motion, {0: GR.SPLIT, 1: GR.KEEP, 2: GR.SPLIT})
    assert_storage_parts(motion.storage_parts(2, hard=True), 4, 24, 0)
    set_gates(motion, {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SPLIT})
    assert_storage_parts(motion.storage_parts(2, hard=True), 13, 13 * 24, n * 24)
    before_hard = motion.storage_parts(2, hard=True)
    before_soft = motion.storage_parts(2)
    with torch.no_grad():
        for value in motion.region_translation:
            value.fill_(1000.)
        for value in motion.region_rotation:
            value.fill_(1.)
        motion.gs_translation.fill_(1000.)
        motion.gs_rotation.fill_(1.)
    for before, after in zip((before_hard, before_soft),
                             (motion.storage_parts(2, hard=True), motion.storage_parts(2))):
        for name in before:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)
    wide = storage_fixture(torch.float64)
    set_gates(wide, {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SPLIT})
    assert wide.reference_storage_bytes(n) == n * 6 * 8
    assert_storage_parts(wide.storage_parts(2, hard=True), 13, 13 * 48,
                         n * 48, dtype=torch.float64)


def test_gated_storage_soft_gradient_and_refine_matches_hard():
    motion = storage_fixture()
    with torch.no_grad():
        for logits in motion.gate_logits:
            logits[:, :] = torch.tensor([.2, .1, 0.])
    assert motion.storage_scale(1) == 0.
    assert motion.storage_scale(2) == .5
    assert motion.storage_scale(3) == 1.
    soft = motion.storage_parts(2)
    total = sum(soft.values())
    assert torch.isfinite(total)
    total.backward()
    for logits in motion.gate_logits:
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0
    motion(4, training=True)
    assert all(not logits.requires_grad for logits in motion.gate_logits)
    frozen_soft = motion.storage_parts(4)
    frozen_hard = motion.storage_parts(4, hard=True)
    for name in frozen_soft:
        torch.testing.assert_close(frozen_soft[name], frozen_hard[name], rtol=0, atol=0)


class SyntheticQuantizedBirthView:
    """A quantized birth candidate with one retained and one rejected row."""
    def __init__(self, reference, config):
        self.reference = reference
        self.source = torch.tensor([0, len(reference['xyz']) - 1], dtype=torch.long)
        self.keep = torch.tensor([True, False])
        self.steps = Q.attribute_steps(config, 1.)
        self.symbols = {name: torch.zeros_like(value[self.source], dtype=torch.int32)
                        for name, value in reference.items()}
        self.symbols['xyz'][0, 2] = 1
        self.symbols['rotation'][0, 1] = 1
        self.symbols['logscale'][0, 0] = 2
        self.symbols['opacity'][0, 0] = -1
        self.symbols['sh_dc'][0, 0, 1] = 1
        self.symbols['sh_rest'][0, 1, 2] = -1
        self.symbols['xyz'][1, 0] = 100  # Rejected row never enters materialized raw.

    def raw_value(self, name):
        predictor = self.reference[name][self.source]
        if name == 'rotation':
            predictor = torch.nn.functional.normalize(predictor, dim=-1)
        return predictor + self.symbols[name].to(predictor) * self.steps[name]


def materialize_gated(base_raw, new_view, motion):
    materialized, sources = Q.materialize_quantized_gaussians(base_raw, new_view)
    layout = motion.register_materialized_layout(materialized['xyz'], sources)
    return materialized, sources, layout


def test_gated_two_frame_materialization_preserves_keep_and_birth_owners():
    previous, motion = gated_fixture(motion_init_decay=.6)
    set_gates(motion, {0: GR.SPLIT, 1: GR.SHARE, 2: GR.SHARE})
    with torch.no_grad():
        motion.gate_logits[1][0] = torch.tensor([20., -20., -20.])
        motion.region_translation[0][:, 0] = .5
    moved_xyz, moved_rotation = motion(4)
    base_raw = dict(previous, xyz=moved_xyz, rotation=moved_rotation)
    new_view = SyntheticQuantizedBirthView(base_raw, motion.config)
    direct, sources, layout = materialize_gated(base_raw, new_view, motion)
    assert sources.tolist() == [0]
    assert len(direct['xyz']) == len(previous['xyz']) + 1
    assert all(not value.requires_grad for value in direct.values())
    torch.testing.assert_close(layout['index'][:, -1], motion.index[:, 0], rtol=0, atol=0)
    for name in Q.RAW_FIELDS:
        expected = new_view.raw_value(name)[new_view.keep]
        if name == 'rotation':
            expected = torch.nn.functional.normalize(expected, dim=-1)
        torch.testing.assert_close(direct[name][-1:], expected, rtol=0, atol=0)
    kept = motion.gate_state_arrays()['gs_terminal'] == GR.KEEP
    for name in ('xyz', 'rotation'):
        torch.testing.assert_close(direct[name][:-1][kept], previous[name][kept], rtol=0, atol=0)
    motion.committed = True
    resumed = GR.GatedResidualMotion(motion.config)
    resumed.restore(copy.deepcopy(motion.capture()))
    deleted, protection = resumed.compensation_deletions(torch.sigmoid(direct['opacity']), 1)
    assert protection['deleted_keep_count'] == 0
    rows = Q.survivor_rows(len(direct['xyz']), deleted)
    resumed.setup(direct['xyz'][rows], direct['rotation'][rows], 1.,
                  previous_rows=rows, previous_new_count=1)
    torch.testing.assert_close(resumed.index, layout['index'][:, rows], rtol=0, atol=0)
    for before, after in zip(motion.gate_logits, resumed.gate_logits):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    next_xyz, next_rotation = resumed(4)
    surviving_keep = resumed.gate_state_arrays()['gs_terminal'] == GR.KEEP
    torch.testing.assert_close(next_xyz[surviving_keep], direct['xyz'][rows][surviving_keep], rtol=0, atol=0)
    next_base = {name: value[rows] for name, value in direct.items()}
    next_base.update(xyz=next_xyz, rotation=next_rotation)
    materialized, empty_sources, next_layout = materialize_gated(next_base, None, resumed)
    assert empty_sources.numel() == 0
    for name in Q.RAW_FIELDS:
        torch.testing.assert_close(materialized[name], next_base[name], rtol=0, atol=0)
    empty_view = SyntheticQuantizedBirthView(next_base, resumed.config)
    empty_view.keep[:] = False
    empty_raw, empty_kept, empty_layout = materialize_gated(next_base, empty_view, resumed)
    assert empty_kept.numel() == 0
    torch.testing.assert_close(empty_layout['index'], next_layout['index'], rtol=0, atol=0)
    for name in Q.RAW_FIELDS:
        torch.testing.assert_close(empty_raw[name], materialized[name], rtol=0, atol=0)


def test_gated_checkpoint_restores_topology_refinement_and_warm_start():
    raw, old = gated_fixture(motion_init_decay=.6)
    set_gates(old, {0: GR.SHARE})
    with torch.no_grad():
        old.region_translation[0][:, 0] = .5
    old.committed = True
    old_xyz, old_rotation = old(2)
    materialized, _, layout = materialize_gated(
        dict(raw, xyz=old_xyz, rotation=old_rotation), None, old)
    torch.testing.assert_close(materialized['xyz'], old_xyz.detach(), rtol=0, atol=0)
    saved = copy.deepcopy(old.capture())
    restored = GR.GatedResidualMotion(old.config)
    restored.restore(saved)
    assert restored.committed
    torch.testing.assert_close(restored.materialized_layout['index'], layout['index'], rtol=0, atol=0)
    for original, recovered in zip(old.gate_logits, restored.gate_logits):
        torch.testing.assert_close(recovered, original, rtol=0, atol=0)
    restored.setup(old_xyz.detach(), old_rotation.detach(), 1.,
                   previous_rows=torch.arange(len(raw['xyz'])), previous_new_count=0)
    assert torch.equal(restored.index, old.index)
    assert torch.equal(restored.gate_state_arrays()['states'][0],
                       old.gate_state_arrays()['states'][0])
    initialized_xyz, _ = restored(2)
    expected_offset = torch.zeros_like(old_xyz)
    expected_offset[:, 0] = .3
    torch.testing.assert_close(initialized_xyz - old_xyz, expected_offset, rtol=0, atol=2e-6)
    restored(4, training=True)
    assert all(not value.requires_grad for value in restored.gate_logits)
    midframe = GR.GatedResidualMotion(old.config)
    midframe.restore(copy.deepcopy(restored.capture()))
    assert midframe.frozen_states is not None
    assert all(not value.requires_grad for value in midframe.gate_logits)
    for left, right in zip(restored(4), midframe(4)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(midframe.previous_layout['index'], layout['index'], rtol=0, atol=0)


def test_gated_candidate_motion_gradients_do_not_depend_on_gate_probabilities():
    """Equal candidate poses and upstream signals give equal motion gradients."""
    gradients = []
    for logits in ([.2, .1, 0.], [30., 0., -30.]):
        raw, motion = gated_fixture()
        with torch.no_grad():
            for gate in motion.gate_logits:
                gate[:] = torch.tensor(logits)
            for level in range(3):
                motion.region_translation[level][:, level] = .2
                motion.region_rotation[level][:, (level + 1) % 3] = .1
            motion.gs_translation[:, 0] = .3
            motion.gs_rotation[:, 1] = .2
        xyz, rotation = motion(2, training=True)
        # Even saturated ancestor KEEP gates do not weaken motion candidates.
        torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
        torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)
        loss = (xyz[0] * torch.tensor([.7, -.2, .4])).sum()
        loss = loss + (rotation[0] * torch.tensor([0., .2, .3, -.4])).sum()
        loss.backward()
        parameters = [*motion.region_translation, *motion.region_rotation,
                      motion.gs_translation, motion.gs_rotation]
        assert all(value.grad is not None and torch.isfinite(value.grad).all()
                   for value in parameters)
        assert all(value.grad.abs().sum() > 0 for value in parameters)
        gradients.append([value.grad.clone() for value in parameters])
        assert all(value.grad is not None and torch.isfinite(value.grad).all()
                   for value in motion.gate_logits)
        if logits[0] < 1:
            assert all(value.grad.abs().sum() > 0 for value in motion.gate_logits)
    for ordinary, saturated in zip(*gradients):
        torch.testing.assert_close(ordinary, saturated, rtol=0, atol=0)


def test_gated_translation_gradient_has_no_depth_factor_and_refinement_is_hard():
    raw, motion = gated_fixture()
    with torch.no_grad():
        for gate in motion.gate_logits:
            gate[:] = torch.tensor([20., 0., -20.])
    xyz, _ = motion(2, training=True)
    xyz[0].sum().backward()
    for level, parameter in enumerate(motion.region_translation):
        expected = torch.zeros_like(parameter)
        expected[motion.index[level, 0]] = 1
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    expected = torch.zeros_like(motion.gs_translation)
    expected[0] = 1
    torch.testing.assert_close(motion.gs_translation.grad, expected, rtol=0, atol=0)
    motion.optimizer_step()
    torch.testing.assert_close(motion(2)[0], raw['xyz'], rtol=0, atol=0)
    # Once the cut is frozen, unused candidates must receive no updates.
    set_gates(motion, {0: GR.SHARE, 1: GR.SPLIT, 2: GR.SPLIT})
    motion(4, training=True)[0].sum().backward()
    assert motion.region_translation[0].grad.abs().sum() > 0
    for parameter in [*motion.region_translation[1:], motion.gs_translation]:
        assert parameter.grad is None or parameter.grad.count_nonzero() == 0
    assert all(value.grad is None for value in motion.gate_logits)


def test_gated_compensation_protects_keep_and_stable_opacity_order():
    raw, motion = gated_fixture()
    set_gates(motion, {0: GR.SPLIT, 1: GR.SHARE, 2: GR.KEEP})
    with torch.no_grad():
        motion.gate_logits[1][0] = torch.tensor([20., -20., -20.])
    motion(4)
    motion.committed = True
    protected = motion.gate_state_arrays()['gs_terminal'] == GR.KEEP
    assert protected.any() and not protected.all()
    eligible_old = torch.where(~protected)[0]
    opacity = torch.full((len(raw['xyz']) + 2,), .8)
    opacity[:len(protected)][protected] = .001
    opacity[eligible_old[:2]] = .02  # Stable ties retain original row order.
    deleted, info = motion.compensation_deletions(opacity, 2)
    torch.testing.assert_close(deleted, eligible_old[:2], rtol=0, atol=0)
    assert info['protected_keep_old_count'] == int(protected.sum())
    assert info['deleted_count'] == 2 and info['deleted_keep_count'] == 0
    restored = GR.GatedResidualMotion(motion.config)
    restored.restore(copy.deepcopy(motion.capture()))
    torch.testing.assert_close(restored.compensation_deletions(opacity, 2)[0],
                               deleted, rtol=0, atol=0)


def test_gated_all_keep_deletes_only_newborns_and_zero_budget_is_empty():
    raw, motion = gated_fixture()
    set_gates(motion, {0: GR.KEEP, 1: GR.SPLIT, 2: GR.SPLIT})
    motion(4)
    motion.committed = True
    count = len(raw['xyz'])
    opacity = torch.cat((torch.full((count,), .001), torch.tensor([.9, .8])))
    deleted, info = motion.compensation_deletions(opacity, 2)
    torch.testing.assert_close(deleted, torch.tensor([count + 1, count]), rtol=0, atol=0)
    assert info['protected_keep_old_count'] == count
    assert info['deleted_keep_count'] == 0
    deleted, info = motion.compensation_deletions(opacity[:count], 0)
    assert deleted.numel() == 0 and info['requested_count'] == 0
    try:
        motion.compensation_deletions(opacity, 1)
    except ValueError as exc:
        assert 'counts' in str(exc)
    else:
        raise AssertionError('Protection must reject an inconsistent old/new mapping')


def test_frame_preparation_uses_keep_protection_and_reports_history():
    """Run production frame preparation without importing a CUDA renderer."""
    module_path = project_root / 'pipeline' / 'ReCon-GS' / 'module.py'
    module_tree = ast.parse(module_path.read_text())
    method = next(node for node in ast.walk(module_tree)
                  if isinstance(node, ast.FunctionDef) and node.name == 'prepare_quantized_frame')
    namespace = dict(torch=torch, time=time, model_raw=Q.model_raw,
                     reference_digest=Q.reference_digest, survivor_rows=Q.survivor_rows,
                     GaussianModel=lambda degree: SimpleNamespace())
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(module_path), 'exec'), namespace)
    raw, motion = gated_fixture(motion_init_decay=0., tree_rebuild_interval=5)
    set_gates(motion, {0: GR.SPLIT, 1: GR.SHARE, 2: GR.KEEP})
    with torch.no_grad():
        motion.gate_logits[1][0] = torch.tensor([20., -20., -20.])
    xyz, rotation = motion(4)
    protected = motion.gate_state_arrays()['gs_terminal'] == GR.KEEP
    previous = {name: torch.cat((value, value[:2]), dim=0) for name, value in raw.items()}
    previous['xyz'][:len(xyz)] = xyz.detach()
    previous['rotation'][:len(xyz)] = rotation.detach()
    opacity = torch.full((len(xyz) + 2, 1), .8)
    opacity[:len(xyz)][protected] = .001
    opacity[-2:] = .9
    previous['opacity'] = torch.logit(opacity)
    motion.register_materialized_layout(previous['xyz'], torch.tensor([0, 1]))
    motion.committed = True

    class OldGaussians:
        def __init__(self):
            for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
                setattr(self, attr, previous[name].clone())

        @property
        def get_xyz(self):
            return self._xyz

        @property
        def get_xyz_ori(self):
            return self._xyz

        @property
        def get_opacity(self):
            return torch.sigmoid(self._opacity)

        def prune_points(self, deleted):
            rows = Q.survivor_rows(len(self._xyz), deleted)
            for attr in Q.RAW_ATTRIBUTES:
                setattr(self, attr, getattr(self, attr)[rows])

    subject = SimpleNamespace(
        gaussians=OldGaussians(), residual_motion=motion, current_frame=5,
        config=SimpleNamespace(quantized_residual=motion.config, sh_degree=1),
        quant_first_frame_gs_count=len(xyz), quant_previous_new_count=2,
        quant_point_birth_frame=torch.tensor([0] * len(xyz) + [1, 1]),
        datamodule=SimpleNamespace(cameras_extent=1.), renderer=SimpleNamespace(),
        trainer=SimpleNamespace(synchronize=lambda: None),
        deletion_diagnostic_view=lambda count: None, freeze_quantized_base=lambda: None,
        new_quant_birth_control=lambda: {})
    namespace['prepare_quantized_frame'](subject)
    deleted = subject.quant_deleted_indices.long()
    expected = torch.where(~protected)[0][:2]
    torch.testing.assert_close(deleted, expected, rtol=0, atol=0)
    kept_rows = Q.survivor_rows(len(previous['xyz']), deleted)
    torch.testing.assert_close(subject.gaussians.get_xyz, previous['xyz'][kept_rows], rtol=0, atol=0)
    assert subject.quant_birth_control['deletion_protection']['deleted_keep_count'] == 0
    history = subject.quant_birth_control['motion_history']
    assert history['old_motion_rows'] == len(xyz) - 2
    assert history['newborn_rows'] == 2 and history['first_frame_rows'] == 0
    assert history['initialization_decay'] == 0.
    assert motion.tree_setup['action'] == 'rebuild'
    assert motion.tree_setup['initialization'] == 'motion_knn_gate_membership'
    assert motion.stage_seconds['tree_rebuild'] > 0


def test_tree_rebuild_config_validation():
    gated_residual_test_config(tree_rebuild_interval=5, tree_rebuild_neighbors=3).validate()
    for name, value in [('tree_rebuild_interval', -1), ('tree_rebuild_interval', True),
                        ('tree_rebuild_interval', 2.5), ('tree_rebuild_neighbors', 0),
                        ('tree_rebuild_neighbors', False)]:
        try:
            gated_residual_test_config(**{name: value}).validate()
        except ValueError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f'{name}={value} must be rejected')
    try:
        residual_test_config(tree_rebuild_interval=5).validate()
    except ValueError as exc:
        assert 'enable_learned_gates' in str(exc)
    else:
        raise AssertionError('Nongated mode must not silently ignore the interval')


def test_rebuild_knn_translation_and_quaternion_inheritance():
    centers = torch.tensor([[0., 0., 0.], [2., 0., 0.], [100., 0., 0.]])
    values = torch.zeros(3, 6)
    values[:, 0] = torch.tensor([2., 4., 90.])
    values[:2, 5] = torch.tensor([170., -170.]) * torch.pi / 180
    inherited = GR._inherit_region_priors(
        centers, torch.tensor([[.1, 0., 0.]]), values, 2)
    torch.testing.assert_close(inherited[:, :3], torch.tensor([[3., 0., 0.]]), rtol=0, atol=1e-6)
    # Rotation vectors must not be averaged directly: +170/-170 mean ~180, not 0.
    quaternion = R.rotation_vector_quaternion(inherited[:, 3:])
    torch.testing.assert_close(quaternion.abs(), torch.tensor([[0., 0., 0., 1.]]), rtol=0, atol=1e-6)
    single = GR._inherit_region_priors(
        centers[:1], centers[1:2], values[:1], 3)
    torch.testing.assert_close(single, values[:1], rtol=0, atol=1e-6)


def test_periodic_rebuild_preserves_base_pose_and_inherits_only_same_level():
    raw, old = gated_fixture(count=54, tree_rebuild_interval=5, motion_init_decay=.6)
    set_gates(old, {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SPLIT})
    with torch.no_grad():
        for level in range(3):
            old.region_translation[level][:, 0] = level + 1.
        old.gs_translation[:, 1] = torch.arange(54).remainder(2) * 10.
    xyz, rotation = old(4)
    materialize_gated(dict(raw, xyz=xyz, rotation=rotation), None, old)
    old.committed = True
    packet = copy.deepcopy(old.capture())
    new_index, _, _ = Q.stable_layout(xyz, old.config.fine_region_gs)
    assert not torch.equal(new_index, old.index), 'Fixture must require regrouping'
    for frame, action in [(4, 'reuse'), (5, 'rebuild'), (6, 'reuse'), (10, 'rebuild')]:
        rebuilt = GR.GatedResidualMotion(old.config)
        rebuilt.restore(copy.deepcopy(packet))
        rebuilt.setup(xyz.detach(), rotation.detach(), 1., frame_index=frame)
        assert rebuilt.tree_setup['action'] == action
        torch.testing.assert_close(rebuilt.base_xyz, xyz, rtol=0, atol=0)
        torch.testing.assert_close(rebuilt.base_rotation, rotation, rtol=0, atol=0)
        torch.testing.assert_close(rebuilt.index, new_index if action == 'rebuild' else old.index,
                                   rtol=0, atol=0)
        for level in range(3):
            expected = torch.zeros_like(rebuilt.region_translation[level])
            expected[:, 0] = .6 * (level + 1)
            torch.testing.assert_close(rebuilt.region_translation[level], expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(rebuilt.gs_translation, old.gs_translation * .6, rtol=0, atol=1e-6)
        assert not rebuilt.optimizer.state
        if action == 'rebuild':
            recovered = GR.GatedResidualMotion(old.config)
            recovered.restore(copy.deepcopy(rebuilt.capture()))
            assert recovered.tree_setup == rebuilt.tree_setup
            expected_pose = rebuilt(2)
            recovered_pose = recovered(2)
            for expected, actual in zip(expected_pose, recovered_pose):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gate_membership_majority_agreement_conflict_and_relabeling():
    logits = torch.tensor([[8., -2., -3.], [-4., 7., -1.], [5., -1., -2.]])
    initial = [-1., 0., -.5]
    # New region IDs deliberately differ from their dominant old IDs.
    old = torch.tensor([0, 0, 1, 0, 1, 0, 2, 2, 2, 1, 1])
    new = torch.tensor([3, 3, 3, 0, 0, 1, 1, 1, 1, 2, 2])
    inherited = GR._inherit_gate_logits(old, new, logits, 4, initial)
    expected = torch.stack((torch.tensor(initial), logits[2], logits[1], logits[0]))
    torch.testing.assert_close(inherited, expected, rtol=0, atol=0)
    # No source has a strict majority, but all source states agree.
    agreed = GR._inherit_gate_logits(torch.tensor([0, 2]), torch.tensor([0, 0]),
                                     logits, 1, initial)
    torch.testing.assert_close(agreed[0], (logits[0] + logits[2]) / 2, rtol=0, atol=0)


def test_rebuild_unchanged_members_keep_gates_and_static_pose():
    raw, motion = gated_fixture(tree_rebuild_interval=5, motion_init_decay=.6)
    set_gates(motion, {0: GR.SPLIT, 1: GR.SHARE, 2: GR.KEEP})
    with torch.no_grad():
        motion.gate_logits[1][0] = torch.tensor([20., -20., -20.])
        # Small motion preserves the partition while KNN mixes the three middle regions.
        motion.region_translation[1][1:, 0] = .3
    protected = motion.gate_state_arrays()['gs_terminal'] == GR.KEEP
    xyz, rotation = motion(4)
    materialize_gated(dict(raw, xyz=xyz, rotation=rotation), None, motion)
    motion.committed = True
    old_index = motion.index.clone()
    middle_logits = motion.gate_logits[1].detach().clone()
    motion.setup(xyz.detach(), rotation.detach(), 1., frame_index=5)
    assert motion.tree_setup['action'] == 'rebuild'
    torch.testing.assert_close(motion.index, old_index, rtol=0, atol=0)
    torch.testing.assert_close(motion.gate_logits[1], middle_logits, rtol=0, atol=0)
    moved, rotated = motion(2)
    assert bool(protected.any())
    torch.testing.assert_close(moved[protected], xyz[protected], rtol=0, atol=0)
    torch.testing.assert_close(rotated[protected], rotation[protected], rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in motion.gate_logits)
    assert motion.frozen_states is None


def test_rebuild_does_not_inherit_closed_candidate_motion():
    raw, motion = gated_fixture(tree_rebuild_interval=5, motion_init_decay=.6)
    set_gates(motion, {0: GR.KEEP, 1: GR.SPLIT, 2: GR.SPLIT})
    with torch.no_grad():
        for value in motion.region_translation:
            value.fill_(100.)
        motion.gs_translation.fill_(100.)
    materialize_gated(raw, None, motion)
    motion.committed = True
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=5)
    for value in list(motion.region_translation) + [motion.gs_translation]:
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
    xyz, rotation = motion(2)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)


if __name__ == '__main__':
    checks = [value for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for check in checks:
        check()
    print(f'All {len(checks)} nested prototype numerical checks passed.')
