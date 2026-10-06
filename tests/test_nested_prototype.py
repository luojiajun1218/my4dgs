"""Numerical checks without loading the renderer or a scene."""
import importlib
import ast
import copy
import itertools
import json
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F

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
SH = importlib.import_module('pipeline.ReCon-GS.model.sparse_hierarchical')
FC = importlib.import_module('pipeline.ReCon-GS.model.flat_cluster')
BR = importlib.import_module('pipeline.ReCon-GS.model.binary_refinement')
MC = importlib.import_module('pipeline.ReCon-GS.model.motion_clustering')
SL_CODEC = importlib.import_module('pipeline.ReCon-GS.model.sparse_layout_codec')
GS_ACTIVITY = importlib.import_module('pipeline.ReCon-GS.model.gs_activity_cost')
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


class SlowWorkerOnlyPairDataset(WorkerOnlyPairDataset):
    def __getitem__(self, index):
        time.sleep(.5)
        return super().__getitem__(index)

    def get_frame_dataset(self, frame):
        return SlowWorkerOnlyPairDataset(frame)


def test_selective_pairs_use_workers_and_frame_cache():
    before_workers = {worker.pid for worker in multiprocessing.active_children()}
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
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        new_workers = {worker.pid for worker in multiprocessing.active_children()} - before_workers
        if not new_workers:
            break
        time.sleep(.05)
    assert not new_workers


def test_selective_pair_loader_timeout_is_bounded_and_cleans_worker():
    before_workers = {worker.pid for worker in multiprocessing.active_children()}
    data = DATA.DataModule.__new__(DATA.DataModule)
    data.config = SimpleNamespace(extra_dataloader_kwargs=dict(num_workers=1, pin_memory=False))
    data.scene = SimpleNamespace(train_dataset=SlowWorkerOnlyPairDataset(0))
    data.current_frame = 1
    started = time.monotonic()
    with patch.object(DATA, 'SELECTIVE_PAIR_WORKER_TIMEOUT_SECONDS', .05):
        try:
            data.selective_training_pairs(['a.png', 'b.png'], 'cpu', 0.0)
        except RuntimeError as error:
            assert 'timed out' in str(error).lower()
        else:
            raise AssertionError('A delayed paired-view worker did not time out')
    assert time.monotonic() - started < 3
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        new_workers = {worker.pid for worker in multiprocessing.active_children()} - before_workers
        if not new_workers:
            break
        time.sleep(.05)
    assert not new_workers


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
        shared_path = Path(directory) / 'motion_shared.npz'
        deform.export(shared_path, index_encoding='shared_table')
        assert stats['active_anchors'] < stats['candidate_anchors']
        with np.load(path) as packet:
            assert packet['representation'].item() == 'nested_gated_four_sparse_v2'
            assert 'gate_l2' not in packet.files
            for level in (2, 3, 4):
                rows = packet[f'blend_rows_l{level}']
                indices = packet[f'blend_indices_l{level}']
                weights = packet[f'blend_weights_l{level}']
                assert indices.ndim == 2 and indices.shape[0] == len(rows)
                assert indices.shape[1] > 0 and weights.shape == indices.shape
                assert np.all(rows[1:] > rows[:-1])
                assert np.all(indices[weights > 0] >= 0)
        decoded = G.GatedFourDeformation.decode_export_motion(path)
        with np.load(path) as v2, np.load(shared_path) as v3:
            assert v3['representation'].item() == 'nested_gated_four_sparse_v3_shared_index'
            for level in (2, 3, 4):
                table = v3[f'blend_index_table_l{level}']
                groups = v3[f'blend_index_group_l{level}']
                np.testing.assert_array_equal(table[groups], v2[f'blend_indices_l{level}'])
                np.testing.assert_array_equal(v3[f'blend_weights_l{level}'],
                                              v2[f'blend_weights_l{level}'])
                assert f'blend_indices_l{level}' not in v3.files
        shared_decoded = G.GatedFourDeformation.decode_export_motion(shared_path)
        np.testing.assert_array_equal(shared_decoded, decoded)
        expected_xyz, expected_rot = deform(xyz)
        torch.testing.assert_close(torch.from_numpy(decoded[:, :3]), expected_xyz,
                                   rtol=0, atol=1e-6)
        decoded_rot = torch.nn.functional.normalize(torch.from_numpy(decoded[:, 3:]), dim=-1)
        torch.testing.assert_close(decoded_rot, expected_rot, rtol=0, atol=1e-6)

        # Closed future levels must produce valid empty sparse rows and decode
        # exactly like forward at every partially opened gate depth.
        saved_gate_count = deform.active_gate_count
        saved_masks = {level: deform._gate_mask(level).clone() for level in (2, 3, 4)}
        try:
            for gate_count in (0, 1, 2):
                with torch.no_grad():
                    for level, mask in saved_masks.items():
                        deform._gate_mask(level).copy_(mask)
                    deform.active_gate_count = gate_count
                    for level in range(gate_count + 2, 5):
                        deform._gate_mask(level).zero_()
                partial_path = Path(directory) / f'motion_{gate_count}.npz'
                deform.export(partial_path)
                shared_partial_path = Path(directory) / f'motion_shared_{gate_count}.npz'
                deform.export(shared_partial_path, index_encoding='shared_table')
                with np.load(partial_path) as packet:
                    assert int(packet['active_gate_count']) == gate_count
                    for level in range(gate_count + 2, 5):
                        assert packet[f'blend_rows_l{level}'].size == 0
                    for level in range(2, gate_count + 2):
                        assert packet[f'blend_rows_l{level}'].size > 0
                partial = G.GatedFourDeformation.decode_export_motion(partial_path)
                shared_partial = G.GatedFourDeformation.decode_export_motion(
                    shared_partial_path)
                np.testing.assert_array_equal(shared_partial, partial)
                partial_xyz, partial_rot = deform(xyz)
                torch.testing.assert_close(torch.from_numpy(partial[:, :3]), partial_xyz,
                                           rtol=0, atol=1e-6)
                torch.testing.assert_close(
                    torch.nn.functional.normalize(torch.from_numpy(partial[:, 3:]), dim=-1),
                    partial_rot, rtol=0, atol=1e-6)
        finally:
            with torch.no_grad():
                deform.active_gate_count = saved_gate_count
                for level, mask in saved_masks.items():
                    deform._gate_mask(level).copy_(mask)

        # A remapped inactive child is encoded as -1 and contributes zero,
        # rather than accidentally indexing the last residual anchor.
        with np.load(path) as packet:
            sparse = {key: packet[key] for key in packet.files}
        positive_slots = ((sparse['blend_indices_l2'] >= 0) &
                          (sparse['blend_weights_l2'] > 0))
        row, slot = np.argwhere(positive_slots)[0]
        output_row = int(sparse['blend_rows_l2'][row])
        removed_index = int(sparse['blend_indices_l2'][row, slot])
        removed_value = (sparse['delta'][removed_index] *
                         sparse['blend_weights_l2'][row, slot])
        sparse['blend_indices_l2'][row, slot] = -1
        sentinel_path = Path(directory) / 'motion_sentinel.npz'
        np.savez_compressed(sentinel_path, **sparse)
        sentinel = G.GatedFourDeformation.decode_export_motion(sentinel_path)
        assert np.isfinite(sentinel).all()
        np.testing.assert_allclose(decoded[output_row] - sentinel[output_row], removed_value,
                                   rtol=0, atol=1e-7)

        # The supported ratio=2 configuration has two blend slots, not three.
        ratio2_config = D.DeformConfig(hierarchy_mode='nested', num_grid_levels=4,
                                       max_gs_per_grid=5, grid_level_ratio=2, momentum=None)
        ratio2_deform = G.GatedFourDeformation(ratio2_config)
        ratio2_deform.setup(xyz)
        with torch.no_grad():
            ratio2_deform.active_gate_count = 3
            for level in (2, 3, 4):
                ratio2_deform._gate_mask(level).fill_(True)
            ratio2_deform.delta[:, :3].normal_(0, .01)
            ratio2_deform.delta[:, 4:].normal_(0, .01)
        ratio2_path = Path(directory) / 'motion_ratio2.npz'
        ratio2_deform.export(ratio2_path)
        ratio2_shared_path = Path(directory) / 'motion_ratio2_shared.npz'
        ratio2_deform.export(ratio2_shared_path, index_encoding='shared_table')
        with np.load(ratio2_path) as packet:
            for level in (2, 3, 4):
                assert packet[f'blend_indices_l{level}'].shape[1] == 2
        ratio2_decoded = G.GatedFourDeformation.decode_export_motion(ratio2_path)
        ratio2_shared_decoded = G.GatedFourDeformation.decode_export_motion(
            ratio2_shared_path)
        np.testing.assert_array_equal(ratio2_shared_decoded, ratio2_decoded)
        ratio2_xyz, ratio2_rot = ratio2_deform(xyz)
        torch.testing.assert_close(torch.from_numpy(ratio2_decoded[:, :3]), ratio2_xyz,
                                   rtol=0, atol=1e-6)
        torch.testing.assert_close(
            torch.nn.functional.normalize(torch.from_numpy(ratio2_decoded[:, 3:]), dim=-1),
            ratio2_rot, rtol=0, atol=1e-6)


def test_sparse_decoder_masks_nonfinite_sentinel_placeholder():
    packet = dict(
        representation=np.array('nested_gated_four_sparse_v2'),
        delta=np.array([[np.nan, np.inf, -np.inf, 0., 0., 0., 0.],
                        [.25, -.5, .75, 1., 0., 0., 0.]], dtype=np.float32),
        base_index=np.array([1], dtype=np.int32),
        active_gate_count=np.array(1, dtype=np.int32),
        blend_rows_l2=np.array([0], dtype=np.int32),
        blend_indices_l2=np.array([[-1]], dtype=np.int32),
        blend_weights_l2=np.array([[1.]], dtype=np.float32),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'nonfinite_sentinel.npz'
        np.savez_compressed(path, **packet)
        decoded = G.GatedFourDeformation.decode_export_motion(path)
    np.testing.assert_array_equal(decoded, packet['delta'][packet['base_index']])
    assert np.isfinite(decoded).all()

    shared_packet = dict(
        representation=np.array('nested_gated_four_sparse_v3_shared_index'),
        delta=packet['delta'],
        base_index=packet['base_index'],
        active_gate_count=np.array(1, dtype=np.int32),
        blend_rows_l2=np.array([0], dtype=np.int32),
        blend_index_table_l2=np.array([[-1]], dtype=np.int32),
        blend_index_group_l2=np.array([0], dtype=np.int32),
        blend_weights_l2=np.array([[1.]], dtype=np.float32),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'nonfinite_shared_sentinel.npz'
        np.savez_compressed(path, **shared_packet)
        shared_decoded = G.GatedFourDeformation.decode_export_motion(path)
    np.testing.assert_array_equal(shared_decoded, packet['delta'][packet['base_index']])
    assert np.isfinite(shared_decoded).all()


def test_sparse_motion_export_is_connected_to_renderer():
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'module.py'
    tree = ast.parse(source_path.read_text())
    module_class = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == 'Module')
    method_names = {'load_sparse_motion_export', 'forward_sparse_motion_export',
                    'clear_sparse_motion_export', 'advance_sparse_motion_export'}
    methods = [node for node in module_class.body
               if isinstance(node, ast.FunctionDef) and node.name in method_names]
    scope = dict(torch=torch, TensorSpace=object,
                 GatedFourDeformation=G.GatedFourDeformation)

    class PlaybackGaussian:
        def __init__(self, sh_degree=1):
            self._xyz = torch.empty(0, 3)
            self._rotation = torch.empty(0, 4)
            self.delta_xyz = None
            self.delta_rot = None
            self._opacity = torch.empty(0)

        @property
        def get_xyz_ori(self):
            return self._xyz

        @property
        def get_xyz(self):
            return self._xyz if self.delta_xyz is None else self._xyz + self.delta_xyz

        @property
        def get_rotation(self):
            if self.delta_rot is None:
                return self._rotation
            return GAUSS.quaternion_multiply(self._rotation, self.delta_rot)

        @property
        def get_opacity(self):
            return self._opacity

        def load_ply(self, path):
            assert path == 'incremental.ply'
            self._xyz = torch.tensor([[2., 0., 0.]])
            self._rotation = torch.tensor([[1., 0., 0., 0.]])
            self._opacity = torch.tensor([.9])
            self._features_dc = torch.zeros(1, 1, 3)
            self._features_rest = torch.zeros(1, 3, 3)
            self._scaling = torch.zeros(1, 3)
            self.max_radii2D = torch.zeros(1)
            self.xyz_gradient_accum = torch.zeros(1, 1)
            self.denom = torch.ones(1, 1, dtype=torch.int64)

        def apply_deformation(self, delta_xyz, delta_rotation):
            self._xyz = self.get_xyz + delta_xyz
            self._rotation = GAUSS.quaternion_multiply(self.get_rotation, delta_rotation)

        def extend(self, model):
            self._xyz = torch.cat([self._xyz, model._xyz])
            self._rotation = torch.cat([self._rotation, model._rotation])
            self._opacity = torch.cat([self._opacity, model._opacity])

        def prune_points(self, indices):
            keep = torch.ones(len(self._xyz), dtype=torch.bool, device=self._xyz.device)
            keep[indices] = False
            self._xyz = self._xyz[keep]
            self._rotation = self._rotation[keep]
            self._opacity = self._opacity[keep]

    class PlaybackMerged:
        def __init__(self, models):
            self.models = models

        @property
        def get_xyz(self):
            return torch.cat([model.get_xyz for model in self.models])

        @property
        def get_rotation(self):
            return torch.cat([model.get_rotation for model in self.models])

    class PlaybackRenderer:
        def __init__(self, gaussians):
            self.gaussians = gaussians
            self.fail = False

        def __call__(self, viewpoint, background):
            assert viewpoint == 'camera'
            assert background.shape == (3,)
            if self.fail:
                raise RuntimeError('fixture renderer failure')
            return dict(render=self.gaussians.get_xyz.clone(),
                        rotation=self.gaussians.get_rotation.clone())

    scope.update(GaussianModel=PlaybackGaussian, MergedGaussianModel=PlaybackMerged)
    scope.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), 'exec'), scope)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'sparse.npz'
        delta = np.zeros((2, 7), dtype=np.float32)
        delta[0, 3] = 1.
        delta[1, 0] = .4
        delta[1, 3] = 1.
        np.savez_compressed(path,
            representation=np.array('nested_gated_four_sparse_v2'), delta=delta,
            base_index=np.array([0, 0], dtype=np.int32),
            active_gate_count=np.array(1, dtype=np.int32),
            blend_rows_l2=np.array([1], dtype=np.int32),
            blend_indices_l2=np.array([[1, -1]], dtype=np.int32),
            blend_weights_l2=np.array([[.25, .75]], dtype=np.float32))
        base = PlaybackGaussian()
        base._xyz = torch.zeros(2, 3, dtype=torch.float64)
        base._rotation = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]],
                                      dtype=torch.float64)
        base._opacity = torch.tensor([.2, .3], dtype=torch.float64)
        previous_delta_xyz = torch.full((2, 3), .03, dtype=torch.float64)
        previous_delta_rot = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]],
                                          dtype=torch.float64)
        base.delta_xyz = previous_delta_xyz
        base.delta_rot = previous_delta_rot
        renderer = PlaybackRenderer(base)
        playback = SimpleNamespace(
            deform=G.GatedFourDeformation(D.DeformConfig(
                hierarchy_mode='nested', refinement_mode='gradhet', num_grid_levels=4)),
            gaussians=base, renderer=renderer,
            config=SimpleNamespace(sh_degree=1, merge_to_base=True),
            background=torch.zeros(3), _sparse_motion_export=None)
        playback.clear_sparse_motion_export = lambda: scope[
            'clear_sparse_motion_export'](playback)
        loaded = scope['load_sparse_motion_export'](playback, path, 'incremental.ply')
        assert loaded == dict(motion_rows=2, incremental_rows=1)
        incremental = playback._sparse_motion_export[2]
        for name in ('_xyz', '_features_dc', '_features_rest', '_opacity',
                     '_scaling', '_rotation', 'max_radii2D', 'xyz_gradient_accum'):
            value = getattr(incremental, name)
            assert value.device == base.get_xyz_ori.device
            assert value.dtype == base.get_xyz_ori.dtype
        assert incremental.denom.dtype == torch.int64
        result = scope['forward_sparse_motion_export'](playback, 'camera')['render']
        torch.testing.assert_close(result, torch.tensor([[0., 0., 0.],
                                                          [.1, 0., 0.],
                                                          [2., 0., 0.]], dtype=torch.float64),
                                   rtol=0, atol=1e-7)
        expected_rotation = GAUSS.quaternion_multiply(
            base._rotation, torch.tensor([[1., 0., 0., 0.],
                                          [1., 0., 0., 0.]], dtype=torch.float64))
        torch.testing.assert_close(
            scope['forward_sparse_motion_export'](playback, 'camera')['rotation'][:2],
            expected_rotation, rtol=0, atol=1e-7)
        assert renderer.gaussians is base
        assert base.delta_xyz is previous_delta_xyz
        assert base.delta_rot is previous_delta_rot
        renderer.fail = True
        try:
            scope['forward_sparse_motion_export'](playback, 'camera')
        except RuntimeError as error:
            assert str(error) == 'fixture renderer failure'
        else:
            raise AssertionError('renderer failure did not propagate')
        assert renderer.gaussians is base
        assert base.delta_xyz is previous_delta_xyz
        assert base.delta_rot is previous_delta_rot
        renderer.fail = False
        base.delta_xyz = None
        base.delta_rot = None
        advanced = scope['advance_sparse_motion_export'](playback)
        assert advanced == dict(base_rows=2, merged_incremental_rows=1)
        torch.testing.assert_close(
            base.get_xyz, torch.tensor([[.1, 0., 0.], [2., 0., 0.]], dtype=torch.float64),
            rtol=0, atol=1e-7)
        assert renderer.gaussians is base
        assert playback._sparse_motion_export is None
        scope['clear_sparse_motion_export'](playback)
        assert playback._sparse_motion_export is None


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
                      ('max_new_gaussians', 2.0), ('max_retained_gaussians', 0),
                      ('max_retained_gaussians', False), ('max_retained_gaussians', 2.0)):
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


def test_gated_fixed_coarse_cut_keeps_model_capacity_but_ignores_gate_logits():
    raw, fixed = gated_fixture(gate_policy='fixed_coarse')
    _, learned = gated_fixture(gate_policy='learned')
    fixed_capacity = sum(parameter.numel() for parameter in fixed.parameters())
    learned_capacity = sum(parameter.numel() for parameter in learned.parameters())
    assert fixed_capacity == learned_capacity
    initial = fixed.gate_state_arrays()
    assert bool((initial['states'][0] == GR.SHARE).all())
    assert not initial['visited'][1].any() and not initial['visited'][2].any()

    set_gates(fixed, {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SPLIT})
    fixed.gate_logits[0].grad = None
    with torch.no_grad():
        fixed.region_translation[0][:, 0] = 0.2
        fixed.region_translation[1][:, 0] = 50.
        fixed.gs_translation[:, 0] = 50.
    xyz, _ = fixed(fixed.config.motion_steps, training=True)
    assert bool((fixed.gate_state_arrays()['states'][0] == GR.SHARE).all())
    assert not fixed.gate_state_arrays()['visited'][1].any()
    assert fixed.frozen_states is None
    assert fixed.storage_parts(fixed.config.motion_steps, hard=True)['gate_bytes'] == 0
    xyz.sum().backward()
    assert fixed.gate_logits[0].grad is None
    assert bool((xyz[:, 0] < 50.).all())


def test_camera_name_split_is_ordered_unique_and_disjoint():
    dataset = SimpleNamespace(items=[
        SimpleNamespace(path=Path('cam02.png')),
        SimpleNamespace(path=Path('cam01.png')),
        SimpleNamespace(path=Path('cam03.png')),
    ])
    assert DATA.DataModule._camera_indices(dataset, ['cam03.png', 'cam02.png']) == [2, 0]
    try:
        DATA.DataModule._camera_indices(dataset, ['cam99.png'])
    except ValueError as error:
        assert 'missing views' in str(error)
    else:
        raise AssertionError('Unknown camera names must be rejected')
    data = DATA.DataModule.__new__(DATA.DataModule)
    data.config = SimpleNamespace(training_camera_names=['cam01.png', 'cam02.png'],
                                  validation_camera_names=['cam02.png'])
    try:
        data._configured_camera_split()
    except ValueError as error:
        assert 'overlap' in str(error)
    else:
        raise AssertionError('Overlapping train/validation camera names must be rejected')


def test_camera_name_split_builds_disjoint_train_and_validation_loaders():
    data = DATA.DataModule.__new__(DATA.DataModule)
    data.config = SimpleNamespace(
        training_camera_names=['a.png'], validation_camera_names=['b.png'],
        extra_dataloader_kwargs=dict(num_workers=0, pin_memory=False),
        training_view_seed=0, eval_train=True, eval_test=False,
        first_frame_eval_train=None)
    data.scene = SimpleNamespace(train_dataset=WorkerOnlyPairDataset(0), test_dataset=None)
    data.current_frame = 0
    train_loader = data.train_dataloader()
    validation_loaders = data.val_dataloader()
    assert train_loader.dataset.indices == [1]
    assert len(validation_loaders) == 1
    assert validation_loaders[0].dataset.indices == [0]
    assert data.eval_names == ['validation']


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
    namespace.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
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
        new_quant_birth_control=lambda: {}, quantized_quality_snapshot=lambda *args: None)
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


def test_final_birth_budget_is_separate_and_preserves_materialized_row_order():
    cfg = residual_test_config(max_new_gaussians=None, max_retained_gaussians=2)
    reference = residual_raw_fixture(count=5)
    model = SimpleNamespace()
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(reference[name].clone()))
    with torch.no_grad():
        model._opacity[:, 0] = torch.tensor([-10., 2., 2., 1., 0.])
    view = Q.QuantizedNewGaussianView(model, cfg, 1., reference)
    assert cfg.max_new_gaussians is None
    assert int(view.keep.sum()) == 4  # Training is not limited to two candidates.
    source = view.source.clone()
    report = view.apply_retention_budget(cfg.max_retained_gaussians)
    assert report['eligible'] == 4 and report['pruned'] == 2
    assert view.keep.tolist() == [False, True, True, False, False]
    torch.testing.assert_close(view.keep_st, view.keep.float(), rtol=0, atol=0)
    materialized, retained_sources = Q.materialize_quantized_gaussians(reference, view)
    assert len(materialized['xyz']) == 7
    torch.testing.assert_close(retained_sources, source[1:3], rtol=0, atol=0)
    for name in Q.RAW_FIELDS:
        torch.testing.assert_close(materialized[name][:5], reference[name], rtol=0, atol=0)
    # Equal scores prefer the earlier source row; a larger budget does not create points.
    tied = Q.QuantizedNewGaussianView(model, cfg, 1., reference)
    tied.apply_retention_budget(1)
    assert tied.keep.tolist() == [False, True, False, False, False]
    for limit in [None, 200]:
        unchanged = Q.QuantizedNewGaussianView(model, cfg, 1., reference)
        assert unchanged.apply_retention_budget(limit)['pruned'] == 0
        assert int(unchanged.keep.sum()) == 4
    with torch.no_grad():
        model._opacity.fill_(-10.)
    empty = Q.QuantizedNewGaussianView(model, cfg, 1., reference)
    assert empty.apply_retention_budget(200)['eligible'] == 0
    raw, sources = Q.materialize_quantized_gaussians(reference, empty)
    assert len(raw['xyz']) == 5 and sources.numel() == 0


def test_finalization_budget_updates_storage_tree_and_next_frame_count():
    raw, motion = gated_fixture(count=5, enable_incremental=True, incremental_steps=1,
                                max_new_gaussians=None, max_retained_gaussians=2)
    cfg = motion.config
    base, candidates = SimpleNamespace(), SimpleNamespace()
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(base, attr, raw[name].clone())
        setattr(candidates, attr, torch.nn.Parameter(raw[name].clone()))
    with torch.no_grad():
        candidates._opacity[:, 0] = torch.tensor([-10., 2., 2., 1., 0.])
    candidates.get_xyz = candidates._xyz
    view = Q.QuantizedNewGaussianView(candidates, cfg, 1., raw)
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'module.py'
    tree = ast.parse(source_path.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
               and node.name in {'finalize_quantized_frame', 'quantized_storage', 'new_quant_birth_control'}]
    scope = dict(torch=torch, time=time, json=json, np=np, RAW_ATTRIBUTES=Q.RAW_ATTRIBUTES,
                 materialize_quantized_gaussians=Q.materialize_quantized_gaussians,
                 model_raw=Q.model_raw, reference_digest=Q.reference_digest,
                 GaussianModel=lambda degree: SimpleNamespace())
    scope.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), 'exec'), scope)
    with tempfile.TemporaryDirectory() as directory:
        fake = SimpleNamespace(
            config=SimpleNamespace(quantized_residual=cfg, sh_degree=1,
                                   motion_mode='quantized_residual'),
            current_frame=1, global_step=cfg.total_steps, output_dir=Path(directory),
            residual_motion=motion, gaussians=base, gaussians_incr=candidates,
            quant_deleted_indices=torch.empty(0, dtype=torch.int32),
            quant_point_birth_frame=torch.zeros(5, dtype=torch.long), quant_first_frame_gs_count=5,
            trainer=SimpleNamespace(synchronize=lambda: None, record_io=lambda *args: None),
            make_quantized_new_view=lambda: view,
            quantized_quality_snapshot=lambda *args: None,
            quant_stats_window_start=lambda: cfg.densification_stats_start)
        fake.install_quantized_raw = lambda value: setattr(fake, 'installed_raw', value)
        fake.quantized_storage = lambda *args, **kwargs: scope['quantized_storage'](fake, *args, **kwargs)
        fake.quant_birth_control = scope['new_quant_birth_control'](fake)
        scope['finalize_quantized_frame'](fake)
        report = json.loads((Path(directory) / 'frame_summary/quantized_frame_001.json').read_text())
        birth = report['birth_control']
        assert (birth['final_candidate_count'], birth['opacity_pruned'],
                birth['retention_budget_pruned'], birth['final_retained']) == (5, 1, 2, 2)
        assert report['new_count'] == fake.quant_previous_new_count == 2
        bytes_per_birth = 4 + sum(value[0].numel() * value.element_size() for value in raw.values())
        assert report['storage_bytes_by_class']['new_bytes'] == 2 * bytes_per_birth
        assert report['expected_storage_bytes_by_class']['new_bytes'] == 2 * bytes_per_birth
        assert len(fake.installed_raw['xyz']) == 7
        assert len(fake.quant_point_birth_frame) == 7
        assert motion.materialized_layout['index'].shape[1] == 7
        torch.testing.assert_close(motion.materialized_layout['index'][:, 5:],
                                   motion.index[:, view.source[view.keep]], rtol=0, atol=0)


def test_candidate_prune_reason_counts_overlap_without_double_deletion():
    for screen_limit, expected_count in [(20., 4), (None, 2)]:
        fake = SimpleNamespace(
            _xyz=torch.zeros(5, 3), xyz_gradient_accum=torch.zeros(5, 1), denom=torch.ones(5, 1),
            get_opacity=torch.tensor([[.001], [.5], [.5], [.001], [.5]]),
            get_scaling=torch.tensor([[.01]*3, [.01]*3, [.3]*3, [.3]*3, [.01]*3]),
            max_radii2D=torch.tensor([30., 30., 0., 0., 0.]),
            adaptive_densify_and_split=lambda *args, **kwargs: dict(eligible=0, added=0, blocked=0))
        fake.prune_points = lambda mask: setattr(fake, '_xyz', fake._xyz[~mask])
        result = GAUSS.GaussianModel.adaptive_densify_and_prune(fake, .1, .01, 1., screen_limit)
        assert result['pruned'] == expected_count
        assert len(fake._xyz) == 5 - expected_count
        expected = dict(low_opacity=2, screen_size=2, world_size=2, multiple=2) if screen_limit else dict(
            low_opacity=2, screen_size=0, world_size=0, multiple=0)
        assert result['prune_reasons'] == expected


def test_zero_refinement_keeps_gates_trainable_through_last_motion_step():
    raw, motion = gated_fixture(gate_refine_steps=0)
    cfg = motion.config
    for step in [1, cfg.motion_steps - 1, cfg.motion_steps]:
        assert motion.stage(step) == 'gate_joint'
        motion(step, training=True)
        assert motion.frozen_states is None
        assert all(value.requires_grad for value in motion.gate_logits)
    restored = GR.GatedResidualMotion(cfg)
    restored.restore(copy.deepcopy(motion.capture()))
    assert restored.frozen_states is None
    assert all(value.requires_grad for value in restored.gate_logits)
    for left, right in zip(motion(cfg.motion_steps), restored(cfg.motion_steps)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


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


def test_motion_gradient_modes_preserve_forward_and_gate_gradient():
    for cut in ({0: GR.KEEP, 1: GR.SPLIT, 2: GR.SPLIT},
                {0: GR.SPLIT, 1: GR.SHARE, 2: GR.SPLIT},
                {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SPLIT}):
        results = []
        for mode in ('all_split', 'hard_path', 'keep_probe'):
            raw, motion = gated_fixture(motion_gradient_mode=mode)
            set_gates(motion, cut)
            with torch.no_grad():
                for i in range(3):
                    motion.region_translation[i][:, i] = .2
                    motion.region_rotation[i][:, (i + 1) % 3] = .1
                motion.gs_translation[:, 0] = .3
                motion.gs_rotation[:, 1] = .2
            xyz, rotation = motion(2, training=True)
            loss = xyz.sum() + (rotation * torch.tensor([.1, .2, .3, .4])).sum()
            loss.backward()
            grads = [p.grad.clone() for p in [*motion.region_translation,
                     *motion.region_rotation, motion.gs_translation, motion.gs_rotation]]
            results.append((xyz, rotation, [p.grad.clone() for p in motion.gate_logits], grads))
        for result in results[1:]:
            for index in (0, 1):
                torch.testing.assert_close(results[0][index], result[index], rtol=0, atol=0)
            for a, b in zip(results[0][2], result[2]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        if cut[0] == GR.KEEP:
            assert all(g.count_nonzero() == 0 for g in results[1][3])
            assert any(g.count_nonzero() > 0 for g in results[0][3])
            assert results[2][3][0].count_nonzero() > 0
            for index in (1, 2, 4, 5, 6, 7):
                assert results[2][3][index].count_nonzero() == 0
        elif cut[1] == GR.SHARE:
            # The third-level and local motion are closed, with no soft scaling.
            for index in (2, 5, 6, 7):
                assert results[1][3][index].count_nonzero() == 0
                assert results[0][3][index].count_nonzero() > 0
        else:
            for result in results[1:]:
                for a, b in zip(results[0][3], result[3]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_keep_probe_learns_closed_candidate_without_changing_pose_or_gate_prior():
    raw, motion = gated_fixture(motion_gradient_mode='keep_probe', gate_refine_steps=0,
                                motion_xyz_step_extent=.001)
    set_gates(motion, {0: GR.KEEP, 1: GR.SHARE, 2: GR.SHARE})
    logits = [value.detach().clone() for value in motion.gate_logits]
    xyz, rotation = motion(1, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)
    (xyz - (raw['xyz'] + .1)).square().mean().backward()
    assert motion.region_translation[0].grad.abs().sum() > 0
    for parameter in list(motion.region_translation)[1:] + [motion.gs_translation]:
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad), rtol=0, atol=0)
    motion.optimizer_step()
    for before, after in zip(logits, motion.gate_logits):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    xyz, _ = motion(2, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    (xyz - (raw['xyz'] + .1)).square().mean().backward()
    assert motion.gate_logits[0].grad.count_nonzero() > 0
    restored = GR.GatedResidualMotion(motion.config)
    restored.restore(copy.deepcopy(motion.capture()))
    for before, after in zip(motion(2), restored(2)):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_gs_activity_support_ignores_identity_and_quaternion_hemisphere():
    raw, motion = gated_fixture()
    regions = [torch.zeros(count, 6, requires_grad=True) for count in motion.counts]
    local = torch.zeros(len(raw['xyz']), 6, requires_grad=True)
    for rotation in (raw['rotation'], -raw['rotation']):
        support = GS_ACTIVITY.candidate_activity_support(
            raw['xyz'], rotation, motion.active_layout, regions, local, .0005, .0005)
        assert support.shape == (len(raw['xyz']), 4)
        assert support.dtype == torch.bool and not support.any()
        assert not support.requires_grad


def test_gs_activity_cost_counts_terminal_pose_after_ancestor_cancellation():
    raw, motion = gated_fixture()
    regions = [torch.zeros(count, 6) for count in motion.counts]
    regions[0][:, 0] = .01
    regions[1][:, 0] = -.01
    local = torch.zeros(len(raw['xyz']), 6)
    support = GS_ACTIVITY.candidate_activity_support(
        raw['xyz'], raw['rotation'], motion.active_layout, regions, local, .0005, .0005)
    assert support[:, 0].all() and not support[:, 1:].any()

    def probabilities(choices):
        return [F.one_hot(torch.full((count,), choices[level]), 3).float()
                for level, count in enumerate(motion.counts)]

    # A parent SPLIT does not itself count changed GS if the terminal cancels it.
    for choices, expected in (([GR.SHARE] * 3, 1.),
                              ([GR.SPLIT, GR.SHARE, GR.SHARE], 0.),
                              ([GR.SPLIT, GR.KEEP, GR.SPLIT], 0.)):
        value = GS_ACTIVITY.expected_activity_fraction(probabilities(choices), motion.index, support)
        torch.testing.assert_close(value, torch.tensor(expected), rtol=0, atol=0)


def test_gs_activity_probability_expectation_and_gate_only_gradient():
    raw, motion = gated_fixture(gate_refine_steps=0)
    logits = [torch.tensor([.2, .5, .3]).log().expand(count, 3).clone().requires_grad_()
              for count in motion.counts]
    probabilities = [value.softmax(-1) for value in logits]
    support = torch.ones(len(raw['xyz']), 4, requires_grad=True)
    value = GS_ACTIVITY.expected_activity_fraction(probabilities, motion.index, support)
    expected = .5 + .3 * .5 + .3 ** 2 * .5 + .3 ** 3
    torch.testing.assert_close(value, torch.tensor(expected), rtol=1e-6, atol=1e-7)
    value.backward()
    assert all(parameter.grad is not None for parameter in logits)
    assert bool((logits[0].grad[:, GR.KEEP] < 0).all())
    assert support.grad is None
    assert all(parameter.grad is None for parameter in motion.parameters())


def test_gs_activity_cost_counts_affected_gs_instead_of_parameter_tuples():
    raw, motion = gated_fixture()
    n = len(raw['xyz'])
    split = [F.one_hot(torch.full((count,), GR.SPLIT), 3).float()
             for count in motion.counts]
    support = torch.zeros(n, 4, dtype=torch.bool)
    support[0, 3] = True
    value = GS_ACTIVITY.expected_activity_fraction(split, motion.index, support)
    torch.testing.assert_close(value, torch.tensor(1 / n), rtol=1e-6, atol=1e-7)
    share = [F.one_hot(torch.full((count,), GR.SHARE), 3).float()
             for count in motion.counts]
    support[:, 0] = True
    torch.testing.assert_close(GS_ACTIVITY.expected_activity_fraction(share, motion.index, support),
                               torch.tensor(1.), rtol=0, atol=0)


def test_gs_activity_support_matches_hard_rigid_endpoints_and_local_cancellation():
    raw, motion = gated_fixture()
    regions = [torch.zeros(count, 6) for count in motion.counts]
    regions[0][:, 0] = .01
    regions[1][:, 1] = .003
    regions[2][:, 5] = .01
    local = torch.zeros(len(raw['xyz']), 6)
    states = [torch.full((count,), GR.SPLIT) for count in motion.counts]
    prefix_xyz, prefix_rotation = GR._motion(raw['xyz'], raw['rotation'], motion.active_layout,
                                            regions, local, states)
    canceled = torch.arange(len(local)) % 2 == 0
    local[canceled, :3] = raw['xyz'][canceled] - prefix_xyz[canceled]
    inverse_motion = R.quaternion_multiply(
        raw['rotation'][canceled], Q.inverse_unit_quaternion(prefix_rotation[canceled]))
    local[canceled, 3:] = Q.shortest_rotation_vector(inverse_motion)
    support = GS_ACTIVITY.candidate_activity_support(
        raw['xyz'], raw['rotation'], motion.active_layout, regions, local, .0005, .0005)
    assert not support[canceled, 3].any() and support[~canceled, 3].all()
    for endpoint in range(4):
        choices = [torch.full((count,), GR.SPLIT) for count in motion.counts]
        if endpoint < 3:
            choices[endpoint].fill_(GR.SHARE)
        xyz, rotation = GR._motion(raw['xyz'], raw['rotation'], motion.active_layout,
                                   regions, local, choices)
        relative = R.quaternion_multiply(F.normalize(rotation, dim=-1),
                                        Q.inverse_unit_quaternion(F.normalize(raw['rotation'], dim=-1)))
        expected = ((xyz - raw['xyz']).norm(dim=-1) > .0005) | (
            Q.shortest_rotation_vector(relative).norm(dim=-1) > .0005)
        assert torch.equal(support[:, endpoint], expected)


def test_hard_path_motion_gradient_warmup_changes_only_candidate_jacobian():
    results = {}
    for warmup, step in ((0, 1), (2, 1), (0, 2), (2, 2), (0, 3), (2, 3)):
        raw, motion = gated_fixture(motion_gradient_mode='hard_path',
                                    motion_gradient_warmup_steps=warmup)
        set_gates(motion, {0: GR.KEEP, 1: GR.SPLIT, 2: GR.SPLIT})
        with torch.no_grad():
            for level in range(3):
                motion.region_translation[level][:, 0] = .2
            motion.gs_translation[:, 0] = .3
        xyz, rotation = motion(step, training=True)
        loss = xyz.sum() + rotation.sum()
        loss.backward()
        motion_grads = [p.grad.clone() for p in
                        [*motion.region_translation, *motion.region_rotation,
                         motion.gs_translation, motion.gs_rotation]]
        gate_grads = [parameter.grad.clone() for parameter in motion.gate_logits]
        results[warmup, step] = (xyz, rotation, gate_grads, motion_grads)

    # Both the first and final warmup steps use all_split candidate Jacobians,
    # while preserving the same hard pose and gate proxy at that exact step.
    for step in (1, 2):
        hard, warmed = results[0, step], results[2, step]
        for index in (0, 1):
            torch.testing.assert_close(hard[index], warmed[index], rtol=0, atol=0)
        for expected, actual in zip(hard[2], warmed[2]):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert all(grad.count_nonzero() == 0 for grad in hard[3])
        assert any(grad.count_nonzero() > 0 for grad in warmed[3])

    # After the schedule, both candidate motion and gate gradients match hard_path.
    hard, warmed = results[0, 3], results[2, 3]
    for index in (0, 1):
        torch.testing.assert_close(hard[index], warmed[index], rtol=0, atol=0)
    for expected, actual in zip(hard[2], warmed[2]):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for expected, actual in zip(hard[3], warmed[3]):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_soft_motion_ancestor_product_vs_hard_path_gradient():
    raw, motion = gated_fixture(motion_gradient_mode='hard_path')
    set_gates(motion, {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SPLIT})
    regions = [torch.zeros(count, 6) for count in motion.counts]
    local = torch.zeros(len(raw['xyz']), 6, requires_grad=True)
    probabilities = [torch.tensor([.2, .6, .2]).expand(count, 3) for count in motion.counts]
    states = motion.gate_state_arrays()['states']
    soft_xyz, _ = GR._motion(raw['xyz'], raw['rotation'], motion.active_layout,
                             regions, local, states, soft_probs=probabilities)
    soft_grad = torch.autograd.grad(soft_xyz.sum(), local)[0][:, :3]
    hard_xyz, _ = GR._motion(raw['xyz'], raw['rotation'], motion.active_layout,
                             regions, local, states)
    hard_grad = torch.autograd.grad(hard_xyz.sum(), local)[0][:, :3]
    torch.testing.assert_close(soft_grad, torch.full_like(soft_grad, .2 ** 3), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(hard_grad, torch.ones_like(hard_grad), rtol=0, atol=0)
    for value in ('soft', '', None):
        try:
            gated_residual_test_config(motion_gradient_mode=value).validate()
        except ValueError as error:
            assert 'motion_gradient_mode' in str(error)
        else:
            raise AssertionError('Unknown gradient mode accepted')
    for value in (-1, 4, True, 1.5):
        try:
            gated_residual_test_config(motion_gradient_warmup_steps=value).validate()
        except ValueError as error:
            assert 'motion_gradient_warmup_steps' in str(error)
        else:
            raise AssertionError(f'Invalid gradient warmup accepted: {value!r}')


def test_quality_snapshot_restores_rng_renderer_and_rigid_overrides():
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'module.py'
    method = next(node for node in ast.walk(ast.parse(source_path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == 'quantized_quality_snapshot')
    scope = dict(torch=torch, np=np, l1_loss=torch.nn.functional.l1_loss,
                 psnr=lambda image, gt: -10 * (image - gt).square().mean().log10(),
                 ssim=lambda image, gt: 1 - (image - gt).abs().mean(),
                 MergedGaussianModel=lambda models: tuple(models))
    scope.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), 'exec'), scope)
    def loader():
        torch.rand(5)  # Emulate DataLoader's global worker-seed consumption.
        return [SimpleNamespace(path=Path('cam01.png'), image=torch.zeros(3, 4, 4),
                                to=lambda device: camera)]
    camera = SimpleNamespace(path=Path('cam01.png'), image=torch.zeros(3, 4, 4))
    camera.to = lambda device: camera
    seen = []
    seen_overrides = []
    class SnapshotRenderer:
        gaussians = 'original'
        def __call__(self, viewpoint, background):
            assert not torch.is_grad_enabled()
            seen.append(self.gaussians)
            seen_overrides.append(base.rigid_xyz)
            return dict(render=torch.full_like(viewpoint.image, .1))
    override = torch.ones(1, 3)
    base = SimpleNamespace(rigid_xyz=override, rigid_rotation=None)
    subject = SimpleNamespace(config=SimpleNamespace(lambda_dssim=.2,
        quantized_diagnostics=SimpleNamespace(control_quality_enabled=True)),
        renderer=SnapshotRenderer(), gaussians=base, background=None, device='cpu',
        datamodule=SimpleNamespace(train_dataloader=lambda **kwargs: loader(),
                                   test_dataloader=loader),
        quant_control_diagnostics={}, save_quantized_control_diagnostics=lambda: None)
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    scope['quantized_quality_snapshot'](subject, 'before_retention', 'new_view')
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert subject.renderer.gaussians == 'original'
    assert base.rigid_xyz is override and base.rigid_rotation is None
    assert seen == [(base, 'new_view'), (base, 'new_view')]
    assert seen_overrides == [None, None]
    scope['quantized_quality_snapshot'](subject, 'prediction', preserve_motion=True)
    assert seen_overrides[-2:] == [override, override]
    assert base.rigid_xyz is override
    quality = subject.quant_control_diagnostics['before_retention']
    torch.testing.assert_close(torch.tensor(quality['train']['mean']['image_objective']),
                               torch.tensor(.1), rtol=1e-6, atol=1e-7)
    # First-frame-derived models may not have rigid overrides yet.
    del base.rigid_xyz, base.rigid_rotation
    scope['quantized_quality_snapshot'](subject, 'before_deletion')
    assert base.rigid_xyz is None and base.rigid_rotation is None


def test_paired_camera_rng_ignores_model_random_consumption():
    data = DATA.DataModule.__new__(DATA.DataModule)
    data.current_frame = 3
    data.config = SimpleNamespace(training_view_seed=0,
                                 extra_dataloader_kwargs=dict(num_workers=0, pin_memory=False))
    data.get_train_dataset = lambda scale: list(range(20))
    sequences = []
    for perturb in (1, 1000):
        torch.manual_seed(23)
        torch.rand(perturb)
        loader = data.train_dataloader()
        fetcher = DATA.PairedFetcher(loader, True, None, seed=3)
        sequence = []
        for _ in range(60):
            torch.rand(perturb)
            sequence.append(fetcher.next())
        sequences.append(sequence)
    assert sequences[0] == sequences[1]
    assert all(sorted(sequences[0][start:start + 20]) == list(range(20)) for start in (0, 20, 40))


def test_fixed_budget_random_preserves_counts_rng_gradients_and_checkpoint():
    raw, reference = gated_fixture(count=108)
    root_count = reference.counts[0]
    counts = [[root_count - 2, 1, 1], [1, 1, 1], [1, 2, 0]]
    before = torch.random.get_rng_state().clone()
    first = GR.fixed_budget_states(reference.active_layout, counts, seed=7)
    second = GR.fixed_budget_states(reference.active_layout, counts, seed=7)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    selected = GR._selection(reference.active_layout, first)
    for level in range(3):
        visited = selected[0][level]
        assert [int(((first[level] == action) & visited).sum())
                for action in range(3)] == counts[level]
    try:
        GR.fixed_budget_states(reference.active_layout, [[1, 0, 0]] * 3, seed=7)
    except ValueError as error:
        assert 'Infeasible' in str(error)
    else:
        raise AssertionError('Impossible reachable counts must fail')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'budget.json'
        path.write_text(json.dumps(dict(frames={'1': dict(level_counts=counts)})))
        cfg = gated_residual_test_config(gate_policy='fixed_budget_random',
                                        fixed_gate_budget_path=str(path), fixed_gate_seed=6)
        fixed = GR.GatedResidualMotion(cfg)
        fixed.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
        assert sum(p.numel() for p in fixed.parameters()) == sum(
            p.numel() for p in reference.parameters())
        with torch.no_grad():
            for parameter in fixed.region_translation:
                parameter.fill_(.1)
        xyz, _ = fixed(cfg.motion_steps, training=True)
        xyz.sum().backward()
        assert all(p.grad is None for p in fixed.gate_logits)
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in fixed.region_translation)
        keep = fixed.gate_state_arrays()['gs_terminal'] == GR.KEEP
        torch.testing.assert_close(xyz[keep], raw['xyz'][keep], rtol=0, atol=0)
        restored = GR.GatedResidualMotion(cfg)
        restored.restore(copy.deepcopy(fixed.capture()))
        for left, right in zip(fixed(cfg.motion_steps), restored(cfg.motion_steps)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_inherited_keep_zero_candidates_have_no_image_learning_signal():
    raw, motion = gated_fixture(motion_gradient_mode='hard_path', gate_refine_steps=0)
    set_gates(motion, {0: GR.KEEP, 1: GR.KEEP, 2: GR.KEEP})
    xyz, _ = motion(1, training=True)
    target = raw['xyz'] + .1
    (xyz - target).square().mean().backward()
    for parameter in motion.parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad),
                                       rtol=0, atol=1e-7)
    motion.committed = True
    motion.register_materialized_layout(raw['xyz'], torch.empty(0, dtype=torch.long))
    original_packet = copy.deepcopy(motion.capture())
    packet = copy.deepcopy(original_packet)
    packet['config']['reset_keep_gate_priors'] = True
    cfg = gated_residual_test_config(motion_gradient_mode='hard_path',
                                    gate_refine_steps=0, reset_keep_gate_priors=True)
    reopened = GR.GatedResidualMotion(cfg)
    reopened.restore(packet)
    reopened.setup(raw['xyz'], raw['rotation'], 1., previous_rows=torch.arange(len(raw['xyz'])),
                   previous_new_count=0, frame_index=2)
    assert reopened.tree_setup['reset_keep_gate_nodes'] == reopened.counts
    assert bool((reopened.gate_state_arrays()['states'][0] == GR.SHARE).all())
    assert reopened.statistics()['levels'][0]['refreshed_keep_final_counts'] == [
        0, reopened.counts[0], 0]
    xyz, _ = reopened(1, training=True)
    (xyz - target).square().mean().backward()
    assert reopened.region_translation[0].grad.abs().sum() > 0
    assert original_packet['config']['reset_keep_gate_priors'] is False


def test_trainer_restore_maps_cpu_checkpoint_to_training_device():
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'trainer.py'
    trainer = next(node for node in ast.parse(source_path.read_text(encoding='utf-8')).body
                   if isinstance(node, ast.ClassDef) and node.name == 'Trainer')
    method = next(node for node in trainer.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'setup')
    scope = dict(torch=torch)
    scope.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), 'exec'), scope)
    xyz = torch.tensor([[1., 2., 3.]])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'cpu.pth'
        torch.save(dict(global_step=200, current_frame=99, xyz=xyz), path)
        for target in ('cpu', 'meta'):
            events, packets = [], []
            module = SimpleNamespace(
                setup=lambda: events.append('module_setup'),
                on_load_checkpoint=lambda packet: packets.append(packet))
            module.to = lambda device: module
            subject = SimpleNamespace(device=torch.device(target), output_dir=Path(directory),
                init_logger=lambda: events.append('logger'),
                datamodule=SimpleNamespace(setup=lambda output: events.append('data_setup'),
                    on_load_checkpoint=lambda packet: events.append('data_restore')))
            scope['setup'](subject, module, path)
            assert packets[0]['xyz'].device.type == target
            assert packets[0]['xyz'].shape == xyz.shape and packets[0]['xyz'].dtype == xyz.dtype
            assert packets[0]['current_frame'] == 99 and subject.global_step == 200
            assert events == ['logger', 'data_setup', 'module_setup', 'data_restore']
            assert module.trainer is subject
            if target == 'cpu':
                torch.testing.assert_close(packets[0]['xyz'], xyz, rtol=0, atol=0)


def test_actual_motion_statistics_measures_keep_and_quaternion_sign():
    xyz = torch.zeros(3, 3)
    rotation = torch.tensor([[1., 0., 0., 0.]] * 3)
    moved = xyz.clone()
    moved[1, 0] = .2
    rotated = rotation.clone()
    rotated[1] = R.rotation_vector_quaternion(torch.tensor([0., 0., .01]))
    rotated[2] *= -1
    stats = GR.actual_motion_statistics(xyz, rotation, moved, rotated, 2.,
                                       [0., .05, .15], [0., .005, .02],
                                       torch.tensor([GR.KEEP, GR.SHARE, GR.KEEP]))
    assert stats['keep_position_changed_count'] == 0
    assert stats['keep_rotation_changed_count'] == 0
    assert stats['any_pose_changed_count'] == 1
    assert [x['count'] for x in stats['translation_extent']['above_threshold']] == [1, 1, 0]
    assert [x['count'] for x in stats['rotation_radians']['above_threshold']] == [1, 1, 0]
    torch.testing.assert_close(torch.tensor(stats['rotation_radians']['quantiles']['max']),
                               torch.tensor(.01), rtol=1e-5, atol=1e-7)




def test_gs_activity_configuration_validation_and_legacy_zero_weight_restore():
    gated_residual_test_config(gs_activity_loss_weight=.001).validate()
    for bad in (-.001, float('nan'), float('inf'), True):
        try:
            gated_residual_test_config(gs_activity_loss_weight=bad).validate()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid GS activity weight accepted')
    raw, motion = gated_fixture()
    packet = copy.deepcopy(motion.capture())
    packet['config'].pop('gs_activity_loss_weight')
    restored = GR.GatedResidualMotion(motion.config)
    restored.restore(packet)
    for original, current in zip(motion(2), restored(2)):
        torch.testing.assert_close(original, current, rtol=0, atol=0)


def test_gated_gs_activity_fraction_preserves_motion_and_counts_cancellation():
    raw, motion = gated_fixture(gs_activity_loss_weight=.001,
                                motion_xyz_step_extent=.001, gate_refine_steps=0)
    set_gates(motion, {0: GR.SPLIT, 1: GR.SHARE, 2: GR.SHARE})
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .01
        motion.region_translation[1][:, 0] = -.01
    torch.testing.assert_close(motion.gs_activity_fraction(2, hard=True), torch.tensor(0.),
                               rtol=0, atol=0)
    assert float(motion.storage_parts(2, hard=True)['region_motion_bytes']) > 0
    set_gates(motion, {0: GR.SHARE, 1: GR.SHARE, 2: GR.SHARE})
    torch.testing.assert_close(motion.gs_activity_fraction(2, hard=True), torch.tensor(1.),
                               rtol=0, atol=0)
    before = [value.detach().clone() for value in motion(2)]
    value = motion.gs_activity_fraction(2)
    assert 0 <= float(value) <= 1
    value.backward()
    assert motion.gate_logits[0].grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in motion.region_translation)
    for previous, current in zip(before, motion(2)):
        torch.testing.assert_close(previous, current, rtol=0, atol=0)


def test_gs_activity_training_cost_skips_zero_and_caches_post_commit_value():
    import textwrap
    source = (project_root / 'pipeline/ReCon-GS/module.py').read_text(encoding='utf8')
    start = source.index('            loss_activity = loss_storage.new_zeros(())')
    end = source.index("            metrics.update({'storage/'", start)
    block = textwrap.dedent(source[start:end])
    calls = []
    candidate = torch.tensor(.4, requires_grad=True)
    motion = SimpleNamespace(committed=False, last_metrics={'gs_activity/expected_fraction': .1},
        storage_scale=lambda step: 1., gs_activity_fraction=lambda step: calls.append(step) or candidate)
    def calculate(weight, step):
        initial = torch.tensor(.023)
        scope = dict(loss=initial, loss_storage=torch.tensor(.003), metrics={},
                     cfg=SimpleNamespace(gs_activity_loss_weight=weight),
                     self=SimpleNamespace(residual_motion=motion,
                                          config=SimpleNamespace(motion_mode='quantized_residual')),
                     current_step=step)
        scope.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
        exec(block, scope)
        return initial, scope
    initial, disabled = calculate(0., 133)
    assert disabled['loss'] is initial and calls == []
    torch.testing.assert_close(disabled['metrics']['loss/gs_activity'], torch.tensor(0.),
                               rtol=0, atol=0)
    _, joint = calculate(.001, 133)
    torch.testing.assert_close(joint['loss'], torch.tensor(.0234), rtol=1e-6, atol=1e-7)
    joint['loss'].backward()
    torch.testing.assert_close(candidate.grad, torch.tensor(.001), rtol=1e-6, atol=1e-7)
    motion.committed = True
    candidate = torch.tensor(.6)
    _, committed = calculate(.001, 134)
    torch.testing.assert_close(committed['loss'], torch.tensor(.0236), rtol=1e-6, atol=1e-7)
    candidate = torch.tensor(.9)
    _, cached = calculate(.001, 135)
    torch.testing.assert_close(cached['loss'], committed['loss'], rtol=0, atol=0)
    assert calls == [133, 134]


def test_gs_activity_cost_allows_mixed_subregions_with_exact_kept_pose():
    raw, motion = gated_fixture(gs_activity_loss_weight=.001,
                                motion_xyz_step_extent=.001, gate_refine_steps=0)
    set_gates(motion, {0: GR.SPLIT, 1: GR.KEEP, 2: GR.KEEP})
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .02
        motion.gate_logits[1][0] = torch.tensor([-20., 20., -20.])
    moved = motion.index[1] == 0
    assert moved.any() and (~moved).any()
    xyz, rotation = motion(2)
    torch.testing.assert_close(xyz[~moved], raw['xyz'][~moved], rtol=0, atol=0)
    torch.testing.assert_close(rotation[~moved], raw['rotation'][~moved], rtol=0, atol=0)
    expected_translation = torch.tensor([.02, 0., 0.]).expand(int(moved.sum()), 3)
    torch.testing.assert_close(xyz[moved] - raw['xyz'][moved], expected_translation,
                               rtol=1e-6, atol=1e-7)
    expected_cost = moved.float().mean()
    torch.testing.assert_close(motion.gs_activity_fraction(2, hard=True), expected_cost,
                               rtol=1e-6, atol=1e-7)
    assert 0 < float(expected_cost) < 1
    set_gates(motion, {0: GR.SHARE})
    torch.testing.assert_close(motion.gs_activity_fraction(2, hard=True), torch.tensor(1.),
                               rtol=0, atol=0)



def keep_local_probe_fixture(mode, choices=None, mixed=False):
    raw, motion = gated_fixture(motion_gradient_mode=mode, gate_refine_steps=0,
        motion_xyz_step_extent=.001, motion_rotation_step_rad=.001,
        gs_activity_loss_weight=.001)
    choices = {0: GR.SPLIT, 1: GR.SHARE, 2: GR.SPLIT} if choices is None else choices
    with torch.no_grad():
        for level, logits in enumerate(motion.gate_logits):
            logits.fill_(-.7)
            logits[:, choices.get(level, GR.KEEP)] = .9
            motion.region_translation[level][:] = torch.tensor([.02, -.003, .004]) * (level + 1)
            motion.region_rotation[level][:] = torch.tensor([.003, .002, -.005]) * (level + 1)
        motion.gs_translation[:] = torch.tensor([.002, -.001, .003])
        motion.gs_rotation[:] = torch.tensor([-.002, .004, .001])
        if mixed:
            motion.gate_logits[1][0] = torch.tensor([.9, -.7, -.7])
            motion.gate_logits[1][1] = torch.tensor([-.7, -.7, .9])
            fine = torch.where(motion.parents[1] == 1)[0][0]
            motion.gate_logits[2][fine] = torch.tensor([.9, -.7, -.7])
    return raw, motion


def test_keep_local_probe_preserves_hard_pose_gate_gradients_and_storage():
    results = []
    for mode in ('keep_probe', 'keep_local_probe'):
        raw, motion = keep_local_probe_fixture(mode, mixed=True)
        xyz, rotation = motion(2, training=True)
        weights = torch.linspace(.3, 1.5, len(xyz))[:, None]
        activity = motion.gs_activity_fraction(2)
        storage = sum(motion.storage_parts(2).values())
        loss = (xyz.square() * weights).sum() + (rotation * weights).sum()
        loss = loss + .001 * activity + .0001 * storage
        loss.backward()
        results.append((xyz.detach(), rotation.detach(),
            [p.grad.clone() for p in motion.gate_logits], activity.detach(), storage.detach()))
        assert any(p.grad.abs().sum() > 0 for p in motion.gate_logits)
        with torch.no_grad():
            hard_xyz, hard_rotation = motion(2)
        torch.testing.assert_close(xyz, hard_xyz, rtol=0, atol=0)
        torch.testing.assert_close(rotation, hard_rotation, rtol=0, atol=0)
    for index in (0, 1, 3, 4):
        torch.testing.assert_close(results[0][index], results[1][index], rtol=0, atol=0)
    for a, b in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_keep_local_probe_blocks_kept_descendant_ancestor_motion_only():
    outcomes = []
    for mode in ('keep_probe', 'keep_local_probe'):
        raw, motion = keep_local_probe_fixture(mode)
        with torch.no_grad():
            motion.gate_logits[1][0] = torch.tensor([.9, -.7, -.7])
        kept = motion.index[1] == 0
        xyz, rotation = motion(2, training=True)
        (xyz[kept].sum() + rotation[kept].sum()).backward()
        outcomes.append(motion)
        torch.testing.assert_close(xyz[kept], raw['xyz'][kept], rtol=0, atol=0)
        torch.testing.assert_close(rotation[kept], raw['rotation'][kept], rtol=0, atol=0)
        assert motion.region_translation[1].grad[0].abs().sum() > 0
        assert motion.region_rotation[1].grad[0].abs().sum() > 0
        for p in (*motion.region_translation[2:], *motion.region_rotation[2:],
                  motion.gs_translation, motion.gs_rotation):
            torch.testing.assert_close(p.grad, torch.zeros_like(p), rtol=0, atol=0)
    old, new = outcomes
    assert old.region_translation[0].grad.abs().sum() > 0
    assert old.region_rotation[0].grad.abs().sum() > 0
    for p in (new.region_translation[0], new.region_rotation[0]):
        torch.testing.assert_close(p.grad, torch.zeros_like(p), rtol=0, atol=0)
    for a, b in zip(old.gate_logits, new.gate_logits):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def test_keep_local_probe_preserves_active_chains_and_root_keep_probe():
    cases = ({0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SPLIT},
             {0: GR.SPLIT, 1: GR.SHARE}, {0: GR.SHARE}, {0: GR.KEEP})
    for choices in cases:
        outcomes = []
        for mode in ('keep_probe', 'keep_local_probe'):
            raw, motion = keep_local_probe_fixture(mode, choices=choices)
            xyz, rotation = motion(2, training=True)
            weights = torch.linspace(.3, 1.5, len(xyz))[:, None]
            ((xyz.square() * weights).sum() + (rotation * weights).sum()).backward()
            parameters = [*motion.region_translation, *motion.region_rotation,
                          motion.gs_translation, motion.gs_rotation]
            outcomes.append([p.grad.clone() for p in parameters])
            assert motion.region_translation[0].grad.abs().sum() > 0
        for a, b in zip(*outcomes):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_keep_local_probe_validation_and_checkpoint_roundtrip():
    gated_residual_test_config(motion_gradient_mode='keep_local_probe').validate()
    for options in (dict(enable_learned_gates=False, storage_loss_weight=0),
                    dict(gate_policy='fixed_coarse')):
        try:
            gated_residual_test_config(motion_gradient_mode='keep_local_probe', **options).validate()
        except ValueError as error:
            assert 'keep_local_probe' in str(error)
        else:
            raise AssertionError('Isolated KEEP probe accepted without learned gates')
    raw, motion = keep_local_probe_fixture('keep_local_probe', mixed=True)
    restored = GR.GatedResidualMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    assert restored.config.motion_gradient_mode == 'keep_local_probe'
    for a, b in zip(motion(2), restored(2)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_keep_candidate_loss_preserves_hard_path_and_gate_proxy():
    outcomes = []
    for mode in ('hard_path', 'keep_candidate_loss'):
        raw, motion = keep_local_probe_fixture(mode, mixed=True)
        xyz, rotation = motion(2, training=True)
        storage = sum(motion.storage_parts(2).values())
        activity = motion.gs_activity_fraction(2)
        ((xyz.square()).sum() + rotation.sum() + .001 * storage + .001 * activity).backward()
        outcomes.append((xyz.detach(), rotation.detach(), storage.detach(), activity.detach(),
            [p.grad.clone() for p in (*motion.region_translation, *motion.region_rotation,
                                     motion.gs_translation, motion.gs_rotation, *motion.gate_logits)]))
    for index in range(4):
        torch.testing.assert_close(outcomes[0][index], outcomes[1][index], rtol=0, atol=0)
    for a, b in zip(outcomes[0][4], outcomes[1][4]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_keep_candidate_own_pose_gradients_exclude_gates_ancestors_and_active_nodes():
    raw, motion = keep_local_probe_fixture('keep_candidate_loss')
    with torch.no_grad():
        motion.gate_logits[1][0] = torch.tensor([.9, -.7, -.7])
    kept = motion.index[1] == 0
    xyz, rotation, count = motion.keep_share_candidate_pose()
    assert count == 1
    assert not torch.equal(xyz[kept], raw['xyz'][kept])
    (xyz.sum() + rotation.sum()).backward()
    for parameters in (motion.region_translation, motion.region_rotation):
        torch.testing.assert_close(parameters[0].grad, torch.zeros_like(parameters[0]), rtol=0, atol=0)
        assert parameters[1].grad[0].abs().sum() > 0
        torch.testing.assert_close(parameters[1].grad[1:], torch.zeros_like(parameters[1][1:]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(parameters[2].grad, torch.zeros_like(parameters[2]), rtol=0, atol=0)
    assert all(p.grad is None for p in (*motion.gate_logits, motion.gs_translation, motion.gs_rotation))
    set_gates(motion, {0: GR.SHARE})
    assert motion.keep_share_candidate_pose() is None


def test_keep_candidate_own_loss_corrects_overshot_closed_translation_gradient():
    raw, motion = gated_fixture(motion_gradient_mode='keep_candidate_loss',
                                gate_refine_steps=0, motion_xyz_step_extent=.001)
    with torch.no_grad():
        for logits in motion.gate_logits:
            logits[:] = torch.tensor([.9, -.7, -.7])
        motion.region_translation[0][:, 0] = .3
    target = raw['xyz'][:, 0] + .1
    hard_xyz, _ = motion(2, training=True)
    hard_loss = (hard_xyz[:, 0] - target).square().mean()
    hard_loss.backward()
    torch.testing.assert_close(motion.region_translation[0].grad,
                               torch.zeros_like(motion.region_translation[0]), rtol=0, atol=0)
    motion.zero_grad()
    xyz, _, _ = motion.keep_share_candidate_pose()
    loss = (xyz[:, 0] - target).square().mean()
    loss.backward()
    torch.testing.assert_close(loss.detach(), torch.tensor(.04), rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(motion.region_translation[0].grad[:, 0].sum(), torch.tensor(.4),
                               rtol=1e-5, atol=1e-7)
    assert all(p.grad is None for p in motion.gate_logits)


def test_keep_candidate_loss_schedule_validation_and_checkpoint():
    cfg = gated_residual_test_config(motion_gradient_mode='keep_candidate_loss',
                                     motion_steps=133, gate_refine_steps=0,
                                     keep_candidate_fit_interval=10, keep_candidate_fit_weight=1.)
    cfg.validate()
    assert [step for step in range(1, 201) if cfg.keep_candidate_fit_due(step)] == list(range(1, 134, 10))
    assert len([step for step in range(1, 201) if cfg.keep_candidate_fit_due(step)]) == 14
    for options in (dict(keep_candidate_fit_interval=0), dict(keep_candidate_fit_interval=True),
                    dict(keep_candidate_fit_interval=1.2), dict(keep_candidate_fit_weight=0.),
                    dict(keep_candidate_fit_weight=float('nan')), dict(keep_candidate_fit_weight=True),
                    dict(motion_gradient_warmup_steps=1),
                    dict(enable_learned_gates=False, storage_loss_weight=0),
                    dict(gate_policy='fixed_coarse')):
        try:
            gated_residual_test_config(motion_gradient_mode='keep_candidate_loss', **options).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid candidate configuration accepted: {options}')
    raw, motion = keep_local_probe_fixture('keep_candidate_loss', mixed=True)
    restored = GR.GatedResidualMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    assert restored.config.motion_gradient_mode == 'keep_candidate_loss'
    for a, b in zip(motion(2), restored(2)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_candidate_render_audit_samples_before_image_outcomes():
    path = project_root / 'scripts' / 'audit_closed_candidate_render.py'
    parsed = ast.parse(path.read_text())
    function = next(node for node in parsed.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'sample_keep_nodes')
    namespace = {'torch': torch, 'np': np}
    namespace.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    counts = torch.arange(1, 19)
    index = torch.repeat_interleave(torch.arange(18), counts)
    states = [torch.zeros(18, dtype=torch.long) for _ in range(3)]
    visited = [torch.ones(18, dtype=torch.bool) for _ in range(3)]
    for choice, reach in zip(states, visited):
        choice[0] = 1
        reach[-3:] = False
    selection = dict(states=states, visited=visited)
    layout = dict(index=[index.clone() for _ in range(3)])
    sample = namespace['sample_keep_nodes']
    first, population = sample(selection, layout, 37, 8, 3)
    assert sample(selection, layout, 37, 8, 3) == (first, population)
    assert sample(selection, layout, 38, 8, 3)[0] != first
    assert len(first) == 24
    assert all(item['reachable_keep_nodes'] == 14 for item in population)
    for level in range(3):
        selected = [item for item in first if item['level'] == level]
        assert [sum(item['size_bin'] == group for item in selected) for group in range(3)] == [3, 3, 2]
        assert len({item['node'] for item in selected}) == 8
        assert all(1 <= item['node'] < 15 and item['members'] == item['node'] + 1
                   for item in selected)


def test_fine_split_candidate_fits_closed_local_without_active_or_gate_gradients():
    raw, motion = gated_fixture(motion_gradient_mode='keep_candidate_loss',
                                gate_refine_steps=0, fit_fine_split_candidates=True)
    set_gates(motion, {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SHARE})
    hard = motion(2)
    candidate = motion.keep_share_candidate_pose()
    assert candidate is not None and candidate[2] > 0
    # A zero local candidate initially has the same values but a fitting Jacobian.
    for a, b in zip(hard, candidate[:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    (candidate[0].sum() + candidate[1].sum()).backward()
    assert motion.gs_translation.grad.abs().sum() > 0
    assert motion.gs_rotation.grad.abs().sum() > 0
    for p in (*motion.region_translation, *motion.region_rotation):
        torch.testing.assert_close(p.grad, torch.zeros_like(p), rtol=0, atol=0)
    assert all(p.grad is None for p in motion.gate_logits)
    with torch.no_grad():
        motion.gs_translation[:, 0] = 2 * motion.motion_steps_vector[0]
    changed = motion.keep_share_candidate_pose()
    assert not torch.equal(changed[0], hard[0])
    # Shadow fitting never changes the selected hard SHARE pose.
    for a, b in zip(hard, motion(2)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fine_split_candidate_gradients_exclude_keep_and_open_local_rows():
    raw, motion = gated_fixture(count=72, motion_gradient_mode='keep_candidate_loss',
                                gate_refine_steps=0, fit_fine_split_candidates=True)
    set_gates(motion, {0: GR.SPLIT, 1: GR.SPLIT, 2: GR.SHARE})
    with torch.no_grad():
        motion.gate_logits[2][0] = torch.tensor([20., -20., -20.])
        motion.gate_logits[2][1] = torch.tensor([-20., -20., 20.])
    selection = motion.gate_state_arrays()
    eligible = (selection['states'][2] == GR.SHARE)[motion.index[2]]
    assert eligible.any() and (~eligible).any()
    pose = motion.keep_share_candidate_pose()
    (pose[0].sum() + pose[1].sum()).backward()
    assert motion.gs_translation.grad[eligible].abs().sum() > 0
    for p in (motion.gs_translation, motion.gs_rotation):
        torch.testing.assert_close(p.grad[~eligible], torch.zeros_like(p[~eligible]), rtol=0, atol=0)
    hard = motion(2)
    kept = selection['gs_terminal'] == GR.KEEP
    torch.testing.assert_close(hard[0][kept], raw['xyz'][kept], rtol=0, atol=0)
    assert all(p.grad is None for p in motion.gate_logits)


def test_fine_split_candidate_config_resume_and_graph_budget():
    cfg = gated_residual_test_config(motion_gradient_mode='keep_candidate_loss',
        motion_steps=133, gate_refine_steps=0, fit_fine_split_candidates=True)
    cfg.validate()
    assert sum(cfg.keep_candidate_fit_due(step) for step in range(1, 201)) == 14
    for options in (dict(fit_fine_split_candidates=1),
                    dict(fit_fine_split_candidates=True, motion_gradient_mode='hard_path')):
        try:
            gated_residual_test_config(**options).validate()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid fine candidate settings accepted')
    raw, motion = gated_fixture(motion_gradient_mode='keep_candidate_loss',
                                gate_refine_steps=0, fit_fine_split_candidates=True)
    restored = GR.GatedResidualMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    assert restored.config.fit_fine_split_candidates
    for a, b in zip(motion(2), restored(2)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def sparse_hierarchical_fixture(**overrides):
    # Synthetic fixture values; these are not approved training parameters.
    values = dict(residual_test_config().__dict__)
    values.update(motion_steps=6, shared_warmup_steps=0,
                  dense_fit_steps=2, gate_compress_steps=2, gate_refine_steps=2,
                  enable_learned_gates=True, gate_policy='learned', gate_lr=.05,
                  gate_temperature_start=1., gate_temperature_end=1.,
                  initial_gate_logit=2., gate_hard_threshold=.5,
                  storage_loss_weight=.001, motion_init_decay=0.)
    values.update(overrides)
    raw = residual_raw_fixture(18)
    motion = SH.SparseHierarchicalMotion(SH.SparseHierarchicalConfig(**values))
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    assert motion.tree_setup['initialization_decay'] == 0.
    return raw, motion


def test_sparse_hierarchical_binary_m_forward_keeps_closed_pose_and_trains_gate():
    raw, motion = sparse_hierarchical_fixture(
        m_hard_forward=True, m_initial_gate_logit=0., gate_hard_threshold=.51,
        motion_xyz_step_extent=1e-6)
    with torch.no_grad():
        motion.gs_translation[:, 0] = .02
    xyz, rotation = motion(3, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)
    (-xyz[:, 0].sum()).backward()
    assert torch.isfinite(motion.m_logits.grad).all()
    assert (motion.m_logits.grad < 0).all()
    assert all(value.grad is None for value in motion.motion_parameters())
    motion.optimizer_step()
    assert (motion.m_logits.detach() > 0).all()
    xyz, _ = motion(4, training=True)
    _, _, local = motion.gate_probabilities(4)
    torch.testing.assert_close(xyz[:, 0], raw['xyz'][:, 0] + .02 * local,
                               rtol=0, atol=1e-7)


def test_sparse_hierarchical_binary_m_hardening_rebase_preserves_full_selected_motion():
    raw, motion = sparse_hierarchical_fixture(
        m_hard_forward=True, rebase_on_hardening=True,
        motion_xyz_step_extent=1e-6, motion_rotation_step_rad=1e-5)
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .03
        motion.gs_translation[:, 1] = .01
        motion.gs_rotation[:, 2] = .02
        motion.m_logits[0] = -8.
    motion(3, training=True)
    soft_xyz, soft_q = motion(4)
    hard_xyz, hard_q = motion(5, training=True)
    torch.testing.assert_close(hard_xyz, soft_xyz, rtol=0, atol=8e-7)
    torch.testing.assert_close(hard_q, soft_q, rtol=0, atol=8e-6)
    torch.testing.assert_close(hard_xyz[0], raw['xyz'][0], rtol=0, atol=0)
    assert hard_xyz[1, 0] - raw['xyz'][1, 0] > .025
    restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    for actual, expected in zip(restored(5), (hard_xyz, hard_q)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_hierarchical_warm_motion_survives_rebuild_deletion_and_births():
    raw, motion = sparse_hierarchical_fixture(
        motion_init_decay=.6, tree_rebuild_interval=2,
        motion_xyz_step_extent=1e-6, motion_rotation_step_rad=1e-5)
    with torch.no_grad():
        motion.base_rotation.copy_(R.rotation_vector_quaternion(
            torch.arange(len(raw['xyz']))[:, None] * torch.tensor([[.002, .003, -.001]])))
        motion.region_translation[0][:, 0] = .015
        motion.region_rotation[1][:, 2] = .023
        motion.gs_translation[:, 1] = torch.arange(len(raw['xyz'])) * .001
        motion.m_logits[0] = -8.
    prior_xyz, prior_q = motion(5, training=True)
    motion.committed = True
    old_displacement, old_relative = motion.final_quantized_total_motion()
    born_xyz = prior_xyz[2:3].detach() + .002
    full_xyz = torch.cat((prior_xyz.detach(), born_xyz))
    full_q = torch.cat((prior_q.detach(), prior_q[2:3].detach()))
    motion.register_materialized_layout(full_xyz, [2])
    rows = torch.tensor([0] + list(range(2, len(full_xyz))))
    xyz, rotation = full_xyz[rows], full_q[rows]
    expected_translation = torch.cat((old_displacement[:1], old_displacement[2:],
                                      torch.zeros_like(born_xyz))) * .6
    expected_vector = torch.cat((Q.shortest_rotation_vector(old_relative[:1]),
                                Q.shortest_rotation_vector(old_relative[2:]),
                                torch.zeros_like(born_xyz))) * .6
    motion.setup(xyz, rotation, 1., previous_rows=rows, previous_new_count=1, frame_index=2)
    actual_xyz, actual_q = motion(1, training=False)
    torch.testing.assert_close(actual_xyz, xyz + expected_translation, rtol=0, atol=7e-7)
    expected_q = R.quaternion_multiply(R.rotation_vector_quaternion(expected_vector), rotation)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=8e-6)
    assert motion.tree_setup['initialization_decay'] == .6
    assert motion.tree_setup['action'] == 'rebuild'
    assert any(bool(value.abs().any()) for value in motion.region_translation)
    assert all(not bool(value.abs().any()) for value in motion.region_rotation)
    with torch.no_grad():
        motion.m_logits[0] = -8.
    hard_xyz, hard_q = motion(5, training=True)
    torch.testing.assert_close(hard_xyz[0], xyz[0], rtol=0, atol=0)
    torch.testing.assert_close(hard_q[0], rotation[0], rtol=0, atol=0)
    restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    for actual, expected in zip(restored(5), (hard_xyz, hard_q)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_hierarchical_warm_motion_rejects_uncommitted_and_wrong_identity():
    raw, motion = sparse_hierarchical_fixture(motion_init_decay=.6)
    try:
        motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=2)
    except ValueError as error:
        assert 'committed' in str(error)
    else:
        raise AssertionError('Uncommitted sparse motion history must be rejected')
    with torch.no_grad():
        motion.gs_translation[:, 0] = .02
    xyz, rotation = motion(5, training=True)
    motion.committed = True
    for rows, new_xyz in [(torch.arange(len(xyz)).flip(0), xyz.detach()),
                          (torch.arange(len(xyz)), xyz.detach() + .01)]:
        try:
            motion.setup(new_xyz, rotation.detach(), 1., previous_rows=rows,
                         previous_new_count=0, frame_index=2)
        except ValueError as error:
            assert 'identities' in str(error)
        else:
            raise AssertionError('Wrong sparse motion identities must be rejected')


def test_sparse_hierarchical_zero_decay_and_restore_skip_history_initialization():
    raw, motion = sparse_hierarchical_fixture(motion_init_decay=0.)
    with torch.no_grad():
        motion.gs_translation[:, 0] = .02
    xyz, rotation = motion(5, training=True)
    motion.setup(xyz.detach(), rotation.detach(), 1., frame_index=2)
    for value in motion.motion_parameters():
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
    assert motion.tree_setup['initialization_decay'] == 0.
    raw, warm = sparse_hierarchical_fixture(motion_init_decay=.6)
    with torch.no_grad():
        warm.gs_translation[:, 0] = .02
    warm.setup(raw['xyz'], raw['rotation'], 1., warm_start=False, frame_index=2)
    for value in warm.motion_parameters():
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)


def test_sparse_hierarchical_closed_m_restores_exact_previous_pose():
    raw, motion = sparse_hierarchical_fixture()
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .4
        motion.region_rotation[0][:, 2] = .2
        motion.gs_translation[:, 1] = .2
        motion.m_logits.fill_(-20.)
    xyz, rotation = motion(5, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)


def test_sparse_hierarchical_child_moves_with_closed_parent_residual():
    raw, motion = sparse_hierarchical_fixture()
    owner = motion.index[2, 0]
    with torch.no_grad():
        motion.m_logits.fill_(20.)
        for logits in motion.z_region_logits:
            logits.fill_(-20.)
        motion.z_local_logits.fill_(-20.)
        motion.z_region_logits[2][owner] = 20.
        motion.region_translation[2][owner, 0] = .4
    xyz, rotation = motion(5, training=True)
    expected = raw['xyz'].clone()
    expected[motion.index[2] == owner, 0] += .4
    torch.testing.assert_close(xyz, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)


def test_sparse_hierarchical_three_stages_freeze_the_other_parameter_group():
    _, motion = sparse_hierarchical_fixture()
    motion(1, training=True)
    assert all(p.requires_grad for p in motion.motion_parameters())
    assert all(not p.requires_grad for p in motion.gate_parameters())
    motion(3, training=True)
    assert all(not p.requires_grad for p in motion.motion_parameters())
    assert all(p.requires_grad for p in motion.gate_parameters())
    motion(5, training=True)
    assert all(p.requires_grad for p in motion.motion_parameters())
    assert all(not p.requires_grad for p in motion.gate_parameters())


def test_sparse_hierarchical_soft_render_gradients_only_train_gates():
    _, motion = sparse_hierarchical_fixture()
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .4
        motion.gs_translation[0, 1] = .2
    xyz, rotation = motion(3, training=True)
    (xyz.sum() + rotation.sum() + sum(motion.storage_parts(3).values()) * .001).backward()
    assert motion.m_logits.grad is not None and motion.m_logits.grad.abs().sum() > 0
    assert motion.z_region_logits[0].grad.abs().sum() > 0
    assert motion.z_local_logits.grad.abs().sum() > 0
    assert all(p.grad is None for p in motion.motion_parameters())
    previous = [p.detach().clone() for p in motion.motion_parameters()]
    motion.optimizer_step()
    for before, after in zip(previous, motion.motion_parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_sparse_hierarchical_eval_does_not_change_training_phase_or_optimizer():
    _, motion = sparse_hierarchical_fixture()
    motion(3, training=True)
    optimizer = motion.optimizer
    gates = [p.requires_grad for p in motion.gate_parameters()]
    with torch.no_grad():
        motion(5, training=False)
    assert motion.current_phase == 'gate_compress'
    assert motion.optimizer is optimizer and motion.hard_masks is None
    assert [p.requires_grad for p in motion.gate_parameters()] == gates


def test_sparse_hierarchical_refine_drops_closed_row_momentum():
    raw, motion = sparse_hierarchical_fixture()
    xyz, rotation = motion(1, training=True)
    (xyz.sum() + rotation.sum()).backward()
    motion.optimizer_step()
    with torch.no_grad():
        motion.m_logits.fill_(-20.)
    previous = [p.detach().clone() for p in motion.motion_parameters()]
    xyz, rotation = motion(5, training=True)
    (xyz.sum() + rotation.sum()).backward()
    motion.optimizer_step()
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    for before, after in zip(previous, motion.motion_parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_sparse_hierarchical_shared_m_reaches_unobserved_fine_members():
    _, motion = sparse_hierarchical_fixture(
        m_initial_gate_logit=0., gate_hard_threshold=.51,
        m_fine_sharing=True, m_fine_gate_lr=.05)
    motion(3, training=True)
    owners = motion.index[2]
    node = int(torch.bincount(owners).argmax())
    members = torch.where(owners == node)[0]
    assert len(members) >= 2
    observed, unobserved = members[:2]
    probability = motion.gate_probabilities(3)[0]
    (-probability[observed]).backward()
    assert motion.m_fine_logits.grad[node] < 0
    torch.testing.assert_close(motion.m_logits.grad[unobserved], torch.tensor(0.), rtol=0, atol=0)
    motion.optimizer_step()
    assert motion._hard_gate_masks()['m'][unobserved]
    torch.testing.assert_close(motion.m_logits[unobserved], torch.tensor(0.), rtol=0, atol=0)
    restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    torch.testing.assert_close(restored.gate_probabilities(3)[0],
                               motion.gate_probabilities(3)[0], rtol=0, atol=0)


def test_sparse_hierarchical_shared_m_legacy_checkpoint_defaults_to_no_sharing():
    _, motion = sparse_hierarchical_fixture()
    motion(3, training=True)
    packet = copy.deepcopy(motion.capture())
    packet['config'].pop('m_fine_sharing')
    packet['config'].pop('m_fine_gate_lr')
    packet['state'].pop('m_fine_logits')
    restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    restored.restore(packet)
    assert not restored.config.m_fine_sharing
    torch.testing.assert_close(restored.gate_probabilities(3)[0],
                               motion.gate_probabilities(3)[0], rtol=0, atol=0)
    for invalid in (None, 0., -1., float('nan'), float('inf'), True):
        try:
            sparse_hierarchical_fixture(m_fine_sharing=True, m_fine_gate_lr=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid shared m learning rate accepted')


def test_sparse_hierarchical_neutral_m_initialization_preserves_z_and_defaults_to_keep():
    raw, motion = sparse_hierarchical_fixture(m_initial_gate_logit=0., gate_hard_threshold=.51)
    torch.testing.assert_close(motion.m_logits, torch.zeros_like(motion.m_logits), rtol=0, atol=0)
    for value in [*motion.z_region_logits, motion.z_local_logits]:
        torch.testing.assert_close(value, torch.full_like(value, 2.), rtol=0, atol=0)
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .2
        motion.gs_translation[:, 1] = .1
    pose = motion(5, training=True)
    torch.testing.assert_close(pose[0], raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(pose[1], raw['rotation'], rtol=0, atol=0)
    assert not motion.hard_masks['m'].any()
    _, legacy = sparse_hierarchical_fixture()
    torch.testing.assert_close(legacy.m_logits, torch.full_like(legacy.m_logits, 2.), rtol=0, atol=0)
    for invalid in (float('nan'), float('inf'), True):
        try:
            sparse_hierarchical_fixture(m_initial_gate_logit=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid initial m logit accepted')


def test_sparse_hierarchical_hardening_rebase_preserves_selected_soft_poses_and_codec():
    raw, motion = sparse_hierarchical_fixture(
        rebase_on_hardening=True, motion_xyz_step_extent=1e-6,
        motion_rotation_step_rad=1e-5)
    with torch.no_grad():
        raw['rotation'] = R.rotation_vector_quaternion(
            torch.arange(len(raw['xyz']))[:, None] * torch.tensor([[.003, .001, -.002]]))
        motion.base_rotation.copy_(raw['rotation'])
        motion.region_translation[0][:, 0] = .03
        motion.region_rotation[0][:, 2] = .02
        motion.region_translation[2][:, 1] = .01
        motion.gs_translation[:, 2] = .01
        motion.gs_rotation[:, 0] = .02
        motion.m_logits[0] = -2.
        motion.z_local_logits[1] = -2.
        motion.z_region_logits[0][0] = -2.
    motion(3, training=True)
    target_xyz, target_rotation = motion(4, training=False)
    expected_masks = motion._hard_gate_masks()
    hard_xyz, hard_rotation = motion(5, training=True)
    active = expected_masks['m'] & expected_masks['z_local']
    torch.testing.assert_close(hard_xyz[active], target_xyz[active], rtol=0, atol=6e-7)
    torch.testing.assert_close(hard_rotation[active], target_rotation[active], rtol=0, atol=5e-6)
    torch.testing.assert_close(hard_xyz[~expected_masks['m']], raw['xyz'][~expected_masks['m']],
                               rtol=0, atol=0)
    torch.testing.assert_close(hard_rotation[~expected_masks['m']],
                               raw['rotation'][~expected_masks['m']], rtol=0, atol=0)
    for key in ('m', 'z_local'):
        assert torch.equal(motion.hard_masks[key], expected_masks[key])
    assert all(torch.equal(a, b) for a, b in zip(
        motion.hard_masks['z_regions'], expected_masks['z_regions']))
    restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    for a, b in zip(motion(5), restored(5)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'rebased.npz'
        motion.export_motion(path)
        decoded = SH.SparseHierarchicalMotion.decode_export_motion(
            path, raw['xyz'], raw['rotation'])
        for a, b in zip(motion(5), decoded):
            torch.testing.assert_close(a, torch.as_tensor(b), rtol=1e-6, atol=1e-6)


def test_sparse_hierarchical_refine_lr_scale_only_changes_refinement():
    _, motion = sparse_hierarchical_fixture(refine_lr_scale=.1)
    motion(1, training=True)
    fitting_lrs = torch.tensor([group['lr'] for group in motion.motion_optimizer.param_groups])
    motion(3, training=True)
    motion(5, training=True)
    refine_lrs = torch.tensor([group['lr'] for group in motion.motion_optimizer.param_groups])
    torch.testing.assert_close(refine_lrs, fitting_lrs * .1, rtol=1e-6, atol=0)
    packet = copy.deepcopy(motion.capture())
    restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    restored.restore(packet)
    restored_lrs = torch.tensor([group['lr'] for group in restored.motion_optimizer.param_groups])
    torch.testing.assert_close(restored_lrs, refine_lrs, rtol=0, atol=0)
    for invalid in (0., -1., float('nan'), float('inf'), True):
        try:
            sparse_hierarchical_fixture(refine_lr_scale=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid refinement learning-rate scale accepted')


def test_sparse_hierarchical_m_tree_roundtrip_and_expected_code_length():
    raw = residual_raw_fixture(4)
    index, counts, parents = Q.stable_layout(raw['xyz'], 2)
    layout = GR._layout(index, counts, parents, raw['xyz'])
    probability = torch.tensor([.2, .4, .6, .8], dtype=torch.float64, requires_grad=True)
    expected = probability.new_zeros(())
    for choices in itertools.product((False, True), repeat=4):
        mask = torch.tensor(choices)
        data, bits = SH.encode_m_tree(mask, layout)
        assert torch.equal(SH.decode_m_tree(data, bits, layout), mask)
        mass = torch.where(mask, probability, 1 - probability).prod()
        expected = expected + mass * bits
    proxy = SH._expected_m_tree_bits(probability, layout)
    torch.testing.assert_close(proxy, expected, rtol=1e-12, atol=1e-12)
    proxy.backward()
    assert probability.grad is not None and torch.isfinite(probability.grad).all()


def test_sparse_hierarchical_float32_storage_underflow_has_finite_correct_gradient():
    # Match the run's 24-GS leaves, 72-GS middle nodes and 216-GS roots.
    # With logit=2, a middle node's all-off probability underflows in float32.
    rows = torch.arange(432)
    layout = dict(index=torch.stack((rows // 216, rows // 72, rows // 24)),
                  counts=[2, 6, 18],
                  parents=[torch.arange(6) // 3, torch.arange(18) // 3])
    for initial_logit in (-7., -2., 2., 7.):
        logits = torch.full((len(rows),), initial_logit, requires_grad=True)
        actual = SH._expected_m_tree_bits(logits.sigmoid(), layout,
            (F.logsigmoid(logits), F.logsigmoid(-logits)))
        actual.backward()
        assert torch.isfinite(actual)
        assert torch.isfinite(logits.grad).all()

        # Independent closed-form reference for this regular tree, in float64.
        reference_logit = torch.tensor(initial_logit, dtype=torch.float64,
                                       requires_grad=True)
        p = reference_logit.sigmoid()
        uniform = lambda count: p.pow(count) + (1 - p).pow(count)
        leaf_bits = 2 + 24 * (1 - uniform(24))
        middle_bits = 2 + 3 * leaf_bits - 6 * uniform(72)
        reference = 2 * (2 + 3 * middle_bits - 6 * uniform(216))
        reference.backward()
        torch.testing.assert_close(actual.double(), reference, rtol=2e-5, atol=1e-4)
        expected_gradient = (reference_logit.grad / len(rows)).expand_as(logits)
        torch.testing.assert_close(logits.grad.double(), expected_gradient,
                                   rtol=2e-4, atol=1e-6)


def test_sparse_hierarchical_disabled_storage_has_no_backward_graph():
    source_path = project_root / 'pipeline/ReCon-GS/module.py'
    module_tree = ast.parse(source_path.read_text())
    block = next(node for node in ast.walk(module_tree)
                 if isinstance(node, ast.If) and any(
                     isinstance(statement, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == 'storage_weight'
                         for target in statement.targets) for statement in node.body))
    def assignment_index(name):
        return next(i for i, statement in enumerate(block.body)
                    if isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == name
                        for target in statement.targets))
    statements = block.body[assignment_index('cfg'):assignment_index('loss_activity')]
    code = compile(ast.Module(body=statements, type_ignores=[]), str(source_path), 'exec')
    for phase, weight, ramp in [('gate_compress', 0., 1.),
                                ('gate_compress', .001, 0.), ('dense_fit', .001, 1.)]:
        image_parameter = torch.tensor(2., requires_grad=True)
        storage_parameter = torch.tensor(-1., requires_grad=True)
        grad_enabled = []
        def count_storage(view):
            grad_enabled.append(torch.is_grad_enabled())
            # Deliberately invalid diagnostics must not enter the image loss.
            value = storage_parameter.sqrt()
            return value, {'motion_region_bytes': value}
        fake = SimpleNamespace(
            config=SimpleNamespace(motion_mode='sparse_hierarchical',
                                   quantized_residual=SimpleNamespace(storage_loss_weight=weight)),
            residual_motion=SimpleNamespace(stage=lambda step: phase,
                                            storage_scale=lambda step: ramp,
                                            reference_storage_bytes=lambda count: 24),
            quant_first_frame_gs_count=1, quantized_storage=count_storage)
        namespace = dict(torch=torch, self=fake, render_pkg={}, current_step=101,
                         loss=image_parameter.square(), loss_rgb=image_parameter.square())
        namespace.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
        exec(code, namespace)
        namespace['loss'].backward()
        assert grad_enabled == []  # Sparse storage diagnostics are disabled too.
        assert not namespace['storage'].requires_grad
        assert storage_parameter.grad is None
        torch.testing.assert_close(namespace['loss_storage'], torch.tensor(0.), rtol=0, atol=0)
        torch.testing.assert_close(image_parameter.grad, torch.tensor(4.), rtol=0, atol=0)


def test_sparse_hierarchical_storage_probability_keeps_tiny_region_gradients():
    _, motion = sparse_hierarchical_fixture(m_initial_gate_logit=-25.)
    motion(3, training=True)
    log_off = motion._m_storage_log_probabilities(3)[1]
    for level in range(3):
        probability = motion._region_open_probability(log_off, level)
        reference = -torch.expm1(-torch.bincount(motion.index[level],
            minlength=motion.counts[level]).double() * F.softplus(torch.tensor(-25., dtype=torch.float64)))
        torch.testing.assert_close(probability.double(), reference, rtol=1e-6, atol=0)
        assert (probability > 0).all()
    total = sum(motion.storage_parts(3).values())
    total.backward()
    assert torch.isfinite(total)
    assert torch.isfinite(motion.m_logits.grad).all()
    assert (motion.m_logits.grad > 0).all()
    assert all(value.grad is None for value in motion.motion_parameters())


def test_sparse_hierarchical_activity_uses_soft_training_and_fixed_render_masks():
    _, motion = sparse_hierarchical_fixture(gs_activity_loss_weight=.001,
        m_initial_gate_logit=0., gate_hard_threshold=.51, m_fine_sharing=True,
        m_fine_gate_lr=.02)
    torch.testing.assert_close(motion.gs_activity_fraction(1), torch.tensor(1.), rtol=0, atol=0)
    motion(3, training=True)
    activity = motion.gs_activity_fraction(3)
    torch.testing.assert_close(activity, torch.tensor(.5), rtol=0, atol=0)
    activity.backward()
    assert (motion.m_logits.grad > 0).all()
    assert (motion.m_fine_logits.grad > 0).all()
    assert all(value.grad is None for value in [*motion.z_region_logits, motion.z_local_logits])
    assert all(value.grad is None for value in motion.motion_parameters())
    motion(5, training=True)
    # Soft probabilities are .5, but the renderer's .51 threshold keeps all GSs.
    torch.testing.assert_close(motion.gs_activity_fraction(5), torch.tensor(0.), rtol=0, atol=0)
    torch.testing.assert_close(motion.gs_activity_fraction(7), torch.tensor(0.), rtol=0, atol=0)
    for bad in (-.001, float('nan'), float('inf'), True):
        try:
            sparse_hierarchical_fixture(gs_activity_loss_weight=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid sparse activity weight accepted')


def test_sparse_hierarchical_regularizers_reach_gates_and_newborns_in_correct_phases():
    import textwrap
    source = (project_root / 'pipeline/ReCon-GS/module.py').read_text()
    start = source.index('            cfg = self.config.quantized_residual\n            storage_active')
    end = source.index('            loss_candidate =', start)
    code = textwrap.dedent(source[start:end])
    for parameter_format, step in itertools.product(('legacy', 'sparse_parameters'), (1, 3, 5, 7)):
        _, motion = sparse_hierarchical_fixture(gs_activity_loss_weight=.002,
                                              motion_export_format=parameter_format)
        motion(min(step, 5), training=True)
        motion.committed = step == 7
        image_parameter = torch.tensor(2., requires_grad=True)
        birth_parameter = torch.tensor(.5, requires_grad=True)
        observed = []
        def storage(view):
            observed.append(torch.is_grad_enabled())
            parts = motion.storage_parts(step)
            parts['new_bytes'] = birth_parameter * 24 if step == 7 else birth_parameter.new_zeros(())
            return sum(parts.values()), parts
        fake = SimpleNamespace(config=SimpleNamespace(motion_mode='sparse_hierarchical',
                                                     quantized_residual=motion.config),
                               residual_motion=motion, quant_first_frame_gs_count=18,
                               quantized_storage=storage)
        namespace = dict(self=fake, torch=torch, render_pkg={}, current_step=step,
            loss=image_parameter.square(), loss_rgb=image_parameter.square(), metrics={})
        namespace.setdefault('SPARSE_MOTION_MODES', {'sparse_hierarchical', 'flat_cluster'})
        exec(code, namespace)
        namespace['loss'].backward()
        if step == 3:
            assert observed == [True]
            assert namespace['loss_storage'] > 0 and namespace['loss_activity'] > 0
            assert (motion.m_logits.grad > 0).all()
            assert all((value.grad > 0).all() for value in [*motion.z_region_logits, motion.z_local_logits])
            assert all(value.grad is None for value in motion.motion_parameters())
        elif step == 7:
            assert observed == [True] and namespace['loss_storage'] > 0
            torch.testing.assert_close(birth_parameter.grad,
                torch.tensor(motion.config.storage_loss_weight / 18), rtol=1e-6, atol=0)
            assert namespace['loss_activity'] == 0
            assert all(value.grad is None for value in motion.gate_parameters())
        else:
            assert observed == []
            assert namespace['loss_storage'] == 0 and namespace['loss_activity'] == 0
            assert all(value.grad is None for value in motion.gate_parameters())


def test_sparse_hierarchical_nonfinite_gate_gradient_aborts_before_optimizer_update():
    _, motion = sparse_hierarchical_fixture()
    motion(3, training=True)
    previous = motion.m_logits.detach().clone()
    motion.m_logits.grad = torch.ones_like(motion.m_logits)
    motion.m_logits.grad[0] = float('nan')
    try:
        motion.optimizer_step()
    except FloatingPointError as error:
        assert 'm_logits.grad' in str(error)
        assert 'frame=1, step=3' in str(error)
    else:
        raise AssertionError('Nonfinite gate gradients must stop before Adam updates')
    torch.testing.assert_close(motion.m_logits, previous, rtol=0, atol=0)
    assert not motion.gate_optimizer.state


def test_sparse_hierarchical_nonfinite_logits_cannot_be_hardened_as_closed_gates():
    for invalid in (float('nan'), float('inf'), -float('inf')):
        _, motion = sparse_hierarchical_fixture()
        with torch.no_grad():
            motion.m_logits[0] = invalid
        try:
            motion._hard_gate_masks()
        except FloatingPointError as error:
            assert 'm_logits' in str(error)
        else:
            raise AssertionError('Nonfinite logits must not become binary gate masks')


def test_sparse_hierarchical_storage_counts_plain_arrays_and_dense_training_separately():
    _, motion = sparse_hierarchical_fixture()
    with torch.no_grad():
        motion.m_logits.fill_(-20.)
        motion.m_logits[:3] = 20.
        motion.gs_translation[:3, 0] = .2
    motion(5, training=True)
    parts = motion.storage_parts(5, hard=True)
    assert set(parts) == {'motion_region_bytes', 'motion_local_bytes',
                          'm_mask_bytes', 'z_mask_bytes', 'topology_bytes', 'pivot_bytes'}
    regions, count = sum(motion.counts), len(motion.base_xyz)
    assert parts['m_mask_bytes'] == count
    assert parts['z_mask_bytes'] == count + regions
    assert parts['topology_bytes'] == 4 * (count + sum(motion.counts[1:]) + 3)
    assert parts['pivot_bytes'] == 12 * regions
    arrays = motion._raw_parameter_arrays()
    actual_parameter_bytes = sum(value.nbytes for key, value in arrays.items()
        if key not in ('representation', 'motion_export_format', 'base_digest'))
    torch.testing.assert_close(sum(parts.values()), torch.tensor(float(actual_parameter_bytes)),
                               rtol=0, atol=0)
    dense = sum(value.numel() * value.element_size() for value in motion.parameters())
    assert motion.dense_training_parameter_bytes() == dense
    assert not hasattr(motion, 'latent_decoders')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.npz'
        report = motion.export_motion(path)
        with np.load(path, allow_pickle=False) as packet:
            byte_count = packet['m_mask'].nbytes + packet['z_mask'].nbytes
            byte_count += packet['local_payload'].nbytes
            byte_count += sum(packet[f'region_payload_l{level}'].nbytes for level in range(3))
        assert report['payload_estimate_bytes'] == byte_count
        assert byte_count < actual_parameter_bytes  # Legacy packing is not the model cost.


def test_sparse_hierarchical_raw_storage_expectation_and_gradients_match_enumeration():
    _, motion = sparse_hierarchical_fixture(fine_region_gs=2)
    raw = residual_raw_fixture(4)
    motion = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    motion(3, training=True)
    with torch.no_grad():
        motion.m_logits.copy_(torch.tensor([-1., .2, 1., -.8]))
    m, z_regions, z_local = motion.gate_probabilities(3)
    proxy = sum(motion.storage_parts(3).values())
    regions, count = sum(motion.counts), len(m)
    fixed = count * 2 + regions + 4 * (count + sum(motion.counts[1:]) + 3) + regions * 12
    expected = proxy.new_tensor(0., dtype=torch.float64)
    for choices in itertools.product((False, True), repeat=count):
        mask = torch.tensor(choices)
        mass = torch.where(mask, m.double(), 1 - m.double()).prod()
        row_bytes = (mask.to(z_local) * z_local).double().sum() * 24
        for level in range(3):
            relevant = torch.bincount(motion.index[level][mask], minlength=motion.counts[level]) > 0
            row_bytes = row_bytes + (relevant.to(z_regions[level]) * z_regions[level]).double().sum() * 24
        expected = expected + mass * (row_bytes + fixed)
    torch.testing.assert_close(proxy.double(), expected, rtol=2e-6, atol=1e-5)
    parameters = motion.gate_parameters()
    actual_grad = torch.autograd.grad(proxy, parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, parameters)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=1e-5)


def test_sparse_hierarchical_raw_parameter_save_load_and_checkpoint_preserve_poses():
    import zipfile
    for moving in (False, True):
        raw, motion = sparse_hierarchical_fixture(motion_export_format='raw_parameters',
            rebase_on_hardening=True)
        with torch.no_grad():
            motion.m_logits.fill_(20. if moving else -20.)
            motion.m_logits[0] = -20.
            motion.gs_translation[:, 0] = .003
            motion.region_rotation[0][:, 2] = .02
        expected = motion(5, training=True)
        assert motion.hard_storage_cache is None
        recovered = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
        recovered.restore(copy.deepcopy(motion.capture()))
        for left, right in zip(expected, recovered(5)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parameters.npz'
            report = recovered.export_motion(path)
            with zipfile.ZipFile(path) as archive:
                assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
            with np.load(path, allow_pickle=False) as arrays:
                assert arrays['local_payload'].dtype == np.float32
                assert arrays['m_mask'].dtype == np.bool_
                assert not any(name.startswith('latent') for name in arrays.files)
                actual_bytes = SH.SparseHierarchicalMotion.raw_parameter_bytes(arrays)
            assert report['parameter_bytes'] == actual_bytes
            assert report['archive_bytes'] >= actual_bytes
            assert report['layout_dependency'] is None
            torch.testing.assert_close(sum(recovered.storage_parts(5, hard=True).values()),
                                       torch.tensor(float(actual_bytes)), rtol=0, atol=0)
            decoded = SH.SparseHierarchicalMotion.decode_export_motion(path, raw['xyz'], raw['rotation'])
            for left, right in zip(expected, decoded):
                torch.testing.assert_close(left, torch.from_numpy(right), rtol=1e-6, atol=1e-6)


def test_sparse_parameter_arrays_roundtrip_masks_paths_and_checkpoint():
    import zipfile
    for selection in ('closed', 'one', 'mixed', 'all'):
        raw, motion = sparse_hierarchical_fixture(motion_export_format='sparse_parameters',
                                                 rebase_on_hardening=False)
        with torch.no_grad():
            motion.m_logits.fill_(-20. if selection != 'all' else 20.)
            if selection == 'one':
                motion.m_logits[7] = 20.
            elif selection == 'mixed':
                motion.m_logits[::3] = 20.
            motion.z_local_logits[::2] = -20.
            motion.z_region_logits[0].fill_(-20.)  # A child remains able to move.
            motion.z_region_logits[1][::2] = -20.
            motion.gs_translation[:, 0] = .003
            motion.region_rotation[2][:, 2] = .02
        expected = motion(5, training=True)
        restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
        restored.restore(copy.deepcopy(motion.capture()))
        for left, right in zip(expected, restored(5)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert motion.hard_storage_cache is None
        arrays = restored._sparse_parameter_arrays()
        m = restored.hard_masks['m']
        np.testing.assert_array_equal(arrays['m_rows'], torch.where(m)[0].numpy())
        assert arrays['m_rows'].dtype == np.uint32 and arrays['fine_index'].dtype == np.uint16
        assert len(arrays['fine_index']) == int(m.sum())
        for level in range(3):
            relevant = torch.unique(restored.index[level][m], sorted=True)
            assert arrays[f'pivots_l{level}'].shape == (len(relevant), 3)
        byte_count = SH.SparseHierarchicalMotion.raw_parameter_bytes(arrays)
        torch.testing.assert_close(sum(restored.storage_parts(5, hard=True).values()),
                                   torch.tensor(float(byte_count)), rtol=0, atol=0)
        assert byte_count < SH.SparseHierarchicalMotion.raw_parameter_bytes(restored._raw_parameter_arrays())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.npz'
            report = restored.export_motion(path)
            assert report['parameter_bytes'] == byte_count
            assert report['motion_export_format'] == 'sparse_parameters'
            assert report['layout_dependency'] is None
            with zipfile.ZipFile(path) as archive:
                assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
            decoded = SH.SparseHierarchicalMotion.decode_export_motion(path, raw['xyz'], raw['rotation'])
            for left, right in zip(expected, decoded):
                torch.testing.assert_close(left, torch.from_numpy(right), rtol=1e-6, atol=1e-6)


def test_sparse_parameter_expected_cost_and_gradient_match_exact_enumeration():
    _, fixture = sparse_hierarchical_fixture(motion_export_format='sparse_parameters',
        fine_region_gs=2, m_fine_sharing=True, m_fine_gate_lr=.02)
    raw = residual_raw_fixture(4)
    motion = SH.SparseHierarchicalMotion(copy.deepcopy(fixture.config))
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    motion(3, training=True)
    with torch.no_grad():
        motion.m_logits.copy_(torch.tensor([-1., .2, 1., -.8]))
        motion.m_fine_logits.fill_(.1)
    m, z_regions, z_local = motion.gate_probabilities(3)
    proxy = sum(motion.storage_parts(3).values())
    expected = proxy.new_tensor(0., dtype=torch.float64)
    # Enumerate the m selections; conditionally marginalize independent z.
    parent_bytes = sum(motion.counts[level + 1] * SH._region_index_dtype(motion.counts[level]).itemsize
                       for level in range(2))
    for choices in itertools.product((False, True), repeat=len(m)):
        mask = torch.tensor(choices)
        mass = torch.where(mask, m.double(), 1 - m.double()).prod()
        rows = int(mask.sum())
        cost = proxy.new_tensor(float(parent_bytes + 16 + rows * 7), dtype=torch.float64)
        cost = cost + (mask.to(z_local) * z_local).double().sum() * 24
        for level in range(3):
            relevant = torch.bincount(motion.index[level][mask], minlength=motion.counts[level]) > 0
            cost = cost + int(relevant.sum()) * 13
            cost = cost + (relevant.to(z_regions[level]) * z_regions[level]).double().sum() * 24
        expected = expected + mass * cost
    torch.testing.assert_close(proxy.double(), expected, rtol=2e-6, atol=1e-5)
    actual_grad = torch.autograd.grad(proxy, motion.gate_parameters(), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, motion.gate_parameters())
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=1e-5)
        assert torch.isfinite(actual).all() and (actual > 0).all()
    assert all(value.grad is None for value in motion.motion_parameters())


def test_sparse_parameter_index_width_boundary_and_large_fine_owner_roundtrip():
    assert SH._region_index_dtype(65536) == np.dtype(np.uint16)
    assert SH._region_index_dtype(65537) == np.dtype(np.uint32)
    raw, motion = sparse_hierarchical_fixture(motion_export_format='sparse_parameters')
    raw = residual_raw_fixture(2)
    index = torch.tensor([[0, 0], [0, 0], [0, 65536]])
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1,
        layout_override=dict(index=index, counts=[1, 1, 65537],
            parents=[torch.zeros(1, dtype=torch.long), torch.zeros(65537, dtype=torch.long)]))
    with torch.no_grad():
        motion.m_logits.fill_(20.)
        motion.gs_translation[:, 1] = .003
    expected = motion(5, training=True)
    arrays = motion._sparse_parameter_arrays()
    assert arrays['fine_index'].dtype == np.uint32
    np.testing.assert_array_equal(arrays['fine_index'], np.array([0, 65536], dtype=np.uint32))
    torch.testing.assert_close(sum(motion.storage_parts(5, hard=True).values()),
        torch.tensor(float(SH.SparseHierarchicalMotion.raw_parameter_bytes(arrays))), rtol=0, atol=0)
    actual = SH.SparseHierarchicalMotion._load_sparse_parameter_pose(arrays, raw['xyz'], raw['rotation'])
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, torch.from_numpy(right), rtol=1e-6, atol=1e-6)


def test_sparse_parameter_reader_rejects_inconsistent_arrays():
    raw, motion = sparse_hierarchical_fixture(motion_export_format='sparse_parameters')
    motion(5, training=True)
    arrays = motion._sparse_parameter_arrays()
    invalid = []
    for field, value in (
        ('source_row_count', np.array(len(raw['xyz']) + 1, dtype=np.uint32)),
        ('m_rows', arrays['m_rows'][::-1].copy()),
        ('m_rows', np.zeros_like(arrays['m_rows'])),
        ('m_rows', np.full_like(arrays['m_rows'], len(raw['xyz']))),
        ('fine_index', arrays['fine_index'].astype(np.float32)),
        ('parents_l0', arrays['parents_l0'].astype(np.int32)),
        ('z_local_mask', arrays['z_local_mask'][:-1]),
        ('local_payload', arrays['local_payload'][:-1]),
        ('pivots_l0', np.full_like(arrays['pivots_l0'], np.nan)),
    ):
        packet = dict(arrays)
        packet[field] = value
        invalid.append(packet)
    for packet in invalid:
        try:
            SH.SparseHierarchicalMotion._load_sparse_parameter_pose(packet, raw['xyz'], raw['rotation'])
        except ValueError:
            pass
        else:
            raise AssertionError('Inconsistent sparse parameter packet accepted')


def test_sparse_hierarchical_export_and_refine_checkpoint_roundtrip():
    _, motion = sparse_hierarchical_fixture()
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .2
        motion.region_rotation[0][:, 2] = .1
        motion.gs_translation[0, 1] = .2
        motion.m_logits[0] = -20.
    xyz, rotation = motion(5, training=True)
    (xyz.sum() + rotation.sum()).backward()
    motion.optimizer_step()
    before = motion(5, training=False)
    restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    for left, right in zip(before, restored(5, training=False)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for old, new in zip(motion.motion_parameters(), restored.motion_parameters()):
        old_state = motion.motion_optimizer.state.get(old, {})
        new_state = restored.motion_optimizer.state.get(new, {})
        assert set(old_state) == set(new_state)
        for name in old_state:
            torch.testing.assert_close(old_state[name], new_state[name], rtol=0, atol=0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.npz'
        restored.export_motion(path)
        decoded = SH.SparseHierarchicalMotion.decode_export_motion(
            path, restored.base_xyz, restored.base_rotation)
        for expected, actual in zip(before, decoded):
            torch.testing.assert_close(expected, torch.as_tensor(actual), rtol=1e-6, atol=1e-6)


def test_sparse_layout_gap_boundaries_and_small_unsigned_ids():
    values = np.array([0, 1, 127, 128, 16383, 16384, 2097152, 268435456,
                       4294967295], dtype=np.uint64)
    decoded = SL_CODEC.decode_gaps(SL_CODEC.encode_gaps(values))
    torch.testing.assert_close(torch.from_numpy(decoded),
                               torch.from_numpy(values.astype(np.int64)), rtol=0, atol=0)
    assert SL_CODEC.compact_unsigned(np.array([0, 255])).dtype == np.uint8
    assert SL_CODEC.compact_unsigned(np.array([0, 65535])).dtype == np.uint16
    assert SL_CODEC.compact_unsigned(np.array([65536])).dtype == np.uint32
    for invalid in ([128], [128, 0], [255, 255, 255, 255, 31]):
        try:
            SL_CODEC.decode_gaps(np.asarray(invalid, dtype=np.uint8))
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed packed row gaps accepted')


def test_sparse_layout_grouped_members_restore_original_order_and_empty_regions():
    fine = np.array([3, 0, 3, 1, 0], dtype=np.int64)
    parents = [np.array([0, 0]), np.array([0, 0, 1, 1])]
    middle = parents[1][fine]
    layout = dict(index=np.stack((parents[0][middle], middle, fine)),
                  counts=np.array([1, 2, 4]), parents=parents,
                  pivots=[np.zeros((count, 3), dtype=np.float32) for count in [1, 2, 4]])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'layout.npz'
        arrays = SL_CODEC.encode_layout(layout, path)
        assert 'index' not in arrays
        assert arrays['fine_group_sizes'].tolist() == [2, 1, 0, 2]
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as packet:
            decoded = SL_CODEC.decode_layout(path, packet)
        torch.testing.assert_close(torch.from_numpy(decoded['index']),
                                   torch.from_numpy(layout['index']), rtol=0, atol=0)


def test_sparse_hierarchical_compact_and_legacy_packets_decode_identical_poses():
    raw, motion = sparse_hierarchical_fixture()
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .02
        motion.region_rotation[1][:, 2] = .01
        motion.gs_translation[0, 1] = .03
        motion.m_logits[1] = -20.
    expected = motion(5, training=True)
    with tempfile.TemporaryDirectory() as directory:
        compact = Path(directory) / 'compact.npz'
        legacy = Path(directory) / 'legacy.npz'
        report = motion.export_motion(compact)
        np.savez_compressed(legacy, **motion._make_export_arrays())
        assert report['layout_encoding'] == SL_CODEC.GROUPED
        assert report['layout_dependency'] is None
        assert report['archive_bytes'] == compact.stat().st_size
        with np.load(compact, allow_pickle=False) as packet:
            assert 'index' not in packet.files
            assert packet['counts'].dtype.itemsize < 4
        decoded = [SH.SparseHierarchicalMotion.decode_export_motion(
            path, raw['xyz'], raw['rotation']) for path in [compact, legacy]]
        for value, left, right in zip(expected, *decoded):
            torch.testing.assert_close(torch.as_tensor(left), torch.as_tensor(right),
                                       rtol=0, atol=0)
            torch.testing.assert_close(value, torch.as_tensor(left), rtol=1e-6, atol=1e-6)


def test_sparse_hierarchical_layout_reuse_handles_birth_deletion_and_changed_pivots():
    raw, motion = sparse_hierarchical_fixture(tree_rebuild_interval=5)
    motion(5, training=True)
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        path = Path(directory) / 'motion_001.npz'
        motion.export_motion(path)
        paths.append(path)
        for frame in [2, 3, 4, 5]:
            previous_xyz, previous_rotation = motion(5, training=False)
            sources = torch.tensor([1, 3])
            materialized_xyz = torch.cat((previous_xyz, previous_xyz[sources] + .005))
            materialized_rotation = torch.cat((previous_rotation, previous_rotation[sources]))
            motion.register_materialized_layout(materialized_xyz, sources)
            keep = torch.ones(len(materialized_xyz), dtype=torch.bool)
            keep[0] = False
            keep[-1] = False  # A newly appended GS may itself be deleted.
            rows = torch.where(keep)[0]
            base_xyz, base_rotation = materialized_xyz[rows], materialized_rotation[rows]
            motion.setup(base_xyz, base_rotation, 1., previous_rows=rows,
                         previous_new_count=2, frame_index=frame)
            with torch.no_grad():
                motion.gs_translation[:, 0] = .01
                motion.region_rotation[0][:, 2] = .005
            expected = motion(5, training=True)
            path = Path(directory) / f'motion_{frame:03d}.npz'
            report = motion.export_motion(path)
            with np.load(path, allow_pickle=False) as packet:
                restored_layout = SL_CODEC.decode_layout(path, packet)
                assert 'index' not in packet.files
                if frame < 5:
                    assert report['layout_encoding'] == SL_CODEC.REUSE
                    assert report['layout_dependency'] == paths[-1].name
                    assert 'counts' not in packet.files
                    assert 'fine_group_gaps' not in packet.files
                    assert len(packet['layout_appended_fine']) == 2
                    assert any(name.startswith('pivot_xor_l') for name in packet.files)
                else:
                    assert report['layout_encoding'] == SL_CODEC.GROUPED
                    assert report['layout_dependency'] is None
            torch.testing.assert_close(torch.from_numpy(restored_layout['index']),
                                       motion.index.cpu(), rtol=0, atol=0)
            for value, pivot in zip(motion.active_layout['pivots'], restored_layout['pivots']):
                torch.testing.assert_close(value.cpu(), torch.from_numpy(pivot), rtol=0, atol=0)
            decoded = SH.SparseHierarchicalMotion.decode_export_motion(path, base_xyz, base_rotation)
            for value, actual in zip(expected, decoded):
                torch.testing.assert_close(value, torch.as_tensor(actual), rtol=1e-6, atol=1e-6)
            paths.append(path)


def test_sparse_hierarchical_layout_dependencies_verified_and_new_directory_self_contained():
    raw, motion = sparse_hierarchical_fixture()
    motion(5, training=True)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / 'first.npz'
        motion.export_motion(first)
        rows = torch.arange(len(raw['xyz']))
        motion.setup(raw['xyz'], raw['rotation'], 1., previous_rows=rows, frame_index=2)
        motion(5, training=True)
        second = root / 'second.npz'
        assert motion.export_motion(second)['layout_encoding'] == SL_CODEC.REUSE
        relocated = root / 'new_run' / 'second.npz'
        assert motion.export_motion(relocated)['layout_encoding'] == SL_CODEC.GROUPED
        restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
        restored.restore(copy.deepcopy(motion.capture()))
        assert restored.export_motion(root / 'resumed.npz')['layout_encoding'] == SL_CODEC.GROUPED
        # A changed predecessor must fail instead of silently applying a wrong tree.
        first.write_bytes(first.read_bytes() + b'changed')
        try:
            SH.SparseHierarchicalMotion.decode_export_motion(second, raw['xyz'], raw['rotation'])
        except ValueError as error:
            assert 'hash mismatch' in str(error)
        else:
            raise AssertionError('Changed layout predecessor accepted')
        # A packet made self-contained in a new directory needs no old files.
        first.unlink()
        second.unlink()
        expected = motion(5, training=False)
        for value, actual in zip(expected, SH.SparseHierarchicalMotion.decode_export_motion(
                relocated, raw['xyz'], raw['rotation'])):
            torch.testing.assert_close(value, torch.as_tensor(actual), rtol=1e-6, atol=1e-6)


def motion_probe_fixture(builder='spatial', **overrides):
    parameters = dict(tree_builder=builder, tree_probe_steps=4, motion_steps=10,
                      tree_cluster_neighbors=8, tree_cluster_iterations=3,
                      tree_cluster_spatial_weight=.001,
                      tree_cluster_orientation_extent=.01,
                      tree_probe_min_observations=2, motion_xyz_step_extent=1e-6,
                      motion_rotation_step_rad=1e-5)
    parameters.update(overrides)
    return sparse_hierarchical_fixture(**parameters)


def test_motion_fit_recovers_rotation_despite_nonuniform_displacements():
    generator = np.random.default_rng(17)
    source = generator.normal(size=(40, 3))
    angle = .3
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    translation = np.array([.2, -.1, .04])
    target = source @ rotation.T + translation
    assert np.std(target - source, axis=0).max() > .1
    actual_r, actual_t, _, _ = MC.fit_rigid_groups(source, target,
        np.repeat(rotation[None], len(source), axis=0), np.zeros(len(source), np.int64),
        1, np.ones(len(source)), .01)
    torch.testing.assert_close(torch.from_numpy(actual_r[0]), torch.from_numpy(rotation),
                               rtol=0, atol=1e-12)
    torch.testing.assert_close(torch.from_numpy(actual_t[0]), torch.from_numpy(translation),
                               rtol=0, atol=1e-12)


def test_motion_clustering_improves_shared_fit_and_keeps_nested_budget():
    generator = np.random.default_rng(3)
    source = generator.normal(size=(120, 3)) * .04
    root = np.repeat(np.arange(2), 60)
    moving = np.r_[np.zeros(45), np.ones(15), np.ones(45), np.zeros(15)].astype(bool)
    source_q = np.tile([1., 0., 0., 0.], (120, 1))
    target_q = source_q.copy()
    target_q[moving] = [np.cos(.1), 0., 0., np.sin(.1)]
    relative = MC.relative_matrices(source_q, target_q)
    target = np.einsum('nij,nj->ni', relative, source)
    target[moving, 0] += .05
    seed = np.stack((root, root * 3 + np.arange(120) % 3,
                     (root * 3 + np.arange(120) % 3) * 3 + np.arange(120) // 3 % 3))
    parents = [np.repeat(np.arange(2), 3), np.repeat(np.arange(6), 3)]
    observations = np.full(120, 2)
    observations[0] = 1
    options = dict(extent=1., neighbors=8, iterations=3, spatial_weight=.001,
                   orientation_extent=.01, min_observations=2)
    index, diagnostics = MC.cluster_motion_layout(source, source_q, target, target_q,
        seed, [2, 6, 18], parents, observations, **options)
    repeated, repeated_diagnostics = MC.cluster_motion_layout(source, source_q, target,
        target_q, seed, [2, 6, 18], parents, observations, **options)
    np.testing.assert_array_equal(index, repeated)
    assert diagnostics == repeated_diagnostics
    assert diagnostics['levels'][0]['after']['objective'] < diagnostics['levels'][0]['before']['objective'] * .2
    assert index[0, 0] == seed[0, 0]
    for level in range(2):
        np.testing.assert_array_equal(parents[level][index[level + 1]], index[level])
    assert all((index[level] < count).all() for level, count in enumerate([2, 6, 18]))
    # No qualified observations must leave the spatial seed deterministic and finite.
    unknown, unknown_metrics = MC.cluster_motion_layout(source, source_q, target, target_q,
        seed, [2, 6, 18], parents, np.zeros(120), **options)
    np.testing.assert_array_equal(unknown[0], seed[0])
    assert unknown_metrics['qualified_rows'] == 0
    assert all(np.isfinite(level['after']['objective']) for level in unknown_metrics['levels'])


def test_tree_probe_uses_distinct_training_views_and_only_local_motion():
    raw, motion = motion_probe_fixture()
    xyz, rotation = motion(1, training=True)
    (xyz.sum() + rotation.sum()).backward()
    assert all(value.grad is None for value in (*motion.region_translation,
        *motion.region_rotation, *motion.gate_parameters()))
    assert motion.gs_translation.grad is not None
    visible = torch.ones(len(xyz), dtype=torch.bool)
    motion.observe_tree_probe(visible, 'cam01.png')
    motion.observe_tree_probe(visible, 'cam01.png')
    torch.testing.assert_close(motion.probe_observations, torch.ones(len(xyz)), rtol=0, atol=0)
    motion.observe_tree_probe(visible, 'cam02.png')
    torch.testing.assert_close(motion.probe_observations, torch.full((len(xyz),), 2.), rtol=0, atol=0)
    masks = motion._hard_masks_for_step(1)
    assert not any(value.any() for value in masks['z_regions'])
    assert masks['m'].all() and masks['z_local'].all()
    assert [motion.stage(step) for step in [1, 4, 5, 6, 7, 8, 9, 10, 11]] == [
        'tree_probe', 'tree_probe', 'dense_fit', 'dense_fit', 'gate_compress',
        'gate_compress', 'gate_refine', 'gate_refine', 'incremental']


def test_probe_projection_preserves_teacher_pose_and_resume_for_both_builders():
    for builder in ['spatial', 'rigid_cluster']:
        raw, motion = motion_probe_fixture(builder)
        motion(4, training=True)
        with torch.no_grad():
            motion.gs_translation[:, 0] = .01
            motion.gs_rotation[:, 2] = .02
        visible = torch.ones(len(raw['xyz']), dtype=torch.bool)
        motion.observe_tree_probe(visible, 'cam01.png')
        motion.observe_tree_probe(visible, 'cam02.png')
        target = motion(4)
        restored = SH.SparseHierarchicalMotion(copy.deepcopy(motion.config))
        restored.restore(copy.deepcopy(motion.capture()))
        restored.observe_tree_probe(visible, 'cam01.png')
        torch.testing.assert_close(restored.probe_observations, motion.probe_observations, rtol=0, atol=0)
        old_optimizer = motion.motion_optimizer
        actual = motion(5, training=True)
        resumed = restored(5, training=True)
        assert motion.tree_probe_complete and motion.motion_optimizer is not old_optimizer
        for expected, observed, continued in zip(target, actual, resumed):
            torch.testing.assert_close(observed, expected, rtol=0, atol=2e-5)
            torch.testing.assert_close(continued, observed, rtol=0, atol=0)
        assert motion.tree_setup['probe']['qualified_rows'] == len(raw['xyz'])
        assert motion.tree_setup['probe']['training_view_names'] == ['cam01.png', 'cam02.png']
        if builder == 'spatial':
            assert motion.tree_setup['probe']['changed_owners'] == [0, 0, 0]
        else:
            for level in range(2):
                torch.testing.assert_close(motion.parents[level][motion.index[level + 1]],
                                           motion.index[level], rtol=0, atol=0)


def test_tree_probe_parameters_are_explicit_and_full_configs_have_equal_budgets():
    import yaml
    configs = [yaml.safe_load((project_root / 'config' / name).read_text()) for name in [
        'fs_spatial_probe_p40_300_20261005.yaml', 'fs_rigid_cluster_p40_300_20261005.yaml']]
    for config in configs:
        sparse = config['module']['sparse_hierarchical']
        assert sparse['motion_steps'] + sparse['incremental_steps'] == config['trainer']['num_incr_steps'] == 340
        assert config['data']['extra_dataset_kwargs']['num_frames'] == 300
        assert sparse['storage_warmup_steps'] == sparse['tree_probe_steps'] + sparse['dense_fit_steps'] == 140
    first, second = copy.deepcopy(configs)
    first.pop('experiment_name'); second.pop('experiment_name')
    first['module']['sparse_hierarchical'].pop('tree_builder')
    second['module']['sparse_hierarchical'].pop('tree_builder')
    assert first == second
    for overrides in [dict(tree_cluster_neighbors=None), dict(tree_cluster_iterations=0),
                      dict(tree_probe_min_observations=5), dict(tree_cluster_spatial_weight=-1.),
                      dict(tree_cluster_orientation_extent=0.), dict(motion_init_decay=.5)]:
        try:
            motion_probe_fixture(**overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid probe options accepted: {overrides}')


def flat_cluster_fixture(**overrides):
    values = dict(residual_test_config().__dict__)
    values.update(motion_steps=10, tree_probe_steps=4, shared_warmup_steps=0,
        dense_fit_steps=2, gate_compress_steps=2, gate_refine_steps=2,
        enable_learned_gates=True, gate_policy='learned', gate_lr=.05,
        gate_temperature_start=1., gate_temperature_end=1.,
        initial_gate_logit=2., m_initial_gate_logit=2., gate_hard_threshold=.51,
        storage_loss_weight=.001, motion_init_decay=0., cluster_gs=6,
        tree_builder='rigid_cluster', tree_cluster_neighbors=8,
        tree_cluster_iterations=3, tree_cluster_spatial_weight=0.,
        tree_cluster_orientation_extent=.01, tree_probe_min_observations=2,
        motion_xyz_step_extent=1e-6, motion_rotation_step_rad=1e-5,
        motion_export_format='sparse_parameters', tree_rebuild_interval=5)
    values.update(overrides)
    raw = residual_raw_fixture(18)
    motion = FC.FlatClusterMotion(FC.FlatClusterConfig(**values))
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    return raw, motion


def test_flat_cluster_seed_is_deterministic_and_has_no_parents():
    for count, budget in [(1, 24), (18, 6), (64, 7), (64, 1)]:
        xyz = torch.zeros(count, 3)  # Coordinate ties must use stable GS IDs.
        left = FC.stable_flat_layout(xyz, budget)
        right = FC.stable_flat_layout(xyz, budget)
        torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
        assert left[0].shape == (1, count) and left[1] == [(count + budget - 1) // budget]
        assert left[2] == right[2] == []
        assert len(torch.unique(left[0])) == left[1][0]
        layout = GR._layout(*left, xyz)
        arrays = SL_CODEC.numpy_layout(layout)
        SL_CODEC.validate_layout(arrays)
        assert SL_CODEC.layout_digest(arrays) == SL_CODEC.layout_digest(SL_CODEC.numpy_layout(layout))


def test_flat_clustering_uses_shared_motion_and_preserves_unobserved_rows():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(120, 3)) * .04
    seed = np.repeat(np.arange(2), 60)
    moving = np.r_[np.zeros(45), np.ones(15), np.ones(45), np.zeros(15)].astype(bool)
    base_q = np.tile([1., 0., 0., 0.], (120, 1))
    target_q = base_q.copy()
    target_q[moving] = [np.cos(.1), 0., 0., np.sin(.1)]
    target = np.einsum('nij,nj->ni', MC.relative_matrices(base_q, target_q), source)
    target[moving, 0] += .05
    observations = np.full(120, 2)
    observations[0] = 1
    options = dict(extent=1., neighbors=8, iterations=3,
                   orientation_extent=.01, min_observations=2)
    owner, report = MC.cluster_flat_motion_layout(source, base_q, target, target_q,
        seed, 2, observations, **options)
    repeated, repeated_report = MC.cluster_flat_motion_layout(source, base_q, target,
        target_q, seed, 2, observations, **options)
    np.testing.assert_array_equal(owner, repeated)
    assert report == repeated_report and owner.shape == (1, 120)
    assert owner[0, 0] == seed[0]
    assert report['levels'][0]['after']['objective'] < report['levels'][0]['before']['objective'] * .2
    unknown, unknown_report = MC.cluster_flat_motion_layout(source, base_q, target,
        target_q, seed, 2, np.zeros(120), **options)
    np.testing.assert_array_equal(unknown[0], seed)
    assert unknown_report['qualified_rows'] == 0
    assert np.isfinite(unknown_report['levels'][0]['after']['objective'])


def test_flat_probe_projection_and_resume_preserve_teacher_pose():
    raw, motion = flat_cluster_fixture(m_fine_sharing=True, m_fine_gate_lr=.02)
    motion(4, training=True)
    with torch.no_grad():
        motion.gs_translation[:, 0] = .01
        motion.gs_rotation[:, 2] = .02
    visibility = torch.ones(len(raw['xyz']), dtype=torch.bool)
    motion.observe_tree_probe(visibility, 'cam01.png')
    motion.observe_tree_probe(visibility, 'cam01.png')
    torch.testing.assert_close(motion.probe_observations, torch.ones(18), rtol=0, atol=0)
    motion.observe_tree_probe(visibility, 'cam02.png')
    teacher = motion(4)
    restored = FC.FlatClusterMotion(copy.deepcopy(motion.config))
    restored.restore(copy.deepcopy(motion.capture()))
    actual, resumed = motion(5, training=True), restored(5, training=True)
    assert motion.index.shape == (1, 18) and motion.parents == []
    assert len(motion.region_translation) == len(motion.region_rotation) == 1
    assert len(motion.z_region_logits) == 1 and len(motion.statistics()['levels']) == 1
    assert motion.tree_setup['probe']['qualified_rows'] == 18
    for expected, observed, continued in zip(teacher, actual, resumed):
        torch.testing.assert_close(observed, expected, rtol=0, atol=2e-5)
        torch.testing.assert_close(continued, observed, rtol=0, atol=0)


def test_flat_shared_motion_local_residual_and_closed_m_are_independent():
    raw, motion = flat_cluster_fixture()
    motion(5, training=True)
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .1
        motion.gs_translation[:, 1] = .02
        motion.m_logits.fill_(20.)
        motion.m_logits[0] = -20.
        motion.z_region_logits[0].fill_(20.)
        motion.z_region_logits[0][motion.index[0, 1]] = -20.
        motion.z_local_logits.fill_(-20.)
        motion.z_local_logits[1] = 20.
    xyz, rotation = motion(9, training=True)
    torch.testing.assert_close(xyz[0], raw['xyz'][0], rtol=0, atol=0)
    torch.testing.assert_close(rotation[0], raw['rotation'][0], rtol=0, atol=0)
    torch.testing.assert_close(xyz[1] - raw['xyz'][1], torch.tensor([0., .02, 0.]),
                               rtol=0, atol=1e-6)
    masks = motion._hard_masks_for_step(9)
    encoded = SH.encode_m_tree(masks['m'], motion.active_layout)
    torch.testing.assert_close(SH.decode_m_tree(*encoded, motion.active_layout),
                               masks['m'], rtol=0, atol=0)
    assert torch.isfinite(motion.storage_bit_counts(9)['m_bits'])


def test_flat_parameter_storage_export_and_checkpoint_roundtrip():
    torch.manual_seed(41)
    for parameter_format, all_closed in itertools.product(
            ['raw_parameters', 'sparse_parameters'], [False, True]):
        raw, motion = flat_cluster_fixture(motion_export_format=parameter_format,
            rebase_on_hardening=True, m_fine_sharing=True, m_fine_gate_lr=.02)
        motion(5, training=True)
        with torch.no_grad():
            for value in motion.motion_parameters():
                value.copy_(torch.randn_like(value) * .01)
            motion.m_logits.fill_(-20. if all_closed else 20.)
            if not all_closed:
                motion.m_logits[::2] = -20.
            motion.z_region_logits[0][1] = -20.
            motion.z_local_logits[1::3] = -20.
        expected = motion(9, training=True)
        masks = motion._hard_masks_for_step(9)
        restored = FC.FlatClusterMotion(copy.deepcopy(motion.config))
        restored.restore(copy.deepcopy(motion.capture()))
        for left, right in zip(expected, restored(9)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'flat.npz'
            exported = motion.export_motion(path)
            with np.load(path, allow_pickle=False) as packet:
                assert packet['counts'].shape == (1,)
                assert not any(name.startswith('parents_l') or name.endswith('_l1')
                               or name.endswith('_l2') for name in packet.files)
                arrays = {name: packet[name] for name in packet.files}
            actual_bytes = motion.raw_parameter_bytes(arrays)
            assert exported['parameter_bytes'] == actual_bytes
            torch.testing.assert_close(sum(motion.storage_parts(9, hard=True).values()),
                                       torch.tensor(float(actual_bytes)), rtol=0, atol=0)
            decoded = FC.FlatClusterMotion.decode_export_motion(path, raw['xyz'], raw['rotation'])
            for left, right in zip(expected, decoded):
                torch.testing.assert_close(left, torch.from_numpy(right), rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(torch.from_numpy(decoded[0])[~masks['m']],
                                       raw['xyz'][~masks['m']], rtol=0, atol=0)
            try:
                SH.SparseHierarchicalMotion.decode_export_motion(path, raw['xyz'], raw['rotation'])
            except ValueError:
                pass
            else:
                raise AssertionError('Hierarchy decoder must reject flat representation')


def test_flat_survivor_and_birth_memberships_reuse_without_parent_tables():
    raw, motion = flat_cluster_fixture()
    xyz, rotation = motion(9, training=True)
    motion.committed = True
    sources = torch.tensor([0, 4])
    materialized_xyz = torch.cat((xyz, xyz[sources]))
    materialized_rotation = torch.cat((rotation, rotation[sources]))
    old_index = motion.index.clone()
    motion.register_materialized_layout(materialized_xyz, sources)
    rows = torch.tensor([0, 2, 4, 7, 12, 18, 19])
    expected = torch.cat((old_index, old_index[:, sources]), dim=1)[:, rows]
    motion.setup(materialized_xyz[rows], materialized_rotation[rows], 1.,
                 previous_rows=rows, previous_new_count=2, frame_index=2)
    torch.testing.assert_close(motion.index, expected, rtol=0, atol=0)
    assert motion.parents == [] and motion.tree_setup['action'] == 'reuse'
    assert motion.tree_setup['region_counts'] == [3]
    motion.setup(materialized_xyz[rows], materialized_rotation[rows], 1., frame_index=5)
    assert motion.parents == [] and motion.index.shape == (1, 7)
    assert motion.tree_setup['action'] == 'rebuild' and motion.counts == [2]


def test_flat_gate_phase_losses_have_finite_gradients_and_freeze_motion():
    _, motion = flat_cluster_fixture(gs_activity_loss_weight=.001,
                                     m_fine_sharing=True, m_fine_gate_lr=.02)
    motion(7, training=True)
    storage = sum(motion.storage_parts(7).values())
    loss = storage / motion.reference_storage_bytes(18) + motion.gs_activity_fraction(7)
    loss.backward()
    assert all(value.grad is None for value in motion.motion_parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in motion.gate_parameters())
    motion.optimizer_step()
    assert len(motion.region_translation) == 1


def test_flat_fullres_candidate_budget_and_actual_image_size_guard():
    import yaml
    path = project_root / 'config' / 'fs_flat_cluster_p40_fullres_300.yaml'
    config = yaml.safe_load(path.read_text())
    parameters = FC.FlatClusterConfig(**config['module']['flat_cluster'])
    parameters.validate()
    assert parameters.cluster_gs == 24 and parameters.tree_cluster_spatial_weight == 0.
    assert parameters.motion_steps == config['module']['deformation']['num_stage1_steps'] == 240
    assert parameters.total_steps == config['trainer']['num_incr_steps'] == 340
    assert config['data']['extra_dataset_kwargs']['num_frames'] == 300
    assert config['data']['resolution'] == 1
    assert config['data']['expected_image_size'] == [2689, 2016]
    view = SimpleNamespace(image=torch.zeros(3, 4, 6))
    assert DATA._collate_single_view([view], [6, 4]) is view
    assert DATA._collate_single_view([view]) is view
    try:
        DATA._collate_single_view([view], [3, 2])
    except ValueError:
        pass
    else:
        raise AssertionError('Actual tensor size must reject unexpected downsampling')
    for changes in [dict(cluster_gs=0), dict(tree_cluster_spatial_weight=.001),
                    dict(motion_export_format='legacy')]:
        invalid = copy.deepcopy(config['module']['flat_cluster'])
        invalid.update(changes)
        try:
            FC.FlatClusterConfig(**invalid).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid flat settings accepted: {changes}')


def test_module_setup_selects_flat_model_and_shared_birth_config():
    _, fixture = flat_cluster_fixture()
    path = project_root / 'pipeline' / 'ReCon-GS' / 'module.py'
    module_class = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name == 'Module')
    method = next(node for node in module_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'setup')

    class SetupGaussian:
        def __init__(self, degree):
            self.degree = degree

        def create_from_pcd(self, point_cloud, extent):
            self.extent = extent

        def training_setup(self, config):
            self.config = config

    scope = dict(torch=torch, SPARSE_MOTION_MODES={'sparse_hierarchical', 'flat_cluster'},
        QUANTIZED_MOTION_MODES={'quantized_residual', 'sparse_hierarchical', 'flat_cluster'},
        FlatClusterMotion=FC.FlatClusterMotion, SparseHierarchicalMotion=SH.SparseHierarchicalMotion,
        GaussianModel=SetupGaussian, Renderer=lambda config, model: SimpleNamespace(gaussians=model),
        Deformation=lambda config: SimpleNamespace(), Timer=lambda: SimpleNamespace())
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    subject = SimpleNamespace(device='cpu',
        config=SimpleNamespace(motion_mode='flat_cluster', sparse_export_index_encoding='per_row',
            flat_cluster=copy.deepcopy(fixture.config), random_background=False,
            selective_diagnostics_config=None, sh_degree=1, renderer=None, gaussian=None,
            quantized_diagnostics=SimpleNamespace(validate=lambda: None),
            deformation=SimpleNamespace(hierarchy_mode='legacy', num_grid_levels=3, num_stage1_steps=10)),
        trainer=SimpleNamespace(config=SimpleNamespace(num_incr_steps=fixture.config.total_steps)),
        datamodule=SimpleNamespace(config=SimpleNamespace(white_background=False),
            scene=SimpleNamespace(point_cloud=None), cameras_extent=1.))
    subject.register_buffer = lambda name, value: setattr(subject, name, value)
    scope['setup'](subject)
    assert isinstance(subject.residual_motion, FC.FlatClusterMotion)
    assert subject.config.quantized_residual is subject.config.flat_cluster
    assert subject.residual_motion.num_motion_levels == 1


def binary_refinement_fixture(**overrides):
    import yaml
    path = project_root / 'config' / 'fs_binary_quantized_refine_24f_20261006.yaml'
    values = yaml.safe_load(path.read_text(encoding='utf-8'))['module']['binary_refinement']
    # Synthetic budgets and units, not experiment parameter changes.
    values.update(coarse_region_gs=12, min_child_gs=2, coarse_fit_steps=2,
                  dense_fit_steps=4, gate_compress_steps=4, gate_refine_steps=2,
                  motion_steps=10, incremental_steps=2, storage_warmup_steps=4,
                  storage_ramp_steps=1, motion_xyz_step_extent=.1,
                  motion_rotation_step_rad=.1, densification_stats_steps=2)
    values.update(overrides)
    raw = residual_raw_fixture(18)
    motion = BR.BinaryRefinementMotion(BR.BinaryRefinementConfig(**values))
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    return raw, motion


def test_binary_spatial_tree_is_deterministic_and_has_only_real_children():
    for count in (1, 3, 18, 49):
        xyz = torch.zeros(count, 3)
        left = BR.stable_binary_forest(xyz, 12, 2, 3)
        right = BR.stable_binary_forest(xyz, 12, 2, 3)
        torch.testing.assert_close(left['index'], right['index'], rtol=0, atol=0)
        assert left['counts'] == right['counts']
        for level, parents in enumerate(left['parents']):
            children = torch.bincount(parents, minlength=left['counts'][level])
            assert bool(((children == 0) | (children == 2)).all())
            owner = left['index'][level + 1]
            sizes = torch.bincount(owner[owner >= 0], minlength=left['counts'][level + 1])
            assert bool((sizes >= 2).all())
            valid = owner >= 0
            torch.testing.assert_close(parents[owner[valid]], left['index'][level, valid],
                                       rtol=0, atol=0)
    terminal = BR.stable_binary_forest(torch.zeros(3, 3), 12, 2, 3)
    assert terminal['counts'] == [1, 0, 0]
    assert not bool((terminal['index'][1:] >= 0).any())


def test_binary_stop_inherits_parent_and_closes_all_descendants():
    raw, motion = binary_refinement_fixture()
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .2
        motion.region_translation[1][:, 1] = .3
        motion.region_translation[2][:, 2] = .4
        motion.refine_logits[0].fill_(-20.)
        motion.refine_logits[1].fill_(20.)
    xyz, rotation = motion(9, training=True)
    torch.testing.assert_close(xyz, raw['xyz'] + torch.tensor([.2, 0., 0.]),
                               rtol=0, atol=1e-6)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)
    assert [row['reachable_nodes'] for row in motion.statistics()['levels']][1:] == [0, 0]
    assert motion.statistics()['gs']['terminal_depth_counts'] == [len(raw['xyz']), 0, 0]


def test_binary_zero_parent_can_refine_and_quantization_zero_reuses_exact_pose():
    raw, motion = binary_refinement_fixture()
    with torch.no_grad():
        for gate in motion.refine_logits:
            gate.fill_(20.)
        motion.region_translation[1][:, 0] = .3
    xyz, _ = motion(9, training=True)
    torch.testing.assert_close(xyz, raw['xyz'] + torch.tensor([.3, 0., 0.]),
                               rtol=0, atol=1e-6)
    assert motion.statistics()['levels'][0]['nonzero_motion_records'] == 0
    assert motion.statistics()['levels'][1]['nonzero_motion_records'] > 0
    raw, zero = binary_refinement_fixture()
    raw['rotation'] = raw['rotation'] * 2.3
    # Shared checkpoints contain raw, non-unit quaternions. Identity motion
    # must not normalize them and thereby create an unencoded pose update.
    zero.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    with torch.no_grad():
        for value in zero.motion_parameters():
            value.fill_(.049)  # Less than half the synthetic 0.1 bin.
    for actual, expected in zip(zero(9, training=True), (raw['xyz'], raw['rotation'])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'zero.npz'
        zero.export_motion(path)
        decoded = BR.BinaryRefinementMotion.decode_export_motion(path, raw['xyz'], raw['rotation'])
        for actual, expected in zip(decoded, (raw['xyz'], raw['rotation'])):
            torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=0, atol=0)


def test_binary_closed_ste_gate_has_image_gradient_and_joint_motion_updates():
    raw, motion = binary_refinement_fixture()
    with torch.no_grad():
        motion.region_translation[1][:, 0] = .3
        motion.refine_logits[0].fill_(-.1)
    xyz, _ = motion(5, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    (-xyz[:, 0].sum()).backward()
    assert bool((motion.refine_logits[0].grad < 0).all())
    assert torch.isfinite(motion.region_translation[0].grad).all()
    assert motion.region_translation[0].requires_grad
    assert motion.refine_logits[0].requires_grad
    before = motion.refine_logits[0].detach().clone()
    motion.optimizer_step()
    assert bool((motion.refine_logits[0].detach() > before).all())
    assert motion.gate_optimizer.defaults['eps'] == 1e-15
    assert motion.motion_optimizer.defaults['eps'] == 1e-8
    _, coarse = binary_refinement_fixture()
    coarse(1, training=True)
    assert coarse.region_translation[0].requires_grad
    assert not coarse.region_translation[1].requires_grad
    assert not coarse.refine_logits[0].requires_grad
    coarse(3, training=True)
    assert coarse.region_translation[1].requires_grad
    assert not coarse.refine_logits[0].requires_grad


def test_binary_sparse_export_cost_decode_and_checkpoint_roundtrip():
    torch.manual_seed(63)
    for closed in (False, True):
        raw, motion = binary_refinement_fixture()
        with torch.no_grad():
            for value in motion.motion_parameters():
                value.copy_(torch.randn_like(value) * .13)
            motion.refine_logits[0].fill_(-20. if closed else 20.)
            motion.refine_logits[1][::2] = -20.
        expected = motion(9, training=True)
        restored = BR.BinaryRefinementMotion(copy.deepcopy(motion.config))
        restored.restore(copy.deepcopy(motion.capture()))
        for actual, target in zip(restored(9), expected):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'binary.npz'
            report = motion.export_motion(path)
            with np.load(path, allow_pickle=False) as packet:
                arrays = {name: packet[name] for name in packet.files}
            cost = BR.BinaryRefinementMotion.raw_parameter_bytes(arrays)
            assert cost == report['parameter_bytes']
            torch.testing.assert_close(sum(motion.storage_parts(9, hard=True).values()),
                                       torch.tensor(float(cost)), rtol=0, atol=0)
            decoded = BR.BinaryRefinementMotion.decode_export_motion(path, raw['xyz'], raw['rotation'])
            for actual, target in zip(decoded, expected):
                torch.testing.assert_close(torch.from_numpy(actual), target,
                                           rtol=0, atol=0)
            if closed:
                assert arrays['node_rows_l1'].size == arrays['motion_rows_l1'].size == 0
                assert arrays['node_rows_l2'].size == arrays['motion_rows_l2'].size == 0
            try:
                BR.BinaryRefinementMotion.decode_export_motion(path, raw['xyz'] + 1., raw['rotation'])
            except ValueError as error:
                assert 'digest' in str(error)
            else:
                raise AssertionError('Wrong motion base must be rejected')


def test_binary_small_terminal_tree_and_expected_storage_gradients_are_finite():
    _, motion = binary_refinement_fixture(coarse_region_gs=100, min_child_gs=12)
    assert motion.counts == [1, 0, 0]
    xyz, rotation = motion(5, training=True)
    loss = xyz.sum() + rotation.sum() + sum(motion.storage_parts(5).values()) / 1000
    loss.backward()
    motion.optimizer_step()
    assert torch.isfinite(motion.region_translation[0]).all()
    _, joint = binary_refinement_fixture()
    with torch.no_grad():
        joint.region_translation[1][:, 0] = .3
    joint(5, training=True)
    sum(joint.storage_parts(5).values()).backward()
    for gate in joint.refine_logits:
        assert gate.grad is not None and torch.isfinite(gate.grad).all()
    assert bool((joint.refine_logits[0].grad > 0).all())


def test_binary_joint_optimizer_resume_and_readonly_eval_are_exact():
    raw, motion = binary_refinement_fixture()
    with torch.no_grad():
        motion.region_translation[0][:, 0] = .1
        motion.region_translation[1][:, 1] = .2

    def train_one(subject, step):
        xyz, _ = subject(step, training=True)
        target = raw['xyz'] + torch.tensor([.2, .1, 0.])
        loss = (xyz - target).square().mean() + 1e-6 * sum(subject.storage_parts(step).values())
        loss.backward()
        subject.optimizer_step()

    train_one(motion, 5)
    packet = copy.deepcopy(motion.capture())
    restored = BR.BinaryRefinementMotion(copy.deepcopy(motion.config))
    restored.restore(packet)
    phase, optimizer = restored.current_phase, restored.optimizer
    restored(9, training=False)
    assert restored.current_phase == phase and restored.optimizer is optimizer
    assert restored.hard_masks is None
    train_one(motion, 6)
    train_one(restored, 6)
    for left, right in zip(motion.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_binary_decoder_rejects_child_under_stopped_parent():
    raw, motion = binary_refinement_fixture()
    with torch.no_grad():
        for gate in motion.refine_logits:
            gate.fill_(20.)
        motion.region_translation[2][:, 0] = .2
    motion(9, training=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'binary.npz'
        motion.export_motion(path)
        with np.load(path, allow_pickle=False) as packet:
            arrays = {name: packet[name] for name in packet.files}
        arrays['refine_mask_l0'][0] = False
        damaged = Path(directory) / 'damaged.npz'
        np.savez(damaged, **arrays)
        try:
            BR.BinaryRefinementMotion.decode_export_motion(damaged, raw['xyz'], raw['rotation'])
        except ValueError:
            pass
        else:
            raise AssertionError('A stopped parent cannot own a reachable child')


def test_binary_experiment_config_matches_confirmed_parameters():
    import yaml
    path = project_root / 'config' / 'fs_binary_quantized_refine_24f_20261006.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    parameters = BR.BinaryRefinementConfig(**config['module']['binary_refinement'])
    parameters.validate()
    assert config['data']['extra_dataset_kwargs']['num_frames'] == 25
    assert config['data']['resolution'] == -1
    assert config['data']['expected_image_size'] == [1600, 1199]
    assert parameters.coarse_fit_steps == 50 and parameters.dense_fit_steps == 100
    assert parameters.gate_compress_steps == 80 and parameters.gate_refine_steps == 20
    assert parameters.total_steps == config['trainer']['num_incr_steps'] == 300
    assert parameters.motion_steps == config['module']['deformation']['num_stage1_steps'] == 200
    assert parameters.coarse_region_gs == 96 and parameters.min_child_gs == 12
    assert parameters.max_motion_levels == 3
    assert parameters.motion_xyz_step_extent == 1e-4
    assert parameters.motion_rotation_step_rad == 1e-3


if __name__ == '__main__':
    checks = [value for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for check in checks:
        check()
    print(f'All {len(checks)} nested prototype numerical checks passed.')
