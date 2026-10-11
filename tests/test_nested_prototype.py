"""Numerical checks without loading the renderer or a scene."""
import importlib
import ast
import copy
from dataclasses import asdict, fields
import itertools
import json
import math
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
# Archived configs remain test inputs after experiment-directory cleanup.
legacy_config_root = project_root / 'tests' / 'fixtures' / 'config'
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
D = importlib.import_module('pipeline.ReCon-GS.archive.model.deformation')
N = importlib.import_module('pipeline.ReCon-GS.archive.model.nested')
G = importlib.import_module('pipeline.ReCon-GS.archive.model.gated_four')
R = importlib.import_module('pipeline.ReCon-GS.archive.model.region_rigid')
S = importlib.import_module('pipeline.ReCon-GS.archive.model.selective_motion')
DATA = importlib.import_module('pipeline.ReCon-GS.data')
Q = importlib.import_module('pipeline.ReCon-GS.archive.model.residual_motion')
GR = importlib.import_module('pipeline.ReCon-GS.archive.model.gated_residual')
SH = importlib.import_module('pipeline.ReCon-GS.archive.model.sparse_hierarchical')
FC = importlib.import_module('pipeline.ReCon-GS.archive.model.flat_cluster')
BR = importlib.import_module('pipeline.ReCon-GS.archive.model.binary_refinement')
SAB = importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary')
CC = importlib.import_module('pipeline.ReCon-GS.archive.model.common_contrast')
SHADOW = importlib.import_module('pipeline.ReCon-GS.archive.model.shadow_exploration')
GD = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_diagnostics')
GPS = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_probe_snapshot')
NM = importlib.import_module('pipeline.ReCon-GS.archive.model.node_motion')
PPC = importlib.import_module('pipeline.ReCon-GS.archive.model.parent_precision_compensation')
MC = importlib.import_module('pipeline.ReCon-GS.archive.model.motion_clustering')
SL_CODEC = importlib.import_module('pipeline.ReCon-GS.archive.model.sparse_layout_codec')
GS_ACTIVITY = importlib.import_module('pipeline.ReCon-GS.archive.model.gs_activity_cost')
GAUSS = importlib.import_module('pipeline.ReCon-GS.archive.model.gaussian')
ACTIVE_GAUSS = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
SPLIT_SCORES = importlib.import_module('pipeline.ReCon-GS.archive.model.split_region_scores')
DIRECT_TREE = importlib.import_module('pipeline.ReCon-GS.archive.model.direct_tree')
DIRECT_TRANSACTION = importlib.import_module('pipeline.ReCon-GS.archive.direct_transaction')
NODE_EDITS = importlib.import_module('pipeline.ReCon-GS.archive.model.tree_edit_codec')
BIRTH = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_candidates')
COLOR = importlib.import_module('pipeline.ReCon-GS.archive.model.sparse_color')
RIGID = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
RIGID_CODEC = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
RIGID_MODULE = importlib.import_module('pipeline.ReCon-GS.module')
RIGID_TRAINER = importlib.import_module('pipeline.ReCon-GS.trainer')
RIGID_BIRTH_CODEC = importlib.import_module(
    'pipeline.ReCon-GS.model.rigid_tree_birth_codec')
RIGID_REPLAY = importlib.import_module('scripts.replay_rigid_tree')


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
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'legacy_module.py'
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
    module_path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'legacy_module.py'
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
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'legacy_module.py'
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
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'legacy_module.py'
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
    source_path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'legacy_trainer.py'
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
    source = (project_root / 'pipeline/ReCon-GS/archive/legacy_module.py').read_text(encoding='utf8')
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
    path = project_root / 'scripts' / 'archive' / 'audit_closed_candidate_render.py'
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
    source_path = project_root / 'pipeline/ReCon-GS/archive/legacy_module.py'
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
    source = (project_root / 'pipeline/ReCon-GS/archive/legacy_module.py').read_text()
    start = source.index('            cfg = self.config.quantized_residual\n            stage =')
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


def test_tree_probe_parameters_are_explicit_and_fixtures_have_equal_budgets():
    configs = [asdict(motion_probe_fixture(builder)[1].config)
               for builder in ('spatial', 'rigid_cluster')]
    for config in configs:
        assert config['motion_steps'] == (config['tree_probe_steps'] + config['dense_fit_steps']
                                         + config['gate_compress_steps'] + config['gate_refine_steps'])
    first, second = copy.deepcopy(configs)
    first.pop('tree_builder')
    second.pop('tree_builder')
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


def test_flat_candidate_budget_and_actual_image_size_guard():
    _, motion = flat_cluster_fixture()
    parameters = motion.config
    parameters.validate()
    assert parameters.cluster_gs == 6 and parameters.tree_cluster_spatial_weight == 0.
    assert parameters.motion_steps == 10
    assert parameters.total_steps == parameters.motion_steps + parameters.incremental_steps
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
        invalid = asdict(parameters)
        invalid.update(changes)
        try:
            FC.FlatClusterConfig(**invalid).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid flat settings accepted: {changes}')


def test_module_setup_selects_flat_model_and_shared_birth_config():
    _, fixture = flat_cluster_fixture()
    path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'legacy_module.py'
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


def retained_experiment_config():
    import yaml
    path = legacy_config_root / 'fs_storage_aware_polish03_24f_20261007.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def retained_binary_values():
    values = retained_experiment_config()['module']['storage_aware_binary']
    names = {item.name for item in fields(BR.BinaryRefinementConfig)}
    values = {name: value for name, value in values.items() if name in names}
    values.update(max_motion_levels=3, tree_rebuild_interval=1)
    return values


def binary_refinement_fixture(**overrides):
    values = retained_binary_values()
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


def test_binary_config_uses_retained_experiment_budgets():
    config = retained_experiment_config()
    parameters = BR.BinaryRefinementConfig(**retained_binary_values())
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
    assert parameters.motion_xyz_step_extent == 1e-3
    assert parameters.motion_rotation_step_rad == 1e-2


def storage_aware_binary_fixture(**overrides):
    values = retained_experiment_config()['module']['storage_aware_binary']
    values.update(coarse_region_gs=12, min_child_gs=2, max_motion_levels=4,
        coarse_fit_steps=2, dense_fit_steps=4, gate_compress_steps=4,
        gate_refine_steps=2, motion_steps=10, incremental_steps=2,
        storage_warmup_steps=4, storage_ramp_steps=1,
        translation_steps_extent=[.1, .05, .02, .01, .005],
        rotation_steps_rad=[.1, .05, .02, .01, .005],
        motion_xyz_step_extent=.1, motion_rotation_step_rad=.1,
        densification_stats_steps=2)
    values.update(overrides)
    raw = residual_raw_fixture(48)
    config = SAB.StorageAwareBinaryConfig(**values)
    motion = SAB.StorageAwareBinaryMotion(config)
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    return raw, motion


def common_contrast_fixture(**overrides):
    import yaml
    values = yaml.safe_load((legacy_config_root /
        'fs_common_contrast_fixed16_100f.yaml').read_text())['module']['common_contrast_binary']
    values.update(coarse_region_gs=12, min_child_gs=2, max_motion_levels=4,
        coarse_fit_steps=2, dense_fit_steps=4, gate_compress_steps=4,
        gate_refine_steps=2, motion_steps=10, incremental_steps=2,
        storage_warmup_steps=4, storage_ramp_steps=1,
        translation_step_extent=.01, rotation_step_rad=.005,
        motion_xyz_step_extent=.1, motion_rotation_step_rad=.1,
        densification_stats_steps=2)
    values.update(overrides)
    raw = residual_raw_fixture(48)
    config = CC.CommonContrastConfig(**values)
    motion = CC.CommonContrastMotion(config)
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    return raw, motion


def test_common_contrast_sub_half_step_records_are_exact_identity():
    raw, motion = common_contrast_fixture(component_modes=True, rotation_moves_positions=False)
    with torch.no_grad():
        motion.root_translation.fill_(float(motion._steps[0]) * .49)
        motion.root_rotation.fill_(float(motion._steps[3]) * .49)
        for value in motion.contrast_translation:
            value.fill_(float(motion._steps[0]) * .49)
        for value in motion.contrast_rotation:
            value.fill_(float(motion._steps[3]) * .49)
        for value in motion.refine_logits:
            value.fill_(20.)
    _, _, root_symbols, contrast_symbols, _, _ = motion._quantized_parameters()
    assert not bool(root_symbols.ne(0).any())
    assert all(not bool(value.ne(0).any()) for value in contrast_symbols)
    xyz, rotation = motion(motion.config.motion_steps)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)
    assert float(motion.storage_parts(motion.config.motion_steps, hard=True)['fixed_symbol_bytes']) == 0
    metadata = motion.motion_quantization_metadata()
    assert metadata['translation_step_extent'] == .01
    assert metadata['rotation_step_rad'] == .005
    assert metadata['unused_legacy_motion_fields'] == ['motion_xyz_step_extent', 'motion_rotation_step_rad']


def test_common_contrast_zero_local_records_inherit_one_shared_motion():
    raw, motion = common_contrast_fixture(component_modes=True, rotation_moves_positions=False)
    with torch.no_grad():
        motion.root_translation[:, 0] = float(motion._steps[0]) * 1.25
        for value in motion.contrast_translation:
            value.fill_(float(motion._steps[0]) * .49)
        for value in motion.refine_logits:
            value.fill_(20.)
    _, _, root_symbols, contrast_symbols, _, _ = motion._quantized_parameters()
    assert bool(root_symbols[:, 0].eq(1).all())
    assert all(not bool(value.ne(0).any()) for value in contrast_symbols)
    xyz, _ = motion(motion.config.motion_steps)
    expected = raw['xyz'].clone()
    expected[:, 0] += motion._steps[0]
    torch.testing.assert_close(xyz, expected, rtol=0, atol=0)
    assert int((xyz != raw['xyz']).any(-1).sum()) == len(xyz)
    assert int(root_symbols.ne(0).any(-1).sum()) == motion.counts[0] < len(xyz)


def test_common_contrast_member_weights_produce_subgrid_displacements():
    raw, motion = common_contrast_fixture(component_modes=True, rotation_moves_positions=False,
                                          min_child_gs=1)
    level, pair_position = next(
        (level, position)
        for level, weights in enumerate(motion._cached_split_weights)
        for position in range(len(weights))
        if float(weights[position].min()) < .5)
    parameter_position = int(motion._cached_split_positions[level][pair_position])
    with torch.no_grad():
        motion.contrast_translation[level][parameter_position, 0] = float(motion._steps[0]) * .75
        for value in motion.refine_logits:
            value.fill_(20.)
    weights = motion._cached_split_weights[level][pair_position]
    side = 0 if weights[1] < weights[0] else 1
    child = motion._cached_split_children[level][pair_position, side]
    members = motion.index[level + 1] == child
    xyz, _ = motion(motion.config.motion_steps)
    delta = xyz[members, 0] - raw['xyz'][members, 0]
    expected = (-weights[1] if side == 0 else weights[0]) * motion._steps[0]
    torch.testing.assert_close(delta, expected.expand_as(delta), rtol=1e-5, atol=1e-7)
    assert bool(delta.ne(0).all())
    assert bool((delta.abs() < motion._steps[0] * .5).all())


def test_common_contrast_config_matches_full_run_protocol():
    import yaml
    config = yaml.safe_load((legacy_config_root /
        'fs_common_contrast_fixed16_100f.yaml').read_text())
    parameters = CC.CommonContrastConfig(**config['module']['common_contrast_binary'])
    parameters.validate()
    assert config['module']['motion_mode'] == 'common_contrast_binary'
    assert config['data']['extra_dataset_kwargs']['num_frames'] == 100
    assert config['data']['resolution'] == -1
    assert config['trainer']['num_incr_steps'] == parameters.total_steps == 300
    assert parameters.motion_steps == config['module']['deformation']['num_stage1_steps'] == 200
    assert parameters.fixed_motion_bits == 16
    assert parameters.translation_step_extent == 1e-5
    assert parameters.rotation_step_rad == 1e-4
    assert parameters.storage_loss_weight == .01


def test_common_contrast_single_parameter_derives_weighted_zero_mean_children():
    _, motion = common_contrast_fixture()
    level = next(level for level, mask in enumerate(motion._cached_binary)
                 if bool(mask.any()))
    node = int(torch.where(motion._cached_binary[level])[0][0])
    position = motion._cached_gate_positions[level][node]
    contrast = torch.tensor([.02, -.03, .01, .04, -.02, .03])
    with torch.no_grad():
        motion.contrast_translation[level][position] = contrast[:3]
        motion.contrast_rotation[level][position] = contrast[3:]
    root, values, *_ = motion._quantized_parameters()
    regions = motion._regions(root, values)
    children = [child for child in motion._cached_children[level][node]
                if bool(motion._cached_occupancy[level + 1][child])]
    wl, wr = motion._cached_weights[level][position]
    torch.testing.assert_close(wl * regions[level + 1][children[0]] +
                               wr * regions[level + 1][children[1]],
                               torch.zeros(6), rtol=0, atol=1e-8)
    assert len(motion.contrast_translation[level]) == len(motion.active_layout['refine_rows'][level])


def test_common_contrast_zero_parent_split_can_reach_nonzero_deeper_contrast():
    raw, motion = common_contrast_fixture()
    with torch.no_grad():
        for value in motion.refine_logits:
            value.fill_(20.)
        for value in motion.contrast_translation:
            value.zero_()
        deep = next(level for level in reversed(range(len(motion._cached_binary)))
                    if bool(motion._cached_binary[level].any()))
        node = int(torch.where(motion._cached_binary[deep])[0][0])
        motion.contrast_translation[deep][motion._cached_gate_positions[deep][node], 0] = .04
    xyz, _ = motion(7)
    assert bool((xyz != raw['xyz']).any())
    if deep:
        # The immediate shallower contrast can remain exactly zero while the
        # open subtree exposes this deeper detail.
        assert all(not bool(value.ne(0).any()) for value in motion.contrast_translation[:deep])


def test_common_contrast_codec_roundtrip_matches_forward_and_exact_storage():
    raw, motion = common_contrast_fixture()
    with torch.no_grad():
        motion.root_translation[0] = torch.tensor([.03, -.02, .01])
        motion.root_rotation[0] = torch.tensor([.02, .01, -.03])
        for logits in motion.refine_logits:
            logits.fill_(20.)
        for level, mask in enumerate(motion._cached_binary):
            rows = motion.active_layout['refine_rows'][level]
            valid = torch.where(mask[rows])[0]
            if len(valid):
                motion.contrast_translation[level][valid[0], 1] = .02
                break
    motion.hard_masks = motion._hard_gate_masks()
    expected = motion(10)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.bin'
        export = motion.export_motion(path, frame_index=1,
            previous_state_digest=Q.reference_digest(raw),
            post_motion_digest='00' * 32, previous_count=len(raw['xyz']),
            deleted_count=0, birth_count=0)
        actual = CC.CommonContrastMotion.decode_motion(path, raw['xyz'],
            raw['rotation'], motion.config, motion.active_layout)
        dispatched = SAB.decoder_for_binary_stream(path, motion.config,
                                                    retain_frame_state=True)
        assert isinstance(dispatched, CC.CommonContrastDecoder)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        hard_bytes = sum(float(value) for value in motion.storage_parts(10, hard=True).values())
        assert export['archive_bytes'] == hard_bytes
        assert export['payload_bits'] == motion.statistics()['hard_coded_payload_bits']


def test_common_contrast_checkpoint_restores_parameters_gates_and_optimizer():
    raw, motion = common_contrast_fixture()
    with torch.no_grad():
        motion.root_translation[:, 0] = .02
    for _ in range(2):
        xyz, _ = motion(5, training=True)
        loss = (xyz - raw['xyz'] - .03).square().mean()
        loss.backward()
        motion.optimizer_step()
    packet = copy.deepcopy(motion.capture())
    restored = CC.CommonContrastMotion(copy.deepcopy(motion.config))
    restored.restore(packet)
    for actual, expected in zip(restored.state_dict().values(), motion.state_dict().values()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(restored(5), motion(5)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for candidate in (motion, restored):
        xyz, _ = candidate(5, training=True)
        (xyz - raw['xyz'] - .03).square().mean().backward()
        candidate.optimizer_step()
    for actual, expected in zip(restored.state_dict().values(), motion.state_dict().values()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert motion.motion_optimizer.state[motion.root_translation]['step'] > 0
    motion(9, training=True)
    hardened = CC.CommonContrastMotion(copy.deepcopy(motion.config))
    hardened.restore(copy.deepcopy(motion.capture()))
    assert all(torch.equal(a, b) for a, b in zip(
        hardened.hard_masks['refine'], motion.hard_masks['refine']))


def test_common_contrast_empty_child_becomes_unary_without_gate_or_contrast_cost():
    raw, source = common_contrast_fixture()
    level = next(level for level, mask in enumerate(source._cached_binary)
                 if bool(mask.any()))
    node = int(torch.where(source._cached_binary[level])[0][0])
    children = [child for child in source._cached_children[level][node]
                if bool(source._cached_occupancy[level + 1][child])]
    remove_child = children[1]
    keep = source.active_layout['index'][level + 1] != remove_child
    subset = torch.where(keep)[0]
    override = dict(index=source.active_layout['index'][:, subset],
        counts=source.counts, parents=source.parents,
        pivots=source.active_layout['pivots'])
    motion = CC.CommonContrastMotion(copy.deepcopy(source.config))
    motion.setup(raw['xyz'][subset], raw['rotation'][subset], 1., frame_index=2,
                 layout_override=override)
    assert not bool(motion._cached_binary[level][node])
    assert int(motion._cached_unary[level][node]) == children[0]
    gates = [torch.zeros_like(value) for value in motion.refine_logits]
    paths = motion._paths(gates)
    assert bool(paths[level + 1][children[0]])
    position = motion._cached_gate_positions[level][node]
    with torch.no_grad():
        motion.contrast_translation[level][position].fill_(1.)
    root, contrasts, *_ = motion._quantized_parameters()
    regions = motion._regions(root, contrasts)
    torch.testing.assert_close(regions[level + 1][children[0]], torch.zeros(6), rtol=0, atol=0)
    # Inverse factorization of a unary branch must remove the pivot-induced
    # parent displacement even though the branch has no local parameter.
    layout = copy.deepcopy(motion.active_layout)
    for depth, pivot in enumerate(layout['pivots']):
        layout['pivots'][depth] = pivot + torch.tensor([.01, -.006, .004]) * (depth + 1)
    root_values = raw['xyz'].new_zeros(layout['counts'][0], 6)
    root_values[:, 3:] = torch.tensor([.21, -.13, .08])
    split_values = [raw['xyz'].new_zeros(len(rows), 6)
                    for rows in layout['refine_rows']]
    cumulative_q, cumulative_delta = CC.reconstruct_cumulative_transforms(
        layout, root_values, split_values)
    all_paths = [raw['xyz'].new_ones(count) for count in layout['counts']]
    terminal = NM.terminal_node_rows(layout, all_paths, selected_only=False)
    factored = CC.decompose_terminal_transforms(layout,
        torch.cat(cumulative_q)[terminal], torch.cat(cumulative_delta)[terminal])
    result_q, result_delta = CC.reconstruct_cumulative_transforms(layout, *factored)
    equivalent = torch.minimum(
        (torch.cat(result_q)[terminal] - torch.cat(cumulative_q)[terminal]).abs().amax(-1),
        (torch.cat(result_q)[terminal] + torch.cat(cumulative_q)[terminal]).abs().amax(-1))
    assert float(equivalent.max()) < 3e-6
    torch.testing.assert_close(torch.cat(result_delta)[terminal],
        torch.cat(cumulative_delta)[terminal], rtol=2e-5, atol=3e-6)


def test_common_contrast_arbitrary_terminal_factorization_handles_general_pivots():
    torch.manual_seed(7103)
    raw, motion = common_contrast_fixture()
    remove = None
    for level, mask in enumerate(motion._cached_binary):
        for node in torch.where(mask)[0].tolist():
            child = [value for value in motion._cached_children[level][node]
                     if bool(motion._cached_occupancy[level + 1][value])]
            rows = torch.where(motion.active_layout['index'][level + 1] == child[0])[0]
            if len(rows) > 1:
                remove = int(rows[0])
                break
        if remove is not None:
            break
    assert remove is not None
    keep = torch.arange(len(raw['xyz'])) != remove
    raw = {name: value[keep] for name, value in raw.items()}
    layout = BR._binary_layout(motion.active_layout['index'][:, keep], motion.counts,
        motion.parents, raw['xyz'], motion.active_layout['pivots'])
    # Deliberately violate centroid-pivot identities. The general delta formula
    # must still invert and reconstruct the cumulative terminal transforms.
    for level, pivot in enumerate(layout['pivots']):
        layout['pivots'][level] = pivot + torch.randn_like(pivot) * (.01 + level * .003)
    split_count = [len(rows) for rows in layout['refine_rows']]
    roots = torch.randn(layout['counts'][0], 6) * .025
    contrasts = [torch.randn(count, 6) * .02 for count in split_count]
    rotations, deltas = CC.reconstruct_cumulative_transforms(layout, roots, contrasts)
    paths = [raw['xyz'].new_ones(count) for count in layout['counts']]
    terminal = NM.terminal_node_rows(layout, paths, selected_only=False)
    flat_q, flat_delta = torch.cat(rotations), torch.cat(deltas)
    terminal_q, terminal_delta = flat_q[terminal], flat_delta[terminal]
    factored_roots, factored_contrasts = CC.decompose_terminal_transforms(
        layout, terminal_q, terminal_delta)
    actual_q, actual_delta = CC.reconstruct_cumulative_transforms(
        layout, factored_roots, factored_contrasts)
    actual_terminal_q = torch.cat(actual_q)[terminal]
    equivalent = torch.minimum((actual_terminal_q - terminal_q).abs().amax(-1),
                               (actual_terminal_q + terminal_q).abs().amax(-1))
    assert float(equivalent.max()) < 3e-6
    torch.testing.assert_close(torch.cat(actual_delta)[terminal], terminal_delta,
                               rtol=2e-5, atol=3e-6)
    effective = CC._effective_splits(layout)
    weights = torch.cat([value[value.sum(-1) > 0] for value in effective[-1]])
    assert bool(((weights[:, 0] - .5).abs() > 1e-3).any())
    # Independent production pose check, rather than helper self-comparison.
    root_q = CC.rotation_vector_quaternion(factored_roots[:, 3:])
    regions = [factored_roots]
    occupancy, children, _, positions, binary, _, split_weights = CC._effective_splits(layout)
    for level in range(len(layout['parents'])):
        local = roots.new_zeros(layout['counts'][level + 1], 6)
        for node in torch.where(binary[level])[0].tolist():
            position = positions[level][node]
            child = [value for value in children[level][node]
                     if bool(occupancy[level + 1][value])]
            wl, wr = split_weights[level][position]
            local[child[0]] = -wr * factored_contrasts[level][position]
            local[child[1]] = wl * factored_contrasts[level][position]
        regions.append(local)
    production_xyz, production_rotation = NM.node_conditional_transform(
        raw['xyz'], raw['rotation'], layout, regions, paths, selected_only=False,
        terminal=terminal)
    expected_xyz = (raw['xyz'] + NM.rotation_delta(terminal_q,
        raw['xyz'] - torch.cat(layout['pivots'])[terminal]) + terminal_delta)
    expected_rotation = Q.quaternion_multiply(terminal_q, raw['rotation'])
    torch.testing.assert_close(production_xyz, expected_xyz, rtol=2e-5, atol=3e-6)
    equivalent = torch.minimum((production_rotation - expected_rotation).abs().amax(-1),
                               (production_rotation + expected_rotation).abs().amax(-1))
    assert float(equivalent.max()) < 3e-6


def test_common_contrast_common_world_rigid_has_zero_split_contrasts():
    raw, motion = common_contrast_fixture()
    layout = copy.deepcopy(motion.active_layout)
    for level, pivot in enumerate(layout['pivots']):
        layout['pivots'][level] = pivot + (level + 1) * torch.tensor([.011, -.007, .004])
    omega = torch.tensor([[.31, -.17, .09]])
    world_q = R.rotation_vector_quaternion(omega)[0]
    world_b = torch.tensor([.04, -.03, .02])
    paths = [raw['xyz'].new_ones(count) for count in layout['counts']]
    terminal = NM.terminal_node_rows(layout, paths, selected_only=False)
    pivot = torch.cat(layout['pivots'])[terminal]
    terminal_q = world_q.expand(len(terminal), -1)
    terminal_delta = NM.rotation_delta(terminal_q, pivot) + world_b
    roots, contrasts = CC.decompose_terminal_transforms(layout, terminal_q, terminal_delta)
    for value in contrasts:
        torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=2e-6)


def test_common_contrast_quaternion_sign_and_near_pi_branch_are_stable():
    raw, motion = common_contrast_fixture()
    layout = motion.active_layout
    roots = raw['xyz'].new_zeros(layout['counts'][0], 6)
    contrasts = [raw['xyz'].new_zeros(len(rows), 6) for rows in layout['refine_rows']]
    level = next(level for level, value in enumerate(motion._cached_split_positions)
                 if len(value))
    position = int(motion._cached_split_positions[level][0])
    contrasts[level][position, 3] = math.pi - 2e-4
    rotations, deltas = CC.reconstruct_cumulative_transforms(layout, roots, contrasts)
    paths = [raw['xyz'].new_ones(count) for count in layout['counts']]
    terminal = NM.terminal_node_rows(layout, paths, selected_only=False)
    q = torch.cat(rotations)[terminal]
    q[::2].neg_()  # identical SO(3) representatives with alternating signs
    factored = CC.decompose_terminal_transforms(layout, q, torch.cat(deltas)[terminal])
    actual_q, actual_delta = CC.reconstruct_cumulative_transforms(layout, *factored)
    result = torch.cat(actual_q)[terminal]
    equivalent = torch.minimum((result - q).abs().amax(-1), (result + q).abs().amax(-1))
    assert float(equivalent.max()) < 5e-5
    torch.testing.assert_close(torch.cat(actual_delta)[terminal], torch.cat(deltas)[terminal],
                               rtol=2e-5, atol=3e-6)


def test_common_contrast_closed_ancestor_blocks_deep_motion_but_gate_gets_signal():
    raw, motion = common_contrast_fixture(gate_hard_threshold=.5)
    # Find an effective binary split with an effective binary descendant.
    chosen = None
    for level, rows in enumerate(motion.active_layout['refine_rows'][:-1]):
        for position, node in enumerate(rows.tolist()):
            if not bool(motion._cached_binary[level][node]):
                continue
            descendants = [child for child in motion._cached_children[level][node]
                           if bool(motion._cached_binary[level + 1][child])]
            if descendants:
                chosen = level, position, descendants[0]
                break
        if chosen:
            break
    assert chosen is not None
    level, position, deep_node = chosen
    deep_position = motion._cached_gate_positions[level + 1][deep_node]
    with torch.no_grad():
        for value in motion.refine_logits:
            value.fill_(20.)
        motion.refine_logits[level][position] = -2.
        motion.contrast_translation[level + 1][deep_position, 0] = .05
    xyz, _ = motion(5, training=True)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    loss = (xyz[:, 0] * torch.linspace(.1, 1., len(xyz))).sum()
    loss.backward()
    deep_grad = motion.contrast_translation[level + 1].grad
    assert deep_grad is None or float(deep_grad[deep_position].abs().max()) == 0.
    gate_grad = motion.refine_logits[level].grad
    assert gate_grad is not None and abs(float(gate_grad[position])) > 0
    gates = motion._gate_values(5)
    assert float(gates[level][position]) == 0.
    assert all(bool(((value.detach() == 0) | (value.detach() == 1)).all()) for value in gates)


def test_storage_aware_binary_config_matches_confirmed_parameters():
    config = retained_experiment_config()
    assert config['module']['motion_mode'] == 'storage_aware_binary'
    parameters = SAB.StorageAwareBinaryConfig(**config['module']['storage_aware_binary'])
    parameters.validate()
    assert config['data']['extra_dataset_kwargs']['num_frames'] == 25
    assert config['data']['resolution'] == -1
    assert config['data']['expected_image_size'] == [1600, 1199]
    assert config['trainer']['num_incr_steps'] == parameters.total_steps == 300
    assert parameters.motion_steps == config['module']['deformation']['num_stage1_steps'] == 200
    assert (parameters.coarse_fit_steps, parameters.dense_fit_steps,
            parameters.gate_compress_steps, parameters.gate_refine_steps,
            parameters.incremental_steps) == (50, 100, 80, 20, 100)
    assert (parameters.coarse_region_gs, parameters.min_child_gs,
            parameters.max_motion_levels, parameters.tree_rebuild_interval) == (96, 12, 4, 5)
    assert parameters.precision_bits == [8, 12, 16, 24, 32]
    assert parameters.translation_steps_extent == [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    assert parameters.rotation_steps_rad == [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    assert parameters.storage_loss_weight == .01
    assert parameters.storage_warmup_steps == 100 and parameters.storage_ramp_steps == 20
    assert parameters.gate_lr == parameters.precision_lr == .02
    assert parameters.initial_gate_logit == 0.
    assert max(range(5), key=parameters.precision_initial_logits.__getitem__) == 4


def test_storage_aware_binary_phase_schedule_and_ste_gradients():
    _, motion = storage_aware_binary_fixture()
    assert motion.tree_rebuilt
    assert [motion.stage(step) for step in (1, 2, 3, 4, 5, 8, 9, 10, 11)] == [
        'dense_fit', 'dense_fit', 'dense_fit', 'dense_fit', 'gate_compress',
        'gate_compress', 'gate_refine', 'gate_refine', 'incremental']
    assert motion.gate_probabilities(5)[0].eq(.5).all()
    assert all(torch.equal(value.argmax(-1), torch.full((len(value),), 4,
                     dtype=torch.long, device=value.device))
               for value in motion._precision_probabilities(5))

    motion(1, training=True)
    assert motion.region_translation[0].requires_grad
    assert not motion.region_translation[1].requires_grad
    assert not motion.refine_logits[0].requires_grad
    motion(3, training=True)
    assert motion.region_translation[1].requires_grad
    assert not motion.refine_logits[0].requires_grad
    with torch.no_grad():
        for value in motion.region_translation[1:]:
            value[:, 0] = .37
        motion.refine_logits[0].fill_(-.1)
    xyz, _ = motion(5, training=True)
    assert motion.region_translation[0].requires_grad
    assert motion.refine_logits[0].requires_grad
    assert motion.precision_logits[0].requires_grad
    storage = sum(motion.storage_parts(5).values())
    loss = xyz[:, 0].sum() + storage / 1000
    loss.backward()
    assert motion.refine_logits[0].grad is not None
    assert torch.isfinite(motion.refine_logits[0].grad).all()
    assert motion.precision_logits[1].grad is not None
    assert torch.isfinite(motion.precision_logits[1].grad).all()
    assert motion.precision_logits[1].grad.abs().sum() > 0


def test_storage_aware_binary_bitstream_exact_cost_and_decode():
    raw, motion = storage_aware_binary_fixture()
    with torch.no_grad():
        for value in motion.region_translation:
            value[:, 0] = .37
        for gate in motion.refine_logits:
            gate.fill_(4.)
    expected_xyz, expected_rotation = motion(10, training=True)
    hard_bits = motion._hard_rate_bits(10)
    payload, payload_bits, _, _ = motion._encode_payload(10)
    assert payload_bits == hard_bits
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'frame_001.bin'
        digest = Q.reference_digest(raw)
        post_motion = {**raw, 'xyz': expected_xyz, 'rotation': expected_rotation}
        export = motion.export_motion(path, frame_index=1,
            previous_state_digest=digest,
            post_motion_digest=Q.reference_digest(post_motion),
            previous_count=len(raw['xyz']), deleted_count=0, birth_count=0)
        assert isinstance(export['precision_bits'], list)
        header = SAB.StorageAwareBinaryMotion.read_header(path)
        assert header['extent'] == 1.
        assert header['payload_bits'] == payload_bits
        assert path.stat().st_size == SAB.CODEC_HEADER_BYTES + len(payload)
        assert export['archive_bytes'] == path.stat().st_size
        coded_storage = sum(motion.storage_parts(10, hard=True).values())
        torch.testing.assert_close(coded_storage, torch.tensor(float(path.stat().st_size)),
                                   rtol=0, atol=0)
        decoder = SAB.StorageAwareBinaryDecoder(copy.deepcopy(motion.config))
        decoded, info = decoder.advance(path, None, raw)
        assert info['tree_rebuilt']
        torch.testing.assert_close(decoded['xyz'], expected_xyz, rtol=0, atol=0)
        torch.testing.assert_close(decoded['rotation'], expected_rotation, rtol=0, atol=0)
        assert Q.reference_digest(decoded) == Q.reference_digest(
            {**raw, 'xyz': expected_xyz, 'rotation': expected_rotation})
        assert motion.statistics()['hard_coded_payload_bits'] == payload_bits


def test_storage_aware_binary_tree_rebuilds_every_fifth_frame():
    raw, motion = storage_aware_binary_fixture()
    for frame in range(1, 8):
        if frame > 1:
            rows = torch.arange(len(raw['xyz']))
            motion.setup(raw['xyz'], raw['rotation'], 1., previous_rows=rows,
                         previous_new_count=0, frame_index=frame)
        assert motion.tree_rebuilt == ((frame - 1) % 5 == 0)
        motion.register_materialized_layout(raw['xyz'], torch.empty(0, dtype=torch.long))


def test_storage_aware_newborn_quantization_can_change_nearest_source_without_breaking_decoder():
    raw = residual_raw_fixture(2)
    raw['xyz'] = torch.tensor([[0., 0., 0.], [.15, 0., 0.]])
    _, motion = storage_aware_binary_fixture(coarse_region_gs=1, min_child_gs=1,
                                            new_xyz_step_extent=.1)
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    candidate = GAUSS.GaussianModel(1)
    for name, attribute in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(candidate, attribute, torch.nn.Parameter(raw[name][:1].clone()))
    with torch.no_grad():
        candidate._xyz[0, 0] = .074  # Nearest old GS is row 0 before quantization.
    view = Q.QuantizedNewGaussianView(candidate, motion.config, 1., raw)
    materialized, predictors = Q.materialize_quantized_gaussians(raw, view)
    assert predictors.tolist() == [0]
    torch.testing.assert_close(materialized['xyz'][-1], torch.tensor([.1, 0., 0.]), rtol=0, atol=0)
    decoded_source = SAB._nearest_source_rows(raw['xyz'], materialized['xyz'][2:])
    assert decoded_source.tolist() == [1]
    original_index = motion.index.clone()
    motion.register_materialized_layout(materialized['xyz'], predictors)
    torch.testing.assert_close(motion.materialized_layout['index'][:, -1], original_index[:, 1], rtol=0, atol=0)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        birth = GAUSS.GaussianModel(1)
        for name, attribute in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
            setattr(birth, attribute, torch.nn.Parameter(materialized[name][-1:].clone()))
        birth_path = directory / 'birth.ply'
        birth.save_ply(str(birth_path))
        motion_path = directory / 'motion.bin'
        motion.export_motion(motion_path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(raw), previous_count=2, deleted_count=0, birth_count=1)
        decoder = SAB.StorageAwareBinaryDecoder(copy.deepcopy(motion.config), retain_frame_state=True)
        decoded, _ = decoder.advance(motion_path, birth_path, raw)
        assert Q.reference_digest(decoded) == Q.reference_digest(materialized)
        torch.testing.assert_close(decoder.materialized_layout['index'],
                                   motion.materialized_layout['index'], rtol=0, atol=0)
        recovery = importlib.import_module('scripts.archive.recover_storage_aware_checkpoint')
        recovered = recovery.motion_from_decoded(motion.config, decoder.last_frame_state, decoder.materialized_layout)
        decoded_motion = recovered(recovered.config.motion_steps)
        torch.testing.assert_close(decoded_motion[0], raw['xyz'], rtol=0, atol=0)
        torch.testing.assert_close(decoded_motion[1], raw['rotation'], rtol=0, atol=0)
        restored = SAB.StorageAwareBinaryMotion(copy.deepcopy(motion.config))
        restored.restore(recovered.capture())
        deleted, _ = restored.compensation_deletions(decoded['opacity'], 1)
        assert deleted.tolist() == [2]
        rows = Q.survivor_rows(len(decoded['xyz']), deleted)
        restored.setup(decoded['xyz'][rows], decoded['rotation'][rows], 1.,
            previous_rows=rows, previous_new_count=1, frame_index=2)
        torch.testing.assert_close(restored.index, original_index, rtol=0, atol=0)
    before = motion.materialized_layout['index'].clone()
    for invalid in ([2], [-1], [[0]]):
        try:
            motion.register_materialized_layout(materialized['xyz'], invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed predictor rows must be rejected')
        torch.testing.assert_close(motion.materialized_layout['index'], before, rtol=0, atol=0)


def test_storage_aware_recovered_motion_matches_nonzero_mixed_precision_bitstream():
    recovery = importlib.import_module('scripts.archive.recover_storage_aware_checkpoint')
    raw, motion = storage_aware_binary_fixture()
    torch.manual_seed(59)
    motion.prepare_step(5)
    with torch.no_grad():
        for level, (translation, rotation, logits) in enumerate(zip(
                motion.region_translation, motion.region_rotation, motion.precision_logits)):
            translation.copy_(torch.randn_like(translation) * .23)
            rotation.copy_(torch.randn_like(rotation) * .11)
            logits.fill_(-100.)
            choices = torch.arange(len(logits)) % len(motion.config.precision_bits)
            logits.scatter_(1, choices[:, None], 0.)
        for gate in motion.refine_logits:
            gate.fill_(3.)
            gate[1::3] = -3.
    motion.prepare_step(10)
    xyz, rotation = motion(10)
    materialized = {**raw, 'xyz': xyz.detach(), 'rotation': rotation.detach()}
    motion.register_materialized_layout(xyz, [])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.bin'
        motion.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(materialized), previous_count=len(raw['xyz']),
            deleted_count=0, birth_count=0)
        decoder = SAB.StorageAwareBinaryDecoder(copy.deepcopy(motion.config), retain_frame_state=True)
        decoder.advance(path, None, raw)
        recovered = recovery.motion_from_decoded(motion.config, decoder.last_frame_state, decoder.materialized_layout)
        restored_xyz, restored_rotation = recovered(10)
        assert Q.reference_digest({**raw, 'xyz': restored_xyz, 'rotation': restored_rotation}) == Q.reference_digest(materialized)
        expected = motion._encode_payload(10)
        actual = recovered._encode_payload(10)
        assert actual[0] == expected[0] and actual[1] == expected[1]
        assert recovered.committed and not any(p.requires_grad for p in recovered.parameters())


def test_storage_aware_binary_sequence_recreates_births_deletions_and_cached_tree():
    check_storage_aware_binary_sequence()


def test_node_motion_sequence_recreates_exact_states():
    check_storage_aware_binary_sequence(coarse_region_gs=24, motion_compute='node', cache_static_layout=True,
        selected_precision_only=True, cache_frozen_storage=True)


def check_storage_aware_binary_sequence(**overrides):
    raw, motion = storage_aware_binary_fixture(**overrides)
    decoder = SAB.StorageAwareBinaryDecoder(copy.deepcopy(motion.config))
    with torch.no_grad():
        for value in motion.region_translation:
            value[:, 0] = .37
        for gate in motion.refine_logits:
            gate.fill_(4.)
    moved_xyz, moved_rotation = motion(10, training=True)
    birth = {name: value[:1].clone() for name, value in raw.items()}
    birth['xyz'] = moved_xyz[:1] + torch.tensor([[.001, 0., 0.]])
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        birth_model = GAUSS.GaussianModel(1)
        for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
            setattr(birth_model, attr, torch.nn.Parameter(birth[name].clone()))
        birth_path = directory / 'birth_001.ply'
        birth_model.save_ply(str(birth_path))
        assert birth_path.stat().st_size == SAB.birth_ply_size(1, 1)
        source = SAB._nearest_source_rows(moved_xyz, birth['xyz'])
        full_xyz = torch.cat((moved_xyz, birth['xyz']), dim=0)
        motion.register_materialized_layout(full_xyz, source)
        motion_path = directory / 'motion_001.bin'
        digest = Q.reference_digest(raw)
        post_motion = {**raw, 'xyz': moved_xyz, 'rotation': moved_rotation}
        motion.export_motion(motion_path, frame_index=1,
            previous_state_digest=digest,
            post_motion_digest=Q.reference_digest(post_motion),
            previous_count=len(raw['xyz']), deleted_count=0, birth_count=1)
        decoded, first = decoder.advance(motion_path, birth_path, raw)
        assert first['tree_rebuilt'] and first['birth_count'] == 1
        expected = {name: value.clone() for name, value in raw.items()}
        expected['xyz'] = torch.cat((moved_xyz, birth['xyz']), dim=0)
        expected['rotation'] = torch.cat((moved_rotation, birth['rotation']), dim=0)
        for name in Q.RAW_FIELDS:
            if name not in ('xyz', 'rotation'):
                expected[name] = torch.cat((raw[name], birth[name]), dim=0)
        assert Q.reference_digest(decoded) == Q.reference_digest(expected)

        motion.committed = True
        deleted, _ = motion.compensation_deletions(decoded['opacity'], 1)
        rows = Q.survivor_rows(len(decoded['xyz']), deleted)
        base = {name: value[rows].clone() for name, value in decoded.items()}
        motion.setup(base['xyz'], base['rotation'], 1., previous_rows=rows,
                     previous_new_count=1, frame_index=2)
        assert not motion.tree_rebuilt
        with torch.no_grad():
            for value in motion.region_translation:
                value[:, 1] = .11
            for gate in motion.refine_logits:
                gate.fill_(4.)
        frame2_xyz, frame2_rotation = motion(10, training=True)
        frame2_path = directory / 'motion_002.bin'
        expected2 = {name: value.clone() for name, value in base.items()}
        expected2['xyz'], expected2['rotation'] = frame2_xyz, frame2_rotation
        motion.export_motion(frame2_path, frame_index=2,
            previous_state_digest=Q.reference_digest(decoded),
            post_motion_digest=Q.reference_digest(expected2),
            previous_count=len(decoded['xyz']), deleted_count=1, birth_count=0)
        decoded2, second = decoder.advance(frame2_path, None, decoded)
        assert second['deleted_rows'] == deleted.tolist()
        assert not second['tree_rebuilt']
        assert Q.reference_digest(decoded2) == Q.reference_digest(expected2)


def test_strict_initial_gs_budget_selects_clone_split_by_gradient_and_still_prunes():
    class BudgetModel:
        def __init__(self, prune=False):
            self._xyz = torch.zeros(4, 3)
            self.xyz_gradient_accum = torch.tensor([[5.], [9.], [9.], [1.]])
            self.denom = torch.ones(4, 1)
            self.clone_rows = self.split_rows = None
            self.remove_one = prune

        @property
        def get_opacity(self):
            values = torch.ones(len(self._xyz), 1)
            if self.remove_one:
                values[0] = 0
            return values

        def densify_and_clone(self, grads, threshold, extent):
            self.clone_rows = torch.where((grads[:, 0] >= threshold) &
                                          torch.tensor([True, True, False, False]))[0]
            self._xyz = torch.cat((self._xyz, torch.zeros(len(self.clone_rows), 3)))

        def densify_and_split(self, grads, threshold, extent):
            self.split_rows = torch.where((grads[:, 0] >= threshold) &
                                          torch.tensor([False, False, True, True]))[0]
            self._xyz = torch.cat((self._xyz, torch.zeros(len(self.split_rows), 3)))

        def prune_points(self, mask):
            self._xyz = self._xyz[~mask]

    model = BudgetModel()
    result = GAUSS.GaussianModel.densify_and_prune(model, .5, .005, 1., None, max_points=6)
    assert model.clone_rows.tolist() == [1] and model.split_rows.tolist() == [2]
    assert result['after'] == 6 and result['selected'] == 2
    full = BudgetModel(prune=True)
    result = GAUSS.GaussianModel.densify_and_prune(full, .5, .005, 1., None, max_points=4)
    assert result['selected'] == 0 and result['after'] == 3


def test_render_attribute_cache_tracks_inplace_replacement_override_and_gradients():
    raw = residual_raw_fixture(8)
    model = GAUSS.GaussianModel(1)
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(raw[name].clone()))
    model.render_attribute_cache = True
    with torch.no_grad():
        old = model.get_opacity
        assert old is model.get_opacity
        model._opacity.add_(.3)
        changed = model.get_opacity
        assert changed is not old
        torch.testing.assert_close(changed, torch.sigmoid(model._opacity), rtol=0, atol=0)
        model._opacity = torch.nn.Parameter(model._opacity.clone() + .2)
        replaced = model.get_opacity
        assert replaced is not changed
        assert model._render_cache['opacity'][1][0] is model._opacity
        model.rigid_rotation = raw['rotation'].clone()
        first = model.get_rotation
        model.rigid_rotation = torch.flip(raw['rotation'], dims=[0]).clone()
        second = model.get_rotation
        assert second is not first
        torch.testing.assert_close(second, F.normalize(model.rigid_rotation, dim=-1), rtol=0, atol=0)
        model.rigid_rotation = None
        model.get_scaling
        model.get_features
    model.get_scaling.sum().backward()
    assert model._scaling.grad is not None and model._scaling.grad.abs().sum() > 0
    model.get_features.sum().backward()
    assert model._features_dc.grad is not None
    model._scaling.requires_grad_(False)
    assert model.get_scaling is model.get_scaling
    model.rigid_rotation = raw['rotation'].clone().requires_grad_()
    assert model.get_rotation.requires_grad
    assert model.get_rotation is not model.get_rotation


def test_node_motion_values_and_ste_gradients_match_reference():
    torch.manual_seed(17)
    raw = residual_raw_fixture(48)
    for dtype, atol, rtol in ((torch.float64, 2e-10, 2e-8),
                             (torch.float32, 3e-5, 3e-4)):
        xyz, rotation = raw['xyz'].to(dtype), raw['rotation'].to(dtype)
        layout = BR.stable_binary_forest(xyz, 24, 2, 4)
        assert all(layout['counts'])
        source = [torch.randn(count, 6, dtype=dtype) * .04 for count in layout['counts']]
        logits = [torch.linspace(-1., 1., len(rows), dtype=dtype)
                  for rows in layout['refine_rows']]
        xyz_weight = torch.randn_like(xyz)
        rotation_weight = torch.randn_like(rotation)

        def evaluate(transform):
            regions = [value.clone().requires_grad_() for value in source]
            gates = [value.clone().requires_grad_() for value in logits]
            paths = [xyz.new_ones(layout['counts'][0])]
            for level, parent in enumerate(layout['parents']):
                prob = gates[level].sigmoid()
                weight = (prob >= .5).to(prob) + (prob - prob.detach())
                node_gate = xyz.new_zeros(layout['counts'][level]).index_copy(
                    0, layout['refine_rows'][level], weight)
                paths.append(paths[level][parent] * node_gate[parent])
            result = transform(xyz, rotation, layout, regions, paths)
            loss = (result[0] * xyz_weight).sum() + (result[1] * rotation_weight).sum()
            inputs = regions + gates
            gradients = torch.autograd.grad(loss, inputs, allow_unused=True)
            gradients = [torch.zeros_like(value) if gradient is None else gradient
                         for value, gradient in zip(inputs, gradients)]
            return result, gradients

        reference, reference_grad = evaluate(BR._conditional_transform)
        actual, actual_grad = evaluate(NM.node_conditional_transform)
        for left, right in zip(actual, reference):
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        for left, right in zip(actual_grad, reference_grad):
            torch.testing.assert_close(left, right, atol=atol, rtol=rtol)


def test_inference_attribute_read_does_not_poison_frozen_training_cache():
    model = GAUSS.GaussianModel(1)
    model.render_attribute_cache = True
    model._scaling = torch.nn.Parameter(torch.zeros(8, 3), requires_grad=False)
    with torch.inference_mode():
        inference_scales = model.get_scaling
        assert torch.is_inference(inference_scales)
    scales = model.get_scaling
    assert not torch.is_inference(scales)
    probe = torch.ones_like(scales, requires_grad=True)
    (probe * scales).sum().backward()
    torch.testing.assert_close(probe.grad, scales, rtol=0, atol=0)


def test_node_motion_exact_zero_paths_keep_raw_pose_and_opening_gradient():
    raw = residual_raw_fixture(48)
    raw['rotation'] = raw['rotation'] * 3
    layout = BR.stable_binary_forest(raw['xyz'], 24, 2, 4)
    regions = [raw['xyz'].new_zeros(count, 6) for count in layout['counts']]
    regions[1][:, 0] = .03
    gate = torch.zeros(layout['counts'][0], requires_grad=True)
    paths = [torch.ones_like(gate), gate[layout['parents'][0]]]
    for level in range(2, 4):
        paths.append(paths[level - 1][layout['parents'][level - 1]])
    xyz, rotation = NM.node_conditional_transform(raw['xyz'], raw['rotation'], layout, regions, paths)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    torch.testing.assert_close(rotation, raw['rotation'], rtol=0, atol=0)
    xyz[:, 0].sum().backward()
    assert gate.grad is not None and gate.grad.abs().sum() > 0


def test_cached_storage_and_selected_precision_preserve_costs_and_gradients():
    _, reference = storage_aware_binary_fixture()
    optimized = copy.deepcopy(reference)
    optimized.config.cache_static_layout = True
    optimized.config.selected_precision_only = True
    optimized.config.cache_frozen_storage = True
    torch.manual_seed(23)
    with torch.no_grad():
        for a, b in zip(reference.region_translation, optimized.region_translation):
            a.copy_(torch.randn_like(a) * .2)
            b.copy_(a)
    for step in (1, 3, 5, 9, 11):
        reference.prepare_step(step)
        optimized.prepare_step(step)
        for a, b in zip(reference(step), optimized(step)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for hard in (False, True):
            left, right = reference.storage_parts(step, hard), optimized.storage_parts(step, hard)
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        assert optimized._hard_rate_bits(step) == reference._hard_rate_bits(step)
    assert optimized._frozen_storage_parts
    assert all(not value.requires_grad for value in optimized.storage_parts(11).values())


def test_reference_motion_preserves_original_call_order_and_diagnostics():
    _, motion = storage_aware_binary_fixture()
    motion.config.motion_compute = 'reference'
    motion.config.selected_precision_only = False
    motion.prepare_step(1)
    events = []
    original_regions, original_paths = motion._region_motion, motion._paths

    def regions(step=None, **kwargs):
        events.append(('regions', step))
        return original_regions(step, **kwargs)

    def paths(gates):
        events.append(('paths', None))
        return original_paths(gates)

    motion._region_motion, motion._paths = regions, paths
    motion(1)
    assert events == [('regions', 1), ('paths', None)]
    events.clear()
    motion._hard_masks_for_step(3)
    assert events == [('paths', None), ('regions', None)]

    with torch.no_grad():
        for logits in motion.precision_logits:
            logits.fill_(-30.)
            logits[:, 0] = 0.
            logits[:, 1] = 2e-8
    expected = [value.argmax(-1) for value in motion._precision_weights(
        5, differentiable=False)]
    for logits, reference in zip(motion.precision_logits, expected):
        torch.testing.assert_close(logits.argmax(-1), torch.ones_like(reference),
                                   rtol=0, atol=0)
        torch.testing.assert_close(reference, torch.zeros_like(reference),
                                   rtol=0, atol=0)

    def direct_indices(step):
        raise AssertionError('Reference diagnostics must keep the original softmax path')

    motion._precision_indices = direct_indices
    choices, _, _ = motion._selected_symbols(5)
    for actual, reference in zip(choices, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    motion._hard_rate_bits(5)


def test_storage_aware_ste_is_exact_hard_forward_with_soft_gradients():
    _, motion = storage_aware_binary_fixture(coarse_region_gs=24)
    motion.prepare_step(5)
    with torch.no_grad():
        for logits in motion.precision_logits:
            logits.fill_(-100.)
            logits[:, 0] = 0.
            logits[:, 1] = float(np.log(4.))
        for logits in motion.refine_logits:
            logits.fill_(float(np.log(4.)))

    weights = motion._precision_weights(5)
    probabilities = motion._precision_probabilities(5)
    costs = torch.arange(len(motion.config.precision_bits), dtype=weights[0].dtype)
    for actual, probability in zip(weights, probabilities):
        expected = F.one_hot(probability.argmax(-1), probability.shape[-1]).to(actual)
        torch.testing.assert_close(actual.detach(), expected, rtol=0, atol=0)
    hard_loss = sum((value * costs).sum() for value in weights)
    soft_loss = sum((value * costs).sum() for value in probabilities)
    hard_grad = torch.autograd.grad(hard_loss, list(motion.precision_logits),
                                    retain_graph=True)
    soft_grad = torch.autograd.grad(soft_loss, list(motion.precision_logits))
    for actual, expected in zip(hard_grad, soft_grad):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    captured = []
    original_paths = motion._paths

    def capture_paths(gates):
        captured.extend(gates)
        return original_paths(gates)

    motion._paths = capture_paths
    motion(5)
    for value in captured:
        torch.testing.assert_close(value.detach(), torch.ones_like(value), rtol=0, atol=0)
    gate_grad = torch.autograd.grad(sum(value.sum() for value in captured),
                                   list(motion.refine_logits))
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in gate_grad)

    raw = torch.zeros(64, 6)
    raw[:, 0] = torch.linspace(.051, .09, len(raw))
    raw.requires_grad_()
    support, hard_support = motion._support_by_precision(raw)
    torch.testing.assert_close(support.detach(), hard_support.to(support), rtol=0, atol=0)
    support.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0


def gate_diagnostic_fixture(**overrides):
    values = dict(enabled=True, frames=[1], trace_steps=[5], probe_steps=[5],
                  camera_names=['train_a.png', 'train_b.png'], fit_steps=3,
                  fit_lr_scale=1., seed=37)
    values.update(overrides)
    config = GD.GateDiagnosticsConfig(**values)
    _, motion = storage_aware_binary_fixture()
    config.validate(motion.config)
    motion.prepare_step(5)
    with torch.no_grad():
        for logits in motion.refine_logits:
            logits.fill_(2.)
        motion.refine_logits[0][0] = -2.
        for translation in motion.region_translation:
            translation.fill_(.03)
    views = [SimpleNamespace(path=Path(name)) for name in config.camera_names]
    target = motion.base_xyz + .1

    def objective(view, xyz, rotation):
        scale = 1. if view.path.name == config.camera_names[0] else 2.
        return scale * (xyz - target).square().mean() + .01 * rotation.square().mean()

    return motion, config, views, objective


def assert_diagnostic_state_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_diagnostic_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_diagnostic_state_equal(a, b)
    else:
        assert left == right


def test_gate_diagnostic_default_off_and_100_frame_candidate_preserves_algorithm():
    import yaml
    assert not GD.GateDiagnosticsConfig().enabled
    GD.GateDiagnosticsConfig().validate()
    config = yaml.safe_load((legacy_config_root /
        'fs_gate_mechanism_diagnostics_100f.yaml').read_text())
    baseline = retained_experiment_config()
    assert config['data']['extra_dataset_kwargs']['num_frames'] == 100
    assert config['experiment_name'] != baseline['experiment_name']
    assert config['module']['storage_aware_binary'] == baseline['module']['storage_aware_binary']
    assert config['trainer']['num_incr_steps'] == 300
    diagnostics = GD.GateDiagnosticsConfig(**config['module']['gate_diagnostics'])
    diagnostics.validate(SAB.StorageAwareBinaryConfig(**config['module']['storage_aware_binary']))
    assert diagnostics.frames[-1] == 99 and diagnostics.precision_policies == ['fixed']
    for invalid in (dict(nodes_per_state=0), dict(fit_steps=True), dict(seed=-1),
                    dict(fit_lr_scale=float('nan')), dict(frames=[1, 1]),
                    dict(precision_policies=['learned']), dict(probe_steps=[301]),
                    dict(camera_names=[]), dict(fixed_precision_bits=9)):
        try:
            gate_diagnostic_fixture(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid diagnostic config accepted: {invalid}')


def test_gate_trace_separates_unreachable_closed_and_gate_motion_gradients():
    motion, config, _, _ = gate_diagnostic_fixture()
    xyz, rotation = motion(5)
    (xyz.sum() + rotation.sum()).backward()
    closed = GD.node_trace(motion, 5, 0, 0)
    assert closed['reachable'] and not closed['gate_open']
    assert closed['gate_gradient']['available'] and closed['gate_gradient']['l2'] > 0
    children = [row for row in closed['descendants'] if row['level'] == 1]
    assert children and all(not row['reachable'] for row in children)
    assert all(row['translation_gradient']['available'] and
               row['translation_gradient']['l2'] == 0 for row in children)
    samples = GD.sample_nodes(motion, 5, 1, config)
    assert samples == GD.sample_nodes(motion, 5, 1, config)
    assert {'reachable_closed', 'reachable_open', 'unreachable'}.issubset(
        {row['stratum'] for row in samples})
    reachable = GD.sample_nodes(motion, 5, 1, config, reachable_only=True)
    assert all(row['stratum'] != 'unreachable' for row in reachable)
    unreachable = next(row for row in samples if row['stratum'] == 'unreachable')
    try:
        GD.DiagnosticCandidate(motion, 5, unreachable['level'], unreachable['node'], True, config, 'fixed')
    except ValueError:
        pass
    else:
        raise AssertionError('An unreachable gate cannot support a local stop/refine conclusion')


def test_gate_candidates_freeze_ancestors_siblings_and_stop_descendants():
    motion, config, views, objective = gate_diagnostic_fixture(precision_policies=['fixed', 'learned'])
    for policy in config.precision_policies:
        for opened in (False, True):
            candidate = GD.DiagnosticCandidate(motion, 5, 0, 0, opened, config, policy)
            source_state = {name: value.clone() for name, value in motion.state_dict().items()}
            before_xyz, before_rotation = candidate()
            before_xyz, before_rotation = before_xyz.detach(), before_rotation.detach()
            mask = motion.index[0] == 0
            for _ in range(3):
                xyz, rotation = candidate()
                candidate.fit_step(objective(views[0], xyz, rotation))
            xyz, rotation = candidate()
            torch.testing.assert_close(xyz[~mask], before_xyz[~mask], rtol=0, atol=0)
            torch.testing.assert_close(rotation[~mask], before_rotation[~mask], rtol=0, atol=0)
            for level, frozen in enumerate(candidate.train_masks):
                for group in ('region_translation', 'region_rotation', 'precision_logits'):
                    actual = getattr(candidate, group)[level]
                    torch.testing.assert_close(actual[~frozen], source_state[f'{group}.{level}'][~frozen], rtol=0, atol=0)
            if not opened:
                assert all(not value.any() for value in candidate.train_masks[1:])
            assert candidate.region_translation[0].data_ptr() != motion.region_translation[0].data_ptr()
            assert candidate.base_xyz.data_ptr() != motion.base_xyz.data_ptr()
    # A deeper comparison also keeps ancestors unchanged.
    root = 1
    child = int(torch.where(motion.parents[0] == root)[0][0])
    candidate = GD.DiagnosticCandidate(motion, 5, 1, child, True, config, 'fixed')
    xyz, rotation = candidate()
    candidate.fit_step(objective(views[0], xyz, rotation))
    torch.testing.assert_close(candidate.region_translation[0], motion.region_translation[0], rtol=0, atol=0)
    torch.testing.assert_close(candidate.region_rotation[0], motion.region_rotation[0], rtol=0, atol=0)


def test_gate_actual_cost_covers_reachable_subtree_flags_selectors_and_frame_padding():
    motion, config, _, _ = gate_diagnostic_fixture()
    counts = []
    for opened in (False, True):
        candidate = GD.DiagnosticCandidate(motion, 5, 0, 0, opened, config, 'fixed')
        cost = GD.exact_cost(candidate, 5, birth_bytes=123, subtree=candidate.subtree_masks)
        payload, bits, _, _ = candidate._encode_payload(5)
        assert cost['payload_bits'] == bits == sum(cost['components'].values())
        assert cost['motion_bytes'] == SAB.CODEC_HEADER_BYTES + len(payload)
        assert cost['total_bytes'] == cost['motion_bytes'] + 123
        assert cost['padding_bits'] == (-bits) % 8
        assert bits + cost['padding_bits'] == len(payload) * 8
        if not opened:
            assert cost['subtree_components']['presence_bits'] == 1
            assert cost['subtree_components']['refine_flag_bits'] == 1
        else:
            assert cost['subtree_components']['presence_bits'] > 1
        counts.append(cost)
    assert counts[1]['total_bytes'] > counts[0]['total_bytes']
    local_delta = sum(counts[1]['subtree_components'].values()) - sum(counts[0]['subtree_components'].values())
    assert local_delta == counts[1]['payload_bits'] - counts[0]['payload_bits']
    # Stale production incremental-storage caches must not influence diagnostic bytes.
    candidate.config.cache_frozen_storage = True
    candidate._frozen_storage_parts[True] = {'wrong': candidate.base_xyz.new_tensor(1e9)}
    assert GD.exact_cost(candidate, 11)['motion_bytes'] == counts[1]['motion_bytes']


def test_gate_probe_preserves_source_state_gradients_rng_and_records_actual_gain():
    import random
    motion, config, views, objective = gate_diagnostic_fixture(precision_policies=['fixed', 'learned'])
    xyz, rotation = motion(5)
    objective(views[0], xyz, rotation).backward()
    motion.motion_optimizer.step()  # Include nonempty momentum in the isolation check.
    saved = copy.deepcopy(motion.capture())
    grads = [None if p.grad is None else p.grad.clone() for p in motion.parameters()]
    phase, last_step = motion.current_phase, motion.last_step
    torch.manual_seed(43)
    np.random.seed(43)
    random.seed(43)
    rng = torch.get_rng_state().clone()
    numpy_rng, python_rng = copy.deepcopy(np.random.get_state()), random.getstate()
    with GD.isolated_rng('cpu'):
        torch.rand(2)
        np.random.rand(2)
        random.random()
        record = GD.compare_node(motion, 5, 0, 0, views, objective, config,
                                  reference_bytes=1000., rate_weight=.01, birth_bytes=12)
    assert_diagnostic_state_equal(saved, motion.capture())
    assert phase == motion.current_phase and last_step == motion.last_step
    for parameter, gradient in zip(motion.parameters(), grads):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    assert random.getstate() == python_rng
    assert [row['precision_policy'] for row in record['comparisons']] == ['fixed', 'learned']
    for row in record['comparisons']:
        stop, refine = row['arms']['stop'], row['arms']['refine']
        expected = (stop['after']['image_loss'] - refine['after']['image_loss'] - .01 *
            (refine['after']['cost']['total_bytes'] - stop['after']['cost']['total_bytes']) / 1000.)
        assert row['gain'] == expected and row['supports_refine'] == (expected > 0)
        assert [r['camera'] for r in stop['fit_losses']] == [r['camera'] for r in refine['fit_losses']]
        assert stop['first_fit_gradient']['gate_gradient']['available'] is False
        assert stop['fit_seconds'] >= 0 and refine['fit_seconds'] >= 0
        if row['precision_policy'] == 'fixed':
            assert stop['first_fit_gradient']['descendants'][0]['precision_gradient']['available'] is False


def test_gate_probe_report_is_node_addressable_and_refuses_overwrite():
    from omegaconf import OmegaConf
    from libgs.utils.config import to_structured
    motion, config, views, objective = gate_diagnostic_fixture(fit_steps=1)
    # Reproduce the production libgs conversion, which passes ListConfig fields through.
    config, unused = to_structured(OmegaConf.create(asdict(config)), GD.GateDiagnosticsConfig)
    assert not unused
    with tempfile.TemporaryDirectory() as directory:
        path = GD.run_probes(motion, frame=1, step=5, context='online_pre_update',
            views=views, image_loss=objective, config=config, reference_bytes=1000.,
            rate_weight=.01, output_dir=directory)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows and all(row['frame'] == 1 and row['step'] == 5 for row in rows)
        assert all(row['context'] == 'online_pre_update' and row['stage'] == 'gate_compress' for row in rows)
        assert all(row['sample']['node'] == row['node'] for row in rows)
        assert path.with_suffix('.plan.json').is_file()
        plan = json.loads(path.with_suffix('.plan.json').read_text())
        assert plan['precision_policies'] == ['fixed']
        assert plan['camera_names'] == [view.path.name for view in views]
        assert json.loads(json.dumps(asdict(config)))['precision_policies'] == ['fixed']
        summary_module = importlib.import_module('scripts.archive.summarize_gate_diagnostics')
        summary = summary_module.summarize(Path(directory))
        assert len(summary['rows']) == len(rows)
        assert {row['context'] for row in summary['rows']} == {'online_pre_update'}
        assert all(row['frame'] == 1 and row['step'] == 5 for row in summary['aggregates'])
        try:
            GD.run_probes(motion, frame=1, step=5, context='online_pre_update',
                views=views, image_loss=objective, config=config, reference_bytes=1000.,
                rate_weight=.01, output_dir=directory)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Completed probe results must not be overwritten')


def test_gate_diagnostic_entry_loads_native_resolved_yaml_lists():
    import yaml
    from omegaconf import OmegaConf
    from libgs.utils.config import to_structured
    entry = importlib.import_module('scripts.archive.diagnose_binary_gates')
    path = legacy_config_root / 'fs_gate_mechanism_diagnostics_100f.yaml'
    values = OmegaConf.load(path).module.gate_diagnostics
    direct, unused = to_structured(values, GD.GateDiagnosticsConfig)
    assert not unused
    for name in ('frames', 'trace_steps', 'probe_steps', 'camera_names', 'precision_policies'):
        assert type(getattr(direct, name)) is list
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / 'diagnostic.yaml'
        fixture.write_text(yaml.safe_dump(OmegaConf.to_container(values, resolve=True)))
        loaded = entry.load_config(fixture, GD.GateDiagnosticsConfig)
        assert asdict(loaded) == asdict(direct)
        assert json.loads(json.dumps(dict(diagnostic=asdict(loaded))))['diagnostic']['fit_steps'] == 20
        fixture.write_text('enabled: false\nseed: 7\nframes: ["${seed}"]\n')
        resolved = entry.load_config(fixture, GD.GateDiagnosticsConfig)
        assert resolved.frames == [7] and type(resolved.frames[0]) is int


def test_gate_candidate_initial_pose_matches_source_at_all_phases_and_backends():
    config = GD.GateDiagnosticsConfig(fit_steps=1)
    for backend in ('reference', 'node'):
        _, motion = storage_aware_binary_fixture(motion_compute=backend,
            cache_static_layout=True, selected_precision_only=(backend == 'node'))
        with torch.no_grad():
            for values in motion.region_translation:
                values.fill_(.03)
            for values in motion.region_rotation:
                values[:, 0] = .04
        for step in (1, 3, 5, 9, 11):
            motion.prepare_step(step)
            opened = bool(motion.refinement_masks(step)[0][0])
            candidate = GD.DiagnosticCandidate(motion, step, 0, 0, opened, config, 'fixed')
            for actual, expected in zip(candidate(), motion(step)):
                torch.testing.assert_close(actual, expected,
                    rtol=0 if backend == 'reference' else 1e-6,
                    atol=0 if backend == 'reference' else 1e-6)


def test_gate_frozen_render_objective_keeps_source_attributes_and_pose_separate():
    from libgs.metric import psnr, ssim
    path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'gate_diagnostic_render.py'
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))

    class FakeRenderer:
        def __init__(self, config, model):
            self.gaussians = model

        def __call__(self, view, background):
            image = self.gaussians.get_xyz.mean(0)[:, None, None].expand(3, 12, 12)
            return dict(render=image)

    scope = dict(torch=torch, copy=copy, RAW_ATTRIBUTES=Q.RAW_ATTRIBUTES,
                 GaussianModel=GAUSS.GaussianModel, Renderer=FakeRenderer, ssim=ssim, psnr=psnr)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), scope)
    raw = residual_raw_fixture(4)
    model = GAUSS.GaussianModel(1)
    model.active_sh_degree = 1
    for name, attribute in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attribute, torch.nn.Parameter(raw[name].clone()))
    original = {attribute: getattr(model, attribute).clone() for attribute in Q.RAW_ATTRIBUTES}
    objective = scope['FrozenImageObjective'](model, SimpleNamespace(), torch.zeros(3), .2, 3)
    xyz = model._xyz[:3].detach().clone().requires_grad_()
    rotation = model._rotation[:3].detach().clone().requires_grad_()
    view = SimpleNamespace(image=torch.zeros(3, 12, 12))
    loss = objective(view, xyz, rotation)
    loss.backward()
    assert xyz.grad is not None and xyz.grad.abs().sum() > 0
    assert objective.model.rigid_xyz is None and objective.model.rigid_rotation is None
    assert objective.model.active_sh_degree == model.active_sh_degree
    for attribute in Q.RAW_ATTRIBUTES:
        assert getattr(model, attribute).grad is None
        torch.testing.assert_close(getattr(model, attribute), original[attribute], rtol=0, atol=0)
        assert not getattr(objective.model, attribute).requires_grad


def test_online_gate_hook_is_read_only_and_keeps_tracing_nodes_after_gate_changes():
    path = project_root / 'pipeline' / 'ReCon-GS' / 'archive' / 'legacy_module.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Module')
    definition = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'record_gate_diagnostics')
    scope = dict(__name__='pipeline.ReCon-GS.archive.legacy_module', __package__='pipeline.ReCon-GS.archive',
        torch=torch, time=time, isolated_rng=GD.isolated_rng,
        sample_nodes=GD.sample_nodes, node_trace=GD.node_trace, write_trace=GD.write_trace,
        run_probes=GD.run_probes, private_view_plan=GD.private_view_plan)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), scope)
    motion, config, views, objective = gate_diagnostic_fixture(fit_steps=1, trace_steps=[5, 6])
    fake_renderer_module = ModuleType('pipeline.ReCon-GS.archive.gate_diagnostic_render')
    fake_renderer_module.FrozenImageObjective = lambda *args: objective
    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            'pipeline.ReCon-GS.archive.gate_diagnostic_render': fake_renderer_module}):
        owner = SimpleNamespace(current_frame=1, global_step=5, device='cpu', residual_motion=motion,
            quant_first_frame_gs_count=48, gaussians=object(), background=torch.zeros(3),
            config=SimpleNamespace(motion_mode='storage_aware_binary', gate_diagnostics=config,
                                   renderer=SimpleNamespace(), lambda_dssim=.2),
            trainer=SimpleNamespace(is_first_frame=False, output_dir=Path(directory), synchronize=lambda: None),
            datamodule=SimpleNamespace(config=SimpleNamespace(training_camera_names=config.camera_names),
                selective_training_pairs=lambda *args: [(view, view) for view in views]))
        xyz, rotation = motion(5)
        objective(views[0], xyz, rotation).backward()
        original = copy.deepcopy(motion.capture())
        seconds = scope['record_gate_diagnostics'](owner, views[0])
        assert seconds > 0
        assert_diagnostic_state_equal(motion.capture(), original)
        first_nodes = set(owner._gate_trace_samples)
        with torch.no_grad():
            for logits in motion.refine_logits:
                logits.fill_(-2.)
        owner.global_step = 6
        scope['record_gate_diagnostics'](owner, views[0])
        assert first_nodes.issubset(owner._gate_trace_samples)
        rows = [json.loads(line) for line in
            (Path(directory) / 'gate_diagnostics' / 'frame_001_trace.jsonl').read_text().splitlines()]
        second_nodes = {(row['level'], row['node']) for row in rows if row['step'] == 6}
        assert first_nodes.issubset(second_nodes)
        config.enabled = False
        assert scope['record_gate_diagnostics'](owner, views[0]) == 0.


def joint_diagnostic_fixture():
    motion, config, views, objective = gate_diagnostic_fixture(fit_steps=2,
        candidate_mode='joint_precision', stop_parent_precision_bits=[8, 32],
        refine_parent_precision_bits=[8, 32], refine_descendant_precision_bits=[8, 32],
        evaluation_camera_names=['check_a.png'], view_sequence='seeded_balanced_cycle')
    for view in views:
        view.image = torch.zeros(3, 4, 4)
        view.full_proj_transform = torch.eye(4)
    checks = [SimpleNamespace(path=Path('check_a.png'), image=torch.zeros(3, 4, 4),
                              full_proj_transform=torch.eye(4))]
    return motion, config, views, checks, objective


def test_joint_precision_yaml_keeps_main_parameters_and_has_32_restricted_candidates():
    import yaml
    config = yaml.safe_load((legacy_config_root / 'fs_joint_precision_mechanism_100f.yaml').read_text())
    original = yaml.safe_load((legacy_config_root / 'fs_gate_mechanism_diagnostics_100f.yaml').read_text())
    assert config['data'] == original['data'] and config['trainer'] == original['trainer']
    assert config['module']['storage_aware_binary'] == original['module']['storage_aware_binary']
    diagnostics = GD.GateDiagnosticsConfig(**config['module']['gate_diagnostics'])
    diagnostics.validate(SAB.StorageAwareBinaryConfig(**config['module']['storage_aware_binary']))
    specs = GD.candidate_specs(diagnostics)
    assert len(specs) == 32 and len({item['name'] for item in specs}) == 32
    assert sum(item['opened'] and item['parent_bits'] is not None for item in specs) == 25
    assert diagnostics.evaluation_camera_names == ['cam18.png', 'cam19.png', 'cam20.png']
    assert diagnostics.fit_steps == 20 and diagnostics.save_probe_snapshots and diagnostics.record_component_gradients
    for name in ('stop_parent_precision_bits', 'refine_parent_precision_bits', 'refine_descendant_precision_bits'):
        assert getattr(diagnostics, name) == config['module']['storage_aware_binary']['precision_bits']


def test_joint_parent_and_descendant_precision_assignments_are_distinct_and_local():
    motion, config, _, _, _ = joint_diagnostic_fixture()
    candidate = GD.DiagnosticCandidate(motion, 5, 0, 0, True, config, 'fixed', parent_bits=32, descendant_bits=8)
    assert candidate.diagnostic_precision[0][0] == 4
    assert all(bool((choice[mask] == 0).all()) for choice, mask in
               zip(candidate.diagnostic_precision[1:], candidate.train_masks[1:]))
    source_precision = motion._selected_symbols(5)[0]
    for choice, original, mask in zip(candidate.diagnostic_precision, source_precision, candidate.subtree_masks):
        torch.testing.assert_close(choice[~mask], original[~mask], rtol=0, atol=0)
    gates = motion.refinement_masks(5)
    torch.testing.assert_close(candidate.diagnostic_gates[0][1:], gates[0][1:], rtol=0, atol=0)
    for actual, original in zip(candidate.diagnostic_gates[1:], gates[1:]):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    stop = GD.DiagnosticCandidate(motion, 5, 0, 0, False, config, 'fixed', parent_bits=8)
    assert stop.diagnostic_precision[0][0] == 0
    assert all(not value.any() for value in stop.train_masks[1:])
    for actual, original in zip(stop.diagnostic_precision[1:], source_precision[1:]):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)


def test_joint_candidate_fits_share_private_views_and_check_views_do_not_fit():
    motion, config, views, checks, objective = joint_diagnostic_fixture()
    saved = copy.deepcopy(motion.capture())
    plan = GD.private_view_plan(config, 1, 5, 0, 0)
    assert plan == GD.private_view_plan(config, 1, 5, 0, 0)
    record = GD.compare_node(motion, 5, 0, 0, views, objective, config,
        reference_bytes=1152., rate_weight=.01, evaluation_views=checks, view_plan=plan)
    assert len(record['candidates']) == 8 and record['schema'] == 'joint_precision_candidates_v1'
    expected_order = [views[index].path.name for index in plan['indices']]
    for candidate in record['candidates']:
        assert [row['camera'] for row in candidate['fit_losses']] == expected_order
        assert candidate['after']['evaluation_views'][0]['role'] == 'nonfitting_training_view'
        assert candidate['after']['evaluation_views'][0]['camera'] == 'check_a.png'
        assert all(row['camera'] != 'check_a.png' for row in candidate['fit_losses'])
        delta = candidate['after']['cost']['total_bytes'] - record['baseline']['cost']['total_bytes']
        image_gain = record['baseline']['image_loss'] - candidate['after']['image_loss']
        assert candidate['gain'] == image_gain - .01 * delta / 1152.
        assert len(candidate['per_view_gains']) == 3
    assert_diagnostic_state_equal(motion.capture(), saved)


def test_crossfold_feedback_protocol_and_summary_keep_check_views_held_out():
    module = importlib.import_module('scripts.archive.replay_joint_feedback_crossfold')
    protocol = module.load_protocol(
        legacy_config_root / 'fs_joint_feedback_crossfold_3nodes.yaml')
    assert len(protocol['nodes']) == 3 and protocol['expected_candidate_count'] == 32
    for fold in protocol['folds']:
        assert len(fold['fit_camera_names']) == len(fold['check_camera_names']) == 10
        assert not (set(fold['fit_camera_names']) & set(fold['check_camera_names']))
        assert 'cam00.png' not in fold['fit_camera_names'] + fold['check_camera_names']

    def candidate(name, fit_gain, check_gain):
        parent = None if name in {'stop_original', 'refine_original'} else 12
        return dict(spec=dict(name=name, parent_bits=parent),
                    gain=fit_gain, evaluation_gain=check_gain)

    # check favors refine_parent8_descendant8, but fitting selected parent16.
    record = dict(
        selected_by_fitting_rate_score='refine_parent16_descendant16',
        selected_by_check_rate_score='refine_parent8_descendant8',
        best_stop='stop_parent12', best_refine='refine_parent16_descendant16',
        candidates=[
            candidate('stop_original', .1, .3),
            candidate('refine_original', .2, .1),
            candidate('stop_parent12', .4, .25),
            candidate('refine_parent16_descendant16', .5, .35),
            candidate('refine_parent8_descendant8', .3, .9),
        ])
    summary = module.summarize_fold(record)
    assert summary['fit_selected']['winner'] == 'refine_parent16_descendant16'
    assert summary['selection_analysis_only']['check_selected_winner'] == 'refine_parent8_descendant8'
    gains = summary['heldout_rate_distortion_gains']
    assert gains['fit_winner_vs_source'] == .35
    assert gains['fit_original_vs_source'] == .1  # original chosen by fitting, not by check.
    torch.testing.assert_close(torch.tensor(gains['fit_winner_vs_fit_original']),
                               torch.tensor(.25), rtol=0, atol=1e-7)
    torch.testing.assert_close(torch.tensor(gains['fit_refine_minus_fit_stop']),
                               torch.tensor(.1), rtol=0, atol=1e-7)
    assert gains['fit_joint_vs_fit_original'] == gains['fit_winner_vs_fit_original']


def test_crossfold_balanced_plans_use_each_fit_view_twice_and_swap_roles():
    module = importlib.import_module('scripts.archive.replay_joint_feedback_crossfold')
    protocol = module.load_protocol(
        legacy_config_root / 'fs_joint_feedback_crossfold_3nodes.yaml')
    base = asdict(joint_diagnostic_fixture()[1])
    base.update(stop_parent_precision_bits=[8, 12, 16, 24, 32],
                refine_parent_precision_bits=[8, 12, 16, 24, 32],
                refine_descendant_precision_bits=[8, 12, 16, 24, 32])
    motion = storage_aware_binary_fixture()[1]
    packet = dict(diagnostic_config=base, motion=dict(config=asdict(motion.config)))
    for fold in protocol['folds']:
        config = module.configure_fold(packet, fold, protocol, GD)
        plan = GD.private_view_plan(config, 5, 120, 1, 2552)
        counts = [plan['indices'].count(index) for index in range(10)]
        assert counts == [2] * 10
        assert set(config.camera_names).isdisjoint(config.evaluation_camera_names)
    assert set(protocol['folds'][0]['fit_camera_names']) == set(
        protocol['folds'][1]['check_camera_names'])


def test_joint_feedback_selects_same_candidate_across_folds_with_source_fallback():
    module = importlib.import_module('scripts.archive.prototype_joint_feedback_transaction')

    def fold(values):
        return dict(candidates=[dict(spec=dict(name=name), evaluation_gain=gain)
                                for name, gain in values.items()])

    # Per-fold argmax differs, while candidate b is positive in both directions.
    selected = module.select_crossfit_candidate([
        fold(dict(a=3., b=2., c=-1.)), fold(dict(a=-2., b=1., c=4.))])
    assert selected['selected']['name'] == 'b'
    assert selected['selected']['directional_gains'] == [2., 1.]
    rejected = module.select_crossfit_candidate([
        fold(dict(a=3., b=-2.)), fold(dict(a=-1., b=4.))])
    assert rejected['selected']['name'] == 'source'


def test_joint_feedback_transaction_learns_hard_choices_and_rolls_back_exact_state():
    module = importlib.import_module('scripts.archive.prototype_joint_feedback_transaction')
    motion, config, _, _, _ = joint_diagnostic_fixture()
    config.stop_parent_precision_bits = [8, 32]
    config.refine_parent_precision_bits = [8, 32]
    config.refine_descendant_precision_bits = [8, 32]
    candidate = GD.DiagnosticCandidate(motion, 5, 0, 0, True, config, 'fixed',
                                       parent_bits=8, descendant_bits=8)
    deployment = GD.clone_motion(motion)
    for parameter in deployment.refine_logits:
        parameter.grad = torch.arange(len(parameter), dtype=parameter.dtype) + 1
    for parameter in deployment.precision_logits:
        parameter.grad = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter) + 1
    with torch.no_grad():
        candidate.region_translation[0][candidate.train_masks[0], 0] += .2
    transaction = module.capture_transaction(deployment)
    before_unrelated = deployment.region_translation[0][~candidate.train_masks[0]].clone()
    changes, _ = module.apply_candidate_motion(deployment, candidate)
    protocol = dict(feedback_lr=.02, feedback_betas=[.9, .999], feedback_eps=1e-15,
                    max_feedback_logit_steps=128)
    feedback, _ = module.optimize_feedback_logits(deployment, candidate, 5, 0, 0, protocol)
    assert changes and feedback['final']['matches'] and feedback['steps'] <= 128
    torch.testing.assert_close(deployment.region_translation[0][~candidate.train_masks[0]],
                               before_unrelated, rtol=0, atol=0)
    gate_position = deployment._refine_position(0, 0)
    gate_expected = transaction['gradients']['refine_logits.0']
    gate_actual = deployment.refine_logits[0].grad
    torch.testing.assert_close(gate_actual[:gate_position], gate_expected[:gate_position], rtol=0, atol=0)
    torch.testing.assert_close(gate_actual[gate_position + 1:], gate_expected[gate_position + 1:], rtol=0, atol=0)
    assert gate_actual[gate_position] == 0
    for depth, mask in enumerate(candidate.train_masks):
        actual = deployment.precision_logits[depth].grad
        expected = transaction['gradients'][f'precision_logits.{depth}']
        torch.testing.assert_close(actual[~mask], expected[~mask], rtol=0, atol=0)
        assert not actual[mask].any()
    assert not module.accept_or_restore(deployment, transaction, -1., True)
    assert module.transaction_matches(deployment, transaction)
    assert_diagnostic_state_equal(deployment.capture(), transaction['motion'])
    for name, parameter in deployment.named_parameters():
        expected = transaction['gradients'][name]
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


def test_joint_feedback_continuation_protocol_balances_future_views():
    module = importlib.import_module('scripts.archive.continue_joint_feedback_pair')
    protocol = module.load_protocol(
        legacy_config_root / 'fs_joint_feedback_continuation_pair.yaml')
    assert [(row['frame'], row['step']) for row in protocol['nodes']] == [(5, 120), (99, 180)]
    for frame, start, current, cycles in ((5, 120, 'cam06.png', 4),
                                          (99, 180, 'cam13.png', 1)):
        schedule = module.balanced_continuation_schedule(
            protocol['ordinary_camera_names'], 0, frame, start, 200, current)
        assert schedule[0] == dict(step=start, camera=current,
                                   role='recomputed_snapshot_step')
        assert schedule[-1]['step'] == 200 and len(schedule) == 201 - start
        counts = {name: sum(row['camera'] == name for row in schedule[1:])
                  for name in protocol['ordinary_camera_names']}
        assert set(counts.values()) == {cycles}
        assert schedule == module.balanced_continuation_schedule(
            protocol['ordinary_camera_names'], 0, frame, start, 200, current)


def test_joint_feedback_continuation_recomputes_and_hardens_with_native_stage():
    module = importlib.import_module('scripts.archive.continue_joint_feedback_pair')
    raw, motion = storage_aware_binary_fixture()
    # Simulate a saved post-backward/pre-update step-8 gate-compress snapshot.
    xyz, rotation = motion(8, training=True)
    (xyz.square().mean() + rotation.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in motion.parameters())
    cleared = module.clear_saved_gradients(motion)
    assert cleared and all(parameter.grad is None for parameter in motion.parameters())

    class Objective:
        def __call__(self, view, xyz, rotation):
            return (xyz - raw['xyz']).square().mean() + (rotation - raw['rotation']).square().mean()

    view = SimpleNamespace(path=Path('cam01.png'))
    first = module.training_update(motion, 8, view, Objective(), 1152.)
    assert first['phase'] == 'gate_compress' and not first['hardened_this_step']
    assert first['storage_proxy_differentiable']
    second = module.training_update(motion, 9, view, Objective(), 1152.)
    assert second['phase'] == 'gate_refine' and second['hardened_this_step']
    assert second['hard_masks_frozen'] and second['gate_precision_storage_frozen']
    # Main training keeps the hard gate/precision choices fixed, while its
    # symbol-support STE can still send storage gradients to motion rows.
    assert second['storage_proxy_differentiable']
    assert motion.hard_masks is not None and motion.last_step == 9
    expected_lr = motion.config.region_translation_lr * motion.config.refine_lr_scale
    assert second['motion_optimizer_lrs'][0] == expected_lr
    torch.testing.assert_close(torch.tensor(motion.motion_optimizer.param_groups[0]['lr']),
                               torch.tensor(expected_lr), rtol=0, atol=0)


def test_joint_feedback_continuation_source_fallback_is_complete_noop_pair():
    module = importlib.import_module('scripts.archive.continue_joint_feedback_pair')
    class Objective:
        def __call__(self, view, xyz, rotation):
            return (xyz - raw['xyz']).square().mean() + (rotation - raw['rotation']).square().mean()

    view = SimpleNamespace(path=Path('cam01.png'))
    analyzer = importlib.import_module('scripts.archive.analyze_joint_feedback_continuation')
    for status, with_feedback in (('source_retained', False), ('rolled_back', True)):
        raw, source = storage_aware_binary_fixture()
        # Populate optimizer state before cloning so equality covers Adam moments and steps.
        xyz, rotation = source(8, training=True)
        ((xyz - raw['xyz']).square().mean() + (rotation - raw['rotation']).square().mean()
         + .01 * sum(source.storage_parts(8).values()) / 1152.).backward()
        source.optimizer_step()
        record = dict(status=status, accepted=False)
        if with_feedback:
            record['feedback'] = dict(changed_rows=dict(
                precision=[dict(level=1, rows=[7, 8])]))
        feedback, mode = module.initialize_feedback_motion(
            source, record, Path('/does/not/exist.pth'), 'cpu', SAB)
        assert mode == 'source_noop_after_' + status
        assert module._exact_equal(module._motion_runtime_state(source),
                                   module._motion_runtime_state(feedback))
        for step in (8, 9, 10):
            feedback, rows = module.paired_training_update(
                source, feedback, mode, step, view, Objective(), 1152., SAB)
            assert rows['source'] == rows['feedback']
            assert module._exact_equal(module._motion_runtime_state(source),
                                       module._motion_runtime_state(feedback))
        zero = dict(feedback_minus_source_image_loss=0., feedback_minus_source_psnr=0.,
                    feedback_minus_source_bytes=0, feedback_rd_gain=0.)
        synthetic = dict(feedback_mode=mode, deployed_packet=None,
            deployed_packet_sha256=None, no_op_final_state_equal=True,
            evaluations=[dict(comparison=zero)], final_validation=dict(
                comparison=dict(feedback_minus_source_image_loss=0.,
                                feedback_minus_source_psnr=0.)))
        assert analyzer.noop_consistency_issues(synthetic, 1) == []
    assert module.tracked_precision_rows({}, 1, 7) == [dict(level=1, rows=[7])]
    feedback_record = dict(feedback=dict(changed_rows=dict(
        precision=[dict(level=1, rows=[7, 8])])))
    assert module.tracked_precision_rows(feedback_record, 1, 7) == [
        dict(level=1, rows=[7, 8])]


def test_joint_feedback_continuation_analyzes_source_and_rollback_noops():
    analyzer = importlib.import_module('scripts.archive.analyze_joint_feedback_continuation')
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory)
        snapshots = run / 'source_snapshot'
        snapshots.mkdir()
        live_source = run / 'live_source.py'
        copied_source = snapshots / '00_live_source.py'
        live_source.write_text('synthetic continuation source\n', encoding='utf-8')
        copied_source.write_bytes(live_source.read_bytes())
        code_hash = analyzer.digest(live_source)
        (snapshots / 'manifest.json').write_text(json.dumps([dict(
            source=str(live_source), snapshot=str(copied_source),
            sha256=code_hash, bytes=live_source.stat().st_size)]), encoding='utf-8')

        motion_packet = run / 'motion.pth'
        torch.save(dict(motion=dict(config=dict(
            region_translation_lr=.03, region_rotation_lr=.04,
            refine_lr_scale=.3))), motion_packet)
        packet_hash = analyzer.digest(motion_packet)
        cameras = ['cam01.png']
        schedule = [dict(step=180, camera='cam01.png', role='recomputed_snapshot_step'),
                    dict(step=181, camera='cam01.png', role='seeded_balanced_future')]

        def state(step):
            return dict(level=0, node=1, position=0, phase='gate_refine',
                hard_masks_frozen=step >= 181, gate_open=False, gate_logit=-2.,
                gate_hard_margin=2., precision=[dict(
                    level=0, row=1, choice=0, bits=8, logit_margin=1.)])

        def update(step):
            return dict(step=step, camera='cam01.png', phase='gate_refine',
                image_loss=.1, storage_proxy_bytes=1., storage_weight=.01,
                storage_loss=.001, total_loss=.101,
                storage_proxy_differentiable=True,
                gate_precision_storage_frozen=True,
                motion_optimizer_lrs=[.009, .012], expected_refine_lr_scale=.3,
                hardened_this_step=step == 181, hard_masks_frozen=step >= 181,
                schedule_role=schedule[step - 180]['role'], target_state=state(step))

        cost = dict(components=dict(motion=8), payload_bits=8,
                    header_bytes=2, total_bytes=3)

        def evaluation(step, timing):
            arm = dict(step=step, image_loss=.1, psnr=20.,
                views=[dict(camera='cam01.png', image_loss=.1, psnr=20.)],
                cost=cost, rate_weight=.01, actual_rd_score=.101)
            comparison = dict(step=step, feedback_minus_source_image_loss=0.,
                feedback_minus_source_psnr=0., feedback_minus_source_bytes=0,
                feedback_rd_gain=0.)
            return dict(timing=timing, arms=dict(
                source=copy.deepcopy(arm), feedback=copy.deepcopy(arm)),
                comparison=comparison)

        nodes = []
        for frame, status in ((1, 'source_retained'), (2, 'rolled_back')):
            updates = [update(180), update(181)]
            nodes.append(dict(node=dict(frame=frame, step=180, level=0, node=1),
                source_snapshot=str(motion_packet), source_snapshot_sha256=packet_hash,
                transaction_status=status, selected='source',
                feedback_mode='source_noop_after_' + status,
                deployed_packet=None, deployed_packet_sha256=None,
                no_op_final_state_equal=True,
                actual_compute_budget=dict(training_updates=2,
                    all20_evaluation_sets=3, final_validation_renders=1,
                    no_op_feedback_reuses_source_compute=True),
                snapshot_context='synthetic', snapshot_training_camera='cam01.png',
                semantics='synthetic no-op mirror', cleared_saved_gradient_norms={},
                schedule=schedule, future_camera_counts={'cam01.png': 1},
                reference_bytes=100., dependencies={},
                target_rows=dict(gate=dict(level=0, node=1),
                                 precision=[dict(level=0, rows=[1])]),
                objective=dict(image='synthetic', storage_loss_weight=.01,
                    gs_activity_loss_weight=0.,
                    gs_activity_omitted_because_source_weight_is_zero=True,
                    keep_candidate_omitted_because_sparse_motion_path_disables_it=True),
                initial_target_state=dict(source=state(180), feedback=state(180)),
                updates=dict(source=copy.deepcopy(updates), feedback=copy.deepcopy(updates)),
                evaluations=[evaluation(180, 'pre_update'),
                             evaluation(180, 'post_update'),
                             evaluation(181, 'post_update')],
                final_validation=dict(source=dict(camera='cam00.png', image_loss=.2, psnr=15.),
                    feedback=dict(camera='cam00.png', image_loss=.2, psnr=15.),
                    comparison=dict(feedback_minus_source_image_loss=0.,
                                    feedback_minus_source_psnr=0.)),
                hardening=dict(source_steps=[181], feedback_steps=[181]),
                immutable_sources_verified=True))
        summary = dict(schema='joint_feedback_controlled_continuation_v1',
            protocol=dict(expected_nodes=2, end_step=181,
                ordinary_camera_names=cameras, evaluation_steps=[180, 181],
                validation_camera_name='cam00.png'), nodes=nodes)
        (run / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        result = analyzer.analyze(run)
        assert result['issues'] == [] and result['completed_nodes'] == 2
        assert result['actual_compute_budget'] == dict(
            training_updates=4, all20_evaluation_sets=6,
            all20_render_count=6, final_validation_renders=2)


def test_independent_image_rate_gradients_explain_closed_motion_without_contamination():
    motion, config, views, _, objective = joint_diagnostic_fixture()
    with torch.no_grad():
        for logits in motion.precision_logits:
            logits.fill_(-100.)
            logits[:, 0] = 0.  # Nonzero smooth support derivative despite zero coded motion.
    xyz, rotation = motion(5)
    image = objective(views[0], xyz, rotation)
    rate = .01 * sum(motion.storage_parts(5).values()) / 1152.
    (image + rate).backward()
    saved = copy.deepcopy(motion.capture())
    gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                 for name, parameter in motion.named_parameters()}
    components = GD.collect_gradient_components(motion, 5, views[0], objective,
                                                 reference_bytes=1152., rate_weight=.01)
    children = torch.where(motion.parents[0] == 0)[0]
    assert bool((components['image']['region_translation.1'][children] == 0).all())
    assert components['rate_proxy']['region_translation.1'][children].abs().sum() > 0
    for name, gradient in gradients.items():
        parameter = dict(motion.named_parameters())[name]
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
            a, b = components['image'][name], components['rate_proxy'][name]
            expected = (torch.zeros_like(gradient) if a is None else a) + (torch.zeros_like(gradient) if b is None else b)
            torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-8)
    assert_diagnostic_state_equal(motion.capture(), saved)
    row = GD.node_trace(motion, 5, 0, 0, components)
    assert row['gate_gradient_components']['image']['available']
    assert row['descendants'][1]['gradient_components']['translation']['image']['l2'] == 0.


def test_stage_snapshot_restores_complete_source_and_replays_identical_candidates():
    motion, config, views, checks, objective = joint_diagnostic_fixture()
    xyz, rotation = motion(5)
    objective(views[0], xyz, rotation).backward()
    motion.motion_optimizer.step()  # Nonempty Adam state must also be isolated in the snapshot.
    motion.motion_optimizer.zero_grad(set_to_none=True)
    xyz, rotation = motion(5)
    objective(views[0], xyz, rotation).backward()
    samples = [dict(level=0, node=0, stratum='reachable_closed', population=1)]
    plans = {'0:0': GD.private_view_plan(config, 1, 5, 0, 0)}
    saved = copy.deepcopy(motion.capture())
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        snapshot = GPS.save_probe_snapshot(directory / 'stage.pth', motion=motion, frame=1, step=5,
            diagnostic_config=config, render_state=dict(old_count=48, attributes=residual_raw_fixture(48)),
            data_config=dict(root='fixture'), module_config=dict(lambda_dssim=.2), renderer_config={},
            background=torch.zeros(3), first_frame_gs_count=48, samples=samples, view_plans=plans,
            views=views + checks, training_camera=views[0].path.name, source_rng=GPS.capture_rng())
        packet = torch.load(snapshot, map_location='cpu', weights_only=False)
        restored = GPS.restore_probe_motion(packet, 'cpu')
        assert_diagnostic_state_equal(saved, restored.capture())
        for (name, parameter), (_, original) in zip(restored.named_parameters(), motion.named_parameters()):
            assert parameter.requires_grad == original.requires_grad
            if original.grad is not None:
                torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0)
        GPS.verify_snapshot_views(packet, views + checks)
        a = GD.compare_node(motion, 5, 0, 0, views, objective, config, reference_bytes=1152.,
                            rate_weight=.01, evaluation_views=checks, view_plan=plans['0:0'])
        b = GD.compare_node(restored, 5, 0, 0, views, objective, config, reference_bytes=1152.,
                            rate_weight=.01, evaluation_views=checks, view_plan=packet['view_plans']['0:0'])
        for left, right in zip(a['candidates'], b['candidates']):
            assert left['spec'] == right['spec'] and left['gain'] == right['gain']
            assert left['before'] == right['before'] and left['after'] == right['after']
            assert left['fit_losses'] == right['fit_losses']
        motion.motion_optimizer.step()
        assert_diagnostic_state_equal(saved, packet['motion'])
        views[0].image.add_(1.)
        try:
            GPS.verify_snapshot_views(packet, views + checks)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed replay image dependencies must be rejected')


def test_known_static_rigid_and_local_motion_are_representable_with_assigned_gates():
    _, motion = storage_aware_binary_fixture(coarse_region_gs=1000)
    motion.prepare_step(5)
    with torch.no_grad():
        for gates in motion.refine_logits:
            gates.fill_(-100.)
    xyz, rotation = motion(5)
    torch.testing.assert_close(xyz, motion.base_xyz, rtol=0, atol=0)
    torch.testing.assert_close(rotation, motion.base_rotation, rtol=0, atol=0)
    with torch.no_grad():
        motion.region_translation[0][0, 0] = .1
        motion.region_rotation[0][0, 2] = .1
    xyz, rotation = motion(5)
    theta = torch.tensor(.1)
    matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.],
                           [torch.sin(theta), torch.cos(theta), 0.], [0., 0., 1.]])
    pivot = motion.active_layout['pivots'][0][0]
    expected = (motion.base_xyz - pivot) @ matrix.T + pivot + torch.tensor([.1, 0., 0.])
    torch.testing.assert_close(xyz, expected, rtol=1e-6, atol=1e-5)
    expected_q = torch.tensor([torch.cos(theta / 2), 0., 0., torch.sin(theta / 2)])
    torch.testing.assert_close(rotation, expected_q.expand_as(rotation), rtol=1e-6, atol=1e-6)
    with torch.no_grad():
        motion.region_translation[0].zero_()
        motion.region_rotation[0].zero_()
        motion.refine_logits[0].fill_(100.)
        motion.region_translation[1][0, 0] = .1
        motion.region_translation[1][1, 0] = -.1
    xyz, rotation = motion(5)
    expected = motion.base_xyz.clone()
    expected[:, 0] += torch.where(motion.index[1] == 0, .1, -.1)
    torch.testing.assert_close(xyz, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(rotation, motion.base_rotation, rtol=0, atol=0)


def test_full_online_joint_hook_saves_snapshot_components_and_joint_summary():
    from dataclasses import dataclass, field
    from typing import Optional
    path = project_root / 'pipeline/ReCon-GS/archive/legacy_module.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Module')
    definition = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'record_gate_diagnostics')
    scope = dict(__name__='pipeline.ReCon-GS.archive.legacy_module', __package__='pipeline.ReCon-GS.archive',
        torch=torch, time=time, isolated_rng=GD.isolated_rng, capture_rng=GPS.capture_rng,
        sample_nodes=GD.sample_nodes, node_trace=GD.node_trace, write_trace=GD.write_trace,
        run_probes=GD.run_probes, private_view_plan=GD.private_view_plan,
        collect_gradient_components=GD.collect_gradient_components, save_probe_snapshot=GPS.save_probe_snapshot)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), scope)
    motion, config, views, checks, objective = joint_diagnostic_fixture()
    config.save_probe_snapshots = True
    config.record_component_gradients = True

    @dataclass
    class RenderConfig:
        debug: bool = False

    @dataclass
    class ModelConfig:
        gate_diagnostics: object = field(default_factory=lambda: config)
        motion_mode: str = 'storage_aware_binary'
        renderer: object = field(default_factory=RenderConfig)
        lambda_dssim: float = .2

    @dataclass
    class DataConfig:
        root: str = 'fixture'
        training_camera_names: object = None

    class Objective:
        def __call__(self, view, xyz, rotation):
            return objective(view, xyz, rotation)

        def snapshot_state(self):
            return dict(old_count=len(motion.base_xyz), attributes=residual_raw_fixture(48))

    fake = ModuleType('pipeline.ReCon-GS.archive.gate_diagnostic_render')
    fake.FrozenImageObjective = lambda *args: Objective()
    all_views = {view.path.name: view for view in views + checks}
    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {
            'pipeline.ReCon-GS.archive.gate_diagnostic_render': fake}):
        directory = Path(directory)
        owner = SimpleNamespace(current_frame=1, global_step=5, device='cpu', residual_motion=motion,
            quant_first_frame_gs_count=48, gaussians=object(), background=torch.zeros(3), config=ModelConfig(),
            trainer=SimpleNamespace(is_first_frame=False, output_dir=directory, synchronize=lambda: None),
            datamodule=SimpleNamespace(config=DataConfig(training_camera_names=list(all_views)),
                selective_training_pairs=lambda names, *args: [(all_views[name], all_views[name]) for name in names]))
        xyz, rotation = motion(5)
        (objective(views[0], xyz, rotation) + .01 * sum(motion.storage_parts(5).values()) / 1152.).backward()
        original = copy.deepcopy(motion.capture())
        gradients = {name: None if parameter.grad is None else parameter.grad.clone()
                     for name, parameter in motion.named_parameters()}
        rng = torch.get_rng_state().clone()
        scope['record_gate_diagnostics'](owner, views[0])
        assert_diagnostic_state_equal(motion.capture(), original)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for name, parameter in motion.named_parameters():
            if gradients[name] is not None:
                torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
        snapshot = directory / 'gate_diagnostics/snapshots/frame_001_step_0005.pth'
        packet = torch.load(snapshot, map_location='cpu', weights_only=False)
        assert packet['gradient_components']['context']['camera'] == views[0].path.name
        assert len(packet['views']) == 3 and packet['purpose'] == 'training_probe'
        summary = importlib.import_module('scripts.archive.summarize_gate_diagnostics').summarize(directory)
        assert len(summary['joint_nodes']) == len(packet['samples'])
        assert len(summary['joint_candidates']) == 8 * len(packet['samples'])
        assert not summary['rows']
        assert all(row['after']['evaluation_views'][0]['role'] == 'nonfitting_training_view'
                   for row in summary['joint_candidates'])


def shadow_explorer_fixture(motion, **overrides):
    values = dict(
        start_step=5, end_step=5, interval=1, max_backward_per_frame=1,
        levels=[0, 1, 2], seed=0, camera_policy='seeded_balanced_cycle',
        frontier_policy='reachable_closed_same_level_all',
        descendant_policy='preserve_source_hard',
        precision_policy='current_argmax_detached', loss='image_only',
        translation_lr=motion.config.region_translation_lr,
        rotation_lr=motion.config.region_rotation_lr,
        adam_betas=list(motion.config.adam_betas), adam_eps=motion.config.adam_eps,
        fraction=1.)
    values.update(overrides)
    config = SHADOW.ShadowExplorationConfig(**values)
    return SHADOW.ShadowMotionExplorer(motion, config)


def test_shadow_frontier_preserves_deeper_gates_and_targets_occupied_new_rows():
    _, motion = storage_aware_binary_fixture()
    motion.prepare_step(5)
    with torch.no_grad():
        motion.refine_logits[0].fill_(-5.)
        for logits in motion.refine_logits[1:]:
            logits.fill_(5.)
    plan = motion.shadow_frontier_plan(5, 0)
    occupancy = motion._occupancy_values()
    assert plan['step'] == 5 and len(plan['gate_nodes']) > 0
    for level in range(1, len(plan['source_gates'])):
        torch.testing.assert_close(plan['shadow_gates'][level],
                                   plan['source_gates'][level], rtol=0, atol=0)
    for actual, source, shadow, occupied in zip(
            plan['target_rows'], plan['source_paths'], plan['shadow_paths'], occupancy):
        expected = shadow.bool() & ~source.bool() & occupied.bool()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not bool(plan['target_rows'][0].any())
    assert bool(plan['target_rows'][1].any()) and bool(plan['target_rows'][2].any())
    assert all(bool(occupancy[level][rows].all())
               for level, rows in enumerate(plan['target_rows']) if bool(rows.any()))


def test_shadow_step_isolates_rows_optimizers_and_main_hard_pose():
    _, motion = storage_aware_binary_fixture()
    motion.prepare_step(5)
    with torch.no_grad():
        motion.refine_logits[0].fill_(-5.)
        for logits in motion.refine_logits[1:]:
            logits.fill_(5.)
        for value in motion.region_translation:
            value.uniform_(-.03, .03)
    explorer = shadow_explorer_fixture(motion)
    plan = explorer.plan(5)
    before_parameters = {name: value.detach().clone()
                         for name, value in motion.named_parameters()}
    before_main_optimizers = (copy.deepcopy(motion.motion_optimizer.state_dict()),
                              copy.deepcopy(motion.gate_optimizer.state_dict()))
    with torch.no_grad():
        before_pose = tuple(value.clone() for value in motion(5))
    xyz, rotation = explorer.pose(5, plan)
    weights = torch.linspace(.1, 1., xyz.shape[0], device=xyz.device)[:, None]
    loss = (xyz * weights).sum() + .01 * (rotation * weights).sum()
    gradients = explorer.backward_and_step(5, loss, plan)
    assert any(row['active_rows'] and row['gradients']['translation'] > 0
               for row in gradients)
    for name, value in motion.named_parameters():
        old = before_parameters[name]
        if name.startswith('region_translation.') or name.startswith('region_rotation.'):
            level = int(name.split('.')[1])
            changed = ((value.detach() != old).flatten(1).any(-1)
                       if value.shape[0] else torch.zeros(0, dtype=torch.bool))
            assert not bool((changed & ~plan['target_rows'][level]).any())
        else:
            torch.testing.assert_close(value, old, rtol=0, atol=0)
    assert_diagnostic_state_equal(motion.motion_optimizer.state_dict(),
                                  before_main_optimizers[0])
    assert_diagnostic_state_equal(motion.gate_optimizer.state_dict(),
                                  before_main_optimizers[1])
    with torch.no_grad():
        after_pose = motion(5)
    for before, after in zip(before_pose, after_pose):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert explorer.completed_steps == [5]


def test_fractional_shadow_frontier_is_deterministic_uniform_and_disjoint():
    _, motion = storage_aware_binary_fixture(coarse_region_gs=4)
    motion.prepare_step(5)
    with torch.no_grad():
        motion.refine_logits[0].fill_(-5.)
        for logits in motion.refine_logits[1:]:
            logits.fill_(5.)
    explorer = shadow_explorer_fixture(motion,
        frontier_policy='reachable_closed_same_level_uniform_without_replacement',
        fraction=.25)
    full = explorer.full_plan(5)
    torch_rng = torch.get_rng_state().clone()
    numpy_rng = copy.deepcopy(np.random.get_state())
    first = explorer.plan(5, frame_index=7, full_plan=full)
    second = explorer.plan(5, frame_index=7, full_plan=full)
    assert torch.equal(torch_rng, torch.get_rng_state())
    current_numpy = np.random.get_state()
    assert numpy_rng[0] == current_numpy[0]
    assert np.array_equal(numpy_rng[1], current_numpy[1])
    assert numpy_rng[2:] == current_numpy[2:]
    candidate_count = len(full['candidate_gate_positions'])
    assert len(first['gate_positions']) == math.ceil(.25 * candidate_count)
    torch.testing.assert_close(first['gate_positions'], second['gate_positions'], rtol=0, atol=0)
    assert len(first['gate_positions']) == len(torch.unique(first['gate_positions']))
    candidate = set(full['candidate_gate_positions'].tolist())
    assert set(first['gate_positions'].tolist()) <= candidate
    rng = np.random.default_rng(np.random.SeedSequence([0, 7, 5, 0, 0x53484144]))
    offsets = np.sort(rng.choice(candidate_count, math.ceil(.25 * candidate_count),
                                 replace=False))
    expected = full['candidate_gate_positions'][torch.as_tensor(offsets)]
    torch.testing.assert_close(first['gate_positions'], expected, rtol=0, atol=0)
    selected_nodes = set(first['gate_nodes'].tolist())
    for depth, mask in enumerate(first['target_rows']):
        rows = torch.where(mask)[0]
        for _ in range(depth):
            rows = motion.active_layout['parents'][depth - 1][rows]
            depth -= 1
        assert set(rows.tolist()) <= selected_nodes


def test_fractional_shadow_frontier_empty_selection_skips_cleanly():
    _, motion = storage_aware_binary_fixture(coarse_region_gs=4)
    motion.prepare_step(5)
    with torch.no_grad():
        for logits in motion.refine_logits:
            logits.fill_(5.)
    explorer = shadow_explorer_fixture(motion,
        frontier_policy='reachable_closed_same_level_uniform_without_replacement',
        fraction=.25)
    full = explorer.full_plan(5)
    plan = explorer.plan(5, frame_index=3, full_plan=full)
    assert len(plan['candidate_gate_positions']) == len(plan['gate_positions']) == 0
    assert not any(bool(mask.any()) for mask in plan['target_rows'])


def test_parent_precision_compensation_preserves_continuous_child_pose():
    _, motion = storage_aware_binary_fixture()
    snapshot_step, evaluation_step, level = 5, 10, 0
    with torch.no_grad():
        for logits in motion.refine_logits:
            logits.fill_(5.)
        motion.region_translation[0][:] = torch.tensor([.037, -.021, .014])
        motion.region_rotation[0][:] = torch.tensor([.081, -.027, .043])
        motion.region_translation[1][:] = torch.tensor([-.018, .029, .011])
        motion.region_rotation[1][:] = torch.tensor([-.052, .031, .047])
    PPC.freeze_hard_choices(motion, snapshot_step)
    node = int(motion.active_layout['refine_rows'][level][0])
    children, parent_members = PPC.immediate_child_coverage(
        motion.active_layout, level, node)
    assert len(children) == 2 and int(parent_members.sum()) > 0
    source_regions = motion._region_motion(evaluation_step, hard=True)
    paths = motion._paths(motion.hard_masks['refine'])
    source_rotation, source_displacement = PPC.cumulative_node_poses(
        motion.active_layout, source_regions, paths)
    report = PPC.rebase_immediate_children(motion, evaluation_step, level, node, 0)
    assert report['children'].tolist() == children.tolist()
    torch.testing.assert_close(report['continuous_position_error'],
                               torch.zeros_like(report['continuous_position_error']),
                               rtol=0, atol=2e-7)
    torch.testing.assert_close(report['continuous_rotation_error'],
                               torch.zeros_like(report['continuous_rotation_error']),
                               rtol=0, atol=1e-6)

    # Cross-check the analytic node calculation against the production
    # reference transform over every Gaussian, before child re-quantization.
    source_pose = BR._conditional_transform(
        motion.base_xyz, motion.base_rotation, motion.active_layout,
        source_regions, paths)
    continuous_regions = [value.detach().clone() for value in source_regions]
    alternate_regions = motion._region_motion(evaluation_step, hard=True)
    continuous_regions[level][node] = alternate_regions[level][node]
    continuous_regions[level + 1][children, :3] = report['target_translation']
    continuous_regions[level + 1][children, 3:] = report['target_rotation_vector']
    continuous_pose = BR._conditional_transform(
        motion.base_xyz, motion.base_rotation, motion.active_layout,
        continuous_regions, paths)
    torch.testing.assert_close(continuous_pose[0], source_pose[0], rtol=0, atol=5e-7)
    source_q = torch.nn.functional.normalize(source_pose[1], dim=-1)
    continuous_q = torch.nn.functional.normalize(continuous_pose[1], dim=-1)
    sign = torch.where((source_q * continuous_q).sum(-1, keepdim=True) < 0, -1., 1.)
    torch.testing.assert_close(continuous_q * sign, source_q, rtol=0, atol=5e-7)

    # Re-quantization is intentionally separate from the continuous identity.
    requantized = motion._region_motion(evaluation_step, hard=True)
    actual_rotation, actual_displacement = PPC.cumulative_node_poses(
        motion.active_layout, requantized, paths)
    assert torch.isfinite(actual_displacement[level + 1][children]).all()
    assert torch.isfinite(actual_rotation[level + 1][children]).all()
    assert not torch.equal(requantized[level + 1][children],
                           torch.cat((report['target_translation'],
                                      report['target_rotation_vector']), dim=-1))
    assert source_displacement[level + 1][children].shape == (2, 3)
    assert source_rotation[level + 1][children].shape == (2, 4)


def test_parent_precision_compensation_rejects_stop_and_incomplete_coverage():
    _, motion = storage_aware_binary_fixture()
    snapshot_step, evaluation_step, level = 5, 10, 0
    with torch.no_grad():
        motion.refine_logits[0].fill_(-5.)
    PPC.freeze_hard_choices(motion, snapshot_step)
    node = int(motion.active_layout['refine_rows'][level][0])
    try:
        PPC.rebase_immediate_children(motion, evaluation_step, level, node, 0)
    except ValueError as error:
        assert 'STOP' in str(error)
    else:
        raise AssertionError('A STOP parent must not be compensated through children')

    with torch.no_grad():
        motion.refine_logits[0].fill_(5.)
    PPC.freeze_hard_choices(motion, snapshot_step)
    try:
        PPC.rebase_immediate_children(motion, snapshot_step, level, node, 0)
    except ValueError as error:
        assert 'frozen hard choices' in str(error)
    else:
        raise AssertionError('Gate-compress step must not silently ignore frozen precision')

    broken = copy.deepcopy(motion.active_layout)
    children = torch.where(broken['parents'][level] == node)[0]
    row = torch.where(broken['index'][level + 1] == children[0])[0][0]
    broken['index'][level + 1][row] = -1
    try:
        PPC.immediate_child_coverage(broken, level, node)
    except ValueError as error:
        assert 'cover' in str(error)
    else:
        raise AssertionError('Incomplete child coverage must be rejected')


def test_parent_precision_child_requantization_zero_and_saturation():
    raw = torch.tensor([
        [.049, 0., 0., 0., 0., 0.],
        [.051, 0., 0., 0., 0., 0.],
        [20., -20., 0., 0., 0., 0.],
    ])
    steps = torch.full_like(raw, .1)
    decoded, symbols, saturated = PPC.quantize_rows(
        raw, steps, torch.tensor([8, 8, 4]))
    assert not bool(symbols[0].ne(0).any())
    assert bool(symbols[1].ne(0).any())
    assert bool(saturated[2])
    assert symbols[2, 0] == 7 and symbols[2, 1] == -8
    torch.testing.assert_close(decoded[1, 0], torch.tensor(.1), rtol=0, atol=0)


def test_parent_precision_compensation_ignores_empty_structural_child():
    _, motion = storage_aware_binary_fixture()
    snapshot_step, evaluation_step, level = 5, 10, 0
    with torch.no_grad():
        for logits in motion.refine_logits:
            logits.fill_(5.)
    node = int(motion.active_layout['refine_rows'][level][0])
    children = torch.where(motion.active_layout['parents'][level] == node)[0]
    owner = motion.active_layout['index'][level + 1]
    owner[owner == children[0]] = children[1]
    motion._cached_occupancy = SAB._occupancy(motion.active_layout)
    PPC.freeze_hard_choices(motion, snapshot_step)
    report = PPC.rebase_immediate_children(
        motion, evaluation_step, level, node, 0)
    assert report['structural_children'].tolist() == children.tolist()
    assert report['children'].tolist() == [int(children[1])]


def test_fixed_gate_value_states_change_only_declared_hard_choices_and_children():
    _, source = storage_aware_binary_fixture()
    snapshot_step, evaluation_step, level = 5, 10, 0
    with torch.no_grad():
        for logits in source.refine_logits:
            logits.fill_(5.)
        source.region_translation[0][:] = torch.tensor([.037, -.021, .014])
        source.region_rotation[0][:] = torch.tensor([.081, -.027, .043])
    PPC.freeze_hard_choices(source, snapshot_step)
    packet = source.capture()
    node = int(source.active_layout['refine_rows'][level][0])
    position = source._refine_position(level, node)
    original_precision = int(source.hard_masks['precision'][level][node])

    def clone():
        result = SAB.StorageAwareBinaryMotion(copy.deepcopy(source.config))
        result.restore(copy.deepcopy(packet))
        return result

    states = {name: clone() for name in ('a', 'b', 'c', 'd', 'e')}
    PPC.set_frozen_node_action(states['b'], level, node, opened=False)
    PPC.set_frozen_node_action(states['c'], level, node, opened=False,
                               parent_precision_index=0)
    PPC.set_frozen_node_action(states['d'], level, node, opened=True,
                               parent_precision_index=0)
    PPC.set_frozen_node_action(states['e'], level, node, opened=True)
    report = PPC.rebase_immediate_children(
        states['e'], evaluation_step, level, node, 0)
    assert bool(states['a'].hard_masks['refine'][level][position])
    assert not bool(states['b'].hard_masks['refine'][level][position])
    assert not bool(states['c'].hard_masks['refine'][level][position])
    assert bool(states['d'].hard_masks['refine'][level][position])
    assert bool(states['e'].hard_masks['refine'][level][position])
    assert int(states['a'].hard_masks['precision'][level][node]) == original_precision
    assert all(int(states[name].hard_masks['precision'][level][node]) == 0
               for name in ('c', 'd', 'e'))
    for name in ('b', 'c', 'd'):
        for actual, expected in zip(states[name].parameters(), states['a'].parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    children = report['children']
    for depth in range(len(source.counts)):
        for field in ('region_translation', 'region_rotation'):
            actual = getattr(states['e'], field)[depth]
            expected = getattr(states['a'], field)[depth]
            changed = (actual != expected).any(-1)
            allowed = torch.zeros_like(changed)
            if depth == level + 1:
                allowed[children] = True
            assert not bool((changed & ~allowed).any())
    # A/B/C/D retain every non-target hard choice exactly.
    for name in ('b', 'c', 'd'):
        for depth, (actual, expected) in enumerate(zip(
                states[name].hard_masks['refine'], states['a'].hard_masks['refine'])):
            difference = actual != expected
            allowed = torch.zeros_like(difference)
            if name in ('b', 'c') and depth == level:
                allowed[position] = True
            assert not bool((difference & ~allowed).any())
        for depth, (actual, expected) in enumerate(zip(
                states[name].hard_masks['precision'], states['a'].hard_masks['precision'])):
            difference = actual != expected
            allowed = torch.zeros_like(difference)
            if name in ('c', 'd') and depth == level:
                allowed[node] = True
            assert not bool((difference & ~allowed).any())


def test_split_region_scores_average_visible_views_before_norm():
    gradients = torch.tensor([
        [[2., 0., 0.], [2., 0., 0.], [float('nan'), 0., 0.]],
        [[-2., 0., 0.], [100., 0., 0.], [4., 0., 0.]],
        [[0., 0., 0.], [6., 0., 0.], [8., 0., 0.]],
    ], dtype=torch.float64)
    visible = torch.tensor([[True, True, True], [True, False, True], [False, True, True]])
    evidence = SPLIT_SCORES.average_position_gradients(
        gradients, visible, min_visible_views=2)
    torch.testing.assert_close(evidence.mean,
        torch.tensor([[0., 0., 0.], [4., 0., 0.], [6., 0., 0.]], dtype=torch.float64),
        rtol=0, atol=0)
    assert evidence.visible_views.tolist() == [2, 2, 2]
    assert evidence.supported.tolist() == [True, True, True]
    assert not evidence.mean.requires_grad


def test_split_region_scores_do_not_cancel_opposite_gs_gradients():
    gradients = torch.tensor([[[3., 0., 0.], [-3., 0., 0.],
                               [0., 2., 0.], [0., 2., 0.]]], dtype=torch.float64)
    evidence = SPLIT_SCORES.average_position_gradients(
        gradients, torch.ones(1, 4, dtype=torch.bool), min_visible_views=1)
    owners = torch.tensor([0, 0, 1, 1])
    scores = SPLIT_SCORES.score_split_regions(evidence, owners, region_count=3)
    torch.testing.assert_close(scores.score, torch.tensor([3., 2., 0.], dtype=torch.float64),
                               rtol=0, atol=0)
    assert scores.total_gs.tolist() == [2, 2, 0]
    assert scores.supported_gs.tolist() == [2, 2, 0]


def test_split_region_scores_preserve_unknown_support_and_normalize_size():
    gradients = torch.tensor([[[4., 0., 0.], [4., 0., 0.], [4., 0., 0.],
                               [4., 0., 0.], [100., 0., 0.]]], dtype=torch.float64)
    visible = torch.tensor([[True, True, True, True, False]])
    evidence = SPLIT_SCORES.average_position_gradients(gradients, visible, min_visible_views=1)
    scores = SPLIT_SCORES.score_split_regions(evidence,
        torch.tensor([0, 1, 1, 1, 0]), region_count=3)
    torch.testing.assert_close(scores.score, torch.tensor([4., 4., 0.], dtype=torch.float64),
                               rtol=0, atol=0)
    assert scores.total_gs.tolist() == [2, 3, 0]
    assert scores.supported_gs.tolist() == [1, 3, 0]
    selected = SPLIT_SCORES.select_split_regions(scores,
        torch.tensor([True, True, True]), top_k=2)
    assert selected.tolist() == [0, 1]
    assert SPLIT_SCORES.select_split_regions(scores,
        torch.tensor([False, True, True]), top_k=2).tolist() == [1]
    assert SPLIT_SCORES.select_split_regions(scores,
        torch.ones(3, dtype=torch.bool), top_k=0).numel() == 0


def gradient_split_fixture(**overrides):
    import yaml
    values = yaml.safe_load((legacy_config_root /
        'fs_gradient_cluster_tree_30f.yaml').read_text())['module']['common_contrast_binary']
    # Small deterministic fixture budgets, separate from experiment settings.
    values.update(coarse_region_gs=12, min_child_gs=1, coarse_fit_steps=2,
        dense_fit_steps=4, gate_compress_steps=4, gate_refine_steps=2,
        motion_steps=10, incremental_steps=0, enable_incremental=False,
        storage_warmup_steps=4, storage_ramp_steps=1, densification_stats_steps=2,
        direct_candidate_windows=0, direct_split_cluster_iterations=8,
        direct_split_top_k=2, direct_max_trial_splits=2)
    values.update(overrides)
    raw = residual_raw_fixture(48)
    motion = CC.CommonContrastMotion(CC.CommonContrastConfig(**values))
    motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    motion.last_step = motion.config.dense_fit_steps + motion.config.gate_compress_steps
    return raw, motion


def test_gradient_split_clustering_scales_blocks_and_preserves_unknowns():
    xyz = torch.tensor([[-3., 0., 0.], [-2., 0., 0.], [-1., 0., 0.],
                        [1., 0., 0.], [2., 0., 0.], [3., 0., 0.], [9., 0., 0.]],
                       dtype=torch.float64)
    gradients = torch.tensor([[[-2., 0., 0.]] * 3 + [[2., 0., 0.]] * 3 +
                              [[float('nan'), 0., 0.]]], dtype=torch.float64)
    evidence = SPLIT_SCORES.average_position_gradients(gradients,
        torch.ones(1, 7, dtype=torch.bool), min_visible_views=1)
    kwargs = dict(min_child_gs=2, iterations=10, position_weight=1., gradient_weight=1.)
    labels = SPLIT_SCORES.cluster_split_region(xyz, evidence, **kwargs)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 1]
    scaled = SPLIT_SCORES.PositionGradientEvidence(evidence.mean * .001,
        evidence.visible_views, evidence.supported)
    again = SPLIT_SCORES.cluster_split_region(xyz * 1000 + 20, scaled, **kwargs)
    assert torch.equal(labels, again)
    assert torch.equal(labels, SPLIT_SCORES.cluster_split_region(xyz, evidence, **kwargs))


def test_split_region_scores_allow_unlimited_or_fractional_candidates():
    scores = SPLIT_SCORES.SplitRegionScores(
        torch.tensor([6., 5., 4., 3., 2., 1., 100., 0.]),
        torch.ones(8, dtype=torch.long), torch.ones(8, dtype=torch.long))
    legal = torch.tensor([True] * 6 + [False, True])
    selected = SPLIT_SCORES.select_split_regions(scores, legal, top_k=None)
    assert selected.tolist() == [0, 1, 2, 3, 4, 5]
    selected = SPLIT_SCORES.select_split_regions(scores, legal, top_k=None, candidate_fraction=.5)
    assert selected.tolist() == [0, 1, 2]
    larger = SPLIT_SCORES.SplitRegionScores(torch.arange(12, 0, -1).float(),
        torch.ones(12, dtype=torch.long), torch.ones(12, dtype=torch.long))
    assert SPLIT_SCORES.select_split_regions(larger,
        torch.ones(12, dtype=torch.bool), top_k=None, candidate_fraction=.5).numel() == 6


def test_gradient_split_more_than_two_regions_produce_decodable_edits():
    raw, source = gradient_split_fixture(direct_split_top_k=None,
                                         direct_max_trial_splits=None,
                                         direct_split_candidate_fraction=.5)
    gradient = torch.zeros(2, len(raw['xyz']), 3)
    gradient[:, :, 0] = torch.arange(1, len(raw['xyz']) + 1).float()
    evidence = SPLIT_SCORES.average_position_gradients(gradient,
        torch.ones(2, len(raw['xyz']), dtype=torch.bool), min_visible_views=2)
    candidate, report = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is not None
    selection = report['split_selection']
    assert len(selection['selected_regions']) == math.ceil(selection['eligible_regions'] * .5)
    applied = [record for record in selection['selected_regions'] if record.get('edit_applied')]
    assert len(applied) > 2 and candidate.counts[-1] == 2 * len(applied)
    xyz, rotation = candidate(candidate.config.motion_steps)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'many_splits.bin'
        moved = dict(raw, xyz=xyz, rotation=rotation)
        candidate.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(moved), previous_count=len(xyz),
            deleted_count=0, birth_count=0)
        decoded, _ = CC.CommonContrastDecoder(candidate.config).advance(path, None, raw)
        assert Q.reference_digest(decoded) == Q.reference_digest(moved)


def test_gradient_split_unlimited_neighbor_windows_are_not_capped_at_two():
    _, source = gradient_split_fixture(direct_gradient_split_enabled=False,
        direct_candidate_windows=None, direct_max_trial_splits=None, direct_split_top_k=None)
    raw = residual_raw_fixture(96)
    source = CC.CommonContrastMotion(copy.deepcopy(source.config))
    source.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    source.last_step = source.config.dense_fit_steps + source.config.gate_compress_steps
    candidate, report = DIRECT_TREE.propose_local_batch(source)
    assert candidate is not None and len(report['windows']) > 2


def test_gradient_split_clustering_rejects_tiny_child_and_missing_evidence():
    xyz = torch.zeros(6, 3, dtype=torch.float64)
    gradients = torch.tensor([[[-1., 0., 0.]] * 5 + [[1., 0., 0.]]], dtype=torch.float64)
    evidence = SPLIT_SCORES.average_position_gradients(gradients,
        torch.ones(1, 6, dtype=torch.bool), min_visible_views=1)
    kwargs = dict(min_child_gs=2, iterations=10, position_weight=1., gradient_weight=1.)
    assert SPLIT_SCORES.cluster_split_region(xyz, evidence, **kwargs) is None
    evidence.supported[2:] = False
    assert SPLIT_SCORES.cluster_split_region(xyz, evidence, **kwargs) is None


def test_gradient_split_candidates_are_independent_and_roundtrip_on_cuda_or_cpu():
    for device in ('cpu', 'cuda') if torch.cuda.is_available() else ('cpu',):
        raw, source = gradient_split_fixture()
        raw = {name: value.to(device) for name, value in raw.items()}
        source.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
        source.last_step = source.config.dense_fit_steps + source.config.gate_compress_steps
        with torch.no_grad():
            source.root_translation[:] = source.base_xyz.new_tensor([.03, -.02, .01])
            source.root_rotation[:] = source.base_xyz.new_tensor([.04, -.02, .03])
            for logit in source.refine_logits:
                logit.fill_(-2.)
        gradients = torch.zeros(2, len(raw['xyz']), 3, device=device)
        gradients[:, :, 0] = torch.linspace(1., 2., len(raw['xyz']), device=device)
        evidence = SPLIT_SCORES.average_position_gradients(gradients,
            torch.ones(2, len(raw['xyz']), device=device, dtype=torch.bool), min_visible_views=2)
        xyz, _ = source(source.last_step)
        before = source(source.config.motion_steps)
        candidate, report = DIRECT_TREE.propose_local_batch(source,
            split_xyz=xyz, split_evidence=evidence)
        assert candidate is not None
        assert len(report['split_selection']['selected_regions']) == 2
        assert all(record['actions'] == ['gradient_cluster_terminal_split']
                   for record in report['windows'])
        assert candidate.counts[-1] == 4
        assert all(record['child_gs'][0] >= source.config.min_child_gs and
                   record['child_gs'][1] >= source.config.min_child_gs
                   for record in report['split_selection']['selected_regions'])
        xyz, rotation = candidate(candidate.config.motion_steps)
        torch.testing.assert_close(xyz, before[0], rtol=0, atol=1e-4)
        torch.testing.assert_close(rotation, before[1], rtol=0, atol=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gradient_tree.bin'
            moved = dict(raw, xyz=xyz, rotation=rotation)
            candidate.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
                post_motion_digest=Q.reference_digest(moved), previous_count=len(xyz),
                deleted_count=0, birth_count=0)
            decoder = CC.CommonContrastDecoder(candidate.config)
            decoded, _ = decoder.advance(path, None, raw)
            assert Q.reference_digest(decoded) == Q.reference_digest(moved)
            assert round(float(sum(candidate.storage_parts(candidate.config.motion_steps, hard=True).values()))) == path.stat().st_size
            restored = CC.CommonContrastMotion(copy.deepcopy(candidate.config))
            restored.restore(candidate.capture())
            assert restored._structural_payload() == candidate._structural_payload()
            candidate.committed = True
            candidate.register_materialized_layout(xyz, torch.empty(0, device=device, dtype=torch.long))
            candidate.setup(xyz, rotation, 1., frame_index=2,
                previous_rows=torch.arange(len(xyz), device=device), previous_new_count=0)
            assert not candidate.tree_rebuilt and candidate.counts[-1] == 4
            xyz, rotation = candidate(candidate.config.motion_steps)
            path2 = Path(directory) / 'next.bin'
            candidate.export_motion(path2, frame_index=2, previous_state_digest=Q.reference_digest(decoded),
                post_motion_digest=Q.reference_digest(dict(decoded, xyz=xyz, rotation=rotation)),
                previous_count=len(xyz), deleted_count=0, birth_count=0)
            decoded, _ = decoder.advance(path2, None, decoded)
            assert SAB.layout_digest(decoder.materialized_layout) == SAB.layout_digest(candidate.active_layout)


def test_gradient_split_no_signal_and_node_budget_make_no_proposal():
    raw, source = gradient_split_fixture()
    gradients = torch.zeros(2, len(raw['xyz']), 3)
    evidence = SPLIT_SCORES.average_position_gradients(gradients,
        torch.ones(2, len(raw['xyz']), dtype=torch.bool), min_visible_views=2)
    candidate, report = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is None and not report['split_selection']['selected_regions']
    source.config.direct_node_budget = sum(source.counts)
    evidence.mean.fill_(1.)
    candidate, report = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is None and report['split_selection']['eligible_regions'] == 0


def test_gradient_split_component_modes_omit_blocks_and_decode_old_version():
    raw, motion = gradient_split_fixture(direct_tree_enabled=False,
                                         direct_gradient_split_enabled=False)
    with torch.no_grad():
        for logit in motion.refine_logits:
            logit.fill_(-2.)
    with tempfile.TemporaryDirectory() as directory:
        for translation, rotation in ((True, False), (False, True), (True, True), (False, False)):
            with torch.no_grad():
                motion.root_translation.zero_()
                motion.root_rotation.zero_()
                if translation:
                    motion.root_translation[:, 0] = .03
                if rotation:
                    motion.root_rotation[:, 2] = .04
            xyz, q = motion(motion.config.motion_steps)
            moved = dict(raw, xyz=xyz, rotation=q)
            sizes = []
            for component_modes in (True, False):
                motion.config.component_modes = component_modes
                path = Path(directory) / f'{translation}_{rotation}_{component_modes}.bin'
                motion.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
                    post_motion_digest=Q.reference_digest(moved), previous_count=len(xyz),
                    deleted_count=0, birth_count=0)
                dx, dq, *_ = motion.decode_motion(path, raw['xyz'], raw['rotation'],
                                                  motion.config, motion.active_layout)
                assert Q.reference_digest(dict(raw, xyz=dx, rotation=dq)) == Q.reference_digest(moved)
                assert round(float(sum(motion.storage_parts(motion.config.motion_steps, hard=True).values()))) == path.stat().st_size
                sizes.append(path.stat().st_size)
            if translation != rotation:
                assert sizes[0] < sizes[1]


def test_gradient_split_probe_restores_state_and_uses_only_image_world_gradients():
    torch.manual_seed(20261008)
    parameter = torch.nn.Parameter(torch.zeros(3, 3))
    parameter.grad = torch.full_like(parameter, 7.)
    cfg = SimpleNamespace(direct_split_camera_names=['cam01.png', 'cam02.png'],
                          direct_split_min_visible_views=2)
    class Motion:
        config = cfg
        last_step = 10
        def __call__(self, step):
            return parameter, torch.tensor([[1., 0., 0., 0.]]).expand(3, -1)
    class Data:
        config = SimpleNamespace(training_camera_names=None, validation_camera_names=None)
        _camera_indices = staticmethod(DATA.DataModule._camera_indices)
        def get_train_dataset(self):
            return SimpleNamespace(items=[SimpleNamespace(path=Path(name))
                                          for name in cfg.direct_split_camera_names])
        def train_dataloader(self, random, indices):
            torch.rand(1)
            return [SimpleNamespace(path=Path(name), image=torch.ones(3, 1, 3),
                to=lambda device, name=name: SimpleNamespace(path=Path(name), image=torch.ones(3, 1, 3)))
                for name in cfg.direct_split_camera_names]
    sentinel_xyz, sentinel_rotation = object(), object()
    gs = SimpleNamespace(rigid_xyz=sentinel_xyz, rigid_rotation=sentinel_rotation)
    original_model = object()
    class Renderer:
        gaussians = original_model
        def __call__(self, view, background):
            assert self.gaussians is gs and gs.rigid_xyz.is_leaf and gs.rigid_xyz.requires_grad
            torch.testing.assert_close(gs.rigid_xyz, parameter.detach(), rtol=0, atol=0)
            return dict(render=gs.rigid_xyz.T[:, None, :], visibility_filter=torch.ones(3, dtype=torch.bool))
    subject = SimpleNamespace(residual_motion=Motion(), datamodule=Data(),
        renderer=Renderer(), gaussians=gs, config=SimpleNamespace(lambda_dssim=0.),
        background=torch.zeros(3), device='cpu')
    rng = torch.random.get_rng_state().clone()
    with patch.object(DIRECT_TRANSACTION, 'ssim', lambda image, target: image.sum() * 0 + 1):
        xyz, evidence, report = DIRECT_TRANSACTION.collect_split_position_gradients(subject)
    torch.testing.assert_close(evidence.mean, torch.full((3, 3), -1. / 9), rtol=0, atol=1e-8)
    assert evidence.supported.all() and report['cameras'] == cfg.direct_split_camera_names
    assert torch.equal(rng, torch.random.get_rng_state())
    assert subject.renderer.gaussians is original_model
    assert gs.rigid_xyz is sentinel_xyz and gs.rigid_rotation is sentinel_rotation
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 7.), rtol=0, atol=0)
    torch.testing.assert_close(xyz, parameter.detach(), rtol=0, atol=0)
    with patch.object(DIRECT_TRANSACTION, 'ssim', side_effect=RuntimeError('probe failure')):
        try:
            DIRECT_TRANSACTION.collect_split_position_gradients(subject)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Probe failure did not propagate')
    assert subject.renderer.gaussians is original_model
    assert gs.rigid_xyz is sentinel_xyz and gs.rigid_rotation is sentinel_rotation
    cfg.direct_split_camera_names = ['cam00.png']
    try:
        DIRECT_TRANSACTION.collect_split_position_gradients(subject)
    except ValueError:
        pass
    else:
        raise AssertionError('Held-out scoring camera was accepted')


def test_gradient_split_config_requires_explicit_parameters():
    _, source = gradient_split_fixture()
    for changes in (dict(direct_split_min_visible_views=6), dict(direct_split_top_k=3),
                    dict(direct_split_candidate_fraction=0.), dict(direct_split_candidate_fraction=1.1),
                    dict(direct_split_position_weight=0.), dict(direct_split_cluster_iterations=None),
                    dict(direct_tree_enabled=False)):
        values = asdict(source.config)
        values.update(changes)
        try:
            CC.CommonContrastConfig(**values).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid gradient split settings accepted: {changes}')


def node_edit_fixture(**overrides):
    settings = dict(direct_edit_encoding='node_ops', direct_prune_inactive_splits=True,
                    direct_proposal_step=4, direct_split_min_child_gs=1,
                    direct_enable_merges=False)
    settings.update(overrides)
    raw, source = gradient_split_fixture(**settings)
    source.last_step = source.config.dense_fit_steps
    gradient = torch.zeros(2, len(raw['xyz']), 3)
    gradient[:, :, 0] = torch.linspace(1., 2., len(raw['xyz']))
    evidence = SPLIT_SCORES.average_position_gradients(gradient,
        torch.ones(2, len(raw['xyz']), dtype=torch.bool), min_visible_views=2)
    return raw, source, evidence


def test_node_edit_codec_roundtrips_packed_membership_and_rejects_corruption():
    operations = [dict(kind='swap', left=4, right=10000), dict(kind='merge', a=2, b=7),
                  dict(kind='split', level=2, node=41, labels=[0, 1, 0, 1, 0, 1, 1, 0, 1])]
    wire = NODE_EDITS.encode_node_edits(operations)
    assert NODE_EDITS.decode_node_edits(wire) == operations
    assert len(wire) < 30
    for bad in (wire[:-1], wire + b'\x00', b'NED1\x01\x7f', b'NED1\x01\x03\x02\x01\x08'):
        try:
            NODE_EDITS.decode_node_edits(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt node edit was decoded')


def test_node_edit_candidate_has_trainable_gates_and_exact_v4_cost():
    raw, source, evidence = node_edit_fixture()
    candidate, report = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate.hard_masks is None
    candidate.prepare_step(candidate.config.dense_fit_steps + 1)
    assert all(parameter.requires_grad for parameter in candidate.refine_logits)
    cost = candidate.storage_parts(candidate.config.dense_fit_steps + 1)['structure_edit_bytes']
    gradients = torch.autograd.grad(cost, list(candidate.refine_logits), allow_unused=True)
    assert any(gradient is not None and bool(gradient.abs().sum() > 0) for gradient in gradients)
    wire = candidate._structural_payload()
    assert wire.startswith(NODE_EDITS.MAGIC)
    replay = DIRECT_TREE.apply_layout_edits(source.active_layout, wire, raw['xyz'])
    assert SAB.layout_digest(replay) == SAB.layout_digest(candidate.active_layout)
    assert len(wire) < len(DIRECT_TREE.layout_edits(source.active_layout, candidate.active_layout))
    xyz, rotation = candidate(candidate.config.motion_steps)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'v4.bin'
        moved = dict(raw, xyz=xyz, rotation=rotation)
        candidate.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(moved), previous_count=len(xyz), deleted_count=0, birth_count=0)
        assert candidate.read_header(path)['codec'] == 'common_contrast_node_edits_v4'
        decoded, _ = CC.CommonContrastDecoder(candidate.config).advance(path, None, raw)
        assert Q.reference_digest(decoded) == Q.reference_digest(moved)
        assert round(float(sum(candidate.storage_parts(candidate.config.motion_steps, hard=True).values()))) == path.stat().st_size
        resumed = CC.CommonContrastMotion(copy.deepcopy(candidate.config))
        resumed.restore(candidate.capture())
        assert resumed._pending_edit_operations == candidate._pending_edit_operations
        assert resumed._structural_payload() == wire


def test_node_edit_pruning_removes_only_closed_new_routes_without_pose_change():
    raw, source, evidence = node_edit_fixture()
    candidate, _ = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    with torch.no_grad():
        candidate.root_translation[:] = torch.tensor([.03, -.02, .01])
        candidate.root_rotation[:] = torch.tensor([.04, -.02, .03])
        candidate.contrast_translation[2][:, 0] = .002
    candidate.hard_masks = dict(refine=[torch.ones(len(rows), dtype=torch.bool)
                                       for rows in candidate.active_layout['refine_rows']])
    first = candidate._pending_edit_operations[0]
    position = candidate._cached_gate_positions[first['level']][first['node']]
    candidate.hard_masks['refine'][first['level']][position] = False
    candidate.last_step = candidate.config.motion_steps
    before = candidate(candidate.config.motion_steps)
    pruned, report = DIRECT_TREE.prune_inactive_split_edits(candidate)
    assert report['removed_splits'] == 1 and pruned.counts[-1] == 2
    after = pruned(pruned.config.motion_steps)
    torch.testing.assert_close(after[0], before[0], rtol=0, atol=0)
    torch.testing.assert_close(after[1], before[1], rtol=0, atol=0)
    assert len(pruned._pending_edit_operations) == 1
    decoded_layout = DIRECT_TREE.apply_layout_edits(pruned._frame_coding_layout,
        pruned._structural_payload(), raw['xyz'])
    assert SAB.layout_digest(decoded_layout) == SAB.layout_digest(pruned.active_layout)


def test_node_edit_merge_replay_matches_existing_topology_edits():
    _, source, _ = node_edit_fixture(direct_gradient_split_enabled=False,
        direct_candidate_windows=None, direct_enable_merges=True, direct_prune_inactive_splits=False)
    raw = residual_raw_fixture(96)
    source = CC.CommonContrastMotion(copy.deepcopy(source.config))
    source.setup(raw['xyz'], raw['rotation'], 1., frame_index=1)
    source.last_step = source.config.dense_fit_steps
    candidate, _ = DIRECT_TREE.propose_local_batch(source)
    operations = NODE_EDITS.decode_node_edits(candidate._structural_payload())
    assert any(operation['kind'] == 'merge' for operation in operations)
    replay = DIRECT_TREE.apply_layout_edits(source.active_layout,
        candidate._structural_payload(), raw['xyz'])
    assert SAB.layout_digest(replay) == SAB.layout_digest(candidate.active_layout)


def test_node_edit_merge_disagreement_filters_incompatible_roots():
    raw, source, evidence = node_edit_fixture(direct_candidate_windows=None,
        direct_enable_merges=True, direct_merge_max_disagreement=.01)
    with torch.no_grad():
        source.root_translation[:, 0] = torch.arange(source.counts[0]) * .1
    evidence.mean.zero_()
    candidate, report = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is None


def test_node_edit_warm_prior_opens_edited_paths_for_joint_gate_learning():
    raw, source, evidence = node_edit_fixture(direct_candidate_gate_logit=1.)
    candidate, _ = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    for operation in candidate._pending_edit_operations:
        if operation['kind'] != 'split':
            continue
        level, node = operation['level'], operation['node']
        while level >= 0:
            position = candidate._cached_gate_positions[level].get(node)
            if position is not None:
                assert float(candidate.refine_logits[level][position]) >= 1.
            if level == 0:
                break
            node = int(candidate.parents[level - 1][node])
            level -= 1
    assert candidate.hard_masks is None


def test_node_edit_four_level_fixed_control_bootstrap_decodes_identically():
    raw, source, _ = node_edit_fixture(direct_initial_levels=4,
                                      direct_gradient_split_enabled=False)
    assert source.counts[-1] > 0
    xyz, rotation = source(source.config.motion_steps)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'fixed4.bin'
        source.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(dict(raw, xyz=xyz, rotation=rotation)),
            previous_count=len(xyz), deleted_count=0, birth_count=0)
        decoded, _ = CC.CommonContrastDecoder(source.config).advance(path, None, raw)
        assert Q.reference_digest(decoded) == Q.reference_digest(dict(raw, xyz=xyz, rotation=rotation))


def test_node_edit_swap_replay_matches_full_four_level_motion():
    raw, source, _ = node_edit_fixture(direct_initial_levels=4,
        direct_gradient_split_enabled=False, direct_candidate_windows=None,
        direct_enable_merges=False)
    with torch.no_grad():
        source.contrast_translation[0][:, 0] = .04
    candidate, _ = DIRECT_TREE.propose_local_batch(source)
    assert candidate is not None
    operations = NODE_EDITS.decode_node_edits(candidate._structural_payload())
    assert any(operation['kind'] == 'swap' for operation in operations)
    replay = DIRECT_TREE.apply_layout_edits(source.active_layout,
        candidate._structural_payload(), raw['xyz'])
    assert SAB.layout_digest(replay) == SAB.layout_digest(candidate.active_layout)
    xyz, rotation = candidate(candidate.config.motion_steps)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'swap4.bin'
        source_state = Q.reference_digest(raw)
        target_state = Q.reference_digest(dict(raw, xyz=xyz, rotation=rotation))
        candidate.export_motion(path, frame_index=1, previous_state_digest=source_state,
            post_motion_digest=target_state, previous_count=len(xyz), deleted_count=0, birth_count=0)
        decoded, _ = CC.CommonContrastDecoder(candidate.config).advance(path, None, raw)
        assert Q.reference_digest(decoded) == target_state


def test_rigid_gradient_projection_removes_shared_translation_rotation_and_unknowns():
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                        [0., 0., 1.], [1., 1., 1.], [500., 500., 500.]], dtype=torch.float64)
    omega = torch.tensor([.3, -.2, .4], dtype=torch.float64)
    gradients = torch.cross(omega.expand_as(xyz), xyz, dim=-1) + torch.tensor([2., -1., 3.])
    gradients[-1] = float('nan')
    support = torch.tensor([True, True, True, True, True, False])
    evidence = SPLIT_SCORES.PositionGradientEvidence(gradients, support.long() * 2, support)
    owners = torch.zeros(6, dtype=torch.long)
    residual = SPLIT_SCORES.rigid_gradient_residual(xyz, evidence, owners, region_count=2)
    torch.testing.assert_close(residual.mean, torch.zeros_like(xyz), rtol=0, atol=0)
    assert torch.equal(residual.supported, support)
    scores = SPLIT_SCORES.score_split_regions(residual, owners, region_count=2)
    torch.testing.assert_close(scores.score, torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
    assert scores.supported_gs.tolist() == [5, 0]


def test_rigid_gradient_projection_keeps_internal_conflict_and_is_scale_invariant():
    xyz = torch.zeros(8, 3, dtype=torch.float64)
    xyz[:, 0] = torch.arange(8, dtype=torch.float64)
    gradient = torch.zeros_like(xyz)
    gradient[:, 0] = torch.tensor([-1., -1., -1., -1., 1., 1., 1., 1.])
    evidence = SPLIT_SCORES.PositionGradientEvidence(
        gradient, torch.full((8,), 2), torch.ones(8, dtype=torch.bool))
    owners = torch.zeros(8, dtype=torch.long)
    residual = SPLIT_SCORES.rigid_gradient_residual(xyz, evidence, owners, region_count=1)
    shifted = SPLIT_SCORES.rigid_gradient_residual(xyz * 100 + 17, evidence, owners, region_count=1)
    torch.testing.assert_close(residual.mean, gradient, rtol=0, atol=1e-12)
    torch.testing.assert_close(shifted.mean, residual.mean, rtol=0, atol=1e-12)
    child_owners = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    child = SPLIT_SCORES.rigid_gradient_residual(xyz, evidence, child_owners, region_count=2)
    torch.testing.assert_close(child.mean, torch.zeros_like(xyz), rtol=0, atol=0)
    # Collinear/zero-radius groups use a pseudoinverse, without NaNs.
    coincident = SPLIT_SCORES.rigid_gradient_residual(torch.zeros_like(xyz), evidence,
                                                   owners, region_count=1)
    assert bool(torch.isfinite(coincident.mean).all())


def test_rigid_split_rejects_shared_field_and_preserves_existing_motion_parameters():
    raw, source, evidence = node_edit_fixture(direct_split_score_mode='rigid_residual',
        direct_preserve_split_motion=True, direct_candidate_windows=0)
    evidence.mean.fill_(2.)
    candidate, report = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is None and report['split_selection']['eligible_regions'] == 0
    evidence.mean.zero_()
    evidence.mean[:, 0] = torch.linspace(1., 2., len(raw['xyz']))
    with torch.no_grad():
        source.root_translation[:, 1].fill_(.0001234)
        source.root_rotation[:, 2].fill_(.0012345)
        source.contrast_translation[0][:, 2].fill_(.0002345)
    candidate, report = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is not None and report['preserved_split_roots']
    assert report['split_selection']['score_mode'] == 'rigid_residual'
    torch.testing.assert_close(candidate.root_translation, source.root_translation, rtol=0, atol=0)
    torch.testing.assert_close(candidate.root_rotation, source.root_rotation, rtol=0, atol=0)
    for level, rows in enumerate(candidate.active_layout['refine_rows']):
        previous = {int(node): pos for pos, node in enumerate(source.active_layout['refine_rows'][level])}
        for position, node in enumerate(rows.tolist()):
            for name in ('contrast_translation', 'contrast_rotation'):
                value = getattr(candidate, name)[level][position]
                expected = (getattr(source, name)[level][previous[node]] if node in previous else
                            torch.zeros_like(value))
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
    old_xyz, old_q = source(source.last_step)
    new_xyz, new_q = candidate(candidate.last_step)
    torch.testing.assert_close(new_xyz, old_xyz, rtol=0, atol=2e-6)
    torch.testing.assert_close(new_q, old_q, rtol=0, atol=2e-6)


def test_rigid_gradient_projection_and_adaptive_codec_on_cuda_when_available():
    if not torch.cuda.is_available():
        return
    raw, previous, evidence = node_edit_fixture(direct_split_score_mode='rigid_residual',
        direct_preserve_split_motion=True, direct_candidate_windows=0)
    owners = torch.zeros(len(raw['xyz']), dtype=torch.long)
    expected = SPLIT_SCORES.rigid_gradient_residual(raw['xyz'], evidence, owners, region_count=1)
    gpu_raw = {name: value.cuda() for name, value in raw.items()}
    gpu_evidence = SPLIT_SCORES.PositionGradientEvidence(
        evidence.mean.cuda(), evidence.visible_views.cuda(), evidence.supported.cuda())
    actual = SPLIT_SCORES.rigid_gradient_residual(gpu_raw['xyz'], gpu_evidence,
                                               owners.cuda(), region_count=1)
    torch.testing.assert_close(actual.mean.cpu(), expected.mean, rtol=1e-5, atol=1e-8)
    motion = CC.CommonContrastMotion(copy.deepcopy(previous.config))
    motion.setup(gpu_raw['xyz'], gpu_raw['rotation'], 1., frame_index=1)
    motion.last_step = motion.config.dense_fit_steps
    with torch.no_grad():
        motion.root_translation[:, 1].fill_(.0001234)
        motion.root_rotation[:, 2].fill_(.0012345)
    candidate, _ = DIRECT_TREE.propose_local_batch(motion, split_xyz=gpu_raw['xyz'],
                                                  split_evidence=gpu_evidence)
    assert candidate is not None and candidate.counts[-1] > 0
    with torch.no_grad():
        candidate.contrast_translation[-1][:, 0].fill_(.00005)
    xyz, rotation = candidate(candidate.config.motion_steps)
    target = dict(gpu_raw, xyz=xyz, rotation=rotation)
    packet, _ = candidate.encode_motion(frame_index=1,
        previous_state_digest=Q.reference_digest(gpu_raw), post_motion_digest=Q.reference_digest(target),
        previous_count=len(xyz), deleted_count=0, birth_count=0)
    decoded, _ = CC.CommonContrastDecoder(candidate.config).advance(packet, None, gpu_raw)
    assert Q.reference_digest(decoded) == Q.reference_digest(target)


def test_common_contrast_memory_packet_matches_file_and_validates_corruption():
    raw, source, evidence = node_edit_fixture()
    candidate, _ = DIRECT_TREE.propose_local_batch(source,
        split_xyz=raw['xyz'], split_evidence=evidence)
    xyz, rotation = candidate(candidate.config.motion_steps)
    target = dict(raw, xyz=xyz, rotation=rotation)
    arguments = dict(frame_index=1, previous_state_digest=Q.reference_digest(raw),
        post_motion_digest=Q.reference_digest(target), previous_count=len(xyz),
        deleted_count=0, birth_count=0)
    packet, report = candidate.encode_motion(**arguments)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'motion.bin'
        exported = candidate.export_motion(path, **arguments)
        assert packet == path.read_bytes()
        assert report['archive_bytes'] == exported['archive_bytes'] == len(packet)
        layout = DIRECT_TREE.apply_layout_edits(candidate._frame_coding_layout,
            candidate.read_header(packet)['structure_edits'], raw['xyz'])
        memory_xyz, memory_q, *_ = candidate.decode_motion(packet, raw['xyz'], raw['rotation'],
                                                         candidate.config, layout)
        file_xyz, file_q, *_ = candidate.decode_motion(path, raw['xyz'], raw['rotation'],
                                                     candidate.config, layout)
        torch.testing.assert_close(memory_xyz, file_xyz, rtol=0, atol=0)
        torch.testing.assert_close(memory_q, file_q, rtol=0, atol=0)
        assert Q.reference_digest(dict(raw, xyz=memory_xyz, rotation=memory_q)) == Q.reference_digest(target)
    try:
        candidate.read_header(packet[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError('Truncated in-memory packet was accepted')


def test_direct_transaction_can_skip_trial_snapshots_without_skipping_decoded_selection():
    raw, control, evidence = node_edit_fixture(direct_save_trial_artifacts=False,
        direct_split_score_mode='rigid_residual', direct_preserve_split_motion=True)
    candidate, proposal = DIRECT_TREE.propose_local_batch(control,
        split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is not None
    proposal['proposal_seconds'] = 0.
    gaussians = SimpleNamespace(rigid_xyz=None, rigid_rotation=None,
        xyz_gradient_accum=torch.zeros(len(raw['xyz']), 1), denom=torch.ones(len(raw['xyz']), 1))
    for name, attribute in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(gaussians, attribute, torch.nn.Parameter(raw[name].clone()))
    image = torch.ones(3, 16, 16)
    views = []
    for index in range(1, 21):
        view = SimpleNamespace(path=Path(f'cam{index:02d}.png'), image=image)
        view.to = lambda device, view=view: view
        views.append(view)
    with tempfile.TemporaryDirectory() as temporary:
        module = SimpleNamespace(residual_motion=candidate, gaussians=gaussians,
            output_dir=Path(temporary), current_frame=1, device=torch.device('cpu'),
            quant_previous_reference=raw, quant_previous_sha256=Q.reference_digest(raw),
            quant_deleted_indices=torch.empty(0, dtype=torch.int32),
            quant_first_frame_gs_count=len(raw['xyz']), quant_stats_window_initialized=False,
            config=SimpleNamespace(lambda_dssim=.2), background=torch.zeros(3),
            trainer=SimpleNamespace(synchronize=lambda: None),
            datamodule=SimpleNamespace(train_dataloader=lambda random=False: views),
            renderer=lambda view, background: dict(render=image))
        transaction = DIRECT_TRANSACTION.DirectTransaction(module, control, proposal)
        with patch.object(torch, 'save', side_effect=AssertionError('Unexpected trial snapshot')):
            transaction.finish(module)
        directory = Path(temporary) / 'direct_tree_trials' / 'frame_001'
        assert {path.name for path in directory.iterdir()} == {'report.json'}
        report = json.loads((directory / 'report.json').read_text())
        assert not report['trial_artifacts_saved'] and report['trial_artifact_write_seconds'] == 0
        assert report['trial_codec_seconds'] > 0
        assert set(report['branches']) == {'control', 'candidate'}
        assert not report['accepted'] and module.residual_motion is control


def test_periodic_birth_retention_preserves_low_opacity_fitting_and_final_storage():
    reference = residual_raw_fixture(3)
    model = SimpleNamespace()
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(reference[name][:1].clone()))
    with torch.no_grad():
        model._opacity.fill_(-5.3)  # About 0.005, below the 0.01 keep threshold.
        model._features_dc.fill_(.4)
    legacy = Q.QuantizedNewGaussianView(model, residual_test_config(), 1., reference)
    signal = legacy.get_features.sum((1, 2)) + legacy.get_xyz.sum(-1)
    (legacy.get_opacity.flatten() * signal).sum().backward()
    assert not legacy.keep.any()
    torch.testing.assert_close(model._features_dc.grad, torch.zeros_like(model._features_dc), rtol=0, atol=0)
    torch.testing.assert_close(model._xyz.grad, torch.zeros_like(model._xyz), rtol=0, atol=0)
    for attr in Q.RAW_ATTRIBUTES:
        getattr(model, attr).grad = None
    config = residual_test_config(birth_retention_mode='periodic_prune')
    config.validate()
    view = Q.QuantizedNewGaussianView(model, config, 1., reference)
    assert view.keep.all() and not view.keep_st.requires_grad
    assert float(view.get_opacity.max()) < config.opacity_keep_threshold
    signal = view.get_features.sum((1, 2)) + view.get_xyz.sum(-1)
    (view.get_opacity.flatten() * signal).sum().backward()
    assert bool((model._features_dc.grad != 0).all())
    assert bool((model._xyz.grad != 0).all())
    assert bool(torch.isfinite(model._opacity.grad).all())
    # Every rendered candidate costs one stored row and survives finalization.
    assert float(view.keep_st.sum()) == int(view.keep.sum()) == 1
    report = view.apply_retention_budget(None)
    assert report['eligible'] == 1 and report['pruned'] == 0
    raw, sources = Q.materialize_quantized_gaussians(reference, view)
    assert len(raw['xyz']) == 4 and len(sources) == 1
    for field in Q.RAW_FIELDS:
        expected = view.raw_value(field)
        if field == 'rotation':
            expected = F.normalize(expected, dim=-1)
        torch.testing.assert_close(raw[field][-1:], expected, rtol=0, atol=0)


def test_periodic_birth_retention_preserves_explicit_budget_and_empty_pool():
    reference = residual_raw_fixture(3)
    config = residual_test_config(birth_retention_mode='periodic_prune')
    model = SimpleNamespace()
    for name, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(reference[name].clone()))
    with torch.no_grad():
        model._opacity[:, 0] = torch.tensor([-6., -5., -7.])
    view = Q.QuantizedNewGaussianView(model, config, 1., reference)
    assert view.keep.all()  # All are below 0.01, but no periodic prune happened.
    report = view.apply_retention_budget(1)
    assert report['eligible'] == 3 and report['pruned'] == 2
    assert view.keep.tolist() == [False, True, False]
    assert int((view.get_opacity > 0).sum()) == 1
    empty = SimpleNamespace()
    for attr in Q.RAW_ATTRIBUTES:
        setattr(empty, attr, torch.nn.Parameter(getattr(model, attr)[:0].detach().clone()))
    empty_view = Q.QuantizedNewGaussianView(empty, config, 1., reference)
    assert empty_view.get_opacity.shape == (0, 1)
    raw, sources = Q.materialize_quantized_gaussians(reference, empty_view)
    assert len(raw['xyz']) == len(reference['xyz']) and not len(sources)


def test_periodic_birth_retention_matches_stream_bytes_and_sequential_decoding():
    raw, motion = common_contrast_fixture(birth_retention_mode='periodic_prune')
    xyz, rotation = motion(motion.config.motion_steps)
    moved = dict(raw, xyz=xyz, rotation=rotation)
    model = SimpleNamespace()
    for field, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
        setattr(model, attr, torch.nn.Parameter(moved[field][:1].clone()))
    with torch.no_grad():
        model._opacity.fill_(-5.3)
    view = Q.QuantizedNewGaussianView(model, motion.config, 1., moved)
    final, _ = Q.materialize_quantized_gaussians(moved, view)
    assert len(final['xyz']) == len(raw['xyz']) + 1
    assert float(torch.sigmoid(final['opacity'][-1])) < motion.config.opacity_keep_threshold
    module_ast = ast.parse((project_root / 'pipeline/ReCon-GS/archive/legacy_module.py').read_text())
    module_class = next(node for node in module_ast.body if isinstance(node, ast.ClassDef) and node.name == 'Module')
    storage_function = next(node for node in module_class.body if isinstance(node, ast.FunctionDef)
                            and node.name == 'quantized_storage')
    scope = dict(torch=torch, np=np, birth_ply_size=SAB.birth_ply_size,
                 SPARSE_MOTION_MODES={'common_contrast_binary'})
    exec(compile(ast.Module(body=[storage_function], type_ignores=[]), 'module_storage', 'exec'), scope)
    fake = SimpleNamespace(residual_motion=motion, global_step=motion.config.total_steps,
        config=SimpleNamespace(motion_mode='common_contrast_binary', sh_degree=1),
        quant_deleted_indices=torch.empty(0, dtype=torch.int32))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        birth_model = GAUSS.GaussianModel(1)
        for field, attr in zip(Q.RAW_FIELDS, Q.RAW_ATTRIBUTES):
            setattr(birth_model, attr, torch.nn.Parameter(final[field][-1:].clone()))
        birth_path = path / 'birth.ply'
        birth_model.save_ply(birth_path)
        motion_path = path / 'motion.bin'
        motion.export_motion(motion_path, frame_index=1,
            previous_state_digest=Q.reference_digest(raw), post_motion_digest=Q.reference_digest(moved),
            previous_count=len(raw['xyz']), deleted_count=0, birth_count=1)
        expected, expected_parts = scope['quantized_storage'](fake, view, hard=False)
        hard, hard_parts = scope['quantized_storage'](fake, view, hard=True)
        assert int(hard_parts['new_bytes']) == int(expected_parts['new_bytes']) == birth_path.stat().st_size
        assert int(hard) == motion_path.stat().st_size + birth_path.stat().st_size
        torch.testing.assert_close(expected, hard, rtol=0, atol=.875)  # Byte padding is hard-only.
        decoded, _ = CC.CommonContrastDecoder(motion.config).advance(motion_path, birth_path, raw)
        assert Q.reference_digest(decoded) == Q.reference_digest(final)


def test_birth_retention_mode_validates_and_old_checkpoint_keeps_legacy_behavior():
    for mode in ('unknown', None, False, [], {}):
        try:
            residual_test_config(birth_retention_mode=mode).validate()
        except ValueError as error:
            assert 'birth_retention_mode' in str(error)
        else:
            raise AssertionError(f'Invalid birth retention mode accepted: {mode!r}')
    raw = residual_raw_fixture()
    config = residual_test_config()
    motion = Q.QuantizedResidualMotion(config)
    motion.setup(raw['xyz'], raw['rotation'], 1.)
    packet = copy.deepcopy(motion.capture())
    packet['config'].pop('birth_retention_mode')
    restored = Q.QuantizedResidualMotion(config)
    restored.restore(packet)
    assert restored.config.birth_retention_mode == 'opacity_threshold'
    try:
        Q.QuantizedResidualMotion(residual_test_config(birth_retention_mode='periodic_prune')).restore(packet)
    except ValueError:
        pass
    else:
        raise AssertionError('Changing birth retention across a checkpoint was accepted')


def test_appearance_birth_budget_is_independent_and_stable_without_geometry_signal():
    scores = torch.tensor([0., 4., 4., 9., 1., 0., 0., 0.], dtype=torch.float64)
    eligible = torch.tensor([True, True, True, False, True, True, True, True])
    selected = BIRTH.select_appearance_birth_sources(scores, eligible, .25)
    assert selected.tolist() == [1, 2]
    assert not len(BIRTH.select_appearance_birth_sources(scores, eligible, 0.))
    assert not len(BIRTH.select_appearance_birth_sources(torch.zeros_like(scores), eligible, 1.))
    for bad in (-.1, 1.1, float('nan'), True):
        try:
            BIRTH.select_appearance_birth_sources(scores, eligible, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid appearance fraction accepted: {bad!r}')


def test_position_gradient_births_exclude_zero_gradient_sources_and_obey_optional_budget():
    def model(geometry=True):
        fake = SimpleNamespace(_xyz=torch.arange(12, dtype=torch.float64).reshape(4, 3),
            xyz_gradient_accum=torch.tensor([[.2], [0.], [.3], [0.]], dtype=torch.float64)
                if geometry else torch.zeros(4, 1, dtype=torch.float64),
            denom=torch.ones(4, 1, dtype=torch.float64), percent_dense=.01,
            _features_dc=torch.arange(12, dtype=torch.float64).reshape(4, 1, 3),
            _features_rest=torch.zeros(4, 3, 3, dtype=torch.float64),
            _opacity=torch.zeros(4, 1, dtype=torch.float64),
            _scaling=torch.full((4, 3), -20., dtype=torch.float64),
            _rotation=torch.tensor([[1., 0., 0., 0.]], dtype=torch.float64).repeat(4, 1))
        fake.get_scaling = fake._scaling.exp()
        fake.densification_postfix = lambda *values: setattr(fake, 'births', values)
        return fake
    torch.manual_seed(117)
    fake = model()
    result = GAUSS.GaussianModel.adaptive_densify_and_clone(fake, .1, 1.)
    assert result == dict(eligible=2, added=2, blocked=0)
    torch.testing.assert_close(fake.births[0], fake._xyz[[0, 2]], rtol=0, atol=1e-7)
    torch.testing.assert_close(fake.births[1], fake._features_dc[[0, 2]], rtol=0, atol=0)
    limited = model()
    result = GAUSS.GaussianModel.adaptive_densify_and_clone(limited, .1, 1.,
        max_new_gaussians=1)
    assert result == dict(eligible=2, added=1, blocked=1)
    torch.testing.assert_close(limited.births[1], limited._features_dc[[2]], rtol=0, atol=0)
    inactive = model(geometry=False)
    result = GAUSS.GaussianModel.adaptive_densify_and_clone(inactive, .1, 1.)
    assert result == dict(eligible=0, added=0, blocked=0)
    assert len(inactive.births[0]) == 0


def test_appearance_birth_probe_is_read_only_and_excludes_validation_views():
    cfg = residual_test_config(birth_appearance_fraction=.25,
                              birth_appearance_camera_names=['cam01.png', 'cam02.png'])
    cfg.validate()
    model = SimpleNamespace(_xyz=torch.zeros(4, 3),
        _features_dc=torch.nn.Parameter(torch.zeros(4, 1, 3), requires_grad=False),
        get_scaling=torch.full((4, 3), .005), percent_dense=.01)
    model.get_xyz = model._xyz
    original_dc = model._features_dc
    original_binding = object()
    class ProbeRenderer:
        gaussians = original_binding
        def __call__(self, view, background):
            weights = torch.tensor([1., 3., 0., 2.])
            rgb = (self.gaussians._features_dc[:, 0] * weights[:, None]).sum(0)
            return dict(render=rgb[:, None, None].expand(3, 16, 16),
                        visibility_filter=torch.ones(4, dtype=torch.bool))
    views = []
    for name in ['cam01.png', 'cam02.png']:
        view = SimpleNamespace(path=Path(name), image=torch.ones(3, 16, 16))
        view.to = lambda device, view=view: view
        views.append(view)
    dataset = SimpleNamespace(items=views)
    def loader(random, indices):
        torch.rand(1)  # Simulate a worker seed without changing the main RNG.
        return [views[index] for index in indices]
    datamodule = SimpleNamespace(config=SimpleNamespace(training_camera_names=['cam01.png','cam02.png'],
        validation_camera_names=['cam00.png']), cameras_extent=1., get_train_dataset=lambda: dataset,
        _camera_indices=DATA.DataModule._camera_indices, train_dataloader=loader)
    module = SimpleNamespace(gaussians=model, renderer=ProbeRenderer(), datamodule=datamodule,
        residual_motion=SimpleNamespace(config=cfg), config=SimpleNamespace(lambda_dssim=.2),
        device=torch.device('cpu'), background=torch.zeros(3), trainer=SimpleNamespace(synchronize=lambda: None))
    before = torch.random.get_rng_state().clone()
    rows, report = BIRTH.collect_appearance_birth_sources(module)
    assert rows.tolist() == [1] and report['extra_renders'] == 2
    assert model._features_dc is original_dc and not original_dc.requires_grad and original_dc.grad is None
    assert module.renderer.gaussians is original_binding
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    class FailedRenderer(ProbeRenderer):
        def __call__(self, view, background):
            raise RuntimeError('probe failure')
    module.renderer = FailedRenderer()
    try:
        BIRTH.collect_appearance_birth_sources(module)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Probe failure did not propagate')
    assert model._features_dc is original_dc and module.renderer.gaussians is original_binding
    cfg.birth_appearance_camera_names = ['cam00.png']
    try:
        BIRTH.collect_appearance_birth_sources(module)
    except ValueError:
        pass
    else:
        raise AssertionError('Validation camera was allowed to select births')


def test_appearance_birth_config_and_old_checkpoint_defaults():
    for values in (dict(birth_appearance_fraction=-.1), dict(birth_appearance_fraction=1.1),
                   dict(birth_appearance_fraction=True), dict(birth_appearance_fraction=.1),
                   dict(birth_appearance_fraction=.1,birth_appearance_camera_names='cam'),
                   dict(birth_appearance_fraction=.1,birth_appearance_camera_names=['cam01.png','cam01.png'])):
        try:
            residual_test_config(**values).validate()
        except ValueError as error:
            assert 'birth_appearance' in str(error)
        else:
            raise AssertionError('Invalid appearance-birth configuration accepted')
    config = residual_test_config()
    raw = residual_raw_fixture()
    motion = Q.QuantizedResidualMotion(config)
    motion.setup(raw['xyz'], raw['rotation'], 1.)
    packet = copy.deepcopy(motion.capture())
    packet['config'].pop('birth_appearance_fraction')
    packet['config'].pop('birth_appearance_camera_names')
    restored = Q.QuantizedResidualMotion(config)
    restored.restore(packet)
    assert restored.config.birth_appearance_fraction == 0.


def test_sparse_color_reuse_changes_only_selected_rows_and_codec_is_exact():
    raw = residual_raw_fixture(12)
    update = COLOR.SparseColorUpdate(raw['sh_dc'], torch.tensor([5, 1, 8]), .001, .0375, .1)
    with torch.no_grad():
        update.delta[0] = torch.tensor([.0113, -.0036, 0.])
        update.delta[2] = torch.tensor([.1234, .0056, -.0087])
    color = update(raw['sh_dc'])
    changed = color.ne(raw['sh_dc']).any(-1).any(-1)
    assert torch.where(changed)[0].tolist() == [5, 8]
    color.sum().backward()
    torch.testing.assert_close(update.delta.grad, torch.ones_like(update.delta), rtol=0, atol=0)
    packet = update.encode()
    assert int(update.storage_bytes(hard=True)) == len(packet)
    decoded = COLOR.apply_sparse_color_packet(packet, raw)
    torch.testing.assert_close(decoded['sh_dc'], color.detach(), rtol=0, atol=0)
    for key in Q.RAW_FIELDS:
        if key != 'sh_dc':
            torch.testing.assert_close(decoded[key], raw[key], rtol=0, atol=0)
    for bad in (packet[:12], packet[:-2], packet[:COLOR.HEADER.size] + b'corrupt'):
        try:
            COLOR.apply_sparse_color_packet(bad, raw)
        except (ValueError, __import__('zlib').error):
            pass
        else:
            raise AssertionError('Corrupt sparse color stream was accepted')


def test_sparse_color_checkpoint_restores_next_optimizer_step_and_zero_cost():
    raw = residual_raw_fixture(8)
    update = COLOR.SparseColorUpdate(raw['sh_dc'], torch.tensor([1, 3]), .001, .02, .1)
    assert update.encode() == b'' and float(update.storage_bytes(hard=True)) == 0.
    loss = (update(raw['sh_dc']) - .2).square().mean()
    loss.backward(); update.optimizer_step()
    restored = COLOR.SparseColorUpdate.restore(copy.deepcopy(update.capture()), raw['sh_dc'])
    for obj in (update, restored):
        loss = (obj(raw['sh_dc']) - .2).square().mean()
        loss.backward(); obj.optimizer_step()
    torch.testing.assert_close(update.delta, restored.delta, rtol=0, atol=0)
    assert update.encode() == restored.encode()


def test_common_contrast_v5_requires_and_applies_sparse_color_before_state_hash():
    raw, motion = common_contrast_fixture(birth_appearance_action='color_update', component_modes=True)
    update = COLOR.SparseColorUpdate(raw['sh_dc'], torch.tensor([2, 4]), .001, .02, .1)
    with torch.no_grad():
        update.delta[:, 0] = .02
    xyz, rotation = motion(motion.config.motion_steps)
    moved = dict(raw, xyz=xyz, rotation=rotation)
    target = dict(moved, sh_dc=update(moved['sh_dc']).detach())
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)
        appearance = path / 'appearance.bin'
        update.export(appearance)
        stream = path / 'motion.bin'
        motion.export_motion(stream, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(target), previous_count=len(xyz), deleted_count=0,
            birth_count=0, appearance_present=True)
        assert motion.read_header(stream)['codec'] == 'common_contrast_sparse_color_v5'
        try:
            CC.CommonContrastDecoder(motion.config).advance(stream, None, raw)
        except ValueError as error:
            assert 'color stream' in str(error)
        else:
            raise AssertionError('Missing appearance stream was accepted')
        decoded, _ = CC.CommonContrastDecoder(motion.config).advance(stream, None, raw, appearance)
        assert Q.reference_digest(decoded) == Q.reference_digest(target)
    if torch.cuda.is_available():
        gpu = {key: value.cuda() for key, value in raw.items()}
        gpu_update = COLOR.SparseColorUpdate(gpu['sh_dc'], torch.tensor([2, 4]), .001, .02, .1)
        with torch.no_grad():
            gpu_update.delta[:, 0] = .02
        decoded = COLOR.apply_sparse_color_packet(gpu_update.encode(), gpu)
        torch.testing.assert_close(decoded['sh_dc'], gpu_update(gpu['sh_dc']), rtol=0, atol=0)


def test_common_motion_prior_inherits_only_reached_quantized_motion():
    raw, motion = common_contrast_fixture(motion_prior_decay=.6, direct_tree_enabled=True,
        direct_initial_levels=3, component_modes=True)
    with torch.no_grad():
        motion.root_translation[:, 0] = .03
        motion.contrast_translation[0][:, 1] = .05
        for gate in motion.refine_logits:
            gate.fill_(-20.)
    xyz, rotation = motion(motion.config.motion_steps)
    previous_root, *_ = motion._quantized_parameters()
    motion.committed = True
    motion.register_materialized_layout(xyz, torch.empty(0, dtype=torch.long))
    motion.setup(xyz, rotation, 1., frame_index=2, previous_rows=torch.arange(len(xyz)), previous_new_count=0)
    torch.testing.assert_close(motion.root_translation, previous_root[:, :3].detach()*.6, rtol=0, atol=0)
    assert all(not bool(value.any()) for value in motion.contrast_translation)
    assert motion.tree_setup['initialization_decay'] == .6


def test_pure_tree_split_preserves_existing_adam_moments():
    raw, source, evidence = node_edit_fixture(direct_preserve_optimizer_moments=True,
        direct_preserve_split_motion=True, direct_enable_swaps=False, direct_candidate_windows=0)
    source.prepare_step(source.last_step)
    loss = sum(parameter.sum() for parameter in source.motion_parameters())
    loss.backward(); source.optimizer_step()
    before = copy.deepcopy(source.motion_optimizer.state[source.root_translation])
    candidate, report = DIRECT_TREE.propose_local_batch(source, split_xyz=raw['xyz'], split_evidence=evidence)
    assert candidate is not None and report['optimizer_moments_preserved']
    for key in ('exp_avg','exp_avg_sq','step'):
        torch.testing.assert_close(candidate.motion_optimizer.state[candidate.root_translation][key], before[key], rtol=0, atol=0)
        torch.testing.assert_close(source.motion_optimizer.state[source.root_translation][key], before[key], rtol=0, atol=0)


def test_motion_entropy_envelope_roundtrip_and_dispatch_preserve_old_stream():
    raw, motion = common_contrast_fixture(motion_entropy_compression=True)
    xyz, rotation = motion(motion.config.motion_steps)
    target = dict(raw, xyz=xyz, rotation=rotation)
    args = dict(frame_index=1, previous_state_digest=Q.reference_digest(raw),
        post_motion_digest=Q.reference_digest(target), previous_count=len(xyz), deleted_count=0, birth_count=0)
    packet, report = motion.encode_motion(**args)
    assert packet.startswith(CC.ENTROPY_MAGIC) and report['entropy_compressed']
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'motion.bin'
        path.write_bytes(packet)
        decoded, _ = SAB.decoder_for_binary_stream(path,motion.config).advance(path,None,raw)
        assert Q.reference_digest(decoded) == Q.reference_digest(target)
    try:
        motion.read_header(packet[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError('Truncated entropy stream was accepted')


def test_batched_bit_reader_preserves_unaligned_fields_and_end_checks():
    widths = [1, 2, 16, 32, 3, 5, 31, 1, 16, 2, 7, 32]
    values = [(index * 198731 + 95127) & ((1 << width) - 1)
              for index, width in enumerate(widths)]
    writer = SAB._BitWriter()
    for value, width in zip(values, widths):
        writer.write(value, width)
    reader = SAB._BitReader(writer.finish(), sum(widths))
    assert [reader.read(width) for width in widths] == values
    assert reader.cursor == sum(widths)
    try:
        reader.read()
    except ValueError:
        pass
    else:
        raise AssertionError('Reader accepted a field beyond the declared bit count')


def test_batched_effective_splits_match_scalar_weights_with_empty_children():
    raw, source = common_contrast_fixture()
    original = source.active_layout
    removed_child = original['parents'][0].tolist().index(0)
    keep = original['index'][1] != removed_child
    keep[1::7] = False
    subset = torch.where(keep)[0]
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for dtype in (torch.float32, torch.float64):
            layout = BR._binary_layout(original['index'][:, subset].to(device),
                original['counts'], original['parents'], raw['xyz'][subset].to(device=device, dtype=dtype),
                [value.to(device=device, dtype=dtype) for value in original['pivots']])
            occupancy, children, members, positions, binary, unary, weights = CC._effective_splits(layout)
            for level, groups in enumerate(children):
                expected = torch.zeros_like(weights[level])
                for node, group in enumerate(groups):
                    occupied = [child for child in group if bool(occupancy[level + 1][child])]
                    assert bool(binary[level][node]) == (len(occupied) == 2)
                    assert int(unary[level][node]) == (occupied[0] if len(occupied) == 1 else -1)
                    if len(occupied) == 2:
                        left, right = occupied
                        total = members[level + 1][left] + members[level + 1][right]
                        expected[positions[level][node], 0] = members[level + 1][left] / total
                        expected[positions[level][node], 1] = members[level + 1][right] / total
                torch.testing.assert_close(weights[level], expected, rtol=0, atol=0)


def test_lossless_birth_packet_replays_attributes_and_tree_attachment():
    birth_codec = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_codec')
    raw, motion = common_contrast_fixture(birth_export_format='raw_zlib')
    xyz, rotation = motion(motion.config.motion_steps)
    base = dict(raw, xyz=xyz, rotation=rotation)
    births = {name: value[:3].detach().clone() for name, value in base.items()}
    births['xyz'] += .0001
    final = {name: torch.cat((base[name], births[name])) for name in Q.RAW_FIELDS}
    layout = motion.register_materialized_layout(final['xyz'], torch.arange(3))
    packet = birth_codec.encode_birth_packet(births, motion.birth_layout_sources, len(xyz))
    assert birth_codec.HEADER.size + birth_codec.HEADER.unpack_from(packet)[4] == birth_codec.raw_birth_size(3, 1)
    decoded_births, rows = birth_codec.decode_birth_packet(packet, base)
    for name in Q.RAW_FIELDS:
        torch.testing.assert_close(decoded_births[name], births[name], rtol=0, atol=0)
    assert torch.equal(rows, motion.birth_layout_sources)
    with tempfile.TemporaryDirectory() as temporary:
        birth_path, path = Path(temporary)/'birth.bin', Path(temporary)/'motion.bin'
        birth_path.write_bytes(packet)
        motion.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(base), previous_count=len(xyz), deleted_count=0, birth_count=3)
        decoder = SAB.decoder_for_binary_stream(path, motion.config)
        actual, _ = decoder.advance(path, birth_path, raw)
        for name in Q.RAW_FIELDS:
            torch.testing.assert_close(actual[name], final[name], rtol=0, atol=0)
        assert SAB.layout_digest(decoder.materialized_layout) == SAB.layout_digest(layout)
    for invalid in (packet[:-1], packet + b'extra'):
        try:
            birth_codec.decode_birth_packet(invalid, base)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid lossless birth packet accepted')


def test_bincount_pivots_match_sequential_add_at_for_mixed_magnitudes():
    import numpy as np
    torch.manual_seed(917)
    xyz = torch.randn(48, 3).double()
    xyz[::3] *= 1e8
    xyz[1::3] *= 1e-8
    _, source = common_contrast_fixture()
    old = source.active_layout
    layout = BR._binary_layout(old['index'], old['counts'], old['parents'], xyz)
    for level, count in enumerate(old['counts']):
        owners = old['index'][level].numpy()
        valid = owners >= 0
        sums = np.zeros((count, 3), dtype=np.float64)
        np.add.at(sums, owners[valid], xyz.numpy()[valid])
        sizes = np.bincount(owners[valid], minlength=count)[:, None]
        expected = torch.from_numpy(sums / np.maximum(sizes, 1))
        torch.testing.assert_close(layout['pivots'][level], expected, rtol=0, atol=0)


def test_simple_clone_uses_recongs_jitter_and_raw_initialization():
    scales = torch.tensor([[.02, .001, .003], [.01, .002, .004]], dtype=torch.float64)
    rotation = torch.tensor([[2 ** -.5, 0., 0., 2 ** -.5], [1., 0., 0., 0.]], dtype=torch.float64)
    model = SimpleNamespace(_xyz=torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64),
        xyz_gradient_accum=torch.ones(2, 1), denom=torch.ones(2, 1), percent_dense=.1,
        get_scaling=scales, _features_dc=torch.zeros(2, 1, 3, dtype=torch.float64),
        _features_rest=torch.zeros(2, 3, 3, dtype=torch.float64), _opacity=torch.zeros(2, 1, dtype=torch.float64),
        _scaling=scales.log(), _rotation=rotation)
    model.densification_postfix = lambda *values: setattr(model, 'births', values)
    torch.manual_seed(377)
    expected_legacy = torch.normal(model._xyz, scales * .5)
    torch.manual_seed(377)
    report = GAUSS.GaussianModel.adaptive_densify_and_clone(model, .1, 1., sigma_scale=.5)
    assert report == dict(eligible=2, added=2, blocked=0)
    torch.testing.assert_close(model.births[0], expected_legacy, rtol=0, atol=0)
    torch.testing.assert_close(model.births[3], torch.full((2, 1), .1,
                                dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(model.births[4], model._scaling, rtol=0, atol=0)
    torch.testing.assert_close(model.births[-1], torch.tensor([[1., 0., 0., 0.]],
                                dtype=torch.float64).repeat(2, 1), rtol=0, atol=0)


def test_split_residual_matches_covariance_only_position_sharing():
    xyz = torch.tensor([[-1., 0., 0.], [1., 0., 0.], [0., -1., 0.],
                        [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]], dtype=torch.float64)
    translation = torch.tensor([.2, -.3, .1], dtype=torch.float64)
    omega = torch.tensor([.1, .2, .3], dtype=torch.float64)
    gradient = translation + torch.cross(omega.expand_as(xyz), xyz, dim=-1)
    supported = torch.ones(len(xyz), dtype=torch.bool)
    evidence = SPLIT_SCORES.PositionGradientEvidence(gradient,
        torch.full((len(xyz),), 2, dtype=torch.long), supported)
    owners = torch.zeros(len(xyz), dtype=torch.long)
    rigid = SPLIT_SCORES.rigid_gradient_residual(xyz, evidence, owners, region_count=1)
    covariance_only = SPLIT_SCORES.rigid_gradient_residual(xyz, evidence, owners,
        region_count=1, rotation_moves_positions=False)
    torch.testing.assert_close(rigid.mean, torch.zeros_like(gradient), rtol=0, atol=1e-14)
    torch.testing.assert_close(covariance_only.mean, gradient - translation, rtol=0, atol=1e-14)
    assert float(covariance_only.mean.norm()) > 0
    constant = SPLIT_SCORES.PositionGradientEvidence(translation.expand_as(xyz),
        evidence.visible_views, supported)
    for rotates in (False, True):
        residual = SPLIT_SCORES.rigid_gradient_residual(xyz, constant, owners,
            region_count=1, rotation_moves_positions=rotates)
        torch.testing.assert_close(residual.mean, torch.zeros_like(gradient), rtol=0, atol=0)


def test_simple_birth_training_rejects_old_extensions_without_breaking_decoder_config():
    raw, motion = common_contrast_fixture(component_modes=True)
    cfg = motion.config
    cfg.require_simple_birth_training()
    for name, value in [('birth_appearance_fraction', .002),
                        ('birth_appearance_action', 'color_update'),
                        ('birth_residual_enabled', True),
                        ('birth_lifetime_enabled', True),
                        ('birth_clone_geometry', 'parent_covariance')]:
        old = getattr(cfg, name)
        setattr(cfg, name, value)
        try:
            cfg.require_simple_birth_training()
        except ValueError as error:
            assert 'position-gradient cloning only' in str(error)
        else:
            raise AssertionError(f'Old birth experiment option remained trainable: {name}')
        setattr(cfg, name, old)
    # Old settings remain valid when a v6/v7 reader replays an existing stream.
    cfg.birth_lifetime_enabled = True
    cfg.validate()


def test_residual_birth_pixel_centers_roundtrip_row_camera_matrices():
    residual_birth = importlib.import_module('pipeline.ReCon-GS.archive.model.residual_birth')
    projection = torch.zeros(4, 4, dtype=torch.float64)
    projection[0, 0] = projection[1, 1] = projection[2, 2] = projection[2, 3] = 1.
    world = torch.eye(4, dtype=torch.float64)
    world[3, :3] = torch.tensor([.3, -.2, .1])
    view = SimpleNamespace(image=torch.zeros(3, 6, 8, dtype=torch.float64),
        fovx=math.pi / 2, fovy=math.pi / 2, world_view_transform=world,
        full_proj_transform=world @ projection)
    pixels = torch.tensor([[0., 0.], [7., 5.], [3., 2.]], dtype=torch.float64)
    depth = torch.tensor([1., 2., 3.], dtype=torch.float64)
    xyz = residual_birth.backproject_pixels(pixels, depth, view)
    ndc, eye_z = residual_birth.project_points(xyz, view)
    expected = (pixels + .5) * torch.tensor([2 / 8, 2 / 6]) - 1
    torch.testing.assert_close(ndc, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(eye_z, depth, rtol=0, atol=1e-12)


def test_residual_birth_multiview_support_rejects_occlusion_and_duplicate_peaks():
    residual_birth = importlib.import_module('pipeline.ReCon-GS.archive.model.residual_birth')
    projection = torch.zeros(4, 4)
    projection[0, 0] = projection[1, 1] = projection[2, 2] = projection[2, 3] = 1.
    rgb = torch.tensor([.4, .5, .6])[:, None, None].expand(3, 8, 8).clone()
    view = SimpleNamespace(image=rgb, world_view_transform=torch.eye(4), full_proj_transform=projection)
    package = (view, torch.full((8, 8), .3), torch.full((8, 8), 3.), torch.ones(8, 8))
    xyz = torch.tensor([[0., 0., 2.], [.001, 0., 2.], [0., 0., 4.], [100., 0., 2.]])
    scores, colors, support = residual_birth.multiview_birth_evidence(xyz, [package, package], .1, .05, .03)
    assert support.tolist() == [2, 2, 0, 0]
    torch.testing.assert_close(colors[:2], torch.tensor([[.4, .5, .6]]).repeat(2, 1), rtol=0, atol=1e-7)
    selected = residual_birth.select_spaced_births(xyz, torch.full((4,), .01), scores, support, 2, 4, 1.5)
    assert selected.tolist() == [0]
    peaks = residual_birth.grid_error_peaks(torch.full((8, 8), .3), 4, 4, .1)
    assert peaks.tolist() == [[0., 0.], [4., 0.], [0., 4.], [4., 4.]]


def test_birth_opacity_stream_updates_only_selected_rows_and_resumes_adam():
    lifetime = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_lifetime')
    raw = residual_raw_fixture(7)
    update = lifetime.BirthOpacityUpdate(raw['opacity'], torch.tensor([2, 5]), .01, .1, .1)
    with torch.no_grad():
        update.delta[:] = torch.tensor([[.015], [-2.007]])
    decoded = lifetime.apply_opacity_packet(update.encode(), raw)
    expected = update(raw['opacity'])
    torch.testing.assert_close(decoded['opacity'], expected, rtol=0, atol=0)
    for name in Q.RAW_FIELDS:
        if name != 'opacity':
            assert torch.equal(decoded[name], raw[name])
    assert torch.equal(decoded['opacity'][[0, 1, 3, 4, 6]], raw['opacity'][[0, 1, 3, 4, 6]])
    update.values().square().sum().backward()
    update.optimizer_step()
    restored = lifetime.BirthOpacityUpdate.restore(copy.deepcopy(update.capture()), raw['opacity'])
    for model in (update, restored):
        model.values().square().sum().backward()
        model.optimizer_step()
    torch.testing.assert_close(update.delta, restored.delta, rtol=0, atol=0)


def test_birth_expiration_prioritizes_dead_births_and_preserves_visible_keep():
    lifetime = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_lifetime')
    opacity = torch.tensor([[-20.], [-20.], [1.], [-20.], [-20.], [-20.]])
    origins = torch.tensor([0, 0, 0, 1, 1, 2])
    changed = torch.zeros(5, dtype=torch.bool)
    deleted, report = lifetime.expiration_deletions(opacity, 1, changed, origins, .0039)
    assert deleted.tolist() == [3, 4, 5]
    assert report['expired_births'] == 3 and report['deleted_keep_count'] == 2
    opacity[3:] = 1.
    deleted, report = lifetime.expiration_deletions(opacity, 1, changed, origins, .0039)
    assert deleted.tolist() == [0] and report['expired_births'] == 0
    assert report['capacity_deleted_invisible_first_frame'] == 1


def test_v6_opacity_packet_and_expiration_replay_preserve_state_and_layout():
    lifetime = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_lifetime')
    birth_codec = importlib.import_module('pipeline.ReCon-GS.archive.model.birth_codec')
    raw, motion = common_contrast_fixture(component_modes=True, birth_lifetime_enabled=True,
                                          birth_export_format='raw_zlib')
    raw['opacity'][:4] = -20.
    decoder = CC.CommonContrastDecoder(motion.config)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        origins = torch.zeros(len(raw['xyz']), dtype=torch.long)
        previous_birth_count = 0
        for frame in range(1, 4):
            previous = {name: value.detach().clone() for name, value in raw.items()}
            if frame > 1:
                deleted, _ = motion.compensation_deletions(raw['opacity'], previous_birth_count, origins)
                rows = Q.survivor_rows(len(raw['xyz']), deleted)
                raw = {name: value[rows] for name, value in raw.items()}
                origins = origins[rows]
                motion.setup(raw['xyz'], raw['rotation'], 1., frame_index=frame,
                    previous_rows=rows, previous_new_count=previous_birth_count)
            else:
                deleted = torch.empty(0, dtype=torch.long)
            xyz, rotation = motion(motion.config.motion_steps)
            base = dict(raw, xyz=xyz, rotation=rotation)
            update_path = None
            if frame == 2:
                update = lifetime.BirthOpacityUpdate(base['opacity'], torch.where(origins > 0)[0], .01, .1, .1)
                with torch.no_grad():
                    update.delta.fill_(-20.)
                base = dict(base, opacity=update(base['opacity']).detach())
                update_path = folder/f'opacity_{frame}.bin'
                update.export(update_path)
            born_count = 2 if frame < 3 else 0
            births = {name: value[:born_count].detach().clone() for name, value in base.items()}
            if frame == 1:
                births['opacity'].zero_()
            if frame == 2:
                births['opacity'].fill_(-20.)
            final = {name: torch.cat((base[name], births[name])) for name in Q.RAW_FIELDS}
            layout = motion.register_materialized_layout(final['xyz'], torch.arange(born_count))
            birth_path = None
            if born_count:
                birth_path = folder/f'birth_{frame}.bin'
                birth_codec.export_birth_packet(birth_path, births, motion.birth_layout_sources, len(base['xyz']))
            path = folder/f'motion_{frame}.bin'
            motion.export_motion(path, frame_index=frame, previous_state_digest=Q.reference_digest(previous),
                post_motion_digest=Q.reference_digest(base), previous_count=len(previous['xyz']),
                deleted_count=len(deleted), birth_count=born_count, opacity_present=update_path is not None)
            actual, info = decoder.advance(path, birth_path, previous, opacity_path=update_path)
            for name in Q.RAW_FIELDS:
                torch.testing.assert_close(actual[name], final[name], rtol=0, atol=0)
            assert SAB.layout_digest(decoder.materialized_layout) == SAB.layout_digest(layout)
            assert torch.equal(decoder.birth_frames[:len(origins)], origins)
            if frame == 3:
                assert len(info['deleted_rows']) == 4 > previous_birth_count
            motion.committed = True
            raw = final
            origins = torch.cat((origins, torch.full((born_count,), frame, dtype=torch.long)))
            previous_birth_count = born_count


def test_covariance_only_rotation_preserves_positions_and_decodes_v7():
    raw, motion = common_contrast_fixture(component_modes=True, rotation_moves_positions=False)
    with torch.no_grad():
        motion.root_rotation[:, 2] = .7
    xyz, rotation = motion(motion.config.motion_steps)
    torch.testing.assert_close(xyz, raw['xyz'], rtol=0, atol=0)
    assert not torch.equal(rotation, raw['rotation'])
    target = dict(raw, xyz=xyz, rotation=rotation)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'motion.bin'
        motion.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(target), previous_count=len(xyz), deleted_count=0, birth_count=0)
        header = motion.read_header(path)
        assert header['codec'] == 'common_contrast_covariance_rotation_v7'
        assert not header['rotation_moves_positions']
        actual, _ = SAB.decoder_for_binary_stream(path, motion.config).advance(path, None, raw)
        for name in Q.RAW_FIELDS:
            torch.testing.assert_close(actual[name], target[name], rtol=0, atol=0)


def test_v7_decoder_honors_old_rigid_stream_position_semantics():
    raw, motion = common_contrast_fixture(component_modes=True)
    with torch.no_grad():
        motion.root_rotation[:, 2] = .7
    xyz, rotation = motion(motion.config.motion_steps)
    assert not torch.equal(xyz, raw['xyz'])
    target = dict(raw, xyz=xyz, rotation=rotation)
    config = copy.deepcopy(motion.config)
    config.rotation_moves_positions = False
    config.validate()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary)/'motion.bin'
        motion.export_motion(path, frame_index=1, previous_state_digest=Q.reference_digest(raw),
            post_motion_digest=Q.reference_digest(target), previous_count=len(xyz), deleted_count=0, birth_count=0)
        actual, _ = SAB.decoder_for_binary_stream(path, config).advance(path, None, raw)
        for name in Q.RAW_FIELDS:
            torch.testing.assert_close(actual[name], target[name], rtol=0, atol=0)


def test_canonical_camera_order_pairs_differently_shuffled_frame_datasets():
    class CameraNames(torch.utils.data.Dataset):
        def __init__(self, names):
            self.items = [SimpleNamespace(path=Path(name), frame=3) for name in names]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index].path.name

    names = [f'cam{index:02d}.png' for index in range(1, 21)]
    data = DATA.DataModule.__new__(DATA.DataModule)
    data.current_frame = 3
    data.config = SimpleNamespace(training_view_seed=0,
        canonical_training_view_order=True,
        extra_dataloader_kwargs=dict(num_workers=0, pin_memory=False))
    sequences = []
    for order in (names, list(reversed(names))):
        dataset = CameraNames(order)
        data.get_train_dataset = lambda scale: dataset
        loader = data.train_dataloader()
        fetcher = DATA.PairedFetcher(loader, True, None, seed=3)
        sequences.append([fetcher.next() for _ in range(60)])
        assert [item.path.name for item in dataset.items] == order
    assert sequences[0] == sequences[1]
    assert all(sorted(sequences[0][offset:offset + 20]) == names
               for offset in (0, 20, 40))
    assert list(data.train_dataloader(random=False, indices=[])) == names
    data.config.training_camera_names = ['cam20.png', 'cam01.png']
    data.config.validation_camera_names = ['cam11.png']
    assert list(data.train_dataloader(random=False)) == ['cam01.png', 'cam20.png']
    data.config.training_camera_names = None
    data.config.validation_camera_names = None
    # Default/legacy order stays tied to the original dataset permutation.
    data.config.canonical_training_view_order = False
    legacy = []
    for order in (names, list(reversed(names))):
        dataset = CameraNames(order)
        data.get_train_dataset = lambda scale: dataset
        legacy.append(list(data.train_dataloader()))
    assert legacy[0] != legacy[1]


def test_canonical_training_order_rejects_missing_sampling_seed():
    data = DATA.DataModule.__new__(DATA.DataModule)
    data.current_frame = 3
    data.config = SimpleNamespace(training_view_seed=None,
        canonical_training_view_order=True,
        extra_dataloader_kwargs=dict(num_workers=0, pin_memory=False))
    data.get_train_dataset = lambda scale: list(range(20))
    try:
        data.train_dataloader()
    except ValueError as error:
        assert 'training_view_seed' in str(error)
    else:
        raise AssertionError('Canonical sampling must have an explicit seed')


def test_rigid_birth_source_rows_follow_split_and_prune():
    if not torch.cuda.is_available():
        return  # libgs' split operation allocates directly on CUDA.
    torch.manual_seed(17)
    device = 'cuda'
    model = ACTIVE_GAUSS.BirthGaussianModel(0)
    values = dict(
        _xyz=torch.tensor([[0., 0., 0.], [5., 0., 0.]], device=device),
        _features_dc=torch.zeros(2, 1, 3, device=device),
        _features_rest=torch.zeros(2, 0, 3, device=device),
        _opacity=torch.zeros(2, 1, device=device),
        _scaling=torch.log(torch.tensor([[.2, .2, .2], [.01, .01, .01]], device=device)),
        _rotation=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], device=device),
    )
    for name, value in values.items():
        setattr(model, name, torch.nn.Parameter(value))
    model.source_rows = torch.tensor([4, 8], device=device)
    model.training_setup(SimpleNamespace(
        percent_dense=.1, position_lr_init=.01, position_lr_final=.001,
        position_lr_delay_mult=1., position_lr_max_steps=100,
        feature_lr=.01, opacity_lr=.01, scaling_lr=.01, rotation_lr=.01))
    model.max_radii2D = torch.zeros(2, device=device)
    stats = model.adaptive_densify_and_split(
        torch.tensor([2., 0.], device=device), 1., 1., N=2)
    assert stats['added'] == 1
    torch.testing.assert_close(model.source_rows.cpu(), torch.tensor([8, 4, 4]),
                               rtol=0, atol=0)
    model.prune_points(torch.tensor([1], device=device))
    torch.testing.assert_close(model.source_rows.cpu(), torch.tensor([8, 4]),
                               rtol=0, atol=0)
    model.prune_points(torch.tensor([0], device=device))
    torch.testing.assert_close(model.source_rows.cpu(), torch.tensor([4]),
                               rtol=0, atol=0)


def rigid_tree_fixture(**overrides):
    options = dict(region_count=3, motion_steps=100, birth_steps=100,
                   tree_edit_interval=10, tree_edit_step=60,
                   tree_edit_neighbors=2, motion_xyz_step_extent=.01,
                   motion_rotation_step_rad=.01)
    options.update(overrides)
    model = RIGID.RigidResidualTree(RIGID.RigidTreeConfig(**options))
    xyz = torch.tensor([[-2., 0., 0.], [-1., 0., 0.],
                        [1., 0., 0.], [2., 0., 0.]])
    raw_q = torch.tensor([[1., 0., 0., 0.]] * len(xyz))
    model.setup(xyz, raw_q, 1., frame_index=1)
    return model


def test_rigid_tree_parent_child_pose_is_ordered_about_distinct_pivots():
    dtype = torch.float64
    xyz = torch.tensor([[2., 0., 0.]], dtype=dtype)
    base_q = RIGID.rotation_vector_quaternion(
        torch.tensor([[0., torch.pi / 2, 0.]], dtype=dtype))
    owner = torch.tensor([2])
    parents = torch.tensor([-1, 0, 1])
    pivots = torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]], dtype=dtype)
    translation = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=dtype)
    rotation = torch.tensor([[0., 0., 0.], [0., 0., torch.pi / 2],
                             [torch.pi / 2, 0., 0.]], dtype=dtype)
    zeros = torch.zeros(1, 3, dtype=dtype)
    moved, q = RIGID.compose_pose(xyz, base_q, owner, parents, pivots,
                                  translation, rotation, zeros, zeros)
    torch.testing.assert_close(moved, torch.tensor([[1., 2., 1.]], dtype=dtype),
                               rtol=0, atol=1e-12)
    rotated_x = RIGID.quaternion_rotate(q, torch.tensor([[1., 0., 0.]], dtype=dtype))
    torch.testing.assert_close(rotated_x, torch.tensor([[0., 1., 0.]], dtype=dtype),
                               rtol=0, atol=1e-12)


def test_rigid_tree_zero_pose_is_exact_identity_with_nonzero_gradients():
    model = rigid_tree_fixture()
    xyz, q = model.forward(1, training=True)
    assert torch.equal(xyz, model.base_xyz)
    assert torch.equal(q, model.base_rotation)
    (xyz[:, 1].sum() + q[:, 2].sum()).backward()
    assert model.region_rotation.grad is not None
    assert model.region_rotation.grad[1:].abs().sum() > 0
    assert model.gs_translation.grad.abs().sum() > 0
    assert model.gs_rotation.grad.abs().sum() > 0


def test_rigid_tree_zero_local_residual_follows_shared_and_root_stays_static():
    model = rigid_tree_fixture()
    with torch.no_grad():
        model.region_translation[1, 0] = .25
        model.region_translation[2, 1] = -.15
        model.owner[0] = 0
    moved, q = model.forward(1)
    torch.testing.assert_close(moved[0], model.base_xyz[0], rtol=0, atol=0)
    torch.testing.assert_close(q[0], model.base_rotation[0], rtol=0, atol=0)
    owner_one = (model.owner == 1).nonzero(as_tuple=True)[0]
    assert len(owner_one)
    torch.testing.assert_close(moved[owner_one] - model.base_xyz[owner_one],
                               torch.tensor([[.25, 0., 0.]]).expand(len(owner_one), -1),
                               rtol=0, atol=1e-6)


def test_rigid_tree_stage_and_first_layout_are_deterministic():
    model = rigid_tree_fixture(region_count=6)
    assert model.stage(60) == 'motion_fit'  # Frame 1 cannot edit the tree.
    model.frame_index = 10
    assert [model.stage(step) for step in (59, 60, 61, 70, 100)] == [
        'motion_fit', 'tree_edit', 'motion_fit', 'motion_fit', 'motion_fit']
    owner, parents = RIGID.initial_layout(model.base_xyz, 6)
    torch.testing.assert_close(owner, model.owner, rtol=0, atol=0)
    torch.testing.assert_close(parents, model.parents, rtol=0, atol=0)
    assert owner.tolist() == [1, 3, 2, 4]
    assert parents.tolist() == [-1, 0, 0, 0, 0, 0]
    assert model.update_topology(59) == []


def test_rigid_tree_translation_rotation_symbols_and_relative_gradients_are_independent():
    model = rigid_tree_fixture()
    with torch.no_grad():
        model.region_translation[1, 0] = .02
        model.region_rotation[2, 2] = .03
        model.gs_translation[0, 1] = .04
        model.gs_rotation[3, 0] = .05
    rt, rr, gt, gr = model._values(hard=True)
    assert rt[1].tolist() == [2, 0, 0] and not bool(rr[1].any())
    assert rr[2].tolist() == [0, 0, 3] and not bool(rt[2].any())
    assert gt[0].tolist() == [0, 4, 0] and not bool(gr[0].any())
    assert gr[3].tolist() == [5, 0, 0] and not bool(gt[3].any())
    rate = model.loss_terms(60)['rate_loss']
    assert not rate.requires_grad
    model.loss_terms(60)['relative_motion_loss'].backward()
    for name in ('region_translation', 'region_rotation'):
        grad = getattr(model, name).grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    assert model.gs_translation.grad is None
    assert model.gs_rotation.grad is None
    diagnostics = model.loss_terms(60)
    assert diagnostics['gs_translation_active_blocks'] == 1
    assert diagnostics['gs_rotation_active_blocks'] == 1


def test_rigid_tree_region_proximal_selects_four_sparse_block_choices():
    model = rigid_tree_fixture(region_count=5, rate_loss_weight=1.,
                               motion_xyz_step_extent=1e-4,
                               motion_rotation_step_rad=1e-4)
    with torch.no_grad():
        model.region_translation[0, 0] = .5
        model.region_rotation[0, 0] = .4
        model.region_translation[1:, 0] = torch.tensor([.2, .2, .0002, .0002])
        model.region_rotation[1:, 0] = torch.tensor([.1, .0002, .1, .0002])
    for parameter, strong_rows in ((model.region_translation, (1, 2)),
                                   (model.region_rotation, (1, 3))):
        state = model.optimizer.state[parameter]
        state['step'] = torch.tensor(1.)
        state['exp_avg'] = torch.zeros_like(parameter)
        state['exp_avg_sq'] = torch.full_like(parameter, 1e-12)
        state['exp_avg_sq'][list(strong_rows)] = 1.
    model.last_step = 60
    model._proximal_region_blocks()
    torch.testing.assert_close(model.region_translation[0, 0], torch.tensor(.5),
                               rtol=0, atol=0)
    torch.testing.assert_close(model.region_rotation[0, 0], torch.tensor(.4),
                               rtol=0, atol=0)
    assert model.region_translation[1, 0] == .2
    assert model.region_rotation[1, 0] == .1
    assert model.region_translation[2, 0] == .2
    assert not bool(model.region_rotation[2].any())
    assert not bool(model.region_translation[3].any())
    assert model.region_rotation[3, 0] == .1
    assert not bool(model.region_translation[4].any())
    assert not bool(model.region_rotation[4].any())
    assert model.last_metrics['motion/prox_zero_region_translation_blocks'] == 2
    assert model.last_metrics['motion/prox_zero_region_rotation_blocks'] == 2


def test_rigid_tree_region_subgrid_motion_accumulates_through_proximal_step():
    model = rigid_tree_fixture(rate_loss_weight=.01,
                               region_translation_lr=.001)
    for _ in range(3):
        model.last_step = 60
        model.region_translation.grad = torch.zeros_like(model.region_translation)
        model.region_translation.grad[1, 0] = -1.
        model.optimizer_step()
    assert 0 < float(model.region_translation[1, 0]) < .005
    assert not bool(model._values(hard=True)[0][1].any())
    for _ in range(3):
        model.region_translation.grad = torch.zeros_like(model.region_translation)
        model.region_translation.grad[1, 0] = -1.
        model.optimizer_step()
    assert model.region_translation[1, 0] > .005
    assert bool(model._values(hard=True)[0][1].any())


def test_rigid_tree_shared_steps_still_train_region_with_sparse_proximal():
    model = rigid_tree_fixture(motion_steps=6, birth_steps=2,
                               shared_steps=3, tree_edit_step=5,
                               rate_warmup_steps=2,
                               motion_xyz_step_extent=1e-4)
    xyz, _ = model.forward(1, training=True)
    (xyz - model.base_xyz - .1).square().sum().backward()
    model.optimizer_step()
    assert bool(model.region_translation[1:].ne(0).any())
    assert not bool(model.gs_translation.any())
    assert model.gs_translation not in model.optimizer.state


def test_rigid_tree_proximal_rate_prunes_weak_gs_blocks():
    model = rigid_tree_fixture(rate_loss_weight=1.)
    with torch.no_grad():
        model.gs_translation[0, 1] = .01
        model.gs_rotation[1, 0] = .01
    for parameter in (model.gs_translation, model.gs_rotation):
        state = model.optimizer.state[parameter]
        state['step'] = torch.tensor(1.)
        state['exp_avg'] = torch.zeros_like(parameter)
        state['exp_avg_sq'] = torch.full_like(parameter, 1e-12)
    model.last_step = 60
    model._proximal_gs_blocks()
    assert not bool(model.gs_translation.any())
    assert not bool(model.gs_rotation.any())
    assert model.last_metrics['motion/prox_zero_translation_blocks'] >= 1
    assert model.last_metrics['motion/prox_zero_rotation_blocks'] >= 1


def test_rigid_tree_quantized_pose_has_ste_gradient():
    model = rigid_tree_fixture()
    with torch.no_grad():
        model.region_translation[1, 0] = .013
        model.region_rotation[1, 2] = .013
    rt, rr, _, _ = model._values()
    torch.testing.assert_close(rt[1, 0], torch.tensor(.01), rtol=0, atol=1e-8)
    torch.testing.assert_close(rr[1, 2], torch.tensor(.01), rtol=0, atol=1e-8)
    xyz, q = model.forward(1, training=True)
    (xyz[:, 0].sum() + xyz[:, 1].sum() + q[:, 3].sum()).backward()
    assert model.region_translation.grad[1, 0].abs() > 0
    assert model.region_rotation.grad[1, 2].abs() > 0


def test_rigid_tree_cuda_pose_and_packet_match_cpu_when_available():
    if not torch.cuda.is_available():
        return
    model = rigid_tree_fixture()
    with torch.no_grad():
        model.region_translation[1, 0] = .02
        model.region_rotation[2, 2] = .03
        model.gs_translation[0, 1] = -.04
        model.gs_rotation[3, 0] = .05
    cpu_xyz, cpu_q = model.forward(60)
    gpu = RIGID.RigidResidualTree(model.config)
    gpu.setup(model.base_xyz.cuda(), model.base_rotation.cuda(), 1., frame_index=1)
    with torch.no_grad():
        for name in ('region_translation', 'region_rotation',
                     'gs_translation', 'gs_rotation'):
            getattr(gpu, name).copy_(getattr(model, name).cuda())
    gpu_xyz, gpu_q = gpu.forward(60)
    torch.testing.assert_close(gpu_xyz.cpu(), cpu_xyz, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gpu_q.cpu(), cpu_q, rtol=1e-6, atol=1e-6)
    values = model._values(hard=True)
    prior = model.codec_previous_state
    args = (prior, 1, 1., torch.arange(4), model.owner, model.parents,
            *values, torch.empty(0, dtype=torch.long), torch.empty(0, 3),
            torch.empty(0, 4), .01, .01)
    cpu_packet = RIGID_CODEC.encode_motion(*args)
    gpu_args = [value.cuda() if isinstance(value, torch.Tensor) else value
                for value in args]
    assert RIGID_CODEC.encode_motion(*gpu_args) == cpu_packet


def test_rigid_tree_owner_reassignment_preserves_decoded_target():
    model = rigid_tree_fixture()
    model.frame_index = 10
    original_owner = model.owner.clone()
    row = int((model.owner == 1).nonzero(as_tuple=True)[0][0])
    with torch.no_grad():
        model.region_translation[2, 0] = .2
        model.gs_translation[row, 0] = .2
    before_xyz, before_q = (value.detach().clone() for value in model.forward(60))
    edits = model.update_topology(60)
    assert any(edit['kind'] == 'owner' and edit['row'] == row
               for edit in edits)
    assert int(model.owner[row]) != int(original_owner[row])
    after_xyz, after_q = model.forward(60)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=.0051)
    torch.testing.assert_close(after_q, before_q, rtol=0, atol=.0051)


def test_rigid_tree_reparent_deduplicates_shared_rigid_motion_without_pose_drift():
    model = rigid_tree_fixture()
    model.frame_index = 10
    with torch.no_grad():
        model.region_translation[1, 0] = .2
        model.region_translation[2, 0] = .2
    before_xyz, before_q = (value.detach().clone() for value in model.forward(60))
    edits = model.update_topology(60)
    parents = [edit for edit in edits if edit['kind'] == 'parent']
    assert len(parents) == 1
    node, parent = parents[0]['node'], parents[0]['parent']
    assert int(model.parents[node]) == parent
    assert len(RIGID._topological_order(model.parents)) == len(model.parents) - 1
    # The new child transform is exactly zero and inherits its parent's +0.2.
    torch.testing.assert_close(model.region_translation[node], torch.zeros(3),
                               rtol=0, atol=1e-6)
    torch.testing.assert_close(model.region_translation[parent, 0], torch.tensor(.2),
                               rtol=0, atol=1e-6)
    after_xyz, after_q = model.forward(60)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=.0051)
    torch.testing.assert_close(after_q, before_q, rtol=0, atol=.0051)


def test_rigid_tree_rotation_tolerance_uses_true_angle_for_parent_candidates():
    raw_q = torch.tensor([-.3029974103, -1.7618416548,
                          .6348446012, -.8043590784])
    q = raw_q.repeat(4, 1)
    # Float32 self-dot fails the old near-one cosine check even with no edit.
    old_dot = (F.normalize(q, dim=-1) * F.normalize(q, dim=-1)).sum(-1)
    assert bool((old_dot < math.cos(math.sqrt(3) * .001 / 4)).all())
    same = RIGID.quaternion_angle_error(q, q)
    negated = RIGID.quaternion_angle_error(q, -q)
    torch.testing.assert_close(same, torch.zeros_like(same), rtol=0, atol=1e-12)
    torch.testing.assert_close(negated, torch.zeros_like(negated), rtol=0, atol=1e-12)
    outside = RIGID.rotation_vector_quaternion(
        torch.tensor([[0., 0., .002]]))
    assert float(RIGID.quaternion_angle_error(
        torch.tensor([[1., 0., 0., 0.]]), outside)) > math.sqrt(3) * .001 / 2

    model = rigid_tree_fixture(gs_xyz_step_extent=.01,
                               gs_rotation_step_rad=.001)
    model.frame_index = 10
    model.base_rotation = q.clone()
    with torch.no_grad():
        model.region_translation[1, 0] = .2
        model.region_translation[2, 0] = .2
    before_xyz, before_q = (value.detach().clone() for value in model.forward(60))
    edits = model.update_topology(60)
    assert sum(edit['kind'] == 'parent' for edit in edits) == 1
    after_xyz, after_q = model.forward(60)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=.0051)
    assert float(RIGID.quaternion_angle_error(after_q, before_q).amax()) < 1e-6


def test_rigid_tree_checkpoint_restores_pose_and_adam_next_step():
    model = rigid_tree_fixture()
    (model.forward(1, training=True)[0] - model.base_xyz - .03).square().sum().backward()
    model.optimizer_step()
    saved = copy.deepcopy(model.capture())
    recovered = RIGID.RigidResidualTree(model.config)
    recovered.restore(saved)
    for name in ('region_translation', 'region_rotation',
                 'gs_translation', 'gs_rotation'):
        torch.testing.assert_close(getattr(recovered, name), getattr(model, name),
                                   rtol=0, atol=0)
    for original, clone in zip(model.optimizer.state.values(),
                               recovered.optimizer.state.values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(original[key], clone[key], rtol=0, atol=0)
    for subject in (model, recovered):
        (subject.forward(2, training=True)[0] - subject.base_xyz - .07).square().sum().backward()
        subject.optimizer_step()
    for name in ('region_translation', 'region_rotation',
                 'gs_translation', 'gs_rotation'):
        torch.testing.assert_close(getattr(recovered, name), getattr(model, name),
                                   rtol=0, atol=0)


def test_rigid_tree_codec_sparse_fields_birth_and_two_frame_replay():
    prior_model = rigid_tree_fixture()
    prior = prior_model.codec_previous_state
    zero_r = torch.zeros(3, 3, dtype=torch.int16)
    zero_g = torch.zeros(4, 3, dtype=torch.int16)
    region_t = zero_r.clone(); region_t[1, 0] = 20
    region_r = zero_r.clone(); region_r[2, 2] = 10
    gs_t = torch.zeros(3, 3, dtype=torch.int16); gs_t[0, 1] = 5
    gs_r = torch.zeros(3, 3, dtype=torch.int16); gs_r[2, 0] = 4
    rows = torch.tensor([0, 2, 3])
    owners = prior.owner[rows].clone()
    birth_xyz = torch.tensor([[8., 1., 0.]])
    birth_q = torch.tensor([[1., 0., 0., 0.]])
    args = (prior, 1, 1., rows, owners, prior.parents,
            region_t, region_r, gs_t, gs_r, torch.tensor([1]),
            birth_xyz, birth_q, .01, .01)
    packet = RIGID_CODEC.encode_motion(*args)
    state = RIGID_CODEC.decode_motion(packet, prior)
    assert state.frame_index == 1 and len(state.xyz) == 4
    torch.testing.assert_close(state.xyz[-1], birth_xyz[0], rtol=0, atol=0)
    assert int(state.owner[-1]) == int(owners[1])
    expected_xyz, expected_q = RIGID.compose_pose(
        prior.xyz[rows], prior.rotation[rows], owners, prior.parents,
        RIGID.region_pivots(prior.xyz[rows], owners, prior.parents),
        region_t.float() * .01, region_r.float() * .01,
        gs_t.float() * .01, gs_r.float() * .01)
    torch.testing.assert_close(state.xyz[:3], expected_xyz, rtol=0, atol=1e-6)
    torch.testing.assert_close(state.rotation[:3], expected_q, rtol=0, atol=1e-6)
    zero_packet = RIGID_CODEC.encode_motion(
        prior, 1, 1., torch.arange(4), prior.owner, prior.parents,
        zero_r, zero_r, zero_g, zero_g, torch.empty(0, dtype=torch.long),
        torch.empty(0, 3), torch.empty(0, 4), .01, .01)
    assert len(zero_packet) < len(packet)
    second_rows = torch.tensor([0, 2, 3])
    packet2 = RIGID_CODEC.encode_motion(
        state, 2, 1., second_rows, state.owner[second_rows], state.parents,
        zero_r, zero_r, torch.zeros(3, 3, dtype=torch.int16),
        torch.zeros(3, 3, dtype=torch.int16), torch.tensor([2]),
        torch.tensor([[9., 0., 0.]]), birth_q, .01, .01)
    next_state = RIGID_CODEC.decode_motion(packet2, state)
    torch.testing.assert_close(next_state.xyz[:3], state.xyz[second_rows],
                               rtol=0, atol=0)
    assert next_state.frame_index == 2
    assert int(next_state.owner[-1]) == int(state.owner[2])


def test_rigid_tree_codec_rejects_cycles_wrong_state_and_corruption():
    model = rigid_tree_fixture()
    state = model.codec_previous_state
    zero_r = torch.zeros(3, 3, dtype=torch.int16)
    zero_g = torch.zeros(4, 3, dtype=torch.int16)
    args = (state, 1, 1., torch.arange(4), state.owner, state.parents,
            zero_r, zero_r, zero_g, zero_g, torch.empty(0, dtype=torch.long),
            torch.empty(0, 3), torch.empty(0, 4), .01, .01)
    packet = RIGID_CODEC.encode_motion(*args)
    for damaged in (packet[:-1], packet + b'X', b'BAD!' + packet[4:]):
        try:
            RIGID_CODEC.decode_motion(damaged, state)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupted rigid stream was accepted')
    wrong = copy.deepcopy(state)
    wrong.xyz[0, 0] += 1
    try:
        RIGID_CODEC.decode_motion(packet, wrong)
    except ValueError as error:
        assert 'digest' in str(error)
    else:
        raise AssertionError('Wrong prior state was accepted')
    cycle = torch.tensor([-1, 2, 1])
    bad = list(args); bad[5] = cycle
    try:
        RIGID_CODEC.encode_motion(*bad)
    except ValueError as error:
        assert 'cycle' in str(error).lower()
    else:
        raise AssertionError('Parent cycle was accepted')


def test_rigid_tree_all_zero_identity_stream_does_not_grow_with_gs_count():
    sizes = []
    for count in (256, 1024):
        xyz = torch.zeros(count, 3)
        rotation = torch.zeros(count, 4); rotation[:, 0] = 1
        owner = torch.ones(count, dtype=torch.long)
        parents = torch.tensor([-1, 0])
        state = RIGID_CODEC.DecodedState(xyz, rotation, owner, parents, 0)
        payload = RIGID_CODEC.encode_motion(
            state, 1, 1., torch.arange(count), owner, parents,
            torch.zeros(2, 3, dtype=torch.int16),
            torch.zeros(2, 3, dtype=torch.int16),
            torch.zeros(count, 3, dtype=torch.int16),
            torch.zeros(count, 3, dtype=torch.int16),
            torch.empty(0, dtype=torch.long), torch.empty(0, 3),
            torch.empty(0, 4), .01, .01)
        decoded = RIGID_CODEC.decode_motion(payload, state)
        torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=0)
        sizes.append(len(payload))
    assert sizes[0] == sizes[1]


def test_rigid_tree_hard_rate_matches_actual_packet_without_double_penalty():
    model = rigid_tree_fixture()
    with torch.no_grad():
        model.region_translation[1, 0] = .02
        model.gs_translation[0, 1] = .99  # 99 signed grid units.
    hard = int(model.loss_terms(60)['rate_bytes_hard'])
    surrogate = model.loss_terms(60)['rate_loss']
    assert not surrogate.requires_grad
    assert model.region_translation.grad is None
    assert model.gs_translation.grad is None
    xyz, q = model.materialized_pose()
    model.register_materialized_layout(xyz, q)
    with tempfile.TemporaryDirectory() as directory:
        info = model.export_motion(Path(directory) / 'motion.bin')
    assert hard == info['raw_stream_bytes']
    assert info['bytes'] <= hard


def test_rigid_tree_birth_inherits_owner_across_survivor_deletions():
    model = rigid_tree_fixture()
    pose_xyz, pose_q = model.materialized_pose()
    birth_xyz = torch.tensor([[7., 0., 0.]])
    birth_q = torch.tensor([[1., 0., 0., 0.]])
    source = torch.tensor([1])
    owner = int(model.owner[1])
    model.register_materialized_layout(torch.cat((pose_xyz, birth_xyz)),
                                       torch.cat((pose_q, birth_q)), source)
    assert int(model.materialized_owner[-1]) == owner
    with tempfile.TemporaryDirectory() as directory:
        model.export_motion(Path(directory) / 'motion.bin')
    state = model.codec_previous_state
    surviving = torch.tensor([0, 2, 4])
    model.setup(state.xyz[surviving], state.rotation[surviving], 1.,
                frame_index=2, previous_rows=surviving)
    torch.testing.assert_close(model.owner, state.owner[surviving], rtol=0, atol=0)
    assert int(model.owner[-1]) == owner
    assert torch.isfinite(model.pivots).all()


def test_rigid_birth_appearance_stream_roundtrips_and_rejects_bad_counts():
    old = dict(xyz=torch.zeros(4, 3), rotation=torch.tensor([[1., 0., 0., 0.]] * 4),
               logscale=torch.zeros(4, 3), opacity=torch.zeros(4, 1),
               sh_dc=torch.zeros(4, 1, 3), sh_rest=torch.zeros(4, 2, 3))
    new = dict(logscale=torch.tensor([[.1, .2, .3], [.4, .5, .6]]),
               opacity=torch.tensor([[.7], [.8]]),
               sh_dc=torch.arange(6, dtype=torch.float32).reshape(2, 1, 3),
               sh_rest=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3))
    xyz = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    rotation = torch.tensor([[1., 0., 0., 0.]] * 2)
    packet = RIGID_BIRTH_CODEC.encode_birth_attributes(new, 4)
    decoded = RIGID_BIRTH_CODEC.decode_birth_attributes(packet, old, xyz, rotation)
    for key, expected in {**new, 'xyz': xyz, 'rotation': rotation}.items():
        torch.testing.assert_close(decoded[key], expected, rtol=0, atol=0)
    for bad_packet, bad_xyz in ((packet, xyz[:1]), (packet[:-1], xyz),
                                (packet + b'X', xyz)):
        try:
            RIGID_BIRTH_CODEC.decode_birth_attributes(
                bad_packet, old, bad_xyz, rotation)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt birth appearance packet was accepted')


def test_rigid_birth_quantization_uses_lineage_and_keeps_periodic_survivors():
    birth_quantization = importlib.import_module(
        'pipeline.ReCon-GS.model.birth_quantization')
    reference = dict(xyz=torch.tensor([[0., 0., 0.], [10., 0., 0.]]),
                     rotation=torch.tensor([[1., 0., 0., 0.]] * 2),
                     logscale=torch.zeros(2, 3), opacity=torch.zeros(2, 1),
                     sh_dc=torch.zeros(2, 1, 3), sh_rest=torch.zeros(2, 0, 3))
    candidate = SimpleNamespace(
        _xyz=torch.nn.Parameter(torch.tensor([[.01, 0., 0.], [9.99, 0., 0.]])),
        _rotation=torch.nn.Parameter(reference['rotation'].clone()),
        _scaling=torch.nn.Parameter(reference['logscale'].clone()),
        _opacity=torch.nn.Parameter(torch.full((2, 1), -10.)),
        _features_dc=torch.nn.Parameter(reference['sh_dc'].clone()),
        _features_rest=torch.nn.Parameter(reference['sh_rest'].clone()))
    source = torch.tensor([1, 0])  # Deliberately disagrees with nearest position.
    view = birth_quantization.QuantizedNewGaussianView(
        candidate, RIGID_MODULE.BirthConfig(), 1., reference, source)
    raw, decoded_source = birth_quantization.materialize_quantized_gaussians(
        reference, view)
    torch.testing.assert_close(decoded_source, source, rtol=0, atol=0)
    assert len(raw['xyz']) == 4
    assert bool((torch.sigmoid(raw['opacity'][-2:]) < .01).all())
    view.get_opacity.sum().backward()
    assert candidate._opacity.grad is not None
    assert bool((candidate._opacity.grad.abs() > 0).all())
    try:
        birth_quantization.QuantizedNewGaussianView(
            candidate, RIGID_MODULE.BirthConfig(), 1., reference,
            torch.tensor([2, 0]))
    except ValueError as error:
        assert 'lineage' in str(error)
    else:
        raise AssertionError('Birth source outside reference was accepted')


def test_rigid_full_raw_stream_replays_motion_birth_and_manifest_two_frames():
    import hashlib
    model = rigid_tree_fixture()
    state0 = model.codec_previous_state
    n = len(state0.xyz)
    raw0 = dict(xyz=state0.xyz, rotation=state0.rotation,
                logscale=torch.arange(3 * n, dtype=torch.float32).reshape(n, 3) / 10,
                opacity=torch.arange(n, dtype=torch.float32).reshape(n, 1) / 10,
                sh_dc=torch.arange(3 * n, dtype=torch.float32).reshape(n, 1, 3) / 10,
                sh_rest=torch.arange(6 * n, dtype=torch.float32).reshape(n, 2, 3) / 10)
    newborn_xyz = torch.tensor([[7., 1., 0.]])
    newborn_q = torch.tensor([[1., 0., 0., 0.]])
    newborn_attrs = dict(logscale=torch.tensor([[.3, .4, .5]]),
                         opacity=torch.tensor([[.7]]),
                         sh_dc=torch.tensor([[[.11, .12, .13]]]),
                         sh_rest=torch.tensor([[[.1, .2, .3], [.4, .5, .6]]]))
    region_t = torch.zeros(3, 3, dtype=torch.int16)
    region_t[1, 0] = 2
    zero_r = torch.zeros(3, 3, dtype=torch.int16)
    zero_g = torch.zeros(n, 3, dtype=torch.int16)
    edited_owner = state0.owner.clone()
    edited_owner[2] = 1
    packet1 = RIGID_CODEC.encode_motion(
        state0, 1, 1., torch.arange(n), edited_owner, state0.parents,
        region_t, zero_r, zero_g, zero_g, torch.tensor([2]),
        newborn_xyz, newborn_q, .01, .01)
    state1 = RIGID_CODEC.decode_motion(packet1, state0)
    torch.testing.assert_close(state1.birth_sources, torch.tensor([2]),
                               rtol=0, atol=0)
    assert int(state1.owner[-1]) == int(edited_owner[2]) != int(state0.owner[2])
    raw1 = {name: torch.cat((raw0[name], newborn_attrs[name]))
            for name in ('logscale', 'opacity', 'sh_dc', 'sh_rest')}
    raw1.update(xyz=state1.xyz, rotation=state1.rotation)
    zero_g_next = torch.zeros(n + 1, 3, dtype=torch.int16)
    packet2 = RIGID_CODEC.encode_motion(
        state1, 2, 1., torch.arange(n + 1), state1.owner, state1.parents,
        zero_r, zero_r, zero_g_next, zero_g_next,
        torch.empty(0, dtype=torch.long), torch.empty(0, 3),
        torch.empty(0, 4), .01, .01)
    state2 = RIGID_CODEC.decode_motion(packet2, state1)

    def raw_digest(raw):
        digest = hashlib.sha256()
        for name in Q.RAW_FIELDS:
            value = raw[name].detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    with tempfile.TemporaryDirectory() as directory:
        stream = Path(directory)
        reference = stream / 'reference_frame0.npz'
        np.savez_compressed(reference, **{name: value.numpy()
                       for name, value in raw0.items()},
                       owner=state0.owner.numpy(), parents=state0.parents.numpy(),
                       frame_index=np.array(0, dtype=np.int32))
        (stream / 'motion_001.bin').write_bytes(packet1)
        (stream / 'motion_002.bin').write_bytes(packet2)
        (stream / 'birth_001.bin').write_bytes(
            RIGID_BIRTH_CODEC.encode_birth_attributes(newborn_attrs, n))
        manifest1 = dict(frame=1, decoded_state_sha256=state1.digest().hex(),
                         raw_state_sha256=raw_digest(raw1))
        manifest2 = dict(frame=2, decoded_state_sha256=state2.digest().hex(),
                         raw_state_sha256=raw_digest(raw1))
        (stream / 'frame_001.json').write_text(json.dumps(manifest1))
        (stream / 'frame_002.json').write_text(json.dumps(manifest2))
        actual_state, actual_raw = RIGID_REPLAY.replay(stream, reference, through=2)
        assert actual_state.digest() == state2.digest()
        for name in Q.RAW_FIELDS:
            torch.testing.assert_close(actual_raw[name], raw1[name], rtol=0, atol=0)
        manifest2['raw_state_sha256'] = '0' * 64
        (stream / 'frame_002.json').write_text(json.dumps(manifest2))
        try:
            RIGID_REPLAY.replay(stream, reference, through=2)
        except ValueError as error:
            assert 'digest' in str(error)
        else:
            raise AssertionError('Wrong replay manifest digest was accepted')


def test_rigid_pipeline_motion_freezes_all_old_attributes_and_birth_updates_only_new():
    class FakeGaussians:
        def __init__(self, xyz):
            self._xyz = torch.nn.Parameter(xyz.clone())
            self._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]] * len(xyz)))
            self._scaling = torch.nn.Parameter(torch.zeros(len(xyz), 3))
            self._opacity = torch.nn.Parameter(torch.zeros(len(xyz), 1))
            self._features_dc = torch.nn.Parameter(torch.zeros(len(xyz), 1, 3))
            self._features_rest = torch.nn.Parameter(torch.zeros(len(xyz), 0, 3))
            self.rigid_xyz = None
            self.rigid_rotation = None

        @property
        def get_xyz(self):
            return self._xyz if self.rigid_xyz is None else self.rigid_xyz

    class FakeRenderer:
        def __init__(self, gaussians):
            self.gaussians = gaussians

        def __call__(self, viewpoint, background):
            return dict(render=self.gaussians.get_xyz.mean().expand(3, 2, 2))

    motion = rigid_tree_fixture()
    old = FakeGaussians(motion.base_xyz)
    fake = SimpleNamespace(gaussians=old, motion=motion, renderer=FakeRenderer(old),
        frame_prepared=True, motion_committed=False, frame_finalized=False,
        trainer=SimpleNamespace(global_step=1), background=torch.zeros(3))
    RIGID_MODULE.Module._freeze_base(fake)
    before = {name: getattr(old, name).detach().clone() for name in Q.RAW_ATTRIBUTES}
    for step in (1, 100):
        fake.trainer.global_step = step
        RIGID_MODULE.Module.forward(fake, None, training=True)['render'].sum().backward()
        motion.optimizer_step()
        assert fake.renderer.gaussians is old
        assert old.rigid_xyz is None and old.rigid_rotation is None
    for name, value in before.items():
        parameter = getattr(old, name)
        assert not parameter.requires_grad and parameter.grad is None
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    assert model_parameter_changed(motion.region_translation)

    newborn = torch.nn.Parameter(torch.tensor([[2., 0., 0.]]))
    newborn.sum().backward()
    births = SimpleNamespace(_xyz=newborn, max_radii2D=torch.zeros(1),
        add_densification_stats=lambda *args, **kwargs: None,
        optimizer=torch.optim.SGD([newborn], lr=.1))
    fake.births = births
    fake.motion_committed = True
    fake.trainer.global_step = 101
    fake.motion_steps = 100
    fake.total_steps = 200
    fake.config = SimpleNamespace(birth=SimpleNamespace(densify_interval=40))
    fake.birth_control = dict(split_added=0, pruned=0)
    package = dict(visibility_filter=torch.ones(len(old._xyz) + 1, dtype=torch.bool),
                   radii=torch.ones(len(old._xyz) + 1),
                   viewspace_points=torch.zeros(len(old._xyz) + 1, 3))
    RIGID_MODULE.Module.post_training_step(fake, package)
    torch.testing.assert_close(newborn.detach(), torch.tensor([[1.9, -.1, -.1]]),
                               rtol=0, atol=1e-6)
    for name, value in before.items():
        torch.testing.assert_close(getattr(old, name), value, rtol=0, atol=0)


def model_parameter_changed(parameter):
    return bool(parameter.detach().abs().sum() > 0)


def test_rigid_pipeline_only_one_method_and_native_first_checkpoint_contract():
    import yaml
    from dataclasses import fields
    config = yaml.safe_load((project_root / 'config' /
        'fs_rigid_residual_tree_300f_20261011.yaml').read_text())
    module = config['module']
    assert module['rigid_tree']['motion_steps'] == 100
    assert module['rigid_tree']['birth_steps'] == 100
    assert config['trainer']['num_init_steps'] == 15000
    assert config['data']['resolution'] == -1
    assert config['data']['extra_dataset_kwargs']['num_frames'] == 300
    assert 'rigid_tree' in {field.name for field in fields(RIGID_MODULE.ModuleConfig)}
    assert not ({'motion_mode', 'quantized_residual', 'flat_cluster',
                 'sparse_hierarchical'} & module.keys())
    assert not hasattr(RIGID_MODULE.ModuleConfig(), 'motion_mode')
    legacy = importlib.import_module('pipeline.ReCon-GS.archive')
    assert hasattr(legacy.ModuleConfig(), 'motion_mode')
    for path in (project_root / 'pipeline' / 'ReCon-GS' / name
                 for name in ('module.py', 'trainer.py', 'data.py')):
        imports = [node.module or '' for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, ast.ImportFrom)]
        assert not any('archive' in name or 'legacy' in name for name in imports)
    assert RIGID_TRAINER.Trainer.num_steps.fget(SimpleNamespace(
        is_first_frame=False, module=SimpleNamespace(total_steps=200))) == 200

    events = []
    gaussians = SimpleNamespace(_xyz=torch.zeros(3, 3),
                                restore=lambda packet, cfg: events.append('gaussians'))
    fake = SimpleNamespace(gaussians=gaussians,
        renderer=SimpleNamespace(gaussians=None),
        config=SimpleNamespace(gaussian=object(), gaussian_stage2=object(),
                               rigid_tree=RIGID.RigidTreeConfig(),
                               initial_opacity_threshold=0.0),
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=15000)),
        current_frame=0, _freeze_base=lambda: events.append('frozen'),
        reference_count=None)
    RIGID_MODULE.Module.on_load_checkpoint(fake,
        dict(global_step=15000, gaussians='shared-first-frame'))
    assert events == ['gaussians', 'frozen'] and fake.reference_count == 3
    assert fake.initial_reference['removed_count'] == 0
    fake.current_frame = 1
    try:
        RIGID_MODULE.Module.on_load_checkpoint(fake,
            dict(global_step=15000, gaussians='shared-first-frame'))
    except ValueError as error:
        assert 'first-frame' in str(error)
    else:
        raise AssertionError('Native frame-zero checkpoint accepted on later frame')


def test_rigid_pipeline_mid_birth_checkpoint_restores_lineage_and_adam():
    if not torch.cuda.is_available():
        return  # The libgs Gaussian training setup allocates CUDA statistics.
    device = 'cuda'
    gaussian_cfg = RIGID_MODULE.GaussianConfig()
    birth = ACTIVE_GAUSS.BirthGaussianModel(0)
    fields = dict(_xyz=torch.tensor([[2., 0., 0.]], device=device),
                  _rotation=torch.tensor([[1., 0., 0., 0.]], device=device),
                  _scaling=torch.zeros(1, 3, device=device),
                  _opacity=torch.zeros(1, 1, device=device),
                  _features_dc=torch.zeros(1, 1, 3, device=device),
                  _features_rest=torch.zeros(1, 0, 3, device=device))
    for name, value in fields.items():
        setattr(birth, name, torch.nn.Parameter(value))
    birth.spatial_lr_scale = 1.
    birth.training_setup(gaussian_cfg)
    birth.max_radii2D = torch.zeros(1, device=device)
    birth.source_rows = torch.tensor([2], device=device)
    birth._xyz.sum().backward()
    birth.optimizer.step()
    birth.optimizer.zero_grad(set_to_none=True)

    motion = rigid_tree_fixture()
    config = RIGID_MODULE.ModuleConfig()
    config.rigid_tree = motion.config
    config.gaussian = gaussian_cfg
    config.gaussian_stage2 = gaussian_cfg
    base = SimpleNamespace(_xyz=torch.zeros(4, 3, device=device),
                           capture=lambda: 'gaussians',
                           restore=lambda packet, cfg: None)
    source = SimpleNamespace(config=config, gaussians=base, births=birth,
        motion=motion, frame_prepared=True, frame_finalized=False,
        motion_committed=True, reference_count=4, frame_report=None,
        initial_reference=dict(source='native_frame0_checkpoint', threshold=0.0,
                               original_count=4, retained_count=4, removed_count=0),
        birth_control=dict(clone_added=1))
    packet = {}
    RIGID_MODULE.Module.on_save_checkpoint(source, packet)
    packet = copy.deepcopy(packet)
    restored_birth = ACTIVE_GAUSS.BirthGaussianModel(0)
    events = []
    resumed = SimpleNamespace(config=config, gaussians=base,
        renderer=SimpleNamespace(gaussians=None), births=restored_birth,
        motion=RIGID.RigidResidualTree(motion.config), device=torch.device(device),
        _freeze_base=lambda: events.append('base_frozen'), current_frame=1,
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=15000)))
    RIGID_MODULE.Module.on_load_checkpoint(resumed, packet)
    assert resumed.motion_committed and not resumed.frame_finalized
    assert resumed.initial_reference == source.initial_reference
    assert not resumed.motion.committed
    assert all(not parameter.requires_grad for parameter in resumed.motion.parameters())
    assert events == ['base_frozen']
    torch.testing.assert_close(restored_birth.source_rows.cpu(), torch.tensor([2]),
                               rtol=0, atol=0)
    torch.testing.assert_close(restored_birth._xyz, birth._xyz, rtol=0, atol=0)
    original_moments = birth.optimizer.state[birth._xyz]
    restored_moments = restored_birth.optimizer.state[restored_birth._xyz]
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(restored_moments[key], original_moments[key],
                                   rtol=0, atol=0)
    for candidate in (birth, restored_birth):
        candidate._xyz.square().sum().backward()
        candidate.optimizer.step()
    torch.testing.assert_close(restored_birth._xyz, birth._xyz, rtol=0, atol=0)


def test_rigid_pipeline_mid_birth_restore_freezes_motion_on_cpu():
    motion = rigid_tree_fixture()
    assert not motion.committed
    for parameter in motion.parameters():
        parameter.sum().backward(retain_graph=True)
    packet = motion.capture()
    config = RIGID_MODULE.ModuleConfig()
    config.rigid_tree = motion.config
    base = SimpleNamespace(_xyz=torch.zeros(4, 3),
                           restore=lambda saved, cfg: None)
    resumed = SimpleNamespace(config=config, gaussians=base,
        renderer=SimpleNamespace(gaussians=None),
        motion=RIGID.RigidResidualTree(motion.config),
        _freeze_base=lambda: None, current_frame=1,
        trainer=SimpleNamespace(config=SimpleNamespace(num_init_steps=15000)))
    checkpoint = dict(gaussians='base', rigid_tree_version=1,
        rigid_tree_config=asdict(config.rigid_tree),
        rigid_tree_module_config=asdict(config),
        rigid_tree_frame=dict(prepared=True, finalized=False,
            motion_committed=True, reference_count=4, report=None,
            birth_control=dict(clone_added=0)),
        rigid_tree_motion=packet, births=None)
    RIGID_MODULE.Module.on_load_checkpoint(resumed, checkpoint)
    assert resumed.motion_committed and not resumed.motion.committed
    assert all(not parameter.requires_grad for parameter in resumed.motion.parameters())
    assert all(parameter.grad is None for parameter in resumed.motion.parameters())


def test_rigid_pipeline_single_validation_loader_averages_its_views():
    logged = []
    fake = SimpleNamespace(datamodule=SimpleNamespace(eval_names=['test']),
                           log_dict=lambda record: logged.append(record))
    rows = [dict(psnr=20., l1=.2), dict(psnr=30., l1=.1)]
    result = RIGID_MODULE.Module.validation_end(fake, rows, num_loaders=1)
    torch.testing.assert_close(torch.tensor([result['test']['psnr'], result['test']['l1']]),
                               torch.tensor([25., .15]), rtol=0, atol=1e-7)
    assert len(logged) == 1
    torch.testing.assert_close(torch.tensor([logged[0]['eval-test/psnr'],
                                            logged[0]['eval-test/l1']]),
                               torch.tensor([25., .15]), rtol=0, atol=1e-7)


if __name__ == '__main__':
    checks = [value for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for check in checks:
        check()
    print(f'All {len(checks)} nested prototype numerical checks passed.')
