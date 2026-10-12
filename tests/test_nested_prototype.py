"""Deterministic numerical checks for the active rigid-tree motion model."""

import ast
import copy
import importlib
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import numerical modules without running the pipeline package initializer.
pipeline_package = ModuleType("pipeline")
pipeline_package.__path__ = [str(PROJECT_ROOT / "pipeline")]
recon_package = ModuleType("pipeline.ReCon-GS")
recon_package.__path__ = [str(PROJECT_ROOT / "pipeline" / "ReCon-GS")]
pipeline_package.__dict__["ReCon-GS"] = recon_package
sys.modules["pipeline"] = pipeline_package
sys.modules["pipeline.ReCon-GS"] = recon_package

RIGID = importlib.import_module("pipeline.ReCon-GS.model.rigid_residual_tree")
RIGID_MODULE = importlib.import_module("pipeline.ReCon-GS.module")


def rigid_tree_fixture(**overrides):
    options = dict(
        region_count=3,
        motion_steps=100,
        birth_steps=100,
        tree_edit_interval=10,
        tree_edit_step=60,
        tree_edit_neighbors=2,
        motion_xyz_step_extent=0.01,
        motion_rotation_step_rad=0.01,
    )
    options.update(overrides)
    model = RIGID.RigidResidualTree(RIGID.RigidTreeConfig(**options))
    xyz = torch.tensor(
        [[-2.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * len(xyz))
    model.setup(xyz, rotation, 1.0, frame_index=1)
    return model


def test_rigid_tree_parent_child_pose_uses_distinct_pivots():
    dtype = torch.float64
    xyz = torch.tensor([[2.0, 0.0, 0.0]], dtype=dtype)
    base_rotation = RIGID.rotation_vector_quaternion(
        torch.tensor([[0.0, torch.pi / 2, 0.0]], dtype=dtype)
    )
    owner = torch.tensor([2])
    parents = torch.tensor([-1, 0, 1])
    pivots = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=dtype,
    )
    translation = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
    )
    rotation = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, torch.pi / 2],
         [torch.pi / 2, 0.0, 0.0]],
        dtype=dtype,
    )
    zeros = torch.zeros(1, 3, dtype=dtype)

    moved, composed = RIGID.compose_pose(
        xyz, base_rotation, owner, parents, pivots,
        translation, rotation, zeros, zeros,
    )

    torch.testing.assert_close(
        moved, torch.tensor([[1.0, 2.0, 1.0]], dtype=dtype),
        rtol=0, atol=1e-12,
    )
    rotated_x = RIGID.quaternion_rotate(
        composed, torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype)
    )
    torch.testing.assert_close(
        rotated_x, torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype),
        rtol=0, atol=1e-12,
    )


def test_rigid_tree_zero_pose_is_identity_with_motion_gradients():
    model = rigid_tree_fixture()
    xyz, rotation = model.forward(1, training=True)

    assert torch.equal(xyz, model.base_xyz)
    assert torch.equal(rotation, model.base_rotation)

    (xyz[:, 1].sum() + rotation[:, 2].sum()).backward()
    assert model.region_rotation.grad[1:].abs().sum() > 0
    assert model.gs_translation.grad.abs().sum() > 0
    assert model.gs_rotation.grad.abs().sum() > 0


def test_rigid_tree_schedule_and_initial_layout_are_deterministic():
    model = rigid_tree_fixture(region_count=6)
    assert model.stage(60) == "motion_fit"

    model.frame_index = 10
    assert [model.stage(step) for step in (59, 60, 61, 70, 100)] == [
        "motion_fit", "tree_edit", "motion_fit", "motion_fit", "motion_fit"
    ]
    owner, parents = RIGID.initial_layout(model.base_xyz, 6)
    torch.testing.assert_close(owner, model.owner, rtol=0, atol=0)
    torch.testing.assert_close(parents, model.parents, rtol=0, atol=0)
    assert owner.tolist() == [1, 3, 2, 4]
    assert parents.tolist() == [-1, 0, 0, 0, 0, 0]


def test_translation_and_rotation_quantization_are_independent():
    model = rigid_tree_fixture()
    with torch.no_grad():
        model.region_translation[1, 0] = 0.02
        model.region_rotation[2, 2] = 0.03
        model.gs_translation[0, 1] = 0.04
        model.gs_rotation[3, 0] = 0.05

    region_t, region_r, gs_t, gs_r = model._values(hard=True)
    assert region_t[1].tolist() == [2, 0, 0]
    assert not bool(region_r[1].any())
    assert region_r[2].tolist() == [0, 0, 3]
    assert not bool(region_t[2].any())
    assert gs_t[0].tolist() == [0, 4, 0]
    assert not bool(gs_r[0].any())
    assert gs_r[3].tolist() == [5, 0, 0]
    assert not bool(gs_t[3].any())


def test_relative_motion_loss_only_regularizes_regions():
    model = rigid_tree_fixture(relative_motion_loss_weight=1.0)
    model.extent = 2.0
    with torch.no_grad():
        model.region_translation[1] = torch.tensor([2.0, -4.0, 0.0])
        model.region_rotation[1] = torch.tensor([-0.3, 0.0, 0.1])
        model.region_translation[2] = torch.tensor([0.0, 2.0, -2.0])
        model.region_rotation[2] = torch.tensor([0.0, 0.2, 0.0])

    loss = model.loss_terms(1, diagnostics=False)["relative_motion_loss"]
    torch.testing.assert_close(loss, torch.tensor(2.8), rtol=0, atol=1e-6)
    loss.backward()

    assert model.region_translation.grad.abs().sum() > 0
    assert model.region_rotation.grad.abs().sum() > 0
    assert model.gs_translation.grad is None
    assert model.gs_rotation.grad is None


def test_sparse_proximal_step_removes_weak_gs_blocks():
    model = rigid_tree_fixture(rate_loss_weight=1.0)
    with torch.no_grad():
        model.gs_translation[0, 1] = 0.01
        model.gs_rotation[1, 0] = 0.01
    for parameter in (model.gs_translation, model.gs_rotation):
        state = model.optimizer.state[parameter]
        state["step"] = torch.tensor(1.0)
        state["exp_avg"] = torch.zeros_like(parameter)
        state["exp_avg_sq"] = torch.full_like(parameter, 1e-12)

    model.last_step = 60
    model._proximal_gs_blocks()

    assert not bool(model.gs_translation.any())
    assert not bool(model.gs_rotation.any())


def test_owner_reassignment_preserves_quantized_pose():
    model = rigid_tree_fixture()
    model.frame_index = 10
    original_owner = model.owner.clone()
    row = int((model.owner == 1).nonzero(as_tuple=True)[0][0])
    with torch.no_grad():
        model.region_translation[2, 0] = 0.2
        model.gs_translation[row, 0] = 0.2

    before_xyz, before_rotation = (
        value.detach().clone() for value in model.forward(60)
    )
    edits = model.update_topology(60)

    assert any(
        edit["kind"] == "owner" and edit["row"] == row for edit in edits
    )
    assert int(model.owner[row]) != int(original_owner[row])
    after_xyz, after_rotation = model.forward(60)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=0.0051)
    torch.testing.assert_close(
        after_rotation, before_rotation, rtol=0, atol=0.0051
    )


def test_checkpoint_restores_pose_and_next_adam_step():
    model = rigid_tree_fixture()
    loss = (model.forward(1, training=True)[0] - model.base_xyz - 0.03).square().sum()
    loss.backward()
    model.optimizer_step()

    recovered = RIGID.RigidResidualTree(model.config)
    recovered.restore(copy.deepcopy(model.capture()))

    for name in (
        "region_translation", "region_rotation", "gs_translation", "gs_rotation"
    ):
        torch.testing.assert_close(
            getattr(recovered, name), getattr(model, name), rtol=0, atol=0
        )

    for subject in (model, recovered):
        next_loss = (
            subject.forward(2, training=True)[0] - subject.base_xyz - 0.07
        ).square().sum()
        next_loss.backward()
        subject.optimizer_step()

    for name in (
        "region_translation", "region_rotation", "gs_translation", "gs_rotation"
    ):
        torch.testing.assert_close(
            getattr(recovered, name), getattr(model, name), rtol=0, atol=0
        )


def test_production_pipeline_exposes_only_rigid_tree_motion():
    config_fields = {item.name for item in fields(RIGID_MODULE.ModuleConfig)}
    assert "rigid_tree" in config_fields
    assert "motion_mode" not in config_fields

    root = PROJECT_ROOT / "pipeline" / "ReCon-GS"
    assert not (root / "archive").exists()
    for path in (root / "module.py", root / "trainer.py", root / "data.py"):
        imports = [
            node.module or ""
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom)
        ]
        assert not any("archive" in name or "legacy" in name for name in imports)
