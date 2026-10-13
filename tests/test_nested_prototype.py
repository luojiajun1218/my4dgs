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
CLUSTER = importlib.import_module("pipeline.ReCon-GS.model.shared_motion_clustering")
CODEC = importlib.import_module("pipeline.ReCon-GS.model.rigid_tree_codec")
TREE_STATS = importlib.import_module("pipeline.ReCon-GS.model.tree_statistics")
HEIR = importlib.import_module("pipeline.ReCon-GS.model.heir_motion_hierarchy")
RESIDUAL_CLUSTER = importlib.import_module("pipeline.ReCon-GS.model.residual_region_clustering")


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


def grow_rate_fixture(model):
    """Append source-linked births and start the next decoded frame."""
    import tempfile

    xyz, rotation = (value.detach().clone() for value in model.materialized_pose())
    sources = torch.arange(len(xyz), device=xyz.device)
    model.register_materialized_layout(torch.cat((xyz, xyz)),
                                      torch.cat((rotation, rotation)), sources)
    with tempfile.TemporaryDirectory() as directory:
        model.export_motion(Path(directory) / 'motion.bin')
    decoded = model.codec_previous_state
    model.setup(decoded.xyz, decoded.rotation, 1., frame_index=model.frame_index + 1)


def test_reference_rate_keeps_one_motion_blocks_price_after_births():
    model = rigid_tree_fixture(rate_loss_weight=1., rate_normalization='reference')

    def added_block_price(subject):
        with torch.no_grad():
            subject.gs_translation.zero_()
        empty = subject.loss_terms(60)['rate_loss']
        with torch.no_grad():
            subject.gs_translation[0, 0] = .1
        return subject.loss_terms(60)['rate_loss'] - empty

    before = added_block_price(model)
    grow_rate_fixture(model)
    assert len(model.owner) == 8 and model.rate_reference_count == 4
    torch.testing.assert_close(added_block_price(model), before, rtol=0, atol=1e-7)
    terms = model.loss_terms(60)
    torch.testing.assert_close(terms['rate_byte_price'], torch.tensor(1 / 96),
                               rtol=0, atol=1e-8)


def test_reference_rate_does_not_loosen_old_gs_gate_after_births():
    for normalization in ('reference', 'frame'):
        model = rigid_tree_fixture(rate_loss_weight=1., rate_normalization=normalization)
        grow_rate_fixture(model)
        with torch.no_grad():
            model.gs_translation[0, 0] = .1
        # Returning to zero costs 0.06 in this Adam metric. The block price
        # is 8/96 for the four-point reference, and would fall to 8/192
        # if four births could change the price of an existing point.
        variance = (0.012 - model.config.adam_eps) ** 2 * (1 - model.config.adam_betas[1])
        model.optimizer.state[model.gs_translation] = dict(step=torch.tensor(1.),
            exp_avg=torch.zeros_like(model.gs_translation),
            exp_avg_sq=torch.full_like(model.gs_translation, variance))
        model.last_step = 60
        model._proximal_gs_blocks()
        expected = 0. if normalization == 'reference' else .1
        torch.testing.assert_close(model.gs_translation[0, 0], torch.tensor(expected),
                                   rtol=0, atol=1e-7)


def test_reference_rate_checkpoint_keeps_initial_count_and_legacy_default():
    model = rigid_tree_fixture(rate_normalization='reference')
    grow_rate_fixture(model)
    packet = copy.deepcopy(model.capture())
    restored = RIGID.RigidResidualTree(copy.deepcopy(model.config))
    restored.restore(packet)
    assert restored.rate_reference_count == 4 and len(restored.owner) == 8
    for name, value in model.loss_terms(60).items():
        torch.testing.assert_close(restored.loss_terms(60)[name], value, rtol=0, atol=0)
    packet['rate_reference_count'] = 0
    try:
        restored.restore(packet)
    except ValueError as error:
        assert 'shared-reference GS count' in str(error)
    else:
        raise AssertionError('Invalid reference count must be rejected')
    legacy = rigid_tree_fixture()
    packet = copy.deepcopy(legacy.capture())
    packet['config'].pop('rate_normalization')
    packet.pop('rate_reference_count')
    legacy.restore(packet)
    assert legacy.config.rate_normalization == 'frame'
    assert legacy.rate_normalization_count == len(legacy.owner)


def test_tree_audit_distinguishes_label_permutations_from_membership_changes():
    audit = importlib.import_module('scripts.audit_heir_tree_method')
    previous = [0, 0, 1, 1, 2, 2]
    permuted = audit.ownership_overlap(previous, [0, 0, 2, 2, 1, 1], 3, 3)
    assert permuted['root_to_motion'] == 0 and permuted['motion_to_root'] == 0
    assert permuted['matched_all_fraction'] == 1.
    assert permuted['unmatched_all_fraction'] == 0.
    merged = audit.ownership_overlap(previous, [0, 0, 1, 1, 1, 1], 3, 2)
    assert merged['matched_moving_fraction'] == .5
    assert merged['old_motion_group_retention'] == 1.
    assert merged['best_many_to_one_owner_edit_fraction'] == 0.
    changed = audit.ownership_overlap(previous, [0, 1, 1, 0, 1, 2], 3, 3)
    assert changed['root_to_motion'] == 1 and changed['motion_to_root'] == 1
    torch.testing.assert_close(torch.tensor(changed['matched_moving_fraction'], dtype=torch.float64),
                               torch.tensor(2 / 3, dtype=torch.float64), rtol=0, atol=1e-12)
    assert changed['unmatched_all_fraction'] == .5
    root_only = audit.ownership_overlap([0, 0], [0, 0], 1, 1)
    assert root_only['matched_moving_fraction'] is None
    assert root_only['matched_all_fraction'] == 1.


def test_tree_audit_rotation_witness_uses_actual_noncommuting_composition():
    audit = importlib.import_module('scripts.audit_heir_tree_method')
    witness = audit.linear_rotation_witness(RIGID)
    assert witness['additive_rotation_error_rad'] > .5
    torch.testing.assert_close(torch.tensor(witness['actual_relative_rotation_error_rad'], dtype=torch.float64),
                               torch.tensor(0., dtype=torch.float64), rtol=0, atol=1e-12)


def test_se3_parent_candidates_reject_a_fast_parent_for_a_nearly_static_child():
    q = torch.tensor([[[1., 0., 0., 0.]] * 3])
    b = torch.tensor([[[0., 0., 0.], [.03, 0., 0.], [.001, 0., 0.]]])
    p = torch.zeros(1, 3, 3)
    costs, allowed = HEIR.se3_parent_costs(q, b, p, torch.ones(1, 3, dtype=torch.bool),
                                         1., torch.tensor([1e-5] * 3 + [1e-4] * 3))
    assert not bool(allowed[2, 1]) and bool(allowed[2, 0])
    assert bool(allowed[1, 2]) and not bool(allowed.diagonal().any())
    torch.testing.assert_close(costs[2, 1], torch.tensor(.029), rtol=0, atol=1e-7)
    torch.testing.assert_close(costs[2, 0], torch.tensor(.001), rtol=0, atol=1e-7)


def test_se3_parent_history_does_not_treat_missing_observations_as_static():
    q = torch.tensor([[[1., 0., 0., 0.]] * 3] * 2)
    b = torch.tensor([[[0., 0., 0.], [.03, 0., 0.], [.04, 0., 0.]],
                      [[0., 0., 0.], [.03, 0., 0.], [.04, 0., 0.]]])
    valid = torch.tensor([[True, False, True], [True, True, True]])
    _, allowed = HEIR.se3_parent_costs(q, b, torch.zeros(2, 3, 3), valid,
                                     1., torch.tensor([1e-5] * 3 + [1e-4] * 3))
    assert not bool(allowed[2, 1])  # New parent has no evidence for the older child's past.
    assert bool(allowed[1, 2]) and bool(allowed[2, 0])


def test_se3_hierarchy_samples_learned_probabilities_without_second_softmax():
    torch.manual_seed(91)
    model = HEIR.HeirMotionHierarchy(6, HEIR.HeirHierarchyConfig(geometry='se3'))
    weights = torch.tensor([[0., 0., 0.], [.95, 0., .05], [1., 0., 0.]], requires_grad=True)
    samples = torch.stack([model.sample_hierarchy(weights, 1.6) for _ in range(512)])
    frequency = samples[:, 1, 0].mean()
    assert .90 < float(frequency) < .99
    assert not bool(samples[:, 1, 1].any()) and not bool(samples[:, 2, 1:].any())
    samples[:, 1, 0].sum().backward()
    assert bool(torch.isfinite(weights.grad).all())


def test_temporal_heir_ignores_unobserved_sample_values():
    torch.manual_seed(92)
    model = HEIR.HeirMotionHierarchy(6, HEIR.HeirHierarchyConfig(geometry='se3'))
    motion = torch.tensor([[[0.] * 6, [.03, 0., 0., 0., 0., 0.], [0.] * 6],
                           [[0.] * 6, [.02, 0., 0., 0., 0., 0.], [.025, 0., 0., 0., 0., 0.]]])
    valid = torch.tensor([[True, True, False], [True, True, True]])
    first = model.parent_weights(motion, valid)
    changed = motion.clone()
    changed[0, 2] = 1000.
    torch.testing.assert_close(model.parent_weights(changed, valid), first, rtol=0, atol=0)


def test_motion_history_tracks_gs_ids_instead_of_historical_region_ids():
    history = importlib.import_module('pipeline.ReCon-GS.model.motion_history').MotionHistory(2)
    source = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [3., 0., 0.], [4., 0., 0.], [3., 1., 0.]])
    target = source + torch.tensor([[.03, 0., 0.]] * 3 + [[0., .04, 0.]] * 3)
    history.record(1, source, target, torch.tensor([[1., 0., 0., 0.]] * 6), torch.ones(6, dtype=torch.long))
    # Node IDs swapped; a later newborn has no observation before its birth.
    owner = torch.tensor([2, 2, 2, 1, 1, 1, 3])
    pooled = history.pool(owner, 4, .01, 1)
    torch.testing.assert_close(torch.tensor(pooled['offset'][0][2]),
                               torch.tensor([.03, 0., 0.], dtype=torch.float64), rtol=0, atol=1e-7)
    torch.testing.assert_close(torch.tensor(pooled['offset'][0][1]),
                               torch.tensor([0., .04, 0.], dtype=torch.float64), rtol=0, atol=1e-7)
    assert pooled['valid'][0].tolist() == [True, True, True, False]
    restored = importlib.import_module('pipeline.ReCon-GS.model.motion_history').MotionHistory(2)
    restored.restore(history.capture())
    assert restored.records[0]['frame'] == 1
    torch.testing.assert_close(restored.records[0]['source'], source, rtol=0, atol=0)


def se3_hierarchy_fixture(device='cpu'):
    base = residual_hierarchy_fixture(device=device)
    config = copy.deepcopy(base.config)
    config.heir = HEIR.HeirHierarchyConfig(geometry='se3', history_frames=3)
    config.hdbscan.feature_mode = 'point_displacement'
    model = RIGID.RigidResidualTree(config)
    model.setup(base.base_xyz, base.base_rotation, 1., frame_index=1)
    model.mark_observed(torch.ones(len(model.owner), device=device, dtype=torch.bool))
    return model


def export_fixture_frame(model, directory):
    xyz, rotation = model.materialized_pose()
    model.register_materialized_layout(xyz, rotation)
    model.export_motion(Path(directory) / f'motion_{model.frame_index:03d}.bin')


def test_se3_parent_validation_preserves_pose_and_codec_when_removing_countermotion():
    import tempfile

    model = se3_hierarchy_fixture()
    install_residual_fixture_motion(model)
    model.update_topology(100)
    model.update_topology(200)
    with tempfile.TemporaryDirectory() as directory:
        export_fixture_frame(model, directory)
        reference = model.codec_previous_state
        model.setup(reference.xyz, reference.rotation, 1., frame_index=2)
        model.mark_observed(torch.ones(50, dtype=torch.bool))
        with torch.no_grad():
            model.parents[:] = torch.tensor([-1, 0, 1])
            model.region_translation[1, 0] = .03
            model.region_translation[2, 0] = -.029
        target_xyz, target_q = (value.detach().clone() for value in model.forward(100))
        assert model.stage(100) == 'tree_validate'
        model.update_topology(100)
        assert model.parents.tolist() == [-1, 0, 0]
        torch.testing.assert_close(model.forward(100)[0], target_xyz, rtol=0, atol=9e-5)
        torch.testing.assert_close(model.forward(100)[1], target_q, rtol=0, atol=9e-4)
        export_fixture_frame(model, directory)
        decoded = CODEC.decode_motion((Path(directory) / 'motion_002.bin').read_bytes(), reference)
        torch.testing.assert_close(decoded.xyz, target_xyz, rtol=0, atol=9e-5)
        assert decoded.parents.tolist() == [-1, 0, 0]
        assert [record['frame'] for record in model.motion_history.records] == [1, 2]


def test_temporal_heir_preserves_learning_across_frames_and_uses_past_observations():
    import tempfile

    model = se3_hierarchy_fixture()
    with torch.no_grad():
        model.gs_translation[:16, 0] = .03
        model.gs_translation[16:32, 0] = .033
    model.update_topology(100)
    loss = model.heir_loss_terms(101)['heir_loss']
    loss.backward()
    model.heir_optimizer.step()
    model.heir_optimizer.zero_grad(set_to_none=True)
    learned = [parameter.detach().clone() for parameter in model.heir_hierarchy.parameters()]
    steps = [state['step'].clone() for state in model.heir_optimizer.state.values()]
    network = model.heir_hierarchy
    model.update_topology(200)
    with tempfile.TemporaryDirectory() as directory:
        export_fixture_frame(model, directory)
        for frame in range(2, 10):
            decoded = model.codec_previous_state
            model.setup(decoded.xyz, decoded.rotation, 1., frame_index=frame)
            model.mark_observed(torch.ones(50, dtype=torch.bool))
            model.update_topology(100)
            export_fixture_frame(model, directory)
        decoded = model.codec_previous_state
        model.setup(decoded.xyz, decoded.rotation, 1., frame_index=10)
        model.mark_observed(torch.ones(50, dtype=torch.bool))
        with torch.no_grad():
            model.gs_translation[:16, 0] = .03
            model.gs_translation[16:32, 0] = .033
        model.update_topology(100)
        assert model.heir_hierarchy is network
        assert model.heir_features.shape[0] == 4
        assert model.last_metrics['heir/retained_network'] == 1
        for actual, expected in zip(model.heir_hierarchy.parameters(), learned):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for state, expected in zip(model.heir_optimizer.state.values(), steps):
            torch.testing.assert_close(state['step'], expected, rtol=0, atol=0)


def test_temporal_heir_checkpoint_preserves_history_and_next_stochastic_step():
    model = se3_hierarchy_fixture()
    with torch.no_grad():
        model.gs_translation[:16, 0] = .03
        model.gs_translation[16:32, 0] = .033
    model.update_topology(100)
    target = model.forward(100)[0].detach().clone()

    def advance(subject, step):
        terms = subject.heir_loss_terms(step)
        loss = (subject.forward(step, training=True)[0] - target).square().mean() + terms['heir_loss']
        loss.backward()
        subject.optimizer_step()

    advance(model, 101)
    packet = copy.deepcopy(model.capture())
    rng = torch.get_rng_state()
    restored = RIGID.RigidResidualTree(copy.deepcopy(model.config))
    restored.restore(packet)
    torch.set_rng_state(rng)
    advance(model, 102)
    torch.set_rng_state(rng)
    advance(restored, 102)
    for actual, expected in zip(restored.parameters(), model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.heir_past['quaternion'].shape == model.heir_past['quaternion'].shape


def test_partition_byte_acceptance_keeps_previous_membership_without_pose_loss():
    import tempfile

    model = se3_hierarchy_fixture()
    install_residual_fixture_motion(model)
    model.update_topology(100)
    model.update_topology(200)
    with tempfile.TemporaryDirectory() as directory:
        export_fixture_frame(model, directory)
        previous = model.codec_previous_state
        model.setup(previous.xyz, previous.rotation, 1., frame_index=2)
        model.previous_partition = model.owner.clone()
        model.mark_observed(torch.ones(50, dtype=torch.bool))
        # A spurious reassignment on an otherwise unchanged fitted frame.
        model.owner[0] = 2
        target_xyz, target_q = (value.detach().clone() for value in model.forward(100))
        model._select_region_partition(100)
        assert model.last_metrics['tree_edit/previous_partition_retained'] == 1
        assert int(model.owner[0]) == int(previous.owner[0])
        assert model.last_metrics['tree_edit/retained_partition_bytes'] <= model.last_metrics['tree_edit/proposed_partition_bytes']
        torch.testing.assert_close(model.forward(100)[0], target_xyz, rtol=0, atol=9e-5)
        torch.testing.assert_close(model.forward(100)[1], target_q, rtol=0, atol=9e-4)


def test_old_heir_checkpoint_fields_receive_legacy_geometry_defaults():
    model = residual_hierarchy_fixture()
    install_residual_fixture_motion(model)
    model.update_topology(100)
    packet = copy.deepcopy(model.capture())
    packet['config']['heir'].pop('geometry')
    packet['config']['heir'].pop('history_frames')
    for key in ('heir_past', 'previous_partition', 'motion_history'):
        packet.pop(key)
    restored = RIGID.RigidResidualTree(copy.deepcopy(model.config))
    restored.restore(packet)
    assert restored.config.heir.geometry == 'linear' and not restored.motion_history.records


def test_cuda_temporal_se3_tree_keeps_history_on_cpu_and_preserves_stream_pose():
    import tempfile

    if not torch.cuda.is_available():
        return
    model = se3_hierarchy_fixture(device='cuda')
    install_residual_fixture_motion(model)
    model.update_topology(100)
    terms = model.heir_loss_terms(101)
    assert all(bool(torch.isfinite(value)) for value in terms.values())
    model.update_topology(200)
    with tempfile.TemporaryDirectory() as directory:
        export_fixture_frame(model, directory)
        assert model.motion_history.records[0]['source'].device.type == 'cpu'
        first = model.codec_previous_state
        model.setup(first.xyz.cuda(), first.rotation.cuda(), 1., frame_index=2)
        model.mark_observed(torch.ones(50, device='cuda', dtype=torch.bool))
        with torch.no_grad():
            model.parents[:] = torch.tensor([-1, 0, 1], device='cuda')
            model.region_translation[1, 0] = .03
            model.region_translation[2, 0] = -.029
        target = model.forward(100)[0].detach().cpu().clone()
        model.update_topology(100)
        export_fixture_frame(model, directory)
        torch.testing.assert_close(model.codec_previous_state.xyz, target, rtol=0, atol=9e-5)
        assert model.codec_previous_state.parents.tolist() == [-1, 0, 0]


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


def motion_cluster_fixture():
    model = RIGID.RigidResidualTree(RIGID.RigidTreeConfig(
        region_count=8193, adaptive_clustering=True,
        motion_steps=100, birth_steps=100, shared_steps=0,
        bootstrap_tree_step=20, tree_edit_step=60, rate_loss_weight=0.,
        motion_xyz_step_extent=1e-4, motion_rotation_step_rad=1e-4,
        gs_xyz_step_extent=1e-3, gs_rotation_step_rad=1e-3,
        clustering=CLUSTER.MotionClusteringConfig(
            translation_tolerance_extent=.003, rotation_tolerance_rad=.003,
            min_cluster_gs=4, split_noise_steps=1.)))
    torch.manual_seed(17)
    xyz = torch.randn(16, 3) * .2
    xyz[8:] += torch.tensor([10., 0., 0.])
    rotation = torch.tensor([[1., 0., 0., 0.]] * len(xyz))
    model.setup(xyz, rotation, 1., frame_index=1)
    model.mark_observed(torch.ones(len(xyz), dtype=torch.bool))
    return model


def test_shared_rigid_motion_groups_ignore_spatial_separation():
    import numpy as np
    torch.manual_seed(11)
    source = torch.randn(24, 3, dtype=torch.float64)
    source[12:] += 100
    q = RIGID.rotation_vector_quaternion(torch.tensor([[0., 0., .2]], dtype=torch.float64))
    quaternion = q.expand(len(source), -1)
    target = RIGID.quaternion_rotate(quaternion, source) + torch.tensor([.1, .2, -.1])
    config = CLUSTER.MotionClusteringConfig(translation_tolerance_extent=1e-5,
        rotation_tolerance_rad=1e-5, min_cluster_gs=4)
    groups = CLUSTER.shared_motion_groups(source.numpy(), target.numpy(), quaternion.numpy(),
                                         np.ones(len(source)), 1., config)
    assert len(groups) == 1
    assert groups[0]['rows'].tolist() == list(range(len(source)))
    assert groups[0]['position_error'].max() < 1e-10
    assert groups[0]['rotation_error'].max() < 1e-10


def test_motion_clustering_count_follows_distinct_rigid_motions():
    import numpy as np
    torch.manual_seed(12)
    source = torch.randn(16, 3, dtype=torch.float64)
    vectors = torch.zeros_like(source)
    vectors[:8, 2] = .2
    vectors[8:, 1] = -.3
    quaternion = RIGID.rotation_vector_quaternion(vectors)
    target = RIGID.quaternion_rotate(quaternion, source)
    target[:8] += torch.tensor([.1, 0., 0.])
    target[8:] += torch.tensor([0., .2, 0.])
    config = CLUSTER.MotionClusteringConfig(translation_tolerance_extent=.001,
        rotation_tolerance_rad=.001, min_cluster_gs=4)
    groups = CLUSTER.shared_motion_groups(source.numpy(), target.numpy(), quaternion.numpy(),
                                         np.ones(len(source)), 1., config)
    assert [group['rows'].tolist() for group in groups] == [list(range(8)), list(range(8, 16))]
    # Repeat with the same input: ownership does not depend on RNG or ordering
    # in a hash table, and rotating points are fitted as one rigid motion.
    repeated = CLUSTER.shared_motion_groups(source.numpy(), target.numpy(), quaternion.numpy(),
                                           np.ones(len(source)), 1., config)
    assert [group['rows'].tolist() for group in repeated] == [group['rows'].tolist() for group in groups]


def test_cluster_split_compares_motion_amplitude_with_parent_and_noise():
    zero = [0., 0., 0.]
    assert CLUSTER.residual_exceeds_parent([.2, 0., 0.], zero, [.1, 0., 0.], zero, .01, .01)
    assert CLUSTER.residual_exceeds_parent(zero, [0., .2, 0.], zero, [0., .1, 0.], .01, .01)
    assert not CLUSTER.residual_exceeds_parent([.05, 0., 0.], zero, [.1, 0., 0.], zero, .01, .01)
    assert not CLUSTER.residual_exceeds_parent([.001, 0., 0.], zero, zero, zero, .01, .01)


def test_adaptive_tree_starts_minimal_and_bootstraps_inside_normal_motion_steps():
    model = motion_cluster_fixture()
    assert len(model.parents) == 2  # region_count=8193 is not a cluster target.
    assert model.config.motion_steps + model.config.birth_steps == 200
    assert model.config.shared_steps == 0
    assert model.stage(19) == 'motion_fit'
    assert model.stage(20) == 'tree_edit'
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
    target = [value.detach().clone() for value in model.forward(20)]
    model.last_step = 20
    edits = model.update_topology(20)
    assert any(edit['kind'] == 'split' for edit in edits)
    assert len(model.parents) == 3
    assert model.bootstrap_finalized
    assert model.stage(60) == 'motion_fit'
    assert model.frame_reference_state.frame_index == 0
    torch.testing.assert_close(model.frame_reference_state.xyz, model.base_xyz, rtol=0, atol=0)
    assert not torch.equal(model.frame_reference_state.xyz, target[0])
    actual = model.forward(20)
    torch.testing.assert_close(actual[0], target[0], rtol=0, atol=9e-4)
    torch.testing.assert_close(actual[1], target[1], rtol=0, atol=9e-4)
    model._topology_updated = False
    model.frame_index = 9
    assert model.stage(60) == 'motion_fit'
    model.frame_index = 10
    assert model.stage(60) == 'tree_edit'


def test_motion_tree_edit_preserves_unaffected_adam_rows():
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
    for parameter in model.parameters():
        model.optimizer.state[parameter] = dict(step=torch.tensor(3.),
            exp_avg=torch.full_like(parameter, .02), exp_avg_sq=torch.full_like(parameter, .03))
    untouched = {name: model.optimizer.state[getattr(model, name)]['exp_avg'].clone()
                 for name in ('gs_translation', 'gs_rotation', 'region_translation', 'region_rotation')}
    model.last_step = 20
    model.update_topology(20)
    for name in ('gs_translation', 'gs_rotation'):
        torch.testing.assert_close(model.optimizer.state[getattr(model, name)]['exp_avg'][8:],
                                   untouched[name][8:], rtol=0, atol=0)
        torch.testing.assert_close(model.optimizer.state[getattr(model, name)]['step'],
                                   torch.tensor(3.), rtol=0, atol=0)
    for name in ('region_translation', 'region_rotation'):
        torch.testing.assert_close(model.optimizer.state[getattr(model, name)]['exp_avg'][:2],
                                   untouched[name], rtol=0, atol=0)


def test_variable_tree_checkpoint_preserves_next_optimizer_update():
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
    model.last_step = 20
    model.update_topology(20)
    recovered = RIGID.RigidResidualTree(model.config)
    recovered.restore(copy.deepcopy(model.capture()))
    assert len(recovered.parents) == 3
    assert recovered.bootstrap_finalized
    for subject in (model, recovered):
        loss = (subject.forward(21, training=True)[0] - subject.base_xyz - .08).square().mean()
        loss.backward()
        subject.optimizer_step()
    for name in ('region_translation', 'region_rotation', 'gs_translation', 'gs_rotation'):
        torch.testing.assert_close(getattr(model, name), getattr(recovered, name), rtol=0, atol=0)


def test_dynamic_codec_replays_region_growth_and_merge_redirect():
    source = motion_cluster_fixture()
    prior = CODEC.DecodedState(source.base_xyz, source.base_rotation,
                              source.owner.clone(), source.parents.clone(), 0)
    parents = torch.tensor([-1, 0, 1])
    owner = prior.owner.clone()
    owner[:8] = 2
    pivots = RIGID.region_pivots(prior.xyz, owner, parents)
    rt, rr = torch.zeros(3, 3, dtype=torch.int16), torch.zeros(3, 3, dtype=torch.int16)
    rt[2, 0] = 1000
    gt, gr = torch.zeros(16, 3, dtype=torch.int16), torch.zeros(16, 3, dtype=torch.int16)
    empty = torch.empty(0, dtype=torch.long)
    packet = CODEC.encode_motion(prior, 1, 1., torch.arange(16), owner, parents, rt, rr, gt, gr,
        empty, torch.empty(0, 3), torch.empty(0, 4), 1e-4, 1e-4,
        gs_translation_step_extent=1e-3, gs_rotation_step_rad=1e-3,
        region_map=torch.arange(2), pivots=pivots)
    assert packet[:4] == CODEC.DYNAMIC_MAGIC
    decoded = CODEC.decode_motion(CODEC.compress_motion_packet(packet), prior)
    expected = RIGID.compose_pose(prior.xyz, prior.rotation, owner, parents, pivots,
                                  rt.float() * 1e-4, rr.float() * 1e-4, gt.float() * 1e-3, gr.float() * 1e-3)
    torch.testing.assert_close(decoded.xyz, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(decoded.parents, parents, rtol=0, atol=0)
    # A merge changes the region count; redirecting node2's members takes one
    # mapping entry rather than explicit per-GS owner edits.
    merged_parents = torch.tensor([-1, 0])
    merged_owner = torch.ones(16, dtype=torch.long)
    merged_pivots = RIGID.region_pivots(decoded.xyz, merged_owner, merged_parents)
    packet2 = CODEC.encode_motion(decoded, 2, 1., torch.arange(16), merged_owner,
        merged_parents, rt[:2] * 0, rr[:2] * 0, gt, gr, empty,
        torch.empty(0, 3), torch.empty(0, 4), 1e-4, 1e-4,
        gs_translation_step_extent=1e-3, gs_rotation_step_rad=1e-3,
        region_map=torch.tensor([0, 1, 1]), pivots=merged_pivots)
    merged = CODEC.decode_motion(packet2, decoded)
    torch.testing.assert_close(merged.owner, merged_owner, rtol=0, atol=0)
    torch.testing.assert_close(merged.xyz, decoded.xyz, rtol=0, atol=0)
    assert len(merged.parents) == 2


def test_two_frame_reference_uses_frame_zero_attributes_and_learned_layout():
    import tempfile
    import numpy as np
    from types import SimpleNamespace
    trainer = importlib.import_module('pipeline.ReCon-GS.trainer')
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
    model.last_step = 20
    model.update_topology(20)
    gaussians = SimpleNamespace(_xyz=model.base_xyz, _rotation=model.base_rotation,
        _scaling=torch.zeros(16, 3), _opacity=torch.zeros(16, 1),
        _features_dc=torch.zeros(16, 1, 3), _features_rest=torch.zeros(16, 3, 3))
    module = SimpleNamespace(gaussians=gaussians,
        config=SimpleNamespace(rigid_tree=model.config),
        initial_reference=dict(original_count=16, retained_count=16, removed_count=0, threshold=0.))
    with tempfile.TemporaryDirectory() as directory:
        worker = SimpleNamespace(output_dir=Path(directory), module=module,
                                 datamodule=SimpleNamespace(cameras_extent=1.))
        trainer.Trainer._write_reference(worker, 0, model.frame_reference_state)
        with np.load(worker.decoder_reference_path, allow_pickle=False) as packet:
            assert int(packet['frame_index']) == 0
            assert int(packet['region_count']) == 3
            torch.testing.assert_close(torch.from_numpy(packet['xyz']), model.base_xyz, rtol=0, atol=0)
            torch.testing.assert_close(torch.from_numpy(packet['owner']), model.owner, rtol=0, atol=0)


def test_child_residual_below_parent_does_not_split_even_when_nonzero():
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.region_translation[1, 0] = .2
        model.gs_translation[:8, 0] = .1
    model.last_step = 20
    target = model.forward(20)[0].detach().clone()
    edits = model.update_topology(20)
    assert not any(edit['kind'] == 'split' for edit in edits)
    assert len(model.parents) == 2
    torch.testing.assert_close(model.forward(20)[0], target, rtol=0, atol=9e-4)


def test_merging_shared_motion_compacts_nodes_and_keeps_pose_and_wire_bytes():
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
        model.gs_translation[8:, 0] = .12
    model.last_step = 20
    model.update_topology(20)
    assert len(model.parents) == 4
    with torch.no_grad():
        model.parents[:] = torch.tensor([-1, 0, 1, 1])
        model.region_translation[3] = model.region_translation[2]
    prior = CODEC.DecodedState(model.base_xyz.detach().clone(),
        model.base_rotation.detach().clone(), model.owner.clone(), model.parents.clone(), 9)
    model.frame_index = 10
    model.frame_reference_state = model.codec_previous_state = prior
    model.region_map = torch.arange(4)
    model._topology_updated = False
    model._metadata_cache = None
    before = [value.detach().clone() for value in model.forward(60)]
    edits = model.update_topology(60)
    assert any(edit['kind'] == 'merge' for edit in edits)
    assert len(model.parents) == 3
    torch.testing.assert_close(model.forward(60)[0], before[0], rtol=0, atol=9e-4)
    rt, rr, gt, gr = model._values(hard=True)
    packet = CODEC.encode_motion(prior, 10, model.extent, model.previous_rows,
        model.owner, model.parents, rt, rr, gt, gr, torch.empty(0, dtype=torch.long),
        torch.empty(0, 3), torch.empty(0, 4), model.config.motion_xyz_step_extent,
        model.config.motion_rotation_step_rad, gs_translation_step_extent=model.config.gs_xyz_step_extent,
        gs_rotation_step_rad=model.config.gs_rotation_step_rad,
        region_map=model.region_map, pivots=model.pivots)
    assert int(model._hard_motion_bytes()) + model._metadata_bytes() == len(packet)
    decoded = CODEC.decode_motion(packet, prior)
    torch.testing.assert_close(decoded.xyz, model.forward(60)[0], rtol=0, atol=0)
    torch.testing.assert_close(decoded.parents, model.parents, rtol=0, atol=0)


def test_new_motion_region_survives_its_first_proximal_projection():
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
    model.last_step = 20
    model.update_topology(20)
    model.config.rate_loss_weight = 1.
    retained = model.region_translation[2].detach().clone()
    # No Adam variance exists for a freshly appended region. It must get a
    # normal backward update before being eligible for the sparse projection.
    model._proximal_region_blocks()
    torch.testing.assert_close(model.region_translation[2], retained, rtol=0, atol=0)


def test_module_two_frame_initialization_runs_exactly_100_plus_100_steps():
    import tempfile
    from types import SimpleNamespace
    trainer_module = importlib.import_module('pipeline.ReCon-GS.trainer')
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
    reference_xyz = model.base_xyz.clone()
    gaussians = SimpleNamespace(_xyz=reference_xyz.clone(), _rotation=model.base_rotation.clone(),
        _scaling=torch.zeros(16, 3), _opacity=torch.zeros(16, 1),
        _features_dc=torch.zeros(16, 1, 3), _features_rest=torch.zeros(16, 3, 3),
        xyz_gradient_accum=torch.zeros(16, 1), denom=torch.zeros(16, 1),
        add_densification_stats=lambda *args: None)
    subject = SimpleNamespace(config=RIGID_MODULE.ModuleConfig(rigid_tree=model.config),
        motion=model, gaussians=gaussians, motion_committed=False,
        motion_steps=100, total_steps=200, birth_control=dict(tree_edit_count=0),
        tree_statistics=[], reference_count=16,
        births=SimpleNamespace(_xyz=torch.empty(0, 3)),
        initial_reference=dict(original_count=16, retained_count=16, removed_count=0, threshold=0.))
    events = []
    with tempfile.TemporaryDirectory() as directory:
        worker = SimpleNamespace(global_step=0, output_dir=Path(directory), module=subject,
            bootstrap_pending=True, datamodule=SimpleNamespace(cameras_extent=1.))
        subject.trainer = worker
        subject.current_frame = 1
        subject.output_dir = Path(directory)
        subject._write_tree_statistics = lambda step: RIGID_MODULE.Module._write_tree_statistics(subject, step)

        def write_reference(frame_index, state):
            events.append(('reference', worker.global_step))
            torch.testing.assert_close(gaussians._xyz, reference_xyz, rtol=0, atol=0)
            trainer_module.Trainer._write_reference(worker, frame_index, state)

        def begin_births():
            events.append(('birth', worker.global_step))
            subject.motion_committed = True
            gaussians._xyz = model.forward(100)[0].detach().clone()

        worker._write_reference = write_reference
        subject._begin_births = begin_births
        subject.finalize_frame = lambda: events.append(('final', worker.global_step))
        target = model.forward(1)[0].detach().clone()
        motion_updates = 0
        for step in range(1, 201):
            worker.global_step = step
            if not subject.motion_committed:
                loss = (model.forward(step, training=True)[0] - target).square().mean()
                loss.backward()
                motion_updates += 1
            package = dict(visibility_filter=torch.ones(16, dtype=torch.bool),
                           viewspace_points=torch.zeros(16, 3))
            RIGID_MODULE.Module.post_training_step(subject, package)
        assert motion_updates == 100
        assert events == [('reference', 100), ('birth', 100), ('final', 200)]
        assert not worker.bootstrap_pending
        assert len(subject.tree_statistics) == 1
        assert subject.tree_statistics[0]['step'] == 20
        assert (Path(directory) / 'tree_stats/frame_001_step_020.csv').is_file()


def test_adaptive_configuration_rejects_extra_optimization_phases():
    config = motion_cluster_fixture().config
    for field_name, value in (('motion_steps', 1200), ('birth_steps', 200),
                              ('shared_steps', 20), ('tree_edit_interval', 5)):
        invalid = copy.deepcopy(config)
        setattr(invalid, field_name, value)
        try:
            invalid.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid adaptive schedule accepted: {field_name}')


def test_tree_statistics_distinguish_direct_ownership_and_inherited_motion():
    import csv
    import tempfile

    model = rigid_tree_fixture()
    with torch.no_grad():
        model.owner[:] = torch.tensor([0, 1, 2, 2])
        model.parents[:] = torch.tensor([-1, 0, 1])
        model.region_translation[1, 0] = .02
        model.gs_translation[3, 0] = .04
    original_owner = model.owner.clone()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'tree_stats/frame_001_step_060.csv'
        record = TREE_STATS.write_tree_statistics(path, model, 2, 60)
        with path.open(newline='') as file:
            rows = list(csv.DictReader(file))
    assert record['nodes'] == 3
    assert record['original_frame0_gs'] == 2 and record['old_gs'] == 4
    assert [int(row['original_frame0_gs_count']) for row in rows] == [1, 1, 0]
    assert [int(row['old_gs_count']) for row in rows] == [1, 1, 2]
    assert sum(float(row['old_gs_percent']) for row in rows) == 100.
    assert sum(float(row['original_frame0_gs_percent']) for row in rows) == 100.
    assert rows[2]['parent_id'] == '1' and rows[2]['depth'] == '2'
    assert rows[2]['local_motion_nonzero'] == '0'
    assert rows[2]['inherited_motion_nonzero'] == '1'
    assert rows[2]['old_gs_with_local_residual'] == '1'
    torch.testing.assert_close(model.owner, original_owner, rtol=0, atol=0)


def test_tree_statistics_written_on_update_even_without_topology_changes():
    import tempfile
    from types import SimpleNamespace

    model = motion_cluster_fixture()
    model.frame_index = 10
    model.forward(60, training=True)[0].sum().mul(0).backward()
    with tempfile.TemporaryDirectory() as directory:
        subject = SimpleNamespace(config=RIGID_MODULE.ModuleConfig(rigid_tree=model.config),
            motion=model, motion_committed=False, motion_steps=100, total_steps=200,
            tree_statistics=[], reference_count=16, current_frame=10,
            output_dir=Path(directory), birth_control={},
            trainer=SimpleNamespace(global_step=60))
        subject._write_tree_statistics = lambda step: RIGID_MODULE.Module._write_tree_statistics(subject, step)
        package = dict(visibility_filter=torch.ones(16, dtype=torch.bool))
        RIGID_MODULE.Module.post_training_step(subject, package)
        assert not model.topology_edits
        assert len(subject.tree_statistics) == 1
        assert (Path(directory) / 'tree_stats/frame_010_step_060.csv').is_file()
        subject.trainer.global_step = 61
        model.forward(61, training=True)[0].sum().mul(0).backward()
        RIGID_MODULE.Module.post_training_step(subject, package)
        assert len(subject.tree_statistics) == 1


def test_module_checkpoint_keeps_tree_statistics_and_reads_older_checkpoints():
    from types import SimpleNamespace

    config = RIGID_MODULE.ModuleConfig()
    record = dict(path='tree_stats/frame_010_step_060.csv', frame=10, step=60)
    subject = SimpleNamespace(config=config, current_frame=10,
        gaussians=SimpleNamespace(capture=lambda: None, restore=lambda *args: None),
        renderer=SimpleNamespace(),
        frame_prepared=False, frame_finalized=True, motion_committed=False,
        reference_count=16, frame_report={}, birth_control={},
        tree_statistics=[record], initial_reference=dict(
            source='native_frame0_checkpoint', threshold=0., original_count=16,
            retained_count=16, removed_count=0),
        births=SimpleNamespace(source_rows=None), _freeze_base=lambda: None)
    packet = {}
    RIGID_MODULE.Module.on_save_checkpoint(subject, packet)
    subject.tree_statistics = []
    RIGID_MODULE.Module.on_load_checkpoint(subject, packet)
    assert subject.tree_statistics == [record]
    packet['rigid_tree_frame'].pop('tree_statistics')
    RIGID_MODULE.Module.on_load_checkpoint(subject, packet)
    assert subject.tree_statistics == []


def test_heir_ancestor_decoder_reconstructs_known_translation_hierarchy():
    absolute = torch.tensor([[[0., 0.], [1., .2], [1.3, .4]]], dtype=torch.float64)
    hierarchy = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    relative = absolute - hierarchy @ absolute
    reconstructed = HEIR.reconstruct_motion(relative, hierarchy, 3)
    torch.testing.assert_close(reconstructed, absolute, rtol=0, atol=1e-12)


def test_heir_attention_and_straight_through_sampling_have_parent_gradients():
    torch.manual_seed(23)
    motion = torch.tensor([[0., 0.], [1., 0.], [1., .2], [-.5, .1]])
    config = HEIR.load_heir_config(PROJECT_ROOT / 'config/heir_hierarchy_core.yaml')
    config.hierarchy_samples = 2
    model = HEIR.HeirMotionHierarchy(2, config)
    assert model.make_optimizer().param_groups[0]['lr'] == config.learning_rate
    weights = model.parent_weights(motion)
    torch.testing.assert_close(weights[1:].sum(-1), torch.ones(3), rtol=0, atol=1e-6)
    assert torch.equal(weights[0], torch.zeros(4))
    assert torch.equal(weights.diag(), torch.zeros(4))
    sampled = model.sample_hierarchy(weights, 1.6)
    assert torch.equal(sampled[1:].sum(-1), torch.ones(3))
    assert torch.equal(sampled.diag(), torch.zeros(4))
    terms = model.loss_terms(motion, .5)
    assert bool(torch.isfinite(terms['heir_loss']))
    terms['heir_loss'].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(float(parameter.grad.abs().sum()) for parameter in model.parameters()) > 0


def test_heir_export_breaks_cycles_and_keeps_root_and_depth_limit():
    scores = torch.tensor([[0., 0., 0.], [.1, 0., .9], [.1, .9, 0.]])
    parents = HEIR.acyclic_parent_projection(scores, 2)
    assert parents[0] == -1
    assert parents[1] != 1 and parents[2] != 2
    order = RIGID._topological_order(parents)
    assert len(order) == 2
    depth = [0] * len(parents)
    for node in order:
        depth[node] = depth[int(parents[node])] + 1
    assert max(depth) <= 2
    shallow = HEIR.acyclic_parent_projection(scores, 1)
    assert shallow.tolist() == [-1, 0, 0]


def test_heir_identity_only_graph_has_finite_zero_loss():
    model = HEIR.HeirMotionHierarchy(6, HEIR.HeirHierarchyConfig())
    motion = torch.zeros(1, 6)
    loss = model.loss_terms(motion, 0)['heir_loss']
    torch.testing.assert_close(loss, torch.tensor(0.), rtol=0, atol=0)
    loss.backward()
    assert model.infer_parents(motion, 8).tolist() == [-1]


def test_motion_warm_start_transfers_global_increment_and_newborn_source():
    import tempfile
    model = motion_cluster_fixture()
    model.config.motion_warm_start = True
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
    model.last_step = 20
    model.update_topology(20)
    with torch.no_grad():
        model.region_rotation[2, 2] = .2
    global_q, global_b = model._global_regions()
    _, _, gs_t, gs_r = model._values()
    moved, rotation = (value.detach().clone() for value in model.forward(100))
    births = torch.tensor([0])
    moved = torch.cat((moved, moved[:1] + torch.tensor([[.025, 0., 0.]])))
    rotation = torch.cat((rotation, rotation[:1]))
    model.register_materialized_layout(moved, rotation, births)
    owner = model.materialized_owner.clone()
    expected_xyz = RIGID.quaternion_rotate(global_q[owner], moved) + global_b[owner]
    expected_xyz += torch.cat((gs_t, gs_t[births]))
    expected_rotation = RIGID.quaternion_multiply(
        RIGID.rotation_vector_quaternion(torch.cat((gs_r, gs_r[births]))),
        RIGID.quaternion_multiply(global_q[owner], rotation))
    with tempfile.TemporaryDirectory() as directory:
        model.export_motion(Path(directory) / 'motion_001.bin')
    model.setup(model.codec_previous_state.xyz, model.codec_previous_state.rotation,
                1., frame_index=2)
    actual_xyz, actual_rotation = model.forward(1)
    torch.testing.assert_close(actual_xyz, expected_xyz, rtol=0, atol=5e-4)
    torch.testing.assert_close(actual_rotation, expected_rotation, rtol=0, atol=5e-4)
    assert len(model.owner) == 17
    assert model.codec_previous_state.frame_index == 1
    assert model.stage(20) == 'motion_fit'


def test_retired_spatial_training_cannot_silently_run_1400_steps():
    from types import SimpleNamespace
    trainer = importlib.import_module('pipeline.ReCon-GS.trainer')
    subject = SimpleNamespace(module=SimpleNamespace(config=SimpleNamespace(
        rigid_tree=RIGID.RigidTreeConfig(motion_steps=1200, birth_steps=200))))
    try:
        trainer.Trainer.training_loop(subject)
    except ValueError as error:
        assert 'retired' in str(error)
    else:
        raise AssertionError('Retired fixed spatial training was accepted')


def test_fully_inherited_leaf_merges_without_resetting_gs_adam():
    model = motion_cluster_fixture()
    with torch.no_grad():
        model.gs_translation[:8, 0] = .1
        model.gs_translation[8:, 0] = .12
    model.last_step = 20
    model.update_topology(20)
    assert len(model.parents) == 4
    with torch.no_grad():
        model.parents[:] = torch.tensor([-1, 0, 1, 2])
        model.region_translation[3] = 0
        model.region_rotation[3] = 0
    for parameter in (model.gs_translation, model.gs_rotation):
        model.optimizer.state[parameter] = dict(step=torch.tensor(10.),
            exp_avg=torch.full_like(parameter, .07), exp_avg_sq=torch.full_like(parameter, .09))
    before = model.forward(60)[0].detach().clone()
    edits = model._collapse_identity_leaves()
    model._compact_motion_regions()
    assert any(edit['kind'] == 'merge' for edit in edits)
    assert len(model.parents) == 3
    torch.testing.assert_close(model.forward(60)[0], before, rtol=0, atol=1e-6)
    for parameter in (model.gs_translation, model.gs_rotation):
        torch.testing.assert_close(model.optimizer.state[parameter]['exp_avg'],
                                   torch.full_like(parameter, .07), rtol=0, atol=0)


def residual_hierarchy_fixture(device='cpu', frame=1):
    torch.manual_seed(41)
    config = RIGID.RigidTreeConfig(
        adaptive_clustering=True, residual_hierarchy=True, region_count=1,
        bootstrap_tree_step=100, tree_edit_step=100, tree_refine_steps=100,
        motion_xyz_step_extent=1e-5, motion_rotation_step_rad=1e-4,
        gs_xyz_step_extent=1e-4, gs_rotation_step_rad=1e-3,
        rate_loss_weight=0., relative_motion_loss_weight=0.,
        hdbscan=RESIDUAL_CLUSTER.ResidualHdbscanConfig(
            min_cluster_size=8, min_samples=4, cluster_selection_epsilon=.01))
    xyz = torch.randn(50, 3).to(device)
    rotation = RIGID.rotation_vector_quaternion(torch.randn(50, 3).to(device) * .05)
    model = RIGID.RigidResidualTree(config)
    model.setup(xyz, rotation, 1., frame_index=frame)
    observed = torch.ones(50, dtype=torch.bool, device=device)
    observed[-1] = False
    model.mark_observed(observed)
    return model


@torch.no_grad()
def install_residual_fixture_motion(model):
    model.gs_translation[:16] = model.base_xyz.new_tensor([.03, 0., 0.])
    model.gs_translation[16:32] = model.base_xyz.new_tensor([-.03, .05, 0.])
    model.gs_translation[48] = model.base_xyz.new_tensor([.8, .2, .1])
    model.gs_translation[49] = model.base_xyz.new_tensor([.11, -.04, 0.])


def test_hdbscan_uses_shared_rigid_motion_and_separates_unknown_from_static():
    import numpy as np
    from scipy.spatial.transform import Rotation

    source = np.random.default_rng(14).normal(size=(50, 3)).astype(np.float32).astype(np.float64)
    source[::2] += 20.  # Spatially distant members still share one rigid motion.
    target = source.copy()
    target[:16] += [.03125, 0., 0.]
    rotation = Rotation.from_rotvec([0., 0., .04])
    target[16:32] = rotation.apply(source[16:32]) + [-.03, .05, 0.]
    target[48] += [.8, .2, .1]
    quaternion = np.zeros((50, 4))
    quaternion[:, 0] = 1.
    quaternion[16:32] = rotation.as_quat()[[3, 0, 1, 2]]
    observations = np.ones(50, dtype=np.int64)
    observations[49] = 0  # Zero fitted motion with no observations is unknown.
    config = RESIDUAL_CLUSTER.ResidualHdbscanConfig(
        min_cluster_size=8, min_samples=4, cluster_selection_epsilon=.01)
    groups, labels = RESIDUAL_CLUSTER.hdbscan_motion_regions(
        source, target, quaternion, observations, 1., config)
    assert [group.tolist() for group in groups] == [list(range(16)), list(range(16, 32))]
    assert labels[32:48].tolist() == [-2] * 16
    assert labels[48:].tolist() == [-1, -3]


def test_center_static_candidates_keep_independent_covariance_rotation():
    model = residual_hierarchy_fixture()
    model.config.hdbscan.feature_mode = 'point_displacement'
    with torch.no_grad():
        model.gs_rotation[:16, 2] = .01
    before_xyz, before_q = (value.detach().clone() for value in model.forward(100))
    model.update_topology(100)
    assert model.parents.tolist() == [-1]
    assert model.cluster_labels[:16].tolist() == [-2] * 16
    assert bool(model.gs_rotation[:16].ne(0).any(-1).all())
    after_xyz, after_q = model.forward(100)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=0)
    torch.testing.assert_close(after_q, before_q, rtol=0, atol=1e-7)


def test_gmm_bic_recovers_observed_translation_modes_and_keeps_static_unknown():
    import numpy as np

    random = np.random.default_rng(63)
    source = random.normal(size=(82, 3))
    target = source.copy()
    target[:32] += np.array([.03, 0., 0.]) + random.normal(scale=1e-5, size=(32, 3))
    target[32:64] += np.array([-.03, .05, 0.]) + random.normal(scale=1e-5, size=(32, 3))
    quaternion = np.zeros((82, 4))
    quaternion[:, 0] = 1.
    observations = np.ones(82, dtype=np.int64)
    observations[-1] = 0
    features = RESIDUAL_CLUSTER.ResidualHdbscanConfig(feature_mode='point_displacement')
    config = RESIDUAL_CLUSTER.ResidualGmmConfig(components=(1, 2, 3),
        min_cluster_size=8, max_fit_points=100, random_seed=63)
    groups, labels, metrics = RESIDUAL_CLUSTER.gmm_motion_regions(
        source, target, quaternion, observations, 1., features, config)
    assert metrics['gmm_selected_components'] == 2
    assert metrics['gmm_converged'] == 1
    assert [group.tolist() for group in groups] == [list(range(32)), list(range(32, 64))]
    assert labels[64:81].tolist() == [-2] * 17 and labels[81] == -3


def test_gmm_region_builder_and_checkpoint_preserve_pose_with_bic_configuration():
    model = residual_hierarchy_fixture()
    model.config.hdbscan.feature_mode = 'point_displacement'
    model.config.residual_clustering_method = 'gmm'
    model.config.gmm = RESIDUAL_CLUSTER.ResidualGmmConfig(components=(2, 3),
        min_cluster_size=8, max_fit_points=100, random_seed=41)
    install_residual_fixture_motion(model)
    before_xyz, before_q = (value.detach().clone() for value in model.forward(100))
    edits = model.update_topology(100)
    assert edits[0]['method'] == 'gmm'
    assert len(model.parents) == 3
    assert model.last_metrics['tree_edit/gmm_selected_components'] >= 2
    restored = RIGID.RigidResidualTree(copy.deepcopy(model.config))
    restored.restore(copy.deepcopy(model.capture()))
    after_xyz, after_q = restored.forward(100)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=9e-5)
    torch.testing.assert_close(after_q, before_q, rtol=0, atol=9e-4)
    assert restored.config.residual_clustering_method == 'gmm'


def test_identity_only_residual_fit_has_finite_losses_and_valid_codec_metadata():
    model = residual_hierarchy_fixture()
    assert model.parents.tolist() == [-1]
    assert model.owner.tolist() == [0] * 50
    for step in (1, 50, 99):
        terms = model.loss_terms(step)
        assert all(bool(torch.isfinite(value)) for value in terms.values())
        torch.testing.assert_close(terms['relative_motion'], torch.tensor(0.), rtol=0, atol=0)
    layout, owner = CODEC.dynamic_layout_bytes(model.frame_reference_state,
        model.previous_rows, model.parents, model.pivots, model.region_map)
    assert layout == b'\x00'
    assert torch.equal(owner, model.owner)


def test_hdbscan_displacements_do_not_treat_covariance_rotation_as_body_motion():
    import numpy as np
    from scipy.spatial.transform import Rotation

    random = np.random.default_rng(52)
    source = random.normal(size=(50, 3)).astype(np.float32).astype(np.float64)
    source[::2] += 20.
    target = source.copy()
    target[:16] += [.03125, 0., 0.]
    target[16:32] += [-.0625, .046875, 0.]
    target[48] += [.8, .2, .1]
    q = np.zeros((50, 4))
    q[:, 0] = 1.
    q[:32] = Rotation.from_rotvec(random.normal(size=(32, 3)) * .02).as_quat()[:, [3, 0, 1, 2]]
    observations = np.ones(50, dtype=np.int64)
    observations[-1] = 0
    config = RESIDUAL_CLUSTER.ResidualHdbscanConfig(feature_mode='point_displacement',
        min_cluster_size=8, min_samples=4, cluster_selection_epsilon=.01)
    groups, labels = RESIDUAL_CLUSTER.hdbscan_motion_regions(source, target, q,
        observations, 1., config)
    assert [group.tolist() for group in groups] == [list(range(16)), list(range(16, 32))]
    assert labels[48:].tolist() == [-1, -3]


def test_hdbscan_tree_edit_preserves_pose_and_gs_adam_while_keeping_noise_motion():
    model = residual_hierarchy_fixture()
    install_residual_fixture_motion(model)
    for parameter in (model.gs_translation, model.gs_rotation):
        model.optimizer.state[parameter] = dict(step=torch.tensor(10.),
            exp_avg=torch.full_like(parameter, .07), exp_avg_sq=torch.full_like(parameter, .09))
    before_xyz, before_q = (value.detach().clone() for value in model.forward(100))
    edits = model.update_topology(100)
    assert edits[0]['method'] == 'hdbscan'
    assert len(model.parents) == 3
    assert model.owner[:16].tolist() == [1] * 16
    assert model.owner[16:32].tolist() == [2] * 16
    assert model.owner[32:].tolist() == [0] * 18
    assert model.cluster_labels[48:].tolist() == [-1, -3]
    after_xyz, after_q = model.forward(100)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=9e-5)
    torch.testing.assert_close(after_q, before_q, rtol=0, atol=9e-4)
    torch.testing.assert_close(model.gs_translation[48:],
        torch.tensor([[.8, .2, .1], [.11, -.04, 0.]]), rtol=0, atol=1e-7)
    for parameter in (model.gs_translation, model.gs_rotation):
        torch.testing.assert_close(model.optimizer.state[parameter]['exp_avg'],
            torch.full_like(parameter, .07), rtol=0, atol=0)


def test_residual_rebuild_returns_parent_cancelling_static_points_to_identity():
    model = residual_hierarchy_fixture()
    install_residual_fixture_motion(model)
    model.update_topology(100)
    with torch.no_grad():
        model.gs_translation[:16] = -model.region_translation[1]
    stationary_xyz, stationary_q = model.forward(100)
    torch.testing.assert_close(stationary_xyz[:16], model.base_xyz[:16], rtol=0, atol=1e-7)
    model.frame_index = 10
    model._topology_updated = False
    model.update_topology(100)
    assert model.owner[:16].tolist() == [0] * 16
    assert model.cluster_labels[:16].tolist() == [-2] * 16
    torch.testing.assert_close(model.gs_translation[:16], torch.zeros(16, 3), rtol=0, atol=0)
    after_xyz, after_q = model.forward(100)
    torch.testing.assert_close(after_xyz[:16], stationary_xyz[:16], rtol=0, atol=1e-7)
    torch.testing.assert_close(after_q[:16], stationary_q[:16], rtol=0, atol=1e-7)


def test_heir_refinement_checkpoint_preserves_next_stochastic_adam_step():
    model = residual_hierarchy_fixture()
    install_residual_fixture_motion(model)
    model.update_topology(100)
    target = model.forward(100)[0].detach().clone()

    def advance(subject, step):
        terms = subject.heir_loss_terms(step)
        loss = (subject.forward(step, training=True)[0] - target).square().mean()
        (loss + terms['heir_loss']).backward()
        subject.optimizer_step()

    advance(model, 101)
    checkpoint = copy.deepcopy(model.capture())
    rng = torch.get_rng_state()
    restored = RIGID.RigidResidualTree(copy.deepcopy(model.config))
    restored.restore(checkpoint)
    assert restored.motion_steps == 200 and restored.stage(102) == 'tree_refine'
    torch.set_rng_state(rng)
    advance(model, 102)
    torch.set_rng_state(rng)
    advance(restored, 102)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(model.heir_hierarchy.parameters(), restored.heir_hierarchy.parameters()):
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(model.heir_optimizer.state[left][key],
                restored.heir_optimizer.state[right][key], rtol=0, atol=0)
    restored.update_topology(200)
    completed = RIGID.RigidResidualTree(copy.deepcopy(model.config))
    completed.restore(copy.deepcopy(restored.capture()))
    assert completed.heir_finalized and completed.bootstrap_finalized
    assert all(not parameter.requires_grad for parameter in completed.heir_hierarchy.parameters())


def test_heir_parent_projection_preserves_rotated_gs_pose_and_bootstrap_reference():
    from unittest import mock

    model = residual_hierarchy_fixture()
    install_residual_fixture_motion(model)
    model.update_topology(100)
    with torch.no_grad():
        model.region_rotation[1, 2] = .04
        model.region_rotation[2, 1] = -.02
    target_xyz, target_q = (value.detach().clone() for value in model.forward(200))
    parents = torch.tensor([-1, 0, 1])
    with mock.patch.object(model.heir_hierarchy, 'infer_parents', return_value=parents):
        edits = model.update_topology(200)
    assert any(edit['kind'] == 'parent' for edit in edits)
    assert model.parents.tolist() == [-1, 0, 1]
    after_xyz, after_q = model.forward(200)
    torch.testing.assert_close(after_xyz, target_xyz, rtol=0, atol=9e-5)
    assert float(RIGID.quaternion_angle_error(after_q, target_q).max()) < 9e-4
    torch.testing.assert_close(model.frame_reference_state.xyz, model.base_xyz, rtol=0, atol=0)
    torch.testing.assert_close(model.frame_reference_state.rotation, model.base_rotation, rtol=0, atol=0)
    assert model.bootstrap_finalized and model.heir_finalized


def test_module_residual_tree_uses_300_steps_on_tree_frames_and_200_otherwise():
    import tempfile
    from types import SimpleNamespace

    trainer_module = importlib.import_module('pipeline.ReCon-GS.trainer')
    for frame in (1, 2, 10):
        model = residual_hierarchy_fixture(frame=frame)
        install_residual_fixture_motion(model)
        reference_xyz = model.base_xyz.clone()
        gaussians = SimpleNamespace(_xyz=reference_xyz.clone(), _rotation=model.base_rotation.clone(),
            _scaling=torch.zeros(50, 3), _opacity=torch.zeros(50, 1),
            _features_dc=torch.zeros(50, 1, 3), _features_rest=torch.zeros(50, 3, 3),
            xyz_gradient_accum=torch.zeros(50, 1), denom=torch.zeros(50, 1),
            add_densification_stats=lambda *args: None)
        subject = SimpleNamespace(config=RIGID_MODULE.ModuleConfig(rigid_tree=model.config),
            current_frame=frame, motion=model, gaussians=gaussians, motion_committed=False,
            birth_control=dict(tree_edit_count=0), tree_statistics=[], reference_count=50,
            births=SimpleNamespace(_xyz=torch.empty(0, 3)),
            initial_reference=dict(original_count=50, retained_count=50, removed_count=0, threshold=0.))
        subject.motion_steps = RIGID_MODULE.Module.motion_steps.fget(subject)
        subject.total_steps = RIGID_MODULE.Module.total_steps.fget(subject)
        events = []
        with tempfile.TemporaryDirectory() as directory:
            subject.output_dir = Path(directory)
            worker = SimpleNamespace(global_step=0, output_dir=Path(directory), module=subject,
                bootstrap_pending=frame == 1, datamodule=SimpleNamespace(cameras_extent=1.))
            subject.trainer = worker
            subject._write_tree_statistics = lambda step: RIGID_MODULE.Module._write_tree_statistics(subject, step)

            def write_reference(frame_index, state):
                events.append(('reference', worker.global_step))
                torch.testing.assert_close(gaussians._xyz, reference_xyz, rtol=0, atol=0)
                trainer_module.Trainer._write_reference(worker, frame_index, state)

            def begin_births():
                events.append(('birth', worker.global_step))
                subject.motion_committed = True
                gaussians._xyz, gaussians._rotation = (
                    value.detach().clone() for value in model.materialized_pose())

            worker._write_reference = write_reference
            subject._begin_births = begin_births
            subject.finalize_frame = lambda: events.append(('final', worker.global_step))
            target = model.forward(1)[0].detach().clone()
            motion_updates, heir_updates = 0, 0
            for step in range(1, subject.total_steps + 1):
                worker.global_step = step
                if not subject.motion_committed:
                    loss = (model.forward(step, training=True)[0] - target).square().mean()
                    terms = model.heir_loss_terms(step)
                    if terms:
                        loss = loss + terms['heir_loss']
                        heir_updates += 1
                    loss.backward()
                    motion_updates += 1
                package = dict(visibility_filter=torch.ones(50, dtype=torch.bool),
                               viewspace_points=torch.zeros(50, 3))
                RIGID_MODULE.Module.post_training_step(subject, package)
            tree_frame = frame in (1, 10)
            assert motion_updates == (200 if tree_frame else 100)
            assert heir_updates == (100 if tree_frame else 0)
            assert events == ([('reference', 200)] if frame == 1 else []) + [
                ('birth', subject.motion_steps), ('final', subject.total_steps)]
            assert [record['step'] for record in subject.tree_statistics] == ([100, 200] if tree_frame else [])
            if tree_frame:
                assert (Path(directory) / f'tree_stats/frame_{frame:03d}_step_200.csv').is_file()


def test_dynamic_codec_handles_identity_only_tree_growth_and_return_to_root():
    import tempfile

    model = residual_hierarchy_fixture()
    previous = model.frame_reference_state
    with tempfile.TemporaryDirectory() as directory:
        for frame in (1, 2, 3):
            if frame > 1:
                model.setup(model.materialized_xyz, model.materialized_rotation, 1., frame_index=frame)
                model.mark_observed(torch.ones(50, dtype=torch.bool))
            if frame == 2:
                install_residual_fixture_motion(model)
            # Exercise the codec's layout transitions independently of cadence.
            model._build_residual_regions(100)
            model._finalize_heir_tree(model.motion_steps)
            if frame == 1:
                previous = model.frame_reference_state
            assert len(model.parents) == (3 if frame == 2 else 1)
            xyz, rotation = model.materialized_pose()
            model.register_materialized_layout(xyz, rotation)
            path = Path(directory) / f'motion_{frame:03d}.bin'
            info = model.export_motion(path)
            decoded = CODEC.decode_motion(path.read_bytes(), previous)
            torch.testing.assert_close(decoded.xyz, xyz, rtol=0, atol=1e-6)
            torch.testing.assert_close(decoded.rotation, rotation, rtol=0, atol=1e-6)
            assert torch.equal(decoded.parents, model.parents)
            assert info['region_count'] == len(model.parents)
            previous = decoded


def test_residual_region_rejects_cancelled_mean_without_freezing_individual_motion():
    import numpy as np
    from unittest import mock

    model = residual_hierarchy_fixture()
    with torch.no_grad():
        model.base_xyz[8:16] = model.base_xyz[:8]
        model.base_rotation[8:16] = model.base_rotation[:8]
        model.gs_translation[:8, 0] = .01
        model.gs_translation[8:16, 0] = -.01
    before = model.forward(100)[0].detach().clone()
    labels = np.full(50, -2, dtype=np.int64)
    labels[:16] = 0
    with mock.patch.object(importlib.import_module('pipeline.ReCon-GS.model.adaptive_motion_tree'),
                           'hdbscan_motion_regions', return_value=([np.arange(16)], labels)):
        model.update_topology(100)
    assert model.parents.tolist() == [-1]
    assert model.cluster_labels[:16].tolist() == [-1] * 16
    assert model.last_metrics['tree_edit/near_identity_motion_groups'] == 1
    assert model.gs_translation[:16].ne(0).any(-1).all()
    torch.testing.assert_close(model.forward(100)[0], before, rtol=0, atol=1e-7)


def test_residual_region_rejects_shared_fit_that_saves_no_gs_support():
    import numpy as np
    from unittest import mock

    model = residual_hierarchy_fixture()
    with torch.no_grad():
        model.base_xyz[8:16] = model.base_xyz[:8]
        model.base_rotation[8:16] = model.base_rotation[:8]
        model.gs_translation[:8, 0] = .01
        model.gs_translation[8:16, 0] = -.008
    before = model.forward(100)[0].detach().clone()
    labels = np.full(50, -2, dtype=np.int64)
    labels[:16] = 0
    with mock.patch.object(importlib.import_module('pipeline.ReCon-GS.model.adaptive_motion_tree'),
                           'hdbscan_motion_regions', return_value=([np.arange(16)], labels)):
        model.update_topology(100)
    assert model.parents.tolist() == [-1]
    assert model.last_metrics['tree_edit/nonsparse_motion_groups'] == 1
    torch.testing.assert_close(model.forward(100)[0], before, rtol=0, atol=1e-7)


def test_useful_shared_motion_can_refine_before_exact_gs_support_shrinks():
    import numpy as np
    from unittest import mock

    model = residual_hierarchy_fixture()
    with torch.no_grad():
        model.base_xyz[8:16] = model.base_xyz[:8]
        model.base_rotation[8:16] = model.base_rotation[:8]
        model.gs_translation[:8, 0] = .06
        model.gs_translation[8:16, 0] = .04
    before = model.forward(100)[0].detach().clone()
    labels = np.full(50, -2, dtype=np.int64)
    labels[:16] = 0
    with mock.patch.object(importlib.import_module('pipeline.ReCon-GS.model.adaptive_motion_tree'),
                           'hdbscan_motion_regions', return_value=([np.arange(16)], labels)):
        model.update_topology(100)
    assert model.parents.tolist() == [-1, 0]
    assert model.last_metrics['tree_edit/deferred_sparse_motion_groups'] == 1
    assert bool(model.gs_translation[:16].ne(0).any(-1).all())
    torch.testing.assert_close(model.forward(100)[0], before, rtol=0, atol=1e-7)


def test_residual_identity_leaf_one_can_merge_into_root_without_pose_change():
    model = residual_hierarchy_fixture()
    install_residual_fixture_motion(model)
    model.update_topology(100)
    with torch.no_grad():
        model.region_translation.zero_()
        model.region_rotation.zero_()
    before = model.forward(200)[0].detach().clone()
    edits = model._collapse_identity_leaves()
    model._compact_motion_regions()
    assert any(row['node'] == 1 for row in edits)
    assert model.parents.tolist() == [-1]
    torch.testing.assert_close(model.forward(200)[0], before, rtol=0, atol=0)


def test_shared_translation_can_save_payload_when_gs_rotation_remains_active():
    import numpy as np
    from unittest import mock

    model = residual_hierarchy_fixture()
    with torch.no_grad():
        model.base_xyz[8:16] = model.base_xyz[:8]
        model.base_rotation[8:16] = model.base_rotation[:8]
        model.gs_translation[:16, 0] = .03
        model.gs_rotation[:8, 2] = .01
        model.gs_rotation[8:16, 2] = -.01
    before_xyz, before_q = (value.detach().clone() for value in model.forward(100))
    labels = np.full(50, -2, dtype=np.int64)
    labels[:16] = 0
    with mock.patch.object(importlib.import_module('pipeline.ReCon-GS.model.adaptive_motion_tree'),
                           'hdbscan_motion_regions', return_value=([np.arange(16)], labels)):
        model.update_topology(100)
    assert model.parents.tolist() == [-1, 0]
    assert model.owner[:16].tolist() == [1] * 16
    assert not bool(model.gs_translation[:16].any())
    assert bool(model.gs_rotation[:16].ne(0).any(-1).all())
    after_xyz, after_q = model.forward(100)
    torch.testing.assert_close(after_xyz, before_xyz, rtol=0, atol=1e-7)
    torch.testing.assert_close(after_q, before_q, rtol=0, atol=1e-7)


def test_cuda_residual_tree_clustering_and_codec_keep_cpu_reference():
    import tempfile

    if not torch.cuda.is_available():
        return
    model = residual_hierarchy_fixture(device='cuda')
    install_residual_fixture_motion(model)
    assert all(bool(torch.isfinite(value)) for value in model.loss_terms(1).values())
    model.update_topology(100)
    model.update_topology(200)
    xyz, rotation = model.materialized_pose()
    previous = model.frame_reference_state
    model.register_materialized_layout(xyz, rotation)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.bin'
        model.export_motion(path)
        decoded = CODEC.decode_motion(path.read_bytes(), previous)
        torch.testing.assert_close(decoded.xyz, xyz.cpu(), rtol=1e-5, atol=1e-6)
        assert decoded.xyz.device.type == 'cpu'
        assert decoded.parents.device.type == 'cpu'


if __name__ == '__main__':
    checks = [value for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for check in checks:
        check()
    print(f'{len(checks)} numerical checks passed')
