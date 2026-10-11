"""Focused lossless motion packet and actual-byte regressions."""

import hashlib
import importlib
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if 'pipeline.ReCon-GS' not in sys.modules:
    pipeline = ModuleType('pipeline')
    pipeline.__path__ = [str(ROOT / 'pipeline')]
    recon = ModuleType('pipeline.ReCon-GS')
    recon.__path__ = [str(ROOT / 'pipeline' / 'ReCon-GS')]
    sys.modules['pipeline'] = pipeline
    sys.modules['pipeline.ReCon-GS'] = recon

CODEC = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
RIGID = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')


def packet_fixture(count=256):
    xyz = torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 3).clone()
    rotation = torch.zeros(count, 4)
    rotation[:, 0] = 1
    owner = torch.ones(count, dtype=torch.long)
    parents = torch.tensor([-1, 0], dtype=torch.long)
    prior = CODEC.DecodedState(xyz, rotation, owner, parents, 0)
    region = torch.zeros(2, 3, dtype=torch.int16)
    gs_translation = torch.full((count, 3), 42, dtype=torch.int16)
    gs_rotation = torch.full((count, 3), 7, dtype=torch.int16)
    args = (prior, 1, 1., torch.arange(count), owner, parents,
            region, region, gs_translation, gs_rotation,
            torch.tensor([0]), torch.tensor([[999., 0., 0.]]),
            torch.tensor([[1., 0., 0., 0.]]), .01, .01)
    return prior, args


def test_compressed_and_legacy_packets_replay_identically():
    prior, args = packet_fixture()
    legacy = CODEC.encode_motion(*args)
    packed = CODEC.encode_motion(*args, compression='auto')
    assert legacy.startswith(CODEC.MAGIC)
    assert packed.startswith(CODEC.COMPRESSED_MAGIC)
    assert len(packed) < len(legacy)
    old_state = CODEC.decode_motion(legacy, prior)
    new_state = CODEC.decode_motion(packed, prior)
    for name in ('xyz', 'rotation', 'owner', 'parents', 'birth_sources'):
        torch.testing.assert_close(getattr(new_state, name), getattr(old_state, name),
                                   rtol=0, atol=0)
    assert new_state.digest() == old_state.digest()
    info = CODEC.motion_packet_info(packed)
    assert info == dict(format='RRC1-zlib', stored_bytes=len(packed),
                        raw_stream_bytes=len(legacy),
                        compression_savings_bytes=len(legacy) - len(packed))


def test_auto_compression_keeps_incompressible_legacy_packet():
    seed = b'rigid-tree-incompressible-fixture'
    body = b''.join(hashlib.sha256(seed + i.to_bytes(4, 'little')).digest()
                    for i in range(16))
    raw = CODEC.MAGIC + body
    raw += hashlib.sha256(raw).digest()[:CODEC.CHECKSUM_BYTES]
    assert CODEC.compress_motion_packet(raw, mode='auto') == raw


def test_compressed_packet_rejects_mutation_truncation_trailing_and_bomb():
    prior, args = packet_fixture()
    packet = CODEC.encode_motion(*args, compression='zlib')
    damaged = bytearray(packet)
    damaged[len(damaged) // 2] ^= 1
    for candidate in (bytes(damaged), packet[:-1], packet + b'X',
                      CODEC.COMPRESSED_MAGIC + CODEC._uvar(
                          CODEC.MAX_RAW_PACKET_BYTES + 1) + packet[5:]):
        try:
            CODEC.decode_motion(candidate, prior)
        except ValueError:
            pass
        else:
            raise AssertionError('Damaged compressed stream was accepted')


def test_export_reports_actual_stored_and_uncompressed_bytes():
    prior, args = packet_fixture()
    config = RIGID.RigidTreeConfig(region_count=2,
                                   motion_xyz_step_extent=.01,
                                   motion_rotation_step_rad=.01)
    model = RIGID.RigidResidualTree(config)
    model.setup(prior.xyz, prior.rotation, 1., frame_index=1)
    with torch.no_grad():
        model.gs_translation[:] = .42
        model.gs_rotation[:] = .07
    pose = model.materialized_pose()
    birth_xyz, birth_rotation = args[11], args[12]
    model.register_materialized_layout(
        torch.cat((pose[0], birth_xyz)), torch.cat((pose[1], birth_rotation)),
        birth_source_rows=torch.tensor([0]))
    uncompressed = int(model.loss_terms(100)['rate_bytes_hard_uncompressed'])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'motion.bin'
        info = model.export_motion(path)
        assert path.stat().st_size == info['bytes'] == info['payload_bytes']
        assert info['stored_bytes'] == info['bytes']
        assert info['raw_stream_bytes'] == uncompressed
        assert info['compression_savings_bytes'] == uncompressed - info['bytes']
        assert CODEC.decode_motion(path.read_bytes(), model.frame_reference_state).digest() == (
            model.codec_previous_state.digest())


def dual_step_model(**overrides):
    xyz = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                        [2., 0., 0.], [3., 0., 0.]])
    rotation = torch.zeros(4, 4)
    rotation[:, 0] = 1
    options = dict(region_count=3, motion_steps=4, birth_steps=2,
                   shared_steps=1, tree_edit_step=3,
                   motion_xyz_step_extent=.01, motion_rotation_step_rad=.01,
                   gs_xyz_step_extent=.02, gs_rotation_step_rad=.03,
                   gs_translation_lr=.01, gs_rotation_lr=.01,
                   rate_loss_weight=.0001)
    options.update(overrides)
    config = RIGID.RigidTreeConfig(**options)
    model = RIGID.RigidResidualTree(config)
    model.setup(xyz, rotation, 1., frame_index=1)
    return model


def test_dual_step_stream_replays_distinct_region_and_gs_grids():
    model = dual_step_model()
    with torch.no_grad():
        model.region_translation[1, 0] = .025
        model.gs_translation[0, 1] = .041
        model.gs_rotation[0, 2] = .071
    shared_xyz, shared_q = model.forward(1)
    with torch.no_grad():
        saved_t, saved_r = model.gs_translation.clone(), model.gs_rotation.clone()
        model.gs_translation.zero_()
        model.gs_rotation.zero_()
        region_xyz, region_q = model.forward(2)
        model.gs_translation.copy_(saved_t)
        model.gs_rotation.copy_(saved_r)
    torch.testing.assert_close(shared_xyz, region_xyz, rtol=0, atol=0)
    torch.testing.assert_close(shared_q, region_q, rtol=0, atol=0)
    xyz, rotation = model.materialized_pose()
    model.register_materialized_layout(xyz, rotation)
    with tempfile.TemporaryDirectory() as directory:
        info = model.export_motion(Path(directory) / 'motion.bin')
        assert info['format'] in ('RRT2-raw', 'RRC2-zlib')
        torch.testing.assert_close(
            torch.tensor([info['region_translation_step_world'],
                          info['gs_translation_step_world'],
                          info['gs_rotation_step_rad']]),
            torch.tensor([.01, .02, .03]), rtol=1e-6, atol=0)
        assert int(model.loss_terms(4)['rate_bytes_hard_uncompressed']) == (
            info['raw_stream_bytes'])
        torch.testing.assert_close(model.codec_previous_state.xyz, xyz.detach(),
                                   rtol=0, atol=1e-6)
        torch.testing.assert_close(model.codec_previous_state.rotation,
                                   rotation.detach(), rtol=0, atol=1e-6)


def test_shared_only_ignores_gs_and_optimizer_does_not_update_it():
    model = dual_step_model()
    with torch.no_grad():
        model.gs_translation[0, 0] = .04
    before = model.gs_translation.detach().clone()
    xyz, _ = model.forward(1, training=True)
    torch.testing.assert_close(xyz, model.base_xyz, rtol=0, atol=0)
    model.gs_translation.grad = torch.ones_like(model.gs_translation)
    model.gs_rotation.grad = torch.ones_like(model.gs_rotation)
    model.optimizer_step()
    torch.testing.assert_close(model.gs_translation, before, rtol=0, atol=0)
    assert model.gs_translation not in model.optimizer.state


def test_proximal_gs_keeps_large_signal_and_zeroes_small_signal():
    model = dual_step_model(gs_xyz_step_extent=.001, gs_rotation_step_rad=.001)
    model.last_step = 2
    translation_grad = torch.zeros_like(model.gs_translation)
    rotation_grad = torch.zeros_like(model.gs_rotation)
    translation_grad[0, 0] = rotation_grad[0, 0] = 1e-8
    translation_grad[1, 0] = rotation_grad[1, 0] = 1e-2
    model.gs_translation.grad = translation_grad
    model.gs_rotation.grad = rotation_grad
    model.optimizer_step()
    assert bool((model.gs_translation[0] == 0).all())
    assert bool((model.gs_rotation[0] == 0).all())
    assert bool(model.gs_translation[1].ne(0).any())
    assert bool(model.gs_rotation[1].ne(0).any())
    assert model.last_metrics['motion/prox_zero_translation_blocks'] >= 1
    diagnostics = model.loss_terms(2, diagnostics=True)
    assert int(diagnostics['gs_translation_active_blocks']) == 1
    assert int(diagnostics['gs_rotation_active_blocks']) == 1


def test_proximal_gs_handles_one_optimizer_block_without_moments():
    model = dual_step_model(gs_xyz_step_extent=.001, gs_rotation_step_rad=.001)
    model.last_step = 2
    gradient = torch.zeros_like(model.gs_translation)
    gradient[0, 0] = 1e-8
    gradient[1, 0] = 1e-2
    model.gs_translation.grad = gradient
    model.optimizer_step()
    assert model.gs_rotation not in model.optimizer.state
    assert bool((model.gs_translation[0] == 0).all())
    assert bool(model.gs_translation[1].ne(0).any())


def test_rate_gradient_charges_regions_while_gs_support_uses_proximal_step():
    model = dual_step_model()
    with torch.no_grad():
        model.region_translation[1, 0] = .015
        model.gs_translation[0, 0] = .03
    model.loss_terms(2)['rate_loss'].backward()
    assert model.region_translation.grad[1].abs().sum() > 0
    assert model.gs_translation.grad is None
    assert model.loss_terms(2)['rate_bytes_surrogate'] > 0


def test_tree_edit_reports_exact_uncompressed_cost_and_bounded_pose_error():
    model = dual_step_model(tree_edit_interval=10)
    model.frame_index = 10
    row = int((model.owner == 1).nonzero(as_tuple=True)[0][0])
    with torch.no_grad():
        model.region_translation[2, 0] = .2
        model.gs_translation[row, 0] = .2
    target_xyz, target_q = (value.detach().clone() for value in model.forward(3))
    before = int(model.loss_terms(3)['rate_bytes_hard_uncompressed'])
    model.update_topology(3)
    metrics = model.last_metrics
    after = int(model.loss_terms(3)['rate_bytes_hard_uncompressed'])
    assert metrics['tree_edit/uncompressed_bytes_before'] == before
    assert metrics['tree_edit/uncompressed_bytes_after'] == after
    assert metrics['tree_edit/uncompressed_bytes_delta'] == after - before
    assert metrics['tree_edit/owner_edits'] >= 1
    xyz, rotation = model.forward(3)
    assert metrics['tree_edit/pose_max_xyz_error_world'] <= (
        3 ** .5 * float(model.gs_steps[0]) / 2 + 1e-6)
    torch.testing.assert_close(xyz, target_xyz,
                               rtol=0, atol=3 ** .5 * float(model.gs_steps[0]) / 2 + 1e-6)
    dot = torch.nn.functional.normalize(rotation, dim=-1).mul(
        torch.nn.functional.normalize(target_q, dim=-1)).sum(-1).abs()
    assert bool((dot >= torch.cos(model.gs_steps[3] * (3 ** .5 / 4))).all())


def test_tree_metrics_checkpoint_and_next_frame_boundary():
    model = dual_step_model()
    expected = {'tree_edit/owner_edits': 3,
                'tree_edit/pose_max_xyz_error_world': .002}
    model.last_metrics.update(expected)
    packet = model.capture()
    restored = RIGID.RigidResidualTree(model.config)
    restored.restore(packet)
    assert restored.last_metrics == expected
    packet['last_metrics']['tree_edit/owner_edits'] = 9
    assert restored.last_metrics == expected

    xyz, rotation = model.materialized_pose()
    model.register_materialized_layout(xyz, rotation)
    with tempfile.TemporaryDirectory() as directory:
        model.export_motion(Path(directory) / 'motion.bin')
    prior = model.codec_previous_state
    model.setup(prior.xyz, prior.rotation, 1., frame_index=2)
    assert model.last_metrics == {}


if __name__ == '__main__':
    checks = [value for name, value in list(globals().items())
              if name.startswith('test_') and callable(value)]
    for check in checks:
        check()
    print(f'Passed {len(checks)} rigid tree compression checks.')
