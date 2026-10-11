"""Audit real frame5 motion inheritance and prepare one decay .6 vs 0 pair."""
import copy
import hashlib
import json
import runpy
from pathlib import Path

import torch
import yaml

from scripts.archive.prepare_keep_local_probe_replay import assert_same_state


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    root = Path('output/fs_core_keep_candidate_ownloss_seed0_300f_20261003/20261003T150101_38c8c8')
    source = root / 'ckpt-5-200.pth'
    original = torch.load(source, map_location='cpu', weights_only=False)
    saved = original['quantized_residual_motion']
    assert original['current_frame'] == 5 and original['global_step'] == 200
    assert saved['config']['motion_gradient_mode'] == 'keep_candidate_loss'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    scope = runpy.run_path('tests/test_nested_prototype.py', run_name='inheritance_preflight')
    GR, Q = scope['GR'], scope['Q']
    xyz, rotation, opacity = [original['gaussians'][index].detach() for index in (1, 5, 6)]
    reference = GR.GatedResidualMotion(Q.QuantizedResidualConfig(**saved['config']))
    reference.restore(copy.deepcopy(saved))
    count = original['quantized_frame_state']['previous_new_count']
    deleted, protection = reference.compensation_deletions(torch.sigmoid(opacity), count)
    rows = Q.survivor_rows(len(xyz), deleted)
    symbols = reference._dense_symbols(integer=True)
    states = reference.gate_state_arrays()
    checks = []
    for frame in (6, 10):
        models = []
        for decay in (.6, 0.):
            packet = copy.deepcopy(saved)
            packet['config']['motion_init_decay'] = decay
            motion = GR.GatedResidualMotion(Q.QuantizedResidualConfig(**packet['config']))
            motion.restore(packet)
            motion.setup(xyz[rows], rotation[rows], saved['extent'], previous_rows=rows,
                         previous_new_count=count, frame_index=frame)
            torch.testing.assert_close(motion.base_xyz, xyz[rows], rtol=0, atol=0)
            torch.testing.assert_close(motion.base_rotation, rotation[rows], rtol=0, atol=0)
            with torch.no_grad():
                first, second = motion(1), motion(1)
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            if decay == 0.:
                assert all(torch.count_nonzero(value) == 0 for value in
                           [*motion.region_translation, *motion.region_rotation,
                            motion.gs_translation, motion.gs_rotation])
                torch.testing.assert_close(first[0], xyz[rows], rtol=0, atol=0)
                torch.testing.assert_close(torch.nn.functional.normalize(first[1], dim=-1),
                                           torch.nn.functional.normalize(rotation[rows], dim=-1),
                                           rtol=0, atol=2e-7)
            if frame == 6:
                for level in range(3):
                    active = states['visited'][level] & (states['states'][level] != GR.KEEP)
                    expected = symbols[f'region_{level}'].float() * reference.motion_steps_vector * decay
                    expected = torch.where(active[:, None], expected, torch.zeros_like(expected))
                    torch.testing.assert_close(motion.region_translation[level], expected[:, :3], rtol=0, atol=0)
                    torch.testing.assert_close(motion.region_rotation[level], expected[:, 3:], rtol=0, atol=0)
            models.append(motion)
        for a, b in zip(models[0].gate_logits, models[1].gate_logits):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(models[0].index, models[1].index, rtol=0, atol=0)
        checks.append(dict(frame=frame, tree_action=models[0].tree_setup['action'],
            same_gate_logits=True, same_membership=True, same_materialized_base=True,
            zero_decay_identity_pose=True, repeated_forward_does_not_accumulate=True,
            reuse_active_region_prior_exact=frame == 6))
    records, packets = [], []
    for arm, decay in (('decay06', .6), ('decay00', 0.)):
        name = f'fs_core_inherit_{arm}_from5_10f_20261003'
        directory = Path('output') / name / 'replay_input'
        config_path = Path('config') / f'{name}.yaml'
        if directory.exists() or config_path.exists():
            raise FileExistsError(name)
        directory.mkdir(parents=True)
        packet = copy.deepcopy(original)
        packet['quantized_residual_motion']['config']['motion_init_decay'] = decay
        normalized = copy.deepcopy(packet)
        normalized['quantized_residual_motion']['config']['motion_init_decay'] = .6
        tensors = assert_same_state(original, normalized)
        assert tensors == 81
        path = directory / 'start.pth'
        torch.save(packet, path)
        cfg = yaml.safe_load((root / 'config.yaml').read_text())
        cfg.update(experiment_name=name, ckpt_path=str(path))
        cfg['data']['extra_dataset_kwargs']['num_frames'] = 16
        cfg['trainer']['saving_ckpt_frames'] = [15]
        cfg['module']['saving_gs_frames'] = [15]
        cfg['module']['quantized_residual']['motion_init_decay'] = decay
        config_path.write_text('# 同失败own-loss frame5完整状态回放6–15；唯一机制变量motion_init_decay .6 vs0。\n'
            '# 只关闭上一帧实际应用运动的预测；gate logits/映射继续同样继承。\n'
            '# candidate loss interval10/weight1、GS/storage .001、200步、16/4视角全部冻结。\n'
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        record = dict(arm=arm, decay=decay, source_checkpoint=str(source), source_sha256=source_hash,
            input_checkpoint=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            exact_tensors=tensors, only_input_field_changed='quantized_residual_motion.config.motion_init_decay',
            frames=[6, 15], gate_inheritance_unchanged=True,
            limitations='Fresh seed0 replay, source has no full RNG state; compare these two arms, not the original continuous trajectory.')
        (directory / 'lineage.json').write_text(json.dumps(record, indent=2))
        records.append(record)
        packets.append(normalized)
        print(config_path, flush=True)
    assert assert_same_state(*packets) == 81
    audit = dict(source_checkpoint=str(source), source_sha256=source_hash, checks=checks,
        deletion_count=len(deleted), deletion_protection=protection, inputs=records,
        interpretation='Motion decay rescales applied quantized region translation/rotation-vector and open local residual; gate logits inherit separately. Materialized prior pose is a fresh base; repeated forward has no in-place accumulation.',
        limits='CPU state/pose invariants do not prove absence of all renderer/optimization bugs. Fixed actual-render phase probes in both replays test image-loss consequences.')
    Path('/tmp/recongs_motion_inheritance_preflight_20261003.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(dict(checks=checks, tensors=81, source_sha256=source_hash)), flush=True)


if __name__ == '__main__':
    main()
