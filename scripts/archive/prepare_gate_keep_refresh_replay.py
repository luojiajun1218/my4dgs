"""Create two documented local replay inputs from one completed frame state.

Only the method configuration flag differs. Historical checkpoints are read-only.
Each replay starts with the same model/optimizer tensors and seed; this avoids
comparing a freshly seeded replay against a continuous run's random birth stream.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def state_digest(packet):
    digest = hashlib.sha256()
    motion = packet['quantized_residual_motion']
    for name, tensor in sorted(motion['state'].items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_run', type=Path)
    parser.add_argument('--start', type=int, default=149)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--suffix', default='20261003')
    parser.add_argument('--arm', choices=('both', 'control', 'resetkeep', 'keepprobe'),
                        default='both')
    args = parser.parse_args()
    checkpoint = args.source_run / f'ckpt-{args.start}-200.pth'
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if original['current_frame'] != args.start or original['global_step'] != 200:
        raise ValueError('Replay needs the specified completed frame checkpoint')
    if args.frames < 1:
        raise ValueError('Replay frame count must be positive')
    motion_digest = state_digest(original)
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = yaml.safe_load((args.source_run / 'config.yaml').read_text(encoding='utf-8'))
    last = args.start + args.frames
    arms = ((False, 'control'), (True, 'resetkeep'))
    if args.arm != 'both':
        arms = ((args.arm == 'resetkeep', args.arm),)
    for flag, arm in arms:
        name = f'fs_core_gate_replay_{arm}_from{args.start}_{args.frames}f_{args.suffix}'
        directory = Path('output') / name / 'replay_input'
        directory.mkdir(parents=True, exist_ok=False)
        # Explicitly amend just this configuration field; optimizer/model state
        # and previous materialized GS remain untouched.
        original['quantized_residual_motion']['config']['reset_keep_gate_priors'] = flag
        if arm == 'keepprobe':
            original['quantized_residual_motion']['config']['motion_gradient_mode'] = 'keep_probe'
        target = directory / 'start.pth'
        torch.save(original, target)
        assert state_digest(original) == motion_digest
        cfg = json.loads(json.dumps(config))
        cfg.update(experiment_name=name, ckpt_path=str(target))
        cfg['data']['extra_dataset_kwargs']['num_frames'] = last + 1
        cfg['module']['quantized_residual']['reset_keep_gate_priors'] = flag
        cfg['module']['quantized_diagnostics']['prediction_quality_enabled'] = True
        if arm == 'keepprobe':
            cfg['module']['quantized_residual']['motion_gradient_mode'] = 'keep_probe'
        cfg['trainer']['saving_ckpt_frames'] = [last]
        cfg['module']['saving_gs_frames'] = [last]
        path = Path('config') / f'{name}.yaml'
        if path.exists():
            raise FileExistsError(path)
        path.write_text('# 同一完成帧checkpoint的局部反事实；只改变指定的gate/候选梯度开关。\n'
                        '# 两臂均重新使用seed0；相机顺序逐项核验，cam00仅终态被动评测。\n'
                        + ('# keep_probe保留gate先验和硬forward，只拟合当前KEEP终端的SHARE候选梯度。\n'
                           if arm == 'keepprobe' else '')
                        + yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        (directory / 'lineage.json').write_text(json.dumps(dict(
            source_checkpoint=str(checkpoint), source_sha256=source_hash,
            motion_state_sha256=motion_digest, start_frame=args.start,
            replay_frames=[args.start + 1, last], reset_keep_gate_priors=flag,
            motion_gradient_mode=original['quantized_residual_motion']['config']['motion_gradient_mode'],
            change=('Only motion_gradient_mode=keep_probe; gate priors retained; all tensors unchanged.'
                    if arm == 'keepprobe' else
                    'Only config.reset_keep_gate_priors; all checkpoint tensors unchanged.')), indent=2))
        print(path)


if __name__ == '__main__':
    main()
