"""Prepare a two-arm replay changing only the KEEP candidate Jacobian mode."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def assert_same_state(a, b, path='', count=None):
    count = [0] if count is None else count
    if path == 'quantized_residual_motion.config.motion_gradient_mode':
        return count[0]
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            raise AssertionError(f'Tensor type differs: {path}')
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        count[0] += 1
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_same_state(a[key], b[key], f'{path}.{key}' if path else key, count)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b)):
            assert_same_state(left, right, f'{path}[{index}]', count)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    else:
        assert a == b, path
    return count[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_run', type=Path)
    parser.add_argument('--start', type=int, default=5)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--suffix', default='20261003')
    args = parser.parse_args()
    source = args.source_run / f'ckpt-{args.start}-200.pth'
    original = torch.load(source, map_location='cpu', weights_only=False)
    if original['current_frame'] != args.start or original['global_step'] != 200:
        raise ValueError('Replay requires the completed source frame state')
    if args.frames < 1:
        raise ValueError('Replay frame count must be positive')
    if original['quantized_residual_motion']['config']['motion_gradient_mode'] != 'keep_probe':
        raise ValueError('Control input must use keep_probe')
    base_config = yaml.safe_load((args.source_run / 'config.yaml').read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    records = []
    packets = []
    for arm, mode in (('control', 'keep_probe'), ('isolated', 'keep_local_probe')):
        name = f'fs_core_keep_local_{arm}_from{args.start}_{args.frames}f_{args.suffix}'
        directory = Path('output') / name / 'replay_input'
        config_path = Path('config') / f'{name}.yaml'
        if directory.exists() or config_path.exists():
            raise FileExistsError(name)
        directory.mkdir(parents=True)
        packet = copy.deepcopy(original)
        packet['quantized_residual_motion']['config']['motion_gradient_mode'] = mode
        tensors = assert_same_state(original, packet)
        path = directory / 'start.pth'
        torch.save(packet, path)
        cfg = copy.deepcopy(base_config)
        cfg.update(experiment_name=name, ckpt_path=str(path))
        cfg['data']['extra_dataset_kwargs']['num_frames'] = args.start + args.frames + 1
        cfg['module']['quantized_residual']['motion_gradient_mode'] = mode
        cfg['trainer']['saving_ckpt_frames'] = [args.start + args.frames]
        cfg['module']['saving_gs_frames'] = [args.start + args.frames]
        config_path.write_text('# 同frame5完整状态重放6–15：只比较KEEP候选祖先梯度隔离。\n'
            '# 两臂GS成本.001、硬forward/gate代理/先验/200步/全部初始化相同。\n'
            '# 相同输入张量与fresh seed0；不使用动静mask或手工ROI。\n'
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        record = dict(arm=arm, source_checkpoint=str(source), source_sha256=source_hash,
                      input_checkpoint=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      start_frame=args.start, frames=[args.start + 1, args.start + args.frames],
                      mode=mode, exact_tensors=tensors, all_other_fields_identical=True,
                      limitations='Source lacks complete RNG state. Both arms are freshly seeded; '
                                  'compare these replays to each other, not to the original continuous random stream.')
        (directory / 'lineage.json').write_text(json.dumps(record, indent=2))
        records.append(record)
        packets.append(packet)
        print(config_path)
    assert_same_state(*packets)
    Path('/tmp/recongs_keep_local_pair_inputs_20261003.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
