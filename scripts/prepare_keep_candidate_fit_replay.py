"""Prepare exact-state keep_probe versus sparse candidate-own image fitting."""
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from prepare_keep_local_probe_replay import assert_same_state


def main():
    source_run = Path('output/fs_core_gsactivity_weight001_seed0_300f_20261003/20261003T134910_37e8cf')
    source = source_run / 'ckpt-5-200.pth'
    original = torch.load(source, map_location='cpu', weights_only=False)
    assert original['current_frame'] == 5 and original['global_step'] == 200
    # Backward-compatible configuration defaults; values/tensors are unchanged.
    original['quantized_residual_motion']['config'].setdefault('keep_candidate_fit_interval', 10)
    original['quantized_residual_motion']['config'].setdefault('keep_candidate_fit_weight', 1.)
    records, packets = [], []
    for arm, mode in (('control', 'keep_probe'), ('ownloss', 'keep_candidate_loss')):
        name = f'fs_core_keep_candidate_{arm}_from5_10f_20261003'
        directory = Path('output') / name / 'replay_input'
        config_path = Path('config') / f'{name}.yaml'
        if directory.exists() or config_path.exists():
            raise FileExistsError(name)
        directory.mkdir(parents=True)
        packet = copy.deepcopy(original)
        packet['quantized_residual_motion']['config']['motion_gradient_mode'] = mode
        count = assert_same_state(original, packet)
        input_path = directory / 'start.pth'
        torch.save(packet, input_path)
        cfg = yaml.safe_load((source_run / 'config.yaml').read_text())
        cfg.update(experiment_name=name, ckpt_path=str(input_path))
        cfg['data']['extra_dataset_kwargs']['num_frames'] = 16
        cfg['trainer']['saving_ckpt_frames'] = [15]
        cfg['module']['saving_gs_frames'] = [15]
        cfg['module']['quantized_residual'].update(motion_gradient_mode=mode,
            keep_candidate_fit_interval=10, keep_candidate_fit_weight=1.)
        config_path.write_text('# 同frame5状态，回放6–15；仅关闭候选的学习机制不同。\n'
            '# interval10：1,11,…131步最多14次额外候选渲染，200基础渲染之外+7%。\n'
            '# weight1：与硬图像相同photometric尺度；候选只更新可达KEEP自身运动。\n'
            '# 原gate代理/GS成本.001/storage.001/先验/布局/补点不变。\n'
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        record = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            input_checkpoint=str(input_path), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
            start_frame=5, frames=[6, 15], mode=mode, exact_tensors=count, all_other_fields_identical=True,
            new_compatible_defaults=dict(keep_candidate_fit_interval=10, keep_candidate_fit_weight=1.),
            limitations='Checkpoint has no complete RNG state; both new arms freshly seeded. '
                        'Own-loss adds at most14 counted candidate render/backward graphs per200-step frame.')
        (directory / 'lineage.json').write_text(json.dumps(record, indent=2))
        records.append(record)
        packets.append(packet)
        print(config_path)
    assert assert_same_state(*packets) == 81
    Path('/tmp/recongs_keep_candidate_pair_inputs_20261003.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
