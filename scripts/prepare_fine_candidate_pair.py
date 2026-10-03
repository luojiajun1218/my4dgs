"""Prepare a 20-camera replay pair with identical tensors/optimizer input state."""
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml


def main():
    source_run = Path('output/fs_core_keep_candidate_zero_inherit_refaligned_seed0_300f_20261003/20261003T174228_5fe082')
    source = source_run / 'ckpt-5-200.pth'
    packet = torch.load(source, map_location='cpu', weights_only=False)
    assert packet['current_frame'] == 5 and packet['global_step'] == 200
    config = yaml.safe_load((source_run / 'config.yaml').read_text())
    assert config['data']['training_camera_names'] is None
    assert config['data']['validation_camera_names'] is None
    entries = []
    for name, enabled in (('control', False), ('fitted', True)):
        experiment = f'fs_core_fine_candidate_{name}_refaligned_from5_10f_20261003'
        checkpoint = Path('output') / experiment / 'replay_input/start.pth'
        checkpoint.parent.mkdir(parents=True, exist_ok=False)
        saved = copy.deepcopy(packet)
        saved['quantized_residual_motion']['config']['fit_fine_split_candidates'] = enabled
        torch.save(saved, checkpoint)
        current = copy.deepcopy(config)
        current['experiment_name'] = experiment
        current['ckpt_path'] = str(checkpoint)
        current['data']['extra_dataset_kwargs']['num_frames'] = 16
        current['module']['quantized_residual']['fit_fine_split_candidates'] = enabled
        current['trainer']['saving_ckpt_frames'] = [15]
        current['trainer']['saving_ckpt_every_n_frames'] = 1000
        current['module']['saving_gs_frames'] = [15]
        current['module']['saving_gs_every_n_frames'] = 1000
        path = Path('config') / f'{experiment}.yaml'
        path.write_text('# 同正式frame5完整状态回放6–15；20train/test00，无独立val。\n'
            '# 唯一机制变量fit_fine_split_candidates；200基础步+最多14候选图。\n'
            '# false: KEEP候选；true: 同图额外拟合fine SHARE的局部SPLIT候选，不强制开gate。\n'
            + yaml.safe_dump(current, allow_unicode=True, sort_keys=False))
        entries.append(dict(name=name, enabled=enabled, config=str(path),
                            checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    loaded = [torch.load(row['checkpoint'], map_location='cpu', weights_only=False) for row in entries]
    def compare(a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            return 1
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            return sum(compare(a[key], b[key]) for key in a)
        if isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            return sum(compare(x, y) for x, y in zip(a, b))
        assert a == b
        return 0
    for value in loaded:
        value['quantized_residual_motion']['config'].pop('fit_fine_split_candidates')
    tensors = compare(*loaded)
    summary = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   entries=entries, identical_input_tensors=tensors,
                   tensor_rtol=0, tensor_atol=0,
                   only_checkpoint_metadata_difference='fit_fine_split_candidates',
                   frames=[6, 15], train_cameras='cam01–20', test_camera='cam00',
                   selection_test=False, independent_validation=False)
    Path('/tmp/recongs_fine_candidate_pair_inputs_20261003.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
