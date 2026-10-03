"""Analyze the predeclared training-side criteria of the 20-camera replay pair."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml


def analyze(control, fitted, output):
    cameras = {f'cam{i:02d}.png' for i in range(1, 21)}
    rows = []
    for run, enabled in ((control, False), (fitted, True)):
        cfg = yaml.safe_load((run / 'config.yaml').read_text())
        assert cfg['data']['training_camera_names'] is None
        assert cfg['data']['validation_camera_names'] is None
        assert cfg['module']['quantized_residual']['fit_fine_split_candidates'] is enabled
    for frame in range(6, 16):
        row = dict(frame=frame)
        traces = []
        for name, run in (('control', control), ('fitted', fitted)):
            packet = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())
            assert 'wall' in packet['timing']
            assert set(packet['quantized_evaluation']) == {'test'}
            diagnostic = json.loads((run / 'frame_summary' / f'control_{frame:03d}.json').read_text())
            trace = json.loads((run / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views']
            assert len(trace) == 200 and set(trace) == cameras
            traces.append(trace)
            probe = diagnostic['quality']['after_retention']['train_probe']['mean']
            stream = packet['stream']
            actual = stream['actual_motion']
            assert actual['keep_position_changed_count'] == actual['keep_rotation_changed_count'] == 0
            assert stream['candidate_fit']['extra_renders'] <= 14
            values = dict(train_probe_image_objective=probe['image_objective'],
                train_probe_psnr=probe['psnr'], test_psnr=packet['quantized_evaluation']['test']['psnr'],
                hard_bytes=stream['hard_storage_bytes'], training_s=packet['timing']['training'],
                inclusive_training_s=packet['timing']['training_loop_including_state_update'],
                wall_s=packet['timing']['wall'], local_fraction=stream['activities']['gs_local_active_fraction'],
                actual_changed_fraction=actual['any_pose_changed_count'] / actual['old_surviving_count'],
                extra_renders=stream['candidate_fit']['extra_renders'],
                local_gradient_samples=sum(
                    sample['gradients']['gs_translation']['nonzero_rows'] > 0 or
                    sample['gradients']['gs_rotation']['nonzero_rows'] > 0
                    for sample in diagnostic['gradients']))
            row.update({f'{name}_{key}': value for key, value in values.items()})
        assert traces[0] == traces[1], 'Paired actual 200-step camera sequences differ'
        for key in values:
            row[f'delta_{key}'] = row[f'fitted_{key}'] - row[f'control_{key}']
        rows.append(row)
    means = {key: statistics.fmean(row[key] for row in rows) for key in rows[0] if key != 'frame'}
    result = dict(control=str(control), fitted=str(fitted), frames=[6, 15], means=means,
        actual_camera_traces_identical=True, KEEP_violations=0, max_aux_graphs=14,
        train_probe_image_objective_improved=means['delta_train_probe_image_objective'] < 0,
        local_gradient_samples_present=means['fitted_local_gradient_samples'] > 0,
        test_used_for_selection=False,
        limitations=['One short replay pair, not a long-sequence result.',
            'Five fixed training probes are not final evaluation of all20 training cameras.',
            'CUDA optimization can be nondeterministic despite exact input tensors/camera traces.',
            'Local gradients include combined main/aux/storage/GS-activity losses.',
            'Candidate pose difference needs separate checkpoint numerical inspection.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fine_candidate_pair_summary.json').write_text(json.dumps(result, indent=2))
    with (output / 'fine_candidate_pair_frames.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('fitted', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.control, args.fitted, args.output)
