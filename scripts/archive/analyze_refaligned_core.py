"""Passive test-cam00 comparison; never selects or changes training parameters."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml


def average(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def analyze(run, reference, output):
    config = yaml.safe_load((run / 'config.yaml').read_text())
    assert config['data']['training_camera_names'] is None
    assert config['data']['validation_camera_names'] is None
    assert config['data']['split_train_test'] and config['data']['eval_test']
    assert config['data']['resolution'] == -1
    assert config['trainer']['num_incr_steps'] == 200
    reference_rows = {item['frame']: item for line in reference.read_text().splitlines()
                      if (item := json.loads(line))['frame'] > 0}
    rows, views, phases = [], [], []
    cameras = {f'cam{index:02d}.png' for index in range(1, 21)}
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        try:
            packet = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        frame = packet['frame']
        if frame == 0 or 'wall' not in packet.get('timing', {}):
            continue
        trace = json.loads((run / 'frame_summary' /
                            f'training_views_{frame:03d}.json').read_text())['views']
        assert len(trace) == 200 and set(trace) <= cameras
        assert cameras <= set(trace), 'Every training camera must occur in 200 steps'
        evaluation = json.loads((run / 'frame_summary' /
                                 f'evaluation_{frame:03d}_step200.json').read_text())
        test_views = [view for view in evaluation['views'] if view['split'] == 'test']
        assert len(test_views) == 1 and test_views[0]['camera'] == 'cam00.png'
        assert 'validation' not in packet['quantized_evaluation']
        ref = reference_rows[frame]
        assert ref['steps'] == 200 and ref['complete_training_steps']
        ref_test = ref['evaluation']['test']
        assert [view['view'] for view in ref_test['views']] == ['cam00.png']
        stream = packet['stream']
        actual = stream['actual_motion']
        assert actual['keep_position_changed_count'] == 0
        assert actual['keep_rotation_changed_count'] == 0
        test = packet['quantized_evaluation']['test']
        row = dict(frame=frame, test_psnr=test['psnr'], test_ssim=test['ssim'],
                   test_l1=test['l1'], reference_test_psnr=ref_test['psnr'],
                   reference_test_ssim=ref_test['ssim'],
                   reference_test_l1=ref_test['l1'],
                   test_delta_psnr=test['psnr'] - ref_test['psnr'],
                   actual_changed_fraction=actual['any_pose_changed_count'] /
                   actual['old_surviving_count'],
                   keep_fraction=stream['activities']['gs']['fractions']['KEEP'],
                   local_fraction=stream['activities']['gs_local_active_fraction'],
                   hard_bytes=stream['hard_storage_bytes'],
                   expected_bytes=stream['expected_storage_bytes'],
                   extra_candidate_renders=stream['candidate_fit']['extra_renders'],
                   born=stream['new_count'], deleted=stream['deleted_count'],
                   old_surviving_gs=actual['old_surviving_count'],
                   total_gs=actual['old_surviving_count'] + stream['new_count'],
                   reference_born=ref['birth_ply']['rows'],
                   reference_deleted=ref['transition']['deleted_rows'],
                   reference_total_gs=ref['transition']['survivor_rows'] + ref['birth_ply']['rows'],
                   training_s=packet['timing']['training'],
                   training_loop_s=packet['timing']['training_loop_including_state_update'],
                   wall_s=packet['timing']['wall'],
                   reference_training_s=ref['timing']['training_steps_wall_synchronized_s'],
                   reference_wall_s=ref['timing']['frame_loop_wall_synchronized_s_including_validation_and_io'],
                   test_forward_s=test_views[0]['forward_seconds'],
                   reference_test_forward_s=ref_test['views'][0]['single_call_synchronized_seconds'])
        for name, value in stream['storage_bytes_by_class'].items():
            row[name] = value
        for level in stream['activities']['levels']:
            prefix = f"level{level['level']}"
            row[f'{prefix}_visited'] = level['visited_count']
            for action, fraction in level['fractions'].items():
                row[f'{prefix}_{action}'] = fraction
        rows.append(row)
        views.extend(dict(frame=frame, arm='candidate', **view) for view in evaluation['views'])
        views.extend(dict(frame=frame, arm='reference', **view) for view in ref_test['views'])
        control = json.loads((run / 'frame_summary' /
                              f'control_{frame:03d}.json').read_text())['quality']
        for stage, splits in control.items():
            assert 'validation_probe' not in splits
            for split, item in splits.items():
                phases.extend(dict(frame=frame, stage=stage, split=split, **view)
                              for view in item.get('views', []))
    assert rows and [row['frame'] for row in rows] == list(range(1, rows[-1]['frame'] + 1))
    keys = sorted({key for row in rows for key in row if key != 'frame'})
    blocks = []
    for start, end in ((1, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299)):
        subset = [row for row in rows if start <= row['frame'] <= end]
        if subset:
            blocks.append(dict(start=start, end=end, completed=len(subset),
                               means={key: average(subset, key) for key in keys}))
    summary = dict(run=str(run), completed=len(rows), complete299=len(rows) == 299,
                   protocol=dict(train_cameras=sorted(cameras), test_camera='cam00.png',
                                 validation_cameras=[], selection_test00=False,
                                 per_frame_steps=200, candidate_extra_limit=14,
                                 resolution=[1600, 1199]),
                   frame0_sha256=hashlib.sha256(Path(config['ckpt_path']).read_bytes()).hexdigest(),
                   means={key: average(rows, key) for key in keys}, blocks=blocks,
                   endpoint=rows[-1],
                   fps=dict(candidate=len(rows) / sum(row['test_forward_s'] for row in rows),
                            reference=len(rows) / sum(row['reference_test_forward_s'] for row in rows)),
                   extra_candidate_fraction=sum(row['extra_candidate_renders'] for row in rows) /
                   (200 * len(rows)),
                   limitations=[
                       'One seed; algorithm and model structures differ from native REF.',
                       'Shared native frame0; aggregate contains dynamic frames1–299 only.',
                       'Test cam00 is passive reporting, never tuning or early stopping.',
                       '200 base steps plus candidate graphs; no equal-compute claim.',
                       'Full measured costs include diagnostics and checkpoint IO; REF instrumentation differs.',
                       'FPS uses synchronized single-view forward+clamp, not batch benchmark FPS.',
                       'Effective parameter bytes exclude membership metadata, file headers and training artifacts.',
                       'Training camera order is seeded; not asserted identical to native REF global RNG.'])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'refaligned_summary.json').write_text(json.dumps(summary, indent=2))
    for name, data in (('frames.csv', rows), ('float_views.csv', views), ('phase_views.csv', phases)):
        if data:
            with (output / name).open('w') as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in data for key in row}))
                writer.writeheader()
                writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
    frames = [row['frame'] for row in rows]
    axes[0].plot(frames, [row['test_psnr'] for row in rows], label='Frozen candidate')
    axes[0].plot(frames, [row['reference_test_psnr'] for row in rows], label='Native REF')
    axes[1].plot(frames, [row['test_delta_psnr'] for row in rows], label='Candidate - REF')
    axes[1].axhline(0, color='gray', linewidth=.8)
    axes[2].plot(frames, [row['actual_changed_fraction'] for row in rows], label='Old GS pose changed')
    axes[2].plot(frames, [row['local_fraction'] for row in rows], label='GS-local path open')
    axes[3].plot(frames, [row['hard_bytes'] / 1024 for row in rows], label='Effective parameter estimate')
    axes[4].plot(frames, [row['training_loop_s'] for row in rows], label='Candidate loop incl. state update')
    axes[4].plot(frames, [row['reference_training_s'] for row in rows], label='REF recorded training loop')
    axes[5].plot(frames, [row['wall_s'] for row in rows], label='Candidate full frame')
    axes[5].plot(frames, [row['reference_wall_s'] for row in rows], label='REF full frame')
    labels = ('Test cam00 PSNR (dB)', 'Test difference (dB)', 'GS fraction',
              'Estimated parameters (KiB)', 'Training seconds', 'Full frame seconds')
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.legend()
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('Dynamic frame from shared native frame0')
    figure.suptitle(f'Train cam01-20 / test cam00: {len(rows)}/299 completed\n'
                   'Candidate 133+67 plus <=14 auxiliary graphs; REF 100+100')
    figure.tight_layout()
    figure.savefig(output / 'refaligned_comparison.png', dpi=150)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.run, args.reference, args.output)
    print(json.dumps(dict(completed=summary['completed'], means=summary['means'],
                          fps=summary['fps'], extra_candidate_fraction=summary['extra_candidate_fraction'])))


if __name__ == '__main__':
    main()
