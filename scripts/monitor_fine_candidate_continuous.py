"""Report frozen fine-candidate progress and the predeclared TRAIN-only guard."""
import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--guard-config', type=Path, required=True)
    args = parser.parse_args()
    guard = yaml.safe_load(args.guard_config.read_text())
    baseline = Path(guard['reference_run'])
    ref_file = Path('output/recongs_native_stream_20261002/recongs_flame_steak_native_delta_index_seed0_20261002_rerun01/20261002T231157_base02/stream_metrics/frames.jsonl')
    ref = {row['frame']: row for line in ref_file.read_text().splitlines()
           if (row := json.loads(line))['frame'] > 0}
    frames = {}
    for path in (args.run / 'frame_summary').glob('frame_*.json'):
        try:
            row = json.loads(path.read_text())
            if row['frame'] > 0 and 'wall' in row.get('timing', {}):
                frames[row['frame']] = row
        except (ValueError, KeyError):
            continue
    last = max(frames, default=0)
    if not last:
        print(json.dumps(dict(frame=0, numerical_failure=False,
                             severe_training_regression=False)))
        return
    assert sorted(frames) == list(range(1, last + 1))
    cameras = {f'cam{i:02d}.png' for i in range(1, 21)}
    current, previous = {}, {}
    for frame, row in frames.items():
        assert set(row['quantized_evaluation']) == {'test'}
        trace = json.loads((args.run / 'frame_summary' / f'training_views_{frame:03d}.json').read_text())['views']
        assert len(trace) == 200 and set(trace) == cameras
        actual = row['stream']['actual_motion']
        assert actual['keep_position_changed_count'] == actual['keep_rotation_changed_count'] == 0
        assert row['stream']['candidate_fit']['extra_renders'] <= guard['maximum_extra_graphs']
        for root, target in ((args.run, current), (baseline, previous)):
            target[frame] = json.loads((root / 'frame_summary' / f'control_{frame:03d}.json').read_text())['quality']['after_retention']['train_probe']['mean']['image_objective']
    window = guard['window_frames']
    windows = []
    consecutive = 0
    severe = False
    for end in range(window, last + 1, window):
        indices = list(range(end - window + 1, end + 1))
        numerator = statistics.fmean(current[f] for f in indices)
        denominator = statistics.fmean(previous[f] for f in indices)
        ratio = numerator / denominator
        consecutive = consecutive + 1 if ratio >= guard['severe_train_image_objective_ratio'] else 0
        windows.append(dict(start=indices[0], end=end, train_image_objective=numerator,
                            baseline_train_image_objective=denominator, ratio=ratio))
        if end >= guard['first_quality_check_frame'] and consecutive >= guard['consecutive_nonoverlapping_windows']:
            severe = True
    indices = list(range(max(1, last - 9), last + 1))
    def mean(call):
        return statistics.fmean(call(frames[f], f) for f in indices)
    test = mean(lambda row, f: row['quantized_evaluation']['test']['psnr'])
    old = statistics.fmean(json.loads((baseline / 'frame_summary' / f'frame_{f:03d}.json').read_text())['quantized_evaluation']['test']['psnr'] for f in indices)
    native = statistics.fmean(ref[f]['evaluation']['test']['psnr'] for f in indices)
    result = dict(frame=last, recent_window=indices, test_cam00_recent=test,
        old_formal_test_recent=old, delta_old_formal=test-old,
        native_reference_recent=native, delta_native=test-native,
        train_probe_image_recent=statistics.fmean(current[f] for f in indices),
        old_train_probe_image_recent=statistics.fmean(previous[f] for f in indices),
        actual_changed_recent=mean(lambda row, f: row['stream']['actual_motion']['any_pose_changed_count'] / row['stream']['actual_motion']['old_surviving_count']),
        actual_local_open_recent=mean(lambda row, f: row['stream']['activities']['gs_local_active_fraction']),
        hard_bytes_recent=mean(lambda row, f: row['stream']['hard_storage_bytes']),
        inclusive_training_seconds_recent=mean(lambda row, f: row['timing']['training_loop_including_state_update']),
        wall_seconds_recent=mean(lambda row, f: row['timing']['wall']),
        extra_graphs_recent=sum(frames[f]['stream']['candidate_fit']['extra_renders'] for f in indices),
        checkpoint_frames=sorted(int(p.name.split('-')[1]) for p in args.run.glob('ckpt-*-200.pth')),
        actual_all20_training=True, actual_only_test00=True, test_used_for_stop=False,
        numerical_failure=not all(math.isfinite(value) for f, row in frames.items() for value in
            (row['quantized_evaluation']['test']['psnr'], row['stream']['hard_storage_bytes'],
             row['stream']['last_training_metrics']['loss/image'], current[f])),
        severe_training_regression=severe, guard_windows=windows,
        utc=datetime.now(timezone.utc).isoformat())
    output = args.run / 'analysis/progress'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'latest.json').write_text(json.dumps(result, indent=2))
    state_file = Path('/tmp/recongs_active_task_state_20261003.json')
    state = json.loads(state_file.read_text())
    state['fine_candidate_continuous']['latest_progress'] = result
    if state.get('active_training') and state['active_training'].get('run') == str(args.run):
        state['active_training']['last_completed_frame'] = last
    state_file.write_text(json.dumps(state, indent=2))
    history_file = output / 'milestones.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else []
    if not history or last - history[-1]['frame'] >= 10 or last == 299:
        history.append(result)
        history_file.write_text(json.dumps(history, indent=2))
        with Path('/tmp/recongs_parent_status_20261003_1402.md').open('a') as handle:
            handle.write(f'\nFineLONG frame{last}: test{test:.5f}, oldformal{old:.5f} delta{test-old:+.5f},native{native:.5f};local{result["actual_local_open_recent"]:.4%}/actualGS{result["actual_changed_recent"]:.3%},bytes{result["hard_bytes_recent"]:.1f},inclusive{result["inclusive_training_seconds_recent"]:.3f}s,wall{result["wall_seconds_recent"]:.3f}s.20train/test00only/KEEP0/aux<=14; severeTRAINguard={severe}; testnotstop.\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
