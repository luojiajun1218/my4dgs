"""Reuse logged training renders for the frozen 20-camera experiment."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((args.run / 'config.yaml').read_text())
    assert config['data']['training_camera_names'] is None
    assert config['data']['validation_camera_names'] is None
    motion_steps = config['module']['quantized_residual']['motion_steps']
    steps = config['trainer']['num_incr_steps']
    assert steps == 200 and motion_steps == 133
    completed = []
    for path in sorted((args.run / 'frame_summary').glob('frame_*.json')):
        try:
            item = json.loads(path.read_text())
            if item['frame'] > 0 and 'wall' in item.get('timing', {}):
                completed.append(item['frame'])
        except (ValueError, KeyError):
            continue
    assert completed == list(range(1, len(completed) + 1))
    accumulator = EventAccumulator(str(args.run), size_guidance={'scalars': 0},
                                   purge_orphaned_data=False).Reload()
    tags = set(accumulator.Tags()['scalars'])
    required = ['psnr', 'loss/image']
    assert all(tag in tags for tag in required)
    events = {tag: accumulator.Scalars(tag) for tag in required}
    cameras = {f'cam{index:02d}.png' for index in range(1, 21)}
    rows = []
    for index, frame in enumerate(completed):
        start = index * steps
        if any(len(values) < start + steps for values in events.values()):
            break  # Last event batch can still be buffered during an active run.
        chunks = {tag: values[start:start + steps] for tag, values in events.items()}
        assert all([value.step for value in values] == list(range(1, steps + 1))
                   for values in chunks.values())
        trace = json.loads((args.run / 'frame_summary' /
                            f'training_views_{frame:03d}.json').read_text())['views']
        assert len(trace) == steps and set(trace) == cameras
        row = dict(frame=frame, last20_distinct_cameras=len(set(trace[-20:])))
        for name, begin, end in (('all', 0, steps), ('motion', 0, motion_steps),
                                 ('birth', motion_steps, steps), ('last20', steps - 20, steps)):
            for tag, values in chunks.items():
                metric = 'psnr' if tag == 'psnr' else 'image_objective'
                row[f'{name}_{metric}'] = statistics.fmean(value.value for value in values[begin:end])
        for camera in sorted(cameras):
            indices = [i for i, name in enumerate(trace) if name == camera]
            row[f'{camera}_samples'] = len(indices)
            for tag, values in chunks.items():
                metric = 'psnr' if tag == 'psnr' else 'image_objective'
                row[f'{camera}_{metric}'] = statistics.fmean(values[i].value for i in indices)
        rows.append(row)
    assert rows
    keys = sorted({key for row in rows for key in row if key != 'frame'})
    blocks = []
    for start, end in ((1, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299)):
        subset = [row for row in rows if start <= row['frame'] <= end]
        if subset:
            blocks.append(dict(start=start, end=end, completed=len(subset),
                               means={key: statistics.fmean(row[key] for row in subset) for key in keys}))
    summary = dict(completed_frames=len(completed), logged_complete_frames=len(rows),
                   means={key: statistics.fmean(row[key] for row in rows) for key in keys},
                   blocks=blocks,
                   semantics='200 changing optimization renders per frame; 20-camera sampled trajectory, not fixed final-pose evaluation. Last20 remains an optimization trajectory even if all cameras occur once. No new renders or updates.',
                   stage_steps=[133, 67], selection_test00=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'training_trajectory_summary.json').write_text(json.dumps(summary, indent=2))
    with (args.output / 'training_trajectory.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['frame'] + keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(completed_frames=len(completed), logged_complete_frames=len(rows),
                          all_psnr=summary['means']['all_psnr'],
                          all_image_objective=summary['means']['all_image_objective'])))


if __name__ == '__main__':
    main()
