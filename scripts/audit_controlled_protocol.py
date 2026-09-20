"""Fail closed on missing frames, step/view mismatches, storage or source errors."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

from omegaconf import OmegaConf

from summarize_controlled import read_jsonl


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run, reference=None):
    config = OmegaConf.load(run / 'config.yaml')
    frames = list(range(1, int(config.data.extra_dataset_kwargs.num_frames)))
    steps = int(config.trainer.num_incr_steps)
    motion_steps = int(config.module.deformation.num_stage1_steps)
    views = defaultdict(list)
    for row in read_jsonl(run / 'training_views.jsonl'):
        if row['frame'] > 0:
            views[row['frame']].append(row)
    require(sorted(views) == frames, 'Incomplete training frame set')
    times = [r for r in read_jsonl(run / 'frame_times.jsonl') if r['frame'] > 0]
    require(sorted(r['frame'] for r in times) == frames, 'Incomplete/duplicate frame times')
    tests = defaultdict(list)
    for row in read_jsonl(run / 'evaluation.jsonl'):
        if row['frame'] > 0 and row['split'] == 'test':
            tests[row['frame']].append(row)
    require(sorted(tests) == frames, 'Incomplete evaluation frame set')

    source_manifest = json.loads((run / 'controlled_manifest.json').read_text())
    snapshot_manifest = json.loads((run / 'source_snapshot_manifest.json').read_text())
    with zipfile.ZipFile(run / 'source_snapshot.zip') as archive:
        require(set(archive.namelist()) == set(snapshot_manifest), 'Snapshot file list differs')
        for name, expected in snapshot_manifest.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected,
                    f'Snapshot hash mismatch: {name}')
        for name, expected in source_manifest['source_sha256'].items():
            require(snapshot_manifest['source/' + name] == expected,
                    f'Source manifest mismatch: {name}')

    rounds = accepted = mixed_depth_rounds = total_extra = 0
    max_error = 0.
    for frame in frames:
        main = views[frame]
        require([r['step'] for r in main] == list(range(1, steps + 1)),
                f'Frame {frame}: missing/reordered steps')
        training_paths = {r['path'] for r in main}
        require(len(training_paths) == 20, f'Frame {frame}: expected 20 training views')
        for name, lo, hi in (('motion', 0, motion_steps), ('incremental', motion_steps, steps)):
            counts = Counter(r['path'] for r in main if lo < r['step'] <= hi)
            require(set(counts) == training_paths and set(counts.values()) == {(hi - lo) // 20}
                    and (hi - lo) % 20 == 0, f'Frame {frame}: nonuniform {name} cameras')
        final = [r for r in tests[frame] if r['step'] == steps]
        require(len(final) == 1 and Path(final[0]['path']).stem == 'cam00',
                f'Frame {frame}: wrong final test view/step')
        require(final[0]['path'] not in training_paths, f'Frame {frame}: test leakage')
        directory = run / 'frame_storage'
        stats = json.loads((directory / f'frame_{frame:03d}.json').read_text())
        motion_bytes = (directory / f'motion_{frame:03d}.npz').stat().st_size
        incremental_bytes = (directory / f'incremental_{frame:03d}.ply').stat().st_size
        require(stats['motion_bytes'] == motion_bytes and stats['incremental_bytes'] == incremental_bytes
                and stats['total_bytes'] == motion_bytes + incremental_bytes,
                f'Frame {frame}: actual file bytes differ')
        event_path = run / 'hierarchy_stats' / f'frame_{frame:03d}_binary.json'
        events = json.loads(event_path.read_text()) if event_path.exists() else []
        require(sum(e['extra_backward_steps'] for e in events) == stats.get('extra_backward_steps', 0),
                f'Frame {frame}: unaccounted trial backward calls')
        for event in events:
            rounds += 1
            accepted += event['accepted']
            mixed_depth_rounds += len(set(event['depths'])) > 1
            require(set(event['train_views']) <= training_paths
                    and set(event['decision_views']) <= training_paths,
                    f'Frame {frame}: nontraining trial camera')
            if not event['decision_uses_all_training_views']:
                require(set(event['train_views']).isdisjoint(event['decision_views']),
                        f'Frame {frame}: trial/decision views overlap')
            require(event['accepted'] == (event['objective_after'] < event['objective_before']),
                    f'Frame {frame}: decision inconsistent with RD cost')
        total_extra += stats.get('extra_backward_steps', 0)
        for key in ('decode_xyz_max_error', 'decode_rotation_max_error'):
            max_error = max(max_error, stats.get(key, 0))
        if reference is not None:
            relative = Path('image/frames') / f'{frame:04d}' / 'cam00_gt.png'
            require(hashlib.sha256((run / relative).read_bytes()).digest() ==
                    hashlib.sha256((reference / relative).read_bytes()).digest(),
                    f'Frame {frame}: test ground truth differs')
    require(max_error <= 1e-7, 'Exported motion round-trip error exceeds tolerance')
    result = dict(run=str(run.resolve()), dynamic_frames=len(frames), regular_steps_per_frame=steps,
                  motion_steps=motion_steps, incremental_steps=steps-motion_steps,
                  extra_backward_calls=total_extra, trials=rounds, accepted_trials=accepted,
                  mixed_depth_candidate_rounds=mixed_depth_rounds,
                  exported_max_decode_error=max_error, source_snapshot_verified=True,
                  all_test_gt_identical_to_reference=reference is not None)
    (run / 'protocol_audit.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.reference), indent=2))
