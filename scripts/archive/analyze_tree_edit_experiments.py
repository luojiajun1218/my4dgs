"""Read completed tree-edit runs without modifying their files or checkpoints."""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics

import yaml


def inspect_run(run, frames=None):
    run = Path(run).resolve()
    config = yaml.safe_load((run / 'config.yaml').read_text())
    rows = []
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        data = json.loads(path.read_text())
        frame = data['frame']
        if not frame:
            continue
        stream, quality = data['stream'], data['quantized_evaluation']['test']
        motion = run / 'frame_storage/common_contrast_binary' / f'motion_{frame:03d}.bin'
        birth = motion.with_name(f'incremental_{frame:03d}.ply')
        actual = motion.stat().st_size + (birth.stat().st_size if birth.exists() else 0)
        if actual != stream['inference_file_bytes']:
            raise AssertionError(f'Frame {frame} file sizes differ from its report')
        birth_control = stream.get('birth_control', {})
        activities = stream['activities']
        changed = activities.get('gs', {})
        retained_limit = birth_control.get('max_retained_gaussians')
        rows.append(dict(frame=frame, psnr=quality['psnr'], ssim=quality['ssim'],
            lpips=quality['lpips'], fps=quality['fps'], necessary_bytes=actual,
            motion_bytes=motion.stat().st_size, birth_bytes=actual-motion.stat().st_size,
            structure_bytes=stream['motion_export']['structure_edit_bytes'],
            new_gs=stream['new_count'], fine_nodes=activities['region_counts'][-1],
            retained_cap_hit=int(retained_limit is not None and stream['new_count'] >= retained_limit),
            retention_pruned=birth_control.get('retention_budget_pruned', 0),
            changed_gs_fraction=changed.get('changed', 0) / max(changed.get('denominator', 0), 1),
            remaining_first_frame=birth_control.get('point_origins', {}).get('remaining_first_frame', 0),
            wall_seconds=data['timing']['wall'], training_seconds=data['timing']['training']))
    if frames is not None:
        if [row['frame'] for row in rows] != list(range(1, frames + 1)):
            raise AssertionError('Requested completed frame range is missing')
        if not (run / f'ckpt-{frames}-{config["trainer"]["num_incr_steps"]}.pth').exists():
            raise AssertionError('Final checkpoint is missing')
    trials = []
    for path in sorted((run / 'direct_tree_trials').glob('*/report.json')):
        data = json.loads(path.read_text())
        selection = data.get('split_selection') or {}
        candidates = selection.get('selected_regions', [])
        operations = collections.Counter(action for window in data.get('windows', []) for action in window['actions'])
        trial = dict(frame=int(path.parent.name.rsplit('_', 1)[-1]), reason=data['reason'],
            accepted=data.get('accepted', False), selected=len(candidates),
            clustered=sum(node['clustered'] for node in candidates),
            applied=sum(node.get('edit_applied', False) for node in candidates), actions=dict(operations),
            pruning=data.get('split_pruning'), branches=data.get('branches'),
            proposal_seconds=data.get('proposal_seconds'), proposal_step=data.get('proposal_step'))
        trials.append(trial)
    numeric = [key for key in rows[0] if key != 'frame'] if rows else []
    means = {key: statistics.fmean(row[key] for row in rows) for key in numeric}
    snapshots = {}
    for entry in json.loads((run / 'source_snapshot/manifest.json').read_text()):
        path = run / 'source_snapshot' / entry['snapshot']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise AssertionError('Saved source snapshot hash differs')
        if not path.suffix == '.yaml':
            snapshots[Path(entry['source']).name] = digest
    blocks = [dict(first=block[0]['frame'], last=block[-1]['frame'], frames=len(block),
                   means={key: statistics.fmean(row[key] for row in block) for key in numeric})
              for start in range(0, len(rows), 50) if (block := rows[start:start + 50])]
    return dict(run=str(run), config=config, dynamic_frames=len(rows), rows=rows, means=means,
                blocks_50_frames=blocks, worst_frames=sorted(rows, key=lambda row: row['psnr'])[:10],
                trials=trials, accepted_trials=sum(trial['accepted'] for trial in trials),
                dynamic_bytes=sum(row['necessary_bytes'] for row in rows), source_hashes=snapshots)


def compare(control, candidate):
    left, right = control['config'], candidate['config']
    for section in ('data', 'trainer'):
        if left[section] != right[section]:
            raise ValueError(f'Comparison protocols differ in {section}')
    if left['ckpt_path'] != right['ckpt_path'] or left['random_seed'] != right['random_seed']:
        raise ValueError('First-frame source or seed differs')
    permitted = {'direct_gradient_split_enabled', 'direct_candidate_windows', 'direct_enable_merges',
                 'direct_initial_levels'}
    differences = {key: [left['module']['common_contrast_binary'].get(key), value]
                   for key, value in right['module']['common_contrast_binary'].items()
                   if left['module']['common_contrast_binary'].get(key) != value}
    if set(differences) - permitted:
        raise ValueError(f'Additional method/settings differences: {differences}')
    if control['source_hashes'] != candidate['source_hashes']:
        raise ValueError('Motion source snapshots differ; review before comparing')
    paired = []
    baseline = {row['frame']: row for row in control['rows']}
    for row in candidate['rows']:
        if row['frame'] not in baseline:
            continue
        old = baseline[row['frame']]
        paired.append(dict(frame=row['frame'], control_psnr=old['psnr'], candidate_psnr=row['psnr'],
            delta_psnr=row['psnr']-old['psnr'], control_bytes=old['necessary_bytes'],
            candidate_bytes=row['necessary_bytes'], delta_bytes=row['necessary_bytes']-old['necessary_bytes'],
            fine_nodes=row['fine_nodes']))
    return dict(protocol_differences=differences, paired=paired,
        mean_delta_psnr=statistics.fmean(row['delta_psnr'] for row in paired),
        worst_delta_psnr=min(row['delta_psnr'] for row in paired),
        best_delta_psnr=max(row['delta_psnr'] for row in paired),
        candidate_byte_ratio=sum(row['candidate_bytes'] for row in paired)/sum(row['control_bytes'] for row in paired))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--frames', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    runs = [inspect_run(run, args.frames) for run in args.run]
    result = dict(runs=runs, limitations=['FPS excludes decoding and data loading.',
        'Dynamic file bytes exclude the shared first frame and configuration.',
        'Sequential state equality requires a separate decoder audit.'])
    if len(runs) == 2:
        result['comparison'] = compare(*runs)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        for index, run in enumerate(runs):
            with (args.output / f'frames_{index}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(run['rows'][0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(run['rows'])
    compact = [dict(run=run['run'], frames=run['dynamic_frames'], means=run['means'],
                    accepted_trials=run['accepted_trials']) for run in runs]
    print(json.dumps(dict(runs=compact, comparison={key:value for key,value in result.get('comparison',{}).items()
                                                  if key != 'paired'}), ensure_ascii=False))


if __name__ == '__main__':
    main()
