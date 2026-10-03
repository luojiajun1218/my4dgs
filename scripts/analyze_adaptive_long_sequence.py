"""Summarize passive long-sequence diagnostics without selecting a checkpoint."""
import argparse
import csv
import json
import statistics
from pathlib import Path


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def analyze(run, reference):
    ref = {int(r['frame']): r for r in
           (json.loads(line) for line in reference.read_text().splitlines() if line.strip())}
    rows = []
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        d = json.loads(path.read_text())
        frame = d['frame']
        stream = d.get('stream')
        if stream is None or frame == 0:
            continue
        quality = d['quantized_evaluation']
        test = quality.get('test', {})
        validation = quality.get('validation', {})
        baseline = ref.get(frame, {}).get('evaluation', {}).get('test', {})
        ref_views = ref.get(frame, {}).get('evaluation', {}).get('train', {}).get('views', [])
        ref_validation = [v for v in ref_views if v.get('view') in
                          ('cam17.png', 'cam18.png', 'cam19.png', 'cam20.png')]
        actual = stream.get('actual_motion') or {}
        activity = stream['activities']
        births = stream['birth_control']
        row = dict(frame=frame, test_psnr=test.get('psnr'), test_ssim=test.get('ssim'),
                   test_l1=test.get('l1'), validation_psnr=validation.get('psnr'),
                   validation_ssim=validation.get('ssim'), ref_psnr=baseline.get('psnr'),
                   ref_validation_psnr=(statistics.fmean(v['psnr'] for v in ref_validation)
                                        if ref_validation else None),
                   ref_ssim=baseline.get('ssim'), ref_l1=baseline.get('l1'),
                   hard_bytes=stream['hard_storage_bytes'],
                   training_s=d['timing']['training'], wall_s=d['timing']['wall'],
                   keep_fraction=activity['gs']['fractions']['KEEP'],
                   local_fraction=activity['gs_local_active_fraction'],
                   root_split_fraction=activity['levels'][0]['fractions']['SPLIT'],
                   born=stream['new_count'], deleted=stream['deleted_count'],
                   total_gs=births['point_origins']['final_total'],
                   remaining_first_frame=births['point_origins']['remaining_first_frame'],
                   tree_action=stream['tree_setup']['action'],
                   actual_changed_fraction=(actual['any_pose_changed_count'] /
                                            actual['old_surviving_count'] if actual else None),
                   keep_position_violations=actual.get('keep_position_changed_count'),
                   keep_rotation_violations=actual.get('keep_rotation_changed_count'))
        if row['test_psnr'] is not None and row['ref_psnr'] is not None:
            row['delta_psnr'] = row['test_psnr'] - row['ref_psnr']
            row['delta_ssim'] = row['test_ssim'] - row['ref_ssim']
            row['delta_l1'] = row['test_l1'] - row['ref_l1']
        if row['validation_psnr'] is not None and row['ref_validation_psnr'] is not None:
            row['validation_delta_psnr'] = row['validation_psnr'] - row['ref_validation_psnr']
        control_path = path.parent / f'control_{frame:03d}.json'
        if control_path.exists():
            control = json.loads(control_path.read_text())
            # save_quantized_control_diagnostics wraps phase snapshots in quality.
            control = control.get('quality', control)
            for split in ('train_probe', 'validation_probe'):
                for stage in ('before_deletion', 'after_deletion', 'after_initialization', 'after_motion',
                              'before_retention', 'after_retention'):
                    value = control.get(stage, {}).get(split, {}).get('mean', {}).get('psnr')
                    row[f'{split}_{stage}_psnr'] = value
                for name, before, after in (
                        ('deletion', 'before_deletion', 'after_deletion'),
                        ('prediction', 'after_deletion', 'after_initialization'),
                        ('motion_fit', 'after_initialization', 'after_motion'),
                        ('motion', 'after_deletion', 'after_motion'),
                        ('birth', 'after_motion', 'before_retention'),
                        ('retention', 'before_retention', 'after_retention')):
                    a = row.get(f'{split}_{before}_psnr')
                    b = row.get(f'{split}_{after}_psnr')
                    row[f'{split}_{name}_delta'] = b - a if a is not None and b is not None else None
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    blocks = []
    for start, end in ((1, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 299)):
        block = [r for r in rows if start <= r['frame'] <= end]
        if block:
            blocks.append(dict(start=start, end=end, completed=len(block),
                               means={key: mean(block, key) for key in fields if key != 'tree_action'}))
    summary = dict(run=str(run), reference=str(reference), completed_frames=len(rows),
                   last_frame=rows[-1]['frame'] if rows else 0, blocks=blocks,
                   means={key: mean(rows, key) for key in fields if key != 'tree_action'},
                   limitations=['REF trained on 20 cameras; candidate trained on 16 with 4 validation. '
                                 'Same-frame differences are descriptive, not a paired method effect.',
                                 'Storage is an effective parameter estimate, not encoded bytes.',
                                 'Probe stage metrics use one train and one validation camera.',
                                 'KEEP is not a semantic background label; no foreground masks are assumed.'])
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    rows, summary = analyze(args.run, args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (args.output / 'frames.csv').open('w') as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if args.plot and rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
        frames = [r['frame'] for r in rows]
        axes[0].plot(frames, [r['test_psnr'] for r in rows], label='Adaptive cam00')
        axes[0].plot(frames, [r['ref_psnr'] for r in rows], label='REF cam00 (20 train cameras)')
        axes[0].set_ylabel('PSNR (dB)')
        axes[1].plot(frames, [r.get('delta_psnr') for r in rows], label='Adaptive - REF')
        axes[1].axhline(0, color='gray', linewidth=.8)
        axes[1].set_ylabel('PSNR difference (dB)')
        axes[2].plot(frames, [r['keep_fraction'] for r in rows], label='Terminal KEEP')
        axes[2].plot(frames, [r['actual_changed_fraction'] for r in rows], label='Actual pose changed')
        axes[2].plot(frames, [r['local_fraction'] for r in rows], label='GS local open')
        axes[2].set_ylabel('Old GS fraction')
        axes[3].plot(frames, [r['hard_bytes'] / 1024 for r in rows], label='Effective parameters (KiB)')
        axes[3].set_ylabel('KiB estimate')
        axes[3].set_xlabel('Dynamic frame')
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(loc='best')
        fig.suptitle('Continuous adaptive hierarchy: passive diagnostics')
        fig.tight_layout()
        fig.savefig(args.output / 'long_sequence.png', dpi=150)
        plt.close(fig)
    print(json.dumps(dict(completed=summary['completed_frames'], last_frame=summary['last_frame'],
                          means=summary['means'], blocks=summary['blocks']), indent=2))


if __name__ == '__main__':
    main()
