"""Read existing experiments and export compact, auditable research records.

This script never trains, renders, loads checkpoints, or writes into source runs.
Frame indices are normalized to zero-based indices; primary means exclude frame 0.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
SOURCE_FILES = {}


def display(path):
    path = Path(path).absolute()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_text(path):
    path = Path(path)
    raw = path.read_bytes()
    SOURCE_FILES[display(path)] = {
        'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
    }
    return raw.decode('utf-8')


def read_json(path):
    return json.loads(read_text(path))


def parse_namespace(text):
    """Parse argparse's historical Namespace repr without executing it."""
    node = ast.parse(text.strip(), mode='eval').body
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == 'Namespace' and not node.args):
        raise ValueError('Expected a keyword-only Namespace(...) snapshot')
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def parse_s2gs_progress(text):
    """Keep completed frames only, deduplicating repeated tqdm refreshes.

    A row at 14/300 with Frame index=15 is still in progress. Its later
    15/300 row is complete. Values have only the precision printed by tqdm.
    """
    fields = {
        'psnr': 'PSNR (Test)', 'legacy_test_loss': 'Loss (Test)', 'fps': 'FPS',
        'legacy_frame_seconds': 'Frame time', 'legacy_frame_io_seconds': 'Frame time IO',
        'legacy_estimated_size_mb': 'Size (MB)', 'gs_count': 'Num points',
        'update_gs': 'Update points', 'active_gates': 'Active gates',
    }
    complete = {}
    for line in re.split(r'[\r\n]', text):
        if 'Training progress:' not in line or 'Frame index=' not in line:
            continue
        count = re.search(r'\|\s*(\d+)/\d+\s*\[', line)
        frame = re.search(r'Frame index=(\d+)', line)
        if count is None or int(count[1]) != int(frame[1]):
            continue
        row = {'frame': int(frame[1]) - 1, 'original_frame_index': int(frame[1])}
        for key, label in fields.items():
            value = re.search(re.escape(label) + r'=([-+\d.eE]+)(?=,|\])', line)
            if value:
                row[key] = float(value[1])
        if 'psnr' in row:
            complete[row['frame']] = row
    return [complete[k] for k in sorted(complete)]


def parse_recongs_log(text):
    frame, pending = None, None
    rows, starts, timers = [], set(), {}
    for line in text.splitlines():
        start = re.search(r'==> Start learning frame (\d+)', line)
        if start:
            frame = int(start[1])
            starts.add(frame)
        evaluation = re.search(r'Evaluate (\S+) dataset at step (\d+):', line)
        if evaluation:
            if frame is None:
                raise ValueError('Evaluation appeared before a frame marker')
            pending = {'frame': frame, 'step': int(evaluation[2]), 'split': evaluation[1]}
        elif pending and 'PSNR:' in line:
            values = re.findall(r'(L1|PSNR|SSIM|FPS): ([-+\d.eE]+)', line)
            pending.update({key.lower(): float(value) for key, value in values})
            rows.append(pending)
            pending = None
        timing = re.match(r'\s*frame:\s*([\d.]+).*\(n=1, sum=', line)
        if timing and frame is not None:
            timers[frame] = float(timing[1])
    return rows, starts, timers


def refine_from_tensorboard(run, rows):
    """Pair event order with log frame markers, verifying every step/value.

    ReCon-GS resets TensorBoard step each frame. Treating step=200 as a
    unique global index would discard almost the entire video.
    """
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

    events = defaultdict(list)
    for path in sorted(run.glob('events.out.tfevents.*')):
        # Hash only event files, never checkpoints or image data.
        raw = path.read_bytes()
        SOURCE_FILES[display(path)] = {
            'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
        }
        for event in LegacyEventFileLoader(str(path)).Load():
            for value in event.summary.value:
                if value.tag.startswith('eval-') and value.HasField('simple_value'):
                    events[value.tag].append((event.step, value.simple_value))
    if not events:
        return False
    for split in sorted({row['split'] for row in rows}):
        subset = [row for row in rows if row['split'] == split]
        for metric in ('l1', 'psnr', 'ssim', 'fps'):
            values = events[f'eval-{split}/{metric}']
            if len(values) != len(subset):
                raise ValueError(f'{run}: event/log count differs for {split}/{metric}')
            for row, (step, value) in zip(subset, values):
                # Logs round to four decimals; Python FPS is additionally
                # rounded to float32 when TensorBoard stores its scalar.
                tolerance = 0.000051 + abs(value) * 2**-24
                if step != row['step'] or abs(value - row[metric]) > tolerance:
                    raise ValueError(f'{run}: event/log mismatch for {metric}: {row}, event={value}')
                row[metric] = value
    return True


def means(rows, keys):
    result = {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'Non-finite {key} in research records')
            result[key] = statistics.fmean(values)
    return result


def inventory_bytes(run):
    """Logical file sizes; do not follow or double-count symlinks/hard links."""
    sizes, counts, seen = Counter(), Counter(), set()
    for root, dirs, names in os.walk(run, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(root) / name).is_symlink()]
        for name in names:
            path = Path(root) / name
            if path.is_symlink():
                continue
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                continue
            seen.add(inode)
            category = ('checkpoint' if path.suffix in {'.pth', '.pkl'} else
                        'ply' if path.suffix == '.ply' else
                        'images' if path.suffix in {'.png', '.jpg', '.jpeg'} else
                        'encoded_bin' if path.suffix == '.bin' else 'other')
            sizes[category] += stat.st_size
            counts[category] += 1
    return {'total_bytes': sum(sizes.values()), 'bytes_by_category': dict(sizes),
            'files_by_category': dict(counts), 'semantics': 'Logical file sizes, unique inodes, no symlink traversal'}


def stream_frame(run, report):
    stream = report['stream']
    frame = report['frame']
    packet = run / 'incremental_stream' / f'frame_{frame:03d}.bin'
    actual_bytes = packet.stat().st_size
    if actual_bytes != stream['actual']['total_bytes']:
        raise ValueError(f'{packet}: actual size does not match frame report')
    bootstrap_bytes = stream['bootstrap_bytes']
    if actual_bytes + bootstrap_bytes != stream['total_stream_bytes']:
        raise ValueError(f'{packet}: bootstrap accounting mismatch')
    if sum(stream['actual_bytes_by_class'].values()) != stream['total_stream_bytes']:
        raise ValueError(f'{packet}: class totals do not match container bytes')
    sidecar = read_json(packet.with_suffix('.json'))
    for key in ('frame', 'total_stream_bytes', 'estimated_total_bits', 'decoded_reference_sha256'):
        if sidecar[key] != stream[key]:
            raise ValueError(f'{packet}: sidecar and summary differ at {key}')
    activity = stream['activities']['gs']['fractions']
    return dict(stream_bytes=stream['total_stream_bytes'], bootstrap_bytes=bootstrap_bytes,
                estimated_bits=stream['estimated_total_bits'], births=stream['new_count'],
                deletions=stream['deleted_count'], old_gs_count=stream['activities']['gs']['denominator'],
                gs_keep_fraction=activity['KEEP'], gs_shared_fraction=activity['SHARED'],
                gs_local_fraction=activity['SPLIT'],
                **stream['actual_bytes_by_class'],
                **{f'codec_{key}_seconds': value for key, value in stream['stage_seconds'].items()})


def summarize_run(run, external=False):
    name = run.name if (run / 'cfg_args').exists() else run.parent.name
    config_path = run / ('cfg_args' if (run / 'cfg_args').exists() else 'config.yaml')
    config_text = read_text(config_path)
    legacy = config_path.name == 'cfg_args'
    config = parse_namespace(config_text) if legacy else yaml.safe_load(config_text)
    log_path = run / ('train.log' if legacy else 'logging.INFO')
    log = read_text(log_path) if log_path.exists() else ''
    details_path = run / 'training_details.log'
    details = read_text(details_path) if details_path.exists() else ''
    entry = dict(experiment=name, run_path=display(run), external_reference=external,
                 config_source=display(config_path), family='s2gs_legacy' if legacy else 'recongs',
                 source_precision='json_2_decimals' if legacy else 'float32', warnings=[])
    stages, frames, starts, timers = [], [], set(), {}
    if legacy:
        planned = config['max_frames']
        step = (config.get('shared_motion_stage1_steps', 0) + config.get('shared_motion_stage2_steps', 0)
                if config.get('shared_motion') else config.get('hierarchy_stage1_steps', 0)
                + config.get('hierarchy_stage2_steps', config.get('hierarchy_gs_steps', 0))
                + config.get('hierarchy_growth_steps', 0))
        entry.update(sh_degree=config['sh_degree'], resolution=config['resolution'],
                     dataset=config['source_path'], shared_first_frame=None,
                     initial_steps=config['epochs'], subsequent_steps=step)
        metric_path = run / 'training_metrics.json'
        if metric_path.exists():
            metrics = read_json(metric_path)
            for row in metrics:
                frames.append(dict(frame=int(row['Frame index']) - 1,
                    original_frame_index=int(row['Frame index']), psnr=row['PSNR (Test)'],
                    legacy_test_loss=row['Loss (Test)'], fps=row['FPS'], gs_count=row['Num points'],
                    update_gs=float(row['Update points']), active_gates=row['Active gates'],
                    legacy_frame_seconds=row['Frame time'], legacy_frame_io_seconds=row['Frame time IO'],
                    legacy_estimated_size_mb=row['Size (MB)'], source=display(metric_path)))
            avg_path = run / 'avg_metrics.json'
            if avg_path.exists():
                entry['legacy_saved_average'] = read_json(avg_path)
        else:
            frames = parse_s2gs_progress(log)
            for row in frames:
                row['source'] = display(log_path)
            entry['source_precision'] = 'tqdm_3_significant_digits'
            entry['warnings'].append('No final JSON; completed tqdm rows are rounded, approximate metrics.')
        starts = {int(match) - 1 for match in re.findall(r'Frame (\d+):', log)}
        starts.update(row['frame'] for row in frames)
        entry['warnings'].append('Historical S2GS source/CLI is absent from current checkout; independent frame-0 training.')
        if 'KeyboardInterrupt' in log:
            entry['termination'] = 'KeyboardInterrupt'
    else:
        trainer, data, module = config['trainer'], config['data'], config['module']
        stage_rows, starts, timers = parse_recongs_log(log + '\n' + details)
        planned = data.get('extra_dataset_kwargs', {}).get('num_frames') or (max(starts) + 1 if starts else None)
        step = trainer['num_incr_steps']
        entry.update(sh_degree=module['sh_degree'], resolution=data['resolution'],
                     dataset=data['root'], shared_first_frame=config.get('ckpt_path'),
                     initial_steps=trainer['num_init_steps'], subsequent_steps=step,
                     motion_mode=module.get('motion_mode', 'legacy'))
        reports = sorted((run / 'frame_summary').glob('frame_*.json'))
        if reports:
            for path in reports:
                report = read_json(path)
                quality = (report.get('decoded_evaluation') or report.get('evaluation', {})).get('test')
                if not quality:
                    entry['warnings'].append(f'{path.name}: missing final test evaluation; excluded.')
                    continue
                row = dict(frame=report['frame'], step=report.get('final_step', step),
                           **quality, source=display(path))
                if row['step'] != step:
                    raise ValueError(f'{path}: final step differs from saved experiment budget')
                row.update({f'{key}_seconds': value for key, value in report.get('timing', {}).items()})
                storage = report.get('debug_storage') or report.get('storage')
                if storage:
                    row['debug_storage_bytes'] = storage['total_bytes']
                if report.get('stream'):
                    row.update(stream_frame(run, report))
                if report.get('selection'):
                    selection = report['selection']
                    row['gs_shared_fraction'] = selection['gs_weighted']['fractions']['SHARED']
                    row['active_split_nodes'] = selection['active_tree']['counts']['SPLIT']
                    row['terminal_shared_nodes'] = selection['terminal_regions']['counts']['SHARED']
                    row['coarsest_shared_nodes'] = selection['terminal_regions']['depth_by_state']['SHARED'].get('0', 0)
                frames.append(row)
            for path in sorted((run / 'frame_summary').glob('evaluation_*.json')):
                report = read_json(path)
                for split, metrics in report['evaluation'].items():
                    stages.append(dict(frame=report['frame'], step=report['step'], split=split,
                                       **metrics, source=display(path)))
            completed = {row['frame']: row for row in frames}
            for row in stages:
                final = completed.get(row['frame'])
                if final and row['split'] == 'test' and row['step'] == final['step']:
                    for key in ('psnr', 'ssim', 'l1'):
                        if row[key] != final[key]:
                            raise ValueError(f'{run}: final evaluation/summary mismatch for {key}')
        else:
            precise = refine_from_tensorboard(run, stage_rows)
            entry['source_precision'] = 'tensorboard_float32' if precise else 'log_4_decimals'
            for row in stage_rows:
                row['source'] = display(log_path)
                stages.append(row)
                if row['split'] == 'test' and row['step'] == (trainer['num_init_steps'] if row['frame'] == 0 else step):
                    frames.append({key: value for key, value in row.items() if key != 'split'})
            for row in frames:
                if row['frame'] in timers:
                    row['legacy_timer_frame_seconds'] = timers[row['frame']]
        if not data.get('extra_dataset_kwargs', {}).get('num_frames'):
            entry['warnings'].append('Frame count inferred from log; saved config uses dataset default.')
        hashes = re.findall(r'Git commit hash: ([a-f0-9]+)', log)
        entry['logged_git_commits'] = sorted(set(hashes))
        entry['warnings'].append('Logged Git hash alone does not identify uncommitted training source.')
        if 'Finished pipeline' in log:
            entry['termination'] = 'finished_pipeline'
        elif 'KeyboardInterrupt' in log + details:
            entry['termination'] = 'KeyboardInterrupt'
        storage_rows = [read_json(path) for path in sorted((run / 'frame_storage').glob('frame_*.json'))]
        if storage_rows:
            entry['legacy_storage'] = means(storage_rows, ['motion_bytes', 'incremental_bytes', 'total_bytes',
                'active_anchors', 'candidate_anchors', 'active_l2_parents', 'active_l3_parents', 'active_l4_parents'])
            entry['legacy_storage']['semantics'] = 'Motion NPZ and incremental GS files; not an entropy-coded stream.'
        gate_counts = defaultdict(lambda: Counter())
        for path in sorted((run / 'hierarchy_stats').glob('frame_*_gate_l*_stats.csv')):
            level = re.search(r'_gate_l(\d+)_', path.name)[1]
            for row in csv.DictReader(read_text(path).splitlines()):
                gate_counts[level]['parents'] += 1
                gate_counts[level]['eligible'] += row['eligible'] == 'True'
                gate_counts[level]['active'] += row['active'] == 'True'
        if gate_counts:
            entry['gates'] = {level: dict(counts) for level, counts in gate_counts.items()}
        if module.get('motion_mode') == 'selective':
            diagnostics = [read_json(path) for path in sorted((run / 'selective_motion').glob('frame_*.json'))]
            comparisons = [row for report in diagnostics for row in report['comparisons']]
            topology = Counter()
            for report in diagnostics:
                for event in report.get('topology_events', []):
                    for key, value in event.items():
                        if key != 'event' and isinstance(value, int):
                            topology[f'{event["event"]}.{key}'] += value
            entry['rule_diagnostics'] = dict(frames=len(diagnostics),
                all_comparisons=len(comparisons), shared_blocked=sum(row['shared_blocked'] for row in comparisons),
                rollbacks=sum(report['final_review']['rolled_back'] for report in diagnostics),
                mean_shared_regions=statistics.fmean(report['shared_motion_regions'] for report in diagnostics),
                topology_operation_counts=dict(topology),
                stage_budgets=diagnostics[0]['stage_budgets'] if diagnostics else None)
    if len({row['frame'] for row in frames}) != len(frames):
        raise ValueError(f'{run}: duplicate final-frame records')
    frames.sort(key=lambda row: row['frame'])
    dynamic = [row for row in frames if row['frame'] > 0]
    expected = set(range(1, planned)) if planned else set()
    present = {row['frame'] for row in dynamic}
    entry.update(planned_total_frames=planned, completed_dynamic_frames=len(dynamic),
                 completed_frame_indices=sorted(present), missing_dynamic_frames=sorted(expected - present),
                 started_but_incomplete_frames=sorted(starts - present - {0}),
                 status='complete' if expected and present == expected else 'partial',
                 primary_range=[dynamic[0]['frame'], dynamic[-1]['frame']] if dynamic else None,
                 mean=means(dynamic, sorted({key for row in dynamic for key, value in row.items()
                    if isinstance(value, (int, float)) and key not in {'frame', 'step', 'original_frame_index'}})),
                 all_frame_mean=means(frames, ['psnr', 'ssim', 'l1', 'legacy_test_loss']),
                 first_frame=next((row for row in frames if row['frame'] == 0), None),
                 artifacts=inventory_bytes(run))
    if dynamic:
        entry['last_frame'] = dynamic[-1]
        entry['psnr_min'] = min(dynamic, key=lambda row: row['psnr'])
        entry['psnr_max'] = max(dynamic, key=lambda row: row['psnr'])
    streamed = [row for row in dynamic if 'stream_bytes' in row]
    if streamed:
        bootstrap_path = run / 'incremental_stream' / 'bootstrap.json'
        bootstrap = read_json(bootstrap_path)
        bootstrap_bytes = bootstrap_path.stat().st_size
        if sum(row['bootstrap_bytes'] for row in streamed) != bootstrap_bytes:
            raise ValueError(f'{run}: bootstrap was not counted exactly once')
        expected_bins = {f'frame_{row["frame"]:03d}.bin' for row in streamed}
        actual_bins = {path.name for path in (run / 'incremental_stream').glob('frame_*.bin')}
        if expected_bins != actual_bins:
            raise ValueError(f'{run}: encoded bins and final frame records do not match')
        entry['stream'] = dict(total_bytes=sum(row['stream_bytes'] for row in streamed),
            bootstrap_bytes=bootstrap_bytes, frames=len(streamed),
            total_births=sum(row['births'] for row in streamed), total_deletions=sum(row['deletions'] for row in streamed),
            reference_sha256=bootstrap['reference_sha256'], reference_gs_count=bootstrap['first_frame_gs_count'],
            validated='All bin lengths, sidecars, class sums, final metrics and one bootstrap; no independent replay.',
            semantics='Actual incremental containers plus bootstrap once, excluding external frame-0 model and debug files.')
    image_paths = sorted((run / 'image/frames').glob('*/cam00_gt.png'))
    if image_paths:
        from PIL import Image
        with Image.open(image_paths[0]) as image:
            entry['saved_test_image_size'] = list(image.size)
    camera_path = run / 'cameras.json'
    if camera_path.exists():
        cameras = read_json(camera_path)
        entry['camera_metadata_image_sizes'] = [list(size) for size in sorted({
            (camera['width'], camera['height']) for camera in cameras})]
        entry['test_camera'] = 'cam00' if image_paths else 'test_indices=' + str(config.get('test_indices'))
    for row in frames + stages:
        row['experiment'] = name
    entry['config'] = config
    return entry, frames, stages, config_text


def config_diff(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(set(left) | set(right)):
            rows.extend(config_diff(left.get(key), right.get(key), f'{prefix}.{key}' if prefix else key))
        return rows
    return [] if left == right else [{'key': prefix, 'left': left, 'right': right}]


def paired_comparisons(entries, rows):
    by_run = defaultdict(dict)
    for row in rows:
        if row['frame'] > 0:
            by_run[row['experiment']][row['frame']] = row
    references = [entry for entry in entries if entry.get('external_reference')]
    result = []
    windows = [(1, 5), (1, 20), (1, 27), (21, 27), (1, 299), (1, 50),
               (51, 100), (101, 150), (151, 200), (201, 250), (251, 299)]
    for reference in references:
        for entry in entries:
            if entry['family'] != 'recongs' or entry['experiment'] == reference['experiment']:
                continue
            for lo, hi in windows:
                indices = list(range(lo, hi + 1))
                current, baseline = by_run[entry['experiment']], by_run[reference['experiment']]
                if not all(frame in current and frame in baseline for frame in indices):
                    continue
                a, b = [current[i] for i in indices], [baseline[i] for i in indices]
                result.append(dict(experiment=entry['experiment'], reference=reference['experiment'],
                    start_frame=lo, end_frame=hi, n=len(indices),
                    psnr=statistics.fmean(row['psnr'] for row in a),
                    reference_psnr=statistics.fmean(row['psnr'] for row in b),
                    delta_psnr=statistics.fmean(x['psnr'] - y['psnr'] for x, y in zip(a, b)),
                    ssim=statistics.fmean(row['ssim'] for row in a),
                    reference_ssim=statistics.fmean(row['ssim'] for row in b),
                    psnr_wins=sum(x['psnr'] > y['psnr'] for x, y in zip(a, b))))
    return result


def write_csv(path, rows, first=()):
    keys = list(first) + sorted({key for row in rows for key in row} - set(first))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot_records(destination, entries, frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    series = defaultdict(list)
    for row in frames:
        if row['frame'] > 0:
            series[row['experiment']].append(row)
    labels = {'flame_steak_fullres': 'ReCon-GS reference (100+100)',
              'flame_steak_gradhet4_recongs_first_checkpoint': 'GradHet 4 levels (100+100)',
              'flame_steak_selective_motion_21': 'Rule selection (150+50)',
              'fs_learned_selective_300': 'Learned selection (400+50, partial)',
              'fs_residual_rate0_300_steps200': 'Residual, rate=0 (150+50)',
              'fs_residual_rate0001_300_steps200': 'Residual, rate=1e-4 (150+50)',
              'fs_motion_only_300_steps200': 'Motion only (200+0, decay=0.6)'}
    colors = dict(zip(labels, ['#222222', '#9675ba', '#ce7446', '#c9507c', '#ddab32', '#428c7d', '#3776b6']))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout='constrained')
    for name, label in labels.items():
        rows = series.get(name, [])
        if not rows:
            continue
        x, y = [row['frame'] for row in rows], [row['psnr'] for row in rows]
        axes[0, 0].plot(x, y, lw=1.25, label=label, color=colors[name], alpha=.9)
        early = [row for row in rows if row['frame'] <= 27]
        axes[0, 1].plot([row['frame'] for row in early], [row['psnr'] for row in early],
                        lw=1.5, color=colors[name])
        if rows and 'stream_bytes' in rows[0]:
            axes[1, 0].plot(x, [row['stream_bytes'] / 1000 for row in rows],
                            label=label, color=colors[name], lw=1.3)
    axes[0, 0].set(title='Final test PSNR; shared frame 0 excluded', ylabel='PSNR (dB)')
    axes[0, 1].set(title='First 27 subsequent frames', ylabel='PSNR (dB)')
    axes[1, 0].set(title='Actual incremental bytes; frame-0 model excluded', ylabel='kB / frame (1000 bytes)')
    axes[1, 1].set(title='PSNR difference from ReCon-GS; 20-frame rolling mean', ylabel='PSNR difference (dB)')
    ref = {row['frame']: row for row in series.get('flame_steak_fullres', [])}
    for name in labels:
        rows = series.get(name, [])
        if not ref or name == 'flame_steak_fullres' or len(rows) < 299:
            continue
        delta = [row['psnr'] - ref[row['frame']]['psnr'] for row in rows]
        smoothed = [statistics.fmean(delta[max(0, i - 19):i + 1]) for i in range(len(delta))]
        axes[1, 1].plot([row['frame'] for row in rows], smoothed, color=colors[name], lw=1.8)
    axes[1, 1].axhline(0, color='#555555', lw=.8, ls='--')
    for ax in axes.flat:
        ax.set_xlabel('Zero-based frame index')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc='lower left')
    axes[1, 0].legend(fontsize=8)
    fig.savefig(destination / 'recongs_results.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
    for entry in entries:
        if entry['family'] != 's2gs_legacy':
            continue
        rows = series[entry['experiment']]
        ax.plot([row['frame'] for row in rows], [row['psnr'] for row in rows],
                label=entry['experiment'] + (' (partial, rounded log)' if entry['status'] != 'complete' else ''), lw=1.3)
    ax.set(title='Historical S2GS branch: independent first frames and different budgets',
           xlabel='Zero-based frame index; frame 0 excluded', ylabel='Test PSNR (dB)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.savefig(destination / 's2gs_history.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=REPO / 'output')
    parser.add_argument('--reference', type=Path, action='append', default=[])
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    roots = sorted(path for path in args.output_root.iterdir() if path.is_dir())
    entries, frames, stages, snapshots = [], [], [], {}
    runs = []
    for root in roots:
        candidates = ([root] if any((root / name).exists() for name in ('config.yaml', 'cfg_args')) else
                      [path for path in sorted(root.iterdir()) if path.is_dir() and (path / 'config.yaml').exists()])
        if not candidates:
            entries.append(dict(experiment=root.name, run_path=display(root), family='unknown',
                status='empty' if not any(root.iterdir()) else 'unrecognized', completed_dynamic_frames=0,
                warnings=['No saved config or usable run; no result claimed.']))
        runs.extend((path, False) for path in candidates)
    runs.extend((path, True) for path in args.reference)
    destination = args.destination.resolve()
    for source in [args.output_root] + args.reference:
        if destination == source.resolve() or source.resolve() in destination.parents:
            raise ValueError('Destination must be outside every historical source directory')
    for run, external in runs:
        entry, final_rows, eval_rows, snapshot = summarize_run(run, external)
        if entry['experiment'] in snapshots:
            raise ValueError(f'Duplicate experiment name: {entry["experiment"]}; select distinct source runs.')
        entries.append(entry)
        frames.extend(final_rows)
        stages.extend(eval_rows)
        snapshots[entry['experiment']] = snapshot
    comparisons = paired_comparisons(entries, frames)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_dir = destination / 'configs'
    snapshot_dir.mkdir(exist_ok=True)
    for entry in entries:
        name = entry['experiment']
        if name not in snapshots:
            continue
        suffix = '.json' if entry['family'] == 's2gs_legacy' else '.yaml'
        path = snapshot_dir / (name + suffix)
        path.write_text(json.dumps(entry['config'], indent=2, ensure_ascii=False) + '\n'
                        if suffix == '.json' else snapshots[name])
        entry['config_record'] = str(path.relative_to(destination))
    by_name = {entry['experiment']: entry for entry in entries}
    differences = []
    base = 'fs_residual_rate0_300_steps200'
    for target in ('fs_residual_rate0001_300_steps200', 'fs_motion_only_300_steps200'):
        if base in by_name and target in by_name:
            differences.append(dict(left=base, right=target,
                changes=config_diff(by_name[base]['config'], by_name[target]['config'])))
    git_head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    report = dict(schema_version=1, generated_at=datetime.now(timezone.utc).isoformat(),
                  statistics_base_commit=git_head, local_experiment_directories=len(roots),
                  external_references=len(args.reference),
                  primary_semantics='Arithmetic mean of final test metrics on completed frames, excluding frame 0.',
                  entries=entries, comparisons=comparisons, config_differences=differences,
                  source_manifest=SOURCE_FILES)
    (destination / 'inventory.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    write_csv(destination / 'frames.csv', frames, ('experiment', 'frame', 'psnr', 'ssim', 'l1'))
    write_csv(destination / 'evaluations.csv', stages, ('experiment', 'frame', 'step', 'split', 'psnr', 'ssim'))
    write_csv(destination / 'comparisons.csv', comparisons, ('experiment', 'reference', 'start_frame', 'end_frame', 'n'))
    summaries = []
    for entry in entries:
        summaries.append(dict(experiment=entry['experiment'], family=entry['family'], status=entry['status'],
            completed_dynamic_frames=entry['completed_dynamic_frames'], planned_total_frames=entry.get('planned_total_frames'),
            **entry.get('mean', {}), artifact_bytes=entry.get('artifacts', {}).get('total_bytes'),
            source_precision=entry.get('source_precision'), run_path=entry['run_path']))
    write_csv(destination / 'summary.csv', summaries, ('experiment', 'family', 'status', 'completed_dynamic_frames', 'psnr', 'ssim'))
    if args.plots:
        plot_records(destination, entries, frames)
    print(json.dumps([dict(experiment=entry['experiment'], status=entry['status'],
        frames=entry['completed_dynamic_frames'],
        psnr=entry.get('mean', {}).get('psnr'), ssim=entry.get('mean', {}).get('ssim'),
        mean_stream_bytes=entry.get('mean', {}).get('stream_bytes')) for entry in entries], indent=2))


if __name__ == '__main__':
    main()
