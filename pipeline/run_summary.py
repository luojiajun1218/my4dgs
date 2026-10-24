"""End-of-training statistics from completed frame reports and packet files."""

import json
import math
import time
from pathlib import Path
from statistics import mean

from .timing import summarize_frame_timings


FPS_REFERENCE_NOTICE = 'FPS仅供参考'


def time_render(render, synchronize, warmups=10, repeats=50, clock=time.perf_counter):
    """Time full renders after warmup, excluding queued work and metrics."""
    for _ in range(warmups):
        render()
    synchronize()
    started = clock()
    for _ in range(repeats):
        render()
    synchronize()
    seconds = clock() - started
    return dict(warmup_renders=warmups, timed_renders=repeats,
                render_seconds=seconds, render_fps=repeats / seconds,
                fps_notice=FPS_REFERENCE_NOTICE)


def build_run_summary(run, expected_frames, *, initial_report=None,
                      reference_path=None, elapsed_seconds=None):
    run = Path(run)
    rows = [json.loads(path.read_text()) for path in
            sorted((run / 'frame_summary').glob('frame_*.json'))]
    rows = [row for row in rows if 0 < row['frame'] < expected_frames]
    ids = [row['frame'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Run summary needs unique frame IDs')
    rows.sort(key=lambda row: row['frame'])
    if initial_report is None and (run / 'reference_evaluation.json').is_file():
        initial_report = json.loads((run / 'reference_evaluation.json').read_text())
    evaluated = ([initial_report] if initial_report is not None else []) + rows
    quality = {}
    for group in sorted({name for row in evaluated for name in row.get('evaluation', {})}):
        metrics, counts = {}, {}
        for name in ('psnr', 'ssim', 'lpips'):
            values = [row['evaluation'][group][name] for row in evaluated
                      if name in row.get('evaluation', {}).get(group, {})]
            values = [value for value in values if value is not None and math.isfinite(value)]
            if values:
                metrics[name], counts[name] = mean(values), len(values)
        quality[group] = dict(metrics=metrics, metric_frame_counts=counts)

    reference = Path(reference_path) if reference_path else (
        run / 'frame_storage/rigid_tree/reference_frame0.npz')
    reference_bytes = reference.stat().st_size if reference.is_file() else None
    has_frame0 = reference_bytes is not None and reference.name == 'reference_frame0.npz'
    payloads, updates = [], []
    for row in rows:
        manifest_path = run / row.get('frame_manifest',
            f'frame_storage/rigid_tree/frame_{row["frame"]:03d}.json')
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        paths = [manifest_path.parent / manifest[key] for key in
                 ('motion', 'birth', 'opacity_update', 'feature_dc_update') if manifest.get(key)]
        if not paths or not all(path.is_file() for path in paths):
            continue
        payload = sum(path.stat().st_size for path in paths)
        payloads.append(payload)
        updates.append(payload + manifest_path.stat().st_size)
    storage_complete = len(updates) == len(rows)
    complete = has_frame0 and set(ids) == set(range(1, expected_frames))
    total_bytes = (reference_bytes + sum(updates)
                   if complete and storage_complete else None)
    timings = summarize_frame_timings(rows)
    renders = [row['render_timing'] for row in evaluated
               if row.get('render_timing') and row['render_timing'].get('render_seconds', 0) > 0]
    render_seconds = sum(row['render_seconds'] for row in renders)
    render_count = sum(row['timed_renders'] for row in renders)
    initial_count = (initial_report or {}).get('total_count')
    if initial_count is None and rows:
        initial_count = (rows[0].get('initial_reference') or {}).get('retained_count')
    return dict(schema=1, run=str(run), expected_frames=expected_frames,
        recorded_frames=len(rows) + int(has_frame0), sequence_complete=complete,
        dynamic_frames=len(rows), quality=quality,
        storage=dict(measured_dynamic_frames=len(updates),
            payload_kb_mean=mean(payloads) / 1000 if storage_complete and payloads else None,
            complete_update_kb_mean=mean(updates) / 1000 if storage_complete and updates else None,
            reference_mb=reference_bytes / 1e6 if reference_bytes is not None else None,
            total_mb=total_bytes / 1e6 if total_bytes is not None else None,
            reference_amortized_kb_per_frame=(total_bytes / expected_frames / 1000
                                             if total_bytes is not None else None)),
        initial_gaussians=initial_count,
        final_gaussians=rows[-1].get('total_count') if rows else initial_count,
        model_training_seconds_mean=timings['model_training']['mean_seconds']['training_seconds'],
        frame_wall_seconds_mean=timings['all_dynamic']['mean_seconds']['frame_wall_seconds'],
        model_training_measured_frames=timings['model_training']['measured_frame_count'],
        fit_elapsed_seconds=elapsed_seconds,
        pure_render_fps=render_count / render_seconds if render_seconds else None,
        fps_notice=FPS_REFERENCE_NOTICE,
        render_measured_frames=len(renders), render_seconds=render_seconds,
        timed_renders=render_count,
        render_scope='Validation-time synchronized forward+clamp; excludes decode, upload, '
                     'loading, warmups and quality metrics; GPU exclusivity is not audited.',
        storage_units='KB=1000 bytes; MB=1000000 bytes')


def format_run_summary(summary):
    def number(value, digits=2):
        return f'{value:.{digits}f}' if value is not None else 'N/A'

    expected = summary['expected_frames']
    lines = ['================ Training summary ================',
             f'Frames recorded: {summary["recorded_frames"]}/{expected} | '
             f'dynamic: {summary["dynamic_frames"]} | '
             f'{"complete" if summary["sequence_complete"] else "partial coverage"}']
    for group, row in summary['quality'].items():
        parts = []
        for name, value in row['metrics'].items():
            unit, digits = (' dB', 4) if name == 'psnr' else ('', 6)
            parts.append(f'{name.upper()}={value:.{digits}f}{unit} '
                         f'({row["metric_frame_counts"][name]}/{expected} frames)')
        lines.append(f'{group} mean: ' + ' | '.join(parts))
    storage = summary['storage']
    lines.extend([
        'Storage (dynamic payload+index, excludes reference): '
        f'{number(storage["complete_update_kb_mean"])} KB/frame '
        f'({storage["measured_dynamic_frames"]}/{summary["dynamic_frames"]} dynamic frames)',
        'Storage (including reference, amortized): '
        f'{number(storage["reference_amortized_kb_per_frame"])} KB/frame | '
        f'total={number(storage["total_mb"])} MB | reference={number(storage["reference_mb"])} MB',
        f'GS: {summary["initial_gaussians"]} -> {summary["final_gaussians"]}',
        f'Time: model training={number(summary["model_training_seconds_mean"])} s/dynamic frame | '
        f'frame wall={number(summary["frame_wall_seconds_mean"])} s/dynamic frame'])
    if summary['fit_elapsed_seconds'] is not None:
        seconds = round(summary['fit_elapsed_seconds'])
        lines.append(f'This run (setup, training and evaluation): '
                     f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}')
    lines.append('Pure render (validation, excludes decode/upload): '
                 f'{number(summary["pure_render_fps"])} FPS '
                 f'({summary["render_measured_frames"]}/{expected} frames) | '
                 f'{summary.get("fps_notice", FPS_REFERENCE_NOTICE)}')
    lines.append(f'Summary saved: {Path(summary["run"]) / "run_summary.json"}')
    lines.append('==================================================')
    return lines
