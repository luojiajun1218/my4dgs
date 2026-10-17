"""Benchmark full rigid-tree rendering and sequential decoded playback.

Materialized FPS times Module.forward plus clamp at cam00. Playback FPS times
each motion/birth file read, checked decode, host-to-GPU materialization, and
the same Module.forward plus clamp. Dataset loading, reference loading,
integrity preflight, quality metrics, and warmups are outside timed scopes.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW_FIELDS = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')
CAPTURE_INDEX = dict(xyz=1, sh_dc=2, sh_rest=3, logscale=4,
                     rotation=5, opacity=6)


def _process_state(pid: int) -> str | None:
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except (FileNotFoundError, PermissionError, IndexError):
        return None


def _proc_gpu_processes() -> tuple[list[dict], int]:
    """Fallback when NVML is unavailable; count inaccessible FD tables."""
    records, inaccessible = [], 0
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal() or int(process.name) == os.getpid():
            continue
        pid = int(process.name)
        fds = process / 'fd'
        try:
            entries = list(fds.iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            inaccessible += 1
            continue
        devices = set()
        for fd in entries:
            try:
                target = os.readlink(fd)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if target.startswith('/dev/nvidia'):
                devices.add(target)
        if devices:
            records.append(dict(pid=pid, state=_process_state(pid), source='proc_fd',
                                devices=sorted(devices),
                                compute_context='/dev/nvidia-uvm' in devices))
    return records, inaccessible


def check_gpu_exclusivity() -> dict:
    """Reject runnable foreign GPU users; state uncertainty remains explicit."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
             '--format=csv,noheader,nounits'], capture_output=True,
            text=True, check=False, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        result = None
        nvml_problem = str(error)
    else:
        nvml_problem = ((result.stderr or result.stdout).strip()
                        if result.returncode else None)
    if result is not None and result.returncode == 0:
        records = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            pid_text = line.split(',', 1)[0].strip()
            if not pid_text.isdecimal():
                raise RuntimeError(f'Unparseable nvidia-smi process row: {line}')
            pid = int(pid_text)
            if pid != os.getpid():
                records.append(dict(pid=pid, state=_process_state(pid),
                                    source='nvidia_smi', row=line.strip()))
        method, inaccessible = 'nvidia_smi', 0
    else:
        records, inaccessible = _proc_gpu_processes()
        method = 'proc_fd_fallback'
        warnings.warn('nvidia-smi/NVML unavailable; checking open /dev/nvidia '
                      f'file descriptors instead. NVML error: {nvml_problem}',
                      RuntimeWarning, stacklevel=2)
    running = [row for row in records if row['state'] not in ('T', 't', 'Z') and
               (method == 'nvidia_smi' or row['compute_context'])]
    if running:
        raise RuntimeError(f'Runnable foreign GPU contexts prevent reliable FPS: {running}')
    if records:
        warnings.warn(f'Other GPU device users (graphics may share the GPU): {records}',
                      RuntimeWarning, stacklevel=2)
    if inaccessible:
        warnings.warn(f'Cannot inspect {inaccessible} process FD tables; '
                      'GPU exclusivity is not fully verified.',
                      RuntimeWarning, stacklevel=2)
    return dict(method=method, foreign_contexts=records,
                inaccessible_process_fd_tables=inaccessible,
                verified=(method == 'nvidia_smi'),
                nvml_error=nvml_problem)


def _load_config(run: Path):
    from omegaconf import OmegaConf
    from libgs.pipeline.loader import load_pipeline
    from libgs.utils.config import to_structured

    Config, _ = load_pipeline('my4dgs')
    from importlib import import_module
    normalize = import_module('pipeline.model.rigid_residual_tree').saved_config_with_default_motion_lr
    saved = OmegaConf.to_container(OmegaConf.load(run / 'config.yaml'), resolve=True)
    saved['module']['rigid_tree'] = normalize(saved['module']['rigid_tree'])
    saved['module'].pop('old_opacity', None)
    config, extra = to_structured(OmegaConf.create(saved), Config)
    if extra:
        raise ValueError(f'Unrecognized run configuration: {extra}')
    return config


def _cam00(datamodule, frame: int, device):
    datamodule.current_frame = frame
    dataset = datamodule.get_test_dataset()
    if dataset is None:
        raise ValueError(f'Frame {frame} has no test dataset')
    names = [Path(str(item.path)).name for item in dataset.items]
    indices = [i for i, name in enumerate(names) if name == 'cam00.png']
    if len(indices) != 1:
        raise ValueError(f'Frame {frame} needs exactly one cam00.png, found {indices}')
    return dataset[indices[0]].to(device), len(dataset)


def _make_module(config, device):
    """Use the active Module.forward path without allocating a train optimizer."""
    import torch
    from libgs.renderer import Renderer

    data = importlib.import_module('pipeline.data')
    module_source = importlib.import_module('pipeline.module')
    gaussian_source = importlib.import_module('pipeline.model.gaussian')
    import libgs.data.dataset.base as dataset_base

    # The direct single-view load below has no DataLoader worker.
    dataset_base.warning = lambda: None
    config.data.shuffle = False
    datamodule = data.DataModule(config.data)
    datamodule.setup()
    module = module_source.Module(config.module, datamodule=datamodule)
    module.trainer = SimpleNamespace(
        device=device, global_step=(config.module.rigid_tree.motion_steps +
                                    config.module.rigid_tree.birth_steps))
    module.gaussians = gaussian_source.GaussianModel(config.module.sh_degree)
    module.gaussians.active_sh_degree = config.module.sh_degree
    module.renderer = Renderer(config.module.renderer, module.gaussians)
    color = 1. if config.data.white_background else 0.
    module.register_buffer('background', torch.full((3,), color, device=device))
    module.frame_prepared = False
    module.motion_committed = False
    module.frame_finalized = True
    module.eval()
    return module, datamodule


def _install_raw(module, raw, device):
    import torch

    model = module.gaussians
    source = importlib.import_module('pipeline.model.birth_quantization')
    for name, attribute in zip(source.RAW_FIELDS, source.RAW_ATTRIBUTES):
        value = raw[name].detach().to(device=device, dtype=torch.float32)
        setattr(model, attribute, torch.nn.Parameter(value, requires_grad=False))
    model._render_cache = {}
    model.rigid_xyz = None
    model.rigid_rotation = None


def _capture_raw(checkpoint: Path, frame: int, total_steps: int):
    import torch

    packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (packet.get('rigid_tree_version') != 1 or
            packet.get('current_frame') != frame or
            packet.get('global_step') != total_steps or
            not packet.get('rigid_tree_frame', {}).get('finalized')):
        raise ValueError('Benchmark requires a finalized rigid-tree checkpoint')
    capture = packet['gaussians']
    raw = {name: capture[index].detach().cpu().float().contiguous()
           for name, index in CAPTURE_INDEX.items()}
    return raw, int(capture[0])


def _load_reference(path: Path, region_count: int | None):
    import torch
    codec = importlib.import_module('pipeline.model.rigid_tree_codec')

    with np.load(path, allow_pickle=False) as packet:
        raw = {name: torch.from_numpy(packet[name].copy()) for name in RAW_FIELDS}
        if region_count is not None and int(packet['region_count']) != region_count:
            raise ValueError('Stream reference region count differs from run config')
        if int(packet['region_count']) != len(packet['parents']):
            raise ValueError('Stream reference region count differs from stored tree')
        state = codec.DecodedState.from_reference(packet)
    codec._validate_state(state)
    return state, raw


def _decode_frame(stream: Path, frame: int, state, raw):
    from scripts.replay_rigid_tree import decode_frame
    return decode_frame(stream, frame, state, raw)


def _render_checked(module, view):
    import torch

    rendered = module.forward(view)['render']
    if rendered.shape != view.image.shape or not bool(torch.isfinite(rendered).all()):
        raise ValueError('Renderer produced wrong shape or nonfinite pixels')
    return rendered.clamp(0., 1.)


def _quality(module, view, expected, tolerance):
    import torch
    from libgs.metric import psnr, ssim

    if (view.image.shape != (3, 1199, 1600) or
            not bool(torch.isfinite(view.image).all()) or
            bool((view.image < -1e-6).any()) or
            bool((view.image > 1 + 1e-6).any())):
        raise ValueError('cam00 input must be finite RGB in [0,1] at 1600x1199')
    image = _render_checked(module, view)
    measured = dict(psnr=float(psnr(image, view.image)),
                    ssim=float(ssim(image, view.image)),
                    l1=float(torch.nn.functional.l1_loss(image, view.image)))
    difference = None
    if expected is not None:
        difference = {name: measured[name] - expected[name]
                      for name in measured}
        if (abs(difference['psnr']) > tolerance['psnr'] or
                abs(difference['ssim']) > tolerance['ssim'] or
                abs(difference['l1']) > tolerance['l1']):
            raise ValueError(f'Materialized cam00 quality differs from saved evaluation: {difference}')
    return measured, difference, image


def _time_materialized(module, view, warmups, timed, rounds):
    import torch

    samples = []
    for repeat in range(rounds):
        gpu_status = check_gpu_exclusivity()
        with torch.no_grad():
            for _ in range(warmups):
                module.forward(view)['render'].clamp(0., 1.)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(timed):
                module.forward(view)['render'].clamp(0., 1.)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
        check_gpu_exclusivity()
        samples.append(dict(round=repeat, seconds=seconds, fps=timed / seconds,
                            gpu_check=gpu_status))
    return dict(samples=samples,
                fps=rounds * timed / sum(row['seconds'] for row in samples))


def _time_playback(module, view, stream, reference_state, reference_raw,
                   end_frame, rounds, device):
    import torch

    passes = []
    frame_count = end_frame - reference_state.frame_index
    if frame_count < 1:
        raise ValueError('Stream target must follow the reference frame')
    for repeat in range(rounds):
        gpu_status = check_gpu_exclusivity()
        state = reference_state
        raw = {name: value.clone() for name, value in reference_raw.items()}
        rows = []
        final_image = None
        with torch.no_grad():
            for frame in range(reference_state.frame_index + 1, end_frame + 1):
                torch.cuda.synchronize()
                started = time.perf_counter()
                state, raw = _decode_frame(stream, frame, state, raw)
                decoded = time.perf_counter()
                _install_raw(module, raw, device)
                final_image = module.forward(view)['render'].clamp(0., 1.)
                torch.cuda.synchronize()
                completed = time.perf_counter()
                rows.append(dict(frame=frame, file_decode_seconds=decoded - started,
                                 upload_forward_clamp_seconds=completed - decoded,
                                 end_to_end_seconds=completed - started,
                                 gs=len(raw['xyz'])))
        check_gpu_exclusivity()
        seconds = sum(row['end_to_end_seconds'] for row in rows)
        passes.append(dict(round=repeat, frames=rows, seconds=seconds,
                           fps=frame_count / seconds, gpu_check=gpu_status))
    return dict(passes=passes,
                fps=frame_count * rounds / sum(row['seconds'] for row in passes),
                final_image=final_image, final_raw=raw)


def summarize_render_frames(rows):
    """Aggregate synchronized full-render samples; decoding never enters FPS."""
    if not rows:
        raise ValueError('At least one rendered frame is required')
    frame_ids = [int(row['frame']) for row in rows]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError('Render frame IDs must be unique')
    renders = sum(int(row['timed_renders']) for row in rows)
    seconds = sum(float(row['render_seconds']) for row in rows)
    if renders < 1 or seconds <= 0:
        raise ValueError('Render counts and seconds must be positive')
    fps_values = sorted(int(row['timed_renders']) /
                        float(row['render_seconds']) for row in rows)
    rank = .05 * (len(fps_values) - 1)
    low, high = int(rank), min(len(fps_values) - 1, int(rank) + 1)
    p05 = fps_values[low] + (fps_values[high] - fps_values[low]) * (rank - low)
    psnrs = [float(row['quality']['psnr']) for row in rows]
    result = dict(first_frame=min(frame_ids), last_frame=max(frame_ids),
        frame_count=len(rows), total_timed_renders=renders,
        total_render_seconds=seconds, aggregate_fps=renders / seconds,
        frame_fps_min=fps_values[0], frame_fps_p05=p05,
        frame_fps_median=statistics.median(fps_values),
        frame_fps_max=fps_values[-1],
        psnr_mean=sum(psnrs) / len(psnrs), psnr_min=min(psnrs),
        psnr_max=max(psnrs))
    for metric in ('ssim', 'l1'):
        if all(metric in row['quality'] for row in rows):
            values = [float(row['quality'][metric]) for row in rows]
            result[f'{metric}_mean'] = sum(values) / len(values)
            result[f'{metric}_min'] = min(values)
            result[f'{metric}_max'] = max(values)
    return result


def _summarize_storage(rows):
    """Keep legacy payload metrics and separately count necessary JSON indices.

    Historical rows without measured index sizes retain their payload metrics,
    but complete-update metrics stay unknown rather than inventing zero bytes.
    """
    dynamic = [row for row in rows if int(row['frame']) > 0]
    sizes = sorted(int(row['encoded_bytes']) for row in dynamic)
    motion = sum(int(row['motion_bytes']) for row in dynamic)
    birth = sum(int(row['birth_bytes']) for row in dynamic)
    opacity = sum(int(row.get('opacity_update_bytes', 0)) for row in dynamic)
    for row in dynamic:
        payload = (int(row['motion_bytes']) + int(row['birth_bytes']) +
                   int(row.get('opacity_update_bytes', 0)))
        if int(row['encoded_bytes']) != payload or int(row.get('payload_bytes', payload)) != payload:
            raise ValueError('Dynamic payload bytes must equal actual motion, birth and optional opacity bytes')

    def statistics_for(values):
        if not values:
            return dict(total=0, mean=None, p95=None, max=None)
        values = sorted(values)
        rank = .95 * (len(values) - 1)
        low, high = int(rank), min(len(values) - 1, int(rank) + 1)
        return dict(total=sum(values), mean=sum(values) / len(values),
                    p95=values[low] + (values[high] - values[low]) * (rank - low),
                    max=values[-1])

    payload_stats = statistics_for(sizes)
    result = dict(dynamic_frame_count=len(sizes), motion_bytes_total=motion,
                  birth_bytes_total=birth,
                  opacity_update_bytes_total=opacity,
                  manifest_index_measured=all('manifest_index_bytes' in row for row in dynamic),
                  encoded_bytes_scope='payload only: motion, birth and optional old-opacity packet files',
                  complete_update_bytes_scope='payload plus necessary per-frame manifest JSON file')
    for name, value in payload_stats.items():
        result[f'encoded_bytes_{name}'] = value  # Preserve legacy payload names.
        result[f'payload_bytes_{name}'] = value
    if result['manifest_index_measured']:
        indices, complete = [], []
        for row in dynamic:
            index_bytes = int(row['manifest_index_bytes'])
            if index_bytes < 0:
                raise ValueError('Manifest index bytes must be nonnegative')
            complete_bytes = int(row['encoded_bytes']) + index_bytes
            if int(row.get('complete_update_bytes', complete_bytes)) != complete_bytes:
                raise ValueError('Complete update bytes must include payload and manifest index')
            indices.append(index_bytes)
            complete.append(complete_bytes)
        index_stats, complete_stats = statistics_for(indices), statistics_for(complete)
    else:
        index_stats = complete_stats = dict(total=None, mean=None, p95=None, max=None)
    for name, value in index_stats.items():
        result[f'manifest_index_bytes_{name}'] = value
    for name, value in complete_stats.items():
        result[f'complete_update_bytes_{name}'] = value
    return result


def _summarize_sequence_storage(rows, reference_bytes):
    """Separate payload-compatible totals from complete exported sequence size."""
    dynamic = _summarize_storage(rows)
    payload_total = int(reference_bytes) + dynamic['payload_bytes_total']
    complete_total = (None if dynamic['complete_update_bytes_total'] is None else
                      int(reference_bytes) + dynamic['complete_update_bytes_total'])
    return dict(bootstrap_reference_bytes=int(reference_bytes), dynamic_packets=dynamic,
                manifest_index_bytes_total=dynamic['manifest_index_bytes_total'],
                complete_update_bytes_total=dynamic['complete_update_bytes_total'],
                total_reference_and_dynamic_bytes=payload_total,  # Legacy payload scope.
                total_reference_and_dynamic_payload_bytes=payload_total,
                total_reference_and_complete_update_bytes=complete_total,
                accounting_version=2,
                scopes=dict(total_reference_and_dynamic_bytes='legacy: bootstrap reference plus dynamic payload only',
                    total_reference_and_complete_update_bytes='bootstrap reference plus every dynamic payload and manifest index'))


def _sequence_groups(rows, segment_size):
    if segment_size < 1:
        raise ValueError('segment_size must be positive')
    frames = [int(row['frame']) for row in rows]
    if frames != list(range(len(rows))):
        raise ValueError('Complete sequence must contain ordered frames 0..N-1')
    dynamic = rows[1:]
    def group(part):
        summary = summarize_render_frames(part)
        summary['storage'] = _summarize_storage(part)
        return summary

    all_segments = [group(rows[start:start + segment_size])
                    for start in range(0, len(rows), segment_size)]
    dynamic_segments = [group(
        dynamic[start:start + segment_size])
        for start in range(0, len(dynamic), segment_size)]
    return dict(bootstrap_frame0=group(rows[:1]),
                all_frames=group(rows),
                dynamic_frames=group(dynamic),
                all_segments=all_segments,
                dynamic_segments=dynamic_segments)


def _validate_stream_manifest(stream, frame, decoded, raw, reference):
    reference_digest = importlib.import_module(
        'pipeline.model.birth_quantization').reference_digest
    manifest_path = stream / f'frame_{frame:03d}.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('frame') != frame or
            manifest.get('decoded_state_sha256') != decoded.digest().hex() or
            manifest.get('raw_state_sha256') != reference_digest(raw) or
            manifest.get('total_count') != len(raw['xyz']) or
            manifest.get('reference') != str(reference)):
        raise ValueError(f'Frame {frame} decoded state differs from manifest')
    return manifest


def _actual_frame_storage(stream, frame, manifest, summary, reference_bytes,
                          relative_reference):
    """Stat packet files and reject stale or misleading saved size metadata."""
    motion_name = f'motion_{frame:03d}.bin'
    birth_name = f'birth_{frame:03d}.bin'
    opacity_name = f'opacity_{frame:03d}.bin'
    if (manifest.get('motion') != motion_name or manifest.get('birth') not in (None, birth_name) or
            manifest.get('opacity_update') not in (None, opacity_name)):
        raise ValueError(f'Frame {frame} manifest packet names differ from expected files')
    motion_bytes = (stream / motion_name).stat().st_size
    birth_bytes = (stream / birth_name).stat().st_size if manifest['birth'] else 0
    opacity_bytes = (stream / opacity_name).stat().st_size if manifest.get('opacity_update') else 0
    if not manifest.get('opacity_update') and (stream / opacity_name).exists():
        raise ValueError('Opacity packet is not declared by frame manifest')
    manifest_index_bytes = (stream / f'frame_{frame:03d}.json').stat().st_size
    saved_motion = summary['motion']['bytes']
    saved_birth = summary['birth']['archive_bytes'] if summary.get('birth') else 0
    saved_opacity = summary['opacity_update']['archive_bytes'] if summary.get('opacity_update') else 0
    payload = motion_bytes + birth_bytes + opacity_bytes
    if (saved_motion != motion_bytes or saved_birth != birth_bytes or saved_opacity != opacity_bytes or
            summary['encoded_bytes'] != payload or
            summary['shared_reference_bytes'] != reference_bytes or
            summary['decoder_reference'] != relative_reference or
            (summary.get('birth') is None) != (manifest['birth'] is None)):
        raise ValueError(f'Frame {frame} saved storage summary differs from actual files')
    result = dict(motion_bytes=motion_bytes, birth_bytes=birth_bytes,
                encoded_bytes=payload,
                payload_bytes=payload,
                manifest_index_bytes=manifest_index_bytes,
                complete_update_bytes=payload + manifest_index_bytes)
    if 'opacity_update' in manifest:
        result['opacity_update_bytes'] = opacity_bytes
    return result


def _time_sequence_render(module, view, warmups, timed):
    import torch

    with torch.no_grad():
        for _ in range(warmups):
            module.forward(view)['render'].clamp(0., 1.)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(timed):
            module.forward(view)['render'].clamp(0., 1.)
        torch.cuda.synchronize()
    return time.perf_counter() - started


def _sequence_quality(module, view, expected, tolerance, cache_enabled):
    import torch

    module.gaussians.render_attribute_cache = False
    with torch.no_grad():
        uncached = _render_checked(module, view)
    module.gaussians.render_attribute_cache = cache_enabled
    module.gaussians._render_cache = {}
    with torch.no_grad():
        quality, delta, materialized = _quality(module, view, expected, tolerance)
    difference = float((materialized - uncached).abs().max())
    if cache_enabled and not torch.equal(materialized, uncached):
        raise ValueError(f'Cached and uncached render differ at {view.path}')
    return quality, delta, difference


def _selected_render_cache(module, args):
    return (module.gaussians.render_attribute_cache
            if args.render_attribute_cache is None else
            args.render_attribute_cache)


def _run_sequence(args, config, run, device, first_gpu_check):
    import torch

    if (args.mode not in ('auto', 'stream') or args.checkpoint is not None):
        raise ValueError('Complete sequence requires the decoded stream, not a checkpoint')
    if args.sequence_warmups < 10 or args.sequence_renders < 50:
        raise ValueError('Complete-sequence protocol requires >=10 warmups and '
                         '>=50 timed full renders per frame')
    if args.segment_size < 1 or args.gpu_check_interval < 1:
        raise ValueError('Sequence segment and GPU-check intervals must be positive')
    stream = args.stream_dir or run / 'frame_storage' / 'rigid_tree'
    reference = (args.reference or stream / 'reference_frame0.npz').resolve()
    module, datamodule = _make_module(config, device)
    num_frames = datamodule.num_frames
    if num_frames < 2:
        raise ValueError('Complete sequence needs frame0 and dynamic frames')
    for frame in range(1, num_frames):
        for path in (stream / f'motion_{frame:03d}.bin',
                     stream / f'frame_{frame:03d}.json',
                     run / 'frame_summary' / f'frame_{frame:03d}.json'):
            if not path.is_file():
                raise FileNotFoundError(f'Incomplete sequence: {path}')
    state, raw = _load_reference(reference, None)
    if state.frame_index != 0:
        raise ValueError('Complete sequence must start from reference frame0')
    cache_enabled = _selected_render_cache(module, args)
    reference_bytes = reference.stat().st_size
    checks = [dict(after_frame=None, status=first_gpu_check)]
    rows = []
    tolerance = dict(psnr=args.psnr_tolerance,
                     ssim=args.ssim_tolerance, l1=args.l1_tolerance)
    relative_reference = json.loads(
        (stream / 'frame_001.json').read_text())['reference']
    for frame in range(num_frames):
        if frame:
            read_started = time.perf_counter()
            state, raw = _decode_frame(stream, frame, state, raw)
            manifest = _validate_stream_manifest(stream, frame, state, raw,
                                                 relative_reference)
            decode_verify_seconds = time.perf_counter() - read_started
        else:
            decode_verify_seconds = 0.0
        upload_started = time.perf_counter()
        _install_raw(module, raw, device)
        torch.cuda.synchronize()
        upload_seconds = time.perf_counter() - upload_started
        view, test_views = _cam00(datamodule, frame, device)
        if test_views != 1:
            raise ValueError('Complete-sequence cam00 PSNR needs one test view per frame')
        expected = None
        storage = dict(motion_bytes=0, birth_bytes=0, opacity_update_bytes=0, encoded_bytes=0,
                       payload_bytes=0, manifest_index_bytes=0, complete_update_bytes=0)
        if frame:
            summary = json.loads((run / 'frame_summary' /
                f'frame_{frame:03d}.json').read_text())
            expected = summary['evaluation']['test']
            storage = _actual_frame_storage(
                stream, frame, manifest, summary, reference_bytes,
                relative_reference)
        quality, difference, cache_difference = _sequence_quality(
            module, view, expected, tolerance, cache_enabled)
        seconds = _time_sequence_render(
            module, view, args.sequence_warmups, args.sequence_renders)
        rows.append(dict(frame=frame, camera=view.path.name,
            gaussian_count=len(raw['xyz']),
            quality=quality, quality_saved_difference=difference,
            cache_pixel_max_abs_difference=cache_difference,
            warmup_renders=args.sequence_warmups,
            timed_renders=args.sequence_renders,
            render_seconds=seconds, render_fps=args.sequence_renders / seconds,
            file_decode_verify_seconds=decode_verify_seconds,
            upload_seconds=upload_seconds, **storage))
        if (frame + 1) % 50 == 0 or frame == num_frames - 1:
            completed = summarize_render_frames(rows)
            total_bytes = sum(row['encoded_bytes'] for row in rows)
            complete_bytes = sum(row['complete_update_bytes'] for row in rows)
            print(f"Sequence {frame + 1}/{num_frames}: "
                  f"render {completed['aggregate_fps']:.2f} FPS, "
                  f"mean PSNR {completed['psnr_mean']:.3f} dB, "
                  f"dynamic payload {total_bytes} bytes, "
                  f"with manifest indices {complete_bytes} bytes", flush=True)
        if ((frame + 1) % args.gpu_check_interval == 0 or
                frame == num_frames - 1):
            checks.append(dict(after_frame=frame,
                               status=check_gpu_exclusivity()))
        del view
    groups = _sequence_groups(rows, args.segment_size)
    result = dict(schema='rigid_residual_tree_complete_sequence_benchmark_v1',
        run=str(run), stream_dir=str(stream.resolve()), reference=str(reference),
        expected_frames=num_frames, measured_frames=len(rows),
        image_shape=[3, 1199, 1600], cache_enabled=cache_enabled,
        gpu_name=torch.cuda.get_device_name(device),
        storage=_summarize_sequence_storage(rows, reference_bytes),
        timing_protocol=dict(warmups_per_frame=args.sequence_warmups,
            timed_full_renders_per_frame=args.sequence_renders,
            cuda_synchronize_before_and_after=True,
            counted_operation='Module.forward(cam00)[render].clamp(0,1)'),
        groups=groups, frames=rows,
        decode_verify_seconds_total=sum(
            row['file_decode_verify_seconds'] for row in rows),
        upload_seconds_total=sum(row['upload_seconds'] for row in rows),
        gpu_checks=checks,
        scopes=dict(render_fps='Only synchronized full materialized render + clamp '
                    'after per-frame decode/upload; warmups, camera loading, quality '
                    'checks and decode are excluded.',
                    decode='Motion/birth/optional-opacity file reads, CPU decode and manifest verification '
                           'reported separately; initial reference load is one-time '
                           'and excluded.'),
        no_training_updates=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(all_frames_fps=groups['all_frames']['aggregate_fps'],
        dynamic_frames_fps=groups['dynamic_frames']['aggregate_fps'],
        all_frames_psnr=groups['all_frames']['psnr_mean'],
        dynamic_frames_psnr=groups['dynamic_frames']['psnr_mean'],
        dynamic_payload_bytes_mean=result['storage']['dynamic_packets']['payload_bytes_mean'],
        complete_update_bytes_mean=result['storage']['dynamic_packets']['complete_update_bytes_mean'],
        output=str(args.output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    frame_selection = parser.add_mutually_exclusive_group(required=True)
    frame_selection.add_argument('--frame', type=int)
    frame_selection.add_argument('--sequence', action='store_true',
                                 help='Evaluate every frame from reference frame0')
    parser.add_argument('--mode', choices=('auto', 'checkpoint', 'stream', 'both'),
                        default='auto')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--stream-dir', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--warmup-renders', type=int, default=5)
    parser.add_argument('--timed-renders', type=int, default=30)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--playback-rounds', type=int, default=2)
    parser.add_argument('--sequence-warmups', type=int, default=10)
    parser.add_argument('--sequence-renders', type=int, default=50)
    parser.add_argument('--segment-size', type=int, default=50)
    parser.add_argument('--gpu-check-interval', type=int, default=50)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument('--render-attribute-cache', dest='render_attribute_cache',
                       action='store_true', help='Explicitly enable the frozen-GS cache')
    cache.add_argument('--no-render-attribute-cache', dest='render_attribute_cache',
                       action='store_false', help='Disable the cache for an A/B check')
    parser.set_defaults(render_attribute_cache=None)
    parser.add_argument('--psnr-tolerance', type=float, default=0.05)
    parser.add_argument('--ssim-tolerance', type=float, default=0.001)
    parser.add_argument('--l1-tolerance', type=float, default=0.0001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.frame is not None and args.frame < 1:
        parser.error('Single-frame target must follow frame0')
    if min(args.warmup_renders, args.timed_renders,
           args.rounds, args.playback_rounds) < 1:
        parser.error('Timing counts must be positive')
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for materialized rendering FPS')
    device = torch.device('cuda')
    first_gpu_check = check_gpu_exclusivity()
    run = args.run.resolve()
    config = _load_config(run)
    if config.data.resolution != -1 or config.data.expected_image_size != [1600, 1199]:
        raise ValueError('Benchmark requires full-resolution 1600x1199 configuration')
    if args.sequence:
        _run_sequence(args, config, run, device, first_gpu_check)
        return
    total_steps = config.module.rigid_tree.total_steps_for_frame(args.frame)
    checkpoint = args.checkpoint or run / f'rigid-ckpt-{args.frame}-{total_steps}.pth'
    stream = args.stream_dir or run / 'frame_storage' / 'rigid_tree'
    manifest_path = stream / f'frame_{args.frame:03d}.json'
    has_checkpoint = checkpoint.is_file()
    has_stream = manifest_path.is_file()
    mode = (('both' if has_checkpoint and has_stream else
             'checkpoint' if has_checkpoint else 'stream')
            if args.mode == 'auto' else args.mode)
    if mode in ('checkpoint', 'both') and not has_checkpoint:
        raise FileNotFoundError(checkpoint)
    if mode in ('stream', 'both') and not has_stream:
        raise FileNotFoundError(manifest_path)
    if mode == 'stream' and args.checkpoint is not None:
        raise ValueError('--checkpoint is unused in stream mode')
    module, datamodule = _make_module(config, device)
    view, test_views = _cam00(datamodule, args.frame, device)
    if tuple(view.image.shape) != (3, 1199, 1600):
        raise ValueError(f'cam00 loaded at unexpected shape {tuple(view.image.shape)}')

    expected = None
    summary_path = run / 'frame_summary' / f'frame_{args.frame:03d}.json'
    if test_views == 1 and summary_path.is_file():
        expected = json.loads(summary_path.read_text())['evaluation']['test']
    elif summary_path.is_file():
        warnings.warn('Test evaluation averages several cameras; cam00 quality '
                      'will be reported without an invalid mean comparison.',
                      RuntimeWarning, stacklevel=2)
    tolerance = dict(psnr=args.psnr_tolerance,
                     ssim=args.ssim_tolerance, l1=args.l1_tolerance)

    checkpoint_raw, checkpoint_degree = None, None
    if mode in ('checkpoint', 'both'):
        checkpoint_raw, checkpoint_degree = _capture_raw(
            checkpoint, args.frame, total_steps)
        if checkpoint_degree != config.module.sh_degree:
            raise ValueError('Checkpoint SH degree differs from run configuration')

    reference_path, reference_state, reference_raw = None, None, None
    stream_raw = None
    if mode in ('stream', 'both'):
        from scripts.replay_rigid_tree import replay

        manifest = json.loads(manifest_path.read_text())
        reference_path = args.reference or run / manifest['reference']
        reference_state, reference_raw = _load_reference(
            reference_path, None)
        if reference_state.frame_index >= args.frame:
            raise ValueError('Decoder reference must precede benchmark frame')
        # Validate hashes and all raw attributes before measuring; this also
        # makes later file timing explicitly warm-cache local I/O.
        _, stream_raw = replay(stream, reference_path, args.frame)
        for frame in range(reference_state.frame_index + 1, args.frame + 1):
            other, _ = _cam00(datamodule, frame, device)
            if (other.image.shape != view.image.shape or
                    other.fovx != view.fovx or other.fovy != view.fovy or
                    any(not torch.equal(getattr(other, key), getattr(view, key))
                        for key in ('world_view_transform', 'full_proj_transform',
                                    'camera_center'))):
                raise ValueError(f'cam00 camera changed at frame {frame}')
            del other

    exact_checkpoint_stream = None
    if mode == 'both':
        exact_checkpoint_stream = {
            name: torch.equal(checkpoint_raw[name], stream_raw[name])
            for name in RAW_FIELDS}
        if not all(exact_checkpoint_stream.values()):
            raise ValueError(f'Checkpoint and decoded stream differ: {exact_checkpoint_stream}')
    selected_raw = checkpoint_raw if checkpoint_raw is not None else stream_raw
    _install_raw(module, selected_raw, device)
    cache_enabled = _selected_render_cache(module, args)
    module.gaussians.render_attribute_cache = False
    with torch.no_grad():
        uncached_image = _render_checked(module, view)
    module.gaussians.render_attribute_cache = cache_enabled
    module.gaussians._render_cache = {}
    cache_pixel_max_abs_difference = None
    with torch.no_grad():
        quality, quality_delta, checkpoint_image = _quality(
            module, view, expected, tolerance)
        if cache_enabled:
            cache_pixel_max_abs_difference = float(
                (checkpoint_image - uncached_image).abs().max())
            if not torch.equal(checkpoint_image, uncached_image):
                raise ValueError('Cached and uncached full renders differ')
        materialized = _time_materialized(
            module, view, args.warmup_renders, args.timed_renders, args.rounds)

    playback = None
    stream_quality = None
    if mode in ('stream', 'both'):
        with torch.no_grad():
            playback = _time_playback(module, view, stream, reference_state,
                                      reference_raw, args.frame,
                                      args.playback_rounds, device)
            stream_image = playback.pop('final_image')
            playback.pop('final_raw')
            if not torch.allclose(stream_image, checkpoint_image,
                                  rtol=0, atol=1e-5):
                raise ValueError('Stream and materialized renders disagree at target frame')
            stream_quality = dict(pixel_max_abs_difference=float(
                (stream_image - checkpoint_image).abs().max()))
    result = dict(schema='rigid_residual_tree_benchmark_v1',
        run=str(run), frame=args.frame, mode=mode,
        checkpoint=str(checkpoint.resolve()) if checkpoint_raw is not None else None,
        stream_dir=str(stream.resolve()) if stream_raw is not None else None,
        reference=str(reference_path.resolve()) if reference_path else None,
        image_shape=list(view.image.shape), camera=view.path.name,
        test_view_count=test_views, quality=quality,
        quality_reference=expected, quality_difference=quality_delta,
        checkpoint_stream_raw_exact=exact_checkpoint_stream,
        stream_render_comparison=stream_quality,
        gaussian_count=len(selected_raw['xyz']),
        render_attribute_cache=cache_enabled,
        cache_pixel_max_abs_difference=cache_pixel_max_abs_difference,
        materialized=materialized, playback=playback,
        gpu_name=torch.cuda.get_device_name(device),
        initial_gpu_check=first_gpu_check,
        scopes=dict(
            materialized='CUDA-synchronized Module.forward plus clamp; includes Python '
                         'property activation and rasterizer setup, excludes decoding, transfer, '
                         'camera/GT loading and metrics.',
            playback='Per dynamic frame: local motion/birth file reads, integrity-checked CPU '
                     'decode, raw GS host-to-GPU install, CUDA-synchronized Module.forward '
                     'plus clamp. Excludes one-time reference, camera/GT loading, manifest '
                     'verification and metrics. Files were preflight-read, so OS cache may help.'),
        no_training_updates=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(materialized_fps=materialized['fps'],
                          playback_fps=(playback['fps'] if playback else None),
                          psnr=quality['psnr'], output=str(args.output))))


if __name__ == '__main__':
    main()
