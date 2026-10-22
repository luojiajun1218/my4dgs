"""Run the five remaining N3DV sequences after their input audits finish.

Training, reference evaluation, and render benchmarks run in separate processes
so a monitoring process cannot leave a foreign CUDA context in the benchmark.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
FLAME_RESULT = ROOT / 'output/n3dv/flame_steak_20261019T091653_half_final480w'
SCENES = ('cook_spinach', 'cut_roasted_beef', 'sear_steak', 'coffee_martini', 'flame_salmon_1')
sys.path.insert(0, str(ROOT))
from scripts.n3dv_output_layout import scene_output


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def power(expected_limit=480):
    result = subprocess.run(['nvidia-smi', '--query-gpu=power.limit,power.draw,temperature.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError('GPU unavailable: ' + (result.stdout + result.stderr).strip())
    values = [float(part.strip()) for part in result.stdout.strip().split(',')]
    if len(values) != 3 or values[0] != expected_limit:
        raise ValueError(f'Expected the declared {expected_limit} W limit; observed ' + result.stdout.strip())
    return dict(limit_w=values[0], draw_w=values[1], temperature_c=values[2])


def thermal_events():
    path = os.environ.get('N3DV_THERMAL_EVENTS')
    return Path(path).read_text().splitlines() if path and Path(path).exists() else []


def ready_scene(audit, name):
    row = read(audit / 'ready_scenes.json').get('ready_scenes', {}).get(name)
    if not row or row.get('status') != 'passed':
        return None
    for filename in ('frame_alignment_checks.json', 'frame0_my4dgs_checks.json'):
        if read(audit / filename).get(name, {}).get('status') != 'passed':
            return None
    support = read(audit / 'seed_support_checks.json').get(name, {})
    if support.get('seed_points_with_fewer_than_two_fusion_cameras') != 0:
        return None
    first = read(audit / 'fresh_frame0_results.json').get(name, {})
    if first.get('status') != 'complete' or first.get('global_step') != 15000:
        return None
    manifest = read(Path(row['source']) / 'preprocessing_manifest.json')
    if manifest.get('status') != 'complete' or manifest.get('frames') != 300:
        return None
    return row


@contextmanager
def gpu_lease(output):
    path = Path(os.environ.get('N3DV_GPU_LOCK', str(output / 'gpu.lock')))
    with path.open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        yield


@contextmanager
def quiet_benchmark(output):
    flag = output / 'pause_background_for_benchmark.json'
    write(flag, dict(owner_pid=os.getpid()))
    try:
        time.sleep(2)
        yield
    finally:
        flag.unlink(missing_ok=True)


def verify_flame():
    directory = FLAME_RESULT
    manifest = read(directory / 'preservation_manifest.json')
    for filename, key in (('frame0.pth', 'bootstrap_checkpoint_sha256'),
                          ('final_checkpoint.pth', 'final_checkpoint_sha256')):
        if sha256(directory / filename) != manifest[key]:
            raise ValueError('Protected flame_steak artifact changed: ' + filename)


def contiguous_reports(run):
    reports = []
    for frame in range(1, 300):
        report = read(run / 'frame_summary' / f'frame_{frame:03d}.json')
        if not report:
            break
        reports.append(report)
    return reports


def require_complete(run):
    expected = [f'frame_{frame:03d}.json' for frame in range(1, 300)]
    if sorted(p.name for p in (run / 'frame_summary').glob('frame_*.json')) != expected:
        raise ValueError('Expected all 299 dynamic-frame summaries, without extra frames')
    reports = contiguous_reports(run)
    for frame, report in enumerate(reports, 1):
        if report['frame'] != frame or report['train_loss_steps'] != 300:
            raise ValueError('Frame/step coverage differs from the fixed 300-frame recipe')
        for name in (f'motion_{frame:03d}.bin', f'frame_{frame:03d}.json'):
            if not (run / 'frame_storage/rigid_tree' / name).is_file():
                raise FileNotFoundError(name)
    if not (run / 'rigid-ckpt-299-300.pth').is_file():
        raise FileNotFoundError('Missing final frame-299, step-300 checkpoint')
    return reports


def measure_reference(run, output):
    import torch
    from libgs.metric import lpips, psnr, ssim
    from scripts.benchmark_rigid_residual_tree import (
        _cam00, _install_raw, _load_config, _load_reference, _make_module,
    )
    torch.set_num_threads(int(os.environ.get('N3DV_CPU_THREADS', '4')))
    config = _load_config(run)
    module, data = _make_module(config, torch.device('cuda'))
    reference = run / 'frame_storage/rigid_tree/reference_frame0.npz'
    _, raw = _load_reference(reference, None)
    _install_raw(module, raw, torch.device('cuda'))
    view, count = _cam00(data, 0, torch.device('cuda'))
    if count != 1:
        raise ValueError('Expected held-out cam00 only')
    with torch.no_grad():
        image = module.forward(view)['render'].clamp(0, 1)
        result = dict(frame=0, camera='cam00.png', reference_sha256=sha256(reference),
                      gaussian_count=len(raw['xyz']), psnr=float(psnr(image, view.image).mean()),
                      ssim=float(ssim(image, view.image)),
                      lpips=float(lpips(image, view.image, net_type='vgg').mean()))
    write(output, result)


def check_final_state(run):
    import torch
    from scripts.benchmark_rigid_residual_tree import _capture_raw, _decode_frame, _load_reference
    torch.set_num_threads(int(os.environ.get('N3DV_CPU_THREADS', '4')))
    stream = run / 'frame_storage/rigid_tree'
    state, raw = _load_reference(stream / 'reference_frame0.npz', None)
    for frame in range(1, 300):
        state, raw = _decode_frame(stream, frame, state, raw)
    captured, degree = _capture_raw(run / 'rigid-ckpt-299-300.pth', 299, 300)
    exact = {name: torch.equal(value, captured[name]) for name, value in raw.items()}
    if not all(exact.values()) or degree != 1:
        raise ValueError('Sequential decoded state differs from the final checkpoint')
    return dict(status='passed', frame=299, step=300, raw_attributes_exact=exact,
                final_checkpoint_sha256=sha256(run / 'rigid-ckpt-299-300.pth'))


def model_timing_fields(reports):
    from pipeline.timing import summarize_frame_timings
    summary = summarize_frame_timings(reports)
    model = summary['model_training']
    return dict(timing_versions=sorted({row.get('timing', {}).get('version', 0) for row in reports}),
        model_training_measured_dynamic_frames=model['measured_frame_count'],
        model_training_seconds_mean=model['mean_seconds']['training_seconds'],
        model_training_scope=('Forward/backward, optimizer updates and training regularization; '
            'excludes offline preprocessing, frame setup, image loading/transfers, final encoding, '
            'progress diagnostics, evaluation and checkpoint writes. Version-1 reports do not measure this scope.'))


def write_frame_metrics(reports, benchmark, measurement, lpips_values, model_time_valid=True):
    with (measurement / 'per_frame_metrics.csv').open('w', newline='') as file:
        columns = ['frame', 'psnr', 'ssim', 'lpips_vgg', 'complete_update_kb', 'pure_render_fps',
                   'gaussians', 'model_training_seconds', 'model_phase_elapsed_seconds',
                   'training_loop_seconds', 'preparation_seconds', 'data_loading_seconds',
                   'encoding_seconds', 'evaluation_seconds', 'checkpoint_seconds', 'frame_wall_seconds']
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(benchmark['frames']):
            frame_timing = reports[index - 1].get('timing', {}) if index else {}
            model = (frame_timing.get('training_seconds') if frame_timing.get('version') == 2 else None)
            writer.writerow(dict(frame=index, psnr=row['quality']['psnr'], ssim=row['quality']['ssim'],
                lpips_vgg=lpips_values[index], complete_update_kb=row['complete_update_bytes'] / 1000,
                pure_render_fps=row['render_fps'], gaussians=row['gaussian_count'],
                model_training_seconds=model if model_time_valid else None,
                model_phase_elapsed_seconds=model,
                training_loop_seconds=(frame_timing.get('training_loop_seconds')
                    if frame_timing.get('version') == 2 else frame_timing.get('training_seconds')),
                **{key: frame_timing.get(key) for key in columns[-6:]}))


def collect(run, measurement, scene, image_size, power_limit=480):
    reports = require_complete(run)
    benchmark = read(measurement / 'decoded_sequence.json')
    initial = read(measurement / 'reference_metrics.json')
    if not benchmark.get('sequence_complete') or benchmark.get('measured_frames') != 300:
        raise ValueError('An incomplete benchmark must not be reported as 300 frames')
    if not all(row['status']['verified'] for row in benchmark['gpu_checks']):
        raise ValueError('GPU exclusivity was not verified for every benchmark segment')
    if abs(benchmark['frames'][0]['quality']['psnr'] - initial['psnr']) > 1e-5:
        raise ValueError('Independent reference metrics do not match sequence frame0')
    timing = benchmark['training_timing']['all_dynamic']
    if not timing['all_frames_measured'] or timing['measured_frame_count'] != 299:
        raise ValueError('Pure training timing coverage must include all dynamic frames')
    quality = benchmark['groups']['all_frames']
    storage = benchmark['storage']
    lpips_values = [initial['lpips']] + [row['evaluation']['test']['lpips'] for row in reports]
    result = dict(scene=scene, status='verified', frames=300, dynamic_frames=299,
                  power_limit_w=power_limit,
                  image_size=image_size, run=str(run), measurement=str(measurement),
                  psnr_db=quality['psnr_mean'], ssim=quality['ssim_mean'],
                  lpips_vgg=statistics.mean(lpips_values),
                  dynamic_complete_kb=storage['dynamic_packets']['complete_update_bytes_mean'] / 1000,
                  reference_mb=storage['bootstrap_reference_bytes'] / 1e6,
                  reference_amortized_kb_per_frame=storage['total_reference_and_complete_update_bytes'] / 300000,
                  pure_render_fps=quality['aggregate_fps'],
                  timed_renders=quality['total_timed_renders'], render_seconds=quality['total_render_seconds'],
                  initial_gaussians=benchmark['frames'][0]['gaussian_count'],
                  final_gaussians=benchmark['frames'][-1]['gaussian_count'],
                  timing_mean_seconds=timing['mean_seconds'],
                  timing_excluding_first_dynamic=benchmark['training_timing']['excluding_first_dynamic'],
                  final_state=check_final_state(run),
                  max_saved_psnr_difference=max(abs(row['quality_saved_difference']['psnr'])
                      for row in benchmark['frames'][1:]),
                  max_cache_pixel_difference=max(row['cache_pixel_max_abs_difference']
                      for row in benchmark['frames']))
    result.update(model_timing_fields(reports))
    write_frame_metrics(reports, benchmark, measurement, lpips_values)
    write(measurement / 'summary.json', result)
    return result


def refresh_completed_statistics(output):
    """Refresh derived paths/time labels without rerendering or rewriting evidence."""
    results = read(output / 'results.json')
    if not results:
        raise ValueError('No verified scene results are available yet')
    for name, result in results.items():
        if result.get('status') != 'verified' or result.get('frames') != 300:
            raise ValueError('Only verified complete scene statistics can be refreshed')
        run, measurement = Path(result['run']).resolve(), Path(result['measurement']).resolve()
        reports = require_complete(run)
        benchmark = read(measurement / 'decoded_sequence.json')
        if not benchmark.get('sequence_complete') or benchmark.get('measured_frames') != 300:
            raise ValueError('Missing complete independent measurement')
        initial = read(measurement / 'reference_metrics.json')
        result.update(model_timing_fields(reports), run=str(run), measurement=str(measurement),
                      scene_output=str(measurement.parent))
        valid = result.get('training_timing_excludes_thermal_pauses', True)
        result['model_phase_elapsed_seconds_mean'] = result['model_training_seconds_mean']
        if not valid:
            result['model_training_seconds_mean'] = None
        result['training_loop_seconds_mean'] = statistics.mean(
            row['timing'].get('training_loop_seconds', row['timing']['training_seconds']) for row in reports)
        for key in ('preparation_seconds', 'data_loading_seconds', 'encoding_seconds', 'diagnostics_seconds'):
            result[key + '_mean'] = (statistics.mean(row['timing'][key] for row in reports)
                if all(row['timing'].get('version') == 2 for row in reports) else None)
        lpips_values = [initial['lpips']] + [row['evaluation']['test']['lpips'] for row in reports]
        write_frame_metrics(reports, benchmark, measurement, lpips_values, model_time_valid=valid)
        write(measurement / 'summary.json', result)
    write(output / 'results.json', results)
    if set(results) == set(SCENES):
        publish(output, results)
    return results


def publish(output, results):
    evidence = ROOT / 'docs/experiment/evidence' / output.name
    evidence.mkdir(parents=True, exist_ok=True)
    write(evidence / 'summary.json', results)
    rows = [results[name] for name in SCENES]
    for row in rows:
        seconds = row.get('model_training_seconds_mean')
        row['training_time_display'] = (f'{seconds:.3f}' if seconds is not None and
            row.get('training_timing_excludes_thermal_pauses', True) else '未单独测得')
    table = '\n'.join(f"| {r['scene']} | {r['image_size'][0]}×{r['image_size'][1]} | "
        f"{r['psnr_db']:.4f} | {r['ssim']:.6f} | {r['lpips_vgg']:.6f} | "
        f"{r['dynamic_complete_kb']:.3f} | {r['pure_render_fps']:.2f} | "
        f"{r['training_time_display']} |" for r in rows)
    flame = read(FLAME_RESULT / 'measurement/decoded_sequence.json')
    all_six_psnr = statistics.mean([r['psnr_db'] for r in rows] +
                                  [flame['groups']['all_frames']['psnr_mean']])
    all_six_kb = statistics.mean([r['dynamic_complete_kb'] for r in rows] +
        [flame['storage']['dynamic_packets']['complete_update_bytes_mean'] / 1000])
    fg = flame['groups']['all_frames']
    all_six_fps = (sum(r['timed_renders'] for r in rows) + fg['total_timed_renders']) / (
        sum(r['render_seconds'] for r in rows) + fg['total_render_seconds'])
    aggregate = dict(scene_count=6, psnr_db=all_six_psnr, dynamic_complete_kb=all_six_kb,
                     aggregate_pure_render_fps=all_six_fps, frame_count=1800,
                     training_seconds_all_six=None,
                     new_scenes_power_limit_w=rows[0]['power_limit_w'], flame_power_limit_w=480,
                     note='Includes the immutable flame_steak final result; its pure training timing is unknown. Power limits are recorded separately.')
    write(output / 'six_scene_aggregate.json', aggregate)
    write(evidence / 'six_scene_aggregate.json', aggregate)
    import shutil
    for row in rows:
        target = evidence / row['scene']
        target.mkdir(exist_ok=True)
        for name in ('summary.json', 'per_frame_metrics.csv', 'reference_metrics.json'):
            shutil.copy2(Path(row['measurement']) / name, target / name)
    power_note = ('两轮 GPU 上限均为 480 W；本轮五景按 CPU 温度限制并发，资源条件与旧 flame 测量并非完全相同。'
                  if rows[0]['power_limit_w'] == 480 else
                  '两轮 GPU 功率上限不同，合并值不能写成统一功率条件的基准。')
    document = f'''# N3DV 其余五景完整 300 帧

五景均从本轮重建的相机、图像和 40000 点初始化云出发，首帧各自从零训练 15000 步，然后以 flame_steak 正式版固定配方完成第 0–299 帧。每动态帧 300 步、运动每步 2 视角、新增每步 1 视角，cam00 留出，半分辨率 `resolution: 1`，本轮动态运行功率上限 {rows[0]['power_limit_w']} W。未做场景专属调参，未改变运动方法。数据流程见[预处理与首帧记录](20261019_n3dv_other5_from_raw_and_fresh_frame0.md)。完成时间：{datetime.now().astimezone().isoformat()}。

新增 `scripts/run_n3dv_scenes.py` 顺序调度训练、独立解码和测量；逐场景检查完成后即进入 GPU 队列，无需等待其他场景；GPU 重建、首帧训练和动态训练通过同一文件锁互斥。仅调整恢复点保存策略：保存第 1 帧和每 30 帧末尾的 step-300 状态，省去 step-20/70/200 中间检查点；训练步数、视角及优化参数不变。源码冻结于 `{output / 'source'}`，各场景配置保留在输出的 `configs/` 和运行目录，原始训练产物不写入 Git。

本轮按作者控制 CPU 温度的要求限制并发，实际 CPU 亲和性和计算库线程数保存在 `supervision/` 的阶段记录，数据加载 worker 为 0，CPU 预处理可与 GPU 训练重叠，所有 CUDA 工作互斥；70°C 暂停、55°C 恢复。温控事件单独留证；出现暂停的训练分项不宣称为纯训练时间，渲染计时若被暂停则作废并重测。

| 场景 | 分辨率 | PSNR/dB | SSIM | LPIPS/VGG | 完整动态 KB/帧 | 纯渲染 FPS | 纯训练秒/动态帧 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

质量统计覆盖 0–299 帧；动态存储和训练时间覆盖 1–299 帧。KB=1000 B，完整更新含运动、新增、透明度、区域 DC 及必要 manifest，首帧参考另计。首帧参考体积和摊销结果保存在 JSON，不能将动态更新均值当作含首帧的整段码率。

每景顺序解码全部 299 个更新，验证已保存 PSNR/SSIM/L1，比较缓存与直接渲染，并额外将最终六种属性与第 299 帧 checkpoint 逐张量精确比较。LPIPS 的第 0 帧基于实际量化参考独立重算，动态帧取训练终态评估值；未声称独立重算全部动态 LPIPS。

FPS 使用每帧 10 次预热和 50 次完整 `Module.forward(cam00)+clamp`，前后 CUDA 同步，共每景 15000 次计时渲染，GPU 独占检查全部通过。此数值不含解码、上传、图像加载或质量计算。模型训练时间只统计前向、反向、参数更新及训练正则；数据预处理、模型初始化、取图/传输、最终码流编码、日志、评估及 checkpoint 另计。旧版训练循环时间不能改称模型训练时间，缺少分项时显示“未单独测得”；同时保留排除首动态帧冷启动后的记录。

连同既有 flame_steak 正式版，六景平均 PSNR **{all_six_psnr:.4f} dB**，平均完整动态更新 **{all_six_kb:.3f} KB/帧**，按全部计时渲染数除以总 GPU 渲染秒数的合并 FPS **{all_six_fps:.2f}**。flame 的旧结果缺少纯训练分项，因此六景纯训练平均保持未知。{power_note} 本轮不是功率对照实验。

结果根目录：`{output}`。每景 `measurement/summary.json`、`per_frame_metrics.csv`、`decoded_sequence.json` 分别保存汇总、逐帧数据与完整解码/计时证据。flame 正式结果入口为 `{FLAME_RESULT}`，目录位置按作者要求统一，原始训练和测量内容保持不变。
'''
    (ROOT / 'docs/experiment' / (datetime.now().strftime('%Y%m%d') + '_n3dv_other5_full300.md')).write_text(document)


def main(args):
    output = args.output.resolve()
    audit = args.preparation_audit.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'batch_status.json'
    # A later preparation supervisor may join the already active GPU queue.
    # The file lock prevents it from starting a duplicate training run.
    batch_lock = (output / 'batch.lock').open('a')
    try:
        fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        while True:
            current = read(state_path)
            if current.get('status') == 'complete':
                return
            if current.get('status') == 'failed':
                raise RuntimeError('The active GPU batch failed: ' + str(current.get('error')))
            try:
                fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(5)
            else:
                raise RuntimeError('The active batch exited without a completed result')
    if state_path.exists():
        previous = read(state_path)
        if previous.get('status') == 'complete':
            if set(read(output / 'results.json')) != set(SCENES):
                raise ValueError('Completed batch has missing scene results')
            return
        if not args.resume_waiting or previous.get('dynamic_training_started'):
            raise FileExistsError('Inspect existing dynamic progress before resuming a batch')
        attempts = output / 'attempts'
        attempts.mkdir(exist_ok=True)
        write(attempts / (datetime.now().strftime('%Y%m%dT%H%M%S') + '_batch_status.json'), previous)
    state = dict(status='waiting_for_preprocessing', scenes=list(SCENES), completed_scenes=[],
                 pid=os.getpid(),
                 started_at=datetime.now().astimezone().isoformat(), current_scene=None,
                 dynamic_training_started=False, power_limit_w=args.power_limit)

    def update(**fields):
        state.update(fields, updated_at=datetime.now().astimezone().isoformat())
        write(state_path, state)

    def event(**fields):
        value = dict(at=datetime.now().astimezone().isoformat(), **fields)
        with (output / 'progress.jsonl').open('a') as file:
            file.write(json.dumps(value, ensure_ascii=False) + '\n')
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def run(command, cwd, log, callback=None, uninterrupted=False):
        thermal_before = len(thermal_events())
        power_before = power(args.power_limit)
        threads = os.environ.get('N3DV_CPU_THREADS', '8')
        with log.open('w') as file:
            child = subprocess.Popen(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                         OPENBLAS_NUM_THREADS=threads))
        update(child_pid=child.pid, command=[str(x) for x in command])
        try:
            while child.poll() is None:
                if callback:
                    callback()
                sample = power(args.power_limit)
                with (output / 'gpu_samples.jsonl').open('a') as file:
                    file.write(json.dumps(dict(at=datetime.now().astimezone().isoformat(),
                        scene=state['current_scene'], phase=state['status'], **sample)) + '\n')
                time.sleep(20)
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
        if callback:
            callback()
        during = [json.loads(line) for line in thermal_events()[thermal_before:]]
        if uninterrupted and any(event['kind'] == 'paused' for event in during):
            raise RuntimeError('Temperature pause interrupted render timing; discard this benchmark and rerun after cooling')
        return dict(command=[str(x) for x in command], cwd=str(cwd),
                    power_before=power_before, power_after=power(args.power_limit),
                    thermal_events=during,
                    thermal_paused_seconds=sum(event.get('paused_seconds', 0) for event in during))

    results = {}
    try:
        verify_flame()
        update()
        base = yaml.safe_load((ROOT / 'config/n3dv/flame_steak_half_final480w.yaml').read_text())
        config_directory = output / 'configs'
        config_directory.mkdir(exist_ok=True)
        snapshot = output / 'source'
        subprocess.run([sys.executable, str(ROOT / 'scripts/freeze_experiment_source.py'), str(snapshot),
                        '--data-path', 'data/colmap4d_half/cook_spinach'], check=True, cwd=ROOT)
        from scripts.benchmark_rigid_residual_tree import check_gpu_exclusivity
        configs, ready = {}, {}
        for name in SCENES:
            update(status='waiting_for_scene_inputs', current_scene=name)
            while True:
                row = ready_scene(audit, name)
                if row is not None:
                    break
                live = read(audit / 'live_status.json')
                if live.get('status') in ('needs_attention', 'failed', 'stopped'):
                    raise RuntimeError('Preprocessing requires attention: ' + str(live.get('error', live.get('issues'))))
                time.sleep(10)
            ready[name] = row
            checkpoint = args.frame0_root.resolve() / name / 'frame0.pth'
            if sha256(checkpoint) != read(audit / 'fresh_frame0_results.json')[name]['checkpoint_sha256']:
                raise ValueError('Fresh first-frame checkpoint changed: ' + name)
            config = json.loads(json.dumps(base))
            config['output_dir'] = str(output)
            config['experiment_name'] = name + '/training'
            config['data']['root'] = row['adapter']
            config['data']['expected_image_size'] = row['image_size']
            config['ckpt_path'] = str(checkpoint)
            config['trainer']['saving_ckpt_steps'] = [300]
            config['trainer']['saving_ckpt_frames'] = [1, *range(29, 300, 30)]
            if 'N3DV_DATALOADER_WORKERS' in os.environ:
                config['data'].setdefault('extra_dataloader_kwargs', {})['num_workers'] = int(os.environ['N3DV_DATALOADER_WORKERS'])
            path = config_directory / (name + '_half_full300.yaml')
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            configs[name] = path
            update(status='waiting_for_gpu', current_scene=name)
            with gpu_lease(output):
                check_gpu_exclusivity()
                update(status='training', current_scene=name, completed_frames=1, run=None, dynamic_training_started=True)
                directory = scene_output(output, name,
                    datetime.now().strftime('%Y%m%dT%H%M%S'), args.recipe_name)
                update(scene_output=str(directory))
                emitted = set()

                def progress():
                    candidates = list((directory / 'training').glob('*/config.yaml'))
                    if len(candidates) > 1:
                        raise ValueError('More than one training run for ' + name)
                    if not candidates:
                        return
                    run_dir = candidates[0].parent
                    reports = contiguous_reports(run_dir)
                    count = len(reports) + 1
                    update(completed_frames=count, run=str(run_dir))
                    for boundary in range(30, count + 1, 30):
                        if boundary in emitted:
                            continue
                        part = reports[max(0, boundary - 31):boundary - 1]
                        encoded = [row['encoded_bytes'] + (run_dir / 'frame_storage/rigid_tree' /
                                   f"frame_{row['frame']:03d}.json").stat().st_size for row in part]
                        event(scene=name, stage='training', completed_frames=boundary,
                              first_dynamic_frame=part[0]['frame'], last_dynamic_frame=part[-1]['frame'],
                              psnr_db=statistics.mean(r['evaluation']['test']['psnr'] for r in part),
                              dynamic_complete_kb=statistics.mean(encoded) / 1000,
                              training_seconds=statistics.mean(r['training_seconds'] for r in part))
                        emitted.add(boundary)

                provenance = run([sys.executable, '-u', str(snapshot / 'main.py'), '--config=' + str(configs[name])],
                                 snapshot, directory / 'training.log', progress)
                candidates = list((directory / 'training').glob('*/config.yaml'))
                if len(candidates) != 1:
                    raise ValueError('Expected exactly one complete run for ' + name)
                run_dir = candidates[0].parent
                require_complete(run_dir)
                measurement = directory / 'measurement'
                measurement.mkdir(exist_ok=True)
                update(status='benchmarking', completed_frames=300)
                command = [sys.executable, str(snapshot / 'scripts/benchmark_rigid_residual_tree.py'),
                    '--run', str(run_dir), '--sequence', '--mode=stream', '--sequence-warmups=10',
                    '--sequence-renders=50', '--segment-size=30', '--gpu-check-interval=30',
                    '--psnr-tolerance=0.00001', '--ssim-tolerance=0.000001', '--l1-tolerance=0.000001',
                    '--output', str(measurement / 'decoded_sequence.json')]
                with quiet_benchmark(output):
                    for attempt in range(3):
                        attempt_output = measurement / f'decoded_sequence_attempt_{attempt + 1}.json'
                        try:
                            provenance['benchmark'] = run([*command[:-1], str(attempt_output)], snapshot,
                                measurement / f'benchmark_attempt_{attempt + 1}.log', uninterrupted=True)
                            attempt_output.replace(measurement / 'decoded_sequence.json')
                            break
                        except RuntimeError as error:
                            if 'Temperature pause interrupted render timing' not in str(error) or attempt == 2:
                                raise
                            event(scene=name, stage='benchmark_retry_after_thermal_pause', attempt=attempt + 1)
                provenance['reference_metrics'] = run([sys.executable, str(snapshot / 'scripts/run_n3dv_scenes.py'),
                    '--measure-reference', str(run_dir), '--metric-output', str(measurement / 'reference_metrics.json')],
                    snapshot, measurement / 'reference_metrics.log')
                update(status='verifying_final_state')
                result = collect(run_dir, measurement, name, ready[name]['image_size'], args.power_limit)
                result['cpu_affinity'] = sorted(os.sched_getaffinity(0))
                result['thermal_paused_seconds_during_training'] = provenance['thermal_paused_seconds']
                result['training_timing_excludes_thermal_pauses'] = not provenance['thermal_paused_seconds']
                if provenance['thermal_paused_seconds']:
                    result['timing_note'] = 'Recorded training wall durations include temperature pauses; pure training timing is unavailable for this run.'
                write(measurement / 'summary.json', result)
                write(directory / 'launch_provenance.json', provenance)
                results[name] = result
                write(output / 'results.json', results)
                update(completed_scenes=list(results))
                event(scene=name, stage='verified', **{key: result[key] for key in
                    ('psnr_db', 'dynamic_complete_kb', 'pure_render_fps', 'timing_mean_seconds')})
        verify_flame()
        publish(output, results)
        update(status='complete', completed_at=datetime.now().astimezone().isoformat(), child_pid=None)
        event(stage='complete', scenes=list(results), frames=1500)
    except BaseException as error:
        update(status='failed', error=repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation-audit', type=Path,
                        default=ROOT / 'output/n3dv_preparation/audit')
    parser.add_argument('--frame0-root', type=Path,
                        default=ROOT / 'output/n3dv_preparation/frame0')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/n3dv' /
                        ('_batch_' + datetime.now().strftime('%Y%m%dT%H%M%S') + '_half_final480w'))
    parser.add_argument('--recipe-name', default='half_final480w')
    parser.add_argument('--refresh-summary', action='store_true',
                        help='Refresh derived timing labels and canonical paths for verified results only')
    parser.add_argument('--measure-reference', type=Path)
    parser.add_argument('--metric-output', type=Path)
    parser.add_argument('--power-limit', type=int, default=480,
                        help='Expected measured limit, not a command that changes hardware settings')
    parser.add_argument('--resume-waiting', action='store_true',
                        help='Resume a failed/waiting batch only if dynamic training never began')
    arguments = parser.parse_args()
    if arguments.refresh_summary:
        refresh_completed_statistics(arguments.output.resolve())
    elif arguments.measure_reference:
        measure_reference(arguments.measure_reference, arguments.metric_output)
    else:
        main(arguments)
