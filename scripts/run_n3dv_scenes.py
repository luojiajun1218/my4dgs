"""Run the five remaining N3DV sequences after their input audits finish.

Training, reference evaluation, and render benchmarks run in separate processes
so a monitoring process cannot leave a foreign CUDA context in the benchmark.
"""

import argparse
from datetime import datetime
import csv
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
SCENES = ('cook_spinach', 'cut_roasted_beef', 'sear_steak', 'coffee_martini', 'flame_salmon_1')
sys.path.insert(0, str(ROOT))


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


def verify_flame():
    directory = ROOT / 'output/final'
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
    torch.set_num_threads(4)
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
    torch.set_num_threads(4)
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
    with (measurement / 'per_frame_metrics.csv').open('w', newline='') as file:
        columns = ['frame', 'psnr', 'ssim', 'lpips_vgg', 'complete_update_kb', 'pure_render_fps',
                   'gaussians', 'training_seconds', 'evaluation_seconds', 'checkpoint_seconds', 'frame_wall_seconds']
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(benchmark['frames']):
            training = reports[index - 1] if index else {}
            writer.writerow(dict(frame=index, psnr=row['quality']['psnr'], ssim=row['quality']['ssim'],
                lpips_vgg=lpips_values[index], complete_update_kb=row['complete_update_bytes'] / 1000,
                pure_render_fps=row['render_fps'], gaussians=row['gaussian_count'],
                **{key: training.get(key) for key in columns[-4:]}))
    write(measurement / 'summary.json', result)
    return result


def publish(output, results):
    evidence = ROOT / 'docs/experiment/evidence/n3dv_other5_full300_20261019'
    evidence.mkdir(parents=True, exist_ok=True)
    write(evidence / 'summary.json', results)
    rows = [results[name] for name in SCENES]
    table = '\n'.join(f"| {r['scene']} | {r['image_size'][0]}×{r['image_size'][1]} | "
        f"{r['psnr_db']:.4f} | {r['ssim']:.6f} | {r['lpips_vgg']:.6f} | "
        f"{r['dynamic_complete_kb']:.3f} | {r['pure_render_fps']:.2f} | "
        f"{r['timing_mean_seconds']['training_seconds']:.3f} |" for r in rows)
    flame = read(ROOT / 'output/final/measurement/decoded_sequence.json')
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
    document = f'''# N3DV 其余五景完整 300 帧（2026-10-19 启动）

五景均从本轮重建的相机、图像和 40000 点初始化云出发，首帧各自从零训练 15000 步，然后以 flame_steak 正式版固定配方完成第 0–299 帧。每动态帧 300 步、运动每步 2 视角、新增每步 1 视角，cam00 留出，半分辨率 `resolution: 1`，本轮动态运行功率上限 {rows[0]['power_limit_w']} W。未做场景专属调参，未改变运动方法。数据流程见[预处理与首帧记录](20261019_n3dv_other5_from_raw_and_fresh_frame0.md)。完成时间：{datetime.now().astimezone().isoformat()}。

新增 `scripts/run_n3dv_scenes.py` 顺序调度训练、独立解码和测量；全部数据检查完成后才开始动态训练。仅调整恢复点保存策略：保存第 1 帧和每 30 帧末尾的 step-300 状态，省去 step-20/70/200 中间检查点；训练步数、视角及优化参数不变。源码冻结于 `{output / 'source'}`，各场景配置保留在 `config/n3dv/` 和运行目录，原始训练产物不写入 Git。

| 场景 | 分辨率 | PSNR/dB | SSIM | LPIPS/VGG | 完整动态 KB/帧 | 纯渲染 FPS | 纯训练秒/动态帧 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

质量统计覆盖 0–299 帧；动态存储和训练时间覆盖 1–299 帧。KB=1000 B，完整更新含运动、新增、透明度、区域 DC 及必要 manifest，首帧参考另计。首帧参考体积和摊销结果保存在 JSON，不能将动态更新均值当作含首帧的整段码率。

每景顺序解码全部 299 个更新，验证已保存 PSNR/SSIM/L1，比较缓存与直接渲染，并额外将最终六种属性与第 299 帧 checkpoint 逐张量精确比较。LPIPS 的第 0 帧基于实际量化参考独立重算，动态帧取训练终态评估值；未声称独立重算全部动态 LPIPS。

FPS 使用每帧 10 次预热和 50 次完整 `Module.forward(cam00)+clamp`，前后 CUDA 同步，共每景 15000 次计时渲染，GPU 独占检查全部通过。此数值不含解码、上传、图像加载或质量计算。纯训练时间按新增分项计时汇总，排除阶段留出渲染、评估和 checkpoint；同时保留排除首动态帧冷启动后的值。

连同既有 flame_steak 正式版，六景平均 PSNR **{all_six_psnr:.4f} dB**，平均完整动态更新 **{all_six_kb:.3f} KB/帧**，按全部计时渲染数除以总 GPU 渲染秒数的合并 FPS **{all_six_fps:.2f}**。flame 的旧结果缺少纯训练分项，因此六景纯训练平均保持未知。flame 既有测量上限为 480 W，本轮五景上限为 {rows[0]['power_limit_w']} W，合并值不能写成统一功率条件的基准，也不能当作功率对照实验。

结果根目录：`{output}`。每景 `measurement/summary.json`、`per_frame_metrics.csv`、`decoded_sequence.json` 分别保存汇总、逐帧数据与完整解码/计时证据。本轮不修改 `output/final` 的 flame 正式结果。
'''
    (ROOT / 'docs/experiment/20261019_n3dv_other5_full300.md').write_text(document)


def main(args):
    output = args.output.resolve()
    audit = args.preparation_audit.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'batch_status.json'
    if state_path.exists():
        previous = read(state_path)
        if not args.resume_waiting or previous.get('dynamic_training_started'):
            raise FileExistsError('Inspect existing dynamic progress before resuming a batch')
        attempts = output / 'attempts'
        attempts.mkdir(exist_ok=True)
        write(attempts / (datetime.now().strftime('%Y%m%dT%H%M%S') + '_batch_status.json'), previous)
    state = dict(status='waiting_for_preprocessing', scenes=list(SCENES), completed_scenes=[],
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

    def run(command, cwd, log, callback=None):
        power_before = power(args.power_limit)
        with log.open('w') as file:
            child = subprocess.Popen(command, cwd=cwd, stdout=file, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8'))
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
        return dict(command=[str(x) for x in command], cwd=str(cwd),
                    power_before=power_before, power_after=power(args.power_limit))

    results = {}
    try:
        verify_flame()
        update()
        while not (audit / 'completed_summary.json').is_file():
            live = read(audit / 'live_status.json')
            if live.get('status') == 'needs_attention':
                raise RuntimeError('Preprocessing requires attention: ' + str(live.get('issues')))
            update(preprocessing=live)
            time.sleep(20)
        ready = read(audit / 'ready_scenes.json')['ready_scenes']
        for name in SCENES:
            if ready[name]['status'] != 'passed' or read(audit / 'frame0_my4dgs_checks.json')[name]['status'] != 'passed':
                raise ValueError('Unverified dataset or native first frame: ' + name)
            checkpoint = args.frame0_root.resolve() / name / 'frame0.pth'
            if sha256(checkpoint) != read(audit / 'fresh_frame0_results.json')[name]['checkpoint_sha256']:
                raise ValueError('Fresh first-frame checkpoint changed: ' + name)
        base = yaml.safe_load((ROOT / 'config/n3dv/flame_steak_half_final480w.yaml').read_text())
        configs = {}
        for name in SCENES:
            config = json.loads(json.dumps(base))
            config['output_dir'] = str(output)
            config['experiment_name'] = name + '/training'
            config['data']['root'] = ready[name]['adapter']
            config['data']['expected_image_size'] = ready[name]['image_size']
            config['ckpt_path'] = str(args.frame0_root.resolve() / name / 'frame0.pth')
            config['trainer']['saving_ckpt_steps'] = [300]
            config['trainer']['saving_ckpt_frames'] = [1, *range(29, 300, 30)]
            path = ROOT / 'config/n3dv' / (name + '_half_final480w_full300.yaml')
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            configs[name] = path
        snapshot = output / 'source'
        subprocess.run([sys.executable, str(ROOT / 'scripts/freeze_experiment_source.py'), str(snapshot),
                        '--data-path', 'data/colmap4d_half/cook_spinach'], check=True, cwd=ROOT)
        from scripts.benchmark_rigid_residual_tree import check_gpu_exclusivity
        check_gpu_exclusivity()
        for name in SCENES:
            update(status='training', current_scene=name, completed_frames=1, run=None, dynamic_training_started=True)
            directory = output / name
            directory.mkdir(exist_ok=True)
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
            provenance['benchmark'] = run(command, snapshot, measurement / 'benchmark.log')
            provenance['reference_metrics'] = run([sys.executable, str(snapshot / 'scripts/run_n3dv_scenes.py'),
                '--measure-reference', str(run_dir), '--metric-output', str(measurement / 'reference_metrics.json')],
                snapshot, measurement / 'reference_metrics.log')
            update(status='verifying_final_state')
            result = collect(run_dir, measurement, name, ready[name]['image_size'], args.power_limit)
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
                        default=ROOT / 'output/n3dv_prepare_other5_20261019')
    parser.add_argument('--frame0-root', type=Path,
                        default=ROOT / 'output/n3dv_standard_other5_20261019')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/n3dv_other5_full300_20261019')
    parser.add_argument('--measure-reference', type=Path)
    parser.add_argument('--metric-output', type=Path)
    parser.add_argument('--power-limit', type=int, default=480,
                        help='Expected measured limit, not a command that changes hardware settings')
    parser.add_argument('--resume-waiting', action='store_true',
                        help='Resume a failed/waiting batch only if dynamic training never began')
    arguments = parser.parse_args()
    if arguments.measure_reference:
        measure_reference(arguments.measure_reference, arguments.metric_output)
    else:
        main(arguments)
