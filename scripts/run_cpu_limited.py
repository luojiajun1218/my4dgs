"""Run one process tree on a small CPU set, pausing it when the CPU gets hot."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def cpu_sensor():
    for directory in Path('/sys/class/hwmon').glob('hwmon*'):
        name = directory / 'name'
        if name.exists() and name.read_text().strip() in ('k10temp', 'coretemp'):
            return directory / 'temp1_input'
    raise RuntimeError('No readable CPU temperature sensor; workload was not started')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def supervise(args):
    sensor = args.sensor or cpu_sensor()
    sensors = [sensor] if args.sensor else sorted(sensor.parent.glob('temp*_input'))
    policy_path = args.policy
    if policy_path is None and args.status.parent.name == 'supervision':
        policy_path = args.status.parent.parent / 'resource_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path and policy_path.exists() else {}
    cpus = set(policy.get('cpus', [int(value) for value in args.cpus.split(',')]))
    threads = int(policy.get('library_threads', 1))
    if threads < 1:
        raise ValueError('Library thread count must be positive')
    if not cpus or not cpus <= os.sched_getaffinity(0):
        raise ValueError('Requested CPU set is unavailable')
    if not args.resume_below < args.pause_at < args.abort_at:
        raise ValueError('Expected resume < pause < abort temperatures')
    os.sched_setaffinity(0, cpus)
    environment = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'N3DV_CPU_THREADS'):
        environment[name] = str(threads)
    environment.update(MAX_JOBS='1', MAGICK_THREAD_LIMIT='1')
    environment['N3DV_DATALOADER_WORKERS'] = '0'
    environment['N3DV_THERMAL_EVENTS'] = str(args.status.with_suffix('.events.jsonl').resolve())
    if args.status.parent.name == 'supervision':
        environment['N3DV_GPU_LOCK'] = str((args.status.parent.parent / 'gpu.lock').resolve())
    args.status.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='waiting_for_cooling', cpus=sorted(cpus), pause_at_c=args.pause_at,
                 library_threads=threads, resource_policy_file=str(policy_path) if policy else None,
                 resume_below_c=args.resume_below, abort_at_c=args.abort_at,
                 sensor=str(sensor), command=args.command, pause_count=0, paused_seconds=0.0,
                 cpu_temperature_inputs=[str(path) for path in sensors],
                 started_at=datetime.now().astimezone().isoformat())
    events = args.status.with_suffix('.events.jsonl')
    samples = args.status.with_suffix('.temperatures.csv')
    if not samples.exists():
        samples.write_text('timestamp,cpu_temperature_c,status\n')
    child = None
    paused_at = None
    max_temperature = 0.0
    next_gpu_check = 0.0
    background_pause = args.status.parent.parent / 'pause_background_for_benchmark.json'
    gpu_queue = any(Path(token).name == 'run_n3dv_scenes.py' for token in args.command)

    def benchmark_requests_pause():
        if gpu_queue or not background_pause.exists():
            return False
        try:
            owner = json.loads(background_pause.read_text()).get('owner_pid')
        except FileNotFoundError:
            return False
        return bool(owner and Path('/proc' + '/' + str(owner)).exists())

    def event(kind, **fields):
        with events.open('a') as file:
            file.write(json.dumps(dict(at=datetime.now().astimezone().isoformat(),
                                       kind=kind, **fields)) + '\n')

    def group_signal(kind):
        if child is not None:
            try:
                os.killpg(child.pid, kind)
            except ProcessLookupError:
                pass

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Supervisor received signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while True:
            temperature = max(int(path.read_text()) / 1000 for path in sensors)
            if not 0 < temperature < 150:
                raise RuntimeError('CPU temperature sensor returned an invalid reading')
            max_temperature = max(max_temperature, temperature)
            state.update(temperature_c=temperature, max_temperature_c=max_temperature,
                         updated_at=datetime.now().astimezone().isoformat())
            if temperature >= args.abort_at:
                raise RuntimeError(f'CPU reached the conservative abort threshold: {temperature:.1f} C')
            if child is None:
                if temperature < args.pause_at and not benchmark_requests_pause():
                    child = subprocess.Popen(args.command, env=environment, start_new_session=True)
                    state.update(status='running', child_pid=child.pid)
                    event('started', child_pid=child.pid, temperature_c=temperature)
            elif child.poll() is not None:
                state.update(status='complete' if child.returncode == 0 else 'failed',
                             returncode=child.returncode)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, args.command)
                break
            elif paused_at is None and (temperature >= args.pause_at or benchmark_requests_pause()):
                group_signal(signal.SIGSTOP)
                paused_at = time.monotonic()
                reason = 'benchmark' if benchmark_requests_pause() else 'temperature'
                state.update(status='paused_for_benchmark' if reason == 'benchmark' else 'paused_for_cooling',
                             pause_count=state['pause_count'] + 1)
                event('paused', temperature_c=temperature, reason=reason)
            elif paused_at is not None and temperature <= args.resume_below and not benchmark_requests_pause():
                seconds = time.monotonic() - paused_at
                state['paused_seconds'] += seconds
                group_signal(signal.SIGCONT)
                paused_at = None
                state['status'] = 'running'
                event('resumed', paused_seconds=seconds, temperature_c=temperature)
            if args.require_gpu and time.monotonic() >= next_gpu_check:
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                     capture_output=True, text=True, timeout=5)
                if gpu.returncode or not gpu.stdout.strip():
                    raise RuntimeError('GPU unavailable; stopping this process tree')
                next_gpu_check = time.monotonic() + 15
            write(args.status, state)
            with samples.open('a') as file:
                file.write(f"{state['updated_at']},{temperature:.3f},{state['status']}\n")
            time.sleep(args.interval)
    except BaseException as error:
        state.update(status='stopped' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        event('stopped', error=repr(error))
        raise
    finally:
        group_signal(signal.SIGTERM)
        group_signal(signal.SIGCONT)
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                group_signal(signal.SIGKILL)
                child.wait()
        state['finished_at'] = datetime.now().astimezone().isoformat()
        write(args.status, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpus', default='0,1')
    parser.add_argument('--pause-at', type=float, default=70)
    parser.add_argument('--resume-below', type=float, default=55)
    parser.add_argument('--abort-at', type=float, default=80)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--sensor', type=Path)
    parser.add_argument('--policy', type=Path,
                        help='CPU/library policy read once at this job boundary; thermal thresholds remain enforced')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    if arguments.command[:1] == ['--']:
        arguments.command.pop(0)
    if not arguments.command or arguments.interval <= 0:
        parser.error('A command and a positive sampling interval are required')
    supervise(arguments)
