"""Serial N3DV preprocessing, fresh initialization and five full training runs.

Every compute subprocess inherits a bounded CPU affinity and a temperature
supervisor. This launcher never changes power limits or installs boot services.
"""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'output/n3dv_preparation/audit'
FRAME0 = ROOT / 'output/n3dv_preparation/frame0'
SOURCE = ROOT.parent / 'data/n3dv/my4dgs_standard_20261019'
SCENES = ('cook_spinach', 'cut_roasted_beef', 'sear_steak', 'coffee_martini', 'flame_salmon_1')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'low_cpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = output / 'supervisor_status.json'
    if status.exists():
        raise FileExistsError('Use a new output directory or inspect the previous supervisor progress')
    state = dict(status='starting', pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 resource_policy_file=str(output / 'resource_policy.json'),
                 cpus=args.cpus, cpu_threads=1, dataloader_workers=0,
                 pause_at_c=70, resume_below_c=55, abort_at_c=80,
                 gpu_power_limit_w=480, completed_preparation=[],
                 started_at=datetime.now().astimezone().isoformat(),
                 authorization='Author requested continuation on 2026-10-20 with reduced CPU usage.')

    def update(**fields):
        state.update(fields, updated_at=datetime.now().astimezone().isoformat())
        write(status, state)
        live = dict(state)
        live['automatic_continuation_enabled'] = state['status'] not in ('failed', 'stopped', 'complete')
        write(AUDIT / 'live_status.json', live)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Stop requested: signal ' + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    active = None

    def run(command, stage, scene=None):
        nonlocal active
        prefix = output / 'supervision' / ((scene + '_' if scene else '') + stage)
        prefix.parent.mkdir(exist_ok=True)
        thermal = prefix.with_suffix('.json')
        wrapper = [sys.executable, '-u', str(ROOT / 'scripts/run_cpu_limited.py'),
                   '--cpus', args.cpus, '--status', str(thermal), '--require-gpu', '--', *command]
        update(status=stage, scene=scene, thermal_status=str(thermal), command=command)
        print(json.dumps(dict(stage=stage, scene=scene), ensure_ascii=False), flush=True)
        with prefix.with_suffix('.log').open('w') as log:
            active = subprocess.Popen(wrapper, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      env=dict(os.environ, N3DV_ONESHOT='1'))
            update(child_pid=active.pid)
            while active.poll() is None:
                observed = read(thermal)
                update(temperature_c=observed.get('temperature_c'),
                       thermal_state=observed.get('status'),
                       max_temperature_c=observed.get('max_temperature_c'))
                time.sleep(5)
            code = active.returncode
            active = None
            if code:
                raise subprocess.CalledProcessError(code, wrapper)

    try:
        # Real CUDA allocation confirms more than mere NVML visibility after reboot.
        run([sys.executable, '-c', "import torch; x=torch.ones(16,device='cuda'); assert x.sum().item()==16; print(torch.cuda.get_device_name())"], 'cuda_check')
        for scene in SCENES:
            run([sys.executable, '-u', str(ROOT / 'scripts/prepare_n3dv_from_videos.py'),
                 '--raw-root', str(ROOT.parent / 'data/n3dv'), '--destination-root', str(SOURCE),
                 '--scenes', scene, '--resume', '--threads=1', '--image-workers=1'], 'prepare', scene)
            for worker in ('validate_as_ready', 'check_alignment_as_ready', 'seed_support_as_ready',
                           'train_fresh_frame0_as_ready', 'verify_frame0_as_ready'):
                run([sys.executable, '-u', str(AUDIT / (worker + '.py'))], worker, scene)
            ready = read(AUDIT / 'ready_scenes.json')
            for filename in ('frame_alignment_checks.json', 'frame0_my4dgs_checks.json'):
                if read(AUDIT / filename).get(scene, {}).get('status') != 'passed':
                    raise ValueError('Incomplete verification for ' + scene + ': ' + filename)
            result = read(AUDIT / 'fresh_frame0_results.json')[scene]
            if hashlib.sha256((FRAME0 / scene / 'frame0.pth').read_bytes()).hexdigest() != result['checkpoint_sha256']:
                raise ValueError('Completed first-frame checkpoint changed: ' + scene)
            if ready['ready_scenes'].get(scene, {}).get('status') != 'passed':
                raise ValueError('Training input not ready for ' + scene)
            ready['automatic_continuation_enabled'] = True
            ready['status'] = 'complete' if set(ready['ready_scenes']) == set(SCENES) else 'in_progress'
            write(AUDIT / 'ready_scenes.json', ready)
            state['completed_preparation'].append(scene)
            update()
        run([sys.executable, '-u', str(AUDIT / 'finalize_preparation_audit.py')], 'publish_preparation')
        run([sys.executable, '-u', str(ROOT / 'scripts/run_n3dv_scenes.py'),
             '--preparation-audit', str(AUDIT), '--frame0-root', str(FRAME0),
             '--output', str(output), '--power-limit=480'], 'full300_training_and_verification')
        batch = read(output / 'batch_status.json')
        results = read(output / 'results.json')
        if batch.get('status') != 'complete' or set(results) != set(SCENES):
            raise ValueError('All five scenes must finish and verify before marking this task complete')
        update(status='complete', completed_scenes=list(results), child_pid=None,
               completed_at=datetime.now().astimezone().isoformat())
    except BaseException as error:
        update(status='stopped' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
        raise
    finally:
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=25)
            except subprocess.TimeoutExpired:
                active.kill()
                active.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/n3dv' /
                        ('_batch_' + datetime.now().strftime('%Y%m%dT%H%M%S') + '_half_final480w'))
    parser.add_argument('--cpus', default='0,1')
    main(parser.parse_args())
