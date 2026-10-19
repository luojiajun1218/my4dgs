"""Resume the explicitly requested five-scene batch after a recovery reboot.

Run as the ordinary project user, under ReConGS. Systemd applies the temporary
power cap outside this script and restores the default when this job exits.
This script never reboots the computer or stores administrator credentials.
"""

from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'output/n3dv_prepare_other5_20261019'
MODELS = ROOT / 'output/n3dv_standard_other5_20261019'
OUTPUT = ROOT / 'output/n3dv_other5_full300_20261019'
SOURCE = Path('/home/gamer-02/Documents/DeepLearning/data/n3dv/my4dgs_standard_20261019')
SCENES = ('cook_spinach', 'cut_roasted_beef', 'sear_steak', 'coffee_martini', 'flame_salmon_1')
sys.path.insert(0, str(AUDIT))
from watcher_io import read_json, write_json


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'automation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    print('Recovery boot supervisor:', boot, flush=True)
    if read_json(OUTPUT / 'batch_status.json').get('status') == 'complete':
        (OUTPUT / 'batch_complete').touch()
        return
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,power.limit', '--format=csv,noheader'],
                         check=True, capture_output=True, text=True)
    if '350.00 W' not in gpu.stdout:
        raise RuntimeError('Recovery batch expects the declared temporary 350 W cap')
    attempt = datetime.now().strftime('%Y%m%dT%H%M%S') + '_' + boot[:8]
    write_json(OUTPUT / ('recovery_launch_' + attempt + '.json'), dict(
        boot_id=boot, started_at=datetime.now().astimezone().isoformat(),
        gpu=gpu.stdout.strip(), power_limit_w=350,
        training_recipe_unchanged=True, administrator_credentials_stored=False))
    # Keep interrupted native attempts for diagnosis; completed models are kept
    # exactly in place. An interrupted first frame is retrained from its seed.
    completed = read_json(AUDIT / 'fresh_frame0_results.json')
    for scene in SCENES:
        directory = MODELS / scene
        if scene not in completed and (directory / 'frame0').exists():
            archive = directory / 'failed_attempts' / attempt
            archive.mkdir(parents=True)
            for name in ('frame0', 'frame0_config.yaml', 'frame0_launch.json', 'frame0_launcher.log'):
                item = directory / name
                if item.exists():
                    shutil.move(str(item), archive / name)
        marker = SOURCE / scene / 'preprocessing_manifest.json'
        manifest = read_json(marker)
        if manifest.get('status') == 'failed':
            manifest['status'] = 'awaiting_resume'
            manifest['resume_queued_at'] = datetime.now().astimezone().isoformat()
            write_json(marker, manifest)
    children = []

    def launch(command, log, threads=4):
        with log.open('a') as file:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=file, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                         OPENBLAS_NUM_THREADS=str(threads)))
        children.append(child)
        return child

    try:
        if not (AUDIT / 'completed_summary.json').is_file():
            launch([sys.executable, '-u', 'scripts/prepare_n3dv_from_videos.py',
                '--raw-root', '/home/gamer-02/Documents/DeepLearning/data/n3dv',
                '--destination-root', str(SOURCE), '--scenes', *SCENES, '--resume'],
                AUDIT / 'prepare_after_reboot.log', threads=16)
            workers = {}
            for name, log in (
                ('validate_as_ready', 'validation'), ('check_alignment_as_ready', 'alignment_watcher'),
                ('seed_support_as_ready', 'seed_support'), ('train_fresh_frame0_as_ready', 'fresh_frame0_training'),
                ('verify_frame0_as_ready', 'frame0_verification'), ('finalize_preparation_audit', 'finalize')):
                workers[name] = launch([sys.executable, '-u', str(AUDIT / (name + '.py'))],
                                       AUDIT / (log + '.log')).pid
            write_json(AUDIT / 'resume_worker_pids.json', workers)
            launched_at = time.time()
            launch([sys.executable, '-u', str(AUDIT / 'monitor_resumed_batch.py')], AUDIT / 'monitor.log')
            for _ in range(60):
                live = AUDIT / 'live_status.json'
                if live.exists() and live.stat().st_mtime >= launched_at:
                    if read_json(live).get('status') == 'running':
                        break
                    raise RuntimeError('Preprocessing recovery failed; inspect live_status.json')
                time.sleep(1)
            else:
                raise TimeoutError('The resumed preprocessing monitor did not become ready')
        batch = launch([sys.executable, '-u', 'scripts/run_n3dv_scenes.py',
                        '--resume-waiting', '--power-limit=350'], OUTPUT / 'batch.log', threads=8)
        while batch.poll() is None:
            time.sleep(20)
        if batch.returncode or read_json(OUTPUT / 'batch_status.json').get('status') != 'complete':
            raise RuntimeError('The five-scene batch has not completed; inspect batch_status.json')
        (OUTPUT / 'batch_complete').touch()
        print('All five 300-frame sequences completed and verified.', flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    main()
