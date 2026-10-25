#!/usr/bin/python3
"""Capture GPU failures and reboot into ordinary OpenSSH over Tailscale.

Installed by install_gpu_recovery.sh. The default action schedules a reboot;
--check, --status and --collect-only never schedule one. No password is stored.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

CONFIG = Path('/etc/gpu-recovery/config.json')
STATE = Path('/var/lib/gpu-recovery')
LOGS = Path('/var/log/gpu-recovery')
PENDING = STATE / 'pending.json'
SERVICES = ('NetworkManager.service', 'ssh.service', 'ssh.socket',
            'clash-verge-service.service', 'tailscaled.service')


def now():
    return datetime.now().astimezone().isoformat()


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def capture(args, timeout=10):
    """Bound even NVIDIA diagnostics whose children can hang after GPU loss."""
    try:
        child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, start_new_session=True)
    except OSError as error:
        return {'code': 127, 'output': str(error), 'timed_out': False}
    try:
        output, _ = child.communicate(timeout=timeout)
        return {'code': child.returncode, 'output': output, 'timed_out': False}
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            output, _ = child.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            output = error.output or ''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
        return {'code': 124, 'output': output, 'timed_out': True}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=path.name + '.tmp.', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def validate_proxy(owner, desired, config):
    uid = config['uid']
    vendor = Path(config['proxy_state_dir'])
    identity = owner.get('identity', {}).get('Unix', {})
    if owner.get('owner_key') != str(uid) or identity.get('uid') != uid:
        raise ValueError('Clash active owner does not match the configured account')
    core = (desired.get('last_clash_config') or {}).get('core_config') or {}
    runtime = vendor / 'users' / str(uid) / 'runtime'
    expected_ipc = Path(config['proxy_run_dir']) / 'users' / str(uid) / 'verge-mihomo.sock'
    if (Path(core.get('config_dir', '')) != runtime
            or Path(core.get('config_path', '')) != runtime / 'config.yaml'
            or Path(core.get('core_ipc_path', '')) != expected_ipc):
        raise ValueError('Unsupported Clash runtime paths; recovery was not armed')
    executable = Path(core.get('core_path', ''))
    if executable.resolve().parent != (vendor / 'cores').resolve():
        raise ValueError('Clash core is outside the administrator-approved directory')
    if not executable.is_file() or not (runtime / 'config.yaml').is_file():
        raise ValueError('The saved Clash core or runtime config is missing')


def snapshot_proxy(config):
    vendor = Path(config['proxy_state_dir'])
    owner = load_json(vendor / 'active-owner.json')
    desired = load_json(vendor / 'users' / str(config['uid']) / 'desired-state.json')
    validate_proxy(owner, desired, config)
    return {'active_owner': owner, 'desired_state': desired}


def boot_wifi_available(config):
    """Check stored system Wi-Fi credentials without printing or copying them."""
    for profile in config['wifi_profiles']:
        metadata = capture(['nmcli', '-g', 'connection.permissions,connection.autoconnect,802-11-wireless-security.psk-flags',
                            'connection', 'show', profile], timeout=5)
        if metadata['code'] or metadata['output'].splitlines() != ['', 'yes', '0']:
            continue
        secret = capture(['nmcli', '--show-secrets', '-g', '802-11-wireless-security.psk',
                          'connection', 'show', profile], timeout=5)
        if secret['code'] == 0 and secret['output'].strip():
            return True
    return False


def prepare_boot(config):
    """Replay saved proxy intent only after a recovery reboot, once per boot."""
    if not PENDING.exists():
        return False
    pending = load_json(PENDING)
    current_boot = boot_id()
    if (pending['origin_boot_id'] == current_boot
            or pending.get('proxy_prepared_boot_id') == current_boot):
        return False
    snapshot = pending['proxy_snapshot']
    vendor = Path(config['proxy_state_dir'])
    owner_path = vendor / 'active-owner.json'
    desired_path = vendor / 'users' / str(config['uid']) / 'desired-state.json'
    owner = load_json(owner_path) if owner_path.exists() else snapshot['active_owner']
    try:
        desired = load_json(desired_path)
        validate_proxy(owner, desired, config)
    except (OSError, ValueError, KeyError, TypeError):
        owner, desired = snapshot['active_owner'], snapshot['desired_state']
    validate_proxy(owner, desired, config)
    # GUI shutdown may have recorded "stopped" after the operator requested recovery.
    # Keep its most recent valid paths/configuration; restore just the run intent.
    desired = {**desired, 'core_should_be_running': True}
    atomic_json(owner_path, owner)
    atomic_json(desired_path, desired)
    pending['proxy_prepared_boot_id'] = current_boot
    atomic_json(PENDING, pending)
    print('Recovery boot: restored saved Clash core startup intent.', flush=True)
    return True


def port_open(port, banner=False):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=2) as connection:
            return connection.recv(128).startswith(b'SSH-') if banner else True
    except OSError:
        return False


def remote_ready(status):
    # Local prechecks, not proof of login from an external client. Retain relay
    # warnings in the report. GPU availability must not gate SSH recovery.
    return bool(status['sshd_config_ok'] and status['ssh_banner_ok']
                and status['default_route'] and status['proxy_port_open']
                and status['tailscale_running'] and status['tailscale_online'])


def status_now(config, include_gpu=True):
    ssh = capture(['/usr/sbin/sshd', '-t'])
    route = capture(['ip', '-4', 'route', 'show', 'default'])
    tail = capture(['tailscale', 'status', '--json'], timeout=5)
    try:
        ts = json.loads(tail['output']) if tail['code'] == 0 else {}
    except ValueError:
        ts = {}
    addresses = [x for x in ts.get('TailscaleIPs', []) if ':' not in x]
    addresses = addresses or [config['tailscale_ip']]
    result = dict(at=now(), boot_id=boot_id(), sshd_config_ok=ssh['code'] == 0,
                  ssh_banner_ok=port_open(22, banner=True),
                  proxy_port_open=port_open(config['proxy_port']),
                  default_route=route['output'].strip() if route['code'] == 0 else '',
                  tailscale_running=ts.get('BackendState') == 'Running',
                  tailscale_online=ts.get('Self', {}).get('Online') is True,
                  tailscale_health=ts.get('Health') or [],
                  ssh_commands=[f'ssh {config["user"]}@{address}' for address in addresses])
    if include_gpu:
        result['gpu'] = capture(['nvidia-smi', '--query-gpu=name,power.limit', '--format=csv,noheader'])
    result['remote_checks_ready'] = remote_ready(result)
    return result


def print_status(status):
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


def collect(config):
    directory = LOGS / (datetime.now().strftime('%Y%m%dT%H%M%S%f') + '-' + boot_id()[:8])
    directory.mkdir(parents=True, mode=0o700, exist_ok=False)
    jobs = {
        'kernel.log': (['journalctl', '-k', '-b', '-n', '3000', '--no-pager'], 15),
        'nvidia-smi.log': (['nvidia-smi', '-q'], 10),
        'pci-gpu.log': (['lspci', '-vv', '-s', config['gpu_pci']], 5),
        'pci-upstream.log': (['lspci', '-vv', '-s', config['upstream_pci']], 5),
        'services.log': (['systemctl', 'show', *SERVICES, '-p', 'Id', '-p', 'ActiveState', '-p', 'UnitFileState'], 10),
        'tailscale.log': (['tailscale', 'status'], 5),
    }
    command_status = {}
    for name, (args, timeout) in jobs.items():
        result = capture(args, timeout)
        (directory / name).write_text(result['output'])
        command_status[name] = {key: result[key] for key in ('code', 'timed_out')}
    report = Path('/usr/bin/nvidia-bug-report.sh')
    if report.is_file():
        # NVIDIA's script appends .gz itself.
        result = capture([str(report), '--safe-mode', '--output-file', str(directory / 'nvidia-bug-report.log')], timeout=60)
        (directory / 'bug-report-command.log').write_text(result['output'])
        command_status['bug_report'] = {key: result[key] for key in ('code', 'timed_out')}
    atomic_json(directory / 'collection.json', {'at': now(), 'boot_id': boot_id(), 'commands': command_status})
    print(f'Diagnostics saved: {directory}', flush=True)
    return directory


def start_recovery(config):
    # These preconditions prevent a scripted reboot with a broken SSH daemon.
    if capture(['/usr/sbin/sshd', '-t'])['code']:
        raise RuntimeError('SSH configuration check failed; reboot was not scheduled')
    if not boot_wifi_available(config):
        raise RuntimeError('No configured system Wi-Fi profile has stored boot credentials; reboot was not scheduled')
    enabled = capture(['systemctl', 'enable', '--now', *SERVICES], timeout=30)
    if enabled['code']:
        raise RuntimeError('Could not enable remote access services; reboot was not scheduled: ' + enabled['output'])
    proxy = snapshot_proxy(config)
    directory = collect(config)
    previous = load_json(PENDING) if PENDING.exists() else None
    pending = dict(at=now(), origin_boot_id=boot_id(), diagnostics=str(directory),
                   proxy_snapshot=proxy, expected_ssh=f'ssh {config["user"]}@{config["tailscale_ip"]}')
    atomic_json(PENDING, pending)
    # Read both previous diagnostics and the freshly armed recovery record after boot.
    os.sync()
    scheduled = capture(['shutdown', '-r', '+1', 'GPU recovery: diagnostics saved; remote access services enabled.'])
    if scheduled['code']:
        if previous is None:
            PENDING.unlink(missing_ok=True)
        else:
            atomic_json(PENDING, previous)
        raise RuntimeError('The reboot request failed: ' + scheduled['output'])
    print('Reboot scheduled in one minute. Running jobs will stop.', flush=True)
    print(f'After boot: {pending["expected_ssh"]}', flush=True)
    print('Cancel before reboot: gpu-recover --cancel', flush=True)


def boot_check(config):
    if not PENDING.exists():
        return 0
    pending = load_json(PENDING)
    if pending['origin_boot_id'] == boot_id():
        print('Recovery is armed for the next boot; not checking the current boot.')
        return 0
    deadline = time.monotonic() + 600
    proxy_retry = time.monotonic() + 60
    while True:
        status = status_now(config, include_gpu=False)
        status.update(origin_boot_id=pending['origin_boot_id'], diagnostics=pending['diagnostics'])
        if status['remote_checks_ready'] or time.monotonic() >= deadline:
            status['gpu'] = capture(['nvidia-smi', '--query-gpu=name,power.limit', '--format=csv,noheader'])
            atomic_json(STATE / 'last-boot.json', status)
            print_status(status)
            if status['remote_checks_ready']:
                atomic_json(STATE / 'last-recovery.json', {key: value for key, value in pending.items() if key != 'proxy_snapshot'})
                PENDING.unlink(missing_ok=True)
                return 0
            return 1
        atomic_json(STATE / 'last-boot.json', status)
        capture(['systemctl', 'start', 'ssh.service', 'tailscaled.service'], timeout=15)
        if not status['proxy_port_open'] and time.monotonic() >= proxy_retry:
            capture(['systemctl', 'restart', 'clash-verge-service.service'], timeout=20)
            proxy_retry = time.monotonic() + 120
        # Tailscale already retries coordination/relay connections itself. Do not
        # restart a live VPN just because a remote relay is temporarily unhealthy.
        time.sleep(15)


def ensure_root():
    if os.geteuid() != 0:
        os.execvp('sudo', ['sudo', '/usr/local/bin/gpu-recover', *sys.argv[1:]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true', help='Check SSH/VPN readiness without rebooting')
    actions.add_argument('--status', action='store_true', help='Show current readiness and the last boot report')
    actions.add_argument('--collect-only', action='store_true', help='Collect diagnostics without rebooting')
    actions.add_argument('--cancel', action='store_true', help='Cancel the scheduled recovery reboot')
    actions.add_argument('--prepare-boot', action='store_true', help=argparse.SUPPRESS)
    actions.add_argument('--boot-check', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    ensure_root()
    os.umask(0o077)
    config = load_json(CONFIG)
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.prepare_boot:
        prepare_boot(config)
        return 0
    if args.boot_check:
        return boot_check(config)
    if args.check or args.status:
        status = status_now(config)
        if args.status and (STATE / 'last-boot.json').exists():
            status['last_boot_report'] = load_json(STATE / 'last-boot.json')
        print_status(status)
        return 0 if status['remote_checks_ready'] else 2
    # Serialize only operations that arm/cancel a reboot or capture a new report.
    with (STATE / 'operation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.cancel:
            result = capture(['shutdown', '-c'])
            if result['code']:
                raise RuntimeError('Could not cancel a shutdown: ' + result['output'])
            if PENDING.exists() and load_json(PENDING)['origin_boot_id'] == boot_id():
                PENDING.unlink()
            print('Recovery reboot cancelled. Diagnostics were retained.')
        elif args.collect_only:
            collect(config)
        else:
            start_recovery(config)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'gpu-recover: {error}', file=sys.stderr)
        raise SystemExit(1)
