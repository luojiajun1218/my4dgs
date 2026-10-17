"""Exercise recovery state without root, host service writes, or reboot requests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'gpu_recovery', Path(__file__).resolve().parents[1] / 'scripts/gpu_recover.py')
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / 'state'
        self.state.mkdir()
        self.pending = self.state / 'pending.json'
        self.vendor = self.root / 'vendor'
        runtime = self.vendor / 'users/1000/runtime'
        runtime.mkdir(parents=True)
        (runtime / 'config.yaml').write_text('mixed-port: 7897\n')
        core = self.vendor / 'cores/verge-mihomo'
        core.parent.mkdir()
        core.write_text('test fixture, never executed')
        self.config = {'uid': 1000, 'user': 'gamer-02',
                       'proxy_state_dir': str(self.vendor),
                       'proxy_run_dir': str(self.root / 'run')}
        self.owner = {'owner_key': '1000', 'identity': {'Unix': {'uid': 1000, 'gid': 1000}}}
        self.desired = {'core_should_be_running': True, 'generation': 7,
                        'last_clash_config': {'core_config': {
                            'core_path': str(core), 'config_path': str(runtime / 'config.yaml'),
                            'config_dir': str(runtime),
                            'core_ipc_path': str(self.root / 'run/users/1000/verge-mihomo.sock')}}}
        self.pending_data = {'origin_boot_id': 'old-boot', 'proxy_snapshot': {
            'active_owner': self.owner, 'desired_state': self.desired}}
        self.patchers = [patch.object(RECOVERY, 'STATE', self.state),
                         patch.object(RECOVERY, 'PENDING', self.pending),
                         patch.object(RECOVERY, 'boot_id', return_value='new-boot')]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def arm(self):
        RECOVERY.atomic_json(self.pending, self.pending_data)

    def test_current_boot_does_not_replay_pending_state(self):
        self.pending_data['origin_boot_id'] = 'new-boot'
        self.arm()
        self.assertFalse(RECOVERY.prepare_boot(self.config))
        self.assertFalse((self.vendor / 'active-owner.json').exists())

    def test_new_boot_recovers_owner_and_config_after_gui_cleanup(self):
        self.arm()
        self.assertTrue(RECOVERY.prepare_boot(self.config))
        self.assertEqual(RECOVERY.load_json(self.vendor / 'active-owner.json'), self.owner)
        desired = RECOVERY.load_json(self.vendor / 'users/1000/desired-state.json')
        self.assertTrue(desired['core_should_be_running'])
        self.assertEqual(desired['last_clash_config'], self.desired['last_clash_config'])
        self.assertEqual((self.vendor / 'active-owner.json').stat().st_mode & 0o777, 0o600)

    def test_valid_newer_config_is_preserved_and_replay_runs_once(self):
        self.arm()
        RECOVERY.atomic_json(self.vendor / 'active-owner.json', self.owner)
        current = {**self.desired, 'core_should_be_running': False, 'generation': 8}
        path = self.vendor / 'users/1000/desired-state.json'
        RECOVERY.atomic_json(path, current)
        self.assertTrue(RECOVERY.prepare_boot(self.config))
        self.assertEqual(RECOVERY.load_json(path)['generation'], 8)
        self.assertTrue(RECOVERY.load_json(path)['core_should_be_running'])
        RECOVERY.atomic_json(path, current)
        self.assertFalse(RECOVERY.prepare_boot(self.config))
        self.assertFalse(RECOVERY.load_json(path)['core_should_be_running'])

    def test_invalid_core_path_cannot_be_restored(self):
        self.pending_data['proxy_snapshot']['desired_state']['last_clash_config']['core_config']['core_path'] = '/tmp/unapproved-core'
        self.arm()
        with self.assertRaises(ValueError):
            RECOVERY.prepare_boot(self.config)
        self.assertFalse((self.vendor / 'active-owner.json').exists())

    def test_remote_ssh_readiness_does_not_require_a_working_gpu(self):
        status = dict(sshd_config_ok=True, ssh_banner_ok=True, default_route='wifi',
                      proxy_port_open=True, tailscale_running=True, tailscale_online=True,
                      gpu={'code': 1, 'output': 'GPU is lost'})
        self.assertTrue(RECOVERY.remote_ready(status))
        status['tailscale_online'] = False
        self.assertFalse(RECOVERY.remote_ready(status))

    def test_check_mode_never_arms_or_schedules_a_reboot(self):
        config_path = self.root / 'config.json'
        RECOVERY.atomic_json(config_path, self.config)
        with patch.object(RECOVERY, 'CONFIG', config_path), \
             patch.object(RECOVERY, 'ensure_root'), \
             patch.object(RECOVERY, 'status_now', return_value={'remote_checks_ready': True}), \
             patch.object(RECOVERY, 'print_status'), \
             patch.object(RECOVERY, 'start_recovery') as start:
            self.assertEqual(RECOVERY.main(['--check']), 0)
            start.assert_not_called()
        self.assertFalse(self.pending.exists())

    def test_collect_only_never_arms_or_schedules_a_reboot(self):
        config_path = self.root / 'config.json'
        RECOVERY.atomic_json(config_path, self.config)
        with patch.object(RECOVERY, 'CONFIG', config_path), \
             patch.object(RECOVERY, 'ensure_root'), \
             patch.object(RECOVERY, 'collect') as collect, \
             patch.object(RECOVERY, 'start_recovery') as start:
            self.assertEqual(RECOVERY.main(['--collect-only']), 0)
            collect.assert_called_once_with(self.config)
            start.assert_not_called()
        self.assertFalse(self.pending.exists())

    def test_hung_diagnostic_is_bounded(self):
        started = time.monotonic()
        result = RECOVERY.capture([sys.executable, '-c', 'import time; time.sleep(60)'], timeout=.1)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['code'], 124)
        self.assertLess(time.monotonic() - started, 4)

    def test_missing_diagnostic_tool_is_reported_without_stopping_recovery(self):
        result = RECOVERY.capture([str(self.root / 'nonexistent-tool')])
        self.assertEqual(result['code'], 127)
        self.assertFalse(result['timed_out'])

    def test_failed_shutdown_restores_the_previous_pending_record(self):
        self.arm()
        previous = RECOVERY.load_json(self.pending)

        def fake_capture(command, timeout=10):
            return {'code': 1 if command[0] == 'shutdown' else 0,
                    'output': 'simulated response', 'timed_out': False}

        with patch.object(RECOVERY, 'capture', side_effect=fake_capture), \
             patch.object(RECOVERY, 'boot_wifi_available', return_value=True), \
             patch.object(RECOVERY, 'snapshot_proxy', return_value=previous['proxy_snapshot']), \
             patch.object(RECOVERY, 'collect', return_value=self.root), \
             patch.object(RECOVERY.os, 'sync'):
            with self.assertRaisesRegex(RuntimeError, 'reboot request failed'):
                RECOVERY.start_recovery({**self.config, 'tailscale_ip': '100.0.0.1'})
        self.assertEqual(RECOVERY.load_json(self.pending), previous)

    def test_successful_recovery_saves_proxy_state_and_schedules_one_reboot(self):
        snapshot = self.pending_data['proxy_snapshot']
        with patch.object(RECOVERY, 'capture', return_value={'code': 0, 'output': ''}) as commands, \
             patch.object(RECOVERY, 'boot_wifi_available', return_value=True), \
             patch.object(RECOVERY, 'snapshot_proxy', return_value=snapshot), \
             patch.object(RECOVERY, 'collect', return_value=self.root), \
             patch.object(RECOVERY.os, 'sync'), patch('builtins.print'):
            RECOVERY.start_recovery({**self.config, 'tailscale_ip': '100.0.0.1'})
        pending = RECOVERY.load_json(self.pending)
        self.assertEqual(pending['proxy_snapshot'], snapshot)
        self.assertEqual(pending['origin_boot_id'], 'new-boot')
        shutdowns = [call.args[0] for call in commands.call_args_list if call.args[0][0] == 'shutdown']
        self.assertEqual(len(shutdowns), 1)
        self.assertEqual(shutdowns[0][:3], ['shutdown', '-r', '+1'])

    def test_reboot_is_refused_when_boot_wifi_is_not_configured(self):
        with patch.object(RECOVERY, 'capture', return_value={'code': 0}) as commands, \
             patch.object(RECOVERY, 'boot_wifi_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Wi-Fi'):
                RECOVERY.start_recovery(self.config)
        self.assertEqual(commands.call_count, 1)
        self.assertEqual(commands.call_args.args[0], ['/usr/sbin/sshd', '-t'])
        self.assertFalse(self.pending.exists())

    def test_boot_report_marks_ssh_ready_even_if_gpu_is_still_lost(self):
        self.pending_data['diagnostics'] = str(self.root / 'diagnostics')
        self.arm()
        ready = {'remote_checks_ready': True, 'ssh_commands': ['ssh gamer-02@100.0.0.1']}
        with patch.object(RECOVERY, 'status_now', return_value=ready), \
             patch.object(RECOVERY, 'capture', return_value={'code': 1, 'output': 'GPU is lost'}), \
             patch.object(RECOVERY, 'print_status'):
            self.assertEqual(RECOVERY.boot_check(self.config), 0)
        report = RECOVERY.load_json(self.state / 'last-boot.json')
        self.assertTrue(report['remote_checks_ready'])
        self.assertEqual(report['gpu']['code'], 1)
        self.assertFalse(self.pending.exists())


if __name__ == '__main__':
    unittest.main()
