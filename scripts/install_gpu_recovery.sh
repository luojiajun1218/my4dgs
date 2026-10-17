#!/bin/bash
# Install a single recovery command; installation never reboots the machine.
set -euo pipefail
[[ ${EUID} -eq 0 ]] || { printf '%s\n' 'Run this installer with sudo.' >&2; exit 1; }
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
helper="$source_dir/gpu_recover.py"
[[ -f "$helper" && $(id -u gamer-02) == 1000 ]]
[[ ! -e /var/lib/gpu-recovery/pending.json ]] || { printf '%s\n' 'A recovery reboot is already pending; review it before reinstalling.' >&2; exit 1; }
/usr/sbin/sshd -t
install -d -o root -g root -m 0755 /usr/local/libexec/gpu-recovery /etc/gpu-recovery
install -d -o root -g root -m 0700 /var/lib/gpu-recovery /var/log/gpu-recovery
install -o root -g root -m 0755 "$helper" /usr/local/libexec/gpu-recovery/recovery.py
cat > /etc/gpu-recovery/config.json <<'JSON'
{
  "user": "gamer-02",
  "uid": 1000,
  "tailscale_ip": "100.124.74.30",
  "wifi_profiles": ["106_5G", "106_2.4G"],
  "proxy_state_dir": "/var/lib/clash-verge-service",
  "proxy_run_dir": "/run/clash-verge-service",
  "proxy_port": 7897,
  "gpu_pci": "21:00.0",
  "upstream_pci": "20:03.1"
}
JSON
chmod 0644 /etc/gpu-recovery/config.json
cat > /usr/local/bin/gpu-recover <<'SH'
#!/bin/sh
exec /usr/bin/python3 /usr/local/libexec/gpu-recovery/recovery.py "$@"
SH
chmod 0755 /usr/local/bin/gpu-recover
install -d -o root -g root -m 0755 /etc/systemd/system/clash-verge-service.service.d
cat > /etc/systemd/system/clash-verge-service.service.d/20-create-owner-ipc.conf <<'UNIT'
[Service]
ExecStartPre=/usr/bin/install -d -o root -g root -m 0755 /run/clash-verge-service/users
ExecStartPre=/usr/bin/install -d -o root -g root -m 0700 /run/clash-verge-service/users/1000
UNIT
cat > /etc/systemd/system/clash-verge-service.service.d/30-gpu-recovery-restore.conf <<'UNIT'
[Service]
# Ignore a preparation failure so the vendor IPC server can still start.
ExecStartPre=-/usr/bin/python3 /usr/local/libexec/gpu-recovery/recovery.py --prepare-boot
UNIT
cat > /etc/systemd/system/gpu-recovery-boot-check.service <<'UNIT'
[Unit]
Description=Check SSH and Tailscale after an operator-requested GPU recovery reboot
Wants=network-online.target ssh.service tailscaled.service clash-verge-service.service
After=network-online.target ssh.service tailscaled.service clash-verge-service.service
ConditionPathExists=/var/lib/gpu-recovery/pending.json
StartLimitIntervalSec=0

[Service]
Type=simple
ExecStart=/usr/bin/python3 /usr/local/libexec/gpu-recovery/recovery.py --boot-check
Restart=on-failure
RestartSec=60
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
UNIT
chmod 0644 /etc/systemd/system/clash-verge-service.service.d/20-create-owner-ipc.conf /etc/systemd/system/clash-verge-service.service.d/30-gpu-recovery-restore.conf /etc/systemd/system/gpu-recovery-boot-check.service
# Validate the installed configuration and saved proxy state before arming anything.
/usr/bin/python3 - <<'PY'
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('gpu_recovery', '/usr/local/libexec/gpu-recovery/recovery.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.snapshot_proxy(module.load_json(module.CONFIG))
assert module.boot_wifi_available(module.load_json(module.CONFIG)), 'No system Wi-Fi credentials available before desktop login.'
assert not module.PENDING.exists(), 'An existing recovery is pending; review it before reinstalling.'
PY
systemd-analyze verify /etc/systemd/system/gpu-recovery-boot-check.service clash-verge-service.service ssh.service tailscaled.service
systemctl daemon-reload
systemctl enable --now NetworkManager.service ssh.service ssh.socket tailscaled.service clash-verge-service.service
systemctl enable gpu-recovery-boot-check.service
printf '%s\n' 'Installed: gpu-recover. No reboot was scheduled and live proxy/VPN services were not restarted.'
/usr/local/bin/gpu-recover --check
