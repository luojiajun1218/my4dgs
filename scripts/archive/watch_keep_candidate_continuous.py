"""Watch one authorized run; stop only under its predeclared validation rule."""
import argparse
import json
import math
import os
import signal
import statistics
import time
from pathlib import Path

import yaml


def load_frames(run):
    frames = {}
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        try:
            packet = json.loads(path.read_text())
            if packet['frame'] > 0 and 'wall' in packet['timing']:
                frames[packet['frame']] = packet
        except (json.JSONDecodeError, KeyError):
            # A summary may be between its initial and final write.
            continue
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--watch-config', type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.watch_config.read_text())
    run_cfg = yaml.safe_load((args.run / 'config.yaml').read_text())
    expected_config = f"--config=config/{run_cfg['experiment_name']}.yaml"
    anchors = {name: load_frames(Path(cfg[name])) for name in ('original_core', 'keep_probe')}
    output = args.run / 'analysis' / 'watch'
    output.mkdir(parents=True, exist_ok=True)
    checks, consecutive, announced, seen = [], 0, 0, set()
    stop = None
    while True:
        frames = load_frames(args.run)
        last = max(frames, default=0)
        process = Path(f'/proc/{args.pid}/cmdline')
        try:
            command = process.read_bytes().replace(b'\0', b' ').decode()
        except FileNotFoundError:
            command = ''
        if command and expected_config not in command:
            raise RuntimeError('PID identity changed; refusing to signal another process')
        for endpoint in range(cfg['first_check_frame'], min(last, cfg['last_check_frame']) + 1,
                              cfg['check_interval_frames']):
            if endpoint in seen:
                continue
            indices = list(range(endpoint - cfg['window_frames'] + 1, endpoint + 1))
            if not all(frame in frames for frame in indices):
                continue
            means = {name: statistics.fmean(
                source[frame]['quantized_evaluation']['validation']['psnr'] for frame in indices)
                for name, source in dict(candidate=frames, **anchors).items()}
            differences = {name: means['candidate'] - means[name] for name in anchors}
            failed = (differences['keep_probe'] < -cfg['worse_than_keep_probe_db'] and
                      differences['original_core'] < -cfg['worse_than_original_core_db'])
            nonfinite = not all(math.isfinite(value) for value in means.values())
            consecutive = consecutive + 1 if failed else 0
            check = dict(endpoint=endpoint, window=indices, validation_means=means,
                differences=differences, failed=failed, consecutive_failed_checks=consecutive,
                passive_cam00_used=False)
            checks.append(check)
            seen.add(endpoint)
            print('Validation check ' + json.dumps(check), flush=True)
            if nonfinite or consecutive >= cfg['consecutive_failed_checks']:
                stop = dict(reason='nonfinite_validation' if nonfinite else
                            cfg.get('failure_reason', 'predeclared_early_reversal'),
                            at_completed_frame=last, triggering_check=check, pid=args.pid)
                break
        state = dict(run=str(args.run), pid=args.pid, completed_frames=last, checks=checks,
                     stop_decision=stop, process_running=bool(command), complete299=last == 299)
        (output / 'status.json').write_text(json.dumps(state, indent=2))
        if stop:
            (output / 'stop_decision.json').write_text(json.dumps(stop, indent=2))
            if command:
                os.kill(args.pid, signal.SIGTERM)
            print('Stopped authorized run: ' + json.dumps(stop), flush=True)
            break
        if last != announced and last:
            packet = frames[last]
            stream = packet['stream']
            print(json.dumps(dict(frame=last,
                validation=packet['quantized_evaluation']['validation']['psnr'],
                passive_cam00=packet['quantized_evaluation']['test']['psnr'],
                training_s=packet['timing']['training'],
                extra_renders=stream['candidate_fit']['extra_renders'],
                hard_bytes=stream['hard_storage_bytes'])), flush=True)
            announced = last
        if not command or last == 299:
            print('Watch finished ' + json.dumps(state), flush=True)
            break
        time.sleep(cfg['poll_seconds'])


if __name__ == '__main__':
    main()
