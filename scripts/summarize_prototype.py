"""Collect one-row-per-frame test metrics and measured export storage."""
import csv
import json
import re
from pathlib import Path

rows = []
for config in sorted(Path('output').glob('prototype_*/**/config.yaml')):
    root = config.parent
    logs = list(root.glob('logging.INFO')) or list(root.glob('logging.*.log.INFO.*'))
    if not logs:
        continue
    text = logs[0].read_text()
    frame = None
    metrics = {}
    for line in text.splitlines():
        match = re.search(r'Start learning frame (\d+)', line)
        if match:
            frame = int(match[1])
        match = re.search(r'Evaluate test dataset at step (\d+):', line)
        if match and frame is not None:
            step = int(match[1])
        match = re.search(r'\bPSNR: ([\d.]+)', line)
        if match and 'step' in locals() and frame is not None:
            metrics.setdefault(frame, {})[step] = float(match[1])
    for frame, values in metrics.items():
        if frame == 0:
            continue
        storage = root / 'frame_storage' / f'frame_{frame:03d}.json'
        row = dict(method=config.parts[1].removeprefix('prototype_'), frame=frame,
                   psnr50=values.get(50), psnr99=values.get(99), psnr100=values.get(100),
                   psnr200=values.get(200), run_dir=str(root))
        if storage.exists():
            row.update(json.loads(storage.read_text()))
        rows.append(row)
fields = ['method','frame','psnr50','psnr99','psnr100','psnr200','active_fine','candidate_fine',
          'incremental_gs','motion_bytes','incremental_bytes','total_bytes','run_dir']
path = Path('output/prototype_summary.csv')
with path.open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
print(f'Saved {path}')
