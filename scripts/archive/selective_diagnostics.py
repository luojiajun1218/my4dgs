#!/usr/bin/env python3
"""Offline wrapper for the same region diagnostics used when training saves images."""
import argparse
import json
import sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'pipeline/ReCon-GS/model'))
from selective_plot import draw_frame

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New diagnostic directory; must not already exist')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/diagnostics/selective_motion.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not config['enabled']:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cameras = json.loads((args.experiment / 'cameras.json').read_text())
    config['camera'] = config['offline_camera']
    summaries = []
    for frame in config['offline_frames']:
        candidates = [c for c in cameras if c['img_name'] == config['offline_camera'] and c.get('frame') == frame]
        if len(candidates) != 1:
            raise ValueError(f'Expected one camera {config["offline_camera"]} at frame {frame}, got {len(candidates)}')
        packet = torch.load(args.experiment / 'selective_motion' / f'frame_{frame:03d}.pt', map_location='cpu', weights_only=False)
        image_path = args.experiment / 'image/frames' / f'{frame:04d}' / f'{config["offline_camera"]}_step{config["image_step"]}.png'
        summaries.append(draw_frame(packet, candidates[0], image_path, args.output, frame, config))
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    (args.output / 'display_config.yaml').write_text(args.config.read_text())


if __name__ == '__main__':
    main()
