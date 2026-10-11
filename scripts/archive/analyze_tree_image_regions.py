"""Read saved cam00 PNGs and measure errors in explicitly supplied rectangles."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
import yaml


def parse_region(value):
    name, coordinates = value.split(':', 1)
    bounds = tuple(float(item) for item in coordinates.split(','))
    if (len(bounds) != 4 or not 0 <= bounds[0] < bounds[2] <= 1 or
            not 0 <= bounds[1] < bounds[3] <= 1):
        raise ValueError('Use name:x0,y0,x1,y1 with normalized bounds in [0, 1]')
    return name, bounds


def inspect_images(run, regions, frames):
    config = yaml.safe_load((run / 'config.yaml').read_text())
    steps = config['trainer']['num_incr_steps']
    rows = []
    for frame in frames:
        directory = run / f'image/frames/{frame:04d}'
        with Image.open(directory / 'cam00_gt.png') as image:
            truth = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        with Image.open(directory / f'cam00_step{steps}.png') as image:
            rendered = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        if truth.shape != rendered.shape:
            raise ValueError(f'Frame {frame} image sizes differ')
        error = np.square(truth - rendered).mean(axis=-1, dtype=np.float64)
        height, width = error.shape
        row = dict(frame=frame)
        for name, (x0, y0, x1, y1) in regions.items():
            crop = error[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)]
            if not crop.size:
                raise ValueError(f'Empty pixel rectangle: {name}')
            row[f'{name}_mse'] = float(crop.mean())
            row[f'{name}_psnr'] = float(-10 * np.log10(max(crop.mean(), 1e-20)))
        rows.append(row)
    return dict(run=str(run), rows=rows, means={key: statistics.fmean(row[key] for row in rows)
                for key in rows[0] if key != 'frame'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--region', action='append', default=[], help='name:x0,y0,x1,y1')
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.first < 1 or args.last < args.first:
        raise ValueError('Expected a positive contiguous dynamic frame range')
    regions = {'whole': (0, 0, 1, 1)}
    for value in args.region:
        name, bounds = parse_region(value)
        if not name or name in regions:
            raise ValueError(f'Duplicate or empty region name: {name}')
        regions[name] = bounds
    runs = [inspect_images(run.resolve(), regions, range(args.first, args.last + 1)) for run in args.run]
    result = dict(regions=regions, runs=runs,
        semantics='PSNR from 8-bit saved PNGs. Rectangles are explicit diagnostic crops, '
                  'may contain multiple objects and overlap. They are not segmentation masks '
                  'or the floating-point renderer evaluation used in the main result.')
    if len(runs) == 2:
        result['candidate_minus_control'] = {key: runs[1]['means'][key] - value
                                            for key, value in runs[0]['means'].items()}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for index, run in enumerate(runs):
        with (args.output / f'regions_{index}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(run['rows'][0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(run['rows'])
    print(json.dumps(dict(means=[run['means'] for run in runs],
                         comparison=result.get('candidate_minus_control'))))


if __name__ == '__main__':
    main()
