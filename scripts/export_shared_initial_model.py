"""Export a shared first-frame checkpoint as a lossless standalone model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = yaml.safe_load((run / 'config.yaml').read_text())
    source = Path(config['ckpt_path'])
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    values = checkpoint['gaussians']
    raw = dict(xyz=values[1], sh_dc=values[2], sh_rest=values[3], logscale=values[4],
               rotation=values[5], opacity=values[6])
    arrays = {name: value.detach().cpu().numpy() for name, value in raw.items()}
    destination = run / 'first_reference.npz'
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(destination, **arrays)
    with np.load(destination, allow_pickle=False) as restored:
        if any(not np.array_equal(restored[name], value) for name, value in arrays.items()):
            raise AssertionError('Lossless initial-model roundtrip differs')
    report = dict(source=str(source.resolve()), path=str(destination),
        archive_bytes=destination.stat().st_size, raw_attribute_bytes=sum(value.nbytes for value in arrays.values()),
        gaussian_count=len(arrays['xyz']), active_sh_degree=int(values[0]),
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), exact_attributes=True,
        semantics='Standalone raw GS attributes only; no optimizer, motion parameters or dynamic-frame checkpoint is required.')
    (run / 'initial_model_export.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
