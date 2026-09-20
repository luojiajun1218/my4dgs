"""Read-only saved-gradient benchmark; compare exact output and CPU runtime."""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'config/idea_rd_tree_projection_benchmark.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    name = config['experiment_name']
    if Path(name).name != name or config['repeats'] < 1:
        raise ValueError('Invalid benchmark configuration')
    output = ROOT / 'output' / (name + '_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    files = [Path(__file__).resolve(), args.config.resolve(),
             ROOT / 'pipeline/ReCon-GS/model/rd_tree_projection.py']
    for path in files:
        shutil.copy2(path, snapshot / path.name)
    inputs = ROOT / config['input_directory']
    prior = json.loads((inputs / 'inputs.json').read_text())
    checkpoint_path = Path(prior['checkpoint'])
    assert digest(checkpoint_path) == prior['checkpoint_sha256']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    xyz = checkpoint['gaussians'][1].detach().double().numpy()
    gradient = np.load(inputs / 'gradient.npz')['gradient'].astype(np.float64)
    rho = np.asarray(config['curvature'])
    target = -gradient / rho
    rate = config['normalized_rate'] / len(xyz)
    rows = []
    reference_tree_digest = None
    for partition in config['partitions']:
        started = time.perf_counter()
        tree = P.build_candidate_tree(xyz, config['max_depth'], config['min_leaf_size'],
                                      partition, config['clustering_iterations'])
        tree_seconds = time.perf_counter() - started
        workspace = P.ProjectionWorkspace(tree)
        if partition == 'geometry_2means':
            reference_tree_digest = tree.fingerprint()
        if partition == 'geometry_2means_batched' and reference_tree_digest is not None:
            assert tree.fingerprint() == reference_tree_digest, 'Batched candidate tree changed on benchmark'
        reference_packet = None
        # Alternate order to reduce systematic warm-cache ordering bias.
        timings = {False: [], True: []}
        for repeat in range(config['repeats']):
            for fast in ((False, True) if repeat % 2 == 0 else (True, False)):
                started = time.perf_counter()
                result = P.project_rd_tree(target, tree, config['quant_step'], rate, rho,
                                           vectorized_symbols=fast, vectorized_statistics=fast,
                                           workspace=workspace if fast else None)
                timings[fast].append(time.perf_counter() - started)
                packet = P.encode_projection(result, tree)
                if reference_packet is None:
                    reference_packet = packet
                assert packet == reference_packet, 'Reference/vectorized or repeated packet mismatch'
                assert np.array_equal(P.decode_projection(packet, tree), result.field)
        (output / f'{partition}.rdp').write_bytes(reference_packet)
        row = {'partition': partition, 'nodes': len(tree.nodes), 'tree_seconds': tree_seconds,
               'reference_seconds': timings[False], 'vectorized_seconds': timings[True],
               'median_speedup': float(np.median(timings[False]) / np.median(timings[True])),
               'identical_packets': True, 'packet_bytes': len(reference_packet)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    record = {'config': config, 'results': rows,
              'input_gradient_sha256': digest(inputs / 'gradient.npz'),
              'checkpoint_sha256': prior['checkpoint_sha256'],
              'source_hashes': {str(path.relative_to(ROOT)): digest(path) for path in files},
              'scope': 'CPU projection only; excludes tree construction, encoding and rendering'}
    (output / 'results.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Completed:', output, flush=True)


if __name__ == '__main__':
    main()
