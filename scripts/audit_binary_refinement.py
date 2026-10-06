"""Audit sequential integer-motion playback against complete training states.

Usage: python scripts/audit_binary_refinement.py RUN OUTPUT_JSON
CUDA matches the training arithmetic for exact SHA256 verification.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from plyfile import PlyData
import torch
import yaml

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
package = ModuleType('pipeline')
package.__path__ = [str(repo / 'pipeline')]
sys.modules['pipeline'] = package
package = ModuleType('pipeline.ReCon-GS')
package.__path__ = [str(repo / 'pipeline/ReCon-GS')]
sys.modules['pipeline.ReCon-GS'] = package
BR = importlib.import_module('pipeline.ReCon-GS.model.binary_refinement')
Q = importlib.import_module('pipeline.ReCon-GS.model.residual_motion')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
run, output = args.run, args.output
if output.exists():
    raise FileExistsError(f'Refusing to overwrite audit output: {output}')
device = 'cuda'
config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))


def capture_raw(packet):
    state = packet['gaussians']
    slots = dict(xyz=1, sh_dc=2, sh_rest=3, logscale=4, rotation=5, opacity=6)
    return {name: state[index].detach().to(device) for name, index in slots.items()}


def birth_raw(path):
    vertex = PlyData.read(path)['vertex']
    names = vertex.data.dtype.names
    def fields(prefix):
        ordered = sorted((n for n in names if n.startswith(prefix)),
                         key=lambda n: int(n.rsplit('_', 1)[1]))
        return np.stack([vertex[name] for name in ordered], axis=1).astype(np.float32)
    arrays = dict(
        xyz=np.stack([vertex[name] for name in ('x', 'y', 'z')], axis=1),
        rotation=fields('rot_'), logscale=fields('scale_'),
        opacity=np.asarray(vertex['opacity'])[:, None],
        sh_dc=fields('f_dc_').reshape(-1, 3, 1).transpose(0, 2, 1),
        sh_rest=fields('f_rest_').reshape(len(vertex), 3, -1).transpose(0, 2, 1))
    return {name: torch.from_numpy(value.copy()).to(device) for name, value in arrays.items()}


raw = capture_raw(torch.load(config['ckpt_path'], map_location='cpu', weights_only=False))
checks = []
for frame in range(1, config['data']['extra_dataset_kwargs']['num_frames']):
    report = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text(encoding='utf-8'))['stream']
    manifest = json.loads(Path(report['frame_manifest']['path']).read_text(encoding='utf-8'))
    assert Q.reference_digest(raw) == manifest['previous_reference_sha256'], ('previous digest', frame)
    rows = Q.survivor_rows(len(raw['xyz']), manifest['deleted_indices']).to(device)
    raw = {name: value[rows].clone() for name, value in raw.items()}
    motion_path = Path(report['motion_export']['path'])
    xyz, rotation = BR.BinaryRefinementMotion.decode_export_motion(motion_path, raw['xyz'], raw['rotation'])
    raw['xyz'] = torch.from_numpy(xyz).to(device)
    raw['rotation'] = torch.from_numpy(rotation).to(device)
    with np.load(motion_path, allow_pickle=False) as packet:
        parameter_bytes = BR.BinaryRefinementMotion.raw_parameter_bytes({key: packet[key] for key in packet.files})
    assert parameter_bytes == report['motion_export']['parameter_bytes'], ('payload bytes', frame)
    birth = report.get('incremental_export')
    if birth:
        newborn = birth_raw(birth['path'])
        raw = {name: torch.cat((raw[name], newborn[name].to(raw[name])), dim=0) for name in raw}
    digest = Q.reference_digest(raw)
    assert digest == report['reference_sha256'], ('materialized digest', frame, digest, report['reference_sha256'])
    checkpoints = list(run.glob(f'ckpt-{frame}-*.pth'))
    for path in checkpoints:
        saved = capture_raw(torch.load(path, map_location='cpu', weights_only=False))
        for name in raw:
            torch.testing.assert_close(raw[name], saved[name], rtol=0, atol=0)
    checks.append(dict(frame=frame, exact_materialized_sha256=digest,
                       motion_parameter_bytes=parameter_bytes, checkpoint_matches=len(checkpoints)))
output.write_text(json.dumps(dict(frames=len(checks), device=device,
    semantics='Sequentially decode integer motion, apply explicit deletion IDs and append birth PLY. '
              'Every complete raw state matches the training SHA256; saved checkpoints match every '
              'attribute bit-for-bit on the same CUDA device.', checks=checks), indent=2), encoding='utf-8')
print('Sequential raw-state audit passed for', len(checks), 'frames')
