"""Audit sequential storage-aware binary decoding against saved frame states.

Usage: python scripts/audit_storage_aware_binary.py RUN OUTPUT_JSON [--device cuda]
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import torch
import yaml

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
package = ModuleType('pipeline')
package.__path__ = [str(repo / 'pipeline')]
sys.modules['pipeline'] = package
package = ModuleType('pipeline.ReCon-GS')
package.__path__ = [str(repo / 'pipeline' / 'ReCon-GS')]
sys.modules['pipeline.ReCon-GS'] = package
SAB = importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary')
Q = importlib.import_module('pipeline.ReCon-GS.model.residual_motion')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--device', default='cuda')
parser.add_argument('--frames', type=int,
                    help='Audit this completed prefix; defaults to the full configured run')
args = parser.parse_args()
run, output = args.run.resolve(), args.output.resolve()
if output.exists():
    raise FileExistsError(f'Refusing to overwrite audit output: {output}')
if args.device.startswith('cuda') and not torch.cuda.is_available():
    raise RuntimeError('Exact training-arithmetic audit requires a CUDA device')

config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
if config['module']['motion_mode'] != 'storage_aware_binary':
    raise ValueError('Expected a storage_aware_binary run')
motion_config = SAB.StorageAwareBinaryConfig(**config['module']['storage_aware_binary'])
motion_config.validate()
device = torch.device(args.device)


def capture_raw(packet):
    state = packet['gaussians']
    slots = dict(xyz=1, sh_dc=2, sh_rest=3, logscale=4, rotation=5, opacity=6)
    return {name: state[index].detach().to(device) for name, index in slots.items()}


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, run / path, repo / path)
    return next((candidate for candidate in candidates if candidate.exists()), run / path)


first_checkpoint = Path(config['ckpt_path'])
if not first_checkpoint.is_absolute():
    first_checkpoint = (repo / first_checkpoint).resolve()
raw = capture_raw(torch.load(first_checkpoint, map_location='cpu', weights_only=False))
decoder = SAB.StorageAwareBinaryDecoder(motion_config)
dynamic_frames = config['data']['extra_dataset_kwargs']['num_frames'] - 1
if args.frames is not None:
    if args.frames < 1 or args.frames > dynamic_frames:
        raise ValueError(f'--frames must be in [1, {dynamic_frames}]')
    dynamic_frames = args.frames
checks = []

for frame in range(1, dynamic_frames + 1):
    report_path = run / 'frame_summary' / f'frame_{frame:03d}.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))['stream']
    motion_export = report['motion_export']
    motion_path = resolve_path(motion_export['path'])
    birth_export = report.get('incremental_export')
    birth_path = resolve_path(birth_export['path']) if birth_export else None
    manifest_path = resolve_path(report['frame_manifest']['path'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if 'deleted_indices' in manifest or 'owner_ids' in manifest or 'terminal_ids' in manifest:
        raise ValueError(f'Per-row or per-node IDs leaked into frame {frame} manifest')
    if motion_path.stat().st_size != motion_export['archive_bytes']:
        raise ValueError(f'Motion archive byte count mismatch at frame {frame}')
    if birth_export and birth_path.stat().st_size != birth_export['archive_bytes']:
        raise ValueError(f'Birth PLY byte count mismatch at frame {frame}')

    raw, detail = decoder.advance(motion_path, birth_path, raw)
    digest = Q.reference_digest(raw)
    if digest != report['reference_sha256']:
        raise ValueError(f'Materialized state digest mismatch at frame {frame}: {digest}')
    if detail['tree_rebuilt'] != ((frame - 1) % motion_config.tree_rebuild_interval == 0):
        raise ValueError(f'Unexpected tree rebuild cadence at frame {frame}')
    if detail['birth_count'] != report['new_count']:
        raise ValueError(f'Birth count mismatch at frame {frame}')
    if len(detail['deleted_rows']) != report['deleted_count']:
        raise ValueError(f'Deletion count mismatch at frame {frame}')
    file_bytes = motion_path.stat().st_size + (birth_path.stat().st_size if birth_path else 0)
    if file_bytes != report['inference_file_bytes']:
        raise ValueError(f'Necessary inference file total mismatch at frame {frame}')

    checkpoint_matches = 0
    for checkpoint in sorted(run.glob(f'ckpt-{frame}-*.pth')):
        saved = capture_raw(torch.load(checkpoint, map_location='cpu', weights_only=False))
        for name in raw:
            torch.testing.assert_close(raw[name], saved[name], rtol=0, atol=0)
        checkpoint_matches += 1
    checks.append(dict(frame=frame, reference_sha256=digest,
        motion_bytes=motion_path.stat().st_size,
        birth_bytes=birth_path.stat().st_size if birth_path else 0,
        inference_bytes=file_bytes, coded_payload_bits=detail['coded_payload_bits'],
        tree_rebuilt=detail['tree_rebuilt'], deleted_count=len(detail['deleted_rows']),
        birth_count=detail['birth_count'], checkpoint_matches=checkpoint_matches))

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(frames=len(checks), device=str(device), checks=checks,
    semantics='Each frame verifies the prior full-state hash, reproduces deterministic deletions, '
    'rebuilds or remaps the cached binary tree, decodes the preorder stream, appends the birth PLY, '
    'and matches the materialized training-state digest. Saved frame checkpoints are compared '
    'attribute-by-attribute on the same device.'), indent=2), encoding='utf-8')
print(f'Sequential storage-aware binary audit passed for {len(checks)} frames')
