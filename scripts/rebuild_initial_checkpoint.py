"""Rebuild a trainable static checkpoint from a lossless frame-0 reference.

The rigid-tree reference stores every rendered Gaussian attribute but not the
Adam state used by the original static fit.  This utility copies only the
checkpoint envelope and optimizer-group metadata from a compatible template,
installs the reference attributes bit-exactly, and intentionally starts with
empty Adam moments.  Historical files are never modified.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


RAW_TO_CAPTURE = {
    'xyz': 1,
    'sh_dc': 2,
    'sh_rest': 3,
    'logscale': 4,
    'rotation': 5,
    'opacity': 6,
}
EXPECTED_SHAPES = {
    'xyz': (3,),
    'sh_dc': (1, 3),
    'logscale': (3,),
    'rotation': (4,),
    'opacity': (1,),
}
EXPECTED_OPTIMIZER_GROUPS = ('xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(path):
    """Return validated float32 raw tensors, extent and complete SH degree."""
    with np.load(path, allow_pickle=False) as source:
        missing = (set(RAW_TO_CAPTURE) | {'extent'}) - set(source.files)
        if missing:
            raise ValueError(f'Reference is missing fields: {sorted(missing)}')
        arrays = {name: np.array(source[name], copy=True) for name in RAW_TO_CAPTURE}
        extent = float(source['extent'])
        frame_index = int(source['frame_index']) if 'frame_index' in source else 0
    if frame_index != 0:
        raise ValueError('Expected a frame-0 reference')
    count = len(arrays['xyz'])
    if count < 1 or not math.isfinite(extent) or extent <= 0:
        raise ValueError('Reference needs at least one Gaussian and a positive finite extent')
    for name, array in arrays.items():
        if array.dtype != np.float32 or len(array) != count or not np.isfinite(array).all():
            raise ValueError(f'Invalid {name}: expected {count} finite float32 rows')
        if name in EXPECTED_SHAPES and array.shape[1:] != EXPECTED_SHAPES[name]:
            raise ValueError(f'Invalid {name} shape: {array.shape}')
    rest = arrays['sh_rest']
    if rest.ndim != 3 or rest.shape[2] != 3:
        raise ValueError(f'Invalid sh_rest shape: {rest.shape}')
    degree = math.isqrt(rest.shape[1] + 1) - 1
    if degree < 0 or (degree + 1) ** 2 - 1 != rest.shape[1]:
        raise ValueError(f'sh_rest does not contain a complete SH basis: {rest.shape}')
    return {name: torch.from_numpy(value) for name, value in arrays.items()}, extent, degree


def rebuild_packet(template, raw, extent, degree, global_step):
    """Install raw attributes and reset row-dependent training state."""
    if template.get('current_frame') != 0:
        raise ValueError('Template must be a completed static frame-0 checkpoint')
    capture = template.get('gaussians')
    if not isinstance(capture, (tuple, list)) or len(capture) != 12:
        raise ValueError('Unsupported Gaussian checkpoint capture')
    optimizer = copy.deepcopy(capture[10])
    groups = tuple(group.get('name') for group in optimizer.get('param_groups', []))
    if groups != EXPECTED_OPTIMIZER_GROUPS:
        raise ValueError(f'Unexpected optimizer groups: {groups}')
    if [group.get('params') for group in optimizer['param_groups']] != [[i] for i in range(6)]:
        raise ValueError('Template optimizer parameter layout is incompatible')
    optimizer['state'] = {}
    count = len(raw['xyz'])
    values = list(copy.deepcopy(capture))
    values[0] = degree
    for name, index in RAW_TO_CAPTURE.items():
        values[index] = torch.nn.Parameter(raw[name].clone(), requires_grad=True)
    values[7] = torch.zeros(count, dtype=torch.float32)
    values[8] = torch.zeros(count, 1, dtype=torch.float32)
    values[9] = torch.zeros(count, 1, dtype=torch.float32)
    values[10] = optimizer
    values[11] = extent
    packet = copy.deepcopy(template)
    packet['global_step'] = int(global_step)
    packet['current_frame'] = 0
    packet['gaussians'] = tuple(values)
    return packet


def verify_packet(packet, raw, extent, degree, global_step):
    capture = packet['gaussians']
    if (packet['current_frame'] != 0 or packet['global_step'] != global_step
            or capture[0] != degree or float(capture[11]) != extent):
        raise AssertionError('Rebuilt checkpoint metadata differs')
    for name, index in RAW_TO_CAPTURE.items():
        value = capture[index]
        if not isinstance(value, torch.nn.Parameter) or not value.requires_grad:
            raise AssertionError(f'{name} is not a trainable leaf Parameter')
        if not torch.equal(value.detach(), raw[name]):
            raise AssertionError(f'{name} differs from the reference')
    count = len(raw['xyz'])
    if tuple(capture[7].shape) != (count,) or any(tuple(capture[i].shape) != (count, 1)
                                                   for i in (8, 9)):
        raise AssertionError('Row-dependent buffers have incorrect shapes')
    if capture[10]['state']:
        raise AssertionError('Rebuilt optimizer must start without Adam moments')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True,
                        help='Compatible static checkpoint used only for its packet envelope')
    parser.add_argument('--global-step', type=int, required=True,
                        help='Training step represented by the reference attributes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix('.json')
    if args.global_step < 0:
        raise ValueError('global-step must be nonnegative')
    if args.output.exists() or report_path.exists():
        raise FileExistsError(args.output if args.output.exists() else report_path)
    if any((parent / 'config.yaml').is_file() for parent in args.output.resolve().parents):
        raise ValueError('Do not write reconstructed assets into a historical training run')

    raw, extent, degree = load_reference(args.reference)
    template = torch.load(args.template, map_location='cpu', weights_only=False)
    packet = rebuild_packet(template, raw, extent, degree, args.global_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packet, args.output)
    restored = torch.load(args.output, map_location='cpu', weights_only=False)
    verify_packet(restored, raw, extent, degree, args.global_step)
    report = {
        'reference': str(args.reference.resolve()),
        'reference_sha256': sha256(args.reference),
        'template': str(args.template.resolve()),
        'template_sha256': sha256(args.template),
        'output': str(args.output.resolve()),
        'output_sha256': sha256(args.output),
        'global_step': args.global_step,
        'gaussian_count': len(raw['xyz']),
        'SH_degree': degree,
        'spatial_lr_scale': extent,
        'attributes_bit_exact': True,
        'trainable_leaf_parameters': True,
        'optimizer_state_reset': True,
        'semantics': ('Reference attributes are lossless; Adam moments are unavailable and '
                      'intentionally reset before any resumed fitting.'),
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
