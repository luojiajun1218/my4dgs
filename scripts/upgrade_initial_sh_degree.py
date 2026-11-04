"""Expand a static first-frame checkpoint to a higher SH degree losslessly.

Existing coefficients and optimizer moments are copied exactly.  Newly added
coefficients and their moments start at zero, so the upgraded checkpoint must
render identically before any colour-only refinement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch


ACTIVE_SH_INDEX = 0
FEATURE_REST_INDEX = 3
OPTIMIZER_INDEX = 10


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _degree_from_rest(value: torch.Tensor) -> int:
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f'Expected SH-rest shape [N, C, 3], got {tuple(value.shape)}')
    side = math.isqrt(int(value.shape[1]) + 1)
    if side * side != int(value.shape[1]) + 1:
        raise ValueError(f'Invalid SH-rest coefficient count: {value.shape[1]}')
    return side - 1


def _expanded_zeros(value: torch.Tensor, target_coefficients: int) -> torch.Tensor:
    result = value.new_zeros((value.shape[0], target_coefficients, value.shape[2]))
    result[:, : value.shape[1]].copy_(value)
    return result


def upgrade_checkpoint(checkpoint: dict, target_degree: int) -> tuple[dict, dict]:
    """Return an upgraded checkpoint and a compact verification report."""
    if int(checkpoint.get('current_frame', -1)) != 0:
        raise ValueError('Only a frame-0 static checkpoint can be upgraded')
    if target_degree not in range(4):
        raise ValueError('target_degree must be between 0 and 3')
    if 'gaussians' not in checkpoint or not isinstance(checkpoint['gaussians'], tuple):
        raise ValueError('Expected a native Gaussian checkpoint tuple')
    if len(checkpoint['gaussians']) <= OPTIMIZER_INDEX:
        raise ValueError('Gaussian checkpoint tuple is incomplete')

    upgraded = copy.deepcopy(checkpoint)
    gaussian = list(upgraded['gaussians'])
    source_rest = checkpoint['gaussians'][FEATURE_REST_INDEX]
    source_degree = _degree_from_rest(source_rest)
    active_degree = int(checkpoint['gaussians'][ACTIVE_SH_INDEX])
    if not 0 <= active_degree <= source_degree:
        raise ValueError('Active SH degree exceeds checkpoint capacity')
    if target_degree <= source_degree:
        raise ValueError(
            f'Target degree {target_degree} must exceed source degree {source_degree}')

    target_coefficients = (target_degree + 1) ** 2 - 1
    target_rest = _expanded_zeros(source_rest, target_coefficients)
    gaussian[ACTIVE_SH_INDEX] = target_degree
    gaussian[FEATURE_REST_INDEX] = target_rest

    optimizer = gaussian[OPTIMIZER_INDEX]
    if not isinstance(optimizer, dict):
        raise ValueError('Expected a Gaussian optimizer state dictionary')
    groups = [group for group in optimizer.get('param_groups', [])
              if group.get('name') == 'f_rest']
    if len(groups) != 1 or len(groups[0].get('params', [])) != 1:
        raise ValueError('Could not uniquely identify the f_rest optimizer parameter')
    parameter_id = groups[0]['params'][0]
    state = optimizer.get('state', {}).get(parameter_id)
    if state is not None:
        for name, value in tuple(state.items()):
            if not torch.is_tensor(value) or value.ndim == 0:
                continue
            if tuple(value.shape) != tuple(source_rest.shape):
                raise ValueError(
                    f'Unexpected f_rest optimizer tensor {name}: {tuple(value.shape)}')
            state[name] = _expanded_zeros(value, target_coefficients)

    gaussian[OPTIMIZER_INDEX] = optimizer
    upgraded['gaussians'] = tuple(gaussian)

    if not torch.equal(target_rest[:, : source_rest.shape[1]], source_rest):
        raise AssertionError('Existing SH coefficients changed during upgrade')
    if torch.count_nonzero(target_rest[:, source_rest.shape[1]:]):
        raise AssertionError('New SH coefficients are not zero')
    for index, (before, after) in enumerate(zip(checkpoint['gaussians'], upgraded['gaussians'])):
        if index in (ACTIVE_SH_INDEX, FEATURE_REST_INDEX, OPTIMIZER_INDEX):
            continue
        if torch.is_tensor(before):
            if not torch.equal(before, after):
                raise AssertionError(f'Gaussian field {index} changed during upgrade')
        elif before != after:
            raise AssertionError(f'Gaussian field {index} changed during upgrade')

    report = {
        'frame': int(checkpoint['current_frame']),
        'global_step': int(checkpoint['global_step']),
        'points': int(source_rest.shape[0]),
        'source_degree': source_degree,
        'source_active_degree': active_degree,
        'target_degree': target_degree,
        'source_rest_shape': list(source_rest.shape),
        'target_rest_shape': list(target_rest.shape),
        'source_rest_sha256': _tensor_sha256(source_rest),
        'retained_rest_sha256': _tensor_sha256(
            target_rest[:, : source_rest.shape[1]]),
        'new_coefficients_all_zero': not bool(torch.count_nonzero(
            target_rest[:, source_rest.shape[1]:])),
        'non_sh_fields_bit_exact': True,
        'optimizer_moments_expanded': state is not None,
    }
    return upgraded, report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target-degree', type=int, required=True, choices=range(4))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    upgraded, report = upgrade_checkpoint(checkpoint, args.target_degree)
    temporary = output.with_name(output.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        torch.save(upgraded, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    report.update({
        'source_checkpoint': str(source),
        'source_checkpoint_sha256': _file_sha256(source),
        'output_checkpoint': str(output),
        'output_checkpoint_sha256': _file_sha256(output),
        'scope': ('SH capacity expansion only. Verify identical rendering before '
                  'using this checkpoint for colour-only refinement.'),
    })
    provenance = output.with_suffix('.json')
    if provenance.exists():
        raise FileExistsError(provenance)
    provenance.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
