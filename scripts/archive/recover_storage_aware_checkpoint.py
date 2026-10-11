"""Recover a completed frame from its verified native stream, without training.

Requires an earlier finalized checkpoint and the completed frame reports/bitstreams.
Creates one new checkpoint. Hidden candidates, old Adam moments and RNG states are
not recoverable; the reconstructed committed motion is verified for next-frame
deletions and cached-tree reuse. The next frame initializes its own motion afresh.
"""

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import torch
import yaml

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))


def numerical_modules():
    for name, path in (('pipeline', repo / 'pipeline'),
                       ('pipeline.ReCon-GS', repo / 'pipeline/ReCon-GS')):
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    return (importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary'),
            importlib.import_module('pipeline.ReCon-GS.archive.model.residual_motion'))


@torch.no_grad()
def motion_from_decoded(config, snapshot, materialized_layout):
    """Recreate the selected motion and verify its exact original encoded payload."""
    sab, _ = numerical_modules()
    motion = sab.StorageAwareBinaryMotion(copy.deepcopy(config))
    header = snapshot['header']
    motion.setup(snapshot['base_xyz'], snapshot['base_rotation'], header['extent'],
                 frame_index=header['frame'], layout_override=snapshot['layout'])
    precision = snapshot['coding']['precision']
    symbols = snapshot['coding']['symbols']
    for level, target in enumerate(symbols):
        steps = motion._candidate_steps()[precision[level]]
        raw = target.to(motion.base_xyz) * steps
        # Floating-point multiply/divide need not be reciprocal. Choose an adjacent
        # float when necessary so that re-quantization recovers the exact symbols.
        for _ in range(8):
            rounded = torch.round(raw / steps).to(torch.long)
            exact = rounded == target
            if bool(exact.all()):
                break
            direction = torch.where(rounded < target,
                torch.full_like(raw, float('inf')), torch.full_like(raw, -float('inf')))
            raw = torch.where(exact, raw, torch.nextafter(raw, direction))
        else:
            raise ValueError('Could not reconstruct original quantization symbols')
        motion.region_translation[level].copy_(raw[:, :3])
        motion.region_rotation[level].copy_(raw[:, 3:])
        motion.precision_logits[level].fill_(-100.)
        motion.precision_logits[level].scatter_(1, precision[level][:, None], 0.)
    for logits, gates in zip(motion.refine_logits, snapshot['gates']):
        logits.copy_(torch.where(gates, 100., -100.))
    motion.hard_masks = dict(refine=[value.clone() for value in snapshot['gates']],
                             precision=[value.clone() for value in precision])
    motion.prepare_step(config.motion_steps)
    motion.committed = True
    for parameter in motion.parameters():
        parameter.requires_grad_(False)
    motion.materialized_layout = sab._copy_layout(materialized_layout)
    motion.previous_layout = sab._copy_layout(snapshot['previous_layout'])
    payload, bits, _, _ = motion._encode_payload(config.motion_steps)
    if payload != header['payload'] or bits != header['payload_bits']:
        raise ValueError('Recovered motion does not reproduce the original bitstream')
    return motion


def cpu_packet(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_packet(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_packet(item) for item in value)
    if isinstance(value, list):
        return [cpu_packet(item) for item in value]
    return value


def recover(run, checkpoint, target_frame, output, device):
    sab, residual = numerical_modules()
    run, checkpoint, output = Path(run).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError('Recovery output must be new and outside the historical run')
    config = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    if config['module']['motion_mode'] != 'storage_aware_binary':
        raise ValueError('Expected a storage-aware binary source run')
    motion_config = sab.StorageAwareBinaryConfig(**config['module']['storage_aware_binary'])
    motion_config.validate()
    packet = torch.load(checkpoint, map_location=device, weights_only=False)
    first = int(packet['current_frame'])
    if (not packet['quantized_frame_state']['finalized'] or
            packet['global_step'] != config['trainer']['num_incr_steps'] or target_frame <= first):
        raise ValueError('Recovery requires an earlier completed checkpoint')
    slots = dict(xyz=1, sh_dc=2, sh_rest=3, logscale=4, rotation=5, opacity=6)
    raw = {name: packet['gaussians'][slot].detach() for name, slot in slots.items()}
    report = json.loads((run / 'frame_summary' / f'frame_{first:03d}.json').read_text())['stream']
    if residual.reference_digest(raw) != report['reference_sha256']:
        raise ValueError('Starting checkpoint disagrees with the source frame report')
    motion = sab.StorageAwareBinaryMotion(copy.deepcopy(motion_config))
    motion.restore(packet['quantized_residual_motion'])
    xyz, rotation = motion(motion_config.motion_steps)
    decoder = sab.StorageAwareBinaryDecoder(motion_config, retain_frame_state=True)
    decoder.materialized_layout = motion.materialized_layout
    decoder.previous_changed = ((xyz != motion.base_xyz).any(-1) |
                                (rotation != motion.base_rotation).any(-1)).detach()
    decoder.previous_birth_count = packet['quantized_frame_state']['previous_new_count']
    decoder.last_frame = first
    origins = packet['quantized_frame_state']['point_birth_frame'].detach().cpu()
    verified = []
    del motion, xyz, rotation
    for frame in range(first + 1, target_frame + 1):
        report = json.loads((run / 'frame_summary' / f'frame_{frame:03d}.json').read_text())['stream']
        motion_path = Path(report['motion_export']['path'])
        birth_export = report.get('incremental_export')
        birth_path = Path(birth_export['path']) if birth_export else None
        raw, detail = decoder.advance(motion_path, birth_path, raw)
        digest = residual.reference_digest(raw)
        if digest != report['reference_sha256']:
            raise ValueError(f'Recovered materialized state disagrees with frame {frame}')
        rows = residual.survivor_rows(len(origins), detail['deleted_rows'])
        origins = torch.cat((origins[rows], torch.full((detail['birth_count'],), frame, dtype=torch.long)))
        if len(origins) != len(raw['xyz']):
            raise ValueError('Recovered point origins disagree with GS order')
        verified.append(dict(frame=frame, reference_sha256=digest,
                             birth_count=detail['birth_count'], deleted_count=len(detail['deleted_rows'])))
        print(f'Recovered and hash-verified frame {frame}', flush=True)
    restored = motion_from_decoded(motion_config, decoder.last_frame_state, decoder.materialized_layout)
    restored.tree_setup = report['tree_setup']
    restored.stage_seconds = dict(report['stage_seconds'])
    restored.last_metrics = dict(report['last_training_metrics'])
    xyz, rotation = restored(motion_config.motion_steps)
    old_count = len(restored.base_xyz)
    post_motion = {name: raw[name][:old_count] for name in residual.RAW_FIELDS}
    post_motion.update(xyz=xyz, rotation=rotation)
    if residual.reference_digest(post_motion) != decoder.last_frame_state['header']['post_motion_digest']:
        raise ValueError('Recovered motion poses differ from the exact decoded state')
    deleted, _ = restored.compensation_deletions(raw['opacity'], decoder.previous_birth_count)
    expected = sab._delete_rows(raw, decoder.previous_birth_count, decoder.previous_changed)
    torch.testing.assert_close(deleted, expected, rtol=0, atol=0)
    gaussian_state = list(packet['gaussians'])
    for name, slot in slots.items():
        gaussian_state[slot] = raw[name]
    count = len(raw['xyz'])
    gaussian_state[7] = raw['xyz'].new_zeros(count)
    gaussian_state[8] = raw['xyz'].new_zeros(count, 1)
    gaussian_state[9] = raw['xyz'].new_zeros(count, 1)
    gaussian_state[10] = dict(gaussian_state[10], state={})
    packet.update(current_frame=target_frame, global_step=config['trainer']['num_incr_steps'],
        gaussians=tuple(gaussian_state), gaussians_incr=None, quantized_residual_motion=restored.capture())
    packet['quantized_frame_state'] = dict(packet['quantized_frame_state'],
        previous_new_count=decoder.previous_birth_count, previous_reference=raw,
        deleted_indices=torch.tensor(detail['deleted_rows'], device=device, dtype=torch.long),
        report=report, finalized=True, stats_window_initialized=True,
        birth_control=report['birth_control'], point_birth_frame=origins)
    packet['native_recovery'] = dict(source_run=str(run), source_checkpoint=str(checkpoint),
        restored_frame=target_frame, verified_frames=verified,
        semantics='Exact native GS state, selected motion symbols/precision, deletion policy and cached topology. '
            'Hidden candidates, learned logits, Adam moments and original RNG state are unavailable. '
            'Use to continue at the next frame, whose motion initializes afresh.')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_packet(packet), output)
    output.with_suffix('.recovery.json').write_text(json.dumps(packet['native_recovery'], indent=2), encoding='utf-8')
    print(f'Recovered checkpoint: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    with torch.no_grad():
        recover(args.run, args.checkpoint, args.frame, args.output, args.device)


if __name__ == '__main__':
    main()
