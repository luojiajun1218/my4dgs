"""Decode consecutive motion streams from shared frame0, not training latents."""
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_motion')
G = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
C = importlib.import_module('pipeline.ReCon-GS.module')
CODEC = importlib.import_module('pipeline.ReCon-GS.model.motion_codec')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--compact-zero-subtrees', action='store_true',
                        help='Also save losslessly compacted packets in the new audit directory')
    args = parser.parse_args()
    audit_started = time.perf_counter()
    run = args.run.resolve()
    config = yaml.safe_load((run / 'config.yaml').read_text())
    initial = Path(config['ckpt_path'])
    checkpoint = torch.load(initial, map_location='cuda', weights_only=False)
    if checkpoint['current_frame'] != 0 or not config['module']['merge_to_base']:
        raise ValueError('Audit requires shared frame0 and merge-to-base protocol')
    model = G.GaussianModel(config['module']['sh_degree'])
    model.restore(checkpoint['gaussians'], C.GaussianConfig(**config['module']['gaussian']))
    del checkpoint
    records, hashes = [], {str(initial): digest(initial)}
    fields = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
    tuple_indices = (1, 2, 3, 4, 5, 6)
    storage = run / 'frame_storage'
    total_frames = config['data']['extra_dataset_kwargs']['num_frames']
    incremental = None
    output = ROOT / 'output' / ('idea_rd_tree_decode_audit_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    if args.compact_zero_subtrees:
        from zero_subtree_compaction import compact_packet
        (output / 'compacted_motion').mkdir()
    for frame in range(1, total_frames):
        torch.cuda.synchronize()
        decode_started = time.perf_counter()
        if incremental is not None:
            count = len(incremental.get_xyz)
            if count:
                model.extend(incremental)
                indices = model.get_opacity.flatten().sort().indices
                model.prune_points(indices[:count])
        packet_path = storage / f'motion_{frame:03d}.rdp'
        protocol_path = packet_path.with_suffix('.rdp.json')
        protocol = json.loads(protocol_path.read_text())
        decoded = M.decode_tree_motion(packet_path.read_bytes(),
                                      model.get_xyz_ori.detach().cpu().double().numpy(), protocol)
        compaction = None
        if args.compact_zero_subtrees:
            compact_started = time.perf_counter()
            xyz = model.get_xyz_ori.detach().cpu().double().numpy()
            tree = M.build_candidate_tree(xyz, protocol['max_depth'], protocol['min_leaf_size'],
                                          protocol['partition'], protocol['clustering_iterations'])
            compacted, compaction = compact_packet(packet_path.read_bytes(), tree)
            compact_field = M.decode_projection(compacted, tree)
            if not np.array_equal(compact_field, decoded):
                raise AssertionError('Compacted motion differs from original decoded motion')
            # Apply the compacted stream, not just validate it alongside original application.
            decoded = compact_field
            target = output / 'compacted_motion' / packet_path.name
            with target.open('xb') as stream:
                stream.write(compacted)
            target.with_suffix('.rdp.json').symlink_to(protocol_path)
            compaction['verification_and_recode_seconds'] = time.perf_counter() - compact_started
            compaction['packet_sha256'] = digest(target)
            del tree, compact_field, xyz
        value = torch.as_tensor(decoded, dtype=model.get_xyz_ori.dtype, device='cuda')
        model.apply_deformation(value[:, :3], CODEC.rotation_vector_to_quaternion(value[:, 3:]))
        torch.cuda.synchronize()
        decode_seconds = time.perf_counter() - decode_started
        # Checkpoints are opened only AFTER decoding, and only to compare output.
        checkpoint_path = run / f"ckpt-{frame}-{config['trainer']['num_incr_steps']}.pth"
        reference = (torch.load(checkpoint_path, map_location='cuda', weights_only=False)
                     if checkpoint_path.exists() else None)
        errors = {} if reference is not None else None
        if reference is not None:
            for name, index in zip(fields, tuple_indices):
                actual, expected = getattr(model, name).detach(), reference['gaussians'][index]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                errors[name] = float((actual - expected).abs().max()) if actual.numel() else 0.
        incremental_path = storage / f'incremental_{frame:03d}.ply'
        incremental = G.GaussianModel(config['module']['sh_degree'])
        incremental.load_ply(incremental_path)
        checked = reference is not None
        if checked:
            for name, index in zip(fields, tuple_indices):
                torch.testing.assert_close(getattr(incremental, name), reference['gaussians_incr'][index],
                                           rtol=0, atol=0)
            hashes[str(checkpoint_path)] = digest(checkpoint_path)
        del reference
        for path in (packet_path, protocol_path, incremental_path):
            hashes[str(path)] = digest(path)
        records.append({'frame': frame, 'gaussians': len(model.get_xyz),
                        'merge_tree_decode_apply_seconds': decode_seconds,
                        'incremental_gaussians': len(incremental.get_xyz),
                        'base_max_absolute_errors': errors, 'checkpoint_compared': checked,
                        'incremental_exact': True if checked else None})
        if compaction is not None:
            records[-1]['zero_subtree_compaction'] = compaction
        print(json.dumps(records[-1]), flush=True)
    if not records or not records[-1]['checkpoint_compared']:
        raise ValueError('Continuous decode audit requires a final checkpoint comparison')
    sources = [Path(__file__).resolve(), ROOT / 'pipeline/ReCon-GS/model/rd_tree_motion.py',
               ROOT / 'pipeline/ReCon-GS/model/rd_tree_projection.py',
               ROOT / 'pipeline/ReCon-GS/model/gaussian.py']
    if args.compact_zero_subtrees:
        sources.append(Path(__file__).with_name('zero_subtree_compaction.py'))
    (output / 'report.json').write_text(json.dumps(
        {'run': str(run), 'frames': records, 'hashes': hashes,
         'audit_wall_seconds': time.perf_counter() - audit_started,
         'applied_zero_subtree_compacted_stream': args.compact_zero_subtrees,
         'source_hashes': {str(p): digest(p) for p in sources},
         'scope': ('Same runtime/device/order protocol. Every packet is decoded; only available '
                   'checkpoints are compared, explicitly marked per frame. '
                   'Does not prove cross-platform bit-exactness.')}, indent=2) + '\n')
    print('Audit passed:', output, flush=True)


if __name__ == '__main__':
    main()
