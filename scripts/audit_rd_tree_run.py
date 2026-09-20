"""Read-only run audit; prefix checks do not claim full-sequence completion.

Checks packet headers and accounting, not independent motion reconstruction.
Use audit_rd_tree_stream.py separately for the latter.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

from omegaconf import OmegaConf


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(path):
    # Only terminated records: the writer may be appending during prefix audits.
    data = path.read_bytes()
    return [json.loads(line) for line in data.split(b'\n')[:-1] if line]


def by_frame(path):
    result = defaultdict(list)
    for row in rows(path):
        result[row['frame']].append(row)
    return result


def audit(run, reference=None, through=None):
    config = OmegaConf.load(run / 'config.yaml')
    last = int(config.data.extra_dataset_kwargs.num_frames) - 1
    end = last if through is None else through
    require(1 <= end <= last, 'Invalid audit prefix')
    steps = int(config.trainer.num_incr_steps)
    motion = int(config.module.deformation.num_stage1_steps)
    accum = int(config.module.deformation.rd_tree.accumulation_steps)
    views = by_frame(run / 'training_views.jsonl')
    ref_views = by_frame(reference / 'training_views.jsonl') if reference is not None else None
    evaluation = by_frame(run / 'evaluation.jsonl')
    times = by_frame(run / 'frame_times.jsonl')
    manifest = json.loads((run / 'controlled_manifest.json').read_text())
    hashes = json.loads((run / 'source_snapshot_manifest.json').read_text())
    with zipfile.ZipFile(run / 'source_snapshot.zip') as archive:
        require(set(archive.namelist()) == set(hashes), 'Snapshot membership')
        for name, digest in hashes.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest, name)
        for name, digest in manifest['source_sha256'].items():
            require(hashes['source/' + name] == digest, 'Training source mismatch')
    for frame in range(1, end + 1):
        require([r['step'] for r in views[frame]] == list(range(1, steps + 1)),
                f'Frame {frame}: training steps')
        if ref_views is not None:
            require(views[frame] == ref_views[frame], f'Frame {frame}: view sequence differs')
        paths = {r['path'] for r in views[frame]}
        require(len(paths) == 20, 'Expected 20 training views')
        ev = evaluation[frame]
        train = [r for r in ev if r['split'] == 'train']
        test = [r for r in ev if r['split'] == 'test']
        require(len(ev) == 21 and len(train) == 20 and len(test) == 1,
                f'Frame {frame}: incomplete/duplicate evaluation')
        require({r['path'] for r in train} == paths, 'Evaluation train view set')
        require(Path(test[0]['path']).stem == 'cam00' and test[0]['path'] not in paths,
                'Test view/leakage')
        require(all(r['step'] == steps and all(math.isfinite(r[k]) for k in
                    ('psnr', 'ssim', 'l1')) for r in ev), 'Nonfinite/wrong-step metrics')
        require(len(times[frame]) == 1 and math.isfinite(times[frame][0]['wall_seconds'])
                and times[frame][0]['wall_seconds'] > 0, 'Missing/invalid wall time')
        directory = run / 'frame_storage'
        stats = json.loads((directory / f'frame_{frame:03d}.json').read_text())
        packet = (directory / f'motion_{frame:03d}.rdp').read_bytes()
        require(len(packet) >= 80 and packet[:4] == b'RDP1', 'Packet magic/header')
        points, channels, bits = struct.unpack('<III', packet[4:16])
        require(points > 0 and channels == 6 and bits == stats['payload_bits'], 'Packet dimensions/bits')
        require(len(packet) == 32 + 8 * channels + (bits + 7) // 8, 'Packet length')
        padding = (-bits) % 8
        require(not padding or packet[-1] & ((1 << padding) - 1) == 0, 'Nonzero padding')
        protocol_path = directory / f'motion_{frame:03d}.rdp.json'
        protocol = json.loads(protocol_path.read_text())
        require(protocol['rd_tree_protocol'] == 1, 'Protocol version')
        for key in ('max_depth', 'min_leaf_size', 'partition', 'clustering_iterations'):
            require(protocol[key] == config.module.deformation.rd_tree[key], key)
        protocol_bytes = protocol_path.stat().st_size
        incr = (directory / f'incremental_{frame:03d}.ply').stat().st_size
        expected = dict(packet_bytes=len(packet), protocol_bytes=protocol_bytes,
                        motion_bytes=len(packet) + protocol_bytes, incremental_bytes=incr,
                        total_bytes=len(packet) + protocol_bytes + incr,
                        updates=motion // accum, projection_calls=motion // accum + 1)
        require(all(stats[k] == v for k, v in expected.items()), 'Storage/update accounting')
    return dict(run=str(run), frames_checked=end, configured_dynamic_frames=last,
                full_sequence=end == last,
                exact_training_views_match=True if ref_views is not None else None,
                snapshot_verified=True, packet_accounting_verified=True,
                independent_motion_reconstruction_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--same-schedule-reference', type=Path,
                        help='Optional same-seed peer; omitted means sequence matching is not checked')
    parser.add_argument('--through', type=int)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.same_schedule_reference, args.through), indent=2))
