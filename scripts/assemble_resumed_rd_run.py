"""Create a provenance-labelled analysis view of two completed frame segments.

Never edits source runs or copies checkpoints. Not a claim of uninterrupted or
bit-exact training resume. Independent stream audit must be run afterwards.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import yaml


def read_rows(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n')[:-1] if line]


def select_rows(prefix, suffix, boundary, last):
    return ([row for row in prefix if 1 <= row['frame'] <= boundary]
            + [row for row in suffix if boundary < row['frame'] <= last])


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assemble(prefix, suffix, boundary):
    prefix, suffix = prefix.resolve(), suffix.resolve()
    configs = [yaml.safe_load((p / 'config.yaml').read_text()) for p in (prefix, suffix)]
    last = int(configs[0]['data']['extra_dataset_kwargs']['num_frames']) - 1
    if not 1 <= boundary < last:
        raise ValueError('Invalid segment boundary')
    steps = int(configs[0]['trainer']['num_incr_steps'])
    checkpoint = prefix / f'ckpt-{boundary}-{steps}.pth'
    if Path(configs[1]['ckpt_path']).resolve() != checkpoint or not checkpoint.is_file():
        raise ValueError('Suffix did not resume the specified boundary checkpoint')
    normalized = copy.deepcopy(configs)
    for config in normalized:
        for key in ('experiment_name', 'output_dir', 'ckpt_path'):
            config.pop(key, None)
    if normalized[0] != normalized[1]:
        raise ValueError('Training configurations differ beyond resume paths')
    manifests = [json.loads((p / 'controlled_manifest.json').read_text())
                 for p in (prefix, suffix)]
    for key in ('seed', 'source_sha256', 'frame_seed', 'sampler_seed'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f'Resume mismatch: {key}')
    logs = {}
    for name, count in (('training_views.jsonl', steps), ('evaluation.jsonl', 21),
                        ('frame_times.jsonl', 1)):
        logs[name] = select_rows(read_rows(prefix / name), read_rows(suffix / name),
                                 boundary, last)
        for frame in range(1, last + 1):
            frame_rows = [row for row in logs[name] if row['frame'] == frame]
            if len(frame_rows) != count:
                raise ValueError(f'Incomplete or duplicate {name} frame {frame}')
            if name == 'training_views.jsonl' and [r['step'] for r in frame_rows] != list(range(1, steps + 1)):
                raise ValueError(f'Training step sequence mismatch: {frame}')
    if not (suffix / f'ckpt-{last}-{steps}.pth').is_file():
        raise ValueError('Suffix final checkpoint missing')
    sources = {}
    for frame in range(1, last + 1):
        run = prefix if frame <= boundary else suffix
        for name in (f'frame_{frame:03d}.json', f'motion_{frame:03d}.rdp',
                     f'motion_{frame:03d}.rdp.json', f'incremental_{frame:03d}.ply'):
            path = run / 'frame_storage' / name
            if not path.is_file():
                raise ValueError(f'Missing artifact: {path}')
            sources['frame_storage/' + name] = path
        path = run / f'ckpt-{frame}-{steps}.pth'
        if path.is_file():
            sources[path.name] = path
    output = prefix.parent.parent / ('idea_resumed_rd_composite_'
                                     + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    (output / 'frame_storage').mkdir()
    for name, path in sources.items():
        (output / name).symlink_to(path)
    for name in ('source_snapshot.zip', 'source_snapshot_manifest.json'):
        (output / name).symlink_to(prefix / name)
    lineage = dict(kind='resumed_segment_analysis_view', prefix=str(prefix), suffix=str(suffix),
                   boundary=boundary, prefix_frames=[1, boundary], suffix_frames=[boundary + 1, last],
                   checkpoint_sha256=file_hash(checkpoint),
                   initial_checkpoint=str(configs[0]['ckpt_path']),
                   uninterrupted_training=False, bit_exact_resume=False,
                   independent_stream_audit_completed=False,
                   note='Overlapping prefix tail and suffix boundary no-op excluded. '
                        'Wall means combine selected actual frame calls; restart overhead excluded.',
                   source_log_sha256={str(run / name): file_hash(run / name)
                                      for run in (prefix, suffix) for name in logs},
                   assembly_script_sha256=file_hash(Path(__file__)))
    for name, rows in logs.items():
        with (output / name).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    (output / 'config.yaml').write_text(yaml.safe_dump(configs[0], sort_keys=False))
    manifest = dict(manifests[0], analysis_lineage=lineage)
    (output / 'controlled_manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'resume_lineage.json').write_text(json.dumps(lineage, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--suffix', type=Path, required=True)
    parser.add_argument('--boundary', type=int, required=True)
    args = parser.parse_args()
    print(assemble(args.prefix, args.suffix, args.boundary))
