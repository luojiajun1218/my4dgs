"""Losslessly repack a completed rigid-tree stream into its own benchmarkable run.

The source run is read only. This tool copies the necessary decoder reference
and birth packets, rewrites only motion sparse row IDs, and verifies every
decoded motion/raw state against the source manifests. It never copies
checkpoints, images, training logs, or dataset contents.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
codec = importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec')
birth_codec = importlib.import_module(
    'pipeline.ReCon-GS.model.rigid_tree_birth_codec')
quantization = importlib.import_module(
    'pipeline.ReCon-GS.model.birth_quantization')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _reference(path):
    with np.load(path, allow_pickle=False) as packet:
        raw = {name: torch.from_numpy(packet[name].copy())
               for name in quantization.RAW_FIELDS}
        state = codec.DecodedState(
            raw['xyz'], raw['rotation'],
            torch.from_numpy(packet['owner'].copy()).long(),
            torch.from_numpy(packet['parents'].copy()).long(),
            int(packet['frame_index']))
    if state.frame_index != 0:
        raise ValueError('A full-run repack needs a frame0 decoder reference')
    return state, raw


def _validated_input(source, through, expected):
    stream = source / 'frame_storage' / 'rigid_tree'
    summaries = source / 'frame_summary'
    first = json.loads((stream / 'frame_001.json').read_text())
    relative_reference = first['reference']
    if relative_reference != 'frame_storage/rigid_tree/reference_frame0.npz':
        raise ValueError('Expected the standard frame0 decoder reference path')
    reference = (source / relative_reference).resolve(strict=True)
    if not reference.is_relative_to(source):
        raise ValueError('Decoder reference must stay inside the source run')
    if reference.name != 'reference_frame0.npz':
        raise ValueError('Expected an independent frame0 decoder reference')
    for frame in range(1, through + 1):
        manifest_path = stream / f'frame_{frame:03d}.json'
        summary_path = summaries / f'frame_{frame:03d}.json'
        motion_path = stream / f'motion_{frame:03d}.bin'
        for path in (manifest_path, summary_path, motion_path):
            if not path.is_file():
                raise FileNotFoundError(f'Incomplete source frame {frame}: {path}')
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('frame') != frame or
                manifest.get('reference') != relative_reference or
                manifest.get('motion') != motion_path.name):
            raise ValueError(f'Inconsistent source manifest for frame {frame}')
        birth_name = manifest.get('birth')
        if birth_name is not None:
            if birth_name != f'birth_{frame:03d}.bin' or not (
                    stream / birth_name).is_file():
                raise ValueError(f'Incomplete birth packet for frame {frame}')
    if through == expected:
        if (stream / f'motion_{expected + 1:03d}.bin').exists():
            raise ValueError('Source has motion frames beyond configured num_frames')
    return stream, relative_reference, reference


def repack(source, destination, through=None):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if not source.is_dir() or destination == source or source in destination.parents:
        raise ValueError('Destination must be a separate run directory')
    if destination.exists():
        raise FileExistsError(destination)
    config_path = source / 'config.yaml'
    config = yaml.safe_load(config_path.read_text())
    expected = int(config['data']['extra_dataset_kwargs']['num_frames']) - 1
    if expected < 1:
        raise ValueError('Source config needs dynamic frames')
    target = expected if through is None else int(through)
    if target < 1 or target > expected:
        raise ValueError('through must be between frame1 and configured final frame')
    source_stream, relative_reference, source_reference = _validated_input(
        source, target, expected)
    state, raw = _reference(source_reference)
    source_reference_bytes = source_reference.stat().st_size
    staging = destination.with_name(destination.name + '.building-' +
                                    uuid.uuid4().hex[:8])
    staging.mkdir(parents=True, exist_ok=False)
    dest_stream = staging / 'frame_storage' / 'rigid_tree'
    dest_summary = staging / 'frame_summary'
    dest_stream.mkdir(parents=True)
    dest_summary.mkdir()
    try:
        shutil.copyfile(config_path, staging / 'config.yaml')
        shutil.copyfile(source_reference, dest_stream / source_reference.name)
        copied_reference = dest_stream / source_reference.name
        if (_sha256(copied_reference) != _sha256(source_reference) or
                copied_reference.stat().st_size != source_reference_bytes):
            raise ValueError('Copied decoder reference changed')
        rows = []
        for frame in range(1, target + 1):
            old_motion_path = source_stream / f'motion_{frame:03d}.bin'
            old_motion = old_motion_path.read_bytes()
            original_info = codec.motion_packet_info(old_motion)
            new_motion = codec.repack_legacy_motion_packet(old_motion)
            new_info = codec.motion_packet_info(new_motion)
            old_state = codec.decode_motion(old_motion, state)
            new_state = codec.decode_motion(new_motion, state)
            if (old_state.digest() != new_state.digest() or
                    not torch.equal(old_state.birth_sources,
                                    new_state.birth_sources)):
                raise ValueError(f'Frame {frame} motion state changed in repack')
            manifest = json.loads(
                (source_stream / f'frame_{frame:03d}.json').read_text())
            summary = json.loads(
                (source / 'frame_summary' / f'frame_{frame:03d}.json').read_text())
            if (manifest['decoded_state_sha256'] != new_state.digest().hex() or
                    manifest['old_count'] != len(state.xyz) or
                    manifest['total_count'] != len(new_state.xyz) or
                    summary['motion']['bytes'] != len(old_motion) or
                    summary['encoded_bytes'] != len(old_motion) +
                    (summary['birth']['archive_bytes'] if summary.get('birth') else 0) or
                    summary['shared_reference_bytes'] != source_reference_bytes or
                    summary['decoder_reference'] != relative_reference):
                raise ValueError(f'Frame {frame} source storage metadata differs')
            current = {name: value.clone() for name, value in raw.items()}
            current['xyz'] = new_state.xyz[:len(state.xyz)]
            current['rotation'] = new_state.rotation[:len(state.xyz)]
            birth_name = manifest['birth']
            birth_bytes = 0
            if birth_name is not None:
                birth_path = source_stream / birth_name
                birth_packet = birth_path.read_bytes()
                birth_bytes = len(birth_packet)
                if (summary['birth']['archive_bytes'] != birth_bytes or
                        manifest['birth_count'] != len(new_state.birth_sources)):
                    raise ValueError(f'Frame {frame} source birth metadata differs')
                newborn = birth_codec.decode_birth_attributes(
                    birth_packet, current, new_state.xyz[len(state.xyz):],
                    new_state.rotation[len(state.xyz):], new_state.birth_sources)
                if not torch.equal(new_state.owner[len(state.xyz):],
                                   new_state.owner[:len(state.xyz)][
                                       new_state.birth_sources]):
                    raise ValueError(f'Frame {frame} birth owner changed')
                for name in quantization.RAW_FIELDS:
                    current[name] = torch.cat((current[name], newborn[name]))
                (dest_stream / birth_name).write_bytes(birth_packet)
            elif manifest['birth_count'] != 0:
                raise ValueError(f'Frame {frame} omitted a required birth packet')
            digest = quantization.reference_digest(current)
            if (manifest['raw_state_sha256'] != digest or
                    summary['materialized_sha256'] != digest):
                raise ValueError(f'Frame {frame} raw appearance/pose changed')
            dest_motion_path = dest_stream / f'motion_{frame:03d}.bin'
            dest_motion_path.write_bytes(new_motion)
            if dest_motion_path.stat().st_size != new_info['stored_bytes']:
                raise ValueError(f'Frame {frame} motion write was incomplete')
            manifest.update(motion_bytes=len(new_motion),
                            motion_format=new_info['format'],
                            motion_raw_stream_bytes=new_info['raw_stream_bytes'],
                            transcode_source_motion_sha256=hashlib.sha256(
                                old_motion).hexdigest())
            (dest_stream / f'frame_{frame:03d}.json').write_text(
                json.dumps(manifest, indent=2))
            original_hard = dict(summary.get('motion_rate_diagnostics', {}))
            summary['source_training_motion_rate_diagnostics'] = original_hard
            for name in ('rate_bytes_hard', 'rate_bytes_hard_uncompressed'):
                if name in summary.get('motion_rate_diagnostics', {}):
                    summary['motion_rate_diagnostics'][name] = float(
                        new_info['raw_stream_bytes'])
            summary['motion'].update(path=str(destination /
                                              'frame_storage' / 'rigid_tree' /
                                              dest_motion_path.name),
                                     bytes=len(new_motion),
                                     payload_bytes=len(new_motion),
                                     format=new_info['format'],
                                     stored_bytes=len(new_motion),
                                     raw_stream_bytes=new_info['raw_stream_bytes'],
                                     compression_savings_bytes=(
                                         new_info['compression_savings_bytes']))
            if birth_name is not None:
                summary['birth']['path'] = str(destination / 'frame_storage' /
                                               'rigid_tree' / birth_name)
            summary['encoded_bytes'] = len(new_motion) + birth_bytes
            summary['frame_manifest'] = str(Path(relative_reference).parent /
                                            f'frame_{frame:03d}.json')
            original_checkpoint = summary.get('checkpoint')
            summary['source_training_checkpoint'] = (
                str((source / original_checkpoint).resolve())
                if original_checkpoint and not Path(original_checkpoint).is_absolute()
                else original_checkpoint)
            summary['checkpoint'] = None  # No checkpoint is copied into this run.
            summary['transcode_source'] = dict(
                run=str(source), motion_path=str(old_motion_path),
                motion_sha256=hashlib.sha256(old_motion).hexdigest(),
                motion_bytes=len(old_motion), motion_format=original_info['format'],
                raw_stream_bytes=original_info['raw_stream_bytes'],
                source_summary=str(source / 'frame_summary' /
                                   f'frame_{frame:03d}.json'))
            (dest_summary / f'frame_{frame:03d}.json').write_text(
                json.dumps(summary, indent=2))
            rows.append(dict(frame=frame, source_motion_bytes=len(old_motion),
                             repacked_motion_bytes=len(new_motion),
                             birth_bytes=birth_bytes,
                             original_decoded_state_sha256=old_state.digest().hex(),
                             repacked_decoded_state_sha256=new_state.digest().hex(),
                             raw_state_sha256=digest))
            state, raw = new_state, current
        report = dict(schema='rigid_tree_lossless_sparse_repack_v1',
                      source_run=str(source), output_run=str(destination),
                      source_experiment_name=config['experiment_name'],
                      source_config_sha256=_sha256(config_path),
                      codec_file_sha256=_sha256(Path(codec.__file__)),
                      repack_script_sha256=_sha256(Path(__file__)),
                      source_reference=str(source_reference),
                      reference_sha256=_sha256(copied_reference),
                      reference_bytes=source_reference_bytes,
                      configured_dynamic_frames=expected,
                      repacked_dynamic_frames=target,
                      complete=(target == expected),
                      partial=(target != expected),
                      codec='positive sparse row gaps; all other motion symbols '
                            'and birth bytes unchanged',
                      checkpoints_copied=False,
                      training_metrics_preserved=True,
                      source_motion_bytes=sum(row['source_motion_bytes'] for row in rows),
                      repacked_motion_bytes=sum(row['repacked_motion_bytes']
                                                for row in rows),
                      birth_bytes=sum(row['birth_bytes'] for row in rows),
                      frames=rows)
        report['source_reference_and_packets_bytes'] = (
            source_reference_bytes + report['source_motion_bytes'] +
            report['birth_bytes'])
        report['repacked_reference_and_packets_bytes'] = (
            source_reference_bytes + report['repacked_motion_bytes'] +
            report['birth_bytes'])
        report['saved_packet_bytes'] = (report['source_motion_bytes'] -
                                        report['repacked_motion_bytes'])
        if destination.exists():
            raise FileExistsError(destination)
        (staging / 'stream_repack.json').write_text(json.dumps(report, indent=2))
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-run', type=Path, required=True)
    parser.add_argument('--through', type=int,
                        help='Explicit partial CPU check; default requires the full '
                             'configured frame sequence')
    args = parser.parse_args()
    report = repack(args.source_run, args.output_run, args.through)
    print(json.dumps({name: report[name] for name in (
        'output_run', 'complete', 'repacked_dynamic_frames',
        'source_motion_bytes', 'repacked_motion_bytes', 'birth_bytes')}))


if __name__ == '__main__':
    main()
