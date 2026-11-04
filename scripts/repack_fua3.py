"""Copy a committed run prefix and losslessly repack FUA2 to FUA3.

This never writes to the source. It requires a new output run directory and
verifies both sequential decodes, all six float32 attribute bit patterns,
survivor/birth mappings and saved state digests on every frame. Frame summaries
are copied with packet-size accounting and paths updated for benchmarking.
"""
import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.model import full_attribute_update as fua
from pipeline.model.birth_quantization import reference_digest
from pipeline.model.rigid_tree_codec import DecodedState
from scripts import benchmark_rigid_residual_tree as benchmark
from scripts.replay_rigid_tree import decode_frame, fields


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_code_hash(root):
    """Hash main.py and pipeline/**/*.py; external libraries are not covered."""
    root = root.resolve()
    main, pipeline = root / 'main.py', root / 'pipeline'
    if not main.is_file() or not (pipeline / 'model/full_attribute_update.py').is_file():
        raise ValueError('Training source must contain main.py and pipeline/model/full_attribute_update.py')
    files = [main] + sorted(pipeline.rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError('Training source contains an external Python file link')
        relative = path.relative_to(root).as_posix().encode()
        digest.update(struct.pack('<I', len(relative)))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return dict(path=str(root), sha256=digest.hexdigest(), file_count=len(files),
                scope='main.py and pipeline/**/*.py; excludes external libraries and data',
                attribution='Supplied by caller; source run does not independently attest code identity')


def repack_packet(packet, *, frame_index):
    """Return the shorter original/FUA3 packet, after strict FUA2 validation.

    FUA1 stays as-is to preserve its legacy decoder semantics, and FUA3 is
    validated and left as-is. The source reference hash is preserved verbatim;
    sequential replay supplies and validates its actual reference.
    """
    if len(packet) < fua.HEADER.size:
        raise ValueError('Truncated full attribute packet')
    magic, frame, count, rest, *tail = fua.HEADER.unpack_from(packet)
    steps, length, prior, checksum = tuple(tail[:3]), tail[3], tail[4], tail[5]
    dense_expected = count * (6 + 3 * rest) * 2
    sparse_max = 12 + count * (15 + 2 * (6 + 3 * rest))
    if (magic not in (fua.MAGIC, fua.SPARSE_MAGIC, fua.PLANAR_MAGIC) or
            frame != frame_index or count < 1 or
            (magic == fua.MAGIC and length != dense_expected) or
            (magic != fua.MAGIC and not 12 <= length <= sparse_max) or
            length > fua.MAX_BODY or any(not math.isfinite(s) or s <= 0 for s in steps)):
        raise ValueError('Invalid full attribute packet metadata')
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(packet[fua.HEADER.size:], length + 1)
    except zlib.error as exc:
        raise ValueError('Damaged full attribute packet') from exc
    if len(raw) != length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Invalid full attribute compressed length')
    prefix = fua.HEADER.pack(magic, frame, count, rest, *steps, length, prior, 0)
    if zlib.crc32(prefix + raw) != checksum:
        raise ValueError('Full attribute checksum mismatch')
    if magic == fua.MAGIC:
        return packet
    if magic == fua.PLANAR_MAGIC:
        fua._sparse_zigzag_byteplane(raw, count, rest, inverse=True)
        return packet
    planar = fua._sparse_zigzag_byteplane(raw, count, rest)
    candidate = fua._encode_body(fua.PLANAR_MAGIC, planar, frame, count, rest, steps, prior)
    # FUA3 conversion can never enlarge the complete committed packet.
    return min((packet, candidate), key=len)


def _reference(path):
    with np.load(path, allow_pickle=False) as packet:
        raw = {key: torch.from_numpy(packet[key].copy()) for key in fields}
        state = DecodedState.from_reference(packet)
    return state, raw


def _equal_optional(a, b):
    return (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))


def _check_frame(manifest, old_state, old_raw, new_state, new_raw):
    if (old_state.digest() != new_state.digest() or
            old_state.digest().hex() != manifest['decoded_state_sha256'] or
            reference_digest(old_raw) != manifest['raw_state_sha256'] or
            reference_digest(new_raw) != manifest['raw_state_sha256']):
        raise ValueError('Repacked frame digest differs from original manifest')
    for name in ('survivor_rows', 'birth_sources', 'gs_translation_rows'):
        if not _equal_optional(getattr(old_state, name), getattr(new_state, name)):
            raise ValueError(f'Repacked frame {name} differs')
    old_count = len(old_state.survivor_rows)
    for name in fields:
        a, b = old_raw[name].contiguous(), new_raw[name].contiguous()
        if a.shape != b.shape or a.dtype != b.dtype or a.numpy().tobytes() != b.numpy().tobytes():
            raise ValueError(f'Repacked {name} differs')
        if (a[:old_count].numpy().tobytes() != b[:old_count].numpy().tobytes() or
                a[old_count:].numpy().tobytes() != b[old_count:].numpy().tobytes()):
            raise ValueError(f'Repacked old/newborn {name} differs')


def repack_run(source_run: Path, output_run: Path, through: int, *, training_source: Path):
    source_run, output_run = source_run.resolve(), output_run.resolve()
    if not source_run.is_dir() or not (source_run / 'frame_storage/rigid_tree').is_dir():
        raise ValueError('Source must be an existing run with a rigid-tree stream')
    if not (source_run / 'config.yaml').is_file():
        raise ValueError('Source run must contain its saved config.yaml')
    if output_run.exists() or output_run.is_relative_to(source_run) or source_run.is_relative_to(output_run):
        raise ValueError('Output must be a new directory outside the source run')
    if through < 1:
        raise ValueError('through must be at least 1')
    training_code = _training_code_hash(training_source)
    decoder_code = _training_code_hash(ROOT)
    source_config = source_run / 'config.yaml'
    saved_config = OmegaConf.to_container(OmegaConf.load(source_config), resolve=False)
    if not isinstance(saved_config, dict) or not isinstance(saved_config.get('experiment_name'), str) or not saved_config['experiment_name']:
        raise ValueError('Source configuration needs an experiment_name')
    repacked_config = dict(saved_config)
    repacked_config['experiment_name'] = saved_config['experiment_name'] + '_fua3_repack_20261104'
    stream = source_run / 'frame_storage/rigid_tree'
    summary_dir = source_run / 'frame_summary'
    # Validate required manifest/summary names before creating any artifact.
    manifests, summaries = {}, {}
    for frame in range(1, through + 1):
        manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
        summary = json.loads((summary_dir / f'frame_{frame:03d}.json').read_text())
        if manifest.get('frame') != frame:
            raise ValueError('Frame manifest index differs')
        reference = manifest.get('reference')
        if (not isinstance(reference, str) or
                Path(reference).parts[:2] != ('frame_storage', 'rigid_tree') or
                len(Path(reference).parts) != 3 or
                Path(reference).name != 'reference_frame0.npz'):
            raise ValueError('Unsupported shared reference path')
        reference_path = source_run / reference
        if not reference_path.is_file():
            raise ValueError('Missing reference')
        benchmark._actual_frame_storage(stream, frame, manifest, summary,
                                        reference_path.stat().st_size, reference)
        manifests[frame], summaries[frame] = manifest, summary
    output_run.mkdir(parents=True, exist_ok=False)
    incomplete = output_run / 'REPACK_INCOMPLETE'
    incomplete.write_text('Repack is incomplete or verification failed; do not treat this directory as verified.\n')
    shutil.copyfile(source_config, output_run / 'source_config.yaml')
    OmegaConf.save(OmegaConf.create(repacked_config), output_run / 'config.yaml')
    check_config = OmegaConf.to_container(OmegaConf.load(output_run / 'config.yaml'), resolve=False)
    if check_config != repacked_config:
        raise ValueError('Repacked config differs beyond experiment_name')
    out_stream = output_run / 'frame_storage/rigid_tree'
    out_summary = output_run / 'frame_summary'
    out_stream.mkdir(parents=True)
    out_summary.mkdir()
    shutil.copyfile(stream / 'reference_frame0.npz', out_stream / 'reference_frame0.npz')
    records, adjusted_summaries = [], {}
    for frame in range(1, through + 1):
        manifest, summary = manifests[frame], summaries[frame]
        shutil.copyfile(stream / f'frame_{frame:03d}.json', out_stream / f'frame_{frame:03d}.json')
        before, after, old_magic, new_magic = 0, 0, None, None
        for key in ('motion', 'birth', 'opacity_update', 'feature_dc_update', 'logscale_update', 'full_attribute_update'):
            name = manifest.get(key)
            if name is None:
                continue
            source = stream / name
            if key == 'full_attribute_update':
                packet = source.read_bytes()
                replacement = repack_packet(packet, frame_index=frame)
                (out_stream / name).write_bytes(replacement)
                before, after = len(packet), len(replacement)
                old_magic, new_magic = packet[:4].decode(), replacement[:4].decode()
                summary[key]['archive_bytes'] = after
                summary[key]['format'] = new_magic
            else:
                shutil.copyfile(source, out_stream / name)
            if isinstance(summary.get(key), dict) and 'path' in summary[key]:
                summary[key]['path'] = str(out_stream / name)
        for name in ('packet_bytes', 'encoded_bytes', 'complete_dynamic_update_bytes'):
            if name in summary:
                summary[name] += after - before
        benchmark._actual_frame_storage(out_stream, frame, manifest, summary,
                                        (out_stream / 'reference_frame0.npz').stat().st_size,
                                        manifest['reference'])
        if manifest.get('full_attribute_update'):
            written = (out_stream / manifest['full_attribute_update']).read_bytes()
            if (len(written) != summary['full_attribute_update']['archive_bytes'] or
                    written[:4].decode() != summary['full_attribute_update']['format']):
                raise ValueError('Repacked format or bytes disagree with summary')
        adjusted_summaries[frame] = summary
        records.append(dict(frame=frame, original_fua_bytes=before, repacked_fua_bytes=after,
                            original_format=old_magic, repacked_format=new_magic))
    old_state, old_raw = _reference(stream / 'reference_frame0.npz')
    new_state, new_raw = _reference(out_stream / 'reference_frame0.npz')
    for frame in range(1, through + 1):
        old_state, old_raw = decode_frame(stream, frame, old_state, old_raw)
        new_state, new_raw = decode_frame(out_stream, frame, new_state, new_raw)
        _check_frame(manifests[frame], old_state, old_raw, new_state, new_raw)
        records[frame - 1].update(decoded_state_sha256=old_state.digest().hex(),
                                  raw_state_sha256=reference_digest(old_raw),
                                  surviving_old_count=len(old_state.survivor_rows),
                                  birth_count=len(old_state.xyz) - len(old_state.survivor_rows),
                                  bit_exact_six_fields_and_sources=True)
    # Only after the entire prefix passed bitwise validation may copied
    # evaluation values appear in artifact summaries, clearly marked inherited.
    for frame, summary in adjusted_summaries.items():
        record = records[frame - 1]
        if 'full_attribute_rate_diagnostics' in summary:
            summary['source_full_attribute_rate_diagnostics'] = summary.pop('full_attribute_rate_diagnostics')
        summary['lossless_repack_provenance'] = dict(
            source_run=str(source_run), verified_prefix_through=through,
            all_six_fields_and_sources_bit_exact=True, new_training_steps=0,
            psnr_retested=False, evaluation='Inherited from source after bitwise verification',
            training_losses_and_rate_diagnostics='Source training diagnostics; not recomputed',
            original_fua_bytes=record['original_fua_bytes'], repacked_fua_bytes=record['repacked_fua_bytes'])
        (out_summary / f'frame_{frame:03d}.json').write_text(json.dumps(summary, indent=2) + '\n')
    report = dict(source_run=str(source_run), through=through, verified=True,
                  output_run=str(output_run), experiment_name=repacked_config['experiment_name'],
                  source_experiment_name=saved_config['experiment_name'], new_training_steps=0,
                  psnr_retested=False, evaluation='Inherited from source after bitwise verification',
                  source_training_config_sha256=_sha256_file(source_config),
                  source_training_code=training_code, decoder_code=decoder_code,
                  decoder_fua_sha256=_sha256_file(ROOT / 'pipeline/model/full_attribute_update.py'),
                  repack_script_sha256=_sha256_file(Path(__file__).resolve()),
                  artifact_config_sha256=_sha256_file(output_run / 'config.yaml'),
                  config_note='source_config.yaml is original; config.yaml changes experiment_name only; mode and training values describe source',
                  frames=records)
    (output_run / 'fua3_repack_report.json').write_text(json.dumps(report, indent=2) + '\n')
    incomplete.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('output_run', type=Path, help='New output directory outside source_run')
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--training-source', type=Path, required=True,
                        help='Frozen source checkout used for original training; hashed for provenance')
    args = parser.parse_args()
    result = repack_run(args.source_run, args.output_run, args.through,
                        training_source=args.training_source)
    print(json.dumps({'output_run': result['output_run'], 'through': result['through'],
                      'verified': result['verified']}, indent=2))


if __name__ == '__main__':
    main()
