"""Replay the necessary rigid-tree frame stream from a shared GS reference."""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
codec = importlib.import_module('pipeline.model.rigid_tree_codec')
birth_codec = importlib.import_module('pipeline.model.rigid_tree_birth_codec')
opacity_codec = importlib.import_module('pipeline.model.legacy_opacity_codec')
fields = ('xyz', 'rotation', 'logscale', 'opacity', 'sh_dc', 'sh_rest')


def apply_stream_opacity_update(directory, frame, current, old_count, manifest=None):
    """Apply optional old-row logits at the timing specified by the manifest."""
    if manifest is None:
        import json
        path = directory / f'frame_{frame:03d}.json'
        manifest = json.loads(path.read_text()) if path.exists() else {}
    name = manifest.get('opacity_update')
    expected = f'opacity_{frame:03d}.bin'
    if name not in (None, expected):
        raise ValueError('Opacity manifest filename differs from expected frame packet')
    path = directory / expected
    if name is None:
        if path.exists():
            raise ValueError('Unexpected opacity packet absent from frame manifest')
        return current
    current['opacity'][:old_count] = opacity_codec.decode_opacity_delta(
        path.read_bytes(), current['opacity'][:old_count], frame_index=frame).logits
    return current


def decode_frame(directory: Path, frame: int, state, raw):
    """Materialize survivors first, then decode births using surviving sources."""
    motion_file = directory / f'motion_{frame:03d}.bin'
    decoded = codec.decode_motion(motion_file.read_bytes(), state)
    rows = decoded.survivor_rows
    if rows is None:
        raise ValueError('Motion decoder lost previous-row survivor mapping')
    old_count = len(rows)
    birth_count = len(decoded.xyz) - old_count
    if any(len(value) != len(state.xyz) for value in raw.values()):
        raise ValueError('Previous raw attributes differ from decoded GS count')
    current = {name: value[rows].clone() for name, value in raw.items()}
    current['xyz'] = decoded.xyz[:old_count]
    current['rotation'] = decoded.rotation[:old_count]
    import json
    manifest_path = directory / f'frame_{frame:03d}.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    timing = manifest.get('opacity_update_timing', 'after_births')
    if timing not in ('before_births', 'after_births'):
        raise ValueError('Unknown opacity update timing')
    if timing == 'before_births':
        current = apply_stream_opacity_update(directory, frame, current, old_count, manifest)
    feature_name = manifest.get('feature_dc_update')
    expected_feature_name = f'shdc_{frame:03d}.bin'
    if feature_name not in (None, expected_feature_name):
        raise ValueError('DC feature manifest filename differs from expected packet')
    if manifest.get('feature_dc_update_timing', 'before_births') != 'before_births':
        raise ValueError('Unknown DC feature update timing')
    feature_path = directory / expected_feature_name
    if feature_name is not None:
        feature_codec = importlib.import_module('pipeline.model.sparse_features')
        current['sh_dc'] = feature_codec.decode_feature_dc_delta(
            feature_path.read_bytes(), current['sh_dc'], frame_index=frame,
            owner=decoded.owner[:old_count], node_count=len(decoded.parents))
    elif feature_path.exists():
        raise ValueError('Unexpected DC feature packet absent from frame manifest')
    birth_file = directory / f'birth_{frame:03d}.bin'
    if birth_count:
        newborn = birth_codec.decode_birth_attributes(
            birth_file.read_bytes(), current, decoded.xyz[old_count:],
            decoded.rotation[old_count:], decoded.birth_sources)
        sources = decoded.birth_sources
        if sources is None or len(sources) != birth_count:
            raise ValueError('Motion stream lost birth attachment rows')
        expected_owner = decoded.owner[:old_count][sources]
        if decoded.spatial_centers is not None:
            from importlib import import_module
            spatial = import_module('pipeline.model.spatial_gate')
            expected_paths = spatial.spatial_state_route(decoded.xyz[old_count:], decoded)
            expected_owner = spatial.spatial_owner(expected_paths, decoded.spatial_open)
        if not torch.equal(decoded.owner[old_count:], expected_owner):
            raise ValueError('Birth owner differs from encoded source owner')
        for name in fields:
            current[name] = torch.cat((current[name], newborn[name]))
        if (not torch.allclose(current['xyz'], decoded.xyz, rtol=1e-5, atol=1e-6) or
                not torch.allclose(current['rotation'], decoded.rotation, rtol=1e-5, atol=1e-6)):
            raise ValueError('Birth attributes disagree with decoded motion')
    elif birth_file.exists():
        raise ValueError('Unexpected birth stream for empty birth frame')
    if timing == 'after_births':
        current = apply_stream_opacity_update(directory, frame, current, old_count, manifest)
    return decoded, current


def replay(directory: Path, reference: Path, through: int):
    with np.load(reference, allow_pickle=False) as packet:
        raw = {key: torch.from_numpy(packet[key].copy()) for key in fields}
        state = codec.DecodedState.from_reference(packet)
    if through < state.frame_index:
        raise ValueError('Requested frame precedes the decoder reference')
    for frame in range(state.frame_index + 1, through + 1):
        decoded, current = decode_frame(directory, frame, state, raw)
        manifest_path = directory / f'frame_{frame:03d}.json'
        if manifest_path.exists():
            import json
            import hashlib
            manifest = json.loads(manifest_path.read_text())
            digest = hashlib.sha256()
            for name in fields:
                value = current[name].detach().cpu().contiguous()
                digest.update(name.encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.numpy().tobytes())
            if (manifest['frame'] != frame or
                    manifest['decoded_state_sha256'] != decoded.digest().hex() or
                    manifest['raw_state_sha256'] != digest.hexdigest()):
                raise ValueError('Frame manifest state digest mismatch')
        state, raw = decoded, current
    return state, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stream_dir', type=Path)
    parser.add_argument('--reference', type=Path,
                        help='Reference NPZ; defaults to the first reference in stream_dir')
    parser.add_argument('--through', type=int,
                        help='Last frame to decode; defaults to the last motion file')
    parser.add_argument('--save-npz', type=Path,
                        help='Optionally write decoded raw GS attributes')
    args = parser.parse_args()
    directory = args.stream_dir
    references = sorted(directory.glob('reference_frame*.npz'))
    reference = args.reference or (references[0] if references else None)
    if reference is None:
        parser.error('No decoder reference found')
    motions = sorted(directory.glob('motion_*.bin'))
    if args.through is None and not motions:
        parser.error('No motion frames found; pass --through for reference only')
    through = args.through if args.through is not None else int(motions[-1].stem[-3:])
    state, raw = replay(directory, reference, through)
    if args.save_npz:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_npz,
            **{name: value.cpu().numpy() for name, value in raw.items()})
    print(f'frame={state.frame_index} gs={len(state.xyz)} '
          f'decoder_sha256={state.digest().hex()}')


if __name__ == '__main__':
    main()
