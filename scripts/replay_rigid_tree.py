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
    manifest_path = directory / f'frame_{frame:03d}.json'
    if (directory / 'population_window.json').exists() and not manifest_path.is_file():
        raise ValueError('Population-window frame has not been committed by a manifest')
    motion_file = directory / f'motion_{frame:03d}.bin'
    decoded = codec.decode_motion(motion_file.read_bytes(), state)
    rows = decoded.survivor_rows
    if rows is None:
        raise ValueError('Motion decoder lost previous-row survivor mapping')
    old_count = len(rows)
    birth_count = len(decoded.xyz) - old_count
    if any(len(value) != len(state.xyz) for value in raw.values()):
        raise ValueError('Previous raw attributes differ from decoded GS count')
    import json
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Decoder survivor IDs are strictly ordered. If every old point survives,
    # the mapping is identity, so do not gather and clone each full tensor.
    # The decoded pose replaces xyz/rotation. A full-attribute packet creates
    # independent output tensors for these three fields; defer their copies
    # until that decoder rather than cloning each field twice.
    all_survive = old_count == len(state.xyz)
    full_will_copy = bool(manifest.get('full_attribute_update'))
    current = {}
    for name, value in raw.items():
        if name == 'xyz':
            current[name] = decoded.xyz[:old_count]
        elif name == 'rotation':
            current[name] = decoded.rotation[:old_count]
        elif all_survive and full_will_copy and name in ('logscale', 'sh_dc', 'sh_rest'):
            current[name] = value
        else:
            current[name] = value.clone() if all_survive else value.index_select(0, rows)
    timing = manifest.get('opacity_update_timing', 'after_births')
    if timing not in ('before_births', 'after_births'):
        raise ValueError('Unknown opacity update timing')
    if timing == 'before_births':
        current = apply_stream_opacity_update(directory, frame, current, old_count, manifest)
    feature_name = manifest.get('feature_dc_update')
    if feature_name is not None:
        from pipeline.model.sparse_features import decode_feature_dc_delta
        if (feature_name != f'shdc_{frame:03d}.bin' or
                manifest.get('feature_dc_update_timing', 'before_births') != 'before_births'):
            raise ValueError('Shared color packet must precede newborn attribute prediction')
        current['sh_dc'] = decode_feature_dc_delta(
            (directory / feature_name).read_bytes(), current['sh_dc'], frame_index=frame,
            owner=decoded.owner[:old_count], node_count=len(decoded.parents))
    elif (directory / f'shdc_{frame:03d}.bin').exists():
        raise ValueError('Unexpected DC feature packet absent from frame manifest')
    region_scale = manifest.get('regional_logscale_update')
    if region_scale is not None:
        from pipeline.model.regional_logscale import decode_regional_logscale
        if (region_scale != f'regscale_{frame:03d}.bin' or
                manifest.get('regional_logscale_update_timing') != 'before_full_attribute_update'):
            raise ValueError('Invalid regional scale packet timing or filename')
        current['logscale'] = decode_regional_logscale((directory / region_scale).read_bytes(),
            current['logscale'], frame_index=frame, owner=decoded.owner[:old_count],
            node_count=len(decoded.parents))
    elif (directory / f'regscale_{frame:03d}.bin').exists():
        raise ValueError('Unexpected regional scale packet')
    color_version = manifest.get('color_stream_version')
    if color_version is not None and (type(color_version) is not int or color_version not in (2, 3)):
        raise ValueError('Unknown color stream version')
    if color_version == 3 and manifest.get('logscale_update') is not None:
        raise ValueError('Sparse attribute v3 forbids separate logscale packets')
    logscale_name = manifest.get('logscale_update')
    if logscale_name is not None:
        from pipeline.model.sparse_logscale import decode_logscale_delta
        if (logscale_name != f'logscale_{frame:03d}.bin' or
                manifest.get('logscale_update_timing', 'before_births') != 'before_births'):
            raise ValueError('Log-scale packet must precede newborn attribute prediction')
        if decoded.gs_translation_rows is None:
            raise ValueError('Motion decoder lost point-translation support')
        current['logscale'] = decode_logscale_delta(
            (directory / logscale_name).read_bytes(), current['logscale'],
            allowed_rows=decoded.gs_translation_rows, frame_index=frame).logscale
    elif (directory / f'logscale_{frame:03d}.bin').exists():
        raise ValueError('Unexpected log-scale packet absent from frame manifest')
    if color_version in (2, 3):
        budget = manifest.get('color_point_budget')
        if (type(budget) is not int or budget < 1 or
                'color_point_mask' in manifest or 'color_point_mask_count' in manifest or
                manifest.get('full_update') is not True or
                'full_attribute_update' not in manifest or
                manifest.get('full_attribute_update_timing') != 'after_regional_dc_before_births' or
                (color_version == 2 and 'color_attribute_fields' in manifest) or
                (color_version == 3 and manifest.get('color_attribute_fields') != ['sh_dc', 'logscale'])):
            raise ValueError('Invalid sparse color stream manifest')
    point_mask_name = manifest.get('color_point_mask')
    point_mask = None
    expected_mask = f'pointmask_{frame:03d}.bin'
    if point_mask_name is not None:
        from pipeline.model.color_point_mask import decode_color_point_mask
        if point_mask_name != expected_mask:
            raise ValueError('Color point mask filename differs from frame')
        point_mask = torch.from_numpy(decode_color_point_mask((directory / point_mask_name).read_bytes(), frame, old_count))
        if (int(point_mask.sum()) != manifest.get('color_point_mask_count') or
                int(point_mask.sum()) > manifest.get('color_point_budget', -1)):
            raise ValueError('Color point mask metadata differs from packet')
    elif (directory / expected_mask).exists():
        raise ValueError('Unexpected color point mask packet absent from manifest')
    elif manifest.get('color_point_mask_count', 0):
        raise ValueError('Nonempty color point mask is missing')
    elif 'color_point_budget' in manifest and color_version not in (2, 3):
        point_mask = torch.zeros(old_count, dtype=torch.bool)
    full_name = manifest.get('full_attribute_update')
    if full_name is not None:
        from pipeline.model.full_attribute_update import decode_full_attribute_delta
        if (manifest.get('full_update') is not True or
                full_name != f'fullattr_{frame:03d}.bin' or
                manifest.get('full_attribute_update_timing') != 'after_regional_dc_before_births'):
            raise ValueError('Full attributes must follow regional DC and precede newborn prediction')
        reference = {name: current[name] for name in ('logscale', 'sh_dc', 'sh_rest')}
        current.update(decode_full_attribute_delta((directory / full_name).read_bytes(),
                                                   reference, frame_index=frame,
                                                   color_only_budget=budget if color_version in (2, 3) else None,
                                                   allowed_fields=('sh_dc', 'logscale') if color_version == 3 else ('sh_dc',)))
        if point_mask is not None and not torch.equal(current['sh_dc'][~point_mask], reference['sh_dc'][~point_mask]):
            raise ValueError('Decoded point DC changed outside its transmitted mask')
        if 'color_point_budget' in manifest and ((color_version != 3 and not torch.equal(current['logscale'], reference['logscale'])) or
                not torch.equal(current['sh_rest'], reference['sh_rest'])):
            raise ValueError('Color-only residual packet changed non-DC attributes')
    elif (directory / f'fullattr_{frame:03d}.bin').exists():
        raise ValueError('Unexpected full attribute packet absent from frame manifest')
    regional_name = manifest.get('regional_full_attributes')
    if regional_name is not None:
        from pipeline.model.regional_full_attributes import decode_regional_full
        if (regional_name != f'regfull_{frame:03d}.bin' or
                manifest.get('regional_full_timing') != 'after_point_attributes_before_births'):
            raise ValueError('Invalid regional full attribute packet timing')
        current.update(decode_regional_full((directory / regional_name).read_bytes(), current,
            decoded.owner[:old_count], len(decoded.parents), frame))
    elif (directory / f'regfull_{frame:03d}.bin').exists():
        raise ValueError('Unexpected regional full attribute packet')
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
