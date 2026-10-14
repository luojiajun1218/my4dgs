"""Read completed persistent streams; attribute exact raw sparse entry bytes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_online_flat_regions import CODEC, read_symbol_state


def sparse_entry_bytes(owner, translation, rotation, region_count):
    """Attribute delta-gap+flag+int16 bytes using the full GS row stream.

    The shared sparse-count varint header is reported separately. Compression
    of the complete packet is not separable into per-region byte contributions.
    """
    owner = torch.as_tensor(owner).cpu().long()
    translation, rotation = translation.cpu(), rotation.cpu()
    t_active, r_active = translation.ne(0).any(-1), rotation.ne(0).any(-1)
    rows = torch.where(t_active | r_active)[0]
    cost = [0] * region_count
    last = -1
    for row in rows.tolist():
        gap = row - last
        last = row
        row_bytes = len(CODEC._uvar(gap)) + 1 + 6 * (int(t_active[row]) + int(r_active[row]))
        cost[int(owner[row])] += row_bytes
    header = len(CODEC._uvar(len(rows)))
    actual = bytearray()
    CODEC._sparse_write(actual, translation, rotation)
    assert sum(cost) + header == len(actual)
    return dict(raw_entry_bytes_by_slot=cost, raw_count_header_bytes=header,
                raw_sparse_section_bytes=len(actual),
                root_entry_fraction=cost[0] / sum(cost) if sum(cost) else None,
                compression_attribution='not separable; shares refer to raw sparse entries')


def analyze(run):
    stream = run / 'frame_storage' / 'rigid_tree'
    with np.load(stream / 'reference_frame0.npz', allow_pickle=False) as reference:
        state = CODEC.DecodedState(
            torch.from_numpy(reference['xyz'].copy()), torch.from_numpy(reference['rotation'].copy()),
            torch.from_numpy(reference['owner'].copy()).long(),
            torch.from_numpy(reference['parents'].copy()).long(), int(reference['frame_index']))
    frames = []
    for path in sorted((run / 'frame_summary').glob('frame_*.json')):
        summary = json.loads(path.read_text())
        frame = summary['frame']
        motion_path = stream / f'motion_{frame:03d}.bin'
        state, rt, rr, gt, gr = read_symbol_state(motion_path.read_bytes(), state)
        manifest_path = stream / f'frame_{frame:03d}.json'
        manifest = json.loads(manifest_path.read_text())
        assert state.digest().hex() == manifest['decoded_state_sha256']
        assert len(gt) == summary['old_count']
        assert len(state.owner) == summary['total_count']
        diagnostic = summary['flat_region_diagnostics']
        regional = diagnostic['regions']
        qcount = diagnostic['refinement']['after']['quantized_motion']
        active = gt.ne(0).any(-1) | gr.ne(0).any(-1)
        assert int(active.sum()) == qcount['gs_active_rows']
        owner = state.owner[:len(gt)]
        root_active = int((active & owner.eq(0)).sum())
        raw = sparse_entry_bytes(owner, gt, gr, len(state.parents))
        assert raw['raw_sparse_section_bytes'] == diagnostic['refinement']['after']['packet_score']['raw_residual_motion_bytes']
        motion_bytes = motion_path.stat().st_size
        birth_bytes = (stream / f'birth_{frame:03d}.bin').stat().st_size if manifest.get('birth') else 0
        opacity_name = manifest.get('opacity_update')
        if opacity_name not in (None, f'opacity_{frame:03d}.bin'):
            raise ValueError('Unexpected opacity update filename')
        opacity_bytes = (stream / opacity_name).stat().st_size if opacity_name else 0
        assert motion_bytes == summary['motion']['bytes']
        assert opacity_bytes == (summary['opacity_update']['archive_bytes'] if summary.get('opacity_update') else 0)
        assert motion_bytes + birth_bytes + opacity_bytes == summary['encoded_bytes']
        frames.append(dict(
            frame=frame, psnr=summary['evaluation']['test']['psnr'],
            motion_bytes=motion_bytes, birth_bytes=birth_bytes, opacity_update_bytes=opacity_bytes,
            payload_bytes=motion_bytes + birth_bytes + opacity_bytes,
            manifest_index_bytes=manifest_path.stat().st_size,
            complete_update_bytes=motion_bytes + birth_bytes + opacity_bytes + manifest_path.stat().st_size,
            total_gs=len(state.owner), survivors=len(gt), newborns=len(state.owner) - len(gt),
            region_ids=regional.get('region_ids'), parent_slots=state.parents.tolist(),
            populated_regions=regional['populated_region_count'], allocated_slots=len(state.parents),
            root_gs=int(state.owner.eq(0).sum()), residual_nonzero_gs=int(active.sum()),
            root_residual_nonzero_gs=root_active,
            root_residual_nonzero_fraction=root_active / int(active.sum()) if active.any() else None,
            persistent=diagnostic.get('persistent'), raw_residual=raw,
            before_quantized=diagnostic['refinement']['before']['quantized_motion'],
            after_quantized=qcount,
            before_packet_bytes=diagnostic['refinement']['before']['packet_score']['bytes'],
            after_packet_bytes=diagnostic['refinement']['after']['packet_score']['bytes']))
    count = len(frames)
    if not count:
        raise ValueError('No completed dynamic frames')
    return dict(run=str(run), dynamic_frame_count=count, frames=frames,
        means={key: sum(row[key] for row in frames) / count for key in
               ('psnr', 'motion_bytes', 'birth_bytes', 'opacity_update_bytes', 'payload_bytes', 'manifest_index_bytes',
                'complete_update_bytes')},
        pooled_root_residual_nonzero_fraction=sum(row['root_residual_nonzero_gs'] for row in frames) /
            max(1, sum(row['residual_nonzero_gs'] for row in frames)),
        pooled_root_raw_residual_entry_fraction=sum(row['raw_residual']['raw_entry_bytes_by_slot'][0]
            for row in frames) / max(1, sum(sum(row['raw_residual']['raw_entry_bytes_by_slot']) for row in frames)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.run.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
