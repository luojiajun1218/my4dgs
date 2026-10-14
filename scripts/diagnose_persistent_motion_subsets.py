"""Read-only integer residual subgroups and same-pose leaf retirement costs."""

import argparse
from dataclasses import is_dataclass
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.analyze_online_flat_regions import CODEC, read_symbol_state
from scripts.analyze_persistent_motion_run import sparse_entry_bytes


def diagnose(run, checkpoint_frame=29, cap_frame=37):
    torch.set_num_threads(2)
    packet = torch.load(run / f'rigid-ckpt-{checkpoint_frame}-300.pth',
                        map_location='cpu', weights_only=False)
    capture = packet['rigid_tree_motion']
    rigid = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
    defaults = rigid.RigidTreeConfig()
    config = rigid.RigidTreeConfig(**{key: (type(getattr(defaults, key))(**value)
        if is_dataclass(getattr(defaults, key)) else value)
        for key, value in capture['config'].items()})
    model = rigid.RigidResidualTree(config)
    model.restore(capture)
    history = model.online_history.records
    wanted = {int(record['frame_index']) for record in history}
    stream = run / 'frame_storage' / 'rigid_tree'
    with np.load(stream / 'reference_frame0.npz', allow_pickle=False) as ref:
        state = CODEC.DecodedState(torch.from_numpy(ref['xyz'].copy()),
            torch.from_numpy(ref['rotation'].copy()), torch.from_numpy(ref['owner'].copy()).long(),
            torch.from_numpy(ref['parents'].copy()).long(), int(ref['frame_index']))
    symbols, cap = {}, {}
    original_compose = CODEC.compose_pose

    def remember_compose(*args):
        cap['inputs'] = tuple(value.detach().clone() for value in args)
        return original_compose(*args)

    for frame in range(1, max(checkpoint_frame, cap_frame) + 1):
        prior = state
        if frame == cap_frame:
            with patch.object(CODEC, 'compose_pose', side_effect=remember_compose):
                state, rt, rr, gt, gr = read_symbol_state((stream / f'motion_{frame:03d}.bin').read_bytes(), prior)
            cap.update(prior=prior, state=state)
        else:
            state, rt, rr, gt, gr = read_symbol_state((stream / f'motion_{frame:03d}.bin').read_bytes(), prior)
        manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
        assert state.digest().hex() == manifest['decoded_state_sha256']
        if frame in wanted:
            record = next(record for record in history if int(record['frame_index']) == frame)
            assert len(record['ids']) == len(gt)
            symbols[frame] = torch.cat((gt, gr), dim=-1).long().abs().sum(-1).numpy() / 6.

    current_ids = model.online_ids.numpy()
    owners = model._persistent_owner_ids().numpy()
    units, observed = [], []
    for record in history:
        ids = record['ids']; order = np.argsort(ids)
        where = np.searchsorted(ids[order], current_ids)
        present = where < len(ids)
        present[present] &= ids[order[where[present]]] == current_ids[present]
        value = np.zeros(len(current_ids)); mask = np.zeros(len(current_ids), bool)
        rows = order[where[present]]
        value[present] = symbols[int(record['frame_index'])][rows]
        mask[present] = record['observed'][rows]
        units.append(value); observed.append(mask)
    units, observed = np.stack(units, axis=1), np.stack(observed, axis=1)
    high = units > config.persistent_regions.split_p90_units
    streak = (observed[:, -1] & high[:, -1]).astype(np.int64)
    stopped = streak == 0
    for index in range(len(history) - 2, -1, -1):
        valid = observed[:, index] & ~stopped
        streak += valid & high[:, index]
        stopped |= valid & ~high[:, index]
    latest = units[:, -1]
    gt = torch.round(model.gs_translation / model.gs_steps[:3]).short()
    gr = torch.round(model.gs_rotation / model.gs_steps[3:]).short()
    raw = sparse_entry_bytes(model.owner, gt, gr, len(model.parents))
    entry_cost = np.zeros(len(gt), np.int64)
    translation_active, rotation_active = gt.ne(0).any(-1), gr.ne(0).any(-1)
    previous_row = -1
    for row in torch.where(translation_active | rotation_active)[0].tolist():
        entry_cost[row] = len(CODEC._uvar(row - previous_row)) + 1 + 6 * (
            int(translation_active[row]) + int(rotation_active[row]))
        previous_row = row
    assert int(entry_cost.sum()) + raw['raw_count_header_bytes'] == raw['raw_sparse_section_bytes']
    latest_summary = json.loads((run / 'frame_summary' / f'frame_{checkpoint_frame:03d}.json').read_text())
    regions = []
    for slot, logical in enumerate(model.persistent_state.region_ids.tolist()):
        members = owners == logical; active = members & (latest > 0)
        final_high = members & observed[:, -1] & high[:, -1]
        total_l1 = float(latest[members].sum() * 6)
        regions.append(dict(region_id=logical, gs=int(members.sum()), observed=int((members & observed[:, -1]).sum()),
            nonzero=int(active.sum()), high_observed=int(final_high.sum()),
            high_fraction_all=float(final_high.sum() / max(1, members.sum())),
            high_valid_streak2=int((members & (streak >= 2)).sum()),
            high_valid_streak3=int((members & (streak >= 3)).sum()),
            integer_l1=int(total_l1), high_integer_l1=int(latest[final_high].sum() * 6),
            high_l1_fraction=float(latest[final_high].sum() * 6 / total_l1) if total_l1 else None,
            raw_sparse_entry_bytes=raw['raw_entry_bytes_by_slot'][slot],
            high_raw_sparse_entry_bytes=int(entry_cost[final_high].sum()),
            high_raw_entry_fraction=float(entry_cost[final_high].sum() / max(1, entry_cost[members].sum())),
            p90_all_observed_units=float(np.quantile(latest[members & observed[:, -1]], .9)),
            p90_active_observed_units=float(np.quantile(latest[active & observed[:, -1]], .9)) if (active & observed[:, -1]).any() else None,
            evidence=latest_summary['flat_region_diagnostics']['persistent']['evidence'].get(str(logical))))

    # Reconstruct actual cap-frame decoded survivor motion, not an estimated fit.
    source, source_q, owner, parents, pivots, rt_float, rr_float, gt_float, gr_float = cap['inputs']
    assert len(source) == len(cap['prior'].xyz) and torch.equal(source, cap['prior'].xyz)
    model.base_xyz, model.base_rotation = source, source_q
    model.owner, model.parents, model.pivots = owner, parents, pivots
    model.previous_rows = torch.arange(len(source))
    model.codec_previous_state = model.frame_reference_state = cap['prior']
    model.frame_index = cap_frame
    model.region_translation, model.region_rotation = nn.Parameter(rt_float), nn.Parameter(rr_float)
    model.gs_translation, model.gs_rotation = nn.Parameter(gt_float), nn.Parameter(gr_float)
    summary = json.loads((run / 'frame_summary' / f'frame_{cap_frame:03d}.json').read_text())
    model.persistent_state.region_ids = torch.tensor(summary['flat_region_diagnostics']['persistent']['region_ids'])
    model.persistent_state.reference_region_ids = model.persistent_state.region_ids.clone()
    model.region_map = torch.arange(len(parents))
    model._metadata_cache = None
    target_xyz, target_q = (value.detach().clone() for value in model.forward(200))
    torch.testing.assert_close(target_xyz, cap['state'].xyz[:len(source)], rtol=0, atol=0)
    torch.testing.assert_close(target_q, cap['state'].rotation[:len(source)], rtol=0, atol=0)
    baseline = model._online_packet_score(target_xyz, target_q)
    assert baseline['valid']
    original = model._persistent_layout_snapshot()
    retirements = []
    rt, rr, gt, gr = model._values(hard=True)
    active = gt.ne(0).any(-1) | gr.ne(0).any(-1)
    candidates = sorted((int((active & owner.eq(slot)).sum()), slot)
        for slot in range(1, len(parents)) if not bool((parents[1:] == slot).any()))[:8]
    for _, slot in candidates:
        model._restore_persistent_layout(original)
        logical = int(model.persistent_state.region_ids[slot]); rows = torch.where(model.owner == slot)[0]
        logical_owner = model._persistent_owner_ids(); logical_owner[rows] = 0
        layout = model._compact_persistent_layout(logical_owner)
        kept = [int(torch.where(model.persistent_state.region_ids == identifier)[0][0])
                for identifier in layout['region_ids']]
        new_gt, new_gr = model.gs_translation.detach().clone(), model.gs_rotation.detach().clone()
        residual = model._retarget_rows(rows, source.new_tensor([1., 0., 0., 0.]), source.new_zeros(3), target_xyz, target_q)
        if residual is None:
            retirements.append(dict(region_id=logical, overflow=True)); continue
        new_gt[rows], new_gr[rows] = residual
        model.region_translation = nn.Parameter(model.region_translation.detach()[kept].clone())
        model.region_rotation = nn.Parameter(model.region_rotation.detach()[kept].clone())
        model.gs_translation, model.gs_rotation = nn.Parameter(new_gt), nn.Parameter(new_gr)
        model.pivots = model.pivots[kept].clone()
        model.owner, model.parents, model.region_map = layout['owner'], layout['parents'], layout['region_map']
        model.persistent_state.region_ids = layout['region_ids']
        model._metadata_cache = None
        after = model._online_packet_score(target_xyz, target_q)
        _, _, agt, agr = model._values(hard=True)
        retirements.append(dict(region_id=logical, gs=len(rows),
            before_active=int(active[rows].sum()), after_active=int((agt[rows].ne(0).any(-1) | agr[rows].ne(0).any(-1)).sum()),
            before_integer_l1=int(gt[rows].long().abs().sum() + gr[rows].long().abs().sum()),
            after_integer_l1=int(agt[rows].long().abs().sum() + agr[rows].long().abs().sum()),
            packet_delta=after['bytes'] - baseline['bytes'], raw_residual_delta=after['raw_residual_motion_bytes'] - baseline['raw_residual_motion_bytes'],
            before_packet=baseline, after_packet=after,
            pose_guard_valid=after['valid']))
    return dict(run=str(run), checkpoint_frame=checkpoint_frame, history_frames=[int(r['frame_index']) for r in history],
        high_definition='observed GS mean absolute six integer residual channels >4; missing holds, observed low resets',
        hidden_subsets=sorted(regions, key=lambda r: r['integer_l1'], reverse=True),
        cap_frame=cap_frame, retirement_trials=retirements,
        retirement_scope='CPU read-only hypothetical retirement of one leaf; other shared motions unchanged; no optimization/births, actual complete compressed motion packet decode guard; not applied to training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = diagnose(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=str(args.output), hidden_subsets=result['hidden_subsets'][:5],
        retirement_trials=result['retirement_trials']), allow_nan=False))
