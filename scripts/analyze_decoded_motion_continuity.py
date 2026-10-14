"""CPU posterior continuity of real decoded GS motion; never a GT trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def summary(previous, current):
    count = len(current)
    if not count:
        return dict(gs=0)
    npast, ncurrent = np.linalg.norm(previous, axis=1), np.linalg.norm(current, axis=1)
    error = np.linalg.norm(current - previous, axis=1)
    valid = (npast > 0) & (ncurrent > 0)
    cos = np.einsum('ij,ij->i', previous[valid], current[valid]) / (npast[valid] * ncurrent[valid])
    xp, xc = previous - previous.mean(0), current - current.mean(0)
    denominator = np.linalg.norm(xp) * np.linalg.norm(xc)
    rms, baseline = float(np.sqrt(np.square(error).mean())), float(np.sqrt(np.square(ncurrent).mean()))
    return dict(gs=count,
        centered_vector_correlation=float(np.einsum('ij,ij->', xp, xc) / denominator) if denominator > 1e-20 else None,
        cosine_supported_gs=len(cos), mean_cosine=float(cos.mean()) if len(cos) else None,
        median_cosine=float(np.median(cos)) if len(cos) else None,
        reverse_fraction=float((cos < 0).mean()) if len(cos) else None,
        previous_displacement_p50=float(np.quantile(npast, .5)),
        previous_displacement_p90=float(np.quantile(npast, .9)),
        current_displacement_p50=float(np.quantile(ncurrent, .5)),
        current_displacement_p90=float(np.quantile(ncurrent, .9)),
        constant_velocity_error_p50=float(np.quantile(error, .5)),
        constant_velocity_error_p90=float(np.quantile(error, .9)),
        constant_velocity_error_p99=float(np.quantile(error, .99)),
        constant_velocity_error_rms=rms, zero_motion_baseline_error_rms=baseline,
        constant_velocity_to_zero_error_rms_ratio=rms / baseline if baseline > 1e-20 else None)


def analyze(path):
    before = path.stat()
    x = torch.load(path, map_location='cpu', weights_only=False)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Checkpoint changed during read; wait for a completed save.')
    motion = x['rigid_tree_motion']
    records = motion['online_history']['records']
    cfg, extent = motion['config'], motion['extent']
    step = (cfg['gs_xyz_step_extent'] or cfg['motion_xyz_step_extent']) * extent
    moving_units = cfg['online_clustering']['static_translation_extent'] / (cfg['gs_xyz_step_extent'] or cfg['motion_xyz_step_extent'])
    state = motion['persistent_state']
    ids = motion['materialized_online_ids'].cpu().numpy()
    owners = motion['tensors']['materialized_owner'].cpu().numpy()
    region_ids = state['region_ids'].cpu().numpy()
    sort = np.argsort(ids)
    ids, owners = ids[sort], region_ids[owners[sort]]
    totals_p, totals_c, total_owners = [], [], []
    pairs = []
    for prior, current in zip(records, records[1:]):
        fp, fc = int(prior['frame_index']), int(current['frame_index'])
        if fc != fp + 1:
            pairs.append(dict(previous_frame=fp, current_frame=fc, skipped='not_adjacent_actual_frames'))
            continue
        common, ip, ic = np.intersect1d(prior['ids'], current['ids'], return_indices=True)
        seen_p, seen_c = prior['observed'][ip], current['observed'][ic]
        both = seen_p & seen_c
        p = (prior['target'][ip] - prior['source'][ip]).astype(np.float64)
        c = (current['target'][ic] - current['source'][ic]).astype(np.float64)
        moving_p = np.linalg.norm(np.rint(p / step), axis=1) > moving_units
        moving_c = np.linalg.norm(np.rint(c / step), axis=1) > moving_units
        loc = np.searchsorted(ids, common)
        if np.any(loc == len(ids)) or not np.array_equal(ids[loc], common):
            raise ValueError('Matched historical GS ID absent from checkpoint cohort; deletion needs an explicit mapping.')
        cohort = owners[loc]
        chain_error = np.linalg.norm(current['source'][ic[both]].astype(np.float64) - prior['target'][ip[both]], axis=1)
        pair = dict(previous_frame=fp, current_frame=fc, id_intersection=len(common),
            current_ids_without_previous_transition=len(current['ids']) - len(common),
            both_observed_gs=int(both.sum()), previous_observed_current_missing_gs=int((seen_p & ~seen_c).sum()),
            previous_moving_current_missing_gs=int((seen_p & ~seen_c & moving_p).sum()),
            current_observed_previous_missing_gs=int((~seen_p & seen_c).sum()),
            starts_gs=int((both & ~moving_p & moving_c).sum()),
            stops_gs=int((both & moving_p & ~moving_c).sum()),
            source_equals_prior_decoded_target_max_error=float(chain_error.max()) if len(chain_error) else None,
            all_both_observed=summary(p[both], c[both]),
            both_moving=summary(p[both & moving_p & moving_c], c[both & moving_p & moving_c]),
            either_moving=summary(p[both & (moving_p | moving_c)], c[both & (moving_p | moving_c)]),
            current_moving=summary(p[both & moving_c], c[both & moving_c]), regions=[])
        for identifier in np.unique(cohort[both & (moving_p | moving_c)]):
            members = both & (cohort == identifier)
            pair['regions'].append(dict(region_id=int(identifier), both_observed_gs=int(members.sum()),
                both_moving=summary(p[members & moving_p & moving_c], c[members & moving_p & moving_c]),
                either_moving=summary(p[members & (moving_p | moving_c)], c[members & (moving_p | moving_c)])))
        pairs.append(pair)
        totals_p.append(p[both]); totals_c.append(c[both]); total_owners.append(cohort[both])
    report = dict(checkpoint=str(path.resolve()), checkpoint_frame=motion['frame_index'],
        scope='posterior decoded-estimate motion continuity; not GT, training effect, or an initialization experiment',
        displacement='actual complete WORLD target-source, stable GS IDs, both observed for each pair',
        prediction='previous displacement added to next actual decoded source; no future input to predictor',
        region_scope='checkpoint final logical-owner cohorts projected by stable GS ID; not historical region memberships',
        observation_scope='saved renderer visibility/radii proxy; not proof of unoccluded pixel contribution',
        history_frames=[int(row['frame_index']) for row in records],
        gs_translation_step_world=step, moving_threshold_GS_units=moving_units,
        moving_rule='L2(round(WORLD displacement / GS step)) greater than existing static threshold in GS units',
        reverse_rule='previous dot current < 0, only nonzero vectors; no invented direction for zero motion',
        correlated_samples_scope='GS-transition pairs; the same GS can contribute in several adjacent pairs', pairs=pairs)
    if totals_p:
        p, c = np.concatenate(totals_p), np.concatenate(totals_c)
        cohort = np.concatenate(total_owners)
        moving_p = np.linalg.norm(np.rint(p / step), axis=1) > moving_units
        moving_c = np.linalg.norm(np.rint(c / step), axis=1) > moving_units
        report['pooled'] = dict(all_both_observed=summary(p, c),
            both_moving=summary(p[moving_p & moving_c], c[moving_p & moving_c]),
            either_moving=summary(p[moving_p | moving_c], c[moving_p | moving_c]),
            current_moving=summary(p[moving_c], c[moving_c]),
            starts_gs_pairs=int((~moving_p & moving_c).sum()), stops_gs_pairs=int((moving_p & ~moving_c).sum()))
        report['pooled_regions'] = [dict(region_id=int(identifier),
            both_moving=summary(p[(cohort == identifier) & moving_p & moving_c], c[(cohort == identifier) & moving_p & moving_c]),
            either_moving=summary(p[(cohort == identifier) & (moving_p | moving_c)], c[(cohort == identifier) & (moving_p | moving_c)]))
            for identifier in np.unique(cohort[moving_p | moving_c])]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint.resolve().parent in args.output.resolve().parents:
        raise ValueError('Use a fresh independent output outside the existing run.')
    torch.set_num_threads(2)
    report = analyze(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(history_frames=report['history_frames'], pooled=report.get('pooled'),
        pairs=[{key: row[key] for key in ('previous_frame', 'current_frame', 'both_moving',
            'starts_gs', 'stops_gs', 'previous_moving_current_missing_gs')} for row in report['pairs']
            if 'both_moving' in row]), indent=2))
    print(f'Analysis saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
