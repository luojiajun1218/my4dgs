"""Audit and summarize a completed joint-feedback cross-fold replay."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-15):
        raise AssertionError(f'{actual!r} != {expected!r}')


def audit_candidate(candidate, baseline, fit_names, check_names, reference_bytes, rate_weight):
    fit_losses = candidate['fit_losses']
    if len(fit_losses) != 20:
        raise AssertionError('Every candidate must receive exactly 20 fitting updates')
    counts = {name: sum(row['camera'] == name for row in fit_losses) for name in fit_names}
    if set(counts.values()) != {2} or any(row['camera'] in check_names for row in fit_losses):
        raise AssertionError('Fitting cameras are not balanced or a check camera entered fitting')
    before, after = candidate['before'], candidate['after']
    for state in (before, after):
        if ([row['camera'] for row in state['views']] != fit_names or
                [row['camera'] for row in state['evaluation_views']] != check_names):
            raise AssertionError('Candidate view roles or ordering changed')
    byte_delta = after['cost']['total_bytes'] - baseline['cost']['total_bytes']
    penalty = rate_weight * byte_delta / reference_bytes
    fit_image_gain = baseline['image_loss'] - after['image_loss']
    check_image_gain = baseline['evaluation_image_loss'] - after['evaluation_image_loss']
    close(candidate['byte_delta'], byte_delta)
    close(candidate['rate_penalty'], penalty)
    close(candidate['image_gain'], fit_image_gain)
    close(candidate['gain'], fit_image_gain - penalty)
    close(candidate['evaluation_gain'], check_image_gain - penalty)
    per_view = candidate['per_view_gains']
    if len(per_view) != len(fit_names) + len(check_names):
        raise AssertionError('Per-view gains do not cover all twenty views')
    for row in per_view:
        close(row['rate_gain'], row['image_gain'] - penalty)
        if row['byte_delta'] != byte_delta:
            raise AssertionError('Per-view rate accounting changed candidate bytes')
    return dict(byte_delta=byte_delta, rate_penalty=penalty,
                fit_image_gain=fit_image_gain, fit_rate_gain=fit_image_gain - penalty,
                check_image_gain=check_image_gain, check_rate_gain=check_image_gain - penalty)


def analyze(run):
    run = Path(run).resolve()
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    expected_candidates = summary['protocol']['candidate_count']
    if not summary['nodes'] or len(summary['protocol']['folds']) != 2:
        raise AssertionError('Cross-fold replay must contain nodes and two swapped folds')
    manifest = json.loads((run / 'source_snapshot/manifest.json').read_text(encoding='utf-8'))
    if manifest != summary.get('source_code_snapshot'):
        raise AssertionError('Source manifest and cross-fold summary disagree')
    for entry in manifest:
        if (file_digest(entry['snapshot']) != entry['sha256']
                or Path(entry['snapshot']).stat().st_size != entry['bytes']):
            raise AssertionError(f'Captured source changed: {entry["snapshot"]}')
    candidate_rows, fold_rows, issues = [], [], []
    seen = set()
    for node_entry in summary['nodes']:
        address = node_entry['node']
        frame, step = address['frame'], address['step']
        level, node = address['level'], address['node']
        snapshot = Path(node_entry['source_snapshot'])
        if file_digest(snapshot) != node_entry['source_snapshot_sha256']:
            raise AssertionError(f'Historical snapshot changed: {snapshot}')
        if not node_entry['dependencies']['saved_snapshot_dependencies_verified']:
            raise AssertionError('Saved image/camera fingerprints were not verified')
        saved = {row['camera'] for row in node_entry['dependencies']['saved_views']}
        added = {row['camera'] for row in node_entry['dependencies']['added_views']}
        if saved | added != {f'cam{index:02d}.png' for index in range(1, 21)}:
            raise AssertionError('Recorded dependencies do not cover cam01..cam20')
        for fold_spec in summary['protocol']['folds']:
            fold_name = fold_spec['name']
            path = run / (f'frame_{frame:03d}_step_{step:04d}_level_{level}_'
                          f'node_{node}_{fold_name}.json')
            fold = json.loads(path.read_text(encoding='utf-8'))
            record = fold['record']
            key = (frame, step, level, node, fold_name)
            if key in seen:
                raise AssertionError(f'Duplicate fold: {key}')
            seen.add(key)
            fit_names, check_names = fold['fit_camera_names'], fold['check_camera_names']
            if set(fit_names) & set(check_names) or 'cam00.png' in fit_names + check_names:
                raise AssertionError('Fit/check split is not disjoint or includes cam00')
            plan_names = [fit_names[index] for index in fold['view_plan']['indices']]
            if len(plan_names) != 20 or any(plan_names.count(name) != 2 for name in fit_names):
                raise AssertionError('Saved fitting plan is not the balanced 20-step plan')
            candidates = record['candidates']
            if len(candidates) != expected_candidates or len({row['spec']['name'] for row in candidates}) != expected_candidates:
                raise AssertionError('Fold does not contain 32 unique candidates')
            baseline = record['baseline']
            if ([row['camera'] for row in baseline['views']] != fit_names or
                    [row['camera'] for row in baseline['evaluation_views']] != check_names):
                raise AssertionError('Unfitted source baseline does not cover the configured split')
            by_name = {row['spec']['name']: row for row in candidates}
            if {'stop_original', 'refine_original'} - set(by_name):
                raise AssertionError('Same-budget original-precision controls are missing')
            if max(candidates, key=lambda row: row['gain'])['spec']['name'] != record['selected_by_fitting_rate_score']:
                raise AssertionError('Fit-selected overall winner is inconsistent')
            if max(candidates, key=lambda row: row['evaluation_gain'])['spec']['name'] != record['selected_by_check_rate_score']:
                raise AssertionError('Check-selected analysis winner is inconsistent')
            for candidate in candidates:
                metrics = audit_candidate(candidate, baseline, fit_names, check_names,
                    record['reference_bytes'], record['rate_weight'])
                candidate_rows.append(dict(frame=frame, step=step, level=level, node=node,
                    fold=fold_name, candidate=candidate['spec']['name'],
                    opened=candidate['spec']['opened'],
                    parent_bits=candidate['spec']['parent_bits'],
                    descendant_bits=candidate['spec']['descendant_bits'], **metrics))
            fit_winner = by_name[record['selected_by_fitting_rate_score']]
            fit_stop, fit_refine = by_name[record['best_stop']], by_name[record['best_refine']]
            fit_original = max((by_name['stop_original'], by_name['refine_original']),
                               key=lambda row: row['gain'])
            fit_joint = max((row for row in candidates if row['spec']['parent_bits'] is not None),
                            key=lambda row: row['gain'])
            fold_rows.append(dict(frame=frame, step=step, level=level, node=node,
                fold=fold_name, source_gate_open=record['original']['gate_open'],
                fit_winner=fit_winner['spec']['name'],
                fit_winner_check_rate_gain=fit_winner['evaluation_gain'],
                fit_best_stop=fit_stop['spec']['name'], fit_best_refine=fit_refine['spec']['name'],
                fit_selected_refine_minus_stop_check_rate_gain=
                    fit_refine['evaluation_gain'] - fit_stop['evaluation_gain'],
                fit_original=fit_original['spec']['name'],
                fit_original_check_rate_gain=fit_original['evaluation_gain'],
                fit_joint=fit_joint['spec']['name'],
                fit_joint_check_rate_gain=fit_joint['evaluation_gain'],
                fit_joint_minus_fit_original_check_rate_gain=
                    fit_joint['evaluation_gain'] - fit_original['evaluation_gain'],
                check_selected_analysis_only=record['selected_by_check_rate_score']))

    expected_folds = len(summary['nodes']) * len(summary['protocol']['folds'])
    if len(seen) != expected_folds or len(candidate_rows) != expected_folds * expected_candidates:
        raise AssertionError('Fold or candidate count differs from the recorded protocol')
    by_node = {}
    for row in fold_rows:
        key = (row['frame'], row['step'], row['level'], row['node'])
        by_node.setdefault(key, []).append(row)
    node_rows = []
    for key, rows in by_node.items():
        if len(rows) != 2:
            raise AssertionError(f'Node lacks a swapped fold: {key}')
        preferences = [row['fit_selected_refine_minus_stop_check_rate_gain'] > 0 for row in rows]
        joint = [row['fit_joint_minus_fit_original_check_rate_gain'] > 0 for row in rows]
        source = [row['fit_winner_check_rate_gain'] > 0 for row in rows]
        node_rows.append(dict(frame=key[0], step=key[1], level=key[2], node=key[3],
            heldout_refine_preference_same_sign=preferences[0] == preferences[1],
            both_fit_winners_beat_source=all(source),
            both_fit_joint_beat_fit_original=all(joint),
            fold_fit_winners=[row['fit_winner'] for row in rows],
            fold_refine_minus_stop_check_gains=[
                row['fit_selected_refine_minus_stop_check_rate_gain'] for row in rows],
            fold_joint_minus_original_check_gains=[
                row['fit_joint_minus_fit_original_check_rate_gain'] for row in rows]))

    def write_csv(name, rows):
        with (run / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv('candidates.csv', candidate_rows)
    write_csv('folds.csv', fold_rows)
    analysis = dict(schema='joint_feedback_crossfold_audit_v1',
        completed_candidates=len(candidate_rows), completed_folds=len(fold_rows),
        historical_snapshots_verified=len(summary['nodes']),
        source_code_files_verified=len(manifest), issues=issues,
        nodes=node_rows, folds=fold_rows,
        semantics={
            'fit_winner_check_rate_gain': 'Winner selected on fit cameras, evaluated unchanged on disjoint check cameras versus the unfitted source.',
            'fit_joint_minus_fit_original_check_rate_gain': 'Best explicit joint precision candidate and best original-precision candidate are both selected on fit cameras and compared on check cameras.',
            'check_selected_analysis_only': 'Selected on check cameras and therefore not held-out evidence.',
        })
    (run / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding='utf-8')
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    print(json.dumps(dict(completed_candidates=result['completed_candidates'],
                          completed_folds=result['completed_folds'],
                          historical_snapshots_verified=result['historical_snapshots_verified'],
                          issues=result['issues']), indent=2))


if __name__ == '__main__':
    main()
