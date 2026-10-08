"""Independently audit a completed joint-feedback clone transaction."""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def close(left, right):
    if not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-15):
        raise AssertionError(f'{left!r} != {right!r}')


def independently_select(fold_records):
    folds = [{row['spec']['name']: row for row in record['candidates']}
             for record in fold_records]
    if len(folds) != 2 or set(folds[0]) != set(folds[1]):
        raise AssertionError('Cross-fold candidate spaces disagree')
    eligible = []
    for name in folds[0]:
        gains = [fold[name]['evaluation_gain'] for fold in folds]
        if all(value > 0 for value in gains):
            eligible.append((sum(gains) / 2, name, gains))
    return (max(eligible) if eligible else (0.0, 'source', [0.0, 0.0]))


def analyze(run):
    run = Path(run).resolve()
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('schema') != 'joint_feedback_transaction_v1':
        raise AssertionError('Unexpected transaction schema')
    issues, rows = [], []
    manifest = json.loads((run / 'source_snapshot/manifest.json').read_text(encoding='utf-8'))
    if manifest != summary['source_code_snapshot']:
        raise AssertionError('Source manifest and run summary disagree')
    for entry in manifest:
        snapshot = Path(entry['snapshot'])
        if digest(snapshot) != entry['sha256'] or snapshot.stat().st_size != entry['bytes']:
            raise AssertionError(f'Captured source changed: {snapshot}')

    diagnostic = importlib.import_module('pipeline.ReCon-GS.model.gate_diagnostics')
    storage = importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary')
    for result in summary['nodes']:
        address = result['node']
        frame, step = address['frame'], address['step']
        snapshot = Path(result['source_snapshot'])
        if digest(snapshot) != result['source_snapshot_sha256']:
            raise AssertionError(f'Historical snapshot changed: {snapshot}')
        fold_records = []
        for path, expected in result['source_fold_files'].items():
            if digest(path) != expected:
                raise AssertionError(f'Source fold changed: {path}')
            fold_records.append(json.loads(Path(path).read_text(encoding='utf-8'))['record'])
        mean_gain, selected, directional = independently_select(fold_records)
        if (selected != result['selected'] or directional != result['selection']['selected']['directional_gains']):
            raise AssertionError('Recorded same-candidate cross-fold selection is inconsistent')
        source = result['source_evaluation']
        if len(source['views']) != 20 or [row['camera'] for row in source['views']] != [
                f'cam{index:02d}.png' for index in range(1, 21)]:
            raise AssertionError('Source realization does not cover cam01..cam20')
        if selected == 'source':
            if result['status'] != 'source_retained' or result['accepted']:
                raise AssertionError('Source fallback unexpectedly changed the clone')
            rows.append(dict(frame=frame, step=step, selected=selected,
                             status=result['status'], selection_mean_gain=mean_gain,
                             realized_gain=0.0, byte_delta=0, logit_steps=0))
            continue

        if (len(result['fit_losses']) != 20 or
                sorted(row['camera'] for row in result['fit_losses']) !=
                [f'cam{index:02d}.png' for index in range(1, 21)]):
            raise AssertionError('Deployment fit must use every training camera exactly once')
        candidate, realized = result['candidate_after'], result['realized_evaluation']
        predicted_delta = candidate['cost']['total_bytes'] - source['cost']['total_bytes']
        predicted_gain = (source['image_loss'] - candidate['image_loss'] -
            result['rate_weight'] * predicted_delta / result['reference_bytes'])
        realized_delta = realized['cost']['total_bytes'] - source['cost']['total_bytes']
        realized_gain = (source['image_loss'] - realized['image_loss'] -
            result['rate_weight'] * realized_delta / result['reference_bytes'])
        close(predicted_gain, result['predicted_all20_gain'])
        close(realized_gain, result['realized_gain'])
        close(predicted_gain, realized_gain)
        if predicted_delta != result['predicted_byte_delta'] or realized_delta != result['realized_byte_delta']:
            raise AssertionError('Actual byte deltas are inconsistent')
        close(candidate['image_loss'], realized['image_loss'])
        if candidate['cost']['total_bytes'] != realized['cost']['total_bytes']:
            raise AssertionError('Candidate was not exactly realized in the deployment clone')
        feedback = result['feedback']
        if feedback['steps'] > feedback['max_steps']:
            raise AssertionError('Feedback optimizer exceeded its configured budget')
        if result['status'] == 'committed_to_clone':
            if not result['accepted'] or not feedback['final']['matches']:
                raise AssertionError('Committed transaction is not marked accepted')
            deployed_path = run / f'deployed_frame_{frame:03d}_step_{step:04d}.pth'
            packet = torch.load(deployed_path, map_location='cpu', weights_only=False)
            motion = storage.StorageAwareBinaryMotion(storage.StorageAwareBinaryConfig(**packet['config']))
            motion.restore(packet)
            deployed_cost = diagnostic.exact_cost(motion, step)
            if deployed_cost['total_bytes'] != realized['cost']['total_bytes']:
                raise AssertionError('Saved deployed motion has a different production bitstream cost')
            if not result['gate_optimizer_resets'] or any(
                    not row['scalar_step_preserved'] for row in result['gate_optimizer_resets']):
                raise AssertionError('Accepted gate optimizer reset semantics are incomplete')
            if any(not row['scalar_step_preserved'] for row in result['motion_optimizer_resets']):
                raise AssertionError('Accepted motion optimizer reset semantics are incomplete')
        elif result['status'] == 'rolled_back':
            rejection_reason = (not feedback['final']['matches']
                                or not math.isfinite(realized_gain) or realized_gain <= 0)
            if result['accepted'] or not rejection_reason or not result['rollback_verified']:
                raise AssertionError('Rejected selected transaction did not roll back exactly')
            if (run / f'deployed_frame_{frame:03d}_step_{step:04d}.pth').exists():
                raise AssertionError('Rolled-back transaction unexpectedly saved a deployed packet')
        else:
            raise AssertionError('Selected transaction has an invalid terminal status')
        rows.append(dict(frame=frame, step=step, selected=selected,
            status=result['status'], selection_mean_gain=mean_gain,
            realized_gain=realized_gain, byte_delta=realized_delta,
            logit_steps=feedback['steps'], gate_before=feedback['initial']['gate'],
            gate_after=feedback['final']['gate'],
            final_gate_margin=feedback['final']['gate_target_margin'],
            final_precision_margin=feedback['final']['minimum_precision_target_margin']))

    counts = dict(committed=sum(row['status'] == 'committed_to_clone' for row in rows),
                  source_retained=sum(row['status'] == 'source_retained' for row in rows),
                  rolled_back=sum(row['status'] == 'rolled_back' for row in rows))
    if counts != {name: summary[name] for name in counts}:
        raise AssertionError('Transaction status counts disagree')
    analysis = dict(schema='joint_feedback_transaction_audit_v1', issues=issues,
        source_snapshots_verified=len(rows), source_code_files_verified=len(manifest),
        status=counts, nodes=rows,
        conclusion='Every selected candidate was checked against its realized clone; '
                   'source fallbacks and rollbacks retained their source state.')
    (run / 'analysis.json').write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding='utf-8')
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))


if __name__ == '__main__':
    main()
