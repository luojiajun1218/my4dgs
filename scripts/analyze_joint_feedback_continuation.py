"""Audit and summarize a completed controlled feedback continuation run."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def _choice(state):
    return dict(gate_open=state['gate_open'], precision=[
        dict(level=row['level'], row=row['row'], bits=row['bits'])
        for row in state['precision']])


def _transitions(node, arm):
    sequence = [dict(step='pre_update', state=node['initial_target_state'][arm])]
    sequence.extend(dict(step=row['step'], state=row['target_state'])
                    for row in node['updates'][arm])
    result, previous = [], None
    for row in sequence:
        choice = _choice(row['state'])
        if choice != previous:
            result.append(dict(step=row['step'], **choice,
                gate_logit=row['state']['gate_logit']))
            previous = choice
    return result


def noop_consistency_issues(node, frame):
    if node['feedback_mode'] == 'deployed_feedback':
        return []
    issues = []
    if (node['deployed_packet'] is not None or node['deployed_packet_sha256'] is not None
            or not node['no_op_final_state_equal']):
        issues.append(f'frame {frame}: source fallback/rollback is not an exact no-op pair')
    names = ('feedback_minus_source_image_loss', 'feedback_minus_source_psnr',
             'feedback_minus_source_bytes', 'feedback_rd_gain')
    if any(any(evaluation['comparison'][name] != 0 for name in names)
           for evaluation in node['evaluations']):
        issues.append(f'frame {frame}: no-op evaluation differs between arms')
    if any(node['final_validation']['comparison'][name] != 0 for name in (
            'feedback_minus_source_image_loss', 'feedback_minus_source_psnr')):
        issues.append(f'frame {frame}: no-op cam00 evaluation differs between arms')
    return issues


def analyze(run):
    run = Path(run).resolve()
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    config = summary['protocol']
    issues = []
    if summary.get('schema') != 'joint_feedback_controlled_continuation_v1':
        issues.append('unexpected summary schema')
    expected_nodes = config.get('expected_nodes')
    if expected_nodes is None or len(summary.get('nodes', [])) != expected_nodes:
        issues.append('completed node count differs from the configured protocol')

    manifest = json.loads((run / 'source_snapshot/manifest.json').read_text(encoding='utf-8'))
    for row in manifest:
        if digest(row['snapshot']) != row['sha256']:
            issues.append(f'source snapshot copy hash mismatch: {row["snapshot"]}')
        if digest(row['source']) != row['sha256']:
            issues.append(f'live source changed after launch: {row["source"]}')

    node_rows = []
    conceptual_training_update_records = 0
    conceptual_all20_evaluation_sets = 0
    actual_training_updates = 0
    actual_all20_evaluation_sets = 0
    actual_final_validation_renders = 0
    for node in summary['nodes']:
        frame, start = node['node']['frame'], node['node']['step']
        expected_steps = list(range(start, config['end_step'] + 1))
        schedule = node['schedule']
        if [row['step'] for row in schedule] != expected_steps:
            issues.append(f'frame {frame}: schedule does not cover every continuation step')
        if schedule[0]['camera'] != node['snapshot_training_camera']:
            issues.append(f'frame {frame}: current step did not reuse snapshot camera')
        if config['validation_camera_name'] in [row['camera'] for row in schedule]:
            issues.append(f'frame {frame}: cam00 leaked into training schedule')
        expected_visits = (config['end_step'] - start) // len(config['ordinary_camera_names'])
        if set(node['future_camera_counts'].values()) != {expected_visits}:
            issues.append(f'frame {frame}: future camera sequence is not balanced')
        if node['objective']['gs_activity_loss_weight'] != 0:
            issues.append(f'frame {frame}: unimplemented GS activity term was nonzero')
        if not node['immutable_sources_verified']:
            issues.append(f'frame {frame}: immutable source verification failed')
        if digest(node['source_snapshot']) != node['source_snapshot_sha256']:
            issues.append(f'frame {frame}: source snapshot digest changed')
        if node['feedback_mode'] == 'deployed_feedback':
            if digest(node['deployed_packet']) != node['deployed_packet_sha256']:
                issues.append(f'frame {frame}: feedback packet digest changed')

        source_packet = torch.load(node['source_snapshot'], map_location='cpu', weights_only=False)
        motion_config = source_packet['motion']['config']
        expected_lrs = [motion_config['region_translation_lr'] * motion_config['refine_lr_scale'],
                        motion_config['region_rotation_lr'] * motion_config['refine_lr_scale']]
        for arm in ('source', 'feedback'):
            updates = node['updates'][arm]
            conceptual_training_update_records += len(updates)
            if [row['step'] for row in updates] != expected_steps:
                issues.append(f'frame {frame} {arm}: missing training updates')
            if [row['camera'] for row in updates] != [row['camera'] for row in schedule]:
                issues.append(f'frame {frame} {arm}: camera schedule mismatch')
            if node['hardening'][f'{arm}_steps'] != [181]:
                issues.append(f'frame {frame} {arm}: hardening did not occur exactly at step181')
            refine = next((row for row in updates if row['step'] == 181), None)
            if (refine is None or refine['phase'] != 'gate_refine'
                    or refine['motion_optimizer_lrs'] != expected_lrs):
                issues.append(f'frame {frame} {arm}: refine optimizer LR mismatch')
            if any(not math.isfinite(row['total_loss']) for row in updates):
                issues.append(f'frame {frame} {arm}: nonfinite loss')

        expected_eval_steps = [start] + [step for step in config['evaluation_steps']
                                         if start <= step <= config['end_step']]
        actual_eval_steps = [row['comparison']['step'] for row in node['evaluations']]
        if actual_eval_steps != expected_eval_steps:
            issues.append(f'frame {frame}: evaluation checkpoints differ from protocol')
        conceptual_all20_evaluation_sets += 2 * len(node['evaluations'])
        for evaluation in node['evaluations']:
            for arm in ('source', 'feedback'):
                value = evaluation['arms'][arm]
                if [row['camera'] for row in value['views']] != config['ordinary_camera_names']:
                    issues.append(f'frame {frame} {arm}: all20 evaluation camera mismatch')
                cost = value['cost']
                if (sum(cost['components'].values()) != cost['payload_bits']
                        or cost['total_bytes'] != cost['header_bytes']
                        + math.ceil(cost['payload_bits'] / 8)):
                    issues.append(f'frame {frame} {arm}: actual bitstream accounting mismatch')
        validation = node['final_validation']
        if any(validation[arm]['camera'] != config['validation_camera_name']
               for arm in ('source', 'feedback')):
            issues.append(f'frame {frame}: final validation camera mismatch')
        if node['feedback_mode'] != 'deployed_feedback':
            issues.extend(noop_consistency_issues(node, frame))

        budget = node.get('actual_compute_budget', {})
        expected_multiplier = 2 if node['feedback_mode'] == 'deployed_feedback' else 1
        expected_budget = dict(
            training_updates=len(schedule) * expected_multiplier,
            all20_evaluation_sets=len(node['evaluations']) * expected_multiplier,
            final_validation_renders=expected_multiplier,
            no_op_feedback_reuses_source_compute=(expected_multiplier == 1),
        )
        if budget != expected_budget:
            issues.append(f'frame {frame}: actual compute budget record is inconsistent')
        actual_training_updates += int(budget.get('training_updates', 0))
        actual_all20_evaluation_sets += int(budget.get('all20_evaluation_sets', 0))
        actual_final_validation_renders += int(budget.get('final_validation_renders', 0))

        final = node['evaluations'][-1]['comparison']
        node_rows.append(dict(node=node['node'], transaction_status=node['transaction_status'],
            feedback_mode=node['feedback_mode'], training_updates_per_arm=len(schedule),
            initial_feedback_rd_gain=node['evaluations'][0]['comparison']['feedback_rd_gain'],
            final_feedback_rd_gain=final['feedback_rd_gain'],
            final_feedback_minus_source_bytes=final['feedback_minus_source_bytes'],
            final_feedback_minus_source_psnr=final['feedback_minus_source_psnr'],
            cam00_feedback_minus_source_image_loss=validation['comparison'][
                'feedback_minus_source_image_loss'],
            cam00_feedback_minus_source_psnr=validation['comparison'][
                'feedback_minus_source_psnr'],
            transitions={arm: _transitions(node, arm) for arm in ('source', 'feedback')}))

    result = dict(schema='joint_feedback_controlled_continuation_analysis_v1',
        run=str(run), source_summary_sha256=digest(summary_path),
        completed_nodes=len(summary['nodes']),
        conceptual_pair_records=dict(
            training_updates=conceptual_training_update_records,
            all20_evaluation_sets=conceptual_all20_evaluation_sets,
            final_validation_renders=2 * len(summary['nodes'])),
        actual_compute_budget=dict(
            training_updates=actual_training_updates,
            all20_evaluation_sets=actual_all20_evaluation_sets,
            all20_render_count=(actual_all20_evaluation_sets
                                * len(config['ordinary_camera_names'])),
            final_validation_renders=actual_final_validation_renders),
        source_code_files_verified=len(manifest), issues=issues, nodes=node_rows,
        conclusion=('Every configured node is reported. Committed feedback arms are compared with '
                    'same-state continuations; source fallbacks and rollbacks are exact no-op pairs.'))
    (run / 'analysis.json').write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    if issues:
        raise SystemExit(1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    analyze(args.run)


if __name__ == '__main__':
    main()
