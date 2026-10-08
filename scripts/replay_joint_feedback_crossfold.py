"""Cross-fit all joint gate/precision candidates from immutable stage snapshots.

The replay fits each of the 32 existing candidates on ten ordinary training
cameras and evaluates the fit-selected candidates on the disjoint ten cameras.
The second fold swaps the two camera groups. No candidate is written back to the
source motion and cam00 is never loaded.
"""

import argparse
import copy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def load_protocol(path):
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = {
        'experiment_name', 'source_run', 'output_root', 'device', 'nodes',
        'folds', 'seed', 'fit_steps', 'fit_lr_scale', 'view_sequence',
        'expected_candidate_count', 'expected_source',
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f'Cross-fold replay config is missing: {missing}')
    if not isinstance(values['experiment_name'], str) or not values['experiment_name']:
        raise ValueError('experiment_name must be a nonempty string')
    if isinstance(values['seed'], bool) or not isinstance(values['seed'], int) or values['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (isinstance(values['fit_steps'], bool) or not isinstance(values['fit_steps'], int)
            or values['fit_steps'] < 1):
        raise ValueError('fit_steps must be a positive integer')
    if (isinstance(values['fit_lr_scale'], bool) or
            not math.isfinite(values['fit_lr_scale']) or values['fit_lr_scale'] <= 0):
        raise ValueError('fit_lr_scale must be finite and positive')
    if values['view_sequence'] not in {'round_robin', 'seeded_balanced_cycle'}:
        raise ValueError('Unsupported view_sequence')
    if (isinstance(values['expected_candidate_count'], bool) or
            not isinstance(values['expected_candidate_count'], int) or
            values['expected_candidate_count'] < 1):
        raise ValueError('expected_candidate_count must be a positive integer')
    nodes = [(item['frame'], item['step'], item['level'], item['node'])
             for item in values['nodes']]
    if (not nodes or len(set(nodes)) != len(nodes) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for item in nodes for value in item)):
        raise ValueError('nodes must contain unique nonnegative integer addresses')
    expected_pool = {f'cam{index:02d}.png' for index in range(1, 21)}
    if len(values['folds']) != 2:
        raise ValueError('Cross-fold replay requires two swapped folds')
    first_fit = set(values['folds'][0]['fit_camera_names'])
    first_check = set(values['folds'][0]['check_camera_names'])
    second_fit = set(values['folds'][1]['fit_camera_names'])
    second_check = set(values['folds'][1]['check_camera_names'])
    for fold in values['folds']:
        fit, check = fold['fit_camera_names'], fold['check_camera_names']
        if len(fit) != 10 or len(check) != 10 or len(set(fit)) != 10 or len(set(check)) != 10:
            raise ValueError('Each fold needs ten unique fit and ten unique check cameras')
        if set(fit) & set(check) or set(fit) | set(check) != expected_pool:
            raise ValueError('Each fold must partition cam01..cam20')
        if 'cam00.png' in fit + check:
            raise ValueError('cam00 is reserved for final evaluation')
        if values['fit_steps'] % len(fit):
            raise ValueError('fit_steps must divide evenly across fitting cameras')
    if first_fit != second_check or first_check != second_fit:
        raise ValueError('The second fold must exchange fitting and checking cameras')
    return values


def _same_value(actual, expected):
    if isinstance(expected, list):
        return list(actual) == expected
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=0.0)
    return actual == expected


def validate_source(packet, protocol, diagnostic):
    motion_config = packet['motion']['config']
    for name, expected in protocol['expected_source'].items():
        actual = motion_config.get(name)
        if not _same_value(actual, expected):
            raise ValueError(f'Source snapshot {name}={actual!r}, expected {expected!r}')
    specs = diagnostic.candidate_specs(diagnostic.GateDiagnosticsConfig(
        **packet['diagnostic_config']))
    if len(specs) != protocol['expected_candidate_count']:
        raise ValueError('Source snapshot no longer defines the expected 32 candidates')


def configure_fold(packet, fold, protocol, diagnostic):
    values = copy.deepcopy(packet['diagnostic_config'])
    values.update(enabled=True, seed=protocol['seed'], fit_steps=protocol['fit_steps'],
                  fit_lr_scale=protocol['fit_lr_scale'], view_sequence=protocol['view_sequence'],
                  camera_names=list(fold['fit_camera_names']),
                  evaluation_camera_names=list(fold['check_camera_names']),
                  save_probe_snapshots=False, record_component_gradients=False)
    config = diagnostic.GateDiagnosticsConfig(**values)
    motion_config = importlib.import_module(
        'pipeline.ReCon-GS.model.storage_aware_binary').StorageAwareBinaryConfig(
            **packet['motion']['config'])
    config.validate(motion_config)
    if len(diagnostic.candidate_specs(config)) != protocol['expected_candidate_count']:
        raise ValueError('Fold configuration changed the candidate count')
    return config


def summarize_fold(record):
    """Report only fit-selected candidates as held-out evidence.

    Check-selected winners are retained under a clearly separate selection-only
    field. This prevents the held-out cameras from being described as evidence
    for a candidate they selected themselves.
    """
    by_name = {row['spec']['name']: row for row in record['candidates']}
    fit_winner = by_name[record['selected_by_fitting_rate_score']]
    fit_stop = by_name[record['best_stop']]
    fit_refine = by_name[record['best_refine']]
    original = max((by_name[name] for name in ('stop_original', 'refine_original')),
                   key=lambda row: row['gain'])
    joint = max((row for row in record['candidates']
                 if row['spec'].get('parent_bits') is not None), key=lambda row: row['gain'])
    heldout = {
        'fit_winner_vs_source': fit_winner['evaluation_gain'],
        'fit_original_vs_source': original['evaluation_gain'],
        'fit_winner_vs_fit_original': fit_winner['evaluation_gain'] - original['evaluation_gain'],
        'fit_joint_vs_fit_original': joint['evaluation_gain'] - original['evaluation_gain'],
        'fit_refine_minus_fit_stop': fit_refine['evaluation_gain'] - fit_stop['evaluation_gain'],
    }
    return {
        'fit_selected': {
            'winner': fit_winner['spec']['name'],
            'best_stop': fit_stop['spec']['name'],
            'best_refine': fit_refine['spec']['name'],
            'best_original_precision': original['spec']['name'],
            'best_joint_precision': joint['spec']['name'],
        },
        'heldout_rate_distortion_gains': heldout,
        'heldout_supports_fit_selected_refine': heldout['fit_refine_minus_fit_stop'] > 0,
        'selection_analysis_only': {
            'check_selected_winner': record['selected_by_check_rate_score'],
            'meaning': 'Selected by check cameras; never used as held-out generalization evidence.',
        },
    }


def summarize_crossfold(folds):
    selected = [fold['summary']['fit_selected'] for fold in folds]
    gains = [fold['summary']['heldout_rate_distortion_gains'] for fold in folds]
    return {
        'same_fit_selected_winner': selected[0]['winner'] == selected[1]['winner'],
        'same_fit_selected_gate_action':
            selected[0]['winner'].startswith('refine') == selected[1]['winner'].startswith('refine'),
        'heldout_refine_preference_same_sign':
            (gains[0]['fit_refine_minus_fit_stop'] > 0) ==
            (gains[1]['fit_refine_minus_fit_stop'] > 0),
        'both_fit_winners_beat_source_on_heldout':
            all(value['fit_winner_vs_source'] > 0 for value in gains),
        'both_joint_winners_beat_fitted_original_on_heldout':
            all(value['fit_joint_vs_fit_original'] > 0 for value in gains),
        'semantics': 'Each gain evaluates a candidate selected only by the opposite fitting split. '
                     'The three nodes are mechanism-debug cases, not a population estimate.',
    }


def _append_progress(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def replay(protocol_path, output=None):
    from libgs.renderer import RendererConfig
    from libgs.utils.config import to_structured
    from scripts.diagnose_binary_gates import file_digest

    package = importlib.import_module('pipeline.ReCon-GS')
    diagnostic = importlib.import_module('pipeline.ReCon-GS.model.gate_diagnostics')
    snapshots = importlib.import_module('pipeline.ReCon-GS.model.gate_probe_snapshot')
    render = importlib.import_module('pipeline.ReCon-GS.gate_diagnostic_render')
    gaussian = importlib.import_module('pipeline.ReCon-GS.model.gaussian')

    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    source_run = _resolve_repo_path(protocol['source_run']).resolve()
    if output is None:
        stamp = time.strftime('%Y%m%dT%H%M%S')
        output = _resolve_repo_path(protocol['output_root']) / stamp
    output = Path(output).resolve()
    if (output.exists() or source_run.is_relative_to(output)
            or output.is_relative_to(source_run)):
        raise ValueError('Replay output must be new and outside the source run')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(protocol_path.read_text(encoding='utf-8'), encoding='utf-8')
    source_snapshot_dir = output / 'source_snapshot'
    source_snapshot_dir.mkdir()
    source_code_paths = [Path(__file__).resolve(), protocol_path,
        Path(diagnostic.__file__).resolve(), Path(snapshots.__file__).resolve(),
        Path(render.__file__).resolve(),
        Path(importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary').__file__).resolve()]
    source_code = []
    for index, path in enumerate(source_code_paths):
        destination = source_snapshot_dir / f'{index:02d}_{path.name}'
        payload = path.read_bytes()
        destination.write_bytes(payload)
        source_code.append(dict(source=str(path), snapshot=str(destination),
                                sha256=file_digest(path), bytes=len(payload)))
    (source_snapshot_dir / 'manifest.json').write_text(
        json.dumps(source_code, indent=2), encoding='utf-8')
    progress_path = output / 'progress.jsonl'
    started = time.perf_counter()
    all_nodes = []

    for node_spec in protocol['nodes']:
        frame, step = node_spec['frame'], node_spec['step']
        level, node = node_spec['level'], node_spec['node']
        snapshot = source_run / 'gate_diagnostics' / 'snapshots' / f'frame_{frame:03d}_step_{step:04d}.pth'
        source_digest = file_digest(snapshot)
        packet = torch.load(snapshot, map_location='cpu', weights_only=False)
        if packet['frame'] != frame or packet['step'] != step:
            raise ValueError('Snapshot identity does not match the configured node')
        validate_source(packet, protocol, diagnostic)
        motion = snapshots.restore_probe_motion(packet, protocol['device'])
        trace = diagnostic.node_trace(motion, step, level, node)
        if not trace['reachable']:
            raise ValueError(f'Configured node is not reachable: {node_spec}')

        data_config, unused = to_structured(packet['data_config'], package.DataConfig)
        if unused:
            raise ValueError(f'Unknown saved data fields: {unused}')
        renderer_config, unused = to_structured(packet['renderer_config'], RendererConfig)
        if unused:
            raise ValueError(f'Unknown saved renderer fields: {unused}')
        state = packet['render_state']
        model = gaussian.GaussianModel(state['max_sh_degree'])
        model.active_sh_degree = state['active_sh_degree']
        model.render_attribute_cache = state['render_attribute_cache']
        for name, value in state['attributes'].items():
            setattr(model, name, torch.nn.Parameter(value.to(protocol['device']), requires_grad=False))
        objective = render.FrozenImageObjective(model, renderer_config,
            packet['background'].to(protocol['device']), packet['module_config']['lambda_dssim'],
            state['old_count'])

        camera_pool = sorted({name for fold in protocol['folds']
                              for key in ('fit_camera_names', 'check_camera_names')
                              for name in fold[key]})
        saved_names = [entry['camera'] for entry in packet['views']]
        load_names = list(dict.fromkeys(saved_names + camera_pool))
        with diagnostic.isolated_rng(protocol['device']):
            data = package.DataModule(data_config)
            data.setup()
            data.current_frame = frame
            loaded = [now for now, _ in data.selective_training_pairs(
                load_names, protocol['device'], 0.)]
            by_name = {view.path.name: view for view in loaded}
            snapshots.verify_snapshot_views(packet, [by_name[name] for name in saved_names])
            dependencies = {
                'saved_snapshot_dependencies_verified': True,
                'saved_views': packet['views'],
                'added_views': [snapshots.view_fingerprint(by_name[name])
                                for name in camera_pool if name not in saved_names],
                'dataset_root_resolved': packet['dataset_root_resolved'],
            }
            reference = motion.reference_storage_bytes(packet['first_frame_gs_count'])
            rate_weight = motion.config.storage_loss_weight * motion.storage_scale(step)
            if motion.stage(step) not in {'gate_compress', 'gate_refine'}:
                rate_weight = 0.0
            fold_outputs = []
            for fold in protocol['folds']:
                config = configure_fold(packet, fold, protocol, diagnostic)
                fit_views = [by_name[name] for name in config.camera_names]
                check_views = [by_name[name] for name in config.evaluation_camera_names]
                plan = diagnostic.private_view_plan(config, frame, step, level, node)
                counts = {name: plan['indices'].count(index)
                          for index, name in enumerate(config.camera_names)}
                expected_visits = config.fit_steps // len(config.camera_names)
                if set(counts.values()) != {expected_visits}:
                    raise AssertionError('Balanced plan must visit every fitting camera equally')
                _append_progress(progress_path, dict(event='fold_started', frame=frame, step=step,
                    level=level, node=node, fold=fold['name'], elapsed_seconds=time.perf_counter() - started))
                record = diagnostic.compare_joint_node(motion, step, level, node, fit_views,
                    objective, config, reference_bytes=reference, rate_weight=rate_weight,
                    evaluation_views=check_views, view_plan=plan)
                fold_result = dict(name=fold['name'], fit_camera_names=config.camera_names,
                    check_camera_names=config.evaluation_camera_names, view_plan=plan,
                    summary=summarize_fold(record), record=record)
                fold_path = output / f'frame_{frame:03d}_step_{step:04d}_level_{level}_node_{node}_{fold["name"]}.json'
                fold_path.write_text(json.dumps(fold_result, indent=2, allow_nan=False), encoding='utf-8')
                fold_outputs.append(fold_result)
                _append_progress(progress_path, dict(event='fold_completed', frame=frame, step=step,
                    level=level, node=node, fold=fold['name'], result=str(fold_path),
                    elapsed_seconds=time.perf_counter() - started))

        if file_digest(snapshot) != source_digest:
            raise AssertionError('Source snapshot changed during replay')
        node_result = dict(node=node_spec, source_snapshot=str(snapshot),
            source_snapshot_sha256=source_digest, source_trace=trace, dependencies=dependencies,
            reference_bytes=reference, rate_weight=rate_weight,
            actual_cost_scope='Production motion bitstream bytes; birth PLY is candidate-constant and omitted.',
            folds=[dict(name=row['name'], summary=row['summary']) for row in fold_outputs],
            crossfold=summarize_crossfold(fold_outputs))
        all_nodes.append(node_result)
        (output / 'summary.partial.json').write_text(json.dumps(dict(nodes=all_nodes), indent=2,
            allow_nan=False), encoding='utf-8')

    summary = dict(schema='joint_feedback_crossfold_v1', experiment_name=protocol['experiment_name'],
        source_run=str(source_run), config=str(protocol_path), source_code_snapshot=source_code,
        total_seconds=time.perf_counter() - started,
        protocol=dict(seed=protocol['seed'], fit_steps=protocol['fit_steps'],
            fit_lr_scale=protocol['fit_lr_scale'], view_sequence=protocol['view_sequence'],
            candidate_count=protocol['expected_candidate_count'], folds=protocol['folds']),
        nodes=all_nodes, limitations=[
            'The three nodes are mechanism-debug cases and do not estimate overall error rates.',
            'Candidate fits are finite 20-step local optimizations, not global convergence.',
            'No candidate parameters, gate logits or precision logits are written back to training.',
            'Check-selected winners are selection analysis only; held-out claims use fit-selected candidates.',
        ])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    _append_progress(progress_path, dict(event='completed', elapsed_seconds=summary['total_seconds']))
    print(f'Cross-fold replay completed: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    replay(args.config, args.output)


if __name__ == '__main__':
    main()
