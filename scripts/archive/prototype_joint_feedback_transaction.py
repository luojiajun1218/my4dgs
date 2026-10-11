"""Apply cross-fit joint feedback transactionally to independent snapshot clones."""

import argparse
import copy
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def load_protocol(path):
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = {'experiment_name', 'source_crossfold_run', 'output_root', 'device',
        'nodes', 'require_bidirectional_positive', 'source_candidate_gain', 'camera_names',
        'seed', 'deployment_fit_steps', 'fit_lr_scale', 'view_sequence',
        'max_feedback_logit_steps', 'feedback_lr', 'feedback_betas', 'feedback_eps',
        'acceptance_requires_strict_positive_realized_gain'}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f'Feedback transaction config is missing: {missing}')
    if not isinstance(values['experiment_name'], str) or not values['experiment_name']:
        raise ValueError('experiment_name must be a nonempty string')
    nodes = [(row['frame'], row['step'], row['level'], row['node']) for row in values['nodes']]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError('nodes must be nonempty and unique')
    expected_cameras = [f'cam{index:02d}.png' for index in range(1, 21)]
    if values['camera_names'] != expected_cameras or 'cam00.png' in values['camera_names']:
        raise ValueError('Deployment fitting must use cam01..cam20 in order and exclude cam00')
    for name in ('require_bidirectional_positive',
                 'acceptance_requires_strict_positive_realized_gain'):
        if not isinstance(values[name], bool):
            raise ValueError(f'{name} must be bool')
    for name in ('seed', 'deployment_fit_steps', 'max_feedback_logit_steps'):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'{name} has an invalid integer budget')
    if values['deployment_fit_steps'] % len(values['camera_names']):
        raise ValueError('Deployment fitting must balance all cameras equally')
    for name in ('fit_lr_scale', 'feedback_lr', 'feedback_eps'):
        value = values[name]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if (len(values['feedback_betas']) != 2 or
            not all(0 <= value < 1 for value in values['feedback_betas'])):
        raise ValueError('feedback_betas must contain two values in [0, 1)')
    if values['view_sequence'] not in {'round_robin', 'seeded_balanced_cycle'}:
        raise ValueError('Unsupported view_sequence')
    if values['source_candidate_gain'] != 0.0:
        raise ValueError('The unchanged source action must have exactly zero gain')
    return values


def select_crossfit_candidate(fold_records, require_bidirectional_positive=True):
    """Select one identical candidate across folds; source is the zero-gain fallback."""
    if len(fold_records) != 2:
        raise ValueError('Cross-fit selection requires two swapped folds')
    by_fold = [{row['spec']['name']: row for row in record['candidates']}
               for record in fold_records]
    if set(by_fold[0]) != set(by_fold[1]):
        raise ValueError('The two folds have different candidate spaces')
    rows = []
    for name in sorted(by_fold[0]):
        gains = [fold[name]['evaluation_gain'] for fold in by_fold]
        eligible = (all(value > 0 for value in gains)
                    if require_bidirectional_positive else sum(gains) / len(gains) > 0)
        rows.append(dict(name=name, directional_gains=gains,
            mean_gain=sum(gains) / len(gains), eligible=eligible,
            spec=copy.deepcopy(by_fold[0][name]['spec'])))
    eligible = [row for row in rows if row['eligible']]
    selected = (max(eligible, key=lambda row: (row['mean_gain'], row['name']))
                if eligible else dict(name='source', directional_gains=[0.0, 0.0],
                                      mean_gain=0.0, eligible=True, spec=None))
    return dict(selected=selected, candidates=rows,
        semantics='Both check directions participate in candidate selection; these gains are selection values, not held-out evidence.')


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def capture_transaction(motion):
    return dict(motion=copy.deepcopy(motion.capture()),
        gradients={name: None if parameter.grad is None else parameter.grad.detach().clone()
                   for name, parameter in motion.named_parameters()},
        requires_grad={name: parameter.requires_grad for name, parameter in motion.named_parameters()},
        caches=copy.deepcopy(dict(fixed_paths=getattr(motion, '_fixed_paths', None),
            fixed_rows=getattr(motion, '_fixed_rows', None),
            fixed_terminal=getattr(motion, '_fixed_terminal', None),
            frozen_storage_parts=getattr(motion, '_frozen_storage_parts', {}))),
        rng=_rng_state())


def restore_transaction(motion, packet):
    motion.restore(copy.deepcopy(packet['motion']))
    for name, parameter in motion.named_parameters():
        parameter.requires_grad_(packet['requires_grad'][name])
        gradient = packet['gradients'][name]
        parameter.grad = None if gradient is None else gradient.to(parameter).clone()
    motion._fixed_paths = copy.deepcopy(packet['caches']['fixed_paths'])
    motion._fixed_rows = copy.deepcopy(packet['caches']['fixed_rows'])
    motion._fixed_terminal = copy.deepcopy(packet['caches']['fixed_terminal'])
    motion._frozen_storage_parts = copy.deepcopy(packet['caches']['frozen_storage_parts'])
    _restore_rng(packet['rng'])


def _exact_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and
                left.dtype == right.dtype and left.device == right.device and
                left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and
                left.dtype == right.dtype and left.shape == right.shape and
                np.array_equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict) and
                left.keys() == right.keys() and
                all(_exact_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right) and
                all(_exact_equal(a, b) for a, b in zip(left, right)))
    return left == right


def transaction_matches(motion, expected):
    actual = capture_transaction(motion)
    return _exact_equal(actual, expected)


def _zero_optimizer_rows(optimizer, parameter, mask):
    """Clear target-row moments while retaining Adam's parameter-wide scalar step."""
    state = optimizer.state.get(parameter, {})
    cleared = []
    with torch.no_grad():
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = state.get(name)
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[mask] = 0
                cleared.append(name)
    return cleared


def apply_candidate_motion(deployment, candidate):
    changed, optimizer_resets = [], []
    with torch.no_grad():
        for level, mask in enumerate(candidate.train_masks):
            rows = torch.where(mask)[0]
            if not len(rows):
                continue
            for field in ('region_translation', 'region_rotation'):
                target = getattr(deployment, field)[level]
                source = getattr(candidate, field)[level]
                delta_l2 = float((target[rows] - source[rows]).norm())
                target[rows] = source[rows]
                cleared = _zero_optimizer_rows(deployment.motion_optimizer, target, mask)
                if target.grad is not None:
                    target.grad[mask] = 0
                changed.append(dict(parameter=f'{field}.{level}', rows=rows.cpu().tolist(),
                                    copied_delta_l2=delta_l2))
                optimizer_resets.append(dict(parameter=f'{field}.{level}', rows=len(rows),
                    cleared=cleared, scalar_step_preserved=True))
    deployment._fixed_paths = deployment._fixed_rows = deployment._fixed_terminal = None
    deployment._frozen_storage_parts = {}
    return changed, optimizer_resets


def _hard_feedback_state(motion, candidate, step, level, node):
    position = motion._refine_position(level, node)
    gate = bool(motion.refinement_masks(step)[level][position])
    target_gate = bool(candidate.diagnostic_gates[level][position])
    precision = [value.argmax(-1) for value in
                 motion._precision_weights(step, differentiable=False)]
    matches = []
    margins = []
    for depth, mask in enumerate(candidate.train_masks):
        rows = torch.where(mask)[0]
        if not len(rows):
            continue
        target = candidate.diagnostic_precision[depth][rows]
        matches.append(bool((precision[depth][rows] == target).all()))
        logits = motion.precision_logits[depth][rows]
        chosen = logits.gather(1, target[:, None]).squeeze(1)
        alternatives = logits.clone()
        alternatives.scatter_(1, target[:, None], float('-inf'))
        margins.append(float((chosen - alternatives.max(-1).values).min()))
    gate_logit = float(motion.refine_logits[level][position])
    return dict(matches=gate == target_gate and all(matches), gate=gate,
        target_gate=target_gate, gate_logit=gate_logit,
        gate_target_margin=gate_logit if target_gate else -gate_logit,
        minimum_precision_target_margin=min(margins) if margins else None,
        precision_rows=sum(int(mask.sum()) for mask in candidate.train_masks))


def optimize_feedback_logits(deployment, candidate, step, level, node, protocol):
    if deployment.hard_masks is not None:
        raise ValueError('Feedback prototype requires a pre-hardening gate-compress snapshot')
    position = deployment._refine_position(level, node)
    gate_parameter = deployment.refine_logits[level]
    precision_parameters = [(depth, parameter, mask) for depth, (parameter, mask) in
        enumerate(zip(deployment.precision_logits, candidate.train_masks)) if bool(mask.any())]
    parameters = [gate_parameter] + [parameter for _, parameter, _ in precision_parameters]
    saved_gradients = {id(parameter): None if parameter.grad is None else parameter.grad.detach().clone()
                       for parameter in parameters}
    optimizer = torch.optim.Adam(parameters, lr=protocol['feedback_lr'],
        betas=tuple(protocol['feedback_betas']), eps=protocol['feedback_eps'])
    initial = _hard_feedback_state(deployment, candidate, step, level, node)
    losses, steps = [], 0
    while steps < protocol['max_feedback_logit_steps']:
        state = _hard_feedback_state(deployment, candidate, step, level, node)
        if state['matches']:
            break
        optimizer.zero_grad(set_to_none=True)
        target_gate = gate_parameter.new_tensor(float(state['target_gate']))
        terms = [F.binary_cross_entropy_with_logits(gate_parameter[position], target_gate)]
        for depth, parameter, mask in precision_parameters:
            terms.append(F.cross_entropy(parameter[mask], candidate.diagnostic_precision[depth][mask]))
        loss = sum(terms)
        loss.backward()
        gate_mask = torch.zeros_like(gate_parameter, dtype=torch.bool)
        gate_mask[position] = True
        if gate_parameter.grad is not None:
            gate_parameter.grad[~gate_mask] = 0
        for _, parameter, mask in precision_parameters:
            if parameter.grad is not None:
                parameter.grad[~mask] = 0
        optimizer.step()
        steps += 1
        losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    gate_saved = saved_gradients[id(gate_parameter)]
    gate_parameter.grad = (None if gate_saved is None else gate_saved.to(gate_parameter).clone())
    if gate_parameter.grad is not None:
        gate_parameter.grad[position] = 0
    for _, parameter, mask in precision_parameters:
        saved = saved_gradients[id(parameter)]
        parameter.grad = None if saved is None else saved.to(parameter).clone()
        if parameter.grad is not None:
            parameter.grad[mask] = 0
    final = _hard_feedback_state(deployment, candidate, step, level, node)
    changed = dict(gate=dict(level=level, node=node, position=position,
        target_open=final['target_gate']), precision=[dict(level=depth,
        rows=torch.where(mask)[0].cpu().tolist()) for depth, _, mask in precision_parameters])
    return dict(steps=steps, max_steps=protocol['max_feedback_logit_steps'],
        initial=initial, final=final, first_loss=losses[0] if losses else None,
        last_loss=losses[-1] if losses else None, changed_rows=changed), precision_parameters


def reset_accepted_gate_optimizer(deployment, level, node, precision_parameters):
    position = deployment._refine_position(level, node)
    gate = deployment.refine_logits[level]
    gate_mask = torch.zeros_like(gate, dtype=torch.bool)
    gate_mask[position] = True
    rows = [dict(parameter=f'refine_logits.{level}', rows=[position],
                 cleared=_zero_optimizer_rows(deployment.gate_optimizer, gate, gate_mask),
                 scalar_step_preserved=True)]
    if gate.grad is not None:
        gate.grad[gate_mask] = 0
    for depth, parameter, mask in precision_parameters:
        rows.append(dict(parameter=f'precision_logits.{depth}',
            rows=torch.where(mask)[0].cpu().tolist(),
            cleared=_zero_optimizer_rows(deployment.gate_optimizer, parameter, mask),
            scalar_step_preserved=True))
        if parameter.grad is not None:
            parameter.grad[mask] = 0
    return rows


def accept_or_restore(motion, transaction, realized_gain, hard_matches, strict_positive=True):
    accepted = bool(hard_matches and math.isfinite(realized_gain) and
                    (realized_gain > 0 if strict_positive else realized_gain >= 0))
    if not accepted:
        restore_transaction(motion, transaction)
    return accepted


def evaluate_motion(motion, step, views, objective, diagnostic):
    rows = []
    with torch.no_grad():
        xyz, rotation = motion(step)
        for view in views:
            metrics = objective.evaluate(view, xyz, rotation)
            rows.append(dict(camera=view.path.name, image_loss=float(metrics['image_loss']),
                             psnr=float(metrics['psnr'])))
    return dict(image_loss=sum(row['image_loss'] for row in rows) / len(rows), views=rows,
                cost=diagnostic.exact_cost(motion, step))


def _configure_deployment(packet, protocol, diagnostic):
    values = copy.deepcopy(packet['diagnostic_config'])
    values.update(camera_names=list(protocol['camera_names']), evaluation_camera_names=[],
        fit_steps=protocol['deployment_fit_steps'], fit_lr_scale=protocol['fit_lr_scale'],
        seed=protocol['seed'], view_sequence=protocol['view_sequence'],
        save_probe_snapshots=False, record_component_gradients=False)
    config = diagnostic.GateDiagnosticsConfig(**values)
    storage = importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary')
    config.validate(storage.StorageAwareBinaryConfig(**packet['motion']['config']))
    return config


def run(protocol_path, output=None):
    from libgs.renderer import RendererConfig
    from libgs.utils.config import to_structured
    from scripts.archive.diagnose_binary_gates import file_digest

    package = importlib.import_module('pipeline.ReCon-GS.archive')
    diagnostic = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_diagnostics')
    snapshots = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_probe_snapshot')
    render = importlib.import_module('pipeline.ReCon-GS.archive.gate_diagnostic_render')
    gaussian = importlib.import_module('pipeline.ReCon-GS.archive.model.gaussian')

    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    source_run = _repo_path(protocol['source_crossfold_run']).resolve()
    source_summary = json.loads((source_run / 'summary.json').read_text(encoding='utf-8'))
    source_analysis = json.loads((source_run / 'analysis.json').read_text(encoding='utf-8'))
    expected_candidates = (len(protocol['nodes']) * len(source_summary.get('protocol', {}).get('folds', []))
                           * source_summary.get('protocol', {}).get('candidate_count', 0))
    if (source_summary.get('schema') != 'joint_feedback_crossfold_v1' or
            source_analysis.get('completed_candidates') != expected_candidates
            or source_analysis.get('issues')):
        raise ValueError('Source cross-fold run has not passed the expected audit')
    if output is None:
        output = _repo_path(protocol['output_root']) / time.strftime('%Y%m%dT%H%M%S')
    output = Path(output).resolve()
    historical_run = Path(source_summary['source_run']).resolve()
    protected_roots = (source_run, historical_run)
    if (output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output)
                               for root in protected_roots)):
        raise ValueError('Feedback output must be new and outside all read-only source runs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(protocol_path.read_text(encoding='utf-8'), encoding='utf-8')
    source_snapshot_dir = output / 'source_snapshot'
    source_snapshot_dir.mkdir()
    source_code_paths = [Path(__file__).resolve(),
        Path(diagnostic.__file__).resolve(), Path(snapshots.__file__).resolve(),
        Path(importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary').__file__).resolve()]
    source_code = []
    for index, path in enumerate(source_code_paths):
        destination = source_snapshot_dir / f'{index:02d}_{path.name}'
        payload = path.read_bytes()
        destination.write_bytes(payload)
        source_code.append(dict(source=str(path), snapshot=str(destination),
                                sha256=file_digest(path), bytes=len(payload)))
    (source_snapshot_dir / 'manifest.json').write_text(
        json.dumps(source_code, indent=2), encoding='utf-8')
    source_files = [source_run / 'summary.json', source_run / 'analysis.json']
    source_hashes = {str(path): file_digest(path) for path in source_files}
    started, results = time.perf_counter(), []

    source_nodes = {(row['node']['frame'], row['node']['step'], row['node']['level'],
                     row['node']['node']): row for row in source_summary['nodes']}
    for address in protocol['nodes']:
        key = (address['frame'], address['step'], address['level'], address['node'])
        if key not in source_nodes:
            raise ValueError(f'Node is absent from source cross-fold run: {key}')
        node_source = source_nodes[key]
        frame, step, level, node = key
        fold_paths, fold_packets = [], []
        for fold in source_summary['protocol']['folds']:
            path = source_run / (f'frame_{frame:03d}_step_{step:04d}_level_{level}_'
                                 f'node_{node}_{fold["name"]}.json')
            fold_paths.append(path)
            fold_packets.append(json.loads(path.read_text(encoding='utf-8')))
        fold_hashes = {str(path): file_digest(path) for path in fold_paths}
        selection = select_crossfit_candidate([row['record'] for row in fold_packets],
            protocol['require_bidirectional_positive'])
        selected = selection['selected']
        snapshot_path = Path(node_source['source_snapshot'])
        snapshot_hash = file_digest(snapshot_path)
        if snapshot_hash != node_source['source_snapshot_sha256']:
            raise AssertionError('Historical source snapshot hash changed')
        packet = torch.load(snapshot_path, map_location='cpu', weights_only=False)
        source = snapshots.restore_probe_motion(packet, protocol['device'])

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

        saved_names = [row['camera'] for row in packet['views']]
        load_names = list(dict.fromkeys(saved_names + protocol['camera_names']))
        with diagnostic.isolated_rng(protocol['device']):
            data = package.DataModule(data_config)
            data.setup()
            data.current_frame = frame
            loaded = [now for now, _ in data.selective_training_pairs(
                load_names, protocol['device'], 0.)]
            by_name = {view.path.name: view for view in loaded}
            snapshots.verify_snapshot_views(packet, [by_name[name] for name in saved_names])
            views = [by_name[name] for name in protocol['camera_names']]
            fingerprints = [snapshots.view_fingerprint(view) for view in views]
            recorded_dependencies = {row['camera']: row for row in
                node_source['dependencies']['saved_views'] + node_source['dependencies']['added_views']}
            if {row['camera']: row for row in fingerprints} != {
                    name: recorded_dependencies[name] for name in protocol['camera_names']}:
                raise ValueError('All-20 deployment dependencies differ from the cross-fold run')
            source_evaluation = evaluate_motion(source, step, views, objective, diagnostic)
            reference = source.reference_storage_bytes(packet['first_frame_gs_count'])
            rate_weight = source.config.storage_loss_weight * source.storage_scale(step)
            if source.stage(step) not in {'gate_compress', 'gate_refine'}:
                rate_weight = 0.0

            result = dict(node=address, selection=selection, selected=selected['name'],
                source_snapshot=str(snapshot_path), source_snapshot_sha256=snapshot_hash,
                source_fold_files=fold_hashes, view_fingerprints=fingerprints,
                source_evaluation=source_evaluation, reference_bytes=reference,
                rate_weight=rate_weight, status='source_retained', accepted=False,
                semantics='Cross-fold check values are used for selection. All-20 fitting/evaluation '
                          'uses training/selection cameras and is not independent generalization evidence.')
            if selected['name'] != 'source':
                config = _configure_deployment(packet, protocol, diagnostic)
                plan = diagnostic.private_view_plan(config, frame, step, level, node)
                if (len(plan['indices']) != len(views) or
                        sorted(plan['indices']) != list(range(len(views)))):
                    raise AssertionError('Deployment plan must use each of the twenty cameras once')
                spec = selected['spec']
                candidate = diagnostic.DiagnosticCandidate(source, step, level, node,
                    spec['opened'], config, 'fixed', parent_bits=spec['parent_bits'],
                    descendant_bits=spec['descendant_bits'])
                candidate_before = diagnostic._evaluate(candidate, views, objective, step, 0)
                fit_losses = []
                for fit_index, camera_index in enumerate(plan['indices']):
                    view = views[camera_index]
                    xyz, rotation = candidate()
                    loss = objective(view, xyz, rotation)
                    fit_losses.append(dict(step=fit_index + 1, camera=view.path.name,
                                           image_loss=float(loss.detach())))
                    candidate.fit_step(loss)
                candidate_after = diagnostic._evaluate(candidate, views, objective, step, 0)
                predicted_byte_delta = (candidate_after['cost']['total_bytes'] -
                                        source_evaluation['cost']['total_bytes'])
                predicted_gain = (source_evaluation['image_loss'] - candidate_after['image_loss'] -
                    rate_weight * predicted_byte_delta / reference)

                deployment = diagnostic.clone_motion(source)
                transaction = capture_transaction(deployment)
                motion_changes, motion_resets = apply_candidate_motion(deployment, candidate)
                feedback, precision_parameters = optimize_feedback_logits(
                    deployment, candidate, step, level, node, protocol)
                realized = evaluate_motion(deployment, step, views, objective, diagnostic)
                realized_byte_delta = (realized['cost']['total_bytes'] -
                                       source_evaluation['cost']['total_bytes'])
                realized_gain = (source_evaluation['image_loss'] - realized['image_loss'] -
                    rate_weight * realized_byte_delta / reference)
                accepted = accept_or_restore(deployment, transaction, realized_gain,
                    feedback['final']['matches'],
                    protocol['acceptance_requires_strict_positive_realized_gain'])
                gate_resets = (reset_accepted_gate_optimizer(deployment, level, node,
                    precision_parameters) if accepted else [])
                if accepted:
                    torch.save(snapshots.cpu_copy(deployment.capture()),
                               output / f'deployed_frame_{frame:03d}_step_{step:04d}.pth')
                result.update(status='committed_to_clone' if accepted else 'rolled_back',
                    accepted=accepted, deployment_view_plan=plan, fit_losses=fit_losses,
                    candidate_before=candidate_before, candidate_after=candidate_after,
                    predicted_all20_gain=predicted_gain,
                    predicted_byte_delta=predicted_byte_delta,
                    feedback=feedback, motion_changes=motion_changes,
                    motion_optimizer_resets=motion_resets,
                    gate_optimizer_resets=gate_resets,
                    realized_evaluation=realized, realized_gain=realized_gain,
                    realized_byte_delta=realized_byte_delta,
                    rollback_verified=(transaction_matches(deployment, transaction)
                                       if not accepted else None))
                if not accepted and not result['rollback_verified']:
                    raise AssertionError('Rejected feedback transaction did not restore exact state')
            results.append(result)
            (output / f'frame_{frame:03d}_step_{step:04d}_level_{level}_node_{node}.json').write_text(
                json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')

        if file_digest(snapshot_path) != snapshot_hash:
            raise AssertionError('Historical snapshot changed during feedback transaction')
        for path, digest in fold_hashes.items():
            if file_digest(path) != digest:
                raise AssertionError('Immutable cross-fold record changed during feedback transaction')

    for path, digest in source_hashes.items():
        if file_digest(path) != digest:
            raise AssertionError('Source cross-fold summary changed during feedback transaction')
    summary = dict(schema='joint_feedback_transaction_v1', experiment_name=protocol['experiment_name'],
        source_crossfold_run=str(source_run), config=str(protocol_path),
        total_seconds=time.perf_counter() - started,
        source_code_snapshot=source_code,
        committed=sum(row['accepted'] for row in results), source_retained=sum(
            row['selected'] == 'source' for row in results), rolled_back=sum(
            row['status'] == 'rolled_back' for row in results), nodes=results,
        optimizer_reset_semantics='Only target-row Adam moments exp_avg/exp_avg_sq/max_exp_avg_sq '
            'are zeroed. The parameter-wide scalar step is preserved.',
        limitations=['All twenty cameras are training/selection data, not independent validation.',
            'This is an offline clone transaction prototype and is not connected to production training.',
            'BCE/CE realizes a search-selected policy target; it does not alone establish a new learning method.'])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Joint feedback transaction completed: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    raise SystemExit('Archived training launcher; use main.py with the rigid-tree config')
