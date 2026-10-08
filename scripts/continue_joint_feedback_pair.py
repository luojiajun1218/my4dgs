"""Continue source and feedback motion clones under one controlled schedule."""

import argparse
import copy
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def load_protocol(path):
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = {
        'experiment_name', 'source_transaction_run', 'output_root', 'device',
        'nodes', 'ordinary_camera_names', 'validation_camera_name', 'seed',
        'end_step', 'current_camera_from_snapshot', 'future_view_sequence',
        'evaluation_steps', 'actual_cost', 'final_validation_step',
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f'Continuation config is missing: {missing}')
    if not isinstance(values['experiment_name'], str) or not values['experiment_name']:
        raise ValueError('experiment_name must be a nonempty string')
    cameras = values['ordinary_camera_names']
    expected = [f'cam{index:02d}.png' for index in range(1, 21)]
    if cameras != expected or values['validation_camera_name'] in cameras:
        raise ValueError('ordinary_camera_names must be cam01..cam20 and exclude validation')
    if values['validation_camera_name'] != 'cam00.png':
        raise ValueError('The final validation camera must be cam00.png')
    for name in ('current_camera_from_snapshot', 'actual_cost'):
        if not isinstance(values[name], bool) or not values[name]:
            raise ValueError(f'{name} must be true for the controlled continuation')
    if values['future_view_sequence'] != 'seeded_balanced_cycle':
        raise ValueError('future_view_sequence must be seeded_balanced_cycle')
    for name in ('seed', 'end_step', 'final_validation_step'):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    nodes = []
    for row in values['nodes']:
        address = tuple(row.get(name) for name in ('frame', 'step', 'level', 'node'))
        if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in address) or not isinstance(row.get('deployed_packet'), str)
                or not row['deployed_packet']):
            raise ValueError('Each node needs a nonnegative address and deployed_packet')
        if row['step'] > values['end_step']:
            raise ValueError('Node step cannot exceed end_step')
        nodes.append(address)
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError('nodes must be nonempty and unique')
    steps = values['evaluation_steps']
    if (not steps or steps != sorted(set(steps)) or any(
            isinstance(step, bool) or not isinstance(step, int) or step < 1
            for step in steps)):
        raise ValueError('evaluation_steps must contain unique ascending positive integers')
    if values['final_validation_step'] != values['end_step']:
        raise ValueError('Final validation must run exactly at end_step')
    return values


def balanced_continuation_schedule(camera_names, seed, frame, start_step, end_step,
                                   current_camera):
    """Use the saved current camera, then balanced seeded permutations."""
    if current_camera not in camera_names:
        raise ValueError('Snapshot training camera is outside the ordinary camera pool')
    if start_step > end_step:
        raise ValueError('start_step must not exceed end_step')
    remaining = end_step - start_step
    if remaining % len(camera_names):
        raise ValueError('Future continuation steps must balance the ordinary camera pool')
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(frame), int(start_step), int(end_step)]))
    future = []
    while len(future) < remaining:
        future.extend(camera_names[int(index)] for index in rng.permutation(len(camera_names)))
    return ([dict(step=start_step, camera=current_camera, role='recomputed_snapshot_step')]
            + [dict(step=step, camera=camera, role='seeded_balanced_future')
               for step, camera in zip(range(start_step + 1, end_step + 1), future)])


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {name: _move(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def restore_motion_capture(packet, device, storage):
    config = storage.StorageAwareBinaryConfig(**packet['config'])
    motion = storage.StorageAwareBinaryMotion(config)
    motion.restore(_move(packet, device))
    return motion


def _exact_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right)
                and left.dtype == right.dtype and left.device == right.device
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (isinstance(left, dict) and isinstance(right, dict)
                and left.keys() == right.keys()
                and all(_exact_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_exact_equal(a, b) for a, b in zip(left, right)))
    return left == right


def _motion_runtime_state(motion):
    cache_prefixes = ('_cached_', '_fixed_', '_frozen_', '_static_')
    return dict(capture=motion.capture(),
        gradients={name: None if parameter.grad is None else parameter.grad.detach().clone()
                   for name, parameter in motion.named_parameters()},
        requires_grad={name: parameter.requires_grad
                       for name, parameter in motion.named_parameters()},
        caches={name: copy.deepcopy(value) for name, value in vars(motion).items()
                if name.startswith(cache_prefixes)})


def clone_source_noop(source, storage):
    """Deep-copy every runtime field for a mirrored source fallback/rollback arm."""
    del storage  # Kept in the signature so callers share one initialization path.
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (torch.cuda.get_rng_state(source.base_xyz.device).clone()
                if source.base_xyz.is_cuda else None)
    clone = copy.deepcopy(source)
    if (not torch.equal(cpu_rng, torch.get_rng_state())
            or (cuda_rng is not None and not torch.equal(
                cuda_rng, torch.cuda.get_rng_state(source.base_xyz.device)))):
        raise AssertionError('No-op feedback cloning changed RNG state')
    if not _exact_equal(_motion_runtime_state(source), _motion_runtime_state(clone)):
        raise AssertionError('No-op feedback clone differs from complete source runtime state')
    return clone


def tracked_precision_rows(transaction_record, level, node):
    """Track feedback target rows, or the sampled parent when source was selected."""
    feedback = transaction_record.get('feedback')
    if feedback is None:
        return [dict(level=int(level), rows=[int(node)])]
    return copy.deepcopy(feedback['changed_rows']['precision'])


def transaction_feedback_mode(record):
    status = record.get('status')
    accepted = bool(record.get('accepted'))
    if status == 'committed_to_clone' and accepted:
        return 'deployed_feedback'
    if status in {'source_retained', 'rolled_back'} and not accepted:
        return 'source_noop_after_' + status
    raise ValueError('Transaction status and accepted flag disagree')


def initialize_feedback_motion(source, record, deployed_packet, device, storage):
    """Restore a committed packet or create an exact source clone for no-op cases."""
    mode = transaction_feedback_mode(record)
    if mode == 'deployed_feedback':
        capture = torch.load(deployed_packet, map_location='cpu', weights_only=False)
        feedback = restore_motion_capture(capture, device, storage)
    else:
        feedback = clone_source_noop(source, storage)
    return feedback, mode


def clear_saved_gradients(motion):
    norms = {}
    for name, parameter in motion.named_parameters():
        if parameter.grad is not None:
            norms[name] = float(parameter.grad.norm())
    motion.optimizer_zero_grad()
    if any(parameter.grad is not None for parameter in motion.parameters()):
        raise AssertionError('Continuation must clear every saved gradient')
    return norms


def _precision_margin(logits, choice):
    chosen = logits[choice]
    others = logits.clone()
    others[choice] = float('-inf')
    return float(chosen - others.max())


def target_state(motion, step, level, node, precision_rows):
    position = motion._refine_position(level, node)
    gates = motion.refinement_masks(step)
    precision = motion._precision_indices(step)
    rows = []
    for entry in precision_rows:
        depth = int(entry['level'])
        for row in entry['rows']:
            choice = int(precision[depth][row])
            logits = motion.precision_logits[depth][row].detach()
            rows.append(dict(level=depth, row=int(row), choice=choice,
                bits=int(motion.config.precision_bits[choice]),
                logit_margin=_precision_margin(logits, choice)))
    gate_logit = float(motion.refine_logits[level][position].detach())
    return dict(level=level, node=node, position=position,
        phase=motion.stage(step), hard_masks_frozen=motion.hard_masks is not None,
        gate_open=bool(gates[level][position]), gate_logit=gate_logit,
        gate_hard_margin=abs(gate_logit), precision=rows)


def training_update(motion, step, view, objective, reference_bytes):
    before_hard = motion.hard_masks is not None
    xyz, rotation = motion(step, training=True)
    phase = motion.stage(step)
    image_loss = objective(view, xyz, rotation)
    storage_weight = (motion.config.storage_loss_weight * motion.storage_scale(step)
                      if phase in {'gate_compress', 'gate_refine'} else 0.)
    storage = image_loss.new_zeros(())
    if storage_weight > 0:
        storage = sum(motion.storage_parts(step, hard=False).values())
    loss_storage = storage_weight * storage / reference_bytes
    loss = image_loss + loss_storage
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f'Nonfinite continuation loss at step {step}')
    loss.backward()
    motion.optimizer_step()
    after_hard = motion.hard_masks is not None
    return dict(step=step, camera=view.path.name, phase=phase,
        image_loss=float(image_loss.detach()), storage_proxy_bytes=float(storage.detach()),
        storage_weight=float(storage_weight), storage_loss=float(loss_storage.detach()),
        total_loss=float(loss.detach()), storage_proxy_differentiable=bool(storage.requires_grad),
        gate_precision_storage_frozen=phase != 'gate_compress',
        motion_optimizer_lrs=[float(group['lr']) for group in motion.motion_optimizer.param_groups],
        expected_refine_lr_scale=float(motion.config.refine_lr_scale),
        hardened_this_step=not before_hard and after_hard,
        hard_masks_frozen=after_hard)


def paired_training_update(source, feedback, feedback_mode, step, view,
                           objective, reference_bytes, storage):
    """Advance independent feedback, or mirror one source update for exact no-op."""
    source_row = training_update(source, step, view, objective, reference_bytes)
    if feedback_mode == 'deployed_feedback':
        feedback_row = training_update(feedback, step, view, objective, reference_bytes)
    else:
        feedback = clone_source_noop(source, storage)
        feedback_row = copy.deepcopy(source_row)
    return feedback, dict(source=source_row, feedback=feedback_row)


def evaluate_motion(motion, step, views, objective, diagnostic, reference_bytes):
    rows = []
    with torch.no_grad():
        xyz, rotation = motion(step)
        for view in views:
            metrics = objective.evaluate(view, xyz, rotation)
            rows.append(dict(camera=view.path.name,
                image_loss=float(metrics['image_loss']), psnr=float(metrics['psnr'])))
    cost = diagnostic.exact_cost(motion, step)
    image_loss = sum(row['image_loss'] for row in rows) / len(rows)
    rate_weight = motion.config.storage_loss_weight * motion.storage_scale(step)
    return dict(step=step, image_loss=image_loss,
        psnr=sum(row['psnr'] for row in rows) / len(rows), views=rows, cost=cost,
        rate_weight=rate_weight,
        actual_rd_score=image_loss + rate_weight * cost['total_bytes'] / reference_bytes)


def compare_evaluations(source, feedback):
    if source['step'] != feedback['step']:
        raise ValueError('Paired evaluations must use one step')
    return dict(step=source['step'],
        feedback_minus_source_image_loss=feedback['image_loss'] - source['image_loss'],
        feedback_minus_source_psnr=feedback['psnr'] - source['psnr'],
        feedback_minus_source_bytes=(feedback['cost']['total_bytes']
                                     - source['cost']['total_bytes']),
        feedback_rd_gain=source['actual_rd_score'] - feedback['actual_rd_score'])


def _load_validation_view(data, name, device):
    loader = data.test_dataloader()
    if loader is None:
        raise ValueError('Saved data configuration has no test loader for cam00')
    iterator = iter(loader)
    views = []
    try:
        for _ in range(len(loader.dataset)):
            views.append(next(iterator).to(device))
    finally:
        shutdown = getattr(iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()
    matches = [view for view in views if view.path.name == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one validation view {name}, found {len(matches)}')
    return matches[0]


def _snapshot_source_code(output, paths, digest):
    directory = output / 'source_snapshot'
    directory.mkdir()
    manifest = []
    for index, path in enumerate(paths):
        path = Path(path).resolve()
        payload = path.read_bytes()
        destination = directory / f'{index:02d}_{path.name}'
        destination.write_bytes(payload)
        manifest.append(dict(source=str(path), snapshot=str(destination),
            sha256=digest(path), bytes=len(payload)))
    (directory / 'manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def _append_progress(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


def run(protocol_path, output=None):
    from libgs.renderer import RendererConfig
    from libgs.utils.config import to_structured
    from scripts.diagnose_binary_gates import file_digest

    package = importlib.import_module('pipeline.ReCon-GS')
    diagnostic = importlib.import_module('pipeline.ReCon-GS.model.gate_diagnostics')
    snapshots = importlib.import_module('pipeline.ReCon-GS.model.gate_probe_snapshot')
    render = importlib.import_module('pipeline.ReCon-GS.gate_diagnostic_render')
    gaussian = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
    storage = importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary')
    binary = importlib.import_module('pipeline.ReCon-GS.model.binary_refinement')

    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    transaction_run = _repo_path(protocol['source_transaction_run']).resolve()
    transaction_summary_path = transaction_run / 'summary.json'
    transaction_summary = json.loads(transaction_summary_path.read_text(encoding='utf-8'))
    transaction_count = sum(transaction_summary.get(name, 0)
                            for name in ('committed', 'source_retained', 'rolled_back'))
    if (transaction_summary.get('schema') != 'joint_feedback_transaction_v1'
            or transaction_count != len(protocol['nodes'])):
        raise ValueError('Source feedback transaction is incomplete or unaudited')
    crossfold_run = Path(transaction_summary['source_crossfold_run']).resolve()
    crossfold_summary = json.loads((crossfold_run / 'summary.json').read_text(encoding='utf-8'))
    historical_run = Path(crossfold_summary['source_run']).resolve()
    if output is None:
        output = _repo_path(protocol['output_root']) / time.strftime('%Y%m%dT%H%M%S')
    output = Path(output).resolve()
    protected_roots = (transaction_run, crossfold_run, historical_run)
    if (output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output)
                               for root in protected_roots)):
        raise ValueError('Continuation output must be new and outside all read-only source runs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(
        protocol_path.read_text(encoding='utf-8'), encoding='utf-8')
    source_code = _snapshot_source_code(output, [Path(__file__), protocol_path,
        Path(diagnostic.__file__), Path(snapshots.__file__), Path(render.__file__),
        Path(storage.__file__), Path(binary.__file__)], file_digest)

    transaction_nodes = {(row['node']['frame'], row['node']['step'],
                          row['node']['level'], row['node']['node']): row
                         for row in transaction_summary['nodes']}
    immutable = [transaction_summary_path, crossfold_run / 'summary.json']
    started = time.perf_counter()
    results = []
    progress = output / 'progress.jsonl'

    for node_spec in protocol['nodes']:
        key = tuple(node_spec[name] for name in ('frame', 'step', 'level', 'node'))
        record = transaction_nodes.get(key)
        if record is None or record.get('status') not in {
                'committed_to_clone', 'source_retained', 'rolled_back'}:
            raise ValueError(f'Configured node has no complete feedback transaction: {key}')
        frame, start_step, level, node = key
        source_snapshot = Path(record['source_snapshot']).resolve()
        feedback_mode = transaction_feedback_mode(record)
        feedback_committed = feedback_mode == 'deployed_feedback'
        deployed_packet = (transaction_run / node_spec['deployed_packet']).resolve()
        if (deployed_packet.parent != transaction_run
                or (feedback_committed and not deployed_packet.is_file())):
            raise ValueError('Committed deployed_packet must be an existing direct child of transaction run')
        immutable.extend([source_snapshot,
            transaction_run / f'frame_{frame:03d}_step_{start_step:04d}_level_{level}_node_{node}.json'])
        if feedback_committed:
            immutable.append(deployed_packet)
        initial_hashes = {str(path): file_digest(path) for path in immutable}
        packet = torch.load(source_snapshot, map_location='cpu', weights_only=False)
        if (packet.get('context') != 'online_after_backward_before_optimizer_update'
                or packet['frame'] != frame or packet['step'] != start_step):
            raise ValueError('Source snapshot does not have the expected pre-update identity')
        if file_digest(source_snapshot) != record['source_snapshot_sha256']:
            raise ValueError('Historical source snapshot hash differs from feedback transaction')
        source = snapshots.restore_probe_motion(packet, protocol['device'])
        source_cleared = clear_saved_gradients(source)
        feedback, feedback_mode = initialize_feedback_motion(
            source, record, deployed_packet, protocol['device'], storage)
        if (source.last_step != start_step or feedback.last_step != start_step
                or source.hard_masks is not None or feedback.hard_masks is not None):
            raise ValueError('Continuation requires two matching pre-hardening states')
        if source.config != feedback.config or source.counts != feedback.counts:
            raise ValueError('Source and feedback arms have different motion configurations')
        if source.config.gs_activity_loss_weight != 0:
            raise ValueError('Controlled continuation does not implement a nonzero GS activity loss')
        feedback_cleared = clear_saved_gradients(feedback)

        data_config, unused = to_structured(packet['data_config'], package.DataConfig)
        if unused:
            raise ValueError(f'Unknown saved data fields: {unused}')
        renderer_config, unused = to_structured(packet['renderer_config'], RendererConfig)
        if unused:
            raise ValueError(f'Unknown saved renderer fields: {unused}')
        if str(Path(data_config.root).resolve()) != packet['dataset_root_resolved']:
            raise ValueError('Resolved dataset root differs from frozen snapshot dependency')
        state = packet['render_state']
        model = gaussian.GaussianModel(state['max_sh_degree'])
        model.active_sh_degree = state['active_sh_degree']
        model.render_attribute_cache = state['render_attribute_cache']
        for name, value in state['attributes'].items():
            setattr(model, name, torch.nn.Parameter(
                value.to(protocol['device']), requires_grad=False))
        objective = render.FrozenImageObjective(model, renderer_config,
            packet['background'].to(protocol['device']),
            packet['module_config']['lambda_dssim'], state['old_count'])

        with diagnostic.isolated_rng(protocol['device']):
            data = package.DataModule(data_config)
            data.setup()
            data.current_frame = frame
            pairs = data.selective_training_pairs(
                protocol['ordinary_camera_names'], protocol['device'], 0.)
            ordinary_views = [current for current, _ in pairs]
            by_name = {view.path.name: view for view in ordinary_views}
            if list(by_name) != protocol['ordinary_camera_names']:
                raise ValueError('Ordinary views did not load in the configured order')
            saved_names = [entry['camera'] for entry in packet['views']]
            snapshots.verify_snapshot_views(packet, [by_name[name] for name in saved_names])
            data._selective_pairs_cache = None
            del pairs
            validation_view = _load_validation_view(
                data, protocol['validation_camera_name'], protocol['device'])
            dependencies = dict(saved_snapshot_dependencies_verified=True,
                saved_views=packet['views'],
                added_ordinary_views=[snapshots.view_fingerprint(by_name[name])
                    for name in protocol['ordinary_camera_names'] if name not in saved_names],
                final_validation_view=snapshots.view_fingerprint(validation_view),
                dataset_root_resolved=packet['dataset_root_resolved'])

            schedule = balanced_continuation_schedule(protocol['ordinary_camera_names'],
                protocol['seed'], frame, start_step, protocol['end_step'],
                packet['training_camera'])
            expected_visits = (protocol['end_step'] - start_step) // len(by_name)
            future_counts = {name: sum(row['camera'] == name for row in schedule[1:])
                             for name in protocol['ordinary_camera_names']}
            if set(future_counts.values()) != {expected_visits}:
                raise AssertionError('Future continuation camera schedule is not balanced')
            reference_bytes = source.reference_storage_bytes(packet['first_frame_gs_count'])
            precision_rows = tracked_precision_rows(record, level, node)
            arms = dict(source=source, feedback=feedback)
            updates = {name: [] for name in arms}
            evaluations = []

            pre_source = evaluate_motion(source, start_step, ordinary_views,
                                         objective, diagnostic, reference_bytes)
            pre = dict(source=pre_source,
                feedback=(evaluate_motion(feedback, start_step, ordinary_views,
                                          objective, diagnostic, reference_bytes)
                          if feedback_committed else copy.deepcopy(pre_source)))
            source_state = target_state(source, start_step, level, node, precision_rows)
            pre_states = dict(source=source_state,
                feedback=(target_state(feedback, start_step, level, node, precision_rows)
                          if feedback_committed else copy.deepcopy(source_state)))
            evaluations.append(dict(timing='pre_update', arms=pre,
                                     comparison=compare_evaluations(pre['source'], pre['feedback'])))
            _append_progress(progress, dict(event='node_started', frame=frame,
                step=start_step, pre_feedback_rd_gain=evaluations[-1]['comparison']['feedback_rd_gain'],
                elapsed_seconds=time.perf_counter() - started))

            for schedule_row in schedule:
                step, camera = schedule_row['step'], schedule_row['camera']
                feedback, paired_updates = paired_training_update(
                    source, feedback, feedback_mode, step, by_name[camera],
                    objective, reference_bytes, storage)
                arms['feedback'] = feedback
                source_state = target_state(source, step, level, node, precision_rows)
                feedback_state = (target_state(feedback, step, level, node, precision_rows)
                                  if feedback_committed else copy.deepcopy(source_state))
                states = dict(source=source_state, feedback=feedback_state)
                for arm, update in paired_updates.items():
                    update['schedule_role'] = schedule_row['role']
                    update['target_state'] = states[arm]
                    updates[arm].append(update)
                if step in protocol['evaluation_steps']:
                    source_evaluation = evaluate_motion(
                        source, step, ordinary_views, objective, diagnostic, reference_bytes)
                    paired = dict(source=source_evaluation,
                        feedback=(evaluate_motion(feedback, step, ordinary_views,
                                                  objective, diagnostic, reference_bytes)
                                  if feedback_committed else copy.deepcopy(source_evaluation)))
                    evaluation = dict(timing='post_update', arms=paired,
                        comparison=compare_evaluations(paired['source'], paired['feedback']))
                    evaluations.append(evaluation)
                    _append_progress(progress, dict(event='checkpoint', frame=frame,
                        step=step, feedback_rd_gain=evaluation['comparison']['feedback_rd_gain'],
                        elapsed_seconds=time.perf_counter() - started))

            validation = {}
            with torch.no_grad():
                xyz, rotation = source(protocol['final_validation_step'])
                metrics = objective.evaluate(validation_view, xyz, rotation)
                validation['source'] = dict(camera=validation_view.path.name,
                    image_loss=float(metrics['image_loss']), psnr=float(metrics['psnr']))
                if feedback_committed:
                    xyz, rotation = feedback(protocol['final_validation_step'])
                    metrics = objective.evaluate(validation_view, xyz, rotation)
                    validation['feedback'] = dict(camera=validation_view.path.name,
                        image_loss=float(metrics['image_loss']), psnr=float(metrics['psnr']))
                else:
                    validation['feedback'] = copy.deepcopy(validation['source'])
            validation['comparison'] = dict(
                feedback_minus_source_image_loss=(validation['feedback']['image_loss']
                                                  - validation['source']['image_loss']),
                feedback_minus_source_psnr=(validation['feedback']['psnr']
                                            - validation['source']['psnr']))

        no_op_final_state_equal = (None if feedback_committed else _exact_equal(
            _motion_runtime_state(source), _motion_runtime_state(feedback)))
        if not feedback_committed and not no_op_final_state_equal:
            raise AssertionError('No-op feedback arm diverged from source continuation')

        if any(file_digest(Path(path)) != digest for path, digest in initial_hashes.items()):
            raise AssertionError('A read-only source artifact changed during continuation')
        result = dict(node={name: node_spec[name] for name in ('frame', 'step', 'level', 'node')},
            source_snapshot=str(source_snapshot), source_snapshot_sha256=file_digest(source_snapshot),
            transaction_status=record['status'], selected=record['selected'],
            feedback_mode=feedback_mode,
            deployed_packet=str(deployed_packet) if feedback_committed else None,
            deployed_packet_sha256=file_digest(deployed_packet) if feedback_committed else None,
            no_op_final_state_equal=no_op_final_state_equal,
            actual_compute_budget=dict(
                training_updates=len(schedule) * (2 if feedback_committed else 1),
                all20_evaluation_sets=len(evaluations) * (2 if feedback_committed else 1),
                final_validation_renders=2 if feedback_committed else 1,
                no_op_feedback_reuses_source_compute=not feedback_committed),
            snapshot_context=packet['context'], snapshot_training_camera=packet['training_camera'],
            semantics=(
                ('Both arms discard the saved current-step gradient and independently recompute it. '
                 if feedback_committed else
                 'The source discards the saved current-step gradient and recomputes it; the no-op '
                 'feedback record mirrors the complete post-update source state and metrics. ')
                + 'This is a controlled continuation through the original motion stages, '
                  'not an exact replay of the historical camera/RNG trajectory.'),
            cleared_saved_gradient_norms=dict(source=source_cleared, feedback=feedback_cleared),
            schedule=schedule, future_camera_counts=future_counts,
            reference_bytes=reference_bytes, dependencies=dependencies,
            target_rows=dict(gate=dict(level=level, node=node), precision=precision_rows),
            objective=dict(image='0.8 L1 + 0.2 DSSIM',
                storage_loss_weight=float(source.config.storage_loss_weight),
                gs_activity_loss_weight=float(source.config.gs_activity_loss_weight),
                gs_activity_omitted_because_source_weight_is_zero=True,
                keep_candidate_omitted_because_sparse_motion_path_disables_it=True),
            initial_target_state=pre_states, updates=updates, evaluations=evaluations,
            final_validation=validation,
            hardening=dict(source_steps=[row['step'] for row in updates['source']
                                         if row['hardened_this_step']],
                           feedback_steps=[row['step'] for row in updates['feedback']
                                           if row['hardened_this_step']]),
            immutable_sources_verified=True)
        results.append(result)
        result_path = output / f'frame_{frame:03d}_step_{start_step:04d}_continuation.json'
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        (output / 'summary.partial.json').write_text(json.dumps(
            dict(nodes=[dict(node=row['node'], evaluations=row['evaluations'],
                             final_validation=row['final_validation']) for row in results]),
            indent=2, allow_nan=False), encoding='utf-8')
        _append_progress(progress, dict(event='node_completed', frame=frame,
            step=start_step, result=str(result_path), elapsed_seconds=time.perf_counter() - started))

    summary = dict(schema='joint_feedback_controlled_continuation_v1',
        experiment_name=protocol['experiment_name'], config=str(protocol_path),
        source_transaction_run=str(transaction_run), source_crossfold_run=str(crossfold_run),
        historical_run=str(historical_run), source_code_snapshot=source_code,
        total_seconds=time.perf_counter() - started,
        protocol=dict(seed=protocol['seed'], end_step=protocol['end_step'],
            expected_nodes=len(protocol['nodes']),
            ordinary_camera_names=protocol['ordinary_camera_names'],
            evaluation_steps=protocol['evaluation_steps'],
            validation_camera_name=protocol['validation_camera_name'],
            current_step_policy='clear saved gradients and recompute snapshot training camera',
            future_view_sequence=protocol['future_view_sequence']),
        nodes=results, limitations=[
            'The future camera order is a controlled balanced schedule, not the historical training order.',
            'Only old-GS motion is continued; no commit, densification, newborn fitting, or full-frame claim is made.',
            'cam01..cam20 are training/selection views. cam00 is evaluated once at the end and never updates either arm.',
            'No-op feedback mirrors one source computation exactly; it is not a second independent GPU replay.',
            'These mechanism-debug nodes do not estimate population-level performance.',
        ])
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    _append_progress(progress, dict(event='completed',
        elapsed_seconds=summary['total_seconds']))
    print(f'Controlled continuation completed: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
