"""Measure independent no-feedback replay noise for accepted feedback nodes."""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import torch
import yaml


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def load_protocol(path):
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = {
        'experiment_name', 'source_transaction_run', 'output_root', 'device',
        'nodes', 'ordinary_camera_names', 'validation_camera_name', 'seed',
        'end_step', 'future_view_sequence', 'final_validation_step',
        'arm_order', 'rng_reset_between_arms',
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f'No-op stability config is missing: {missing}')
    expected_cameras = [f'cam{index:02d}.png' for index in range(1, 21)]
    if values['ordinary_camera_names'] != expected_cameras:
        raise ValueError('ordinary_camera_names must be cam01..cam20')
    if values['validation_camera_name'] != 'cam00.png':
        raise ValueError('validation_camera_name must be cam00.png')
    if values['future_view_sequence'] != 'seeded_balanced_cycle':
        raise ValueError('future_view_sequence must be seeded_balanced_cycle')
    if values['arm_order'] != ['arm_a', 'arm_b']:
        raise ValueError('arm_order must preserve the existing paired protocol')
    if values['rng_reset_between_arms'] is not False:
        raise ValueError('This audit must reproduce the existing no-reset arm order')
    if values['final_validation_step'] != values['end_step']:
        raise ValueError('Final validation must run at end_step')
    addresses = []
    for row in values['nodes']:
        address = tuple(row.get(name) for name in ('frame', 'step', 'level', 'node'))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in address):
            raise ValueError('Every node needs a nonnegative integer address')
        if row['step'] > values['end_step']:
            raise ValueError('Node step cannot exceed end_step')
        addresses.append(address)
    if not addresses or len(addresses) != len(set(addresses)):
        raise ValueError('nodes must be nonempty and unique')
    return values


def parameter_difference(left, right):
    left_parameters = dict(left.named_parameters())
    right_parameters = dict(right.named_parameters())
    if left_parameters.keys() != right_parameters.keys():
        raise AssertionError('Independent arms expose different parameter names')
    maximum = 0.0
    maximum_name = None
    maximum_changed = 0
    total_changed = 0
    for name, left_value in left_parameters.items():
        right_value = right_parameters[name]
        if (left_value.dtype != right_value.dtype
                or left_value.device != right_value.device
                or left_value.shape != right_value.shape):
            raise AssertionError(f'Parameter metadata differs for {name}')
        difference = (left_value.detach() - right_value.detach()).abs()
        changed = int(torch.count_nonzero(difference))
        current = float(difference.max()) if difference.numel() else 0.0
        total_changed += changed
        if current > maximum:
            maximum = current
            maximum_name = name
            maximum_changed = changed
    return dict(max_abs=maximum, max_parameter=maximum_name,
                max_parameter_changed_elements=maximum_changed,
                total_changed_elements=total_changed)


def rng_fingerprint(device):
    cpu = hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    cuda = None
    if torch.device(device).type == 'cuda':
        cuda = hashlib.sha256(
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
    return dict(cpu_sha256=cpu, cuda_sha256=cuda)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run(protocol_path, output=None):
    from libgs.renderer import RendererConfig
    from libgs.utils.config import to_structured
    from scripts.archive.continue_joint_feedback_pair import (
        _exact_equal,
        _load_validation_view,
        _motion_runtime_state,
        _snapshot_source_code,
        balanced_continuation_schedule,
        clear_saved_gradients,
        compare_evaluations,
        evaluate_motion,
        training_update,
    )
    from scripts.archive.diagnose_binary_gates import file_digest

    package = importlib.import_module('pipeline.ReCon-GS.archive')
    diagnostic = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_diagnostics')
    snapshots = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_probe_snapshot')
    render = importlib.import_module('pipeline.ReCon-GS.archive.gate_diagnostic_render')
    gaussian = importlib.import_module('pipeline.ReCon-GS.archive.model.gaussian')
    storage = importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary')
    binary = importlib.import_module('pipeline.ReCon-GS.archive.model.binary_refinement')

    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    transaction_run = _repo_path(protocol['source_transaction_run']).resolve()
    transaction_summary_path = transaction_run / 'summary.json'
    transaction_summary = json.loads(
        transaction_summary_path.read_text(encoding='utf-8'))
    if transaction_summary.get('schema') != 'joint_feedback_transaction_v1':
        raise ValueError('Source transaction has the wrong schema')
    transaction_nodes = {
        tuple(row['node'][name] for name in ('frame', 'step', 'level', 'node')): row
        for row in transaction_summary['nodes']
    }
    configured = {
        tuple(row[name] for name in ('frame', 'step', 'level', 'node'))
        for row in protocol['nodes']
    }
    accepted = {key for key, row in transaction_nodes.items()
                if row.get('accepted') and row.get('status') == 'committed_to_clone'}
    if configured != accepted:
        raise ValueError('Configured nodes must exactly equal the accepted transaction nodes')

    if output is None:
        output = _repo_path(protocol['output_root']) / time.strftime('%Y%m%dT%H%M%S')
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(transaction_run):
        raise ValueError('Audit output must be new and outside the source transaction')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(
        protocol_path.read_text(encoding='utf-8'), encoding='utf-8')
    source_code = _snapshot_source_code(output, [
        Path(__file__), protocol_path,
        REPO / 'scripts/archive/continue_joint_feedback_pair.py',
        Path(diagnostic.__file__), Path(snapshots.__file__), Path(render.__file__),
        Path(storage.__file__), Path(binary.__file__),
    ], file_digest)

    started = time.perf_counter()
    results = []
    source_hash = file_digest(transaction_summary_path)
    for node_spec in protocol['nodes']:
        key = tuple(node_spec[name] for name in ('frame', 'step', 'level', 'node'))
        record = transaction_nodes[key]
        frame, start_step, level, node = key
        source_snapshot = Path(record['source_snapshot']).resolve()
        source_snapshot_hash = file_digest(source_snapshot)
        if source_snapshot_hash != record['source_snapshot_sha256']:
            raise ValueError('Source snapshot hash differs from the transaction record')
        packet = torch.load(source_snapshot, map_location='cpu', weights_only=False)
        if (packet.get('context') != 'online_after_backward_before_optimizer_update'
                or packet['frame'] != frame or packet['step'] != start_step):
            raise ValueError('Source snapshot identity does not match the configured node')

        arm_a = snapshots.restore_probe_motion(packet, protocol['device'])
        cleared = clear_saved_gradients(arm_a)
        rng_before_clone = rng_fingerprint(protocol['device'])
        arm_b = copy.deepcopy(arm_a)
        rng_after_clone = rng_fingerprint(protocol['device'])
        initial_runtime_exact = _exact_equal(
            _motion_runtime_state(arm_a), _motion_runtime_state(arm_b))
        if not initial_runtime_exact or rng_before_clone != rng_after_clone:
            raise AssertionError('Independent arms did not start from one exact runtime state')
        initial_parameter_difference = parameter_difference(arm_a, arm_b)

        data_config, unused = to_structured(packet['data_config'], package.DataConfig)
        if unused:
            raise ValueError(f'Unknown saved data fields: {unused}')
        renderer_config, unused = to_structured(
            packet['renderer_config'], RendererConfig)
        if unused:
            raise ValueError(f'Unknown saved renderer fields: {unused}')
        state = packet['render_state']
        model = gaussian.GaussianModel(state['max_sh_degree'])
        model.active_sh_degree = state['active_sh_degree']
        model.render_attribute_cache = state['render_attribute_cache']
        for name, value in state['attributes'].items():
            setattr(model, name, torch.nn.Parameter(
                value.to(protocol['device']), requires_grad=False))
        objective = render.FrozenImageObjective(
            model, renderer_config, packet['background'].to(protocol['device']),
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
                raise ValueError('Ordinary cameras loaded in an unexpected order')
            data._selective_pairs_cache = None
            del pairs
            validation_view = _load_validation_view(
                data, protocol['validation_camera_name'], protocol['device'])
            schedule = balanced_continuation_schedule(
                protocol['ordinary_camera_names'], protocol['seed'], frame,
                start_step, protocol['end_step'], packet['training_camera'])
            reference_bytes = arm_a.reference_storage_bytes(packet['first_frame_gs_count'])
            updates = []
            for schedule_row in schedule:
                step = schedule_row['step']
                view = by_name[schedule_row['camera']]
                row_a = training_update(
                    arm_a, step, view, objective, reference_bytes)
                row_b = training_update(
                    arm_b, step, view, objective, reference_bytes)
                updates.append(dict(
                    step=step, camera=schedule_row['camera'],
                    role=schedule_row['role'],
                    arm_a_total_loss=row_a['total_loss'],
                    arm_b_total_loss=row_b['total_loss'],
                    arm_b_minus_a_total_loss=(row_b['total_loss']
                                              - row_a['total_loss']),
                    parameter_difference=parameter_difference(arm_a, arm_b),
                    arm_a_hardened_this_step=row_a['hardened_this_step'],
                    arm_b_hardened_this_step=row_b['hardened_this_step']))

            evaluation_a = evaluate_motion(
                arm_a, protocol['end_step'], ordinary_views, objective,
                diagnostic, reference_bytes)
            evaluation_b = evaluate_motion(
                arm_b, protocol['end_step'], ordinary_views, objective,
                diagnostic, reference_bytes)
            comparison = compare_evaluations(evaluation_a, evaluation_b)
            with torch.no_grad():
                validation = {}
                for name, motion in (('arm_a', arm_a), ('arm_b', arm_b)):
                    xyz, rotation = motion(protocol['final_validation_step'])
                    metrics = objective.evaluate(validation_view, xyz, rotation)
                    validation[name] = dict(
                        camera=validation_view.path.name,
                        image_loss=float(metrics['image_loss']),
                        psnr=float(metrics['psnr']))
                validation['comparison'] = dict(
                    arm_b_minus_a_image_loss=(validation['arm_b']['image_loss']
                                              - validation['arm_a']['image_loss']),
                    arm_b_minus_a_psnr=(validation['arm_b']['psnr']
                                        - validation['arm_a']['psnr']))

        if file_digest(source_snapshot) != source_snapshot_hash:
            raise AssertionError('Read-only source snapshot changed during audit')
        result = dict(
            node=dict(frame=frame, step=start_step, level=level, node=node),
            source_snapshot=str(source_snapshot),
            source_snapshot_sha256=source_snapshot_hash,
            selected_feedback=record['selected'],
            initial_runtime_exact=initial_runtime_exact,
            initial_rng_unchanged_by_deepcopy=rng_before_clone == rng_after_clone,
            initial_parameter_difference=initial_parameter_difference,
            cleared_saved_gradient_norms=cleared,
            schedule=schedule,
            updates=updates,
            final_parameter_difference=parameter_difference(arm_a, arm_b),
            final_all20=dict(arm_a=evaluation_a, arm_b=evaluation_b,
                             comparison=comparison),
            final_cam00=validation,
            immutable_source_verified=True,
        )
        results.append(result)
        _write_json(output / f'frame_{frame:03d}_step_{start_step:04d}_noop.json', result)
        _write_json(output / 'summary.partial.json', dict(nodes=results))

    if file_digest(transaction_summary_path) != source_hash:
        raise AssertionError('Read-only transaction summary changed during audit')
    summary = dict(
        schema='joint_feedback_noop_stability_v1',
        experiment_name=protocol['experiment_name'],
        config=str(protocol_path),
        source_transaction_run=str(transaction_run),
        source_transaction_summary_sha256=source_hash,
        source_code_snapshot=source_code,
        total_seconds=time.perf_counter() - started,
        protocol=dict(
            seed=protocol['seed'], end_step=protocol['end_step'],
            ordinary_camera_names=protocol['ordinary_camera_names'],
            validation_camera_name=protocol['validation_camera_name'],
            arm_order=protocol['arm_order'],
            rng_reset_between_arms=protocol['rng_reset_between_arms'],
            independent_updates=True, feedback_applied=False,
            mirrored_compute=False),
        nodes=results,
        limitations=[
            'One independent repeat per node estimates numerical scale but not a noise distribution.',
            'The fixed arm order matches the prior continuation and may include order-dependent GPU effects.',
            'cam01..cam20 are training views; cam00 is evaluated only after continuation.',
        ])
    _write_json(output / 'summary.json', summary)
    print(f'No-op stability audit completed: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
