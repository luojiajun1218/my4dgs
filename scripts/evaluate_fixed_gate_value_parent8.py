"""Render a fixed-state gate-value diagnostic for 36 original open nodes.

Each node uses five prescribed states without fitting: source REFINE, source-bit
STOP, 8-bit STOP, naive 8-bit REFINE, and compensated 8-bit REFINE.  Cam00 is
never loaded.  Costs come from the production storage-aware motion writer.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def load_protocol(path):
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = {
        'experiment_name', 'source_run', 'output_root', 'device', 'frames',
        'snapshot_steps', 'sample_stratum', 'expected_snapshot_count',
        'expected_node_count', 'expected_states', 'expected_view_renders',
        'camera_names', 'forbidden_camera_name', 'resolution', 'equal_view_weight',
        'no_grad', 'fit_steps', 'rate_weight', 'rate_reference', 'actual_cost',
        'birth_bytes_in_delta', 'target_parent_bits', 'allowed_source_parent_bits',
        'states', 'codec_position_atol', 'codec_rotation_atol_rad',
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f'Gate-value protocol is missing: {missing}')
    expected_cameras = [f'cam{index:02d}.png' for index in range(1, 21)]
    if values['camera_names'] != expected_cameras:
        raise ValueError('Views must be cam01..cam20 in order')
    if values['forbidden_camera_name'] != 'cam00.png':
        raise ValueError('cam00 must remain the forbidden evaluation view')
    if values['forbidden_camera_name'] in values['camera_names']:
        raise ValueError('cam00 cannot appear in the render pool')
    if values['sample_stratum'] != 'reachable_open':
        raise ValueError('This diagnostic uses only original reachable-open samples')
    if values['device'] != 'cuda' or values['resolution'] != -1:
        raise ValueError('The approved protocol requires CUDA and saved full resolution')
    if values['equal_view_weight'] is not True or values['no_grad'] is not True:
        raise ValueError('All views must be equally weighted under no_grad')
    if values['fit_steps'] != 0:
        raise ValueError('This diagnostic cannot fit motion')
    if values['rate_weight'] != .01:
        raise ValueError('The approved actual-byte weight is 0.01')
    if values['rate_reference'] != 'source_first_frame_gs_count_times_24_bytes':
        raise ValueError('The source reference byte convention changed')
    if values['actual_cost'] != 'storage_aware_binary_preorder_v1_whole_motion_bin':
        raise ValueError('The production whole-motion writer is required')
    if values['birth_bytes_in_delta'] is not False:
        raise ValueError('Birth bytes are fixed and omitted from candidate deltas')
    if values['target_parent_bits'] != 8:
        raise ValueError('This is the single approved 8-bit intervention')
    if values['allowed_source_parent_bits'] != [12, 32]:
        raise ValueError('Source parent bits must be explicitly restricted to 12 or 32')
    expected_states = {
        'A': {'gate': 'REFINE', 'parent_bits': 'source', 'compensation': False},
        'B': {'gate': 'STOP', 'parent_bits': 'source', 'compensation': False},
        'C': {'gate': 'STOP', 'parent_bits': 8, 'compensation': False},
        'D': {'gate': 'REFINE', 'parent_bits': 8, 'compensation': False},
        'E': {'gate': 'REFINE', 'parent_bits': 8,
              'compensation': 'immediate_children'},
    }
    if values['states'] != expected_states:
        raise ValueError('A/B/C/D/E semantics differ from the approved protocol')
    if values['expected_snapshot_count'] != len(values['frames']) * len(values['snapshot_steps']):
        raise ValueError('Snapshot count must equal frames x snapshot steps')
    if values['expected_node_count'] != values['expected_snapshot_count'] * 3:
        raise ValueError('There must be one original open sample per gate level')
    if values['expected_states'] != values['expected_snapshot_count'] + 4 * values['expected_node_count']:
        raise ValueError('State count must share A once per snapshot')
    if values['expected_view_renders'] != values['expected_states'] * len(values['camera_names']):
        raise ValueError('Render budget does not match states x cameras')
    return values


def restore_frozen(packet, device, snapshots, compensation):
    motion = snapshots.restore_probe_motion(packet, device)
    compensation.freeze_hard_choices(motion, int(packet['step']))
    evaluation_step = int(motion.config.motion_steps)
    if motion.stage(evaluation_step) not in {'gate_refine', 'incremental'}:
        raise ValueError('Evaluation step does not consume frozen hard choices')
    for parameter in motion.parameters():
        parameter.grad = None
    return motion, evaluation_step


def hard_choice_differences(source, candidate):
    gate = []
    precision = []
    for level, (left, right) in enumerate(zip(
            source['refine'], candidate.hard_masks['refine'])):
        gate.extend((level, int(row)) for row in torch.where(left != right)[0].tolist())
    for level, (left, right) in enumerate(zip(
            source['precision'], candidate.hard_masks['precision'])):
        precision.extend((level, int(row)) for row in torch.where(left != right)[0].tolist())
    return dict(gate=[list(value) for value in gate],
                precision=[list(value) for value in precision])


def validate_choice_differences(actual, *, level, node, position, gate_changed,
                                precision_changed):
    expected_gate = [[level, position]] if gate_changed else []
    expected_precision = [[level, node]] if precision_changed else []
    if actual['gate'] != expected_gate or actual['precision'] != expected_precision:
        raise AssertionError('Candidate changed an undeclared gate or precision choice')


def roundtrip_pose_and_cost(motion, evaluation_step, path, frame, storage,
                            diagnostic, compensation, protocol):
    with torch.no_grad():
        forward = tuple(value.detach().clone() for value in motion(evaluation_step))
    report = motion.export_motion(
        path, frame_index=int(frame), previous_state_digest='00' * 32,
        post_motion_digest='00' * 32, previous_count=len(motion.base_xyz),
        deleted_count=0, birth_count=0)
    exact = diagnostic.exact_cost(motion, evaluation_step)
    if exact['birth_bytes'] != 0 or exact['total_bytes'] != report['archive_bytes']:
        raise AssertionError('Exact diagnostic cost differs from the production motion.bin')
    decoded = storage.StorageAwareBinaryMotion.decode_motion(
        path, motion.base_xyz, motion.base_rotation, motion.config,
        motion.active_layout)
    pose = (decoded[0], decoded[1])
    position = (pose[0].to(torch.float64) - forward[0].to(torch.float64)).norm(dim=-1)
    angle = compensation.quaternion_angle_error(pose[1], forward[1])
    max_position = float(position.max())
    max_angle = float(angle.max())
    if max_position > protocol['codec_position_atol']:
        raise AssertionError('Decoded XYZ differs from the rendered candidate pose')
    if max_angle > protocol['codec_rotation_atol_rad']:
        raise AssertionError('Decoded rotation differs from the rendered candidate pose')
    return pose, dict(archive_bytes=int(report['archive_bytes']),
        payload_bits=int(report['payload_bits']),
        payload_padding_bits=int(report['payload_padding_bits']),
        codec_header_bytes=int(report['codec_header_bytes']),
        saturated_motion_nodes=int(report['saturated_motion_nodes']),
        sha256=file_digest(path), exact_components=exact['components'],
        decoded_forward_position_max=max_position,
        decoded_forward_rotation_angle_max_rad=max_angle)


def evaluate_state(motion, evaluation_step, views, objective, motion_path,
                   frame, reference_bytes, protocol, storage, diagnostic,
                   compensation, render_counter):
    pose, cost = roundtrip_pose_and_cost(
        motion, evaluation_step, motion_path, frame, storage,
        diagnostic, compensation, protocol)
    rows = []
    with torch.no_grad():
        for view in views:
            if view.path.name == protocol['forbidden_camera_name']:
                raise AssertionError('cam00 reached the fixed gate-value evaluator')
            metrics = objective.evaluate(view, *pose)
            rows.append(dict(camera=view.path.name,
                image_loss=float(metrics['image_loss']), psnr=float(metrics['psnr'])))
            render_counter[0] += 1
    if [row['camera'] for row in rows] != protocol['camera_names']:
        raise AssertionError('State views are not cam01..cam20 in configured order')
    rate_term = protocol['rate_weight'] * cost['archive_bytes'] / reference_bytes
    for row in rows:
        row['actual_rate_term'] = rate_term
        row['actual_rd_score'] = row['image_loss'] + rate_term
    return dict(image_loss=sum(row['image_loss'] for row in rows) / len(rows),
        psnr=sum(row['psnr'] for row in rows) / len(rows),
        actual_rate_term=rate_term,
        actual_rd_score=sum(row['actual_rd_score'] for row in rows) / len(rows),
        reference_bytes=reference_bytes, cost=cost, views=rows)


def sign(value):
    return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def compare_node(states):
    by_name = {name: {row['camera']: row for row in states[name]['views']}
               for name in states}
    rows = []
    for camera in [row['camera'] for row in states['A']['views']]:
        state = {name: by_name[name][camera] for name in states}
        gains = dict(
            original_bit_refine_gain=(state['B']['actual_rd_score'] -
                                      state['A']['actual_rd_score']),
            parent8_naive_refine_gain=(state['C']['actual_rd_score'] -
                                       state['D']['actual_rd_score']),
            parent8_compensated_refine_gain=(state['C']['actual_rd_score'] -
                                             state['E']['actual_rd_score']))
        row = dict(camera=camera,
            image_loss={name: state[name]['image_loss'] for name in states},
            actual_rd_score={name: state[name]['actual_rd_score'] for name in states},
            gate_opening_net_gain=gains,
            gate_opening_gain_sign={name: sign(value) for name, value in gains.items()},
            compensated_minus_naive_image_loss=(state['E']['image_loss'] -
                                                 state['D']['image_loss']),
            compensated_minus_naive_rd=(state['E']['actual_rd_score'] -
                                         state['D']['actual_rd_score']))
        row['compensated_minus_naive_image_loss_sign'] = sign(
            row['compensated_minus_naive_image_loss'])
        rows.append(row)
    gains = dict(
        original_bit_refine_gain=(states['B']['actual_rd_score'] -
                                  states['A']['actual_rd_score']),
        parent8_naive_refine_gain=(states['C']['actual_rd_score'] -
                                   states['D']['actual_rd_score']),
        parent8_compensated_refine_gain=(states['C']['actual_rd_score'] -
                                         states['E']['actual_rd_score']))
    return dict(gate_opening_net_gain=gains,
        gate_opening_gain_sign={name: sign(value) for name, value in gains.items()},
        compensated_minus_naive_image_loss=(states['E']['image_loss'] -
                                             states['D']['image_loss']),
        compensated_minus_naive_actual_bytes=(states['E']['cost']['archive_bytes'] -
                                               states['D']['cost']['archive_bytes']),
        compensated_minus_naive_rd=(states['E']['actual_rd_score'] -
                                     states['D']['actual_rd_score']),
        views=rows)


def scalar_summary(values):
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return dict(count=len(values), mean=float(tensor.mean()), median=float(tensor.median()),
                minimum=float(tensor.min()), maximum=float(tensor.max()),
                positive=int((tensor > 0).sum()), negative=int((tensor < 0).sum()),
                zero=int((tensor == 0).sum()))


def summarize(nodes):
    gain_names = ('original_bit_refine_gain', 'parent8_naive_refine_gain',
                  'parent8_compensated_refine_gain')
    node_gains = {name: scalar_summary([
        row['comparison']['gate_opening_net_gain'][name] for row in nodes])
        for name in gain_names}
    view_gains = {name: scalar_summary([
        view['gate_opening_net_gain'][name] for row in nodes
        for view in row['comparison']['views']]) for name in gain_names}
    by_level = {}
    for level in sorted(set(row['node']['level'] for row in nodes)):
        selected = [row for row in nodes if row['node']['level'] == level]
        by_level[str(level)] = {name: scalar_summary([
            row['comparison']['gate_opening_net_gain'][name] for row in selected])
            for name in gain_names}
    return dict(nodes=len(nodes), node_gate_opening_net_gain=node_gains,
        per_view_gate_opening_net_gain=view_gains, by_level=by_level,
        compensated_minus_naive_image_loss=scalar_summary([
            row['comparison']['compensated_minus_naive_image_loss'] for row in nodes]),
        compensated_minus_naive_actual_bytes=scalar_summary([
            row['comparison']['compensated_minus_naive_actual_bytes'] for row in nodes]),
        compensated_minus_naive_rd=scalar_summary([
            row['comparison']['compensated_minus_naive_rd'] for row in nodes]))


def snapshot_source_code(output, protocol_path, modules):
    paths = [Path(__file__).resolve(), Path(protocol_path).resolve(),
        Path(modules['compensation'].__file__).resolve(),
        Path(modules['storage'].__file__).resolve(),
        Path(modules['diagnostic'].__file__).resolve(),
        Path(modules['snapshots'].__file__).resolve(),
        Path(modules['render'].__file__).resolve(),
        Path(modules['binary'].__file__).resolve(),
        REPO / 'tests/test_nested_prototype.py']
    directory = output / 'source_code'
    directory.mkdir()
    manifest = []
    for index, source in enumerate(paths):
        target = directory / f'{index:02d}_{source.name}'
        shutil.copy2(source, target)
        manifest.append(dict(source=str(source), path=str(target.relative_to(output)),
                             sha256=file_digest(target), bytes=target.stat().st_size))
    write_json(directory / 'manifest.json', manifest)
    return manifest


def run(protocol_path, output=None):
    from libgs.renderer import RendererConfig
    from libgs.utils.config import to_structured

    package = importlib.import_module('pipeline.ReCon-GS')
    modules = {
        'compensation': importlib.import_module(
            'pipeline.ReCon-GS.model.parent_precision_compensation'),
        'storage': importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary'),
        'diagnostic': importlib.import_module('pipeline.ReCon-GS.model.gate_diagnostics'),
        'snapshots': importlib.import_module('pipeline.ReCon-GS.model.gate_probe_snapshot'),
        'render': importlib.import_module('pipeline.ReCon-GS.gate_diagnostic_render'),
        'binary': importlib.import_module('pipeline.ReCon-GS.model.binary_refinement'),
        'gaussian': importlib.import_module('pipeline.ReCon-GS.model.gaussian'),
    }
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    source_run = repo_path(protocol['source_run']).resolve()
    if output is None:
        output = repo_path(protocol['output_root']) / time.strftime('%Y%m%dT%H%M%S')
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source_run) or source_run.is_relative_to(output):
        raise ValueError('Output must be new and outside the immutable source run')
    output.mkdir(parents=True, exist_ok=False)
    bins = output / 'motion_bins'
    bins.mkdir()
    (output / 'config.yaml').write_text(
        protocol_path.read_text(encoding='utf-8'), encoding='utf-8')
    source_code = snapshot_source_code(output, protocol_path, modules)
    progress_path = output / 'progress.jsonl'
    started = time.perf_counter()
    render_counter = [0]
    state_count = 0
    snapshot_count = 0
    input_manifest = []
    nodes = []
    shared_a = []

    for frame in protocol['frames']:
        for snapshot_step in protocol['snapshot_steps']:
            stem = f'frame_{frame:03d}_step_{snapshot_step:04d}'
            snapshot_path = source_run / 'gate_diagnostics/snapshots' / f'{stem}.pth'
            plan_path = source_run / 'gate_diagnostics' / f'{stem}_online_pre_update.plan.json'
            snapshot_hash, plan_hash = file_digest(snapshot_path), file_digest(plan_path)
            packet = torch.load(snapshot_path, map_location='cpu', weights_only=False)
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            if packet['frame'] != frame or packet['step'] != snapshot_step:
                raise ValueError(f'Snapshot identity mismatch: {stem}')
            if packet['samples'] != plan['samples']:
                raise ValueError(f'Plan and snapshot samples differ: {stem}')
            samples = [row for row in plan['samples']
                       if row['stratum'] == protocol['sample_stratum']]
            if len(samples) != 3 or sorted(row['level'] for row in samples) != [0, 1, 2]:
                raise ValueError(f'{stem} does not retain three original open samples')
            if packet['data_config']['resolution'] != protocol['resolution']:
                raise ValueError('Snapshot resolution differs from the full-resolution protocol')
            saved_names = [row['camera'] for row in packet['views']]
            if protocol['forbidden_camera_name'] in saved_names:
                raise ValueError('Snapshot dependencies unexpectedly contain cam00')

            data_config, unused = to_structured(packet['data_config'], package.DataConfig)
            if unused:
                raise ValueError(f'Unknown saved data fields: {unused}')
            renderer_config, unused = to_structured(packet['renderer_config'], RendererConfig)
            if unused:
                raise ValueError(f'Unknown saved renderer fields: {unused}')
            state = packet['render_state']
            model = modules['gaussian'].GaussianModel(state['max_sh_degree'])
            model.active_sh_degree = state['active_sh_degree']
            model.render_attribute_cache = state['render_attribute_cache']
            for name, value in state['attributes'].items():
                setattr(model, name, torch.nn.Parameter(
                    value.to(protocol['device']), requires_grad=False))
            objective = modules['render'].FrozenImageObjective(
                model, renderer_config, packet['background'].to(protocol['device']),
                packet['module_config']['lambda_dssim'], state['old_count'])
            load_names = list(dict.fromkeys(saved_names + protocol['camera_names']))
            if protocol['forbidden_camera_name'] in load_names:
                raise AssertionError('cam00 entered the data loader request')

            with modules['diagnostic'].isolated_rng(protocol['device']):
                data = package.DataModule(data_config)
                data.setup()
                data.current_frame = frame
                loaded = [now for now, _ in data.selective_training_pairs(
                    load_names, protocol['device'], 0.)]
                by_name = {view.path.name: view for view in loaded}
                modules['snapshots'].verify_snapshot_views(
                    packet, [by_name[name] for name in saved_names])
                views = [by_name[name] for name in protocol['camera_names']]

                a_motion, evaluation_step = restore_frozen(
                    packet, protocol['device'], modules['snapshots'],
                    modules['compensation'])
                if a_motion.config.storage_loss_weight != protocol['rate_weight']:
                    raise ValueError('Source storage weight differs from lambda=0.01')
                reference_bytes = a_motion.reference_storage_bytes(
                    packet['first_frame_gs_count'])
                expected_reference = packet['first_frame_gs_count'] * 24
                if reference_bytes != expected_reference:
                    raise AssertionError('Source reference bytes changed')
                source_hard = dict(
                    refine=[value.detach().clone() for value in a_motion.hard_masks['refine']],
                    precision=[value.detach().clone() for value in a_motion.hard_masks['precision']])
                a_result = evaluate_state(
                    a_motion, evaluation_step, views, objective,
                    bins / f'{stem}_A_source_refine.bin', frame, reference_bytes,
                    protocol, modules['storage'], modules['diagnostic'],
                    modules['compensation'], render_counter)
                a_result['state'] = 'A'
                a_result['action'] = protocol['states']['A']
                shared_a.append(dict(frame=frame, snapshot_step=snapshot_step,
                                     evaluation_step=evaluation_step, result=a_result))
                state_count += 1
                del a_motion

                for sample in samples:
                    level, node = int(sample['level']), int(sample['node'])
                    source_motion, check_step = restore_frozen(
                        packet, protocol['device'], modules['snapshots'],
                        modules['compensation'])
                    if check_step != evaluation_step:
                        raise AssertionError('Independent clones use different evaluation steps')
                    position = source_motion._refine_position(level, node)
                    paths = source_motion._paths(source_motion.hard_masks['refine'])
                    occupancy = source_motion._occupancy_values()
                    if (not bool(paths[level][node]) or not bool(occupancy[level][node]) or
                            not bool(source_motion.hard_masks['refine'][level][position])):
                        raise ValueError('Configured sample is not source reachable+open')
                    original_index = int(source_motion.hard_masks['precision'][level][node])
                    original_bit = source_motion.config.precision_bits[original_index]
                    if (original_bit not in protocol['allowed_source_parent_bits'] or
                            original_bit == protocol['target_parent_bits']):
                        raise ValueError('Source parent bit is outside the approved 12/32 set')
                    target_index = source_motion.config.precision_bits.index(
                        protocol['target_parent_bits'])
                    structural_children, _ = modules['compensation'].immediate_child_coverage(
                        source_motion.active_layout, level, node)
                    occupied_children = structural_children[
                        occupancy[level + 1][structural_children]]
                    child_bits = [source_motion.config.precision_bits[int(value)] for value in
                        source_motion.hard_masks['precision'][level + 1][occupied_children]]
                    del source_motion

                    state_results = {'A': a_result}
                    state_diffs = {'A': dict(gate=[], precision=[])}
                    rebase_metadata = None
                    for name in ('B', 'C', 'D', 'E'):
                        motion, candidate_step = restore_frozen(
                            packet, protocol['device'], modules['snapshots'],
                            modules['compensation'])
                        before_parameters = ({parameter_name: value.detach().clone()
                            for parameter_name, value in motion.named_parameters()}
                            if name == 'E' else None)
                        action = protocol['states'][name]
                        opened = action['gate'] == 'REFINE'
                        precision_index = (None if action['parent_bits'] == 'source'
                                           else target_index)
                        if name == 'E':
                            modules['compensation'].set_frozen_node_action(
                                motion, level, node, opened=True)
                            rebase_metadata = modules['compensation'].rebase_immediate_children(
                                motion, candidate_step, level, node, target_index)
                        else:
                            modules['compensation'].set_frozen_node_action(
                                motion, level, node, opened=opened,
                                parent_precision_index=precision_index)
                        differences = hard_choice_differences(source_hard, motion)
                        validate_choice_differences(differences, level=level, node=node,
                            position=position, gate_changed=name in ('B', 'C'),
                            precision_changed=name in ('C', 'D', 'E'))
                        if name == 'E':
                            children = rebase_metadata['children']
                            for parameter_name, value in motion.named_parameters():
                                changed = (value.detach() != before_parameters[parameter_name])
                                if value.ndim:
                                    changed_rows = changed.reshape(len(value), -1).any(-1)
                                else:
                                    changed_rows = changed.reshape(1)
                                allowed = torch.zeros_like(changed_rows)
                                prefix = f'region_translation.{level + 1}'
                                rotation_prefix = f'region_rotation.{level + 1}'
                                if parameter_name in (prefix, rotation_prefix):
                                    allowed[children] = True
                                if bool((changed_rows & ~allowed).any()):
                                    raise AssertionError('Compensation changed a non-child parameter row')
                        motion_path = bins / (
                            f'{stem}_level_{level}_node_{node}_{name}.bin')
                        result = evaluate_state(
                            motion, candidate_step, views, objective, motion_path,
                            frame, reference_bytes, protocol, modules['storage'],
                            modules['diagnostic'], modules['compensation'], render_counter)
                        result['state'] = name
                        result['action'] = action
                        state_results[name] = result
                        state_diffs[name] = differences
                        state_count += 1
                        del motion

                    comparison = compare_node(state_results)
                    node_result = dict(node=dict(frame=frame, snapshot_step=snapshot_step,
                        evaluation_step=evaluation_step, level=level, node=node,
                        stratum=sample['stratum'], population=sample['population']),
                        original_parent_bit=original_bit,
                        target_parent_bit=protocol['target_parent_bits'],
                        structural_children=list(map(int, structural_children.tolist())),
                        occupied_children=list(map(int, occupied_children.tolist())),
                        occupied_child_precision_bits=child_bits,
                        hard_choice_differences=state_diffs,
                        compensation=(None if rebase_metadata is None else dict(
                            continuous_position_max=float(
                                rebase_metadata['continuous_position_error'].max()),
                            continuous_rotation_angle_max_rad=float(
                                rebase_metadata['continuous_rotation_error'].max()))),
                        states=state_results, comparison=comparison,
                        semantics=('Positive gate_opening_net_gain means the fixed REFINE state has '
                            'lower instantaneous J than its paired STOP state on the same view/state.'))
                    node_path = output / (
                        f'{stem}_level_{level}_node_{node}.json')
                    write_json(node_path, node_result)
                    nodes.append(node_result)
                    append_jsonl(progress_path, dict(event='node_completed',
                        frame=frame, snapshot_step=snapshot_step, level=level, node=node,
                        result=str(node_path), state_count=state_count,
                        view_renders=render_counter[0],
                        elapsed_seconds=time.perf_counter() - started))
                    write_json(output / 'summary.partial.json', dict(
                        completed_nodes=len(nodes), state_count=state_count,
                        view_renders=render_counter[0], nodes=nodes))

            if file_digest(snapshot_path) != snapshot_hash or file_digest(plan_path) != plan_hash:
                raise AssertionError('Immutable source changed during evaluation')
            input_manifest.extend([
                dict(path=str(snapshot_path), sha256=snapshot_hash),
                dict(path=str(plan_path), sha256=plan_hash),
            ])
            snapshot_count += 1
            del objective, model, data, loaded, by_name, views
            torch.cuda.empty_cache()

    if snapshot_count != protocol['expected_snapshot_count']:
        raise AssertionError('Snapshot count differs from the protocol')
    if len(nodes) != protocol['expected_node_count']:
        raise AssertionError('Node count differs from the protocol')
    if state_count != protocol['expected_states']:
        raise AssertionError('Independent state count differs from the protocol')
    if render_counter[0] != protocol['expected_view_renders']:
        raise AssertionError('Actual render count differs from the fixed budget')
    summary = dict(schema='fixed_gate_value_parent8_render_v1',
        experiment_name=protocol['experiment_name'], source_run=str(source_run),
        config=str(protocol_path), device=protocol['device'],
        training=False, fitting=False, cam00_loaded=False,
        snapshot_count=snapshot_count, node_count=len(nodes),
        state_count=state_count, view_renders=render_counter[0],
        rate_weight=protocol['rate_weight'],
        source_code_manifest=source_code, input_manifest=input_manifest,
        shared_source_refine_states=shared_a, summary=summarize(nodes), nodes=nodes,
        limitations=[
            'This is an instantaneous fixed-motion gate diagnostic, not a converged STOP-versus-REFINE comparison.',
            'No motion, gate, or precision parameter is fitted or written back.',
            'Per-view and mean signs are recorded without a statistical significance threshold.',
            'Cam01..cam20 are the complete equally weighted render pool; cam00 is never loaded.',
            'Birth bytes are candidate-constant and omitted from J; motion bytes use the actual whole-tree writer.',
            'Standalone motion headers use zero digest placeholders and are not a sequential playback stream.',
        ], total_seconds=time.perf_counter() - started)
    write_json(output / 'summary.json', summary)
    hashes = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'sha256.json':
            hashes.append(dict(path=str(path.relative_to(output)),
                               sha256=file_digest(path), bytes=path.stat().st_size))
    write_json(output / 'sha256.json', hashes)
    append_jsonl(progress_path, dict(event='completed', state_count=state_count,
        view_renders=render_counter[0], elapsed_seconds=summary['total_seconds']))
    print(json.dumps(dict(output=str(output), snapshot_count=snapshot_count,
        node_count=len(nodes), state_count=state_count,
        view_renders=render_counter[0], summary=summary['summary'],
        total_seconds=summary['total_seconds']), indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == '__main__':
    main()
