"""Audit analytic parent-to-child compensation with the production bit writer.

This is a CPU-only experiment over immutable gate-probe snapshots. It does not
train, render, or modify the production training/export path.
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


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_protocol(path):
    values = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    required = {
        'experiment_name', 'source_run', 'output_root', 'device', 'frames',
        'snapshot_steps', 'sample_stratum', 'expected_snapshot_count',
        'expected_node_count', 'expected_precision_bits', 'alternate_parent_bits',
        'compensation_scope', 'recursive_pushdown', 'candidate_modes', 'writer',
        'birth_bytes_in_delta', 'continuous_position_atol',
        'continuous_rotation_atol_rad', 'codec_position_atol',
        'codec_rotation_atol_rad',
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f'Protocol is missing keys: {missing}')
    if values['device'] != 'cpu':
        raise ValueError('This audit is CPU-only')
    if values['sample_stratum'] != 'reachable_open':
        raise ValueError('The audit must use the original reachable-open samples')
    if values['alternate_parent_bits'] != 'all_except_source':
        raise ValueError('Every non-source parent precision must be compared')
    if values['compensation_scope'] != 'occupied_immediate_children':
        raise ValueError('Only occupied immediate children may be compensated')
    if values['recursive_pushdown'] is not False:
        raise ValueError('Recursive compensation is outside this audit')
    if values['candidate_modes'] != ['naive', 'compensated']:
        raise ValueError('The paired candidate modes must be naive and compensated')
    if values['writer'] != 'storage_aware_binary_preorder_v1':
        raise ValueError('The production storage-aware writer is required')
    if values['birth_bytes_in_delta'] is not False:
        raise ValueError('Birth bytes stay fixed and must not enter the motion delta')
    if values['expected_snapshot_count'] != len(values['frames']) * len(values['snapshot_steps']):
        raise ValueError('Snapshot count does not match frames x steps')
    if values['expected_node_count'] != values['expected_snapshot_count'] * 3:
        raise ValueError('Expected nodes must retain three open samples per snapshot')
    if (len(set(values['frames'])) != len(values['frames']) or
            len(set(values['snapshot_steps'])) != len(values['snapshot_steps'])):
        raise ValueError('Frames and snapshot steps must be unique')
    for name in ('continuous_position_atol', 'continuous_rotation_atol_rad',
                 'codec_position_atol', 'codec_rotation_atol_rad'):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f'{name} must be positive')
    return values


def restore_frozen(packet, snapshots):
    motion = snapshots.restore_probe_motion(packet, 'cpu')
    snapshots_step = int(packet['step'])
    hard = importlib.import_module(
        'pipeline.ReCon-GS.archive.model.parent_precision_compensation').freeze_hard_choices(
            motion, snapshots_step)
    evaluation_step = int(motion.config.motion_steps)
    if motion.stage(evaluation_step) not in {'gate_refine', 'incremental'}:
        raise ValueError('Configured final motion step does not consume frozen hard choices')
    # Keep an explicit audit copy; no later probability/temperature call may
    # define the candidate's gate or precision selections.
    frozen = dict(refine=[value.clone() for value in hard['refine']],
                  precision=[value.clone() for value in hard['precision']])
    motion.hard_masks = frozen
    return motion, evaluation_step


def tensor_stats(values):
    values = values.detach().to(torch.float64).flatten()
    if not len(values):
        return dict(count=0, mean=0., rms=0., median=0., p95=0., maximum=0.)
    return dict(count=len(values), mean=float(values.mean()),
                rms=float(torch.sqrt(torch.mean(values.square()))),
                median=float(torch.quantile(values, .5)),
                p95=float(torch.quantile(values, .95)), maximum=float(values.max()))


def pose_error(reference, actual, parent_members, compensation):
    xyz_reference, rotation_reference = reference
    xyz_actual, rotation_actual = actual
    position = (xyz_actual.to(torch.float64) - xyz_reference.to(torch.float64)).norm(dim=-1)
    angle = compensation.quaternion_angle_error(rotation_actual, rotation_reference)
    changed_components = ((xyz_actual != xyz_reference).any(-1) |
                          (rotation_actual != rotation_reference).any(-1))
    geometric_changed = (position != 0) | (angle != 0)
    outside = ~parent_members
    return dict(position=tensor_stats(position), rotation_angle_rad=tensor_stats(angle),
                changed_component_rows=int(changed_components.sum()),
                changed_geometric_rows=int(geometric_changed.sum()),
                parent_member_rows=int(parent_members.sum()),
                changed_component_rows_outside_parent=int((changed_components & outside).sum()),
                changed_geometric_rows_outside_parent=int((geometric_changed & outside).sum()),
                outside_parent_position_max=float(position[outside].max()) if bool(outside.any()) else 0.,
                outside_parent_rotation_angle_max_rad=(float(angle[outside].max())
                                                        if bool(outside.any()) else 0.))


def roundtrip(motion, evaluation_step, path, frame, storage, protocol):
    forward = tuple(value.detach().clone() for value in motion(evaluation_step))
    report = motion.export_motion(
        path, frame_index=int(frame), previous_state_digest='00' * 32,
        post_motion_digest='00' * 32, previous_count=len(motion.base_xyz),
        deleted_count=0, birth_count=0)
    coding = {}
    decoded = storage.StorageAwareBinaryMotion.decode_motion(
        path, motion.base_xyz, motion.base_rotation, motion.config,
        motion.active_layout, coding_state=coding)
    decoded_pose = (decoded[0], decoded[1])
    position = (decoded_pose[0].to(torch.float64) - forward[0].to(torch.float64)).norm(dim=-1)
    compensation = importlib.import_module(
        'pipeline.ReCon-GS.archive.model.parent_precision_compensation')
    rotation = compensation.quaternion_angle_error(decoded_pose[1], forward[1])
    maximum_position = float(position.max())
    maximum_rotation = float(rotation.max())
    if maximum_position > protocol['codec_position_atol']:
        raise AssertionError('Production decode XYZ differs from candidate forward')
    if maximum_rotation > protocol['codec_rotation_atol_rad']:
        raise AssertionError('Production decode rotation differs from candidate forward')
    if report['archive_bytes'] != path.stat().st_size:
        raise AssertionError('Writer report disagrees with motion.bin size')
    header = storage.StorageAwareBinaryMotion.read_header(path)
    return decoded_pose, dict(archive_bytes=report['archive_bytes'],
        payload_bits=report['payload_bits'], payload_padding_bits=report['payload_padding_bits'],
        codec_header_bytes=report['codec_header_bytes'], saturated_motion_nodes=report['saturated_motion_nodes'],
        sha256=file_digest(path), decoded_forward_position_max=maximum_position,
        decoded_forward_rotation_angle_max_rad=maximum_rotation,
        header_payload_bits=header['payload_bits'])


def child_symbol_change(source, candidate, level, children, evaluation_step):
    source_precision, source_symbols, source_saturation = source._selected_symbols(evaluation_step)
    candidate_precision, candidate_symbols, candidate_saturation = candidate._selected_symbols(
        evaluation_step)
    child_level = level + 1
    source_present = source_symbols[child_level][children].ne(0).any(-1)
    candidate_present = candidate_symbols[child_level][children].ne(0).any(-1)
    return dict(child_nodes=list(map(int, children.tolist())),
        source_bits=[source.config.precision_bits[int(value)]
                     for value in source_precision[child_level][children]],
        candidate_bits=[candidate.config.precision_bits[int(value)]
                        for value in candidate_precision[child_level][children]],
        source_nonzero=int(source_present.sum()), candidate_nonzero=int(candidate_present.sum()),
        zero_to_nonzero=int((~source_present & candidate_present).sum()),
        nonzero_to_zero=int((source_present & ~candidate_present).sum()),
        changed_symbol_rows=int((source_symbols[child_level][children] !=
                                 candidate_symbols[child_level][children]).any(-1).sum()),
        source_saturated=int(source_saturation[child_level][children].sum()),
        candidate_saturated=int(candidate_saturation[child_level][children].sum()))


def snapshot_sources(output, protocol_path):
    sources = [
        Path(__file__).resolve(),
        REPO / 'pipeline/ReCon-GS/archive/model/parent_precision_compensation.py',
        REPO / 'pipeline/ReCon-GS/archive/model/storage_aware_binary.py',
        REPO / 'pipeline/ReCon-GS/archive/model/node_motion.py',
        REPO / 'pipeline/ReCon-GS/archive/model/binary_refinement.py',
        REPO / 'pipeline/ReCon-GS/archive/model/residual_motion.py',
        REPO / 'pipeline/ReCon-GS/archive/model/gate_probe_snapshot.py',
        REPO / 'tests/test_nested_prototype.py',
        Path(protocol_path).resolve(),
    ]
    directory = output / 'source_code'
    directory.mkdir()
    manifest = []
    for index, source in enumerate(sources):
        target = directory / f'{index:02d}_{source.name}'
        shutil.copy2(source, target)
        manifest.append(dict(source=str(source), path=str(target.relative_to(output)),
                             sha256=file_digest(target)))
    write_json(directory / 'manifest.json', manifest)
    return manifest


def summarize(rows):
    summary = {}
    for mode in ('naive', 'compensated'):
        pose = [row[mode]['pose_error'] for row in rows]
        byte_delta = [row[mode]['writer']['archive_bytes_delta'] for row in rows]
        summary[mode] = dict(candidates=len(rows),
            mean_position_error=float(sum(row['position']['mean'] for row in pose) / len(pose)),
            max_position_error=max(row['position']['maximum'] for row in pose),
            mean_rotation_angle_rad=float(
                sum(row['rotation_angle_rad']['mean'] for row in pose) / len(pose)),
            max_rotation_angle_rad=max(row['rotation_angle_rad']['maximum'] for row in pose),
            mean_motion_bin_bytes_delta=float(sum(byte_delta) / len(byte_delta)),
            min_motion_bin_bytes_delta=min(byte_delta), max_motion_bin_bytes_delta=max(byte_delta),
            changed_rows_outside_parent=sum(row['changed_geometric_rows_outside_parent']
                                            for row in pose))
    summary['paired'] = dict(
        compensated_lower_mean_position=sum(
            row['compensated']['pose_error']['position']['mean'] <
            row['naive']['pose_error']['position']['mean'] for row in rows),
        compensated_lower_mean_rotation=sum(
            row['compensated']['pose_error']['rotation_angle_rad']['mean'] <
            row['naive']['pose_error']['rotation_angle_rad']['mean'] for row in rows),
        compensated_same_motion_bin_bytes=sum(
            row['compensated']['writer']['archive_bytes'] ==
            row['naive']['writer']['archive_bytes'] for row in rows),
        child_zero_to_nonzero=sum(row['compensated']['children']['zero_to_nonzero'] for row in rows),
        child_nonzero_to_zero=sum(row['compensated']['children']['nonzero_to_zero'] for row in rows),
        compensated_child_saturation=sum(
            row['compensated']['children']['candidate_saturated'] for row in rows))
    return summary


def run(protocol_path, output=None):
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    source_run = repo_path(protocol['source_run']).resolve()
    if output is None:
        output = repo_path(protocol['output_root']) / time.strftime('%Y%m%dT%H%M%S')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    bins = output / 'motion_bins'
    bins.mkdir()
    source_manifest = snapshot_sources(output, protocol_path)

    snapshots = importlib.import_module('pipeline.ReCon-GS.archive.model.gate_probe_snapshot')
    storage = importlib.import_module('pipeline.ReCon-GS.archive.model.storage_aware_binary')
    compensation = importlib.import_module(
        'pipeline.ReCon-GS.archive.model.parent_precision_compensation')
    input_manifest, rows, sample_records = [], [], []
    snapshot_count = node_count = 0
    for frame in protocol['frames']:
        for snapshot_step in protocol['snapshot_steps']:
            stem = f'frame_{frame:03d}_step_{snapshot_step:04d}'
            snapshot_path = source_run / 'gate_diagnostics/snapshots' / f'{stem}.pth'
            plan_path = source_run / 'gate_diagnostics' / f'{stem}_online_pre_update.plan.json'
            if not snapshot_path.is_file() or not plan_path.is_file():
                raise FileNotFoundError(f'Missing immutable source for {stem}')
            packet = torch.load(snapshot_path, map_location='cpu', weights_only=False)
            plan = json.loads(plan_path.read_text(encoding='utf-8'))
            if packet['frame'] != frame or packet['step'] != snapshot_step:
                raise ValueError(f'Snapshot identity mismatch for {stem}')
            if packet['samples'] != plan['samples']:
                raise ValueError(f'Snapshot and plan samples differ for {stem}')
            open_samples = [row for row in plan['samples']
                            if row['stratum'] == protocol['sample_stratum']]
            if len(open_samples) != 3 or sorted(row['level'] for row in open_samples) != [0, 1, 2]:
                raise ValueError(f'{stem} no longer has one original open sample per gate level')
            source, evaluation_step = restore_frozen(packet, snapshots)
            if source.config.motion_compute != 'reference':
                raise ValueError('Candidate and decoder comparison requires the source reference path')
            if list(source.config.precision_bits) != protocol['expected_precision_bits']:
                raise ValueError('Source precision choices differ from the fixed protocol')
            source_bin = bins / f'{stem}_source.bin'
            source_pose, source_writer = roundtrip(
                source, evaluation_step, source_bin, frame, storage, protocol)
            source_precision, source_symbols, source_saturation = source._selected_symbols(
                evaluation_step)
            input_manifest.extend([
                dict(path=str(snapshot_path), sha256=file_digest(snapshot_path)),
                dict(path=str(plan_path), sha256=file_digest(plan_path)),
            ])
            snapshot_count += 1

            for sample in open_samples:
                level, node = int(sample['level']), int(sample['node'])
                gates = source.hard_masks['refine']
                paths = source._paths(gates)
                occupancy = source._occupancy_values()
                position = source._refine_position(level, node)
                if (not bool(paths[level][node]) or not bool(occupancy[level][node]) or
                        not bool(gates[level][position])):
                    raise ValueError(f'Original sample is not reachable+open: {stem} L{level} N{node}')
                structural_children, parent_members = compensation.immediate_child_coverage(
                    source.active_layout, level, node)
                occupied_children = structural_children[
                    occupancy[level + 1][structural_children]]
                original_index = int(source.hard_masks['precision'][level][node])
                original_bit = source.config.precision_bits[original_index]
                alternatives = [bit for bit in source.config.precision_bits if bit != original_bit]
                if len(alternatives) != 4:
                    raise ValueError('Each node must have exactly four alternate parent precisions')
                sample_records.append(dict(frame=frame, snapshot_step=snapshot_step,
                    level=level, node=node, stratum=sample['stratum'],
                    population=sample['population'], original_parent_bit=original_bit,
                    parent_members=int(parent_members.sum()),
                    structural_children=list(map(int, structural_children.tolist())),
                    occupied_children=list(map(int, occupied_children.tolist()))))
                node_count += 1

                for alternate_bit in alternatives:
                    alternate_index = source.config.precision_bits.index(alternate_bit)
                    tag = f'{stem}_level_{level}_node_{node}_parent_{alternate_bit}bit'

                    naive, naive_step = restore_frozen(packet, snapshots)
                    naive.hard_masks['precision'][level][node] = alternate_index
                    naive_bin = bins / f'{tag}_naive.bin'
                    naive_pose, naive_writer = roundtrip(
                        naive, naive_step, naive_bin, frame, storage, protocol)
                    naive_children = child_symbol_change(
                        source, naive, level, occupied_children, evaluation_step)
                    if naive_children['changed_symbol_rows']:
                        raise AssertionError('Naive parent-bit change modified child symbols')

                    rebased, rebased_step = restore_frozen(packet, snapshots)
                    rebase = compensation.rebase_immediate_children(
                        rebased, rebased_step, level, node, alternate_index)
                    if (float(rebase['continuous_position_error'].max()) >
                            protocol['continuous_position_atol']):
                        raise AssertionError('Continuous compensation did not preserve child displacement')
                    if (float(rebase['continuous_rotation_error'].max()) >
                            protocol['continuous_rotation_atol_rad']):
                        raise AssertionError('Continuous compensation did not preserve child rotation')
                    rebased_bin = bins / f'{tag}_compensated.bin'
                    rebased_pose, rebased_writer = roundtrip(
                        rebased, rebased_step, rebased_bin, frame, storage, protocol)
                    rebased_children = child_symbol_change(
                        source, rebased, level, occupied_children, evaluation_step)

                    for writer in (naive_writer, rebased_writer):
                        writer['archive_bytes_delta'] = (
                            writer['archive_bytes'] - source_writer['archive_bytes'])
                        writer['payload_bits_delta'] = (
                            writer['payload_bits'] - source_writer['payload_bits'])
                    row = dict(frame=frame, snapshot_step=snapshot_step,
                        evaluation_step=evaluation_step, level=level, node=node,
                        stratum=sample['stratum'], population=sample['population'],
                        original_parent_bit=original_bit, alternate_parent_bit=alternate_bit,
                        parent_members=int(parent_members.sum()),
                        structural_children=list(map(int, structural_children.tolist())),
                        occupied_children=list(map(int, occupied_children.tolist())),
                        source=dict(writer=source_writer,
                            parent_nonzero=bool(source_symbols[level][node].ne(0).any()),
                            parent_saturated=bool(source_saturation[level][node]),
                            pose_error=dict(position=tensor_stats(torch.zeros(len(source.base_xyz))),
                                rotation_angle_rad=tensor_stats(torch.zeros(len(source.base_xyz))),
                                changed_component_rows=0, changed_geometric_rows=0,
                                parent_member_rows=int(parent_members.sum()),
                                changed_component_rows_outside_parent=0,
                                changed_geometric_rows_outside_parent=0,
                                outside_parent_position_max=0.,
                                outside_parent_rotation_angle_max_rad=0.)),
                        naive=dict(writer=naive_writer,
                            pose_error=pose_error(source_pose, naive_pose, parent_members, compensation),
                            children=naive_children,
                            parent_saturated=bool(naive._selected_symbols(naive_step)[2][level][node])),
                        compensated=dict(writer=rebased_writer,
                            pose_error=pose_error(source_pose, rebased_pose, parent_members, compensation),
                            children=rebased_children,
                            parent_saturated=bool(rebased._selected_symbols(rebased_step)[2][level][node]),
                            continuous_pre_requantization=dict(
                                position_max=float(rebase['continuous_position_error'].max()),
                                rotation_angle_max_rad=float(
                                    rebase['continuous_rotation_error'].max()))))
                    rows.append(row)
                    write_json(output / f'{tag}.json', row)

    if snapshot_count != protocol['expected_snapshot_count']:
        raise AssertionError('Processed snapshot count differs from the protocol')
    if node_count != protocol['expected_node_count']:
        raise AssertionError('Processed node count differs from the protocol')
    expected_candidates = protocol['expected_node_count'] * 4
    if len(rows) != expected_candidates:
        raise AssertionError('Candidate pair count differs from 36 nodes x 4 alternate bits')

    result = dict(schema='parent_precision_compensation_codec_audit_v1',
        experiment_name=protocol['experiment_name'], protocol=str(protocol_path),
        source_run=str(source_run), device='cpu', training=False, rendering=False,
        snapshot_count=snapshot_count, node_count=node_count,
        alternate_comparisons=len(rows), candidate_motion_bins=len(rows) * 2,
        source_motion_bins=snapshot_count,
        hard_choice_semantics=('Snapshot gates and precision argmax are frozen at snapshot_step; '
            'all forward/writer/decode calls use config.motion_steps in a hard-mask-consuming phase.'),
        target_semantics=('All candidate errors use the source production-writer decoded pose. '
            'The continuous analytic identity is reported separately before child re-quantization.'),
        byte_semantics=('archive_bytes is the actual whole-tree production motion.bin including its '
            'fixed codec header. Birth bytes are held fixed and omitted from every delta.'),
        compensation_semantics=('Only occupied immediate children are rebased; deeper residuals, all '
            'gates, and all other precision choices remain unchanged. There is no recursive pushdown.'),
        samples=sample_records, summary=summarize(rows), comparisons=rows,
        input_manifest=input_manifest, source_code_manifest=source_manifest)
    write_json(output / 'summary.json', result)
    hashes = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'sha256.json':
            hashes.append(dict(path=str(path.relative_to(output)), sha256=file_digest(path),
                               bytes=path.stat().st_size))
    write_json(output / 'sha256.json', hashes)
    print(json.dumps(dict(output=str(output), snapshot_count=snapshot_count,
        node_count=node_count, comparisons=len(rows), summary=result['summary']), indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == '__main__':
    main()
