"""Read-only audit of one trained motion tree; never runs an optimizer."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import sys
from types import ModuleType

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def load_numerical_modules(source):
    # Avoid importing the renderer or pipeline initializer during a CPU audit.
    for name, path in (('pipeline', source / 'pipeline'),
                       ('pipeline.ReCon-GS', source / 'pipeline/ReCon-GS')):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    modules = (importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree'),
               importlib.import_module('pipeline.ReCon-GS.model.rigid_tree_codec'))
    if any(not Path(module.__file__).resolve().is_relative_to(source.resolve()) for module in modules):
        raise ValueError('Cached numerical modules do not come from the requested source snapshot')
    return modules


def restore_motion(rigid, checkpoint):
    packet = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = packet['rigid_tree_motion']
    config = rigid.saved_config_with_default_motion_lr(saved['config'])
    classes = dict(clustering=rigid.MotionClusteringConfig,
                   hdbscan=rigid.ResidualHdbscanConfig,
                   gmm=rigid.ResidualGmmConfig, heir=rigid.HeirHierarchyConfig)
    for name, cls in classes.items():
        config[name] = cls(**config[name])
    model = rigid.RigidResidualTree(rigid.RigidTreeConfig(**config))
    model.restore(saved)
    assert asdict(model.config) == rigid.saved_config_with_default_motion_lr(saved['config'])
    return model


def linear_rotation_witness(rigid):
    """An additive rotation-vector residual need not be an SO(3) residual."""
    dtype = torch.float64
    parent = rigid.rotation_vector_quaternion(torch.tensor([[torch.pi / 2, 0., 0.]], dtype=dtype))
    relative = rigid.rotation_vector_quaternion(torch.tensor([[0., torch.pi / 2, 0.]], dtype=dtype))
    child = rigid.quaternion_multiply(relative, parent)
    difference = rigid.quaternion_rotvec(child) - rigid.quaternion_rotvec(parent)
    additive_recovery = rigid.quaternion_multiply(
        rigid.rotation_vector_quaternion(difference), parent)
    correct_relative = rigid.quaternion_multiply(child, rigid.quaternion_inverse(parent))
    correct_recovery = rigid.quaternion_multiply(correct_relative, parent)
    torch.testing.assert_close(correct_recovery, child, rtol=0, atol=1e-12)
    return dict(parent_rotation_vector=rigid.quaternion_rotvec(parent)[0].tolist(),
                child_rotation_vector=rigid.quaternion_rotvec(child)[0].tolist(),
                additive_rotation_error_rad=float(rigid.quaternion_angle_error(additive_recovery, child)),
                actual_relative_rotation_error_rad=float(rigid.quaternion_angle_error(correct_recovery, child)),
                scope='Large-angle mathematical witness, not measured sequence error. '
                      'The current finalizer computes the correct relative quaternion.')


def ownership_overlap(previous, current, old_regions, new_regions):
    """Match region IDs by overlap; a mere label permutation has zero churn."""
    previous = np.asarray(previous, dtype=np.int64)
    current = np.asarray(current, dtype=np.int64)
    if previous.shape != current.shape or not len(previous):
        raise ValueError('Ownership must cover the same surviving GS rows')
    matrix = np.bincount(previous * new_regions + current,
                         minlength=old_regions * new_regions).reshape(old_regions, new_regions)
    old_ids, new_ids = linear_sum_assignment(-matrix[1:, 1:])
    matched_moving = int(matrix[1:, 1:][old_ids, new_ids].sum())
    moving_to_moving = int(matrix[1:, 1:].sum())
    matched_all = int(matrix[0, 0]) + matched_moving
    mapped_all = int(matrix[0, 0]) + int(matrix[1:].max(axis=1).sum())
    return dict(old_regions=old_regions, new_regions=new_regions, old_gs=len(previous),
                root_to_motion=int(matrix[0, 1:].sum()),
                motion_to_root=int(matrix[1:, 0].sum()),
                moving_to_moving=moving_to_moving,
                matched_moving_fraction=(matched_moving / moving_to_moving if moving_to_moving else None),
                old_motion_group_retention=(int(matrix[1:, 1:].max(axis=1).sum()) / moving_to_moving
                                            if moving_to_moving else None),
                new_motion_group_purity=(int(matrix[1:, 1:].max(axis=0).sum()) / moving_to_moving
                                        if moving_to_moving else None),
                best_many_to_one_owner_edit_fraction=1 - mapped_all / len(previous),
                matched_all_fraction=matched_all / len(previous),
                unmatched_all_fraction=1 - matched_all / len(previous))


def owner_edit_section(codec, payload, previous):
    """Measure the existing dynamic packet's owner records before compression."""
    raw = codec.uncompressed_motion_packet(payload)
    reader = codec._Reader(raw[:-codec.CHECKSUM_BYTES])
    if bytes(reader.take(4)) != codec.DYNAMIC_MAGIC:
        raise ValueError('Owner audit requires the active dynamic layout format')
    reader.take(32)
    frame, regions, count, _ = [reader.integer() for _ in range(4)]
    reader.take(40)
    if reader.take(1)[0] != 0 or count != len(previous.xyz):
        raise ValueError('This audit expects the active no-deletion stream')
    mode = reader.take(1)[0]
    if mode == 1:
        for _ in range(len(previous.parents) + regions - 1):
            reader.integer()
        reader.take(12 * regions)
    elif mode != 0:
        raise ValueError('Unknown dynamic layout mode')
    started = reader.cursor
    count_edits = reader.integer()
    for _ in range(count_edits):
        reader.integer()
        reader.integer()
    size = reader.cursor - started
    return dict(frame=frame, actual_owner_edits=count_edits,
                actual_owner_edit_fraction=count_edits / count,
                owner_edit_section_raw_bytes=size,
                owner_edit_section_share_of_raw_packet=size / len(raw))


def reaches_cycle(parents):
    for node in range(1, len(parents)):
        seen = set()
        while node != 0:
            if node in seen:
                return True
            seen.add(node)
            node = int(parents[node])
    return False


@torch.no_grad()
def checkpoint_audit(rigid, path):
    model = restore_motion(rigid, path)
    xyz, rotation = model.forward(model.motion_steps)
    flat_t, flat_r = model._residual_for_target(xyz, rotation, torch.zeros_like(model.owner))
    rt, rr, gt, gr = model._values(hard=True)
    active = lambda value: value.ne(0).any(-1)
    flat_blocks = int(active(flat_t).sum() + active(flat_r).sum())
    shared_blocks = int(active(gt).sum() + active(gr).sum())
    _, step = path.stem.rsplit('-', 1)
    result = dict(frame=model.frame_index, checkpoint_step=int(step), old_gs=len(model.owner),
                  regions=len(model.parents),
                  flat_gs_translation_blocks=int(active(flat_t).sum()),
                  flat_gs_rotation_blocks=int(active(flat_r).sum()),
                  shared_gs_translation_blocks=int(active(gt).sum()),
                  shared_gs_rotation_blocks=int(active(gr).sum()),
                  saved_gs_payload_blocks=flat_blocks - shared_blocks,
                  region_translation_blocks=int(active(rt[1:]).sum()),
                  region_rotation_blocks=int(active(rr[1:]).sum()),
                  heir_feature_shape=(None if model.heir_features is None else list(model.heir_features.shape)),
                  heir_temporal_samples=(None if model.heir_features is None else
                                        1 if model.heir_features.ndim == 2 else model.heir_features.shape[0]))
    q, offset = model._global_regions()
    features = model._heir_motion_features()
    edges = []
    length = model.config.clustering.orientation_length_extent * model.extent
    for node in range(1, len(model.parents)):
        parent = int(model.parents[node])
        t, r = model._local_from_global(node, parent, model.pivots[node], q, offset)
        true_feature = torch.cat((t / (model.config.hdbscan.translation_scale_extent * model.extent),
                                  r / model.config.hdbscan.rotation_scale_rad))
        root_t, root_r = model._local_from_global(node, 0, model.pivots[node], q, offset)
        physical_cost = float(rigid.relative_motion_magnitude(t[None], r[None], model.extent))
        root_cost = float(rigid.relative_motion_magnitude(root_t[None], root_r[None], model.extent))
        proxy = features[node] - features[parent]
        proxy_q = rigid.rotation_vector_quaternion((proxy[3:] * model.config.hdbscan.rotation_scale_rad)[None])
        inherited = rigid.quaternion_multiply(proxy_q, q[parent:parent + 1])
        rows = torch.where(model.owner == node)[0]
        edge = dict(node=node, parent=parent, members=len(rows),
                    linear_feature_l1=float(proxy.abs().sum()),
                    linear_root_feature_l1=float(features[node].abs().sum()),
                    physical_local_feature_l1=float(true_feature.abs().sum()),
                    physical_regularizer_l1=physical_cost,
                    physical_root_regularizer_l1=root_cost,
                    parent_reduces_physical_regularizer=physical_cost < root_cost,
                    additive_rotation_error_rad=float(rigid.quaternion_angle_error(inherited, q[node:node + 1])))
        if len(rows):
            predicted_xyz = rigid.quaternion_rotate(q[node].expand(len(rows), -1), model.base_xyz[rows]) + offset[node]
            predicted_q = rigid.quaternion_multiply(q[node].expand(len(rows), -1), model.base_rotation[rows])
            weights = model.observations[rows].clamp_min(1).to(xyz)
            root_xyz_error = ((xyz[rows] - model.base_xyz[rows]).square().sum(-1) * weights).sum()
            root_q_error = ((length * rigid.quaternion_angle_error(rotation[rows], model.base_rotation[rows])).square() * weights).sum()
            shared_xyz_error = ((xyz[rows] - predicted_xyz).square().sum(-1) * weights).sum()
            shared_q_error = ((length * rigid.quaternion_angle_error(rotation[rows], predicted_q)).square() * weights).sum()
            root_error = root_xyz_error + root_q_error
            edge.update(shared_fit_reduction=(float(1 - (shared_xyz_error + shared_q_error) / root_error)
                                               if float(root_error) > 1e-20 else None),
                        position_fit_reduction=(float(1 - shared_xyz_error / root_xyz_error)
                                                if float(root_xyz_error) > 1e-20 else None),
                        rotation_fit_reduction=(float(1 - shared_q_error / root_q_error)
                                                if float(root_q_error) > 1e-20 else None))
        edges.append(edge)
    result['edges'] = edges
    if model.heir_hierarchy is not None and len(model.parents) > 1:
        weights = model.heir_hierarchy.parent_weights(features)
        raw = torch.cat((model.parents.new_tensor([-1]), weights[1:].argmax(-1)))
        projected = model.heir_hierarchy.infer_parents(features, model.config.parent_max_depth)
        # The current sampler feeds normalized w as logits, literally Eq. (4).
        scores = weights[1:].masked_fill(torch.eye(len(weights), dtype=torch.bool)[1:], -torch.inf)
        sampling_probabilities = scores.softmax(-1)
        result['hierarchy_attention'] = dict(
            parent_confidence_mean=float(weights[1:].amax(-1).mean()),
            sampling_max_probability_mean=float(sampling_probabilities.amax(-1).mean()),
            uniform_parent_probability=1 / (len(weights) - 1),
            argmax_has_cycle=reaches_cycle(raw.tolist()),
            projection_changed_edges=int((projected[1:] != raw[1:]).sum()),
            soft_relative_l1_sum=float((features - weights @ features)[1:].abs().sum()))
    return result


def stage_audit(run):
    snapshots = {}
    for path in sorted((run / 'tree_stats').glob('*.csv')):
        with path.open(newline='') as file:
            rows = list(csv.DictReader(file))
        snapshots[int(rows[0]['frame']), int(rows[0]['step'])] = dict(
            regions=len(rows), old_gs=sum(int(x['old_gs_count']) for x in rows),
            local_residual_rows=sum(int(x['old_gs_with_local_residual']) for x in rows))
    results = []
    for frame in sorted({frame for frame, _ in snapshots}):
        start, finish = snapshots[frame, 100], snapshots[frame, 200]
        assert start['old_gs'] == finish['old_gs']
        results.append(dict(frame=frame, before=start, after=finish,
                            local_residual_rows_delta=finish['local_residual_rows'] - start['local_residual_rows']))
    return results


def stream_audit(codec, run):
    stream = run / 'frame_storage/rigid_tree'
    with np.load(stream / 'reference_frame0.npz', allow_pickle=False) as reference:
        state = codec.DecodedState(*(torch.from_numpy(reference[name].copy())
            for name in ('xyz', 'rotation', 'owner', 'parents')), int(reference['frame_index']))
    results = []
    summaries = sorted((run / 'frame_summary').glob('*.json'))
    for path in summaries:
        report = json.loads(path.read_text())
        frame = report['frame']
        old = state
        state = codec.decode_motion((stream / f'motion_{frame:03d}.bin').read_bytes(), old)
        assert len(old.xyz) == report['old_count']
        assert len(state.xyz) == report['total_count']
        manifest = json.loads((stream / f'frame_{frame:03d}.json').read_text())
        assert state.digest().hex() == manifest['decoded_state_sha256']
        if frame > 1 and frame % 10 == 0:
            row = ownership_overlap(old.owner.numpy(), state.owner[:len(old.xyz)].numpy(),
                                    len(old.parents), len(state.parents))
            results.append(dict(**row, **owner_edit_section(codec,
                (stream / f'motion_{frame:03d}.bin').read_bytes(), old)))
        elif frame > 1:
            assert torch.equal(old.owner, state.owner[:len(old.xyz)])
        if frame % 50 == 0 or frame == 299:
            print(json.dumps(dict(decoded_frame=frame, gs=len(state.xyz))), flush=True)
    assert state.frame_index == 299 and len(summaries) == 299
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--source-snapshot', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='New audit directory outside the historical run')
    args = parser.parse_args()
    run, source, output = args.run.resolve(), args.source_snapshot.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(run) or output.is_relative_to(source):
        raise ValueError('Audit output must be new and outside historical inputs')
    manifest = json.loads((source / 'source_manifest.json').read_text())
    for name in ('rigid_residual_tree', 'adaptive_motion_tree', 'heir_motion_hierarchy', 'rigid_tree_codec'):
        relative = f'pipeline/ReCon-GS/model/{name}.py'
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == manifest['hashes'][relative]
    output.mkdir(parents=True)
    rigid, codec = load_numerical_modules(source)
    torch.manual_seed(0)
    checkpoints = []
    for path in sorted(run.glob('rigid-ckpt-*.pth'),
                       key=lambda path: tuple(int(x) for x in path.stem.split('-')[2:])):
        frame, step = (int(x) for x in path.stem.split('-')[2:])
        if step == (300 if frame == 1 or frame % 10 == 0 else 200):
            checkpoints.append(checkpoint_audit(rigid, path))
    stages = stage_audit(run)
    preliminary = dict(source=str(source), run=str(run), no_training=True,
        rotation_witness=linear_rotation_witness(rigid), checkpoints=checkpoints, refinement_stages=stages,
        caveats=['Linear rotation witness does not mean the current finalizer composes rotations incorrectly.',
                 'Flat and shared GS payload counts use the GS grid tolerance, not bit-exact identical poses.',
                 'Fit reductions describe estimated model motion, not scene motion ground truth.',
                 'Stage 100->200 includes motion optimization and tree finalization, not an isolated tree effect.'])
    (output / 'checkpoint_audit.json').write_text(json.dumps(preliminary, indent=2))
    transitions = stream_audit(codec, run)
    preliminary['region_transitions'] = transitions
    preliminary['summary'] = dict(
        audited_dynamic_frames=299, region_rebuilds_after_initialization=len(transitions),
        heir_temporal_samples=sorted({x['heir_temporal_samples'] for x in checkpoints
                                     if x['heir_temporal_samples'] is not None}),
        matched_moving_fraction_median=statistics.median(x['matched_moving_fraction'] for x in transitions
                                                        if x['matched_moving_fraction'] is not None),
        unmatched_all_fraction_median=statistics.median(x['unmatched_all_fraction'] for x in transitions),
        refinement_rows_increase_frames=sum(x['local_residual_rows_delta'] > 0 for x in stages),
        refinement_rows_decrease_frames=sum(x['local_residual_rows_delta'] < 0 for x in stages),
        refinement_local_residual_rows_delta_mean=statistics.fmean(x['local_residual_rows_delta'] for x in stages))
    (output / 'audit.json').write_text(json.dumps(preliminary, indent=2))
    with (output / 'region_transitions.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(transitions[0]))
        writer.writeheader()
        writer.writerows(transitions)
    print(json.dumps(preliminary['summary'], indent=2))


if __name__ == '__main__':
    main()
