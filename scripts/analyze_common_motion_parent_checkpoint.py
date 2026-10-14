"""CPU-only proposal diagnosis; never writes a checkpoint or existing run.

This explicitly evaluates the unactivated helper outside its maintenance
schedule. It measures representation changes, not a training/quality benefit.
"""

import argparse
from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
from scipy.spatial.transform import Rotation
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name, directory in (('pipeline', ROOT / 'pipeline'),
                        ('pipeline.ReCon-GS', ROOT / 'pipeline' / 'ReCon-GS')):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules[name] = package
rigid = importlib.import_module('pipeline.ReCon-GS.model.rigid_residual_tree')
common = importlib.import_module('pipeline.ReCon-GS.model.common_motion_parent')
math_tree = importlib.import_module('pipeline.ReCon-GS.model.online_motion_tree')


def numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def world_pose(source, rotation, owner, q, b, gs_t, gs_r):
    sharing = Rotation.from_quat(q[owner][:, [1, 2, 3, 0]])
    xyz = sharing.apply(source) + b[owner] + gs_t
    orientation = Rotation.from_rotvec(gs_r) * sharing * Rotation.from_quat(rotation[:, [1, 2, 3, 0]])
    return xyz, orientation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh independent analysis output.')
    if args.checkpoint.resolve().parent in args.output.resolve().parents:
        raise ValueError('Analysis output must be outside the historical run.')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    packet = checkpoint['rigid_tree_motion']
    config = rigid.RigidTreeConfig()
    for key, value in packet['config'].items():
        existing = getattr(config, key)
        setattr(config, key, type(existing)(**value) if is_dataclass(existing) else value)
    model = rigid.RigidResidualTree(config)
    model.restore(packet)
    source, owner = numpy(model.base_xyz).astype(np.float64), numpy(model.owner)
    ids = numpy(model.persistent_state.region_ids)
    parents, pivots = numpy(model.parents), numpy(model.pivots).astype(np.float64)
    target_xyz = model.materialized_xyz[:len(owner)].clone()
    target_q = model.materialized_rotation[:len(owner)].clone()
    with torch.no_grad():
        world_q, world_b = model._global_regions()
    q, b = numpy(world_q).astype(np.float64), numpy(world_b).astype(np.float64)
    history = model.online_history.aligned(numpy(model.online_ids))
    observed = numpy(model.observations >= config.online_clustering.min_observations)
    cfg = common.CommonMotionParentConfig()
    original_interval = cfg.update_interval
    cfg.update_interval = 1  # Explicit diagnostic call; never changes active config.
    evaluations = []

    def trace_evaluation(frame, event, value):
        # Observe the unmodified helper's returns; this adds no admission gate
        # or alternate parameter setting to the actual CPU proposal.
        if frame.f_code.co_name == 'evaluate' and frame.f_code.co_filename == common.__file__:
            if event == 'return':
                local = frame.f_locals
                item = dict(child_slots=[int(slot) for slot in local['branches']],
                            returned_candidate=value is not None)
                for name in ('old_cost', 'new_cost', 'history_before', 'history_after', 'history_details'):
                    if name in local:
                        item[name] = local[name]
                if 'fit' in local and local['fit'] is not None:
                    item['virtual_rms'] = local['fit']['virtual_rms']
                if local.get('history_before', 0) and 'motion' in local:
                    motion = local['motion']
                    cq, cb = math_tree.local_to_world_motion(motion['parents'], motion['pivots'],
                        motion['t'], motion['r'])
                    cr = Rotation.from_quat(cq[owner][:, [1, 2, 3, 0]])
                    old_r = Rotation.from_quat(q[owner][:, [1, 2, 3, 0]])
                    error = cr.apply(source) + cb[owner] - old_r.apply(source) - b[owner]
                    item['candidate_prequant_pose_max_xyz'] = float(np.linalg.norm(error, axis=1).max())
                    item['candidate_prequant_pose_max_rotation_rad'] = float((cr * old_r.inv()).magnitude().max())
                evaluations.append(item)
            return trace_evaluation
        return None

    started = time.perf_counter()
    sys.settrace(trace_evaluation)
    try:
        proposal = common.propose_common_motion_parents(source, owner, ids, q, b,
            model.extent, numpy(model.steps[:3]).astype(np.float64),
            numpy(model.steps[3:]).astype(np.float64), cfg, parents=parents,
            pivots=pivots, observed=observed, history=history,
            next_region_id=model.persistent_state.next_region_id, frame_index=model.frame_index)
    finally:
        sys.settrace(None)
    elapsed = time.perf_counter() - started
    new_q, new_b = math_tree.local_to_world_motion(proposal.parents, proposal.pivots,
        proposal.local_translation, proposal.local_rotvec)
    args_pose = (source, numpy(model.base_rotation).astype(np.float64), owner)
    gs_t, gs_r = (numpy(packet['parameters'][key]).astype(np.float64)
                   for key in ('gs_translation', 'gs_rotation'))
    old_xyz, old_rotation = world_pose(*args_pose, q, b, gs_t, gs_r)
    new_xyz, new_rotation = world_pose(*args_pose, new_q, new_b, gs_t, gs_r)
    position_error = np.linalg.norm(new_xyz - old_xyz, axis=1)
    rotation_error = (new_rotation * old_rotation.inv()).magnitude()
    _, depth = math_tree.validate_motion_tree(proposal.parents, owner, cfg.max_depth)
    report = dict(checkpoint=str(args.checkpoint.resolve()),
        analysis_scope='CPU proposal only; no training, packet export or PSNR measurement',
        frame=int(model.frame_index), gs_count=len(owner), observed_gs=int(observed.sum()),
        current_world_scope='actual hard-quantized CPU _global_regions from final checkpoint',
        history_frames=history['frames'], helper_history_frames=proposal.metrics['history_frames'],
        history_support_by_frame=[int(history['observed'][:, i].sum()) for i in range(len(history['frames']))],
        history_membership_scope='current stable GS IDs and current leaf membership applied to decoded past observations',
        original_update_interval=original_interval,
        original_schedule_due=bool(model.frame_index == 1 or model.frame_index % original_interval == 0),
        diagnostic_schedule_override='evaluate current frame once; admission mathematics unchanged',
        helper_config=asdict(cfg), proposal_seconds=elapsed,
        regions_before=len(ids), regions_after=len(proposal.region_ids),
        added_parent_count=len(proposal.metrics['inserted_parents']), max_depth=int(depth.max()),
        region_ids_before=ids.tolist(), region_ids_after=proposal.region_ids.tolist(),
        parents_before=parents.tolist(), parents_after=proposal.parents.tolist(),
        original_slot_ids_preserved=bool(np.array_equal(proposal.region_ids[:len(ids)], ids)),
        internal_nodes_own_no_gs=True,
        prequantized_same_residual_pose_error=dict(max_xyz=float(position_error.max()),
            rms_xyz=float(np.sqrt(np.square(position_error).mean())),
            max_rotation_rad=float(rotation_error.max()),
            scope='float64 WORLD normalization and exact reparameterization; identical independent GS residuals'),
        regional_sparse_bytes_scope='raw regional sparse section only; excludes topology, ownership, GS residuals, compression and manifest',
        candidate_evaluations_observed=evaluations,
        helper_metrics=proposal.metrics)
    # Exercise the existing CPU installation/retarget method on this in-memory
    # restored copy; no optimizer, training or codec file is produced.
    model.parents = torch.as_tensor(proposal.parents, dtype=torch.long)
    model.pivots = torch.as_tensor(proposal.pivots, dtype=model.base_xyz.dtype)
    with torch.no_grad():
        valid = model._install_tree_motion_and_retarget(proposal.local_translation,
            proposal.local_rotvec, target_xyz, target_q)
        report['quantized_install_and_retarget_valid'] = bool(valid)
        if valid:
            decoded_xyz, decoded_q = model.forward(model.motion_steps)
            report['quantized_retarget_pose_error'] = dict(
                max_xyz=float((decoded_xyz - target_xyz).norm(dim=-1).max()),
                rms_xyz=float((decoded_xyz - target_xyz).square().sum(-1).mean().sqrt()),
                max_rotation_rad=float(rigid.quaternion_angle_error(decoded_q, target_q).max()),
                gs_translation_step=float(model.gs_steps[0]),
                gs_rotation_step=float(model.gs_steps[3]),
                scope='actual float32 CPU region quantization and existing GS residual retarget against final surviving-GS target')
            report['gs_integer_l1_before'] = sum(int(torch.round(packet['parameters'][key] /
                model.gs_steps[offset:offset + 3]).long().abs().sum())
                for key, offset in (('gs_translation', 0), ('gs_rotation', 3)))
            report['gs_integer_l1_after_retarget'] = sum(int(torch.round(value /
                model.gs_steps[offset:offset + 3]).long().abs().sum())
                for value, offset in ((model.gs_translation, 0), (model.gs_rotation, 3)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('regions_before', 'regions_after',
        'added_parent_count', 'prequantized_same_residual_pose_error',
        'quantized_install_and_retarget_valid', 'quantized_retarget_pose_error')}, indent=2))
    print(json.dumps({'before': proposal.metrics['before'], 'after': proposal.metrics['after'],
        'inserted_parents': proposal.metrics['inserted_parents'],
        'rejection_counts': proposal.metrics['rejection_counts']}, indent=2))
    print(f'Analysis saved: {args.output.resolve()}')


if __name__ == '__main__':
    main()
