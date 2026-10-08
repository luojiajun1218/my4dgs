"""Controlled six-frame source/shadow continuation from saved step-120 states."""
import argparse, copy, importlib, json, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch, yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.continue_joint_feedback_pair import (_exact_equal, _load_validation_view,
    _motion_runtime_state, balanced_continuation_schedule, clear_saved_gradients,
    compare_evaluations, evaluate_motion, training_update)
from scripts.diagnose_binary_gates import file_digest


def load_protocol(path):
    p = yaml.safe_load(Path(path).read_text())
    expected = [f'cam{i:02d}.png' for i in range(1, 21)]
    if p['ordinary_camera_names'] != expected or p['validation_camera_name'] != 'cam00.png':
        raise ValueError('Camera protocol must be cam01..cam20 with cam00 final-only')
    if p['frames'] != [1, 5, 25, 50, 75, 99] or p['source_step'] != 120 or p['end_step'] != 200:
        raise ValueError('This controlled protocol is fixed to six step120..200 continuations')
    if not p['final_validation_only'] or p['main_camera_policy'] != 'saved_current_then_seeded_balanced_cycle':
        raise ValueError('Invalid validation or main camera policy')
    return p


def shadow_camera_schedule(names, seed, frame, steps):
    rng = np.random.default_rng(np.random.SeedSequence([seed, frame, 0x53484144]))
    sequence = []
    while len(sequence) < len(steps):
        sequence.extend(names[int(i)] for i in rng.permutation(len(names)))
    return dict(zip(steps, sequence[:len(steps)]))


def exact_named_parameters(before, motion, target_rows):
    changed_target = 0
    for name, value in motion.named_parameters():
        old = before[name]
        if name.startswith(('region_translation.', 'region_rotation.')):
            level = int(name.split('.')[1])
            changed = ((value.detach() != old).flatten(1).any(-1) if value.shape[0]
                       else torch.zeros(0, dtype=torch.bool, device=value.device))
            if bool((changed & ~target_rows[level]).any()):
                return False, changed_target
            changed_target += int((changed & target_rows[level]).sum())
        elif not torch.equal(value.detach(), old):
            return False, changed_target
    return True, changed_target


def mechanism_metrics(motion, step):
    gates = motion.refinement_masks(step)
    paths = motion._paths(gates)
    occupancy = motion._occupancy_values()
    precision = motion._precision_indices(step)
    levels = []
    for level in range(len(motion.counts)):
        occupied = occupancy[level].bool()
        reachable = paths[level].bool() & occupied
        histogram = {str(bits): int((reachable & (precision[level] == index)).sum())
                     for index, bits in enumerate(motion.config.precision_bits)}
        row = dict(level=level, occupied_motion_rows=int(occupied.sum()),
                   reachable_motion_rows=int(reachable.sum()),
                   precision_bit_histogram=histogram)
        if level < len(gates):
            internal = motion.active_layout['refine_rows'][level]
            visited = reachable[internal]
            row.update(occupied_refine_gates=int(occupied[internal].sum()),
                       reachable_open_gates=int((visited & gates[level].bool()).sum()),
                       reachable_closed_gates=int((visited & ~gates[level].bool()).sum()))
        levels.append(row)
    return dict(levels=levels)


def paired_mechanism_metrics(source, shadow, step):
    source_gates, shadow_gates = source.refinement_masks(step), shadow.refinement_masks(step)
    occupancy = source._occupancy_values()
    gate_differences = []
    for level, (left, right) in enumerate(zip(source_gates, shadow_gates)):
        internal = source.active_layout['refine_rows'][level]
        gate_differences.append(int(((left.bool() != right.bool()) &
                                     occupancy[level][internal].bool()).sum()))
    return dict(source=mechanism_metrics(source, step), shadow=mechanism_metrics(shadow, step),
                hard_gate_difference_by_level=gate_differences)


def hard_identities(motion, step):
    gates = motion.refinement_masks(step)
    paths = motion._paths(gates)
    occupancy = motion._occupancy_values()
    gate_rows = []
    for level, gate in enumerate(gates):
        internal = motion.active_layout['refine_rows'][level]
        occupied = occupancy[level][internal].bool()
        gate_rows.append(dict(
            level=level,
            open_positions=torch.where(occupied & gate.bool())[0].cpu().tolist(),
            open_nodes=internal[occupied & gate.bool()].cpu().tolist(),
            closed_positions=torch.where(occupied & ~gate.bool())[0].cpu().tolist(),
            closed_nodes=internal[occupied & ~gate.bool()].cpu().tolist()))
    return dict(gates=gate_rows, paths=[dict(level=level,
        reachable_rows=torch.where(path.bool() & occupancy[level].bool())[0].cpu().tolist())
        for level, path in enumerate(paths)])


def run(config_path, output=None):
    from libgs.renderer import RendererConfig
    from libgs.utils.config import to_structured
    package = importlib.import_module('pipeline.ReCon-GS')
    diagnostic = importlib.import_module('pipeline.ReCon-GS.model.gate_diagnostics')
    snapshots = importlib.import_module('pipeline.ReCon-GS.model.gate_probe_snapshot')
    render = importlib.import_module('pipeline.ReCon-GS.gate_diagnostic_render')
    gaussian = importlib.import_module('pipeline.ReCon-GS.model.gaussian')
    shadow_module = importlib.import_module('pipeline.ReCon-GS.model.shadow_exploration')
    protocol = load_protocol(config_path)
    source_run = (REPO / protocol['source_run']).resolve()
    if output is None:
        output = (REPO / protocol['output_root'] / time.strftime('%Y%m%dT%H%M%S')).resolve()
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(source_run) or source_run.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the source run')
    output.mkdir(parents=True)
    (output/'config.yaml').write_text(Path(config_path).read_text())
    source_files = [Path(__file__), Path(config_path), Path(shadow_module.__file__),
        Path(importlib.import_module('pipeline.ReCon-GS.model.storage_aware_binary').__file__),
        REPO/'scripts/continue_joint_feedback_pair.py', REPO/'scripts/diagnose_binary_gates.py',
        Path(diagnostic.__file__), Path(snapshots.__file__), Path(render.__file__),
        Path(gaussian.__file__), Path(package.__file__)]
    snap_dir = output/'source_snapshot'; snap_dir.mkdir()
    manifest=[]
    for i,p in enumerate(source_files):
        p=p.resolve(); dst=snap_dir/f'{i:02d}_{p.name}'; dst.write_bytes(p.read_bytes())
        manifest.append(dict(source=str(p), snapshot=str(dst), sha256=file_digest(p)))
    (snap_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    started=time.perf_counter(); results=[]
    for frame in protocol['frames']:
        source_snapshot=(source_run/'gate_diagnostics/snapshots'/f'frame_{frame:03d}_step_0120.pth').resolve()
        source_hash=file_digest(source_snapshot)
        packet=torch.load(source_snapshot,map_location='cpu',weights_only=False)
        if packet['frame'] != frame or packet['step'] != 120 or packet['context'] != 'online_after_backward_before_optimizer_update':
            raise ValueError('Unexpected source snapshot identity')
        source=snapshots.restore_probe_motion(packet,protocol['device'])
        clear_saved_gradients(source)
        shadow=copy.deepcopy(source)
        initial_exact=_exact_equal(_motion_runtime_state(source),_motion_runtime_state(shadow))
        if not initial_exact: raise AssertionError('Initial source/shadow runtime differs')
        scfg=shadow_module.ShadowExplorationConfig(**protocol['shadow'])
        explorer=shadow_module.ShadowMotionExplorer(shadow,scfg)
        data_config,unused=to_structured(packet['data_config'],package.DataConfig)
        renderer_config,unused2=to_structured(packet['renderer_config'],RendererConfig)
        if unused or unused2: raise ValueError(f'Unknown saved config: {unused} {unused2}')
        state=packet['render_state']; model=gaussian.GaussianModel(state['max_sh_degree'])
        model.active_sh_degree=state['active_sh_degree']; model.render_attribute_cache=state['render_attribute_cache']
        for name,value in state['attributes'].items():
            setattr(model,name,torch.nn.Parameter(value.to(protocol['device']),requires_grad=False))
        objective=render.FrozenImageObjective(model,renderer_config,packet['background'].to(protocol['device']),
                                               packet['module_config']['lambda_dssim'],state['old_count'])
        with diagnostic.isolated_rng(protocol['device']):
            data=package.DataModule(data_config); data.setup(); data.current_frame=frame
            pairs=data.selective_training_pairs(protocol['ordinary_camera_names'],protocol['device'],0.)
            views=[a for a,_ in pairs]; by_name={v.path.name:v for v in views}; data._selective_pairs_cache=None
            validation=_load_validation_view(data,'cam00.png',protocol['device'])
            schedule=balanced_continuation_schedule(protocol['ordinary_camera_names'],protocol['main_seed'],
                frame,120,200,packet['training_camera'])
            shadow_cameras=shadow_camera_schedule(protocol['ordinary_camera_names'],scfg.seed,frame,scfg.steps())
            reference_bytes=source.reference_storage_bytes(packet['first_frame_gs_count'])
            evaluations=[]; events=[]; source_updates=[]; shadow_updates=[]
            pre_s=evaluate_motion(source,120,views,objective,diagnostic,reference_bytes)
            pre_h=evaluate_motion(shadow,120,views,objective,diagnostic,reference_bytes)
            evaluations.append(dict(timing='pre_update',arms=dict(source=pre_s,shadow=pre_h),
                comparison=compare_evaluations(pre_s,pre_h),
                mechanism=paired_mechanism_metrics(source,shadow,120)))
            for row in schedule:
                step=row['step']; view=by_name[row['camera']]
                source_updates.append(training_update(source,step,view,objective,reference_bytes))
                if explorer.due(step):
                    shadow.prepare_step(step)
                    full_plan=explorer.full_plan(step)
                    plan=explorer.plan(step,frame_index=frame,full_plan=full_plan)
                    serial=explorer.serializable_plan(plan)
                    target_count=sum(len(x) for x in serial['target_rows'])
                    full_target_count=sum(int(mask.sum()) for mask in full_plan['target_rows'])
                    level=int(plan['gate_level'])
                    probabilities=shadow.gate_probabilities(step)[level].detach()
                    selected_probabilities=probabilities[plan['gate_positions']].cpu().tolist()
                    diagnostic_view=by_name[shadow_cameras[step]]
                    with torch.no_grad():
                        hard_xyz,hard_rotation=shadow(step)
                        hard_loss=objective(diagnostic_view,hard_xyz,hard_rotation)
                        full_xyz,full_rotation=explorer.pose(step,full_plan)
                        full_loss=objective(diagnostic_view,full_xyz,full_rotation)
                    event=dict(step=step,camera=shadow_cameras[step],plan=serial,
                        candidate_gate_count=len(serial['candidate_gate_nodes']),
                        selected_gate_count=len(serial['gate_nodes']),
                        selected_gate_probabilities=selected_probabilities,
                        selection_seed_components=[scfg.seed,frame,step,level,0x53484144],
                        full_target_row_count=full_target_count,
                        target_row_count=target_count,used_backward=False,
                        hard_image_loss=float(hard_loss),
                        full_frontier_image_loss=float(full_loss),
                        full_minus_hard_image_loss=float(full_loss-hard_loss),
                        diagnostic_no_grad_forwards=2)
                    if target_count:
                        before={n:p.detach().clone() for n,p in shadow.named_parameters()}
                        main_opt=(copy.deepcopy(shadow.motion_optimizer.state_dict()),
                                  copy.deepcopy(shadow.gate_optimizer.state_dict()))
                        before_pose=(hard_xyz.clone(),hard_rotation.clone())
                        xyz,rot=explorer.pose(step,plan)
                        loss=objective(diagnostic_view,xyz,rot)
                        grads=explorer.backward_and_step(step,loss,plan)
                        isolated,changed=exact_named_parameters(before,shadow,plan['target_rows'])
                        with torch.no_grad(): after_pose=shadow(step)
                        event.update(used_backward=True,image_loss=float(loss.detach()),
                            sampled_minus_hard_image_loss=float(loss.detach()-hard_loss),
                            sampled_minus_full_image_loss=float(loss.detach()-full_loss),
                            gradient_rows=grads,
                            parameter_isolation_exact=isolated,changed_target_rows=changed,
                            main_motion_optimizer_exact=_exact_equal(main_opt[0],shadow.motion_optimizer.state_dict()),
                            main_gate_optimizer_exact=_exact_equal(main_opt[1],shadow.gate_optimizer.state_dict()),
                            main_hard_pose_exact=all(torch.equal(a,b) for a,b in zip(before_pose,after_pose)))
                        if not all((event['parameter_isolation_exact'],event['main_motion_optimizer_exact'],
                                    event['main_gate_optimizer_exact'],event['main_hard_pose_exact'])):
                            raise AssertionError(f'Shadow isolation failed at frame {frame} step {step}')
                    events.append(event)
                shadow_updates.append(training_update(shadow,step,view,objective,reference_bytes))
                if step in protocol['evaluation_steps']:
                    se=evaluate_motion(source,step,views,objective,diagnostic,reference_bytes)
                    he=evaluate_motion(shadow,step,views,objective,diagnostic,reference_bytes)
                    evaluations.append(dict(timing='post_update',arms=dict(source=se,shadow=he),
                        comparison=compare_evaluations(se,he),
                        mechanism=paired_mechanism_metrics(source,shadow,step)))
            final_validation={}
            with torch.no_grad():
                for name,motion in [('source',source),('shadow',shadow)]:
                    xyz,rot=motion(200); metrics=objective.evaluate(validation,xyz,rot)
                    final_validation[name]=dict(camera='cam00.png',image_loss=float(metrics['image_loss']),psnr=float(metrics['psnr']))
            final_validation['comparison']=dict(shadow_minus_source_image_loss=final_validation['shadow']['image_loss']-final_validation['source']['image_loss'],
                shadow_minus_source_psnr=final_validation['shadow']['psnr']-final_validation['source']['psnr'])
        if file_digest(source_snapshot)!=source_hash: raise AssertionError('Source snapshot changed')
        final_paths=shadow._paths(shadow.refinement_masks(200))
        final_occupancy=shadow._occupancy_values()
        exposure_final=[]
        for level, exposures in enumerate(explorer.row_exposures):
            trained=exposures.to(final_paths[level].device) > 0
            reachable=final_paths[level].bool() & final_occupancy[level].bool()
            exposure_final.append(dict(level=level,ever_shadow_trained_rows=int(trained.sum()),
                trained_rows_finally_reachable=int((trained & reachable).sum()),
                trained_rows_finally_unreachable=int((trained & ~reachable).sum())))
        result=dict(frame=frame,source_step=120,end_step=200,source_snapshot=str(source_snapshot),
            source_snapshot_sha256=source_hash,initial_runtime_exact=initial_exact,
            main_schedule=schedule,shadow_camera_schedule=shadow_cameras,shadow_events=events,
            evaluations=evaluations,final_validation=final_validation,
            final_hard_identities=dict(source=hard_identities(source,200),
                                       shadow=hard_identities(shadow,200)),
            shadow_exposure_final=exposure_final,
            actual_compute_budget=dict(main_backward_per_arm=len(schedule),shadow_backward_used=sum(e['used_backward'] for e in events),
                shadow_events=len(events),diagnostic_no_grad_forwards=sum(e['diagnostic_no_grad_forwards'] for e in events),
                all20_evaluations_per_arm=len(evaluations),cam00_renders=2),
            semantics='Shadow runs before its arm main backward; main loss is freshly recomputed. Empty frontiers skip backward. Deeper hard gates are preserved.')
        (output/f'frame_{frame:03d}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        results.append(result)
        (output/'summary.partial.json').write_text(json.dumps(dict(schema='shadow_frontier_pair_v2',nodes=results),indent=2,allow_nan=False))
        print(json.dumps(dict(frame=frame,shadow_backward_used=result['actual_compute_budget']['shadow_backward_used'],
            final_rd_gain=evaluations[-1]['comparison']['feedback_rd_gain'],cam00_psnr=final_validation['comparison']['shadow_minus_source_psnr']),allow_nan=False),flush=True)
    summary=dict(schema='shadow_frontier_pair_v2',status='complete',experiment_name=protocol['experiment_name'],
        config=str(Path(config_path).resolve()),source_run=str(source_run),source_code=manifest,nodes=results,
        elapsed_seconds=time.perf_counter()-started)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (output/'exit_code.txt').write_text('0\n')
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output')
    args=ap.parse_args(); print(run(args.config,args.output))
