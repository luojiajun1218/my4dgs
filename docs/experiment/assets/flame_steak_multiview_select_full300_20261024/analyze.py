"""Recompute completed multiview-select sequence statistics from local artifacts."""
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUNS = {
    'previous': 'output/n3dv/flame_steak_1020/130k_multiview_mse_180_80_40/20261024T002601_88a2fc',
    'current': 'output/n3dv/flame_steak_1020/130k_multiview_select_ssim_180_80_40/20261024T095641_3b0b28',
}


def mean(values):
    return float(np.mean(values)) if len(values) else None


def byte_size(path):
    return path.stat().st_size if path.exists() else 0


def analyze(relative):
    run = ROOT / relative
    summary = json.loads((run / 'run_summary.json').read_text())
    reference = json.loads((run / 'reference_evaluation.json').read_text())
    paths = sorted((run / 'frame_summary').glob('frame_*.json'))
    packets = run / 'frame_storage/rigid_tree'
    frames, accept_kinds, reasons = [], Counter(), Counter()
    checks = dict(frame_numbers=True, stage_count=True, quality_matches_stage=True,
                  heldout_diagnostic_only=True, test_camera=True,
                  count_identity=True, count_tolerance=True,
                  protected_previous_births=True, encoded_payload=True,
                  reference_identity=True, fixed_count_during_refit=True)
    previous_count = reference['total_count']
    for expected, path in enumerate(paths, start=1):
        source = json.loads(path.read_text())
        frame = source['frame']
        checks['frame_numbers'] &= frame == expected
        stages = {stage['step']: stage for stage in source['stage_image_metrics']}
        checks['stage_count'] &= set(stages) == {180, 260, 300}
        checks['quality_matches_stage'] &= abs(source['evaluation']['test']['psnr'] - stages[300]['psnr']) < 1e-6
        checks['heldout_diagnostic_only'] &= all(not stage['participates_in_training_or_selection'] for stage in stages.values())
        checks['test_camera'] &= all(Path(stage['camera']).name == 'cam00.png' for stage in stages.values())
        checks['fixed_count_during_refit'] &= stages[260]['gaussian_count'] == stages[300]['gaussian_count'] == source['total_count']
        born, retired = source['birth_count'], source['retired_count']
        checks['count_identity'] &= source['total_count'] == previous_count + born - retired
        previous_count = source['total_count']
        checks['count_tolerance'] &= (math.ceil(.9 * born - 1e-9) <= retired <= math.floor(1.1 * born + 1e-9))
        control = source['birth_control']
        retirement = control['retirement']
        checks['protected_previous_births'] &= retirement['retired_previous_birth_count'] == 0
        admission = control['birth_admission']
        accepted = admission['accepted']
        base = admission['baseline']
        trials = admission['trial_metrics']
        best = max(trials, key=lambda trial: trial['psnr']) if trials else base
        selected_index = admission.get('selected_trial')
        selected = trials[selected_index] if selected_index is not None else None
        utility = admission.get('newborn_utility', {})
        candidates = retirement['candidate_count']
        positive = utility.get('positive_count')
        kind = None
        if accepted:
            if born == candidates:
                kind = 'full_batch'
            elif born == positive:
                kind = 'positive_all'
            elif positive is not None and born == math.ceil(positive * .5):
                kind = 'positive_top_half'
            elif positive is not None and born == math.ceil(positive * .25):
                kind = 'positive_top_quarter'
            else:
                kind = 'other_subset'
            accept_kinds[kind] += 1
        else:
            if any(trial['psnr'] - base['psnr'] >= admission['min_gain_db'] for trial in trials):
                reasons['psnr_pass_but_ssim_rejected'] += 1
            else:
                reasons['no_trial_passed_psnr'] += 1
        payload = sum(byte_size(packets / f'{prefix}_{frame:03d}.bin')
                      for prefix in ('motion', 'birth', 'opacity'))
        index_bytes = (packets / f'frame_{frame:03d}.json').stat().st_size
        checks['encoded_payload'] &= payload == source['encoded_bytes']
        checks['reference_identity'] &= source['shared_reference_bytes'] == summary['storage']['reference_mb'] * 1e6
        frames.append(dict(
            frame=frame, motion_psnr=stages[180]['psnr'], selected_psnr=stages[260]['psnr'],
            final_psnr=stages[300]['psnr'], birth_selection_gain=stages[260]['psnr'] - stages[180]['psnr'],
            refit_gain=stages[300]['psnr'] - stages[260]['psnr'],
            birth_net_gain=stages[300]['psnr'] - stages[180]['psnr'],
            ssim=source['evaluation']['test']['ssim'], lpips=source['evaluation']['test']['lpips'],
            born=born, retired=retired, total=source['total_count'], candidates=candidates,
            accepted=accepted, accepted_kind=kind, positive_utility_count=positive,
            validation_best_psnr_gain=best['psnr'] - base['psnr'],
            validation_best_ssim_change=best['ssim'] - base['ssim'],
            validation_selected_psnr_gain=selected['psnr'] - base['psnr'] if selected else None,
            validation_selected_ssim_change=selected['ssim'] - base['ssim'] if selected else None,
            trial_count=len(trials), payload_bytes=payload, dynamic_update_bytes=payload+index_bytes,
            motion_bytes=byte_size(packets / f'motion_{frame:03d}.bin'),
            birth_bytes=byte_size(packets / f'birth_{frame:03d}.bin'),
            opacity_bytes=byte_size(packets / f'opacity_{frame:03d}.bin'),
            training_seconds=source['training_seconds'], frame_wall_seconds=source['frame_wall_seconds'],
            phase_seconds=source['timing']['phases'],
            prune_reasons=control.get('prune_reasons', {}),
        ))
    accepted = [frame for frame in frames if frame['accepted']]
    all_quality = [reference['evaluation']['test']['psnr']] + [frame['final_psnr'] for frame in frames]
    checks['quality_recomputed'] = abs(mean(all_quality) - summary['quality']['test']['metrics']['psnr']) < 1e-9
    checks['dynamic_storage_recomputed'] = abs(mean([frame['dynamic_update_bytes'] for frame in frames])/1000 - summary['storage']['complete_update_kb_mean']) < 1e-9
    checks['sequence_complete'] = summary['sequence_complete'] and len(frames) == 299
    stats = dict(
        dynamic_frames=len(frames), accepted=len(accepted), rejected=len(frames)-len(accepted),
        accepted_fraction=len(accepted)/len(frames), accept_kinds=dict(accept_kinds), rejected_reasons=dict(reasons),
        motion_psnr_mean=mean([frame['motion_psnr'] for frame in frames]),
        selected_psnr_mean=mean([frame['selected_psnr'] for frame in frames]),
        final_dynamic_psnr_mean=mean([frame['final_psnr'] for frame in frames]),
        birth_selection_gain_mean=mean([frame['birth_selection_gain'] for frame in frames]),
        refit_gain_mean=mean([frame['refit_gain'] for frame in frames]),
        birth_net_gain_mean=mean([frame['birth_net_gain'] for frame in frames]),
        accepted_net_gain_mean=mean([frame['birth_net_gain'] for frame in accepted]),
        accepted_test_positive=sum(frame['birth_net_gain']>1e-6 for frame in accepted),
        accepted_test_negative=sum(frame['birth_net_gain']<-1e-6 for frame in accepted),
        accepted_test_equal=sum(abs(frame['birth_net_gain'])<=1e-6 for frame in accepted),
        accepted_selection_test_negative=sum(frame['birth_selection_gain']<-1e-6 for frame in accepted),
        accepted_refit_test_negative=sum(frame['refit_gain']<-1e-6 for frame in accepted),
        accepted_validation_psnr_gain_mean=mean([frame['validation_selected_psnr_gain'] for frame in accepted]),
        accepted_validation_ssim_change_mean=mean([frame['validation_selected_ssim_change'] for frame in accepted]),
        candidates_total=sum(frame['candidates'] for frame in frames),
        born_total=sum(frame['born'] for frame in frames), retired_total=sum(frame['retired'] for frame in frames),
        initial_count=reference['total_count'], final_count=frames[-1]['total'],
        min_count=min([reference['total_count']]+[frame['total'] for frame in frames]),
        max_count=max([reference['total_count']]+[frame['total'] for frame in frames]),
        payload_components_kb_mean={name: mean([frame[name+'_bytes'] for frame in frames])/1000 for name in ('motion','birth','opacity')},
        prune_reasons_total={name:sum(frame['prune_reasons'].get(name,0) for frame in frames) for name in ('low_opacity','screen_size','world_size','multiple')},
        phase_seconds_mean={phase:mean([frame['phase_seconds'].get(phase,0.) for frame in frames]) for phase in ('motion','birth','refit','encoding','validation','stage_evaluation')},
        dynamic_update_kb_quantiles=dict(zip(['min','p50','p90','max'], np.quantile([frame['dynamic_update_bytes']/1000 for frame in frames],[0,.5,.9,1]).tolist())),
        dynamic_frames_over_50kb=sum(frame['dynamic_update_bytes']>50000 for frame in frames),
        birth_gain_quantiles=dict(zip(['min','p10','p50','p90','max'],np.quantile([frame['birth_net_gain'] for frame in frames],[0,.1,.5,.9,1]).tolist())),
        target_gap_db=34.2-summary['quality']['test']['metrics']['psnr'],
    )
    blocks=[]
    for start in (0,50,100,150,200,250):
        subset=[frame for frame in frames if start<=frame['frame']<start+50]
        quality=[frame['final_psnr'] for frame in subset]
        if start==0:quality=[reference['evaluation']['test']['psnr']]+quality
        blocks.append(dict(start=start,end=start+49,quality_count=len(quality),
                           psnr_mean=mean(quality),motion_psnr_mean=mean([frame['motion_psnr'] for frame in subset]),
                           birth_net_gain_mean=mean([frame['birth_net_gain'] for frame in subset]),
                           accepted=sum(frame['accepted'] for frame in subset),dynamic_frames=len(subset),
                           dynamic_update_kb_mean=mean([frame['dynamic_update_bytes']/1000 for frame in subset]),
                           end_count=subset[-1]['total']))
    return dict(run=relative,run_summary=summary,reference_evaluation=reference,
                consistency_checks=checks,stats=stats,blocks=blocks,frames=frames)


results={name:analyze(path) for name,path in RUNS.items()}
a,b=results['previous'],results['current']
results['comparison']=dict(
    final_psnr_delta=b['run_summary']['quality']['test']['metrics']['psnr']-a['run_summary']['quality']['test']['metrics']['psnr'],
    dynamic_motion_psnr_delta=b['stats']['motion_psnr_mean']-a['stats']['motion_psnr_mean'],
    dynamic_birth_net_gain_delta=b['stats']['birth_net_gain_mean']-a['stats']['birth_net_gain_mean'],
    dynamic_update_kb_delta=b['run_summary']['storage']['complete_update_kb_mean']-a['run_summary']['storage']['complete_update_kb_mean'],
    same_reference_hash=b['reference_evaluation']['reference_sha256']==a['reference_evaluation']['reference_sha256'],
    note='Sequence differences are observations, not isolated birth-method causal effects. Changed newborns alter subsequent motion inputs; CUDA training is not certified deterministic.',
)
diagnosis = json.loads((OUT/'fixed_attribute_diagnosis.json').read_text())
results['fixed_attribute_diagnosis'] = dict(
    source='fixed_attribute_diagnosis.json', scope=diagnosis['scope'],
    frames=[row['frame'] for row in diagnosis['frames']],
    components={key:dict(mean=mean([row[key] for row in diagnosis['frames']]),
                         minimum=min(row[key] for row in diagnosis['frames']),
                         maximum=max(row[key] for row in diagnosis['frames']))
                for key in ('adding_final_births_to_all_old_db',
                            'deleting_old_with_births_present_db',
                            'deleting_old_without_births_db','total_change_db')},
)
for name in RUNS:
    assert all(results[name]['consistency_checks'].values()), results[name]['consistency_checks']
(OUT/'comparison.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for name,color,label in [('previous','#8792a2','Previous MSE'),('current','#126cbe','Current subset + SSIM')]:
    rows=results[name]['frames'];x=np.array([row['frame'] for row in rows])
    final=np.array([row['final_psnr'] for row in rows])
    axes[0,0].plot(x,final,color=color,alpha=.2,lw=.6)
    axes[0,0].plot(x[9:-10],np.convolve(final,np.ones(20)/20,mode='valid'),color=color,label=label)
    gain=np.array([row['birth_net_gain'] for row in rows])
    axes[0,1].plot(x[9:-10],np.convolve(gain,np.ones(20)/20,mode='valid'),color=color,label=label)
    axes[1,0].plot(x,[row['dynamic_update_bytes']/1000 for row in rows],color=color,alpha=.75,lw=.8,label=label)
    axes[1,1].plot(x,[row['total']/1000 for row in rows],color=color,label=label)
axes[0,0].axhline(34.2,color='#b6503c',ls='--',lw=.8,label='34.2 dB target')
axes[0,0].set(title='cam00 final PSNR (20-frame moving mean)',ylabel='PSNR (dB)')
axes[0,1].axhline(0,color='black',lw=.5)
axes[0,1].set(title='Same-frame birth + refit gain (20-frame mean)',ylabel='PSNR gain (dB)')
axes[1,0].axhline(50,color='#b6503c',ls='--',lw=.8)
axes[1,0].set(title='Dynamic update: payload + index, excludes reference',ylabel='KB / frame (1000 bytes)')
axes[1,1].set(title='Final Gaussian count',ylabel='Thousand Gaussians')
for axis in axes.flat:
    axis.set_xlabel('Frame');axis.grid(alpha=.2);axis.legend(fontsize=8)
fig.savefig(OUT/'sequence_comparison.png',dpi=160)
plt.close(fig)
print(json.dumps({name:results[name]['stats'] for name in RUNS},indent=2))
print(json.dumps(results['comparison'],indent=2))
