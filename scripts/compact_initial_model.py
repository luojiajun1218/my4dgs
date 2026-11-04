"""Prune a static reference by native rendered-image influence, then save a seed.

For a scalar opacity gate, Rademacher image VJPs estimate the squared norm
of its image Jacobian. This approximates individual removal influence;
simultaneous removals interact, so validate the pruned result separately.
Scores use training cameras only. Historical checkpoints are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from libgs.utils.config import to_structured
from libgs.metric import psnr,ssim
from pipeline.entry import Config
from pipeline.model.birth_quantization import RAW_FIELDS,model_raw,reference_digest
from pipeline.model.residual_birth import RawView
from pipeline.model.rigid_tree_codec import DecodedState
from scripts import benchmark_rigid_residual_tree as base


def cpu_tree(value):
    if isinstance(value,torch.nn.Parameter):
        return torch.nn.Parameter(value.detach().cpu(),requires_grad=value.requires_grad)
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def coverage_selection(scores, groups, target, minimum=2):
    """Keep each occupied cell's best rows, then fill by global influence."""
    if (scores.ndim != 1 or not scores.is_floating_point() or not torch.isfinite(scores).all()
            or groups.shape != scores.shape or groups.dtype != torch.long
            or groups.device != scores.device or (groups < 0).any()):
        raise ValueError('Coverage needs finite scores and matching nonnegative int64 region IDs')
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError('Coverage minimum must be a positive integer')
    if not 0 < target <= len(scores):
        raise ValueError('Invalid coverage target')
    order = torch.argsort(scores, descending=True, stable=True)
    by_group = order[torch.argsort(groups[order], stable=True)]
    _, counts = torch.unique_consecutive(groups[by_group], return_counts=True)
    starts = (counts.cumsum(0) - counts).repeat_interleave(counts)
    rank = torch.arange(len(scores), device=scores.device) - starts
    mandatory = by_group[rank < minimum]
    if len(mandatory) > target:
        raise ValueError(f'Coverage requires {len(mandatory)} rows but target is {target}')
    selected = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    selected[mandatory] = True
    selected[order[~selected[order]][:target-len(mandatory)]] = True
    return selected.nonzero(as_tuple=True)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--render-config',type=Path,required=True,
        help='Rendering configuration for the checkpoint dataset and resolution')
    parser.add_argument('--output',type=Path,required=True)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--target-count',type=int)
    budget.add_argument('--keep-fraction',type=float)
    parser.add_argument('--probes',type=int,default=2)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--opacity-threshold',type=float,default=.0075)
    parser.add_argument('--coverage-reference',type=Path,
        help='Keep at least two available GS in each occupied finest candidate cell; requires the same frame0 xyz')
    args=parser.parse_args()
    paths=[args.output,args.output.with_suffix('.json'),args.output.with_suffix('.scores.pt')]
    if any(p.exists() for p in paths):raise FileExistsError(paths)
    if any((parent/'config.yaml').is_file() for parent in args.output.resolve().parents):
        raise ValueError('Do not modify a historical run')
    if not 0<=args.opacity_threshold<1 or args.probes<1:raise ValueError('Invalid score settings')
    if args.keep_fraction is not None and not 0<args.keep_fraction<1:
        raise ValueError('keep-fraction must be between0 and1')
    torch.set_num_threads(2)
    config,extra=to_structured(OmegaConf.load(args.render_config),Config)
    if extra:raise ValueError(extra)
    expected_size=config.data.expected_image_size
    if expected_size is None or len(expected_size)!=2:
        raise ValueError('render-config must specify expected_image_size as [width, height]')
    expected_shape=(3,int(expected_size[1]),int(expected_size[0]))
    config.data.extra_dataset_kwargs['num_frames']=1
    packet=torch.load(args.checkpoint,map_location='cuda',weights_only=False)
    if packet.get('current_frame')!=0:raise ValueError('Expected static frame0')
    config.module.sh_degree=int(packet['gaussians'][0])
    module,dm=base._make_module(config,torch.device('cuda'))
    model=module.gaussians
    model.restore(packet['gaussians'],config.module.gaussian)
    initial_count=len(model._xyz)
    remove=model.get_opacity.detach().flatten()<args.opacity_threshold
    if remove.all():raise ValueError('No opacity survivors')
    if remove.any():model.prune_points(remove)
    old_count=len(model._xyz)
    target=args.target_count if args.target_count is not None else int(old_count*args.keep_fraction)
    if not 0<target<old_count:raise ValueError('Budget must remove at least one point and retain at least one')
    raw=model_raw(model)
    raw={k:v.detach() for k,v in raw.items()}
    coverage_state = None
    if args.coverage_reference is not None:
        import numpy as np
        with np.load(args.coverage_reference,allow_pickle=False) as reference:
            coverage_state = DecodedState.from_reference(reference)
        if (coverage_state.frame_index != 0 or coverage_state.spatial_paths is None
                or not torch.equal(coverage_state.xyz,raw['xyz'].cpu())):
            raise ValueError('Coverage reference must have identical post-opacity frame0 positions')
    view,count=base._cam00(dm,0,torch.device('cuda'))
    if count!=1 or tuple(view.image.shape)!=expected_shape:
        raise ValueError(f'Wrong evaluation camera or image size: expected {expected_shape}')
    alpha=model.get_opacity.detach().clone().requires_grad_(True)
    module.renderer.gaussians=RawView(raw,config.module.sh_degree,model,alpha)
    dm.current_frame=0
    dataset=dm.get_train_dataset()
    ordered=sorted((Path(str(v.path)).name,i) for i,v in enumerate(dataset.items))
    if any(name=='cam00.png' for name,_ in ordered):raise ValueError('Heldout camera in score set')
    scores=torch.zeros(old_count,device='cuda',dtype=torch.float64)
    checks=[]
    for view_index,(name,index) in enumerate(ordered):
        view=dataset[index].to('cuda')
        image=module.renderer(view,module.background)['render']
        for probe in range(args.probes):
            generator=torch.Generator(device='cuda').manual_seed(args.seed+100003*view_index+probe)
            signs=torch.randint(0,2,image.shape,device='cuda',generator=generator,dtype=torch.int32).to(image).mul_(2).sub_(1)
            gradient=torch.autograd.grad((image*signs).sum(),alpha,retain_graph=probe+1<args.probes)[0]
            gated=(alpha.detach()*gradient).flatten()
            scores.add_(gated.double().square()/args.probes)
            if view_index==0 and probe==0:
                # Forward perturbations independently check the native VJP.
                selected=gated.abs().argsort(descending=True)[:3]
                for row in selected.tolist():
                    perturbed=alpha.detach().clone();perturbed[row]*=.95
                    previous=module.renderer.gaussians
                    module.renderer.gaussians=RawView(raw,config.module.sh_degree,model,perturbed)
                    with torch.no_grad():trial=module.renderer(view,module.background)['render']
                    module.renderer.gaussians=previous
                    actual=float(((trial-image.detach())*signs).double().sum())
                    predicted=-.05*float(gated[row])
                    checks.append(dict(row=row,predicted_projection=predicted,actual_projection=actual,
                        relative_error=abs(actual-predicted)/max(abs(actual),abs(predicted),1e-8)))
        print(f'INFLUENCE {view_index+1}/{len(ordered)} {name}',flush=True)
        del image,view
    if not torch.isfinite(scores).all():raise ValueError('Non-finite influence')
    selected=scores.argsort(descending=True,stable=True)[:target]
    coverage_report = None
    if coverage_state is not None:
        groups = coverage_state.spatial_paths[:,-1].to(scores.device)
        coverage_selected = coverage_selection(scores,groups,target)
        original_counts = torch.bincount(groups,minlength=len(coverage_state.parents))
        global_counts = torch.bincount(groups[selected],minlength=len(original_counts))
        final_counts = torch.bincount(groups[coverage_selected],minlength=len(original_counts))
        required = original_counts.clamp_max(2)
        assert bool(final_counts.ge(required).all())
        global_mask = torch.zeros_like(scores,dtype=torch.bool);global_mask[selected]=True
        coverage_report = dict(reference=str(args.coverage_reference.resolve()),
            reference_sha256=hashlib.sha256(args.coverage_reference.read_bytes()).hexdigest(),
            original_occupied_regions=int(original_counts.gt(0).sum()),minimum_per_region=2,
            unguarded_emptied_regions=int((original_counts.gt(0)&global_counts.eq(0)).sum()),
            unguarded_regions_below_available_minimum=int(global_counts.lt(required).sum()),
            retained_occupied_regions=int(final_counts.gt(0).sum()),
            guarded_regions_below_available_minimum=int(final_counts.lt(required).sum()),
            selected_rows_replaced=int((~global_mask[coverage_selected]).sum()))
        selected = coverage_selected
    remove=torch.ones(old_count,dtype=torch.bool,device='cuda');remove[selected]=False
    view,_=base._cam00(dm,0,torch.device('cuda'))

    @torch.no_grad()
    def evaluate():
        module.renderer.gaussians=model
        image=module.forward(view)['render'].clamp(0,1)
        return dict(points=len(model._xyz),project_PSNR=float(psnr(image,view.image)),
            global_RGB_PSNR=float(-10*(image-view.image).square().mean().log10()),SSIM=float(ssim(image,view.image)))

    before=evaluate()
    model.prune_points(remove)
    after=evaluate()
    packet['gaussians']=cpu_tree(model.capture())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cpu_tree(packet),args.output)
    # Check both retained attributes and Adam row layout after serialization.
    saved=torch.load(args.output,map_location='cpu',weights_only=False)
    actual={k:saved['gaussians'][i] for k,i in base.CAPTURE_INDEX.items()}
    assert reference_digest(actual)==reference_digest(model_raw(model))
    for state in saved['gaussians'][10]['state'].values():
        for key in ('exp_avg','exp_avg_sq'):
            if key in state:assert state[key].shape[0]==target
    torch.save(dict(scores=scores.cpu(),selected_rows=(~remove).nonzero(as_tuple=True)[0].cpu()),paths[2])
    report=dict(source=str(args.checkpoint.resolve()),source_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        output=str(args.output.resolve()),global_step=int(packet['global_step']),SH_degree=config.module.sh_degree,
        original_points=initial_count,after_opacity_points=old_count,target_points=target,
        opacity_threshold=args.opacity_threshold,probes_per_training_view=args.probes,training_cameras=[n for n,_ in ordered],
        native_VJP_finite_perturbation_checks=checks,before=before,after=after,
        retained_attribute_sha=reference_digest(actual),optimizer_rows_verified=True,
        coverage=coverage_report,
        score_quantiles=torch.quantile(scores.cpu(),torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),
        scope='Native training-camera image-Jacobian energy estimated by Rademacher VJP, not exact joint deletion cost. No GT pixels in per-point scores. cam00 evaluates model selection only, not independent generalization. Retained attributes are unchanged; validate dynamic30/300 before dynamic claims.')
    paths[1].write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
