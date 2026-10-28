"""Prune a static reference by native rendered-image influence, then save a seed.

For a scalar opacity gate, Rademacher image VJPs estimate the squared norm
of its image Jacobian. This approximates individual removal influence;
simultaneous removals interact, so use progressive pruning and recovery fit.
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
from scripts import benchmark_rigid_residual_tree as base


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--render-config',type=Path,default=ROOT/'config/fs_units_300.yaml')
    parser.add_argument('--output',type=Path,required=True)
    budget=parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--target-count',type=int)
    budget.add_argument('--keep-fraction',type=float)
    parser.add_argument('--probes',type=int,default=2)
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--opacity-threshold',type=float,default=.0075)
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
    remove=torch.ones(old_count,dtype=torch.bool,device='cuda');remove[selected]=False
    view,count=base._cam00(dm,0,torch.device('cuda'))
    if count!=1 or tuple(view.image.shape)!=(3,1008,1345):raise ValueError('Wrong evaluation camera')

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
        score_quantiles=torch.quantile(scores.cpu(),torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist(),
        scope='Native training-camera image-Jacobian energy estimated by Rademacher VJP, not exact joint deletion cost. No GT pixels in per-point scores. cam00 evaluates model selection only, not independent generalization. Recover with static fitting, then validate dynamic30/300 before claims.')
    paths[1].write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
