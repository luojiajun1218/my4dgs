"""Prototype residual-pixel spawning and contribution-ranked retirement.

Saved step-180 models are fixed inputs. Uses training cameras for all candidate
and retirement decisions. Does not change the training pipeline or write codec.
"""
from pathlib import Path
import argparse
import json
import math
import random
import statistics
import sys
import time
import warnings
import torch
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.model.gaussian import GaussianModel
from pipeline.model.birth_quantization import RAW_ATTRIBUTES
from libgs.utils.config import to_structured
from libgs.renderer import Renderer
from libgs.metric import psnr
from libgs.model.merged_gaussian import MergedGaussianModel
from libgs.utils.sh import eval_sh,C0
from pipeline.entry import Config
from pipeline.data import DataModule
from pipeline.model.birth_quantization import RAW_FIELDS,QuantizedNewGaussianView
from pipeline.module import motion_photometric_loss
from pipeline.model.multiview_birth import (MultiViewBirthConfig,
    propose_multiview_births, validation_edit_gain)
from libgs.metric import ssim

def raw_capture(capture):
    return dict(zip(('xyz','sh_dc','sh_rest','logscale','rotation','opacity'),
                    [value.detach() for value in capture[1:7]]))


def model(raw):
    result=GaussianModel(1)
    result.active_sh_degree=1
    for key,attr in zip(RAW_FIELDS,RAW_ATTRIBUTES):
        setattr(result,attr,torch.nn.Parameter(raw[key],requires_grad=False))
    return result


def project_path(value):
    path=Path(value).expanduser()
    return path if path.is_absolute() else ROOT/path


def evaluate(renderer,views,cfg):
    with torch.no_grad():
        scores=[];losses=[];errors=[];structural=[]
        for v in views:
            im=renderer(v,torch.zeros(3,device='cuda'))['render']
            scores.append(float(psnr(im.clamp(0,1),v.image)))
            losses.append(float(motion_photometric_loss(im,v.image,cfg.module.lambda_dssim)))
            errors.append(float((im.clamp(0,1)-v.image).square().mean()))
            structural.append(float(ssim(im.clamp(0,1),v.image)))
        return {'psnr':statistics.mean(scores),'photo':statistics.mean(losses),
                'mse':statistics.mean(errors),'ssim':statistics.mean(structural)}

@torch.no_grad()
def place_candidates(renderer,views,full,cfg,settings):
    points=[];colors=[];footprints=[];view_indices=[]
    h,w=views[0].image.shape[-2:];bg=torch.zeros(3,device='cuda')
    homogeneous=torch.cat((full['xyz'],torch.ones_like(full['xyz'][:,:1])),1)
    # 5x5 pooling reduces sensor/pixel noise. One candidate per 16x16 tile.
    for vi,v in enumerate(views):
        im=renderer(v,bg)['render']
        z=(homogeneous@v.world_view_transform)[:,2]
        attr=torch.stack((z,torch.ones_like(z),torch.zeros_like(z)),1)
        acc=renderer(v,bg,override_color=attr)['render']
        valid=(acc[1]>.5)&(acc[0]>0)
        error=torch.nn.functional.avg_pool2d((im-v.image).abs().mean(0)[None,None],5,1,2)[0,0]
        error=error.masked_fill(~valid,0)
        values,indices=torch.nn.functional.max_pool2d(error[None,None],16,16,return_indices=True)
        top=torch.topk(values.flatten(),min(settings.candidates_per_view,values.numel())).indices
        top=top[values.flatten()[top]>0]
        if not len(top):
            continue
        pixel=indices.flatten()[top];y=torch.div(pixel,w,rounding_mode='floor');x=pixel%w
        depth=acc[0,y,x]/acc[1,y,x]
        ndcx=(x.float()+.5)*2/w-1;ndcy=(y.float()+.5)*2/h-1
        cam=torch.stack((ndcx*math.tan(v.fovx*.5)*depth,
                         ndcy*math.tan(v.fovy*.5)*depth,depth,torch.ones_like(depth)),1)
        world=(cam@torch.linalg.inv(v.world_view_transform))[:,:3]
        projected=torch.cat((world,torch.ones_like(world[:,:1])),1)@v.full_proj_transform
        uv=(projected[:,:2]/projected[:,3:4]+1)*projected.new_tensor((w,h))*.5-.5
        assert float((uv-torch.stack((x,y),1)).abs().max())<.01
        points.append(world);colors.append(v.image[:,y,x].T)
        footprints.append(depth*(2*math.tan(v.fovx*.5)/w)*settings.initial_sigma_pixels)
        view_indices.extend([vi]*len(world))
    if not points:
        return {k:v[:0].clone() for k,v in full.items()},torch.empty(0,device='cuda',dtype=torch.long)
    xyz=torch.cat(points);rgb=torch.cat(colors)
    _,nn=cKDTree(full['xyz'].cpu().numpy()).query(xyz.cpu().numpy(),k=1)
    source=torch.as_tensor(nn,device='cuda',dtype=torch.long)
    # Remove near-duplicate samples from different views without a count quota.
    cell=torch.round(xyz/(1e-3*cfg.data_extent)).long()
    unique={};keep=[]
    for i,key in enumerate(cell.cpu().tolist()):
        key=tuple(key)
        if key not in unique:unique[key]=i;keep.append(i)
    keep=torch.as_tensor(keep,device='cuda');xyz=xyz[keep];rgb=rgb[keep];source=source[keep]
    raw={k:v[source].clone() for k,v in full.items()};raw['xyz']=xyz
    limit=torch.cat(footprints)[keep].clamp_min(1e-6)
    largest=raw['logscale'].exp().max(1).values
    raw['logscale']+=torch.minimum(torch.ones_like(limit),limit/largest).log()[:,None]
    # Retain higher SH; correct DC so this training view evaluates to sampled RGB.
    for j,index in enumerate(keep.cpu().tolist()):
        v=views[view_indices[index]]
        direction=torch.nn.functional.normalize(xyz[j:j+1]-v.camera_center,dim=-1)
        features=torch.cat((raw['sh_dc'][j:j+1],raw['sh_rest'][j:j+1]),1).transpose(1,2)
        old_color=eval_sh(1,features,direction)[0]+.5
        raw['sh_dc'][j,0]+=(rgb[j]-old_color)/C0
    alpha=settings.initial_alpha
    raw['opacity'].fill_(math.log(alpha/(1-alpha)))
    return raw,source

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,
        default=ROOT/'config/diagnostic/flame_steak_birth_revision.yaml')
    args=parser.parse_args()
    settings=OmegaConf.load(args.config)
    run=project_path(settings.run_dir)
    output=project_path(settings.result_json)
    output.parent.mkdir(parents=True,exist_ok=True)
    if (settings.fit_steps<1 or settings.views_per_step<1 or settings.candidates_per_view<1
            or not 0<settings.initial_alpha<1 or settings.initial_sigma_pixels<=0
            or not 0<=settings.retirement_tolerance<1
            or not 0<=settings.minimum_relative_validation_gain<1):
        raise ValueError('Invalid diagnostic settings')
    torch.set_num_threads(4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        cfg,_=to_structured(OmegaConf.load(run/'config.yaml'),Config)
    if cfg.module.sh_degree!=1:
        raise ValueError('This Flame Steak diagnostic expects SH degree 1')
    data=DataModule(cfg.data);data.setup()
    cfg.data_extent=data.cameras_extent
    results=[]
    for frame in settings.frames:
        start=time.monotonic();data.current_frame=frame
        ds=data.get_train_dataset();order=sorted(range(len(ds)),key=lambda i:ds.items[i].path.name)
        views=[ds[i].to('cuda') for i in order]
        validation_names=set(settings.validation_cameras)
        validation=[v for v in views if v.path.name in validation_names]
        train=[v for v in views if v.path.name not in validation_names]
        if len(validation)!=len(validation_names) or not validation or not train:
            raise ValueError('Validation cameras must form a nonempty proper subset of training cameras')
        ds=data.get_test_dataset();test=[ds[i].to('cuda') for i in range(len(ds))]
        ck=torch.load(run/f'rigid-ckpt-{frame}-180.pth',map_location='cuda',weights_only=False)
        full=raw_capture(ck['gaussians']);base=model(full)
        renderer=Renderer(cfg.module.renderer,base)
        baseline={'train':evaluate(renderer,train,cfg),'validation':evaluate(renderer,validation,cfg),'test':evaluate(renderer,test,cfg)}
        proposal_report={}
        if settings.get('candidate_method','pixel_depth')=='multiview_residual':
            multiview=MultiViewBirthConfig(**OmegaConf.to_container(settings.multiview))
            raw,source,proposal_report=propose_multiview_births(renderer,train,full,
                torch.zeros(3,device='cuda'),data.cameras_extent,multiview)
        else:
            raw,source=place_candidates(renderer,train,full,cfg,settings)
        if not len(source):
            record=dict(frame=frame,new_count=0,retired_count=0,
                accepted_by_validation_loss=False,baseline=baseline,
                committed_new_count=0,committed_retired_count=0,
                committed_test_psnr=baseline['test']['psnr'])
            results.append(record)
            output.write_text(json.dumps(dict(settings=OmegaConf.to_container(settings),records=results),indent=2)+'\n')
            print(json.dumps(record),flush=True)
            continue
        new=model(raw)
        groups=[]
        rates={'_xyz':settings.position_lr_extent*data.cameras_extent,
                        '_features_dc':settings.color_lr,
                        '_scaling':settings.scale_lr,'_opacity':settings.alpha_lr}
        if settings.get('rest_lr') is not None:
            rates['_features_rest']=settings.rest_lr
        for attr,lr in rates.items():
            parameter=getattr(new,attr);parameter.requires_grad_(True)
            groups.append({'params':[parameter],'lr':lr,'initial_lr':lr})
        optimizer=torch.optim.Adam(groups,eps=1e-15)
        def refresh(base_model=base):
            renderer.gaussians=MergedGaussianModel([base_model,QuantizedNewGaussianView(
                new,cfg.module.birth,data.cameras_extent,full,source)])
        refresh();initial={'train':evaluate(renderer,train,cfg),'validation':evaluate(renderer,validation,cfg),'test':evaluate(renderer,test,cfg)}
        rng=random.Random(0);indices=[]
        while len(indices)<settings.fit_steps*settings.views_per_step:
            cycle=list(range(len(train)));rng.shuffle(cycle);indices.extend(cycle)
        stages={}
        for step in range(1,settings.fit_steps+1):
            for vi in indices[settings.views_per_step*(step-1):settings.views_per_step*step]:
                v=train[vi];refresh()
                im=renderer(v,torch.zeros(3,device='cuda'))['render']
                image_loss=((im-v.image).square().mean() if settings.get('fit_loss','photo')=='mse'
                    else motion_photometric_loss(im,v.image,cfg.module.lambda_dssim))
                loss=image_loss/settings.views_per_step
                loss.backward()
            for group in optimizer.param_groups:
                group['lr']=group['initial_lr']*(settings.lr_final_ratio**((step-1)/max(1,settings.fit_steps-1)))
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            if step in {*settings.evaluation_steps,settings.fit_steps}:
                refresh();stages[step]={'train':evaluate(renderer,train,cfg),'validation':evaluate(renderer,validation,cfg),'test':evaluate(renderer,test,cfg)}
        # Contribution = sum over image pixels of the actual alpha*T blending
        # weights. Color-override derivative gives this exactly in one backward.
        refresh();n=len(full['xyz']);k=len(new._xyz)
        contribution=torch.zeros(n,device='cuda')
        for index,v in enumerate(train):
            labels=torch.zeros((n+k,3),device='cuda',requires_grad=True)
            rendered=renderer(v,torch.zeros(3,device='cuda'),override_color=labels)['render']
            grad,=torch.autograd.grad(rendered.sum(),labels)
            if not bool(torch.isfinite(grad).all()) or bool((grad < -1e-6).any()):
                raise AssertionError('Compositing weights must be finite and nonnegative')
            if index==0:
                with torch.no_grad():
                    alpha_image=renderer(v,torch.zeros(3,device='cuda'),
                        override_color=torch.ones_like(labels))['render'][0]
                # Independent render verifies the sum of color derivatives.
                torch.testing.assert_close(grad[:,0].sum(),alpha_image.sum(),rtol=1e-4,atol=.05)
            contribution=torch.maximum(contribution,grad[:n].mean(1))
        protected=torch.zeros(n,device='cuda',dtype=torch.bool);protected[source]=True
        eligible=(~protected).nonzero(as_tuple=True)[0]
        ranked=eligible[torch.argsort(contribution[eligible],stable=True)]
        trials=[]
        for d in sorted(set((math.ceil((1-settings.retirement_tolerance)*k),k,
                              math.floor((1+settings.retirement_tolerance)*k)))):
            if d>len(eligible):
                raise ValueError('Not enough unprotected old points for the retirement band')
            keep=torch.ones(n,device='cuda',dtype=torch.bool);keep[ranked[:d]]=False
            trial=model({key:value[keep] for key,value in full.items()})
            refresh(trial)
            score=evaluate(renderer,validation,cfg)
            criterion=(-score['psnr'] if settings.get('admission_metric','photo')=='psnr' else score['photo'])
            trials.append((criterion,d,keep,score))
        _,d,keep,validation_metrics=min(trials,key=lambda v:(v[0],v[1]))
        if settings.get('admission_metric','photo')=='psnr':
            accepted,_=validation_edit_gain(baseline['validation'],validation_metrics,
                min_gain_db=settings.get('minimum_validation_gain_db',0.),
                max_ssim_drop=settings.get('max_ssim_drop',1e-4))
        else:
            accepted=validation_metrics['photo']<baseline['validation']['photo']*(1-settings.minimum_relative_validation_gain)
        trial=model({key:value[keep] for key,value in full.items()});refresh(trial)
        final={'train':evaluate(renderer,train,cfg),'validation':validation_metrics,'test':evaluate(renderer,test,cfg)}
        record={'frame':frame,'new_count':k,'retired_count':d,'accepted_by_validation_loss':accepted,
            'committed_new_count':k if accepted else 0,'committed_retired_count':d if accepted else 0,
            'baseline':baseline,'initial':initial,'proposal_report':proposal_report,
            'validation_camera_names':sorted(validation_names),'minimum_relative_validation_gain':settings.minimum_relative_validation_gain,'stages':stages,'retirement_trial':final,
            'wall_seconds':time.monotonic()-start,'committed_test_psnr':final['test']['psnr'] if accepted else baseline['test']['psnr'],
            'scope':'Fixed saved motion. Candidate fitting and retirement trials only; all candidate rows retained during fitting. No sequential training, checkpoint resume or codec measurement.'}
        results.append(record);output.write_text(json.dumps(dict(settings=OmegaConf.to_container(settings),records=results),indent=2)+'\n');print(json.dumps(record),flush=True)
        del ck,full,base,new,renderer,optimizer,groups,raw,train,test,labels,rendered,grad,im,loss,trials,trial

if __name__=='__main__':main()
