"""Birth priors from measured 2D residual support and inherited depth support.

Image moments are real native T*alpha error moments. Depth still comes from
the native compositor's parent centre depth, and ray thickness is a compact prior;
neither is a measured surface or triangulated geometry.
"""
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from .support_reincarnation_proposal import _rotation
from .early_replacement_model import _logits


def _matrix_quaternion(matrix):
    r=matrix
    roots=torch.stack((1+r[:,0,0]+r[:,1,1]+r[:,2,2],
        1+r[:,0,0]-r[:,1,1]-r[:,2,2],1-r[:,0,0]+r[:,1,1]-r[:,2,2],
        1-r[:,0,0]-r[:,1,1]+r[:,2,2]),-1).clamp_min(0).sqrt()
    candidates=torch.stack((
        torch.stack((roots[:,0].square(),r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]),-1),
        torch.stack((r[:,2,1]-r[:,1,2],roots[:,1].square(),r[:,1,0]+r[:,0,1],r[:,0,2]+r[:,2,0]),-1),
        torch.stack((r[:,0,2]-r[:,2,0],r[:,1,0]+r[:,0,1],roots[:,2].square(),r[:,2,1]+r[:,1,2]),-1),
        torch.stack((r[:,1,0]-r[:,0,1],r[:,2,0]+r[:,0,2],r[:,2,1]+r[:,1,2],roots[:,3].square()),-1)),1)
    best=roots.argmax(-1);row=torch.arange(len(r),device=r.device)
    return F.normalize(candidates[row,best]/(2*roots[row,best,None].clamp_min(1e-15)),dim=-1)


def support_distribution(first,second,width,height):
    """Conditional grid mean/covariance, including finite pixel-cell area."""
    mass=first[:,0].clamp_min(0)
    denominator=mass.clamp_min(torch.finfo(first.dtype).tiny)
    mean=first[:,1:]/denominator[:,None]
    bounds=mean.new_tensor([1-1/width,1-1/height])
    mean=torch.minimum(torch.maximum(mean,-bounds),bounds)
    values=second/denominator[:,None]
    covariance=torch.stack((values[:,0]-mean[:,0].square(),values[:,1]-mean[:,0]*mean[:,1],
        values[:,1]-mean[:,0]*mean[:,1],values[:,2]-mean[:,1].square()),-1).reshape(-1,2,2)
    eigen,basis=torch.linalg.eigh(covariance)
    covariance=(basis*eigen.clamp_min(0)[:,None])@basis.transpose(-1,-2)
    # A pixel has normalized-grid width 2/W, hence uniform-cell variance 1/(3W²).
    covariance=covariance+torch.diag(mean.new_tensor([1/(3*width**2),1/(3*height**2)]))
    return mass,mean,covariance


@torch.no_grad()
def residual_support_opacity(parent_opacity,rows,cache):
    """Conservative amplitude prior for relocated, previously untested support.

    Average residual RGB energy under the real parent compositor weights. This
    is an initialization heuristic, not a measured physical opacity or an exact
    photometric optimum. Copying an opaque parent's alpha onto a new footprint
    would assert that untested footprint already explains the whole foreground.
    """
    energy=parent_opacity.new_zeros(len(rows),dtype=torch.float64)
    mass=torch.zeros_like(energy)
    for item in cache:
        energy+=item['support_first_moments'][rows,0].double().clamp_min(0)
        mass+=item['native_mass'][rows].double().clamp_min(0)
    rms=(energy/(3*mass.clamp_min(torch.finfo(mass.dtype).tiny))).clamp(0,1).sqrt()
    alpha=torch.minimum(parent_opacity[:,0].double().sigmoid(),rms)
    if not bool(alpha.isfinite().all()):raise ValueError('Residual support opacity is nonfinite')
    return _logits(alpha[:,None].to(parent_opacity))


@torch.no_grad()
def residual_support_seed(raw,rows,cache,position_step):
    if not math.isfinite(position_step) or position_step<=0:
        raise ValueError('Residual support needs a finite positive position grid')
    seed={key:raw[key][rows].clone() for key in ('xyz','rotation','logscale')}
    count=len(rows);valid_any=torch.zeros(count,dtype=torch.bool,device=rows.device)
    if not count:return seed,valid_any,dict(policy='native_error_footprint_v1',seeded_count=0)
    if not cache:raise ValueError('Residual support needs training-view moments')
    if bool(seed['rotation'].double().norm(dim=-1).le(0).any()):
        raise ValueError('Residual support source quaternion is zero')
    xyz=seed['xyz'].double();rotation=_rotation(seed['rotation'].double())
    scale=seed['logscale'].double().exp()
    if not all(bool(torch.isfinite(v).all()) for v in (xyz,rotation,scale)) or bool(scale.le(0).any()):
        raise ValueError('Residual support source geometry is nonfinite')
    cache=sorted(cache,key=lambda c:c.get('name',Path(str(c['view'].path)).name))
    proposals=[];covariances=[];weights=[];valid_views=[];directions=[]
    for item in cache:
        if 'support_first_moments' not in item or 'support_second_moments' not in item:
            raise ValueError('Residual support moments were not collected')
        view=item['view'];height,width=view.image.shape[-2:]
        first=item['support_first_moments'][rows].double();second=item['support_second_moments'][rows].double()
        if first.shape!=(count,3) or second.shape!=(count,3) or not bool(torch.isfinite(first).all()&torch.isfinite(second).all()):
            raise ValueError('Residual support moments must be finite aligned triples')
        mass,mean,covariance=support_distribution(first,second,width,height)
        matrix=view.world_view_transform.to(xyz);inverse=torch.linalg.inv(matrix)
        origin=inverse[3,:3];camera=xyz@matrix[:3,:3]+matrix[3,:3]
        tx,ty=math.tan(float(view.fovx)*.5),math.tan(float(view.fovy)*.5)
        camera_ray=torch.cat((mean*mean.new_tensor([tx,ty]),torch.ones(count,1,device=xyz.device,dtype=xyz.dtype)),-1)
        ray=camera_ray@inverse[:3,:3]
        # Native compositing sorts each whole Gaussian at its centre depth.
        # Reuse that prior instead of claiming a new depth/plane observation.
        depth=camera[:,2]
        position=origin+depth[:,None]*ray
        valid=(mass>0)&item['observed'][rows]&camera[:,2].gt(.2)&depth.gt(.2)
        valid&=torch.isfinite(position).all(-1)
        tangent=torch.stack((depth[:,None]*tx*inverse[0,:3],depth[:,None]*ty*inverse[1,:3]),-1)
        world_cov=tangent@covariance@tangent.transpose(-1,-2)
        # Along-ray depth extent is not observed by this footprint. Initialize
        # it with the geometric mean tangent radius, rather than copying an
        # arbitrarily broad or almost singular parent covariance.
        ray_variance=depth.square()*tx*ty*torch.linalg.det(covariance).clamp_min(0).sqrt()
        direction=F.normalize(ray,dim=-1)
        world_cov+=ray_variance[:,None,None]*direction[:,:,None]*direction[:,None,:]
        world_cov+=torch.eye(3,device=xyz.device,dtype=xyz.dtype)[None]*position_step**2
        proposals.append(position);covariances.append(world_cov);directions.append(direction)
        weights.append(torch.where(valid,mass,torch.zeros_like(mass)));valid_views.append(valid)
    positions=torch.stack(proposals,1);covariances=torch.stack(covariances,1)
    weights=torch.stack(weights,1);valid_views=torch.stack(valid_views,1);valid_any=valid_views.any(-1)
    # Select a real view-supported proposal closest to the weighted consensus;
    # do not average conflicting rays into a new unsupported 3D position.
    distance=(positions[:,:,None]-positions[:,None,:]).square().sum(-1)
    cost=(distance*weights[:,None,:]).sum(-1).masked_fill(~valid_views,torch.inf)
    chosen=cost.argmin(-1);index=torch.arange(count,device=xyz.device)
    position=positions[index,chosen];covariance=covariances[index,chosen]
    parent_radius=scale.amax(-1).clamp_min(position_step)
    normalized_distance=(positions-position[:,None]).norm(dim=-1)/parent_radius[:,None]
    normalized_distance=normalized_distance.masked_fill(~valid_views,torch.inf)
    ordered_distance=normalized_distance.sort(-1).values
    second_view_distance=(ordered_distance[:,1] if len(cache)>1 else
                          ordered_distance[:,0].new_full((count,),torch.inf))
    weighted_spread=((positions-position[:,None]).square().sum(-1)*weights).sum(-1)
    weighted_spread=(weighted_spread/weights.sum(-1).clamp_min(
        torch.finfo(weights.dtype).tiny)).sqrt()/parent_radius
    # Separate the new primitive from an exactly tied source-depth layer by
    # the codec's coordinate uncertainty, rather than a parent-size jitter.
    position=position-2*position_step*torch.stack(directions,1)[index,chosen]
    eigen,basis=torch.linalg.eigh(covariance);eigen=eigen.clamp_min(position_step**2)
    determinant=torch.linalg.det(basis)
    basis[:,:,0]*=torch.where(determinant<0,-torch.ones_like(determinant),torch.ones_like(determinant))[:,None]
    flips=xyz.new_tensor([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    choices=basis[:,None]*flips[None,:,None,:]
    closeness=(choices*rotation[:,None]).sum((-1,-2))
    basis=choices[index,closeness.argmax(-1)]
    seed.update(xyz=position.to(seed['xyz']),rotation=_matrix_quaternion(basis).to(seed['rotation']),
                logscale=(.5*eigen.log()).to(seed['logscale']))
    if not all(bool(torch.isfinite(v).all()) for v in seed.values()):
        raise ValueError('Residual support produced nonfinite seeds')
    ratios=eigen.sqrt().amax(-1)/scale.amax(-1)
    def quantiles(value):
        value=value[valid_any&torch.isfinite(value)]
        if not len(value):return None
        return {str(q):float(torch.quantile(value,q)) for q in (.1,.25,.5,.75,.9,.95,.99)}

    report=dict(policy='native_error_footprint_v1',seeded_count=int(valid_any.sum()),
        single_view_count=int((valid_views.sum(-1)==1).sum()),
        support_view_count_quantiles=quantiles(valid_views.sum(-1).double()),
        second_view_distance_parent_sigma_quantiles=quantiles(second_view_distance),
        weighted_proposal_spread_parent_sigma_quantiles=quantiles(weighted_spread),
        consistent_second_view_count={str(limit):int((valid_any&(second_view_distance<=limit)).sum())
                                      for limit in (1.,2.,4.,8.)},
        child_to_parent_max_scale_median=float(ratios[valid_any].median()) if bool(valid_any.any()) else None,
        selected_view_counts=torch.bincount(chosen[valid_any],minlength=len(cache)).cpu().tolist(),
        view_names=[c.get('name',Path(str(c['view'].path)).name) for c in cache],
        depth_semantics='native parent centre-depth prior with codec-scale forward separation; not measured or triangulated depth',
        covariance_semantics='measured image-error footprint plus compact unobserved ray-depth prior')
    return seed,valid_any,report
