"""Small, two-view residual colour nudges; not inverse alpha compositing."""
import math
import torch
from .support_reincarnation_proposal import _rotation, _sample


@torch.no_grad()
def conservative_colour_seed(raw, seed, rows, alpha, cache, gain=.5, max_rgb_delta=.05):
    """Approximate delta C = gain * (GT-I)/(T*alpha_new/2).

    Native parent coverage mass estimates T from the projected Gaussian area;
    the half factor is the Gaussian footprint's average alpha response. This
    assumes nearby parent/child footprints and similar depth ordering. It does
    not recover the occluded colour behind a newly inserted child. Therefore
    require support and signed agreement in BOTH proposal views, and clamp the
    resulting RGB nudge before converting to SH DC. SH remains freely learnable.
    """
    n=len(rows)
    report=dict(enabled=True,accepted=0,candidates=n,proposal_views=len(cache),
                gain=gain,max_rgb_delta=max_rgb_delta,
                semantics='bounded residual nudge using approximate parent T and child alpha; not exact compositing inversion')
    if not n:return report
    if len(cache)!=2:raise ValueError('Colour initialization needs exactly two training proposal views')
    xyz=seed['xyz'];values=[];valid_views=[]
    # Seed support lattice is only a sampling prior. Native response below is
    # the actual parent full-scene T*alpha contribution, not bbox visibility.
    rotation=_rotation(seed['rotation']);scale=seed['logscale'].exp()
    axes=scale.argsort(-1)[:,1:]
    basis=rotation.gather(2,axes[:,None].expand(-1,3,-1))
    lattice=xyz.new_tensor([[x,y] for y in (-1.,0.,1.) for x in (-1.,0.,1.)])
    world=xyz[:,None]+torch.einsum('nij,npj->npi',basis,lattice[None]*scale.gather(1,axes)[:,None])
    kernel=torch.exp(-.5*lattice.square().sum(-1))
    source_rotation=_rotation(raw['rotation'][rows])
    source_cov=(source_rotation*raw['logscale'][rows].exp()[:,None])
    for item in cache:
        view=item['view']
        if item['name']=='cam00.png' or bool(getattr(view,'is_test',False)):
            raise ValueError('Colour initialization cannot use held-out cameras')
        matrix=view.world_view_transform.to(xyz)
        camera=world@matrix[:3,:3]+matrix[3,:3];z=camera[...,2]
        grid=torch.stack((camera[...,0]/z.clamp_min(1e-12)/math.tan(float(view.fovx)*.5),
                          camera[...,1]/z.clamp_min(1e-12)/math.tan(float(view.fovy)*.5)),-1)
        valid=(z>0)&(grid.abs()<1).all(-1)
        weights=valid*kernel[None]
        sampled=-_sample(item['residual'],grid)
        local=(sampled*weights[...,None]).sum(1)/weights.sum(-1).clamp_min(1e-12)[:,None]
        mass=item['native_mass'][rows]
        signed=item['signed_response'][rows]/mass.clamp_min(1e-12)[:,None]
        # Project the parent covariance in pixel units to estimate its ideal
        # unoccluded alpha mass; native mass then supplies an average T proxy.
        p=raw['xyz'][rows]@matrix[:3,:3]+matrix[3,:3];depth=p[:,2].clamp_min(1e-12)
        height,width=item['residual'].shape[-2:]
        fx=width/(2*math.tan(float(view.fovx)*.5));fy=height/(2*math.tan(float(view.fovy)*.5))
        jac=xyz.new_zeros(n,2,3)
        jac[:,0,0]=fx/depth;jac[:,1,1]=fy/depth
        jac[:,0,2]=-fx*p[:,0]/depth.square();jac[:,1,2]=-fy*p[:,1]/depth.square()
        camera_basis=matrix[:3,:3].T[None]@source_cov
        projected=jac@camera_basis
        cov=projected@projected.transpose(-1,-2)
        cov=cov+torch.eye(2,device=xyz.device,dtype=xyz.dtype)[None]*.3
        area=2*math.pi*torch.linalg.det(cov).clamp_min(1e-12).sqrt()
        transmittance=(mass/(area*raw['opacity'][rows].sigmoid().flatten().clamp_min(1e-4))).clamp(0,1)
        response=.5*alpha.flatten()*transmittance
        # Exact native weighted residual and nearby seed-probe residual must
        # point in the same direction before using their conservative mean.
        coherent=(local*signed).sum(-1)>.5*local.norm(dim=-1)*signed.norm(dim=-1)
        valid_views.append(item['observed'][rows]&valid.sum(-1).ge(5)&mass.gt(1e-3)&
                           response.ge(.02)&coherent&signed.norm(dim=-1).gt(.002))
        values.append((local+signed)*.5/response.clamp_min(.02)[:,None])
    a,b=values
    consistent=(a*b).sum(-1)>.5*a.norm(dim=-1)*b.norm(dim=-1)
    accepted=valid_views[0]&valid_views[1]&consistent
    # Do not create a colour offset in channels with opposite evidence signs.
    delta=((a+b)*.5).clamp(-max_rgb_delta,max_rgb_delta)*gain
    delta=torch.where((a*b)>0,delta,torch.zeros_like(delta))
    delta=torch.where(accepted[:,None],delta,torch.zeros_like(delta))
    seed['sh_dc']+=delta[:,None]/.28209479177387814
    report.update(accepted=int(accepted.sum()),fallback=int((~accepted).sum()),
                  max_applied_rgb_delta=float(delta.abs().max()),
                  native_response_floor=.02)
    return report
