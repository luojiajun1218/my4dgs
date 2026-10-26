"""Inherited-support births selected by native residual-SSE responsibility.

No depth mixture, dominant ID, inferred plane, source-size limit or signed-DC
gradient is used to decide which old support deserves another representation.
A selected parent is copied in all six raw fields. Its nine covariance probes
are a direction prior only, not newly triangulated geometry or rendered cover.
"""
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS,model_raw


def _ordered_views(views):
    views=list(views.values()) if isinstance(views,dict) else list(views)
    ordered=sorted(((Path(str(view.path)).name,view) for view in views),key=lambda item:item[0])
    names=[name for name,_ in ordered]
    if not names or len(set(names))!=len(names):
        raise ValueError('Training camera names must be nonempty and unique')
    if 'cam00.png' in names or any(bool(getattr(view,'is_test',False)) for _,view in ordered):
        raise ValueError('Support proposals and cross-fitting cannot use external test cameras')
    if len(names)>63:raise ValueError('Training-view evidence bits require at most 63 cameras')
    return ordered


def _native_responsibility(module,views,residual_floor=0.,collect_colour_evidence=False):
    ordered=_ordered_views(views);old=module.gaussians
    if any(getattr(old,name,None) is not None for name in
           ('rigid_xyz','rigid_rotation','rigid_alpha','rigid_opacity','rigid_sh_dc')):
        raise ValueError('Support reincarnation requires a materialized incumbent')
    xyz=old.get_xyz
    scores=xyz.new_zeros(len(xyz));observed_bits=torch.zeros(len(xyz),dtype=torch.long,device=xyz.device)
    responsibility_bits=torch.zeros_like(observed_bits)
    renderer=module.renderer;previous_model=renderer.gaussians
    previous_collect=getattr(renderer,'collect_contributions',False)
    has_last=hasattr(renderer,'last_contributions')
    previous_last=getattr(renderer,'last_contributions',None)
    cache=[];per_view=[];total_error=0.
    renderer.gaussians=old
    try:
        for index,(name,view) in enumerate(ordered):
            with torch.no_grad():
                renderer.collect_contributions=True
                package=renderer(view,module.background)
                observed=package.get('motion_contribution_filter')
                if observed is None or observed.shape!=(len(xyz),):
                    raise ValueError('Responsibility selection requires native actual contribution masks')
                observed=observed.detach().bool()
                residual=(package['render']-view.image).detach()
                error=residual.square()
                if residual_floor:
                    mse=error.mean(0,keepdim=True)
                    error=error*((mse-float(residual_floor)**2).clamp_min(0)/mse.clamp_min(1e-30))
                error_sum=float(error.sum())
            renderer.collect_contributions=False
            # This forward is zero (black background), but its colour Jacobian
            # is the real full-scene alpha/T response. Squared residual prevents
            # opposite signs or already-converged DC gradients cancelling.
            with torch.enable_grad():
                probe=torch.zeros_like(xyz,requires_grad=True)
                zero_rgb=renderer(view,torch.zeros_like(module.background),override_color=probe)['render']
                objective=(zero_rgb*error).sum()
                gradient=torch.autograd.grad(objective,probe,only_inputs=True,
                    retain_graph=collect_colour_evidence)[0]
                colour_evidence={}
                if collect_colour_evidence:
                    # Reuse the same native colour Jacobian: no extra render.
                    signed=torch.autograd.grad((zero_rgb*(-residual)).sum(),probe,
                        retain_graph=True,only_inputs=True)[0]
                    mass=torch.autograd.grad(zero_rgb.sum(),probe,only_inputs=True)[0].mean(-1)
                    colour_evidence=dict(native_mass=mass.detach().clamp_min(0),
                                         signed_response=signed.detach())
            current=gradient.detach().sum(-1)
            if not bool(torch.isfinite(current).all()):
                raise ValueError('Native residual responsibility is nonfinite')
            # The exact derivative is nonnegative; clamp only arithmetic noise.
            current=current.clamp_min(0)
            scores+=current
            observed_bits|=observed.long()*(1<<index)
            responsibility_bits|=(current>0).long()*(1<<index)
            total_error+=error_sum
            per_view.append(dict(camera=name,old_RGB_SSE=error_sum,
                native_responsibility_SSE=float(current.sum()),
                observed_old_count=int(observed.sum()),positive_responsibility_count=int((current>0).sum())))
            cache.append(dict(view=view,name=name,residual=residual,image=view.image.detach(),
                              observed=observed,responsibility=current,**colour_evidence))
    finally:
        renderer.gaussians=previous_model
        renderer.collect_contributions=previous_collect
        if has_last:renderer.last_contributions=previous_last
    report=dict(training_camera_names=[name for name,_ in ordered],external_test_used=False,
        old_RGB_SSE=total_error,native_responsibility_SSE=float(scores.sum()),
        native_responsibility_fraction_of_total_SSE=float(scores.sum())/max(total_error,1e-30),
        observed_old_count=int(observed_bits.ne(0).sum()),
        positive_responsibility_count=int(scores.gt(0).sum()),per_view=per_view,
        score_semantics='sum over standard train RGB pixels of actual native T*alpha_i * residual_RGB^2',
        coverage_semantics='additive error-responsibility mass, not bbox area or union of covered pixels',
        source_scale_limit=None,signed_DC_gradient_used=False,mutation_before_transaction=False,
        native_render_calls=2*len(ordered),
        native_colour_backward_calls=(3 if collect_colour_evidence else 1)*len(ordered))
    return scores,observed_bits,responsibility_bits,report,cache


def score_support_responsibility(module,views,residual_floor=0.):
    """Return scores[N], actual-observation bits, positive-error bits, report.

    Exactly one old RGB and one differentiable zero-colour native pass per view.
    Geometry, alpha and ordering remain the full incumbent in both passes.
    Only the temporary colour leaf receives an autograd gradient.
    """
    return _native_responsibility(module,views,residual_floor=residual_floor)[:4]


def _rotation(q):
    w,x,y,z=F.normalize(q,dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


def _sample(image,grid):
    shape=grid.shape[:-1]
    values=F.grid_sample(image[None],grid.reshape(1,-1,1,2),align_corners=False,
                         padding_mode='zeros')[0,:,:,0].T
    return values.reshape(*shape,image.shape[0])


def _support_evidence(base,rows,cache):
    count=len(rows);xyz=base['xyz'][rows];scale=base['logscale'][rows].exp()
    normal=scale.argmin(-1)
    axes=torch.arange(3,device=xyz.device).expand(count,3)
    tangent=axes[axes!=normal[:,None]].reshape(count,2)
    basis=_rotation(base['rotation'][rows]).gather(2,tangent[:,None].expand(-1,3,-1))
    sigma=scale.gather(1,tangent)
    lattice=xyz.new_tensor([[x,y] for y in (-1.,0.,1.) for x in (-1.,0.,1.)])
    world=xyz[:,None]+torch.einsum('fij,fpj->fpi',basis,lattice[None]*sigma[:,None])
    color_sum=xyz.new_zeros(count,9,3);residual_sum=torch.zeros_like(color_sum)
    seen=xyz.new_zeros(count,9);energy=xyz.new_zeros(count,9)
    pattern_sum=torch.zeros_like(color_sum);pattern_square=torch.zeros_like(color_sum)
    kernel=torch.exp(-.5*lattice.square().sum(-1))
    residual_scale=xyz.new_zeros(count)
    per_view=[]
    for item in cache:
        view=item['view'];matrix=view.world_view_transform.to(world)
        camera=world@matrix[:3,:3]+matrix[3,:3];z=camera[...,2]
        grid=torch.stack((camera[...,0]/z.clamp_min(1e-12)/math.tan(float(view.fovx)*.5),
                          camera[...,1]/z.clamp_min(1e-12)/math.tan(float(view.fovy)*.5)),-1)
        valid=(grid.abs()<1).all(-1)&(z>0)&item['observed'][rows,None]
        color=_sample(item['image'],grid);residual=_sample(item['residual'],grid)
        color_sum+=color*valid[...,None];residual_sum+=residual*valid[...,None]
        energy+=residual.square().sum(-1)*valid;seen+=valid
        weight=valid*kernel[None]
        mean=(residual*weight[...,None]).sum(1)/weight.sum(-1).clamp_min(1e-30)[:,None]
        centered=residual-mean[:,None]
        pattern_sum+=centered*valid[...,None];pattern_square+=centered.square()*valid[...,None]
        residual_scale=torch.maximum(residual_scale,(residual.abs()*valid[...,None]).amax((1,2)))
        per_view.append(valid)
    color=color_sum/seen.clamp_min(1)[...,None]
    residual=residual_sum/seen.clamp_min(1)[...,None]
    spatial=torch.exp(-.5*lattice.square().sum(-1))[None]*seen
    pattern=pattern_sum/seen.clamp_min(1)[...,None]
    variance=(pattern_square/seen.clamp_min(1)[...,None]-pattern.square()).clamp_min(0)
    coherent=(pattern.square()-variance/seen.clamp_min(1)[...,None]).clamp_min(0)
    coherent*=seen.ge(2)[...,None]
    arithmetic=(torch.finfo(xyz.dtype).eps*(len(cache)+len(lattice)+8)*residual_scale).square()
    return dict(world=world,color=color,residual=residual,energy=energy/seen.clamp_min(1),
                mass=spatial/spatial.sum(-1,keepdim=True).clamp_min(1e-30),counts=seen,
                tangent_axes=tangent,per_view=per_view,pattern=pattern,coherent=coherent,
                arithmetic_bound=arithmetic,local=lattice[None]*sigma[:,None])


def _split_direction(probes,index):
    """A deterministic inherited-support direction, not a geometry fit."""
    local=probes['local'][index].detach().cpu();pattern=probes['pattern'][index].detach().cpu()
    mass=probes['mass'][index].detach().cpu();count=probes['counts'][index].detach().cpu()
    coherent=probes['coherent'][index].detach().cpu().mean(-1)
    bound=float(probes['arithmetic_bound'][index])
    evidence=float((coherent*mass).sum())
    ids=((count>=2)&(mass>0)).nonzero().flatten()
    if len(ids)<2 or evidence<=bound:
        return False,local.new_zeros(2),evidence,'no_observable_coherent_pattern',0.
    weight=mass[ids];total=weight.sum()
    features=torch.cat((local[ids],pattern[ids]),-1)
    mean=(pattern[ids]*weight[:,None]).sum(0)/total
    before=((pattern[ids]-mean).square().sum(-1)*weight).sum()
    best=None
    for axis in range(features.shape[1]):
        ordered=ids[features[:,axis].argsort(stable=True)]
        cumulative=mass[ordered].cumsum(0)
        cut=max(1,min(len(ordered)-1,int(torch.searchsorted(cumulative,cumulative[-1]*.5))+1))
        cost=weight.new_zeros(());positions=[]
        for part in (ordered[:cut],ordered[cut:]):
            w=mass[part];s=w.sum()
            color=(pattern[part]*w[:,None]).sum(0)/s
            cost+=((pattern[part]-color).square().sum(-1)*w).sum()
            positions.append((local[part]*w[:,None]).sum(0)/s)
        gain=float((before-cost)/total)
        direction=(positions[0]-positions[1])*.5
        if best is None or gain>best[0]:best=(gain,direction)
    eligible=best[0]>bound and bool(best[1].square().sum()>0)
    sigma=local.abs().amax(0)
    rho2=float((best[1]/sigma).square().sum())
    if eligible and rho2>=1:
        return False,best[1],evidence,'outside_family_one_sigma_domain',rho2
    return (eligible,best[1] if eligible else local.new_zeros(2),evidence,
        'supported' if eligible else 'no_explained_spatial_pattern',rho2)


@torch.no_grad()
def propose_support_reincarnations(module,views,source_quota=128,prefilter=512):
    """Native responsibility selection and complete 1:1 raw parent copies.

    ``spec['raw']`` has no batch dimension. Copying all specs while removing
    their unique parents preserves the initial geometry/alpha/SH scene exactly.
    Fine refinement may use the inherited nine-probe lattice, but must not call
    that lattice a measured surface, per-probe alpha/T map, or native coverage.
    """
    for name,value in (('source_quota',source_quota),('prefilter',prefilter)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    if prefilter<source_quota:raise ValueError('Prefilter must contain the requested source quota')
    scores,observed,responsibility,report,cache=_native_responsibility(module,views)
    valid=(scores>0)&observed.ne(0)
    candidates=valid.nonzero().flatten()
    ordered=candidates[scores[candidates].argsort(descending=True,stable=True)]
    prefiltered=ordered[:prefilter];rows=prefiltered[:source_quota]
    base=model_raw(module.gaussians)
    probes=_support_evidence(base,rows,cache) if len(rows) else None
    specs=[]
    for index,row in enumerate(rows.tolist()):
        mass=probes['mass'][index]*scores[row]
        split_eligible,offset,contrast_evidence,split_reason,rho2=_split_direction(probes,index)
        specs.append(dict(old_row=row,raw={name:base[name][row].detach().clone() for name in RAW_FIELDS},
            source_row=row,responsibility=float(scores[row]),observed_view_bits=int(observed[row]),
            responsibility_view_bits=int(responsibility[row]),support_world=probes['world'][index].clone(),
            support_color=probes['color'][index].clone(),support_mass=mass.clone(),
            support_residual=probes['residual'][index].clone(),support_error=probes['energy'][index].clone(),
            support_view_count=probes['counts'][index].clone(),support_axes=probes['tangent_axes'][index].clone(),
            split_eligible=split_eligible,tangent_offset=offset.to(base['xyz']),
            contrast_evidence=contrast_evidence,split_reason=split_reason,split_offset_rho_squared=rho2,
            support_kind='inherited_covariance_not_new_surface',
            support_semantics='nine prior probes with source-level native observation; not per-probe Talpha responsibility or rendered cover'))
    selected_mass=float(scores[rows].sum())
    report.update(prefilter_budget=prefilter,source_quota=source_quota,
        prefilter_count=len(prefiltered),selected_count=len(rows),
        prefilter_rows=prefiltered.detach().cpu().tolist(),selected_rows=rows.detach().cpu().tolist(),
        selected_native_responsibility_SSE=selected_mass,
        selected_fraction_of_total_SSE=selected_mass/max(report['old_RGB_SSE'],1e-30),
        selected_fraction_of_native_responsibility=selected_mass/max(report['native_responsibility_SSE'],1e-30),
        split_eligible_count=sum(s['split_eligible'] for s in specs),
        unsupported_detail_offsets_kept_coarse=sum(s['split_reason']=='outside_family_one_sigma_domain' for s in specs),
        excluded_exact_zero=int(scores.eq(0).sum()),excluded_unobserved=int(observed.eq(0).sum()),
        initial_semantics='complete original raw support/opacity/SH copied; source retires only after atomic group acceptance',
        rendering_equivalence_caveat='Native equal-depth stable-sort ordering can change on delete-and-append even with identical raw attributes')
    return specs,report


@torch.no_grad()
def camera_complementary_groups(views):
    """Deterministic geometric 10/10-style complementary cross-fit groups.

    Pair nearby camera poses, put one member in each group, then choose pair
    orientation to balance the spatial/directional feature sums. Every training
    camera belongs to exactly one group; neither group is held out forever.
    """
    ordered=_ordered_views(views)
    if len(ordered)<2:raise ValueError('Complementary camera groups need at least two views')
    centers=[];forward=[]
    for _,view in ordered:
        inverse=torch.linalg.inv(view.world_view_transform.detach().cpu().double())
        centers.append(inverse[3,:3]);forward.append(F.normalize(inverse[2,:3],dim=0))
    centers=torch.stack(centers);direction=torch.stack(forward)
    centered=centers-centers.mean(0)
    radius=centered.norm(dim=-1).median().clamp_min(torch.finfo(centers.dtype).eps)
    features=torch.cat((centered/radius,direction),-1)
    distance=torch.cdist(features,features).square()
    remaining=list(range(len(ordered)));pairs=[]
    while len(remaining)>1:
        pairs_left=[(float(distance[a,b]),a,b) for p,a in enumerate(remaining) for b in remaining[p+1:]]
        _,a,b=min(pairs_left);pairs.append((a,b));remaining.remove(a);remaining.remove(b)
    groups=[[],[]];sums=features.new_zeros(2,features.shape[1])
    for a,b in pairs:
        straight=(sums[0]+features[a]-sums[1]-features[b]).square().sum()
        swapped=(sums[0]+features[b]-sums[1]-features[a]).square().sum()
        if swapped<straight:a,b=b,a
        groups[0].append(a);groups[1].append(b);sums[0]+=features[a];sums[1]+=features[b]
    if remaining:
        target=0 if len(groups[0])<=len(groups[1]) else 1
        groups[target].append(remaining[0]);sums[target]+=features[remaining[0]]
    names=tuple(tuple(ordered[i][0] for i in sorted(group)) for group in groups)
    report=dict(groups=[list(g) for g in names],all_training_cameras_used=True,external_test_used=False,
        pairing=[[ordered[a][0],ordered[b][0]] for a,b in pairs],
        feature_mean_difference_norm=float((sums[0]/len(groups[0])-sums[1]/len(groups[1])).norm()),
        semantics='nearby camera pose pairs split across complementary training groups; native GT/test metrics never choose the split')
    return names,report
