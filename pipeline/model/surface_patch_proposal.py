"""Training-evidence surface domains spanning multiple incumbent Gaussians.

This module is a CPU proposal stage, not a renderer or an admission policy.
Native contribution masks and rendered depth/normal *moments* are supplied by
the caller. They are inherited surface evidence, never ground-truth geometry.
Region-assigned pixel area below must not be reported as actual rendered cover.
"""
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from libgs.utils.sh import eval_sh


_CELL = 8
_NORMAL_COS = math.cos(math.pi/6)
_C0 = .28209479177387814


def _cpu(value, dtype=None):
    return torch.as_tensor(value).detach().to(device='cpu', dtype=dtype).clone()


def _rotation(q):
    w,x,y,z = F.normalize(q,dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)


def _project(cache, xyz):
    m=cache['matrix'];p=xyz@m[:3,:3]+m[3,:3];z=p[:,2]
    grid=torch.stack((p[:,0]/z.clamp_min(1e-12)/cache['tanx'],
                      p[:,1]/z.clamp_min(1e-12)/cache['tany']),-1)
    return grid,z


def _sample(image,grid):
    if image.ndim==2:image=image[None]
    return F.grid_sample(image[None],grid[None,:,None],align_corners=False,
        padding_mode='zeros')[0,:,:,0].T


def _unproject(cache,pixels,depth):
    grid=(2*pixels+1)/pixels.new_tensor([cache['width'],cache['height']])-1
    p=torch.stack((grid[:,0]*cache['tanx']*depth,grid[:,1]*cache['tany']*depth,depth),-1)
    inv=cache['inverse'];return p@inv[:3,:3]+inv[3,:3]


def _pooled_sum(image):
    average=F.avg_pool2d(image[None,None],_CELL,_CELL,ceil_mode=True,
                         count_include_pad=False)[0,0]
    height,width=image.shape
    rows=(height-torch.arange(average.shape[0],device=image.device)*_CELL).clamp_max(_CELL)
    cols=(width-torch.arange(average.shape[1],device=image.device)*_CELL).clamp_max(_CELL)
    return average*rows[:,None]*cols[None]


def _cache_evidence(item,count,dtype,index):
    view=item['view'];name=Path(str(getattr(view,'path',f'train_{index}'))).name
    if name=='cam00.png' or item.get('is_test',False):
        raise ValueError('Surface proposals only accept standard training cameras')
    if 'observed_old' not in item:
        raise ValueError('Surface proposals require native old contribution evidence')
    image=_cpu(item.get('current_gt',view.image),dtype)
    residual=item.get('residual_rgb')
    if residual is None:
        residual=_cpu(item['rendered'],dtype)-image
    else:residual=_cpu(residual,dtype)
    if residual.ndim==3:energy=residual.square().mean(0)
    elif residual.ndim==2:energy=residual.square()
    else:raise ValueError('Residual must have HxW or RGBxHxW shape')
    previous=item.get('previous_image')
    change=torch.zeros_like(energy) if previous is None else (image-_cpu(previous,dtype)).square().mean(0)
    # Persistent low-frequency error retains its complete DC component. A
    # temporal change reduces priority softly; it does not create an ROI.
    mass=energy*energy/(energy+change+torch.finfo(dtype).tiny)
    height,width=energy.shape
    m=_cpu(view.world_view_transform,dtype)
    cache=dict(name=name,image=image,energy=energy,mass=mass,
        depth=_cpu(item['depth'],dtype).reshape(height,width),
        coverage=_cpu(item['coverage'],dtype).reshape(height,width),
        normal=_cpu(item['normal'],dtype).reshape(3,height,width),
        variance=_cpu(item.get('depth_variance',torch.zeros_like(energy)),dtype).reshape(height,width).clamp_min(0),
        observed=_cpu(item['observed_old']).bool().flatten(),
        matrix=m,inverse=torch.linalg.inv(m),tanx=math.tan(float(view.fovx)*.5),
        tany=math.tan(float(view.fovy)*.5),width=width,height=height,
        temporal_available=previous is not None)
    if cache['observed'].shape!=(count,):raise ValueError('Native old contribution mask has wrong length')
    cache['focal']=width/(2*cache['tanx'])
    return cache


def _surface_cells(cache):
    mass=_pooled_sum(cache['mass']);h,w=cache['mass'].shape
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    cx=_pooled_sum(cache['mass']*xx)/mass.clamp_min(1e-30)
    cy=_pooled_sum(cache['mass']*yy)/mass.clamp_min(1e-30)
    cells=(mass>0).nonzero();pixels=torch.stack((cx[cells[:,0],cells[:,1]],cy[cells[:,0],cells[:,1]]),-1)
    grid=(2*pixels+1)/pixels.new_tensor([w,h])-1
    depth=_sample(cache['depth'],grid)[:,0]
    variance=_sample(cache['variance'],grid)[:,0]
    coverage=_sample(cache['coverage'],grid)[:,0]
    normals=_sample(cache['normal'],grid);norm=normals.norm(dim=-1)
    footprint=depth/cache['focal']*_CELL
    valid=(depth>0)&torch.isfinite(depth)&torch.isfinite(norm)&(coverage>torch.finfo(depth.dtype).eps)
    valid&=(norm>=_NORMAL_COS)&(variance<=(footprint*.5).square())
    cells,pixels,depth,variance,footprint,normals=[x[valid] for x in (cells,pixels,depth,variance,footprint,normals)]
    if not len(cells):return None
    normals=F.normalize(normals,dim=-1);points=_unproject(cache,pixels,depth)
    weights=mass[cells[:,0],cells[:,1]]
    row_size=(h-cells[:,0]*_CELL).clamp_max(_CELL)
    col_size=(w-cells[:,1]*_CELL).clamp_max(_CELL)
    lookup={tuple(cell):i for i,cell in enumerate(cells.tolist())}
    p,n,fp=points.tolist(),normals.tolist(),footprint.tolist()
    remaining=set(range(len(cells)));groups=[]
    # Neighbouring pixels are connected only if both local planes agree.
    for first in range(len(cells)):
        if first not in remaining:continue
        remaining.remove(first);pending=[first];part=[]
        while pending:
            i=pending.pop();part.append(i);y,x=cells[i].tolist()
            for key in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                j=lookup.get(key)
                if j not in remaining:continue
                cosine=abs(sum(a*b for a,b in zip(n[i],n[j])))
                d=[a-b for a,b in zip(p[i],p[j])]
                distance=max(abs(sum(a*b for a,b in zip(d,n[i]))),abs(sum(a*b for a,b in zip(d,n[j]))))
                if cosine<_NORMAL_COS or distance>(fp[i]+fp[j])*.5:continue
                remaining.remove(j);pending.append(j)
        if len(part)>=3:groups.append(torch.tensor(part,dtype=torch.long))
    return dict(cells=cells,pixels=pixels,xyz=points,normal=normals,weight=weights,
        footprint=footprint,area=(row_size*col_size).to(depth),groups=groups,
        color=_sample(cache['image'],grid[valid]))


def _plane(points,normals,weights):
    weights=weights/weights.sum().clamp_min(1e-30)
    aligned=normals*torch.where((normals*normals[:1]).sum(-1,keepdim=True)<0,-1.,1.)
    normal=F.normalize((aligned*weights[:,None]).sum(0),dim=0)
    center=(points*weights[:,None]).sum(0)
    seed=F.one_hot(normal.abs().argmin(),3).to(points)
    tangent=F.normalize(seed-normal*(seed*normal).sum(),dim=0)
    frame=torch.stack((tangent,torch.cross(normal,tangent,dim=-1),normal),-1)
    uv=(points-center)@frame[:,:2]
    covariance=(uv*weights[:,None]).T@uv
    _,evec=torch.linalg.eigh(covariance)
    t0=frame[:,:2]@evec[:,-1]
    t0=t0*torch.where((t0*tangent).sum()<0,-1.,1.)
    frame=torch.stack((t0,torch.cross(normal,t0,dim=-1),normal),-1)
    return center,frame


def _planar_parts(data,ids):
    pending=[ids];result=[]
    while pending:
        ids=pending.pop()
        if len(ids)<3:continue
        xyz=data['xyz'][ids];normal=data['normal'][ids];weight=data['weight'][ids]
        center,frame=_plane(xyz,normal,weight)
        uv=(xyz-center)@frame[:,:2];error=((xyz-center)*frame[:,2]).sum(-1).abs()
        compatible=(normal@frame[:,2]).abs()>=_NORMAL_COS
        planar=bool((error<=data['footprint'][ids]*.5).all() and compatible.all())
        if planar:
            # A line of samples is not a two-dimensional supported patch.
            spread=torch.linalg.eigvalsh((uv-uv.mean(0)).T@(uv-uv.mean(0))/len(ids))
            if spread[0]>data['footprint'][ids].median().square()/48:
                result.append(ids)
        elif len(ids)>=6:
            axis=(uv.amax(0)-uv.amin(0)).argmax();ordered=ids[uv[:,axis].argsort(stable=True)]
            cut=len(ids)//2;pending.extend((ordered[:cut],ordered[cut:]))
    return result


def _visible_support(cache,points,normal):
    grid,z=_project(cache,points)
    depth=_sample(cache['depth'],grid)[:,0];variance=_sample(cache['variance'],grid)[:,0]
    normals=_sample(cache['normal'],grid);norm=normals.norm(dim=-1)
    coverage=_sample(cache['coverage'],grid)[:,0]
    tolerance=2*z.abs()/cache['focal']+3*variance.sqrt()
    valid=(grid.abs()<1).all(-1)&(z>0)&(depth>0)&(coverage>torch.finfo(z.dtype).eps)
    valid&=(z-depth).abs()<=tolerance
    valid&=(norm>=_NORMAL_COS)&((F.normalize(normals,dim=-1)*normal).sum(-1).abs()>=_NORMAL_COS)
    # High depth spread cannot certify a single opaque sheet (e.g. flame).
    valid&=variance<=(z.abs()/cache['focal']*_CELL*.5).square()
    return valid,_sample(cache['image'],grid)


def _photo_consistent(reference,other):
    delta=other-reference
    # Separate camera-wide offset from structure; bounded material agreement
    # is still required, so a specular/transparent patch cannot use HDR to pass.
    offset=delta.median(0).values
    centered=delta-offset
    structure=(reference-reference.mean(0)).square().mean()+(other-other.mean(0)).square().mean()
    return bool(offset.abs().max()<=.1 and centered.square().mean()<=.04**2+.25*structure)


def _certify(points,colors,normal,caches):
    # Uniformly sample the whole support for cross-view certification rather
    # than approving a large domain from one matching centre pixel.
    ids=torch.linspace(0,len(points)-1,min(128,len(points))).round().long().unique()
    points,colors=points[ids],colors[ids];support=[]
    for cache in caches:
        valid,other=_visible_support(cache,points,normal)
        if int(valid.sum())<max(3,math.ceil(len(points)/4)):continue
        if _photo_consistent(colors[valid],other[valid]):support.append(cache['name'])
    return support


def _ownership(base,old_normals,cache,data,ids,center,frame):
    xyz=base['xyz'];grid,z=_project(cache,xyz)
    pixels=((grid+1)*grid.new_tensor([cache['width'],cache['height']])*.5).floor().long()
    cell_y=(pixels[:,1]//_CELL);cell_x=(pixels[:,0]//_CELL)
    cell_mask=torch.zeros((math.ceil(cache['height']/_CELL),math.ceil(cache['width']/_CELL)),dtype=torch.bool)
    cells=data['cells'][ids];cell_mask[cells[:,0],cells[:,1]]=True
    valid=(grid.abs()<1).all(-1)&(z>0)&cache['observed']
    valid&=cell_mask[cell_y.clamp(0,cell_mask.shape[0]-1),cell_x.clamp(0,cell_mask.shape[1]-1)]
    rows=valid.nonzero().flatten()
    if not len(rows):return rows
    support,_=_visible_support(cache,xyz[rows],frame[:,2])
    sigma=base['logscale'][rows].exp().amin(-1)
    plane_error=((xyz[rows]-center)*frame[:,2]).sum(-1).abs()
    support&=plane_error<=data['footprint'][ids].median()*.5+3*sigma
    support&=(old_normals[rows]@frame[:,2]).abs()>=_NORMAL_COS
    return rows[support]


def _merge_candidates(candidates):
    parents=list(range(len(candidates)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(candidates):
        for j in range(i):
            b=candidates[j]
            if not a['old_set'].intersection(b['old_set']):continue
            if abs(float(a['frame'][:,2]@b['frame'][:,2]))<_NORMAL_COS:continue
            distance=max(abs(float((a['center']-b['center'])@a['frame'][:,2])),
                         abs(float((a['center']-b['center'])@b['frame'][:,2])))
            if distance>min(float(a['footprint'].median()),float(b['footprint'].median()))*.5:continue
            parents[root(i)]=root(j)
    groups={}
    for i,candidate in enumerate(candidates):groups.setdefault(root(i),[]).append(candidate)
    merged=[]
    for members in groups.values():
        item={key:torch.cat([c[key] for c in members]) for key in ('xyz','normal','weight','color','footprint','area')}
        item['old_set']=set().union(*(c['old_set'] for c in members))
        item['views']=set().union(*(c['views'] for c in members))
        item['center'],item['frame']=_plane(item['xyz'],item['normal'],item['weight'])
        # A chain of locally compatible charts must not become a forced plane.
        error=((item['xyz']-item['center'])*item['frame'][:,2]).sum(-1).abs()
        if bool((error<=item['footprint']*.5).all()):merged.append(item)
        else:merged.extend(members)
    return merged


def _farthest(points,weights,count):
    first=int(weights.argmax());chosen=[first]
    distance=(points-points[first]).square().sum(-1)
    while len(chosen)<count:
        distance[chosen]=-1
        index=int((distance*(weights/weights.max().clamp_min(1e-30)).sqrt()).argmax())
        if float(distance[index])<=torch.finfo(points.dtype).eps**2:break
        chosen.append(index);distance=torch.minimum(distance,(points-points[index]).square().sum(-1))
    return torch.tensor(chosen,dtype=torch.long)


def _materialize(candidate,budget,base,old_normals,caches):
    center,frame=candidate['center'],candidate['frame'];points=candidate['xyz']
    uv=(points-center)@frame[:,:2];weight=candidate['weight'];footprint=candidate['footprint']
    chosen=_farthest(uv,weight,min(budget,len(uv)))
    kernel_uv=uv[chosen]
    assignment=torch.cat([torch.cdist(part,kernel_uv).argmin(-1) for part in uv.split(1024)])
    count=len(chosen);total=weight.new_zeros(count).scatter_add_(0,assignment,weight)
    sums=uv.new_zeros(count,2).scatter_add_(0,assignment[:,None].expand(-1,2),uv*weight[:,None])
    centers=sums/total.clamp_min(1e-30)[:,None]
    variance=uv.new_zeros(count,2).scatter_add_(0,assignment[:,None].expand(-1,2),(uv-centers[assignment]).square()*weight[:,None])
    variance/=total.clamp_min(1e-30)[:,None]
    pixel_area=weight.new_zeros(count).scatter_add_(0,assignment,footprint.square()*weight)/total.clamp_min(1e-30)
    # Uniform-cell second moment gives sigma=half-cell; footprint is justified
    # by observed depth pixels, not extrapolated from a large image rectangle.
    scales=(3*variance+pixel_area[:,None]/4).sqrt()
    old_rows=torch.tensor(sorted(candidate['old_set']),dtype=torch.long)
    thickness=base['logscale'][old_rows].exp().amin(-1).median().clamp_min(torch.finfo(uv.dtype).eps)
    kernel_xyz=center+centers@frame[:,:2].T
    colors=[];valid_views=[]
    for cache in caches:
        valid,color=_visible_support(cache,kernel_xyz,frame[:,2])
        if cache['name'] not in candidate['views']:valid[:]=False
        valid_views.append(valid)
        colors.append(color.masked_fill(~valid[:,None],float('nan')))
    colors=torch.stack(colors)
    color=colors.nanmedian(0).values
    fallback=candidate['color'][chosen]
    color=torch.where(torch.isfinite(color),color,fallback).clamp(0,1)
    distances,neighbours=[],[]
    for part in kernel_xyz.split(64):
        values,rows=torch.cdist(part,base['xyz'][old_rows]).topk(min(3,len(old_rows)),largest=False)
        distances.append(values);neighbours.append(old_rows[rows])
    distances,material_rows=torch.cat(distances),torch.cat(neighbours)
    material_weight=1/distances.clamp_min(torch.finfo(uv.dtype).eps)
    material_weight/=material_weight.sum(-1,keepdim=True)
    # Material provenance is fixed in physical surface coordinates. A later
    # codec predictor reassignment must not replace this inherited SH prior.
    sources=material_rows[:,0]
    rest=(base['sh_rest'][material_rows]*material_weight[...,None,None]).sum(1)
    degree=math.isqrt(rest.shape[1]+1)-1
    if (degree+1)**2!=rest.shape[1]+1:
        raise ValueError('Inherited surface SH coefficient count is not a complete degree')
    sh=torch.cat((rest.new_zeros((count,1,3)),rest),1).transpose(1,2)
    dc_estimates=[]
    for index,cache in enumerate(caches):
        direction=F.normalize(kernel_xyz-cache['inverse'][3,:3],dim=-1)
        angular=eval_sh(degree,sh,direction)
        estimate=(colors[index]-.5-angular)/_C0
        dc_estimates.append(estimate.masked_fill(~valid_views[index][:,None],float('nan')))
    dc=torch.stack(dc_estimates).nanmedian(0).values
    # A centre can land between observed support samples. In that case retain
    # the actual physical material prior, never add SH to an RGB-derived DC.
    fallback_dc=(base['sh_dc'][material_rows]*material_weight[...,None,None]).sum(1)[:,0]
    prior_dc=torch.where(torch.isfinite(dc),dc,fallback_dc)[:,None]
    margin=footprint[:,None]*.5
    bounds=torch.stack(((uv-margin).amin(0),(uv+margin).amax(0)))
    error=((points-center)*frame[:,2]).sum(-1).abs()
    return dict(center=center.clone(),frame=frame.clone(),normal=frame[:,2].clone(),
        uv_bounds=bounds,thickness=thickness,kernel_uv=centers,kernel_scale=scales,
        kernel_color=color,kernel_dc=prior_dc.clone(),kernel_sh_rest=rest,
        prior_dc=prior_dc,prior_sh_rest=rest.clone(),
        old_rows=old_rows,source_rows=sources,material_rows=material_rows,
        material_weights=material_weight,score=float(weight.sum()/len(caches)),
        support_uv=uv.clone(),support_world=points.clone(),support_weights=weight.clone(),
        support_color=candidate['color'].clone(),support_mass=weight.clone(),
        diagnostics=dict(evidence='inherited_multiview_surface_moments_and_RGB',
            support_views=sorted(candidate['views']),old_anchor_count=len(old_rows),
            assigned_support_pixels=float(candidate['area'].sum()),
            assigned_error_mass=float(weight.sum()),plane_error_max=float(error.max()),
            plane_error_rms=float((error.square()*weight).sum().div(weight.sum()).sqrt()),
            actual_native_new_coverage=None,
            area_scope='summed_training_view_cell_assignment_not_native_rendered_coverage',
            color_prior='angular_corrected_multiview_DC_and_trusted_surface_SH_rest_interpolation',
            codec_source_is_material_prior=False))


@torch.no_grad()
def build_surface_patches(base_raw,view_evidence,max_kernels,max_patches=None):
    """Return ``(patch_specs, report)`` from standard training evidence only.

    Each evidence dict contains view, current_gt, previous_image, residual_rgb,
    coverage, depth, normal, depth_variance, observed_old. ``depth``/``normal``
    are native alpha/T-weighted moments, and observed_old must be a native
    contribution mask. Inputs may reside on GPU; this stage explicitly copies
    detached evidence to CPU and never mutates the scene or camera datasets.
    Independent novel/MVS proposals can be appended by the caller; a rejected
    opaque-sheet hypothesis must retain its incumbent carriers.
    """
    if isinstance(max_kernels,bool) or not isinstance(max_kernels,int) or max_kernels<1:
        raise ValueError('max_kernels must be a positive integer')
    if max_patches is None:max_patches=max(1,min(32,max_kernels//4))
    if isinstance(max_patches,bool) or not isinstance(max_patches,int) or max_patches<1:
        raise ValueError('max_patches must be a positive integer')
    base={name:_cpu(value) for name,value in base_raw.items()}
    dtype=base['xyz'].dtype;count=len(base['xyz'])
    caches=[_cache_evidence(e,count,dtype,i) for i,e in enumerate(view_evidence.values() if isinstance(view_evidence,dict) else view_evidence)]
    if len(caches)<3:raise ValueError('A surface patch needs evidence from at least three training views')
    if len({c['name'] for c in caches})!=len(caches):raise ValueError('Training evidence camera names must be unique')
    R=_rotation(base['rotation']);normal_axis=base['logscale'].argmin(-1)
    old_normals=R.gather(2,normal_axis[:,None,None].expand(-1,3,1)).squeeze(-1)
    candidates=[];report=dict(training_camera_names=[c['name'] for c in caches],
        external_test_used=False,input_old_count=count,geometric_regions=0,
        rejected_without_multiple_old=0,rejected_crossview_surface=0,
        allocated_kernels=0,selected_patches=0,
        area_scope='training-view observed cell assignments, not actual new render coverage',
        persistent_mean_residual_included=True,
        temporal_evidence_views=sum(c['temporal_available'] for c in caches))
    for cache in caches:
        data=_surface_cells(cache)
        if data is None:continue
        parts=[part for group in data['groups'] for part in _planar_parts(data,group)]
        report['geometric_regions']+=len(parts)
        parts.sort(key=lambda ids:float(data['weight'][ids].sum()),reverse=True)
        for ids in parts[:max(8,max_patches*2)]:
            center,frame=_plane(data['xyz'][ids],data['normal'][ids],data['weight'][ids])
            old=_ownership(base,old_normals,cache,data,ids,center,frame)
            if len(old)<3:
                report['rejected_without_multiple_old']+=1;continue
            views=_certify(data['xyz'][ids],data['color'][ids],frame[:,2],caches)
            if len(views)<3:
                report['rejected_crossview_surface']+=1;continue
            item={key:data[key][ids].clone() for key in ('xyz','normal','weight','color','footprint','area')}
            item.update(center=center,frame=frame,old_set=set(old.tolist()),views=set(views))
            candidates.append(item)
    candidates=_merge_candidates(candidates)
    candidates.sort(key=lambda c:float(c['weight'].sum()),reverse=True)
    selected=[];owned=set()
    for candidate in candidates:
        if candidate['old_set'].intersection(owned):continue
        selected.append(candidate);owned.update(candidate['old_set'])
        if len(selected)>=min(max_patches,max(1,max_kernels//3)):break
    if not selected:return [],report
    # Every chart gets three kernels; remaining capacity follows integrated
    # residual mass across the chart, never the number of source GS gradients.
    minimum=min(3,max_kernels//len(selected));budgets=[minimum]*len(selected)
    remaining=max_kernels-sum(budgets)
    scores=torch.tensor([float(c['weight'].sum()) for c in selected],dtype=torch.float64)
    shares=remaining*scores/scores.sum();extra=shares.floor().long()
    for i in range(remaining-int(extra.sum())):extra[int((shares-extra).argmax())]+=1
    budgets=[b+int(e) for b,e in zip(budgets,extra)]
    specs=[_materialize(c,b,base,old_normals,caches) for c,b in zip(selected,budgets)]
    report['allocated_kernels']=sum(len(s['kernel_uv']) for s in specs)
    report['selected_patches']=len(specs)
    report['owned_old_count']=len(owned)
    report['patch_scores']=[s['score'] for s in specs]
    report['diagnostics']=[s['diagnostics'] for s in specs]
    return specs,report
