"""One source-family shadow colour/density proposal, never SGD or admission.

Only the accepted prefix and this family appear in native probes. A bounded
2x2 child DC block is fitted once. Three fixed-colour images define a quadratic
image surrogate and a quartic MSE. Its cubic derivative is solved once. Native
skip/termination/output clamps make this only a proposal; the controller must
perform the single actual-codec error/byte check before accepting the family.
"""
import numpy as np
import torch

from .birth_radiance import _isolated_renderer,_preclamp_rgb,_view_list,Y00
from .child_dc_initializer import _pair_image_moments,solve_box2
from .residual_birth import RawView


def density_coefficients(i0,ihalf,i1,target):
    """Mean RGB error polynomial in ascending order c0..c4; raw RGB only."""
    x,half,one,y=[torch.as_tensor(z).detach().double() for z in (i0,ihalf,i1,target)]
    if not x.numel() or any(z.shape!=x.shape or not bool(torch.isfinite(z).all()) for z in (x,half,one,y)):
        raise ValueError('Finite, same-shaped nonempty native images are required')
    dhalf,done=half-x,one-x
    a=4*dhalf-done;b=2*done-4*dhalf;r=x-y
    return torch.stack((r.square().mean(),2*(r*a).mean(),a.square().mean()+2*(r*b).mean(),
                        2*(a*b).mean(),b.square().mean()))


def solve_density_quartic(coefficients):
    """Global minimum of one real quartic on [0,1], using cubic roots/ends.

    No density grid, line search, repeated refit, or optimizer iteration. Near
    multiple real roots can acquire small imaginary roundoff; their real parts
    are harmless additional feasible candidates, evaluated on the polynomial.
    """
    c=(coefficients.detach().cpu().double().numpy() if torch.is_tensor(coefficients)
       else np.asarray(coefficients,dtype=np.float64))
    if c.shape!=(5,) or not np.isfinite(c).all():raise ValueError('Five finite quartic coefficients required')
    derivative=np.asarray((4*c[4],3*c[3],2*c[2],c[1]))
    scale=max(float(np.abs(derivative).max()),np.finfo(np.float64).tiny)
    first=0
    while first<len(derivative) and abs(derivative[first])<=128*np.finfo(np.float64).eps*scale:first+=1
    roots=np.roots(derivative[first:]/scale) if len(derivative)-first>1 else np.asarray([])
    tolerance=64*np.cbrt(np.finfo(np.float64).eps)
    candidates=[0.,1.]
    for z in roots:
        if abs(z.imag)<=tolerance*max(1.,abs(z.real)) and -tolerance<=z.real<=1+tolerance:
            candidates.append(float(np.clip(z.real,0.,1.)))
    candidates=sorted(set(candidates));best=0.;change=0.
    variation=max(float(np.abs(c[1:]).max()),np.finfo(np.float64).tiny)
    values=[]
    for rho in candidates:
        # Do not subtract nearly equal full MSEs to assess tiny improvements.
        delta=rho*(c[1]+rho*(c[2]+rho*(c[3]+rho*c[4])))
        values.append(dict(rho=rho,surrogate_mse=float(c[0]+delta)))
        if delta<change-256*np.finfo(np.float64).eps*variation:best,change=rho,float(delta)
    return best,dict(coefficients=c.tolist(),stationary_roots=[[float(z.real),float(z.imag)] for z in roots],
        candidates=values,proposal_rho=best,proposal_surrogate_mse=float(c[0]+change),
        solve='one cubic-root solve plus interval endpoints, no optimizer/grid/backtracking')


@torch.no_grad()
def propose_shadow_family(module,views,root_index,render_density,probe_scene):
    """Fit/write just this family's DC and return one JSON density proposal.

    ``probe_scene() -> (raw, global_child_rows[2])`` supplies the actual-codec
    unit-density scene (alpha=.99 for this pair). ``render_density(view,rho)``
    returns raw native RGB, or {render: RGB}, using the current DC leaf version.
    Its rho=0 is the true accepted prefix, with the original old source still
    present. Other unaccepted families must be absent from ALL probes.

    This helper never accepts, updates a prefix, changes alpha/floors, or calls
    backward/optimizer.step. A must do the sole real-codec alltrain error/byte
    acceptance and restore this pair's DC on rejection. Exceptions restore DC.
    """
    names,views=_view_list(views);model=module.births;root=int(root_index)
    rows=model.reincarnation_groups[root]
    report=dict(root_index=root,rho=0.,views=names,view_count=len(views),SGD_steps=0,optimizer_steps=0,
        method='sequential_prefix_shadow_2x2_DC_then_quartic_density',density_alpha_unit=.99,
        scope='one model, all supplied standard training views, accepted prefix plus this family only',
        proposal_only=True,actual_codec_acceptance_required=True,
        polynomial_boundary='fixed-order/support raw RGB idealization; native skip, termination and output clamp require the sole actual-codec check')
    if len(rows)!=3 or not views:return dict(report,reason='no_two_child_family_or_views')
    children=rows[1:];leaf=model._features_dc;initial=leaf[children].detach().clone()
    try:
        raw,global_rows=probe_scene();global_rows=torch.as_tensor(global_rows,device=raw['xyz'].device,dtype=torch.long)
        if global_rows.shape!=(2,) or len(global_rows.unique())!=2:
            raise ValueError('Shadow native probe must identify exactly two distinct child rows')
        report.update(model_child_rows=children.tolist(),packed_scene_child_rows=global_rows.tolist())
        scene=RawView(raw,module.config.sh_degree,module.gaussians)
        h=torch.zeros(1,3,2,2,dtype=torch.float64,device=raw['xyz'].device);g=torch.zeros(1,3,2,dtype=torch.float64,device=h.device)
        colours=torch.zeros_like(raw['xyz']);colours[global_rows,torch.arange(2,device=h.device)]=1
        bg=torch.zeros_like(module.background)
        with _isolated_renderer(module.renderer):
            module.renderer.gaussians=scene
            for view in views:
                prediction=module.renderer(view,module.background)['render']
                weights=module.renderer(view,bg,override_color=colours)['render'][:2]
                shader=_preclamp_rgb(scene.get_features[global_rows],scene.get_xyz[global_rows],scene.active_sh_degree,view)>0
                active=(prediction>=0)&(prediction<=1)
                residual=view.image.to(prediction)-prediction.clamp(0,1)
                for channel in range(3):
                    j=Y00*weights*shader[:,channel,None,None]*active[channel][None]
                    hh,gg=_pair_image_moments(j.flatten(1).double()[None],residual[channel].flatten().double())
                    h[:,channel]+=hh/prediction.numel();g[:,channel]+=gg/prediction.numel()
        lower,upper=model.intrinsic_dc_bounds()
        current=raw['sh_dc'][global_rows,0]
        lo,hi=lower[children,0],upper[children,0]
        delta,colour_report=solve_box2(h/len(views),g/len(views),
            (lo-current).T[None].clamp_max(0),(hi-current).T[None].clamp_min(0))
        delta=delta[0].T.to(current)
        target=(current+delta).maximum(lo).minimum(hi)
        target=torch.where(delta==0,initial[:,0],target)
        leaf[children,0]=target;model._clear_reincarnation_cache()
        report.update(colour_solve=colour_report,changed_child_DC_components=int((target!=initial[:,0]).sum()),
            max_abs_DC_change=float((target-initial[:,0]).abs().max()),native_colour_response_renders=2*len(views))
        coeff=torch.zeros(5,dtype=torch.float64);raw_errors=[];clamped_errors=[]
        with _isolated_renderer(module.renderer):
            for view in views:
                images=[]
                for rho in (0.,.5,1.):
                    value=render_density(view,rho);image=value['render'] if isinstance(value,dict) else value
                    images.append(image.detach())
                truth=view.image.to(images[0])
                coeff+=density_coefficients(*images,truth).cpu()/len(views)
                raw_errors.append([float((x-truth).square().mean()) for x in images])
                clamped_errors.append([float((x.clamp(0,1)-truth).square().mean()) for x in images])
        rho,fit=solve_density_quartic(coeff)
        report.update(rho=rho,physical_child_alpha=.99*rho,density_fit=fit,
            probe_densities=[0.,.5,1.],probe_raw_mean_mse=np.mean(raw_errors,axis=0).tolist(),
            probe_clamped_mean_mse=np.mean(clamped_errors,axis=0).tolist(),native_density_probe_renders=3*len(views),
            reason='unique_density_proposal_ready_for_actual_codec_check' if rho>0 else 'prior_is_surrogate_minimum')
        return report
    except BaseException:
        leaf[children]=initial;model._clear_reincarnation_cache()
        raise
