"""Joint native-compositing DC initialization for independent newborns.

Geometry, alpha and directional SH stay fixed. Only the original two training
proposal cameras are used. This is extra colour fitting at birth, not candidate
screening; it has no decoder state and changes no quantization precision.
"""
import math

import torch

from .birth_quantization import RAW_FIELDS
from .birth_radiance import _isolated_renderer, _preclamp_rgb
from .reincarnation_model import C0, sh1_sphere_range
from .residual_birth import RawView
from .support_reincarnation_proposal import _ordered_views


@torch.no_grad()
def projected_colour_fit(rhs, operator, lower, upper, active, iterations=16):
    """Box-constrained least squares, diagonally majorized FISTA.

    For nonnegative alpha-compositing J, H=J.T@J has nonnegative entries.
    diag(H@1)-H is positive semidefinite, so row sums supply a valid diagonal
    majorizer without choosing an image-dependent learning rate. Inactive
    channels are held at zero in both the operator and its row sums.
    """
    if rhs.shape!=lower.shape or rhs.shape!=upper.shape or rhs.shape!=active.shape:
        raise ValueError('Colour solve arrays must have the same shape')
    if not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1:
        raise ValueError('Colour solve iterations must be positive')
    if not bool(torch.isfinite(rhs).all()&torch.isfinite(lower).all()&torch.isfinite(upper).all()):
        raise ValueError('Colour solve inputs must be finite')
    if bool((lower>0).any()|(upper<0).any()|(lower>upper).any()):
        raise ValueError('Colour box must contain the unchanged zero offset')
    if not bool(active.any()):
        return torch.zeros_like(rhs),dict(iterations=0,normal_operator_calls=0,active_channels=0)
    mask=active.to(dtype=rhs.dtype)
    calls=0
    def apply(value):
        nonlocal calls
        calls+=1
        result=operator((value*mask).to(rhs.dtype)).double()*mask
        if result.shape!=rhs.shape or not bool(torch.isfinite(result).all()):
            raise ValueError('Colour normal operator returned invalid values')
        return result
    majorizer=apply(mask).clamp_min(1e-20)
    x=torch.zeros_like(rhs,dtype=torch.float64);y=x.clone();momentum=1.
    b=rhs.double()*mask
    for index in range(iterations):
        hy=apply(y) if index else torch.zeros_like(y)
        new=((y+(b-hy)/majorizer).clamp(lower.double(),upper.double()))*mask
        next_momentum=(1+math.sqrt(1+4*momentum*momentum))/2
        y=new+(momentum-1)/next_momentum*(new-x)
        x=new;momentum=next_momentum
    return x.to(rhs.dtype),dict(iterations=iterations,normal_operator_calls=calls,
                               active_channels=int(active.sum()))


@torch.no_grad()
def native_colour_seed(renderer, template, old, newborn, views, background,
                       *,dc_step,predictor_dc,iterations=16):
    """Return quantized SH DCs with coherent two-view evidence.

    The solve includes overlap and occlusion of the complete old+new scene.
    It is linear in a common RGB/DC shift while SH1 is nonnegative over the
    entire sphere. Channels outside that condition, without two actual
    responses, or with opposing per-view gradients retain their original DC.
    The box prevents creating new HDR range beyond max(1,inherited maximum).
    Later ordinary all-attribute optimization remains unconstrained.
    """
    ordered=_ordered_views(views)
    if len(ordered)!=2:
        raise ValueError('Native colour birth initialization needs exactly two training views')
    if not math.isfinite(dc_step) or dc_step<=0:
        raise ValueError('DC quantization step must be finite and positive')
    count=len(newborn['xyz'])
    report=dict(enabled=True,mode='native_joint_dc',candidates=count,
        proposal_views=[name for name,_ in ordered],test_used=False,
        colour_units='common RGB offset; stored as parent-predicted quantized SH DC',
        objective='fixed-geometry un-clamped native RGB squared error',
        directional_SH_unchanged=True,geometry_and_alpha_unchanged=True,
        screening=False)
    if not count:
        return newborn['sh_dc'].clone(),{**report,'accepted':0,'additional_render_calls':0,'additional_backward_calls':0}
    if predictor_dc.shape!=newborn['sh_dc'].shape:
        raise ValueError('DC predictor must align with newborns')
    raw={name:torch.cat((old[name].detach(),newborn[name].detach())) for name in RAW_FIELDS}
    scene=RawView(raw,template.max_sh_degree,template)
    scene.active_sh_degree=template.active_sh_degree
    directional=newborn['sh_rest'] if scene.active_sh_degree else newborn['sh_rest'][:,:0]
    low,high=sh1_sphere_range(newborn['sh_dc'],directional)
    lower=-low.clamp_min(0)
    upper=(high.clamp_min(1)-high).clamp_min(0)
    sphere_valid=low.ge(0)&torch.isfinite(high)
    old_zeros=torch.zeros_like(old['xyz'])
    baseline=[];projections=[];masses=[]
    calls=0;backwards=0
    with _isolated_renderer(renderer):
        renderer.gaussians=scene
        for name,view in ordered:
            colors=_preclamp_rgb(scene.get_features,scene.get_xyz,scene.active_sh_degree,view).clamp_min(0)
            image=renderer(view,background,override_color=colors)['render'].detach();calls+=1
            target=view.image.to(image)-image
            baseline.append(dict(camera=name,mse=float(target.square().mean()),
                clamped_mse=float((image.clamp(0,1)-view.image.to(image)).square().mean())))
            with torch.enable_grad():
                variable=torch.zeros_like(newborn['xyz'],requires_grad=True)
                response=renderer(view,background*0,override_color=torch.cat((old_zeros,variable)))['render'];calls+=1
                projection=torch.autograd.grad((response*target).sum(),variable,retain_graph=True)[0]
                mass=torch.autograd.grad(response.sum(),variable)[0];backwards+=2
            projections.append(projection.detach());masses.append(mass.detach())
        a,b=projections
        observed=torch.stack(masses).gt(1e-12).all(0)
        coherent=(a.double()*b.double()).gt(0)
        active=observed&coherent&sphere_valid
        rhs=a+b

        def operator(value):
            nonlocal calls,backwards
            total=torch.zeros_like(value)
            for name,view in ordered:
                with torch.enable_grad():
                    variable=value.detach().clone().requires_grad_(True)
                    response=renderer(view,background*0,override_color=torch.cat((old_zeros,variable)))['render'];calls+=1
                    gradient=torch.autograd.grad(.5*response.square().sum(),variable)[0];backwards+=1
                total+=gradient.detach()
            return total

        delta,fit=projected_colour_fit(rhs,operator,lower,upper,active,iterations)
    # Snap to the existing predictor lattice while keeping the global SH1
    # colour box. This is the same residual step, not a new precision choice.
    rgb_step=C0*dc_step
    minimum=torch.ceil(lower.double()/rgb_step)
    maximum=torch.floor(upper.double()/rgb_step)
    correction=torch.round(delta.double()/rgb_step).clamp(minimum,maximum)*active
    original_symbols=((newborn['sh_dc']-predictor_dc)/dc_step).round()
    symbols=original_symbols+correction[:,None].to(original_symbols)
    dc=predictor_dc+symbols*dc_step
    report.update(fit,accepted=int(correction.ne(0).any(-1).sum()),
        sphere_invalid_channels=int((~sphere_valid).sum()),
        opposed_or_zero_projection_channels=int((~coherent).sum()),
        missing_two_view_response_channels=int((~observed).sum()),
        max_applied_rgb_delta=float((correction*rgb_step).abs().max()),
        additional_render_calls=calls,additional_backward_calls=backwards,
        baseline=baseline)
    return dc,report
