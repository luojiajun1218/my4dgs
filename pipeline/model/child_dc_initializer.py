"""One bounded two-child DC solve from native residuals; zero SGD steps.

Each RGB/family solves a 2x2 block with its true sibling overlap. Cross-family
blocks are omitted; one full-scene native/codec check accepts or rolls back the
whole proposal. There is no iterative fitting, damping search or Adam update.
"""
import torch

from .birth_radiance import (_isolated_renderer, _response_scene, _sh_active,
                            _view_list, _errors, Y00)

C1 = .4886025119029199


def _pair_image_moments(flat, residual):
    """Exact sibling Gram/RHS reductions without a huge-K skinny cuBLAS GEMM.

    Inputs keep their float64 multiplication and accumulation. For full-size
    images, explicit five dot reductions avoid the severely under-utilized
    batched (2 x pixels) @ (pixels x 2) kernel on consumer GPUs.
    """
    plus,minus=flat.unbind(1)
    h00=(plus*plus).sum(-1)
    h01=(plus*minus).sum(-1)
    h11=(minus*minus).sum(-1)
    h=torch.stack((h00,h01,h01,h11),-1).reshape(-1,2,2)
    b=torch.stack(((plus*residual).sum(-1),(minus*residual).sum(-1)),-1)
    return h,b


def solve_box2(gram, rhs, lower, upper):
    """Batched exact convex box QP by its interior, four edges and corners."""
    g, b, lo, hi = [torch.as_tensor(x).detach().to(device='cpu', dtype=torch.float64)
                    for x in (gram, rhs, lower, upper)]
    if (g.shape[-2:] != (2, 2) or b.shape != g.shape[:-1] or lo.shape != b.shape
            or hi.shape != b.shape or any(not bool(torch.isfinite(x).all()) for x in (g,b,lo,hi))
            or bool((lo > hi).any()) or bool((lo > 0).any()) or bool((hi < 0).any())):
        raise ValueError('Finite 2x2 Gram/box shapes must agree and zero must be feasible')
    shape=b.shape;g,b,lo,hi=g.reshape(-1,2,2),b.reshape(-1,2),lo.reshape(-1,2),hi.reshape(-1,2)
    scale=torch.maximum(g.abs().amax((1,2)),b.abs().amax(1)).clamp_min(torch.finfo(g.dtype).tiny)
    eps=256*torch.finfo(g.dtype).eps
    if bool(((g-g.transpose(1,2)).abs().amax((1,2))>eps*scale).any()):
        raise ValueError('Gram must be symmetric')
    g=(g+g.transpose(1,2))*.5
    if bool((torch.linalg.eigvalsh(g)[:,0]<-eps*scale).any()):
        raise ValueError('Gram must be positive semidefinite')
    gn,bn=g/scale[:,None,None],b/scale[:,None]
    candidates=[torch.zeros_like(b)]
    interior=(torch.linalg.pinv(gn,hermitian=True)@bn[...,None])[...,0]
    valid=((interior>=lo)&(interior<=hi)).all(1)
    candidates.append(torch.where(valid[:,None],interior,torch.zeros_like(interior)))
    for fixed in (0,1):
        free=1-fixed
        for bound in (lo,hi):
            x=torch.zeros_like(b);x[:,fixed]=bound[:,fixed]
            numerator=bn[:,free]-gn[:,free,fixed]*x[:,fixed]
            denominator=gn[:,free,free]
            optimum=numerator/denominator.clamp_min(torch.finfo(g.dtype).tiny)
            # Degenerate affine edge: its endpoint is the exact minimizer.
            affine=torch.where(numerator>0,hi[:,free],torch.where(numerator<0,lo[:,free],torch.zeros_like(numerator)))
            x[:,free]=torch.where(denominator>0,optimum,affine).maximum(lo[:,free]).minimum(hi[:,free])
            candidates.append(x)
    for x0 in (lo[:,0],hi[:,0]):
        for x1 in (lo[:,1],hi[:,1]):candidates.append(torch.stack((x0,x1),1))
    best=torch.zeros_like(b);value=torch.zeros(len(b),dtype=g.dtype)
    for x in candidates[1:]:
        objective=torch.einsum('bi,bij,bj->b',x,g,x)-2*(x*b).sum(1)
        better=objective<value-eps*scale
        best=torch.where(better[:,None],x,best);value=torch.where(better,objective,value)
    return best.reshape(shape),dict(objective_change=value.reshape(shape[:-1]).tolist(),
        zero_response_blocks=int((g.abs().sum((1,2))==0).sum()),solve='interior/edge/corner closed forms; no optimizer steps')


@torch.no_grad()
def _native_moments(module,views,scene,old_count,children):
    f=len(children);h=torch.zeros(f,3,2,2,dtype=torch.float64,device=children.device)
    b=torch.zeros(f,3,2,dtype=torch.float64,device=children.device)
    colours=scene.get_xyz.new_zeros(len(scene.get_xyz),3);bg=torch.zeros_like(module.background)
    before=[];maxdiff=0.
    for view in views:
        prediction=module.forward(view)['render'].detach()
        module.renderer.gaussians=scene
        static=module.renderer(view,module.background)['render']
        maxdiff=max(maxdiff,float((static-prediction).abs().max()))
        if not torch.allclose(static,prediction,atol=2e-6,rtol=2e-5):
            raise ValueError('Initializer response scene differs from actual training scene')
        image=prediction.clamp(0,1);residual=view.image.to(image)-image
        active=(prediction>=0)&(prediction<=1);shader=_sh_active(scene,view,old_count)
        before.append((image-view.image.to(image)).square().mean().double())
        for start in range(0,f,3):
            pair=children[start:start+3];ids=pair.flatten();maps=[]
            for offset in range(0,len(ids),3):
                batch=ids[offset:offset+3];channels=torch.arange(len(batch),device=ids.device)
                colours.zero_();colours[old_count+batch,channels]=1
                maps.append(module.renderer(view,bg,override_color=colours)['render'][:len(batch)])
            weights=torch.cat(maps).reshape(len(pair),2,*image.shape[-2:])
            for channel in range(3):
                j=Y00*weights*shader[pair,channel][...,None,None]*active[channel][None,None]
                flat=j.flatten(2).double();r=residual[channel].flatten().double()
                pair_h,pair_b=_pair_image_moments(flat,r)
                h[start:start+len(pair),channel]+=pair_h/image.numel()
                b[start:start+len(pair),channel]+=pair_b/image.numel()
    return h/len(views),b/len(views),torch.stack(before),maxdiff


@torch.no_grad()
def initialize_child_dc_once(module, views, validate_scene=None):
    """Write one quantized, source-bounded child-DC proposal, or exact rollback.

    Caller supplies all standard training cameras and calls once at birth in
    one model. ``validate_scene()`` may return same-population actual-codec per-view
    MSEs or ``{mse, per_view_mse, codec_bytes}`` (preferred); otherwise the
    ordinary quantized native forward is used.
    No Parameter/grad/Adam state is replaced or updated. Mother, geometry,
    opacity and SH-rest remain unchanged. Counterfactual residual renders are
    analysis/closed-form initialization, not hidden SGD steps.
    """
    names,views=_view_list(views);model=module.births
    children=[rows[1:] for rows in model.reincarnation_groups if len(rows)==3]
    report=dict(method='one_shot_bounded_two_child_DC',SGD_steps=0,optimizer_steps=0,
                views=names,endpoint='source theta0 intact mother plus ordinary positive-alpha details',
                accepted=False,pair_count=len(children),cross_family_H='omitted; one full-scene check',
                view_scope='all supplied standard training cameras, one model',
                acceptance='one proposal only, strict all-training mean-MSE decrease or exact rollback')
    if not children or not views:return dict(report,reason='no_child_pairs_or_views')
    children=torch.stack(children).to(model._xyz.device);leaf=model._features_dc
    initial=leaf.detach().clone();committed=False
    def validate():
        value=validate_scene() if validate_scene else _errors(module,views)
        metadata=dict(value) if isinstance(value,dict) else {}
        if isinstance(value,dict):value=value.get('per_view_mse',value.get('mse'))
        if value is None:raise ValueError('Validation must return MSE or a dict containing mse/per_view_mse')
        return torch.as_tensor(value,dtype=torch.float64).detach(),metadata
    before,before_info=validate();report.update(before_mean_mse=float(before.mean()),before_validation=before_info,
        fit_scene='current full quantized native training scene',
        validation_scene='caller actual codec scene' if validate_scene else 'quantized native training scene')
    try:
        with _isolated_renderer(module.renderer):
            scene,old_count=_response_scene(module)
            h,b,native_before,maxdiff=_native_moments(module,views,scene,old_count,children)
            report['response_scene_max_abs_difference']=maxdiff
            qview=module.make_birth_view();current=qview.raw_value('sh_dc').detach()[:,0]
            rest=qview.raw_value('sh_rest').detach()
            if hasattr(model,'intrinsic_dc_bounds'):
                lo,hi=model.intrinsic_dc_bounds();lo,hi=lo[:,0],hi[:,0]
            else:
                low,high=model._radiance_bounds();radius=C1*rest.square().sum(1).sqrt()
                lo=(low+radius-.5)/Y00;hi=(high-radius-.5)/Y00
            lower=(lo-current)[children].permute(0,2,1);upper=(hi-current)[children].permute(0,2,1)
            # Numerical source-box endpoints can differ from current decoded
            # pristine values by an ulp. Preserve zero without widening farther.
            lower=lower.clamp_max(0);upper=upper.clamp_min(0)
            delta,math=solve_box2(h,b,lower,upper);report['math']=math
            delta=delta.to(current).permute(0,2,1)
            ids=children.flatten();step=current.new_tensor(qview.steps['sh_dc'])
            predictor=qview.predictor['sh_dc'][:,0].detach()
            desired=current[ids]+delta.flatten(0,1)
            k=((desired-predictor[ids])/step).round()
            kl=torch.ceil((lo[ids]-predictor[ids])/step);kh=torch.floor((hi[ids]-predictor[ids])/step)
            k=k.maximum(kl).minimum(kh);decoded=predictor[ids]+k*step
            valid=(kl<=kh)&(decoded>=lo[ids])&(decoded<=hi[ids])
            # An analytically zero component must retain its exact pristine
            # preimage, rather than acquiring a predictor-grid rounding edit.
            target=torch.where(valid&(delta.flatten(0,1)!=0),decoded,initial[ids,0])
            leaf[ids,0]=target
            model._clear_reincarnation_cache()
            after_view=module.make_birth_view()
            unchanged_rest=torch.equal(after_view.raw_value('sh_rest').detach().contiguous().view(torch.uint8),rest.contiguous().view(torch.uint8))
            report['SHrest_bitwise_unchanged']=unchanged_rest
            if not unchanged_rest:
                report['reason']='DC_proposal_changed_fixed_SHrest';return report
            after,after_info=validate();finite=bool(torch.isfinite(before).all()&torch.isfinite(after).all())
            committed=finite and float(after.mean())<float(before.mean())
            report.update(accepted=committed,after_mean_mse=float(after.mean()) if finite else None,
                after_validation=after_info,native_before_mean_mse=float(native_before.mean()),changed_child_DC_components=int((target!=initial[ids,0]).sum()),
                max_abs_DC_update=float((target-initial[ids,0]).abs().max()),
                reason='alltrain_mean_mse_improved' if committed else 'single_proposal_not_better')
            return report
    finally:
        if not committed:leaf.copy_(initial);model._clear_reincarnation_cache()

