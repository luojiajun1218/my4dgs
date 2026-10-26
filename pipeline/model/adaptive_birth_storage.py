"""Differentiable newborn storage pressure, deliberately not a byte predictor.

SPG5/6 uses a source uvarint and seven signed pose uvarints per point. RTB2
uses four whole int16/int32 arrays followed by zlib, not per-symbol varints.
The appearance term below is only a compressibility proxy on integer symbols.
"""
import torch
from torch.nn import functional as F
from .birth_quantization import quantized_symbols
from .early_replacement_model import _rotate, _multiply, _rotation_vector


def storage_ramp(step, warmup_steps=20, ramp_steps=70, birth_step=70):
    return min(max((step-birth_step-warmup_steps)/ramp_steps, 0.), 1.)


def newborn_storage_terms(view, threshold=.01):
    """Return byte-scaled surrogate and diagnostics, with STE symbols.

    Continuous probability alpha supplies support gradients even inside a
    quantization bin; the existing final prune uses the model alpha threshold.
    s=a/(a+t) does not saturate at the observable deletion threshold. Negative
    opacity offsets retain their forward price but do not pull alpha upward
    toward an opaque parent. Their support gradient still encourages removal.
    """
    alpha=view.model._opacity.clamp(0, 1).flatten()
    support=alpha/(alpha+threshold)
    sources=view.source
    # IDs are discrete lineage metadata, never optimized. This stays on device.
    source_bytes=torch.ones_like(alpha)
    for bits in (7,14,21,28):source_bytes=source_bytes+(sources>=(1<<bits)).to(alpha.dtype)
    count=len(alpha)
    if not count:
        zero=alpha.sum()
        return dict(surrogate_bytes=zero,support_points=zero,base_bytes=zero,
                    pose_extra_bytes=zero,appearance_entropy_bytes=zero)
    base=source_bytes+7
    pose_extra=alpha.new_zeros(count)
    appearance_extra=alpha.new_zeros(count)
    for name,symbol in view.symbols.items():
        if name=='rotation':
            # Pose codec normalizes the quantized quaternion once more before
            # forming its seven parent-offset symbols.
            symbol=quantized_symbols(view.get_rotation-view.predictor[name],view.steps[name])
        if name in ('xyz','rotation'):
            # The byte pressure belongs to the newborn residual. Preserve the
            # rendered symbols exactly, but prevent its gradient from charging
            # the shared parent's already separately regularized motion.
            model=view.model
            value=(_rotate(model._source_q.detach(),model.local_xyz)+model._source_xyz.detach()
                   if name=='xyz' else F.normalize(_multiply(model._source_q.detach(),
                       _rotation_vector(model.local_rotvec)),dim=-1))
            if name=='rotation':
                value=torch.where((value*view.predictor[name]).sum(-1,keepdim=True)<0,-value,value)
            carrier=(value-view.predictor[name])/view.steps[name]
            symbol=carrier+(symbol-carrier).detach()
        if name=='opacity':
            symbol=torch.where(symbol.detach()>=0, symbol, symbol.detach())
        if name in ('xyz','rotation'):
            # Continuous log-length relaxation of zigzag uvarints. Actual
            # varint lengths have hard boundaries and zlib acts on whole pools.
            pose_extra+=torch.log2(1+2*symbol.abs()).reshape(count,-1).sum(-1)/7
        else:
            dimensions=symbol[0].numel() if count else symbol.shape[1:].numel()
            base=base+2*dimensions  # RTB2 minimum int16 storage before zlib.
            appearance_extra+=torch.log2(1+symbol.abs()).reshape(count,-1).sum(-1)/8
    raw=(support*(base+pose_extra+appearance_extra)).sum()
    return dict(surrogate_bytes=raw, support_points=support.sum(),
                base_bytes=(support*base).sum(),
                pose_extra_bytes=(support*pose_extra).sum(),
                appearance_entropy_bytes=(support*appearance_extra).sum())
