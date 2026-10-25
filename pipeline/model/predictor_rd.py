"""Local Adam-metric cost after coding a proposal around a shared predictor.

This is a local optimization surrogate, not an image Hessian or a derivative
through the lossless compressor. Its targets and metric come from photo-only
Adam before either proximal penalty has removed the proposed corrections.
"""
import torch


def predictor_cost(target, predictor, metric, steps, l1, byte_price, row_bytes):
    """Return per-point envelope cost and the chosen quantized residual.

    Try the same T/R block support choices as GS proximal sparsification. Hold
    the chosen residual fixed in backward (the envelope derivative); only the
    predictor learns here. Gates receive these costs through path probabilities.
    """
    target, metric = target.detach(), metric.detach()
    with torch.no_grad():
        difference = target - predictor.detach()
        residual = difference.sign() * (difference.abs() - l1 / metric.clamp_min(1e-20)).clamp_min(0)
        residual = (residual / steps).round().clamp(-32767, 32767) * steps
        options = torch.stack((residual, torch.cat((residual[:, :3], residual[:, 3:] * 0), -1),
            torch.cat((residual[:, :3] * 0, residual[:, 3:]), -1), residual * 0))
        active_t = options[:, :, :3].ne(0).any(-1)
        active_r = options[:, :, 3:].ne(0).any(-1)
        byte_cost = ((active_t | active_r) * row_bytes + 6 * (active_t.int() + active_r.int()))
        local = (.5 * metric * (predictor.detach()[None] + options - target[None]).square()).sum(-1)
        local += (l1 * options.abs()).sum(-1) + byte_price * byte_cost
        chosen = local.argmin(0)
        rows = torch.arange(len(target), device=target.device)
        selected = options[chosen, rows]
        selected_bytes = byte_cost[chosen, rows]
    distortion = (.5 * metric * (predictor + selected - target).square()).sum(-1)
    penalty = (l1 * selected.abs()).sum(-1)
    return distortion + penalty + byte_price * selected_bytes, selected, selected_bytes
