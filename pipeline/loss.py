"""动态帧训练损失的统一入口。

负责图像重建损失及其 SSIM 缓存、RMSE 和时间权重，并按当前训练阶段组合
运动正则、逐点及区域属性的存储损失；可按需返回加权后的损失分项供检查。
训练步骤和参数更新由 per_frame_training.py 与 module.py 调度，存储开销的
估算由 model/storage_rate.py 负责。本文件不管理首帧静态训练或码流编解码。
"""
from collections import OrderedDict
from pathlib import Path
import weakref

import torch
from torch.nn.functional import conv2d, interpolate, l1_loss
from libgs.metric import psnr, ssim
from libgs.metric.ssim import create_window

from .per_frame_training import fitting_old_motion, regional_attribute_repair_active


def motion_image_factor(step, motion_steps, enabled=False, fitting_motion=False,
                        bootstrap_steps=None):
    """Track broad image structure first, then fit full-resolution details."""
    if not enabled or not fitting_motion:
        return 1
    bootstrap_steps = (2 * motion_steps // 5 if bootstrap_steps is None else bootstrap_steps)
    if step <= bootstrap_steps // 2:
        return 4
    if step <= bootstrap_steps:
        return 2
    return 1

class _TrainingSSIMCache:
    """Cache fixed target-image SSIM terms for the current frame."""

    def __init__(self, max_bytes=768 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._windows = {}
        self._targets = OrderedDict()
        self._target_bytes = 0
        self._budgets = {}

    def clear(self):
        self._targets.clear()
        self._target_bytes = 0
        self._budgets.clear()

    def _window(self, image):
        key = (image.shape[-3], image.device, image.dtype)
        if key not in self._windows:
            self._windows[key] = create_window(11, image.shape[-3]).to(
                device=image.device, dtype=image.dtype)
        return self._windows[key]

    def prepare(self, image, target, size):
        if (target.requires_grad or image.device != target.device or image.dtype != target.dtype or
                torch.is_autocast_enabled(image.device.type)):
            return None
        if image.device not in self._budgets:
            # Leave room for the renderer and saved backward buffers.
            free = torch.cuda.mem_get_info(image.device)[0] if image.is_cuda else 4 * self.max_bytes
            self._budgets[image.device] = min(self.max_bytes, free // 4)
        budget = self._budgets[image.device]
        pixels = image.numel() // (image.shape[-2] * image.shape[-1]) * size[0] * size[1]
        resized_bytes = pixels * image.element_size()
        needed = resized_bytes * (2 + (tuple(target.shape[-2:]) != tuple(size)))
        if needed > budget:
            return None
        try:
            version = target._version
        except RuntimeError:  # Inference tensors do not expose a version counter.
            return None
        key = (id(target), version, tuple(size))
        entry = self._targets.get(key)
        if entry is not None and entry[0]() is target:
            self._targets.move_to_end(key)
            return (target if entry[1] is None else entry[1], *entry[2:5])
        if entry is not None:
            self._target_bytes -= entry[5]
            del self._targets[key]
        # Device-cached views retain their target tensors across steps. With
        # CPU-only view caching, each device copy can die after one step; its
        # cached SSIM moments cannot be reused and should not occupy GPU RAM.
        for stale_key, stale in list(self._targets.items()):
            if stale[0]() is None:
                self._target_bytes -= stale[5]
                del self._targets[stale_key]
        resized = (interpolate(target[None], size=size, mode='area')[0]
                   if tuple(target.shape[-2:]) != tuple(size) else target)
        window = self._window(image)
        with torch.no_grad():
            mean = conv2d(resized, window, padding=5, groups=image.shape[-3])
            variance = conv2d(resized * resized, window, padding=5,
                              groups=image.shape[-3]) - mean.pow(2)
        stored_resized = None if resized is target else resized
        nbytes = sum(t.numel() * t.element_size() for t in
                     (mean, variance) + (() if stored_resized is None else (stored_resized,)))
        if nbytes <= budget:
            while self._targets and self._target_bytes + nbytes > budget:
                _, old = self._targets.popitem(last=False)
                self._target_bytes -= old[5]
            self._targets[key] = (weakref.ref(target), stored_resized, mean, variance, window, nbytes)
            self._target_bytes += nbytes
        return resized, mean, variance, window

    @staticmethod
    def ssim(image, target, mean_target, variance_target, window):
        channel = image.shape[-3]
        mean_image = conv2d(image, window, padding=5, groups=channel)
        mean_image_sq = mean_image.pow(2)
        mean_target_sq = mean_target.pow(2)
        mean_product = mean_image * mean_target
        variance_image = conv2d(image * image, window, padding=5, groups=channel) - mean_image_sq
        covariance = conv2d(image * target, window, padding=5, groups=channel) - mean_product
        return (((2 * mean_product + 0.01**2) * (2 * covariance + 0.03**2)) /
                ((mean_image_sq + mean_target_sq + 0.01**2) *
                 (variance_image + variance_target + 0.03**2))).mean()

def stabilized_rmse(image, target):
    """Pixel-scale RMSE with finite gradient and an exact zero at a match."""
    mse = (image - target).square().mean()
    return _stabilized_rmse_from_mse(mse)

def _stabilized_rmse_from_mse(mse):
    return mse / ((mse + 1e-8).sqrt() + 1e-4)

@torch.no_grad()
def temporal_repair_weights(target, previous, strength, scale):
    """Soft temporal evidence in a fixed training camera; never a hard mask."""
    if target.shape != previous.shape or target.ndim != 3:
        raise ValueError('Temporal repair images must have matching CHW shapes')
    change = (target.detach() - previous.detach()).abs().mean(dim=0, keepdim=True)
    change = torch.nn.functional.avg_pool2d(change.unsqueeze(0), 5, stride=1, padding=2)[0]
    return 1 + strength * (change / scale).clamp(0, 1)

def weighted_repair_rmse(image, target, weights):
    if image.shape != target.shape or weights.shape != (1, *image.shape[-2:]):
        raise ValueError('Repair weights must match the image')
    mse = ((image - target).square() * weights).sum() / (weights.sum() * image.shape[0])
    return mse / ((mse + 1e-8).sqrt() + 1e-4)


def mixed_image_loss(image, target, photometric_loss, mix, weights=None):
    """Combine the photometric term with optional temporally weighted RMSE."""
    squared_error = None
    if weights is None:
        squared_error = (image - target).square()
        rmse = _stabilized_rmse_from_mse(squared_error.mean())
    else:
        rmse = weighted_repair_rmse(image, target, weights)
    return (1 - mix) * photometric_loss + mix * rmse, rmse, squared_error


def _add_component(total, value, name, components):
    if components is not None:
        components[name] = value.detach()
    return total + value

def motion_photometric_loss(image, target, lambda_dssim, factor=1, *, metric='l1', cache=None):
    if metric not in ('l1', 'rmse'):
        raise ValueError('Photometric pixel loss must be l1 or rmse')
    size = (tuple(max(1, value // factor) for value in image.shape[-2:])
            if factor > 1 else tuple(image.shape[-2:]))
    cached = cache.prepare(image, target, size) if cache is not None else None
    if factor > 1:
        image = interpolate(image[None], size=size, mode='area')[0]
        if cached is None:
            target = interpolate(target[None], size=size, mode='area')[0]
    if cached is not None:
        target, mean_target, variance_target, window = cached
    # Global RMS keeps pixel-scale loss units while giving small errors small
    # gradients. The offset makes an exact match have zero finite loss/gradient.
    if metric == 'l1':
        pixel = l1_loss(image, target)
    else:
        pixel = stabilized_rmse(image, target)
    structural = (ssim(image, target) if cached is None else
                  cache.ssim(image, target, mean_target, variance_target, window))
    return (1 - lambda_dssim) * pixel + lambda_dssim * (1 - structural)


def compute_training_loss(self, viewpoint, step, regularization_weight=1., *,
                          early_replacement=False, components=None):
    """Return the real training objective, metrics and rendered package.

    Optional ``components`` receives detached, already-weighted terms from this
    same forward pass; no diagnostic work or extra render is performed by default.
    """
    if components is not None:
        components.clear()
    package = self.forward(viewpoint, training=True)
    image = package['render']
    factor = motion_image_factor(step, self.motion_steps,
        enabled=self.config.motion_coarse_to_fine,
        fitting_motion=fitting_old_motion(self),
        bootstrap_steps=self.config.rigid_tree.spatial_gate.start_step)
    metric = getattr(self.config, 'photometric_loss', 'l1')
    if (self.motion_committed or (early_replacement and self.early_initialized)) and self.config.birth.photometric_loss is not None:
        metric = self.config.birth.photometric_loss
    cache = getattr(self, '_training_ssim_cache', None)
    if cache is None:
        cache = self._training_ssim_cache = _TrainingSSIMCache()
    image_loss = motion_photometric_loss(image, viewpoint.image,
                                         self.config.lambda_dssim, factor, metric=metric, cache=cache)
    loss = image_loss
    full_cfg = getattr(self.config, 'full_update', None)
    mix = ((full_cfg.repair_rmse_mix if step > self.motion_steps else full_cfg.motion_rmse_mix)
           if full_cfg is not None else 0.)
    temporal_objective = bool((regional_attribute_repair_active(self) or
                               getattr(self.config, 'ablation_motion_only', False)) and mix > 0)
    squared_error = None
    if temporal_objective:
        weights = self._temporal_repair_weights(viewpoint) if full_cfg.repair_temporal_weight else None
        loss, repair_rmse, squared_error = mixed_image_loss(
            image, viewpoint.image, image_loss, mix, weights)
    if components is not None:
        components['image'] = loss.detach()
    # Leave scalar diagnostics on device until all views have finished.
    # Converting each one to float would synchronize CUDA before backward.
    early = early_replacement and self.early_initialized
    if early:
        squared_error = ((image.detach() - viewpoint.image.detach()).square()
                         if squared_error is None else squared_error.detach())
        channel_mse = squared_error.view(image.shape[0], -1).mean(1, keepdim=True)
        image_psnr = (20 * torch.log10(1. / torch.sqrt(channel_mse))).mean()
    elif squared_error is not None:
        channel_mse = squared_error.detach().view(image.shape[0], -1).mean(1, keepdim=True)
        image_psnr = (20 * torch.log10(1. / torch.sqrt(channel_mse))).mean()
    else:
        image_psnr = psnr(image.detach(), viewpoint.image).detach()
    metrics = dict(image_loss=image_loss.detach(), motion_image_factor=factor,
                   psnr=image_psnr,
                   gs=len(self.gaussians._xyz))
    if temporal_objective:
        metrics['repair_rmse'] = repair_rmse.detach()
    if early:
        error=squared_error.mean(0)
        weight=self.early_focus_weights.get(Path(viewpoint.path).name)
        metrics['early_batch_mse']=error.mean().detach()
        if weight is not None:
            metrics['early_focus_mse']=((error*weight).sum()/weight.sum().clamp_min(1e-12)).detach()
    proposal_weight = (getattr(self.motion, 'proposal_image_weight', 0.)
                       if not self.motion_committed else 0.)
    if proposal_weight:
        proposal = self.forward(viewpoint, training=True, proposal_only=True)['render']
        proposal_loss = motion_photometric_loss(proposal, viewpoint.image,
                                                self.config.lambda_dssim, factor, metric=metric, cache=cache)
        loss = _add_component(loss, proposal_weight * proposal_loss, 'proposal_image', components)
        metrics['proposal_image_loss'] = proposal_loss.detach()
    predictor_weight = (getattr(self.motion, 'predictor_image_weight', 0.)
                        if not self.motion_committed else 0.)
    if predictor_weight:
        prediction = self.forward(viewpoint, training=True, predictor_only=True)['render']
        prediction_loss = motion_photometric_loss(prediction, viewpoint.image,
            self.config.lambda_dssim, factor, metric=metric, cache=cache)
        loss = _add_component(loss, predictor_weight * prediction_loss, 'predictor_image', components)
        metrics['predictor_image_loss'] = prediction_loss.detach()
    if fitting_old_motion(self) and regularization_weight:
        loss = _add_component(loss, regularization_weight * self.motion.regularization_loss(step),
                              'motion_regularization', components)
    if (regularization_weight and getattr(self, 'opacity_update', None) is not None and
            (not getattr(self.config.birth,'frame_training_plan',False) or step >= 101 or
             (getattr(self, 'full_update_active', False) and
              not self.config.full_update.regional_repair)) and
            (not self.motion_committed or
             (not self.frame_finalized and (self.config.opacity_update.joint_with_birth or
                                           self.config.birth.joint_old_opacity)))):
        terms = self.opacity_update.rate_terms(self._opacity_rate_price(step),
            width=self.config.rigid_tree.sparse_rate_width_bins)
        loss = _add_component(loss, regularization_weight * terms['loss'], 'opacity_storage', components)
    if (regularization_weight and getattr(self, 'feature_dc_update', None) is not None and
            (not self.motion_committed or
             (not self.frame_finalized and self.config.feature_dc_update.joint_with_birth))):
        terms = self.feature_dc_update.rate_terms(self._rate_byte_price(step) * self.config.full_update.shared_dc_rate_scale,
            width=self.config.rigid_tree.sparse_rate_width_bins)
        loss = _add_component(loss, regularization_weight * terms['loss'], 'feature_storage', components)
    logscale_update = getattr(self, 'logscale_update', None)
    if (regularization_weight and logscale_update is not None and
            logscale_update.ready and not self.frame_finalized):
        terms = logscale_update.rate_terms(self._rate_byte_price(step))
        loss = _add_component(loss, regularization_weight * terms['loss'], 'logscale_storage', components)
    full = getattr(self, 'full_attribute_update', None)
    regional_full = getattr(self, 'regional_full_attributes', None)
    reuse_point_symbols = bool(regional_full is not None and step >= 101 and
        self.config.full_update.attribute_aware_gate and not self.motion._hard_gate_frozen and
        self.motion.gate_logits.requires_grad)
    point_symbols = None
    if (regularization_weight and getattr(self, 'full_update_active', False) and full is not None and
            full.config.storage_loss_enabled and not self.frame_finalized and
            not (step <= self.motion_steps and full.initial_zero_frozen())):
        terms = full.rate_terms(self._rate_byte_price(step) * self.config.full_update.point_dc_rate_scale,
                                include_scalar_counts=False, include_symbols=reuse_point_symbols)
        loss = _add_component(loss, regularization_weight * terms['loss'], 'point_storage', components)
        point_symbols = terms.get('symbols')
    if regularization_weight and regional_full is not None and step >= 101 and not self.frame_finalized:
        price = self._rate_byte_price(step)
        regional_terms = regional_full.rate_terms(price)
        loss = _add_component(loss, regularization_weight * regional_terms['loss'], 'regional_storage', components)
        metrics['regional_storage_loss'] = regional_terms['loss'].detach()
        metrics['regional_storage_raw_bytes'] = regional_terms['raw_bytes_surrogate'].detach()
        if self.config.full_update.attribute_aware_gate:
            gate_terms = regional_full.gate_rate_terms(full, self.opacity_update, price,
                                                       point_symbols=point_symbols)
            loss = _add_component(loss, regularization_weight * gate_terms['proxy'], 'attribute_gate', components)
            metrics['attribute_gate_hard_weighted_bytes'] = gate_terms['hard_weighted_bytes'].detach()
            metrics['attribute_gate_expected_weighted_bytes'] = gate_terms['expected_weighted_bytes'].detach()
            if step in (self.config.rigid_tree.spatial_gate.joint_gate_start_step, 151, 165, 180) and gate_terms['proxy'].requires_grad:
                gate_gradient = torch.autograd.grad(gate_terms['proxy'], self.motion.gate_logits,
                    retain_graph=True, allow_unused=True)[0]
                reachable = regional_full.reachable_gate_mask()
                logits = self.motion.gate_logits.detach()
                self.birth_control.setdefault('attribute_gate_rate_checks', []).append(dict(
                    step=step,
                    gradient_norm=0. if gate_gradient is None else float(gate_gradient.detach().norm()),
                    nonzero_gates=0 if gate_gradient is None else int(gate_gradient.detach().ne(0).sum()),
                    reachable_gates=int(reachable.sum()),
                    reachable_nonzero=0 if gate_gradient is None else int((reachable & gate_gradient.detach().ne(0)).sum()),
                    reachable_towards_switch=0 if gate_gradient is None else int((reachable & (gate_gradient.detach()*logits).gt(0)).sum()),
                    reachable_within_one=int((reachable & logits.abs().lt(1.)).sum()),
                    hard_weighted_bytes=float(gate_terms['hard_weighted_bytes']),
                    expected_weighted_bytes=float(gate_terms['expected_weighted_bytes'])))
    if (regularization_weight and self.config.birth.population_loss_weight and
            ((getattr(self.config.birth,'frame_training_plan',False) and step >= 101) or self.early_initialized)):
        penalty,population_metrics=self.per_frame_training._adaptive_population_regularization(step)
        loss = _add_component(loss, regularization_weight * penalty, 'population', components)
        metrics.update({name:value*regularization_weight for name,value in population_metrics.items()})
    return loss, metrics, package
