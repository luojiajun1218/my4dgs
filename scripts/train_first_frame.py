"""首帧静态 GS 训练入口及可选的增密、多视角和颜色微调策略。

负责命令行配置、首帧续训，以及可选的多视角、增密预算和颜色微调策略。
静态训练模型与调度核心位于 pipeline/first_frame；本文件不训练后续动态帧。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


STATIC_FIELDS = ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity')
COLOUR_FIELDS = ('_features_dc', '_features_rest')


def _top_candidates(mask, score, limit):
    """Return eligible indices in stable descending score order."""
    candidates = mask.nonzero(as_tuple=True)[0]
    count = min(limit, len(candidates))
    if not count:
        return candidates[:0]
    order = torch.argsort(score[candidates], descending=True, stable=True)
    return candidates[order[:count]]


@torch.no_grad()
def average_optimizer_gradients(optimizer, count):
    """Average accumulated model gradients without scaling view-space evidence."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Gradient averaging count must be a positive integer')
    if count == 1:
        return
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            if parameter.grad is not None:
                parameter.grad.div_(count)


@torch.no_grad()
def enforce_world_scale_cap(model, scene_extent, maximum_extent_fraction):
    """Project covariance axes onto the native large-splat safety bound.

    The original 3DGS pruning rule treats an axis above 0.1 scene extents as
    oversized, but only activates that rule together with its optional screen-
    size phase.  Some exact-15k configs deliberately place the opacity reset
    beyond the training horizon, so the safety rule never runs and scale Adam
    can create held-out-view occluders.  This projection makes the same world-
    space concept explicit and clears only the clipped Adam components.
    """
    if (isinstance(scene_extent, bool) or not isinstance(scene_extent, (int, float)) or
            not math.isfinite(float(scene_extent)) or float(scene_extent) <= 0 or
            isinstance(maximum_extent_fraction, bool) or
            not isinstance(maximum_extent_fraction, (int, float)) or
            not math.isfinite(float(maximum_extent_fraction)) or
            not 0 < float(maximum_extent_fraction) <= 1):
        raise ValueError('World-scale safety needs a positive finite extent and fraction')
    scaling = model._scaling
    if (not isinstance(scaling, torch.nn.Parameter) or scaling.ndim != 2 or
            scaling.shape[1] != 3 or not bool(torch.isfinite(scaling).all())):
        raise ValueError('World-scale safety needs finite log-scale parameters [N,3]')
    maximum = float(scene_extent) * float(maximum_extent_fraction)
    log_maximum = scaling.new_tensor(maximum).log()
    clipped = scaling > log_maximum
    rows = clipped.any(dim=1)
    before_log_max = float(scaling.max()) if scaling.numel() else float('-inf')
    if bool(clipped.any()):
        scaling.clamp_(max=log_maximum)
        optimizer = getattr(model, 'optimizer', None)
        if optimizer is None:
            raise ValueError('World-scale safety requires the live Gaussian optimizer')
        state = optimizer.state.get(scaling, {})
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            value = state.get(name)
            if value is not None:
                if value.shape != scaling.shape or not bool(torch.isfinite(value).all()):
                    raise ValueError(f'Invalid scale Adam state: {name}')
                value.masked_fill_(clipped, 0)
    if (not bool(torch.isfinite(scaling).all()) or
            bool((scaling > log_maximum + 2 * torch.finfo(scaling.dtype).eps).any())):
        raise ValueError('World-scale projection failed')
    return dict(maximum_world_scale=maximum,
                maximum_extent_fraction=float(maximum_extent_fraction),
                clipped_rows=int(rows.sum()),
                clipped_components=int(clipped.sum()),
                before_log_scale_max=before_log_max,
                after_log_scale_max=(float(scaling.max()) if scaling.numel()
                                     else float('-inf')))


@torch.no_grad()
def accumulate_extra_densification_views(module, packages: Iterable[dict]):
    """Add extra-view evidence before the native one-shot densification event.

    The first package is deliberately excluded here. The native
    ``densify_gaussians`` call consumes it and performs the sole topology event
    for the global step after all extra packages have contributed statistics.
    """
    packages = list(packages)
    for package in packages[1:]:
        visible = package['visibility_filter']
        radii = package['radii']
        module.gaussians.max_radii2D[visible] = torch.maximum(
            module.gaussians.max_radii2D[visible], radii[visible])
        module.gaussians.add_densification_stats(
            package['viewspace_points'], visible)


@torch.no_grad()
def coverage_budgeted_selection(gradient_sum, observations, sample_count, scaling,
                                percent_dense, scene_extent, grad_threshold,
                                remaining):
    """Choose a deterministic, coverage-weighted subset within a hard GS budget.

    The native statistic is mean screen-space gradient over visible samples.
    Multiplying it by the square root of visibility frequency keeps the native
    units while preferring errors supported across the sampled camera stream.
    Clone and two-way split each add exactly one net Gaussian, so ``remaining``
    is also the maximum number of selected parents.
    """
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int) or
            sample_count < 1 or isinstance(remaining, bool) or
            not isinstance(remaining, int) or remaining < 0):
        raise ValueError('Coverage-budgeted densification needs valid sample and slot counts')
    if (gradient_sum.ndim != 2 or gradient_sum.shape[1:] != (1,) or
            observations.shape != gradient_sum.shape or scaling.ndim != 2 or
            scaling.shape[0] != len(gradient_sum) or
            not bool(torch.isfinite(gradient_sum).all()) or
            not bool(torch.isfinite(observations).all()) or
            not bool(torch.isfinite(scaling).all()) or
            bool((observations < 0).any())):
        raise ValueError('Coverage-budgeted densification received invalid aligned statistics')
    denominator = observations.clamp_min(1)
    native = gradient_sum / denominator
    native = torch.where(observations > 0, native, torch.zeros_like(native)).flatten()
    support = (observations.flatten() / float(sample_count)).clamp(0, 1)
    score = native * support.sqrt()
    scale = scaling.max(dim=1).values
    small = scale <= float(percent_dense) * float(scene_extent)
    eligible = score >= float(grad_threshold)
    selected = torch.zeros(len(score), dtype=torch.bool, device=score.device)
    selected[_top_candidates(eligible, score, remaining)] = True
    return dict(selected=selected, clone=selected & small, split=selected & ~small,
                native_gradient=native, coverage=support, score=score,
                eligible=eligible)


@torch.no_grad()
def camera_frontier_budgeted_selection(gradient_sum, observations,
                                       camera_gradient_sum, camera_observations,
                                       scaling, percent_dense, scene_extent,
                                       grad_threshold, remaining, frontier_fraction,
                                       sample_count,
                                       camera_offset=0):
    """Re-rank only coverage-safe candidates with rotating camera quotas.

    A per-camera gradient is not sufficient evidence for changing shared 3D
    topology: a one-view outlier can otherwise consume a quota, be repeatedly
    split, and become a large held-out-view occluder.  The original multi-view
    coverage score therefore remains the sole eligibility gate.  Camera
    frontiers only order that safe set; an unfilled quota stays unfilled until
    the global coverage-score pass, and the strict cap is allowed to underfill.
    """
    if (camera_gradient_sum.ndim != 2 or
            camera_observations.shape != camera_gradient_sum.shape or
            camera_gradient_sum.shape[1] != len(gradient_sum) or
            not bool(torch.isfinite(camera_gradient_sum).all()) or
            not bool(torch.isfinite(camera_observations).all()) or
            bool((camera_observations < 0).any()) or
            isinstance(frontier_fraction, bool) or
            not isinstance(frontier_fraction, (int, float)) or
            not 0 < frontier_fraction <= 1):
        raise ValueError('Camera-frontier statistics or fraction are invalid')
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
        raise ValueError('Camera-frontier remaining budget is invalid')
    # Reuse the baseline implementation so frontier and non-frontier paths
    # cannot silently diverge on either the support definition or threshold.
    baseline = coverage_budgeted_selection(
        gradient_sum, observations, sample_count, scaling, percent_dense,
        scene_extent, grad_threshold, 0)
    native = baseline['native_gradient']
    coverage = baseline['coverage']
    coverage_score = baseline['score']
    safe_eligible = baseline['eligible']
    camera_mean = torch.where(camera_observations > 0,
        camera_gradient_sum / camera_observations.clamp_min(1),
        torch.zeros_like(camera_gradient_sum))
    presence = camera_observations.gt(0).sum(0).clamp_min(1)
    frontier_score = camera_mean / presence.to(camera_mean).sqrt()[None]
    camera_eligible = (camera_observations > 0) & safe_eligible[None]
    camera_count = len(camera_mean)
    if camera_count < 1:
        raise ValueError('Camera-frontier selection needs at least one training camera')
    if (isinstance(camera_offset, bool) or not isinstance(camera_offset, int) or
            not 0 <= camera_offset < camera_count):
        raise ValueError('Camera-frontier rotation offset is invalid')
    reserve = min(remaining, max(min(remaining, camera_count),
                                 int(round(remaining * float(frontier_fraction)))))
    base, extra = divmod(reserve, camera_count)
    selected = torch.zeros(len(native), dtype=torch.bool, device=native.device)
    allocations = [0] * camera_count
    for turn in range(camera_count):
        index = (camera_offset + turn) % camera_count
        quota = base + (turn < extra)
        chosen = _top_candidates(camera_eligible[index] & ~selected, frontier_score[index], quota)
        selected[chosen] = True
        allocations[index] = len(chosen)
    frontier_selected = selected.clone()
    slots = remaining - int(selected.sum())
    if slots:
        selected[_top_candidates(safe_eligible & ~selected, coverage_score, slots)] = True
    scale = scaling.max(dim=1).values
    small = scale <= float(percent_dense) * float(scene_extent)
    selected_below_gate = selected & ~safe_eligible
    if bool(selected_below_gate.any()):
        raise ValueError('Camera frontier selected outside the multi-view coverage gate')
    return dict(selected=selected, clone=selected & small, split=selected & ~small,
                native_gradient=native,
                coverage=coverage, score=coverage_score,
                eligible=safe_eligible, frontier_selected=frontier_selected,
                global_selected=selected & ~frontier_selected,
                camera_allocations=allocations, frontier_reserve=reserve,
                frontier_fraction=float(frontier_fraction),
                camera_offset=int(camera_offset),
                frontier_score=frontier_score,
                selected_below_coverage_gate=selected_below_gate)


@torch.no_grad()
def accumulate_camera_frontier_stats(module, packages):
    names = getattr(module, '_frontier_camera_names', None)
    gradient = getattr(module, '_frontier_gradient_sum', None)
    observations = getattr(module, '_frontier_observations', None)
    if not names or gradient is None or observations is None:
        raise ValueError('Camera-frontier accumulators were not initialized')
    lookup = {name: index for index, name in enumerate(names)}
    count = len(module.gaussians.get_xyz)
    if gradient.shape != observations.shape or gradient.shape != (len(names), count):
        raise ValueError('Camera-frontier accumulator layout differs from current GS')
    for package in packages:
        name = package.get('_static_camera_name')
        if name not in lookup:
            raise ValueError(f'Unknown static training camera in frontier statistics: {name}')
        visible = package['visibility_filter']
        points = package['viewspace_points']
        values = points.grad
        if values is None:
            raise ValueError('Camera-frontier statistics require view-space gradients')
        if len(values) != len(visible):
            values = values[:len(visible)]
        index = lookup[name]
        gradient[index, visible] += torch.norm(values[visible, :2], dim=-1)
        observations[index, visible] += 1


def _quantiles(values):
    if not len(values):
        return None
    result = torch.quantile(values.float(), values.new_tensor([0., .25, .5, .75, 1.]))
    return [float(value) for value in result]


def progressive_densification_cap(current_step, from_step, until_step, interval,
                                  start_cap, final_cap):
    """Linearly expose the population budget over actual topology events."""
    values = (current_step, from_step, until_step, interval, start_cap, final_cap)
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in values) or
            interval < 1 or from_step < 0 or until_step <= from_step or
            start_cap < 1 or final_cap < start_cap):
        raise ValueError('Invalid progressive densification schedule')
    first = (from_step // interval + 1) * interval
    last = ((until_step - 1) // interval) * interval
    if first > last or current_step < first or current_step > last or current_step % interval:
        raise ValueError('Progressive cap requested outside an actual densification event')
    events = (last - first) // interval + 1
    index = (current_step - first) // interval + 1
    return start_cap + (final_cap - start_cap) * index // events


@torch.no_grad()
def coverage_budgeted_densify_gaussians(module, current_step, visibility_filter,
                                         viewspace_point_tensor, radii, sample_count,
                                         population_cap=None, frontier_fraction=None):
    """Native topology/pruning with coverage scoring and an exact population cap."""
    model = module.gaussians
    model.max_radii2D[visibility_filter] = torch.maximum(
        model.max_radii2D[visibility_filter], radii[visibility_filter])
    model.add_densification_stats(viewspace_point_tensor, visibility_filter)

    cfg = module.config.densify
    started = current_step > cfg.from_step
    at_interval = current_step % cfg.interval == 0
    event = bool(started and at_interval)
    report = None
    if event:
        before = len(model.get_xyz)
        cap = cfg.max_gaussians if population_cap is None else population_cap
        if (isinstance(cap, bool) or not isinstance(cap, int) or
                cap < 1 or cap > cfg.max_gaussians or before > cap):
            raise ValueError(f'Population {before} exceeds invalid/current strict cap {cap}')
        if frontier_fraction is None:
            stats = coverage_budgeted_selection(
                model.xyz_gradient_accum, model.denom, sample_count,
                model.get_scaling, model.percent_dense, module.datamodule.cameras_extent,
                cfg.grad_threshold, cap - before)
        else:
            first = (cfg.from_step // cfg.interval + 1) * cfg.interval
            offset = ((current_step - first) // cfg.interval) % len(module._frontier_camera_names)
            stats = camera_frontier_budgeted_selection(
                model.xyz_gradient_accum, model.denom,
                module._frontier_gradient_sum, module._frontier_observations,
                model.get_scaling, model.percent_dense, module.datamodule.cameras_extent,
                cfg.grad_threshold, cap - before, frontier_fraction, sample_count, offset)
        selected = stats['selected']
        # Feed a binary selection carrier into the unchanged native clone/split
        # implementations. Both operations remain responsible for optimizer
        # tensor extension, split sampling and parent removal.
        carrier = selected.to(model._xyz).unsqueeze(1)
        model.densify_and_clone(carrier, .5, module.datamodule.cameras_extent)
        model.densify_and_split(carrier, .5, module.datamodule.cameras_extent)

        after_densify = len(model.get_xyz)
        exceed_reset_interval = current_step > cfg.opacity_reset_interval
        size_threshold = 20 if exceed_reset_interval else None
        prune = (model.get_opacity < .005).squeeze()
        if size_threshold:
            prune |= (model.max_radii2D > size_threshold)
            prune |= model.get_scaling.max(dim=1).values > .1 * module.datamodule.cameras_extent
        pruned = int(prune.sum())
        model.prune_points(prune)
        after = len(model.get_xyz)
        if after > cap or after_densify > cap:
            raise ValueError('Coverage-budgeted densification exceeded its exact GS cap')
        candidates = stats['eligible']
        report = dict(step=int(current_step), pre_gaussians=before,
            scheduled_cap=int(cap), configured_final_cap=int(cfg.max_gaussians),
            remaining_slots=int(cap-before),
            eligible_count=int(candidates.sum()), selected_count=int(selected.sum()),
            selected_clone_count=int(stats['clone'].sum()),
            selected_split_count=int(stats['split'].sum()),
            post_densify_gaussians=after_densify, pruned_count=pruned,
            post_gaussians=after, sample_count=int(sample_count),
            coverage_quantiles_eligible=_quantiles(stats['coverage'][candidates]),
            coverage_quantiles_selected=_quantiles(stats['coverage'][selected]),
            score_quantiles_eligible=_quantiles(stats['score'][candidates]),
            score_quantiles_selected=_quantiles(stats['score'][selected]),
            score=('multi-view coverage gate; per-camera frontier re-rank then coverage score'
                   if frontier_fraction is not None else
                   'mean view-space gradient * sqrt(visible samples / all camera samples)'),
            selection=('coverage-safe equal camera quotas plus stable coverage-score global top-k'
                       if frontier_fraction is not None else
                       'stable descending top-k under exact net-population slots'))
        if frontier_fraction is not None:
            report.update(frontier_fraction=stats['frontier_fraction'],
                frontier_reserve=stats['frontier_reserve'],
                camera_rotation_offset=stats['camera_offset'],
                frontier_selected_count=int(stats['frontier_selected'].sum()),
                global_selected_count=int(stats['global_selected'].sum()),
                selected_below_coverage_gate_count=int(
                    stats['selected_below_coverage_gate'].sum()),
                camera_frontier_allocations=stats['camera_allocations'],
                camera_names=list(module._frontier_camera_names))
        torch.cuda.empty_cache()

    white_background = module.datamodule.config.white_background
    at_reset_interval = current_step % cfg.opacity_reset_interval == 0
    at_densify_start = current_step == cfg.from_step
    if at_reset_interval or (white_background and at_densify_start):
        model.reset_opacity()
    return event, report


def configure_static_parameters(model, colour_only, preserve_sh_degree=None):
    """Set trainability on restored leaves without changing optimizer references."""
    if preserve_sh_degree is not None and not colour_only:
        raise ValueError('Preserving low-order SH requires colour-only fitting')
    for name in STATIC_FIELDS:
        value = getattr(model, name)
        if not value.is_leaf:
            raise ValueError(f'Static checkpoint field is not a leaf: {name}')
        value.grad = None
        if preserve_sh_degree is None:
            trainable = not colour_only or name in COLOUR_FIELDS
        else:
            trainable = name == '_features_rest'
        value.requires_grad_(trainable)
        if not any(value is parameter for group in model.optimizer.param_groups
                   for parameter in group['params']):
            raise ValueError(f'Static optimizer does not reference restored field: {name}')


def install_high_order_only_training(model, preserve_sh_degree):
    """Freeze DC/low-order SH slices, including their existing Adam momentum."""
    if preserve_sh_degree not in range(3):
        raise ValueError('preserve_sh_degree must be between 0 and 2')
    rest = model._features_rest
    preserved = (preserve_sh_degree + 1) ** 2 - 1
    if rest.ndim != 3 or rest.shape[1] <= preserved:
        raise ValueError('Checkpoint has no higher-order SH coefficients to fit')
    groups = [group for group in model.optimizer.param_groups
              if any(parameter is rest for parameter in group['params'])]
    if len(groups) != 1:
        raise ValueError('Could not uniquely identify the SH-rest optimizer group')
    if groups[0].get('weight_decay', 0) != 0:
        raise ValueError('High-order-only fitting requires zero SH weight decay')
    state = model.optimizer.state.get(rest, {})
    for name, value in state.items():
        if not torch.is_tensor(value) or value.ndim == 0:
            continue
        if tuple(value.shape) != tuple(rest.shape):
            raise ValueError(f'Unexpected SH optimizer tensor {name}: {tuple(value.shape)}')
        value[:, :preserved].zero_()

    def mask_low_order(gradient):
        gradient = gradient.clone()
        gradient[:, :preserved].zero_()
        return gradient

    return rest.register_hook(mask_low_order), preserved


def tensor_sha256(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def static_field_digests(model):
    return {name: tensor_sha256(getattr(model, name)) for name in STATIC_FIELDS}


def verify_colour_only(before, after):
    changed = [name for name in STATIC_FIELDS if before[name] != after[name]]
    if any(name not in COLOUR_FIELDS for name in changed):
        raise ValueError(f'Colour-only fit changed frozen attributes: {changed}')
    if not changed:
        raise ValueError('Colour-only fit did not change any SH weights')
    return changed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate the static pipeline and config without training')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--name')
    parser.add_argument('--output-dir',type=Path,
        help='Override the config output root so a candidate never reuses a historical run')
    parser.add_argument('--data-root',type=Path,
        help='Override the frame-0 scene data root in the config')
    parser.add_argument('--sh-degree',type=int,choices=range(4),
        help='Override the static model SH degree for a new shared-reference candidate')
    parser.add_argument('--num-init-steps',type=int,
        help='Override the static fit length (for example a fixed 10k recipe across scenes)')
    parser.add_argument('--max-gaussians',type=int,
        help='Override only the native densification population cap')
    parser.add_argument('--views-per-step',type=int,default=1,
        help='Average this many training views into each static Adam step; '
             'the checkpoint step count and topology-event count stay unchanged')
    parser.add_argument('--coverage-budgeted-densification',action='store_true',
        help='Use coverage-weighted stable top-k densification with a strict max-gaussians cap; '
             'intended for fresh multi-view static training only')
    parser.add_argument('--densify-start-cap',type=int,
        help='With coverage budgeting, expose max-gaussians linearly from this first-frame '
             'population through the final scheduled topology event')
    parser.add_argument('--camera-frontier-fraction',type=float,
        help='Within the multi-view coverage-safe set, reserve this fraction of each '
             'strict event budget as equal per-camera frontier quotas, then fill by '
             'the baseline coverage score; unsupported quotas may remain empty')
    parser.add_argument('--max-world-scale-extent',type=float,
        help='After every static Adam update, cap each Gaussian covariance axis to this '
             'fraction of the scene extent and clear clipped scale-Adam components; '
             '0.1 matches the native large-world-splat safety definition')
    parser.add_argument('--colour-only', action='store_true',
        help='Fit initial SH only; keep xyz, scale, rotation and opacity bit-exact')
    parser.add_argument('--preserve-sh-degree', type=int, choices=range(3),
        help='With --colour-only, preserve DC and SH through this degree bit-exact; '
             'fit only newly added higher-order coefficients')
    args=parser.parse_args()
    from omegaconf import OmegaConf
    from libgs.utils.config import to_structured
    import pipeline.first_frame as native
    from pipeline.first_frame import trainer as trainer_native

    class MultiViewModule(native.Module):
        @torch.no_grad()
        def post_training_step_init(self, render_pkg):
            if not isinstance(render_pkg, (tuple, list)):
                return super().post_training_step_init(render_pkg)
            packages = list(render_pkg)
            if len(packages) != args.views_per_step or not packages:
                raise ValueError('Static multi-view package count changed within a step')
            if self.global_step < self.config.densify.until_step:
                # View-space tensors received unscaled per-view backward
                # gradients. Accumulating each observation before the native
                # call preserves its per-visible-view statistic while allowing
                # exactly one topology event at this global step.
                if args.camera_frontier_fraction is not None:
                    accumulate_camera_frontier_stats(self, packages)
                accumulate_extra_densification_views(self, packages)
                package = packages[0]
                if args.coverage_budgeted_densification:
                    samples = getattr(self, '_coverage_densification_samples', 0) + len(packages)
                    cfg = self.config.densify
                    topology_event = (self.global_step > cfg.from_step and
                                      self.global_step % cfg.interval == 0)
                    cap = cfg.max_gaussians
                    if args.densify_start_cap is not None and topology_event:
                        cap = progressive_densification_cap(
                            self.global_step, cfg.from_step, cfg.until_step, cfg.interval,
                            args.densify_start_cap, cfg.max_gaussians)
                    event, report = coverage_budgeted_densify_gaussians(
                        self, self.global_step, package['visibility_filter'],
                        package['viewspace_points'], package['radii'], samples,
                        population_cap=cap,
                        frontier_fraction=args.camera_frontier_fraction)
                    self._coverage_densification_samples = 0 if event else samples
                    if report is not None:
                        self._coverage_densification_events.append(report)
                    if event and args.camera_frontier_fraction is not None:
                        shape=(len(self._frontier_camera_names),len(self.gaussians.get_xyz))
                        self._frontier_gradient_sum=self.gaussians._xyz.new_zeros(shape)
                        self._frontier_observations=self.gaussians._xyz.new_zeros(shape)
                else:
                    self.densify_gaussians(
                        self.global_step,
                        package['visibility_filter'],
                        package['viewspace_points'],
                        package['radii'])
            self.gaussians.optimizer.step()
            if args.max_world_scale_extent is not None:
                item = enforce_world_scale_cap(
                    self.gaussians, self.datamodule.cameras_extent,
                    args.max_world_scale_extent)
                state = getattr(self, '_world_scale_safety')
                state['calls'] += 1
                state['total_clipped_rows'] += item['clipped_rows']
                state['total_clipped_components'] += item['clipped_components']
                state['max_clipped_rows_per_step'] = max(
                    state['max_clipped_rows_per_step'], item['clipped_rows'])
                if item['clipped_rows']:
                    state['steps_with_clipping'] += 1
                    if state['first_clipped_step'] is None:
                        state['first_clipped_step'] = int(self.global_step)
                    state['last_clipped_step'] = int(self.global_step)
                state['last_projection'] = item
            self.gaussians.optimizer.zero_grad(set_to_none=True)

    class ResumeTrainer(native.Trainer):
        def setup(self, module, ckpt_path=None):
            # The native loader preserves the saved device. Pruning exports CPU
            # checkpoints, so restore them onto the initialized training device
            # before model parameters determine the fetcher's image device.
            super().setup(module, None)
            if ckpt_path is not None:
                checkpoint = torch.load(ckpt_path, map_location=self.device,
                                        weights_only=False)
                self.global_step = checkpoint['global_step']
                self.datamodule.on_load_checkpoint(checkpoint)
                self.module.on_load_checkpoint(checkpoint)

        def frame_training_loop(self):
            if not self.is_first_frame:
                raise ValueError('This entry is limited to the static first frame')
            start_step, end_step = self.global_step, self.num_steps
            if start_step >= end_step:
                raise ValueError('Static fitting needs remaining training steps')
            configure_static_parameters(self.module.gaussians, args.colour_only,
                                        args.preserve_sh_degree)
            before = static_field_digests(self.module.gaussians)
            sh_hook = None
            preserved = None
            preserved_before = None
            if args.preserve_sh_degree is not None:
                sh_hook, preserved = install_high_order_only_training(
                    self.module.gaussians, args.preserve_sh_degree)
                preserved_before = tensor_sha256(self.module.gaussians._features_rest[:, :preserved])
            trainable = [name for name in STATIC_FIELDS
                         if getattr(self.module.gaussians, name).requires_grad]
            if self.global_step > 0:
                self.module.timer.clock('frame').start()
            try:
                result = (super().frame_training_loop()
                          if args.views_per_step == 1 else
                          self._multi_view_frame_training_loop())
            finally:
                if sh_hook is not None:
                    sh_hook.remove()
            after = static_field_digests(self.module.gaussians)
            changed = (verify_colour_only(before, after) if args.colour_only else
                       [name for name in STATIC_FIELDS if before[name] != after[name]])
            preserved_after = None
            if preserved is not None:
                preserved_after = tensor_sha256(self.module.gaussians._features_rest[:, :preserved])
                if preserved_before != preserved_after:
                    raise ValueError('Preserved low-order SH changed during high-order fit')
            report = dict(colour_only=args.colour_only, start_step=start_step,
                end_step=end_step, trainable_fields=trainable, changed_fields=changed,
                views_per_step=args.views_per_step,
                coverage_budgeted_densification=args.coverage_budgeted_densification,
                camera_frontier_fraction=args.camera_frontier_fraction,
                coverage_densification_events=getattr(self.module,'_coverage_densification_events',[]),
                world_scale_safety=getattr(self.module, '_world_scale_safety',
                    dict(enabled=False)),
                image_samples=(end_step-start_step)*args.views_per_step,
                before_sha256=before, after_sha256=after,
                preserve_sh_degree=args.preserve_sh_degree,
                preserved_sh_coefficients=preserved,
                preserved_sh_sha256_before=preserved_before,
                preserved_sh_sha256_after=preserved_after,
                frozen_fields_bit_exact=all(before[name] == after[name]
                    for name in STATIC_FIELDS if name not in trainable))
            (self.output_dir / 'first_frame_trainability.json').write_text(json.dumps(report, indent=2)+'\n')
            print('STATIC_TRAINABILITY_VERIFIED', json.dumps(report), flush=True)
            return result

        def _multi_view_frame_training_loop(self):
            """Native static loop with one averaged Adam update per global step."""
            train_dataloader = self.datamodule.train_dataloader()
            self.module._coverage_densification_samples = 0
            self.module._coverage_densification_events = []
            self.module._world_scale_safety = dict(
                enabled=args.max_world_scale_extent is not None,
                maximum_extent_fraction=args.max_world_scale_extent,
                calls=0, steps_with_clipping=0, total_clipped_rows=0,
                total_clipped_components=0, max_clipped_rows_per_step=0,
                first_clipped_step=None, last_clipped_step=None,
                last_projection=None)
            if args.camera_frontier_fraction is not None:
                dataset=self.datamodule.get_train_dataset()
                names=sorted(Path(str(item.path)).name for item in dataset.items)
                if not names or len(names)!=len(set(names)):
                    raise ValueError('Camera-frontier densification requires unique training camera names')
                self.module._frontier_camera_names=names
                shape=(len(names),len(self.module.gaussians.get_xyz))
                self.module._frontier_gradient_sum=self.module.gaussians._xyz.new_zeros(shape)
                self.module._frontier_observations=self.module.gaussians._xyz.new_zeros(shape)
            if (args.coverage_budgeted_densification and
                    len(self.module.gaussians.get_xyz) > self.module.config.densify.max_gaussians):
                raise ValueError('Initial GS population already exceeds the strict densification cap')
            if (args.densify_start_cap is not None and
                    len(self.module.gaussians.get_xyz) > args.densify_start_cap):
                raise ValueError('Initial GS population already exceeds densify-start-cap')
            trainer_native.logging.info(
                'Number of views in training dataset: %d; views per static step: %d',
                len(train_dataloader), args.views_per_step)
            move = lambda value: value.to(self.device, True)
            fetcher = trainer_native.Fetcher(
                train_dataloader, self.config.cache_data_on_device, move)
            metrics_ema = {'loss': 0., 'psnr': 0.}
            progress = trainer_native.tqdm(
                range(self.global_step, self.num_steps),
                desc=f'Training frame {self.current_frame}')

            def update_metric(name, value):
                metrics_ema[name] = .6 * metrics_ema[name] + .4 * value

            for self.global_step in range(self.global_step + 1, self.num_steps + 1):
                self.module.pre_training_step()
                packages = []
                mean_loss = 0.
                mean_psnr = 0.
                metrics = None
                for _ in range(args.views_per_step):
                    self.module.timer.clock('data').start()
                    viewpoint = fetcher.next()
                    self.module.timer.clock('data').stop()
                    loss, metrics, package = self.module.training_step(
                        viewpoint, self.global_step)
                    self.module.timer.clock('backward').start()
                    # Keep view-space gradients in their native one-view units
                    # for densification. Only optimizer parameter gradients are
                    # averaged after all views have contributed.
                    loss.backward()
                    self.module.timer.clock('backward').stop()
                    mean_loss += float(loss.detach()) / args.views_per_step
                    mean_psnr += float(metrics['psnr'].detach()) / args.views_per_step
                    package['_static_camera_name']=Path(str(viewpoint.path)).name
                    packages.append(package)
                average_optimizer_gradients(
                    self.module.gaussians.optimizer, args.views_per_step)

                self.module.timer.clock('progress updating').start()
                update_metric('loss', mean_loss)
                update_metric('psnr', mean_psnr)
                if self.global_step % 10 == 0:
                    displayed = {key: f'{value:.4f}' for key, value in metrics_ema.items()}
                    displayed['gs'] = metrics['gs']
                    progress.set_postfix(displayed)
                    progress.update(10)
                if self.global_step == self.num_steps:
                    progress.close()
                self.module.timer.clock('progress updating').stop()

                self.module.post_training_step(packages)
                saving = self.config.saving_ckpt_steps + [self.num_steps]
                if self.global_step in saving:
                    self.save_checkpoint()
                if self.global_step in self.config.testing_steps + [self.num_steps]:
                    self.validation_loop()
            self.global_step = 0

    class FirstFramePipeline(native.Pipeline):
        def setup_trainer_and_modules(self):
            output_dir = Path(self.config.output_dir) / self.config.experiment_name
            self.datamodule = native.DataModule(self.config.data)
            self.module = MultiViewModule(self.config.module, datamodule=self.datamodule)
            self.trainer = ResumeTrainer(self.config.trainer, output_dir=output_dir)

    config,extra=to_structured(OmegaConf.load(args.config.resolve()),native.Config)
    if extra:raise ValueError(f'Unrecognized first-frame config: {extra}')
    if config.data.extra_dataset_kwargs.get('num_frames')!=1:
        raise ValueError('This entry is limited to one static first frame')
    if args.colour_only and (config.module.noise_scale is not None or
                            config.module.densify.until_step != 0):
        raise ValueError('Colour-only fitting requires noise_scale=null and densify.until_step=0')
    if args.preserve_sh_degree is not None and not args.colour_only:
        raise ValueError('--preserve-sh-degree requires --colour-only')
    if args.preserve_sh_degree is not None and args.checkpoint is None:
        raise ValueError('--preserve-sh-degree requires --checkpoint')
    if (isinstance(args.views_per_step, bool) or args.views_per_step < 1):
        raise ValueError('--views-per-step must be a positive integer')
    if args.views_per_step > 1 and (args.checkpoint is not None or args.colour_only):
        raise ValueError('Static multi-view fitting requires a fresh full-model run')
    if args.coverage_budgeted_densification and args.views_per_step < 2:
        raise ValueError('Coverage-budgeted densification requires multi-view static fitting')
    if args.densify_start_cap is not None and not args.coverage_budgeted_densification:
        raise ValueError('--densify-start-cap requires coverage-budgeted densification')
    if args.densify_start_cap is not None and args.densify_start_cap < 1:
        raise ValueError('--densify-start-cap must be positive')
    if args.camera_frontier_fraction is not None and (
            not args.coverage_budgeted_densification or
            not 0 < args.camera_frontier_fraction <= 1):
        raise ValueError('--camera-frontier-fraction needs coverage budgeting and must be in (0,1]')
    if args.max_world_scale_extent is not None and (
            not math.isfinite(args.max_world_scale_extent) or
            not 0 < args.max_world_scale_extent <= 1 or
            args.views_per_step < 2 or args.checkpoint is not None or args.colour_only):
        raise ValueError('--max-world-scale-extent needs a value in (0,1] and a fresh '
                         'multi-view full-model run')
    if args.checkpoint is not None:config.ckpt_path=args.checkpoint.resolve()
    if args.name is not None:config.experiment_name=args.name
    if args.output_dir is not None:config.output_dir=args.output_dir.resolve()
    if args.data_root is not None:config.data.root=str(args.data_root.resolve())
    if not config.data.root:
        raise ValueError('Set data.root in the config or pass --data-root')
    if args.sh_degree is not None:config.module.sh_degree=args.sh_degree
    if args.num_init_steps is not None:
        if args.num_init_steps < 1:raise ValueError('--num-init-steps must be positive')
        config.trainer.num_init_steps=args.num_init_steps
    if args.max_gaussians is not None:
        if args.max_gaussians < 1:raise ValueError('--max-gaussians must be positive')
        config.module.densify.max_gaussians=args.max_gaussians
    if args.validate_only:
        print(f'Static pipeline: pipeline.first_frame; steps: {config.trainer.num_init_steps}; '
              f'GS cap: {config.module.densify.max_gaussians}; views/step: {args.views_per_step}; '
              f'data: {config.data.root}')
        return
    FirstFramePipeline(config).run()


if __name__=='__main__':main()
