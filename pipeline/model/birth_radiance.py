"""Training-camera calibration of coherent intrinsic DC residuals.

Geometry and directional SH stay fixed. A real quantized forward supplies the
residual projection; native, zero-background colour probes supply each point's
occlusion-aware diagonal response. The source's directional color envelope
bounds the solve before a full-scene quantized line search. No optimizer or
codec state changes.
"""

from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from .birth_quantization import RAW_FIELDS, model_raw
from .residual_birth import RawView


Y00 = 0.28209479177387814
RESPONSE_EPS = 1.e-12
MAX_BACKTRACKS = 8


def _coherent_dc_delta(projection, response, eps=RESPONSE_EPS):
    """Aggregate (view, point, RGB) ``sum(J*r)`` and ``sum(J**2)``.

The returned delta is in SH-DC coefficient units, not RGB units. At least two
views must actually respond in a channel. Cross-point Hessian terms are omitted;
the later full-scene line search accounts for their effect on acceptance.
"""
    if projection.shape != response.shape or projection.ndim != 3 or projection.shape[-1] != 3:
        raise ValueError('DC projection and response must both have shape (views, points, 3)')
    projection, response = projection.double(), response.double()
    valid = (torch.isfinite(projection) & torch.isfinite(response) & (response > eps))
    h = torch.where(valid, response, 0.)
    b = torch.where(valid, projection, 0.)
    support = valid.sum(0)
    total_h, total_b = h.sum(0), b.sum(0)
    mean = total_b / total_h.clamp_min(eps)
    second_moment = torch.where(valid, b.square() / h.clamp_min(eps), 0.).sum(0)
    coherence = (total_b.square() / (total_h * second_moment + eps)).clamp(0, 1)
    eligible = support >= 2
    delta = torch.where(eligible, coherence * mean, 0.)
    return delta, dict(coherence=coherence, support=support, eligible=eligible,
                       valid=valid, mean=mean, response=h)


@contextmanager
def _isolated_renderer(renderer):
    """Keep response probes from changing subsequent renderer diagnostics."""
    previous = renderer.gaussians
    attributes = {name: getattr(renderer, name) for name in
                  ('collect_contributions', 'last_contributions') if hasattr(renderer, name)}
    try:
        if 'collect_contributions' in attributes:
            renderer.collect_contributions = False
        yield
    finally:
        renderer.gaussians = previous
        for name, value in attributes.items():
            setattr(renderer, name, value)


@torch.no_grad()
def _response_scene(module):
    """Snapshot exactly the cohort's rendered population, without source remaps."""
    tx = module.birth_transaction
    fixed = getattr(tx, 'fixed_reference', None)
    if fixed is not None:
        old = {name: fixed[name].detach().clone() for name in RAW_FIELDS}
        active = maximum = module.config.sh_degree
    else:
        old = model_raw(module.gaussians)
        keep = torch.ones(len(old['xyz']), dtype=torch.bool, device=old['xyz'].device)
        parents = tx.replacements
        keep[parents[parents >= 0]] = False
        old = {name: value[keep] for name, value in old.items()}
        active = getattr(module.gaussians, 'active_sh_degree', module.config.sh_degree)
        maximum = getattr(module.gaussians, 'max_sh_degree', module.config.sh_degree)
    new = module.make_birth_view()
    if new is None:
        raise ValueError('Nonempty births require a quantized birth view')
    new_raw = {name: new.raw_value(name).detach().clone() for name in RAW_FIELDS}
    # Module.forward uses normalized birth rotations, including when their
    # predictor is a previously decoded, quantized quaternion.
    new_raw['rotation'] = new.get_rotation.detach().clone()
    raw = {name: torch.cat((old[name], new_raw[name]), 0) for name in RAW_FIELDS}
    scene = RawView(raw, maximum, module.gaussians)
    scene.active_sh_degree, scene.max_sh_degree = active, maximum
    # Avoid normalizing already normalized birth quaternions a second time.
    scene.get_rotation = torch.cat((F.normalize(old['rotation'], dim=-1), new_raw['rotation']), 0)
    return scene, len(old['xyz'])


def _probe_view(module, view, original_dc):
    """Real STE-quantized MSE derivative, even inside a no_grad cohort hook."""
    births = module.births
    probe = torch.nn.Parameter(original_dc.detach().clone(), requires_grad=True)
    try:
        births._features_dc = probe
        with torch.enable_grad():
            image = module.forward(view)['render']
            loss = (image.clamp(0, 1) - view.image.to(image)).square().mean()
            gradient = (torch.autograd.grad(loss, probe, allow_unused=True)[0]
                        if loss.requires_grad else None)
        if gradient is None:
            gradient = torch.zeros_like(probe)
        # MSE = sum((I-GT)^2)/(3*H*W); r = GT-I.
        projection = gradient.detach()[:, 0, :].double() * (-.5 * image.numel())
        # Match torch.clamp's derivative, which includes its exact boundaries.
        pixel_active = (image.detach() >= 0) & (image.detach() <= 1)
        return loss.detach().double(), projection, pixel_active, image.detach()
    finally:
        # The optimizer retains this exact Parameter and any existing state.
        births._features_dc = original_dc


@torch.no_grad()
def _sh_active(scene, view, old_count):
    color = _preclamp_rgb(scene.get_features[old_count:], scene.get_xyz[old_count:],
                          scene.active_sh_degree, view)
    # At the SH clamp's exact kink we conservatively do not authorize an update.
    return torch.isfinite(color) & (color > 0)


def _preclamp_rgb(features, xyz, degree, view):
    from libgs.utils.sh import eval_sh

    directions = xyz - view.camera_center.to(xyz)[None]
    directions = F.normalize(directions, dim=-1)
    return eval_sh(degree, features.transpose(1, 2), directions) + .5


@torch.no_grad()
def _source_envelope(module, views, scene, old_count):
    """DC box preserving the cohort source's SH range at the current geometry."""
    current = scene.get_features[old_count:]
    xyz = scene.get_xyz[old_count:]
    raw = getattr(module.birth_transaction, 'raw', None)
    if isinstance(raw, dict) and 'sh_dc' in raw and 'sh_rest' in raw:
        prior = torch.cat((raw['sh_dc'], raw['sh_rest']), 1).to(current)
        if prior.shape != current.shape:
            raise ValueError('Cohort-initial source SH must align with the current birth population')
        source = 'transaction.raw_cohort_initial_dc_and_sh_rest'
    else:
        prior = current
        source = 'current_quantized_features_fallback_missing_transaction_prior'
    colors = torch.stack([_preclamp_rgb(current, xyz, scene.active_sh_degree, view)
                          for view in views]).double()
    prior_colors = torch.stack([_preclamp_rgb(prior, xyz, scene.active_sh_degree, view)
                                for view in views]).double()
    if not bool(torch.isfinite(colors).all() & torch.isfinite(prior_colors).all()):
        raise ValueError('Intrinsic radiance envelope received nonfinite preclamp source colors')
    current_min, current_max = colors.amin(0), colors.amax(0)
    prior_low = prior_colors.amin(0).clamp_max(0.)
    prior_high = prior_colors.amax(0).clamp_min(1.)
    # A continuous tx.raw can differ by a quantization fraction from the actual
    # frozen shader. Include its current decoded colors so zero remains feasible.
    low, high = torch.minimum(prior_low, current_min), torch.maximum(prior_high, current_max)
    lower, upper = (low-current_min)/Y00, (high-current_max)/Y00
    expanded = (low != prior_low) | (high != prior_high)
    report = dict(source=source, view_count=len(views),
        meaning='per point/RGB source preclamp SH range at current quantized xyz over all supplied training directions; includes [0,1]',
        zero_step='minimal hull with current quantized preclamp range keeps zero feasible',
        zero_step_expanded_rgb_channels=int(expanded.sum()),
        zero_step_expanded_points=int(expanded.any(-1).sum()),
        zero_step_expansion_abs=_summary(torch.cat(((prior_low-low).flatten(), (high-prior_high).flatten()))),
        low_rgb=low.tolist(), high_rgb=high.tolist(),
        current_rgb_min=_summary(current_min), current_rgb_max=_summary(current_max),
        prior_rgb_min=_summary(prior_colors.amin(0)), prior_rgb_max=_summary(prior_colors.amax(0)),
        delta_lower=_summary(lower), delta_upper=_summary(upper))
    return lower, upper, report


@torch.no_grad()
def _box_quantized_candidate(initial, delta, scale, quantized_view, lower, upper):
    """Project rounded trial symbols inward too, so decoding cannot leave the box.

Normal interior trials keep their original continuous preimage. Boundary trials
use the closest feasible symbol; no half-quantum color overshoot is authorized.
    """
    predictor = quantized_view.predictor['sh_dc']
    step = torch.as_tensor(quantized_view.steps['sh_dc'], device=initial.device, dtype=initial.dtype)
    base_symbols = quantized_view.symbols['sh_dc'].detach()
    decoded_before = quantized_view.raw_value('sh_dc').detach()
    k_low = torch.ceil(lower[:, None] / step.double())
    k_high = torch.floor(upper[:, None] / step.double())
    desired = initial + scale * delta
    symbols = ((desired-predictor)/step).round()
    increment = (symbols-base_symbols).double().maximum(k_low).minimum(k_high).to(initial)
    bounded_symbols = base_symbols + increment
    target = predictor + bounded_symbols * step
    candidate = torch.where(symbols == bounded_symbols, desired, target)
    # Verify actual float-precision decoder arithmetic rather than assuming its
    # result equals baseline + k*step exactly near a box boundary.
    for _ in range(2):
        decoded = predictor + ((candidate-predictor)/step).round() * step
        difference = (decoded-decoded_before).double()[:, 0]
        below, above = difference < lower, difference > upper
        invalid = below | above
        if not bool(invalid.any()):
            return candidate
        inward = increment - above[:, None].to(increment) + below[:, None].to(increment)
        inward = torch.where(increment > 0, inward.clamp_min(0), inward.clamp_max(0))
        increment = torch.where(invalid[:, None], inward, increment)
        target = predictor + (base_symbols+increment) * step
        candidate = torch.where(invalid[:, None], torch.where(increment == 0, initial, target), candidate)
    # Zero is an exact feasible decoded fallback for any residual roundoff.
    decoded = predictor + ((candidate-predictor)/step).round() * step
    difference = (decoded-decoded_before).double()[:, 0]
    invalid = (difference < lower) | (difference > upper)
    return torch.where(invalid[:, None], initial, candidate)


@torch.no_grad()
def _view_response(module, view, scene, old_count, pixel_active, point_mask=None):
    """Three candidates per native render; all points still occlude normally."""
    count = len(scene.get_xyz) - old_count
    response = torch.zeros(count, 3, dtype=torch.float64, device=scene.get_xyz.device)
    active_sh = _sh_active(scene, view, old_count)
    colors = scene.get_xyz.new_zeros(len(scene.get_xyz), 3)
    background = torch.zeros_like(module.background)
    selected = (torch.arange(count, device=colors.device) if point_mask is None else
                point_mask.nonzero(as_tuple=True)[0])
    previous = module.renderer.gaussians
    try:
        module.renderer.gaussians = scene
        for start in range(0, len(selected), 3):
            rows = selected[start:start+3]
            width = len(rows)
            colors.zero_()
            channels = torch.arange(width, device=colors.device)
            colors[old_count + rows, channels] = 1.
            weights = module.renderer(view, background, override_color=colors)['render'][:width]
            if not bool(torch.isfinite(weights).all()):
                raise ValueError('Native color-response render produced nonfinite weights')
            squared = weights.square()
            for channel in range(3):
                h = (squared * pixel_active[channel][None]).sum((-1, -2), dtype=torch.float64)
                response[rows, channel] = h * Y00**2 * active_sh[rows, channel]
    finally:
        module.renderer.gaussians = previous
    return response


@torch.no_grad()
def _errors(module, views):
    return torch.stack([(module.forward(view)['render'].clamp(0, 1) - view.image).square().mean()
                        for view in views]).double()


def _summary(values):
    values = values.detach().double().flatten()
    values = values[torch.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=0., min=0., max=0.)
    return dict(count=len(values), mean=float(values.mean()), min=float(values.min()), max=float(values.max()))


def _view_list(views):
    if isinstance(views, dict):
        return [str(name) for name in views], list(views.values())
    views = list(views)
    return [Path(str(view.path)).name if hasattr(view, 'path') else str(index)
            for index, view in enumerate(views)], views


@torch.no_grad()
def calibrate_intrinsic_dc(module, views, point_mask=None):
    """Calibrate births once at cohort end using all standard training cameras.

    ``views`` is an ordered name->camera dict or a camera sequence. Test cameras
    must not be supplied. The report contains only JSON-compatible values. DC
    updates are in place; the original Parameter, requires_grad flags, SH-rest
    and Adam state survive renderer failure and line-search rejection as well.
    """
    names, views = _view_list(views)
    count = len(module.births._xyz)
    selected = (torch.ones(count, dtype=torch.bool, device=module.births._xyz.device)
                if point_mask is None else torch.as_tensor(point_mask, device=module.births._xyz.device))
    if selected.shape != (count,) or selected.dtype != torch.bool:
        raise ValueError('Intrinsic DC point_mask must be a Boolean vector aligned with births')
    report = dict(method='source_envelope_bounded_intrinsic_dc_v6_1', views=names, view_count=len(views),
        candidate_count=count, selected_candidate_count=int(selected.sum()), status='skipped', reason=None,
        before_per_view_mse=[], after_per_view_mse=[], before_mean_mse=None, after_mean_mse=None,
        scene_max_abs_difference=0., coherence={}, support={}, dc_update={}, source_envelope={},
        line_search=dict(accepted=False, accepted_scale=0., max_attempts=MAX_BACKTRACKS, trials=[]),
        stats_semantics=dict(
            view_scope='caller_supplied_standard_training_cameras_only',
            residual='GT minus clamp(full quantized forward, 0, 1)',
            projection='sum(J*r), recovered from real MSE gradient times -3*H*W/2',
            response='sum(J^2), native alpha*T color responses times Y00, with SH and image clamp masks',
            response_approximation='diagonal across candidates; conservative zero response at SH clamp kink',
            coherence='(sum H*d)^2 / (sum H * sum H*d^2 + eps), d=sum(J*r)/H',
            support='H > 1e-12 in at least two actual responding training views per RGB channel',
            box='coherent Newton DC delta projected to inherited source preclamp RGB envelope; rounded trial symbols projected inward',
            acceptance='strict decrease of mean full quantized training-camera MSE; no per-camera veto',
            dc_units='SH DC coefficient units; SH-rest never changes'))
    if count == 0 or len(views) < 2 or not bool(selected.any()):
        report['reason'] = ('zero_births' if count == 0 else
                            'zero_selected_points' if not bool(selected.any()) else 'fewer_than_two_views')
        return report
    if not getattr(module, 'motion_committed', True) or getattr(module, 'frame_finalized', False):
        raise ValueError('Intrinsic DC calibration requires an unfinished, motion-committed birth cohort')

    original_dc = module.births._features_dc
    initial_dc = original_dc.detach().clone()
    original_flag = original_dc.requires_grad
    committed = False
    try:
        with _isolated_renderer(module.renderer):
            scene, old_count = _response_scene(module)
            losses, projections, responses = [], [], []
            for view in views:
                loss, projection, pixel_active, prediction = _probe_view(module, view, original_dc)
                # This inexpensive guard detects stale sources, old opacity or
                # degree mismatches before using any physical response estimate.
                module.renderer.gaussians = scene
                static_prediction = module.renderer(view, module.background)['render']
                difference = float((static_prediction - prediction).abs().max())
                report['scene_max_abs_difference'] = max(report['scene_max_abs_difference'], difference)
                if not torch.allclose(static_prediction, prediction, atol=2.e-6, rtol=2.e-5):
                    raise ValueError('Intrinsic DC response scene differs from the actual full quantized forward')
                losses.append(loss)
                projections.append(projection)
                responses.append(_view_response(module, view, scene, old_count, pixel_active, selected))
            before = torch.stack(losses)
            if not bool(torch.isfinite(before).all()):
                raise ValueError('Intrinsic DC calibration received nonfinite training MSE')
            report.update(before_per_view_mse=before.tolist(), after_per_view_mse=before.tolist(),
                          before_mean_mse=float(before.mean()), after_mean_mse=float(before.mean()))
            delta, stats = _coherent_dc_delta(torch.stack(projections), torch.stack(responses))
            supported_channels = stats['support'] >= 2
            point_views = stats['valid'].any(-1).sum(0)
            report['coherence'] = _summary(stats['coherence'][supported_channels])
            report['support'] = dict(
                responding_points=int((point_views > 0).sum()),
                multiview_points=int(supported_channels.any(-1).sum()),
                multiview_rgb_channels=int(supported_channels.sum()),
                per_view_responding_points=stats['valid'].any(-1).sum(-1).tolist(),
                per_point_responding_views=point_views.tolist(),
                per_point_rgb_responding_views=stats['support'].tolist(),
                response=_summary(stats['response'][stats['valid']]))
            lower, upper, envelope = _source_envelope(module, views, scene, old_count)
            bounded_delta = delta.maximum(lower).minimum(upper)
            bounded = bounded_delta != delta
            report['source_envelope'] = envelope
            proposed = bounded_delta.to(initial_dc).unsqueeze(1)
            report['dc_update'] = dict(unbounded_abs=_summary(delta.abs()),
                                      bounded_abs=_summary(bounded_delta.abs()),
                                      bounded_points=int(bounded.any(-1).sum()),
                                      bounded_rgb_channels=int(bounded.sum()),
                                      lower_boundary_points=int((bounded & (bounded_delta == lower)).any(-1).sum()),
                                      upper_boundary_points=int((bounded & (bounded_delta == upper)).any(-1).sum()),
                                      proposed_abs=_summary(proposed.abs()),
                                      committed_abs=_summary(torch.zeros_like(proposed)),
                                      quantized_changed_points=0, quantized_changed_components=0)
            if not bool(stats['valid'].any()):
                report['reason'] = 'zero_response'
                return report
            if not bool(proposed.any()):
                report['reason'] = 'no_coherent_multiview_residual'
                return report

            initial_view = module.make_birth_view()
            quantized_before = initial_view.raw_value('sh_dc').detach().clone()
            for attempt in range(MAX_BACKTRACKS):
                scale = .5**attempt
                candidate = _box_quantized_candidate(initial_dc, proposed, scale, initial_view, lower, upper)
                original_dc.copy_(torch.where(selected[:, None, None], candidate, initial_dc))
                quantized = module.make_birth_view().raw_value('sh_dc').detach()
                changed = quantized != quantized_before
                trial = dict(scale=scale, quantized_changed_components=int(changed.sum()),
                             mean_mse=None, per_view_mse=[], accepted=False)
                report['line_search']['trials'].append(trial)
                if not bool(changed.any()):
                    # Along a line back toward the same original DC, smaller
                    # steps cannot re-enter a different uniform quantizer bin.
                    report['reason'] = 'quantized_update_is_zero'
                    break
                after = _errors(module, views)
                finite = torch.isfinite(after)
                trial.update(mean_mse=float(after.mean()) if bool(finite.all()) else None,
                             per_view_mse=[float(value) if bool(valid) else None
                                           for value, valid in zip(after, finite)])
                if bool(finite.all()) and float(after.mean()) < float(before.mean()) - 1.e-10:
                    trial['accepted'] = committed = True
                    report.update(status='accepted', reason='mean_training_mse_improved',
                                  after_per_view_mse=after.tolist(), after_mean_mse=float(after.mean()))
                    report['line_search'].update(accepted=True, accepted_scale=scale)
                    report['dc_update'].update(committed_abs=_summary((original_dc-initial_dc).abs()),
                        quantized_changed_points=int(changed.flatten(1).any(-1).sum()),
                        quantized_changed_components=int(changed.sum()))
                    break
            if not committed and report['reason'] is None:
                report.update(status='rejected', reason='line_search_did_not_improve_mean_training_mse')
            return report
    finally:
        module.births._features_dc = original_dc
        original_dc.requires_grad_(original_flag)
        if not committed:
            original_dc.copy_(initial_dc)
