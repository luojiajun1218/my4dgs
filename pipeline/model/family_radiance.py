"""Coherent, bounded DC contrast for conservative mother/plus/minus families.

Each family has one quantized RGB contrast z, applied as +z/-z to its two
children. The mature mother and the intrinsic child mean never receive a color
update. This intrinsic constraint does not imply equal rendered brightness in
every camera: native alpha/T and the piecewise color clamps still apply.
"""

import torch

from .birth_radiance import (Y00, MAX_BACKTRACKS, RESPONSE_EPS, _coherent_dc_delta,
    _errors, _isolated_renderer, _preclamp_rgb, _response_scene, _sh_active, _summary, _view_list)


def _coherent_family_delta(projection, response):
    """The same weighted coherence ratio, independent of response magnitude.

    Support still requires actual H>RESPONSE_EPS. Normalizing H avoids placing
    an absolute epsilon with H-squared units in the dimensionless coherence.
    """
    _, stats = _coherent_dc_delta(projection, response)
    h, valid = stats['response'], stats['valid']
    tiny = torch.finfo(torch.float64).tiny
    weight = h / h.sum(0).clamp_min(tiny)
    d = torch.where(valid, projection.double()/h.clamp_min(RESPONSE_EPS), 0.)
    mean = (weight*d).sum(0)
    second = (weight*d.square()).sum(0)
    coherence = (mean.square()/second.clamp_min(tiny)).clamp(0, 1)
    delta = torch.where(stats['eligible'], coherence*mean, 0.)
    stats.update(coherence=coherence, mean=mean)
    return delta, stats


def _contrast_response(plus, minus, plus_active, minus_active, pixel_active):
    """Exact within-family diagonal: square the signed response, retaining cross H.

    ``plus/minus`` are (families,H,W) native alpha*T images; SH masks are
    (families,RGB), and the final image-clamp mask is (RGB,H,W).
    """
    response = torch.zeros(len(plus), 3, dtype=torch.float64, device=plus.device)
    for channel in range(3):
        k = (plus * plus_active[:, channel, None, None] -
             minus * minus_active[:, channel, None, None]) * Y00
        response[:, channel] = (k.square() * pixel_active[channel][None]).sum(
            (-1, -2), dtype=torch.float64)
    return response


@torch.no_grad()
def _family_view_response(module, view, scene, old_count, rows, pixel_active):
    """Six children / two RGB renders at a time, without a full image Jacobian."""
    response = torch.zeros(len(rows), 3, dtype=torch.float64, device=scene.get_xyz.device)
    active = _sh_active(scene, view, old_count)
    colors = scene.get_xyz.new_zeros(len(scene.get_xyz), 3)
    background = torch.zeros_like(module.background)
    previous = module.renderer.gaussians
    try:
        module.renderer.gaussians = scene
        for start in range(0, len(rows), 3):
            pairs = rows[start:start+3, 1:]
            children = pairs.flatten()
            maps = []
            for offset in range(0, len(children), 3):
                batch = children[offset:offset+3]
                channels = torch.arange(len(batch), device=colors.device)
                colors.zero_()
                colors[old_count+batch, channels] = 1.
                image = module.renderer(view, background, override_color=colors)['render'][:len(batch)]
                if not bool(torch.isfinite(image).all()):
                    raise ValueError('Family native color-response render produced nonfinite weights')
                maps.append(image)
            weights = torch.cat(maps, 0)
            response[start:start+len(pairs)] = _contrast_response(
                weights[0::2], weights[1::2], active[pairs[:, 0]], active[pairs[:, 1]], pixel_active)
    finally:
        module.renderer.gaussians = previous
    return response


def _family_probe_view(module, view, contrast):
    """Get the actual paired-STE gradient without replacing any Parameter."""
    with torch.enable_grad():
        image = module.forward(view)['render']
        loss = (image.clamp(0, 1) - view.image.to(image)).square().mean()
        gradient = (torch.autograd.grad(loss, contrast, allow_unused=True)[0]
                    if loss.requires_grad else None)
    if gradient is None:
        raise ValueError('Family quantized forward must retain the family_contrast STE gradient')
    projection = gradient.detach().double() * (-.5 * image.numel())
    active = (image.detach() >= 0) & (image.detach() <= 1)
    return loss.detach().double(), projection, active, image.detach()


def _pair_increment_bounds(lower, upper):
    """Intersect +dz child bounds with the negated -dz child bounds."""
    if lower.shape != upper.shape or lower.ndim != 3 or lower.shape[1:] != (2, 3):
        raise ValueError('Family child DC bounds must have shape (families,2,3)')
    return torch.maximum(lower[:, 0], -upper[:, 1]), torch.minimum(upper[:, 0], -lower[:, 1])


@torch.no_grad()
def _family_source_envelope(module, views, scene, old_count, rows):
    """The frozen mother's inherited SH bounds each child at its current xyz."""
    births = module.births
    prior = getattr(births, 'family_anchor_raw', None)
    if not isinstance(prior, dict) or 'sh_dc' not in prior or 'sh_rest' not in prior:
        raise ValueError('Conservative families require fixed family_anchor_raw appearance')
    anchor = torch.cat((prior['sh_dc'], prior['sh_rest']), 1).to(scene.get_features)
    if len(anchor) != len(rows):
        raise ValueError('Fixed family anchor appearance must align with family_rows')
    children = rows[:, 1:].flatten()
    xyz = scene.get_xyz[old_count+children]
    current = scene.get_features[old_count+children]
    inherited = anchor.repeat_interleave(2, 0)
    colors = torch.stack([_preclamp_rgb(current, xyz, scene.active_sh_degree, view)
                          for view in views]).double().reshape(len(views), len(rows), 2, 3)
    prior_colors = torch.stack([_preclamp_rgb(inherited, xyz, scene.active_sh_degree, view)
                                for view in views]).double().reshape_as(colors)
    if not bool(torch.isfinite(colors).all() & torch.isfinite(prior_colors).all()):
        raise ValueError('Family source envelope received nonfinite preclamp SH colors')
    current_min, current_max = colors.amin(0), colors.amax(0)
    prior_low, prior_high = prior_colors.amin(0).clamp_max(0.), prior_colors.amax(0).clamp_min(1.)
    low, high = torch.minimum(prior_low, current_min), torch.maximum(prior_high, current_max)
    child_lower, child_upper = (low-current_min)/Y00, (high-current_max)/Y00
    lower, upper = _pair_increment_bounds(child_lower, child_upper)
    expanded = (low != prior_low) | (high != prior_high)
    report = dict(source='family_anchor_raw_fixed_mature_mother_dc_and_sh_rest',
        meaning='inherited preclamp SH range at current child xyz over all supplied training directions; includes [0,1]',
        zero_step='minimal hull with current quantized child colors keeps zero feasible',
        zero_step_expanded_child_rgb_channels=int(expanded.sum()),
        low_child_rgb=low.tolist(), high_child_rgb=high.tolist(),
        contrast_delta_lower=_summary(lower), contrast_delta_upper=_summary(upper),
        prior_rgb_min=_summary(prior_colors.amin(0)), prior_rgb_max=_summary(prior_colors.amax(0)))
    return lower, upper, report


def _decoded_contrast(value, step):
    return (value / step).round() * step


@torch.no_grad()
def _paired_box_candidate(initial, delta, scale, step, lower, upper):
    """Round one contrast symbol, keeping its decoded increment inside both boxes."""
    step = torch.as_tensor(step, device=initial.device, dtype=initial.dtype)
    if not bool(torch.isfinite(step).all() & (step > 0).all()):
        raise ValueError('Family DC quantization step must be finite and positive')
    symbols = (initial / step).round()
    before = symbols * step
    low_symbol, high_symbol = torch.ceil(lower/step.double()), torch.floor(upper/step.double())
    increment = (scale * delta.double() / step.double()).round().maximum(low_symbol).minimum(high_symbol).to(initial)
    candidate = torch.where(increment == 0, initial, (symbols+increment)*step)
    for _ in range(2):
        difference = (_decoded_contrast(candidate, step)-before).double()
        below, above = difference < lower, difference > upper
        invalid = below | above
        if not bool(invalid.any()):
            return candidate
        inward = increment - above.to(increment) + below.to(increment)
        inward = torch.where(increment > 0, inward.clamp_min(0), inward.clamp_max(0))
        increment = torch.where(invalid, inward, increment)
        target = torch.where(increment == 0, initial, (symbols+increment)*step)
        candidate = torch.where(invalid, target, candidate)
    difference = (_decoded_contrast(candidate, step)-before).double()
    return torch.where((difference < lower) | (difference > upper), initial, candidate)


@torch.no_grad()
def calibrate_family_contrast_dc(module, views):
    """Fit one frozen family RGB contrast per family using training cameras only.

    Geometry, mother/common DC, SH-rest and optimizer state are untouched. The
    original contrast Parameter stays installed; requires_grad and any existing
    .grad are preserved even when rendering or acceptance fails.
    """
    names, views = _view_list(views)
    births = module.births
    rows = getattr(births, 'family_rows', None)
    count = len(rows) if rows is not None else 0
    report = dict(method='conservative_family_contrast_dc_v7', family_count=count,
        views=names, view_count=len(views), status='skipped', reason=None,
        before_per_view_mse=[], after_per_view_mse=[], before_mean_mse=None, after_mean_mse=None,
        scene_max_abs_difference=0., coherence={}, support={}, source_envelope={}, contrast_update={},
        line_search=dict(accepted=False, accepted_scale=0., max_attempts=MAX_BACKTRACKS, trials=[]),
        stats_semantics=dict(
            view_scope='caller_supplied_standard_training_cameras_only',
            projection='sum(K*r), real family_contrast MSE gradient times -3*H*W/2',
            response='sum((J_plus-J_minus)^2), native alpha*T responses with actual SH/image clamp masks',
            response_approximation='exact within each scalar family/RGB contrast; cross-family Hessian omitted',
            coherence='weighted mean(d)^2 / weighted mean(d^2), H normalized over actual supporting views; zero denominator gives zero',
            support='K squared response > 1e-12 in at least two actual training views per RGB channel',
            box='intersection of plus-child source box and negated minus-child source box',
            quantization='one contrast integer symbol; plus/minus receive opposite decoded increments',
            conservation='mother DC and intrinsic child mean stay fixed; per-view rendered brightness can change',
            acceptance='strict decrease of mean full quantized training-camera MSE; no per-camera veto'))
    if count == 0 or len(views) < 2:
        report['reason'] = 'zero_families' if count == 0 else 'fewer_than_two_views'
        return report
    contrast = births.family_contrast
    if contrast.shape != (count, 3) or not isinstance(contrast, torch.nn.Parameter):
        raise ValueError('family_contrast must be the original leaf Parameter with shape (families,3)')
    rows = torch.as_tensor(rows, dtype=torch.long, device=contrast.device)
    if (rows.shape != (count, 3) or bool(((rows < 0) | (rows >= len(births._xyz))).any()) or
            len(rows.unique()) != rows.numel()):
        raise ValueError('Family mother/plus/minus rows must be valid and disjoint')
    if not getattr(module, 'motion_committed', True) or getattr(module, 'frame_finalized', False):
        raise ValueError('Family calibration requires an unfinished, motion-committed cohort')
    initial, flag = contrast.detach().clone(), contrast.requires_grad
    step = torch.as_tensor(module.config.birth.new_sh_step, device=contrast.device, dtype=contrast.dtype)
    decoded_before = _decoded_contrast(initial, step)
    committed = False
    try:
        with _isolated_renderer(module.renderer):
            scene, old_count = _response_scene(module)
            original_dc = scene.get_features[old_count:, :1].clone()
            contrast.requires_grad_(True)
            losses, projections, responses = [], [], []
            for view in views:
                loss, projection, pixel_active, prediction = _family_probe_view(module, view, contrast)
                module.renderer.gaussians = scene
                static = module.renderer(view, module.background)['render']
                difference = float((static-prediction).abs().max())
                report['scene_max_abs_difference'] = max(report['scene_max_abs_difference'], difference)
                if not torch.allclose(static, prediction, atol=2.e-6, rtol=2.e-5):
                    raise ValueError('Family response scene differs from actual full quantized forward')
                losses.append(loss)
                projections.append(projection)
                responses.append(_family_view_response(module, view, scene, old_count, rows, pixel_active))
            # Keep calibration-only color leaves frozen outside the probes.
            contrast.requires_grad_(flag)
            before = torch.stack(losses)
            if not bool(torch.isfinite(before).all()):
                raise ValueError('Family calibration received nonfinite training MSE')
            report.update(before_per_view_mse=before.tolist(), after_per_view_mse=before.tolist(),
                          before_mean_mse=float(before.mean()), after_mean_mse=float(before.mean()))
            delta, stats = _coherent_family_delta(torch.stack(projections), torch.stack(responses))
            supported = stats['eligible']
            report['coherence'] = _summary(stats['coherence'][supported])
            report['support'] = dict(responding_families=int(stats['valid'].any(0).any(-1).sum()),
                multiview_families=int(supported.any(-1).sum()),
                multiview_rgb_channels=int(supported.sum()),
                per_view_responding_families=stats['valid'].any(-1).sum(-1).tolist(),
                per_family_rgb_responding_views=stats['support'].tolist(),
                response=_summary(stats['response'][stats['valid']]))
            lower, upper, envelope = _family_source_envelope(module, views, scene, old_count, rows)
            bounded = delta.maximum(lower).minimum(upper)
            touched = bounded != delta
            report['source_envelope'] = envelope
            report['contrast_update'] = dict(unbounded_abs=_summary(delta.abs()),
                bounded_abs=_summary(bounded.abs()), bounded_families=int(touched.any(-1).sum()),
                bounded_rgb_channels=int(touched.sum()), committed_abs=_summary(torch.zeros_like(bounded)),
                quantized_changed_families=0, quantized_changed_components=0,
                mother_dc_max_abs_change=0., intrinsic_child_mean_max_abs_change=0.)
            if not bool(stats['valid'].any()) or not bool(bounded.any()):
                report['reason'] = ('zero_contrast_response' if not bool(stats['valid'].any()) else
                                    'no_coherent_multiview_contrast')
                return report
            for attempt in range(MAX_BACKTRACKS):
                scale = .5**attempt
                contrast.copy_(_paired_box_candidate(initial, bounded, scale, step, lower, upper))
                decoded = _decoded_contrast(contrast.detach(), step)
                changed = decoded != decoded_before
                trial = dict(scale=scale, quantized_changed_components=int(changed.sum()),
                             mean_mse=None, per_view_mse=[], accepted=False)
                report['line_search']['trials'].append(trial)
                if not bool(changed.any()):
                    report['reason'] = 'quantized_contrast_update_is_zero'
                    break
                current_dc = module.make_birth_view().raw_value('sh_dc').detach()
                mother_change = float((current_dc[rows[:, 0]]-original_dc[rows[:, 0]]).abs().max())
                mean_change = float(((current_dc[rows[:, 1]]+current_dc[rows[:, 2]])*.5 -
                                     (original_dc[rows[:, 1]]+original_dc[rows[:, 2]])*.5).abs().max())
                if mother_change != 0. or not torch.allclose(
                        (current_dc[rows[:, 1]]+current_dc[rows[:, 2]])*.5,
                        (original_dc[rows[:, 1]]+original_dc[rows[:, 2]])*.5, atol=2.e-6, rtol=2.e-5):
                    raise ValueError('Quantized family contrast changed mother DC or intrinsic child mean')
                after = _errors(module, views)
                finite = torch.isfinite(after)
                trial.update(mean_mse=float(after.mean()) if bool(finite.all()) else None,
                             per_view_mse=[float(value) if bool(valid) else None for value, valid in zip(after, finite)])
                if bool(finite.all()) and float(after.mean()) < float(before.mean()) - 1.e-10:
                    trial['accepted'] = committed = True
                    report.update(status='accepted', reason='mean_training_mse_improved',
                                  after_per_view_mse=after.tolist(), after_mean_mse=float(after.mean()))
                    report['line_search'].update(accepted=True, accepted_scale=scale)
                    report['contrast_update'].update(committed_abs=_summary((decoded-decoded_before).abs()),
                        quantized_changed_families=int(changed.any(-1).sum()),
                        quantized_changed_components=int(changed.sum()),
                        mother_dc_max_abs_change=mother_change, intrinsic_child_mean_max_abs_change=mean_change)
                    break
            if not committed and report['reason'] is None:
                report.update(status='rejected', reason='line_search_did_not_improve_mean_training_mse')
            return report
    finally:
        contrast.requires_grad_(flag)
        if not committed:
            contrast.copy_(initial)


calibrate_family_contrast = calibrate_family_contrast_dc
