"""Post-fit admission of current-frame births using disjoint training views.

The fixed candidate tree supplies groups; the tested scene is decoded from
the same packet codecs used for export. No heldout test image participates.
"""
import time
import sys
from pathlib import Path

import torch

from .model.birth_quantization import RAW_FIELDS, materialize_quantized_gaussians, model_raw
from .model.residual_birth import RawView
from .model.rigid_tree_birth_codec import (encode_birth_attributes, encode_birth_symbols,
                                           decode_birth_attributes)
from .model.rigid_tree_codec import encode_motion, decode_motion
from .model.spatial_gate import spatial_owner, spatial_state_route


def checked_training_pool(module, cached):
    names = list(cached)
    if (len(names) < 8 or len(set(names)) != len(names) or 'cam00.png' in names or
            any(Path(str(view.path)).name != name for name, view in cached.items())):
        raise ValueError('Post-fit admission requires at least eight distinct non-cam00 training cameras')
    requested = getattr(module.datamodule.config, 'training_camera_names', None)
    if requested is not None and (len(requested) != len(set(requested)) or set(requested) != set(names)):
        raise ValueError('Post-fit admission cannot use a cached pool wider than the configured training split')
    validation = getattr(module.datamodule.config, 'validation_camera_names', None)
    if validation is not None and set(validation).intersection(names):
        raise ValueError('Post-fit admission cannot use configured validation cameras')
    return names


def reserve_admission_cameras(module):
    cached = module.transaction_camera_views()
    names = checked_training_pool(module, cached)
    proposed = [Path(str(view.path)).name for view in module._plan_step_views]
    if len(proposed) != 4 or len(set(proposed)) != 4 or not set(proposed) <= set(names):
        raise ValueError('Post-fit admission needs the actual four step-200 proposal cameras')
    eligible = [name for name in names if name not in proposed]
    if len(eligible) < 4:
        raise ValueError('Post-fit admission needs four training cameras outside the first proposal')
    seed = int(module.datamodule.config.training_view_seed or 0)
    generator = torch.Generator(device='cpu').manual_seed(seed + 1000003 * int(module.current_frame) + 98765431)
    selected = [eligible[i] for i in torch.randperm(len(eligible), generator=generator)[:4].tolist()]
    module._plan_admission_camera_names = selected
    module.birth_control['regional_postfit_admission_cameras'] = dict(
        proposal_step_200=proposed, validation=selected,
        semantics='training cameras excluded from proposal 200 and all fitting/proposals 201-300')


@torch.no_grad()
def _prospective_decoded_scene(module):
    """Encode/decode all fitted candidates once before group decisions."""
    motion = module.motion
    old = model_raw(module.gaussians)
    scale = getattr(module, 'logscale_update', None)
    if scale is not None and scale.ready:
        old['logscale'] = scale.canonical_logscale(old['logscale'])
    view = module.make_birth_view()
    materialized, sources = materialize_quantized_gaussians(old, view)
    old_count = len(old['xyz'])
    rt, rr, gt, gr = [value.detach().cpu() for value in motion._values(hard=True)]
    cfg, birth = module.config.rigid_tree, module.config.birth
    packet = encode_motion(motion.frame_reference_state, motion.frame_index, motion.extent,
        motion.previous_rows.cpu(), motion.owner.cpu(), motion.parents.cpu(), rt, rr, gt, gr,
        sources.cpu(), materialized['xyz'][old_count:].cpu(), materialized['rotation'][old_count:].cpu(),
        cfg.motion_xyz_step_extent, cfg.motion_rotation_step_rad, compression='auto',
        gs_translation_step_extent=cfg.gs_xyz_step_extent,
        gs_rotation_step_rad=cfg.gs_rotation_step_rad, spatial_open=motion.opened.cpu(),
        spatial_relative=motion.relative_regions,
        inherit_birth_rotation=(not birth.rotation_updates and not cfg.rotation_updates),
        birth_pose_steps=(birth.new_xyz_step_extent * module.datamodule.cameras_extent,
                          birth.new_rotation_step))
    decoded = decode_motion(packet, motion.frame_reference_state)
    old = {name: value.detach().cpu().clone() for name, value in old.items()}
    old['xyz'], old['rotation'] = decoded.xyz[:old_count], decoded.rotation[:old_count]
    if len(sources):
        attrs = {name: materialized[name][old_count:] for name in ('logscale', 'opacity', 'sh_dc', 'sh_rest')}
        birth_packet = encode_birth_symbols(attrs, old_count,
            symbols={name: view.symbols[name] for name in attrs},
            steps={name: view.steps[name] for name in attrs},
            predictor={name: view.predictor[name] for name in attrs}, source_rows=sources)
        if birth_packet is None:
            birth_packet = encode_birth_attributes(attrs, old_count)
        newborn = decode_birth_attributes(birth_packet, old, decoded.xyz[old_count:],
                                           decoded.rotation[old_count:], decoded.birth_sources)
        raw = {name: torch.cat((old[name], newborn[name])) for name in RAW_FIELDS}
    else:
        raw = old
    return raw, decoded, sources.detach().cpu()


def leaf_groups(paths, opened, max_group_size=32):
    """Sorted fixed-tree leaf IDs, then original row order in chunks of 32."""
    if paths.ndim != 2 or max_group_size < 1:
        raise ValueError('Admission grouping needs candidate paths and a positive group size')
    owner = spatial_owner(paths, opened)
    leaves = paths[:, -1]
    groups = []
    for leaf in torch.unique(leaves, sorted=True).tolist():
        rows = leaves.eq(leaf).nonzero(as_tuple=True)[0]
        for start in range(0, len(rows), max_group_size):
            selected = rows[start:start + max_group_size]
            active = owner[selected].unique()
            if len(active) != 1:
                raise ValueError('A candidate leaf cannot have multiple active owners')
            groups.append((int(leaf), int(active[0]), selected))
    return groups


def _synchronize(module):
    if torch.device(module.device).type == 'cuda':
        torch.cuda.synchronize(module.device)


@torch.no_grad()
def _mse_per_view(module, raw, views):
    renderer = module.renderer
    previous = renderer.gaussians
    missing = object()
    collect = getattr(renderer, 'collect_contributions', missing)
    last = getattr(renderer, 'last_contributions', missing)
    try:
        if collect is not missing:
            renderer.collect_contributions = False
        renderer.gaussians = RawView(raw, module.gaussians.active_sh_degree, module.gaussians)
        values = []
        for view in views:
            rendered = renderer(view, module.background)['render'].clamp(0, 1)
            error = (rendered - view.image).double().square().mean()
            if not bool(torch.isfinite(error)):
                raise ValueError('Post-fit admission produced a nonfinite validation MSE')
            values.append(float(error))
        return values
    finally:
        renderer.gaussians = previous
        if collect is not missing:
            renderer.collect_contributions = collect
        if last is missing:
            if hasattr(renderer, 'last_contributions'):
                delattr(renderer, 'last_contributions')
        else:
            renderer.last_contributions = last


def _tolerance(before, after):
    return max(1e-12, 8 * torch.finfo(torch.float32).eps * max(abs(before), abs(after)))


@torch.no_grad()
def admit_fitted_births(module):
    from .trainer import Trainer
    rng = Trainer._capture_rng(module.trainer)
    _synchronize(module)
    started = time.perf_counter()
    try:
        cached = module.transaction_camera_views()
        checked_training_pool(module, cached)
        names = getattr(module, '_plan_admission_camera_names', None)
        if (not names or len(names) != 4 or len(set(names)) != 4 or
                any(name not in cached or name == 'cam00.png' for name in names)):
            raise ValueError('Post-fit admission lost its four reserved training views')
        views = [cached[name] for name in names]
        raw_cpu, decoded, sources = _prospective_decoded_scene(module)
        old_count = len(module.gaussians._xyz)
        count = len(sources)
        if count != len(module.births._xyz):
            raise ValueError('Post-fit admission candidate count disagrees with decoded packet')
        paths = spatial_state_route(decoded.xyz[old_count:], decoded)
        groups = leaf_groups(paths, decoded.spatial_open.cpu())
        if sum(len(rows) for _, _, rows in groups) != count:
            raise ValueError('Post-fit admission did not cover every fitted candidate')
        raw = {name: value.to(module.device) for name, value in raw_cpu.items()}
        accepted = torch.zeros(count, dtype=torch.bool)
        render_seconds = 0.

        def render_selection(mask):
            nonlocal render_seconds
            indices = mask.nonzero(as_tuple=True)[0].to(module.device)
            scene = {name: torch.cat((value[:old_count], value[old_count:][indices]))
                     for name, value in raw.items()}
            _synchronize(module)
            render_started = time.perf_counter()
            errors = _mse_per_view(module, scene, views)
            _synchronize(module)
            render_seconds += time.perf_counter() - render_started
            return errors

        baseline = render_selection(accepted)
        initial = list(baseline)
        group_reports = []
        for leaf, owner, rows in groups:
            trial = accepted.clone()
            trial[rows] = True
            errors = render_selection(trial)
            before_mean, after_mean = sum(baseline) / 4, sum(errors) / 4
            tolerance = _tolerance(before_mean, after_mean)
            take = before_mean - after_mean > tolerance
            group_reports.append(dict(leaf=leaf, active_owner=owner, rows=rows.tolist(),
                candidate_count=len(rows), accepted=take, baseline_mse=before_mean,
                trial_mse=after_mean, baseline_per_view=baseline, trial_per_view=errors,
                improvement=before_mean-after_mean, numerical_tolerance=tolerance))
            if take:
                accepted, baseline = trial, errors
        selected = accepted.nonzero(as_tuple=True)[0]
        module.births.select_rows_inplace(selected.to(module.device), module.motion.optimizer)
        expected = {name: torch.cat((value[:old_count], value[old_count:][selected]))
                    for name, value in raw_cpu.items()}
        if not torch.equal(module.births.source_rows.cpu(), sources[selected]):
            raise ValueError('Post-fit admission changed an old-GS codec reference')
        _synchronize(module)
        report = dict(cameras=list(names), old_count=old_count, candidates=count,
            leaf_count=len(set(leaf for leaf, _, _ in groups)), groups=group_reports,
            group_count=len(groups), accepted_groups=sum(row['accepted'] for row in group_reports),
            accepted_count=len(selected), rejected_count=count-len(selected),
            old_only_per_view_mse=initial, old_only_mean_mse=sum(initial)/4,
            accepted_per_view_mse=baseline, accepted_mean_mse=sum(baseline)/4,
            selection_render_seconds=render_seconds,
            extra_seconds=time.perf_counter()-started,
            semantics='decoded candidate attributes; each complete clamped render is compared '
                      'by mean MSE over four disjoint training views, in fixed-leaf order')
        module.birth_control['regional_postfit_admission'] = report
        module._plan_admission_verification = (expected, sources[selected].clone(), views, list(baseline))
    finally:
        Trainer._restore_rng(rng)


@torch.no_grad()
def verify_encoded_admission(module):
    from .trainer import Trainer
    expected, expected_sources, views, baseline = module._plan_admission_verification
    rng = Trainer._capture_rng(module.trainer)
    _synchronize(module)
    started = time.perf_counter()
    try:
        # Re-read the published final packets using the independent decoder.
        # Mid-frame checkpoints persist the preceding raw reference; replay
        # provides a safe fallback for older compatible checkpoints.
        path = list(sys.path)
        try:
            from scripts.replay_rigid_tree import decode_frame, replay
        finally:
            sys.path[:] = path
        directory = module.output_dir / 'frame_storage' / 'rigid_tree'
        prior = module.motion.frame_reference_state
        previous_raw = getattr(module, '_plan_diagnostic_reference', None)
        if previous_raw is None:
            prior_decoded, previous_raw = replay(directory, module.trainer.decoder_reference_path,
                                                 module.current_frame - 1)
            if prior_decoded.digest() != prior.digest():
                raise ValueError('Post-fit admission replay recovered a different previous frame')
        decoded, wire = decode_frame(directory, module.current_frame, prior, previous_raw)
        if decoded.digest() != module.motion.codec_previous_state.digest():
            raise ValueError('Post-fit admission decoded motion differs from the published state')
        actual = model_raw(module.gaussians)
        errors = {}
        for name in RAW_FIELDS:
            value = wire[name].detach().cpu()
            wanted = expected[name]
            if (value.dtype != wanted.dtype or value.shape != wanted.shape or
                    not torch.equal(value.contiguous().view(torch.uint8),
                                    wanted.contiguous().view(torch.uint8))):
                raise ValueError(f'Encoded post-fit admission changed validated {name} attributes')
            model_value = actual[name].detach().cpu()
            if (model_value.dtype != value.dtype or model_value.shape != value.shape or
                    not torch.equal(model_value.contiguous().view(torch.uint8),
                                    value.contiguous().view(torch.uint8))):
                raise ValueError(f'Materialized post-fit admission differs from decoded {name} attributes')
            errors[name] = True
        sources = decoded.birth_sources
        if sources is None:
            sources = torch.empty(0, dtype=torch.long)
        if not torch.equal(sources.detach().cpu().long(), expected_sources):
            raise ValueError('Encoded post-fit admission changed the validated old-GS source rows')
        _synchronize(module)
        render_started = time.perf_counter()
        decoded_scene = {name: value.to(module.device) for name, value in wire.items()}
        mse = _mse_per_view(module, decoded_scene, views)
        _synchronize(module)
        render_seconds = time.perf_counter() - render_started
        for before, after in zip(baseline, mse):
            if abs(before-after) > _tolerance(before, after):
                raise ValueError('Encoded post-fit admission changed its validated scene MSE')
        _synchronize(module)
        module.birth_control['regional_postfit_admission']['codec_verification'] = dict(
            independent_stream_decoded=True, model_matches_decoded=True,
            six_attributes_exact=errors, source_rows_exact=True, per_view_mse=mse,
            mean_mse=sum(mse)/4, render_seconds=render_seconds,
            extra_seconds=time.perf_counter()-started)
    finally:
        module._plan_admission_verification = None
        Trainer._restore_rng(rng)
