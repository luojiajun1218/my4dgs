"""诊断动态帧训练中 gate、区域参数和逐点属性的梯度。

负责选择动态帧、步骤、训练相机和参数，在优化器更新前记录图像目标、运动存储
及相对运动损失的分项梯度，并提供 gate 分层统计；分项不等同于完整训练目标。
本文件是临时诊断入口，复用正式训练流程，不负责定义或替换逐帧训练方案。
示例：python scripts/diagnose_training_gradients.py --config=config/best_coffee_martini.yaml \
    --probe_frames=1,10 --probe_steps=50,151 --probe_cameras=scheduled \
    --probe_targets=gate,regional_motion,point_motion \
    output_dir=tmp experiment_name=gate_probe data.extra_dataset_kwargs.num_frames=11
结果写入该次实验目录的 training_gradient_diagnostics，不修改历史实验。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from absl import flags

from pipeline.loss import (motion_image_factor, motion_photometric_loss,
                           mixed_image_loss)
from pipeline.per_frame_training import fitting_old_motion, regional_attribute_repair_active
from pipeline.module import uses_early_replacement
from pipeline.trainer import Trainer
import main as project_main

flags.DEFINE_list('probe_frames', ['1'], 'Dynamic frame indices, comma-separated.')
flags.DEFINE_list('probe_steps', ['151'], 'Training steps, comma-separated, in 1..200.')
flags.DEFINE_list('probe_cameras', ['scheduled'], 'scheduled, all, or training camera filenames.')
flags.DEFINE_list('probe_targets', ['all'],
                  'all, or groups gate,regional_motion,point_motion,regional_attributes,point_attributes.')
flags.DEFINE_bool('probe_stop_after', True, 'Stop after the final requested frame and step.')


class ProbeFinished(Exception):
    pass


def _indices(values, name, lower, upper=None):
    try:
        result = sorted({int(value) for value in values})
    except ValueError as error:
        raise ValueError(f'{name} must contain comma-separated integers') from error
    if not result or any(value < lower or (upper is not None and value > upper) for value in result):
        raise ValueError(f'{name} must be in {lower}..{upper or "infinity"}')
    return result


def _parameters(module, targets):
    motion = module.motion
    regional = getattr(module, 'regional_full_attributes', None)
    point = getattr(module, 'full_attribute_update', None)
    opacity = getattr(module, 'opacity_update', None)
    groups = {
        'gate': {'gate': motion.gate_logits},
        'regional_motion': {'regional.translation': motion.region_translation,
                            'regional.rotation': motion.region_rotation},
        'point_motion': {'point.translation': motion.gs_translation,
                         'point.rotation': motion.gs_rotation},
        'regional_attributes': {f'regional.{name}': value for name, value in
                                getattr(regional, 'parameters', {}).items()},
        'point_attributes': {f'point.{name}': value for name, value in
                             getattr(point, 'parameters', {}).items()},
    }
    if opacity is not None and opacity.delta is not None:
        groups['point_attributes']['point.opacity'] = opacity.delta
    if targets == ['all']:
        targets = list(groups)
    unknown = set(targets) - groups.keys()
    if unknown or not targets:
        raise ValueError(f'Invalid probe_targets {targets}; choose all or {list(groups)}')
    return ({name: value for group in dict.fromkeys(targets)
             for name, value in groups[group].items()},
            [group for group in dict.fromkeys(targets) if not groups[group]])


def _gradients(value, parameters, *, retain_graph=False):
    result = {name: None for name in parameters}
    active = [(name, parameter) for name, parameter in parameters.items() if parameter.requires_grad]
    if not active or not torch.is_tensor(value) or not value.requires_grad:
        return result
    gradients = torch.autograd.grad(value, [parameter for _, parameter in active],
                                    allow_unused=True, retain_graph=retain_graph)
    result.update((name, gradient) for (name, _), gradient in zip(active, gradients))
    return result


def _parameter_stats(parameter, gradient):
    result = dict(shape=list(parameter.shape), requires_grad=bool(parameter.requires_grad),
                  gradient_available=gradient is not None)
    if gradient is None:
        return result
    value = gradient.detach()
    result.update(norm=float(value.norm()), nonzero=int(value.ne(0).sum()),
                  finite=bool(torch.isfinite(value).all()),
                  mean_abs=float(value.abs().mean()) if value.numel() else None,
                  max_abs=float(value.abs().max()) if value.numel() else None)
    return result


def _all_stats(parameters, gradients):
    return {name: _parameter_stats(value, gradients[name]) for name, value in parameters.items()}


def _stats(gradient, mask, logits):
    count = int(mask.sum())
    if gradient is None:
        return dict(count=count, gradient_available=False, norm=None, nonzero=None,
                    towards_switch=None)
    values = gradient.detach()[mask]
    return dict(count=count, gradient_available=True, norm=float(values.norm()),
                nonzero=int(values.ne(0).sum()),
                towards_switch=int((values * logits.detach()[mask] > 0).sum()))


def _cosine(a, b, mask=None):
    if a is None or b is None:
        return None
    a, b = a.detach(), b.detach()
    if mask is not None:
        a, b = a[mask], b[mask]
    denominator = a.norm() * b.norm()
    return None if float(denominator) == 0 else float((a * b).sum() / denominator)


def _gate_masks(motion):
    paths, owner = motion.spatial_paths, motion.owner
    matches = paths.eq(owner[:, None])
    if not bool(matches.sum(1).eq(1).all()):
        raise ValueError('Each GS must have exactly one owner on its spatial path')
    depth = matches.long().argmax(1)
    reaches = torch.arange(paths.shape[1], device=paths.device)[None, :] <= depth[:, None]
    eligible = motion.split_eligible
    reachable = torch.zeros_like(eligible)
    reachable[paths[reaches]] = True
    masks = dict(eligible=eligible, reachable=reachable & eligible,
                 unreachable=eligible & ~reachable, invalid=~eligible)
    for level in range(paths.shape[1]):
        nodes = torch.zeros_like(eligible)
        nodes[paths[:, level]] = True
        masks[f'depth_{level + 1}'] = nodes & eligible
    return masks


def _capture_sampling(trainer, motion):
    generator = getattr(motion, '_gate_rng', None)
    noise = getattr(motion, '_sampled_gate_noise', None)
    return (trainer._capture_rng(), None if generator is None else generator.get_state(),
            None if noise is None else noise.clone(), getattr(motion, '_sampled_gate_step', None))


def _restore_sampling(trainer, motion, state):
    rng, gate_rng, noise, sampled_step = state
    trainer._restore_rng(rng)
    if gate_rng is not None:
        motion._gate_rng.set_state(gate_rng)
    if hasattr(motion, '_sampled_gate_noise'):
        motion._sampled_gate_noise = noise
        motion._sampled_gate_step = sampled_step


def _image_probe(module, view, step, parameters, factor, metric):
    image = module.forward(view, training=True)['render']
    photo = motion_photometric_loss(image, view.image, module.config.lambda_dssim,
                                    factor, metric=metric)
    config = module.config.full_update
    mix = config.repair_rmse_mix if step > module.motion_steps else config.motion_rmse_mix
    active = (regional_attribute_repair_active(module) or
              getattr(module.config, 'ablation_motion_only', False)) and mix > 0
    if active:
        weights = module._temporal_repair_weights(view) if config.repair_temporal_weight else None
        objective, _, _ = mixed_image_loss(image, view.image, photo, mix, weights)
        photo_gradient = _gradients(photo, parameters, retain_graph=True)
        gradient = _gradients(objective, parameters)
    else:
        objective = photo
        gradient = photo_gradient = _gradients(photo, parameters)
    return (image.detach(), float(photo.detach()), float(objective.detach()),
            photo_gradient, gradient)


def _views(module, step, cameras):
    views = module.transaction_camera_views()
    if cameras == ['scheduled']:
        # The normal training call records this view later; avoid a duplicate audit entry.
        trace = module.birth_control.get('point_schedule_camera_trace')
        previous_length = len(trace) if trace is not None else 0
        view = module.per_frame_training.frame_plan_training_views(step, 1)[0]
        trace = module.birth_control.get('point_schedule_camera_trace')
        if trace is not None:
            del trace[previous_length:]
        return [(Path(str(view.path)).name, view)]
    if cameras == ['all']:
        return ((name, views[name]) for name in views)
    if 'scheduled' in cameras or 'all' in cameras:
        raise ValueError('Use scheduled or all alone, or provide training camera filenames')
    missing = [name for name in cameras if name not in views]
    if missing:
        raise ValueError(f'Unknown training camera(s): {missing}; available: {list(views)}')
    return ((name, views[name]) for name in dict.fromkeys(cameras))


def _diagnose(trainer, step, cameras, targets):
    module, motion = trainer.module, trainer.module.motion
    gate = motion.gate_logits
    parameters, unavailable_groups = _parameters(module, targets)
    masks = _gate_masks(motion) if 'gate' in parameters else None
    factor = motion_image_factor(step, module.motion_steps,
        enabled=module.config.motion_coarse_to_fine, fitting_motion=fitting_old_motion(module),
        bootstrap_steps=module.config.rigid_tree.spatial_gate.start_step)
    metric = module.config.photometric_loss
    if (module.motion_committed or (uses_early_replacement(module.config) and module.early_initialized)) and module.config.birth.photometric_loss is not None:
        metric = module.config.birth.photometric_loss
    original_ste = bool(getattr(motion, 'refine_image_gate_ste', False))
    state = _capture_sampling(trainer, motion)
    try:
        _restore_sampling(trainer, motion, state)
        if motion._candidate_initialized and not motion.amplitude_refit_active:
            terms = motion.loss_terms(step, diagnostics=False)
            rate = _gradients(terms['rate_loss'], parameters, retain_graph=True)
            relative = _gradients(terms['relative_motion_loss'], parameters)
            del terms
        else:
            rate = relative = {name: None for name in parameters}
        results = []
        for name, view in _views(module, step, cameras):
            _restore_sampling(trainer, motion, state)
            image, photo, objective, photo_gradient, gradient = _image_probe(
                module, view, step, parameters, factor, metric)
            row = dict(camera=name, photometric_loss=photo, image_objective=objective,
                       image_objective_gradient=_all_stats(parameters, gradient),
                       photometric_gradient=_all_stats(parameters, photo_gradient),
                       image_rate_cosine={name: _cosine(gradient[name], rate[name])
                                          for name in parameters})
            if masks is not None:
                row['gate_detail'] = dict(
                    image_gradient={key: _stats(gradient['gate'], mask, gate) for key, mask in masks.items()},
                    photometric_gradient={key: _stats(photo_gradient['gate'], mask, gate) for key, mask in masks.items()},
                    image_rate_cosine_reachable=_cosine(gradient['gate'], rate['gate'], masks['reachable']))
            # This compares the configured path with the same path with the optional
            # refine STE disabled; it does not force an unsupported STE configuration.
            if original_ste:
                _restore_sampling(trainer, motion, state)
                motion.refine_image_gate_ste = False
                baseline, baseline_photo, baseline_objective, _, baseline_gradient = _image_probe(
                    module, view, step, parameters, factor, metric)
                motion.refine_image_gate_ste = original_ste
                difference = (image - baseline).abs()
                row['without_refine_ste'] = dict(photometric_loss=baseline_photo,
                    image_objective=baseline_objective,
                    image_objective_gradient=_all_stats(parameters, baseline_gradient),
                    forward_equal=bool(torch.equal(image, baseline)),
                    forward_max_abs=float(difference.max()))
            results.append(row)
        config = module.config.full_update
        report = dict(frame=int(trainer.current_frame), step=int(step),
            phase=module.per_frame_training.per_frame_phase(step),
            targets=targets, unavailable_groups=unavailable_groups,
            gate_requires_grad=bool(gate.requires_grad), hard_gate_frozen=bool(motion._hard_gate_frozen),
            refine_image_gate_ste=original_ste, image_relaxation=motion.config.spatial_gate.image_relaxation,
            photometric_objective=dict(metric=metric, lambda_dssim=module.config.lambda_dssim, factor=factor,
                note='Photometric component; image_objective also includes any configured RMSE mix.'),
            motion_rmse_mix=float(config.motion_rmse_mix),
            motion_regularization={name: _all_stats(parameters, gradient)
                                   for name, gradient in (('rate_loss', rate), ('relative_motion_loss', relative))},
            note='Gradient towards_switch counts eligible logits where gradient descent moves toward zero. '
                 'These separate gradients exclude attribute-rate and other total-objective terms.',
            cameras=results)
        if masks is not None:
            report['gate_motion_regularization'] = {
                name: {key: _stats(gradient['gate'], mask, gate) for key, mask in masks.items()}
                for name, gradient in (('rate_loss', rate), ('relative_motion_loss', relative))}
        directory = trainer.output_dir / 'training_gradient_diagnostics'
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'frame_{trainer.current_frame:03d}_step_{step:03d}.json'
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'Training gradient diagnostic: {destination}', flush=True)
    finally:
        motion.refine_image_gate_ste = original_ste
        _restore_sampling(trainer, motion, state)


_original_fit = Trainer._fit_views
_original_resolve_config = project_main.resolve_config
_seen = set()


def _resolve_probe_config(argv, config_type):
    config, extra = _original_resolve_config(argv, config_type)
    destination = Path(config.output_dir) / config.experiment_name
    if destination.exists():
        raise ValueError(f'Probe output already exists: {destination}. '
                         'Choose a new output_dir and experiment_name.')
    return config, extra


def _probe_fit(self, fetcher, step, *, defer_metrics=False):
    frames = _indices(flags.FLAGS.probe_frames, 'probe_frames', 1)
    steps = _indices(flags.FLAGS.probe_steps, 'probe_steps', 1, 200)
    frame = int(self.current_frame)
    if frame in frames and step in steps:
        _diagnose(self, step, list(flags.FLAGS.probe_cameras), list(flags.FLAGS.probe_targets))
        _seen.add((frame, step))
        if flags.FLAGS.probe_stop_after and frame == frames[-1] and step == steps[-1]:
            raise ProbeFinished()
    return _original_fit(self, fetcher, step, defer_metrics=defer_metrics)


Trainer._fit_views = _probe_fit
project_main.resolve_config = _resolve_probe_config


if __name__ == '__main__':
    try:
        project_main.app.run(project_main.main)
    except ProbeFinished:
        print('Requested final probe completed; training stopped before its optimizer update.', flush=True)
    finally:
        if flags.FLAGS.is_parsed():
            requested = {(frame, step) for frame in _indices(flags.FLAGS.probe_frames, 'probe_frames', 1)
                         for step in _indices(flags.FLAGS.probe_steps, 'probe_steps', 1, 200)}
            missing = requested - _seen
            if missing:
                print(f'Requested probes not reached (check frame count or resume position): {sorted(missing)}', flush=True)
