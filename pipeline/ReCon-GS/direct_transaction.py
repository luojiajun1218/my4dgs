"""Direct topology decision before a single normal newborn-fitting phase."""

import json
import time

import torch
from torch.nn.functional import l1_loss
from libgs.metric import ssim

from .model.common_contrast import CommonContrastMotion
from .model.direct_tree import apply_layout_edits, propose_local_batch, prune_inactive_split_edits
from .model.residual_motion import model_raw, reference_digest
from .model.split_region_scores import average_position_gradients


@torch.enable_grad()
def collect_split_position_gradients(module):
    """Image-only world-position derivatives without optimizer/GS mutation."""
    motion, cfg = module.residual_motion, module.residual_motion.config
    names = list(cfg.direct_split_camera_names)
    dataset = module.datamodule.get_train_dataset()
    indices = module.datamodule._camera_indices(dataset, names)
    configured = getattr(module.datamodule.config, 'training_camera_names', None)
    validation = set(getattr(module.datamodule.config, 'validation_camera_names', None) or [])
    allowed = set(configured) if configured is not None else {item.path.name for item in dataset.items}
    if not set(names) <= allowed or set(names) & validation or 'cam00.png' in names:
        raise ValueError('Split scoring cameras must belong to the optimization training split')
    with torch.no_grad():
        xyz, rotation = motion(motion.last_step)
    xyz = xyz.detach()
    rotation = rotation.detach()
    original_model = module.renderer.gaussians
    original_xyz, original_rotation = module.gaussians.rigid_xyz, module.gaussians.rigid_rotation
    gradients, visible, observed = [], [], []
    try:
        module.renderer.gaussians = module.gaussians
        module.gaussians.rigid_rotation = rotation
        # The loader must not advance optimization's random streams.
        with torch.random.fork_rng(devices=[]):
            for view in module.datamodule.train_dataloader(random=False, indices=indices):
                view = view.to(module.device)
                observed.append(view.path.name)
                leaf_xyz = xyz.clone().requires_grad_(True)
                module.gaussians.rigid_xyz = leaf_xyz
                pkg = module.renderer(view, module.background)
                weight = module.config.lambda_dssim
                loss = ((1 - weight) * l1_loss(pkg['render'], view.image) +
                        weight * (1 - ssim(pkg['render'], view.image)))
                gradient, = torch.autograd.grad(loss, leaf_xyz)
                gradients.append(gradient.detach())
                visible.append(pkg['visibility_filter'].detach())
    finally:
        module.renderer.gaussians = original_model
        module.gaussians.rigid_xyz, module.gaussians.rigid_rotation = original_xyz, original_rotation
    if observed != names:
        raise ValueError('Scoring loader did not return exactly the requested training cameras')
    evidence = average_position_gradients(torch.stack(gradients), torch.stack(visible),
                                         min_visible_views=cfg.direct_split_min_visible_views)
    return xyz, evidence, dict(cameras=observed, supported_gs=int(evidence.supported.sum()),
                              semantics='One fixed rendered GS state; visible finite world XYZ '
                              'image gradients only. No storage term or optimizer step.')


class DirectTransaction:
    def __init__(self, module, control, proposal):
        self.control = control
        self.report = proposal
        self.report.update(extra_control_steps=0, extra_control_seconds=0.,
                           training_cameras=[], acceptance_seconds=0.)
        self.view_names = []
        self.control_stats = dict(accum=module.gaussians.xyz_gradient_accum.clone(),
                                  denom=module.gaussians.denom.clone())
        self.control_stats_window_initialized = module.quant_stats_window_initialized

    def train_control(self, module, viewpoint):
        started = time.perf_counter()
        active = module.residual_motion
        module.residual_motion = self.control
        try:
            with torch.enable_grad():
                pkg = module.forward_quantized_residual(viewpoint, training=True)
                image = pkg['render']
                weight = module.config.lambda_dssim
                loss = (1 - weight) * l1_loss(image, viewpoint.image) + weight * (1 - ssim(image, viewpoint.image))
                cfg = self.control.config
                storage, _ = module.quantized_storage()
                rate_weight = cfg.storage_loss_weight * self.control.storage_scale(module.global_step)
                loss = loss + rate_weight * storage / self.control.reference_storage_bytes(module.quant_first_frame_gs_count)
                loss.backward()
                if cfg.enable_incremental and module.global_step >= cfg.densification_stats_start:
                    if not self.control_stats_window_initialized:
                        if cfg.densification_stats_steps is not None:
                            self.control_stats['accum'].zero_()
                            self.control_stats['denom'].zero_()
                        self.control_stats_window_initialized = True
                    visible = pkg['visibility_filter']
                    grad = pkg['viewspace_points'].grad
                    self.control_stats['accum'][visible] += grad[visible, :2].norm(dim=-1, keepdim=True)
                    self.control_stats['denom'][visible] += 1
                self.control.optimizer_step()
        finally:
            module.residual_motion = active
        module.trainer.synchronize()
        self.report['extra_control_seconds'] += time.perf_counter() - started
        self.report['extra_control_steps'] += 1
        self.view_names.append(viewpoint.path.name)

    @torch.no_grad()
    def finish(self, module):
        started = time.perf_counter()
        candidate = module.residual_motion
        candidate, pruning = prune_inactive_split_edits(candidate)
        module.residual_motion = candidate
        self.report['split_pruning'] = pruning
        directory = module.output_dir / 'direct_tree_trials' / f'frame_{module.current_frame:03d}'
        directory.mkdir(parents=True, exist_ok=False)
        raw = model_raw(module.gaussians)
        io_seconds = 0.
        if candidate.config.direct_save_trial_artifacts:
            io_started = time.perf_counter()
            torch.save(dict(control=self.control.capture(), candidate=candidate.capture(),
                            raw=raw, report=self.report, frame=module.current_frame,
                            control_stats=self.control_stats, view_names=self.view_names),
                       directory / 'fitted_motion_snapshot.pt')
            module.trainer.synchronize()
            io_seconds += time.perf_counter() - io_started
        branches = {}
        codec_started = time.perf_counter()
        codec_io_seconds = 0.
        for name, motion in (('control', self.control), ('candidate', candidate)):
            xyz, rotation = motion(motion.config.motion_steps)
            moved = dict(raw, xyz=xyz.detach(), rotation=rotation.detach())
            packet, export = motion.encode_motion(frame_index=module.current_frame,
                previous_state_digest=module.quant_previous_sha256,
                post_motion_digest=reference_digest(moved), previous_count=len(module.quant_previous_reference['xyz']),
                deleted_count=int(module.quant_deleted_indices.numel()), birth_count=0)
            header = motion.read_header(packet)
            layout = apply_layout_edits(motion._frame_coding_layout, header['structure_edits'], motion.base_xyz)
            dx, dq, *_ = motion.decode_motion(packet, motion.base_xyz, motion.base_rotation, motion.config, layout)
            if reference_digest(dict(raw, xyz=dx, rotation=dq)) != header['post_motion_digest']:
                raise AssertionError('Trial encoded/decoded motion hash mismatch')
            if candidate.config.direct_save_trial_artifacts:
                io_started = time.perf_counter()
                path = directory / f'{name}.bin'
                path.write_bytes(packet)
                export['path'] = str(path)
                duration = time.perf_counter() - io_started
                io_seconds += duration
                codec_io_seconds += duration
            branches[name] = dict(motion=motion, xyz=dx, rotation=dq, export=export, objectives=[])
        module.trainer.synchronize()
        codec_seconds = max(0., time.perf_counter() - codec_started - codec_io_seconds)
        # Acceptance uses all available cam01--20 training cameras, no holdout.
        cameras = []
        original_xyz, original_q = module.gaussians.rigid_xyz, module.gaussians.rigid_rotation
        try:
            for view in module.datamodule.train_dataloader(random=False):
                view = view.to(module.device)
                if view.path.name not in {f'cam{i:02d}.png' for i in range(1, 21)}:
                    raise ValueError('Acceptance camera is outside cam01--20')
                cameras.append(view.path.name)
                for branch in branches.values():
                    module.gaussians.rigid_xyz, module.gaussians.rigid_rotation = branch['xyz'], branch['rotation']
                    image = module.renderer(view, module.background)['render']
                    weight = module.config.lambda_dssim
                    distortion = (1 - weight) * l1_loss(image, view.image) + weight * (1 - ssim(image, view.image))
                    branch['objectives'].append(float(distortion))
        finally:
            module.gaussians.rigid_xyz, module.gaussians.rigid_rotation = original_xyz, original_q
        if set(cameras) != {f'cam{i:02d}.png' for i in range(1, 21)}:
            raise ValueError('Direct transaction requires exactly twenty training cameras')
        reference_bytes = candidate.reference_storage_bytes(module.quant_first_frame_gs_count)
        weight = candidate.config.storage_loss_weight
        values = {}
        for name, branch in branches.items():
            distortion = sum(branch['objectives']) / len(branch['objectives'])
            size = branch['export']['archive_bytes']
            values[name] = dict(distortion=distortion, bytes=size,
                               structure_edit_bytes=branch['export']['structure_edit_bytes'],
                               objective=distortion + weight * size / reference_bytes)
        accepted = values['candidate']['objective'] < values['control']['objective']
        module.residual_motion = candidate if accepted else self.control
        if not accepted and candidate.config.enable_incremental:
            module.gaussians.xyz_gradient_accum.copy_(self.control_stats['accum'])
            module.gaussians.denom.copy_(self.control_stats['denom'])
        module.trainer.synchronize()
        self.report.update(accepted=accepted, branches=values, training_cameras=cameras,
                           continuation_view_order=self.view_names,
                           reference_bytes=reference_bytes, rate_weight=weight,
                           trial_artifacts_saved=candidate.config.direct_save_trial_artifacts,
                           trial_artifact_write_seconds=io_seconds,
                           trial_codec_seconds=codec_seconds,
                           acceptance_seconds=time.perf_counter() - started,
                           semantics='One entire decoded motion batch; equal remaining motion/gate budgets '
                           'per branch after the proposal. Conditional motion comparison before newborn fitting. '
                           'Newborn fitting runs once on the selected branch; future newborn bytes are '
                           'reported in final-frame accounting. Control improvements '
                           'survive rollback. Adam moments reset equally at proposal boundary. No cam00 selection.')
        (directory / 'report.json').write_text(json.dumps(self.report, indent=2))
        module.residual_motion.stage_seconds['direct_control'] = self.report['extra_control_seconds']
        module.residual_motion.stage_seconds['direct_acceptance'] = self.report['acceptance_seconds']
        module.residual_motion.stage_seconds['direct_proposal'] = self.report['proposal_seconds']
        module.residual_motion.stage_seconds['direct_trial_artifact_write'] = io_seconds
        module.residual_motion.stage_seconds['direct_trial_codec'] = codec_seconds


@torch.no_grad()
def start_transaction(module, render_pkg):
    motion, cfg = module.residual_motion, module.residual_motion.config
    if not cfg.direct_tree_enabled:
        return
    if module.current_frame % cfg.direct_update_interval:
        return
    if (not cfg.direct_reset_on_noop and cfg.direct_candidate_windows == 0
            and not cfg.direct_gradient_split_enabled):
        return
    if (not cfg.direct_preserve_optimizer_moments and cfg.direct_proposal_step is not None
            and cfg.direct_proposal_step < cfg.dense_fit_steps + cfg.gate_compress_steps):
        # Apply the same boundary reset even in fixed-tree control runs.
        motion._make_optimizers()
        motion.current_phase = None
    if cfg.direct_candidate_windows == 0 and not cfg.direct_gradient_split_enabled:
        # Normal gate_refine resets its Adam moments on the next step too.
        return
    module.trainer.synchronize()
    started = time.perf_counter()
    screen_grad = render_pkg['viewspace_points'].grad
    saliency = screen_grad[:, :2].detach().norm(dim=-1) if screen_grad is not None else None
    split_kwargs, scoring_report = {}, None
    if cfg.direct_gradient_split_enabled:
        xyz, evidence, scoring_report = collect_split_position_gradients(module)
        split_kwargs = dict(split_xyz=xyz, split_evidence=evidence)
    candidate, report = propose_local_batch(motion, saliency, **split_kwargs)
    if scoring_report is not None:
        report['split_gradient_probe'] = scoring_report
    module.trainer.synchronize()
    report['proposal_seconds'] = time.perf_counter() - started
    report['proposal_step'] = module.global_step
    if candidate is None:
        directory = module.output_dir / 'direct_tree_trials' / f'frame_{module.current_frame:03d}'
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'report.json').write_text(json.dumps(report, indent=2))
        return
    module.direct_transaction = DirectTransaction(module, motion, report)
    module.residual_motion = candidate
