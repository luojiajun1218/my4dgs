"""Single production path: parent-relative rigid motion and quantized births."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch.nn.functional import l1_loss
from torchvision.utils import save_image

from libgs.metric import lpips, psnr, ssim
from libgs.model.merged_gaussian import MergedGaussianModel
from libgs.pipeline import Module as BaseModule
from libgs.pipeline import ModuleConfig as BaseModuleConfig
from libgs.renderer import Renderer, RendererConfig

from .model.gaussian import BirthGaussianModel, GaussianModel, select_budget_mask
from .model.rigid_tree_birth_codec import export_birth_attributes
from .model.birth_quantization import (RAW_ATTRIBUTES, RAW_FIELDS,
    QuantizedNewGaussianView, materialize_quantized_gaussians, model_raw,
    reference_digest)
from .model.rigid_residual_tree import (RigidResidualTree, RigidTreeConfig,
    quaternion_angle_error, saved_config_with_default_motion_lr)
from .model.tree_statistics import write_tree_statistics
from .model.sparse_opacity import (SparseOpacityDelta, QuantizedOldOpacityView,
                                  decode_opacity_delta)
from .model.contribution_renderer import ContributionRenderer


@dataclass
class GaussianConfig:
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 15000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    percent_dense: float = 0.01


@dataclass
class BirthConfig:
    """Stage-two candidate fitting and exact quantized attribute grid."""

    clone_sigma_scale: float = 2.0
    densification_stats_steps: int = 30
    densify_interval: int = 40
    densify_grad_threshold: float = 1.5e-4
    opacity_threshold: float = 0.01
    new_max_screen_size: float = 20.0
    max_new_gaussians: Optional[int] = None
    new_xyz_step_extent: float = 1e-5
    new_rotation_step: float = 0.001
    new_logscale_step: float = 0.001
    new_opacity_step: float = 0.01
    new_sh_step: float = 0.001

    def validate(self, motion_steps):
        if (isinstance(self.densification_stats_steps, bool) or
                not isinstance(self.densification_stats_steps, int) or
                not 1 <= self.densification_stats_steps <= motion_steps):
            raise ValueError('birth.densification_stats_steps must fit the motion phase')
        if (isinstance(self.densify_interval, bool) or
                not isinstance(self.densify_interval, int) or
                self.densify_interval < 1 or
                not math.isfinite(self.densify_grad_threshold) or
                self.densify_grad_threshold <= 0):
            raise ValueError('Birth densification interval and threshold must be positive')
        for name in ('clone_sigma_scale', 'new_max_screen_size',
                     'new_xyz_step_extent', 'new_rotation_step',
                     'new_logscale_step', 'new_opacity_step', 'new_sh_step'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'birth.{name} must be positive')
        if not 0 < self.opacity_threshold < 1:
            raise ValueError('birth.opacity_threshold must be in (0, 1)')
        if self.max_new_gaussians is not None and (
                isinstance(self.max_new_gaussians, bool) or
                not isinstance(self.max_new_gaussians, int) or
                self.max_new_gaussians < 1):
            raise ValueError('birth.max_new_gaussians must be positive or null')


@dataclass
class OldOpacityConfig:
    enabled: bool = False
    mode: str = 'logit'
    step: float = .01
    learning_rate: float = .05
    storage_price_scale: float = 1.0

    def validate(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('old_opacity.enabled must be boolean')
        if self.mode not in ('logit', 'alpha', 'saturated_alpha'):
            raise ValueError('old_opacity.mode must be logit, alpha or saturated_alpha')
        for name in ('step', 'learning_rate'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'old_opacity.{name} must be positive')
        if (isinstance(self.storage_price_scale, bool) or
                not math.isfinite(self.storage_price_scale) or self.storage_price_scale < 0):
            raise ValueError('old_opacity.storage_price_scale must be nonnegative')


@dataclass
class ModuleConfig(BaseModuleConfig):
    sh_degree: int = 1
    effective_motion_observations: bool = False
    initial_opacity_threshold: float = 0.0
    lambda_dssim: float = 0.2
    num_saving_images: int = 5
    lpips_required: bool = True
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    gaussian_stage2: GaussianConfig = field(default_factory=GaussianConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    rigid_tree: RigidTreeConfig = field(default_factory=RigidTreeConfig)
    birth: BirthConfig = field(default_factory=BirthConfig)
    old_opacity: OldOpacityConfig = field(default_factory=OldOpacityConfig)


def validate_initial_opacity_threshold(threshold):
    if (isinstance(threshold, bool) or
            not isinstance(threshold, (int, float)) or
            not math.isfinite(threshold) or
            not 0 <= threshold < 1):
        raise ValueError('module.initial_opacity_threshold must be in [0, 1)')


@torch.no_grad()
def prune_native_initial_gaussians(model, threshold):
    """Prune only the loaded native frame0, before the decoder reference exists."""
    validate_initial_opacity_threshold(threshold)
    original_count = len(model._xyz)
    if original_count < 1:
        raise ValueError('Native first frame contains no Gaussians')
    removed_count = 0
    if threshold > 0:
        remove = torch.sigmoid(model._opacity.detach().flatten()) < threshold
        removed_count = int(remove.sum())
        if removed_count == original_count:
            raise ValueError('Initial opacity pruning would remove every Gaussian')
        if removed_count:
            model.prune_points(remove)
    retained_count = len(model._xyz)
    if retained_count != original_count - removed_count:
        raise ValueError('Initial opacity pruning produced an inconsistent GS count')
    return dict(source='native_frame0_checkpoint', threshold=float(threshold),
                original_count=original_count, retained_count=retained_count,
                removed_count=removed_count)


def old_gaussian_pose_activity(decoded_xyz, decoded_rotation,
                               prior_xyz, prior_rotation, region_steps, gs_steps):
    """Count old GS with decoded pose changes above half a quantization step."""
    translation_threshold = min(float(region_steps[0]), float(gs_steps[0])) / 2
    rotation_threshold = min(float(region_steps[3]), float(gs_steps[3])) / 2
    moved_xyz = (decoded_xyz - prior_xyz).norm(dim=-1)
    moved_rotation = quaternion_angle_error(decoded_rotation, prior_rotation)
    return dict(
        old_gs_with_changed_pose=int(((moved_xyz > translation_threshold) |
                                      (moved_rotation > rotation_threshold)).sum()),
        changed_pose_translation_threshold_world=translation_threshold,
        changed_pose_rotation_threshold_rad=rotation_threshold)


class Module(BaseModule):
    """Motion freezes appearance; birth optionally fits sparse old opacity."""

    def setup(self):
        cfg = self.config
        validate_initial_opacity_threshold(cfg.initial_opacity_threshold)
        cfg.birth.validate(cfg.rigid_tree.motion_steps)
        cfg.old_opacity.validate()
        if not isinstance(cfg.effective_motion_observations, bool):
            raise ValueError('effective_motion_observations must be boolean')
        if not 0 <= cfg.lambda_dssim <= 1:
            raise ValueError('module.lambda_dssim must be in [0, 1]')
        bg = torch.ones if self.datamodule.config.white_background else torch.zeros
        self.register_buffer('background', bg(3, device=self.device))
        self.gaussians = GaussianModel(cfg.sh_degree)
        self.gaussians.create_from_pcd(self.datamodule.scene.point_cloud,
                                       self.datamodule.cameras_extent)
        self.gaussians.training_setup(cfg.gaussian)
        self.births = BirthGaussianModel(cfg.sh_degree)
        renderer_type = ContributionRenderer if cfg.effective_motion_observations else Renderer
        self.renderer = renderer_type(cfg.renderer, self.gaussians)
        self.motion = RigidResidualTree(cfg.rigid_tree)
        self.frame_prepared = False
        self.frame_finalized = False
        self.motion_committed = False
        self.reference_count = None
        self.initial_reference = None
        self.frame_report = None
        self.birth_control = {}
        self.tree_statistics = []
        self.old_opacity_delta = None
        self.old_opacity_optimizer = None
        self.old_opacity_metrics = {}
        self.stage_image_metrics = []
        self._freeze_base()

    @property
    def current_frame(self):
        return self.datamodule.current_frame

    @property
    def motion_steps(self):
        return self.config.rigid_tree.motion_steps_for_frame(self.current_frame)

    @property
    def total_steps(self):
        return self.motion_steps + self.config.rigid_tree.birth_steps

    def _freeze_base(self):
        for attr in RAW_ATTRIBUTES:
            getattr(self.gaussians, attr).requires_grad_(False)

    def on_save_checkpoint(self, ckpt):
        ckpt['rigid_tree_version'] = 1
        ckpt['rigid_tree_config'] = asdict(self.config.rigid_tree)
        ckpt['rigid_tree_module_config'] = asdict(self.config)
        ckpt['rigid_tree_initial_reference'] = self.initial_reference
        ckpt['gaussians'] = self.gaussians.capture()
        ckpt['rigid_tree_motion'] = self.motion.capture() if self.frame_prepared else None
        ckpt['rigid_tree_frame'] = dict(prepared=self.frame_prepared,
            finalized=self.frame_finalized, motion_committed=self.motion_committed,
            reference_count=self.reference_count, report=self.frame_report,
            birth_control=self.birth_control, tree_statistics=self.tree_statistics)
        ckpt['rigid_tree_frame']['stage_image_metrics'] = self.stage_image_metrics
        ckpt['old_opacity_delta'] = (dict(
            state={name: value.detach().cpu().clone() for name, value in
                   self.old_opacity_delta.state_dict().items()},
            optimizer=self.old_opacity_optimizer.state_dict(),
            metrics=self.old_opacity_metrics)
            if getattr(self, 'old_opacity_delta', None) is not None and
               self.motion_committed and not self.frame_finalized else None)
        ckpt['births'] = (self.births.capture() if self.motion_committed and
                          not self.frame_finalized and self.births.optimizer is not None else None)
        ckpt['birth_source_rows'] = (self.births.source_rows.detach().cpu()
                                     if self.births.source_rows is not None else None)

    def on_load_checkpoint(self, ckpt):
        self.gaussians.restore(ckpt['gaussians'], self.config.gaussian)
        self.renderer.gaussians = self.gaussians
        version = ckpt.get('rigid_tree_version')
        if version is None:
            if self.current_frame != 0 or ckpt['global_step'] != self.trainer.config.num_init_steps:
                raise ValueError('Only a completed native first-frame checkpoint can bootstrap this method')
            self.initial_reference = prune_native_initial_gaussians(
                self.gaussians, self.config.initial_opacity_threshold)
            self.reference_count = self.initial_reference['retained_count']
            self._freeze_base()
            return
        self._freeze_base()
        if (version != 1 or
                saved_config_with_default_motion_lr(ckpt['rigid_tree_config']) !=
                asdict(self.config.rigid_tree)):
            raise ValueError('Rigid-tree checkpoint needs the same method configuration')
        saved_config = ckpt.get('rigid_tree_module_config')
        if saved_config is not None and 'initial_opacity_threshold' not in saved_config:
            saved_config = {**saved_config, 'initial_opacity_threshold': 0.0}
        if isinstance(saved_config, dict) and 'effective_motion_observations' not in saved_config:
            saved_config = {**saved_config, 'effective_motion_observations': False}
        if isinstance(saved_config, dict) and 'old_opacity' not in saved_config:
            saved_config = {**saved_config, 'old_opacity': asdict(OldOpacityConfig())}
        if isinstance(saved_config, dict) and isinstance(saved_config.get('old_opacity'), dict):
            saved_config = {**saved_config, 'old_opacity': {
                'mode': 'logit', **saved_config['old_opacity']}}
        if isinstance(saved_config, dict) and 'rigid_tree' in saved_config:
            saved_config = {**saved_config, 'rigid_tree':
                saved_config_with_default_motion_lr(saved_config['rigid_tree'])}
        if saved_config != asdict(self.config):
            raise ValueError('Rigid-tree checkpoint needs the same module configuration')
        state = ckpt['rigid_tree_frame']
        self.frame_prepared = state['prepared']
        self.frame_finalized = state['finalized']
        self.motion_committed = state['motion_committed']
        self.reference_count = state['reference_count']
        self.initial_reference = ckpt.get('rigid_tree_initial_reference')
        if self.initial_reference is None:
            self.initial_reference = dict(
                source='native_frame0_checkpoint', threshold=0.0,
                original_count=self.reference_count,
                retained_count=self.reference_count, removed_count=0)
        self.frame_report = state['report']
        self.birth_control = state['birth_control']
        self.tree_statistics = state.get('tree_statistics', [])
        self.stage_image_metrics = state.get('stage_image_metrics', [])
        self.old_opacity_delta = None
        self.old_opacity_optimizer = None
        self.old_opacity_metrics = {}
        if self.frame_prepared:
            self.motion.restore(ckpt['rigid_tree_motion'])
            if self.motion_committed:
                for parameter in self.motion.parameters():
                    parameter.requires_grad_(False)
                self.motion.optimizer.zero_grad(set_to_none=True)
        if ckpt['births'] is not None:
            self.births.restore(ckpt['births'], self.config.gaussian_stage2)
            source = ckpt['birth_source_rows']
            if source is None or len(source) != len(self.births._xyz):
                raise ValueError('Birth checkpoint lost source lineage')
            self.births.source_rows = source.to(self.device)
        opacity = ckpt.get('old_opacity_delta')
        if opacity is not None:
            if not self.config.old_opacity.enabled or self.frame_finalized or not self.motion_committed:
                raise ValueError('Pending opacity checkpoint has incompatible frame phase')
            self._initialize_old_opacity()
            self.old_opacity_delta.load_state_dict(opacity['state'])
            if (self.old_opacity_delta.reference.detach().cpu().numpy().tobytes() !=
                    self.gaussians._opacity.detach().cpu().numpy().tobytes()):
                raise ValueError('Pending opacity reference differs from old decoded logits')
            self.old_opacity_optimizer.load_state_dict(opacity['optimizer'])
            self.old_opacity_metrics = opacity['metrics']
        elif self.config.old_opacity.enabled and self.motion_committed and not self.frame_finalized:
            raise ValueError('Mid-birth opacity checkpoint lost pending updates')

    def prepare_frame(self):
        if self.frame_prepared:
            return
        if self.current_frame < 1:
            raise ValueError('Rigid motion starts after the shared first frame')
        xyz, rotation = self.gaussians._xyz.detach(), self.gaussians._rotation.detach()
        previous_rows = (None if self.current_frame == 1 else
                         torch.arange(len(xyz), device=xyz.device))
        self.motion.setup(xyz, rotation, self.datamodule.cameras_extent,
                          frame_index=self.current_frame, previous_rows=previous_rows)
        self.frame_prepared = True
        self.frame_finalized = False
        self.motion_committed = False
        self.frame_report = None
        self.birth_control = dict(clone_added=0, split_added=0, pruned=0,
                                  source_tracked=True)
        self.tree_statistics = []
        self.old_opacity_delta = None
        self.old_opacity_optimizer = None
        self.old_opacity_metrics = {}
        self.stage_image_metrics = []
        self._freeze_base()

    def pre_training_step(self):
        self.prepare_frame()

    def begin_next_frame(self):
        if not self.frame_finalized:
            raise ValueError('Cannot advance before quantized finalization')
        self.frame_prepared = False
        self.frame_finalized = False
        self.motion_committed = False
        self.frame_report = None

    def _write_tree_statistics(self, step):
        path = (self.output_dir / 'tree_stats' /
                f'frame_{self.current_frame:03d}_step_{step:03d}.csv')
        record = write_tree_statistics(path, self.motion, self.reference_count, step)
        record['path'] = str(path.relative_to(self.output_dir))
        record['source_run'] = str(self.output_dir)
        self.tree_statistics.append(record)

    def make_birth_view(self):
        if not len(self.births._xyz):
            return None
        reference = model_raw(self.gaussians)
        return QuantizedNewGaussianView(self.births, self.config.birth,
            self.datamodule.cameras_extent, reference,
            source_rows=self.births.source_rows)

    def forward(self, viewpoint, training=False, old_only=False):
        old_model = self.renderer.gaussians
        effective = getattr(self.config, 'effective_motion_observations', False)
        previous_collection = getattr(self.renderer, 'collect_contributions', False)
        if effective:
            self.renderer.collect_contributions = bool(training and self.frame_prepared and
                                                       not self.motion_committed)
        try:
            if self.frame_prepared and not self.motion_committed:
                xyz, rotation = self.motion(self.trainer.global_step, training=training)
                self.gaussians.rigid_xyz = xyz
                self.gaussians.rigid_rotation = rotation
            elif self.motion_committed and not self.frame_finalized and not old_only:
                delta = getattr(self, 'old_opacity_delta', None)
                old_view = (QuantizedOldOpacityView(self.gaussians, delta)
                            if delta is not None else self.gaussians)
                self.renderer.gaussians = old_view
                birth_view = self.make_birth_view()
                if birth_view is not None:
                    self.renderer.gaussians = MergedGaussianModel(
                        [old_view, birth_view])
            return self.renderer(viewpoint, self.background)
        finally:
            self.gaussians.rigid_xyz = None
            self.gaussians.rigid_rotation = None
            self.renderer.gaussians = old_model
            if effective:
                self.renderer.collect_contributions = previous_collection

    def training_step(self, viewpoint, step):
        package = self.forward(viewpoint, training=True)
        image = package['render']
        image_loss = ((1 - self.config.lambda_dssim) *
                      l1_loss(image, viewpoint.image) +
                      self.config.lambda_dssim * (1 - ssim(image, viewpoint.image)))
        loss = image_loss
        metrics = dict(image_loss=float(image_loss.detach()),
                       psnr=float(psnr(image.detach(), viewpoint.image).detach()),
                       gs=len(self.gaussians._xyz) +
                       (len(self.births._xyz) if self.motion_committed else 0))
        if not self.motion_committed:
            terms = self.motion.loss_terms(step, diagnostics=False)
            loss = loss + terms['rate_loss'] + terms['relative_motion_loss']
            if 'child_relative_motion_loss' in terms:
                loss = loss + terms['child_relative_motion_loss']
            heir_terms = ({} if self.config.rigid_tree.online_flat_clustering else
                          self.motion.heir_loss_terms(step))
            if heir_terms:
                loss = loss + heir_terms['heir_loss']
        return loss, metrics, package

    @torch.no_grad()
    def _begin_births(self):
        xyz, rotation = self.motion(self.motion_steps, training=False)
        self.gaussians.apply_selected_motion(xyz, rotation)
        self.motion_committed = True
        for parameter in self.motion.parameters():
            parameter.requires_grad_(False)
        self.motion.optimizer.zero_grad(set_to_none=True)
        count = len(self.gaussians._xyz)
        grads = self.gaussians.xyz_gradient_accum / self.gaussians.denom
        grads[grads.isnan()] = 0.
        scores = grads.norm(dim=-1)
        selected = ((scores >= self.config.birth.densify_grad_threshold) &
                    (self.gaussians.get_scaling.max(dim=1).values <=
                     self.gaussians.percent_dense * self.datamodule.cameras_extent))
        selected = select_budget_mask(selected, scores,
                                      self.config.birth.max_new_gaussians)
        source_rows = selected.nonzero(as_tuple=True)[0]
        clone = self.gaussians.adaptive_densify_and_clone(
            self.config.birth.densify_grad_threshold,
            self.datamodule.cameras_extent,
            sigma_scale=self.config.birth.clone_sigma_scale,
            max_new_gaussians=self.config.birth.max_new_gaussians)
        if clone['added'] != len(source_rows):
            raise AssertionError('Initial birth source rows disagree with clones')
        self.birth_control['clone_added'] = clone['added']
        if clone['added']:
            self.births.restore(self.gaussians.capture(),
                                self.config.gaussian_stage2)
            new_mask = torch.arange(len(self.gaussians._xyz),
                                    device=self.device) >= count
            self.gaussians.prune_points(new_mask)
            self.births.prune_points(~new_mask)
            self.births.source_rows = source_rows
            self.births.initialize_incremental_optimizer(
                self.config.gaussian_stage2)
        self._freeze_base()
        self._initialize_old_opacity()

    def _initialize_old_opacity(self):
        if not self.config.old_opacity.enabled:
            return
        cfg = self.config.old_opacity
        self.old_opacity_delta = SparseOpacityDelta(self.gaussians._opacity.detach(), cfg.step, mode=cfg.mode)
        self.old_opacity_optimizer = torch.optim.Adam(
            [self.old_opacity_delta.delta], lr=cfg.learning_rate,
            betas=tuple(self.config.rigid_tree.adam_betas), eps=self.config.rigid_tree.adam_eps)
        self.old_opacity_metrics = dict(projected_rows=0, proximal_zeroed_rows=0)

    @torch.no_grad()
    def _step_old_opacity(self):
        if getattr(self, 'old_opacity_delta', None) is None:
            return
        self.old_opacity_optimizer.step()
        self.old_opacity_optimizer.zero_grad(set_to_none=True)
        # Motion's ramp is already complete at the birth boundary. Reuse its
        # resolved normalization and byte price, rather than a new strong prior.
        ramp = min(max((self.motion_steps - self.config.rigid_tree.rate_warmup_steps) /
                       self.config.rigid_tree.rate_ramp_steps, 0.), 1.)
        price = (self.config.rigid_tree.rate_loss_weight * ramp /
                 self.motion.rate_normalization_bytes * self.config.old_opacity.storage_price_scale)
        metrics = self.old_opacity_delta.proximal_zero(self.old_opacity_optimizer, price)
        for name in ('projected_rows', 'proximal_zeroed_rows'):
            self.old_opacity_metrics[name] += metrics[name]
        self.old_opacity_metrics.update(byte_price=price,
            normalization_count=self.motion.rate_normalization_count,
            storage_price_scale=self.config.old_opacity.storage_price_scale,
            support_cost_proxy='absolute row varint bytes + signed int16 value')

    @torch.no_grad()
    def _prune_terminal_births(self):
        """Remove weak births without adding unfitted GS on the final step."""
        model = self.births
        cfg = self.config.birth
        low_opacity = model.get_opacity.flatten() < cfg.opacity_threshold
        big_screen = model.max_radii2D > cfg.new_max_screen_size
        big_world = (model.get_scaling.max(dim=1).values >
                     0.1 * self.datamodule.cameras_extent)
        remove = low_opacity | big_screen | big_world
        removed = int(remove.sum())
        if removed:
            model.prune_points(remove)
        return removed

    @torch.no_grad()
    def post_training_step(self, package, pre_topology_callback=None):
        step = self.trainer.global_step
        if not self.motion_committed:
            frustum = package['visibility_filter']
            observed = (package.get('motion_contribution_filter')
                if getattr(self.config, 'effective_motion_observations', False) else frustum)
            if observed is None or observed.dtype != torch.bool or observed.shape != frustum.shape:
                raise ValueError('Motion evidence requires an aligned actual compositing mask')
            self.motion.mark_observed(observed)
            self.birth_control['motion_frustum_samples'] = (
                self.birth_control.get('motion_frustum_samples', 0) + int(frustum.sum()))
            self.birth_control['motion_observed_samples'] = (
                self.birth_control.get('motion_observed_samples', 0) + int(observed.sum()))
            tree_update_due = self.motion.stage(step) in ('tree_edit', 'tree_finalize')
            flat_update_due = self.motion.stage(step) == 'flat_partition'
            edits = self.motion.optimizer_step(pre_topology_callback=pre_topology_callback)
            if not self.config.rigid_tree.adaptive_clustering:
                edits = self.motion.update_topology(step)
            if tree_update_due:
                self._write_tree_statistics(step)
            if flat_update_due:
                directory = self.output_dir / 'region_statistics'
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'frame_{self.current_frame:03d}.json').write_text(
                    json.dumps(self.motion.flat_region_diagnostics, indent=2))
            if edits:
                self.birth_control['tree_edit_count'] = (
                    self.birth_control.get('tree_edit_count', 0) + len(edits))
            start = self.motion_steps - self.config.birth.densification_stats_steps + 1
            if step == start:
                self.gaussians.xyz_gradient_accum.zero_()
                self.gaussians.denom.zero_()
            if step >= start:
                self.gaussians.add_densification_stats(
                    package['viewspace_points'], package['visibility_filter'])
            if step == self.motion_steps:
                self.motion.finalize_online_refinement()
                if getattr(self.trainer, 'bootstrap_pending', False):
                    if not self.motion.bootstrap_finalized:
                        raise ValueError('First motion frame has not finalized its initial region tree')
                    # The base attributes are still frame0 here. Write the
                    # learned layout before _begin_births installs frame1 pose.
                    self.trainer._write_reference(
                        frame_index=0, state=self.motion.frame_reference_state)
                    self.trainer.bootstrap_pending = False
                self._begin_births()
        else:
            if len(self.births._xyz):
                old_count = len(self.gaussians._xyz)
                visible = package['visibility_filter'][old_count:]
                radii = package['radii'][old_count:]
                self.births.max_radii2D[visible] = torch.maximum(
                    self.births.max_radii2D[visible], radii[visible])
                self.births.add_densification_stats(
                    package['viewspace_points'], visible, align='right')
                birth_step = step - self.motion_steps
                if step == self.total_steps:
                    # Apply the last fitted gradient before pruning replaces
                    # optimizer parameters; no new GS may be born here.
                    self.births.optimizer.step()
                    self.births.optimizer.zero_grad(set_to_none=True)
                    self.birth_control['pruned'] += self._prune_terminal_births()
                else:
                    if birth_step % self.config.birth.densify_interval == 0:
                        result = self.births.adaptive_densify_and_prune(
                            self.config.birth.densify_grad_threshold,
                            self.config.birth.opacity_threshold,
                            self.datamodule.cameras_extent,
                            self.config.birth.new_max_screen_size,
                            max_points=self.config.birth.max_new_gaussians)
                        self.birth_control['split_added'] += result['added']
                        self.birth_control['pruned'] += result['pruned']
                    self.births.optimizer.step()
                    self.births.optimizer.zero_grad(set_to_none=True)
            if getattr(self, 'old_opacity_delta', None) is not None:
                self._step_old_opacity()
            if step == self.total_steps:
                self.finalize_frame()

    @torch.no_grad()
    def finalize_frame(self):
        if self.frame_finalized:
            return
        view = self.make_birth_view()
        raw, sources = materialize_quantized_gaussians(
            model_raw(self.gaussians), view)
        stream_dir = self.output_dir / 'frame_storage' / 'rigid_tree'
        stream_dir.mkdir(parents=True, exist_ok=True)
        motion_path = stream_dir / f'motion_{self.current_frame:03d}.bin'
        region_xyz, region_rot, gs_xyz, gs_rot = self.motion._values(hard=True)
        activity = dict(
            moving_regions=int((region_xyz[1:].ne(0).any(-1) |
                                region_rot[1:].ne(0).any(-1)).sum()),
            old_gs_with_local_residual=int((gs_xyz.ne(0).any(-1) |
                                            gs_rot.ne(0).any(-1)).sum()),
            old_gs=len(gs_xyz), topology_edits=self.birth_control.get('tree_edit_count', 0))
        self.motion.register_materialized_layout(raw['xyz'], raw['rotation'],
                                                 birth_source_rows=sources)
        rate_terms = self.motion.loss_terms(self.motion_steps, diagnostics=True)
        motion_info = self.motion.export_motion(motion_path)
        if self.config.rigid_tree.online_flat_clustering:
            directory = self.output_dir / 'region_statistics'
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f'frame_{self.current_frame:03d}.json').write_text(
                json.dumps(self.motion.flat_region_diagnostics, indent=2))
        decoded = self.motion.codec_previous_state
        prior = self.motion.frame_reference_state
        old_count = len(self.motion.owner)
        activity.update(old_gaussian_pose_activity(
            decoded.xyz[:old_count], decoded.rotation[:old_count],
            prior.xyz, prior.rotation, self.motion.steps, self.motion.gs_steps))
        raw['xyz'] = decoded.xyz.to(device=self.device, dtype=raw['xyz'].dtype)
        raw['rotation'] = decoded.rotation.to(device=self.device,
                                               dtype=raw['rotation'].dtype)
        birth_info = None
        if len(sources):
            birth_path = stream_dir / f'birth_{self.current_frame:03d}.bin'
            birth_info = export_birth_attributes(birth_path,
                {name: raw[name][-len(sources):] for name in
                 ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                len(raw['xyz']) - len(sources),
                symbols={name: view.symbols[name] for name in
                         ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                steps={name: view.steps[name] for name in
                       ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                predictor={name: view.predictor[name] for name in
                           ('logscale', 'opacity', 'sh_dc', 'sh_rest')},
                source_rows=sources)
        opacity_info = None
        delta = getattr(self, 'old_opacity_delta', None)
        if delta is not None:
            if (raw['opacity'][:old_count].detach().cpu().numpy().tobytes() !=
                    delta.reference.detach().cpu().numpy().tobytes()):
                raise ValueError('Old appearance changed before opacity packet materialization')
            symbols = delta.symbols(False)
            packet = delta.packet(self.current_frame)
            actual = delta(False)
            if packet is not None:
                checked = decode_opacity_delta(packet, delta.reference.detach().cpu(),
                                               frame_index=self.current_frame)
                if checked.logits.numpy().tobytes() != actual.detach().cpu().numpy().tobytes():
                    raise ValueError('Opacity training and decoded logits differ')
                path = stream_dir / f'opacity_{self.current_frame:03d}.bin'
                path.write_bytes(packet)
                opacity_info = dict(path=str(path), archive_bytes=len(packet),
                    format='sparse_old_opacity_delta_zlib_v' +
                           str({'logit': 1, 'alpha': 2, 'saturated_alpha': 3}[delta.mode]),
                    coordinate_mode=delta.mode, updated_rows=len(checked.rows))
                raw['opacity'][:old_count] = checked.logits.to(raw['opacity'])
            elif actual.detach().cpu().numpy().tobytes() != delta.reference.cpu().numpy().tobytes():
                raise ValueError('Empty opacity packet unexpectedly changes old logits')
            self.old_opacity_metrics.update(enabled=True, mode=delta.mode, step=float(delta.step),
                saturated_alpha_eligible=(int(delta._saturated.sum())
                    if delta.mode == 'saturated_alpha' else None),
                effective_saturated_alpha_step=(float(delta.step) * .25
                    if delta.mode == 'saturated_alpha' else None),
                old_gs=old_count, active_rows=int(symbols.ne(0).sum()),
                increased_rows=int(symbols.gt(0).sum()), decreased_rows=int(symbols.lt(0).sum()),
                max_abs_logit_delta=float((actual - delta.reference).abs().max()) if old_count else 0.,
                max_abs_alpha_delta=float((actual.sigmoid() - delta.reference.sigmoid()).abs().max()) if old_count else 0.,
                saturated_before=int(delta.reference.sigmoid().eq(1).sum()),
                saturated_after=int(actual.sigmoid().eq(1).sum()),
                zero_alpha_before=int(delta.reference.sigmoid().eq(0).sum()),
                zero_alpha_after=int(actual.sigmoid().eq(0).sum()),
                archive_bytes=(opacity_info['archive_bytes'] if opacity_info else 0),
                predictor_order='birth uses pre-update appearance; opacity applied to old prefix afterward')
        for name, attr in zip(RAW_FIELDS, RAW_ATTRIBUTES):
            setattr(self.gaussians, attr,
                    torch.nn.Parameter(raw[name].detach().clone()))
        self.gaussians.training_setup(self.config.gaussian)
        self.gaussians.max_radii2D = self.gaussians._xyz.new_zeros(len(raw['xyz']))
        self._freeze_base()
        self.renderer.gaussians = self.gaussians
        self.births = BirthGaussianModel(self.config.sh_degree)
        self.old_opacity_delta = None
        self.old_opacity_optimizer = None
        self.frame_finalized = True
        manifest = dict(version=1, frame=self.current_frame,
            reference=str(self.trainer.decoder_reference_path.relative_to(
                self.output_dir)),
            motion=motion_path.name,
            birth=(Path(birth_info['path']).name if birth_info else None),
            old_count=len(raw['xyz']) - len(sources), birth_count=len(sources),
            total_count=len(raw['xyz']),
            decoded_state_sha256=decoded.digest().hex(),
            raw_state_sha256=reference_digest(raw),
            no_old_deletions=True)
        if self.config.old_opacity.enabled:
            manifest['opacity_update'] = (Path(opacity_info['path']).name if opacity_info else None)
        manifest_path = stream_dir / f'frame_{self.current_frame:03d}.json'
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.frame_report = dict(frame=self.current_frame,
            initial_reference=self.initial_reference,
            region_count=len(self.motion.parents),
            initialization=dict(two_frame=self.config.rigid_tree.adaptive_clustering and not self.config.rigid_tree.online_flat_clustering,
                is_initialization_frame=self.config.rigid_tree.adaptive_clustering and self.current_frame == 1,
                initial_tree_built=self.motion.bootstrap_finalized or not self.config.rigid_tree.adaptive_clustering,
                initial_tree_seconds=self.motion.bootstrap_seconds if self.current_frame == 1 else 0.,
                bootstrap_tree_step=self.config.rigid_tree.bootstrap_tree_step,
                prestructure_mode=('shared_and_gs' if self.motion.shared_prestructure_active else 'independent_gs'),
                prestructure_steps=self.config.rigid_tree.motion_steps,
                # Keep the legacy step-count key for readers of older reports;
                # prestructure_mode describes which parameters actually learn.
                free_motion_steps=self.config.rigid_tree.motion_steps,
                post_partition_steps=self.config.rigid_tree.post_partition_steps,
                optimization_steps=self.total_steps),
            old_count=len(raw['xyz']) - len(sources), birth_count=len(sources),
            total_count=len(raw['xyz']), motion=motion_info, birth=birth_info,
            opacity_update=opacity_info, old_opacity_diagnostics=self.old_opacity_metrics,
            motion_observation_diagnostics=dict(
                kind=('actual_compositing' if self.config.effective_motion_observations else 'frustum'),
                frustum_samples=self.birth_control.get('motion_frustum_samples', 0),
                used_samples=self.birth_control.get('motion_observed_samples', 0),
                old_gs_with_no_motion_observation=int(self.motion.observations.eq(0).sum()),
                scope='surviving GS; current motion optimization views; excludes birth and heldout renders'),
            stage_image_metrics=self.stage_image_metrics,
            encoded_bytes=motion_path.stat().st_size +
                (birth_info['archive_bytes'] if birth_info else 0) +
                (opacity_info['archive_bytes'] if opacity_info else 0),
            shared_reference_bytes=self.trainer.decoder_reference_path.stat().st_size,
            materialized_sha256=reference_digest(raw),
            decoder_reference=manifest['reference'],
            frame_manifest=str(manifest_path.relative_to(self.output_dir)),
            motion_activity=activity,
            motion_rate_diagnostics={name: float(value.detach())
                                     for name, value in rate_terms.items()},
            motion_tree_diagnostics={name: float(value)
                                     for name, value in self.motion.last_metrics.items()
                                     if name.startswith('tree_edit/')},
            heir_diagnostics={name: float(value) for name, value in self.motion.last_metrics.items()
                              if name.startswith('heir/')},
            flat_region_diagnostics=self.motion.flat_region_diagnostics,
            tree_statistics=self.tree_statistics,
            birth_control=self.birth_control,
            semantics='Necessary motion, birth and optional sparse old-opacity files; shared first frame counted once. '
                      'No old-GS deletion; scale and SH remain unchanged for old GS.')

    @torch.no_grad()
    def validation_step(self, viewpoint, idx, loader_idx):
        image = self.forward(viewpoint)['render'].clamp(0, 1)
        if idx < self.config.num_saving_images and self.trainer.global_step == self.total_steps:
            root = Path(self.datamodule.config.root)
            path = (self.output_dir / 'image' /
                    viewpoint.path.relative_to(root))
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(image, path.with_name(path.stem +
                       f'_step{self.trainer.global_step}' + path.suffix))
            gt_path = path.with_name(path.stem + '_gt' + path.suffix)
            if not gt_path.exists() and not gt_path.is_symlink():
                save_image(viewpoint.image, gt_path)
        metrics = dict(l1=float(l1_loss(image, viewpoint.image)),
                       psnr=float(psnr(image, viewpoint.image)),
                       ssim=float(ssim(image, viewpoint.image)))
        if self.config.lpips_required:
            metrics['lpips'] = float(lpips(image, viewpoint.image,
                                          net_type='vgg').mean())
        return metrics

    def validation_end(self, results, num_loaders=1):
        groups = [results] if num_loaders == 1 else results
        summary = {}
        for name, rows in zip(self.datamodule.eval_names, groups):
            if rows:
                summary[name] = {key: sum(row[key] for row in rows) / len(rows)
                                 for key in rows[0]}
                self.log_dict({f'eval-{name}/{key}': value
                               for key, value in summary[name].items()})
        return summary
