"""200-step point-only ablation, with inert spatial metadata for codec compatibility."""
import torch

from .model.spatial_motion_training import FullUpdateSpatialMotionTraining
from .per_frame_training import PerFrameTraining as ExistingPerFrameTraining


class PointOnlySpatialMotionTraining(FullUpdateSpatialMotionTraining):
    @property
    def shared_discovery_active(self):
        return False

    def _gate_probabilities(self):
        probabilities = torch.zeros_like(self.gate_logits)
        probabilities[0] = 1
        return probabilities

    @torch.no_grad()
    def _prepare_region_gate_step(self, step, previous_step):
        self._hard_gate_frozen = True
        self.gate_logits.requires_grad_(False)
        self.gate_logits.grad = None

    @torch.no_grad()
    def _begin_joint_frame(self, proposal, previous_regions, previous_residuals):
        super()._begin_joint_frame(proposal, previous_regions, previous_residuals)
        self.region_translation.zero_()
        self.region_rotation.zero_()
        self._hard_gate_frozen = True

    @torch.no_grad()
    def prepare_plan_step(self, step):
        super().prepare_plan_step(step)
        for parameter in (self.region_translation, self.region_rotation, self.gate_logits):
            parameter.requires_grad_(False)
            parameter.grad = None
        self.gs_translation.requires_grad_(True)
        self.gs_rotation.requires_grad_(step > 100)
        if step <= 100:
            self.gs_rotation.grad = None
        self._hard_gate_frozen = True

    @torch.no_grad()
    def _select_frontier(self, freeze=True, full_motion=None, preserve_remainder=False):
        return super()._select_frontier(freeze=True, full_motion=full_motion,
                                        preserve_remainder=preserve_remainder)

    @torch.no_grad()
    def optimizer_step(self):
        edits = super().optimizer_step()
        if torch.count_nonzero(self.region_translation) or torch.count_nonzero(self.region_rotation):
            raise RuntimeError('Point-only ablation must not have regional motion')
        return edits


class PerFrameTraining(ExistingPerFrameTraining):
    def validate_per_frame_training(self, config):
        if (not config.full_update.enabled or not config.full_update.early_joint_repair or
                not config.full_update.dense_color_200 or config.full_update.regional_full_200 or
                config.full_update.attribute_aware_gate or config.ablation_motion_only or
                config.rigid_tree.motion_steps != 200 or config.rigid_tree.spatial_gate.final_joint_refine_steps):
            raise ValueError('Point-only requires the 200-step all-point plan without regional attributes')

    def per_frame_phase(self, step):
        return 'point_translation' if step <= 100 else 'point_motion_attributes'

    def _early_prepare_rates(self, step):
        super()._early_prepare_rates(step)
        context = self.context
        # The legacy point-only color variant freezes scale; this experiment
        # retains all point attributes in the second stage.
        context.full_attribute_update.parameters['logscale'].requires_grad_(step > 100)
        motion = context.motion
        if (motion.region_translation.requires_grad or motion.region_rotation.requires_grad or
                motion.gate_logits.requires_grad or not motion.gs_translation.requires_grad or
                motion.gs_rotation.requires_grad != (step > 100)):
            raise RuntimeError(f'Invalid point-only parameter schedule at step {step}')
        if step in (1, 100, 101, 151, 180, 200):
            context.birth_control.setdefault('point_only_freeze_audit', []).append(dict(
                step=int(step), region_translation_trainable=motion.region_translation.requires_grad,
                region_rotation_trainable=motion.region_rotation.requires_grad,
                gate_trainable=motion.gate_logits.requires_grad,
                point_translation_trainable=motion.gs_translation.requires_grad,
                point_rotation_trainable=motion.gs_rotation.requires_grad,
                opacity_trainable=context.opacity_update.delta.requires_grad,
                point_attribute_trainable={name: p.requires_grad for name, p in context.full_attribute_update.parameters.items()}))

    def adjust_manifest(self, manifest):
        manifest['training_step_accounting'].update(
            shared_gate=0, joint_gate_residual_early=0, fixed_gate_residual=0,
            joint_gate_residual=0, final_motion=0, point_translation_steps=200,
            point_rotation_steps=100, regional_full_attribute_steps=0,
            shared_adam_steps=0, point_adam_steps=200)
        manifest['regional_full_rate_config'] = None
        manifest['regional_full_timing'] = None
        manifest['point_only_ablation'] = dict(
            region_motion=False, gate_learning=False, regional_attributes=False,
            point_translation_steps=200, point_rotation_steps=100, point_attribute_steps=100,
            codec_spatial_metadata='inert fixed routing metadata; no shared motion or attributes')
