"""固定候选空间树的区域深度消融：共享前 100 步、冻结后 100 步。

仅在 RECONGS_HIERARCHY_ABLATION 明确选中时导入。复用现有逐帧优化、
相机采样、属性更新和编码；本方案只控制 gate 学习与固定层级归属。
"""
import os
from types import MethodType

import torch

from .per_frame_training import PerFrameTraining as ExistingPerFrameTraining
from .model.spatial_region_gate import spatial_owner


class PerFrameTraining(ExistingPerFrameTraining):
    def __init__(self, context):
        mode = os.environ.get('RECONGS_HIERARCHY_ABLATION')
        if mode not in ('adaptive', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5'):
            raise ValueError('RECONGS_HIERARCHY_ABLATION must be adaptive or depth1 through depth5')
        self.mode = mode
        self.depth = None if mode == 'adaptive' else int(mode[-1])
        super().__init__(context)

    def validate_per_frame_training(self, config):
        super().validate_per_frame_training(config)
        gate = config.rigid_tree.spatial_gate
        if (config.rigid_tree.shared_discovery_steps != 100 or gate.max_depth != 5 or
                not gate.frame_training_plan or not gate.shared_discovery or
                gate.image_relaxation != 'stochastic_ste'):
            raise ValueError('Hierarchy ablation requires the common depth-five, 100+100 stochastic frame plan')

    def per_frame_phase(self, step):
        if step <= 100:
            return 'adaptive_gate_shared' if self.depth is None else 'fixed_shared'
        return 'frozen_gate_motion_attributes'

    def configure_motion(self, motion):
        """Bind this experiment's selection policy to this model instance only."""
        depth = self.depth
        original_begin = motion._begin_joint_frame
        original_select = motion._select_frontier
        original_prepare_gate = motion._prepare_region_gate_step
        original_record_diagnostics = motion.record_final_diagnostics

        def fixed_probabilities(model):
            result = torch.zeros_like(model.gate_logits)
            result[model.spatial_paths[:, :depth - 1].reshape(-1)] = 1
            result[0] = 1
            return result

        def fixed_weights(model, *, sample_if_needed=True):
            return model._path_weights(fixed_probabilities(model))

        def assert_fixed(model):
            if not torch.equal(model.owner, model.spatial_paths[:, depth - 1]):
                raise RuntimeError(f'Fixed depth {depth} ownership changed at step {model.last_step}')

        @torch.no_grad()
        def begin(model, proposal, previous_regions, previous_residuals):
            model._ablation_owner_at_100 = None
            if model.spatial_paths.shape[1] != 5:
                raise ValueError('The comparison requires the same five-level candidate tree')
            if depth is not None:
                model.opened = fixed_probabilities(model).bool()
                model.owner = spatial_owner(model.spatial_paths, model.opened)
            original_begin(proposal, previous_regions, previous_residuals)
            if depth is not None:
                assert_fixed(model)

        @torch.no_grad()
        def prepare_gate(model, step, previous_step):
            if depth is None and step <= 100:
                original_prepare_gate(step, previous_step)
            else:
                model._hard_gate_frozen = True
                model.gate_logits.requires_grad_(False)
                model.gate_logits.grad = None

        @torch.no_grad()
        def select(model, freeze=True, full_motion=None, preserve_remainder=False):
            freeze = bool(freeze or depth is not None or model.last_step >= 100)
            edits = original_select(freeze=freeze, full_motion=full_motion,
                                    preserve_remainder=preserve_remainder)
            if depth is not None:
                assert_fixed(model)
            elif model.last_step == 100:
                model._ablation_owner_at_100 = model.owner.detach().clone()
            elif model.last_step > 100:
                if model._ablation_owner_at_100 is None or not torch.equal(model.owner, model._ablation_owner_at_100):
                    raise RuntimeError(f'Adaptive ownership changed after step 100 at {model.last_step}')
            return edits

        def record_diagnostics(model):
            result = original_record_diagnostics()
            model.flat_region_diagnostics['configured_gate_freeze_step'] = model.config.spatial_gate.freeze_step
            model.flat_region_diagnostics['gate_freeze_step'] = 100 if depth is None else 0
            model.flat_region_diagnostics['hierarchy_ablation_mode'] = self.mode
            return result

        motion._begin_joint_frame = MethodType(begin, motion)
        motion._prepare_region_gate_step = MethodType(prepare_gate, motion)
        motion._select_frontier = MethodType(select, motion)
        motion.record_final_diagnostics = MethodType(record_diagnostics, motion)
        if depth is not None:
            motion._gate_probabilities = MethodType(fixed_probabilities, motion)
            motion._sampled_path_weights = MethodType(fixed_weights, motion)

    def _early_prepare_rates(self, step):
        super()._early_prepare_rates(step)
        context = self.context
        motion = context.motion
        early = step <= 100
        expected_gate = early and self.depth is None
        if motion.gate_logits.requires_grad != expected_gate:
            raise RuntimeError(f'Unexpected gate freezing at step {step}')
        if not motion.region_translation.requires_grad:
            raise RuntimeError(f'Regional motion unexpectedly frozen at step {step}')
        for parameter in (motion.gs_translation, motion.gs_rotation, context.opacity_update.delta,
                          *context.full_attribute_update.parameters.values()):
            if parameter.requires_grad == early:
                raise RuntimeError(f'Unexpected residual or attribute freezing at step {step}')
        if step in (1, 100, 101, 151, 180, 200):
            context.birth_control.setdefault('hierarchy_ablation_freeze_audit', []).append(dict(
                step=int(step), gate_trainable=motion.gate_logits.requires_grad,
                region_translation_trainable=motion.region_translation.requires_grad,
                region_rotation_trainable=motion.region_rotation.requires_grad,
                point_translation_trainable=motion.gs_translation.requires_grad,
                point_rotation_trainable=motion.gs_rotation.requires_grad,
                opacity_trainable=context.opacity_update.delta.requires_grad,
                point_attribute_trainable={name: value.requires_grad for name,value in
                                           context.full_attribute_update.parameters.items()},
                hard_gate_frozen=bool(motion._hard_gate_frozen)))

    def adjust_manifest(self, manifest):
        accounting = manifest['training_step_accounting']
        accounting.update(shared_gate=100 if self.depth is None else 0,
            fixed_shared_motion=100 if self.depth is not None else 0,
            joint_gate_residual_early=0, fixed_gate_residual=100,
            joint_gate_residual=0, final_motion=0)
        manifest['hierarchy_ablation'] = dict(mode=self.mode, fixed_depth=self.depth,
            first_stage_steps=100, second_stage_steps=100,
            adaptive_gate_freezes_after_step=100,
            spatial_membership='per_frame_previous_decoded_position',
            fixed_ownership_per_frame=self.depth is not None)
        regional = manifest.get('regional_full_rate_config')
        if regional is not None:
            regional['joint_gate_start_step'] = None
            regional['gate_reopen_logit_limit'] = None
            regional['gate_alpha_rebase'] = 'not used: gate frozen in attribute stage'
